# DC-SAE: Deep Compression Semantic Autoencoder for Faster Diffusion Convergence

Xu Huang<sup>1∗</sup> Ye Huang<sup>1∗</sup> Zijun Liao<sup>1∗</sup> Yuwei Niu<sup>1</sup> Xiaojie Li Menghan Zhou<sup>2</sup> De Wen Soh<sup>2</sup> Xiaotong Li<sup>1</sup> Daquan Zhou<sup>1†</sup>

<sup>1</sup>Peking University, <sup>2</sup>Singapore University of Technology and Design

<sup>∗</sup>Equal contribution, <sup>†</sup>Corresponding author

## Abstract

High-compression tokenizers are essential for scaling latent image generative models. However, aggressive compression creates a fundamental tradeof between reconstruction fidelity and generation eficiency: high compression image encoder always increases the learning dificulty of difusion training, resulting in slow model convergence. Recent representation autoencoders speed up the difusion training by improving the latent feature’s expressive capability by replacing VAE encoders with pretrained semantic encoders, yet they are typically limited to moderate compression and lose pixel-level details necessary for faithful reconstruction. To achieve both high compression and fast difusion training, we propose DC-SAE, a Decoupled Compact Semantic Autoencoder designed for high-compression image generation with accelerated difusion model convergence. DC-SAE consists of two key components: (1) a macro-level architecture design that leverages semantic encoders to enable higher compression ratios, and (2) a pixel-level encoder that preserves low-level details, ensuring high-fidelity image reconstruction. We empirically demonstrate that DC-SAE performs strongly on image generation tasks, achieving both compact latent representations and eficient training dynamics. Specifically, on the ImageNet dataset with 512 × 512 resolution, DC-SAE achieves 32× spatial compression, with 29.79 PSNR and 3.37 gFID, substantially outperforming the previous state-of-the-art high-compression tokenizer baselines DC-AE by 13.5% and 54.9% on PSNR and gFID, respectively, maintaining comparable throughput and faster difusion model training convergence. Beyond class-conditional generation, a 1.6B-parameter DiT using DC-SAE achieves 0.84 on GenEval and 86.007 on DPG-Bench for text-to-image generation at 1024 × 1024 resolution.

Project Page: https://dagroup-pku.github.io/DCSAE

GitHub Repo: https://github.com/DAGroup-PKU/DCSAE

Huggingface Repo: https://huggingface.co/DAGroup-PKU/DCSAE

## 1 Introduction

High-compression visual tokenizers such as DC-AE [6, 7] are crucial because they substantially reduce the latent sequence length and computational cost of all latent-based generative models. However, despite their high reconstruction quality, the resulting latent spaces are often less favorable for difusion model training in terms of convergence speed: although high-compression tokenizers reduce the cost of each forward pass, difusion models typically require more training steps to converge, which can ultimately reduce the benefits for the overall training cost.

1024x1024, batch 32 on H200, imgs/s  
![](images/58cde77895ba1ba905dc953bc89718d0c99cded33d411d2a9e8927590d31447e.jpg)  
Source

![](images/493a50c02c7d6c1a2179bca277b526ae935a47c81425037062eb8a69b57c9a29.jpg)

![](images/e75d3c948243f79f67292da05b8d43dc213d0663351ea07cca2ceb3f164fb8d1.jpg)

![](images/fb26a7f8bd1970780faf06b63b0bd2e0c6d6b0a16e1d5112fa6b0032f8ad18ee.jpg)  
DC-SAE (32x) PSNR 29.79 / rFID 0.16

![](images/b09a32b1e4fe9919b281115e599127d43998f29fc97bfbdce0cdbabdc17dcf15.jpg)  
(b) Generation quality

![](images/439152887ae967e10c27cbe7bb48d568e9ae944b9941090fa84d80ede8caa533.jpg)  
(a) Reconstruction comparison  
(c) Autoencoder throughput  
Figure 1 Overview of DC-SAE under 32× spatial compression. (a) DC-SAE preserves reconstruction details missed by RAE and improves over DC-AE. (b) On ImageNet 512 × 512, DC-SAE achieves better gFID than DC-AE. (c) DC-SAE also provides substantially higher autoencoder throughput.

We are therefore motivated to ask: how can we design a tokenizer that achieves both a high input compression ratio and fast difusion model convergence? To explore this question, we draw inspiration from recent representation autoencoders (RAEs) [30, 37, 50, 52], which have been shown to enable faster difusion model training convergence than conventional variational autoencoders (VAEs). However, naively increasing the compression ratio of pretrained semantic encoders such as DINOv2 [30] to a deep-compression regime, e.g., 32×, substantially degrades the reconstruction quality of RAEs, rendering them impractical for high-fidelity generation.

Through a detailed analysis, we find that a potential root cause of the poor reconstruction performance of RAE-like models is their lack of low-level visual information, as semantic encoders primarily capture high-level abstract representations of input images.

We therefore propose a novel dual-stream encoder design, consisting of a semantic encoder for extracting high-level features and an unconstrained pixel-level encoder for preserving low-level visual information. The high-level semantic features capture the major components of visual inputs and map visually similar objects to similar latent representations, yielding a more compact and structured latent space that accelerates difusion model convergence. Meanwhile, the low-level pixel features compensate for the missing fine-grained details in semantic representations, enabling high-quality reconstruction even under a high compression ratio. The new scheme was termed DC-SAE. DC-SAE defines a new family of autoencoders that enjoys both a high-compression ratio and faster difusion model convergence. To further improve the reconstruction quality of DC-SAE, we introduce a new module termed Spatial DeMerger. It expands the compact fused latent only on the decoder side, and thus keeps the generator-facing sequence short while providing denser spatial conditioning for high-fidelity reconstruction.

We conduct extensive experiments to validate the efectiveness of DC-SAE. Under the challenging 32× compression regime, DC-SAE consistently outperforms the previous high-compression tokenizer DC-AE in both reconstruction and generation quality. Specifically, DC-SAE achieves 29.79 PSNR and 3.37 gFID, compared with 26.25 PSNR and 7.47 gFID obtained by DC-AE. Beyond quality improvements, DC-SAE also delivers substantially higher end-to-end throughput. As shown in Tab. 7, DC-SAE achieves 4.64×, 4.75×, and 4.41× higher throughput than DC-AE at 256 × 256, 512 × 512, and 1024 × 1024 resolutions, respectively. These results demonstrate that DC-SAE simultaneously improves reconstruction fidelity, generative performance, and computational eficiency under aggressive latent compression. We also extend the evaluation to text-to-image generation with a 1.6B-parameter DiT and a Qwen3-1.7B text encoder. At 1024 × 1024 resolution with classifier-free guidance (CFG), this model achieves 0.84 on GenEval and 86.007 on DPG-Bench, demonstrating the applicability of DC-SAE beyond class-conditional ImageNet generation.

We further provide an in-depth analysis of the proposed semantic–pixel dual-stream design. Under the 32× compression setting, DiT trained on the joint semantic–pixel latent space converges substantially faster than models trained on either semantic-only or pixel-only latents. Visualization results Fig. 5 further reveal that the pixel branch complements the semantic branch by recovering fine-grained local appearance details that are largely absent from high-level semantic representations. Together, these findings suggest that combining semantic compactness with pixel-level fidelity is key to building high-compression tokenizers that remain both reconstruction-friendly and difusion-friendly. Our contributions are summarized as follows:

• High-compression semantic encoders. We propose a simple, yet extremely efective way to adapt the semantic autoencoders from 16× to 32× compression, without re-training them. This makes it possible to utilize the pre-trained semantic encoders such as series works of DINOs [4, 30, 38] and CLIPs [34, 50].

• Fast diffusion model training convergence. We propose a novel dual-stream encoder mechanism that achieves 32× latent compression while maintaining intrinsically fast convergence for difusion model training. Unlike prior methods that rely on explicit and often complex latent-space constraints—such as VFM alignment [44], self-supervised token losses [21], or channel masking [7]—our approach improves convergence through the architectural design of the tokenizer itself. Moreover, it is complementary to these existing techniques and can potentially benefit from incorporating them further.

• State-of-the-art deep compression autoencoder. By combining these designs, we successfully train DC-SAE, which establishes a new state of the art in both reconstruction quality and difusion model training eficiency. On ImageNet, DC-SAE improves PSNR by 13.5% and FID by 54.9%, while achieving a 4.41× speedup over the previous state-of-the-art deep-compression method.

## 2 Related work

## 2.1 Representation encoders and unified tokenizers for generation

Pretrained visual representation encoders have recently been adopted as alternatives to conventional VAE encoders for latent generation. Self-supervised and vision-language models such as DINOv2, SigLIP, CLIP, and MAE provide semantically structured features that are useful for recognition and alignment [13, 30, 34, 50]. Representation autoencoders (RAEs) further show that replacing the reconstruction-only VAE encoder with a pretrained representation encoder can produce semantically meaningful latents and accelerate difusion transformer training [52]. However, these encoders are not originally designed for pixel-faithful reconstruction or generative tokenization. Their objectives often discard texture, precise local geometry, color statistics, and high-frequency details that are weakly relevant for semantic understanding but crucial for reconstruction. Complementary eforts explore semantic priors and latent-space regularization to improve latent representations and downstream generation [8, 24, 31, 44, 48].

Another related line of work pursues unified tokenizers for both visual understanding and generation [26, 28, 33, 39, 47]. TokenFlow and UniTok improve discrete token spaces through decoupled or multi-codebook designs [26, 33], while UniLiP and UniFlow adapt semantic features for reconstruction and generation via self-distillation or pixel-level decoding mechanisms [39, 47]. LatentUM further studies unified multimodal modeling in a shared semantic latent space [17]. Recent continuous or general-purpose visual tokenizers, such as UniFluid, Ming-UniVision, and AToken, also explore unified representations for visual understanding and generation across images, videos, or multimodal tasks [9, 16, 25].

The closest work to ours is SVG, which augments frozen DINO features with a lightweight residual branch to recover missing perceptual details for latent difusion [37]. While SVG shows that VFM features can serve as a generative latent space, our focus is diferent: we study how to convert a frozen semantic encoder into a highcompression visual tokenizer. Under deep compression, reconstruction becomes especially challenging because the compact semantic latent has limited spatial capacity. DC-SAE therefore introduces an unconstrained pixel encoder to complement the frozen semantic branch with local appearance information, without relying on SVG-style self-distillation or semantic regularization. Despite this simple design, DC-SAE improves reconstruction over SVG and recent regularized representation-tokenizer methods such as UniLiP [39], while retaining the fast generation convergence of semantic latents.

![](images/d3daf7e50ef23936e5268e53f7418895fbbe2872bb89b53d6d433df255e7a95c.jpg)  
Figure 2 Pipeline of DC-SAE. We first adapt a frozen semantic encoder to a 32× token grid. Since the resulting semantic autoencoder sufers from poor reconstruction under aggressive compression, we introduce an unconstrained pixel branch to preserve local image details. The semantic and pixel branches are fused and decoded by a ViT decoder with Spatial Demerger.

## 2.2 Deep-compression visual tokenizers

Visual tokenizers compress images into compact latent representations for eficient generative modeling [35]. As image generation scales to higher resolutions and longer videos, deep-compression tokenizers have become increasingly important because the latent sequence length directly determines the cost of difusion or transformer-based generators. DC-AE and DC-AE 1.5 push continuous autoencoders to much higher spatial compression ratios through residual autoencoding, decoupled high-resolution adaptation, and structured latent spaces [6, 7]. TexTok uses text descriptions to provide high-level semantic information during tokenization, allowing image tokens to focus more on fine visual details under small token budgets [49]. MAETok shows that masked modeling can improve latent-space structure for difusion models without relying on variational constraints [5]. TC-AE studies deep compression from the perspective of token capacity, using staged token compression and self-supervised token structuring to reduce representation collapse [21]. Recent image and video tokenizers further explore high-compression design from diferent perspectives, including eficient video autoencoding in H3AE, neural image/video tokenization in Cosmos Tokenizer, difusion-guided decoding in DGAE, and high-fidelity discrete tokenization in WeTok [23, 29, 43, 53]. Other recent works, such as FlexTok, GloTok, and TokBench, study adaptive token allocation, global semantic structure, and tokenizer evaluation for visual generation [2, 42, 51].

These methods reveal a common reconstruction–generation dilemma: increasing latent capacity can improve reconstruction, but may also make the latent distribution harder for downstream generators to learn [7, 21]. Our work addresses this dilemma in the setting of semantic autoencoders. Instead of requiring a single compact latent to preserve both semantics and pixel detail, DC-SAE uses a decoupled design: a compact semantic branch provides the generator-facing representation, while an unconstrained pixel branch supplies local appearance information needed for high-fidelity reconstruction. The two branches are concatenated and expanded by a Spatial DeMerger before decoding, which restores a denser spatial layout without requiring the generator to model a dense latent sequence. This allows us to extend semantic autoencoders beyond the common 16× regime toward higher compression, while preserving the generation eficiency benefits of compact semantic latents.

## 3 Method

Our goal is to build a high-compression visual tokenizer that preserves the fast convergence property of semantic representations while achieving faithful image reconstruction. Recent semantic autoencoders show that pretrained semantic encoders provide structured latent spaces that are highly efective for difusion training. However, most existing semantic autoencoders operate at the standard 16× spatial compression ratio. To improve throughput, we first investigate whether such semantic autoencoders can be directly scaled to 32× compression.

![](images/5f7a2f09918e2f03b57f6a703bd1dd60ae996e0b91470d4535f15d1480761c84.jpg)

![](images/2759213bb058776c16355d68aeafc242deb45f9c285e310da3e36b8d8c01ff1c.jpg)  
Figure 3 (a) Validation PSNR comparison between DeMerger and No-DeMerger variants. (b) DiT training convergence on joint semantic–pixel, semantic-only, and pixel-only latents under 32× compression. (c) Illustration of pre-merge and post-merge semantic compression for constructing a 32× semantic tokenizer.

## 3.1 Limitations of Semantic Autoencoders under Deep Compression Regime

Semantic encoders are trained to preserve high-level visual structure, such as object identity, scene layout, and semantic relations. These structures are usually stable under moderate image resizing. Motivated by this observation, we adopt a simple strategy to construct a 32× semantic autoencoder from a standard semantic encoder. Specifically, we resize the input image to half resolution, which is equivalent to using a larger efective patch on the original image. The resized image is then patchified, linearly projected, and fed into the frozen semantic encoder. This directly produces a compact 32× semantic latent.

The resulting semantic autoencoder is trained with the standard reconstruction objective:

$$
\mathcal { L } _ { \mathrm { r e c } } = \lambda _ { 1 } \| \mathbf { x } - \hat { \mathbf { x } } \| _ { 1 } + \lambda _ { \mathrm { p e r c } } \mathcal { L } _ { \mathrm { p e r c } } ( \mathbf { x } , \hat { \mathbf { x } } ) + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } } ,\tag{1}
$$

where $\mathcal { L } _ { \mathrm { p e r c } }$ denotes perceptual loss and ${ \mathcal { L } } _ { \mathrm { a d v } }$ denotes the optional adversarial loss.

However, we observe that this direct strategy is insuficient for constructing an efective high-compression tokenizer. While resizing largely preserves the global semantic structure, it substantially degrades the reconstruction fidelity. At $2 5 6 \times 2 5 6$ resolution, the resulting 32× semantic autoencoder achieves only around 15 PSNR. More importantly, the learned latent no longer exhibits the fast difusion convergence typically expected from pretrained semantic representations. After 80 epochs of DiT training, it reaches only 7.15 gFID.

One may suspect that this failure comes from the specific input-level resizing design rather than semantic compression itself. To rule out this possibility, we further test common MLLM-style post-merge compression strategies, such as average pooling or merging already-extracted semantic tokens. As discussed in Sec. 4.4, these alternatives perform even worse than the resizing-based pre-merge strategy. This suggests that the bottleneck is not merely the choice of compression implementation.

We attribute this failure to the objective mismatch of semantic autoencoding. A pretrained semantic encoder is not trained for invertible reconstruction. It keeps information useful for recognition and representation learning, but discards many pixel-level details, such as color statistics, textures, local boundaries, and high-frequency patterns. When the compression ratio is increased from 16× to 32×, this information loss is further amplified. Therefore, a semantic branch alone is insuficient for high-compression visual tokenization.

## 3.2 Semantic-guided Unconstrained AutoEncoder

The failure of the direct 32× semantic autoencoder shows that semantic latents alone are insuficient for high-fidelity reconstruction under aggressive compression. A straightforward way to improve reconstruction is to use an unconstrained, high-channel autoencoder, since a larger latent capacity can preserve more pixel-level information. However, prior works have shown that such reconstruction-oriented latents are often dificult for difusion models to learn. To improve latent generability, existing methods usually introduce additional constraints or regularization, such as KL regularization, representation alignment, structured latent design, or channel masking [7, 19, 35, 44]. This reflects a common trade-of: increasing autoencoder capacity improves reconstruction, but may hurt difusion convergence.

Our key motivation is that this trade-of can be alleviated by semantic guidance. Recent works suggest that semantic representation latents can provide useful structure for difusion training, and Latent Forcing further indicates that semantic latents can guide pixel-level generation trajectories [1]. Therefore, we hypothesize that, even under high compression, a frozen semantic encoder can serve as a structural guide for an unconstrained high-channel pixel autoencoder. Instead of constraining the pixel latent itself, we pair it with a semantic latent and jointly train the two branches inside a single autoencoder.

As shown in Fig. 2, DC-SAE introduces an unconstrained pixel encoder in parallel with the frozen 32× semantic branch. The semantic branch provides a compact and structured latent space that captures object-level layout and semantic relations. The pixel branch is trained from scratch and focuses on reconstruction-critical details, such as texture, color statistics, local boundaries, and high-frequency information. The two latents are concatenated along the channel dimension, fused by a lightweight projection, and decoded jointly. In this way, the final latent space combines the generative structure of semantic representations with the reconstruction capacity of unconstrained pixel features. Since the pixel branch is directly optimized for reconstruction, the decoder may over-rely on it and under-utilize the frozen semantic branch. We therefore apply sample-wise dropout to the pixel latent during tokenizer training, forcing the decoder to use semantic structure when the pixel branch is dropped and to treat the pixel branch as complementary detail when it is present.

This decomposition leads to a more suitable latent space for high-compression difusion training. As shown in Fig. 3(b), the joint semantic–pixel latent converges faster than both semantic-only and pixel-only alternatives, reaching 4.27 gFID after 80 epochs compared with 7.15 and 11.04, respectively. This indicates that neither branch alone is suficient: semantic features provide a structured, difusion-friendly representation but lose reconstruction-critical details, whereas pixel latents preserve such details but are harder to model at high channel capacity. By combining them, DC-SAE obtains a latent representation that supports both faithful reconstruction and eficient generative modeling.

After tokenizer training, the entire tokenizer is frozen, and DiT is trained on the fused latent. Because the semantic and pixel branches are jointly optimized through the same decoder, the resulting latent is more generation-friendly than simply concatenating an independently trained pixel autoencoder with semantic features.

## 3.3 ViT Decoder with Spatial Demerger

The semantic–pixel latent introduced above improves the information content of the tokenizer under 32× compression. However, the compact latent grid also creates a decoder-side spatial bottleneck. For difusion modeling, a sparse 32× grid is desirable because it significantly reduces the sequence length. For reconstruction, however, this grid is less suitable for a ViT decoder, which relies on spatial tokens and attention to recover local image details. A denser decoder-side token grid can thus provide more spatial positions for modeling textures, boundaries, and high-frequency structures.

To address this mismatch, we introduce Spatial DeMerger before the ViT decoder. Spatial DeMerger expands each compact latent token into a small group of spatial sub-tokens before decoding. Importantly, this expansion is applied only on the decoder side. The DiT still operates on the original compact 32× compressed latent, so the eficiency benefit of high compression is preserved, while the decoder receives a denser grid for high-fidelity reconstruction.

This design decouples the latent grid used for difusion modeling from the token grid used for image decoding. The compact grid keeps DiT training and sampling eficient, whereas the expanded decoder-side grid improves the reconstruction ability of the ViT decoder. In Sec. 4.4, we further provide ablation studies showing that Spatial DeMerger consistently improves reconstruction quality under high compression.

Table 1 Comparison of 16× semantic autoencoders on reconstruction quality and ImageNet 256 × 256 generation after 80 epochs. DC-SAE-16x achieves strong reconstruction while retaining competitive generation eficiency, making it a flexible baseline for high-compression tokenizer design. Notice: The 16× DC-SAE setting is used only in this ablation; all other DC-SAE results in this paper use the 32× setting.
<table><tr><td colspan="8">Metric RAE [52] SD-VAE [35] VA-VAE [44] SVG [37] REPA [46]</td></tr><tr><td colspan="8">Reconstruction</td></tr><tr><td>PSNR ↑</td><td>19.20</td><td>24.08</td><td>27.96</td><td>23.89</td><td>一</td><td></td><td>27.80</td></tr><tr><td>rFID ↓</td><td>0.62</td><td>0.87</td><td>0.28</td><td>0.65</td><td></td><td></td><td>0.26</td></tr><tr><td colspan="8">Generation</td></tr><tr><td>gFID ↓</td><td>2.16</td><td></td><td>5.96</td><td>6.57</td><td>7.90</td><td></td><td>3.09</td></tr><tr><td>IS↑</td><td>214.80</td><td></td><td>128.00</td><td></td><td>137.90</td><td>118.60</td><td>189.95</td></tr></table>

![](images/d41654733a9ac41f2e4b69a20f84853ce9c96a6cb0cae50a0de2349b4f54bd9b.jpg)  
Figure 4 Qualitative results on ImageNet. The images are generated with 512 × 512 resolutions in the latent space of DC-SAE under 32× compression, with randomly sampled class conditions.

## 4 Experiments

## 4.1 Experimental Setting

We evaluate DC-SAE on class-conditional ImageNet generation at both 256 × 256 and 512 × 512 resolutions, as well as text-to-image generation at 1024 × 1024 resolution. For ImageNet tokenizer training, we first train DC-SAE at 256 × 256 resolution and then finetune it at 512 × 512 resolution under the target spatial compression ratio. Detailed tokenizer training settings are provided in Appendix A.2. After tokenizer training, the tokenizer is frozen, and a latent difusion/transformer generator is trained in the learned latent space. For ImageNet difusion training, we follow the training protocol of DC-AE [6]. We report reconstruction quality using rFID, and ImageNet generation quality using gFID and Inception Score. Text-to-image training and evaluation are detailed in Sec. 4.3.

## 4.2 Main Result

ImageNet 256 × 256 generation We first compare DC-SAE with DC-AE and DC-AE-1.5 under the ImageNet 256 × 256 setting. This setting evaluates whether the proposed semantic–pixel decoupled tokenizer can improve generation quality under the same high-compression regime. As shown in Tab. 5, DC-SAE improves the reconstruction rFID from 0.69 to 0.45 and achieves a gFID of 3.31 with an Inception Score of 180.71 using a DiT-XL generator. Compared with DC-AE and DC-AE-1.5 under the same DiT-XL backbone, DC-SAE provides substantially better generation quality, showing that the proposed tokenizer produces a more generation-friendly latent space while preserving reconstruction fidelity. These results suggest that DC-SAE is already a strong high-compression tokenizer at 256 × 256 resolution. Although our generator uses the standard DiT-XL backbone rather than the specialized USiT architecture, it achieves generation quality comparable to or better than stronger DC-AE baselines, while also improving reconstruction rFID. Following RAE, we also use a DiT-XL variant with a DDT-wide head for difusion training [41].

Table 2 Comparison with state-of-the-art image generative models on ImageNet 512 × 512 class-conditional generation. DC-SAE uses a 32× high-compression semantic tokenizer and trains a DiT-XL generator in the learned latent space.
<table><tr><td colspan="10">ImageNet 512 × 512</td></tr><tr><td rowspan="3">Image Generative Model</td><td rowspan="3">Autoencoder</td><td rowspan="3">Compression Rate Params (B)</td><td rowspan="3"></td><td colspan="2">gFID ↓</td><td rowspan="3">Inception Score ↑ PSNR ↑ rFiD ↓</td><td colspan="2"></td><td rowspan="3"></td></tr><tr><td colspan="2"></td><td></td></tr><tr><td>w/o CFG</td><td>w/CFG</td><td></td><td></td><td></td></tr><tr><td>DiT-XL [32]</td><td>Flux-VAE-f8c16 [20]</td><td>8×</td><td>0.68</td><td>27.35</td><td>8.72</td><td>53.09</td><td></td><td></td><td></td></tr><tr><td>DiT-XL [32]</td><td>SD-VAE-f8c4 [35]</td><td>8×</td><td>0.67</td><td>12.03</td><td>3.04</td><td>105.25</td><td></td><td></td><td></td></tr><tr><td>SiT-XL [27]</td><td>SD-VAE-f8c4 [35]</td><td>8×</td><td>0.67</td><td></td><td>2.62</td><td></td><td></td><td></td><td></td></tr><tr><td>MAGVIT-v2 [45]</td><td></td><td></td><td></td><td>3.07</td><td>1.91</td><td></td><td>213.10</td><td></td><td></td></tr><tr><td>MAR-L [22]</td><td></td><td></td><td></td><td>2.74</td><td>1.73</td><td></td><td>205.20</td><td></td><td></td></tr><tr><td>EDM2-XXL [18]</td><td></td><td></td><td></td><td>1.91</td><td>1.81</td><td></td><td></td><td></td><td></td></tr><tr><td>SiT-XL [27]</td><td>DC-AE-f32c32 [6]</td><td>32×</td><td>0.67</td><td>7.47</td><td>2.41</td><td></td><td>131.37</td><td>26.25</td><td>0.20</td></tr><tr><td>USiT-H [6]</td><td>DC-AE-f32c32 [6]</td><td>32×</td><td>0.50 1.58</td><td>3.80 2.90</td><td>1.89 1.72</td><td></td><td>174.58 187.68</td><td></td><td></td></tr><tr><td>USiT-2B [6]</td><td>DC-AE-f32c32 [6]</td><td>32×</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DiT-XL + DDT wide head [41] DC-SAE</td><td></td><td>32×</td><td>0.83</td><td>3.37</td><td>1.99</td><td></td><td>205.83</td><td>29.79</td><td>0.16</td></tr></table>

Table 3 Text-to-image generation results with CFG. The DC-SAE model is evaluated at 1024 × 1024 resolution. Generator sizes and tokenizer compression ratios are listed explicitly; the baselines use diferent model and training configurations. Higher scores are better for both benchmarks.
<table><tr><td>Image Generative Model</td><td>Autoencoder</td><td>Compression Rate Params (B)</td><td></td><td>GenEval ↑</td><td>DPG-Bench ↑</td></tr><tr><td>DC-Gen-FLUX.1-Krea-12B [14]</td><td>DC-AE f32c32 [6]</td><td>32×</td><td>12</td><td>0.72</td><td>87.073</td></tr><tr><td>DC-Gen-FLUX.1-Krea-12B [14]</td><td>DC-AE-1.5 f64c128 [7]</td><td>64×</td><td>12</td><td>0.59</td><td>75.439</td></tr><tr><td>1.6B-DiT</td><td>DC-SAE</td><td>32×</td><td>1.6</td><td>0.84</td><td>86.007</td></tr></table>

ImageNet 512 × 512 generation We further evaluate DC-SAE on ImageNet 512 × 512 class-conditional generation. The tokenizer is finetuned at 512 × 512 resolution under a 32× spatial compression ratio, and a DiT-XL generator is trained in the frozen DC-SAE latent space for 400 epochs following the DC-AE training setting. As shown in Tab. 2, DC-SAE achieves a gFID of 3.37 without CFG under the 512 × 512 setting, demonstrating that the proposed tokenizer scales to higher-resolution generation while maintaining a compact latent representation. We provide qualitative generation samples in Fig. 4.

## 4.3 Text-to-Image Generation

Experimental setup. To evaluate DC-SAE beyond class-conditional ImageNet generation, we train a text-toimage model with a 1.6B-parameter DiT backbone and Qwen3-1.7B [40] as the text encoder, using DC-SAE at 32× spatial compression. Training uses a subset of internal data together with LAION-COCO [36]. We progressively pretrain the model at 256 × 256 resolution for 100K iterations with a batch size of 1024, followed by another 100K iterations at 512 × 512 with the same batch size. We then perform supervised fine-tuning (SFT) at 1024 × 1024 resolution for 40K iterations with a batch size of 256. The final model is evaluated with classifier-free guidance (CFG) on GenEval [10] and DPG-Bench [15], which assess compositional text-to-image alignment and dense-prompt following, respectively.

Results. As shown in Tab. 3, our model achieves 0.84 on GenEval, exceeding the DC-AE and DC-AE-1.5- based DC-Gen-FLUX.1-Krea-12B baselines by 0.12 and 0.25, respectively. On DPG-Bench, our model obtains 86.007, compared with 87.073 for the DC-AE baseline and 75.439 for the DC-AE-1.5 baseline. Thus, the 1.6B-parameter generator achieves the highest GenEval score among these models and remains competitive on DPG-Bench against the 12B DC-AE-based generator. These results support the applicability of DC-SAE to open-ended text-to-image generation with a compact latent grid.

Table 4 Comparison between post-merge and pre-merge strategies for constructing a 32× semantic tokenizer.
<table><tr><td>Semantic Encoder PSNR ↑ rFID ↓ gFID ↓</td></tr><tr><td>Post-merge</td></tr><tr><td>Qwen-ViT Frozen Merger 23.40 0.62 25.96</td></tr><tr><td>66.83 Qwen-ViT AvgPool Merger 23.07 0.61 17.71 84.45</td></tr><tr><td>Qwen-ViT Learnable Merger 23.85 0.58 16.84 78.34</td></tr><tr><td>DINOv2 Learnable Merger 24.03 0.35 8.60 108.11</td></tr><tr><td>Pre-merge</td></tr><tr><td>DINOv2 23.13 0.45 3.31 180.71</td></tr></table>

Table 5 Comparison with DC-AE and DC-AE-1.5 on ImageNet 256 × 256 class-conditional generation.
<table><tr><td colspan="6">ImageNet 256 × 256</td></tr><tr><td>Diffusion Model</td><td>Setting</td><td>Autoencoder</td><td>rFID ↓</td><td></td><td>gFID ↓ Inception Score ↑</td></tr><tr><td>DiT-XL [32]</td><td>f32c32</td><td>DC-AE [6]</td><td>0.69</td><td>10.18</td><td>107.49</td></tr><tr><td>DiT-XL [32]</td><td>f32c128</td><td>DC-AE [6]</td><td>0.26</td><td>26.44</td><td>53.41</td></tr><tr><td>DiT-XL [32]</td><td>f32c32</td><td>DC-AE-1.5 [7]</td><td></td><td>10.50</td><td>107.99</td></tr><tr><td>DiT-XL [32]</td><td>f32c128</td><td>DC-AE-1.5 [7]</td><td>0.26</td><td>17.31</td><td>80.38</td></tr><tr><td>DiT-XL [32]</td><td>f32c64</td><td>DC-SAE (Ours)</td><td>0.45</td><td>5.67</td><td>156.68</td></tr><tr><td>USiT-H [6]</td><td>f32c32</td><td>DC-AE [6]</td><td>0.69</td><td>3.89</td><td>167.40</td></tr><tr><td>USiT-H [7]</td><td>f32c32</td><td>DC-AE-1.5 [7]</td><td></td><td>4.15</td><td>166.94</td></tr><tr><td>DiT-XL [32]</td><td>f32c128</td><td>TC-AE [21]</td><td>0.35</td><td>7.16</td><td></td></tr><tr><td>DiT-XL + DDT wide head [41] f32c64</td><td></td><td>DC-SAE (Ours)</td><td>0.45</td><td>3.31</td><td>180.71</td></tr></table>

## 4.4 Ablation Studies

We conduct several ablation studies to understand the four design choices of DC-SAE. Specifically, we ask: whether the proposed semantic–pixel decoupled design is efective under the standard 16× setting, how to adapt pretrained semantic encoders to the 32× token grid, whether Spatial DeMerger is necessary for high-compression ViT decoding, and whether the gain comes from joint semantic–pixel autoencoding rather than simply exposing DiT to semantic features.

Validation under 16× compression. We first evaluate the proposed semantic–pixel decoupled design under the standard 16× setting. This setting isolates the efect of the pixel branch, since pretrained semantic encoders already provide a suficiently dense token grid at 16× compression. As shown in Tab. 1, DC-SAE substantially improves reconstruction quality over semantic-only autoencoders. Compared with RAE, which obtains around 19 PSNR, DC-SAE reaches 27.80 PSNR and 0.26 rFID. Meanwhile, its latent space remains efective for difusion training, achieving strong ImageNet 256 × 256 generation performance after 80 epochs. This result confirms the basic motivation of our design. The frozen semantic branch provides a structured latent space that benefits generation, but it lacks reconstruction-critical details. Adding a trainable pixel branch closes most of this reconstruction gap without destroying the difusion-friendly structure of the semantic latent.

Pre-merge versus post-merge semantic compression. We next compare diferent ways of adapting pretrained semantic encoders to 32× compression. Motivated by recent MLLM eforts on high-compression vision tokenizers, a straightforward approach is to first extract dense visual tokens and then reduce the token count through local token merging. As shown in Fig. 3(c). In our setting, this corresponds to first extracting dense 16× semantic tokens and then merging each local token group into a compact 32× token. We refer to this family of methods as post-merge compression, since compression is applied after semantic feature extraction. We evaluate several variants, including average pooling [11, 12], frozen merger modules [3], and learnable mergers. Although these methods are simple and commonly used for reducing visual token count, they are not ideal for generative tokenization. As shown in Tab. 4, post-merge variants consistently underperform. For example, Qwen-ViT, with its frozen merger, obtains poor generation quality, and even learnable merger variants remain limited. This indicates that reducing tokens after semantic abstraction discards information that is dificult to recover later. In contrast, the resizing-based strategy used in Sec. 3.1 compresses the input before semantic abstraction. We refer to this strategy as pre-merge compression, since the semantic encoder directly forms representations under the target 32× token budget rather than compressing already-extracted dense semantic tokens. Empirically, this strategy leads to better generation quality, showing that compression should be introduced before semantic encoding.

Efectiveness of Spatial DeMerger. Finally, we evaluate the role of Spatial DeMerger in the decoder. At 32× compression, the latent grid is highly compact, which is favorable for difusion throughput but challenging for reconstruction. This issue is particularly important for ViT decoders, since they benefit from a denser spatial token grid for local detail recovery.

As shown in Fig. 3(a), under the standard DINOv2 pre-merge setting on ImageNet 256 × 256 with 32× compression, adding Spatial DeMerger consistently improves validation PSNR over the No-DeMerger variant. This supports our motivation that the compact difusion latent and the decoder-side token grid should be treated diferently: the difusion model should operate on the sparse 32× latent for eficiency, while the decoder benefits from a denser grid for high-fidelity reconstruction. Additional results with QwenViT-based semantic encoders are provided in Appendix A.4.

Joint semantic–pixel training versus late concatenation. A natural question is whether the gain of DC-SAE comes from joint semantic–pixel autoencoding, or simply from exposing DiT to semantic features. To isolate this factor, we compare two settings in Tab. 6: Joint and Late-Concat. Joint is the DC-SAE baseline, where the decoder reconstructs images from the concatenated semantic and pixel latents. Thus, the pixel branch is learned together with the semantic branch inside the autoencoder. In contrast, Late-Concat first trains a pixel autoencoder alone, whose decoder only receives the pixel latent. The frozen DINOv2 semantic latent is concatenated with the pixel latent only later, during DiT training. Therefore, DiT receives semantic features in both settings, but only Joint learns a coupled semantic–pixel latent space during autoencoding.

As shown in Tab. 6, late concatenation does not match the joint training variant. Although Late-Concat achieves better reconstruction quality than Joint, it gives worse generation performance, with an 80-epoch FID of 5.64 versus 4.02. To rule out a reconstruction–generation conflict, Late-Concat-E uses an earlier checkpoint of the same pixel autoencoder, whose reconstruction quality is closer to Joint. It still underperforms, reaching an 80-epoch FID of 5.40. These results indicate that DC-SAE’s gain is not merely from adding semantic features to DiT, but from jointly training the decoder to use semantic

Table 6 Standalone autoencoder ablation. Entries are FID / IS after DiT training.
<table><tr><td>Metric</td><td>Joint</td><td>Late-Concat</td><td>Late-Concat-E</td></tr><tr><td>PSNR ↑ rFID ↓</td><td>29.79 0.17</td><td>31.91</td><td>29.15 0.17</td></tr><tr><td></td><td></td><td>0.09</td><td></td></tr><tr><td>20 ep.</td><td>8.12 134.23</td><td>10.95 122.48</td><td>10.35 127.78</td></tr><tr><td>40 ep. 60 ep.</td><td>5.23 166.76</td><td>7.41 153.16</td><td>6.91 156.97</td></tr><tr><td></td><td>4.33 180.31</td><td>6.22 168.13</td><td>6.01 167.48</td></tr><tr><td>80 ep.</td><td>4.02 185.46</td><td>5.64 174.52</td><td>5.40 173.00</td></tr></table>

and pixel latents as a compatible representation for generation.

## 4.5 Discussion: Understanding Component Contributions to Reconstruction Fidelity

We further analyze where the reconstruction ability of DC-SAE comes from. Since DC-SAE contains both a semantic branch and a pixel branch, the decoder may rely on either branch for reconstruction. To study this, we reconstruct images using the full latent, the latent with the semantic component masked, and the latent with the pixel component masked.

As shown in Figure 5, masking out the semantic component has little efect on reconstruction quality, while masking out the pixel component causes severe degradation. This suggests that high-fidelity reconstruction mainly comes from the pixel branch. The semantic branch, in contrast, primarily provides high-level structure and a generation-friendly representation space rather than detailed appearance information.

This finding is consistent with the design of DC-SAE. The pixel branch preserves local texture, color, and spatial details for reconstruction, while the semantic branch guides the joint latent toward a structure that is easier for DiT to learn. Thus, DC-SAE does not require the frozen semantic encoder to carry all low-level information; instead, reconstruction and generative structure are handled by two complementary branches.

![](images/4f6e4be598fc89b4a2a5ebc01ca8e8529086a616be2949e798df8868c815afdd.jpg)  
Figure 5 Investigating where DC-SAE’s reconstruction ability comes from. We compare three reconstruction settings: using the full latent, masking the semantic component, and masking the pixel component. Masking the semantic branch has little impact on reconstruction quality, whereas masking the pixel branch causes severe degradation. This indicates that DC-SAE primarily relies on the pixel branch for reconstruction.

## 5 Conclusion

We propose DC-SAE, a decoupled compact semantic autoencoder for high-compression latent image generation. DC-SAE addresses the reconstruction–generation tradeof by combining a frozen semantic encoder for generation-friendly structure with an unconstrained pixel branch for low-level reconstruction details. We further identify post-hoc semantic token merging as a key obstacle to extending semantic autoencoders to 32× compression, as it discards spatial and low-level details after semantic abstraction. We address this with input-level semantic compression and a decoder-side Spatial DeMerger, enabling high-compression semantic tokenization with strong reconstruction fidelity. Experiments on ImageNet show that DC-SAE achieves strong reconstruction quality, faster difusion convergence, and competitive generation performance, reaching 29.79 PSNR and 3.37 gFID at 512 × 512 resolution with 32× compression. Text-to-image experiments further demonstrate its applicability beyond ImageNet: a 1.6B-parameter DiT using DC-SAE achieves 0.84 on GenEval and 86.007 on DPG-Bench at 1024 × 1024 resolution with CFG.

## References

[1] Alan Baade, Eric Ryan Chan, Kyle Sargent, Changan Chen, Justin Johnson, Ehsan Adeli, and Li Fei-Fei. Latent forcing: Reordering the difusion trajectory for pixel-space image generation, 2026. URL https://arxiv.org/ abs/2602.11401.

[2] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, Oğuzhan Fatih Kar, Elmira Amirloo, Alaaeldin El-Nouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. arXiv preprint arXiv:2502.13967, 2025.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

[4] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.

[5] Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are efective tokenizers for difusion models. In International Conference on Machine Learning, 2025.

[6] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, and Song Han. Deep compression autoencoder for eficient high-resolution difusion models. In International Conference on Learning Representations, 2025.

[7] Junyu Chen, Dongyun Zou, Wenkun He, Junsong Chen, Enze Xie, Song Han, and Han Cai. DC-AE 1.5: Accelerating difusion model convergence with structured latent space. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. arXiv:2508.00413.

[8] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3.5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

[9] Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, et al. Unified autoregressive visual generation and understanding with continuous tokens. arXiv preprint arXiv:2503.13436, 2025.

[10] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. GenEval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv. org/abs/2310.11513.

[11] Google DeepMind. Gemma 4. https://deepmind.google/models/gemma/gemma-4/, 2026. Accessed: 2026-05-02.

[12] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

[13] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009, 2022.

[14] Wenkun He, Yuchao Gu, Junyu Chen, Junyi Wu, Wenhang Ge, Dongyun Zou, Yujun Lin, Zhekai Zhang, Haocheng Xi, Muyang Li, et al. Dc-gen: Post-training difusion acceleration with deeply compressed latent space. In European Conference on Computer Vision, pages 249–269. Springer, 2026.

[15] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. ELLA: Equip difusion models with LLM fo enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024. URL https://arxiv.org/abs/2403.05135.

[16] Ziyuan Huang, Dandan Zheng, Cheng Zou, Rui Liu, Xiaolong Wang, Kaixiang Ji, Weilong Chai, Jianxin Sun, Libin Wang, Yongjie Lv, et al. Ming-univision: Joint image understanding and generation with a unified continuous tokenizer. arXiv preprint arXiv:2510.06590, 2025.

[17] Jiachun Jin, Zetong Zhou, Xiao Yang, Hao Zhang, Pengfei Liu, Jun Zhu, and Zhijie Deng. Latentum: Unleashing the potential of interleaved cross-modal reasoning via a latent-space unified model. arXiv preprint arXiv:2604.02097, 2026.

[18] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of difusion models. In Proc. CVPR, 2024.

[19] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. URL https://arxiv.org/abs/ 1312.6114.

[20] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

[21] Teng Li, Ziyuan Huang, Cong Chen, Yangfu Li, Yuanhuiyi Lyu, Dandan Zheng, Chunhua Shen, and Jun Zhang TC-AE: Unlocking token capacity for deep compression autoencoders. arXiv preprint arXiv:2604.07340, 2026.

[22] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization, 2024. URL https://arxiv.org/abs/2406.11838.

[23] Dongxu Liu, Yuang Peng, Haomiao Tang, Yuwei Chen, Chunrui Han, Zheng Ge, Daxin Jiang, and Mingxue Liao. Dgae: Difusion-guided autoencoder for eficient latent representation learning. arXiv preprint arXiv:2506.09644, 2025.

[24] Shizhan Liu, Xinran Deng, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Jie Tang. Delving into latent spectral biasing of video VAEs for superior difusability. arXiv preprint arXiv:2512.05394, 2025.

[25] Jiasen Lu, Liangchen Song, Mingze Xu, Byeongjoo Ahn, Yanjun Wang, Chen Chen, Afshin Dehghan, and Yinfei Yang. Atoken: A unified tokenizer for vision. arXiv preprint, 2025.

[26] Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321, 2025.

[27] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Bofi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and difusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.

[28] Meituan LongCat Team. LongCat-Next: Lexicalizing modalities as discrete tokens. arXiv preprint arXiv:2603.27538, 2026. URL https://arxiv.org/abs/2603.27538.

[29] NVIDIA. Cosmos tokenizer: A suite of image and video neural tokenizers. arXiv preprint arXiv:2501.03575, 2025.

[30] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[31] Yueming Pan, Ruoyu Feng, Qi Dai, Yuqi Wang, Wenfeng Lin, Mingyu Guo, Chong Luo, and Nanning Zheng. Semantics lead the way: Harmonizing semantic and texture modeling with asynchronous latent difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[32] William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

[33] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.

[35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[36] Christoph Schuhmann, Andreas Köpf, Richard Vencu, Theo Coombes, and Romain Beaumont. LAION COCO: 600M synthetic captions from LAION2B-en. https://laion.ai/blog/laion-coco/, 2022.

[37] Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent difusion model without variational autoencoder. In International Conference on Learning Representations, 2026. arXiv:2510.15301.

[38] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[39] Hao Tang, Chenwei Xie, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278, 2025.

[40] Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

[41] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled difusion transformer, 2025. URL https://arxiv.org/abs/2504.05741.

[42] Junfeng Wu, Dongliang Luo, Weizhi Zhao, Zhihao Xie, Yuanhao Wang, Junyi Li, Xudong Xie, Yuliang Liu, and Xiang Bai. Tokbench: Evaluating your visual tokenizer before visual generation. arXiv preprint arXiv:2505.18142, 2025.

[43] Yushu Wu, Yanyu Li, Ivan Skorokhodov, Anil Kag, Willi Menapace, Sharath Girish, Aliaksandr Siarohin, Yanzhi Wang, and Sergey Tulyakov. H3ae: High compression, high speed, and high quality autoencoder for video difusion models. arXiv preprint arXiv:2504.10567, 2025.

[44] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent difusion models. arXiv preprint arXiv:2501.01423, 2025.

[45] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats difusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.

[46] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training difusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.

[47] Zhengrong Yue, Haiyu Zhang, Xiangyu Zeng, Boyu Chen, Chenting Wang, Shaobin Zhuang, Lu Dong, Yi Wang, Limin Wang, and Yali Wang. Uniflow: A unified pixel flow tokenizer for visual understanding and generation. arXiv preprint arXiv:2510.10575, 2025.

[48] Zhengrong Yue, Taihang Hu, Mengting Chen, Haiyu Zhang, Zihao Pan, Tao Liu, Zikang Wang, Jinsong Lan, Xiaoyong Zhu, Bo Zheng, and Yali Wang. What matters for difusion-friendly latent manifold? prior-aligned autoencoders for latent difusion. arXiv preprint arXiv:2605.07915, 2026.

[49] Kaiwen Zha, Lijun Yu, Alireza Fathi, David A. Ross, Cordelia Schmid, Dina Katabi, and Xiuye Gu. Languageguided image tokenization for generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[50] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023.

[51] Xuan Zhao, Zhongyu Zhang, Yuge Huang, Yuxi Mi, Guodong Mu, Shouhong Ding, Jun Wang, Rizen Guo, and Shuigeng Zhou. Glotok: Global perspective tokenizer for image reconstruction and generation. arXiv preprint arXiv:2511.14184, 2025.

[52] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Difusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

[53] Shaobin Zhuang, Yiwei Guo, Canmiao Fu, Zhipeng Huang, Zeyue Tian, Zhang Ying, Chen Li, and Yali Wang. Wetok: Powerful discrete tokenization for high-fidelity visual reconstruction. arXiv preprint arXiv:2508.05599, 2025.

## A Appendix

## A.1 Limitations and Future Work.

Our evaluation covers class-conditional ImageNet generation and text-to-image generation at 1024 × 1024 resolution. However, the text-to-image study uses one 1.6B DiT configuration and a subset of internal data together with LAION-COCO; broader validation across training scales, datasets, and generator architectures remains future work. In addition, while we observe that semantic latents can accelerate difusion training and that input-level pre-merge compression outperforms post-hoc token merging, we do not yet have a complete understanding of which properties of semantic encoders are most important for generation. A deeper analysis of semantic encoder selection, as well as a principled explanation of when pre-merge or post-merge compression is preferable, remains an important direction for future work.

## A.2 Tokenizer Training Details

Datasets. We use ImageNet-1K for tokenizer training. Most experiments are conducted at $2 5 6 \times 2 5 6$ resolution. For $5 1 2 \times 5 1 2$ generation, we initialize the tokenizer from the trained $2 5 6 \times 2 5 6$ checkpoint and further finetune it on $5 1 2 \times 5 1 2$ images.

Tokenizer Architecture. Our tokenizer contains a frozen semantic encoder, a trainable pixel encoder, a fusion projection, and a ViT decoder with Spatial DeMerger. The semantic encoder is always frozen. The pixel encoder, fusion projection, Spatial DeMerger, and decoder are trained for reconstruction. The decoder architecture and discriminator largely follow RAE [52].

Training Configuration. We follow the basic tokenizer training setting of RAE [52], including the decoder and discriminator training setup. We set the L1 loss weight to λ = 1,LPIPS loss weight to $\lambda _ { \mathrm { l p i p s } } = 1$ and the adversarial loss weight to $\lambda _ { \mathrm { G A N } } = 0 . 5 .$ . At 256 × 256 resolution, we train the tokenizer for 30 epochs with a batch size of 512. For 512 × 512 resolution, we initialize from the converged $2 5 6 \times 2 5 6$ checkpoint and continue training for 20 epochs with a batch size of 512.

## A.3 Throughput Comparison with DC-AE

As shown in Tab. 7, we compare the end-to-end throughput of DC-AE and DC-SAE under the same settings. All throughput numbers are measured with a batch size of 32 on a single NVIDIA H200 GPU using bf16, 32 batch-sizes with 10 warmup iterations and 50 measurement iterations. DC-SAE consistently achieves higher end-to-end throughput than the corresponding DC-AE baseline across all evaluated resolutions, with the speedup remaining above 4× at every resolution.

Table 7 End-to-end throughput comparison between DC-AE and DC-SAE (imgs/s).
<table><tr><td>Model</td><td> $2 5 6 \times 2 5 6$ </td><td> $5 1 2 \times 5 1 2$ </td><td> $1 0 2 4 \times 1 0 2 4$ </td></tr><tr><td>DC-AE-f32c32</td><td>265.6</td><td>69.7</td><td>17.4</td></tr><tr><td>DC-SAE-f32c64</td><td>1233.5</td><td>330.8</td><td>76.8</td></tr><tr><td>DC-SAE-f32c256</td><td>1054.5</td><td>284.4</td><td>67.9</td></tr></table>

## A.4 Additional Analysis of Spatial DeMerger

To further demonstrate the efectiveness of Spatial DeMerger, we conduct additional experiments with QwenViT-based semantic encoders. In particular, we consider two common vision-encoder configurations used in MLLM settings. The first one uses the original frozen QwenViT with its native merger module. This follows the design used in Qwen-VL, where the vision encoder consists of a 16× patch-based QwenViT followed by a 2 × 2 merger trained together with the MLLM. The second one replaces the original QwenViT merger with a simple 2 × 2 average pooling operation, which is another commonly used way to reduce the spatial resolution of vision tokens in MLLM vision encoders.

In all experiments, the semantic encoder is kept fully frozen, and no additional pixel encoder is used, unlike the settings studied in the main paper. This setup isolates the role of Spatial DeMerger under a stricter reconstruction setting, where the decoder must recover spatial details only from the frozen semantic representation. As shown in Tab. 8, Spatial DeMerger consistently improves reconstruction quality for both the original QwenViT merger and the $2 \times 2$ average pooling variant. These results suggest that Spatial DeMerger is especially useful for RAE-like settings, where the encoder is fully frozen, and the decoder receives only sparse semantic tokens without auxiliary pixel-level features.

Table 8 Additional analysis of Spatial DeMerger with QwenViT-based semantic encoders. We compare the original QwenViT merger and a 2 × 2 average pooling variant, with and without Spatial DeMerger. Both semantic encoders are frozen, and no additional pixel encoder is used.
<table><tr><td rowspan="2">Semantic encoder setting</td><td colspan="2">Spatial DeMerger</td></tr><tr><td>w/o DeMerger w/ DeMerger</td><td></td></tr><tr><td>QwenViT w/ original merger</td><td>15.39</td><td>17.52</td></tr><tr><td>QwenViT w/ 2 × 2 avg pooling</td><td>15.50</td><td>17.64</td></tr></table>