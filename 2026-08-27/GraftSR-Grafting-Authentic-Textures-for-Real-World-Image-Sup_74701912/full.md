# GraftSR: Grafting Authentic Textures for Real-World Image Super-Resolution via Identical-Instance Guidance

Qifan Yu<sup>1</sup>, Haoran Bai<sup>1</sup>, Zongyao He<sup>1</sup>, Weijie He<sup>1</sup>, Sibin Deng<sup>1</sup>, Honggang Qi<sup>2</sup>, Ying Chen<sup>1∗</sup>,

<sup>1</sup>Taobao & Tmall Group of Alibaba, <sup>2</sup>University of Chinese Academy of Sciences

{yuqifan.yqf, baihaoran.bhr}@taobao.com

https://yuqifan1117.github.io/GraftSR/

## Abstract

Difusion-based real-world image super-resolution (SR) achieves impressive perceptual quality but inherently sufers from severe texture hallucination. To overcome this limitation, we propose GraftSR, a texture-reference-guided generative SR framework that leverages reference images of the identical instance to anchor the restoration of authentic textures. However, severe spatial misalignment between low-quality inputs and their references poses significant challenges, often leading to ambiguous transfer targets and background feature leakage. To address these issues, GraftSR employs a novel dual-mask reference guidance mechanism that systematically decouples the cross-view texture injection process. By explicitly isolating what authentic textures to extract from the reference and precisely localizing where to apply them within the target, GraftSR achieves robust texture transfer without relying on brittle spatial alignment. Furthermore, to bridge the critical gap in appropriate training data, we construct TexRefSR-141K, the first large-scale dataset providing high-quality reference tuples equipped with complementary spatial masks. Extensive experiments on our newly established benchmark, TexRefSR-Eval, demonstrate that GraftSR sets a new state-of-the-art. Notably, it reduces LPIPS by 20.2% over top-performing baselines, achieving superior reference-faithful restoration.

## Introduction

Difusion-based real-world image super-resolution (SR) (Wu et al. 2024; Duan et al. 2025; Fang et al. 2026) aims to restore high-quality (HQ) images from degraded observations by harnessing the immense potential of generative models (Wu et al. 2025; Peebles and Xie 2023; Labs et al. 2025; Podell et al. 2024), achieving perceptual quality far beyond that of earlier CNN-based approaches (Zhang et al. 2017; Soh and Cho 2022; Cui et al. 2024). However, a fundamental limitation persists: the generative restoration process inherently hallucinates textures that deviate from their authentic appearance due to the ill-posed nature of SR (see Fig. 1(a)). This severe hallucination destroys the fine-grained identity of the subject, strictly limiting its deployment in realworld scenarios that demand precise physical fidelity (e.g., e-commerce applications).

This limitation motivates the integration of reference images to anchor the generative process, aiming to restore a lowquality (LQ) input into a HQ output whose textures are faithful to a reference of the identical instance. Recently, generative editing approaches (e.g., Gemini-3-Pro) have emerged that can formally incorporate reference images to transfer customized textures into target regions. However, while these methods can achieve texture transfer, they often do so at the expense of the original geometric structure, resulting in poor content fidelity (see Fig. 1(c)). Therefore, it is of great interest to develop an efective texture-reference-guided generative SR framework that seamlessly transfers authentic reference textures while strictly preserving the structural and content consistency of the input image.

However, realizing this goal is highly challenging due to the severe cross-view spatial misalignments (e.g., varying poses, scales, and viewpoints) inherent in real-world identical-instance images. Existing reference-based SR methods rely heavily on brittle feature-level matching, sufering from severe artifacts beyond locally matched regions under such spatial misalignments (see Fig. 1(b)). Thus, achieving faithful cross-view texture transfer faces two intertwined challenges: (i) background feature leakage, since irrelevant reference contexts make it hard to isolate exactly what to reference; and (ii) ambiguous transfer target, since it is dificult to precisely localize where to transfer the reference textures.

To overcome these challenges, we present GraftSR, a novel generative framework driven by a dual-mask reference guidance mechanism. Built upon MMDiT backbone (Wu et al. 2025), it systematically tackles the aforementioned bottlenecks by explicitly injecting reference textures through two synergistic condition tokens: (i) mask-modulated reference tokens that isolate valid reference regions to explicitly determine what authentic textures to extract; and (ii) regionaware semantic tokens that precisely localize where the texture should be applied within the target. By jointly attending to these tokens within a unified sequence, GraftSR implicitly learns robust cross-view texture aggregation without relying on brittle spatial matching. Finally, an adversarial training objective (Fang et al. 2026) distills the generative process for eficient one-step image SR framework.

Furthermore, to address the critical absence of identicalinstance training data, we curate TexRefSR-141K, the first large-scale dataset tailored for this task. Through an automated data construction pipeline, we pair images of strictly identical instances from multi-view galleries and supplement them with explicit complementary spatial masks. Specifically, the reference mask isolates the authentic texture source, while the target mask demarcates the exact restoration region. Comprising over 141K high-quality tuples across 61K distinct instances, this dataset provides the essential spatial selectivity to decouple textures from complex backgrounds, establishing a vital data foundation for the community.

![](images/fc4cb3de61d739bce1e5dc9ea1499abb3c243fa11aa1cef4eb7bdd919a8b7a01.jpg)  
Figure 1: GraftSR achieves reference-faithful image SR of fine-grained details (e.g., material textures and delicate prints), whereas existing methods exhibit distinct limitations: (a) Difusion-based SR generates hallucinated textures; (b) Referencebased SR introduces artifacts beyond matched regions; and (c) Generative editing model sufers from poor contentfidelity.

Extensive experiments on our newly established TexRefSR-Eval benchmark, which enables faithful evaluation against identical-instance references, validate the superiority of GraftSR against state-of-the-art baselines. Quantitatively, a notable 20.2% reduction in LPIPS over top-performing methods confirms that the generated details are faithfully anchored to reference textures rather than arbitrarily hallucinated. Qualitatively, GraftSR achieves highly precise crossview texture transfer while strictly preserving the original geometric structures, delivering reference-faithful restoration (see Fig. 1(d)). In summary, our main contributions are:

• We propose GraftSR, a novel texture-reference-guided generative SR framework that suppresses generative hallucinations and achieves highly faithful texture transfer from cross-view identical-instance references.

• We construct TexRefSR-141K and TexRefSR-Eval, the first large-scale dataset and benchmark suite providing highquality identical-instance tuples equipped with complementary spatial masks.

• Extensive experiments demonstrate that GraftSR establishes a new state-of-the-art in reference-guided SR, bridging the gap between perceptual quality and texture fidelity for faithful texture restoration.

## Related Work

## Difusion-based Image Restoration

Recent advances in difusion-based generative models (Rombach et al. 2022; Esser et al. 2024; Wu et al. 2025) have shifted image restoration from pixel-level regression (Zhang et al. 2018b; Liang et al. 2021) to perceptual synthesis. Early multi-step approaches have successfully adapted largescale difusion models for image super-resolution (Yu et al. 2024; Duan et al. 2025). To mitigate the heavy computational overhead, one-step distillation models have recently emerged. Specifically, OSEDif (Wu et al. 2024) leverages variational score distillation, while PiSA-SR (Sun et al. 2025) and ODTSR (Fang et al. 2026) decouple generation branches to optimize perceptual quality and semantic control simultaneously. To capture finer visual details, VOSR (Wu et al. 2026) departs from standard T2I paradigms by proposing a vision-only framework that substitutes textual prompts with deep visual features extracted from pretrained encoders. Despite their impressive performance, these blind restoration methods lack explicit target texture priors, often resulting in spurious artifacts or hallucinated details. In contrast, our work leverages reference images of the identical instance as precise texture anchors to faithfully transfer reference textures during generative restoration.

## Reference-based Image Restoration

To bypass the ill-posed issues of single-image paradigms, reference-based super-resolution (RefSR) restores highfidelity details based on auxiliary images. Early matchingbased methods (Zhang et al. 2019; Yang et al. 2020; Zhang et al. 2022; Jiang et al. 2021; Cao et al. 2022; Zhang et al. 2023; Lu et al. 2021) and advanced weighting strategies (Guo et al. 2024; Varanka et al. 2024) heavily rely on localized patch alignment. Although recent difusion-based RefSR models (Wang et al. 2026; Zhao et al. 2026) attempt to process unaligned references, deriving reference pairs from semantically approximate images or single-image crops inherently restricts them to global feature matching. Consequently, they inevitably sufer from background leakage when facing unaligned references. In contrast, our method achieves robust texture transfer via dual-mask reference guidance.

## Method

Unlike conventional blind super-resolution, our texturereference-guided SR framework aims to reconstruct degraded observations by faithfully transferring authentic textures from identical-instance reference images. Given a LQ input $I _ { L Q }$ and a HQ texture reference I of the identical instance, the goal is to synthesize a HQ output $\hat { I } _ { H Q }$ that strictly preserves the spatial structure of $I _ { L Q }$ while achieving precise texture fidelity to $I _ { R E F }$

![](images/87c2cb156c0486f7bede774cfe360c0e03b819475a9d1e1c6b05f15b37537ca2.jpg)  
Figure 2: Overview architecture of GraftSR. Driven by a dual-mask reference guidance mechanism, it extracts mask-modulated reference tokens and region-aware semantic tokens to condition the MMDiT blocks for one-step reference-faithful restoration.

## Dual-Mask Reference-Guided Architecture

To achieve authentic texture fidelity without relying on brittle spatial alignment, we propose GraftSR, introducing a novel dual-mask reference guidance mechanism to explicitly resolve the critical ambiguities of what to reference and where to transfer. Built upon the MMDiT backbone (Wu et al. 2025), we design two synergistic conditioning tokens, namely mask-modulated reference tokens and region-aware semantic tokens, to efectively overcome these two challenges. As illustrated in Fig. 2, these extracted tokens jointly condition the cascaded MMDiT blocks to denoise the dualnoise latent, which is ultimately distilled via a one-step adversarial objective (Fang et al. 2026) for eficient image SR.

Semantic and Spatial Condition Extraction. To explicitly establish the semantic and spatial correspondences of the identical instance between the LQ input $I _ { L Q }$ and the HQ reference $I _ { R E F } ,$ we employ an automated extraction process prior to the difusion model. First, we leverage a Vision-Language Model (VLM) (Bai et al. 2025) to jointly analyze both images. The VLM identifies the semantic name of the shared instance and generates a comprehensive text caption $T$ for $I _ { L Q }$ , which acts as an optional textual condition for the MMDiT backbone. Subsequently, using the identified instance name as a prompting cue, we employ SAM3 (Carion et al. 2025) to perform zero-shot segmentation. This step extracts the target region mask $M _ { T G T }$ from $I _ { L Q }$ and the reference mask $M _ { R E F }$ from $I _ { R E F }$ , explicitly localizing the identical instance across both views. These extracted masks and the caption seamlessly serve as the fundamental conditions for the subsequent token generation.

Dual-Mask Synergistic Tokens. To explicitly decouple the cross-view texture injection process, we formulate two synergistic conditioning tokens driven by the extracted masks. First, to determine what to reference, we encode $I _ { R E F }$ using a frozen VAE. Since unaligned identical-instance references inevitably contain irrelevant backgrounds, we modulate the reference latent with the corresponding spatial mask to explicitly isolate valid texture-bearing regions:

$$
z _ { \mathrm { r e f } } = \mathcal { E } _ { \mathrm { V A E } } ( I _ { R E F } ) + \phi ( \mathcal { D } ( M _ { R E F } ) ) ,\tag{1}
$$

where $z _ { \mathrm { r e f } }$ denotes the mask-modulated reference tokens, ${ \mathcal { E } } _ { \mathrm { V A E } }$ denotes the frozen VAE encoder, D is the downsampling operation, and $\phi ( \cdot )$ is a learnable projection. Rather than applying a hard mask that might disrupt the continuous latent space, this additive modulation softly intervenes in the feature space, isolating authentic textures while preserving the model’s general generative capability.

Second, to accurately designate where to transfer these extracted textures, we establish a high-level spatial-semantic understanding of the target. We extract multi-modal condition tokens via the frozen Qwen2.5-VL encoder (Bai et al. 2025):

$$
z _ { \mathrm { s e m } } = \mathcal { E } _ { \mathrm { V L M } } ( I _ { L Q } , ~ M _ { T G T } , ~ I _ { R E F } , ~ T ) ,\tag{2}
$$

where ${ \mathcal { E } } _ { \mathrm { V L M } }$ denotes the Qwen2.5-VL encoder. By jointly embedding the text prompt T with these visual-spatial clues, ${ \mathcal { E } } _ { \mathrm { V L M } }$ yields region-aware semantic tokens $z _ { \mathrm { s e m } }$ . Together, $z _ { \mathrm { r e f } }$ and $z _ { \mathrm { s e m } }$ form a complementary pair, precisely guiding the network to transfer the right textures to the right locations.

Model Training. Following (Fang et al. 2026), we adopt a dual-noise strategy to optimally balance perceptual quality and texture fidelity. Specifically, dual-level noises are injected into the latent representation of the LQ input to formulate the noisy target tokens. To train the MMDiT backbone as an eficient one-step SR model, these noisy tokens are concatenated along the sequence dimension with our explicitly extracted conditions and subsequently processed by the MMDiT blocks. The predicted denoised latent is then decoded back to the pixel space, and the model is optimized using a combination of MSE, DISTS (Ding et al. 2020), and an adversarial objectives. For adversarial supervision, we employ a relativistic GAN loss driven by a Wan2.1-1.3B-based discriminator (Wan et al. 2025) to guarantee high-fidelity perceptual generation. More training details are deferred to the Supplementary Materials.

![](images/5a4bf18806193646e13d0ce3c2814f0877d1e819c7df33625800b9e6350b1525.jpg)  
Figure 3: The proposed VLM-driven data curation pipeline. It retrieves identical-instance triplets from multi-view galleries and extracts the complementary masks and unified caption to support authentic texture transfer.

## Identical-Instance Dataset Construction

To efectively train GraftSR, the availability of large-scale, strictly identical-instance data is imperative. Prevailing reference-based SR datasets merely pair semantically approximate images, inevitably synthesizing textures that deviate from the target instance’s authentic appearance. To fundamentally shift this paradigm, we exploit large-scale ecommerce multi-view image galleries, which uniquely capture identical instances across diverse viewpoints, thus providing authentic textures for cross-view transfer. Building on this insight, we design an automated VLM-driven curation pipeline and construct TexRefSR-141K, the first large-scale dataset tailored for identical-instance texture-reference SR. As illustrated in Fig. 3, the pipeline first assembles identicalinstance triplets from the multi-view gallery, and then annotates them with a unified caption and complementary mask pairs via our semantic and spatial condition extraction.

Identical-Instance Triplet. Departing from conventional reference paradigms, we retrieve triplets that depict strictly identical instances from multi-view galleries, guaranteeing genuine texture correspondence rather than mere semantic resemblance. Since faithful restoration demands a texturerich reference, we deliberately anchor the retrieval process on the reference image. Specifically, a quality filter selects an optimal HQ product image to serve as $I _ { R E F }$ . Guided by this anchor, the pipeline retrieves a relevant HQ view of the identical instance to act as the ground-truth $I _ { H Q }$ . A highorder degradation model (Wang et al. 2021) is then applied to $I _ { H Q }$ to synthesize the LQ input $I _ { L Q }$ , formulating the core $\{ \dot { I } _ { R E F } , I _ { H Q } , I _ { L Q } \}$ triplet. During practical inference, $I _ { L Q }$ originates directly from real-world observations, and its corresponding $I _ { R E F }$ is retrieved from the instance gallery.

Caption and Complementary Masks. Beyond mere image pairing, we also equip each triplet with explicit spatialsemantic guidance to resolve the critical ambiguities of what and where to transfer. Leveraging the same semantic and spatial condition extraction introduced in our architecture, we obtain the unified caption T and the complementary mask pair $\{ M _ { R E F } , M _ { T G T } \}$ for each triplet. These masks cleanly isolate valid texture regions from irrelevant backgrounds and precisely localize the target instance across unaligned views. To maximize textual supervision quality during training, we derives $T$ and $M _ { T G T }$ utilizing the clearer $I _ { H Q }$ . Additionally, random morphological dilation is applied to both spatial masks to ensure robustness along delicate boundaries (e.g., lace and decorative edges).

![](images/b4a5c42964c2d0c52ce504d36cda8ddb8649c271025b75506b505222b6ff218c.jpg)

![](images/c3da5f2e27734e548090817636d210703d089cebfa47fc962575ee2cb1870c2c.jpg)  
Figure 4: Dataset statistics. (a) Reference distribution over the top-15 leaf categories, with long-tail categories merged as "Others". (b) Number of HQ images per category (bars) and the average number of HQ images per reference (line).

Quality Assessment and Dataset Statistics. To guarantee the reliability of reference guidance, an initial filtering stage strictly assesses candidate images before masking: it computes embedding similarities to discard mismatched pairs, eliminates degraded samples via a quality predictor, and rejects extreme close-ups that lose recognizable instance identity. Through this rigorous pipeline, we construct TexRefSR-141K, comprising over 141K high-quality tuples across 61K diverse fashion products (Fig. 4 (a)). Crucially, each reference maps to multiple target ground-truths (Fig. 4 (b)). This 1-to-N identical-instance mapping yields rich cross-view texture correspondences, endowing the network with the spatial selectivity essential for authentic texture-reference-guided SR.

Beyond the training data, we further curate TexRefSR-Eval, the first benchmark dedicated to identical-instance referenceguided SR, comprising 300 manually verified test cases. It consists of two complementary subsets: TexRefSR-Eval-Syn (250 samples), whose LQ inputs are synthesized from highquality ground truths for paired full-reference evaluation, and TexRefSR-Eval-Real (50 samples), whose LQ inputs are naturally degraded images collected from raw e-commerce galleries for real-world generalization. Together, they form a complete foundation for training and benchmarking this task.

## Experiments

## Experimental Setup

Benchmarks. We evaluate reference-guided real-world image SR on our newly established TexRefSR-Eval benchmark, whose scale factor follows each subset’s degradation (4× for the Syn subset and native resolution for the Real subset). To further assess general SR performance, we adopt two standard real-world datasets, RealSR (Cai et al. 2019) and DRealSR (Wei et al. 2020), both under 4× upscaling following OSEDif (Wu et al. 2024) for fair comparison.

Baselines. We compare against methods from three categories: (i) Generative SR, including multi-step methods SUPIR (Yu et al. 2024) and DiT4SR (Duan et al. 2025), and one-step methods OSEDif (Wu et al. 2024), PiSA-SR (Sun et al. 2025), ODTSR (Fang et al. 2026), and VOSR (Wu et al. 2026); (ii) Reference-based SR, including RefSR (Guo et al. 2024), AdaRefSR (Wang et al. 2026), and Garment-Zoom (Zhao et al. 2026); and (iii) Commercial editing models, including Gemini-3-Pro and GPT-Image-2. For opensource methods, we obtain all results using their oficial codes and models under default settings, while closed-source commercial models are evaluated through their oficial APIs.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="4">Full-Reference</td><td colspan="4">No-Reference</td></tr><tr><td>LPIPS↓</td><td>DISTS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MUSIQ↑</td><td>NIQE↓</td><td>CLIPIQA↑</td><td>MANIQA↑</td></tr><tr><td rowspan="6">Open-source Generative SR</td><td>OSEDiff</td><td>0.2326</td><td>0.1444</td><td>28.3752</td><td>0.8471</td><td>65.2201</td><td>4.8256</td><td>0.6631</td><td>0.5250</td></tr><tr><td>SUPIR</td><td>0.2208</td><td>0.1321</td><td>29.3151</td><td>0.8382</td><td>61.5577</td><td>5.2317</td><td>0.5922</td><td>0.5225</td></tr><tr><td>DiT4SR</td><td>0.3816</td><td>0.2147</td><td>25.6121</td><td>0.8071</td><td>33.1845</td><td>7.3649</td><td>0.5769</td><td>0.4253</td></tr><tr><td>ODTSR</td><td>0.1983</td><td>0.1048</td><td>29.5098</td><td>0.8431</td><td>69.2601</td><td>4.7789</td><td>0.6176</td><td>0.5129</td></tr><tr><td>PiSA-SR</td><td>0.2247</td><td>0.1343</td><td>28.5478</td><td>0.8458</td><td>66.7754</td><td>4.6765</td><td>0.6831</td><td>0.5338</td></tr><tr><td>VOSR-1.4b</td><td>0.2349</td><td>0.1247</td><td>28.3661</td><td>0.8290</td><td>71.1585</td><td>4.3381</td><td>0.6873</td><td>0.5523</td></tr><tr><td rowspan="3">Open-source Reference-based SR</td><td>RefSR</td><td>0.4384</td><td>0.2685</td><td>28.6190</td><td>0.8297</td><td>24.0680</td><td>8.1108</td><td>0.4106</td><td>0.3259</td></tr><tr><td>AdaRefSR</td><td>0.6071</td><td>0.2785</td><td>16.2155</td><td>0.7154</td><td>32.1215</td><td>7.7052</td><td>0.5645</td><td>0.4367</td></tr><tr><td>GarmentZoom</td><td>0.4156</td><td>0.2643</td><td>25.8811</td><td>0.7255</td><td>34.1616</td><td>6.1379</td><td>0.4360</td><td>0.4160</td></tr><tr><td>Commercial Editing Models</td><td>GPT-Image-2</td><td>0.4474</td><td>0.2154</td><td>18.3255</td><td>0.7665</td><td>34.4500</td><td>7.4755</td><td>0.5781</td><td>0.4355</td></tr><tr><td rowspan="2"></td><td>Gemini-3-Pro</td><td>0.2065</td><td>0.1034</td><td>26.7874</td><td>0.8222</td><td>64.9845</td><td>4.9920</td><td>0.6216</td><td>0.5048</td></tr><tr><td>Graft (Ours)</td><td>0.1583</td><td>0.0986</td><td>30.1060</td><td>0.8494</td><td>69.8535</td><td>4.9346</td><td>0.7275</td><td>0.5270</td></tr></table>

Table 1: Quantitative comparison on the TexRefSR-Eval-Syn benchmark, where ground-truth references are available. Both full-reference and no-reference metrics are reported. Red bold and blue underline denote the best and the second-best results.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="4">No-Reference</td><td colspan="3">VLM Evaluation (GPT-4.1-global)</td></tr><tr><td>NIQE↓</td><td>MUSIQ↑</td><td>CLIPIQA↑</td><td>MANIQA↑</td><td>PQ↑</td><td>TC↑</td><td>Overall↑</td></tr><tr><td rowspan="6">Open-source Generative SR</td><td>OSEDiff</td><td>3.9329</td><td>70.5668</td><td>0.6395</td><td>0.6292</td><td>3.320</td><td>1.740</td><td>2.530</td></tr><tr><td>SUPIR</td><td>4.2105</td><td>59.6691</td><td>0.4778</td><td>0.5741</td><td>3.380</td><td>1.960</td><td>2.670</td></tr><tr><td>DiT4SR</td><td>3.7190</td><td>72.5782</td><td>0.6428</td><td>0.6485</td><td>3.720</td><td>2.040</td><td>2.880</td></tr><tr><td>ODTSR</td><td>4.2809</td><td>71.5261</td><td>0.6140</td><td>0.6580</td><td>2.760</td><td>1.560</td><td>2.160</td></tr><tr><td>PiSA-SR</td><td>4.2523</td><td>65.5855</td><td>0.4913</td><td>0.5789</td><td>2.840</td><td>1.660</td><td>2.250</td></tr><tr><td>VOSR-1.4b</td><td>3.8273</td><td>69.7636</td><td>0.6227</td><td>0.6399</td><td>3.880</td><td>2.380</td><td>3.130</td></tr><tr><td rowspan="3">Open-source Reference-based SR</td><td>RefSR</td><td>5.0782</td><td>53.8617</td><td>0.3251</td><td>0.5030</td><td>2.200</td><td>1.280</td><td>1.740</td></tr><tr><td>AdaRefSR</td><td>5.7189</td><td>46.2674</td><td>0.3625</td><td>0.5087</td><td>2.580</td><td>1.560</td><td>2.070</td></tr><tr><td>GarmentZoom</td><td>5.6298</td><td>31.4101</td><td>0.3625</td><td>0.4417</td><td>2.380</td><td>1.840</td><td>2.110</td></tr><tr><td>Commercial</td><td>GPT-Image-2</td><td>4.9825</td><td>65.9262</td><td>0.5667</td><td>0.6159</td><td>3.020</td><td>2.100</td><td>2.560</td></tr><tr><td rowspan="2">Editing Models</td><td>Gemini-3-Pro</td><td>5.1077</td><td>56.5937</td><td>0.4633</td><td>0.5277</td><td>3.880</td><td>2.700</td><td>3.290</td></tr><tr><td>GraftSR (Ours)</td><td>3.9881</td><td>73.0045</td><td>0.6409</td><td>0.6507</td><td>3.980</td><td>2.840</td><td>3.410</td></tr></table>

Table 2: Quantitative comparison on the TexRefSR-Eval-Real benchmark with no-reference metrics and VLM-based metrics.

Evaluation Metrics. For evaluation, we adopt both fullreference (FR) and no-reference (NR) metrics. The FR metrics consist of the pixel-level metrics PSNR and SSIM (Wang et al. 2004) and the perceptual-level metrics LPIPS (Zhang et al. 2018a) and DISTS (Ding et al. 2020). For NR image quality assessment, we report the results of NIQE (Mittal et al. 2012), MUSIQ (Ke et al. 2021), CLIPIQA (Wang, Chan, and Loy 2023), and MANIQA (Yang et al. 2022). In addition, to evaluate real-world reference-guided restoration, we introduce two VLM-based measures alongside their average (Overall): perceptual quality (PQ) of the predicted images to assess the overall visual realism of the restoration, and texture consistency (TC) computed within the masked regions between the reference and predicted images to evaluate their texture fidelity.

## Main Results on Texture-Guided Real-World SR

Quantitative comparisons on the TexRefSR-Eval benchmarks are reported in Tables 1 and 2. Based on these results, we draw the following conclusions: (i) Our method achieves the optimal balance between reference fidelity and perceptual quality. As shown in Table 1, GraftSR dominates all full-reference metrics. Specifically, it reduces LPIPS by 20.2% compared to the second-best method (ODTSR) and is the only approach to achieve a DISTS score strictly below 0.10. This physical fidelity is accomplished without compromising visual aesthetics, as evidenced by securing the highest CLIPIQA (0.7275) and highly competitive MUSIQ scores. (ii) Our method yields the best texture consistency in real-world e-commerce scenarios. On TexRefSR-Eval-Real (Table 2), GraftSR leads the VLM evaluation, which assesses semantic texture alignment. It achieves the highest texture consistency (2.840, compared to 2.700 for Gemini-3-Pro and 2.380 for VOSR), perceptual quality (3.980), and overall score (3.410). (iii) Our method outperforms powerful closed-source commercial models in texture-guided realworld SR tasks. Compared to GPT-Image-2 and Gemini-3-Pro, our method demonstrates significantly higher fidelity on the synthetic benchmark (PSNR of 30.11 dB vs. 18.33 dB and 26.79 dB) and a superior VLM overall score on the realworld benchmark (3.410 vs. 2.560 and 3.290). This gap arises because commercial models tend to over-edit and introduce hallucinated details, whereas GraftSR strictly preserves the pixel-level textures critical for reliable restoration.

![](images/d42150b9ad329bfc94e0986fa84f7c12e2c85ca4bfab4914c20f2e2dd6943fb4.jpg)  
Figure 5: Qualitative comparison on TexRefSR-Eval, including one synthetic case with GT (top) and two real cases (bottom). Our method achieves reference-faithful restoration with superior fidelity and texture consistency. (Zoom in for best view)

-quality texture-reference-guided image super-resolution.

## Main Results on General Real-World SR

Table 3 reports the results on the RealSR and DRealSR benchmarks where the reference branch is left empty. This setup explicitly evaluates whether our framework maintains its general restoration capability without reference guidance. Notably, GraftSR achieves the highest fidelity on both benchmarks, outperforming the second-best PiSA-SR by a significant margin on DRealSR (+1.09,dB PSNR and +0.029 SSIM). Furthermore, GraftSR maintains a solid balance between fidelity and perception, ranking second in LPIPS while securing competitive no-reference scores (MUSIQ and MANIQA). These results demonstrate the strong robustness of our GraftSR when references are unavailable. We attribute this to injecting reference cues via sequence concatenation, which makes the reference branch a detachable component that gracefully disengages when references are absent, allowing the framework to subsume general real-world SR.

<table><tr><td>Dataset</td><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>MANIQA↑</td></tr><tr><td rowspan="7">ReSR</td><td>SUPIR</td><td>23.65</td><td>0.6620</td><td>0.3541</td><td>62.09</td><td>0.5780</td></tr><tr><td>DiT4SR</td><td>23.50</td><td>0.6657</td><td>0.3215</td><td>67.76</td><td>0.6564</td></tr><tr><td>OSEDiff</td><td>25.15</td><td>0.7341</td><td>0.2920</td><td>69.08</td><td>0.6335</td></tr><tr><td>PiSA-SR</td><td>25.40</td><td>0.7418</td><td>0.2672</td><td>70.14</td><td>0.6551</td></tr><tr><td>VOSR-1.4B</td><td>25.23</td><td>0.7175</td><td>0.2732</td><td>70.58</td><td>0.6443</td></tr><tr><td>ODTSR</td><td>25.07</td><td>0.7361</td><td>0.2398</td><td>68.29</td><td>0.6622</td></tr><tr><td>GraftSR (Ours)</td><td>25.51</td><td>0.7512</td><td>0.2559</td><td>70.37</td><td>0.6676</td></tr><tr><td rowspan="7">DRSR</td><td>SUPIR</td><td>25.09</td><td>0.6460</td><td>0.4243</td><td>58.79</td><td>0.5471</td></tr><tr><td>DiT4SR</td><td>25.40</td><td>0.6657</td><td>0.3869</td><td>65.75</td><td>0.6287</td></tr><tr><td>OSEDiff</td><td>27.92</td><td>0.7835</td><td>0.2968</td><td>64.65</td><td>0.5895</td></tr><tr><td>PiSA-SR</td><td>28.32</td><td>0.7804</td><td>0.2960</td><td>66.11</td><td>0.6146</td></tr><tr><td>VOSR-1.4B</td><td>27.88</td><td>0.7413</td><td>0.3260</td><td>66.04</td><td>0.6053</td></tr><tr><td>ODTSR</td><td>28.14</td><td>0.7736</td><td>0.2592</td><td>62.86</td><td>0.6227</td></tr><tr><td>GraftSR (Ours)</td><td>29.04</td><td>0.8052</td><td>0.2858</td><td>66.73</td><td>0.6376</td></tr></table>

Table 3: Quantitative comparison on geneal real-world SR. our GraftSR still achieves strong generalization in both fidelity and perceptual quality even without reference images.

![](images/3196785b64c94bc47a510792d87685d43e3ac13b56e80554581f46e25a8cc08d.jpg)  
Figure 6: Robustness analysis of GraftSR with absent or mismatched references during inference.

<table><tr><td>Variant</td><td>NIQE↓</td><td>MUSIQ↑</td><td>CLIPIQA↑</td><td>MANIQA↑</td><td>PQ↑</td><td>TC↑</td></tr><tr><td colspan="7">Ablation on Texture Reference Injection</td></tr><tr><td>Cross-attn</td><td>4.2210</td><td>70.55</td><td>0.5236</td><td>0.5874</td><td>2.660</td><td>1.840</td></tr><tr><td>Cross-attn + Dual-Mask</td><td>3.9818</td><td>71.83</td><td>0.5974</td><td>0.6168</td><td>3.000</td><td>2.000</td></tr><tr><td>Joint-attn</td><td>4.4200</td><td>71.07</td><td>0.5835</td><td>0.6278</td><td>3.400</td><td>2.100</td></tr><tr><td colspan="7">Ablation on Dual-Mask Reference Guidance</td></tr><tr><td>w/o zref</td><td>4.2911</td><td>72.60</td><td>0.6222</td><td>0.6469</td><td>3.500</td><td>2.220</td></tr><tr><td>w/o MTGT</td><td>4.6184</td><td>71.25</td><td>0.5688</td><td>0.6279</td><td>3.500</td><td>2.360</td></tr><tr><td>wlo MREF</td><td>4.4125</td><td>72.23</td><td>0.5959</td><td>0.6375</td><td>3.640</td><td>2.440</td></tr><tr><td>GraftSR (Ours)</td><td>3.9881</td><td>73.00</td><td>0.6409</td><td>0.6507</td><td>3.980</td><td>2.840</td></tr></table>

Table 4: Ablation studies on the texture reference injection strategy and the dual-mask reference guidance module.

## Qualitative Comparison

Figure 5 presents a qualitative comparison on TexRefSR-Eval, including one synthetic case with ground truth (GT) and two real cases. On the synthetic case, our result attains both high fidelity to the GT and strong texture consistency with the reference on the delicate lace (j1), whereas Gemini-3-Pro alters the background bag, degrading fidelity to the GT (i1). On real cases, blind generative SR methods (e.g., ODTSR) hallucinate textures inconsistent with the reference (f2–f3), while reference-based SR methods (e.g., AdaRefSR, GarmentZoom) fail beyond matched regions and produce artifacts due to brittle spatial misalignment (g2–g3, h2–h3). In contrast, only GraftSR faithfully recovers the flufy sherpa texture (j2) and the fine knit with correct buttons (j3), demonstrating its robustness in texture-reference-guided SR.

![](images/e91750c95a5b57379b39a66cf0fca10494346a5382406eb9b4833e8ff1d79615.jpg)  
Figure 7: Visual comparison of the ablation studies.

![](images/e63bbe6344e9b7fd289bcc88818c221cf0fab8687a149757fae6cc597631d434.jpg)  
Figure 8: The user study results. Our method is consistently preferred over SOTA ODTSR and Gemini-3-Pro.

## In-depth Analysis

Texture Reference Injection. Table 4 (top) and Fig. 7 (c–d) evaluate injection strategies. Cross-attn lacks texture consistency despite mask guidance. Unmasked Joint-attn improves texture transfer but introduces structural artifacts (higher NIQE). Ultimately, our full model synergizes joint attention for maximal transfer and dual masks for precise alignment.

Dual-Mask Reference Guidance. As shown in Table 4 (bottom) and Fig. 7 (e–g), removing the reference token z<sub>ref</sub> degrades texture consistency, failing to synthesize lace textures (Fig. 7 (e)). In addition, discarding either spatial mask (M<sub>TGT</sub> or M<sub>REF</sub>) disrupts the reference-target alignment, causing textures to nearly vanish (Fig. 7 (f–g)). In contrast, full GraftSR accurately localizes delicate textures (Fig. 7 (h)), confirming all components are indispensable.

User Study. We conduct a Two-Alternative Forced Choice (2AFC) study across 50 diverse scenes, where participants judge our results against the strongest SOTA methods based on perceptual quality and texture consistency. As shown in Fig. 8, our method wins 82% of votes against ODTSR and 74% against the leading Gemini-3-Pro. This demonstrates a strong alignment with human visual preferences.

Robustness Analysis and Limitations. As shown in Fig. 6, we test GraftSR with a mismatched reference or no reference during inference. In both cases, it still produces faithful textures comparable to using the correct reference and clearly outperforms ODTSR. Nevertheless, when inputs are severely degraded, GraftSR trades of some fine-grained details to preserve fidelity, which we leave for future work.

## Conclusion

In this paper, we present GraftSR, a texture-reference-guided generative SR framework that restores authentic textures from identical-instance references to overcome the texture hallucination of difusion-based real-world SR. To handle the inherent misalignment between low-quality inputs and their references, GraftSR introduces a dual-mask reference guidance mechanism that decouples what textures to extract from where to apply them, enabling robust transfer without brittle spatial alignment. We further construct TexRefSR-141K and TexRefSR-Eval, the first large-scale identical-instance dataset and benchmark with complementary spatial masks. Extensive experiments show that GraftSR sets a new stateof-the-art for highly faithful texture restoration.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Cai, J.; Zeng, H.; Yong, H.; Cao, Z.; and Zhang, L. 2019. Toward real-world single image super-resolution: A new benchmark and a new model. In Proceedings of the IEEE/CVF international conference on computer vision, 3086–3095.

Cao, J.; Liang, J.; Zhang, K.; Li, Y.; Zhang, Y.; Wang, W.; and Gool, L. V. 2022. Reference-based image super-resolution with deformable attention transformer. In European conference on computer vision, 325–342. Springer.

Carion, N.; Gustafson, L.; Hu, Y.-T.; Debnath, S.; Hu, R.; Suris, D.; Ryali, C.; Alwala, K. V.; Khedr, H.; Huang, A.; et al. 2025. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719.

Cui, Y.; Ren, W.; Cao, X.; and Knoll, A. 2024. Revitalizing convolutional network for image restoration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12): 9423–9438.

Ding, K.; Ma, K.; Wang, S.; and Simoncelli, E. P. 2020. Image quality assessment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence, 44(5): 2567–2581.

Duan, Z.-P.; Zhang, J.; Jin, X.; Zhang, Z.; Xiong, Z.; Zou, D.; Ren, J. S.; Guo, C.; and Li, C. 2025. Dit4sr: Taming difusion transformer for real-world image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 18948–18958.

Esser, P.; Kulal, S.; Blattmann, A.; Entezari, R.; Müller, J.; Saini, H.; Levi, Y.; Lorenz, D.; Sauer, A.; Boesel, F.; et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning.

Fang, Y.; Chen, Y.; Yin, S.; Hu, Q.; Yao, J.; Zhang, Y.; Zhang, X.; and Wang, Y. 2026. One-step difusion transformer for controllable real-world image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23440–23450.

Guo, H.; Dai, T.; Ouyang, Z.; Zhang, T.; Zha, Y.; Chen, B.; and Xia, S.-t. 2024. Refir: Grounding large restoration models with retrieval augmentation. Advances in Neural Information Processing Systems, 37: 46593–46621.

Jiang, Y.; Chan, K. C.; Wang, X.; Loy, C. C.; and Liu, Z. 2021. Robust reference-based super-resolution via C2-matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Jolicoeur-Martineau, A. 2018. The relativistic discriminator: a key element missing from standard GAN. arXiv preprint arXiv:1807.00734.

Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, 5148–5157.

Labs, B. F.; Batifol, S.; Blattmann, A.; Boesel, F.; Consul, S.; Diagne, C.; Dockhorn, T.; English, J.; English, Z.; Esser, P.; et al. 2025. FLUX. 1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv preprint arXiv:2506.15742.

Liang, J.; Cao, J.; Sun, G.; Zhang, K.; Van Gool, L.; and Timofte, R. 2021. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, 1833–1844.

Lin, S.; Xia, X.; Ren, Y.; Yang, C.; Xiao, X.; and Jiang, L. 2025. Difusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316.

Lu, L.; Li, W.; Tao, X.; Lu, J.; and Jia, J. 2021. Masa-sr: Matching acceleration and spatial adaptation for referencebased image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6368–6377.

Mittal, A.; et al. 2012. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3): 209– 212.

Peebles, W.; and Xie, S. 2023. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, 4195–4205.

Podell, D.; English, Z.; Lacey, K.; Blattmann, A.; Dockhorn, T.; Müller, J.; Penna, J.; and Rombach, R. 2024. Sdxl: Improving latent difusion models for high-resolution image synthesis. In International Conference on Learning Representations, volume 2024, 1862–1874.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Soh, J. W.; and Cho, N. I. 2022. Variational deep image restoration. IEEE Transactions on Image Processing, 31: 4363–4376.

Sun, L.; Wu, R.; Ma, Z.; Liu, S.; Yi, Q.; and Zhang, L. 2025. Pixel-level and semantic-level adjustable super-resolution: A dual-lora approach. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2333–2343.

Varanka, T.; Toivonen, T.; Tripathy, S.; Zhao, G.; and Acar, E. 2024. Pfstorer: Personalized face restoration and superresolution. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2372–2381.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

Wang, J.; Chan, K. C.; and Loy, C. C. 2023. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, volume 37, 2555–2563.

Wang, X.; Xie, L.; Dong, C.; and Shan, Y. 2021. Realesrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, 1905–1914.

Wang, Y.; Wan, Y.; Zheng, S.; Li, B.; Hou, Q.; and Jiang, P.-T. 2026. Trust but Verify: Adaptive Conditioning for Reference-Based Difusion Super-Resolution via Implicit Reference Correlation Modeling. arXiv preprint arXiv:2602.01864.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.

Wei, P.; Xie, Z.; Lu, H.; Zhan, Z.; Ye, Q.; Zuo, W.; and Lin, L. 2020. Component divide-and-conquer for real-world image super-resolution. In European conference on computer vision, 101–117. Springer.

Wu, C.; Li, J.; Zhou, J.; Lin, J.; Gao, K.; Yan, K.; Yin, S.-m.; Bai, S.; Xu, X.; Chen, Y.; et al. 2025. Qwen-image technical report. arXiv preprint arXiv:2508.02324.

Wu, R.; Sun, L.; Ma, Z.; and Zhang, L. 2024. One-step efective difusion network for real-world image super-resolution. Advances in Neural Information Processing Systems, 37: 92529–92553.

Wu, R.; Sun, L.; Zhang, Z.; Kong, X.; Zhao, J.; Wang, S.; and Zhang, L. 2026. VOSR: A Vision-Only Generative Model for Image Super-Resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16311–16321.

Yang, F.; Yang, H.; Fu, J.; Lu, H.; and Guo, B. 2020. Learning texture transformer network for image super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Yang, S.; Wu, T.; Shi, S.; Lao, S.; Gong, Y.; Cao, M.; Wang, J.; and Yang, Y. 2022. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 1191–1200.

Yao, S.; Liu, M.; Zhang, Z.; Wan, Z.; Ji, Z.; Bai, J.; and Zuo, W. 2025. MDIQA: Unified Image Quality Assessment for Multi-dimensional Evaluation and Restoration. arXiv preprint arXiv:2508.16887.

Yu, F.; Gu, J.; Li, Z.; Hu, J.; Kong, X.; Wang, X.; He, J.; Qiao, Y.; and Dong, C. 2024. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 25669–25680.

Zhang, K.; Zuo, W.; Gu, S.; and Zhang, L. 2017. Learning deep CNN denoiser prior for image restoration. In Proceedings of the IEEE conference on computer vision and pattern recognition, 3929–3938.

Zhang, L.; Li, X.; He, D.; Li, F.; Ding, E.; and Zhang, Z. 2023. LMR: A large-scale multi-reference dataset for referencebased super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 13118–13127.

Zhang, L.; Li, X.; He, D.; Li, F.; Wang, Y.; and Zhang, Z. 2022. Rrsr: Reciprocal reference-based image superresolution with progressive feature alignment and selection. In European conference on computer vision, 648–664. Springer.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018a. The unreasonable efectiveness of deep features as a perceptual metric. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Zhang, Y.; Li, K.; Li, K.; Wang, L.; Zhong, B.; and Fu, Y. 2018b. Image super-resolution using very deep residual channel attention networks. In Proceedings ofthe European conference on computer vision, 286–301.

Zhang, Y.; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.; Xie, P.; Yang, A.; Liu, D.; Lin, J.; et al. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

Zhang, Z.; Wang, Z.; Lin, Z.; and Qi, H. 2019. Image superresolution by neural texture transfer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 7982–7991.

Zhao, R.; Ma, J.; Cao, H. H.; Curless, B.; Seitz, S. M.; and Kemelmacher-Shlizerman, I. 2026. GarmentZoom: Generating Zoomable Images from Garment Listings. arXivpreprint arXiv:2606.29535.

# GraftSR: Grafting Authentic Textures for Real-World Image Super-Resolution via Identical-Instance Guidance

Supplementary Material

## Overview

In this supplementary material, we present:

• Method details, including the architecture details and the data construction procedure.

• Implementation details, including hyperparameter settings, prompts, and training strategies.

• User study details.

• More visualization results on each dataset.

## Method Details

In this section, we elaborate on the training details omitted from the main paper, including the one-step training formulation, the detailed training objectives, and the detail of our identical-instance dataset construction.

## One-Step Training Formulation

Dual-Noise Latent Construction. As described in the main paper, we adopt a dual-noise strategy (Fang et al. 2026) to balance perception and fidelity. Given the low-quality latent $z _ { L Q } \bar { = } \bar { \mathcal { E } _ { \mathrm { V A E } } } ( I _ { L Q } )$ , both the perception and control branches follow the flow-matching forward process:

$$
\begin{array} { c } { z = ( 1 - \sigma _ { p } ) z _ { L Q } + \sigma _ { p } \epsilon , \sigma _ { p } = \sigma ( s _ { p } ) , \nonumber } \\ { z _ { c } = ( 1 - \sigma _ { c } ) z _ { L Q } + \sigma _ { c } \epsilon , \sigma _ { c } = \sigma ( s _ { c } ) , } \end{array} \epsilon \sim { \mathcal N } ( 0 , { \bf I } ) ,\tag{1}
$$

where $\sigma _ { p }$ and $\sigma _ { c }$ are the noise levels at timesteps $s _ { p } { = } 7 5 0$ and $s _ { c } \sim \mathcal { U } ( 7 5 0 , 9 9 9 )$ . A larger sampled $s _ { c }$ yields weaker noise, allowing the control branch to preserve fine details.

One-Step Velocity Prediction. To convert the multi-step denoising into a single forward pass, we adopt one-step adversarial training (Lin et al. 2025) and skip the consistencydistillation stage, since our input is a noised low-quality latent rather than pure noise. The generator predicts the velocity field to update the prior-noise latent to the high-quality target in a single step:

$$
z _ { \mathrm { p r e d } } = z + \left( 0 - \sigma _ { p } \right) v _ { \theta } ( z , z _ { \mathrm { c } } , z _ { \mathrm { s e m } } , z _ { \mathrm { r e f } } ) ,\tag{2}
$$

which is decoded to $I _ { \mathrm { p r e d } } = D ( z _ { \mathrm { p r e d } } )$ for supervision. Note that when GPU memory is limited, high-resolution inputs can be processed in a tiled manner. In this case, in addition to the dual-noise encoding of each cropped patch, we encode the full image with the VAE and concatenate its latent as an additional global structural token into the control branch, so that each patch retains access to the global context. This extends the conditioning input of $v _ { \theta }$ with one extra token, while the velocity-prediction and update rule remain unchanged.

## Training Objectives

Generator Objective. The generator is trained with a reconstruction loss ${ \mathcal { L } } _ { \mathrm { r e c } }$ in RGB space and an adversarial loss $\mathcal { L } _ { \mathrm { a d v } } ^ { G }$ in latent space. Beyond the standard MSE and LPIPS terms, we introduce the DISTS loss (Ding et al. 2020). Unlike LPIPS, which strictly penalizes spatial shifts, DISTS tolerates mild misalignment, making it well suited to our cross-view texture restoration. It extracts multi-stage feature maps from a pre-trained VGG network and jointly measures structural and textural similarity:

$$
\mathcal { L } _ { \mathrm { { D I S T S } } } = 1 - \sum _ { i = 0 } ^ { n } \left( \alpha _ { i } l _ { s } ( \hat { F } _ { i } , F _ { i } ) + \beta _ { i } l _ { t } ( \hat { F } _ { i } , F _ { i } ) \right) ,\tag{3}
$$

where $\hat { F } _ { i }$ and $F _ { i }$ are the feature maps of $I _ { \mathrm { p r e d } }$ and $I _ { H Q }$ at the i-th VGG layer of a pre-trained VGG16 network(i = 0 denotes the input image and $n = 5$ corresponds to the five convolutional stages), $\bar { l } _ { s }$ and $l _ { t }$ measure structural and textural similarity, and $\alpha _ { i } , \beta _ { i }$ are the pre-trained weights from (Ding et al. 2020). This encourages texture fidelity without enforcing rigid pixel-to-pixel matching. The overall reconstruction loss is then formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } = \mathcal { L } _ { \mathrm { M S E } } ( I _ { \mathrm { p r e d } } , I _ { H Q } ) + \lambda _ { \mathrm { l p i p s } } \mathcal { L } _ { \mathrm { L P I P S } } + \lambda _ { \mathrm { d i s t s } } \mathcal { L } _ { \mathrm { D I S T S } } . } \end{array}\tag{4}
$$

For the adversarial term, we adopt a relativistic GAN loss (Jolicoeur-Martineau 2018) to avoid mode collapse:

$$
\mathcal { L } _ { \mathrm { a d v } } ^ { G } = - \mathbb { E } [ \log ( 1 - R ( z _ { H Q } , z _ { \mathrm { p r e d } } ) ) ] - \mathbb { E } [ \log ( R ( z _ { \mathrm { p r e d } } , z _ { H Q } ) ) ] ,\tag{5}
$$

where $z _ { H Q } = \mathcal { E } _ { \mathrm { V A E } } ( I _ { H Q } )$ is the clean latent encoded from the high-quality target image, and $R ( a , b ) = \mathcal { S } ( D ( a ) -$ $D ( b ) )$ measures the relativistic realism between two latents. Since the VAE encoder $\mathcal { E } _ { \mathrm { V A E } } ^ { ' }$ in the prior noise branch is trainable, we further introduce a VAE loss to prevent it from drifting away from the target distribution. Specifically, its encoded LQ latent $z _ { L Q }$ is decoded by the frozen VAE decoder and supervised against the HQ target:

$$
\mathcal { L } _ { \mathrm { v a e } } = \mathcal { L } _ { \mathrm { M S E } } ( D \left( \mathcal { E } _ { \mathrm { V A E } } ^ { \prime } ( I _ { L Q } ) \right) - I _ { H Q } )\tag{6}
$$

The overall generator objective is:

$$
\mathcal { L } _ { G } = \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } } ^ { G } + \lambda _ { \mathrm { v a e } } \mathcal { L } _ { \mathrm { v a e } }\tag{7}
$$

Discriminator Objective. Following the one-step SR paradigm (Fang et al. 2026), our discriminator is built upon the Wan2.1 1.3B (Wan et al. 2025) architecture, which shares the same VAE encoder as the Qwen-Image backbone and injects textual conditioning through cross-attention. Instead of introducing extra cross-attention layers to regress a single scalar as in as in Adversarial Post-Training (APT) (Lin et al.

2025), we simply append two 2D convolutional layers to the transformer outputs, yielding patch-wise realism scores. The discriminator is initialized from pre-trained weights and is optimized with an adversarial loss $\mathcal { L } _ { \mathrm { a d v } } ^ { D }$ that is symmetric to the generator counterpart:

$$
\mathcal { L } _ { \mathrm { a d v } } ^ { D } = - \mathbb { E } [ \log ( R ( z _ { H Q } , z _ { \mathrm { p r e d } } ) ) ] - \mathbb { E } [ \log ( 1 - R ( z _ { \mathrm { p r e d } } , z _ { H Q } ) ) ] .\tag{8}
$$

To stabilize training, we further impose an approximated R1 regularization that suppresses the discriminator gradients on real samples:

$$
\mathcal { L } _ { R } = \left\| D ( z _ { H Q } ) - D ( z _ { H Q } + \delta ) \right\| _ { 2 } ^ { 2 } , \quad \delta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } { \bf I } ) ,\tag{9}
$$

where δ is a small Gaussian perturbation added to the real latent $z _ { H Q }$ , encouraging consistent predictions between the clean and perturbed inputs. The overall discriminator objective is:

$$
\begin{array} { r } { \mathcal { L } _ { D } = \mathcal { L } _ { \mathrm { a d v } } ^ { D } + \lambda _ { R } \mathcal { L } _ { R } . } \end{array}\tag{10}
$$

## Dataset Construction Details

We supplement two concrete details of our curation pipeline omitted from the main paper: the mask dilation scheme and the pair quality assessment rules.

Mask Dilation. To avoid artifacts around delicate peripheral structures $( e . g .$ ., lace, tassels, decorative edges), we dilate the complementary masks with an elliptical kernel followed by a light Gaussian smoothing. During training, the dilation radius is randomly sampled per mask as a boundarypreserving augmentation, using 3–6 pixels for the target mask and a larger 5–10 pixels for the reference mask to tolerate cross-view misalignment; at inference, a fixed radius is adopted for deterministic behavior.

Pair Quality Assessment. As mentioned in the main paper, an initial filtering stage strictly assesses candidate images before masking. We detail its three criteria here: (1) we compute the semantic embedding similarity (Zhang et al. 2025) between $I _ { R E F }$ and $I _ { H Q }$ and discard any tuple below a strict threshold of 0.85, which eliminates mismatched pairs such as a garment matched with an image showing only textbased specifications; (2) we adopt the MDIQA (Yao et al. 2025) quality predictor to eliminate degraded samples with severe motion blur or compression artifacts, avoiding unreliable texture supervision; and (3) we apply heuristic rules to reject extreme close-ups (e.g., macro shots of fabric) that lose recognizable instance identity and lack suficient structural context.

## Implementation Details

In this section, we provide the full implementation details of GraftSR to facilitate reproducibility. We first present the complete hyperparameter settings for both the generator and the discriminator. We then describe the training strategies adopted to enhance robustness and generalization. Finally, we detail the prompts used for VLM throughout the pipeline.

## Hyperparameter Settings

We build the generator based on the Qwen-Image-Edit (Wu et al. 2025) (2511 version) backbone and fine-tune it with LoRA, while the Wan2.1-1.3B discriminator is fully finetuned from its pre-trained weights. The rank of LoRA is 128. Both networks are optimized using RMSprop $( \alpha { = } 0 . 9 $ momentum=0) with constant learning rates of $5 \times 1 0 ^ { - 5 }$ for the generator and $5 \times 1 0 ^ { - 6 }$ for the discriminator. We train for one epoch with a total batch size of 32 on 32 NVIDIA H20 GPUs, taking roughly 30 hours, with gradient checkpointing enabled and the discriminator updated once per generator step. Training is conducted on $\bar { 5 } 1 2 \times 5 1 2$ patches. The perception and control branches inject noise starting from timesteps $s _ { p } { = } 7 5 0$ and $s _ { c } { \sim } \mathcal { U } ( 7 5 0 , 9 9 9 )$ , respectively. For the loss weights, we set the LPIPS, DISTS, and VAE weights to $\lambda _ { \mathrm { l p i p s } } { = } \lambda _ { \mathrm { d i s t s } } { = } \lambda _ { \mathrm { v a e } } { = } 1 . 0$ with the MSE term weighted by 1.0 by default, and the adversarial loss weight to $\lambda _ { \mathrm { a d v } } { = } 0 . 0 2 \nonumber$ . The R1 regularization weight is set to $\lambda _ { R } { = } 5 . 0$ with a perturbation variance $\sigma ^ { 2 } { = } 0 . 0 0 5$

## Training Strategies

To enhance robustness and generalization, we adopt several stochastic training strategies. To construct training patches, we randomly crop a $5 1 2 \times 5 1 2$ region, where with a probability of 0.5 the crop is centered at a point sampled within the mask region to ensure the target instance is well covered, and otherwise a random region is selected. To prevent the model from over-relying on the auxiliary conditions, we randomly drop the entire reference branch and the spatial mask each with a probability of 0.1, encouraging the network to remain efective even when the reference or mask is unavailable. In addition, with a probability of 0.5 the control branch directly receives the clean low-quality latent without noise injection, which further balances detail preservation and restoration capability.

## VLM Prompts

Annotation Prompt. To jointly obtain the unified caption T and the complementary masks, we prompt the VLM with the reference and target images that depict the same instance, asking it to identify the shared entity, describe its visual attributes, and localize the matched semantic region in both views for cross-view alignment:

Task. You will receive two images depicting the same product and its associated outfit: a scene/model image (SRC) and a high-quality reference image (REF). Localize the corresponding region of the same product in both images and return the matched semantic bounding boxes for afine alignment.

Key requirements. (1) Semantic consistency: the two boxes must enclose the same visible part of the same product; if SRC shows only the upper part, the REF box should cover only the corresponding upper part rather than the whole product. (2) Completeness: under semantic consistency, each box should cover the product’s visible region as completely as possible.

Output. product\_entity (the plain object noun, e.g., dress, T-shirt); product\_description (visual attributes such as color, material, texture, and style); src\_image\_description (full visual content of SRC); match\_description; and src\_matched\_bbox / ref\_matched\_bbox (the matched region in each image).

Evaluation Prompt. For the VLM-based evaluation on real-world SR, where no ground truth is available, we adopt a pointwise scoring protocol. The prediction is presented as a global view together with a full-resolution crop of the key region, and is graded on its intrinsic quality and its consistency with the reference crop. The VLM is instructed by the following system and task prompts, and rates each dimension on an integer scale of 1–5:

System. You are a rigorous, objective image-quality evaluator for reference-based super-resolution. For each image you get a global view and a full-resolution crop of the key region: use the global view for content/composition and the crop for fine detail/texture. Return only a valid JSON object. Each score is an integer 1–5 (1=worst, 5=best), and the full range should be used.

Task. Evaluate a real-world image restoration result. The inputs, each consisting of a global view and a key-region crop, are: (i) PRED, the model output to evaluate; and (ii) REF, the reference showing the true textures of the key item. Grade PRED on its intrinsic quality and its consistency with REF along the dimensions below, and return the scores in the specified JSON format.

Evaluation dimensions (the key goal is restoring correct, faithful texture of the key item):

(1) Perceptual quality: overall visual quality — sharpness, natural realistic details, correct tone/color, and absence of degradation. Penalize blur, insuficient enhancement, noise, artifacts, over-smoothing, over-sharpening halos, over-saturation, and color shift. Note that real highresolution images naturally contain fine high-frequency textures; a visibly sharper yet faithful result should score higher, and only false details (halos, ringing, hallucinated textures) are penalized.

(2) Texture consistency: whether the key item’s textures/- patterns match the REF, and whether they blend seamlessly in lighting, perspective, and style. Penalize destroyed, fabricated, or mismatched textures.

## User Study Details

We conduct a two-alternative forced-choice (2AFC) study with 10 participants over 50 diverse scenes. In each trial, the low-quality input and the reference image are presented alongside two anonymized candidates, namely our result and that ofa competing method, whose left and right positions are randomly shufled across trials to eliminate positional bias. Participants are forced to select the better result according to perceptual quality and texture consistency with the reference.

To ensure consistent judgments, participants follow a set of evaluation criteria. First, regarding perceptual quality, each candidate is compared against the low-quality input, and a candidate is assigned a lower preference ifit remains blurry or fails to recover basic details. Second, regarding texture consistency, each candidate is compared against the reference, focusing on whether the large-scale material and texture details within the product region are consistent, while highly localized regions are ignored. In addition, candidates are assigned a lower preference if their textures are inconsistent with the reference or their appearances are over-smoothed and deviate from the authentic material. Overall, sharpness is prioritized, such that a candidate without efective quality enhancement is not preferred, and among candidates with comparable sharpness the one with better texture preservation is selected. Furthermore, we exclude the option of whether invalid, such as those with an unusable reference or corrupted images, prior to aggregation. The user study interface is shown in Fig. 1.

![](images/f14de0185d99d6ec2739dff591d83c335e15ef9a5af29d0d989c107b9a4f6e08.jpg)  
Figure 1: User Study Interface.

## More Visualization Results

We provide additional qualitative comparisons to further demonstrate the efectiveness of GraftSR. Figure 2 and Figure 3 present more results on the synthetic and real-world splits of our TexRefSR-Eval benchmark, respectively, where reference guidance is available. Figure 4 further reports results on general image super-resolution benchmarks. Across all settings, our method consistently restores sharper structures and more faithful textures than competing approaches, while avoiding the over-smoothing and hallucinated details commonly observed in other methods.

![](images/1e025643e662cb96a54cec3f0f8e8ed52786c55e84d77fe1a5fa0d1b9423356a.jpg)  
Figure 2: More qualitative comparisons on the TexRefSR-Eval-Syn dataset, where low-quality inputs are produced by the synthetic degradation model. Guided by the reference, our method restores textures that are more faithful to the ground truth, recovering fine patterns and structural details that other methods either over-smooth or hallucinate incorrectly.

![](images/06ee8e88a669a619655e2378e211c1d746e511938a1cb7fb55f2b4b921850f91.jpg)  
Figure 3: More qualitative comparisons on the TexRefSR-Eval-Real dataset, which contains real-world low-quality images without ground truth. Our method produces perceptually sharper results whose textures remain consistent with the reference, whereas other methods tend to yield blurry or unfaithful reconstructions.

![](images/fac54f151379dbd268ba92d7a91e391df72040b7fc0046e487c6b5e6872cfd4d.jpg)  
(a) Zoomed Input  
(b) Zoomed GT  
(c) OSEDiff  
(d) PiSA-SR  
(e) ODTSR  
(f) Ours  
Figure 4: More qualitative comparisons on general real-world SR benchmarks (RealSR and DRealSR). Although no reference is available in this setting, our method still generalizes well to natural scenes and recovers clean, realistic details that are competitive with or superior to state-of-the-art general SR methods.