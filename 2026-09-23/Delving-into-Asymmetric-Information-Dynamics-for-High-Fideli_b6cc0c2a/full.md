# Delving into Asymmetric Information Dynamics for High-Fidelity Virtual Try-On

Zishu Qin<sup>1</sup>, Zhiyu Jin<sup>1</sup>, Pipei Huang<sup>1</sup>, Hao Zhou<sup>1,⊠</sup>

<sup>1</sup>Alibaba Group

<sup>⊠</sup>Corresponding author

## Abstract

Virtual try-on (VTON) requires precise pixel-level fidelity, yet mainstream Difusion Transformers (DiTs) often sufer from texture degradation and structural drift. We identify symmetric interactions in standard joint-attention mechanisms as a source of these failures. Although such interactions support semantic flexibility in general-purpose editing, they allow stochastic noise to corrupt deterministic garment features in VTON. We analyze this problem through asymmetric information dynamics and introduce two diagnostic indicators: Conditional Attention Entropy (CAE) for feature unbiasedness and Injected Information Flux (IIF) for injection efectiveness. Our analysis suggests that symmetric bidirectional attention can corrupt conditional features and attenuate the conditional signal. To address these limitations, we propose RealFit, a framework that combines Unidirectional Information Flow (UIF) with Decoupled Timestep Modulation (DTM). UIF isolates the garment condition from stochastic noise to preserve garment identity, while DTM optimizes the modulation scale to maintain a strong conditional signal. The resulting time-invariant condition branch enables a conditional KV cache that reduces inference time by approximately 75%. RealFit ofers a principled approach to conditional generation with state-of-the-art fidelity and eficiency.

## 1 Introduction

Image-based virtual try-on (VTON) is an important task in digital fashion and e-commerce. Unlike general image editing, VTON [10] requires the textures, patterns, silhouettes, and design identity of a garment to be accurately preserved when it is transferred onto a reference person. Despite the strong generative capabilities of large-scale Difusion Transformers (DiTs) [9, 14, 16, 17, 39], achieving consistently high fidelity remains challenging. Existing models can introduce artifacts or degrade garment features, causing the generated garment to difer from the reference in color, texture, or fine detail. We argue that these limitations are partly architectural: many generative frameworks are designed for general-purpose editing and are not well suited to the strict identitypreservation requirements of VTON.

General-purpose models typically prioritize flexibility and global semantic alignment, whereas VTON requires fidelity to a specific reference garment. We argue that a key bottleneck in DiT-based VTON methods is the symmetric interaction pattern used in standard joint attention. In models such as FLUX.1 Kontext [20], noise and condition tokens are treated similarly and processed through bidirectional attention between the two streams. However, VTON is an asymmetric information task: the noise latent is stochastic and requires guidance from the condition, while the garment condition is deterministic and should remain largely unafected by noise. To analyze this interaction, we introduce two informationtheoretic indicators: (1) Conditional Attention Entropy (CAE), which measures feature unbiasedness by quantifying semantic dispersion and potential degradation in conditional features, and (2) Injected Information Flux (IIF), which measures injection efectiveness by quantifying the cumulative signal transferred from the condition to the noise latent.

![](images/31f7a01a5cbbda7503e22c9a115db280ae0e8a95e7e0b9fb7fd27963db956318.jpg)

![](images/c772ebd46a3563bafa4fe589462b35bf4b167e1cd2aa20a481d196168ac963f8.jpg)  
Figure 1 RealFit achieves high-fidelity virtual try-on. (Left) RealFit preserves garment identity across categories, while prior DiTs exhibit errors in silhouette, detail, color, and texture. (Right) Fidelity correlates strongly with CAE and IIF. RealFit reduces CAE and increases IIF to overcome fidelity bottlenecks in symmetric architectures.

The empirical results in Figure 1 suggest that standard bidirectional DiTs face an information bottleneck characterized by high CAE and low IIF. This bottleneck involves two attention paths. (1) The Condition-on-Noise (C-on-N) path allows deterministic garment tokens to attend to the stochastic noise latent. This reverse information flow increases CAE and corrupts the unbiased garment representation, causing conditional features to drift from the reference during denoising. (2) The Noise-on-Condition (N-on-C) path transfers garment information from the condition to the noise latent and directly determines IIF. In practice, timestep-dependent modulation designed for noisy latents can attenuate the conditional signal during critical stages of denoising. The resulting low IIF provides insuficient guidance for faithful reconstruction of the target garment.

Building on these insights, we propose RealFit, a high-fidelity VTON framework that addresses this bottleneck through two complementary mechanisms. (1) Unidirectional Information Flow (UIF): We block the C-on-N path while allowing noise tokens to attend to the garment condition. The condition attends only to itself and remains isolated from the stochastic noise latent. This design minimizes CAE, preserves high-frequency details, and maintains an unbiased garment representation throughout denoising. (2) Decoupled Timestep Modulation (DTM): We decouple the modulation of the noise and condition branches to maximize IIF. Rather than following a dynamic schedule, the condition branch uses a fixed, optimal modulation state that maintains a strong conditional signal. This allows deterministic garment features to guide the entire denoising trajectory, improving visual fidelity.

Together, UIF and DTM make the condition branch independent of both the denoising timestep and the noise latent, so its features remain time-invariant. We use this property to implement a conditional KV cache: garment features are computed once and reused throughout denoising, avoiding redundant computation in subsequent iterations. This design accelerates inference and reduces GPU memory usage. Our contributions are as follows:

• We formulate VTON as an asymmetric information task based on the distinction between stochastic noise latents and deterministic conditional signals. This perspective motivates a diagnostic framework based on CAE and IIF for quantifying feature unbiasedness and injection efectiveness in generative models.

• We propose RealFit, which combines UIF and DTM to address the limitations of symmetric DiT architectures. By protecting the deterministic condition from stochastic corruption and maximizing injection efectiveness, RealFit overcomes these structural bottlenecks while preserving garment identity.

• Quantitative benchmarks and extensive qualitative evaluations show that RealFit achieves state-of-the-art performance. Its conditional KV cache reduces inference time by approximately 75%, making high-fidelity generation more practical for deployment.

## 2 Related Work

## 2.1 General-Purpose Image Editing

General-purpose image editing has advanced substantially with the introduction of Difusion Transformers (DiTs) [25]. DiT-based frameworks such as FLUX.1 Kontext [20] and OmniGen [32] use symmetric joint attention for in-context editing and instruction following without auxiliary conditioning networks. UniVG [7] and Lumina-Image 2.0 [26] process multimodal signals as joint sequences, supporting diverse tasks with a single set of model weights. The Show-o series [33, 34] unifies autoregressive modeling and discrete difusion to handle interleaved modalities. ICEdit [38] improves instruction adherence through in-context generation with minimal parameter-eficient fine-tuning. Training-free methods such as Stable Flow [1] and SpotEdit [27] enable localized editing by identifying critical layers or skipping computation in unchanged regions.

However, the symmetric bidirectional interactions used by these architectures are poorly suited to VTON. The garment condition must remain deterministic and retain its details, yet standard DiTs allow stochastic noise to corrupt conditional features and blur fine textures. Building on FLUX.1 Kontext, we introduce unidirectional information flow and decoupled modulation to preserve conditional integrity and maintain an unbiased garment representation.

## 2.2 Virtual Try-On

Difusion models have become a dominant approach to generative modeling, with strong performance in high-fidelity image synthesis [3, 30], structured image editing [11, 22], and spatially conditioned generation [36]. Image-based VTON aims to generate an image of a person wearing a target garment while preserving both the person’s identity and garment fidelity. Early methods typically relied on explicit geometric warping and extensive preprocessing. More recent methods build on large pretrained difusion models and incorporate garment and person conditions through attention, improving photorealism and robustness. LADI-VTON [24] uses textual inversion for guidance, while OOTDifusion [35] and IDM-VTON [5] use pretrained latent difusion backbones to align and inject garment details through outfitting fusion or auxiliary modules, avoiding explicit warping. CatVTON [6] simplifies the architecture by spatially concatenating inputs within adapted selfattention modules, achieving competitive eficiency without redundant encoders. ITA-MDT [13] uses image-timestep adaptive feature aggregation to capture fine details in salient garment regions. Other methods emphasize scalability and versatility: Any-Fit [21] introduces parallel attention blocks for compositional clothing, and Any2AnyTryon [9] reduces reliance on auxiliary signals through adaptive positional embeddings. StyleVTON [14] supports crossdomain story visualization, while FitDiT [15] allocates more capacity to high-resolution features using frequency-domain objectives. CORAL [17] improves person-garment correspondence in unpaired settings through attention entropy minimization.

RealFit uses attention entropy to characterize the unbiasedness of deterministic conditional features. Our focus is on protecting reference features from stochastic interference and preserving their information throughout denoising.

## 3 Methodology

## 3.1 Problem Formulation

RealFit performs high-fidelity VTON using a garment image $x _ { g }$ and a reference person image $x _ { r } .$ . We generate a try-on image $x _ { p }$ in which the person wears the garment in $x _ { g }$ , while preserving the person’s identity, pose, body shape, and surrounding scene.

RealFit is built on a FLUX-style DiT architecture [19]. A standard FLUX-style DiT processes the noise latent $\mathbf { x } _ { t }$ and the visual condition c derived from $x _ { g }$ and $x _ { r }$ symmetrically, using bidirectional attention and shared timestep-dependent modulation. FLUX contains both double-stream and single-stream blocks, which difer primarily in how the visual and text modalities interact. Our design targets visual features in both block types.

![](images/033b313ea43667a426fd4d415585ed1b1284dbeb2c0739b00fd46c05c66d86c4.jpg)  
Figure 2 Overview of RealFit. High CAE in attention and low IIF in modulation can create an information bottleneck. RealFit addresses this bottleneck through an asymmetric design that combines UIF and DTM for high-fidelity VTON.

## 3.2 Bottlenecks in Symmetric Interaction

Standard DiT-based generation and editing models use symmetric interactions between noise and condition tokens. Treating these tokens equivalently supports flexible editing but conflicts with the strict fidelity requirements of VTON. The two streams represent distinct information states: the noise latent $\mathbf { x } _ { t }$ is stochastic and requires structural guidance, whereas the visual condition c is a deterministic reference that encodes pixel-level identity. Symmetric attention allows noise to influence the condition, forcing conditional features to adapt to intermediate noisy states and potentially corrupting them. We formalize this interaction as follows.

Let $\mathbf { N } \in \mathbb { R } ^ { L _ { n } \times d }$ and $\mathbf { C } \in \mathbb { R } ^ { L _ { c } \times d }$ denote the noise and condition token sequences, respectively. The symmetric joint-attention matrix A is given by

$$
\mathbf { A } = \mathrm { S o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } \right) = \left[ \mathbf { A } _ { N \mathrm { - o n } - N } \quad \mathbf { A } _ { N \mathrm { - o n } - C } \right] ,\tag{1}
$$

where $\mathbf { A } _ { I - \mathrm { o n } - J }$ denotes the attention weights between queries from I and keys from J. This decomposition reveals two aspects of the information bottleneck.

Conditional Attention Entropy The $\mathbf { A } _ { C - \mathrm { o n } - N }$ path allows stochastic noise to corrupt the deterministic condition. For a condition query i, the updated feature is a weighted sum of condition and noise values:

$$
\mathbf { c } _ { i } ^ { \prime } = \sum _ { j \in \mathrm { C o n d i t i o n } } \mathbf { A } _ { i , j } \mathbf { v } _ { j } + \sum _ { k \in \mathrm { N o i s e } } \mathbf { A } _ { i , k } \mathbf { v } _ { k } ,\tag{2}
$$

where the second term introduces variability from the stochastic latent. For high-fidelity VTON, we argue that the condition stream should remain unbiased: each condition token should concentrate its attention on a sparse set of deterministic keys to preserve highfrequency details.

We define Conditional Attention Entropy (CAE) as the sum of Shannon entropies over all condition-query attention distributions. CAE quantifies how concentrated these distributions are and characterizes the preservation of deterministic conditional features during joint attention:

$$
\mathrm { C A E } = \sum _ { i = L _ { n } + 1 } ^ { L _ { n } + L _ { c } } \left( - \sum _ { j = 1 } ^ { L _ { n } + L _ { c } } \mathbf { A } _ { i , j } \ln \mathbf { A } _ { i , j } \right) .\tag{3}
$$

Higher entropy indicates a more difuse distribution. In Equation 2, high CAE indicates dispersed attention, which can dilute garment identity when condition queries attend to stochastic noise rather than deterministic garment features. Decoupling the information pathways and blocking the C-on-N path prevents this reverse influence. Minimizing CAE supports conditional integrity and preserves detailed, unbiased garment features throughout denoising.

Injected Information Flux In addition to protecting the condition, high-fidelity VTON requires noise tokens to receive the garment’s deterministic details efectively. For a noise query $i \in \{ 1 , \ldots , L _ { n } \}$

![](images/7b425885fe165839b63350d48ae97097e8452d9a272a00e0f835ba40e87c993b.jpg)  
(a) Visual comparisons on public benchmarks, RealFit displays superior details.

RealFiT

![](images/2d31a871e8b5fd7b9222f06017651fdc0fbcb8aedd54c717914b18b99da8a902.jpg)  
(b) A Wide Range of High-Fidelity Fashion Injection Tasks.

Figure 3 Qualitative results. (a) RealFit protects deterministic garment features, achieving higher fidelity and sharpe textures than the baselines. (b) Generalization across fashion items demonstrates broad applicability.

a symmetric layer updates its feature $\mathbf { n } _ { i } ^ { \prime }$ as follows:

$$
\mathbf { n } _ { i } ^ { \prime } = \underbrace { \sum _ { j = 1 } ^ { L _ { n } } \mathbf { A } _ { i , j } \mathbf { v } _ { j } } _ { \mathrm { S e l f - r e f n i n g } } + \underbrace { \sum _ { k = L _ { n } + 1 } ^ { L _ { n } + L _ { c } } \mathbf { A } _ { i , k } \mathbf { v } _ { k } } _ { \mathrm { I n j e c t e d ~ S i g n a l } } ,\tag{4}
$$

where the second term injects garment information into the noise stream. To quantify this transfer, we define Injected Information Flux (IIF) as the sum of attention weights $\mathbf { A } _ { i , k }$ over the localized garment region. IIF measures how efectively the conditional signal guides the noise latent during denoising.

In standard denoising DiTs, query and key magnitudes are modulated by a shared timestep-dependent scale $\gamma ( t )$ , for example through AdaLN. Because $\mathbf { A } _ { i , k } \propto \exp ( \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { k } / \sqrt { d } )$ , IIF is sensitive to the norm of the condition keys $\| \mathbf { k } _ { k } \|$ , which is scaled by γ(t). Under common modulation schedules, γ(t) is relatively small early in denoising (large t) and increases as t → 0. This time-dependent suppression may be unnecessary for a deterministic garment condition, whose information content remains relatively stable. It can weaken conditional guidance precisely when the latent is establishing the garment’s global structure, potentially causing structural hallucinations or identity drift. We therefore decouple condition modulation from noise modulation and fix the condition branch to the maximal-scale state (t = 0). This encourages consistently stronger IIF and helps preserve structural integrity from the earliest denoising steps.

## 3.3 Asymmetric Information for VTON

RealFit (Figure 2) addresses these bottlenecks with unidirectional attention and decoupled modulation.

Unidirectional Information Flow To reduce CAE and preserve feature unbiasedness, we replace symmetric interactions with unidirectional attention. A structured attention mask retains the N-on-N, Non-C, and C-on-C paths while blocking the C-on-N path. This confines condition-token interactions to the deterministic condition stream, shielding it from stochastic noise. Unidirectional Information Flow (UIF) thus helps preserve the reference garment’s identity throughout denoising.

Decoupled Timestep Modulation To increase IIF and strengthen conditional injection, we modulate the noise and condition branches separately. The noise branch follows the current timestep t to accommodate changing noise levels. The condition branch uses a fixed timestep t<sup>⋆</sup>, which yields a relatively large, stable modulation scale throughout denoising. Keeping noise modulation dynamic and condition modulation fixed helps maintain the magnitude of condition keys and limits signal attenuation. DTM thus strengthens N-on-C attention and increases IIF to preserve fine garment details.

Asymmetric Text Coupling To maintain both instruction adherence and visual fidelity, we extend the asymmetric design to the text stream. The noise branch interacts with a text representation conditioned on the current timestep to support semantic planning throughout denoising. The condition branch instead uses a text representation conditioned on a fixed timestep, providing stable identity constraints from the garment description. This improves crossmodal consistency, allowing text and visual conditions to jointly guide generation.

Table 1 Results on VITON-HD and DressCode. Best and second-best results are in bold and underlined, respectively.
<table><tr><td rowspan="3">Method</td><td colspan="6">VITON-HD</td><td colspan="6">DressCode</td></tr><tr><td colspan="4">Paired</td><td colspan="2">Unpaired</td><td colspan="3">Paired</td><td colspan="2">Unpaired</td></tr><tr><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID ↓</td><td>KID↓</td><td>FID ↓</td><td>KID↓</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID ↓</td><td>KID ↓</td><td>FID ↓</td><td>KID ↓</td></tr><tr><td>LADI-VTON</td><td>0.875</td><td>0.091</td><td>9.42</td><td>1.63</td><td>14.64</td><td>8.75</td><td>0.902</td><td>0.052</td><td>6.94</td><td>2.33</td><td>10.67</td><td>5.78</td></tr><tr><td>IDM-VTON</td><td>0.881</td><td>0.078</td><td>9.12</td><td>1.03</td><td>9.84</td><td>1.12</td><td>0.923</td><td>0.046</td><td>5.32</td><td>1.24</td><td>9.54</td><td>4.32</td></tr><tr><td>AnyFit</td><td>0.893</td><td>0.075</td><td>8.60</td><td>0.55</td><td>8.06</td><td>1.37</td><td>0.904</td><td>0.044</td><td>4.51</td><td>0.48</td><td>7.33</td><td>2.59</td></tr><tr><td>OOTDiffusion</td><td>0.878</td><td>0.071</td><td>8.81</td><td>0.82</td><td>12.40</td><td>4.68</td><td>0.885</td><td>0.053</td><td>4.61</td><td>0.95</td><td>12.56</td><td>6.62</td></tr><tr><td>CatVTON</td><td>0.870</td><td>0.056</td><td>5.42</td><td>0.41</td><td>9.01</td><td>1.09</td><td>0.892</td><td>0.045</td><td>3.99</td><td>0.81</td><td>6.13</td><td>1.40</td></tr><tr><td>ITA-MDT</td><td>0.885</td><td>0.083</td><td>5.46</td><td>0.63</td><td>8.67</td><td>1.21</td><td>0.922</td><td>0.055</td><td>6.16</td><td>0.94</td><td>11.02</td><td>1.75</td></tr><tr><td>FitDiT</td><td>0.898</td><td>0.066</td><td>4.73</td><td>0.19</td><td>8.20</td><td>0.34</td><td>0.925</td><td>0.043</td><td>2.63</td><td>0.49</td><td>4.73</td><td>0.90</td></tr><tr><td>RealFit (Ours)</td><td>0.907</td><td>0.049</td><td>4.29</td><td>0.15</td><td>7.74</td><td>0.25</td><td>0.933</td><td>0.034</td><td>2.31</td><td>0.41</td><td>3.96</td><td>0.75</td></tr></table>

Time-Invariant KV Cache The asymmetric design makes the condition branch independent of both the noise latent and the denoising timestep t, so its features remain constant during inference. We therefore compute the condition keys $\left( \mathbf { K } _ { c } \right)$ and values (V<sub>c</sub>) once at the initial step and cache them for subsequent iterations. This conditional KV cache eliminates redundant computation and reduces endto-end inference time by over 75% while preserving generation quality.

## 4 Experiments

## 4.1 Experimental Settings

Datasets We evaluate RealFit on two widely used public benchmarks, VITON-HD [4] and Dress-Code [23], as well as an additional fashion dataset collected to evaluate generalization across diverse fashion items. VITON-HD contains 11,647 training and 2,032 test samples of women’s upper-body garments. DressCode contains 48,392 training and 5,400 test samples spanning upper-body garments, lower-body garments, and dresses. To address the lack of reference person images in these benchmarks, we use open-source editing models to augment the original garment–ground-truth pairs into garment– reference person–ground-truth triplets. The reference and ground-truth images depict the same person in diferent clothing. We will release the augmented dataset to support further VTON research.

Implementation Details We train RealFit for 40 epochs using the AdamW optimizer [18] with a learning rate of $3 \times 1 0 ^ { - 5 }$ . Training uses DeepSpeed ZeRO-2 [28, 29] on 8 NVIDIA H200 GPUs with a total batch size of 48. At inference time, we use 25 sampling steps on a single NVIDIA RTX 4090 GPU.

Evaluation Metrics Following prior work, we evaluate generation quality in both paired and unpaired settings. In the paired setting, we use SSIM [31], LPIPS [37], FID [12], and KID [2]. In the unpaired setting, we use FID and KID to assess overall fidelity and diversity. To evaluate garment-level perceptual fidelity, we use Gemini 3.1 Pro Preview to score four attributes: detail, texture, silhouette, and color. We also report sample-averaged CAE and IIF in selected experiments, unless stated otherwise.

## 4.2 Qualitative Results

We qualitatively compare RealFit with representative methods, focusing on garment fidelity and robustness across try-on scenarios.

Figure 3(a) compares RealFit with representative baselines, including OOTDifusion [35], AnyFit [21], CatVTON [6], and FitDiT [15]. Existing symmetric methods often preserve the overall garment category but blur textures or alter garment identity when handling complex patterns and fine details. These artifacts are primarily associated with information corruption and signal attenuation in symmetric interactions. RealFit produces sharper textures and preserves garment identity more faithfully. UIF protects deterministic garment features, while DTM strengthens information injection, allowing the model to retain reference details and naturally adapt the garment to the person’s body and pose.

To assess the versatility of the asymmetric framework, we also evaluate RealFit on challenging fashion categories using collected in-the-wild images. As shown in Figure 3(b), RealFit produces consistent, realistic results for diverse items, including hats, bags, shoes, and full suits. It preserves their distinctive material textures and fine structural details. These results support our central premise: preserving the condition as a deterministic signal and injecting its information efectively are important for high-fidelity conditional generation. They highlight RealFit’s potential in vir-

Table 2 Ablation study of RealFit’s components.
<table><tr><td rowspan="2">Method</td><td colspan="2">Generation</td><td colspan="2">Information</td></tr><tr><td>SSIM↑</td><td>LPIPS ↓</td><td>CAE↓</td><td>IIF ↑</td></tr><tr><td>Baseline</td><td>0.854</td><td>0.156</td><td>8.0</td><td>438.7</td></tr><tr><td>+ Decouple</td><td>0.857</td><td>0.150</td><td>7.9</td><td>442.6</td></tr><tr><td>+ UIF</td><td>0.867</td><td>0.068</td><td>3.9</td><td>440.1</td></tr><tr><td>+ DTM</td><td>0.907</td><td>0.049</td><td>3.6</td><td>696.5</td></tr></table>

Table 3 Efects of UIF and DTM across backbones.
<table><tr><td>Backbone</td><td>Config</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>KID↓</td></tr><tr><td>SDXL</td><td>Baseline</td><td>0.826</td><td>0.491</td><td>7.75</td><td>2.33</td></tr><tr><td rowspan="2"></td><td>+ UIF &amp; DTM</td><td>0.896</td><td>0.078</td><td>4.87</td><td>0.31</td></tr><tr><td>Baseline</td><td>0.843</td><td>0.231</td><td>7.46</td><td>1.63</td></tr><tr><td>SD3-DiT</td><td>+ UIF &amp; DTM</td><td>0.901</td><td>0.057</td><td>4.52</td><td>0.28</td></tr></table>

tual fashion.

## 4.3 Quantitative Results

Table 1 compares RealFit with the baselines on VITON-HD and DressCode. RealFit outperforms the compared methods on all reported metrics, achieving state-of-the-art results.

RealFit achieves the highest SSIM scores, indicating better preservation of garment structure. Its lower LPIPS scores also reflect improved perceptual similarity and finer texture preservation. These results support our objective of high-fidelity information injection. RealFit additionally obtains the lowest FID and KID scores, indicating improved photorealism and closer alignment between the generated and real image distributions. This suggests that the model produces plausible try-on images while maintaining diversity. The gains in the unpaired setting further demonstrate robust generalization.

## 4.4 Ablation Studies

We conduct ablation studies on VITON-HD to examine the contribution of each component. As shown in Table 2, the components provide incremental, complementary gains. Row 1 reports the standard symmetric DiT baseline. Row 2 decouples the QKV projections of the noise and condition branches, yielding a modest improvement. This may reflect the distributional diference between noise latents and garment features: separate projections allow the model to encode each stream more efectively. Adding UIF in Row 3 substantially reduces both LPIPS and CAE, indicating that unidirectional attention limits semantic corruption and feature dispersion while preserving conditional unbiasedness. Adding DTM in Row 4 further improves SSIM and IIF, demonstrating that branchspecific modulation strengthens the conditional signal and improves injection efectiveness.

![](images/52fa5a295eb78df4f9b16894275bbc8da75cef724f3247ec1360d3fa2619e300.jpg)

Figure 4 Spearman correlations between CAE/IIF and perceptual fidelity.  
![](images/03022dd1f652b2aef03f57e231b8c0ccf9d010f9bbb2d90589a766d20deef8c7.jpg)  
Figure 5 CAE/IIF trends with perceptual fidelity.

Table 3 evaluates the proposed design with diferent backbones. Applying UIF and DTM yields substantial improvements on both SDXL and SD3-DiT, supporting the efectiveness of asymmetric information injection across architectures.

## 4.5 Perceptual Fidelity and CAE/IIF

We further examine the relationship between CAE/IIF and perceptual fidelity. Using Gemini 3.1 Pro Preview [8], we score 2,032 VITON-HD samples on detail, texture, silhouette, and color. Regression analyses relating these scores to CAE and IIF yield p-values below 0.05 in all cases, indicating statistically significant associations with generation quality. Figure 4 reports the Spearman correlations between the two indicators and the four fidelity scores. CAE is more strongly correlated with high-frequency attributes, namely detail and texture, whereas IIF is more strongly correlated with low-frequency attributes, namely silhouette and color. This suggests that the indicators capture complementary aspects of perceptual fidelity, consistent with our analysis: optimizing CAE helps preserve a detailed, unbiased condition, while optimizing IIF provides the noise latent with suficient deterministic guidance. Figure 5 shows how each indicator varies with the fidelity scores most strongly associated with it. The trends support reducing CAE and increasing IIF as an effective optimization direction.

## 4.6 Computational Efficiency

We evaluate RealFit’s computational cost, memory usage, and inference time. Because the condition branch is independent of the noise latent and timestep t, its keys and values can be computed once and reused throughout denoising. As shown in Table 4, the conditional KV cache removes redundant computation and reduces end-to-end inference time by approximately 75% without compromising fidelity. It also reduces peak GPU memory usage by avoiding repeated processing of condition tokens, improving the practicality of deployment.

Table 4 Computational cost and inference time with and without the conditional KV cache.
<table><tr><td>Methods</td><td> $\mathrm { F L O P s \ ( T ) }$ </td><td>Memory (MB)</td><td>Time (s)</td></tr><tr><td> $\mathrm { w / o }$  Cache</td><td>0.052</td><td>25032</td><td>102.2</td></tr><tr><td> $\mathrm { w } / $  Cache</td><td>0.027</td><td>19436</td><td>24.7</td></tr></table>

## 5 Conclusion

We present RealFit, a high-fidelity VTON framework based on asymmetric information dynamics in DiTs. Two diagnostic indicators, Conditional Attention Entropy and Injected Information Flux, characterize the information bottleneck in symmetric architectures. Unidirectional Information Flow protects deterministic garment features from stochastic noise, while Decoupled Timestep Modulation maintains a strong conditional signal. RealFit achieves state-of-the-art fidelity across benchmarks, and its time-invariant condition branch supports a conditional KV cache that reduces inference time by approximately 75% without additional training. These results highlight the potential of asymmetric interactions for high-fidelity, identity-preserving generation.

## References

[1] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel Cohen-Or. Stable flow: Vital layers for training-free image editing. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[2] Mikołaj Bińkowski, Dougal J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. In International Conference on Learning Representations (ICLR), 2018.

[3] Andreas Blattmann, Robin Rombach, Huan Ling, Seung Wook Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: Highresolution video synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[4] Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual tryon via misalignment-aware normalization. In Proceed-

ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14126– 14135, 2021.

[5] Yewon Choi, Sangkyung Kwak, Kyungjune Lee, Hyungjin Choi, and Jaegul Shin. Idm-vton: Improving difusion models for authentic virtual try-on in the wild. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. URL https://arxiv.org/abs/2403.05139.

[6] Zheng Chong, Xiao Dong, Haonan Li, Shiyuan Zhang, Wenqi list Zhang, Xuesong Zhang, Han Zhao, Deqiang Jiang, and Xiaoran Liang. Catvton: Concatenation is all you need for virtual try-on with difusion models. In International Conference on Learning Representations (ICLR), 2025.

[7] Tsu-Jui Fu, Yusu Qian, Chen Chen, Wenze Hu, Zhe Gan, and Yinfei Yang. Univg: A generalist difusion model for unified image generation and editing. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[8] Google. Gemini 3 pro preview. https://deepmind. google/technologies/gemini/, 2025. Accessed: 2026-01-20.

[9] Hailong Guo, Bohan Zeng, Yiren Song, Wentao Zhang, Chuang Zhang, and Jiaming Liu. Any2anytryon: Leveraging adaptive position embeddings for versatile virtual clothing tasks, 2025.

[10] Xintong Han, Zuxuan Wu, Zhe Wu, Ruichi Yu, and Larry S. Davis. Viton: An image-based virtual try-on network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7543–7552, 2018.

[11] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

[12] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

[13] Ji Woo Hong, Tri Ton, Trung X. Pham, Gwanhyeong Koo, Sunjae Yoon, and Chang D. Yoo. Ita-mdt: Image-timestep-adaptive masked difusion transformer framework for image-based virtual tryon. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[14] Tasin Islam, Alina Miron, Xiaohui Liu, and Yongmin Li. Stylevton: A multi-pose virtual tryon with identity and clothing detail preservation.

Neurocomputing, 594:127887, 2024. ISSN 0925- 2312. doi: https://doi.org/10.1016/j.neucom.2024. 127887. URL https://www.sciencedirect.com/ science/article/pii/S0925231224006581.

[15] Boyuan Jiang, Xiaobin Hu, Donghao Luo, Qingdong He, Chengming Xu, Jinlong Peng, Jiangning Zhang, Chengjie Wang, Yunsheng Wu, and Yanwei Fu. Fitdit: Advancing the authentic garment details for high-fidelity virtual try-on, 2024.

[16] Jeongho Kim, Guojung Gu, Minho Park, Sunghyun Park, and Jaegul Choo. Stableviton: Learning semantic correspondence with latent difusion model for virtual try-on. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8176–8185, 2024.

[17] Jiyoung Kim, Youngjin Shin, Siyoon Jin, Dahyun Chung, Jisu Nam, Tongmin Kim, Jongjae Park, Hyeonwoo Kang, and Seungryong Kim. Coral: Correspondence alignment for improved virtual try-on, 2026.

[18] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[19] Black Forest Labs. Flux.1: Text-to-image generation with 12b rectified flow transformers. Technical Report, 2024. URL https://blackforestlabs.ai/.

[20] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, and Jack English. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

[21] Yuhan Li, Hao Zhou, Wenxiang Shang, Ran Lin, Xuanhong Chen, and Bingbing Ni. Anyfit: Controllable virtual try-on for any combination of attire across any scenario, 2024.

[22] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic diferential equations. In International Conference on Learning Representations (ICLR), 2021.

[23] Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: High-resolution multi-category virtual try-on. In Proceedings of the European Conference on Computer Vision (ECCV), pages 345–362, 2022.

[24] Davide Morelli, Alberto Baldrati, Giuseppe Cartella, Marcella Cornia, Marco Bertini, and Rita Cucchiara. Ladi-vton: Latent difusion textual-inversion enhanced virtual try-on. In Proceedings of the ACM International Conference on Multimedia, 2023.

[25] William Peebles and Saining Xie. Scalable difusion models with transformers. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[26] Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, et al. Lumina-image 2.0: A unified and eficient image generative framework. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[27] Zhibin Qin, Zhenxiong Tan, Zeqing Wang, Songhua Liu, and Xinchao Wang. Spotedit: Selective region editing in difusion transformers. arXiv preprint arXiv:2512.22323, 2025.

[28] Samyam Rajbhandari, Jef Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.

[29] Jef Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.

[30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[31] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[32] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.

[33] Jinheng Xie, Weijia Mao, Zechen Bai Pillai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, et al. Show-o: One single transformer to unify multimodal understanding and generation. International Conference on Learning Representations (ICLR), 2025.

[34] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. Advances in Neural Information Processing Systems (NeurIPS), 2025.

[35] Yuhao Xu, Tao Gu, Weifeng Chen, and Arlene Chen. Ootdifusion: Outfitting fusion based latent difusion for controllable virtual try-on. arXiv preprint arXiv:2403.01779, 2024.

[36] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image difusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[37] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586– 595, 2018.

[38] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale difusion transformer. Advances in Neural Information Processing Systems (NeurIPS), 2025.

[39] Luyang Zhu, Dawei Yang, Tyler Zhu, Fitsum Reda, William Chan, Chitwan Saharia, Mohammad Norouzi, and Ira Kemelmacher-Shlizerman. Tryondifusion: A tale of two unets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4606–4615, 2023.