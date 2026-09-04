# Exploring the Potential of Contrastive Language-Image Pre-training for Multi-Source Remote Sensing Data

Xiangyang Miao<sup>1</sup> <sup>∗</sup>, Kelu Yao<sup>2</sup> <sup>∗</sup>, Yekai Huang<sup>1</sup>, Xiaogang Xu <sup>1</sup>, Junxiao Xue<sup>2</sup>, Minjun Shen<sup>2</sup>, Chenghui Lv<sup>1</sup>, Shanji Liu<sup>1</sup>, Yaying Chen<sup>1</sup>, Chao Li<sup>2</sup>

<sup>1</sup>School of Computer Science and Technology, Zhejiang University <sup>2</sup>Space-based Computing System Research Center, Zhejiang Lab

## Abstract

Contrastive language-image learning (CLIP) has become a key paradigm for remote sensing vision-language understanding. However, existing remote sensing contrastive learning methods are mostly built on RGB-oriented CLIP architectures, making it dificult to exploit heterogeneous sensors such as SAR, multi-spectral imaging (MSI), and hyperspectral imaging (HSI). To address this limitation, we propose OmniRSCLIP, an end-to-end contrastive learning framework that supports multi-source sensor inputs for remote sensing vision-language modeling. The key idea is to extend CLIP beyond its fixed RGB input interface without breaking the pretrained visual knowledge. To this end, OmniRSCLIP introduces Spectral-Spatial Basis Decomposition (SSBD), which formulates arbitrary-channel adaptation as a basis recomposition problem: pretrained CLIP patch embeddings provide transferable spatial bases, while wavelength-conditioned coeficients span sensor-specific embedding kernels within a constrained visual prior space. This design avoids forcing heterogeneous sensors into a fixed-channel input space, while aligning them in a unified image-text semantic space. We further introduce a spectral-context-aware mask-based contrastive learning scheme to suppress modality-specific redundant features and enhance fine-grained image-text alignment. Finally, to support multi-modal training, we construct OmniRS5M, the first large-scale remote sensing image-text corpus covering RGB, SAR, MSI, and HSI. Experiments on retrieval, zero-shot classification, and semantic localization show that OmniRSCLIP preserves strong RGB-domain performance while efectively extending CLIP to heterogeneous remote sensing modalities.

## Introduction

Language-image contrastive learning, exemplified by CLIP (Radford et al. 2021), has achieved remarkable progress in cross-modal semantic alignment and transferable representation learning, supporting tasks such as image-text retrieval (Huang et al. 2026; Qin et al. 2025; Tschannen et al. 2025), open-vocabulary segmentation (Liu et al. 2025; Li et al. 2026) and anomaly detection (Gao et al. 2026; Ma et al. 2025). This contrastive pre-training paradigm has also been introduced into the remote sensing community, ofering a promising route toward general-purpose earth observation representations. However, unlike natural images that are typically represented as RGB inputs, remote sensing data are collected from diverse sensors (Wolf, Rolih, and Zajc

![](images/4d07fc29db79404e6f17bd62023fa0ef4e112e6b8b7a6e06a67328d4e8174156.jpg)  
Figure 1: Comparison of remote sensing image-text alignment paradigms. (a) Existing methods are mainly built for RGB imagery and are dificult to extend to heterogeneous sensor inputs. (b) OmniRSCLIP uses SSBD as a modalityadaptive visual embedding interface, enabling a single model to process RGB, SAR, MSI, and HSI data within a unified image-text alignment framework.

2026; Yang et al. 2026b), including optical, SAR, MSI, and HSI, which difer substantially in channel numbers, spectral responses, and imaging mechanisms. Most existing remote sensing contrastive learning frameworks (Chen et al. 2025; Yang et al. 2026a) remain centered on RGB-oriented CLIP architectures or optical imagery, making it dificult to exploit heterogeneous sensor inputs in a unified manner. Therefore, developing an end-to-end remote sensing contrastive learning framework that can accommodate arbitrarychannel modalities while retaining the transferable knowledge of CLIP remains an important challenge.

This challenge first raises a fundamental model-design question: how can an RGB-oriented CLIP model be extended to heterogeneous remote sensing sensors without sacrificing the pretrained visual knowledge that makes contrastive transfer efective? A straightforward strategy is to transform non-RGB data into a three-channel RGB-compatible format, for example, by projecting MSI and HSI data with PCA (Gong et al. 2025; Zhao et al. 2025). Although this enables direct use of the original CLIP visual encoder, rich spectral observations are forced into a fixed input format, which can discard sensor-specific information. For example, HSI captures material-specific absorption signatures across hundreds of contiguous narrow bands, whereas projecting them onto only three principal components may discard subtle but discriminative spectral diferences. Recent arbitrary-channel embedding methods (Marimo et al. 2025; Xiong et al. 2024) ofer a more flexible direction, but they often rely on reinitialized or independently generated patch embedding layers. During end-to-end optimization, this may break the compatibility between the embedding head and the pretrained CLIP visual encoder, forcing the model to relearn lowlevel spectral-spatial projections and weakening the inherited RGB-domain prior. Therefore, the key issue is not merely to accept more input channels, but to build an adaptation mechanism that remains anchored to the pretrained CLIP embedding space while modeling sensor-specific spectral responses. Meanwhile, data supervision also limits multimodal remote sensing contrastive learning: existing optica image-text datasets (Lu et al. 2017; Yuan et al. 2021; Qu et al. 2016) mainly contain coarse scene-level captions, while paired language supervision for SAR, MSI, and HSI remains scarce.

To address these challenges, we propose OmniRSCLIP, the first end-to-end contrastive learning framework that supports multi-source sensor inputs for remote sensing visionlanguage modeling, as shown in Fig. 1. The core of OmniRSCLIP is the SSBD module, which serves as a modalityadaptive visual embedding interface between heterogeneous sensor inputs and the CLIP visual backbone. Rather than treating arbitrary-channel adaptation as an unconstrained kernel prediction problem, SSBD formulates it as a basis recomposition problem. Specifically, the pretrained CLIP patch embedding provides transferable spatial bases that retain generic local visual primitives, while wavelength-conditioned coefficients recompose channel-specific spectral-spatial kernels within this constrained visual prior space. This formulation reflects a natural property of multi-source remote sensing observations: diferent spectral bands and sensors often reflect similar spatial structures but require diferent spectral responses (Li et al. 2025; Hong et al. 2024). Consequently, OmniRSCLIP can align RGB, SAR, MSI, and HSI observations within a unified image-text semantic space without forcing heterogeneous data into a three-channel representation or relearning the input layer from scratch. Furthermore, although SSBD preserves sensor-specific information from heterogeneous inputs, not all resulting visual features contribute equally to alignment with rich textual descriptions. We therefore develop a spectral-context-aware maskbased contrastive learning scheme, emphasizing the most text-relevant visual features to further refine cross-modal alignment.

Finally, to support large-scale multi-modal training, we construct OmniRS5M, the first large-scale remote sensing image-text corpus covering RGB, SAR, MSI, and HSI modalities. OmniRS5M is built through a hybrid data collection and modality-aware caption distillation pipeline. Building on existing reliable image-text datasets, we further expand language supervision by converting visual-only RGB, SAR, MSI, and HSI datasets using modality-specific prompts and advanced VLMs. The resulting global scene descriptions and local details are further recombined into short and long caption candidates, providing multi-granularity supervision for contrastive pre-training. After filtering and manual quality inspection, OmniRS5M contains 4.67 million images and 73.90 million text candidates, including both short and long captions, ofering the data foundation for exploring contrastive language-image pre-training on multi-source remote sensing data.

The main contributions of this paper are summarized as follows:

• We propose OmniRSCLIP, an end-to-end remote sensing vision-language framework for unified modeling of heterogeneous sensor inputs. To bridge arbitrary-channel imagery with the CLIP visual backbone, we introduce SSBD as a modality-adaptive visual embedding interface, which generates wavelength-aware spectral-spatial kernels while inheriting the RGB-domain visual prior of CLIP.

• We construct OmniRS5M, the first large-scale multimodal remote sensing image-text corpus covering RGB, SAR, MSI, and HSI modalities. With modality-aware caption distillation and global-local text synthesis, OmniRS5M provides diverse multi-granularity supervision for learning unified remote sensing vision-language representations.

• Extensive experiments on short-text, long-text, and multimodal retrieval benchmarks validate the efectiveness of OmniRSCLIP. The results demonstrate its advantages over existing remote sensing vision-language models and alternative arbitrary-channel adaptation strategies across heterogeneous remote sensing modalities.

## Related Work

## Contrastive Language-Image Pre-training

CLIP (Radford et al. 2021) learns a shared visionlanguage embedding space through large-scale image-text contrastive pre-training, providing a solid foundation for open-vocabulary recognition and image-text retrieval. Recent studies in the natural image domain further show that extending the text context and strengthening fine-grained semantic alignment can substantially improve CLIP’s ability to understand complex scenes (Zhang et al. 2024a; Asokan, Wu, and Albreiki 2025; Xie et al. 2025a; Wu et al. 2026). These advances provide an important basis for transferring CLIPstyle contrastive learning to remote sensing. In the remote sensing community, early studies first fine-tune CLIP with large-scale short-caption satellite image-text datasets, including RemoteCLIP (Liu et al. 2024), GeoRSCLIP (Zhang et al. 2024b), and SkyScript (Wang et al. 2024). Later works gradually move toward richer semantic supervision: DGTRS-CLIP (Chen et al. 2025) introduces dual-granularity semantic alignment to improve the understanding of complex remote sensing scenes, while GeoAlignCLIP (Yang et al. 2026a) explicitly enforces global-local multi-view consistency. However, unlike natural images, remote sensing data are acquired by diverse sensors with diferent channel numbers, spectral responses, and imaging mechanisms. Existing methods are still primarily designed for RGB imagery. Although several studies have made preliminary attempts to extend contrastive learning to MSI and HSI data, they often require separate visual encoders or modality-specific embedding heads for diferent sensors, making it dificult to build a unified framework for heterogeneous remote sensing inputs (Jain et al. 2026; Marimo et al. 2025). In contrast, OmniRSCLIP supports multi-source remote sensing inputs within a single contrastive learning framework while preserving the pretrained CLIP visual prior.

## Channel-Adaptive Embedding

Earth observation data exhibit highly diverse channel configurations, ranging from single-channel SAR data to HSI with hundreds of narrow bands. To accommodate such variablechannel inputs, recent studies have explored several forms of channel-adaptive visual embedding. HyperFree (Li et al. 2025) adopts wavelength-aware dictionaries but sufers limited cross-modality transferability due to fixed spectral range constraints. EarthDial (Soni et al. 2025) processes arbitrary modalities by iteratively grouping input channels into threechannel subsets, while Any-Optical-Model (Li, Li, and Hong 2026) applies a shared convolutional kernel across spectral bands; both strategies improve input compatibility but may insuficiently model channel-specific spectral responses. The most related method to ours is DOFA (Xiong et al. 2024), which dynamically predicts patch-embedding weights from spectral wavelengths. However, generating the full embedding kernel introduces a large prediction space and may reduce compatibility with the pretrained CLIP embedding prior. In contrast, SSBD formulates arbitrary-channel adaptation as compact spectral-spatial basis recomposition, preserving CLIP-inherited spatial priors while predicting only wavelength-conditioned basis coeficients.

## Method

This section details OmniRSCLIP, an end-to-end remote sensing vision-language framework that generalizes CLIP to heterogeneous Earth observation modalities. OmniRSCLIP addresses arbitrary-channel sensor inputs through its core SSBD module, which serves as a modality-adaptive visual embedding interface between multi-source imagery and the CLIP visual backbone. In addition, a spectral-context-aware mask-based contrastive learning scheme is adopted during training to further improve cross-modal alignment.

## Architecture Overview

As illustrated in Fig. 2, OmniRSCLIP follows the dual-tower design of CLIP, where visual and text encoders project remote sensing images and language descriptions into a shared contrastive space. To support heterogeneous sensors, we replace the fixed RGB patch embedding with SSBD: given an arbitrary-channel image $\mathbf { X } \in \mathbb { R } ^ { C \times H \times W }$ , where C denotes the number of input channels and H and W denote the image height and width, together with its wavelength sequence $\bar { \lambda = } \{ \bar { \lambda _ { i } } \} _ { i = 1 } ^ { C }$ , SSBD dynamically generates spectral-spatial kernels to extract visual features compatible with the CLIP visual backbone. Thus, RGB, SAR, MSI, and HSI inputs can share the same visual backbone. For the language branch, we employ KPS (Zhang et al. 2024a) to extend the context length from 77 to 248 tokens for long remote sensing descriptions. Moreover, we extend the mask-based alignment mechanism (Xie et al. 2025b) by injecting SSBD-derived spectral context into the Mask Network, enabling captionand sensor-aware feature selection for refined cross-modal alignment.

## Motivation

The central dificulty in adapting CLIP to multi-source remote sensing data is the mismatch between the fixed RGB patch embedding and variable-channel sensor observations. The original CLIP patch embedding is parameterized by $\mathbf { W } _ { \mathrm { c l i p } } \ \in \ \mathbb { R } ^ { D \times 3 \times K \times K }$ , where D denotes the embedding feature dimension and K is the patch size, and is therefore tied to three-channel RGB inputs. A straightforward solution is to project non-RGB data into RGB space, for example by applying PCA to MSI or HSI data before feeding them into CLIP (Gong et al. 2025; Zhao et al. 2025). Although this strategy preserves the original CLIP parameters, it compresses high-dimensional spectral observations into three channels and may discard sensor-specific cues. Another representative direction, DOFA (Xiong et al. 2024), predicts the entire convolutional embedding layer from wavelength information. For an input with C channels, this layer contains $D C K ^ { 2 }$ kernel parameters. This ofers arbitrary-channel flexibility, but for high-dimensional sensors, the large parameter-generation space can increase the early optimization burden and cause the generated embeddings to deviate from the pretrained CLIP visual prior.

To build a structured adaptation mechanism, we use the notion of basis representation from linear algebra. For a vector w $\in \mathbb { R } ^ { M }$ , a set of M linearly independent basis vectors $\lbrace  { \mathbf { b } } _ { m } \rbrace _ { m = 1 } ^ { M }$ can represent any vector in this space as

$$
\mathbf { w } = \sum _ { m = 1 } ^ { M } \alpha _ { m } \mathbf { b } _ { m } , \quad \mathbf { b } _ { m } \in \mathbb { R } ^ { M } ,\tag{1}
$$

where $\alpha _ { m }$ is the coordinate of w under the m-th basis vector. This view separates the representation into two components: a set of basis vectors that define the coordinate system, and a set of coeficients that specify how a particular vector is composed.

SSBD applies this idea to patch embedding kernels, but uses a compact learnable dictionary rather than a complete basis of the full kernel space. For the i-th spectral channel, the corresponding channel-wise kernel is $\mathbf { W } _ { i } \in \mathbb { R } ^ { D \times K \times K }$ which can be flattened into a vector $\mathbf { w } _ { i } \in \mathbb { R } ^ { M }$ with $M = D K ^ { 2 }$ . Instead of predicting all M entries of each $\mathbf { w } _ { i }$ directly, SSBD approximates the channel-wise kernel using N learnable spatial bases $\{ \mathbf { b } _ { n } \} _ { n = 1 } ^ { N }$ with $\mathbf { b } _ { n } \in \mathbb { R } ^ { M }$

$$
\mathbf { w } _ { i } \approx \sum _ { n = 1 } ^ { N } \alpha _ { i , n } ( \pmb { \lambda } ) \mathbf { b } _ { n } , \quad N \ll M ,\tag{2}
$$

![](images/8d7bef34ebbf14e2befe3a7fb74485a392c2626f18233e8f9c475fa8ffb11926.jpg)  
Figure 2: Overview of the proposed OmniRSCLIP framework. The key contribution is SSBD, a wavelength-conditioned input interface that dynamically generates embedding weights to adapt multi-source remote sensing images to a shared CLIP visual backbone, followed by feature selection guided by both text and spectral context via a mask network for image-text alignment.

where $\boldsymbol { \lambda }$ is the central wavelength of the sampling sensor channel; $\alpha _ { i , n } ( \cdot )$ represents a wavelength-conditioned coeficient function that assigns the n-th spatial basis to the i-th spectral channel according to the full spectral configuration of the input. After reshaping $\mathbf { w } _ { i }$ back to $\mathbf { W } _ { i } ,$ , it becomes the patch embedding kernel for the corresponding spectral channel. Therefore, SSBD reduces the dynamic prediction target from CM full embedding parameters to CN wavelengthconditioned coeficients, where $N \ll M$ . This compact subspace approximation is motivated by the observation that channel-specific embedding kernels share reusable spatial structures across bands and modalities. In practice, our ablation study in Fig. 4 shows that a basis number far smaller than the full kernel dimensionality is suficient to achieve strong performance, indicating that this compact parameterization provides an efective trade-of between adaptation capacity and optimization stability.

## Spatial Basis Generation

Efective spatial bases should retain the visual common sense learned by CLIP, since edges, textures, and local structures remain useful across heterogeneous sensors. Therefore, instead of learning the basis kernels from scratch, SSBD adopts a prior inheritance strategy. Given the original CLIP patch embedding weight $\mathbf { W } _ { \mathrm { c l i p } } \in \mathbb { R } ^ { D \times 3 \times K \times K }$ , we first collapse the RGB channel dimension to extract a spectral-agnostic spatial prototype $\mathbf { B } _ { \mathrm { b a s e } } \in \mathbb { R } ^ { D \times K \times K }$

$$
\mathbf { B } _ { \mathrm { b a s e } } = \mathrm { A v g P o o l } ( \mathbf { W } _ { \mathrm { c l i p } } ) ,\tag{3}
$$

where $D$ denotes the embedding dimension and $K$ is the patch size. This simple averaging operation removes the color-specific channel dependency while preserving the dominant spatial patterns encoded in the pretrained patch embedding. We then construct N spatial bases by adding small-magnitude perturbations to the pretrained prototype:

$$
{ \bf B } _ { n } = { \bf B } _ { \mathrm { b a s e } } + \epsilon _ { n } , \quad \epsilon _ { n } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \quad n = 1 , \ldots , N .\tag{4}
$$

In this way, all spatial bases are anchored in the pretrained feature space of the CLIP visual backbone at initialization, while the perturbations provide enough diversity for diferent bases to specialize during end-to-end multi-modal training.

## Wavelength-Conditioned Coeficient Generation

A spectral band should not be treated as an isolated input channel. Its visual response depends on both its own spectral position and the wavelength configuration of the sensor, since neighboring bands, spectral gaps, and bandwidth density jointly determine what physical information the channel carries. Therefore, SSBD formulates coeficient generation as a sequence-level spectral reasoning problem: for each input sample, the model predicts how the shared spatial bases should be mixed according to the complete wavelength sequence.

Given the central wavelength $\lambda _ { i }$ of the i-th channel, we first map it into a high-dimensional spectral token:

$$
\mathbf { e } _ { i } = \phi \left( \operatorname { P E } ( \lambda _ { i } ) \right) ,\tag{5}
$$

where $\mathrm { P E } ( \cdot )$ denotes sinusoidal wavelength encoding and $\phi ( \cdot )$ is a lightweight residual MLP. Next, to further capture inter-band dependencies, we prepend a learnable spectral query $\mathbf { e _ { s } }$ to the wavelength-token sequence and feed the full sequence into a Transformer encoder:

$$
[ \mathbf { h } _ { s } , \mathbf { h } _ { 1 } , \dots , \mathbf { h } _ { C } ] = \mathrm { T r a n s f o r m e r } \left( [ \mathbf { e } _ { \mathrm { s } } , \mathbf { e } _ { 1 } , \dots , \mathbf { e } _ { C } ] \right) ,\tag{6}
$$

where $\mathbf { h } _ { s }$ summarizes the global spectral context, and $\mathbf { h } _ { i }$ is the contextualized representation of the i-th spectral band. Unlike an independent per-channel mapping, this selfattention formulation allows the coeficient of each band to depend on its relative location within the whole spectral profile. The basis mixing coeficients are then predicted by:

$$
\begin{array} { r } { \pmb { \alpha } _ { i } ( \pmb { \lambda } ) = \operatorname { L i n e a r } _ { \alpha } ( \mathbf { h } _ { i } ) , \quad \pmb { \alpha } _ { i } ( \pmb { \lambda } ) \in \mathbb { R } ^ { N } . } \end{array}\tag{7}
$$

In this way, $\alpha _ { i }$ becomes a wavelength-conditioned basis coordinate rather than a freely learned channel-specific parameter. The same spatial basis can be reused across sensors, while diferent wavelength configurations induce diferent basis mixtures for RGB, SAR, MSI, and HSI inputs. In particular, for SAR inputs, whose imaging mechanism difers from optical spectral sensing, the corresponding entry in λ denotes the C-band operating wavelength rather than an optical spectral band.

## Dynamic Spectral-Spatial Patch Embedding

With the wavelength-conditioned coeficients, SSBD synthesizes a channel-wise patch embedding kernel for each input band by combining the shared spatial bases:

$$
\mathbf { W } _ { i } = \sum _ { n = 1 } ^ { N } \alpha _ { i , n } ( \pmb { \lambda } ) \mathbf { B } _ { n } , \quad \mathbf { W } _ { i } \in \mathbb { R } ^ { D \times K \times K } .\tag{8}
$$

Here, $\mathbf { B } _ { n }$ is the reshaped kernel form of $\mathbf { b } _ { n }$ in Eq. 2. In practice, to improve training stability, we further perform mean-variance normalization on the synthesized kernels, and denote the normalized kernel as $\widehat { \mathbf { W } } _ { i }$ . The normalized kernels are then used to embed an arbitrary-channel image by summing channel-wise patch responses:

$$
\mathbf { Z } = \sum _ { i = 1 } ^ { C } \mathbf { X } _ { i } \circledast { \widehat { \mathbf { W } } } _ { i } ,\tag{9}
$$

where ⊛ denotes convolution operation. Thus, the output $\mathbf { Z } \in \mathbb { R } ^ { D \times H ^ { \prime } \times W ^ { \prime } }$ , where $H ^ { \prime }$ and $W ^ { \prime }$ denote the height and width of the embedded feature map, can be directly fed into the CLIP visual backbone. In this way, SSBD serves as a modality-adaptive embedding interface: spatial bases inherit the pretrained prior of CLIP, while wavelength-conditioned coeficients adapt these bases to arbitrary input channels without requiring a separate embedding layer for each sensor.

## Spectral-Context-Aware Mask-Based Alignment

Text-conditioned feature selection (Xie et al. 2025b) is efective for fine-grained image-text alignment, as it can suppress irrelevant visual dimensions and emphasize caption-related semantics. We extend this method to heterogeneous remote sensing by introducing a spectral-context-aware Mask Network, which incorporates the spectral context produced by SSBD and makes feature selection aware of both caption semantics and sensor characteristics. Let T denote the text features from the text encoder, and $\mathbf { h } _ { s }$ is the spectral token generated by SSBD. Subsequently, T and $\mathbf { h } _ { s }$ are jointly input into the Mask Network $\bar { G } _ { \mathrm { m a s k } } ( \cdot )$ to produce a text- and spectral-context-aware feature mask::

$$
\mathbf { m } = \sigma ( G _ { \mathrm { m a s k } } ( \mathbf { T } , \mathbf { h } _ { s } ) ) ,\tag{10}
$$

where $\sigma ( \cdot )$ is the sigmoid activation function. We retain mask values larger than 0.5 to select the activated visual features. Following (Xie et al. 2025b), the final training objective is:

$$
\mathcal { L } = \lambda _ { \mathrm { a l i g n } } ( \mathcal { L } _ { \mathrm { s i d m } } + \mathcal { L } _ { \mathrm { d i s m } } ) + \lambda _ { \mathrm { s p a r s e } } \mathcal { L } _ { \mathrm { s p a r s e } } ,\tag{11}
$$

where $\mathcal { L } _ { \mathrm { s i d m } }$ encourages the mask to select visual dimensions that are most relevant to the paired text; ${ \mathcal { L } } _ { \mathrm { d i s m } }$ enforces the masked visual representation to remain discriminative across images; $\mathcal { L } _ { \mathrm { s p a r s e } }$ encourages compact feature activation and improves the eficiency of the selected representation. Further details are provided in the supplementary material.

## Dataset Construction

To support the end-to-end training and evaluation of OmniRSCLIP, we construct OmniRS5M, a large-scale multimodal remote sensing image-text dataset covering RGB, SAR, MSI, and HSI. OmniRS5M is built through a data synthesis pipeline that integrates multi-source data collection, modality-aware semantic distillation, structured XML organization, and dynamic text-supervised caption synthesis. During data collection, OmniRS5M is organized into train and test subsets: for source datasets with oficial public splits, we follow the released partitions whenever available; otherwise, we randomly divide the collected samples into disjoint subsets, ensuring image-level separation between training and testing. The final synthesized semantic files are further recombined into short and long caption candidates, providing multi-granularity supervision for contrastive pretraining. Detailed dataset sources, split protocols, prompt templates, XML schema, and synthesis rules are provided in the supplementary material.

Overall, OmniRS5M contains 4.67M remote sensing images and 73.90M text candidates, including 27.82M short captions and 46.08M long captions. It covers 4.32M RGB images, 236.13K SAR images, 71.87K MSI images, and 47.71K HSI images. To the best of our knowledge, OmniRS5M is the first large-scale remote sensing image-text dataset thatjointly integrates RGB, SAR, MSI, and HSI modalities.

## Experiment

## Implementation Details

We implement OmniRSCLIP with CLIP-style vision backbones, including ViT-B/16 and ViT-L/14. For both backbones, the original fixed-channel patch embedding layer is replaced by the proposed SSBD module, and the number of spatial bases is set to 64. The loss weights $\lambda _ { \mathrm { a l i g n } }$ and $\lambda _ { \mathrm { s p a r s e } }$ are set to 10 and 2, respectively. To mitigate training data imbalance, we adopt a two-stage training strategy, training each stage for 5 epochs. The first stage uses only RGB image-text pairs, while the second incorporates multi-modal data. The global batch sizes are 1024 and 384 in the first and second stages, respectively. For both the vision and text encoders, the learning rate is $1 \dot { \times } 1 0 ^ { - 4 }$ in the first stage and $1 \times 1 0 ^ { - 5 }$ in the second, while that of the Mask Network remains $1 \times 1 0 ^ { - 3 }$ throughout. All ViT-B/16 experiments use 8 NVIDIA A100 GPUs, whereas ViT-L/14 experiments use 16.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="2">RGB</td><td colspan="2">MSI</td><td colspan="2">HSI</td><td colspan="2">SAR</td><td rowspan="2">Mean</td></tr><tr><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td></tr><tr><td rowspan="2">LongCLIP†</td><td></td><td></td><td>ViT-B/16 23.74/65.16 27.55/69.67</td><td>8.84/19.08</td><td>8.95/19.05</td><td>18.15/31.49</td><td>16.83/32.69</td><td>16.10/28.4015.00/27.70 26.77</td><td></td><td></td></tr><tr><td></td><td>ViT-L/1423.28/73.5132.05/77.69</td><td></td><td>10.51/23.44</td><td>10.39/23.55</td><td>20.31/33.53</td><td>18.87/33.17</td><td>19.30/34.6017.80/33.2030.32</td><td></td><td></td></tr><tr><td rowspan="2">HiMoCLIP†</td><td></td><td>ViT-B/1621.74/68.27</td><td>25.90/69.24</td><td>11.30/24.05</td><td>9.91/22.58</td><td>20.55/35.10</td><td>16.83/33.29</td><td>17.60/31.3016.10/30.4028.38</td><td></td><td></td></tr><tr><td></td><td>ViT-L/1424.54/74.0830.31/75.24</td><td></td><td>13.48/29.93</td><td>13.60/29.37</td><td>21.63/40.62</td><td>22.36/42.67</td><td></td><td>18.40/33.0018.10/34.1032.59</td><td></td></tr><tr><td rowspan="2">Ours</td><td></td><td>ViT-B/1630.90/75.7930.36/76.79</td><td></td><td>14.38/31.46</td><td>13.58/31.68</td><td>22.48/38.70</td><td>20.91/38.34</td><td>19.20/35.0017.10/34.2033.18</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>ViT-L/1431.45/76.7131.41/77.4014.35/30.4714.16/31.1921.03/38.5819.35/39.42</td><td>19.80/36.3017.80/37.1033.53</td><td></td><td></td></tr></table>

Table 1: Comparison of cross-modal retrieval performance on multi-source remote sensing data. Each cell reports short text/long-text Recall@1; Mean is the average of all Recall@1 values. <sup>†</sup> denotes fine-tuned models.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="2">Image-to-Text</td><td colspan="2">Text-to-Image</td></tr><tr><td>R@1</td><td>R@5</td><td>R@1</td><td>R@5</td></tr><tr><td></td><td>Results on RSITMD</td><td></td><td> Dataset</td><td></td><td></td></tr><tr><td rowspan="2">CLIP</td><td>ViT-B/16</td><td>7.96</td><td>21.46</td><td>8.36</td><td>26.11</td></tr><tr><td>ViT-L/14</td><td>10.62</td><td>27.43</td><td>10.04</td><td>33.33</td></tr><tr><td>DGTRSCLIP</td><td>ViT-B/16 ViT-L/14</td><td>21.90 22.12</td><td>39.82 42.26</td><td>15.18 17.35</td><td>40.80 43.27</td></tr><tr><td>RemoteCLIP</td><td>ViT-B/32 ViT-L/14</td><td>27.88 28.76</td><td>50.66 52.43</td><td>22.17 23.63</td><td>56.46 59.42</td></tr><tr><td>Ours</td><td>ViT-B/16 ViT-L/14</td><td>30.53</td><td>53.54</td><td>23.10</td><td>51.37</td></tr><tr><td></td><td>Results on RSICD Dataset</td><td>33.19</td><td>56.19</td><td>22.61</td><td>55.44</td></tr><tr><td>CLIP</td><td>ViT-B/16</td><td>5.67</td><td>14.82</td><td>5.29</td><td>16.96</td></tr><tr><td></td><td>ViT-L/14</td><td>6.59</td><td>17.75</td><td>4.96</td><td>18.72</td></tr><tr><td>DGTRSCLIP</td><td>ViT-B/16 ViT-L/14</td><td>11.53 14.36</td><td>29.19 28.36</td><td>9.92 11.22</td><td>28.60 31.86</td></tr><tr><td>RemoteCLIP</td><td>ViT-B/32 ViT-L/14</td><td>17.02 18.39</td><td>37.97 37.42</td><td>13.71 14.73</td><td>37.11 39.93</td></tr><tr><td>Ours</td><td>ViT-B/16 ViT-L/14</td><td>23.06 23.97</td><td>45.11 46.57</td><td>17.00 17.75</td><td>42.34 44.87</td></tr></table>

Table 2: Comparison of cross-modal retrieval performance on RSITMD and RSICD.

## Cross-Modal Retrieval

Following (Liu et al. 2024; Chen et al. 2025), we adopt cross-modal retrieval as the primary task to evaluate the performance of OmniRSCLIP. Specifically, in contrast to existing methods that predominantly focus on the RGB modality, we extend the evaluation scope to diverse remote sensing modalities: SAR, MSI, and HSI.

First, we evaluate OmniRSCLIP on public RGB short-text benchmarks, as summarized in Tab. 2. Although designed for heterogeneous sensors beyond RGB, OmniRSCLIP remains competitive on standard RGB retrieval tasks. With the ViT-L/14 backbone, it achieves the best image-to-text results on RSITMD (Yuan et al. 2021) and the best performance across all reported metrics on RSICD (Lu et al. 2017). These results show that OmniRSCLIP preserves strong RGB-domain alignment while extending CLIP to arbitrary-modality remote sensing data.

Beyond standard RGB benchmarks, Tab. 1 evaluates multisource image-text retrieval on RGB, SAR, MSI, and HSI.

![](images/542a4597c96b36a26b1401f340665d2d766068b5b2ac73b94a1fa9cddc6985a3.jpg)  
Figure 3: Aggregated heatmaps of diferent methods.

Since existing RGB-oriented VLMs cannot directly handle arbitrary-channel inputs, we replace the patch embedding layers of LongCLIP (Zhang et al. 2024a) and HiMoCLIP (Wu et al. 2026) with SSBD and fine-tune them under the same multi-modal setting. Thus, this comparison focuses on different baselines under a unified arbitrary-channel input interface, while diferent input adaptation strategies are evaluated in Tab. 5. We report both short- and long-text retrieval to examine the efect ofsemantic richness. OmniRSCLIP achieves the best mean performance, suggesting its efectiveness for unified alignment across heterogeneous modalities.

## Zero-shot Classification

Zero-shot classification evaluates the category-level transferability of the learned vision-language space without taskspecific supervision. We average prompt embeddings into class prototypes and predict the class with the highest imagetext cosine similarity. As shown in Tab. 3, OmniRSCLIP shows strong classification performance: ViT-B/16 achieves the best results among the compared methods on all four datasets, while ViT-L/14 also remains consistently competitive. These results indicate that OmniRSCLIP preserves category-level recognition ability while extending CLIP to heterogeneous sensor inputs.

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="4">Testing Dataset</td></tr><tr><td>AID</td><td>WHU</td><td>OPT</td><td>RSC</td></tr><tr><td rowspan="3">ViT-B/16</td><td>CLIP</td><td>66.38</td><td>81.29</td><td>73.44</td><td>61.44</td></tr><tr><td>DGTRSCLIP</td><td>74.97</td><td>90.15</td><td>85.43</td><td>73.78</td></tr><tr><td>Ours</td><td>85.21</td><td>95.32</td><td>89.68</td><td>82.14</td></tr><tr><td rowspan="4">ViT-L/14</td><td>CLIP</td><td>69.47</td><td>85.77</td><td>80.86</td><td>63.67</td></tr><tr><td>GeoRSCLIP</td><td>75.46</td><td>92.14</td><td>85.91</td><td>75.89</td></tr><tr><td>RemoteCLIP</td><td>88.53</td><td>96.12</td><td>88.17</td><td>70.78</td></tr><tr><td>DGTRSCLIP</td><td>77.49</td><td>93.03</td><td>89.19</td><td>82.63</td></tr><tr><td></td><td>Ours</td><td>85.35</td><td>96.52</td><td>90.32</td><td>79.30</td></tr></table>

Table 3: Comparison of zero-shot classification accuracy. AID, WHU, OPT, and RSC denote AID (Xia et al. 2017), WHU-RS19 (Xia et al. 2010), OPTIMAL-31 (Wang et al. 2018), and RS-C11 (Zhao, Tang, and Huo 2016), respectively.
<table><tr><td>SSBD</td><td>Mask Network</td><td>RGB</td><td>SAR</td><td>MSI</td><td>HSI</td></tr><tr><td rowspan="4">√</td><td></td><td>65.70</td><td>47.67</td><td>34.89</td><td>43.87</td></tr><tr><td></td><td>63.82</td><td>47.31</td><td>38.16</td><td>48.94</td></tr><tr><td>√</td><td>67.92</td><td>47.29</td><td>35.94</td><td>52.48</td></tr><tr><td>√</td><td>69.08</td><td>48.81</td><td>41.10</td><td>53.64</td></tr></table>

Table 4: Ablation of core components.

## Semantic Localization

To further examine fine-grained cross-modal grounding, we provide qualitative semantic localization results on the AIR-SLT (Yuan et al. 2022) dataset. Given a natural-language query, we decompose each image into multi-scale local views, compute their similarities with the query, and project the responses back to the original image to form a heatmap. As shown in Fig. 3, compared with existing remote sensing VLMs, OmniRSCLIP produces more concentrated responses on text-related regions and fewer activations on irrelevant areas, indicating stronger fine-grained grounding in complex remote sensing scenes.

## Ablation Study

We conduct ablation studies using ViT-B/16 and, unless otherwise specified, report the per-modality mean of R@1, R@5, and R@10 over both short- and long-text retrieval.

Efect of Core Components Table 4 validates the contributions of SSBD and Mask Network. Without SSBD, MSI and HSI inputs are projected to three channels by PCA, while SAR inputs are replicated to fit the original CLIP interface. In the absence of Mask Network, SSBD improves the average recall on MSI and HSI from 34.89/43.87 to 38.16/48.94, demonstrating its ability to better exploit spectral information than fixed three-channel adaptation. Incorporating Mask Network further improves performance across all modalities, with the full configuration achieving the best overall results. This confirms their complementarity: SSBD provides modality-adaptive spectral embedding, while the Mask Network further selects alignment-relevant features using semantic and spectral context.

<table><tr><td>Method</td><td>CLIP Prior</td><td>RGB</td><td>SAR</td><td>MSI</td><td>HSI</td></tr><tr><td>PCA</td><td>X</td><td>65.70</td><td>47.67</td><td>34.89</td><td>43.87</td></tr><tr><td>PCA</td><td>√</td><td>67.92</td><td>47.29</td><td>35.94</td><td>52.48</td></tr><tr><td>DOFA</td><td>X</td><td>67.54</td><td>46.94</td><td>40.95</td><td>52.31</td></tr><tr><td>SSBD</td><td>X</td><td>64.92</td><td>45.69</td><td>36.67</td><td>49.10</td></tr><tr><td>SSBD</td><td>√</td><td>69.08</td><td>48.81</td><td>41.10</td><td>53.64</td></tr></table>

Table 5: Efect of preserving the CLIP patch-embedding prior. Results are per-modality average recall.

![](images/2e5dab15579af5614ef68dc31c684b9160807875aa63291ab2ddf7be5156e25a.jpg)

![](images/c7a07cf913f1b14a9880eabcac4e9998646b5ee6a19b246d3cb8aed224a443dd.jpg)  
(a) Image-to-text  
(b) Text-to-image  
Figure 4: Efect of the number of spatial bases in SSBD.

Efect of CLIP Prior We further examine the efect of preserving the CLIP visual prior, defined as initializing patch embeddings or spatial bases from vanilla CLIP rather than random weights. As shown in Tab. 5, prior inheritance consistently improves the overall performance of both PCAand SSBD-based adaptation. In particular, CLIP-initialized SSBD achieves gains across all four modalities and outperforms DOFA, demonstrating that basis decomposition efectively adapts heterogeneous inputs while retaining transferable visual knowledge from CLIP.

Efect of the Number of Spatial Bases Unlike full-kernel generation, SSBD only learns a compact basis set and predicts wavelength-conditioned coeficients for basis recombination. As shown in Fig. 4, using only one basis leads to a clear performance drop, because it is equivalent to applying the same spatial transformation to all spectral bands and therefore cannot model inter-band spectral diferences. Increasing the basis number quickly improves the performance on MSI and HSI, while the curves become stable around 48 to 64 bases. Further increasing the dictionary to 128 brings limited additional gains. We thus adopt 64 bases by default for a favorable balance between performance and complexity.

## Conclusion

This paper introduces OmniRSCLIP, a unified visionlanguage framework for heterogeneous remote sensing image-text alignment. With SSBD as a modality-adaptive embedding interface and spectral-text-aware feature selection, OmniRSCLIP enables a single CLIP-based model to align RGB, SAR, MSI, and HSI observations within a shared semantic space. Experiments on retrieval, zero-shot classification, qualitative semantic localization, and ablations show that OmniRSCLIP extends CLIP beyond RGB imagery while preserving strong semantic transferability.

## References

Asokan, M.; Wu, K.; and Albreiki, F. 2025. FineLIP: Extending CLIP’s reach via fine-grained alignment with longer text inputs. In Proceedings of the Computer Vision and Pattern Recognition Conference, 14495–14504.

Chen, W.; Deng, Y.; Jin, W.; Chen, J.; Chen, J.; Feng, Y.; Xi, Z.; Liu, D.; Li, K.; and Meng, Y. 2025. DGTRSD and DGTRSCLIP: A dual-granularity remote sensing image–text dataset and vision–language foundation model for alignment. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing, 18: 29113–29130.

Gao, B.-B.; Zhou, Y.; Yan, J.; Cai, Y.; Zhang, W.; Wang, M.; Liu, J.; Liu, Y.; Wang, L.; and Wang, C. 2026. Adaptclip: Adapting clip for universal visual anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 4095–4103.

Gong, L.; Bai, R.; Li, Y.; Chen, Y.; Fan, F.; Zhao, S.; and Li, C. 2025. Bridging HSI and LiDAR data with frequencydomain hierarchical fusion for enhanced classification. IEEE Transactions on Geoscience and Remote Sensing.

Hong, D.; Zhang, B.; Li, X.; Li, Y.; Li, C.; Yao, J.; Yokoya, N.; Li, H.; Ghamisi, P.; Jia, X.; et al. 2024. SpectralGPT: Spectral remote sensing foundation model. IEEE transactions on pattern analysis and machine intelligence, 46(8): 5227–5244.

Huang, W.; Wu, A.; Yang, Y.; Luo, X.; Yang, Y.; Naseem, U.; Wang, C.; Dai, Q.; Dai, X.; Chen, D.; et al. 2026. LLM2CLIP: Powerful Language Model Unlocks Richer Cross-Modality Representation. In Proceedings ofthe AAAI Conference onArtificial Intelligence, volume 40, 5131–5139.

Jain, P.; Marcos, D.; Ienco, D.; Interdonato, R.; and Berchoux, T. 2026. TimeSenCLIP: A time series vision– language model for remote sensing. ISPRS Journal of Photogrammetry and Remote Sensing, 236: 99–119.

Li, B.; Dong, H.; Zhang, D.; Zhao, Z.; Sun, H.; and Gao, J. 2026. Exploring eficient open-vocabulary segmentation in the remote sensing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 5982–5991.

Li, J.; Liu, Y.; Wang, X.; Peng, Y.; Sun, C.; Wang, S.; Sun, Z.; Ke, T.; Jiang, X.; Lu, T.; et al. 2025. HyperFree: A channeladaptive and tuning-free foundation model for hyperspectral remote sensing imagery. In Proceedings of the Computer Vision and Pattern Recognition Conference, 23048–23058.

Li, X.; Li, C.; and Hong, D. 2026. Any-Optical-Model: A universal foundation model for optical remote sensing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 6539–6547.

Liu, F.; Chen, D.; Guan, Z.; Zhou, X.; Zhu, J.; Ye, Q.; Fu, L.; and Zhou, J. 2024. Remoteclip: A vision language foundation model for remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–16.

Liu, Y.; Wang, G.; Zhang, J.; Liu, Q.; and Huang, D. 2025. Unveiling the knowledge of clip for training-free open-vocabulary semantic segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 5649–5657.

Lu, X.; Wang, B.; Zheng, X.; and Li, X. 2017. Exploring models and data for remote sensing image caption generation. IEEE Transactions on Geoscience and Remote Sensing, 56(4): 2183–2195.

Ma, W.; Zhang, X.; Yao, Q.; Tang, F.; Wu, C.; Li, Y.; Yan, R.; Jiang, Z.; and Zhou, S. K. 2025. Aa-clip: Enhancing zero-shot anomaly detection via anomaly-aware clip. In Proceedings of the Computer Vision and Pattern Recognition Conference, 4744–4754.

Marimo, C. T.; Blumenstiel, B.; Nitsche, M.; Jakubik, J.; and Brunschwiler, T. 2025. Beyond the visible: Multispectral vision-language learning for earth observation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 359–375. Springer.

Qin, X.; Zhang, P.; Yang, J. J. O.; Zeng, G.; Li, Y.; Wang, Y.; Zhang, W.; and Dai, P. 2025. Clip is almost all you need: Towards parameter-eficient scene text retrieval without OCR. In Proceedings of the Computer Vision and Pattern Recognition Conference, 24873–24883.

Qu, B.; Li, X.; Tao, D.; and Lu, X. 2016. Deep semantic understanding of high resolution remote sensing image. In 2016 International Conference on Computer, Information and Telecommunication Systems, 1–5.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, 8748–8763. PmLR.

Soni, S.; Dudhane, A.; Debary, H.; Fiaz, M.; Munir, M. A.; Danish, M. S.; Fraccaro, P.; Watson, C. D.; Klein, L. J.; Khan, F. S.; et al. 2025. Earthdial: Turning multi-sensory earth observations to interactive dialogues. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 14303–14313.

Tschannen, M.; Gritsenko, A.; Wang, X.; Naeem, M. F.; Alabdulmohsin, I.; Parthasarathy, N.; Evans, T.; Beyer, L.; Xia, Y.; Mustafa, B.; et al. 2025. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786.

Wang, Q.; Liu, S.; Chanussot, J.; and Li, X. 2018. Scene classification with recurrent attention of VHR remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 57(2): 1155–1167.

Wang, Z.; Prabha, R.; Huang, T.; Wu, J.; and Rajagopal, R. 2024. Skyscript: A large and semantically diverse visionlanguage dataset for remote sensing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 5805–5813.

Wolf, F.; Rolih, B.; and Zajc, L. Č. 2026. Brewing stronger features: Dual-teacher distillation for multispectral Earth observation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 27815–27826.

Wu, R.; Chen, P.; Shen, F.; Zhao, S.; Hui, Q.; Gao, H.; Lu,T.; Liu, Z.; Zhao, F.; Wang, K.; et al. 2026. HiMo-CLIP:

Modeling semantic hierarchy and monotonicity in visionlanguage alignment. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 26974–26982.

Xia, G.-S.; Hu, J.; Hu, F.; Shi, B.; Bai, X.; Zhong, Y.; Zhang, L.; and Lu, X. 2017. AID: A benchmark data set for performance evaluation of aerial scene classification. IEEE Transactions on Geoscience and Remote Sensing, 55(7): 3965– 3981.

Xia, G.-S.; Yang, W.; Delon, J.; Gousseau, Y.; Sun, H.; and Maître, H. 2010. Structural high-resolution satellite image indexing. In ISPRS TC VII Symposium-100 Years ISPRS, volume 38, 298–303.

Xie, C.; Wang, B.; Kong, F.; Li, J.; Liang, D.; Zhang, G.; Leng, D.; and Yin, Y. 2025a. FG-CLIP: Fine-Grained Visual and Textual Alignment. In International Conference on Machine Learning, 68777–68793. PMLR.

Xie, S.; Lingjing, L.; Zheng, Y.; Yao, Y.; Tang, Z.; Xing, E. P.; Chen, G.; and Zhang, K. 2025b. Smartclip: Modular vision-language alignment with identification guarantees. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, 29780–29790.

Xiong, Z.; Wang, Y.; Zhang, F.; Stewart, A. J.; Hanna, J.; Borth, D.; Papoutsis, I.; Saux, B. L.; Camps-Valls, G.; and Zhu, X. X. 2024. Neural plasticity-inspired multimodal foundation model for earth observation. arXiv preprint arXiv:2403.15356.

Yang, X.; Fu, R.; Duan, Z.; Lin, Z.; Liu, X.; and Yang, B. 2026a. Geoalignclip: Enhancing fine-grained visionlanguage alignment in remote sensing via multi-granular consistency learning. arXiv preprint arXiv:2603.09566.

Yang, Y.; Tang, S.; Zhao, Q.; Zhang, H.; Wang, X.; and Deng, Z. 2026b. Sar-disentdm: a semantic-disentangled difusion model for limited-data sar image synthesis. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 11793–11801.

Yuan, Z.; Zhang, W.; Fu, K.; Li, X.; Deng, C.; Wang, H.; and Sun, X. 2021. Exploring a fine-grained multiscale method for cross-modal remote sensing image retrieval. IEEE Transactions on Geoscience and Remote Sensing, 60: 1–19.

Yuan, Z.; Zhang, W.; Li, C.; Pan, Z.; Mao, Y.; Chen, J.; Li, S.; Wang, H.; and Sun, X. 2022. Learning to evaluate performance of multimodal semantic localization. IEEE Transactions on Geoscience and Remote Sensing, 60: 3207171.

Zhang, B.; Zhang, P.; Dong, X.; Zang, Y.; and Wang, J. 2024a. Long-clip: Unlocking the long-text capability of clip. In European conference on computer vision, 310–325. Springer.

Zhang, Z.; Zhao, T.; Guo, Y.; and Yin, J. 2024b. RS5M and GeoRSCLIP: A large-scale vision-language dataset and a large vision-language model for remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 62: 1–23.

Zhao, L.; Tang, P.; and Huo, L. 2016. Feature significancebased multibag-of-visual-words model for remote sensing image scene classification. Journal of Applied Remote Sensing, 10(3): 035004–035004.

Zhao, Y.; Gao, F.; Jin, X.; Dong, J.; and Du, Q. 2025. Dynamic Frequency Feature Fusion Network for Multi-Source Remote Sensing Data Classification. IEEE Geoscience and Remote Sensing Letters.