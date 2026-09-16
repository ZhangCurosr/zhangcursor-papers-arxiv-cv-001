# Bridging the Perceptual Gap: Residual-Enhanced Downscaling and Manifold-Aware Perception Alignment Adaptation for NR-IQA

Yu Li <sup>1</sup> Zhengran Shen <sup>1</sup> Yachun Mi <sup>1</sup> Puchao Zhou <sup>1</sup> Shaohui Liu <sup>1</sup>

## Abstract

Leveraging Large Vision-Language Models like CLIP has recently set new benchmarks for No-Reference Image Quality Assessment (NR-IQA). However, the contrastive pretraining of CLIP inherently prioritizes semantic invariance, which often suppresses subtle perceptual signals, a phenomenon we term perceptual submergence. Furthermore, standard preprocessing techniques (e.g., cropping and interpolation) further exacerbate the loss of critical high-frequency quality cues. In this paper, we propose the Cross-modal Perception Alignment Adapter (CMPA), a manifold-aware framework designed to disentangle perceptual distortions from dominant semantics. CMPA introduces a Perception-Sensitive Feature Extractor (PFE) that projects CLIP features into a compact, low-dimensional subspace, explicitly magnifying distortion-induced off-manifold deviations. Subsequently, a Cross-Modal Perception Alignment Injector (PAI) aligns these features with qualityaware text anchors and re-injects them into the backbone. To ensure input fidelity, we also devise a Residual-enhanced Perceptual Downscaling strategy that adaptively compensates for resolution-induced information loss using Just Noticeable Difference (JND) guided frequency reinjection. Extensive evaluations on several benchmark datasets demonstrate that our approach significantly outperforms state-of-the-art methods, effectively recovering the perceptual signals submerged in semantic-dense representations.

## 1. Introduction

No-reference image quality assessment (NR-IQA) is a fundamental yet challenging task in computer vision, aiming to perceive visual degradations without access to pristine references. The task is particularly arduous for real-world images from User-Generated Content (UGC), which are afflicted by diverse and uncontrolled distortions such as motion blur, sensor noise, and mixed compression artifacts. While early works relied on handcrafted statistics (Mittal et al., 2012a;b; Saad et al., 2012) and CNNs (Su et al., 2020), recent state-of-the-art methods have pivoted towards leveraging Vision-Language Models, specifically CLIP (Radford et al., 2021), to harness rich semantic priors for quality prediction (Wang et al., 2023; Zhang et al., 2023).

![](images/f4bae3ecebfa7350f159b393e7f284e3333c70b876455f88507c5045c443c1d9.jpg)  
(b)  
Figure 1. Motivation of the proposed method. (a) Low-dimensional projection uncovers perceptual discrepancies by mapping pristine features onto a manifold while isolating distorted samples as outliers. (b) Our residual re-injection compensates for perceptual information loss inherent in conventional downsampling, achieving thumbnails with both semantic clarity and structural fidelity.

Despite their success, we argue that a critical theoretical disconnect persists in existing CLIP-based methods—ranging from early explorers (Wang et al., 2023; Zhang et al., 2023) to recent zero-shot frameworks like QualiCLIP (Agnolucci et al., 2024), typically treat CLIP embeddings as monolithic feature pools. These approaches typically treat pretrained CLIP embeddings as generic feature pools, overlooking the fundamental distributional characteristics of the feature space. CLIP is optimized for semantic alignment via contrastive learning, which explicitly encourages semantic invariance—the model is trained to map images with the same content (e.g., a sharp bird and a blurry bird) to the same semantic concept. Consequently, in the original highdimensional hypersphere, perceptual distortion signals are effectively treated as ”non-semantic noise” and suppressed.

As a conceptual illustration in Fig. 1(a), we hypothesize a phenomenon termed ’perceptual submergence’, where subtle distortion-induced variations are masked by dominant semantic features in the high-dimensional embedding space. In the original high-dimensional space, features of pristine (blue) and distorted (orange) images cluster tightly based on content, rendering quality-related deviations mathematically indistinguishable from semantic variations. However, our preliminary observation in Fig. 1(a) also suggests that when these features are projected into a compact subspace, the distortion-induced off-manifold perturbations become significantly more distinguishable.

Inspired by the Manifold Hypothesis (Bengio et al., 2013), we propose the Cross-modal Perception Alignment Adapter (CMPA). Our key insight is that while CLIP features are dominated by high-dimensional semantic manifolds, perceptually salient cues reside in a more compact, separable low-dimensional subspace. CMPA disentangles these cues through two synergistic components: (1) the Perception-Sensitive Feature Extractor (PFE), which projects highdimensional embeddings into a bottleneck subspace to filter out semantic redundancies and expose latent perceptual manifolds; and (2) the Cross-Modal Perception Alignment Injector (PAI), which aligns these visual residuals with qualityaware text anchors. This low-dimensional design is empirically corroborated by findings in LoDA (Xu et al., 2024), where IQA performance was found to peak at a remarkably low intrinsic dimension (e.g., d=64), further validating the efficacy of our projection strategy. MA-CLIP (Liao et al., 2025) also identified that CLIP’s semantic-dense features tend to overlook critical quality cues, further justifying our motivation to disentangle perceptual signals from the dominant semantic manifold.

Our framework also addresses the critical loss of perceptual cues often overlooked during mandatory input resizing. To fit fixed input constraints, practitioners often resort to grid-based cropping or traditional interpolation. As analyzed in Fig. 1(b), grid-based methods suffer from Semantic Fragmentation (Wu et al., 2022), while traditional interpolation induces Perceptual Aliasing (Parmar et al., 2022), indiscriminately smoothing out high-frequency textures that are vital for quality discrimination. To resolve this, we introduce a Residual-enhanced Perceptual Downscaling (RPD) preprocessor. By decomposing the raw image into base and residual frequencies, we adaptively re-inject lost high-frequency details into the downsampled input, guided by Just Noticeable Difference (JND) weighted map. As conceptually illustrated in Fig. 1(b), RPD is designed to preserve the clear details of the original capture, aiming to prevent the model from being misled by artifacts typically introduced by traditional resizing. In summary, our contributions are threefold:

1. We propose CMPA, a manifold-aware adapter comprising a Perception-Sensitive Feature Extractor and a Cross-Modal Perception Alignment Injector. This architecture effectively disentangles subtle quality cues from dominant semantic noise by projecting features into a low-dimensional perceptual subspace and subsequently re-injecting aligned perceptual anchors into the backbone.

2. We introduce a RPD preprocessor that adaptively compensates for resolution-induced information loss using frequency separation and JND-guided residual injection. This approach resolves the perception-distortion mismatch caused by traditional resizing, ensuring the model’s predictions are grounded in the intrinsic quality of the raw image rather than preprocessing artifacts.

3. Extensive evaluations on several benchmark datasets demonstrate that our framework consistently outperforms SOTA methods, validating the effectiveness of our methods.

## 2. Related Works

## 2.1. Learning based Image Quality Assessment

Deep learning has revolutionized NR-IQA by directly mapping image content to perceptual quality scores. Early approaches based on Convolutional Neural Networks (CNNs) typically employed pre-trained networks, such as ResNet (He et al., 2016), as backbones to leverage hierarchical feature extraction for quality prediction. Hyper-IQA (Su et al., 2020) introduced adaptive hyper-networks to decouple the IQA process into content understanding and quality prediction stages, thereby mimicking the contentsensitive characteristics of the Human Visual System (HVS). To capture long-range dependencies, Transformer-based methods (Vaswani et al., 2017), exemplified by MUSIQ (Ke et al., 2021) and MANIQA (Yang et al., 2022), utilize multiscale attention mechanisms to facilitate interaction between global and local regions. Concurrently, self-supervised methods (Madhusudana et al., 2022; Zhao et al., 2023) leverage contrastive learning to mitigate the scarcity of labeled data; however, they often struggle to bridge the domain gap between synthetic pre-training tasks and authentic distortions. Recently, Vision-Language Models (VLMs) have shifted the paradigm from pure visual regression to semantic-assisted quality assessment. Wang et al. (Wang et al., 2023) explored the zero-shot capabilities of CLIP but encountered misalignments between high-level semantics and low-level distortion patterns. Although subsequent finetuning strategies (Zhang et al., 2023; Li et al., 2025; Mi et al., 2024) incorporated scene and distortion priors, they typically process visual and textual branches independently prior to fusion. Crucially, these methods lack deep, layerwise interaction mechanisms, failing to explicitly guide image features to query quality-relevant definitions from the text encoder, thereby limiting fine-grained alignment. In this work, we take advantage of well-pretrained VLMs and adapt them via layer-wise interaction between visual features and quality-aware linguistic priors.

## 2.2. Efficient Model Fine-Tuning Methods

In recent years, visual foundation models have witnessed an exponential surge in parameter scale, evolving from early EfficientNet (0.48B parameters) (Pham et al., 2021) to contemporary Transformer-based massive models reaching 22B parameters (Dehghani et al., 2023). The shift from full fine-tuning to Parameter-Efficient Fine-Tuning (PEFT) has become essential as foundation model scales escalate (Dehghani et al., 2023). Current PEFT methodologies primarily bifurcate into two streams: Approaches like LoRA (Hu et al., 2022) and FacT (Jie & Deng, 2023) inject trainable low-rank matrices to adapt to downstream tasks. However, these methods treat the low-rank subspace primarily as an optimization tool, lacking a dedicated mechanism for perceptual feature analysis. Methods such as Adapt-Former (Chen et al., 2022) and ConvPass (Jie & Deng, 2022) insert lightweight bottleneck or convolutional modules. Recently, (Xu et al., 2024; Li et al., 2026) further validated the feasibility of employing such Transformer adaptation techniques to enhance IQA performance. Nevertheless, being designed for semantic reconstruction, these generic adapters tend to suppress subtle feature perturbations induced by image distortions. In contrast, our work moves beyond generic modulation by introducing a specialized mechanism designed to decouple quality-sensitive cues from dominant semantic features within a learned manifold structure.

## 2.3. Perceptual Image Pre-processing Strategies

Modern IQA backbones typically require fixed-size inputs, necessitating strategies to balance perceptual detail with semantic integrity. Current methodologies primarily rely on patch-based sampling to handle high-resolution inputs. Early works such as MUSIQ and FastIQA (Wu et al., 2022) extract multiple local patches to preserve fine-grained fidelity. To further capture scale-variant distortions, some works (Liu et al., 2024; Mi et al., 2025) introduces a multiresolution patch sampling strategy, aggregating girds from different scales to enrich the representation. While these methods effectively mitigate the information loss of a single resolution, the discrete nature of patch-based approaches inevitably disrupts the global semantic layout—a critical prior for Vision-Language Models like CLIP, and often requires complex aggregation mechanisms. Alternative adaptive sampling or downsampling methods aim to maintain the global structure, yet standard bicubic operators or ratedistortion-driven approaches (Liang et al., 2024) often suppress quality-aware high-frequency details like subtle noise or blur. In this work, we propose Residual-enhanced Perceptual Downscaling (RPD) to perform unified downsampling while adaptively re-injecting lost perceptual cues, ensuring a holistic and quality-aware input for assessment.

## 3. Methods

## 3.1. Overview

The overall architecture of our proposed framework is illustrated in Fig. 2. We introduce a parameter-efficient paradigm for BIQA, leveraging a frozen CLIP backbone augmented with our Residual-enhanced Perceptual Downscaling (RPD) and Cross-modal Perception Alignment (CMPA) modules.

To adapt raw high-resolution inputs to the fixed resolution required by pretrained encoders without sacrificing qualitysensitive details, we first apply the RPD preprocessor. Unlike standard interpolation that indiscriminately smooths out high-frequency transients, RPD explicitly captures and re-injects residual textures from the original image into the downsampled representation. This ensures that the backbone receives a perception-consistent input where critical distortion cues are preserved.

The downsampling image, alongside quality-related linguistic prompts such as [”Low quality image”, ”High quality image”], is then fed into the frozen CLIP image and text encoders. To address the phenomenon of perceptual submergence, we insert CMPA adapters across the transformer blocks to decouple perceptual signals from the dominant semantic manifold. Within each CMPA, the Perceptionsensitive Feature Extractor (PFE) projects high-dimensional features into a compact, low-dimensional bottleneck subspace—a design motivated by the hypothesis that the intrinsic dimension of perceptual quality is significantly lower than that of semantics. Subsequently, the Perception Alignment Injector (PAI) facilitates deep cross-modal interaction via bidirectional cross-attention, calibrating the visual features with quality-aware linguistic priors. Finally, the quality score is derived by computing the cosine similarity between the aligned visual global output and the textual quality anchors.

## 3.2. Perception-Sensitive Feature Extractor

To expose the subtle perceptual deviations often diluted in high dimensional semantic spaces, the PFE explicitly maps the multi-modal embeddings into a compact latent bottleneck. Let $F _ { \mathrm { i m g } } \in \mathbb { R } ^ { L \times D }$ and $F _ { \mathrm { t x t } } \in \mathbb { R } ^ { 2 \times D }$ denote the visual and textual features from the frozen CLIP blocks. We first perform a dimensionality reduction to project both modalities into a low-dimensional subspace $d \ll D ;$

![](images/d1b74b1aa826806ead29fd7d9d6b5c7c911ba9986f84094c2b1a57f23f00aa2d.jpg)  
Figure 2. Overall architecture of the proposed framework. Given an input image at original resolution, the RPD is first employed to produce a downsampled version that fits the backbone’s input constraints. Subsequently, the CMPA is integrated into the frozen CLIP blocks via a layer-wise scheme. Specifically, CMPA extracts and aligns perceptual features in a low-dimensional manifold through cross-modal guidance, which are then re-injected into the original CLIP features. The final image quality assessment is performed by computing the similarity score between the perception-aligned image embeddings and text prompts.

$$
Z _ { m } = \mathrm { L i n e a r } _ { D \to d } ( F _ { m } ) , \quad m \in \{ \mathrm { i m g } , \mathrm { t x t } \}\tag{1}
$$

This projection serves as a structural bottleneck that filters out redundant semantic invariants. To facilitate initial crossmodal synergy within this subspace, we employ a Shared MLP to refine the projected features:

$$
\hat { Z } _ { m } = \sigma \big ( W _ { 2 } \sigma ( W _ { 1 } Z _ { m } ) \big )\tag{2}
$$

By enforcing weight sharing across modalities, the Shared MLP acts as a modality-agnostic perceptual filter, forcing the bottleneck to capture unified distortion patterns rather than modality-specific noise.

## 3.3. Cross-Modal Perception Alignment Injector

Building upon the distilled bottleneck representations, the PAI stage executes deep cross-modal alignment to ensure that the extracted manifold deviations are perceptually consistent across both modalities. As illustrated in the detailed CMPA architecture, we employ a dual-path multi-head Cross-Attention(MHCA) mechanism to facilitate mutual modulation between visual and textual features.

The interaction in the d-dimensional subspace is defined by two complementary paths, where one modality acts as the query to calibrate the other: the visual features ${ \hat { Z } } _ { \mathrm { i m g } }$ serve as the query, while the textual anchors $\hat { Z } _ { \mathrm { t x t } }$ provide the key-value pairs to guide the perceptual alignment. The refined visual feature $\breve { F } _ { \mathrm { i m g } } ^ { \mathrm { a l i g n } }$ is formulated as:

$$
F _ { \mathrm { i m g } } ^ { \mathrm { a l i g n } } = \mathrm { M H C A } ( \hat { Z } _ { \mathrm { i m g } } , \hat { Z } _ { \mathrm { t x t } } , \hat { Z } _ { \mathrm { t x t } } )\tag{3}
$$

Symmetrically, the textual prompt embeddings query the visual patch features to capture local, distortion-sensitive cues, yielding the aligned textual feature $F _ { \mathrm { t x t } } ^ { \mathrm { a l i g n } }$

$$
\begin{array} { r } { F _ { \mathrm { t x t } } ^ { \mathrm { a l i g n } } = \mathrm { M H C A } ( \hat { Z } _ { \mathrm { t x t } } \hat { Z } _ { \mathrm { i m g } } , \hat { Z } _ { \mathrm { i m g } } ) } \end{array}\tag{4}
$$

This dual-path mechanism allows the linguistic quality anchors to dynamically “supervise” the visual extraction while simultaneously updating the text embeddings with imagespecific distortion evidence.

Following the cross-modal interaction, the aligned features are projected back to the original D-dimensional space. To preserve the robust semantic priors of the CLIP backbone while augmenting it with perceptual sensitivity, we apply a residual-style injection:

$$
F _ { m } ^ { \mathrm { f i n a l } } = F _ { m } + \mathrm { L i n e a r } _ { d \to D } ( F _ { m } ^ { \mathrm { a l i g n } } ) , \quad m \in \{ \mathrm { i m g } , \mathrm { t x t } \}\tag{5}
$$

where $F _ { m }$ denotes the original high-dimensional features from the frozen encoders. By re-injecting these refined “perceptual residuals,” the framework effectively bridges the gap between high-level semantic understanding and lowlevel degradation sensitivity.

## 3.4. Residual-Enhanced Perceptual Downscaling

To mitigate the fidelity loss inherent in standard resizing, we propose Residual-enhanced Perceptual Downscaling (RPD). As illustrated in Fig. 3, RPD rectifies the information deficit by decomposing the raw signal into multi-frequency components and re-injecting sampling residuals extracted via a Residual Enhanced Downscaling (RED) module. Given an input component $\mathbf { I } _ { \mathrm { i n } }$ , the RED operator $\mathcal { R } ( \cdot )$ extracts the sampling residual by measuring the deviation between the original signal and its reconstruction through a down-up sampling cycle:

![](images/fa3eb135acef4ff2760d8b1e43855358552acc6336836232fe5a9eecbbf8582d.jpg)  
Figure 3. Architecture of the RPD. Input images are decomposed into frequency-specific components and processed via the RED module to extract structural residuals. These residuals are then adaptively fused using a JND-guided weight map to generate the perceptually-consistent output $\dot { I } _ { t a r g e t }$

$$
\mathcal { R } ( \mathbf { I } _ { \mathrm { i n } } ) = \mathrm { d o w n } ( \mathbf { I } _ { \mathrm { i n } } ) + \mathrm { d o w n } ( \mathbf { I } _ { \mathrm { i n } } - \mathrm { u p } ( \mathrm { d o w n } ( \mathbf { I } _ { \mathrm { i n } } ) ) )\tag{6}
$$

where down(·) and up(·) denote the downsampling and upsampling operators, respectively. These residuals $\mathbf { I } _ { \mathrm { i n } } -$ $\mathrm { u p } ( \mathrm { d o w n } ( \mathbf { I } _ { \mathrm { i n } } )$ explicitly represent the high-frequency de tails and structural information that are typically diluted in standard downsampling.

To ensure the residual injection is perceptually optimal, we introduce an adaptive gain map A to modulate the injection intensity. We first estimate a Just Noticeable Difference (JND) map $\mathbf { M } _ { \mathrm { j n d } }$ based on luminance adaptation $F _ { \mathrm { l a } }$ (Chou & Li, 1995) and visual masking $F _ { \mathrm { v m } }$ (Yang et al., 2005):

$$
\mathbf { M } _ { \mathrm { j n d } } = \frac { 1 } { 2 } \Big ( F _ { \mathrm { l a } } ( L _ { m } ) + \boldsymbol { \beta } \cdot F _ { \mathrm { v m } } ( G ) \Big )\tag{7}
$$

where $L _ { m }$ denotes the local mean luminance estimated via Gaussian smoothing, and G is the texture gradient magnitude computed using Sobel operators. $\beta$ is a weighting coefficient. To optimize residual integration, we introduce an adaptive gain map A to modulate the injection intensity. Grounded in the Weber-Fechner Law, which characterizes the logarithmic nature of human visual perception, the gain is formulated as:

$$
\mathbf { A } = 1 + \log _ { 2 } \left( 1 + \left( 1 - { \frac { 1 } { d } } \right) \right) \cdot ( \mathbf { M } _ { \mathrm { j n d } } \cdot { \boldsymbol { \alpha } } )\tag{8}
$$

where d is the downsampling ratio and α is a scaling factor. The term $\log _ { 2 } ( 2 - 1 / d )$ serves as a differentiable weight within [0,1) that monotonically increases with the d. This ensures the injection intensity adaptively compensates for information loss, prioritizing residuals in regions where the HVS is most sensitive to distortions.

Finally, RPD performs frequency separation by applying a Gaussian filter $G ( k , \sigma )$ to decompose the raw image into low-frequency $\mathbf { I } _ { L }$ and high-frequency ${ \mathbf { I } } _ { H }$ components. The target representation $\mathbf { I } _ { t a r g e t }$ is resynthesized by recursively aggregating the base downsampled features with the residuals extracted by the RED operator $\mathcal { R } ( \cdot )$

$$
\mathbf { I } _ { t a r g e t } = \mathcal { R } ( \mathbf { I } _ { L } ) + \mathbf { A } \odot ( \mathcal { R } ( \mathbf { I } _ { H } ) )\tag{9}
$$

where $\odot$ denotes element-wise multiplication. By leveraging the RED-derived residuals, RPD provides a high-fidelity input that bridges the gap between semantic understanding and distortion sensitivity. Further rationale of the feasibility of RPD can be found in Appendix. B.

## 4. Experiments

## 4.1. Experimental Settings

Datasets: Our method is evaluated on classical IQA datasets, including four synthetic datasets, LIVE (Sheikh et al., 2006), CSIQ (Larson & Chandler, 2010), TID2013 (Ponomarenko et al., 2015), KADID-10k (Lin et al., 2019) and four authentic datasets, LIVEC (Ghadiyaram & Bovik, 2015), KonIQ-10k (Hosu et al., 2020), SPAQ (Fang et al., 2020), and FLIVE (Ying et al., 2021).

Implementation details: To ensure a fair comparison, we follow the experimental setup of LoDa (Xu et al., 2024). A key distinction in our approach is the omission of image resizing to better evaluate the effectiveness of the proposed RPD and alignment-amplification mechanism. To maintain data diversity consistent with LoDa’s strategy (which resizes the short edge to 384 and then crops to 224), we adopt a random cropping ratio of $3 8 4 / 2 2 4 \approx 0 . 6 $ relative to the shortest edge. Patches smaller than 224 are up-interpolated to meet the backbone’s input requirements. We employ the AdamW optimizer with a learning rate of 1e-3 and weight decay of 0.01, using a batch size of 128. Unless otherwise specified, we utilize CLIP with the ViT-B/32 backbone as default. Additional Implementation details and settings are provided in the Appendix. C.

## 4.2. Performance Comparison with SOTA

We compare our CMPA fine-tuning framework against a wide range of state-of-the-art NR-IQA methods (Zhang et al., 2015; 2020; Zhu et al., 2020; Ying et al., 2021; Su et al., 2020; Ke et al., 2021; Zhang et al., 2023; Qin et al., 2023; Saha et al., 2023; Guan et al., 2024; Li et al., 2025; Xu et al., 2024). The results are summarized in Table 1. Our method consistently outperforms existing approaches across both synthetically distorted and authentically distorted datasets, achieving the highest or near-highest SRCC and PLCC in most cases. This substantial performance margin demonstrates the effectiveness of our proposed CMPA framework in capturing perceptual quality across diverse distortion types. Notably, our method achieves these results with highly parameter-efficient fine-tuning: only 1.8M learnable parameters are tuned, far fewer than full fine-tuning of CLIP-based models or other parameter-heavy approaches. This highlights the parameter-friendliness of CMPA, making it practical for resource-constrained settings while delivering superior perceptual alignment. Compared to other CLIPbased methods (e.g., LIQE, GRMP-IQA), our approach benefits from explicit cross-modal guidance and refinement in the low-dimensional perceptual subspace. This design enables stronger perceptual sensitivity, leading to improved performance on both synthetic and authentic distortions.

Table 1. Performance comparison measured by medians of SRCC and PLCC, where the numbers within parentheses indicate the finetuned parameters of the model and bold entries indicate the top two results.
<table><tr><td rowspan="2">Method</td><td>LIVE</td><td>CSIQ</td><td></td><td>TID2013</td><td>KADID-10k</td><td>KonIQ-10k</td><td>LIVEC</td><td>SPAQ</td><td></td><td>FLIVE</td></tr><tr><td colspan="3">SRCC PLCC SRCC PLCC</td><td colspan="2">SRCC PLCC</td><td colspan="2">SRCC PLCC| SRCC PLCC</td><td colspan="2">SRCC PLCC SRCC</td><td colspan="2">PLCC SRCC PLCC</td></tr><tr><td>ILNIQE</td><td>0.902 0.906</td><td>0.822</td><td>0.865</td><td>0.521 0.648</td><td>0.534 0.558</td><td>0.523</td><td>0.537 0.508</td><td>0.508</td><td>0.712</td><td>0.294</td><td>0.332</td></tr><tr><td>DBCNN</td><td>0.968 0.971</td><td>0.946</td><td>0.959</td><td>0.816 0.865</td><td>0.851</td><td>0.856 0.875</td><td>0.884</td><td>0.851 0.869</td><td>0.713 0.911</td><td>0.915 0.545</td><td>0.551</td></tr><tr><td>MetaIQA</td><td>0.960 0.959</td><td>0.899</td><td>0.908</td><td>0.856 0.868</td><td>0.762</td><td>0.775 0.887</td><td>0.856</td><td>0.835 0.802</td><td></td><td>0.540</td><td></td></tr><tr><td>P2P-BM</td><td>0.959 0.958</td><td>0.899</td><td>0.902</td><td>0.862 0.856</td><td>0.840 0.849</td><td>0.872</td><td>0.885 0.844</td><td>0.842</td><td></td><td>0.526</td><td>0.507</td></tr><tr><td>HyperIQA(27M)</td><td>0.962 0.966</td><td>0.923</td><td>0.942</td><td>0.840 0.858</td><td>0.852</td><td>0.845 0.906</td><td>0.917</td><td>0.859 0.882</td><td>0.911</td><td>0.915 0.544</td><td>0.598 0.602</td></tr><tr><td>MUSIQ(27M)</td><td>0.940 0.911</td><td>0.871</td><td>0.893</td><td>0.773 0.815</td><td>0.875</td><td>0.872 0.916</td><td>0.928</td><td>0.702 0.746</td><td>0.918</td><td>0.921 0.566</td><td></td></tr><tr><td>TReS(152M)</td><td>0.969 0.968</td><td>0.922</td><td>0.942</td><td>0.863 0.883</td><td>0.859</td><td>0.858 0.915</td><td>0.928</td><td>0.846 0.877</td><td></td><td>0.544</td><td>0.661 0.625</td></tr><tr><td>LIQE(151M)</td><td>0.970 0.951</td><td>0.943</td><td>0.946</td><td></td><td>0.930</td><td>0.931 0.919</td><td>0.908</td><td>0.904 0.910</td><td></td><td></td><td></td></tr><tr><td>DEIQT(24M)</td><td>0.980 0.982</td><td></td><td></td><td>0.892 0.908</td><td>0.889</td><td>0.887 0.921</td><td>0.934</td><td>0.875 0.894</td><td>0.919</td><td>0.923 0.571</td><td>0.663</td></tr><tr><td>Re-IQA(48M)</td><td>0.970 0.971</td><td>0.945</td><td>0.960</td><td>0.804 0.861</td><td>0.872</td><td>0.885 0.914</td><td>0.923</td><td>0.840 0.854</td><td>0.918</td><td>0.925 0.575</td><td>0.675</td></tr><tr><td>QMamba(30M)</td><td>0.959 0.958</td><td>0.950</td><td>0.952</td><td>0.896 0.913</td><td>0.923</td><td>0.938 0.928</td><td>0.943</td><td>0.863 0.903</td><td>0.927</td><td>0.933 0.574</td><td>0.672</td></tr><tr><td>GRMP-IQA</td><td>0.981 0.983</td><td>0.949</td><td>0.955</td><td></td><td></td><td>0.934</td><td>0.945</td><td>0.897 0.916</td><td>0.927</td><td>0.932 0.616</td><td>0.704</td></tr><tr><td>LoDa(9M)</td><td>0.975 0.979</td><td></td><td></td><td>0.869 0.901</td><td>0.931</td><td>0.936 0.932</td><td>0.944</td><td>0.876 0.899</td><td>0.925</td><td>0.928 0.578</td><td>0.679</td></tr><tr><td>Ours(w/o PAI 1.5M)</td><td>0.977 0.981</td><td>0.947</td><td>0.957</td><td>0.873 0.902</td><td>0.936</td><td>0.938 0.933</td><td>0.944</td><td>0.901 0.911</td><td>0.925</td><td>0.927 0.579</td><td>0.679</td></tr><tr><td>Ours(1.8M)</td><td>0.982 0.985</td><td>0.952</td><td>0.962</td><td>0.878 0.905</td><td>0.937</td><td>0.940 0.935</td><td>0.945</td><td>0.903 0.915</td><td>0.927</td><td>0.931 0.584</td><td>0.686</td></tr><tr><td>Ours(with RPD)</td><td>0.983 0.987</td><td>0.951</td><td>0.961</td><td>0.899 0.911</td><td>0.938</td><td>0.941</td><td>0.938 0.948</td><td>0.903 0.918</td><td>0.929</td><td>0.934 0.587</td><td>0.689</td></tr></table>

Table 2. SRCC on the cross datasets validation. The best performances are highlighted with boldface, and subsequent tables maintain the same.
<table><tr><td rowspan="2">Training Testing</td><td colspan="2">FLIVE</td><td>LIVEC</td><td>KonIQ</td></tr><tr><td>KonIQ</td><td>LIVEC</td><td>KonIQ</td><td>LIVEC</td></tr><tr><td>DBCNN</td><td>0.716</td><td>0.724</td><td>0.754</td><td>0.755</td></tr><tr><td>P2P-BM</td><td>0.755</td><td>0.738</td><td>0.740</td><td>0.770</td></tr><tr><td>HyperIQA</td><td>0.758</td><td>0.735</td><td>0.772</td><td>0.785</td></tr><tr><td>TReS</td><td>0.713</td><td>0.740</td><td>0.733</td><td>0.786</td></tr><tr><td>DEIQT</td><td>0.733</td><td>0.781</td><td>0.744</td><td>0.794</td></tr><tr><td>LoDa</td><td>0.763</td><td>0.805</td><td>0.745</td><td>0.811</td></tr><tr><td>Ours(w/o PAI)</td><td>0.811</td><td>0.821</td><td>0.791</td><td>0.840</td></tr><tr><td>Ours</td><td>0.825</td><td>0.820</td><td>0.801</td><td>0.846</td></tr></table>

## 4.3. Cross-Dataset Evaluation

We further compare the generalizability of our method against competitive BIQA models in a cross-dataset setting following (Qin et al., 2023). Training is performed on one specific dataset, and testing is conducted on a different dataset without any fine-tuning or parameter adaptation.

Table 3. Data-efficient learning validation with the training set containing 20%, 40% and 60% images.
<table><tr><td rowspan="2">Mode</td><td rowspan="2">Methods</td><td colspan="2">KonIQ</td><td colspan="2">LIVEC</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td rowspan="4">20%</td><td>HyperIQA</td><td>0.869</td><td>0.873</td><td>0.776</td><td>0.809</td></tr><tr><td>DEIQT</td><td>0.888</td><td>0.908</td><td>0.792</td><td>0.822</td></tr><tr><td>LoDa</td><td>0.907</td><td>0.923</td><td>0.815</td><td>0.854</td></tr><tr><td>Ours</td><td>0.917</td><td>0.933</td><td>0.850</td><td>0.873</td></tr><tr><td rowspan="4">40%</td><td>HyperIQA</td><td>0.892</td><td>0.908</td><td>0.832</td><td>0.849</td></tr><tr><td>DEIQT</td><td>0.903</td><td>0.922</td><td>0.838</td><td>0.855</td></tr><tr><td>LoDa</td><td>0.922</td><td>0.935</td><td>0.849</td><td>0.879</td></tr><tr><td>Ours</td><td>0.926</td><td>0.941</td><td>0.885</td><td>0.906</td></tr><tr><td rowspan="4">60%</td><td>HyperIQA</td><td>0.901</td><td>0.914</td><td>0.843</td><td>0.862</td></tr><tr><td>DEIQT</td><td>0.914</td><td>0.931</td><td>0.848</td><td>0.877</td></tr><tr><td>LoDa</td><td>0.928</td><td>0.940</td><td>0.869</td><td>0.891</td></tr><tr><td>Ours</td><td>0.931</td><td>0.946</td><td>0.898</td><td>0.915</td></tr></table>

The experimental results, reported as the median SRCC across four datasets, are shown in Table 2. As observed, our method achieves the best performance on all datasets. Even without the further interactive guidance and refinement provided by PAI, our method can achieve strong capabilities simply through low-dimensional cross-modal alignment. These results manifest the generalization capability of the proposed method.

## 4.4. Data-Efficient Learning Validation

Following the evaluation protocol in DEIQT (Qin et al., 2023), we investigate the robustness of our fine-tuning framework under varying amounts of training data. We train on 20%, 40%, and 60% of the full training set from KonIQ-10k (randomly subsampled while preserving the original distribution), and evaluate on the standard test splits of KonIQ-10k and LIVEC using SRCC and PLCC. Our method consistently outperforms HyperIQA, DEIQT, and LoDa across all data regimes and both test sets. Notably, with only 40% of the training data, our approach achieves performance comparable to or better than the competitors full-data results. Even at 20% data, we remain highly competitive, demonstrating strong data efficiency. These results highlight the effectiveness of our fine-tuning framework in low-data regimes, where perceptual alignment is preserved with minimal samples. This efficiency is particularly valuable for real-world scenarios with limited labeled perceptual data. Results are summarized in Table 3.

Table 4. Comparison of different CLIP fine-tuning strategies on KADID-10k, KonIQ-10k, and SPAQ (SRCC / PLCC). Best results in bold.
<table><tr><td rowspan="2">Fine-tuning Method</td><td colspan="2">KADID-10k</td><td colspan="2">KonIQ-10k</td><td colspan="2">SPAQ</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td>CLIP (Original)</td><td>0.666</td><td>0.671</td><td>0.625</td><td>0.727</td><td>0.738</td><td>0.735</td></tr><tr><td>CLIP (Full fine-tune)</td><td>0.869</td><td>0.879</td><td>0.928</td><td>0.894</td><td>0.898</td><td>0.902</td></tr><tr><td>Adapter-CLIP (Gao et al., 2024)</td><td>0.928</td><td>0.930</td><td>0.874</td><td>0.893</td><td>0.914</td><td>0.915</td></tr><tr><td>LoRA (Hu et al., 2022)</td><td>0.910</td><td>0.911</td><td>0.918</td><td>0.923</td><td>0.912</td><td>0.915</td></tr><tr><td>COOP (Zhou et al., 2022)</td><td>0.901</td><td>0.938</td><td>0.895</td><td>0.904</td><td>0.864</td><td>0.866</td></tr><tr><td>Ours (w/o attn)</td><td>0.936</td><td>0.933</td><td>0.933</td><td>0.945</td><td>0.925</td><td>0.927</td></tr><tr><td>Ours</td><td>0.937</td><td>0.940</td><td>0.935</td><td>0.945</td><td>0.927</td><td>0.931</td></tr></table>

## 4.5. Comparison with Other Fine-Tuning Strategies

To fairly assess the effectiveness of our CMPA fine-tuning framework, we compare it against several established CLIPbased adaptation methods under identical experimental conditions: all models use the same random crop-based preinput processing (without our RPD sampling). Results are reported in Table 4. Full-parameter fine-tuning of CLIP yields reasonable gains over zero-shot, but common parameter-efficient methods (LoRA, COOP) underperform or show only marginal improvement. This is likely because they operate directly in the original high-dimensional feature space, where semantic and perceptual signals are entangled, diluting the perceptual signal during adaptation. Adapter-CLIP achieves the second-best performance among baselines, thanks to its moderate dimensionality reduction (to half the original dimension) followed by up-projection. This partial dimensionality reduction inadvertently isolates some perceptual-sensitive components, providing partial support for our low-dimensional perceptual subspace hypothesis. Our CMPA framework outperforms all compared methods across all datasets. Ablating the PAI attention module (Ours w/o attn) already yields strong results, demonstrating the effectiveness of the perceptual feature enhancement (PFE) component alone. Adding the PAI module further boosts performance by guiding low-dimensional features toward perceptually sensitive directions through attention-based interaction and alignment. This incremental gain validates our core design: combining explicit perceptual subspace enhancement with guided alignment yields the strongest perceptual fidelity. These results confirm that directly finetuning in the original CLIP space is suboptimal for perceptual tasks, while our low-dimensional, perceptually guided approach provides a more effective pathway.

![](images/40e25a13a9281ad1d886621f422155b32003d6206f7445e6b362e06676e9189f.jpg)  
Figure 4. Visualization of representational alignment and feature distributions. (a) RDM results show that the injected features achieve a significantly higher correlation with human perception (0.562) compared to the original CLIP (0.033). (b) t-SNE visualizations demonstrate that while original CLIP is dominated by semantic clustering, our injected residuals and fused features exhibit perceptual discriminability and hierarchical structure.

## 4.6. Qualitative Analysis

To validate the hypothesis of a low-dimensional perceptual subspace, we visualize the feature distributions using RSA and t-SNE. As shown in Fig. 4(a), the original CLIP features exhibit negligible correlation with human perceptual judgments (RSA=0.033). In contrast, the injected perceptual residuals achieve a significantly higher RSA score of 0.562, confirming that our CMPA effectively isolates perceptually salient information from the semantic-dominant CLIP space. The t-SNE projections in Fig. 4(b) further reveal that while original CLIP features form rigid semantic clusters, the injected residuals exhibit a hierarchical structure sensitive to distortion levels and textures rather than high-level semantics. The fused ”Total Feature” achieves superior perceptual discriminability by dispersing representations along perceptual axes while preserving necessary semantic coherence. These visualizations provide direct, qualitative support for our core hypothesis: perceptual information resides in a separable low-dimensional subspace within CLIP embeddings, and our CMPA framework—through guided cross-modal alignment—successfully extracts and enhances this subspace. The improved perceptual clustering in the fused features directly explains the superior performance observed in quantitative evaluations.

We further visualize the downsampled images produced by our RPD method alongside traditional interpolation baselines (Bicubic and Lanczos). The results, including zoomedin regions with corresponding SSIM and LPIPS scores relative to the original high-resolution image, are shown in Figure 5. RPD consistently achieves higher SSIM and lower LPIPS scores compared to Bicubic and Lanczos, indicating superior structural preservation and perceptual fidelity.

![](images/3f47867a7b28d10d260fdf0b8b872a18aa80ddbaf319bf2a3148d48caaf57c0e.jpg)  
Figure 5. Visual comparison with other pre-input Downsampling methods. Zoom in for better.

Visually, RPD outputs exhibit sharper details, reduced aliasing, and better texture retention, aligning more closely with human perception of the original image. For instance, in textured areas, RPD maintains fine-grained details that are blurred or lost in baselines. These qualitative comparisons demonstrate that RPD not only excels in objective metrics but also provides outputs with enhanced visual consistency to the originals. Additional visualizations are provided in the Appendix. E.

## 4.7. Ablation Studies

We conduct a series of ablation studies to validate the contribution. More ablation experiments can be found in the Appendix. D.

Impact of Low-Dimensional Perceptual Subspace Dimension: Following LoDa (Xu et al., 2024), we ablate the dimensionality of the perceptual subspace (24, 32, 48, 64, 128) while keeping other settings fixed. Performance is evaluated on KADID-10k and KonIQ-10k using SRCC and PLCC, as shown in Figure 6. The results reveal a nonmonotonic trend: performance peaks at 48 dimensions on both datasets, then declines as dimensionality increases toward the original CLIP embedding size (128). This inverted-U shape is strikingly consistent with observations in LoDa and provides strong empirical support for our hypothesis: perceptual information is most salient and separable in a compact low-dimensional subspace. Excessive dimensionality reintroduces semantic entanglement, diluting perceptual sensitivity. More experiments related to the sensitivity of perception in low-dimensional spaces can be found in the Appendix. A.

![](images/009688c6260e72dc06d12e2e63ee7bf34d07dd408b05d0ba3b8f3a50051ed473.jpg)

![](images/61170efee9d67967706d98064934f6f4a6f3f6a2b8a7f16538021f1e59f7cedc.jpg)  
Figure 6. Impact of the perceptual subspace dimensionality. Performance across two datasets exhibits a non-monotonic trend, peaking at 48 dimensions before declining as the dimension increases towards 128. This inverted-U shape confirms that a compact lowdimensional space effectively isolates perceptual information from semantic entanglement.

Table 5. Performance gain when applied to HyperIQA and MANIQA.
<table><tr><td rowspan="2">Method</td><td colspan="2">LIVEC</td><td colspan="2">KonIQ-10k</td><td colspan="2">SPAQ</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td>HyperIQA</td><td>0.859</td><td>0.882</td><td>0.906</td><td>0.917</td><td>0.911</td><td>0.915</td></tr><tr><td>+ RPD (Ours)</td><td>0.868</td><td>0.887</td><td>0.911</td><td>0.923</td><td>0.916</td><td>0.918</td></tr><tr><td>MANIQA</td><td>0.868</td><td>0.891</td><td>0.930</td><td>0.945</td><td>0.920</td><td>0.923</td></tr><tr><td>+ RPD (Ours)</td><td>0.870</td><td>0.901</td><td>0.932</td><td>0.946</td><td>0.922</td><td>0.927</td></tr></table>

RPD Framework Generality Ablation : To verify the generality of RPD beyond our CMPA framework, we apply it as a plug-and-play preprocessing step to two representative baselines: HyperIQA (CNN-based) and MANIQA (Transformer-based). Results are shown in Table 5. RPD consistently improves both baselines across all datasets. This framework-agnostic enhancement confirms that RPD’s residual-enhanced perceptual downsampling is not tied to our specific CMPA architecture but provides general perceptual fidelity benefits when used as a preprocessing module. More reliability experiments related to RPD can be found in the Appendix. F.

## 5. Conclusion

In this paper, we address the discrepancy between CLIP’s semantic invariance and the perceptual requirements of NR-IQA via the Cross-modal Perception Alignment Adapter (CMPA), which mitigates perceptual submergence by decoupling quality cues through low-dimensional manifold projection. This framework is further supported by Residual-enhanced Perceptual Downscaling (RPD) to preserve high-frequency fidelity during mandatory resizing to get perception-consistent input. Extensive benchmarks demonstrate that our approach offers a competitive, parameter-efficient solution, highlighting that manifoldaware decoupling and input fidelity are essential for adapting foundation models to fine-grained perceptual tasks. This work represents a meaningful step toward more robust and perception-aware VLM adaptation.

## 6. Additional Rebuttal Response

## 6.1. More Detailed Component Ablation Experiments

To further disentangle the source of performance gains, we conduct a fine-grained ablation on the bottleneck structure, cross-modal interaction, and attention design. As shown in Table 6, image-only tuning already provides the major improvement over the frozen CLIP baseline, while independently tuning both text and image branches brings almost no additional gain. In contrast, introducing the proposed Perceptual Feature Extraction (PFE) module with a shared lowdimensional subspace significantly improves performance, and the proposed Perceptual Attention Interaction (PAI) further provides consistent gains through explicit bidirectional cross-modal alignment. These results demonstrate that the improvements originate from the combination of compact perceptual subspace learning and active cross-modal guidance, rather than from generic bottleneck regularization alone.

Table 6. Detailed ablation of CMPA components on KADID-10k.
<table><tr><td>Method</td><td>SRCC</td><td>PLCC</td></tr><tr><td>Frozen CLIP</td><td>0.656</td><td>0.671</td></tr><tr><td>+ Image-only Adapter (high-d)</td><td>0.929</td><td>0.932</td></tr><tr><td>+ Image-only Adapter (low-d)</td><td>0.928</td><td>0.930</td></tr><tr><td>+ Independent Text+Image Tuning</td><td>0.928</td><td>0.931</td></tr><tr><td>+ PFE (shared MLP, high-d)</td><td>0.912</td><td>0.916</td></tr><tr><td>+ PFE (shared MLP, low-d)</td><td>0.936</td><td>0.938</td></tr><tr><td>+ PAI (text→image)</td><td>0.938</td><td>0.941</td></tr><tr><td>+ PAI (Full CMPA)</td><td>0.938</td><td>0.941</td></tr></table>

## 6.2. Hyperparameter Sensitivity Analysis

We further analyze the sensitivity of the RPD hyperparameters $( \alpha , \beta , \eta )$ by varying each parameter within {0.5, 0.8, 1.0, 1.2, 1.5} while fixing the remaining two. As shown in Table 7, the performance remains highly stable across a broad range, demonstrating that RPD does not rely on delicate hyperparameter tuning.

Table 7. Hyperparameter sensitivity analysis of RPD on KonIQ-10k (SRCC/SSIM).
<table><tr><td>Value</td><td>α</td><td> $\beta$ </td><td>η</td></tr><tr><td>0.5</td><td>0.932/0.860</td><td>0.934/0.851</td><td>0.930/0.859</td></tr><tr><td>0.8</td><td>0.934/0.874</td><td>0.935/0.869</td><td>0.933/0.865</td></tr><tr><td>1.0</td><td>0.938/0.871</td><td>0.938/0.871</td><td>0.938/0.871</td></tr><tr><td>1.2</td><td>0.935/0.857</td><td>0.936/0.861</td><td>0.935/0.851</td></tr><tr><td>1.5</td><td>0.930/0.865</td><td>0.931/0.832</td><td>0.929/0.821</td></tr></table>

## 6.3. More Detailed Analysis of Time Consumption

We provide a detailed efficiency analysis including sampling latency, inference time, GPU memory footprint, GFLOPs, and parameter count in Table 8. The proposed CMPA introduces only 1.8M trainable parameters (< 2% of CLIP ViT-B), while RPD itself is entirely parameter-free. Although RPD mathematically requires two interpolation passes, the practical overhead remains extremely small compared with the backbone computation. Specifically, RPD increases sampling latency from only 0.005s to 0.011s, while inference latency remains nearly unchanged (1.08 ms/image). These results demonstrate that the proposed framework preserves strong practicality for large-scale training and real-time deployment.

Table 8. Detailed efficiency comparison (batch size = 128).
<table><tr><td>Method</td><td>Samp. (s)</td><td>Infer. (ms/img)</td><td>Mem. (MB)</td><td>GFLOPs</td><td>Total Params</td><td>Train. Params</td></tr><tr><td>CLIP (Resize)</td><td>0.005</td><td>0.99</td><td>1652</td><td>27.29</td><td>151.5M</td><td></td></tr><tr><td>CMPA only</td><td>0.005</td><td>1.08</td><td>2192</td><td>27.46</td><td>153.3M</td><td>1.8M</td></tr><tr><td>RPD + CMPA</td><td>0.011</td><td>1.08</td><td>2192</td><td>27.46</td><td>153.3M</td><td>1.8M</td></tr></table>

## 6.4. Transferability and CMPA Non-Regularizer Analysis

To validate the generalizability of CMPA, we extend experiments to additional VLM backbones including BLIP, SigLIP, and ALIGN. As shown in Table 9, CMPA consistently achieves strong performance on BLIP and SigLIP, demonstrating that the proposed low-dimensional perceptual decoupling generalizes well across modern high-quality multimodal encoders. Interestingly, the gains on ALIGN are significantly weaker. We argue that this phenomenon provides evidence that CMPA is not merely a generic regularizer. BLIP and SigLIP are trained on carefully filtered or curated data, resulting in well-structured latent spaces where coherent perceptual manifolds exist and can be effectively isolated by CMPA. In contrast, ALIGN is trained on extremely noisy web-scale image-text pairs, causing perceptual cues to be heavily corrupted. Consequently, CMPA cannot effectively recover a stable perceptual manifold from such noisy representations. This contrast strongly supports our core hypothesis that CMPA specifically unlocks structured perceptual representations rather than acting as a generic capacity-reduction module.

Table 9. Transferability of CMPA across different VLM backbones (SRCC).
<table><tr><td>Method</td><td>KADID-10k</td><td>KonIQ-10k</td><td>LIVEC</td></tr><tr><td>ALIGN</td><td>0.910</td><td>0.907</td><td>0.838</td></tr><tr><td>BLIP</td><td>0.936</td><td>0.936</td><td>0.895</td></tr><tr><td>SigLIP</td><td>0.933</td><td>0.937</td><td>0.897</td></tr></table>

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Agnolucci, L., Galteri, L., and Bertini, M. Quality-aware image-text alignment for opinion-unaware image quality assessment. arXiv preprint arXiv:2403.11176, 2024.

Arslan, S. S., Vogelsang, L., Fux, M., and Sinha, P. Uniform resampling vs. image blur: Aliasing approximation via isotropic gaussian filtering. arXiv preprint arXiv:2502.11605, 2025.

Bengio, Y., Courville, A., and Vincent, P. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8): 1798–1828, 2013.

Chen, C. and Mo, J. IQA-PyTorch: Pytorch toolbox for image quality assessment. [Online]. Available: https:// github.com/chaofengc/IQA-PyTorch, 2022.

Chen, S., Ge, C., Tong, Z., Wang, J., Song, Y., Wang, J., and Luo, P. Adaptformer: Adapting vision transformers for scalable visual recognition. Advances in Neural Information Processing Systems, 35:16664–16678, 2022.

Chou, C.-H. and Li, Y.-C. A perceptually tuned subband image coder based on the measure of just-noticeabledistortion profile. IEEE Transactions on circuits and systemsfor video technology, 5(6):467–476, 1995.

Dehghani, M., Djolonga, J., Mustafa, B., Padlewski, P., Heek, J., Gilmer, J., Steiner, A. P., Caron, M., Geirhos, R., Alabdulmohsin, I., et al. Scaling vision transformers to 22 billion parameters. In International conference on machine learning, pp. 7480–7512. PMLR, 2023.

Fang, Y., Zhu, H., Zeng, Y., Ma, K., and Wang, Z. Perceptual quality assessment of smartphone photography. In In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3674–3683, 2020.

Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., and Qiao, Y. Clip-adapter: Better vision-language models with feature adapters. Int. J. Comput. Vis., 2024.

Ghadiyaram, D. and Bovik, A. C. Massive online crowdsourced study of subjective and objective picture quality. IEEE Transactions on Image Processing, 25(1):372–387, 2015.

Guan, F., Li, X., Yu, Z., Lu, Y., and Chen, Z. Q-mamba: On first exploration of vision mamba for image quality assessment. arXiv preprint arXiv:2406.09546, 2024.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

Hosu, V., Lin, H., Sziranyi, T., and Saupe, D. Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Transactions on Image Processing, 29:4041–4056, 2020.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Jie, S. and Deng, Z.-H. Convolutional bypasses are better vision transformer adapters. arXiv preprint arXiv:2207.07039, 2022.

Jie, S. and Deng, Z.-H. Fact: Factor-tuning for lightweight adaptation on vision transformer. In Proceedings ofthe AAAI conference on artificial intelligence, volume 37, pp. 1060–1068, 2023.

Ke, J., Wang, Q., Wang, Y., Milanfar, P., and Yang, F. Musiq: Multi-scale image quality transformer. In Proceedings ofthe IEEE/CVF international conference on computer vision, pp. 5148–5157, 2021.

Larson, E. C. and Chandler, D. M. Most apparent distortion: full-reference image quality assessment and the role of strategy. Journal of Electronic Imaging, 19(1):011006, 2010.

Li, D., Jiang, T., and Jiang, M. Norm-in-norm loss with faster convergence and better performance for image quality assessment. In Proceedings of the 28th ACM International conference on multimedia, pp. 789–797, 2020.

Li, X., Huang, Z., Zhang, Y., Shen, Y., Li, K., Zheng, X., Cao, L., and Ji, R. Few-shot image quality assessment via adaptation of vision-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 10442–10452, 2025.

Li, Y., Mi, Y., Zhou, P., Wu, Y., Wang, X., and Liu, S. Unleashing vision transformer potential in image quality assessment via global-local adaptive interaction. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 10567–10571. IEEE, 2026.

Liang, Y., Garg, B., Rosin, P., and Qin, Y. Deep generative model based rate-distortion for image downscaling

assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19363–19372, 2024.

Liao, Z., Wu, D., Shi, Z., Mai, S., Zhu, H., Zhu, L., Jiang, Y., and Chen, B. Beyond cosine similarity magnitude-aware clip for no-reference image quality assessment. arXiv preprint arXiv:2511.09948, 2025.

Lin, H., Hosu, V., and Saupe, D. Kadid-10k: A largescale artificially distorted iqa database. In 2019 Eleventh International Conference on Quality ofMultimedia Experience (QoMEX), 2019.

Liu, Y., Quan, Y., Xiao, G., Li, A., and Wu, J. Scaling and masking: A new paradigm of data sampling for image and video quality assessment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 3792–3801, 2024.

Ma, K., Wu, Q., Wang, Z., Duanmu, Z., Yong, H., Li, H., and Zhang, L. Group mad competition-a new methodology to compare objective image quality models. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1664–1673, 2016.

Madhusudana, P. C., Birkbeck, N., Wang, Y., Adsumilli, B., and Bovik, A. C. Image quality assessment using contrastive learning. IEEE Transactions on Image Processing, 31:4149–4161, 2022.

Mi, Y., Shu, Y., Li, Y., Hui, C., Zhou, P., and Liu, S. Clifvqa: Enhancing video quality assessment by incorporating high-level semantic information related to human feelings. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pp. 9989–9998, 2024.

Mi, Y., Li, Y., Meng, W., Chen, C., Hui, C., and Liu, S. Mvqa: Mamba with unified sampling for efficient video quality assessment. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18498– 18509, 2025.

Mittal, A., Moorthy, A. K., and Bovik, A. C. No-reference image quality assessment in the spatial domain. IEEE Transactions on image processing, 21(12):4695–4708, 2012a.

Mittal, A., Soundararajan, R., and Bovik, A. C. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012b.

Parmar, G., Zhang, R., and Zhu, J.-Y. On aliased resizing and surprising subtleties in gan evaluation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 11410–11420, 2022.

Pham, H., Dai, Z., Xie, Q., and Le, Q. V. Meta pseudo labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11557–11568, 2021.

Ponomarenko, N., Jin, L., Ieremeiev, O., Lukin, V., Egiazarian, K., and Astola, J. Image database tid2013: Peculiarities, results and perspectives. Signal Processing: Image Communication, 30:57–77, 2015.

Qin, G., Hu, R., Liu, Y., Zheng, X., Liu, H., Li, X., and Zhang, Y. Data-efficient image quality assessment with attention-panel decoder. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 2091– 2100, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Saad, M. A., Bovik, A. C., and Charrier, C. Blind image quality assessment: A natural scene statistics approach in the dct domain. IEEE Transactions on Image Processing, 21(8):3339–3352, 2012.

Saha, A., Mishra, S., and Bovik, A. C. Re-iqa: Unsupervised learning for image quality assessment in the wild. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5846–5855, 2023.

Sheikh, H. R., Sabir, M. F., and Bovik, A. C. A statistical evaluation of recent full reference image quality assessment algorithms. IEEE Transactions on Image Processing, 15(11):3440–3451, 2006.

Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, X., Sun, J., and Zhang, Y. Blindly assess image quality in the wild guided by a self-adaptive hyper network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3667–3676, 2020.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Wang, J., Chan, K. C., and Loy, C. C. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 2555–2563, 2023.

Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., and Change Loy, C. Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops, pp. 0–0, 2018.

Wu, H., Chen, C., Hou, J., Liao, L., Wang, A., Sun, W., Yan, Q., and Lin, W. Fast-vqa: Efficient end-to-end video quality assessment with fragment sampling. In European conference on computer vision, pp. 538–554. Springer, 2022.

Xu, K., Liao, L., Xiao, J., Chen, C., Wu, H., Yan, Q., and Lin, W. Boosting image quality assessment through efficient transformer adaptation with local feature enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2662– 2672, 2024.

Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., and Yang, Y. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1191–1200, 2022.

Yang, X., Ling, W., Lu, Z., Ong, E. P., and Yao, S. Just noticeable distortion model and its applications in video coding. Signal processing: Image communication, 20(7): 662–680, 2005.

Ying, Z., Mandal, M., Ghadiyaram, D., and Bovik, A. Patchvq:’patching up’the video quality problem. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14019–14029, 2021.

Zhang, L., Zhang, L., and Bovik, A. C. A feature-enriched completely blind image quality evaluator. IEEE Transactions on Image Processing, 24(8):2579–2591, 2015.

Zhang, W., Ma, K., Yan, J., Deng, D., and Wang, Z. Blind image quality assessment using a deep bilinear convolutional neural network. IEEE Transactions on Circuits and Systemsfor Video Technology, 30(1):36–47, 2020.

Zhang, W., Zhai, G., Wei, Y., Yang, X., and Ma, K. Blind image quality assessment via vision-language correspondence: A multitask learning perspective. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14071–14081, 2023.

Zhao, K., Yuan, K., Sun, M., Li, M., and Wen, X. Qualityaware pre-trained models for blind image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22302– 22313, 2023.

Zhou, K., Yang, J., Loy, C. C., and Liu, Z. Learning to prompt for vision-language models. International Journal ofComputer Vision, 130(9):2337–2349, 2022.

Zhu, H., Li, L., Wu, J., Dong, W., and Shi, G. Metaiqa: Deep meta-learning for no-reference image quality assessment. In In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14131–14140, 2020.

## A. Statistical Validation of the Subspace Distortion Amplification Effect

## A.1. Dimensionality Reduction Toy Experiment

One might hypothesize that the performance gain comes from reduced overfitting due to fewer parameters. However, our RDM analysis (Figure. 4a) reveals that the correlation with human perception increases significantly in the subspace, confirming that the gain stems from semantic disentanglement rather than mere regularization. To further validate our hypothesis of a low-dimensional perceptual subspace within CLIP embeddings, we perform an additional dimensionality reduction study. Features are extracted using a frozen CLIP image encoder. PCA is applied to project these into a lower dimensional subspace, after which the reduced features are interpolated back to the original dimensionality. The residual between this interpolated representation and the original features constitutes our perceptual feature, consistent with the residual injection strategy in CMPA.

We conduct a controlled toy experiment comparing distortion sensitivity between the original high-dimensional features and the residual subspace. We sample distorted images from the KADID-10k (Lin et al., 2019). Deviation intensity (normalized displacement from pristine) and Signal-to-Noise Ratio (SNR) are computed for both representations across distortion levels.

As shown in Figure 7, the residual subspace consistently exhibits higher distortion sensitivity, with SNR gains up to 1.59× at subtle distortion levels and ranging from 1.24× to 1.51× overall. Box plots of deviation intensity further reveal amplified responses, particularly to mild distortions. These findings demonstrate that linear dimensionality reduction can isolate and enhance perceptually salient components, providing strong evidence for the viability of our low-dimensional perceptua subspace hypothesis and its potential utility in perceptual alignment tasks.

![](images/6255bd90c2120d77df105675225604f63a6a1bb428140612ce86342454a91f3c.jpg)  
Figure 7. Distortion sensitivity analysis in high-dimensional vs. low-dimensional spaces. Our residual subspace consistently yields higher Signal-to-Noise Ratio (SNR) and amplified deviation intensity across all distortion levels compared to the original CLIP space. The significant sensitivity gain (up to 1.59×) at subtle distortions validates that a compact subspace effectively isolates and enhances perceptually salient information.

## B. Optimization-inspired Theoretical Analysis of RPD

We present a heuristic framework viewing the proposed downsampling method as a first-order approximation to a perceptually guided convex optimization problem, capturing the essence of Human Visual System (HVS), driven error minimization while opening doors for further refinement.

## B.1. Problem Formulation

Le $\mathbf { \Psi } : \mathbf { x } \in \mathbb { R } ^ { N }$ be the high-resolution input and $\mathbf { y } \in \mathbb { R } ^ { M } ( M < N )$ the target low-resolution output. With linear downsampling $\mathcal { D }$ and upsampling U, perceptual downsampling seeks $\mathbf { y } ^ { * }$ minimizing the HVS-weighted reconstruction error:

$$
\operatorname* { m i n } _ { \mathbf { y } } { f ( \mathbf { y } ) } = \frac { 1 } { 2 } \| \mathbf { W } ( \mathcal { U } \mathbf { y } - \mathbf { x } ) \| _ { 2 } ^ { 2 }\tag{10}
$$

where $\mathbf { W } = \mathrm { d i a g } ( w _ { i } )$ encodes NSS-derived JND weights reflecting local luminance adaptation and texture masking.

## B.2. Adaptive Gain via Weber-Fechner Law

The Weber-Fechner law describes the relationship between the physical intensity of a stimulus I and its perceived magnitude S. In its classical form, it states that the perceived change in sensation is proportional to the relative change in stimulus intensity:

$$
\Delta S = k \cdot { \frac { \Delta I } { I } } \quad { \mathrm { o r , i n ~ i n t e g r a t e d ~ f o r m , } } \quad S = k \ln I + C ,\tag{11}
$$

where k is a constant and $C$ is an integration constant. This implies that human perception follows a logarithmic response: larger absolute changes are required to produce the same perceived difference at higher stimulus levels (diminishing marginal sensitivity).

In the context of perceptual downsampling, we introduce an adaptive gain G to compensate for information loss while respecting this perceptual nonlinearity. When the downsampling ratio $d = N / M$ increases, more high-frequency content is discarded, but human observers do not perceive the loss linearly. Instead, the need for compensation grows sub-linearly with $d .$

We therefore define the gain as

$$
G ( d , \mathbf { W } ) = \mathbf { 1 } + \eta \cdot \log _ { 2 } \left( 1 + \left( 1 - \frac { 1 } { d } \right) \right) \cdot \mathbf { W } ,\tag{12}
$$

where $\eta > 0$ is a scaling factor controlling the strength of enhancement, and W incorporates local JND-based weights. The logarithmic term $\log _ { 2 } ( 1 + ( 1 - 1 / d ) )$ reflects Weber-Fechner’s principle: as d grows larger, additional compensation yields progressively smaller perceptual benefit, naturally preventing over-sharpening or halo artifacts in regions where further enhancement would be less noticeable.

This formulation ensures that the residual injection is perceptually economical — stronger enhancement is applied only where it is most likely to be noticed (guided by W) and scales sub-linearly with the degree of downsampling, aligning the method with human visual sensitivity.

## B.3. First-Order Approximation to the Optimum

The convex objective (1) has gradient $\nabla _ { \mathbf y } f = \mathcal { U } ^ { \top } \mathbf W ^ { 2 } ( \mathcal { U } \mathbf y - \mathbf x )$ . Starting from $\mathbf { y } _ { 0 } = \mathcal { D } \mathbf { x }$ , one-step gradient descent yields:

$$
\begin{array} { r } { \mathbf { y } _ { 1 } = \mathcal { D } \mathbf { x } + \alpha \mathcal { U } ^ { \top } \mathbf { W } ^ { 2 } ( \mathbf { x } - \mathcal { U } \mathcal { D } \mathbf { x } ) } \end{array}\tag{13}
$$

Approximating $x \mathbf { W } ^ { 2 } \approx G ( d , \mathbf { W } )$ (with $\mathbf { W } ^ { 2 } \approx c \mathbf { W }$ under small/normalized W) and $\boldsymbol { \mathcal { U } } ^ { \top } \approx \boldsymbol { \mathcal { D } }$ , we obtain the practical form:

$$
\mathbf { y } _ { \mathrm { f i n a l } } \approx \mathcal { D } \mathbf { x } + G ( d , \mathbf { W } ) \odot \mathcal { D } ( \mathbf { R } )\tag{14}
$$

where $\mathbf { R } = \mathbf { x } - \mathcal { U } \mathcal { D } \mathbf { x }$ is the lost high-frequency residual. This closed-form residual injection approximates perceptual error minimization, with bounded approximation error $O ( \alpha ^ { 2 } \| \mathbf { R } \| ^ { 2 } )$ under mild conditions.

## B.4. Information-Theoretic Insight

Standard downsampling discards frequencies above the Nyquist limit. The residual enhancement selectively projects perceptually salient high frequencies back into the low-resolution domain, guided by JND thresholds. This maximizes perceptual mutual information $I _ { \mathrm { H V S } } ( \mathbf { x } ; \mathbf { y } )$ under the resolution constraint, echoing perceptual losses in modern superresolution (e.g., ESRGAN (Wang et al., 2018)) and inspiring extensions to deep learning, compression, and content-aware processing.

This heuristic framework bridges classical optimization and HVS principles, offering a foundation for perceptually intelligent image algorithms with room for rigorous refinement and broader impact.

## C. Additional implementation details

More Dataset Information : For the synthetic datasets, they contain a few pristine im-ages that are synthetically distorted by various distortion types. LIVE (Sheikh et al., 2006) contains 779 synthetically distorted images with 5 distortion types. The CSIQ (Larson & Chandler, 2010) dataset is a synthetic IQA benchmark containing 30 reference images and 866 distorted images with various distortion types and subjective quality scores. TID2013 (Ponomarenko et al., 2015)andKADID-10k (Lin et al., 2019) consist of 3,000 and 10,125 synthetically distorted images involving 24 and 25 distortion types, respectively. For the authentic datasets, LIVEC (Ghadiyaram & Bovik, 2015) consists of 1,162 images with diverse authentic distortions captured by mobile devices. KonIQ10k (Hosu et al., 2020) contains 10,073 images which are selected from YFCC100M and the selected images cover a wide and uniform range of distortions such as brightness colorfulness, contrast, noise, sharpness, etc. SPAQ (Fang et al., 2020) consists of 11,125 images captured by different mobile devices, covering a large variety of scene categories. FLIVE (Ying et al., 2021) is the largest in-the-wild IQA dataset by far, which contains 39,810 real-world images with diverse contents, sizes, and aspect ratios.

Training and Evaluation Specifics: To ensure the sampling process captures comprehensive image information, random horizontal and vertical flips are applied during training. We adopt a standard 80/20 train-test split ratio. To minimize performance bias and ensure statistical robustness, each experiment is repeated 10 times, and the median Spearman Rank-Order Correlation Coefficient (SRCC) and Pearson Linear Correlation Coefficient (PLCC) are reported. All models are trained and evaluated on a single NVIDIA RTX 4090 GPU. $\alpha , \beta$ are all set to 1.0 in the experiments. The σ of the Gaussian kernel separated in the frequency domain is inspired by (Arslan et al., 2025) and designed to be 0.5 times the scaling ratio d. Loss Function: For the loss function, we use Pearson Linear Correlation Coefficient (PLCC) loss (Li et al., 2020), which can be formally expressed as

$$
L _ { \mathrm { P L C C } } = 1 - \frac { \sum _ { i = 1 } ^ { m } ( \tilde { y } _ { i } - \tilde { a } ) ( y _ { i } - a ) } { \sqrt { \sum _ { i = 1 } ^ { m } ( \tilde { y } _ { i } - \tilde { a } ) ^ { 2 } \sum _ { i = 1 } ^ { m } ( y _ { i } - a ) ^ { 2 } } } )\tag{15}
$$

where $\tilde { y } _ { i }$ is the predicted quality score for the i-th image, $y _ { i }$ is the true quality score for the i-th image, $\begin{array} { r } { \tilde { a } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } } \end{array}$ y˜ and $\textstyle a = { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } y _ { i }$ are the mean values of the predicted and true quality scores, respectively, m is the number of images in the training batch.

## D. Additional ablation experiments

## D.1. Study on the scale of pre-trained CLIP

We further examine how the pre-training scale of the CLIP backbone affects CMPA performance. We compare three ViT variants: ViT-L/14, ViT-B/16, and ViT-B/32. Performance scales with backbone capacity, with ViT-L/14 achieving the

Table 10. Impact of large-scale pretrained model sizes.
<table><tr><td rowspan="2">Backbone</td><td colspan="2">KADID-10k</td><td colspan="2">KonIQ-10k</td><td colspan="2">SPAQ</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td>ViT-L-14</td><td>0.951</td><td>0.953</td><td>0.936</td><td>0.945</td><td>0.929</td><td>0.932</td></tr><tr><td>ViT-B-16</td><td>0.934</td><td>0.935</td><td>0.934</td><td>0.943</td><td>0.925</td><td>0.929</td></tr><tr><td>ViT-B-32</td><td>0.938</td><td>0.941</td><td>0.935</td><td>0.945</td><td>0.927</td><td>0.931</td></tr></table>

strongest results on synthetic distortions (KADID-10k) and remaining highly competitive on authentic datasets. Notably, ViT-B/16 yields near-top performance across the board, demonstrating that our fine-tuning approach efficiently harnesses mid-scale pre-trained representations without excessive computational cost. These findings validate the effectiveness of CMPA and its ability to benefit from richer pre-training in perceptual quality assessment tasks.

## D.2. Performance on Individual Distortion Types

To verify that our Residual-Enhanced Perceptual Downsampling (RPD) does not cause over-enhancement for specific distortions, we evaluate per-distortion performance on CSIQ and LIVE datasets using SRCC and PLCC. Results are shown in Table 11. Our method outperforms MANIQA on most distortion types across both datasets (e.g., superior on JPEG, WN, GB in both; JP2K in CSIQ and LIVE). This demonstrates the effectiveness and broad robustness of RPD.

On Additive Pink Gaussian Noise (FN) in CSIQ, our performance is slightly lower than MANIQA. This is expected: RPD injects high-frequency residual compensation, which partially offsets the natural high-frequency attenuation of FN, which emphasizes more low-frequency structures, while the high-frequency components are relatively suppressed. However, JND-guided weighting limits the influence of these residuals, resulting in only negligible overall difference.

These results confirm that RPD enhances perceptual sensitivity without introducing systematic bias or significant degradation for any major distortion type.

Table 11. Performance comparison on LIVE and CSIQ datasets on Individual Distortion Types (PLCC).
<table><tr><td rowspan="2">Dataset</td><td colspan="5">LIVE</td><td colspan="6">CSIQ</td></tr><tr><td>JP2K</td><td>JPEG</td><td>WN</td><td>GB</td><td>FF</td><td>WN</td><td>JPEG</td><td>JP2K</td><td>FN</td><td>GB</td><td>CC</td></tr><tr><td>BRISQUE</td><td>0.929</td><td>0.965</td><td>0.982</td><td>0.964</td><td>0.828</td><td>0.723</td><td>0.806</td><td>0.840</td><td>0.378</td><td>0.820</td><td>0.804</td></tr><tr><td>ILNIQE</td><td>0.894</td><td>0.941</td><td>0.981</td><td>0.915</td><td>0.833</td><td>0.850</td><td>0.899</td><td>0.906</td><td>0.874</td><td>0.858</td><td>0.501</td></tr><tr><td>HOSA</td><td>0.935</td><td>0.954</td><td>0.975</td><td>0.954</td><td>0.954</td><td>0.604</td><td>0.733</td><td>0.818</td><td>0.500</td><td>0.841</td><td>0.716</td></tr><tr><td>BIECON</td><td>0.952</td><td>0.974</td><td>0.980</td><td>0.956</td><td>0.923</td><td>0.902</td><td>0.942</td><td>0.954</td><td>0.884</td><td>0.946</td><td>0.523</td></tr><tr><td>WaDIQaM</td><td>0.942</td><td>0.953</td><td>0.982</td><td>0.938</td><td>0.923</td><td>0.974</td><td>0.853</td><td>0.947</td><td>0.882</td><td>0.976</td><td>0.923</td></tr><tr><td>PQR</td><td>0.953</td><td>0.965</td><td>0.981</td><td>0.944</td><td>0.921</td><td>0.915</td><td>0.934</td><td>0.955</td><td>0.926</td><td>0.921</td><td>0.837</td></tr><tr><td>HyperIQA</td><td>0.949</td><td>0.961</td><td>0.982</td><td>0.926</td><td>0.934</td><td>0.927</td><td>0.934</td><td>0.960</td><td>0.931</td><td>0.915</td><td>0.874</td></tr><tr><td>MANIQA</td><td>0.870</td><td>0.895</td><td>0.984</td><td>0.959</td><td>0.896</td><td>0.966</td><td>0.971</td><td>0.973</td><td>0.977</td><td>0.956</td><td>0.946</td></tr><tr><td>Ours</td><td>0.988</td><td>0.990</td><td>0.984</td><td>0.983</td><td>0.971</td><td>0.984</td><td>0.982</td><td>0.986</td><td>0.973</td><td>0.976</td><td>0.949</td></tr></table>

## D.3. Study on Sampling Time and Performance Trade-off

To quantify the additional overhead introduced by our Residual-Enhanced Perceptual Downsampling (RPD), we measure sampling time and perceptual quality on the KonIQ-10k dataset using SSIM and LPIPS (higher is better for SSIM, while lower is better for LPIPS). Results are reported in Table 12 for single-step RPD (Ours), iterative RPD (Ours iter), and baselines Lanczos and Bicubic

Table 12. Sampling time (seconds) , perceptual quality (SSIM / LPIPS) and IQA metrics (PLCC/SRCC) on KonIQ-10k. Best perceptual scores in bold.
<table><tr><td>Method</td><td>Sampling Time (s)</td><td>SSIM</td><td>LPIPS</td><td>SRCC</td><td>PLCC</td></tr><tr><td>Ours_iter</td><td>0.053</td><td>0.821</td><td>0.310</td><td>0.910</td><td>0.924</td></tr><tr><td>Ours</td><td>0.011</td><td>0.871</td><td>0.241</td><td>0.929</td><td>0.945</td></tr><tr><td>Lanczos</td><td>0.007</td><td>0.852</td><td>0.265</td><td>0.914</td><td>0.931</td></tr><tr><td>Bicubic</td><td>0.005</td><td>0.854</td><td>0.265</td><td>0.921</td><td>0.935</td></tr></table>

Our single-step RPD incurs a modest time overhead (∼2–3× over traditional interpolation) but delivers superior perceptual quality across both metrics. This pre-input sampling strategy adds negligible impact to model parameters or GFLOPs, as it occurs before the network forward pass. Iterative application of RPD (Ours iter) further increases time cost and degrades performance due to excessive high-frequency enhancement, which disrupts perceptual structure and information fidelity. This highlights a clear trade-off: RPD provides strong perceptual gains at low single-step cost, but benefits most from non-iterative use to avoid over-compensation.

## D.4. Full Component Ablation

Although Table 1 has some ablation experimental results, further explanations are given here in order to avoid ambiguity. Table 13 compares the performance of the original CLIP (zero-shot), CLIP with CMPA fine-tuning, and the full framework with both CMPA and RPD. Evaluations are performed on KADID-10k, KonIQ-10k, and SPAQ using SRCC and PLCC. CLIP alone exhibits limited perceptual alignment in zero-shot settings. Applying CMPA fine-tuning yields substantial improvements across all datasets, demonstrating the effectiveness of our perceptual subspace enhancement and cross-modal alignment. Adding RPD further boosts performance, particularly on authentic distortion benchmarks (e.g., KonIQ-10k SRCC from 0.935 to 0.938, SPAQ PLCC from 0.931 to 0.934). These incremental gains confirm that both CMPA and RPD contribute meaningfully, with RPD compensating for high-frequency information loss during downsampling.

Table 13. Full component ablation: Impact of CMPA and RPD on perceptual quality assessment.
<table><tr><td rowspan="2">Backbone</td><td colspan="2">KADID-10k</td><td colspan="2">KonIQ-10k</td><td colspan="2">SPAQ</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td>CLIP (Original)</td><td>0.656</td><td>0.671</td><td>0.625</td><td>0.727</td><td>0.738</td><td>0.735</td></tr><tr><td>+CMPA</td><td>0.937</td><td>0.940</td><td>0.935</td><td>0.945</td><td>0.927</td><td>0.931</td></tr><tr><td>+CMPA &amp; RPD</td><td>0.938</td><td>0.941</td><td>0.938</td><td>0.948</td><td>0.929</td><td>0.934</td></tr></table>

## D.5. Component Ablation of CMPA

To gain deeper insight into how our CMPA framework operates, we perform a detailed component ablation study on its key modules: text-only fine-tuning, image-only fine-tuning, dual text-image fine-tuning, addition of the Perceptual Feature Enhancement (PFE) module with shared MLP, and the full CMPA with Perceptual Alignment Interaction (PAI) module. Results are summarized in Table 14. Fine-tuning only the text encoder yields moderate gains over zero-shot CLIP, but

Table 14. Impact of text-only, image-only, PFE, and PAI modules.
<table><tr><td rowspan="2">Method</td><td colspan="2">KADID-10k</td><td colspan="2">KonIQ-10k</td><td colspan="2">SPAQ</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td>CLIP (Original)</td><td>0.656</td><td>0.671</td><td>0.625</td><td>0.727</td><td>0.738</td><td>0.735</td></tr><tr><td>+ Only Text</td><td>0.792</td><td>0.793</td><td>0.796</td><td>0.873</td><td>0.893</td><td>0.886</td></tr><tr><td>+ Only Image</td><td>0.928</td><td>0.930</td><td>0.928</td><td>0.933</td><td>0.914</td><td>0.923</td></tr><tr><td>+ Text &amp; Image</td><td>0.928</td><td>0.931</td><td>0.920</td><td>0.934</td><td>0.914</td><td>0.920</td></tr><tr><td>+ PFE (shared MLP)</td><td>0.936</td><td>0.938</td><td>0.933</td><td>0.945</td><td>0.925</td><td>0.927</td></tr><tr><td>+ PAI (Full CMPA)</td><td>0.938</td><td>0.941</td><td>0.935</td><td>0.945</td><td>0.927</td><td>0.931</td></tr></table>

performance remains limited. This is because the image features remain in the original high-dimensional space, where perceptual signals are diluted by semantic information, leading to low perceptual sensitivity. Image-only fine-tuning achieves results comparable to CLIP-Adapter, as both operate primarily on the visual branch without strong cross-modal guidance. Adding the PFE module (shared MLP with perception-aware text prompts) forces image features toward the perceptua subspace, delivering a clear performance boost (e.g., SRCC from 0.928 to 0.936 on KADID-10k). This confirms the effectiveness of explicit perceptual enhancement. Incorporating the PAI module provides further improvement through attention-based cross-modal interaction and alignment, yielding the best results across all datasets. The incremental gain from PAI validates our design: guided interaction in the low-dimensional perceptual subspace refines alignment and enhances perceptual discriminability.

These ablations demonstrate that the full CMPA framework—combining perception-aware text guidance, feature enhancement (PFE), and cross-modal interaction (PAI), is essential for achieving superior perceptual quality assessment performance.

## D.6. Component Ablation of RPD

We performed ablation on the components of the RPD sampling, mainly eliminating the adaptive perceptual JND weights based on the scaling coefficient. The specific results are shown in the Table 15. It can be seen that without the adaptive weighting supplement method, although it improves the information retention of the sampling to a certain extent and enhances the perceptual performance on the real dataset, the perceptual performance on synthetic datasets such as Tid2013 remains poor. This is probably due to the overfitting caused by directly fusing the lost high-frequency information. With the addition of JND, the effect has been improved to some extent. Moreover, with the addition of perceptual weights, the effect has become even better, which indicates the importance of integrating human visual system and perceptual laws.

Table 15. Result on component ablation of RPD.
<table><tr><td rowspan="2">Methods</td><td colspan="2">KonIQ</td><td colspan="2">Tid2013</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td>Bicubic</td><td>0.926</td><td>0.936</td><td>0.879</td><td>0.899</td></tr><tr><td>+ Residual information</td><td>0.932</td><td>0.943</td><td>0.870</td><td>0.879</td></tr><tr><td>+ JND weight map</td><td>0.933</td><td>0.945</td><td>0.883</td><td>0.907</td></tr><tr><td>+ Perceptual weight</td><td>0.938</td><td>0.948</td><td>0.899</td><td>0.911</td></tr></table>

![](images/ce09185f28d3574a9e5005132b1ef2b95ccc0b8a062b375242df198a1402e17c.jpg)  
(a)

![](images/48231b5f5cd7219f9d196326971f9245247ddb0800aaa44ef14f4a794226a1af.jpg)  
(b)

![](images/9022f8d5a87e00c24803da0a96e53bc4ea7bdbe154aa1038ef2c1acefbd50cc1.jpg)  
(c)

![](images/fe39a0a66aa0f3f0d2b5608ec2ca368a90ec24c40c2716b43d48eb42aa0ca52b.jpg)  
(d)  
Figure 8. Representative gMAD pairs between proposed methods and CLIPIQA+ on synthetic distortions. From left to right: Fixing Ours at the low-quality. Fixing Ours at the high-quality. Fixing CLIPIQA+ at the low-quality. Fixing CLIPIQA+ at the high-quality.

## E. More Qualitative experiments

## E.1. More Qualitative gMAD result

To illustrate perceptual superiority, we conduct a qualitative gMAD competition against CLIPIQA+. The results are shown in Figure 8. Our method consistently produces downsampled images that are more perceptually distinguishable from each other in pairwise human-like judgments, outperforming CLIPIQA+ by a clear margin. This qualitative evidence reinforces the robustness and effectiveness of our fine-tuning strategy in generating perceptually faithful representations.

To further illustrate the perceptual superiority of our Residual-Enhanced Perceptual Downsampling (RPD), we conducted a qualitative GMAD (Ma et al., 2016) competition between RPD and standard Bicubic downsampling. GMAD evaluates how well different methods produce images that are perceptually distinguishable by human observers.

The results are shown in Figure 9. Our method consistently ranks higher in the GMAD pairwise comparison, generating downsampled images that better preserve perceptual details and overall quality compared to Bicubic. These qualitative findings confirm the stronger perceptual fidelity of RPD.

Additionally, we visualize the Fourier spectrum of the pre-input downsampled images produced by RPD, Lanczos, Bicubic, and the original high-resolution image (averaged over 128 images from SPAQ). The results are presented in Figure 10.

As shown in the Fourier spectra and relative log amplitudes of the transformed feature maps, RPD downsampled images retain significantly more high-frequency information than traditional interpolation methods (Lanczos and Bicubic). Moreover, the relative log amplitude distribution of RPD is much closer to that of the original high-resolution image. This indicates that RPD effectively compensates for high-frequency loss during downsampling, resulting in a more faithful information distribution that aligns better with the original content. Together, these qualitative and frequency-domain analyses provide compelling evidence of the perceptual advantages and robustness of the proposed RPD method over conventional downsampling techniques.

![](images/46db56e31d42e4cb9da4505ef93300f25ea5fc997c34f8d3f05e2d0f33605fee.jpg)  
(a)

![](images/40a3a72e044e12c3a693cf3d1604b1e3e18ec1e0e93e5d67fe7421c9d5018198.jpg)  
(b)

![](images/2d26a362fc153fe69bc40a8b5db720e2cfaddeabb82cb2cd159846bbb36a9e74.jpg)  
(c)

![](images/95bc064777403f54302f8a84fc7344dde36c9c4bebfcfb578596e59a9a3712f1.jpg)  
(d)  
Figure 9. Representative gMAD pairs between proposed methods and Bicubic on synthetic distortions. From left to right: Fixing Ours at the low-quality. Fixing Ours at the high-quality. Fixing Bicubic at the low-quality. Fixing Bicubic at the high-quality.

## E.2. Qualitative Analysis of Saliency Maps

To qualitatively assess the perceptual focus of our CMPA framework, we visualize saliency maps for the original images, full-fine-tuned CLIP, CLIP-Adapter, our CMPA method, and CMPA with downsampling (using RPD). The saliency maps highlight regions of high activation, revealing where each method attends during perceptual quality assessment. Results are shown in Figure 11.

In the full-fine-tuned CLIP and CLIP-Adapter variants, saliency often concentrates on semantically dominant regions (e.g., central objects or high-contrast edges), with scattered activations that may overlook subtle perceptual distortions like texture degradation or noise. This suggests a bias toward semantic features rather than fine-grained perceptual details. In contrast, our CMPA method produces more focused and perceptually relevant saliency: activations are stronger in areas sensitive to human perception, such as textured surfaces, edges with potential aliasing, or regions prone to luminance variations. When combined with RPD downsampling, the saliency maps exhibit even greater refinement, with enhanced emphasis on high-frequency details that are typically lost in standard downsampling. This results in more accurate highlighting of perceptual artifacts (e.g., blurring in backgrounds or color shifts in complex scenes).

These qualitative patterns demonstrate that CMPA, especially with RPD, shifts attention toward perceptually salient features, aligning better with human visual sensitivity. The visualizations further validate the effectiveness of our low-dimensiona perceptual subspace guidance and residual enhancement in producing robust, perception-aware representations.

## F. Rationality analysis of RPD

## F.1. Generalization of RPD without Additional Data Augmentation

In-Dataset Evaluation: To further demonstrate the intrinsic perceptual preservation capability of our Residual-Enhanced Perceptual Downsampling (RPD) and rule out potential confounding effects from other data augmentation strategies, we conduct an ablation study using RPD alone for downsampling during training—without any additional augmentations such as random cropping or flipping. All patches are of size 1 (full-image level), and other training settings remain identical to the main experiments. Results are shown in Table 16.

Table 16. Performance with RPD-only downsampling (no random crop/flip) vs. traditional downscaling methods.
<table><tr><td rowspan="3">Method</td><td colspan="2">LIVE</td><td colspan="2">KonIQ-10k</td><td colspan="2">SPAQ</td><td colspan="2">FLIVE</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td>Bicubic</td><td>0.863</td><td>0.882</td><td>0.921</td><td>0.935</td><td>0.915</td><td>0.918</td><td>0.561</td><td>0.673</td></tr><tr><td>Lanczos</td><td>0.851</td><td>0.863</td><td>0.914</td><td>0.931</td><td>0.901</td><td>0.907</td><td>0.545</td><td>0.665</td></tr><tr><td>Ours (No-Aug)</td><td>0.877</td><td>0.893</td><td>0.929</td><td>0.945</td><td>0.923</td><td>0.928</td><td>0.585</td><td>0.685</td></tr></table>

![](images/36b097e0f11a330e638048751e123619c6722f40e08f830e3ee9c745fdff0b5a.jpg)

(a)  
![](images/1010dea65002083386a121ac2e634a0784c00a6ca275e7621f681dcd93bd69e1.jpg)  
(b)  
Figure 10. Fourier analysis was performed on the downsampled images obtained by the conventional interpolation and RPD methods. (a) Fourier spectrum of all methods. (b) Relative log amplitude of the Fourier transform feature map. (a) and (b) show that RPD captures more high-frequency signals while the frequency energy distribution is closer to the original image

Even without any conventional data augmentation to increase training diversity, our RPD-based training achieves competitive or near state-of-the-art performance across all datasets. This substantial improvement over Bicubic and Lanczos underscores that RPD’s perceptual preservation is a fundamental property of the downsampling process itself, rather than an artifact of augmentation-induced diversity.

Cross-Dataset Evaluation: To rule out the possibility that this strong performance stems from dataset overfitting rather than genuine perceptual learning, we further perform cross-dataset evaluation. We train on one dataset and test on another, still using RPD-only downsampling without augmentation. Results are reported in Table 17.

Table 17. Cross-dataset generalization (SRCC) without data augmentation.
<table><tr><td rowspan="2">Training Testing</td><td colspan="2">FLIVE</td><td>LIVEC</td><td>KonIQ</td></tr><tr><td>KonIQ</td><td>LIVEC</td><td>KonIQ</td><td>LIVEC</td></tr><tr><td>DBCNN</td><td>0.716</td><td>0.724</td><td>0.754</td><td>0.755</td></tr><tr><td>P2P-BM</td><td>0.755</td><td>0.738</td><td>0.740</td><td>0.770</td></tr><tr><td>HyperIQA</td><td>0.758</td><td>0.735</td><td>0.772</td><td>0.785</td></tr><tr><td>TReS</td><td>0.713</td><td>0.740</td><td>0.733</td><td>0.786</td></tr><tr><td>DEIQT</td><td>0.733</td><td>0.781</td><td>0.744</td><td>0.794</td></tr><tr><td>LoDa</td><td>0.763</td><td>0.805</td><td>0.745</td><td>0.811</td></tr><tr><td>Ours (No-Aug)</td><td>0.781</td><td>0.793</td><td>0.766</td><td>0.828</td></tr></table>

Despite the absence of augmentation-induced diversity, our method still achieves stronger cross-domain generalization than LoDa and remains competitive with augmentation-equipped baselines. This confirms that RPD enables the model to learn genuine perceptual representations rather than merely overfitting to training distribution artifacts. These ablation results highlight the robustness and effectiveness of RPD as a standalone perceptual downsampling strategy.

Origin Image  
Full-Finetune CLIP  
CLIP-Adapter  
Ours  
With RPD  
![](images/e2367220757b89e19cbe1ac45b803efc07e69e7352a1b3e25d63947fe3918f19.jpg)  
Figure 11. Visualization of Gram-Cam Map of models. Our CMPA enables the model’s attention to focus more intensively on perceptually relevant regions, while the RPD further refines and strengthens this perceptual emphasis.

## F.2. Perceptual Fidelity and Plug-and-Play Generality of RPD

To demonstrate that our Residual-Enhanced Perceptual Downsampling (RPD) better preserves perceptual quality during downsampling, we evaluate the upsampled results against the original high-resolution images using two full-reference metrics: SSIM and LPIPS (higher SSIM and lower LPIPS indicate better structural and perceptual fidelity).

Results are reported as medians in Table 18 across several widely used IQA datasets.

Table 18. Median SSIM and LPIPS after upsampling to original resolution. Bold indicates the top results.
<table><tr><td rowspan="2">Method</td><td colspan="2">LIVE</td><td colspan="2">CSIQ</td><td colspan="2">TID2013</td><td colspan="2">KADID-10k</td><td colspan="2">KonIQ-10k</td><td colspan="2">LIVEC</td><td colspan="2">SPAQ</td><td colspan="2">FLIVE</td></tr><tr><td>SSIM</td><td>LPIPS</td><td>SSIM</td><td>LPIPS</td><td>SSIM</td><td>LPIPS</td><td>SSIM</td><td>LPIPS</td><td>SSIM</td><td>LPIPS</td><td>SSIM</td><td>LPIPS</td><td>SSIM</td><td>LPIPS</td><td>SSIM</td><td>LPIPS</td></tr><tr><td>Bicubic</td><td>0.928</td><td>0.257</td><td>0.852</td><td>0.223</td><td>0.861</td><td>0.125</td><td>0.893</td><td>0.099</td><td>0.854</td><td>0.262</td><td>0.913</td><td>0.210</td><td>0.811</td><td>0.545</td><td>0.880</td><td>0.138</td></tr><tr><td>Lanczos</td><td>0.920</td><td>0.264</td><td>0.847</td><td>0.230</td><td>0.857</td><td>0.120</td><td>0.890</td><td>0.095</td><td>0.852</td><td>0.265</td><td>0.909</td><td>0.156</td><td>0.811</td><td>0.546</td><td>0.878</td><td>0.135</td></tr><tr><td>Ours</td><td>0.948</td><td>0.249</td><td>0.869</td><td>0.210</td><td>0.875</td><td>0.112</td><td>0.904</td><td>0.090</td><td>0.871</td><td>0.241</td><td>0.921</td><td>0.153</td><td>0.819</td><td>0.504</td><td>0.895</td><td>0.125</td></tr></table>

Compared to traditional Bicubic and Lanczos interpolation, RPD consistently achieves higher SSIM and lower LPIPS across all datasets. This indicates superior structural preservation and reduced perceptual distance to the original image, confirming the effectiveness of RPD in compensating for perceptual information lost during downsampling.

To further verify that RPD provides more perceptually faithful downsampled images, we evaluate the plug-and-play generality of RPD by applying it as a simple pre-processing step to two off-the-shelf no-reference IQA models (NIQE and MANIQA, from the pyiqa library (Chen & Mo, 2022)) in zero-shot settings. The results are shown in Table 19. Applying RPD as a plug-and-play preprocessing step consistently improves the zero-shot performance of both NIQE and MANIQA across all evaluated datasets. These gains demonstrate the strong generality of RPD: it enhances perceptual alignment for existing NR-IQA models without any retraining or fine-tuning, highlighting its ability to produce downsampled images that

are more perceptually representative of the original content.

Table 19. Zero-shot NR-IQA performance (SRCC / PLCC) with and without RPD pre-processing. Bold entries indicate the top results. The upper block reports results evaluated by NIQE, while the lower block corresponds to MANIQA.
<table><tr><td rowspan="2">Method</td><td colspan="2">LIVE</td><td colspan="2">CSIQ</td><td colspan="2">TID2013</td><td colspan="2">KADID-10k</td><td colspan="2">KonIQ-10k</td><td colspan="2">LIVEC</td><td colspan="2">SPAQ</td><td colspan="2">FLIVE</td></tr><tr><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC PLCC</td><td></td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC PLCC</td><td></td><td>SRCC PLCC</td><td></td></tr><tr><td colspan="9">NIQE-Zero-shot Evaluation</td><td colspan="9"></td></tr><tr><td>Bicubic</td><td>0.206</td><td>0.214</td><td>0.283</td><td>0.497</td><td>0.667</td><td>0.803</td><td>0.567</td><td>0.623</td><td>0.158</td><td>0.299</td><td>0.157</td><td>0.381</td><td>-0.306</td><td>-0.169 0.493</td><td></td><td>0.609</td></tr><tr><td>Lanczos</td><td>0.206</td><td>0.188</td><td>0.293 0.435</td><td>0.519 0.650</td><td>0.638</td><td>0.790</td><td>0.536</td><td>0.625</td><td>0.174</td><td>0.307</td><td>0.161</td><td>0.386</td><td>-0.303</td><td>-0.167</td><td>0.514</td><td>0.607</td></tr><tr><td>Ours</td><td colspan="4">0.397 0.398</td><td>0.662</td><td>0.791</td><td>0.527</td><td>0.529</td><td>0.183</td><td>0.331</td><td>0.263</td><td>0.437</td><td>0.004</td><td>0.067</td><td>0.507</td><td>0.605</td></tr><tr><td colspan="9">MANIQA-Zero-shot Evaluation</td><td colspan="4"></td><td colspan="4"></td></tr><tr><td>Bicubic</td><td>-0.025 -0.027 0.662</td><td></td><td></td><td>0.688</td><td>0.503</td><td>0.562</td><td>0.477</td><td>0.503</td><td>0.698</td><td>0.763</td><td>0.776</td><td>0.795</td><td>0.834</td><td>0.831</td><td>0.339</td><td>0.405</td></tr><tr><td>Lanczos</td><td>-0.025-0.0280.659</td><td></td><td></td><td>0.684</td><td>0.504</td><td>0.560</td><td>0.475</td><td>0.501</td><td>0.698</td><td>0.762</td><td>0.776</td><td>0.794</td><td>0.834</td><td>0.830</td><td>0.337</td><td>0.405</td></tr><tr><td>Ours</td><td>–0.012 –0.018 0.705</td><td></td><td></td><td>0.742</td><td>0.514</td><td>0.589</td><td>0.523</td><td>0.552</td><td>0.709</td><td>0.779</td><td>0.805</td><td>0.829</td><td>0.829</td><td>0.827</td><td>0.344</td><td>0.418</td></tr></table>

## F.3. More visual comparison results

Additional qualitative visualization results, as shown in Figure. 12, including more examples of downsampled images under various scenes and distortion types, are provided in the appendix for further reference.

![](images/c98ca69fa6b91e1ed1cbab0e980872d7f75dc868e17eb341d86c8481fcb9042b.jpg)  
Figure 12. More pre-input downsampling results for visual comparison.

## G. Discussion and Limitations

Our experiments and theoretical analysis provide initial evidence that low-dimensional subspaces within feature embeddings exhibit heightened perceptual sensitivity to distortions. This finding aligns qualitatively with prior observations in LoDa (Xu et al., 2024), yet we did not systematically search for the optimal perceptual subspace dimensionality. Instead, we explored a subset of subspace sizes. Determining the theoretically optimal or empirically best dimension remains an important direction for future investigation.

The proposed Residual-Enhanced Perceptual Downscaling (RPD) serves as an intuitive and lightweight compensation strategy for high-frequency information lost during downsampling. However, we acknowledge that RPD is not a universa solution for all resolution rescaling tasks. Specifically, this work primarily compares RPD against standard interpolation baselines and does not extensively evaluate it against modern learnable image resizers or deep-learning-based downsampling methods. This choice was motivated by our goal to maintain a plug-and-play preprocessor without introducing significant computational overhead or inference latency, which are common drawbacks of deep-learning-based scalers. Additionally, because RPD explicitly injects high-frequency residuals, it may introduce minor perceptual confusion on certain high frequency-attenuating distortions, such as Pink Gaussian Noise. Consequently, RPD should be viewed as a practical, first-order approximation rather than an optimal perceptually consistent downsampling method. In future work, we plan to investigate more sophisticated, perception-consistent downsampling architectures using lightweight deep learning techniques, aiming to achieve a better balance between input fidelity and computational efficiency in NR-IQA pipelines