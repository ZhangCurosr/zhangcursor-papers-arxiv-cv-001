# MambaMPD: A Mamba-Driven Segmentation Framework for Marine Pollution Detection from Remote Sensing Imagery

Shuaiyu Chen<sup>a</sup>,Wei Han<sup>b</sup>, Peng Ren<sup>c</sup>, Chunbo Luo<sup>a</sup> and Zeyu Fu<sup>a</sup>

<sup>a</sup>Department of Computer Science, University of Exeter, Exeter, United Kingdom; <sup>b</sup>School of Computer Science, China University of Geosciences, Wuhan, China; <sup>c</sup>College of Oceanography and Space Informatics, China University of Petroleum (East China), Qingdao, China

## ABSTRACT

Accurate detection of marine pollution is essential for protecting coastal ecosystems and marine biodiversity. Recently, vision Mamba-based approaches have shown promise in remote sensing semantic segmentation due to their ability to eficiently capture long-range dependencies and global context. However, their potential remains largely unexplored in the context of Marine Pollution Detection (MPD) with distinct challenges, including low signal-to-noise ratios, spatial fragmentation of pollution patterns, and indistinct boundaries caused by strong visual similarity between pollutants and the surrounding marine environment. To address these challenges, we propose MambaMPD, an enhanced Mamba-based framework tailored for marine pollution detection. MambaMPD incorporates two targeted modules that enhance the Mamba encoder with complementary structural priors: the Frequency-Aware Augmentation (FAA) module and the multi-scale Edge-Guided Attention (EGA) module. The FAA module strengthens the encoding process by integrating Wavelet Transforms, which decompose features into multi-scale frequency subbands. This enables the model to eficiently capture both low-frequency contextual semantics and high-frequency structural details, essential for identifying small, low-contrast, and irregular pollution patterns. Meanwhile, the EGA module adaptively integrates hierarchical Laplacian-derived multi-scale boundary cues into deep semantic representations, guiding the refinement of encoder features before decoding, thereby sharpening boundary delineation and alleviating ambiguity in visually confusing and spatially fragmented marine scenes. Additionally, a U-Net-style decoder equipped with squeeze-and-excitation attention and deep supervision is employed to progressively recover and refine semantic and spatial features across multiple scales. Extensive experiments on two benchmark marine pollution detection datasets demonstrate that the proposed method outperforms the compared methods in mIoU while maintaining significantly lower computational cost than foundation-model-based approaches. On MADOS it surpasses OSDMamba by 3.6% in F1, and on M4D it improves Oil Spill IoU by 6.82% over TransOilSeg. The source code of our approach will be available at https://github.com/Multimodal-Intelligence-Lab-MIL/MambaMPD.

## KEYWORDS

Marine pollution detection ; State space model ; Remote sensing imagery ; Wavelet transform ; Edge-guided attention

## 1. Introduction

Marine pollution monitoring is a highly demanding yet societally vital task. Pollutants such as oil spills [1], plastic debris [2], and chemical contaminants [3] pose serious threats to fragile coastal ecosystems and marine biodiversity, while also disrupting navigation and undermining the sustainability of the blue economy [4–6]. Recent advances in aerospace and sensor technologies have greatly improved access to remote sensing data, including synthetic aperture radar (SAR)[7] and multispectral imagery [8]. Accurately delineating polluted regions from satellite or aerial imagery is therefore crucial for both rapid emergency response and long-term environmental management [9–11], and contributes directly to the United Nations Sustainable Development Goals, in particular SDG 14, in line with the recognised role of remote sensing in advancing the SDGs [12].

Traditional marine pollution detection methods, such as thresholding [13–15], clustering [16, 17], and Markov Random Field-based segmentation [18–20] primarily rely on hand-crafted features like colour, texture, and spatial consistency. While computationally eficient, these approaches often lack robustness and generalisability in complex marine environments. Recent progress has seen a shift toward learningbased models, evolving from CNN-based methods [21, 22], to Transformer-based architectures [23–26], and more recently, foundation model-based approaches like SAM-OIL [27]. These developments reflect a transition from task-specific pipelines to prompt-driven, globally-aware segmentation frameworks. More recently, the Mamba architectures [28–30], based on state-space models (SSMs) [28], have gained attention for their ability to eficiently capture long-range dependencies and model global context. While Mamba-based models have shown promise in general remote sensing segmentation [31], their potential remains largely unexplored in the context of marine pollution detection.

Despite significant progress in learning-based methods for marine pollution detection, several fundamental challenges persist across both multispectral and SAR imagery, as illustrated in Fig.1. These challenges are not specific to any particular architecture but remain largely unresolved across existing CNN-based, Transformer-based, and state-space model-based approaches:

1)Pollution targets such as oil spills and plastic patches typically manifest as small, dispersed regions, often captured by medium-resolution sensors like Sentinel-2 and SAR platforms [8]. These modalities lack the spatial or textural granularity required to capture fine structures, making models highly sensitive to noise [32].

2)Pollutants (such as emulsified oils, rainbow sheens, and plastics) often share similar visual or backscatter patterns with non-pollutant features such as seawater, ship wakes, plankton, and algae [33, 34], leading to substantial semantic ambiguity and misclassification. [35].

3)Dynamic sea surface conditions, driven by waves, currents, and biological activity, further obscure the boundaries between pollution and background regions [36].

While Mamba-based architectures have demonstrated strong capability in capturing long-range dependencies with linear complexity, they still lack dedicated mechanisms to address the above challenges in the context of marine pollution detection. A recent work, OSDMamba [37], adapts Mamba to SAR oil-spill segmentation but retains a standard VSS encoder, and therefore ofers no mechanism for the high-frequency cues of small, low-contrast targets or for boundary preservation under speckle and dynamic sea states; it is moreover confined to binary, single-modality detection. To this end, we propose MambaMPD, a Mamba-driven segmentation framework tailored for marine pollution detection, as shown in Fig. 2 (a). Built upon the expressive capacity of VMamba [29], MambaMPD incorporates several targeted adaptations to address the unique challenges posed by marine pollutants across both SAR and multispectral imagery. A key component of MambaMPD is the Frequency-Aware Augmentation (FAA) Module, which mainly integrates Wavelet Transforms into the Mamba encoder to decompose features into multi-scale frequency subbands. This allows the model to eficiently capture both contextual semantics and fine structural details, enhancing its ability to detect small, low-contrast, and irregular marine pollution targets.

![](images/bac35fb953649eb99dad4000e0e3bbd085cabda8b35d6d2ff0531d8d4ff950cd.jpg)

![](images/2a25377bebdb23c37b784e54662ddc6b54ffa1332298743546b18bb17c03cc73.jpg)

![](images/b62c01302e58d892ed18393c3736a85a6451aa7f2c92e496e5d2189d86365a88.jpg)

![](images/d4c33cf5c26218af58e5647724fef81d5da81fea815a6e1bae3476e0bbf9eac1.jpg)  
(a)

![](images/257b550d005043945cf401cc1d31e0b93fc407a05f948f944cf69d12e9658b1f.jpg)  
(b)

![](images/fad0ba3d902a7f6a4c19c293eea5e48b4ec0dd0af892376ea321dc828f50461e.jpg)  
(c)  
Figure 1. Illustration of key challenges in marine pollution detection across SAR imagery (top row, from M4D dataset) and multispectral imagery (bottom row, from MADOS dataset). (a) Small and dispersed pollution targets (highlighted by red boxes) that are easily obscured by background clutter and noise, making finegrained detection dificult. (b) Visual and backscatter ambiguity, where oil spills, look-alike dark spots, and natural marine features (e.g., algae, turbid water) exhibit highly similar appearances, leading to frequent misclassification. (c) Indistinct and fragmented pollution boundaries caused by speckle noise in SAR and dynamic sea surface conditions in optical imagery, posing challenges for precise boundary delineation.

Another key component of MambaMPD is the multi-scale Edge-Guided Attention (EGA) module, which adaptively integrates Laplacian-based edge cues with semantic features extracted at multiple encoder depths. EGA is specifically designed to enhance boundary preservation under conditions of spatial sparsity. Additionally, a U-Net-style decoder equipped with Squeeze-and-Excitation (SE) attention and deep supervision is employed to progressively recover and refine semantic and spatial features across multiple scales. To validate the efectiveness of the proposed MambaMPD framework, we conduct comprehensive experiments on two challenging marine pollution detection datasets, M4D [7] and MADOS [8], featuring realistic sea conditions, ambiguous boundaries, and pollutant-background similarities. Following Kikaki et al. [8], our experiments focus on oil spills and marine debris, the two major marine pollutants observable in satellite imagery; extension to other pollution types is discussed in Section 6.

In summary, our contributions are shown as follows:

1) We propose MambaMPD tailored for marine pollution detection, addressing domain-specific challenges in SAR and multispectral imagery.

2) We introduce an FAA module, which augments the Mamba encoder by incorporating WT to capture both low-frequency contextual semantics and high-frequency structural cues, crucial for detecting small, low-contrast, and irregular pollution targets, without introducing noticeable parameter overhead.

3) We introduce a multi-scale EGA module to efectively fuse multi-scale edge features with deep semantic representations. This module improves boundary localisation and reduces confusion in visually similar marine pollution classes.

![](images/6c669c3a3a9f851813104afe06fbdfc5a3c6794fa0cc3f39f10657e9d1d31576.jpg)  
Figure 2. Overall architecture of the proposed framework. The encoder extracts multi-scale features from the input image through a wavelet-enhanced hierarchical backbone consisting of a WT block, patch embedding, patch merging, and stacked VSS blocks. To strengthen boundary modeling, Laplacian features from diferent stages are incorporated into edge-guided attention modules, which refine encoder representations before they are passed to the decoder. The decoder progressively restores spatial resolution via stacked UnetrUpBlocks and SE blocks, where skip connections explicitly transfer encoder features to the corresponding decoder stages for structural detail recovery. Deep supervision is further applied to intermediate decoding layers, and the final prediction is generated through the output mapping layer.

4) Experimental results show that the proposed MambaMPD achieves the best mIoU among compared methods, outperforming TransOilSeg by 6.82% in Oil Spill IoU on M4D and MariNeXt by 4.2% in F1-score on MADOS.

## 2. Related Work

## 2.1. Marine Pollution Detection from Remote Sensing

Early approaches to marine pollution detection in remote sensing primarily relied on expert-defined heuristics and rule-based strategies. These traditional methods [33– 35, 38, 39], such as spectral thresholding, morphological filtering, and geometric descriptors, were typically tailored to detect a single pollutant type. However, they struggle in multi-class scenarios due to limited adaptability to competing marine elements.

For instance, Kikaki et al. [8] observed that rule-based systems often confuse plastic debris with natural water patterns and misclassify clean water regions when appearance overlap is high. Similar issues have been reported in optical imagery, where marine mucilage is mistaken for floating waste [40], and in SAR data where fine debris is difficult to distinguish from ships or sea clutter due to similar radar backscatter [41]. While fast and interpretable, these methods’ reliance on low-level cues and rigid rules limits their scalability and generalisation in real-world applications.

To address these shortcomings, conventional machine learning (ML) techniques have been explored, where handcrafted features, such as texture, shape, or contextual descriptors, are paired with classifiers like support vector machines or random forests [42–44]. These approaches introduce greater adaptability and can incorporate spatial patterns to detect complex marine targets. In particular, ML models have shown promise in recognising small-scale or texture-rich pollutants under varying conditions [40]. Nevertheless, their success is often constrained by the quality of manually extracted features and the inherent imbalance in marine datasets. Moreover, specific pollutant types, such as large plastic patches or dispersed oil films, present sporadic appearances and irregular shapes [45, 46], further challenging the generalisation capabilities of these models.

With the rise of deep learning, semantic segmentation techniques have been increasingly adopted in remote sensing applications. Classic encoder–decoder architectures such as U-Net [47] and DeepLab [48] extract hierarchical features and leverage skip connections and atrous convolutions to maintain both global semantic context and finegrained spatial details [49–51]. These approaches have been progressively extended to the task of marine pollution detection. A series of early works focused on oil spill detection using SAR imagery as a binary segmentation task. For example, Mahmoud et al. [21] employed a standard U-Net to separate oil spills from background water, achieving promising results under simple scenarios. However, this setup lacks scalability for more complex or diverse marine scenes. To address blurry boundaries and noisy textures in SAR imagery, CBD-Net [52] introduced contextual and boundary supervision, enhancing small target segmentation and achieving state-of-the-art performance on the SOS dataset. Further improving SAR-based segmentation, DGNet [53] incorporated the intrinsic distribution of SAR backscatter values into a latent variable inference generation loop, demonstrating data-eficient training and superior generalisation with limited annotations. Similarly, SRCNet [54] designed a competing dual-network structure leveraging a SAR-specific image representation, which enhanced learning from small-scale datasets and achieved accurate segmentation under constrained supervision.

Beyond binary segmentation, recent works began addressing the need to distinguish between multiple types of marine targets in SAR images, such as ship wakes, look-alike dark spots, and other oil-like artefacts. The M4D dataset [7] represents an efort in this direction, ofering a richer annotation schema for evaluating fine-grained pollutant detection. Building on this, Wu et al. [27] proposed SAM-OIL, leveraging vision foundation models for few-shot and interactive segmentation. However, its performance is constrained by prompt quality and the absence of explicit structural cues, limiting precision in noisy, cluttered marine scenes. Chai et al. [26] proposed TransOilSeg, which performs well on large-scale imagery with clear oil slicks but may struggle in scenes with small, fragmented, or low-contrast targets.

To extend beyond SAR, Duarte et al. [55] developed a deep learning pipeline for detecting marine debris in multispectral imagery. Kikaki et al. [8] introduced the MADOS dataset and proposed MariNeXt, a CNN-based model that fuses spectral and spatial cues. While efective across pollutant types and sensors, MariNeXt lacks mechanisms to capture structural or edge information crucial for complex scenes.

In summary, existing models still struggle with key challenges such as small object size, low contrast and boundary confusion [56–58]. Distinct from prior works, our proposed MambaMPD explicitly addresses these limitations through the introduction of the Frequency-Aware Augmentation (FAA) module and Edge-Guided Attention (EGA) module, which enhance structural sensitivity, preserve contextual semantics, and improve the delineation of fine-grained object boundaries.

## 2.2. Vision Mamba in Remote Sensing Segmentation

A growing body of research has explored the adaptation of the Mamba architecture [28] to remote sensing image segmentation, showcasing its eficiency in modeling long-range dependencies for high-resolution and large-scale spatial tasks. For instance, RS-Mamba [59] introduces an omnidirectional selective scan to process gigapixel-scale remote sensing images with linear computational cost, achieving competitive results in land cover segmentation and change detection. RS3Mamba [60] complements a convolutional backbone with a state-space auxiliary branch, enabling efective multi-level feature fusion for urban semantic segmentation. In the bi-temporal domain, Change-Mamba [61] employs a Vision-Mamba encoder and a set of specialized decoders to capture temporal correlations, surpassing traditional CNN and Transformer baselines on several change detection benchmarks. Meanwhile, RSMamba [31] further demonstrates the potential of Mamba in spatial classification by introducing dynamic multipath activation to remove causal constraints and better adapt to 2D spatial tasks. Some preliminary segmentation works, such as in [59, 62], have verified the feasibility of Mamba in land cover mapping and building footprint extraction.

Despite their promising capabilities, these eforts concentrate on land-oriented tasks such as land cover mapping, urban segmentation, and change detection; marine pollution detection, which demands sharper boundary delineation and sensitivity to highfrequency structural details for small, sparse, and visually ambiguous pollutants, has received little attention. Among Mamba-based approaches, OSDMamba [37] is the closest to our work, applying a standard VSS encoder to binary oil-spill segmentation in SAR imagery. MambaMPD difers in three respects: it augments the encoder with frequency-aware decomposition (FAA) and injects multi-scale edge priors into decoding (EGA); it extends the task from binary oil-spill segmentation to multi-class marine pollution detection; and it is validated on both SAR and multispectral data. A quantitative comparison controlling for model capacity is given in Section 4.4.2.

## 3. METHOD

Marine pollution detection from remote sensing imagery can be formulated as a pixelwise multiclass semantic segmentation task. Given an input image $X \in \mathbb { R } ^ { H \times \dot { W } \times C _ { \mathrm { i n } } }$ the goal is to learn a mapping $Y = \mathcal { F } _ { \theta } ( X ) \in \mathbb { R } ^ { C \times H \times W }$ that assigns each pixel a probability distribution over C semantic categories. In our formulation, X can represent either multispectral optical images or single-/dual-polarization SAR data, enabling the framework to operate in a modality-agnostic manner. The following subsections present the proposed MambaMPD framework in detail.

![](images/e12a5d427c5f087d07b5a528b061deb89a71968654b2cca00b83a9b808139884.jpg)  
Figure 3. Illustration of the proposed Frequency-Aware Augmentation Module (FAA). Given an input feature, FAA employs a dual-branch design consisting of a residual branch and a wavelet transform branch. The wavelet branch performs level-1 wavelet decomposition to separate low- and high-frequency components, while the high-frequency information is further refined through an additional wavelet-based convolution operation. The processed wavelet features are fused with the residual features and subsequently recalibrated by a channel-attention mechanism to emphasize informative frequency responses. The final enhanced representation is produced through residual aggregation, followed by a 7 × 7 convolution and instance normalization.

## 3.1. Mamba-based Encoder

MambaMPD adopts the Visual State Space (VSS) block [29] as its fundamental building unit. The VSS block replaces traditional attention mechanisms with selective statespace dynamics, eficiently modeling long-range dependencies with linear complexity. Given a sequence of visual tokens $\mathbf { \bar { \Psi } } _ { X } \in \mathbf { \bar { \mathbb { R } } } ^ { H \times W \times C }$ , the core state-space recurrence is defined as:

$$
s _ { t } = A \cdot s _ { t - 1 } + B \cdot { \mathrm { C o n v } } _ { \mathrm { p r o j } } ( x _ { t } ) , \qquad y _ { t } = \sigma ( G _ { t } ) \odot ( C \cdot s _ { t } ) ,\tag{1}
$$

where A and B are learnable transition matrices governing the system dynamics, $G _ { t }$ is a learned spatial gate, and C is a readout matrix. The output is restored to the original spatial dimension via a convolutional projection $\hat { x } _ { t } = \mathrm { C o n v } _ { \mathrm { o u t } } ( y _ { t } )$ . To capture spatial context from multiple orientations, the VSS block further employs a two-dimensional selective scanning (SS2D) mechanism [30] that unfolds the feature map along four directional paths and merges the resulting representations.

Here, $s _ { t } \in \mathbb { R } ^ { N }$ denotes the hidden state of dimension N, $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { C }$ is the feature vector of a single visual token, and $A \in \mathbb { R } ^ { N \times N } , B \in \mathbb { R } ^ { \dot { N } \times C } , C \in \mathbb { R } ^ { C \times N }$ are the state-space matrices. To apply this 1-D recurrence to 2-D feature maps, SS2D [29] unfolds each $H \times W$ feature map along four directional paths (left-to-right, right-toleft, top-to-bottom, bottom-to-top) to produce four sequences of length $H \times W$ . Each sequence is processed independently; the four outputs are then summed element-wise and reshaped to $H \times W \times C$ . The overall encoder follows the hierarchical design of VMamba-Tiny [29] and Swin-UMamba [30], performing 2× spatial downsampling at each stage. The four encoder stages are composed of {2, 2, 9, 2} VSS blocks, respectively, providing a balanced trade-of between depth and eficiency. However, as shown in Fig. 4(a), our empirical observations reveal that the standard Mamba-based encoder sufers from limited frequency sensitivity and poor edge preservation, particularly for small, sparsely distributed marine pollutants under complex backgrounds.

## 3.2. Frequency-Aware Augmentation Module

Frequency decomposition is motivated by the physics of marine pollution imagery. In SAR, oil spills dampen capillary and short gravity waves, producing smooth dark patches; the large-scale backscatter contrast is carried by low-frequency components, while the sharp gradient at slick boundaries falls in high-frequency components [63]. In multispectral data, pollutants difer from clean water subtly at a global scale but abruptly at their edges. Wavelet decomposition separates these two cues: lowfrequency sub-bands encode region-level semantics and high-frequency sub-bands preserve boundary detail [64]. As shown in Fig. 2, the FAA module sits before the patch embedding layer and operates on the raw input $X \in \mathbb { R } ^ { H \times W \times C _ { \mathrm { i n } } }$ . Placing it this early injects frequency cues before spatial downsampling, so high-frequency detail is not lost to successive patch merging. To address the limited frequency sensitivity of the standard encoder, we propose a Frequency-Aware Augmentation (FAA) module, as illustrated in Fig. 3. FAA introduces hierarchical wavelet decomposition together with lightweight convolutional refinement, enabling the encoder to simultaneously capture global contextual structures and fine-grained boundary details that are critical for marine pollution segmentation.

![](images/17c55cfd614b5e9418d7a3a26b020fa209407b8580fd0b487339de519fca6fe9.jpg)  
Image

![](images/89fba05a3b8206551e706ffac09bf2e73fa1d7619087eb87c0bb5b5614d1257b.jpg)  
(a)

![](images/721cdac9ad034f080ff9eb0e14ea56e0fc01f0af8b3227114c8e0be7a5e1471d.jpg)  
(b)  
Figure 4. (a) shows the feature response using the baseline Mamba encoder, while (b) demonstrates the enhanced performance achieved by integrating FAA into the baseline Mamba encoder. Red boxes highlight regions where the FAA enhances feature responses in small targets and boundary regions, which are critical for detecting sparse marine pollutants in complex backgrounds.

Given an input feature tensor $X \in \mathbb { R } ^ { C \times H \times W }$ , we first apply a separable 2D Haar wavelet transform to decompose X into four frequency sub-bands:

$$
[ X _ { L L } , X _ { L H } , X _ { H L } , X _ { H H } ] = \mathrm { W T } ( X ) ,\tag{2}
$$

where $X _ { L L }$ denotes the low-frequency approximation component, and $X _ { L H } , \ X _ { H L } .$ $X _ { H H }$ capture high-frequency details along the horizontal, vertical, and diagonal directions, respectively. Each sub-band is refined by a shared 3 × 3 convolutional operator $\phi ( \cdot )$ and reconstructed via the inverse wavelet transform (IWT).

To further enlarge the efective receptive field and strengthen multi-scale frequency perception, we adopt a hierarchical decomposition strategy where only the low-frequency approximation branch from the previous level is further decomposed:

$$
[ X _ { L L } ^ { ( i ) } , X _ { L H } ^ { ( i ) } , X _ { H L } ^ { ( i ) } , X _ { H H } ^ { ( i ) } ] = \mathrm { W T } ( X _ { L L } ^ { ( i - 1 ) } ) .\tag{3}
$$

![](images/c3876dca7f710e588d11263d45dddd53c7cb9f67658df0970bd71b324ae5c83e.jpg)  
Image

![](images/be50ef4c063b272e2a1651edb08ee28b270b4b296d7ec8c5f8f95636d62bd46d.jpg)  
(a)

![](images/5dc14280b40ac1a5314e674477e8aaebb60ef0bafaeedd69892442ce0fa8034f.jpg)  
(b)  
Figure 5. (a) presents the baseline feature response, and (b) shows the enhanced feature response with the SE-ResDecoder. Red boxes show that our approach produces more accurate boundaries and consistent segmentation masks, demonstrating improved robustness in small-scale and ambiguous marine pollutant regions.

This enables the model to progressively capture increasingly global structures while preserving high-frequency details at each level. The features are then reconstructed in a coarse-to-fine manner:

$$
Z ^ { ( i ) } = \mathrm { I W T } \big ( y _ { L L } ^ { ( i ) } + \uparrow ( Z ^ { ( i + 1 ) } ) , y _ { L H } ^ { ( i ) } , y _ { H L } ^ { ( i ) } , y _ { H H } ^ { ( i ) } \big ) ,\tag{4}
$$

where $y _ { L L } ^ { ( i ) } , y _ { L H } ^ { ( i ) } , y _ { H L } ^ { ( i ) }$ , and $y _ { H H } ^ { ( i ) }$ are the refined sub-band features at level $i , \uparrow ( \cdot )$ denotes upsampling to match the spatial resolution of the current level, and $Z ^ { ( i + 1 ) }$ is the reconstructed output from the next coarser level. Through this top-down reconstruction scheme, FAA progressively integrates coarse contextual information with fine structural details across decomposition levels.

The DWT and IWT use fixed Haar wavelet filters implemented as stride-2 convolutions and transposed convolutions, respectively; their weights are frozen and receive no gradient updates. The only trainable components in FAA are the $3 \times 3$ refinement convolutions $\phi ( \cdot )$ applied to each sub-band and the subsequent SE channel attention block. Gradients pass through the fixed wavelet layers via the standard chain rule, as through any non-learnable linear operation. Table 10 confirms this: FAA adds few trainable parameters. The reconstructed feature is then fused with a residual branch and passed through a squeeze-and-excitation (SE) block to adaptively recalibrate channel responses. Finally, a $7 \times 7$ depthwise convolution followed by instance normalization is applied to enhance local-global interaction and generate the final augmented feature. As illustrated in Fig. 4(b), incorporating FAA into the encoder leads to more discriminative responses in small-target and boundary regions. The ablation study (Table 6) confirms this quantitatively: FAA alone improves Oil Spill IoU by +5.29% over the baseline, which are particularly important for detecting sparse marine pollutants under complex backgrounds. Notably, as reported in Table 10, this additional cost is negligible in practice. As reported in Table 10, FAA adds only 0.30 M trainable parameters while providing the largest single-module gain in Oil Spill IoU.

## 3.3. Decoder

To improve feature recalibration and semantic refinement during decoding, we adopt a Squeeze-and-Excitation Residual Decoder (SE-ResDecoder), as shown in Fig. 2. Each decoder stage consists of a UnetrUpBlock followed by an SE block. The UnetrUpBlock first upsamples the coarser decoder feature via transpose convolution and concatenates it with the corresponding encoder skip feature. The fused representation is then refined through two residual convolutional blocks composed of convolution, normalization, and LeakyReLU activation. This design enables the decoder to progressively align semantic information from coarse decoder features with structural details from shallow encoder features. Subsequently, a squeeze-and-excitation block performs channel-wise recalibration to emphasize the most informative feature channels.

In addition, deep supervision is applied at each decoder stage: a 1 × 1 convolution generates an auxiliary segmentation prediction that is upsampled to the original resolution and supervised by the ground-truth mask, facilitating stable gradient propagation and encouraging consistent semantic recovery across decoder stages. As illustrated in Fig. 5(a) and Fig. 5(b), the proposed SE-ResDecoder enables the network to recover more semantically coherent and spatially precise predictions, particularly in challenging regions with ambiguous boundaries, fragmented structures, and small targets.

## 3.4. Edge-Guided Attention (EGA) Module

Although the Mamba-based encoder–decoder architecture exhibits strong global contextual modeling capabilities, Fig. 6(a) shows that purely semantic decoding often struggles to recover fine-scale pollution structures, particularly in low-contrast or visually ambiguous ocean regions. These failure cases suggest that decoder features alone are insuficient to provide the geometric cues required for accurate boundary reconstruction. To address this limitation, we introduce an Edge-Guided Attention (EGA) module, as schematically illustrated in Fig. 7. The purpose of EGA is to inject structural guidance into the decoding process by jointly exploiting global context, local structural cues, and edge-aware guidance derived from the decoder prediction.

Let $\hat { f } _ { i } ^ { e } \in \mathbb { R } ^ { H _ { i } \times W _ { i } \times C _ { i } }$ denote the encoder feature at stage i, and let $\hat { f } _ { i + 1 } ^ { d } \in \mathbb { R } ^ { H _ { i } \times W _ { i } \times 1 }$ denote the decoder-side prediction aligned to the same spatial resolution. $\hat { f } _ { i + 1 } ^ { d }$ is an intermediate auxiliary prediction from the deep supervision branch at decoder stage i+1. As described in Section 3.3, each decoder stage produces a single-channel prediction via a $1 \times 1$ convolution and sigmoid activation. This prediction is bilinearly interpolated to the resolution of the encoder feature $\hat { f } _ { i } ^ { e }$ before entering EGA. Using these stage-wise predictions as edge guidance lets EGA access semantic cues that sharpen as decoding progresses, rather than depending on a single fixed output. Following the upper-left branch of Fig. 7, we first derive complementary global and local spatial response maps $f _ { i } ^ { g }$ and $f _ { i } ^ { l o c }$ from the encoder feature via a Global Feature Extractor and a Local Feature Extractor, respectively, where $f _ { i } ^ { g }$ captures broad contextual saliency and $f _ { i } ^ { l o c }$ preserves local structural details. To explicitly inject boundary cues, the decoder prediction is transformed into an edge-aware map through a Laplacian operator:

$$
f _ { i } ^ { \mathrm { e d g e } } = \mathrm { N o r m } \left( \frac { 1 } { 3 } \sum _ { k \in \{ 1 , 2 , 4 \} } \dag \left( \left| \mathbf { L } \ast \mathrm { M a x P o o l } ( \hat { f } _ { i + 1 } ^ { d } ; k ) \right| \right) \right) , \quad \mathbf { L } = \left[ \begin{array} { l l l } { 0 } & { - 1 } & { 0 } \\ { - 1 } & { 4 } & { - 1 } \\ { 0 } & { - 1 } & { 0 } \end{array} \right]\tag{5}
$$

![](images/49fb01db969cbe6fdd9c7cce93cf99c5569c833b6358c17b9cfa0d67621f10d1.jpg)  
Image

![](images/7b5a81f42a219087328530acf97d9ebcad02682acc2df3f4c7f8048dba3d11c2.jpg)  
(a)

![](images/abad4603618a3ba1c1288437e9882d0759bd45d143cdd76feac74c2fdafb6bce.jpg)  
(b)  
Figure 6. (a) represents the baseline model without edge guidance and (b) represents the baseline model with the proposed EGA module. Red boxes highlight that the baseline with the EGA module produces sharper boundaries and more coherent feature responses.

The global and local responses are combined and modulated by the edge prior to form a structurally guided attention map:

$$
f _ { i } ^ { m } = ( f _ { i } ^ { g } + f _ { i } ^ { l o c } ) \odot f _ { i } ^ { e d g e } ,\tag{6}
$$

where ⊙ denotes element-wise multiplication. The resulting map encodes structurally guided spatial attention informed by both context and predicted boundaries.

This modulated map, together with the global and local response maps, is concatenated with the encoder feature, refined by a $3 \times 3$ convolution, and combined with the original encoder feature through a residual connection to preserve the original representation while injecting guided spatial emphasis:

$$
\begin{array} { r } { f _ { i } ^ { c } = \mathrm { C o n v } _ { 3 \times 3 } \Big ( \mathrm { C o n c a t } [ \hat { f } _ { i } ^ { e } , f _ { i } ^ { g } , f _ { i } ^ { \mathrm { l o c } } , f _ { i } ^ { m } ] \Big ) + \hat { f } _ { i } ^ { e } . } \end{array}\tag{7}
$$

Finally, an Eficient Channel Attention (ECA) block [65] is applied to adaptively emphasize informative channels. The ECA block computes a channel descriptor via global average pooling and obtains attention weights through a lightweight 1D convolution, producing the edge-enhanced output $f _ { i } ^ { a }$ for subsequent decoding. A detailed algorithmic summary of the full EGA pipeline, including multi-scale edge prior construction and cross-scale feature fusion, is provided in Table 1. As shown in Fig. 6(b), the proposed EGA module substantially improves boundary fidelity, reduces false positives in ambiguous regions, and facilitates coherent segmentation of thin, irregular, and highly fragmented pollution structures.

![](images/98755610d055c20e33a93dd3c55bfbacc0ab93b258703e44c102d4f19e3e387f.jpg)  
Figure 7. Conceptual overview of the proposed Edge-Guided Attention (EGA) module. The encoder feature is processed by parallel global and local feature extractors to capture complementary contextual and detailed spatial cues. In parallel, the decoder prediction is transformed into an edge-aware guidance map, which is used to modulate the feature representation. The modulated features are then concatenated and refined, followed by channel-wise recalibration using the Eficient Channel Attention (ECA) module, to produce the final edgeenhanced output. This figure is intended as a schematic illustration of the overall interaction mechanism, while the detailed implementation and mathematical formulation are provided in Eqs. (5)-(7).

Table 1. Pseudo-code of the proposed Edge-Guided Attention (EGA) at decoder stage i.  
```latex
Input: Encoder feature $f _ { i } ^ { e } \in \mathbb { R } ^ { C _ { i } \times H _ { i } \times W _ { i } }$ , decoder prediction $f _ { i + 1 } ^ { d } \in \mathbb { R } ^ { 1 \times H _ { i } \times W _ { i } }$
Output: Edge-enhanced encoder feature $f _ { i } ^ { a } \in \mathbb { R } ^ { C _ { i } \times H _ { i } \times W _ { i } }$
Phase I: Multi-scale edge prior construction
1. Fine-scale Laplacian response: ${ \cal E } ^ { ( 1 ) } \gets \Big | { \bf L } * f _ { i + 1 } ^ { d } \Big | .$
2. For each coarse scale k $\in \{ 2 , 4 \}$ , compute $\tilde { f } _ { i + 1 } ^ { ( k ) } \gets$ MaxPool $\displaystyle f _ { i + 1 } ^ { d } ; k )$ and $E ^ { ( k ) } \gets \mathrm { U }$ psample $\left( \left| \mathbf { L } * \tilde { f } _ { i + 1 } ^ { ( k ) } \right| \right)$
3. Aggregate cross-scale edge cues: $\begin{array} { r } { f _ { i } ^ { \mathrm { e d g e } } \gets \mathrm { N o r m } \Big ( \frac { 1 } { 3 } \sum _ { k \in \{ 1 , 2 , 4 \} } E ^ { ( k ) } \Big ) } \end{array}$
Phase II: Spatial modulation
4. Global and local spatial responses: $f _ { i } ^ { g } \gets \mathrm { G E } ( f _ { i } ^ { e } ) , \quad f _ { i } ^ { \mathrm { l o c } } \gets \mathrm { L E } ( f _ { i } ^ { e } )$
5. Edge-modulated attention map: $f _ { i } ^ { m } \stackrel { \cdot } {  } ( f _ { i } ^ { g } + f _ { i } ^ { \mathrm { { l o c } } } ) \odot \stackrel { \cdot } { f _ { i } ^ { \mathrm { e d g e } } } .$
Phase III: Feature refinement and channel recalibration
6. Concatenation and residual refinement: $f _ { i } ^ { c } \gets \mathrm { C o n v } _ { 3 \times 3 } \big ( \mathrm { C o n c a t } ( f _ { i } ^ { e } , f _ { i } ^ { g } , f _ { i } ^ { \mathrm { l o c } } , f _ { i } ^ { m } ) \big ) + f _ { i } ^ { e }$
7. Channel recalibration via ECA: $f _ { i } ^ { a } \gets f _ { i } ^ { c } \stackrel { \cdot } { \otimes } \sigma ($ Conv $\mathrm { D } _ { \kappa } \big ( \mathrm { G A P } ( f _ { i } ^ { c } ) \big ) \big )$ $\kappa = \psi ( C _ { i } ) .$
```

## 4. Experiments

## 4.1. Datasets

1) MADOS Dataset [8]: The MADOS (Marine Debris and Oil Spill) dataset is a globally distributed benchmark dataset specifically designed for detecting marine pollution, including oil spills and marine debris. It contains 174 multispectral Sentinel-2 (S2) satellite images, captured between 2015 and 2022, with approximately 1.5 million pixels annotated. The dataset covers 15 diferent thematic categories, such as oil spills, marine debris, ships, phytoplankton, turbid waters, etc. The S2 images in the MADOS dataset cover global coastal waters with spatial resolutions of 10 meters, 20 meters, and 60 meters, and a revisit cycle of about 5 days. During the data processing, ACOLITE atmospheric correction [79] was used to extract Rayleigh reflectance, and annotations were provided by multiple experts, with a focus on labelling major pollutants like oil spills and marine debris. Additionally, the dataset exhibits significant diversity, covering various geographic distributions and environmental conditions, capturing diferent weather and ocean states. We adopt the oficial scene-level data split provided by Kikaki et al. [8], where training and test samples correspond to geographically distinct Sentinel-2 acquisitions, preventing spatial leakage between partitions.

Table 2. Quantitative comparison of MambaMPD on the MADOS dataset(Excluding foundation model). Note: U-Net is included here as one of the compared methods for overall benchmarking; the module-wise ablation on the core Mamba architecture is reported in Table 6.
<table><tr><td rowspan=1 colspan=1>Model                 F1 (%, ↑) mIoU $( \% , \uparrow )$  OA (%, ↑)</td></tr><tr><td rowspan=1 colspan=1>RF[66]                      56.6            43.9             67.1</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { R F + + } [ 6 7 ]$                   64.4            52.4             83.8</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { U - N e t } [ \mathrm { 4 } 7 ]$                    63.8            51.0             82.9</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { S e g N e { \bar { X } t } { \bar { [ 6 8 ] } } }$                60.6            49.2            86.6</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { M a r i N e X \bar { t } \ [ \bar { 8 } ] }$               70.6            59.2             81.6</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { O S D M a m b a } [ 3 7 ]$            71.2            68.1             82.3 $\mathrm { M a m b a M P D \ ( o u r s ) }$       74.8           69.8            83.1</td></tr></table>

Table 3. Quantitative comparison of our MambaMPD model against recent remote sensing foundation models on the MADOS dataset.
<table><tr><td>Model</td><td>mIoU (%, ↑)</td></tr><tr><td>DOFA[69]</td><td>59.58</td></tr><tr><td> $\mathrm { G F M - } \dot { \mathrm { S } } \mathrm { w i n } [ 7 0 ]$ </td><td>64.7</td></tr><tr><td>prithvi[71]</td><td>49.9</td></tr><tr><td> $\mathrm { R e m o t i e C L I P [ 7 2 ] }$ </td><td>60.0</td></tr><tr><td> $\mathrm { S a t l a s N e t } [ 7 3 ]$ </td><td>55.9</td></tr><tr><td> $\mathrm { S c a l e - M A E } [ \bar { 7 } 4 ]$ </td><td>57.3</td></tr><tr><td> $\mathrm { S p e c t r a l G P T } [ \bar { 7 } 5 ]$ </td><td>57.9</td></tr><tr><td> $\mathrm { S S L 4 E O  – S 1 2 – M o C o [ 7 6 ] }$ </td><td>51.8</td></tr><tr><td> $\mathrm { S S L 4 E O  – S 1 2 – D I N O \bar { / } 7 6 \bar { ] } }$ </td><td>49.4</td></tr><tr><td> $\mathrm { S S L 4 E O  – S 1 2 – M A E [ \bar { 7 } 6 ] }$ </td><td></td></tr><tr><td> $\mathrm { S S L 4 E O - S 1 2 - D a t a 2 V e c [ 7 6 ] }$ </td><td>49.9</td></tr><tr><td></td><td>44.4</td></tr><tr><td> $\mathrm { T e r r a M i n d - B \ ( T e r r a M e s h ) [ 7 7 ] }$ </td><td>69.5</td></tr><tr><td> $\mathrm { P A N G A E A ( U { - } N e t ) [ 7 8 ] }$ </td><td>57.8</td></tr><tr><td> $\mathrm { P A N G A E A ( V I T B  – i 6 ) [ 7 8 ] }$   $\mathbf { M a m b a M P D \ ( o u r s ) }$ </td><td>48.2 69.8</td></tr><tr><td></td><td></td></tr></table>

2) M4D Dataset [7]: The M4D dataset was developed for marine oil spill detection research. The dataset consists of images extracted from satellite synthetic aperture radar (SAR) data, covering oil spills and other related semantic categories, with corresponding ground truth masks and labels [7]. Specifically, the dataset comprises 1002 training images and 110 test images, each with corresponding labels. A total of 5 semantic classes are annotated, including: sea surface, oil spill, look-alike, ship, and land. We use the fixed 1002/110 train/test split of Krestenitis et al. [7]; training and test images come from separate SAR acquisitions with no spatial overlap. We hold out 5% of the training set for validation.

## 4.2. Implementation Details

Our network is built using the PyTorch framework. In the M4D dataset, the training template we used follows the design in HTSM [80]. We trained the model using an SGD optimizer with a momentum of 0.9 and a weight decay of 1e-4. Additionally, we set the initial learning rate to 0.001 and employed a “Poly” decay strategy. All experiments were implemented on an NVIDIA 4070Ti GPU. The batch size was set to 8, with a maximum of 100 epochs. Similarly, in the MADOS dataset, we used the oficial training framework provided by [8]. For classical CNN methods, the backbone networks employed during the encoder stage were pretrained on ImageNet. Following OSDMamba [37], we use a hybrid objective that combines Focal Loss and Jaccard Loss, formulated as $\begin{array} { r } { L = - \alpha ( 1 - p _ { t } ) ^ { \gamma } \log ( p _ { t } ) + \left( 1 - \frac { \sum _ { i } T P _ { i } } { \sum _ { i } T P _ { i } + \sum _ { i } F P _ { i } + \sum _ { i } F N _ { i } } \right) } \end{array}$ , where α balances class frequencies, $p _ { t }$ is the predicted probability of the true class, γ modulates the emphasis on hard examples, and $T P _ { i } , F P _ { i } , F N _ { i }$ denote pixel-wise statistics for class i. The model was initialized with pre-trained weights from ImageNet [81]. A batch size of 4 was used, and the model was trained for 100 epochs across all stages. Deep supervision is applied across decoding stages, and the final training objective sums the hybrid loss with all auxiliary prediction losses using predefined layer-wise weights. All methods are compared on the oficial train/test split of each dataset with identical metric computation. The CNN- and Transformer-based results in Tables 2 and 4 are obtained on these oficial splits, with each model trained end-to-end from an ImageNet-pretrained backbone under its original configuration. The foundationmodel results in Table 3 follow the PANGAEA evaluation framework [78]: each model is initialised with its publicly released weights obtained by large-scale pre-training on Earth observation data (masked autoencoding, contrastive learning, or supervised pre-training), its encoder is frozen as a feature extractor, and a UPerNet decoder fed with four intermediate feature levels is trained on the oficial MADOS training split, using the same decoder, optimiser, schedule, and band adaptation for all models. MambaMPD is trained end-to-end from ImageNet-pretrained weights on the same splits.

Table 4. Quantitative performance comparison on the M4D dataset.
<table><tr><td>Model</td><td>Sea Surface(%)</td><td></td><td>Oil Spill(%) Look alike(%)</td><td>Ship(%)</td><td>Land(%)</td><td>mIoU(%)</td><td>FPS</td></tr><tr><td>Unet[47]</td><td>93.90</td><td>53.79</td><td>39.55</td><td>44.93</td><td>92.68</td><td>64.97</td><td>52.16</td></tr><tr><td>LinkNet[82]</td><td>94.99</td><td>51.53</td><td>43.24</td><td>40.23</td><td>93.97</td><td>64.79</td><td>66.45</td></tr><tr><td>PSPNet[83]</td><td>92.78</td><td>40.10</td><td>33.79</td><td>24.42</td><td>86.90</td><td>55.60</td><td>25.23</td></tr><tr><td>DeepLabv2[84]</td><td>94.09</td><td>25.57</td><td>40.30</td><td>11.41</td><td>74.99</td><td>49.27</td><td>15.62</td></tr><tr><td>DeepLabv2 (msc)[84]</td><td>95.39</td><td>49.53</td><td>49.28</td><td>31.26</td><td>88.65</td><td>62.83</td><td>16.14</td></tr><tr><td>DeepLabv3+[48]</td><td>96.43</td><td>53.38</td><td>55.40</td><td>27.63</td><td>92.44</td><td>65.06</td><td>55.10</td></tr><tr><td>YOLOv8-SAM[27]</td><td>94.34</td><td>41.84</td><td>48.15</td><td>52.48</td><td>87.65</td><td>64.89</td><td></td></tr><tr><td>SAM-OIL[27]</td><td>96.05</td><td>51.60</td><td>55.60</td><td>52.55</td><td>91.81</td><td>69.52</td><td></td></tr><tr><td>TransOilSeg[26]</td><td>97.02</td><td>61.38</td><td>62.41</td><td>33.49</td><td>94.39</td><td>69.74</td><td>8.86</td></tr><tr><td>OSDMamba[37]</td><td>96.47</td><td>65.59</td><td>47.57</td><td>46.85</td><td>94.76</td><td>70.25</td><td>10.00</td></tr><tr><td>MambaMPD (ours)</td><td>96.30</td><td>68.20</td><td>43.60</td><td>52.36</td><td>93.79</td><td>70.85</td><td>15.19</td></tr></table>

## 4.3. Evaluation Metrics

For the M4D dataset, we use the mean Intersection over Union (mIoU) score over the union to evaluate the model performance. In the MADOS dataset, both mIoU and average F1 (Ave.F1) scores are used to evaluate the model performance. These two evaluation metrics are based on the confusion matrix, which contains four components: True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). For each class, IoU is defined as the ratio of the intersection and union of the predicted and ground truth values. In addition, we report macro-averaged Precision, Recall, and F1-score on the M4D dataset (Table 5) to provide a more comprehensive assessment of model performance.

![](images/270ef2c7df4ff478b646cd23258d98e98e02c20df990f35433d6d813ffc22f56.jpg)  
Figure 8. Comparison of confusion matrices on the MADOS dataset.

Table 5. Additional evaluation metrics on the M4D dataset. Precision, Recall, and F1-score are computed as the macro-average across all five classes.
<table><tr><td>Model</td><td>Precision (%)</td><td>Recall (%)</td><td>F1 (%)</td><td>mIoU (%)</td></tr><tr><td>U-Net</td><td>77.80</td><td>74.75</td><td>75.50</td><td>64.97</td></tr><tr><td>DeepLabV3+</td><td>72.72</td><td>77.05</td><td>74.57</td><td>65.06</td></tr><tr><td>TransOilSeg</td><td>79.77</td><td>78.33</td><td>78.99</td><td>69.74</td></tr><tr><td>OSDMamba</td><td>80.13</td><td>78.21</td><td>77.35</td><td>70.25</td></tr><tr><td>MambaMPD (ours)</td><td>81.02</td><td>79.41</td><td>78.18</td><td>70.85</td></tr></table>

## 4.4. Comparison With State-of-the-Art

## 4.4.1. Quantitative Results on MADOS Dataset

We evaluated and compared our MambaMPD with RF [66], RF++ [67], U-Net [47], SegNeXt [68], MariNeXt (reproduced) [8], and approaches involving fine-tuning of remote sensing foundation models, such as CROMA, DOFA, GFM-Swin, Prithvi, RemoteCLIP, SatlasNet, Scale-MAE, SpectralGPT, SSL4EO, TerraMind-B, and PAN-GAEA [69–78, 85]. Table 2 and Table 3 report the segmentation performance of each method on the MADOS dataset, further demonstrating the efectiveness of our proposed MambaMPD. Our MambaMPD achieves an F1 score of 74.8% and an mIoU of 69.8%, outperforming the other compared methods. Compared to classical ensemble methods such as RF (F1: 56.6%, mIoU: 43.9%) and RF++ (F1: 64.4%, mIoU: 52.4%), MambaMPD shows substantial improvements, particularly in terms of segmentation quality. When compared to U-Net [47] and SegNeXt [68], which are popular CNNbased architectures, MambaMPD delivers significant gains of 18.8% and 20.6% in mIoU, respectively. Moreover, it also outperforms MariNeXt [8], a multi-branch CNN designed for multi-type marine pollutant recognition, by a notable margin of 10.6% in mIoU and 4.2% in F1 score. These results clearly demonstrate the superior capability of MambaMPD in handling complex and heterogeneous marine scenes. Its strong performance across all three metrics reflects not only accurate segmentation but also robustness in classifying fine-grained and ambiguous targets under real-world oceanic conditions.

Furthermore, we computed the confusion matrix obtained by applying MambaMPD on the MADOS test set, as shown in Fig. 8. Our model shows better performance compared with other models. Although the number of test pixels for oil spill categories is relatively low compared to others, our model still achieves good segmentation performance. Specifically, the F1 scores for small-object classes consistently exceed 70%, reflecting the model’s robustness in addressing class imbalance. Notably, the accuracy for the Ship class reaches 90.4%, indicating that our model exhibits strong discriminative capability for key categories, efectively reducing misclassification.

![](images/1ad621bca477d1870370a288e8d032114a4a6a84686ee12c7561a666135af6b4.jpg)  
Figure 9. Qualitative performance comparison on selected test samples of the MADOS dataset.

## 4.4.2. Quantitative Results on M4D Dataset

We further evaluated and compared the proposed MambaMPD with several existing methods, including Unet [47], LinkNet [82], PSPNet [83], Deeplabv2 [84], Deeplabv2 (msc) [84], Deeplabv3+ [48], YOLOv8-SAM [27], SAM-OIL [27] and TransOilSeg [26] on the M4D dataset.

Table 4 demonstrates that our MambaMPD outperforms other methods in both oil spill IoU and total mIoU. We further compared MambaMPD with models built upon the Segment Anything Model (SAM), specifically YOLO-SAM and SAM-OIL, which leverage prompt-based or detection-driven strategies to address oil spill segmentation. While these methods benefit from SAM’s general-purpose segmentation capability, they lack dedicated mechanisms for modelling structured context or preserving fine-grained boundaries. In contrast, our MambaMPD leverages the selective scanning mechanism of the Mamba architecture to explicitly capture long-range dependencies and improve contextual understanding across sparse oceanic targets. In addition, we compared MambaMPD with mainstream CNN-based segmentation networks. DeepLabV3+ [48] utilises atrous spatial pyramid pooling (ASPP) to encode multi-scale context, while U-Net employs hierarchical skip connections to preserve spatial resolution. Although both achieve reasonable performance, they are constrained by local receptive fields and struggle with long-range semantic modeling. MambaMPD significantly outperforms these baselines, especially in the oil spill category, where it achieves a remarkable 14.82% improvement in Oil Spill IoU over DeepLabV3+ and a 26.36% gain compared to YOLO-SAM. These results clearly demonstrate the advantage of combining state-space modeling with frequency- and edge-aware modules for structure-sensitive marine pollutant detection.

![](images/dd1386493a778a02ada3fe9268a77ac414fbad0b3133e70c7f102e715e350c40.jpg)  
Figure 10. Qualitative performance comparison on the selected test samples of the M4D dataset.

OSDMamba shares the same VSS backbone family and training objective, which makes it the most direct reference for assessing the proposed modules. MambaMPD improves Oil Spill IoU from 65.59% to 68.20% on M4D and F1 from 71.2% to 74.8% on MADOS. The ablation study attributes these gains to the two modules: FAA alone raises Oil Spill IoU by +5.29% over the Mamba baseline (Table 6), and replacing EGA with SE or CBAM attention degrades all boundary-sensitive classes (Table 7). The improvements therefore stem from the frequency-aware and edge-guided designs rather than from model capacity.

To complement the IoU-based evaluation, Table 5 further reports macro-averaged Precision, Recall, and F1-score. MambaMPD attains the highest Precision (81.02%) and Recall (79.41%); its F1-score (78.18%) is second to TransOilSeg (78.99%), while its mIoU remains the highest among all compared methods.

## 4.4.3. Qualitative Analysis

Additional examples on the MADOS test set are shown in Fig. 9. Neither a classical Random Forest classifier nor mainstream encoder–decoder architectures (e.g., U-Net) nor the recent MariNeXt network succeed in recovering satisfactory edge or texture detail from the multi-spectral data: land–sea boundaries are frequently mis-labelled, and objects with blurred contours are often missed. In contrast, MambaMPD faithfully reconstructs subtle high-frequency structures, such as ship wakes and thin pollutant filaments, while simultaneously suppressing noise. This superior visual performance underscores the model’s capacity to exploit edge-aware priors and long-range context, enabling the reconstruction of details that closely match the true spatial distribution of marine pollutants. A visual comparison with all state-of-the-art baselines is presented in Fig. 10. In the majority of evaluated scenes, MambaMPD delivers more accurate and visually coherent delineation of pollution targets, generally outperforming both traditional convolutional and modern Transformer-based encoders. In rows (a) and (b) of Fig. 10, conventional encoder–decoder networks such as PSPNet misclassify large portions of the slick, reflecting their limited capability to resolve fine-scale boundaries. Although earlier deep learning methods better capture overall shape and edge information, they remain susceptible to texture-induced artefacts. By contrast, MambaMPD obtains sharp, artefact-free borders that follow the ground-truth contour with high fidelity, demonstrating the benefit of its wavelet-enhanced Mamba encoder and edgeguided attention. In the cluttered scene highlighted in Fig. 10, all competing models are distracted by background objects and consequently yield false positives. Owing to its stronger discriminative power in visually ambiguous and cluttered scenes, MambaMPD more reliably isolates the pollutant region correctly and preserves background integrity. These observations suggest that our framework can exploit richer structural priors, thereby improving recognition robustness in the tested marine environments.

Table 6. Ablation study of the proposed modules on the M4D dataset.
<table><tr><td></td><td></td><td></td><td></td><td></td><td>FAA SER EGA | SeaSurface (%, ↑) Oil Spill (%, ↑) Look-alike (%, ↑) Ship (%, ↑) Land (%, ↑) | MIoU (%, ↑) Acc (%, ↑)</td><td></td><td></td><td></td><td></td></tr><tr><td>√</td><td></td><td></td><td>94.93</td><td>61.68</td><td>45.14</td><td>45.23</td><td>91.01</td><td>67.51</td><td>95.73</td></tr><tr><td rowspan="4"></td><td></td><td></td><td>95.43</td><td>66.97</td><td>46.09</td><td>37.58</td><td>91.82</td><td>67.58</td><td>96.15</td></tr><tr><td>√</td><td></td><td>95.58</td><td>66.70</td><td>45.51</td><td>37.55</td><td>93.26</td><td>67.72</td><td>96.24</td></tr><tr><td></td><td>√</td><td>96.40</td><td>65.46</td><td>46.70</td><td>38.36</td><td>94.59</td><td>68.30</td><td>97.00</td></tr><tr><td>√</td><td></td><td>96.29</td><td>65.38</td><td>47.62</td><td>43.29</td><td>94.14</td><td>69.34</td><td>96.93</td></tr><tr><td>√</td><td>√</td><td>√</td><td>95.73</td><td>67.36</td><td>47.38</td><td>41.88</td><td>93.18</td><td>69.11</td><td>96.49</td></tr><tr><td>√</td><td></td><td>√</td><td>96.27</td><td>67.04</td><td>45.16</td><td>43.36</td><td>95.57</td><td>69.48</td><td>95.39</td></tr><tr><td>√</td><td>√</td><td>√</td><td>96.30</td><td>68.20</td><td>43.60</td><td>52.36</td><td>93.79</td><td>70.85</td><td>97.21</td></tr></table>

## 4.4.4. Failure Case Analysis

MambaMPD achieves the highest overall mIoU but drops to 43.60% on the Lookalike class, well behind TransOilSeg and SAM-OIL. The frequency-enhanced and edgeguided features push the model to detect dark-patch structures aggressively: this raises Oil Spill IoU to 68.20%, the best among all methods, but also misclassifies lookalike regions such as biogenic films and low-wind areas as Oil Spill. Fig. 11 shows a representative case. The darkest and most homogeneous segment of an elongated look-alike formation, where the low-backscatter signature of oil is locally strongest, is predicted as Oil Spill; the remainder of the formation is correctly identified. Separating such segments requires context beyond local backscatter, such as shape regularity or wind conditions, which the current features do not capture.

TransOilSeg exhibits the reverse trade-of, leading on Look-alike but falling to 61.38% on Oil Spill, so the two classes currently sit on a recall–precision frontier rather than being jointly solved by either design. Future work will explore class-aware contrastive learning and look-alike-specific auxiliary losses to better separate these visually similar categories.

## 4.5. Ablation Study

To thoroughly evaluate the efectiveness of each proposed module in our architecture, we conduct extensive ablation experiments on the M4D dataset. The baseline model is a MambaU-Net consisting of a VSS encoder with {2, 2, 9, 2} blocks and a standard UnetrUpBlock decoder, without any of the proposed modules (FAA, SER, or EGA). Each module is then individually or jointly added to this Mamba-based baseline to assess its contribution. The results are shown in Table 6. The investigated modules include: the Frequency-Aware Augmentation Module (FAA) for frequency-aware feature extraction, the SE-ResDecoder (SER) for semantic enrichment, and the Edge-Guided Attention (EGA) for fine boundary preservation. We also analyse the role of their combinations and the cumulative efect when all modules are used together.

## 4.5.1. Efect of Single Modules

Without applying FAA, SER, or EGA, the baseline MambaU-Net achieves an Oil IoU of 61.68% and an mIoU of 67.51%. This setting serves as the reference point for evaluating the contribution of each individual module.

When only the Frequency-Aware Augmentation (FAA) module is added, the Oil IoU improves significantly from 61.68% to 66.97%, and the mIoU rises from 67.51% to 67.58%. This confirms that FAA enhances the model’s ability to extract discriminative multi-frequency representations, which are crucial for capturing subtle structural, textural, and contextual diferences between marine pollutants and ocean backgrounds.

Introducing only the SE-ResDecoder (SER) module yields an Oil IoU of 66.70% and an mIoU of 67.72%. The improvement compared with the baseline demonstrates that SER efectively enriches geometric and structural cues, enabling better discrimination of irregular pollutant shapes within SAR imagery.

Using only the Edge-Guided Attention (EGA) module increases Oil IoU to 65.46% and mIoU to 68.30%, representing the strongest single-module enhancement. This verifies that explicitly modeling boundary priors greatly benefits marine pollution segmentation by mitigating the edge-blurring tendency of SSM-based architectures.

## 4.5.2. Efect of Dual Module Combinations

We then evaluate the synergy between two modules. The combination of FAA and SER improves the segmentation accuracy significantly, yielding an oil spill IoU of 65.38% and an mIoU of 69.34%. This suggests that frequency-aware and semantic-enhancing modules complement each other by capturing global context while preserving classspecific features. Combining FAA and EGA (without SER) also achieves strong results, with an oil spill IoU of 67.04% and mIoU of 69.48%, indicating that frequency and boundary information together help the model to suppress noise and highlight transition regions. On the other hand, the SER + EGA setting (without FAA) results in 67.36% oil spill IoU and 69.11% mIoU, which is inferior to the configurations involving FAA, highlighting the foundational role of frequency-aware encoding in marine pollution scenarios.

## 4.5.3. Full Model with All Modules

Finally, when all three modules (FAA + SER + EGA) are integrated, the model achieves the best performance across nearly all categories. Specifically, it obtains an oil spill IoU of 68.20%, an overall mIoU of 70.85%, and a pixel-level accuracy of 97.21%. This configuration shows superior performance not only in the oil spill class but also across the Sea Surface and Ship classes, suggesting that each module contributes uniquely and efectively to the final representation. These results clearly demonstrate that the three modules provide complementary advantages: FAA improves global frequency perception, SER enhances semantic discriminability for rare categories, and EGA refines boundaries and edge structures. Their joint efect enables the model to capture both global context and fine-grained details, which are critical for robust oil spill segmentation under real-world, complex marine conditions.

Table 7. Sensitivity of decoder attention strategies on the M4D dataset. Backbone (VSS {2, 2, 9, 2}), FAA, and SE-ResDecoder remain fixed; only the attention module is replaced.
<table><tr><td>Attention type</td><td>Oil(%)</td><td>Look-alike(%) mIoU(%)</td><td></td></tr><tr><td>None (no att.)</td><td>65.38</td><td>47.62</td><td>69.34</td></tr><tr><td>SE</td><td>69.44</td><td>41.82</td><td>69.88</td></tr><tr><td>CBAM</td><td>69.87</td><td>42.10</td><td>70.12</td></tr><tr><td>EGA (ours)</td><td>68.20</td><td>43.60</td><td>70.85</td></tr></table>

![](images/c33a51c1c9789abef10d022d403673ad0108e73cdbf59be4aaf1d4483c8d033a.jpg)  
Figure 11. Representative failure case on the M4D test set. SAR image containing an elongated dark formation; Label, in which the entire formation is annotated as Look-alike; MambaMPD prediction. Most of the formation is correctly identified, but its darkest and most homogeneous segment is labelled as Oil Spill: locally this segment carries the low-backscatter signature that the frequency- and edge-enhanced features amplify.

## 5. Further Analysis

To more comprehensively assess the robustness and eficiency of the proposed MambaMPD framework, we conduct additional analyses from three complementary perspectives: architectural sensitivity to wavelet configurations in the FAA module, decoder attention strategies, and overall computational complexity. These studies provide deeper insights into why MambaMPD exhibits strong boundary modelling capability while maintaining its lightweight nature.

## 5.1. Sensitivity to Wavelet Configuration in FAA

The FAA module leverages multi–level discrete wavelet transform (DWT) to extract high–frequency components that are critical for modelling thin oil films and fragmented look–alike structures. As wavelet-based frequency decomposition may, in principle, be influenced by the choice of wavelet settings, we evaluate the sensitivity of FAA to two key hyperparameters: (i) the wavelet basis and (ii) the decomposition level J. All experiments are performed on the M4D dataset using the standard VSS backbone configuration {2, 2, 9, 2} and the same training protocol as in the main study. As shown

Table 8. Sensitivity of the FAA module to diferent wavelet bases. The decomposition level is fixed to J = 2. Results are reported on both M4D and MADOS to verify cross-modality robustness.
<table><tr><td colspan="5">M4D MADOS</td></tr><tr><td>Wavelet basis</td><td>OA</td><td>mIoU</td><td>Oil IoU</td><td>OA</td><td>mIoU</td><td>F1</td></tr><tr><td>Haar</td><td>97.11</td><td>70.78</td><td>67.91</td><td>83.22</td><td>69.73</td><td>74.15</td></tr><tr><td>Daubechies-2 (db2)</td><td>97.64</td><td>70.53</td><td>67.01</td><td>82.75</td><td>69.11</td><td>74.69</td></tr><tr><td>Daubechies-4 (db4)</td><td>97.65</td><td>70.27</td><td>67.69</td><td>83.34</td><td>69.60</td><td>74.71</td></tr></table>

Table 9. Sensitivity of the FAA module to the wavelet decomposition level J. The wavelet basis is fixed to Haar. Results are reported on both M4D and MADOS.
<table><tr><td colspan="5">M4D MADOS</td></tr><tr><td>Level J OA</td><td>mIoU</td><td>Oil IoU</td><td>OA</td><td>mIoU</td><td>F1</td></tr><tr><td>J = 1</td><td>97.63</td><td>70.16</td><td>67.44</td><td>83.17</td><td>68.93 74.68</td></tr><tr><td>J = 2</td><td>97.11</td><td>70.78</td><td>67.91 83.22</td><td>69.73</td><td>74.15</td></tr><tr><td>J = 3</td><td>97.29</td><td>70.14 67.42</td><td>83.47</td><td>69.06</td><td>74.34</td></tr></table>

in Table 8 and Table 9.

Wavelet basis. We fix the decomposition level to J = 2 and examine three commonly used wavelets: Haar, Daubechies-2 (db2), and Daubechies-4 (db4). As summarised in Table 8, the resulting diferences in OA, mIoU, and oil–spill IoU are minimal, with mIoU fluctuating within 0.51% and Oil IoU within 0.90%. This demonstrates that the FAA is largely insensitive to the specific wavelet family chosen, confirming that the gains of MambaMPD do not depend on a finely tuned handcrafted basis. Decomposition level. Next, we vary the decomposition depth $J \in \{ 1 , 2 , 3 \}$ while fixing the wavelet basis to Haar (Table 9). A shallow single-level decomposition $( J = 1 )$ results in a slightly reduced mIoU, indicating that insuficient frequency resolution weakens multi–scale structure modelling. Increasing the depth to $J = 2$ yields the best performance, while J = 3 produces a marginal decline due to over-decomposition and loss of fine spatial details. Importantly, the performance remains stable across all settings, highlighting the robustness of FAA to reasonable changes in its wavelet configuration.

## 5.2. Sensitivity to Attention Strategies

To assess whether the benefits of EGA depend on a particular attention form, we compare EGA against three alternatives while keeping the VSS backbone, FAA, and SE-ResDecoder fixed: (i) no attention, (ii) SE-based channel attention, and (iii) CBAM, a widely used spatial–channel attention module. As shown in Table 7, adding attention generally improves performance over the plain skip-connection baseline. However, EGA achieves the best overall mIoU, the most boundary-ambiguous class; although SE and CBAM attain slightly higher Oil IoU, EGA delivers the strongest overall performance by markedly improving this harder class. The improvements are particularly pronounced in boundary-sensitive categories, demonstrating that explicit integration of multi–scale edge cues provides more efective structural guidance than generic at-

Table 10. Module-wise computational overhead analysis. Starting from the baseline MambaU-Net, each proposed module is cumulatively added to quantify its impact on parameters, FLOPs, and inference speed.
<table><tr><td>Model</td><td>Params (M)</td><td>FLOPs (G)</td><td>FPS</td></tr><tr><td>Baseline MambaU-Net</td><td>36.59</td><td>164.21</td><td>17.19</td></tr><tr><td>+ FAA</td><td>36.89</td><td>185.61</td><td>16.17</td></tr><tr><td>+ EGA</td><td>36.77</td><td>182.89</td><td>15.40</td></tr><tr><td>+ FAA + EGA (Full)</td><td>37.06</td><td>204.29</td><td>15.19</td></tr></table>

tention mechanisms.

Moreover, the overall variation across attention strategies remains small, confirming that MambaMPD is not overly sensitive to the attention formulation. The superior performance of EGA arises from its edge-guided reweighting rather than reliance on a fragile hyperparameter setup.

## 5.3. Computational Complexity

To quantify the overhead introduced by each module, we measure trainable parameters, FLOPs, and inference speed. FLOPs are computed using ptflops under a 1250 × 650 input resolution, and throughput is measured as the average over 100 forward passes on an NVIDIA RTX 4070Ti GPU. Results are reported in Table 10.

FAA adds 0.30 M parameters and 21.40 G FLOPs, reducing throughput from 17.19 to 16.17 FPS. EGA adds 0.18 M parameters and 18.68 G FLOPs. With both modules active, the full MambaMPD model contains 37.06 M parameters and 204.29 G FLOPs—an increase of only 0.47 M parameters and 40.08 G FLOPs over the baseline, while throughput remains at 15.19 FPS. By comparison, foundation-model-based approaches such as SAM-OIL carry over 600 M parameters, placing MambaMPD at roughly 1/16 of that capacity. These figures confirm that the performance gains stem from the structural design of the frequency-aware and edge-guided modules rather than from scaling model size.

All current experiments use fixed-size patches; scaling to sliding-window inference over full satellite swaths is left to future work.

## 6. Conclusion

This paper presented MambaMPD, a segmentation framework for marine pollution detection from remote sensing imagery. MambaMPD builds on selective state-space modelling and introduces two problem-driven modules: Frequency-Aware Augmentation (FAA) and Edge-Guided Attention (EGA). Mamba ofers linear-complexity sequence modelling but is weak at capturing frequency structure and preserving edges; FAA compensates through hierarchical wavelet decomposition that survives spatial downsampling, while EGA injects multi-scale Laplacian edge priors into the decoder, together addressing the low contrast and boundary ambiguity of marine pollutants in SAR and multispectral imagery.

Relative to prior work, MambaMPD difers from CNN and Transformer-based detectors, which are limited by local receptive fields, and from foundation-model-based approaches, which carry hundreds of millions of parameters. To our knowledge, it is also the first Mamba-based framework designed for multi-class marine pollution detection across both SAR and multispectral modalities, rather than binary oil-spill segmentation in a single modality. This positioning translates into three practical advantages. First, it attains the best mIoU among the compared methods on both benchmarks, improving Oil Spill IoU by 6.82% over TransOilSeg on M4D and F1 by 3.6% over OSDMamba on MADOS. Second, these gains come at low cost: the full model adds only 0.47 M parameters over the baseline and runs faster than the strongest baselines, TransOilSeg and OSDMamba, making it suitable for single-GPU deployment. Third, its modality-agnostic design transfers between SAR and optical pipelines without architectural changes. These properties make MambaMPD a candidate for operational settings such as oil-spill emergency response and routine coastal surveillance, and a practical tool for maritime safety agencies, environmental monitoring organisations, and coastal management authorities.

Several limitations remain. Look-alike discrimination remains imperfect: the learned feature space does not yet separate pollutants from visually similar backgrounds well enough. As analysed in Section 4.4.4, this reflects a deliberate emphasis on recall: a missed spill delays emergency response and its cost grows with time, whereas false alarms are filtered during routine human verification of alerts, so high recall on actual pollutants is prioritised in operational monitoring [4, 63]. Future work will address these gaps through joint SAR–optical fusion, class-aware contrastive learning to suppress look-alike confusion, and sliding-window inference over full satellite swaths. As suitable annotated benchmarks become available, the framework can further be extended to additional pollutant types, such as chemical contaminants and microplastics, supporting more reliable and comprehensive marine pollution assessment.

## 7. Data Availability Statement

The datasets used in this study are publicly available. The MADOS (Marine Debris and Oil Spill) dataset [86] is openly available on Zenodo at https://doi.org/ 10.5281/zenodo.10664073. The Oil Spill Detection Dataset (M4D) [7] was originally provided by the Multimodal Data Fusion and Analytics Group at the Centre for Research and Technology Hellas (CERTH). All processed data used in the experiments of this study have been deposited on Zenodo and are publicly accessible at https://doi.org/10.5281/zenodo.19386441 [87]. The source code of the proposed MambaMPD framework is publicly available at https://github.com/ Multimodal-Intelligence-Lab-MIL/MambaMPD [88] under the CC0 v1.0 licence.

## 8. Funding

This work was supported in part by China Scholarship Council and in part by the University of Exeter Ph.D. Scholarships.

## 9. Disclosure Statement

No potential conflict of interest was reported by the author(s).

## References

[1] Saeid Dehghani-Dehcheshmeh, Mehdi Akhoondzadeh, and Saeid Homayouni. Oil spills detection from sar earth observations based on a hybrid cnn transformer networks. Marine Pollution Bulletin, 190:114834, 2023.

[2] Jingwu Ma, Renfeng Ma, Qi Pan, Xianjun Liang, Jianqing Wang, and Xinxin Ni. A global review of progress in remote sensing and monitoring of marine pollution. Water, 15(19): 3491, 2023.

[3] Matthew J McCarthy, Hannah V Herrero, Stephanie A Insalaco, Melissa T Hinten, and Assaf Anyamba. Satellite remote sensing for environmental sustainable development goals: A review of applications for terrestrial and marine protected areas. Remote Sensing Applications: Society and Environment, page 101450, 2025.

[4] Anne HS Solberg, Camilla Brekke, and Per Ove Husoy. Oil spill detection in radarsat and envisat sar images. IEEE Transactions on Geoscience and Remote Sensing, 45(3): 746–755, 2007.

[5] Ferdenant A Mkrtchyan and Costas A Varotsos. A new monitoring system for the surface marine anomalies. Water, Air, & Soil Pollution, 229(8):273, 2018. doi: 10.1007/s11270-018-3938-3.

[6] Costas A Varotsos, Vladimir F Krapivin, and Ferdenant A Mkrtchyan. New optical tools for water quality diagnostics. Water, Air, & Soil Pollution, 230(8):177, 2019. doi: 10.1007/s11270-019-4228-4.

[7] Marios Krestenitis, Georgios Orfanidis, Konstantinos Ioannidis, Konstantinos Avgerinakis, Stefanos Vrochidis, and Ioannis Kompatsiaris. Oil spill identification from satellite images using deep neural networks. Remote Sensing, 11(15):1762, 2019.

[8] Katerina Kikaki, Ioannis Kakogeorgiou, Ibrahim Hoteit, and Konstantinos Karantzalos. Detecting marine pollutants and sea surface features with deep learning in sentinel-2 imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 210:39–54, 2024.

[9] Alexandra Cernian. Advances in detecting and identifying marine plastic waste: A comprehensive review. Journal of Coastal Research, 113(SI):1076–1081, 2025.

[10] Zhipeng Wan, Sheng Wang, Wei Han, Yuewei Wang, Xiaohui Huang, Xiaohan Zhang, Xiaodao Chen, and Yunliang Chen. A systematic survey and meta-analysis of the segment anything model in remote sensing image processing: Challenges, advances, applications, and opportunities. ISPRS Journal of Photogrammetry and Remote Sensing, 229:436–466, 2025.

[11] Xiaodao Chen, Yupeng Liu, Wei Han, Xiongwei Zheng, Sheng Wang, Jun Wang, and Lizhe Wang. A vision-language foundation model-based multi-modal retrieval-augmented generation framework for remote sensing lithological recognition. ISPRS Journal of Photogrammetry and Remote Sensing, 225:328–340, 2025.

[12] Costas A Varotsos and Arthur P Cracknell. Remote Sensing Letters contribution to the success of the Sustainable Development Goals-UN 2030 agenda. Remote Sensing Letters, 11(8):715–719, 2020. doi: 10.1080/2150704X.2020.1753338.

[13] Brian R Silliman, Philip M Dixon, Cameron Wobus, Qiang He, Pedro Daleo, Brent B Hughes, Matthew Rissing, Jonathan M Willis, and Mark W Hester. Thresholds in marsh resilience to the deepwater horizon oil spill. Scientific Reports, 6(1):32520, 2016.

[14] Sharon E Hook. Beyond thresholds: A holistic approach to impact assessment is needed to enable accurate predictions of environmental risk from oil spills. Integrated Environmental Assessment and Management, 16(6):813–830, 2020.

[15] Jin Xu, Haixia Wang, Can Cui, Baigang Zhao, and Bo Li. Oil spill monitoring of shipborne radar image features using svm and local adaptive threshold. Algorithms, 13(3):69, 2020.

[16] Tong Yang. Dynamic assessment of environmental damage based on the optimal clustering criterion–taking oil spill damage to marine ecological environment as an example. Ecological Indicators, 51:53–58, 2015.

[17] Giacomo Capizzi, Grazia Lo Sciuto, Marcin Wo´zniak, and Robertas Damaˇsevicius. A clustering based system for automated oil spill detection by satellite remote sensing. In

Artificial Intelligence and Soft Computing: 15th International Conference, ICAISC 2016, Zakopane, Poland, June 12-16, 2016, Proceedings, Part II 15, pages 613–623. Springer, 2016.

[18] Miguel Moctezuma, Flavio Parmiggiani, and Ludwin Lopez Lopez. Measuring marine oil spill extent by markov random fields. In Remote Sensing of the Ocean, Sea Ice, Coastal Waters, and Large Water Regions 2014, volume 9240, pages 57–62. SPIE, 2014.

[19] Linlin Xu, M Javad Shafiee, Alex Wong, Fan Li, Lei Wang, and David Clausi. Oil spill candidate detection from sar imagery using a thresholding-guided stochastic fully-connected conditional random field model. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 79–86, 2015.

[20] Ludwin Lopez, Miguel Moctezuma, and Flavio Parmiggiani. Contextual approach for oil spill detection in sar images using image fusion and markov random fields. In 2006 49th IEEE International Midwest Symposium on Circuits and Systems, volume 2, pages 137–139. IEEE, 2006.

[21] Amira S Mahmoud, Sayed A Mohamed, Reda A El-Khoriby, Hisham M AbdelSalam, and Ihab A El-Khodary. Oil spill identification based on dual attention unet model using synthetic aperture radar images. Journal of the Indian Society of Remote Sensing, 51(1): 121–133, 2023.

[22] Chunshan Li, Mingzhi Wang, Xiaofei Yang, and Dianhui Chu. Ds-unet: Dual-stream unet for oil spill detection of sar image. IEEE Geoscience and Remote Sensing Letters, 20: 1–5, 2023.

[23] Hai-Feng Zhong, Qing Sun, Hong-Mei Sun, and Rui-Sheng Jia. Nt-net: A semantic segmentation network for extracting lake water bodies from optical remote sensing images based on transformer. IEEE Transactions on Geoscience and Remote Sensing, 60:1–13, 2022.

[24] Yuheng Liu, Yifan Zhang, Ye Wang, and Shaohui Mei. Rethinking transformers for semantic segmentation of remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 61:1–15, 2023.

[25] Cheng Zhang, Wanshou Jiang, Yuan Zhang, Wei Wang, Qing Zhao, and Chenjie Wang. Transformer and cnn hybrid deep neural network for semantic segmentation of veryhigh-resolution remote sensing imagery. IEEE Transactions on Geoscience and Remote Sensing, 60:1–20, 2022.

[26] Yu Chai, Xinhai Han, Yiqi Wang, Dan Luo, Jingsong Yang, Peng Chen, and Gang Zheng. Transoilseg: A novel sar oil spill detection method addressing data limitations and lookalike confusions. IEEE Transactions on Geoscience and Remote Sensing, 2025.

[27] Wenhui Wu, Man Sing Wong, Xinyu Yu, Guoqiang Shi, Coco Yin Tung Kwok, and Kang Zou. Compositional oil spill detection based on object detector and adapted segment anything model from sar images. IEEE Geoscience and Remote Sensing Letters, 2024.

[28] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

[29] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. Vmamba: Visual state space model. Advances in neural information processing systems, 37:103031–103063, 2024.

[30] Jiarun Liu, Hao Yang, Hong-Yu Zhou, Yan Xi, Lequan Yu, Cheng Li, Yong Liang, Guangming Shi, Yizhou Yu, Shaoting Zhang, et al. Swin-umamba: Mamba-based unet with imagenet-based pretraining. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 615–625. Springer, 2024.

[31] Keyan Chen, Bowen Chen, Chenyang Liu, Wenyuan Li, Zhengxia Zou, and Zhenwei Shi. Rsmamba: Remote sensing image classification with state space model. IEEE Geoscience and Remote Sensing Letters, 2024.

[32] Rami Al-Ruzouq, Mohamed Barakat A Gibril, Abdallah Shanableh, Abubakir Kais, Osman Hamed, Saeed Al-Mansoori, and Mohamad Ali Khalil. Sensors, features, and machine learning for oil spill detection and monitoring: A review. Remote Sensing, 12(20):3338, 2020.

[33] Srikanta Sannigrahi, Bidroha Basu, Arunima Sarkar Basu, and Francesco Pilla. Development of automated marine floating plastic detection system using sentinel-2 imagery and machine learning models. Marine Pollution Bulletin, 178:113527, 2022.

[34] Alex Sol´e G´omez, Leonardo Scandolo, and Elmar Eisemann. A learning approach for river <sup>\`</sup> debris detection. International Journal ofApplied Earth Observation and Geoinformation, 107:102682, 2022.

[35] Henning Heiselberg. Ship-iceberg classification in sar and multispectral satellite images with neural networks. Remote Sensing, 12(15):2353, 2020.

[36] Peng Liu. A survey of remote-sensing big data. frontiers in Environmental Science, 3:45, 2015.

[37] Shuaiyu Chen, Fu Wang, Peng Ren, Chunbo Luo, and Zeyu Fu. Osdmamba: Enhancing oil spill detection from remote sensing images using selective state-space model. IEEE Geoscience and Remote Sensing Letters, 22:1–5, 2025. doi: 10.1109/LGRS.2025.3583965.

[38] Costas A Varotsos and Vladimir F Krapivin. Pollution of Arctic waters has reached a critical point: An innovative approach to this problem. Water, Air, & Soil Pollution, 229 (11):343, 2018. doi: 10.1007/s11270-018-4004-x.

[39] Costas A Varotsos, Vladimir F Krapivin, Ferdenant A Mkrtchyan, Suren A Gevorkyan, and Tengfei Cui. A novel approach to monitoring the quality of lakes water by optical and modeling tools: Lake Sevan as a case study. Water, Air, & Soil Pollution, 231(8): 435, 2020. doi: 10.1007/s11270-020-04792-8.

[40] Chuanmin Hu, Lin Qi, Yuyuan Xie, Shuai Zhang, and Brian B Barnes. Spectral characteristics of sea snot reflectance observed from satellites: Implications for remote sensing of marine debris. Remote Sensing of Environment, 269:112842, 2022.

[41] Lin Qi, Menghua Wang, Chuanmin Hu, and Benjamin Holt. On the capacity of sentinel-1 synthetic aperture radar in detecting floating macroalgae and other floating matters. Remote Sensing of Environment, 280:113188, 2022.

[42] Katerina Kikaki, Ioannis Kakogeorgiou, Paraskevi Mikeli, Dionysios E Raitsos, and Konstantinos Karantzalos. Marida: A benchmark for marine debris detection from sentinel-2 remote sensing data. PloS one, 17(1):e0262247, 2022.

[43] Jamila Mifdal, Nicolas Long´ep´e, and Marc Rußwurm. Towards detecting floating objects on a global scale with learned spatial features using sentinel 2. ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 3:285–293, 2021.

[44] Paraskevi Mikeli. Sentinel-2 satellite multispectral data analysis of marine debris and other characteristics on the sea surface. Master’s thesis, National Technical University of Athens, 2022.

[45] Heidi M Dierssen. Hyperspectral measurements, parameterizations, and atmospheric correction of whitecaps and foam from visible to shortwave infrared for ocean color remote sensing. Frontiers in Earth Science, 7:14, 2019.

[46] Shungudzemwoyo P Garaba and Heidi M Dierssen. An airborne remote sensing case study of synthetic hydrocarbon detection using short wave infrared absorption features identified from marine-harvested macro-and microplastics. Remote Sensing of Environment, 205: 224–235, 2018.

[47] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.

[48] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schrof, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 801–818, 2018.

[49] Xiangtai Li, Henghui Ding, Haobo Yuan, Wenwei Zhang, Jiangmiao Pang, Guangliang Cheng, Kai Chen, Ziwei Liu, and Chen Change Loy. Transformer-based visual segmentation: A survey. IEEE transactions on pattern analysis and machine intelligence, 2024.

[50] Ioannis Kotaridis and Maria Lazaridou. Remote sensing image segmentation advances: A meta-analysis. ISPRS Journal of Photogrammetry and Remote Sensing, 173:309–322,

2021.

[51] Vivek Dey, Yun Zhang, and Ming Zhong. A review on image segmentation techniques with remote sensing perspective, volume 38. na Vienna, Austria, 2010.

[52] Qiqi Zhu, Yanan Zhang, Ziqi Li, Xiaorui Yan, Qingfeng Guan, Yanfei Zhong, Liangpei Zhang, and Deren Li. Oil spill contextual and boundary-supervised detection network based on marine sar images. IEEE Transactions on Geoscience and Remote Sensing, 60: 1–10, 2022. doi: 10.1109/TGRS.2021.3115492.

[53] Fang Chen, Heiko Balzter, Feixiang Zhou, Peng Ren, and Huiyu Zhou. Dgnet: Distribution guided eficient learning for oil spill image segmentation. IEEE Transactions on Geoscience and Remote Sensing, 61:1–17, 2023. doi: 10.1109/TGRS.2023.3240579.

[54] Fang Chen, Heiko Balzter, Peng Ren, and Huiyu Zhou. Srcnet: Seminal image representation collaborative network for oil spill segmentation in sar imagery. IEEE Transactions on Geoscience and Remote Sensing, 62:1–18, 2024. doi: 10.1109/TGRS.2024.3463404.

[55] Miguel M Duarte and Leonardo Azevedo. Automatic detection and identification of floating marine debris using multispectral satellite imagery. IEEE Transactions on Geoscience and Remote Sensing, 61:1–15, 2023.

[56] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, Victor Villena-Martinez, and Jose Garcia-Rodriguez. A review on deep learning techniques applied to semantic segmentation. arXiv preprint arXiv:1704.06857, 2017.

[57] Shijie Hao, Yuan Zhou, and Yanrong Guo. A brief survey on semantic segmentation with deep learning. Neurocomputing, 406:302–321, 2020.

[58] Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, and Demetri Terzopoulos. Image segmentation using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(7):3523–3542, 2021.

[59] Sijie Zhao, Hao Chen, Xueliang Zhang, Pengfeng Xiao, Lei Bai, and Wanli Ouyang. Rsmamba: Remote sensing mamba for large remote sensing image dense prediction. arXiv preprint arXiv:2404.02668, 2024.

[60] Xianping Ma, Xiaokang Zhang, and Man-On Pun. Rs3mamba: Visual state space model for remote sensing images semantic segmentation. arXiv preprint arXiv:2404.02457, 2024.

[61] Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, and Naoto Yokoya. Changemamba: Remote sensing change detection with spatiotemporal state space model. IEEE Transactions on Geoscience and Remote Sensing, 62:1–20, 2024. doi: 10.1109/TGRS. 2024.3417253.

[62] Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, Ge Cui, and Lei Li. Mamba-unet: Unet-like pure visual mamba for medical image segmentation. arXiv preprint arXiv:2402.05079, 2024.

[63] Werner Alpers, Benjamin Holt, and Kan Zeng. Oil spill detection by imaging radars: Challenges and pitfalls. Remote sensing of environment, 201:133–147, 2017.

[64] Pengju Liu, Hongzhi Zhang, Kai Zhang, Liang Lin, and Wangmeng Zuo. Multi-level wavelet-cnn for image restoration. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 773–782, 2018.

[65] Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, and Qinghua Hu. Eca-net: Eficient channel attention for deep convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11534– 11542, 2020.

[66] Leo Breiman. Random forests. Machine learning, 45:5–32, 2001.

[67] Yuliya V Karpievitch, Elizabeth G Hill, Anthony P Leclerc, Alan R Dabney, and Jonas S Almeida. Rf++: Generalized random forest-based classifier for cluster-correlated data. The Annals of Applied Statistics, 3(2):812–838, 2009.

[68] Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zhengning Liu, Ming-Ming Cheng, and Shi-Min Hu. Segnext: Rethinking convolutional attention design for semantic segmentation. Advances in Neural Information Processing Systems, 35:1140–1156, 2022.

[69] Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J. Stewart, Jo¨elle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand Le Saux, Gustau Camps-Valls, and Xiao Xiang Zhu. Neural

plasticity-inspired multimodal foundation model for earth observation, 2025. URL https: //arxiv.org/abs/2403.15356.

[70] Siqi Lu, Junlin Guo, James R Zimmer-Dauphinee, Jordan M Nieusma, Xiao Wang, Steven A Wernke, Yuankai Huo, et al. Vision foundation models in remote sensing: A survey. IEEE Geoscience and Remote Sensing Magazine, 2025.

[71] Daniela Szwarcman, Sujit Roy, Paolo Fraccaro, Thorsteinn El´ı G´ıslason, Benedikt Blumenstiel, Rinki Ghosal, Pedro Henrique de Oliveira, Joao Lucas de Sousa Almeida, Rocco Sedona, Yanghui Kang, et al. Prithvi-eo-2.0: A versatile multi-temporal foundation model for earth observation applications. arXiv preprint arXiv:2412.02732, 2024.

[72] Fan Liu, Delong Chen, Zhangqingyun Guan, Xiaocong Zhou, Jiale Zhu, Qiaolin Ye, Liyong Fu, and Jun Zhou. Remoteclip: A vision language foundation model for remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 62:1–16, 2024.

[73] Favyen Bastani, Piper Wolters, Ritwik Gupta, Joe Ferdinando, and Aniruddha Kembhavi. Satlaspretrain: A large-scale dataset for remote sensing image understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16772–16782, 2023.

[74] Colorado J Reed, Ritwik Gupta, Shufan Li, Sarah Brockman, Christopher Funk, Brian Clipp, Kurt Keutzer, Salvatore Candido, Matt Uyttendaele, and Trevor Darrell. Scalemae: A scale-aware masked autoencoder for multiscale geospatial representation learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4088–4099, 2023.

[75] Danfeng Hong, Bing Zhang, Xuyang Li, Yuxuan Li, Chenyu Li, Jing Yao, Naoto Yokoya, Hao Li, Pedram Ghamisi, Xiuping Jia, et al. Spectralgpt: Spectral remote sensing foundation model. arXiv preprint arXiv:2311.07113, 2023.

[76] Yi Wang, Nassim Ait Ali Braham, Zhitong Xiong, Chenying Liu, Conrad M Albrecht, and Xiao Xiang Zhu. Ssl4eo-s12: A large-scale multimodal, multitemporal dataset for self-supervised learning in earth observation [software and data sets]. IEEE Geoscience and Remote Sensing Magazine, 11(3):98–106, 2023.

[77] Benedikt Blumenstiel, Paolo Fraccaro, Valerio Marsocci, Johannes Jakubik, Stefano Maurogiovanni, Mikolaj Czerkawski, Rocco Sedona, Gabriele Cavallaro, Thomas Brunschwiler, Juan Bernabe Moreno, et al. Terramesh: A planetary mosaic of multimodal earth observation data. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2394–2402, 2025.

[78] Valerio Marsocci, Yuru Jia, Georges Le Bellier, David Kerekes, Liang Zeng, Sebastian Hafner, Sebastian Gerard, Eric Brune, Ritu Yadav, Ali Shibli, Heng Fang, Yifang Ban, Maarten Vergauwen, Nicolas Audebert, and Andrea Nascetti. Pangaea: A global and inclusive benchmark for geospatial foundation models, 2025. URL https://arxiv.org/ abs/2412.04204.

[79] Quinten Vanhellemont and Kevin Ruddick. Atmospheric correction of metre-scale optical satellite data for inland and coastal water applications. Remote sensing of environment, 216:586–597, 2018.

[80] Abhishek Ramanathapura Satyanarayana and Maruf A Dhali. Oil spill segmentation using deep encoder-decoder models. arXiv preprint arXiv:2305.01386, 2023.

[81] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 248–255. IEEE, June 2009.

[82] Abhishek Chaurasia and Eugenio Culurciello. Linknet: Exploiting encoder representations for eficient semantic segmentation. In 2017 IEEE visual communications and image processing (VCIP), pages 1–4. IEEE, 2017.

[83] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2881–2890, 2017.

[84] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous con-

volution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence, 40(4):834–848, 2017.

[85] Anthony Fuller, Koreen Millard, and James Green. Croma: Remote sensing representations with contrastive radar-optical masked autoencoders. Advances in Neural Information Processing Systems, 36:5506–5538, 2023.

[86] Katerina Kikaki, Ioannis Kakogeorgiou, Ibrahim Hoteit, and Konstantinos Karantzalos. MADOS - Marine Debris and Oil Spill dataset, 2024. URL https://doi.org/10.5281/ zenodo.10664073.

[87] Marios Krestenitis, Georgios Orfanidis, Konstantinos Ioannidis, Konstantinos Avgerinakis, Stefanos Vrochidis, and Ioannis Kompatsiaris. Oil Spill Detection Dataset (M4D), 2019. URL https://doi.org/10.5281/zenodo.19386441.

[88] Shuaiyu Chen, Wei Han, Peng Ren, Chunbo Luo, and Zeyu Fu. MambaMPD: Source code for marine pollution detection. https://github.com/ Multimodal-Intelligence-Lab-MIL/MambaMPD, 2026. Licensed under CC0-1.0.