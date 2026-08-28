# FU-Mamba: A Frequency-Enhanced Dynamic Scanning Framework for Oralscan Image Segmentation

Xinxin Zhao<sup>a</sup>, Jinpeng Ye<sup>a</sup>, Bo Wei<sup>a</sup>, Liqin Wu<sup>b</sup>, Mahmoud Hassaballah<sup>c,j</sup>, Karen Egiazarian<sup>d</sup>, Aura Conci<sup>e</sup>, Victor Hugo C. de Albuquerque<sup>f</sup>, Abdulkadir Sengur<sup>g</sup>, Leszek Rutkowski<sup>h,i</sup>, Yan Tian<sup>a,∗</sup>

<sup>a</sup>School of Computer Science and Technology, Zhejiang Gongshang University, 310018, Hangzhou, China

<sup>b</sup>Department of Stomatology, Tongxiang Hospital of Traditional Chinese Medicine, 314500, Tongxiang, China

<sup>c</sup>Department of Computer Science, Prince Sattam Bin Abdulaziz University, 16278, AlKharj, Saudi Arabia

<sup>d</sup>Department of Computing Sciences, Tampere University, 33720, Tampere, Finland <sup>e</sup>Department of Computer Science, Universidade Federal Fluminense, 24210-346, Niteroi, Brazil

<sup>f</sup>Department of Teleinformatics Engineering, Federal University of Ceara, 60020-181, Fortaleza, Brazil

<sup>g</sup>Department of Electrical and Electronic Engineering, Faculty of Technology, Firat University, 23000, Elazig, Turkey

<sup>h</sup>Systems Research Institute, Polish Academy of Sciences, 01-447, Warsaw, Poland <sup>i</sup>AGH University of Krakow, 30-059, Krakow, Poland

<sup>j</sup>Department of Computer Science, Qena University, 83523, Qena, Egypt

## Abstract

Oralscan image segmentation is essential for computer-aided diagnosis and treatment planning in digital dentistry. However, existing visual state space models (SSMs) often rely on manually designed scanning orders to flatten image patches into sequences, which disrupts the semantic spatial continuity and hinders coherent feature extraction from key foreground regions. Moreover, elements such as inconsistent lighting, reflective surfaces, and noise during data acquisition disrupt the frequency distribution by diminishing high-frequency details while enhancing low-frequency components, consequently hindering the accurate localization of boundaries. In response to these challenges, we introduce FU-Mamba, an innovative framework that incorporates dynamic scanning and frequency domain enhancement within the SSM architecture. Specifi-

cally, the Dynamic Mamba Block (DMB) adaptively learns sampling ofsets via a trainable ofset prediction network and performs flexible bilinear interpolation, enabling content-aware scanning that preserves spatial coherence. Furthermore, a frequency domain enhancement block balances spectral components through wavelet-guided decomposition and spectrum pooling, improving robustness under adverse imaging conditions. Experimental findings indicate that FU-Mamba attains a notable enhancement in segmentation accuracy, evidenced by a 1.1% increase in the mean intersection over union (mIoU) metric when evaluated on the dental segmentation dataset. Project page: https://byte2bite.github.io/FU-Mamba/

Keywords: Image Segmentation, Digital Dentistry, Linear Attention Mechanism, Computer Vision

## 1. Introduction

Tooth segmentation plays a vital role in digital dentistry, with potential applications in disease diagnosis [1], treatment monitoring [2, 3], and image-based clinical analysis. Efective segmentation enhances the detection of dental conditions [4], including caries, periodontal disease, and various oral abnormalities, thus allowing for more prompt and accurate diagnoses.

State space models (SSMs) [5], as exemplified by Mamba [6], have garnered significant interest in recent times due to their ability to selectively retain or discard information across sequences via the S6 block while ofering both lineartime computation and a global receptive field. Inspired by Mamba’s success in natural language processing, several works, such as Vim [7], have broadened its applicability to the field of computer vision by transforming two-dimensional images into one-dimensional sequences through the implementation of diverse scanning methodologies. This approach illustrates the capabilities of SSMs in visual tasks. As demonstrated in Fig. 1(a), PlainMamba [8] contends that traditional raster scanning fails to account for the significance of spatial continuity in images. Consequently, it proposes a continuous scanning approach aimed at maintaining the correlation between neighboring patches. Likewise, as shown in Fig. 1(b), LocalMamba [9] proposes a local scanning strategy to better capture spatial positional relationships. Nevertheless, these scanning approaches disrupt the spatial adjacency of semantic structures after flattening [10], leading to the loss of critical structural information. By contrast, as shown in Fig. 1(c), our approach adaptively determines the scanning order based on image content. While recent advances like DefMamba [11] have explored deformable state spaces for natural images, our approach uniquely synergizes this content-aware dynamic scanning with frequency balancing, enabling more flexible modeling capabilities tailored for complex dental artifacts. To demonstrate our scanning approach, the figure displays six sample points, though the full process includes additional points.

![](images/1177ec541c7ecb7266f673f0fa7f2eea49f2d843a289af18af020e3085069225.jpg)  
（a）Continuous Scanning

![](images/c372770d566ceb3f165f36ef59241e7af5324ccad3c83be6962ad03bae830244.jpg)  
（b）Local Scanning

![](images/6c27e008723a0013c5a2289de963305b2c1e11e4f6c40050e6446fd37ab23843.jpg)  
（c）Dynamic Scanning

![](images/9dff8fd236efb2d3c7af731ecf71472dafced31abe21cb02aaab315321f4baa8.jpg)  
（d）Original Image

![](images/e429fe57f5ccebddca66bd56ca02fda248d30afc973e58f27fad39d0a3dfa449.jpg)  
（e）U-Mamba

![](images/02b57379070e6db25146ed47228d48c12dd8ccb0778cb9419d113faebfacd168.jpg)  
（f）FU-Mamba  
Figure 1: Comparison of diferent scanning strategies in visual state space models. (a) Continuous scanning strategy proposed by PlainMamba; (b) Local scanning strategy proposed by LocalMamba; (c) Our proposed dynamic scanning strategy; (d) Original dental image under poor lighting conditions; (e) Low-quality segmentation mask produced by U-Mamba; (f) Segmentation mask generated by our approach. Yellow dots indicate reference points, and yellow arrows represent the scanning order.

Efectively segmenting images with complex and variable frequency characteristics remains a challenging task, particularly in dental image analysis, where the input is often afected by uneven lighting, reflective surfaces like enamel and metal restorations, motion blur, and noise during image capture. These artifacts modify the original frequency distribution of the image by attenuating high-frequency components, such as edges and textures, while redistributing the overall energy towards lower frequencies. Consequently, the model struggles to precisely detect subtle features like tooth edges and gum shapes. As shown in Fig. 1(e), U-Mamba fails to efectively balance high- and low-frequency components, resulting in inaccurate tooth boundary segmentation and an inability to completely segment the tooth under poor lighting conditions. In comparison, as demonstrated in Fig. 1(f), our approach attains a more efective equilibrium among frequency components and successfully delineates the entire tooth region, even under less than ideal illumination conditions.

Our work is inspired by recent advances in visual state space modeling and frequency-aware representation learning. In computer vision applications, the Mamba approach breaks down 2D images into smaller patches. These patches are then converted into several 1D sequences by employing specific scanning techniques from diferent angles. This approach successfully integrates the Mamba model for visual applications, achieving strong results and demonstrating the promise of SSMs in computer vision. In parallel, WTConv [12] utilizes wavelet-based decomposition to improve the network’s capacity to identify hierarchical spatial patterns. This approach is especially advantageous for the representation of large-scale structural information in natural images. SPAM [13] introduces a frequency-aware convolutional mixer that adaptively integrates spectral cues into spatial representations, thereby strengthening feature discriminability.

To address the challenges of spatial structure disruption and frequency imbalance in dental image segmentation, we propose a novel framework named FU-Mamba, which integrates a dynamic scanning mechanism and frequencydomain enhancement within a state space model. Specifically, we introduce a DMB that adaptively learns sampling positions and scanning sequences during training. By dynamically adjusting the scanning path to prioritize important information, this block improves the acquisition and processing of pertinent input features. As a result, the model attains enhanced flexibility in its modeling capabilities while preserving linear computational complexity and a comprehensive global modeling capacity. Additionally, to augment the model’s proficiency in managing intricate frequency characteristics, we integrate a frequency domain enhancement block (FEB) specifically engineered to equilibrate various frequency components. This module integrates a wavelet-guided spectral pooling module (WSPM), which not only amplifies frequency-domain information but also emphasizes mid-frequency signals aligned with human visual sensitivity. Fig. 2 presents a comprehensive depiction of the methodology we have proposed. The key contributions of this study are delineated as follows:

• A unified approach named FU-Mamba is developed by integrating dynamic scanning and frequency-aware modeling into a visual state space model for accurate and eficient dental image segmentation.

• A DMB is introduced to adaptively adjust sampling positions and scanning sequences based on input features, thereby preserving spatial continuity and enhancing structural representation.

• A frequency domain enhancement block is proposed to balance highand low-frequency components by integrating wavelet-based decomposition and spectrum pooling, improving robustness under challenging imaging conditions.

Experimental results on the dental segmentation dataset (DSD) [14] and OralVision [15] dataset demonstrate that the proposed FU-Mamba framework efectively generates high-quality segmentation masks.

The layout of the subsequent sections is as follows: Section 2 ofers a summary of studies focused on vision mamba and frequency domain analysis. Section 3 outlines the proposed methodology in detail. Section 4 presents the findings from the experiments conducted. Finally, Section 5 concludes the paper and explores possible avenues for future research.

## 2. Related Work

We present a concise overview of the contemporary literature concerning dental image segmentation, vision mamba, and frequency domain analysis. Additionally, it ofers a comparative analysis of the advantages and disadvantages associated with the current methodologies in these areas.

## 2.1. Dental Image Segmentation

Dental image segmentation [16] represents a critical area of inquiry within the domain of computer vision, especially in the context of dental radiological diagnostics. The eficacy of segmentation algorithms is paramount for the precise identification of dental structures and the development of subsequent treatment strategies. In recent years, scholars have introduced a range of novel segmentation techniques [17, 14, 18, 2, 19, 3] aimed at overcoming the challenges associated with dental image segmentation.

Du et al. [20] introduced a hierarchical segmentation level set model that markedly enhances segmentation accuracy by mitigating the transfer and accumulation of segmentation errors. Their approach provides a promising foundation for further studies in this field. In the realm of deep learning, Joshi et al. [21] employed the UNet model to segment panoramic X-ray images, demonstrating the power of advanced machine learning in medical imaging. Furthermore, Harsh et al. [22] extended this line of work by incorporating an attention module into the UNet framework and fine-tuning it for dental image analysis. The incorporation of an attention gate allows this approach to dynamically concentrate on critical feature regions within dental images while minimizing the influence of extraneous information, thus improving both segmentation accuracy and robustness.

Additionally, Hou et al. [23] introduced Teeth UNet, an enhanced UNet variant designed specifically for dental panoramic X-ray segmentation. This model enhances the delineation of tooth boundaries and contextual semantic information through the implementation of dense skip connections, multi-scale aggregation attention blocks, and dilated hybrid attention blocks, efectively addressing challenges such as indistinct tooth boundaries and low contrast. This model outperforms the standard UNet architecture by a significant margin, delivering sharper segmentation results and more precise feature detection, especially when dealing with intricate dental anatomy and images with poor contrast. In their quest for streamlined solutions, Lin et al. [24] pioneered a lean deep learning framework tailored for segmenting dental X-ray images, making it viable for edge device implementation. By leveraging knowledge distillation techniques, they refined the neural network’s design, reducing computational overhead and parameter counts without compromising accuracy. This breakthrough paves the way for real-time clinical use, striking an ideal balance between eficiency and performance. However, despite the significant progress made by these approaches in dental image segmentation, their segmentation results still exhibit certain limitations when dealing with oral edge regions and under insuficient lighting conditions. Beyond traditional mechanisms, recent advancements have explored diverse paradigms. For instance, novel backbones like U-KAN [25] and Light-UNETR [26] have demonstrated strong capabilities. Furthermore, interactive and ambiguity-aware models such as MedSAM-Agent [27], P2SAM [28], and others [29] empower segmentation through reinforcement learning or probabilistic prompting. Innovations like GTP-4O [30] and global correlation networks [31, 32, 33] further advance biomedical representation.

## 2.2. Vision Mamba

Mamba [6] was originally introduced for natural language processing, where it demonstrated strong performance in sequence modeling by selectively propagating input-dependent information through an eficient state space mechanism. This success has motivated the application of Mamba to various visual tasks, including medical image segmentation [34, 35], remote sensing image segmentation [36, 37, 38, 39], and object detection [40]. Unlike natural language sequences, images do not have a unique causal ordering, making the design of suitable scanning strategies crucial for visual Mamba models. To address this issue, Vim [7] introduced a bidirectional scanning mechanism to model image sequences and enlarge the efective receptive field. PlainMamba [8] further adopted a four-directional scanning strategy to enhance contextual modeling in images.

Recently, the eficiency and long-range modeling capabilities of SSMs and RNN-like architectures have been extensively explored across broader visual tasks. For instance, MambaAD [41] utilizes state space models to capture complex normal patterns for multi-class unsupervised anomaly detection, while recent comparative studies [42] have investigated Mamba and RWKV architectures for high-quality and high-eficiency visual representations in vision foundation models. Motivated by these advances, several studies [43, 44] have integrated U-Net architectures with visual Mamba blocks and achieved promising results in medical image segmentation. The rapid evolution of SSMs has further led to advanced architectures tailored for complex medical data. For instance, PV-SSM [45] explored pure visual state space models for high-dimensional medical image analysis. For volumetric data, SegMamba [46] and SegMambav2 [47] extended long-range sequential modeling to 3D medical image segmentation. To improve computational eficiency, UltraLight VM-UNet [26] proposed a lightweight visual Mamba design for skin lesion segmentation.

However, many existing visual Mamba methods still rely on predefined scanning patterns, which may limit their ability to preserve fine-grained local structures in images with irregular boundaries. To address local structure modeling, LocalMamba [9] introduced local window-based scanning to capture local dependencies while maintaining global context. To reduce the limitation of manually designed scanning patterns, DefMamba [11] proposed a deformable visual state space model for adaptively capturing spatial contexts in general vision tasks. Diferent from these works, FU-Mamba focuses on 2D dental image segmentation and integrates content-adaptive dynamic sampling with frequency-domain enhancement. This design aims to address dental-specific challenges, including complex tooth boundaries, low-contrast regions, uneven illumination, and reflection-induced frequency imbalance, thereby improving the modeling of finegrained two-dimensional oral structures.

## 2.3. Frequency Domain Analysis

In recent years, frequency-domain analysis has garnered considerable interest within the domain of computer vision, especially regarding its contribution to improving the robustness and generalization capabilities of deep learning models. Early studies have shown that convolutional neural networks (CNNs) exhibit an implicit bias toward high-frequency components, making them vulnerable to adversarial perturbations. For example, Caro et al. [48] demonstrated that local convolutions inherently favor high-frequency features, leading to vulnerabilities in adversarial settings. To address these challenges, researchers have explored approaches for balancing frequency components within neural networks. Zhang et al. [49] introduced the high-frequency feature disentanglement and recalibration (HFDR) module, which strategically isolates and recalibrates frequency-specific features to mitigate low-frequency bias and enhance adversarial robustness. In addition, wavelet-based approaches have been proposed to capture multi-scale frequency information. For instance, WTConv [12] utilizes wavelet transforms to expand the receptive field without excessive parameterization, thereby enhancing the modeling of low-frequency features. Meanwhile, SPAM [13] further designed frequency-aware convolutional mixers that dynamically adjust feature fusion based on the spectral distribution. Despite these advancements, existing approaches still face limitations in efectively balancing frequency components, especially under complex imaging conditions such as dental images, which are prone to noise and uneven illumination. These challenges highlight the need for more adaptive approaches that can dynamically respond to varying frequency characteristics.

## 3. Our Approach

We design a dental image segmentation approach based on the UNet architecture, which takes dental images as input and outputs corresponding segmentation masks. The fundamental principles of state space models are introduced in Section 3.1. Specifically, the dynamic scanning module is employed for feature extraction (see Section 3.2), while the FEB is used to efectively balance high- and low-frequency features (see Section 3.3). This overall design facilitates the generation of high-quality segmentation masks. The detailed architecture is illustrated in Fig. 2.

![](images/a3f54f4fc852f2cc38a2ea7a169846efc42ffb216998323ea7395216482dd4f9.jpg)  
Figure 2: The comprehensive framework of the proposed approach. The proposed approach follows the UNet architecture and comprises multiple encoders, decoders, and skip connections. In particular, the skip connections are constructed using frequency domain enhancement blocks. Meanwhile, encoders are built upon DMBs, ensuring efective feature extraction and reconstruction. In the figure, FEB refers to the frequency domain enhancement block, WSPM denotes a wavelet-guided spectral pooling module, and SPF represents the spectrum pooling filter. The blue area represents the mask generated after image segmen tation.

## 3.1. Preliminaries: State Space Model

SSMs were originally developed to analyze dynamic systems and time-series data, making it inherently unsuitable for processing two-dimensional images that lack causal relationships. To address this limitation, images are often converted into sequences, where scanning mechanisms help capture spatial features and transform them into visual representations. In SSMs, the temporal progression of the system’s states is represented by a hidden latent state denoted as ${ \bf h } ( t ) \in \mathbb { R } ^ { N }$ . This latent state facilitates the mapping of an input sequence $\mathbf { x } ( t ) \in \mathbb { R }$ to an output sequence $\mathbf { y } ( t ) \in \mathbb { R }$ . The dynamics of state transitions and observations are described by the subsequent linear diferential equations:

$$
\mathbf { h } ^ { \prime } ( \mathbf { t } ) = \alpha \mathbf { h } ( \mathbf { t } ) + \beta \mathbf { x } ( \mathbf { t } ) ,\tag{1}
$$

$$
\mathbf { y } ( \mathbf { t } ) = \theta \mathbf { h } ( \mathbf { t } ) + \delta \mathbf { x } ( \mathbf { t } ) .\tag{2}
$$

where the parameter matrices $\pmb { \alpha } \in \mathbb { C } ^ { N \times N }$ 2 $\boldsymbol { \beta } \in \mathbb { C } ^ { N }$ ， $\pmb \theta \in \mathbb { C } ^ { N }$ , and $\pmb { \delta } \in \mathbb { C } ^ { 1 }$ define the dependencies between system states, inputs, and outputs.

To enhance computational eficiency and adapt SSM for sequence modeling, the Mamba framework introduces a discretization step by incorporating a timestep parameter $\Delta t .$ This transformation facilitates the conversion of continuous system parameters, denoted as $_ { \pmb { \alpha } }$ and $\beta .$ , into their discrete equivalents, represented as $\overline { { \alpha } }$ and ${ \overline { { \beta } } } .$ A commonly utilized technique for executing this conversion is the zero-order hold (ZOH) approach, which is mathematically characterized as follows:

$$
\overline { { \alpha } } = \exp ( \Delta \alpha ) ,\tag{3}
$$

$$
\begin{array} { r } { \overline { { \beta } } = ( \Delta \alpha ) ^ { - 1 } ( \exp ( \Delta \alpha ) - \mathbf { I } ) \cdot \Delta \beta . } \end{array}\tag{4}
$$

With this discretization, the system dynamics are reformulated as follows:

$$
\begin{array} { r } { \mathbf { h } _ { \mathbf { t } } = \overline { { \alpha } } \mathbf { h } _ { \mathbf { t - 1 } } + \overline { { \beta } } \mathbf { x } _ { \mathbf { t } } , } \end{array}\tag{5}
$$

$$
\mathbf { y _ { t } } = \theta \mathbf { h _ { t } } + \delta \mathbf { x _ { t } } .\tag{6}
$$

Unlike traditional SSMs, Mamba adapts parameters $\beta$ and $\pmb { \theta }$ dynamically with input. Consequently, this modification ensures that Mamba maintains linear computational complexity during forward propagation, making it more eficient in sequence modeling tasks.

![](images/de0ab7a42dd134908251834f622787dc7a520f8510f9bcbe4edf381f60e28b4a.jpg)  
Figure 3: Illustration of the DMB, where only six reference points are displayed for clarity. Each starting reference point corresponds to a patch’s origin, with its ofset calculated via the OPN. Utilizing the predicted two-dimensional coordinates, bilinear interpolation is applied to extract features from significant areas, which are subsequently input into the SSM. To demonstrate our scanning approach, the figure displays six sample points, though the full process includes additional points.

## 3.2. Dynamic Mamba Block (DMB)

In visual data processing, images are typically represented in a two-dimensional structure. However, manually designed scanning patterns tailored for twodimensional images often overlook the continuous correlation between image patches. Although continuous scanning maintains consistency between neighboring patches, it often sacrifices fine-grained spatial details. On the other hand, localized scanning techniques preserve spatial relationships but struggle to establish connections across distant regions. These approaches depend on rigid, predetermined scanning patterns that fail to adjust dynamically to an image’s unique features.

To address this constraint, as illustrated in Fig. 3, we introduce a dynamic scanning technique that adjusts the search areas based on the input image’s features. This approach enables the model to prioritize key foreground details while disregarding irrelevant background noise [50], resulting in a smarter, more targeted scanning process that adapts to the visual context. Moreover, the proposed DMB is integrated across multiple feature scales, progressively reducing spatial resolution while increasing feature dimensionality at each stage. Within each DMB, depthwise convolutions (DWConv) and convolutional feed-forward networks (ConvFFN) are employed to enhance local feature extraction, thereby facilitating more precise spatial modeling and improving the overall representational capacity of the network.

Dynamic scanning (DS). DS is a mechanism designed to model the relationships between image patches by dynamically adjusting sampling locations based on key regions in the feature map. The core idea is to leverage an ofset prediction network (OPN) to learn content-aware sampling positions, extract corresponding features from the feature map using bilinear interpolation, and then feed them into the SSM for feature aggregation. Since the objective of the OPN is to regress smooth and spatially coherent ofsets, it is intentionally designed as a lightweight module to ensure eficiency and stable optimization. Specifically, the OPN adopts a lightweight architecture consisting of a depthwise convolution layer, followed by Layer Normalization (LN), a GELU activation function, and a linear projection layer for ofset regression. The depthwise convolution employs a $3 \times 3$ kernel to enhance local spatial sensitivity while maintaining low computational complexity. The predicted ofsets are further constrained within the normalized coordinate range using a tanh operation to stabilize dynamic sampling and avoid abrupt spatial shifts during training. To prevent invalid sampling positions, all predicted coordinates are clipped to the valid feature map range before bilinear interpolation. Coordinates exceeding the boundary are projected back to the nearest valid location, ensuring stable feature extraction near tooth boundaries and low-contrast regions.

Specifically, DS first employs OPN to predict the coordinate ofsets for each image patch relative to its original position:

$$
\Delta r = O P N ( [ x _ { 1 } , x _ { 2 } , . . . , x _ { n } ] ) .\tag{7}
$$

The ofsets identify which areas warrant greater focus, directing where features should be sampled. In mathematical terms, OPN processes the feature map and generates coordinate ofsets $( \Delta \mathrm { ~ r ~ } )$ for every patch. These ofsets are combined with the initial coordinates (p) to compute the adjusted sampling

positions $\left( \mathrm { p ^ { \prime } } \right)$ , as shown below:

$$
p ^ { \prime } [ h , w , : ] = p [ h , w , : ] + \Delta r [ h , w , : ] .\tag{8}
$$

Here, h and w denote the patch’s positional indices along the height and width axes. To maintain consistency across varying feature map resolutions, these coordinates are normalized to fall within the range [-1, 1].

After identifying the updated sampling points $p ^ { \prime }$ , the DS module employs bilinear interpolation to fetch relevant features from the feature map. For a predicted sampling coordinate $p _ { h , w } ^ { \prime } = ( a _ { x } , a _ { y } )$ , the sampled feature is computed as:

$$
X ^ { \prime } [ h , w ] = \phi ( p _ { h , w } ^ { \prime } , X ) ,\tag{9}
$$

where $\phi ( \cdot )$ denotes the bilinear interpolation function. Specifically, it is formulated as:

$$
\phi ( p _ { h , w } ^ { \prime } , X ) = \sum _ { ( r _ { x } , r _ { y } ) \in \mathcal { N } ( p _ { h , w } ^ { \prime } ) } g ( a _ { x } , r _ { x } , a _ { y } , r _ { y } ) X [ r _ { y } , r _ { x } , : ] ,\tag{10}
$$

where $\mathcal { N } ( p _ { h , w } ^ { \prime } )$ denotes the four nearest integer grid points around $p _ { h , w } ^ { \prime }$ . The interpolation weight is defined as:

$$
g ( a _ { x } , r _ { x } , a _ { y } , r _ { y } ) = \operatorname* { m a x } ( 0 , 1 - | a _ { x } - r _ { x } | ) \times \operatorname* { m a x } ( 0 , 1 - | a _ { y } - r _ { y } | ) .\tag{11}
$$

Here, $a _ { x }$ and $a _ { y }$ represent the horizontal and vertical coordinates of the predicted sampling position, while $r _ { x }$ and $r _ { y }$ denote the indices of neighboring feature map locations. The function $g ( a _ { x } , r _ { x } , a _ { y } , r _ { y } )$ computes the bilinear interpolation weight according to the spatial distance between the predicted coordinate and its neighboring grid points.

After feature sampling, the collected features are reordered according to the spatial proximity of the updated coordinates $p ^ { \prime }$ . Specifically, neighboring sampled positions in the dynamic coordinate space are arranged adjacently in the generated sequence, allowing the SSM to better preserve local structural continuity during sequential modeling. After collecting the sampled features, DS reorganizes them according to their updated spatial coordinates before feeding them into the SSM for sequential modeling. To adaptively extract informative regions, we employ a dynamic scanning mechanism, whose complete workflow is outlined in Algorithm 1. Notably, the learned sampling positions in DS provide additional relative position information, which helps the SSM better capture local feature relationships and improve overall feature aggregation capability. In summary, DS dynamically adjusts sampling locations, allowing the feature extraction process to flexibly focus on key regions while using bilinear interpolation to achieve smooth information aggregation.

Algorithm 1: Dynamic Scanning   
Require: Input feature map $\overline { { \boldsymbol { X } \in \mathbb { R } ^ { H \times W \times C } } }$   
Ensure: Reordered feature sequence $X ^ { \prime } \in \mathbb { R } ^ { N \times C }$   
1: Predict ofsets: $\Delta r \gets \mathrm { O P N } ( X )$   
2: Compute updated coordinates: $p ^ { \prime } \gets p + \Delta r$   
3: Bilinear feature sampling: $f [ h , w ] \gets \phi ( p ^ { \prime } [ h , w ] , X )$   
4: Generate semantic-aware ordering: $\mathcal { T }  \mathrm { s o r t } ( p ^ { \prime } )$   
5: for $k = 1$ to N do   
6: $( h _ { k } , w _ { k } )  \mathcal { T } _ { k }$   
7: $X ^ { \prime } [ k ] \gets f [ h _ { k } , w _ { k } ]$   
8: end for   
9: return $X ^ { \prime }$

## 3.3. Frequency Domain Enhancement Block

The segmentation of dental images poses distinct challenges attributable to the intrinsic complexity of their frequency composition. On one hand, highfrequency artifacts, including scanner-induced noise and sharp discontinuities along tooth boundaries, can interfere with accurate localization. On the other hand, low-frequency structures like smooth gingival contours and the general geometry of teeth encode critical global information. Traditional approaches often fail to capture and reconcile these contrasting frequency components effectively, especially under conditions of poor contrast between teeth and soft tissue, individual anatomical variability, or inconsistent lighting. These limitations lead to degraded segmentation quality and poor model robustness across diverse cases.

In order to tackle these challenges, we propose a frequency-aware enhancement mechanism that emphasizes perceptually important structures while suppressing irrelevant high-frequency disturbances. Motivated by the contrast sensitivity profile of the human visual system, which is most responsive to midfrequency signals, we introduce the WSPM. This module is intended to enhance low-frequency components while simultaneously integrating them with highfrequency details in a balanced manner, thus directing the spectral distribution towards mid-range frequencies.

To ensure adaptability across multiple spatial scales, the WSPM employs a multi-branch design with variable convolutional depths and kernel shapes. The low-frequency region is defined as a centered circular area in the frequency spectrum. To ensure adaptability across diferent feature scales, its radius is dynamically determined according to the spatial resolution of the input feature map. Specifically, the radius is set to one-eighth of the minimum feature map dimension at each encoder stage:

$$
r = { \frac { \operatorname* { m i n } ( H , W ) } { 8 } } .\tag{12}
$$

This adaptive setting enables consistent low-frequency coverage across diferent feature scales while avoiding excessive suppression of mid-frequency components.

Directional sensitivity is strengthened using separable convolutions with kernels sized 1×n and n×1. Simultaneously, dilated convolutions (rates: 1, 3, 5, 7) widen the receptive field. Appropriate padding ensures alignment across branches, facilitating smooth concatenation during feature aggregation. Features extracted from diferent branches are concatenated along the channel dimension and subsequently fused using a 1 × 1 convolution layer for adaptive channel aggregation. This adaptive fusion strategy allows the network to learn branch-wise feature importance automatically rather than relying on uniform averaging. The combination of asymmetric convolutions and diferent dilation rates enables the network to capture directional structures and mediumfrequency contextual patterns at multiple receptive fields, thereby enhancing structural continuity while suppressing excessive high-frequency noise.

The key concept of deep wavelet convolution (DWT-Conv) combines wavelet transforms with convolutional operations to eficiently capture and refine multiscale image features. Specifically, DWT-Conv first applies wavelet decomposition to the input image, typically using the Haar wavelet transform. This process separates the image into low-frequency (LL) and high-frequency (LH, HL, HH) components. In our implementation, a single-level Haar wavelet decomposition is employed at each encoder stage. Fixed Haar wavelet kernels are adopted instead of learnable wavelet bases to maintain stable frequency decomposition and avoid introducing additional trainable parameters. This design also improves computational eficiency while preserving high-frequency edge structures commonly observed in dental images. This process can be represented as:

$$
[ X _ { L L } , X _ { L H } , X _ { H L } , X _ { H H } ] = \mathrm { D W T } ( X ) .\tag{13}
$$

where $X \in \mathbb { R } ^ { H \times W \times C }$ represents the input feature map. The DWT decomposes it into four subbands $( X _ { L L } , X _ { L H } , X _ { H L } , X _ { H H } )$ , each with reduced spatial dimensions and preserved channels R ${ \begin{array} { l } { { \frac { H } { 2 } } \times { \frac { W } { 2 } } \times C } \end{array} }$ . Subsequently, a $1 \times 1$ convolution operation is employed on the concatenated subbands along the channel dimension (yielding an intermediate tensor of R $\scriptstyle { \frac { H } { 2 } } \times { \frac { W } { 2 } } \times 4 C  $ . The convolution outputs a refined tensor of the same size $4 C ,$ , which is then split back into four enhanced subbands to augment low-frequency features while eficiently modeling cross-band relationships:

$$
[ X _ { L L } ^ { \prime } , X _ { L H } ^ { \prime } , X _ { H L } ^ { \prime } , X _ { H H } ^ { \prime } ] = \mathrm { { S p l i t } } ( { \mathrm { C o n v } } _ { 1 \times 1 } ( { \mathrm { C o n c a t } } ( X _ { L L } , X _ { L H } , X _ { H L } , X _ { H H } ) ) ) .\tag{14}
$$

The final step involves applying an inverse wavelet transform to combine the

refined subbands, producing the enhanced feature map as follows:

$$
Y = \mathrm { I W T } ( [ X _ { L L } ^ { \prime } , X _ { L H } ^ { \prime } , X _ { H L } ^ { \prime } , X _ { H H } ^ { \prime } ] ) .\tag{15}
$$

To alleviate potential information loss during inverse reconstruction, a residual connection is introduced between the input feature map and the reconstructed output. In addition, Layer Normalization is applied after subband fusion to stabilize the feature distribution under varying illumination and noisy conditions.

This operation not only enhances low-frequency information but also preserves the spatial resolution of the image to some extent, providing a richer information foundation for subsequent feature extraction and segmentation tasks. It is particularly useful for handling the complex relationships between structures such as teeth and gums in dental images.

The spectrum pooling filter (SPF) serves as a key mechanism for harmonizing high- and low-frequency data, enhancing feature representation through balanced spectral processing. $\mathrm { B y }$ converting spatial feature maps into the frequency domain via a 2D discrete Fourier transform (DFT), the SPF dynamically regulates various frequency components. This transformation enables precise modulation of spectral information, ensuring optimal feature extraction across diferent frequency bands. This procedure can be illustrated as follows:

$$
Z = F ( z ) ,\tag{16}
$$

where z represents the feature map after deep convolution, and $F$ denotes the 2D DFT operation. Then, the spectrum is centralized by focusing the lowfrequency components in the center and removing the remaining parts outside the spectrum. This procedure can be illustrated as follows:

$$
S _ { l f } ( u , v ) = \left\{ \begin{array} { l l } { G ( Z ) ( u , v ) , } & { \mathrm { i f ~ } ( u , v ) \in A _ { l f } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{17}
$$

$$
S _ { h f } = G ( Z ) - S _ { l f } .\tag{18}
$$

Here, $G ( \cdot )$ is the Fourier transform centralization function, and $A l _ { f }$ defines the low-frequency region centered at the origin. The centralization operation only reorders the spatial arrangement of frequency components through FFT shift without altering their original magnitude or phase values, thereby preserving structural boundary consistency during inverse transformation. Next, the processed frequency features are mapped back to the spatial domain through inverse transformations:

$$
f _ { l p } ( Z ) = F ^ { - 1 } ( G ^ { - 1 } ( S _ { l f } ) ) ,\tag{19}
$$

$$
f _ { h p } ( Z ) = F ^ { - 1 } ( G ^ { - 1 } ( S _ { h f } ) ) .\tag{20}
$$

Finally, the features from diferent frequency bands are fused to generate the spectrum-pooled feature map:

$$
\tilde { Z } = \lambda f _ { l p } ( Z ) + ( 1 - \lambda ) f _ { h p } ( Z ) ,\tag{21}
$$

where λ is the balancing parameter that controls the proportion of high- and low-frequency components.

This approach allows SPF to minimize disruptions caused by high-frequency noise while maintaining critical low-frequency structural details. As a result, it significantly improves the precision and reliability of dental image segmentation, especially when dealing with complex interfaces between teeth and surrounding soft tissues.

## 4. Results

We assess the eficacy of our methodology and conduct a comparative analysis with alternative methodologies utilizing two distinct datasets.

## 4.1. Datasets and Evaluation Criteria

## 4.1.1. Datasets

In order to assess the eficacy of our proposed semantic segmentation approach, we performed evaluations utilizing two publicly accessible dental datasets: the DSD [14] and the Oralvision [15] dataset.

DSD used in this study is derived from the dental segmentation data described in Tian et al. [14]. Specifically, we extracted 2D dental image frames and their corresponding pixel-wise segmentation masks from the original dental video data for 2D dental image segmentation experiments. The DSD subset follows a multi-class semantic segmentation setting, where each pixel is annotated into one of several oral structure categories, including teeth, surrounding soft tissues, and background, rather than a binary tooth-background label. The resulting DSD subset consists of 4900 training images, 151 validation images, and 151 test images. The training, validation, and test sets were constructed without image overlap, and all splits were mutually exclusive to prevent potential data leakage. The detailed data augmentation strategy is described in Section 4.2.

The Oralvision dataset comprises intraoral images captured at a resolution of 640×480 using dental scanning equipment in clinical settings. These images were collected from randomly selected patients and manually annotated by six university researchers utilizing LabelM. The dataset categorizes oral structures into diferent tissue types, including gums, lips, teeth, tongue, and cheeks. Furthermore, it incorporates real-world challenges by retaining visual noise elements such as food particles, dental calculus, implant attachments, and saliva. The dataset is comprised of 4000 images designated for training purposes, 120 images allocated for validation, and an additional 120 images set aside for testing. The training, validation, and test sets were constructed without image overlap, and all splits were mutually exclusive to prevent potential data leakage. All OralVision images were resized to a unified input resolution during preprocessing while preserving challenging artifacts such as saliva, dental calculus, and illumination variations for robustness evaluation.

## 4.1.2. Evaluation Criteria

To assess segmentation quality, we adopt mean Intersection over Union (mIoU) together with boundary mean IoU (mBIoU). These metrics respectively evaluate overall spatial overlap and edge delineation precision across all semantic categories. Per-class IoU scores are first computed and subsequently averaged to yield mIoU and mBIoU, thereby reflecting model behavior across the entire tissue taxonomy.

Boundary IoU is determined according to standard practice: for each category, its contour is extracted and dilated by a tolerance of t = 3 pixels to form a narrow band around the boundary. Denoting the dilated boundary pixel sets of the ground-truth and predicted masks as $B _ { g t }$ and $B _ { p r e d }$ , the boundary IoU is defined as

$$
\mathrm { I o U } _ { \mathrm { b o u n d a r y } } = \frac { | B _ { g t } \cap B _ { p r e d } | } { | B _ { g t } \cup B _ { p r e d } | } .\tag{22}
$$

We report parameter size, computational complexity, and inference latency as eficiency indicators. FLOPs were estimated at the same input resolution for all methods. For SAM-based methods, one bounding-box prompt was used during evaluation. Inference latency was measured under single-sample inference with a batch size of 1.

## 4.2. Implementation Details

All experiments were conducted on a high-performance workstation equipped with an Intel Core i9-13900KF CPU, 32 GB RAM, and four NVIDIA RTX 4090D GPUs. For the eficiency benchmarks, all methods were evaluated on a single NVIDIA RTX 4090D GPU with batch size 1 and an input resolution of $6 4 0 ~ \times ~ 4 8 0$ . To ensure fair and reproducible comparison, no multi-GPU inference was used during latency measurement, and all compared methods were benchmarked under the same hardware and input settings. All methods were implemented under the PyTorch framework to ensure a unified experimental environment and fair comparison. All experiments were repeated three times with diferent random seeds, and the reported results correspond to the mean and standard deviation. For fair comparison, all competing methods were trained and evaluated under identical data splits and random seed settings. Oficial implementations were adopted for all compared methods whenever publicly available, with only necessary modifications to accommodate unified dental image input resolutions and output categories.

For training eficiency, the batch size was set to 32 for the DSD dataset and 24 for the OralVision dataset, respectively, according to GPU memory consumption and convergence stability under diferent input resolutions. All models were trained for 45 epochs using the AdamW optimizer with an initial learning rate of 0.01 and a weight decay of $3 \times 1 0 ^ { - 5 }$ . A cosine annealing learning rate schedule was adopted throughout training to gradually reduce the learning rate and improve optimization stability.

The overall optimization objective consists of a hybrid segmentation loss and an ofset regularization term. Specifically, the segmentation loss combines Dice loss and cross-entropy loss to jointly optimize region overlap and pixel-wise classification accuracy:

$$
\begin{array} { r } { \mathcal { L } _ { s e g } = \mathcal { L } _ { D i c e } + \mathcal { L } _ { C E } . } \end{array}\tag{23}
$$

Dice loss was introduced to alleviate the class imbalance commonly observed in multi-class dental image segmentation.

To stabilize dynamic sampling, an additional ofset regularization loss was introduced:

$$
\mathcal { L } _ { r e g } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \| \Delta { r } _ { i } \| _ { 2 } ^ { 2 } ,\tag{24}
$$

where $\Delta r _ { i }$ denotes the predicted ofset at the i-th spatial location. The final optimization objective is defined as:

$$
\mathcal { L } = \mathcal { L } _ { s e g } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { r e g } ,\tag{25}
$$

where $\lambda _ { \mathrm { r e g } }$ was empirically set to 0.01.

An early stopping strategy was adopted to mitigate overfitting, where training was terminated if the validation performance did not improve for 8 consecutive epochs. The minimum improvement threshold for validation mIoU was set to 0.001. In addition, gradient clipping with a threshold of 1.0 was applied during optimization to further improve training stability.

To improve robustness against intra-oral viewpoint variations and illumination changes, data augmentation strategies including random horizontal flipping, random rotation, random scaling, and brightness adjustment were applied during training. The probability of applying each augmentation operation was set to 0.5, while the brightness adjustment factor was randomly sampled within the range of [0.8, 1.2]. Elastic deformation and CutMix augmentation were not adopted because they may introduce unrealistic anatomical distortions in dental structures.

For the proposed dynamic scanning module, the OPN adopts a lightweight architecture consisting of a $3 \times 3$ depthwise convolution layer, followed by LN, GELU activation, and a linear projection layer for ofset regression. To stabilize dynamic sampling, the predicted ofsets were normalized using a tanh activation function and constrained within the range [−0.25, 0.25] relative to the normalized coordinate grid. In addition, the ofset regularization term in Eq. (24) further constrains excessive spatial displacement and improves the smoothness of neighboring sampling locations during optimization. In addition, all predicted sampling coordinates were clipped to the valid feature map range before bilinear interpolation to avoid invalid boundary sampling.

For the DMB design, the channel dimensions across the four encoder stages were set to {64, 128, 256, 512}, respectively. The ConvFFN expansion ratio was set to 4, while all depthwise convolutions employed a kernel size of $3 \times 3$

For the frequency enhancement module, the WSPM adopted a cascaded wavelet convolution structure with a $1 \times 1$ convolution kernel and stride 1. Two parallel SPF branches were introduced, with frequency balancing weights empirically set to $\lambda _ { 1 } = 0 . 7$ to balance low-frequency structural information and high-frequency detail preservation. All DWT and DFT operations were performed on downsampled deep feature maps to reduce computational overhead while preserving frequency-aware representation capability.

## 4.3. Ablation Study

In order to assess the eficacy of this module, we undertook comprehensive experiments designed to examine its influence on the overall performance of the model. Specifically, we executed ablation studies utilizing the dental segmentation dataset and Oralvision dataset to evaluate the module’s role in enhancing segmentation accuracy.

Backbone. To evaluate the efectiveness of our proposed methodology, we employ U-Mamba as the foundational architecture for our ablation studies. U-Mamba features a symmetric encoder-decoder structure that combines the strong feature extraction capabilities of UNet with the extensive context modeling benefits of the Mamba framework. This architectural design has proven to be efective in medical image segmentation tasks.

Table 1: Ablation study on the scanning design. CS refers to the continuous scanning strategy, LS denotes the local scanning strategy, and DMB denotes the proposed Dynamic Mamba Block. Results are reported as mean ± standard deviation over three independent runs.
<table><tr><td>Scanning Strategy</td><td></td><td>DSD (mIoU) OralVision (mIoU)</td></tr><tr><td>CS</td><td> $8 9 . 7 \pm 0 . 4$ </td><td> $8 9 . 1 \pm 0 . 5$ </td></tr><tr><td>LS</td><td> $9 0 . 2 \pm 0 . 3$ </td><td> $8 9 . 8 \pm 0 . 4$ </td></tr><tr><td>DMB</td><td>²  ${ \bf 9 0 . 9 \pm 0 . 3 }$ </td><td> ${ \bf 9 0 . 4 } \pm { \bf 0 . 3 }$ </td></tr></table>

Impact of the DMB. In order to assess the impact of the DMB, we conduct a comparative analysis with two alternative scanning methodologies: continuous scanning (CS) and local scanning (LS), while keeping all other components unchanged. As shown in Table 1, DMB markedly enhances the model’s capacity to discern intricate details and adjust to spatial variations present in dental images. Specifically, it achieves a 1.2% increase in mIoU compared to CS on the DSD. This improvement is mainly attributed to DMB’s capacity to dynamically adjust sampling positions based on input features, thereby enhancing structural awareness and local context modeling. As a result, DMB enhances the precision of boundary delineation and elevates segmentation performance within intricate oral environments.

![](images/2aca575a72ea51bfd380a82b4309e9c09c14fbe885a956ccd8a9a0f454a3c722.jpg)  
(a) Dental Segmentation Dataset

![](images/e8bb837ac339547a713cc3904ca71320633a30dda998de543ff80ad8f3ff4f71.jpg)  
(b) OralVision Dataset  
Figure 4: Comparison of mIoU scores on the dental segmentation dataset and the OralVision dataset with and without the FEB, as well as with the FEB-${ \bf w } / { \bf o }$ WSPM variant. Here, FEB-w/o WSPM refers to a configuration in which the WSPM component within FEB is replaced by DWT-Conv. As shown, the model incorporating the full frequency domain enhancement block achieves higher mIoU than the other two configurations.

Impact of the Frequency Domain Enhancement Block. We assess the eficacy of the FU-Mamba model by comparing its performance with and without the FEB on both the dental segmentation dataset and the OralVision dataset. As shown in Fig. 4(a), the leftmost result corresponds to the FU-Mamba model with only skip connections, the center represents a variant named FEB-w/o WSPM in which the WSPM component within FEB is replaced by DWT-Conv, and the rightmost result shows our full version incorporating WSPM. The same comparison is presented in Fig. 4(b). The FU-Mamba model without FEB achieves 1.4% lower mIoU compared to the version with the FEB. Similarly, the FEB-w/o WSPM variant also underperforms the full version that includes WSPM. We further visualize the segmentation results with and without the FEB module. As illustrated in Fig. 5, removing FEB leads to a notable degradation in segmentation quality, particularly in regions afected by noise and insuficient illumination. The observed change in performance can be ascribed to the module’s capacity to equilibrate high- and low-frequency characteristics, thereby eficiently attenuating high-frequency noise while maintaining critical structural information.

To further investigate the efectiveness of diferent components within the proposed FEB, we conduct an additional ablation study on the DSD dataset, as summarized in Table 2. Specifically, we separately evaluate the contributions of the wavelet-based DWT-Conv module and the SPF module.

![](images/ba8d8462e41e82007261a8963d26f28fe319d05d0c04ec562e2d4515580a04df.jpg)  
Image

![](images/6d44f5262623bd6fe02051fcfc0db7b410f12a1017980407757cbfd2ff65b1fb.jpg)  
w/o FEB

![](images/44f05006329161b2f600a76d7cc1842ff96bb8ed0a82f3b31013eaa241777850.jpg)  
w/ FEB

![](images/281b4e09dbc4de75002776aedc29e64a0d8ce64e037ef8a032c5320f959ada08.jpg)  
GT  
Figure 5: Visualization comparison of the frequency domain enhancement block. The FEB module enables the segmentation results on low-light dental images to more closely align with the ground truth. The blue regions indicate the predicted segmentation results, while the yellow dashed boxes highlight diferences across various settings, emphasizing the advantages of incorporating FEB under challenging lighting conditions.

Starting from the baseline model without FEB, introducing only the DWT-Conv module improves the mIoU from 90.9% to 91.7%, demonstrating that wavelet-guided decomposition efectively enhances multi-scale structural representation and frequency-aware feature extraction for complex dental regions. When only the SPF module is employed, the mIoU increases to 91.4%, indicating that adaptive spectral balancing helps suppress high-frequency noise and improves semantic consistency under challenging illumination conditions.

By jointly integrating DWT-Conv and SPF, the complete FEB achieves the best performance with 92.3% mIoU and 88.9% mBIoU. These results demonstrate that the two components provide complementary benefits, where DWT-Conv serves as the primary contributor for frequency decomposition and structural preservation, while SPF further enhances discriminative spectral fusion and robustness against noise interference. Consequently, the combined design achieves more accurate and stable dental image segmentation.

Hyperparameter Analysis. To further evaluate the sensitivity of the proposed frequency enhancement mechanism, we conducted a hyperparameter analysis on the frequency balancing weight λ in the Spectrum Pooling Filter (SPF).

Table 2: Ablation study on the efectiveness of diferent frequency enhancement components in FEB.
<table><tr><td>SPF</td><td>DWT-Conv</td><td>mIoU (%) 1</td><td>mBIoU (%)</td></tr><tr><td rowspan="3">V</td><td></td><td> $9 0 . 9 \pm 0 . 3$ </td><td> $8 6 . 9 \pm 0 . 2$ </td></tr><tr><td></td><td> $9 1 . 4 \pm 0 . 3$ </td><td> $8 7 . 5 \pm 0 . 3$ </td></tr><tr><td>√</td><td> $9 1 . 7 \pm 0 . 2$ </td><td> $8 7 . 9 \pm 0 . 3$ </td></tr><tr><td>V</td><td>V _</td><td> ${ \bf 9 2 . 3 \ : \pm { \ : 0 . 2 } }$ </td><td>一  ${ \bf 8 8 . 9 \pm 0 . 3 }$ </td></tr></table>

The parameter λ controls the relative contribution between low-frequency structural information and high-frequency detail features during frequency-domain fusion. Specifically, a smaller λ emphasizes high-frequency components, while a larger λ strengthens low-frequency semantic information.

Fig. 6 reports the segmentation performance under diferent λ settings on the DSD dataset. When λ is set to 0.3, the model tends to overly focus on highfrequency details and becomes more sensitive to noise interference, resulting in relatively lower segmentation accuracy. As λ increases, the model achieves better structural consistency and boundary representation. The best performance is obtained at $\lambda = 0 . 7$ , achieving an mIoU of 92.3%. However, when λ is further increased to 0.9, excessive emphasis on low-frequency information weakens fine-grained boundary modeling capability, leading to a slight performance decrease.

![](images/29b15b2ed3a37f70fcef2c5b446185e19d17db56a8c90a1b5ff39ad2bdd3b74a.jpg)  
Figure 6: Efect of diferent frequency balancing weights λ.

Incremental Ablation and Complexity Analysis. To further evaluate the efectiveness of each proposed component, an incremental ablation study is conducted on the proposed FU-Mamba framework. Starting from the baseline

U-Mamba architecture, the DMB and FEB are progressively introduced to analyze both segmentation performance improvement and computational overhead. The quantitative results on the DSD dataset are summarized in Table 3.

Table 3: Incremental ablation and complexity analysis on the DSD.
<table><tr><td>DMB</td><td>FEB</td><td>mIoU (%)</td><td>mBIoU (%)</td><td>Params (M)</td><td>Latency (ms)</td></tr><tr><td></td><td></td><td> $\overline { { 9 0 . 1 \pm 0 . 3 } }$ </td><td> $8 5 . 8 \pm 0 . 4$ </td><td>32.8</td><td>21.3</td></tr><tr><td>√</td><td></td><td> $9 0 . 9 \pm 0 . 3$ </td><td> $8 6 . 7 \pm 0 . 3$ </td><td>34.1</td><td>25.7</td></tr><tr><td>√</td><td>√</td><td>_  ${ \bf 9 2 . 3 \pm 0 . 2 }$ </td><td> ${ \bf 8 8 . 9 \ \pm \ 0 . 3 }$ </td><td>36.2</td><td>31.8</td></tr></table>

Specifically, after introducing the DMB module, the model achieves noticeable improvement in mIoU owing to the adaptive dynamic scanning strategy, which enhances spatial continuity modeling and global context aggregation.

When further incorporating the FEB module, the segmentation accuracy is further improved, particularly in complex boundary regions and low-contrast structures. This improvement mainly benefits from the explicit modeling of frequency-domain information through wavelet decomposition and spectrumaware enhancement. Meanwhile, the additional computational overhead is primarily introduced by the DWT and DFT operations in the frequency-domain branch.

## 4.4. Evaluation of Tooth Segmentation

We conduct a comparative analysis of our methodology against various leading segmentation techniques, including those based on SAM [51, 52, 53, 54] and Mamba-like architectures [7, 55, 43, 44], under consistent experimental settings. We also include T-Mamba [56], a unified frequency-based Mamba framework for 2D and 3D tooth segmentation, to provide a domain-specific baseline.

In a fair comparison under identical experimental conditions, our algorithm demonstrated superior performance on the DSD and Oralvision dataset compared to the aforementioned approaches in Table 4. Specifically, our approach achieved a 1.1% improvement in mIoU on the DSD and a 1.3% improvement on the Oralvision dataset, compared to the HQ-SAM model. These results highlight the superior segmentation quality ofered by our approach over SAM-based approaches. Furthermore, when compared to other linear mechanism-based approaches, such as U-Mamba and SegMan, our algorithm not only outperformed these approaches in terms of both mIoU and mBIoU but did so with significant improvements, underscoring the efectiveness of our approach in providing high-quality segmentation results for dental images.

Table 4: Experimental results on the test set of DSD and the Oralvision dataset. Our experimental results are highlighted in purple.
<table><tr><td rowspan="2">Approach</td><td colspan="2">DSD</td><td colspan="2">OralVision</td></tr><tr><td>mIoU</td><td>mBIoU</td><td>mIoU</td><td>mBIoU</td></tr><tr><td>SAM [51]</td><td> $8 8 . 7 \pm 0 . 4$ </td><td> $8 5 . 3 \pm 0 . 5$ </td><td> $8 8 . 1 \pm 0 . 3$ </td><td> $8 5 . 0 \pm 0 . 4$ </td></tr><tr><td>HQ-SAM [52]</td><td> $9 1 . 2 \pm 0 . 3$ </td><td> $8 8 . 1 \pm 0 . 4$ </td><td> $9 0 . 3 \pm 0 . 3$ </td><td> $8 7 . 2 \pm 0 . 3$ </td></tr><tr><td>EfficientSAM [53]</td><td> $8 7 . 2 \pm 0 . 5$ </td><td> $8 4 . 4 \pm 0 . 5$ </td><td> $8 6 . 9 \pm 0 . 4$ </td><td> $8 3 . 8 \pm 0 . 3$ </td></tr><tr><td>GroundedSAM [54]</td><td> $8 8 . 6 \pm 0 . 4$ </td><td> $8 5 . 5 \pm 0 . 3$ </td><td> $8 7 . 6 \pm 0 . 3$ </td><td> $8 4 . 7 \pm 0 . 3$ </td></tr><tr><td>nnU-Net [57]</td><td> $8 9 . 7 \pm 0 . 3$ </td><td> $8 6 . 1 \pm 0 . 3$ </td><td> $8 9 . 3 \pm 0 . 3$ </td><td> $8 5 . 6 \pm 0 . 4$ </td></tr><tr><td>U-KAN [25]</td><td> $8 9 . 1 \pm 0 . 4$ </td><td> $8 5 . 4 \pm 0 . 3$ </td><td> $8 8 . 7 \pm 0 . 4$ </td><td> $8 5 . 0 \pm 0 . 4$ </td></tr><tr><td>MedNeXt [58]</td><td> $9 0 . 2 \pm 0 . 3$ </td><td> $8 6 . 6 \pm 0 . 4$ </td><td> $9 0 . 0 \pm 0 . 3$ </td><td> $8 6 . 1 \pm 0 . 3$ </td></tr><tr><td>T-Mamba [56]</td><td> $9 0 . 4 \pm 0 . 3$ </td><td> $8 6 . 8 \pm 0 . 3$ </td><td> $8 9 . 8 \pm 0 . 3$ </td><td> $8 6 . 0 \pm 0 . 4$ </td></tr><tr><td>Vim [7]</td><td> $9 0 . 6 \pm 0 . 3$ </td><td> $8 6 . 4 \pm 0 . 4$ </td><td> $8 9 . 9 \pm 0 . 4$ </td><td> $8 6 . 3 \pm 0 . 3$ </td></tr><tr><td>VRWKV [55]</td><td> $8 9 . 8 \pm 0 . 4$ </td><td> $8 5 . 7 \pm 0 . 3$ </td><td> $8 9 . 2 \pm 0 . 3$ </td><td> $8 4 . 8 \pm 0 . 3$ </td></tr><tr><td>U-Mamba [43]</td><td> $9 0 . 1 \pm 0 . 3$ </td><td> $8 5 . 8 \pm 0 . 4$ </td><td> $9 0 . 0 \pm 0 . 3$ </td><td> $8 5 . 2 \pm 0 . 3$ </td></tr><tr><td>Mamba-UNet [44]</td><td> $9 0 . 3 \pm 0 . 2$ </td><td> $8 6 . 0 \pm 0 . 3$ </td><td> $8 9 . 7 \pm 0 . 3$ </td><td> $8 4 . 9 \pm 0 . 4$ </td></tr><tr><td>Ours(FU-Mamba)</td><td>_  ${ \bf 9 2 . 3 \pm 0 . 2 }$  </td><td> ${ \bf 8 8 . 9 \pm 0 . 3 }$  </td><td> ${ \bf 9 1 . 6 \pm 0 . 3 }$  </td><td> ${ \bf 8 8 . 0 \pm 0 . 4 }$ </td></tr></table>

We conducted a qualitative comparison with U-Mamba and HQ-SAM, and visualized the segmentation results on the DSD in Fig. 7. As shown, our approach demonstrates a greater ability to adapt to complex oral environments, thereby producing more fine-grained segmentation results.

During the acquisition of oral images, shadow occlusion is often unavoidable, leading to uneven illumination and degraded segmentation performance. In the first example, the segmentation result is adversely afected by both lighting inconsistencies and the presence of metal inlays. Consequently, U-Mamba fails to completely segment the patient’s molars, while HQ-SAM produces a result with numerous holes. In contrast, our approach not only successfully segments the entire molar region but also captures more fine details, enabling it to better resist such interference and generate masks with fewer holes. Moreover, due to the high visual similarity between the cheek and the gum, previous approaches often struggle to distinguish the boundary, resulting in incorrect segmentation. However, in the third example, our approach successfully diferentiates between the gum and cheek regions, efectively suppressing noise from shadows and reflections, and thus achieves a more refined segmentation outcome.

![](images/edf6b4a8253485df60d492a83103218e19f489fed20ed52c98d320206d893449.jpg)  
Image

![](images/c2f84f750d929a3d19d2050f0b713140b43a25568d87019465b1c37973dd7aff.jpg)  
U-Mamba

![](images/9620efc75d0527803936adb01b8fa45c45a80f04f62e70c7892e8251a343d886.jpg)  
HQ-SAM

![](images/4e53d827858ca10bc9f59abad207d5131057a517507d73524a27122154eb6334.jpg)  
Ours

![](images/c960b40cc653052cec41c50880d24c0322b14ce87e9f83be3886bfddbc808206.jpg)  
GT  
Figure 7: Visualization results of U-Mamba, HQ-SAM, and our approach on DSD. Our approach generates finer segmentation masks compared to others. The blue area shows the segmentation mask, and the white dashed box highlights diferences between our approach and the others.

Table 5: Complexity comparison of diferent segmentation methods.
<table><tr><td>Approach</td><td>Params (M)</td><td>FLOPs (G)</td><td>Latency (ms)</td></tr><tr><td>Vim</td><td>26.1</td><td>41.3</td><td>20.7</td></tr><tr><td>U-Mamba</td><td>32.8</td><td>48.6</td><td>24.9</td></tr><tr><td>HQ-SAM</td><td>91.4</td><td>168.2</td><td>69.5</td></tr><tr><td>FU-Mamba (Ours)</td><td>36.2</td><td>53.1</td><td>28.2</td></tr></table>

Although FU-Mamba introduces additional modules such as the Ofset Prediction Network and frequency-domain enhancement operations, the computational overhead remains within a reasonable range. As shown in Table 5, FU-Mamba achieves a favorable trade-of between segmentation accuracy and computational eficiency. Compared with U-Mamba, the proposed approach introduces only moderate increases in parameters, FLOPs, and inference latency, while still maintaining substantially lower complexity than foundation-based models such as HQ-SAM. Specifically, FU-Mamba achieves over 2% improvement in mIoU on the DSD dataset with only 4.5G additional FLOPs and 3.3 ms extra inference latency.

## 4.5. Discussion

Dental image segmentation plays a vital role in the digital transformation of clinical oral diagnosis and treatment. The proposed FU-Mamba segmentation approach builds upon the UNet architecture by integrating the advantages of a dynamic scanning mechanism and a frequency-balancing strategy, thereby enabling the generation of high-quality segmentation results for dental images.

The experimental findings derived from both the dental segmentation dataset and the Oralvision dataset illustrate the eficacy of the proposed methodology. In particular, the dynamic scanning mechanism is shown to more efectively capture local information, which in turn improves the model’s adaptability to diverse image content. Furthermore, by balancing high- and low-frequency features, our approach improves dental segmentation performance in challenging oral environments, such as those afected by high-frequency noise and poor lighting conditions.

The DMB in FU-Mamba demonstrates superior performance in dental image segmentation, primarily due to its efective modeling of the spatial continuity inherent in oral anatomical structures. By adaptively predicting sampling ofsets, the model reorganizes image patches in a structure-aware manner that better aligns with the semantic layout of the image. This optimized sequence enhances the state space model’s capacity to learn spatial dependencies among various regions, thereby improving its ability to delineate boundaries between teeth and adjacent tissues. As a result, this approach markedly advances the modeling of essential structures and contributes to the production of high-quality, detailed segmentation masks.

The frequency domain enhancement block demonstrates notable advantages under challenging conditions such as low illumination or imaging noise, making it a critical component in enhancing the segmentation performance of FU-

Mamba. Specifically, FEB integrates wavelet decomposition with frequencyaware pooling to amplify mid-frequency features that carry structural semantics while simultaneously achieving a balanced representation between local details and global structures. Consequently, the module enables the model to retain essential shape information and boundary cues even when fine image details are degraded.

![](images/dc38cc01d8067007a3b69bf555452d5761a7c7fd0926c9722d3554324ac8a5ca.jpg)

![](images/a4b516b6f26e90a36d6fb3af3920308385d6df240230526d5c8819418639d75e.jpg)

![](images/711c3c06b9e2c890dd1b4d59199b2fc39af3f09b25d3754331f03756d0f3290b.jpg)  
Image

![](images/715ef75d35b758cc6455275fc88230262ed2248d52b9986709dc1a4f3976844d.jpg)

![](images/ead3651c49bff3f9e07b5e7e7f9db233b1dcfb040791dcdb2e56e45c89034056.jpg)  
Ours

![](images/5581d2bb95d9bff94fd0fc0a1795f2c12c8deb8b1bcd8a55401f08c05c376c82.jpg)  
GT  
Figure 8: Failure cases of our approach in DSD. The blue regions represent the masks, and the white dashed boxes highlight segmentation diferences.

As illustrated in Fig. 8, although the proposed method achieves robust performance on most dental images, failure cases still occur under extremely challenging conditions, such as severe noise, strong reflections, low contrast, and heavy dental calculus occlusion. To further evaluate robustness, we additionally analyzed the failure distribution on the DSD test set and observed that approximately 6.5% of the samples showed noticeable segmentation degradation, mainly involving boundary leakage and local structure omission. From a module-level perspective, these failures are primarily related to the limitations of dynamic ofset estimation and frequency-domain enhancement under adverse imaging conditions. In the DMB module, unreliable local features may cause the Ofset Prediction Network to generate unstable sampling ofsets, leading to inaccurate structural aggregation during SSM modeling. Meanwhile, although the Frequency Enhancement Block enhances discriminative frequency components, its efectiveness may decrease when high-frequency noise and reflection artifacts dominate the spectrum, making boundary-related information dificult to distinguish from interference signals. These observations suggest that improving the robustness of dynamic sampling and adaptive frequency modeling under extreme imaging conditions remains an important direction for future research.

Future work will focus on addressing the current limitations by incorporating multimodal data and enhancing frequency-aware processing. Specifically, introducing additional modalities such as depth or near-infrared images may provide complementary structural information, thereby improving robustness in occluded or low-contrast regions. Moreover, refining the frequency domain enhancement block to adaptively adjust the balance between high- and lowfrequency components based on local context may further suppress noise and improve boundary precision.

## 5. Conclusion

This paper presents FU-Mamba, a dental image segmentation approach that integrates a dynamic scanning mechanism with a frequency domain enhancement block. The experimental findings indicate that the proposed methodology attains competitive segmentation performance across two dental datasets. Specifically, the DMB generates reordered sequences based on adaptively adjusted positions, thereby enabling more accurate capture of critical structures during segmentation. Concurrently, the FEB facilitates the equilibrium of spec tral components, thereby augmenting the model’s ability to accurately depict intricate frequency characteristics. However, the model still exhibits limitations in capturing fine-grained structures under noisy conditions, which may result in incomplete or inaccurate boundary delineation. Future work will focus on incorporating multi-modal data and improving frequency-aware modeling to further enhance segmentation robustness and accuracy.

## Declaration of Competing Interest

All authors declare that they have no conflicts of interest with respect to the publication of this article.

## Availability of Data and Materials

Data supporting the findings of this study are available from the corresponding author on a reasonable request.

## Acknowledgements

This work was supported by the Zhejiang Province Natural Science Foundation (No. LZ24F020001), the Open Project Program of State Key Laboratory of Advanced Medical Materials and Devices (No. SQ2022SKL01089-2025-14), and the Open Project Program of State Key Laboratory of Virtual Reality Technology and Systems, Beihang University (No. VRLAB2026B13).

## References

[1] Y. Peng, H. Zou, K. Yang, Q. Kong, Adml: Asymmetric deformationguided mutual learning for semi-supervised medical image segmentation, Neurocomputing 675 (2026) 132940.

[2] Y. Tian, G. Jian, J. Wang, H. Chen, L. Pan, Z. Xu, J. Li, R. Wang, A revised approach to orthodontic treatment monitoring from oralscan video, IEEE Journal of Biomedical and Health Informatics 27 (12) (2023) 5827– 5836.

[3] Y. Tian, H. Fu, H. Wang, Y. Liu, Z. Xu, H. Chen, J. Li, R. Wang, RGB oralscan video-based orthodontic treatment monitoring, Science China Information Sciences 67 (1) (2024) 112107.

[4] P. Azad, A. Heidari, C. G. Akcora, A. Khonsari, S. H. Rastegar, A unified graph neural network-based approach for few-shot learning with task nodes and difpool abstraction, Neurocomputing 676 (2026) 133003.

[5] C. Lu, Y. Schroecker, A. Gu, E. Parisotto, J. Foerster, S. Singh, F. Behbahani, Structured state space models for in-context reinforcement learning, Advances in Neural Information Processing Systems 36 (2024) 47016– 47031.

[6] A. Gu, T. Dao, Mamba: Linear-time sequence modeling with selective state spaces, in: International Conference on Language Modeling, 2024, pp. 1–12.

[7] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, X. Wang, Vision mamba: Eficient visual representation learning with bidirectional state space model, in: Proceedings of the International Conference on Machine Learning, 2024, pp. 62429–62442.

[8] C. Yang, Z. Chen, M. Espinosa, L. Ericsson, Z. Wang, J. Liu, E. J. Crowley, Plainmamba: Improving non-hierarchical mamba in visual recognition, in: British Machine Vision Conference, 2024, pp. 1–19.

[9] T. Huang, X. Pei, S. You, F. Wang, C. Qian, C. Xu, Localmamba: Visual state space model with windowed selective scan, in: European Conference on Computer Vision, 2024, pp. 12–22.

[10] Y. Tian, P. Xue, W. Ding, M. Hassaballah, K. Egiazarian, A. Conci, A. Sengur, L. Rutkowski, DM-CFO: A difusion model for compositional 3D tooth generation with collision-free optimization, IEEE Transactions on Visualization and Computer Graphics (2026) 1–13.

[11] L. Liu, M. Zhang, J. Yin, T. Liu, W. Ji, Y. Piao, H. Lu, Defmamba: Deformable visual state space model, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 8838– 8847.

[12] S. E. Finder, R. Amoyal, E. Treister, O. Freifeld, Wavelet convolutions for large receptive fields, in: European Conference on Computer Vision, 2024, pp. 363–380.

[13] G. Yun, J. Yoo, K. Kim, J. Lee, D. H. Kim, Spanet: Frequency-balancing token mixer using spectral pooling aggregation modulation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 6113–6124.

[14] Y. Tian, G. Cheng, J. Gelernter, S. Yu, C. Song, B. Yang, Joint temporal context exploitation and active learning for video segmentation, Pattern Recognition 100 (2020) 107158.

[15] O. Contributors, Oralvision dataset, available at: https://bit.ly/3Qry3Ry (2024).

[16] X. Zhao, J. Jiang, Y. Tian, L. Wu, Z. Xu, W.-f. Yang, Y. Zou, X. Wang, Innovative tooth segmentation using hierarchical features and bidirectional sequence modeling, Pattern Recognition 175 (2026) 113045.

[17] H. Awari, N. Subramani, A. Janagaraj, G. Balasubramaniapillai Thanammal, J. Thangarasu, R. Kohar, Three-dimensional dental image segmentation and classification using deep learning with tunicate swarm algorithm, Expert Systems 41 (6) (2024) e13198.

[18] Y. Tian, Y. Zhang, W.-G. Chen, D. Liu, H. Wang, H. Xu, J. Han, Y. Ge, 3D tooth instance segmentation learning objectness and afinity in point cloud, ACM Transactions on Multimedia Computing, Communications, and Applications 18 (4) (2022) 1–16.

[19] C. Chen, J. Han, K. Debattista, Virtual category learning: A semisupervised learning method for dense prediction with extremely limited labels, IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (8) (2024) 5595–5611.

[20] D. Wenjie, W. Yuanjun, Combined layer-by-layer segmentation level set model applied to dental CBCT image segmentation, The Journal of Dentists 12 (2024) 44–50.

[21] R. Joshi, Segmentation of teeth in panoramic x-ray image using u-net algorithm, in: International Conference on Artificial Intelligence For Internet of Things, 2024, pp. 1–6.

[22] P. Harsh, R. Chakraborty, S. Tripathi, K. Sharma, Attention u-net architecture for dental image segmentation, in: International Conference on Intelligent Technologies, 2021, pp. 1–5.

[23] S. Hou, T. Zhou, Y. Liu, P. Dang, H. Lu, H. Shi, Teeth u-net: A segmentation model of dental panoramic x-ray images for context semantics and contrast enhancement, Computers in Biology and Medicine 152 (2023) 106296.

[24] S. Lin, X. Hao, Y. Liu, D. Yan, J. Liu, M. Zhong, Lightweight deep learning methods for panoramic dental x-ray image segmentation, Neural Comput ing and Applications 35 (11) (2023) 8295–8306.

[25] C. Li, X. Liu, W. Li, C. Wang, H. Liu, Y. Liu, Z. Chen, Y. Yuan, U-kan makes strong backbone for medical image segmentation and generation, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 4652–4660.

[26] R. Wu, Y. Liu, G. Ning, P. Liang, Q. Chang, Ultralight vm-unet: Parallel vision mamba significantly reduces parameters for skin lesion segmentation, Patterns 6 (11) (2025).

[27] S. Liu, L. Bao, Q. Yang, W. Geng, B. Zheng, C. Li, W. Chen, H. Peng, Y. Yuan, MedSAM-Agent: Empowering interactive medical image segmentation with multi-turn agentic reinforcement learning, arXiv preprint arXiv:2602.03320 (2026).

[28] Y. Huang, C. Li, H. Liu, H. Xu, Y. Huang, X. Ding, Y. Yuan, P2SAM: Probabilistically prompted SAMs are eficient segmentator for ambiguous medical images, in: Proceedings of the ACM International Conference on Multimedia, 2024, pp. 9779–9788.

[29] C. Li, Y. Huang, W. Li, H. Liu, X. Liu, Q. Xu, Z. Chen, Y. Huang, Y. Yuan, Flaws can be applause: Unleashing potential of segmenting ambiguous objects in SAM, in: Advances in Neural Information Processing Systems, 2024, pp. 45578–45599.

[30] C. Li, X. Liu, C. Wang, Y. Liu, W. Yu, J. Shao, Y. Yuan, GTP-4O: Modality-prompted heterogeneous graph learning for omni-modal biomedical representation, in: European Conference on Computer Vision, 2024, pp. 168–187.

[31] L. Sun, C. Li, X. Ding, Y. Huang, G. Wang, Y. Yu, Few-shot medical image segmentation using a global correlation network with discriminative embedding, Computers in Biology and Medicine 140 (2022) 105067.

[32] Y. Liu, D. Zhang, Q. Zhang, J. Han, Part-object relational visual saliency, IEEE Transactions on Pattern Analysis and Machine Intelligence 44 (7) (2021) 3688–3704.

[33] Y. Liu, C. Li, S. Xu, J. Han, Part-whole relational fusion towards multimodal scene understanding, International Journal of Computer Vision 133 (7) (2025) 4483–4503.

[34] Y. Yue, Z. Li, Medmamba: Vision mamba for medical image classification, CoRR abs/2403.03849 (2024).

[35] M. Zhang, Y. Yu, S. Jin, L. Gu, T. Lin, X. Tao, Vm-unet-v2: Rethinking vision mamba unet for medical image segmentation, in: Bioinformatics Research and Applications International Symposium, Vol. 14954, 2024, pp. 335–346.

[36] X. Ma, X. Zhang, M.-O. Pun, Rs<sup>3</sup>mamba: Visual state space model for remote sensing image semantic segmentation, IEEE Geoscience and Remote Sensing Letters 21 (2024) 1–5.

[37] L. Wang, D. Li, S. Dong, X. Meng, X. Zhang, D. Hong, Pyramidmamba: Rethinking pyramid feature fusion with selective space state model for semantic segmentation of remote sensing imagery, International Journal of Applied Earth Observation and Geoinformation 144 (2025) 104884.

[38] Y. Hou, Y. Wang, X. Xia, Y. Tian, Z. Li, T. Q. Quek, Toward secure SAR image generation via federated angle-aware generative difusion framework, IEEE Internet of Things Journal 13 (2) (2025) 2713–2730.

[39] Y. Hou, S. Zhao, X. Xia, M. Liwang, Z. Li, N. Xu, D. Wu, Y. Tian, T. Q. Quek, FedC-DAC: A federated clustering with dynamic aggregation and calibration method for SAR image target recognition, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 19 (2025) 3726–3745.

[40] Z. Zhao, P. He, Yolo-mamba: Object detection method for infrared aerial images, Signal Image Video Processing 18 (12) (2024) 8793–8803.

[41] H. He, Y. Bai, J. Zhang, Q. He, H. Chen, Z. Gan, C. Wang, X. Li, G. Tian, L. Xie, MambaAD: Exploring state space models for multi-class unsupervised anomaly detection, in: Advances in Neural Information Processing Systems, 2024, pp. 71162–71187.

[42] H. Yuan, X. Li, L. Qi, T. Zhang, M.-H. Yang, S. Yan, C. C. Loy, Mamba or rwkv: Exploring high-quality and high-eficiency segment anything model, arXiv preprint arXiv:2406.19369 (2024).

[43] J. Ma, F. Li, B. Wang, U-Mamba: Enhancing long-range dependency for biomedical image segmentation (2024) 473–483.

[44] Z. Wang, J.-Q. Zheng, Y. Zhang, G. Cui, L. Li, Mamba-UNet: UNet-like pure visual mamba for medical image segmentation (2024) 2385–2394.

[45] C. Wang, X. Liu, C. Li, Y. Liu, Y. Yuan, PV-SSM: Exploring pure visual state space model for high-dimensional medical data analysis, in: IEEE International Conference on Bioinformatics and Biomedicine, 2024, pp. 2542– 2549.

[46] Z. Xing, T. Ye, Y. Yang, G. Liu, L. Zhu, Segmamba: Long-range sequential modeling mamba for 3D medical image segmentation, in: proceedings of Medical Image Computing and Computer Assisted Intervention, 2024.

[47] Z. Xing, T. Ye, Y. Yang, D. Cai, B. Gai, X.-J. Wu, et al., Segmambav2: Long-range sequential modeling mamba for general 3d medical image segmentation, IEEE Transactions on Medical Imaging 45 (1) (2025) 4–15.

[48] J. O. Caro, Y. Ju, R. Pyle, S. Dey, W. Brendel, F. Anselmi, A. Patel, Local convolutions cause an implicit bias towards high frequency adversarial examples, in: Annual Conference on Neural Information Processing Systems Workshop, 2022, pp. 1–11.

[49] K. Zhang, J. Weng, Y. Cai, S. Li, Z. Luo, Mitigating low-frequency bias: Feature recalibration and frequency attention regularization for adversarial robustness, Neural Networks 193 (2026) 108070.

[50] D. S. Ong, Y. Liu, C. Shang, G. Ding, Q. Shen, J. Han, Fixing background misclassification in few-shot object detection via product of experts, IEEE Transactions on Pattern Analysis and Machine Intelligence 48 (2) (2025) 1825–1841.

[51] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al., Segment anything, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026.

[52] L. Ke, M. Ye, M. Danelljan, Y.-W. Tai, C.-K. Tang, F. Yu, et al., Segment anything in high quality, Advances in Neural Information Processing Systems 36 (2023) 29914–29934.

[53] Y. Xiong, B. Varadarajan, L. Wu, X. Xiang, F. Xiao, C. Zhu, X. Dai, D. Wang, F. Sun, F. Iandola, et al., Eficientsam: Leveraged masked image pretraining for eficient segment anything, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16111–16121.

[54] T. Ren, S. Liu, A. Zeng, J. Lin, K. Li, H. Cao, J. Chen, X. Huang, Y. Chen, F. Yan, et al., Grounded sam: Assembling open-world models for diverse visual tasks, in: International Conference on Computer Vision, 2023, pp. 108–117.

[55] Y. Duan, W. Wang, Z. Chen, X. Zhu, L. Lu, T. Lu, Y. Qiao, H. Li, J. Dai, W. Wang, Vision-RWKV: Eficient and scalable visual perception with RWKV-like architectures, in: International Conference on Learning Representations, 2025, pp. 83166–83182.

[56] J. Hao, Y. Zhu, L. He, M. Liu, J. K. H. Tsoi, K. F. Hung, T-mamba: a unified framework with long-range dependency in dual-domain for 2D & 3D tooth segmentation, IEEE Transactions on Multimedia (2026) 1–14.

[57] F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, K. H. Maier-Hein, NNU-Net: A self-configuring method for deep learning-based biomedical image segmentation, Nature Methods 18 (2) (2021) 203–211.

[58] S. Roy, G. Koehler, C. Ulrich, M. Baumgartner, J. Petersen, F. Isensee, P. F. Jaeger, K. H. Maier-Hein, Mednext: Transformer-driven scaling of convnets for medical image segmentation, in: International Conference on Medical Image Computing and Computer-assisted Intervention, 2023, pp. 405–415.