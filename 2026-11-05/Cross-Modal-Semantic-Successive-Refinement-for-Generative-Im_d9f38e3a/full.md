> REPLACE THIS LINE WITH YOUR MANUSCRIPT ID NUMBER (DOUBLE-CLICK HERE TO EDIT) <

# Cross-Modal Semantic Successive Refinement for Generative Image Communication over Underwater Acoustic Channels

Xu Pei, Bin Wang, Member, IEEE, Guangxu Zhu, Member, IEEE, Yi Zhang, Member, IEEE and Haiyan Fan

Abstract — Underwater acoustic communication faces challenges such as limited available bandwidth and complex channel environments, which impose demanding requirements on efficient image transmission and reliable reconstruction. To address these challenges, this paper proposes a generative image communication framework based on cross-modal semantic successive refinement. At the transmitter, text descriptions, compact bottleneck features, and visual side information compressed by residual vector quantization are transmitted. At the receiver, text and bottleneck features are first fused to predict a base estimate of the intermediate visual representation. The received RVQ indices are then used to recover supplementary visual information and perform residual correction, followed by image reconstruction using a one-step Pixel MeanFlow model. During training, a CLIP-based text–image semantic consistency constraint is introduced to preserve the semantic correspondence between generated results and the original text descriptions. During transmission, protection resources are allocated according to the importance of different information branches and RVQ quantization stages. Simulations using COCO 2017 images and measured underwater acoustic channel records from Watermark show that text and bottleneck features provide complementary semantic and structural conditions, RVQ visual side information further improves reconstruction quality, and layered unequal error protection helps improve recovery at low SNRs. Code-rate allocation experiments reveal a tradeoff between the reliability of fundamental conditions and visual-detail reconstruction under a limited visual-feature coding budget, providing guidance for allocating protection resources in underwater acoustic image transmission.

Index Terms — Underwater acoustic communication, generative semantic communication, semantic successive refinement, Pixel MeanFlow, residual vector quantization, crossmodal semantic consistency.

## I. INTRODUCTION

attention because it focuses on conveying taskrelevant meaning rather than ensuring bit-level recovery. Shannon's information theory established the foundations of source compression and reliable channel transmission [1]. Weaver further distinguished the technical, semantic, and effectiveness levels of communication [2]. Early studies explored the quantification of semantic information [3], followed by Bao et al.'s work on semantic information measurement and coding [4]. More recent research has advanced SC in semantic representation, communication architectures, and task-oriented transmission, with increasing

attention to contextual information and data significance [5], [6], [7], [8], [9], [10], [11], [12]. In essence, SC aims to identify and transmit information relevant to the receiver's task, enabling the intended content to be reconstructed or interpreted from the received semantic representation and available context.

Underwater visual information is important for applications such as oceanographic observation, marine biology, environmental monitoring, and underwater exploration [13]. Because radio-frequency signals attenuate rapidly in water, acoustic waves are commonly used for underwater wireless communication. However, underwater acoustic channels are characterized by limited bandwidth, long propagation delays, multipath propagation, and Doppler effects, which make reliable image transmission challenging [13]. These constraints motivate the development of communication methods that reduce transmission requirements while preserving important visual information. Semantic communication provides a promising approach by prioritizing semantically relevant content.

Despite the potential of semantic communication for underwater image transmission, several fundamental challenges remain. First, images contain rich and diverse visual information, making it difficult to obtain compact representations while preserving the important information required for accurate reconstruction. Second, the constrained and time-varying nature of underwater acoustic channels makes the reliable transmission of visual information challenging, as channel impairments may lead to information loss and degradation. Third, reconstructing high-quality images from the information available at the receiver remains difficult, particularly when the transmitted representations are incomplete or distorted. Addressing these challenges requires effective approaches to visual representation, reliable information transmission, and image reconstruction in underwater acoustic environments.

To address these challenges, researchers have developed learning-based methods for image representation and transmission. Bourtsoulatze et al. proposed DeepJSCC, which uses a CNN encoder and decoder to jointly encode images and transmit them over a channel [14]. Later studies extended this framework by adding receiver feedback [15], combining it with OFDM [16], introducing attention modules [17], and adapting the transmission bandwidth [18]. Dai et al. proposed NTSCC, which uses learned nonlinear transforms and an entropy model to represent image information for transmission [19]. In parallel, different neural architectures have been explored to improve semantic feature extraction. CNNs are effective in capturing local visual structures, while Transformer-based models can capture long-range dependencies. For image communication, SwinJSCC employs a hierarchical Swin Transformer architecture to extract visual features for joint source-channel coding [20].To support digital transmission, some studies have explored discrete representations. DeepJSCC-Q constrains transmitted symbols to a finite constellation [21], while Hu et al. proposed a masked VQ-VAE framework that represents visual features as discrete codebook indices[22]. These methods provide different ways to represent and transmit image information.

Generative models have also been used to improve image reconstruction at the receiver. Erdemir et al. incorporated generative models into JSCC to improve perceptual quality under limited transmission resources [23]. Wu et al. proposed CDDM, which uses diffusion-based denoising to reduce channel noise [24]. Qiao et al. used pretrained diffusion models with semantic conditioning for generative semantic communication [25], while Zhang et al. incorporated semantic guidance into diffusion-based wireless image reconstruction [26]. In addition, CLIP maps images and text into a shared embedding space [27], while latent diffusion models enable conditional image generation in a compressed latent space [28]. These studies demonstrate the potential of generative models and cross-modal information for image reconstruction.

These methods have also been explored for underwater image communication. Peng et al. incorporated physical priors into underwater image semantic communication [29]. Chen et al. used large language models to understand image semantics and prioritize information, together with conditional diffusion for image recovery [30]. Liang and Ma applied adaptive RVQ to obtain compact visual representations for underwater image transmission [31]. These studies have explored different ways to use physical knowledge, language models, and discrete visual representations in underwater communication.

Overall, previous studies have advanced image representation, digital transmission, generative reconstruction, and underwater semantic communication. However, several issues remain. First, textual descriptions convey the main semantic content of an image but cannot fully represent its spatial structure and fine visual details. Although generative models can infer missing content from learned priors, the generated details may differ from those in the original image when sufficient visual information is unavailable. Second, transmitting supplementary visual information can improve reconstruction, but increases transmission overhead and exposes the additional information to channel errors. Therefore, efficient compression and reliable transmission of visual information remain important. Third, generative reconstruction needs to balance image quality and inference efficiency. Moreover, a visually plausible image does not necessarily preserve the semantics of its corresponding text description. These issues call for a framework that combines cross-modal visual recovery, efficient visual information transmission, and semantically consistent image generation.

To address these issues, we propose a cross-modal semantic successive-refinement framework for generative image communication over underwater acoustic channels. The proposed framework combines textual semantics, compact visual features, and compressed visual side information to progressively reconstruct the transmitted image. First, at the receiver, textual semantics and the received Bottleneck features are used to predict a base estimate of the intermediate visual representation. The RVQ-decoded visual side information is then used for residual correction, further refining the base estimate with additional image-specific visual details. This process improves the recovery of visual information that cannot be fully represented by the text and Bottleneck features alone. Second, RVQ converts high-dimensional visual side information into compact discrete codebook indices to reduce transmission overhead, while unequal error protection (UEP) assigns different protection levels to the transmitted information according to its importance, improving transmission reliability. Finally, the refined intermediate representation and text embedding are fed into a one-step Pixel MeanFlow model [32] to generate the final RGB image, reducing the inference cost associated with iterative generation. During training, a CLIP-based text–image consistency loss [27] is further introduced to encourage semantic alignment between the reconstructed image and its corresponding text description.

The main contributions of this paper are summarized as follows:

1) We propose a cross-modal semantic successiverefinement framework for underwater image communication. The receiver first obtains a base estimate of the intermediate visual representation from textual semantics and Bottleneck features, and then refines it using the received visual side information. A one-step Pixel MeanFlow model subsequently reconstructs the final RGB image, enabling cascaded recovery from semantic and visual representations to the pixel domain.

2) We develop an RVQ-based visual side-information compression and unequal error protection scheme. RVQ converts high-dimensional visual information into compact discrete codebook indices, reducing the amount of information required for transmission. UEP further allocates different protection levels to transmitted information according to its importance, including different RVQ quantization stages.

3) We introduce a CLIP-based cross-modal semantic consistency constraint for image reconstruction. Frozen CLIP image and text encoders are used to measure the semantic similarity between the reconstructed image and its corresponding text description. This training-time constraint encourages the generated image to preserve the textual semantics throughout the cascaded reconstruction process.

## II. GENERAL SYSTEM MODEL

## A. Multimodal Representation for Semantic Transmission

The overall architecture of the proposed generative system is shown in Fig. 1, which is capable for cross-modal data semantic transmission with successive refinement. To start with, an RGB image $X \in R ^ { 3 \times H \times W }$ is to be transmitted, let its paired text description be �, such textual description can be generated by a well-trained Vision-Language-Model (VLM). At the transmitter, a semantic encoder $E _ { s } ( \cdot )$ is leveraged to extract the several levels of visual representations of the input image. In our work, two levels ofrepresentations, named ofthe bottleneck feature $F _ { B }$ and the intermediate feature $F _ { I } ,$ , are extracted:

![](images/6052b5baedc0416d3b1481093cc0be49a2f29f019327c88fa50131faaa8c8e1d.jpg)  
Fig. 1. Overall architecture of the cross-modal semantic successive-refinement generative communication system.

$$
( F _ { B } , F _ { I } ) = E _ { s } ( X )\tag{1}
$$

Specifically, the semantic encoder is instantiated by the combination of a convolutional neural network and a Swin-Transformer-fashion [33] network. The bottleneck feature is highly compressed and mainly retains abstract information that is irrecognizable, whereas the intermediate feature is lesscompressed, which can be further used to recover more details in the reconstruction phase.

To note that, the raw image, the intermediate feature and bottleneck feature form a represenation dependency chain, i.e. intuitively both the intermediate and bottleneck feature are the representation of the raw image, the bottleneck feature can be also regarded as the representation of the intermediate feature. Such chain is useful to illustrate the following hierarchical reconstruction.

The text description � provides cross-modal information about object categories, scene relationships, and overall semantics. These different levels or types of semantic information play distinct but complementary roles during the reconstruction process at the receiver.

During the practical data transmission, the complete intermediate feature contains a large amount of data, which significantly increase the transmission overhead. Therefore, a compression method should be specifically applied to the intermediate feature. Unlike the bottleneck feature, which achieves compression through task-oriented semantic abstraction and may discard fine-grained feature information, the proposed approach adopts a separate feature-level compression pathway (e.g., vector quantization). This pathway directly compresses the intermediate feature for transmission, aiming to reduce communication overhead while preserving its original feature representation as much as possible.

Denoting the compression process of the intermediate feature as ����(·), the compression result of the massive intermediate information is

$$
S = C o m p ( F _ { I } )\tag{2}
$$

where � represents the intermediate compression information, which is actually transmitted over the underwater acoustic link. At the receiver side, the decoder jointly uses textual semantics, bottleneck features as well as the intermediate feature that is additionally recovered from the intermediate compression information � .

## B. Cascaded Reconstruction for Generative Reception

As mentioned above, the textual description, bottleneck features, and compressed intermediate features are gathered, encoded, and transmitted through the underwater acoustic channel. Conventionally, these heterogeneous information sources are assembled into a single transmission block and protected using the same channel coding scheme, thereby receiving a uniform level of error protection.

However, such a strategy overlooks their distinct roles in receiver-side reconstruction. Specifically, the textual description provides explicit high-level semantic guidance, conveying essential information such as object categories, scene attributes, and semantic relationships. The bottleneck features encode abstract semantic representations and global contextual information, which are crucial for maintaining the overall scene structure and semantic consistency. In contrast, the intermediate features preserve richer spatial information and fine-grained details, such as object boundaries, local structures, and textures, complementing the semantic abstraction of the bottleneck features.

Consequently, transmission errors in these three information streams may lead to different types and degrees of reconstruction degradation. This motivates the adoption of unequal error protection (UEP), whereby different levels of channel protection are assigned to each information stream according to its contribution and sensitivity to reconstruction quality. Each stream is then quantized and transmitted to the receiver through a digital communication link with an appropriately configured error-protection level. The specific underwater acoustic channel configuration is introduced in the experimental settings in Section IV.

To elaborate, the receiver performs image reconstruction through two cascaded generative stages. First, the received text description $\hat { T }$ is converted into a text embedding $\boldsymbol { { \widehat { v } } } _ { T }$ by a frozen CLIP Text Encoder [27]. Together with the recovered bottleneck feature, this embedding is fed into the feature reconstruction network, named as the Inter Predictor $P _ { \theta }$ , to obtain an initial prediction of the visual-side intermediate representation:

$$
\hat { F } _ { I } ^ { P } = P _ { \theta } \big ( \hat { v } _ { T } , \hat { F } _ { B } \big ) \in R ^ { C _ { I } \times H _ { I } \times W _ { I } }\tag{3}
$$

where $\hat { F } _ { B }$ denotes the received bottleneck feature after the acoustic channel. $P _ { \theta } ( \cdot )$ denotes the feature-domain Inter Predictor.

As the predicted intermediate feature $\hat { F } _ { I } ^ { P }$ is obtained, the receiver uses the recovered visual-side information to correct the prediction and obtain a more enhanced intermediate visual representation. This process of intermediate feature enhancement can be expressed in a compact form as

$$
\widehat { F _ { I } } = F _ { \psi } \big ( \widehat { F } _ { I } ^ { P } , \widehat { S } \big ) \in R ^ { C _ { I } \times H _ { I } \times } \mathsf { W } _ { I }\tag{4}
$$

where $\hat { S }$ denotes the intermediate compression information recovered from the acoustic channel and $F _ { \psi } ( \cdot )$ descripts the procedure of a CNN-based intermediate feature recovery and a subsequent fusion operation.

To elaborate, the Equation (4) is envisioned by the RVQ scheme that is mainly adopted in this paper – the received discrete indices are used to select the subset of the shared codebook with transmitter, then obtain the recovered intermediate compression information $\hat { S }$ by aggregating all the selected codewords. The predicted intermediate representation $\hat { F } _ { I } ^ { P }$ is further enhanced to $\widehat { F } _ { I }$ with the assistance of �<sup>̂</sup>.

After the restored enhanced intermediate representation $\widehat { F } _ { I }$ has been obtained, in the second stage, a one-step Pixel MeanFlow generator reconstructs the image in pixel-domain. The recovered intermediate representation and text embedding are jointly leveraged as generation conditions, yielding the reconstructed image:

$$
\widehat { X } \sim P _ { \omega } \left( X \vert \widehat { v } _ { T } , \widehat { F } _ { I } \right)\tag{5}
$$

where $P _ { \omega } \big ( X \big | \widehat { v } _ { T } , \widehat { F } _ { I } \big )$ denotes the conditional image distribution modeled by the one-step Pixel MeanFlow generator. Through the two-stage cascaded reconstruction process, the receiver first recovers a richer intermediate representation from compact cross-modal semantic conditions and then reconstructs a pixeldomain image from this representation and, consequently, achieving semantic successive refinement across representation domains.

Additionally, to maintain semantic consistency between the progressively reconstructed image and its corresponding textual description, a cross-modal semantic consistency loss is introduced. Such loss encourages the reconstructed visual content to preserve the high-level semantic information conveyed by the transmitted text, thereby promoting semantic alignment across modalities. Specifically, the reconstructed image $\hat { X }$ and its corresponding received textual description �<sup>̂</sup> are fed into the image encoder and text encoder of a pretrained CLIP model as

$$
\left\{ \begin{array} { l } { { \hat { v } _ { I } = C L I P _ { I } ( \hat { X } ) } } \\ { { \hat { v } _ { T } = C L I P _ { T } ( \hat { T } ) } } \end{array} \right.\tag{6}
$$

respectively, the resulting image and text embeddings are mapped into a shared semantic space, where their cosine similarity is calculated to quantify the cross-modal semantic consistency,

$$
\mathcal { L } _ { C L I P } = 1 - \boldsymbol { \widehat { v } _ { I } } ^ { T } \boldsymbol { \widehat { v } _ { T } }\tag{7}
$$

which is introduced to the training process of the multimodal semantic receiver.

## III. IMPLEMENTATION DETAILS

The previous section presents a general formulation of the proposed system end-to-end. In this section, the details of the realization are elaborated.

## A. Hybrid Source and Channel Coding

To start with, we introduce the transmission procedure. In conventional communication systems, the data to be transmitted are first processed by a source encoder, which removes statistical redundancy to reduce the data volume, such as JPEG for image compression and H.264 or H.265 for video compression. The resulting compressed bitstream is then fed into a channel encoder, which introduces redundancy to improve transmission reliability over noisy channels. For example, Polar codes are adopted for control-channel coding, while LDPC codes are employed for data-channel coding in 5G New Radio (NR).

In contrast to the conventional paradigm, our transmitter adopts a hybrid approach to multimodal information representation and transmission. Specifically, it takes an image � and its corresponding textual description $T ,$ generated by a pretrained vision-language model (VLM), as inputs. For the image modality, the semantic encoder $E _ { s }$ extracts two feature representations at different abstraction levels, namely, the bottleneck feature $F _ { B }$ and the intermediate feature $F _ { I }$ as defined in (1). To reduce the transmission overhead associated with the intermediate feature, $F _ { I }$ is further compressed to obtain a compact representation � . For the textual modality, the original text $\bar { T }$ is retained as the source representation and directly used for transmission, without additional semantic feature extraction.

Once these multimodal representations, i.e., $F _ { B } , S ,$ , and $T ,$ are obtained, they are independently fed into dedicated channel encoders. Since the three representations play distinct roles in receiver-side reconstruction and exhibit different sensitivities to transmission errors, different levels of channel protection are assigned to each stream. This enables the transmitter to allocate channel-coding redundancy according to the respective importance of each information stream.

After the source representation and bitstream preparation, the three resulting bitstreams corresponding to $F _ { B } , S$ , and $T$ are separately fed into three independent LDPC encoders, each configured with a different code rate. Specifically, let $K _ { j }$ and $N _ { j }$ denote the input and output bit lengths of the j-th stream, respectively. Its effective code rate is defined as

$$
\mathcal { R } _ { j } = K _ { j } / N _ { j } , \qquad j \in B , S , T .\tag{8}
$$

By assigning different code rates to the three streams according to their respective importance and sensitivity to transmission errors, unequal error protection (UEP) is achieved. In particular, a lower code rate introduces more redundancy and generally provides stronger error protection. The three channel-coded bitstreams are then concatenated in a predefined order to form a single coded bitstream:

$$
b = b _ { B } | | b _ { S } | | b _ { T }\tag{9}
$$

where $b _ { B } , b _ { S }$ , and $b _ { T }$ denote the LDPC-coded bitstreams, respectively, and || represents bitstream concatenation. Finally, the concatenated bitstream is mapped onto QPSK symbols, with

![](images/1363e79a7ab7206613e06fce864b713d90f0cbb329aad50c20641ab55c7cf59a.jpg)  
Fig. 2. RVQ-based Intermediate information transmission.

every two consecutive coded bits forming one modulation symbol, yielding the transmitted symbol sequence:

$$
\phi _ { m } = \mu _ { \mathrm { Q P S K } } ( b _ { 2 m - 1 } , b _ { 2 m } )\tag{10}
$$

Such symbols are furtherly modulated with OFDM scheme and transmitted into the noisy acoustic channel environment.

## B. Base Prediction of Intermediate Feature Reconstruction

At the receiver, the received acoustic signal undergoes the corresponding reverse processing procedure, including QPSK demodulation, bitstream separation, and LDPC decoding. This yields three recovered multimodal representations, namely, the bottleneck feature ${ \hat { F } } _ { B } ,$ the intermediate feature indices $\boldsymbol { \dot { \hat { S } } } ,$ and the textual description $\hat { T }$ . Based on these decoded representations, the receiver proceeds with the reconstruction process.

In the first reconstruction branch, the receiver generates a base prediction of the intermediate feature $\hat { F } _ { I } ^ { P }$ by leveraging the decoded bottleneck feature ${ \hat { F } } _ { B }$ and the received textual description ${ \hat { T } } ,$ as formulated in (3). Specifically, to incorporate textual information as a semantic condition, $\hat { T }$ is first fed into a frozen CLIP text encoder to obtain a global semantic embedding the same as (6)

$$
\widehat { v } _ { T } = C L I P _ { T } ( \widehat { T } )\tag{11}
$$

This embedding is then transformed into a set of textconditioning tokens through a learnable multilayer perceptron (MLP).

$$
\hat { C } _ { T } = M L P _ { \eta } ( \hat { v } _ { T } )\tag{12}
$$

The text-conditioning tokens provide high-level semantic cues, including object categories and scene relationships.

In parallel, the bottleneck feature $\hat { F } _ { B }$ is processed by a CNNbased projection network $G _ { \phi }$ to generate visual-conditioning tokens while preserving its spatial layout,

$$
\hat { V } _ { B } ^ { \bar { ( ) } } = \hat { G } _ { \phi } \big ( \hat { F } _ { B } \big )\tag{13}
$$

These tokens provide complementary visual structural information, such as spatial organization and coarse structural characteristics. By jointly exploiting the semantic guidance from the text and the structural priors encoded in the bottleneck feature, the Inter Predictor generates a base prediction of the intermediate feature, which serves as the initial representation for subsequent residual refinement.

The Inter Predictor is implemented as a conditional prediction network consisting of eight stacked Transformer blocks. Specifically, the text-conditioning tokens and bottleneck-conditioning tokens are first concatenated along the token dimension to form a unified multimodal sequence:

$$
Z _ { 0 } = \left[ \hat { C } _ { T } ; \hat { V } _ { B } \right]\tag{14}
$$

where $\hat { C } _ { T }$ and $\hat { V } _ { B }$ denote the text-conditioning and visualconditioning tokens, respectively. The concatenated sequence is then processed by the Transformer blocks, where multi-head self-attention enables interactions between textual and visual representations, facilitating the fusion of high-level semantic cues and spatial structural information. The resulting multimodal representation is subsequently passed through an output projection layer to generate the predicted intermediate feature:

$$
\hat { F } _ { I } ^ { P } = G _ { o u t } \left( T _ { \theta } ^ { ( 8 ) } ( Z _ { 0 } ) \right)\tag{15}
$$

where $T _ { \boldsymbol { \theta } } ^ { ( 8 ) }$ denotes the stack of eight Transformer blocks and $G _ { o u t }$ represents the output projection.

During training, the ground-truth intermediate feature $F _ { I }$ extracted by the transmitter-side semantic encoder, serves as the supervision target. To ensure consistent feature scaling, the prediction and ground-truth feature are normalized before calculating the mean squared error (MSE) loss:

$$
\mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { D _ { I } } \big \| N \big ( \widehat { F } _ { I } ^ { P } \big ) - N ( F _ { I } ) \big \| _ { F } ^ { 2 }\tag{16}
$$

where �(·) denotes the feature normalization operation and $D _ { I }$ is the total number of elements in the intermediate feature. By minimizing this loss, the Inter Predictor learns to reconstruct the intermediate visual representation from the compact multimodal conditioning information.

After training, the Inter Predictor is frozen to generate a base estimate of the intermediate feature from the decoded bottleneck feature and textual description. When only these two sources are transmitted, the base estimate provides visual conditioning for downstream image reconstruction. When RVQ indices are additionally received, the decoded residual features are used to further refine the estimate. By maintaining a fixed prediction baseline, the frozen Inter Predictor provides a stable reference for the subsequent residual refinement and feature fusion.

## C. RVQ-based Intermediate Feature Residue Prediction

The frozen Inter Predictor described in the previous subsection generates a base estimate of the intermediate feature $\hat { F } _ { I } ^ { P }$ from the received bottleneck feature ${ \hat { F } } _ { B }$ and textual description $\hat { T } .$ . Although this estimate approximates the groundtruth intermediate feature $F _ { I } { \mathrm { : } }$ , fine-grained visual details cannot be fully recovered from these compact conditions alone. Therefore, additional visual side information is transmitted and used to learn the residue part of $F _ { I }$ , in order to refine the base estimate.

To elaborate this residue branch, specifically, let’s revisit equation (2) in the transmission part, $F _ { I }$ in compressed by a CNN-based encoder $S _ { e n c } ,$ as illustrated in Fig. 2. This encoder reduces its spatial resolution and channel dimensionality while preserving information relevant to receiver-side refinement. It is trained end-to-end to retain the visual details that cannot be adequately recovered by the Inter Predictor. The encoder outputs a compressed feature tensor

$$
S _ { r a w } \in R ^ { C _ { S } \times H _ { S } \times W _ { S } }\tag{17}
$$

Before quantization, $S _ { r a w }$ is reshaped into a matrix by flattening its spatial dimensions:

$$
S \stackrel { \cdot } { = } r e s h a p e ( S _ { r a w } ) ^ { T } \in R ^ { H _ { S } W _ { S } \times C _ { S } }\tag{18}
$$

Consequently, each row of � represents a �-dimensional latent vector, resulting in $H _ { S } W _ { S }$ latent vectors in total. These vectors are then quantized using a multistage Residual Vector Quantizer (RVQ)[34]. Specifically, at each quantization stage, the residuals of all latent vectors are quantized using a shared codebook. Different quantization stages employ distinct codebooks, while all codewords within a codebook have the same dimensionality $C _ { S }$ as the latent vectors. The resulting discrete codebook indices represent the quantized form of � and are transmitted over the digital link. By varying the number of quantization stages, RVQ supports different trade-offs between transmission overhead and reconstruction fidelity.

RVQ uses multiple quantization stages to encode the residual error that remains insufficiently represented by the preceding stage. First, the original latent representation is used as the initial quantization residual, i.e., $r _ { 0 } = S$ . At the � -th quantization stage, quantizer $Q ( r ; R )$ selects the nearest codeword from its codebook � according to the current residual � , and the selected codeword is denoted by � . The portion already represented by this codeword is then subtracted from the current residual:

$$
[ q _ { l } , i _ { l } ] = Q ( r _ { l - 1 } , R _ { l } )\tag{19a}
$$

$$
r _ { l } = r _ { l - 1 } - q _ { l } = r _ { l - 1 } - R _ { l } \{ i _ { l } \}\tag{19b}
$$

where $q _ { l }$ represents the codeword obtained at quantization stage $l , i _ { l }$ denotes the index of $q _ { l }$ in the l-th level codebook $R _ { l }$ and $r _ { l }$ denotes the residual error that remains unrepresented after this stage. Thus, the first stage provides a coarse approximation of the original latent feature, while subsequent stages continue to represent visual information not adequately captured by the preceding stages. What is actually sent through the digital link is not the codeword vectors $q _ { l }$ themselves, but the codebook indices $i _ { l }$ selected by quantizer of each stage, thereby substantially reducing the amount of transmitted visual side information.

At the reception part, with � RVQ stages, the receiver retrieves the corresponding indices of codewords from the shared codebooks with transmitter using the received indices and summates them to obtain the recovered latent visual representation:

$$
\hat { S } = \sum _ { l = 1 } ^ { L } R _ { l } \{ \hat { \boldsymbol { \imath } } _ { l } \}\tag{20}
$$

where $\hat { \iota } _ { l }$ denotes the received index at stage �. It can be seen that, due to the erroneous channel, the codeword indices may deviate from their correct values and in turn, causes the error latent vectors. To solve this, a learnable error-correction network is inserted to mitigate the effects from the codeword indices error.

In particular, as shown in Fig. 2 the erroneous $\hat { S }$ is firstly decompressed by another CNN-based decoder $S _ { d e c } ,$ which is a reverse procedure of $S _ { e n c } .$ The output of the decoder is

$$
\widehat { F } _ { I } ^ { d e c } = S _ { d e c } ( \widehat { S } ) \in R ^ { \bar { C } } \overline { { d e c } } ^ { \times H _ { I } \times W _ { I } }\tag{21}
$$

whose dimension is slightly different from $\widehat { F } _ { I } \in R ^ { C _ { I } \times H _ { I } \times W _ { I } }$ Then, the $\hat { F } _ { I } ^ { P }$ and $\hat { F } _ { I } ^ { d e c }$ are concatenated as the input of an errorcorrection network $C _ { R V Q }$ to predict the difference between the ground truth intermediate feature $F _ { I }$ and the base-estimated one $\widehat { F } _ { I } ^ { P }$ . Finally, they are summed up to generate a more accurate intermediate feature

$$
\hat { F } _ { I } = \hat { F } _ { I } ^ { P } + C _ { \mathrm { R V Q } } \big ( \mathrm { C o n c a t } \big [ \hat { F } _ { I } ^ { d e c } , \hat { F } _ { I } ^ { P } \big ] \big )\tag{22}
$$

which is the detail implementation of equation (4).

Based on the base estimate $\hat { F } _ { I } ^ { P }$ generated by the frozen Inter Predictor (Eq. (15)), we further train the RVQ-based residual refinement branch. During this process, the Inter Predictor remains frozen, while the original intermediate feature $F _ { I }$ serves as the reconstruction target.

To enable effective reconstruction with different numbers of RVQ stages, a stage-wise reconstruction supervision strategy is applied. Specifically, for a three-stage RVQ configuration, three reconstruction cases are considered, using the cumulative quantized representations from the first stage, the first two stages, and all three stages, respectively. The corresponding latent representations are given by

$$
\hat { S } ^ { ( k ) } = \sum _ { l = 1 } ^ { k } R _ { l } \left\{ \hat { \imath } _ { l } \right\} , \quad k \in \{ 1 , 2 , 3 \}\tag{23}
$$

Each cumulative representation is processed by the same CNNbased decoder $S _ { d e c }$ and the same error-correction network $C _ { R V Q } .$ Together with the fixed base estimate $\widehat { F } _ { I } ^ { P }$ , the decoded feature is used to generate an enhanced intermediate feature:

$$
\widehat { F } _ { I } ^ { ( k ) } = \widehat { F } _ { I } ^ { P } + C _ { R V Q } \big ( \mathrm { C o n c a t } \big [ S _ { \mathrm { d e c } } \big ( \hat { S } ^ { ( k ) } \big ) , \widehat { F } _ { I } ^ { P } \big ] \big )\tag{24}
$$

Accordingly, three reconstructed features, $\widehat { F } _ { I } ^ { ( 1 ) } , \widehat { F } _ { I } ^ { ( 2 ) }$ , and $\widehat { F } _ { I } ^ { ( 3 ) }$ are obtained and individually supervised against the groundtruth intermediate feature $F _ { I }$ . The overall reconstruction loss is defined as the average of the three stage-wise reconstruction losses:

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { 3 } \sum _ { k = 1 } ^ { 3 } \mathsf { M S E } \big ( \widehat { F } _ { I } ^ { ( k ) } , F _ { I } \big )\tag{25}
$$

![](images/715b093e1e3d11b5ac0fa8e3e7473fb5435fb953f7a0b86eaec9ae4380cee6fd.jpg)  
Fig. 3. Image reconstruction using Pixel MeanFlow.

By jointly optimizing these three reconstruction cases, the residual refinement branch is explicitly trained to exploit different levels of RVQ information. In particular, the firststage supervision ensures that even a single quantization stage can provide useful residual correction, while subsequent stages progressively improve the reconstruction by incorporating additional residual information. Consequently, the convolutional encoder $S _ { e n c } ,$ latent decoder $S _ { d e c } ,$ and correction network $C _ { R V Q }$ are optimized jointly, and the whole residue branch network is encouraged to maintain effective reconstruction performance at different RVQ depths, rather than relying solely on the complete three-stage representation.

In addition to the reconstruction loss, a commitment loss is introduced to encourage the input to each quantizer to remain close to its selected codewords. For the three-stage model, this loss and the combined training objective are expressed as

$$
\mathcal { L } _ { c o m } = \frac { 1 } { 3 } \sum _ { l = 1 } ^ { 3 } M S E \big ( r _ { l - 1 } , s g ( q _ { l } ) \big )\tag{26a}
$$

$$
\mathcal { L } _ { R V Q } = \mathcal { L } _ { r e c } + \lambda _ { c o m } \mathcal { L } _ { c o m }\tag{26b}
$$

respectively, where $r _ { l - 1 }$ is the input residual at stage �, with $r _ { 0 } = S$ , and $s g ( \cdot )$ denotes the stop-gradient operation [31], which preserves its argument during forward computation but prevents gradients from propagating through it. The coefficient $\lambda _ { c o m }$ controls the weight of the commitment loss. In summary, these objectives encourage the encoder to produce latent representations that support reconstruction at different refinement levels while remaining amenable to quantization.

The residual branch network is optimized through backpropagation. Since RVQ involves discrete codeword selection, a straight-through estimator (STE) is employed to enable reconstruction gradients to propagate through the quantization operation to the encoder. Meanwhile, the codebooks are initialized by sampling latent vectors for the first stage and residual vectors for subsequent stages. During training, they are updated using an exponential moving average (EMA), which gradually moves each codeword toward the centroid of its assigned vectors. In this way, the encoder learns latent representations suitable for reconstruction and quantization, while the codebooks progressively adapt to the evolving latent and residual distributions. After training, the codebooks are fixed and shared between the transmitter and receiver.

D. Low-Rank Intermediate Representations Based on Tucker and CP

As alternative non-learned representations to the RVQbased approach, we further consider two low-rank tensor decomposition methods, namely, Tucker decomposition and CP decomposition [35]. Unlike RVQ, which relies on training data to learn the encoder, codebooks, and decoder, these methods directly approximate the intermediate feature tensor through low-rank decomposition, without requiring offline model training or an additional training dataset. They provide an alternative approach to visual side-information compression, enabling a comparative investigation of learned discrete representations and non-learned low-rank representations.

Given an intermediate feature tensor $F _ { I }$ Tucker decomposition approximates it using a compact core tensor and a set of factor matrices:

$$
F _ { I } \approx G \times _ { 1 } U ^ { ( 1 ) } \times _ { 2 } U ^ { ( 2 ) } \times _ { 3 } U ^ { ( 3 ) }\tag{27}
$$

where � denotes the core tensor, and $U ^ { ( 1 ) } , U ^ { ( 2 ) }$ , and $U ^ { ( 3 ) }$ are factor matrices capturing the low-rank subspaces along the respective tensor modes. By restricting the decomposition rank of each mode, Tucker decomposition exploits interdimensional correlations to reduce the representation size. The original feature can subsequently be reconstructed from the core tensor and factor matrices at the receiver.

CP decomposition provides another low-rank approximation by expressing the original tensor as a sum of rank-one components:

$$
\begin{array} { r } { F _ { I } \approx \sum _ { r = 1 } ^ { R } \lambda _ { r } a _ { r } \circ b _ { r } \circ c _ { r } } \end{array}\tag{28}
$$

where � denotes the CP rank, $a _ { r } , b _ { r }$ , and $c _ { r }$ are factor vectors along the three tensor modes, and $\lambda _ { r }$ is the weight of the �th rank-one component, and ∘ denotes the outer product. Unlike Tucker decomposition, CP does not require a separate core tensor, resulting in a more compact factorized representation. However, its approximation performance and numerical stability are sensitive to the selected rank and the decomposition procedure.

At the receiver, the recovered low-rank representations from Tucker or CP decomposition serve as additional visual side information. They are fused with the base intermediate feature prediction from the cross-modal predictor through crossattention, providing complementary visual information for subsequent reconstruction. These methods, together with RVQ, allow us to investigate the differences between learned discrete residual quantization and non-learned continuous low-rank decomposition for visual feature transmission over underwater acoustic channels.

E. Pixel MeanFlow RGB Reconstruction and Cross-Modal Semantic Consistency Loop

With the recovered intermediate feature $\hat { F } _ { I }$ and text embedding $\boldsymbol { { \widehat { v } } } _ { T }$ , the receiver employs Pixel MeanFlow model [32] to reconstruct the RGB image in a single network forward pass. The generation process is formulated through two complementary representations of the image state $X _ { t }$

The first representation defines a linear interpolation between a ground-truth image � and an independent Gaussian noise sample �:

$$
X _ { t } = ( 1 - t ) X + t \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) , \quad t \in [ 0 , 1 ]\tag{29}
$$

Here, $t = 0$ corresponds to the clean image, whereas $t = 1$ corresponds to pure noise. Since both � and � are known during training, differentiating the interpolation yields the ground-truth instantaneous velocity:

$$
\begin{array} { r } { v _ { \mathrm { t a r g e t } } = \frac { d X _ { t } } { d t } = \epsilon - X } \end{array}\tag{30}
$$

The second representation describes the same time-indexed state in terms of the displacement along a generative trajectory conditioned on the recovered intermediate feature $\hat { F } _ { I }$ and text embedding $\widehat { v } _ { T } .$ :

$$
X _ { t } = X _ { r } + ( t - r ) u \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big ) , \quad 0 \leq r < t \leq 1\tag{31}
$$

Here, $X _ { r }$ denotes the image state at the earlier target time $r _ { \mathrm { { : } } }$ while $u \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big ) = ( X _ { t } - X _ { r } ) / ( t - r )$ represents the average velocity over the interval $[ r , t ]$ . Unlike the linear interpolation in (29), this representation characterizes the net displacement along the conditional generative trajectory. Since the target state $X _ { r }$ is unavailable during generation, the average velocity is estimated by a network $u _ { \omega } \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big )$ , which uses $\hat { F } _ { I }$ and $\boldsymbol { { \widehat { v } } } _ { T }$ as visual and semantic conditions, respectively.

Differentiating (31) with respect to �, while holding � and its corresponding trajectory state fixed, with the conditioning variables $\hat { F } _ { I }$ and $\boldsymbol { { \widehat { v } } } _ { T }$ also held fixed, relates the estimated average velocity to the instantaneous velocity:

$$
\begin{array} { r } { V _ { \omega } \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big ) = u _ { \omega } + ( t - r ) \frac { d u _ { \omega } } { d t } } \end{array}\tag{32}
$$

The derivative is the total derivative along the time-dependent state $X _ { t }$ , with $\hat { F } _ { I }$ and $\boldsymbol { { \widehat { v } } } _ { T }$ treated as fixed conditions. This relationship enables the average-velocity network to be trained using the ground-truth instantaneous velocity from (30):

$$
\mathcal { L } _ { P M F } = \mathbb { E } _ { X , \epsilon , \hat { F } _ { I } , \hat { v } _ { T } , r , t } \left[ \left| V _ { \omega } \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big ) - v _ { \mathrm { t a r g e t } } \right| _ { 2 } ^ { 2 } \right]\tag{33}
$$

Pixel MeanFlow parameterizes the average velocity through an image-valued network output $\tilde { X } _ { \omega } \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big )$ , which jointly depends on the noisy image state, time interval, recovered visual feature, and text embedding. The corresponding targetstate prediction is

$$
\begin{array} { r } { u _ { \omega } \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big ) = \frac { X _ { t } - \tilde { X } _ { \omega } \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big ) } { t } } \end{array}\tag{34a}
$$

$$
\widehat { X } _ { r } = X _ { t } - ( t - r ) u _ { \omega } = \frac { r } { t } X _ { t } + \left( 1 - \frac { r } { t } \right) \widetilde { X } _ { \omega }\tag{34b}
$$

For one-step reconstruction, setting $t = 1 , r = 0$ , and $X _ { 1 } =$ � gives

$$
\hat { X } _ { 0 } = \tilde { X } _ { \omega } \big ( \epsilon , 0 , 1 ; \hat { F } _ { I } , \hat { v } _ { T } \big )\tag{35}
$$

As illustrated in Fig. 3, the Pixel MeanFlow network incorporates the recovered intermediate feature $\hat { F } _ { I }$ and text embedding $\widehat { v } _ { T }$ as complementary visual and semantic conditions. The noisy image, intermediate feature, text embedding, and time information are projected into corresponding tokens and jointly processed by Transformer blocks, enabling the image tokens to attend to both the visual structure provided by ${ \bf { \bar { \it F } } } _ { I }$ and the semantic information conveyed by $\boldsymbol { { \widehat { v } } } _ { T }$ . Additionally, spatially aligned intermediate features derived from $\hat { F } _ { I }$ are added to the corresponding image tokens to preserve spatial correspondence. The resulting image tokens are projected and rearranged to form the conditional image-valued output $\tilde { X } _ { \omega } \big ( X _ { t } , r , t ; \hat { F } _ { I } , \hat { v } _ { T } \big )$

To further improve reconstruction fidelity and semantic consistency, L1 and LPIPS losses are employed to constrain pixel-level and perceptual discrepancies, while a CLIP-based loss encourages agreement between the reconstructed image and its textual description. The overall training objective is

$$
\mathcal { L } _ { \mathrm { t r a i n } } = \mathcal { L } _ { \sf P M F } + \lambda _ { L P I P S } \mathcal { L } _ { \mathrm { L P I P S } } + \lambda _ { L 1 } \mathcal { L } _ { L 1 } + \lambda _ { C L I P } \mathcal { L } _ { \mathrm { C L I P } }\tag{36}
$$

The velocity derivative and auxiliary losses are evaluated only during training. At inference, the recovered intermediate feature $\hat { F } _ { I }$ and text embedding $\widehat { v } _ { T }$ guide the trained Pixel MeanFlow network to reconstruct the final RGB image directly from Gaussian noise in a single forward pass, with the visual and semantic conditions jointly determining the reconstructed image content.

## IV. EXPERIMENTAL RESULTS AND ANALYSIS

## A. Experimental Settings

In our work, a subset of 50,000 images is selected from the COCO 2017 dataset [36], with 40,000 images used for training, 5,000 for validation, and 5,000 for testing. Each image is associated with five manually annotated natural-language descriptions, and the corresponding official captions are used as text descriptions throughout the training, validation, and testing stages. In the experiments, one official caption is selected for each image as its text condition, and all images are uniformly preprocessed into RGB images of size $1 2 \bar { 8 } \times 1 2 8$ . For the underwater channel, we use the NOF1 measured channel records from the Watermark V1 dataset [37]. The Watermark dataset contains underwater acoustic channel measurements collected in various representative marine environments and reflects channel variations under different propagation distances, water depths, frequency bands, and multipath conditions. The NOF1 record selected in this work was collected in a shallow fjord environment with relatively fixed transmitter and receiving-hydrophone positions. Its sounding signal has a duration of 32.9 s, an operating frequency band of 10–18 kHz, a propagation distance of approximately $7 5 0 \mathrm { m } .$ , and a water depth of approximately 10 m. The record was repeatedly collected over an observation period of about 33 min and can describe the time-varying characteristics of the underwater acoustic channel.

The system consists of a semantic encoder, an Inter Predictor, a side-information compression module, and a Pixel MeanFlow image generator. The semantic encoder combines a CNN and a Swin Transformer and outputs a bottleneck feature of size 16 × 8 × 8 and an intermediate feature of size $9 6 \times 3 2 \times 3 2$ . The RVQ encoder maps the intermediate feature to a latent space of size $6 4 \times 1 6 \times 1 6 .$ , after which multistage residual quantization produces discrete codebook indices. Tucker and CP decompositions are also considered as the alternative trainingfree low-rank visual side-information representations. Both the Inter Predictor and Pixel MeanFlow employ conditional DiT architectures with a hidden dimension of 384 and eight Transformer blocks. The principal network architectures and experimental parameters are summarized in Table I. All experiments were conducted on a server equipped with an NVIDIA GeForce RTX 3090 GPU and an Intel Xeon Gold 6240 CPU. The time of the inference for a single image, (i.e., including the end-to-end encoding, transmission, reception and decoding, but excluding the period of the underwater propagation and hardware-based channel coding), varies from 40-100 ms, which is practical for real-world deployment.

Table I. Main network architectures and experimental settings
<table><tr><td>Module</td><td>Configuration</td></tr><tr><td>Semantic encoder</td><td>CNN-Swin; four stages with channels [96, 192, 256, 192]; depths [2, 2, 6, 2];</td></tr><tr><td>bottleneck feature intermediate feature</td><td>window size 4  $1 6 \times 8 \times 8$   $9 6 \times 3 2 \times 3 2$ </td></tr><tr><td>Inter Predictor</td><td>Conditional DiT; hidden dimension 384; eight blocks  $; 2 \times 2$  intermediate</td></tr><tr><td></td><td>patch</td></tr><tr><td>RVQ latent vectors</td><td> $6 4 \times 1 6 \times 1 6$ </td></tr><tr><td>RVQ codebook size</td><td>64, 128, 256, 512</td></tr><tr><td>RVQ depth Tucker ranks</td><td>1,2,3</td></tr><tr><td>CP ranks</td><td>(24, 10, 10), (24, 16, 16)</td></tr><tr><td></td><td> $\mathtt { R } = 3 3 , \mathtt { R } = 5 9$ </td></tr><tr><td>Pixel MeanFlow generator</td><td>Conditional DiT; hidden dimension 384; eight blocks; 8 × 8 RGB patch</td></tr></table>

## B. Effects of RVQ Depth and Codebook Size

To investigate the effects of RVQ quantization depth and codebook size on reconstruction performance, we vary these two parameters while maintaining a fixed channel SNR of 10 dB, and evaluate the resulting image reconstruction quality.

![](images/f7289955f105796c650dadede1649a92c5fafa7dc29a3ec5907ecf3eaad1095b.jpg)  
(a)

![](images/29331d6671d975fee8140cc75b4bea179c1f95639cd1e7fef6c6b54fcdce9db3.jpg)  
(b)  
Fig. 4. Reconstruction performance for different RVQ codebook sizes and quantization depths at 10 dB channel SNR.

The channel coding rates for the bottleneck and text streams are both fixed at 1/3, whereas all RVQ stages use a code rate of 1/2, ensuring a consistent channel coding rate across different

RVQ configurations. Since the bottleneck and text representations contribute to multiple stages of the reconstruction process, they carry more critical information and are therefore assigned lower code rates to provide stronger error protection. As shown in Fig. 4, increasing the RVQ quantization depth leads to consistent improvements in both PSNR and SSIM, indicating that subsequent residual quantization stages capture complementary visual information that remains inadequately represented by preceding stages. As shown in Fig. 4(a), at an SNR of 10 dB and a codebook size of 256, increasing the RVQ depth from one to two and three stages raises the PSNR from 26.7 to 27.4 and 27.9 dB, respectively. However, these improvements come at the cost of increased transmission overhead, as additional quantization stages require transmitting more codebook indices. Similarly, larger codebooks size generally require more bits to represent each index. Fig. 4(a) also shows that, for three-stage RVQ at an SNR of 10 dB, increasing the codebook size from 64 to 128, 256, and 512 yields PSNR values of 27.4, 27.8, 27.9, and 28.2 dB, respectively. These results highlight the trade-off between reconstruction quality and transmission efficiency.

## C. Comparison of Tucker and CP Tensor-Decomposition Representations

Tucker and CP tensor decompositions (TD) are leveraged as the alternative training-free visual side-information representations for comparison. As shown in Fig. 5, the higher channel SNR, the better reconstruction performance for both types of low-rank representation, indicating that more reliable channel transmission enables an increased TD representation capacity to provide the receiver with more intact supplementary visual information. As shown in Fig. 5(a), when the channel SNR increases from 5 to 10 dB, the PSNR of Tucker decomposition with ranks (24, 16, 16) increases from 25.9 to 27.4 dB, while that of CP decomposition with rank 59 increases from 25.8 to 28.4 dB. In addition, for both Tucker and CP TD, higher-rank ones also promote the reconstruction performance due to the preservation of the original intermediate feature information, yet indicating higher demands on transmission overhead. The rank comparison in Fig. 5(a) shows that, at an SNR of 10 dB, increasing the Tucker ranks from (24, 10, 10) to (24, 16, 16) improves the PSNR from 26.4 to 27.4 dB. Similarly, increasing the CP rank from 33 to 59 improves the PSNR from 26.9 to 28.4 dB.

From the perspective of representation volume, the visual side information decomposition for Tucker and CP directly consists of a core tensor, factor matrices, or factor vectors, which is substantially larger than those of RVQ, i.e., only the transmission of several discrete indices are needed for the latter. For a fair comparison, Tucker and CP TD take advantages in eliminating model training and reducing dependence on the training distribution, whereas their relatively large representation volume limits the application in resourceconstrained underwater acoustic links.

## D. Ablation Study

To evaluate the contributions of different information branches and the effectiveness of unequal error protection in cascaded reconstruction, we consider five transmission strategies: “Bneck+Text”, “Text+RVQ”, “Bneck+RVQ”, “Bneck+Text+RVQ”, and “Bneck+Text+UEP\_RVQ”. Here, “Bneck” denotes the bottleneck feature, “Text” denotes the textual stream, and “RVQ” represents the transmission of indices generated by Residual Vector Quantization. “UEP\_RVQ” further indicates that the RVQ indices at different quantization stages are assigned unequal channel protection levels. The first three configurations serve as ablation studies of the information branches: “Bneck+Text” excludes the RVQbased visual side information, “Text+RVQ” removes the bottleneck branch, and “Bneck+RVQ” removes the text branch. The last two configurations retain all three information branches but differ in their channel protection strategies

![](images/185f24a5535fbf12b5793b76d969c691d560c624e4af0a2930411caacffe03d4.jpg)  
(a)

![](images/36b1dd068618cd2b784c3b33504b59391931732c0cef28aedc95eade8062a0a5.jpg)  
(b)

![](images/8efbeeb65aa9eccc6bb1fbeee47830125692111bf94ab5e3c03a82ba8235a4fa.jpg)  
(c)

![](images/6e36575bd41817e5569d11fc5387041133bfdeca3d84d9a957e7f895b6913d43.jpg)  
(d)

Fig. 5. Image reconstruction performance of Tucker and CP low-rank visual side-information representations at different SNR.  
![](images/a3342a9fa586f81a31022d21d24ce50f8500c1e9069f3e71943c3dd9508e03ca.jpg)  
(a)

![](images/d1693110746bcd8662cc4ca4fa9dae37205cb4e8f7b878eedbeb047fe4966ed2.jpg)  
(b)

![](images/3b0a1a939bba07e20a826cf8e2b173b322c55791d8540a0e79752bf820fe611a.jpg)  
(c)

![](images/9c077e596677c46b68899f855d8ed5dfcbb5917ddcb0624bffa10bab82025de3.jpg)  
(d)  
Fig. 6. Ablation results for different information branches at various SNR.

Regarding channel coding, the “Bneck” and “Text” streams are consistently assigned a code rate of 1/3 whenever they are included. For conventional RVQ transmission, all three stages use an equal code rate of 1/2. In contrast, “UEP\_RVQ” adopts stage-dependent code rates of 1/3, 4/7, and 4/5 for the first, second, and third quantization stages, respectively. Since a lower code rate introduces more redundancy and provides stronger error protection, the first RVQ stage receives the highest protection level, followed by the second and third stages. The experimental results are presented in Fig. 6.

Compared with the “Bneck+Text” strategy, incorporating RVQ (“Bneck+Text+RVQ”) improves reconstruction quality, particularly in the higher-SNR region. As shown in Fig. 6(a), at an SNR of 10 dB, “Bneck+Text” achieves a PSNR of 25.8 dB, whereas “Bneck+Text+RVQ” achieves 28.1 dB, corresponding to a gain of 2.3 dB from the additional RVQ side information. This demonstrates that the transmitted visual side information provides complementary details that cannot be fully recovered from the compact cross-modal conditions alone. The qualitative results in Fig. 7 further support this observation. At 10 dB, “Bneck+Text” successfully recovers the overall layout of the surfing scene, but local details of the person, surfboard, and waves remain relatively blurred. With RVQ, these details become clearer, particularly in terms of object boundaries and local textures.

Removing the bottleneck branch (“Text+RVQ”) results in a noticeable degradation in reconstruction performance, especially at lower SNRs. The results in Fig. 6(a) show that, at an SNR of 5 dB, removing the bottleneck branch reduces the PSNR from 26.3 dB for “Bneck+Text+RVQ” to 17.7 dB for “Text+RVQ”, representing a decrease of 8.6 dB. This highlights the importance of bottleneck features in providing fundamental visual and structural information for image reconstruction, which cannot be adequately replaced by textual guidance and RVQ side information alone.

Finally, comparing “Bneck+RVQ” with “Bneck+Text+RVQ” reveals that their PSNR and SSIM values are comparable in the higher-SNR region. However, the inclusion of the textual branch helps maintain higher cross-modal semantic similarity in the lower-SNR region. As shown in Fig. 6(d), at an SNR of 2.5 dB, adding the text branch increases CLIP similarity from 0.214 for “Bneck+RVQ” to 0.235 for “Bneck+Text+RVQ”. This indicates that, beyond contributing to visual reconstruction, textual information provides explicit semantic guidance that helps preserve consistency between the reconstructed image and its corresponding description.

Overall, the ablation results demonstrate that the Bneck, Text, and RVQ branches provide complementary contributions to reconstruction, particularly under low-SNR conditions. As the channel quality improves, the marginal benefits of incorporating additional information become less pronounced, although their effects may vary across evaluation metrics. A similar SNR-dependent trend is observed for layered UEP: its advantage is more evident at low SNR but diminishes as SNR increases, with equal protection even yielding slightly better results for some metrics. Fig. 6(a) illustrates how the relative performance of the two protection strategies changes with SNR. At 2.5 dB, “Bneck+Text+UEP\_RVQ” achieves a PSNR of 22.8 dB, compared with 20.8 dB for “Bneck+Text+RVQ”. At 10 dB, the corresponding values are 27.8 and 28.1 dB, respectively, indicating that the low-SNR advantage of unequal protection disappears at the higher SNR. These observations motivate a further investigation of channel coding rate allocation across the five information streams, namely, Bneck, Text, and the RVQ indices from the three quantization stages, as presented in the next subsection.

Original  
Bneck+Text  
Text+RVQ  
Bneck+RVQ  
![](images/6ce4581c8090c17e2d4c3be86c0378f2cef441b1233d5f16a599d4ed0fb69cdd.jpg)  
Bneck+Text +RVQ  
Bneck+Text +UEP\_RVQ

Fig. 7. Example image reconstructions under different conditions at SNR = 10 dB.  
![](images/c24ecb87f7b4818d207040c1780b3f9f83703af9d3bd556e5eebcb1354e70ed7.jpg)  
(a)

![](images/3c1dc8d2ec8d0257da6d7c5d01ee2695369aa2c5b8ad15d65b02db71e1d58a17.jpg)  
(b)

![](images/1ac9795db4aedf8d062f9109ec6a4f084fc5b7701932a3e21daafd0395013253.jpg)  
(c)

![](images/4e0c25549512a1e4ee4b090edb4afe058119d3b2871fadd14ad3193406b11101.jpg)  
(d)  
Fig. 8. Comparison of reconstruction performance under different code-rate allocation configurations.

## E. Effects of Channel Coding Rate Allocation

To further investigate the impact of unequal channel protection resource allocation on reconstruction performance, we consider five deliberately designed channel coding rate configurations, as summarized in Table II, while retaining all three information branches: “Bneck”, “Text”, and the threestage RVQ. To ensure a fair comparison, the overall transmission budget allocated to Bneck and the three RVQ stages is kept constant across all five configurations, while the transmission overhead of the “Text” branch is excluded from this budget. The “Text” branch adopts the same channel coding rate as “Bneck” in each configuration. Accordingly, the five configurations explore different protection resource allocation strategies under a comparable transmission budget. Specifically, Configuration A applies equal protection across the three RVQ stages, whereas Configuration B introduces progressively weaker protection for later stages. Configurations C and D further vary the protection allocated to “Bneck” relative to the RVQ stages, while Configuration E assigns an identical code rate to all four budgeted streams.

Table II. Channel coding rate allocation configurations
<table><tr><td>Config.</td><td>Bneck / Text</td><td>RVQ-1</td><td>RVQ-2</td><td>RVQ-3</td></tr><tr><td>A</td><td>0.333</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>B</td><td>0.333</td><td>0.333</td><td>0.571</td><td>0.8</td></tr><tr><td>C</td><td>0.307</td><td>0.5</td><td>0.6</td><td>0.75</td></tr><tr><td>D</td><td>0.286</td><td>0.667</td><td>0.8</td><td>0.8</td></tr><tr><td>E</td><td>0.389</td><td>0.389</td><td>0.389</td><td>0.389</td></tr></table>

The results in Fig. 8 reveal a tradeoff in channel protection resource allocation between the foundational information streams (“Bneck” and Text) and the RVQ-based visual enhancement stream. The former provides essential structural and semantic conditions for reconstruction, whereas the latter supplies complementary visual details.

In the low-SNR region, stronger protection of the "Bneck” and Text branches leads to better reconstruction performance. As the allocation shifts from Configuration B to Configuration D, more protection resources are assigned to these foundational branches. Specifically, as the channel coding rate of Bneck and Text decreases from 0.333 (config B) to 0.286(config D), the corresponding PSNR values at 2.5 dB SNR increase in ascending order: 22.8, 25.1, and 25.8, as shown in Fig. 8(a). This indicates that under severe channel conditions, maintaining the reliability of the fundamental structural and semantic information is particularly important for reconstruction.

However, this trend reverses as the channel condition improves. At 5 dB SNR, the corresponding PSNR values from Configuration B to Configuration D exhibit a descending trend, namely, 26.9, 26.3, and 25.9. This suggests that further strengthening the protection of “Bneck” and Text does not necessarily yield better reconstruction performance. Instead, allocating more protection resources to the RVQ branches becomes increasingly beneficial, as improved channel conditions allow the receiver to make more effective use of the supplementary visual information. These results demonstrate that channel protection resource allocation should account for the prevailing channel conditions. Rather than consistently favoring either the foundational information or the RVQ-based visual enhancement, an appropriate balance between their protection levels is essential for achieving robust reconstruction across different SNR regimes.

Furthermore, Configurations A and E are included to examine two uniform protection schemes: Configuration A assigns an identical code rate to all RVQ stages, whereas

Configuration E applies a uniform code rate across all information streams. As shown in Fig. 8, the relative performance of these configurations varies with channel SNR, and neither consistently outperforms the other allocation strategies across all channel conditions and evaluation metrics. This further suggests that a fixed channel coding rate allocation may not be optimal under varying channel conditions. Instead, channel-state feedback could be exploited to guide the adaptive allocation of channel coding resources, motivating future research into channel-aware protection strategy learning.

## V. CONCLUSION

This paper proposes a generative image communication framework based on cross-modal semantic successive refinement for underwater acoustic channels. At the receiver, the received text description and bottleneck features are first fed into the Inter Predictor to generate a base estimate of the intermediate feature. The received RVQ indices are then decoded to recover visual side information, which is used by the residual refinement branch to further refine the base estimate. Finally, a one-step Pixel MeanFlow model reconstructs the RGB image from the refined intermediate feature conditioned on the text embedding. In addition, a CLIPbased text–image semantic consistency constraint is incorporated during training to encourage semantic consistency between the reconstructed image and its corresponding text description.

Experimental results show that removing the bottleneck branch leads to a substantial degradation in reconstruction quality, particularly under low-SNR conditions, while incorporating the text branch improves the semantic similarity between reconstructed images and their corresponding descriptions. The additional RVQ-based visual side information further improves reconstruction quality, especially at higher SNRs. Increasing the RVQ quantization depth and codebook size also generally improves reconstruction performance, although at the cost of increased transmission overhead. Furthermore, the channel protection experiments show that stronger protection of the bottleneck and text branches is more beneficial under low-SNR conditions. As the channel SNR increases, allocating relatively more protection resources to the RVQ stages becomes increasingly beneficial, rather than continuously strengthening the protection of the bottleneck and text branches. The results also show that unequal protection across RVQ stages improves reconstruction performance under low-SNR conditions, while its advantage diminishes at higher SNRs. These findings demonstrate the importance of adapting channel coding rate allocation to the channel conditions. Future work will investigate channel-aware adaptive protection strategies that dynamically adjust the protection levels of the bottleneck, text, and RVQ branches according to channel variations, aiming to further improve reconstruction performance under diverse underwater acoustic conditions.

## REFERENCES

[1] C. E. Shannon, “A Mathematical Theory of Communication”.

[2] W. Weaver, “Recent Contributions to The Mathematical Theory of Communication”.

[3] R. Carnap and Y. Bar-Hillel, “AN OUTLINE OF A THEORY OF SEMANTIC INFORMATION”.

[4] J. Bao et al., “Towards a theory of semantic communication,” in 2011 IEEE Network Science Workshop, Jun. 2011, pp. 110–117. doi: 10.1109/NSW.2011.6004632.

[5] Q. Lan et al., “What is Semantic Communication? A View on Conveying Meaning in the Era of Machine Intelligence,” Journal of Communications and Information Networks, vol. 6, no. 4, pp. 336–371, Dec. 2021, doi: 10.23919/JCIN.2021.9663101.

[6] E. C. Strinati and S. Barbarossa, “6G Networks: Beyond Shannon Towards Semantic and Goal-Oriented Communications,” Feb. 17, 2021, arXiv: arXiv:2011.14844. doi: 10.48550/arXiv.2011.14844.

[7] D. Gunduz et al., “Beyond Transmitting Bits: Context, Semantics, and Task-Oriented Communications,” Oct. 03, 2022, arXiv: arXiv:2207.09353. doi: 10.48550/arXiv.2207.09353.

[8] Z. Qin, X. Tao, J. Lu, W. Tong, and G. Y. Li, “Semantic Communications: Principles and Challenges,” Jun. 27, 2022, arXiv: arXiv:2201.01389. doi: 10.48550/arXiv.2201.01389.

[9] W. Yang et al., “Semantic Communications for Future Internet: Fundamentals, Applications, and Challenges,” IEEE Commun. Surv. Tutorials, vol. 25, no. 1, pp. 213–250, 2023, doi: 10.1109/COMST.2022.3223224.

[10] Y. Liu, X. Wang, Z. Ning, M. Zhou, L. Guo, and B. Jedari, “A survey on semantic communications: Technologies, solutions, applications and challenges,” Digital Communications and Networks, vol. 10, no. 3, pp. 528–545, Jun. 2024, doi: 10.1016/j.dcan.2023.05.010.

[11] M. Kountouris and N. Pappas, “Semantics-Empowered Communication for Networked Intelligent Systems,” Mar. 10, 2021, arXiv: arXiv:2007.11579. doi: 10.48550/arXiv.2007.11579.

[12] E. Uysal et al., “Semantic Communications in Networked Systems: A Data Significance Perspective,” IEEE Network, vol. 36, no. 4, pp. 233– 240, Jul. 2022, doi: 10.1109/MNET.106.2100636.

[13] Z. Li, M. Chitre, and M. Stojanovic, “Underwater acoustic communications,” Nat Rev Electr Eng, vol. 2, no. 2, pp. 83–95, Dec. 2024, doi: 10.1038/s44287-024-00122-w.

[14] E. Bourtsoulatze, D. Burth Kurka, and D. Gündüz, “Deep Joint Source-Channel Coding for Wireless Image Transmission,” IEEE Transactions on Cognitive Communications and Networking, vol. 5, no. 3, pp. 567– 579, Sep. 2019, doi: 10.1109/TCCN.2019.2919300.

[15] D. B. Kurka and D. Gündüz, “DeepJSCC-f: Deep Joint Source-Channel Coding of Images With Feedback,” IEEE Journal on Selected Areas in Information Theory, vol. 1, no. 1, pp. 178–193, May 2020, doi: 10.1109/JSAIT.2020.2987203.

[16] M. Yang, C. Bian, and H.-S. Kim, “Deep Joint Source Channel Coding for Wireless Image Transmission with OFDM,” in ICC 2021 - IEEE International Conference on Communications, Jun. 2021, pp. 1–6. doi: 10.1109/ICC42927.2021.9500996.

[17] J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, and M. Rodrigues, “Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 4, pp. 2315–2328, Apr. 2022, doi: 10.1109/TCSVT.2021.3082521.

[18] D. B. Kurka and D. Gündüz, “Bandwidth-Agile Image Transmission With Deep Joint Source-Channel Coding,” IEEE Transactions on Wireless Communications, vol. 20, no. 12, pp. 8081–8095, Dec. 2021, doi: 10.1109/TWC.2021.3090048.

[19] J. Dai et al., “Nonlinear Transform Source-Channel Coding for Semantic Communications,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 8, pp. 2300–2316, Aug. 2022, doi: 10.1109/JSAC.2022.3180802.

[20] K. Yang, S. Wang, J. Dai, X. Qin, K. Niu, and P. Zhang, “SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding,” Jul. 05, 2024, arXiv: arXiv:2308.09361. doi: 10.48550/arXiv.2308.09361.

[21] T.-Y. Tung, D. B. Kurka, M. Jankowski, and D. Gündüz, “DeepJSCC-Q: Constellation Constrained Deep Joint Source-Channel Coding,” IEEE Journal on Selected Areas in Information Theory, vol. 3, no. 4, pp. 720–731, Dec. 2022, doi: 10.1109/JSAIT.2022.3231042.

[22] Q. Hu, G. Zhang, Z. Qin, Y. Cai, G. Yu, and G. Y. Li, “Robust Semantic Communications With Masked VQ-VAE Enabled Codebook,” IEEE Transactions on Wireless Communications, vol. 22, no. 12, pp. 8707– 8722, Dec. 2023, doi: 10.1109/TWC.2023.3265201.

[23] E. Erdemir, T.-Y. Tung, P. L. Dragotti, and D. Gündüz, “Generative Joint Source-Channel Coding for Semantic Image Transmission,” IEEE

Journal on Selected Areas in Communications, vol. 41, no. 8, pp. 2645– 2657, Aug. 2023, doi: 10.1109/JSAC.2023.3288243.

[24] T. Wu et al., “CDDM: Channel Denoising Diffusion Models for Wireless Communications,” May 16, 2023, arXiv: arXiv:2305.09161. doi: 10.48550/arXiv.2305.09161.

[25] L. Qiao, M. B. Mashhadi, Z. Gao, C. H. Foh, P. Xiao, and M. Bennis, “Latency-Aware Generative Semantic Communications with Pre-Trained Diffusion Models,” Jul. 13, 2024, arXiv: arXiv:2403.17256. doi: 10.48550/arXiv.2403.17256.

[26] M. Zhang, H. Wu, G. Zhu, R. Jin, X. Chen, and D. Gündüz, “Semantics-Guided Diffusion for Deep Joint Source-Channel Coding in Wireless Image Transmission,” Jul. 21, 2025, arXiv: arXiv:2501.01138. doi: 10.48550/arXiv.2501.01138.

[27] A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” in Proceedings of the 38th International Conference on Machine Learning, PMLR, Jul. 2021, pp. 8748–8763. Accessed: Oct. 08, 2026. [Online]. Available: https://proceedings.mlr.press/v139/radford21a.html

[28] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-Resolution Image Synthesis with Latent Diffusion Models,” Apr. 13, 2022, arXiv: arXiv:2112.10752. doi: 10.48550/arXiv.2112.10752.

[29] Y. Peng, J. Liu, Y. Zhu, R. Zhang, and F. Yuan, “Efficient Semantic Communication for Underwater Images Guided by Physical Priors,” IEEE Signal Process. Lett., vol. 32, pp. 3072–3076, 2025, doi: 10.1109/LSP.2025.3592591.

[30] W. Chen et al., “Semantic Communication Based on Large Language Model for Underwater Image Transmission,” IEEE Transactions on Mobile Computing, vol. 25, no. 2, pp. 2060–2075, Feb. 2026, doi: 10.1109/TMC.2025.3607717.

[31] J. Liang and L. Ma, “Adaptive Semantic Communication for Underwater Images Based on Residual Vector Quantization,” in OCEANS 2026 Sanya, May 2026, pp. 1–7. doi: 10.1109/OCEANS66983.2026.11617191.

[32] Y. Lu et al., “One-step Latent-free Image Generation with Pixel Mean Flows,” Jan. 29, 2026, arXiv: arXiv:2601.22158. doi: 10.48550/arXiv.2601.22158.

[33] Z. Liu et al., “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows,” Aug. 17, 2021, arXiv: arXiv:2103.14030. doi: 10.48550/arXiv.2103.14030.

[34] D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han, “Autoregressive Image Generation using Residual Quantization,” Mar. 09, 2022, arXiv: arXiv:2203.01941. doi: 10.48550/arXiv.2203.01941.

[35] T. G. Kolda and B. W. Bader, “Tensor Decompositions and Applications,” SIAM Rev., vol. 51, no. 3, pp. 455–500, Aug. 2009, doi: 10.1137/07070111X.

[36] T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” Feb. 21, 2015, arXiv: arXiv:1405.0312. doi: 10.48550/arXiv.1405.0312.

[37] P. A. Van Walree, F.-X. Socheleau, R. Otnes, and T. Jenserud, “The Watermark Benchmark for Underwater Acoustic Modulation Schemes,” IEEE J. Oceanic Eng., vol. 42, no. 4, pp. 1007–1018, Oct. 2017, doi: 10.1109/JOE.2017.2699078.