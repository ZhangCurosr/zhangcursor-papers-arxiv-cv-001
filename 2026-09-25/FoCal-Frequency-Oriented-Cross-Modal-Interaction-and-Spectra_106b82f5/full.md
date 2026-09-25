# FoCal: Frequency-Oriented Cross-Modal Interaction and Spectral Calibration for Aerial Visible-Infrared Object Detection

Ben Liang, Graduate Student Member, IEEE, Chao Sui, Junqi Bai, Yuan Liu, Member, IEEE, Chunlai Li, Xiubao Sui, and Qian Chen

Abstract—In aerial RGB–IR object detection, effectively exploiting complementary information across modalities is critical for robust perception under complex illumination and environmental conditions. Existing multimodal detectors mainly focus on spatial-domain interaction or frequency-specific feature enhancement, while the cross-modal interaction patterns of different frequency components remain insufficiently explored. Moreover, spectral discrepancy itself may contain both useful complementary cues and unreliable modality-specific responses, making indiscriminate frequency fusion suboptimal. To address these issues, we propose FoCal, a frequency-oriented framework for aerial RGB–IR object detection. First, a Frequency-Aware Dual-Domain Calibration (FADC) module is developed to explicitly model frequency-dependent cross-modal interaction. Lowfrequency components are collaboratively consolidated into a shared structural consensus, whereas high-frequency components preserve modality-specific information through selective crossmodal exchange. The resulting frequency-aware cues are further transferred to the original feature domain to regulate cross-modal calibration. Second, we introduce a Discrepancy-Guided Spectral Modulation (DGSM) module, which characterizes cross-modal spectral imbalance using confidence-weighted relative amplitude discrepancy and transforms it into a bounded signed gate for adaptive enhancement, preservation, or attenuation of the joint multimodal spectrum. Extensive experiments on DroneVehicle, ESCVehicle, and ATR-UMOD demonstrate the effectiveness of FoCal, yielding mAP<sub>50</sub> values of 83.5%, 54.8%, and 64.6%, respectively. Meanwhile, with only 3.0M parameters, FoCal achieves 113.6 FPS while preserving leading detection accuracy, highlighting a favorable accuracy–efficiency trade-off. Code is available at https://github.com/universeliang/FoCal.

Index Terms—object detection, multimodal, cross-modal interaction, frequency fusion, modality-specific.

![](images/25fa4f057ac3a4746d986ed5506bb55fa266c815842746973fa793a6651110ea.jpg)  
Fig. 1. Comparison of cross-modal fusion paradigms. (a) Spatial-domain fusion directly combines RGB and IR features. (b) FreFusion combines same-frequency components before applying band-specific fusion modules. (c) FoCal differentiates the interaction itself: low-frequency components form a shared structural consensus, whereas high-frequency components undergo selective exchange while preserving modality-specific details.

## I. INTRODUCTION

In recent years, single-modality object detection based on visible (RGB) imagery has achieved remarkable progress and has been widely applied in various scenarios, including intelligent transportation [1], [2], disaster rescue [3], [4], and remote sensing [5]–[7]. Nevertheless, RGB imaging is highly dependent on external illumination, and its visual quality can deteriorate substantially under low illumination, strong light interference, and adverse weather conditions, leading to severe detail loss and contrast degradation [8], [9]. In contrast, infrared (IR) sensors capture the thermal radiation emitted by objects and are therefore less sensitive to illumination variations, while also providing complementary object contours and structural cues under challenging environments [10], [11]. These complementary imaging characteristics make RGB–IR multimodal fusion an effective paradigm for robust all-weather visual perception [12]–[14].

Multimodal fusion detection (MFD) aims to exploit both shared and modality-specific information from heterogeneous sensors to construct robust representations for object detection [15]. Most existing approaches perform cross-modal interaction primarily in the spatial domain, where semantic alignment, feature enhancement, or attention-based aggregation is employed to improve multimodal complementarity [14], [16]. However, spatial-domain interaction alone does not explicitly distinguish structural information from fine-grained modalitydependent responses. Recent studies have therefore introduced frequency-domain decomposition to separate low-frequency structures from high-frequency details and assign different enhancement operators to individual frequency bands [17], [18]. Despite their effectiveness, these approaches mainly focus on how different frequency components are processed after decomposition, while a more fundamental question remains underexplored: should cross-modal components at different frequencies follow the same interaction pattern in the first place?

![](images/27dcb384dc623390b6082dd6367bbf5bd6bdb95690cbba6537950489fb715945.jpg)  
Fig. 2. Comparison of model accuracy and computational efficiency. Petal lengths are linearly scaled to the original values within each metric, with the maximum value assigned the longest petal. FoCal achieves the highest m $\mathrm { A P _ { 5 0 } }$ and FPS while requiring the fewest parameters and FLOPs.

As shown in Fig. 1, we argue that frequency characteristics should determine not only the subsequent processing operators, but also the cross-modal information-flow topology. Lowfrequency components mainly encode coarse contours, object layouts, and large-scale structures, which generally exhibit stronger correspondence between RGB and IR modalities. They are therefore more suitable for consensus formation, where both modalities collaboratively establish a shared structural representation. High-frequency components, in contrast, contain fine edges and local textures together with modalitydependent responses and sensor-specific noise. Prematurely collapsing them into a single representation may suppress useful modality-specific details or propagate unreliable responses across modalities. From this perspective, low- and high-frequency components require fundamentally different cross-modal interaction patterns: a many-to-one consensus for low-frequency structures and an individuality-preserving twoto-two exchange for high-frequency details. More importantly, frequency-domain interaction should not be regarded as the endpoint of multimodal fusion. We consider the frequency domain as an auxiliary space for estimating the reliability of cross-modal interaction. The resulting frequency-conditioned cues are transferred back to the original feature domain to regulate information injection between RGB and IR streams. In this manner, frequency-specific interaction and originaldomain feature calibration are explicitly coupled while serving different purposes: the former determines how the modalities should interact, whereas the latter determines where and to what extent counterpart information should be introduced.

Based on these observations, we propose FoCal, a frequency-oriented framework for aerial RGB–IR object detection. First, we propose a Frequency-Aware Dual-Domain Calibration (FADC) module that explicitly models frequencydependent cross-modal interaction. It promotes structural consensus in low-frequency components while preserving modality individuality through selective high-frequency exchange, and further transfers the learned frequency cues to calibrate the original feature streams. Second, we propose a Discrepancy-Guided Spectral Modulation (DGSM) module that exploits confidence-weighted spectral discrepancy as a control signal to adaptively regulate the joint multimodal spectrum. It converts the learned discrepancy cues into a signed spectral gate, enabling frequency-wise enhancement, preservation, or attenuation of the joint representation. Extensive experiments are conducted on three aerial-view multimodal datasets, including DroneVehicle [8], ESCVehicle [19], and ATR-UMOD [20]. FoCal achieves $\mathrm { m A P _ { 5 0 } }$ values of 83.5%, 54.8%, and 64.6% on the three datasets, respectively, consistently delivering stateof-the-art performance across all three benchmarks. As illustrated in Fig. 2, FoCal further achieves 113.6 FPS with only 3.0M parameters and 8.4G FLOPs, highlighting its favorable accuracy–efficiency trade-off. Overall, our main contributions are summarized as follows:

(1) We propose FoCal, a lightweight frequency-oriented framework for aerial RGB–IR object detection. Rather than merely assigning different processing operators to different frequency bands, FoCal explicitly models frequencydependent cross-modal interaction and further exploits spectral discrepancy to regulate multimodal representation learning.

(2) We propose a Frequency-Aware Dual-Domain Calibration (FADC) module that differentiates the cross-modal interaction patterns of low- and high-frequency components. Lowfrequency representations are collaboratively consolidated to establish structural consensus, whereas high-frequency representations preserve modality individuality through selective cross-modal exchange. The resulting frequency-aware cues are further transferred to the original feature domain for crossmodal calibration.

(3) We propose a Discrepancy-Guided Spectral Modulation (DGSM) module that characterizes cross-modal spectral imbalance using confidence-weighted relative amplitude discrepancy. Instead of directly treating discrepancy as informative content, DGSM converts it into a bounded signed control signal to adaptively enhance, preserve, or attenuate the joint multimodal spectrum.

## II. RELATED WORK

## A. Multimodal Object Detection

Multimodal fusion aims to exploit complementary information from different modalities to produce high-quality fused representations [21], thereby improving the performance of MFD. For example, CDDFuse [22] and MambaDFuse [23] employ Transformer- and Mamba-based architectures, respectively, to enhance pixel-level fusion through global contextual modeling, and subsequently transfer the fused representations to downstream detection tasks. Owing to the complementary properties of different modalities, there has been growing interest in using RGB–IR image pairs for multispectral pedestrian and vehicle detection. UA-CMDet [8] introduces DroneVehicle, an RGB–IR vehicle detection benchmark captured from UAV viewpoints, and improves detection performance by leveraging illumination estimation to model cross-modal uncertainty. CFT [24] and YOLOFusion [25] enhance multispectral object detection by modeling global feature interactions both across and within modalities. ICAFusion [14] proposes a query-guided iterative cross-attention mechanism to capture cross-modal complementary information and improve feature discriminability. FusionMamba [26] further projects cross-modal features into a hidden state space via Mamba for enhanced representation learning, thereby suppressing pseudotarget responses. Although these methods have achieved promising performance, there remains substantial room for improvement. In particular, frequency-domain disentanglement provides a promising route to separate informative components for more refined cross-modal fusion, which is especially beneficial for small-object detection in complex aerial scenarios.

## B. Frequency in computer vision

Frequency-domain analysis has been increasingly explored in recent vision tasks due to its ability to disentangle lowand high-frequency components for more effective representation learning [27], [28]. Among representative frequency domain techniques, Fast Fourier Transform (FFT) and Discrete Wavelet Transform (DWT) exhibit complementary characteristics: FFT is particularly effective at modeling global frequency interactions, whereas DWT is better suited for local multi-scale decomposition. DFFormer [29] replaces conventional MHSA with an FFT-based token mixer to enhance global modeling, achieving excellent performance on high-resolution images with lower computational cost. SFDFusion [30] employs FFT to fuse the amplitude and phase of infrared and visible images, and further cascades the spectral representations with spatialdomain features, thereby effectively improving multimodal image fusion quality. WTConv [31] effectively enlarges the effective receptive field through cascaded wavelet transforms. HaarFuse [32] decomposes features into high- and lowfrequency subbands using wavelet transform, and then applies CNNs and Transformers to finely refine different components, resulting in improved fusion quality. In multimodal detection, WaveMamba [18] efficiently combines wavelet decomposition with Vision Mamba, enhancing the capability of multimodal object detection. DEPFusion [33] leverages Mamba and FFT to enhance multi-scale context and texture details in RGB lowfrequency components, thereby improving UAV multispectral object detection performance. Despite recent progress, most existing multimodal detection methods still assume equal contributions of frequency components from different modalities, while overlooking the trade-off relationships between same-frequency components across modalities. Consequently, frequency-dependent cross-modal interaction and discrepancyaware spectral modeling remain insufficiently explored for refined multimodal fusion. Moreover, these methods usually treat frequency-enhanced features as the final fusion outputs, without further exploiting them to regulate the subsequent interaction of the original feature streams.

## III. METHODS

In this section, we first present the overall architecture of the proposed FoCal network, as shown in Fig. 3. We then

elaborate on the design rationale of the FADC and DGSM modules, respectively.

## A. Frequency-Aware Dual-Domain Calibration

RGB and infrared sensors exhibit substantially different imaging characteristics, resulting in distinct cross-modal behaviors across frequency bands. Existing frequency-aware fusion methods typically distinguish low- and high-frequency components by assigning them different enhancement operators after decomposition. However, such a strategy mainly focuses on how frequency components are processed after interaction, while paying less attention to a more fundamental issue: whether different frequency bands should follow the same cross-modal interaction pattern in the first place. We argue that frequency specificity should determine not only the subsequent processing operator, but also the cross-modal informationflow topology. In particular, low-frequency components mainly encode coarse structures and semantic layouts that exhibit relatively strong cross-modal correspondence, and are therefore suitable for establishing a shared structural consensus. In contrast, high-frequency components contain modality-dependent contours, textures, and local details, together with sensorspecific noise. Prematurely collapsing them into a single fused representation may suppress useful modality-specific cues or propagate unreliable responses across modalities.

Motivated by this observation, we propose a Frequency-Aware Dual-Domain Calibration (FADC) module. FADC explicitly assigns different interaction topologies to different frequency bands: low-frequency representations are collaboratively consolidated into a shared structural consensus, whereas high-frequency representations remain modality-specific and interact through recipient-conditioned selective exchange. The resulting frequency-conditioned representations are further used to estimate cross-modal reliability and calibrate information transfer over the original feature streams.

Given the feature maps $X _ { r g b } , X _ { i r } \in \mathbb { R } ^ { C \times H \times W }$ from the RGB and IR branches, we first employ a Discrete Wavelet Transform (DWT) with the Haar basis to separate coarse structures from directional details. Specifically, each input feature is decomposed into one low-frequency approximation component and three directional high-frequency components:

$$
\{ \mathcal { A } , \mathcal { H } ^ { h } , \mathcal { H } ^ { v } , \mathcal { H } ^ { d } \} = \mathcal { W } ( X ) ,\tag{1}
$$

where $\mathcal { W } ( \cdot )$ denotes the Haar-based DWT, A represents the low-frequency approximation, and $\mathcal { H } ^ { h } , \mathcal { H } ^ { v }$ , and $\mathcal { H } ^ { d }$ denote horizontal, vertical, and diagonal high-frequency responses, respectively. For each modality, the directional high-frequency components are aggregated as

$$
\mathcal { H } _ { m } = \sum _ { k \in \{ h , v , d \} } \mathcal { H } _ { m } ^ { k } , \qquad m \in \{ r g b , i r \} .\tag{2}
$$

Accordingly, each modality is represented by a low-frequency structural component $A _ { m }$ and a high-frequency detail component $\mathcal { H } _ { m }$

![](images/b9bc937ba5bf613de291f21d8fa49331b099ae215455aa0d794d3402d0db3edd.jpg)  
Fig. 3. Overall architecture of the proposed FoCal. The upper part presents the macroscopic detection pipeline, which consists of a Backbone for feature extraction, a Path Aggregation Network (PAN) for multi-scale feature fusion, and a Detection Head.

1) Frequency-Dependent Interaction Topology: A key design principle of FADC is that low- and high-frequency components should not simply adopt different post-fusion operators; instead, they should follow different cross-modal interaction topologies according to their information characteristics.

Low-Frequency Consensus Formation. Low-frequency responses primarily describe object layouts, coarse contours, and large-scale semantic structures, which generally exhibit stronger cross-modal correspondence than fine-grained details. Therefore, instead of independently enhancing $\mathcal { A } _ { r g b }$ and $\mathcal { A } _ { i r }$ after direct aggregation, we explicitly establish a shared structural anchor before subsequent feature calibration. Specifically, the two low-frequency components are concatenated and fed into a lightweight convolutional gating function followed by Softmax normalization to obtain complementary weights $\eta _ { r g b }$ and $\eta _ { i r }$ . The shared low-frequency consensus is formulated as

$$
\mathcal { A } _ { s h a r e d } = \eta _ { r g b } \odot \mathcal { A } _ { r g b } + \eta _ { i r } \odot \mathcal { A } _ { i r } , \quad \eta _ { r g b } + \eta _ { i r } = 1 .\tag{3}
$$

This operation implements a many-to-one interaction topology,

$$
( \mathcal { A } _ { r g b } , \mathcal { A } _ { i r } )  \mathcal { A } _ { s h a r e d } ,\tag{4}
$$

where both modalities collaboratively determine a common structural representation. The adaptive weighting prevents the shared representation from degenerating into a fixed average and allows the network to dynamically adjust the contribution of each modality according to local observation quality.

High-Frequency Selective Exchange. High-frequency components exhibit a fundamentally different cross-modal relationship. They contain useful modality-specific edges and textures, but also sensor-dependent noise and local disturbances. Directly collapsing $\mathcal { H } _ { r g b }$ and $\mathcal { H } _ { i r }$ into a single high-frequency representation may therefore discard modality-specific details or allow unreliable responses from one modality to dominate the other. Instead of forming a shared high-frequency representation, FADC preserves both modality-specific streams and performs recipient-conditioned selective exchange. We first derive channel-aggregated spatial descriptors using channelwise average and max pooling. The descriptors from the two modalities are jointly encoded by a lightweight convolution-Sigmoid gating function to generate two modality-specific reception masks, $\mathcal { M } _ { r g b }$ and $\mathcal { M } _ { i r }$ . The high-frequency components are then updated as

$$
\tilde { \mathcal { H } } _ { r g b } = \mathcal { H } _ { r g b } + \mathcal { M } _ { r g b } \odot \mathcal { H } _ { i r } , \quad \tilde { \mathcal { H } } _ { i r } = \mathcal { H } _ { i r } + \mathcal { M } _ { i r } \odot \mathcal { H } _ { r g b } .\tag{5}
$$

Unlike winner-take-all selection or direct high-frequency aggregation, Eq. (5) follows a two-to-two interaction topology,

$$
( \mathcal { H } _ { r g b } , \mathcal { H } _ { i r } ) \to ( \tilde { \mathcal { H } } _ { r g b } , \tilde { \mathcal { H } } _ { i r } ) ,\tag{6}
$$

where each modality preserves its intrinsic high-frequency responses while selectively absorbing complementary details from its counterpart. Importantly, $\mathcal { M } _ { r g b }$ and $\mathcal { M } _ { i r }$ characterize the reception preference of the corresponding target modalities rather than the standalone saliency of the source modality. This recipient-conditioned interaction prevents premature modality collapse and reduces the propagation of modality-specific noise.

The low- and high-frequency representations are subsequently recomposed into two frequency-conditioned features:

$$
\tilde { X } _ { r g b } = \mathcal { A } _ { s h a r e d } + \tilde { \mathcal { H } } _ { r g b } , \quad \tilde { X } _ { i r } = \mathcal { A } _ { s h a r e d } + \tilde { \mathcal { H } } _ { i r } .\tag{7}
$$

Therefore, the two modalities share the same low-frequency structural anchor while retaining individually calibrated high frequency details.

2) Frequency-Guided Reliability Calibration: The frequency-conditioned features in Eq. (7) are not directly regarded as the final fused representation. Instead, they serve as auxiliary cues for estimating where and what information can be reliably transferred between the two original modality streams. Specifically, we derive spatial and channel reliability masks from $\tilde { X } _ { r g b }$ and $\tilde { X } _ { i r }$

$$
W _ { s } = \mathcal { S } ( \tilde { X } _ { r g b } , \tilde { X } _ { i r } ) , \quad W _ { c } = \mathcal { C } ( \tilde { X } _ { r g b } , \tilde { X } _ { i r } ) ,\tag{8}
$$

$$
W = W _ { s } \odot W _ { c } ,\tag{9}
$$

where $W ~ = ~ \{ W _ { r g b } , W _ { i r } \}$ . The spatial weighting function $\boldsymbol { \mathcal { S } } ( \cdot )$ extracts position-sensitive reliability from channel-wise average and maximum responses, whereas the channel weighting function $\mathcal { C } ( \cdot )$ models channel dependencies from pooled bimodal statistics through a shared MLP. Their combination provides modality-specific frequency-conditioned reliability estimates. Meanwhile, the original RGB and IR features are jointly encoded to obtain a bimodal contextual prior:

$$
\mathcal { O } _ { j o i n t } = \mathcal { G } ( [ X _ { r g b } , X _ { i r } ] ) ,\tag{10}
$$

where $\mathcal { G } ( \cdot )$ denotes a lightweight gating function. We then combine the joint bimodal prior with the frequencyconditioned reliability masks to obtain modality-specific calibration masks:

$$
\begin{array} { r } { \mathcal { R } _ { k } = \mathcal { O } _ { j o i n t } \odot W _ { k } , \qquad k \in \{ r g b , i r \} . } \end{array}\tag{11}
$$

Based on $\mathcal { R } _ { k }$ , complementary information is selectively injected into each original modality stream:

$$
X _ { k } ^ { a u g } = X _ { k } + \mathcal { R } _ { k } \odot X _ { \bar { k } } , \qquad k \in \{ r g b , i r \} ,\tag{12}
$$

where $\bar { k }$ denotes the counterpart modality. In this formulation, frequency-domain interaction does not directly replace the original modality representation; instead, it determines the reliability of cross-modal information transfer.

To further regulate the augmented representation, we construct a reliability-conditioned reference feature:

$$
X _ { k } ^ { r e f } = X _ { k } ^ { a u g } \odot \mathcal { R } _ { k } , \qquad \Delta X _ { k } = X _ { k } ^ { r e f } - X _ { k } ^ { a u g } ,\tag{13}
$$

and perform residual calibration as

$$
X _ { k } ^ { o u t } = \sigma ( \Delta X _ { k } ) \odot X _ { k } ^ { a u g } + X _ { k } ^ { r e f } ,\tag{14}
$$

where $\sigma ( \cdot )$ denotes the Sigmoid function. This residual calibration further adjusts the contribution of the cross-modally augmented responses according to the learned reliability reference.

Overall, FADC differs from conventional frequency-aware fusion in that frequency decomposition is not merely used to assign different enhancement modules to different frequency bands. Instead, it explicitly imposes frequency-dependent cross-modal interaction topologies: low-frequency components undergo structural consensus formation, whereas highfrequency components preserve modality individuality and perform recipient-conditioned selective exchange. The resulting frequency-conditioned representations are then converted into reliability cues to regulate cross-modal transfer over the original feature streams, thereby avoiding premature modality collapse while retaining complementary information from both modalities.

## B. Discrepancy-Guided Spectral Modulation

Although FADC regulates cross-modal interaction accord ing to frequency characteristics, the resulting modality representations may still exhibit spectral imbalance caused by heterogeneous imaging mechanisms and modality-dependent degradation. Simply aggregating the two modalities may consequently weaken complementary responses or propagate modality-specific interference. To address this issue, we introduce a Discrepancy-Guided Spectral Modulation (DGSM) module. DGSM employs cross-modal spectral discrepancy to guide the modulation of an independently constructed joint spectrum. A confidence-weighted relative discrepancy is first derived from the RGB and IR spectra and then converted into a signed gate for adaptive spectral regulation. The resulting modulation selectively enhances, preserves, or attenuates joint spectral responses according to the learned discrepancy cues, thereby improving spectral complementarity while reducing unreliable modality-specific interference.

Given the modality-specific features $ { \boldsymbol { X } } _ { r g b } ,  { \boldsymbol { X } } _ { i r }  { \mathrm { ~  ~ \varphi ~ } } \in$ $\mathbb { R } ^ { C \times H \times W }$ , we first map them into a common comparison space using a shared projection function $\mathcal { P } ( \cdot )$ . During training, the RGB and IR features are jointly normalized to reduce modality-dependent feature-scale bias before spectral comparison:

$$
X _ { r g b } ^ { p } , X _ { i r } ^ { p } = \mathcal { P } ( X _ { r g b } , X _ { i r } ) .\tag{15}
$$

The projected features are then transformed into the frequency domain using the Fast Fourier Transform (FFT):

$$
Z _ { r g b } = \mathcal { F } ( X _ { r g b } ^ { p } ) , \qquad Z _ { i r } = \mathcal { F } ( X _ { i r } ^ { p } ) ,\tag{16}
$$

where $\mathcal F ( \cdot )$ denotes the orthonormally normalized FFT. Their amplitude spectra are given by

$$
A _ { r g b } = | Z _ { r g b } | , \qquad A _ { i r } = | Z _ { i r } | .\tag{17}
$$

We therefore define a scale-normalized relative spectral discrepancy as

$$
D _ { r e l } = \frac { | A _ { r g b } - A _ { i r } | } { A _ { r g b } + A _ { i r } + \epsilon } ,\tag{18}
$$

where ϵ is a small constant for numerical stability. In this formulation, $D _ { r e l }$ characterizes the relative spectral imbalance between the two modalities rather than their absolute energy difference. However, a large relative discrepancy does not necessarily imply useful complementarity. In particular, two nearly inactive spectral responses may still yield a large $D _ { r e l }$ due to their small denominator. To avoid overemphasizing such unreliable differences, we further introduce an energy confidence term. Let

![](images/22892758102c239b1c19da441c9631ed0bfac638fcf5d7979a47606a88d17539.jpg)  
Fig. 4. Overall architecture of the proposed DGSM

$$
{ \cal A } _ { \Sigma } = { \cal A } _ { r g b } + { \cal A } _ { i r } ,\tag{19}
$$

and normalize its spectral energy as

$$
E = \frac { A _ { \Sigma } } { \mathrm { M e a n } _ { \Omega } ( A _ { \Sigma } ) + \epsilon } ,\tag{20}
$$

where $\mathrm { M e a n } _ { \Omega } ( \cdot )$ denotes averaging over the frequency plane. The corresponding confidence is defined as

$$
C _ { e } = \frac { E } { 1 + E } .\tag{21}
$$

The final discrepancy descriptor is therefore

$$
D = D _ { r e l } \odot C _ { e } .\tag{22}
$$

Consequently, relative differences occurring at nearly inactive spectral locations receive lower confidence, whereas discrepancies supported by meaningful spectral responses are retained for subsequent modulation.

Importantly, DGSM does not directly inject the discrepancy descriptor $D$ into the multimodal representation. Instead, D is transformed into a learnable signed modulation gate:

$$
G _ { d } = \operatorname { t a n h } \left( { \mathcal { G } } _ { d } ( D ) \right) , \qquad G _ { d } \in [ - 1 , 1 ] ,\tag{23}
$$

where $\mathcal { G } _ { d } ( \cdot )$ denotes a lightweight $1 \times 1$ convolution. The signed formulation allows the model to learn different responses to cross-modal spectral imbalance. Therefore, spectral discrepancy itself is not assumed to be intrinsically informative; instead, the detection objective determines how each discrepancy pattern should affect multimodal representation learning.

In parallel with the discrepancy branch, the original RGB and IR features are concatenated and projected into a compact joint representation:

$$
X _ { j o i n t } = \mathcal { P } _ { j } ( [ X _ { r g b } , X _ { i r } ] ) ,\tag{24}
$$

where $\mathcal { P } _ { j } ( \cdot )$ denotes a lightweight channel compression operator. Its joint spectrum is obtained as

$$
Z _ { j o i n t } = \mathcal { F } ( X _ { j o i n t } ) .\tag{25}
$$

We then use the signed discrepancy gate to perform bounded residual modulation:

$$
Z _ { m o d } = Z _ { j o i n t } \odot ( 1 + \gamma G _ { d } ) ,\tag{26}
$$

where $\gamma$ controls the maximum modulation magnitude and is empirically set to 0.5. In this way, the original joint spectrum serves as a stable information basis, while cross-modal spectral discrepancy only controls an additional residual adjustment.

To further model interactions between the real and imaginary components, we convert the modulated complex spectrum into a real-valued representation:

$$
Q = [ \mathrm { R e } ( Z _ { m o d } ) , \mathrm { I m } ( Z _ { m o d } ) ] .\tag{27}
$$

A lightweight spectral transformation $\Phi ( \cdot )$ is then applied in a residual manner:

$$
Q ^ { o u t } = Q + \Phi ( Q ) .\tag{28}
$$

The resulting real and imaginary components are recombined into a complex spectrum and transformed back to the spatial domain:

$$
X _ { f f t } = \mathcal { F } ^ { - 1 } \left( \mathrm { C o m p l e x } ( Q _ { r e a l } ^ { o u t } , Q _ { i m a g } ^ { o u t } ) \right) .\tag{29}
$$

Finally, the reconstructed feature is projected back to the original multimodal dimensionality and combined with the concatenated input through a residual connection.

Overall, DGSM separates discrepancy estimation from multimodal content fusion: the discrepancy branch identifies where cross-modal spectral imbalance occurs, while the joint branch independently preserves the content to be fused. The former serves only as a learned control signal for the latter. This discrepancy-as-control formulation allows DGSM to exploit complementary spectral differences while reducing the risk of indiscriminately amplifying modality-specific noise.

## IV. EXPERIMENTS

## A. Datasets and Evaluation Metrics

Datasets. We evaluate our model on three commonly used drone-based visible–infrared object detection datasets: DroneVehicle [8], ESCVehicle [19], and ATR-UMOD [20].

(1) DroneVehicle Dataset. The DroneVehicle dataset is a large-scale RGB–IR benchmark captured from aerial perspective, containing 953,087 vehicle instances across 28,439 RGB– IR image pairs. It covers diverse scenarios, including urban roads, residential areas, and parking lots, under both daytime and nighttime conditions. The dataset provides annotations for five vehicle categories: car, bus, truck, van, and freight car. Following the official data split, we use 17,990 image pairs for training, 1,469 for validation, and 8,980 for testing.

(2) ESCVehicle dataset. The ESCVehicle dataset is a dronebased visible-infrared benchmark containing 10,727 aligned RGB–IR image pairs with 369,714 annotated vehicle instances. It covers seven vehicle categories and diverse scenarios, including urban areas, parking lots, roads, tree-covered regions, rain, fog, snow, lakes, and hills. Following the official split, we use 6,983 image pairs for training, 507 for validation, and 3,237 for testing. The diverse scene annotations make ESCVehicle particularly suitable for evaluating the robustness of multimodal vehicle detection methods under challenging environmental conditions.

TABLE I  
COMPARISON OF FOCAL WITH OTHER METHODS ON THE DRONEVEHICLE TEST SET. THE TERMS $\mathbf { \ddot { \Gamma } } \mathbf { R } \mathbf { G B } ^ { * }$ AND $\mathbf { \vec { \mu } } ^ { 4 } \mathbf { I R } ^ { 3 } \mathbf { \vec { \mu } }$ DENOTE DETECTION OUTCOMES FROM VISIBLE IMAGES AND INFRARED IMAGES, RESPECTIVELY, WHIL $\mathrm { ; \ ^ {  } R G B { + } I R ^ { \ } } $ REFLECTS COMBINED DETECTION OUTCOMES FROM FUSED INFRARED AND VISIBLE IMAGES. THE BEST AND SECOND-BEST RESULTS IN EACH CATEGORY ARE HIGHLIGHTED IN RED AND BLUE, RESPECTIVELY. <sup>†</sup> DENOTES THE RESULTS REPRODUCED BY OURSELVES UNDER THE SAME TRAINING AND EVALUATION SETTINGS.
<table><tr><td>Methods</td><td>Pub.</td><td>Modality</td><td> $\mathrm { \ m A P 5 0 }$ </td><td>mAP</td><td>Car</td><td>Truck</td><td>Bus</td><td>Van</td><td>Freight car</td><td>Param#</td></tr><tr><td>S2ANet [35]</td><td>TGRS’21</td><td>RGB</td><td>61.0</td><td>31.4</td><td>80.0</td><td>54.2</td><td>84.9</td><td>43.8</td><td>42.2</td><td>70.7M</td></tr><tr><td>YOLO11n † [34]</td><td>Ultralytics’24</td><td>RGB</td><td>69.1</td><td>47.9</td><td>93.6</td><td>62.0</td><td>90.4</td><td>50.4</td><td>49.2</td><td>2.7M</td></tr><tr><td>S2ANet [35]</td><td>TGRS’21</td><td>IR</td><td>67.5</td><td>40.4</td><td>89.9</td><td>54.5</td><td>88.9</td><td>48.4</td><td>55.8</td><td>70.7M</td></tr><tr><td> $\mathrm { Y O L O 1 1 n } ^ { \dag } \ [ 3 4 ]$ </td><td>Ultralytics’24</td><td>IR</td><td>79.2</td><td>63.3</td><td>98.3</td><td>76.8</td><td>95.0</td><td>60.4</td><td>65.7</td><td>2.7M</td></tr><tr><td>YOLO11n-Dual † [34]</td><td>Ultralytics’24</td><td>RGB+IR</td><td>80.8</td><td>65.2</td><td>98.5</td><td>80.5</td><td>95.8</td><td>62.2</td><td>67.0</td><td>4.0M</td></tr><tr><td>C2Former [36]</td><td>TGRS&#x27;24</td><td> $\operatorname { R G B + I R }$ </td><td>74.2</td><td>47.5</td><td>90.2</td><td>68.3</td><td>89.8</td><td>58.5</td><td>64.4</td><td>118.5M</td></tr><tr><td>CALNet [37]</td><td>ACM MM&#x27;23</td><td> $\operatorname { R G B + I R }$ </td><td>75.4</td><td>48.2</td><td>90.3</td><td>76.2</td><td>89.1</td><td>58.5</td><td>63.0</td><td>39.5M</td></tr><tr><td>OAFA [38]</td><td>CVPR&#x27;24</td><td> $\operatorname { R G B + I R }$ </td><td>77.1</td><td>50.1</td><td>90.1</td><td>75.4</td><td>89.8</td><td>61.8</td><td>68.2</td><td></td></tr><tr><td>M2D-LIF [39]</td><td>ICCV’25</td><td> $\operatorname { R G B + I R }$ </td><td>81.4</td><td>68.1</td><td>97.8</td><td>81.0</td><td>96.0</td><td>64.6</td><td>67.9</td><td>37.1M</td></tr><tr><td>FusionMamba [26]</td><td>TMM&#x27;25</td><td> $\operatorname { R G B + I R }$ </td><td>79.2</td><td>56.0</td><td>96.7</td><td>80.2</td><td>95.3</td><td>64.0</td><td>59.8</td><td>287.6M</td></tr><tr><td>WaveMamba [18]</td><td>ICCV’25</td><td> $\operatorname { R G B + I R }$ </td><td>79.8</td><td>60.5</td><td>95.0</td><td>80.4</td><td>90.6</td><td>64.5</td><td>68.5</td><td>69.1M</td></tr><tr><td> $\mathrm { C ^ { 2 } D F F  – N e t } ^ { \mathrm { ~ \dagger ~ } } \left[ 4 0 \right]$ </td><td>TGRS&#x27;25</td><td> $\operatorname { R G B + I R }$ </td><td>82.0</td><td>66.7</td><td>98.6</td><td>82.2</td><td>95.9</td><td>64.0</td><td>69.5</td><td>6.6M</td></tr><tr><td> $\mathrm { C O M O } ^ { \dag } \ [ 4 1 ]$ </td><td>Inf. Fusion&#x27;26</td><td> $\operatorname { R G B + I R }$ </td><td>81.2</td><td>66.7</td><td>97.8</td><td>81.3</td><td>94.8</td><td>63.9</td><td>68.4</td><td>6.5M</td></tr><tr><td> $\mathrm { C r o s s W e a v e r } ^ { \textsf { \textsf { T } } } [ 4 2 ]$ </td><td>CVPR&#x27;26</td><td> $\operatorname { R G B + I R }$ </td><td>80.6</td><td>57.3</td><td>98.4</td><td>79.2</td><td>95.0</td><td>63.2</td><td>67.3</td><td>5.4M</td></tr><tr><td> $\operatorname { F o C a l } \ ( \mathrm { O u r s } )$ </td><td></td><td> $\operatorname { R G B + I R }$ </td><td>83.5</td><td>68.6</td><td>98.6</td><td>83.6</td><td>96.2</td><td>66.9</td><td>72.0</td><td>3.0M</td></tr></table>

(3) ATR-UMOD Dataset. The ATR-UMOD dataset is a high-diversity UAV-based RGB–IR benchmark containing 13,353 aligned image pairs across 11 object categories. It covers diverse imaging conditions, including flight altitudes from 80 m to 300 m, camera angles from $0 ^ { \circ } ~ \mathrm { t o } ~ 7 5 ^ { \circ }$ , all-day and all-year acquisition, as well as various weather, illumination, and scenario conditions. Each image pair is additionally annotated with six condition attributes, including altitude, angle, time, weather, illumination, and scenario, providing a comprehensive benchmark for evaluating multimodal detection under complex conditions. Following the official split, we use 11,850 image pairs for training and 1,503 for testing.

Evaluation Metrics. We adopt standard evaluation metrics, including $\mathrm { \ m A P _ { 5 0 } }$ and mAP, to assess model performance on different datasets. Specifically, $\mathrm { \ m A P _ { 5 0 } }$ denotes the mean Average Precision at an IoU threshold of 0.50, while mAP is averaged over IoU thresholds from 0.50 to 0.95 with a step size of 0.05.

## B. Implementation Details

All experiments are conducted on a single NVIDIA RTX 4090 GPU with 24 GB memory, using CUDA 12.1 and PyTorch 2.5.1. Our model is implemented by extending the Ultralytics [34] single-modal framework into a dual-stream multimodal architecture. To ensure a fair evaluation of the proposed architectural designs, all models are trained entirely from scratch without relying on any pre-trained weights. We train all models using the SGD optimizer with a momentum of 0.937, a weight decay of 0.0005, and an initial learning rate of 0.01. For DroneVehicle, ESCVehicle, and ATR-UMOD, the batch size is set to 16, and the number of training epochs is set to 150. Unless otherwise specified, the loss functions, other hyperparameters, and data augmentation settings follow the default configuration.

## C. Comparisons with State-of-the-Art Methods

Comparison on DroneVehicle. Table I reports the comparison with representative single-modal and RGB–IR object detectors on the DroneVehicle test set. FoCal achieves the best overall performance, reaching 83.5% m $\mathrm { A P _ { 5 0 } }$ and 68.6% mAP. Specifically, it surpasses the second-best $\mathrm { C ^ { 2 } D F F  – N e t }$ by 1.5 percentage points in $\mathrm { m A P _ { 5 0 } }$ and M<sup>2</sup>D-LIF by 0.5 percentage points in mAP. Compared with the YOLO11n-Dual baseline, FoCal further improves $\mathrm { m A P _ { 5 0 } }$ from 80.8% to 83.5% and mAP from 65.2% to 68.6%, corresponding to gains of 2.7 and 3.4 percentage points, respectively. This demonstrates the effectiveness of the proposed cross-modal interaction and spectral calibration beyond conventional dual-stream fusion. At the category level, FoCal achieves APs of 98.6%, 83.6%, 96.2%, 66.9%, and 72.0% for car, truck, bus, van, and freight car, respectively. It ranks first on four of the five categories and ties for the best result on car. Particularly, FoCal improves the second-best AP by 1.4, 2.3, and 2.5 percentage points for truck, van, and freight car, respectively, indicating consistent improvements across object categories. FoCal also maintains high computational efficiency. With only 3.0M parameters, it has the smallest model size among the compared RGB–IR methods with reported parameter counts, while outperforming substantially larger models such as M<sup>2</sup>D-LIF (37.1M), WaveMamba (69.1M), and FusionMamba (287.6M). Moreover, FoCal reduces the parameter count by 25% relative to YOLO11n-Dual (4.0M) while achieving higher detection accuracy, demonstrating a favorable accuracy–complexity tradeoff.

Comparison on ESCVehicle. Table II reports the comparison with representative single-modal and RGB–IR detectors on the ESCVehicle dataset. FoCal achieves the best overall performance, reaching 54.8% $\mathrm { m A P _ { 5 0 } }$ and 41.7% mAP. Compared with the second-best $\mathrm { C ^ { 2 } \mathrm { - } V e D }$ , FoCal improves $\mathrm { n A P _ { 5 0 } }$ and mAP by 2.4 and 3.1 percentage points, respectively. It also consistently outperforms other representative RGB– IR methods, demonstrating the effectiveness of the proposed cross-modal interaction and spectral calibration. Notably, Fo-Cal achieves these improvements with only 3.0M parameters and 8.4G FLOPs, both being the lowest among all compared methods. Compared with $\mathrm { C ^ { 2 } \mathrm { - } V e D }$ , FoCal reduces the model size from 16.5M to 3.0M and the computational cost from 13.1G to 8.4G while achieving higher accuracy. These results demonstrate that FoCal simultaneously improves detection effectiveness and computational efficiency.

COMPARISON OF FOCAL WITH OTHER METHODS ON THE ESCVEHICLE DATASET.  
TABLE II
<table><tr><td>Method</td><td>Modality</td><td>mAP50</td><td>mAP</td><td>Param#</td><td>FLOPs</td></tr><tr><td>RoI Transformer [43]</td><td>RGB</td><td>37.2</td><td>19.3</td><td>87.2M</td><td>148.7G</td></tr><tr><td>S2ANet [35]</td><td>RGB</td><td>35.5</td><td>17.5</td><td>70.7M</td><td>120.6G</td></tr><tr><td>Oriented R-CNN [44]</td><td>RGB</td><td>39.3</td><td>20.6</td><td>73.2M</td><td>134.8G</td></tr><tr><td>KLD [45]</td><td>RGB</td><td>32.9</td><td>16.2</td><td>41.9M</td><td>131.2G</td></tr><tr><td>RoI Transformer [43]</td><td>IR</td><td>22.0</td><td>10.5</td><td>87.2M</td><td>148.7G</td></tr><tr><td>S2ANet [35]</td><td>IR</td><td>21.1</td><td>9.7</td><td>70.7M</td><td>120.6G</td></tr><tr><td>Oriented R-CNN [44]</td><td>IR</td><td>24.3</td><td>11.4</td><td>73.2M</td><td>134.8G</td></tr><tr><td>OSIV-Net [46]</td><td>IR</td><td>37.0</td><td>19.1</td><td>10.8M</td><td>29.38G</td></tr><tr><td>CFT [24]</td><td>RGB+IR</td><td>41.7</td><td>23.9</td><td>44.7M</td><td>44.0G</td></tr><tr><td>ICAFusion [14]</td><td>RGB+IR</td><td>45.0</td><td>30.3</td><td>24.5M</td><td>39.2G</td></tr><tr><td>CALNet [37]</td><td>RGB+IR</td><td>48.8</td><td>31.8</td><td>39.5M</td><td>52.9G</td></tr><tr><td>C2Former [36]</td><td>RGB+IR</td><td>48.7</td><td>30.8</td><td>36.8M</td><td>80.3G</td></tr><tr><td>CMA-Det [47]</td><td>RGB+IR</td><td>47.6</td><td>27.9</td><td>28.9M</td><td>39.8G</td></tr><tr><td>C2-VeD [19]</td><td>RGB+IR</td><td>52.4</td><td>38.6</td><td>16.5M</td><td>13.1G</td></tr><tr><td>COMO † [41]</td><td>RGB+IR</td><td>49.5</td><td>37.8</td><td>6.5M</td><td>25.5G</td></tr><tr><td>CrossWeaver † [42]</td><td>RGB+IR</td><td>45.4</td><td>31.0</td><td>5.4M</td><td>19.5G</td></tr><tr><td>FoCal (Ours)</td><td>RGB+IR</td><td>54.8</td><td>41.7</td><td>3.0M</td><td>8.4G</td></tr></table>

TABLE III

COMPARISON OF FOCAL WITH OTHER METHODS ON THE ATR-UMOD DATASET. ALL METHODS PERFORM LOCALIZATION AND CLASSIFICATION USING ORIENTED BOUNDING BOX (OBB) HEADS. THE CATEGORIES, CAR, SUV, VAN, BUS, FREIGHT CAR, TRUCK, MOTORCYCLE, TRAILER, EXCAVATOR, CRANE, AND TANK TRUCK, ARE ABBREVIATED AS CR, SV, VN, BS, FC, TK, ME, TR, ER, CE, AND TT, RESPECTIVELY.
<table><tr><td>Detectors</td><td>Modality</td><td>CR</td><td>SV</td><td>VN</td><td>BS</td><td>FC</td><td>TK</td><td>TT</td><td>TR</td><td>CE</td><td>ER</td><td>ME</td><td> $\mathrm { \ m A P 5 0 }$ </td></tr><tr><td rowspan="3">S²ANet [35] ReDet [48] RoI Transformer [43] Oriented R-CNN [44]</td><td rowspan="3">RGB</td><td>34.2 37.2</td><td>44.9 52.5</td><td>45.9 51.6</td><td>69.2</td><td>24.4</td><td>37.4</td><td>5.6</td><td>22.5</td><td>49.5</td><td>31.2</td><td>25.6</td><td>35.5</td></tr><tr><td>36.9</td><td>51.9</td><td></td><td>74.8</td><td>33.5</td><td>48.1 46.8</td><td>16.7 18.2</td><td>40.7 36.3</td><td>61.4 58.9</td><td>36.8 38.3</td><td>32.9 25.3</td><td>41.1</td></tr><tr><td>53.3 36.9 52.5</td><td>51.6</td><td>71.5 74.8</td><td>30.1 33.5</td><td>48.1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>42.5 44.2</td></tr><tr><td rowspan="3">YOLOv5s [34] S2ANet [35] ReDet [48]</td><td rowspan="3">IR</td><td>45.8</td><td>60.7</td><td>57.5</td><td>75.2</td><td>41.6</td><td>52.1</td><td>16.7 18.2</td><td>40.7 42.3</td><td>61.7 68.7</td><td>36.8 47.5</td><td>32.9 47.4</td><td>50.7</td></tr><tr><td></td><td></td><td></td><td></td><td>35.5</td><td>24.3</td><td>31.4</td><td>16.0</td><td>10.8</td><td>1.0</td><td>32.0</td><td></td></tr><tr><td>50.2 35.9 57.4 42.6</td><td>31.8 38.8</td><td>59.9 70.4</td><td>42.3</td><td>31.5</td><td></td><td>52.0</td><td>15.9</td><td>33.7</td><td></td><td></td><td>29.9 37.9</td></tr><tr><td rowspan="3">RoI Transformer [43] Oriented R-CNN [44] YOLOv5s [34] UA-CMDet [8]</td><td rowspan="3"></td><td>54.6 57.5</td><td>41.8</td><td>38.7</td><td>64.0</td><td>43.1</td><td>33.6</td><td>61.0</td><td>23.4</td><td>32.8</td><td>8.7 7.0</td><td>23.1 23.4</td><td>38.5</td></tr><tr><td></td><td>41.6 36.8</td><td></td><td>63.8</td><td>43.5</td><td>28.6</td><td>64.3</td><td>28.5</td><td>44.2</td><td>6.9</td><td>23.9</td><td>40.0</td></tr><tr><td>65.8</td><td>51.2 51.6</td><td>75.3</td><td>53.1</td><td>38.9</td><td></td><td>83.3</td><td>46.2</td><td>57.9</td><td>12.0</td><td>42.7</td><td>52.5</td></tr><tr><td rowspan="9">C2Former [36] TINet [49] CALNet [37] OAFA [38] RGB+IR YOLOrs [50] CrossWeaver † [42] PCDF [20]</td><td rowspan="9"></td><td>50.9</td><td>43.3</td><td>47.9</td><td>75.8</td><td>51.4</td><td>44.5</td><td>42.8</td><td>40.1</td><td>54.8</td><td>39.6</td><td>23.2</td><td>46.8</td></tr><tr><td>60.5</td><td>53.3</td><td>51.6</td><td>81.6</td><td>46.1</td><td>44.7</td><td>46.6</td><td>29.3</td><td>56.3</td><td>36.8</td><td>40.0</td><td>49.7</td></tr><tr><td>60.2</td><td>51.4</td><td>54.4</td><td>74.5</td><td>50.2</td><td>46.0</td><td>44.6</td><td>39.7</td><td>59.0</td><td>47.5</td><td>27.0</td><td>50.4</td></tr><tr><td>71.9</td><td>65.5</td><td>71.0</td><td>78.4</td><td>53.6</td><td>51.2</td><td>37.7</td><td>35.3</td><td>56.3</td><td>31.9</td><td>38.6</td><td>53.8</td></tr><tr><td>70.4</td><td>59.6</td><td>63.1</td><td>81.5</td><td>60.1</td><td>47.5</td><td>80.1</td><td>32.4</td><td>59.0</td><td>33.0</td><td>50.1</td><td>57.9</td></tr><tr><td>73.2</td><td>62.6</td><td>66.3</td><td>81.8</td><td>61.1</td><td>48.2</td><td>70.3</td><td>37.6</td><td>64.3</td><td>41.8</td><td>52.9</td><td>60.0</td></tr><tr><td>74.2</td><td>59.8</td><td>62.3</td><td>87.3</td><td>61.8</td><td>48.6</td><td>81.6</td><td>43.5</td><td>67.5</td><td>34.9</td><td>56.6</td><td>61.7</td></tr><tr><td>70.8</td><td>60.6</td><td>65.4</td><td>84.3</td><td>62.1</td><td>51.3</td><td>86.1</td><td>42.5</td><td>71.1</td><td>49.0</td><td>51.2</td><td>63.1</td></tr><tr><td>76.0</td><td>62.9</td><td>67.6</td><td>87.1</td><td>63.8</td><td>47.6</td><td>82.8</td><td>47.6</td><td>75.4</td><td>42.2</td><td>59.2</td><td>64.2</td></tr><tr><td>COMO† [41] FoCal (Ours)</td><td></td><td>68.3</td><td>60.7</td><td>64.6</td><td>91.1</td><td>61.5</td><td>46.4</td><td>82.3</td><td>52.1</td><td>78.3</td><td>58.5</td><td>47.0</td><td>64.6</td></tr></table>

Comparison on ATR-UMOD. Table III presents the comparison with representative single-modal and RGB–IR detectors on the ATR-UMOD dataset. FoCal achieves the best overall performance with an m $\mathrm { A P _ { 5 0 } }$ of 64.6%. It also consistently outperforms other representative multimodal methods, including OAFA, COMO, and CrossWeaver, demonstrating the effectiveness of the proposed frequency-dependent cross-modal interaction and discrepancy-guided spectral modulation. At the category level, FoCal achieves the best AP on bus, trailer, crane, and excavator, reaching 91.1%, 52.1%, 78.3%, and 58.5%, respectively. The pronounced gains on these categories indicate that FoCal can effectively exploit complementary RGB–IR information across objects with diverse appearance, contributing to its superior overall detection performance.

## D. Ablation Studies

To validate the effectiveness of the proposed components in FoCal, we conduct ablation studies on DroneVehicle to examine the contributions of the architectural redesign, FADC, and DGSM, followed by a comparison of frequency interaction strategies.

TABLE IV  
ABLATION STUDY OF INDIVIDUAL COMPONENTS IN FOCAL ON THEDRONEVEHICLE DATASET.
<table><tr><td>RR</td><td>FADC</td><td>DGSM</td><td> $\mathrm { \ m A P 5 0 }$ </td><td> $\mathrm { \ m A P 7 5 }$ </td><td>mAP</td><td>Param#</td><td>FLOPs</td></tr><tr><td>x</td><td>x</td><td>x</td><td>80.8</td><td>76.6</td><td>65.2</td><td>4.01M</td><td>9.5G</td></tr><tr><td>√</td><td>x</td><td>x</td><td>81.9</td><td>78.5</td><td>67.3</td><td>2.63M</td><td>6.8G</td></tr><tr><td>√</td><td>√</td><td>x</td><td>82.9</td><td>79.5</td><td>68.2</td><td>2.75M</td><td>7.3G</td></tr><tr><td>√</td><td>x</td><td>√</td><td>82.6</td><td>79.1</td><td>68.0</td><td>2.84M</td><td>7.9G</td></tr><tr><td>√</td><td>√</td><td>√</td><td>83.5</td><td>79.9</td><td>68.6</td><td>2.96M</td><td>8.4G</td></tr></table>

Redundancy Reduction. Considering the structural redundancy of dual-stream feature extraction, we adopt a lightweight redundancy reduction (RR) strategy. Specifically, the modalityspecific branches are retained up to P4, while a shared P5 representation is constructed from the fused P4 feature through parameter-free downsampling followed by high-level semantic encoding, thereby avoiding duplicated deep feature extraction. In addition, RR employs a compact two-scale prediction hierarchy on P4 and P5. The high-resolution P3 representation remains involved in bidirectional multi-scale aggregation and continuously supplies fine-grained spatial cues to subsequent features, while final predictions are concentrated on the semantically stronger P4 and P5 levels. This design decouples feature-scale aggregation from prediction-scale allocation and provides a compact detection architecture without discarding high-resolution information. As shown in Table IV, RR improves m $\mathrm { A P _ { 5 0 } , m A P _ { 7 5 } }$ , and mAP from 80.8%, 76.6%, and 65.2% to 81.9%, 78.5%, and 67.3%, respectively, while reducing the parameter count from 4.01M to 2.63M and FLOPs from 9.5G to 6.8G. This avoids duplicated high-level computation and redundant dense predictions, yielding a more compact architecture while preserving detection accuracy.

Effectiveness of Individual Components. Introducing RR improves these metrics to 81.9%, 78.5%, and 67.3%, respectively, while reducing the parameter count from 4.01M to 2.63M and FLOPs from 9.5G to 6.8G. This provides a more compact and effective starting point for subsequent cross-modal modeling. Building upon RR, FADC further improves $\mathrm { \ m A P _ { 5 0 } }$ , $\mathrm { \ m A P _ { 7 5 } , }$ and mAP to 82.9%, 79.5%, and 68.2%, respectively, with only 0.12M additional parameters and 0.5G FLOPs. This verifies the effectiveness of modeling different cross-modal interaction patterns for low- and highfrequency components and transferring the resulting frequency cues back to the original feature streams for calibration. Independently adding DGSM to RR achieves $8 2 . 6 \% \mathrm { \ m A P _ { 5 0 } }$ 79.1% $\mathrm { \ m A P _ { 7 5 } }$ , and 68.0% mAP. The consistent improvement demonstrates that confidence-weighted spectral discrepancy provides complementary guidance for regulating the joint multimodal spectrum. Combining FADC and DGSM yields the best overall performance, reaching 83.5% ${ \mathrm { m A P } } _ { 5 0 } ,$ , 79.9% $\mathrm { \ m A P _ { 7 5 } }$ , and 68.6% mAP. These results indicate that the two modules provide complementary benefits: FADC focuses on frequency-dependent cross-modal interaction and originaldomain calibration, whereas DGSM regulates the joint spectral representation using cross-modal discrepancy cues.

Effectiveness of Frequency-Dependent Interaction. To investigate whether different frequency components favor different cross-modal interaction patterns, we compare four configurations in Table V. Applying selective exchange to both frequency bands (Sym.Ex.) achieves 82.3% $\mathrm { \ m A P _ { 5 0 } } .$ , 78.8% $\mathrm { \ m A P _ { 7 5 } }$ , and 67.6% mAP, while uniformly adopting structural consensus (Sym.Con.) improves the results to 82.7%, 79.2%, and 68.0%, respectively. These results show that both interaction mechanisms are effective, but a uniform strategy does not explicitly account for the distinct characteristics of different frequency components. We further reverse the proposed assignment by applying structural consensus to high-frequency components and selective exchange to low-frequency components (Reverse). Under the same model complexity as our design, this configuration obtains 82.4% $\mathrm { \ m A P _ { 5 0 } }$ , 78.9% $\mathrm { \ m A P _ { 7 5 } }$ , and 67.7% mAP. In contrast, our frequency-dependent strategy applies selective exchange to high-frequency components and structural consensus to low-frequency components, improving the three metrics to 82.9%, 79.5%, and 68.2%, respectively. These results support the proposed frequencydependent interaction principle: low-frequency components benefit from cross-modal structural consensus, whereas highfrequency components are better modeled by preserving modality-specific information through selective exchange.

TABLE V  
ABLATION STUDY OF DIFFERENT FREQUENCY INTERACTION STRATEGIES ON THE DRONEVEHICLE DATASET. “EX.” AND “CON.” DENOTE SELECTIVE EXCHANGE AND STRUCTURAL CONSENSUS, RESPECTIVELY.
<table><tr><td>Variant</td><td>HF</td><td>LF</td><td> $\mathrm { \ m A P 5 0 }$ </td><td> $\mathrm { \ m A P 7 5 }$ </td><td>mAP</td><td>Param#</td><td>FLOPs</td></tr><tr><td>Sym.</td><td>Ex.</td><td>Ex.</td><td>82.3</td><td>78.8</td><td>67.6</td><td>2.71M</td><td>7.1G</td></tr><tr><td>Sym.</td><td>Con.</td><td>Con.</td><td>82.7</td><td>79.2</td><td>68.0</td><td>2.79M</td><td>7.5G</td></tr><tr><td>Reverse</td><td>Con.</td><td>Ex.</td><td>82.4</td><td>78.9</td><td>67.7</td><td>2.75M</td><td>7.3G</td></tr><tr><td>Ours</td><td>Ex.</td><td>Con.</td><td>82.9</td><td>79.5</td><td>68.2</td><td>2.75M</td><td>7.3G</td></tr></table>

## E. Visualization

We further provide qualitative visualizations to examine the behavior of FoCal in challenging aerial scenes. As shown in Fig. 5, the compared examples contain densely distributed vehicles under low illumination and weak visual contrast. In the first scene, both the baseline and CrossWeaver miss the highlighted target, whereas COMO and FoCal successfully detect it. In the second scene, the baseline and COMO fail to identify the highlighted vehicle within the densely arranged parking area, while CrossWeaver and FoCal recover it. In both cases, FoCal successfully preserves the target responses while maintaining stable detections for surrounding vehicles. The paired RGB and IR observations further illustrate the complementary characteristics of the two modalities, especially when target visibility is degraded in the visible spectrum. These results qualitatively demonstrate that FoCal can more effectively exploit complementary cross-modal cues in complex aerial environments.

Fig. 6 visualizes the intermediate frequency representations produced by FADC. The RGB and IR low-frequency maps exhibit different response distributions before interaction, while the consensus operation progressively consolidates them into a shared representation with clearer structural responses around vehicle regions. In contrast, the high-frequency branch maintains two modality-specific streams after selective exchange, preserving distinct local details while incorporating complementary cues from the counterpart modality. The response heatmaps in Fig. 7 further illustrate the effect of this interaction on the final representation. Compared with the baseline, FoCal produces more concentrated activations around vehicle locations and weaker diffuse responses in background regions, particularly under low-contrast RGB conditions. The IR branch also maintains stable responses on target regions, indicating that complementary thermal cues are effectively retained during fusion.

![](images/9aa4606411356d53296b383314c10f6be9eb86d3ca595e1cf205d0dad625a6a6.jpg)

Fig. 5. Qualitative comparison of the proposed method against other competing methods under complex aerial scenarios.  
![](images/0bee1ca7f2abf2a993efff149af5efac53775c12c75447935c3f132b487b8350.jpg)  
Fig. 6. Visualization of the FADC frequency pathway at P3. The top and bottom rows correspond to RGB and IR. From left to right, the panels show the input images, raw low-frequency (LF) components, the shared LF representation after cooperative interaction, raw high-frequency (HF) components, HF components after competitive interaction, and the recomposed LF+HF features. One shared LF map is produced, whereas the HF and recomposed features remain modality-specific.

TABLE VI

COMPARISON OF MODEL COMPLEXITY AND INFERENCE EFFICIENCY. LATENCY AND FPS ARE MEASURED USING THE TRAINED .P T MODELS IN PYTORCH ON AN NVIDIA RTX 4090 WITH A BATCH SIZE OF 1.

<table><tr><td>Method</td><td>Param#</td><td>FLOPs</td><td> $\mathrm { \ m A P _ { 5 0 } }$ </td><td>Latency (ms/pair)↓</td><td>FPS ↑</td></tr><tr><td>CrossWeaver [42]</td><td>5.4M</td><td>19.5G</td><td>80.6</td><td>32.1</td><td>31.2</td></tr><tr><td>COMO [41]</td><td>6.5M</td><td>25.5G</td><td>81.2</td><td>15.5</td><td>64.5</td></tr><tr><td>FoCal (Ours)</td><td>3.0M</td><td>8.4G</td><td>83.5</td><td>8.8</td><td>113.6</td></tr></table>

## V. DISCUSSION

Overall, the experimental results on DroneVehicle, ESCVehicle, and ATR-UMOD demonstrate that FoCal consistently achieves strong detection performance across different aerial RGB–IR benchmarks. In particular, the improvements in both overall detection accuracy and category-level performance indicate that frequency-dependent cross-modal interaction can effectively exploit complementary information between visible and infrared modalities. Meanwhile, the compact architecture of FoCal suggests that these accuracy gains do not rely on increasing model capacity, providing a favorable basis for practical multimodal detection.

![](images/2d5bac726b5568b5d87d8ec9e128b8a985bff60703d3f76977978c752a9b71ef.jpg)  
Fig. 7. Comparison of heatmap visualizations across different input modalities. From top to bottom: visible and infrared modalities. From left to right: baseline model and the proposed FoCal.

Beyond detection accuracy, computational efficiency is particularly important for aerial platforms, where real-time processing and limited computing resources are common constraints. As shown in Table VI, FoCal achieves 83.5% mAP with only 3.0M parameters and 8.4G FLOPs, while reaching 113.6 FPS with a latency of 8.8 ms per RGB–IR pair on an NVIDIA RTX 4090. In comparison, COMO achieves 81.2% m $\mathrm { \bf A P _ { 5 0 } }$ at 64.5 FPS, while CrossWeaver obtains 80.6% mAP<sub>50</sub> at 31.2 FPS. The lower FPS of COMO and CrossWeaver may be associated with their more elaborate cross-modal interaction pipelines, including cross-Mamba-based global– local modeling in COMO and deformation-aware hierarchical interaction in CrossWeaver. In contrast, FoCal adopts a more compact frequency-oriented interaction design, reducing inference overhead while preserving strong detection accuracy. Thus, FoCal simultaneously provides higher detection accuracy, lower computational complexity, and substantially higher forward-pass throughput. These characteristics indicate its strong potential for real-time RGB–IR object detection and make it particularly suitable for future deployment on aerial and other latency-sensitive multimodal perception platforms.

Future work will therefore investigate end-to-end deployment optimization, including mixed-precision inference, TensorRT acceleration, and hardware-aware optimization on embedded and edge computing platforms.

## VI. CONCLUSION

In this work, we proposed FoCal, a frequency-oriented framework for aerial RGB–IR object detection. FoCal explicitly exploits the distinct characteristics of different frequency components to improve cross-modal interaction. Specifically, FADC establishes structural consensus in the low-frequency branch while preserving modality-specific information through selective high-frequency exchange, and further transfers the resulting frequency-aware cues to calibrate the original feature streams. DGSM complements this process by converting confidence-weighted spectral discrepancy into a signed modulation signal for adaptive regulation of the joint multimodal spectrum. Extensive experiments on DroneVehicle, ESCVehicle, and ATR-UMOD demonstrate that FoCal consistently achieves state-of-the-art performance, with m $\mathrm { A P _ { 5 0 } }$ values of 83.5%, 54.8%, and 64.6%, respectively. Meanwhile, the model contains only 3.0M parameters and exhibits favorable inference efficiency, highlighting a strong trade-off between detection accuracy and computational cost. Future work will explore hardware-aware optimization and extend frequencydependent interaction to temporal and more diverse multimodal perception scenarios.

## REFERENCES

[1] X. Liu, H. Zou, J. Li, L. Pan, S. He, X. Cao, Y. Zhang, J. Li, and W. Chen, “D2-detr: Detr with dual-domain frequency-spatial modeling for unmanned aerial vehicle imagery object detection,” Knowledge-Based Systems, p. 115788, 2026.

[2] B. Liang, Y. Liu, C. Sui, Y. Wang, L. Xiao, X. Sui, and Q. Chen, “Multi-scale pattern-aware task-gating network for aerial small object detection,” Neural Networks, p. 108680, 2026.

[3] R. Wang, J. Yang, M. Li, and D. Hao, “Efficient small object detection based on multi-level implicit feature enhancement and presence region mask guidance,” Knowledge-Based Systems, p. 115020, 2025.

[4] Y. Liu, X. Ji, H. Wei, B. Qiu, Y. Liu, X. Sui, and Q. Chen, “Dmsa-net: Dual-domain multiscale attentive network for robust sar ship detection,” IEEE Transactions on Aerospace and Electronic Systems, 2025.

[5] H. Wei, N. Wang, Y. Liu, P. Ma, D. Pang, X. Sui, and Q. Chen, “Spatio-temporal feature fusion and guide aggregation network for remote sensing change detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–16, 2024.

[6] H. Wei, Y. Liu, Y. Ma, D. Pang, Y. Ye, X. Sui, and Q. Chen, “Multi-scale spatial frequency fusion and prior change guidance network for remote sensing change detection,” IEEE Transactions on Instrumentation and Measurement, 2025.

[7] Q. Chen, T. Xu, Y. Sun, B. Wang, and Y. Han, “Tiny object detection via implicit feature fusion and hybrid metric adaptive label assignment,” Knowledge-Based Systems, p. 116218, 2026.

[8] Y. Sun, B. Cao, P. Zhu, and Q. Hu, “Drone-based rgb-infrared crossmodality vehicle detection via uncertainty-aware learning,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 10, pp. 6700–6713, 2022.

[9] Y. Shi, Y. Liu, T. Liu, C. Li, L. Yuan, X. Sui, and Q. Chen, “Spgfuse: A dual-task semantic perception guided network for infrared and visible image fusion,” Optics and Lasers in Engineering, vol. 195, p. 109240, 2025.

[10] T. Jiang, X. Kuang, S. Wang, T. Liu, Y. Liu, X. Sui, and Q. Chen, “Crossdomain colorization of unpaired infrared images through contrastive learning guided by color feature selection attention,” Optics Express, vol. 32, no. 9, pp. 15008–15024, 2024.

[11] T. Li, M. Ye, T. Wu, N. Li, S. Li, S. Tang, and L. Ji, “Pseudo visible feature fine-grained fusion for thermal object detection,” in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6710– 6719, 2025.

[12] L. Tang, Z. Chen, J. Huang, and J. Ma, “Camf: An interpretable infrared and visible image fusion network based on class activation mapping,” IEEE Transactions on Multimedia, vol. 26, pp. 4776–4791, 2023.

[13] X. Li, S. Chen, C. Tian, H. Zhou, and Z. Zhang, “M2fnet: Mask-guided multi-level fusion for rgb-t pedestrian detection,” IEEE Transactions on Multimedia, vol. 26, pp. 8678–8690, 2024.

[14] J. Shen, Y. Chen, Y. Liu, X. Zuo, H. Fan, and W. Yang, “Icafusion: Iterative cross-attention guided feature fusion for multispectral object detection,” Pattern Recognition, vol. 145, p. 109913, 2024.

[15] K. Chen, J. Liu, and H. Zhang, “Igt: Illumination-guided rgb-t object detection with transformers,” Knowledge-Based Systems, vol. 268, p. 110423, 2023.

[16] J. Zhang, J. Lei, W. Xie, Z. Fang, Y. Li, and Q. Du, “Superyolo: Super resolution assisted object detection in multimodal remote sensing imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–15, 2023.

[17] K. Li, D. Wang, Z. Hu, S. Li, W. Ni, L. Zhao, and Q. Wang, “Fd2- net: Frequency-driven feature decomposition network for infrared-visible object detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 4797–4805, 2025.

[18] H. Zhu, W. Dong, L. Yang, H. Li, Y. Yang, Y. Ren, Q. Zhu, Z. Feng, C. Li, S. Lin, et al., “Wavemamba: Wavelet-driven mamba fusion for rgb-infrared object detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11219–11229, 2025.

[19] J. Song, N. Zhang, Z. Wang, and T. Tian, “Escvehicle: A drone-based visible-infrared vehicle benchmark with extensive scene coverage,” IEEE Transactions on Geoscience and Remote Sensing, 2026.

[20] C. Chen, K. Bin, T. Hu, J. Qi, X. Liu, T. Liu, Z. Liu, Y. Liu, and P. Zhong, “Fusion meets diverse conditions: A high-diversity benchmark and baseline for uav-based multimodal object detection with condition cues,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 27958–27967, IEEE, 2025.

[21] Y. Shi, Y. Liu, B. Qiu, T. Liu, X. Sui, and Q. Chen, “Infrared and visible image fusion via pre-fusion compensation and visual-gradient saliency detection,” Infrared Physics & Technology, p. 106000, 2025.

[22] Z. Zhao, H. Bai, J. Zhang, Y. Zhang, S. Xu, Z. Lin, R. Timofte, and L. Van Gool, “Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 5906–5916, 2023.

[23] Z. Li, H. Pan, K. Zhang, Y. Wang, and F. Yu, “Mambadfuse: A mambabased dual-phase model for multi-modality image fusion,” arXiv preprint arXiv:2404.08406, 2024.

[24] F. Qingyun, H. Dapeng, and W. Zhaokui, “Cross-modality fusion transformer for multispectral object detection,” arXiv preprint arXiv:2111.00273, 2021.

[25] F. Qingyun and W. Zhaokui, “Cross-modality attentive feature fusion for object detection in multispectral remote sensing imagery,” Pattern Recognition, vol. 130, p. 108786, 2022.

[26] W. Dong, H. Zhu, S. Lin, X. Luo, Y. Shen, G. Guo, and B. Zhang, “Fusion-mamba for cross-modality object detection,” IEEE Transactions on Multimedia, 2025.

[27] G. Zhang, X. Ji, B. Qiu, Y. Cai, Y. Liu, X. Sui, and Q. Chen, “Sfnet: A dual-enhanced rgbt tracker via global-local modality refinement and frequency-spatial cross-modal fusion,” Optics and Lasers in Engineering, vol. 194, p. 109201, 2025.

[28] Z. Shi, J. Hu, J. Ren, H. Ye, X. Yuan, Y. Ouyang, J. He, B. Ji, and J. Guo, “Hs-fpn: High frequency and spatial perception fpn for tiny object detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 6896–6904, 2025.

[29] Y. Xiang, K. Zhao, Z. Yu, X. Yuan, G. Huang, J. Tian, and J. Li, “Dfformer: Capturing dynamic frequency features to locate image manipulation through adaptive frequency transformer and prototype learning,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[30] K. Hu, Q. Zhang, M. Yuan, and Y. Zhang, “Sfdfusion: an efficient spatial-frequency domain fusion network for infrared and visible image fusion,” arXiv preprint arXiv:2410.22837, 2024.

[31] S. E. Finder, R. Amoyal, E. Treister, and O. Freifeld, “Wavelet convolutions for large receptive fields,” in European conference on computer vision, pp. 363–380, Springer, 2024.

[32] Y. Wang, J. Liu, J. Wang, L. Yang, B. Dong, and Z. Li, “Haarfuse: A dual-branch infrared and visible light image fusion network based on haar wavelet transform,” Pattern Recognition, vol. 164, p. 111594, 2025.

[33] S. Li, Z. Liu, Z. Hong, Z. Zhou, and X. Cao, “Depfusion: Dual-domain enhancement and priority-guided mamba fusion for uav multispectral object detection,” arXiv preprint arXiv:2509.07327, 2025.

[34] G. Jocher and J. Qiu, “Ultralytics yolo11,” 2024.

[35] J. Han, J. Ding, J. Li, and G.-S. Xia, “Align deep features for oriented object detection,” IEEE transactions on geoscience and remote sensing, vol. 60, pp. 1–11, 2021.

[36] M. Yuan and X. Wei, “C<sup>2</sup>former: Calibrated and complementary transformer for rgb-infrared object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–12, 2024.

[37] X. He, C. Tang, X. Zou, and W. Zhang, “Multispectral object detection via cross-modal conflict-aware learning,” in Proceedings of the 31st ACM International Conference on Multimedia, pp. 1465–1474, 2023.

[38] C. Chen, J. Qi, X. Liu, K. Bin, R. Fu, X. Hu, and P. Zhong, “Weakly misalignment-free adaptive feature alignment for uavs-based multimodal object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26836–26845, 2024.

[39] T. Zhao, B. Liu, Y. Gao, Y. Sun, M. Yuan, and X. Wei, “Rethinking multi-modal object detection from the perspective of mono-modality feature learning,” in 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6364–6373, IEEE, 2025.

[40] Y. Zhang, J. Chen, J. Wang, D. Shi, S. Han, and L. Deng, “C2dffnet for object detection in multimodal remote sensing images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–16, 2025.

[41] C. Liu, X. Ma, X. Yang, Y. Zhang, and Y. Dong, “Como: Cross-mamba interaction and offset-guided fusion for multimodal object detection,” Information Fusion, vol. 125, p. 103414, 2026.

[42] H. Yang, J. Fang, Y. Zhu, X. Zhao, Y. Guo, X. Zhang, X. Hu, X. Yang, and Q. Ming, “Crossweaver: Towards efficient cross-modal interweaving and decoupling for weakly-aligned multispectral object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pp. 6361–6370, June 2026.

[43] J. Ding, N. Xue, Y. Long, G.-S. Xia, and Q. Lu, “Learning roi transformer for oriented object detection in aerial images,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2844–2853, IEEE, 2019.

[44] X. Xie, G. Cheng, J. Wang, X. Yao, and J. Han, “Oriented r-cnn for object detection,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3500–3509, IEEE, 2021.

[45] X. Yang, X. Yang, J. Yang, Q. Ming, W. Wang, Q. Tian, and J. Yan, “Learning high-precision bounding box for rotated object detection via kullback-leibler divergence,” Advances in Neural Information Processing Systems, vol. 34, pp. 18381–18394, 2021.

[46] N. Zhang, B. Chai, J. Song, T. Tian, P. Zhu, J. Ma, and J. Tian, “Omniscene infrared vehicle detection: An efficient selective aggregation approach and a unified benchmark,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 223, pp. 244–260, 2025.

[47] K. Song, X. Xue, H. Wen, Y. Ji, Y. Yan, and Q. Meng, “Misaligned visible-thermal object detection: A drone-based benchmark and baseline,” IEEE Transactions on Intelligent Vehicles, vol. 9, no. 11, pp. 7449– 7460, 2024.

[48] J. Han, J. Ding, N. Xue, and G.-S. Xia, “Redet: A rotation-equivariant detector for aerial object detection,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2785–2794, IEEE, 2021.

[49] Y. Zhang, H. Yu, Y. He, X. Wang, and W. Yang, “Illuminationguided rgbt object detection with inter-and intra-modality fusion,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–13, 2023.

[50] M. Sharma, M. Dhanaraj, S. Karnam, D. G. Chachlakis, R. Ptucha, P. P. Markopoulos, and E. Saber, “Yolors: Object detection in multimodal remote sensing imagery,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 1497–1508, 2020.