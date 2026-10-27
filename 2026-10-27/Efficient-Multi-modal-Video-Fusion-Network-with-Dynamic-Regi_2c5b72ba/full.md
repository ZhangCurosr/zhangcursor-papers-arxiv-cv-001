# Efficient Multi-modal Video Fusion Network with Dynamic Region-Guided Temporal Interaction

Jingxue Huang, Xiaosong Li, Yi Lin

Abstract—Multi-modal video fusion aims to generate temporally coherent videos with salient targets and rich structural details by integrating complementary information from heterogeneous sensors. However, existing video fusion methods commonly rely on optical flow estimation and feature warping for interframe alignment and cross-frame information aggregation to maintain temporal consistency, thereby introducing considerable computational overhead and making the fusion performance vulnerable to flow estimation errors. To address these limitations, this paper proposes an efficient multi-modal video fusion network with dynamic region-guided temporal interaction. Specifically, the proposed spatio-temporal interaction module leverages the Test-Time Training paradigm to model temporal dependencies with linear complexity. Building on this design, the module restricts temporal modeling to key dynamic regions in the video sequence, as static background regions provide limited temporal cues. By excluding low-variation tokens from unnecessary temporal interaction, this selective strategy contributes to improved fusion quality and temporal consistency. In addition, leveraging the strong ability of pretrained MAE features to capture global structural dependencies, we introduce a feature-level temporal consistency constraint to suppress local flickering by aligning the inter-frame variation directions of the fused and source videos. We further propose a new MAE-based temporal metric that quantifies directional distance between their temporal variations within dynamic regions. Experimental results demonstrate that, compared with existing state-of-the-art methods, our method achieves superior fusion quality and temporal consistency on multiple benchmark datasets, while exhibiting high computational efficiency under edge-device deployment settings.

Index Terms—Multi-modal video fusion, dynamic regions, sequence modeling, feature-level temporal consistency constraint, temporal metric.

## I. INTRODUCTION

D <sup>UE</sup> <sup>to</sup> <sup>the</sup> <sup>inherent</sup> <sup>limitations</sup> <sup>of</sup> <sup>modality-specific</sup> <sup>imag-</sup>ing mechanisms, a single modality is generally unable to ing mechanisms, a single modality is generally unable to provide a comprehensive representation of a scene [1]. Specifically, visible sensors provide rich texture, color, and structural details, which are beneficial for preserving the natural visual representation of scenes. However, their imaging quality is highly susceptible to illumination variations, nighttime conditions, smoke, and adverse weather. In contrast, infrared sensors capture thermal radiation from targets, offering stronger target perception under low-light and other degraded conditions. Nevertheless, infrared imaging often suffers from insufficient texture details and relatively low spatial resolution [2], [3].

Multi-modal video fusion (MMVF) jointly exploits complementary information from video sequences captured by heterogeneous sensors to generate unified and information-rich video representations [4]–[7]. This capability supports reliable scene understanding in complex environments, facilitating applications such as video surveillance, autonomous driving, low-altitude observation, and disaster detection [8]–[12].

Existing advanced video fusion methods commonly resort to optical-flow estimation and explicit feature warping to achieve cross-frame information modeling and rigid temporal alignment [13]. For example, UniVF [4] introduces such techniques into multi-frame learning to establish inter-frame relationships and achieves promising video fusion performance. However, as illustrated in Fig. 1, flow-based methods suffer from two fundamental bottlenecks. The first one lies in their low computational efficiency, while the second is their high sensitivity to estimation accuracy. Specifically, the fusion quality is strictly constrained by the accuracy and stability of optical-flow estimation. In complex dynamic scenarios, optical flow is often difficult to estimate reliably, and the accumulated errors over long video sequences can be further propagated into the fusion method. In addition, if optical flow is discarded and global self-attention-based Transformers are adopted for direct spatiotemporal information aggregation, even heavier computational overhead will be introduced. Because establishing correlations along any dimension with conventional global attention incurs quadratic complexity, the computational cost grows prohibitively with the sequence length, making it difficult to scale to long video sequences. These issues severely hinder the practical deployment of MMVF on resource-constrained edge devices, where real-time inference and low power consumption are typically required.

To address these limitations, this work aims to explore a lightweight inter-frame relation modeling strategy that efficiently models temporal dependencies without relying on explicit optical-flow alignment or dense global spatio-temporal attention. Recently, Test-Time Training (TTT) [14] has shown strong potential for efficient visual sequence modeling. By reformulating token interaction as an online learning process over a compact inner model, TTT avoids explicitly constructing pairwise attention maps and enables sequence modeling with linear computational complexity. Although existing visual TTT architectures have demonstrated the effectiveness of TTT in modeling intra-frame spatial context, the use of TTT-based sequence modeling for efficient cross-frame interaction in MMVF remains underexplored.

In this paper, we propose an efficient MMVF network equipped with a spatio-temporal interaction module (STIM) for lightweight temporal dynamic relation modeling. Specifically, STIM employs TTT-based sequence modeling along the temporal dimension to efficiently capture cross-frame dependencies. Nevertheless, uniformly applying temporal interaction to all inter-frame features would still introduce substantial redundant feature interactions, as different spatial regions provide unequal temporal cues. Most static background regions exhibit only minor variations across adjacent frames, and modeling these regions may provide limited temporal benefit while propagating unnecessary noise. In contrast, moving targets, regions with changing boundaries, and areas exhibiting distinct cross-modal response differences are more likely to cause temporal discontinuities in the fused video. Therefore, as shown in Fig. 1, we introduce a dynamic region guidance mechanism into STIM, allowing it to gather the dynamic features from each frame, reorganize them into a sequence of dynamic features, and perform TTT-based temporal interaction over the sequence. In this way, each selected feature can more effectively aggregate inter-frame context from the video clip, enabling sparse temporal modeling focused on dynamically informative regions. In addition, to further enhance the temporal stability of fused videos, we introduce a featurelevel temporal consistency constraint based on pretrained MAE representations. Benefiting from their strong ability to capture global structural dependencies, MAE features provide a structure-aware representation space in which patch-level structural and regional variations between frames can be effectively characterized. We therefore align the adjacent-frame feature variations of the fused video with those of the source videos. The main contributions of this work are summarized as follows:

![](images/e600ef0abb5bd03fe03ac67e7eb63e61028781eab3373357e2b369395ef0a836.jpg)  
Fig. 1. Comparison with an existing video fusion method [4] in terms of temporal interaction schemes, fusion quality and computational efficiency. Al reported metrics are evaluated on the VTMOT dataset.

\- We propose an efficient MMVF framework that employs linear-complexity TTT-based sequence modeling for temporal interaction and dynamic-region guidance to further focus inter-frame modeling on dynamically informative content.

\- We propose a feature-level temporal consistency constraint that exploits MAE representations to align the temporal variations of fused videos with those of source videos, thereby suppressing inter-frame flickering in MMVF.

\- We introduce a new feature-level temporal metric, termed Feature-Level Dynamic Direction Distance (FD3), to evaluate temporal consistency from the perspective of region-level dynamic variations in the feature space.

\- Extensive experiments on multiple benchmark datasets demonstrate that our method achieves superior fusion quality. We further validate that the proposed method exhibits higher computational efficiency than the compared methods when deployed on the Jetson AGX Orin platform.

## II. RELATED WORK

## A. Multi-modal Image Fusion

With the rapid advancement of deep learning, multi-modal image fusion (MMIF) has progressively evolved from conventional paradigms relying on handcrafted rules and traditional image-processing operators [15], [16] toward datadriven learning paradigms centered on neural network architectures [17]–[19], leading to substantial improvements in overall fusion quality. According to the underlying network architectures [20], existing MMIF methods can be broadly categorized into convolutional neural network (CNN)-based [21]– [23], Transformer-based [24]–[26], generative adversarial network (GAN)-based [27]–[29], diffusion model-based [11], [30], [31], Mamba-based [32]–[34], and hybrid-architecture methods [7], [35], [36]. For CNN-based methods, Li et al. [21] proposed LRRNet, which unfolds the optimization process of low-rank representation into a lightweight feed-forward convolutional network, thereby reducing reliance on empirical architecture design. For Transformer-based methods, Wu et al. [25] developed FCFormer based on a generalized fullyconnected self-attention mechanism. It exploits both local and non-local correlations among multiple source images and provides a unified framework for different image fusion tasks. For GAN-based methods, Huang et al. [27] introduced an end-to-end framework for trimodal medical image fusion. The method mainly incorporates primitive relationship reasoning to enhance global semantic information. For diffusionbased methods, Shi et al. [30] proposed a degradation-aware diffusion framework for MMIF under arbitrary degradation conditions. The model directly regresses the fused image for implicit denoising and introduces a joint observation-model correction mechanism to enforce both degradation and fusion constraints during sampling. For Mamba-based methods, Cao et al. [32] proposed Shuffle Mamba, which adopts random and inverse shuffle operations to alleviate the scanning bias caused by fixed token-ordering strategies. The framework further performs modality-aware representation and cross-modal interaction along both spatial and channel dimensions. For hybrid architectures, Xia et al. [36] revisited the integration of CNNs and Transformers from a frequency perspective, aiming to better coordinate the local-detail modeling capability of CNNs with the global-context modeling capability of Transformers.

## B. Multi-modal Video Fusion

To address the temporal instability of frame-wise MMIF in dynamic scenes, recent studies have shifted their attention to MMVF. Gong et al. [5] proposed an infrared–visible video fusion framework, which incorporates temporal cooperative modules and targeted guidance to achieve superior fusion results. Tang et al. [1] proposed a unified multi-modal video fusion and restoration framework with bi-temporal co-attention, which exploits temporal and cross-modal complementary cues for information restoration and fusion. Although these methods improve temporal coherence through strong spatio-temporal modeling, heavy multi-frame aggregation may over-couple features under complex motion or large temporal variations, causing detail loss and inter-frame artifacts. Zhao et al. [4] introduced a unified video fusion framework, employing an additional optical flow network to estimate inter-frame pixel displacement and perform warping. Although these MMVF methods achieve superior fusion quality, their complex temporal interaction or warping mechanisms often incur substantial computational overhead as well as high inference latency, limiting real-time deployment on resource-constrained edge devices. In this paper, the proposed method employs TTTbased sequence modeling along the temporal dimension and performs inter-frame interaction within dynamic regions, without relying on explicit optical-flow alignment or heavy multiframe aggregation.

## C. Efficient Visual Sequence Modeling

The Vision Transformer (ViT) architecture [37] has been widely adopted in image fusion with remarkable performance. However, standard Softmax attention explicitly computes pairwise interactions between queries and keys, resulting in a computational complexity of $\mathcal { O } ( N ^ { 2 } )$ with respect to the sequence length. Consequently, its computational cost increases substantially when processing long visual sequences. To alleviate this computational bottleneck, linear attention has emerged as an alternative owing to its $\mathcal { O } ( N )$ complexity [38]. Specifically, it replaces the Softmax operation with a linear kernel mapping and reorganizes the computation order from $( Q K ^ { \top } ) V \ \mathrm { \bar { t o } } \ Q ( K ^ { \top } V )$ , thereby avoiding the construction of a complete attention matrix. Despite its considerable efficiency gains, aggregating the key–value information into a shared context $K ^ { \top } V$ restricts nonlinear relation modeling and finegrained query–key interactions, leading to limited representational capacity [39], [40]. To better balance computational efficiency and representation capability, Test-Time Training reformulates attention as an online learning process, in which a compact nonlinear internal model is updated at inference time using key–value pairs, thereby enabling adaptive token– context interactions with $\mathcal { O } ( N )$ complexity [41].

## III. METHODOLOGY

## A. Overall Framework

Given paired multi-modal video sequences $S ^ { 1 } , S ^ { 2 } \in$ $\mathbb { R } ^ { B \times T \times 3 \times H \times W }$ , where B, T, H, and W denote the batch size, number of frames, height, and width, respectively, the proposed method aims to generate high-quality fused results $S ^ { F }$ with high computational efficiency. As illustrated in Fig. 2, the overall framework is organized into three stages.

First, in the encoder, the input sequences are sequentially fed into convolutional layers and Squeeze-and-Excitation (SE) Blocks [42] to perform shallow feature extraction and mutual feature enhancement. Since convolutional operations are inherently limited by their local receptive fields, modeling global contextual dependencies is essential for preserving spatial structures. Therefore, the enhanced features are further delivered to VITTT [41] to capture intra-frame long-range dependencies. Compared with conventional vision Transformers with quadratic self-attention complexity, VITTT enables global context modeling with linear computational complexity, making it well suited for efficient MMVF. Subsequently, the extracted features are fed into the proposed STIM to establish inter-frame temporal dependencies.

Second, in the fusion stage, the features extracted by VITTT and STIM are fused through TokenFusion, where a lightweight fully connected layer is employed to integrate multi-modal features. Specifically, the outputs of the first and third VITTT blocks, together with those of the second and third STIM blocks, are selected as inputs to TokenFusion.

Finally, in the decoder, the fused features are progressively upsampled and reconstructed. Meanwhile, the enhanced shallow features extracted by the SE Blocks are integrated and introduced into the final head through a residual connection to facilitate the final reconstruction, producing the fused sequences.

![](images/c6454fe2d9902ba8d9a84cb077855b1d33d9f0b16a306968dad9777533cda538.jpg)  
Fig. 2. Overview of the proposed multi-modal video fusion framework.

## B. Spatio-Temporal Interaction Module

Modeling temporal dependencies across consecutive frames is essential for maintaining temporal consistency in fused videos. To this end, we propose the STIM, which consists of two key components: dynamic region estimation and efficient spatio-temporal interaction. Specifically, a dynamic region map is generated for each frame to identify regions that are most relevant to temporal variation. In addition, temporal interactions are performed within these dynamic regions. Since both components are subsequently implemented at the patch level, each frame is divided into N non-overlapping patches of size $p \times p ,$ where $h _ { p } = H / p , w _ { p } = W / p ,$ and $N = h _ { p } w _ { p } .$ Dynamic Region Estimation. Given a pair of multi-modal video sequences $S ^ { 1 }$ and $S ^ { 2 }$ , we first compute the temporal mean representation of each modality to characterize temporal reference, as follows:

$$
\bar { S } ^ { m } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } S _ { t } ^ { m } , \qquad m \in \{ 1 , 2 \} .\tag{1}
$$

Based on the predefined patch partition, the dynamic response of the n-th patch is calculated by averaging the pixel-

wise deviations of the t-th frame from the temporal mean:

$$
R _ { t , n } ^ { m } = \frac { 1 } { p ^ { 2 } } \sum _ { ( u , v ) \in \Omega _ { n } } \left| S _ { t } ^ { m } ( u , v ) - \bar { S } ^ { m } ( u , v ) \right| ,\tag{2}
$$

where $\Omega _ { n }$ denotes the set of spatial locations within the n-th patch, and $( u , v )$ denotes a pixel coordinate in $\Omega _ { n }$

The dynamic responses of the two sequences are combined as:

$$
R _ { t , n } = R _ { t , n } ^ { 1 } + R _ { t , n } ^ { 2 } .\tag{3}
$$

Finally, the dynamic regions $\mathcal { D } _ { t }$ are identified by selecting the Top-K patches of frame t according to their dynamic scores:

$$
\mathcal { D } _ { t } = \mathrm { T o p } \cdot K _ { n } \left( R _ { t , n } ; K \right) .\tag{4}
$$

where $K ~ = ~ \mathrm { m a x } \left( 1 , \mathrm { r o u n d } ( \rho N ) \right)$ denotes the number of selected patches, and $\rho$ is a hyperparameter controlling the proportion of selected dynamic patches.

Spatio-Temporal Interaction. Given the patch embeddings $\dot { Z } \in \mathbb { R } ^ { B \times T \times N \times D }$ output by the VITTT block, STIM directly operates on these token representations, where D denotes the embedding dimension.

As shown in Fig. 2, STIM consists of two parallel branches: a sparse temporal interaction branch and a dense spatial refinement branch. Guided by D, the sparse branch performs spatio-temporal interaction at the selected patch locations, while the dense branch performs local spatial refinement over the full grid.

Before describing the two branches, we first introduce the TTT-based formulation for inner-model adaptation. Let $f _ { \theta }$ denote an arbitrary inner model parameterized by the learnable initial fast weights θ. Given M key–value pairs, the fast weights are adapted once using the dot-product adaptation objective:

$$
\mathcal { L } _ { \mathrm { a d a p t } } ( \theta ; K , V ) = - \frac { \gamma } { M } \langle f _ { \theta } ( K ) , V \rangle _ { F } ,\tag{5}
$$

where $\langle \cdot , \cdot \rangle _ { F }$ denotes the Frobenius inner product. The corresponding one-step fast-weight update is defined as:

$$
\begin{array} { r } { \theta ^ { * } = \theta - \nabla _ { \theta } \mathcal { L } _ { \mathrm { a d a p t } } ( \theta ; K , V ) , } \end{array}\tag{6}
$$

where $\theta ^ { * }$ denotes the adapted fast weights obtained after an inner-loop update.

Sparse temporal interaction branch. The selected patch embeddings are first gathered as:

$$
Z _ { s } = \mathrm { G a t h e r } \left( Z , \mathcal { D } \right) , \qquad Z _ { s } \in \mathbb { R } ^ { B \times T \times K \times D } .\tag{7}
$$

To preserve the frame identity of the selected patch embeddings during subsequent spatio-temporal interaction, a sinusoidal temporal positional encoding is added:

$$
Z _ { s } ^ { \mathrm { p o s } } = Z _ { s } + P ,\tag{8}
$$

where $P \in \mathbb { R } ^ { 1 \times T \times 1 \times D }$ denotes the frame-wise temporal positional encoding. The temporal and selected-patch dimensions are then merged to form a token sequence $\dot { Z _ { s } ^ { \mathrm { s e q } } } \in \mathbb { R } ^ { B \times T K \times D }$ concatenated along the temporal dimension.

Queries, keys, and values are generated from the spatiotemporal sequence through learnable linear projections:

$$
Q _ { s } = Z _ { s } ^ { \mathrm { s e q } } W _ { Q _ { s } } , \qquad K _ { s } = Z _ { s } ^ { \mathrm { s e q } } W _ { K _ { s } } , \qquad V _ { s } = Z _ { s } ^ { \mathrm { s e q } } W _ { V _ { s } } ,\tag{9}
$$

where $Q _ { s } , K _ { s } , V _ { s } \in \mathbb { R } ^ { B \times T K \times D }$

For the sparse temporal interaction branch, the inner model is implemented as a simplified SwiGLU parameterized by $\theta _ { s } = \{ W _ { 1 } , W _ { 2 } \}$ . For an arbitrary input $U ,$ it is defined as:

$$
f _ { \theta _ { s } } ( U ) = ( U W _ { 1 } ) \odot \mathrm { S i L U } ( U W _ { 2 } ) ,\tag{10}
$$

where ⊙ denotes element-wise multiplication. Using $K _ { s }$ and $V _ { s }$ as the key–value inputs and substituting $\theta = \theta _ { s } , K = K _ { s } ,$ $V = V _ { s } ,$ , and $M = T K$ into Eq. (5) defines the adaptation objective for this branch. Applying the one-step update in Eq. (6) yields the adapted parameter set $\theta _ { s } ^ { * } = \{ W _ { 1 } ^ { * } , W _ { 2 } ^ { * } \}$

The adapted inner model is then applied to the queries to produce the output of this branch:

$$
O _ { s } ^ { \mathrm { s e q } } = f _ { \theta _ { s } ^ { * } } ( Q _ { s } ) = ( Q _ { s } W _ { 1 } ^ { * } ) \odot \mathrm { S i L U } ( Q _ { s } W _ { 2 } ^ { * } ) .\tag{11}
$$

After restoring the temporal and selected-patch dimensions, the interacted embeddings are scattered back to their original spatial locations according to D, where $O _ { s } \in \mathbb { R } ^ { B \times T \times \overline { { K } } \times D }$ denotes the dimension-restored form of $O _ { s } ^ { \mathrm { s e q } } \in \mathbb { R } ^ { B \times T K \times D }$

$$
\widetilde { Z } _ { s } = \mathrm { S c a t t e r } \left( { O } _ { s } , { \mathcal { D } } ; N \right) , \qquad \widetilde { Z } _ { s } \in \mathbb { R } ^ { B \times T \times N \times D } .\tag{12}
$$

Dense spatial refinement branch. In parallel with the sparse temporal interaction, the dense branch operates on the full grid and adapts the convolutional inner-model design for local spatial refinement [41]. The complete patch embeddings are projected into queries, keys, and values, which are subsequently rearranged according to their original patch locations as $Q _ { l } , \bar { K _ { l } } , \bar { V _ { l } } \in \bar { \mathbb { R } } ^ { B T \times C \times h _ { p } \times w _ { p } }$ , where the batch and temporal dimensions are merged and C denotes the channel dimension.

The convolutional inner model is parameterized by the learnable initial fast weights $\theta _ { l } ~ = ~ \{ W _ { 3 } \}$ and implemented using a $3 \times 3$ depth-wise convolution:

$$
f _ { \theta _ { l } } ( U ) = \mathrm { D W C o n v } _ { 3 \times 3 } ( U ; W _ { 3 } ) .\tag{13}
$$

The full grid key–value features $K _ { l }$ and $V _ { l }$ are used to instantiate the adaptation formulation in Eqs. (5) and (6), with $\theta = \theta _ { l } , K = K _ { l } , V = V _ { l }$ , and $M = N$ . This produces the adapted parameter set $\theta _ { l } ^ { * } = \{ W _ { 3 } ^ { * } \}$

The adapted inner model is then applied to the full grid queries to obtain the locally refined features:

$$
O _ { l } = f _ { \theta _ { l } ^ { * } } ( Q _ { l } ) = \mathrm { D W C o n v } _ { 3 \times 3 } ( Q _ { l } ; W _ { 3 } ^ { * } ) ,\tag{14}
$$

where $O _ { l } ~ \in ~ \mathbb { R } ^ { B T \times C \times h _ { p } \times w _ { p } }$ . The resulting feature maps are finally rearranged back into token form, yielding $\tilde { Z } _ { l } \in$ $\mathbb { R } ^ { B \times T \times N \times D }$ . Finally, the two branch outputs, $\widetilde { Z } _ { l }$ and $\widetilde { Z } _ { s }$ , are linearly fused.

## C. Loss Function

Pixel-level Supervision. To preserve frame-wise intensity and structural information, we adopt two pixel-level constraints: an intensity loss and a gradient loss. Accordingly, the pixel-level objective is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p i x e l } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { i n t } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { g r a d } } , } \end{array}\tag{15}
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are hyperparameters. Then, the intensity and gradient terms are respectively formulated as:

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } _ { \mathrm { i n t } } = \frac { 1 } { T H W } \sum _ { t = 1 } ^ { T } \left. S _ { t } ^ { F } - \operatorname* { m a x } \left( S _ { t } ^ { 1 } , S _ { t } ^ { 2 } \right) \right. _ { 1 } , } \\ & { \displaystyle \mathcal { L } _ { \mathrm { g r a d } } = \frac { 1 } { T H W } \sum _ { t = 1 } ^ { T } \left. \left. \nabla S _ { t } ^ { F } \right. - \operatorname* { m a x } \left( \left. \nabla S _ { t } ^ { 1 } \right. , \left. \nabla S _ { t } ^ { 2 } \right. \right) \right. _ { 1 } . } \end{array}\tag{16}
$$

Here, $\operatorname* { m a x } ( \cdot , \cdot )$ denotes element-wise maximum selection, and ∇ denotes the Sobel gradient operator.

Feature-level Supervision. Although pretrained MAE representations [43], [44] have been increasingly adopted as visual priors, their patch-level contextual modeling capability remains underexplored for supervising temporal transitions in MMVF. Masked image modeling [45] encourages a model to recover masked image regions from their visible context, enabling the learned representations to capture contextual dependencies among image patches. Consequently, features extracted by a pretrained MAE encoder are responsive to variations in structural patterns, object boundaries, and regional content. When applied to video sequences, the temporal variations of these features can effectively characterize the structural- and region-level dynamic evolution of content.

Based on this property, we employ a frozen MAE encoder as a shared representation space for feature-level temporal supervision. Given the fused sequence and the two source sequences, $S ^ { F } , S ^ { 1 }$ , and $S ^ { 2 }$ , the MAE feature of each frame is extracted as:

$$
E _ { t } ^ { m } = \Phi ( S _ { t } ^ { m } ) , \quad m \in \{ F , 1 , 2 \} ,\tag{17}
$$

where Φ denotes the frozen MAE encoder, and $E _ { t } ^ { F } , E _ { t } ^ { 1 }$ , and $E _ { t } ^ { 2 }$ represent the features of the fused and the two source sequences at frame t, respectively.

To characterize inter-frame feature variations $\dot { E } _ { t } ^ { m }$ , the formulation is given as follows:

$$
\dot { E } _ { t } ^ { m } = E _ { t + 1 } ^ { m } - E _ { t } ^ { m } , \qquad 1 \leq t < T , \quad m \in \{ F , 1 , 2 \} .\tag{18}
$$

Then, we define a temporal consistency loss to align the temporal transition directions of the fused sequence with those of both source sequences between consecutive frames:

$$
\mathcal { L } _ { \mathrm { t e m p } } = \frac { 1 } { 2 ( T - 1 ) } \sum _ { t = 1 } ^ { T - 1 } \sum _ { m = 1 } ^ { 2 } \left[ 1 - \cos \left( \dot { E } _ { t } ^ { F } , \dot { E } _ { t } ^ { m } \right) \right] .\tag{19}
$$

where $\cos ( \cdot , \cdot )$ denotes cosine similarity between the corresponding feature variations.

In addition, we impose a time-reversal consistency constraint on the fused sequence as a self-consistency regularizer to reduce the model’s dependence on a specific temporal processing direction. Specifically, the reversed input sequences $S _ { \mathrm { r e } } ^ { 1 }$ and $S _ { \mathrm { r e } } ^ { 2 }$ are fused to obtain $S _ { \mathrm { r e } } ^ { F } .$ . After restoring it to the original temporal order, its feature variations are expected to be consistent with those of the original fusion result $S ^ { F }$ . By applying Eqs. (17) and (18), we obtain its patch-level feature $E ^ { \mathrm { r e } }$ and the corresponding adjacent-frame temporal feature variations $\dot { E } _ { t } ^ { \mathrm { r e } }$ . This constraint suppresses temporal-direction bias and promotes more stable temporal modeling. The timereversal consistency loss is then defined as:

$$
\mathcal { L } _ { \mathrm { r e v } } = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \left[ 1 - \cos \left( \dot { E } _ { t } ^ { F } , \dot { E } _ { t } ^ { \mathrm { r e } } \right) \right] .\tag{20}
$$

Beyond temporal supervision, the frozen MAE features are also used to provide structural and semantic guidance over the intermediate features of the network. Specifically, for each modality, the output features of all three ViTTT blocks are aligned with the final output features of MAE. Let $H _ { m } ^ { l }$ denote the output feature of the l-th ViTTT block for modality $m \in \{ 1 , 2 \}$ , where $l \in \{ 1 , 2 , 3 \}$ . Before feature alignment, a modality- and layer-specific projection head maps $H _ { m } ^ { l }$ to the MAE feature dimension, yielding $\widehat { H } _ { m } ^ { l }$ . The distillation loss is defined as:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \frac { 1 } { 6 T } \sum _ { t = 1 } ^ { T } \sum _ { m = 1 } ^ { 2 } \sum _ { l = 1 } ^ { 3 } \left[ 1 - \cos \left( \widehat { H } _ { t , m } ^ { l } , E _ { t } ^ { m } \right) \right] .\tag{21}
$$

Therefore, the feature-level objective is defined as:

$$
\mathcal { L } _ { \mathrm { f e a t } } = \lambda _ { 3 } \mathcal { L } _ { \mathrm { t e m p } } + \lambda _ { 4 } \mathcal { L } _ { \mathrm { r e v } } + \lambda _ { 5 } \mathcal { L } _ { \mathrm { d i s t i l l } } ,\tag{22}
$$

where $\lambda _ { 3 } , \lambda _ { 4 }$ and $\lambda _ { 5 }$ are hyperparameters.

Overall, the final training loss of the proposed model is as follows:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { p i x e l } } + \mathcal { L } _ { \mathrm { f e a t } }\tag{23}
$$

## IV. EXPERIMENTS

## A. Experiment Settings

Datasets. The proposed model is trained on three datasets: HDO [46], M3SVD [1], and VTUAV [47]. For HDO, 22 sequences are used for training, with an average of approximately 240 frames per sequence, while the remaining five sequences are used for testing. For M3SVD, 50 sequences are used for training, with an average of approximately 200 frames per sequence, and eight sequences are selected from the remaining data for testing. For VTUAV, 38 sequences are used for training, with an average of approximately 100 frames per sequence, and eight sequences are selected from the remaining data for testing. To evaluate the generalization ability of the proposed method, eight sequences from the VTMOT dataset [4] are additionally selected for testing.

Training Details. All experiments are implemented in Py-Torch and conducted on a server equipped with two NVIDIA GeForce RTX 4090 GPUs. The model is trained for a total of $5 0 { , } 0 0 0$ iterations using a two-stage progressive training strategy. During the first 25,000 iterations, five-frame video clips are randomly cropped to $2 5 6 \times 2 5 6$ pixels for training, with a batch size of five. During the remaining 25,000 iterations, the spatial resolution and temporal length are increased to $3 8 4 \times 3 8 4$ pixels and seven frames, respectively, while the batch size is reduced to three. AdamW is adopted as the optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ . The patch size is set to $1 6 \times 1 6$ pixels, and the ρ for dynamic-region selection is set to 0.25. The fixed scaling factor γ is set to $9 ^ { - { \frac { 1 } { 2 } } }$ . The loss weights are empirically set to $\lambda _ { 1 } = 5 , \lambda _ { 2 } = 1 0$ $\lambda _ { 3 } = 2 , \lambda _ { 4 } = 2 ,$ , and $\lambda _ { 5 } = 1 0$

Comparative Methods. In this paper, we select seven stateof-the-art fusion methods for comparison, including three video fusion methods (VideoFusion [1], TemCoCo [5], and UniVF [4]) and four image fusion methods (DSPFusion [48], ISFusion [49], GIFNet [50], and CoCoNet [51]). For fair comparison, all competing methods are evaluated using the parameter settings reported in their original papers.

Evaluation Metrics. At the image level, we employ $\mathrm { Q _ { G } }$ $\mathrm { Q _ { M } }$ , FSIM, and VIF to evaluate fusion quality from multiple perspectives [1], [52], [53]. Specifically, $\mathrm { Q } _ { \mathrm { G } }$ measures the preservation of edge strength and orientation from the source images. $\mathrm { Q _ { M } }$ evaluates the retention of multi-scale edge and high-frequency information. FSIM assesses the similarity of salient local features and structure. VIF evaluates the fidelity of fusion results in accordance with human visual perception.

For temporal quality assessment, flowD and feaCD are adopted as existing evaluation metrics [5]. flowD quantifies frame-to-frame temporal alignment errors through opticalflow-based frame prediction. feaCD evaluates the directional consistency of inter-frame variations between the fused and source sequences over the entire frame by mapping each frame into the ResNet-18 output space. In addition, we introduce a new metric (FD3), which quantifies the dynamic direction distance between the inter-frame feature variations of the fused sequence and those of each source sequence within local dynamic regions in the pretrained MAE feature space.

![](images/819a99b54ef4ce9ddf4a097e42903f8401696d4fa787624c4d42403fdd39c603.jpg)  
Fig. 3. Qualitative comparison of all methods on the M3SVD, VTUAV, HDO and VTMOT datasets.

Following the feature-level supervision introduced in Sec. III-C, a frozen MAE encoder is employed to extract spatially aligned patch features from each frame. Let

$$
\mathbf { E } _ { t } ^ { m } = \left[ \mathbf { e } _ { t , 1 } ^ { m } , \mathbf { e } _ { t , 2 } ^ { m } , \ldots , \mathbf { e } _ { t , N } ^ { m } \right] ^ { \top } \in \mathbb { R } ^ { N \times D } , \qquad m \in \{ F , 1 , 2 \} ,\tag{24}
$$

where $\mathbf { e } _ { t , n } ^ { m } \in \mathbb { R } ^ { D }$ denotes the MAE patch feature corresponding to the n-th spatial region of sequence m at frame t. For each fixed spatial region, we first compute its temporal mean feature and then derive the feature deviation from this mean:

$$
\overline { { \mathbf { e } } } _ { n } ^ { m } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { e } _ { t , n } ^ { m } , \qquad \Delta \mathbf { e } _ { t , n } ^ { m } = \mathbf { e } _ { t , n } ^ { m } - \overline { { \mathbf { e } } } _ { n } ^ { m } .\tag{25}
$$

The resulting variations are weak in static regions but pronounced in regions containing moving targets, evolving boundaries, and temporally varying content. Accordingly, FD3 is defined as

$$
\mathrm { F D 3 } = \frac { 1 } { T N } \sum _ { t = 1 } ^ { T } \sum _ { n = 1 } ^ { N } \sum _ { m \in \{ 1 , 2 \} } \left[ 1 - \cos \left( \Delta \mathbf { e } _ { t , n } ^ { F } , \Delta \mathbf { e } _ { t , n } ^ { m } \right) \right] .\tag{26}
$$

A lower FD3 value indicates smaller directional discrepancies between the region-level feature variations of the fused sequence and those of the two source sequences.

## B. Multi-modal Video Fusion Experiment

Qualitative Comparison. As shown in Fig. 3, we present four representative scenes from different datasets. In the first case, different fusion methods are compared in terms of their ability to depict camouflaged persons in a jungle environment. The results of CoCoNet, VideoFusion, and TemCoCo tend to excessively preserve the salient information from infrared images, leading to the loss of visible details. In contrast, the proposed method achieves a better balance between crossmodal information. It not only highlights camouflaged persons in jungle scenes with dense vegetation and complex background textures, but also preserves fine visible details, such as clothing wrinkles and hats. Similarly, in the second case, under sufficient illumination conditions, DSPFusion, Co-CoNet, and VideoFusion fail to achieve desirable information balance, resulting in degraded vehicle details. Our method moderately enhances the saliency of vehicles on the road while preserving their original structural details. In the third case, obvious artifacts can be observed in the results of VideoFusion and TemCoCo. These artifacts mainly arise from abnormal residual contents propagated from adjacent frames, indicating the issue of cross-frame information leakage. The proposed method effectively integrates and decouples temporal information, and achieves better contrast enhancement and structural information preservation under insufficient illumination. In the last case, the enlarged regions show that, compared with other methods, the proposed method exhibits stronger detail preservation capability for weak-texture regions, such as zebra crossings on the road and traffic signs. Overall, from the perspective of visual evaluation, our method achieves better performance in terms of contrast, structural information preservation, and cross-modal information balance.

Quantitative Comparison. Table I reports the quantitative comparison between the proposed method and seven stateof-the-art methods on multiple objective metrics. Overall, the proposed method achieves clear advantages on most metrics. For image quality evaluation, all results of our method rank first except for FSIM on the M3SVD dataset and $\mathrm { Q _ { G } }$ on the VTMOT dataset, where our method ranks second. The higher $\mathrm { Q } _ { \mathrm { G } }$ and $\mathrm { Q _ { M } }$ values indicate that the proposed method preserves richer textures and detailed information. The advantage in FSIM demonstrates that the fused results maintain strong feature-level consistency with the source sequences. Moreover, the best VIF values suggest that the fused results provide better visual fidelity in terms of perceptually relevant information, such as contrast, color and appearance details.

For temporal evaluation, our method ranks first on all datasets in terms of feaCD and FD3, indicating that it can better maintain source-consistent temporal variation directions both over the global representation and within the local dynamic regions. For flowD, although the proposed method does not achieve the best results on all datasets, it consistently ranks among the top three. Specifically, it obtains the second-best result on the M3SVD dataset, only behind UniVF, and ranks third on the other three datasets, following VideoFusion and TemCoCo. These results demonstrate the strong competitiveness of our method in matching the motion continuity of the source sequences. VideoFusion and TemCoCo achieve lower flowD values, which may partly benefit from over-smoothed fused results. In Sec. IV-E, we further analyze the sensitivity of different temporal metrics to temporal smoothing to provide a more comprehensive discussion.

TABLE I  
QUANTITATIVE COMPARISONS ACROSS THE M3SVD, VTUAV, HDO, AND VTMOT DATASETS. THE BEST, SECOND-BEST, AND THIRD-BEST RESULTS ARE DENOTED BY BESTCOLOR , SECONDCOLOR , AND THIRDCOLOR , RESPECTIVELY.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Pub.</td><td rowspan="2">Method</td><td colspan="2">QG ↑ QM↑</td><td>VIF↑</td><td>FSIM↑ feaCD ↓</td><td>flowD↓</td><td>FD3↓</td><td></td><td colspan="2">QG↑ QM↑</td><td>VIF↑</td><td>FSIM ↑ feaCD↓</td><td>flowD↓ Dataset</td><td></td><td>FD3↓</td></tr><tr><td colspan="3"></td><td colspan="2">M3SVD Dataset</td><td colspan="2"></td><td colspan="7">VTUAV</td></tr><tr><td rowspan="6">Image</td><td>TIP 26</td><td>DSPFusion</td><td>0.61</td><td>0.58</td><td>0.31</td><td>0.782</td><td>1.76</td><td>8.72</td><td>1.15</td><td>0.57</td><td>0.93</td><td>0.34</td><td>0.793</td><td>1.73</td><td>12.51 9.35</td><td>1.09</td></tr><tr><td>CVPR 26</td><td>ISFusion</td><td>0.51</td><td>0.35</td><td>0.25</td><td>0.777</td><td>1.69</td><td>6.96</td><td>1.12</td><td>0.48</td><td>0.74</td><td>0.33</td><td>0.808</td><td>1.77</td><td></td><td>1.05</td></tr><tr><td>CVPR 25</td><td>GIFNet</td><td>0.47</td><td>0.25</td><td>0.24</td><td>0.776</td><td>1.78</td><td>6.32</td><td>1.14</td><td>0.39</td><td>0.55</td><td>0.27</td><td>0.804</td><td>1.81</td><td>10.23</td><td>1.09</td></tr><tr><td>IJCV 24</td><td>CoCoNet</td><td>0.35</td><td>0.19</td><td>0.18</td><td>0.697</td><td>1.76</td><td>11.15</td><td>1.30</td><td>0.33</td><td>0.41</td><td>0.21</td><td>0.739</td><td>1.83</td><td>17.47</td><td>1.18</td></tr><tr><td>CVPR 26</td><td>VideoFusion</td><td>0.36</td><td>0.30</td><td>0.30</td><td>0.802</td><td>1.70</td><td>5.81</td><td>1.33</td><td>0.29</td><td>0.49</td><td>0.30</td><td>0.811</td><td>1.94</td><td>7.95</td><td>1.36</td></tr><tr><td>ICCV 25</td><td>TemCoCo</td><td>0.42</td><td>0.26</td><td>0.28</td><td>0.788</td><td>1.88</td><td>8.22</td><td>1.35</td><td>0.32</td><td>0.50</td><td>0.25</td><td>0.808</td><td>1.97</td><td>6.70</td><td>1.37</td></tr><tr><td rowspan="6">Video</td><td rowspan="3">NeurIPS 25</td><td>UniVF</td><td>0.63</td><td>0.62</td><td>0.37</td><td>0.796</td><td>1.88</td><td>3.46</td><td>1.13</td><td>0.57</td><td>0.89</td><td>0.39</td><td>0.813</td><td>1.73</td><td>8.72</td><td>1.14</td></tr><tr><td></td><td>0.63</td><td>0.82</td><td>0.39</td><td>0.798</td><td>1.66</td><td>5.68</td><td>1.06</td><td>0.61</td><td>1.26</td><td>0.41</td><td>0.813</td><td>1.61</td><td>8.07</td><td>1.04</td></tr><tr><td colspan="3">HDO Dataset</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>VTMOT Dataset</td><td></td><td></td><td></td></tr><tr><td colspan="2">DSPFusion</td><td>0.53</td><td>0.52</td><td>0.26</td><td>0.775</td><td>1.76</td><td>14.47</td><td>1.13</td><td>0.54</td><td>0.58</td><td>0.33</td><td>0.804</td><td>1.70</td><td>19.66</td><td>1.05</td></tr><tr><td>CVPR 26</td><td>ISFusion</td><td>0.46</td><td>0.36</td><td>0.24</td><td>0.772</td><td>1.77</td><td>11.38</td><td>1.03</td><td>0.42</td><td>0.37</td><td>0.27</td><td>0.774</td><td>1.77</td><td>15.00</td><td>1.03</td></tr><tr><td>CVPR 25 IJCV 24</td><td>GIFNet</td><td>0.32</td><td>0.23</td><td>0.19</td><td>0.775</td><td>1.83</td><td>11.80</td><td>1.25</td><td>0.31</td><td>0.23</td><td>0.21</td><td>0.783</td><td>1.85</td><td>15.84</td><td>1.13</td></tr><tr><td rowspan="4">Video</td><td>CVPR 26</td><td>CoCoNet</td><td>0.28</td><td>0.20</td><td>0.17</td><td>0.722</td><td>1.84</td><td>16.58</td><td>1.29</td><td>0.23</td><td>0.19</td><td>0.15</td><td>0.678</td><td>1.88</td><td>23.60</td><td>1.17</td></tr><tr><td>ICCV 25</td><td>VideoFusion TemCoCo</td><td>0.30 0.35</td><td>0.30 0.28</td><td>0.26</td><td>0.803</td><td>1.91</td><td>6.41</td><td>1.32</td><td>0.27</td><td>0.27</td><td>0.29</td><td>0.815</td><td>1.89</td><td>9.82</td><td>1.26</td></tr><tr><td></td><td>UniVF</td><td>0.44</td><td></td><td>0.23</td><td>0.801</td><td>1.87</td><td>9.66</td><td>1.30</td><td>0.32</td><td>0.27</td><td>0.24</td><td>0.808</td><td>1.89</td><td>13.16</td><td>1.20</td></tr><tr><td>NeurIPS 25</td><td>Ours</td><td>0.55</td><td>0.41 0.65</td><td>0.27 0.31</td><td>0.803 0.809</td><td>1.78 1.71</td><td>11.02 10.74</td><td>1.24 1.03</td><td>0.43 0.52</td><td>0.49 0.60</td><td>0.34 0.37</td><td>0.816 0.816</td><td>1.74 1.67</td><td>14.24 14.15</td><td>1.10 1.01</td></tr></table>

![](images/cf5f59d9aafd01e09d706a07572642d5f1115fe623c15969a2f9b942b4fed086.jpg)  
Fig. 4. Qualitative comparisons of the temporal consistency for source and fused videos in Scene ’0111 1803’ from the M3SVD dataset. We visualize the pixels of the selected column (the red line).

Temporal Comparison. As illustrated in Fig. 4, the pixels along the fixed column indicated by the red line are extracted from each fused video sequence and temporally concatenated to provide an intuitive visualization of the temporal variations produced by different fusion methods. It can be observed that several image fusion methods suffer from pronounced temporal flickering, particularly DSPFusion and GIFNet. Although VideoFusion achieves favorable performance in terms of temporal metrics, its fused results tend to be overly smooth, leading to degraded edge sharpness and reduced clarity of local details. Furthermore, Fig. 5 presents the frame-wise scores of different evaluation metrics over consecutive frames. The metric curves of several methods exhibit noticeable fluctuations, with UniVF and ISFusion showing particularly unstable behavior. Their relatively large standard deviations further confirm such temporal variations from a statistical perspective. Overall, the proposed method demonstrates consistently stable performance in both subjective visual comparisons and objective quantitative evaluations.

![](images/109d302d9b55af55e71ccb0e4eea721a174c85bc4c22de40cd03db76099172e4.jpg)  
Fig. 5. Temporal curves of different metrics over consecutive frames of the sequence from Scene ’0111 1803’ of the M3SVD dataset. Mean denotes average value and SD represents standard deviation.

Downstream Task Comparison. To further validate the advantage of the proposed method in temporal stability, we employ SAMWISE [54] as a unified segmentation model to perform video segmentation on the sequences generated by different video fusion methods. As shown in Fig. 6, at time T + 6, all comparison methods exhibit varying degrees of segmentation errors, whereas our method still maintains relatively accurate target segmentation. This demonstrates that our method preserves smoother inter-frame transitions in the fused sequence, thereby providing more stable visual information for downstream segmentation tasks.

![](images/74f58e435c45be842dcd2f2d6285c99c61838ac5bf5d8a9c2fdb7258f6a16fee.jpg)  
Fig. 6. Comparison of segmentation results produced by different video fusion methods on scene ‘0118 1913’ of the M3SVD dataset. The bottom row of each image presents cropped views of the blue-boxed region at two-frame intervals from T to T + 8. The segmentation prompt used for this scene is ’the person on the right side wearing a hoodie.

![](images/16add108a2553a522ce912900f42b093d00359b1f3678097e276de56ab7bd94b.jpg)  
Fig. 7. Bubble chart comparing the computational efficiency and fusion performance of different methods. The bubble size is determined by the average rank of each method over all datasets and evaluation metrics, where a larger bubble denotes better overall performance.

Efficiency Comparison. To evaluate the practical inference efficiency of different methods under resource-constrained conditions, we further conduct runtime experiments on an NVIDIA Jetson AGX Orin 64GB Developer Kit. For a fair comparison, all methods are evaluated using FP32 inference under the same software environment and timing protocol, using FP32 tensors with TF32 acceleration enabled on supported NVIDIA GPUs, at an input resolution of 640×480. As shown in Fig. 7, although UniVF achieves competitive fusion performance, it does so at the expense of substantially reduced computational efficiency. In contrast, our method achieves the highest inference speed of 27.83 FPS while requiring only 63 GFLOPs/frame, representing the second-lowest computational cost among all compared methods. These results demonstrate a superior fusion performance–speed trade-off for real-time inference on the edge platform.

![](images/4f35877f04e1f8f32edc693b223b088e567855687e80b093d1433738140bdf4b.jpg)  
Fig. 8. Visualization of variation responses in the pixel space and different pretrained feature spaces for Scene ’002’ from the HDO dataset. (a) Adjacentframe variation. (b) Variation from the temporal mean.

TABLE II  
IOU OF THE TOP-K HIGH-RESPONSE REGIONS SELECTED FROM THE PIXEL-LEVEL VARIATION MAP AND DIFFERENT PRETRAINED FEATURE MAPS. THE BEST RESULT IN EACH ROW IS HIGHLIGHTED IN BOLD.
<table><tr><td></td><td>CLIP</td><td>DINO</td><td>MAE</td></tr><tr><td>ρ = 0.10</td><td>0.21</td><td>0.40</td><td>0.58</td></tr><tr><td>ρ = 0.20</td><td>0.30</td><td>0.46</td><td>0.60</td></tr><tr><td>ρ = 0.25</td><td>0.33</td><td>0.48</td><td>0.60</td></tr><tr><td>ρ = 0.30</td><td>0.36</td><td>0.49</td><td>0.60</td></tr><tr><td> $\rho = 0 . 4 0$ </td><td>0.42</td><td>0.52</td><td>0.62</td></tr></table>

## C. Analysis of the MAE Feature Space

To investigate whether the MAE feature space can more effectively characterize structural- and region-level variations when applied to video sequences, we conduct comprehensive analyses by comparing MAE with other pretrained models. For the qualitative analysis, as illustrated in Fig. 8, MAE exhibits the highest spatial consistency with the pixel-level responses among the compared pretrained models under both variation formulations, particularly in the high-response regions. For the quantitative analysis, we further evaluate the spatial consistency between the high-response regions obtained from different pretrained feature spaces and the corresponding pixel-level responses using IoU. As shown in Table II, MAE consistently achieves the highest IoU across different ρ settings, substantially outperforming the other pretrained models. These qualitative and quantitative results jointly demonstrate that MAE provides a stronger capability for characterizing structural- and region-level variations in video sequences.

## D. Analysis of the Hyperparameter ρ

As shown in Table III, compared with full-region temporal modeling without dynamic region guidance $( \rho = 1 . 0 0 )$ , the guided variants achieve better performance on the objective evaluation. This indicates that dynamic region guidance enables STIM to focus temporal interactions on tokens containing more informative variations, thereby improving the effectiveness of temporal modeling. Among the different dynamicregion ratios, our method $( \rho = 0 . 2 5 )$ achieves the best overall performance.

## E. Analysis of the FD3 Metric

As shown in Table IV, we construct a stress benchmark for the temporal metrics comprising three types of perturbations: temporal misalignment, local content lag, and temporal

TABLE III  
QUANTITATIVE COMPARISON WITH DIFFERENT ρ ON HDO DATASET. THE BEST AND SECOND-BEST RESULTS ARE DENOTED BY BESTCOLOR AND SECONDCOLOR RESPECTIVELY.
<table><tr><td></td><td>QG↑</td><td>QM↑</td><td>VIF↑</td><td>FSIM↑</td><td>feaCD↓</td><td>flowD↓</td><td>FD3↓</td></tr><tr><td> $\overline { { \rho = 0 . 1 0 } }$ </td><td>0.553</td><td>0.646</td><td>0.310</td><td>0.808</td><td>1.715</td><td>10.795</td><td>1.041</td></tr><tr><td> $\rho = 0 . 2 5$ </td><td>0.554</td><td>0.649</td><td>0.308</td><td>0.809</td><td>1.712</td><td>10.738</td><td>1.034</td></tr><tr><td> $\rho = 0 . 5 0$ </td><td>0.549</td><td>0.640</td><td>0.307</td><td>0.809</td><td>1.713</td><td>10.650</td><td>1.037</td></tr><tr><td> $\rho = 1 . 0 0$ </td><td>0.544</td><td>0.615</td><td>0.303</td><td>0.807</td><td>1.721</td><td>10.751</td><td>1.038</td></tr></table>

## TABLE IV

SENSITIVITY COMPARISON OF TEMPORAL METRICS UNDER DIFFERENT DEGRADATION TYPES. EACH DEGRADATION TYPE IS EVALUATED UNDER PROGRESSIVELY MORE CHALLENGING SETTINGS. VALUES IN PARENTHESES DENOTE RELATIVE CHANGES WITH RESPECT TO THE CLEAN BASELINE. THE BASELINE VALUES ARE OBTAINED FROM THE PROPOSED METHOD ON THE HDO DATASET WITHOUT DEGRADATION. THE MOST DESIRABLE RESPONSE IN EACH ROW, JOINTLY CONSIDERING SENSITIVITY, CORRECT RESPONSE DIRECTION, AND MONOTONICITY, IS HIGHLIGHTED IN BOLD.
<table><tr><td>Perturbation</td><td>Setting</td><td>feaCD↓</td><td>flowD↓</td><td>FD3↓</td></tr><tr><td>Clean</td><td>Baseline</td><td>1.71</td><td>10.74</td><td>1.03</td></tr><tr><td rowspan="3">Misalign</td><td>t=1</td><td>2.04 (+19.0%)</td><td>10.85 (+1.1%)</td><td>1.36 (+31.1%)</td></tr><tr><td>t=2</td><td>2.02 (+18.0%)</td><td>11.19 (+4.2%)</td><td>1.50 (+44.9%)</td></tr><tr><td>t=3</td><td>1.99 (+16.0%)</td><td>11.39 (+6.0%)</td><td>1.59 (+53.6%)</td></tr><tr><td rowspan="3">Local Lag</td><td>area=0.6</td><td>1.97 (+14.9%)</td><td>11.22 (+4.5%)</td><td>1.38 (+34.0%)</td></tr><tr><td>area=0.5</td><td>1.96 (+14.5%)</td><td>11.16 (+3.9%)</td><td>1.33 (+29.1%)</td></tr><tr><td>area=0.4</td><td>1.96 (+14.2%)</td><td>11.08 (+3.2%)</td><td>1.28 (+24.2%)</td></tr><tr><td rowspan="3">Smoothing</td><td>windows=3</td><td>1.92 (+12.1%)</td><td>6.39 (-40.5%)</td><td>1.16 (+12.5%)</td></tr><tr><td>windows=5</td><td>1.97 (+15.1%)</td><td>4.77 (-55.6%)</td><td>1.24 (+19.9%)</td></tr><tr><td>windows=7</td><td>1.97 (+15.2%)</td><td>3.90 (-63.7%)</td><td>1.30 (+26.1%)</td></tr></table>

smoothing. By progressively introducing temporal degradations of different types and severity levels, the benchmark systematically evaluates the degradation-awareness capability of temporal quality metrics in terms of sensitivity, responsedirection consistency, and monotonicity. On this basis, FD3 is compared with two existing temporal metrics, flowD and feaCD, to assess their ability to perceive different forms of temporal degradation. The detailed generation procedures for these perturbations are provided in the supplementary material.

Under temporal misalignment, flowD changes by only 1.1%–6.0% relative to the unperturbed baseline, indicating limited sensitivity to this type of degradation. feaCD exhibits a more pronounced deviation from the baseline, suggesting that it can detect temporal misalignment. However, as the temporal offset increases from one to three frames, its relative increase unexpectedly decreases from 19.0% to 16.0%. The non-monotonic response is inconsistent with the increasing degradation severity. In contrast, FD3 not only exhibits a substantially larger deviation from the baseline, but also shows a continuously increasing response from 31.1% to 53.6% as the misalignment becomes more severe. This demonstrates its stronger capability to distinguish different levels of temporal misalignment.

Under local content lag, both flowD and feaCD can, to some extent, discriminate the perturbed sequences from the original baseline. Nevertheless, their responses vary only marginally with the area of the affected region, indicating limited capability to differentiate the spatial extent of local degradation. By comparison, FD3 exhibits a clearer severity-dependent response. As the affected-region ratio increases from 0.4 to 0.6, its relative increase rises from 24.2% to 34.0%, demonstrating greater sensitivity and more desirable monotonic behavior.

TABLE V  
SENSITIVITY COMPARISON OF FD3 USING CLIP [55], DINO [56], AND MAE AS PRETRAINED FEATURE EXTRACTORS UNDER LOCAL CONTENT LAG PERTURBATIONS. PERCENTAGES IN PARENTHESES DENOTE RELATIVE CHANGES FROM THE CLEAN BASELINE. THE BASELINE VALUES ARE OBTAINED BY EVALUATING EACH METRIC ON THE UNPERTURBED FUSION RESULTS OF THE PROPOSED METHOD ON THE HDO DATASET. THE LARGEST RELATIVE INCREASE FROM THE CORRESPONDING CLEAN BASELINE IS HIGHLIGHTED IN BOLD.
<table><tr><td></td><td>Baseline</td><td>area=0.60</td><td>area=0.50</td><td>area=0.40</td></tr><tr><td>CLIP</td><td>1.72</td><td>1.84 (+7%)</td><td>1.83 (+6%)</td><td>1.81 (+6%)</td></tr><tr><td>DINO</td><td>1.35</td><td>1.60 (+19%)</td><td>1.57 (+17%)</td><td>1.54 (+14%)</td></tr><tr><td>MAE</td><td>1.03</td><td>1.38 (+34%)</td><td>1.33 (+29%)</td><td>1.28 (+24%)</td></tr></table>

Under temporal smoothing, flowD continuously decreases as the smoothing strength increases. Since a lower flowD value is interpreted as better temporal quality, this behavior implies that more severe oversmoothing is incorrectly assigned a better evaluation score. This observation suggests that flowD tends to favor smoother inter-frame transitions, but such a preference fails to properly reflect the temporal quality degradation caused by excessive smoothing. Although the response direction of feaCD is generally consistent with expectation, its relative increase grows only from 12.1% to 15.2% and gradually saturates under stronger smoothing levels, thereby limiting its ability to distinguish different degrees of temporal smoothing. In contrast, the relative increase of FD3 steadily rises from 12.5% to 26.1%, allowing it to continuously capture the temporal quality degradation induced by increasingly severe smoothing.

Across all three perturbations, feaCD exhibits relatively weak responses to changes in degradation severity, which may be attributed to its global information aggregation mechanism. In particular, when static backgrounds occupy a large proportion of the test sequence, temporal degradations occurring in local dynamic regions can be diluted by the dominant stable background information, thereby weakening the metric response to the severity of localized temporal variations.

On the other hand, Table V further investigates the influence of pretrained feature spaces on the local degradation sensitivity of FD3. Under local content lag, CLIP exhibits only weak responses to changes in the affected-region ratio, whereas DINO produces more noticeable variations. MAE consistently yields the largest relative changes across all settings. Meanwhile, as the affected-region ratio decreases, the relative change of MAE-based FD3 progressively declines from 34% to 24%, forming a clearer monotonic hierarchy. These results suggest that MAE is the most suitable pretrained feature extractor for FD3, enabling more sensitive and discriminative responses to local temporal variations across different spatial extents.

Overall, FD3 is not intended as a simple replacement for existing temporal metrics. Instead, it provides a more interpretable temporal diagnostic perspective that is better aligned with the requirements of MMVF. The stress-benchmark results demonstrate that FD3 consistently exhibits higher sensitivity, more reliable response directions, and clearer monotonic trends across all degradation types, validating its effectiveness in quantifying localized discrepancies in dynamic variation directions between the fused and source sequences and thereby complementing existing temporal metrics.

TABLE VI  
ABLATION STUDY OF ARCHITECTURAL VARIANTS, PRETRAINED FEATURESPACES, AND LOSS TERMS ON THE HDO DATASET. THE BEST RESULTSARE HIGHLIGHTED IN BOLD.
<table><tr><td></td><td> $\overline { { \mathrm { ~ Q G ~ } } }$ </td><td> $\overline { { \mathrm { ~ Q _ { M } ~ } } }$ </td><td>VIF</td><td>FSIM feaCD</td><td>flowD</td><td>FD3</td></tr><tr><td>w ViT</td><td>0.551 0.635 0.306</td><td></td><td>0.808</td><td>1.713</td><td>10.771</td><td>1.037</td></tr><tr><td>w/o STIM</td><td>0.543 0.633 0.305</td><td></td><td>0.809</td><td>1.717</td><td>10.537 1.052</td><td></td></tr><tr><td>w CLIP</td><td>0.522 0.621 0.309</td><td></td><td>0.809</td><td>1.728</td><td>10.427 1.052</td><td></td></tr><tr><td>w DINO</td><td>0.5390.643 0.321</td><td></td><td>0.812</td><td>1.713</td><td></td><td>10.4761.051</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { r e v } }$ </td><td>0.5530.6410.308</td><td></td><td>0.809</td><td>1.720</td><td>10.7641.035</td><td></td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { t e m p } } \& \mathcal { L } _ { \mathrm { r e v } }$ </td><td>0.536 0.628 0.302</td><td></td><td>0.806</td><td>1.722</td><td>10.9841.055</td><td></td></tr><tr><td>Ours</td><td>0.554 0.6490.308</td><td></td><td>0.809</td><td>1.712</td><td>10.7381.034</td><td></td></tr></table>

![](images/14c97d04736435e577001daabfb95bcaf91c178f2195c24bfb7d3ac7cdeaf089.jpg)

![](images/74c70dd3b009cfa661f70280bdb8fae6b6869af3695ea4a1ae65badc1806f40d.jpg)  
Fig. 9. Computational efficiency of ViT and Ours with respect to the number of input frames at two spatial resolutions. For each line chart, the left and right groups correspond to resolutions of 640×480 and 1280×1024, respectively.

## F. Ablation Experiment

To validate the effectiveness of the key components in the proposed method, we conduct six ablation experiments: (I) w ViT: retaining the dynamic region-guided mechanism while replacing the proposed STIM with a conventional ViT; (II) w/o STIM: directly removing STIM; (III) w CLIP: replacing MAE with CLIP as the feature extractor for the temporal loss; (IV) w DINO: replacing MAE with DINO as the feature extractor for the temporal loss; (V) w/o $\mathcal { L } _ { \mathrm { r e v } } .$ removing the ${ \mathcal { L } } _ { \mathrm { r e v } }$ loss; and (VI) w/o ${ \mathcal { L } } _ { \mathrm { r e v } }$ & $\mathcal { L } _ { \mathrm { t e m p } } \mathrm { . }$ simultaneously removing both ${ \mathcal { L } } _ { \mathrm { r e v } }$ and $\mathcal { L } _ { \mathrm { t e m p } } .$

As shown in Table VI, replacing STIM with ViT results in a slight degradation in the overall quantitative performance. More importantly, as illustrated in Fig. 9, the computational cost of ViT increases rapidly as the number of input frames or the spatial resolution grows, accompanied by a substantial reduction in inference speed. In contrast, STIM employs a linear-complexity sequence modeling mechanism and therefore maintains relatively stable computational efficiency across different input scales.

For experiments (II) and (VI), the visual comparisons in Fig. 10 show that the regions highlighted by the boxes suffer from varying degrees of blurring, whereas the proposed model preserves smoother content transitions in the corresponding regions. Moreover, at the edge locations indicated by the triangles, both experiments exhibit conspicuous abrupt dark variations, suggesting the occurrence of temporal flickering. The quantitative results in Table VI further show that experiments (II) and (VI) perform considerably worse than the proposed model across the overall metrics, demonstrating the importance of STIM and the temporal constraints.

![](images/334bda7d3e0f52316fa84585ee3a0c64f2c6f6de727741ddb93c2a6b34486e2a.jpg)  
${ \mathrm { F i g . } }$ 10. Visual comparison of temporal consistency between two ablation variants: (II) w/o STIM and (VI) w/o $\mathcal { L } _ { \mathrm { t e m p } } \& \mathcal { L } _ { \mathrm { r e v } }$

For the ablations on pretrained feature extractors, namely experiments (III) and (IV), CLIP- and DINO-based variants achieve the best results on a few individual metrics. Nevertheless, their overall performance remains inferior to that of the proposed configuration using MAE, indicating that MAE features are more suitable for constructing the proposed temporal loss.

Finally, the comparison between experiment (V) and the proposed model shows that introducing ${ \mathcal { L } } _ { \mathrm { r e v } }$ yields consistent improvements across the evaluation metrics, further validating its effectiveness in enhancing the overall performance of the proposed model.

## V. LIMITATION AND FUTURE WORK

The proposed method adopts a sequence-based fusion paradigm that enables effective sequence-level temporal interaction. In online inference, however, fixed-length clips must be buffered before joint processing, resulting in additional latency. Future work will investigate a streaming fusion framework with temporal state propagation, enabling incremental state updates to further reduce computational cost and improve inference speed.

## VI. CONCLUSION

This paper proposes an efficient MMVF algorithm that mainly relies on a TTT-based sequence modeling mechanism to effectively interact with dynamic regions along the temporal dimension. In addition, to suppress temporal flickering, we introduce a temporal consistency constraint that employs MAE as the feature extractor. The proposed framework not only produces high-quality fusion results on multiple public datasets, but also demonstrates higher inference efficiency when evaluated on an edge device. Finally, we introduce FD3 to evaluate the dynamic direction distance of temporal variations within dynamic regions in the pretrained MAE feature space.

## REFERENCES

[1] L. Tang, Y. Wang, M. Gong, Z. Li, Y. Deng, X. Yi, C. Li, H. Xu, H. Zhang, and J. Ma, “Videofusion: A spatio-temporal collaborative network for multi-modal video fusion,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 19 559–19 569.

[2] X. Qin, Y. Cui, S. Sun, R. Chen, W. Ren, A. Knoll, and X. Cao, “Disentangle to fuse: Towards content preservation and cross-modality consistency for multi-modality image fusion,” IEEE Transactions on Image Processing, 2026.

[3] J. Chen, W. Yu, X. Tian, J. Huang, and J. Ma, “Mdbfusion++: A visible and infrared image fusion framework capable for motion deblurring,” IEEE Transactions on Image Processing, vol. 35, pp. 3647–3661, 2026.

[4] Z. Zhao, H. Bai, B. Ke, Y. Cui, L. Deng, Y. Zhang, K. Zhang, and K. Schindler, “A unified solution to video fusion: From multi-frame learning to benchmarking,” Advances in Neural Information Processing Systems, vol. 38, pp. 128 504–128 535, 2026.

[5] M. Gong, H. Zhang, X. Yi, L. Tang, and J. Ma, “Temcoco: Temporally consistent multi-modal video fusion with visual-semantic collaboration,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 14 326–14 335.

[6] C. Cheng, T. Xu, X.-J. Wu, T. Zhou, H. Li, Z. Tang, and J. Kittler, “Evanet: Towards more efficient and consistent infrared and visible image fusion assessment,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[7] Z.-H. Cao, Y.-J. Liang, L.-J. Deng, and G. Vivone, “An efficient image fusion network exploiting unifying language and mask guidance,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[8] J. Xin, B. Shi, Z. Liang, J. Hao, X. Song, N. Wang, J. Li, and X. Gao, “Interpretable general image fusion via scalable autoregressive modeling,” IEEE Transactions on Image Processing, vol. 35, pp. 6632– 6644, 2026.

[9] J. Chen, L. Yang, W. Yu, W. Gong, Z. Cai, and J. Ma, “Sdsfusion: A semantic-aware infrared and visible image fusion network for degraded scenes,” IEEE Transactions on Image Processing, 2025.

[10] H. Xu, X. Yi, C. Lu, G. Liu, and J. Ma, “Urfusion: Unsupervised unified degradation-robust image fusion network,” IEEE Transactions on Image Processing, 2025.

[11] L. Tang, C. Li, and J. Ma, “Mask-difuser: A masked diffusion model for unified unsupervised image fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[12] J. Zhang, X. Yu, and Y. Lin, “Rethinking occlusion modeling for uav tracking,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 13 563–13 573.

[13] Z. Zhao, Y. Cui, L. Deng, H. Bai, H. Qin, T. Feng, and K. Schindler, “Mambavf: State space model for efficient video fusion,” arXiv preprint arXiv:2602.06017, 2026.

[14] Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Chen, X. Wang, S. Koyejo et al., “Learning to (learn at test time): Rnns with expressive hidden states,” in International Conference on Machine Learning. PMLR, 2025, pp. 57 503–57 522.

[15] X. Li, X. Li, T. Tan, H. Li, and T. Ye, “Umcfuse: A unified multiple complex scenes infrared and visible image fusion framework,” IEEE Transactions on Image Processing, 2025.

[16] X. Li, F. Zhou, and H. Tan, “Joint image fusion and denoising via threelayer decomposition and sparse representation,” Knowl. Based Syst., vol. 224, p. 107087, 2021.

[17] X. Li, X. Li, and W. Jiang, “Text-guided channel perturbation and pretrained knowledge integration for unified multi-modality image fusion,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 40, no. 8, 2026, pp. 6521–6529.

[18] X. Li, H. Liu, X. Li, T. Ye, Z. Kuang, and H. Li, “Awm-fuse: Multimodality image fusion for adverse weather via global and local text perception,” IEEE Transactions on Image Processing, 2026.

[19] J. Huang, X. Li, T. Tan, X. Li, F. Zhou, and H. Li, “Mma-unet: A multi-modal asymmetric unet architecture for infrared and visible image fusion,” IEEE Transactions on Emerging Topics in Computational Intelligence, 2026.

[20] J. Liu, G. Wu, Z. Liu, D. Wang, Z. Jiang, L. Ma, W. Zhong, X. Fan, and R. Liu, “Infrared and visible image fusion: From data compatibility to task adaption,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 4, pp. 2349–2369, 2024.

[21] H. Li, T. Xu, X.-J. Wu, J. Lu, and J. Kittler, “Lrrnet: A novel representation learning guided fusion network for infrared and visible images,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 11 040–11 052, 2023.

[22] J. Huang, X. Li, H. Tan, L. Yang, G. Wang, and P. Yi, “Dednet: Infrared and visible image fusion with noise removal by decomposition-driven network,” Measurement, p. 115092, 2024.

[23] C. He, Z. Jia, H. Zhao, S. Song, J. Wang, and F. Shi, “Image fusion based on prior information.” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. PP, 2026.

[24] Q. Wang, Z. Li, S. Zhang, N. Chi, and Q. Dai, “Wavefusion: A novel wavelet vision transformer with saliency-guided enhancement for multimodal image fusion,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[25] X. Wu, Z. H. Cao, T. Z. Huang, L. J. Deng, J. Chanussot, and G. Vivone, “Fully-connected transformer for multi-source image fusion,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 3, pp. 2071–2088, 2025.

[26] J. Shi, P. Duan, X. Ma, J. Chi, and Y. Dai, “Frefusion: Frequency domain transformer for infrared and visible image fusion,” IEEE Transactions on Multimedia, 2025.

[27] J. Huang, X. Li, H. Tan, and X. Cheng, “Generative adversarial network for trimodal medical image fusion using primitive relationship reasoning,” IEEE Journal ofBiomedical and Health Informatics, vol. 28, no. 10, pp. 5729–5741, 2024.

[28] J. Ma, H. Xu, J. Jiang, X. Mei, and X.-P. Zhang, “Ddcgan: A dual-discriminator conditional generative adversarial network for multiresolution image fusion,” IEEE Transactions on Image Processing, vol. 29, pp. 4980–4995, 2020.

[29] J. Zhang, L. Jiao, W. Ma, F. Liu, X. Liu, L. Li, P. Chen, and S. Yang, “Transformer based conditional gan for multimodal image fusion,” IEEE Transactions on Multimedia, vol. 25, pp. 8988–9001, 2023.

[30] Y. Shi, Y. Liu, Z.-C. Wu, J. Cheng, H. Li, and X. Chen, “Degradationrobust fusion: An efficient degradation-aware diffusion framework for multimodal image fusion in arbitrary degradation scenarios,” arXiv preprint arXiv:2604.08922, 2026.

[31] Z. Wang, J. Zhang, T. Guan, Y. Zhou, X. Li, M. Dong, and J. Liu, “Efficient rectified flow for image fusion,” Advances in Neural Information Processing Systems, vol. 38, pp. 22 927–22 948, 2026.

[32] K. Cao, X. He, T. Hu, C. Xie, M. Zhou, and J. Zhang, “Shuffle mamba: State space models with random shuffle for multi-modal image fusion,” IEEE Transactions on Circuits and Systems for Video Technology, 2026.

[33] C. Zhu, S. Deng, X. Song, Y. Li, and Q. Wang, “Mamba collaborative implicit neural representation for hyperspectral and multispectral remote sensing image fusion,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[34] X. Wei, Y. Qiu, J. Xu, and J. Zhang, “Hcfmanet: A novel holistic crossmodal fusion mamba network for multi-modal medical image fusion,” IEEE Transactions on Multimedia, 2026.

[35] J. Huang, T. Tan, X. Li, T. Ye, and Y. Wu, “Multiple attention channels aggregated network for multimodal medical image fusion,” Medical Physics, vol. 52, pp. 2356–2374, 2024.

[36] J. Xia, H. Song, J. Li, S. Ma, Y. Lin, Y. Huang, and J. Yang, “Rethinking the cnn-transformer hybrid architecture for infrared and visible image fusion from a frequency perspective,” IEEE Transactions on Multimedia, 2026.

[37] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[38] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, “Transformers are rnns: Fast autoregressive transformers with linear attention,” in International Conference on Machine Learning. PMLR, 2020, pp. 5156–5165.

[39] D. Han, X. Pan, Y. Han, S. Song, and G. Huang, “Flatten transformer: Vision transformer using focused linear attention,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 5961–5971.

[40] Z. Qin, W. Sun, H. Deng, D. Li, Y. Wei, B. Lv, J. Yan, L. Kong, and Y. Zhong, “cosformer: Rethinking softmax in attention,” in International Conference on Learning Representations.

[41] D. Han, Y. Li, T. Li, Z. Cao, Z. Wang, J. Song, Y. Cheng, B. Zheng, and G. Huang, “ViT<sup>3</sup>: Unlocking test-time training in vision,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 51–61.

[42] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 7132–7141.

[43] J. Li, J. Jiang, P. Liang, J. Ma, and L. Nie, “Maefuse: Transferring omni features with pretrained masked autoencoders for infrared and visible image fusion via guided training,” IEEE Transactions on Image Processing, 2025.

[44] M. Zhou, J. Huang, N. Zheng, and C. Li, “Learned image reasoning prior penetrates deep unfolding network for panchromatic and multi-spectral image fusion,” 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12 364–12 373, 2023.

[45] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked au-´ toencoders are scalable vision learners,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 000–16 009.

[46] H. Xie, M. Sang, Y. Zhang, Y. Yang, S. Zhao, and J. Zhong, “Rcvs: A unified registration and fusion framework for video streams,” IEEE Transactions on Multimedia, vol. 26, pp. 11 031–11 043, 2024.

[47] P. Zhang, J. Zhao, D. Wang, H. Lu, and X. Ruan, “Visible-thermal uav tracking: A large-scale benchmark and new baseline,” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8876–8885, 2022.

[48] L. Tang, C. Li, Y. Wang, G. Wang, Y. Yuan, and J. Ma, “Dspfusion: Image fusion via degradation and semantic dual-prior guidance,” IEEE Transactions on Image Processing, vol. 35, pp. 6331–6345, 2026.

[49] X. Wang, Z. Guan, W. Qian, C. Wang, and R. Ma, “Multi-modal image fusion via intervention-stable feature learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 33 827–33 837.

[50] C. Cheng, T. Xu, Z. Feng, X. Wu, Z. Tang, H. Li, Z. Zhang, S. Atito, M. Awais, and J. Kittler, “One model for all: Low-level task interaction is a key to task-agnostic image fusion,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 28 102–28 112.

[51] J. Liu, R. Lin, G. Wu, R. Liu, Z. Luo, and X. Fan, “Coconet: Coupled contrastive learning network with multi-level feature ensemble for multimodality image fusion,” International Journal of Computer Vision, vol. 132, no. 5, pp. 1748–1775, 2024.

[52] Z. Liu, E. Blasch, Z. Xue, J. Zhao, R. Laganiere, and W. Wu, “Objective\` assessment of multiresolution image fusion algorithms for context enhancement in night vision: A comparative study,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, pp. 94–109, 2012.

[53] L. Zhang, L. Zhang, X. Mou, and D. Zhang, “Fsim: A feature similarity index for image quality assessment,” IEEE Transactions on Image Processing, vol. 20, pp. 2378–2386, 2011.

[54] C. Cuttano, G. Trivigno, G. Rosi, C. Masone, and G. Averta, “Samwise: Infusing wisdom in sam2 for text-driven video segmentation,” 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3395–3405, 2025.

[55] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning. PmLR, 2021, pp. 8748–8763.

[56] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” Transactions on Machine Learning Research Journal, 2024.