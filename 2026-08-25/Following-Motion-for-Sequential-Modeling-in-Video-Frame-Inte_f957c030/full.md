# Following Motion for Sequential Modeling in Video Frame Interpolation

Jaehyun Park<sup>1</sup> and Nam Ik Cho<sup>1,2,3</sup>

<sup>1</sup> IPAI, Seoul National University

2 Department of Electrical and Computer Engineering, INMC, Seoul National

University

3 SNUAILab

{jaep0805,nicho}@snu.ac.kr

Abstract. State Space Models (SSMs) have surfaced as a promising architecture in Video Frame Interpolation (VFI), as they can capture long-range dependencies with linear computational complexity. However, their predefined scanning order limits their efectiveness in modeling the dynamic motion trajectories inherent in VFI problems. To tackle this challenge, we propose Motion-Guided Mamba for Video Frame Interpolation (MGMVFI), an adaptation of the selective state space model tailored explicitly for VFI. MGMVFI introduces Motion-Guided Serialization (MGS), which leverages optical flow to define a motionadaptive 1D input order for the SSM. This aligns the causal state updates with semantically related tokens, enabling motion-consistent feature propagation, particularly for large and dynamic motions. Additionally, to mitigate the unreliable feature representations caused by inaccurate optical flow estimates, we introduce contextual synthesis that utilizes the surrounding spatial context for robust inter-frame feature synthesis. These components are seamlessly integrated within our tailored Mamba architecture, which also employs a lightweight refinement block to enhance local detail reconstruction at a reduced computational cost. Extensive experiments on standard VFI benchmarks demonstrate that MGMVFI achieves state-of-the-art performance, particularly on complex and dynamic motions, thereby establishing a new direction for sequence modeling in video interpolation.

Keywords: Video Frame Interpolation · State Space Models · Mamba

## 1 Introduction

Video Frame Interpolation (VFI) is a long-standing task in computer vision that aims to synthesize intermediate frames between consecutive input frames in a video sequence. This task plays a crucial role in numerous applications, such as video frame rate up-conversion [1,2,4,14], slow-motion video generation [1, 19], and novel view synthesis [6, 20, 52]. By increasing temporal resolution, VFI improves visual smoothness and performance of downstream video tasks.

![](images/a18de9567e9c83a4b86cade7ee82050e6620b27c87edab65058de64b6ffe8dbb.jpg)  
Fig. 1: We present Motion-Guided Serialization (MGS), which uses optical flow to sample from motion-aligned feature maps $F _ { 0 } ^ { \prime }$ & F<sup>′</sup>. MGS results in consecutive semantically consistent tokens (Patch 7 & 9) for a. Horizontal and b. Vertical scans.

The efectiveness of deep learning-based VFI methods depends on how well the model captures the spatial and temporal dependencies in the input frame pair. Conventional approaches have primarily relied on convolutional neural networks (CNNs) [1,4,14,23] and Transformer-based architectures [31,41,48]. However, CNNs sufer from limited receptive fields, making them less suitable for modeling large motions that span an extensive range. Transformers, on the other hand, ofer global attention mechanisms but sufer from quadratic computational complexity, limiting their scalability for high-resolution, real-time video processing.

Recently, state space models (SSMs), notably the Selective State Space (S6/ Mamba) family, have emerged as promising alternatives that combine the efficiency of convolutional operations with the dynamic modeling capability of recurrent structures [7–9]. S6-based architectures overcome the drawbacks of CNNs and Transformers by updating hidden states in a selective and recurrent manner, enabling the capture of long-range dependencies with linear complexity [7]. The inherent sequential nature of SSMs makes them highly dependent on the scanning order of the input features [16]. This sequence order is very pivotal, as it fundamentally dictates the flow of information through state updates, shaping the model’s understanding of spatial and temporal relationships.

To process video inputs, prior works [7,10,11,25,28] have expanded 2D scanning strategies, such as horizontal, vertical, spiral scans, and Z-scans, by scanning each spatial coordinate sequentially for all images. Furthermore, 3D scans, such as the interleaved [47] and Hilbert curve-based scans [18], have also been specifically developed to learn features that cover both spatial and temporal dimensions. However, these pre-defined, motion-agnostic scanning paths are fundamentally misaligned with the dynamic nature of video. As shown in Figure 1, interleaved scanning [47] interleaves the columns of $F _ { 0 }$ and $F _ { 1 }$ to incorporate the temporal dimension. However, this simple approach often aggregates semantically inconsistent features across time. For instance, Patches 7 and 9 represent features belonging to the same moving object. However, they have a large sequential distance (Gray line) in the interleaved scan’s input sequence, forcing the SSM to both estimate the motion trajectories and model the inter-frame features. Moreover, the sequential distance depends on not only the magnitude of the motion but also the direction of the scan path, introducing additional variability that further hampers the estimation of motion trajectories.

Crucially, we reinterpret motion alignment as a serialization problem for sequence models. Classic flow-based VFI methods [1,19,37] estimate intermediate flows and visibility to warp features for spatial synthesis. In contrast, our goal is to define the 1D input sequence order for the SSM: motion trajectories determine how tokens are serialized, so the causal state updates occur between semantically related patches. To this end, we propose a Motion-Guided Serialization (MGS) method that utilizes optical flow to dynamically rearrange input features along estimated motion paths. We combine this with patch-level interleaved scans to ensure that adjacent patches in the SSM input correspond to semantically-related and temporally-coherent content. The comparison is illustrated in Figure 1. Compared to interleaved scanning [47], our motion-aligned serialization method produces a semantically consistent input sequence, demonstrated by the adjacent features of the moving car in the serialized SSM input sequence (Patch 7 and 9). Additionally, because we interleave at patch-level, the semantically-related features stay adjacent regardless of scan direction.

However, MGS’s dependence on optical flow estimation inherently creates inaccuracies that arise from occlusion/disocclusion, abrupt motion and lighting changes. We therefore further design a complementary masked adapter module to handle areas with inaccurate flow with contextual synthesis. Contextual synthesis identifies these information gaps and synthesizes robust features as replacements by leveraging the surrounding valid spatial context. The resulting motion-aligned sequence provides a rich, coherent input that vastly relaxes the motion estimation and inter-frame modeling objective for the SSM. This allows the downstream SSM to eficiently handle temporal propagation followed by the modified Eficient Discriminative Frequency-domain FFN (EDFFN) for enhanced local detail reconstruction at reduced computational cost [21].

Our contribution can be summarized in threefold:

1. Motion-Guided Serialization (MGS): A novel serialization mechanism that leverages optical flow to dynamically define the SSM input sequence order along estimated motion paths, ensuring semantically consistent sequence modeling for complex and large-magnitude motions.

2. Contextual Synthesis: A robust strategy to restore corrupted feature representations in regions with inaccurate or missing flow by leveraging valid surrounding spatial context to synthesize replacement features.

3. Adaptation of the Mamba architecture: The integration of a modified Eficient Discriminative Frequency-domain FFN (EDFFN) into the Mamba framework to enhance local texture and detail reconstruction at a low cost.

## 2 Related Work

Video Frame Interpolation. Early deep learning-based VFI methods were predominantly built on CNNs. These approaches can be broadly categorized into flow-based and kernel-based methods. Flow-based VFI methods first estimate the optical flow between input frames and then warp these frames to the intermediate time step. Pioneering works, such as [30], introduced this concept, which has been followed by numerous refinements. These include direct estimation of intermediate flows [19, 29], bilateral motion estimation [39, 40], and the use of pre-trained contextual features to enhance the warping process [36]. More recent eforts have focused on eficiency, such as combining estimation and refinement into a single encoder [22] or using distillation [14]. Even in the modern landscape, advanced CNN architectures, such as SGM-VFI [27], continue to demonstrate competitive performance. Kernel-based VFI methods, pioneered by [38], bypass explicit flow estimation. Instead, they estimate spatially-adaptive convolution kernels that sample and weigh neighboring pixels from the input frames to directly synthesize the intermediate frame. To overcome the limitations of fixed local kernels, subsequent works explored adaptive filters [23] and deformable convolutions [2, 3] to better handle large motions and occlusions.

To address the limited receptive fields of CNNs and better model long-range dependencies, researchers began adopting the Transformer architecture. Initial works [31, 48] demonstrated the power of attention mechanisms for VFI. The Transformer was also applied to kernel-based methods [41] to address contentagnostic weights. Recently, EMA-VFI [48] has refined the Transformer architecture, achieving a balance of performance and eficiency that makes it a key state-of-the-art baseline.

A distinct line of research has focused on enhancing the perceptual quality of interpolated frames, particularly in challenging scenarios involving complex motion. Generative models, particularly Denoising Difusion Probabilistic Models [12], have demonstrated remarkable potential. Works such as [17] and [5] have demonstrated that generative priors can yield more realistic results, albeit at a significant computational cost. More recently, EDEN [50] integrated a Transformer-based tokenizer with a DiT-style difusion backbone, while Lyu et al. [32,33] adapted Brownian Bridge Difusion Models [24] to set a high standard for perceptual quality and robustness. In this work, we set generative models aside and focus on deterministic models and their reconstruction performance.

State Space Models. State Space Models (SSMs) [9] were initially proposed for sequential signal modeling and later adapted for use in deep learning, utilizing HiPPO-based initialization and eficient structured representations. S4 [8] introduced parallelized updates to the hidden state, and S6 (Mamba) [7] further advanced the design with selective gating and hardware-aware implementations.

In vision applications, SSMs have been adapted to 2D and video domains [16]. Liu et al. [28] extended selective scanning to image patches, ofering global receptive fields, while VideoMamba [25] incorporated hierarchical bidirectional state updates for eficient video sequence modeling. [49,51] modeled motion-aware dynamic trajectories and temporal dependencies across frames to capture complex temporal dynamics in video sequences. In image restoration, MambaIR [11] introduced local enhancement and channel attention to address pixel forgetting, followed by semantic-based reordering for local-global reasoning [10].

![](images/f31557903fe39602e8a70e2bdec7bcbde27517ba4c8d251ce2c789af1abfc0b2.jpg)  
Fig. 2: Overall pipeline of our model. Our MGMVFI block first rearranges input features based on their motion trajectory. It then augments features in regions with unreliable optical flow using spatial context from the surrounding (highlighted) patches. The Contextual Synthesis is visualized for $\textrm { a } 3 \times 3$ hollow kernel case.

For frame interpolation, VFIMamba [47] was the first to adopt S6-based modeling, employing interleaved frame scanning and curriculum-based training to outperform both CNN- and transformer-based baselines. Afterward, LC-Mamba [18] adopted the Hilbert-curve scan method with shifted windows to maintain spatial continuity across window boundaries and extended this to a 3D voxel-level scan for better spatiotemporal modeling between frames. Our work builds on this line by introducing Motion-Guided Serialization. Unlike prior SSM-based models, which used fixed or heuristic scans, we leverage optical flow to dynamically rearrange scan paths along true motion trajectories, making the sequence modeling itself motion-adaptive.

## 3 Proposed Method

## 3.1 Overview

Our goal is to synthesize an intermediate frame $I _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ at an arbitrary timestep $t \in ( 0 , 1 )$ given two consecutive input frames, $I _ { 0 } , I _ { 1 } \in \mathbb { R } ^ { H \times W \times 3 }$ . As illustrated in Figure 2, our model consists of three stages:

1. Feature Extraction and Optical Flow Estimation: A shared CNN encoder first extracts multi-scale deep feature maps, $F _ { 0 }$ and $F _ { 1 }$ , from the input frames $I _ { 0 }$ and $I _ { 1 }$ . Furthermore, a pre-trained optical flow model is used to estimate the corresponding flow maps: $f _ { 0 \to 1 }$ and $f _ { 1  0 }$

2. Motion-Guided Feature Processing: Our motion-guided pipeline processes the extracted features and optical flow with our proposed Motion-Guided Mamba for Video Frame Interpolation (MGMVFI) block. Within MGMVFI, the features are first processed using Motion-Guided Serialization and Contextual Synthesis to define a motion-adaptive serialization of the inputs. Then, the refined features are passed to our inter-frame feature modeling stage, which uses an SSM and the modified EDFFN block [21] to eficiently capture local details and high-frequency information.

3. Frame Synthesis: Finally, a frame synthesis module takes the motionguided feature map $F _ { t }$ and additional flow priors to synthesize the final RGB output frame, $I _ { t } .$ We utilize the established generation modules presented in [48] and [14] to focus on the adaptation of SSMs.

## 3.2 Motion-Guided Serialization

The core of our MGMVFI block is Motion-Guided Serialization (MGS), which reformulates how features are prepared for the SSM sequence model. Rather than using motion alignment for direct synthesis, MGS uses motion trajectories to define the serialization of features, i.e., the 1D token order fed into the SSM block. Concretely, we warp features to the intermediate grid so that standard raster scans over the resulting map correspond to motion-aligned sequences. This allows the causal flow of information to follow semantic motion, so SSM updates occur between related tokens and can focus on local residuals rather than global trajectory tracking. In practice, we serialize features by sampling along continuous motion trajectories, avoiding the limitations of fixed raster scans or linear blending that frequently produce ghosting artifacts near object boundaries.

To establish these trajectories, we must first estimate the intermediate flow fields originating from the target frame $I _ { t }$ . Given the bi-directional optical flow between the source frames, $f _ { 0 \to 1 }$ and $f _ { 1 \to 0 }$ , we assume linear motion trajectories to scale the flow vectors. We employ forward splatting [37], denoted by $s ,$ to push these scaled vectors onto the intermediate grid $p _ { t } \colon$

$$
f _ { t  0 } = S ( - t \cdot f _ { 0  1 } ) , \qquad f _ { t  1 } = S ( - ( 1 - t ) \cdot f _ { 1  0 } ) ,\tag{1}
$$

Then, calculate the source coordinates $p _ { 0 }$ and $p _ { 1 }$ by applying these intermediate flow fields to the target grid $p _ { t } \colon$

$$
p _ { 0 } = p _ { t } + f _ { t  0 } , \qquad p _ { 1 } = p _ { t } + f _ { t  1 }\tag{2}
$$

Using these coordinates, we perform backward warping via bilinear sampling, denoted by $B ( \cdot , \cdot )$ , to fetch the motion-aligned features from $F _ { 0 }$ and $F _ { 1 }$ :

$$
F _ { 0 } ^ { \prime } = \mathcal { B } ( F _ { 0 } , p _ { 0 } ) , \qquad F _ { 1 } ^ { \prime } = \mathcal { B } ( F _ { 1 } , p _ { 1 } )\tag{3}
$$

Examples of the sampled $F _ { 0 } ^ { \prime }$ and $F _ { 1 } ^ { \prime }$ are illustrated in Figure 1. By sampling from these motion-aligned feature maps, semantically consistent patches (e.g.,

Patch 7 and 9) are always consecutively ordered in the 1D sequence, simplifying the learning objective of the SSM to focus solely on inter-frame feature modeling. To quantify this, we further define Sequential Feature Similarity on the 1D serialized sequence as

$$
S = \frac { 1 } { N - 1 } \sum _ { n = 1 } ^ { N - 1 } \exp \left( - \frac { \| \phi ( z _ { n + 1 } ) - \phi ( z _ { n } ) \| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right)\tag{4}
$$

where N is the sequence length and $\phi ( z _ { n } )$ represents the feature representation of the n-th token in the sequence. We use this metric in our experiments (Table 5) to verify that MGS yields higher S, indicating smoother feature similarities between consecutive tokens and a simpler modeling target for the SSM.

## 3.3 Contextual Synthesis for Unreliable Regions

While backward warping populates the entire feature map, it fetches invalid or unreliable features in regions where accurate optical flow is unavailable. These failures occur not only due to physical occlusions and disocclusions, but also estimation inaccuracies caused by large motions or abrupt brightness changes. Consequently, the SSM receives invalid or corrupted tokens from these areas, resulting in substantial reconstruction errors. To address this, we introduce a novel Contextual Synthesis strategy designed to reconstruct these problematic regions by leveraging valid surrounding spatial information.

Flow Consistency Check for Reliability Masking. To identify such regions, we first identify unreliable motion paths using a standard forward-backward consistency check on the optical flow maps $f _ { 0 \to 1 }$ and $f _ { 1 \to 0 }$ . A motion path is considered unreliable if a point $p$ warped from $I _ { 0 }$ to $I _ { 1 }$ and back does not return to its original location:

$$
\epsilon _ { 0 \to 1 } ( p ) = \Big | \Big | f _ { 0 \to 1 } ( p ) + f _ { 1 \to 0 } \big ( p + f _ { 0 \to 1 } ( p ) \big ) \Big | \Big | _ { 2 } ^ { 2 }\tag{5}
$$

A binary consistency mask, which we denote as $M _ { 0 \to 1 } \in \{ 0 , 1 \} ^ { H ^ { \prime } \times W ^ { \prime } \times 1 }$ , is produced by thresholding this squared L2-norm against the average flow magnitude. This identifies pixels where the motion path has failed:

$$
M _ { 0 \to 1 } ( p ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } \epsilon _ { 0 \to 1 } ( p ) > \alpha \cdot { \overline { { | | f _ { 0 \to 1 } | } } } } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{6}
$$

This check is particularly crucial for handling algorithmic failures, such as those caused by abrupt changes in brightness. When lighting variations violate the brightness constancy assumption of the flow network, it often defaults to a zero-magnitude vector. If applied to a moving object, backward warping with this zero vector fetches incorrect spatial locations (e.g., the static background instead of the moving object), causing severe ghosting artifacts. Because our consistency check evaluates the bidirectional agreement of the flow, it successfully flags these zero-vector failures, ensuring unreliable regions are masked regardless of whether they stem from physical occlusions or lighting variations. This bi-directional check produces two distinct binary masks:

$M _ { 0 \to 1 } \colon$ A forward consistency mask identifying regions in $I _ { 0 }$ whose features become unreliable when mapped to $I _ { 1 }$ (e.g. occluded regions).

$M _ { 1  0 } \colon \mathrm { A }$ backward consistency mask identifying regions in $I _ { 1 }$ whose features are unreliable when mapped back to $I _ { 0 }$ (e.g. disoccluded regions).

These masks remain in the coordinates of their respective source frames.

Contextual Synthesis via Masked Adapter. Synthesizing these unreliable regions is challenging because the corresponding features fetched from the opposite frame are corrupted. An intuitive example of this failure occurs during physical occlusion, as seen with Patch 9 in the interleaved scan of Figure 1. Because the tree trunk visible in $F _ { 0 }$ (Patch 9) is covered by the moving car, features fetched from the same estimated location in $F _ { 1 }$ (Patch 9) are incorrect. A straightforward solution is to use the mask $M _ { 0 \to 1 }$ to simply zero out these incorrect features from $F _ { 1 } ^ { \prime } .$ . However, this forces the model to rely solely on the features from $F _ { 0 } .$ , leading to low-quality synthesis characterized by blur and fading artifacts. To fill this information gap, we introduce a masked adapter module $\mathcal { A } _ { 1 }$ that replaces the corrupted features of $F _ { 1 } ^ { \prime }$ with valid surrounding context. As shown by the highlighted patches in Figure 2, $\mathcal { A } _ { 1 }$ uses a $7 \times 7$ hollow convolution, which convolves over the surrounding patches while explicitly excluding the unreliable center patch:

$$
F _ { 1 } ^ { c } = \mathcal { A } _ { 1 } ( F _ { 1 } ^ { \prime } ) = C o n v _ { 1 \times 1 } ( \mathrm { G E L U } ( C o n v H o l l o w _ { 7 \times 7 } ( F _ { 1 } ^ { \prime } ) ) ) + F _ { 1 } ^ { \prime }\tag{7}
$$

By excluding the identified failure patch, we guide the adapter $\mathcal { A } _ { 1 }$ to leverage only the robust, surrounding valid context to supplement the missing feature. A symmetric process is used to handle backward flow failures using a separate adapter $\mathcal { A } _ { \mathrm { 0 } }$ with the surrounding contextual features of $F _ { 0 } ^ { \prime } .$

Refined Feature Composition. The final feature maps for each stream, $F ^ { r e f }$ are composed by selecting either the motion-sampled feature $F ^ { \prime }$ or the adapter’s synthesized output $F ^ { c }$ , guided by the appropriate mask:

$$
F _ { 1 } ^ { r e f } = ( 1 - M _ { 0  1 } ) \cdot F _ { 1 } ^ { \prime } + M _ { 0  1 } \cdot F _ { 1 } ^ { c } , \qquad F _ { 0 } ^ { r e f } = ( 1 - M _ { 1  0 } ) \cdot F _ { 0 } ^ { \prime } + M _ { 1  0 } \cdot F _ { 0 } ^ { c }\tag{8}
$$

Finally, these two refined feature maps from (8) are fused with the original features via an additive connection before being passed to the main SSM block:

$$
F _ { 1 } ^ { f u s e d } = F _ { 1 } + F _ { 1 } ^ { r e f } , \quad F _ { 0 } ^ { f u s e d } = F _ { 0 } + F _ { 0 } ^ { r e f }\tag{9}
$$

For each spatial location i in the target grid, we interleave the corresponding features from $F _ { 0 } ^ { f u s e d }$ and $F _ { 1 } ^ { f u s e d }$ to form a sequence token input $z _ { i } \dot { : }$

$$
\begin{array} { r } { z _ { i } = [ F _ { 0 } ^ { f u s e d } ( i ) \parallel F _ { 1 } ^ { f u s e d } ( i ) ] } \end{array}\tag{10}
$$

where ∥ denotes concatenation along the feature dimension. The final 1D sequence $Z = \{ z _ { 1 } , z _ { 2 } , \dots , z _ { N } \}$ is then fed into the Mamba SSM block.

## 3.4 Inter-frame Feature Modeling

The feature modeling in our pipeline utilizes the following two key components. SS2D Block. We adopt the SS2D block [28] as our SSM. Following its design, we perform $K = 4$ diferent raster scans (horizontal, vertical, and their backward counterparts) on the motion-aligned input sequence Z. Outputs are then aggregated and multiplied by the original input features to modulate intensity according to the spatiotemporal context captured by the SSM. This gated representation then undergoes a final linear layer to produce the refined feature map. This adaptation allows robust estimation of motion trajectories and inter-frame feature modeling within a unified pipeline.

EDFFN Block. While our MGS simplifies the SSM’s objective by semantically aligning features, it also introduces computational overhead. Furthermore, a recent study [34] suggests that the Mamba block tends to focus on low-frequency information, necessitating a complementary module to capture fine-grained details. To ofset the computational cost and simultaneously enhance local feature modeling, we modify and adapt the Eficient Discriminative Frequencydomain FFN (EDFFN). EDFFN is uniquely suited to our aligned features, as its frequency-domain gating mechanism selectively refines local spatial details and enhances texture reconstruction, both of which are crucial for high-fidelity interpolation. On the other hand, the EDFFN’s eficient design, which performs frequency operations on lower dimensions, provides significant computational savings.

## 4 Implementation Details

Training Datasets and Details. We train our model on the Vimeo-90K [46] triplet dataset, a large-scale collection known for its diverse motions. During training, the images are randomly cropped to $2 5 6 \times 2 5 6$ and augmented using horizontal, vertical, and temporal flipping, along with random rotations (90<sup>◦</sup>, 180<sup>◦</sup>, 270<sup>◦</sup>). We empirically set $S = 2$ and $N = 3$ , repeating the feature processing for ×8 and ×16 downsampled features.

The model is trained for 150 epochs with a batch size of 8 on three NVIDIA RTX 3090 GPUs (approx. 3 days). We use the AdamW optimizer $( \beta _ { 1 } = 0 . 9 ;$ $\beta _ { 2 } = 0 . 9 9 9$ , weight decay $1 \times 1 0 ^ { - 4 } )$ . The learning rate is initialized at $2 \times 1 0 ^ { - 4 }$ and gradually decreased to $2 \times 1 0 ^ { - 5 }$ using a cosine annealing schedule. To reduce computational overhead, we employ the pre-trained large RAFT [45] model and pre-compute the bi-directional flow maps. The flow consistency check threshold α is set to 1.1, which empirically yields the best results.

Objective Functions. Our proposed model is trained by minimizing a composite objective function, $\mathcal { L } _ { t o t a l }$ , which combines a primary reconstruction loss and an auxiliary warping loss:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { l a p } + \lambda _ { 1 } \mathcal { L } _ { w a r p } ,\tag{11}
$$

where $\mathcal { L } _ { l a p }$ is the $\mathcal { L } _ { 1 }$ loss computed between the Laplacian pyramids of the final interpolated frame $I _ { t }$ and the ground-truth frame $I _ { t } ^ { \bar { G } T }$ . The warping loss, $\mathcal { L } _ { w a r p } ,$

provides supervision on the image synthesized by the feature $F _ { t } ^ { s }$ of each s-th scale of our multiscale architecture. It is defined as the sum of Laplacian losses between the intermediate warped frames $I _ { t } ^ { s }$ and the ground truth:

$$
\mathcal { L } _ { \mathrm { w a r p } } = \sum _ { s } \mathcal { L } _ { \mathrm { l a p } } ( I _ { t } ^ { s } , I _ { t } ^ { \mathrm { G T } } )\tag{12}
$$

Based on empirical validation, the weighting coeficient $\lambda _ { 1 }$ is set to 0.5.

## 5 Experiment Results

We compare our model against recent state-of-the-art (SotA) VFI models. As a deterministic model, we specifically choose to compare against CNN-based [1, 4, 13, 14, 22, 23, 26, 27, 37, 42], Transformer-based [48, 50], and S6-based [18, 47] models. The metrics reported for LC-Mamba are from its balanced model (Ours-B). For a more comprehensive qualitative assessment, we also include the difusion-based generative model EDEN [50]. As a generative model that focuses on perceptual metrics, EDEN serves as a strong reference for qualitative performance.

Evaluation Datasets. Our model is evaluated on four test benchmarks. Vimeo90K [46] (448×256) and UCF101 [43] (256×256) consist of low-resolution real-world scenes, composed of varying motion magnitude. SNU-FILM [4] is a higher resolution dataset (1280×720) that is divided into four dificulty splits(Easy, Medium, Hard, and Extreme) depending on the motion magnitude. Xiph [35] consists of 4K-resolution frames, which we followed [37] to reproduce 2K-resolution frames and evaluate on both.

## 5.1 Quantitative Evaluation

To quantitatively evaluate our model, we report PSNR and SSIM, the standard reconstruction benchmarks in VFI literature. As shown in Table 1, our model achieves state-of-the-art (SotA) performance on the Vimeo-90K and UCF101 datasets across various motion magnitudes, exceeding prior methods. The primary advantage of MGMVFI is most evident on the challenging SNU-FILM benchmark. While it performs competitively on Easy and Medium splits, MG-MVFI establishes new SotA results on the Hard and Extreme splits, which feature significant, complex motion. On the Extreme split, MGMVFI outperforms the next-best S6-based model, VFIMamba, by 0.07 dB, confirming the robustness of our motion-aware approach for high-fidelity synthesis.

The eficiency and high-resolution scalability of MGMVFI are further underscored by the Xiph benchmark results in Table 2. The performance on Xiph 4K highlights the importance of trajectory-aware serialization. While fixed scanning orders often fail to capture temporal dependencies at high resolutions where related features are spatially distant, MGMVFI dynamically aligns the sequence with the true motion path to efectively preserve structural integrity and fine textures. Furthermore, despite the motion-guided serialization step, our model maintains a competitive inference runtime of 427.5 ms. This eficiency stems from the EDFFN module, which selectively refines high-frequency details in the frequency domain, allowing for a streamlined sequence-modeling backbone.

Table 1: Quantitative Evaluation across three benchmark datasets. The best and second best results are indicated by bold and underline.
<table><tr><td rowspan="2"></td><td rowspan="2">Train Dataset</td><td rowspan="2">Vimeo-90K [46] UCF101 [43]</td><td rowspan="2"></td><td colspan="4">SNU-FILM [4]</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>Extreme</td></tr><tr><td>DAIN [1]</td><td>V</td><td>34.71/0.9756</td><td>34.99/0.9683</td><td>39.73/0.9902</td><td>35.46/0.9780</td><td>30.17/0.9335</td><td>25.09/0.8584</td></tr><tr><td>AdaCof [23]</td><td>V</td><td>34.47/0.9730</td><td>34.90/0.9680</td><td>39.80/0.9900</td><td>35.05/0.9754</td><td>29.46/0.9244</td><td>24.31/0.8439</td></tr><tr><td>CAIN [4]</td><td>V</td><td>34.65/0.9730</td><td>34.91/0.9690</td><td>39.89/0.9900</td><td>35.61/0.9776</td><td>29.90/0.9292</td><td>24.78/0.8507</td></tr><tr><td>Softsplat [37]</td><td>V</td><td>36.13/0.9805</td><td>35.39/0.9697</td><td>40.26/0.9911</td><td>36.07/0.9798</td><td>30.53/0.9365</td><td>25.16/0.8604</td></tr><tr><td>XVFI [42]</td><td>V</td><td>35.09/0.9759</td><td>35.17/0.9685</td><td>39.93/0.9907</td><td>35.37/0.9782</td><td>29.58/0.9276</td><td>24.17/0.8450</td></tr><tr><td>M2M-VFI [13]</td><td>V</td><td>35.47/0.9778</td><td>35.28/0.9694</td><td>39.66/0.9904</td><td>35.74/0.9794</td><td>30.30/0.9360</td><td>25.08/0.8604</td></tr><tr><td>RIFE [14]</td><td>V</td><td>35.61/0.9779</td><td>35.28/0.9690</td><td>39.80/0.9903</td><td>35.76/0.9787</td><td>30.36/0.9351</td><td>25.27/0.8601</td></tr><tr><td>IFRNet-L [22]</td><td>V</td><td>36.20/0.9808</td><td>35.42/0.9698</td><td>40.10/0.9906</td><td>36.12/0.9797</td><td>30.63/0.9368</td><td>25.26/0.8609</td></tr><tr><td>AMT-L [26]</td><td>V</td><td>36.35/0.9815</td><td>35.39/0.9698</td><td></td><td>39.95/0.991336.09/0.9805</td><td>30.75/0.9384</td><td>25.41/0.8638</td></tr><tr><td>AMT-G [26]</td><td>V</td><td>36.53/0.9817</td><td>35.41/0.9699</td><td></td><td>39.88/0.991336.12/0.9805</td><td>30.78/0.9385</td><td>25.43/0.8644</td></tr><tr><td>SGM-VFI [27]</td><td>V</td><td>35.81/0.9793</td><td>35.34/0.9693</td><td>40.14/0.9907</td><td>36.06/0.9795</td><td>30.81/0.9375</td><td>25.59/0.8646</td></tr><tr><td>EMA-VFI-S [48]</td><td>V</td><td>36.07/0.9797</td><td>35.34/0.9696</td><td>39.81/0.9906</td><td>35.88/0.9795</td><td>30.69/0.9375</td><td>25.47/0.8632</td></tr><tr><td>EMA-VFI [48]</td><td>V</td><td>36.64/0.9819</td><td>35.48/0.9701</td><td>39.98/0.9910</td><td>36.09/0.9801</td><td>30.94/0.9392</td><td>25.69/0.8661</td></tr><tr><td>EDEN [50]</td><td>LAVIB</td><td>32.66/0.9587</td><td>34.92/0.9676</td><td>38.69/0.9879</td><td>34.66/0.9744</td><td>29.59/0.9279</td><td>24.94/0.8536</td></tr><tr><td>VFIMamba [47]</td><td>V+X</td><td>36.64/0.9819</td><td>35.45/0.9702</td><td>40.51/0.9912</td><td>36.40/0.9805</td><td>30.99/0.9401</td><td>25.79/0.8682</td></tr><tr><td>LCMamba(B) [18]</td><td>V</td><td>36.43/0.9813</td><td>35.39/0.9698</td><td>40.07/0.9909</td><td>36.08/0.9801</td><td>30.59/0.9375</td><td>25.35/0.8630</td></tr><tr><td>Ours</td><td>V</td><td>36.67/0.982035.53/0.9710</td><td></td><td>40.45/0.9910</td><td>36.38/ /0.9804</td><td>31.04/0.941125.86/0.8732</td><td></td></tr></table>

Table 2: Quantitative evaluation on Xiph [35] dataset and eficiency comparison. Runtimes (RT) are measured for a 2048×1024 resolution input on an RTX A6000.
<table><tr><td></td><td>Xiph 2K</td><td></td><td>Xiph 4K</td><td>Params(M)</td><td>RT(ms)</td></tr><tr><td>SGM-VFI</td><td>36.57</td><td>0.9424</td><td>34.23 0.9021</td><td>20.8</td><td>1168.9</td></tr><tr><td>EMA-VFI</td><td>36.74</td><td>0.9445</td><td>34.54 0.9054</td><td>65.6</td><td>332.3</td></tr><tr><td>EDEN</td><td>33.36</td><td>0.9043</td><td>26.84 0.7303</td><td>157.8</td><td>1304.4</td></tr><tr><td>VFIMamba</td><td>37.13</td><td>0.9451</td><td>34.62 0.9059</td><td>66.1</td><td>414.7</td></tr><tr><td>LCMamba(B)</td><td>36.90</td><td>0.9456</td><td>34.26 0.9040</td><td>16.2</td><td>785.6</td></tr><tr><td>Ours</td><td>37.15</td><td>0.9454</td><td>34.65 0.9059</td><td>78.4</td><td>427.5</td></tr></table>

## 5.2 Qualitative Evaluation

Figure 3 provides a qualitative comparison against leading VFI models using challenging scenes from the Vimeo90K and SNU-FILM Hard datasets. Scene 1 (columns 1–2) illustrates a rapid-motion example where the motion-agnostic scanning orders of prior S6-based models fail. Their fixed interleaved and Hilbertcurve scans provide semantically inconsistent patches, causing structural distortion. In contrast, our MGS module processes features along the motion path, providing a coherent sequence that simplifies the SSM’s objective and produces structurally sound, sharp interpolations. Scene 2 (column 3) highlights a dificult disocclusion scenario. Baselines, including prior S6 models and EMA-VFI, exhibit prominent ghosting and background blur due to insuficient information in newly revealed regions. Conversely, our results demonstrate the efectiveness of the Contextual Synthesis mechanism; the masked adapter synthesizes robust features from the surrounding valid spatial context. Combined with our modified EDFFN, the model handles information gaps to produce a crisp, artifact-free background. Scene 3 (column 4) demonstrates our model’s superiority in reconstructing high-frequency details. While baselines sufer from blur or washout artifacts that sacrifice faithfulness—as seen in the generative EDEN model—our model renders the sharpest textures. This performance is attributable to the EDFFN module, whose frequency-domain gating is uniquely suited for texture reconstruction, as validated by the 0.14 dB gain in Table 3.

![](images/d24cfb57822c885be6923ef30c669e1963a3ad8d3555e92873230c40ba9b560b.jpg)  
Fig. 3: Visual comparison of the interpolated results from the SNU-FILM [4] Hard and Vimeo90K [46] dataset. Best viewed zoomed in.

## 5.3 Ablation Study

Progressive Module Analysis. We conduct a progressive ablation study on the Vimeo-90K, UCF101, and SNU-FILM Hard datasets to validate the contributions of our proposed modules, with quantitative results and visual comparisons provided in Table 3 and Figure 4, respectively. We begin with a baseline model consisting of core SSM blocks, which struggles to handle dynamic movement, resulting in significant blur and ghosting artifacts. Integrating the modified EDFFN block yields a consistent improvement in sharpness across all benchmarks, raising the PSNR on the SNU-FILM Hard split to 29.80 dB. The addition of our core MGS method provides a substantial performance leap, increasing the SNU-FILM Hard PSNR by 0.44 dB to 30.24 dB. Visually, the +EDFFN and MGS configuration produces much clearer images with reduced structural artifacts, demonstrating the efectiveness of aligning the serialization order with estimated motion paths. Finally, introducing the Contextual Synthesis module yields the most significant gain, boosting the SNU-FILM Hard PSNR to 31.04 dB and the Vimeo-90K PSNR to 36.67 dB. This full model efectively reconstructs complex occlusions, such as a sheep’s head, while correctly eliminating disoccluded background artifacts (hind leg) as guided by the consistency mask $M _ { 1  0 }$

![](images/bfdd72093fb03e59f0651d6b9aeed04b755a3df0d04cd379aa6ead87af858971.jpg)  
Fig. 4: Visual comparison of the progressive module analysis from Vimeo-90K [46].

Table 3: Quantitative report of the progressive module analysis
<table><tr><td rowspan="2">Scan method</td><td colspan="4">Vimeo-90K [46] UCF101 [43] Hard [4]</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR SSIM</td><td>PSNR SSIM</td></tr><tr><td>SSM</td><td>35.96</td><td>0.9793</td><td>35.37 0.9699</td><td>29.71 0.9293</td></tr><tr><td>+ EDFFN</td><td>36.10</td><td>0.9797</td><td>35.38 0.9701</td><td>29.80 0.9300</td></tr><tr><td>+ MGS</td><td>36.26</td><td>0.9802</td><td>35.56 0.9713</td><td>30.24 0.9345</td></tr><tr><td>+ Cont. Synth. 36.67</td><td></td><td>0.9820</td><td>35.53 0.9710</td><td>31.04 0.9411</td></tr></table>

Table 4: Performance comparison with diferent optical flow backbones on SNU-FILM Hard [4]
<table><tr><td></td><td colspan="3">Flow Backbone|RT(ms) Params(M) |PSNR SSIM</td></tr><tr><td>LiteFlowNet</td><td>185.12</td><td>4.86</td><td>30.890.9405</td></tr><tr><td>PWC-Net</td><td>218.23</td><td>9.37</td><td>30.700.9388</td></tr><tr><td>RAFT - Small</td><td>277.88</td><td>0.99</td><td>30.920.9408</td></tr><tr><td>RAFT - Large</td><td>415.90</td><td>5.26</td><td>31.040.9411</td></tr></table>

Optical Flow Sensitivity Analysis. For our model, we employed the RAFT-Large model [45] to generate the flow maps $f _ { 0 \to 1 }$ and $f _ { 1 \to 0 } .$ In this section, we examine changes in performance when using flow estimators [15,44,45] at varying levels of accuracy and computational cost. As shown in Table 4, while using RAFT-Large yields the highest fidelity, our model maintains robust performance even with lightweight estimators such as RAFT-Small. The performance drop is marginal (approx. 0.12dB), indicating that MGMVFI is not reliant on perfect flow estimation. This is further supported by the performance of our model across diferent scan methods presented below (Table 5), indicating that the SSM itself possesses inherent motion-modeling capabilities. Furthermore, utilizing a smaller flow backbone significantly reduces the overall inference latency, ofering a flexible trade-of for resource-constrained applications.

Table 5: Comparison of MGS against previ ous scanning methods
<table><tr><td rowspan="3">Scan method</td><td colspan="2">Vimeo-90K [46]  $S \uparrow$ </td><td>Hard [4]</td></tr><tr><td>PSNR</td><td>SSIM PSNR</td><td>SSIM</td></tr><tr><td colspan="3">0.15 35.39</td></tr><tr><td>Sequential Spiral</td><td colspan="2">0.9791 0.12 35.17</td><td>29.45 0.9336 28.83 0.9302</td></tr><tr><td>Z-scan</td><td colspan="2">0.9781 0.29 35.50 0.9792</td><td>29.79 0.9335</td></tr><tr><td>Interleaved</td><td colspan="2">0.35 36.56</td><td>30.88 0.9390</td></tr><tr><td>Hilbert Curve</td><td colspan="2">0.9817 36.45 0.9813</td><td>30.57 0.9376</td></tr><tr><td>0.33</td><td colspan="2"></td><td></td></tr><tr><td>Ours (MGS)</td><td colspan="2">0.46 36.67 0.9820</td><td>31.04 0.9411</td></tr></table>

![](images/e71bb5946464f5b81871501d2d7852ccbe4ec06bb7324015b5940a2452c175f0.jpg)  
Fig. 5: Computation comparison on diferent input resolution

Scan methods. Table 5 evaluates our Motion-Guided Serialization (MGS) against conventional single-image (Sequential, Spiral, Z-scan) and video-specific (Interleaved, Hilbert Curve) scanning methods. To quantify the semantic continuity of the 1D serialized sequence, we introduce the Sequential Feature Similarity (S) metric (4) measured across 100 randomly sampled Vimeo90K images. Our results show that MGS significantly outperforms all baselines, particularly in complex motion scenarios. Specifically, MGS achieves a superior S of 0.46— a 31% improvement over the next best method (Interleaved). This semantic consistency directly translates into reconstruction accuracy, with MGS achieving a SotA PSNR of 31.04 dB on the SNU-FILM Hard benchmark. This highlights MGS’s semantic consistency, simplifying the sequence modeling task for the S6 block and enabling more accurate local feature reconstruction.

Computation scalability. Finally, the scalability of the S6 architecture is evaluated by measuring computational load against input resolution in Figure 5. While competitive at lower scales, the MGMVFI block exhibits a clear advantage at high resolutions, where its linear scaling results in substantially lower resource consumption than other architectures.

## 6 Limitations

Despite MGMVFI’s strong performance for SSM-based interpolation, several avenues for improvement remain. MGMVFI relies on external optical flow for MGS; flow errors can corrupt token ordering and propagate through the SSM. While contextual synthesis is beneficial, it does not address the core problem. Furthermore, MGS uses a single motion-adaptive serialization, which can underperform in scenes with multiple objects and small motions. We leave new research directions, such as joint flow refinement and multi-trajectory serialization, for future work.

## 7 Conclusion

We have proposed MGMVFI — Motion-Guided Mamba for Video Frame Interpolation — a motion-aware framework that redefines how S6 models handle VFI by moving beyond fixed scanning patterns. By implementing the MGS module, we leveraged optical flow to create a motion-adaptive serialization that preserved temporal coherence and streamlined interpolation. This was complemented by contextual synthesis to mitigate occlusions and an optimized EDFFN block that improved computational eficiency while restoring fine details. MG-MVFI achieved state-of-the-art performance on challenging benchmarks, proving the eficacy of motion-guided serialization. We anticipate that this motion-guided serialization approach will provide a scalable foundation for broader video synthesis and understanding tasks.

## Acknowledgements

This work was supported by Samsung Electronics Co., Ltd(IO251211-14315-01).

## References

1. Bao, W., Lai, W.S., Ma, C., Zhang, X., Gao, Z., Yang, M.H.: Depth-aware video frame interpolation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3703–3712 (2019)

2. Cheng, X., Chen, Z.: Video frame interpolation via deformable separable convolution. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 34, pp. 10607–10614 (2020)

3. Cheng, X., Chen, Z.: Multiple video frame interpolation via enhanced deformable separable convolution. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(10), 7029–7045 (2021)

4. Choi, M., Kim, H., Han, B., Xu, N., Lee, K.M.: Channel attention is all you need for video frame interpolation. In: Proceedings of the AAAI conference on artificial intelligence. vol. 34, pp. 10663–10671 (2020)

5. Danier, D., Zhang, F., Bull, D.: Ldmvfi: Video frame interpolation with latent difusion models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 1472–1480 (2024)

6. Flynn, J., Neulander, I., Philbin, J., Snavely, N.: Deepstereo: Learning to predict new views from the world’s imagery. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5515–5524 (2016)

7. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. In: First conference on language modeling (2024)

8. Gu, A., Goel, K., Ré, C.: Eficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396 (2021)

9. Gu, A., Johnson, I., Goel, K., Saab, K., Dao, T., Rudra, A., Ré, C.: Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems 34, 572–585 (2021)

10. Guo, H., Guo, Y., Zha, Y., Zhang, Y., Li, W., Dai, T., Xia, S.T., Li, Y.: Mambairv2: Attentive state space restoration. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 28124–28133 (2025)

11. Guo, H., Li, J., Dai, T., Ouyang, Z., Ren, X., Xia, S.T.: Mambair: A simple baseline for image restoration with state-space model. In: European conference on computer vision. pp. 222–241. Springer (2024)

12. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

13. Hu, P., Niklaus, S., Sclarof, S., Saenko, K.: Many-to-many splatting for eficient video frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3553–3562 (2022)

14. Huang, Z., Zhang, T., Heng, W., Shi, B., Zhou, S.: Real-time intermediate flow estimation for video frame interpolation. In: European Conference on Computer Vision. pp. 624–642. Springer (2022)

15. Hui, T.W., Tang, X., Loy, C.C.: Liteflownet: A lightweight convolutional neural network for optical flow estimation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 8981–8989 (2018)

16. Ibrahim, F., Liu, G., Wang, G.: A survey on mamba architecture for vision applications. arXiv preprint arXiv:2502.07161 (2025)

17. Jain, S., Watson, D., Tabellion, E., Poole, B., Kontkanen, J., et al.: Video interpolation with difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7341–7351 (2024)

18. Jeong, M.W., Rhee, C.E.: Lc-mamba: Local and continuous mamba with shifted windows for frame interpolation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17671–17681 (2025)

19. Jiang, H., Sun, D., Jampani, V., Yang, M.H., Learned-Miller, E., Kautz, J.: Super slomo: High quality estimation of multiple intermediate frames for video interpolation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 9000–9008 (2018)

20. Kalantari, N.K., Wang, T.C., Ramamoorthi, R.: Learning-based view synthesis for light field cameras. ACM Transactions on Graphics (TOG) 35(6), 1–10 (2016)

21. Kong, L., Dong, J., Tang, J., Yang, M.H., Pan, J.: Eficient visual state space model for image deblurring. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 12710–12719 (2025)

22. Kong, L., Jiang, B., Luo, D., Chu, W., Huang, X., Tai, Y., Wang, C., Yang, J.: Ifrnet: Intermediate feature refine network for eficient frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1969–1978 (2022)

23. Lee, H., Kim, T., Chung, T.y., Pak, D., Ban, Y., Lee, S.: Adacof: Adaptive collaboration of flows for video frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5316–5325 (2020)

24. Li, B., Xue, K., Liu, B., Lai, Y.K.: Bbdm: Image-to-image translation with brownian bridge difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition. pp. 1952–1961 (2023)

25. Li, K., Li, X., Wang, Y., He, Y., Wang, Y., Wang, L., Qiao, Y.: Videomamba: State space model for eficient video understanding. In: European conference on computer vision. pp. 237–255. Springer (2024)

26. Li, Z., Zhu, Z.L., Han, L.H., Hou, Q., Guo, C.L., Cheng, M.M.: Amt: All-pairs multi-field transforms for eficient frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9801– 9810 (2023)

27. Liu, C., Zhang, G., Zhao, R., Wang, L.: Sparse global matching for video frame interpolation with large motion. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19125–19134 (2024)

28. Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Jiao, J., Liu, Y.: Vmamba: Visual state space model. Advances in neural information processing systems 37, 103031–103063 (2024)

29. Liu, Z., Yeh, R.A., Tang, X., Liu, Y., Agarwala, A.: Video frame synthesis using deep voxel flow. In: Proceedings of the IEEE international conference on computer vision. pp. 4463–4471 (2017)

30. Long, G., Kneip, L., Alvarez, J.M., Li, H.: Learning image matching by simply watching video (2016)

31. Lu, L., Wu, R., Lin, H., Lu, J., Jia, J.: Video frame interpolation with transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3532–3542 (2022)

32. Lyu, Z., Chen, C.: Tlb-vfi: Temporal-aware latent brownian bridge difusion for video frame interpolation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16260–16269 (2025)

33. Lyu, Z., Li, M., Jiao, J., Chen, C.: Frame interpolation with consecutive brownian bridge difusion. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 3449–3458 (2024)

34. Ma, X., Ni, Z., Chen, X.: Tinyvim: Frequency decoupling for tiny hybrid vision mamba. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 23519–23529 (2025)

35. Montgomery, C., Lars, H.: Xiph. org video test media (derf’s collection). Online, https://media. xiph. org/video/derf 6, 1 (1994)

36. Niklaus, S., Liu, F.: Context-aware synthesis for video frame interpolation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1701–1710 (2018)

37. Niklaus, S., Liu, F.: Softmax splatting for video frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5437–5446 (2020)

38. Niklaus, S., Mai, L., Liu, F.: Video frame interpolation via adaptive separable convolution. In: Proceedings of the IEEE international conference on computer vision. pp. 261–270 (2017)

39. Park, J., Ko, K., Lee, C., Kim, C.S.: Bmbc: Bilateral motion estimation with bilateral cost volume for video interpolation. In: Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XIV 16. pp. 109–125. Springer (2020)

40. Park, J., Lee, C., Kim, C.S.: Asymmetric bilateral motion estimation for video frame interpolation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 14539–14548 (2021)

41. Shi, Z., Xu, X., Liu, X., Chen, J., Yang, M.H.: Video frame interpolation transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17482–17491 (2022)

42. Sim, H., Oh, J., Kim, M.: Xvfi: extreme video frame interpolation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 14489–14498 (2021)

43. Soomro, K., Zamir, A.R., Shah, M.: Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402 (2012)

44. Sun, D., Yang, X., Liu, M.Y., Kautz, J.: Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 8934–8943 (2018)

45. Teed, Z., Deng, J.: Raft: Recurrent all-pairs field transforms for optical flow. In: European conference on computer vision. pp. 402–419. Springer (2020)

46. Xue, T., Chen, B., Wu, J., Wei, D., Freeman, W.T.: Video enhancement with task-oriented flow. International Journal of Computer Vision 127(8), 1106–1125 (2019)

47. Zhang, G., Liu, C., Cui, Y., Zhao, X., Ma, K., Wang, L.: Vfimamba: Video frame interpolation with state space models. Advances in Neural Information Processing Systems 37, 107225–107248 (2024)

48. Zhang, G., Zhu, Y., Wang, H., Chen, Y., Wu, G., Wang, L.: Extracting motion and appearance via inter-frame attention for eficient video frame interpolation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5682–5692 (2023)

49. Zhang, Z., Liu, A., Reid, I., Hartley, R., Zhuang, B., Tang, H.: Motion mamba: Eficient and long sequence motion generation. In: European Conference on Computer Vision. pp. 265–282. Springer (2024)

50. Zhang, Z., Chen, H., Zhao, H., Lu, G., Fu, Y., Xu, H., Wu, Z.: Eden: Enhanced difusion for high-quality large-motion video frame interpolation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2105–2115 (2025)

51. Zhou, C., Wu, T., Liu, W., Wu, X., Fu, Y.: Mvssm: Motion-aware visual state space model for eficient video deblurring. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4855–4865 (2026)

52. Zhou, T., Tulsiani, S., Sun, W., Malik, J., Efros, A.A.: View synthesis by appearance flow. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14. pp. 286–301. Springer (2016)