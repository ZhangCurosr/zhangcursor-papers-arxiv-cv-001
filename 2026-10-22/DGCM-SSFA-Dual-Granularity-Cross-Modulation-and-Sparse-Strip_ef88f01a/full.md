# DGCM-SSFA: Dual-Granularity Cross-Modulation and Sparse Strip Frequency Attention for Efficient 3T-to-7T MRI Synthesis

Zhichun Fan, Zhiqiang Chen, Lanling Liu, Lingjun Liu, Zhonghua Xie<sup>∗</sup> School of Computer Science and Engineering, Huizhou University, Huizhou, China

Abstract—Synthesizing high-fidelity 7T MRI from routinely acquired 3T scans offers a practical pathway to enhanced neuroimaging without the prohibitive cost of ultra-high-field scanners. While existing deep learning approaches have achieved promising results, state-of-the-art methods typically rely on either large-scale architectures exceeding 45 million parameters or computationally intensive ensemble strategies, limiting their practicality in resource-constrained clinical environments. In this work, we propose a compact multi-domain attention framework, termed DGCM-SSFA, that achieves competitive synthesis quality with only 11M parameters. The proposed architecture integrates two complementary modules into a U-Net Transformer backbone. First, a Dual-Granularity Cross-Modulated Feature Modulator (DGCM) is embedded at multiple hierarchical levels to enhance local edge and texture details through Sobel-guided spatial attention and depthwise frequency-aware channel attention with cross-branch gating. Second, a Sparse Strip Frequency Attention (SSFA) module at the bottleneck captures global spatial context via strip-pooled attention with near-linear complexity, while simultaneously performing FFT-based spectral modulation in the frequency domain. These two modules operate synergistically with the backbone’s native channel attention, forming a triplegranularity feature refinement pipeline. Extensive experiments on the UNC and BNU 3T-to-7T MRI datasets across T1w and T2w modalities demonstrate that DGCM-SSFA achieves strong structural fidelity, attaining an SSIM of 0.8431 on BNU T2w— approaching the best benchmark value of 0.8624—and 0.8100 on BNU T1w, while using substantially fewer parameters than Transformer-based alternatives such as ResViT (123.4M) and FS-RWKV (45.4M). These results validate that carefully designed multi-domain attention provides a practical and parameterefficient solution for clinical MRI synthesis.

Index Terms—MRI synthesis, 3T-to-7T translation, multidomain attention, frequency-aware modulation, sparse strip attention, compact network, medical image synthesis

## I. INTRODUCTION

Magnetic resonance imaging at 7 Tesla (7T) represents the current gold standard for structural neuroimaging, offering an order-of-magnitude gain in signal-to-noise ratio and spatial resolution over conventional 3T scanners. This elevated tissue contrast enables the delineation of subcortical nuclei, cortical microarchitecture, and subtle white matter lesions that remain invisible at lower field strengths. Despite these compelling diagnostic advantages, 7T adoption is severely limited by hardware costs exceeding several million dollars, specialized site requirements, and stringent safety protocols—factors that collectively restrict access to a handful of elite research centers worldwide [1].

Synthesizing 7T-equivalent images from routine 3T acquisitions has therefore attracted substantial interest [2]–[4]. The task is to learn a mapping that recovers the high-resolution anatomical details latent in clinical scans, yet this mapping must capture fine-grained edge structures at tissue boundaries, long-range anatomical dependencies across entire brain regions, and subtle textural variations, which remains an open challenge.

Early approaches to this problem employed deep convolutional architectures [7]–[9] and generative adversarial formulations [10]–[12]. These methods exhibit a characteristic tradeoff: CNNs faithfully reconstruct local intensity patterns but fail to model dependencies beyond their limited receptive field, whereas GANs can hallucinate visually convincing textures at the cost of introducing spurious anatomical features that undermine diagnostic trustworthiness. More recent Transformerbased designs [13]–[15] address the global context deficit through self-attention, but their quadratic complexity forces aggressive downsampling that discards precisely the highfrequency details most critical for clinical interpretation. The latest generation of state space models [16], [17] partially resolves this tension by offering linear-complexity global modeling; however, they operate predominantly in the spatial domain and provide no explicit mechanism for frequencyselective feature refinement.

A deeper issue cuts across these paradigms. Anatomical structures in MRI span multiple scales and exhibit distinct signatures in both the spatial and frequency domains. Edge contours at tissue boundaries are best captured by local gradient operators, global intensity distributions across an entire slice are naturally modeled by long-range attention, and fine periodic textures are most compactly represented in the Fourier domain. Yet no existing cross-field synthesis framework integrates all three forms of attention—local edgeguided, global spatial, and frequency-domain—within a single unified architecture. This motivates the central premise of our work: that a carefully orchestrated synergy of complementary attention mechanisms can achieve competitive synthesis fidelity without the massive parameter counts of monolithic

Transformer or state space models.

We propose DGCM-SSFA, a compact multi-domain attention network built upon a U-Net Transformer backbone. Its first component (DGCM) is deployed at multiple encoder levels and combines an edge-aware spatial branch with a frequency-aware channel branch; the two branches gate each other [18]. The second component (SSFA) operates at the bottleneck, compressing the key-value token space from $H \times W$ to $H { + } W$ for near-linear global attention [19], with a parallel FFT branch that performs spectral modulation of the magnitude component. Both modules are detailed in Section III.

The principal contributions of this work are:

1) A multi-domain attention architecture, DGCM-SSFA, that integrates dual-granularity cross-modulated feature enhancement and sparse strip frequency attention into a U-Net Transformer for 3T-to-7T MRI synthesis, operating at 11M parameters.

2) The DGCM module, which unifies Sobel-structured edge priors, depthwise spectral estimation, and spatialchannel gating to amplify local anatomical detail.

3) The SSFA module, which achieves global context modeling with near-linear complexity through strip-pooled attention and incorporates FFT-based spectral modulation for frequency-aware refinement.

4) Extensive experiments on the UNC and BNU datasets across T1w and T2w modalities, demonstrating that DGCM-SSFA achieves strong structural fidelity— attaining an SSIM of 0.8431 on BNU T2w, approaching the best benchmark value of 0.8624 while using only 11M parameters, substantially fewer than Transformerbased alternatives.

## II. RELATED WORK

## A. Deep Learning for Cross-Field MRI Synthesis

Mapping 3T MRI to its 7T counterpart lies at the intersection of super-resolution and cross-modal translation. Early approaches used convolutional architectures trained with pixelwise regression [7], [8], achieving reasonable intensity fidelity but producing over-smoothed outputs that lacked crisp anatomical boundaries. Adversarial training [10]–[12] substantially improved perceptual quality, but introduces a well-documented risk of hallucinating spurious anatomical features [10], [12]—a dangerous failure mode in diagnostic imaging.

Transformer-based architectures [6], [13], [14] address the smoothness issue through self-attention, capturing long-range dependencies that CNNs miss, but ResViT [6] combines residual convolutional blocks with vision transformers at the cost of 123M parameters. State space formulations [2], [16], [17] offer a more efficient alternative, with FS-RWKV [2] showing that linear-complexity recurrent architectures can approach Transformer-level quality. Despite these advances, the dominant design philosophy remains one of scaling, leaving a gap for compact architectures that achieve competitive quality through structural innovation.

## B. Attention Mechanisms in Image Restoration

Channel attention, as popularized by SENet [20] and refined by CBAM [21], recalibrates feature responses by modeling interdependencies between channels through global pooling operations. Spatial attention [22] complements channel attention by generating a 2D weight map that highlights informative regions; however, standard spatial attention based on max- and average-pooling statistics lacks sensitivity to structural edges, motivating the use of explicit gradient operators such as Sobel filters [23] to inject edge-awareness.

A parallel line of work explores frequency-domain processing for image restoration [24]–[27]. Different degradation types manifest in distinct frequency bands: high-frequency components carry edge and texture information, while lowfrequency components encode global illumination and contrast. Fast Fourier Convolution [24] performs convolution in the complex frequency domain through a dual-branch design, whereas DCT-based learning [26] applies channel selection for efficient spectral processing.

For global context aggregation, strip pooling [19] offers an attractive compromise between the limited receptive field of local operations and the quadratic cost of dense self-attention by compressing the attention token space from O(HW) to $O ( H + W )$ . This technique is increasingly adopted in restoration tasks where global illumination and contrast consistency are critical.

## C. Multi-Domain and Multi-Granularity Feature Fusion

A recurring theme in recent restoration literature is combining complementary attention mechanisms. CBAM [21] pioneered this idea by stacking channel and spatial attention, and subsequent work extended it to the frequency domain [25], [26]. Others use cross-modal gating [18], [28], where one branch modulates another. Despite these advances, existing MRI synthesis frameworks typically employ at most two attention types and rarely integrate frequency-domain processing or edge-aware spatial attention within a unified design. Our work addresses this gap by orchestrating three complementary attention mechanisms—local edge-guided spatial, global strip-pooled spatial, and frequency-domain spectral—coupled through cross-branch and cross-domain gating.

## III. METHOD

## A. Overall Pipeline

The overall architecture of DGCM-SSFA is illustrated in Fig. 1(a). Given a degraded 3T MRI image $I \in \mathbb { R } ^ { H \times W \times C } ,$ a convolutional layer first extracts shallow features $F _ { 0 } \in$ $\mathbb { R } ^ { H \times W \times C }$ . The encoder then progressively downsamples the feature maps through a stack of FFN modules and convolutional layers, capturing hierarchical representations at multiple scales. A DGCM module is inserted after each encoder level to perform dual-granularity cross-modulated feature refinement, enhancing local edge and texture details at progressively deeper semantic levels.

At the bottleneck, the most compact latent representation is processed by the SSFA module, which captures global spatial context via strip-pooled attention and performs FFT-based spectral modulation in the frequency domain. The decoder symmetrically upsamples the features back to the original resolution through Transformer blocks with self-attention. A skip connection from the encoder output is concatenated with the decoder features to preserve low-level structural information.

![](images/bd38d08d8c8d669c412cc5dd3a4b9558507aa7fd5247ad1ffca076ca7771b09a.jpg)  
Fig. 1. Overall architecture of the proposed DGCM-SSFA. (a) The framework adopts a U-Net-style encoder-decoder backbone. The encoder stacks FFN and Conv layers for hierarchical feature extraction. DGCM modules are inserted after each encoder level and after the refinement stage for local detail enhancement. A single SSFA module is deployed at the bottleneck for global spatial-frequency attention. A global residual connection produces the final output. (b) DGCM consists of an edge-aware spatial branch and a frequency-aware channel branch with cross-modulation fusion. (c) SSFA combines sparse strip attention (SSA) and frequency spectral attention (FSA) with cross-gated fusion.

The decoded features are further processed by a refinement stage consisting of additional Transformer blocks, followed by a fourth DGCM for final detail enhancement. The refined features are projected to the residual image $R \in \mathbb { R } ^ { H \times W \times C }$ via a convolutional layer. The final restored 7T-like MRI image is obtained through a global residual connection: <sup>ˆ</sup>I = I + R.

The encoder and decoder consist of repeated Transformer blocks, each with a Multi-head Self-Attention (MSA) module and a Feed-Forward Network (FFN) applying a pointwise convolution with GELU activation. The encoder downsamples via stride-2 convolution across three levels, while the decoder symmetrically upsamples through bilinear interpolation. Skip connections concatenate low-level encoder features with highlevel decoder features, preserving fine-grained spatial information lost during compression.

The two core modules, DGCM and SSFA, are detailed in Fig. 1(b) and (c), respectively. DGCM employs an edge-aware spatial branch and a frequency-aware channel branch with cross-modulation gating to enhance local feature discrimination. SSFA combines sparse strip attention with FFT-based spectral attention under a cross-gated fusion mechanism for efficient global context modeling. These two modules operate in a complementary manner: DGCM focuses on local edge and texture refinement at multiple hierarchical levels, while SSFA captures long-range dependencies at the bottleneck.

## B. DGCM: Dual-Granularity Cross-Modulated Feature Modulator

The DGCM module, illustrated in Fig. 1(b), is deployed at four hierarchical positions: after the first, second, and third encoder levels, and after the refinement stage. At each position, DGCM takes a feature map $F _ { \mathrm { i n } } \in \mathbb { R } ^ { C \times H \times W }$ and produces a refined output $F _ { \mathrm { o u t } }$ of identical dimensions, following a localglobal residual design:

$$
F _ { \mathrm { o u t } } = F _ { \mathrm { i n } } + \mathcal { R } ( F _ { \mathrm { f u s e d } } ) ,\tag{1}
$$

where R denotes a lightweight residual refinement block composed of two convolutional layers with ReLU activation.

1) Edge-Aware Spatial Branch: The spatial branch constructs a three-channel descriptor from the input feature map:

a channel-wise maximum map highlighting strong activations, a channel-wise mean map capturing overall intensity distribution, and a gradient magnitude map computed via fixed Sobel operators applied to the mean map. The Sobel kernels $G _ { x }$ and $G _ { y }$ are registered as non-trainable buffers, ensuring zero additional parameter cost for edge extraction. The gradient magnitude is computed as:

$$
G _ { \mathrm { m a g } } = \sqrt { ( G _ { x } * F _ { \mathrm { m e a n } } ) ^ { 2 } + ( G _ { y } * F _ { \mathrm { m e a n } } ) ^ { 2 } + \epsilon } ,\tag{2}
$$

where ∗ denotes convolution and ϵ is a small constant for numerical stability. The three-channel descriptor is passed through a lightweight convolutional head consisting of two $3 \times$ 3 convolutions with an intermediate ReLU activation, producing a single-channel spatial attention map $W _ { s } \in [ 0 , 1 ] ^ { \hat { 1 } \times H \times W }$ via sigmoid activation. The spatially modulated output is:

$$
F _ { \mathrm { s p } } = W _ { s } \odot F _ { \mathrm { i n } } ,\tag{3}
$$

where ⊙ denotes element-wise multiplication with broadcasting.

2) Frequency-Aware Channel Branch: The channel branch first transforms the input through two consecutive $1 \times 1$ convolutions to obtain an intermediate representation $F _ { d } .$ Three statistical descriptors are then computed: a max-pooled vector $z _ { \operatorname* { m a x } } \in \mathbb { R } ^ { C \times 1 \times i }$ , an average-pooled vector $z _ { \mathrm { a v g } } ,$ , and a frequency-aware vector $z _ { \mathrm { f r e q } }$ obtained by applying a depthwise $3 \times 3$ convolution to $F _ { d } ,$ , modulating the result with the spatial attention map $W _ { s }$ , and globally average-pooling the absolute response:

$$
z _ { \mathrm { f r e q } } = \mathrm { G A P } \big ( | \operatorname { D W C o n v } _ { 3 \times 3 } ( F _ { d } ) \odot W _ { s } | \big ) .\tag{4}
$$

The three descriptors are concatenated and processed by a twolayer MLP with a reduction ratio of 4, producing a channel attention vector $W _ { c } \in [ 0 , 1 ] ^ { C \times 1 \times 1 }$ . The channel-modulated output is:

$$
F _ { \mathrm { c h } } = W _ { c } \odot F _ { \mathrm { i n } } .\tag{5}
$$

3) Cross-Modulation Fusion: Rather than simply summing the two branch outputs, DGCM employs a learnable gated fusion mechanism. The spatial and channel outputs are concatenated and processed by two parallel pathways: a fusion pathway that generates an integrated feature map $F _ { \mathrm { c o n v } }$ through a $1 \times 1 { - } 3 \times 3$ convolution sequence, and a gating pathway that produces a sigmoid-gated mask $G \in [ 0 , 1 ] ^ { \overline { { C } } \times H \times \overline { { W } } }$

$$
\begin{array} { r } { F _ { \mathrm { f u s e d } } = G \odot \mathrm { C o n v _ { 3 \times 3 } } \big ( \mathrm { C o n v _ { 1 \times 1 } } ( [ F _ { \mathrm { s p } } , F _ { \mathrm { c h } } ] ) \big ) } \\ { + ( 1 - G ) \odot \frac { 1 } { 2 } ( F _ { \mathrm { s p } } + F _ { \mathrm { c h } } ) . } \end{array}\tag{6}
$$

This formulation allows the network to dynamically blend the fused representation with the direct average of the two branches, providing a soft selection mechanism that adapts to the local feature characteristics at each spatial position.

## C. SSFA: Sparse Strip Frequency Attention

While DGCM excels at local detail enhancement, capturing long-range anatomical dependencies—such as global intensity gradients across the brain, bilateral symmetry, and large-scale tissue contrast—requires a global attention mechanism. SSFA, deployed at the bottleneck (dimension 192), addresses this through two parallel pathways operating in the spatial and frequency domains, coupled by cross-domain gating.

1) Sparse Strip Spatial Attention: Standard dense selfattention incurs $\bar { \mathcal { O } } ( \bar { H ^ { 2 } } W ^ { 2 } )$ complexity, which is prohibitive for high-resolution feature maps. SSFA adopts strip pooling to compress the key and value token space from HW to $H + W$ . Given the query, key, and value projections $Q , K , V ~ \in ~ \mathbb { R } ^ { B \times \operatorname { h e a d } \times H W \times d }$ , the key and value tensors are reshaped to 2D spatial form and average-pooled along each axis independently:

$$
K _ { h } = \mathrm { A v g P o o l } _ { W } ( K _ { 2 D } ) , \quad K _ { w } = \mathrm { A v g P o o l } _ { H } ( K _ { 2 D } ) ,\tag{7}
$$

$$
K _ { \mathrm { s t r i p } } = \operatorname { C o n c a t } ( K _ { h } , K _ { w } ) \in \mathbb { R } ^ { B \times \operatorname { h e a d } \times ( H + W ) \times d } ,\tag{8}
$$

with analogous operations for V. The attention is then computed as:

$$
\mathrm { A t t n } _ { \mathrm { s t r i p } } = \mathrm { S o f t m a x } \left( \frac { Q K _ { \mathrm { s t r i p } } ^ { \mathsf { T } } } { \sqrt { d } } \right) V _ { \mathrm { s t r i p } } ,\tag{9}
$$

yielding $\mathcal { O } ( H W ( H + W ) )$ complexity—a substantial reduction from the quadratic cost of dense attention, particularly at the bottleneck where spatial dimensions are smallest.

2) Frequency Spectral Attention: In parallel, the input feature map is transformed to the frequency domain via the 2D real-valued Fast Fourier Transform (rFFT). The resulting complex spectrum is decomposed into magnitude $M = | { \mathcal { F } } ( F ) |$ and phase $\Phi = \angle { \mathcal { F } } ( F )$ . A channel attention mechanism is applied exclusively to the magnitude component:

$$
M ^ { \prime } = M \odot \sigma \bigl ( \mathrm { M L P } ( \mathrm { G A P } ( M ) ) \bigr ) ,\tag{10}
$$

where σ denotes the sigmoid function and the MLP consists of two $1 \times 1$ convolutions with a reduction ratio of 4. The phase component is preserved unchanged, as it encodes the structural positioning of features. The refined spectrum is reconstructed via inverse rFFT: $F _ { \mathrm { f r e q } } ^ { \prime } = \mathcal { F } ^ { - 1 } ( M ^ { \prime } \cdot \overline { { e ^ { j \Phi } } } )$ . This design enables the network to selectively amplify or suppress specific frequency bands, enhancing high-frequency tissue boundaries while preserving low-frequency contrast.

3) Cross-Domain Gating: The spatial and frequency pathways are not independent. A cross-domain gating mechanism enables mutual refinement: the spatial output is gated by a channel-wise vector derived from the frequency output, and vice versa:

$$
\begin{array} { r } { F _ { \mathrm { s p } } ^ { \prime } = F _ { \mathrm { s p } } \odot \sigma \big ( \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { G A P } ( F _ { \mathrm { f r e q } } ^ { \prime } ) ) \big ) , } \end{array}\tag{11}
$$

$$
F _ { \mathrm { f r e q } } ^ { \prime \prime } = F _ { \mathrm { f r e q } } ^ { \prime } \odot \sigma \big ( \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { G A P } ( F _ { \mathrm { s p } } ) ) \big ) .\tag{12}
$$

The gated outputs are concatenated and fused through a $1 \times 1 -$ $3 \times 3$ convolution sequence, followed by a final $1 \times 1 { - } 1 \times 1$ projection and a global residual addition:

$$
F _ { \mathrm { S S F A } } = F _ { \mathrm { i n } } + F _ { \mathrm { p r o j } } .\tag{13}
$$

## D. Training Objective

The model is trained with a simple L1 reconstruction loss between the synthesized 7T image $\hat { I } _ { 7 T }$ and the ground truth I<sub>7T</sub>:

$$
\mathcal { L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert \hat { I } _ { 7 T } ^ { ( i ) } - I _ { 7 T } ^ { ( i ) } \rVert _ { 1 } ,\tag{14}
$$

where N denotes the number of pixels. The L1 loss is chosen over L2 for its robustness to outliers and its tendency to produce sharper outputs in image restoration tasks [29]. The global residual formulation—where the network predicts the difference between 3T and 7T rather than the 7T image directly—further stabilizes training.

## E. Implementation Details

DGCM-SSFA is implemented in PyTorch with the BasicSR toolbox [30], and trained with AdamW [31] (initial learning rate $3 ~ \times ~ 1 0 ^ { - 4 }$ , weight decay $1 \times 1 0 ^ { - 4 } )$ under a cosine annealing schedule [32]. Training runs for 30,000 iterations with a two-stage progressive patch size (128 × 128 for the first 10,000 iterations, then 256 × 256), a batch size of 8 per GPU on two NVIDIA RTX 5090 GPUs, and standard geometric augmentation (random flipping, 90-degree rotation, and cropping).

## IV. EXPERIMENTS

## A. Datasets and Preprocessing

We evaluate on two publicly available paired 3T–7T MRI datasets: UNC and BNU [2]. UNC provides paired 3T (Siemens Prisma) and 7T (Siemens Terra) brain scans from 10 participants (T1w: MPRAGE/MP2RAGE; T2w: SPACE/TSE); BNU provides 20 subjects scanned on 3T Siemens Prisma and 7T Siemens Magnetom (T1w: MPRAGE/3D MPRAGE; T2w: TSE/3D SPACE). Following standard preprocessing, all volumes are resampled to isotropic 1.0 mm, skull-stripped, rigidly registered across field strengths using FSL FLIRT [33], and normalized to [0, 1]. Each volume is sliced along the axial plane into 256 × 256 2D images.

## B. Evaluation Metrics

We adopt three widely used quantitative metrics for image synthesis quality assessment: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM) [34] computed with an 11×11 Gaussian kernel (σ = 1.5), and Root Mean Square Error (RMSE). PSNR and RMSE capture pixellevel reconstruction fidelity, while SSIM reflects perceptual structural consistency. Higher PSNR and SSIM values indicate better performance, whereas lower RMSE denotes superior reconstruction accuracy.

## C. Comparison with State-of-the-Art Methods

To comprehensively assess the synthesis capability of DGCM-SSFA, we compare it against eight representative methods spanning four architectural paradigms. The CNNbased category includes DDCNN [8] and WATNet [35];

GAN-based methods comprise CycleGAN [10], pix2pix [11], and RegGAN [12]; Transformer-based approaches cover ResViT [6] and PTNet [5]; and the SSM-based baseline FS-RWKV [2] represents the latest state-space model. All methods are trained and evaluated under a unified protocol on the same data splits.

Quantitative Results. Table I reports the numerical comparison across four modality-site combinations. Our DGCM-SSFA achieves competitive performance with only 11M parameters. On the BNU T2w dataset, it attains the secondhighest SSIM of 0.8431, trailing only FS-RWKV (0.8624), demonstrating the effectiveness of the multi-granularity attention design in preserving structural fidelity. On the BNU T1w dataset, the SSIM of 0.8100 ranks second only to FS-RWKV (0.8388). In terms of parameter efficiency, DGCM-SSFA contains 11M parameters—significantly fewer than FS-RWKV (45.40M), ResViT (123.44M), pix2pix (57.18M), and PTNet (28.13M)—while maintaining competitive or superior structural similarity.

Qualitative Analysis. Visual comparisons of synthesized 7T-like images are presented in Fig. 2. DGCM-SSFA recovers sharper tissue boundaries and finer anatomical textures compared to competing methods, particularly in cortical gray matter and subcortical regions where high-frequency details are critical.

## D. Ablation Studies

We conduct ablation experiments on the BNU T2w dataset to validate the contribution of each core component; quantitative results are summarized in Table II and visualized in Fig. 3.

Removing all DGCM modules yields a PSNR of 25.88 dB and SSIM of 0.8206, degradations of 0.39 dB and 0.0225, confirming that the edge-aware spatial and frequency-aware channel branches jointly preserve fine anatomical boundaries. Removing SSFA gives 26.02 dB and 0.8270 (drops of 0.25 dB and 0.0161), with the degradation most pronounced in regions of elongated anatomical structures, validating the contribution of strip-pooled global attention and FFT-based spectral modulation. With both modules removed, the backbone alone reaches 25.66 dB and 0.8048. The individual contributions (0.39 + 0.25 = 0.64 dB) closely match the joint contribution (0.61 dB), leaving only 0.03 dB of functional overlap and indicating that the two modules operate on largely orthogonal feature spaces.

## V. DISCUSSION AND CONCLUSION

DGCM-SSFA achieves competitive synthesis quality with only 11M parameters, showing that carefully designed multidomain attention can rival larger architectures. On BNU T2w it attains the second-highest SSIM of 0.8431, approaching FS-RWKV (0.8624) while using only 24% of its parameters. The ablation study further reveals that DGCM and SSFA operate on largely orthogonal feature spaces with only 0.03 dB of functional overlap. Despite strong structural fidelity, the model yields moderate PSNR, reflecting a preference for anatomical boundary preservation over pixel-wise intensity accuracy— a trade-off that is clinically acceptable given that diagnostic confidence depends primarily on the clear delineation of tissue interfaces. Several limitations merit future investigation: the 2D slice-wise formulation does not exploit inter-slice anatomical consistency, and a volumetric 3D extension could improve both fidelity metrics [37]; hybrid loss functions combining adversarial, perceptual [36], and frequency-domain objectives [3] may better balance structural and pixel-wise quality; and validation on pathological cases remains an important step toward clinical translation. In summary, DGCM-SSFA offers a practical and parameter-efficient solution for high-fidelity 3Tto-7T MRI synthesis.

Error Map (GT - Ours)  
TABLE I  
QUANTITATIVE COMPARISON ON MRI CROSS-MODALITY SYNTHESIS (3T → 7T)
<table><tr><td rowspan="2">Method</td><td colspan="3">UNC T1w</td><td colspan="3">UNC T2w</td><td colspan="3">BNU T1w</td><td colspan="3">BNU T2w</td><td rowspan="2">Params (M)</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>RMSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>RMSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>RMSE↓</td><td>PSNR↑</td><td>SSIM↑</td><td>RMSE↓</td></tr><tr><td>CycleGAN</td><td>19.98</td><td>0.6789</td><td>0.1008</td><td>24.88</td><td>0.7538</td><td>0.0586</td><td>21.93</td><td>0.7881</td><td>0.0807</td><td>26.30</td><td>0.8053</td><td>0.0493</td><td>28.29</td></tr><tr><td>pix2pix</td><td>19.96</td><td>0.6847</td><td>0.1015</td><td>24.74</td><td>0.7550</td><td>0.0591</td><td>22.74</td><td>0.8102</td><td>0.0739</td><td>26.47</td><td>0.8182</td><td>0.0481</td><td>57.18</td></tr><tr><td>DDĆNN</td><td>20.75</td><td>0.6412</td><td>0.0924</td><td>24.21</td><td>0.7286</td><td>0.0640</td><td>23.02</td><td>0.7732</td><td>0.0717</td><td>26.88</td><td>0.8229</td><td>0.0463</td><td>0.29</td></tr><tr><td>WATNet</td><td>20.17</td><td>0.6753</td><td>0.0990</td><td>23.07</td><td>0.6698</td><td>0.0719</td><td>21.13</td><td>0.7319</td><td>0.0886</td><td>26.13</td><td>0.7625</td><td>0.0503</td><td>1.35</td></tr><tr><td>RegGAN</td><td>19.46</td><td>0.6423</td><td>0.1073</td><td>23.97</td><td>0.6994</td><td>0.0654</td><td>22.10</td><td>0.7897</td><td>0.0793</td><td>25.61</td><td>0.7703</td><td>0.0531</td><td>2.76</td></tr><tr><td>ResViT</td><td>20.14</td><td>0.6696</td><td>0.0994</td><td>23.47</td><td>0.7140</td><td>0.0687</td><td>22.78</td><td>0.8069</td><td>0.0733</td><td>26.58</td><td>0.8251</td><td>0.0477</td><td>123.44</td></tr><tr><td>PTNet</td><td>20.82</td><td>0.7142</td><td>0.0916</td><td>24.66</td><td>0.7495</td><td>0.0602</td><td>22.98</td><td>0.7903</td><td>0.0717</td><td>27.34</td><td>0.8242</td><td>0.0439</td><td>28.13</td></tr><tr><td>FS-RWKV</td><td>21.01</td><td>0.7258</td><td>0.0898</td><td>25.31</td><td>0.7807</td><td>0.0565</td><td>23.36</td><td>0.8388</td><td>0.0689</td><td>27.49</td><td>0.8624</td><td>0.0431</td><td>45.40</td></tr><tr><td>Ours</td><td>20.21</td><td>0.6924</td><td>0.0972</td><td>24.50</td><td>0.7505</td><td>0.0614</td><td>22.39</td><td>0.8100</td><td>0.0765</td><td>26.27</td><td>0.8431</td><td>0.0496</td><td>11</td></tr></table>

Input (3T)  
GT (7T)  
CycleGAN  
DDCNN  
pix2pix  
FS-RWKV  
Ours  
![](images/911f988dc507a6874b40c2fcdfcc6bb3e6781eac1b6219a1bed26de58da1a668.jpg)  
Fig. 2. Qualitative comparison of synthesized 3T-to-7T MRI on four datasets (UNC T1w, UNC T2w, BNU T1w, BNU T2w). From left to right: 3T input, 7T ground truth, CycleGAN, DDCNN, pix2pix, FS-RWKV, the proposed DGCM-SSFA, and error map (error magnitude relative to GT). Yellow and cya boxes in the main image denote regions magnified below.

TABLE II  
ABLATION STUDY ON THE KEY COMPONENTS OF DGCM-SSFA. ALL MODELS ARE TRAINED WITHOUT ENSEMBLE AND EVALUATED ON BNU T2W. THE BEST RESULT IN EACH COLUMN IS HIGHLIGHTED IN BOLD.
<table><tr><td>Variant</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>Full DGCM-SSFA</td><td>26.27</td><td>0.8431</td></tr><tr><td>w/o DGCM</td><td>25.88</td><td>0.8206</td></tr><tr><td>w/o SSFA</td><td>26.02</td><td>0.8270</td></tr><tr><td>w/o Both</td><td>25.66</td><td>0.8048</td></tr></table>

![](images/3de055313d11d62c87cb2670738d4e461fe13814ae8c36ef985da1d0f4d3dd9b.jpg)

![](images/50c2ea97e1b1cd999172883079150d16db1c4b2b31e6047acadfd1c3454e63d6.jpg)  
Fig. 3. Ablation study on the key components of DGCM-SSFA. PSNR (left) and SSIM (right) are reported for four variants. All models are trained without ensemble and evaluated on BNU T2w.

## REFERENCES

[1] S. M. Seyedsaadat, S. N. Seyed Saadat, V. Patel, S. J. S. Sandhu, D. Paech, A. Desai, X. Zhou, S. Tao, V. Gupta, and E. H. Middlebrooks, “Ultrahigh-field 7-T MRI in neuroradiology: A comprehensive clinical review,” RadioGraphics, vol. 46, no. 1, e250194, 2026.

[2] Y. Lei, Z. Li, Y. Liu, X. Chen, and C.-M. Pun, “FS-RWKV: leveraging frequency spatial-aware RWKV for 3T-to-7T MRI translation,” in Proc. IEEE Int. Conf. Bioinf. Biomed. (BIBM), 2025, pp. 1–6.

[3] X. Chen, C. Jing, Z. Li, Y. Shen, G. Huang, Y. Liu, B. Lei, C.-M. Pun, and S. Wang, “From image synthesis to digital twins: A paradigm shift in AI-driven medical image computing,” IEEE Trans. Consum. Electron., vol. 72, no. 3, pp. 6132–6144, 2026.

[4] X. Chen, Y. Shen, M. Mahmud, M. Kwok-Po Ng, K.-F. Tsang, S. Wang, C.-M. Pun, and S. Wang, “ConnectomeDiffuser: Generative AI enables brain network construction from diffusion tensor imaging,” IEEE Trans. Consum. Electron., vol. 71, no. 3, pp. 7835–7847, 2025.

[5] X. Zhang, X. He, J. Guo, N. Ettehadi, N. Aw, D. Semanek, J. Posner, A. Laine, and Y. Wang, “PTNet: a high-resolution infant MRI synthesizer based on transformer,” arXiv:2105.13993, 2021.

[6] O. Dalmaz, M. Yurt, and T. C¸ ukur, “ResViT: residual vision transformers for multimodal medical image synthesis,” IEEE Trans. Med. Imaging, vol. 41, no. 10, pp. 2598–2614, 2022.

[7] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: convolutional networks for biomedical image segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent. (MICCAI), 2015, pp. 234–241.

[8] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a Gaussian denoiser: residual learning of deep CNN for image denoising,” IEEE Trans. Image Process., vol. 26, no. 7, pp. 3142–3155, 2017.

[9] X. Chen, Z. Zhou, Z. Li, X. Zhang, Y. Dong, and K.-F. Tsang, “DocUnfold: Leveraging unfolding network and a real-world large-scale dataset for handwriting contamination removal in documents,” IEEE Trans. Consum. Electron., vol. 72, no. 1, pp. 1138–1145, 2026.

[10] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2223–2232.

[11] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 1125–1134.

[12] L. Kong, C. Lian, D. Huang, Z. Li, Y. Hu, and Q. Zhou, “Breaking the dilemma of medical image-to-image translation,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021, pp. 1964–1978.

[13] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, et al., “An image is worth 16x16 words: transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[14] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, et al., “Swin transformer: hierarchical vision transformer using shifted windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 10012– 10022.

[15] H. Shi, H. Tao, G. Huang, J. Jiang, X. Chen, C.-M. Pun, S. Wang, and P. Pan, “Intrinsic concept extraction based on compositional interpretability,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 38969–38978.

[16] A. Gu and T. Dao, “Mamba: linear-time sequence modeling with selective state spaces,” in Proc. Conf. Lang. Model. (COLM), 2024.

[17] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, et al., “VMamba: visual state space model,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 37, 2024.

[18] T. Takikawa, D. Acuna, V. Jampani, and S. Fidler, “Gated-SCNN: gated shape CNNs for semantic segmentation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 5222–5231.

[19] Q. Hou, L. Zhang, M.-M. Cheng, and J. Feng, “Strip pooling: rethinking spatial pooling for scene parsing,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 4003–4012.

[20] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7132– 7141.

[21] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: convolutional block attention module,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 3–19.

[22] X. Wang, R. Girshick, A. Gupta, and K. He, “Non-local neural networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 7794–7803.

[23] I. Sobel, “An isotropic 3x3 image gradient operator,” in Machine Vision for Three-Dimensional Scenes, H. Freeman, Ed. Academic Press, 1990, pp. 376–379.

[24] L. Chi, B. Jiang, and Y. Mu, “Fast Fourier convolution,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020, pp. 4479–4488.

[25] H. Huang, R. He, Z. Sun, and T. Tan, “Wavelet domain generative adversarial network for multi-scale face hallucination,” Int. J. Comput. Vis., vol. 127, no. 6, pp. 763–784, 2019.

[26] K. Xu, M. Qin, F. Sun, Y. Wang, Y.-K. Chen, and F. Ren, “Learning in the frequency domain,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 1740–1749.

[27] X. Chen, Z. Li, X. Zhang, Y. Lei, Y. Dong, X. Zhang, and C.-M. Pun, “AquaClarity: Vision transformer enhances underwater image via nonlinear feature fusion and color fidelity improvement,” IEEE Trans. Consum. Electron., vol. 72, no. 1, pp. 1122–1129, 2026.

[28] X. Xu, Y. Pan, X. Zhang, X. Chen, and K.-F. Tsang, “Manifold-guided geometric enhancement for reliable multimodal interaction in consumer electronics,” IEEE Trans. Consum. Electron., vol. 72, no. 1, pp. 1156– 1163, 2026.

[29] H. Zhao, O. Gallo, I. Frosio, and J. Kautz, “Loss functions for image restoration with neural networks,” IEEE Trans. Comput. Imaging, vol. 3, no. 1, pp. 47–57, 2017.

[30] X. Wang, K. Yu, K. Zhou, W. Luo, and C. C. Loy, “BasicSR: Open source image and video restoration toolbox,” https://github.com/ XPixelGroup/BasicSR, 2023.

[31] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.

[32] I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2017.

[33] M. Jenkinson, P. Bannister, M. Brady, and S. Smith, “Improved optimization for the robust and accurate linear registration and motion correction of brain images,” NeuroImage, vol. 17, no. 2, pp. 825–841, 2002.

[34] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.

[35] L. Qu, Y. Zhang, S. Wang, P.-T. Yap, and D. Shen, “Synthesized 7T MRI from 3T MRI via deep learning in spatial and wavelet domains,” Med. Image Anal., vol. 62, p. 101663, 2020.

[36] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016, pp. 694–711.

[37] X. Chen, A. Li, Y. Dong, Y. Mao, and Z. Liang, “Multivariate featureguided infrared image synthesis with high-frequency fidelity,” IEEE Trans. Consum. Electron., early access, 2026.