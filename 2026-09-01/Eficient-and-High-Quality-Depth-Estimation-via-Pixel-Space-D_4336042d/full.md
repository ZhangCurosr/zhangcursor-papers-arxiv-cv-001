# Eficient and High-Quality Depth Estimation via Pixel-Space Difusion with Linear Attention

Bingde Liu<sup>1,2</sup>, Wu Ran<sup>1</sup>, Jinglei Zhang<sup>1</sup>, Huanhuan Yuan<sup>1</sup>, and Chao Ma<sup>1,⋆</sup>

<sup>1</sup> MoE Key Lab of Artificial Intelligence, Institute of AI, Shanghai Jiao Tong University, Shanghai, China

<sup>2</sup> Zhiyuan College, Shanghai Jiao Tong University, Shanghai, China   
{cylarus1024, bonjourlemonde, zhangjinglei168, yuanhuanhuan, chaoma}@sjtu.edu.cn Code: https://github.com/VISION-SJTU/Lapis

![](images/e1cb1153d621cce86d3ce2f459141d54ee0f1a6565445f40604ac2ad29c43c03.jpg)

![](images/411fd1a2c07775d24f03c371eb6c89772ed5ca250d56adbeb678ef37e3fa91cb.jpg)

![](images/c5dceaafd451dca915a62d3ee484815d65420229d4aa17c2b8030dd0c8829748.jpg)

![](images/8496b0fa8a3bc20ea1aefc15f1c680971d8e054627b555ddbf27fd8a7b2b7cd7.jpg)  
Fig. 1: Lapis: a one-step pixel-space difusion framework with linear attention for eficient and high-quality monocular depth estimation. Lapis delivers precise geometry and intricate structures across resolutions, consistently outperforming SOTA models in accuracy and boundary sharpness with competitive latency.

Abstract. This work presents Lapis, a linear-attention-based pixelspace generative framework that achieves eficient and high-fidelity depth estimation with one-step difusion. While generative frameworks have significantly advanced monocular depth estimation with superior detail fidelity, the O(N<sup>2</sup>) complexity of standard attention and the multi-step denoising process introduce prohibitive computational costs when scaling them to high-resolution image applications. Although linear attention and one-step prediction are intuitively viable, directly applying them

leads to poor structural consistency, detail loss, and noise. Lapis rectifies these limitations through a coarse-to-fine hierarchy. Specifically, a Patch-level Consistency Module restores structural coherence by integrating semantic and spatial priors. Subsequently, a Pixel-level Refinement Module recovers sharp geometric boundaries via skip-connectionbased pixel correspondence. Furthermore, to mitigate sampling noise inherent in one-step difusion, we leverage the manifold assumption and adopt a direct x-prediction strategy to target the clean data manifold. Extensive evaluations on multiple benchmarks demonstrate that Lapis consistently achieves state-of-the-art (SOTA) accuracy and boundary sharpness across various resolutions, reducing inference latency by up to 7.6× at 1080P and 10.9× at 1440P resolution compared to previous SOTA generative models.

Keywords: Monocular Depth Estimation · Difusion Models · Linear Attention

## 1 Introduction

Monocular depth estimation is a cornerstone of computer vision, powering critical applications in real-world 3D/4D perception and reconstruction [28, 33, 37, 40, 67, 72, 80]. Driven by the demand for fine-grained geometric perception, the field has recently transitioned from discriminative regression [41,54,64,65,74,75] to generative frameworks [18, 20, 25, 30, 31, 46, 70, 71]. Compared to their discriminative counterparts, generative models significantly advance fine-grained structural details. However, as applications increasingly demand higher resolutions to capture intricate, pixel-level geometry, as illustrated in Fig. 2, existing generative frameworks inevitably face a fundamental trade-of between fidelity and eficiency. On the one hand, most models [18, 20, 25, 30, 31, 46, 71] inherit latent-space U-Net [57] architectures from pre-trained image generators [56] for eficiency. Such a modeling paradigm is not only restricted by the irreversible information loss of the VAE [32] bottleneck, but also struggles to capture longrange dependencies necessary for global consistency at high resolutions. On the other hand, while recent works [70] employ pixel-space Difusion Transformers (DiTs) [50] to eliminate the latent bottleneck and maintain long-range coherence, their $\mathcal { \bar { O } } ( N ^ { 2 } )$ complexity becomes computationally prohibitive at high resolutions. Moreover, the iterative multi-step sampling in both paradigms multiplies these overheads. Under these constraints, achieving a trajectory that harmonizes pixel-level precision with inference eficiency remains an open challenge.

In search of an architecture capable of capturing long-range consistency in pixel-space without quadratic complexity or multi-step overhead, an intuitively viable choice is to adopt linearized alternatives [29] for attention mechanisms [63] in DiTs with a one-step sampling strategy. Yet, such naive adaptation incurs significant precision loss compared to vanilla DiTs [50]. Specifically, we identify this degradation as stemming from two distinct scales: a patch-level structural misalignment that disrupts global consistency, and a pixel-level fidelity loss that hampers fine-grained geometric recovery. Beyond these architectural weaknesses, the widely-adopted v-prediction trajectories [1, 42, 43] exhibit severe instability in pixel-space when restricted to a single forward pass, yielding noise-corrupted results without iterative refinement.

Test-time Resolution Scaling  
![](images/fda40f05a3fb060b70c5025eec8eecb5bbe938850785fd6592a265b5b319cf6d.jpg)  
Fig. 2: Visualization of test-time resolution scaling using our model. A depth estimation model produces more fine-grained details as input resolution increases.

To address these multifaceted challenges, we propose three core innovations. First, we introduce the Patch-level Consistency Module (PCM) to rectify structural misalignment. The PCM leverages pre-trained semantic priors [49] to enforce global coherence while incorporating local convolutional biases to ensure seamless structural continuity across adjacent patches. Second, to mitigate pixellevel fidelity loss, we develop the Pixel-level Refinement Module (PRM), which employs skip-connection-based pixel correspondence to recover sharp boundaries and intricate structures. Third, we stabilize one-step inference by adopting direct pixel-space x-prediction over the volatile v-prediction. Grounded in the manifold assumption [38], which posits that natural data resides on a lowdimensional manifold, whereas noise-corrupted quantities do not, this strategy targets the clean data manifold directly, efectively suppressing the sampling instability inherent in one-step generation. Building on these components, we present Lapis, the first linear-attention-based pixel-space generative framework to enable consistent and high-fidelity depth reconstruction in a single forward pass while maintaining O(N) computational scaling in the difusion process.

Experiments across 12 benchmarks demonstrate that Lapis achieves SOTA accuracy and boundary sharpness, consistently surpassing leading discriminative and generative baselines across varying resolutions with competitive eficiency, as shown in Fig. 1. For instance, at 1080P resolution, Lapis maintains a 5.1 average AbsRel (Tab. 2), a 10% relative improvement over the strongest competitor, while requiring only 13% of its latency (Tab. 4). These results underscore Lapis as an eficient and high-quality solution for high-resolution 3D perception.

Our main contributions are summarized as follows:

– We present Lapis, the first linearized pixel-space difusion model for monocular depth estimation. It enables linear-time depth difusion within a single forward pass, significantly outperforming generative alternatives in eficiency.

We introduce a coarse-to-fine strategy to rectify the structural and detail degradation of linear attention. This includes a Patch-level Consistency Module that restores global structural coherence by integrating semantic and spatial priors, and a Pixel-level Refinement Module that recovers fine-grained geometric details via skip-connection-based pixel correspondence.

– We leverage the manifold assumption to address the unstable one-step pixelspace sampling via a direct x-prediction strategy. This approach efectively suppresses sampling noise and produces clean depth maps.

– Extensive evaluations show that Lapis achieves SOTA accuracy and boundary sharpness across multiple benchmarks while maintaining competitive latency, ofering a holistic solution for high-resolution 3D perception.

## 2 Related Work

## 2.1 Generative Monocular Depth Estimation

As a cornerstone for 3D vision, monocular depth estimation (MDE) has evolved from early CNN-based regression on specific domains [13, 17, 35] toward opendomain expansion leveraging diverse datasets [54, 76], and finally to the architectural transformation aforded by large-scale Vision Transformers (ViTs) [2,11,12,53]. Recent SOTA discriminative models, such as [41,64,65,74,75], utilize powerful pre-trained encoders [49] and massive training data to achieve remarkable zero-shot robustness. Despite their impressive accuracy and eficiency, these regression-based paradigms often produce over-smoothed predictions that discard fine-grained geometric details. To overcome this challenge, generative modeling works treat depth estimation as a conditional synthesis task. One prominent branch [18,20, 24,25,30,31,46,71] fine-tunes pre-trained latent-space image synthesis models [34, 56] to recover intricate structures during the diffusion process. However, these models remain inadequate for high-resolution depth estimation where high-fidelity geometric details are paramount. Their U-Net backbones [57] struggle with long-range consistency, while VAE latent downsampling [32] irreversibly discards fine-grained geometric details. To circumvent these constraints, recent research [70] has shifted toward pixel-space difusion Transformers (DiTs) [50] to preserve complete spatial information for superior detail restoration. However, pixel-space difusion predominantly relies on standard attention [63] with $\mathcal { O } ( N ^ { 2 } )$ complexity, rendering it computationally prohibitive for high-resolution applications. Furthermore, iterative difusion sampling compounds the latency overhead, posing significant challenges for eficient high-resolution deployment. By contrast, Lapis bridges the gap between geometric precision and inference eficiency via a linearized pixel-space DiT that enables high-fidelity, single-pass reconstruction. Unlike naive adaptations, our dual-level refinement and x-prediction strategy ensure that these eficiency gains enhance, rather than compromise, the restoration of intricate geometric structures.

## 2.2 Sampling Strategies for Generative Depth Estimation

Early generative MDE frameworks [18, 30] leverage pre-trained latent difusion models (LDMs) [56] with ϵ-prediction [27]. Consequently, they inherit the heavy iterative sampling schedule (10–50 steps), leading to substantial computational latency. To accelerate inference, recent research has branched into two primary directions. On one hand, several approaches [20, 31, 70] employ flow-matching with v-prediction for straighter sampling trajectories, but still require at least 4 steps to suficiently suppress noise artifacts. On the other hand, existing onestep x-prediction models [25, 46, 71] remain fundamentally constrained by the LDM paradigm. Specifically, the VAE-based latent space [32] is explicitly regularized toward an isotropic Gaussian distribution, inherently compromising the low-dimensional manifold of natural geometry [38]. As a result, performing onestep x-prediction within such a distorted space leads to poor results. In contrast to existing v-prediction or latent-space x-prediction models, Lapis performs xprediction directly in pixel-space, thereby preserving the original geometric manifold and achieving SOTA accuracy within a single forward pass.

## 2.3 Linear Attention

To maintain global context while bypassing the quadratic scaling of standard selfattention [63], linear attention [29] utilizes kernel approximations to eliminate the Softmax operation, achieving O(N) complexity. Despite its widespread adoption in natural language processing [51,63,69] for long-sequence modeling, it remains disproportionately sparse in visual tasks. Existing explorations [6,15,16,22,23,45] primarily refine the linear attention formulation for categorical tasks with limited data, such as image classification, object detection, instance segmentation, and semantic segmentation, which rely on coarse-grained semantic features and are inherently tolerant of feature fluctuations. In contrast, MDE is a dense regression task requiring high-precision numerical fidelity, making it far more sensitive to the representational decay often associated with linear approximations. More recently, generative models [7,68] have adopted linear attention for visual synthesis tasks. But in these contexts, visual plausibility, e.g., FID [26] and FVD [61], often takes precedence over metric accuracy. Lapis stands as the first application of linear attention for MDE. We demonstrate that linearized models can achieve high-quality 3D dense perception eficiently without compromising accuracy.

## 3 Method

Lapis is designed as a high-eficiency, one-step generative framework for relative monocular depth estimation. As illustrated in Fig. 3, the model operates directly in pixel-space, taking a concatenated noisy depth map and a conditioning image as input to predict a clean depth map without the need for latent autoencoders. To circumvent the quadratic complexity inherent in standard DiTs [50], we replace all standard multi-head self-attention (MSA) layers with linearized attention kernels [29]. To bridge the representational fidelity gap often associated with linear approximations, we augment the transformer blocks with two synergistic components: the Patch-level Consistency Module (PCM) and the Pixel-level Refinement Module (PRM). While the PCM enforces spatial alignment across the token grid at the patch-level, the PRM restores high-frequency precision at the pixel-level, collectively harmonizing linear-time computational eficiency with high-fidelity depth estimation.

![](images/b10b6e21f9cc9e35926d738cb76ff13819aaaf81d90ea138f75d71df3f71cc84.jpg)  
Fig. 3: Architecture overview. Lapis performs pixel-space x-prediction to enable high-quality one-step denoising. It first leverages a linearized DiT backbone to establish a coherent global depth layout, integrating semantic and spatial priors via our Patch-level Consistency Module. Subsequently, the Pixel-level Refinement Module restores fine-grained geometry via skip-connection-based pixel correspondence. The overall framework enables eficient and accurate depth estimation with pixel-level details.

## 3.1 One-step Pixel-Space Difusion

To eliminate the multi-step sampling latency inherent in iterative difusion models, we adopt a native one-step framework. Guided by the low-dimensional manifold assumption [38], which posits that natural geometric data resides on a structured and continuous manifold, our model directly predicts the clean depth map in pixel-space. Specifically, we formulate the difusion process following the flow-matching paradigm [1, 42, 43]. Given a clean depth sample $\mathbf { x } _ { 0 } ,$ we sample

Gaussian noise $\varepsilon \sim \mathcal { N } ( 0 , 1 )$ , time step $t \sim [ \tau , 1 ]$ with a small constant $\tau \ ( e . g .$ 0.05), and apply the linear schedule as:

$$
\mathbf { x } _ { t } = ( 1 - t ) \cdot \mathbf { x } _ { 0 } + t \cdot \varepsilon .\tag{1}
$$

Our model $x _ { \theta } ( \cdot )$ then predicts $\mathbf { x } _ { \mathrm { 0 } }$ with $\mathbf { x } _ { t } ,$ t and corresponding image c as:

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { 0 } = x _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } ) . } \end{array}\tag{2}
$$

Training losses. Our model is optimized via a joint objective that combines velocity-based flow supervision and multi-scale geometric regularization. The velocity loss is defined in v-space following flow-matching frameworks [1,42,43], where the target velocity is given by:

$$
\mathbf { v } _ { t } = \frac { d \mathbf { x } _ { t } } { d t } = \pmb { \varepsilon } - \mathbf { x } _ { 0 } .\tag{3}
$$

We supervise the network by minimizing the Mean Squared Error (MSE) between the predicted velocity $\hat { \mathbf { v } } _ { t } = \left( \mathbf { x } _ { t } - \hat { \mathbf { x } } _ { 0 } \right) / t$ and its ground truth:

$$
\mathcal { L } _ { \mathrm { v e l o c i t y } } = \mathbb { E } _ { \mathbf { x } _ { 0 } , \epsilon , t } \left[ | | \hat { \mathbf { v } } _ { t } - \mathbf { v } _ { t } | | _ { l _ { 2 } } ^ { 2 } \right] .\tag{4}
$$

We further incorporate a multi-scale gradient matching loss directly in x-space following [54, 75] to enhance the sharpness of depth maps:

$$
\mathcal { L } _ { \mathrm { g r a d } } = \mathbb { E } _ { \mathbf { x } _ { 0 } , \boldsymbol { \epsilon } , t } \left[ \sum _ { s \in S } w _ { s } \left\| \nabla _ { s } \mathbf { x } _ { 0 } - \nabla _ { s } \hat { \mathbf { x } } _ { 0 } \right\| _ { l _ { 1 } } \right] ,\tag{5}
$$

where $\nabla _ { s }$ denotes the spatial gradient operator at diferent resolution scales S and $w _ { s }$ is the scale-specific weight. The final loss is a weighted combination:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { v e l o c i t y } } + \lambda _ { \mathrm { g r a d } } \mathcal { L } _ { \mathrm { g r a d } } ,\tag{6}
$$

where $\lambda _ { \mathrm { g r a d } }$ is a hyperparameter balancing accuracy and boundary sharpness.

## 3.2 Linearized Transformer Architecture

Linear attention formulation. In standard self-attention [63] (for single head case), given query, key, and value matrices $Q , K , V \in \mathbb { R } ^ { N \times d }$ , where N is the number of tokens and d is the feature dimension, the attention output is computed as:

$$
\mathrm { A t t n } ( Q , K , V ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { d } } \right) V ,\tag{7}
$$

which incurs a quadratic time complexity of $\mathcal { O } ( N ^ { 2 } d )$ . This cost becomes prohibitive for high-resolution input where N is large (e.g., 1024 × 768 pixels equal to 3072 tokens under patch size 16). Instead, linear attention [29] approximates the softmax similarity with a non-negative kernel function $\phi ( \cdot )$ . By leveraging the associativity of matrix multiplication, the computation can be reformulated as:

![](images/02f1f7e2e4a18b37129ff10c5e6dd59c766ce3d49be83050e04411bf885eb5dd.jpg)  
Fig. 4: Qualitative ablation of architectural components. We visually validate the necessity of each model component in Lapis. In comparison with the full model, removing the global PCM leads to severe global inconsistencies, omitting the local PCM introduces discontinuous patch artifacts, and the absence of the PRM results in a significant loss of sharp details.

$$
{ \mathrm { A t t n } } _ { \mathrm { L i n e a r } } ( Q _ { i } , K , V ) = { \frac { \phi ( Q _ { i } ) \sum _ { j = 1 } ^ { N } \phi ( K _ { j } ) ^ { T } V _ { j } } { \phi ( Q _ { i } ) \sum _ { j = 1 } ^ { N } \phi ( K _ { j } ) ^ { T } } } { \mathrm { ~ f o r ~ e a c h ~ } } i \in [ 1 , N ] .\tag{8}
$$

With pre-computed global context terms $\begin{array} { r } { \sum _ { j = 1 } ^ { N } \phi ( K _ { j } ) ^ { T } V _ { j } \in \mathbb { R } ^ { d \times d } } \end{array}$ and normalization factor $\begin{array} { r } { \sum _ { j = 1 } ^ { N } \phi ( K _ { j } ) ^ { T } \in \mathbb { R } ^ { d \times 1 } } \end{array}$ , Eq. (8) avoids the explicit construction of the $N \times N$ attention matrix and reduces the time complexity to $\mathcal { O } ( N d ^ { 2 } )$ . We adopt the standard DiT [50] as our generative backbone for its structural simplicity and strong scaling properties. By substituting all MSA layers inside it with Eq. (8), we efectively reduce its complexity to $\mathcal { O } ( N )$

The fidelity gap in linearization. Despite its eficiency, we identify a duallevel fidelity gap when applying linear attention to dense depth estimation. First, the low-rank nature of linear kernels [15,22] often leads to patch-level structural misalignment, where geometric coherence is weakened. Second, the iterative linear averaging inherent in these layers results in a significant loss of pixel-level detail. To resolve these issues, we introduce the Patch-level Consistency Module (PCM) and the Pixel-Level Refinement Module (PRM), as detailed below.

## 3.3 Patch-level Consistency Module

Global relational consistency. The core limitation of linearized attention lies in the sacrifice of the exponential softmax kernel, which provides a sharp attentional focalization in long-range modeling [15,22]. We observe that forcing linear attention to simultaneously reconstruct global scene layouts and local geometric details is highly impractical. As illustrated in the second column of Fig. 4, the model struggles to preserve global geometry layout and subsequently blurs local details. To alleviate this, we supervise macro-level global geometry with semantic priors, thereby exempting the linearized backbone from the complexities of global spatial coordination. Specifically, let $\mathbf { s } \in \mathbb { R } ^ { N \times D }$ denote the DiT tokens, where N and D represent the sequence length and embedding dimension, respectively. We align s with semantic descriptors $\mathbf { z } \in \mathbb { R } ^ { N \times D ^ { \prime } }$ , which are extracted from the input image c via a frozen DINOv2 [49] encoder. Here, $D ^ { \prime }$ corresponds to the feature dimension of the DINOv2 backbone. To align and refine high-level semantics, we apply a gating mechanism to z as follows:

$$
\hat { \mathbf { z } } _ { i } = \mathbf { W } _ { o } \left( \mathbf { W } _ { v } \bar { \mathbf { z } } _ { i } \odot \sigma ( \mathbf { W } _ { g } \bar { \mathbf { z } } _ { i } ) \right) ,\tag{9}
$$

where $\{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N }$ are tokens normalized from $\mathbf { z } , \sigma ( \cdot )$ denotes the sigmoid function, ⊙ represents the element-wise product, and $\mathbf { W } _ { v } , \mathbf { W } _ { g } , \mathbf { W } _ { o }$ are learnable channelwise projection matrices. The refined tokens zˆ are subsequently fused with the timestep embedding t to condition the feature flow in DiT blocks using an extended AdaLN-Zero [50] mechanism:

$$
\mathrm { A d a L N - Z e r o } ( f , \mathbf { s } , \mathbf { t } , \hat { \mathbf { z } } ) = \alpha \odot f \left( \mathrm { L a y e r N o r m } ( \mathbf { s } ) \odot ( 1 + \gamma ) + \beta \right) ,\tag{10}
$$

where the modulation parameters $( \gamma , \beta , \alpha )$ are regressed from the joint embedding $\hat { \mathbf { z } } + \mathbf { t }$ via a multi-layer perceptron (MLP). We apply such a mechanism to both the linear attention layers and the feed-forward network (FFN) layers [63], which are represented by $f ( \cdot )$ . In this way, we efectively anchor the feature flow within a consistent global geometric layout without introducing additional parameters to the modulation process. These semantically-modulated blocks are interleaved in the linearized DiT backbone for computational eficiency.

Local spatial continuity. Despite the global structural rectification, linear attention still struggles to maintain local patch-wise continuity. As illustrated in the third column of Fig. 4, this deficiency manifests as localized depth discontinuities and perceptible grid artifacts. We address this structural fragmentation by incorporating a local inductive bias into the FFN layer via a residual $3 \times 3$ depth-wise convolution (DWC), a structure we term ConvFFN. Specifically, we formulate the feature transition in the ConvFFN as follows:

$$
\mathbf { s } ^ { \prime } = \mathrm { S i L U } ( \mathrm { L i n e a r } ( \mathbf { s } ) ) , \quad \mathbf { s } _ { \mathrm { o u t } } = \mathrm { L i n e a r } ( \mathrm { S i L U } ( \mathrm { D W C } ( \mathbf { s } ^ { \prime } ) ) + \mathbf { s } ^ { \prime } ) ,\tag{11}
$$

where SiLU(·) denotes the SiLU activation function [14]. This local inductive bias enforces spatial correlation across adjacent tokens, efectively suppressing the aforementioned grid artifacts to ensure smooth geometric transitions.

## 3.4 Pixel-Level Refinement Module

Although the PCM ensures token-level coherence, mapping discretized tokens back to the continuous pixel-space still entails a significant fidelity gap. As previously analyzed, the iterative averaging efect in deep linear-attention-based architectures leads to cumulative signal dilution, where detailed structures progressively fade across layers. Such degradation is evident in the fourth column of Fig. 4, where a backbone without PRM fails to resolve sharp geometric boundaries. To counteract this, the Pixel-level Refinement Module (PRM) re-integrates pixel-level input cues via a long-range skip connection to efectively recover sharp geometric boundaries. Specifically, the original input $y$ is concatenated with a learnable patch-wise positional encoding and transformed by an MLP to yield the PRM input $y ^ { \prime } \in \bar { \mathbb { R } } ^ { D _ { \mathrm { p i x e l } } \times H \times W }$ , where $D _ { \mathrm { p i x e l } }$ denotes the embedding dimension of the PRM. To bridge the gap between patch-level representation and pixel-level representation, latent tokens $\mathbf { s } ~ \in ~ \mathbb { R } ^ { \tilde { N } \times D }$ from the final DiT block are first reshaped into a spatial grid $\mathbb { R } ^ { D \times h \times w }$ and then projected and upsampled via PixelShufle [60] to match the target pixel-level resolution, following the transformation chain as follows:

![](images/8ac7386b0f93d96f2e78e0e0ff5957a09b5c364fac150553f36a4cbd421e95ba.jpg)  
Fig. 5: Qualitative comparison of monocular depth estimation models. Our model excels at capturing thin structures while preserving the accurate global geometry.

$$
\begin{array} { r } { \textbf { s } \xrightarrow { \mathrm { M L P } } \textbf { s } ^ { \prime } \in \mathbb { R } ^ { ( D _ { \mathrm { p i x e l } } \times p ^ { 2 } ) \times h \times w } \xrightarrow { \mathrm { P i x e l S h u f f e } } \textbf { s } _ { \mathrm { p i x e l } } \in \mathbb { R } ^ { D _ { \mathrm { p i x e l } } \times H \times W } , } \end{array}\tag{12}
$$

where $p$ denotes the patch size, and $H = h \cdot p$ and $W = w \cdot p$ represent the height and width of the original input image. The PRM is architected as a sequence of Pixel Refiner blocks, each using an AdaLN-Zero-modulated ConvFFN layer to decode the pixel-level latents into depth values with pixel-level cues:

$$
\mathrm { A d a L N - Z e r o } ( g , \mathbf { y } ^ { \prime } , \mathbf { s } _ { \mathrm { p i x e l } } ) = \alpha \odot g \left( \mathrm { L a y e r N o r m } ( \mathbf { y } ^ { \prime } ) \odot ( 1 + \gamma ) + \beta \right) ,\tag{13}
$$

where $g ( \cdot )$ denotes the ConvFFN layer, and the scaling and shifting parameters $( \gamma , \beta , \alpha )$ are regressed from the features $\mathbf { s } _ { \mathrm { p i x e l } }$ via an MLP. This design efectively recovers pixel-level details from the raw input pixels during the decoding stage. Finally, a standard linear prediction head in DiT [50] is used to project the refined pixel-level features into a high-fidelity depth map d $\in \mathbb { R } ^ { H \times W }$

Table 1: Zero-shot relative depth estimation on native resolution. Lapis surpasses both established discriminative models and recent generative baselines.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td rowspan="2">Training Data</td><td colspan="2">NYUv2</td><td colspan="2">KITTI AbsRel↓</td><td colspan="2">ETH3D</td><td colspan="2">ScanNet</td><td colspan="2">DIODE</td><td colspan="2">Average</td></tr><tr><td>AbsRel↓</td><td>δ1↑</td><td></td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td></tr><tr><td rowspan="10">Discttie</td><td>MiDaS [54]</td><td>2M</td><td>11.1</td><td>88.5</td><td>23.6</td><td>63.0</td><td>18.4</td><td>75.2</td><td>12.1</td><td>84.6</td><td></td><td></td><td></td><td></td></tr><tr><td>LeRes [77]</td><td>354K</td><td>9.0</td><td>91.6</td><td>14.9</td><td>78.4</td><td>17.1</td><td>77.7</td><td>9.1</td><td>91.7</td><td></td><td></td><td></td><td></td></tr><tr><td>Omnidata [12]</td><td>12.2M</td><td>7.4</td><td>94.5</td><td>14.9</td><td>83.5</td><td>16.6</td><td>77.8</td><td>7.5</td><td>93.6</td><td></td><td></td><td></td><td></td></tr><tr><td>DPT [53]</td><td>1.4M</td><td>9.8</td><td>90.3</td><td>10.0</td><td>90.1</td><td>7.8</td><td>94.6</td><td>8.2</td><td>93.4</td><td></td><td></td><td></td><td></td></tr><tr><td>HDN [79]</td><td>300K</td><td>6.9</td><td>94.8</td><td>11.5</td><td>86.7</td><td>12.1</td><td>83.3</td><td>8.0</td><td>93.9</td><td></td><td></td><td></td><td></td></tr><tr><td>Lotus-D [25]</td><td>59K</td><td>5.1</td><td>97.2</td><td>8.1</td><td>93.1</td><td>6.1</td><td>97.0</td><td>5.5</td><td>96.5</td><td></td><td></td><td></td><td></td></tr><tr><td>Depth Anything [74] *</td><td>62.6M</td><td>4.5</td><td>97.9</td><td>8.7</td><td>93.6</td><td>6.6</td><td>98.1</td><td>4.4</td><td>98.0</td><td>11.1</td><td>94.8</td><td>7.0</td><td>96.5</td></tr><tr><td>Depth Anything V2 [75] * *</td><td>62.6M</td><td>4.5</td><td>97.9</td><td>7.9</td><td>93.6</td><td>7.4</td><td>97.7</td><td>4.2</td><td>97.8</td><td>10.0</td><td>94.9</td><td>6.8</td><td>96.4</td></tr><tr><td>Depth Anything 3 [41] </td><td>1.2M (Scenes)</td><td>4.1</td><td>97.4</td><td>7.0</td><td>95.6</td><td>4.2</td><td>97.7</td><td>4.1</td><td>97.6</td><td>7.1</td><td>94.6</td><td>5.3</td><td>96.6</td></tr><tr><td>MoGe [64] * MoGe-2  $[ 6 5 ] ~ *$ </td><td>9M 8.9M</td><td>3.7 3.5</td><td>97.9 98.0</td><td>5.9 5.9</td><td>97.3 97.1</td><td>3.5 4.5</td><td>98.4 96.7</td><td>3.6 3.4</td><td>98.3 98.3</td><td>7.7 6.7</td><td>94.8</td><td>4.9 4.8</td><td>97.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>95.5</td><td></td><td>97.1</td></tr><tr><td rowspan="10">Genative</td><td>GeoWizard [18]</td><td>280K</td><td>5.6</td><td>96.3</td><td>14.4</td><td>82.0</td><td>6.6</td><td>95.8</td><td>6.4</td><td>95.0</td><td></td><td></td><td></td><td></td></tr><tr><td>GenPercept [71]</td><td>74K</td><td>5.6</td><td>96.0</td><td>13.0</td><td>84.2</td><td>7.0</td><td>95.6</td><td>6.2</td><td>96.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Diffusion-E2E-FT [46]</td><td>74K</td><td>5.4</td><td>96.5</td><td>9.6</td><td>92.1</td><td>6.4</td><td>95.9</td><td>5.8</td><td>96.5</td><td></td><td>一</td><td></td><td></td></tr><tr><td>Lotus-G [25]</td><td>59K</td><td>5.4</td><td>96.8</td><td>8.5</td><td>92.2</td><td>5.9</td><td>97.0</td><td>5.9</td><td>95.7</td><td></td><td></td><td></td><td></td></tr><tr><td>Marigold v1.0 [30]</td><td>74K</td><td>5.5</td><td>96.4</td><td>9.9</td><td>91.6</td><td>6.5</td><td>95.9</td><td>6.4</td><td>95.2</td><td>11.1*</td><td>89.5*</td><td>7.9</td><td>93.7</td></tr><tr><td>Marigold v1.1 [31]</td><td>74K</td><td>5.5</td><td>96.4</td><td>10.5</td><td>90.2</td><td>6.9</td><td>95.7</td><td>5.8</td><td>96.3</td><td>10.1*</td><td>91.0*</td><td>7.7</td><td>93.9</td></tr><tr><td>Lotus-2 [24]</td><td>59K</td><td>4.1</td><td>97.6</td><td>6.7</td><td>94.5</td><td>4.6</td><td>98.1</td><td>4.2</td><td>97.6</td><td>8.4*</td><td>94.7*</td><td>5.6</td><td>96.5</td></tr><tr><td>Pixel-Perfect Depth  $\mathbf { \Pi } _ { \downarrow } ^ { \left[ 7 0 \right] } { } ^ { * }$ </td><td>125K</td><td>3.7</td><td>98.1</td><td>5.4</td><td>97.1</td><td>3.1</td><td>99.2</td><td>3.8</td><td>98.2</td><td>4.9</td><td>97.5</td><td>4.2</td><td>98.0</td></tr><tr><td>Lapis (Ours) *</td><td>122K</td><td>3.8</td><td>98.2</td><td>5.1</td><td>97.7</td><td>3.2</td><td>99.2</td><td>3.8</td><td>98.2</td><td>4.8</td><td>97.9</td><td>4.1</td><td>98.2</td></tr></table>

## 4 Experiments

## 4.1 Implementation Details

Depth normalization. Given raw depth map d, our model follows [70] to predict the normalized log-scale depth $\mathbf { x } _ { \mathrm { 0 } }$ as:

$$
\mathbf { x } _ { 0 } = \frac { \hat { \mathbf { d } } - \hat { \mathbf { d } } _ { 2 } } { \hat { \mathbf { d } } _ { 9 8 } - \hat { \mathbf { d } } _ { 2 } } - 0 . 5 ,\tag{14}
$$

where $\hat { \mathbf { d } } = \log ( \mathbf { d } + \epsilon )$ with the ofset ϵ fixed at 0.01 to ensure numerical stability. $\hat { \mathbf { d } } _ { 2 }$ and $\hat { \mathbf { d } } _ { 9 8 }$ represent the $2 ^ { \mathrm { n d } }$ and $9 8 ^ { \mathrm { t h } }$ percentiles computed independently for each depth map. We further rescale $\mathbf { x } _ { \mathrm { 0 } }$ to ensure unit variance, facilitating a more stable data distribution.

Training datasets. We adopt a total of about 122K training samples from five synthetic datasets: Hypersim [55] (53.8K), TartanAir [66] (28.2K), VKITTI2 [5] (21.3K), UrbanSyn [21] (7.5K), and MatrixCity [39] (11.6K) to train our model. All training datasets are directly resized to 1024 × 768 resolution and simply concatenated together without weighted sampling.

Model architecture details. Our model comprises 520M parameters, maintaining consistency with the DiT-Large [50] configuration in hidden dimensions and the number of blocks while transitioning to a fully linearized backbone. The kernel feature map $\phi ( \cdot )$ in Eq. (8) is implemented as $\phi ( x ) = \mathrm { e l u } ( x ) + 1$ [8]. Following [70], we employ the DINOv2-Large [49] encoder fine-tuned by MoGe-2 [65] as the frozen semantic encoder to provide robust structural priors. For the PRM, we use $M = 3$ blocks and an embedding dimension of $D _ { \mathrm { p i x e l } } = 3 2$

Table 2: Zero-shot relative depth estimation on 1080P and 1440P. Notably, Lapis demonstrates the exceptional resolution-robustness of our dual-level PCM and PRM modeling and largely outperforms existing baselines in accuracy metrics.
<table><tr><td rowspan=1 colspan=14>Middlebury   Booster   Cityscapes DrivingStereo  ETH3D     AverageRes.     Method      1920 × 1080  4112 × 3008  2048 × 1024   1762 × 800   6048 × 4032AbsRel↓δ1↑AbsRel↓δ1↑AbsRel↓δ1↑AbsRel↓δ1↑AbsRel↓δ1↑AbsRel↓δ1↑</td></tr><tr><td rowspan=1 colspan=2>Depth Anything V2 [75]</td><td rowspan=1 colspan=1>6.6</td><td rowspan=1 colspan=1>96.8</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>99.7</td><td rowspan=1 colspan=2>10.1  89.4</td><td rowspan=1 colspan=2>6.4  96.4</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>98.1</td><td rowspan=1 colspan=1>6.4</td><td rowspan=1 colspan=1>96.1</td></tr><tr><td rowspan=1 colspan=1>Depth Anything 3 [41]</td><td rowspan=1 colspan=1>3 [41]</td><td rowspan=1 colspan=1>7.6</td><td rowspan=1 colspan=1>93.4</td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>99.8</td><td rowspan=1 colspan=1>18.9</td><td rowspan=1 colspan=1>75.3</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>96.8</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>97.5</td><td rowspan=1 colspan=1>7.9</td><td rowspan=1 colspan=1>92.6</td></tr><tr><td rowspan=1 colspan=1>MoGe-2 [65</td><td rowspan=1 colspan=1>]</td><td rowspan=1 colspan=1>7.5</td><td rowspan=1 colspan=1>93.8</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>99.9</td><td rowspan=1 colspan=1>11.8</td><td rowspan=1 colspan=1>89.2</td><td rowspan=1 colspan=1>5.8</td><td rowspan=1 colspan=1>96.5</td><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>96.9</td><td rowspan=1 colspan=1>6.3</td><td rowspan=1 colspan=1>95.3</td></tr><tr><td rowspan=2 colspan=1>1080   Marigold v1.1 [Lotus-2 [24</td><td rowspan=1 colspan=1>31]</td><td rowspan=1 colspan=1>13.8</td><td rowspan=1 colspan=1>81.4</td><td rowspan=1 colspan=1>8.6</td><td rowspan=1 colspan=1>94.3</td><td rowspan=1 colspan=1>24.0</td><td rowspan=1 colspan=1>60.0</td><td rowspan=1 colspan=1>7.1</td><td rowspan=1 colspan=1>95.9</td><td rowspan=1 colspan=1>11.5</td><td rowspan=1 colspan=1>87.2</td><td rowspan=1 colspan=1>13.0</td><td rowspan=1 colspan=1>83.8</td></tr><tr><td rowspan=1 colspan=1>]</td><td rowspan=1 colspan=1>14.5</td><td rowspan=1 colspan=1>76.2</td><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=1>91.3</td><td rowspan=1 colspan=1>31.8</td><td rowspan=1 colspan=1>58.1</td><td rowspan=1 colspan=1>6.1</td><td rowspan=1 colspan=1>96.6</td><td rowspan=1 colspan=1>6.6</td><td rowspan=1 colspan=1>97.5</td><td rowspan=1 colspan=1>13.4</td><td rowspan=1 colspan=1>83.9</td></tr><tr><td rowspan=2 colspan=2>Pixel-Perfect Depth [70]Lapis (Ours)</td><td rowspan=1 colspan=1>6.3</td><td rowspan=1 colspan=1>96.5</td><td rowspan=1 colspan=1>2.9</td><td rowspan=1 colspan=1>99.8</td><td rowspan=1 colspan=1>9.3</td><td rowspan=1 colspan=1>93.5</td><td rowspan=1 colspan=1>6.1</td><td rowspan=1 colspan=1>96.8</td><td rowspan=1 colspan=1>4.1</td><td rowspan=1 colspan=1>99.1</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>97.1</td></tr><tr><td rowspan=1 colspan=1>5.6</td><td rowspan=1 colspan=1>96.8</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>99.9</td><td rowspan=1 colspan=1>8.7</td><td rowspan=1 colspan=1>92.5</td><td rowspan=1 colspan=1>5.6</td><td rowspan=1 colspan=1>97.0</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>99.3</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>97.1</td></tr><tr><td rowspan=2 colspan=2>Depth Anything V2 [75]Depth Anything 3 [41]</td><td rowspan=1 colspan=1>6.9</td><td rowspan=1 colspan=1>96.5</td><td rowspan=1 colspan=1>3.4</td><td rowspan=1 colspan=1>99.6</td><td rowspan=1 colspan=1>10.6</td><td rowspan=1 colspan=1>88.7</td><td rowspan=1 colspan=1>6.4</td><td rowspan=1 colspan=1>96.5</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=1>97.9</td><td rowspan=1 colspan=1>6.6</td><td rowspan=1 colspan=1>95.8</td></tr><tr><td rowspan=1 colspan=1>3 [41]</td><td rowspan=1 colspan=1>7.9</td><td rowspan=1 colspan=1>93.4</td><td rowspan=1 colspan=1>2.9</td><td rowspan=1 colspan=1>99.9</td><td rowspan=1 colspan=1>20.4</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>5.6</td><td rowspan=1 colspan=1>96.9</td><td rowspan=1 colspan=1>6.2</td><td rowspan=1 colspan=1>96.6</td><td rowspan=1 colspan=1>8.6</td><td rowspan=1 colspan=1>91.6</td></tr><tr><td rowspan=1 colspan=1>MoGe-2 [65]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.7</td><td rowspan=1 colspan=1>93.5</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>99.9</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>88.9</td><td rowspan=1 colspan=1>5.8</td><td rowspan=1 colspan=1>96.5</td><td rowspan=1 colspan=1>4.6</td><td rowspan=1 colspan=1>96.8</td><td rowspan=1 colspan=1>6.4</td><td rowspan=1 colspan=1>95.1</td></tr><tr><td rowspan=2 colspan=1>140P   Marigold v1.1 [Lotus-2 [24]</td><td rowspan=1 colspan=1>31]</td><td rowspan=1 colspan=1>17.1</td><td rowspan=1 colspan=1>73.9</td><td rowspan=1 colspan=1>10.4</td><td rowspan=1 colspan=1>91.2</td><td rowspan=1 colspan=1>40.8</td><td rowspan=1 colspan=1>42.1</td><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1>95.3</td><td rowspan=1 colspan=1>15.8</td><td rowspan=1 colspan=1>78.6</td><td rowspan=1 colspan=1>18.3</td><td rowspan=1 colspan=1>76.2</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>16.6</td><td rowspan=1 colspan=1>71.1</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>88.9</td><td rowspan=1 colspan=1>65.1</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1>6.2</td><td rowspan=1 colspan=1>96.6</td><td rowspan=1 colspan=1>8.4</td><td rowspan=1 colspan=1>96.3</td><td rowspan=1 colspan=1>21.3</td><td rowspan=1 colspan=1>80.1</td></tr><tr><td rowspan=1 colspan=1>Pixel-Perfect Dept</td><td rowspan=1 colspan=1>h [70]</td><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=1>95.3</td><td rowspan=1 colspan=1>3.9</td><td rowspan=1 colspan=1>99.8</td><td rowspan=1 colspan=1>10.0</td><td rowspan=1 colspan=1>92.9</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>98.5</td><td rowspan=1 colspan=1>6.5</td><td rowspan=1 colspan=1>96.6</td></tr><tr><td rowspan=1 colspan=2>Lapis (Ours)</td><td rowspan=1 colspan=1>5.8</td><td rowspan=1 colspan=1>96.4</td><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>99.9</td><td rowspan=1 colspan=1>9.1</td><td rowspan=1 colspan=1>91.8</td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=2>97.0  3.4</td><td rowspan=1 colspan=1>99.2</td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=1>96.9</td></tr></table>

Training details. We optimize the model using the total loss defined in Eq. (6), with the gradient loss weight set to $\lambda _ { \mathrm { g r a d } } = 1 . 5$ . Training is conducted using the AdamW optimizer [44] with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ and no weight decay. The model is trained for 400K steps on 4 NVIDIA RTX Pro 6000 Blackwell GPUs, with a per-GPU batch size of 8. An Exponential Moving Average (EMA) with a decay rate of 0.9999 is applied to the model weights during training, and all evaluations are performed using the EMA weights. Both training and inference are conducted in bfloat16 mixed precision.

## 4.2 Zero-shot Relative Depth Evaluation

Native resolution. We evaluate zero-shot relative depth estimation on five mainstream unseen real-world datasets: NYUv2 [48] (indoor), KITTI [19] (outdoor), ETH3D [59] (various), ScanNet [10] (indoor), and DIODE [62] (various). We adopt two widely accepted metrics: Absolute Relative Error (AbsRel) and $\delta _ { 1 }$ accuracy. <sup>3</sup> Each model is evaluated at its training resolution (usually lower than 720P) to assess its peak accuracy. As summarized in Tab. 1, Lapis achieves the best performance with an averaged AbsRel of 4.1% and a $\delta _ { 1 }$ accuracy of 98.2%. Besides, as demonstrated in Fig. 5, our model excels in resolving thin structures and intricate details while maintaining a coherent global geometric layout.

Test-time resolution scaling at 1080P and 1440P. Benefiting from linearized attention and one-step difusion, Lapis enables eficient scaling to high resolutions for enhanced geometric details. To validate that such a capability does not come at the cost of degraded global accuracy, we evaluate zero-shot relative depth at 1080P and 1440P test-time resolution on five datasets with high-resolution annotations: Middlebury [58] (indoor), Booster [52,78] (indoor),

Table 3: Boundary sharpness across diferent resolutions. We report the F1- score (↑) for depth boundary accuracy. Lapis consistently outperforms all existing methods across various resolutions, while achieving near-monotonic improvements in boundary sharpness as the test-time resolution scales up.
<table><tr><td rowspan="2">Method</td><td colspan="4">Hypersim F1↑</td><td colspan="4">Sintel F1↑</td><td colspan="4">Spring F1↑</td><td colspan="4">Average F1↑</td></tr><tr><td>480P 720P</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> 1080P 1440P 480P 720P 1080P 1440P 480P 720P 1080P 1440P 480P 720P 1080P 1440P</td><td></td><td></td><td></td></tr><tr><td>Depth Anything V2 [75]</td><td>13.1</td><td>20.9</td><td>24.5</td><td>24.1</td><td>17.3</td><td>23.0</td><td>23.7</td><td>23.6</td><td>6.1</td><td>9.2</td><td>11.0</td><td>11.7</td><td>12.2</td><td>17.7</td><td>19.7</td><td>19.8</td></tr><tr><td>Depth Anything 3 [41]]</td><td>12.5</td><td>19.3</td><td>23.1</td><td>24.4</td><td>15.8</td><td>20.9</td><td>23.7</td><td>24.6</td><td>5.8</td><td>8.7</td><td>10.2</td><td>10.3</td><td>11.4</td><td>16.3</td><td>19.0</td><td>19.8</td></tr><tr><td>MoGe-2 [65]</td><td>19.5</td><td>28.4</td><td>31.7</td><td>32.1</td><td>21.4</td><td>27.8</td><td>29.2</td><td>28.9</td><td>5.8</td><td>8.4</td><td>10.3</td><td>10.9</td><td>15.6</td><td>21.5</td><td>23.7</td><td>24.0</td></tr><tr><td>Marigold v1.1 [31]</td><td>20.9</td><td>22.2</td><td>8.9</td><td>7.7</td><td>17.5</td><td>18.8</td><td>6.1</td><td>5.6</td><td>5.0</td><td>5.6</td><td>2.9</td><td>2.6</td><td>14.5</td><td>15.6</td><td>6.0</td><td>5.3</td></tr><tr><td>Lotus-2 [24]</td><td>17.1</td><td>21.9</td><td>22.6</td><td>22.0</td><td>14.5</td><td>18.3</td><td>19.2</td><td>18.5</td><td>4.7</td><td>7.5</td><td>9.3</td><td>9.4</td><td>12.1</td><td>15.9</td><td>17.0</td><td>16.6</td></tr><tr><td>Pixel-Perfect Depth [70]</td><td>26.5 31.2</td><td></td><td>32.7</td><td>30.6</td><td>19.1</td><td>21.5</td><td>21.0</td><td>20.8</td><td>6.5</td><td>10.6</td><td>12.0</td><td>13.2</td><td>17.4</td><td>21.1</td><td>21.1</td><td>22.3</td></tr><tr><td>Lapis (Ours)</td><td>27.4 32.2</td><td></td><td>34.8</td><td>33.6</td><td>26.2</td><td>30.9</td><td>31.4</td><td>31.7</td><td>8.0</td><td>13.0</td><td>15.5</td><td>15.8</td><td>20.5</td><td>25.4</td><td>27.2</td><td>27.0</td></tr></table>

Table 4: Runtime statistics across diferent resolutions. We report latency (ms), throughput (Hz), and peak GPU memory (GB). Lapis achieves throughput competitive with leading discriminative models while consistently outperforming existing generative frameworks by a significant margin.
<table><tr><td rowspan="2">Method</td><td colspan="3">480P</td><td colspan="3">720P</td><td colspan="3">1080P</td><td colspan="3">1440P</td></tr><tr><td>Lat ↓</td><td>TP↑Mem ↓</td><td></td><td>Lat ↓</td><td>TP↑</td><td>Mem ↓</td><td>Lat ↓</td><td>TP↑</td><td>Mem ↓</td><td>Lat ↓</td><td></td><td>TP↑Mem ↓</td></tr><tr><td>Depth Anything V2 [75]</td><td>18.6</td><td>53.7</td><td>1.3</td><td>33.9</td><td>29.5</td><td>1.5</td><td>80.6</td><td>12.4</td><td>1.7</td><td>183.5</td><td>5.5</td><td>2.1</td></tr><tr><td>Depth Anything 3 [41]</td><td>14.9</td><td>67.0</td><td>1.3</td><td>58.1</td><td>17.2</td><td>1.3</td><td>141.2</td><td>7.1</td><td>1.3</td><td>207.6</td><td>4.8</td><td>1.4</td></tr><tr><td>MoGe-2 [65]</td><td>24.1</td><td>41.5</td><td>1.3</td><td>49.3</td><td>20.3</td><td>1.5</td><td>115.3</td><td>8.7</td><td>1.7</td><td>245.5</td><td>4.1</td><td>2.1</td></tr><tr><td>Marigold v1.1 [31]</td><td>119.1</td><td>8.4</td><td>6.0</td><td>328.1</td><td>3.0</td><td>8.0</td><td>758.1</td><td>1.3</td><td>11.7</td><td>1479.0</td><td>0.7</td><td>16.9</td></tr><tr><td>Lotus-2 [24]</td><td>1258.6</td><td>0.8</td><td>37.5</td><td>3095.3</td><td>0.3</td><td>39.5</td><td>6952.7</td><td>0.1</td><td>43.3</td><td>13747.0</td><td>0.1</td><td>48.5</td></tr><tr><td>Pixel-Perfect Depth [70]</td><td>75.3</td><td>13.3</td><td>3.2</td><td>250.2</td><td>4.0</td><td>3.3</td><td>905.5</td><td>1.1</td><td>3.5</td><td>2457.0</td><td>0.4</td><td>3.8</td></tr><tr><td>Lapis (Ours)</td><td>27.5</td><td>36.4</td><td>3.3</td><td>55.4</td><td>18.0</td><td>3.4</td><td>119.0</td><td>8.4</td><td>3.6</td><td>224.7</td><td>4.5</td><td>3.9</td></tr></table>

Cityscapes [9] (outdoor), DrivingStereo [73] (outdoor), and ETH3D [59] (various). As reported in Tab. 2, Lapis maintains remarkable accuracy metrics, outperforming current SOTA models by approximately 10% and 19% in AbsRel, respectively. This demonstrates that our architecture not only scales eficiently but also preserves global structural integrity under significant resolution shifts.

## 4.3 Boundary Sharpness Evaluation

To quantify the details of the estimated depth maps, we evaluate the boundary sharpness F1-score proposed by [3] across three high-resolution synthetic benchmarks: Hypersim [55] (indoor), Sintel [4] (various), and Spring [47] (outdoor). Unlike global error metrics, this evaluation specifically measures the model’s ability to resolve sharp occlusion boundaries and fine-grained structural transitions. As demonstrated in Tab. 3, Lapis consistently outperforms both discriminative and generative baselines in boundary sharpness across all test resolutions, validating the eficacy of our PRM in recovering high-frequency spatial details.

## 4.4 Runtime Analysis

We evaluate the runtime performance of Lapis against competitive methods on a single NVIDIA RTX Pro 6000 Blackwell GPU with 1 sample per batch. As summarized in Tab. 4, Lapis significantly outperforms all generative baselines in terms of inference latency. By leveraging a one-step difusion formulation and linearized attention, our model achieves a processing speed that is highly competitive with purely regression-based architectures.

![](images/b99f818450448c3d5ca2fc085f0afba3f29fc4d0cc17840b287184475bf415de.jpg)

Table 5: Ablation study of Lapis components. We evaluate the contribution of each module across five datasets.
<table><tr><td rowspan="2">Components</td><td colspan="2">NYUv2</td><td colspan="2">KITTI</td><td colspan="2">ETH3D</td><td colspan="2">ScanNet</td><td colspan="2">DIODE</td><td colspan="2">Average</td></tr><tr><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1 ↑</td></tr><tr><td>Full Model</td><td>4.3</td><td>98.0</td><td>5.7</td><td>96.9</td><td>4.0</td><td>98.6</td><td>4.5</td><td>97.7</td><td>5.6</td><td>97.0</td><td>4.8</td><td>97.7</td></tr><tr><td>— w/o Global PCM</td><td>19.6</td><td>68.2</td><td>20.2</td><td>69.9</td><td>19.1</td><td>71.4</td><td>19.1</td><td>69.1</td><td>21.2</td><td>71.9</td><td>19.8</td><td>70.1</td></tr><tr><td>w/o Local PCM</td><td>4.5</td><td>97.7</td><td>6.4</td><td>96.5</td><td>4.3</td><td>98.5</td><td>4.5</td><td>97.7</td><td>5.6</td><td>97.0</td><td>5.1</td><td>97.5</td></tr><tr><td>w/o PRM</td><td>4.5</td><td>97.8</td><td>5.7</td><td>96.9</td><td>4.0</td><td>98.6</td><td>4.8</td><td>97.6</td><td>5.5</td><td>97.1</td><td>4.9</td><td>97.6</td></tr></table>

![](images/af69cd230dcd97cd0edc4a942ae2d9158f8e483ef1cf2448fbf5f310f0642d7e.jpg)  
Fig. 6: Quantitative comparison under various denoising steps and qualitative comparison for the impact of the PRM on one-step generation. A one-step x-prediction strategy is optimal in both eficiency and geometric fidelity.

## 4.5 Ablation Study

To facilitate eficient analysis, we follow [41, 65] to adopt a lightweight configuration based on the DiT-Base [50] backbone. For the PRM, we reduce the complexity to M = 2 blocks with an embedding dimension of $D _ { \mathrm { p i x e l } } = 1 6$

Efectiveness of components. We conduct a component-wise ablation to validate the synergy of the PCM and the PRM, as reported in Tab. 5 and Fig. 4. First, removing Global PCM causes a sharp accuracy drop, confirming the necessity of semantic supervision for global coherence. Second, Local PCM is critical for inter-patch continuity. Its absence increases AbsRel (4.8 → 5.1) and introduces grid artifacts. Finally, while the PRM has a marginal impact on accuracy metrics, it is essential for restoring high-frequency details otherwise diluted in the deep linearized backbone to ensure detailed geometry.

Prediction targets and difusion steps. We compare x-prediction with vprediction across varying sampling steps (1, 2, 4, 8). As shown in Fig. 6 (left), both strategies peak at one-step sampling, with x-prediction slightly outperforming in AbsRel. We further analyze pixel-level continuity. As visualized in

Fig. 6 (right), while the PRM efectively suppresses noise in the v-prediction path, x-prediction consistently yields more spatially coherent depth maps. Consequently, a one-step x-prediction strategy is identified as our optimal choice, as it inherently aligns with the clean data manifold and maximizes both eficiency and geometric fidelity.

## 5 Conclusion

In this paper, we presented Lapis, a pixel-space generative model based on linear attention for eficient and high-quality depth estimation, especially under highresolution inputs. By decoupling patch-level context modeling from pixel-level refinement and leveraging a direct x-prediction strategy, we efectively resolve the degradations in the naively linearized backbone and suppress the noise in one-step pixel-space difusion. Extensive experiments demonstrate that Lapis achieves SOTA zero-shot performance while significantly reducing inference latency. Our approach provides a scalable and computationally eficient foundation for real-world 3D perception, bridging the gap between generative quality and discriminative speed.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant Nos. 62322113, 62376156), as well as the Shanghai Municipal Special Program for Basic Research on General AI Foundation Models (Grant No. 2025SHZDZX025G15).

## References

1. Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571 (2023)

2. Bhat, S.F., Birkl, R., Wofk, D., Wonka, P., Müller, M.: Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288 (2023)

3. Bochkovskii, A., Delaunoy, A., Germain, H., Santos, M., Zhou, Y., Richter, S.R., Koltun, V.: Depth pro: Sharp monocular metric depth in less than a second. In: International Conference on Learning Representations (2025)

4. Butler, D.J., Wulf, J., Stanley, G.B., Black, M.J.: A naturalistic open source movie for optical flow evaluation. In: A. Fitzgibbon et al. (Eds.) (ed.) Proceedings of the European Conference on Computer Vision. pp. 611–625. Part IV, LNCS 7577, Springer-Verlag (Oct 2012)

5. Cabon, Y., Murray, N., Humenberger, M.: Virtual kitti 2. arXiv preprint arXiv:2001.10773 (2020)

6. Cai, H., Li, J., Hu, M., Gan, C., Han, S.: Eficientvit: Multi-scale linear attention for high-resolution dense prediction. arXiv preprint arXiv:2205.14756 (2024)

7. Chen, J., Zhao, Y., Yu, J., Chu, R., Chen, J., Yang, S., Wang, X., Pan, Y., Zhou, D., Ling, H., et al.: Sana-video: Eficient video generation with block linear difusion transformer. arXiv preprint arXiv:2509.24695 (2025)

8. Clevert, D.A., Unterthiner, T., Hochreiter, S.: Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289 (2016)

9. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2016)

10. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2017)

11. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2021)

12. Eftekhar, A., Sax, A., Malik, J., Zamir, A.: Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 10786–10796 (2021)

13. Eigen, D., Puhrsch, C., Fergus, R.: Depth map prediction from a single image using a multi-scale deep network. Advances in Neural Information Processing Systems 27 (2014)

14. Elfwing, S., Uchibe, E., Doya, K.: Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks 107, 3–11 (2018)

15. Fan, Q., Huang, H., Ai, Y., He, R.: Rectifying magnitude neglect in linear attention. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 21505–21514 (2025)

16. Fan, Q., Huang, H., He, R.: Breaking the low-rank dilemma of linear attention. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 25271–25280 (2025)

17. Fu, H., Gong, M., Wang, C., Batmanghelich, K., Tao, D.: Deep ordinal regression network for monocular depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2002–2011 (2018)

18. Fu, X., Yin, W., Hu, M., Wang, K., Ma, Y., Tan, P., Shen, S., Lin, D., Long, X.: Geowizard: Unleashing the difusion priors for 3d geometry estimation from a single image. In: Proceedings of the European Conference on Computer Vision (2024)

19. Geiger, A., Lenz, P., Stiller, C., Urtasun, R.: Vision meets robotics: The kitti dataset. International Journal of Robotics Research (2013)

20. Gui, M., Schusterbauer, J., Prestel, U., Ma, P., Kotovenko, D., Grebenkova, O., Baumann, S.A., Hu, V.T., Ommer, B.: Depthfm: Fast generative monocular depth estimation with flow matching. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 3203–3211 (2025)

21. Gómez, J.L., Silva, M., Seoane, A., Borràs, A., Noriega, M., Ros, G., Iglesias-Guitian, J.A., López, A.M.: All for one, and one for all: Urbansyn dataset, the third musketeer of synthetic driving scenes. Neurocomputing 637, 130038 (2025)

22. Han, D., Pan, X., Han, Y., Song, S., Huang, G.: Flatten transformer: Vision transformer using focused linear attention. In: Proceedings of the IEEE International Conference on Computer Vision (2023)

23. Han, D., Wang, Z., Xia, Z., Han, Y., Pu, Y., Ge, C., Song, J., Song, S., Zheng, B., Huang, G.: Demystify mamba in vision: A linear attention perspective. Advances in Neural Information Processing Systems 37, 127181–127203 (2024)

24. He, J., Li, H., Sheng, M., Chen, Y.C.: Lotus-2: Advancing geometric dense prediction with powerful image generative model. arXiv preprint arXiv:2512.01030 (2025)

25. He, J., Li, H., Yin, W., Liang, Y., Li, L., Zhou, K., Zhang, H., Liu, B., Chen, Y.: Lotus: Difusion-based visual foundation model for high-quality dense prediction. In: International Conference on Learning Representations. vol. 2025, pp. 89454– 89467 (2025)

26. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017)

27. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in Neural Information Processing Systems 33, 6840–6851 (2020)

28. Huang, J., Miao, S., Yang, B., Ma, Y., Liao, Y.: Vivid4d: Improving 4d reconstruction from monocular video by video inpainting. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 12592–12604 (2025)

29. Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.: Transformers are rnns: Fast autoregressive transformers with linear attention. In: International Conference on Machine Learning. pp. 5156–5165. PMLR (2020)

30. Ke, B., Obukhov, A., Huang, S., Metzger, N., Daudt, R.C., Schindler, K.: Repurposing difusion-based image generators for monocular depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

31. Ke, B., Qu, K., Wang, T., Metzger, N., Huang, S., Li, B., Obukhov, A., Schindler, K.: Marigold: Afordable adaptation of difusion-based image generators for image analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

32. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2022)

33. Kumar, R., Vats, V.: Few-shot novel view synthesis using depth aware 3d gaussian splatting. In: Proceedings of the European Conference on Computer Vision. pp. 1–13. Springer (2024)

34. Labs, B.F.: Flux. https://github.com/black- forest- labs/flux (2024), accessed: June 30, 2026

35. Lee, J.H., Han, M.K., Ko, D.W., Suh, I.H.: From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv preprint arXiv:1907.10326 (2019)

36. Lefaudeux, B., Massa, F., Liskovich, D., Xiong, W., Caggiano, V., Naren, S., Xu, M., Hu, J., Tintore, M., Zhang, S., Labatut, P., Haziza, D., Wehrstedt, L., Reizenstein, J., Sizov, G.: xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers (2022), accessed: June 30, 2026

37. Li, J., Zhang, J., Bai, X., Zheng, J., Ning, X., Zhou, J., Gu, L.: Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20775–20785 (June 2024)

38. Li, T., He, K.: Back to basics: Let denoising generative models denoise. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 36115–36125 (2026)

39. Li, Y., Jiang, L., Xu, L., Xiangli, Y., Wang, Z., Lin, D., Dai, B.: Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 3205–3215 (2023)

40. Li, Z., Yu, Z., Austin, D., Fang, M., Lan, S., Kautz, J., Alvarez, J.M.: FB-OCC: 3D occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492 (2023)

41. Lin, H., Chen, S., Liew, J.H., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647 (2025)

42. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2023)

43. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022)

44. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2019)

45. Lu, J., Yao, J., Zhang, J., Zhu, X., Xu, H., Gao, W., Xu, C., Xiang, T., Zhang, L.: Soft: Softmax-free transformer with linear complexity. arXiv preprint arXiv:2110.11945 (2022)

46. Martin Garcia, G., Knaebel, K., Schmidt, C., de Geus, D., Hermans, A., Leibe, B.: Fine-tuning image-conditional difusion models is easier than you think. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (2025)

47. Mehl, L., Schmalfuss, J., Jahedi, A., Nalivayko, Y., Bruhn, A.: Spring: A highresolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023)

48. Nathan Silberman, Derek Hoiem, P.K., Fergus, R.: Indoor segmentation and support inference from rgbd images. In: Proceedings of the European Conference on Computer Vision (2012)

49. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Howes, R., Huang, P.Y., Xu, H., Sharma, V., Li, S.W., Galuba, W., Rabbat, M., Assran, M., Ballas, N., Synnaeve, G., Misra, I., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

50. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 4195–4205 (2023)

51. Qin, Z., Sun, W., Deng, H., Li, D., Wei, Y., Lv, B., Yan, J., Kong, L., Zhong, Y.: cosformer: Rethinking softmax in attention. In: International Conference on Learning Representations (2022)

52. Ramirez, P.Z., Costanzino, A., Tosi, F., Poggi, M., Salti, S., Mattoccia, S., Stefano, L.D.: Booster: A benchmark for depth from images of specular and transparent surfaces. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(1), 85–102 (2024)

53. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction. Proceedings of the IEEE International Conference on Computer Vision (2021)

54. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., Koltun, V.: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(3) (2022)

55. Roberts, M., Ramapuram, J., Ranjan, A., Kumar, A., Bautista, M.A., Paczan, N., Webb, R., Susskind, J.M.: Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In: Proceedings of the IEEE International Conference on Computer Vision (2021)

56. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10684–10695 (2022)

57. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 234–241. Springer (2015)

58. Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang, X., Westling, P.: High-resolution stereo datasets with subpixel-accurate ground truth. In: German Conference on Pattern Recognition. pp. 31–42. Springer (2014)

59. Schöps, T., Schönberger, J.L., Galliani, S., Sattler, T., Schindler, K., Pollefeys, M., Geiger, A.: A multi-view stereo benchmark with high-resolution images and multi-camera videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2017)

60. Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D., Wang, Z.: Real-time single image and video super-resolution using an eficient subpixel convolutional neural network. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1874–1883 (2016)

61. Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: FVD: A new metric for video generation (2019), https://openreview.net/ forum?id=rylgEULtdN, accessed: June 30, 2026

62. Vasiljevic, I., Kolkin, N., Zhang, S., Luo, R., Wang, H., Dai, F.Z., Daniele, A.F., Mostajabi, M., Basart, S., Walter, M.R., Shakhnarovich, G.: DIODE: A Dense Indoor and Outdoor DEpth Dataset. arXiv preprint arXiv:1908.00463 (2019)

63. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017)

64. Wang, R., Xu, S., Dai, C., Xiang, J., Deng, Y., Tong, X., Yang, J.: Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5261–5271 (2025)

65. Wang, R., Xu, S., Dong, Y., Deng, Y., Xiang, J., Lv, Z., Sun, G., Tong, X., Yang, J.: Moge-2: Accurate monocular geometry with metric scale and sharp details. In: Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., Chen, N. (eds.) Advances in Neural Information Processing Systems. vol. 38, pp. 35928– 35959. Curran Associates, Inc. (2025)

66. Wang, W., Zhu, D., Wang, X., Hu, Y., Qiu, Y., Wang, C., Hu, Y., Kapoor, A., Scherer, S.: Tartanair: A dataset to push the limits of visual slam. In: IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 4909–4916. IEEE (2020)

67. Wu, Y., Zheng, W., Zuo, S., Huang, Y., Zhou, J., Lu, J.: Embodiedocc: Embodied 3d occupancy prediction for vision-based online scene understanding. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 26360–26370 (October 2025)

68. Xie, E., Chen, J., Chen, J., Cai, H., Tang, H., Lin, Y., Zhang, Z., Li, M., Zhu, L., Lu, Y., et al.: Sana: Eficient high-resolution image synthesis with linear difusion transformers. arXiv preprint arXiv:2410.10629 (2024)

69. Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., Singh, V.: Nystr"omformer: A nystr"om-based algorithm for approximating self-attention. arXiv preprint arXiv:2102.03902 (2021)

70. Xu, G., Lin, H., Luo, H., Wang, X., YAO, J., Zhu, L., Pu, Y., Chi\_, C., Sun, H., Wang, B., Chen, G., Ye, H., Peng, S., Yang, X.: Pixel-perfect depth with semanticsprompted difusion transformers. In: Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., Chen, N. (eds.) Advances in Neural Information Processing Systems. vol. 38, pp. 174731–174755. Curran Associates, Inc. (2025)

71. Xu, G., Liu, M., Fan, C., Xie, K., Zhao, Z., Chen, H., Shen, C., et al.: What matters when repurposing difusion models for general dense perception tasks? In: International Conference on Learning Representations. vol. 2025, pp. 6786–6799 (2025)

72. Xu, J., Deng, K., Fan, Z., Wang, S., Xie, J., Yang, J.: Ad-gs: Object-aware b-spline gaussian splatting for self-supervised autonomous driving. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 24770–24779 (October 2025)

73. Yang, G., Song, X., Huang, C., Deng, Z., Shi, J., Zhou, B.: Drivingstereo: A largescale dataset for stereo matching in autonomous driving scenarios. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2019)

74. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., Zhao, H.: Depth anything: Unleashing the power of large-scale unlabeled data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

75. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., Zhao, H.: Depth anything v2. arXiv preprint arXiv:2406.09414 (2024)

76. Yin, W., Liu, Y., Shen, C., Yan, Y.: Enforcing geometric constraints of virtual normal for depth prediction. In: Proceedings of the IEEE International Conference on Computer Vision (2019)

77. Yin, W., Zhang, J., Wang, O., Niklaus, S., Mai, L., Chen, S., Shen, C.: Learning to recover 3d scene shape from a single image. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2021)

78. Zama Ramirez, P., Tosi, F., Poggi, M., Salti, S., Di Stefano, L., Mattoccia, S.: Open challenges in deep stereo: the booster dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022)

79. Zhang, C., Yin, W., Wang, Z., Yu, G., Fu, B., Shen, C.: Hierarchical normalization for robust monocular depth estimation. Advances in Neural Information Processing Systems (2022)

80. Zhang, W., Liu, H., Qi, Z., Wang, Y., Yu, X., Zhang, J., Dong, R., He, J., Wang, H., Zhang, Z., et al.: Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge. Advances in Neural Information Processing Systems 38, 24195–24228 (2026)

## A More Implementation Details

Table F shows the hyperparameters of Lapis and its ablation variant. Both configurations are designed to remain strictly comparable with DiT-L and DiT-B [50] in hidden dimensions and the number of blocks, respectively.

Table F: Detailed architectural configurations for Lapis.
<table><tr><td rowspan="2">Model</td><td colspan="5">Linearized DiT Backbone</td><td>PCM</td><td colspan="3">PRM</td></tr><tr><td>Patch Size</td><td>Blocks</td><td>Embed Dim Attn Heads</td><td></td><td>FFN Dim</td><td>DINOv2</td><td>Blocks</td><td>Embed Dim</td><td>FFN Dim</td></tr><tr><td>Lapis (Full)</td><td>16 × 16</td><td>24</td><td>1024</td><td>16</td><td>4096</td><td>ViT-L/14</td><td>3</td><td>32</td><td>128</td></tr><tr><td>Lapis (Ablation)</td><td>16 × 16</td><td>12</td><td>768</td><td>12</td><td>3072</td><td>ViT-B/14</td><td>2</td><td>16</td><td>64</td></tr></table>

## B Evaluation Protocols

## B.1 Relative Depth Estimation

For zero-shot relative depth evaluation, we adopt the open-source protocol from Marigold [30, 31] for pre-processing, least-squares alignment, and metric computation. Additionally, we downsample ETH3D [59] to 2048 × 1360, and use a Sobel filter with a gradient threshold of 0.3 to mask out unreliable depth transitions caused by sensor noise in DIODE [62]. All models perform inference at their respective scales (e.g., native, 1080P, or 1440P), with predictions resized to ground-truth resolution before alignment and metric computation.

## B.2 Boundary Sharpness

For boundary sharpness evaluation, we adopt the open-source protocol from Depth Pro [3] for metric computation. Consistent with our relative depth evaluation, all models perform inference at their respective scales, with the results subsequently resized to the ground-truth resolution. These predictions are then aligned via least-squares and clipped to the valid depth range of each dataset before computing the final sharpness metrics.

## B.3 Runtime Analysis

We evaluate the runtime eficiency on a single NVIDIA RTX Pro 6000 Blackwell GPU with a batch size of 1. We incorporate Memory Eficient Attention [36] for standard Softmax attention [63] operations, and further optimize all models via compilation to minimize execution overhead. The reported results are the average of 100 iterations after 20 warm-up rounds to eliminate cold-start overhead. Notably, Lapis significantly outperforms existing generative models in eficiency even without specialized CUDA kernel optimizations for linear attention.

Table G: Edge-aware point cloud evaluation on the test split of Hypersim [55]. We report the log-space Chamfer Distance (CD) near depth discontinuities.
<table><tr><td>Method</td><td></td><td></td><td></td><td>Depth Any. V2 [75] Depth Any. 3 [41] MoGe-2 [65] Marigold v1.1 [31] PPD [70] Lapis (Ours)</td><td></td><td></td></tr><tr><td>Log-Space CD ↓</td><td>0.19</td><td>0.17</td><td>0.14</td><td>0.18</td><td>0.12</td><td>0.12</td></tr></table>

![](images/9b168a758c41bd459858230a23c94ccb07146d16b539323fe9356df1cfd0cfa7.jpg)  
Fig. G: Qualitative comparisons for point clouds without post-processing. Compared to discriminative [65, 75] and latent-space generative [31] methods, Lapis significantly suppresses flying pixel artifacts (highlighted in red boxes). Furthermore, while previous pixel-space generative models [70] often sufer from “snow-like” noise in flat regions, Lapis produces more spatially consistent surfaces. Best viewed zoomed-in.

## C Edge-Aware Point Cloud Evaluation

While current discriminative and latent-space generative models often sufer from flying pixel artifacts due to over-smoothing or the lossy compression of VAEs [32], Lapis preserves uncompromising geometric fidelity as a pixel-space difusion framework. To validate this, we follow [70] to evaluate edge-aware point clouds on the oficial test split of Hypersim [55]. Unlike the boundary F1-score, this evaluation provides a more rigorous assessment from the perspective of 3D structural integrity. Specifically, predicted depth maps are first aligned with the ground truth and back-projected into 3D point clouds with camera intrinsics. We then apply a Canny detector to the ground-truth depth maps in log-space to extract edge masks and subsequently compute the log-space Chamfer Distance (CD) near these regions. As demonstrated in Tab. G, Lapis achieves the lowest log-space CD (0.12), matching the SOTA pixel-space generative model [70] and significantly outperforming discriminative and latent-space generative ones [31,41,65,75]. The visualization in Fig. G further highlights that Lapis produces smoother and more spatially consistent surfaces.

## D Extended Ablation Study

In this section, we additionally investigate the architectural choices of Lapis, focusing on linear attention kernels and semantic conditioning strategies.

Table H: Ablation study of diferent kernel functions. Our linearized configurations maintain performance parity with the standard Softmax attention, while the ELU + 1 kernel yields a slightly superior overall result.
<table><tr><td rowspan="2">Kernel</td><td colspan="2">NYUv2</td><td colspan="2">KITTI</td><td colspan="2">ETH3D</td><td colspan="2">ScanNet</td><td colspan="2">DIODE</td><td colspan="2">Average</td></tr><tr><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td> $\delta _ { 1 } \uparrow$ </td></tr><tr><td>Softmax (Baseline)</td><td>4.6</td><td>98.0</td><td>5.7</td><td>96.9</td><td>4.1</td><td>98.6</td><td>5.0</td><td>97.6</td><td>5.6</td><td>97.1</td><td>5.0</td><td>97.6</td></tr><tr><td>ReLU</td><td>4.5</td><td>97.9</td><td>5.7</td><td>97.0</td><td>4.0</td><td>98.7</td><td>4.9</td><td>97.6</td><td>5.5</td><td>97.1</td><td>4.9</td><td>97.7</td></tr><tr><td>Focused ReLU, p = 3</td><td>4.5</td><td>97.9</td><td>5.7</td><td>96.9</td><td>3.9</td><td>98.7</td><td>4.7</td><td>97.7</td><td>5.6</td><td>97.0</td><td>4.9</td><td>97.6</td></tr><tr><td>ELU + 1</td><td>4.3</td><td>98.0</td><td>5.7</td><td>96.9</td><td>4.0</td><td>98.6</td><td>4.5</td><td>97.7</td><td>5.6</td><td>97.0</td><td>4.8</td><td>97.7</td></tr></table>

Table I: Ablation study of semantic conditioning strategies. Methods enforcing a direct one-to-one spatial correspondence (MLP and AdaLN variants) significantly outperform global Linear Cross-Attention in preserving the geometric layout.
<table><tr><td rowspan="2">Conditioning</td><td colspan="2">NYUv2</td><td colspan="2">KITTI</td><td colspan="2">ETH3D</td><td colspan="2">ScanNet</td><td colspan="2">DIODE</td><td colspan="2">Average</td></tr><tr><td>AbsRel.↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel↓</td><td>δ1↑</td><td>AbsRel.↓</td><td>δ1↑</td></tr><tr><td>Linear Cross-Attention</td><td>10.3</td><td>89.6</td><td>13.9</td><td>82.7</td><td>9.8</td><td>91.2</td><td>9.7</td><td>90.5</td><td>10.4</td><td>91.4</td><td>10.8</td><td>89.1</td></tr><tr><td>MLP</td><td>4.4</td><td>98.0</td><td>5.8</td><td>96.9</td><td>4.0</td><td>98.7</td><td>4.6</td><td>97.8</td><td>5.6</td><td>97.0</td><td>4.9</td><td>97.7</td></tr><tr><td>AdaLN (Int.)</td><td>4.5</td><td>97.9</td><td>5.8</td><td>96.9</td><td>4.0</td><td>98.7</td><td>4.7</td><td>97.7</td><td>5.6</td><td>97.1</td><td>4.9</td><td>97.7</td></tr><tr><td>AdaLN-Zero (Full)</td><td>4.4</td><td>97.9</td><td>5.8</td><td>96.7</td><td>4.0</td><td>98.6</td><td>4.8</td><td>97.5</td><td>5.6</td><td>97.1</td><td>4.9</td><td>97.6</td></tr><tr><td>AdaLN-Zero (Int.)</td><td>4.3</td><td>98.0</td><td>5.7</td><td>96.9</td><td>4.0</td><td>98.6</td><td>4.5</td><td>97.7</td><td>5.6</td><td>97.0</td><td>4.8</td><td>97.7</td></tr></table>

## D.1 Influence of Kernel Functions

We evaluate various activations within the linearized DiT backbone of the complete Lapis architecture, with a standard Softmax attention variant as a baseline for comparison. Specifically, we contrast the ReLU kernel adopted in [68], the ELU+1 kernel as the canonical setting for linear attention [29], and the Focused ReLU kernel [22], which enhances the focus ability of ReLU via a power function $\begin{array} { r } { f _ { p } ( x ) = \frac { \| x \| } { \| x ^ { p } \| } x ^ { p } } \end{array}$ . As shown in Table H, the linearized models achieve performance competitive with, or even slightly exceeding, the Softmax baseline, validating the efectiveness of our architectural design. Among them, ELU + 1 yields slightly superior average performance and is thus selected as our final configuration.

## D.2 Semantic Conditioning Strategies

We investigate several mechanisms for semantic conditioning inside the PCM, drawing inspiration from the class-label conditioning used in DiT [50]. These include: 1) Linear Cross-Attention, which treats semantic features as keys and values for global interaction; 2) a simple MLP that concatenates features with image tokens; and 3) AdaLN variants that modulate features through adaptive layer normalization. As shown in Tab. I, Linear Cross-Attention yields significantly inferior results. We attribute this to the absence of a patch-wise inductive bias. Specifically, cross-attention allows each token to aggregate information from the entire semantic map, which blurs the precise spatial alignment between semantic tokens and DiT tokens, and compromises the global geometric layout that the PCM is intended to regularize. In contrast, all methods that preserve this localized inductive bias achieve consistently superior and comparable performance.

![](images/2eccbd0de47701eeb8c25a724a46aa43753aa4ecabf58c2a7b439553eaf7fba3.jpg)  
Fig. H: Qualitative comparisons on three synthetic benchmarks. We visualize depth estimation results in Hypersim [55], Sintel [4], and Spring [47]. Lapis demonstrates robust performance in preserving sharp structural boundaries across diverse scenarios, ranging from complex indoor geometries to wide-scale outdoor landscapes.

Among them, interleaved AdaLN-Zero is selected as our final configuration, as it ofers slightly better average results without introducing additional parameters.

## E More Visual Results

We provide additional qualitative comparisons on synthetic benchmarks, including Hypersim [55], Sintel [4], and Spring [47], as well as open-domain images sourced from publicly available web resources, e.g., Pexels <sup>4</sup> and Unsplash <sup>5</sup>, in Fig. H and Fig. I, respectively. These results further demonstrate the robustness and structural fidelity of Lapis across diverse scenarios.

## F Limitations and Future Work

Currently, linear attention exhibits numerical instability in reduced-precision formats (e.g., fp16). This challenge arises from the lack of explicit normalization in the attention score computation process. Addressing this through more stable formulations and optimized kernels will enhance both numerical robustness and throughput. Furthermore, Lapis presently focuses on single-image relative depth estimation. Extending this framework to metric depth or video-consistent estimation would significantly broaden its utility in downstream applications, such as robotics and temporal-coherent scene reconstruction.

![](images/591e0d33868298061e5159dad184a498bb068a0ad588caf71f8d5cccc744397d.jpg)  
Fig. I: Qualitative results on open-domain images. Best viewed zoomed-in.