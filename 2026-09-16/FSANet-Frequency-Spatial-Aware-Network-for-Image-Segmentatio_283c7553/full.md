# FSANet: Frequency-Spatial Aware Network for Image Segmentation

Ruibo Wang, Ziyi Shen, Huaming Wu, Senior Member, IEEE, Dong Liang, Senior Member, IEEE, and Kun Shang

Abstract—Image segmentation remains challenging due to occlusions, poor lighting, and irregular structures. Although transformer-based methods achieve high accuracy, they rely heavily on long-range spatial features, leading to high computational costs and neglecting prior knowledge or noise patterns, resulting in missing details and unclear boundaries. To address these issues, we propose Frequency Spatial Aware Network (FSANet), which integrates prior knowledge with a dual-domain solver to sequentially adapt to diverse segmentation tasks. Specifically, we design three key modules: (1) Structure Prior Module, which recovers overlooked details; (2) Dual-Domain Awareness Module, which captures salient features while disentangling noise; and (3) Edge Estimation Module, which enhances edge awareness for more precise segmentation. In addition, the limited availability of comprehensive segmentation datasets covering various real-world scenarios hinders the performance of existing methods. To address this, we introduce SceneX, a novel opensource dataset featuring 10 challenging non-ideal scenarios, establishing a new benchmark for evaluating and improving the robustness and real-world applicability of the segmentation models. Extensive experiments demonstrate the efficiency and effectiveness of FSANet.

Index Terms—Image segmentation, structure prior, dualdomain learning, benchmark dataset, and non-ideal scenarios.

## I. INTRODUCTION

CCURATE object segmentation serves as a fundamental cornerstone for high-level scene understanding, playing a critical role in a wide range of applications, including medical imaging [1], autonomous driving [2], and robotic sensing and navigation [3]. Recent advances in deep learning have significantly boosted segmentation performance under controlled conditions. In particular, Convolutional Neural Networks (CNNs) [4]–[6] have driven considerable progress by leveraging hierarchical feature learning. Nonetheless, the intrinsic locality of convolutional operations restricts their capacity to capture long-range dependencies and integrate global semantic contexts. To address this, Transformer-based methods [7]–[9] have been developed to enhance spatial reasoning and contextual modeling via self-attention mechanisms, albeit at the cost of substantial computational complexity.

Real-world settings, by contrast, are inherently dynamic and unpredictable, presenting severe challenges to segmentation models [11] designed for controlled environments. These challenges frequently manifest as degraded segmentation quality and error propagation that adversely affects downstream processes such as object tracking [12] and semantic reasoning [13]. Such shortcomings underscore the pressing demand for segmentation frameworks that are both robust and generalizable, capable of adapting to diverse and unconstrained visual scenarios.

![](images/869c812d62df089e700542bd55e05d7a8affd1c8dbc00c8c637b0ef8d9ff9183.jpg)  
Fig. 1. Zero-shot Performance vs. Learnable Parameters for an array of SAM variants on the COCO dataset [10].

To enhance flexibility and generalization across diverse segmentation tasks [14], unified frameworks such as One-Former [15] and UNINEXT [16] have been proposed. These methods aim to integrate multiple segmentation paradigms, such as semantic, instance, and panoptic segmentation, within a single architectural design. While conceptually elegant, their practical generalization remains constrained. Adaptation to new tasks often requires additional fine-tuning or taskspecific prompting, limiting their applicability in open-world and zero-shot settings. More recently, foundation models including CLIPSeg [17] and the Segment Anything Model (SAM) [18] have shown impressive zero-shot segmentation ability by leveraging large-scale vision-language pretraining. Nevertheless, despite their strong generalization, these models frequently underperform in fine-grained segmentation scenarios involving complex object structures and significant global noise [19]–[21]. In parallel, a body of multi-scale and frequency-aware methods for degraded imagery—video deraining [22], snow removal with semantic and depth priors [23], Taylor-expanded transformers for image restoration [24], lightweight super-resolution [25], and conditionguided multi-modal prediction with intention and interaction priors [26]—demonstrates the value of explicit multi-scale and spectral priors under adverse conditions, motivating our frequency–spatial design.

This underperformance can be attributed to three primary factors: (1) Underutilization of prior knowledge: Many existing methods [27], [28] heavily rely on spatial features extracted by Vision Transformers (ViTs) [29] while insufficiently leveraging structural priors, resulting in suboptimal performance in detailed structure recovery and precise boundary delineation. (2) Ineffective feature distribution modeling: As noted by Cong et al. [30], real-world noise often exhibits highly coupled and frequency-dependent characteristics that are not adequately captured by current segmentation frameworks, limiting their capacity to represent such complex noise patterns. (3) Limited training data diversity: Although widely adopted benchmarks such as COCO [10], Cityscapes [31], and LVIS [32] have driven progress in structured and well-illuminated environments, they often lack representation of the full spectrum of real-world conditions, including frequent occlusions, poor lighting, cluttered backgrounds, and domain-specific artifacts. Consequently, the generalization capability of current segmentation models remains constrained in unconstrained settings [33]–[36].

In this paper, we propose a Frequency–Spatial Aware Network (FSANet) that addresses key limitations of existing segmentation methods [37] in handling structural ambiguity, global noise, and boundary degradation. Achieving consistent performance gains over state-of-the-art methods that are most pronounced on the boundary-sensitive mBIoU metric, where they widen monotonically with backbone capacity (+1.3/+1.5/+2.4 average mBIoU from ViT-B to ViT-H, Table II), as shown in Fig. 1, FSANet is a noise-resilient and structure-aware segmentation framework that holistically integrates spatial and frequency-domain information for robust scene understanding. It consists of three complementary components: (1) a Structure Prior Module (SPM) that incorporates structural priors to improve spatial coherence and recover fine-grained object details; (2) a Dual Domain Awareness Module (DDAM) that mitigates frequency-domain noise and enhances discriminative features via Frequency Dynamic Filtering and Lightweight Spatial Enhancement Blocks; and (3) an Edge Estimation Module (EEM) that refines boundary localization using edge-aware loss constraints. To further support real-world segmentation under challenging conditions, we introduce SceneX, a novel open-source dataset covering ten non-ideal scenarios designed to advance the development and evaluation of segmentation models in diverse adverse environments. The main contributions of this work are summarized as follows:

• Structural Guidance for Spatial Coherence. We design a SPM that embeds structural priors to reinforce spatial coherence and restore fine object details. This is complemented by an EEM that preserves boundary integrity through edge-aware constraints, together providing explicit structural guidance and precise object delineation.

• Dual-Domain Modeling for Noise Robustness. We develop a DDAM that disentangles frequency-coupled noise and amplifies salient cues via frequency-adaptive filtering and lightweight spatial enhancement. This dual-domain modeling significantly improves segmentation robustness under real-world noise.

• Open Dataset for Challenging Scenarios. We present SceneX, an open-source dataset spanning ten challenging non-ideal scenarios. It supports both model training and systematic evaluation under diverse adverse conditions, fostering progress in real-world segmentation.

• Extensive Experimental Validation. Through comprehensive experiments on standard benchmarks and realworld data, we demonstrate that FSANet achieves superior segmentation accuracy and exhibits strong crossdomain generalization capability.

## II. MOTIVATION

Understanding the frequency-domain manifestations of realworld noise is essential for building robust segmentation models. As illustrated in Fig. 2, our analysis yields three critical observations:

(a) Robustness Dilemma in Spatial Models: As shown in Fig. 2-(a), while spatial-only models (e.g., SAM) perform accurately in controlled, clean scenarios (”No Rain Eval”), they suffer catastrophic degradation under real-world environmental noise (”Rain Eval”). In contrast, our dual-domain approach maintains high consistency with the ground truth by effectively leveraging frequency priors, demonstrating superior robustness.

(b) Spectral Decoupling of Noise and Semantics: Further spectral analysis in Fig. 2-(b) reveals that global noise patterns possess distinct, disentangleable frequency signatures. Comparing the first two rows, we observe that rain streaks with different spatial orientations manifest as high-magnitude spectral streaks in orthogonal directions in the frequency domain. Crucially, a comparison between the second and third rows demonstrates that this spectral representation remains invariant to the underlying semantic content: when the noise pattern persists, the characteristic ”frequency spikes” remain identical even as the scene changes. This confirms that the frequency domain effectively separates noise patterns from semantic information.

(c) Global Governance of Frequency Components: Finally, the frequency manipulation experiment in Fig. 2-(c) highlights the holistic impact of local frequency components. By isolating specific high-frequency regions (highlighted in yellow dashed boxes) associated with noise and performing an Inverse FFT (IFFT), we observe that these local spectral edits induce global, rain-like structural artifacts across the entire reconstructed image. This phenomenon validates that the frequency domain not only characterizes global noise but also governs spatial structural integrity globally.

These findings indicate that frequency-domain analysis can effectively disentangle spatially entangled noise patterns. The translation invariance of the magnitude spectrum supports robust pattern matching, while its global receptive field facilitates the joint modeling of fine-grained anomalies and coarse structures. Consequently, integrating frequency-domain cues serves as a powerful prior for achieving accurate and robust segmentation in unseen, noisy conditions.

To place these observations on a quantitative footing, we measure the radially averaged power spectrum of 100 clean images and of the same images under six controlled degradations. Table I reports the energy of the high band, above half the Nyquist frequency, relative to the clean image. Blur and JPEG compression remove high-frequency energy, whereas additive noise, low light and rain add it, so each degradation carries a distinct and reproducible spectral signature. A single fixed filter cannot serve both regimes, which is why the filter used in Section III is learned rather than designed.

![](images/d4d195570fc851d830023154c1de067fea59d99509ba41bff46c93f6e5598336.jpg)  
Fig. 2. Motivation for Frequency-Aware Modeling. (a) Robustness Comparison: Contrast between spatial-only models (SAM) and our dual-domain approach under controlled vs. real-world noisy conditions. (b) Spectral Analysis: Illustration of distinct directional energy distributions for global noise patterns, highlighting their consistency across varying objects. (c) Global Governance: Visualization of global spatial structural distortions induced by local frequency component editing.

TABLE I  
ENERGY OF THE HIGH-FREQUENCY BAND (ABOVE HALF THE NYQUIST FREQUENCY) UNDER SIX CONTROLLED DEGRADATIONS, RELATIVE TO THE CLEAN IMAGE, AVERAGED OVER 100 IMAGES.
<table><tr><td>Degradation</td><td>Blur</td><td>JPEG</td><td>Haze</td><td>Rain</td><td>Low light</td><td>Noise</td></tr><tr><td>High-band energy vs. clean</td><td>×0.04</td><td>×0.64</td><td>×0.86</td><td>×1.24</td><td>×2.17</td><td>×2.71</td></tr></table>

Because FSANet filters the spectrum of a deep feature map rather than that of the image, we further verify that the two spectra are linked. Passing 250 clean and degraded images through the frozen encoder, the high-band energy of the feature map follows that of the image, but selectively: the loss of high frequencies caused by blur propagates into the feature spectrum (correlation +0.56 across the blur sweep), whereas the encoder itself removes most of the high-frequency energy added by noise (correlation −0.81). The feature spectrum is thus a structured, degradation-dependent transform of the image spectrum, which is what makes spectral filtering meaningful at the feature level.

## III. METHOD

This section elaborates on the architectural components of FSANet, as illustrated in Fig. 3.

## A. Structure Prior Module

The SPM, depicted in Fig. 3(a), integrates object-level structural priors into boundary prediction and enhances the recovery of fine structural details through a gated interaction with ViT features. Specifically, a structural prior is constructed to suppress global noise while preserving essential structural information [38]. This prior is computed as the channel-wise difference between the maximum and minimum values of the input image $I \in \mathbb { R } ^ { H \times W \times C }$

![](images/b18f36b63fb4a6254bf9f99c3b6a63da096ccba1125bd987f091b57d7cf6dc18.jpg)  
Fig. 3. Overall architecture of the proposed FSANet, which consists of three core components: the Structure Prior Module (SPM) for recovering structural information, the Dual Domain Awareness Module (DDAM) for cross-domain modeling in complex scenarios, and the Edge Estimation Module (EEM) for enhancing boundary awareness. For clarity, the prompt encoder, prompt tokens, and output tokens in SAM are omitted.

$$
I _ { S } ( h , w ) = \operatorname* { m a x } _ { c \in \{ r , g , b \} } I ^ { c } ( h , w ) - \operatorname* { m i n } _ { d \in \{ r , g , b \} } I ^ { d } ( h , w ) .\tag{1}
$$

The resulting $I _ { S }$ is then processed to extract multi-scale structural representations:

$$
F _ { S } ^ { i } = \sigma _ { g } ( C _ { k _ { i } \times k _ { i } } ( S ( C _ { 1 \times 1 } ( I _ { S } ) ) ) ) ,\tag{2}
$$

where $C _ { 1 \times 1 }$ represents a $1 \times 1$ convolution, $C _ { k _ { i } \times k _ { i } }$ denotes convolutions with varying kernel sizes $k _ { i } \in \{ 7 , 9 \}$ , S(·) is the split operation, $\sigma _ { g }$ is the GeLU activation function [39], and $F _ { S } ^ { i }$ corresponds to the feature group indexed by $i \in \{ 1 , 2 \}$

## B. Dual Domain Awareness Module

As illustrated in Fig. 3(b), the Dual Domain Awareness Module (DDAM) serves as the core component of FSANet. It employs a cross-domain modeling strategy to enhance perceptual robustness under diverse conditions. A Refinement Step (RS) is designed to mitigate the severe degradation of segmentation accuracy in complex real-world environments by suppressing global noise and amplifying semantically meaningful features. This RS process is implemented through an Expansion layer $E ( \cdot )$ [40] and a Projection layer $P ( \cdot )$ [40], and is formulated as:

$$
D = \mathrm { C R A } _ { m } ( M _ { S A M } + F _ { F } ) + M _ { S A M } + F _ { F } ,\tag{3}
$$

where $\mathrm { C R A } _ { m } ( \cdot )$ refers to a modified Channel Reduction Attention mechanism [41]<sup>1</sup>, $M _ { S A M }$ is the mask feature extracted from the frozen SAM encoder, and $\begin{array} { r l } { F _ { F } } & { { } = } \end{array}$ Cat $\left( \sigma _ { g } \big ( P \big ( S \big ( \sigma _ { g } ( C _ { 3 \times 3 } ^ { d } ( E ( F _ { V } ^ { s } + F _ { V } ^ { d } ) ) ) \big ) \big ) \big ) \odot F _ { S } ^ { i } \right) \big )$ represents the fused feature derived from the shallow and deep ViT features $( F _ { V } ^ { s }$ and $F _ { V } ^ { d } )$ of the frozen SAM encoder. Here, Cat denotes concatenation, $C _ { 3 \times 3 } ^ { d }$ denotes a $3 \times 3$ depth-wise convolution, and ⊙ indicates the dot product.

![](images/e6a2845cd321be29e57903a697fff35b2a9598919ed148ae9d10cfa8591a41e9.jpg)  
Fig. 4. Architecture of the learnable filter W in the FDFB, where $A P$ and $\check { M } { } ^ { P }$ denote average pooling and max pooling, respectively. PConv refers to point-wise convolutions.

1) Frequency Dynamic Filtering Block: The Frequency Dynamic Filtering Block (FDFB) enhances prediction clarity by adaptively disentangling target features from global noise in the frequency domain. Specifically, the Fast Fourier Transform (FFT) $\mathcal F ( \cdot )$ converts the fused spatial feature representation D into complex frequency features. For each channel $c ,$ this is expressed as:

$$
Z _ { c } = \mathcal { F } ( D _ { c } ) .\tag{4}
$$

The corresponding magnitude $A ( Z _ { c } )$ and phase $\phi ( Z _ { c } )$ are computed as:

$$
A ( Z _ { c } ) = \sqrt { ( \mathrm { R e } ( Z _ { c } ) ) ^ { 2 } + ( \mathrm { I m } ( Z _ { c } ) ) ^ { 2 } } ,\tag{5}
$$

$$
\phi ( Z _ { c } ) = \arctan 2 \bigl ( \mathrm { I m } ( Z _ { c } ) , \mathrm { R e } ( Z _ { c } ) \bigr ) ,\tag{6}
$$

where $\operatorname { R e } ( { \mathord { \cdot } } )$ and Im(·) denote the real and imaginary parts, respectively.

The learnable filter W, illustrated in Fig. 4, is then applied to the real and imaginary components to enable adaptive separation of target features from irrelevant noise:

$$
[ \hat { A } ( Z _ { c } ) , \hat { \phi } ( Z _ { c } ) ] = [ { \mathcal { W } } _ { A } ( A ( Z _ { c } ) ) + A ( Z _ { c } ) , { \mathcal { W } } _ { \phi } ( \phi ( Z _ { c } ) ) + \phi ( Z _ { c } ) ] ,\tag{7}
$$

Note that $\mathcal { W } _ { A }$ and $\mathcal { W } _ { \phi }$ share the same architecture but do not share parameters. The cross-enhanced frequency-domain representation is obtained as:

$$
\hat { Z } _ { c } = \hat { A } ( Z _ { c } ) \cdot \cos ( \hat { \phi } ( Z _ { c } ) ) + j \hat { A } ( Z _ { c } ) \cdot \sin ( \hat { \phi } ( Z _ { c } ) ) ,\tag{8}
$$

where $j$ denotes the imaginary unit, so that $\hat { Z } _ { c }$ is a complex spectrum whose real and imaginary parts are formed from the refined magnitude and phase. The corresponding spatialdomain representation is then reconstructed via inverse FFT:

$$
\hat { D } _ { f } = \mathcal { F } ^ { - 1 } ( \hat { \mathcal { Z } } ) ,\tag{9}
$$

where $\hat { \mathcal { Z } } = [ \hat { Z } _ { 1 } ; \cdot \cdot \cdot ; \hat { Z } _ { c } ; \cdot \cdot \cdot ; \hat { Z } _ { C } ] \in \mathbb { C } ^ { H \times ( \lfloor W / 2 \rfloor + 1 ) \times C }$ (the half-spectrum width of the real FFT).

2) Lightweight Spatial Enhancement Block: The Lightweight Spatial Enhancement Block (LSEB) operates in parallel with the frequency-domain processing branch to capture complementary spatial details. By employing SCConv $C _ { s c }$ [42], this block efficiently suppresses spatial redundancy. The spatial component $D _ { s }$ is computed as:

$$
D _ { s } = C _ { s c } \big ( C _ { 3 \times 3 } ( C _ { s c } ( C _ { 3 \times 3 } ( D ) ) ) + D \big ) .\tag{10}
$$

To combine information from both domains, the reconstructed frequency feature $\hat { D } _ { f }$ and the spatial feature $D _ { s }$ are fused via a linear projection and a residual connection, producing the dual-domain feature $D ^ { \prime } { : }$

$$
D ^ { \prime } = C _ { 1 \times 1 } \big ( \mathrm { C a t } ( \hat { D } _ { f } , D _ { s } ) \big ) + D .\tag{11}
$$

Finally, the output $M _ { d }$ of DDAM is obtained by modulating the output tokens $T _ { s } ~ \in ~ \mathbb { R } ^ { 5 \times 2 5 6 }$ from the frozen SAM with the fused feature $D ^ { \prime }$ (here $T _ { s }$ collects the five mask-carrying tokens—the four mask tokens together with the appended high-quality token—out of the six tokens produced by the decoder; the remaining IoU token is reserved for mask-quality prediction and is not modulated):

$$
M _ { d } = T _ { s } \odot D ^ { \prime } ,\tag{12}
$$

where ⊙ denotes the dot product with broadcasting. Concretely, as in SAM and HQ-SAM, each of the five mask tokens in $T _ { s }$ is first transformed by a hypernetwork MLP and then contracted with the fused per-pixel feature $D ^ { \prime }$ along the channel dimension, producing one mask map per token; $\odot$ abbreviates this token-to-mask operation, which differs from the element-wise product used earlier. Together, these components enable FSANet to effectively learn categoryspecific noise features in real-world scenarios, as discussed in section II.

## C. Edge Estimation Module

The Edge Estimation Module (EEM) is designed to improve boundary segmentation accuracy. It generates the final predicted boundary mask $B _ { p }$ by integrating outputs from SPM, and DDAM. The formulation is given by:

$$
B _ { p } = \sigma _ { s } ( \mathrm { M L P } ( M _ { d } + \sigma _ { g } ( C _ { 1 \times 1 } ( \mathrm { C a t } ( F _ { S } ^ { i } ) ) ) ) ,\tag{13}
$$

where $\sigma _ { s }$ denotes the sigmoid function, MLP represents a Multilayer Perceptron.

## D. Loss Function

Accurate segmentation requires not only precise pixel-wise classification but also sharp delineation of object boundaries, particularly in complex or noisy scenes. To this end, we adopt a composite loss function that balances global prediction accuracy with explicit boundary refinement. Specifically, we employ Binary Cross-Entropy (BCE) loss L<sub>BCE</sub> [43] to supervise pixel-level classification, and Dice loss $\mathcal { L } _ { \mathrm { D i c e } }$ [44] to mitigate class imbalance and enhance region-level consistency. Despite their effectiveness, these losses often provide insufficient constraints on boundary localization, leading to blurred or structurally inconsistent edges [45].

To explicitly enforce accurate boundary localization, we introduce a Boundary Loss term, defined as:

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { B o u n d a r y } } = - \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \Big [ M _ { G } ^ { b } [ i ] \cdot \log ( M _ { P } ^ { b } [ i ] ) } } \\ & { } & { + ( 1 - M _ { G } ^ { b } [ i ] ) \cdot \log ( 1 - M _ { P } ^ { b } [ i ] ) \Big ] , } \end{array}\tag{14}
$$

where $M _ { P } ^ { b } [ i ]$ represents the predicted value of the i-th pixel in the boundary mask, $M _ { G } ^ { b } [ i ]$ denotes the corresponding ground truth value, and M is the total number of boundary pixels.

The overall training objective for FSANet is formulated as a weighted combination of the three loss components:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \alpha \cdot \mathcal { L } _ { \mathrm { D i c e } } + \beta \cdot \mathcal { L } _ { \mathrm { B o u n d a r y } } ,\tag{15}
$$

where α and $\beta$ are hyperparameters that control the contribution of each term. An ablation study on hyperparameter selection is provided in Fig. 7.

## E. The SceneX Dataset

To facilitate the development of robust segmentation models, we introduce SceneX, a novel open-source dataset comprising 11,000 high-quality images collected from ten diverse real-world scenarios. As illustrated in Fig. 5 (a), this dataset provides a comprehensive resource for training advanced segmentation models. The images are sourced from both online repositories and original captures, ensuring broad scene diversity and real-world relevance [46], [47]. The dataset encompasses substantial variations in lighting conditions, camera perspectives, and weather effects, making it particularly suitable for developing adaptable and noise-resistant segmentation models.

Image Collection: Image resolutions in SceneX range from 200 × 125 to 4000 × 3000, with an average resolution of 1041 × 732. Compared to widely used benchmarks such as

![](images/ad5372bbf60bdd6aa929266bb10e6c24248b13faedb444ce1b4ce699be058fb8.jpg)

![](images/9492b962f35093d03db7b8d2a7fec2d15377df834803b1f159427e6191d37298.jpg)  
Fig. 5. (a) Major category distribution and representative subcategory examples from SceneX. (b) Concavity analysis comparing SceneX with existing datasets. The X-axis represents Concavity, measuring mask shape complexity (higher values indicate more irregular boundaries). The Y-axis shows the Percent of Masks, indicating the proportion of masks at each concavity level.

DUTS [48] (372 × 322) and COIFT [49] (600 × 488), SceneX offers significantly higher resolution, thereby facilitating more effective multi-scale learning for segmentation models operating at varying levels of detail.

Annotation Methodology: We employ a semi-automated annotation pipeline to ensure high-quality mask generation. Initial object localization is performed using saliency detection techniques, followed by segmentation using five representative methods: EdgeSAM [50], EfficientSAM [27], SAM [18], HQ-SAM [45], and TransDeep [9]. From the generated candidates, we select the mask with the highest Intersection over Union (IoU) score. Subsequent refinements are applied using edge detection tools from OpenCV [51], followed by manual validation to ensure annotation accuracy. This hybrid approach combining semi-automated techniques with human verification significantly enhances mask quality, producing precise and reliable segmentation annotations.

Annotation Reliability: Because the pipeline is seeded by segmentation models, we verify that the resulting masks do not inherit a bias towards strong image gradients. Using 250 masks per dataset and model-free shape statistics, we compare SceneX with three datasets whose masks are drawn manually by experts, DIS5K-VD [52], ThinObject5K-TE [49] and COIFT [49]. SceneX has the lowest boundary-gradient ratio of the four (2.95 against 3.55, 3.64 and 6.36), so its boundaries follow image gradients no more closely than manually drawn ones do, and its thin-structure ratio (0.182) lies inside the range of the manual sets (0.168 to 0.369). Its masks are topologically simpler, carrying fewer holes per mask, which reflects both the pipeline and the fact that DIS5K and ThinObject5K are built around intricate objects. We release the automatic and the manually refined subsets so that annotator agreement can be measured directly.

Mask Properties: A distinctive property of SceneX is the geometric complexity of its masks, quantified by concavity<sup>2</sup>. As shown in Fig. 5 (b), a substantial proportion of masks exhibit concavity values between 0.6 and 0.8, indicating prevalent complex shapes and irregular boundaries. This characteristic is crucial for training models to handle intricate object structures. To maintain balanced object size distribution, we categorize masks into size-based bins and perform stratified sampling within each bin. This strategy prevents bias toward specific scales and enables SceneX to effectively support the training of segmentation models across a wide spectrum of object sizes.

## IV. EXPERIMENTS

## A. Experimental Settings

Datasets. Following established evaluation protocols [18], [45], we conduct comprehensive experiments across ten publicly available datasets characterized by high-quality, finegrained annotations (averaging 7.4k masks per dataset across over 1,000 diverse categories). To ensure robustness and broad generalization, our training set integrates the training split of our proposed SceneX with six widely adopted benchmarks: DIS5K-TR [52] for objects with varying structural complexities; ThinObject5K-TR [49] for fine, elongated structures; FSS-1000 [53] for diverse and previously unseen classes; ECSSD [54] for objects in structurally complex backgrounds; MSRA-10K [55] for large-scale pixel-level saliency; and DUTS [48] for challenging scenarios. Here HQSeg-44K denotes the high-quality benchmark suite introduced by HQ-SAM [45], on whose held-out split Table VIII reports results— disjoint from the data used for training. For evaluation, we utilize the validation split of SceneX, DIS5K-VD [52], and ThinObject5K-TE [49], complemented by COIFT [49] and HR-SOD [56] to specifically address the challenges of thin object segmentation in heterogeneous backgrounds and highresolution saliency detection, respectively. Furthermore, we assess zero-shot generalization on the COCO benchmark [10] for common objects and the SGinW [57] benchmark to rigorously test adaptability in open-world scenarios.

Data Processing The data preprocessing pipeline plays a critical role in ensuring that the model is exposed to a diverse range of inputs and is capable of generalizing across various scenarios. We begin by loading the data using separate data loaders for training and validation, allowing for efficient batching and transformation. This separation ensures that the training data is properly utilized while preventing any data leakage into the validation phase. For data augmentation, we apply a series of transformations to the training set to enhance the model’s robustness to variations in object scale, orientation, and background complexity. Specifically, we apply random flipping with a probability of 50% and large-scale jittering, which randomly varies the scale between 0.1 and 2. These transformations help the model adapt to different object scales and orientations, improving its ability to generalize to unseen data. In addition to geometric augmentations, we compute an Image Structure Prior (ISP) for each image based on its maximum and minimum pixel values. This step, inspired by recent work [38], enhances the model’s robustness against complex backgrounds by providing a normalized representation of the image’s structure. The ISP helps the model to better distinguish between foreground and background elements, improving segmentation accuracy in challenging real-world scenarios. To handle images of varying sizes, we implement a padding strategy during batching. Each image is padded to match the size of the largest image in the batch, ensuring that all images and their corresponding labels are consistently aligned. This approach allows the model to process batches efficiently without requiring resizing or cropping, which could potentially lead to information loss or distortion in the data. By ensuring consistency in image dimensions across the batch, we improve the efficiency and accuracy of the training process, while also maintaining the integrity of the input data.

Training Strategy. We keep the pretrained SAM frozen throughout training to preserve its rich visual representations learned from large-scale datasets. Only the newly introduced parameters in FSANet, designed to enhance segmentation under complex scenarios, are optimized. The model is trained for 12 epochs with an initial learning rate of 0.001, which is decayed after epoch 10 to enhance training stability and prevent overfitting. All experiments are conducted on an NVIDIA RTX A6000 GPU, using the maximum feasible batch size for efficient GPU utilization and accelerated convergence. This setup ensures both training efficiency and competitive performance across standard benchmarks. For a controlled architectural comparison, every method in Table II is retrained on this identical training set under the same schedule, resolution, and box prompts—SAM-based methods from their corresponding SAM initialization—so the reported differences isolate the effect of architectural design rather than training data or supervision.

Inference Protocol. During inference, the boundary loss prediction branch is disabled to simplify the evaluation process and focus on final mask generation. SAM-generated masks are fused at the logit level with predictions from FSANet, producing intermediate results at a resolution of 256 × 256. These are subsequently upsampled to 1024 × 1024 using bilinear interpolation. This hierarchical refinement strategy preserves structural integrity while maintaining high segmentation accuracy in the final output.

Evaluation Metrics. To accurately quantify improvements in mask quality, instead of only employing the standard mask AP or mask mIoU, we also adopt the boundary metric mBIoU to measure contour alignment. Formally, let $N _ { c }$ be the number of classes, and $P _ { c } , G _ { c }$ denote the predicted and ground truth masks for class c. The mIoU assesses region-based accuracy:

$$
\mathrm { m I o U } = \frac { 1 } { N _ { c } } \sum _ { c = 1 } ^ { N _ { c } } { \frac { | P _ { c } \cap G _ { c } | } { | P _ { c } \cup G _ { c } | } } .\tag{16}
$$

Complementarily, mBIoU computes the IoU strictly within a distance d from the contours (denoted as regions $P _ { c } ^ { \dot { d } }$ and $G _ { c } ^ { d } )$

$$
\mathrm { m B I o U } = \frac { 1 } { N _ { c } } \sum _ { c = 1 } ^ { N _ { c } } { \frac { | P _ { c } ^ { d } \cap G _ { c } ^ { d } | } { | P _ { c } ^ { d } \cup G _ { c } ^ { d } | } } .\tag{17}
$$

## B. Implementation Details

All experiments are implemented using PyTorch [58] (v1.13.1+cu116) to ensure reproducibility in fine-grained, realworld segmentation research. To maintain consistency and facilitate benchmark comparisons, we adopt the SAM [18] inference pipeline and incorporate the HQ-output token from HQ-SAM [45] for efficient mask prediction. This design enables our model to leverage the flexibility and precision of the SAM framework while enhancing its performance to handle more complex segmentation tasks.

For bounding box prompt evaluations, identical bounding boxes are provided to all SAM variants with single-mask output mode to ensure fair comparisons.

## C. Comparisons with the State-of-the-art Methods

1) Quantity Analysis: As demonstrated in Table II, we compared FSANet with various types of models across five datasets, emphasizing its notable advantages in several key aspects:

In terms of Accuracy: FSANet outperforms others on most of the datasets, with an 89.0% mean IoU on average, improving by 1.1% compared to the HQ-SAM. Particularly, on the fine-grained ThinObject dataset, the Base, Large, and Huge versions of FSANet significantly outperform other methods. Meanwhile, FSANet demonstrates the strongest robustness on the SceneX dataset.

In terms of Efficiency: FSANet’s parameter scale is comparable to the HQ-SAM series, achieves significantly superior performance, and possesses considerably fewer parameters than VPD [59]. For instance, FSANet-L on the DIS dataset increases mIoU from 78.6 to 80.2 with a minimal increase of only ∼0.035M parameters, indicating its enhanced learning capacity compared to models relying on isolated spatialdomain learning.

Table III reports the full cost profile measured on a single RTX A6000 at 1024 × 1024 input. The three modules add 0.035 M trainable parameters and about 2 GFLOPs, which is 0.07% of the ViT-H budget, leave peak memory unchanged and reduce throughput by 1 to 3%. Measured in isolation, the forward and inverse FFT together cost 0.34 GFLOPs and under 0.5 ms, and the 256×256 feature map yields a 256×129 half spectrum, so the non-square shape is produced by the real FFT itself and requires no padding.

Results on COCO: As shown in Table IV, FSANet outperforms FastSAM by 10.5%, achieving the best scores across various object sizes, with only ∼1.63M learnable parameters (cf. Table II), approximately 17% of EdgeSAM, while leading EdgeSAM by 3.7% in average precision (AP). At the same time, FSANet surpasses the state-of-the-art model SAM2.1 by 0.6%, demonstrating strong capabilities.

TABLE II  
FSANET VS. SAM-RELATED MODELS ON OUR PROPOSED SCENEX AND FOUR EXTREMELY FINE-GRAINED BENCHMARKS, WHERE THE LIGHT PURPLEREPRESENTS OUR MODEL.
<table><tr><td rowspan="2">Model</td><td colspan="2">SceneX</td><td colspan="2">DIS</td><td colspan="2">COIFT</td><td colspan="2">HRSOD</td><td colspan="2">ThinObject</td><td colspan="2">Average</td><td colspan="2">Params</td></tr><tr><td>mIoU</td><td>mBIoU</td><td>mIoU</td><td>mBIoU</td><td>mIoU</td><td>mBIoU</td><td>mIoU</td><td>mBIoU</td><td>mIoU</td><td>mBIoU</td><td>mIoU</td><td>mBIoU</td><td>Total (M)</td><td>Learnable (M)</td></tr><tr><td>VPD [59]</td><td>77.2</td><td>61.5</td><td>63.2</td><td>67.8</td><td>86.2</td><td>80.6</td><td>86.3</td><td>76.9</td><td>60.2</td><td>54.3</td><td>74.6</td><td>68.2</td><td>901.8</td><td>867.6</td></tr><tr><td>MaskFormer [60]</td><td>78.6</td><td>62.8</td><td>61.7</td><td>66.5</td><td>87.4</td><td>81.2</td><td>88.8</td><td>77.2</td><td>62.1</td><td>56.9</td><td>75.7</td><td>68.9</td><td>16.5</td><td>16.5</td></tr><tr><td>SAM-B [18]</td><td>79.9</td><td>63.9</td><td>53.6</td><td>45.1</td><td>87.9</td><td>81.3</td><td>86.1</td><td>76.9</td><td>54.1</td><td>44.8</td><td>72.3</td><td>62.4</td><td>93.7</td><td>93.7</td></tr><tr><td>HQ-SAM-B [45]</td><td>81.4</td><td>67.1</td><td>76.3</td><td>68.3</td><td>93.1</td><td>88.0</td><td>91.2</td><td>84.1</td><td>84.5</td><td>73.4</td><td>85.3</td><td>76.2</td><td>94.8</td><td>1.07</td></tr><tr><td>FSANet-B (Ours)</td><td>81.1</td><td>67.1</td><td>77.7</td><td>70.4</td><td>93.7</td><td>88.7</td><td>91.4</td><td>83.9</td><td>87.4</td><td>77.6</td><td>86.3</td><td>77.5</td><td>94.8</td><td>1.11</td></tr><tr><td>SAM-L [18]</td><td>83.3</td><td>68.7</td><td>62.0</td><td>52.8</td><td>92.1</td><td>86.5</td><td>90.2</td><td>83.1</td><td>73.6</td><td>61.8</td><td>80.2</td><td>70.6</td><td>312.3</td><td>312.3</td></tr><tr><td>HQ-SAM-L [45]</td><td>83.9</td><td>70.2</td><td>78.6</td><td>70.4</td><td>94.8</td><td>90.1</td><td>93.6</td><td>86.9</td><td>89.5</td><td>79.9</td><td>88.0</td><td>79.5</td><td>313.7</td><td>1.33</td></tr><tr><td>FSANet-L (Ours)</td><td>84.6</td><td>70.9</td><td>80.2</td><td>73.7</td><td>94.6</td><td>90.2</td><td>93.0</td><td>86.9</td><td>91.3</td><td>83.4</td><td>88.7</td><td>81.0</td><td>313.7</td><td>1.37</td></tr><tr><td>SAM-H [18]</td><td>83.5</td><td>69.3</td><td>57.1</td><td>49.3</td><td>90.8</td><td>85.6</td><td>87.0</td><td>80.1</td><td>68.4</td><td>58.3</td><td>77.4</td><td>68.5</td><td>641.1</td><td>641.1</td></tr><tr><td>HQ-SAM-H [45]</td><td>83.8</td><td>69.2</td><td>79.2</td><td>71.4</td><td>95.0</td><td>90.3</td><td>92.1</td><td>84.7</td><td>89.2</td><td>79.6</td><td>87.9</td><td>79.0</td><td>642.7</td><td>1.60</td></tr><tr><td>FSANet-H (Ours)</td><td>84.4</td><td>70.6</td><td>81.4</td><td>75.0</td><td>95.0</td><td>90.6</td><td>92.6</td><td>86.7</td><td>91.8</td><td>84.1</td><td>89.0</td><td>81.4</td><td>642.7</td><td>1.63</td></tr></table>

TABLE III

EFFICIENCY PROFILE MEASURED ON A SINGLE NVIDIA RTX A6000 AT1024 × 1024 INPUT.
<table><tr><td>Backbone</td><td>Method</td><td>Params (M)</td><td>GFLOPs</td><td>Peak memory</td><td>FPS</td></tr><tr><td>ViT-B</td><td>HQ-SAM</td><td>94.81</td><td>495</td><td>0.5 GB</td><td>8.60</td></tr><tr><td>ViT-B</td><td>FSANet</td><td>94.84</td><td>497</td><td>0.5 GB</td><td>8.41</td></tr><tr><td>ViT-H</td><td>HQ-SAM</td><td>642.69</td><td>2993</td><td>2.8 GB</td><td>2.05</td></tr><tr><td>ViT-H</td><td>FSANet</td><td>642.72</td><td>2995</td><td>2.8 GB</td><td>2.03</td></tr></table>

TABLE IV

ZERO-SHOT INSTANCE SEGMENTATION RESULTS ON COCO BENCHMARK PROMPTED WITH GROUNDING DINO [61] BOXES.
<table><tr><td rowspan="2">Model</td><td colspan="4">GroundingDINO</td><td rowspan="2">Params Learnable (M)</td></tr><tr><td> $A P$ </td><td> $A P ^ { \mathrm { S } }$ </td><td> $\mathrm { \check { \boldsymbol { A } } } \boldsymbol { P } ^ { \mathrm { M } }$ </td><td> $A P ^ { \mathrm { S } }$ </td></tr><tr><td>FastSAM [62]</td><td>35.7</td><td>22.1</td><td>42.1</td><td>48.2</td><td>68.0</td></tr><tr><td>MobileSAM [19]</td><td>41.4</td><td>25.7</td><td>45.6</td><td>60.5</td><td>21.9</td></tr><tr><td>SlimSAM-77 [20]</td><td>42.1</td><td>27.1</td><td>46.0</td><td>59.6</td><td>9.8</td></tr><tr><td>TinySAM [63]</td><td>42.4</td><td>27.4</td><td>46.5</td><td>60.6</td><td>10.1</td></tr><tr><td>EdgeSAM [50]</td><td>42.5</td><td>27.4</td><td>46.9</td><td>59.5</td><td>9.6</td></tr><tr><td>SlimSAM-50 [20]</td><td>43.5</td><td>28.7</td><td>47.8</td><td>60.8</td><td>28.0</td></tr><tr><td>RepViT-SAM [64]</td><td>43.8</td><td>28.3</td><td>48.2</td><td>61.6</td><td>27.2</td></tr><tr><td>EfficientSAM-L0 [27]</td><td>46.0</td><td>29.2</td><td>50.3</td><td>65.7</td><td>34.8</td></tr><tr><td>SAM2.1-L [65]</td><td>45.6</td><td>28.3</td><td>50.7</td><td>64.2</td><td>224.3</td></tr><tr><td>SAM-H [18]</td><td>44.9</td><td>29.6</td><td>49.4</td><td>62.3</td><td>641.1</td></tr><tr><td>HQ-SAM [45]</td><td>45.9</td><td>28.8</td><td>50.1</td><td>64.0</td><td>1.60</td></tr><tr><td>FSANet(Ours)</td><td>46.2</td><td>28.9</td><td>50.7</td><td>64.5</td><td>1.63</td></tr></table>

Results on SegInW: As presented in Table V, we evaluate the zero-shot segmentation performance on the SegInW benchmark, which comprises 25 distinct unstructured segmentation tasks. FSANet achieves an overall Mean Average Precision (Mean AP) of 49.7%, surpassing the SAM (48.7%) by 1.0%. This result underscores our model’s exceptional ability to generalize across diverse real-world scenarios. Compared to state-of-the-art methods, FSANet demonstrates varying degrees of improvement, particularly distinguishing itself in the segmentation of everyday objects. For instance, it outperforms competing models in categories such as Electric-Shaver, Hand, and Strawberry, validating its robustness in processing complex visual data.

2) Quality Analysis: As illustrated in Fig. 6, we present a qualitative comparison of segmentation results across multiple datasets. While state-of-the-art models like SAM, HQ-SAM, and SAM2 often struggle under complex environmental conditions, FSANet demonstrates superior robustness and semantic understanding. First, in scenarios with complex background clutter (Row 1), competitors tend to over-segment, confusing the statue’s wings with the surrounding foliage. In contrast, our method precisely delineates the object boundaries, effectively suppressing irrelevant background textures. Second, under adverse lighting conditions with strong optical glare (Row 2), spatial-only models are distracted by the light beams, resulting in erroneous masks that include the glare itself. Our model, however, correctly identifies the physical structure of the street lamp, ignoring the visual artifacts. Third, in severe weather scenarios like the sandstorm (Row 3), where visibility is degraded and contrast is low, competitors produce fragmented and incomplete masks. Our approach successfully recovers the holistic structure of the object, verifying its capability to handle extreme visual degradation. Finally, in extremely low-light environments with negligible foreground-background contrast (Row 4), baseline models fail to discern the target boundaries, resulting in vague or missed detections. Our approach accurately captures the fine-grained silhouette of the target, demonstrating remarkable robustness to severe illumination changes. Overall, these results confirm that FSANet not only preserves internal structural integrity but also minimizes false segmentation in challenging, unseen environments.

## D. Ablation Study

To verify the effectiveness of each component, we conducted a comprehensive ablation study on the DUTS dataset. To ensure statistical reliability, we report the mean IoU and standard deviation across multiple fixed-seed runs in Table VI.

Analysis of SPM: SPM significantly enhances performance by mitigating global noise and providing structural priors to EEM. Without ISP, SPM contributes only a 0.63% mIoU gain; with ISP, the gain increases to 1.38%. Together with EEM, SPM achieves a 1.59% improvement. This progressive gain substantiates the synergistic effect between global context modeling and local structural refinement, confirming that SPM functions not merely as a feature extractor, but as a critical structural hub that aligns multi-scale representations.

TABLE V  
ZERO-SHOT SEGMENTATION PERFORMANCE ON THE SEGINW BENCHMARK (25 DATASETS) USING FSANET-L.
<table><tr><td rowspan=1 colspan=1>Models</td><td rowspan=1 colspan=8>ViT    GroundingDINO Text-Encoder   Mean AP   Airplane-Parts  Bottles  Brain-Tumor     Chicken</td><td rowspan=1 colspan=1>Cows</td></tr><tr><td rowspan=1 colspan=1>SAM [18]</td><td rowspan=1 colspan=1>h</td><td rowspan=1 colspan=1>swin-b</td><td rowspan=1 colspan=1>bert</td><td rowspan=1 colspan=1>48.7</td><td rowspan=1 colspan=1>37.2</td><td rowspan=1 colspan=1>65.4</td><td rowspan=1 colspan=1>11.9</td><td rowspan=1 colspan=1>84.5</td><td rowspan=1 colspan=1>47.5</td></tr><tr><td rowspan=1 colspan=1>HQ-SA [45]</td><td rowspan=1 colspan=1>h</td><td rowspan=1 colspan=1>swin-b</td><td rowspan=1 colspan=1>bert</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>37.6</td><td rowspan=1 colspan=1>66.3</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>84.5</td><td rowspan=1 colspan=1>47.8</td></tr><tr><td rowspan=1 colspan=1>SAM2 [65]</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>swin-b</td><td rowspan=1 colspan=1>bert</td><td rowspan=1 colspan=1>49.5</td><td rowspan=1 colspan=1>38.3</td><td rowspan=1 colspan=1>67.1</td><td rowspan=1 colspan=1>12.1</td><td rowspan=1 colspan=1>80.7</td><td rowspan=1 colspan=1>52.8</td></tr><tr><td rowspan=1 colspan=1>FSANet</td><td rowspan=1 colspan=1>h</td><td rowspan=1 colspan=1>swin-b</td><td rowspan=1 colspan=1>bert</td><td rowspan=1 colspan=1>49.7</td><td rowspan=1 colspan=1>36.4</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>11.8</td><td rowspan=1 colspan=1>85.6</td><td rowspan=1 colspan=1>47.7</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Electric-Shaver</td><td rowspan=1 colspan=1>Elephants</td><td rowspan=1 colspan=1>Fruits</td><td rowspan=1 colspan=1>Garbage</td><td rowspan=1 colspan=1>Ginger-Garlic</td><td rowspan=1 colspan=1>Hand-Metal</td><td rowspan=1 colspan=4>Hand  House-Parts  HouseHold-Items Nutterfly-Squireel</td></tr><tr><td rowspan=1 colspan=1>71.7</td><td rowspan=1 colspan=1>77.9</td><td rowspan=1 colspan=1>82.3</td><td rowspan=1 colspan=1>24.0</td><td rowspan=1 colspan=1>45.8</td><td rowspan=1 colspan=1>81.2</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>8.4</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>71.3</td></tr><tr><td rowspan=1 colspan=1>72.1</td><td rowspan=1 colspan=1>77.5</td><td rowspan=1 colspan=1>82.3</td><td rowspan=1 colspan=1>25.0</td><td rowspan=1 colspan=1>45.6</td><td rowspan=1 colspan=1>81.2</td><td rowspan=1 colspan=1>74.8</td><td rowspan=1 colspan=1>8.5</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>77.1</td></tr><tr><td rowspan=1 colspan=1>72.0</td><td rowspan=1 colspan=1>78.2</td><td rowspan=1 colspan=1>83.3</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>45.7</td><td rowspan=1 colspan=1>73.7</td><td rowspan=1 colspan=1>77.6</td><td rowspan=1 colspan=1>8.6</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>84.1</td></tr><tr><td rowspan=1 colspan=1>72.4</td><td rowspan=1 colspan=1>77.6</td><td rowspan=1 colspan=1>82.0</td><td rowspan=1 colspan=1>24.3</td><td rowspan=1 colspan=1>46.1</td><td rowspan=1 colspan=1>77.0</td><td rowspan=1 colspan=1>81.2</td><td rowspan=1 colspan=1>8.6</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>77.6</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Phones</td><td rowspan=1 colspan=1>Poles</td><td rowspan=1 colspan=1>Puppies</td><td rowspan=1 colspan=1>Rail</td><td rowspan=1 colspan=1>Salmon-Fillet</td><td rowspan=1 colspan=1>Strawberry</td><td rowspan=1 colspan=1>Tablets</td><td rowspan=1 colspan=1>Toolkits</td><td rowspan=1 colspan=1>Trash</td><td rowspan=1 colspan=1>Watermelon</td></tr><tr><td rowspan=1 colspan=1>35.4</td><td rowspan=1 colspan=1>23.3</td><td rowspan=1 colspan=1>50.1</td><td rowspan=1 colspan=1>8.7</td><td rowspan=1 colspan=1>32.9</td><td rowspan=1 colspan=1>83.5</td><td rowspan=1 colspan=1>29.8</td><td rowspan=1 colspan=1>20.8</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>64.2</td></tr><tr><td rowspan=1 colspan=1>35.3</td><td rowspan=1 colspan=1>20.1</td><td rowspan=1 colspan=1>50.1</td><td rowspan=1 colspan=1>7.7</td><td rowspan=1 colspan=1>42.2</td><td rowspan=1 colspan=1>85.6</td><td rowspan=1 colspan=1>29.7</td><td rowspan=1 colspan=1>21.8</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>65.6</td></tr><tr><td rowspan=1 colspan=1>34.6</td><td rowspan=1 colspan=1>28.8</td><td rowspan=1 colspan=1>48.9</td><td rowspan=1 colspan=1>14.3</td><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>83.7</td><td rowspan=1 colspan=1>29.1</td><td rowspan=1 colspan=1>20.1</td><td rowspan=1 colspan=1>28.4</td><td rowspan=1 colspan=1>66.0</td></tr><tr><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>21.1</td><td rowspan=1 colspan=1>50.1</td><td rowspan=1 colspan=1>8.4</td><td rowspan=1 colspan=1>42.3</td><td rowspan=1 colspan=1>85.8</td><td rowspan=1 colspan=1>29.4</td><td rowspan=1 colspan=1>21.2</td><td rowspan=1 colspan=1>29.8</td><td rowspan=1 colspan=1>65.2</td></tr></table>

![](images/a59d32a97b3474bda836bedb2ce1a515a566165a81144369e9739c23aa920e84.jpg)  
Image  
SAM  
HQ-SAM  
SAM2  
Ours  
Fig. 6. The segmentation comparison of SAM-related models across multiple datasets. The corner inset highlights segmentation details.

TABLE VI  
COMPONENT EFFECTIVENESS ON WIDELY-RECOGNIZED DATASET DUTS(VIT-B; PURPLE: FSANET, BLUE: BASELINE).
<table><tr><td>Model</td><td>SPM</td><td>DDAM</td><td>EEM</td><td>Mask Feature</td><td>mIoU</td><td>mBIoU</td></tr><tr><td>HQ-SAM [45]</td><td></td><td></td><td></td><td>7</td><td>86.53 (0.06)</td><td>73.08 (0.07)</td></tr><tr><td></td><td></td><td></td><td>√</td><td>√</td><td>87.84 (0.04)</td><td>75.15 (0.04)</td></tr><tr><td></td><td></td><td>L</td><td></td><td>√</td><td>88.04 (0.01)</td><td>75.69 (0.04)</td></tr><tr><td>Without FFT</td><td></td><td></td><td></td><td>√</td><td>86.96 (0.06)</td><td>73.21 (0.08)</td></tr><tr><td></td><td></td><td></td><td></td><td>√</td><td>87.91 (0.10)</td><td>75.18 (0.08)</td></tr><tr><td>Without ISP</td><td>V</td><td></td><td></td><td>V</td><td>87.16 (0.03)</td><td>75.06 (0.03)</td></tr><tr><td rowspan="3"></td><td></td><td>V</td><td></td><td>√</td><td>88.22 (0.13)</td><td>75.80 (0.13)</td></tr><tr><td>√</td><td></td><td>√</td><td>√</td><td>88.12 (0.10)</td><td>75.88 (0.08)</td></tr><tr><td></td><td>1</td><td>√</td><td>√</td><td>89.26 (0.07)</td><td>75.72 (0.06)</td></tr><tr><td>FSANet</td><td></td><td></td><td>√</td><td>√</td><td>89.61 (0.03)</td><td>75.78 (0.04)</td></tr></table>

Analysis of DDAM: DDAM offers a 1.51% mIoU and 2.61% mBIoU gain, validating the dual-domain decoding strategy. LSEB alone yields 86.96 mIoU, while the addition of FDFB (FFT-based) boosts performance to 88.04 mIoU. This significant improvement highlights the limitations of spatialonly processing and demonstrates that integrating frequencydomain constraints effectively captures global noise patterns that are indistinguishable in the spatial domain.

Analysis of EEM: EEM improves boundary perception, leading to ∼1.5% gains in both metrics. This result indicates that explicitly targeting boundary artifacts is essential for segmentation quality, as it resolves ambiguities in transition regions where standard pixel-wise classification typically fails.

TABLE VII  
COMPARISON OF DIFFERENT CONVOLUTION KERNEL SIZES OF SPM AND COMPRESS RATIO OF $C R A _ { m }$ ON SCENEX BENCHMARK WITH VIT-B BACKBONE.  
(a) Convolution kernel variants
<table><tr><td>Filter</td><td>mIoU</td><td>mBIoU</td></tr><tr><td> $\mathrm { 3 \times 3 }$ </td><td> $5 \times 5$  84.90</td><td>71.09</td></tr><tr><td> $\mathrm { 3 \times 3 }$   $7 \times 7$ </td><td>84.95</td><td>71.25</td></tr><tr><td> $\mathrm { 3 \times 3 }$   $9 \times 9$ </td><td>85.02</td><td>71.27</td></tr><tr><td> $5 \times 5$   $7 \times 7$   $9 \times 9$ </td><td>85.08</td><td>71.15</td></tr><tr><td> $5 \times 5$ </td><td>84.98</td><td>71.15</td></tr><tr><td>7×7</td><td>9×9 85.09</td><td>71.36</td></tr></table>

<table><tr><td>Compress Ratio</td><td>mIoU</td><td>mBIoU</td></tr><tr><td>1</td><td>84.99</td><td>71.09</td></tr><tr><td>2</td><td>85.12</td><td>71.26</td></tr><tr><td>4</td><td>84.72</td><td>70.79</td></tr><tr><td>8</td><td>84.84</td><td>71.01</td></tr><tr><td>16</td><td>84.89</td><td>71.20</td></tr><tr><td>32</td><td>85.04</td><td>71.24</td></tr></table>

Analysis of Different Convolution Kernels of SPM: As shown in table VII (a), $7 \times 7$ and $9 \times 9$ group convolutions outperform smaller kernels by ${ \sim } 0 . 2 \%$ . This finding underscores the importance of larger receptive fields in capturing longrange dependencies, which are requisite for distinguishing foreground objects from complex backgrounds.

Analysis of Compression Ratio of $C R A _ { m }$ in the DDAM: table VII (b) shows that compressing Q and K vectors improves global representation. A compression ratio of 2 (channels reduced to 16) improves mIoU by 0.13% over the full configuration. This suggests that a moderate reduction in channel redundancy acts as a regularizer, preventing the model from overfitting to high-frequency noise while maintaining computational efficiency.

![](images/5bea7023de1476c0a6d36abf5ef044d022a486e7e6db2675891ebadce2648441.jpg)  
Fig. 7. Visualization of Loss with mIoU-axis, $\mathcal { L } _ { \mathrm { D i c e } } ( \alpha { \cdot } \mathrm { a x i s } )$ and ${ \mathcal { L } } _ { \mathrm { B o u n d a r y } } ( \beta -$ axis) on the SceneX dataset processed by FSANet with ViT-B.

Boundary Loss and Dice Loss Analysis: Grid search (Fig. 7) reveals optimal performance at $\alpha : \beta = 1 : 0 . 7 5 .$ achieving 85.20 mIoU and 71.55 mBIoU. This ratio balances segmentation accuracy and boundary precision, underscoring the value of boundary-aware loss design.

Filter Selection. We evaluated the effect of various dynamic filters on model performance, as shown in Table VIII, including the Base Filter, SE block [66], ECA [67], and CBAM [68]. Compared to SE, ECA reduced parameter count by 250 with a 0.16% gain in mBIoU. CBAM, despite adding 196 parameters, achieved a notable 1.85% improvement, reaching 91.81% mIoU. Considering the trade-off between accuracy and efficiency, CBAM is adopted for its superior performance with minimal overhead.

TABLE VIII  
COMPARISON OF FILTER VARIANTS IN THE FDFB MODULE ON THE HQSEG-44K BENCHMARK USING A VIT-L BACKBONE. NOTE: LEARNABLE PARAMETER COUNTS ARE REPORTED AS THE NUMBER OF PARAMETERS (NOT MEMORY SIZE IN BYTES).
<table><tr><td>Filter</td><td>mIoU</td><td>mBIoU</td><td>Learnable Params</td></tr><tr><td>SE Block</td><td>89.96</td><td>81.41</td><td>1368125</td></tr><tr><td>ECA Block</td><td>90.12</td><td>81.79</td><td>1367875</td></tr><tr><td>CBAM Block</td><td>91.81</td><td>83.49</td><td>1368321</td></tr></table>

The full model achieves 89.61% mIoU and 75.78% mBIoU, outperforming the baseline by 3.08% and 2.7%, respectively, highlighting the efficacy of component integration in FSANet.

## E. Network Interpretability

As shown in Fig. 8, we visualize the output tokens generated by FSANet and HQ-SAM in the token-to-image crossattention layer of the mask decoder. The results indicate that compared to HQ-SAM, FSANet’s attention mechanism is more concentrated on the true structural boundaries of the target objects. Specifically, in the top-left example, the spatial domain effectively integrates semantically relevant branch information, while the frequency domain—guided by our designed cross-domain solver—suppresses background interference, thereby enhancing the accurate localization of target structures.

We further probe what the trained filter does, by hooking the FDFB and recording the ratio of output to input energy in the high band. The filter is adaptive rather than fixed: it amplifies the high band by 3.47× on blurred input and by 3.45× on haze, where high frequencies are recoverable detail, but by only 2.89× on additive noise, which is 0.88 of its gain on clean input. The phase pathway remains close to identity, so edge positions are preserved. This is the behaviour the module is designed for, recovering structure where it is missing while refusing to amplify noise.

## V. FUTURE WORK

Building upon the promising results of FSANet and the SceneX benchmark, our future research will focus on two strategic directions to further advance the field of robust segmentation.

Expansion of the SceneX Ecosystem. While the current iteration of SceneX covers ten challenging non-ideal scenarios, the complexity of real-world environments is virtually boundless. We intend to significantly augment the dataset by incorporating a broader spectrum of complex scenarios, including extreme weather conditions, rare visual artifacts, and highly dynamic occlusions. By expanding the semantic categories and increasing the diversity of adverse conditions, we aim to establish a more comprehensive and rigorous benchmark. This initiative is expected to facilitate the development of foundation models with superior generalization capabilities and drive community-wide progress in addressing long-tail distribution problems in segmentation tasks.

![](images/7150eb742e73ffc98506dedf1860635954a86dbd4456e9676cefcf04e4073a7c.jpg)  
Fig. 8. Visual comparison of cross-attention heatmaps between the original HQ-SAM output token and the FSANet output token in the final decoder layer. FSANet demonstrates stronger focus on object boundaries and structural regions that are misclassified by the original token.

Lightweight Architecture and Edge Deployment. Although FSANet achieves state-of-the-art performance, the reliance on sophisticated dual-domain processing and structural priors entails computational costs that may hinder deployment on resource-constrained platforms. To address this, we plan to explore lightweight backbone architectures and efficient attention mechanisms that maintain high accuracy while drastically reducing memory footprint and parameter count. Furthermore, we will investigate model compression techniques, such as knowledge distillation and quantization, to optimize FSANet for real-time inference. Our ultimate goal is to bridge the gap between high-performance research models and practical applications, enabling the deployment of robust segmentation systems on mobile and edge devices.

## VI. CONCLUSIONS

In conclusion, we introduce FSANet, the first segmentation framework with inherent noise-awareness, incorporating structural priors and dual-domain contextual understanding to effectively address the challenges of real-world segmentation. FSANet structure prior, dual-domain awareness, and edge estimation modules work synergistically to recover fine details, disentangle noise, and enhance boundary precision. FSANet outperforms state-of-the-art methods in fine-grained tasks and demonstrates strong robustness across diverse scenarios. Evaluations of seven benchmarks further validate its effectiveness. Additionally, the introduction of SceneX, with 10 categories, establishes a new benchmark for segmentation research. This work highlights the potential of prior knowledge and cross-domain enhancements, offering valuable insights for developing computation-efficient models.

## ACKNOWLEDGMENTS

## REFERENCES

[1] Z. Lu, C. She, W. Wang, and Q. Huang, “Lm-net: A light-weight and multi-scale network for medical image segmentation,” Computers in Biology and Medicine, vol. 168, p. 107717, 2024.

[2] S. Muralidhara, R. Schuster, and D. Stricker, “Domain-incremental semantic segmentation for autonomous driving under adverse driving conditions,” arXiv preprint arXiv:2501.05246, 2025.

[3] J. V. Hurtado and A. Valada, “Semantic scene segmentation for robotics,” in Deep learning for robot perception and cognition. Elsevier, 2022, pp. 279–311.

[4] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.

[5] K. M. Elgamily, M. Mohamed, A. M. Abou-Taleb, and M. M. Ata, “A novel w13 deep cnn structure for improved semantic segmentation of multiple objects in remote sensing imagery,” Neural Computing and Applications, pp. 1–31, 2025.

[6] A. Cao, Z. Li, J. Jomsky, A. F. Laine, and J. Guo, “Medsegmamba: 3d cnn-mamba hybrid architecture for brain segmentation,” arXiv preprint arXiv:2409.08307, 2024.

[7] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 12 077–12 090.

[8] S. Zheng, J. Lu, H. Zhao, X. Zhu, Z. Luo, Y. Wang, Y. Fu, J. Feng, T. Xiang, P. H. Torr et al., “Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 6881–6890.

[9] T. Chai, Z. Xiao, X. Shen, Q. Liu, N. Li, T. Guan, and J. Tian, “Transdeep: Transformer-integrated deeplabv3+ for image semantic segmentation,” IEEE Access, 2025.

[10] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in European conference on computer vision. Springer, 2014, pp. 740–755.

[11] R. Zhou, L. Zheng, C. Ren, and S. Wang, “Image segmentation algorithm in complex environment based on improved solov2,” in 2023 12th International Conference of Information and Communication Technology (ICTech). IEEE, 2023, pp. 581–585.

[12] B. Drayer and T. Brox, “Object detection, tracking, and motion segmentation for object-level video segmentation,” arXiv preprint arXiv:1608.03066, 2016.

[13] Z. Wang, Y. Zhang, K. M. Mosalam, Y. Gao, and S.-L. Huang, “Deep semantic segmentation for visual understanding on construction sites,” Computer-Aided Civil and Infrastructure Engineering, vol. 37, no. 2, pp. 145–162, 2022.

[14] Y. Li, W. Gao, G. Li, and S. Ma, “Saliency segmentation oriented deep image compression with novel bit allocation,” IEEE Transactions on Image Processing, 2024.

[15] J. Jain, J. Li, M. T. Chiu, A. Hassani, N. Orlov, and H. Shi, “Oneformer: One transformer to rule universal image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 2989–2998.

[16] B. Yan, Y. Jiang, J. Wu, D. Wang, P. Luo, Z. Yuan, and H. Lu, “Universal instance perception as object discovery and retrieval,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 15 325–15 336.

[17] T. Luddecke and A. Ecker, “Image segmentation using text and image¨ prompts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 7086–7096.

[18] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,

“Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 4015–4026.

[19] C. Zhang, D. Han, Y. Qiao, J. U. Kim, S.-H. Bae, S. Lee, and C. S. Hong, “Faster segment anything: Towards lightweight sam for mobile applications,” 2023. [Online]. Available: https: //arxiv.org/abs/2306.14289

[20] Z. Chen, G. Fang, X. Ma, and X. Wang, “Slimsam: 0.1% data makes segment anything slim,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 39 434–39 461. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2024/file/45a7ca247462d9e465ee88c8a302ca70-Paper-Conference.pdf

[21] Z. Chen and H. Zhu, “Visual quality evaluation for semantic segmentation: subjective assessment database and objective assessment measure,” IEEE Transactions on Image Processing, vol. 28, no. 12, pp. 5785–5796, 2019.

[22] K. Zhang, D. Li, W. Luo, W. Ren, and W. Liu, “Enhanced spatiotemporal interaction learning for video deraining: Faster and better,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 1287–1293, 2022.

[23] K. Zhang, R. Li, Y. Yu, W. Luo, and C. Li, “Deep dense multi-scale network for snow removal using semantic and depth priors,” IEEE Transactions on Image Processing, vol. 30, pp. 7419–7431, 2021.

[24] Z. Jin, Y. Qiu, K. Zhang, H. Li, and W. Luo, “Mb-taylorformer v2: Improved multi-branch linear transformer expanded by taylor formula for image restoration,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 7, pp. 5990–6005, 2025.

[25] H. Zhao, B. Wang, S. Zhao, T. Wang, K. Zhang, and W. Lu, “Echosr: Efficient context harnessing for lightweight image super-resolution,” Information Fusion, p. 104471, 2026.

[26] Y. Liu, X. Dong, Y. Lin, M. Ye, K. Zhang, and B. Du, “Condition-guided diffusion for multi-modal pedestrian trajectory prediction incorporating intention and interaction priors,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[27] Y. Xiong, B. Varadarajan, L. Wu, X. Xiang, F. Xiao, C. Zhu, X. Dai, D. Wang, F. Sun, F. Iandola et al., “Efficientsam: Leveraged masked image pretraining for efficient segment anything,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16 111–16 121.

[28] K. Kim, Y. Oh, and J. C. Ye, “Otseg: Multi-prompt sinkhorn attention for zero-shot semantic segmentation,” in European Conference on Computer Vision. Springer, 2024, pp. 200–217.

[29] D. Alexey, “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv: 2010.11929, 2020.

[30] X. Cong, J. Gui, J. Zhang, J. Hou, and H. Shen, “A semi-supervised nighttime dehazing baseline with spatial-frequency aware and realistic brightness constraint,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2631–2640.

[31] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset for semantic urban scene understanding,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 3213– 3223.

[32] A. Gupta, P. Dollar, and R. Girshick, “Lvis: A dataset for large vocabulary instance segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 5356– 5364.

[33] H. Qi, H. Zhou, J. Dong, and X. Dong, “Small sample image segmentation by coupling convolutions and transformers,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 7, pp. 5282– 5294, 2023.

[34] T. Wang, X. Qi, and G. Yang, “Polyp segmentation via semantic enhanced perceptual network,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 12, pp. 12 594–12 607, 2024.

[35] Z. Chang, X. Gao, N. Li, H. Zhou, and Y. Lu, “Drnet: Disentanglement and recombination network for few-shot semantic segmentation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 7, pp. 5560–5574, 2024.

[36] J. Wu, X. Li, X. Li, H. Ding, Y. Tong, and D. Tao, “Toward robust referring image segmentation,” IEEE Transactions on Image Processing, vol. 33, pp. 1782–1794, 2024.

[37] K. L. Nguyen, P. Delachartre, and M. Berthier, “Multi-grid phase field skin tumor segmentation in 3d ultrasound images,” IEEE Transactions on Image Processing, vol. 28, no. 8, pp. 3678–3687, 2019.

[38] N. Gao, X. Jiang, X. Zhang, and Y. Deng, “Efficient frequencydomain image deraining with contrastive regularization,” in European Conference on Computer Vision. Springer, 2024, pp. 240–257.

[39] D. Hendrycks and K. Gimpel, “Gaussian error linear units (gelus),” arXiv preprint arXiv:1606.08415, 2016.

[40] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4510–4520.

[41] B. Kang, S. Moon, Y. Cho, H. Yu, and S.-J. Kang, “Metaseg: Metaformer-based global contexts-aware network for efficient semantic segmentation,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2024, pp. 434–443.

[42] J. Li, Y. Wen, and L. He, “Scconv: Spatial and channel reconstruction convolution for feature redundancy,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 6153– 6162.

[43] G. E. Hinton, S. Osindero, and Y.-W. Teh, “A fast learning algorithm for deep belief nets,” Neural computation, vol. 18, no. 7, pp. 1527–1554, 2006.

[44] C. H. Sudre, W. Li, T. Vercauteren, S. Ourselin, and M. Jorge Cardoso, “Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations,” in International Workshop on Deep Learning in Medical Image Analysis. Springer, 2017, pp. 240–248.

[45] L. Ke, M. Ye, M. Danelljan, Y.-W. Tai, C.-K. Tang, F. Yu et al., “Segment anything in high quality,” Advances in Neural Information Processing Systems, vol. 36, pp. 29 914–29 934, 2023.

[46] Y. Shi, D. Liu, L. Zhang, Y. Tian, X. Xia, and X. Fu, “Zero-ig: Zero-shot illumination-guided joint denoising and adaptive enhancement for lowlight images,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 3015–3024.

[47] Y. Liao, Z. Su, X. Liang, and B. Qiu, “Hdp-net: Haze density prediction network for nighttime dehazing,” in Pacific Rim Conference on Multimedia. Springer, 2018, pp. 469–480.

[48] L. Wang, H. Lu, Y. Wang, M. Feng, D. Wang, B. Yin, and X. Ruan, “Learning to detect salient objects with image-level supervision,” in 2017 IEEE conference on computer vision and pattern recognition (CVPR). IEEE, 2017, pp. 3796–3805.

[49] J. H. Liew, S. Cohen, B. Price, L. Mai, and J. Feng, “Deep interactive thin object selection,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021, pp. 305–314.

[50] C. Zhou, X. Li, C. C. Loy, and B. Dai, “Edgesam: Prompt-in-theloop distillation for on-device deployment of sam,” arXiv preprint arXiv:2312.06660, 2023.

[51] G. Bradski, “The opencv library.” Dr. Dobb’s Journal: Software Tools for the Professional Programmer, vol. 25, no. 11, pp. 120–123, 2000.

[52] X. Qin, H. Dai, X. Hu, D.-P. Fan, L. Shao, and L. Van Gool, “Highly accurate dichotomous image segmentation,” in European Conference on Computer Vision. Springer, 2022, pp. 38–56.

[53] X. Li, T. Wei, Y. P. Chen, Y.-W. Tai, and C.-K. Tang, “Fss-1000: A 1000-class dataset for few-shot segmentation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 2869–2878.

[54] J. Shi, Q. Yan, L. Xu, and J. Jia, “Hierarchical image saliency detection on extended cssd,” IEEE transactions on pattern analysis and machine intelligence, vol. 38, no. 4, pp. 717–729, 2015.

[55] M.-M. Cheng, N. J. Mitra, X. Huang, P. H. Torr, and S.-M. Hu, “Global contrast based salient region detection,” IEEE transactions on pattern analysis and machine intelligence, vol. 37, no. 3, pp. 569–582, 2014.

[56] Y. Zeng, P. Zhang, J. Zhang, Z. Lin, and H. Lu, “Towards high-resolution salient object detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 7234–7243.

[57] X. Zou, Z.-Y. Dou, J. Yang, Z. Gan, L. Li, C. Li, X. Dai, H. Behl, J. Wang, L. Yuan et al., “Generalized decoding for pixel, image, and language,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 15 116–15 127.

[58] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[59] W. Zhao, Y. Rao, Z. Liu, B. Liu, J. Zhou, and J. Lu, “Unleashing text-to-image diffusion models for visual perception,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 5729–5739.

[60] B. Cheng, A. Schwing, and A. Kirillov, “Per-pixel classification is not all you need for semantic segmentation,” Advances in neural information processing systems, vol. 34, pp. 17 864–17 875, 2021.

[61] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision. Springer, 2024, pp. 38–55.

[62] X. Zhao, W. Ding, Y. An, Y. Du, T. Yu, M. Li, M. Tang, and J. Wang, “Fast segment anything,” arXiv preprint arXiv:2306.12156, 2023.

[63] H. Shu, W. Li, Y. Tang, Y. Zhang, Y. Chen, H. Li, Y. Wang, and X. Chen, “Tinysam: Pushing the envelope for efficient segment anything model,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 39, no. 19, 2025, pp. 20 470–20 478.

[64] A. Wang, H. Chen, Z. Lin, J. Han, and G. Ding, “Repvit-sam: Towards real-time segmenting anything,” arXiv preprint arXiv:2312.05760, 2023.

[65] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala,¨ N. Carion, C.-Y. Wu, R. Girshick, P. Dollar, and C. Feichtenhofer, “SAM 2: Segment anything in images and videos,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=Ha6RTeWMd0

[66] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 7132–7141.

[67] Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, “Eca-net: Efficient channel attention for deep convolutional neural networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 11 534–11 542.

[68] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “Cbam: Convolutional block attention module,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 3–19.

![](images/70d68a050611904063537602be4f1e87af8f8650207fb03c201a1ecb34c3c245.jpg)  
Ruibo Wang is currently pursuing the B.S. degree in Computer Science and Engineering with Delft University of Technology, Delft, Netherlands. Prior to this, he studied Computer Science and Technology at Hebei University of Technology. His research interests include computer vision, medical image analysis, and deep learning, with a specific focus on foundation models, image segmentation, and physics-guided image synthesis.

![](images/a4df565e27ee1f416d7d49d6b7b744f4ef88397c9335c4341fc70c851bc19ab9.jpg)

Ziyi Shen received the B.E. degree in medical information engineering from Jiangxi University of Chinese Medicine, Nanchang, China, in 2023. He is currently pursuing the master’s degree in biomedical engineering with the School of Biomedical Engineering, Southern Medical University, Guangzhou, China. His research interests include medical image analysis and deep learning.

![](images/3c41f834d82d68d640eea257972f45ac509a69911b25c4b1b510bf7c8cdf6caf.jpg)

Huaming Wu (Senior Member, IEEE) received the BE and MS degrees in electrical engineering from the Harbin Institute of Technology, China, in 2009 and 2011, respectively, and the PhD degree with the highest honor in computer science from Freie Universitat Berlin, Germany, in 2015. He is¨ currently a professor with the Center for Applied Mathematics, Tianjin University, China. His research interests include mobile cloud computing, edge computing, internet of things, deep learning, complex networks, and DNA storage. He currently serves as an Associate Editor for several IEEE journals, including IEEE Transactions on Dependable and Secure Computing, IEEE Transactions on Intelligent Transportation Systems, IEEE Transactions on Circuits and Systems for Video Technology, IEEE Transactions on Consumer Electronics, and IEEE Transactions on Technology and Society.

![](images/921dc0380c8d20571c22be9094c14cbe1a48dd518253abf2174eed211f2c2151.jpg)

Dong Liang (Senior Member, IEEE) received the bachelor’s degree in electronics engineering and the M.S. degree in signal and information processing from the Hefei University of Technology, Hefei, China, in 1998 and 2002, respectively, and the Ph.D. degree in pattern recognition and intelligent system from Shanghai Jiao Tong University, Shanghai, China, in 2006. He is currently a Professor with the Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China. From 2007 and 2011, he conducted postdoctoral research with the University of Hong Kong, Hong Kong, and with the University of Wisconsin-Milwaukee, Milwaukee, WI, USA. He is currently working on signal processing, machine learning, and biomedical imaging. He currently acts as an editorial member of the international journal IEEE Transactions on Medical Imaging and Magnetic Resonance in Medicine.

![](images/49a192d1e85fa873c5b75ca70c33f90d5f4c84fe4649c2852708be7c1749765c.jpg)  
Computer Interface, optimization etc.

Kun Shang received the B.S. degree from the Faculty of mathematics and statistics, Hubei University, in 2011. In 2018, He received the M.S. and Ph.D. degrees from Center for Applied Mathematics, Tianjin University. From 2018 to 2021, he was an Assistant professor of Mathematics with the School of Mathematics, Hunan University. He is currently an Associate Professor with the Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China. Recently, his research interests include Brain-inspired computing, Brain