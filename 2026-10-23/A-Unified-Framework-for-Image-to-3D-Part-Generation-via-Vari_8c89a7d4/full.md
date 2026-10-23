# A Unified Framework for Image-to-3D Part Generation via Variable Granularity

Qiwen Gu<sup>1,2,‡</sup> Xin Zhou<sup>2,3,‡</sup> Xiwu Chen<sup>2,B,†</sup> Zhongyang Zhu<sup>2</sup> Zhuo Zhang<sup>2</sup> Junqiao Zhao<sup>1,B</sup>

<sup>1</sup>Tongji University <sup>2</sup>Chongqing Afari Intelligent Drive Co., Ltd <sup>3</sup>Huazhong University of Science and Technology

2432178@tongji.edu.cn, xzhou03@hust.edu.cn

![](images/560fc3239a29a5306535a8445eae3e40b43e3c51a33c57546729ca9185e0857f.jpg)  
(a) Our unified multi-granularity framework vs. Separated models

Diverse and High-quality 3D Part Generation  
![](images/5bfa7e9a020f79500698aa2c0e29aada859f8b321a5345519d8035c3e075c2b4.jpg)  
(b) Diverse part-aware 3D generation results  
Figure 1: (a) Our unified framework integrates variable-granularity geometric prompts from sparse points to dense masks into one model for flexible part control. (b) The proposed method produces high-fidelity, semantically coherent 3D parts across diverse object categories.

## Abstract

Controllable 3D part generation is pivotal for digital asset creation, yet existing methods lack the flexibility to accommodate diverse user workflows requiring varying levels of control. In this paper, we introduce FlexPart, a unified imageto-3D part generation framework that supports variable-granularity geometric conditions within a single model. By seamlessly integrating heterogeneous inputs, ranging from sparse points to dense bounding boxes and masks, via gated adaptive modulation, FlexPart unlocks a powerful cross-modal synergy. We also propose an asymmetric geometric guidance mechanism that leverages strong spatial priors to align representations across granularities, improving geometric consistency and facilitating inter-task synergy. Under mask conditioning on PartVerseXL, FlexPart reduces Part CD by 16.6% relative to TripoSG + SAM3D and by 10.5% relative to OmniPart, while maintaining competitive whole-mesh quality. The code can be found at https://github.com/Gucci233/FlexPart

## 1 Introduction

Perceiving objects as compositions of semantically coherent parts is essential for various tasks such as animation, simulation, and interactive editing. Unlike holistic 3D generation methods that generate whole entities [16, 39, 40, 48], image-to-3D part generation decomposes visual inputs into distinct, spatially assemblable, structurally detachable, and image-consistent 3D components.

However, this task presents significant challenges due to the inherent ambiguity of part decomposition. For example, the same object can be segmented at various granularities, and users may require different structural specifications depending on their goals. A straightforward solution is to project 2D segmentation masks [45] onto generated 3D meshes. However, such decoupled pipelines can introduce geometric inconsistencies during the lifting process. To address this limitation, recent approaches have integrated part control directly. Some methods rely on dense geometric prompts, such as pixel-wise 2D segmentation masks [3, 9, 46] or explicit bounding box layouts [8]. These inputs provide strong spatial constraints, enabling well-structured part generation. In contrast, other methods explore lighter forms of supervision, such as high-level prompts like part counts [20] or implicit volumetric partitioning [35]. These approaches reduce annotation effort and improve usability but often sacrifice precise spatial control.

While effective within their respective domains, existing methods are typically tailored to a single prompt granularity, as illustrated in Fig. 1(a). This specialization not only limits user flexibility by hindering seamless transitions between fine-grained control via dense masks and lightweight interac tion via sparse points, but also overlooks the inherent structural relationships across different prompt modalities. Such rigidity is suboptimal, as prompts of varying granularities provide complementary information that could be jointly leveraged. This raises a fundamental question: can we develop a unifiedframework that accommodates variable-granularity inputsforflexible part generation, while exploiting their cross-modal synergy to enhance quality?

Driven by this motivation, we propose FlexPart, a unified framework for controllable 3D part generation. As shown in Fig. 1(a), FlexPart integrates sparse points, bounding boxes, and masks into a single Diffusion Transformer (DiT) via gated adaptive modulation. Joint training improves generation quality across prompt granularities. To further amplify this synergy, we introduce an Asymmetric Geometric Guidance mechanism. During training, it employs differentiable optimal transport to let strong spatial priors (masks/boxes) explicitly supervise the ambiguous representations of sparse points. The resulting model supports different prompt types at inference without retraining.

By consolidating diverse geometric constraints, FlexPart establishes a unified paradigm for versatile part creation with variable granularity. Under mask conditioning on PartVerseXL, FlexPart reduces Part CD by 16.6% relative to TripoSG + SAM3D and by 10.5% relative to OmniPart. Thanks to the learned cross-modal synergy, it remains highly competitive even with significantly sparser prompts. Ultimately, FlexPart delivers superior part-level generation quality (Fig. 1(b)) while preserving strong whole-mesh reconstruction fidelity.

Our contributions are summarized as follows: 1) We propose FlexPart, the first unified framework integrating point, box, and mask prompts into a single DiT to unlock cross-modal synergy for controllable 3D part generation; 2) We devise an Asymmetric Geometric Guidance mechanism that effectively transfers structural knowledge from dense priors to enhance the spatial reasoning of sparse inputs; 3) We demonstrate that our method achieves superior generation quality and geometric consistency compared to existing state-of-the-art approaches across diverse input modalities.

## 2 Related Work

General 3D Mesh Generation. Early 3D generation methods relied on optimization-based pipelines, using Score Distillation Sampling (SDS) to distill 2D priors into implicit 3D representations [19, 28, 37, 38]. While pioneering, these approaches often suffer from slow convergence and geometric artifacts. Feed-forward Large Reconstruction Models (LRMs) [11, 34] improve efficiency by directly predicting 3D representations from images. Native 3D generative models use compact latent representations [5, 47, 48] to support efficient diffusion training. Recent methods [16, 40, 42, 49] further improve geometric fidelity through structured sparse latents or flow matching. These object-level generators typically produce holistic meshes without explicit part decomposition, limiting part-level interaction and editing. Complementary work connects 3D analysis and generation. ROAD transfers discriminative semantic and structural priors to a generative model through reciprocal-objective alignment [24], while HERMES and HERMES++ jointly model scene understanding and future geometry generation in autonomous driving [51, 52]. Our work focuses on joint learning across geometric prompt granularities for controllable object-part generation.

3D Shape Analysis and Part Segmentation. Point-cloud analysis has advanced through efficient state-space backbones such as PointMamba [18] and model adaptation methods, including dynamic adapters with prompt tuning [53], spectral-domain fine-tuning [17], and input-aware parameter adaptation for 3D scenes [23]. Here, learned prompts used for model adaptation differ from the user-provided geometric prompts considered in our work. Language-driven approaches such as ShapeLLM [30] and Kestrel [2] further support object understanding and part-aware grounding. For part decomposition, supervised methods [26, 29] analyze existing shapes, while other approaches lift 2D foundation model priors, such as SAM [14] and GLIP [15], to 3D [1, 13, 21, 36, 45, 50]. Recent methods learn open-world 3D feature fields or native segmentation networks [22, 25, 44]. Conventional segmentation primarily identifies regions of existing surfaces. HoloPart [43] extends this setting by completing initially segmented regions into full 3D parts using a generative model. These pipelines rely on input geometry, whereas our task is to generate complete parts directly from an image and geometric prompts.

Part-level 3D Generation. Recent generative models synthesize objects as compositions of semantic components, with different ways of specifying part structure. Mask- and layout-conditioned methods [8, 9, 46] provide explicit spatial control, although precise masks or layouts can require substantial annotation effort. AutoPartGen [4] generates parts autoregressively and supports conditioning on images, part masks, or existing 3D objects. In a related programmatic reconstruction setting, MeshCoder [6] uses an LLM to translate point clouds into editable, part-structured mesh code. Methods such as PartCrafter [20] and PartPacker [35] internalize part decomposition within the generative process without requiring explicit geometric prompts. This simplifies interaction but offers less direct control over spatial part boundaries. A related challenge arises in count-controlled video generation, where NUMINA [33] uses countable latent layouts to improve numerical alignment, illustrating the value of spatial structure for abstract count-based control. Our framework unifies point, box, and mask prompts within one model, allowing users to choose the level of spatial control while benefiting from joint multi-granularity training.

## 3 Preliminaries

Optimal Transport. Optimal Transport (OT) provides a principled geometric framework for modeling soft correspondences between discrete sets. Given two sets of features with cardinalities N and M, OT aims to estimate a transport plan $\mathbf { P } \in \mathbb { R } _ { + } ^ { N \times M }$ , where each entry $P _ { i j }$ represents the mass transported from element i to element j.

Standard OT constrains P to lie within the transport polytope $\mathcal { U } ( { \bf r } , { \bf q } )$ , defined by marginal constraints:

$$
\mathcal { U } ( \mathbf { r } , \mathbf { q } ) = \left\{ \mathbf { P } \in \mathbb { R } _ { + } ^ { N \times M } \mid \mathbf { P } \mathbf { 1 } _ { M } = \mathbf { r } , \mathbf { P } ^ { \top } \mathbf { 1 } _ { N } = \mathbf { q } \right\} ,\tag{1}
$$

where 1 denotes a vector of ones, and r and q are the source and target mass vectors, respectively.

In high-dimensional matching problems, the entropically regularized formulation is widely adopted to enable efficient and differentiable optimization:

$$
\mathbf { P } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { P } \in \mathcal { U } ( \mathbf { r } , \mathbf { q } ) } \langle \mathbf { P } , \mathbf { C } \rangle - \varepsilon H ( \mathbf { P } ) ,\tag{2}
$$

where C and $\varepsilon$ denote the cost matrix and regularization parameter, respectively. The entropy regularizer is defined as:

$$
H ( \mathbf { P } ) = - \sum _ { i , j } P _ { i j } \log P _ { i j } .\tag{3}
$$

The optimal solution admits the factorized form $\mathbf { P } = \mathrm { d i a g } ( \mathbf { u } ) \mathbf { K } \mathrm { d i a g } ( \mathbf { v } )$ , with $\mathbf { K } = \exp ( - \mathbf { C } / \varepsilon )$ and can be efficiently computed using the Sinkhorn–Knopp algorithm.

Partial Optimal Transport. Standard OT enforces strict mass conservation, requiring all elements to be matched, which is ill-suited for scenarios with occlusions or noise. Partial OT addresses this by introducing a dustbin mechanism to absorb unmatched mass. Conceptually, the transport matrix is augmented to $\bar { \mathbf { P } } \in \mathbb { R } ^ { ( N + 1 ) \times ( M + 1 ) }$ allowing elements with no confident correspondence to be assigned to the dustbin row or column. This relaxation handles partial and ambiguous assignments.

![](images/70f68afcbcf87862dd9c216ec98040f5e48ca8f74570c32e1b486aeb243b3d62.jpg)  
Figure 2: Overview of the FlexPart framework. FlexPart supports point, box, and mask prompts, with points as the minimal input. The noisy latent, part ID, and timestep are jointly conditioned on encoded prompts, where learnable null embeddings replace missing inputs. The fused condition modulates each DiT layer for part generation via gated adaptive layer normalization. During training, a multi-granularity geometric guidance branch aligns weak and strong priors via optimal transport.

## 4 Methodology

Our method presents a unified multi-granularity framework for controllable image-to-3D part generation, accommodating diverse geometric user inputs, such as point prompts, 2D bounding boxes, and 2D masks, within a single model. Built upon the DiT architecture, FlexPart leverages a Gated Adaptive Layer Normalization (Gated AdaLN) mechanism to inject these geometric conditions, enabling unified modeling of heterogeneous inputs with varying precision. To further support this versatility, we incorporate a geometric guidance mechanism to maintain robustness under weak supervision. The overall architecture is illustrated in Fig. 2.

## 4.1 Gated Adaptive Modulation

Prompt Definition and Encoding. We consider multiple geometric prompt types for part-level structure, including point, box, and mask prompts. For each part i, we denote their embeddings as $\mathbf { e } _ { i } ^ { \mathrm { { p o i n t } } } , \mathbf { e } _ { i } ^ { \mathrm { { b o x } } } , \mathbf { e } _ { i } ^ { \mathrm { { m a s k } } } \in \mathbb { R } ^ { D }$ , all produced by lightweight modality-specific encoders. Point prompts are encoded via Gaussian Fourier features followed by an MLP to capture continuous 2D locations. Box prompts encode top-left and bottom-right corners separately with a learnable corner-type embedding inspired by SAM [14]. Mask prompts are processed using a small CNN with global average pooling to obtain compact representations. These encoders provide expressive, lightweight representations.

To train the model with different prompt combinations and improve robustness to missing inputs, we employ a probabilistic masking strategy. Although point prompts serve as the mandatory minimal input for users during inference, as shown in Fig. 2, we apply random masking to all modalities, including points, during training. Specifically, we simulate missing geometric prompts by replacing them with learnable null embeddings. For each part i and modality $k \in$ {point, box, mask}, we use a binary drop mask $m _ { i } ^ { k } \in \{ 0 , 1 \}$ with probability $\mathrm { P r } ( m _ { i } ^ { k } = 0 ) = p _ { \mathrm { d r o p } } $ , where $p _ { \mathrm { d r o p } }$ is the manually set probability of masking a given modality. The effective prompt embedding is then defined as:

$$
\tilde { \mathbf { e } } _ { i } ^ { k } = m _ { i } ^ { k } \mathbf { e } _ { i } ^ { k } + ( 1 - m _ { i } ^ { k } ) \mathbf { e } _ { \mathrm { n u l l } } ^ { k } ,\tag{4}
$$

so that any dropped or absent prompt is replaced by its learnable null embedding ${ \mathbf e } _ { \mathrm { n u l l } } ^ { k } \in \mathbb { R } ^ { D }$

The resulting embeddings for each part are aggregated and projected through a learnable fusion MLP ψ(·) to yield a unified part-level conditioning vector:

$$
\mathbf { c } _ { i } = \psi \biggl ( \bigl [ \tilde { \mathbf { e } } _ { i } ^ { \mathrm { p o i n t } } \bigr | \bigr | \tilde { \mathbf { e } } _ { i } ^ { \mathrm { b o x } } \bigr | \bigr | \tilde { \mathbf { e } } _ { i } ^ { \mathrm { m a s k } } \bigr ] \biggr ) ,\tag{5}
$$

where $\left[ \cdot \parallel \cdot \right]$ denotes feature concatenation. This mechanism produces a conditioning vector $\mathbf { c } _ { i }$ of fixed dimensionality for any input prompt combination.

Multi-Prompt Conditional Modulation. Given the fused part condition vector c, we inject geometric control into the DiT via Gated AdaLN. This module dynamically modulates the transformer hidden states, concatenated with the timestep t and learnable part identity, by predicting scale and shift parameters based on c.

A lightweight modulation network $\Phi ( \cdot )$ maps c to adaptive scale, shift, and gating parameters $( \gamma , \beta , \mathbf { g } ) = \Phi ( \mathbf { c } )$ , which are applied to the hidden features H as

$$
\mathbf { H } ^ { \prime } = \mathbf { H } + \mathbf { g } \odot f ( ( 1 + \gamma ) \odot \mathrm { L N } ( \mathbf { H } ) + \boldsymbol { \beta } ) ,\tag{6}
$$

where $\mathrm { L N } ( \cdot )$ denotes layer normalization and $f ( \cdot )$ is a small feed-forward projection. This gated formulation enables adaptive regulation of geometric conditions at each layer. Complementing the geometric prompts, image features from DINOv2 [27] are injected via cross-attention. Cross-attention provides global context, while Gated AdaLN controls part-level geometry.

Part-Aware Attention. Following the design of PartCrafter [20], we adopt a part-aware attention mechanism in the DiT to model both global inter-part interactions and local intra-part geometry. In our setup, we treat individual parts as distinct samples. A batch contains B parts, each represented by L latent tokens, collectively forming a hidden state tensor H $\in \mathbb { R } ^ { B \times L \times D }$

## 4.2 Asymmetric Geometric Guidance

To enrich sparse point prompts with spatial semantics, we propose asymmetric alignment: strong priors (boxes/masks) supervise weak point representations via differentiable optimal transport.

Asymmetric Visual Bridge. We project student point prompts and teacher box/mask prompts into a shared visual space using a Visual Bridge $B ( \cdot )$ with cross-attention over image features $\mathbf { I } _ { \mathrm { e n c } }$ . To prevent the alignment loss from updating the teacher prompt embeddings, we apply stop-gradient to the box/mask embeddings before the Visual Bridge:

$$
\mathbf { f } _ { i } ^ { \mathrm { s t u } } = \mathcal { B } ( \mathbf { e } _ { i } ^ { \mathrm { p o i n t } } , \mathbf { I } _ { \mathrm { e n c } } ) , \quad \mathbf { f } _ { i } ^ { \mathrm { t e a } } = \mathcal { B } ( \mathrm { s g } ( \mathbf { e } _ { i } ^ { \mathrm { b o x / m a s k } } ) , \mathbf { I } _ { \mathrm { e n c } } ) .
$$

Similarity-based Partial Optimal Transport. We align student and teacher features via differentiable partial optimal transport [32], which introduces global competition to resolve ambiguous matches. The affinity matrix is defined as:

$$
\mathbf { S } _ { i j } = \tau \Big ( \alpha \langle \mathbf { f } _ { i } ^ { \mathrm { s t u } } , \mathbf { f } _ { j } ^ { \mathrm { t e a _ { b o x } } } \rangle + ( 1 - \alpha ) \langle \mathbf { f } _ { i } ^ { \mathrm { s t u } } , \mathbf { f } _ { j } ^ { \mathrm { t e a _ { m a s k } } } \rangle \Big ) ,\tag{7}
$$

where $\tau$ is a learnable temperature and α balances the two teacher sources.

To account for missing conditions (e.g., classifier-free guidance [10]), we introduce a learnable dustbin score $\phi$ and construct an augmented matrix $\bar { \mathbf { S } } \in \breve { \mathbb { R } } ^ { ( N + 1 ) \times ( \bar { M } + \bar { 1 } ) }$ by padding S with $\phi .$ The optimal transport plan is computed via log-space Sinkhorn:

$$
\mathbf { P } ^ { * } = \arg \operatorname* { m a x } _ { \mathbf { P } \in \mathcal { U } } \sum _ { i , j } \bar { \mathbf { S } } _ { i j } \mathbf { P } _ { i j } + \varepsilon H ( \mathbf { P } ) ,\tag{8}
$$

where U denotes valid transport plans and $H ( \mathbf { P } )$ is the entropy regularizer.

This mechanism transfers strong geometric priors from boxes and masks to point representations, improving spatial semantics without requiring dense prompts at inference.

Training Objective. We train FlexPart with a unified objective that combines conditional Rectified Flow and feature alignment supervision:

$$
\mathcal { L } = \mathbb { E } _ { \mathcal { Z } _ { 0 } , \epsilon , t } \Big [ \big \| ( \epsilon - \mathcal { Z } _ { 0 } ) - \mathbf { v } _ { \theta } ( \mathcal { Z } _ { t } , t , \mathbf { c } ) \big \| _ { 2 } ^ { 2 } \Big ] - \lambda _ { \mathrm { a u x } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log ( \mathbf { P } _ { i , i } ^ { * } + \delta ) ,\tag{9}
$$

where the first term corresponds to the conditional Rectified Flow objective, encouraging accurate velocity prediction in the latent space, and the second term enforces index-wise alignment between student and teacher features via diagonal transport probabilities. Here, N denotes the number of student point representations in the alignment term, δ is a small constant for numerical stability, and $\lambda _ { \mathrm { a u x } }$ controls the strength of the alignment supervision.

Inference. During inference, we support flexible geometric conditions ranging from sparse points to dense masks. Point prompts are always required as the base condition, while higher-level cues are incorporated hierarchically for flexible control. Specifically, masks can be paired with bounding boxes, and boxes can be combined with point prompts. This design enables precise control of part generation under varying levels of geometric guidance.

## 5 Implementation Details and Metrics

## 5.1 Implementation Details

We use a pretrained 3D VAE from TripoSG [16] to obtain part latents. Each part is encoded into latent tokens following 3DShape2VecSet [47]. Following PartCrafter [20], we initialize the generative backbone with pretrained TripoSG [16] weights and fine-tune it using the following two-stage training strategy. We train FlexPart on a composite dataset derived from Objaverse [7] and PartVerseXL [8]. Stage 1 targets objects with 1–8 parts, each encoded with 512 tokens. To ensure robustness against suboptimal placement, we randomly sample points within masks and apply a 0.5 dropout rate to both boxes and masks. Stage 2 scales to 2–16 parts using 1024 tokens per part to handle varying granularities. In this stage, we set the dropout probability to 0.1 for points and 0.7 for masks, similar to [12, 31]. For bounding boxes, we implement a mixed dropping strategy with a probability of 0.25 for full dropout, and 0.25 for partial masking applied to half of the parts. Training is conducted on 8 NVIDIA H20 GPUs with a batch size of 256, using learning rates of $1 \times 1 0 ^ { - 4 }$ and $5 \times 1 0 ^ { - 5 }$ for the two stages, respectively.

## 5.2 Datasets and Metrics

To assess generalization, we conduct evaluations on a held-out test set from PartVerseXL [8] and PartObjaverse-Tiny [44], ensuring strict disjointness from the training set. For each dataset, we select the view that maximizes the product of visible pixel count and number of visible parts, ensuring sufficient part coverage.

Whole Mesh Evaluation Metrics. We first assemble the generated parts into a complete mesh and evaluate overall geometric quality using Chamfer Distance (CD) and F-Score, where F-Score is computed with a distance threshold of 0.1. Semantic consistency between the generated mesh and the input image is measured using ULIP [41].

Part-level Evaluation Metrics. A primary challenge in evaluating parts is that the number of generated components often differs from the ground truth. To ensure a fair comparison, we first filter out ground truth parts that are fully occluded in the input image. We apply CD-based Hungarian matching between the remaining ground-truth parts and generated parts to obtain one-to-one correspondences. We compute part-level metrics only over matched pairs; unmatched ground-truth and generated parts are excluded without additional penalties. Based on these matched pairs, we report Part CD to measure geometric deviation. Furthermore, we evaluate spatial alignment using part-level bounding-box IoU $\left( \operatorname { I o U } _ { B } \right)$ for coarse extent, and surface-based voxel IoU $( \operatorname { I o U } _ { V } )$ for fine-grained overlap. To handle non-watertight geometry, we define occupancy based on the distance field rather than interior filling. Specifically, we use a voxel grid with a pitch of $h _ { \mathrm { v o x } } = 0 . 0 3$ , where a voxel is considered occupied if its distance to the nearest mesh surface is less than or equal to $h _ { \mathrm { v o x } }$ . Unless otherwise stated, benchmark box inputs are derived from ground-truth masks, and points are sampled from the corresponding mask regions.

Detailed dataset construction and benchmark preparation are provided in App. A.

Table 1: Part-level structure evaluation. Cond. denotes input conditions; PartPacker requires none. For whole-mesh baselines, we use SAM3D to lift multi-view 2D masks to 3D segmentations. Part IoU<sub>V</sub> and Part IoU<sub>B</sub> denote voxel IoU and 3D bounding box IoU. Bold and underlined indicate the best and second-best results, respectively.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Reference</td><td rowspan="2">Cond.</td><td colspan="3">PartVerseXL</td><td colspan="3">PartObjaverse-Tiny</td></tr><tr><td>Part CD ↓</td><td>Part IoUv ↑ Part IoUB ↑</td><td></td><td>Part CD ↓</td><td>Part IoUv ↑ Part IoUB ↑</td><td></td></tr><tr><td>TRELLIS + SAM3D</td><td>CVPR 25</td><td>MV Mask</td><td>0.3471</td><td>0.1885</td><td>0.2173</td><td>0.3916</td><td>0.1226</td><td>0.2031</td></tr><tr><td>TripoSG + SAM3D</td><td>TPAMI 25</td><td>MV Mask</td><td>0.3523</td><td>0.1727</td><td>0.1896</td><td>0.4235</td><td>0.1250</td><td>0.2078</td></tr><tr><td>PartCrafter</td><td>NeurIPS 25</td><td>Number</td><td>0.5235</td><td>0.0907</td><td>0.1257</td><td>0.5193</td><td>0.0802</td><td>0.1455</td></tr><tr><td>PartPacker</td><td>NeurIPS 25</td><td></td><td>0.3933</td><td>0.1717</td><td>0.2210</td><td>0.3932</td><td>0.1588</td><td>0.2327</td></tr><tr><td>OmniPart</td><td>SIGGRAPH Asia 25</td><td>Mask</td><td>0.3283</td><td>0.1961</td><td>0.2831</td><td>0.3681</td><td>0.1568</td><td>0.2649</td></tr><tr><td>FlexPart (ours)</td><td></td><td>Mask</td><td>0.2938</td><td>0.2049</td><td>0.2840</td><td>0.3560</td><td>0.1571</td><td>0.2778</td></tr><tr><td>FlexPart (ours)</td><td></td><td>Box</td><td>0.3042</td><td>0.2078</td><td>0.2810</td><td>0.3675</td><td>0.1503</td><td>0.2693</td></tr><tr><td>FlexPart (ours)</td><td></td><td>Point</td><td>0.3334</td><td>0.1902</td><td>0.2506</td><td>0.3802</td><td>0.1411</td><td>0.2456</td></tr></table>

Table 2: Whole-mesh quality evaluation. TRELLIS and TripoSG are full-mesh generation methods, while FullPart overlaps with the PartVerseXL training set; these entries are grayed out and reported for reference only, and are excluded from direct comparison. Bold and underlined indicate the best and second-best results.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Reference</td><td colspan="3">PartVerseXL</td><td colspan="3">PartObjaverse-Tiny</td></tr><tr><td>Mean CD ↓</td><td>F-Score ↑</td><td>ULIP↑</td><td>Mean CD ↓</td><td>F-Score ↑</td><td>ULIP↑</td></tr><tr><td>TRELLIS</td><td>CVPR 25</td><td>0.2125</td><td>0.7514</td><td>0.2370</td><td>0.2038</td><td>0.7478</td><td>0.2434</td></tr><tr><td>TripoSG</td><td>TPAMI 25</td><td>0.1227</td><td>0.8176</td><td>0.2331</td><td>0.1504</td><td>0.7811</td><td>0.2410</td></tr><tr><td>PartCrafter</td><td>NeurIPS 25</td><td>0.1666</td><td>0.7346</td><td>0.1950</td><td>0.1731</td><td>0.7280</td><td>0.2136</td></tr><tr><td>PartPacker</td><td>NeurIPS 25</td><td>0.1489</td><td>0.7899</td><td>0.2266</td><td>0.1612</td><td>0.7682</td><td>0.2397</td></tr><tr><td>OmniPart</td><td>SIGGRAPH Asia 25</td><td>0.1547</td><td>0.7636</td><td>0.2179</td><td>0.1585</td><td>0.7501</td><td>0.2335</td></tr><tr><td>FullPart</td><td>ICLR 26</td><td></td><td></td><td></td><td>0.1625</td><td>0.7577</td><td>0.1775</td></tr><tr><td>FlexPart (ours)</td><td></td><td>0.1418</td><td>0.7877</td><td>0.2206</td><td>0.1587</td><td>0.7510</td><td>0.2350</td></tr></table>

## 6 Experiments

## 6.1 Main Results

We compare FlexPart against six state-of-the-art methods, including two holistic mesh generation models [16, 40] and four native part-based generation approaches [8, 20, 35, 46].

Part-level Mesh Quality. We first evaluate the geometric fidelity of individual parts (Tab. 1) to assess controllable part generation. For comparison with whole-mesh generation methods, we employ SAM3D [45] to project multi-view masks onto their generated meshes for part segmentation. This post-hoc lifting process often introduces geometric inconsistencies, significantly degrading part-level accuracy. In contrast, FlexPart directly infers compact part geometry. Under the same mask condition, our method outperforms OmniPart by a large margin, reducing Part CD on PartVerseXL by 10.5%. This improvement is consistent with the benefits of direct part generation: OmniPart uses intermediate axis-aligned 3D boxes, which can enclose substantial empty space around non-axis-aligned shapes.

Crucially, FlexPart maintains robust performance across varying input granularities. Relaxing the condition from masks to boxes yields stable quality, with Part CD degrading by merely 3.5% on PartVerseXL. Even under these weaker priors, FlexPart achieves better Part CD and Part IoU than structurally unconstrained methods like PartPacker. FullPart is excluded here due to its reliance on 3D bounding boxes. A fairer evaluation under controlled settings is provided in Sec. 6.2.

Whole Mesh Quality. Complementing our superior part-level performance, we further assess the global quality of the assembled meshes. As shown in Tab. 2, FlexPart achieves strong and consistent mesh-level quality across both benchmarks. On PartVerseXL, we achieve a Mean CD of 0.1418, surpassing OmniPart by 8%, indicating that our improvements in local part precision translate into better global geometric accuracy. On PartObjaverse-Tiny, while some specialized methods achieve marginally better results, FlexPart remains highly competitive. These results show that FlexPart supports flexible part control while maintaining competitive whole-mesh quality.

Input Image&Cond.

FlexPart (ours)

FullPart

PartPacker

OmniPart

PartCrafter

![](images/015046687d3dae600b0fba15b5b51e14fe009e48f96a04821ca5aeb01711a071.jpg)

Figure 3: Qualitative results of different methods. The upper row shows exploded views for inspecting individual parts, and the lower row shows the assembled meshes. Our method uses masks as input, while other methods are evaluated with their respective input conditions.

Qualitative Results. Fig. 3 presents qualitative comparisons across diverse categories. FullPart provides rough layout control via boxes but often yields inaccurate shapes. PartPacker achieves decent global geometry but lacks explicit control to match user-specified structures. OmniPart tends to distort meshes, deviating noticeably from inputs, while PartCrafter, limited to part-count conditioning, struggles to align with intended divisions. In contrast, FlexPart generates distinct and semantically meaningful parts, ensuring visual realism and structural coherence suitable for downstream editing.

## 6.2 Analysis and Ablation Study

To investigate the architecture design, we conduct ablation studies on PartVerseXL unless otherwise specified. Default settings are marked in gray

Effect of Multi-granularity Synergy. Tab. 3 demonstrates that our framework achieves deep crossmodal synergy instead of a naive combination of conditions. Specifically, stronger geometric priors effectively guide weaker ones during training. Our joint model improves Point only inference by 7.9% in IoU and 6.1% in CD because it internalizes precise boundary constraints from dense inputs. Furthermore, joint training consistently outperforms single-modality models across all evaluation settings. For example, under mask conditioning, the unified model reduces Part CD from 0.3224 for the mask-only training baseline to 0.2938. These results confirm that FlexPart learns integrated semantics in a shared space. This formulation provides a versatile solution that maintains high geometric fidelity across diverse interaction modes. Fig. 4 illustrates robust, semantically consistent results across prompt granularities.

Mask Prompts

Table 3: Ablation on multi-granularity prompt synergy. Eval Cond. denotes the evaluation condition, while Train Setting lists the prompt modalities used to train each model. Point only, Box only, and Mask only denote single-modality training baselines. We compare whole-mesh and part-level fidelity within each evaluation condition.
<table><tr><td rowspan="2">Eval Cond.</td><td rowspan="2">Train Setting</td><td colspan="3">Whole Mesh</td><td colspan="3">Part-level</td></tr><tr><td>CD↓</td><td>F-Score ↑</td><td>ULIP↑</td><td>CD↓</td><td>IoUv ↑</td><td> $\mathrm { I o U } _ { B } \uparrow$ </td></tr><tr><td rowspan="3">Point</td><td>Point only</td><td>0.1623</td><td>0.7368</td><td>0.2014</td><td>0.3550</td><td>0.1763</td><td>0.2320</td></tr><tr><td>Point + Mask</td><td>0.1587</td><td>0.7505</td><td>0.2112</td><td>0.3421</td><td>0.1816</td><td>0.2396</td></tr><tr><td>Point + Box + Mask</td><td>0.1525</td><td>0.7689</td><td>0.2171</td><td>0.3334</td><td>0.1902</td><td>0.2506</td></tr><tr><td rowspan="4">Box</td><td>Box only</td><td>0.1489</td><td>0.7756</td><td>0.2171</td><td>0.3155</td><td>0.2011</td><td>0.2756</td></tr><tr><td>Point + Box</td><td>0.1458</td><td>0.7812</td><td>0.2201</td><td>0.3102</td><td>0.2023</td><td>0.2763</td></tr><tr><td> $\mathbf { B o x } + \mathbf { M a s k }$ </td><td>0.1434</td><td>0.7834</td><td>0.2209</td><td>0.3094</td><td>0.2089</td><td>0.2795</td></tr><tr><td>Point + Box + Mask</td><td>0.1447</td><td>0.7828</td><td>0.2219</td><td>0.3042</td><td>0.2078</td><td>0.2810</td></tr><tr><td rowspan="4">Mask</td><td>Mask only</td><td>0.1541</td><td>0.7564</td><td>0.2056</td><td>0.3224</td><td>0.1905</td><td>0.2688</td></tr><tr><td> $\mathrm { P o i n t + M a s k }$ </td><td>0.1501</td><td>0.7712</td><td>0.2169</td><td>0.3202</td><td>0.1921</td><td>0.2712</td></tr><tr><td> $\mathbf { B o x } + \mathbf { M a s k }$ </td><td>0.1450</td><td>0.7848</td><td>0.2195</td><td>0.3076</td><td>0.1998</td><td>0.2822</td></tr><tr><td>Point + Box + Mask</td><td>0.1418</td><td>0.7877</td><td>0.2206</td><td>0.2938</td><td>0.2049</td><td>0.2840</td></tr></table>

Variable Granularity  
Point Prompts  
Box Prompts

![](images/273d96d67c2effb1a998e20acac5df46f56b62a95136fb96590042d9f2069a13.jpg)  
Figure 4: Qualitative generation results of our method under varying geometric prompts.

## Effect of Asymmetric Geometric Guidance.

Building upon the observed multi-granularity synergy, we evaluate the proposed Asymmetric Geometric Guidance by ablating the auxiliary alignment loss. As shown in Tab. 4, it significantly improves performance under sparse point prompts, yielding gains of 7.9% in Part $\mathrm { I o U } _ { B }$ and 5.0% in Part $\mathrm { I o U } _ { V }$ over the baseline. This indicates that OT-based alignment effectively transfers structural priors from dense annotations to regularize learning from sparse inputs.

Table 4: Ablation on Asymmetric Geometric Guidance under point and mask prompts.
<table><tr><td rowspan="2">Cond.</td><td rowspan="2">Guid.</td><td colspan="3">Part-level</td></tr><tr><td>CD↓</td><td>IoUv ↑</td><td> $\mathrm { I o U } _ { B } \uparrow$ </td></tr><tr><td rowspan="2">Points</td><td></td><td>0.3406</td><td>0.1811</td><td>0.2323</td></tr><tr><td>√</td><td>0.3334</td><td>0.1902</td><td>0.2506</td></tr><tr><td rowspan="2">Masks</td><td></td><td>0.3011</td><td>0.2078</td><td>0.2813</td></tr><tr><td>√</td><td>0.2938</td><td>0.2049</td><td>0.2840</td></tr></table>

Performance under mask conditions remains comparable, suggesting positive cross-granularity transfer without conflict. The guidance operates only during training, with no inference overhead.

Analysis of Stronger 3D Conditions. Although our primary focus is on generating 3D parts from lightweight 2D inputs, we also evaluate FlexPart under explicit 3D constraints. For a fair comparison with FullPart, which uses 3D bounding boxes, we retrain a variant condi-

Table 5: Comparison with FullPart under 3D box conditioning on the PartObjaverse-Tiny benchmark.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Cond.</td><td colspan="3">Whole Mesh</td><td colspan="3">Part-level</td></tr><tr><td>CD↓</td><td>F-Score↑</td><td>ULIP↑</td><td>CD↓</td><td>IoUv ↑</td><td>IoUB↑</td></tr><tr><td>FullPart</td><td>3D Box</td><td>0.1625</td><td>0.7577</td><td>0.1775</td><td>0.2488</td><td>0.2064</td><td>0.4370</td></tr><tr><td>Ours</td><td>3D Box</td><td>0.1385</td><td>0.8158</td><td>0.2209</td><td>0.2356</td><td>0.2653</td><td>0.4027</td></tr></table>

tioned solely on 3D boxes and evaluate it on PartObjaverse-Tiny. As shown in Tab. 5, our method outperforms FullPart on most metrics under the same supervision. This result demonstrates that Gated AdaLN handles both sparse 2D inputs and explicit 3D constraints.

Analysis of SAM-based Alternatives. A seemingly straightforward alternative is to convert sparse prompts into dense masks using 2D segmentation models (e.g., SAM) before 3D generation. However, such cascaded pipelines are inherently limited: lacking 3D part-level semantics, SAM often produces localized fragments rather than complete functional parts, leading to accumulated errors and degraded geometry. In contrast, FlexPart directly models heterogeneous conditions end-to-end, avoiding intermediate information loss and enabling consistent part-level generation from sparse prompts. Detailed analyses are provided in App. H.

Robustness to Input Variations. Perturbing the position and size of 2D boxes by up to 20% increases Part CD from 0.3042 to 0.3171, while Part IoU changes from 0.2078 to 0.2032. Providing boxes for half of the parts and points for the remainder yields Part CD of 0.3287. Mixing masks for half of the parts with boxes for the rest achieves Part CD of 0.3190, despite this combination being unseen during training. These results support generation with noisy and heterogeneous prompts. A complementary study evaluates joint point-and-box inputs from three annotators on 50 PartObjaverse-Tiny objects. Full protocols and results, including count-only inputs, are provided in App. D and E.

Sensitivity of Point Prompts. We evaluate four point-placement strategies on PartVerseXL: Center (the part-mask centroid), Edge (near the part boundary), Offset (a perturbed centroid), and Cluster (the patch center closest to the object-level geometric center). Under point-only input, switching from Center to Edge increases Part CD from 0.3302 to 0.4190 and reduces Part IoU from 0.1905 to 0.1342. With additional box or mask prompts, Part CD varies across the four strategies within 0.3036– 0.3178 and 0.2995–0.3087, respectively. Stronger geometric prompts therefore reduce sensitivity to point placement. Experimental details and quantitative and qualitative results appear in App. G.

## 7 Conclusion

We introduced FlexPart, a unified framework for controllable image-to-3D part generation with geometric prompts at varying granularities. Gated adaptive modulation integrates heterogeneous prompts within a single model, while Asymmetric Geometric Guidance uses strong spatial priors to improve sparse point representations during training. Experiments demonstrate improved part-level geometric quality and competitive whole-mesh fidelity across prompt conditions. These results support joint multi-granularity learning as a practical approach to flexible 3D part generation.

Limitations. Occlusion handling remains challenging. Single-view 2D conditions provide limited depth cues, and the model may place occluded parts incorrectly in visible regions. Improving spatial reasoning under occlusion is a direction for future work.

## Acknowledgments

We thank Chongqing Afari Technology Co., Ltd. for providing computational resources.

This work was supported by The Explorers Program of Shanghai (Basic Research Funding) (26TS1404200); the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM312); and Tongji University’s “Medicine+X” Cultivation Program (2026-0648-YB-05).

## References

[1] Ahmed Abdelreheem, Ivan Skorokhodov, Maks Ovsjanikov, and Peter Wonka. Satr: Zero-shot semantic segmentation of 3d shapes. In Proc. ofIEEE Intl. Conf. on Computer Vision, pages 15166–15179, 2023.

[2] Mahmoud Ahmed, Junjie Fei, Jian Ding, Eslam Mohamed Bakr, and Mohamed Elhoseiny. Kestrel: 3d multimodal llm for part-aware grounded description. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 8973–8983, 2025.

[3] Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, and Andrea Vedaldi. Partgen: Part-level 3d generation and reconstruction with multi-view diffusion models. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 5881–5892, 2025.

[4] Minghao Chen, Jianyuan Wang, Roman Shapovalov, Tom Monnier, Hyunyoung Jung, Dilin Wang, Rakesh Ranjan, Iro Laina, and Andrea Vedaldi. Autopartgen: Autogressive 3d part generation and discovery. In Proc. of Advances in Neural Information Processing Systems, 2025.

[5] Rui Chen, Jianfeng Zhang, Yixun Liang, Guan Luo, Weiyu Li, Jiarui Liu, Xiu Li, Xiaoxiao Long, Jiashi Feng, and Ping Tan. Dora: Sampling and benchmarking for 3d shape variational auto-encoders. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 16251–16261, 2025.

[6] Bingquan Dai, Li Ray Luo, Qihong Tang, Jie Wang, Xinyu Lian, Hao Xu, Minghan Qin, Xudong Xu, Bo Dai, Haoqian Wang, et al. Meshcoder: Llm-powered structured mesh code generation from point clouds. In Proc. ofAdvances in Neural Information Processing Systems, 2025.

[7] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 13142–13153, 2023.

[8] Lihe Ding, Shaocong Dong, Yaokun Li, Chenjian Gao, Xiao Chen, Rui Han, Yihao Kuang, Hong Zhang, Bo Huang, Zhanpeng Huang, et al. Fullpart: Generating each 3d part at full resolution. In Proc. ofIntl. Conf. on Learning Representations, 2026.

[9] Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jaehyeok Kim, Chenjian Gao, Zhanpeng Huang, et al. From one to more: Contextual part latents for 3d generation. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 8230–8240, 2025.

[10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[11] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. In Proc. of Intl. Conf. on Learning Representations, 2024.

[12] Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, and Jingren Zhou. Composer: Creative and controllable image synthesis with composable conditions. In Proc. of Intl. Conf. on Machine Learning, 2023.

[13] Hyunjin Kim and Minhyuk Sung. Partstad: 2d-to-3d part segmentation task adaptation. In Proc. of European Conference on Computer Vision, pages 422–439, 2024.

[14] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proc. ofIEEE Intl. Conf. on Computer Vision, pages 4015–4026, 2023.

[15] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 10965–10975, 2022.

[16] Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–18, 2025.

[17] Dingkang Liang, Tianrui Feng, Xin Zhou, Yumeng Zhang, Zhikang Zou, and Xiang Bai. Parameterefficient fine-tuning in spectral domain for point cloud learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[18] Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, and Xiang Bai. PointMamba: A simple state space model for point cloud analysis. In Advances in Neural Information Processing Systems, volume 37, 2024.

[19] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 300–309, 2023.

[20] Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Yiqiang Feng, Yadong Mu, and Katerina Fragkiadaki. Partcrafter: Structured 3d mesh generation via compositional latent diffusion transformers. In Proc. ofAdvances in Neural Information Processing Systems, 2025.

[21] Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. Part123: part-aware 3d reconstruction from a single-view image. In ACM SIGGRAPH Conference on Computer Graphics and Interactive Techniques, pages 1–12, 2024.

[22] Minghua Liu, Mikaela Angelina Uy, Donglai Xiang, Hao Su, Sanja Fidler, Nicholas Sharp, and Jun Gao. Partfield: Learning 3d feature fields for part segmentation and beyond. In Proc. of IEEE Intl. Conf. on Computer Vision, pages 9704–9715, 2025.

[23] Siyuan Liu, Chaoqun Zheng, Xin Zhou, Tianrui Feng, Dingkang Liang, and Xiang Bai. PointTPA: Dynamic network parameter adaptation for 3D scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[24] Xiao Luo, Mingyang Du, Xin Zhou, Tianrui Feng, Xiwu Chen, Xiaofan Li, Jiangning Zhang, and Dingkang Liang. ROAD: Reciprocal-objective alignment of discriminative semantics for 3D shape generation. arXiv preprint arXiv:2607.28581, 2026.

[25] Changfeng Ma, Yang Li, Xinhao Yan, Jiachen Xu, Yunhan Yang, Chunshi Wang, Zibo Zhao, Yanwen Guo, Zhuo Chen, and Chunchao Guo. P3-sam: Native 3d part segmentation. arXiv preprint arXiv:2509.06784, 2025.

[26] Kaichun Mo, Shilin Zhu, Angel X Chang, Li Yi, Subarna Tripathi, Leonidas J Guibas, and Hao Su. Partnet: A large-scale benchmark for fine-grained and hierarchical part-level 3d object understanding. In Proc. of IEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 909–918, 2019.

[27] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russ Howes, Po-Yao (Bernie) Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[28] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In Proc. ofIntl. Conf. on Learning Representations, 2023.

[29] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 652–660, 2017.

[30] Zekun Qi, Runpei Dong, Shaochen Zhang, Haoran Geng, Chunrui Han, Zheng Ge, Li Yi, and Kaisheng Ma. Shapellm: Universal 3d object understanding for embodied interaction. In Proc. of European Conference on Computer Vision, pages 214–238, 2024.

[31] Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, and Lu Jiang. Vincie: Unlocking in-context image editing from video. In Proc. ofIntl. Conf. on Learning Representations, 2026.

[32] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 4938–4947, 2020.

[33] Zhengyang Sun, Yu Chen, Xin Zhou, Xiaofan Li, Xiwu Chen, Dingkang Liang, and Xiang Bai. When numbers speak: Aligning textual numerals and visual instances in text-to-video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.

[34] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In Proc. ofEuropean Conference on Computer Vision, pages 1–18, 2024.

[35] Jiaxiang Tang, Ruijie Lu, Zhaoshuo Li, Zekun Hao, Xuan Li, Fangyin Wei, Shuran Song, Gang Zeng, Ming-Yu Liu, and Tsung-Yi Lin. Efficient part-level 3d object generation via dual volume packing. In Proc. of Advances in Neural Information Processing Systems, 2025.

[36] Anh Thai, Weiyao Wang, Hao Tang, Stefan Stojanov, James M Rehg, and Matt Feiszli. 3× 2: 3d object part segmentation by 2d semantic correspondences. In Proc. of European Conference on Computer Vision, pages 149–166, 2024.

[37] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In Proc. of European Conference on Computer Vision, pages 439–457, 2024.

[38] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In Proc. ofAdvances in Neural Information Processing Systems, volume 36, pages 8406–8441, 2023.

[39] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. In Proc. ofAdvances in Neural Information Processing Systems, volume 37, pages 121859–121881, 2024.

[40] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 21469–21480, 2025.

[41] Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, and Silvio Savarese. Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding. In Proc. ofIEEE Intl. Conf. on Computer Vision and Pattern Recognition, pages 1179–1189, 2023.

[42] Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao, Xinhai Liu, Xinzhou Wang, Qingxiang Lin, Jiaao Yu, et al. Hunyuan3d 1.0: A unified framework for text-to-3d and image-to-3d generation. arXiv preprint arXiv:2411.02293, 2024.

[43] Yunhan Yang, Yuan-Chen Guo, Yukun Huang, Zi-Xin Zou, Zhipeng Yu, Yangguang Li, Yan-Pei Cao, and Xihui Liu. Holopart: Generative 3d part amodal segmentation. In Proc. of Intl. Conf. on Learning Representations, 2025.

[44] Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y Lam, Yan-Pei Cao, and Xihui Liu. Sampart3d: Segment any part in 3d objects. arXiv preprint arXiv:2411.07184, 2024.

[45] Yunhan Yang, Xiaoyang Wu, Tong He, Hengshuang Zhao, and Xihui Liu. Sam3d: Segment anything in 3d scenes. In Proc. of IEEE Intl. Conf. on Computer Vision Workshop, 2023.

[46] Yunhan Yang, Yufan Zhou, Yuan-Chen Guo, Zi-Xin Zou, Yukun Huang, Ying-Tian Liu, Hao Xu, Ding Liang, Yan-Pei Cao, and Xihui Liu. Omnipart: Part-aware 3d generation with semantic decoupling and structural cohesion. In ACM SIGGRAPH Conference on Computer Graphics and Interactive Techniques in Asia, pages 1–12, 2025.

[47] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions ON Graphics, 42(4):1–16, 2023.

[48] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions ON Graphics, 43(4):1–20, 2024.

[49] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.

[50] Ziming Zhong, Yanyu Xu, Jing Li, Jiale Xu, Zhengxin Li, Chaohui Yu, and Shenghua Gao. Meshsegmenter: Zero-shot mesh semantic segmentation via texture synthesis. In Proc. of European Conference on Computer Vision, pages 182–199, 2024.

[51] Xin Zhou, Dingkang Liang, Xiwu Chen, Feiyang Tan, Dingyuan Zhang, Hengshuang Zhao, and Xiang Bai. HERMES++: Toward a unified driving world model for 3D scene understanding and generation. arXiv preprint arXiv:2604.28196, 2026.

[52] Xin Zhou, Dingkang Liang, Sifan Tu, Xiwu Chen, Yikang Ding, Dingyuan Zhang, Feiyang Tan, Hengshuang Zhao, and Xiang Bai. HERMES: A unified self-driving world model for simultaneous 3D scene understanding and generation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2025.

[53] Xin Zhou, Dingkang Liang, Wei Xu, Xingkui Zhu, Yihan Xu, Zhikang Zou, and Xiang Bai. Dynamic adapter meets prompt tuning: Parameter-efficient transfer learning for point cloud analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

## A Dataset Preparation and Benchmark Details

Data Preparation. We construct the training set from subsets of Objaverse [7] and PartVerseXL [8], totaling approximately 250K 3D models. Due to batch-wise part-packing during training, about 180K models are effectively used. The Objaverse subset provides heterogeneous annotations, where some models contain only complete object meshes while others include part-level meshes. PartVerseXL complements this with densely and carefully annotated part-level meshes. Since most Objaverse meshes are non-watertight, we convert all models into watertight representations, following prior work [5, 16, 42, 49], showing that watertight geometry leads to more stable training and improved generation quality. For each mesh, multiple viewpoints are rendered, and the view with the largest visible foreground area is selected as the training image.

For the selected view, we generate 2D part visibility masks via ID-based rendering. Each part is rendered with a unique color, and depth buffering naturally resolves inter-part occlusions. The rendered image is decoded into a part ID map, from which binary visibility masks are extracted. Bounding boxes and point prompts are then derived directly from the corresponding masks, enabling unified geometric supervision at multiple granularities. To simulate user interactions, we sample 10 points within each mask and select the one nearest its centroid.

Benchmarks. For evaluation, we construct benchmark subsets to test generalization. From Part-VerseXL [8], which provides high-quality part annotations, we create a held-out test set with no overlap with our training data by randomly selecting 100 shapes with no more than 16 annotated parts, matching the maximum part count used during training. We also evaluate on PartObjaverse-Tiny [44], a publicly available part-annotated dataset, including all shapes with up to 16 parts. Although some annotations in PartObjaverse-Tiny are less precise, it provides a complementary setting for assessing part-level generalization. For each dataset, we select the view that maximizes the product of visible pixel count and the number of visible parts, ensuring an informative view with sufficient part coverage. Before computing geometric metrics, we align each generated mesh to its ground-truth counterpart using rigid Iterative Closest Point.

## A.1 Evaluation Metric Definitions

Before ICP, each shape is normalized to fit within $[ - 1 , 1 ] ^ { 3 }$ so that its longest bounding-box side is 2, with no additional scaling during rigid alignment.

Whole Mesh Evaluation Metrics. Whole CD is the symmetric unsquared L2 Chamfer distance over 10K surface samples. For sampled point sets X and $\dot { Y }$

$$
\mathrm { C D } ( X , Y ) = \frac { 1 } { | X | } \sum _ { x \in X } \operatorname* { m i n } _ { y \in Y } \| x - y \| _ { 2 } \quad + \frac { 1 } { | Y | } \sum _ { y \in Y } \operatorname* { m i n } _ { x \in X } \| y - x \| _ { 2 } .\tag{10}
$$

F-Score is the harmonic mean of precision and recall at distance threshold 0.1 (5% of the normalized longest side). ULIP [41] measures semantic consistency with the input image. The same CD definition is used for matched part pairs.

Part-level Evaluation Metrics. A primary challenge in evaluating parts is that the number of generated components often differs from the ground truth. To ensure a fair comparison, we first filter out ground truth parts that are fully occluded in the input image. We apply CD-based Hungarian matching between the remaining ground-truth parts and generated parts to obtain one-to-one correspondences. We compute part-level metrics only over matched pairs; unmatched ground-truth and generated parts are excluded without additional penalties. Based on these matched pairs, we report Part CD to measure geometric deviation. Furthermore, we evaluate spatial alignment using part-level bounding-box IoU $( \mathsf { I o U } _ { B } )$ for coarse extent, and surface-based voxel IoU $( \bar { \mathrm { I o U } } _ { V } )$ for fine-grained overlap. To handle non-watertight geometry, we define occupancy based on the distance field rather than interior filling. Specifically, we use a voxel grid with a pitch of $h _ { \mathrm { v o x } } = 0 . 0 3$ , where a voxel is considered occupied if its distance to the nearest mesh surface is less than or equal to $h _ { \mathrm { v o x } } .$ . The normalized longest axis spans approximately 67 voxels at this pitch, which is fixed across all methods. Unless otherwise stated, benchmark boxes are derived from masks, and points are sampled from the corresponding mask regions.

Table 6: Conditioning routes under mask input on PartVerseXL. PartCrafter is a reference using part count and no spatial prompts, not a matched-condition ablation.
<table><tr><td>Conditioning route</td><td>Part CD ↓</td><td>IoUv ↑</td><td>IoUB↑</td></tr><tr><td>Gated AdaLN (ours)</td><td>0.2938</td><td>0.2049</td><td>0.2840</td></tr><tr><td>Prompt tokens in cross-attention</td><td>0.6147</td><td>0.0818</td><td>0.1089</td></tr><tr><td>PartCrafter (reference)</td><td>0.5235</td><td>0.0907</td><td>0.1257</td></tr></table>

## A.2 Baseline Protocol and Data Lineage

All baselines use official checkpoints, without retraining or fine-tuning. TRELLIS and TripoSG receive an image, with multi-view SAM3D lifting for part evaluation; PartCrafter receives an image and GT part count; PartPacker receives only an image; OmniPart receives an image and GT-derived 2D masks; FullPart receives an image and GT 3D boxes. We re-evaluate every method using the same view selection, shape normalization, ICP alignment, and metrics. The main table is a system-level comparison with different inputs and training sets. Controlled comparisons use mask inputs for OmniPart, 3D boxes for FullPart, and part count for PartCrafter.

TRELLIS uses TRELLIS-500K; TripoSG uses curated Objaverse(-XL) and ShapeNet data. PartCrafter uses part-annotated Objaverse, ShapeNet, and ABO data; PartPacker uses filtered/extracted parts from TRELLIS-500K/Objaverse-XL; OmniPart uses self-constructed partlabeled data with a TRELLIS-based backbone. FullPart trains on PartVerseXL and is omitted on that benchmark. Our PartVerseXL evaluation set is strictly held out, whereas PartObjaverse-Tiny shares Objaverse asset lineage with several pretrained methods, including our TripoSG initialization.

## A.3 Alignment Implementation and Overhead

We use 50 log-domain Sinkhorn iterations with a learnable dustbin score and temperature. The box/mask fusion weight is $\alpha = 0 . 5$ , and the auxiliary alignment loss weight is $\lambda _ { \mathrm { a u x } } = 1 . 0$ . The alignment branch has 1.18M parameters (0.077% of the model), and transport matrices are at most 33 × 33. An fp16 profile on one NVIDIA H20 GPU, with 8 parts and 1024 latent tokens per part, measures 8.9 ms for this branch versus 525 ms for the DiT forward pass (1.70%). This measure forward-pass overhead in the profiled configuration, not total training throughput.

## B Conditioning Route Analysis

We compare Gated AdaLN with injecting geometric prompt tokens into the pretrained image crossattention pathway. The token-injection variant uses the full training schedule. Under mask conditioning, it performs worse than Gated AdaLN and the count-conditioned PartCrafter reference (Table 6). A possible explanation is competition between prompt tokens and the 257 DINOv2 tokens within one attention softmax; the separate modulation route avoids this competition. This diagnostic supports our choice for this backbone, without establishing a general disadvantage of cross-attention conditioning.

## C Analysis of Asymmetric Geometric Guidance

## C.1 Effect of Stop-Gradient Design

We further analyze the effect of the stop-gradient design in geometric guidance. We find that it primarily affects whole-mesh reconstruction quality across different prompt granularities. As shown in Tab. 7, enabling gradient flow (w/ Grad) slightly improves performance under sparse point prompts, reducing CD from 0.1525 to 0.1518. However, it degrades performance under stronger geometric conditions, including both box and mask prompts, where CD increases from 0.1447 to 0.1531 and from 0.1418 to 0.1512, respectively.

This behavior suggests that allowing gradients leads to overly symmetric optimization, where strong and weak prompts are pulled toward each other in representation space. Our goal is to guide weak point representations using strong geometric priors while preventing the alignment loss from updating the teacher prompt embeddings. We stop gradients at box/mask embeddings before the Visual Bridge.

Table 7: Ablation on the Stop-Gradient Design in Geometric Guidance. "w/ Grad" allows the alignment loss to update the teacher prompt embeddings, while "wo/ Grad" (our default) stops gradients at the box/mask embeddings before the Visual Bridge.
<table><tr><td>Condition</td><td>Setting</td><td>CD↓</td><td>F-Score ↑</td><td>ULIP↑</td></tr><tr><td rowspan="2">Point</td><td>w/ Grad</td><td>0.1518</td><td>0.7702</td><td>0.2166</td></tr><tr><td>wo/ Grad</td><td>0.1525</td><td>0.7689</td><td>0.2171</td></tr><tr><td rowspan="2">Box</td><td>w/ Grad</td><td>0.1531</td><td>0.7762</td><td>0.2198</td></tr><tr><td>wo/ Grad</td><td>0.1447</td><td>0.7828</td><td>0.2219</td></tr><tr><td rowspan="2">Mask</td><td>w/ Grad</td><td>0.1512</td><td>0.7761</td><td>0.2212</td></tr><tr><td>wo/ Grad</td><td>0.1418</td><td>0.7877</td><td>0.2206</td></tr></table>

Table 8: End-to-end comparison of alignment losses on PartVerseXL. Only the alignment loss is changed; initialization and training budget are held fixed. IoU<sub>V</sub> and $\mathrm { I o U } _ { B }$ denote part-level voxel and bounding-box IoU, respectively.
<table><tr><td>Condition</td><td>Alignment loss</td><td>Part CD ↓</td><td>IoUv ↑</td><td> $\mathrm { I o U } _ { B } \uparrow$ </td></tr><tr><td>Mask</td><td>Partial OT (ours)</td><td>0.2938</td><td>0.2049</td><td>0.2840</td></tr><tr><td></td><td>InfoNCE</td><td>0.3053</td><td>0.1924</td><td>0.2673</td></tr><tr><td>Box</td><td>Partial OT (ours)</td><td>0.3042</td><td>0.2078</td><td>0.2810</td></tr><tr><td></td><td>InfoNCE</td><td>0.3074</td><td>0.1874</td><td>0.2653</td></tr><tr><td>Point</td><td>Partial OT (ours)</td><td>0.3334</td><td>0.1902</td><td>0.2506</td></tr><tr><td></td><td>InfoNCE</td><td>0.3499</td><td>0.1807</td><td>0.2346</td></tr></table>

Table 9: Assignment-operator diagnostic on the same frozen features. These are feature-matching results, rather than generation metrics or separate L2/InfoNCE training experiments.
<table><tr><td>Assignment operator</td><td>Top-1 matching ↑</td><td>Many-to-one collapse ↓</td></tr><tr><td>Row-wise similarity matching</td><td>0.803</td><td>0.133</td></tr><tr><td>Sinkhorn global assignment</td><td>0.895</td><td>0.024</td></tr></table>

## C.2 Partial OT versus Contrastive Alignment

We evaluate whether partial optimal transport (OT) improves generation beyond a simpler contrastive alignment objective. We train an InfoNCE variant from the same initialization and with the same training budget, changing only the alignment loss. Table 8 reports the resulting part-level generation metrics on PartVerseXL. Partial OT achieves lower Part CD and higher voxel and bounding-box IoU under all three prompt conditions. In particular, under point prompts, Part CD decreases from 0.3499 to 0.3334. This experiment evaluates the alignment objective in end-to-end generation.

## C.3 Global Assignment on Frozen Features

Known part correspondences provide diagonal supervision during training, but point features can still be ambiguous among similar strong priors. Paired L2 alignment treats each known pair independently, while InfoNCE normalizes similarities independently within each point row. Neither imposes a column marginal on the induced matching. Sinkhorn instead couples the assignments through both row and column marginals, introducing competition among points for the available priors.

To isolate this assignment effect, we freeze the trained model and use 199 PartObjaverse-Tiny objects containing 1,330 parts. Point, box, and mask prompts are derived from each ground-truth part as in training. We run only the prompt encoders, visual bridge, and DINOv2 feature extractor, without diffusion sampling or mesh decoding. We then apply row-wise similarity matching and Sinkhorn to the same similarity matrix. Top-1 matching measures correct point-to-prior matches, while many-to-one collapse measures duplicated prior selections by multiple points.

On normalized features, an InfoNCE similarity argmax and an L2 nearest-neighbor readout both reduce to row-wise similarity matching, although their training objectives differ. As shown in Table 9, Sinkhorn improves Top-1 matching from 0.803 to 0.895 and reduces many-to-one collapse by approximately 5.5×. Ambiguity is particularly pronounced for symmetric parts: row-wise matching accuracy is 0.516 for symmetric pairs, compared with 0.931 for other pairs. These observations support global competition for resolving ambiguous feature correspondences.

Table 10: Frozen-feature diagnostic with four cross-object box-prior distractors. Both variants receive the same augmented set of priors; the distractors have no valid correspondence in the current object.
<table><tr><td>Transport formulation</td><td>Top-1 matching ↑</td></tr><tr><td>Strict OT</td><td>0.677</td></tr><tr><td>Partial OT with dustbin</td><td>0.747</td></tr></table>

Table 11: Analysis of the adaptability to user input variations.
<table><tr><td>Method</td><td colspan="3">Part-level</td></tr><tr><td></td><td>CD↓</td><td>IoUv ↑</td><td> $\mathrm { I o U } _ { B } \uparrow$ </td></tr><tr><td>2D Box w/ Perturbation</td><td>0.3171</td><td>0.2032</td><td>0.2759</td></tr><tr><td>Half Box + Point</td><td>0.3287</td><td>0.1998</td><td>0.2609</td></tr><tr><td>Half Mask + Half Box</td><td>0.3190</td><td>0.2053</td><td>0.2706</td></tr><tr><td>Part Count Only</td><td>0.4786</td><td>0.1255</td><td>0.1608</td></tr><tr><td>Ours (2D Box)</td><td>0.3042</td><td>0.2078</td><td>0.2810</td></tr></table>

## C.4 Dustbin for Unmatched Priors

The dustbin addresses a separate issue: unmatched entries in batch-level alignment. In the same frozen-feature diagnostic, we append four box-prior columns from another object, none of which corresponds to a part of the current object. We compare strict OT with partial OT using the dustbin under this shared distractor setting. As shown in Table 10, partial OT improves Top-1 matching from 0.677 to 0.747. The dustbin supplies a channel for unmatched mass, relaxing the requirement to assign mass to irrelevant priors. Together with the preceding diagnostic, this separates the role of global competition among valid matches from the role of unmatched-mass handling.

## D Adaptability to Input Variations

We evaluate FlexPart under noisy, partial, and heterogeneous inputs through controlled experiments that approximate common user input variations. Drawn bounding boxes may be misaligned, loosely cover target parts, or specify only a subset of components.

Perturbed bounding boxes. We simulate noisy user annotations by randomly offsetting the position and size of each input 2D bounding box by up to 20%. This perturbation tests the model’s tolerance to spatial inaccuracy while preserving approximate part localization.

Partial box specification. To reflect scenarios where users focus on only a subset of parts, we provide bounding boxes for half of the parts and use point prompts for the remaining parts. This hybrid setting evaluates whether strong and weak geometric prompts can be jointly exploited within the unified framework.

Mixed-condition specification. To further assess the effectiveness of our unified framework, we consider a mixed condition setting that is never observed during training. Specifically, we provide 2D masks for half of the parts and 2D bounding boxes for the remaining parts. This setting evaluates whether FlexPart can seamlessly integrate heterogeneous strong priors and generalize to novel combinations of geometric conditions.

Minimal input with part count only. We also consider the extreme case where users specify only the desired number of parts. Since FlexPart requires at least point-level spatial prompts, we automatically derive point prompts from the input image. Specifically, we extract high-resolution patch features $\mathbf { f } _ { \mathrm { p a t c h } }$ using DINOv2. To improve spatial consistency, we incorporate the 2D coordinates $\mathbf { x } _ { \mathrm { p a t c h } }$ of each patch to form a joint representation $\mathbf { v } _ { \mathrm { p a t c h } } = [ \mathbf { f } _ { \mathrm { p a t c h } } \ \lVert \ \omega _ { \mathrm { s p } } \mathbf { x } _ { \mathrm { p a t c h } } ]$ , where the spatial weight is set to $\omega _ { \mathrm { s p } } = 5 . 0$ . We then perform spectral clustering on this combined representation using $k = 1 5$ nearest neighbors to obtain point prompts from the cluster centroids. In our preliminary experiments, we compared several clustering methods, including K-means and hierarchical clustering; however, spectral clustering consistently yielded the best performance in capturing semantically and spatially coherent part structures. This strategy enables effective part generation even from minimal input, and outperforms PartCrafter when only part counts are provided.

Table 12: Human-provided prompts on 50 PartObjaverse-Tiny objects. All inputs contain both points and 2D boxes. Values are mean ± standard deviation across three annotators. IoU<sub>V</sub> and IoU<sub>B</sub> denote part-level voxel and bounding-box IoU, respectively.
<table><tr><td>Prompt source</td><td>Part CD ↓</td><td> $\mathrm { I o U } _ { V } \uparrow$ </td><td> $\mathrm { I o U } _ { B } \uparrow$ </td></tr><tr><td>Human points + 2D boxes</td><td> $0 . 3 8 4 \pm 0 . 0 2 2$ </td><td> $0 . 1 3 1 \pm 0 . 0 1 0$ </td><td> $0 . 2 3 1 \pm 0 . 0 1 3$ </td></tr></table>

![](images/168d1080b1308d5bf42a8c388c6fc301932cba84c11c9768be0db34bfb0bcbba.jpg)  
(a) Comparison of inference time

![](images/15ed23a94e2b0120d3c4ce72bed06f87a37fb7f1c15ec0f5e500a30b23e207ab.jpg)  
(b) Comparison of GPU memory  
Figure 5: Efficiency comparison between FlexPart and other methods.

The quantitative results for these controlled input variations are summarized in Tab. 11. The results demonstrate a clear trade-off between the strength of geometric conditions and generation accuracy: stronger inputs provide more precise part localization, while weaker inputs offer greater flexibility with reduced supervision. Crucially, thanks to its unified multi-granularity design, FlexPart consistently maintains reasonable performance across all input settings and generalizes well to noisy, partial, and mixed conditions. This highlights the advantage of a unified framework that seamlessly integrates heterogeneous user inputs for robust, practical interactive 3D part generation.

Complementary evaluation with human-provided point and box prompts is reported in App. E.

## E Evaluation with Human-Provided Prompts

The controlled perturbations in App. D approximate common input errors. To complement them with actual human-provided prompts, we invite three annotators to mark their intended parts on the input images of 50 PartObjaverse-Tiny objects using points and 2D bounding boxes. Every evaluated input contains both prompt types; this study evaluates joint point-and-box input. Table 12 reports the mean and standard deviation of the evaluation scores across the three annotators.

We compute these metrics against the dataset parts after Hungarian matching. Since an annotator’s intended decomposition may differ from the dataset annotation, the scores serve as reference measurements of agreement with the dataset, rather than strict measures of whether the user’s intended decomposition is satisfied. The observed cross-annotator variation suggests limited sensitivity to the annotator within this study. This small-scale evaluation provides evidence for generation with human-provided spatial prompts, but does not measure interaction time or downstream usability. Complementary qualitative results with manually provided point or box prompts are shown in Fig. 9.

## F Part Generation Efficiency

Beyond accuracy, we evaluate efficiency in terms of inference latency and memory usage. Specifically, we measure the average time required to generate a single part, covering the entire process from condition encoding to final mesh extraction. Additionally, we analyze peak GPU memory (VRAM) consumption during object generation with varying numbers of parts. All evaluations are conducted on a single NVIDIA A800 GPU using random instances from the PartVerseXL validation set.

![](images/5c5bc8296ddbca074421b65c467c22fd794d344a8397633a0fadce2cae3235b2.jpg)  
Figure 6: Qualitative analysis of point placement under four sampling strategies: Center, Offset, Edge, and Cluster.

Regarding inference speed, Fig. 5(a) illustrates that FlexPart achieves competitive performance relative to existing baselines. Although our method shares a similar Diffusion Transformer backbone with PartCrafter [20], we concatenate part identity embeddings with the latent code rather than adding them directly. This design choice introduces a slight computational overhead during attention operations, resulting in a marginal increase in inference latency. Compared to two-stage approaches like OmniPart [46], which accelerates generation via latent downsampling, our method operates at a higher resolution and thus requires more computation. FlexPart strikes a balanced trade-off between generation quality and speed, leaving potential for further acceleration in future work.

In terms of memory consumption, as shown in Fig. 5(b), FlexPart maintains a modest peak VRAM footprint of ∼6GB to 8GB, which is comparable to the lightweight baseline PartCrafter [20]. In contrast, dense generation methods such as OmniPart [46] and FullPart [8] demand significantly higher resources, consuming nearly twice the memory under identical settings. Furthermore, our approach exhibits robust scalability with respect to object complexity. When doubling the target part count from 8 to 16, the peak memory usage increases only marginally. This footprint suggests that consumer GPUs with sufficient VRAM can support complex multi-part generation.

## G Sensitivity of Point Prompts

We investigate how point placement affects generation quality across input modalities, focusing on spatial ambiguity and the role of point prompts.

Point Sampling Strategies. We consider four representative strategies to simulate diverse user behaviors and annotation noise: 1). Center, the centroid of the part mask; 2). Edge, a point sampled near the part boundary to induce maximal spatial ambiguity; 3). Offset, a randomly perturbed centroid to mimic typical user imprecision; 4). Cluster, the patch center nearest the object’s geometric center.

Quantitative Analysis. As shown in Tab. 13, the sensitivity to point placement strongly depends on the availability of additional geometric priors. When using Point Only inputs, the model exhibits clear sensitivity to suboptimal point locations. In particular, sampling points near object boundaries (Edge) leads to a substantial degradation across all metrics, with CD increasing from 0.3302 to 0.4190 and IoU<sub>V</sub> dropping from 0.1905 to 0.1342, indicating greater spatial ambiguity without supporting cues.

In contrast, when stronger geometric conditions such as bounding boxes or masks are provided, the influence of point placement becomes significantly less pronounced. For example, under the Point + Box setting, the variation across different strategies remains within a narrow range (e.g., CD varies only from 0.3036 to 0.3178), indicating stable performance. A similar trend is observed for Point + Mask, where IoU<sub>V</sub> fluctuates marginally between 0.1962 and 0.2088, suggesting that geometric priors effectively dominate the spatial reasoning process.

Table 13: Robustness analysis of point prompt placement across different input modalities. Metrics are evaluated on the PartVerseXL test set.
<table><tr><td>Input Type</td><td>Point Strategy</td><td>CD↓</td><td> $\mathbf { I o U } _ { V } \uparrow$ </td><td> $\mathbf { I o U } _ { B } \uparrow$ </td></tr><tr><td rowspan="4">Point Only</td><td>Center</td><td>0.3302</td><td>0.1905</td><td>0.2513</td></tr><tr><td>Edge</td><td>0.4190</td><td>0.1342</td><td>0.1697</td></tr><tr><td>Offset</td><td>0.3476</td><td>0.1898</td><td>0.2455</td></tr><tr><td>Cluster</td><td>0.3803</td><td>0.1632</td><td>0.1905</td></tr><tr><td rowspan="4">Point + Box</td><td>Center</td><td>0.3118</td><td>0.2074</td><td>0.2728</td></tr><tr><td>Edge</td><td>0.3178</td><td>0.1996</td><td>0.2703</td></tr><tr><td>Offset</td><td>0.3064</td><td>0.2029</td><td>0.2805</td></tr><tr><td>Cluster</td><td>0.3036</td><td>0.2062</td><td>0.2793</td></tr><tr><td rowspan="4">Point + Mask</td><td>Center</td><td>0.2995</td><td>0.2030</td><td>0.2857</td></tr><tr><td>Edge</td><td>0.3087</td><td>0.1962</td><td>0.2766</td></tr><tr><td>Offset</td><td>0.3011</td><td>0.2088</td><td>0.2809</td></tr><tr><td>Cluster</td><td>0.3058</td><td>0.2042</td><td>0.2796</td></tr></table>

Table 14: Comparison of direct point conditioning and mask-conditioned generation using SAMderived or ground-truth masks. Direct point conditioning outperforms both SAM cascades on the reported part-level metrics.
<table><tr><td>Setting</td><td>Condition</td><td>CD↓</td><td> $\mathbf { I o U } _ { V } \uparrow$ </td><td> $\mathbf { I o U } _ { B } \uparrow$ </td></tr><tr><td>SAM (Random)</td><td>Mask</td><td>0.3498</td><td>0.1796</td><td>0.2375</td></tr><tr><td>SAM (Center)</td><td>Mask</td><td>0.3387</td><td>0.1731</td><td>0.2244</td></tr><tr><td>Direct point (ours)</td><td>Point</td><td>0.3334</td><td>0.1902</td><td>0.2506</td></tr><tr><td>Ground Truth (Random)</td><td>Mask</td><td>0.2938</td><td>0.2049</td><td>0.2840</td></tr></table>

Qualitative Analysis. Consistent with the quantitative findings, our qualitative results shown in Fig. 6 visualize the generation outcomes under the four point sampling strategies in the Point Only setting. The visual comparison clearly demonstrates how ambiguous point inputs, such as those near the edges, lead to a loss of geometric fidelity.

Overall, while FlexPart is sensitive to point locations under point-only conditions, its sensitivity to point placement is substantially reduced when stronger geometric prompts are also provided.

## H Analysis of SAM-based Cascaded Pipelines

A seemingly plausible approach is to first convert sparse inputs (e.g., points) into dense masks using a 2D segmentation model such as Segment Anything (SAM) [14], followed by mask-conditioned 3D generation. We analyze this pipeline to understand whether such a decomposition can replace direct modeling of sparse conditions.

As shown in Tab. 14, generation conditioned on SAM-derived masks performs worse than generation conditioned on ground-truth masks under both evaluated point sampling strategies. Even with centerbased point sampling, CD degrades from 0.2938 to 0.3387, and IoU drops from 0.2049 to 0.1731. Direct point conditioning achieves Part CD 0.3334, IoU<sub>V</sub> 0.1902, and IoU<sub>B</sub> 0.2506, outperforming both SAM-derived mask settings under the same evaluation protocol.

This performance gap may reflect a mismatch between image segmentation and the intended 3D part decomposition. As visualized in Fig. 7, given a sparse prompt, SAM typically segments a localized fragment rather than the complete functional part intended by the user. Incomplete intermediate masks can propagate segmentation errors into 3D generation.

In contrast, our end-to-end framework avoids this by directly modeling heterogeneous conditions. This joint modeling enables cross-modal synergy, allowing the network to learn robust semantics directly from sparse prompts. Furthermore, it naturally scales to structured 3D conditions like 3D bounding boxes, which cannot be elegantly reduced to 2D masks.

Input Image

![](images/e64626ba43e123f5fe18ff8578aa2bf11341a907dac14263e72481811d545bbb.jpg)

SAM Mask

![](images/3c11d7a672e1ec224e25ef023fb7a473406d5518b917c77e8237822c165a6da9.jpg)

SAM Result

![](images/fcec091bec2f4557ade6f5b8eee70dd16b65495607fe2fe19893d34cbc798d68.jpg)

GT Mask

![](images/4b6c58192c3e4101fa5830c43778a4ab98563079eb20090af3fcef6f0721b41a.jpg)

GT Result

![](images/cf4cc6aa9d8a320ff1b46b17026863954fd7178397751a5681f51f0592d674c3.jpg)

![](images/caaee40c28664bbed2a70489ef61456b2784b60879e4d33f9a2ac5a5c103d625.jpg)

![](images/2c53353a57ad5e1520655da000bbdc9d8eb54feedf6a63a63086551a432f8341.jpg)

![](images/42909a506e4b34dcf1986495086ea360bd24dff2ae98f0a84d20526987eb568d.jpg)

![](images/68c5c8f687905f40197b08efeaaed4607176f275cb83deb799094cbf4cbe3678.jpg)

![](images/b0a4c1ffcef54fe2c1e592edf923949168c66f9f4e8dc8996da6ceb8565dec3a.jpg)

![](images/a7b77b65e2c60ec9b1dcf364a344de3c34148a026d2584a1de034a780488d315.jpg)

![](images/cd8c58861170a633f9ea12cea2a83f9c6929e8d3afecbcd8fd6eb97c3442eafd.jpg)

![](images/a2cbf5bdbbe939730a9cd1ff9a6416d989cd53d3359046a5b1aacb68a301a2b1.jpg)

![](images/0862a4b0c5f18d42ea8859596b487e0383fd3755e714495cc6aed7415aa05cee.jpg)

![](images/01fb91dc31e8ed9805cadc93956aa43e7a8549907ac050016cb6c1f1959fa6cb.jpg)  
Figure 7: Qualitative evaluation of the SAM-based interface. We compare generation results under two input conditions across diverse examples. SAM: masks generated by SAM from a single point prompt sampled near the visual center. GT: ground-truth segmentation masks.  
Condition

Input Image

w/ Mask

w/ Box

w/ Point

![](images/3c6e15b81d7138b5ee4aa384d2d670e08abb7687999f2518709023b9772fe748.jpg)  
Figure 8: Additional qualitative results of FlexPart across objects and geometric prompts.

## I Additional Qualitative Results

To further substantiate the versatility of FlexPart, we provide extended qualitative results in Fig. 8 across a spectrum of geometric conditions. In addition, we present a new set of visualizations in Fig. 9 focusing on objects with highly complex topological structures. By directly leveraging image conditioning alongside user-provided point or box prompts, our method successfully synthesizes high-quality, part-level 3D objects that preserve intricate geometric details.

![](images/5a9e2673fb9f0f3d409bc30b40891f8a400bb54f5b56d411aac835d3e1c5a758.jpg)  
Figure 9: Generalization to complex topologies. Our method directly processes user-provided point or box prompts to generate 3D part assets.

These combined results highlight the core advantage of our unified framework, which empowers users to flexibly choose the most suitable prompt type for their specific needs. Regardless of the input modality, from lightweight points to dense masks, FlexPart consistently produces high-fidelity meshes with clean topology and precise alignment to the provided spatial layouts, demonstrating robust generalization across diverse interaction modes.