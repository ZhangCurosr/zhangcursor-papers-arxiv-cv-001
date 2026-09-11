# HIPERVIT: A HIERARCHICAL PERCEIVER–VISION TRANSFORMER ARCHITECTURE FOR MULTI-SCALE TEXTURE RECOGNITION João Pedro C. A. de Sá<sup>1</sup> and Odemir Martinez Bruno<sup>1,2</sup>

<sup>1</sup>Institute of Mathematics and Computer Sciences (ICMC), University of São Paulo (USP), Avenida Trabalhador São-carlense, 400, São Carlos, SP 13566-590, Brazil

<sup>2</sup>São Carlos Institute of Physics (IFSC), University of São Paulo (USP), Avenida Trabalhador São-carlense, 400, São Carlos, SP 13566-590, Brazil

joaosa@usp.br (J. P. C. A. de Sá) bruno@ifsc.usp.br (O. M. Bruno)

Abstract. Texture recognition remains challenging for modern vision models because discriminative evidence is often carried by higher-order spatial statistics rather than by object shape alone. While Vision Transformers provide strong long-range modeling capacity, their standard object-centric representations do not explicitly expose such statistical structure, which limits texture sensitivity in fine-grained recognition settings. We present HiPerViT, a compact vision-only architecture that injects an explicit second-order statistical prior into a transformer-based recognition pipeline. The method combines global and local image views with a compact bilinear descriptor encoded as a statistical token, and integrates this token with first-order spatial representations through Perceiver-style latent distillation. This design enables direct interaction between spatial tokens and second-order feature co-occurrence statistics, providing the model with explicit access to texture-relevant information without requiring multimodal pretraining or ensemble construction. Across six texture recognition benchmarks, HiPerViT achieves consistent improvements over strong vision-only baselines under the reported evaluation protocols, including gains of +3.05 percentage points on DTD, +10.48 on GTOS-Mobile, and +10.10 on 1200Tex. Beyond benchmark performance, our analyses show that these gains are largely invariant to the backbone depth used to extract second-order statistics and to the ordering of interaction and distillation stages. This pattern suggests that the primary source of improvement is not a specific fusion topology, but the explicit availability of second-order statistical information as a first-class representational signal. These results support explicit statistical tokenization as an efective and robust design principle for texture-centric visual recognition.

## 1. Introduction

Texture recognition remains a tough problem for modern vision systems because evidence is often encoded in higher-order spatial statistics rather than object shape alone. In medical imaging, microscopy, remote sensing, robotics, and industrial inspection, successful recognition depends less on canonical object form than on repeated micro-structures, stochastic organization, and feature co-occurrence patterns distributed across space [12, 3]. This makes texture recognition diferent from standard object-centric categorization: robust performance requires sensitivity to local structure, long-range context, and statistical organization across scales.

This distinction exposes a limitation of current visual architectures. Convolutional neural networks (CNNs) provide strong locality and translation priors but don’t preserve higher-order statistical structure as a first-class representation. Vision Transformers (ViTs) ofer a mechanism for modeling long-range interactions through self-attention, yet their standard training regime encourages representations that are efective for global structural recognition without exposing co-occurrence statistics [4, 10]. Large-scale multimodal pre-training can broaden the scope of learned representations, but such strategies are expensive and not always practical in specialized settings with limited labeled data and domain-specific supervision.

A central question remains unresolved: when transformer-based models improve on texture recognition, do gains arise from a particular fusion architecture or from restoring explicit access to statistical information? Much of the recent literature combines multiscale processing, token mixing, pooling, or auxiliary branches, but this often leaves unclear which ingredient is responsible for the improvement. From a representation design standpoint, the issue is whether texture-sensitive recognition benefits primarily from a specific architectural composition or from exposing higher-order statistics as an explicit representational signal.

In this work, we address this question through HiPerViT, a vision-only architecture that augments transformer representations with an explicit second-order statistical prior. HiPer-ViT combines global and local image views, encodes compact bilinear feature co-occurrence statistics as a statistical token [9], and integrates this token with first-order spatial representations through a Perceiver-style latent bottleneck [15]. The resulting design enables direct interaction between spatial tokens and second-order statistical information within a unified fusion pipeline. Importantly, the goal is not to replace first-order visual structure but to make higher-order statistical evidence explicitly available when it is informative for the task.

This representation-centered framing motivates a testable hypothesis. If HiPerViT improves mainly because it uses a particularly favorable fusion design, then its performance should depend strongly on where interaction and latent distillation are placed in the pipeline. If, instead, the main benefit comes from making second-order statistical information explicitly available to the model, then the gain should remain relatively stable across reasonable changes in fusion ordering and in the backbone depth used to extract the statistical descriptor. If observed gains were mainly due to a privileged fusion topology, performance should be highly sensitive to interaction ordering and latent distillation. If they were mainly due to extracting secondorder descriptors at a favorable stage of the backbone, performance should vary strongly with extraction depth. By contrast, if the principal benefit comes from exposing explicit second-order information as a first-class representational signal, then improvements should be robust to both fusion ordering and extraction depth. Our experiments distinguish among these alternatives.

Across six texture recognition benchmarks, HiPerViT achieves competitive or best-reported top-1 accuracy while remaining computationally compact. Under the reported protocols, it reaches 93.05% on DTD, 94.58% on GTOS-Mobile, and 97.50% on 1200Tex, and in matchedprotocol comparisons it matches or exceeds transformer baselines such as ViT-B/16, DeiT, Swin, and VORTEX on all six datasets. Empirical results support the representation-level interpretation above. The gains associated with the statistical token are stable across alternative fusion topologies and backbone depths used to extract second-order statistics. We show that the contribution of the statistical token is strongest in low-data regimes, consistent with the view that explicit second-order tokenization acts as a conditional inductive bias. Cue-conflict analyses indicate that adding the statistical token does not overturn the dominant object-centric perceptual preference of the underlying backbone, suggesting that the proposed mechanism expands texture sensitivity without destabilizing the primary representational prior.

Taken together, the results support a representation-level conclusion. The low-data experiments show that the statistical token is most helpful when supervision is limited, the topology ablations show that the gain persists across diferent interaction–distillation orderings, and the capture-depth analyses show that final accuracy remains stable even when the raw secondorder descriptors difer markedly in standalone separability. Together, these findings suggest that HiPerViT does not depend on a single privileged fusion layout; rather, its main advantage comes from making second-order statistical structure explicitly available for interaction with first-order spatial tokens. HiPerViT provides one eficient realization of this idea, but the broader implication is methodological: explicit statistical tokenization is a robust design principle for texture-oriented visual recognition.

Our main contributions can be summarized as follows: (i) we introduce HiPerViT, a compact vision-only architecture that integrates an explicit second-order statistical token through multiscale token fusion and Perceiver-style latent distillation; (ii) we show that HiPerViT achieves consistently strong performance across six texture recognition benchmarks while remaining more compact than larger vision-only comparison models; (iii) we demonstrate that the contribution of the statistical token is robust to both the backbone depth at which second-order statistics are extracted and the choice of fusion topology, indicating that the improvement arises primarily from the availability of explicit second-order information rather than from a particular architectural configuration; and (iv) we show that the statistical token acts as a conditional inductive bias, providing its strongest contribution under limited supervision while leaving dominant first-order perceptual priors largely intact in object-centric settings.

## 2. Background

The challenge of texture recognition has long fascinated both neuroscientists and computer vision researchers. The human visual system can efortlessly distinguish velvet from denim or granite from sand, even under dramatic changes of scale or illumination. Understanding how to replicate this capability computationally has led to successive paradigm shifts, each echoing, in diferent ways, the hierarchical computations of biological vision.

## 2.1. Statistical Descriptors of Texture

The first wave of computational approaches sought to capture texture through explicit statistics, inspired by the idea that perception relies on the distribution of simple local features. Gabor filters, for example, model orientation- and frequency-selective responses reminiscent of V1 receptive fields [14, 7]. Gray-Level Co-occurrence Matrices (GLCM) [12] quantified how pixel intensities co-occur, encoding regularities that correlate with human judgments of roughness, smoothness, or directionality. Local Binary Patterns (LBP) later distilled these ideas into compact codes of local contrast.

These descriptors demonstrated that even simple statistics could reproduce aspects of texture perception and perform reliably on controlled datasets. Yet, their handcrafted nature limited adaptability: descriptors tuned to specific frequency ranges or co-occurrence patterns often failed to generalize across scales, lighting conditions, or domains.

A breakthrough emerged when researchers extended this statistical view beyond first-order and marginal distributions to explicitly model pairwise interactions. The bilinear pool represented each image as the outer product of the feature maps, capturing second-order correlations between all channels in a translation-invariant way [19]. This paralleled findings in neuroscience that mid-level cortical areas (V2, V4) integrate co-occurrence patterns of oriented filters to represent texture surfaces [8]. However, the full bilinear representation was prohibitively large (e.g., 262k dimensions for 512 channels). Compact Bilinear Pooling (CBP) [9] addressed this by sketching the outer product into a few thousand dimensions with minimal accuracy loss, enabling eficient training and deployment. Refinements such as iSQRT-COV [18] and Shifted Random Maclaurin pooling [38] further optimized this principle, consolidating the role of second-order statistics as a cornerstone of texture recognition.

In summary, from early co-occurrence matrices to modern compact bilinear pooling, the statistical tradition established a durable insight: textures are not merely distributions of local features, but structured arrangements whose identity often lies in the correlations among them. This statistical perspective paved the way for the next generation of approaches, where hierarchical feature learning could be naturally combined with statistical modeling.

## 2.2. CNN Architecture for Texture Analysis

The advent of deep learning radically shifted the field. CNNs [17] ofered hierarchical representations learned directly from data, mirroring the layered architecture of the visual cortex. Convolutional layers acted as learned filter banks, while deeper layers captured increasingly abstract structures. Researchers quickly realized that these features, although trained for object recognition, could be repurposed for material and texture classification.

Seminal works such as FV-CNN [3] treated convolutional responses as dense filter banks and aggregated them orderlessly, achieving invariance to position and scale. Bilinear CNNs [19] went further by encoding all pairwise channel interactions, significantly improving finegrained recognition. Multiscale extensions (e.g., MOP-CNN, MS-CNN) highlighted another principle from biology: robust texture understanding requires integrating both micro- and macro-patterns across scales.

Despite these successes, CNN-based solutions often demanded task-specific modifications, dilated kernels for high-frequency textures, spectral branches for stationary patterns, revealing that their inductive biases were not inherently tuned for texture statistics.

## 2.3. From Sequence Transformers to Vision Transformers

The Transformer architecture originated in natural language processing with Attention Is All You Need [37], which replaced recurrence with self-attention to model long-range dependencies in sequential data. The key innovation was the multi-head self-attention mechanism, allowing each token to attend to every other token through learned query–key–value interactions. This design provided parallelism, scalability, and global context modeling, advantages that soon motivated exploration beyond language.

Early visual adaptations sought to translate these principles to images by tokenizing spatial patches rather than words. Dosovitskiy et al. (2021) [4] introduced the Vision Transformer (ViT), showing that with suficient data and computation, a pure transformer could rival convolutional networks for image classification. ViT dispensed with spatial inductive biases such as locality and translation equivariance, instead relying on large-scale pre-training to learn them implicitly.

Subsequent architectures progressively reintroduced structure and eficiency: DeiT [36] improved data eficiency through distillation; Swin Transformer [21] implemented hierarchical windowed attention to recover multi-scale locality; and Perceiver [15] extended the transformer to high-dimensional or multimodal inputs via a latent bottleneck with asymmetric cross-attention, drastically reducing the cost of global attention.

This lineage, from sequence models to ViT and hierarchical or latent variants, marks a clear evolution: from global attention without structure toward architectures that integrate scale, hierarchy, and eficiency. HiPerViT continues this trajectory by uniting a multi-scale ViT backbone with a Perceiver-style latent bottleneck and compact bilinear pooling, forming a biologically inspired and computationally eficient model for texture understanding.

## 2.4. The Transformer Era in Texture Recognition

With the success of ViTs in large-scale image classification, researchers soon began investigating their suitability for texture and material recognition, domains traditionally dominated by convolutional networks and handcrafted statistical models. The hope was that the global self-attention of transformers could unify local micro-patterns and long-range contextual cues in a single framework.

However, empirical analyses revealed that vanilla ViTs develop strong shape priors, often favoring object contours over fine-grained texture cues [11, 24]. This efect became particularly evident on fine-grained material datasets.

To address this, several transformer variants introduced inductive biases that restore local sensitivity. VORTEX [31] extracted multiscale tokens from frozen ViTs and applied orderless second-order pooling to capture high-frequency texture cues. DBTrans [20] fused global and local branches to balance contour awareness with texture detail. Other approaches combined transformers with convolutional stems or spectral encoders to re-introduce spatial priors lost in pure self-attention.

Parallel to these eforts, multimodal pre-training frameworks such as CLIP [28] demonstrated that coupling images with textual supervision could improve robustness and partially recover texture sensitivity. Yet these gains came at the cost of billions of parameters, heavy compute budgets, and dependence on language data, conditions rarely feasible in specialized domains like medical imaging or remote sensing.

This evolution exposed a fundamental trade-of: transformers ofer unparalleled global reasoning but tend to neglect the local statistical structure that defines textures. Bridging this divide requires architectures that preserve hierarchical multiscale features while maintaining transformer-level global context.

HiPerViT addresses this gap by integrating a multi-scale ViT backbone with a Perceiverstyle latent bottleneck and compact bilinear pooling, enabling explicit second-order statistical modeling within a transformer framework. Sec. 3 discusses the biological intuitions that guided this design.

## 3. Biological Motivation

Human and non-human primates have evolved a visual system that excels at distinguishing textures under variations of scale, illumination, and context, precisely the challenges that computational models still struggle to overcome. Classical neurophysiology revealed that the early visual cortex (V1) acts as a filter bank, decomposing the retinal input into localized, orientation and frequency selective channels, not unlike Gabor filters [14]. This stage emphasizes micro-patterns: edges, gratings, and fine textures. However, vision does not operate as a strictly linear pipeline. Signals are processed in parallel, distributed across specialized cortical streams [39], ensuring that multiple scales and modalities of information can be extracted simultaneously.

As inputs propagate to mid-level areas such as V2 and V4, neurons integrate co-occurrence statistics, curvatures, and surface properties, enabling representations that capture both the fine statistics of texture and its global layout [29]. Evidence indicates that these areas act as hubs of multiscale integration, combining local filter responses with broader contextual cues [8]. Such processing reflects not only hierarchical depth but also parallel specialization: diferent cortical regions emphasize distinct visual attributes (e.g., color, form, motion), a principle known as functional specialization [39]. In this framework, texture perception emerges from the coordinated interaction of specialized modules rather than from a monolithic sequence of transformations.

At higher cortical stages, notably the inferotemporal (IT) cortex, representations become increasingly abstract and tolerant, merging local structure with semantic associations. Importantly, the integration of parallel streams into compact codes ensures both robustness and eficiency: vast amounts of sensory data are condensed into latent representations that remain discriminative under clutter, occlusion, and variability.

These insights guided the design of HiPerViT. Its multi-scale ViT backbone mirrors the hierarchical and parallel nature of cortical processing. The Compact Bilinear Pooling module is inspired by pairwise co-occurrence coding in mid-level vision [27]. The Perceiver-style latent bottleneck is analogous to the convergent compression observed in IT cortex, condensing distributed multi-stream inputs into compact, discriminative codes.

## 4. Model Architecture and Methodology

Our central hypothesis is that robust visual recognition in micro-structure-sensitive domains requires injecting second-order statistical structure directly into the transformer’s representational space. This is operationalized through Statistical Token Injection (STI), defined below. The remaining architectural components, multi-scale extraction, Perceiver-style distillation, and compact bilinear pooling, serve as the supporting infrastructure that makes STI practical and scalable.

## 4.1. Statistical Token Injection (STI)

Statistical Token Injection (STI) is a representation mechanism that converts compact second-order feature statistics into an explicit token that can participate directly in transformer self-attention. Given a feature map $\mathbf { F } \in \mathbb { R } ^ { N \times D }$ extracted from a backbone layer, STI first computes a compact second-order descriptor $\mathbf { z } _ { \mathrm { s r m } } \in \mathbb { R } ^ { M }$ using a Count Sketch bilinear projection, where $M \ll D ^ { 2 }$ . This operation provides a computationally eficient approximation of pairwise feature co-occurrence statistics without explicitly constructing the full outer-product representation.

The resulting descriptor is projected into the same embedding space as the spatial tokens:

$$
\mathbf { T } _ { \mathrm { s r m } } = W _ { e } \mathbf { z } _ { \mathrm { s r m } } + \mathbf { b } _ { e } , \qquad \mathbf { T } _ { \mathrm { s r m } } \in \mathbb { R } ^ { D } ,\tag{1}
$$

where $W _ { e } \in \mathbb { R } ^ { D \times M }$ and $\mathbf { b } _ { e } \in \mathbb { R } ^ { D }$ are learnable parameters. The projected representation $\mathbf { T } _ { \mathrm { s r m } }$ is then inserted as a first-class token into the spatial token sequence. Consequently, standard multi-head self-attention can jointly operate on first-order spatial representations and explicit second-order statistical information.

This formulation difers from conventional bilinear heads or late-fusion strategies, in which second-order descriptors are typically appended only after spatial representation learning has already occurred. In STI, the statistical representation becomes part of the attention process itself. Spatial tokens can therefore attend to global feature co-occurrence statistics, while the statistical token can be contextually modulated by the spatial structures present in the image. We refer to this bidirectional exchange between first- and second-order representations as cross-order interaction.

STI introduces only a small computational overhead. The additional learnable projection $W _ { e }$ accounts for less than 1% of the backbone parameter count in the evaluated configurations, and the mechanism requires no dedicated attention heads or additional attention layers. It can therefore be incorporated into a standard transformer sequence while preserving the underlying self-attention operation.

![](images/e78dbd637d0d573beaea6a63957f7f8d153972aa5ba7f847d6b934d41d31eca6.jpg)  
Figure 1. Architecture of the proposed HiPerViT network. The pipeline consists of: (1) a multi-scale ViT backbone for hierarchical feature extraction, (2) SRM projection to embed second-order statistics, (3) fusion through a Latent Transformer producing Key/Value pairs, (4) a Perceiver-style module with learnable latent queries and a self-attention tower for high-order interactions, and (5) final aggregation and classification.

## 4.2. HiPerViT: Extraction, Interaction, and Distillation

This subsection details the architecture of HiPerViT, a unified model that integrates hierarchical multi-scale feature extraction from a ViT backbone with second-order feature aggregation via compact bilinear pooling and eficient latent compression.

4.2.1. Global-Local Input Pathway. The pipeline begins with a Global-Local Dual Pathway. Texture recognition benefits from analyzing an image at both a broad structural scale and a high-resolution detail scale.

To implement this, and as conceptually depicted by the input section of the architecture in Figure 1, the raw input image I is transformed into two distinct complementary views: a downsampled global context image $X _ { g } ,$ and a high-resolution local foveal crop $X _ { \ell } .$ This dualpathway strategy guarantees that the subsequent feature extractor receives data emphasizing diferent textural scales:

• Global Context $( X _ { g } )$ : Captures overall structural information and long-range dependencies across macro-patterns.

• Local Foveal View $( X _ { \ell } )$ : Preserves fine-grained micro-patterns and high-frequency local statistics.

Each view $( X _ { g }$ and $X _ { \ell } )$ is then independently forwarded through a single, shared-weight ViT backbone; that is, both scales are processed by the same set of parameters rather than independent parallel branches. This weight-sharing design has two important consequences: (i) memory cost remains proportional to a single backbone, since no additional parameters are introduced for the extra view, and (ii) the shared filters are exposed to the same visual content at multiple scales during training, which encourages the emergence of scale-equivariant internal representations and improves generalization. The resulting hierarchical feature maps from both views are subsequently aggregated and passed to the Distillation module.

4.2.2. Hierarchical Feature Extraction and Second-Order Aggregation. Following the globallocal input preparation, the images $X _ { g }$ and $X _ { \ell }$ are processed by the Vision Transformer backbone. We leverage its inherent design to produce a sequence of feature maps at decreasing spatial resolutions, generating a set of Stage Activation Maps (SAMs) from distinct stages of the hierarchy.

The feature extraction step selects a subset of these SAMs, $\mathbf { F } _ { s _ { i } } .$ , corresponding to diferent depths $s _ { i } .$ These selected feature maps are first tokenized and transformed into sequences $\mathbf { T } _ { s _ { i } }$ From this point, the architecture diverges into two parallel streams of representation:

(1) Token Stream $( \mathbf { T } _ { m u l t i } )$ : The individual stage tokens $\mathbf { T } _ { s _ { i } }$ are concatenated along the sequence dimension to form a single, aggregated multi-scale token object, $\mathbf { T } _ { m u l t i } .$

$$
\mathbf { T } _ { m u l t i } = \operatorname { C o n c a t } ( \mathbf { T } _ { s _ { 1 } } , \mathbf { T } _ { s _ { 2 } } , \ldots , \mathbf { T } _ { s _ { k } } )\tag{2}
$$

where $\mathbf { T } _ { m u l t i } \ \in \ \mathbb { R } ^ { N \times D }$ , and $\begin{array} { r } { N = \sum _ { i = 1 } ^ { k } H _ { i } W _ { i } } \end{array}$ . This unified set serves as the highdimensional input for the subsequent attention-based Distillation module.

(2) Second-Order Stream $( { \bf Z } _ { S R M } ) ;$ : The Second-order Representation Module (SRM) is applied independently to the tokens of each stage $\mathbf { T } _ { s _ { i } }$ . Conceptually, this module aims to capture the full second-order statistics of the features, which corresponds to the sum of outer products of the token vectors:

$$
\mathbf { Z } _ { s _ { i } } ^ { \mathrm { f u l l } } = \sum _ { j = 1 } ^ { H _ { i } W _ { i } } \mathbf { t } _ { j } \mathbf { t } _ { j } ^ { \top }\tag{3}
$$

However, directly computing this high-dimensional tensor is computationally prohibitive. Therefore, in our implementation (detailed in Sec. 4.4), we approximate this operation using Compact Bilinear Pooling (CBP) via the Tensor Sketch trick, generating a compact stage-specific descriptor $\mathbf { Z } _ { s _ { i } }$ . These stage-specific second-order representations, $\mathbf { Z } _ { s _ { i } }$ , are then concatenated to form the final, compact multi-scale second-order feature vector $\mathbf { Z } _ { S R M } \mathrm { : }$

$$
{ \bf Z } _ { S R M } = \mathrm { C o n c a t } ( { \bf Z } _ { s _ { 1 } } , { \bf Z } _ { s _ { 2 } } , \ldots , { \bf Z } _ { s _ { k } } )\tag{4}
$$

The aggregated token set $\mathbf { T } _ { m u l t i }$ and the combined second-order vector $\mathbf { Z } _ { S R M }$ are both passed forward to the next stage, ensuring the system integrates both sequential and correlational feature streams.

4.2.3. Early Interaction, Latent Distillation, and Classification. The final phase of the HiPer-ViT pipeline integrates first-order spatial tokens with second-order statistical summaries via cross-order self-attention, followed by an asymmetric distillation. We describe the default interact→distill ordering here; Sec. 5.5 shows that alternative orderings yield statistically equivalent performance.

Cross-Order Visual Encoding: The high-dimensional spatial token sequence $\mathbf { T } _ { m u l t i }$ and the compact multi-scale second-order vector $\mathbf { Z } _ { S R M }$ are not simply concatenated at the output. Instead, $\mathbf { Z } _ { S R M }$ is embedded into the same dimensional space as the spatial tokens to form a statistical token, denoted $\mathbf { T } _ { s r m }$ . An augmented visual sequence is formed by prepending a class token and the statistical token to the multiscale spatial tokens:

$$
\mathbf { Z } _ { v i s } = \mathrm { C o n c a t } ( [ \mathrm { C L S } ] , \mathbf { T } _ { s r m } , \mathbf { T } _ { m u l t i } )\tag{5}
$$

This sequence is processed by a lightweight visual Transformer encoder, facilitating cross-order and cross-scale self-attention. The spatial tokens dynamically attend to the global statistical token (and vice-versa), integrating fine-grained correlations before any dimensionality reduction occurs.

Feature Distillation (The Perceiver-style Latent Bottleneck): The enriched sequence $\mathbf { \widetilde { Z } } _ { v i s } \in \mathbb { R } ^ { ( N + 2 ) \times D }$ must still be condensed. This is achieved through a Perceiver-style latent bottleneck. A set of trainable, low-dimensional Latent Query tokens $\mathbf { L } \in \mathbb { R } ^ { B \times D ^ { \prime } }$ (where $B \ll N )$ uses asymmetric cross-attention to query the interacting visual stream:

$$
\mathbf { L } _ { d i s t i l l e d } = \mathrm { C r o s s A t t e n t i o n } ( \mathbf { L } { \mathrm { ~ a s ~ Q u e r y } } , \widetilde { \mathbf { Z } } _ { v i s } { \mathrm { ~ a s ~ K e y / V a l u e } } )\tag{6}
$$

In the default configuration, HiPerViT distills a sequence where first- and second-order features have already contextually modulated each other. However, controlled topology ablations (Sec. 5.5) demonstrate that reversing this ordering or deferring SRM integration to a late-fusion stage yields equivalent accuracy, confirming that the gain arises from the SRM statistical prior rather than a specific wiring.

Classification: The distilled latent representation $\mathbf { L } _ { d i s t i l l e d }$ is aggregated via mean pooling to produce a final, compact vector:

$$
\mathbf { V } _ { f i n a l } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \mathbf { L } _ { d i s t i l l e d } ^ { ( i ) }\tag{7}
$$

This ultimate feature vector $\mathbf { V } _ { f i n a l }$ is fed into a linear classification head to project onto the number of target classes C:

$$
\hat { y } = \mathrm { S o f t m a x } ( \mathrm { M L P } ( \mathbf { V } _ { f i n a l } ) )\tag{8}
$$

The network is trained end-to-end by minimizing the standard Cross-Entropy loss $\mathcal { L } _ { \mathrm { C E } }$ between the predicted probabilities yˆ and the ground-truth labels y.

## 4.3. Overview and Notation

Given a batch of images

$$
\mathcal { X } = \{ X ^ { ( b ) } \} _ { b = 1 } ^ { B } , \qquad X ^ { ( b ) } \in \mathbb { R } ^ { 3 \times H \times W } ,\tag{9}
$$

we generate two complementary views: a global image $X _ { g }$ and a set of local crops $X _ { \ell } ,$ , mimicking peripheral and foveal vision. Each image is tokenized by a Vision Transformer (ViT) backbone into patch embeddings:

$$
\begin{array} { r } { \mathbf { Z } _ { g } \in \mathbb { R } ^ { B \times n _ { g } \times D } , \qquad \mathbf { Z } _ { \ell } \in \mathbb { R } ^ { B \times n _ { \ell } \times D } , } \end{array}\tag{10}
$$

where $D$ is the embedding dimension and $n _ { g } .$ , n are the number of tokens for global and local views, respectively.

## 4.4. Second-Order Statistical Branch

Textures are defined not only by first-order statistics (mean activations) but also by correlations among features. Inspired by this, we add a compact bilinear pooling module that approximates the outer product $\mathbf { \overline { { f } } ^ { \top } }$ of a feature vector $\mathbf { f } \in \mathbb { R } ^ { C }$ without quadratic cost.

The Count Sketch projection $S : \mathbb { R } ^ { C }  \mathbb { R } ^ { M }$ is defined as

$$
S ( \mathbf { f } ) [ m ] = \sum _ { i = 1 } ^ { C } s _ { i } f _ { i } \mathbf { 1 } _ { h _ { i } = m } ,\tag{11}
$$

where $h \in \{ 1 , \ldots , M \} ^ { C }$ and $s \in \{ \pm 1 \} ^ { C }$ are fixed random vectors. The bilinear feature is then sketched using convolution in the Fourier domain:

$$
{ \widehat { S ( \mathbf { f } \otimes \mathbf { f } } } ) = { \widehat { S ( \mathbf { f } ) } } \odot { \widehat { S ( \mathbf { f } } } ) ,\tag{12}
$$

with ⊙ denoting elementwise multiplication and b· the FFT. The inverse FFT yields the compact bilinear feature $\mathbf { c } \in \mathbb { R } ^ { M }$ , normalized via signed square-root and $\ell _ { 2 } { \mathrm { : } }$

$$
\mathbf { z } _ { \mathrm { s r m } } = { \frac { \mathrm { s i g n } ( \mathbf { c } ) \cdot { \sqrt { | \mathbf { c } | + \varepsilon } } } { \| \mathrm { s i g n } ( \mathbf { c } ) \cdot { \sqrt { | \mathbf { c } | + \varepsilon } } \| _ { 2 } } } .\tag{13}
$$

This vector $\mathbf { z } _ { \mathrm { s r m } }$ forms a statistical token, directly encoding second-order co-occurrences. Neuroscience suggests such pairwise coding is crucial in mid-level vision [8], providing a biological rationale for this branch.

## 4.5. Projection and Cross-Order Visual Encoding

A defining structural characteristic of HiPerViT is the shared interaction space for heterogeneous features. We project all branches into a unified embedding space of dimension D:

(14)

$$
\mathbf p _ { g } = \mathbf Z _ { g } W _ { p } + b _ { p } ,\tag{15}
$$

$$
\begin{array} { r } { \mathbf p _ { \ell } = \mathbf Z _ { \ell } W _ { p } + b _ { p } , } \end{array}\tag{16}
$$

$$
\mathbf { p } _ { \mathrm { s r m } } = \mathbf { z } _ { \mathrm { s r m } } W _ { s } + b _ { s } ,
$$

with $W _ { p } \in \mathbb { R } ^ { D \times D }$ and $W _ { s } \in \mathbb { R } ^ { M \times D }$ . By embedding $\mathbf { z } _ { \mathrm { s r m } }$ as $\mathbf { p } _ { \mathrm { s r m } }$ , we cast the dense second-order summary as a statistical token. We then concatenate these tokens into a single sequence:

$$
\mathbf { Z } _ { \mathrm { v i s } } = [ [ \mathrm { C L S } ] , \mathbf { p } _ { \mathrm { s r m } } , \mathbf { p } _ { g } , \mathbf { p } _ { \ell } ] \in \mathbb { R } ^ { B \times ( 1 + 1 + n _ { g } + n _ { \ell } ) \times D } ,\tag{17}
$$

and refine them with a lightweight Transformer encoder:

$$
\begin{array} { r } { \widetilde { \mathbf { Z } } _ { \mathrm { v i s } } = \mathrm { T r a n s } \mathbf { E n c } ( \mathbf { Z } _ { \mathrm { v i s } } ) . } \end{array}\tag{18}
$$

This explicitly models mutual dependencies: the spatial tokens $\left( \mathbf { p } _ { g } , \mathbf { p } _ { \ell } \right)$ query the global cooccurrence statistics $\left( \mathbf { p } _ { \mathrm { s r m } } \right)$ to weight relevant textural patterns, while the statistical token is contextually modulated by the dominant localized structures in the image.

## 4.6. Latent Distillation via Cross-Attention

The representation $ { \widetilde { \mathbf { Z } } } _ { \mathrm { v i s } }$ remains long and redundant. To distill information, we introduce L learnable latent vectors $\mathbf { Q } \in \mathbb { R } ^ { L \times D }$ , updated by cross-attention:

$$
\mathbf { L } = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } \widetilde { \mathbf { Z } } _ { \mathrm { v i s } } ^ { \top } } { \sqrt { D } } \right) \widetilde { \mathbf { Z } } _ { \mathrm { v i s } } .\tag{19}
$$

The output $\mathbf { L } \in \mathbb { R } ^ { B \times L \times D }$ compresses hundreds of tokens into $L \ll N$ informative latents, condensing distributed multi-scale signals into compact codes for recognition.

The number of latent queries L is a critical hyperparameter governing the trade-of between compression and representational fidelity. A small L forces the model to abstract away highfrequency spatial details, favoring dominant macro-structures and global context. Conversely, a larger L preserves more fine-grained micro-textural information but increases the computational cost of the subsequent self-attention layers. Empirically, we find that a moderate L (e.g., 64– 128) sufices to capture the relevant texture statistics without retaining pixel-perfect spatia reconstruction, efectively filtering out high-frequency noise while preserving the structural signatures required for classification.

## 4.7. Classification

We aggregate latents by mean pooling:

$$
\mathbf { h } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \mathbf { L } _ { i } ,\tag{20}
$$

and predict with a linear classifier:

$$
\hat { \mathbf { y } } = \mathrm { s o f t m a x } ( W _ { c } ^ { \top } \mathbf { h } + b _ { c } ) ,\tag{21}
$$

with $W _ { c } \in \mathbb { R } ^ { D \times C }$ and C the number of classes.

## 4.8. Training Methodology

We train with Soft Target Cross-Entropy, compatible with mixup:

$$
\mathcal { L } ( \hat { \mathbf { y } } , \tilde { \mathbf { t } } ) = - \sum _ { c = 1 } ^ { C } \tilde { t } _ { c } \log \hat { y } _ { c } ,\tag{22}
$$

where $\tilde { \mathbf { t } } = \lambda \mathbf { t } _ { a } + ( 1 - \lambda ) \mathbf { t } _ { b }$ is a convex combination of one-hot labels. We further apply RandAugment and Layer-wise Learning Rate Decay (LLRD):

$$
\eta _ { i } = \eta _ { 0 } \cdot \rho ^ { S - 1 - i } ,\tag{23}
$$

where i indexes blocks of the backbone, S is the total depth, and $\rho < 1$ the decay factor. This schedule applies lower learning rates to early, well-pretrained layers while allowing higher layers to adapt quickly.

## 4.9. Architectural Synthesis

A key consequence of HiPerViT’s design emerges from the role of the statistical token within the fusion pipeline. By making second-order statistics available as explicit tokens to cross-order self-attention, the model decouples final discriminability from the intrinsic linear separability of the SRM vector itself. Cross-order self-attention enables the spatial token stream to conditionally re-weight second-order correlations based on contextual structure: when the SRM signal is highly informative, the attention mechanism amplifies it; when it is noisy or underdetermined, the spatial tokens carry the discriminative burden. Consequently, the system’s classification power becomes a property of token interaction rather than of standalone feature quality. This reframes compact bilinear pooling, traditionally viewed as a static, orderless descriptor appended at the output of a network, into a dynamically contextualized representation mechanism whose utility is determined by the presence of cross-order attention rather than by a specific interaction–distillation ordering or the extraction depth of the embedding. As we demonstrate empirically, this property manifests as two complementary invariances: (i) final accuracy is invariant to the backbone depth at which the SRM is computed, even when raw discriminability varies by more than 30 percentage points (Sec. 5.6), and (ii) final accuracy is invariant to whether the statistical token interacts before or after latent distillation (Sec. 5.5).

## 5. Experiments and Analysis

We conducted a comprehensive set of experiments to validate the Hierarchical Perceiver– Bilinear Vision Transformers (HiPerViT). Our study is organized around five central questions: (1) How does HiPerViT compare with state-of-the-art (SOTA) texture recognition methods across diverse benchmarks? (2) What is the empirical justification for our architectural choices, particularly multi-scale input and hierarchical feature fusion? (3) How do second-order statistics evolve across backbone depth, and what does this reveal about HiPerViT’s robustness to the representational source of its SRM branch? (4) Does the interact→distill fusion preserve generic representation transferability or induce task-specific specialization? (5) How does HiPerViT balance accuracy with computational and memory cost across diferent ViT backbones?

## 5.1. Experimental Setup

Datasets and Evaluation Protocol. We evaluate HiPerViT on six public texture benchmarks with varying sizes, complexity, and acquisition conditions: DTD [2], FMD [34], KTH-TIPS2-b [23], GTOS-Mobile, USPTex [33], and 1200Tex [1]. This selection captures both controlled and in-the-wild conditions, enabling a robust assessment of generalization. To ensure rigorous and replicable evaluation, we follow established splitting protocols: for DTD, results are averaged over the 10 oficial train/test partitions; for 1200Tex, we apply a Repeated Stratified 5-Fold Cross-Validation (2 repeats); for all other datasets, we adopt the standard 5-fold cross-validation or predefined train/test splits as provided by the original authors.

Implementation Details and Optimization. The multi-scale input pathway generates a global view resized to 448×448 pixels via bicubic interpolation and a local foveal view derived from a Random Resized Crop (scale 0.5 to 1.0) of size 224 × 224 during training, transitioning to a deterministic center crop for evaluation. The ViT backbones (Small, Base, and Large) are exclusively initialized using DINOv2 self-supervised weights [25], ensuring no text-supervision is leaked into our vision-only evaluation. We train the model using the AdamW optimizer with a base learning rate of $3 \times 1 0 ^ { - 5 }$ and a weight decay of 0.05. Optimization proceeds for 10 epochs with an efective batch size of 32 (via gradient accumulation), following a cosine annealing schedule with a 10% linear warmup phase. To preserve the pre-trained structural representations, we apply Layer-wise Learning Rate Decay (LLRD) with a decay factor of 0.8 per block from the top down, while the randomly initialized classifier head and Perceiver bottleneck receive a 10× learning rate multiplier $( 3 \times 1 0 ^ { - 4 } )$

Augmentation Strategy. The network is trained with a Soft Target Cross-Entropy loss compatible with mixed augmentations. For unbalanced datasets, explicit class weighting is applied. Specifically, we apply RandAugment with magnitude (4, 12) strictly to the global view. In addition, we apply a stochastic mixing strategy per batch where CutMix $( \alpha = 1 . 0 )$ is selected with 50% probability; otherwise, standard Mixup $( \alpha = 0 . 5 )$ is applied. During evaluation, we report results using Test-Time Augmentation (TTA), averaging the softmax predictions over the original image and horizontal, vertical, and 180<sup>◦</sup> flipped spatial variations.

## 5.2. Comparison with State-of-the-Art

Table 1 consolidates our comparison against the strongest previously published vision-only methods and transformer-based architectures. The unified structure, divided into subtables (a) and (b), enables a clearer side-by-side interpretation of both external SOTA results and controlled transformer baselines.

It is important to contextualize the metrics in these two subtables. The "Previous SOTA" results in Table 1a reflect values exactly as reported in the respective original publications. As such, these numbers are inherently heterogeneous: they stem from distinct underlying architectures (e.g., ConvNeXt-XXL vs. ViT-L), varying pre-training modalities, and highly specific hyperparameter tuning eforts tailored by the original authors.

To provide a more rigorous, apples-to-apples evaluation of our architectural claims, Table 1b compares HiPerViT against standard transformer architectures trained under the identical protocol detailed in Section 5.1. While this protocol establishes a strong, modern baseline anchored around ViT and DINOv2 weights, we acknowledge that a single, unified training recipe may not represent the absolute optimal hyperparameter configuration for every competing architecture (such as Swin or ConvNeXt). Nevertheless, holding the training regime, data splits, and augmentations perfectly constant isolates the performance delta attributable strictly to HiPerViT’s hierarchical fusion and second-order pooling mechanisms.

Overall, HiPerViT establishes new vision-only state-of-the-art on all six benchmarks, with substantial gains on five. On DTD, our model surpasses a strong ViT-L/14 baseline by $+ 3 . 0 5 \mathrm { p p }$ , illustrating the benefit of hierarchical fusion on highly variable textures. The largest improvements relative to previously published results appear on GTOS-Mobile (+10.48 pp over RADAM [32]) and 1200Tex (+10.1 pp over VisGraphNet [6]), demonstrating robustness under challenging illumination and high-resolution settings. On USPTex, HiPerViT achieves perfect accuracy (100.00%), surpassing the previous best (99.82%). However, rather than simply claiming an architectural breakthrough, this perfect score on USPTex signals severe dataset saturation. It suggests that the discriminative power of modern DINOv2 pre-trained features—potentially combined with structural artifacts of the dataset itself, such as highly similar or near-duplicate textural crop samples across splits or preprocessing leakage—is suficient to entirely over-solve the benchmark.

When compared to alternative transformer architectures trained with identical protocols (Table 1b), HiPerViT outperforms all baselines across all datasets, confirming the efectiveness of its multiscale fusion and hierarchical feature integration.

Table 1. Top-1 accuracy (%) across six texture benchmarks. (a) Comparison with previously published vision-only state-of-the-art. (b) Comparison with transformer-based models trained under identical settings. Best results are in bold.

(a) Comparison with previously published vision-only SOTA.
<table><tr><td>Dataset</td><td>HiPerViT (%)</td><td>Previous SOTA</td><td>Accuracy (%)</td></tr><tr><td>DTD</td><td>93.05</td><td>Linear-FT (ViT-L/14)</td><td>90.00</td></tr><tr><td>FMD</td><td>97.80</td><td>RADAM (ConvNeXt-XXL)</td><td>97.10</td></tr><tr><td>KTH-TIPS2-b</td><td>98.42</td><td>RADAM (ConvNeXt-XXL)</td><td>97.40</td></tr><tr><td>GTOS-Mobile</td><td>94.58</td><td>RADAM (ConvNeXt-B)</td><td>84.10</td></tr><tr><td>USPTex</td><td>100.00</td><td>Local Graphs via Random Encoding</td><td>99.82</td></tr><tr><td>1200Tex</td><td>97.50</td><td>VisGraphNet</td><td>87.40</td></tr></table>

(b) Comparison with transformer architectures under identical training settings.
<table><tr><td>Model</td><td>DTD</td><td>FMD</td><td>KTH-TIPS2-b</td><td>GTOS-Mobile</td><td>USPTex</td><td>1200Tex</td></tr><tr><td>HiPerViT</td><td>93.05</td><td>97.80</td><td>98.42</td><td>94.58</td><td>100.00</td><td>97.50</td></tr><tr><td>ViT-B/16</td><td>73.40</td><td>91.50</td><td>85.86</td><td>84.72</td><td>99.78</td><td>97.00</td></tr><tr><td>DeiT</td><td>69.73</td><td>89.00</td><td>82.32</td><td>88.79</td><td>99.78</td><td>97.50</td></tr><tr><td>Swin</td><td>77.34</td><td>92.00</td><td>88.89</td><td>89.20</td><td>100.00</td><td>97.00</td></tr><tr><td>VORTEX</td><td>87.10</td><td>93.90</td><td>95.10</td><td>93.80</td><td>100.00</td><td>97.50</td></tr></table>

Sources for panel (a): Linear-FT (ViT-L/14) [26]; RADAM [32]; Local Graphs via Random Encoding [5]; VisGraphNet [6]. Dataset references: DTD [2], FMD [34], KTH-TIPS2-b [23], USPTex [33], 1200Tex [1].

## 5.3. Impact of Multi-Scale and Hierarchical Fusion

To disentangle design choices, we performed a hyperparameter sweep over input scale pairs and backbone stage pairs. Results are summarized in Figure 2.

Two clear trends emerge: (i) Combining coarse and fine input scales (e.g., 256px and 128px) consistently outperforms single-scale setups, validating the need for complementary global and local context. (ii) Fusing early- or mid-level backbone features with deeper semantic stages yields the best results, outperforming shallow-only or deep-only combinations. This confirms our core claim: HiPerViT succeeds not by depth alone, but by coupling diverse representational levels.

The observation that intermediate backbone stages paired with deeper layers maximize accuracy is consistent with the principle that mid-level integration of local detail and coarse context precedes efective high-level recognition.

## 5.4. Component Ablation: Contribution of Each Module

To isolate the contribution of each architectural component in HiPerViT, we perform a controlled ablation study over the same training protocol, data preprocessing, and evaluation procedure used throughout Sec. 5. We progressively disable or replace individual modules while keeping the remaining pipeline unchanged. In particular, we consider: (i) a plain transformer backbone (Backbone Only); (ii) feeding only the multi-scale image pyramid without hierarchical fusion (Multi-Scale Image Only); (iii) using only the SRM branch (SRM Only); (iv) using only the Perceiver-based bottleneck pathway (Perceiver Only); and (v) the full HiPerViT model. Results (Top-1 accuracy, %) are summarized in Tab. 3.

![](images/f8df517a2d96345cf856ee7a41aee45c90b8f4b1a2b8687022a0940a54dbcc80.jpg)  
Figure 2. Performance heatmaps for input scale pairs (left) and backbone stage pairs (right). Cells show mean accuracy. Axis labels are color-coded by individual component performance tier.

Table 2. Top-performing configurations of scale and stage combinations.
<table><tr><td>Rank</td><td>Scales</td><td>Stages</td><td>Accuracy (%)</td></tr><tr><td>1</td><td>128-64</td><td>1-5</td><td>97.50</td></tr><tr><td>2</td><td>256-128</td><td>1-5</td><td>97.50</td></tr><tr><td>3</td><td>256-64</td><td>8-9</td><td>97.08</td></tr><tr><td>4</td><td>256-128</td><td>1-8</td><td>97.08</td></tr><tr><td>5</td><td>256-128</td><td>4-5</td><td>97.08</td></tr><tr><td></td><td>…</td><td>…</td><td>…</td></tr></table>

Impact on controlled texture benchmarks. Across DTD, 1200Tex, USPTex, and FMD, the full model consistently outperforms the strongest partial variants, indicating that the gains do not stem from a single isolated component but from their combination. Notably, the Perceiver Only variant produces a substantial improvement over the backbone on DTD (from 83.35 to 91.44), suggesting that the bottleneck aggregation mechanism is particularly efective at consolidating cues relevant to diverse texture patterns. However, on 1200Tex and USPTex the backbone alone already achieves near-perfect performance. The Multi-Scale Image Only variant reaches exactly 100.00% on USPTex. The fact that a partial pipeline can perfectly solve USPTex corroborates the notion that this dataset provides limited granular signal for evaluating complex architectural additions, as the task is trivialized by the baseline features and multi-scale pre-processing. The improvements from the full HiPerViT assembly are therefore most reliably observed on challenging benchmarks with robust performance headroom, like DTD and 1200Tex.

Role of SRM and multi-scale cues. The SRM-only configuration improves over the backbone on KTH-TIPS2-b (from 87.79 to 89.23), which is consistent with the hypothesis that high-frequency residual cues can be informative under certain material/illumination changes.

Table 3. Component ablation of HiPerViT using the ViT-S/16 backbone (Top-1 accuracy, %). The full model difers from Table 1 because this ablation uses ViT-S/16, whereas the main comparison reports the best ViT-L/16 configuration. Best results per column are in bold.
<table><tr><td>Variant</td><td>DTD</td><td>1200</td><td>USP</td><td>FMD</td><td>KTH</td><td>GTOS</td></tr><tr><td>Backbone</td><td>83.35</td><td>96.97</td><td>99.74</td><td>95.05</td><td>87.79</td><td>93.42</td></tr><tr><td>Multi-Scale</td><td>82.98</td><td>94.17</td><td>100.00</td><td>95.05</td><td>83.42</td><td>91.48</td></tr><tr><td>SRM</td><td>82.98</td><td>95.00</td><td>99.74</td><td>92.08</td><td>89.23</td><td>93.65</td></tr><tr><td>Perceiver</td><td>91.44</td><td>94.17</td><td>98.95</td><td>96.04</td><td>88.64</td><td>94.92</td></tr><tr><td>HiPerViT</td><td>92.66</td><td>97.50</td><td>99.74</td><td>97.03</td><td>88.39</td><td>93.04</td></tr></table>

By contrast, relying solely on multi-scale imagery without hierarchical fusion does not yield consistent gains and can slightly degrade performance on KTH-TIPS2-b, indicating that scale diversity alone is insuficient unless it is integrated via a structured fusion mechanism.

Complementarity and full-model behavior. The full HiPerViT configuration achieves the best overall results on three of the four fully reported datasets (DTD, 1200Tex, FMD), while remaining competitive on USPTex and KTH-TIPS2-b. This pattern supports the central design premise of HiPerViT: multi-scale inputs, SRM-guided residual cues, and Perceiver-style global aggregation are complementary, and the hierarchical fusion strategy is essential for translating this complementarity into consistent accuracy gains across benchmarks. In full-data settings the efect of the statistical token is modest, but becomes substantially larger under low-data training (Sec. 5.4.1).

Capacity-Dependence of Ablation Trends (ViT-S vs. ViT-L). It should be noted that while our headline state-of-the-art results (Table 1) are achieved using the highest-capacity ViT-L/16 backbone, this ablation study is purposefully conducted using the smaller ViT-S/16 backbone. This deliberate discrepancy serves a structural aim: immense parameter models often possess the raw capacity to implicitly approximate complex visual patterns, which can mask the specific contributions of explicit structural priors like multi-scale fusion or SRM texture encoding. By restricting the study to a lower-capacity backbone, we isolate the architectural impact of the HiPerViT modules, demonstrating that the structural topology itself—and not just parameter count—drives the recognition improvements. While the functional trends of complementarity hold across all scales, the relative magnitude of improvement provided by our modules is naturally more pronounced in compact backbones that rely heavily on these explicit priors.

5.4.1. Data-Eficiency of the Statistical Token. Does the SRM statistical token provide larger benefits when supervision is scarce, i.e., when the backbone cannot easily internalize cooccurrence priors from data? To test this, we subsample the DTD training set using stratified sampling at {10, 20, 50, 100}% of the training data, keeping the test set unchanged. We train three variants under the identical recipe described in Sec. 5.1: (i) SRM token (full HiPerViT), (ii) GAP token (first-order pooled descriptor injected as the statistical token), and (iii) none (no statistical token; Perceiver-only fusion). We report mean±std over 3 seeds with TTA.

The SRM token yields its largest gains in the low-data regime: at 10% of DTD training data, SRM improves over the no-token variant by +1.64 pp, while the advantage narrows as the fraction increases and becomes marginal at 50–100%. This trend suggests that secondorder co-occurrence modeling primarily acts as an inductive bias that compensates for limited supervision; with suficient data and a strong pretrained backbone, the fusion tower can recover comparable performance using first-order summaries or no explicit statistical token. Notably, at 50% data the GAP token slightly outperforms SRM (−0.13 pp), underscoring that the benefit is regime-dependent rather than absolute.

Table 4. Low-data regime on DTD. SRM, GAP, and None variants were trained under identical conditions at each fraction. ∆ columns report the advantage of SRM over the comparison variant. Values are mean±std over 3 seeds.
<table><tr><td>Frac.</td><td>SRM</td><td>GAP</td><td>None</td><td>∆(S-N)</td><td>∆(S-G)</td></tr><tr><td>10%</td><td>81.52±0.44</td><td>80.90±0.43</td><td> $7 9 . 8 8 { \pm } 0 . 2 6 $ </td><td>+1.64</td><td>+0.62</td></tr><tr><td>20%</td><td>88.55±0.25</td><td> $8 7 . 7 1 { \pm } 0 . 9 0 $ </td><td> $8 7 . 9 4 { \pm } 0 . 3 3 $ </td><td>+0.61</td><td>+0.84</td></tr><tr><td>50%</td><td> $9 1 . 4 5 { \pm } 0 . 2 9 $ </td><td> $9 1 . 5 8 { \pm } 0 . 0 9$ </td><td> $9 1 . 2 9 { \pm } 0 . 5 8 \ $ </td><td> $+ 0 . 1 6$ </td><td>-0.13</td></tr><tr><td>100%</td><td>92.77±0.13</td><td>92.57±0.09</td><td> $9 2 . 3 4 { \pm } 0 . 3 7$ </td><td> $+ 0 . 4 3$ </td><td>+0.20</td></tr></table>

Table 5. Fusion topology comparison on DTD (Partition 1, ViT-S/14 DI-NOv2). Mean ± std over 3 seeds. ∆ is computed relative to Topology A under TTA. All topologies are parameter-matched.
<table><tr><td>Topology</td><td>Acc (no TTA) Acc (TTA) ∆ vs A</td><td></td><td></td></tr><tr><td> $\mathrm { ~ A ~ } ( \mathcal { T }  \mathcal { D } )$ </td><td> $8 7 . 1 1 \pm 0 . 2 7$ </td><td> $8 7 . 2 2 \pm 0 . 3 0$ </td><td>0.00</td></tr><tr><td>B  $( \mathcal { D }  \mathcal { T } )$ </td><td> $8 7 . 2 9 \pm 0 . 2 9$ </td><td> $8 7 . 3 2 \pm 0 . 3 5$ </td><td> $+ 0 . 1 1$ </td></tr><tr><td>C (Late fusion)</td><td> $8 7 . 3 6 \pm 0 . 4 0$ </td><td> $8 7 . 4 7 \pm 0 . 3 3$ </td><td> $+ 0 . 2 5$ </td></tr></table>

This complements the mechanistic analysis in Sec. 5.6: while the interact→distill topology is robust to the origin and quality of second-order cues, explicit SRM modeling provides measurable benefits precisely when learning signals are scarce.

## 5.5. Fusion Topology Robustness and Order Invariance

A natural question is whether the performance of HiPerViT depends on the specific ordering of interaction and distillation, or whether the gain stems primarily from the SRM statistical prior itself. To test this, we evaluate three controlled wiring variants that difer only in how the statistical token is integrated with the spatial token stream, while matching parameter counts and FLOPs:

• Topology A (I → D): Interaction before distillation. The statistical token participates in cross-order self-attention with the full spatial token sequence before Perceiver-style compression (the default HiPerViT configuration).

• Topology B $( \mathcal { D }  \mathcal { T } )$ : Distillation before interaction. Spatial tokens are first compressed via the Perceiver bottleneck; the statistical token then interacts with the distilled latent sequence.

• Topology C (Late fusion): No explicit cross-order attention. The SRM descriptor is concatenated with the pooled latent representation at the classifier input.

Table 5 reveals that performance is largely invariant to fusion ordering. On DTD Partition 1 (3 seeds), all three topologies fall within a narrow band $\mathrm { ( \leq 0 . 2 5 p p }$ under TTA), with diferences well within the run-to-run standard deviation. Notably, the best mean accuracy is achieved by late fusion (Topology C), not by the interact→distill ordering (Topology A).

Table 6. SRM capture-depth ablation on DTD (ViT-S/14 DINOv2). Fusion topology is fixed at pre\_interact. Mean ± std over 3 seeds. ∆ and p-values are computed against the Late baseline via paired t-test.
<table><tr><td>SRM source</td><td>Accuracy (%)</td><td>Δ vs Late</td><td>p-value</td></tr><tr><td>SRM@Early ( 25%)</td><td> $8 2 . 9 1 \pm 0 . 1 8$ </td><td>-0.07</td><td>0.75</td></tr><tr><td>SRM@Mid ( 50%)</td><td> $8 2 . 9 1 \pm 0 . 2 6$ </td><td>-0.07</td><td>0.82</td></tr><tr><td>SRM@Late ( 100%)</td><td> $8 2 . 9 8 \pm 0 . 1 7$ </td><td>0.00</td><td></td></tr><tr><td>SRM@Multi (all)</td><td> $8 3 . 0 9 \pm 0 . 1 6$ </td><td>+0.11</td><td>0.37</td></tr></table>

This result carries an important implication: the SRM statistical prior is the key ingredient; the interaction topology is secondary. All three variants share the same backbone, the same compact bilinear pooling module, and the same training recipe; only the integration topology difers. The fact that performance is essentially indistinguishable across wirings indicates that the SRM provides a transferable representational prior whose utility is not sensitive to whether it is injected before or after latent compression. This finding further validates the STI principle: the gain comes from making second-order statistics available to the representation, not from a specific architectural wiring.

Given this near-identical performance, we adopt I → D (Topology A) as the reference topology in all subsequent experiments for two reasons: (i) it ofers a direct token-level interaction site that facilitates mechanistic analysis (Sec. 5.6), and (ii) it provides a unified implementation across datasets without requiring post-hoc concatenation. This choice is motivated by interpretability and consistency, not by a claimed performance advantage. Future work will probe whether regimes of extreme data scarcity, heavy domain shift, or fine-grained spatial corruption reveal conditions under which cross-order attention provides a measurable advantage over late fusion.

## 5.6. Mechanistic Analysis of Second-Order Representations

The component ablation (Sec. 5.4) establishes that the SRM branch contributes to HiPer-ViT’s performance. A natural follow-up question is: does the quality or origin of the secondorder statistics matter? Specifically, since the ViT backbone is known to progressively transform representations from low-level primitives to high-level semantics [4], one might hypothesize that SRM vectors derived from earlier layers, which preserve finer-grained texture primitives, should yield stronger texture discriminability than those from deeper, more semantic layers.

To test this hypothesis rigorously, we conduct three complementary analyses on the DTD benchmark using a ViT-S/14 DINOv2 backbone with the fusion topology fixed at the pre\_interact configuration. In each experiment, we vary only the backbone block that supplies tokens to the SRM module, selecting block indices at approximately 25%, 50%, and 100% of the backbone depth.

5.6.1. SRM Capture-Depth Ablation. We first train the full HiPerViT pipeline while varying only which backbone block supplies tokens to the SRM. All other hyperparameters, including fusion topology, augmentation, and optimizer settings, are held constant. Each variant is trained with three random seeds.

Table 6 reveals a striking result: final classification accuracy is statistically invariant to SRM capture depth. All diferences are ≤0.11 pp, well within the run-to-run standard deviation, and no paired t-test reaches significance $\left( p \ge 0 . 3 7 \right)$ . This finding directly contradicts the hypothesis that early-layer SRM vectors should be superior for texture recognition.

Table 7. Linear probe accuracy on DTD using only the frozen SRM vectors at each backbone depth. No fusion head or fine-tuning is applied.
<table><tr><td>SRM source</td><td>Linear probe acc (%) ∆ vs Late</td><td></td></tr><tr><td>SRM@Early (25%)</td><td>46.17</td><td>-33.56</td></tr><tr><td>SRM@Mid ( 50%)</td><td>71.06</td><td>-8.67</td></tr><tr><td>SRM@Late ( 100%)</td><td>79.73</td><td>0.00</td></tr></table>

Table 8. Pairwise similarity between SRM vectors extracted at diferent backbone depths on DTD. (a) Linear CKA. (b) Mean sample-wise Pearson correlation.

(a) Linear CKA
<table><tr><td>Early</td><td>Mid Late</td></tr><tr><td>Early 1.000</td><td>0.742 0.304</td></tr><tr><td>Mid 0.742</td><td>1.000 0.492</td></tr><tr><td>Late 0.304</td><td>0.492 1.000</td></tr></table>

(b) Mean Pearson correlation
<table><tr><td>Early</td><td>Mid Late</td></tr><tr><td>Early 1.000</td><td>0.317 0.004</td></tr><tr><td>Mid 0.317</td><td>1.000 0.026</td></tr><tr><td>Late 0.004</td><td>0.026 1.000</td></tr></table>

5.6.2. Linear Probe Analysis. The depth invariance of the full model raises a critical question: do SRM vectors at diferent depths actually carry the same discriminative information, or is the fusion head compensating for diferences?

To answer this, we freeze the pretrained DINOv2 backbone, extract SRM vectors at each capture depth for the entire train and test sets, and train a logistic regression classifier on the raw SRM vectors alone, without the fusion head, Perceiver, or fine-tuning.

Table 7 shows that the raw discriminative power of SRM vectors is strongly depthdependent: Early SRM vectors achieve only 46.17% accuracy (barely above chance for 47 classes), while Late vectors reach 79.73%. This 33.56 pp gap demonstrates that texturediscriminative statistics become progressively more linearly separable with backbone depth.

Combined with the depth-invariance result in Table 6, this reveals a key mechanistic property: the interact→distill fusion head can extract equivalent classification performance from SRM vectors of vastly diferent intrinsic quality. Even when the SRM signal is weak (46% linear separability), the cross-order self-attention mechanism compensates by leveraging the complementary spatial token stream.

5.6.3. Representational Similarity Analysis. To understand the geometric relationship between SRM vectors at diferent depths, we compute pairwise linear CKA [16] and mean sample-wise Pearson correlation on the test set.

The results in Table 8 reveal that SRM vectors at diferent depths are not redundant, they occupy geometrically distinct subspaces. Early–Late CKA is only 0.304, and sample-wise Pearson correlation is nearly zero $( r = 0 . 0 0 4 )$ . This means the depth invariance observed in the full model is not a trivial consequence of representational redundancy: the backbone encodes genuinely diferent second-order statistics at each depth, yet the fusion mechanism integrates them equally well.

Table 9. Top-1 accuracy (%) under test-time corruptions on DTD, with SRM source depth swapped at inference. Parentheses show degradation from the clean baseline (83.09%). The checkpoint was trained with SRM@Late.
<table><tr><td>Corruption</td><td>Sev.</td><td>Early</td><td>Mid</td><td>Late</td></tr><tr><td rowspan="3">Gaussian noise</td><td>Mild</td><td>83.35 (+0.3)</td><td>83.40 (+0.3)</td><td>83.51 (+0.4)</td></tr><tr><td>Med.</td><td>83.03 (-0.1)</td><td>83.30 (+0.2)</td><td>83.19 (+0.1)</td></tr><tr><td>Sev.</td><td>82.18 (-0.9)</td><td>82.66 (-0.4)</td><td>82.13 (-1.0)</td></tr><tr><td rowspan="3">Gaussian blur</td><td>Mild</td><td>82.87 (-0.2)</td><td>82.87 (-0.2)</td><td>82.87 (-0.2)</td></tr><tr><td>Med.</td><td>82.39 (-0.7)</td><td>82.39 (-0.7)</td><td>82.39 (-0.7)</td></tr><tr><td>Sev.</td><td>81.76 (-1.3)</td><td>81.76 (-1.3)</td><td>81.76 (-1.3)</td></tr><tr><td rowspan="3">JPEG compr.</td><td>Mild</td><td>81.44 (-1.6)</td><td>81.44 (-1.6)</td><td>81.44 (-1.6)</td></tr><tr><td>Med.</td><td>74.26 (-8.8)</td><td>74.26 (-8.8)</td><td>74.26 (-8.8)</td></tr><tr><td>Sev.</td><td>48.19 (-34.9)</td><td>48.19 (-34.9)</td><td>48.19 (-34.9)</td></tr><tr><td rowspan="3">Contrast red.</td><td>Mild</td><td>83.14 (+0.1)</td><td>83.14 (+0.1)</td><td>83.14 (+0.1)</td></tr><tr><td>Med.</td><td>82.82 (-0.3)</td><td>82.82 (-0.3)</td><td>82.82 (-0.3)</td></tr><tr><td>Sev.</td><td>76.81 (-6.3)</td><td>76.81 (-6.3)</td><td>76.81 (-6.3)</td></tr></table>

Taken together, the linear probe and RSA results paint a coherent picture: modern selfsupervised ViTs progressively refine texture-discriminative statistics across depth (Table 7), encoding them in increasingly distinct geometric subspaces (Table 8), while HiPerViT’s interact→distill topology robustly integrates these statistics regardless of their origin or quality (Table 6).

5.6.4. Corruption Robustness. Finally, we test whether depth sensitivity emerges under image degradation. Using a model trained with the SRM@Late configuration, we apply test-time corruptions at three severity levels and re-evaluate with the SRM source depth swapped to early, mid, or late.

Table 9 confirms that depth invariance persists under corruption. For blur, JPEG, and contrast, all three SRM depths yield identical accuracy at every severity. Under Gaussian noise, minor fluctuations appear (<0.5 pp spread), but no systematic advantage for any particular depth emerges. This indicates that the fusion head’s ability to compensate for SRM quality is robust even when input quality degrades substantially.

Summary. These four analyses collectively demonstrate that: (1) raw SRM discriminability increases substantially with backbone depth; (2) SRM vectors at diferent depths encode genuinely diferent statistics; yet (3) HiPerViT’s interact→distill fusion achieves invariant final performance across depths, even under corruption. This establishes that the architectural topology, not the source depth of second-order statistics, is the primary mechanism underlying HiPerViT’s texture recognition capability.

## 5.7. Cue-Conflict Behavior and Conditional Texture Utilization

The mechanistic analyses above demonstrate that HiPerViT’s fusion topology robustly integrates second-order statistics regardless of their intrinsic quality. A complementary question is whether this integration alters the model’s perceptual bias, specifically, whether injecting an explicit texture channel causes the model to rely more on texture cues at the expense of shape.

Table 10. Cue-conflict analysis on ImageNet-100. Both models are trained with identical recipes. ∆TBI is relative to the ViT baseline. Higher TBI indicates stronger texture reliance; higher SBI indicates shape reliance.
<table><tr><td>Model</td><td>SBI (%) TBI (%) CA (%) ΔTBI</td><td></td><td></td><td></td></tr><tr><td>ViT (baseline)</td><td>88.67</td><td>1.20</td><td>89.87</td><td></td></tr><tr><td>HiPerViT</td><td>91.47</td><td>0.53</td><td>92.00</td><td>-0.67</td></tr></table>

We evaluate this using cue-conflict images generated via AdaIN style transfer [13] on ImageNet-100: each image preserves the global shape (silhouette) of one class while adopting the texture statistics of a diferent class. For each prediction, we determine whether the model matched the shape label $y ^ { s }$ , the texture label $y ^ { t }$ , or neither, and compute three metrics:

(24)

$$
\mathrm { S B I } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ \hat { y } _ { i } = y _ { i } ^ { s } ] ,\tag{25}
$$

$$
\mathrm { T B I } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ \hat { y } _ { i } = y _ { i } ^ { t } ] ,\tag{26}
$$

$$
\mathrm { C A } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } [ \hat { y } _ { i } \in \{ y _ { i } ^ { s } , y _ { i } ^ { t } \} ] .
$$

where SBI is the Shape-Bias Index, TBI is the Texture-Bias Index, and CA is the overall Conflict Accuracy.

Table 10 reveals a key finding: HiPerViT does not increase texture bias relative to a baseline ViT when trained on ImageNet-100. Both models remain strongly shape-dominant (TBI < 2%), with HiPerViT exhibiting slightly higher shape-bias index and marginally improved overall conflict accuracy.

This result clarifies an important property of the interact→distill topology: the statistical token does not override task-optimal cues. Rather, cross-order attention dynamically amplifies the feature stream that maximizes discriminative signal under the training objective. On ImageNet, where shape is the statistically dominant cue, this corresponds to global structure alignment. Combined with our low-data regime analysis (Sec. 5.4.1) and texture-benchmark results (Sec. 5.4), these findings support a conditional-utilization hypothesis: HiPerViT augments the representational space with second-order statistics without imposing texture bias when texture is not task-relevant.

Formally, let T denote spatial tokens and S the statistical token. The cross-attention output can be expressed as:

$$
A ( T , S ) = \alpha ( T , S ) T + \beta ( T , S ) S ,\tag{27}
$$

where the learned coeficients α and $\beta$ adapt based on task statistics. Cue-conflict results suggest that α dominates under object-centric supervision, while texture benchmarks increase the contribution of $\beta .$ This behavior is consistent with the depth-invariance result in Sec. 5.6, where we show that final accuracy is invariant to SRM intrinsic discriminability. The interact→distill topology thus acts as an adaptive gating mechanism rather than a static statistical augmentation.

Table 11. Cross-dataset transfer accuracy (%) via linear probing. Features are extracted at four representation stages of increasing architectural specialization.
<table><tr><td>Direction</td><td>CLS</td><td>Multi</td><td> $\mathbf { P r e - F u s . }$ </td><td>Latent</td></tr><tr><td> $\mathrm { D T D } \to \mathrm { G T O S }$ </td><td>90.74</td><td>90.01</td><td>90.06</td><td>80.14</td></tr><tr><td> $\mathrm { G T O S }  \mathrm { D T D }$ </td><td>81.22</td><td>81.22</td><td>72.71</td><td>64.10</td></tr></table>

## 5.8. Cross-Dataset Transfer and Representation Specialization

The mechanistic analyses in Sec. 5.6 demonstrate that HiPerViT’s interact→distill topology robustly integrates second-order statistics regardless of their intrinsic quality. A natural follow-up question is whether this fusion mechanism preserves the generic transferability of the underlying backbone or induces task-specific alignment that reduces cross-domain generality.

To investigate this, we conduct a controlled cross-dataset transfer study between DTD (47 classes, general textures) and GTOS-Mobile (31 classes, outdoor ground textures). We train HiPerViT on a source dataset and evaluate transfer via linear probing on the target dataset, comparing representations extracted at four progressively more specialized stages:

(1) Frozen DINOv2 CLS: the [CLS] token from the original pretrained backbone (no fine-tuning);

(2) Frozen DINOv2 Multi-stage: concatenated GAP features at the same capture blocks used by HiPerViT, from the frozen backbone;

(3) HiPerViT Backbone: multi-stage features from the fine-tuned backbone, extracted before the SRM and fusion head;

(4) HiPerViT Latent: the post-distillation latent representation after interact→distill fusion.

Table 11 reveals a clear representational gradient. When source diversity is high (DTD→GTOS) the fine-tuned backbone features before fusion match frozen DINOv2 performance (90.06% vs. 90.01–90.74%), indicating that supervised training with hierarchical fusion does not degrade backbone-level universality under favorable source conditions. In contrast, the postdistillation latent features show reduced cross-domain separability (80.14%), confirming that the interact→distill bottleneck induces task-adaptive alignment of second-order correlations.

The reverse direction (GTOS→DTD) exhibits stronger degradation in both pre-fusion and latent features (72.71% and 64.10%), consistent with over-specialization under limited source diversity, GTOS contains fewer classes and a narrower visual vocabulary than DTD.

These results admit a principled interpretation. Let $\mathbf { f } _ { b }$ denote backbone features and $\mathbf { f } _ { l }$ the latent features after cross-attention distillation. Because the latent queries L are optimized under source supervision, the cross-attention operator $A ( \mathbf { L } , \mathbf { Z } _ { \mathrm { v i s } } )$ induces a source-conditioned projection of feature correlations. Consequently, $\mathbf { f } _ { l }$ becomes a distribution-aligned representation rather than a domain-agnostic embedding, which explains why $\mathbf { f } _ { b }$ retains transferability whereas $\mathbf { f } _ { l }$ exhibits task specialization.

These findings complement the depth-invariance analysis (Sec. 5.6): while the interact→distill topology compensates for weak second-order statistics in-domain, it also encourages contextconditioned statistical alignment. This mechanism improves discriminative separability within the training distribution but reduces isotropic generality across domains. Together, these results highlight a controllable trade-of between universality and specialization: HiPerViT preserves general backbone representations while enabling explicit statistical alignment through its fusion bottleneck, ofering practitioners a choice between generic representation reuse and domain-specific optimization.

## 5.9. Computational Eficiency

Finally, we benchmark HiPerViT across four ViT backbones (S, B, L) under the bestperforming configuration. Figure 3 reports FLOPs, latency, and peak GPU memory.

The ViT-L/16 variant is most expensive (27.0 GFLOPs), yet accuracy gains over smaller backbones are modest. In contrast, ViT-B/16 reduces FLOPs by over 60% while maintaining nearly identical accuracy. Remarkably, ViT-S/16 achieves strong performance at only 4.0 GFLOPs and 214 MiB peak memory, making HiPerViT viable for resource-constrained deployment.

Crucially, when compared to prior SOTA, eficiency improvements are stark: ConvNeXt-XXL with RADAM requires >350 GFLOPs yet fails to outperform HiPerViT+ViT-B on FMD or GTOS-Mobile. Thus, HiPerViT not only advances accuracy but does so with orders-ofmagnitude lower cost.

![](images/8488b0940cdaefc16364331fdaa932ad715543977160f6ec19af7464e9888d09.jpg)

HiPerViT Performance Benchmark  
![](images/2f0265b0e52ddf2d96ea24e563106972c674f20505c1ba14105d5f7bec8fb54e.jpg)

![](images/faf30b0e0112708b8e475eb97bec5d84d1ccbad08e42322e5960ae6acfe9a538.jpg)  
Figure 3. Computational benchmarking of HiPerViT with diferent ViT backbones. Metrics include GFLOPs, inference latency, and peak GPU memory.

In summary, HiPerViT consistently outperforms vision-only baselines across datasets, with particularly large margins on fine-grained textures. Its design—multi-scale input, cross-stage fusion, and bilinear pooling—proves more efective and eficient than brute-force backbone scaling. The architecture delivers both SOTA accuracy and superior computational eficiency, underscoring its suitability for real-world texture recognition.

## 5.10. Real-world applications

While Sections 5.1–5.4 validate HiPerViT on established texture benchmarks, a natural question is whether the proposed extract→distill→interact paradigm transfers to real scientific settings, where (i) acquisition conditions are domain-specific, (ii) sample sizes may be limited, and (iii) specialized sensors (e.g., hyperspectral/multispectral systems) are not always available. To probe this, we evaluate HiPerViT on three datasets introduced in recent peer-reviewed works spanning biotechnology (plant-based pollution sensing), medicine (prostate histopathology), and agriculture (greenhouse tomato disease/pest monitoring). Importantly, in all three cases

![](images/cb59a14d8454f8704b2c497f113031fd416d3a8cc4c806aabfd5d003fe1817b1.jpg)  
Figure 4. HiPerViT accuracy (%) on six benchmarks across four ViT backbones. ViT-L/14 DINOv2 achieves the highest mean accuracy, but smaller backbones remain competitive.

HiPerViT is applied without domain-specific architectural tailoring and achieves performance that matches or surpasses the results reported in the original publications.

Biotechnology: We consider hyperspectral confocal microscopy images of Jacaranda caroba leaves exposed to diferent potassium fluoride pollutant levels, recently investigated in [30]. That work explicitly targets whole-image classification of high-resolution hyperspectral data under limited sample size and high dimensionality, proposing a hand-engineered complex-network descriptor (DNAS) with 32 features and reporting 92.6% accuracy. In contrast, HiPerViT operates on standard RGB inputs, i.e., it does not assume access to hyperspectral/multispectral instrumentation. To ensure a strictly fair methodological comparison, we evaluate HiPerViT using the exact same image cross-validation folds/splits as the DNAS paper, but process only the 3 corresponding visible RGB bands rather than the full 32-channel spectral cube. Despite this substantially weaker sensing modality, HiPerViT reaches 95.0% accuracy on the identical classification task. This result is practically significant: it suggests that the representational bias induced by hierarchical multi-scale extraction and explicit second-order interactions can compensate, at least in part, for the absence of rich spectral bands. We hypothesize that by encoding pairwise feature correlations (via the SRM), the model learns to exploit subtle color-spatial dependencies that act as “pseudo-spectral” signatures, efectively recovering discriminative information that would otherwise require hyperspectral instrumentation. This enables more accessible deployments in resource-limited monitoring scenarios.

Table 12. Real-world applications. HiPerViT compared with results reported in the original publications for three datasets spanning biotechnology, medicine, and agriculture.
<table><tr><td>Domain / Dataset</td><td>Best</td><td>HiPerViT</td><td>Notes</td></tr><tr><td>Biotech / J. caroba pollution</td><td>92.6%</td><td>95.0%</td><td>Baseline uses hyperspectral/multispectral data; HiPerViT uses RGB.</td></tr><tr><td>Medicine / SICAPv2</td><td>85.13% / 76.22%</td><td>87.1%</td><td>Histopathology grading-related classification.</td></tr><tr><td>Agriculture / TLID</td><td>86.56%</td><td>87.5%</td><td>Uses non-patch TLID images, not PTLID patches.</td></tr></table>

Sources: J. caroba pollution [30]; SICAPv2 [35, 22]; TLID [40].

Medicine: Next, we evaluate on SICAPv2 prostate biopsy images, a challenging histology benchmark used in [35] and also adopted in [22]. The clinical motivation is strong: Gleason grading and the identification of grade-4 patterns (including cribriform morphology) are time-consuming and subject to inter-observer variability, motivating robust automated decision support. Reported results on SICAPv2 include 76.22% accuracy in [35] and 85.13% accuracy in [22]. To guarantee no data leakage and maintain parity, we strictly follow the oficial patient-level train/test cross-validation splits provided by the dataset authors. Furthermore, we operate directly on the authors’ pre-extracted histological tiles (patches) and apply standard ImageNet RGB normalization without any domain-specific stain normalization routines (e.g., Macenko). Under this rigorous evaluation setup, HiPerViT achieves 87.1% accuracy, exceeding both reported baselines. Beyond the numeric improvement, this experiment highlights that HiPerViT can capture diagnostically relevant micro-architectural patterns in H&E tissue, a domain where discriminative cues often manifest as subtle textural organization across multiple scales.

Agriculture: Finally, we test HiPerViT on the tomato leaf disease dataset introduced in [40], which reflects region-specific disease occurrence under greenhouse cultivation. That study reports 90.48% accuracy when operating on a patch-based variant (PTLID), and 86.56% accuracy on the original non-patch images (TLID). Here we focus on the non-patch setting to assess performance without the additional supervision and engineering implicit in patch extraction. Crucially, we utilize the exact same training and testing data splits as the original paper to ensure a direct comparison. In this configuration, HiPerViT attains 87.5% accuracy, surpassing the original non-patch baseline. This is notable because greenhouse imagery often exhibits complex nuisance factors (illumination variability, cluttered backgrounds, heterogeneous symptom stages), making generalization beyond curated datasets particularly challenging.

Summary and implications. Table 12 summarizes these results. Across three distinct realworld domains, HiPerViT consistently improves upon the best reported numbers from the corresponding sources, including scenarios where the published baselines rely on specialized sensing (hyperspectral/multispectral) or task-specific preprocessing (patch pipelines). Taken together, these findings support a central claim of this work: explicitly structuring computation into (i) multi-scale feature preservation, (ii) attention-based token distillation, and (iii) secondorder interaction modeling yields a representation that is not only competitive on texture benchmarks, but also deployable and robust in practical scientific applications where data are limited and acquisition constraints are non-ideal.

Summary and implications: Table 12 consolidates the outcomes across biotechnology, medicine, and agriculture. Despite the pronounced domain shift, from confocal hyperspectral microscopy to H&E whole-slide histology and greenhouse leaf imagery, HiPerViT remains consistently competitive and, in these cases, exceeds the best numbers reported in the corresponding sources. Two aspects are particularly noteworthy. First, HiPerViT improves over pipelines that rely on stronger sensing modalities (hyperspectral/multispectral) or additional task-specific engineering (e.g., patch-based supervision), indicating that the proposed representation is not merely exploiting dataset idiosyncrasies but capturing transferable visual structure. Second, the gains are achieved without domain-specific architectural modifications, supporting the claim that the inductive bias introduced by our hierarchical multi-scale extraction, attention-based token distillation, and explicit second-order interactions is broadly applicable.

Closing remark: Taken together, these experiments strengthen the central message of this paper: second-order statistical structure is a foundational visual primitive, and explicitly reintroducing it into SSL transformer representations via Statistical Token Injection yields gains that generalize beyond canonical texture benchmarks. Crucially, this design ofers a practical accuracy–eficiency trade-of: by distilling informative multi-scale tokens and encoding their structured interactions, HiPerViT delivers robust performance in heterogeneous, real-world scientific pipelines while operating under realistic constraints on data volume and sensing hardware.

## 6. Discussion

HiPerViT establishes consistent improvements over prior state-of-the-art models across five of six benchmarks (Table 1), demonstrating the practical and conceptual value of Statistical Token Injection within a multi-scale extraction and latent distillation framework. Its multiscale token hierarchy preserves both local micro-patterns and global context, while the compact bilinear head eficiently encodes pairwise feature correlations. The Perceiver-style latent bottleneck integrates information asymmetrically, condensing tens of thousands of tokens into a manageable set of latent queries. Collectively, these mechanisms, unified through the STI principle, allow HiPerViT to deliver robust recognition without relying on excessively large backbone architectures.

The gains are particularly pronounced on GTOS-Mobile (+10.48 pp) and 1200Tex (+10.27 pp), highlighting that the model excels on high-resolution, domain-specific textures where second-order statistics carry strong discriminative signal. On more challenging and unconstrained datasets, such as DTD, HiPerViT maintains a strong +3.05 pp improvement over a ViT-L/14 baseline, demonstrating that eficiency and generalization need not be mutually exclusive. Across all datasets, smaller ViT-B/S backbones paired with the HiPerViT head achieve competitive accuracy at substantially reduced FLOPs and memory usage, confirming that our approach balances state-of-the-art performance with practical deployment constraints.

This cross-dataset success suggests that injecting explicit second-order statistical structure into SSL transformer representations constitutes a broadly applicable strategy. Rather than favoring one architectural family, HiPerViT selectively leverages each paradigm where it is strongest: early-stage token hierarchies for local patterns, transformer bottlenecks for global integration, and compact bilinear statistics for feature interactions. This combination yields a resilient architecture that generalizes across diverse visual domains, acquisition conditions, and resolutions.

## 6.1. Dual Invariance as Architectural Robustness

HiPerViT exhibits two complementary invariances that jointly characterize the robustness of its fusion mechanism. First, depth-invariance (Sec. 5.6): the fusion head extracts equivalent classification performance from SRM vectors of vastly diferent intrinsic quality, indicating that cross-order self-attention compensates for weak second-order signals by leveraging the complementary spatial token stream. Second, topology-invariance (Sec. 5.5): interact→distill, distill→interact, and late fusion all achieve statistically indistinguishable accuracy on DTD, establishing that the discriminative gain originates from the SRM statistical prior rather than from a brittle wiring choice. Together, these properties make HiPerViT robust to hyperparameter choices about both where to tap second-order statistics from the backbone and how to integrate them into the fusion pipeline.

## 6.2. Limitations and Future Work

Despite these advances, the current study exposes clear directions for further research:

• Counterfactual controls: Our ablations establish the importance of the SRM statistical token, but do not yet include counterfactual baselines (e.g., random tokens, shufled SRM, or learned MLP tokens of equivalent capacity). Adding these controls would provide stronger causal evidence for STI.

• Non-texture domains: While HiPerViT’s gains are validated on texture benchmarks and three applied scientific tasks, broader evaluation on histopathology, remote sensing, and fine-grained biological classification would strengthen the claim of domain generality.

• Temporal and dynamic textures: The current evaluation is limited to static images. Extending the STI framework to spatio-temporal domains could enable recognition of dynamic textures, where temporal consistency is critical.

• Statistical rigor: Key invariance claims are based on 3 seeds with limited partitions. Expanding to ≥5 seeds with 95% confidence intervals would increase defensibility.

• Interaction regimes: Topology ablations show invariance on DTD; identifying regimes where cross-order attention outperforms late fusion (e.g., under heavy domain shift or extreme data scarcity) would strengthen the conditional-advantage narrative.

## 6.3. When Does STI Help?

Our experiments suggest that STI is most beneficial when three conditions are jointly met: (i) the task’s discriminative signal resides in pairwise feature correlations rather than first-order averages; (ii) the backbone’s pre-training objective does not explicitly encode second-order structure (as is the case for standard SSL-ViTs); and (iii) supervision is limited, amplifying the value of a structured inductive bias (Sec. 5.4.1). Conversely, STI does not distort representations when these conditions are absent: cue-conflict evaluation on ImageNet-100 (Sec. 5.7) confirms that the statistical token’s influence is conditioned on task relevance. We emphasize this as a conditional inductive bias: STI enriches the representation without overriding the backbone’s dominant priors.

Overall, HiPerViT demonstrates that carefully combining hierarchical, statistical, and attentionbased mechanisms can achieve state-of-the-art performance while remaining computationally eficient, providing a strong blueprint for extending SSL transformer representations with explicit statistical structure.

## 7. Conclusion

We introduced Statistical Token Injection (STI), a lightweight mechanism for reintroducing second-order statistical structure into self-supervised Vision Transformer representations. STI embeds compact bilinear descriptors as first-class tokens within the transformer’s self-attention, enabling cross-order interaction between spatial and statistical features at negligible parameter cost. We validated STI through HiPerViT, a modular architecture pairing STI with multi-scale extraction and Perceiver-style latent distillation. Evaluated on six texture benchmarks and three applied scientific domains, HiPerViT achieves state-of-the-art or competitive vision-only performance across the evaluated benchmarks, with gains of up to +10.48 pp.

Our analysis highlights four key insights. First, transformer scale alone does not substantially drive performance within the evaluated backbone family: the accuracy gap across ViT-S/B/L backbones remains under 2 pp once STI and the Perceiver head are integrated. Second, HiPer-ViT achieves a favorable accuracy–eficiency trade-of, maintaining SOTA-level performance while substantially reducing FLOPs relative to higher-capacity backbone configurations. Third, mechanistic analysis shows that final recognition performance is remarkably insensitive to the backbone depth from which the second-order statistics are extracted. Fourth, controlled topology ablations indicate that the benefit is associated more strongly with the availability of the second-order representation than with a particular interaction–distillation ordering. Together, these results support the view that making second-order statistics explicitly available to the representation is more important than the particular wiring used to integrate them.

These findings suggest a broader principle for the machine intelligence community: explicit statistical inductive biases can be incorporated into SSL transformer representations through lightweight integration mechanisms, providing consistent benefits in tasks where microstructural information is discriminative, without requiring domain-specific architectures, multimodal pre-training, or excessive compute.

## References

[1] D. Casanova, J. J. M. de Mesquita Sá Junior, and O. M. Bruno. Plant leaf identification using gabor wavelets. International Journal of Imaging Systems and Technology, 19(4):236–243, 2009. Introduces the 1200Tex texture dataset used in subsequent studies.

[2] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describable Textures Dataset (DTD). https://www.robots.ox.ac.uk/\~vgg/data/dtd/, 2014. 5,640 images in 47 de scribable texture categories.

[3] Mircea Cimpoi, Subhransu Maji, and Andrea Vedaldi. Deep filter banks for texture recognition and segmentation. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3828–3836, 2015.

[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021.

[5] Ricardo T. Fares, Luan B. Guerra, and Lucas C. Ribas. Exploring Local Graphs via Random Encoding for Texture Representation Learning. In Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 3: VISAPP, pages 200–209. SciTePress, 2025.

[6] João Batista Florindo, Young-Sup Lee, Kyungkoo Jun, Gwanggil Jeon, and Marcelo K. Albertini. Vis-GraphNet: a complex network interpretation of convolutional neural features. CoRR, abs/2108.12490, 2021.

[7] Ilan Fogel and Dov Sagi. Gabor filters as texture discriminator. Biological Cybernetics, 61:103–113, 1989.

[8] Jonathan Freeman, Corey Ziemba, David Heeger, and et al. A functional and perceptual signature of the second visual area in primates. Nature Neuroscience, 16:974–981, 2013.

[9] Yang Gao, Oscar Beijbom, Ning Zhang, and Trevor Darrell. Compact bilinear pooling, 2016.

[10] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In International Conference on Learning Representations (ICLR), 2019.

[11] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness, 2022.

[12] Robert M. Haralick, K. Shanmugam, and Its’Hak Dinstein. Textural features for image classification. IEEE Transactions on Systems, Man, and Cybernetics, SMC-3(6):610–621, 1973.

[13] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization, 2017.

[14] David H. Hubel and Torsten N. Wiesel. Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex. The Journal of Physiology, 160(1):106–154, 1962.

[15] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. Perceiver: General perception with iterative attention, 2021.

[16] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geofrey Hinton. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97 of Proceedings of Machine Learning Research, pages 3519–3529. PMLR, 2019.

[17] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[18] Peihua Li, Jiangtao Xie, Qilong Wang, and Zilin Gao. Towards faster training of global covariance pooling networks by iterative matrix square root normalization. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 947–955, 2018.

[19] Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji. Bilinear cnn models for fine-grained visual recognition. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), December 2015.

[20] Yangqi Liu, Hao Dong, Guodong Wang, and Chenglizhao Chen. Dual-branch network based on transformer for texture recognition. Digital Signal Processing, 153:104612, 2024.

[21] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows, 2021.

[22] Amin Malekmohammadi, Ali Badiezadeh, Seyed Mostafa Mirhassani, Parisa Gifani, and Majid Vafaeezadeh. Classification of gleason grading in prostate cancer histopathology images using deep learning techniques: Yolo, vision transformers, and vision mamba, 2024. arXiv preprint.

[23] P. Mallikarjuna, Alireza Tavakoli Targhi, Mario Fritz, Eric Hayman, Barbara Caputo, and Jan-Olof Eklundh. THE KTH-TIPS2 database. Technical report, Computational Vision and Active Perception Laboratory (CVAP), KTH Royal Institute of Technology, June 2006. Available at: https://www.csc.kth.se/ cvap/databases/kth-tips/kth-tips2.pdf (accessed 12 July 2025).

[24] Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Intriguing properties of vision transformers. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pages 23296–23308, 2021.

[25] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024.

[26] Guillermo Ortiz-Jimenez, Alessandro Favero, and Pascal Frossard. Task arithmetic in the tangent space: Improved editing of pre-trained models, 2023.

[27] Javier Portilla and Eero P. Simoncelli. A parametric texture model based on joint statistics of complex wavelet coeficients. International Journal of Computer Vision, 40:49–70, 2000.

[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.

[29] Maximilian Riesenhuber and Tomaso Poggio. Hierarchical models of object recognition in cortex. Nature Neuroscience, 2(11):1019–1025, 1999.

[30] Leonardo Scabini, Kallil M Zielinski, Marilia Fernandes, Ricardo T Fares, Lucas C Ribas, Rosana M Kolb, and Odemir M Bruno. Directed network of angular similarity (dnas): A complex network approach fo high-resolution hyperspectral image classification. Computers in Biology and Medicine, 2026. Published online Feb 3, 2026.

[31] Leonardo Scabini, Kallil M. Zielinski, Emir Konuk, Ricardo T. Fares, Lucas C. Ribas, Kevin Smith, and Odemir M. Bruno. Vortex: Challenging cnns at texture recognition by using vision transformers with orderless and randomized token encodings, 2025.

[32] Leonardo Scabini, Kallil M. Zielinski, Lucas C. Ribas, Wesley N. Gonçalves, Bernard De Baets, and Odemir M. Bruno. Radam: Texture recognition through randomized aggregated encoding of deep activation maps, 2023.

[33] Scientific Computing Group, IFSC–USP. USPtex: A daily-life texture dataset. https://scg.ifsc.usp.br/ dataset/USPtex.php, 2012. Accessed: 12 July 2025.

[34] Lavanya Sharan, Ruth Rosenholtz, and Edward H. Adelson. Flickr Material Database (FMD). https: //people.csail.mit.edu/lavanya/fmd.html, 2014. 10 material categories, 100 images each.

[35] J. Silva-Rodríguez, A. Colomer, M. A. Sales, R. Molina, and V. Naranjo. Going deeper through the gleason scoring scale: An automatic end-to-end system for histology prostate grading and cribriform pattern detection. Computer Methods and Programs in Biomedicine, 195:105637, October 2020.

[36] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-eficient image transformers & distillation through attention, 2021.

[37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

[38] Tan Yu, Xiaoyun Li, and Ping Li. Fast and compact bilinear pooling by shifted random maclaurin. In Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21), volume 35, pages 3243–3251, Feb 2021.

[40] Grasielli B. Zimmermann, Marcelo E. Pellenz, Yandre M. G. Costa, and Alceu de S. Britto Jr. Enhancing disease and pest detection in greenhouse tomato cultivation using advanced machine learning on new dataset of images. Journal of the Brazilian Computer Society, 2025.

[39] Semir Zeki. A Vision of the Brain. Blackwell Scientific Publications, Oxford, 1993.