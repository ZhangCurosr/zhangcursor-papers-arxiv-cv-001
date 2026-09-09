Existing T2M methods can be broadly categorized into two lines of research. The first line consists of conventional T2M models, which focus on strengthening the generative backbone itself. Representative approaches include generative adversarial networks (GANs) [2], [3], [4], [5], [6], variational autoencoders (VAEs) [7], diffusion models [8], visuallanguage models [9], [10], motion language models [11], and masked generative models [12], [13], [14], [15]. In particular, masked generative models such as MMM [12] and Moproduction [1], virtual reality, and robotics. By synthesizing realistic and diverse human motions, these methods aim to significantly reduce the cost of manual animation while improving the efficiency and flexibility of content creation. Among various paradigms, text-to-motion (T2M) generation has emerged as a particularly intuitive setting, where a natural language description is directly mapped to a sequence of human joint movements.

![](images/77c26cde01cd538e872696617f9191cda177e8cbdb95873108f953e0141832a8.jpg)  
Fig. 1: ReMoMask vs ReMoMask-2. (a) ReMoMask retrieves motion clips from a database outside the generator, so every reference crosses the boundary between the two regions. (b) ReMoMask-2 rebuilds the database from the generator’s own latents, so store, references and mask tokens share one substrate and nothing crosses.

## 1 INTRODUCTION

![](images/c78481170d4ca773caa9beac2036ebb0a32cc14eb93b85d69d02ae127391e016.jpg)

# ReMoMask-2: Latent Retrieval-Augmented Masked Motion Generation

Yiran Wang<sup>∗</sup>, Zeyu Zhang<sup>∗†</sup>, Ling Shao, Fellow, IEEE, Hao Tang<sup>‡</sup>

Abstract—Text-to-motion (T2M) generation maps a natural language description to a sequence of human joint movements, offering an intuitive interface for producing human motion in gaming, film production, virtual reality, and robotics. Retrieval-Augmented Text-to-Motion (RAG-T2M) models improve over conventional T2M approaches, particularly on uncommon and complex textual descriptions, by conditioning generation on motion-text pairs retrieved from an external database. However, existing RAG-T2M models remain limited by two challenges. First, retrieval and fusion are structurally inconsistent with motion topology: coarse-grained text-motion retrieval overlooks the hierarchical, part-level structure of human motion, and retrieved evidence is fused by mechanisms that ignore the spatial-temporal structure of the motion latent. Second, retrieved evidence typically resides in a contrastive semantic space learned separately from the latents the generator manipulates, leaving a representation gap between what is retrieved and what is generated. To address the first challenge, we present ReMoMask, a structure-aware RAG framework that couples Hierarchical Bidirectiona Momentum (HBM) contrastive learning, which employs dual objectives to jointly align global motion semantics and fine-grained part-level features with text; Semantic Spatial-Temporal Attention (SSTA), a topology-aware fusion module that integrates retrieved knowledge via an asymmetric attention mechanism; and Topology Structured Masking (TSM), a training strategy that adaptively masks motion tokens based on semantic relevance, forcing the model to learn robust part-level grounding. To address the second challenge, we further present ReMoMask-2, which rebuilds the retrieval database directly within the generator’s own pre-quantization latent space and aligns text queries to it via a lightweight projector distilled from the retriever, so that the generator consumes the semantic content of the retrieved motion rather than merely registering its presence. Extensive experiments on HumanML3D, KIT-ML, and SnapMoGen demonstrate that our retriever achieves state-of-the-art text-to-motion retrieval, and that ReMoMask-2 attains state-ofthe-art generation fidelity among retrieval-augmented approaches and the lowest FID among all compared methods on KIT-ML and SnapMoGen, while its single mask-transformer stage, without any residual-refinement network, surpasses the full two-stage pipeline of ReMoMask and delivers the fastest inference among compared systems. Code: https://github.com/AIGeeksGroup/ReMoMask-2. Website: https://aigeeksgroup.github.io/ReMoMask-2.

Index Terms—Text-to-Motion Generation, Retrieval-Augmented Generation, Masked Motion Modeling, Contrastive Learning

Mask [13] discretize motion sequences into tokenized representations and perform masked token prediction, achieving high-fidelity and temporally coherent motion synthesis.

TABLE 1: Comparison of different architecture designs.
<table><tr><td rowspan="2">Method</td><td colspan="4">Retrieval</td><td colspan="2">Generation</td></tr><tr><td>Global Align</td><td>Part Align</td><td>Momentum</td><td>Space</td><td>Fusion</td><td>Latent</td></tr><tr><td>MDM [19]</td><td></td><td></td><td></td><td></td><td>concat</td><td>1D</td></tr><tr><td>T2M-GPT [20]</td><td></td><td></td><td></td><td></td><td>concat</td><td>1D</td></tr><tr><td>MoMask [13]</td><td></td><td></td><td></td><td></td><td>concat</td><td>1D</td></tr><tr><td>MARDM [21]</td><td></td><td></td><td></td><td></td><td>concat</td><td>1D</td></tr><tr><td>LaMP [22]</td><td></td><td>一</td><td>2</td><td></td><td>cross-attn</td><td>1D</td></tr><tr><td>TMR [23]</td><td>√</td><td>×</td><td>×</td><td>Semantic</td><td>concat</td><td>1D</td></tr><tr><td>MoRAG [24]</td><td>×</td><td>×</td><td>×</td><td>Semantic</td><td>cross-attn</td><td>1D</td></tr><tr><td>ReMoGPT [18]</td><td>√</td><td>×</td><td>×</td><td>Semantic</td><td>concat</td><td>1D</td></tr><tr><td>ReMoDiffuse [16]</td><td>√</td><td>×</td><td>×</td><td>Semantic</td><td>cross-attn</td><td>1D</td></tr><tr><td>ReMoMask</td><td>√</td><td>√</td><td>√</td><td>Semantic</td><td>SSTA</td><td>2D</td></tr><tr><td>ReMoMask-2</td><td>V</td><td>V</td><td>v</td><td>Latent</td><td>SSTA</td><td>2D</td></tr></table>

The second line of work, known as retrieval-augmented T2M (RAG-T2M), enhances generation by retrieving relevant motion–text pairs from an external database and injecting the retrieved evidence into the generative process. By conditioning generation on exemplar motions, RAG-based approaches improve robustness to uncommon or complex textual inputs. Representative methods include ReMoDiffuse [16], which performs retrieval via text–text similarity using CLIP [17], and ReMoGPT [18], which adopts a crossmodal text–motion retriever.

Despite their promising performance, existing RAG-T2M methods implicitly treat retrieval and fusion as independent modules and largely ignore the structural consistency between retrieval alignment and motion representation. Through careful analysis, we identify two fundamental design axes of this structural consistency in retrievalaugmented motion generation:

(1) Structural Granularity of Alignment. As shown in Table 1, most text–motion retrieval methods rely on contrastive learning to align global motion embeddings with text [23], [25]. However, human motion is inherently hierarchical, organized by skeletal topology and body-part dependencies. Purely global alignment overlooks fine-grained part-level semantics (e.g., left/right limbs or asymmetric actions), limiting retrieval discriminability. Although some works [18], [26] encode part-level motion features, these representations are typically aggregated without part-level cross-modal alignment, weakening structured correspondence.

(2) Structural Compatibility of Fusion. As depicted in Table 1, existing RAG-T2M methods often adopt simple concatenation or vanilla cross-attention, without systematically examining how motion latent structure (e.g., 1D vs. 2D spatial–temporal tokens) interacts with fusion mechanisms. This mismatch between retrieved information and motion representation can limit generation quality.

These observations suggest that performance improvements in RAG-T2M hinge on whether retrieval alignment and fusion design are structurally consistent with motion topology, beyond the strength of individual modules. Taken together, these two axes constitute the first challenge we address in this article.

Beyond these two axes, a second challenge remains, along a third design axis orthogonal to the structural ones: in existing RAG-T2M methods [16], [18], [24], the retrieved evidence lives in a contrastive semantic space learned separately from the latents the generator manipulates, so retrieved motions must cross a representation gap before they can guide generation, as reflected by the Space column of Table 1. This gap persists even when retrieval alignment and fusion are both made structurally consistent with motion topology.

We propose ReMoMask, a retrieval-augmented masked generative framework for text-to-motion generation. To address alignment granularity, we introduce Hierarchical Bidirectional Momentum (HBM) alignment, a structured contrastive learning framework that jointly supervises instance-level (global) and part-level bidirectional text–motion correspondence under a momentum-based paradigm. To ensure structural compatibility during semantic conditioning, we conduct a systematic study (provided in Section 3) on motion latent representations and fusion strategies. Our analysis reveals that preserving motion as 2D spatial–temporal tokens, rather than flattening them into 1D sequences, significantly improves semantic injection and generation stability. Motivated by this observation, we design Semantic Spatial-Temporal Attention (SSTA), an attention mechanism tailored to inject retrieved motion semantics into the generative backbone. Meanwhile, to further strengthen structural modeling within this 2D formulation, we adopt a Topology Structured Masking (TSM) strategy, allowing the network to learn richer topological dependencies across body parts and time.

In this paper, we extend our ECCV 2026 conference framework [27], which inherits this gap, along the third axis of representation consistency and present ReMoMask-2 (Fig. 1), which performs retrieval directly in the generator’s own pre-quantization latent space z<sub>e</sub>. Because retrieval keys are then produced by the same frozen encoder that supplies the generative latents, no cross-space translation has to be learned, and retrieved neighbors are directly comparable to the latents the generator predicts.

Our contributions are summarized as follows:

• We present ReMoMask, first introduced in our conference version [27], a retrieval-augmented masked generative framework that enforces structural consistency along the two structural axes identified above: alignment granularity and fusion compatibility. It couples HBM, a hierarchical bidirectional momentum alignment framework that explicitly models both global and part-level text–motion correspondence for fine-grained semantic grounding, with SSTA, a topology-aware spatial–temporal attention mechanism built upon structured 2D motion tokens, and a TSM masking strategy that strengthens topological dependencies across body parts and time.

• We present ReMoMask-2, which extends ReMoMask along the third design axis, representation consistency between the retrieval space and the generative latent space: it rebuilds the retrieval database in the frozen RVQ-VAE’s pre-quantization latent space z<sub>e</sub> and aligns text queries to it with a lightweight projector distilled from the HBM retriever, closing the gap between retrieved evidence and the generative substrate.

• Extensive experiments on HumanML3D, KIT-ML, and SnapMoGen show that our retriever attains state-ofthe-art text-to-motion retrieval on all three benchmarks (R@1 18.49 on HumanML3D, against 11.00 for the next best), and that ReMoMask-2 is the strongest retrievalaugmented generator on all three, with the lowest FID among all compared methods on KIT-ML (0.138) and SnapMoGen (13.174), surpassing the two-stage Mo-Mask pipeline on HumanML3D (FID 0.042 versus 0.046, Top-1 0.528 versus 0.521) with a single generative stage.

As shown in Fig. 1, ReMoMask-2 extends our ReMo-Mask framework from retrieval in a separate semantic space to retrieval inside the generator’s own latent space, which in turn makes a single generative stage sufficient. A preliminary version of this work was accepted to ECCV 2026 [27]. That conference version generates motion in two stages, a masked transformer followed by a residualrefinement transformer. The extension consists of two coupled changes: the retrieval database is rebuilt in the generator’s pre-quantization latent space $z _ { e }$ and text queries are aligned to it with a lightweight projector distilled from the conference-version retriever, so that retrieved evidence and the generated latents share a single representation; and the residual-refinement transformer is removed, in line with recent single-stage designs [12], [14], [26], since the retrieval-conditioned mask transformer already supplies the fine-grained correction that a residual stage is designed to provide. Our experiments bear this out: under a unified 20-repeat protocol on which we reproduce ReMoMask, the single-stage, mask-only ReMoMask-2 surpasses the full mask-plus-residual conference version, achieving a 65.9% and 75.4% improvement in FID on HumanML3D (0.123 → 0.042) and KIT-ML $( 0 . 5 6 2  0 . 1 3 8 )$ respectively, while reducing per-sample inference time from roughly 0.14s to 0.05s, a saving to which the lighter retrieval front end of the latent-aligned design contributes alongside the removed stage; reattaching the residual-refinement transformer not only adds inference cost but actively hurts fidelity, raising FID from 0.042 to 0.068. Thus, ReMoMask-2 treats retrieval as a native part of the generative representation rather than as external evidence that must be translated into it.

## 2 RELATED WORK

## 2.1 Text-to-Motion Generation

Text-to-motion (T2M) generation aims to synthesize realistic human motion sequences conditioned on natural language descriptions. Early approaches explored adversarial learning to establish text–motion correspondence. With the introduction of vector quantization, TM2T [28] and T2M-GPT [20] discretized motion sequences and leveraged autoregressive transformers for semantic control. However, autoregressive decoding often suffers from error accumulation.

Recent advances focus on improving motion representation and generation paradigms. MoMask [13] introduces hierarchical residual quantization and masked bidirectional transformers for parallel decoding, achieving strong performance on HumanML3D. Diffusion-based methods [11], [29] further enhance generation quality via non-autoregressive denoising processes. MotionGPT [11] unifies multiple generation tasks under a discrete autoregressive framework.

Beyond backbone improvements, several works investigate fine-grained motion structure modeling. ParCo [26] discretizes whole-body motion into part-level components (limbs, backbone, root) to establish structured priors.

TABLE 2: Comparison of different latent structures and fusion strategies. These pilot results follow the conference evaluation protocol on a MoMask backbone; results under the main protocol used throughout this paper appear in Table 4.
<table><tr><td rowspan=2 colspan=1>Generator</td><td rowspan=2 colspan=1>Latent</td><td rowspan=2 colspan=1>Fusion</td><td rowspan=2 colspan=1>FID↓</td><td></td></tr><tr><td rowspan=1 colspan=1>Top1↑</td></tr><tr><td rowspan=2 colspan=1>MoMask</td><td rowspan=1 colspan=1>1D</td><td rowspan=1 colspan=1>concatcrossAttn</td><td rowspan=1 colspan=1> $0 . 0 5 7 ^ { \pm . 0 1 3 }$  $\underline { { 0 . 0 4 3 ^ { \pm . 0 0 4 } } }$ </td><td rowspan=1 colspan=1> $0 . 5 1 1 ^ { \pm . 0 0 3 }$  $\underline { { 0 . 5 2 5 ^ { \pm . 0 0 3 } } }$ </td></tr><tr><td rowspan=1 colspan=1>2D</td><td rowspan=1 colspan=1>concatcrossAttn</td><td rowspan=1 colspan=1> $0 . 0 4 9 ^ { \pm . 0 0 7 }$  $\mathbf { 0 . 0 3 6 ^ { \pm . 0 0 5 } }$ </td><td rowspan=1 colspan=1> $0 . 5 1 8 ^ { \pm . 0 0 3 }$  $\mathbf { 0 . 5 3 6 ^ { \pm . 0 0 2 } }$ </td></tr></table>

## 2.2 Retrieval-Augmented Text-to-Motion

Retrieval-augmented generation (RAG) has been extended beyond NLP to multimodal domains, including motion generation [16], [18], [24], [30]. In retrieval-augmented T2M (RAG-T2M), relevant motion–text pairs are retrieved from an external database and injected into the generative backbone to improve robustness under complex or rare textual conditions.

Existing approaches typically rely on global contrastive alignment between text and motion embeddings [23], [25]. For instance, ReMoDiffuse [16] performs retrieval based on text–text similarity to indirectly guide generation, while ReMoGPT [18] introduces part-aware encoders for crossmodal retrieval. However, these methods often aggregate part features without explicit bidirectional cross-modal supervision, leaving fine-grained structural correspondence between text descriptions and specific body parts underexplored. Furthermore, regarding the integration of retrieved information, prior works commonly adopt feature concatenation or standard cross-attention [16], [18] without systematically considering the compatibility between the retrieved evidence and the underlying motion latent topology. The impact of motion representation structures on fusion effectiveness remains largely uninvestigated.

## 3 DESIGN PRINCIPLES OF RETRIEVAL-AUGMENTED MOTION GENERATION

While retrieval-augmented T2M methods show promise, the interplay between motion representation and fusion strategy remains underexplored. We investigate two key design axes: (1) motion latent topology (1D sequence vs. 2D spatial–temporal grid) and (2) fusion mechanism (concatenation vs. cross-attention).

Using MoMask [13] as a backbone generator, we systematically vary these axes while fixing other components. We construct both 1D $( z _ { 1 d } )$ and 2D $( z _ { 2 d } )$ latents, integrating retrieved embeddings $( R _ { m } , R _ { t }$ from ReMoDiffuse [16]) via concatenation or cross-attention. All evaluations are performed on the HumanML3D benchmark.

As summarized in Table 2, the optimal configuration combines 2D representation with cross-attention (0.036 FID), the design that directly motivates our SSTA module, which fuses retrieved semantics with motion tokens via structureaware cross-attention over a 2D latent grid.

![](images/1bd0c999c70d278f59903e157130fc19586f939a3c8aa8b40482d344227e1549.jpg)  
Fig. 2: Qualitative comparisons. The darker colors indicate the later in time. The motions generated by our method closely align with the descriptions, outperforming others that exhibit degraded motions or improper semantics.

## 4 METHODOLOGY

## 4.1 Overview

As illustrated in Fig. 1, our framework follows a retrievalaugmented generation paradigm. We first employ Hierarchical Bidirectional Momentum (HBM) learning to align text and motion at both instance and part levels, constructing a hierarchically organized embedding space for accurate retrieval. During training, we utilize Topology Structured Masking (TSM), a structure-aware masked modeling objective that adaptively modulates masking probabilities based on hierarchical relevance to strengthen part-level grounding. Given an input text prompt $x ,$ relevant motion-text pairs are retrieved from this structured space to obtain reference embeddings $( R _ { m } , R _ { t } )$ , which are then injected into the generator via our proposed Semantic Spatial–Temporal Attention (SSTA) to enable topology-aware semantic conditioning over 2D motion tokens.

In this pipeline, retrieval operates in the contrastive semantic space constructed by HBM, whereas generation operates on the latents of a 2D RVQ-VAE, two representations learned under disjoint objectives. Section 4.6 presents the extension that defines ReMoMask-2: we migrate retrieval into the generator’s own pre-quantization latent space, so that retrieved evidence arrives, by construction, in the representation the generator natively consumes.

## 4.2 Hierarchical Bidirectional Momentum

Hierarchical Motion Decomposition. As shown in Fig. 3a, we decompose motion m into $K = 6$ semantic parts (e.g., arms, legs, backbone, root), denoted as $m ^ { k }$ . Given a batch

$\{ ( x _ { i } , m _ { i } ) \} _ { i = 1 } ^ { B } ,$ we encode text, global motion, and part motions into a shared space:

$$
t _ { i } = f _ { T } ( x _ { i } ) , \quad g _ { i } = f _ { G } ( m _ { i } ) , \quad p _ { i , k } = f _ { P } ^ { ( k ) } ( m _ { i } ^ { k } ) ,\tag{1}
$$

where $f _ { T } , f _ { G } ,$ and $f _ { P } ^ { ( k ) }$ are the respective encoders.

Momentum-Based Structured Contrast. To stabilize training and enlarge the negative set, we maintain momentum encoders with parameters updated via exponential moving average:

$$
\theta ^ { - }  \mu \theta ^ { - } + ( 1 - \mu ) \theta ,\tag{2}
$$

where $\theta$ and $\theta ^ { - }$ denote online and momentum parameters. The resulting momentum embeddings $( t _ { i } ^ { - } , g _ { i } ^ { - } , \bar { p } _ { i , k } ^ { - } )$ are stored in a queue Q serving as negative keys.

Contrastive Objective. Using cosine similarity sim(·, ·) and temperature $\tau ,$ we define the positive term $\mathcal { P } ( q , k ) \ \stackrel {  } { = } \ \exp ( \sin ( q , k ) / \tau )$ and negative sum $\begin{array} { l l } { { \mathcal { N } ( q ) } } & { { = } } \end{array}$ $\begin{array} { r } { \sum _ { k ^ { - } \in \mathcal { Q } } \exp ( \sin ( q , k ^ { - } ) / \tau ) } \end{array}$ . The InfoNCE loss is:

$$
\mathrm { I n f o N C E } ( q , k ) = - \log { \frac { { \mathcal { P } } ( q , k ) } { { \mathcal { P } } ( q , k ) + { \mathcal { N } } ( q ) } } .\tag{3}
$$

Bidirectional Alignment. We enforce bidirectional alignment at the instance level:

$$
\mathcal { L } _ { \mathrm { I n s t } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \Big ( \mathrm { I n f o N C E } ( t _ { i } , g _ { i } ) + \mathrm { I n f o N C E } ( g _ { i } , t _ { i } ) \Big ) ,\tag{4}
$$

and further align text with each part representation to preserve fine-grained semantics:

$$
\mathcal { L } _ { \mathrm { P a r t } } = \frac { 1 } { B K } \sum _ { i = 1 } ^ { B } \sum _ { k = 1 } ^ { K } \Big ( \mathrm { I n f o N C E } ( t _ { i } , p _ { i , k } ) + \mathrm { I n f o N C E } ( p _ { i , k } , t _ { i } ) \Big ) .\tag{5}
$$

![](images/b9ea49cf14a150e7444484b209f0cdf74d79e9e790339d1e9d2c105b0fc82d3d.jpg)  
Fig. 3: Framework of ReMoMask-2. (a) The retrieval database is built offline by the frozen RVQ-VAE encoder $\varepsilon \colon$ each motion becomes a pre-quantisation latent grid, pooled and $\ell _ { 2 }$ -normalised into a key. (b) A query projector $\phi ,$ distilled from the frozen HBM retriever, maps text into that same key space, so cosine retrieval returns evidence already expressed in the generator’s representation. (c) Topology Structured Masking (TSM) and (d) Semantic Spatial–Temporal Attention (SSTA) are inherited unchanged from ReMoMask, except that $R _ { t }$ may now enter the Value pathway.

![](images/5b1f77ec0a71db02aae41ad763a21496ba370b625869bb8f355dd49f0982d41a.jpg)  
Fig. 4: Part-level motion encoding for hierarchical alignment. The full-body motion sequence is first divided into multiple body parts according to the skeletal structure. Each part motion is encoded independently by a shared part encoder to obtain part-level features, which are then concatenated and aggregated by an instance encoder to produce a global motion representation. This hierarchical encoding enables fine-grained part-level semantics while preserving holistic motion information.

Final Objective. The total HBM loss combines global and part-level supervision:

$$
\mathcal { L } _ { \mathrm { H B M } } = \mathcal { L } _ { \mathrm { I n s t } } + \lambda _ { P } \mathcal { L } _ { \mathrm { P a r t } } .\tag{6}
$$

This hierarchical supervision yields a structurally consistent embedding space for precise retrieval.

## 4.3 Part-Level Motion Encoder

To capture the hierarchical structure of human motion, inspired by prior human motion modeling works [18], [26], we adopt a part-level motion encoding scheme that decomposes full-body motion into multiple semantically meaningful body parts. As illustrated in Fig. $^ { 4 , }$ a motion sez̄quence is first divided into six parts according to the skeletal topology, including the right arm, left arm, right leg, left leg, backbone, and root. Each part motion is processed independently by a shared part encoder to obtain part-level motion features. These features preserve fine-grained local motion semantics while maintaining parameter efficiency through weight sharing. The resulting part-level representations are then concatenated and aggregated by an instance encoder to form a holistic motion representation, which is subsequently used for hierarchical alignment and retrieval. This hierarchical encoding design enables the model to jointly model local part dynamics and global motion coherence, providing a structured motion representation that facilitates fine-grained text–motion alignment in the proposed framework.

## 4.4 Topology Structured Masking

To fully exploit the hierarchical structure learned by HBM during generation, we introduce Topology Structured Masking (TSM) as a structure-aware training objective. Unlike conventional uniform random masking that ignores semantic relevance, TSM adaptively modulates masking probabilities based on the hierarchical text–motion alignment established by HBM, ensuring the generator focuses on semantically critical body parts.

Hierarchical Semantic Relevance. As shown in Fig. 3b, given a text embedding $t _ { i }$ and part-level motion embeddings $\{ p _ { i , k } \} _ { k = 1 } ^ { K }$ from HBM, we compute part-level relevance weights:

$$
\alpha _ { i , k } = \mathrm { S o f t M a x } _ { k } \big ( \sin ( t _ { i } , p _ { i , k } ) \big ) ,\tag{7}
$$

where $\alpha _ { i , k }$ reflects the semantic alignment strength between the input text and the k-th body part.

Topology-Aware Mask Distribution. We define the masking probability for each body part as:

$$
\pi _ { i , k } = \pi _ { \mathrm { b a s e } } \cdot ( 1 - \alpha _ { i , k } ) ,\tag{8}
$$

where $\pi _ { \mathrm { b a s e } }$ is a global masking ratio. This strategy ensures that semantically aligned parts are preserved as structural anchors, while less relevant parts are masked aggressively. By protecting these core semantic features from corruption, we force the generator to learn how to coordinate global motion conditioned on the key body parts specified by the text, rather than attempting to hallucinate critical semantics from context. These part-level probabilities are then propagated to the 2D spatial–temporal token grid $( J \times T )$ according to the skeletal partition, such that all joints belonging to part k share the probability $\pi _ { i , k }$

Structured Masked Modeling Objective. During training, tokens are masked according to $\pi _ { i , k }$ and replaced by a learnable mask token. The generator is optimized to reconstruct the original latent codes z:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T S M } } = \mathbb { E } _ { z } \left[ \| z _ { \mathrm { p r e d } } - z \| _ { \mathrm { m a s k e d } } \right] . } \end{array}\tag{9}
$$

By aligning the masking distribution with semantic relevance, TSM forces the model to focus on reconstructing critical body parts, thereby enhancing part-level grounding and structural consistency.

## 4.5 Semantic Spatial-Temporal Attention

Unlike conventional cross-attention that treats retrieved features as uniform tokens, we design Semantic Spatial– Temporal Attention (SSTA) to explicitly respect the 2D spatial–temporal organization of motion latents through an asymmetric injection mechanism.

We represent motion as a 2D joint–time token grid, flattened into latent sequences $\boldsymbol { z } \in \mathbb { R } ^ { \mathbf { \tilde { B } } \times N \times d }$ (where $\overset { \vartriangle } { N } = \boldsymbol { T } \cdot \boldsymbol { J } )$ As illustrated in Fig. 3(c), SSTA decouples the roles of semantic conditioning and motion synthesis across the Key and Value pathways.

Query: Motion-Centric Focus. To preserve the structural integrity of the generation process, the Query is derived solely from the current motion latents:

$$
\begin{array} { r } { Q = W _ { q } z , } \end{array}\tag{10}
$$

ensuring that attention computation remains anchored in the spatial–temporal topology of the motion being generated.

Semantic Reference Construction. We compress the multi-modal retrieval context (prompt t, retrieved text $R _ { t }$ and retrieved motion ${ \cal R } _ { m } )$ into a compact semantic token:

$$
h _ { \mathrm { s e m } } = \mathrm { M L P } ( \mathrm { c o n c a t } [ t ; R _ { t } ; R _ { m } ] ) ,\tag{11}
$$

where $h _ { \mathrm { s e m } } \in \mathbb { R } ^ { B \times 1 \times d }$ serves as a global modulation signal rather than a sequence of tokens.

Key: Topology-Aware Gating. The Key pathway integrates both motion states and semantic context to determine where to attend:

$$
K = W _ { k } \cdot \mathrm { c o n c a t } ( z , \ h _ { \mathrm { s e m } } ) .\tag{12}
$$

By injecting $h _ { \mathrm { { s e m } } }$ only into the Key, semantic information modulates the attention weights without directly overwriting the motion representation.

Value: Motion-Domain Synthesis. The Value pathway focuses on what information to synthesize, combining current latents with retrieved dynamic priors:

$$
V = W _ { v } \cdot \operatorname { c o n c a t } ( z , R _ { m } ) .\tag{13}
$$

Here, $R _ { m }$ provides concrete motion details from the database, while z ensures continuity with the current generation state. Notably, textual semantics $( t , R _ { t } )$ are excluded from Value to prevent domain mismatch.

Output. The final output is computed via standard scaled dot-product attention:

$$
{ \mathrm { O u t p u t } } = { \mathrm { S o f t M a x } } \left( { \frac { Q K ^ { \top } } { \sqrt { d } } } \right) V ,\tag{14}
$$

which is then reshaped back to the 2D grid structure. This asymmetric design ensures that retrieved semantics guide the attention focus (via Key) and enrich the motion content (via $R _ { m }$ in Value), while strictly preserving the spatial–temporal topology of the generative backbone.

## 4.6 Latent-Aligned Retrieval

The components above enforce structural consistency along the two axes identified in Section 1: HBM aligns text and motion at the granularity dictated by skeletal topology, and SSTA injects retrieved evidence in a form compatible with the 2D spatial–temporal latent grid. A third axis, however, remains open. In ReMoMask, as in retrieval-augmented T2M at large [16], [18], [24], the space in which evidence is retrieved is not the space in which motion is generated: HBM embeds text and motion into a contrastive space $s ,$ optimized to discriminate matching from non-matching pairs, whereas the masked generator operates on the latent space Z of the 2D RVQ-VAE, optimized purely for reconstruction. The conference version’s retrieval evidence $( R _ { m } , R _ { t } ) \in \mathcal { S }$ is therefore consumed by attention layers whose substrate lives in Z: implicitly, the fusion modules are asked to realize a cross-space translation $\psi : { \mathcal { S } }  { \mathcal { Z } } .$ , supervised only indirectly through the generation loss. Since the two spaces never interact during training, nothing constrains their geometries to agree: two motions that are close under contrastive semantics may be far apart as latent trajectories. We refer to this discrepancy as the retrieval–generation representation gap.

ReMoMask-2 removes the gap at its source: rather than translating evidence across spaces, we construct the retrieval database directly in the generator’s own latent space, so that retrieved evidence arrives, by construction, in the representation the generator natively consumes.

Pre-Quantization Latent Keys. We build retrieval keys with the frozen encoder $\mathcal { E }$ of the pretrained 2D RVQ-VAE [15]. Given a motion $m ,$ the encoder produces a prequantization latent grid

$$
\begin{array} { r } { z _ { e } = \mathcal { E } ( m ) \in \mathbb { R } ^ { d _ { e } \times T ^ { \prime } \times J ^ { \prime } } , } \end{array}\tag{15}
$$

with channel dimension $d _ { e } ~ = ~ 1 0 2 4$ over a downsampled grid of $T ^ { \prime } ~ = ~ T / 4$ temporal steps and $J ^ { \prime } = \mathrm { ~  ~ 6 ~ }$ spatial positions. We deliberately operate on the continuous latent before residual quantization rather than on discrete token indices: the pre-quantization latent preserves the continuous geometry of the encoder space and directly supports cosinebased nearest-neighbor search, whereas quantized indices are categorical and discard within-code variation. A motionlevel key is obtained by average pooling over the grid followed by $\ell _ { 2 }$ normalization:

$$
\bar { z } _ { e } = \frac { \hat { z } _ { e } } { \Vert \hat { z } _ { e } \Vert _ { 2 } } , \qquad \hat { z } _ { e } = \frac { 1 } { T ^ { \prime } J ^ { \prime } } \sum _ { t ^ { \prime } = 1 } ^ { T ^ { \prime } } \sum _ { j ^ { \prime } = 1 } ^ { J ^ { \prime } } z _ { e } ( t ^ { \prime } , j ^ { \prime } ) ,\tag{16}
$$

where $\boldsymbol { z } _ { e } ( t ^ { \prime } , j ^ { \prime } ) ~ \in ~ \mathbb { R } ^ { d _ { e } }$ denotes the latent vector at grid position $( t ^ { \prime } , j ^ { \prime } )$ . Applying this to every training motion and expanding each motion over its captions yields the database

$$
\begin{array} { r } { \mathcal { D } = \big \{ \big ( \bar { z } _ { e } ^ { ( j ) } , x ^ { ( j ) } \big ) \big \} _ { j = 1 } ^ { M } , \qquad M = 6 6 , 9 1 2 , } \end{array}\tag{17}
$$

covering 23,384 training clips (counting mirrored copies as separate database entries); at query time, entries originating from the same motion are deduplicated so that the retrieved set contains distinct exemplars. Since keys are unit-normalized, retrieval reduces to cosine similarity over D. The RVQ-VAE remains entirely frozen throughout: the database is a cached view of representations the generator already uses.

Query Projector. Retrieval requires mapping a text prompt into the same key space. We instantiate this as a lightweight projector ϕ applied to the 512-d CLIP ViT-B/32 [17] text embedding t already employed by the framework:

$$
\phi ( t ) = \operatorname { n o r m } \big ( W _ { 2 } \operatorname { G E L U } ( W _ { 1 } t ) \big ) ,\tag{18}
$$

where $W _ { 1 } \in \mathbb { R } ^ { d _ { e } \times { 5 1 2 } } , W _ { 2 } \in \mathbb { R } ^ { d _ { e } \times d _ { e } } .$ , and norm(·) denotes $\ell _ { 2 }$ normalization. The projector holds ∼1.57M parameters and is the only component trained in this stage.

Distillation from the HBM Retriever. Naively regressing ϕ(t) onto the paired key would supervise a single point and ignore the ranking structure that makes retrieval useful. Instead, we distill the HBM retriever of the conference version, which remains a strong cross-modal ranker, into the latent key space, following the soft-target distillation formulation of [31]. For a training caption x with embedding t, let $\Omega ( x )$ denote the κ database entries ranked highest by the teacher $( \kappa = 2 5 6 )$ . Teacher and student induce distributions over Ω(x) from their respective similarities at temperature $\tau = 0 . 0 7 ,$ matching the retriever’s contrastive temperature:

$$
p _ { j } ^ { \bigstar } = \frac { \exp \big ( s _ { j } ^ { \bigstar } / \tau \big ) } { \sum _ { j ^ { \prime } \in \Omega ( x ) } \exp \big ( s _ { j ^ { \prime } } ^ { \bigstar } / \tau \big ) } , \qquad \bigstar \in \{ \mathrm { H B M } , \phi \} ,\tag{19}
$$

where $s _ { j } ^ { \mathrm { H B M } }$ is the teacher’s text–motion similarity for entry $j$ in $s ,$ and $s _ { j } ^ { \phi } \ = \ \sin ( \phi ( t ) , \bar { z } _ { e } ^ { ( j ) } )$ is the student’s cosine similarity in the latent key space. The projector minimizes the KL divergence from teacher to student,

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \mathrm { K L } \bigl ( p ^ { \mathrm { H B M } } \bigr | \bigr | p ^ { \phi } \bigr ) = \sum _ { j \in \Omega ( x ) } p _ { j } ^ { \mathrm { H B M } } \log \frac { p _ { j } ^ { \mathrm { H B M } } } { p _ { j } ^ { \phi } } .\tag{20}
$$

Restricting the support to the teacher’s top-κ candidates concentrates supervision on the ranking decisions that matter for retrieval, and the soft targets transfer the teacher’s graded preferences rather than a single hard label. The projector is trained for 200 epochs with a batch size of 128, while the RVQ-VAE, the teacher, and the CLIP text encoder all remain frozen. In effect, distillation transfers the teacher’s cross-modal ranking into the generative latent space: the resulting retriever is trained to preserve the semantic precision of HBM while returning evidence natively expressed in the generator’s representation.

Integration with SSTA. At generation time, the query ϕ(t) retrieves the nearest database entries by cosine similarity, and both retrieval streams are read out of the latentaligned space: $R _ { m }$ is the pooled key $\bar { z } _ { e } ^ { ( j ) }$ of a retrieved entry, and $R _ { t } ~ = ~ \phi ( t ^ { ( j ) } )$ is the projection of the CLIP embedding t<sup>(j)</sup> of its caption $x ^ { ( j ) ^ { \bullet } }$ . When $k \ > \ 1$ entries are retrieved, each stream is averaged over the k entries, so that the fusion pathways always receive a single $R _ { m }$ and a single $R _ { t }$ regardless of k. As both streams are $d _ { e ^ { - } }$ dimensional while the generator operates at latent width $d ,$ we insert two linear adapters, $\mathbb { R } ^ { d _ { e } ^ { \star } }  \mathbb { R } ^ { d }$ , one per stream, ahead of the fusion pathways. These adapters are the only architectural addition: SSTA itself—the asymmetric Q-K-V routing of Section 4.5, including the three-way semantic token $\bar { h } _ { \mathrm { s e m } } = \mathrm { M L P } |$ (concat[t; $R _ { t } ; R _ { m } ] )$ —is left unchanged. One routing decision, however, is revisited as a direct consequence of the alignment. The conference version excluded textual signals from the Value pathway to prevent domain mismatch (Section 4.5): there, $\bar { \boldsymbol { R } } _ { t }$ was an embedding in $s ,$ foreign to the synthesis substrate. After latent alignment, $R _ { t } \stackrel { \smile } { = } \phi ( t ^ { ( j ) } )$ is itself a point in the generator’s latent geometry, so the motion-domain criterion governing the Value pathway now admits it; we therefore include $R _ { t }$ alongside $R _ { m }$ in the Value content, while the prompt embedding $t ,$ still a CLIP-space vector, remains excluded. Latent-aligned retrieval is thus a plug-and-play replacement for the retrieval space; Table 5, whose rows share the same generator, fusion modules, and training recipe, isolates the effect of the retrieval representation from that of the fusion design.

Single-Stage Deployment. ReMoMask-2 is deployed as a single mask-transformer stage. We also equipped it with a residual-refinement stage mirroring the conference pipeline and found the extra stage harmful: FID degrades from 0.042 to 0.068 (+0.026), a gap far exceeding the 95% confidence intervals of either configuration. With retrieval operating in the generative latent space, the coarse token prediction already absorbs the correction that residual refinement is designed to provide, so the residual stage has no systematic error left to remove and its updates act instead as an additional source of perturbation; we therefore omit it.

## 5 EXPERIMENT

## 5.1 Dataset and Evaluation Metrics

We evaluate our model on HumanML3D [33], KIT-ML [34], and SnapMoGen [14] datasets. The HumanML3D consists of 14,616 motions and 44,970 texts, while KIT-ML consists of 3,911 motions and 6,278 texts. The SnapMoGen is the largest available dataset, comprising 20,450 motions and 122,565 texts. The motion feature dimensions are 263, 251, and 296 for HumanML3D, KIT-ML, and SnapMoGen, respectively. HumanML3D and KIT-ML are split into training, validation, and test sets with a ratio of 0.80:0.05:0.15, while SnapMoGen uses a split of 0.85:0.05:0.10.

TABLE 3: Results of text-to-motion and motion-to-text retrieval benchmark on HumanML3D, KIT-ML, and SnapMoGen datasets. † indicates Results are reproduced by us following the descriptions in the original paper, as no official implementation is publicly available. Best results are highlighted in bold, and second-best results are underlined.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Methods</td><td colspan="6">Text-to-motion retrieval</td><td colspan="6">Motion-to-text retrieval</td></tr><tr><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td><td>R@1↑</td><td>R@2↑</td><td>R@3↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td></tr><tr><td rowspan="5">HumanML3D</td><td>TEMOS [25]</td><td>2.12</td><td>4.09</td><td>5.87</td><td>8.26</td><td>13.52</td><td>173.00</td><td>3.86</td><td>4.54</td><td>6.94</td><td>9.38</td><td>14.00</td><td>183.25</td></tr><tr><td>TMR [23]</td><td>5.68</td><td>10.59</td><td>14.04</td><td>20.34</td><td>30.94</td><td>28.00</td><td>9.95</td><td>12.44</td><td>17.95</td><td>23.56</td><td>32.69</td><td>28.50</td></tr><tr><td>MotionPatches† [32]</td><td>10.80</td><td>14.98</td><td>20.00</td><td>26.72</td><td>38.02</td><td>19.00</td><td>11.25</td><td>13.86</td><td>19.98</td><td>26.86</td><td>37.40</td><td>20.50</td></tr><tr><td>ReMoGPT† [18]</td><td>11.00</td><td>17.02</td><td>22.18</td><td>29.48</td><td>43.43</td><td>14.00</td><td>12.25</td><td>14.95</td><td>21.45</td><td>28.34</td><td>39.11</td><td>19.00</td></tr><tr><td>HBM (Ours)</td><td>18.49</td><td>21.20</td><td>25.63</td><td>32.40</td><td>46.27</td><td>12.00</td><td>14.83</td><td>17.63</td><td>25.60</td><td>30.75</td><td>44.61</td><td>16.00</td></tr><tr><td rowspan="5">KIT-ML</td><td>TEMOS [25]</td><td>7.11</td><td>13.25</td><td>17.59</td><td>24.10</td><td>35.66</td><td>24.00</td><td>11.69</td><td>15.30</td><td>20.12</td><td>26.63</td><td>36.39</td><td>26.50</td></tr><tr><td>TMR [23]</td><td>7.23</td><td>13.98</td><td>20.36</td><td>28.31</td><td>40.12</td><td>17.00</td><td>11.20</td><td>13.86</td><td>20.12</td><td>28.07</td><td>38.55</td><td>18.00</td></tr><tr><td>MotionPatches† [32]</td><td>14.02</td><td>21.08</td><td>28.91</td><td>34.10</td><td>50.00</td><td>10.50</td><td>13.61</td><td>17.26</td><td>27.54</td><td>33.33</td><td>44.77</td><td>13.00</td></tr><tr><td>ReMoGPT† [18]</td><td>13.42</td><td>17.65</td><td>22.24</td><td>32.30</td><td>44.68</td><td>12.50</td><td>11.93</td><td>14.12</td><td>21.59</td><td>30.54</td><td>39.68</td><td>16.50</td></tr><tr><td>HBM (Ours)</td><td>16.75</td><td>23.10</td><td>30.50</td><td>37.56</td><td>52.20</td><td>8.50</td><td>12.14</td><td>16.36</td><td>25.64</td><td>35.58</td><td>46.15</td><td>14.00</td></tr><tr><td rowspan="5">SnapMoGen</td><td>TEMOS [25]</td><td>5.03</td><td>8.51</td><td>10.79</td><td>14.48</td><td>20.36</td><td>42.00</td><td>4.72</td><td>6.67</td><td>7.95</td><td>10.11</td><td>16.74</td><td>64.00</td></tr><tr><td>TMR [23]</td><td>7.29</td><td>11.80</td><td>14.43</td><td>18.82</td><td>27.39</td><td>28.50</td><td>8.03</td><td>10.22</td><td>13.49</td><td>20.91</td><td>29.06</td><td>33.00</td></tr><tr><td>MotionPatches† [32]</td><td>8.92</td><td>14.75</td><td>18.13</td><td>25.46</td><td>34.95</td><td>19.00</td><td>9.08</td><td>12.24</td><td>14.73</td><td>22.42</td><td>33.52</td><td>25.50</td></tr><tr><td>ReMoGPT† [18]</td><td>8.15</td><td>13.27</td><td>16.05</td><td>24.72</td><td>33.48</td><td>20.50</td><td>9.94</td><td>13.55</td><td>16.41</td><td>23.87</td><td>35.53</td><td>23.00</td></tr><tr><td>HBM (Ours)</td><td>12.43</td><td>17.84</td><td>22.54</td><td>28.22</td><td>42.83</td><td>16.50</td><td>11.62</td><td>15.35</td><td>19.32</td><td>26.18</td><td>39.41</td><td>19.00</td></tr></table>

![](images/5da6670557cfd184d4dedcb6210f9e38ccf16d10382fc4fddd3720f53289d51a.jpg)

![](images/f888269edb4b082770730c4cfe0948c6fb95c43ccc901b58623d9fa48722128e.jpg)  
(a) Heatmap between different retrieval models

HBM  
![](images/44791088be497b06fa044c3dd5761da735743b8c8317e01b2f0196c22402903e.jpg)

![](images/b52728c00422cfa103c840e3f65f82264c7a666a09133312da4cbf1c786023fe.jpg)

![](images/0587bf9fe44bad9dbdf860c888b202dca966522c6fbfde59cfab3d163c420d06.jpg)  
(b) Heatmap of HBM training process

![](images/9a924ac04b3e3060f0ee8dc3d91503755e22a46a0a79731ef52276a9ac0f3d55.jpg)  
Fig. 5: Heatmap of similarity matrix. The diagonal represents positive pairs, with darker colors meaning higher semantic similarity, conducted on HumanML3D.

Evaluation Metrics. For text–motion retrieval, we report Recall at different ranks (R@1, R@2, R@3, R@5, and R@10), which measures the proportion of queries whose groundtruth match appears within the top-k retrieved results. We also report the median rank (MedR), where lower values indicate better retrieval performance. For motion generation, following prior work [13], [15], we evaluate textto-motion (T2M) performance from three perspectives: (1) motion quality, measured by the Frechet Inception Distance (FID); (2) generation diversity, evaluated using Diversity and Multi-Modality (MModality); and (3) text–motion alignment, assessed by R-Precision at top 1/2/3 and the Multi-Modal Distance (MM Dist). Formal definitions of each metric are provided below.

We report standard evaluation metrics widely adopted in text-to-motion generation following prior work [33]. All metrics are computed in a shared embedding space obtained from the pretrained evaluation networks introduced in [33].

TABLE 4: Quantitative evaluation on HumanML3D, KIT-ML, and SnapMoGen datasets. We repeat the evaluation 20 times and report the average with 95% confidence interval. Results marked with † are reproduced following the original paper, as no official implementation is publicly available. Within each framework category, bold and underline indicate the best and the second best results.
<table><tr><td rowspan="2"></td><td rowspan="2">Methods</td><td rowspan="2">Framework</td><td colspan="3">R-Precision↑</td><td rowspan="2">FID↓</td><td rowspan="2">MMDIST↓</td><td rowspan="2">Diversity→</td><td rowspan="2">MModality↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td> $\overline { { \mathrm { T o p } 3 } }$ </td></tr><tr><td rowspan="10">HumL3D</td><td>Real MDM [19]</td><td>=</td><td>0.511±.003</td><td>0.703±.003</td><td>0.797±.002</td><td>0.002±0.000</td><td>2.974±0.008</td><td>9.503±0.065</td><td>一</td></tr><tr><td rowspan="7">T2M-GPT [20] MoMask [13] MoGenTS [15]</td><td rowspan="7"></td><td>0.492±.003</td><td></td><td> $0 . 6 1 1 ^ { \pm . 0 0 7 }$ </td><td> $0 . 5 4 4 ^ { \pm . 0 4 4 }$ </td><td> $5 . 5 6 6 ^ { \pm . 0 2 7 }$ </td><td> $\mathbf { 9 . 5 5 9 ^ { \pm . 0 8 6 } }$ </td><td>2.799±.072</td></tr><tr><td></td><td> $0 . 6 7 9 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 7 5 ^ { \pm . 0 0 2 }$ </td><td> $0 . 1 4 1 ^ { \pm . 0 0 5 }$ </td><td> $3 . 1 2 1 ^ { \pm . 0 0 9 }$ </td><td> $9 . 7 6 1 ^ { \pm . 0 7 3 }$ </td><td> $1 . 8 5 6 ^ { \pm . 0 0 3 }$ </td></tr><tr><td>0.521±.003</td><td> $0 . 7 1 0 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 0 5 ^ { \pm . 0 0 1 }$ </td><td> $0 . 0 4 6 ^ { \pm . 0 0 2 }$ </td><td> $2 . 9 6 9 ^ { \pm . 0 0 9 }$ </td><td> $9 . 6 2 8 ^ { \pm . 0 7 0 }$ </td><td> $1 . 2 4 5 ^ { \pm . 0 4 1 }$ </td></tr><tr><td>t2m 0.529±.003</td><td> $0 . 7 1 9 ^ { \pm . 0 0 2 }$ </td><td>0.812±.002</td><td> $0 . 0 3 3 ^ { \pm . 0 0 1 }$ </td><td> $2 . 8 6 7 ^ { \pm . 0 0 6 }$ </td><td> $9 . 5 7 0 ^ { \pm . 0 7 7 }$ </td><td> $1 . 2 0 5 ^ { \pm . 0 4 3 }$ </td></tr><tr><td> $\overline { { 0 . 5 1 7 ^ { \pm . 0 0 2 } } }$ </td><td> $0 . 7 0 9 ^ { \pm . 0 0 2 }$ </td><td> $\overline { { 0 . 8 0 3 ^ { \pm . 0 0 2 } } }$ </td><td> $\overline { { 0 . 0 6 9 ^ { \pm . 0 0 3 } } }$ </td><td> $\overline { { 2 . 9 4 8 ^ { \pm . 0 0 7 } } }$ </td><td> $\overline { { 9 . 6 1 1 ^ { \pm . 0 3 2 } } }$ </td><td> $1 . 1 9 2 ^ { \pm . 0 5 3 }$ </td></tr><tr><td> $0 . 5 0 0 ^ { \pm . 0 0 4 }$ </td><td> $0 . 6 9 5 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 9 5 ^ { \pm . 0 0 3 }$ </td><td> $0 . 1 1 4 ^ { \pm . 0 0 7 }$ </td><td> $2 . 9 4 5 ^ { \pm . 0 0 4 }$ </td><td> $9 . 7 1 8 ^ { \pm . 0 3 1 }$ </td><td> $2 . 2 3 1 ^ { \pm . 0 7 1 }$ </td></tr><tr><td> $\mathbf { 0 . 5 5 7 ^ { \pm . 0 0 3 } }$ </td><td> $\mathbf { 0 . 7 5 1 ^ { \pm . 0 0 2 } }$ </td><td> $\mathbf { 0 . 8 4 3 ^ { \pm . 0 0 1 } }$ </td><td> $\mathbf { 0 . 0 3 2 ^ { \pm . 0 0 2 } }$ </td><td> $\mathbf { 2 . 7 5 9 ^ { \pm . 0 0 7 } }$ </td><td> $9 . 5 7 1 ^ { \pm . 0 6 9 }$ </td><td> $2 . 7 9 4 ^ { \pm . 0 4 1 }$ </td></tr><tr><td>ReMoDiffuse [16]</td><td rowspan="5">RAG-t2m</td><td> $0 . 5 1 0 ^ { \pm . 0 0 5 }$ </td><td> $0 . 6 9 8 ^ { \pm . 0 0 6 }$ </td><td> $0 . 7 9 5 ^ { \pm . 0 0 4 }$ </td><td> $0 . 1 0 3 ^ { \pm . 0 0 4 }$ </td><td> $2 . 9 7 4 ^ { \pm . 0 1 6 }$ </td><td> $9 . 0 1 8 ^ { \pm . 0 7 5 }$ </td><td> $1 . 7 9 5 ^ { \pm . 0 4 3 }$ </td></tr><tr><td>ReMoGPT† [18]</td><td>0.501±.003</td><td>0.688±.003</td><td> $0 . 7 9 2 ^ { \pm . 0 0 5 }$ </td><td> $\overline { { 0 . 2 0 5 ^ { \pm . 0 2 3 } } }$ </td><td> $2 . 9 2 9 ^ { \pm . 0 2 0 }$ </td><td>9.763±.056</td><td>2.816±.003</td></tr><tr><td>RMD† [30]</td><td>0.524±.002</td><td>0.715±.002</td><td>0.811±.001</td><td> $0 . 1 1 1 ^ { \pm . 0 0 5 }$ </td><td> $2 . 8 7 9 ^ { \pm . 0 0 6 }$ </td><td> $\mathbf { 9 . 5 2 7 ^ { \pm . 0 9 0 } }$ </td><td> $2 . 6 0 4 ^ { \pm . 0 8 4 }$ </td></tr><tr><td>MoRAG-Diffuse† [24]</td><td>0.511±.003</td><td> $\overline { { 0 . 6 9 9 ^ { \pm . 0 0 3 } } }$ </td><td> $0 . 7 9 2 ^ { \pm . 0 0 2 }$ </td><td> $0 . 2 7 0 ^ { \pm . 0 1 0 }$ </td><td> $\overline { { 2 . 9 5 0 ^ { \pm . 0 0 1 } } }$ </td><td> $9 . 5 3 6 ^ { \pm . 1 0 4 }$ </td><td> $2 . 7 7 3 ^ { \pm . 0 1 1 }$ </td></tr><tr><td>ReMoMask [27]</td><td>0.485±.003  $\mathbf { 0 . 5 2 8 ^ { \pm . 0 0 2 } }$ </td><td> $0 . 6 7 6 ^ { \pm . 0 0 3 }$   $\mathbf { 0 . 7 1 9 ^ { \pm . 0 0 2 } }$ </td><td> $0 . 7 7 7 ^ { \pm . 0 0 2 }$ </td><td> $0 . 1 2 3 ^ { \pm . 0 0 3 }$ </td><td> $3 . 0 9 0 ^ { \pm . 0 0 8 }$ </td><td> $9 . 3 5 7 ^ { \pm . 0 8 1 }$ </td><td> $\overline { { 1 . 4 4 6 ^ { \pm . 0 4 5 } } }$ </td></tr><tr><td rowspan="12">Real KIT-MI</td><td>ReMoMask-2 (Ours)</td><td> $0 . 4 2 4 ^ { \pm . 0 0 5 }$  =</td><td>0.649±.006</td><td></td><td> $\mathbf { 0 . 8 1 3 ^ { \pm . 0 0 2 } }$   $0 . 7 7 9 ^ { \pm . 0 0 6 }$ </td><td> $\mathbf { 0 . 0 4 2 ^ { \pm . 0 0 2 } }$   $0 . 0 3 1 ^ { \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 2 . 8 6 0 ^ { \pm . 0 0 9 } }$   $2 . 7 8 8 ^ { \pm 0 . 0 1 2 }$ </td><td>9.472±.079  $1 1 . 0 8 0 ^ { \pm . 0 9 7 }$  —</td><td>1.243±.038 一</td></tr><tr><td>MDM [19]</td><td></td><td></td><td></td><td> $0 . 3 9 6 ^ { \pm . 0 0 4 }$ </td><td> $0 . 4 9 7 ^ { \pm . 0 2 1 }$ </td><td> $9 . 1 9 1 ^ { \pm . 0 2 2 }$ </td><td> $1 0 . 8 4 7 ^ { \pm . 1 0 9 }$ </td><td>1.907±.214</td></tr><tr><td>T2M-GPT [20]</td><td>t2m</td><td>0.416±.006 0.627±.006</td><td></td><td>0.745±.006</td><td> $0 . 5 1 4 ^ { \pm . 0 2 9 }$ </td><td> $3 . 0 0 7 ^ { \pm . 0 2 3 }$ </td><td> $1 0 . 8 6 0 ^ { \pm . 0 4 9 }$ </td><td> $\overline { { 1 . 7 9 8 ^ { \pm . 1 5 7 } } }$ </td></tr><tr><td>MoMask [13]</td><td>0.433±.007</td><td> $0 . 6 5 6 ^ { \pm . 0 0 5 }$ </td><td> $0 . 7 8 1 ^ { \pm . 0 0 5 }$ </td><td></td><td> $0 . 2 0 4 ^ { \pm . 0 1 1 }$ </td><td> $2 . 7 7 9 ^ { \pm . 0 2 2 }$ </td><td>10.203±.038</td><td> $1 . 1 3 1 ^ { \pm . 0 4 3 }$ </td></tr><tr><td>MoGenTS [15]</td><td> $\underline { { 0 . 4 4 5 ^ { \pm . 0 0 6 } } }$ </td><td> $0 . 6 7 1 ^ { \pm . 0 0 6 }$ </td><td> $0 . 7 9 7 ^ { \pm . 0 0 5 }$ </td><td> $\underline { { 0 . 1 4 3 ^ { \pm . 0 0 4 } } }$ </td><td></td><td> $2 . 7 1 1 ^ { \pm . 0 2 4 }$ </td><td> $1 0 . 9 1 8 ^ { \pm . 0 9 0 }$ </td><td> $1 . 4 9 3 ^ { \pm . 0 2 4 }$ </td></tr><tr><td>MoMask++ [14]</td><td>0.428±.008 0.387±.006</td><td> $0 . 6 2 8 _ { . } ^ { \pm . 0 0 6 }$ </td><td> $\overline { { 0 . 7 5 5 ^ { \pm . 0 0 7 } } }$ </td><td> $\overline { { 0 . 1 9 4 ^ { \pm . 0 1 6 } } }$ </td><td>2.814±.006</td><td></td><td>10.927±.101</td><td>1.247±.040</td></tr><tr><td>MARDM [21]</td><td></td><td></td><td> $0 . 6 1 0 ^ { \pm . 0 0 6 }$ </td><td>0.749±.006</td><td> $0 . 2 4 2 ^ { \pm . 0 1 4 }$ </td><td> $2 . 9 7 0 ^ { \pm . 0 3 2 }$ </td><td> $\overline { { 1 0 . 8 9 1 ^ { \pm . 0 4 5 } } }$ </td><td> $1 . 3 1 2 ^ { \pm . 0 5 3 }$ </td></tr><tr><td>LaMP [22]</td><td></td><td> $\mathbf { 0 . 4 7 9 ^ { \pm . 0 0 6 } }$ </td><td> $\mathbf { 0 . 6 9 1 ^ { \pm . 0 0 5 } }$ </td><td> $\mathbf { 0 . 8 2 6 ^ { \pm . 0 0 5 } }$ </td><td> $\mathbf { 0 . 1 4 1 ^ { \pm . 0 1 3 } }$ </td><td> $\mathbf { 2 . 7 0 4 ^ { \pm . 0 1 8 } }$ </td><td> $\mathbf { 1 0 . 9 2 9 ^ { \pm . 1 0 1 } }$ </td><td> $\mathbf { 1 . 9 7 3 ^ { \pm . 0 7 1 } }$ </td></tr><tr><td>ReMoDiffuse [16]</td><td></td><td>0.427±.014</td><td>0.641±.004</td><td>0.765±.055</td><td> $0 . 1 5 5 ^ { \pm . 0 0 6 }$ </td><td> $2 . 8 1 4 ^ { \pm . 0 1 2 }$ </td><td> $1 0 . 8 0 0 ^ { \pm . 1 0 5 }$ </td><td> $1 . 2 3 9 ^ { \pm . 0 2 8 }$ </td></tr><tr><td>ReMoGPT† [18]</td><td>RAG-t2m</td><td>0.376±.006</td><td>0.585±.004</td><td> $0 . 7 3 4 ^ { \pm . 0 0 5 }$ </td><td> $\overline { { 0 . 4 2 5 ^ { \pm . 0 2 2 } } }$ </td><td> $\overline { { 2 . 9 3 5 ^ { \pm . 0 0 9 } } }$ </td><td> $1 0 . 7 1 2 ^ { \pm . 0 9 2 }$ </td><td> $1 . 3 6 2 ^ { \pm . 0 2 0 }$ </td></tr><tr><td>RMD† [30]</td><td></td><td> $0 . 4 3 3 ^ { \pm . 0 0 7 }$ </td><td> $0 . 6 5 2 ^ { \pm . 0 0 5 }$ </td><td> $0 . 7 7 6 ^ { \pm . 0 0 4 }$ </td><td> $0 . 3 2 0 ^ { \pm . 0 1 9 }$ </td><td> $2 . 8 6 3 ^ { \pm . 0 1 5 }$ </td><td>10.879±.110</td><td> $\mathbf { i . 3 6 4 ^ { \pm . 1 0 0 } }$ </td></tr><tr><td>MoRAG-Diffuse† [24]</td><td></td><td>0.330±.008</td><td> $0 . 5 4 2 ^ { \pm . 0 0 7 }$ </td><td> $\overline { { 0 . 6 7 5 ^ { \pm . 0 0 4 } } }$ </td><td> $0 . 6 1 4 ^ { \pm . 0 4 3 }$ </td><td> $3 . 2 5 1 ^ { \pm . 0 2 2 }$ </td><td> $\overline { { 1 0 . 2 4 4 ^ { \pm . 0 9 2 } } }$ </td><td> $1 . 2 6 8 ^ { \pm . 0 4 5 }$ </td></tr><tr><td>ReMoMask [27]</td><td></td><td> $0 . 4 0 4 ^ { \pm . 0 0 7 }$ </td><td> $0 . 6 1 9 ^ { \pm . 0 0 5 }$ </td><td> $0 . 7 4 5 ^ { \pm . 0 0 5 }$ </td><td> $0 . 5 6 2 ^ { \pm . 0 1 9 }$ </td><td>3.027±.018</td><td>10.508±.105</td><td> $0 . 7 9 2 ^ { \pm . 0 3 0 }$ </td></tr><tr><td rowspan="10">Real Snappen</td><td>ReMoMask-2 (Ours)</td><td> $\mathbf { 0 . 4 5 7 ^ { \pm . 0 0 6 } }$ </td><td> $\mathbf { 0 . 6 7 3 ^ { \pm . 0 0 5 } }$ </td><td>一  $\mathbf { 0 . 8 0 1 ^ { \pm . 0 0 4 } }$ </td><td> $\mathbf { 0 . 1 3 8 ^ { \pm . 0 0 7 } }$ </td><td> $\mathbf { 2 . 7 4 3 ^ { \pm . 0 1 6 } }$ </td><td> $\mathbf { 1 0 . 9 6 4 ^ { \pm . 1 0 8 } }$ </td><td> $0 . 9 4 7 ^ { \pm . 0 4 3 }$ </td></tr><tr><td></td><td></td><td>0.976±.001</td><td> $0 . 9 8 5 ^ { \pm . 0 0 1 }$ </td><td> $0 . 0 0 1 ^ { \pm . 0 0 0 }$ </td><td> $2 . 8 3 5 ^ { \pm . 0 0 1 }$ </td><td> $1 9 . 8 4 9 ^ { \pm . 0 6 3 }$ </td><td></td></tr><tr><td></td><td>0.940±.001</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MDM [19]</td><td>0.503±.002</td><td> $0 . 6 5 3 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 2 7 ^ { \pm . 0 0 2 }$ </td><td> $5 7 . 7 8 3 ^ { \pm . 0 9 2 }$ </td><td> $8 . 6 2 7 ^ { \pm . 0 1 2 }$ </td><td> $1 8 . 7 0 1 ^ { \pm . 0 4 3 }$ </td><td> $\mathbf { 1 3 . 4 1 2 ^ { \pm . 2 3 1 } }$ </td></tr><tr><td>T2M-GPT [20]</td><td> $0 . 6 1 8 ^ { \pm . 0 0 2 }$   $0 . 7 7 7 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 7 3 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 1 2 ^ { \pm . 0 0 2 }$ </td><td> $3 2 . 6 2 9 ^ { \pm . 0 8 7 }$ </td><td> $6 . 8 5 5 ^ { \pm . 0 3 2 }$ </td><td> $1 8 . 5 1 3 ^ { \pm . 0 3 5 }$ </td><td> $9 . 1 7 2 ^ { \pm . 1 8 1 }$ </td></tr><tr><td>MoMask [13]</td><td>0.742±.004</td><td> $0 . 8 8 8 ^ { \pm . 0 0 2 }$  0.875±.003</td><td> $0 . 9 2 7 ^ { \pm . 0 0 2 }$   $\overline { { 0 . 9 1 2 ^ { \pm . 0 0 4 } } }$ </td><td> $1 7 . 4 0 4 ^ { \pm . 0 5 1 }$ </td><td> $3 . 8 1 2 ^ { \pm . 0 0 8 }$ </td><td> $1 9 . 5 8 3 ^ { \pm . 0 7 4 }$ </td><td> $8 . 1 8 3 ^ { \pm . 1 8 4 }$ </td></tr><tr><td>MoGenTS [15]</td><td></td><td> $\mathbf { 0 . 9 0 5 ^ { \pm . 0 0 2 } }$ </td><td> $\mathbf { 0 . 9 3 8 ^ { \pm . 0 0 1 } }$ </td><td> $\overline { { 2 3 . 9 2 7 ^ { \pm . 0 4 5 } } }$   $\mathbf { 1 5 . 0 6 0 ^ { \pm . 0 6 5 } }$ </td><td> $3 . 9 2 3 ^ { \pm . 0 0 6 }$   $3 . 7 8 4 ^ { \pm . 0 0 6 }$ </td><td> $1 9 . 4 5 7 ^ { \pm . 0 6 1 }$ </td><td> $8 . 8 4 1 ^ { \pm . 1 9 7 }$ </td></tr><tr><td> $\mathbf { M o M a s k + + } \left[ 1 4 \right]$ </td><td> $\mathbf { 0 . 8 0 2 ^ { \pm . 0 0 1 } }$ </td><td> $0 . 8 1 2 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 6 0 ^ { \pm . 0 0 2 }$ </td><td> $2 6 . 8 7 8 ^ { \pm . 1 3 1 }$ </td><td> $\overline { { 3 . 9 4 7 ^ { \pm . 0 0 9 } } }$ </td><td> $\mathbf { 1 9 . 7 6 4 ^ { \pm . 0 2 7 } }$   $1 9 . 5 9 8 ^ { \pm . 0 1 9 }$ </td><td> $7 . 2 5 9 ^ { \pm . 1 8 0 }$ </td></tr><tr><td>MARDM [21] LaMP [22]</td><td> $0 . 6 5 9 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 0 3 ^ { \pm . 0 0 6 }$ </td><td> $0 . 9 0 0 ^ { \pm . 0 0 3 }$ </td><td> $1 9 . 1 1 4 ^ { \pm . 0 8 5 }$ </td><td> $\mathbf { 3 . 6 9 7 ^ { \pm . 0 1 4 } }$ </td><td> $\underline { { 1 9 . 6 1 8 ^ { \pm . 0 3 7 } } }$ </td><td> $9 . 8 1 2 ^ { \pm . 2 8 7 }$ </td></tr><tr><td>ReMoDiffuse [16]</td><td> $0 . 7 1 1 ^ { \pm . 0 0 5 }$   $0 . 4 9 1 ^ { \pm . 0 0 3 }$ </td><td> $0 . 6 7 3 ^ { \pm . 0 0 7 }$ </td><td> $0 . 7 8 5 ^ { \pm . 0 0 2 }$ </td><td></td><td></td><td></td><td> $\overline { { 9 . 3 2 5 ^ { \pm . 2 3 9 } } }$ </td></tr><tr><td>ReMoGPT† [18] RMD† [30]</td><td rowspan="5">RAG-t2m</td><td> $0 . 6 4 7 ^ { \pm . 0 0 2 }$ </td><td></td><td> $0 . 8 3 9 ^ { \pm . 0 0 2 }$ </td><td> $6 8 . 4 6 0 ^ { \pm . 0 9 1 }$ </td><td> $5 . 9 3 0 ^ { \pm . 0 2 9 }$ </td><td> $1 8 . 7 5 3 ^ { \pm . 0 4 6 }$ </td><td> $8 . 2 6 0 ^ { \pm . 2 1 5 }$ </td></tr><tr><td></td><td></td><td> $0 . 7 9 3 ^ { \pm . 0 0 4 }$ </td><td></td><td> $4 4 . 1 2 1 ^ { \pm . 0 1 6 }$ </td><td> $4 . 8 8 3 ^ { \pm . 0 0 8 }$ </td><td> $1 9 . 4 3 5 ^ { \pm . 0 5 1 }$ </td><td> $\mathbf { 9 . 3 4 7 ^ { \pm . 4 5 9 } }$ </td></tr><tr><td>MoRAG-Diffuse† [24]</td><td> $0 . 5 3 4 ^ { \pm . 0 0 2 }$   $0 . 5 9 7 ^ { \pm . 0 0 3 }$ </td><td> $0 . 6 5 8 ^ { \pm . 0 0 2 }$ </td><td> $0 . 7 1 3 ^ { \pm . 0 0 3 }$ </td><td> $4 9 . 0 8 4 ^ { \pm . 0 2 3 }$ </td><td> $5 . 9 1 6 ^ { \pm . 0 1 7 }$   $5 . 9 2 4 ^ { \pm . 0 3 6 }$ </td><td> $1 8 . 5 6 6 ^ { \pm . 0 2 1 }$ </td><td> $7 . 5 5 9 ^ { \pm . 2 7 3 }$ </td></tr><tr><td>ReMoMask [27]</td><td> $0 . 7 5 2 ^ { \pm . 0 0 3 }$ </td><td> $0 . 7 9 2 ^ { \pm . 0 0 2 }$   $0 . 8 8 3 ^ { \pm . 0 0 2 }$ </td><td> $0 . 8 3 5 ^ { \pm . 0 0 2 }$   $0 . 9 2 1 ^ { \pm . 0 0 2 }$ </td><td> $4 1 . 4 1 7 ^ { \pm . 0 3 0 }$   $2 7 . 8 1 6 ^ { \pm . 2 2 6 }$ </td><td> $3 . 7 1 2 ^ { \pm . 0 1 1 }$ </td><td> $1 8 . 9 2 7 ^ { \pm . 0 3 2 }$   $1 9 . 4 7 2 ^ { \pm . 0 5 8 }$ </td><td> $8 . 2 7 8 ^ { \pm . 2 4 7 }$   $8 . 6 1 3 ^ { \pm . 1 9 2 }$ </td></tr><tr><td>ReMoMask-2 (Ours)</td><td> $\mathbf { 0 . 7 8 8 ^ { \pm . 0 0 3 } }$ </td><td> $\overline { { { \bf 0 . 8 9 7 ^ { \pm . 0 0 4 } } } }$ </td><td> $\overline { { { \bf 0 . 9 3 4 } ^ { \pm . 0 0 2 } } }$ </td><td> $\mathbf { i 3 . 1 7 4 ^ { \pm . 0 5 7 } }$ </td><td> $\mathbf { 3 . 7 0 6 ^ { \pm . 0 1 1 } }$ </td><td> $\mathbf { i 9 . 7 0 3 ^ { \pm . 0 4 6 } }$ </td><td> $\overline { { 8 . 3 3 1 ^ { \pm . 1 7 4 } } }$ </td></tr></table>

We denote the feature representations of ground-truth motions, generated motions, and text descriptions as $f _ { g t } , f _ { p r e d } ,$ and f<sub>text</sub>, respectively.

to the Euclidean distance between motion and text features, and report the retrieval accuracy at Top-1, Top-2, and Top-3. A retrieval is considered successful if the ground-truth description appears within the top-k ranked results.

5.1.0.1 Frechet Inception Distance (FID).: FID evaluates the distribution-level similarity between generated motions and ground-truth motions in the feature space. It is defined as

5.1.0.3 Multimodal Distance (MM-Dist).: MM-Dist measures the alignment between generated motions and their corresponding text descriptions at the feature level. Given N text–motion pairs, it is computed as the average Euclidean distance between motion and text features:

$$
\begin{array} { r l } & { \mathrm { F I D } = \| \mu _ { g t } - \mu _ { p r e d } \| ^ { 2 } } \\ & { \qquad + \operatorname { T r } \Bigl ( \Sigma _ { g t } + \Sigma _ { p r e d } - 2 \bigl ( \Sigma _ { g t } \Sigma _ { p r e d } \bigr ) ^ { \frac { 1 } { 2 } } \Bigr ) } \end{array}\tag{21}
$$

$$
\mathrm { M M \mathrm { - } D i s t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert f _ { p r e d , i } - f _ { t e x t , i } \rVert .\tag{22}
$$

where $\mu _ { g t }$ and $\mu _ { p r e d }$ denote the empirical means of $f _ { g t }$ and f<sub>pred</sub>, and Σ<sub>gt</sub> and $\Sigma _ { p r e d }$ are the corresponding covariance matrices.

5.1.0.2 R-Precision (Top-k).: R-Precision assesses text–motion matching accuracy. For each generated motion, its ground-truth text description is combined with 31 randomly sampled mismatched descriptions from the test set to form a candidate pool. We rank all candidates according

5.1.0.4 Diversity.: Diversity quantifies the overall variability of generated motions across the dataset. We randomly sample $S _ { d i s }$ pairs of generated motion features, denoted as $( f _ { p r e d , i } , f _ { p r e d , i } ^ { \prime } )$ , and compute

$$
\mathrm { D i v e r s i t y } = \frac { 1 } { S _ { d i s } } \sum _ { i = 1 } ^ { S _ { d i s } } \lVert f _ { p r e d , i } - f _ { p r e d , i } ^ { \prime } \rVert .\tag{23}
$$

Following [33], we set $S _ { d i s } = 3 0 0$ in all experiments.

5.1.0.5 Multimodality (MModality).: MModality evaluates the diversity of motions generated from the same text description. For each text input, we generate multiple motion samples and randomly select two subsets, each containing 10 motions. Let $( \dot { f _ { p r e d , i , j } } , f _ { p r e d , i , j } ^ { \prime } )$ denote the feature pair from the j-th sample of the i-th text. MModality is computed as

$$
\mathrm { M M o d a l i t y } = \frac { 1 } { 1 0 N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { 1 0 } \lVert f _ { p r e d , i , j } - f _ { p r e d , i , j } ^ { \prime } \rVert .\tag{24}
$$

## 5.2 Implementation Details

For the motion representation, we use a pretrained 2D-RVQ-VAE [15] comprising two 6-layer residual-quantization branches: a joint-level 2D branch with codebooks of 256 codes of 1024 dimensions, whose pre-quantization latent provides our retrieval keys $z _ { e } ,$ and a holistic 1D branch with codebooks of 512 codes of 512 dimensions. For the topology structured masking, we set $\pi _ { b a s e }$ to 0.5. For the text–motion retrieval model, the motion encoder employs a 4-layer transformer for each body part, with both part-level and instance-level motion embeddings set to 512 dimensions. The text encoder is a frozen pretrained CLIP [17] ViT-B/32 model with one additional trainable transformer layer, shared across all models. For the motion masked model, the SSTA is set to have 6 layers, 8 heads, and 512 latent dimensions. The retrieval model is trained for 200 epochs on a Tesla A800 GPU, with a $\lambda _ { P }$ of 1, a batch size of 128, a queue size of 65,536, and a momentum $\mu$ of 0.999. The masked models are trained on 8 Tesla A800 GPUs for up to 2,000 epochs with a batch size of 64. All models are implemented in PyTorch.

Evaluation Protocol. All results in Table 4 are obtained under a unified protocol: metrics are computed by the same evaluation pipeline, repeated 20 times, and reported with 95% confidence intervals; reconstruction measurements, which are deterministic, are the only exception and are reported without intervals. Entries marked † are reproduced by us following the original description, as no public implementation is available. For our conference-version ReMoMask, the table reports our own reproduction under this protocol, obtained by re-running the publicly released checkpoints where available and retraining from the official configuration otherwise. ReMoMask-2 is evaluated under the same protocol throughout.

## 5.3 Main Results

Evaluation of Motion Retrieval. We evaluate ReMoMask on text-motion and motion-text retrieval on HumanML3D, KIT-ML, and SnapMoGen (Table 3), where it attains state-ofthe-art text-to-motion retrieval on all three benchmarks and the best motion-to-text retrieval on HumanML3D and Snap-MoGen. ReMoMask consistently outperforms prior methods on HumanML3D and SnapMoGen, improving text-tomotion R@1 on HumanML3D to 18.49 versus 11.00 for the next-best method, and increasing R@1 on SnapMoGen from 8.92 to 12.43. On KIT-ML, ReMoMask is best on all six textto-motion metrics, while in the motion-to-text direction it leads on R@5 and R@10 and is second on the remaining four metrics. Fig. 5 further shows that HBM yields clearer diagonal structures with suppressed off-diagonal responses, visualizing the same discriminative text–motion alignment reflected in the retrieval metrics above.

Evaluation of Motion Generation. We evaluate motion generation performance on the HumanML3D, KIT-ML, and SnapMoGen datasets, as illustrated in Table 4. Evaluated under the unified protocol of Sec. 5.2, ReMoMask-2 is the strongest retrieval-augmented system on all three benchmarks against both published and reproduced baselines, attaining both the best FID and the best R-precision within that group, and it generally improves substantially over our conference-version ReMoMask on fidelity, R-precision, and MM-Dist across the three benchmarks, while using only a single mask-transformer stage. On HumanML3D it surpasses the full mask-plus-residual MoMask pipeline on both fidelity (0.042 versus 0.046) and Top-1 R-precision (0.528 versus 0.521) while itself running a single generative stage, and its fidelity is comparable to the strongest systems on this benchmark; on KIT-ML (0.138) and SnapMoGen (13.174) ReMoMask-2 attains the lowest FID among all compared methods. Multimodality decreases on HumanML3D and SnapMoGen while increasing on KIT-ML. We attribute the decrease on the first two benchmarks to retrieval conditioning narrowing the distribution of samples drawn for a given prompt, a hypothesis consistent with the trend on HumanML3D and SnapMoGen; on KIT-ML, where multimodality instead increases, a different mechanism is likely at play. It also delivers the fastest inference among the systems compared in Fig. 6.

Qualitative Comparisons. We compare our method with others in Fig. 2. The qualitative examples illustrate that our model’s motions align closely with the text descriptions, while the compared methods more often exhibit degraded motion quality or semantic mismatches; Sec. A reports a controlled user study quantifying this comparison.

Inference Efficiency Comparison. Beyond generation quality, we compare our method with existing approaches in terms of both FID and inference cost. Following [13], the inference cost is quantified as the mean inference time over 100 samples on a single NVIDIA 2080Ti device, measured end to end and including, for the retrieval-augmented systems, the cost of encoding the query and searching the database. As shown in Fig. 6, ReMoMask-2 achieves the fastest inference among the retrieval-augmented and nonretrieval methods plotted there. Two changes contribute to this margin, only one of which is the removal of the residual-refinement stage: latent-aligned retrieval also replaces the conference-version retrieval stack (a full crossmodal encoder run over the prompt, followed by a search in a separate contrastive index) with a 1.57M-parameter projector and a single cosine lookup over cached keys. Both changes act jointly to produce this speedup.

## 5.4 Ablation Study

Unless otherwise stated, every variant is trained and evaluated under the unified protocol of Section 5.2 with singlestage mask-only inference; generation-quality numbers are therefore obtained under the same protocol as Table 4. Two questions inherited from the conference version are re-examined here, because latent alignment changes their premise: how much the conference components still contribute (Table 6), and how the benefit of retrieval scales with the size of the database (Fig. 7). The remaining componentlevel ablations of the conference architecture (the hierarchical and momentum designs of the retriever, the full sweep over the information sources routed into the SSTA key and value pathways, the sensitivity of contrastive retriever training to its hyperparameters, and the cross-backbone transferability of the retriever) are reported in the conference version [27] and are not repeated here.

![](images/8e5e1681c8f72b55cb982306541e12f5927c88a592a68bc01dc03401201a938a.jpg)  
Fig. 6: Comparison on FID and Inference Cost.

TABLE 5: Impact of the retrieval representation on HumanML3D. The generator, fusion modules, and training recipe are identical across rows; only the representation in which evidence is retrieved and expressed changes. The second row keeps the conference-version database in the semantic space $\bar { \boldsymbol { s } }$ but replaces the textual stream entering the fusion pathways with the latent-aligned $R _ { t } = \phi ( \cdot )$ , so that the motion and textual streams can be moved into the generative geometry independently.
<table><tr><td>Retrieval representation</td><td>FID↓</td><td>Top1↑</td></tr><tr><td>Semantic space (HBM, conference design)</td><td> $\overline { { 0 . 1 0 9 ^ { \pm . 0 0 3 } } }$ </td><td> $\overline { { 0 . 5 0 3 ^ { \pm . 0 0 3 } } }$ </td></tr><tr><td>Semantic space, latent-aligned  $R _ { t }$ </td><td> $0 . 1 0 1 ^ { \pm . 0 0 4 }$ </td><td> $0 . { \dot { 5 } } 0 7 ^ { \pm . 0 0 3 }$ </td></tr><tr><td>Quantized indices  $( z _ { q } )$ </td><td> $0 . 0 6 0 ^ { \pm . 0 0 3 }$ </td><td> $0 . 5 1 9 ^ { \pm . 0 0 3 }$ </td></tr><tr><td>Pre-quantization latent (ze, Ours)</td><td> $\mathbf { 0 . 0 4 2 ^ { \pm . 0 0 2 } }$ </td><td> $\mathbf { 0 . 5 2 8 ^ { \pm . 0 0 2 } }$ </td></tr></table>

Retrieval Space. Table 5 isolates the central design choice of ReMoMask-2. Keeping the single-stage generator and the SSTA fusion fixed, retrieving in the HBM semantic space (the conference retrieval design, retrained under the unified protocol) yields an FID of 0.109 for this generator paired with the conference retrieval space, whereas moving retrieval into the generator’s own pre-quantization latent space improves it to 0.042, with Top1 following the same ordering. The two spaces indeed rank candidates very differently: over the test captions, each ranked against the full database of Section $4 . 6 ,$ the mean Spearman correlation between semantic-space and $z _ { e }$ -space rankings is 0.09 and their top-10 candidate sets overlap by 7.3%, quantifying the retrieval–generation representation gap. Retrieving over quantized indices $( z _ { q } )$ recovers most but not all of the benefit (0.060), likely because discretization discards within-code variation that the continuous $z _ { e }$ keys preserve.

This gain reflects an interaction between the two retrieval streams: their combined effect exceeds the sum of their individual, separable improvements. The second row of this table and the unaligned row of Table 9 each move one stream at a time, and the four configurations complete a $2 \times 2 \cdot$ with both streams in the semantic space FID is 0.109; expressing only the retrieved caption in the generative geometry gives 0.101; rebuilding only the motion database there gives 0.098; expressing both there gives 0.042. Each move alone yields a modest reduction: expressing only the caption in the generator’s geometry lowers FID from 0.109 to 0.101, and rebuilding only the motion database there lowers it to 0.098. But moving the caption after the database has already moved lowers FID by 0.056, from 0.098 to 0.042, far more than the 0.008 it is worth on its own: the two streams interact synergistically, and the full gain appears only once both are expressed together. This is what representation consistency means operationally: the evidence must arrive jointly, as a whole, in the generator’s geometry.

TABLE 6: Contribution of the conference-version components under latent-aligned retrieval, on HumanML3D. Each row is a separately trained variant: TSM is replaced by a uniform random masking schedule, and SSTA by a plain crossattention fusion block of matched capacity that attends over $R _ { m }$ and $R _ { t }$ as an unstructured token pair. Both retrieval streams remain available to that block; only the asymmetric Q-K-V routing over the generator’s token grid is removed; and the retrieval space, the database, and the training recipe are left unchanged. Absolute values here are comparable only within this table; Table 7 instead applies inference-time interventions to a single deployed model.
<table><tr><td>Variant</td><td>FID↓</td><td> $\underline { { \mathrm { T o p 1 \uparrow } } }$ </td></tr><tr><td>ReMoMask-2 (full)</td><td> $\mathbf { 0 . 0 4 2 ^ { \pm . 0 0 2 } }$ </td><td> $\mathbf { \overline { { 0 . 5 2 8 ^ { \pm . 0 0 2 } } } }$ </td></tr><tr><td> $\scriptstyle { \mathrm { w } } / 0$  TSM (uniform masking) w/o SSTA (plain cross-attention)</td><td> $0 . 0 5 8 ^ { \pm . 0 0 3 }$   $0 . 0 6 7 ^ { \pm . 0 0 4 }$ </td><td> $0 . 5 2 2 ^ { \pm . 0 0 3 }$   $0 . 5 1 4 ^ { \pm . 0 0 3 }$ </td></tr></table>

Conference Components. Table 6 asks whether the two components of the conference architecture remain loadbearing once retrieval is moved into the generator’s latent space. They do, and they fail in different ways. Replacing the topology structured masking schedule with uniform random masking costs 0.016 FID (0.042 to 0.058); Top1 degrades as well, by 0.006 and beyond the intervals covering the two entries, but by less than half of what the fusion ablation costs. TSM shapes only the training-time masking distribution, so the ablated model still receives the full retrieved evidence at inference and affects fidelity more than alignment. Replacing SSTA with a plain cross-attention block of matched capacity is more damaging along both axes (0.067 FID, Top1 0.514). The retrieval keys are pooled vectors and carry no grid of their own; the plain block appends that evidence as extra tokens instead of anchoring the fusion in the generator’s joint–time token map (Sec. 4.5), so fidelity and alignment degrade together. The relative ordering of the two arms matches the cumulative ablation of the conference architecture [27], in which SSTA also accounted for more of the gain than TSM. Both arms nevertheless cost less than the retrieval-space and routing choices of Tables 5 and $^ { 9 , }$ which places the components as necessary support for the new retrieval design: the extension augments the conference architecture. The SSTA arm and the first row of Table 9 isolate different components (the former keeps both retrieval streams and removes only the routing structure, whereas the latter keeps the routing structure but withholds $R _ { t }$ entirely), so the larger cost of withholding $R _ { t }$ there is consistent with this fusion ablation. Because HBM is no longer the deployed retriever, its contribution to ReMoMask-2 is measured in its new role, as a distillation teacher, in Table 10.

TABLE 7: Graded conditioning ablation on HumanML3D. The first three rows keep real retrieved motion latents and destroy only their relevance to the query; the next two replace the evidence with degenerate constants; the row below the rule removes the retrieval pathway entirely and lies outside this ladder. All rows are inference-time interventions on the deployed model.
<table><tr><td>Condition</td><td>FID↓</td><td>Top1↑</td><td>MMDIST↓</td></tr><tr><td>Real retrieval (Ours)</td><td> $\mathbf { 0 . 0 4 2 ^ { \pm . 0 0 2 } }$  </td><td> $\mathbf { 0 . 5 2 8 ^ { \pm . 0 0 2 } }$ </td><td>2.860±.009</td></tr><tr><td>Shuffled pairing</td><td> $0 . 0 4 8 ^ { \pm . 0 0 2 }$ </td><td> $0 . 5 2 1 ^ { \pm . 0 0 3 }$ </td><td> $2 . 8 8 9 ^ { \pm . 0 0 6 }$ </td></tr><tr><td>Random neighbors</td><td> $0 . 0 5 6 ^ { \pm . 0 0 3 }$ </td><td> $0 . 5 1 7 ^ { \pm . 0 0 3 }$ </td><td> $2 . 8 9 8 ^ { \pm . 0 0 6 }$ </td></tr><tr><td>Zeroed features</td><td> $0 . 0 6 3 ^ { \pm . 0 0 4 }$ </td><td> $0 . 5 2 1 ^ { \pm . 0 0 3 }$ </td><td> $2 . 8 8 4 ^ { \pm . 0 0 9 }$ </td></tr><tr><td>Constant (mean) prior</td><td> $0 . 0 7 0 ^ { \pm . 0 0 5 }$ </td><td> $0 . 5 1 6 ^ { \pm . 0 0 3 }$ </td><td> $2 . 9 0 2 ^ { \pm . 0 0 6 }$ </td></tr><tr><td>No retrieval</td><td> $0 . 0 5 5 ^ { \pm . 0 0 5 }$ </td><td> $0 . 5 0 9 ^ { \pm . 0 0 2 }$ </td><td> $2 . 9 6 8 ^ { \pm . 0 0 7 }$ </td></tr></table>

Conditioning Fidelity. Table 7 tests whether the generator consumes the semantic content of retrieved evidence or merely benefits from its presence. The decisive evidence is the semantic ladder formed by the first three rows, where the retrieved entries remain real motion latents drawn from the same database and only their relevance to the query is destroyed: quality degrades monotonically from real retrieval (0.042) through shuffled pairing (0.048) to random neighbors (0.056), with each adjacent FID pair separated by non-overlapping confidence intervals; Top1 and MM-Dist reproduce the same ordering, though the gap between the two degraded conditions lies within their intervals. This ordering is the signature of genuine semantic consumption: were retrieval acting as a mere statistical regularizer, degrading its content while preserving its distribution would leave quality unchanged. The claim rests on FID, which is the only column in which the ladder separates step by step; we read the other two as corroborating its direction rather than as independent evidence. Removing retrieval altogether lands at 0.055, statistically indistinguishable from random neighbors. On fidelity the two degenerate conditions are worse still: the dataset-mean prior falls behind even the no-retrieval setting by a margin its confidence interval does not cover (0.070 versus 0.055), while the zero vector (0.063) is at best no better than removing retrieval. That ordering does not carry over to the other two columns: on Top1 and MM-Dist the zero-vector condition is indistinguishable from the rows that receive degraded but real evidence, so the degenerate constants are a statement about fidelity, not about alignment.

Database Coverage. Fig. 7 varies the second quantity retrieval depends on. The ladder above destroys the relevance of the retrieved evidence at a fixed database size; here relevance is left intact and the database is thinned instead, which caps the relevance that is available to be retrieved at all. Quality improves with coverage on both metrics: FID moves from 0.052 at 10% coverage to 0.042 at full coverage and Top1 from 0.513 to 0.528, with the two ends of each sweep separated by their confidence intervals. A straight line in $\log _ { 1 0 }$ of coverage describes the swept range well—

![](images/11ab106caf6316ae2e547f5adf1f20f6beb6d63a3232402c828be0a15d7e4937.jpg)  
Fig. 7: Effect of retrieval database coverage on HumanML3D. Subsets of the latent-aligned database are drawn uniformly at random with a shared seed across points and are swapped in at inference time only: the generator, the query projector, and the training recipe are those of the deployed model throughout, so these points share the intervention protocol of Table 7. The dashed line marks the no-retrieval setting of that table; 100% is the deployed configuration.

TABLE 8: Sensitivity to the number of retrieved entries k on HumanML3D.
<table><tr><td>k</td><td>1</td><td>2 (Ours)</td><td>3</td><td>4</td><td>6</td></tr><tr><td>FID↓</td><td> $\overline { { \bf 0 . 0 3 7 ^ { \pm . 0 0 2 } } }$ </td><td> $\overline { { 0 . 0 4 2 ^ { \pm . 0 0 2 } } }$ </td><td> $\overline { { 0 . 0 4 4 ^ { \pm . 0 0 4 } } }$ </td><td> $\overline { { 0 . 0 4 3 ^ { \pm . 0 0 3 } } }$ </td><td> $\overline { { 0 . 0 4 5 ^ { \pm . 0 0 3 } } }$ </td></tr><tr><td>Top1↑</td><td> $0 . 5 2 0 { \scriptstyle \pm . 0 0 3 }$ </td><td> $\mathbf { 0 . 5 2 8 ^ { \pm . 0 0 2 } }$ </td><td> $0 . 5 2 6 ^ { \pm . 0 0 4 }$ </td><td> $0 . 5 2 5 ^ { \pm . 0 0 3 }$ </td><td> $0 . 5 2 7 ^ { \pm . 0 0 2 }$ </td></tr></table>

$R ^ { 2 } ~ = ~ 0 . 9 3$ on FID, at a slope of about 0.010 FID per decade of database size. This fit describes the measured range only; extending it into a scaling-law claim would require sweeping database sizes beyond the single decade tested here. Because no adjacent pair of points is separated by its confidence interval, and points beyond half coverage are mutually indistinguishable, the figure supports the endto-end trend and the flattening of returns as a range-level pattern. The two experiments are complementary rather than redundant: the ladder isolates relevance, this sweep isolates supply, and both point away from a distributionlevel regularizer account. At 10% coverage the point estimate still outperforms the no-retrieval level of Table 7 (0.052 versus 0.055).

Retrieval Count. Table 8 sweeps the number of retrieved entries. The optimum is shallow: k=1 attains the lowest FID (0.037) at a consistent cost in text–motion alignment (Top1 drops from 0.528 to 0.520), while larger k degrades FID only mildly and leaves alignment essentially flat. We adopt k=2 as the balanced default, trading 0.005 FID for 0.008 Top1 against k=1; the criterion is joint fidelity and alignment, applied to the same metric suite that Table 4 reports, so the deployed 0.042 is deliberately not the lowest FID this model attains. The remaining inference hyperparameters (guidance scale, iteration count, sampling temperature) are inherited unchanged from the conference protocol.

Value-Pathway Routing. Table 9 verifies the routing decision revisited in Section 4.6, and its four rows settle the question inside a single system. Routing the semantic-space $\bar { R } _ { t }$ into the Value pathway is harmful: at 0.098 FID it is worse than routing no textual evidence at all (0.081), and it is the only row in which text–motion alignment degrades appreciably as well (Top1 0.512, against 0.525 with no textual stream in the pathway). Routing the latent-aligned $R _ { t }$ b fi l f ll b l h lf f while alignment is essentially unchanged (Top1 0.525 versus 0.528). The generator, the fusion block, the database, and the training recipe are identical across the four rows, so the verdict on this design choice reverses once the textual evidence is expressed in the generator’s own geometry by the distilled projector $\phi ,$ rather than by a separately trained linear adapter. The contentless control separates the two candidate explanations for so large a move from so small an intervention: appending a fixed random vector of the same width, one extra Value entry, no information, does not help and mildly hurts (0.086), so the deployed row’s gain is attributable specifically to the content of $R _ { t } ,$ as distinct from the mere capacity of an additional entry. The degraded rows are not all the same failure. Withholding $R _ { t }$ leaves the Value pathway with evidence that is genuine and in-domain, so alignment is largely preserved and the cost is confined to fidelity, which we attribute to fusion layers that never learn to reconcile motion evidence with its textual counterpart. Supplying the unaligned $R _ { t } ^ { S }$ instead requires every synthesis step to reconcile two incompatible geometries, which corrupts the conditioning content itself and degrades fidelity and alignment together: the reading Table 7 already supports for its degenerate constants, where forcing an out-of-distribution condition through the fusion layers costs more than removing the condition. Since the adapter is trained end-to-end with the rest of the model, the network is free in principle to learn to ignore this input; that it remains harmful is consistent with incompatible geometry. The damage also stops short of retrieving in the wrong space altogether (0.109 in Table 5): here the retrieved neighbors are the correct ones and only the textual branch is foreign to the substrate. Rows in this table, and the retrieval space variants of Table 5, are separately trained, unlike the inference-time interventions of Table $^ { 7 , }$ so their absolute FID reflects that separately-trained protocol, distinct from the no-retrieval row there (0.055).

TABLE 9: Value-pathway routing under latent-aligned retrieval on HumanML3D. All rows are separately trained variants that share the same generator, database, and training recipe; retrieval runs in the latent key space throughout and only the content appended to the Value pathway changes. Row 2 appends a fixed random unit vector ξ of the same width, a contentless control that isolates the effect of adding one more Value entry from the effect of what that entry carries. Row 3 appends the undistilled semantic-space embedding $R _ { t } ^ { S } \in \mathcal { S } ,$ , width-matched by a trainable linear adapter and optimized jointly with the rest of the model.
<table><tr><td>Value content</td><td>FID↓</td><td>Top1↑</td></tr><tr><td>concat(z,  $\overline { { R _ { m } ) } }$ </td><td> $\overline { { 0 . 0 8 1 ^ { \pm . 0 0 3 } } }$ </td><td> $\overline { { 0 . 5 2 5 ^ { \pm . 0 0 3 } } }$ </td></tr><tr><td>concat(z,  $R _ { m } , \xi )$  (random vector)</td><td> $0 . 0 8 6 ^ { \pm . 0 0 4 }$ </td><td> $0 . 5 2 3 ^ { \pm . 0 0 3 }$ </td></tr><tr><td>concat(z,  $R _ { m } , R _ { t } ^ { S } )$  (unaligned)</td><td> $0 . 0 9 8 ^ { \pm . 0 0 4 }$ </td><td> $0 . 5 1 2 ^ { \pm . 0 0 3 }$ </td></tr><tr><td>concat(z  $, R _ { m } , R _ { t } )$  (Ours)</td><td> $\mathbf { 0 . 0 4 2 ^ { \pm . 0 0 2 } }$ </td><td> $\mathbf { 0 . 5 2 8 ^ { \pm . 0 0 2 } }$ </td></tr></table>

TABLE 10: Distillation design for the query projector on HumanML3D. The three rows above the rule share the HBM retriever as teacher and vary the distillation objective; the row below it keeps the KL objective and swaps the teacher for TMR [23]. TC@1 denotes top-1 consistency between each projector and its own teacher, measured over 4,384 held-out captions (one per test motion), and is therefore comparable in level only within the HBM-teacher rows; it is reported as a diagnostic and carries no preferred direction, which is why its column has no arrow. FID is obtained by plugging each projector into the frozen generation stack, with the generator, the database, and all inference settings left untouched.
<table><tr><td>Objective / teacher</td><td>TC@1</td><td>FID↓</td></tr><tr><td>KL, HBM teacher (Ours)</td><td>0.030</td><td> $\mathbf { 0 . 0 4 2 ^ { \pm . 0 0 2 } }$  </td></tr><tr><td>InfoNCE, HBM teacher</td><td>0.051</td><td> $0 . 0 6 4 ^ { \pm . 0 0 3 }$ </td></tr><tr><td>MSE regression, HBM teacher</td><td>0.048</td><td> $0 . 3 7 2 ^ { \pm . 0 0 4 }$ </td></tr><tr><td>KL, TMR teacher</td><td>0.044</td><td> $0 . 0 7 1 ^ { \pm . 0 0 3 }$ </td></tr></table>

Distillation Design. Table 10 ablates how the query projector is aligned, and the outcome cautions against reading a retrieval proxy as a proxy for generation. Among the rows that share the HBM teacher, consistency with the teacher and downstream fidelity order the objectives differently: InfoNCE attains the highest top-1 agreement (0.051) and MSE regression is close behind (0.048), yet plugging those projectors into the frozen generation stack yields FID 0.064 and 0.372, respectively, whereas soft top-κ distillation, the lowest on consistency, at 0.030, is the only objective that reaches 0.042. Top-1 agreement is a deliberately strict statistic here. The two spaces order candidates almost independently (Spearman 0.09, Table 5), so the teacher’s single highest-ranked entry frequently falls outside the set reachable as an argmax in $z _ { e } ,$ and what distillation can transfer is the graded structure over the candidate set rather than its top element. What the deployed projector must do is retrieve relevant motions, and it does: over 4,384 held-out captions against the full database, ϕ recovers the groundtruth motion of a held-out caption at R@1 12.4, R@5 24.9 and R@10 34.6 directly in the latent key space, still enough to drive the best generation quality in this table. Regression is the extreme case: caption-to-motion correspondence is many-to-many, so a hard single target induces label noise, and reproducing individual teacher decisions can come at the cost of the geometry around them; InfoNCE improves on regression but still supervises one positive per caption. The size of the regression failure needs an explanation of its own, since 0.372 lies far outside the range that any inference-time perturbation of the deployed model spans in Table 7. It is a collapse of the retrieved neighborhood rather than a ranking error: the MSE projector’s queries concentrate into a small region of the key space, and across the held-out captions its retrieved sets cover only 6.2% of the distinct database entries the deployed projector reaches. Nearly every prompt is then conditioned on the same handful of exemplars, which is far more destructive than the uniformly random neighbors of Table 7: those at least vary from prompt to prompt and leave the conditioning signal merely uncorrelated with the text, rather than constant across the dataset. The collapse is partial rather than total: top-1 agreement with the teacher remains at 0.048. The last row varies the teacher rather than the objective, and the same conclusion holds along that second axis: distilling the identical KL objective from the external TMR retriever [23] reproduces that teacher’s own top-1 decisions at 0.044, yet generates worse, at FID 0.071. What carries over into generation is therefore how well the teacher ranks in the first place, independent of how faithfully the student reproduces the teacher’s own decisions. The database remains in the $z _ { e }$ space in that row, so the retrieved evidence stays indomain and only the induced neighborhoods change: TMR ranks text–motion pairs less accurately than HBM on this benchmark (Table 3). Read together with Table 5, a weaker ranker operating in the generator’s latent space (0.071) is still far better than the strongest ranker operating in the semantic space (0.109), consistent with our central claim that representation consistency between retrieval and generation matters more than the quality of the ranker itself; within the latent space, replacing this weaker teacher with HBM further improves FID from 0.071 to 0.042. This is where HBM earns its place in ReMoMask-2: as the teacher whose graded ranking the projector inherits, distinct from the deployed retriever role reported in Table 3, so that within this distillation pipeline, the teacher’s retrieval accuracy (Table 3) contributes to generation quality rather than remaining a disconnected number. We therefore read TC@1 as a diagnostic of alignment rather than as a model-selection criterion, and select both the distillation objective and the teacher by downstream generation quality.

TABLE 11: Contribution of the residual-refinement stage across systems on HumanML3D. ∆ is the FID change from adding the residual stage. The MoMask and conferenceversion ReMoMask +Residual entries are the same values as in Table 4.
<table><tr><td>System</td><td>Mask-only</td><td>+ Residual</td><td> $\Delta$ </td></tr><tr><td>MoMask</td><td> $\overline { { 0 . 0 8 5 ^ { \pm . 0 0 3 } } }$ </td><td> $\overline { { 0 . 0 4 6 ^ { \pm . 0 0 2 } } }$ </td><td>-0.039</td></tr><tr><td>ReMoMask</td><td> $0 . 1 4 3 ^ { \pm . 0 0 5 }$ </td><td> $0 . 1 2 3 ^ { \pm . 0 0 3 }$ </td><td>-0.020</td></tr><tr><td>ReMoMask-2 (Ours)</td><td> $\mathbf { 0 . 0 4 2 ^ { \pm . 0 0 2 } }$ </td><td> $0 . 0 6 8 ^ { \pm . 0 0 3 }$ </td><td>+0.026</td></tr></table>

TABLE 12: Reconstruction quality of the frozen 2D RVQ-VAE as a function of the number of decoded quantization layers, on the HumanML3D test set. The tokenizer is the pretrained model of [15]; these numbers are re-measured by us rather than quoted, using the same feature extractor, the same reference statistics and the same set of test motions as the generation FIDs of Table 4, so that reconstruction and generation FID are on a common scale. Reconstruction is deterministic, so no repetition-based confidence intervals are reported.
<table><tr><td>Decoding</td><td>Recon. FID↓</td><td>MPJPE (mm)↓</td></tr><tr><td>Base layer only (used by ReMoMask-2)</td><td>0.064</td><td>36.8</td></tr><tr><td>2 layers</td><td>0.027</td><td>28.4</td></tr><tr><td>3 layers</td><td>0.014</td><td>23.1</td></tr><tr><td>All 6 layers</td><td>0.005</td><td>16.1</td></tr></table>

Single-stage sufficiency. Tables 11 and 12 together explain why ReMoMask-2 omits the residual-refinement stage. For prior two-stage systems the residual stage is load-bearing: it improves MoMask by 0.039 FID and the conferenceversion ReMoMask by 0.020. ReMoMask-2’s single masktransformer stage, decoding only the base quantization layer, already reaches 0.042, surpassing both the conferenceversion ReMoMask full pipeline (0.123) and MoMask’s two-stage result (0.046). Re-attaching a residual-refinement transformer to that stage actively hurts: FID moves from 0.042 to 0.068, a degradation roughly an order of magnitude larger than the confidence intervals involved. Reconstruction FID falls from 0.064 at the base layer to 0.005 with all six layers (Table 12); the returns diminish with depth, as the first residual layer removes the largest share of the baselayer quantization error and each further layer a smaller one. Our generated motions therefore score better (0.042) than the base-layer reconstructions through which they are decoded (0.064). The two numbers measure different things: FID is a distance between distributions rather than a persample fidelity, so the systematic quantization bias shared by all base-layer reconstructions displaces the reconstructed distribution as a whole, whereas the generator, trained to match the data distribution in token space, absorbs part of that displacement. The depth curve reads the same way: adding a single residual layer cuts reconstruction FID by 58% but MPJPE by only 23%, consistent with one residual code already removing most of the shared distributional offset while per-sample error can only be refined code by code. The reconstruction floor thus bounds per-sample accuracy specifically, leaving distributional quality unconstrained by it, and the depth curve marks per-sample-accuracy headroom that a deeper decoding path could still unlock— most of it already within reach at two layers—though, as Table 11 shows, re-attaching a residual transformer forfeits that headroom instead of reaching it.

## 5.5 Qualitative Results

Fig. 8 illustrates ReMoMask-2’s capability in generating diverse human motions. The 16 randomly inferred samples exhibit complex motion patterns such as directional transitions (”walks toward the front, turns to the right”), rhythmic actions (”raises arms three times”), and semantically rich behaviors (”pretending to be a chicken”), suggesting that the model captures nuanced motion dynamics and temporal transitions. Fig. 9 provides a comparative analysis of ReMoMask-2 against the conference-version ReMoMask, MoGenTS, TMR, and ReMoDiffuse. While baseline models generate basic motions like walking or balancing, our approach produces transitions that appear more natural (e.g., ”walks forwards and then stops to take a rest” vs. simple linear motion) and physically plausible motion sequences (e.g., ”walks forward in a clumsy way”); Sec. A reports a controlled user study quantifying motion quality and textmotion correspondence across methods.

## 6 LIMITATIONS

The effectiveness of the proposed retrieval-augmented framework depends on the match between the motion database and the distribution a query is drawn from: when relevant motions are absent or sparsely represented in the retrieval corpus, the benefit that retrieval provides diminishes accordingly. In addition, ReMoMask-2 decodes only

![](images/1f6aa9a9a4519cde2aee2c79c58a98a91d248e2e09d114a69e32cbb589f13e09.jpg)

![](images/5440d9db986a1e8da0ef947bfe0ca2b7af967f5d05b0c6378a9ba37d5368d0c7.jpg)  
a man is pretending to be a chicken. constantly pecking at the ground and waving his arms like a chiken.

![](images/f056fc179c0699f659b3623e8b3590eef32833ac9975823aa5289d23d7da4aa0.jpg)  
a man is walking forward, favoring his left leg and shifting his walk. he is possibly drunk.  
a person jumps forward three times and then walks a few steps.

![](images/1778c718859e88aec133f1c61a271cf15ba85d94712e8979ece7fe3088d7a634.jpg)  
a person jumps forward with both legs and the continues walking until reaching the other side.

![](images/8e190f6966aaa36769cf13a1634ce7d1e507799fde03450945c08c904fb94f2e.jpg)  
a man kicks something or someone with his left leg.

![](images/342cb584b322a0f6df32a33272b8d86ee0f9c40b9c8309c03bc9deb8f3352b1e.jpg)  
a person is looking around, turns to the left, then looks around again.

![](images/1c3a16a9402c1e236f9c5d25d5aea2bba065af2226427e79388ab9ee7781041a.jpg)  
a person walks toward the front, turns to the right, bounces into a squat , and places both arms in front.

![](images/1d9256fd7ef60d6975cfe360c3532407e4b3f02d46e5500af5f737ccb3dc78fd.jpg)  
a person raises its arms then puts them back down 3 times.

![](images/2592fcef9777cbca2e96d0835f2aed385a186d6e29f86c8eb89b6a653abf65b7.jpg)  
a person is walking on a circle.

![](images/748d6f886d580da0a7b635f1b25803f884ee40215b6e80035193e6cc97c1395e.jpg)  
a person walks slowly forward then toward the left hand side and stands facing that direction.

![](images/191f7b15e8eaacc2e49f1c09df5efa503e091246fa1fe53ab761e7fb1af14118.jpg)  
a person raises there arms towards there shoulders.

![](images/5f8b41ff933a08784d9ecc4fd11665198a46bb89a7cb3e0936a0ace9ce096970.jpg)  
person lifts their hands up twice on the same spot.

![](images/33205f5d47ddd8091c598e93e7688bb72b034df67701a7d76666d6a9c2a18e7b.jpg)  
a person walks forward then turns to the right and continues to walk.

![](images/c144174be540ac03422410391b7512437295484de7aeed2a0b237fc51739780f.jpg)  
a person walks to the right in a partial circle.

![](images/7bdd9fcc45af301828ff534b6be1e1099ad18e36c7acb574a29b2f4a601ed861.jpg)  
person turns one direction then other direction standing feet apart and arms side to side.

![](images/c1eaa8ce3fa7ea2f5b4fc41945180183065a3e2b2303f54a11e7850aa5e36014.jpg)  
someone working on the construction site.

Fig. 8: We randomly sample and visualize 16 motions generated by the proposed ReMoMask-2 framework. These examples are conditioned on diverse prompts randomly selected from the HumanML3D [33], providing qualitative evidence of the model’s ability to synthesize a wide range of realistic and semantically coherent motions.

the base quantization layer of the tokenizer, so its persample precision is bounded by base-layer reconstruction quality, leaving the finer detail carried by the deeper quantization layers untapped. Extending latent-aligned retrieval to databases whose domain differs from the target one, detecting when a query falls outside the covered distribution, and unlocking the deeper decoding path within a single generative stage remain important directions for future work.

## 7 CONCLUSION

In this work, we present ReMoMask and its extension ReMoMask-2, retrieval-augmented masked generative frameworks for text-to-motion generation. To overcome the limitations of coarse-grained retrieval and ineffective fusion, we introduce Hierarchical Bidirectional Momentum (HBM) for precise text-motion alignment and Topology Structured Masking (TSM) to enforce structural consistency during training. Complemented by Semantic Spatial-Temporal Attention (SSTA) for knowledge integration, ReMoMask ef-

a man walks forwards and then stops to take a rest.

ReMoMask-2 a man is balancing on something.

![](images/25c4e00efe5a4d000d2103c83fa2234896361d920374f2a5768302037eec96c4.jpg)

![](images/78cd845e87e7f411a6498e7ce9cde3b89bfcaa984a5c41465dfad10d374dcaf6.jpg)

a man walks forward in a clumsy way.

a man walks forward rather slowly.

![](images/6d2e64239f74815efe0c79e90bb3e87d0b04777b41b5d236953be951befb91c5.jpg)

![](images/125c5248b6334a26d37747173f7ba1725aac205f40aee8b3752563bc9e3ea57d.jpg)

ReMoMask

![](images/4ed3a21dc268a3d3fccb9a7523ad3ea21dd575cd9c8a3e064f7fbdee337b812e.jpg)

![](images/b648f06d51b68d39e0525afa62ce762a47795a0a22343eefd6af29262ff41046.jpg)

![](images/f8252574fa11e7f993f75aabffd00b5e9748d4f7ab34bd9173bcb4e8cf6d6461.jpg)

![](images/ed4a8d2fc77c23bb1d040632124c2900d9a60ef0db996a4cb4bf0b3f2e199e1d.jpg)

MoGenTS

![](images/0a4bb747922541e72f6af7938f4540c32adab8d8dee5ae003885a095d90c8476.jpg)

![](images/af20369015e619c9d6cdeca284a1068c6deacfbe9d92ad40e4d042834aa87380.jpg)

![](images/3da60892fbbe0bd2658d399764893304d9b5eef8753c15d6ba20d5c0a11f173d.jpg)

TMR

![](images/58ad8b28374fb075b54dd63dca9e6cd7dcb3d69a27c9b9ac587685f748df21fa.jpg)

![](images/485f11b46df93a530ca3dde58591d8f7d652f4513eee6110c31c57b3fa3ede52.jpg)

![](images/833f7f1f5322aac1ac73f72933251dd4c7103b6bb7c0fe227888261216f7447c.jpg)

![](images/c786ba627dc9c78b6a5c4db1cfc12eafa4768436837878ca55521483ee30d454.jpg)

ReMoDiffuse

![](images/af51d028b7519c1acece81bd48990c29e7dcbb7d18055c64c6b9c790af0c5527.jpg)

![](images/53b01ec17c55a73dfc8bb279f7de0c493f9afbb1695407ab64a161037a404998.jpg)

![](images/1c85512a51c4fad2acb5b9710cbd328cce58d38ad8a50624afb9f83c0640e055.jpg)

![](images/a2a04869fa438496e19786767fff286fe484041d53be6529703acd962bd491cd.jpg)

Fig. 9: Comparison of the proposed ReMoMask-2 with ReMoMask and three state-of-the-art methods: MoGenTS [15], TMR [23], and ReMoDiffuse [16]. We visualize motion sequences generated in response to four distinct text prompts. Each row represents the output of a different method, and each column corresponds to a specific prompt. The results demonstrate that ReMoMask-2 produces more realistic and semantically aligned motions compared to existing approaches.

TABLE 13: Notations and symbols used in ReMoMask.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $_ x$ </td><td>Input text prompt</td></tr><tr><td> $m$ </td><td>Motion sequence</td></tr><tr><td> $m ^ { k }$ </td><td>Motion of the k-th body part</td></tr><tr><td> $t _ { i }$ </td><td>Text embedding of the i-th sample</td></tr><tr><td> $g _ { i }$ </td><td>Global motion embedding</td></tr><tr><td> $p _ { i , k }$ </td><td>Part-level motion embedding for the k-th body part</td></tr><tr><td> $R _ { m }$ </td><td>Retrieved motion embedding</td></tr><tr><td> $R _ { t }$ </td><td>Retrieved text embedding</td></tr><tr><td> $z$ </td><td>Latent motion tokens arranged on  $\textsf { a } T \times J$  spatial-temporal grid</td></tr><tr><td> $z _ { \mathrm { p r e d } }$ </td><td>Reconstructed latent motion tokens predicted by the generator</td></tr><tr><td> $h _ { \mathrm { { s e m } } }$ </td><td>Global semantic token constructed from text and retrieved context</td></tr><tr><td> $Q , K , V$ </td><td>Query, Key, and Value matrices in the attention module</td></tr><tr><td> $\alpha _ { i , k }$ </td><td>Semantic relevance between text and the k-th body part</td></tr><tr><td> $\pi _ { \mathrm { b a s e } }$ </td><td>Base masking probability</td></tr><tr><td> $\pi _ { i , k }$ </td><td>Masking probability of the k-th body part</td></tr><tr><td> $\mathcal { Q }$ </td><td>Momentum queue storing negative embeddings</td></tr><tr><td> $\mu$ </td><td>Momentum coefficient for updating momentum encoders</td></tr><tr><td> $\tau$ </td><td>Temperature parameter in contrastive learning</td></tr><tr><td> $\lambda _ { P }$ </td><td>Weighting coefficient for the part-level contrastive loss</td></tr><tr><td> $B$ </td><td>Batch size</td></tr><tr><td> $K$ </td><td>Number of body parts</td></tr><tr><td> $T$ </td><td>Number of temporal frames</td></tr><tr><td> $J$ </td><td>Number of body joints</td></tr><tr><td> $N$ </td><td>Length of flattened motion tokens  $( N = T \times J )$ </td></tr><tr><td> $d$ </td><td>Dimension of the latent embedding space</td></tr><tr><td> $\mathrm { M L P } ( \cdot )$ </td><td>Multilayer perceptron</td></tr><tr><td> $\mathrm { c o n c a t } ( \cdot )$ </td><td>Concatenation operation</td></tr><tr><td>flatten(·)</td><td>Flattening a 2D latent grid into a 1D sequence</td></tr><tr><td> $s , z$ </td><td>Contrastive semantic space; generative latent space</td></tr><tr><td> $\psi$ </td><td>Implicit cross-space translation  $s \to z$  in ReMoMask</td></tr><tr><td> $\varepsilon$ </td><td>Frozen 2D  $R \mathrm { V } \dot { Q } \mathrm { - } \mathrm { V } \mathrm { A E }$  encoder producing the pre-quantization latent</td></tr><tr><td> $z _ { e }$ </td><td>Pre-quantization continuous latent of the frozen RVQ-VAE encoder</td></tr><tr><td> $\bar { z } _ { e }$ </td><td>Pooled, l2-normalized  $z _ { e }$  used as a retrieval key</td></tr><tr><td> $d _ { e }$ </td><td>Channel dimension of  $z _ { e } ( d _ { e } = 1 0 2 4 )$ </td></tr><tr><td>D</td><td>Latent-aligned retrieval database of  $( \bar { z } _ { e } , x )$  pairs</td></tr><tr><td> $\phi$ </td><td>Query projector mapping CLIP text embeddings into the ze space</td></tr><tr><td> $\kappa$ </td><td>Number of teacher candidates in distillation (top-κ)</td></tr><tr><td> $\Omega ( x )$ </td><td>Top-κ teacher candidate set for caption x</td></tr><tr><td> $p ^ { \mathrm { H B i M } } , p ^ { \phi }$ </td><td>Teacher / student retrieval distributions in KL distillation</td></tr><tr><td> $\mathcal { L } _ { \mathrm { a l i g n } }$ </td><td>KL distillation objective for the query projector</td></tr><tr><td> $t$ </td><td>CLIP text embedding of a single prompt (single-sample form of ti)</td></tr><tr><td> $T ^ { \prime } , J ^ { \prime }$ </td><td>Temporal / spatial size of the downsampled  $z _ { e }$  grid  $\stackrel { \cdot } { ( } T ^ { \prime } { = } T / 4 , J ^ { \prime } { = } 6 )$ </td></tr><tr><td> $s _ { j } ^ { \mathrm { H B M } } , s _ { j } ^ { \phi }$ </td><td></td></tr><tr><td></td><td>Teacher / student similarity scores over candidate j</td></tr></table>

fectively bridges structured retrieval with high-quality generation. Building on this framework, ReMoMask-2 relocates retrieval into the generator’s own pre-quantization latent space, closing the representation gap between retrieved evidence and the generative latents; a graded conditioning analysis confirms that the deployed model genuinely consumes the retrieved semantics rather than merely registering their presence. With retrieval acting in this shared space, a single mask-transformer stage, without a separate residual-refinement network, surpasses the accuracy of the full two-stage pipeline while reducing inference cost, yielding a system that is both more accurate and more efficient. Extensive experiments on HumanML3D, KIT-ML, and SnapMoGen confirm that our retriever delivers state-of-theart text-to-motion retrieval, and that ReMoMask-2 consistently achieves the best generation fidelity among retrievalaugmented approaches, with the lowest FID among all compared methods on KIT-ML and SnapMoGen, while requiring only a single generative stage.

Acknowledgements. This work was supported by the Fundamental Research Funds for the Central Universities, Peking University.

## APPENDIX

We list notations and symbols used in this paper, as shown in Table 13.

To comprehensively evaluate the generation capability of ReMoMask, we conducted a comparative user study. We randomly selected 20 text prompts from the HumanML3D test set and generated motion sequences using ReMoMask, current state-of-the-art retrieval-augmented method (ReMoDiffuse), generative model (MoMask), and ground truth motions.

We employ a forced-choice paradigm in our user study, asking participants two key questions: “Which of the two motions is more realistic?” and “Which of the two motions corresponds better to the text prompt?”. The study is conducted via a Google Forms interface, as illustrated in Fig. 13. To ensure fairness and reduce potential bias, the names of the generative models are hidden, and the order of presentation is randomized for each question. In total, over 50 participants took part in the evaluation.

Empirical results, depicted in Fig. 10 and Fig. 11, underscore ReMoMask’s strong capability to generate motions that are not only realistic but also closely aligned with textual descriptions. Specifically, as shown in Fig. 10, ReMoMask achieves a 42% preference rate over ground truth (GT) in terms of realism. Although GT motions are derived from real human data, this result indicates that ReMoMask is perceived as comparably realistic by human evaluators. Moreover, the model significantly outperforms both baselines: it achieves 67% preference over MoMask and 75% over ReMoDiffuse, demonstrating its strength in producing high-quality, lifelike motion sequences.

In terms of text correspondence (reported in Fig. 11), ReMo-Mask attains a 47% preference rate over GT, suggesting that its generated motions exhibit nearly human-level alignment with text prompts. Compared to the baselines, ReMoMask again shows substantial improvements, with 72% preference over MoMask and 86% over ReMoDiffuse.

![](images/2bfb1418e7cc7fb271c05e3be7299ba44f50a7695b946511540ec3b8e85cc5fc.jpg)  
Fig. 10: Motion Quality User Study

![](images/03d5f7112501f0559baace5c257196baeffceedf3ec1b12f6b0432ba37b22673.jpg)  
Fig. 11: Text-Motion Correspondence User Study

Video Demonstrations. We provide video demonstrations of our generated motions to facilitate qualitative evaluation. Fig. 12 shows representative video samples, where our method produces temporally coherent and semantically aligned motion sequences.

![](images/5154c1af2b5fc96d302c3ee97f74692e578878387f4e54c1f61cd901645f42b9.jpg)  
Fig. 12: Video sample.

![](images/509c1a0d2df2b6a0d5fadd092fae77e1aad4378fb1207a97339cd0c68f9bdc13.jpg)

Fig. 13: This figure illustrates the User Interface (UI) used in the ReMoMask User Study. Participants are presented with two motion videos, labeled as Motion A and Motion B, alongside a shared textual prompt. The motion clips are sampled from outputs generated by different models or the ground truth (GT), with model identities anonymized and video order randomized. Participants are asked to answer two evaluative questions: (1) “Which of the two motions is more realistic?”, assessing the visual plausibility and motion quality; and (2) “Which of the two motions corresponds better to the text prompt?”, evaluating the semantic alignment between the motion and the given description. This dual-question design enables a comprehensive human assessment of both motion realism and text-motion correspondence.

## REFERENCES

[1] Z. Zhang, Y. Wang, B. Wu, S. Chen, Z. Zhang, S. Huang, W. Zhang, M. Fang, L. Chen, and Y. Zhao, “Motion avatar: Generate hu-

man and animal avatars with arbitrary motion,” arXiv preprint arXiv:2405.11286, 2024. 1

[2] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” 2014. [Online]. Available: https://arxiv. org/abs/1406.2661 1

[3] J. Dong, P. Koniusz, X. Qu, and Y.-S. Ong, “Stabilizing modality gap & lowering gradient norms improve zero-shot adversarial robustness of vlms,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, 2025, pp. 236–247. 1

[4] J. Dong, C. Zhang, X. Qu, Z. Ma, P. Koniusz, and Y.-S. Ong, “Robust superalignment: Weak-to-strong robustness generalization for vision-language models,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 1

[5] J. Dong, P. Koniusz, Y. Zhang, H. Zhu, W. Liu, X. Qu, and Y.- S. Ong, “Improving zero-shot adversarial robustness in visionlanguage models by closed-form alignment of adversarial path simplices,” in Forty-second International Conference on Machine Learning, 2025. 1

[6] J. Dong, J. Liu, X. Qu, and Y.-S. Ong, “Confound from all sides, distill with resilience: Multi-objective adversarial paths to zero-shot robustness,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 624–634. 1

[7] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” 2022. [Online]. Available: https://arxiv.org/abs/1312.6114 1

[8] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” 2020. [Online]. Available: https://arxiv.org/abs/2006. 11239 1

[9] Y. Li, J. Yang, Z. Yang, B. Li, H. He, Z. Yao, L. Han, Y. V. Chen, S. Fei, D. Liu et al., “Cama: Enhancing multimodal in-context learning with context-aware modulated attention,” arXiv preprint arXiv:2505.17097, 2025. 1

[10] Y. Zhang, Y. He, Y. Shao, Z. Yao, H. Xu, J. Dong, Z. Yao, and Z. Dong, “Chromouvqa: Benchmarking vision-language models under chromatic camouflaged images,” arXiv preprint arXiv:2512.05137, 2025. 1

[11] B. Jiang, X. Chen, W. Liu, J. Yu, G. Yu, and T. Chen, “Motiongpt: Human motion as a foreign language,” 2023. [Online]. Available: https://arxiv.org/abs/2306.14795 1, 3

[12] E. Pinyoanuntapong, P. Wang, M. Lee, and C. Chen, “Mmm: Generative masked motion model,” 2024. [Online]. Available: https://arxiv.org/abs/2312.03596 1, 3

[13] C. Guo, Y. Mu, M. G. Javed, S. Wang, and L. Cheng, “Momask: Generative masked modeling of 3d human motions,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 1900–1910. 1, 2, 3, 8, 9, 10

[14] C. Guo, I. Hwang, J. Wang, and B. Zhou, “Snapmogen: Human motion generation from expressive texts,” 2025. [Online]. Available: https://arxiv.org/abs/2507.09122 1, 3, 7, 9

[15] W. Yuan, W. Shen, Y. HE, Y. Dong, X. Gu, Z. Dong, L. Bo, and Q. Huang, “Mogents: Motion generation based on spatialtemporal joint modeling,” in Neural Information Processing Systems (NeurIPS), 2024. 1, 6, 8, 9, 10, 14, 16

[16] M. Zhang, X. Guo, L. Pan, Z. Cai, F. Hong, H. Li, L. Yang, and Z. Liu, “Remodiffuse: Retrieval-augmented motion diffusion model,” arXiv preprint arXiv:2304.01116, 2023. 2, 3, 6, 9, 16

[17] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763. 2, 7, 10

[18] Q. Yu, M. Tanaka, and K. Fujiwara, “Remogpt: Part-level retrievalaugmented motion-language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 9635– 9643. 2, 3, 5, 6, 8, 9

[19] G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-or, and A. H. Bermano, “Human motion diffusion model,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=SJ1kSyO2jwu 2, 9

[20] J. Zhang, Y. Zhang, X. Cun, S. Huang, Y. Zhang, H. Zhao, H. Lu, and X. Shen, “T2m-gpt: Generating human motion from textual descriptions with discrete representations,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 3, 9

[21] Z. Meng, Y. Xie, X. Peng, Z. Han, and H. Jiang, “Rethinking

diffusion for text-driven human motion generation,” arXiv preprint arXiv:2411.16575, 2024. 2, 9

[22] Z. Li, W. Yuan, Y. He, L. Qiu, S. Zhu, X. Gu, W. Shen, Y. Dong, Z. Dong, and L. T. Yang, “Lamp: Language-motion pretraining for motion generation, retrieval, and captioning,” 2025. [Online]. Available: https://arxiv.org/abs/2410.07093 2, 9

[23] M. Petrovich, M. J. Black, and G. Varol, “Tmr: Text-to-motion retrieval using contrastive 3d human motion synthesis,” 2023. [Online]. Available: https://arxiv.org/abs/2305.00976 2, 3, 8, 13, 14, 16

[24] S. S. Kalakonda, S. Maheshwari, and R. K. Sarvadevabhatla, “Morag – multi-fusion retrieval augmented generation for human motion,” 2024. [Online]. Available: https://arxiv.org/abs/2409. 12140 2, 3, 6, 9

![](images/741691b3a82dbba265c730334ff1b6d051bc729d214158129bb14fa5620c3768.jpg)

[25] M. Petrovich, M. J. Black, and G. Varol, “Temos: Generating diverse human motions from textual descriptions,” 2022. [Online]. Available: https://arxiv.org/abs/2204.14109 2, 3, 8

[26] Q. Zou, S. Yuan, S. Du, Y. Wang, C. Liu, Y. Xu, J. Chen, and X. Ji, “Parco: Part-coordinating text-to-motion synthesis,” 2024. [Online]. Available: https://arxiv.org/abs/2403.18512 2, 3, 5

[27] Z. Li, S. Wang, Z. Zhang, and H. Tang, “Remomask: Retrievalaugmented masked motion generation,” in Proceedings of the European Conference on Computer Vision (ECCV), 2026, arXiv:2508.02605. 2, 3, 9, 11

[28] C. Guo, X. Zuo, S. Wang, and L. Cheng, “Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts,” in European Conference on Computer Vision. Springer, 2022, pp. 580–597. 3

[29] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” 2022. [Online]. Available: https://arxiv.org/abs/2010. 02502 3

[30] Z. Liao, M. Zhang, W. Wang, L. Yang, and T. Komura, “Rmd: A simple baseline for more general human motion generation via training-free retrieval-augmented motion diffuse,” 2024. [Online]. Available: https://arxiv.org/abs/2412.04343 3, 9

[31] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” 2015. [Online]. Available: https://arxiv.org/ abs/1503.02531 7

![](images/39704b06d1077c58ec9e8615d67e90ca6d4d8dcf17e4fc10ea79b3fbaad83c77.jpg)

[32] Q. Yu, M. Tanaka, and K. Fujiwara, “Exploring vision transformers for 3d human motion-language models with motion patches,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 8

[33] C. Guo, S. Zou, X. Zuo, S. Wang, W. Ji, X. Li, and L. Cheng, “Generating diverse and natural 3d human motions from text,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022, pp. 5152–5161. 7, 8, 10, 15

[34] M. Plappert, C. Mandery, and T. Asfour, “The KIT motionlanguage dataset,” Big Data, vol. 4, no. 4, pp. 236–252, dec 2016. [Online]. Available: http://dx.doi.org/10.1089/big.2016.0028 7

![](images/fa1fa9795238643aa2ccd55b01b5919859b8cec3699f8180970ff04a3866d512.jpg)

Ling Shao (Fellow, IEEE) is a Distinguished Professor with the University of Chinese Academy of Sciences, Beijing, China. He was the founder of the Inception Institute of Artificial Intelligence (IIAI) and the Mohamed bin Zayed University of Artificial Intelligence (MBZUAI), Abu Dhabi, UAE. His research interests include physical AI, multimodal AI, and AI for healthcare. He is a fellow of the IEEE, the IAPR, the BCS and the IET.

Zeyu Zhang is a researcher working on generative AI, with a particular interest in building models that understand and interact with the physical world. He received his bachelor’s degree from the Australian National University, where he was advised by Prof. Richard Hartley and Prof. Ian Reid. His research explores generative modeling for learning physical dynamics from visual data. His work spans world models, multimodal foundation models, embodied AI, and AI for health.

Yiran Wang is a PhD student at the University of Sydney, Australia, advised by Dr Viorela Ila. He received his bachelor’s and master’s degrees from the University of Sydney. His research interests include computer vision and embodied AI, with a focus on human motion generation, retrieval-augmented generation, multimodal representation learning, and robotics.

![](images/32f80ca8d91b289fe4e026b27714affde3909b55175e4976ebd723043165dff3.jpg)

Hao Tang is an Assistant Professor at Peking University, China. Previously, he held postdoctoral positions at CMU, USA, and ETH Zurich,¨ Switzerland. He earned his master’s degree from Peking University, and his Ph.D. from the University of Trento, Italy. He has had the opportunity to visit the University of Oxford, Northeastern University, NUS, and IIAI, among other institutions. His research interests include computer vision, generative AI, spatial intelligence, world model, and embodied AI.