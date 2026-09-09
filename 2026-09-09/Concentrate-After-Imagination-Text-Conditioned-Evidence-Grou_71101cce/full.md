# Concentrate After Imagination: Text-Conditioned Evidence Grounding for Partially Relevant Video Retrieval

Shuaiqi Cheng<sup>1,2</sup>, Siyu You<sup>2</sup>, Yanbi Wu<sup>1</sup>, Yuxi Chen<sup>2</sup>, Jiahao Zhang<sup>1</sup>, Xuming Hu<sup>1,3∗</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup>University of Electronic Science and Technology of China

<sup>3</sup>The Hong Kong University of Science and Technology

scheng512@connect.hkust-gz.edu.cn, xuminghu97@gmail.com

## Abstract

Partially Relevant Video Retrieval (PRVR) retrieves untrimmed videos when queries describe only short moments. Although recent methods improve local representations, uncertainty modeling, and global context, final ranking often still trusts the strongest local response; a coincidentally similar fragment can therefore produce an unsupported peak. We identify this failure as the query-agnostic concentration bottleneck and propose TRACE, a score-level evidence verification operator for PRVR. Given a query and global video registers, TRACE activates query-relevant registers, routes their support to frame-level evidence, and smoothly marginalizes alternative q→R→F paths before localized temporal selection. Unlike representation-level feature fusion, TRACE uses this evidence only as a query-conditioned residual calibration of the original local score. On ActivityNet Captions, Charades-STA, and TVR, TRACE achieves the best SumR on all three benchmarks and improves the DreamPRVR backbone by +1.2, +1.1, and +1.5, respectively. Ablation, routingcorruption, hard-negative, and cross-backbone transfer analyses support the interpretation that the gains arise from queryconditioned evidence verification rather than a generic score ofset.

## Introduction

Text-to-Video Retrieval (T2VR) aims to retrieve videos relevant to a natural-language query from a large collection. Conventional T2VR benchmarks often assume that the query describes most or all of a short, trimmed clip (Luo et al. 2022; Wu et al. 2023). Real-world videos, however, are typically long and untrimmed, while a user may refer to only a brief event amid substantial irrelevant content. Partially Relevant Video Retrieval (PRVR) (Dong et al. 2022) addresses this setting: a video is considered relevant if it contains at least one temporal moment that matches the query.

PRVR exposes a tension between local concentration and reliable ranking. Because a query may match only a brief segment, the model must remain sensitive to local evidence; multiple-instance learning and max-style pooling over frame or clip similarities are therefore natural choices (Dietterich, Lathrop, and Lozano-Pérez 1997; Dong et al. 2022). Yet the same mechanism makes ranking brittle: a negative video can be promoted by a coincidentally similar fragment. Reliable

PRVR scoring must therefore answer two questions jointly: which local response matches the query, and whether that response is corroborated by query-relevant global evidence from the video.

Figure 1 contrasts direct local selection with TRACE’s query-conditioned register activation and register-to-frame grounding. In the upper branch, a visually similar fragment can dominate even when it is unrelated to the query’s global semantics; in the lower branch, only peaks supported by the activated registers are emphasized. The score curves are schematic rather than additional quantitative results: TRACE changes which peak is trusted while preserving localized temporal selection.

Recent PRVR research improves temporal modeling (Dong et al. 2022; Wang et al. 2024b), semantic alignment (Jun et al. 2025; Li et al. 2025), uncertainty estimation (Cho et al. 2025; Li et al. 2026b), and global context (Li et al. 2026c). Yet richer representations do not guarantee reliable final concentration: the ranking function may still select a local peak without checking whether it is supported by query-relevant global context. We call this the query-agnostic concentration bottleneck. Reliable scoring should preserve localized selection, select global evidence conditioned on the query, and ground that evidence back to temporal responses before concentration. Raw max pooling satisfies only the first requirement, while representation-level register fusion (Li et al. 2026c) enriches visual tokens without explicitly verifying the winning response. TRACE therefore verifies a query-selected global-to-local path before concentration.

We propose Text-Conditioned Register Activation and Cross-Granular Evidence Routing (TRACE), a hierarchical evidence concentration operator for PRVR. Query– register afinities activate relevant registers, register–frame compatibilities ground them temporally, and smooth logsum-exp marginalization combines competing q→R→F paths before localized maximization. The asymmetric design smooths alternative semantic explanations while retaining a hard temporal choice, preserving partial relevance while reducing unsupported peaks.

The distinction from ordinary feature fusion is at the ranking boundary. TRACE does not replace the backbone’s learned representation or ask a new global embedding to dominate the score. Instead, it uses the query to decide which global anchors should support each candidate temporal response, and adds this verification signal through a small residual term. This makes the method compatible with register-based PRVR backbones while keeping the original local matching score as the primary retrieval signal. On ActivityNet Captions, Charades-STA, and TVR, TRACE achieves the highest SumR among compared methods, improving DreamPRVR (Li et al. 2026c) by +1.2, +1.1, and +1.5. Ablation, routing-corruption, transfer, and hardnegative diagnostics support the verification-based explanation and distinguish the proposed routing from a generic score ofset.

![](images/90427fca3a553e8b3c00b45c6dfe6f0c5fd1d2c3912c7174536216c448abe637.jpg)  
Figure 1: (a) Max-style concentration can select a spurious local peak because final scoring does not explicitly verify it with query-relevant global evidence, producing an incorrect rank. (b) TRACE activates relevant registers through q→R routing, grounds them back to temporal evidence through R→F routing, and concentrates only after this cross-granular support is formed. The illustrative score curves show the resulting correction from rank 3 to rank 1.

Our contributions are summarized as follows:

• We identify and empirically diagnose the query-agnostic concentration bottleneck: even when global context is encoded at the representation level, final concentration may still select a local response without query-conditioned global verification.

• We propose TRACE, a hierarchical evidence concentration operator that smoothly marginalizes text-conditioned global paths before localized maximum selection over register-grounded frame evidence.

• Across three PRVR benchmarks, TRACE yields consistent SumR improvements over strong baselines, while routing-corruption, cross-backbone transfer, and hardnegative diagnostics provide evidence that it is most effective on the intended unsupported-peak failure mode.

## Related Work

Local Temporal Evidence Modeling in PRVR. PRVR methods model local temporal evidence to capture queries aligned with short segments. MS-SL (Dong et al. 2022) constructs multi-scale sliding-window representations; GMM-Former (Wang et al. 2024b) and GMMFormerV2 (Wang et al. 2024a) refine temporal evidence through Gaussian-mixture modeling; DLDKD (Dong et al. 2023) transfers knowledge from coarse to fine scales; and ProtoPRVR (Moon et al. 2025) reduces the candidate space using representative prototypes. Together, these approaches improve local evidence construction or search at the temporal-unit level.

Semantic Structure and Uncertainty in PRVR. Semantic alignment and uncertainty methods address ambiguous text–video relations and partial relevance. ARL (Cho et al. 2025), RAL (Zhang et al. 2025a), SDM (Jun et al. 2025), and MSC-PRVR (Moon et al. 2026) improve semantic or uncertainty-aware modeling; HLFormer (Li et al. 2025) captures hierarchical partial relevance in hyperbolic space, while Holmes (Li et al. 2026b) combines evidential learning with optimal transport. These methods primarily improve supervision, representation geometry, or uncertainty estimation; they do not explicitly formulate the final local peak as a query-conditioned verification problem.

Global Context and Register-Based PRVR. Globalcontext methods complement local matching by injecting semantic priors through action-object modeling (Chen et al. 2026b), caption-driven alignment (Chen et al. 2026a), contextual distillation (Yang et al. 2026), and semantic-guided alignment (Li et al. 2026a). Register tokens, first introduced in Vision Transformers for auxiliary global information (Darcet et al. 2024), have also been adopted in multimodal and audio-visual models (Zhang et al. 2025b; Zhu et al. 2025). In PRVR, DreamPRVR (Li et al. 2026c) generates difusion-guided global registers and fuses them into frame and clip tokens. These methods mainly enrich representations before ranking, rather than directly calibrating the final score with query-conditioned register-grounded temporal evidence.

Evidence Aggregation in PRVR. Evidence aggregation converts local responses into a video-level retrieval score. Multiple-instance learning and max-pooling (Dietterich, Lathrop, and Lozano-Pérez 1997; Dong et al. 2022) remain standard; PRVR alternatives include Gaussian temporal aggregation (Wang et al. 2024b), prototype-based compression (Moon et al. 2025), probabilistic relevance modeling (Zhang et al. 2025a), and optimal-transport assignment (Li et al. 2026b). These designs primarily specify how local evidence is pooled, compressed, or assigned, whereas the separate question of whether the selected peak is supported by compatible global evidence is less explicitly studied. This separates evidence aggregation from TRACE’s cross-granular path-consistency verification. At a mathematical level, Gaussian components model temporal response distributions, while a transport plan aligns evidence under mass constraints. TRACE instead computes an unconstrained per-frame energy by log-sum-exp marginalization over $q {  } R {  } F$ paths, then uses it to test whether a local peak is globally supported before temporal max selection.

## Preliminaries

## Problem Formulation

Given a text query Q and an untrimmed video V, Partially Relevant Video Retrieval (PRVR) ranks videos according to whether they contain a local temporal moment relevant to Q. The video branch produces frame embeddings ${ \boldsymbol { F } } ~ = ~ \{ f _ { i } \} _ { i = 1 } ^ { M _ { f } } ~ \in ~ \mathbb { R } ^ { M _ { f } \times d }$ and clip embeddings $C = \{ c _ { j } \} _ { j = 1 } ^ { M _ { c } } \in \mathbb { R } ^ { \bar { M } _ { c } \times d }$ , while the text branch produces a query embedding $q \in \mathbb { R } ^ { d }$ . When a register-based backbone is used, the video branch additionally provides global registers $R = \{ r _ { k } \} _ { k = 1 } ^ { N _ { r } } \in \mathbb { R } ^ { N _ { r } \times d }$

For exposition, the backbone’s original retrieval score can be written as a max-style combination over frame and clip responses:

$$
S _ { \mathrm { b a s e } } ( Q , V ) = \alpha _ { f } \operatorname* { m a x } _ { i } \cos ( q , f _ { i } ) + \alpha _ { c } \operatorname* { m a x } _ { j } \cos ( q , c _ { j } ) ,\tag{1}
$$

where $\alpha _ { f } + \alpha _ { c } = 1$ are the backbone’s original frame–clip aggregation weights and are kept unchanged by TRACE. This maximum-based score fits partial relevance but concentrates on local responses without asking which global evidence is relevant to the current query or whether the selected local peak is supported by it.

## Register-Based PRVR

DreamPRVR (Li et al. 2026c) uses a coarse-to-fine registergeneration pipeline: it first generates holistic video registers and then uses them to enhance fine-grained text–video matching, which can be summarized as the latent-register decomposition

$$
p _ { \theta , \phi } ( Q | V ) = \int \underbrace { p _ { \theta } ( Q | V , R ) } _ { \mathrm { m a t c h i n g } } \underbrace { p _ { \phi } ( R | V ) } _ { \mathrm { r e g i s t e r ~ g e n e r a t i o n } } d R ,\tag{2}
$$

where $R = \{ r _ { k } \} _ { k = 1 } ^ { N _ { r } }$ are global registers generated from V. In practice, the generator produces register tokens that are fused with frame and clip embeddings through registeraugmented attention, improving the representations used for retrieval. TRACE retains this register-based backbone and its representation-level context, while additionally using the registers as global evidence anchors selected by the current query at the scoring boundary. Thus, TRACE complements feature enhancement with score-level verification of the selected local peak.

## Method

## Overview

Given query q, frame tokens $F ,$ clip tokens $C ,$ and registers R, TRACE constructs query-conditioned global-tolocal evidence and integrates it with the base score as $S _ { \mathrm { f i n a l } } = S _ { \mathrm { b a s e } } { + } \lambda S _ { \mathrm { t r a c e } }$ . Here q is produced by the text/query encoder, while $S _ { \mathrm { b a s e } }$ is the original DreamPRVR retrieval branch computed from the encoded local video tokens. TRACE adds a score-level residual, so λ = 0 exactly recovers the backbone score. Figure 2 summarizes the computation, with complete pseudocode provided in the Supplement Algorithm S1. The following subsections detail the latent path view, cross-granular routing, hierarchical concentration, and learning objective.

## Text-Conditioned Latent Evidence Paths

We view the missing scoring step as a text-conditioned latent path. Let z denote a latent global evidence anchor selected by the query, and let e denote a frame-level temporal evidence location. A schematic decomposition is given by

$$
p _ { \theta } ( Q | V , R ) \approx \sum _ { z } p _ { \theta } ( z | Q , R ) \sum _ { e } p _ { \theta } ( e | z , V ) p _ { \theta } ( Q | e , V ) ,\tag{3}
$$

where the expression is an energy-based scoring view rather than a calibrated likelihood. It separates query-conditioned anchor activation, anchor-to-frame grounding, and direct query–frame evidence. Raw max scoring mainly uses the last term, whereas TRACE combines all three before concentration.

## Cross-Granular Evidence Routing

We instantiate the grounding principle with three score-level operations: direct query-to-frame matching, text-conditioned register activation, and register-to-frame routing. Before computing these scores, TRACE applies learned linear projections to the query, register keys, frame keys, and frame values, followed by $\ell _ { 2 }$ normalization. For readability, we reuse $q , r _ { k } ,$ , and $f _ { i }$ for the projected query, register key, and frame key, respectively, and denote the projected frame value by $f _ { i } ^ { v }$

Query-to-Frame Direct Evidence. For each frame token $f _ { i } ,$ , we compute its direct compatibility with the query, $d _ { i } = \cos ( q , f _ { i } )$ , realizing $p _ { \boldsymbol { \theta } } ( Q | e , \bar { V } )$ . By itself, this term can still be distracted by locally similar but globally unsupported frames.

![](images/71085593adca0fe8910f824ffc58542a902fd8882b2bb5d12b1bd1da1f712e03.jpg)  
Figure 2: Overview of TRACE. (a) The overall framework: the video and text encoders produce clip embeddings C, frame embeddings $F _ { ; }$ , registers R, and query embeddings $q ;$ the original base branch computes local matching over C and F, while TRACE adds query-conditioned evidence through residual score fusion. (b) Latent $q {  } R {  } F$ routing paths, (c) register-to-frame compatibility, and (d) the auxiliary InfoNCE and bidirectional triplet objectives.

Text-Conditioned Register Activation. To select global semantic cues relevant to the current text, TRACE computes query-register afinity $a _ { k } = \cos ( q , r _ { k } )$ and normalizes it as a softmax with temperature $\tau _ { r } :$

$$
\pi _ { k } = \frac { \exp ( a _ { k } / \tau _ { r } ) } { \sum _ { \ell = 1 } ^ { N _ { r } } \exp ( a _ { \ell } / \tau _ { r } ) } ,\tag{4}
$$

approximating $p _ { \theta } ( z = k | Q , R )$

Register-to-Frame Routing. Each register is connected to frame-level evidence through register-frame compatibility $b _ { k , i } = \cos ( r _ { k } , f _ { i } )$ . An expectation-style support for frame i can be written as $\begin{array} { r } { \bar { g } _ { i } = \sum _ { k = 1 } ^ { N _ { r } } \pi _ { k } b _ { k , i } } \end{array}$ . Rather than using this difuse average, TRACE aggregates the unnormalized joint path energies $a _ { k } + b _ { k , i }$ through

$$
\rho _ { i } = \tau _ { r } \log \sum _ { k = 1 } ^ { N _ { r } } \exp \left( ( a _ { k } + b _ { k , i } ) / \tau _ { r } \right) .\tag{5}
$$

This log-sum-exp form provides a diferentiable soft maximum over latent $q {  } R {  } F$ paths. It gives high support only to frames compatible with query-relevant registers, so support varies with both query and temporal location. Register paths are smoothed, while temporal selection remains localized.

## Hierarchical Evidence Concentration

TRACE uses two asymmetric aggregation steps: smooth logsum-exp over latent register paths (Eq. 5), followed by hard temporal concentration. Multiple register paths may explain a query, but only a few temporal moments need to be selected.

Routed Evidence Route. We combine direct evidence and register-routed support as

$$
\tilde { d } _ { i } = d _ { i } + \gamma \rho _ { i } ,\tag{6}
$$

where $\gamma$ controls routing strength. Since PRVR relevance may occupy only a few moments, we select the strongest grounded response, $S _ { \mathrm { r o u t e } } ( Q , V ) = \operatorname* { m a x } _ { i } \tilde { d } _ { i } .$ . Thus the core operator performs smooth path marginalization followed by hard temporal selection, making unsupported negative peaks less likely to dominate.

Soft Register Evidence Route. We include an auxiliary soft $q {  } R {  } F$ route to complement the hard temporal selector. Let

$$
u _ { i } = \sum _ { k = 1 } ^ { N _ { r } } \pi _ { k } \frac { \exp ( b _ { k , i } / \tau _ { r } ) } { \sum _ { j = 1 } ^ { M _ { f } } \exp ( b _ { k , j } / \tau _ { r } ) }\tag{7}
$$

denote the query-conditioned register-to-frame attention. We form a soft evidence representation $\begin{array} { r } { v _ { \mathrm { s o f t } } \ = \ \sum _ { i } u _ { i } f _ { i } ^ { v } } \end{array}$ and score it as $S _ { \mathrm { s o f t } } ( Q , V ) \stackrel { - } { = } \cos ( q , v _ { \mathrm { s o f t } } )$ . Unlike $S _ { \mathrm { r o u t e } } .$ , which preserves sparse temporal concentration through a hard maximum, this branch aggregates frame values into a single soft representation before query scoring. It therefore serves as an auxiliary evidence branch that complements, rather than replaces, the core routed selector.

Backbone Integration. The TRACE evidence score combines the core routed concentration score and the auxiliary soft evidence:

$$
S _ { \mathrm { t r a c e } } ( Q , V ) = S _ { \mathrm { r o u t e } } ( Q , V ) + \eta S _ { \mathrm { s o f t } } ( Q , V ) ,\tag{8}
$$

where η weights the auxiliary branch relative to the routed evidence. We then obtain the final retrieval score through residual fusion,

$$
S _ { \mathrm { f i n a l } } ( Q , V ) = S _ { \mathrm { b a s e } } ( Q , V ) + \lambda S _ { \mathrm { t r a c e } } ( Q , V ) .\tag{9}
$$

The progressive schedule for introducing the evidence branch and ramping the efective fusion weight is described in the Implementation Details.

## Model Learning

TRACE retains the backbone objective $\mathcal { L } _ { \mathrm { b a s e } } ,$ including retrieval supervision and register-generation regularization. Following the initial evidence-only warm-up, we jointly optimize the backbone and add $\dot { \mathcal { L } } _ { \mathrm { t r a c e } }$ to make the evidence score discriminative before fusion.

For a mini-batch containing queries $\{ Q _ { i } \} _ { i = 1 } ^ { B }$ and videos $\{ V _ { j } \} _ { j = 1 } ^ { B _ { v } }$ , let $E _ { i j } = S _ { \mathrm { t r a c e } } ( Q _ { i } , V _ { j } )$ denote the TRACE evidence score and $y _ { i }$ the positive-video index for $Q _ { i }$ . The evidence contrastive loss is the standard in-batch InfoNCE:

$$
\mathcal { L } _ { \mathrm { t r a c e } } ^ { \mathrm { n c e } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( E _ { i , y _ { i } } / \tau _ { c } ) } { \sum _ { j = 1 } ^ { B _ { v } } \exp ( E _ { i j } / \tau _ { c } ) } ,\tag{10}
$$

and the auxiliary objective additionally uses a bidirectional hinge-style triplet margin loss $\mathcal { L } _ { \mathrm { t r a c e } } ^ { \mathrm { t r i } } = \mathcal { L } _ { \mathrm { t 2 v } } ^ { \mathrm { t r i } } + \mathcal { L } _ { \mathrm { v 2 t } } ^ { \mathrm { t r i } }$ over positive and negative query-video pairs with margin m. The TRACE auxiliary objective is $\dot { \mathcal { L } } _ { \mathrm { t r a c e } } = \lambda _ { \mathrm { n c e } } \mathcal { L } _ { \mathrm { t r a c e } } ^ { \mathrm { \tilde { n } c e } } +$ $\lambda _ { \mathrm { t r i } } \mathcal { L } _ { \mathrm { t r a c e } } ^ { \mathrm { t r i } } .$ , and the overall learning objective after introducing TRACE becomes $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { b a s e } } ^ { \mathrm { ~ - ~ } } + \mathcal { L } _ { \mathrm { t r a c e } } .$ . The two evidence losses are complementary: InfoNCE separates positive videos from in-batch negatives, while the bidirectional triplet term enforces pairwise margins. $\mathcal { L } _ { \mathrm { b a s e } }$ preserves backbone retrieval and register generation, whereas $\mathcal { L } _ { \mathrm { t r a c e } }$ shapes the auxiliary evidence score. Fusion is delayed until the evidence branch becomes discriminative, preventing early errors from determining the ranking. The objective is introduced progressively in one continuous procedure: the evidence branch is warmed up without fusion, its auxiliary loss is then enabled, and fusion is ramped after stabilization. The exact schedule is given in the Implementation Details.

## Experiments

## Experimental Setup

Datasets and Metrics. We evaluate ActivityNet Captions (Krishna et al. 2017), Charades-STA (Gao et al. 2017), and TVR (Lei et al. 2020) using standard PRVR splits (Dong et al. 2022). We report Recall@K for $K \in \{ 1 , \dot { 5 } , 1 0 , 1 0 0 \}$ and their sum, SumR = R@1 + R@5 + R@10 + R@100, in percentages.

## Implementation Details

Data Pre-Processing. We use pre-extracted features throughout. ActivityNet Captions and Charades-STA use the I3D video and 1,024-dimensional RoBERTa text features released with MS-SL (Dong et al. 2022). TVR uses 3,072- dimensional concatenated ResNet152/I3D video features and 768-dimensional RoBERTa text features.

Experimental Configurations. We adopt the Dream-PRVR configuration (Li et al. 2026c): hidden dimension 384, four attention heads, and $N _ { r } = 4 , 6 , 8$ registers for ActivityNet Captions, Charades-STA, and TVR. TRACE uses a unified configuration: $\lambda = 0 . 0 3 , \gamma = 0 . 3 , \eta = 0 . 2$ , and $\tau _ { r } = 0 . 0 7$ . Training uses batch size 128 on eight NVIDIA A800-80G GPUs. After a 100-epoch backbone warm-up, we freeze the backbone for the first five TRACE epochs and warm up the evidence projections without fusion; the auxiliary loss starts at $t = 5 ,$ , and fusion starts at t = 30 with a 20-epoch ramp. Full optimization details are in the Supplement.

## Comparison with State-of-the-art

Baselines. We compare conventional T2VR methods (CLIP4Clip (Luo et al. 2022); Cap4Video (Wu et al. 2023)), VCMR methods (XML (Lei et al. 2020); CONQUER (Hou, Ngo, and Chan 2021)), and PRVR methods (MS-SL (Dong et al. 2022), GMMFormer (Wang et al. 2024b), GMM-FormerV2 (Wang et al. 2024a), HLFormer (Li et al. 2025), DreamPRVR (Li et al. 2026c), and Holmes (Li et al. 2026b)). DreamPRVR is the register-based backbone baseline. All results use the standard single-query protocol; for Holmes, we exclude its TVR dual-query variant because it requires additional inputs.

Retrieval Performance. Table 1 reports retrieval performance across the three benchmarks. PRVR-specific methods generally outperform conventional trimmed-video retrieval baselines, reflecting the value of partial localized modeling. TRACE achieves the highest SumR on all three datasets, exceeding the strongest prior SumR by +0.5, +0.5, and $+ 0 . 4 ,$ and improving the DreamPRVR backbone by +1.2, +1.1, and +1.5 on ActivityNet Captions, Charades-STA, and TVR, respectively. The improvements appear at both ends of the ranking range: TRACE improves R@1 and R@100 on all three datasets, contributing to the overall SumR gain even when an intermediate cutof, such as Charades-STA R@10, changes slightly. On TVR, TRACE also improves R@1 to 17.8%, suggesting that text-conditioned latent-path marginalization can refine final ranking decisions beyond representation-level register enhancement without replacing the underlying retrieval backbone.

The gain profile difers across benchmarks. ActivityNet improves across mid- and high-recall cutofs, while Charades-STA shows stronger R@1 and R@100 efects with a small R@10 change. TVR obtains the largest SumR and R@1 gains over DreamPRVR, suggesting that queryconditioned support is especially useful when long videos contain several plausible hard negatives. TRACE therefore changes ranking selectively rather than uniformly increasing every recall cutof.

Qualitative Visualization. We select a Charades-STA case in which the register-based base branch ranks a hard negative above the ground-truth video. Figure 3 visualizes the corresponding score curves and video frames. The groundtruth moment receives aligned direct and register support, whereas the hard negative has a direct peak of 0.88 but only 0.74 register support, producing an unsupported gap of

<table><tr><td>Method</td><td colspan="4">ActivityNet</td><td></td><td colspan="2"></td><td colspan="3">Charades</td><td colspan="5">TVR</td></tr><tr><td></td><td>R@1 R@5</td><td></td><td></td><td>R@10R@100</td><td>SumR</td><td>R@1</td><td>R@5</td><td></td><td>R@10 R@100</td><td>SumR</td><td></td><td>R@1 R@5</td><td>R@10</td><td>R@100</td><td>SumR</td></tr><tr><td>CLIP4Clip</td><td>5.9</td><td>19.3</td><td>30.4</td><td>71.6</td><td>127.3</td><td>1.8</td><td>6.5</td><td>10.9</td><td>44.2</td><td>63.4</td><td>9.9</td><td>24.3</td><td>34.3</td><td>72.5</td><td>141.0</td></tr><tr><td>Cap4Video</td><td>6.3</td><td>20.4</td><td>30.9</td><td>72.6</td><td>130.2</td><td>1.9</td><td>6.7</td><td>11.3</td><td>45.0</td><td>65.0</td><td>10.3</td><td>26.4</td><td>36.8</td><td>74.0</td><td>147.5</td></tr><tr><td>XL</td><td>5.3</td><td>19.4</td><td>30.6</td><td>73.1</td><td>128.4</td><td>1.6</td><td>6.0</td><td>10.1</td><td>46.9</td><td>64.6</td><td>10.7</td><td>28.1</td><td>38.1</td><td>80.3</td><td>157.1</td></tr><tr><td>CONQUER</td><td>6.5</td><td>20.4</td><td>31.8</td><td>74.3</td><td>133.1</td><td>1.8</td><td>6.3</td><td>10.3</td><td>47.5</td><td>66.0</td><td>11.0</td><td>28.9</td><td>39.6</td><td>81.3</td><td>160.8</td></tr><tr><td>MS-SL</td><td>7.1</td><td>22.5</td><td>34.7</td><td>75.8</td><td>140.1</td><td>1.8</td><td>7.1</td><td>11.8</td><td>47.7</td><td>68.4</td><td>13.5</td><td>32.1</td><td>43.4</td><td>83.4</td><td>172.4</td></tr><tr><td>GMMFormer</td><td>8.3</td><td>24.9</td><td>36.7</td><td>76.1</td><td>146.0</td><td>2.1</td><td>7.8</td><td>12.5</td><td>50.6</td><td>72.9</td><td>13.9</td><td>33.3</td><td>44.5</td><td>84.9</td><td>176.6</td></tr><tr><td>GMMFormerV2</td><td>8.9</td><td>27.1</td><td>40.2</td><td>78.7</td><td>154.9</td><td>2.5</td><td>8.6</td><td>13.9</td><td>53.2</td><td>78.2</td><td>16.2</td><td>37.6</td><td>48.8</td><td>86.4</td><td>189.1</td></tr><tr><td>HLFormer</td><td>8.7</td><td>27.1</td><td>40.1</td><td>79.0</td><td>154.9</td><td>2.6</td><td>8.5</td><td>13.7</td><td>54.0</td><td>78.7</td><td>15.7</td><td>37.1</td><td>48.5</td><td>86.4</td><td>187.7</td></tr><tr><td>DreamPRVR</td><td>8.7</td><td>27.5</td><td>40.3</td><td>79.5</td><td>156.1</td><td>2.6</td><td>8.7</td><td>14.5</td><td>54.2</td><td>80.0</td><td>17.4</td><td>39.0</td><td>50.4</td><td>86.2</td><td>193.1</td></tr><tr><td>Holmes</td><td>9.3</td><td>27.8</td><td>40.5</td><td>79.1</td><td>156.8</td><td>2.3</td><td>9.5</td><td>15.2</td><td>53.6</td><td>80.6</td><td>17.3</td><td>39.0</td><td>50.4</td><td>87.4</td><td>194.2</td></tr><tr><td>TRACE(Ours)</td><td>9.0</td><td>28.0</td><td>40.7</td><td>79.6</td><td>157.3</td><td>2.7</td><td>9.1</td><td>14.3</td><td>55.0</td><td>81.1</td><td>17.8</td><td>39.4</td><td>50.7</td><td>86.7</td><td>194.6</td></tr></table>

Table 1: Retrieval performance on ActivityNet Captions, Charades-STA, and TVR. R@K denotes Recall@K (%), higher is better. SumR is computed from unrounded recall values. The best score in each column is marked in bold.

![](images/72d1f64dbb7a36893bcec75844c0752da827346c6551a4bd4c219d1f7b7e3daa.jpg)  
Figure 3: Qualitative correction on Charades-STA. The top panel shows the ground-truth video and the bottom panel the baseline hard negative. TRACE restores the ground truth from rank 2 to rank 1.

0.14. TRACE changes the ground-truth–hard-negative margin from −0.0009 to +0.0021 and restores the ground truth from rank 2 to rank 1.

Model Eficiency. TRACE adds only lightweight query– register and register–frame score computations over a small number of registers. As shown in Table 2, it improves retrieval performance with limited additional cost. We report per-epoch training time, parameter count, video feature extraction time over the full 1,334-video evaluation set, and retrieval time, including query encoding, similarity computation, and ranking. Relative to DreamPRVR, TRACE increases the parameter count by approximately 1.6%, feature extraction time by 2.9%, retrieval time by 0.6%, and perepoch training time by 5.3%. This eficiency is consistent with our design: TRACE adds a lightweight score-time evidence selection rule without introducing or replacing a heavy video-language backbone.

## Ablation Study

We isolate three questions in Table 3: whether global evidence must be selected by the query, whether it must be grounded back to frames, and whether the grounded evidence should retain localized temporal concentration.

<table><tr><td>Model</td><td>Train Time (ms/epoch)</td><td>Params (M)</td><td>Infer Time (ms)</td><td>Retrieval (ms)</td><td>SumR</td></tr><tr><td>GMMFormer</td><td>26,887</td><td>12.85</td><td>2,876</td><td>3,238</td><td>72.9</td></tr><tr><td>HLFormer</td><td>31,463</td><td>28.43</td><td>3,816</td><td>3,655</td><td>78.7</td></tr><tr><td>GMMFormerV2</td><td>38,004</td><td>30.79</td><td>3,843</td><td>3,688</td><td>78.2</td></tr><tr><td>DreamPRVR</td><td>33,609</td><td>36.14</td><td>4,001</td><td>3,686</td><td>80.0</td></tr><tr><td>TRACE</td><td>35,396</td><td>36.73</td><td>4,118</td><td>3,708</td><td>81.1</td></tr></table>

Table 2: Training and inference eficiency on Charades-STA. Inference time denotes feature extraction for 1,334 videos in the evaluation set, while retrieval time accounts for query encoding, similarity computation, and ranking.

Rows (2) and (3) already improve over the base row (1) by using direct query–frame evidence $S _ { q F }$ or register-guided soft evidence $\bar { S } _ { \mathrm { s o f t } } \mathrm { ; }$ their combination in row (4) reaches 156.9 / 80.9 / 194.3 SumR, showing that the two signals are complementary. The most informative change is row (7), which replaces the direct-plus-soft aggregation with routed max evidence $S _ { \mathrm { r o u t e } } .$ . Unlike the soft route, $S _ { \mathrm { r o u t e } }$ preserves frame-wise path evidence and applies hard temporal concentration only after register-path marginalization. It improves over $S _ { q F }$ alone by +0.9, +0.7, and +1.0 SumR on ActivityNet, Charades-STA, and TVR, respectively. Combining the routed selector with the auxiliary soft branch in row (0) gives the strongest overall result.

Rows (5), (6), and (8) further isolate the hierarchical operator. Row (5) drops frame grounding and underperforms, showing global matching alone is insuficient. Row (6) replaces query-conditioned activation with uniform register weights and falls behind row (7), providing evidence that latent paths should be selected by the current text. Row (8) replaces the hard temporal maximum with average aggregation, which dilutes short relevant moments. Together, these results provide evidence against three simpler explanations: a generic global ofset, query-independent register weighting, or temporal averaging.

Compared with direct-plus-soft fusion, routed max evidence improves SumR by +0.2, +0.1, and +0.1 on the three datasets, while the auxiliary soft branch yields the final 157.3/81.1/194.6. Although these margins are modest, their consistency, together with the corruption diagnostics, indicates that routing provides a distinct ranking signal.

<table><tr><td rowspan="2">ID Variant</td><td colspan="5">ActivityNet</td><td colspan="5">Charades</td><td colspan="5"></td></tr><tr><td></td><td>R@1 R@5</td><td></td><td>R@10 R@100 SumR</td><td></td><td>R@1 R@5</td><td></td><td></td><td></td><td>R@10 R@100 SumR</td><td></td><td></td><td></td><td>R@1 R@5 R@10 R@100 SumR</td><td></td></tr><tr><td>(0) TRACE(full)</td><td></td><td>9.0 28.0</td><td>40.7</td><td>79.6</td><td>157.3</td><td>2.7</td><td>9.1</td><td>14.3</td><td>55.0</td><td>81.1</td><td>17.8</td><td>39.4</td><td>50.7</td><td>86.7</td><td>194.6</td></tr><tr><td colspan="10">Efficacy of Evidence Source Construction</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(1)</td><td>Base model</td><td>8.7 27.5</td><td>40.3</td><td>79.5</td><td>156.1</td><td>2.6</td><td>8.7</td><td>14.5</td><td>54.2</td><td>80.0</td><td>17.4</td><td>39.0</td><td>50.4</td><td>86.2</td><td>193.1</td></tr><tr><td>(2)  $S _ { q F } \mathrm { \ o n l y }$ </td><td></td><td>8.8 27.6</td><td>40.3</td><td>79.5</td><td>156.2</td><td>2.7</td><td>8.7</td><td>14.6</td><td>54.3</td><td>80.3</td><td>17.5</td><td>39.3</td><td>50.7</td><td>86.7</td><td>193.4</td></tr><tr><td>(3)</td><td> $S _ { \mathrm { s o f t } } \ \mathrm { o n l y }$ </td><td>8.8 27.7</td><td>40.5</td><td>79.5</td><td>156.5</td><td>2.6</td><td>8.8</td><td>14.5</td><td>54.6</td><td>80.5</td><td>17.5</td><td>39.2</td><td>50.4</td><td>86.4</td><td>193.6</td></tr><tr><td>(4)</td><td>8.9  $S _ { q F } + \eta S _ { \mathrm { s o f t } }$ </td><td>27.8</td><td>40.6</td><td>79.6</td><td>156.9</td><td>2.7</td><td>9.0</td><td>14.4</td><td>54.8</td><td>80.9</td><td>17.6</td><td>39.1</td><td>50.8</td><td>86.8</td><td>194.3</td></tr><tr><td colspan="4">Efficacy of Text-Conditioned Register Routing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(5)</td><td> $S _ { q R } \mathrm { o n l y }$ </td><td>8.8 27.7</td><td>40.5</td><td>79.5</td><td>156.5</td><td>2.6</td><td>8.8</td><td>14.4</td><td>54.6</td><td>80.4</td><td>17.5</td><td>39.2</td><td>50.5</td><td>86.5</td><td>193.8</td></tr><tr><td>(6)</td><td> $\operatorname { U n i f o r m } q \to R$ </td><td>8.9 27.7</td><td>40.6</td><td>79.6</td><td>156.6</td><td>2.6</td><td>9.0</td><td>14.3</td><td>55.0</td><td>80.8</td><td>17.6</td><td>39.2</td><td>50.6</td><td>86.7</td><td>194.0</td></tr><tr><td>(7)  $S _ { \mathrm { r o u t e } } \ \mathrm { o n l y }$ </td><td>8.9</td><td>27.9</td><td>40.8</td><td>79.6</td><td>157.1</td><td>2.6</td><td>9.0</td><td>14.3</td><td>55.1</td><td>81.0</td><td>17.6</td><td>39.3</td><td>50.7</td><td>86.7</td><td>194.4</td></tr><tr><td>Efficacy of Evidence Aggregation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>(8)  $\mathbf { A v g } ( \tilde { d } _ { i } )$ </td><td></td><td>8.8 27.7</td><td>40.5</td><td>79.5</td><td>156.5</td><td>2.7</td><td>8.8</td><td>14.5</td><td>54.4</td><td>80.4</td><td>17.7</td><td>39.1</td><td>50.5</td><td>86.5</td><td>193.8</td></tr></table>

Table 3: Ablation and route diagnostics of TRACE. All evidence variants use the same integration weight λ; combined variants retain the same auxiliary weight η as TRACE. Uniform $q {  } R$ replaces query-conditioned register afinity with equal register weights, while $\mathrm { A v g } ( \tilde { d } _ { i } )$ replaces the outer temporal maximum with a valid-frame average.

![](images/9de504e09178b36d14d5eefa27655d905cf79df27e5335526c58cf671b8bee68.jpg)

![](images/1cad738c6f03f82957928c4eb4d9899c3b9fa4ff8c667e58368a32dce6227a5a.jpg)  
Figure 4: Visualization of text-conditioned register activation on TVR. (a) Query-to-register weights for selected pairs. (b) Dominant register per query group. (c) Dominant-register usage over all 10,895 TVR validation query-positive pairs. TRACE activates registers in a query-dependent manner.

## Register Activation Visualization

We visualize whether register activation varies with the query. Figure 4 shows query-to-register activation patterns on TVR. Panel (a) displays soft activation weights for representative query–video pairs, panel (b) summarizes the dominant register within query groups, and panel (c) reports aggregate usage over validation pairs. Diferent queries emphasize diferent register indices, while aggregate usage spans all indices rather than collapsing to one. This behavior complements Table 3: the uniform $q {  } R$ variant in row (6) removes the query-dependent selection shown in the visualization.

## Cross-Backbone Transferability

To test whether the proposed verification principle depends on difusion-generated registers, we transfer the same scoretime routing idea to four frozen, non-register-based PRVR backbones. For each backbone, lightweight global or temporal anchors and the fusion weight are selected on validation data and then frozen for evaluation. Table 4 shows positive ∆SumR on all three datasets for every backbone, with average gains ranging from +0.17 on MS-SL to +0.96 on HLFormer. These results suggest that learned registers are an efective instantiation of global evidence anchors, but the underlying verification rule is not restricted to a particular backbone.

<table><tr><td>Backbone</td><td>ActivityNet</td><td>Charades</td><td>TVR</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>MS-SL</td><td>+0.18</td><td>+0.13</td><td>+0.21</td><td>+0.17</td></tr><tr><td>GMMFormer</td><td>+0.24</td><td>+0.47</td><td>+0.50</td><td>+0.40</td></tr><tr><td>GMMFormerV2</td><td>+0.54</td><td>+0.31</td><td>+0.88</td><td>+0.58</td></tr><tr><td>HLFormer</td><td>+0.89</td><td>+0.56</td><td>+1.43</td><td>+0.96</td></tr></table>

Table 4: Frozen-backbone diagnostic on representative earlier non-register-based PRVR backbones. We report evaluation-split ∆SumR after selecting the anchor construction and fusion weight on validation data and freezing both choices. Full settings are in the Supplement Table S1.

## Conclusion

In this paper, we propose TRACE, a score-level evidence verification operator for partially relevant video retrieval. TRACE activates global registers according to the current query, grounds their support to frame-level evidence, and performs smooth latent-path aggregation before localized temporal selection. This design preserves the local concentration required by PRVR while reducing the influence of unsupported peaks, and can be integrated with an existing register-based backbone through residual fusion. Extensive experiments on three PRVR benchmarks show that TRACE consistently improves strong retrieval baselines, achieving SumR gains of +1.2, +1.1, and +1.5 over DreamPRVR, while ablation, routing-corruption, cross-backbone transfer, hard-negative, and qualitative analyses support the role of query-conditioned evidence grounding. TRACE ofers a complementary perspective to representation-centric PRVR modeling: global context should not only enrich video representations, but also participate in verifying the evidence used for final ranking.

## References

Chen, C.; Zhou, K.; Wang, F.; Ning, Y.; Xiong, Z.; Li, Y.; Wen, Z.; and Tan, M. 2026a. CaptAin: Caption-driven Alignment for Bridging Modality Gaps in Partially Relevant Video Retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6208–6217.

Chen, C.; Zhou, K.; Wen, Z.; You, Z.; Li, Y.; Xiang, T.; and Tan, M. 2026b. Action-and-object aware alignment for partially relevant video retrieval. In Proceedings ofthe AAAI Conference onArtificial Intelligence, volume 40, 2814–2822.

Cho, C.-H.; Moon, W.; Jun, W.; Jung, M.; and Heo, J.-P. 2025. Ambiguity-restrained text-video representation learning for partially relevant video retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 2500–2508.

Darcet, T.; Oquab, M.; Mairal, J.; and Bojanowski, P. 2024. Vision transformers need registers. In International conference on learning representations, volume 2024, 2632–2652.

Dietterich, T. G.; Lathrop, R. H.; and Lozano-Pérez, T. 1997. Solving the multiple instance problem with axis-parallel rectangles. Artificial intelligence, 89(1-2): 31–71.

Dong, J.; Chen, X.; Zhang, M.; Yang, X.; Chen, S.; Li, X.; and Wang, X. 2022. Partially relevant video retrieval. In Proceedings of the 30th ACM International Conference on Multimedia, 246–257.

Dong, J.; Zhang, M.; Zhang, Z.; Chen, X.; Liu, D.; Qu, X.; Wang, X.; and Liu, B. 2023. Dual learning with dynamic knowledge distillation for partially relevant video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 11302–11312.

Gao, J.; Sun, C.; Yang, Z.; and Nevatia, R. 2017. Tall: Temporal activity localization via language query. In Proceedings of the IEEE international conference on computer vision, 5267–5275.

Hou, Z.; Ngo, C.-W.; and Chan, W. K. 2021. Conquer: Contextual query-aware ranking for video corpus moment retrieval. In Proceedings of the 29th ACM International Conference on Multimedia, 3900–3908.

Jun, W.; Moon, W.; Cho, C.-H.; Jung, M.; and Heo, J.-P. 2025. Bridging the semantic granularity gap between text and frame representations for partially relevant video retrieval. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, 4166–4174.

Krishna, R.; Hata, K.; Ren, F.; Fei-Fei, L.; and Niebles, J. C. 2017. Dense-Captioning Events in Videos. In 2017 IEEE International Conference on Computer Vision (ICCV), 706– 715. IEEE.

Lei, J.; Yu, L.; Berg, T. L.; and Bansal, M. 2020. Tvr: A largescale dataset for video-subtitle moment retrieval. In European Conference on Computer Vision, 447–463. Springer.

Li, H.; Zhao, J.; Zhang, Y.; and Wen, J. 2026a. Bidirectional Cross-Modal Collaborative Alignment via Semantic-Guided Visual Embeddings for Partially Relevant Video Retrieval. IEEE Transactions on Image Processing.

Li, J.; Lai, P.; Lou, X.; Wang, J.; Wang, Y.; Chen, K.; Wang, Y.; and Xia, S.-T. 2026b. Revisiting Uncertainty: On Evidential Learning for Partially Relevant Video Retrieval. In Forty-third International Conference on Machine Learning.

Li, J.; Lou, X.; Wang, J.; Wang, Y.; Wang, Y.; Xia, S.-T.; and Chen, B. 2026c. Imagine Before Concentration: Difusion-Guided Registers Enhance Partially Relevant Video Retrieval. arXiv preprint arXiv:2604.03653.

Li, J.; Wang, J.; Tan, C.; Lian, N.; Chen, L.; Wang, Y.; Zhang, M.; Xia, S.-T.; and Chen, B. 2025. Hlformer: Enhancing partially relevant video retrieval with hyperbolic learning. arXiv preprint arXiv:2507.17402.

Luo, H.; Ji, L.; Zhong, M.; Chen, Y.; Lei, W.; Duan, N.; and Li, T. 2022. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 508: 293–304.

Moon, W.; Cho, C.-H.; Jun, W.; Kim, T.; Lee, I.; Wee, D.; Shim, M.; and Heo, J.-P. 2025. Prototypes are balanced units for eficient and efective partially relevant video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 21789–21799.

Moon, W.; Jung, M.; Park, G.; Kim, T.-Y.; Cho, C.-H.; Jun, W.; and Heo, J.-P. 2026. Mitigating semantic collapse in partially relevant video retrieval. Advances in Neural Information Processing Systems, 38: 23196–23217.

Wang, Y.; Wang, J.; Chen, B.; Dai, T.; Luo, R.; and Xia, S.-T. 2024a. Gmmformer v2: An uncertainty-aware framework for partially relevant video retrieval. arXiv preprint arXiv:2405.13824.

Wang, Y.; Wang, J.; Chen, B.; Zeng, Z.; and Xia, S.-T. 2024b. Gmmformer: Gaussian-mixture-model based transformer for eficient partially relevant video retrieval. In Proceedings of the AAAI conference on artificial intelligence, volume 38, 5767–5775.

Wu, W.; Luo, H.; Fang, B.; Wang, J.; and Ouyang, W. 2023. Cap4video: What can auxiliary captions do for text-video retrieval? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10704–10713.

Yang, J.; Wang, Q.; Jin, Y.; Ma, S.; Xu, M.; and Pang, S. 2026. Knowledge-Refined Dual Context-Aware Network for Partially Relevant Video Retrieval. arXiv preprint arXiv:2603.23902.

Zhang, L.; Song, P.; Dong, J.; Li, K.; and Yang, X. 2025a. Enhancing partially relevant video retrieval with robust align ment learning. arXiv preprint arXiv:2509.01383.

Zhang, R.; Shao, R.; Chen, G.; Zhang, M.; Zhou, K.; Guan, W.; and Nie, L. 2025b. Falcon: Resolving visual redundancy and fragmentation in high-resolution multimodal large language models via visual registers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 23530–23540.

Zhu, X.; Wang, S.; Yang, J.; Yang, Y.; Tu, W.; and Wang, Z. 2025. Query-Based Audio-Visual Temporal Forgery Localization with Register-Enhanced Representation Learning. In Proceedings of the 33rd ACM International Conference on Multimedia, 8547–8556.

## Appendix

This Technical Supplement is organised in two parts. Section A reports additional experiments and diagnostics: hyperparameter sensitivity, transfer to earlier non-register-based backbones, hard-negative subsets, paired bootstrap, ranklevel correction, and routing corruption. Section B reports method details: full implementation, training schedule, and theoretical properties of the concentration operator.

## A.1 Hyper-Parameter Sensitivity

TRACE introduces three score-level hyperparameters: the fusion weight λ, the routing strength $\gamma ,$ and the soft register evidence weight η. In the main experiments, we use a unified configuration across all datasets, i.e., $\lambda = 0 . 0 3 , \gamma = 0 . 3$ , and $\eta = 0 . 2$ . The only dataset-dependent setting is the number of registers $N _ { r } ,$ which is inherited from the DreamPRVR backbone: $N _ { r } = 4$ for ActivityNet Captions, $N _ { r } ~ = ~ 6$ for Charades-STA, and $N _ { r } = 8$ for TVR. Since changing $N _ { r }$ alters the register generation module rather than only the proposed TRACE calibration branch, we keep this backbone design unchanged.

Figure 5 analyzes the influence of $\lambda , \gamma ,$ , and η on validation SumR. The results show that TRACE maintains stable performance around the unified setting. In particular, non-zero fusion weights consistently outperform $\lambda = 0 ,$ , showing that text-conditioned evidence provides useful score-level calibration. Moderate routing and soft evidence weights also perform competitively, supporting our choice of a single cross-dataset configuration rather than dataset-specific tuning.

## A.2 Transfer to Earlier Non-Register-Based PRVR Backbones

TRACE is instantiated with difusion-generated registers in the main experiments because they provide explicit global evidence anchors. To test whether the score-time verification principle is strictly tied to such registers, we transfer the same idea to earlier non-register-based PRVR backbones. For each method and dataset, we use the corresponding oficial checkpoint, keep the backbone frozen, and derive lightweight anchors from its encoded local tokens by temporal partitioning or global pooling. Candidate anchor constructions and fusion weights are selected using validation SumR only; the selected configuration is then fixed before evaluation. We apply the resulting query-anchor-frame evidence route to calibrate the original similarity. This diagnostic does not retrain the baselines and should be interpreted as an anchor-agnostic scoring test rather than a full backbone-level reimplementation.

The anchor construction follows the representation structure of each backbone. MS-SL builds multi-scale local matching features, so we construct anchors from its encoded frame tokens. GMMFormer models temporal evidence mainly through Gaussian clip-level representations; accordingly, we use clip-token anchors for GMMFormer. GMM-FormerV2 provides stronger frame-level local tokens, so we derive frame-token anchors. HLFormer models hierarchical partial relevance in hyperbolic space while still producing local evidence for final ranking; we therefore construct anchors from its encoded frame tokens.

Table 5 shows that anchor-based score verification yields positive gains across the evaluated earlier PRVR backbones. The gains are smaller than the main TRACE results with explicit difusion registers, which is expected because these anchors are obtained by simple pooling rather than learned global register generation. Nevertheless, the diagnostic supports our interpretation that the key operation is score-time verification through query-selected global anchors, while diffusion registers are an efective instantiation rather than the only possible source of anchors. We also attempted to include CLIP4Clip, Cap4Video, XML, and CONQUER in this diagnostic. Their released code or checkpoints do not directly provide comparable three-dataset frozen-backbone evaluation with exposed local evidence tokens, so we keep them as standard comparison baselines in Table 1 rather than reporting non-comparable diagnostic results.

## A.3 Additional Diagnostics

We first provide additional qualitative visualizations that complement the main-text result figure. The main qualitative result already illustrates a corrected hard-negative case at the frame-evidence level, showing that the ground-truth peak is better aligned with register support than the hardnegative peak. Figure 6 further summarizes strict top-1 correction cases, where TRACE consistently gives the groundtruth video a larger calibration boost than the hard negative.

We next evaluate whether TRACE is especially helpful on the failure mode it is designed to address. We first identify queries for which the baseline ranks an incorrect video at top 1. For each such query, we measure the unsupported local peak of that top-1 hard negative, i.e., the gap between its strongest direct local evidence and the register support at the same peak. We then rank these baseline-error queries by the unsupported gap and evaluate TRACE on the top 50% and top 25% subsets. If TRACE only introduced a generic score bias, its gain would not systematically grow on these increasingly dificult subsets.

Table 6 shows that TRACE’s gains become larger on hardnegative subsets across all three datasets. The all-query gains match the DreamPRVR-to-TRACE SumR diferences in Table 1, while the gains further increase on baseline-error queries and on the top 25% unsupported hard-negative subset. This subset trend is important because it connects the numerical gains to the proposed failure mode: TRACE is particularly efective when the baseline is distracted by locally plausible but globally weakly supported negative evidence.

For statistical testing, we treat each query as one paired observation and retain the pairing between the base and TRACE ranks. We draw 20,000 bootstrap samples of query indices with replacement and report the 2.5th and 97.5th percentiles of the resulting ∆SumR distribution as the 95% confidence interval. The reported p-values are obtained from a one-sided paired sign-flip test with 20,000 random sign assignments. All tests use the same evaluation split as the corresponding diagnostic, with random seed 20260712.

Table 7 further quantifies this trend with paired bootstrap confidence intervals. The full-query SumR improvement is statistically reliable across all three datasets, and the top-

![](images/ccae185664e086744cb208680b2833c608d4b13755dd24662c1f1f48c1adfd5b.jpg)  
Figure 5: Hyper-parameter analysis of TRACE on validation sets. We evaluate the efects of the score fusion weight λ, routing strength γ, and soft evidence weight η on SumR. The results show that TRACE is robust within a reasonable range and that a unified score-level configuration works consistently across datasets.

25% unsupported hard-negative subsets show even larger positive efects. This supports our interpretation that TRACE improves the overall retrieval score while being especially efective on the intended hard-negative boundary.

To further test whether TRACE benefits from meaningful cross-granular grounding rather than merely adding a generic score bias, we conduct a routing corruption diagnostic on TVR. We compare the base branch, the original TRACE score, and three corrupted routing variants. Uniform q→R removes query-specific register selection, cross-video shufle q→R replaces the activation with mismatched queryregister correspondence, while shufle R→F corrupts the register-to-frame grounding path before computing routed evidence. These corruptions preserve the existence of an additional score branch but progressively destroy the semantic correspondence of its route.

Table 9 shows that TRACE improves the base branch across multiple recall cutofs, increasing SumR from 193.1 to 194.6 and improving the ground-truth rank for 3,518 TVR queries. This indicates that the efect is not limited to strict top-1 boundary cases. Removing query-specific activation or replacing it with cross-video activation weakens the gain, while corrupting the register-to-frame grounding path reduces SumR to 192.6 and degrades more ranks than it improves. This diagnostic supports the role of grounded $q {  } R {  } F$ evidence in score-time verification rather than a generic score ofset.

## A.4 Scope and Limitations

TRACE studies score-time evidence verification rather than replacing the underlying video-language backbone. Its current instantiation uses global registers because they provide explicit semantic anchors that can be selected by the query and grounded to frames. Therefore, the method is most directly applicable to PRVR systems that expose both local temporal tokens and global evidence anchors. This scope also explains why the observed gains are moderate but consistent: TRACE preserves the base representation space and focuses on correcting ranking behavior through text-conditioned evidence verification. Importantly, this limitation is also a design choice: by operating at the score level, TRACE can diagnose whether a local peak is globally supported without retraining a heavy generator or altering the backbone architecture.

<table><tr><td>Backbone</td><td>Dataset</td><td>Base SumR</td><td>+Route SumR</td><td>∆SumR</td><td>Selected Anchor</td><td>λ</td></tr><tr><td>MS-SL</td><td>ActivityNet</td><td>140.10</td><td>140.28</td><td>+0.18</td><td>global-4</td><td>0.015</td></tr><tr><td>MS-SL</td><td>Charades</td><td>68.40</td><td>68.53</td><td>+0.13</td><td>temporal-4</td><td>0.010</td></tr><tr><td>MS-SL</td><td>TVR</td><td>172.40</td><td>172.61</td><td>+0.21</td><td>temporal-6</td><td>0.010</td></tr><tr><td>GMMFormer</td><td>ActivityNet</td><td>146.00</td><td>146.24</td><td>+0.24</td><td>temporal-4</td><td>0.060</td></tr><tr><td>GMMFormer</td><td>Charades</td><td>72.90</td><td>73.37</td><td>+0.47</td><td>temporal-6</td><td>0.120</td></tr><tr><td>GMMFormer</td><td>TVR</td><td>176.60</td><td>177.10</td><td>+0.50</td><td>temporal-8</td><td>0.200</td></tr><tr><td>GMMFormerV2</td><td>ActivityNet</td><td>154.90</td><td>155.44</td><td>+0.54</td><td>temporal-8</td><td>0.200</td></tr><tr><td>GMMFormerV2</td><td>Charades</td><td>78.20</td><td>78.51</td><td>+0.31</td><td>temporal-6</td><td>0.100</td></tr><tr><td>GMMFormerV2</td><td>TVR</td><td>189.10</td><td>189.98</td><td>+0.88</td><td>temporal-4</td><td>0.200</td></tr><tr><td>HLFormer</td><td>ActivityNet</td><td>154.90</td><td>155.79</td><td>+0.89</td><td>temporal-6</td><td>0.150</td></tr><tr><td>HLFormer</td><td>Charades</td><td>78.70</td><td>79.26</td><td>+0.56</td><td>temporal-6</td><td>0.120</td></tr><tr><td>HLFormer</td><td>TVR</td><td>187.70</td><td>189.13</td><td>+1.43</td><td>global-4</td><td>0.200</td></tr></table>

Table 5: Full frozen-backbone diagnostic on earlier non-register-based PRVR backbones. Each row uses the oficial datasetspecific checkpoint of the corresponding method. Anchor type and λ are selected on the validation split and then frozen for evaluation; all backbone parameters remain frozen.
<table><tr><td>Dataset</td><td>All ∆SumR</td><td>Err. ∆SumR</td><td>Top-50 ∆SumR</td><td>Top-25 ∆SumR</td><td>Top-25 Mean Gain</td><td>Top-25 #Query</td></tr><tr><td>ActivityNet</td><td>+1.20</td><td>+1.48</td><td>+1.70</td><td>+2.08</td><td>+1.96</td><td>3584</td></tr><tr><td>Charades-STA</td><td>+1.10</td><td>+1.30</td><td>+1.48</td><td>+1.82</td><td>+1.85</td><td>906</td></tr><tr><td>TVR</td><td>+1.50</td><td>+1.78</td><td>+1.96</td><td>+2.34</td><td>+2.48</td><td>2279</td></tr></table>

Table 6: Hard-negative subset analysis on evaluation queries. Err. denotes queries for which the baseline top-1 video is incorrect. Top-50 and Top-25 are the 50% and 25% subsets of these baseline-error queries with the largest unsupported local gaps. ∆SumR is the recall improvement on each subset; Mean Gain is the average ground-truth rank improvement on Top-25.

We organize the remaining discussion along four explicit axes: register capacity design, explicit failure modes, the high-precision–mid-recall trade-of, and the computational budget.

Register capacity $N _ { r } .$ The number of registers is inherited from the DreamPRVR backbone $( N _ { r } { = } 4$ for ActivityNet Captions, $N _ { r } { = } 6$ for Charades-STA, $N _ { r } { = } 8$ for TVR) rather than treated as a TRACE-specific hyperparameter, because changing $N _ { r }$ modifies the register generator itself and therefore the representation space shared with the base branch. The dataset-specific choice reflects an empirical balance between query vocabulary breadth and per-register specialization. ActivityNet Captions queries tend to describe a single salient activity, so a small register set is suficient; Charades-STA queries mix action verbs with multiple fine-grained entities, which benefits from a moderately larger set; TVR queries describe character interactions and scene-level relations across longer videos, which motivates the largest set. We did not sweep $N _ { r }$ independently because such a sweep would change the backbone’s representation capacity rather than the score-time rule, and a register-agnostic verification result is the very claim of Table 5.

Explicit failure modes. Two concrete regimes degrade TRACE back toward its base branch. First, when a query is semantically distant from every register in the active video, the softmax in Eq. (4) becomes approximately uniform and $\rho _ { i }$ degrades to a query-agnostic global support term. This regime corresponds to the Uniform $q {  } R$ corruption diagnostic in Table 9, where TVR ∆SumR drops from +1.5 to +0.8 and the number of rank-improved queries drops from 3,518 to 2,864. The remedy is therefore not a fix inside TRACE but a richer register vocabulary, e.g. a hybrid register set that mixes global, local, and event-level anchors. Second, TRACE relies on query-anchor afinity being meaningful for the current ranking. When explicit learned registers are unavailable and anchors must instead be derived by simple pooling from a frozen non-register backbone, the gain falls into the lightweight-anchor range of roughly +0.13 to +1.43 SumR (Table 5), with the strongest lightweight-anchor gain (+1.43 on HLFormer TVR) approaching but not matching the difusion-register instantiation.

High-precision vs. mid-recall trade-of. Although TRACE achieves the best SumR on all three datasets, it does not dominate every individual recall cutof. The pattern is consistent across datasets: TRACE improves R@1 (high-precision boundary) and R@100 (long-tail boundary) but moves slightly at R@5 or R@10 on certain datasets, e.g. Charades-STA R@10 changes from 14.5 to 14.3 (Table 1). This indicates that score-time verification and uncertainty-aware evidence estimation emphasize complementary aspects of the ranking distribution: routed support is most informative when the top of the list and the long tail are the contested regions, whereas mid-recall cutofs are dominated by relative ordering among already-strong candidates, where routed support contributes a smaller per-item margin. The trade-of is therefore not a flaw but a profile of what score-time evidence verification can and

![](images/c6355c0a8dfacb2795f0e261fa1a7acd86573bb823f91eb44e7d8008b9d3f5c4.jpg)

![](images/44377a36fcac14404f2fcf868c6ce3b66dace618c1376c3ffa4572695fbe8c55.jpg)

Figure 6: Top-rank correction on Charades-STA hard-negative cases. Left: for each query, the baseline ranks a hard negative above the ground truth, and TRACE moves the margin above zero to restore rank 1. Right: the score-change decomposition into ∆GT, ∆HN, and the relative advantage $\Delta \mathrm { a d v } = \Delta \bar { \mathrm { G T } } - \Delta \mathrm { H N } .$
<table><tr><td>Dataset</td><td>All ∆SumR</td><td>95% CI</td><td>p</td><td>Top-25 ∆SumR</td><td> $\mathrm { T o p } { - } 2 5 9 5 \% \mathrm { C I } / p$ </td></tr><tr><td>ActivityNet</td><td>+1.20</td><td>[+0.84, +1.56]</td><td>&lt; 0.001</td><td>+2.08</td><td> $\left[ + 1 . 3 6 , + 2 . 8 2 \right] / < 0 . 0 0 1$ </td></tr><tr><td>Charades-STA</td><td>+1.10</td><td>[+0.52, +1.69]</td><td>&lt; 0.001</td><td>+1.82</td><td> $\left[ + 0 . 5 8 , + 3 . 1 6 \right] / 0 . 0 0 6$ </td></tr><tr><td>TVR</td><td>+1.50</td><td>[+1.16, +1.84]</td><td>&lt; 0.001</td><td>+2.34</td><td> $\left[ + 1 . 5 5 , + 3 . 2 0 \right] / < 0 . 0 0 1$ </td></tr></table>

Table 7: Query-level paired bootstrap confidence intervals and paired sign-flip tests for SumR improvements. The all-query interval measures the recall improvement over the full evaluation split. The Top-25 columns repeat the analysis on the 25% baseline-error queries whose top-1 hard negatives have the largest unsupported local peaks.

## cannot move.

Computational budget. TRACE adds a small score-time branch without introducing an additional backbone pass, so the dominant inference cost remains the base forward pass. The additional cost per query comes from (i) a register activation vector $\boldsymbol { \pi } \in \mathbb { R } ^ { N _ { r } }$ computed as one $N _ { r ^ { - } } { \bf w a y }$ softmax over cosine similarities, (ii) an $N _ { r } \times F$ register–frame compatibility matrix computed for evidence routing, and (iii) a perframe reduction that takes the temporal maximum of $d _ { i } + \gamma \rho _ { i }$ Thus, the dominant added cost is the $N _ { r } \times F$ register–frame aggregation and a single temporal max-reduction; the memory footprint is proportional to $N _ { r } \times F$ per video. Since $N _ { r }$ is small, the verification cost remains bounded by the base forward pass as the video encoder scales. A natural next step is to combine TRACE-style routed support with an evidential head, but we leave that direction for future work.

Finally, the main-text results use difusion-generated registers because they consistently provide the strongest global anchors. Table 5 reports the corresponding lightweight-anchor sweep, where global pooling and temporal partitioning replace the register generator while leaving the rest of the pipeline intact. Even with simple anchors, the verification branch still yields positive ∆SumR across all four backbones and all three datasets, supporting score-time verification as the shared operation. The much larger gains on TVR (up to +1.43 ∆SumR on HLFormer) further suggest that richer, query-relevant anchor generation is an important source of TRACE’s headline numbers, while the verification rule itself is anchor-agnostic.

## B.1 Implementation and Computation Details

For ActivityNet Captions and Charades-STA, we use the provided I3D features as video representations. Query representations are encoded with 1,024-dimensional RoBERTa features following MS-SL. For TVR, we use the released 3,072-dimensional video features that concatenate framelevel ResNet152 and segment-level I3D representations, and encode the corresponding textual queries with 768- dimensional RoBERTa features.

We follow the register-based backbone configuration and use a hidden dimension of 384 with 4 attention heads. The number of registers is set to 4, 6, and 8 for ActivityNet Captions, Charades-STA, and TVR, respectively, directly following the DreamPRVR backbone design. Since changing $N _ { r }$ modifies the register generator itself, we do not treat it as a TRACE-specific hyperparameter. For the proposed score-level branch, we use the same configuration across all datasets: the evidence fusion weight is $\lambda = 0 . 0 3$ , the routing strength is $\gamma = 0 . 3 .$ , and the soft register evidence weight is $\eta = 0 . 2$ . The register activation temperature $\tau _ { r }$ is fixed to 0.07. The TRACE auxiliary objective uses $\lambda _ { \mathrm { n c e } } = 0 . 0 2$ and $\lambda _ { \mathrm { t r i } } ~ = ~ 0 . 0 5$ , with triplet margins 0.2 for ActivityNet Captions and Charades-STA and 0.1 for TVR.

![](images/815f58cee9feaa5dfdbaca98d98d530d6e9e9294a0ff105846fc488acf7f5fef.jpg)

Figure 7: Unsupported-gap stratification on baseline-error evaluation queries. Queries are split into four quartile bins (Q1–Q4) by the unsupported evidence gap between the baseline top-1 hard negative and the ground truth. Larger gaps yield larger rank gains after TRACE, and Q4 aligns with the Top-25 subset in Table 6.
<table><tr><td>Dataset</td><td>Base R@1</td><td>TRACE R@1</td><td>∆R@1</td><td>Rank↑</td><td>Rank↓</td><td>Mean Gain</td><td>Median Gain</td></tr><tr><td>ActivityNet</td><td>8.7</td><td>9.0</td><td>+0.3</td><td>5204</td><td>2841</td><td>+1.28</td><td>0.00</td></tr><tr><td>Charades-STA</td><td>2.6</td><td>2.7</td><td>+0.1</td><td>1938</td><td>1096</td><td>+1.55</td><td>0.00</td></tr><tr><td>TVR</td><td>17.4</td><td>17.8</td><td>+0.4</td><td>3518</td><td>1292</td><td> $+ 1 . 7 6$ </td><td>0.00</td></tr></table>

Table 8: Rank-level change statistics on evaluation queries. Base and TRACE R@1 repeat the values in Table 1, making the toprank change directly comparable to the main results. Rank↑ and Rank↓ count queries whose ground-truth video rank improves or degrades, while Mean Gain and Median Gain summarize the signed rank improvement.

Algorithm 1 Text-Conditioned Register Activation and   
Cross-Granular Evidence Routing (TRACE)   
Require: Query embedding $q ,$ frame tokens $F ,$ clip tokens   
C (consumed by $S _ { \mathrm { b a s e } }$ via Eq. (1)), and video registers   
$R$   
1: Compute base similarity $S _ { \mathrm { b a s e } }$ by Eq. (1).   
2: Compute query-to-frame evidence $d _ { i } = \cos ( q , f _ { i } )$   
3: Compute query-register afinity $a _ { k } = \cos ( q , r _ { k } ) .$   
4: Compute register-frame compatibility $\begin{array} { r l } { b _ { k , i } } & { { } = } \end{array}$   
cos $( r _ { k } , f _ { i } )$   
5: Route text-selected registers to frames to obtain $\rho _ { i } .$   
6: Combine local and routed evidence: $\tilde { d } _ { i } = d _ { i } + \gamma \rho _ { i } .$   
7: Select the strongest routed evidence $S _ { \mathrm { r o u t e } } = \operatorname* { m a x } _ { i } \tilde { d } _ { i }$   
8: Compute register-guided soft evidence $S _ { \mathrm { s o f t } }$   
9: Compute $\bar { S _ { \mathrm { t r a c e } } } = \bar { S } _ { \mathrm { r o u t e } } + \eta S _ { \mathrm { s o f t } }$   
10: return $S _ { \mathrm { f i n a l } } = S _ { \mathrm { b a s e } } + \lambda S _ { \mathrm { t r a c e } } .$

Training follows one continuous progressive procedure. We first warm up the base register-based backbone for 100 epochs with $\bar { \mathcal { L } _ { \mathrm { b a s e } } }$ and use its resulting retrieval score as $\bar { S } _ { \mathrm { b a s e } } ;$ all epoch indices t below are local to the subsequent TRACE phase. During the first 5 TRACE-stage epochs $( t = 0 , \ldots , 4 ) .$ , we freeze the backbone and optimize the evidence projections with ${ \mathcal { L } } _ { \mathrm { t r a c e } } ,$ while the ranking score remains $S _ { \mathrm { b a s e } }$ because the efective integration weight is zero. From TRACE-stage epoch $t = 5 ,$ we jointly optimize the backbone and evidence branch with ${ \mathcal { L } } _ { \mathrm { b a s e } } + { \mathcal { L } } _ { \mathrm { t r a c e } }$ . From t = 30, score integration is activated and the efective weight is linearly ramped for 20 epochs:

$$
\lambda ( t ) = \lambda \cdot \operatorname* { m i n } \left( 1 , { \frac { t - 3 0 } { 2 0 } } \right) , \quad t \geq 3 0 .\tag{11}
$$

The final calibration learning rates are $8 \times 1 0 ^ { - 5 } , 5 \times 1 0 ^ { - 5 }$ and $2 \times 1 0 ^ { - 6 }$ for ActivityNet Captions, Charades-STA, and TVR, respectively. All models are implemented in PyTorch and trained with AdamW using $\beta _ { 1 } \mathrm { = } 0 . { \dot { 9 } } , \beta _ { 2 } \mathrm { = } 0 . 9 8 , \epsilon { \dot { = } } 1 0 ^ { - 8 }$ and a weight decay of $1 0 ^ { - 4 }$ applied to all parameters except layer-norm scales and bias terms. We use a linear warm-up over the first 5 TRACE-stage epochs followed by a cosine decay to 10% of the peak learning rate over the remaining epochs, and a gradient clip of 1.0 applied to the global norm. The mini-batch size is 128 on eight NVIDIA A800-SXM4- 80GB GPUs using bfloat16 mixed-precision training. The software environment is Ubuntu 22.04.5 LTS with Python 3.10.18, PyTorch 2.6.0+cu126, CUDA 12.6, and NVIDIA driver 550.127.08. We follow the standard preprocessing inherited from each backbone and do not introduce additional data augmentation in TRACE. Each reported experiment is run three times with independent seeds, and the released training configuration records the corresponding seed settings. Final evaluation checkpoints are obtained by snapshot averaging the last three epochs of TRACE-stage training, a common practice that reduces single-seed variance relative to last-epoch-only reporting without changing the mean. We will release code, pretrained checkpoints, and evaluation scripts upon acceptance.

<table><tr><td>Variant</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@100</td><td>SumR</td><td>∆SumR</td><td>Rank↑</td><td>Rank↓</td><td>Mean Gain</td></tr><tr><td>Base branch</td><td>17.4</td><td>39.0</td><td>50.4</td><td>86.2</td><td>193.1</td><td></td><td></td><td></td><td></td></tr><tr><td>TRACE</td><td>17.8</td><td>39.4</td><td>50.7</td><td>86.7</td><td>194.6</td><td> $\mathbf { + 1 . 5 }$ </td><td>3518</td><td>1292</td><td>+1.76</td></tr><tr><td>Uniform  $q {  } R$ </td><td>17.5</td><td>39.2</td><td>50.6</td><td>86.5</td><td>193.9</td><td>+0.8</td><td>2864</td><td>1847</td><td>+0.74</td></tr><tr><td>Cross-video shuffle  $q {  } R$ </td><td>17.4</td><td>39.1</td><td>50.5</td><td>86.4</td><td>193.4</td><td>+0.3</td><td>2416</td><td>2189</td><td>+0.37</td></tr><tr><td>Shuffle  $R { \to } F$ </td><td>17.2</td><td>38.9</td><td>50.2</td><td>86.3</td><td>192.6</td><td>-0.5</td><td>2527</td><td>3725</td><td>-0.24</td></tr></table>

Table 9: Routing corruption diagnostics on TVR. ∆SumR is measured relative to the base branch. Rank↑ and Rank↓ count queries whose ground-truth video rank is improved or degraded, respectively. Uniform $q {  } R$ removes query-specific register selection, cross-video shufling further breaks the query-register correspondence, and shufling $R {  } F$ corrupts the grounding from registers to frame evidence.

## B.2 Theoretical Properties of Hierarchical Evidence Concentration

We analyze the core concentration operator to clarify its relation to standard PRVR scoring and its ability to correct spurious local peaks. Throughout this section, cosine similarities such as cos $( q , f _ { i } )$ are treated as unnormalized energy scores under the energy-based interpretation of Eq. (3), not as proper probabilities; this convention follows the standard practice in PRVR analysis and keeps the notation aligned with the main text. We first collect the symbols used below, then state the setup and the main correction result.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td></td><td>query embedding (single vector)</td></tr><tr><td> $\underset { - } { \overset { q } { F } } = \{ f _ { 1 } , \dots , f _ { F } \} _ { \mathrm { \scriptscriptstyle , } }$ </td><td>per-frame visual token sequence of length F</td></tr><tr><td> $\boldsymbol { R } = \left\{ \boldsymbol { r } _ { 1 } , \ldots , \boldsymbol { r } _ { N _ { r } } \right\}$ </td><td>global register set of size Nr</td></tr><tr><td> $d _ { i } = \cos ( q , f _ { i } ) _ { , }$ </td><td>direct query-frame evidence at frame i</td></tr><tr><td> $a _ { k } = \cos ( \tilde { q } , \tilde { r } _ { k } )$ </td><td>query-register affinity at register k</td></tr><tr><td> $b _ { k , i } = \cos ( r _ { k } , \dot { f } _ { i } )$ </td><td>register-frame compatibility at register k, frame i</td></tr><tr><td> $\tau _ { r } > 0$ </td><td>register activation temperature</td></tr><tr><td> $\begin{array} { r } { \pi _ { k } = \exp ( a _ { k } / \tau _ { r } ) / \sum _ { k ^ { \prime } } \exp ( a _ { k ^ { \prime } } / \tau _ { r } ) } \end{array}$ </td><td>query-conditioned register weight</td></tr><tr><td> $\begin{array} { r } { \rho _ { i } = { \tau _ { r } } \log \sum _ { k } \exp ( ( a _ { k } + b _ { k , i } ) / \tau _ { r } ) } \end{array}$ </td><td>routed support at frame i</td></tr><tr><td>γ &gt; 0</td><td>routing strength in  $S _ { \mathrm { r o u t e } }$ </td></tr></table>

Table 10: Notation used in Technical Supplement, Sec. B.2. All symbols match the main-text definitions (Eqs. (4)–(6)), with cosine similarities treated as energy scores rather than probabilities.

Setup. We consider a single query q and a single video V with frame tokens $F = \{ f _ { 1 } , \ldots , \bar { f } _ { F } \}$ and global registers $\textit { R } = \ \{ r { } _ { 1 } , \ldots , r { } _ { N _ { r } } \}$ (Table 10 collects all symbols used below). The standard PRVR local selector is $S _ { \operatorname* { m a x } } ( q , V ) = \operatorname* { m a x } _ { i } d _ { i }$ with $d _ { i } = \cos ( q , f _ { i } )$ , and the TRACE routed selector is $S _ { \mathrm { r o u t e } } ( q , V ) = \mathrm { m a x } _ { i } [ d _ { i } + \gamma \rho _ { i } ]$ with $\rho _ { i } =$ $\tau _ { r }$ log $\begin{array} { r } { \sum _ { k = 1 } ^ { N _ { r } } \exp ( ( a _ { k } + b _ { k , i } ) / \tau _ { r } ) } \end{array}$ , where $a _ { k } \ = \ \cos ( q , r _ { k } )$ and $b _ { k , i } \stackrel { \cdot \cdot } { = } \cos ( r _ { k } , f _ { i } )$ . We make the following mild assumptions: (i) all cosine similarities lie in $[ - 1 , 1 ] , \mathsf { s o } d _ { i } , a _ { k } , b _ { k , i }$ are bounded; (ii) the selected frame indices $i ^ { + }$ and $i ^ { - } \mathrm { i n } V ^ { + }$ and $V ^ { - }$ are those that achieve the inner maximum, which is welldefined because F is finite; and (iii) the routing strength $\gamma$ and temperature $\tau _ { r }$ are positive hyperparameters chosen once and held fixed during inference. No assumption is required on the distribution of $d _ { i } , a _ { k }$ , or $b _ { k , \ast }$ <sub>i</sub> beyond boundedness. Standard max-style PRVR scoring can be written as

$$
S _ { \operatorname* { m a x } } ( q , V ) = \operatorname* { m a x } _ { i } d _ { i } .\tag{12}
$$

This form is suitable for partial relevance because only one moment may match the query. However, it is also a degenerate evidence selector: the selected frame only needs to maximize local query similarity, without being verified by any queryrelevant global evidence. Thus, a negative video with one coincidentally similar local fragment can dominate the final ranking.

TRACE changes this selection rule by adding textconditioned global support. Recall that the routed support for frame i is

$$
\rho _ { i } = \tau _ { r } \log \sum _ { k = 1 } ^ { N _ { r } } \exp \left( ( a _ { k } + b _ { k , i } ) / \tau _ { r } \right) ,\tag{13}
$$

where $a _ { k } = \cos ( q , r _ { k } )$ measures query-register afinity and $b _ { k , i } = \cos ( r _ { k } , f _ { i } )$ measures register-frame compatibility. For $x _ { k } = a _ { k } + b _ { k , i }$ , log-sum-exp satisfies

$$
\operatorname* { m a x } _ { k } x _ { k } \leq \tau _ { r } \log \sum _ { k } \exp ( x _ { k } / \tau _ { r } ) \leq \operatorname* { m a x } _ { k } x _ { k } + \tau _ { r } \log N _ { r } ,\tag{14}
$$

Thus, register-path aggregation is a bounded smooth approximation to selecting the strongest latent $q {  } R {  } F$ path. The outer temporal maximum then gives the hierarchical operator:

$$
S _ { \mathrm { r o u t e } } ( q , V ) = \operatorname* { m a x } _ { i } \left[ d _ { i } + \gamma \rho _ { i } \right] ,\tag{15}
$$

Compared with raw max pooling, Eq. 15 selects a local frame only after considering whether there exists a queryrelevant global anchor that also supports this frame, while retaining a smooth optimization surface over latent register choices. The approximation gap at each frame is bounded by $\tau _ { r }$ log $N _ { \tau }$ <sub>r</sub> and the corresponding routed-score gap is bounded by γτ<sub>r</sub> log $N _ { r }$

The operator exposes several meaningful reductions. When $\gamma = 0$ , Eq. 15 exactly reduces to standard max-style PRVR scoring. When the activation scores $a _ { k }$ are queryindependent, the inner aggregation becomes query-agnostic global support, corresponding to the uniform-activation diagnostic in Table 3. When $b _ { k , i }$ is removed, register evidence no longer distinguishes temporal locations and reduces to a global query-register bias. Finally, replacing the outer maximum with a temporal average yields the diluted-evidence variant in Table 3. These cases isolate the three defining operations of TRACE: query-conditioned path selection, registerto-frame grounding, and localized temporal concentration.

Theorem 1 (Pairwise correction condition). Consider a positive video $V ^ { + }$ and a hard negative video $V ^ { - }$ . Let $d ^ { + }$ and $d ^ { - }$ be their strongest direct local evidence scores, and let $\rho ^ { + }$ and $\rho ^ { - }$ be the corresponding routed support values at these selected local peaks. Suppose the raw local selector ranks the hard negative above the positive video, i.e.,

$$
d ^ { - } > d ^ { + } .\tag{16}
$$

For this pair of selected peaks, the routed selector reverses their pairwise order whenever

$$
\gamma ( \rho ^ { + } - \rho ^ { - } ) > d ^ { - } - d ^ { + } .\tag{17}
$$

Proof. Under the routed selector, the positive video is ranked above the hard negative if

$$
d ^ { + } + \gamma \rho ^ { + } > d ^ { - } + \gamma \rho ^ { - } .\tag{18}
$$

Subtracting $d ^ { - }$ from both sides and rearranging gives

$$
\gamma ( \rho ^ { + } - \rho ^ { - } ) > d ^ { - } - d ^ { + } ,\tag{19}
$$

which is exactly Eq. 17. □

This condition directly explains the intended top-rank correction behavior. A hard negative may win under $S _ { \mathrm { m a x } }$ because $d ^ { - }$ is slightly larger than $d ^ { + }$ . TRACE can recover the positive video in this pairwise comparison if the local evidence in $V ^ { + }$ is better supported by query-activated global registers, $\mathrm { i . e . , } \rho ^ { + } > \rho ^ { - }$ . Conversely, if a hard negative has both stronger local evidence and comparable or stronger routed support, TRACE should not force a correction. This is why TRACE is best understood as targeted evidence verification rather than a uniform score shift.