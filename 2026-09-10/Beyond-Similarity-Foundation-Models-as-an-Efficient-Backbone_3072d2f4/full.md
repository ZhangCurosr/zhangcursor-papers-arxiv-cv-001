# Beyond Similarity: Foundation Models as an Efficient Backbone for Training-Free Composed Video Retrieval

Dmitry Demidov

Muhammad Zaigham Zaheer

Omkar Thawakar

dmitry.demidov@mbzuai.ac.ae

zaigham.zaheer@mbzuai.ac.ae

omkar.thawakar@mbzuai.ac.ae

Abdelrahman Mohamed Shaker

Rao Anwer

abdelrahman.youssief@mbzuai.ac.ae

rao.anwer@mbzuai.ac.ae

Mohamed bin Zayed University of Artificial Intelligence, UAE

## Abstract

Composed video retrieval (CoVR) searches a galleryfor the target video that realizes a natural-language modification of a source clip. However, at gallery scale, this creates a <sub>fundamental tension: compact embeddings enable efficient,</sub>fi reusable search but can miss the transient actions, state changes, and subtle constraints that demand fine-grained video reasoning, whereas applying large multimodal models uniformly sacrifices scalability. To address these limitations, we propose that frozen foundation models should instead occupy complementary roles, with inference depth adapted to query difficulty. Based on this premise, we introduce CoVRAGE, aframeworkfor training-free Composed Video Retrieval with Adaptive Gated Escalation. Specifically, a composed-query embeddingfirst searches reusable video-only gallery representations; uncertain queries undergo bounded reranking and candidate expansion; ambiguous edits trigger target-description generation; and only close leading candidates reach multimodal verification. To support these roles,frame selection, spatial resolution, and time cues are adapted to each stage. Across complete targetgallery evaluations, our method reaches state-of-the-art performance among training-free approaches, with 89.55 and 93.43 R@1 on Dense-WebVid-CoVR and CoVR-R, respectively (with more than +35% and +25% absolute margins to the closest counterpart). These results show that adaptively orchestrating foundation-model capabilities can combine scalable retrieval withfine-grained reasoning without taskspecific training. The source code and all relevant guidelines are available on github.com/demidovd98/CoVRAGE.

## 1. Introduction

The composed video retrieval (CoVR) task aims to search a gallery for the desired target video that satisfies a naturallanguage modification to a reference clip. It offers an intuitive form of example-based search: the source specifies what to retain, while the edit states what to change, without requiring a complete description of the target. Composed image retrieval has progressed from learned composition of visual and textual features and free-form benchmarks to zero-shot and large-model-assisted formulations [3, 15, 28, 37, 40], and retrieval across video and language has shown that pretrained cross-modal representations can support reusable gallery search [2, 45]. Video composition, however, depends on temporal evidence: a decisive event may occupy only a few frames, and action order, duration, or a fleeting state change may distinguish otherwise similar clips. A useful system must therefore resolve fine-grained intent across time without applying expensive multi-frame reasoning to every gallery item [35, 36, 38].

![](images/abac70d2ae8f4eb380407314e1d33b56f56ce970d0d2afd90a9835dde38e74a6.jpg)

![](images/2a03944bd5b8ed4f71d0c9aeddc828e8dfaa6dc068dfadc80f2eea4272c8855e.jpg)  
Figure 1. Two different pipeline paths through our confidence-gated cascade. A confident case skips heavy verification stages, while an ambiguous case utilizes reranking and strict verification to correct the top prediction after coarse retrieval favours a distractor. The footer summarizes complete accuracy and stage-wise workload spread on the CoVR-R dataset.

Recent CoVR systems address increasingly sparse, dense, egocentric, temporal, and reasoning-oriented edits through task-specific fusion, specialized objectives, generated supervision, or contextual descriptions [14, 34–36, 38, 39]. Frozen foundation models, target-description generation, and largemodel reranking have further improved the available inference tools [13, 23, 36]. Yet these advances expose a central tension. Large-scale pretraining yields broadly transferable representations [32, 48], and specialized embedders can compress a source video and edit into an indexable query for efficient comparison with reusable gallery vectors [20, 24]. Such embeddings can nevertheless lose fine-grained visual information [21]; in CoVR, that bottleneck can obscure a transient action, subtle relation, or negative constraint. Applying a large reasoning model uniformly avoids this bottleneck only by multiplying computation over candidates and frames. Moreover, later reasoning can repair a ranking error only if coarse retrieval keeps the target in its candidate pool. Representation, candidate depth, and reasoning policy are therefore coupled: the representation best suited to broad search is not always the one that should make the final decision. Therefore, the resulting question is not whether one foundation model can solve CoVR, but rather how complementary capabilities should be allocated across retrieval, refinement, and reasoning.

Our premise is that frozen foundation models are most effective when assigned specialized roles in an adaptive coarseto-fine cascade. Figure 1 contrasts its shallow exit with its deeper corrective path. We co-design role, routing, and reuse: each capability addresses a particular retrieval failure, confidence determines which queries and candidates receive it, and query-independent target processing is amortized. Rather than committing every query to a fixed inference depth, the cascade treats confidence as evidence that the next capability is warranted. Confident queries terminate after compact retrieval or shallow refinement; ambiguous queries receive progressively finer candidate scoring, semantic decomposition, and multimodal verification. Stage-aware temporal evidence determines what each activated role sees. In this way, inference depth follows query difficulty: retrieve broadly, refine selectively, and reason only where ambiguity demands it.

We realize this principle with our CoVRAGE, a framework for training-free Composed Video Retrieval with Adaptive Gated Escalation. Each gallery video is encoded once, after which a composed-query embedding retrieves a compact candidate pool. A confidence gate bypasses refinement for well-separated results; otherwise, a reranker scores the shortlist and expands it only when uncertainty persists. Terse or ambiguous edits can be converted once into a targetoriented description, and only a close leading cluster reaches strict multimodal verification. Because these stages resolve different ambiguities, their visual evidence follows the same progression: compact retrieval uses a stable uniform view, while later stages receive dynamic temporal-novelty samples, stage-specific resolution, and elapsed-time cues for irregularly spaced frames. Dynamic selection targets temporal change without increasing every frame budget, while timestamps expose elapsed time under irregular sampling. Modular interfaces separate embedding, scoring, generation, and verification. The framework thus preserves a reusable, query-independent gallery path and concentrates candidateconditioned reasoning on a small, progressively refined set.

Across complete target-gallery evaluations, the framework reaches 54.58 R@1 on WebVid-CoVR, 89.55 on Dense-WebVid-CoVR [36], and 93.43 on CoVR-R. The latter two are the highest displayed training-free values under the evaluated public benchmark protocols, including significant absolute gains of 28.34-point R@1 and 37.95-point R@1, respectively. Ablations show that the stages correct distinct failure modes while preserving the reusable-gallery design. Together, these findings show that adaptive allocation can combine gallery-scale retrieval with the fine-grained reasoning needed for difficult edits.

The main contributions of this research work are:

• We introduce CoVRAGE, a fully training-free, rolespecialized cascade combining reusable gallery encoding, compact retrieval, reranking, decomposition, and verification without task-specific parameter updates.

• We couple confidence-controlled depth with stageaware temporal evidence, allocating cheap candidate search, moderate visual detail refinement, and expensive reasoning according to ambiguity.

• We provide a controlled stage-wise study and complete cross-dataset evaluation, while clarifying when and why each component helps.

• We obtain the state-of-the-art results in training-free setup on the evaluated Dense-WebVid-CoVR [35] and CoVR-R [36] benchmarks.

## 2. Related Work

## 2.1. Composed Image and Video Retrieval

Composed image retrieval represents a target from a reference image and a language edit. TIRG established explicit composition of visual and textual features [40]; CIRR introduced natural images and free-form edits [28], and CIRCO extended evaluation to zero-shot, multi-positive retrieval [3]. Dual encoders for video and text independently showed that target videos can be encoded once and searched efficiently [2]. CoVR joined these lines by retrieving a target video from a source visual and an edit, initially with tasktrained BLIP-family composition and automatically constructed triplets [38, 39]. Video adds a systems problem: decisive evidence may occupy few frames, temporal order can change the event, and query-dependent comparisons multiply multi-frame inference. Egocentric, reason-aware, and omni-modal variants further emphasize localized, implicit, or cross-modal changes [14, 17, 36]. Thus frame allocation, temporal evidence, and candidate cost are part of the retrieval problem, not incidental implementation details.

## 2.2. Supervised Composed Video Retrieval

Direct CoVR methods have advanced through increasingly specialized supervision. Early systems fine-tune encoders and fusion modules with generated triplets, often enriching the source with generated context or dense descriptions [34, 35, 38, 39]. Later work models temporal actions, shared and differential semantics, or alignment among the source, edit, and target with task-specific losses and heads [7, 11, 43, 49]. Other designs introduce uncertainty tokens, directional calibration, hierarchical editing, prompt modules, latent compositional schemas, audio, or rank-aware interpolation [5, 10, 12, 16, 22, 41, 47]. Foundation backbones also yield strong composed embeddings after instruction fine-tuning [18]. The adapted UniCVR system learns query alignment from pseudo-triplets and conditionally deepens candidate assessment [42], while interactive retrieval introduces an additional feedback loop [46]. These methods improve accuracy and scope, but their task-derived supervision, optimized heads, or generated training data address how to learn a CoVR model. Nevertheless, such approaches do not establish how frozen embedding, scoring, generation, and verification capabilities should be assigned and compared under one inference boundary.

## 2.3. Video Retrieval with Foundational Models

Instruction-tuned multimodal embedders map text, images, videos, and mixed-modal inputs into indexable representations [24, 29]. Dedicated embedding and reranking checkpoints further separate broad recall from fine-grained candidate scoring [20]. Universal retrieval work additionally studies modality bridging, preservation of visual identity, and any-to-any search across audio, video, and text [4, 26, 48]. Video-retrieval systems use these capabilities through generic video training, adaptive visual representations, or alignment to a frozen gallery [8, 9, 42]. Their supervision ranges from off-the-shelf inference to substantial adaptation, and recent diagnostics examine where videolanguage reasoning fails within a clip [44].

The closest pipelines validate individual ingredients under different policies. CoVR-R performs generative reason-thenretrieve over reusable gallery representations [36]. MoRe adds exhaustive pairwise judgments to multi-objective recall [13], while $\bar { \mathsf { R } ^ { 3 } }$ combines specialized embeddings, largemodel reasoning, and fixed-depth reranking [23]. Parallel challenge systems explore variants that reason, retrieve, and rerank, dual-route recall, fusion of dense and sparse representations, and visual-guided video-LLM reasoning [1, 25, 27, 33]. These systems establish the value of reasoning, reusable recall, and candidate judgment. Our distinction is to study their allocation jointly: reusable gallery encoding, confidence-controlled depth, early and deferred decomposition, relative-cluster verification, temporal evidence, and measured activation within one fully training-free boundary.

## 2.4. Efficient and Adaptive Multimodal Inference

Coarse-to-fine retrieval established the efficiency pattern of fast indexed recall followed by slower interaction over a shortlist [30]; specialized multimodal embedders and rerankers provide a modern instance [20]. Adaptive video encoders and long-video agents also vary visual computation or revisit selected evidence [8, 19]. Adapted staged CoVR systems add pointwise assessment after dense search [42], while training-free methods generate target semantics or compare shortlisted candidates [13, 23]. Fixed candidate depths can nevertheless make multi-frame cost grow linearly or quadratically with pool size. Our policy instead uses independent signals: edit length launches one reusable description, the embedding gap can bypass refinement, the reranker score controls expansion, its margin can activate deferred generation, and a relative-score cluster controls verification. Query-independent novelty sampling selects the evidence seen downstream. This follows the broader principle of using embeddings for routine similarity and reserving large-model judgment for hard decisions [6].

Against this background, we provide a comprehensive stage-wise study of direct CoVR. Representation, capacity, routing, candidate depth, temporal evidence, resolution, and prompting are evaluated under one training-free pipeline and target-gallery boundary. The framework uses modular stage interfaces: reusable vectors support broad search, while confidence signals bound refinement and reserve generation and verification for ambiguity. Its contribution is the joint design and measurement of role, routing, and temporal evidence. Under the evaluated benchmark protocols, this design attains the highest training-free results on Dense-WebVid-CoVR and CoVR-R while controlling inference cost.

## 3. Method

## 3.1. Pipeline Overview

Let $v ^ { r }$ be a source video and m a free-form modification describing the desired change. A video gallery $\mathcal { G } = \{ v _ { i } ^ { t } \} _ { i = 1 } ^ { N } ,$

![](images/b84d33e4b382914b99608fc5716b3f94422c3cf0dacf12809a24f11152675b09.jpg)  
Figure 2. Overview of the training-free retrieval cascade. Reusable gallery encoding supports global search, while confidence gates route only uncertain queries through bounded candidate scoring, target-description generation, and strict multimodal verification. Skipped or unresolved conditional stages retain the preceding valid ranking.

and $v ^ { \star } \in \mathcal G$ is the target video that realizes m relative to $v ^ { r }$ Given the composed query $\boldsymbol { q } = \left( \boldsymbol { v } ^ { r } , m \right)$ , Equation 1 defines the descending gallery order and target rank:

$$
\begin{array} { r l r } & { } & { \qquad \pi _ { q } = \underset { v _ { i } ^ { t } \in \mathcal { G } } { \arg \operatorname { s o r t } S ( q , v _ { i } ^ { t } ) } , } \\ & { } & { \quad \quad \mathrm { r a n k } _ { \pi _ { q } } ( v ^ { \star } ) = 1 + \underset { v _ { i } ^ { t } \neq v ^ { \star } } { \sum } \mathbf { 1 } \big [ S ( q , v _ { i } ^ { t } ) > S ( q , v ^ { \star } ) \big ] . } \end{array}\tag{1}
$$

Our goal is to place $v ^ { \star }$ first without task-specific parameter updates. Our CoVRAGE assigns complementary roles to frozen models: composed-query and gallery encoders $E _ { q } , E _ { g }$ , a candidate relevance scorer R, a target-description generator $D ,$ and a multimodal verifier V . Their interfaces are modular. Gallery videos are encoded independently of any query, making their representations reusable; the cascade then transforms the source and edit into progressively more discriminative evidence over a bounded candidate set.

As illustrated in Figure 2, composed-query embedding searches the complete gallery. Confident results exit, while uncertain queries undergo bounded reranking and expansion. Conditional decomposition supplies an explicit target description, and verification compares source, edit, and candidate only for the remaining near-ties. A skipped or invalid conditional stage preserves the preceding order. Stage-aware inputs combine uniform coverage with novelty-focused evidence. The complete execution flow is detailed in App. A.1, and concrete interfaces and settings are listed in App. C.

## 3.2. Query Embedding and Coarse Retrieval

Coarse retrieval maps the query formed by the source and edit, as well as each target, into a shared space. The query encoder jointly represents a uniform source view and the raw modification; the gallery encoder receives only a target video under a separate neutral instruction. The query thus carries the transformation while target vectors remain reusable:

$$
\begin{array} { r l r } & { z _ { q } = \mathrm { n o r m } ( E _ { q } ( F _ { U } ( v ^ { r } ) , m ) ) , } \\ & { z _ { i } = \mathrm { n o r m } \big ( E _ { g } ( F _ { U } ( v _ { i } ^ { t } ) ) \big ) , \ } & { e _ { i } = z _ { q } ^ { \top } z _ { i } . } \end{array}\tag{2}
$$

In Equation 2, $F _ { U }$ is uniform sampling and $\operatorname { n o r m } ( x ) =$ $x / \Vert x \Vert _ { 2 }$ . Similarities $e _ { i }$ define the initial order after masking a source that reappears as a non-target. Later stages receive only a small leading pool, while the complete order remains their fallback and untouched tail. Coarse retrieval therefore emphasizes gallery-wide recall; candidate-wise models handle the routed subset. Encoder interfaces, target masking, and boundary cases are detailed in App. A.3, with concrete settings in App. C.

## 3.3. Confidence-Gated Adaptive Reranking

For uncertain queries, a specialized multimodal scorer evaluates each candidate against either the raw modification or one source-conditioned target description; the source video is absent. It returns a continuous relevance value $r _ { i }$

Let $\pi ^ { E }$ be the embedding order, P the current candidate pool, and $\pi ^ { R }$ the resulting order. A separated embedding leader bypasses reranking. Otherwise, the initial pool expands in fixed increments only while its best candidate-wise score remains weak:

$$
\begin{array} { r l } & { e _ { 1 } - e _ { 2 } > \tau _ { E } \Rightarrow \pi ^ { R } = \pi ^ { E } , } \\ & { e _ { 1 } - e _ { 2 } \leq \tau _ { E } \Rightarrow P  P \cup \mathrm { n e x t } _ { \Delta K } ( \pi ^ { E } ) , } \\ & { \qquad \mathrm { w h i l e ~ } \underset { i \in P } { \operatorname* { m a x } } r _ { i } < \tau _ { R } , ~ \vert P \vert < K _ { \operatorname* { m a x } } . } \end{array}\tag{3}
$$

Under Equation 3, easy queries take a shallow path, while ambiguous ones draw more candidates from the embedding order. Scored items are reordered by $r _ { i } ,$ while the unscored tail preserves a complete ranking. Provisional scores support deferred decomposition before the final text-conditioned order. Bypass or invalid evidence retains the embedding order. Full routing and fallback semantics are detailed in App. A.4, with concrete thresholds in App. C.

## 3.4. Conditional Query Decomposition

For short or implicit edits, decomposition generates a concise target description from the source and raw modification. This candidate-independent description replaces only the reranker’s text, while embedding and verification retain the original edit.

Let $W ( m )$ measure edit length and $r _ { 1 } , r _ { 2 }$ be the leading provisional reranker scores. Generation uses complementary early and late tests (one equation is broken down into three rows for readability, it reads top to bottom as one equation):

$$
\begin{array} { c } { { \mathrm { r u n } D } } \\ { { \iff } } \\ { { W ( m ) < \tau _ { W } \lor \left( r _ { 1 } \geq \tau _ { D } \land \frac { r _ { 1 } - r _ { 2 } } { r _ { 1 } } < \tau _ { M } \right) . } } \end{array}\tag{4}
$$

In Equation 4, the first branch handles terse edits before scoring and the second handles plausible but poorly separated raw-text candidates. Deferred generation triggers one final reranking pass. Failed or empty output retains the raw edit. Exact boundaries, reuse, and scheduling are detailed in App. A.5.

## 3.5. Confidence-Gated Candidate Verification

For residual near-ties, a generative multimodal verifier receives the source, raw modification, and one candidate and returns a strict full-match verdict. Missing, contradicted, or uncertain requested conditions are rejected; unconstrained attributes need not match.

Verification is restricted to a contiguous leading ambiguity cluster. For reranker scores $r _ { 1 } \geq r _ { 2 } \geq \cdots$

$$
\begin{array} { c } { { \displaystyle { \mathcal { C } = \mathrm { p r e f i x } \bigg \{ i \leq K _ { V } : \frac { r _ { 1 } - r _ { i } } { r _ { 1 } } < \tau _ { V } \bigg \} , } } } \\ { { \mathrm { r u n } V \Longleftrightarrow r _ { 1 } \geq \tau _ { S } \wedge | { \mathcal { C } } | \geq 2 . } } \end{array}\tag{5}
$$

Under Equation 5, candidates in C are checked top-down. The first acceptance is stably promoted and stops the scan; otherwise the reranked order remains. The score floor excludes implausible pools and the relative-gap rule excludes separated candidates. Exact gate and failure behavior are detailed in App. A.6, and the prompt contract and limits are listed in App. C.

## 3.6. Stage-Aware Frame Allocation

The cascade tailors visual evidence to each role. Globa embedding uses equidistant clip views, whereas candidate scoring and generative reasoning use a query-independent novelty view derived from lightweight frame embeddings. This preserves reusable gallery indexing while focusing later stages on temporal change.

Let $h _ { i }$ denote the normalized embedding of temporal probe $i , \delta _ { i } = \operatorname* { m a x } ( 0 , 1 - h _ { i } ^ { \top } h _ { i - 1 } )$ its adjacent novelty, and B the downstream frame budget. The selected probe for slot k is obtained from the inverse cumulative novelty distribution,

$$
\begin{array} { l } { \displaystyle p _ { i } = \left\{ \begin{array} { l l } { \displaystyle \delta _ { i } / \sum _ { j } \delta _ { j } , } & { \sum _ { j } \delta _ { j } > 0 , } \\ { \displaystyle 1 / M , } & { \sum _ { j } \delta _ { j } = 0 , } \end{array} \right. } \\ { \displaystyle w _ { i } = \lambda p _ { i } + \frac { 1 - \lambda } { M } , } \\ { \displaystyle s _ { k } = \operatorname* { m i n } \Biggl \{ i : \sum _ { j \leq i } w _ { j } \geq \frac { k - \frac { 1 } { 2 } } { B } \Biggr \} , \quad k = 1 , \dots , B . } \end{array}\tag{6}
$$

![](images/bb61872829562768160af186b1a7829d11cbcaf3b0f117f7c3e4b9f8f1d09307.jpg)  
Figure 3. Dynamic frame sampling on a representative animation. Six of the selected views are shown for each policy. Uniform spacing repeatedly samples the long title and green-screen holds; novelty weighting reallocates views to cube assembly, title reveal, the screen transition, and final disassembly. The stock watermark is retained.

In Equation 6, the explicit fallback makes w<sub>i</sub> uniform when novelty vanishes. Inverse-CDF allocation covers the temporal support of change rather than only the largest transitions; its ordered, query-independent indices are reusable for targets. The probe schedule, duplicate handling, spatial settings, and edge cases are detailed in App. A.2.

In Figure 3 we illustrate how novelty weighting in dynamic frame sampling reallocates views relative to naive uniform frame sampling.

## 3.7. Timestamp-Aware Conditioning

Irregular samples do not reveal elapsed time from frame order alone. The two generative stages therefore receive source-relative times identified as metadata rather than scene content, supporting ordered descriptions and cross-video temporal comparison. Embedding and reranking remain annotation-free; rendering and prompt details are given in App. A.7.

## 4. Experiments and Analysis

## 4.1. Experimental Setup

## 4.1.1. Datasets and Evaluation Protocol

We evaluate on WebVid-CoVR, whose edits are short and automatically derived [38]; Dense-WebVid-CoVR, which adds detailed attributes, actions, and temporal relations [35]; and CoVR-R, which combines WebVid and Something-Something-V2 examples requiring more implicit reasoning [36]. All evaluations use a target-video gallery with source self-masking. We report R@1, R@5, R@10, R@50, their four-cutoff average (Avg.), and MeanR3 over R@1, R@5, and R@10 for controlled ablations. Table S1 in App. B.1.1 summarizes the evaluated splits and shared gallery/selfmasking protocol, and also defines subset provenance and auxiliary metrics.

## 4.1.2. Implementation Details

The system is training-free: a compact encoder retrieves from reusable video-only gallery representations, a candidate scorer refines a confidence-dependent pool, and description generation and strict verification are invoked only for ambiguous cases. Coarse retrieval uses a uniform temporal view, whereas candidate-level and reasoning stages receive novelty-weighted frames; the generative stages also receive visible timestamps. Tables S15 and S16 in App. C list model identities, thresholds, frame and resolution settings, prompts, and generation parameters; failure handling and execution details are given in App. C.1.

## 4.2. Quantitative Results

As shown in Table 1, our method has the highest displayed training-free R@1 on the complete Dense-WebVid-CoVR and CoVR-R evaluations, reaching 89.55 and 93.43, respectively. On WebVid-CoVR, it reaches 54.58 R@1. Prior papers use “zero-shot” for both frozen inference and crossdataset transfer; we therefore separate rows by whether the reported configuration learns CoVR-specific parameters rather than by the source’s section heading. WebVid-CoVR’s modification candidates were machine-generated and its test examples were manually selected or filtered, but many retained edits remain short and underspecified [36, 39]. This annotation style may partly explain the remaining gap: sparse surface cues can reward literal similarity, while a reasoning cascade can overinterpret an ambiguous request.

## 4.2.1. Cross-Dataset Performance

The largest displayed gaps occur on benchmarks with dense and reasoning-heavy edits: on CoVR-R, the method exceeds the highest other source-reported training-free R@1 retained in Table 1 by 37.95 points. Against MoRe specifically, CoVRAGE is substantially stronger on Dense-WebVid-CoVR: 89.55 versus 49.6 R@1, 95.15 versus 67.0 R@5, and 96.40 versus 77.9 R@10. Unlike the original WebVid-CoVR annotations, the Dense-WebVid-CoVR test modifications are detailed, fully manually verified, and corrected when needed [35], making this a more realistic evaluation of specific user intent. MoRe leads on noisy WebVid-CoVR but trails sharply on its higher-quality dense counterpart. This reversal is consistent with different sensitivity to annotation quality, although the published aggregate results do not establish its precise cause. Task-trained or adapted systems are a contextual, non-resource-equivalent comparison and are reported separately in Table S2 in App. B.2.1; Table S5 there provides our full cutoff-by-cutoff metrics.

Table 1. Training-free approaches. The evaluation protocol assumes the target gallery with source self-masking. The Avg. column is computed from the displayed cutoffs and rounded to two decimals; missing cutoffs are not imputed. Within each dataset, boldface and underlining mark the best and second-best available values. <sup>∗</sup> marks WebVid-CoVR, which includes unverified machine-generated modification requests, which are often too short or generic [35, 36, 39]. CoVR-R challenge-server results using a different validation/test split are excluded.
<table><tr><td>Dataset</td><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>Avg.</td></tr><tr><td rowspan="4">CoVR-R [36]</td><td>CoVR-R (Qwen3-VL-8B + reasoning) [36]</td><td>49.88</td><td>66.99</td><td>72.97</td><td>85.14</td><td>68.75</td></tr><tr><td>CoVR-R (Qwen3-VL-8B, five-round refinement) [36]</td><td>50.56</td><td>74.03</td><td>81.24</td><td>92.17</td><td>74.50</td></tr><tr><td>CoVR-R (Qwen3-VL-72B) [36]</td><td>55.48</td><td>72.69</td><td>78.57</td><td>87.99</td><td>73.68</td></tr><tr><td>CoVRAGE (ours)</td><td>93.43</td><td>94.27</td><td>94.61</td><td>94.95</td><td>94.31</td></tr><tr><td rowspan="6">Dense-WebVid-CoVR [35]</td><td>CoVR-BLIP (Avg.) [38]</td><td>38.44</td><td>64.96</td><td>71.72</td><td>87.12</td><td>65.56</td></tr><tr><td>ECDE (Avg.) [34]</td><td>40.23</td><td>66.38</td><td>74.84</td><td>88.12</td><td>67.39</td></tr><tr><td>BSE-CoVR (Avg.) [35]</td><td>42.41</td><td>68.54</td><td>77.07</td><td>91.24</td><td>69.82</td></tr><tr><td>CoVR-R (Qwen3-VL-8B + reasoning) [36]</td><td>61.21</td><td>83.40</td><td>89.39</td><td>97.61</td><td>82.90</td></tr><tr><td>MoRe [13] CoVRAGE (ours)</td><td>49.60</td><td>67.00</td><td>77.90</td><td></td><td></td></tr><tr><td></td><td>89.55</td><td>95.15</td><td>96.40</td><td>97.26</td><td>94.59</td></tr><tr><td rowspan="10">WebVid-CoVR* [38]</td><td>EgoVLPv2 [31]</td><td>18.10</td><td>30.60</td><td>35.60</td><td></td><td></td></tr><tr><td>LanguageBind [50]</td><td>39.30</td><td>65.20</td><td>74.20</td><td></td><td></td></tr><tr><td>CLIP (Avg.) [32]</td><td>44.37</td><td>69.13</td><td>77.62</td><td>93.00</td><td>71.03</td></tr><tr><td>CoVR-BLIP (Avg.) [38]</td><td>45.46</td><td>70.46</td><td>79.54</td><td>93.27</td><td>72.18</td></tr><tr><td>CoVR-BLIP-2 (Avg.) [39]</td><td>45.66</td><td>71.71</td><td>81.30</td><td>94.80</td><td>73.37</td></tr><tr><td>ECDE (Avg.) [34]</td><td>47.52</td><td>72.18</td><td>82.37</td><td>95.06</td><td>74.28</td></tr><tr><td>CoVR-R [36]</td><td>49.15</td><td>70.72</td><td>79.25</td><td>93.42</td><td>73.14</td></tr><tr><td>MoRe [13]</td><td>63.00</td><td>83.40</td><td>87.60</td><td></td><td></td></tr><tr><td>TFR-CVR [14]</td><td>51.70</td><td>75.30</td><td>80.70</td><td></td><td></td></tr><tr><td>CoVRAGE (ours)</td><td>54.58</td><td>73.12</td><td>79.26</td><td>86.38</td><td>73.34</td></tr></table>

## 4.2.2. Where the Gains Arise

Candidate reranking with routed decomposition contributes most of the improvement over coarse retrieval, especially on the reasoning-heavy benchmarks. Verification supplies a smaller final correction because it only resolves the leading ambiguity cluster. R@50 is preserved by design: the reusable encoder establishes candidate coverage, while later stages reorder only a bounded head. Table S3 in App. B.2.2 reports the complete stage trajectories. A representative measured efficiency profile and final-configuration routing workloads are reported in App. B.5.

## 4.3. Qualitative Results

Qualitative behavior follows the routing policy rather than one fixed inference path. Confident queries terminate after reusable embedding or shallow refinement, whereas uncertain queries receive the particular evidence needed to resolve the remaining ambiguity. Figure 4 shows the deepest successful path: embedding and reranking preserve a plausible passenger clip ahead of the target, but timestamp-aware verification distinguishes whether the requested appearance and train-window setting are jointly satisfied.

The selected source and candidate frames are irregularly spaced and carry elapsed-time cues. Interpreted through the accompanying prompt, these cues help the verifier relate appearance and setting across each clip (with passenger picking up a phone in the middle of the video) rather than over-weighting a single passenger frame. The correction is evidence for the combined overlay-and-prompt package; it does not attribute the gain to either element in isolation.

The other routed mechanisms address earlier bottlenecks. Confidence-based expansion admits a bathtub transformation omitted from the small initial seed, after which reranking promotes it. In a controlled green-line diagnostic, targetoriented decomposition restores the missing moving-network context before retrieval and candidate scoring. These cases show that candidate depth, semantic explicitness, and temporal evidence solve complementary ambiguities. Their full stage paths are reported in App. B.3.3, and the complete qualitative casebook is provided in App. B.3.2.

## 4.4. Ablation Studies

We retain three component-wise studies that directly explain the final cascade. All use the same Dense-WebVid-CoVR subset; further studies appear in Apps. B.4.1 and B.4.5.

![](images/aa8dcfbf3cdc704edc49b5aabf1be967c86d58479b467921c98c123d7636cee7.jpg)  
Figure 4. Timestamp-aware verification resolves a temporal near-tie. The upstream candidate order is unchanged, but the combined timestamp overlay and explanatory prompt let verification reject a plausible passenger clip (without leaning and phone handling in the middle) and promote the train-window target to first.

## 4.4.1. Query Strategy Analysis

Table 2 shows that joint encoding of the source video and edit improves R@1 by 14.50 points over video-only retrieval and by 4.20 over the strongest late-fusion baseline, while giving the best MeanR3 (80.73). Equal-weight late fusion is slightly stronger at R@10 (90.10 versus 89.20), and text alone is substantially weaker (36.60 R@1), supporting source-grounded composition. Role-specific instructions add 3.50 R@1 in the

Table 2. Embedding query strategy. Comparing text only, source video only, equal-weight late fusion, and joint source-video/edit query encoding. Matched training-free runs on the Dense-WebVid-CoVR subset use a target-video gallery. Joint encoding leads R@1 and MeanR3, supporting source-grounded composition; bold marks best results.
<table><tr><td>Embedding query representation</td><td>R@1↑</td><td>MeanR3 ↑</td></tr><tr><td>Text only</td><td>36.60</td><td>51.27</td></tr><tr><td>Source video only</td><td>52.70</td><td>71.13</td></tr><tr><td>Equal-weight late fusion</td><td>63.00</td><td>79.30</td></tr><tr><td>Joint source video + edit</td><td>67.20</td><td>80.73</td></tr></table>

complete sweep. We retain a joint query and reusable target vectors; full results appear in App. B.4.1.

## 4.4.2. Score-Based Candidate Pool Expansion

Table 3 shows that uncertainty-triggered expansion raises R@1 from 86.10 to 89.50 and MeanR3 from 90.97 to 94.97.

The full sweep in App. B.4.2 leaves R@50 unchanged and isolates candidate depth rather than query interface. A small seed therefore suffices only for confident cases.

Table 3. Candidate-pool expansion for reranking. Candidate-pool ablation testing whether low-confidence queries need deeper coverage than the five-item reranker seed. Matched training-free runs on the Dense-WebVid-CoVR subset use a target-video gallery. Expansion improves R@1 and MeanR3, supporting confidence-gated candidate depth; bold marks best results.
<table><tr><td>Reranker&#x27;s pool policy</td><td>R@1↑</td><td>MeanR3 ↑</td></tr><tr><td>Fixed initial pool</td><td>86.10</td><td>90.97</td></tr><tr><td>Confidence-based expansion</td><td>89.50</td><td>94.97</td></tr></table>

## 4.4.3. Text Decomposition for Reranker

Table 4 shows a 4.80-point R@1 gain from always-on target decomposition. Combining concise-edit and ambiguity routes retains 4.10 points while activating for 26.50% of queries. The larger 20.85-point WebVid-CoVR gain supports

Table 4. Text decomposition for candidate reranking. Targetdecomposition ablation testing whether source-conditioned descriptions resolve reranker ambiguity without generation for every query. Matched training-free runs on the Dense-WebVid-CoVR subset use a target-video gallery. Selective routing handles edits under 10 words or ambiguous provisional scores, and Routed is the activatedquery percentage. Always-on decomposition is most accurate, while selective routing retains 4.10 of its 4.80-point R@1 gain at 26.50% activation, supporting the selected accuracy–workload trade-off. Best in bold, most optimal in underline.
<table><tr><td>Decomposition</td><td>R@1↑</td><td>MeanR3 ↑</td><td>Routed↓</td></tr><tr><td>Disabled</td><td>89.50</td><td>94.97</td><td>0%</td></tr><tr><td>Every query</td><td>94.30</td><td>97.20</td><td>100%</td></tr><tr><td>Selective</td><td>93.60</td><td>96.80</td><td>26.50%</td></tr></table>

decomposition for brief edits, but embedding-side injection has mixed effects; the final interface therefore confines generated text to reranking. Routing, placement, and cross-dataset results appear in Apps. B.4.3 and B.4.5.

## 5. Conclusion and Discussion

We present a fully training-free CoVR cascade that combines reusable gallery search with selectively routed reranking, decomposition, and verification under stage-specific temporal evidence. It reaches 89.55 R@1 on Dense-WebVid-CoVR and 93.43 R@1 on CoVR-R, the highest results among the compared training-free methods under these protocols (Table 1). Qualitative and ablation evidence shows complementary corrections from temporal conditioning, joint composition, candidate depth, and semantic enrichment (Figure 4).

Apps. B.3 and D provide broader analysis and limitations; routing calibration and adaptive candidate budgets remain key next steps.

## References

[1] Ali Alavi. Reason, retrieve, re-rank: A zero-shot reasoningaware framework for composed video retrieval, 2026. arXiv preprint and CoVR-R challenge report; peer review not established. 3

[2] Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisserman.¨ Frozen in time: A joint video and image encoder for end-toend retrieval. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021. 2, 3

[3] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. Zero-shot composed image retrieval with textual inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15338– 15347, 2023. 1, 3

[4] Jiawei Cao, Junyi Feng, Jiashen Hua, Ziheng Huang, Bing Deng, Kaijie Wu, Chaochen Gu, and Jieping Ye. Illuminating visual identity in universal multimodal embeddings. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8737–8748, 2026. 3

[5] Zhiwei Chen, Yupeng Hu, Zixu Li, Zhiheng Fu, Haokun Wen, and Weili Guan. HUD: Hierarchical uncertainty-aware disambiguation network for composed video retrieval. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 6143–6152, 2025. 3

[6] Adnan El Assadi, Niklas Muennighoff, and Jinhyuk Lee. The embedder’s dilemma: LLMs are better, but at what cost? In Third Conference on Language Modeling, 2026. 3

[7] Animesh Gupta, Jay Parmar, Ishan Rajendrakumar Dave, and Mubarak Shah. From play to replay: Composed video retrieval for temporally fine-grained videos. In Advances in Neural Information Processing Systems: Datasets and Benchmarks Track, 2025. 3

[8] Rohit Gupta, Jayakrishnan Unnikrishnan, Fan Fei, Sheng Liu, Son Tran, and Mubarak Shah. ViLL-E: Video LLM embeddings for retrieval. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 43239–43258, San Diego, California, United States, 2026. Association for Computational Linguistics. Outstanding Paper. 3

[9] Shaunak Halbe, Bhagyashree Puranik, Jayakrishnan Unnikrishnan, Kushan Thakkar, Vimal Bhat, and Toufiq Parag. VeRVE: Versatile retrieval for videos via unified embeddings, 2026. arXiv preprint arXiv:2601.12193, version 3. 3

[10] Gyuwon Han, Young Kyun Jang, and Chanho Eom. CoVA: Text-guided composed video retrieval for audio-visual content. In 2026 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 12162–12166, 2026. 3

[11] Yupeng Hu, Zixu Li, Zhiwei Chen, Qinlei Huang, Zhiheng Fu, Mingzhu Xu, and Liqiang Nie. REFINE: Composed video retrieval via shared and differential semantics enhancement. ACM Transactions on Multimedia Computing, Communications, and Applications, 22(7):1–24, 2026. 3

[12] Jiale Huang, Zixu Li, Zhiwei Chen, Zhiheng Fu, Chunxiao Wang, and Yupeng Hu. IMAGINE: Adaptive schema-imagery enhanced composition for composed video retrieval. In Proceedings of the 2026 International Conference on Multimedia Retrieval, pages 288–297, 2026. 3

[13] Sihong Huang, Jiaxin Wu, Dongmei Jiang, Yi Cai, Yaowei Wang, and Xiaoyong Wei. Compositional transformation reasoning for composed video retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25644–25653, 2026. 2, 3, 7, 14

[14] Thomas Hummel, Shyamgopal Karthik, Mariana-Iuliana Georgescu, and Zeynep Akata. EgoCVR: An egocentric benchmark for fine-grained composed video retrieval. In Computer Vision – ECCV 2024, pages 1–17, 2024. 2, 3, 7

[15] Chuong Huynh, Jinyu Yang, Ashish Tawari, Mubarak Shah, Son Tran, Raffay Hamid, Trishul Chilimbi, and Abhinav Shrivastava. CoLLM: A large language model for composed image retrieval. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3994– 4004, 2025. 1

[16] Boseung Jeong, Taegyu Park, Donghyeon Kwon, Hyunsouk Cho, and Suha Kwak. Learning sample-wise rank-aware interpolation weights for composed visual data retrieval. In European Conference on Computer Vision, 2026. 3

[17] Junyang Ji, Shengjun Zhang, Da Li, Yuxiao Luo, Yan Wang, Di Xu, Biao Yang, Wei Yuan, Fan Yang, Zhihai He, and Wenming Yang. OmniCVR: A benchmark for omni-composed video retrieval with vision, audio, and text. In International Conference on Learning Representations, 2026. 3

[18] Fanheng Kong, Jingyuan Zhang, Yahui Liu, Hongzhi Zhang, Shi Feng, Xiaocui Yang, Daling Wang, Yu Tian, Victoria W., Fuzheng Zhang, and Guorui Zhou. Modality curation: Building universal embeddings for advanced multimodal information retrieval, 2025. arXiv preprint arXiv:2505.19650; no independently verified peer-reviewed acceptance at bibliography preparation time. 3, 14

[19] Mohammed Irfan Kurpath, Jaseel Muhammad Kaithakkodan, Jinxing Zhou, Sahal Shaji Mullappilly, Mohammad Almansoori, Noor Ahsan, Beknur Kalmakhanbet, Sambal Shikhar, Rishabh Lalla, Jean Lahoud, Mariette Awad, Fahad Shahbaz Khan, Salman Khan, Rao Muhammad Anwer, and Hisham Cholakkal. A benchmark for omni-modal reasoning in long videos, 2026. arXiv preprint arXiv:2512.16978. 3

[20] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A unified framework for state-of-the-art multimodal retrieval and ranking, 2026. arXiv preprint arXiv:2601.04720. 2, 3

[21] Wenyan Li, Raphael Tang, Chengzu Li, Caiqi Zhang, Ivan Vulic, and Anders Søgaard. Lost in embeddings: Information´ loss in vision–language models. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 22676–22693, Suzhou, China, 2025. Association for Computational Linguistics. 2

[22] Zixu Li, Yupeng Hu, Zhiwei Chen, Qinlei Huang, Guozhi Qiu, Zhiheng Fu, and Meng Liu. ReTrack: Evidence-driven dualstream directional anchor calibration network for composed

video retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 23373–23381, 2026. 3

[23] Zixu Li, Yupeng Hu, Zhiheng Fu, Zhiwei Chen, Weili Guan, and Liqiang Nie. R<sup>3</sup>: Composed video retrieval via reasoningguided recalling and re-ranking, 2026. arXiv preprint and CoVR-R challenge report; peer review not established. 2, 3

[24] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. MM-Embed: Universal multimodal retrieval with multimodal LLMs. In International Conference on Learning Representations, 2025. 2, 3

[25] DongQing Liu, MengShi Qi, and HongWei Ji. Reason-thenretrieve for CoVR-R with structured edit prompts and densesparse fusion, 2026. arXiv preprint and CoVR-R challenge report; peer review not established. 3

[26] Yunze Liu, Chi-Hao Wu, Enmin Zhou, and Junxiao Shen. OmniRetriever: Any-to-any audio-video-text retrieval via fusion-as-teacher distillation, 2026. arXiv preprint arXiv:2605.26641. 3

[27] Yang Liu, Qianqian Xu, Peisong Wen, Siran Dai, and Qingming Huang. Training-free composed video retrieval via visual representation-guided video-LLM reasoning, 2026. arXiv preprint and CoVR-R challenge report; arXiv comments mention the CVPR 2026 VidLLMs workshop, but no archival proceedings record was available at bibliography preparation time. 3

[28] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. Image retrieval on real-life images with pretrained vision-and-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2125–2134, 2021. 1, 2

[29] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Raghuveer Thirukovalluru, Xuan Zhang, Zeyuan Chen, Ran Xu, Caiming Xiong, Yingbo Zhou, Wenhu Chen, and Semih Yavuz. VLM2Vec-V2: Advancing multimodal embedding for videos, images, and visual documents. Transactions on Machine Learning Research, 2026. 3

[30] Antoine Miech, Jean-Baptiste Alayrac, Ivan Laptev, Josef Sivic, and Andrew Zisserman. Thinking fast and slow: Efficient text-to-visual retrieval with transformers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9826–9836, 2021. 3

[31] Shraman Pramanick, Yale Song, Sayan Nag, Kevin Qinghong Lin, Hardik Shah, Mike Zheng Shou, Rama Chellappa, and Pengchuan Zhang. EgoVLPv2: Egocentric video-language pre-training with fusion in the backbone. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5285–5297, 2023. 7

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 2, 7

[33] Yuyang Sun, Yongliang Wu, Xingyu Zhu, Yuxia Chen, Zhenxiang Jiang, Yangguang Ji, Wenbo Zhu, Yanxi Shi, Jay Wu,

Shuo Wang, and Xu Yang. Dual-route top-k retrieval with 1v1 VLM reranking for the CoVR-R, 2026. arXiv preprint and CoVR-R challenge technical report; peer review not established. 3

[34] Omkar Thawakar, Muzammal Naseer, Rao Muhammad Anwer, Salman Khan, Michael Felsberg, Mubarak Shah, and Fahad Shahbaz Khan. Composed video retrieval via enriched context and discriminative embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26896–26906, 2024. 2, 3, 7, 14

[35] Omkar Thawakar, Dmitry Demidov, Ritesh Thawkar, Rao Muhammad Anwer, Mubarak Shah, Fahad Shahbaz Khan, and Salman Khan. Beyond simple edits: Composed video retrieval with dense modifications. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20435–20444, 2025. 2, 3, 6, 7, 14, 15

[36] Omkar Thawakar, Dmitry Demidov, Vaishnav Potlapalli, Sai Prasanna Teja Reddy Bogireddy, Viswanatha Reddy Gajjala, Alaa Mostafa Lasheen, Rao Muhammad Anwer, and Fahad Khan. CoVR-R: Reason-aware composed video retrieval. In British Machine Vision Conference, 2026. Accepted; proceedings forthcoming. 2, 3, 6, 7, 14, 15

[37] Rong-Cheng Tu, Zhao Jin, Jingyi Liao, Xiao Luo, Yingjie Wang, Li Shen, and Dacheng Tao. MLLM-guided VLM finetuning with joint inference for zero-shot composed image retrieval, 2025. arXiv preprint arXiv:2505.19707. 1

[38] Lucas Ventura, Antoine Yang, Cordelia Schmid, and Gul¨ Varol. CoVR: Learning composed video retrieval from web video captions. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5270–5279, 2024. 2, 3, 6, 7, 14

[39] Lucas Ventura, Antoine Yang, Cordelia Schmid, and Gul¨ Varol. CoVR-2: Automatic data construction for composed video retrieval. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):11409–11421, 2024. 2, 3, 6, 7, 14, 15

[40] Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. Composing text and image for image retrieval—an empirical odyssey. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6439–6448, 2019. 1, 2

[41] Hao Wang, Fang Liu, Licheng Jiao, Jiahao Wang, Shuo Li, Lingling Li, Puhua Chen, and Xu Liu. Vision-by-prompt: Context-aware dual prompts for composed video retrieval. Pattern Recognition, 172:112378, 2026. 3

[42] Haokun Wen, Xuemeng Song, Haoyu Zhang, Weili Guan, Xiangyu Zhao, and Liqiang Nie. UniCVR: From alignment to reranking for unified zero-shot composed visual retrieval, 2026. arXiv preprint arXiv:2604.20318. 3, 14

[43] Yue Wu, Zhaobo Qi, Yiling Wu, Junshu Sun, Yaowei Wang, and Shuhui Wang. Learning fine-grained representations through textual token disentanglement in composed video retrieval. In International Conference on Learning Representations, 2025. 3, 14

[44] Chenwei Xu, Jianshu Zhang, Shang Wu, Lie Lu, Pranav Maneriker, Fan Du, Manling Li, and Han Liu. VideoCritic: Diagnosing and localizing reasoning errors in video-language models. In CVPR Workshop on Video Large Language Mod-

els, 2026. Workshop publication; public author record and OpenReview PDF available. 3

[45] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. VideoCLIP: Contrastive pretraining for zero-shot video-text understanding. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 6787–6800. Association for Computational Linguistics, 2021. 2

[46] Bingqing Zhang, Yi Zhang, Zhuo Cao, Yang Li, Xue Li, Jiajun Liu, and Sen Wang. ReCoVR: Closing the loop in interactive composed video retrieval, 2026. arXiv preprint arXiv:2605.09836. 3

[47] Shiqi Zhang, Zhiwei Chen, Zixu Li, Zhiheng Fu, Wenbo Wang, Jiajia Nie, Yinwei Wei, and Yupeng Hu. RELATE: Enhance composed video retrieval via minimal-redundancy hierarchical collaboration. In 2026 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 12132–12136, 2026. 3

[48] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Bridging modalities: Improving universal multimodal retrieval by multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9274–9285, 2025. 2, 3

[49] Yuqian Zheng and Mariana-Iuliana Georgescu. X-Aligner: Composed visual retrieval without the bells and whistles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 6073–6082, 2026. 3

[50] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, Hongfa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, Wancai Zhang, Zhifeng Li, Wei Liu, and Li Yuan. Language-Bind: Extending video-language pretraining to n-modality by language-based semantic alignment. In International Conference on Learning Representations, 2024. 7

# Beyond Similarity: Foundation Models as an Efficient Backbone for Training-Free Composed Video Retrieval

Supplementary Material

A. Additional Method Details

## A.1. Pipeline Overview

This section gives the executable order, equality cases, and fallback invariants behind the conceptual cascade in the main paper. Let $e _ { 1 } \geq e _ { 2 }$ be the two leading cosine similarities after source masking and $r _ { 1 } \ \geq \ r _ { 2 }$ the leading candidaterelevance scores. All comparisons use unrounded values.

Reusable gallery construction. After applying any positive query limit, the evaluator reconstructs the target-only gallery from the retained examples. It validates the media, removes duplicate targets by stable identifier, resolves query-independent target selections, and embeds each distinct target exactly once. Each normalized video-only vector is stored in a gallery matrix. The benchmark reconstructs this matrix within an invocation; a deployment may persist the same target selections and vectors. Candidate-frame preparation is also query-independent and reusable, although every relevance call between a query and candidate remains online.

Online execution. Frame indices are resolved before model inference. A terse edit can activate target-description generation while the composed query and gallery are embedded. After source self-masking, the initial five gallery items form the candidate seed. A sufficiently separated embedding leader returns the coarse ranking directly; otherwise, provisional relevance scoring controls candidate expansion and the deferred description route. At most one generated description is reused across all candidates, and unchanged pair scores need not be recomputed in the final reranking pass. The scored head replaces the corresponding portion of the embedding order while the untouched tail remains complete. Finally, only a leading near-tie cluster can reach strict verification.

Skipping, failure, or unresolved output at any conditional stage retains the preceding valid order. Independent items and queries are processed concurrently. Within a query, expansion pages remain sequential because each leading score determines whether another page is needed; verifier calls are likewise sequential because the first accepted candidate terminates the scan.

## A.2. Stage-Aware Frame Allocation

Each downstream video input receives at most $B \ = \ 1 5$ selected frames; short clips and failed decodes can yield fewer, and verification contains separate source and candidate inputs. Global embedding uses equidistant samples. Reranking, decomposition, and verification instead use duration-adaptive novelty allocation computed from queryindependent, low-resolution frame probes. The probe count may exceed the downstream budget and is accounted for separately from the selected frames consumed by a later-stage model.

For duration $T ,$ short [0, 10), medium [10, 30), and long [30, ∞) clips are probed at 8, 4, and 2 frames per second:

$$
\begin{array} { r l } & { \rho ( T ) = \{ \begin{array} { l l } { 8 , } & { 0 \leq T < 1 0 , } \\ { 4 , } & { 1 0 \leq T < 2 0 , } \\ { 2 , } & { T > 2 3 0 , } \\ { 1 , } & { 0 \leq m \leq ( 1 , \operatorname* { m a n h } ( \mathscr { T } P ( T ) ) ) , } \end{array}  } \\ & { \mathcal { M } = \operatorname* { m a x } ( 1 , \operatorname* { m i n } ( F , M _ { \operatorname* { m a x } } ) ) , } \\ & { \begin{array} { r l } { 1 \leq m \leq ( 0 , 1 - \langle h _ { i } , h _ { i - 1 } \rangle ) , } & { 2 \leq i \leq M , } \\ { \delta _ { i } = 1 , M > 1 , } & { i = 1 , M > 1 , } \end{array} } \\ & { \begin{array} { r l } { \delta _ { i } = \{ \begin{array} { l l } { \delta _ { i } , } & { \sum _ { j } \delta _ { j } > 0 , } \\ { 1 / M , } & { \sum _ { j } \delta _ { j } = 0 , } \\ { 1 / M , } & { \sum _ { i } \delta _ { j } = 0 , } \\ { w _ { i } = 3 p _ { i } + ( 1 - \lambda ) / M , } & { \lambda = 1 . } \end{array}  } \end{array} } \\ & { \begin{array} { r l } & {  w _ { i } = \{ \begin{array} { l l } { \delta _ { i } / \sum _ { j } \delta _ { j } , \ > 0 , } & { 0 } \\ { 1 / M , } & { \lambda = \frac { 1 } { \sqrt { 2 } } \delta _ { j } , \ \sum _ { j } \delta _ { j } = 0 , } \\ { 1 / M , } & { \lambda = 1 . } \end{array}  } \end{array} } \\ & { \begin{array} { r l } { w _ { i } = \{ ( 1 - \lambda ) / M ,  } & { \lambda = 1 . } \end{array} } \end{array}\tag{S1}
$$

In Equation S1, F is the original-frame count, $\mathcal { P }$ is the set of M decoded probes, and $h _ { i }$ is the normalized embedding of probe i. The first probe inherits the second probe’s novelty when multiple probes exist. If $M \leq B ,$ all probes are retained. Otherwise, quantiles $( k + 0 . 5 ) / B$ for $k = 0 , \ldots , B - 1$ , are mapped through the inverse CDF of w; duplicate selections are removed, the highest-density unused probes backfill the budget, and the result is sorted in time. Zero novelty triggers the uniform fallback. This distributes frames over novelty mass rather than selecting only the B largest changes.

The benchmark presamples indices over the active source and gallery set. Configurations sharing the same clip and selection are deduplicated, target selections can be cached with the gallery, and a new source selection is online preparation. Every downstream request uses the selected originalframe indices with server-side resampling disabled. Final decoding, resizing, timestamp rendering where applicable, and JPEG packing remain lazy. The spatial settings are aspect-preserving long-side caps: embedding uses uniform 512-pixel frames, novelty probing uses 256-pixel frames, reranking and decomposition use selected 256-pixel frames, and verification uses selected 512-pixel frames.

Table S1. Evaluation datasets and protocol. Every row uses the complete named public split, a target-video gallery, and source self-masking when the query source appears as a non-target item.
<table><tr><td>Dataset</td><td>Evaluation split</td><td>Gallery</td><td>Source self-mask</td></tr><tr><td>WebVid-CoVR</td><td>Complete public test set</td><td>Target videos</td><td>Yes</td></tr><tr><td>Dense-WebVid-CoVR</td><td>Complete public test set</td><td>Target videos</td><td>Yes</td></tr><tr><td>CoVR-R</td><td>Official public split</td><td>Target videos</td><td>Yes</td></tr></table>

## A.3. Composed Query Embedding and Coarse Retrieval

The composed-query encoder receives the source video and unchanged raw edit under a target-oriented retrieval instruction. The gallery encoder receives only a target video under a separate, explicitly empty instruction; the empty role is transmitted rather than omitted. Both outputs are L2-normalized, so their dot product is cosine similarity. The source side uses up to 15 equidistant frames at a 512-pixel long-side cap, and each target is encoded once under the same visual policy. If the source video occurs in the gallery as a non-target item, it is masked before sorting. Strict score ties therefore retain the minimum rank convention defined in Section B.1.1.

The five highest-scoring valid items seed later stages, while the complete order is retained. The compact embedding model is deliberately assigned the gallery-wide role: it must summarize the broad intent expressed by the source and edit and preserve candidate recall rather than make the final fine-grained decision. Its one-vector target representation can be stored and searched with exact or approximate nearestneighbor machinery, allowing more specialized models to operate only on a routed candidate subset. The exact role instructions and model identity are reported in Section C.

## A.4. Confidence-Gated Adaptive Reranking

The candidate scorer receives the raw edit or one decomposed target description together with a candidate video; the source video is absent. The selected reranker exposes a continuous relevance score for its hosted binary-relevance task. The client consumes that score directly and does not reconstruct token probabilities.

Reranking is bypassed only when $e _ { 1 } { - e _ { 2 } } > 0 . 2 5 $ ; equality at 0.25 enters the stage. Otherwise, the initial five candidates are scored. While the best relevance score is strictly below 0.70, the next five items in embedding order are appended and scored. Equality at 0.70 stops expansion. At most nine additional pages are admitted, giving a maximum scored pool of $5 + 9 \times 5 = 5 0$ items, or the gallery size if smaller.

Scored candidates are sorted in decreasing score with stable tie handling. The unscored tail retains its embeddingorder position, and at most the reranked top ten are exposed to verification. A provisional pass supplies the confidence signal for deferred decomposition; the final pass reuses every candidate score whose text and video inputs are unchanged.

Pages are sequential within a query, while different queries can be processed concurrently. A fired embedding-gap gate, missing scores, or query-level reranking failure preserves the complete embedding order and cannot activate downstream score-dependent routing.

## A.5. Conditional Query Decomposition

The decomposer receives the source video and raw edit and requests one or two target-oriented sentences. Its candidateindependent output is generated at most once for each unique pairing of source and edit and reused across candidates and expansion rounds. In the final configuration it replaces only the reranker’s text; embedding and verification continue to use the raw modification.

The early route counts ASCII letter, digit, or apostrophe sequences as words and activates when the count is strictly below 10; an edit of exactly 10 words remains raw. If no usable early description exists, the late route activates only after a valid provisional reranking pass with $r _ { 1 } \ge 0 . 2 5$ and $( r _ { 1 } - r _ { 2 } ) / r _ { 1 } < 0 . 5 0$ . The score floor is inclusive and the relative-margin test is strict. When a second score is absent it is treated as zero; a missing or nonpositive leading score cannot satisfy the gate. The provisional pass includes any adaptive expansion, so its final observed leaders define the decision. An embedding-gap bypass or failed reranking provides no late-route evidence, while an unsuccessful early generation may be reconsidered by the late rule.

Frame selection completes before early generation begins. Early requests can run in a background worker while query and then gallery embeddings are computed, with parallelism within each block; generation joins before reranking. Empty or failed output falls back to the raw edit. The parser, response budget, cache boundary, and failed-item recovery policy appear in Section C and Section C.1.

## A.6. Confidence-Gated Candidate Verification

The verifier receives the source video, raw edit, and one candidate video; a decomposed description is not reused. Its strict prompt requires the candidate to satisfy every modified condition relative to the source. Missing, contradicted, or uncertain conditions yield a negative verdict, whereas unconstrained attributes need not match.

Only the reranked top ten are considered. Verification requires $r _ { 1 } \ge 0 . 1 5$ and a contiguous leading cluster of at least two candidates satisfying $( r _ { 1 } - r _ { i } ) / r _ { 1 } < 0 . 2 5$ . The score floor is inclusive, while equality at the relative gap excludes a candidate and terminates the prefix. Missing or nonpositive leading scores, a one-item cluster, or a score below the floor bypass the stage.

Table S2. Task-trained or adapted approaches. These methods use CoVR-specific training, learned aggregation, or adaptation and therefore provide a contextual rather than resource-equivalent comparison; CA denotes learned cross-attention fusion. Metric cells retain the precision printed by each source; Avg. is computed from the displayed cutoffs and rounded to two decimals, with missing cutoffs not imputed. Within each dataset, boldface and underlining mark the best and second-best available values. marks WebVid-CoVR, which includes unverified machine-generated modification queries, often short or too generic [35, 36, 39]. <sup>†</sup> denotes a method without independently verified peer-reviewed acceptance at the time of verification. “Dense-CoVR” in MoRe refers to BSE-CoVR; we retain the original method name.
<table><tr><td>Dataset</td><td>Method</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>CoVR-R</td><td>BSE-CoVR [35, 36]</td><td>37.90</td><td>57.67</td><td>64.48</td><td>79.47</td><td>59.88</td></tr><tr><td rowspan="4">Dense-WebVid-CoVR</td><td>CoVR-BLIP (CA) [36, 38]</td><td>35.60</td><td>60.80</td><td>70.31</td><td>87.05</td><td>63.44</td></tr><tr><td>ECDE (CA) [34, 36]</td><td>39.20</td><td>64.40</td><td>75.56</td><td>88.90</td><td>67.02</td></tr><tr><td>BSE-CoVR (CA) [13, 35, 36]</td><td>48.08</td><td>73.36</td><td>81.06</td><td>93.78</td><td>74.07</td></tr><tr><td>BSE-CoVR (Dense-trained) [35]</td><td>71.26</td><td>89.12</td><td>94.56</td><td>98.88</td><td>88.46</td></tr><tr><td rowspan="6">WebVid-CoVR*</td><td>FDCA-BLIP (FineCVR pretrained) [43]</td><td>52.23</td><td>79.42</td><td>86.66</td><td>96.91</td><td>78.81</td></tr><tr><td>CoVR-BLIP (WebVid adapted) [34, 38]</td><td>53.13</td><td>79.93</td><td>86.85</td><td>97.69</td><td>79.40</td></tr><tr><td>FDCA-BLIP (WebVid adapted) [43]</td><td>54.80</td><td>82.27</td><td>89.84</td><td>97.70</td><td>81.15</td></tr><tr><td>CoVR-BLIP-2 [39]</td><td>59.82</td><td>83.84</td><td>91.28</td><td>98.24</td><td>83.30</td></tr><tr><td>ECDE (WebVid adapted) [34]</td><td>60.12</td><td>84.32</td><td>91.27</td><td>98.72</td><td>83.61</td></tr><tr><td>UNITEinstruct 7B† [18]</td><td>72.50</td><td>90.80</td><td>95.30</td><td>99.50</td><td>89.53</td></tr><tr><td></td><td>UniCVR Stage II† [42]</td><td>66.77</td><td>86.31</td><td>91.88</td><td>98.29</td><td>85.81</td></tr></table>

Eligible candidates are scanned in rank order. The first strict positive verdict is stably promoted to the leading position and terminates the scan; all other relative positions are preserved. An all-negative scan, an unparseable response, exhaustion of retries, or a bypass leaves the reranked order unchanged. The generative request, parser, reasoning budget, sampling defaults, and recovery policy are specified in Section C.

## A.7. Timestamp-Aware Conditioning

Only decomposition and verification receive visible timestamps. After selection, decoding, and resizing, the client computes elapsed seconds as the original frame index divided by the original frame rate. It renders the value to two decimal places in the top-left corner using white glyphs with a black outline, before JPEG encoding. Embedding, reranking, and novelty probes receive no overlay.

Both generative prompts append the note: “A timestamp is drawn in each frame for reference only and is not part of the video. Videos do not have to be of the same length.” The overlay and prompt note are treated as one timestamp-aware package. Accordingly, the measured ablation is attributed to the package rather than to either rendering or prompt wording in isolation.

## B. Additional Experiments and Analysis

## B.1. Experimental Setup

## B.1.1. Datasets and Evaluation Protocol

WebVid-CoVR contains short automatically derived edits, Dense-WebVid-CoVR supplies denser attribute, action, and temporal descriptions, and CoVR-R combines WebVid with Something-Something-V2 reasoning examples. Main comparisons use each complete named public evaluation. Controlled studies retain their source identity: the Dense-WebVid-CoVR subset remains a dense-description study, whereas the CoVR-R controlled cohort (used for ablations and analysis) is identified simply as its WebVid subset and contains no Something-Something-V2 examples. Table S1 summarizes the datasets and shared protocol.

Our evaluations use a target-video gallery and mask the query source when it reappears as a non-target item. Prior values in the comparison tables follow each cited source’s reported benchmark protocol, which is not always explicit about source self-masking. R@K is the fraction of targets whose score rank is at most K; exact score ties receive the optimistic minimum rank. MeanR3 averages R@1, R@5, and R@10, while the Avg. column in Table S5 also includes R@50. MRR is reported on the unit interval, and stage-prefixed metrics are diagnostic snapshots rather than substitutes for the final ranking. Our averages use unrounded values; prior averages are source-reported or derived from their published recalls.

Table S3. Cumulative full-evaluation performance after composed embedding, adaptive reranking with conditional decomposition, and selective verification. Deltas compare adjacent stages within the same final run and are computed before display rounding.
<table><tr><td>Dataset</td><td>Cumulative stage</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>Avg.</td><td>∆R@1</td></tr><tr><td rowspan="3">WebVid-CoVR</td><td>Embedding</td><td>30.63</td><td>57.24</td><td>67.53</td><td>86.38</td><td>60.45</td><td></td></tr><tr><td>+ Rerank / decompose</td><td>53.25</td><td>73.12</td><td>79.26</td><td>86.38</td><td>73.00</td><td>+22.61</td></tr><tr><td>+ Verify</td><td>54.58</td><td>73.12</td><td>79.26</td><td>86.38</td><td>73.34</td><td>+1.33</td></tr><tr><td rowspan="3">Dense-WebVid-CoVR</td><td>Embedding</td><td>59.65</td><td>84.19</td><td>90.49</td><td>97.26</td><td>82.90</td><td></td></tr><tr><td>+ Rerank / decompose</td><td>88.57</td><td>95.15</td><td>96.40</td><td>97.26</td><td>94.34</td><td>+28.92</td></tr><tr><td>+ Verify</td><td>89.55</td><td>95.15</td><td>96.40</td><td>97.26</td><td>94.59</td><td>+0.98</td></tr><tr><td rowspan="3">CoVR-R</td><td>Embedding</td><td>56.64</td><td>79.08</td><td>84.55</td><td>94.95</td><td>78.81</td><td></td></tr><tr><td>+ Rerank / decompose</td><td>93.32</td><td>94.27</td><td>94.61</td><td>94.95</td><td>94.29</td><td>+36.67</td></tr><tr><td>+ Verify</td><td>93.43</td><td>94.27</td><td>94.61</td><td>94.95</td><td>94.31</td><td>+0.11</td></tr></table>

## B.1.2. Implementation Details

No benchmark example updates model parameters. Qwen3- VL-Embedding-2B jointly encodes source video and edit while keeping target encodings video-only. Qwen3-VL-Reranker-8B scores a small initial pool and grows it in fixed increments while confidence remains low; a large embedding margin bypasses reranking. Qwen3.5-9B supplies conditional target-description generation and strict verification. Description generation is routed by edit concision or reranker ambiguity, while verification scans only a leading relativescore cluster and accepts the first strict match.

Embedding uses a uniform reusable view. Reranking and decomposition use novelty-weighted 256-pixel frames; verification uses separate source and candidate views at 512 pixels. A query-independent 256-pixel probe selects laterstage frames at duration-dependent rates, and decomposition and verification receive the timestamp overlay together with its explanatory prompt note. Tables S15 and S16 in Section C consolidate model identities, exact thresholds, frame limits, generation settings, parsers, retries, and reuse behavior.

Training-free methods form the primary comparison in Table 1; task-trained or adapted systems are separated in Table S2 because they are contextual rather than resourceequivalent. Prior results are source-reported on the named benchmarks, whereas our rows use the target-gallery and source-self-mask protocol described above. App. B.5 profiles the selected configuration on size-matched 1,000-query, 1,000-target cohorts with fixed profiling seeds and reports both serving-dependent time and model-call workload (not a cross-system latency comparison).

## B.2. Quantitative Results

## B.2.1. Cross-Dataset Performance

Table S5 shows that the 89.55/95.15/96.40/97.26 Dense-WebVid-CoVR recall profile yields 94.59 Avg.; the largest improvement is at the head of the ranking, while R@50 is already close to saturation. On CoVR-R, the

93.43/94.27/94.61/94.95 profile exceeds the other directly compared training-free values at every cutoff, with the largest margin again at R@1. The result improves both early discrimination and coverage without repeating gallery-wide reasoning.

WebVid-CoVR’s modification candidates were machinegenerated and its test examples were manually selected or filtered, but many retained edits remain short and underspecified [36, 39]. This annotation style may contribute to the remaining gap: literal similarity can exploit sparse surface cues, whereas the cascade can overinterpret an ambiguous request. Our 73.34 Avg. is less than one point below the best complete training-free average; at R@1, the sourcereported MoRe row is higher under its reported benchmark protocol. The comparison reverses decisively on Dense-WebVid-CoVR, whose test modifications are fully manually verified and corrected [35]: CoVRAGE exceeds MoRe by 39.95, 28.15, and 18.50 points at R@1, R@5, and R@10, respectively. This higher-quality, more specific annotation regime better reflects detailed user intent, although aggregate results alone cannot identify why MoRe’s relative standing reverses across the two benchmarks. Table S2 provides the separate task-trained or adapted comparison; on the denser benchmarks, the frozen cascade compares favorably with the displayed BSE-CoVR rows.

## B.2.2. Where the Gains Arise

Table S3 shows a stable division of labor. Candidate reranking and routed decomposition produce the dominant R@1 gains, while verification adds 1.33, 0.98, and 0.11 points on WebVid-CoVR, Dense-WebVid-CoVR, and CoVR-R. Its smaller contribution is expected because it only resolves the leading ambiguity cluster. R@50 is unchanged after refinement because later stages reorder a bounded head and preserve the embedding tail; the reusable index therefore supplies coverage while specialists improve local ordering.

Table S4. Extended embedding ablations on their stated datasets and subsets, covering query composition, weighted fusion, embedding capacity, and role-specific query/target instructions. Controlled rows are identified textually as the Dense-WebVid-CoVR subset or WebVid subset of CoVR-R.
<table><tr><td>Configuration</td><td>Dataset</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>M3</td><td>M4</td><td>MRR</td></tr><tr><td colspan="9">Query composition</td></tr><tr><td>Text only</td><td>Dense-WebVid-CoVR</td><td>36.60</td><td>54.60</td><td>62.60</td><td>75.60</td><td>51.27</td><td>57.35</td><td>0.4547</td></tr><tr><td>Source video only</td><td>Dense-WebVid-CoVR</td><td>52.70</td><td>77.00</td><td>83.70</td><td>93.70</td><td>71.13</td><td>76.78</td><td>0.6327</td></tr><tr><td>Weighted sum (text weight 0.25)</td><td>Dense-WebVid-CoVR</td><td>57.60</td><td>77.40</td><td>83.50</td><td>93.20</td><td>72.83</td><td>77.92</td><td>0.6678</td></tr><tr><td>Weighted sum (text weight 0.50)</td><td>Dense-WebVid-CoVR</td><td>63.00</td><td>84.80</td><td>90.10</td><td>96.90</td><td>79.30</td><td>83.70</td><td>0.7275</td></tr><tr><td>Weighted sum (text weight 0.75)</td><td>Dense-WebVid-CoVR</td><td>57.90</td><td>80.80</td><td>87.20</td><td>96.20</td><td>75.30</td><td>80.53</td><td>0.6833</td></tr><tr><td>Joint source-video + edit input</td><td>Dense-WebVid-CoVR</td><td>67.20</td><td>85.80</td><td>89.20</td><td>96.00</td><td>80.73</td><td>84.55</td><td>0.7526</td></tr><tr><td colspan="9">How does embedding-model capacity transfer across datasets?</td></tr><tr><td>Dense-WebVid-CoVR / 8B</td><td>Dense-WebVid-CoVR</td><td>67.20</td><td>85.80</td><td>89.20</td><td>96.00</td><td>80.73</td><td>84.55</td><td>0.7526</td></tr><tr><td>Dense-WebVid-CoVR / 2B</td><td>Dense-WebVid-CoVR</td><td>69.60</td><td>90.10</td><td>94.20</td><td>98.80</td><td>84.63</td><td>88.17</td><td>0.7842</td></tr><tr><td>WebVid-CoVR / 8B</td><td>WebVid-CoVR</td><td>44.20</td><td>68.60</td><td>76.50</td><td>89.00</td><td>63.10</td><td>69.58</td><td>0.5552</td></tr><tr><td>WebVid-CoVR / 2B</td><td>WebVid-CoVR</td><td>47.70</td><td>75.00</td><td>83.70</td><td>95.90</td><td>68.80</td><td>75.58</td><td>0.5998</td></tr><tr><td>CoVR-R WebVid-only subset / 8B</td><td>CoVR-R WebVid-only</td><td>73.80</td><td>89.80</td><td>93.60</td><td>98.20</td><td>85.73</td><td>88.85</td><td>0.8094</td></tr><tr><td>CoVR-R WebVid-only subset / 2B</td><td>CoVR-R WebVid-only</td><td>72.50</td><td>92.30</td><td>95.70</td><td>98.80</td><td>86.83</td><td>89.83</td><td>0.8076</td></tr><tr><td colspan="9">Do role-specific embedding instructions improve alignment?</td></tr><tr><td>Shared instruction</td><td>Dense-WebVid-CoVR</td><td>69.60</td><td>90.10</td><td>94.20</td><td>98.80</td><td>84.63</td><td>88.17</td><td>0.7842</td></tr><tr><td>Role-specific instructions</td><td>Dense-WebVid-CoVR</td><td>73.10</td><td>91.50</td><td>95.30</td><td>98.90</td><td>86.63</td><td>89.70</td><td>0.8123</td></tr></table>

## B.3. Qualitative Results

## B.3.1. Candidate Evolution Through the Cascade

Figure 1 in the main paper makes the stage roles visible without relying on trace identifiers. Coarse retrieval keeps several visually related water clips, reranking promotes both the target and a strong river distractor, and verification checks the remaining negative constraint. The wider scene is rejected because rocks and trees violate the request for no background objects; the close-up target satisfies the full edit. Figure 4 complements that progression with a temporal case: the candidate order remains unchanged after embedding and reranking, but the combined timestamp overlay and explanatory prompt let verification distinguish the complete train-window transformation from a passenger-only match. Together, the examples isolate broad recall, candidate ordering, and final constraint checking.

## B.3.2. Positive, Negative, and Ambiguous Retrievals

An easy green-network example follows the shallow path because coarse retrieval already identifies the correct visual transformation, reranking preserves it, and verification is bypassed. At the other extreme, the sparse edit “change to blue” leaves the factory-process target outside the bounded pool, as shown in Figure S1; no later stage can score what coarse retrieval never admits. The corresponding dense edit specifies blue ink, a spatula, the factory setting, and background machinery, placing the same target inside the cascade and showing how query specificity controls upstream coverage.

![](images/7fa89b3b8c0a914ad4493e4f51ca6fccc3ee0fa23293ad8642f3f1af5a6bed75.jpg)  
Figure S1. Sparse edits can fail before reasoning. The correct factory process (green, dashed) remains outside the candidate pool, while the solid branch retrieves an unrelated blue clip. Later stages cannot recover an unseen target.

Some negative samples, like a campfire failure, expose a different limitation. The selected alternative contains orangeyellow flames, logs, embers, and a dark background and plausibly satisfies the request, yet the single-positive protocol marks only a near-duplicate as correct. This should be treated as annotation ambiguity rather than an unequivocal semantic error. Strict verification is corrective on balance but can still reject a correct near-tied candidate.

Table S5. Complete final target-gallery results with all recall cutoffs, MeanR3, MeanR4, MRR, MedianRank, and usable-media coverage.
<table><tr><td>Dataset</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>M3</td><td>M4</td><td>MRR</td><td>MedR</td><td>Coverage</td></tr><tr><td>WebVid-CoVR</td><td>54.58</td><td>73.12</td><td>79.26</td><td>86.38</td><td>68.99</td><td>73.34</td><td>0.6278</td><td>1</td><td>100.00%</td></tr><tr><td>Dense-WebVid-CoVR</td><td>89.55</td><td>95.15</td><td>96.40</td><td>97.26</td><td>93.70</td><td>94.59</td><td>0.9211</td><td>1</td><td>100.00%</td></tr><tr><td>CoVR-R</td><td>93.43</td><td>94.27</td><td>94.61</td><td>94.95</td><td>94.10</td><td>94.31</td><td>0.9393</td><td>1</td><td>100.00%</td></tr></table>

## B.3.3. Relation to the Main Qualitative Results

Candidate expansion rescues a bathtub transformation that lies outside the small initial seed: one confidence-triggered enlargement admits the target and reranking promotes it. This is a coverage correction rather than a semantic rewrite, because the original query remains unchanged. A controlled green-line diagnostic addresses the complementary bottleneck: decomposition enriches “replace the white lines and dots with green” with the missing moving-network context. That diagnostic supplies the generated text to both retrieval and reranking, whereas the final system uses the more stable reranker-only placement. The two cases therefore separate uncertainty about candidate depth from uncertainty about what the edit describes.

## B.4. Ablation Studies

## B.4.1. Embedding Stage

Query Strategy Analysis. Table S4 shows that joint encoding of the source video and edit yields the best R@1 and MeanR3 in the complete fusion sweep, while equal-weight late fusion gives the best R@10. The joint encoder therefore favors the retrieval head without sacrificing a reusable video-only gallery.

Model Size. The compact embedder improves MeanR3 on the Dense-WebVid-CoVR, WebVid-CoVR, and WebVid subset of CoVR-R studies; only CoVR-R R@1 is slightly lower. More gallery-wide parameters are therefore not uniformly better.

Query/Target Decoupled Instructions. Role-specific instructions improve R@1 from 69.60 to 73.10 over a shared instruction. The query carries the transformation, while the target role remains a neutral video representation.

## B.4.2. Reranker Stage

Query Strategy Analysis. Table S6 shows that adding textconditioned candidate scoring raises Dense-WebVid-CoVR R@1 from 73.10 to 86.10. Supplying source video with the edit reaches 90.00, but the selected text interface reuses one raw or decomposed description across every candidate and expansion round.

Embedding-Based Reranker Gating. A large separation between the two leading embedding scores bypasses reranking for 7.50% of queries without changing any reported accuracy metric, avoiding redundant candidate work on clear cases.

Reranker Pool Size. A small fixed seed is sufficient only when paired with conditional growth. It keeps confident cases shallow while allowing uncertain cases to reach deeper candidates, rather than asserting that one pool size is optimal for every query.

Score-Based Candidate Pool Expansion. Confidencetriggered growth improves R@1 from 86.10 to 89.50 and MeanR3 from 90.97 to 94.97. The matched combined-query study shows the same direction, confirming that the gain is a pool-depth effect rather than an artifact of the text interface.

Reranker Model Size. On the WebVid subset of CoVR-R, the larger candidate scorer raises R@1 from 96.57 to 98.17 while reducing both expansion and verification routing. Here, added reranker capacity improves local ordering and reduces downstream ambiguity. Compact decomposition frames preserve accuracy, while compact reranker frames trade a small R@1 decrease for lower repeated visual cost.

## B.4.3. Decomposer Stage

Text Decomposition for Reranker. Table S7 shows that target-oriented decomposition raises Dense-WebVid-CoVR R@1 from 89.50 to 94.30 and produces an even larger gain on WebVid-CoVR. Supplying the generated description to embedding has mixed cross-dataset effects, so the final interface confines it to reranking.

Decomposition Gating. Always-on generation gives the highest controlled accuracy, concise-edit routing is inexpensive but misses many ambiguous cases, and ambiguity routing captures most of the benefit. The selected union of concise and ambiguous edits activates for 26.50% of queries and retains most of the always-on gain.

## B.4.4. Verifier Stage

Verifier Stage After Reranking. Ungated verification improves R@1 from 93.60 to 94.20, confirming that a final multimodal comparison can repair residual candidate-order errors.

Table S6. Extended reranker ablations with full retrieval metrics and workload fields. Rows isolate text versus combined query input, 2B versus 8B capacity, the strict embedding-difference bypass, score-gated candidate expansion, and later-stage resolution using matched target-only protocols.
<table><tr><td>Configuration</td><td>Dataset</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>M3</td><td>M4</td><td>MRR</td><td>Workload</td></tr><tr><td colspan="10">How much does candidate-conditioned reranking repair coarse retrieval?</td></tr><tr><td>Coarse embedding ranking</td><td>Dense-WebVid-CoVR</td><td>73.10</td><td>91.50</td><td>95.30</td><td>98.90</td><td>86.63</td><td>89.70</td><td>0.8123</td><td></td></tr><tr><td>+ 8B text reranker</td><td>Dense-WebVid-CoVR</td><td>86.10</td><td>91.50</td><td>95.30</td><td>98.90</td><td>90.97</td><td>92.95</td><td>0.8933</td><td></td></tr><tr><td colspan="10">Which query representation should the reranker score?</td></tr><tr><td>Edit text</td><td>Dense-WebVid-CoVR</td><td>86.10</td><td>91.50</td><td>95.30</td><td>98.90</td><td>90.97</td><td>92.95</td><td>0.8933</td><td></td></tr><tr><td>Source video + edit</td><td>Dense-WebVid-CoVR</td><td>90.00</td><td>91.50</td><td>95.30</td><td>98.90</td><td>92.27</td><td>93.92</td><td>0.9149</td><td></td></tr><tr><td colspan="10">Can high-margin candidates bypass redundant reranking?</td></tr><tr><td>Difference gate disabled</td><td>Dense-WebVid-CoVR</td><td>86.10</td><td>91.50</td><td>95.30</td><td>98.90</td><td>90.97</td><td>92.95</td><td>0.8933</td><td></td></tr><tr><td>Difference gate enabled</td><td>Dense-WebVid-CoVR</td><td>86.10</td><td>91.50</td><td>95.30</td><td>98.90</td><td>90.97</td><td>92.95</td><td>0.8933</td><td>skip 7.50%</td></tr><tr><td colspan="10">When should adaptive reranking expand the candidate pool?</td></tr><tr><td>Text query / fixed pool</td><td>Dense-WebVid-CoVR</td><td>86.10</td><td>91.50</td><td>95.30</td><td>98.90</td><td>90.97</td><td>92.95</td><td>0.8933</td><td>expand 0.00%, expanded 0, rescues 0, pages 0.00</td></tr><tr><td>Text query / adaptive expansion</td><td>Dense-WebVid-CoVR</td><td>89.50</td><td>97.20</td><td>98.20</td><td>98.90</td><td>94.97</td><td>95.95</td><td>0.9316</td><td>expand 53.10%, 57 rescues, mean depth 8.64</td></tr><tr><td>Combined query / fixed pool Combined query / adaptive expansion</td><td>Dense-WebVid-CoVR Dense-WebVid-CoVR</td><td>90.00 95.90</td><td>91.50 98.00</td><td>95.30 98.30</td><td>98.90 98.90</td><td>92.27 97.40</td><td>93.92 97.78</td><td>0.9149 0.9696</td><td>expand 0.00%, expanded 0, rescues 0, pages 0.00 expand 17.40%, 61 rescues, mean depth 6.85</td></tr><tr><td colspan="10">How does reranker capacity affect accuracy and routed workload?</td></tr><tr><td>2B reranker</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8B reranker</td><td>CoVR-R WebVid subset CoVR-R WebVid subset</td><td>96.57 98.17</td><td>98.83 98.83</td><td>98.98 98.90</td><td>99.12 99.12</td><td>98.13 98.64</td><td>98.37 98.76</td><td>0.9765 0.9849</td><td>expand 82.47%, verify 42.88%, 861 calls</td></tr><tr><td>Can later reasoning retain accuracy with compact frames?</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>expand 11.40%, verify 3.51%, 61 calls</td></tr><tr><td colspan="10">Decomposition frames: 720 px</td></tr><tr><td>Decomposition frames: 256 px</td><td>Dense-WebVid-CoVR Dense-WebVid-CoVR</td><td>94.30 94.30</td><td>98.40 98.50</td><td>98.60</td><td>98.90</td><td>97.10</td><td>97.55</td><td>0.9618</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>98.80</td><td>98.90</td><td>97.20</td><td>97.62</td><td>0.9627</td><td></td></tr></table>

Verifier Gating. The relative-score and score-floor gate reduces verdict work by more than 90% with only a small accuracy change. Restricting the scan to a leading ambiguity cluster is therefore preferable to verifying every candidate set.

Verifier Strict Prompt. Under the matched gate, strict fullcondition checking raises R@1 from 94.00 to 94.50. The result supports a strict, confidence-gated, top-down verifier under the fixed interface comprising the source, edit, and candidate.

## B.4.5. Frame Sampling

Dynamic Sampling. Table S8 reports that all-stage novelty-weighted allocation improves the matched Dense-WebVid-CoVR result from 94.40 to 95.00 R@1 and improves every reported recall cutoff. Figure 3 illustrates how the policy reallocates views. With later stages already dynamic, changing embedding from uniform to dynamic accounts for the remaining 0.30-point R@1 gain; the selected balanced profile keeps gallery-wide embedding uniform and applies dynamic views only to candidate and reasoning stages.

Hyperparameters for Dynamic Frame Sampling. Short, medium, and long clips are probed at 8, 4, and 2 frames per second, respectively, and pure novelty weighting performs best in the measured sweep. Probe work remains separate from the 15-frame model-input ceiling. Table S9 shows that the aggregate gain is not uniform: shorter groups benefit, whereas long clips expose heterogeneity and small-sample uncertainty in the long-duration tail.

Frame Timestamp Overlay. Figure S2 visualizes the elapsed-time overlay, which is evaluated together with its explanatory prompt note. Table S10 shows that enabling the package changes R@1 from 94.70 to 95.00 and MRR from 0.9652 to 0.9663, while R@50 is unchanged. The result supports the combined temporal cue and does not isolate overlay rendering from prompt wording.

Table S11 reports the cross-dataset frame-count and decomposition-placement extensions.

## B.5. Efficiency and Additional Analysis

We profile the selected configuration on size-matched 1,000- query, 1,000-target cohorts from WebVid-CoVR, Dense-WebVid-CoVR, and the WebVid subset of the official CoVR-R annotation. Each run uses a target-video gallery with source self-masking, 16 workers, and identical model, frame, and routing settings; the generative seeds are fixed to 42 only for this controlled profile. Persistent embedding, prediction, and frame-bundle caches are disabled, so earlier runs cannot reduce the measured cost, while reuse within each invocation is retained to reflect normal batched execution.

Table S12 isolates the online query path: decomposition, composed-query embedding, reranking, and verification. Gallery encoding and query-independent temporal probing are excluded from this boundary and reported separately in Table S13. Mean online busy time is 69.751, 38.630, and 17.842 seconds per query on WebVid-CoVR, Dense-WebVid-CoVR, and CoVR-R, respectively. The corresponding compute is 76.264, 50.514, and 23.227 model inferences per query. A gate skip counts as zero calls; the totals therefore use the 996, 261, and 111 descriptions actually generated in the three runs and exclude no-op stage entries.

The cost difference follows the edit distribution rather than a configuration change. WebVid-CoVR activates expansion, decomposition, and verification for 75.4%, 99.6%, and 16.1% of queries; the corresponding rates are 44.2%, 26.1%, and 6.5% on Dense-WebVid-CoVR and 17.4%, 11.1%, and 3.6% on CoVR-R. Thus concise, underspecified WebVid edits traverse the deepest path most often, whereas the more explicit CoVR-R edits usually terminate after a shallow candidate pass.

Table S13 shows that repeated candidate scoring dominates online work: the reranker averages 47.898, 31.848, and 13.741 busy seconds per query across the same three datasets. Query embedding remains nearly constant at 1.434–1.483 seconds and one call per query, while the conditional generative stages scale with their activation rates. The reusable selector performs about 33.9 low-resolution frame-probe inferences per clip–stage selection, for 134,240, 134,240, and 134,112 calls across the three cohorts; gallery encoding adds one call per target. Target-side selections and gallery vectors can be amortized across future queries, while a new source still requires its own query-independent selection.

The evaluation-routine wall times are 5,330.73, 3,233.29, and 2,012.57 seconds, while the online-model interval unions are 4,416.26, 2,488.34, and 1,204.25 seconds. These wall times reflect the measured serving environment, overlap among 16 workers, and any request retries; they are neither hardware-independent latency nor a cross-system speed comparison. Model-call counts provide the complementary interface-level workload measure, but should not be interpreted as FLOPs or monetary cost. Table S14 separately reports routing rates on the three complete evaluations.

The routing distribution depends strongly on edit style. WebVid-CoVR sends nearly all concise edits through decomposition, Dense-WebVid-CoVR relies primarily on ambiguity-based routing, and CoVR-R uses that route exclusively. Verification remains net positive on every benchmark, while the much lower activation on CoVR-R illustrates how a stronger candidate order narrows downstream work.

## C. Exact Model Interfaces and Reproducibility

The retrieval encoder is Qwen3-VL-Embedding-2B. Its final query form is source video plus raw edit under a targetoriented retrieval instruction; target videos use an explicit neutral instruction. The same embedding family supplies the language-free low-resolution temporal probes. Qwen3-

VL-Reranker-8B receives raw or decomposed text with each candidate video and returns a continuous relevance score. Decomposition and verification share Qwen3.5-9B but retain separate prompts, request layouts, output parsers, resolutions, and reusable prediction records.

Both generative roles use an enabled reasoning mode with a 2,048-token reasoning budget and a 2,560-token maximum response. With temperature unset, the effective defaults are temperature 0.6, top-p 0.95, and top-k 20; final accuracy evaluations are unseeded. General requests use a 300-second timeout, bounded transport retries with exponential backoff and jitter, and a shared connection pool. Temporal probes use a shorter timeout and a separate pool. After transport handling, unresolved decomposition or verification items receive bounded evaluator-level retries. The decomposition parser trims whitespace and one enclosing quote pair; the verifier normalizes leading formatting and accepts responses beginning with yes or no.

## C.1. Caching, Presampling, Concurrency, and Reproducibility

Four reuse families are independent. Embeddings use inmemory reuse plus atomic binary vector entries keyed by model, instruction, role, content, resolved frames, and pixel policy. Reranker, decomposer, and verifier predictions use structured records keyed by stage, model, prompt, query content, frame signature, resolution, timestamps, reasoning, and sampling. Temporal-probe vectors additionally key clip, probe model, frame rate, resolution, and selector version; selected-index records include the output budget, novelty mixture, duration bands, and rates. Final frame bundles use a bounded in-memory least-recently-used store and optional compressed structured storage keyed by clip, exact indices, pixel cap, timestamp state, image-encoding policy, and version.

Writes are atomic and malformed entries become misses. Normal accuracy evaluation permits cross-invocation reuse, whereas timed evaluations redirect or disable the relevant persistent stores while retaining safe reuse within the invocation. This prevents previous evaluations from shrinking the measured cold boundary without duplicating work inside one evaluation.

The final configuration resolves temporal probes and selected indices before inference but does not preload complete frame bundles. Final decoding, resizing, timestamp rendering, and image packing remain lazy. Candidate frames are reused across repeated candidate calls, but every uncached query-conditioned reranker or verifier pair remains online work. Independent items and queries run concurrently; expansion rounds and verifier scans remain sequential within a query, and early decomposition can overlap embedding before joining at reranking.

Table S7. Trade-offs between accuracy and workload for decomposition and verification on the Dense-WebVid-CoVR subset, comparing always-on, concise-edit, ambiguity-based, combined, lenient, and strict routing variants.
<table><tr><td>Configuration</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>M3</td><td>M4</td><td>MRR</td><td>Routed</td><td>Rescues</td><td>Calls</td><td>Saved</td></tr><tr><td>Decomposition</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Decomposition disabled</td><td>89.50</td><td>97.20</td><td>98.20</td><td>98.90</td><td>94.97</td><td>95.95</td><td>0.9316</td><td>0.00%</td><td></td><td></td><td></td></tr><tr><td>Decompose every query</td><td>94.30</td><td>98.50</td><td>98.80</td><td>98.90</td><td>97.20</td><td>97.62</td><td>0.9627</td><td>100.00%</td><td></td><td></td><td></td></tr><tr><td>Route concise edits</td><td>91.00</td><td>98.00</td><td>98.50</td><td>98.90</td><td>95.83</td><td>96.60</td><td>0.9424</td><td>4.50%</td><td></td><td></td><td></td></tr><tr><td>Route ambiguous rankings</td><td>92.90</td><td>97.80</td><td>98.40</td><td>98.90</td><td>96.37</td><td>97.00</td><td>0.9526</td><td>24.20%</td><td>47</td><td></td><td></td></tr><tr><td>Route concise or ambiguous queries</td><td>93.60</td><td>98.20</td><td>98.60</td><td>98.90</td><td>96.80</td><td>97.32</td><td>0.9579</td><td>26.50%</td><td>39</td><td></td><td></td></tr><tr><td>Verification</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Verification disabled</td><td>93.60</td><td>98.20</td><td>98.60</td><td>98.90</td><td>96.80</td><td>97.32</td><td>0.9579</td><td>0.00%</td><td></td><td></td><td></td></tr><tr><td>Verify every candidate set</td><td>94.20</td><td>98.30</td><td>98.60</td><td>98.90</td><td>97.03</td><td>97.50</td><td>0.9616</td><td>100.00%</td><td></td><td>1124</td><td></td></tr><tr><td>Confidence-gated verification</td><td>94.00</td><td>98.20</td><td>98.60</td><td>98.90</td><td>96.93</td><td>97.42</td><td>0.9599</td><td>7.80%</td><td></td><td>84</td><td>92.53%</td></tr><tr><td>Confidence-gated strict verification</td><td>94.50</td><td>98.20</td><td>98.60</td><td>98.90</td><td>97.10</td><td>97.55</td><td>0.9625</td><td>7.80%</td><td></td><td>104</td><td>90.75%</td></tr></table>

Table S8. Aggregate frame-policy ablations under a fixed 15-frame output budget per video input. The evaluated dynamic package improves the matched aggregate result at every reported recall cutoff; placement and novelty-mixing rows record the measured accuracy/compute choices. Probe work is accounted separately from the selected output.
<table><tr><td>Configuration</td><td>Dataset</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>M3</td><td>M4</td><td>MRR</td></tr><tr><td colspan="9">Does dynamic sampling improve aggregate retrieval?</td></tr><tr><td>Uniform sampling at every stage</td><td>Dense-WebVid-CoVR</td><td>94.40</td><td>98.20</td><td>98.60</td><td>98.90</td><td>97.07</td><td>97.52</td><td>0.9619</td></tr><tr><td>Dynamic sampling at every stage</td><td>Dense-WebVid-CoVR</td><td>95.00</td><td>98.70</td><td>98.80</td><td>99.20</td><td>97.50</td><td>97.92</td><td>0.9663</td></tr><tr><td colspan="9">Where should dynamic allocation enter the cascade?</td></tr><tr><td>Uniform embedding; dynamic reasoning stages</td><td>Dense-WebVid-CoVR</td><td>94.70</td><td>98.40</td><td>98.50</td><td>98.90</td><td>97.20</td><td>97.62</td><td>0.9635</td></tr><tr><td>Dynamic sampling at every stage</td><td>Dense-WebVid-CoVR</td><td>95.00</td><td>98.70</td><td>98.80</td><td>99.20</td><td>97.50</td><td>97.92</td><td>0.9663</td></tr><tr><td colspan="9">How should novelty and uniform density be mixed?</td></tr><tr><td>Novelty-mixing coefficient 0.8</td><td>Dense-WebVid-CoVR</td><td>93.40</td><td>98.00</td><td>98.20</td><td>98.90</td><td>96.53</td><td>97.12</td><td>0.9548</td></tr><tr><td>Novelty-mixing coefficient 0.9</td><td>Dense-WebVid-CoVR</td><td>94.30</td><td>98.10</td><td>98.50</td><td>98.90</td><td>96.97</td><td>97.45</td><td>0.9611</td></tr><tr><td>Novelty-mixing coefficient 1.0</td><td>Dense-WebVid-CoVR</td><td>94.70</td><td>98.40</td><td>98.50</td><td>98.90</td><td>97.20</td><td>97.62</td><td>0.9635</td></tr></table>

Table S9. Target-duration analysis for matched frame policies. Dynamic allocation helps the aggregate result and the shorter-duration groups, while the long-duration groups expose heterogeneity and greater small-sample uncertainty in the long-duration tail.
<table><tr><td>Policy</td><td>Target duration</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>M3</td><td>MRR</td></tr><tr><td>Uniform at all stages</td><td>[0, 10) s</td><td>94.50</td><td>97.25</td><td>97.71</td><td>98.17</td><td>96.48</td><td>0.9595</td></tr><tr><td>Uniform at all stages</td><td>[10, 30) s</td><td>94.55</td><td>99.00</td><td>99.43</td><td>99.57</td><td>97.66</td><td>0.9659</td></tr><tr><td>Uniform at all stages</td><td>[30, 60) s</td><td>92.21</td><td>93.51</td><td>93.51</td><td>94.81</td><td>93.07</td><td>0.9275</td></tr><tr><td>Uniform at all stages</td><td>[60, ∞) s</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>1.0000</td></tr><tr><td>Uniform embedding; dynamic reasoning</td><td>[0, 10) s</td><td>95.41</td><td>97.25</td><td>97.25</td><td>98.17</td><td>96.64</td><td>0.9627</td></tr><tr><td>Uniform embedding; dynamic reasoning</td><td>[10, 30) s</td><td>95.12</td><td>99.28</td><td>99.43</td><td>99.57</td><td>97.94</td><td>0.9701</td></tr><tr><td>Uniform embedding; dynamic reasoning</td><td>[30, 60) s</td><td>88.31</td><td>93.51</td><td>93.51</td><td>94.81</td><td>91.77</td><td>0.9030</td></tr><tr><td>Uniform embedding; dynamic reasoning</td><td>[60, ∞) s</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>1.0000</td></tr><tr><td>Dynamic at all stages</td><td>[0, 10) s</td><td>96.33</td><td>98.17</td><td>98.17</td><td>99.08</td><td>97.55</td><td>0.9716</td></tr><tr><td>Dynamic at all stages</td><td>[10, 30) s</td><td>95.41</td><td>99.43</td><td>99.57</td><td>99.71</td><td>98.13</td><td>0.9719</td></tr><tr><td>Dynamic at all stages</td><td>[30, 60) S</td><td>87.01</td><td>93.51</td><td>93.51</td><td>94.81</td><td>91.34</td><td>0.8967</td></tr><tr><td>Dynamic at all stages</td><td>[60, ∞) s</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>1.0000</td></tr></table>

![](images/3e5a91d109a2ef864504c7c21c0f77d75cd812e620f9db12e1b2aa56db46bb49.jpg)  
Figure S2. Timestamp overlay. Selected frames carry elapsed seconds in the top-left corner, giving the reasoning stages an absolute temporal cue even when novelty-weighted samples are irregularly spaced. This visualization is separate from the dynamic-allocation comparison in Figure 3.

Table S10. Temporal ablations for the positive aggregate dynamic-sampling result and the combined timestamp-aware package. The timestamp rows jointly change decomposer overlays, verifier overlays, and the timestamp note; they are not component-wise causal estimates.
<table><tr><td>Configuration R@1 R@5 R@10</td></tr><tr><td>R@50 M3 M4 MRR ∆R@1</td></tr><tr><td>Does the dynamic frame-policy package improve aggregate retrieval? Uniform frame allocation 94.40 98.20 98.60 98.90 97.07 97.52 0.9619</td></tr><tr><td>Dynamic frame-policy package 95.00 98.70 98.80 99.20 97.50 97.92 0.9663 +0.60</td></tr><tr><td>Does timestamp-aware reasoning improve temporal grounding?</td></tr><tr><td>Timestamp-aware package disabled 94.70 98.80 98.90 99.20 97.47 97.90 0.9652</td></tr><tr><td>Timestamp-aware package enabled 95.00 98.70 98.80 99.20 97.50 97.92 0.9663 +0.30</td></tr><tr><td></td></tr></table>

Table S11. Cross-dataset extensions for frame count, decomposition activation, and decomposition placement. Full CoVR-R WebVid and Something-Something-V2 subset rows retain their actual subset identities; WebVid-CoVR and Dense-WebVid-CoVR rows remain separately labeled. The results show domain-dependent frame-count preferences and mixed embedding-side decomposition effects, supporting the conservative 15-frame shared ceiling and reranker-only final placement.
<table><tr><td>Configuration</td><td>Dataset</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>M3</td><td>M4</td><td>MRR</td></tr><tr><td colspan="9">Frame budget</td></tr><tr><td>15 frames at all stages</td><td>CoVR-R WebVid subset</td><td>97.15</td><td>97.81</td><td>97.88</td><td>98.17</td><td>97.61</td><td>97.75</td><td>0.9749</td></tr><tr><td>7 frames at all stages</td><td>CoVR-R WebVid subset</td><td>94.96</td><td>95.69</td><td>95.76</td><td>96.13</td><td>95.47</td><td>95.64</td><td>0.9534</td></tr><tr><td>10 embedding frames; 15 later-stage frames</td><td>CoVR-R WebVid subset</td><td>95.69</td><td>96.20</td><td>96.20</td><td>96.49</td><td>96.03</td><td>96.15</td><td>0.9597</td></tr><tr><td colspan="9">Something-Something-V2 frame count</td></tr><tr><td>15 embedding frames</td><td>CoVR-R Something-Something-V2 subset</td><td>78.10</td><td>78.67</td><td>78.67</td><td>79.63</td><td>78.48</td><td>78.77</td><td>0.7857</td></tr><tr><td>10 embedding frames</td><td>CoVR-R Something-Something-V2 subset</td><td>82.95</td><td>83.78</td><td>83.84</td><td>84.61</td><td>83.52</td><td>83.80</td><td>0.8352</td></tr><tr><td>5 embedding frames</td><td>CoVR-R Something-Something-V2 subset</td><td>86.65</td><td>87.48</td><td>87.48</td><td>87.93</td><td>87.21</td><td>87.39</td><td>0.8720</td></tr><tr><td colspan="9">Decomposition activation</td></tr><tr><td>Decomposition off</td><td>WebVid-CoVR</td><td>34.78</td><td>55.63</td><td>65.02</td><td>82.75</td><td>51.81</td><td>59.55</td><td>0.4448</td></tr><tr><td>Reranker-side decomposition</td><td>WebVid-CoVR</td><td>55.63</td><td>73.47</td><td>78.25</td><td>82.75</td><td>69.12</td><td>72.53</td><td>0.6357</td></tr><tr><td>Decomposition off</td><td>Dense-WebVid-CoVR</td><td>86.58</td><td>92.76</td><td>93.46</td><td>94.36</td><td>90.93</td><td>91.79</td><td>0.8935</td></tr><tr><td>Reranker-side decomposition</td><td>Dense-WebVid-CoVR</td><td>88.65</td><td>93.35</td><td>93.82</td><td>94.36</td><td>91.94</td><td>92.54</td><td>0.9084</td></tr><tr><td>Decomposition off</td><td>CoVR-R WebVid subset</td><td>96.49</td><td>97.95</td><td>98.10</td><td>98.17</td><td>97.52</td><td>97.68</td><td>0.9720</td></tr><tr><td>Reranker-side decomposition</td><td>CoVR-R WebVid subset</td><td>97.15</td><td>97.81</td><td>97.88</td><td>98.17</td><td>97.61</td><td>97.75</td><td>0.9749</td></tr><tr><td colspan="9">Decomposition placement</td></tr><tr><td>Reranker only</td><td>Dense-WebVid-CoVR</td><td>88.65</td><td>93.35</td><td>93.82</td><td>94.36</td><td>91.94</td><td>92.54</td><td>0.9084</td></tr><tr><td>Embedding + reranker</td><td>Dense-WebVid-CoVR</td><td>88.14</td><td>92.95</td><td>93.27</td><td>93.93</td><td>91.45</td><td>92.07</td><td>0.9038</td></tr><tr><td>Reranker only</td><td>WebVid-CoVR</td><td>55.63</td><td>73.47</td><td>78.25</td><td>82.75</td><td>69.12</td><td>72.53</td><td>0.6357</td></tr><tr><td>Embedding + reranker</td><td>WebVid-CoVR</td><td>57.55</td><td>76.76</td><td>83.06</td><td>89.16</td><td>72.46</td><td>76.63</td><td>0.6629</td></tr></table>

Table S12. Controlled inference cost for the selected configuration on size-matched 1,000-query, 1,000-target cohorts. Online busy time sums decomposition, query embedding, reranking, and verification model intervals; online wall is their interval union under 16-worker execution. Evaluator wall spans the evaluation routine after setup and before reporting and persistence, additionally including presampling, gallery preparation, retrieval, and analysis. Cross-run caches are disabled and in-run reuse is retained. Calls count issued model inferences, with gated skips counted as zero. CoVR-R denotes its controlled WebVid-subset cohort.

(a) Measured execution and online compute
<table><tr><td rowspan="2">Dataset</td><td colspan="3">Time (s)</td><td colspan="2">Online calls</td></tr><tr><td>Mean busy/query</td><td>Online wall</td><td>Evaluator wall</td><td>Mean/query</td><td>Total</td></tr><tr><td>WebVid-CoVR</td><td>69.751</td><td>4416.26</td><td>5330.73</td><td>76.264</td><td>76,264</td></tr><tr><td>Dense-WebVid-CoVR</td><td>38.630</td><td>2488.34</td><td>3233.29</td><td>50.514</td><td>50,514</td></tr><tr><td>CoVR-R</td><td>17.842</td><td>1204.25</td><td>2012.57</td><td>23.227</td><td>23,227</td></tr></table>

(b) Routing activity on size-matched cohorts
<table><tr><td>Dataset</td><td>Embedding-gap bypass</td><td>Expanded</td><td>Decomposed</td><td>Verified</td></tr><tr><td>WebVid-CoVR</td><td>7.3%</td><td>75.4%</td><td>99.6%</td><td>16.1%</td></tr><tr><td>Dense-WebVid-CoVR</td><td>7.6%</td><td>44.2%</td><td>26.1%</td><td>6.5%</td></tr><tr><td>CoVR-R</td><td>7.7%</td><td>17.4%</td><td>11.1%</td><td>3.6%</td></tr></table>

Table S13. Stage and reusable-preparation breakdown for the controlled profiles in Table S12. Stage cells report the mean busy seconds and issued model calls per query, including zero cost when a conditional stage is bypassed. Temporal selection is query-independent and reports one low-resolution probe call per inspected frame; its unique clip–stage selections and gallery embeddings are reusable.

(a) Mean online stage cost per query: busy seconds / model calls
<table><tr><td>Dataset</td><td colspan="2">Decomposition</td><td colspan="2">Embedding</td><td colspan="2">Reranking</td><td colspan="2">Verification</td></tr><tr><td></td><td>Time</td><td>Calls</td><td>Time</td><td>Calls</td><td>Time</td><td>Calls</td><td>Time</td><td>Calls</td></tr><tr><td>WebVid-CoVR</td><td>15.603</td><td>0.996</td><td>1.434</td><td>1.000</td><td>47.898</td><td>74.015</td><td>4.815</td><td>0.253</td></tr><tr><td>Dense-WebVid-CoVR</td><td>3.733</td><td>0.261</td><td>1.437</td><td>1.000</td><td>31.848</td><td>49.160</td><td>1.613</td><td>0.093</td></tr><tr><td>CoVR-R</td><td>1.897</td><td>0.111</td><td>1.483</td><td>1.000</td><td>13.741</td><td>22.075</td><td>0.721</td><td>0.041</td></tr></table>

(b) Reusable preparation
<table><tr><td>Dataset</td><td>Selection tasks</td><td>Mean s/task</td><td>Calls/task</td><td>Total probe calls</td><td>Gallery s/item</td><td>Gallery calls</td></tr><tr><td>WebVid-CoVR</td><td>3,958</td><td>2.824</td><td>33.916</td><td>134,240</td><td>1.420</td><td>1,000</td></tr><tr><td>Dense-WebVid-CoVR</td><td>3,958</td><td>2.634</td><td>33.916</td><td>134,240</td><td>1.422</td><td>1,000</td></tr><tr><td>CoVR-R</td><td>3,954</td><td>2.889</td><td>33.918</td><td>134,112</td><td>1.425</td><td>1,000</td></tr></table>

Table S14. Routing rates for the three complete evaluations, including embedding-gap skips, expansion and rescue rates, and mean expansion depth; these measurements expose the different workloads induced by each edit distribution.
<table><tr><td>Dataset</td><td>Gap skips</td><td>Expanded</td><td>Rescue rate</td><td>Mean rounds</td></tr><tr><td>WebVid-CoVR</td><td>2.86%</td><td>80.75%</td><td>14.51%</td><td>8.485</td></tr><tr><td>Dense-WebVid-CoVR</td><td>3.17%</td><td>46.50%</td><td>9.78%</td><td>8.000</td></tr><tr><td>CoVR-R</td><td>3.00%</td><td>27.11%</td><td>15.07%</td><td>6.109</td></tr></table>

Table S14. Final routing rates (continued: decomposition work).
<table><tr><td>Dataset</td><td>Decomposed</td><td>Concise-edit route</td><td>Ambiguity route</td><td>Bypassed</td></tr><tr><td>WebVid-CoVR</td><td>99.80%</td><td>99.22%</td><td>0.59%</td><td>0.20%</td></tr><tr><td>Dense-WebVid-CoVR</td><td>34.44%</td><td>3.25%</td><td>31.19%</td><td>65.56%</td></tr><tr><td>CoVR-R</td><td>15.45%</td><td>0.00%</td><td>15.45%</td><td>84.55%</td></tr></table>

Table S14. Final routing rates (continued: verification work).
<table><tr><td>Dataset</td><td>Verified</td><td>Mean depth</td><td>Repair rate</td><td>Regression rate</td><td>Net R@1</td></tr><tr><td>WebVid-CoVR</td><td>24.73%</td><td>2.718</td><td>1.88%</td><td>0.55%</td><td>+1.33</td></tr><tr><td>Dense-WebVid-CoVR</td><td>13.11%</td><td>2.755</td><td>1.06%</td><td>0.08%</td><td>+0.98</td></tr><tr><td>CoVR-R</td><td>6.76%</td><td>2.573</td><td>0.27%</td><td>0.15%</td><td>+0.11</td></tr></table>

Table S15. Effective final configuration, grouped by reusable, always-on, conditional candidate, conditional generative, and executionboundary work. It records model roles, interfaces, frame policies, routing, generation, retries, concurrency, and reuse. The 15-frame setting is an upper bound per video input; short or unreadable clips can yield fewer, verification receives separate source and candidate inputs, and the selector may inspect additional low-resolution probes.
<table><tr><td>Class</td><td>Component</td><td>Model</td><td>Input → output</td><td>Frames / pixels / time</td></tr><tr><td>Reusable / offline</td><td>Gallery preparation</td><td>Deterministic data path</td><td>Validated target-only gallery → Deduplicated target list and query-independent frame selections</td><td>Uniform; up to 15 target frames; 512 px long side; timestamps off</td></tr><tr><td>Reusable for targets; online for a new source</td><td>Dynamic novelty probe</td><td>Qwen3-VL-Embedding-2B</td><td>Individual query-independent probe frames → L2-normalized per-frame vectors and adjacent novelty</td><td>8/4/2 FPS for [0, 10)/[10, 30)/ [30, ∞) s; inverse-CDF selection; novelty-mixing coefficient 1.0; per-clip output budget 15; 256 px probe frames;</td></tr><tr><td>Always-on online</td><td>Composed Query Embedding</td><td>Qwen3-VL-Embedding-2B</td><td>Source video + raw modification text → Dense query vector, L2-normalized</td><td>timestamps off during probing Uniform; up to 15 source frames; 512 px long side; timestamps off</td></tr><tr><td>Reusable / offline</td><td>Reusable Gallery Encoding</td><td>Qwen3-VL-Embedding-2B</td><td>Target video only → One dense vector per target, L2-normalized</td><td>Uniform; up to 15 target frames; 512 px long side; timestamps off</td></tr><tr><td>Always-on online</td><td>Coarse retrieval and masking</td><td>Cosine similarity over normal- ized vectors</td><td>Query vector + reusable target matrix → Initial top five candidates</td><td>No additional video input; n/a; timestamps n/ a</td></tr><tr><td>Conditional online candidate work</td><td>Confidence-Gated Adaptive Reranking</td><td>Qwen3-VL-Reranker-8B</td><td>Raw edit or one decomposed target description + candidate video; source video absent → Hosted continuous relevance score</td><td>Dynamic; up to 15 candidate frames; 256 px long side; timestamps off</td></tr><tr><td>Conditional online generative work</td><td>Conditional Query Decomposition</td><td>Qwen3.5-9B (shared generation model)</td><td>Source video + raw modification text → Generated description; whitespace trimmed and one enclosing quote pair removed (prompt requests one or two sentences)</td><td>Dynamic; up to 15 source frames; 256 px long side; timestamps on and expressed in seconds to two decimal places</td></tr><tr><td>Conditional online generative work</td><td>Confidence-Gated Candidate Veri- fication</td><td>Qwen3.5-9B (shared generation model)</td><td>Source video + raw modification + one candidate video → Boolean or unclear verdict from a leading yes/no parser; strictness is prompt-defined</td><td>Dynamic; up to 15 source and 15 candidate frames; 512 px long side; timestamps on for both clips and expressed in seconds to two decimal places</td></tr><tr><td>Execution boundary</td><td>Concurrency and presampling</td><td>16 evaluation workers; dynamic-probe workers</td><td>16 Full active source/gallery task set → Presampled indices followed by lazy final frame bundles</td><td>Frame indices are presampled; complete frame bundles are not preloaded; stage- specific caps appear above; timestamps are rendered only during final decomposer/</td></tr><tr><td>Execution boundary</td><td>Timeouts and retry layers</td><td>All hosted model clients</td><td>Stateless model requests and failed evaluator items → Bounded transport recovery followed by role-specific failed-item replay</td><td>verifier bundle construction Probe requests use their separate dynamic pol- icy; n/a; timestamps n/a</td></tr></table>

Table S15. Effective final configuration (continued: routing, execution, and caching).
<table><tr><td>Component</td><td>Routing / thresholds</td><td>Sampling / retries</td><td>Cache and reuse</td></tr><tr><td>Gallery preparation</td><td>Source self-mask on; positive limit applied before gallery rebuild</td><td>Deterministic</td><td>Selections and one target embedding are reusable across queries</td></tr><tr><td>Dynamic novelty probe</td><td>No text/query conditioning; uniform fallback if novelty mass is zero</td><td>16 concurrent probe workers in final runs</td><td>Probe vectors key clip/model/FPS/resolution/version; selec- tions also kev budget/ratio/bands/rates</td></tr><tr><td>Composed Query Embedding</td><td>Joint input; role-specific instruction</td><td>One query embedding request</td><td>In-run and atomic cross-run embedding cache keyed by role, content, instruction, model, and resolved frames</td></tr><tr><td>Reusable Gallery Encoding</td><td>An explicit empty target instruction is transmitted</td><td>One embedding request per distinct target</td><td>Stored normalized gallery matrix is reused by every query</td></tr><tr><td>Coarse retrieval and masking</td><td>Mask source item when present; skip reranking only if the gap between the two highest embedding scores exceeds 0.25</td><td>Equality at 0.25 reranks</td><td>Gallery matrix reused; search is query-specific</td></tr><tr><td>Confidence-Gated Adaptive Reranking</td><td>Initial 5; append 5 while the highest reranker score is below 0.70; at most 9 expansions / 50 candidates; expose top 10 downstream Early if ASCII word count is below 10; otherwise late if the highest reranker score is at</td><td>No client instruction is supplied; unchanged pair scores are reused Thinking 2048; max 2560 tokens; temperature</td><td>Prediction key includes model, query form/content, frame sig- nature, resolution, timestamps, and prompt One description is reused by reranking; failure/empty output</td></tr><tr><td></td><td>least 0.25 and its relative gap to the second-highest score is below 0.50; reranker only</td><td>unset -&gt; 0.6, top-p 0.95, top-k 20; 10 failed- item follow-up passes after transport handling; un- seeded Thinking 2048; max 2560 tokens; temperature</td><td>falls back to raw text</td></tr><tr><td>tion</td><td>contiguous top-10 prefix whose relative score decrease from the leader is below 0.25; stop at first yes</td><td>unset -&gt; 0.6, top-p 0.95, top-k 20; 10 failed- item follow-up passes after transport handling; un- seeded</td><td>Stable promotion; skip/failure/all-no retains prior valid order</td></tr><tr><td>Concurrency and presampling</td><td>Queries/items parallel; expansion pages and verifier scans sequential within a query</td><td>Early decomposition overlaps query then gallery embedding after the prepass</td><td>Presampling resolves probes/indices only; decode, resize, over- lay, and JPEG packing remain lazy</td></tr><tr><td>Timeouts and retry layers</td><td>General calls: 300 s timeout, 5 transport retries; dynamic probes: 120 s timeout, 5 transport retries</td><td>Transport retries 429/500/502/503/504 and connec- tion/read failures; backoff factor 0.5 s, cap 30 s, jitter 0.5 s; general pool 64, probe pool 16; decom- poser/verifier add up to 10 failed-item passes with</td><td>Transport retry repeats one stateless request; failed-item passes rerun only unresolved evaluator items</td></tr></table>

Table S16. Exact model-interface text and request contracts. Embedding rows preserve the literal role instructions, the reranker row records the tracked serving-template fields and unset client instruction, and the generative rows preserve the complete runtime system prompts, labeled user-message order, output parsing, timestamp note, effective sampling parameters, thinking budget, maximum generation length, seed policy, and both transport- and evaluator-level retry layers.
<table><tr><td>Role</td><td>Model</td><td>User layout</td><td>Output</td><td>Timestamp / runtime</td></tr><tr><td>Composed query embedding</td><td>Qwen3-VL-Embedding-2B</td><td>source video: raw modification text</td><td>one dense vector</td><td>timestamps off; L2-normalize before cosine search</td></tr><tr><td>Source-video-only embedding ablation</td><td>Owen3-VL-Embedding-2B</td><td>source video</td><td>one dense vector</td><td>timestamps off: Ablation interface: not the final composed-query form</td></tr><tr><td>Text-only embedding ablation</td><td>Qwen3-VL-Embedding-2B</td><td>raw modification text</td><td>one dense vector</td><td>timestamps n/a; Ablation interface; not the final composed-query form</td></tr><tr><td>Target gallery embedding</td><td>Qwen3-VL-Embedding-2B</td><td>target video only; an explicit empty instruction is transmitted</td><td>one dense vector</td><td>timestamps off: L2-normalize once and store</td></tr><tr><td>Query-independent dynamic-frame probe</td><td>Qwen3-VL-Embedding-2B</td><td>one probe image; empty system instruction is transmitted</td><td>one dense vector per probed frame; adjacent 1-cos supplies novelty</td><td>timestamps off; query independent; normalized frame vectors; no language query</td></tr><tr><td>Candidate reranking</td><td>Qwen3-VL-Reranker-8B</td><td>raw edit or decomposed target description as the search request; candidate video as the document; source video absent; no client instruction is supplied</td><td>hosted continuous relevance score consumed directly</td><td>timestamps off; client does not reconstruct yes/no token probabilities</td></tr><tr><td>Conditional query decomposition</td><td>Qwen3.5-9B</td><td>source-video frames followed by the raw modification text</td><td>prompt requests one or two plain sentences; parser trims whitespace and removes one enclosing matching quote pair only</td><td>timestamps on; timestamp note appended; thinking on; budget 2048; max 2560; effective temperature/top-p/top-k 0.6/0.95/ 20; 300 s request timeout + 5 transport retries; up to 10</td></tr><tr><td>Strict candidate verification</td><td>Qwen3.5-9B</td><td>candidate-video frames</td><td>prompt requests one word; parser trims/ lowercases, strips leading quote/backtick/ asterisk/space characters, accepts a response beginning yes or no, and otherwise returns unclear</td><td>failed-item follow-up passes with 2 s waits; unseeded timestamps on for both videos; timestamp note appended; thinking on; budget 2048; max 2560; effective temperature/ top-p/top-k 0.6/0.95/20; 300 s request timeout + 5 transport retries; up to 10 failed-item follow-up passes with 2 s waits; unseeded</td></tr></table>

Table S16. Exact model-interface contracts (continued: embedding instructions).
<table><tr><td>Embedding role</td><td>Exact instruction</td></tr><tr><td>Composed query embedding</td><td>Given this source video with the desired modification text, represent a potential target video for retrieval.</td></tr><tr><td>Source-video-only embedding ablation</td><td>Represent this video for retrieval.</td></tr><tr><td>Text-only embedding ablation</td><td>Represent this text query for retrieving the described video.</td></tr><tr><td>Target gallery embedding</td><td>&lt;empty string transmitted&gt;</td></tr><tr><td>Query-independent dynamic-frame probe</td><td>&lt;empty string transmitted&gt;</td></tr></table>

Table S16. Exact model-interface contracts (continued: reranker serving template).

Candidate reranking Qwen3-VL-Reranker-8B

Prompt contract. The system asks whether the candidate document satisfies the search request under the supplied retrieval instruction, and restricts the verbal judgment to “yes” or “no.” The user turn contains three semantic fields: a retrieval instruction, the search request, and the candidate document. A supplied system instruction fills the first field; otherwise the service uses the default instruction “Given a search query, retrieve relevant candidates that answer the query.” The search request is the raw edit or decomposed target description, and the candidate document is the candidate video; the source video is absent. No client instruction is supplied in our configuration, so the default retrieval instruction applies.

## Table S16. Exact model-interface contracts (continued: decomposition prompt).

## Conditional query decomposition Qwen3.5-9B

You are a professional video analyst helping with composed video retrieval.   
You are given a query video and a modification text describing the changes to apply to it.   
Your task is to write a single, self-contained description of the TARGET video: the video obtained by applying the described modifications to the query video.   
Keep every detail of the query video that the modification does not change (subjects, actions, scene, setting, mood), and apply every requested change (a ,→ modification may alter the subject, the action, or the context, and may bundle several changes at once).   
Write one or two plain sentences describing the target video as it would actually appear. Do not mention the query video, the modification, or the retrieval task;   
,→ output only the description.

A timestamp is drawn in each frame for reference only and is not part of the video. Videos do not have to be of the same length.

## Table S16. Exact model-interface contracts (continued: verification prompt).

## Strict candidate verification Qwen3.5-9B

You are a professional video analyst.   
You are given a query video with a modification text mentioning required changes to the query video, along with a potential target video.   
First, analyze the query video with modification text, and the target video. Next, answer whether the target video is relevant to user's query.   
Answer 'yes' only if everything mentioned in the modification query is clearly presented in the target video, without ambiguities.It can still be 'yes' if the found differences are not specifically mentioned in the modification query.Answer 'no' if at least one thing mentioned in the modification query is missing, only,→ weakly implied, contradicted, or you are unsure.,→   
Only answer 'yes' or 'no' with one word.

A timestamp is drawn in each frame for reference only and is not part of the video. Videos do not have to be of the same length.

## D. Limitations and Future Directions

The study is comprehensive stage-wise but not exhaustive over every foundation-model family, and stochastic generative ablations are primarily single evaluations. Dynamic selection is query-independent and can under-allocate slowly evolving evidence; training-free query-aware allocation is a natural next step. Fixed gates and the bounded candidate ceiling calibrate well on the evaluated galleries but remain a hard recall boundary because targets omitted upstream are unreachable by verification. Target-only galleries and single-positive annotations also under-measure multiplevalid-answer uncertainty, as the campfire case demonstrates. Strict verification is net positive in every final evaluation but can still reject a correct near-tied candidate. Future work should examine calibration under domain shift, longer and streaming video, multiple-valid-target evaluation, alternative interface-compatible model families, query-aware trainingfree selection, and adaptive candidate budgets that preserve the cascade’s reusable/conditional separation.