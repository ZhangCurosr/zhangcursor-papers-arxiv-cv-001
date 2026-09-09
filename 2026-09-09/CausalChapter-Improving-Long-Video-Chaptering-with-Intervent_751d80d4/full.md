# CausalChapter: Improving Long-Video Chaptering with Interventional Dependency Modeling

Xinran Duan, Guozhang Li, Yaoyao Zhong, Mei Wang<sup>\*</sup>, Lizhi Wang, Hua Huang

School of Artificial Intelligence, Beijing Normal University

Beijing Key Laboratory of Artificial Intelligence for Education

Engineering Research Center of Intelligent Technology and Educational Application, Ministry of Education duanxinran@mail.bnu.edu.cn, {liguozhang097, zhongyy}@bnu.edu.cn {wangmei1, wanglizhi, huahuang}@bnu.edu.cn

## Abstract

Long-form instructional videos require automatic chaptering to support browsing, navigation, and knowledge access. Recent longcontext language models can perform chaptering from textualized video inputs, but they remain costly and brittle for content-dense lecture videos with long transcripts, smooth topic transitions, and detailed chapter outputs. A scalable segment-then-caption paradigm reduces this cost, but introduces two new challenges: boundary error propagation and fragmented crosschapter context. We propose CausalChapter, an intervention-inspired framework for longvideo chaptering that estimates prediction-level influence through lightweight masking and removal interventions. For boundary localization, our Local Dependency Shift module detects drops in predictive dependency between adjacent temporal windows; for chapter description generation, our Cross-Segment Support Selection module reranks historical contexts according to their support for the current prediction. Experiments on long-video chaptering benchmarks show that CausalChapter improves boundary localization, chapter description quality, and cross-chapter coherence. CausalChapter

## 1 Introduction

Long-form videos, such as lectures, meeting recordings, and online tutorials, have become an important source of knowledge and communication. To make such videos searchable and navigable, models need to organize them into temporally grounded and semantically coherent units. Dense video captioning (DVC) provides a representative localizeand-describe paradigm, where a model detects multiple events in an untrimmed video and generates a description for each event (Krishna et al., 2017;

Iashin and Rahtu, 2020a; Wang et al., 2021; Yang et al., 2023b). However, DVC mainly focuses on short-term event-level segments, whereas longform video navigation requires higher-level temporal organization. Video chaptering addresses this need by partitioning a long video into consecutive chapters and generating navigable titles or descriptions for each chapter (Yang et al., 2023a; Ventura et al., 2025).

Recent progress in long-video chaptering has been driven by large-scale datasets and longcontext language models. VidChapters-7M introduces a large-scale benchmark of user-annotated chapters for open-domain videos (Yang et al., 2023a). Chapter-Llama converts videos into timestamped Automatic Speech Recognition (ASR) transcripts and frame captions, and predicts chapter boundaries and free-form titles with a long-context LLM in a single forward pass (Ventura et al., 2025). While effective, such holistic long-context modeling faces increasing cost in content-dense instructional videos, where ASR transcripts are long, topic transitions are smooth, and chapter outputs often require detailed descriptions rather than short titles, as illustrated in Fig. 1(a). Under practical context budgets, processing the entire textualized video may require truncation, sparse sampling, or compression (Wang et al., 2021; Yang et al., 2023b; Kim et al., 2024), which may discard fine-grained evidence needed for boundary localization and chapter description generation.

A scalable alternative is the segment-thencaption paradigm (Islam et al., 2024; Zala et al., 2023): the model first predicts chapter boundaries to divide a long video into local segments, and then generates a description for each segment. This decomposition reduces the input length of each generation step and aligns the generation input with chapter-level outputs. However, it also shifts the central challenge from holistic encoding to identifying which local transitions and historical segments truly matter for prediction. First, boundary errors may propagate to the generation stage: a shifted boundary can mix adjacent chapter content into the current segment or omit key semantic units. Second, independent segment-level generation can break cross-chapter context, which is often crucial in lectures and tutorials where later chapters depend on earlier definitions, assumptions, or examples.

![](images/0311c4c87be21b7f2a671f77b7f765d0699c3914302a43ca3f06e10cbb502d99.jpg)

(a) Cost comparison of whole-video and chapter-level inputs. Chapter A：Gradient Descent Chapter-Transition Chapter B: Overfitting  
![](images/7b26b3a0a782488ebe1088e11379fe7df0e980768a4d34b4d9ac2d87b1b77e7e.jpg)

![](images/c78510b2ffc769bb73e454e7992397b58598da68d171fd32db3f967d15548c37.jpg)  
(c) Dependency-aware retrieval preserves predictive cues.  
Figure 1: Motivation of scalable and dependency-aware long-video chaptering. (a) Content-dense instructional videos require detailed chapter descriptions over long ASR transcripts, making whole-video modeling costly in tokens and memory. (b) Smooth lecture transitions can preserve local coherence while predictive dependency drops near the true chapter boundary. (c) Plausible retrieved contexts can induce mechanism drift in chapter generation, whereas prediction-critical context preserves the intended explanation.

A straightforward solution is to augment each segment with additional context retrieved from nearby or semantically similar segments. Existing retrieval- or memory-augmented methods commonly select context based on semantic similarity, temporal proximity, or visual similarity (Kim et al., 2024, 2025). However, in instructional long videos, surface relevance does not necessarily imply prediction utility. A highly similar segment may simply repeat the current content, while an earlier segment with lower lexical or visual similarity may introduce a definition or logical premise that is crucial for describing the current chapter, as shown in Figure 1(c). This suggests a different criterion for longvideo chaptering: a temporal unit should be judged by whether intervening on it changes the model’s prediction, rather than by whether it is visually similar, temporally close, or lexically overlapping with the current segment. The same criterion can also inform boundary localization. Within a coherent chapter, preceding units usually provide predictive support for subsequent units; near a chapter transition, this dependency may drop even when the transition is visually or lexically smooth, as shown in Figure 1(b).

Motivated by this observation, we propose CausalChapter, an intervention-inspired framework for scalable long-video chapter generation. CausalChapter estimates intervention-defined predictive dependency, namely the observable change in model prediction caused by masking, perturbing, or removing input components. Our goal is not to recover the real-world causal structure of video content, but to measure the prediction-level influence of semantic units and context segments as a practical signal of predictive support. For boundary localization, we introduce Local Dependency Shift (LCDS), which masks semantic units in a preceding temporal window and measures how much the intervention affects the reconstruction of the subsequent window. A local drop in this dependency provides an auxiliary signal for detecting smooth chapter transitions. For chapter description generation, we introduce Cross-Segment Support Selection (CSSE), which removes candidate context segments and measures their influence on the generated description. Segments with high estimated support are selected for second-pass generation, improving the completeness and cross-chapter coherence of the final chapter descriptions.

Our contributions are summarized as follows: (1) We introduce LCDS, an intervention-inspired dependency-drop signal for smooth chapter boundary localization. (2) We introduce CSSE, an intervention-based support estimation mechanism for cross-segment context selection. (3) We show that the proposed interventional mechanism mitigates boundary error propagation and context fragmentation in scalable long-video chaptering.

## 2 Related Work

Video Chaptering. Video chaptering aims to partition a long video into consecutive, nonoverlapping, and semantically coherent chapters, while generating titles or summaries for browsing and navigation. A related task is dense video captioning (DVC), which localizes and describes multiple events in untrimmed videos (Iashin and Rahtu, 2020a,b; Yang et al., 2023b; Wang et al., 2021; Kim et al., 2024; Liu et al., 2025; Wu et al., 2025; Xie et al., 2025; Li et al., 2025). Recent DVC studies further incorporate memory or retrievalaugmented mechanisms to improve event-level descriptions (Kim et al., 2024; Xie et al., 2025; Wu et al., 2025; Liu et al., 2025). Although DVC follows a localize-and-describe paradigm, it mainly focuses on short-term event-level segments whose boundaries are often associated with local visual or event changes, making it less suited to long-form videos governed by high-level topic progression.

For long-form video chaptering, VidChapters-7M introduces a large-scale dataset of userannotated chapters and defines several chaptering tasks, including chapter generation and chapter grounding (Yang et al., 2023a). More recently, Chapter-Llama represents long videos as timestamped ASR transcripts and frame captions, and uses a long-context LLM to jointly predict chapter boundaries and free-form chapter titles in a single forward pass (Ventura et al., 2025). These studies demonstrate the effectiveness of textualized video representations and long-context reasoning. Different from holistic long-context chaptering methods, we study a scalable segment-then-caption formulation and explicitly address the boundary error propagation and cross-segment context fragmentation introduced by this decomposition.

Context Augmentation and Selection. Context augmentation has been widely explored in videolanguage generation through memory propagation, retrieval augmentation, and evidence selection (Kim et al., 2024; Li et al., 2023; Yu et al., 2023; Xie et al., 2025; Wu et al., 2025; Liu et al., 2025; Li et al., 2026). Existing methods commonly identify useful context based on semantic similarity, temporal proximity, cross-modal matching, or attention-based relevance. However, for information-dense lectures, surface-level relevance does not necessarily reflect prediction utility. An earlier definition, assumption, or logical premise may be crucial for the current chapter despite low lexical or visual similarity, while a highly similar segment may simply repeat the current content. Our work therefore treats retrieval as candidate construction only, and performs final context selection according to each segment’s observable influence on current chapter generation.

Causal Video Reasoning. Causal and counterfactual reasoning has been introduced into video understanding to reduce spurious correlations, mitigate language priors, and model event relations (Xiao et al., 2021; Niu et al., 2021; Liu et al., 2023; Chen et al., 2026). Prior studies construct causal video question answering benchmarks, apply counterfactual interventions for bias reduction, or discover event-level causal structures for video reasoning. Different from these works, we use lightweight masking and removal interventions to estimate prediction-level influence for longvideo chaptering, making our approach closer to intervention-based utility estimation than to causal structure discovery.

## 3 Method

## 3.1 Task Formulation and Overview

Given a long video V , the goal is to generate a temporally grounded chapter sequence $\mathcal { C } =$ $\{ ( s _ { k } , e _ { k } , y _ { k } ) \} _ { k = 1 } ^ { K }$ , where $s _ { k }$ and $e _ { k }$ denote the start and end timestamps of the k-th chapter, and $y _ { k }$ denotes its chapter-level description. The predicted chapters are expected to be consecutive, nonoverlapping, and semantically coherent.

We propose CausalChapter, a segment-thencaption framework for long-video chapter generation. CausalChapter first predicts chapter boundaries and then generates a description for each resulting segment. This decomposition is scalable, but it introduces two key challenges, namely boundary error propagation and cross-segment context fragmentation. Figure 2 summarizes the overall pipeline and the two intervention-based modules used to address these challenges. We address them with two intervention-inspired modules: Local Dependency Shift improves boundary localization by measuring drops in predictive dependency between adjacent temporal windows. Cross-Segment Support Selection improves chapter generation by selecting historical segments that provide strong predictive support for the current description. Here, predictive dependency denotes the observable change in model prediction under masking or removal interventions, and serves as an operational measure of predictive support rather than a claim about real-world causal structure.

![](images/364e25e9f0971a46a2407b45b325adae9c6bda2546339667c71b1f4a20588d12.jpg)  
Figure 2: Overview of CausalChapter. Given sentence-level semantic units with aligned visual and ASR information, the framework first predicts chapter boundaries with a segment-then-caption backbone enhanced by Local Dependency Shift (LCDS), which captures dependency drops between adjacent temporal windows. The resulting segments are then described by an LLM-based generator, where Cross-Segment Support Selection (CSSE) reranks historical contexts according to their predictive support for the current chapter. The final output is a sequence of temporally grounded chapter descriptions.

## 3.2 Segment-then-Caption Backbone

We instantiate a segment-then-caption backbone over sentence-level semantic units $\mathcal { U } = \{ u _ { i } = $ $( x _ { i } , t _ { i } ^ { s } , t _ { i } ^ { e } ) \} _ { i = 1 } ^ { N }$ , where $x _ { i }$ denotes the i-th ASR sentence and $t _ { i } ^ { s } , t _ { i } ^ { e }$ its timestamps. For each unit, we sample video frames at 1 FPS within $[ t _ { i } ^ { s } , t _ { i } ^ { e } ]$ and extract CLIP visual features (Radford et al., 2021), and fuse them with the ASR representation to obtain a multimodal semantic-unit representation $z _ { i }$

Given $z _ { i } ,$ , a boundary classifier $g _ { \mathrm { b d } }$ forecasts the probability of a chapter boundary occurring after the i-th unit:

$$
o _ { i } ^ { \mathrm { b a s e } } = g _ { \mathrm { b d } } ( z _ { i } ) , p _ { i } ^ { \mathrm { b a s e } } = \sigma ( o _ { i } ^ { \mathrm { b a s e } } ) _ { c _ { b } } ,\tag{1}
$$

where $o _ { i } ^ { \mathrm { b a s e } } \in \mathbb { R } ^ { 2 }$ is boundary logits, $c _ { b }$ is the boundary class index, and $p _ { i } ^ { \mathrm { b a s e } }$ is the predicted boundary probability. The predicted boundaries are used to divide the video into chapter segments $\boldsymbol { S } = \{ S _ { k } \} _ { k = 1 } ^ { K }$

For each segment $S _ { k }$ , we construct a generation input $I _ { k }$ from the segment ASR text, a compressed visual representation, an explicit temporal prompt, and a temporal feature pooled from semantic-unit representations within $S _ { k }$ . The LLM-based generator $G _ { \theta }$ then produces a first-pass chapter description $\tilde { y } _ { k } = G _ { \theta } ( I _ { k } )$ , where $\theta$ includes trainable parameters such as LoRA adapters and lightweight projection modules. This backbone improves scalability, but its boundary prediction mainly relies on local multimodal evidence, and its generation is primarily conditioned on the current segment.

## 3.3 Local Dependency Shift

Smooth topic transitions are difficult to localize from local visual or lexical changes alone, because adjacent units across chapter boundaries may remain semantically coherent. We treat a boundary as positions where the predictive dependency drops between neighboring temporal windows.

For a candidate boundary after $u _ { i }$ , we construct a preceding window $A _ { i } = \left[ z _ { i - W + 1 } , \dotsc , z _ { i } \right]$ and a following window $B _ { i } = [ z _ { i + 1 } , \dots , z _ { i + W } ]$ , where $W$ is the window size. A MLP-based dependency predictor $R _ { \phi }$ reconstructs the following window from the preceding one, $\hat { B } _ { i } = R _ { \phi } ( A _ { i } )$ . To estimate the contribution of each preceding unit, we mask the r-th unit in $A _ { i }$ to obtain $A _ { i } ^ { \setminus r }$ and predict $\hat { B } _ { i } ^ { \setminus r }$ $R _ { \phi } ( A _ { i } ^ { \backslash r } )$

The intervention effect is measured by the representation change in the predicted following window:

$$
g _ { i , r , j } = D _ { \mathrm { r e p } } ( \hat { b } _ { i , j } , \hat { b } _ { i , j } ^ { \setminus r } ) , d _ { i } = \frac { 1 } { W ^ { 2 } } \sum _ { r = 1 } ^ { W } \sum _ { j = 1 } ^ { W } g _ { i , r , j } ,\tag{2}
$$

where $D _ { \mathrm { r e p } }$ denotes cosine distance, $\hat { b } _ { i , j }$ and $\hat { b } _ { i , j } ^ { \setminus r }$ are the $j \cdot$ -th predicted representations before and after intervention. A larger $d _ { i }$ indicates stronger predictive support from the preceding window to the following one, while a smaller $d _ { i }$ indicates weaker temporal dependency.

We then compare $d _ { i }$ with its neighborhood. Let $\mathcal { N } ( i )$ denote neighboring candidate positions around i within a fixed local radius, and let $\bar { d } _ { i } =$ $\begin{array} { r } { \frac { 1 } { | \mathcal { N } ( i ) | } \sum _ { q \in \mathcal { N } ( i ) } d _ { q } } \end{array}$ be the local reference dependency. Since we only care about dependency drops, the final dependency-shift score is

$$
s _ { i } ^ { \mathrm { d e p } } = \mathrm { N o r m } \big ( \operatorname* { m a x } ( 0 , \bar { d } _ { i } - d _ { i } ) \big ) ,\tag{3}
$$

where $\operatorname { N o r m } ( { \mathord { \cdot } } )$ denotes min-max normalization over all candidate boundary positions in the video. A larger $s _ { i } ^ { \mathrm { d e p } }$ suggests a stronger local dependency drop and is therefore more likely to indicate a smooth chapter boundary.

Finally, we inject this score into the boundaryclass logit, $( o _ { i } ^ { \mathrm { e n h } } ) _ { c _ { b } } = ( o _ { i } ^ { \mathrm { b a s e } } ) _ { c _ { b } } + \gamma s _ { i } ^ { \mathrm { d e p } }$ , where $\gamma$ is a learnable scaling coefficient. The enhanced boundary probability is $p _ { i } ^ { \mathrm { e n h } } = \sigma ( o _ { i } ^ { \mathrm { e n h } } ) _ { c _ { b } }$ . LCDS thus complements the base boundary classifier with an intervention-defined structural cue.

## 3.4 Cross-Segment Support Selection

Independent chapter generation often misses longrange prerequisites such as earlier definitions, assumptions, and problem setups. We therefore estimate the predictive support of historical segments through removal interventions. For each segment $S _ { k }$ , we build a segment representation $h _ { k } = E _ { \mathrm { s e g } } ( S _ { k } , \tilde { y } _ { k } )$ using the segment content, temporal information, segmentation-stage features, and first-pass description. To control computation, we first construct a compact candidate context set:

$$
\begin{array} { r } { C _ { k } = C _ { k } ^ { \mathrm { n e a r } } \cup C _ { k } ^ { \mathrm { s e m } } , } \\ { C _ { k } ^ { \mathrm { s e m } } = \mathrm { T o p M } _ { j < k } \mathrm { S i m } ( h _ { k } , h _ { j } ) . } \end{array}\tag{4}
$$

where $C _ { k } ^ { \mathrm { n e a r } }$ contains temporally neighboring historical segments, $C _ { k } ^ { \mathrm { s e m } }$ contains semantically retrieved segments, and M is a small retrieval budget. This stage only narrows the search space and does not determine the final contexts.

Given $C _ { k }$ , the generator first produces a reference sequence with all candidate contexts, $y _ { k } ^ { \mathrm { r e f } } =$ $G _ { \theta } ( I _ { k } , \tilde { y } _ { k } , C _ { k } )$ . For each candidate segment $S _ { j } \in$ $C _ { k }$ , let $C _ { k } ^ { - j } = C _ { k } \backslash \{ S _ { j } \}$ . We compare the teacherforced output distributions under $C _ { k }$ and $C _ { k } ^ { - j }$ over the same reference prefix. Specifically, define

$$
p _ { t } ( C ) = p _ { \theta } ( \cdot \mid y _ { k , < t } ^ { \mathrm { r e f } } , I _ { k } , \tilde { y } _ { k } , C ) .\tag{5}
$$

The output-difference function is then

$$
D _ { \mathrm { o u t } } ( C _ { k } , C _ { k } ^ { - j } ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \bigl ( p _ { t } ( C _ { k } ) \| p _ { t } ( C _ { k } ^ { - j } ) \bigr ) .\tag{6}
$$

Here, $T$ is the number of evaluated output positions. A larger divergence means that removing $S _ { j }$ changes the model’s predictive distribution more strongly, so $S _ { j }$ is considered to provide stronger predictive support for the current chapter. We define its support score as

$$
s _ { j \to k } = D _ { \mathrm { o u t } } ( C _ { k } , C _ { k } ^ { - j } ) ,\tag{7}
$$

and analyze alternative implementations of $D _ { \mathrm { o u t } }$ in Appendix C.4.

The top-ranked contexts are then selected as $\mathcal { R } _ { k } = \mathrm { T o p K } _ { S _ { i } \in C _ { k } } ( s _ { j  k } )$ , and the final chapter description is generated as $y _ { k } = G _ { \theta } ( I _ { k } , \tilde { y } _ { k } , \mathcal { R } _ { k } )$ This shifts context selection from similarity-driven matching to intervention-driven support estimation. Since interventions are performed only on the compact candidate set $C _ { k }$ , the additional cost scales with the candidate size rather than the total number of video segments.

## 3.5 Training Objective and Inference

The boundary module is trained with supervised boundary labels using

$$
\mathcal { L } _ { \mathrm { b d } } = - \sum _ { i } \log p _ { i } ^ { \mathrm { e n h } } ( b _ { i } ^ { * } ) ,\tag{8}
$$

where $b _ { i } ^ { * }$ is the ground-truth boundary label after $u _ { i }$ . The dependency predictor is trained to reconstruct the following window with $\mathcal { L } _ { \mathrm { r e c } } ~ =$ $\begin{array} { r } { \sum _ { i } D _ { \mathrm { r e c } } ( \hat { B } _ { i } , B _ { i } ) } \end{array}$

The generator is trained with the standard autoregressive objective:

$$
\mathcal { L } _ { \mathrm { g e n } } = - \sum _ { k } \sum _ { t } \log p _ { \theta } ( y _ { k , t } ^ { * } \mid y _ { k , < t } ^ { * } , I _ { k } , \mathcal { R } _ { k } ) ,\tag{9}
$$

where $y _ { k } ^ { * }$ is the ground-truth chapter description, together with a timestamp reconstruction loss

$$
\mathcal { L } _ { \mathrm { t i m e } } = - \sum _ { t \in \Omega _ { \mathrm { t i m e } } } \log p _ { \theta } ( y _ { t } \mid y _ { < t } , I _ { k } ) .\tag{10}
$$

The final objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { b d } } + \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { g e n } } + \mathcal { L } _ { \mathrm { t i m e } } ,\tag{11}
$$

At inference time, CausalChapter first encodes semantic units and predicts enhanced chapter boundaries, which define chapter segments. The generator then produces first-pass descriptions, retrieves candidate historical contexts, estimates their support scores through removal interventions, and generates final descriptions with the top supportive contexts. The final output is the chapter sequence $\mathcal { C } = \{ ( s _ { k } , e _ { k } , y _ { k } ) \} _ { k = 1 } ^ { K }$

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate CausalChapter on two long-video chaptering benchmarks. The primary benchmark is AVLecture (Gupta et al., 2023), which contains long lecture videos with ASR transcripts, OCR outputs, visual content, and humanannotated topic boundaries. Since AVLecture does not provide chapter-level descriptions, we augment each ground-truth segment with a human-verified LLM-assisted chapter description. All models are trained and evaluated on the same augmented references. We use 245 videos for training, 35 for validation, and 70 for testing. We further evaluate generalization on VidChapters-7M (Yang et al., 2023a), which provides user-annotated chapter boundaries and titles. Details of the annotation protocol and annotation-sensitivity analysis are provided in Appendix B.1.

Metrics. We evaluate both chapter localization and description generation. For localization, we report F1@30 and tIoU on AVLecture, and follow prior work to report F1 and tIoU on VidChapters-7M. We additionally use exact-boundary F1 in ablation analyses to expose fine-grained differences among boundary variants. For generation, we report CIDEr (Vedantam et al., 2015) on both datasets. On AVLecture, we report SODA\_c (Fujita et al., 2020), computed with SODA typec using IoU-weighted METEOR matching. On VidChapters-7M, we follow the Chapter-Llama evaluation and report its benchmark notation SODA.

Baselines. We compare CausalChapter with representative dense video captioning methods, including PDVC (Wang et al., 2021) and Vid2Seq (Yang et al., 2023b), as well as longvideo LLM and video chaptering baselines, including VTimeLLM (Huang et al., 2024) and Chapter-Llama (Ventura et al., 2025). We also include closed-source LLMs, GPT-4o (OpenAI, 2024), Gemini-2.5-Pro and Gemini-2.0-Flash (Gemini

Team et al., 2023), as reference systems. All trainable baselines are adapted to the same input setting and evaluated with the same references whenever applicable.

Implementation. For state-of-the-art comparison, we evaluate both Qwen2.5-7B (Yang et al., 2025) and LLaMA-3.1-8B (Grattafiori et al., 2024) on AVLecture and VidChapters-7M. For controlled ablations, we use Qwen2.5-3B as the generation backbone and keep it fixed across all variants. All trainable LLMs are adapted with LoRA, where we set the rank and scaling factor to $( r , \alpha ) = ( 3 2 , 6 4 )$ for Qwen2.5-7B and $( r , \alpha ) = ( 8 , 1 6 )$ for Qwen2.5- 3B. Models are trained with AdamW using a learning rate of $5 \times 1 0 ^ { - 5 }$ , batch size 1, and maximum input length 8192. For prompting, all methods use the same segment-level instruction template, which includes the segment ASR, visual summary, timestamp prompt, and optional retrieved contexts. For CSSE, we construct candidate contexts from temporal neighbors and semantic retrieval, and select the top- $K = 5$ segments. For LCDS, we set the window size to $W = 5$ for interventional dependency modeling. We use deterministic decoding with temperature 0 for intervention scoring to ensure that output changes are attributable to context removal, and use the same decoding setting across all compared variants. Experiments are conducted on a Debian GNU/Linux 12 server equipped with a single NVIDIA H800 PCIe GPU with 80 GB memory. Additional efficiency and scalability analyses are reported in Appendix C.5.

## 4.2 Main Result

We compare CausalChapter with dense video captioning methods, long-video LLM baselines, video chaptering models, and closed-source LLMs under zero-shot prompting. Tab. 1 reports results on the augmented AVLecture and VidChapters-7M benchmarks.

Results on AVLecture. CausalChapter performs best overall among trainable models, substantially outperforming the DVC baselines PDVC (Wang et al., 2021) and Vid2Seq (Yang et al., 2023b). Compared with the strongest fine-tuned chaptering baseline, Chapter-Llama (Ventura et al., 2025), it improves CIDEr from 99.78 to 110.12 with LLaMA-3.1-8B and from 95.36 to 116.88 with Qwen2.5-7B; F1@30 and tIoU also increase from 63.93% and 62.18% to 72.97% and 69.95%, respectively. These gains demonstrate improvements in

<table><tr><td>Type</td><td>Method</td><td>CIDEr</td><td>SODA_c</td><td>F1@30</td><td>tIoU</td></tr><tr><td>DVC</td><td>PDVC Vid2Seq1 T5</td><td>6.11 61.22</td><td>9.79</td><td>9.07 9.84</td><td>56.44 53.77</td></tr><tr><td>Chap.</td><td>VTimeLLMVicuna, * Chapter-LlamaLLaMA  $\mathrm { { C h a p t e r - L l a m a } ^ { Q w e n } }$ </td><td>0.02 99.78 95.36</td><td>3.04 8.15 6.91</td><td>47.53 63.93 57.73</td><td>46.72 62.18 59.62</td></tr><tr><td></td><td>CausalChapterLLaMA CausalChapterQwen</td><td>110.12 116.88</td><td>11.73 13.80</td><td>72.97 72.97</td><td>69.95 69.95</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>GPT-40</td><td>1.82</td><td>0.08</td><td>24.89</td><td>35.23</td></tr><tr><td>LLM</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Gemini-2.5-Pro</td><td>2.40</td><td>1.64</td><td>26.72</td><td>36.12</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Gemini-2.0-Flash</td><td>0.28</td><td>0.18</td><td>11.71</td><td>13.54</td></tr></table>

(a) AVLecture
<table><tr><td>Type</td><td>Method</td><td>CIDEr</td><td>SODA</td><td>F1</td><td>tIoU</td></tr><tr><td>DVC</td><td>Vid2Seq1 T5</td><td>55.8</td><td>11.6</td><td>26.7</td><td>58.6</td></tr><tr><td rowspan="3">Chap.</td><td>Chapter-LlamaLLaMA</td><td>100.9</td><td>19.3</td><td>45.3</td><td>71.8</td></tr><tr><td>CausalChapterLLaMA</td><td>101.5</td><td>18.5</td><td>46.2</td><td>72.0</td></tr><tr><td> $\mathrm { { C a u s a l C h a p t e r } ^ { Q w e n } }$ </td><td>109.2</td><td>18.8</td><td>46.2</td><td>72.0</td></tr><tr><td rowspan="2">LLM</td><td>GPT-40</td><td>51.0</td><td>8.1</td><td>37.6</td><td>68.0</td></tr><tr><td>Gemini-2.0-Flash</td><td>69.7</td><td>11.4</td><td>40.2</td><td>69.3</td></tr></table>

(b) VidChapters-7M

Table 1: Comparison of video chaptering methods on AVLecture and VidChapters-7M. For CausalChapter, localization scores come from the shared boundarylocalization stage, while generation scores use the indicated backbone. Superscripts denote backbones: <sup>T5</sup> T5, <sup>Vicuna</sup> Vicuna-7B, <sup>LLaMA</sup> LLaMA-3.1-8B, and <sup>Qwen</sup> Qwen2.5-7B. Methods without superscripts are proprietary or do not report a backbone; <sup>\*</sup> marks VTimeLLM not fine-tuned on AVLecture.

both description quality and boundary localization. Closed-source LLMs under zero-shot prompting obtain lower automatic scores; Appendix C.3 provides a complementary semantic evaluation, and Appendix C.6 provides a qualitative comparison with Chapter-Llama.

Results on VidChapters-7M. Tab. 1(b) shows that our method also generalizes to VidChapters-7M, which contains more diverse open-domain videos with user-annotated chapter boundaries and titles. Despite using a compact backbone, CausalChapter achieves competitive or better performance than long-video LLM baselines on both generation and localization metrics. This suggests that the proposed dependency-enhanced segmentthen-caption framework is not specific to lecture videos and can transfer to broader long-video chaptering scenarios.

## 4.3 Ablation Studies

We ablate CausalChapter on AVLecture using the same Qwen2.5-3B backbone and training data.

Backbone is the base segment-then-caption model; +LCDS and +CSSE add the corresponding localization and generation modules, while Full further includes temporal-aware training.

As shown in Table 2, LCDS improves F1, BS@30, and tIoU from 52.95%, 59.61%, and 66.36% to 56.72%, 63.02%, and 69.95%, respectively. CSSE leaves boundary predictions unchanged while increasing CIDEr from 88.39 to 98.40 and SODA\_c from 10.84 to 12.63. Their complementary effects yield the full model’s best CIDEr, METEOR, and SODA\_c scores of 104.59, 17.87, and 12.73.

## 4.4 Analysis of Interventional Dependency Modeling

We further analyze whether intervention-defined dependency provides more useful signals than simpler relevance-based alternatives. Table 3 compares LCDS with representation-similarity drops (Sim. Drop) and a contrastive objective (CL Loss). Similarity drops improve several threshold-based metrics but reduce tIoU, while contrastive learning gives only marginal gains. Our dependency score performs best on all metrics, increasing F1 from 52.95% to 56.72% and tIoU from 66.36% to 69.95%, supporting predictive dependency as a stronger cue for smooth boundaries.

Table 4 compares CSSE with temporal proximity (Previous-K), semantic retrieval (Sim. Top-K), and LLM-estimated relevance using the same backbone and boundaries. CSSE performs best on all generation metrics, improving CIDEr from 88.39 to 104.59 and SODA\_c from 10.84 to 12.73, showing the advantage of measuring a context’s influence on current generation.

Window-size sensitivity. We further analyze the effect of the LCDS window size W. As shown in Figure 3, W = 5 provides the most balanced performance across boundary-oriented metrics, improving F1, BS@30, and tIoU over the backbone while maintaining competitive F1@30. This suggests that a moderate local window captures sufficient cross-boundary dependency changes without introducing excessive neighboring noise.

Top-K sensitivity. We further analyze the effect of the number of supportive contexts selected by CSSE. As shown in Figure 4, increasing Top-K generally improves generation quality over the backbone, indicating that cross-segment supportive contexts provide useful complementary information for chapter description generation. The setting K = 5 achieves the best METEOR score and a strong CIDEr score, improving CIDEr from 88.39 to 104.59 and METEOR from 16.83 to 17.87. Therefore, we use K = 5 as a balanced setting in our main experiments.

<table><tr><td>Method</td><td>CIDEr</td><td>METEOR</td><td>SODA_c</td><td>F1</td><td>BS@30</td><td>tIoU</td></tr><tr><td>Backbone</td><td>88.39</td><td>16.83</td><td>10.84</td><td>52.95</td><td>59.61</td><td>66.36</td></tr><tr><td>+ LCDS</td><td>91.36</td><td>17.37</td><td>11.90</td><td>56.72</td><td>63.02</td><td>69.95</td></tr><tr><td>+ CSSE</td><td>98.40</td><td>17.15</td><td>12.63</td><td>52.95</td><td>59.61</td><td>66.36</td></tr><tr><td>Full</td><td>104.59</td><td>17.87</td><td>12.73</td><td>56.72</td><td>63.02</td><td>69.95</td></tr></table>

Table 2: Ablation studies on AVLecture. All variants use predicted boundaries. LCDS denotes Local Causal Dependency Shift, CSSE denotes Cross-Segment Causal Support Estimation.
<table><tr><td>Method</td><td>F1</td><td>F1@30</td><td>BS@30</td><td>tIoU</td></tr><tr><td>Baseline</td><td>52.95</td><td>71.28</td><td>59.61</td><td>66.36</td></tr><tr><td>Sim. Drop</td><td>54.51</td><td>72.41</td><td>61.80</td><td>65.68</td></tr><tr><td>CL Loss</td><td>53.28</td><td>70.91</td><td>59.75</td><td>65.23</td></tr><tr><td>Ours</td><td>56.72</td><td>72.97</td><td>63.02</td><td>69.95</td></tr></table>

Table 3: Analysis of LCDS on AVLecture. All methods use the same Qwen2.5-3B backbone.

<table><tr><td>Method</td><td>CIDEr</td><td>METEOR</td><td>SODA_c</td></tr><tr><td>Baseline</td><td>88.39</td><td>16.83</td><td>10.84</td></tr><tr><td>Previous-K</td><td>90.85</td><td>16.70</td><td>11.32</td></tr><tr><td>Sim. Top-K</td><td>95.97</td><td>16.79</td><td>11.70</td></tr><tr><td>LLM Scoring</td><td>96.11</td><td>17.44</td><td>11.53</td></tr><tr><td>Ours</td><td>104.59</td><td>17.87</td><td>12.73</td></tr></table>

Table 4: Analysis of cross-segment context selection on AVLecture. All methods use Qwen2.5-3B as backbone.

## 5 Conclusion

We presented CausalChapter, an interventional dependency modeling framework for long-video chaptering. To address the boundary error propagation and cross-segment context fragmentation introduced by segment-level generation, CausalChapter estimates intervention-defined predictive dependencies as task-oriented support signals. Specifically, LCDS captures local dependency drops between adjacent temporal windows to provide auxiliary evidence for smooth chapter boundaries, while CSSE selects cross-segment contexts according to their intervention-defined support for current chapter generation. Experiments on AVLecture and VidChapters-7M show that CausalChapter improves temporal localization and description quality on AVLecture, while remaining competitive on VidChapters-7M. These results highlight intervention-defined dependency as a useful signal for scalable and coherent long-video chaptering.

![](images/b0ab0a0108ad3b606f6e6832dcf5a952e7910f0ba59b0c9f6b54d194fe1943a5.jpg)

Figure 3: Window-size sensitivity of LCDS on AVLecture.  
![](images/309300012ee4455f50631e98f86499b965fa400483723d67416aa665f6326bd4.jpg)  
Figure 4: Top-K sensitivity of CSSE on AVLecture.

## Limitations

CausalChapter has several limitations. First, intervention-defined dependency should be interpreted as prediction-level influence rather than realworld causal discovery. Our goal is not to recover causal relations among video events or chapters, but to identify semantic units or context segments that affect boundary prediction and chapter-level generation under controlled masking or removal interventions. Second, our augmented AVLecture benchmark uses human-verified LLM-assisted chapter descriptions, which may inherit some stylistic regularities from the annotation pipeline. To reduce this effect, all compared methods are trained and evaluated with the same augmented references, and our annotation-sensitivity analysis examines alternative annotation models and styles. Third, CSSE introduces additional inference cost because it estimates context support through removal interventions. We control this cost by applying interventions only to a compact candidate set constructed from temporal neighbors and semantic retrieval, rather than to all segments or tokens. Further acceleration of support estimation is an interesting direction for future work.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (62437001, 62506040 and 62402051) and the Fundamental Research Funds for the Central Universities (2253500001).

## References

Tieyuan Chen, Huabin Liu, Yi Wang, Yihang Chen, Tianyao He, Chaofan Gan, Huanyu He, and Weiyao Lin. 2026. Mecd+: Unlocking event-level causal graph discovery for video reasoning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(3):2628–2645.

Soichiro Fujita, Tsutomu Hirao, Hidetaka Kamigaito, Manabu Okumura, and Masaaki Nagata. 2020. Soda: Story oriented dense video captioning evaluation framework. In European Conference on Computer Vision, pages 517–531. Springer.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Anchit Gupta, CV Jawahar, Makarand Tapaswi, et al. 2023. Unsupervised audio-visual lecture segmentation. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5232–5241.

Bin Huang, Xin Wang, Hong Chen, Zihan Song, and Wenwu Zhu. 2024. Vtimellm: Empower llm to grasp video moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14271–14280.

Vladimir Iashin and Esa Rahtu. 2020a. A better use of audio-visual cues: Dense video captioning with bi-modal transformer. arXiv preprint arXiv:2005.08271.

Vladimir Iashin and Esa Rahtu. 2020b. Multi-modal dense video captioning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 958–959.

Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, and Gedas Bertasius.

2024. Video recap: Recursive captioning of hourlong videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18198–18208.

Minkuk Kim, Hyeon Bae Kim, Jinyoung Moon, Jinwoo Choi, and Seong Tae Kim. 2024. Do you remember? dense video captioning with cross-modal memory retrieval. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13894–13904.

Minkuk Kim, Hyeon Bae Kim, Jinyoung Moon, Jinwoo Choi, and Seong Tae Kim. 2025. Hicm<sup>2</sup>: Hierarchical compact memory modeling for dense video captioning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 4293– 4301.

Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. 2017. Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision, pages 706–715.

Guozhang Li, De Cheng, Xinpeng Ding, Nannan Wang, Xiaoyu Wang, and Xinbo Gao. 2023. Boosting weakly-supervised temporal action localization with text information. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10648–10657. IEEE.

Guozhang Li, Xinpeng Ding, De Cheng, Jie Li, Nannan Wang, and Xinbo Gao. 2025. Etc: Temporal boundary expand then clarify for weakly supervised video grounding with multimodal large language model. IEEE Transactions on Multimedia, 27:1772–1782.

Guozhang Li, Xinran Duan, Mei Wang, Lizhi Wang, and Hua Huang. 2026. Curvature-guided task synergy for skeleton based temporal action segmentation. In International Conference on Learning Representations, volume 2026, pages 74158–74176.

Yang Liu, Guanbin Li, and Liang Lin. 2023. Crossmodal causal relational reasoning for event-level visual question answering. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):11624–11641.

Zhiyue Liu, Xinru Zhang, and Jinyuan Liu. 2025. Taskspecific information decomposition for end-to-end dense video captioning. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16524– 16536.

Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and Ji-Rong Wen. 2021. Counterfactual vqa: A cause-effect look at language bias. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12700– 12710.

OpenAI. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR.

Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575.

Lucas Ventura, Antoine Yang, Cordelia Schmid, and Gül Varol. 2025. Chapter-llama: Efficient chaptering in hour-long videos with llms. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18947–18958.

Teng Wang, Ruimao Zhang, Zhichao Lu, Feng Zheng, Ran Cheng, and Ping Luo. 2021. End-to-end dense video captioning with parallel decoding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6847–6857.

Kangyi Wu, Pengna Li, Jingwen Fu, Yizhe Li, Yang Wu, Yuhan Liu, Jinjun Wang, and Sanping Zhou. 2025. Event-equalized dense video captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8417–8427.

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. 2021. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786.

Zhuyang Xie, Yan Yang, Yankai Yu, Jie Wang, Yongquan Jiang, and Xiao Wu. 2025. Exploring temporal event cues for dense video captioning in cyclic co-learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 8771–8779.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, and Cordelia Schmid. 2023a. Vidchapters-7m: Video chapters at scale. Advances in Neural Information Processing Systems, 36:49428–49444.

Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and Cordelia Schmid. 2023b. Vid2seq: Largescale pretraining of a visual language model for dense video captioning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10714–10726.

Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. 2023. Self-chained image-language model for video localization and question answering. Advances in Neural Information Processing Systems, 36:76749–76771.

Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, and Mohit Bansal. 2023. Hierarchical video-moment retrieval and stepcaptioning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23056–23065.

## A Method Details

## A.1 Segmentation Head Architecture

The main text abstracts the boundary predictor as $g _ { \mathrm { b d } } ( z _ { i } )$ . In practice, our segmentation head follows the general design of recent multimodal video topic segmentation models: it uses sentence-aligned clips as basic units, projects visual and textual clip features into a shared space, performs middlefusion across modalities, and then predicts whether each unit is a topic boundary with a lightweight binary classifier.

Concretely, for the i-th semantic unit, we first obtain a visual clip representation and a text representation from sampled video frames and the corresponding ASR sentence, where $E _ { v }$ and $E _ { t }$ denote the visual and textual encoders, respectively. After projection into the same hidden dimension by the trainable projection matrices $W _ { v }$ and $W _ { t }$ , the two modalities are fused by a small stack of multimodal fusion layers, denoted by MFL, to produce updated visual and textual states. We then concatenate the fused modality-specific states into the multimodal unit representation $z _ { i }$ , which is used by the segmentation head:

$$
v _ { i } = W _ { v } E _ { v } ( c _ { i } ^ { v } ) , \qquad t _ { i } = W _ { t } E _ { t } ( c _ { i } ^ { t } ) ,\tag{12}
$$

$$
\begin{array} { r } { h _ { i } ^ { v } , h _ { i } ^ { t } = \mathrm { M F L } ( v _ { i } , t _ { i } ) , \qquad z _ { i } = [ h _ { i } ^ { v } ; h _ { i } ^ { t } ] , } \end{array}\tag{13}
$$

$$
o _ { i } ^ { \mathrm { b a s e } } = g _ { \mathrm { b d } } ( z _ { i } ) = W _ { p } z _ { i } + b _ { p } .\tag{14}
$$

Here, $W _ { p }$ and $b _ { p }$ denote the predictor weight matrix and bias term of the final binary classifier. This design keeps the segmentation head lightweight while still allowing cross-modal interaction before boundary prediction. Relative to a late-fusion design, the middle-fusion structure better exposes cross-modal cues such as transcript transitions, slide changes, and visual context shifts to the boundary classifier. In our implementation, the segmentation head is therefore best viewed as a multimodal fusion block followed by a linear classifier that outputs the boundary logits in Eq. 1.

## B Dataset Construction and Annotation

## B.1 AVLecture Benchmark

AVLecture is a long-form instructional-video benchmark introduced to audio-visual lecture segmentation. The full AVLecture collection contains 86 courses with over 2,350 lectures and a total duration of roughly 2,200 hours, covering a broad range of STEM subjects. Each course provides video lectures together with aligned ASR transcripts and OCR signals, and many courses also include auxiliary educational resources such as lecture notes, slides, and assignments. Among the 86 courses, a 15-course subset with 350 lectures is annotated with temporal segmentation boundaries and serves as the standard benchmark for lecture segmentation. In our work, we adopt this segmented subset as the boundary-localization foundation, and use the 245/35/70 train/validation/test split described in the main text. We further extend these lectures with chapter-level reference descriptions so that the same benchmark can support evaluation of both chapter localization and chapter description generation.

LLM-Assisted Annotation Pipeline. For each video, we use GPT-4o (OpenAI, 2024) as the annotation model. The input includes the full video transcript, frame-level captions extracted every 10 seconds, and the start and end timestamps of all ground-truth segments. The model is instructed to output a JSON object, where each segment is paired with a concise chapter-level description. Each description is expected to summarize the main topic of the corresponding segment, remain faithful to the transcript and frame captions, and distinguish the segment from adjacent chapters. When the transcript and frame captions exceed the input budget, we truncate the input while preserving the segment timestamps and the local transcript/caption context around each target segment.

Human Verification and Revision. After GPT-4o-assisted annotation, all segment descriptions are manually checked and revised. We correct descriptions that are overly generic, unsupported by the transcript or frame captions, inconsistent with the segment boundary, or redundant with adjacent chapters. For segments whose content is distributed across multiple stages of the lecture, we manually improve or combine the generated descriptions to better reflect the complete chapter-level semantics. We also normalize the JSON format, description length, and writing style across videos. The final references are fixed before training and evaluation, and all compared methods use the same augmented references, ensuring that the evaluation remains internally consistent.

The verification was conducted by 10 annotators. Each video and all of its chapter descriptions were reviewed by a group of three annotators. When one annotator revised a description, the revised label was circulated to the other two annotators and further refined until all three accepted the final version. We therefore record revision history and final consensus rather than a chance-corrected agreement coefficient. Among the 350 videos, 308 videos (88.0%) required no revision, 39 videos (11.1%) required one revision round, and 3 videos (0.9%) required two revision rounds. All final labels were accepted by the assigned annotator group.

Table 5 gives representative revision examples, illustrating how human annotators remove crossboundary content and make chapter descriptions more specific to the current segment.

Annotation Sensitivity. To examine whether model comparisons depend on a particular annotation model or writing style, we construct four AVLecture reference sets by varying the annotation model and description format: GPT-4o phrase-style references, GPT-4o sentence-style references, Claude phrase-style references, and Claude sentence-style references. For each reference set, models are retrained and evaluated with the same split, model configuration, training procedure, and evaluation code; only the training and evaluation references are changed. Table 6 shows that CausalChapter improves over Chapter-Llama across all four reference settings, suggesting that the gains are not tied to a single annotation style.

## C Additional Experimental Results

## C.1 Evaluation Metrics

For completeness, we briefly summarize the metrics used in the main paper and the appendix tables. CIDEr evaluates how well a generated chapter description matches the reference description using consensus-based n-gram similarity, with higher scores indicating better agreement with human-written references. METEOR measures generation quality through unigram alignment with stemming and synonym matching. SODA\_c jointly evaluates temporal localization and description quality through temporally ordered matching; on AVLecture, we use SODA type-c with IoUweighted METEOR matching, as specified in the main text.

For boundary localization, F1 is the harmonic mean of precision and recall, and evaluates the overall quality of predicted topic boundaries according to how well they match ground-truth boundaries. BS@30 (Boundaries at 30 seconds) measures whether a predicted boundary falls within a 30-second tolerance window around a ground-truth boundary, thus reflecting boundary-detection accuracy under a fixed temporal tolerance. tIoU reports the temporal intersection-over-union between predicted and ground-truth segments, measuring how accurately the predicted chapter partition overlaps with the reference segmentation. F1@30 applies the F1 criterion under the same 30-second matching window, and therefore captures both boundary accuracy and temporal tolerance.

## C.2 Backbone Scaling and Complete Baseline Results

Overview. The main experiments use different backbone sizes for different purposes: we use larger backbones for state-of-the-art comparison, and a smaller Qwen2.5-3B backbone for ablation studies to control experimental cost while keeping variants comparable. This appendix reports the complete results behind these choices. We first present backbone scaling results for CausalChapter across the LLaMA and Qwen2.5 families, and then provide the full LLM-based baseline and closed-source LLM reference results on AVLecture. These supplementary results verify that the gains of CausalChapter are not tied to a single model family or scale.

Table 7 shows two consistent trends. First, stronger backbones generally lead to better chapterlevel generation quality, especially on CIDEr. Second, temporal-aware optimization improves most corresponding settings across both LLaMA and Qwen2.5 families. These results suggest that the proposed framework benefits from model scaling, while the temporal-aware optimization provides additional gains beyond simply increasing backbone size.

Table 8 provides the complete baseline results on AVLecture. Fine-tuned Chapter-Llama variants are much stronger than their zero-shot counterparts, indicating that task adaptation is important for longvideo chaptering. Closed-source LLMs under zeroshot prompting achieve limited performance, especially on generation metrics, suggesting that simply prompting general-purpose LLMs is insufficient for this benchmark. These observations support the need for task-specific modeling of boundary localization and cross-segment context dependency.

<table><tr><td>Issue</td><td>LLM draft</td><td>Human-revised label</td><td>Reason</td></tr><tr><td>Cross-boundary content</td><td>Force range dependence on mediator mass via decay laws</td><td>Force range dependence on mediator mass</td><td>“Decay laws&quot; belongs to the following chapter.</td></tr><tr><td>Overly general description</td><td>Conservation laws in elastic collisions</td><td>Linearizing one-dimensional elastic collisions via relative velocity</td><td>The draft omitted the specific derivation in the current chapter.</td></tr></table>

Table 5: Representative human revisions in the augmented AVLecture benchmark.
<table><tr><td>Method</td><td>R1 C</td><td>R1 S</td><td>R2 C</td><td>R2 S</td><td>R3 C</td><td>R3 S</td><td>R4 C</td><td>R4 S</td><td>Avg. C</td><td>Avg. S</td></tr><tr><td>Chapter-Llama LLaMA</td><td>99.78</td><td>8.15</td><td>101.26</td><td>8.41</td><td>107.25</td><td>10.32</td><td>104.74</td><td>10.13</td><td>103.26</td><td>9.25</td></tr><tr><td> $\mathrm { C a u s a l C h a p t e r ^ { L L a M A } }$ </td><td>110.12</td><td>11.73</td><td>116.12</td><td>14.13</td><td>115.63</td><td>12.67</td><td>128.19</td><td>14.34</td><td>117.52</td><td>13.22</td></tr><tr><td> $\mathrm { { C a u s a l C h a p t e r } ^ { Q w e n } }$ </td><td>116.88</td><td>13.80</td><td>138.99</td><td>15.21</td><td>121.74</td><td>14.63</td><td>145.24</td><td>15.83</td><td>130.71</td><td>14.87</td></tr></table>

Table 6: Annotation-sensitivity results on AVLecture. R1/R2 use GPT-4o references in phrase/sentence styles, and R3/R4 use Claude references in phrase/sentence styles. C and S denote CIDEr and SODA\_c, respectively.

<table><tr><td>Backbone</td><td>temporal-aware</td><td>CIDEr</td><td>METEOR</td></tr><tr><td>LLaMA-3.2-1B</td><td>w/o</td><td>78.58</td><td>10.58</td></tr><tr><td>LLaMA-3.2-1B</td><td>with</td><td>83.16</td><td>13.05</td></tr><tr><td>LLaMA-3.2-3B</td><td>w/o</td><td>93.40</td><td>17.15</td></tr><tr><td>LLaMA-3.2-3B</td><td>with</td><td>98.45</td><td>17.45</td></tr><tr><td>LLaMA-3.1-8B</td><td>w/o</td><td>98.38</td><td>19.08</td></tr><tr><td>LLaMA-3.1-8B</td><td>with</td><td>110.12</td><td>19.41</td></tr><tr><td>Qwen2.5-0.5B</td><td>w/o</td><td>84.56</td><td>10.38</td></tr><tr><td>Qwen2.5-0.5B</td><td>with</td><td>86.48</td><td>13.41</td></tr><tr><td>Qwen2.5-1.5B</td><td>w/o</td><td>90.57</td><td>16.19</td></tr><tr><td>Qwen2.5-1.5B</td><td>with</td><td>97.94</td><td>16.33</td></tr><tr><td>Qwen2.5-3B</td><td>w/o</td><td>97.64</td><td>17.73</td></tr><tr><td>Qwen2.5-3B</td><td>with</td><td>104.59</td><td>17.87</td></tr><tr><td>Qwen2.5-7B</td><td>w/o</td><td>113.98</td><td>18.24</td></tr><tr><td>Qwen2.5-7B</td><td>with</td><td>116.88</td><td>20.43</td></tr></table>

Table 7: Backbone scaling results of CausalChapter on AVLecture. Temporal-aware optimization further injects timestamp information and segmentation-stage temporal features into the generator.

## C.3 Closed-Source LLM Semantic Evaluation

Lexical metrics can underestimate zero-shot closedsource LLMs when their outputs are semantically reasonable but use a different wording or granularity from the reference descriptions. We therefore complement CIDEr with a semantic-similarity evaluation. GPT-4o is given a generated chapter title and the corresponding reference title, and assigns a 0–100 semantic-similarity score based on topic match and specificity. Table 9 reports the averaged scores. The semantic gap is smaller than the CIDEr gap, but fine-tuned chaptering models still achieve stronger semantic alignment with the AVLecture references.

## C.4 CSSE Output-Difference Sensitivity

The main method instantiates $D _ { \mathrm { o u t } }$ as the teacherforced token-level KL divergence between the fullcontext and leave-one-out predictive distributions. To assess whether CSSE depends on this particular distance function, we compare it with LLM-based relevance scoring, embedding distance, and tokenoverlap distance under the same Qwen2.5-3B setting. As shown in Table 10, all variants improve over removing CSSE, while KL divergence performs best overall.

## C.5 Inference Efficiency and Scalability

## Profiling protocol and end-to-end comparison.

We profile Chapter-Llama and CausalChapter on the same 10 AVLecture test videos using an NVIDIA A100 80GB GPU, BF16 precision, identical decoding settings, and the same warm-up procedure. The videos contain 4.3 predicted chapters on average. Chapter-Llama uses its original whole-video inference pipeline and jointly predicts boundaries and descriptions, so its two stages cannot be timed separately. CausalChapter first predicts boundaries and then applies intravideo micro-batching to chapter-level generation and leave-one-out scoring requests; the video-level batch size remains one. Table 11(a) reports the matched end-to-end comparison using LLaMA-3.1- 8B for both methods and a micro-batch size of 8 for CausalChapter.

Intra-video micro-batching. Once boundaries are fixed, requests from different chapters can be executed in parallel. We apply intra-video microbatching to Pass 1 generation, full-context reference generation, leave-one-out KL scoring, and Pass 2 generation. Table 11(b) reports the complete execution breakdown. Gen. and KL denote the average numbers of sequential batched model invocations per video after micro-batching; because they are averaged over videos, they need not be integers.

<table><tr><td>Method</td><td>Backbone</td><td>Type</td><td>CIDEr</td><td>METEOR</td><td>F1</td><td>F1@30</td><td>BS@30</td><td>tIoU</td></tr><tr><td>VTimeLLM</td><td>ChatGLM3-6B</td><td>ZS</td><td>0.12</td><td>1.79</td><td>32.46</td><td>47.66</td><td>45.78</td><td>48.21</td></tr><tr><td>VTimeLLM</td><td>Vicuna-7B</td><td>ZS</td><td>0.02</td><td>2.04</td><td>32.25</td><td>47.53</td><td>50.97</td><td>46.72</td></tr><tr><td>VTimeLLM</td><td>Vicuna-7B</td><td>FT</td><td></td><td>0.59</td><td>9.22</td><td>20.71</td><td>37.38</td><td>27.88</td></tr><tr><td>Chapter-Llama</td><td>LLaMA-3.2-1B</td><td>ZS</td><td>14.77</td><td>8.36</td><td>3.51</td><td>8.19</td><td>8.19</td><td>19.14</td></tr><tr><td>Chapter-Llama</td><td>LLaMA-3.2-1B</td><td>FT</td><td>68.53</td><td>15.67</td><td>48.18</td><td>63.36</td><td>63.36</td><td>59.30</td></tr><tr><td>Chapter-Llama</td><td>LLaMA-3.2-3B</td><td>ZS</td><td>38.77</td><td>12.79</td><td>4.39</td><td>9.12</td><td>9.12</td><td>23.58</td></tr><tr><td>Chapter-Llama</td><td>LLaMA-3.2-3B</td><td>FT</td><td>86.07</td><td>19.40</td><td>47.53</td><td>65.11</td><td>65.11</td><td>60.64</td></tr><tr><td>Chapter-Llama</td><td>LLaMA-3.1-8B</td><td>ZS</td><td>77.32</td><td>13.93</td><td>20.24</td><td>40.40</td><td>40.40</td><td>41.72</td></tr><tr><td>Chapter-Llama</td><td>LLaMA-3.1-8B</td><td>FT</td><td>99.78</td><td>19.04</td><td>47.56</td><td>63.93</td><td>63.93</td><td>62.18</td></tr><tr><td>GPT-40</td><td></td><td>ZS</td><td>1.82</td><td>2.34</td><td>24.32</td><td>24.89</td><td>26.01</td><td>35.23</td></tr><tr><td>GPT-4o-mini</td><td></td><td>ZS</td><td>2.80</td><td>2.95</td><td>15.88</td><td>21.50</td><td>34.27</td><td>31.21</td></tr><tr><td>Gemini-2.5-Pro</td><td></td><td>ZS</td><td>2.40</td><td>10.03</td><td>24.32</td><td>26.72</td><td>26.01</td><td>36.12</td></tr><tr><td>Gemini-2.0-Flash</td><td></td><td>ZS</td><td>0.28</td><td>3.07</td><td>9.64</td><td>11.71</td><td>27.99</td><td>13.54</td></tr><tr><td>Claude-3.5-Sonnet</td><td></td><td>ZS</td><td></td><td>5.44</td><td>16.49</td><td>24.18</td><td>38.14</td><td>31.75</td></tr><tr><td>Claude-Sonnet-4</td><td></td><td>ZS</td><td>0.02</td><td>10.93</td><td>18.08</td><td>25.36</td><td>34.35</td><td>38.48</td></tr></table>

Table 8: Complete long-video LLM baseline and closed-source LLM reference results on AVLecture. ZS denotes zero-shot prompting and FT denotes fine-tuning.

<table><tr><td>Method</td><td>Semantic similarity</td></tr><tr><td>CausalChapter</td><td>82.10</td></tr><tr><td>Chapter-Llama</td><td>79.80</td></tr><tr><td>Gemini-2.5-Pro</td><td>60.40</td></tr></table>

Table 9: Semantic-similarity evaluation for chapter descriptions on AVLecture.

<table><tr><td> $D _ { \mathrm { o u t } }$  implementation</td><td>CIDEr</td><td>SODA_c</td></tr><tr><td>Without CSSE</td><td>88.39</td><td>10.84</td></tr><tr><td>LLM-based scoring</td><td>99.27</td><td>11.68</td></tr><tr><td>Embedding distance</td><td>100.20</td><td>11.79</td></tr><tr><td>Token-overlap distance</td><td>101.93</td><td>12.02</td></tr><tr><td>KL divergence</td><td>104.59</td><td>12.73</td></tr></table>

Table 10: Sensitivity of CSSE to different outputdifference functions on AVLecture.

At micro-batch size 1, the 11.90 generation calls in Table 11(b) comprise 4.30 Pass 1 calls, 3.30 short CSSE reference-generation calls, and 4.30 Pass 2 calls; each reference-generation call produces at most 16 tokens. Increasing the microbatch size to 8 reduces the sequential generation calls from 11.90 to 5.30 and KL calls from 11.20 to 6.60. CSSE time remains approximately 4.6 seconds per video and accounts for 4.59/10.50 (43.7%) of the complete pipeline at size 8. Overall latency decreases from 12.20 to 10.50 seconds per video, while peak memory increases from 16.59 to 19.35 GiB.

(a) End-to-end comparison
<table><tr><td>Method</td><td>Bound.</td><td>Gen.</td><td>Total</td><td>1 Mem.</td></tr><tr><td>Chapter-Llama</td><td></td><td>11.61</td><td>11.61</td><td>19.56</td></tr><tr><td>CausalChapter</td><td>0.19</td><td>10.31</td><td>10.50</td><td>19.35</td></tr></table>

(b) Intra-video micro-batching
<table><tr><td>Batch</td><td>Gen.</td><td>KL</td><td>CSSE</td><td>Total</td><td>Mem.</td></tr><tr><td>1</td><td>11.90</td><td>11.20</td><td>4.66</td><td>12.20</td><td>16.59</td></tr><tr><td>2</td><td>8.10</td><td>8.20</td><td>4.63</td><td>11.10</td><td>17.05</td></tr><tr><td>4</td><td>5.90</td><td>6.90</td><td>4.60</td><td>10.84</td><td>19.35</td></tr><tr><td>8</td><td>5.30</td><td>6.60</td><td>4.59</td><td>10.50</td><td>19.35</td></tr></table>

Table 11: BF16 inference on AVLecture. Times are seconds per video, memory is peak GiB, and Gen./KL are sequential batched calls per video. CausalChapter generation in (a) includes Pass 1, CSSE, and Pass 2.

Candidate-set scaling. Table 12(a) evaluates the cost of expanding the pre-scoring candidate limit. The average number of available candidates saturates at 1.84 per chapter on the profiled videos. Consequently, increasing the limit from 1 to 10 changes the average number of KL calls only from 6.60 to 6.90 per video, while CSSE time increases from 4.43 to 5.31 seconds per video. This indicates that the practical intervention cost is governed by the compact set of available historical candidates rather than by the nominal limit alone.

Top-K after intervention scoring. Top-K is applied only after all candidates have been scored and therefore does not create additional intervention calls. Table 12(b) reports its measured context count, time, and memory. Generation quality is reported once, in Figure 4 in the main paper.

(a) Candidate-set limit
<table><tr><td></td><td>Limit Actual/ch.</td><td>KL</td><td>CSSE</td><td>Total Mem.</td></tr><tr><td>1</td><td>0.77</td><td>6.60</td><td>4.43 9.93</td><td>19.37</td></tr><tr><td>3</td><td>1.60</td><td>6.60</td><td>5.24 10.82</td><td>19.36</td></tr><tr><td>5</td><td>1.81</td><td>6.90</td><td>5.51 10.96</td><td>19.36</td></tr><tr><td>10</td><td>1.84</td><td>6.90</td><td>5.31 10.71</td><td>19.56</td></tr></table>

(b) Top-K after scoring
<table><tr><td>Top-K</td><td>Selected/ch.</td><td>Total</td><td>Mem.</td></tr><tr><td>1</td><td>0.77</td><td>10.71</td><td>19.56</td></tr><tr><td>3</td><td>1.60</td><td>10.98</td><td>19.50</td></tr><tr><td>5</td><td>1.81</td><td>10.84</td><td>19.35</td></tr><tr><td>7</td><td>1.84</td><td>10.79</td><td>19.38</td></tr><tr><td>9</td><td>1.84</td><td>10.43</td><td>19.58</td></tr></table>

Table 12: CSSE scaling on AVLecture. Times are seconds per video, memory is peak GiB, and KL denotes calls per video. In (a), Gen. calls remain 5.90 per video. Top-K in (b) is applied after scoring and adds no leaveone-out evaluations.

End-to-end time remains within 10.43–10.98 seconds per video and peak memory within 19.35– 19.58 GiB across the Top-K sweep. As Figure 4 shows, K = 5 gives the highest METEOR and a strong CIDEr score while using nearly all contexts available on average; we therefore use it as the balanced default.

Quantized inference. We additionally profile 4- bit NF4 inference for memory-constrained settings. Figure 5 reports the latency–memory trade-off; each point is annotated with its exact value, so we do not repeat the measurements in a separate table.

![](images/17b0884b07d107c01df1377a246ca7e3b2634deeaa1e12f3e7c183edc452f336.jpg)  
Figure 5: CausalChapter latency–memory trade-off under 4-bit NF4 quantization. Lower values are better; gold points mark the minimum in each panel.

At micro-batch size 1, quantized CausalChapter uses 8.28 GiB of peak memory, compared with 9.90 GiB for Chapter-Llama under the same NF4 setting. Increasing the size to 8 reduces latency from 47.77 to 19.41 seconds per video while using 13.80 GiB. These results provide an explicit latency–memory operating range rather than a single deployment point.

## C.6 Qualitative Analysis

Beyond quantitative results, we further analyze representative cases in Figure 6 to examine how CausalChapter behaves under smooth boundary transitions and cross-segment context selection.

In smooth-boundary cases, baseline methods often rely on local representation changes or textual similarity drops, and therefore tend to delay or miss boundaries when the topic gradually evolves. In contrast, CausalChapter captures a drop in intervention-defined predictive dependency between adjacent windows through LCDS, allowing it to identify structural changes more accurately. For chapter generation, similarity-based retrieval often selects contexts that are lexically close but largely repetitive. CSSE instead tends to select segments that provide definitions, background, experimental setup, or reasoning premises, leading to descriptions that are more complete and better aligned with the logical structure of the whole video.

The qualitative example is consistent with the quantitative results. LCDS provides a local dependency-shift signal for smooth boundary localization, while CSSE changes context selection from surface similarity matching to support estimation with respect to the current generation. These two behaviors help explain why CausalChapter improves both temporal localization and chapter-level generation.

## C.7 Prompt Design for Two-Pass Generation

We adopt a two-pass prompting strategy for lecture subheading generation. In both passes, the model receives the current segment transcript and generates one concise subheading. The system instruction remains in the non-truncated prefix, whereas the transcript occupies the truncatable middle. This preserves task instructions, visual tokens, examples, contextual titles, and generation markers under truncation. Table 13 summarizes the two passes.

![](images/290f741484d56e3e9ee14cb98cefd5c9925f3264c35d45e95da6095f75ff631b.jpg)  
(b)

Figure 6: Qualitative comparison on AVLecture.
<table><tr><td>Component</td><td>Prompt Content</td></tr><tr><td>System Prompt</td><td>You are a helpful assistant generating lecture subheadings. Given the video segment transcript, generate a concise and informative subheading that captures the main topic. Only output the subheading, nothing else.</td></tr><tr><td>Pass 1: Segment-Level Generation</td><td>Visual context: &lt;VIS_0&gt;&lt;VIS_1&gt;...&lt;VIS_N&gt;. Optional few-shot examples are provided in the format: Segment: [example transcript] Subheading: [example title]. Optional related subheadings from the same lecture are provided as contextual titles. The current segment transcript is then given as: Segment: [current segment transcript]. The model is required to output only the generated subheading.</td></tr><tr><td>Pass 2: Temporal Re- finement</td><td>Visual context: &lt;VIS_0&gt;&lt;VIS_1&gt;.. . &lt;VIS_N&gt;. Temporal and causal rules: every retrieved context is from a segment before the current segment. Use a retrieved context only if it helps clarify the topic transition or dependency. Ignore irrelevant retrieved contexts. Never copy a previous title as the current title. The final subheading must describe only the current segment. The Pass-1 generated title is provided as the initial draft. The model is asked to keep it unchanged if it accurately captures the main concept using correct technical terms. If the draft misses a critical technical term or the main concept is wrong, the model generates a better subheading while preserving all correct technical terms from the draft. Retrieved previous contexts are provided as previous titles with metadata such as temporal distance, source, and retrieval score. The current segment transcript is then given as: Current segment transcript: [current segment transcript]. The model is required to output only the final refined subheading.</td></tr></table>

Table 13: Prompt components used in the two-pass generation framework.

## D LLM Usage Statement

We utilized a large language model (LLM) to improve the grammar, clarity, and overall readability of this manuscript. The LLM’s role was strictly limited to language editing and polishing. All scientific contributions, including the core ideas, methodology, experimental design, data analysis, and conclusions, are the original work of the human authors. The use of the LLM did not alter the scientific content or its interpretation.