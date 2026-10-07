# EC-RAG: Event Chain Retrieval-Augmented Generation for Long Video Understanding

Yuhao Qin qinyuhaoswu@163.com Northwestern Polytechnical University Xi’an, China

Yuke Li liyuke@nwpu.edu.cn Northwestern Polytechnical University Xi’an, China

Junbo Wang jbwang@nwpu.edu.cn Northwestern Polytechnical University Xi’an, China

Yining Zhu yiningzhu@nwpu.edu.cn Northwestern Polytechnical University Xi’an, China

## Abstract

Current large video-language models (LVLMs) still face challenges when dealing with long videos, mainly because frames are often processed independently, making it dificult to capture temporal dependencies across events. Although retrieval-augmented approaches have been introduced to provide additional context, most of them operate at the frame or snippet level, which limits their ability to model how events evolve over time and relate to each other. In this paper, we propose Event Chain Retrieval-Augmented Generation (EC-RAG), a training-free framework that organizes video content into an explicit event chain before question answering. Instead of retrieving isolated frames or text segments, EC-RAG first partitions the video into semantically coherent segments, represents each segment using multi-modal signals, and then links them into a structured chain that preserves temporal order and captures interevent relationships. Given a query, the system identifies relevant events within this chain and gathers supporting evidence from the associated modalities. Our approach ofers several practical advantages: (i) event-level abstraction that better reflects how video content is naturally structured, enabling more reliable localization compared to frame-level retrieval; (ii) structured multi-modal fusion that aggregates speech, text, and visual cues at the event level, allowing complementary information to be more efectively utilized during reasoning; and (iii) plug-and-play compatibility with existing LVLM backbones, requiring no additional training or reliance on proprietary models. Experiments on Video-MME, MLVU, and LongVideoBench show that this event-centric design consistently outperforms frame-level retrieval baselines, highlighting the importance of modeling temporal structure for long-video understanding.

## Keywords

Long Video Understanding, Retrieval-Augmented Generation, Event Chain Reasoning

## 1 Introduction

With the rapid advancement of Large Video-Language Models (LVLMs), significant progress has been made in video understand ing tasks [2, 14, 17, 18, 37, 41]. However, when processing long videos spanning tens of minutes to hours, current LVLMs face critical challenges: the limited context window restricts the number of frames that can be processed, while simply increasing sampled frames leads to information redundancy and reasoning degradation.

![](images/072996392a36a004a9952ce4e38ee4ddc1a325cff081f63e48e718ce6bd754bb.jpg)  
Figure 1: Advantages of our EC-RAG. EC-RAG provides an event-centric, training-free pipeline with structured temporal understanding that is easily compatible with any LVLM.

Recent studies have sought to extend the context length of LVLMs for long video understanding, such as LongLLaVA [29] scales token capacity to ingest more frames. However, this bruteforce expansion proves brittle—Video-MME [6] benchmark shows accuracy drops when additional frames are supplied beyond a threshold. This outcome suggests that simply increasing the number of sampled frames not only leads to information redundancy but also imposes additional challenges for the model to handle complex reasoning. A parallel research direction employs Retrieval-Augmented Generation (RAG) to supplement video queries with externally retrieved information. Early eforts in this direction [16, 39] demonstrate that augmenting visual inputs with auxiliary textual information can efectively enhance video understanding, particularly for content that is dificult to capture through visual frames alone. These methods typically extract three types of auxiliary texts from videos: automatic speech recognition (ASR) for transcribing spoken dialogue and narration, optical character recognition (OCR) for capturing on-screen text such as titles and labels, and object detection (DET) for identifying visual entities and their spatial arrangements. Building upon this foundation, Video-RAG [20] proposes visually-aligned auxiliary texts from OCR, ASR, and DET, achieving notable improvements with minimal computational overhead. TV-RAG [1] further introduces temporal-window retrieval with entropy-based keyframe selection to handle semantic drift in long videos. More recently, AdaVideoRAG [35] employs adaptive dificulty routing and graph databases for complex multi-hop reasoning. Despite these advances, existing video RAG methods share a fundamental limitation: they retrieve at the frame level or text segment level, treating video as a collection of independent units rather than a coherent narrative of interconnected events. Long videos inherently contain sequences of causally connected events—a lecture progresses through topics, a movie unfolds through scenes, a cooking video follows recipe steps. Without modeling this event structure, current methods produce fragmented retrieval results that fail to support questions requiring temporal reasoning.

Based on this observation, we focus on the following question: Can we design a video RAG framework that explicitly captures temporal event dynamics and inter-event relationships for coherent long video understanding?

To this end, we propose Event Chain Retrieval-Augmented Generation (EC-RAG), a training-free framework that organizes video content into explicit event chains before question answering. Instead of retrieving isolated frames or text snippets, EC-RAG first partitions the video into semantically coherent temporal segments, aggregates multi-modal signals (ASR, OCR, DET) within each segment to form events, and links them into a structured chain that preserves temporal order. For multi-modal extraction, we introduce CLIP-Variance guided keyframe selection that combines querysemantic relevance with visual informativeness, ensuring selected frames are both relevant and information-rich. Given a user query, EC-RAG performs event-level localization to identify the most relevant events along with their temporal context, and gathers supporting evidence from associated modalities for answer generation.

We evaluate EC-RAG on Video-MME [6] , MLVU [42] , and LongVideoBench [33] benchmarks. By applying EC-RAG to five diferent open-source LVLMs, we achieve an average performance improvement of 4% on Video-MME. Furthermore, compared with existing frame-level retrieval methods such as Video-RAG and TV-RAG, EC-RAG consistently demonstrates superior performance, validating the efectiveness of our event-chain-based approach. Our contributions are summarized as follows:

• We propose EC-RAG, a novel framework that explicitly models temporal event chains for long video understanding. Unlike existing methods operating at frame or snippet level, EC-RAG captures inter-event relationships and enables coherent reasoning over extended temporal spans.

• We introduce CLIP-Variance guided keyframe selection that jointly considers semantic relevance and visual informativeness, improving the quality of multi-modal information extraction for event construction.

• We design a query-guided event localization mechanism that identifies relevant events within the temporal chain and aggregates multi-modal evidence, providing structured context for LVLM answer generation.

## 2 Related Work

## 2.1 Video Understanding with Large Language Models

The emergence of powerful LLMs has spurred eforts to develop comprehensive video-language systems. Early works like Video-ChatGPT [21] and VideoChat [12] adopt frame-wise feature extraction followed by temporal aggregation. Video-LLaVA [13] introduces a unified projection to bridge visual and textual modalities, while LLaVA-NeXT-Video [39] further refines this through videospecific adaptation. These models perform well on short videos but struggle with longer content due to frame sampling constraints.

To address this limitation, recent methods explore expanding the context window for extended sequences. LongVA [38] and Long-LLaVA [29] leverage pre-training on lengthy documents to develop transferable sequence modeling capabilities. INTP [26] reorganizes visual tokens to accommodate more information within fixed context limits. However, densely sampling frames quickly becomes computationally prohibitive, and the high redundancy in video data often limits the benefits of processing more frames.

## 2.2 LLM-based Agents for Video Understanding

An alternative paradigm employs LLMs as controllers to orchestrate specialized tools for video comprehension [7, 22, 30, 32, 36]. Representative methods such as VideoAgent [4] and MM-VID [15] enable dynamic interaction with video content through iterative retrieval and reasoning. While flexible, these approaches introduce substantial latency from multiple tool calls and often rely on proprietary APIs, limiting broader adoption in open-source settings.

## 2.3 Retrieval-Augmented Generation for Video

RAG-based methods retrieve relevant information from pre-constructed databases to augment video understanding. Video-RAG [20] extracts multimodal cues including speech transcripts and detected objects to build searchable knowledge bases. TV-RAG [1] enhances this with temporal-aware retrieval, while AdaVideoRAG [35] employs adaptive routing for queries of varying complexity. Current approaches, however, retrieve content at the frame or segment granularity without capturing the temporal progression of events, limiting performance on questions requiring understanding of how actions unfold across time.

## 3 Method

We propose EC-RAG, a training-free retrieval-augmented framework for large video-language models that organizes video content as structured event chains. As illustrated in Figure 2, our pipeline comprises four stages: (i) Query Decoupling: The user query is decomposed into modality-specific retrieval requests for ASR, OCR, and DET. (ii) CLIP-Variance Guided Multimodal Extraction: Informative keyframes are selected through joint CLIP similarity and variance weighting, followed by parallel extraction of speech, text, and object information. (iii) Temporal Event Chain Construction: The video is segmented into temporal units and organized as a chain of semantically coherent events with explicit inter-event relations. (iv) Evidence-augmented Multimodal Reasoning: Query-guided localization identifies relevant events, and the LVLM generates answers grounded in the event chain context and multimodal evidence.

![](images/71383c658e93b548294d96e755b9d80ed7188f77e9ecbd79b792704df7aa0171.jpg)  
Figure 2: Overview of the EC-RAG framework. The pipeline comprises four stages: (i) Query Decoupling parses the user question into modality-specific retrieval requests for ASR, OCR, and DET; (ii) CLIP-Variance Guided Multimodal Extraction selects informative keyframes through weighted scoring of visual similarity and frame variance, then extracts speech transcripts, on-screen text, and scene graphs via open-vocabulary object detection; (iii) Temporal Event Chain Construction segments the video into 30-second units and builds a chain of semantically coherent events, where each event integrates its associated multimodal evidence with explicit temporal relations; (iv) Evidence-augmented Multimodal Reasoning performs query-guided event localization to retrieve relevant events, then feeds the event chain summary, located events, and raw evidence to the LVLM for answer generation. By organizing video content as structured event sequences, EC-RAG captures temporal dynamics and inter-event dependencies for complex video understanding.

Problem Formulation. Given a video � and a user query �, a frame sampler extracts � frames $\mathbf { F } = \left\{ F _ { 1 } , F _ { 2 } , \ldots , F _ { N } \right\}$ . These frames are encoded by a visual encoder (e.g., CLIP-L [24]) to obtain visual features $\mathbf { F } _ { v } .$ The standard LVLM inference can be formulated as:

$$
O = \mathrm { L V L M } ( \mathbf { F } _ { v } , Q )\tag{1}
$$

However, this direct approach struggles with long videos where critical information may be sparse across extended durations. EC-RAG addresses this limitation by constructing an event chain (EC) E that captures the temporal structure of video content, and performing query-guided event localization to retrieve relevant context before final reasoning:

$$
O = \mathrm { L V L M } ( \mathbf { F } _ { v } , Q , \mathcal { E } , \mathbf { A } )\tag{2}
$$

where E denotes the temporal event chain and A represents the retrieved multimodal evidence. This formulation enables the model to reason over both visual frames and structured textual context derived from the video.

## 3.1 Query Decoupling

Before processing the video content, we first analyze the user query to understand what types of information are needed to answer it. Motivated by recent methods [13, 20], this query decoupling step separates the retrieval planning from the actual video processing, allowing the system to focus computational resources on relevant modalities. Upon receiving a user query, EC-RAG decouples it into structured retrieval requests without accessing video frames. The LVLM analyzes the query to determine what information sources are needed:

$$
{ \bf R } = \mathrm { L V L M } ( P _ { d } , Q ) = \{ R _ { \mathrm { a s r } } , R _ { \mathrm { o c r } } , R _ { \mathrm { d e t } } \}\tag{3}
$$

where $P _ { d }$ is the decoupling prompt enhanced with few-shot examples that demonstrate how to extract retrieval targets from various question types. Each request may be NULL if the corresponding modality is unnecessary for the given query. Specifically, $R _ { \mathrm { a s r } }$ specifies key phrases to retrieve from speech transcripts, which is particularly useful for questions about dialogue content or narration. $R _ { \mathrm { o c r } }$ defines keywords to search in on-screen text, targeting questions about titles, labels, or displayed information. $R _ { \mathrm { d e t } }$ indicates physical objects to detect in video frames, essential for visual grounding questions.

The decoupling prompt guides the LVLM to identify concrete retrieval targets through structured examples. For detection requests, we enforce that $R _ { \mathrm { d e t } }$ contains only physical entities rather than abstract concepts, ensuring compatibility with vision-based detection models. This constraint prevents the system from attempting to detect non-visual concepts like emotions or intentions.

## 3.2 CLIP-Variance Guided Multimodal Extraction

This stage selects informative keyframes and extracts multimodal information from the video in parallel.

Keyframe Selection. A critical challenge in video understanding is selecting which frames to process for detailed analysis. Uniform sampling may miss important moments, while processing all frames is computationally prohibitive. Rather than uniformly sampling frames or relying solely on semantic similarity, inspired by query-guided frame selection in prior work [1, 20], we propose a CLIP-variance (CV) guided selection mechanism that jointly considers query relevance and visual informativeness. The intuition is that informative frames should both relate to the query content and contain rich visual details rather than static or redundant imagery. For each frame $F _ { t }$ , we compute a weighted score:

$$
s _ { t } = \mathrm { s i m } _ { \mathrm { C L I P } } ( R _ { \mathrm { d e t } } , F _ { t } ) \cdot \sigma _ { t }\tag{4}
$$

where sim $\operatorname { z m } ( \cdot )$ denotes the CLIP similarity between the detection query and frame content. This term ensures that selected frames are semantically relevant to the objects or entities mentioned in the query. The second factor $\sigma _ { t }$ is the normalized visual variance of frame $F _ { t } \colon$

$$
\sigma _ { t } = { \frac { \operatorname { V a r } ( F _ { t } ) } { \operatorname* { m a x } _ { j } \operatorname { V a r } ( F _ { j } ) } }\tag{5}
$$

The variance term captures the visual complexity of each frame, computed from the pixel intensity distribution. Frames with higher variance typically contain more visual structure and details, while low-variance frames often correspond to static backgrounds, black screens, or uniform scenes that provide limited information. By multiplying these two factors, we favor frames that are both queryrelevant and visually informative. Keyframes are selected by thresholding on the weighted scores:

$$
\mathbf { F } _ { \mathrm { k e y } } = \{ F _ { t } \ | \ s _ { t } \ge \tau _ { s } \}\tag{6}
$$

where $\tau _ { s }$ is a score threshold, with constraints on minimum and maximum frame counts to ensure adequate coverage across the video duration.

ASR Extraction. Audio information provides crucial context that is often unavailable through visual analysis alone, including dialogue, narration, and background audio cues. Following recent retrieval-augmented video systems [1, 20], we extract the audio track from the video and transcribe it using Whisper [25], a robust speech recognition model capable of handling diverse audio conditions. The resulting speech transcripts $T _ { \mathrm { a s r } }$ are segmented by timestamps to preserve temporal alignment with video frames. These segments are then encoded using Contriever [8], a dense retrieval encoder trained for semantic similarity matching, producing embeddings that capture the semantic content of each transcript segment. The embeddings are indexed in a FAISS [10] database for eficient similarity search:

$$
\mathrm { D B } _ { \mathrm { a s r } } \overleftarrow { \mathrm { e } } \overleftarrow { \mathrm { e } } \mathrm { ~ B } _ { \mathrm { a s r } } = \mathrm { C o n t r i e v e r } ( T _ { \mathrm { a s r } } )\tag{7}
$$

OCR Extraction. On-screen text frequently contains valuable information such as titles, subtitles, labels, and other textual elements that complement visual and audio content. Current LVLMs often struggle to accurately recognize text within video frames, particularly for small or stylized fonts. To address this limitation, we employ EasyOCR [9], a dedicated text recognition model, to extract on-screen text from each sampled frame. Since OCR models may produce noisy outputs, we apply a confidence threshold to filter low-quality detections:

$$
T _ { \mathrm { o c r } } = \{ t \mid t = \operatorname { E a s y O C R } ( F ) , \operatorname { c o n f } ( t ) > \tau _ { \mathrm { o c r } } \}\tag{8}
$$

This filtering step removes unreliable text detections that could introduce noise into the retrieval process. The filtered texts are encoded and stored in a separate FAISS index following the same procedure as ASR:

$$
\mathrm { D B } _ { \mathrm { o c r } } \overleftarrow { \mathrm { e } } \overleftarrow { \mathrm { e } } \mathrm { I } \mathrm { S } \mathrm { S } \mathrm { \textbf { E } } _ { \mathrm { o c r } } = \mathrm { C o n t r i e v e r } ( T _ { \mathrm { o c r } } )\tag{9}
$$

Object Detection with Scene Graph. While LVLMs demonstrate strong capabilities in object recognition, they continue to face challenges in precise object counting, localization, and understanding spatial relationships between objects. These limitations often lead to hallucinations when answering questions that require fine-grained visual understanding. To provide more accurate objectlevel information, we employ APE [27], an open-vocabulary object detection model that accepts textual prompts specifying which objects to detect. For the selected keyframes $\mathbf { F } _ { \mathrm { k e y } }$ , APE localizes relevant objects based on the detection request:

$$
D _ { t } = \mathrm { A P E } ( F _ { t } , R _ { \mathrm { d e t } } ) , \quad F _ { t } \in { \bf F } _ { \mathrm { k e y } }\tag{10}
$$

The raw detection outputs consisting of object categories and bounding box coordinates are not directly interpretable by language models. To bridge this gap, inspired by scene graph representations in visual reasoning [1, 20], we transform the detection results into semantic scene graph descriptions expressed in natural language. For each keyframe, we generate three complementary types of information: (1) Object Location $A _ { \mathrm { l o c } }$ converts bounding box coordinates into natural language spatial descriptions $( \mathbf { e . g . }$ , “left side of frame”, “upper right corner”), making positional information accessible to the language model; (2) Object Count $A _ { \mathrm { { c n t } } }$ aggregates detection results to provide accurate counts for each object category, addressing the counting limitations of LVLMs; (3) Object Relation $A _ { \mathrm { r e l } }$ computes and describes relative spatial relationships between detected objects (e.g., “person A is to the left of person B”). The complete scene graph representation combines these components:

$$
T _ { \mathrm { d e t } } = \mathrm { S c e n e G r a p h } ( A _ { \mathrm { l o c } } , A _ { \mathrm { c n t } } , A _ { \mathrm { r e l } } )\tag{11}
$$

## 3.3 Temporal Event Chain Construction

Unlike frame-level retrieval methods that treat video as a collection of independent images, EC-RAG explicitly models the temporal structure of video content by constructing an event chain. This representation captures how events unfold over time and enables reasoning about temporal relationships.

Video Segmentation. Long videos contain extended sequences of content that cannot be processed as a single unit. We partition the video into non-overlapping temporal segments offixed duration Δ:

$$
\boldsymbol { S } = \{ S _ { i } = [ t _ { i } , t _ { i } + \Delta ) ~ | ~ i = 0 , 1 , . . . , \lfloor L / \Delta \rfloor \}\tag{12}
$$

where $L$ is the total video duration. Each segment $S _ { i }$ serves as the basic unit for event extraction and is associated with the multimodal information (ASR, OCR, DET) extracted from frames within its time range. This segmentation balances granularity and eficiency: segments should be short enough to capture coherent events while being long enough to contain meaningful content.

Event Extraction. Raw multimodal data from each segment needs to be synthesized into a coherent semantic representation. For each segment $S _ { i } ,$ we aggregate the corresponding ASR, OCR, and DET information and prompt the LVLM to extract a semantic event description that summarizes what happens in that temporal window:

$$
e _ { i } = \mathrm { L V L M } ( P _ { e } , T _ { \mathrm { a s r } } ^ { ( i ) } , T _ { \mathrm { o c r } } ^ { ( i ) } , T _ { \mathrm { d e t } } ^ { ( i ) } )\tag{13}
$$

where $P _ { e }$ is the event extraction prompt that instructs the model to integrate information across modalities. The prompt guides the LVLM to focus on the main action or occurrence while considering all available evidence. Each extracted event $e _ { i }$ comprises three components: a description that concisely summarizes what happens in this segment by integrating speech, text, and visual information; a type that classifies the event as action, dialogue, scene transition, or informational content; and a list of participants identifying key entities (people, objects) involved in the event.

Event Chain Assembly. Individual events are connected to form a coherent narrative structure. The extracted events are organized into a temporal event chain $\mathcal { E } = \{ e _ { 0 } , e _ { 1 } , \ldots , e _ { n } \}$ with explicit temporal ordering that reflects the video’s chronological structure. We establish predecessor-successor relations between adjacent events:

$$
e _ { i } \xrightarrow { \mathrm { t e m p o r a l } } e _ { i + 1 } , \quad \forall i \in [ 0 , n - 1 ]\tag{14}
$$

The complete event chain provides a structured overview of the video’s narrative progression, capturing how events unfold and connect across time. This representation enables the model to understand not just individual moments but also the broader context and temporal dynamics of the video content.

## 3.4 Evidence-augmented Multimodal Reasoning

With the event chain constructed, EC-RAG performs query-guided retrieval to identify relevant events and gather supporting evidence for answer generation.

Query-guided Event Localization. Not all events in a video are relevant to a given query. We identify the most relevant events

through semantic similarity matching between the query and event descriptions. Each event description is encoded using Contriever and compared against the query embedding using cosine similarity:

$$
\sin _ { i } = \cos \left( \operatorname { C o n t r i e v e r } ( Q ) , \operatorname { C o n t r i e v e r } ( e _ { i } . \operatorname { d e s c } ) \right)\tag{15}
$$

This text-to-text matching is more appropriate than image-text matching (e.g., CLIP) since both the query and event descriptions are textual. Events exceeding a relevance threshold $\tau _ { \mathrm { l o c } }$ are selected as located events:

$$
\mathcal { E } _ { \mathrm { l o c } } = \{ e _ { i } \ | \ \mathrm { s i m } _ { i } > \tau _ { \mathrm { l o c } } \}\tag{16}
$$

The located events represent the temporal regions most likely to contain information needed to answer the query.

Evidence Retrieval. While event descriptions provide highlevel summaries, detailed evidence from the original multimodal sources may be necessary for accurate reasoning. For each located event $e _ { i } \in \mathcal { E } _ { \mathrm { l o c } }$ , we retrieve the associated multimodal evidence from the constructed databases. Using the retrieval requests R generated during query decoupling, we query the ASR and OCR databases to obtain relevant text segments:

$$
A _ { \mathrm { a s r } } = \mathrm { R e t r i e v e } ( \mathrm { D B } _ { \mathrm { a s r } } , R _ { \mathrm { a s r } } , \tau ) , \quad A _ { \mathrm { o c r } } = \mathrm { R e t r i e v e } ( \mathrm { D B } _ { \mathrm { o c r } } , R _ { \mathrm { o c r } } , \tau )\tag{17}
$$

The DET evidence is collected from the scene graphs of keyframes falling within the located events’ time ranges:

$$
A _ { \mathrm { d e t } } = \{ T _ { \mathrm { d e t } } ^ { ( t ) } ~ | ~ F _ { t } \in \bf { F }  _ { \mathrm { k e y } } , ~ t \in e _ { i } . { \mathrm { s p a n } } , ~ e _ { i } \in \mathcal { E } _ { \mathrm { l o c } } \}\tag{18}
$$

where the retrieval function returns all database entries with similarity scores exceeding threshold �. The DET evidence $A _ { \mathrm { d e t } }$ is directly obtained from the scene graph descriptions of keyframes within the located events’ time ranges, providing detailed object-level information for visual grounding.

Context Construction and Answer Generation. The final reasoning step integrates multiple levels of context to provide the LVLM with comprehensive information. The context combines three complementary components: (1) Event Chain Summary provides a global overview of all events in the video, giving the model awareness of the full narrative arc even for content not directly relevant to the query; (2) Located Events contain detailed descriptions of query-relevant events along with their temporal predecessors, enabling understanding of causal relationships and temporal context; (3) Multimodal Evidence includes the raw ASR, OCR, and DET information retrieved for the located events, providing specific details that support accurate reasoning. The complete context is assembled as:

$$
C = \mathrm { C o n c a t } ( \mathrm { S u m m a r y } ( { \mathcal { E } } ) , { \mathrm { D e t a i l } } ( { \mathcal { E } } _ { \mathrm { l o c } } ) , A _ { \mathrm { a s r } } , A _ { \mathrm { o c r } } , A _ { \mathrm { d e t } } )\tag{19}
$$

Finally, the LVLM generates the answer by jointly reasoning over the video frames, user query, and constructed context:

$$
O = \mathrm { L V L M } ( \mathbf { F } _ { v } , Q , C )\tag{20}
$$

## 4 Experiments

## 4.1 Datasets

We assess EC-RAG on three widely adopted long-video benchmarks that span diverse durations, domains, and reasoning demands.

Video-MME [6] ofers a systematic testbed for evaluating multimodal comprehension over real-world video content. Its 900 videos are grouped by length into short (< 2 min), medium (4–15 min), and long (30–60 min) subsets, yielding 2,700 multiple-choice questions across 12 task types—ranging from object recognition and counting to temporal reasoning and OCR—and 30 sub-categories drawn from six broad domains including knowledge, sports, and film. This breadth makes Video-MME particularly suitable for stress-testing whether a retrieval-augmented pipeline can supply the right modality of evidence for each question type.

Table 1: Performance on the Video-MME [6] benchmark. #Text indicates the volume of retrieved textual context. EC-RAG is integrated into four open-source LVLMs spanning 8–32 input frames.
<table><tr><td>Model</td><td>#Text</td><td>LLM Params</td><td>Frames</td><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td><td>Gain</td></tr><tr><td colspan="9">Proprietary LVLMs</td></tr><tr><td>GPT-4o [23]</td><td>一</td><td>一</td><td>384</td><td>80.0</td><td>70.3</td><td>65.3</td><td>71.9</td><td>一</td></tr><tr><td>Gemini-1.5-Pro [31]</td><td>一</td><td>1</td><td>0.5 fps</td><td>81.7</td><td>74.3</td><td>67.4</td><td>75.0</td><td>一</td></tr><tr><td colspan="9">Open-Source LVLMs</td></tr><tr><td>Video-LLaVA [13]</td><td></td><td>7B</td><td>8</td><td>44.6</td><td>38.3</td><td>35.8</td><td>39.6</td><td></td></tr><tr><td>Video-LLaVA + EC-RAG</td><td>2.0K</td><td>7B</td><td>8</td><td>50.1</td><td>44.6</td><td>43.4</td><td>46.0</td><td>+6.4</td></tr><tr><td>LLaVA-NeXT-Video [39]</td><td></td><td>7B</td><td>16</td><td>49.4</td><td>43.0</td><td>36.7</td><td>43.0</td><td></td></tr><tr><td>LLaVA-NeXT-Video + EC-RAG</td><td>2.0K</td><td>7B</td><td>16</td><td>55.8</td><td>53.2</td><td>52.5</td><td>53.8</td><td>+10.8</td></tr><tr><td>Long-LLaVA [29]</td><td></td><td>7B</td><td>32</td><td>60.3</td><td>51.4</td><td>44.1</td><td>52.0</td><td></td></tr><tr><td>Long-LLaVA + EC-RAG</td><td>1.9K</td><td>7B</td><td>32</td><td>67.6</td><td>60.3</td><td>60.0</td><td>62.6</td><td>+10.6</td></tr><tr><td>LLaVA-Video [40]</td><td></td><td>7B</td><td>32</td><td>75.7</td><td>59.6</td><td>52.1</td><td>62.4</td><td></td></tr><tr><td>LLaVA-Video + Video-RAG [20]</td><td>2.0K</td><td>7B</td><td>32</td><td>70.9</td><td>63.2</td><td>59.0</td><td>64.3</td><td>+1.9</td></tr><tr><td> $\mathrm { L L a V A  – V i d e o } + \mathrm { T V } \mathrm { - R A G } \left[ 1 \right]$ </td><td>2.0K</td><td>7B</td><td>32</td><td>72.4</td><td>64.1</td><td>59.8</td><td>65.4</td><td>+3.0</td></tr><tr><td> $\mathrm { L L a V A – V i d e o + E C \mathrm { - } R A G }$ </td><td>2.0K</td><td>7B</td><td>32</td><td>77.8</td><td>64.3</td><td>60.4</td><td>67.5</td><td>+5.1</td></tr><tr><td>LLaVA-Video [40]</td><td></td><td>72B</td><td>32</td><td>78.0</td><td>63.7</td><td>59.6</td><td>67.1</td><td></td></tr><tr><td> $\mathrm { L L a V A – V i d e o + E C \mathrm { - } R A G }$ </td><td>2.1K</td><td>72B</td><td>32</td><td>81.9</td><td>72.4</td><td>72.7</td><td>75.7</td><td>+8.6</td></tr></table>

MLVU [42] assembles nine evaluation tasks from videos of3 min to 2 h (mean ≈ 12 min), requiring models to sustain reasoning over extended temporal ranges

LongVideoBench [33] focuses on multimodal retrieval and compositional reasoning within lengthy footage. It provides 6,678 human-written multiple-choice questions spanning 17 thematic categories, testing a model’s ability to pinpoint and integrate evidence scattered across long video timelines.

## 4.2 Implementation Details

All experiments were conducted on NVIDIA RTX 4090 and A800 80 GB GPUs. We adopt LLaVA-Video-7B [40] as the primary LVLM backbone, with three additional 7B architectures and the 72B variant evaluated in Table 1 to verify cross-architecture and cross-scale generalizability. Detection requests produced during query decoupling are post-filtered with spaCy to retain only concrete, CLIPresponsive entities. Both the CLIP similarity threshold and the FAISS [10] retrieval threshold are fixed at � = 0.3, with IndexFlatIP as the similarity backend. Videos are segmented into 30-second windows for event chain construction. Speech is transcribed by Whisper-Large [25], on-screen text is recognized by EasyOCR [9], open-vocabulary detection is handled by APE [27], and dense text retrieval is performed by Contriever [8]. Our ablation suite is built around LLaVA-Video-7B at 32 frames, whose moderate context window makes it a practical testbed for probing the efect of individual components and hyper-parameters.

Video-MME. To verify that EC-RAG generalizes across architectures, model scales, and frame budgets, we integrate it into four open-source LVLMs spanning 8–32 input frames, as reported in Table 1. EC-RAG yields consistent improvements for every backbone, with gains most pronounced on long videos where uniform sampling alone cannot cover the query-relevant content. The benefit is especially evident on frame-constrained backbones, confirming that event-chain retrieval compensates efectively for sparse visual input. The pipeline injects approximately 2 K tokens of supplementary textual context (the #Text column), comparable to roughly 14 extra keyframes in payload, which complement the visual input and help anchor temporal reasoning.

Scaling to the 72B LLaVA-Video, EC-RAG achieves 75.7% overall, surpassing both GPT-4o [23] and Gemini-1.5-Pro [31], showing that structured event-chain retrieval enables open-source models to outperform proprietary systems without additional pretraining. EC-RAG also consistently outperforms Video-RAG [20] and TV-RAG [1] under the same backbone setting, as event-level temporal localization provides more precise context than frame-level retrieval.

MLVU. Table 2 reports performance on the MLVU multiplechoice task. EC-RAG lifts LLaVA-Video-7B from 70.8 to 72.9, outperforming both Video-RAG and TV-RAG on the same backbone. Notably, this 7B model with only 64 frames also surpasses Oryx-1.5 [19], a 32B model operating on 128 frames, highlighting the eficiency of event-chain retrieval over brute-force scaling.

LongVideoBench. As shown in Table 3, EC-RAG also achieves the best result on LongVideoBench, reaching 59.5 with LLaVA-Video-7B at 64 frames. This benchmark requires retrieving and composing evidence distributed across long timelines, a scenario that directly benefits from EC-RAG’s temporal event chain design. Compared with Video-RAG and TV-RAG under the same setting, EC-RAG provides a more substantial boost over the unaugmented baseline, indicating that event-level localization is more efective than flat retrieval when evidence is temporally scattered.

![](images/ca04fb71e14a2d1a3348616d1c0a67ea03ddcee0337087a3b47e606bc9e9a295.jpg)  
Figure 3: Left: Grad-CAM heatmaps on query-relevant and query-irrelevant frames for the baseline and EC-RAG. Right: t-SNE projection of query, vision, and text features, showing tighter cross-modal alignment when EC-RAG is applied.

Table 2: Overall accuracy on the multiple-choice split of the MLVU [42] benchmark. EC-RAG achieves the best result among all 7B-scale models.
<table><tr><td>Model</td><td>#Params</td><td>Frames</td><td>Overall</td></tr><tr><td colspan="2">Proprietary LVLMs</td><td colspan="2"></td></tr><tr><td>GPT-4o [23]</td><td>一</td><td>0.5 fps</td><td>64.6</td></tr><tr><td colspan="4">Open-Source LVLMs</td></tr><tr><td>Video-CCAM [5] Video-XL [28]</td><td>14B</td><td>96</td><td>63.1</td></tr><tr><td>Aria [11]</td><td>7B</td><td>256</td><td>64.9</td></tr><tr><td>LLaVA-Video [40]</td><td>25.3B</td><td>256</td><td>70.6</td></tr><tr><td></td><td>7B</td><td>64</td><td>70.8</td></tr><tr><td>Oryx-1.5 [19]</td><td>32B</td><td>128</td><td>72.3</td></tr><tr><td>LLaVA-Video + Video-RAG [20]</td><td>7B</td><td>64</td><td>72.4</td></tr><tr><td>LLaVA-Video + TV-RAG [1]</td><td>7B</td><td>64</td><td>72.6</td></tr><tr><td>LLaVA-Video + EC-RAG</td><td>7B</td><td>64</td><td>72.9</td></tr></table>

Table 3: Performance on the LongVideoBench [33] validation set.
<table><tr><td>Model</td><td>#Params</td><td>Frames</td><td>Overall</td></tr><tr><td>VideoChat2-Mistral [12]</td><td>7B</td><td>8</td><td>39.3</td></tr><tr><td>ShareGPT4Video [3]</td><td>7B</td><td>8</td><td>39.7</td></tr><tr><td>LLaVA-Next-Mistral [39]</td><td>7B</td><td>8</td><td>49.1</td></tr><tr><td>PLLaVA [34]</td><td>34B</td><td>16</td><td>53.2</td></tr><tr><td>LLaVA-Video [40]</td><td>7B</td><td>64</td><td>56.6</td></tr><tr><td>LLaVA-Video + Video-RAG [20]</td><td>7B</td><td>64</td><td>58.7</td></tr><tr><td>LLaVA-Video + TV-RAG [1]</td><td>7B</td><td>64</td><td>58.8</td></tr><tr><td>LLaVA-Video + EC-RAG</td><td>7B</td><td>64</td><td>59.5</td></tr></table>

## 4.3 Ablation Studies

To quantify each component’s contribution, we ablate EC-RAG on Video-MME with LLaVA-Video-7B at 32 frames (Table 4). Among all modules, ASR has the greatest impact: removing it causes the most significant performance degradation, particularly on medium and long videos, confirming that speech transcription is the most informative auxiliary modality for video QA. The event chain module ranks second in importance, as disabling temporal localization leads to less targeted evidence retrieval. CLIP-variance guided keyframe selection, OCR, and DET each provide meaningful but comparatively smaller gains, reflecting their complementary roles in selecting informative frames, capturing on-screen text, and grounding physical entities respectively.

Table 4: Component ablation of EC-RAG on Video-MME with LLaVA-Video-7B.
<table><tr><td>ASR</td><td>OCR</td><td>DET</td><td>EC</td><td>CV</td><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td></tr><tr><td>X</td><td>√</td><td>√</td><td>√</td><td>√</td><td>74.8</td><td>60.3</td><td>54.7</td><td>63.3</td></tr><tr><td>√</td><td>X</td><td>√</td><td>√</td><td>√</td><td>76.5</td><td>62.8</td><td>59.9</td><td>66.4</td></tr><tr><td>√</td><td>√</td><td>X</td><td>√</td><td>√</td><td>77.1</td><td>62.9</td><td>59.2</td><td>66.4</td></tr><tr><td>√</td><td>√</td><td>√</td><td>X</td><td>√</td><td>77.0</td><td>61.6</td><td>58.4</td><td>65.7</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>76.4</td><td>63.5</td><td>58.9</td><td>66.3</td></tr><tr><td>L</td><td>√</td><td>√</td><td>√</td><td>√</td><td>77.8</td><td>64.3</td><td>60.4</td><td>67.5</td></tr></table>

## 4.4 Qualitative Analysis

Figure 4 presents a representative case from Video-MME in which the baseline LLaVA-Video fails but EC-RAG succeeds. The question asks what happened in the car after the heroine returned from shopping. With 32 uniformly sampled frames, the baseline captures a frame showing the baby in a cheerful state and, lacking any temporal context to disambiguate it from other segments, incorrectly selects option C (“Baby happy”).

EC-RAG addresses this through its pipeline: the query decoupling stage extracts modality-specific retrieval cues (e.g., ASR keywords “car, shopping, unhappy” and DET targets “baby, car seat”). The event chain then localizes the relevant temporal segment, whose ASR transcript explicitly states “He was just not happy in the car.” Simultaneously, CLIP-variance guided keyframe selection identifies a visually informative frame that is forwarded to the object detector, revealing a baby being fed with a recently-cried expression. Equipped with this temporally grounded multimodal evidence, the model correctly selects option D (“Baby unhappy”).

![](images/8160a8f4321a348c618aee6cb72606d65f94e05fd8ab279dfc2f453dce33142e.jpg)  
Figure 4: Qualitative comparison between the baseline LLaVA-Video and EC-RAG on a Video-MME example. The baseline confuses temporally distinct segments, whereas EC-RAG localizes the query-relevant event through its event chain and aggregates ASR and DET evidence to arrive at the correct answer.

Figure 3 further illustrates the efect of EC-RAG through Grad-CAM heatmaps and t-SNE projections. The Grad-CAM visualizations compare the attention distribution ofthe baseline and EC-RAG on both a query-relevant and a query-irrelevant frame. With the supplementary textual evidence provided by EC-RAG, the model’s attention concentrates more tightly on the semantically relevant regions (e.g., the baby and car seat), whereas the baseline distributes attention more difusely. The t-SNE plot projects the query, vision, and text features into a shared 2D space, showing that EC-RAG’s auxiliary tokens pull the vision and query representations closer together, resulting in more aligned cross-modal features that facilitate accurate reasoning.

## 5 Conclusion

We presented EC-RAG, a training-free retrieval-augmented generation framework that structures long-video understanding around temporal event chains. By decomposing videos into semantically coherent events through the fusion of ASR, OCR, and open-vocabulary detection, and localizing query-relevant segments before injecting multimodal evidence into the LVLM prompt, EC-RAG consistently improves diverse backbones across three benchmarks—Video-MME, MLVU, and LongVideoBench—outperforming existing RAG-based methods under every tested setting. Notably, with the 72B LLaVA-Video backbone, EC-RAG surpasses both GPT-4o and Gemini-1.5- Pro on Video-MME using a fully open-source pipeline, demonstrating that structured event-level retrieval enables open-source models to rival and even surpass proprietary systems without additional training.

Beyond the benchmarks evaluated in this work, we believe the event-chain paradigm holds broader potential for other videocentric tasks such as video summarization, temporal grounding, and multi-turn video dialogue, where preserving temporal structure and aggregating cross-modal evidence are equally critical. The training-free nature of EC-RAG also makes it readily applicable to emerging LVLMs as they continue to evolve, ofering a flexible and scalable augmentation strategy.

## References

[1] Zongsheng Cao, Yangfan He, Anran Liu, Jun Xie, Feng Chen, and Zhepeng Wang. 2025. Tv-rag: A temporal-aware and semantic entropy-weighted framework for long video retrieval and understanding. In Proceedings of the 33rd ACM International Conference on Multimedia. 9071–9079.

[2] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. 2024. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision. Springer, 370– 387.

[3] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. 2024. Sharegpt4video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems 37 (2024), 19472–19495.

[4] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. 2024. Videoagent: A memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision. Springer, 75–92.

[5] Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang. 2024. Video-ccam: Enhancing video-language understanding with causal cross attention masks for short and long videos. arXiv preprint arXiv:2408.14023 (2024).

[6] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. 2025. Video mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 24108–24118.

[7] Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 14953–14962.

[8] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118 (2021).

[9] JaidedAI. 2023. EasyOCR. https://github.com/JaidedAI/EasyOCR. Accessed: 2026-03-30.

[10] Jef Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE transactions on big data 7, 3 (2019), 535–547.

[11] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Fan Zhou, Chengen Huang, Yanpeng Li, et al. 2024. Aria: An open multimodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993 (2024).

[12] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2025. Videochat: Chat-centric video understanding. Science China Information Sciences 68, 10 (2025), 200102.

[13] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2024. Video-llava: Learning united visual representation by alignment before projection. In Proceedings ofthe 2024 conference on empirical methods in natural language processing. 5971–5984.

[14] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. 2024. Vila: On pre-training for visual language models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 26689–26699.

[15] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. 2023. MM-VID: Advancing Video Understanding with GPT-4V (ision). CoRR abs/2310.19773 (2023). arXiv preprint arXiv:2310.19773 (2023).

[16] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge. https://llava-vl.github.io/blog/2024-01-30-llava-next/

[17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems 36 (2023), 34892–34916.

[18] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. 2026. Kangaroo: A Powerful Video-Language Model Supporting Long-context Video Input: J. Liu et al. International Journal of Computer Vision 134, 3 (2026), 114.

[19] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. 2024. Oryx mllm: On-demand spatial-temporal understanding at arbitrary reso lution. arXiv preprint arXiv:2409.12961 (2024).

[20] Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin, Haojia Lin, Chaoyou Fu, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, et al. 2024. Video-rag: Visually-aligned retrieval-augmented long video comprehension. arXiv preprint arXiv:2411.13093 (2024).

[21] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. 2024. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 12585–12602.

[22] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho, and Cordelia Schmid. 2024. Morevqa: Exploring modular reasoning models for video question answering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13235–13245.

[23] OpenAI. 2024. GPT-4o System Card. https://openai.com/index/gpt-4o-systemcard/. Accessed: 2026-03-30.

[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[25] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning. PMLR, 28492–28518.

[26] Yuzhang Shang, Bingxin Xu, Weitai Kang, Mu Cai, Yuheng Li, Zehao Wen, Zhen Dong, Kurt Keutzer, Yong Jae Lee, and Yan Yan. 2024. Interpolating videollms: Toward longer-sequence lmms in a training-free manner. arXiv preprint arXiv:2409.12963 (2024).

[27] Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang, Ke Li, Xing Sun, Yunsheng Wu, Shaohui Lin, and Rongrong Ji. 2024. Aligning and prompting everything all at once for universal visual perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13193–13203.

[28] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. 2025. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 26160–26169.

[29] Yin Song, Chen Wu, and Eden Duthie. 2024. aws-prototyping/long-llava-qwen2- 7b. https://huggingface.co/aws-prototyping/long-llava-qwen2-7b. Accessed: 2026-03-30.

[30] Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF international conference on computer vision. 11888–11898.

[31] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).

[32] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. 2024. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision. Springer, 58–76.

[33] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems 37 (2024), 28828–28857.

[34] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. 2024. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994 (2024).

[35] Zhucun Xue, Jiangning Zhang, Xurong Xie, Yuxuan Cai, Yong Liu, Xiangtai Li, and Dacheng Tao. 2025. Adavideorag: Omni-contextual adaptive retrievalaugmented eficient long video understanding. arXiv preprint arXiv:2506.13589 (2025).

[36] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and Gedas Bertasius. 2024. A simple llm framework for long-range video question-answering. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing. 21715–21737.

[37] Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 conference on empirical methods in natural language processing: system demonstrations. 543–553.

[38] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. 2024. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852 (2024).

[39] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. 2024. LLaVA-NeXT: A Strong Zero-shot Video Understanding Model. https://llava-vl.github.io/blog/2024-04-30-llava-nextvideo/

[40] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. 2024. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713 (2024).

[41] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and RongJin. 2024. Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models. arXiv:2406.08487 [cs.CV] https://arxiv.org/abs/2406.08487

[42] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. 2024. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264 2, 5 (2024), 6.

## Supplementary Material

## A Additional Ablation Studies

Frame Sampling Budget. We investigate the sensitivity of EC-RAG to the number of sampled frames by evaluating LLaVA-Video-7B [40] under three frame budgets: 8, 16, and 32 on Video-MME [6]. As shown in Figure 5, EC-RAG consistently outperforms the baseline at every frame budget across all duration categories. Notably, the baseline shows diminishing returns as the frame count increases—particularly on long videos, where doubling from 16 to 32 frames yields only marginal improvement, suggesting that simply adding more visual input cannot compensate for the lack of temporal and textual context. In contrast, EC-RAG maintains substantial gains at all budgets, with the largest absolute improvement observed at 16 frames, indicating that event-chain retrieval is especially efective when the visual input is moderately sparse and the model most needs supplementary evidence to bridge temporal gaps.

Retrieval Threshold. Table 5 investigates the efect of the retrieval similarity threshold � on both eficiency and accuracy. Lower thresholds (�=0.0–0.1) retain nearly all retrieved text, injecting up to 4.4 K supplementary tokens and pushing inference time to over 46 s per question, yet only achieve 64.1–64.7% overall. As � increases, the token budget and latency decrease steadily; at �=0.3, EC-RAG processes 2.4 K tokens in 18 s while reaching the peak overall accuracy of 65.1%, with the best short-video (73.4%) and long-video (61.2%) scores. This non-monotonic pattern indicates that unfiltered retrieval introduces noisy evidence that can mislead the model. Beyond �=0.3, accuracy degrades as useful context is progressively pruned: �=0.5 drops to 60.8% and �=1.0—which effectively discards all retrieved text (0.0 K tokens)—falls to 58.9%. We therefore adopt �=0.3 as the default, which reduces latency by roughly 65% compared to unfiltered retrieval while maximizing accuracy.

Cross-Backbone RAG Comparison. Table 6 compares EC-RAG with Video-RAG [20] and TV-RAG [1] under two additional backbone settings. On Video-LLaVA-7B [13] with 8 frames, all three RAG methods substantially improve over the baseline (39.6%), with EC-RAG achieving the highest overall accuracy of 46.0%, outper forming TV-RAG by +0.5 and Video-RAG by +1.0. The advantage is most pronounced on long videos (+0.8 over TV-RAG), where event-chain retrieval provides more temporally precise context than frame-level or segment-level approaches. On LongVA-7B [38] with 32 frames, the same trend holds: EC-RAG reaches 62.8% overall, surpassing TV-RAG (61.7%) and Video-RAG (60.1%). EC-RAG consistently leads across all duration categories under both backbones, confirming that structured event-level retrieval generalizes across diferent LVLMs and frame budgets.

## B Task-Type Breakdown

Table 7 presents a fine-grained breakdown of accuracy across all 12 task types on Video-MME under 8-frame and 16-frame settings. With 8 frames, EC-RAG improves the average by +5.6 points (54.6%→60.2%), with the largest gains on Information Synopsis (+9.6), Spatial Reasoning (+9.0), and OCR (+7.9)—tasks that heavily rely on textual or spatial cues where event-chain evidence provides direct support. At 16 frames, the average gain further widens to +7.8 points (58.0%→65.8%), with Action Reasoning seeing the most dramatic improvement (+13.7) followed by Attribute Perception (+10.4) and Information Synopsis (+9.9). EC-RAG achieves the best score on 10 out of 12 task types at 8 frames and 12 out of 12 at 16 frames, demonstrating that the event-chain retrieval complements visual features comprehensively across diverse question types rather than benefiting only a narrow subset.

Table 5: Performance with diferent retrieval thresholds on Video-MME [6] with LLaVA-Video-7B [40].
<table><tr><td>τ</td><td>#Token</td><td>Time</td><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td></tr><tr><td>0.0</td><td>4.4K</td><td>52s</td><td>72.4</td><td>61.6</td><td>60.0</td><td>64.7</td></tr><tr><td>0.1</td><td>4.0K</td><td>46s</td><td>71.8</td><td>60.5</td><td>59.9</td><td>64.1</td></tr><tr><td>0.2</td><td>3.5K</td><td>23s</td><td>70.3</td><td>60.4</td><td>57.8</td><td>62.8</td></tr><tr><td>0.3</td><td>2.4K</td><td>18s</td><td>73.4</td><td>60.7</td><td>61.2</td><td>65.1</td></tr><tr><td>0.4</td><td>1.6K</td><td>13s</td><td>70.1</td><td>60.6</td><td>57.5</td><td>62.7</td></tr><tr><td>0.5</td><td>1.2K</td><td>12s</td><td>68.9</td><td>58.2</td><td>55.3</td><td>60.8</td></tr><tr><td>1.0</td><td>0.0K</td><td>10s</td><td>66.5</td><td>56.4</td><td>53.7</td><td>58.9</td></tr></table>

Table 6: Comparison with other RAG methods across diferent backbones on Video-MME [6].
<table><tr><td>Model</td><td>Params</td><td>Frames</td><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td></tr><tr><td>X=Video-LLaVA [13]</td><td>7B</td><td>8</td><td>44.6</td><td>38.3</td><td>35.8</td><td>39.6</td></tr><tr><td>X+Video-RAG [20]</td><td>7B</td><td>8</td><td>49.5</td><td>43.0</td><td>42.5</td><td>45.0</td></tr><tr><td>X+TV-RAG [1]</td><td>7B</td><td>8</td><td>49.7</td><td>44.3</td><td>42.6</td><td>45.5</td></tr><tr><td>X+EC-RAG</td><td>7B</td><td>8</td><td>50.1</td><td>44.6</td><td>43.4</td><td>46.0</td></tr><tr><td>X=LongVA [38]</td><td>7B</td><td>32</td><td>60.9</td><td>49.3</td><td>44.0</td><td>51.4</td></tr><tr><td>X+Video-RAG [20]</td><td>7B</td><td>32</td><td>65.4</td><td>59.1</td><td>55.7</td><td>60.1</td></tr><tr><td>X+TV-RAG [1]</td><td>7B</td><td>32</td><td>66.2</td><td>62.1</td><td>58.7</td><td>61.7</td></tr><tr><td>X+EC-RAG</td><td>7B</td><td>32</td><td>67.1</td><td>62.5</td><td>58.9</td><td>62.8</td></tr></table>

![](images/2ab42aee047228172a5debd6897a881a68df1a5387c7934d91da2f80c6fbc146.jpg)  
Figure 5: Accuracy of LLaVA-Video-7B [40] with and without EC-RAG under diferent frame budgets on Video-MME [6]. EC-RAG provides consistent gains at all budgets, with the largest improvement at 16 frames.

## C More Qualitative Results

We present additional qualitative results of LLaVA-Video-7B [40] with EC-RAG on two representative examples from Video-MME [6] in Figures 6 and 7. Both cases involve temporal reasoning questions from long documentary videos, where the baseline model fails due to the inherent ambiguity of sparsely sampled frames. As illustrated, by constructing temporal event chains from multimodal evidence—particularly ASR transcripts that capture the chronological progression of topics and events—EC-RAG enables the model to accurately resolve temporal ordering and causal relationships that are indistinguishable from visual information alone. The results demonstrate that event-level retrieval efectively bridges the gap between sparse visual sampling and the dense temporal reasoning required for long video understanding.

Table 7: Per-task-type accuracy (%) on Video-MME [6] with LLaVA-Video-7B [40] as the backbone. Gain denotes the absolute improvement over the baseline. Best results per frame setting are in bold.
<table><tr><td>Method</td><td>Frames</td><td>TmpR</td><td>ActR</td><td>ActRec</td><td>AttrP</td><td>SpaP</td><td>SpaR</td><td>TmpP</td><td>InfoS</td><td>OCR</td><td>ObjR</td><td>ObjRec</td><td>Count</td><td>AVG</td><td>Gain</td></tr><tr><td>LLaVA-Video [40]</td><td>8</td><td>41.8</td><td>47.4</td><td>55.6</td><td>69.4</td><td>59.3</td><td>71.4</td><td>56.4</td><td>70.0</td><td>48.9</td><td>51.1</td><td>58.5</td><td>37.7</td><td>54.6</td><td></td></tr><tr><td>LLaVA-Video + EC-RAG</td><td>8</td><td>46.9</td><td>49.8</td><td>55.9</td><td>73.9</td><td>59.6</td><td>80.4</td><td>60.0</td><td>79.6</td><td>56.8</td><td>60.6</td><td>63.6</td><td>40.3</td><td>60.2</td><td>+5.6</td></tr><tr><td>LLaVA-Video [40]</td><td>16</td><td>45.2</td><td>51.2</td><td>57.5</td><td>68.0</td><td>57.4</td><td>73.2</td><td>69.1</td><td>74.0</td><td>50.4</td><td>55.4</td><td>64.1</td><td>41.8</td><td>58.0</td><td></td></tr><tr><td>LLaVA-Video + EC-RAG</td><td>16</td><td>51.4</td><td>64.9</td><td>62.6</td><td>78.4</td><td>61.1</td><td>80.4</td><td>69.6</td><td>83.9</td><td>59.0</td><td>66.1</td><td>67.2</td><td>42.3</td><td>65.8</td><td>+7.8</td></tr></table>

![](images/6ff73c1946926b038cf2633192eb6b823b1990c9d67f5b88cb4683b2f3597108.jpg)  
Figure 6: Qualitative example 1: a history documentary where the question requires reasoning about the temporal order of events after a key battle. The baseline confuses visually similar city depictions, while EC-RAG traces the event chain through ASR narration to identify the correct subsequent event.

![](images/0a52fd096024d92dedae2f288aac92e7bb997cc1d5b709edc36d89df994b1915.jpg)  
Figure 7: Qualitative example 2: a science documentary where the question asks about the introduction order of four topics. The baseline is misled by recurring visual elements across segments, while EC-RAG recovers the correct topic sequence from the structured event chain.