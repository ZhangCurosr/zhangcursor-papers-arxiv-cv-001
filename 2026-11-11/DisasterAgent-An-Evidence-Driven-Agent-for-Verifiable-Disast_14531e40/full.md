# DisasterAgent: An Evidence-Driven Agent for Verifiable Disaster Report Generation From Remote Sensing Imagery

Zhijian Li, Jiawei Song, Chao Ren, Yidan Zhang, Peijin Wang, Xian Sun, Senior Member, IEEE, and Kun Fu, Senior Member, IEEE

Project Page: https://zhijianli2003.github.io/DisasterAgent

Abstract—Remote sensing disaster assessment is moving from simple damage classification toward comprehensive report-level analysis. In this setting, responders need not only damage descriptions but also verifiable evidence to guide response and recovery. Recent vision-language models (VLMs) and tool-augmented agents offer strong multimodal reasoning and perception, but they struggle to provide the evidence required for high-stakes verification. To address these limitations, we propose DisasterAgent, an evidence-driven agent that uses a Perception-Augmentation-Generation workflow to generate verifiable disaster reports from bitemporal optical imagery. DisasterAgent first invokes a specialized vision expert toolkit in the perception phase to build a Spatiotemporal Perception Graph (STPG), serving as a structured memory to store and organize disaster-induced changes across space and time. Evidence-based Augmentation (EA) then extracts evidence from the STPG, attaching inline citations that form a closed evidence loop from each statement to the supporting visual evidence. Extensive experiments on the DisasterM3 and RSCC benchmarks demonstrate that DisasterAgent consistently outperforms general-domain VLMs, remote sensingspecialized models, and existing remote sensing agents. Notably, compared with the original model, it improves the average score for disaster report generation on DisasterM3 by 13.5% and ST5-SCS on RSCC by 15.34%, providing a more accurate and verifiable solution for disaster emergency response. The code will be available at https://github.com/ZhijianLi2003/DisasterAgent.

Index Terms—Disaster assessment, Verifiable disaster report generation, Vision-language models, Spatiotemporal perception enhancement, Perception-Augmentation-Generation workflow, Evidence-driven agent.

## I. INTRODUCTION

Remote sensing imagery has become an indispensable data source for rapid disaster assessment, owing to its extensive spatial coverage, frequent revisit capabilities, and timely capture of large-scale post-disaster impacts [1], [2]. In particular, bitemporal remote sensing images provide crucial visual evidence for identifying disaster-induced changes by comparing

![](images/ee775a28289d8d181556cc3ea14a7b5402ac3e1f4f76f72e07907e3f0c62a772.jpg)

![](images/fe1b8cd5bbe8214fad6bb15a9c88b3372aec052462d1cf5ba5ed874fa2d6dcc1.jpg)

Evaluator(O-5): Precision: 2 | Recall: 1 | Factual Correctness: 0  
![](images/fa995de534ed376ca53cb8d56e6af7378ee6e842b12814cf6c5ae88cd1914c26.jpg)  
Fig. 1. Comparison between standard VLM inference and the evidencegrounded reasoning of DisasterAgent. (a) A user query for a disaster impact caption from a bitemporal optical image pair. (b) An existing VLM generates an unverified report and misidentifies the disaster type. (c) DisasterAgent invokes perception tools and attaches inline citations to pixel-level visual evidence, mitigating hallucinations and producing more accurate captions.

pre- and post-event observations. Recent studies on remote sensing disaster assessment have achieved substantial progress in tasks such as damaged building detection, semantic change detection, and damage severity classification [3]–[5]. However, these methods typically formulate disaster assessment as a pixel-level or object-level perception problem, generating lowlevel structured outputs such as segmentation masks, bounding boxes, or predefined categorical labels [6].

Despite their effectiveness in quantitative damage mapping [7], [8], such perception-oriented methods remain insufficient for high-level decision-making in real-world emergency response. In practical disaster management [9], analysts and responders require more than just localized damage metrics; they often require comprehensive, human-readable reports that summarize the disaster type, spatial distribution of damage, affected infrastructure, environmental changes, and actionable recovery priorities [10], [11]. This requirement is especially critical in complex post-disaster scenarios where multiple land-cover categories, infrastructure systems, and secondary impacts intertwine. Consequently, remote sensing disaster assessment is moving from isolated visual perception toward high-level semantic reasoning and comprehensive report generation.

Recent advances in vision-language models (VLMs) provide a natural way to support this transition. By integrating visual perception with natural language reasoning, VLMs enable flexible interaction with remote sensing imagery, supporting tasks such as visual question answering, change captioning, and disaster report generation [12]–[14]. The emergence of specialized vision-language datasets [15]–[17] has further accelerated this trend by extending disaster assessment from conventional classification to language-based interpretation. Compared with traditional task-specific models, VLMs are highly flexible in handling open-ended queries and can generate textual descriptions that are more accessible to non-expert responders.

However, directly applying VLMs to remote sensing disaster report generation poses significant reliability risks. Most VLMs operate under an end-to-end generation paradigm, mapping visual and textual inputs directly to textual outputs without explicitly exposing the underlying visual evidence or reasoning process [13], [14]. This black-box nature makes it exceedingly difficult to verify whether a generated statement is genuinely supported by the imagery. Consequently, VLMs may produce hallucinated or weakly grounded descriptions, especially in complex post-disaster scenes where subtle spatial changes, small damaged objects, and visually similar disaster patterns can easily trigger fabricated descriptions. While finetuning VLMs on disaster-specific datasets [16], [17] can partially mitigate these issues, it remains a resource-intensive process that fails to resolve the fundamental lack of explicit evidence grounding. In high-stakes emergency response, such unverifiable generation can limit their practical reliability.

To improve grounding and verifiability, recent studies have explored tool-augmented agents [18], [19] for remote sensing analysis. These agents typically employ large language models (LLMs) or VLMs as cognitive planners to invoke external perception tools, such as segmentation or damage assessment models, thereby obtaining more fine-grained information than pure end-to-end VLMs. However, when applied to highstakes disaster assessment, existing remote sensing agents still face two key limitations. On the one hand, their capabilities are largely confined to elementary analytical tasks, such as single-image building counting or damage classification [20], [21], rather than the comprehensive spatiotemporal reasoning required for disaster report generation. On the other hand, their tool outputs are often organized as discrete intermediate results without a unified spatiotemporal structure. This disjointed execution paradigm fails to organize scattered visual information into a coherent reasoning chain, rendering the provenance of generated statements untraceable and making verification difficult.

To address the aforementioned challenges, we propose DisasterAgent, an evidence-driven agent for verifiable disaster report generation from bitemporal remote sensing imagery. As illustrated in Fig. 1, DisasterAgent organizes visual perception outputs into structured evidence and uses them to guide verifiable report generation, instead of relying on unconstrained end-to-end generation or fragmented tool-calling results. By anchoring generated statements to traceable spatiotemporal and historical evidence, DisasterAgent improves the factual reliability and verifiability of disaster reports.

The main contributions of this paper are summarized as follows:

• We propose DisasterAgent, an evidence-driven agent that uses a Perception-Augmentation-Generation workflow to perform comprehensive disaster assessment. The architecture enables pre-trained VLMs to serve as verifiable disaster report generators without fine-tuning for specific tasks, notably improving the average report generation score on the DisasterM3 benchmark by 13.5% and ST5- SCS on the RSCC benchmark by 15.34%.

• To accurately perceive changes caused by disasters, we construct a specialized vision expert toolkit tailored for disaster assessment and design the Spatiotemporal Perception Graph (STPG). Serving as a structured memory, the STPG systematically organizes fragmented tool outputs into coherent spatial relationships and temporal evolution edges across phases.

• To enhance the verifiability of the generated reports, we propose a multi-source Evidence-based Augmentation (EA) mechanism, built upon the STPG, that integrates semantic change claims, pixel-level spatial grounding, and historical experience. Inline citations form a closed evidence loop from each statement to the supporting visual evidence. By anchoring factual claims to this traceable evidence, the mechanism mitigates VLM hallucinations and better supports reliable emergency response.

## II. RELATED WORK

## A. Remote Sensing Disaster Assessment

Traditional remote sensing disaster assessment has predominantly been formulated as dense prediction or object detection tasks. The release of large-scale annotated datasets, such as ISBDA [3], xBD [4], and EBD [22], has driven the rapid development of deep learning methodologies in this domain. Researchers have extensively explored various architectures, including convolutional neural networks (CNNs) [5], [7], vision transformers (ViTs) [6], and state space models (SSMs) [8], to optimize bitemporal feature extraction and semantic change detection. While these methods achieve high precision in localizing damage and grading severity, they are inherently designed to output low-level visual formats (e.g., pixel-level masks).

Furthermore, conventional models are typically tailored for specific disaster types [7] or isolated tasks [15]. However, real-world disasters often trigger cascading effects that simultaneously impact diverse geographical targets, making it difficult for disjointed, single-task models to synthesize a holistic view of the disaster scene. Consequently, there is a pressing need for a unified framework capable of integrating heterogeneous perception results [23], [24] to perform highlevel semantic reasoning. Our work builds upon the robust perception capabilities of these specialized vision models but elevates the output to comprehensive, human-readable disaster reports.

## B. Vision-Language Models for Remote Sensing Disaster Understanding

To support multimodal interpretation in remote sensing, several specialized datasets have been curated. FloodNet [15] provides a Visual Question Answering (VQA) benchmark for post-hurricane damage assessment. RSCC [16] enables large-scale remote sensing change captioning. DisasterM3 [17] further introduces a comprehensive benchmark for multi-task disaster assessment and long-form report generation.

Concurrently, foundational general-domain VLMs (e.g., LLaVA [25], Qwen-VL [26], and InternVL [27]) have inspired the development of specialized remote sensing VLMs. Earth-GPT [12] processes diverse multi-sensor data, GeoChat [13] introduces region-level grounded dialogue in static images, and TEOChat [14] extends conversational reasoning to bitemporal earth observation data for change detection.

To adapt these models to disaster scenarios, recent approaches [16], [17], [28] predominantly rely on instruction fine-tuning. However, this end-to-end generation paradigm has two main limitations. First, it operates as a black box, generating text without revealing pixel-level grounding, making the outputs difficult to verify. Second, it lacks extensibility, as adapting to novel disaster typologies requires resourceintensive data curation and retraining. In contrast, Disaster-Agent employs a training-free, evidence-driven framework that decouples visual perception from language generation, ensuring that reports are explicitly grounded in verifiable evidence and easily extensible to new scenarios.

## C. Tool-Augmented Agents for Remote Sensing Analysis

Tool-augmented agents have emerged as a promising solution to extend the capabilities of LLMs and VLMs by invoking external expert models [18]. By extracting instancelevel or pixel-level grounding information, these agents extend their capabilities beyond pure text generation. For example, ThinkGeo [29] and OpenEarthAgent [30] act as overarching planners that invoke specialized temporal VLMs (such as TEOChat) to handle multi-temporal analysis. Similarly, RS-Agent [20] and EarthAgent [21] specifically target disaster scenarios by routing queries to deep learning models like ChangeOS to conduct precise building damage assessment and severity classification. Other systems explore diverse execution mechanisms: Change-Agent [31] enables interactive visual interpretation by dynamically calling proprietary segmentation tools; IC-EO [32] executes tool invocations by translating user queries into direct Python code; and CangLing-KnowFlow [33] streamlines execution by retrieving and applying predefined workflows from an established knowledge base. However, these existing agents typically rely on linear toolcalling sequences, resulting in isolated intermediate outputs that are difficult to synthesize for complex disaster assessment. DisasterAgent distinguishes itself by introducing the STPG, which systematically organizes fragmented tool outputs into a unified, spatiotemporal-aware structure to guide coherent and verifiable report generation.

## III. METHODOLOGY

## A. Overview of DisasterAgent

DisasterAgent is a skill-conditioned, evidence-driven agent that repurposes a single pre-trained VLM into a verifiable disaster report generator without the need for task-specific fine-tuning. As illustrated in Fig. 2, the system takes a bitemporal pair of remote sensing images (pre- and postdisaster) and a free-form natural language query as input. It processes these inputs through two primary stages. First, a Skill Routing phase dynamically selects the appropriate analytical skill and registers its required tools. Second, a Skill-Guided Workflow sequentially executes Spatiotemporal Change Perception, Evidence-based Augmentation, and Evidence-Fused Verifiable Generation. Ultimately, the system produces a comprehensive disaster report along with a list of citable references, grounding the generated statements in concrete visual and semantic evidence. The design of this three-step workflow is conceptually inspired by the Retrieval-Augmented Generation (RAG) paradigm [34], [35] prevalent in natural language processing (NLP). However, since remote sensing analysis fundamentally relies on precise visual grounding and spatiotemporal awareness rather than mere text retrieval, we tailor this concept to the unique demands of Earth Observation. By evolving the traditional RAG pipeline into a Perception-Augmentation-Generation architecture, the framework establishes a structured, spatiotemporally aware reasoning process driven entirely by a central VLM. The entire workflow is orchestrated as a multi-stage process over a shared state:

1) Skill Routing: The VLM inspects the input query and images to select the most appropriate skill from a Skill Repository (e.g., disaster change description, disaster report generation, or disaster restoration advice). This skill then registers a specialized toolkit comprising spatial, temporal, and damage assessment tools necessary for subsequent analysis.

2) Spatiotemporal Change Perception: Guided by the selected skill, a Plan-and-Execute agent takes over, using function calling to systematically invoke the registered tools. It extracts information to instantiate a Spatiotemporal Perception Graph (STPG), which jointly captures pre-disaster structures, post-disaster structures, and their cross-phase evolution edges.

![](images/41bb505abeb18a3b3c284258e285b6cd2845e1b615ffabb5d5d9eb68dc7de25d.jpg)  
Fig. 2. The overall architecture of DisasterAgent. The framework processes multimodal inputs through two primary stages. (1) Skill Routing: Driven by the VLM’s interpretation of the query, the system dynamically routes to appropriate skills within a pre-defined repository and seamlessly registers the required functional tools. (2) Skill-Guided Workflow: Guided by the registered skills, the framework sequentially executes a three-step pipeline comprising Perception, Evidence-based Augmentation, and Generation.

3) Evidence-based Augmentation: The STPG (serialized as a perception brief and individual graph nodes) drives the extraction of three complementary evidence streams:

• Experts-Experience Evidence (EEE): Retrieves the top-ranked historical disaster case data off-line via multimodal embedding similarity search, providing domain-specific priors and restoration guidance.

• Visual-Grounding Evidence (VGE): Renders precise spatial annotations on-line, such as bounding boxes for buildings and polygons for other regions, directly onto the images.

• Semantic-Changing Evidence (SCE): Reasons over the STPG on-line to generate atomic spatiotemporal claims. Each claim is independently verified by the VLM to obtain a confidence score, filtering out hallucinations and yielding robust changing evidence.

4) Evidence-Fused Verifiable Generation: Finally, the VLM fuses the experience evidence, grounding evidence, and changing evidence to generate the final response. The output includes the comprehensive disaster report with inline citations and a corresponding reference list that maps each citation back to the verified evidence, ensuring the entire generation process is fully verifiable.

## B. Spatiotemporal Change Perception

1) Spatiotemporal Perception Graph: To move beyond unstructured raw pixels while preserving spatially and temporally grounded evidence, we construct the STPG to represent the disaster scene. A key purpose of the STPG is to structurally store the outputs from various agent tool calls, facilitating the verification of low-level visual information during the reasoning process. As illustrated in Fig. 3, the STPG systematically organizes the visual elements into single-phase perception graphs (representing the pre- and post-disaster spatial structures) and cross-phase evolution edges (capturing the temporal changes). Formally, the STPG is defined as:

$$
\mathcal { G } _ { \mathrm { S T P G } } = \left( \mathcal { G } _ { \mathrm { T 1 } } , \mathcal { G } _ { \mathrm { T 2 } } , \mathcal { E } _ { \mathrm { e v o } } \right) ,\tag{1}
$$

where $\mathcal { G } _ { \mathrm { T 1 } }$ and $\mathcal { G } _ { \mathrm { T 2 } }$ denote the spatial perception graphs for the pre-disaster (Time 1) and post-disaster (Time 2) images, respectively, and ${ \mathcal E } _ { \mathrm { e v o } }$ is the set of temporal evolution edges connecting them. Notably, this formulation is highly extensible and can be naturally generalized to multi-temporal scenarios by chaining additional temporal graphs and evolution edges, $\mathbf { e . g . } , ( \mathcal { G } _ { \mathrm { T 1 } } , \mathcal { \bar { E } } _ { \mathrm { e v o } } ^ { ( 1 ) } , \mathcal { G } _ { \mathrm { T 2 } } , \dotsc , \mathcal { E } _ { \mathrm { e v o } } ^ { ( \hat { n } - 1 ) } , \mathcal { \bar { G } } _ { \mathrm { T n } } )$

![](images/18c8ef306e574bb2e764bef4a84f5ef72aac914d2cd551340b13636c6de1977e.jpg)  
Fig. 3. Illustration of the STPG. (a) The Spatial Perception Graph constructs the spatial relationships between diverse geographical targets (e.g., buildings, roads, forests, farmland, and water) within a single phase. (b) The Temporal Evolution Edges establish the cross-phase mappings, explicitly capturing state transitions such as “maintained”, “shrunk”, “expanded”, “transitioned to”, “disappeared”, and “appeared”, while integrating domain-specific damage level assessments (e.g., building damage, road damage).

Spatial Perception Graph. For a single phase (either preor post-disaster), the graph $\mathcal { G } _ { * } = ( \nu _ { * } , \mathcal { R } _ { * } )$ consists of region nodes and their spatial relation edges. Each node $v _ { i } \in \mathcal V _ { * }$ is defined as:

$$
v _ { i } = \big ( i d _ { i } , \ell _ { i } , b _ { i } , m _ { i } , \sigma _ { i } , \phi _ { i } \big ) ,\tag{2}
$$

where $i d _ { i }$ is a unique identifier of the region node, $\ell _ { i }$ is the normalized semantic label $( \mathrm { e . g . }$ ., building, road, farmland, water), $b _ { i } ~ = ~ ( x , y , w , h )$ is the axis-aligned bounding box, $m _ { i }$ represents the COCO-RLE-encoded segmentation mask, $\sigma _ { i } ~ \in ~ [ 0 , 1 ]$ is the detection confidence score, and $\phi _ { i }$ is an extensible property dictionary for storing damage labels. We establish directed edges to connect each region to its k-nearest neighbors based on spatial proximity. In our implementation, we set k = 3. The relation edge $r _ { i j } \in \mathcal { R }$ <sub>∗</sub> is formulated as:

$$
r _ { i j } = \big ( v _ { i } , \rho ( v _ { i } , v _ { j } ) , v _ { j } \big ) , \quad \rho = \mathcal { R } _ { \theta } \big ( \angle ( p _ { j } - p _ { i } ) \big ) ,\tag{3}
$$

where $p _ { i }$ and $p _ { j }$ denote the geometric centers of regions $v _ { i }$ and $v _ { j }$ respectively, and $\mathcal { R } _ { \theta } ( \cdot )$ quantizes the angle $\angle ( p _ { j } - p _ { i } )$ between them into 8 discrete spatial directions (e.g., right, upper-right, above).

Temporal Evolution Edge. To establish accurate temporal mappings across phases, we employ a mask-based region Intersection over Union (IoU) operator:

$$
\mathrm { I o U } ( v _ { i } , v _ { j } ) = { \frac { | m _ { i } \cap m _ { j } | } { | m _ { i } \cup m _ { j } | } } .\tag{4}
$$

Similarly, the area of the mask region corresponding to node $v _ { i }$ is $| v _ { i } | = | m _ { i } |$ . For each region $v _ { i } \in \mathbb { V } _ { \mathrm { T 1 } }$ in the pre-disaster graph $\mathcal { G } _ { \mathrm { T 1 } }$ , we determine its optimal matching region $v _ { i } ^ { \star }$ in the post-disaster graph G<sub>T2</sub>:

$$
v _ { i } ^ { \star } = \arg \operatorname* { m a x } _ { v _ { j } \in \mathcal { V } _ { \mathrm { T 2 } } } \mathrm { I o U } \left( v _ { i } , v _ { j } \right) , \quad u _ { i } = \mathrm { I o U } \left( v _ { i } , v _ { i } ^ { \star } \right) .\tag{5}
$$

We define the area-change ratio as $\alpha _ { i } ~ = ~ | \boldsymbol { v } _ { i } ^ { \star } | / | \boldsymbol { v } _ { i } |$ . The evolution type $\eta _ { i }$ for the edge connecting $v _ { i }$ and $v _ { i } ^ { \star }$ is assigned according to the following piecewise rule:

$$
\eta _ { i } = \left\{ \begin{array} { l l } { D i s a p p e a r e d , } & { u _ { i } < \tau _ { \mathrm { I o U } } , } \\ { T r a n s i t i o n e d ~ t o , } & { \left( u _ { i } \geq \tau _ { \mathrm { I o U } } \right) \wedge \left( \ell _ { i } \neq \ell _ { i } ^ { \star } \right) , } \\ { E x p a n d e d , } & { \left( u _ { i } \geq \tau _ { \mathrm { I o U } } \right) \wedge \left( \ell _ { i } = \ell _ { i } ^ { \star } \right) \wedge \left( \alpha _ { i } > \tau _ { \uparrow } \right) , } \\ { S h r u n k , } & { \left( u _ { i } \geq \tau _ { \mathrm { I o U } } \right) \wedge \left( \ell _ { i } = \ell _ { i } ^ { \star } \right) \wedge \left( \alpha _ { i } < \tau _ { \downarrow } \right) , } \\ { M a i n t a i n e d , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{6}
$$

where $\ell _ { i } ^ { \star }$ denotes the semantic label of $v _ { i } ^ { \star }$ , and $\tau _ { \mathrm { I o U } } , \ \tau _ { \uparrow } ,$ , and $\tau _ { \downarrow }$ are predefined thresholds (set to $\tau _ { \mathrm { I o U } } = 0 . 3 , \tau _ { \uparrow } = 1 . 5 .$ , and $\tau _ { \downarrow } = 0 . 6 6 7$ by default). Regions in $\mathcal { G } _ { \mathrm { T 2 } }$ that fail to match any region in $\mathcal { G } _ { \mathrm { T 1 } } \ ( \mathrm { i . e . }$ , max<sub>i</sub> IoU $\left( v _ { i } , v _ { j } \right) \ : < \ : \tau _ { \mathrm { I o U } } )$ are designated as Appeared. The resulting cross-phase edge set ${ \mathcal E } _ { \mathrm { e v o } }$ thus explicitly encodes temporal dynamics such as existence, identity, and geometry transitions, providing foundational evidence for downstream reasoning.

2) Tool-augmented Visual Perception: Building the aforementioned G requires fusing heterogeneous pixel-level vision models. To achieve this in a decoupled and extensible manner, we delegate the execution to a Plan-and-Execute agent. In this paradigm, the VLM acts purely as a planner, while a sandbox Python runtime acts as the executor.

Given the task description and the skill configuration, the planner emits a structured JSON tool-call plan:

$$
\pi = { \big [ } ( t _ { 1 } , \mathbf { a } _ { 1 } ) , ( t _ { 2 } , \mathbf { a } _ { 2 } ) , \ldots { \big ] } ,\tag{7}
$$

where each step references a registered tool $t _ { k }$ with arguments ${ \bf a } _ { k } .$ . A critical feature of this orchestration strategy is the cross-step data flow expressed via symbolic references $( \mathsf { e } . \mathsf { g } . , \mathsf { N } \hbar \mathsf { S } \mathsf { s t e p } _ { - } 3 " )$ . This allows the STPG to be sequentially updated and threaded through successive tool calls without the VLM needing to handle raw intermediate data. The runtime then executes the plan deterministically, captures all return values and errors, and serializes both the plan and the execution log for later verification.

The registered toolkit is categorized into four functional families: 1) Spatial tools ingest open-vocabulary instancelevel segmentation results to instantiate region nodes in $\mathcal { G } _ { \mathrm { T 1 } }$ and $\mathcal { G } _ { \mathrm { T 2 } } ; ~ 2 )$ Temporal tools compute mask-based IoU to construct the cross-phase evolution edges; 3) Damage tools extract specific disaster impacts (e.g., building damage, road submersion) and project these semantic masks onto the STPG via pixel-voting to update node properties $\phi _ { i } ;$ and 4) Statistics tools aggregate the graph content into compact numerical or textual summaries to facilitate downstream VQA-style reasoning.

TABLE I  
VISION EXPERT TOOLS FOR SPATIOTEMPORAL CHANGE PERCEPTION
<table><tr><td>Vision Expert Tool</td><td>Function Description</td></tr><tr><td>Disaster Type Recognition [36]</td><td>Built upon DINOv3, this tool identifies the disaster type from bitemporal remote sensing images and supports 10 representative disaster categories, including earthquakes, floods, hurricanes, volcanoes, explosions, and conflicts.</td></tr><tr><td>Building Damage Analysis (4-class) [7]</td><td>Implemented with ChangeOS, this tool performs pixel-level building damage assessment based on the Joint Damage Scale, which integrates FEMA and HAZUS standards, and categorizes building conditions into four severity levels: no-damage, minor-damage, major-damage, and destroyed.</td></tr><tr><td>Building Damage Analysis (3-class) [8]</td><td>Based on ChangeMamba, this tool provides pixel-level structural damage perception following FEMA- oriented damage definitions, classifying buildings into three levels: intact, damaged, and destroyed.</td></tr><tr><td>Road Damage Analysis [8]</td><td>Powered by ChangeMamba, this tool extracts pixel-level road damage information from pre- and post-disaster imagery, distinguishing undamaged roads, flood-covered roads, and roads damaged by debris or rubble.</td></tr><tr><td>General Concept Segmentation [37]</td><td>Integrated with SAM3, this tool enables open-vocabulary object detection and segmentation for general disaster-related concepts, such as farmland, forest, water bodies, and other scene elements, supporting zero-shot generalization across diverse disaster scenarios.</td></tr></table>

The specific vision expert models underlying these tools are detailed in Table I. To obtain more accurate pixel-level damage information, some of these tools are pre-trained on the DisasterM3-Instruct and xBD datasets. Because the VLM only plans and never executes the tools directly, tool injection is fully sandboxed. This design ensures that the agent can generalize to new skills or disaster queries simply by modifying the configuration prompt, without requiring any changes to the underlying tool implementations. The designs of the disaster type recognition and building damage assessment tools are provided in Appendix I, as shown in Fig. 10.

## C. Evidence-based Augmentation

To ensure the generated disaster reports are grounded and verifiable, we design three complementary evidence streams (as depicted in the Evidence-based Augmentation block of Fig. 2). These evidence modules are operationally divided into on-line extraction and off-line retrieval mechanisms. Online, the framework operates directly on the STPG to dynamically extract SCE and VGE. Conceptually, SCE encodes what changed and why we believe so at a high semantic level, while VGE provides precise spatial localization, revealing where it changed at the pixel level. Off-line, EEE retrieves domain-specific historical priors from a pre-constructed knowledge base, demonstrating how similar past events were described and remediated. Together, they form a robust, multidimensional evidence base for the subsequent report generation process.

1) On-line Semantic-Changing Evidence Reasoning: As illustrated in Fig. 4, SCE follows a reasoning → verification → filtering pipeline to extract semantic change information from the STPG. This process is divided into three steps:

• Reasoning: The central VLM reads the Perception Brief, which contains the disaster type, evolution edges, and damage labels, to emit a structured Evidence Chain. This chain consists of (i) a set of atomic, verifiable claims $\mathcal { C } = \{ c _ { k } \}$ in natural language (e.g., “The river expanded and submerged adjacent farmland.”), and (ii) a list of evidence references for each claim that cites specific STPG entities (such as evolution edges or specific region nodes). Consequently, rather than generating free-form text, this reasoning step establishes a structured mapping where each claim $c _ { k } ~ \in ~ \mathcal { C }$ is explicitly linked to its supporting STPG evidence references. This structured relationship is then serialized and stored in JSON format, facilitating subsequent verification.

• Verification: Although the claims derived from the reasoning step originate from the perception of visual expert tools and are relatively reliable, they may still contain inaccuracies due to upstream perception errors. To prevent the VLM from simply rubber-stamping these claims, an independent verification mechanism is introduced. This design is motivated by the observation that in complex remote sensing image interpretation tasks, VLMs generally exhibit stronger and more reliable performance as discriminators (i.e., verifying existing claims) than as free-form generators [38]–[40]. For every unverified claim $c _ { k } .$ , a separate VLM call is executed using only the raw image pair $( I _ { \mathrm { T } 1 } , I _ { \mathrm { T } 2 } )$ and the claim $c _ { k }$ as input. Acting as a strict visual auditor, the model returns a confidence score. The returned scalar is written back into the claim node:

$$
\mathrm { c o n f } ( c _ { k } ) \gets \mathrm { V L M } _ { \mathrm { v e r i f y } } ( c _ { k } \mid I _ { \mathrm { T } 1 } , I _ { \mathrm { T } 2 } ) .\tag{8}
$$

• Filtering: Claims satisfying ${ \mathrm { c o n f } } ( c _ { k } ) ~ \geq ~ \tau _ { \mathrm { { c o n f } } }$ (where $\tau _ { \mathrm { c o n f } } ~ = ~ 0 . 7$ by default) are retained to form the final semantic-changing evidence set $\mathcal { E } _ { \mathrm { S C E } }$ . Formally, this set is defined as:

$$
\mathcal { E } _ { \mathrm { S C E } } = \Big \{ c _ { k } \Big | \mathrm { c o n f } ( c _ { k } ) \geq \tau _ { \mathrm { c o n f } } , c _ { k } \in \mathcal { C } \Big \} .\tag{9}
$$

Each surviving claim then serves as a citable reference during the final report generation.

![](images/9ec6b78d064407a89d7fc93b063e8cc99a132ca30d870faffa21d1ceb7f62ea2.jpg)  
Fig. 4. Illustration of the SCE extraction pipeline. The VLM first reasons over the STPG to generate atomic factual claims. Each claim is then independentl verified against the raw bitemporal images to obtain a confidence score. Finally, claims with scores above the threshold are filtered to form the robust SCE set.

This design transforms a monolithic black-box VLM inference into a self-auditing reasoning loop. Only the claims that simultaneously (i) align with the structured STPG and (ii) pass the independent VLM evaluation with a confidence score above the threshold are permitted to enter the final generation stage.

2) On-line Visual-Grounding Evidence Extraction: While SCE supplies global, semantic-level evidence, VGE provides fine-grained, pixel-level evidence by directly rendering the STPG content onto the original images. Specifically, for every region in the spatial perception graphs $\mathcal { G } _ { \mathrm { T 1 } }$ and $\mathcal { G } _ { \mathrm { T 2 } }$ that carries a damage label, we generate precise spatial annotations. For example, buildings are annotated with axis-aligned bounding boxes, while roads and other irregular regions are delineated with polygon contours. The colors of these annotations intuitively encode their respective damage severity (e.g., intact, damaged, or destroyed).

The resulting annotated images, alongside an automatically generated color legend, are subsequently integrated into the VLM’s multimodal context. Formally, the visual-grounding evidence set $\mathcal { E } _ { \mathrm { V G E } }$ is defined as the union of the visually rendered region nodes and their corresponding textual legend $L \colon$

$$
{ \mathcal { E } } _ { \mathrm { V G E } } = \left\{ { \mathrm { R e n d e r } } ( v _ { i } ) ~ { \Big | } ~ v _ { i } \in \mathcal { V } _ { \mathrm { T 1 } } \cup \mathcal { V } _ { \mathrm { T 2 } } \right\} \cup \{ L \} ,\tag{10}
$$

where Render(·) denotes the operation of drawing the bounding box $b _ { i }$ or polygon mask $m _ { i }$ of region $v _ { i }$ onto the image using a color mapped from its damage property $\phi _ { i }$ , and L maps each color to its specific semantic meaning (e.g., “Red: Destroyed Building”). This explicit visual grounding mechanism enables the model to directly attend to the structured perception outcomes anchored to specific instances. Consequently, VGE functions as a highly precise spatial evidence stream, which is particularly crucial for substantiating finegrained damage quantities and spatial distribution patterns within the generated disaster report.

3) Off-line Experts-Experience Evidence Retrieval: Inspired by the practice of emergency-response analysts who consult historical casebooks when assessing a new event, this module supplements the agent’s generation process with external domain expertise. By retrieving historical disaster cases relevant to the current visual findings, the system is provided with essential remote sensing disaster assessment priors that VLMs may not have encountered during their general pre-training phase. Furthermore, compared to taskspecific VLM fine-tuning, this retrieval-based approach offers superior extensibility, as it allows the knowledge base to be updated dynamically without retraining the underlying model.

We construct the source corpus using the training data from the DisasterM3-Instruct dataset. The initial corpus is denoted as:

$$
\begin{array} { r } { \mathcal { S } = \left\{ \left( I _ { \mathrm { T 1 } } ^ { ( j ) } , I _ { \mathrm { T 2 } } ^ { ( j ) } , y _ { j } , D _ { j } , R _ { j } \right) \right\} _ { j = 1 } ^ { N } , } \end{array}\tag{11}
$$

where N is the total number of training cases, $I _ { \mathrm { T 1 } } ^ { ( j ) }$ and $I _ { \mathrm { T } 2 } ^ { ( j ) }$ are the pre- and post-disaster images of the j-th case, $y _ { j }$ is its disaster-type label, and $D _ { j } , R _ { j }$ represent the corresponding expert-verified disaster description and restoration advice, respectively. To ensure efficient retrieval and maintain diversity, we first apply Term Frequency-Inverse Document Frequency (TF-IDF) [41] vectorization on the concatenated text $D _ { j }$ ⊕ $R _ { j }$ for each case in S. Then, for each unique disaster type y, we perform K-means clustering on the corresponding sub-corpus $\mathcal { S } _ { y }$ to obtain M = 50 cluster centroids:

$$
\operatorname* { m i n } _ { \{ \mu _ { k } \} _ { k = 1 } ^ { M } } \sum _ { j \in S _ { y } } \operatorname* { m i n } _ { 1 \leq k \leq M } \big \| \mathrm { T F } \mathrm { - I D F } ( D _ { j } \oplus R _ { j } ) - \pmb { \mu } _ { k } \big \| ^ { 2 } .\tag{12}
$$

By sampling the centroid-nearest case from each cluster, we obtain a lexically and semantically diverse knowledge base

B comprising 50 representative historical cases per disaster type. In this retrieval framework, the multimodal inputs $( I _ { \mathrm { T 1 } } ^ { ( j ) } , I _ { \mathrm { T 2 } } ^ { ( j ) } , y _ { j } )$ act as the query keys, while the textual descriptions $( D _ { j } , R _ { j } )$ serve as the corresponding values to be retrieved.

We employ the Qwen3-VL-Embedding [42] model, denoted as $E _ { \mathrm { m m } } ( \cdot )$ , to extract representations for the query keys. Conditioned on a specific retrieval instruction $\mathcal { T } ^ { * }$ , the off-line embedding $\mathbf { e } _ { j }$ for each historical case in $\boldsymbol { B }$ is calculated as:

$$
\mathbf { e } _ { j } = E _ { \mathrm { m m } } \left( I _ { \mathrm { T 1 } } ^ { ( j ) } \oplus I _ { \mathrm { T 2 } } ^ { ( j ) } \oplus y _ { j } \mid { \mathcal { T } } ^ { * } \right) ,\tag{13}
$$

where ⊕ denotes multimodal sequence concatenation.

During inference, the agent dynamically identifies the disaster type $\hat { y }$ via the previously constructed STPG. The query embedding $\mathbf { q }$ is generated on-line following the same embedding procedure:

$$
\mathbf { q } = E _ { \mathrm { m m } } \big ( I _ { \mathrm { T 1 } } \oplus I _ { \mathrm { T 2 } } \oplus \hat { y } \big | \mathcal { T } ^ { * } \big ) .\tag{14}
$$

The cosine similarity score $s _ { j }$ between the query and each historical case in the knowledge base is computed as:

$$
s _ { j } = { \frac { \mathbf { q } \cdot \mathbf { e } _ { j } } { \| \mathbf { q } \| \| \mathbf { e } _ { j } \| } } , \quad \forall j \in B .\tag{15}
$$

Finally, the evidence set $\mathcal { E } _ { \mathrm { E E E } }$ is constructed by retrieving the top-K most relevant historical text values based on $s _ { j } { \mathrm { : } }$

$$
\begin{array} { r } { \mathcal { E } _ { \mathrm { E E E } } = \Big \{ ( D _ { j } , R _ { j } ) \Big | \operatorname { r a n k } \big ( s _ { j } \big ) \leq K , j \in \mathcal { B } \Big \} , } \end{array}\tag{16}
$$

where rank(·) returns the descending rank of the similarity score in the candidate pool. By default, K is set to 3. The retrieved descriptions and advice are subsequently split and injected into the multi-stage generation prompt, ensuring that historical descriptions inform factual disaster captioning while historical advice provides validated restoration recommendations.

## D. Evidence-Fused Verifiable Generation

The final generation stage fuses the three evidence streams $( \mathcal { E } _ { \mathrm { S C E } } , \mathcal { E } _ { \mathrm { V G E } } , \mathcal { E } _ { \mathrm { E E E } } )$ with the raw image pair $( I _ { \mathrm { T } 1 } , I _ { \mathrm { T } 2 } )$ to produce a comprehensive and structured disaster report. As illustrated in Fig. 5, to ensure the resulting report is highly verifiable, we introduce two key design mechanisms: a cascaded two-stage generation architecture and an explicit perclaim citation system.

Cascaded Two-Stage Generation. Rather than generating the entire disaster report in a single, monolithic VLM call, we decompose the generation process into two cascaded stages. This design mirrors the cognitive process of human responders: first describing the factual impacts, then prescribing subsequent recovery actions:

$$
R _ { \mathrm { d e s c } } = \mathrm { V L M } \big ( I _ { \mathrm { T 1 } } , I _ { \mathrm { T 2 } } , \mathcal { E } _ { \mathrm { S C E } } , \mathcal { E } _ { \mathrm { V G E } } , \mathcal { E } _ { \mathrm { E E E } } ^ { \mathrm { d e s c } } \big ) ,\tag{17}
$$

$$
R _ { \mathrm { r e s } } = \mathrm { V L M } ( I _ { \mathrm { T 1 } } , I _ { \mathrm { T 2 } } , R _ { \mathrm { d e s c } } , \mathcal { E } _ { \mathrm { E E E } } ^ { \mathrm { r e s } } ) ,\tag{18}
$$

where E<sup>desc</sup><sub>EEE</sub> and $\mathcal { E } _ { \mathrm { E E E } } ^ { \mathrm { r e s } }$ represent the historical disaster descriptions and restoration advice fields retrieved from $\mathcal { E } _ { \mathrm { E E E } }$ respectively.

![](images/251386ba9731c1b81a53bf9b2f8e5b9018b4e5ed03f780943f08f4fbf124029f.jpg)  
Fig. 5. Illustration of the Evidence-Fused Verifiable Generation. The framework employs a cascaded two-stage generation process: Stage 1 generates the disaster description grounded in all three evidence streams $( \mathcal { E } _ { \mathrm { S C E } } ,$ E<sub>VGE</sub>, E<sup>desc</sup>), while Stage 2 provides restoration advice conditioned exclusively on the generated description and the retrieved historical advice (E<sup>res</sup> ). Through explicit inline citations (e.g., [1], [2]) mapped to a reference list, the system establishes a closed, traceable evidence loop connecting the final report back to the raw pixel regions.

In Stage 1 (17), the VLM consumes all three evidence streams to ground the factual disaster description $R _ { \mathrm { d e s c } }$ in finegrained visual details and verified semantic claims. Conversely, Stage 2 (18) is deliberately evidence-restricted: the SCE and VGE streams are switched off. Instead, the newly generated $R _ { \mathrm { d e s c } }$ serves as the factual anchor, and only the historical advice field $\mathcal { E } _ { \mathrm { E E E } } ^ { \mathrm { r e s } }$ is injected. This cascaded design yields significant benefits: (i) the restoration advice is strictly conditioned on a committed description supported by citations, effectively eliminating the risk of hallucinating fabricated damage during the recommendation phase; and (ii) the recommendations are guided by historically validated actions tailored to the current disaster type, rather than being inferred directly from raw pixels. This implements an explicit chain of thought at the system level.

Traceable Evidence Loop. In general-domain text generation, linking each statement to supporting evidence by citation is an effective way to make generated text verifiable [43]– [45]. To establish this provenance for disaster reports (shown at the bottom of Fig. 5), the generation prompt is explicitly instructed to cite the indices of the supporting SCE claim when generating the report. Specifically, the model receives a formatted evidence block where claims are enumerated as [i] <text> (confidence score: 0-1). The runtime automatically serializes the generated texts alongside a references list that maps every cited index back to the original claim text, confidence score, and underlying STPG entities. Consequently, human auditors can traverse this closed evidence loop in O(1) complexity per claim to verify the factual accuracy of any statement within the report.

## IV. EXPERIMENTS

To comprehensively evaluate the proposed DisasterAgent, we first introduce the benchmark datasets, evaluation metrics, and implementation details. We then present a quantitative performance comparison against existing VLMs and agent frameworks on two remote sensing disaster benchmarks. Finally, we conduct detailed ablation studies to validate our architectural designs and provide qualitative analyses to intuitively illustrate how DisasterAgent mitigates hallucinations and establishes a traceable evidence loop.

TABLE II  
COMPARISON OF BENCHMARK DATASETS USED FOR EVALUATION
<table><tr><td colspan="2">RSCC [16]</td><td colspan="2">DisasterM3 [17]</td></tr><tr><td>Tasks</td><td>Change Captioning</td><td>Disaster Type Recognition Building Damage Counting Disaster Region Estimation Disaster Captioning Restoration Advice</td></tr><tr><td>Disaster Types</td><td>Earthquake, Flood Wildfire, Volcano Storm, Tsunami</td><td>Earthquake, Flood Hurricane, Volcano Explosion, Wildfire Tornado, Tsunami</td></tr><tr><td>Modality</td><td>Optical</td><td>Optical</td></tr><tr><td>Test Scale</td><td>988 image pairs</td><td>2,512 image pairs</td></tr></table>

## A. Datasets

We evaluate DisasterAgent on two complementary bitemporal remote sensing disaster benchmarks: RSCC [16] and DisasterM3 [17]. Specifically, RSCC is utilized to examine fine-grained change captioning, while DisasterM3 provides a broader scope for comprehensive disaster assessment. A detailed comparison of these two datasets is provided in Table II.

RSCC: RSCC is a large scale change caption benchmark designed for understanding bitemporal images in disaster scenarios. Each sample consists of images taken before and after a disaster, along with a reference caption describing the corresponding scene changes. Compared with conventional caption datasets that focus on single image descriptions or generic land cover changes, RSCC emphasizes temporal transformations caused by disasters, including damaged buildings, inundated areas, vegetation loss, and infrastructure disruption. Following the official evaluation protocol, we evaluate DisasterAgent on the RSCC test benchmark, which contains 988 image pairs. This benchmark assesses whether the model can accurately capture and describe semantic changes from bitemporal observations.

DisasterM3: DisasterM3 is a comprehensive vision and language benchmark encompassing diverse hazards, sensors, and tasks for damage assessment. It contains 26,988 bitemporal satellite images and 123,010 instruction pairs collected from 36 historical events across five continents, covering ten common natural and anthropogenic disaster types. Unlike RSCC, DisasterM3 provides a broader scope for structured assessment and report generation. Its task taxonomy progresses from basic recognition and quantitative damage assessment to long form disaster description and restoration advice generation. In our experiments, we utilize the official DisasterM3 Bench set under the purely optical setting, focusing on the tasks most relevant to structured report generation.

TABLE III  
EVALUATION DIMENSIONS FOR COMPREHENSIVE REPORT GENERATION
<table><tr><td>Task</td><td>Evaluation Dimension</td><td>Score</td><td>Abbr.</td></tr><tr><td rowspan="3">Disaster Captioning</td><td>Damage Assessment Precision</td><td> $0 \sim 5$ </td><td>DAP</td></tr><tr><td>Damage Detail Recall</td><td> $0 \sim 5$ </td><td>DDR</td></tr><tr><td>Factual Correctness</td><td> $0 \sim 5$ </td><td>FC</td></tr><tr><td rowspan="3">Restoration Advice</td><td>Recovery Necessity</td><td> $0 \sim 5$ </td><td>RN</td></tr><tr><td>Action Priority Precision</td><td> $0 \sim 5$ </td><td>APP</td></tr><tr><td>Strategic Completeness</td><td> $0 \sim 5$ </td><td>SC</td></tr></table>

## B. Evaluation Metrics

We adopt task specific evaluation metrics for the two benchmarks, reflecting the different objectives of change captioning and comprehensive disaster report generation.

Metrics on RSCC: For the RSCC benchmark, we evaluate the generated change captions by comparing them with the reference captions using both lexical-overlap and semanticsimilarity metrics. Specifically, we report ROUGE-L [46] and METEOR [47] as classical n-gram-based metrics. ROUGE-L measures the longest common subsequence between the generated and reference captions, while METEOR considers unigram matching with additional alignment and synonymaware matching mechanisms. Since disaster reports can be semantically correct even when their wording differs from the reference caption, we also report ST5-SCS [16], an embedding-based contextual similarity metric. Following the RSCC benchmark, ST5-SCS encodes the generated and reference captions with Sentence-T5 [48] and computes their sharpened cosine similarity. The formula is defined as:

$$
\mathrm { S T 5 - S C S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \frac { f _ { \boldsymbol { \theta } } ( \boldsymbol { \hat { r } } _ { i } ) ^ { \top } f _ { \boldsymbol { \theta } } ( \boldsymbol { r } _ { i } ) } { \| f _ { \boldsymbol { \theta } } ( \boldsymbol { \hat { r } } _ { i } ) \| _ { 2 } \| f _ { \boldsymbol { \theta } } ( \boldsymbol { r } _ { i } ) \| _ { 2 } } \right| ^ { 3 } ,\tag{19}
$$

where N is the total number of evaluated samples, $\hat { r } _ { i }$ and $r _ { i }$ represent the generated caption and the reference caption for the i-th sample, respectively, and $f _ { \theta }$ denotes the Sentence-T5-XXL text encoder. This metric is used to better capture semantic consistency in comprehensive disaster descriptions.

Metrics on DisasterM3: For the DisasterM3 benchmark, we report classification Accuracy (ACC) for Disaster Type Recognition (DTR). Unlike the original multiple choice setup, we formulate Building Damage Counting (BDC) and Disaster Region Estimation (DRE) as open ended numerical generation tasks to better reflect real world emergency response scenarios, where predefined candidate options are typically unavailable. We report the Mean Absolute Error (MAE) between the predicted and ground truth values for these tasks. For comprehensive report generation, we strictly adopt the official evaluation protocol and score the outputs with the official five point $( 0 \sim 5 )$ rubric [17], evaluated by an LLM as a judge. The specific evaluation dimensions for Disaster Captioning and Restoration Advice are detailed in Table III, with the overall score for each task computed as the average of its respective dimensions. Higher scores on this $0 ~ \sim ~ 5$ scale signify more accurate, comprehensive, and verifiable disaster assessment reports.

TABLE IV  
PERFORMANCE COMPARISON ON THE RSCC BENCHMARK
<table><tr><td rowspan="2">Method</td><td colspan="2">N-Gram</td><td>Contextual Similarity</td></tr><tr><td>ROUGE-L (%)</td><td>METEOR (%)</td><td>ST5-SCS (%)</td></tr><tr><td>• General VLM</td><td></td><td></td><td></td></tr><tr><td>Kimi K2.6 [52]</td><td>11.14</td><td>10.98</td><td>52.29</td></tr><tr><td>GLM-4.5V [53]</td><td>13.24</td><td>10.64</td><td>50.57</td></tr><tr><td>InternVL3.5-8B [27]</td><td>12.05</td><td>10.84</td><td>47.98</td></tr><tr><td>Qwen3.5-122B-A10B [51]</td><td>12.27</td><td>10.59</td><td>50.20</td></tr><tr><td>Qwen3.5-27B [51]</td><td>11.68</td><td>11.00</td><td>51.84</td></tr><tr><td>Qwen3.5-9B [51]</td><td>13.23</td><td>11.23</td><td>49.25</td></tr><tr><td>Qwen3.5-4B [51]</td><td>12.82</td><td>12.06</td><td>50.59</td></tr><tr><td colspan="4">• Remote Sensing VLM</td></tr><tr><td>CCExpert-7B [54]</td><td>8.39</td><td>3.73</td><td>40.10</td></tr><tr><td>RSCCM-7B [16]</td><td>14.99</td><td>16.05</td><td>58.52</td></tr><tr><td colspan="4">• Remote Sensing Agent (with the same expert toolkit as ours)</td></tr><tr><td>EarthAgent [21]</td><td>15.31</td><td>13.50</td><td>65.88</td></tr><tr><td>OpenEarthAgent [30]</td><td>16.00</td><td>12.74</td><td>62.21</td></tr><tr><td>• Ours</td><td></td><td></td><td></td></tr><tr><td>DisasterAgent</td><td>15.76</td><td>14.77</td><td>67.18</td></tr><tr><td>vs. Qwen3.5-27B</td><td>↑4.08%</td><td>↑3.77%</td><td>↑15.34%</td></tr></table>

## C. Experimental Details

DisasterAgent is implemented on top of the LangChain [49] and LangGraph [50] frameworks. The entire four stage workflow, comprising Skill routing, Perception, Evidence-based Augmentation, and Generation, is compiled into a stateful directed graph using LangGraph. This architecture allows a shared, typed workflow state to flow seamlessly between nodes. Across all agentic stages, we reuse a single Qwen3.5- 27B [51] as the central brain, accessed via the API provided by Alibaba Cloud’s Bailian platform<sup>1</sup>. The unified VLM brain is wrapped as a standardized client that exposes text, multimodal, and JSON conversational interfaces. To ensure deterministic and reproducible outputs, we employ greedy decoding with the temperature set to 0.0. Notably, no disaster specific finetuning is performed on this VLM.

Within the perception stage, the Plan-and-Execute agent generates structured JSON tool-call plans, which are executed by a local Python runtime through a unified tool interface. The unified tool interface supports two execution modes: offline prediction cache and online model inference. For the offline prediction cache mode, the tool runtime pre-caches all possible prediction results for each tool call as JSON, which allows for fast and deterministic execution. For the online model inference mode, the tool runtime requests on-demand predictions from independently deployed FastAPI services over HTTP. The EEE retrieval service follows the same deployment as the online perception tool services.

For the comparative analysis, we evaluate DisasterAgent against a wide range of mainstream general domain VLMs, specialized remote sensing VLMs, and existing remote sensing agent frameworks. To ensure a fair comparison and isolate the performance gains attributed to our proposed workflow, it is crucial to note that we equip the baseline remote sensing agents with the exact same specialized vision expert toolkit used by DisasterAgent. This rigorous setting guarantees that any observed improvements stem directly from our structured spatiotemporal memory and Evidence-based Augmentation mechanisms, rather than differences in the underlying perception tools. For the LLM as a judge evaluation, we utilize DeepSeek-V4-Flash [55] to score the generated reports. The evaluation prompts strictly follow the settings of the DisasterM3 benchmark [17].

## D. Performance Comparison

As shown in Table IV, DisasterAgent consistently surpasses all zero-shot VLM baselines across all metrics on the RSCC benchmark. Compared to the Qwen3.5-27B base model, it achieves absolute improvements of 4.08%, 3.77%, and 15.34% on ROUGE-L, METEOR, and ST5-SCS, respectively. Remarkably, on the contextual similarity metric, it significantly outperforms the fine-tuned RSCCM-7B model on ST5-SCS (67.18% vs. 58.52%), despite not utilizing any fine-tuning for specific tasks. Regarding the n-gram based metrics, DisasterAgent achieves a higher ROUGE-L score (15.76% vs. 14.99%) than RSCCM-7B, while RSCCM-7B retains a slight edge on METEOR (16.05% vs. 14.77%). This slight edge is expected given its training objective tailored for RSCC, which directly optimizes for exact token matching. Furthermore, when compared to other remote sensing agent frameworks (EarthAgent and OpenEarthAgent) equipped with the same expert toolkit, DisasterAgent achieves the highest ST5-SCS (67.18%) and highly competitive ROUGE-L and METEOR scores. The substantial gain in ST5-SCS underscores that DisasterAgent produces semantically richer and more accurate change descriptions. This improvement is primarily attributed to the STPG and SCE modules, which explicitly capture and reason over spatiotemporal evolution rather than relying on fragmented tool outputs or end-to-end caption generation.

On the DisasterM3 benchmark, we evaluate DisasterAgent against a diverse set of baselines, including general VLMs, remote sensing VLMs, and remote sensing agents. As shown in Table V, DisasterAgent achieves highly competitive performance across all task families, with particularly remarkable advantages in comprehensive report generation.

Compared with the base model Qwen3.5-27B (which serves as the central brain in our framework), DisasterAgent improves the average score for disaster report generation by 13.5%, with increases of 19.8% for Disaster Captioning and 7.2% for Restoration Advice. This substantial improvement highlights that the proposed evidence-driven workflow, integrating visual grounding, semantic verification, and historical experience, effectively mitigates the hallucination issues common in pure VLM generation. Furthermore, among the open source baselines, smaller models (e.g., InternVL3.5-8B, TeoChat-7B) lag considerably behind in generating coherent and factually correct reports. Similarly, while commercial

TABLE V  
PERFORMANCE COMPARISON ON THE DISASTERM3 BENCHMARK
<table><tr><td rowspan="3">Method</td><td rowspan="2">DTR</td><td rowspan="2">BDC</td><td rowspan="2">DRE</td><td colspan="8">Disaster Report Generation</td></tr><tr><td colspan="3">Disaster Captioning</td><td></td><td colspan="3">Restoration Advice</td></tr><tr><td>ACC↑</td><td>MAE↓</td><td>MAE↓</td><td>AVG↑</td><td>DAP↑</td><td>DDR↑</td><td>FC↑</td><td>AVG↑</td><td>RN↑</td><td>APP↑</td><td>SC↑</td></tr><tr><td colspan="10">• General VLM</td><td></td><td></td><td></td></tr><tr><td>Kimi K2.6 [52]</td><td>74.22</td><td>32.63</td><td>4.46</td><td>1.37</td><td>1.61</td><td>1.21</td><td>1.30</td><td>2.96</td><td>3.52</td><td>2.78</td><td>2.57</td></tr><tr><td>MiniMax-M3 [56]</td><td>68.84</td><td>26.50</td><td>3.81</td><td>2.03</td><td>2.31</td><td>1.80</td><td>1.97</td><td>2.79</td><td>3.37</td><td>2.58</td><td>2.42</td></tr><tr><td>Qwen3.5-27B [51]</td><td>70.25</td><td>30.05</td><td>4.81</td><td>1.88</td><td>2.16</td><td>1.65</td><td>1.81</td><td>2.86</td><td>3.47</td><td>2.67</td><td>2.44</td></tr><tr><td>Qwen3.5-9B [51]</td><td>64.02</td><td>35.66</td><td>14.38</td><td>1.71</td><td>2.02</td><td>1.47</td><td>1.65</td><td>2.62</td><td>3.23</td><td>2.43</td><td>2.19</td></tr><tr><td>InternVL3.5-8B [27]</td><td>47.03</td><td>36.70</td><td>26.16</td><td>1.16</td><td>1.43</td><td>0.97</td><td>1.09</td><td>2.55</td><td>3.26</td><td>2.26</td><td>2.14</td></tr><tr><td colspan="10">• Remote Sensing VLM</td><td></td><td></td></tr><tr><td>GeoChat-7B [13]</td><td>42.49</td><td>32.79</td><td>7.07</td><td>0.78</td><td>0.97</td><td>0.34</td><td>1.03</td><td>1.72</td><td>2.18</td><td>1.60</td><td>1.39</td></tr><tr><td>TeoChat-7B [14]</td><td>61.76</td><td>34.03</td><td>11.24</td><td>1.29</td><td>1.56</td><td>0.64</td><td>1.65</td><td>1.62</td><td>2.11</td><td>1.47</td><td>1.29</td></tr><tr><td colspan="10">• Remote Sensing Agent (with the same expert toolkit as ours)</td><td></td><td></td></tr><tr><td>EarthAgent [21]</td><td>95.18</td><td>18.82</td><td>2.21</td><td>2.36</td><td>2.72</td><td>2.03</td><td>2.34</td><td>3.15</td><td>3.76</td><td>2.96</td><td>2.74</td></tr><tr><td>OpenEarthAgent [30]</td><td>94.62</td><td>18.82</td><td>2.21</td><td>2.28</td><td>2.61</td><td>1.94</td><td>2.29</td><td>3.13</td><td>3.76</td><td>2.92</td><td>2.70</td></tr><tr><td colspan="10">• Ours</td><td></td></tr><tr><td>DisasterAgent</td><td>95.47</td><td>19.00</td><td>2.21</td><td>2.87</td><td>3.20</td><td>2.52</td><td>2.88</td><td>3.22</td><td>3.77</td><td>3.05</td><td>2.84</td></tr><tr><td>vs. Qwen3.5-27B</td><td>↑25.22%</td><td>↓11.05</td><td>↓2.60</td><td>↑19.8%</td><td>↑20.8%</td><td>↑17.4%</td><td>↑21.4%</td><td>↑7.2%</td><td>↑6.0%</td><td>↑7.6%</td><td>↑8.0%</td></tr></table>

![](images/b62f62347fd1106ca155b7c632fb7215710bcb766a2b2d9d825bb31753098589.jpg)  
Fig. 6. Disaster captioning scores on the DisasterM3 benchmark across eight disaster types. The VLM, EarthAgent, and DisasterAgent all use Qwen3.5- 27B as the base VLM, and EarthAgent and DisasterAgent both use the same expert toolkit.

APIs (e.g., Kimi K2.6, MiniMax-M3) demonstrate competitive reasoning capabilities, they still fall short of DisasterAgent’s performance. This indicates that neither general domain pretraining nor massive model scale is sufficient for reliable disaster assessment without structured reasoning. Crucially, when comparing agentic frameworks, DisasterAgent, Earth-Agent, and OpenEarthAgent achieve comparable performance in basic recognition and quantitative tasks (DTR, BDC, and DRE). This alignment is expected since all three agents are equipped with the identical vision expert toolkit, which provides the same foundational perception capabilities. However, when focusing on the highly complex comprehensive report generation tasks, DisasterAgent demonstrates a distinct advantage. It outperforms EarthAgent by absolute margins of 0.51 and 0.07 on the average scores for Disaster Captioning and Restoration Advice, respectively. This proves that simply stacking perception tools is inadequate; the structured STPG memory and multi-source evidence augmentation are vital for synthesizing fragmented tool outputs into high quality reports. Fig. 6 further compares disaster captioning scores across eight DisasterM3 types. DisasterAgent outperforms both Qwen3.5- 27B and EarthAgent on every disaster type, with the largest margins on explosion, earthquake, and tsunami scenes.

TABLE VI  
ABLATION STUDY ON THE STRUCTURAL COMPONENTS
<table><tr><td>Group</td><td>STPG</td><td>EA</td><td>Disaster Captioning</td><td>Restoration Advice</td><td>AVG</td></tr><tr><td>w/o Toolkit</td><td>X</td><td>X</td><td>1.88</td><td>2.86</td><td>2.37</td></tr><tr><td rowspan="3">w/ Toolkit</td><td>X</td><td>X</td><td>2.36</td><td>3.15</td><td>2.76</td></tr><tr><td>√</td><td>X</td><td>2.59</td><td>2.96</td><td>2.78</td></tr><tr><td>√</td><td>√</td><td>2.87</td><td>3.22</td><td>3.05</td></tr></table>

## E. Ablation Studies

To systematically evaluate the contribution of each proposed module, we conduct comprehensive ablation studies on the disaster report generation tasks of DisasterM3-Bench. All ablation variants utilize the same Qwen3.5-27B base model, DeepSeek-V4-Flash judge, and decoding configurations, with only the specific framework components toggled. The ablation studies are divided into two parts: first, we investigate the effectiveness of the structural components; second, we conduct a detailed analysis of the individual evidence streams.

![](images/a00e5d0c1c15aabb43650ec61272d3b196898c46242c8de2bb71f65fa2fae861.jpg)

![](images/8f6a80b11a2cf981e58dca1f774d269a4ca8f337fa9524f726eaa3f27579989f.jpg)  
Fig. 7. Detailed ablation results across individual evaluation metrics on the DisasterM3 benchmark. The figure illustrates the incremental performance improvements in (a) disaster captioning and (b) restoration advice when integrating the expert toolkit, STPG, and EA into the base model.

1) Effectiveness of Structural Components: As shown in Table VI, we first establish a w/o Toolkit baseline, where the VLM generates reports end-to-end without any tool invocation, yielding an overall average score of 2.37. By equipping the VLM with a specialized vision expert toolkit via a standard Plan-and-Execute agent, the average score is raised to 2.76. To isolate the impact of these perception tools and specifically evaluate the performance gains attributed to our reasoning workflow, the remaining two settings form a controlled group that shares the exact same expert toolkit. Introducing the STPG to organize the tool outputs primarily boosts the disaster captioning performance, which drives the overall average up to 2.78. Finally, enabling the complete Evidence-based Augmentation (EA) produces the full model with a peak average score of 3.05.

To understand the specific impact of these components on detailed metrics, we analyze the performance trends illustrated in Fig. 7. The transition from the toolkit free baseline to the Plan-and-Execute agent confirms that specialized perception tools provide essential visual cues. However, without a unified spatiotemporal structure, the tool outputs remain fragmented, limiting the disaster captioning score to 2.36. As depicted in Fig. 7(a), introducing the STPG significantly improves disaster captioning to 2.59, demonstrating that explicitly encoding spatial relations and temporal evolution is highly beneficial for factual descriptions. Conversely, Fig. 7(b) shows that restoration advice drops slightly from 3.15 to 2.96 at this stage, suggesting that a structured perception brief without historical priors is less effective for action oriented recommendations. Finally, integrating EA (which comprises SCE, VGE, and EEE) maximizes performance across all dimensions, reaching 2.87 in captioning and 3.22 in restoration advice. Overall, these detailed trends prove that while the expert toolkit is necessary, structured spatiotemporal perception and evidencedriven generation are indispensable for converting fragmented tool outputs into highly verifiable disaster reports.

TABLE VII  
CONTRIBUTION OF MULTI-SOURCE EVIDENCE
<table><tr><td rowspan="2">Task</td><td rowspan="2">Metric</td><td rowspan="2">Full</td><td colspan="3">w/o (Remove)</td><td rowspan="2">Base</td></tr><tr><td>EEE</td><td>VGE</td><td>SCE</td></tr><tr><td rowspan="4">Caponng Disaster</td><td>AVG</td><td>2.87</td><td>2.56</td><td>2.85</td><td>2.58</td><td>1.88</td></tr><tr><td>DAP</td><td>3.20</td><td>2.94</td><td>3.22</td><td>2.87</td><td>2.16</td></tr><tr><td>DDR</td><td>2.52</td><td>2.14</td><td>2.46</td><td>2.30</td><td>1.65</td></tr><tr><td>FC</td><td>2.88</td><td>2.60</td><td>2.89</td><td>2.56</td><td>1.81</td></tr><tr><td rowspan="4">Restton Advie</td><td>AVG</td><td>3.22</td><td>3.08</td><td>-</td><td>-</td><td>2.86</td></tr><tr><td>RN</td><td>3.77</td><td>3.68</td><td>-</td><td>-</td><td>3.47</td></tr><tr><td>APP</td><td>3.05</td><td>2.90</td><td>-</td><td>=</td><td>2.67</td></tr><tr><td>SC</td><td>2.84</td><td>2.65</td><td>-</td><td>=</td><td>2.44</td></tr></table>

2) Contribution of Multi-Source Evidence: To understand the specific role of each evidence stream from the perspective of final report generation, we perform an ablation study on the full model by removing one component at a time, as detailed in Table VII. Removing EEE results in the most noticeable performance degradation, with the Disaster Captioning average score dropping to 2.56 and the Restoration Advice average score decreasing to 3.08. This confirms that EEE serves as a crucial reference, providing essential domain-specific guidance for formulating normative and actionable restoration advice. Furthermore, the substantial drop in the Disaster Captioning average score reveals that historical context is also vital for guiding the VLM to generate accurate factual descriptions of disaster scenarios.

Removing SCE causes the Disaster Captioning average score to fall to 2.58. Notably, this removal leads to a substantial decline in DAP and FC. This validates that the confidencescored, self-auditing SCE mechanism serves as the primary verifiability anchor, effectively filtering out hallucinations during the caption generation stage.

Removing VGE has a relatively marginal effect on the overall Disaster Captioning average score. However, it causes a noticeable drop in DDR. This indicates that while SCE and EEE provide strong semantic and historical priors, VGE supplies fine-grained, pixel-level localization that is essential for accurately grounding damage quantities and spatial distribution.

## F. Qualitative Analysis

To intuitively demonstrate the advantages of our proposed framework, we present a qualitative comparison of disaster captioning between the base VLM (Qwen3.5-27B) and DisasterAgent, as illustrated in Fig. 8.

When relying on direct end-to-end inference, the base VLM suffers from severe hallucinations due to its black-box nature and lack of explicit spatiotemporal grounding. In this specific case, the base model misclassifies the event as a “Tsunami”

![](images/d661fd1e12a3a800d103e1b73a6f9817fc71f98ca96a8de4de024abb40757c4e.jpg)

rather than an explosion. Consequently, it fabricates nonexistent details that align with its erroneous assumption, such as “erosion and sediment redistribution” along the coastline and “alterations to the shoreline.” This unconstrained generation leads to extremely poor evaluation scores, including a Factual Correctness of 0, highlighting the unreliability of standard VLMs for rigorous remote sensing tasks.

In contrast, DisasterAgent leverages its evidence-driven architecture to produce a highly accurate and verifiable disaster report. It correctly identifies the disaster as an “Explosion” and precisely localizes the damage, noting that a “large warehouse structure on the right side” was destroyed while the surrounding roads and water bodies remained intact. Beyond mere accuracy, the most significant advantage of DisasterAgent is its verifiability. As shown in the generated report, every critical statement is appended with a citation tag. These tags correspond to a structured “Verifiable Visual Evidence Reference” that links directly back to the explicitly rendered perception masks and confidence scores from the STPG. By grounding its generation in traceable visual and semantic evidence, DisasterAgent effectively mitigates hallucinations, achieving a perfect Factual Correctness score of 5 and providing the verifiability required for real-world emergency response. To complement the qualitative analysis, we also provide four additional disaster captioning examples covering tsunami, earthquake, flood, and wildfire scenarios in Appendix II, as shown in Figs. 11–14.

TABLE VIII  
SCE CITATION COVERAGE AND ENTITY ASSOCIATIONS
<table><tr><td>Metric</td><td>DisasterM3</td><td>RSCC</td></tr><tr><td>Number of Samples</td><td>2,004</td><td>988</td></tr><tr><td>Samples with SCE Citations (%)</td><td>94.86</td><td>85.32</td></tr><tr><td>Mean STPG Entities per Sample</td><td>182.37</td><td>35.84</td></tr><tr><td>Mean SCE Citations per Sample</td><td>3.40</td><td>1.93</td></tr><tr><td>Mean Linked STPG Entities per SCE Citation</td><td>5.23</td><td>8.02</td></tr></table>

![](images/ef5007185ec5f7746a5366051d2fc04d300468eadff9021abcec010d198a1018.jpg)  
Fig. 9. An SCE citation and its linked STPG entities. The citation in the report excerpt is traced to the corresponding SCE claim and its associated STPG entities, including region nodes and an evolution edge, which are rendered on the pre- and post-disaster images.

## G. Evidence Verifiability Analysis

As reported in Table VIII, SCE citations cover most evaluated samples on both benchmarks, and each citation is associated with multiple STPG entities. Fig. 9 shows how an SCE citation is traced to its linked STPG entities, using a flood example from the RSCC benchmark. In the report excerpt, the citation connects the description of water-body expansion to the corresponding SCE claim. This claim is associated with STPG entities comprising pre-disaster region node 2, postdisaster region node 4, and their evolution edge, which are rendered on the pre- and post-disaster images. The water-body masks contain 79,039 and 173,648 pixels, respectively, giving a post-to-pre area ratio of approximately 2.197. This ratio is consistent with the SCE claim that the water-body area expanded to approximately 2.2 times its pre-disaster extent, and the rendered masks make the locations and boundaries of these regions directly inspectable. The example shows how an SCE citation is linked to its STPG entities and the corresponding image regions, providing explicit spatiotemporal provenance for verification.

## V. CONCLUSION

In this paper, we present DisasterAgent, an evidence-driven agent that elevates remote sensing disaster assessment from basic visual perception to verifiable comprehensive report generation. The framework couples a specialized vision expert toolkit with the STPG, which systematically organizes fragmented tool outputs into coherent spatial relationships and temporal evolution edges. A multi-source evidence augmentation mechanism further integrates visual grounding, semantic verification, and historical experience to guide the generation process. Experiments on the DisasterM3 and RSCC benchmarks suggest the effectiveness of the proposed framework, particularly in mitigating hallucinations and anchoring factual claims to traceable evidence. In future work, we will extend DisasterAgent to multi modal data sources (e.g., Synthetic

Aperture Radar) and multi temporal disaster imagery, enabling more resilient and dynamic disaster assessment systems.

## ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China under Grant 62425115, and partially supported by China Science and Technology Cloud (CSTCloud).

## REFERENCES

[1] K. Kaku, “Satellite remote sensing for disaster management support: A holistic and staged approach based on case studies in sentinel asia,” International Journal of Disaster Risk Reduction, vol. 33, pp. 417–432, 2019.

[2] H. Chen, J. Song, O. Dietrich, C. Broni-Bediako, W. Xuan, J. Wang, X. Shao, Y. Wei, J. Xia, C. Lan et al., “Bright: A globally distributed multimodal building damage assessment dataset with very-highresolution for all-weather disaster response,” Earth System Science Data, vol. 17, no. 11, pp. 6217–6253, 2025.

[3] A. Fujita, K. Sakurada, T. Imaizumi, R. Ito, S. Hikosaka, and R. Nakamura, “Damage detection from aerial images via convolutional neural networks,” in 2017 Fifteenth IAPR international conference on machine vision applications (MVA). IEEE, 2017, pp. 5–8.

[4] R. Gupta, B. Goodman, N. Patel, R. Hosfelt, S. Sajeev, E. Heim, J. Doshi, K. Lucas, H. Choset, and M. Gaston, “Creating xbd: A dataset for assessing building damage from satellite imagery,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2019, pp. 10–17.

[5] Y. Shen, S. Zhu, T. Yang, C. Chen, D. Pan, J. Chen, L. Xiao, and Q. Du, “Bdanet: Multiscale convolutional neural network with cross-directional attention for building damage assessment from satellite images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–14, 2021.

[6] W. Xiao, J. Su, Y. Chen, and G. Cao, “Cross-scale-guided fusion transformer for disaster assessment using satellite imagery,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–12, 2023.

[7] Z. Zheng, Y. Zhong, J. Wang, A. Ma, and L. Zhang, “Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: From natural disasters to manmade disasters,” Remote Sensing of Environment, vol. 265, p. 112636, 2021.

[8] H. Chen, J. Song, C. Han, J. Xia, and N. Yokoya, “Changemamba: Remote sensing change detection with spatiotemporal state space model,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1– 20, 2024.

[9] J. Xu, D. J. Nair, and S. T. Waller, “Implementing equitable wildfire response plans,” Science, vol. 388, no. 6743, pp. 158–159, 2025.

[10] F. Xu, J. Ma, N. Li, and J. C. Cheng, “Large language model applications in disaster management: An interdisciplinary review,” International Journal of Disaster Risk Reduction, vol. 127, p. 105642, 2025.

[11] A. P. Ndigande, J. Wiggins, and S. Ozer, “Survey on disaster management datasets for remote sensing based emergency applications,” arXiv preprint arXiv:2605.08196, 2026.

[12] W. Zhang, M. Cai, T. Zhang, Y. Zhuang, and X. Mao, “Earthgpt: A universal multimodal large language model for multisensor image comprehension in remote sensing domain,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–20, 2024.

[13] K. Kuckreja, M. S. Danish, M. Naseer, A. Das, S. Khan, and F. S. Khan, “Geochat: Grounded large vision-language model for remote sensing,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 27 831–27 840.

[14] J. A. Irvin, E. R. Liu, J. C. Chen, I. Dormoy, J. Kim, S. Khanna, Z. Zheng, and S. Ermon, “Teochat: A large vision-language assistant for temporal earth observation data,” arXiv preprint arXiv:2410.06234, 2024.

[15] M. Rahnemoonfar, T. Chowdhury, A. Sarkar, D. Varshney, M. Yari, and R. R. Murphy, “Floodnet: A high resolution aerial imagery dataset for post flood scene understanding,” IEEE Access, vol. 9, pp. 89 644–89 654, 2021.

[16] Z. Chen, C. Wang, N. Zhang, and F. Zhang, “Rscc: A large-scale remote sensing change caption dataset for disaster events,” Advances in Neural Information Processing Systems, vol. 38, 2025.

[17] J. Wang, W. Xuan, H. Qi, Z. Liu, K. Liu, Y. Wu, H. Chen, J. Song, J. Xia, Z. Zheng et al., “Disasterm3: A remote sensing vision-language dataset for disaster damage assessment and response,” Advances in Neural Information Processing Systems, vol. 38, 2025.

[18] N. A. Talemi, J. Boone, and F. Afghah, “Agentic ai in remote sensing: Foundations, taxonomy, and emerging systems,” arXiv preprint arXiv:2601.01891, 2026.

[19] C. H. Kao, W. Zhao, C. Lam, A. Umap, S. Revankar, S. Speas, S. Bhagat, R. Datta, C. P. Phoo, U. Mall et al., “Towards llm agents for earth observation,” in Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 2597–2611.

[20] W. Xu, Z. Yu, B. Mu, Z. Wei, Y. Zhang, G. Li, J. Wang, and M. Peng, “Rs-agent: Automating remote sensing tasks through intelligent agent,” arXiv preprint arXiv:2406.07089, 2024.

[21] P. Feng, Z. Lv, J. Ye, X. Wang, X. Huo, J. Yu, W. Xu, W. Zhang, L. Bai, C. He et al., “Earth-agent: Unlocking the full landscape of earth observation with agents,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 55 700–55 764.

[22] Z. Wang, C. Wu, F. Zhang, and J. Xia, “Constructing an extensible building damage dataset via semi-supervised fine-tuning across 12 natural disasters,” Journal of Remote Sensing, vol. 5, p. 0733, 2025.

[23] Y. Gao, Z. Ye, and H. Yu, “Cost-efficient computation offloading in sagin: A deep reinforcement learning and perception-aided approach,” IEEE Journal on Selected Areas in Communications, vol. 42, no. 12, pp. 3462–3476, 2024.

[24] Y. Gao, Z. Ye, Z. Lyu, M. Xiao, Y. Xiao, P. Yang, and A. Manolova, “Vision-aided isac in low-altitude economy networks via de-diffused visual priors,” IEEE Transactions on Cognitive Communications and Networking, vol. 12, pp. 3831–3845, 2025.

[25] F. Li, R. Zhang, H. Zhang, Y. Zhang, B. Li, W. Li, Z. Ma, and C. Li, “Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models,” arXiv preprint arXiv:2407.07895, 2024.

[26] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[27] W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao et al., “Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency,” arXiv preprint arXiv:2508.18265, 2025.

[28] Y. Jiang, J. Wang, X. Shen, and K. Dai, “Large language model for postearthquake structural damage assessment of buildings,” Computer-Aided Civil and Infrastructure Engineering, vol. 40, no. 31, pp. 6324–6342, 2025.

[29] A. Shabbir, M. A. Munir, A. Dudhane, M. U. Sheikh, M. H. Khan, P. Fraccaro, J. B. Moreno, F. S. Khan, and S. Khan, “Thinkgeo: Evaluating tool-augmented agents for remote sensing tasks,” arXiv preprint arXiv:2505.23752, 2025.

[30] A. Shabbir, M. Umer Sheikh, M. Akhtar Munir, H. Debary, M. Fiaz, M. Zaigham Zaheer, P. Fraccaro, F. Shahbaz Khan, M. Haris Khan, X. Xiang Zhu et al., “Openearthagent: A unified framework for toolaugmented geospatial agents,” in European Conference on Computer Vision. Springer, 2026, pp. 172–191.

[31] C. Liu, K. Chen, H. Zhang, Z. Qi, Z. Zou, and Z. Shi, “Change-agent: Toward interactive comprehensive remote sensing change interpretation and analysis,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–16, 2024.

[32] L. Lahouel, L. Lopata, S. Gruening, G. Meoni, G. Petit, and S. Lobry, “Ic-eo: Interpretable code-based assistant for earth observation,” arXiv preprint arXiv:2602.00117, 2026.

[33] Z. Chen, H. Wang, J. Yao, P. Ghamisi, J. Zhou, P. M. Atkinson, and B. Zhang, “Cangling-knowflow: A unified knowledge-and-flow-fused agent for comprehensive remote sensing applications,” arXiv preprint arXiv:2512.15231, 2025.

[34] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel¨ et al., “Retrievalaugmented generation for knowledge-intensive nlp tasks,” Advances in neural information processing systems, vol. 33, pp. 9459–9474, 2020.

[35] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global: A graph rag approach to query-focused summarization,” arXiv preprint arXiv:2404.16130, 2024.

[36] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[37] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything with concepts,” arXiv preprint arXiv:2511.16719, 2025.

[38] Y. Li, Y. Du, K. Zhou, J. Wang, X. Zhao, and J.-R. Wen, “Evaluating object hallucination in large vision-language models,” in Proceedings of the 2023 conference on empirical methods in natural language processing, 2023, pp. 292–305.

[39] T.-H. P. Wu, H. Lee, J. Ge, J. Gonzalez, T. Darrell, and D. Chan, “Generate, but verify: Reducing hallucination in vision-language models with retrospective resampling,” Advances in Neural Information Processing Systems, vol. 38, pp. 65 749–65 777, 2025.

[40] H. Li, X. Zhang, and H. Qu, “Ddfav: Remote sensing large vision language models dataset and evaluation benchmark,” Remote Sensing, vol. 17, no. 4, p. 719, 2025.

[41] K. Sparck Jones, “A statistical interpretation of term specificity and its application in retrieval,” Journal of documentation, vol. 28, no. 1, pp. 11–21, 1972.

[42] M. Li, Y. Zhang, D. Long, K. Chen, S. Song, S. Bai, Z. Yang, P. Xie, A. Yang, D. Liu et al., “Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking,” arXiv preprint arXiv:2601.04720, 2026.

[43] J. Menick, M. Trebacz, V. Mikulik, J. Aslanides, F. Song, M. Chadwick, M. Glaese, S. Young, L. Campbell-Gillingham, G. Irving et al., “Teaching language models to support answers with verified quotes,” arXiv preprint arXiv:2203.11147, 2022.

[44] T. Gao, H. Yen, J. Yu, and D. Chen, “Enabling large language models to generate text with citations,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 6465– 6488.

[45] A. Slobodkin, E. Hirsch, A. Cattan, T. Schuster, and I. Dagan, “Attribute first, then generate: Locally-attributable grounded text generation,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024, pp. 3309–3344.

[46] C.-Y. Lin, “Rouge: A package for automatic evaluation of summaries,” in Text summarization branches out, 2004, pp. 74–81.

[47] S. Banerjee and A. Lavie, “Meteor: An automatic metric for mt evaluation with improved correlation with human judgments,” in Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005, pp. 65–72.

[48] J. Ni, G. H. Abrego, N. Constant, J. Ma, K. Hall, D. Cer, and Y. Yang, “Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models,” in Findings of the association for computational linguistics: ACL 2022, 2022, pp. 1864–1874.

[49] H. Chase, “LangChain,” GitHub repository, 2022, accessed: 2026-09-14. [Online]. Available: https://github.com/langchain-ai/langchain

[50] LangChain Inc., “LangGraph,” GitHub repository, accessed: 2026-09- 14. [Online]. Available: https://github.com/langchain-ai/langgraph

[51] Qwen Team, “Qwen3.5: Towards native multimodal agents,” February 2026. [Online]. Available: https://qwen.ai/blog?id=qwen3.5

[52] K. Team, Y. Bai, Y. Bao, Y. Charles, C. Chen, G. Chen, H. Chen, H. Chen, J. Chen, N. Chen et al., “Kimi k2: Open agentic intelligence,” arXiv preprint arXiv:2507.20534, 2025.

[53] W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi, J. Ji, L. Pan et al., “Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning,” arXiv preprint arXiv:2507.01006, 2025.

[54] Z. Wang, M. Wang, S. Xu, Y. Li, and B. Zhang, “Ccexpert: Advancing mllm capability in remote sensing change captioning with difference-aware integration and a foundational dataset,” arXiv preprint arXiv:2411.11360, 2024.

[55] DeepSeek-AI, “DeepSeek-V4: Towards highly efficient milliontoken context intelligence,” https://huggingface.co/deepseek-ai/ DeepSeek-V4-Pro/blob/main/DeepSeek V4.pdf, 2026, technical report, accessed May 11, 2026.

[56] X. Lai, W. Xu, Y. Yang, Q. Chen, Y. Xu, L. Zeng, X. Li, H. Sun, H. Zhu, V. Zhang et al., “Minimax sparse attention,” arXiv preprint arXiv:2606.13392, 2026.

APPENDIX I PERCEPTION TOOL DESIGN  
(a) Disaster Type Recognition Tool  
![](images/9c3a502bdf86885c7f1178078078bded9291414f18b4a18b6407b1895ebf3a98.jpg)

(b) Building Damage Assessment Tool  
![](images/f914b9e9b37895ef6037662d4f551291609fdd0ca958f0737825c32283d88bbb.jpg)  
Fig. 10. Design of the perception tools. (a) Disaster type recognition encodes the pre- and post-disaster images with a shared DINOv3 ViT-L/16, fuses their patch tokens, and predicts one of ten disaster categories together with a confidence score. (b) Building damage assessment segments building instances with SAM3, consolidates them across the two dates, and assigns an instance-level damage label by coverage-aware voting over the masks from ChangeOS or ChangeMamba.

![](images/a81a735d8bf73539c39291c6f1e0aa819896e1b813f7e5f4f5368ce7dddf74d0.jpg)

## APPENDIX II ADDITIONAL QUALITATIVE RESULTS

To complement the qualitative analysis in the main text, we present four additional disaster captioning examples covering tsunami, earthquake, flood, and wildfire scenarios (Fig. 11–14).

![](images/08f201696c93f433ad32b74c6960a19dd43640867e9324f23056902d0f8bc6b9.jpg)

![](images/4e7224d23aa63e30eae460dbae42f99fe855458d1adabcf86c47c2b098416e42.jpg)

DISASTER: Landslide

BUILDING: Significant destruction of residential structures in the lower-left quadrant, with many buildings completely buried or swept away by debris.   
ROAD: Major road networks in the affected area are severed and covered by mud and debris, rendering them impassable.   
VEGETATION: Dense vegetation cover on the slopes has been stripped away, exposing bare earth and indicating a massive movement of soil and rock   
WATER\_BODY: No distinct water bodies are visible; however, the terrain suggests potential channeling of water flow contributing to the erosion.   
AGRICULTURE: Agricultural fields adjacent to the slide path have been inundated with sediment and debris, destroying crops and altering the land surface.   
CONCLUSION: The images depict a catastrophic landslide that has obliterated a residential settlement, destroyed infrastructure, and caused extensive damage to agricultural land and natural vegetation through massive soil displacement.

(c)

![](images/99c13776912e14c5ccef9fedd7daf3bd7f77676527a5711bb2bda66cb168d2b1.jpg)

DisasterAgent (Qwen3.5-27B)

DISASTER: Tsunami[1]

BUILDING: Widespread destruction is evident, with numerous buildings completely obliterated or heavily damaged, particularly in the lower and central regions. Debris fields are scattered extensively across the built environment.[2]

ROAD: The road network has been severely compromised; many segments are obscured by debris and sediment deposition, rendering them impassable or indistinguishable from the surrounding terrain.[3]

VEGETATION: Extensive areas of forest and vegetation have vanished, replaced by exposed soil and debris. Remaining green cover shows significant reduction in density and extent.[4] [6]

WATER\_BODY: No distinct permanent water bodies are visible, though the landscape indicates recent inundation and potential temporary pooling in low-lying agricultural zones.AGRICULTURE: Large tracts of farmland have been washed away or submerged. Some field patterns are disrupted, while new land formations suggest sediment deposition altering previous boundaries.[5] [7]

CONCLUSION: The tsunami caused catastrophic damage to buildings, roads, and vegetation, while significantly altering agricultural landscapes through erosion and sediment deposition.

Precision: 2 Recall: 2 Factual Correctness: 1

![](images/c9bcea91a8bf6ac6b7a0f735a2857b36bcb8c72d282e99e0994688117e91d4fe.jpg)

![](images/4868b303a666a6a03da4c44e28230c113602ef9ba6007563f110d9053adb7ead.jpg)

DISASTER：tsunami

BUILDING: Significant destruction is evident in the central area of the imagery, where numerous buildings that were intact pre-disaster appear partially collapsed or completely obliterated post-disaster.   
Debris is visible surrounding these affected structures, indicating structural failure caused by the tsunami.

ROAD: The primary road network visible in the central region between residential and agricultural areas has suffered significant disruption. Erosion and debris deposits have interrupted roadway continuity, making certain routes impassable.

VEGETATION: Substantial changes in vegetation are apparent. Areas of natural vegetation in the bottom right have been visibly eroded or inundated, as indicated by a reduction in spectral reflectance and the presence of bare ground.

WATER\_BODY: Observations show evidence of water encroachment in the lower portions of the imagery, where no visible standing water existed pre-disaster. This alteration suggests the presence of residual water accumulation from tsunami inundation.

AGRICULTURE: The agricultural fields in the central and right regions show distinct signs of impact, with field patterns disrupted, inundated, or covered by debris. The uniform texture and spectral signatures indicative of cultivated land are absent in several areas.

CONCLUSION: The tsunami has caused widespread damage to built infrastructure, road networks, vegetation, and agricultural areas. Substantial disruption to both natural and human-managed environments is observed, with standing water and debris prominently reflecting the disaster's severity.

Fig. 11. Qualitative DisasterM3 captioning case for a tsunami scene. (a) The input query with the corresponding pre- and post-disaster images. (b) Caption produced by Qwen3.5-27B, which misclassifies the event as a landslide and introduces unsupported descriptions of slope-related soil and rock movement (scores: DAP 2, DDR 2, FC 1). (c) Caption produced by DisasterAgent, which identifies the tsunami and describes damage to buildings, roads, vegetation and farmland, with selected claims linked to numbered evidence references (scores: DAP 5, DDR 4, FC 5). (d) Ground truth caption.

(d)  
(a)  
![](images/1c2d1a9763d8dc48b1939ad716eb8fed9fdc1e6ccfa37971f6a290de4b01a249.jpg)

Analyze the pre-disaster and post-disaster remote-sensing images and assess the disaster impact.

![](images/c3fcfe9f747a86efe8f1fc64c3eb5a32da33e934ee9535d46dda088732990527.jpg)

![](images/e4afc056d4d0ad904e01b63ff52f9ad2c9c1da6343431dfb8a94a6889da9c66b.jpg)

![](images/2a48f71bade40de1eec7620e91e0a178a1cad2e876e9b5bda1f93ba2950e905e.jpg)

![](images/1c86967bd6fa216cdd62f8e46888788e62323f2355a4d74a41436b08b49f8303.jpg)

DISASTER： Earthquake   
BUILDING: Significant structural damage is evident, with numerous buildings showing collapsed roofs, exposed interiors, and piles of rubble. Several structures appear to have partially or completely collapsed, particularly in the lower central area where red-roofed buildings are heavily damaged.   
ROAD: The road network remains largely intact and passable, though some debris from collapsed buildings may be encroaching on the edges of the streets. No major fissures or blockages are immediately visible on the main thoroughfares.   
VEGETATION: Natural vegetation cover appears relatively unaffected by the event, with trees and green spaces retaining their original shape and density.   
WATER\_BODY: No water bodies are present in the analyzed area to assess for changes.   
AGRICULTURE: There is no managed agricultural land visible within the urban footprint of these images.   
CONCLUSION: The imagery depicts a severe earthquake that caused widespread destruction to the built environment, characterized by building collapses and roof failures, while leaving the road infrastructure and natural vegetation largely undisturbed.

![](images/850225f9e4e679d560accc849f11e1723747df7a24421c0843f9fdec26e71b49.jpg)

Precision: 2 Recall: 1 Factual Correctness: 2

(c)

## DisasterAgent (Qwen3.5-27B)

![](images/6005a8ff2ba0c0460ea02f22a30ca4d94edfd4b9faa5d5494e60058f60748ee3.jpg)

DISASTER: Earthquake[1]

BUILDING: The vast majority of buildings remained intact following the event, with no widespread structural collapse observed. Minor surface irregularities or debris are visible on a few specific structures, but overall stability is maintained.[2]

ROAD: The road network appears fully functional and intact. No visible blockages, surface cracking, or deformities disrupt the traffic flow in the main thoroughfares or side streets.[3]

VEGETATION: Vegetation cover shows no discernible changes in spectral reflectance or density between the pre- and post-disaster images, indicating no damage to trees or green spaces.

WATER\_BODY: No water bodies are present within the assessed area; consequently, no changes related to inundation or recession can be identified

AGRICULTURE: There are no active agricultural fields or managed cropland evident in the urban context of the imagery.

CONCLUSION: The earthquake caused minor localized surface damage to a small number of buildings while leaving the road network, vegetation, and other infrastructure unaffected.

![](images/f4af735dc5350dc25cce00cae0ab50bc9516591ec6899cc925f2cfba4de5e0a9.jpg)  
Verifiable Visual Evidence Reference  
[1] The type of disaster that occurred in this area is: earthquake.(Expert Tool)

![](images/f1a6a8817350c6614b8379ac689a784fc23607bdee4220c10081cf35787b6696.jpg)

[2] The vast majority of buildings in the assessed area remained intact following the earthquake. (confidence score:0.95)

[3] Several roads in the region were assessed as intact after the disaster(confidence score: 0.95)

Precision: 5 Recall: 4 Factual Correctness: 5

DISASTER： earthquake

BUILDING: Minor structural disturbances are observed in the built-up areas, primarily roofs shifted or slightly breached. Notable changes are visible on multiple structures near the intersection in the central region of the images, with slight displacement or irregularities in rooftop appearances.

ROAD: The road network remains intact with no visible disruptions, blockages, or surface deformation observed in the images.

VEGETATION: Vegetation in unmanaged areas, including patches visible near the top-left and bottom-right of the images, shows no discernible loss, alterations, or degradation.

WATER\_BODY: No water bodies are detected within the given images, and thus no alterations are observed.

AGRICULTURE: No agricultural fields, defined by managed patterns or irrigation evidence, are visible in either image. Thus, no impacts can be assessed.

CONCLUSION: The earthquake resulted in minor impacts to the built environment, specifically visible in structural elements. No damage is observed in the road network, vegetation, water bodies, or agricultural areas.

Fig. 12. Qualitative DisasterM3 captioning case for an earthquake scene. (a) The input query with the corresponding pre- and post-disaster images. (b) Caption produced by Qwen3.5-27B, which exaggerates minor structural disturbances as widespread building collapse and severe destruction (scores: DAP 2, DDR 1, FC 2). (c) Caption produced by DisasterAgent, which describes buildings as largely intact, acknowledges limited structural disturbances, and identifies intact roads, with selected claims linked to numbered evidence references (scores: DAP 5, DDR 4, FC 5). (d) Ground truth caption.

![](images/1366fa503a0978c7879b919784b5c7a61d990c6de7b60ef4e3398834fcd14486.jpg)

![](images/6c3e960bbe0dbb312c8d453afcae3afd9bee172a370f53ef3459af0c82a30e46.jpg)

Analyze the pre-disaster and post-disaster remote-sensing images and assess the disaster impact.

![](images/18fee0f7a7780ffdbab841106a4c2088b3bbc39c3192b326190633f32938ab8c.jpg)

![](images/129612aabb07f78fe5f1f25fc927d8cdffa2d7aff54d3650213994190f6db9b1.jpg)

(b)

![](images/2cc4a1eaa6e7b3b18da296ea7b47e91a895814917d81d9cb4cbe21f7ff676e90.jpg)

DISASTER:Flood

BUILDING: Buildings are partially submerged or surrounded by water, with only rooftops visible in some areas.

ROAD: Road networks are completely obscured and inundated by floodwaters, rendering them impassable.

VEGETATION: Natural vegetation is largely submerged, with only the tops of trees protruding above the water surface.

WATER\_BODY: Water bodies have expanded significantly, covering vast areas that were previously dry land.

AGRICULTURE: Agricultural fields are entirely flooded, submerging crops and farmland under deep

CONCLUSION: The disaster has caused extensive flooding, submerging buildings, roads, vegetation, and agricultural lands, leading to severe disruption across all observed categories.

(c)

## DisasterAgent (Qwen3.5-27B)

![](images/26b34f2a073019d51941ef561df2d1cb1554ed7e2bd7652ea6216d768bf92525.jpg)

## DISASTER: Flood[1]

BUILDING: Multiple buildings remain intact despite the flooding, with some structures partially surrounded by water but showing no visible structural collapse.[4] A few buildings' status remains uncertain due to heavy inundation obscuring details.[6]

VEGETATION: Natural vegetation areas show signs of inundation, particularly near the expanded water body. Spectral changes suggest waterlogging and partial submersion of tree cover.

ROAD: Several roads are submerged by floodwaters, significantly restricting accessibility across the region. The road network shows evidence of widespread inundation. [3]

WATER\_BODY: Significant expansion of the water body is observed, encroaching into surrounding farmlands and low-lying areas. This indicates severe overflow consistent with major flooding events. AGRICULTURE: Large sections of farmland have been converted into water bodies, with distinct field patterns obscured or completely submerged. Crop loss is likely extensive in these inundated zones.[2] [5]

CONCLUSION: The flood caused significant impacts on agriculture and transportation infrastructure, while most buildings remained structurally sound. Vegetation and water bodies were heavily affected by the widespread inundation.

Factual Correctness: 2

## Verifiable Visual Evidence Reference

Recall: 2

![](images/4c090bb7c2d65b7d1147289b63a463114c840b250d87fe86f1644436905e9d36.jpg)

[1] The type of disaster that occurred in this area is: flood.(Expert Tool)

[2] A section of farmland has been converted into a water body. (confidence score:0.98)

![](images/60e5268bfe741de2bb8e055f25efc00299ce94b865f5ac2aafcf7909d66ef1c3.jpg)

[3] Several roads are submerged by floodwaters. (confidence score: 0.95)

[4] Multiple buildings remain intact despite the flooding. (confidence score:0.95)

![](images/043fc4aa19bb0fce4a520d70dfabb50e8e340fce578a90d15f3b2bb7970d54fc.jpg)

[6] The status of some buildings remains unknown following the event.(confidence score:0.95)

Precision: 3

![](images/64c55550318d60f13163781eb1e78be69875c89a9b0f044f469e782c63616bb3.jpg)

Precision:5 Recall: 4 Factual Correctness:5

(d)

![](images/e40cf5a83c15518d385933f0d8bc3bf9e5bb3e6fc9db41ecfd114264cf4d68bd.jpg)

Ground Truth

DISASTER： Flooding

BUILDING: Buildings remain unaffected as no structural changes, inundation, or visible damage to built infrastructure are observed when comparing pre- and post-disaster images.

ROAD: Road networks in the central portion of the post-disaster image are inundated, rendering them non-functional. Floodwater accumulation has obscured several segments visible in the pre-disaster image.

VEGETATION: Natural vegetation in the upper portions of the post-disaster image shows evidence of inundation, with reduced separation from adjacent waterlogged areas and diminished reflectance signatures typical of healthy vegetation.

WATER\_BODY: Flooding has significantly expanded water coverage in the central and lower portions of the post-disaster image. The increased spread and irregular boundaries of water are apparent compared to discrete features in the pre-disaster image.

AGRICULTURE: Agricultural fields across the center and lower portions of the post-disaster image are heavily inundated, with former field patterns virtually unrecognizable due to waterlogging. This likely indicates substantial crop loss.

CONCLUSION: The flooding has caused significant impacts across road networks, natural vegetation, and agricultural areas, primarily through inundation. While built infrastructure remains unaffected, land use and land cover changes are evident in post-disaster imagery.

Fig. 13. Qualitative DisasterM3 captioning case for a flood scene. (a) The input query with the corresponding pre- and post-disaster images. (b) Caption produced by Qwen3.5-27B, which identifies the flood but overstates its impact on buildings, including unsupported claims that some buildings are submerged with only their rooftops visible (scores: DAP 3, DDR 2, FC 2). (c) Caption produced by DisasterAgent, which distinguishes inundated roads and farmland from buildings assessed as intact and retains uncertainty for some structures, with selected claims linked to numbered evidence references (scores: DAP 5, DDR 4, FC 5). (d) Ground truth caption.

![](images/efc1f631c187deac6b9526c260634340dee48473abef68fd784a9b697f5578ec.jpg)  
Fig. 14. Qualitative DisasterM3 captioning case for a wildfire scene. (a) The input query with the corresponding pre- and post-disaster images. (b) Caption produced by Qwen3.5-27B, which misclassifies the event as a mudslide or debris flow and introduces unsupported claims of sediment deposition, altered drainage channels, and restricted road access (scores: DAP 1, DDR 0, FC 1). (c) Caption produced by DisasterAgent, which identifies the fire, describes extensive vegetation loss, and recognizes that the road network remains intact, with selected claims linked to numbered evidence references (scores: DAP 5, DDR 5, FC 5). (d) Ground truth caption.