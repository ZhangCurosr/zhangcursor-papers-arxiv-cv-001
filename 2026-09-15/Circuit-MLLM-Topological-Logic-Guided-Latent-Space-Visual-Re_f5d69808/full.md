# Circuit-MLLM: Topological Logic-Guided Latent-Space Visual Reasoning for Circuit Schematic Understanding

Jinyuan Deng, Yuqi Jiang, Wenjing Huang, Xin Li, Qi Sun<sup>⋆</sup>, and Cheng Zhuo

Zhejiang University, Hangzhou, China qisunchn@zju.edu.cn

Abstract. Through pre-training on extensive text and image datasets, current multi-modal large language models (MLLMs) achieve strong performance on general tasks. However, circuit schematics present a unique challenge for MLLMs due to their dense component layouts and distinct topological logic, demanding fine-grained structural parsing to extract the electrical semantics. To address this, we propose Circuit-MLLM, a multimodal reasoning framework that reformulates circuit topology analysis as a process of device localization, path tracing, and sequential reasoning within the latent space. We introduce a circuit knowledge mining mechanism that deeply aligns the model’s latent representations with structurally rich features derived from multi-granularity circuit vision experts, enabling the model to efectively internalize topological semantics. Building upon these internalized semantics, we devise a topology-guided sequencing strategy that decouples reasoning from the rigid raster-scan order, enforcing stepwise inference along the circuit’s topological logic in latent space. Across diverse circuit analysis tasks, Circuit-MLLM consistently outperforms strong baselines, notably achieving a 25% higher average score than GPT-5.1, which demonstrates the efectiveness of our framework in circuit schematic topology analysis. Code is publicly available at https://github.com/IC-Yuan/Circuit-MLLM.

Keywords: Multimodal Reasoning · Latent-space · Circuit Schematics

## 1 Introduction

Recent advances in Multimodal Large Language Models (MLLMs) have shown strong performance on general vision–language understanding and reasoning [11, 12]. However, applying MLLMs to circuit schematics remains challenging due to their dense component layouts and intricate topological dependencies [20]. This limitation impedes the deployment of MLLMs in Electronic Design Automation (EDA). Consequently, empowering MLLMs with schematic parsing capabilities is pivotal for automating circuit analysis and accelerating chip design.

![](images/779f7290a94e3ae0feb8207a7aba000b8747c9b580731dbc58e517a692ee6f55.jpg)  
Fig. 1: Motivation of Circuit-MLLM. Existing Visual Chain-of-Thought methods often disrupt the topological integrity of circuits. Meanwhile, latent-space approaches, constrained by raster-order alignment and coarse-grained feature extraction, force the model to deviate from topological paths and result in a lack of fine-grained perception.

Traditional circuit analysis predominantly depends on rigid computer vision pipelines that employ rule-based algorithms for component and connection recognition [16]. Following this paradigm, several MLLM-based approaches for circuit understanding rely on specialized detectors as external front-ends for symbol localization [1], adopting a decoupled design that sufers from error propagation and inhibits the MLLM’s inherent reasoning capabilities. In contrast, end-to-end approaches [22] bypass external detectors and typically rely on supervised finetuning (SFT) with schematic-to-structured language alignment. However, due to the lack of rich instance-level annotations, these models often rely on large-scale synthetic data, which limits their generalization to real-world distributions.

Recent advancements in MLLMs often employ Textual Chain-of-Thought to enhance reasoning, attempting to use textual descriptions to guide the model’s attention toward specific image regions. For circuit schematics, however, describing topology and routing precisely in natural language is inherently brittle, frequently leading to semantic ambiguity and background distraction. Visual Chain-of-Thought (VCoT) [21] ofers a novel perspective by explicitly invoking external tools such as image cropping. Nevertheless, these approaches [2, 13] typically adopt a region-centric “locate-judge” paradigm. This design is fundamentally ill-suited for circuit topology analysis, as discrete bounding boxes often fail to preserve the continuity of thin, long-range wires, resulting in fragmented connections and excessive background clutter.

In contrast, latent-space visual reasoning [4, 10, 19] ofers an alternative by performing inference within hidden states, avoiding explicit bounding-box and text-based representations. This paradigm provides a distinct advantage for analyzing irregular topologies that are inherently dificult to describe. However, directly applying this latent paradigm to complex circuit topologies faces two key challenges: (1) Scarcity of fine-grained features: Current latent-space approaches typically capture only coarse, macroscopic topological regions. These latent representations fail to encode the highly fine-grained connection details and dense component layouts essential for circuit topology analysis. (2) Missing structured order: Existing latent space alignment methods primarily enforce feature-level alignment between latent vectors and auxiliary image patches. Consequently, the sequence of latent representations defaults to a spatial, top-left to bottom-right raster-scan order. However, this scanning order fails to constrain the topology-driven viewpoint progression, resulting in a fundamental misalignment with the topology’s intrinsic logic.

To overcome these challenges, we propose Circuit-MLLM, the first topological logic-guided latent-space visual reasoning framework dedicated to complex circuit topology analysis. First, to address the lack of fine-grained circuit feature perception, we introduce a circuit knowledge mining mechanism based on the latent space. By constructing circuit vision experts, we guide the model to deeply internalize and align complex circuit representations directly within the latent space during training. This mechanism prevents the model from merely focusing on superficial, coarse regional information, thereby endowing it with genuine, fine-grained circuit recognition capabilities. Furthermore, we propose a topology-guided sequencing strategy. This strategy mandates that the generated latent vectors not only accurately encode the topological regions but also strictly follow the intrinsic topological logic in their generation order by specifically progressing from the root component, along the wire topology, to the target component, thereby efectively simulating the topological viewpoint shifts of human engineers entirely within the latent space.

Our contributions are as follows:

1. We propose Circuit-MLLM, a novel latent-space visual reasoning paradigm for end-to-end multimodal analysis of circuit topologies. Unlike verbose Chainof-Thought or external tool-chain methods, our framework directly bridges the semantic gap between spatial visual priors and rigorous circuit logic within the latent space, enabling intrinsic schematic comprehension.

2. We design a circuit knowledge mining mechanism based on latent space guided by a set of vision experts, which can generate highly discriminative fine-grained circuit latent representations without increasing inference overhead, thereby significantly improving topology recognition accuracy.

3. We propose a topology-guided sequencing strategy that decouples latent reasoning from rigid spatial raster-scanning. By dynamically aligning the inference trajectory with the inherent circuit structure, it endows the model with native, human-like topological tracing capabilities.

4. Extensive experiments reveal that Circuit-MLLM achieves a 25% higher average score than GPT-5.1 across diverse circuit analysis tasks, and outperforms latent-space visual baselines by up to 26% on topology reasoning tasks.

![](images/c36af03f87ae4a7011a7f55e343d3b6cff0308d407bf860e26dd9bb7c1798989.jpg)  
Fig. 2: Automated topology annotation workflow for circuit schematics

## 2 Preliminaries

## 2.1 Automated Circuit Schematic Topology Extraction

We have constructed an automated, fine-grained annotation workflow for circuit schematics as shown in Figure 2. First, we collected a total of 12,000 circuit schematics from multiple public datasets [1, 6, 15, 22], comprehensively covering analog, digital, and mixed-signal systems, making it the largest circuit schematic dataset currently available. Subsequently, we manually annotated a subset of the components, text, and junction nodes. A YOLO11 model [9] was then trained on this subset to enable automated detection and bounding box extraction across the entire dataset. Following this, we introduced an OCR framework [3] to perform text recognition and component matching, ensuring the accuracy of attribute assignments. Next, we employed pixel tracking techniques to reconstruct the topological dependencies of the circuits. Given the complexity of circuit topologies and connections, human expert intervention was incorporated to ensure rigorous accuracy. The resulting dataset provides netlist-style descriptions, visual grounding, and pixel-level regional masks. Simultaneously, we constructed latent-space reasoning QA pairs tailored to diferent question types, and introduced Circuit-MLLM-Bench to evaluate the model’s capabilities in circuit topology analysis, as detailed in Section 4.

## 2.2 Related Works

Circuit Schematic Comprehension with MLLMs. Circuit schematics are a fundamental form of structured visual data in EDA, and their understanding has recently attracted growing attention in the MLLM community [17, 18]. Prior work mainly follows two paradigms. Tool-based visual reasoning methods (e.g., Masala CHAI [1]) rely on conventional vision components such as object detectors to produce structured annotations that guide MLLM reasoning. Text-conversion pipelines (e.g., MAPs [22]) first translate circuit schematics into structured netlists using supervised fine-tuned models or of-the-shelf parsers, and subsequently delegate reasoning to text-domain LLMs. However, both paradigms are bottlenecked by intermediate tools and conversions, sufering from error propagation and engineering overhead.

![](images/deaf5437f9575b4f1468b4cbea6bd4dca385ec1d13ade8e8e4af6dcec0fddf47.jpg)  
Fig. 3: Visualization of MLLM intrinsic attention flow in circuit topology analysis. The sequence illustrates native visual tracing from the root component, propagating along topological wire paths to terminal devices.

Multimodal Chain-of-Thought. Chain of Thought (CoT) prompting unlocked progressive reasoning in large language models via intermediate logical steps [5]. Recent studies extended CoT into multimodal settings. For instance, Vision R1 [8] deduces complex logic directly from images via multimodal reinforcement learning, while ICoT [21] interleaves attention-selected image crops with text tokens to improve visual question answering. Visual CoT [13] provides bounding box-grounded reasoning, enhancing spatial localization via explicit visual tokens. Others [7] employ external tools for visual evidence, augmenting multimodal CoT. However, these approaches encounter limitations in circuit schematics, where irregular topological regions are ill-suited for standard bounding boxes and challenging to trace via pure natural language.

Latent Visual Reasoning. Recent studies have introduced latent space reasoning methods [14] into multimodal domains. Specifically, these methods represent intermediate visual steps within the latent space rather than explicitly decoding them into images, thereby preventing the model from being constrained by rigid formats. LVR [10] and Mirage [19] pioneered the Latent Visual Reasoning paradigm, accomplishing latent reasoning by aligning auxiliary image features with vectors in the latent space. Building upon this, ILVR [4] introduced selective latent vector alignment alongside an interleaved multiple round latent space reasoning paradigm. Despite the unconstrained latent space being highly suitable for complex circuit topologies, current methods often overlook its intrinsic logical chains, causing reasoning misalignments in circuit topology analysis.

## 2.3 Visual Attribution of MLLMs in Circuit Analysis

To investigate how the model intrinsically engages in a native reasoning process to analyze circuit schematic topology without any explicit prompt guidance, we conducted a visual attribution of the MLLM’s attention using the Qwen-2.5-VL 7B model. As depicted in Figure 3, we identified two key insights:

– Insight 1: Topology-Guided Attention Flow. When tackling circuit topology tasks, the MLLM exhibits a visual tracking pattern that mirrors circuit connection logic. Rather than scanning components globally, the model initially anchors on a root component, subsequently traces strictly along topological wire paths, and ultimately shifts focus to downstream devices.

![](images/58cdaa7ae68278bcb15927902d53313b8f781458ef0e58349786543fc065fe4a.jpg)  
Fig. 4: Framework of Circuit-MLLM. (1) The circuit knowledge mining mechanism based on latent space provides the model with expert-level circuit representation vectors; (2) a Topology-Guided Sequencing Strategy directs latent-space reasoning to strictly follow the circuit’s topological order; and (3) Text-Latent Joint Supervision Training unifies latent-space feature alignment and explicit text supervision.

Insight 2: Concurrent Branch Tracking. The MLLM demonstrates the capacity to simultaneously attend to multiple wire branches originating from a single node—a behavior likely facilitated by the multi-head attention mechanism. This parallel processing capability stands in contrast to human visual cognition, which typically favors serial, branch-by-branch analysis.

## 3 Methods

In this section, we introduce our Circuit-MLLM framework for sequentially ordered topological reasoning within the latent space. First, we present the latentspace circuit knowledge mining mechanism (Section 3.1) to internalize multiexpert visual capabilities without invoking external tools. Next, Section 3.2 details the topology-guided sequencing strategy, which strictly aligns the model’s latent reasoning sequence with rigorous circuit logic. Finally, Section 3.3 introduces the text-latent joint supervision training to ensure alignment between internal implicit topological reasoning and explicit textual outputs.

## 3.1 Latent-Space Circuit Knowledge Mining Mechanism

While recent latent-space visual reasoning works show promising results, they primarily rely on generic visual encoders to extract basic image features. However, in specialized domains like circuit analysis, fine-grained visual comprehension (e.g., component topologies) is paramount. To acquire these details, conventional methods often integrate supplementary network modules or explicitly invoke external vision tools during inference. This disjointed approach easily introduces cascading errors and increases inference latency.

To overcome these limitations, we propose a Latent-Space Circuit Knowledge Mining Mechanism. Our core philosophy is to abandon inference-time reliance on external tools, enabling the Circuit-MLLM to intrinsically learn and assimilate multi-expert visual features directly within its latent space during training. To this end, we design a Multi-Expert Feature Fusion module that serves as the source of knowledge guidance. For a given circuit image I, we deploy three heterogeneous vision experts to extract complementary structural and semantic priors: HAWP for holistic wireframe and junction detection (F<sub>HAWP</sub>), DeepLSD for fine-grained topological line segment extraction $( \mathcal { F } _ { \mathrm { L S D } } )$ , and DINOv2 for robust patch-level semantic representations (F<sub>DINO</sub>).

Due to architectural disparities among these experts, the extracted feature maps vary in spatial resolutions and channel dimensions. To construct a unified representation, we align them to a target spatial grid $( H _ { s } , W _ { s } )$ determined by the Circuit-MLLM’s visual encoder. Let $\varPhi _ { \mathrm { a l i g n } } ( \cdot )$ denote the bilinear interpolation operator mapping a tensor to this target resolution. The aligned features are concatenated along the channel dimension to form a dense expert representation $\mathcal { F } _ { \mathrm { e x t } }$

$$
\mathcal { F } _ { \mathrm { e x t } } = [ \varPhi _ { \mathrm { a l i g n } } ( \mathcal { F } _ { \mathrm { H A W P } } ) , \varPhi _ { \mathrm { a l i g n } } ( \mathcal { F } _ { \mathrm { L S D } } ) , \varPhi _ { \mathrm { a l i g n } } ( \mathcal { F } _ { \mathrm { D I N O } } ) ] \in \mathbb { R } ^ { ( C _ { 1 } + C _ { 2 } + C _ { 3 } ) \times H _ { s } \times W _ { s } } .\tag{1}
$$

Next, we flatten the spatial dimensions and employ a learnable projection matrix $\mathbf { W } _ { \mathrm { p r o j } }$ to project the concatenated heterogeneous features into the identical latent dimension d of the Circuit-MLLM. Let $\mathbf { E } _ { \mathrm { o r i g } } \in \mathbb { R } ^ { N \times d }$ denote the initial visual tokens extracted by the Circuit-MLLM’s original visual encoder, where $N = H _ { s } \times W _ { s }$ . To imbue these initial tokens with circuit-specific priors, we concatenate them and pass them through a fusion layer $\mathbf { W } _ { \mathrm { f u s e } }$ to obtain the knowledge-enriched feature pool $\mathbf { E } _ { \mathrm { f u s e d } }$

$$
\mathbf { E } _ { \mathrm { e x p e r t } } = \mathrm { F l a t t e n } ( \mathcal { F } _ { \mathrm { e x t } } ) \mathbf { W } _ { \mathrm { p r o j } } ,\tag{2}
$$

$$
\mathbf { E } _ { \mathrm { f u s e d } } = [ \mathbf { E } _ { \mathrm { o r i g } } , \mathbf { E } _ { \mathrm { e x p e r t } } ] \mathbf { W } _ { \mathrm { f u s e } } .\tag{3}
$$

The fused feature pool $\mathbf { E } _ { \mathrm { f u s e d } }$ provides a set of candidate alignment targets for Section 3.2 and Section 3.3, forcing the MLLM to internalize expert-enhanced visual semantics within its latent space.

## 3.2 Topology-Guided Sequencing Strategy

Motivated by the unique attention behaviors observed in our visual attribution analysis in Section 2.3, we introduce the Topology-Guided Sequencing

Strategy. This latent-space reasoning paradigm explicitly guides the model to follow the sequential perspective of topological logic, aligning its inherent tracking capabilities with the rigorous demands of circuit analysis.

Instead of relying on coarse bounding boxes that introduce background noise, we utilize the automated topology extraction workflow (Section 2.1) to obtain exact pixel-level localization for each topological query. Guided by the visual attention shifts of Circuit-MLLM and human domain expertise, we formulate a pixel-wise sequencing map, termed the Topological Logic Mask S(p):

$$
\begin{array} { r } { S ( p ) = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } p \in \varOmega _ { \mathrm { b g } } } \\ { 1 , } & { \mathrm { i f ~ } p \in \varOmega _ { \mathrm { r o o t } } } \\ { 2 + \frac { \mathcal { D } _ { \mathrm { B F S } } \left( p , \varOmega _ { \mathrm { r o o t } } \right) } { \operatorname* { m a x } _ { q \in \varOmega _ { \mathrm { w i r e } } } \mathcal { D } _ { \mathrm { B F S } } \left( q , \varOmega _ { \mathrm { r o o t } } \right) } , } & { \mathrm { i f ~ } p \in \varOmega _ { \mathrm { w i r e } } } \\ { 3 + \gamma _ { i } , } & { \mathrm { i f ~ } p \in \varOmega _ { \mathrm { c o n n } } ^ { ( i ) } } \end{array} \right. } \end{array}\tag{4}
$$

where $\mathcal { D } _ { \mathrm { B F S } }$ denotes the topological distance calculated via Breadth-First Search, capturing the "near-to-far" principle of visual attention. Meanwhile ${ \ , \gamma _ { i } \sim \mathcal { U } ( 0 , 1 ) }$ assigns a distinct continuous value to diferentiate connected devices.

To align this pixel-level mask with the patch-level features $\mathbf { E } _ { \mathrm { f u s e d } }$ , we apply Adaptive Max Pooling, yielding the patch-level Topological Logic Mask $\hat { S } _ { i , j } = \operatorname* { m a x } _ { p \in \mathcal { P } _ { i , j } } S ( p )$ . This acts as a conservative filter to preserve highly localized structures like thin wires. Crucially, after filtering out background features $( \hat { S } _ { i , j } \ > \ 0 )$ , we explicitly inject topological logic by defining a permutation π on the valid feature indices V. This permutation sorts the indices such that their corresponding mask values are monotonically increasing: ${ \hat { S } } _ { \pi ( 1 ) } \leq { \hat { S } } _ { \pi ( 2 ) } \leq$ $\cdots \leq \hat { S } _ { \pi ( | \nu | ) }$ . This seamlessly transforms the unordered 2D spatial features into a strictly topologically-ordered 1D sequence:

$$
\tilde { \mathbf { E } } _ { \mathrm { f u s e d } } = [ \mathbf { E } _ { \mathrm { f u s e d } , \pi ( 1 ) } , \mathbf { E } _ { \mathrm { f u s e d } , \pi ( 2 ) } , \dots , \mathbf { E } _ { \mathrm { f u s e d } , \pi ( | \mathcal { V } | ) } ] .\tag{5}
$$

To accommodate the MLLM’s fixed latent capacity $k ,$ a 1D Adaptive Average Pooling compresses this sequence into a fixed-length target $\mathbf { E } ^ { \ast } \in \mathbb { R } ^ { k \times d }$ :

$$
\mathbf { E } _ { m } ^ { * } = \frac { 1 } { | B _ { m } | } \sum _ { t \in B _ { m } } \tilde { \mathbf { E } } _ { \mathrm { f u s e d } , t } , \quad m \in \{ 1 , 2 , \ldots , k \} .\tag{6}
$$

This ordered, expert-enriched sequence serves as the target latent vectors for knowledge guidance. During the autoregressive generation process, we employ a Sequential Topological-Knowledge Alignment Loss $\left( \mathcal L _ { \mathrm { t o p o } } \right)$ to optimize the model by minimizing the cosine distance between the hidden state $h _ { m }$ at the final layer and the target latent vectors:

$$
\mathcal { L } _ { \mathrm { t o p o } } = \frac { 1 } { k } \sum _ { m = 1 } ^ { k } \left( 1 - \cos ( h _ { m } , \mathbf { E } _ { m } ^ { * } ) \right) .\tag{7}
$$

This mechanism efectively constrains the model to traverse predefined topological trajectories, thereby anchoring its latent representations within a logically structured and dynamically sequenced reasoning space.

## 3.3 Text-Latent Joint Supervision Training

While the latent tokens are supervised by the unified topological knowledge alignment objective $\mathcal { L } _ { \mathrm { t o p o } }$ , the surrounding text tokens are optimized via a standard autoregressive cross-entropy loss. Let x denote the input query and circuit image, and $\pmb { o } _ { \mathrm { p r e } } \ / \ o _ { \mathrm { p o s t } }$ denote the text segments respectively preceding and succeeding the generated latent tokens $\{ h _ { m } \} _ { m = 1 } ^ { k }$ . The textual loss $\mathcal { L } _ { \mathrm { t e x t } }$ is formulated as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { t e x t } } = \displaystyle \sum _ { i = 1 } ^ { | o _ { \mathrm { p r e } } | } \ell _ { \mathrm { C E } } \left( o _ { \mathrm { p r e } , i } , P _ { \theta } ( \pmb { x } , o _ { \mathrm { p r e } , < i } ) \right) } \\ & { \qquad + \displaystyle \sum _ { i = 1 } ^ { | o _ { \mathrm { p o s t } } | } \ell _ { \mathrm { C E } } \left( o _ { \mathrm { p o s t } , i } , P _ { \theta } ( \pmb { x } , o _ { \mathrm { p r e } } , \{ h _ { m } \} _ { m = 1 } ^ { k } , \pmb { o } _ { \mathrm { p o s t } , < i } ) \right) , } \end{array}\tag{8}
$$

where $P _ { \theta } ( \cdot )$ denotes the next-token prediction probability of the Circuit-MLLM. The overall training objective seamlessly combines both terms:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda \mathcal { L } _ { \mathrm { t o p o } } + \mathcal { L } _ { \mathrm { t e x t } } ,\tag{9}
$$

where λ is a balancing coeficient. This joint optimization anchors the latent tokens in the topologically-ordered expert visual space while weaving them naturally into the model’s explicit textual reasoning.

## 4 Experiments

## 4.1 Experimental Settings

Benchmarks. We evaluate our method on Circuit-MLLM-bench, a custom large-scale circuit benchmark. This benchmark is constructed via the Automated Circuit Schematic Topology Extraction pipeline detailed in Section 2.1 and has been rigorously verified by human experts. The benchmark evaluates models across two primary categories: topological analysis tasks, consisting of Connection Judgment and Connection Identification; and conventional detection tasks, which include Total Count, Type Count, and Element Class. Detailed construction procedures are provided in the supplementary material.

Data Synthesis. Driven by our automated topology extraction workflow, we generate question-answer (QA) pairs for each circuit image across the five tasks, alongside auxiliary images and pixel-level masks that delineate semantic regions and govern latent topological sequencing. We sample 5,000 instruction-tuning instances distributed by task complexity: Connection Identification (30%), Connection Judgment (25%), Element Class (15%), Total Count (15%), and Type Count (15%). To prevent data leakage, training images are strictly mutually exclusive from the Circuit-MLLM-bench evaluation set.

Baselines. We compare our approach against four baseline categories: (1) General-Purpose MLLMs: Mainstream models, encompassing closed-source models (e.g.,

Table 1: Performance comparison across all task categories. “Acc.”, “F1”, and “EMR” denote Accuracy, F1-score, and Exact Match Ratio, respectively. “Acc.” is evaluated on single-choice and counting tasks, whereas “F1” and“EMR" are applied to multiple-response tasks. “Avg.” represents the overall average score.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Size</td><td colspan="7">Circuit-MLLM-bench</td></tr><tr><td>Total Count Type Count Element Class Conn.Judge</td><td></td><td></td><td></td><td>Conn.Identi</td><td></td><td>Avg</td></tr><tr><td></td><td></td><td>Acc.↑</td><td>Acc.↑</td><td>Acc.↑</td><td>Acc.↑</td><td>F1↑</td><td>EMR.↑</td><td></td></tr><tr><td>GPT 5.1 GPT-40</td><td>/</td><td>43.60</td><td>60.62</td><td>96.49</td><td>61.11</td><td>76.08</td><td>30.20</td><td>61.35</td></tr><tr><td></td><td>/</td><td>42.01</td><td>60.08</td><td>97.40</td><td>58.50</td><td>78.68</td><td>36.30</td><td>62.16</td></tr><tr><td>Claude-4-Sonnet</td><td>/</td><td>44.91</td><td>58.23</td><td>95.13</td><td>57.13</td><td>69.54</td><td>24.80</td><td>58.17</td></tr><tr><td>Doubao-1.5-vision-pro / deepseek-vl2</td><td>27B</td><td>40.21 24.13</td><td>65.00 43.80</td><td>91.35 77.62</td><td>60.36 56.56</td><td>74.08 59.61</td><td>24.40</td><td>59.23</td></tr><tr><td>GLM-4.5V</td><td>106B</td><td>58.62</td><td>59.50</td><td>90.08</td><td>58.59</td><td>79.37</td><td>17.00 40.00</td><td>46.45 64.36</td></tr><tr><td rowspan="3">Qwen2.5-VL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7B</td><td>21.06</td><td>48.40 57.60</td><td>83.55 88.02</td><td>55.92 60.14</td><td>65.53 68.77</td><td>15.20 14.80</td><td>48.28</td></tr><tr><td>32B 72B</td><td>26.77 29.95</td><td>58.10</td><td>90.60</td><td>57.31</td><td>72.73</td><td>24.60</td><td>52.68 55.55</td></tr><tr><td rowspan="2">Qwen3-VL</td><td>8B</td><td>39.47</td><td>56.00</td><td>90.30</td><td>57.52</td><td>76.86</td><td>38.30</td><td>59.74</td></tr><tr><td>32B</td><td>49.63</td><td>60.00</td><td>92.95</td><td>58.59</td><td>82.02</td><td>42.90</td><td>64.35</td></tr><tr><td>Masala-CHAI</td><td>7B</td><td></td><td>48.10</td><td>93.10</td><td>51.00</td><td>66.35</td><td>19.10</td><td></td></tr><tr><td>Circuit-MLLM</td><td>7B</td><td>27.72 75.87</td><td>65.10</td><td>97.40</td><td>72.60</td><td>88.90</td><td>57.30</td><td>50.88 76.20</td></tr></table>

GPT 5.1,GPT-4o, Claude-4-Sonnet, Doubao-1.5-vision-pro) and open-source models (e.g., Qwen 2.5 and Qwen 3 series, DeepSeek-VL2, and GLM-4.5V). (2) Latent-Space Visual Reasoning: Mirage [19] and ILVR [4].(3) Tool-Augmented Circuit MLLMs: Masala CHAI [1], representing methods that explicitly invoke external tools to modify images prior to analysis. (4) Direct Fine-Tuning: Supervised Fine-Tuning (SFT) and SFT combined with Group Relative Policy Optimization (SFT+GRPO), representing models trained directly on our dataset without utilizing the latent-space mechanism.

Implementation Details. All experiments employ Qwen2.5-VL 7B as the foundational base model. Our framework is implemented using PyTorch and trained on 8 NVIDIA A100 GPUs. We conduct supervised fine-tuning for 15 epochs, utilizing a batch size of 8 alongside a cosine learning rate scheduler with an initial learning rate of 1e-5 across both stages. The latent token size is set to k = 4, and the loss balancing coeficient is configured to γ = 0.6.

## 4.2 Experimental Results

Results on all task categories. Table 1 presents the comparative evaluation results of various open-source and closed-source models across all task categories on Circuit-MLLM-Bench. In terms of average performance, Circuit-MLLM outperforms all existing baseline models, achieving a significant 57.8% relative improvement over its backbone, Qwen-2.5-VL-7B, and an approximate 22.5% improvement over the leading model GPT-4o. This underscores the efectiveness of Circuit-MLLM in solving complex circuit-related problems. Furthermore, although Masala-CHAI, a method utilizing external visual tools, holds a slight edge over Qwen-2.5-VL-7B in total count, element class, and connection identification, its overall gain is merely 4%, indicating the limitations of relying purely on external tools. Most importantly, on the highly challenging connection identification task, our method achieves a breakthrough in the Exact Match Ratio (EMR), reaching approximately three times that of the backbone model, thereby demonstrating its robust capability in parsing complex topologies.

Table 2: Performance comparison on topological analysis tasks. The evaluation encompasses two tasks: Connection Judgment and Connection Identification. The dificulty levels (Easy, Medium, and Hard) are determined by the number of connected components and the total number of components within the circuit diagram.
<table><tr><td rowspan="3">Models</td><td rowspan="3">Methods</td><td colspan="3">Connection Judge.</td><td colspan="6">Connection Identification</td><td rowspan="3">Avg</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Hard</td></tr><tr><td>Acc.↑</td><td>Acc.↑</td><td>Acc.↑</td><td>F1↑</td><td>EMR.↑</td><td>F1↑</td><td>EMR.↑</td><td>F1↑</td><td>EMR.↑</td></tr><tr><td rowspan="5">Qwen2.5-VL 7B</td><td>SFT GRPO</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>59.29</td><td>51.56</td><td>41.52</td><td>75.24</td><td>44.01</td><td>61.70</td><td>7.20</td><td>61.54</td><td>3.78</td><td>45.09</td></tr><tr><td>√</td><td>76.70</td><td>63.28</td><td>56.68</td><td>87.86</td><td>69.46</td><td>89.92</td><td>59.20</td><td>80.40</td><td>26.12</td><td>67.74</td></tr><tr><td>√ √</td><td>76.70</td><td>62.76</td><td>57.40</td><td>87.86</td><td>69.76</td><td>89.85</td><td>58.13</td><td>80.53</td><td>26.80</td><td>67.75</td></tr><tr><td>Masala-CHAI</td><td>55.16</td><td>53.91</td><td>41.88</td><td>75.70</td><td>45.21</td><td>61.87</td><td>6.93</td><td>61.46</td><td>4.47</td><td>45.18</td></tr><tr><td rowspan="2">Latent Reasoning</td><td>Mirage</td><td>77.88</td><td>66.15</td><td>64.98</td><td>90.41</td><td>72.16</td><td>89.48</td><td>58.67</td><td>81.88</td><td>26.80</td><td>69.82</td></tr><tr><td>ILVR</td><td>79.06</td><td>63.28</td><td>63.90</td><td>89.74</td><td>72.46</td><td>88.98</td><td>55.73</td><td>79.47</td><td>22.68</td><td>68.37</td></tr><tr><td>7B</td><td>Circuit-MLLM</td><td>81.42</td><td>70.31</td><td>64.98</td><td>91.44</td><td>76.35</td><td>91.10</td><td>62.67</td><td>83.54</td><td>28.52</td><td>72.26</td></tr></table>

Results on topological analysis tasks. To evaluate Circuit-MLLM in topological analysis, we fine-tuned the latent-space visual baselines, Mirage [19] and ILVR [4], using our latent-space dataset with the same latent size of 4 to ensure a fair comparison. As shown in Table 2, Circuit-MLLM consistently outperforms both Mirage and ILVR across all dificulty levels in Connection Judgment and Connection Identification. The average score increases by approximately 5%, demonstrating the overall efectiveness of our method. Notably, on the Hard dificulty of Connection Identification, our EMR exhibits a substantial 25% improvement over ILVR. Furthermore, Masala-CHAI performs almost identically to the backbone model, Qwen-2.5-VL-7B, on topological tasks, suggesting that explicitly modifying the original image yields marginal benefits. Compared to the SFT+GRPO baseline, Circuit-MLLM achieves an overall improvement of about 7%, with maximum gains reaching 13% on specific tasks. Finally, latentspace visual methods generally surpass the direct SFT baseline (without latent mechanisms) across topological tasks. This consistent superiority strongly proves the validity of conducting topological analysis directly within the latent space.

## 4.3 Ablation Study

Efectiveness of the Proposed Method. As shown in Table 3, the baseline latent reasoning model achieves an average score of 69.85. Introducing the Topology-Guided Sequencing Strategy alone improves the average score to 70.61, bringing consistent gains in Connection Judgment across all dificulty levels. Conversely, incorporating only the Latent-Space Circuit Knowledge mining mechanism raises the average score to 70.64, showing particular efectiveness in the Connection Identification task. When both components are integrated, Circuit-MLLM achieves the highest overall average score of 72.26, a 2.41 improvement over the baseline. The full model demonstrates superior performance across all Connection Judgment tasks and attains the highest EMR (28.52) on the Hard dificulty of Connection Identification. These quantitative results confirm that the two components are complementary, efectively enhancing the model’s topological analysis capabilities within the latent space.

Table 3: Ablation Study of our methods. “Seq.” and “Expert” refer to our two core components: the Topology-Guided Sequencing Strategy and Latent-Space Circuit Knowledge Distillation, respectively.
<table><tr><td rowspan="3">Models</td><td colspan="2">Method</td><td colspan="3">Connection Judge.</td><td colspan="6">Connection Identification</td><td rowspan="3">Avg</td></tr><tr><td rowspan="2"></td><td rowspan="2">Seq. Expert</td><td>Easy</td><td>Medium</td><td>Hard</td><td colspan="2">Easy</td><td colspan="2">Medium</td><td colspan="2">Hard</td></tr><tr><td>Acc.↑</td><td>Acc.↑</td><td>Acc.↑</td><td>F1↑</td><td>EMR.↑</td><td>F1↑</td><td>EMR.↑</td><td>F1↑</td><td>EMR.↑</td></tr><tr><td rowspan="4">Latent Reasoning 7B</td><td rowspan="2"></td><td></td><td>77.88</td><td>65.10</td><td>62.45</td><td>89.97</td><td>73.05</td><td>90.60</td><td>61.87</td><td>82.28</td><td>25.46</td><td>69.85</td></tr><tr><td></td><td>79.06</td><td>66.15</td><td>64.26</td><td>89.99</td><td>73.05</td><td>90.98</td><td>62.40</td><td>83.86</td><td>25.77</td><td>70.61</td></tr><tr><td rowspan="2"></td><td>√</td><td>77.88</td><td>65.62</td><td>62.50</td><td>90.64</td><td>73.35</td><td>91.22</td><td>63.47</td><td>83.83</td><td>27.15</td><td>70.64</td></tr><tr><td>√</td><td>81.42</td><td>70.31</td><td>64.98</td><td>91.44</td><td>76.35</td><td>91.10</td><td>62.67</td><td>83.54</td><td>28.52</td><td>72.26</td></tr></table>

Table 4: Ablation Study of Latent Size. The latent size denotes the total number of latent vectors generated during the inference phase. Across all experiments in this section, the loss balancing coeficient is uniformly configured to $\gamma = 0 . 6$
<table><tr><td rowspan="2">Models</td><td rowspan="2">Latent Size</td><td></td><td></td><td>Total Count Type Count Element Class Conn.Judge</td><td></td><td colspan="2">Conn. Ident</td><td rowspan="2">Avg</td></tr><tr><td>Acc.↑</td><td>Acc.↑</td><td>Acc.↑</td><td>Acc.↑</td><td>F1↑</td><td>EMR.↑</td></tr><tr><td rowspan="4">Latent Reasoning 7B</td><td>2</td><td>74.05</td><td>66.60</td><td>96.20</td><td>67.70</td><td>87.52</td><td>54.50</td><td>74.42</td></tr><tr><td>4</td><td>75.87</td><td>65.10</td><td>97.40</td><td>72.60</td><td>88.92</td><td>57.30</td><td>76.20</td></tr><tr><td>6</td><td>73.75</td><td>66.70</td><td>95.10</td><td>67.80</td><td>87.49</td><td>54.30</td><td>74.19</td></tr><tr><td>8</td><td>75.13</td><td>67.80</td><td>96.50</td><td>67.70</td><td>89.12</td><td>56.30</td><td>75.43</td></tr></table>

Selection of latent size. Table 4 presents the performance of Circuit-MLLM on the Circuit-MLLM-Bench across diferent latent sizes. In terms of the overall score, a latent size of 4 yields the best results, representing the most balanced choice. Furthermore, this indicates that 4 latent pads are suficient to encapsulate the reasoning process for conventional circuit problems.

Efectiveness of diferent circuit visual experts. Table 5 presents the performance of our three circuit visual experts, DINOv2, HAWP, and DeepLSD, across five circuit tasks. Employing all experts simultaneously yields the highest average score, demonstrating their collective efectiveness in circuit problem analysis. Furthermore, the individual contributions of each expert vary significantly across diferent problem types. For instance, utilizing only DINOv2 leads to strong performance on counting tasks, such as Total Count and Type Count, highlighting its precision in general object recognition; however, it struggles with connection-related problems. Conversely, relying solely on HAWP yields better results on Conn. Judge and Conn. Ident., verifying that visual experts specialized in wire detection are indeed highly efective for topology-related tasks.

Table 5: Ablation study of circuit experts. We evaluate the impact of diferent expert models, including DINOv2 , HAWP (Holistically-Attracted Wireframe Parsing, and DeepLSD (Deep Line Segment Detection). In all configurations, the latent size is set to 6, and the loss balancing coeficient is uniformly configured to γ = 0.6.
<table><tr><td colspan="3">Circuit Experts</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Total Count Type Count Element Class Conn.Judge</td><td rowspan="2"></td><td colspan="2">Conn. Ident</td><td rowspan="2">Avg</td></tr><tr><td>DINO HAWP DeepLSD</td><td></td><td>Acc.↑</td><td>Acc.↑</td><td>EMR.↑</td></tr><tr><td>√</td><td></td><td></td><td>75.23</td><td>Acc.↑ 66.80</td><td>94.50</td><td>Acc.↑ 69.00</td><td>F1↑ 88.60</td><td>56.80</td><td>75.16</td></tr><tr><td></td><td>√</td><td></td><td>71.11</td><td>65.90</td><td>95.70</td><td>70.60</td><td>89.16</td><td>57.20</td><td>74.95</td></tr><tr><td>√</td><td>√</td><td></td><td>74.49</td><td>66.50</td><td>96.40</td><td>71.00</td><td>88.16</td><td>54.80</td><td>75.23</td></tr><tr><td>√</td><td>√</td><td>√</td><td>75.87</td><td>65.10</td><td>97.40</td><td>72.60</td><td>88.92</td><td>57.30</td><td>76.20</td></tr></table>

![](images/7d813dd7c3d48aa978bec22632a45dc928a53f8310dd0103a17d3dc3510ba8d6.jpg)  
Fig. 5: Visual comparisons in the latent space. The latent features are captured prior to the generation of each <latent pad> token. Compared to Mirage and ILVR, our Circuit-MLLM achieves both higher fine-grained resolution and precise localization.

Additional Results. Further experimental results, including coeficient γ selection and Amsbench [17] evaluations, are provided in the supplementary material.

## 5 Analysis

Latent Space Attention Analysis. To investigate the visual regions focused on by the MLLM in the latent space, we extract the latent features from the last layer prior to the generation of each <latent pad> token. These features are utilized as latent vectors and mapped to the input image to generate heatmaps as shown in Figure 5. Observations indicate that when the input text queries specific components, our Circuit-MLLM accurately focuses on the root component, adjacent wires, and the final target component within the latent space. In contrast, Mirage and ILVR struggle to localize the regions of interest and exhibit lower fine-grained resolution, making it dificult to capture minute details such as wires. This demonstrates that our proposed Topological Logic Mask and Visual Expert efectively assist the model in capturing these crucial circuit topology features. Furthermore, we note that the latent space naturally avoids noise components and blank areas, proving that the latent space visual approach yields superior performance compared to direct image cropping.

![](images/41d120c3de3ad643e15df5deff04c7a8ae86b728908d500f091ed3ee9f26d5bc.jpg)  
Fig. 6: Dynamic visual shifts within the latent space. We demonstrate this using latent size of 6. Top: Global heatmaps of the penultimate layer’s latent representations overlaid on the input image. Bottom: Localized heatmaps focusing on critical regions for topological reasoning, including the root device, wire, and target device.

Latent Space Visual Shift Analysis. To verify whether Circuit-MLLM achieves dynamic visual shifts in the latent space, we extracted six consecutive latent feature vectors as shown in the Top of Figure 6. We specifically focus on the critical regions for topological reasoning: the root device, the wire, and the target device. As illustrated in the Bottom of Figure 6, a temporal progression of visual attention is observed: the model initially focuses heavily on the root device, followed by a gradual decay of attention in this area; the visual focus then shifts along the connecting wires, ultimately concentrating on the target device. This trajectory aligns with human analytical workflow, proving Circuit-MLLM performs accurate topological reasoning entirely within the latent space.

## 6 Conclusion

In this paper, we proposed Circuit-MLLM, the first topological logic-guided latent-space visual reasoning framework dedicated to complex circuit topology analysis. To address the inherent irregularity of topological regions in circuit diagrams, we pioneered the embedding of the topological reasoning process into the latent space. Extensive experiments demonstrate that by incorporating the circuit knowledge mining mechanism and the topology-guided sequencing strategy, Circuit-MLLM accurately focuses on the intended topological regions, leading to significant performance improvements in complex topological analysis tasks.

## Acknowledgements

This work was supported by NSFC (Grant No. U25A20485, W2412034).

## References

1. Bhandari, J., Bhat, V., He, Y., Rahmani, H., Garg, S., Karri, R.: Masala-chai: A large-scale spice netlist dataset for analog circuits by harnessing ai. arXiv preprint arXiv:2411.14299 (2024)

2. Chen, Z., Zhou, Q., Shen, Y., Hong, Y., Sun, Z., Gutfreund, D., Gan, C.: Visual chain-of-thought prompting for knowledge-based visual reasoning. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 1254–1262 (2024)

3. Cui, C., Sun, T., Liang, S., Gao, T., Zhang, Z., Liu, J., Wang, X., Zhou, C., Liu, H., Lin, M., et al.: Paddleocr-vl: Boosting multilingual document parsing via a 0.9 b ultra-compact vision-language model. arXiv preprint arXiv:2510.14528 (2025)

4. Dong, S., Wang, S., Liu, X., Li, C., Hou, H., Wei, Z.: Interleaved latent visual reasoning with selective perceptual modeling. arXiv preprint arXiv:2512.05665 (2025)

5. Feng, G., Zhang, B., Gu, Y., Ye, H., He, D., Wang, L.: Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems 36, 70757–70798 (2023)

6. Gao, J., Cao, W., Yang, J., Zhang, X.: Analoggenie: A generative engine for automatic discovery of analog circuit topologies. arXiv preprint arXiv:2503.00205 (2025)

7. Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N.A., Krishna, R.: Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems 37, 139348–139379 (2024)

8. Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Xu, Z., Hu, Y., Lin, S.: Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749 (2025)

9. Jocher, G., Qiu, J.: Ultralytics yolo11 (2024), https://github.com/ultralytics/ ultralytics

10. Li, B., Sun, X., Liu, J., Wang, Z., Wu, J., Yu, X., Chen, H., Barsoum, E., Chen, M., Liu, Z.: Latent visual reasoning. arXiv preprint arXiv:2509.24251 (2025)

11. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International conference on machine learning. pp. 12888–12900. PMLR (2022)

12. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)

13. Shao, H., Qian, S., Xiao, H., Song, G., Zong, Z., Wang, L., Liu, Y., Li, H.: Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems 37, 8612–8642 (2024)

14. Shen, Z., Yan, H., Zhang, L., Hu, Z., Du, Y., He, Y.: Codi: Compressing chainof-thought into continuous space via self-distillation. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 677–693 (2025)

15. Shi, Y., Tao, Z., Gao, Y., Huang, L., Wang, H., Yu, Z., Lin, T.J., He, L.: Amsnet 2.0: A large ams database with ai segmentation for net detection. arXiv preprint arXiv:2505.09155 (2025)

16. Shi, Y., Tao, Z., Gao, Y., Zhou, T., Chang, C., Wang, Y., Chen, B., Zhang, G., Liu, A., Yu, Z., et al.: Amsnet-kg: A netlist dataset for llm-based ams circuit autodesign using knowledge graph rag. ACM Transactions on Design Automation of Electronic Systems (2024)

17. Shi, Y., Zhang, Z., Wang, H., Tao, Z., Li, Z., Chen, B., Wang, Y., Yu, Z., Lin, T.J., He, L.: Amsbench: A comprehensive benchmark for evaluating mllm capabilities in ams circuits. arXiv preprint arXiv:2505.24138 (2025)

18. Xiang, K., Li, H., Zhang, T.J., Huang, Y., Liu, Z., Qu, P., He, J., Chen, J., Yuan, Y.J., Han, J., et al.: Seephys: Does seeing help thinking?–benchmarking visionbased physics reasoning. arXiv preprint arXiv:2505.19099 (2025)

19. Yang, Z., Yu, X., Chen, D., Shen, M., Gan, C.: Machine mental imagery: Empower multimodal reasoning with latent visual tokens. arXiv preprint arXiv:2506.17218 (2025)

20. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9556– 9567 (2024)

21. Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., Smola, A.: Multimodal chainof-thought reasoning in language models. arXiv preprint arXiv:2302.00923 (2023)

22. Zhu, E., Liu, Y., Zhang, Z., Li, X., Zhou, J., Yu, X., Huang, M., Wang, H.: Maps: Advancing multi-modal reasoning in expert-level physical science. arXiv preprint arXiv:2501.10768 (2025)

## A Additional Experimental Results

## A.1 Chain-of-Thought Paradigms on Topological Tasks

This section evaluates alternative Chain-of-Thought (CoT) paradigms on topologyrelated tasks to justify our architectural design. We compare two baselines, Visual CoT and Textual CoT, against our proposed Latent-Visual approach:

Visual CoT. Drawing inspiration from Visual SKETCHPAD and Visual CoT, we decompose the reasoning process into a two-stage pipeline. First, the model identifies the relevant topological regions and triggers visual cropping operations to extract and enlarge these areas. Subsequently, both the original global image and the cropped local patches are fed back into the model to formulate a comprehensive evaluation and yield the final result.

Textual CoT. As illustrated in Figure 8, we formulate the determination of connection relationships as a sequential three-step process: locating the root component, tracing along the connecting lines to extract key node coordinates, and ultimately judging the target connected component.

Results. The performance comparison is summarized in Table 6. First, using GPT-4o as the foundation model, we evaluate Visual CoT. While Visual CoT achieves a marginal improvement on the Connection Identification task, it degrades on the Connection Judge task. We attribute this degradation to the dificulty of accurately localizing fine-grained wire regions (as illustrated in Figure 7); the hard-cropping operation inevitably disrupts continuous topological structures and introduces visual artifacts. Furthermore, using Qwen2.5-VL 7B, we benchmark our proposed Latent-Visual approach against the Textual CoT. The Latent-Visual method outperforms Textual CoT across almost all metrics. We attribute this advantage to the inherent fragility of Textual CoT in circuit analysis: forcing the model to explicitly output long sequences of coordinates for topological paths is highly error-prone, whereas our latent-visual representations preserve structural integrity without relying on strict text generation.

Table 6: Performance comparison of diferent Chain-of-Thought (CoT) paradigms on topology tasks. Bold values indicate the best performance. Acc. and EMR denote Accuracy and Exact Match Ratio, respectively.
<table><tr><td rowspan="3">Model</td><td rowspan="3">Method</td><td colspan="3">Connection Judge.</td><td colspan="6">Connection Identification</td><td rowspan="3">Avg</td></tr><tr><td>Easy</td><td>Medium</td><td>Hard</td><td>Easy</td><td></td><td>Medium</td><td></td><td>Hard</td></tr><tr><td>Acc.↑</td><td>Acc.↑</td><td>Acc.↑</td><td>F1↑</td><td>EMR.↑</td><td>F1↑</td><td>EMR.↑</td><td>F1↑</td><td>EMR.↑</td></tr><tr><td rowspan="2">GPT-40</td><td></td><td>78.04</td><td>68.68</td><td>47.41</td><td>82.46</td><td>62.57</td><td>79.10</td><td>30.67</td><td>73.80</td><td>13.40</td><td>59.57</td></tr><tr><td>Visual CoT</td><td>68.44</td><td>60.94</td><td>43.68</td><td>82.54</td><td>60.48</td><td>79.66</td><td>31.73</td><td>72.18</td><td>11.00</td><td>56.74</td></tr><tr><td rowspan="2">Qwen2.5-VL 7B</td><td>Textual CoT</td><td>77.58</td><td>70.00</td><td>63.48</td><td>88.69</td><td>73.65</td><td>89.62</td><td>61.07</td><td>82.99</td><td>29.65</td><td>70.75</td></tr><tr><td>Latent-Visual 81.42</td><td></td><td>70.31</td><td>64.98</td><td>91.44</td><td>76.35</td><td>91.10</td><td>62.67</td><td>83.54</td><td>28.52</td><td>72.26</td></tr></table>

![](images/0780eb0dea1355df072f6532ea6f5dfe7e5c71d0ebaecef055307d6cbc201a48.jpg)

Fig. 7: Question-guided topology region cropping by GPT-4o. The black bounding boxes indicate the corresponding regions successfully cropped by the model based on the given questions, while the red dashed boxes highlight the missed regions.  
![](images/5ef6c263a3d55632c61d26ef3f3f072bc34dc047cb1f8c4d85c0839ecc3301f6.jpg)  
Fig. 8: Illustration of the Textual CoT reasoning process for topology analysis, detailing three sequential stages: locating, tracing, and judging.

## A.2 Sim Weight γ

To explore the optimal similarity weight $\gamma ,$ we conducted experiments across various values of $\gamma$ with a fixed latent size of 4. The overall loss function is defined as $\mathcal { L } _ { \mathrm { t o t a l } } = \gamma \mathcal { L } _ { \mathrm { t o p o } } + \mathcal { L } _ { \mathrm { t e x t } }$ . As shown in Figure 9, performance across all tasks peaks when γ is set to 0.6, demonstrating that latent space alignment and text prediction mutually reinforce each other.

## A.3 Amsbench

In addition to our custom Circuit-MLLM-Bench, we evaluate our approach on AMSBench [17]. We compare against mainstream general-purpose MLLMs, including closed-source (e.g., GPT-5.1, GPT-4o, Claude-4-Sonnet, Doubao-1.5- vision-pro) and open-source models (e.g., Qwen 2.5 and 3 series, DeepSeek-VL2, GLM-4.5V). We also include Masala CHAI [1] as a baseline representing methods that explicitly modify images via external tools prior to analysis.

As shown in Table 7, Circuit-MLLM outperforms all baselines on Type Counting and the topology-focused Connection Identification tasks, demonstrating its strong capability across general circuit tasks. Notably, on Connection

![](images/d004e2b8bcec174a42e7314b2940e0d63cf5777319b3df337f31ca3fc7c11579.jpg)  
Fig. 9: Ablation results of diferent sim weight γ values. Left: Performance on individual tasks. Right: Average scores across all tasks.

Identification, our model achieves a 58% improvement over its backbone, Qwen-2.5-VL 7B, and an approximate 15% increase over GPT-4o, underscoring its proficiency in topology analysis. Additionally, we observe that our Circuit-MLLM underperforms models such as GPT5.1 on general circuit problems, specifically the total count category in amsbench. We attribute this performance gap to the fact that the regions of interest for total count tasks typically span the entire image, which renders additional focusing mechanisms in the latent space less effective. Conversely, Masala CHAI underperforms the Qwen-2.5-VL 7B backbone across almost all tasks, suggesting that explicit pixel-level modifications on the original image ofer limited efectiveness.

Table 7: Performance comparison on Amsbench. We report Accuracy (Acc.) and Mean Squared Error (MSE) for the Total Counting and Type Counting tasks, and the F1-score (F1) for the Connection Identification task.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Size</td><td colspan="5">Amsbench</td></tr><tr><td>Total Counting</td><td></td><td>Type Counting</td><td></td><td>Connection Identification</td></tr><tr><td></td><td></td><td>Acc.↑</td><td>MSE.↓</td><td>Acc.↑</td><td>MSE.↓</td><td>F1↑</td></tr><tr><td>GPT-5.1</td><td>/</td><td>35.53</td><td>17.96</td><td>58.26</td><td>10.92</td><td>65.43</td></tr><tr><td>GPT-40</td><td>/</td><td>51.00</td><td>19.05</td><td>54.00</td><td>28.18</td><td>65.00</td></tr><tr><td>Claude-3.7-Sonnet</td><td>/</td><td>36.00</td><td>18.38</td><td>55.00</td><td>24.18</td><td>71.00</td></tr><tr><td>Doubao-1.5-vision-pro deepseek-vl2</td><td>/ 27B</td><td>24.00 13.40</td><td>38.13 76.83</td><td>51.00 30.93</td><td>24.76 27.11</td><td>64.00 30.85</td></tr><tr><td>GLM-4.5V</td><td>106B</td><td>48.20</td><td>25.59</td><td>58.73</td><td>23.79</td><td>51.27</td></tr><tr><td rowspan="2">Qwen3-VL</td><td></td><td>15.87</td><td>137.57</td><td>35.40</td><td>29.26</td><td></td></tr><tr><td>8B 32B</td><td>27.67</td><td>28.33</td><td>54.80</td><td>19.73</td><td>41.27 46.89</td></tr><tr><td rowspan="3">Qwen2.5-VL</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7B</td><td>17.07</td><td>84.48 60.21</td><td>49.00 50.53</td><td>35.49 31.49</td><td>47.36</td></tr><tr><td>32B 72B</td><td>20.13 43.00</td><td>19.59</td><td>54.33</td><td>18.59</td><td>47.83 52.00</td></tr><tr><td>Masala-CHAI</td><td>7B</td><td>14.00</td><td>95.93</td><td>45.27</td><td>50.44</td><td></td></tr><tr><td>Circuit-MLLM</td><td>7B</td><td>36.05</td><td>38.28</td><td>60.46</td><td>5.64</td><td>44.79 75.01</td></tr></table>

## B Circuit-MLLM-Bench

Unlike AMSBench, our Circuit-MLLM-Bench strictly evaluates the visual information extraction capabilities of MLLMs on circuit diagrams. To construct this benchmark, we aggregated schematics from four primary non-benchmark datasets: AMSnet [16], AnalogGenie [6], Masala-Chai [1], and MAPs [22].

As illustrated in Figure 10, the collected images are first deduplicated. Subsequently, comprehensive topological data is extracted using the automated pipeline detailed in Section 2.1. This topology serves as the foundation for synthesizing five distinct task categories: Connection Judge, Connection Identification, Total Count, Type Count, and Element Class. While Total Count yields exactly one query per schematic, the other categories generate multiple instances, resulting in over 20 question-answer (QA) pairs per image.

For rapid and eficient evaluation, the Circuit-MLLM-Bench utilized in this study comprises a curated subset of 1,000 schematics, containing 1,000 QA pairs per task (5,000 QA pairs in total). Note that this pipeline is highly scalable, capable of generating hundreds of thousands of QA pairs if applied to the entire schematic corpus. To ensure rigorous data quality, the generated dataset underwent preliminary automated verification via Qwen-3-VL (32B), followed by meticulous manual review by domain experts.

![](images/71d1c50dd1919062836a6c37a9a6ae351d4a2b29436ccad33a3870412c933558.jpg)  
Fig. 10: Construction of Circuit-MLLM-Bench.

## C Case Studies on Topology Analysis with Circuit-MLLM

As illustrated in Figure 11, in contrast to GPT-5.1, which frequently struggles to interpret complex and densely routed circuit connections, Circuit-MLLM demonstrates superior performance by efectively utilizing its latent-visual reasoning capabilities to accurately identify all interconnected components.

![](images/7c5f2182c4e2b0355187e4bf8c6b37181f07e15eb58288f0007bd18ce0dd623f.jpg)  
Fig. 11: Representative examples of topology analysis.