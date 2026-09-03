# DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents

Zhuoran Yu <sup>1</sup> Le Thien Phuc Nguyen <sup>1</sup> Jaden Park <sup>1</sup> Xinyi Gu <sup>2</sup> Zexue He <sup>3</sup> Soochahn Lee <sup>4</sup> Rogerio Feris <sup>5</sup> Yong Jae Lee <sup>1</sup>

## Abstract

Multimodal Large Language Models (MLLMs) have achieved strong performance on structured visual understanding tasks such as chart and document question answering. However, existing benchmarks typically evaluate these domains in isolation, leaving underexplored a key capability: whether models can use textual context to determine how chart evidence should be selected, interpreted, and aggregated. We introduce DOCHOP, a benchmark for integrated chart–context reasoning in document-style images. In DOCHOP, the document narrative specifies multi-step compositional constraints, while charts provide the corresponding data values. Questions are grounded on a semantic reference label defined in the narrative, requiring models to resolve target entities from context before aggregating evidence across multiple charts. To enable systematic evaluation, we construct DOCHOP via a stochastic logic-first generation pipeline with controllable reasoning depth and visual density, covering 2,074 examples across six task categories. Experiments on a wide range of proprietary and open-source MLLMs show a substantial gap to human performance: annotators achieve over 90% accuracy, while the best model reaches only 62.83%. Reasoningenhanced models consistently show improved results, but performance degrades as reasoning complexity increases. Overall, DOCHOP provides a controlled testbed for challenging multi-hop document reasoning.

## 1. Introduction

Recent Multimodal Large Language Models (MLLMs) (OpenAI, 2025; Liu et al., 2024b; Bai et al., 2025a;b; Comanici et al., 2025; Liu et al., 2023; Dai et al., 2023) have shown promising results in structured image understanding, spanning domains such as charts (Masry et al., 2022; 2025; Xu et al., 2024; Wang et al., 2024; Xia et al., 2025), documents (Mathew et al., 2021), and webpage-based interfaces (Liu et al., 2024d;c). However, existing benchmarks predominantly evaluate these domains in isolation. For example, chart-centric and document-centric datasets are typically constructed and tested separately, each focusing on a particular form of visual information. This separation overlooks a key aspect of real-world presentation: the interplay between visual data and textual context. In many practical scenarios, charts and text function together, where the surrounding narrative provides definitions, contextual grounding, or reasoning instructions that are essential for correctly interpreting the accompanying visual evidence. These scenarios highlight an important but underexplored capability: models must bind textual constraints to visual evidence and then use the grounded evidence for downstream reasoning.

In this paper, we introduce DOCHOP<sup>1</sup>, a benchmark designed to evaluate MLLMs on integrated document understanding that combines chart-based evidence with surrounding narrative context. The core intuition is to separate reasoning logic from numerical evidence: the document context specifies the multi-step constraints, while the charts provide the corresponding data values. Models must jointly interpret the document narrative and the chart values, using the described reasoning trace to locate and aggregate visual evidence across multiple charts. This design enables a rigorous evaluation of visual-intensive multi-hop reasoning, rather than simple single-hop lookup. In the following paragraphs, we highlight the key properties of DOCHOP.

Integrated Chart-Context Reasoning. DOCHOP consists of document-style images in which multiple charts are interleaved with explanatory narrative context. The narrative does not merely provide background text; it specifies the multi-step reasoning constraints that govern how candidate entities should be selected across charts. Crucially, it concludes by assigning a semantic reference label to the entities obtained by executing this reasoning trace under its logical composition. Questions are then formulated to refer to this label rather than explicit entity names, requiring models to first resolve the target entities from the document context and subsequently retrieve and aggregate the corresponding numerical evidence from the charts. This design ensures that correct answers depend on joint understanding of both the narrative constraints and the visual data. An illustrative example can be found in Figure 1.

![](images/5eb6b3a6ffb52497175b4762e97e9c86a3148ac39865345458ca6bdcf7e18271.jpg)  
Figure 1. Illustration of an example in DocHop. Each document instance corresponds to a stochastically-sampled multi-hop reasoning trace, which is transformed into a narrative context interleaved with multiple charts. The narrative specifies the compositional constraints of the trace and concludes by assigning a semantic reference label to the entities implied by its execution. Questions are then grounded on this label rather than explicit entity names, requiring models to resolve targets from the document context and aggregate numerica evidence from the charts. The colored arrows indicate where the constraints are presented in the document; the black arrow indicates the specific chart the model needs to look at to check if the input entities satisfy the constraint.

Controllable Reasoning Complexity. To achieve a rigorous evaluation, we employ a stochastic logic-generation pipeline rather than relying on noisy web-scraped data. For each instance, we first sample a reasoning trace—a structured chain of conditional checks that necessitates aggregating information from multiple distinct charts. We then prompt an LLM to generate both the underlying numerical data and the accompanying textual document to strictly satisfy this pre-defined trace. This reverse-engineering approach ensures semantic consistency between the text and the charts. Crucially, it allows us to explicitly control difficulty along two axes: reasoning depth (the number of logical hops) and visual density (the number of charts). This results in a diverse difficulty distribution, enabling a systematic diagnostic of model limitations across the entire complexity spectrum—from lower-depth chart–context reasoning to high-load multi-step aggregation—as detailed in Section 3.

Diverse Reasoning Contexts. Our stochastic generation process ensures high variance across logical, visual, and thematic dimensions. The reasoning traces are randomly generated to produce varied structures, ranging from linear chains to multi-branch compositions. Together, these variations broaden the coverage of chart–context reasoning patterns evaluated by the benchmark. Across semantic domains, the benchmark spans 480 distinct topics, and its documents include 7 different chart types. In total, DocHop comprises 2,074 examples covering 6 distinct tasks: Value Retrieval, Counting, Numeric Reasoning, Ranking, Hypothetical Reasoning, and Fact Checking. These are instantiated via 41 unique question templates, ensuring a comprehensive evaluation of model generalizability.

Multi-Stage Quality Verification. To ensure reliability, we implement a multi-stage verification protocol. First, leveraging the structured nature of our generation pipeline, we programmatically validate the numerical data in every example. Since the underlying data tables and reasoning traces are fully accessible, we execute the trace against the data tables to deterministically verify that the generated chart values satisfy the instantiated constraints and that the ground-truth answers are mathematically correct. Second, we conduct human verification of the document context to ensure that the generated narrative faithfully verbalizes the underlying reasoning trace, including the textual constraints, logical compositions, and semantic reference label. Third, we manually inspect the rendered document images to ensure visual readability and completeness, checking for issues such as truncated context, overlapping layout elements, occluded charts, or illegible labels. This combination of programmatic validation and human review supports high-fidelity evaluation standards.

Since DOCHOP is synthetically constructed rather than sampled from naturally occurring documents, we position it as an out-of-domain multi-hop reasoning benchmark for chart– document integration. Its goal is not to model a specific real-world document distribution, but to test whether models can use narrative context to identify the relevant chart evidence and aggregate it across multiple charts.

We evaluate DocHop on a wide range of proprietary and open-source MLLMs. The results show that DocHop remains challenging but not intractable: human annotators achieve over 90% accuracy, while the bestperforming model (GPT-5.2 Reasoning (OpenAI, 2025)) reaches 62.83% overall. Reasoning-enhanced variants consistently outperform their non-reasoning counterparts across both GPT and Gemini families. Proprietary models remain stronger overall, with open-source baselines typically demonstrating 9–24% accuracy in our evaluation.

Moreover, DocHop enables controlled analysis along reasoning depth and chart-count axes, where performance degrades steadily as either complexity factor increases (Figure 5). Finally, we provide qualitative failure case studies to better understand common breakdowns in narrative grounding and cross-chart evidence aggregation.

## 2. Related Work

## 2.1. Multimodal Large Language Models

Recent Multimodal Large Language Models (MLLMs) extend large language models with visual encoders to jointly process images and text. Through large-scale pretraining and instruction tuning, both proprietary and open-source MLLMs have demonstrated strong performance across a wide range of multimodal understanding tasks. Representative proprietary systems include the GPT series (OpenAI, 2025), Gemini (Comanici et al., 2025), and Claude models (Anthropic, 2025a;b). In parallel, open-source models such as LLaVA-style architectures (Liu et al., 2024b; 2023), IDEFICS (Laurenc¸on et al., 2024; Laurenc¸on et al., 2024), Qwen-VL (Bai et al., 2025b;a), Molmo (Deitke et al., 2025), and InternVL (Wang et al., 2025) provide competitive performance and have enabled reproducible research across diverse benchmarks. These general-purpose MLLMs adopt unified architectures that integrate visual perception and language reasoning, enabling zero-shot and few-shot generalization across tasks involving images, documents, and structured layouts. As a result, they have become standard baselines for evaluating multimodal understanding capabilities in contemporary benchmarks. However, their capabilities in complex multi-modal multi-hop reasoning tasks remain underexplored.

## 2.2. Benchmarks for Structured Multimodal Reasoning

A large body of benchmarks has been proposed to evaluate visual reasoning over structured images—particularly in charts, documents, and webpages. Early chart question answering datasets such as FigureQA (Kahou et al., 2017), DVQA (Kafle et al., 2018), LEAF-QA (Chaudhry et al., 2020) , and PlotQA (Methani et al., 2020) rely heavily on questions generated from limited templates.

ChartQA (Masry et al., 2022) introduces human-authored questions on real-world charts and improves linguistic diversity, but still focuses on single-chart reasoning without requiring external textual context. More recent benchmarks on chart and document question answering, including ChartBench (Xu et al., 2024), MMC (Liu et al., 2024a), ChartX (Xia et al., 2025), ChartXiv (Wang et al., 2024), ChartMuseum (Tang et al., 2025), ChartQAPro (Masry et al., 2025), and DocVQA (Mathew et al., 2021) increase image diversity and question difficulty by incorporating scientific figures, multi-chart inputs, or open-ended answers. Recently, DocHop-QA (Park et al., 2025) studies multimodal multi-hop question answering over document collections. However, these benchmarks largely follow a single-hop or limited multi-chart paradigm, where the reasoning logic is implicit in the question rather than explicitly provided and followed across modalities. In contrast to prior benchmarks, DocHop is designed to evaluate multi-round cross-modal reasoning within a single document.

## 3. DocHop

DocHop is constructed with a logic-first, reverse-engineered framework. The key design is to separate reasoning specification from numerical evidence: a symbolic reasoning trace defines the multi-step constraints, the document narrative verbalizes these constraints, and the charts provide the corresponding data values. To enforce joint chart-context reasoning, the narrative concludes by assigning a semantic reference label to the entities implied by executing the trace under its logical composition, and all questions are grounded on this label rather than explicit entity names. This prevents direct chart-only lookup and requires models to resolve targets from the narrative before aggregating evidence from charts. We organize the construction process into four main components: (1) Symbolic Reasoning Trace Generation, (2) Chart and Document Context Generation, (3) Document Composition and QA Curation, and (4) Quality Verification.

![](images/fefbd174de821c86969fabb2a579c72cce67325fe7d30843ce898bee98916a0e.jpg)  
Figure 2. Illustration of the DocHop data generation pipeline. (1) Metadata Pool: We first prompt an LLM to curate a diverse pool of semantic metadata, including topics, entity names, candidate metrics, and timestamps that may appear in an instance. (2) Chart Configuration Sampling: From this pool, we sample a set of chart schemas by selecting chart types (e.g., bar, line) and instantiating their required structural fields from the metadata. (3) Reasoning Trace Generation: Conditioned on the sampled chart configurations, we stochastically construct a multi-hop reasoning trace, where each node corresponds to an instantiated constraint parameterized by a specific chart schema. (4) Chart and Document Context Generation: Given the symbolic trace, we prompt an LLM to generate constraint-satisfying chart tables and an accompanying narrative context that verbalizes the trace, while keeping the two processes independent. (5) QA Curation and Document Rendering: Finally, we assign a semantic reference label in the narrative, instantiate task-specific question–answer pairs grounded on this label, and render the charts and text into a single-page document image for evaluation.

## 3.1. Symbolic Reasoning Trace Generation

This subsection describes the first three stages of the pipeline: metadata pool curation, chart configuration sampling, and symbolic reasoning trace generation. The goal is to construct an abstract reasoning problem before any concrete chart values are generated.

We first establish a semantic context by prompting a large language model (LLM) to generate a diverse metadata pool. Each pool defines the global semantic space of an instance, including the document topic, candidate entities, measurable metrics, semantic groups that subdivide metric values, value units, and available timestamps. From this pool, we sample a set of k chart schemas, where $k \in \{ 2 , 3 , 4 , 6 \}$ . Each schema instantiates a chart structure by selecting metadata fields appropriate for a particular chart type, such as tracking a metric over time or comparing metric values across groups. At this stage, the schemas specify only the structural components of the charts, without any concrete numerical values; the values are generated later conditioned on the reasoning trace in the chart and document context generation stage.

Conditioned on the sampled chart schemas, we construct a symbolic reasoning trace modeled as a logical tree with a target reasoning depth $D \in \{ 2 , 3 , 4 , 5 \}$ . The trace contains two types of nodes. Constraint nodes represent parameterized rules instantiated from templates and filter candidate entities based on chart-grounded conditions. Logical composition nodes combine entity sets from their input nodes using Boolean operations. Directed edges govern how candidate entity sets are passed through the tree. For a child constraint node, the node inherits the candidate entity pool from its parent and applies its rule to further filter this set. Consequently, the entities satisfying a child constraint node form a subset of those satisfying its parent. The trace is generated recursively through the following mechanisms:

Node Parameterization. To promote constraint diversity, we employ a library of 27 distinct rule templates (see Appendix C) designed by human annotators. Each template is formulated as a parameterized function (e.g., <Timestamp>: <Metric> <Operator> <Threshold>) that must be instantiated with concrete values when sampled. During generation, the pipeline first samples a target subset of entities from the valid pool passed down by the parent node, and then populates the remaining slots (e.g., operator and threshold) to form a definite constraint (e.g., Products A and B satisfy the constraint “2024-06: Items Sold ≥ 45”). This ensures that the constraint is both structurally valid and logically consistent with the preceding context.

Logical Composition. Logical composition nodes aggregate constraints by combining entity sets from their input nodes. These nodes are parameterized by sampling a Boolean operator, specifically AND or OR. Unlike constraint nodes that filter entities through attribute-based conditions, the output entity set of a logical composition node is deterministic and logic-driven: it is computed by directly applying the sampled Boolean operator to its input entity sets.

The algorithm iteratively samples and composes these nodes until the target depth D is reached. Due to the stochastic nature of this recursive generation, the process yields diverse reasoning skeletons, ranging from simple linear chains to complex multi-branch trees. The resulting logical structure specifies how entities should be filtered and combined across the sampled chart schemas, and serves as the blueprint for subsequent data synthesis. Examples of these diverse tree structures are provided in Appendix B.

## 3.2. Chart and Document Context Generation

Given a sampled reasoning trace and its associated chart schemas, we synthesize the two complementary sources of information in each DOCHOP instance: chart tables that provide numerical evidence, and document narratives that verbalize the reasoning specification. We generate these two components independently so that the document context describes the constraints without leaking the underlying chart values or the final entity assignments. We use Gemini-2.5-Pro for all LLM-based synthesis steps, and full prompt details are provided in Appendix A.

Chart Data Generation. We generate numerical tables that satisfy the instantiated constraints in the reasoning trace. A naive approach would be to prompt the model to produce all charts jointly conditioned on the full multi-hop trace; however, as reasoning depth and chart count grow, it becomes difficult to reliably ensure that the generated tables satisfy all specified constraints simultaneously. To improve reliability, we adopt a per-chart synthesis strategy: we traverse the reasoning tree and aggregate all regular-node constraints associated with the same chart into a single requirement set. The prompt explicitly specifies which entities must satisfy or violate each condition, and the LLM generates the corresponding chart table independently for each chart. We then verify correctness by executing the reasoning trace programmatically. When converting these tables into chart images, we randomly sample a compatible chart type and color scheme for each chart, introducing diversification in both graphical form and appearance.

Document Context Generation. In parallel, we verbalize the symbolic reasoning trace into a coherent document narrative. Context generation is performed independently from chart synthesis: the LLM has no access to chart values or to the final entity set produced by executing the trace. Instead, it is provided only with the instantiated constraint structure, including the parameterized conditions at each node and their logical composition. This information is sufficient to describe the reasoning procedure in natural language, while preventing the narrative from revealing numerical evidence or directly naming the resulting target entities.

## 3.3. Question and Answer Curation and Document Rendering

Semantic Reference Label. After generating the document narrative, we assign a semantic reference label (e.g., Platinum Tier Night Service in Figure 1) to the entities determined by executing the reasoning trace under its logical composition. Questions are grounded on this label rather than explicit entity names, preventing direct chartonly lookup and requiring models to first identify the relevant entity set from the narrative before aggregating numerical evidence from the charts. For instance, instead of asking for the average metric value of specific hotels (e.g., HTH, SCS, WG in Figure 1), a question is phrased as: “What is the average Linens Replaced under Night Shift in August 2025 of the hotel(s) receiving the Platinum Tier Night Service Commendation?”

Task Definitions. Following prior chart question answering benchmarks, we formulate DOCHOP questions into six reasoning categories:

• Value Retrieval: Retrieve a metric of entities assigned the semantic reference label, or directly ask which entities are assigned the semantic reference label.

• Counting: Count how many entities are assigned the semantic reference label, optionally under an additional chart-grounded condition specified in the question.

• Numeric Reasoning: Perform arithmetic aggregation (e.g., sum or average) over chart entries of entities assigned the semantic reference label.

• Ranking: Identify extrema or order entities assigned the semantic reference label based on their chart values.

• Hypothetical Reasoning: Answer counterfactual queries that modify either chart values while keeping the semantic reference label fixed, or the reasoning trace itself by introducing additional constraints.

• Fact Checking: Verify whether a statement is entailed by jointly applying the textual constraints and chart evidence over entities assigned the semantic reference label.

Question and Answer Construction. For each task type, we start from a corresponding question template and instantiate it using the synthesized document metadata. These templates (41 in total) are designed by human annotators to cover diverse reasoning patterns across tasks. Questions are grounded on the semantic reference label defined in the narrative rather than explicit entity names in the charts, ensuring that answering requires integrating information from the narrative context and the associated charts. For example, a numeric reasoning question may ask: “What is the total Transaction Costs Paid in December 2024 of all investment funds receiving the Certified Prudent Operator designation?”. Since the reasoning trace deterministically defines the target entity set, ground-truth answers are computed exactly by executing the corresponding operations over the generated chart tables. We provide the full list of question templates under each task in Appendix D.

![](images/d5c36f887b3ea5d1ef6fb10d6495278813e3fa39eeae2ed91738e654ac35e506.jpg)  
(a) Distribution by Task

![](images/b0cf219f09bddc86137a096d6380449712a98f71877a5f7ceb70f933ae2436f2.jpg)  
(b) Distribution over Reasoning Depth and Number of Charts  
Figure 3. Dataset distribution of DOCHOP. We report the distribution across task categories (left) and the distribution over reasoning depths and numbers of charts per document (right).

Document Rendering. We render the synthesized document narrative and chart images into a single-page document using ReportLab with a fixed A4-style layout; the question and answer are kept separately and are not included in the rendered document. To ensure that each instance fits on one page, we start from font size 10 and progressively decrease the font size until the narrative fits. We also adjust chart scaling across different subplot layouts so that individual subplots remain roughly comparable in visual size. The final document is rendered at 300 DPI and paired with its corresponding question and verified answer to form one DO-CHOP evaluation instance. The entire data curation pipeline is illustrated in Figure 2.

## 3.4. Quality Verification

We apply a multi-stage verification process to ensure the correctness and readability of DOCHOP instances.

Programmatic Verification of Chart Data. Since the symbolic reasoning trace and generated chart tables are fully accessible, we verify each instance by executing the trace against the chart data. This allows us to deterministically check whether the generated numerical values satisfy all instantiated constraints and whether the resulting target entity set and ground-truth answer are correct. Instances that fail this verification are discarded and regenerated.

Human Verification of Document Context. We then manually inspect whether the generated document narrative faithfully verbalizes the underlying reasoning trace. This step checks whether the textual constraints, logical compositions, and semantic reference label are expressed clearly and consistently with the symbolic trace. Instances with missing, ambiguous, or incorrect descriptions of the reasoning process are rejected.

Final Rendering Verification. Finally, we inspect the rendered document images to ensure that the document context and charts are fully visible and readable. This includes checking for truncated text, missing context, overlapping elements, occluded charts, illegible labels, or other layout artifacts introduced during PDF rendering. Instances that fail the final rendering check are regenerated.

## 3.5. Dataset Statistics

DocHop contains 2,074 document instances, each paired with a single question-answer example. We construct the benchmark with controllable reasoning complexity along two axes: reasoning depth $D \in \{ 2 , 3 , 4 , 5 \}$ and the number of charts per document $k \in \{ 2 , 3 , 4 , 6 \}$ . Each example is assigned to one of six task categories (Value Retrieval, Counting, Numeric Reasoning, Ranking, Hypothetical Reasoning, and Fact Checking), instantiated from a library of question templates.

During generation, we aim to balance the sampling across task types as well as across different depth and chart-count configurations, enabling systematic evaluation over a broad range of reasoning difficulty. The example distributions of DocHop can be found in Figure 3.

DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents
<table><tr><td>Model</td><td>Value Retrieval</td><td>Counting</td><td>Numeric Reasoning</td><td>Ranking</td><td>Hypothetical</td><td>Fact Checking</td><td>Overall</td></tr><tr><td>Human Eval</td><td>95.36</td><td>94.15</td><td>95.53</td><td>93.56</td><td>86.54</td><td>96.67</td><td>93.29</td></tr><tr><td>Random Guessing (Gemini (Comanici et al., 2025))</td><td>0.00</td><td>4.97</td><td>0.69</td><td>0.00</td><td>9.21</td><td>45.56</td><td>10.10</td></tr><tr><td>Random Guessing (GPT (OpenAI, 2025))</td><td>0.00</td><td>8.35</td><td>0.23</td><td>0.00</td><td>10.79</td><td>47.83</td><td>11.27</td></tr><tr><td colspan="8">Proprietary Reasoning Models</td></tr><tr><td>GPT-5.2-Reasoning (OpenAI, 2025)</td><td>66.56</td><td>67.84</td><td>46.93</td><td>60.77</td><td>58.70</td><td>78.79</td><td>62.83</td></tr><tr><td>GPT-5-mini-Reasoning (OpenAI, 2025)</td><td>29.14</td><td>23.68</td><td>9.50</td><td>31.19</td><td>23.67</td><td>55.76</td><td>28.25</td></tr><tr><td>Gemini-2.5-Pro-Reasoning (Comanici et al., 2025)</td><td>41.39</td><td>49.12</td><td>17.60</td><td>39.55</td><td>35.73</td><td>63.33</td><td>40.60</td></tr><tr><td>Gemini-2.5-Flash-Reasoning (Comanici et al., 2025)</td><td>35.10</td><td>40.94</td><td>11.17</td><td>28.62</td><td>22.27</td><td>58.48</td><td>32.02</td></tr><tr><td colspan="8">Proprietary Models</td></tr><tr><td>GPT-5.2 (OpenAI, 2025)</td><td>40.73</td><td>56.14</td><td>19.83</td><td>37.30</td><td>35.50</td><td>55.15</td><td>40.36</td></tr><tr><td>Gemini-2.5-Flash (Comanici et al., 2025)</td><td>27.15</td><td>25.73</td><td>8.94</td><td>28.30</td><td>16.94</td><td>46.36</td><td>24.88</td></tr><tr><td>Claude-4.5-Haiku (Anthropic, 2025a)</td><td>2.98</td><td>19.01</td><td>1.96</td><td>4.82</td><td>11.37</td><td>58.79</td><td>16.35</td></tr><tr><td>Claude-4.5-Sonnet (Anthropic, 2025b)</td><td>4.64</td><td>28.65</td><td>2.51</td><td>7.72</td><td>10.44</td><td>55.15</td><td>17.94</td></tr><tr><td colspan="8">Open-Source Models</td></tr><tr><td>LLaVA-Next-LLaMA3-8B (Liu et al., 2024b)</td><td>0.00</td><td>9.65</td><td>0.56</td><td>0.00</td><td>8.82</td><td>49.09</td><td>11.33</td></tr><tr><td>IDEFICS2-8B (Laurençon et al., 2024)</td><td>2.32</td><td>4.39</td><td>1.40</td><td>1.61</td><td>8.12</td><td>35.45</td><td>8.87</td></tr><tr><td>IDEFICS3-LLaMA3-8B (Laurençon et al., 2024)</td><td>7.62</td><td>9.36</td><td>1.96</td><td>3.86</td><td>5.57</td><td>37.27</td><td>10.66</td></tr><tr><td>Qwen-2.5-VL-7B (Bai et al., 2025b)</td><td>16.89</td><td>13.74</td><td>4.19</td><td>24.12</td><td>12.06</td><td>46.97</td><td>19.05</td></tr><tr><td>Qwen3-VL-8B (Bai et al., 2025a)</td><td>16.89</td><td>22.51</td><td>10.61</td><td>32.15</td><td>16.47</td><td>49.70</td><td>24.16</td></tr><tr><td>Molmo-7B-Q-0924 (Deitke et al., 2025)</td><td>4.30</td><td>23.98</td><td>2.51</td><td>12.22</td><td>11.60</td><td>46.97</td><td>16.73</td></tr><tr><td>Molmo-7B-D-0924 (Deitke et al., 2025)</td><td>9.93</td><td>21.05</td><td>2.23</td><td>13.18</td><td>10.21</td><td>41.52</td><td>16.01</td></tr><tr><td>Ovis1.6-Gemma2-9B (Lu et al., 2024)</td><td>1.32</td><td>8.48</td><td>0.28</td><td>1.93</td><td>7.89</td><td>43.94</td><td>10.56</td></tr><tr><td>InternVL-3.5-8B (Wang et al., 2025)</td><td>3.31</td><td>10.23</td><td>1.96</td><td>4.82</td><td>8.58</td><td>53.03</td><td>13.45</td></tr><tr><td>InternVL-3.5-30B-A3B (Wang et al., 2025)</td><td>2.32</td><td>19.30</td><td>2.79</td><td>2.89</td><td>11.83</td><td>49.39</td><td>14.75</td></tr><tr><td>Qwen-2.5-VL-32B (Bai et al., 2025b)</td><td>21.85</td><td>19.01</td><td>6.70</td><td>22.51</td><td>16.47</td><td>47.27</td><td>21.79</td></tr><tr><td>InternVL-3.5-38B (Wang et al., 2025)</td><td>4.97</td><td>21.93</td><td>4.19</td><td>3.86</td><td>9.05</td><td>50.61</td><td>15.57</td></tr><tr><td>Qwen-2.5-VL-72B (Bai et al., 2025b)</td><td>23.84</td><td>26.32</td><td>7.82</td><td>25.08</td><td>17.87</td><td>48.79</td><td>24.40</td></tr></table>

Table 1. Evaluation results on DocHop across task categories. We report both per-task accuracy and the overall accuracy of DocHop.

## 3.6. Evaluation Metrics

We evaluate model performance using normalized answer accuracy. For non-numeric questions, we use caseinsensitive exact match. For numeric questions, we normalize outputs to a canonical format and apply tolerance-based matching under the required precision. For binary factchecking questions, accuracy is computed over Yes/No predictions. Additional evaluation details, such as prompting instructions, answer parsing rules, and numeric precision handling, are provided in the Appendix E.

## 4. Experiments

## 4.1. Evaluation Setup

Evaluated Models. We evaluate a broad range of state-ofthe-art MLLMs, including both proprietary and open-source systems. Our proprietary model set includes GPT-5.2 (OpenAI, 2025), GPT-5-mini (OpenAI, 2025), Gemini-2.5- Pro/Flash (Comanici et al., 2025), Claude 4.5 Haiku (Anthropic, 2025a), and Claude 4.5 Sonnet (Anthropic, 2025b). We additionally report results for their reasoning-enhanced variants when available (e.g., GPT-5.2-Reasoning, Gemini-2.5-Pro-Reasoning, Gemini-2.5-Flash-Reasoning).

For open-source baselines, we consider representative vision–language models across multiple families and scales, including LLaVA-Next-LLaMA3-8B (Liu et al., 2024b), IDEFICS2-8B (Laurenc¸on et al., 2024), IDEFICS3- LLaMA3-8B (Laurenc¸on et al., 2024), Qwen-2.5-VL (7B/32B/72B) (Bai et al., 2025b), Qwen3-VL-8B (Bai et al.,

2025a), Molmo-7B (O/D) (Deitke et al., 2025), Ovis1.6- Gemma2-9B (Lu et al., 2024) and InternVL-3.5 (8B/30B-A3B/38B) (Wang et al., 2025). Full model configuration settings are provided in Appendix F.

Human and Random Guessing Baselines. We obtain human performance by hiring graduate student annotators, who are presented with the full document image and asked to provide free-form answers. Following ChartXiv (Wang et al., 2024), we also obtain a question-only guessing baseline by prompting GPT-5.2 (OpenAI, 2025) and Gemini-2.5-Pro (Comanici et al., 2025) with only the question text, without providing the document image. This serves as a language-prior baseline.

## 4.2. Main Results

Table 1 summarizes the full evaluation results. Here, we summarize the main takeaways.

Overall Gap to Human Performance. All evaluated models, including both proprietary and open-source ones as well as reasoning variants, remain far below human accuracy. While human annotators achieve over 90% accuracy, the best-performing model (GPT-5.2 Reasoning) reaches only 62.83% overall, indicating a substantial gap in integrated chart-context reasoning. It is also worth noting that it takes approximately five minutes per question on average for human annotators to solve one question, suggesting that DocHop requires careful multi-step reasoning rather than quick visual lookup. The large discrepancy between human and model performance highlights the difficulty of document-level multi-hop reasoning for current MLLMs.

![](images/ab164595df91bc9b463e6d188b441989f7ed102acd1fa5db36c18c2c7665b4fc.jpg)  
Figure 4. Qualitative examples of reasoning outputs. (a) An instance where GPT-5.2 Reasoning answers correctly, while Gemini-2.5-Pro fails by attributing chart values to the wrong entity, leading to an incorrect aggregation. (b) A challenging case where both models make incorrect predictions but for different reasons: GPT identifies the global maximum instead of restricting to entities receiving the semantic label, whereas Gemini resolves the correct entity set from the narrative but makes an OCR mistake in the final value extraction.

Reasoning Models Consistently Outperform Their Non-Reasoning Counterparts. As shown in Table 1, models with explicit reasoning mechanisms achieve substantial gains over their base variants across both GPT and Gemini families. For example, GPT-5.2 improves from 40.36% to 62.83% overall when switching to its reasoning variant, and Gemini-2.5-Flash increases from 24.88% to 32.02%. Gemini-2.5-Pro with a dynamic reasoning mechanism further reaches 40.60% overall accuracy, outperforming the Gemini-2.5-Flash variants. Overall, these results indicate that explicit reasoning remains an important factor for stronger performance on DocHop.

Proprietary Models Remain Stronger Overall. Among all evaluated models, GPT-5.2 Reasoning achieves the best overall performance, reaching 62.83% accuracy on DocHop. Gemini-2.5-Pro Reasoning is the next strongest proprietary baseline at 40.60%, though a substantial gap remains compared to GPT. More broadly, proprietary models generally outperform the open-source baselines in our current evaluation. Most open-source models achieve overall accuracies in the range of roughly 9–24%, indicating that integrated chart–context reasoning remains challenging for current open vision–language models.

## 4.3. Analysis

Model performance degrades with more charts and higher reasoning depth. Figure 5 shows that the performance of both GPT-5.2 Reasoning and Gemini-2.5-Pro Reasoning consistently decreases as the number of charts in the document and the reasoning depth increase. Accuracy is highest in simpler settings with fewer charts (k = 2) and shallow traces (D = 2), and drops steadily as the document becomes visually denser and the reasoning trace requires more hops. This trend suggests that scaling DocHop along either axis imposes additional difficulty for current MLLMs in aggregating evidence across multiple charts under narrative constraints.

Qualitative Examples. Figure 4 shows two representative DocHop instances. (a) Attribution Error. In Figure 4(a), GPT-5.2 Reasoning model answers correctly by grounding the reasoning steps on the right entity throughout. Gemini-2.5-Pro, however, makes an attribution mistake: it confuses the chart values of two entities, leading to an incorrect aggregation despite otherwise plausible intermediate reasoning. (b) Different Failure Modes on the Same Instance. Figure 4(b) illustrates a case where both models fail, but for different reasons. GPT-5.2 Reasoning loses track of the narrative constraint and reports the global maximum rather than the maximum among entities receiving the SLN Designation. Gemini-2.5-Pro correctly identifies the qualified entity set, but produces an incorrect final answer due to a misread chart value. These examples suggest that DocHop remains challenging even for strong proprietary models, with errors in both contextual reasoning and chart understanding.

![](images/647492a743a1cbe70a6dc53e37cb7b127b16b9d5ab4273f9803f806ebf9baba5.jpg)  
(a) Performance by Reasoning Depth

![](images/294a3e94b674288d218c3812a1e1b9ab0458a8d0debcaf186f72fffc0a2a98c4.jpg)  
(b) Performance by Number of Charts  
Figure 5. Performance of GPT-5.2 and Gemini-2.5-Pro under controlled DocHop complexity. We report accuracy of GPT-5.2 Reasoning (OpenAI, 2025) and Gemini-2.5-Pro Reasoning (Comanici et al., 2025), both operating with explicit reasoning enabled, as reasoning depth and number of charts increase. Performance generally degrades for both models as either the reasoning trace becomes deeper or the document becomes visually denser, highlighting the increasing difficulty of cross-chart evidence aggregation under narrative constraints.

## Limitations and Future Work

DOCHOP is designed as a controlled out-of-domain reasoning benchmark rather than a dataset that reproduces the distribution of naturally occurring documents. This design allows us to isolate chart–document integration under controllable reasoning depth and visual density, but it also means that performance on DOCHOP should be interpreted as a diagnostic measure of a specific reasoning capability rather than as a direct estimate of performance on real-world document collections. Future work can extend this logic-grounded evaluation framework to naturally occurring reports, scientific articles, business dashboards, and other document domains.

In addition, DOCHOP focuses on chart-based numerical evidence paired with narrative context. Real documents often contain other structured visual elements, such as tables, diagrams, forms, equations, and user-interface screenshots. Extending the logic-grounded construction framework to these document elements would enable broader evaluation over heterogeneous document structures. Our current rendering pipeline also uses controlled single-page A4-style layouts with readable charts and narrative text. This design reduces confounding factors from severe OCR noise, document parsing failures, and layout artifacts, allowing the benchmark to focus on chart–document reasoning; however, it does not capture the full visual variability of naturally occurring documents. Future work can introduce multipage documents, noisier scans, and more diverse layout styles. Finally, although our construction pipeline combines programmatic verification with human review, synthetic benchmark generation may still leave occasional annotation, wording, or rendering issues. We plan to maintain the released benchmark with versioned updates and incorporate validated community feedback in future releases.

## 5. Conclusion

We introduced DOCHOP, a benchmark for integrated chart– context reasoning in document-style images, where models must resolve narrative constraints and aggregate numerical evidence across multiple charts. A key design is the use of semantic reference labels grounded in the document narrative, which prevents direct chart-only lookup and enforces joint reasoning over text and visual data. Experiments across both proprietary and open-source MLLMs reveal a substantial gap to human performance, while reasoning-enhanced variants consistently improve accuracy over their base counterparts. We further observe a steady degradation as reasoning depth and chart count increase, highlighting persistent challenges in multi-hop evidence aggregation under dense visual contexts. Future work can extend this logic-grounded evaluation to other structured visual elements (e.g., tables, diagrams, and UI documents) to better characterize multimodal reasoning in realistic settings.

## Impact Statement

This paper presents work whose goal is to advance the field of multimodal machine learning through the development of DOCHOP, a benchmark for evaluating integrated chart–context reasoning in document images. Our work is primarily intended to support more reliable assessment of multimodal reasoning capabilities, and does not introduce new deployment-facing model components. Potential societal impacts are therefore indirect, such as improving the robustness and transparency of evaluation for systems used in document understanding. As with other synthetic benchmark datasets, misuse could arise if models are overoptimized for benchmark performance without corresponding real-world generalization. We encourage future work to consider broader coverage of document domains and to complement benchmark-driven progress with responsible evaluation practices.

## Acknowledgment

This work was supported in part by NSF IIS2404180, IBM, Institute of Information & communications Technology Planning& Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration), and Electronics and Telecommunications Research Institute (ETRI) grant (26CB1200, Development and Application of Science-Specialized Multimodal Foundation Models).

## References

Anthropic. Claude haiku 4.5, October 2025a. URL https://www.anthropic.com/news/ claude-haiku-4-5. Accessed: 2026-01-28.

Anthropic. Introducing claude sonnet 4.5, September 2025b. URL https://www.anthropic.com/ news/claude-sonnet-4-5. Accessed: 2026-01- 28.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Chaudhry, R., Shekhar, S., Gupta, U., Maneriker, P., Bansal, P., and Joshi, A. Leaf-qa: Locate, encode & attend for figure question answering. In Proceedings ofthe IEEE/CVF winter conference on applications of computer vision, pp. 3512–3521, 2020.

Comanici, G., Bieber, E., and et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.

Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P. N., and Hoi, S. Instructblip: Towards generalpurpose vision-language models with instruction tuning. Advances in neural information processing systems, 36: 49250–49267, 2023.

Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J. S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., et al. Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 91–104, 2025.

Kafle, K., Price, B., Cohen, S., and Kanan, C. Dvqa: Understanding data visualizations via question answering. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pp. 5648–5656, 2018.

Kahou, S. E., Michalski, V., Atkinson, A., Kad´ ar,´ A.,<sup>´</sup> Trischler, A., and Bengio, Y. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings ofthe 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Laurenc¸on, H., Tronchon, L., Cord, M., and Sanh, V. What matters when building vision-language models? Advances in Neural Information Processing Systems, 37: 87874–87907, 2024.

Laurenc¸on, H., Marafioti, A., Sanh, V., and Tronchon, L. Building and better understanding vision-language models: insights and future directions., 2024.

Liu, F., Wang, X., Yao, W., Chen, J., Song, K., Cho, S., Yacoob, Y., and Yu, D. Mmc: Advancing multimodal chart understanding with large-scale instruction tuning. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 1287–1310, 2024a.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024b. URL https://llava-vl.github.io/blog/ 2024-01-30-llava-next/.

Liu, J., Ou, T., Song, Y., Qu, Y., Lam, W., Xiong, C., Chen, W., Neubig, G., and Yue, X. Harnessing webpage uis for text-rich visual understanding. arXiv preprint arXiv:2410.13824, 2024c.

Liu, J., Song, Y., Lin, B. Y., Lam, W., Neubig, G., Li, Y., and Yue, X. Visualwebbench: How far have multimodal llms evolved in web page understanding and grounding? arXiv preprint arXiv:2404.05955, 2024d.

Lu, S., Li, Y., Chen, Q.-G., Xu, Z., Luo, W., Zhang, K., and Ye, H.-J. Ovis: Structural embedding alignment for multimodal large language model. arXiv preprint arXiv:2405.20797, 2024.

Masry, A., Do, X. L., Tan, J. Q., Joty, S., and Hoque, E. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pp. 2263–2279, 2022.

Masry, A., Islam, M. S., Ahmed, M., Bajaj, A., Kabir, F., Kartha, A., Laskar, M. T. R., Rahman, M., Rahman, S., Shahmohammadi, M., Thakkar, M., Parvez, M. R., Hoque, E., and Joty, S. Chartqapro: A more diverse and challenging benchmark for chart question answering, 2025. URL https://arxiv.org/abs/2504. 05506.

Mathew, M., Karatzas, D., and Jawahar, C. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Methani, N., Ganguly, P., Khapra, M. M., and Kumar, P. Plotqa: Reasoning over scientific plots. In Proceedings ofthe ieee/cvfwinter conference on applications ofcomputer vision, pp. 1527–1536, 2020.

OpenAI. Openai gpt-5 system card, 2025. URL https: //arxiv.org/abs/2601.03267.

OpenAI. Introducing gpt-5.2, December 2025. URL https://openai.com/index/ introducing-gpt-5-2/. Accessed: 2026-01-28.

Park, J., Pyeon, S., Kim, J., Cabal, R. C., Ding, Y., and Han, S. C. Dochop-qa: Towards multi-hop reasoning over multimodal document collections. arXiv preprint arXiv:2508.15851, 2025.

Tang, L., Kim, G., Zhao, X., Lake, T., Ding, W., Yin, F., Singhal, P., Wadhwa, M., Liu, Z. L., Sprague, Z., Namuduri, R., Hu, B., Rodriguez, J. D., Peng, P., and Durrett, G. Chartmuseum: Testing visual reasoning capabilities of large vision-language models, 2025. URL https://arxiv.org/abs/2505.13444.

Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., Wang, Z., Chen, Z., Zhang, H., Yang, G., Wang, H., Wei, Q., Yin, J., Li, W., Cui, E., Chen, G., Ding, Z., Tian, C., Wu, Z., Xie, J., Li, Z., Yang, B., Duan, Y., Wang, X., Hou, Z., Hao, H., Zhang, T., Li, S., Zhao, X., Duan, H., Deng, N., Fu, B., He, Y., Wang, Y., He, C., Shi, B., He, J., Xiong, Y., Lv, H., Wu, L., Shao, W., Zhang, K., Deng, H., Qi, B., Ge, J., Guo, Q., Zhang, W., Zhang, S., Cao, M., Lin, J., Tang, K., Gao, J., Huang, H., Gu, Y., Lyu, C., Tang, H., Wang, R., Lv, H., Ouyang, W., Wang, L., Dou, M., Zhu, X., Lu, T., Lin, D., Dai, J., Su, W., Zhou, B., Chen, K., Qiao, Y., Wang, W., and Luo, G. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. URL https://arxiv.org/abs/2508.18265.

Wang, Z., Xia, M., He, L., Chen, H., Liu, Y., Zhu, R., Liang, K., Wu, X., Liu, H., Malladi, S., et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37: 113569–113697, 2024.

Xia, R., Zhang, B., Ye, H., Yan, X., Liu, Q., Zhou, H., Chen, Z., Ye, P., Dou, M., Shi, B., Yan, J., and Qiao, Y. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning, 2025. URL https: //arxiv.org/abs/2402.12185.

Xu, Z., Du, S., Qi, Y., Xu, C., Yuan, C., and Guo, J. Chartbench: A benchmark for complex visual reasoning in charts, 2024. URL https://arxiv.org/abs/ 2312.15915.

## A. LLM Prompts for Data Curation

We provide the prompt templates used in the construction of DocHop. Given a sampled symbolic reasoning trace (Section 3), we employ an LLM to synthesize both the underlying chart tables and the accompanying document narrative. To ensure reliability and prevent information leakage, chart generation and context generation are performed independently: char prompts include explicit constraint satisfaction requirements, while narrative prompts only receive the instantiated constraint structure without access to chart values or entity assignments.

Across the pipeline, we use three main classes of prompts: (i) Metadata Pool Curation, which generates diverse topics, entities, metrics, and timestamps; (ii) Per-chart Table Synthesis, which produces numerical tables satisfying aggregated chart-specific constraints; and (iii) Narrative Context Generation, which verbalizes the reasoning trace into policy-style document text and introduces the semantic reference label used for question grounding. All prompts are executed with Gemini-2.5-Pro.

## Metadata Pool Prompt.

You are MetaSetWriter. Return ONLY a single JSON object.   
TOPIC: "{topic}"   
WHAT TO RETURN   
{   
"entity\_type": "<singular type label>",   
"entities": ["<E names>"],   
"groups": ["<G orthogonal subgroups>"],   
"metrics": ["<M independent metrics in SAME family>"],   
"unit": "<A concrete plural noun representing the measurement unit>",   
"data\_type": "<’integer’ OR ’float’>",   
"timestamps": ["<T strictly increasing stamps (e.g., Years ’2020’ or Months ’2022-01’)   
>"]   
}   
LOGIC RULES (CRITICAL)   
1. NO MATHEMATICAL REDUNDANCY (INDEPENDENCE):   
- Metrics must be independent variables so that random values do not contradict each   
other.   
- THE "PART vs. WHOLE" RULE:   
- CONFLICT: Do NOT include both a "Total" and its "Parts" in the same list.   
- Bad: ["Total Revenue", "Service Revenue", "Product Revenue"]   
(Because Product + Service might not equal Total in random data).   
- Solution A (Components Only): ["Service Revenue", "Product Revenue", "Consulting   
Revenue"].   
- Solution B (Aggregates Only): ["Total Revenue", "Total Cost", "Total Tax"].   
- THE "FORMULA" RULE:   
- CONFLICT: Do NOT include variables that are simple math functions of others.   
- Bad: ["Revenue", "Cost", "Profit"] (Because Profit = Revenue - Cost).   
- Fix: Pick a level of abstraction. E.g., just ["Revenue", "Cost", "Marketing Spend"]   
OR just ["Net Profit", "EBITDA"].   
2. DESCRIPTIVE METRIC NAMES (NATURAL LANGUAGE FRIENDLY):   
- Choose metric names that sound natural in a sentence (e.g., "The <Metric> was high").   
- GOOD: "Bottles Produced", "Patients Admitted", "Parcels Delivered".   
- BAD: "Total Produced", "Count", "Number".   
- Note: It is OK if the Metric name repeats the Unit name   
(e.g., Metric: "Bottles Produced", Unit: "Bottles") if it improves clarity.   
3. UNIT SCALING & SPECIFICITY:   
- NO generic "counts" or "values". Use specific nouns: "Parcels", "Patients", "Tons",   
Joules".   
- SCALING: Use scale prefixes (Thousand, Million, Billion) for ANY unit if the entity   
scale justifies it.   
- "Million USD" (for Corporations)

```jsonl
"Thousand Liters" (for Water Plants)
- "Metric Tons" (for Mining)
- "Million Users" (for Tech Platforms)
- NO RATES: NO "percent" or "ratio". Metrics must be summable.
4. GROUPS MUST BE UNIVERSAL CATEGORIES (ORTHOGONAL):
- THE GOLDEN RULE: Every single Metric must make sense for Every single Group.
- GOOD: Groups=["North", "South"] (Applies to almost anything).
- BAD: Metric="Virus Cases", Groups=["Viral", "Bacterial"]
(A Virus cannot be Bacterial -> Logic Fail).
- Groups must act as buckets that categorize the metrics, regardless of what the metric
is.
SAFETY & CONTENT GUIDELINES (STRICT)
1. NO BIOLOGICAL FLUIDS/GORE:
- Ban: "Blood", "Organs", "Bodily Fluids", "Kills".
- Alt: "Essence", "Samples", "Elixir", "Units".
2. NO VIOLENCE/HARM:
- Ban: "Fatalities", "Casualties", "Deaths".
- Alt: "Incidents", "Transfers", "Losses", "Expirations".
3. NO SENSITIVE REAL-WORLD TOPICS:
- Avoid real religious figures, political hate speech, or illicit contraband.
- Keep fantasy/cult topics PG-13 (e.g., "Mana", "Relics").
4. NO REAL-WORLD ENTITIES (USE FICTIONAL REALISM):
- Ban: Real country names (e.g., USA, China), real cities (e.g., Paris, Tokyo),
real famous people/politicians, or real company names.
- Ban: Lazy placeholders (e.g., "Region A", "Person X", "Country 1").
- Do: Invent realistic-sounding but fictional names
(e.g., "Westhaven", "Port Meridian", "Director Vance", "Silver Creek", "The
Nordic Union").
{entity_type_clause}
IN-CONTEXT EXAMPLE A (Financial/Annual - No Math Conflicts)
{
"entity_type": "conglomerate",
"entities": ["Aether Corp", "Nebula Logistics", "Terra Firm"],
"groups": ["North America", "EMEA", "APAC"],
"metrics": ["Global Revenue", "Liquidity Assets", "Market Cap"],
"unit": "Million USD",
"data_type": "float",
"timestamps": ["2018","2019","2020","2021","2022","2023"]
}
IN-CONTEXT EXAMPLE B (Operational/Monthly - Scaled Physical Unit)
{
"entity_type": "bottling_plant",
"entities": ["Vortex Springs", "Alpine Pure", "Crystal Cove"],
"groups": ["Glass Bottles", "PET Plastic", "Aluminum Cans"],
"metrics": ["Bottles Produced", "Bottles Scrapped", "Bottles Shipped"],
"unit": "Thousand Bottles",
"data_type": "integer",
"timestamps": ["2023-01","2023-02","2023-03","2023-04","2023-05","2023-06"]
```

## Data Synthesis Prompt–Line Plot as Example.

You must generate the chart below. The chart must strictly follow the chart type, metrics,   
timestamps, groups, and rules assigned to it. All entity-level constraints must be

```csv
satisfied exactly.
IMPORTANT CONSTRAINT INTERPRETATION:
- When a rule specifies that certain entities MUST satisfy a constraint, those entities
must strictly satisfy it.
Entities NOT mentioned in a constraint MUST VIOLATE that constraint.
For example: ’For entities A, B: value >= 70’ means A and B must have value >= 70, while
all other entities must have value < 70.
You must return the final answer in this JSON format:
{
"{chart_id}": "<csv string>"
}
Where the <csv string> is valid CSV, with:
- First column = entity name
All numerical values are in: {unit} .
Note the scale ’{unit}’: A value of 50 means 50 {unit}.
Column headers MUST be exactly the metric/group names provided. DO NOT append the year
e.g. use ’Revenue’, NOT ’Revenue (2017)’).
No commentary, no markdown, no chain-of-thought, no explanation.
You MUST NOT leave the chart empty.
1 CSV string may be long; that is allowed.
You MUST output real values for the chart.
Each value should be a float rounded to two decimal places between 20.00 and 99.99.
Do not make all values end with .00 or .50 to avoid looking artificial.
Example endings can include .23, .87, .14, .69, etc.
Bad examples (DO NOT output):
20.00, 21.00, 22.00, 23.50, 24.50
Good examples (follow these patterns):
20.13, 21.47, 22.84, 23.26, 24.79
CRITICAL VISUALIZATION REQUIREMENT FOR LINE CHARTS:
- When plotted, the lines must be visually distinguishable.
Try your best to curate values so that at each timestamp, no two entities will have
close values.
- i.e, try to avoid two points being too close when plotting the data.
Example Output 1:
{
"{chart_id}": "Entity,2017,2018,2019,2020,2021,2022,2023,2024,2025\n
A,85.23,87.91,86.45,88.17,87.29,89.73,88.42,86.84,87.56\n
B,52.84,58.39,49.17,55.73,61.29,47.84,54.18,59.62,51.47\n
C,38.47,42.91,46.23,50.18,53.84,57.29,60.73,64.18,67.92\n
D,23.91,26.47,29.18,31.84,28.73,32.19,35.42,33.76,36.84"
}
NOTE: A (high 80s, stable), B (mid 50s, highly volatile),
C (rising from 38 to 68), D (low 30s, slow rise)
Example Output 2:
{
"{chart_id}": "Entity,2017,2018,2019,2020,2021,2022,2023,2024,2025\n
A,89.23,84.67,79.41,74.85,70.29,66.73,63.18,59.84,56.47\n
B,44.81,41.29,38.73,36.92,39.47,43.18,47.84,52.29,56.73\n
C,67.34,71.92,76.18,79.84,75.29,70.47,65.91,61.28,57.73\n
D,28.47,24.91,21.73,25.29,29.84,34.18,38.73,43.29,47.91"
}
NOTE: A (high 70s, steady decline), B (mid 40s, U-shaped),
C (high 60s-70s, inverted-U), D (low 30s, V-shaped)
```

Both examples show diverse patterns - your data should also have varied trends.   
## CHART {chart\_id}   
Type: LINE   
Entities: {entity1, entity2, entity3, ...}   
Unit: {unit}   
Metric: {metric\_name}   
Timestamps: {2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025}   
Rules:   
{constraint\_1}   
{constraint\_2}   
{constraint\_3}

## Context Curation Prompt.

You are an expert Policy Drafter. Your goal is to translate a decision tree (provided in   
DOT format) into a professional, natural language policy document.   
Return a single JSON object:   
{   
"title": "A professional, bureaucratic title for the policy",   
"final\_label": "The specific name of the final status/award",   
"text": "The full policy text in Markdown format..."   
}   
I. Generic Graph Grammar & Narrative Rules   
1. Serial Topology (The Filtering Pipeline)   
Structure: ‘Node A -> Node B‘ (Direct arrow connection).   
<sub>\*</sub> Interpretation: A strict dependency. Node B is not an independent step; it operates   
only on the subset of entities that have successfully passed Node A.   
<sub>\*</sub> Narrative Requirement: You must explicitly signal Inherited Eligibility at the start   
of the second node’s description. Clarify that the subsequent criteria apply   
exclusively to the specific pool of candidates that survived the previous filter.   
2. Convergent Topology (Combinatorial Logic)   
<sub>\*</sub> Structure: Multiple arrows converging into a single ‘COMBO‘ node.   
<sub>\*</sub> Edge Order Rule: The order of incoming edges in the ‘dot\_string‘ defines the   
operand roles (Crucial for Subtraction):   
- 1st Edge: Left Operand (Input A).   
- 2nd Edge: Right Operand (Input B).   
3. Logic Definitions & Narrative Requirements   
<sub>\*</sub> AND (Intersection)   
- Meaning: Simultaneous satisfaction of all inputs.   
- Req: Describe as a unified, mandatory standard where no criterion is optional.   
<sub>\*</sub> OR (Union)   
- Meaning: Any input suffices.   
- Req: Describe as alternative pathways or flexible qualification routes.   
II. NARRATIVE ARCHITECTURE (Randomly assigned)   
You must follow the specific structure directed in the user prompt:   
1. STRUCTURE\_A (Conclusion-Led): State the {final\_label} in the first paragraph as   
the policy’s primary goal. Then, detail the prerequisite requirements.   
2. STRUCTURE\_B (Criteria-Led): Detail the requirements phase-by-phase, introducing   
the {final\_label} only in the concluding summary.   
III. WRITING GUIDELINES (FORMULA-TO-POLICY TRANSLATION)   
- STORYTELLER, NOT CALCULATOR: Do not simply list data points. Write the policy as   
a cohesive narrative about the {entity\_type}’s performance history and compliance

```prolog
expectations.
FORMULA IS TRUTH: The user provides logic nodes in mathematical notation
(e.g., A/B >= 50%). These are the absolute ground truth. Your job is to translate
these mathematical relationships into professional policy prose.
DE-MECHANIZATION: Do not simply read the formula aloud.
<sub>*</sub> BAD: "The value of X divided by the sum of Y must be greater than 0.5."
BAD: If a node’s formulate requires comparing with other entities, DO NOT WRITE:
"calculated across all entities in the dataset". USE "all participating
{entity_type}" instead.
PROFESSIONAL PROSE: You have full creative freedom to rephrase the technical labels
from the DOT nodes into smooth, authoritative bureaucratic language.
TERMINOLOGY: Replace generic "entity" with the specific ‘entity_type‘ provided
(e.g., "contract", "farm").
HANDLING EXTREME VALUES (MAX/MIN Constraints):
Context: If a node rule says "value == the max value" or "value == the min value".
<sub>*</sub> Requirement A (Competitive Ranking): You MUST interpret this as a ranking.
Explicitly state that the entity recorded the highest (or lowest) absolute figure
in the comparative set. Use keywords like "Unrivaled," "Market-leading,"
"Highest recorded value."
<sub>*</sub> Requirement B (Scope Abstraction - CRITICAL): If the rule lists specific entity
names (e.g., "among Entity A, Entity B"), DO NOT transcribe these names.
- You must interpret this list as "the specific group of {entity_type}s that
qualified through the previous step".
- Phrasing: Instead of listing names, use phrases like "among the remaining
eligible candidates," "within this refined selection," or "relative to the
qualifying peer group."
IV. AGENCY IDENTITY
Invent a plausible governing authority based on the context
(e.g., "Board of Agricultural Standards").
```

## B. Examples of Symbolic Reasoning Traces

To illustrate the diversity of reasoning structures in DOCHOP, we provide additional examples of the sampled symbolic reasoning traces in this appendix. Each trace is represented as a logical tree, where nodes correspond to parameterized constraints instantiated from rule templates, and edges propagate the valid entity set through successive filtering steps. In addition to simple linear chains, the generation process produces multi-branch compositions with Boolean operators (AND/OR), resulting in varied multi-hop reasoning patterns across documents. These traces serve as the underlying blueprint that is later verbalized into document narratives and grounded through the semantic reference label.

Figure B.1 shows representative reasoning traces with different depths and branching structures.

## C. Atomic Constraint Rule Templates

Atomic Constraint Templates. To construct diverse multi-hop reasoning traces, we define a library of atomic constraint templates (“atoms”) that operate over different chart structures. Each atom specifies a local, parameterized condition grounded in a single chart, such as threshold checks, within-entity comparisons, temporal trends, or extremal selection. During trace generation, these templates are instantiated by sampling concrete timestamps, metrics, groups, operators, and numeric thresholds from the chart schema, yielding fully specified symbolic constraints.

Importantly, we distinguish between non-unique atoms and unique atoms. Non-unique constraints may be satisfied by multiple entities simultaneously (e.g., “Revenue ≥ 70”), enabling gradual filtering across hops. Unique constraints instead enforce a deterministic extremal selection (e.g., “the entity with the highest value”), ensuring that the reasoning trace can resolve to a well-defined target set. Tables C.1–C.3 summarize all atomic templates used in DocHop.

## D. Question Templates

Given a synthesized document instance, we construct the final evaluation question by instantiating a task-specific template. Each template defines a canonical chart question-answering format (e.g., retrieval, counting, aggregation, ranking, hypothetical updates, or fact checking), while leaving key slots—such as the target metric, timestamp, aggregation operator, or additional condition—to be filled from the instance metadata.

![](images/1d00e446a50723f1eaf7fa5036a1952ef3ea46366a0b6b892cb974912c601ae5.jpg)

Figure B.1. Examples of sampled symbolic reasoning traces. Each trace forms a logical tree of instantiated constraints, ranging from linear multi-hop chains to branched compositions with AND/OR operators.
<table><tr><td>Template</td><td>Constraint Meaning</td><td>Example Instantiation</td></tr><tr><td>timestamp_threshold</td><td>Value at a specific timestamp exceeds/falls below a threshold</td><td>&quot;In 2023, Sales ≥ 70.&quot;</td></tr><tr><td>timestamp_vs_across_entity_agg</td><td>Value at a timestamp compared against an aggregate across entities</td><td>“In 2022, A&#x27;s Revenue ≥ the mean Revenue of all entities.&quot;</td></tr><tr><td>timestamp_vs_within_entity_time_agg</td><td>Value at a timestamp compared against an aggregate over time for the same entity</td><td>&quot;In 2021, Profit ≤ A&#x27;s average Profit across all years.&quot;</td></tr><tr><td>two_point_change_threshold</td><td>Absolute change between two timestamps exceeds/falls below a threshold</td><td>“From 2020 to 2022, Growth increases by at least 15.&quot;</td></tr><tr><td>two_point_percentage_change</td><td>Percentage change between two timestamps satisfies a condition</td><td>“From 2019 to 2023, Output rises by more than 20%.&quot;</td></tr><tr><td>window_monotone</td><td>Values over a consecutive time window are monotonic</td><td>“From 2021–2024, Engagement is strictly increasing.&quot;</td></tr><tr><td>window_slope_strict_monotone</td><td>Slope trend across a window follows strict directional consistency</td><td>“The rate of increase accelerates each year from 2020–2023.&quot;</td></tr><tr><td>timestamp_extreme (unique)</td><td>Entity achieves the maximum/minimum value at a timestamp</td><td>&quot;In 2024, A has the highest Satisfaction.&quot;</td></tr><tr><td>window_sum_extreme (unique)</td><td>Entity has the largest/smallest cumulative value over a time window</td><td>“Across 2020–2022, B has the lowest total Cost.&quot;</td></tr></table>

Table C.1. Atomic constraint templates for line charts. These rules impose temporal thresholds, trends, and extremal conditions over time-series values.

Crucially, all templates are grounded on the semantic reference label introduced at the end of the document narrative, rather than directly naming entities from the charts. As a result, questions do not explicitly specify which rows in the chart tables should be queried. Instead, models must first resolve the labeled entity set implied by the narrative constraints, and then retrieve or aggregate the corresponding numerical evidence from the charts.

Across the six task categories, we design a total of 41 diverse templates. These templates cover both direct queries over the labeled entities (e.g., retrieving a metric value or counting the number of qualifying entities) and more compositional variants that introduce additional chart-grounded conditions or counterfactual modifications. This template-based construction enables systematic control over question forms while ensuring that all answers can be computed exactly from the underlying chart tables. The complete template list, grouped by task type, is provided in the following pages.

## E. Evaluation Details

We evaluate model performance using normalized answer accuracy. To reduce formatting variance, we append a shared instruction block to every query, requiring models to return the final answer in a dedicated sentence of the form ‘‘The answer is: <...>’’ and to omit units, currency symbols, and percentage signs. When the question specifies a rounding scheme, models are instructed to follow it strictly; otherwise, integer answers are expected as integers, and non-integer answers are rounded to two decimal places.

DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents
<table><tr><td>Template</td><td>Constraint Meaning</td><td>Example Instantiation</td></tr><tr><td>metric_threshold</td><td>A metric value exceeds/falls below a threshold</td><td>&quot;In 2024, Revenue &gt; 60.&quot;</td></tr><tr><td>row_metric_extreme</td><td>One metric is the largest/smallest within an entity&#x27;s metric profile</td><td>&quot;For A, Profit is the highest among all metrics.&quot;</td></tr><tr><td>row_metric_vs_row_stat</td><td>Metric compared against entity-level statistics (mean/sum)</td><td>“For B, Cost ≤ B&#x27;s average across metrics.&quot;</td></tr><tr><td>row_share_threshold</td><td>Metric contributes at least a fraction of the entity&#x27;s total</td><td>&quot;For C, Marketing accounts for at least 40% of total spend.&quot;</td></tr><tr><td>within_entity_metric_compare</td><td>Direct comparison between two metrics of the same entity</td><td>“For A, Revenue ≥ Expense.&quot;</td></tr><tr><td>metric_difference_threshold</td><td>Difference between two metrics exceeds/falls below a threshold</td><td>&quot;For D, Profit minus Cost ≥ 20.&quot;</td></tr><tr><td>single_metric_cross_entity_extreme (unique)</td><td>Entity is extremal under one metric across all entities</td><td>&quot;A has the lowest Transaction Fees.&quot;</td></tr><tr><td>row_total_cross_entity_extreme (unique)</td><td>Entity has extremal total across all metrics</td><td>&quot;B has the highest overall expenditure.&quot;</td></tr></table>

Table C.2. Atomic constraint templates for multi-metric bar charts. These rules capture within-entity metric comparisons and cross-entity extrema.

<table><tr><td>Template</td><td>Constraint Meaning</td><td>Example Instantiation</td></tr><tr><td>within_entity-group_compare</td><td>Compare values across groups for the same entity</td><td>&quot;For A, North ≥ South.&quot;</td></tr><tr><td>within_entity-group_share_threshold</td><td>A group contributes at least a fraction of entity total</td><td>“For B, Online accounts for at least 50% of total.&quot;</td></tr><tr><td>same_group_value_threshold</td><td>Threshold applied to a specific group value across entities</td><td>“&quot;In Retail, Sales ≥ 70.&quot;</td></tr><tr><td>same_group_share_threshold</td><td>Group share constraint across entities</td><td>“For all entities, APAC contributes less than 30%.&quot;</td></tr><tr><td>same_group_vs_row_stat</td><td>Group value compared against entity-level statistics</td><td>“For C, Europe &lt; C&#x27;s average across regions.&quot;</td></tr><tr><td>group_difference_threshold</td><td>Difference between two groups exceeds/falls below threshold</td><td>“For D, East minus West ≥ 15.&quot;</td></tr><tr><td>row-group_extreme</td><td>Within an entity, one group is maximal/minimal</td><td>“For A, North is the largest region.&quot;</td></tr><tr><td>same_group_value_extreme (unique)</td><td>Entity achieves extremum in a specific group</td><td>&quot;B has the highest value in APAC.&quot;</td></tr><tr><td>same_group_share_extreme (unique)</td><td>Entity has extremal proportional share in a group</td><td>“C has the largest APAC share.&quot;</td></tr><tr><td>row_groups_sum_extreme (unique)</td><td>Entity has extremal total summed across groups</td><td>“&quot;D has the lowest total output across all regions.&quot;</td></tr></table>

Table C.3. Atomic constraint templates for group bar charts. These rules impose within-entity group structure and cross-entity extremal selection.

Prompting Setup (System Instruction). All models are evaluated in a single-image VQA setting where the input is a document-page image containing both narrative text and one or more charts. We use a unified system-style instruction (SHARED INSTRUCTION) appended to the question turn, which enforces (i) a single final answer span, (ii) a canonica numeric formatting policy, and (iii) a fixed output marker (The answer is:) to facilitate robust parsing. The instruction begins with a short preamble describing the input format (e.g., “read the text within the image and analyze the chart(s)”), followed by the shared answer-format constraint described above.

Answer Parsing and Normalization. Given a model response, we extract the predicted answer by searching for the final occurrence of the pattern the answer is: (case-insensitive) and taking the subsequent span; if the marker is missing, we fall back to the last non-empty line. We further normalize extracted strings by stripping leading answer: prefixes, whitespace, and lightweight formatting tokens (e.g., <sub>\*</sub>, backticks).

Correctness Criteria. We apply the same two-step procedure to every question, regardless of task type. We first check for a case-insensitive exact match between the normalized prediction and ground truth (e.g., for Yes/No fact-checking answers or exact entity names). If this fails, we fall back to a numeric comparison: we extract the last numeric token from both prediction and ground truth (allowing an optional minus sign and decimals, and ignoring comma separators) and mark the prediction correct if the two values are close under math.isclose with relative tolerance 0.01 and absolute tolerance 10<sup>−3</sup>.

The image provided is a document page containing both text and chart(s).   
Please read the text within the image and analyze the chart(s) to answer   
the question.   
Your final answer should be a single pure value:   
- If the question asks for an entity, use its complete name exactly as shown   
in the chart

DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents
<table><tr><td rowspan=1 colspan=1>Template ID</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=6>Question Pattern</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Task 1: Valu</td><td rowspan=1 colspan=3>e Retrieval</td></tr><tr><td rowspan=1 colspan=1>1.1</td><td rowspan=1 colspan=1>Value Query</td><td rowspan=1 colspan=3>What is the {metric_desc } of the {en</td><td rowspan=1 colspan=3>tity_type } receiving the {semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>Entity Query</td><td rowspan=1 colspan=3>Which {entity_type} receives the</td><td rowspan=1 colspan=3>{semantic_label}?</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>Task 2: Č</td><td rowspan=1 colspan=3>ounting</td></tr><tr><td rowspan=1 colspan=1>2.1</td><td rowspan=1 colspan=1>Count All</td><td rowspan=1 colspan=6>How many {entity_type }(s) receive the {semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>2.2</td><td rowspan=1 colspan=1>Count w/ Threshold</td><td rowspan=1 colspan=6>Among the {entity_type }(s) receiving the {semantic_label }, how many have {metric_desc} {comparison}{threshold}?</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>Task 3: Numeric Reasoning</td></tr><tr><td rowspan=1 colspan=1>3.1</td><td rowspan=1 colspan=1>Sum</td><td rowspan=1 colspan=6>What is the total {metric_desc } of the {entity_type }(s) receiving the {semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>3.2</td><td rowspan=1 colspan=1>Range</td><td rowspan=1 colspan=6>What is the difference between the highest and lowest {metric_desc} among the {entity_type }(s) receiving the{semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>Max</td><td rowspan=1 colspan=6>What is the highest {metric_desc } among the {entity_type }(s) receiving the {semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>3.4</td><td rowspan=1 colspan=1>Min</td><td rowspan=1 colspan=6>What is the lowest {metric_desc}among the {entity_type }(s) receiving the{semantic_label}?</td></tr><tr><td rowspan=1 colspan=1>3.5</td><td rowspan=1 colspan=1>Average</td><td rowspan=1 colspan=6>What is the average{metric_desc} of the {entity_type}}(s) receiving the {semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>3.6</td><td rowspan=1 colspan=1>Time Range Sum</td><td rowspan=1 colspan=6>What is the total{metric} from {start_time} to {end_time} of all the {entity_type}s receiving the{semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>3.7</td><td rowspan=1 colspan=1>Time Diff</td><td rowspan=1 colspan=6>What is the difference in the total {metric } between {time1 } and {time2} of all the {entity_type }s receiving the{semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>3.8</td><td rowspan=1 colspan=1>Multi-Metric Sum</td><td rowspan=1 colspan=6>What is the total of {metric1 } and {metric2 } in {time } for all the {entity _type }s receiving the {semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>3.9</td><td rowspan=1 colspan=1>Multi-Metric Diff</td><td rowspan=1 colspan=6>What is the difference between the total {metric1 } and total {metric2} in {time} of all the {entity_type}sreceiving the {semantic_label }?</td></tr><tr><td rowspan=1 colspan=1>3.10</td><td rowspan=1 colspan=1>Group Sum</td><td rowspan=1 colspan=6>What is the combined total {metric } across {group1 } and {group2 } in {time} for all the {entity _type }s receivingthe {semantic_label}?</td></tr><tr><td rowspan=1 colspan=1>3.11</td><td rowspan=1 colspan=1>Group Diff</td><td rowspan=1 colspan=6>What is the difference in total {metric } between {group1 } and {group2} in {time} of all the {entity_type}sreceiving the {semantic_label}?</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Task 4: Ranking</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>4.1</td><td rowspan=1 colspan=1>Argmax</td><td rowspan=1 colspan=1>Among the</td><td rowspan=1 colspan=1>{entity_type }</td><td rowspan=1 colspan=2>(s) receiving the</td><td rowspan=1 colspan=1>semantic_label</td><td rowspan=1 colspan=1>}, which one has the highest{metric_desc }?</td></tr><tr><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=1>Argmin</td><td rowspan=1 colspan=1>Among the</td><td rowspan=1 colspan=1>entity_type</td><td rowspan=1 colspan=2>}(s) receiving the</td><td rowspan=1 colspan=1>semantic_label</td><td rowspan=1 colspan=1>}, which one has the lowest{metric_desc }?</td></tr><tr><td rowspan=1 colspan=1>4.3</td><td rowspan=1 colspan=1>K-th Best</td><td rowspan=1 colspan=1>Among the</td><td rowspan=1 colspan=1>entity_type</td><td rowspan=1 colspan=2>}(s) receiving the</td><td rowspan=1 colspan=1>semantic_label</td><td rowspan=1 colspan=1>}, which one has the {k-th }highest {metric_desc}?</td></tr></table>

Table D.1. Question Templates for Chart-based QA

<table><tr><td rowspan=1 colspan=1>Template ID</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=2>Question Pattern</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Task 5: Hypothetical</td></tr><tr><td rowspan=1 colspan=1>5.1</td><td rowspan=1 colspan=1>Global Change</td><td rowspan=1 colspan=2>If the total {metric_desc } of all the {entity_type}s receiving the {semantic_label} {increased/decreased } by{pct} %, what would be the new value?</td></tr><tr><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=1>Extremum Change</td><td rowspan=1 colspan=2>If the {entity_type } with the {highest/lowest } {metric_desc } (among those receiving the {semantic_label }) hadtheir value {increased/decreased } by {pct} %, what would be the new total {metric_desc} for all recipients?</td></tr><tr><td rowspan=1 colspan=1>5.3</td><td rowspan=1 colspan=1>Add Constraint</td><td rowspan=1 colspan=2>Suppose that {entity_type }s that would otherwise qualify for the {semantic_label} are now additionally requiredto satisfy the following constraint: {new_condition }. {downstream_question}</td></tr><tr><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=1>Modify Threshold</td><td rowspan=1 colspan=2>Suppose that, for {entity_type }s seeking the {semantic_label}, the threshold defined under {constraint_name}was adjusted from {old_threshold} to {new_threshold}. {downstream_question }</td></tr><tr><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>Flip Logic</td><td rowspan=1 colspan=1>Suppose that, for {entity _type }s seeking the {semantic_label }, the rule in {section_namesatisfying all listed conditions simultaneously rather than qualifying through only one</td><td rowspan=1 colspan=1>} was modified to require. {downstream_question}</td></tr><tr><td rowspan=1 colspan=1>5.6</td><td rowspan=1 colspan=1>Swap Dimension</td><td rowspan=1 colspan=2>Suppose that, for {entity-type }s seeking the {semantic_label}, the evaluation metric in {section_name} waschanged from &#x27;{old_metric}&#x27; to &#x27;{new_metric}&#x27;. {downstream_question}</td></tr></table>

Table D.2. Question Templates for Chart-based QA (Continued - Hypothetical)

- If the answer is a number: output integers as integers, decimals rounded to   
two places   
(unless the question specifies otherwise)   
- Otherwise, follow the question’s instructions for the expected format   
Do not include units, currency symbols, or percentage signs in your final answer.   
At the end of your response, format the final answer in a separate sentence   
like this:   
The answer is: <your\_final\_answer>

## F. Model Configuration

We report the evaluation configurations of all models considered in our experiments. For proprietary APIs, we specify the exact model versions used at evaluation time. For open-sourced baselines, we list the corresponding HuggingFace checkpoints.

For Qwen-2.5-VL and Qwen3-VL models, we follow the recommended visual tokenization settings, using a minimum pixel resolution of 1280 × 28 × 28 and a maximum resolution of 16384 × 28 × 28.

DocHop: Benchmarking Out-of-domain Multi-hop Reasoning in Information-Dense Documents
<table><tr><td rowspan=1 colspan=1>Template ID</td><td rowspan=1 colspan=2>Subtype               Question Pattern</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Task 6: Fact Checking (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.1 - Entity Verif</td><td rowspan=1 colspan=2>ication</td></tr><tr><td rowspan=1 colspan=1>6.1.1</td><td rowspan=1 colspan=1>Entity Membership</td><td rowspan=1 colspan=1>Is {entity } listed as a recipient of the {semantic_label }? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.1.2</td><td rowspan=1 colspan=1>Value Match</td><td rowspan=1 colspan=1>Did the {entity_type } receiving the {semantic_label } record a {metric_desc } of {value}? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.2 - Count Verifi</td><td rowspan=1 colspan=1>cation</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>6.2.1</td><td rowspan=1 colspan=1>Exact Count</td><td rowspan=1 colspan=1>Are there exactly {count} {entity_type }(s) receiving the {semantic_label}? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3 - Numeric Ver</td><td rowspan=1 colspan=1>ification</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>6.3.1</td><td rowspan=1 colspan=1>Sum Check</td><td rowspan=1 colspan=1>Is the total {metric_desc } of the {entity _type }(s) receiving the {semantic_label } equal to { value }? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.2</td><td rowspan=1 colspan=1>Range Check</td><td rowspan=1 colspan=1>Is the difference between the highest and lowest {metric_desc} among the {entity_type}(s) receiving the{semantic_label} equal to {value}? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.3</td><td rowspan=1 colspan=1>Max Check</td><td rowspan=1 colspan=1>Is the highest {metric_desc } among the {entity_type }(s) receiving the {semantic_label} equal to {value}?(Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.4</td><td rowspan=1 colspan=1>Min Check</td><td rowspan=1 colspan=1>Is the lowest {metric_desc} among the {entity_type }(s) receiving the {semantic_label} equal to {value}?(Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.5</td><td rowspan=1 colspan=1>Average Check</td><td rowspan=1 colspan=1>Is the average {metric_desc} of the {entity_type }(s) receiving the {semantic_label} equal to {value }? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.6</td><td rowspan=1 colspan=1>Time Range Sum Check</td><td rowspan=1 colspan=1>Is the total {metric } from { start_time } to {end_time } for all the {entity_type }s receiving the {semantic_label}equal to {value }? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.7</td><td rowspan=1 colspan=1>Time Diff Check</td><td rowspan=1 colspan=1>Is the difference in the total {metric} between {time1 } and {time2} for all the {entity_type } s receiving the{semantic_label} equal to {value }? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.8</td><td rowspan=1 colspan=1>Multi-Metric Sum Check</td><td rowspan=1 colspan=1>Is the total of {metric1 } and {metric2} in {time } for all the {entity_type }s receiving the {semantic_label } equalto {value}? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.9</td><td rowspan=1 colspan=1>Multi-Metric Diff Check</td><td rowspan=1 colspan=1>Is the difference between the total {metric1 } and total {metric2} in {time} of all the {entity _type} s receivingthe {semantic_label} equal to {value }? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.10</td><td rowspan=1 colspan=1>Group Sum Check</td><td rowspan=1 colspan=1>Is the combined total {metric } across {group1 } and {group2} in {time} for all the {entity_type }s receiving the{semantic_label} equal to {value }? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.3.11</td><td rowspan=1 colspan=1>Group Diff Check</td><td rowspan=1 colspan=1>Is the difference in total {metric} between {group1 } and {group2} in {time } of all the {entity_type }s receivingthe {semantic_label} equal to {value }? (Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.4 - Ranking Ver</td><td rowspan=1 colspan=1>ification</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>6.4.1</td><td rowspan=1 colspan=1>Argmax Check</td><td rowspan=1 colspan=1>Does {entity} have the highest {metric_desc} among the {entity_type }(s) receiving the {semantic_label}?(Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.4.2</td><td rowspan=1 colspan=1>Argmin Check</td><td rowspan=1 colspan=1>Does {entity } have the lowest {metric_desc } among the {entity_type}(s) receiving the {semantic_label}?(Yes/No)</td></tr><tr><td rowspan=1 colspan=1>6.4.3</td><td rowspan=1 colspan=1>K-th Best Check</td><td rowspan=1 colspan=1>Does {entity } have the {k-th } highest {metric_desc } among the {entity _type }(s) receiving the {semantic_label }?(Yes/No)</td></tr></table>

Table D.3. Question Templates for Chart-based QA (Continued - Fact Checking)

<table><tr><td rowspan=1 colspan=1>Model Name</td><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Hugging Face Checkpoint / API</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Reasoning</td><td rowspan=1 colspan=1>Models</td></tr><tr><td rowspan=1 colspan=1>GPT-5.2-Reasoning</td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>gpt-5.2-2025-12-11</td></tr><tr><td rowspan=1 colspan=1>GPT-5-mini-Reasoning</td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>gpt-5-mini-2025-08-07</td></tr><tr><td rowspan=1 colspan=1>Gemini-2.5-Pro-Reasoning</td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>gemini-2.5-pro</td></tr><tr><td rowspan=1 colspan=1>Gemini-2.5-Flash-Reasoning</td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>gemini-2.5-flash</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>Models</td></tr><tr><td rowspan=1 colspan=1>GPT-5.2</td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>gpt-5.2-2025-12-11</td></tr><tr><td rowspan=1 colspan=1>Gemini-2.5-Flash</td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>gemini-2.5-flash</td></tr><tr><td rowspan=1 colspan=1>Claude-4.5-Haiku</td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>claude-haiku-4-5-20251001</td></tr><tr><td rowspan=1 colspan=1>Claude-4.5-Sonnet</td><td rowspan=1 colspan=1>Proprietary</td><td rowspan=1 colspan=1>claude-sonnet-4-5-20250929</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Open-Source M</td><td rowspan=1 colspan=1>odels</td></tr><tr><td rowspan=1 colspan=1>LLaVA-Next-LLaMA3-8B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>lmms-lab/llama3-llava-next-8b</td></tr><tr><td rowspan=1 colspan=1>IDEFICS2-8B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>HuggingFaceM4/idefics2-8b</td></tr><tr><td rowspan=1 colspan=1>IDEFICS3-LLaMA3-8B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>HuggingFaceM4/Idefics3-8B-Llama3</td></tr><tr><td rowspan=1 colspan=1>Qwen-2.5-VL-7B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>Qwen/Qwen2.5-VL-7B-Instruct</td></tr><tr><td rowspan=1 colspan=1>Qwen-2.5-VL-32B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>Qwen/Qwen2.5-VL-32B-Instruct</td></tr><tr><td rowspan=1 colspan=1>Qwen-2.5-VL-72B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>Qwen/Qwen2.5-VL-72B-Instruct</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>Qwen/Qwen3-VL-8B-Instruct</td></tr><tr><td rowspan=1 colspan=1>Molmo-7B-O-0924</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>allenai/Molmo-7B-O-0924</td></tr><tr><td rowspan=1 colspan=1>Molmo-7B-D-0924</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>allenai/Molmo-7B-D-0924</td></tr><tr><td rowspan=1 colspan=1>InternVL-3.5-8B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>OpenGVLab/InternVL3_5-8B</td></tr><tr><td rowspan=1 colspan=1>InternVL-3.5-30B-A3B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>OpenGVLab/InternVL3_5-30B-A3B</td></tr><tr><td rowspan=1 colspan=1>InternVL-3.5-38B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>OpenGVLab/InternVL3_5-38B</td></tr><tr><td rowspan=1 colspan=1>Ovis1.6-Gemma2-9B</td><td rowspan=1 colspan=1>Open-Source</td><td rowspan=1 colspan=1>AIDC-AI/Ovis1.6-Gemma2-9B</td></tr></table>

Table F.1. Model List and Checkpoints

Since DocHop is reasoning-intensive and often requires multi-step aggregation, we set the maximum generation length for all models to 2<sup>14</sup> tokens whenever supported, avoiding premature truncation due to insufficient output budgets.