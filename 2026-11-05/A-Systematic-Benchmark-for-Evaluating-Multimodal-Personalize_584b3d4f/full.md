# A Systematic Benchmark for Evaluating Multimodal Personalized Reasoning in Vision-Language Models

Dohyun Kim<sup>1,∗</sup> Junyong Lhim<sup>2,∗</sup> Kwanyong Park<sup>1,†</sup>

<sup>1</sup>University of Seoul, Republic of Korea <sup>2</sup>Sungkyunkwan University, Republic of Korea

<sup>1</sup>{kdoh04034, kwanyong.park}@uos.ac.kr <sup>2</sup>lhimjy1105@skku.edu

## Abstract

People often make decisions by holistically reasoning over heterogeneous forms of personal information. In this paper, we study the relatively underexplored capability of multimodal personalized reasoning in multimodal large language models (MLLMs). To this end, we introduce MPRBench, a novel and comprehensive benchmark specif ically designed to evaluate personalized reasoning across a wide range of tasks and real-world challenges. MPRBench consists of 12 sub-tasks and additionally supports interpretable analysis of representative error cases. Through extensive experiments, we find that current personalized MLLMs still struggle with personalized reasoning due to several key challenges, including inaccurate recognition of personalized concepts and sensitivity to irrelevant personal information. We hope that MPRBench will stimulate further research on personalized reasoning in MLLMs. Code and data are available at https://github.com/ kkdh15/MPRBench

## 1. Introduction

In daily life, people frequently interact with personal concepts such as friends, belongings, and pets, relying on their own memories. These memories are inherently multimodal and play a central role in human behavior and decisionmaking. For instance, when planning what to do, we often reason holistically by considering the visual appearance of close friends, their habits, past interactions, and the sentiments we expect from them. In other words, humans naturally integrate current context with heterogeneous forms of personal memory, including visual impressions, textual knowledge, past experiences, preferences, and social relationships.

Given this nature of human reasoning, an ideal personalized AI assistant should be able to reason holistically over both the current situation and diverse forms of personal memory. Achieving this goal requires substantial advances in multimodal reasoning over personal information, particularly across heterogeneous modalities such as vision and language, which remains relatively underexplored in current research.

![](images/3f2cae9e3329135be716df3e379a8949a283d6eeef6e83f0a000503fd533d37c.jpg)  
Figure 1. Conceptual Overview and Problem Setup for Multi modal Personalized Reasoning.

Although prior studies on personalized multimodal large language models (MLLMs) [15, 30] have made significant progress, they address this challenge only partially. Existing approaches mainly focus on relatively simple tasks, such as recognizing the presence of personal concepts from visual inputs [30] or reporting their basic status [3]. While current personalized models perform well on such benchmarks, these settings remain far from the complexity of real-world user requests. As a result, it is still unclear whether existing models can faithfully handle realistic scenarios and satisfy user needs, especially when deeper reasoning over multimodal personal information is required.

To bridge this gap, we introduce MPRBench, a novel and comprehensive benchmark for personalization, designed to evaluate the reasoning capabilities of MLLMs with multimodal personal information. In designing MPRBench, we consider emerging challenges in real-world personalized AI scenarios. In particular, we highlight a key challenge of multimodal personalized reasoning: the ability to reject irrelevant personal information during the reasoning process. Although multimodal AI systems may have access to a wide range of personal information in both visual and textual forms, not every user request requires all of that information. Thus, a personalized MLLM should continually determine whether each piece of personal information is relevant to the current reasoning process; otherwise, it should rely on its internal generic knowledge instead.

Based on this insight, we divide our benchmark into three groups—visual-specific, text-specific, and multimodal reasoning—according to which modality is ideally required to solve the problem. Specifically, MPRBench consists of five visual-specific reasoning tasks, three text-specific reasoning tasks, and four multimodal reasoning tasks, resulting in a total of 12 sub-tasks. These design choices allow our benchmark to more faithfully reflect real-world personalized scenarios and to systematically analyze both the performance and failure cases of current models.

On MPRBench, we evaluate a wide range of personalized MLLMs spanning four major categories of prior work: (1) zero-shot, prompting-based personalization [41], (2) methods based on learning specialized personalized tokens [3, 30], (3) retrieval-based models [12, 15] and (4) memory-augmentation-based methods [17, 31]. Through extensive and systematic experiments, we obtain several key findings. First, robust recognition of personal concepts remains a fundamental bottleneck for personalized reasoning. Second, current methods struggle to reject irrelevant personal information across diverse scenarios. Third, reasoning that requires sequential and repeated use of personalized information remains highly challenging for existing models. Together, these findings highlight clear technical barriers that must be addressed.

Our contributions are summarized as follows: (1) We introduce MPRBench, the first comprehensive and interpretable benchmark for evaluating multimodal personalized reasoning. (2) We provide a systematic analysis of current personalized MLLMs and identify key factors that limit their personalized reasoning capabilities.

## 2. Related Works

Multimodal Large Language Models. Inspired by the rapid progress of large language models (LLMs) [8, 43], extensive research efforts have been devoted to extending these advances to diverse modalities, including vision [25] and audio [13]. In particular, for the visual modality, mul timodal large language models (MLLMs) have substantially advanced classical vision-language tasks such as visual question answering [23, 40], image captioning [21, 39] and visual grounding [5, 33, 34, 46]. More recently, the research focus has shifted toward building strong visual reasoning capabilities [18], where models are expected not only to perceive visual content but also to reason over it using commonsense and contextual understanding. To support this trend, many comprehensive multimodal benchmarks [10, 24, 27, 47] have been proposed to cover a wide range of reasoning-oriented tasks. These benchmarks have played an important role in driving progress in visual reasoning and enabling the development of more capable MLLMs [6, 14, 28, 45].

Despite these advances, most prior work has focused on generic visual reasoning, in which reasoning is grounded primarily in common world knowledge. In contrast, our work investigates the relatively unexplored problem of personalized multimodal reasoning, where models must actively leverage multimodal personal information during the reasoning process.

Personalization in MLLMs. Early works such as MyVLM [2] and Yo’LLaVA [30] opened the research area of personalized MLLMs by focusing on recognizing personal concepts and understanding their states. However, such approaches are limited in scalability because they require iterative training for each concept. To overcome this limitation, subsequent studies have attempted to remove concept-wise training costs by introducing large-scale personalization tuning [15, 32, 35] or training-free retrievalbased methods [12, 41].

More recently, several works have broadened the scope of personal information beyond visual identity, incorporating additional attributes such as user preferences [19, 20] or long-term interaction histories [17, 31]. Nevertheless, their target tasks still largely follow a recognition-and-retrieval paradigm, where models first identify a personalized concept and then retrieve or report the associated personal information.

In contrast, we systematically investigate diverse personalized reasoning tasks, in which each concept is represented not only by visual references but also by rich textua attributes. This setting requires models to go beyond concept recognition and attribute retrieval: they must identify relevant personal information and reason over it together with the current visual context and general commonsense knowledge across diverse reasoning scenarios.

## 3. The MPRBenchmark

## 3.1. Overview of MPRBenchmark

Our goal is to assess the multimodal reasoning capabilities of personalized MLLMs in real-world scenarios. In particular, we aim to cover a wide range of reasoning tasks involving personal information and to provide insights for developing more robust personalized models. While existing benchmarks [2, 4, 12, 20, 30] primarily evaluate elementary tasks, such as identifying personal entities, recognizing their states in visual contexts, or retrieving their associated attributes, MPRBench is designed to challenge higher-level reasoning tasks in which multimodal personal information must be used in diverse ways.

Notions and Problem Setup. We define a user-specific entity as a personalized concept, represented by a set of reference image(s) and reference text. The reference image(s) serve as a visual profile that captures the entity’s appearance and functions as a visual identifier. The reference text provides structured personal information in a JSON-like textual format, including attributes such as name, preferences (for persons), ownership (for objects), and relevant history. Given this, the goal of each task is to generate an appropriate answer to a textual query that mimics a user request, by jointly reasoning over both the visual query representing the current circumstance and the known personalized concept. (See Fig. 1)

![](images/7e0daeebc3e3bd034ff42f8a0e2a36f25124ca9b652c77ab4a13e3334bcdc0f3.jpg)  
Figure 2. Illustrative Examples and Statistics of MPRBench.

In this setting, a personalized MLLM must effectively manage personalized concepts composed of reference images and text, and must seamlessly and selectively leverage the relevant information needed to answer the given query.

## 3.2. Key Features of MPRBenchmark.

Broad coverage of personalized reasoning tasks. MPRBench is designed to enable holistic evaluation of personalized reasoning with multimodal personal information. Accordingly, it includes not only tasks that require joint reasoning over both visual and textual personal information, but also tasks for which only one modality is sufficient. Based on this taxonomy, we formulate realistic reasoning-oriented sub-tasks for each group. The resulting benchmark contains 12 sub-tasks in total, supporting evaluation across diverse scenarios and allowing fine-grained analysis of model weaknesses.

Complex visual queries. To better reflect realistic reasoning scenarios, we intentionally construct image queries with complex scenes in which personalized concepts appear alongside other entities within cluttered backgrounds. This design differs substantially from existing benchmarks, which often rely on personalized-object-centric images [30] (e.g., images containing only the target concept) or attribute-centric illustrations that omit the personalized concept itself [19] (e.g., an image showing only a preferred concert). Although this choice significantly increases the burden of data construction, it enables more challenging high-level reasoning tasks and better represents the complexity of real-world environments.

Interpretable benchmark design. We design MPRBench to support interpretable analysis of personalized models. Specifically, we formulate all tasks as multiple-choice question answering problems, where each option is designed to reflect a representative failure case. This design makes it easier to identify major failure modes and provides actionable insights into technical directions for future research.

## 3.3. Task Definition

MPRBench comprises 12 sub-tasks, each designed to capture a distinct challenge in multimodal personalized reasoning. Below, we describe each task group and highlight the key reasoning demands they impose. Representative examples for selected sub-tasks are provided in Fig. 2, the complete set of examples is provided in the supplementary.

Vision-specific reasoning. This group contains tasks for which only visual information—reference image(s) and the visual query—is necessary for successful reasoning. In these tasks, a personalized model must first accurately identify the personalized concept from the reference image(s), and then reason about the current visual scene in relation to that concept. We include personalized variants of several classical vision-oriented reasoning tasks [10, 27, 29, 48]: (1) predictive reasoning, which requires forecasting the concept’s likely near-future action or situation; (2) spatial reasoning, which requires understanding spatial relations involving the concept; (3) numerical reasoning, which requires reasoning about counts or quantities related to the concept; and (4) visual commonsense reasoning, which requires inferring the rationale behind the visual state of the concept based on commonsense knowledge. In addition, we introduce (5) visual appearance change reasoning, a novel task motivated by realistic user needs such as determining whether a close friend has changed their hairstyle. This task requires explicit comparison between stored visual memory from the reference image(s) and the concept’s current appearance in the query image.

![](images/4304e728b53e831855a01ca463be95133854a53793b1150335ed1319c86e418f.jpg)  
Figure 3. Overview of the Data Construction Pipeline for MPRBench.

Text-specific reasoning. In contrast, this group contains tasks for which textual information alone is sufficient, and visual information is unnecessary. These tasks require reasoning over textual descriptions of personalized concepts together with textual user queries. We include (1) predictive reasoning, which involves predicting plausible outcomes from personal preferences, habits, or prior experiences under a given condition, and (2) numerical reasoning, which involves arithmetic or logical reasoning over numerical personal attributes such as birthdays. We further introduce (3) personalized multi-hop reasoning, a novel task that requires sequentially combining multiple pieces of textual personal information. Inspired by generic multi-hop reasoning [16], this task is designed to test whether models can repeatedly retrieve and integrate relevant personal attributes across multiple reasoning steps.

Multimodal reasoning. This group consists of tasks in which both visual and textual personal information are jointly required. As such, it directly evaluates whether a model can align, retrieve, and integrate heterogeneous personal information across modalities. We include (1) predictive reasoning, which requires forecasting likely actions based on personal preferences or habits under a given visual context; (2) sentimental reasoning, which requires inferring the cause or rationale of visually expressed emotion in light of personal events or experiences; and (3) numerical reasoning, which requires counting or aggregating textual attributes associated with concepts present in the visual scene. Finally, we propose (4) multimodal history reasoning, a novel task in which the model must infer when a visual query was captured by matching visual evidence in the current image with historical information about the concept described in text.

We note that both the reference image(s) and reference text are provided to personalized MLLMs in order to reflect real-world conditions, where users often do not explicitly specify which information should be considered.

## 3.4. Dataset Construction Process

We design a semi-automatic dataset construction pipeline to reduce human labor while maintaining overall data quality. We focus here on the high-level construction process (See Fig. 3); more detailed descriptions are provided in the supplementary material.

## 3.4.1. Raw Data Collection

Concept collection. We begin by manually collecting reference images for personalized concepts. These images are sourced either from existing open datasets or from CC BY 4.0 licensed images collected from the internet. For person concepts, we use existing personalization-related datasets such as Yo’LLaVA [30], MC-LLaVA [3], MMPB [19] and CustomConcept101 [22], as well as images of sports players from SoccerAgent [37] and publicly available CC BY 4.0 licensed images of celebrities. For object concepts, we repurpose ConConChi [38] and RPC [44], which were originally constructed for object retrieval and Retail Product Checkout tasks, respectively. To faithfully represent the visual appearance of each concept, we crop the original images to remove complex surrounding contexts. In addition, since some prior personalization methods [30] require multiple reference images, we retain only concepts for which at least five reference images are available.

Complex visual query collection. We also manually collect complex visual query images, with particular emphasis on scene complexity. Specifically, we prioritize images in which the target concept appears together with visually similar concepts (i.e., distractors) and under cluttered backgrounds. These conditions provide valuable opportunities to assess how concept identification difficulty affects downstream reasoning performance. Based on these criteria, we source person-related visual queries from VCR [48] and SoccerAgent [37], and object-related queries from Con-ConChi and RPC. We make substantial efforts to ensure high quality during this stage. For example, we manually inspected approximately 4,000 images from VCR and selected only 120 images that satisfied our quality standards for visual queries.

## 3.4.2. Textual Reference Construction

Constructing high-quality textual references for each concept is a particularly important step, as it strongly affects the quality of subsequent data generation stages. We carefully design a set of core textual attributes: 16 attributes for person concepts, including name, age, favorites, habits, and life events, and 8 attributes for object concepts, including owner, acquisition process, and history. The full attribute list is provided in the appendix. These textual references should be simultaneously plausible, consistent with the visual appearance of the concept, diverse across the dataset, and well correlated with the visual query so as to support the construction of high-level reasoning tasks.

To achieve this, we structure the textual reference construction process into four steps. We describe the process for person concepts as an example.

Step 1. Visually-grounded Attributes. We prompt Gemini-2.5-Pro [11] to infer visually grounded attributes, such as age and gender, that are strongly correlated with the concept’s appearance in the reference images.

Step 2. Diversity-oriented Attributes. To promote diversity, we randomly assign other basic attributes, such as name and birthday, from pre-defined rules when they are not strongly constrained by visual evidence.

Step 3. Context-aware Attributes. We prompt Gemini 2.5-Pro to generate attributes such as preferences and personal events, which later serve as major sources for question generation, conditioned on both the previously fixed attributes and the associated visual query.

Step 4. Group-wise Attributes. We construct interconcept relationships by randomly grouping four people, assigning shared attributes, and generating recurring events within the group, such as having dinner together every Tuesday.

Through this structured pipeline, which combines model-assisted generation with rule-based construction, we aim to preserve attribute naturalness while improving the realism and diversity of the dataset.

![](images/b5160b1cdaa8da98e8c30ed0e80a24a00762d040f6c5db85fad67c43f82607f1.jpg)  
Figure 4. Diagnostic Analysis of Multimodal Reasoning Paths

The construction process for other concept types follows the same overall philosophy. In each case, we account for concept-specific characteristics in both attribute design and detailed generation procedures.

## 3.4.3. Query Construction

After collecting the raw data and constructing the textual references, we use this information to build test instances, each consisting of a visual query, a textual query (i.e., question and answer choices), and the corresponding answer, for evaluating personalized models. During preliminary trials, we found that there is no single efficient and accurate strategy for constructing queries across all sub-tasks and concept types. A major reason is the large variation in data conditions, such as whether useful metadata (e.g., captions or bounding boxes) is available and whether sufficiently complex visual queries are abundant for a given sub-task. We therefore develop three types of query construction methods and choose among them depending on the available data conditions.

Type 1: Template-based. For sub-tasks in which both useful metadata (e.g., captions or bounding boxes) and complex visual queries are available, we use templated questions and answer choices directly from the metadata. For example, this strategy is used for visual-specific tasks that leverage detailed captions in ConConChi and for numerical tasks that use bounding box annotations in RPC.

Type 2: Human-labeled. When complex visual queries are available but supporting metadata is not, we manually construct the textual questions and answer choices. Annotators first identify the core reasoning logic of each problem by considering the nature of the sub-task, the visual query, and the textual reference. Based on this reasoning structure, they then craft appropriate questions and answer choices according to predefined guidelines. Most sub-tasks involving person concepts fall into this category.

Type 3: Generative-model-based. In most cases, the above two methods allow us to construct high-quality queries. However, scaling up the dataset and increasing task difficulty remain challenging due to the scarcity of suitable complex visual queries. To address this limitation, we employ a state-of-the-art image generation model. Specifically, we first generate the core reasoning logic of the problem, followed by the textual question and answer choices, using Gemini-2.5-Pro. We then use Gemini 2.5 Flash Image [11] to generate visual queries that match the generated textual questions. This strategy removes the dependence on pre-existing visual queries and allows us to better exploit the flexibility of generative models.

<table><tr><td></td><td colspan="5">Vision-specific Reasoning</td><td colspan="5">Text-specific Reasoning</td><td colspan="5">Multimodal</td><td></td></tr><tr><td></td><td>predict</td><td>spatial</td><td>numeric</td><td>commsense</td><td>change</td><td>V-Avg</td><td>predict</td><td>numeric</td><td>multi-hop</td><td>T-Avg</td><td>predict</td><td>sentiment</td><td>numeric</td><td>history</td><td>M-Avg</td><td>Avg</td></tr><tr><td>Random Choice</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td><td>25.0</td></tr><tr><td>Proprietary+Prompting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40 [1]+P</td><td>91.6</td><td>64.3</td><td>63.5</td><td>71.6</td><td>65.0</td><td>70.6</td><td>97.5</td><td>70.1</td><td>40.3</td><td>62.5</td><td>88.1</td><td>79.4</td><td>50.0</td><td>49.4</td><td>73.7</td><td>69.4</td></tr><tr><td>GPT-5-mini [42]+P</td><td>90.3</td><td>57.4</td><td>83.0</td><td>66.7</td><td>91.9</td><td>79.9</td><td>95.8</td><td>90.6</td><td>48.4</td><td>70.7</td><td>77.8</td><td>69.0</td><td>58.0</td><td>92.4</td><td>74.4</td><td>75.9</td></tr><tr><td>GPT-5 [42]+P</td><td>88.4</td><td>66.1</td><td>81.2</td><td>72.6</td><td>87.8</td><td>80.3</td><td>98.3</td><td>95.8</td><td>77.4</td><td>87.2</td><td>88.1</td><td>79.4</td><td>64.0</td><td>94.9</td><td>84.1</td><td>83.1</td></tr><tr><td>Gemini-2.5 Pro [11]+P</td><td>85.8</td><td>62.6</td><td>77.7</td><td>63.7</td><td>94.3</td><td>77.9</td><td>98.3</td><td>95.8</td><td>83.0</td><td>90.0</td><td>84.1</td><td>83.3</td><td>59.5</td><td>96.2</td><td>85.1</td><td>83.1</td></tr><tr><td>Open-sourced+Prompting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA [26]+P</td><td>69.7</td><td>46.1</td><td>50.2</td><td>51.9</td><td>90.2</td><td>60.2</td><td>61.7</td><td>41.7</td><td>18.6</td><td>35.5</td><td>34.9</td><td>49.9</td><td>20.0</td><td>39.2</td><td>40.3</td><td>48.2</td></tr><tr><td>Qwen2.5-VL-7B [7]+P</td><td>73.5</td><td>47.8</td><td>46.9</td><td>48.0</td><td>64.2</td><td>55.4</td><td>90.8</td><td>52.1</td><td>17.7</td><td>45.3</td><td>40.5</td><td>66.7</td><td>36.0</td><td>50.6</td><td>49.4</td><td>51.0</td></tr><tr><td>Token-learning based</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Yo&#x27;LLaVA [30]</td><td>47.7</td><td>28.7</td><td>33.9</td><td>47.1</td><td>42.3</td><td>39.0</td><td>43.3</td><td>30.2</td><td>14.0</td><td>25.6</td><td>30.1</td><td>34.9</td><td>30.0</td><td>25.3</td><td>31.9</td><td>33.5</td></tr><tr><td>MC-LLaVA [3]</td><td>56.1</td><td>39.9</td><td>43.5</td><td>39.2</td><td>65.0</td><td>48.4</td><td>57.5</td><td>35.4</td><td>13.6</td><td>30.4</td><td>39.7</td><td>50.0</td><td>34.0</td><td>25.3</td><td>39.6</td><td>41.3</td></tr><tr><td>Retrieval based</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RAP [15]</td><td>50.3</td><td>22.6</td><td>36.9</td><td>43.1</td><td>47.9</td><td>40.1</td><td>53.3</td><td>28.1</td><td>4.52</td><td>23.1</td><td>38.9</td><td>47.6</td><td>22.0</td><td>10.1</td><td>32.3</td><td>33.4</td></tr><tr><td>R2P [12]</td><td>64.5</td><td>39.9</td><td>43.9</td><td>50.9</td><td>83.7</td><td>54.8</td><td>75.8</td><td>40.6</td><td>16.7</td><td>38.2</td><td>56.4</td><td>53.9</td><td>54.0</td><td>34.2</td><td>52.6</td><td>49.8</td></tr><tr><td>Memory-augmentation-based</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TAME [17]</td><td>72.9</td><td>52.2</td><td>50.6</td><td>51.9</td><td>77.2</td><td>59.8</td><td>89.2</td><td>61.5</td><td>9.05</td><td>42.6</td><td>60.3</td><td>61.1</td><td>46.0</td><td>31.7</td><td>50.2</td><td>52.6</td></tr><tr><td>PersonaVLM [31]</td><td>65.2</td><td>47.8</td><td>43.9</td><td>52.9</td><td>78.1</td><td>55.5</td><td>90.8</td><td>62.5</td><td>14.9</td><td>46.2</td><td>62.7</td><td>57.9</td><td>50.0</td><td>34.2</td><td>49.1</td><td>51.3</td></tr></table>

Table 1. Evaluation on proposed MPRBench.

Although we employ three different query construction strategies, we maintain a common design philosophy of building an interpretable benchmark. To this end, regardless of the construction method, we design the incorrect answer choices to reflect the most likely failure cases of personalized models. This allows MPRBench to support comprehensive analysis of model behavior by aggregating statistics over different error types. In particular, the answer options are designed to capture representative failure types including: (1) mis-identification, where the model reasons as if another object or person were the target concept; (2) missing personal information, where the model falls back on generic commonsense instead of using the relevant personal information; and (3) incorrect use of personal information, where the model relies on personal information that is irrelevant to the given problem. As shown in Fig. 4, each incorrect option is grounded in a distinct region of the query image or distinct entry of the personal database. This design allows the selected option to indicate which step of the reasoning process has failed.

## 3.4.4. Human Verification

To ensure the quality of MPRBench, we make substantial manual efforts to review the constructed data and filter out examples that are incorrect or ambiguous. This verification process includes checking the quality of intermediate outputs from raw data collection, textual reference construction, and query construction. Additional details on the human verification procedures used for quality control (e.g., annotator agreement across multiple aspects) are provided in the supplementary.

## 4. Personalized Reasoning Evaluation

## 4.1. Experimental Setup

Baseline models. We benchmark a wide range of personalization methods for MLLMs. These include: (1) zeroshot, prompting-based personalization, which injects personal information as additional context through prompting; (2) personalized special-token learning methods, which learn and use special tokens to represent the visual identity of personalized concepts, including Yo’LLaVA [30] and MC-LLaVA [3]; (3) retrieval-based models, which employ similarity-based retrieval mechanisms for personalized concept recognition, such as RAP [15] and R2P [12]; and (4) memory-augmentation-based models, including TAME [17] and PersonaVLM [31]. Note that memoryaugmentation-based models are originally designed for long-term interactions across multiple turns. For evaluation on MPRBench, we repurpose them for single-turn evaluation by providing all concept-related information available in the benchmark as the model’s single-turn memory input. Evaluation metrics. We report accuracy, defined as the proportion of samples for which a personalized model selects the correct answer, for each sub-task. To obtain representative metrics for each reasoning group (i.e., visionspecific, text-specific, and multimodal reasoning), we compute the accuracy across the sub-tasks within each group, weighted by the number of QA items in each sub-task. We denote these group-level metrics as V-Avg, T-Avg, and M-Avg, respectively. The final overall score, Avg., is defined as the weighted average of these three group-level scores.

![](images/7dae0dc450462dd86511a9beb374d9d3b3f51a09e4fc3bbff544ef56536802ac.jpg)  
Figure 5. Distractor Injection Points in the Input Structure.

## 4.2. Explored Various Distractors

To evaluate whether personalized MLLMs can selectively reason over target-relevant information, we explore various types of distractors introduced at different points in the model input, as illustrated in Fig. 5.

Visually similar distractor. As described in Sec. 3.2, we intentionally construct MPRBench with query images containing concepts visually similar to the target concept. If a model fails to localize the target accurately, it may instead ground its reasoning in a visually similar distractor and produce an answer based on the wrong entity.

Cross-modal distractor. For vision-specific and textspecific tasks, where only one modality is required to answer correctly, we examine the effect of misleading information introduced through the other modality. Such information is superficially related to the query but irrelevant to the correct answer. We further investigate this distractor in the following section.

Inter-concept distractor. Beyond the two distractors above, we expose models to reference images and personal databases of non-target concepts, either directly in the input for prompting-based models or in the concept pool for retrieval-based models. Most models, including proprietary ones, struggle to identify and reason over the target concept while ignoring these additional concepts, leading to performance degradation. We primarily analyze this distractor in

![](images/19d8ab8ef2589d4546307ab24db3f7505d3db093e63a3e37ec10cb7ead1787da.jpg)  
Figure 6. Analysis on Failure Cases.  
the supplementary material.

## 4.3. Evaluation Results and Analysis

Based on MPRBench, we conduct extensive evaluations to assess the personalized reasoning capabilities of personalized MLLMs and to analyze the potential causes behind the observed performance trends.

1) Can current personalized MLLMs perform personalized reasoning effectively? To answer this question, we benchmark baseline models on MPRBench (See 1). Overall, all baselines remain far from perfect, as reflected in the final Avg. score. This contrasts sharply with the near-perfect performance reported on previous personalized recognition benchmarks such as Yo’LLaVA, suggesting that current personalized MLLMs still struggle substantially with personalized reasoning tasks.

More interestingly, methods specifically designed for personalization—both token-learning-based and retrievalbased approaches—often underperform a simple prompting baseline (i.e., LLaVA+Prompt). This observation contrasts with the strong performance claims made in prior works. In addition, RAP, which involves large-scale personalization tuning of the backbone MLLM, shows the weakest performance among the compared methods. We conjecture that tuning on elementary personalized tasks, such as recognition-oriented VQA, may improve personalized concept recognition to some extent, but such improvements do not generalize well to reasoning-intensive settings. One possible reason is that this type of tuning may not preserve, or may even weaken, the intrinsic reasoning capability of the original backbone model.

Memory-augmentation-based methods show only marginal gains over their Qwen2.5-VL-7B backbone with simple prompting. Although their tailored memory construction and retrieval pipelines improve retrieval over long-term histories, these gains do not directly translate to the reasoning required by MPRBench. In MPRBench, retrieved information must be further integrated with a cluttered visual context or commonsense knowledge through multi-step inference, going beyond simple retrieval.

2) What are the common failure modes? Thanks to the interpretable option design of MPRBench, we can analyze model failures by summarizing the distribution of incorrectly selected answer choices. As shown in Fig. 6, the most common failure mode is mis-identification. This indicates that most models still struggle to accurately localize or identify the target concept among distractors during the reasoning process. We argue that this challenge has been somewhat overlooked in previous benchmarks, which typically evaluate personalized recognition in isolation and often rely on object-centric images. In realistic settings, however, personalized recognition itself becomes a major bottleneck for successful reasoning.

To further investigate the impact of personalized recognition on final performance, we conduct a controlled study comparing two settings: ‘reasoning only (location given) versus ‘localization + reasoning’. For this experiment, we use the visual commonsense reasoning sub-task, since the VCR [48] dataset provides precise bounding boxes for concepts. We manually select test samples for which the correct answer could change if mis-identification occurs. We then compare the performance of: (1) the LLaVA+Prompt baseline, where both localization and reasoning are required, and (2) a setting in which the target bounding box is provided as a visual prompt [9], so that only reasoning is required. As shown below, the large performance gap between these two settings indicates that personalized recognition remains a major challenge for reasoning-time performance.

<table><tr><td>reasoning only</td><td>localization + reasoning</td></tr><tr><td>71.43%</td><td>51.43%</td></tr></table>

3) Are personalized MLLMs robust to cross-modal distractors? For the vision-specific and text-specific reasoning groups, it is important for models to reject irrelevant information from the other modality rather than relying on it during reasoning. This becomes particularly challenging when cross-modal personal information is irrelevant but still appears highly plausible or similar to the target concept. To investigate this issue, we conduct a controlled study in which we add hard negative textual personal information and measure its impact on model performance. We use the prediction task as the testbed for this analysis. Specifically, we introduce additional events using keywords taken from incorrect answer choices, while ensuring that the correct answer remains unchanged. As shown in Fig 7, most personalized MLLMs suffer noticeable performance drops when these distractors are added. This result suggests that current personalized MLLMs are not sufficiently robust to irrelevant textual distractors and often fail to properly ignore misleading cross-modal personal information.

4) Which classes of methods excel at understanding finegrained visual details of concepts? Understanding and leveraging fine-grained visual details of a personalized concept (e.g., hairstyle) is crucial in real-world scenarios. To examine which classes of methods are more effective in this setting, we focus on the visual appearance change reasoning task (“Change” in Table 1), which requires detailed comparison between the current visual context and the concept’s stored visual memory. In this task, token-learning-based methods generally perform worse than other three groups, all of which directly access the reference image itself. We suspect that compressing rich visual information into only a few learnable tokens inevitably causes information loss, leading to weaker performance on fine-grained visual understanding. Note that RAP is somewhat exception, which we suspect may stem from its tuning strategy weakening the model’s intrinsic reasoning capability.

![](images/e2564e26f3cdee9821573b8f6e7aa7162cd9466b7fd9e106faf45d919a5bc179.jpg)  
Figure 7. Analysis on Cross-modal Distractor.

5) What unique challenges arise in personalized multihop reasoning? This task introduces the unique challenge of repeatedly leveraging personal information across multiple reasoning steps. Among all sub-tasks, personalized multi-hop reasoning is the most challenging for opensource personalized MLLMs, with most models performing at or even below random chance. Moreover, the performance gap between open-source models and frontier proprietary models is the largest across all sub-tasks, highlighting a critical limitation and the urgent need for more effective methods to support reasoning across multiple retrieval and inference steps.

## 5. Conclusion

In this paper, we introduce MPRBench, a novel benchmark for multimodal personalized reasoning. MPRBench consists of 12 sub-tasks spanning vision-specific, text-specific, and multimodal reasoning, providing broad coverage of personalized reasoning scenarios. In addition, we adopt an interpretable answer design that enables analysis of representative failure modes. Using MPRBench, we find that current personalized MLLMs still face substantial challenges in personalized reasoning, particularly in accurate personalized recognition and robustness to cross-modal distractors during the reasoning process. We hope that our benchmark and analyses will provide useful insights for the development of future personalized MLLMs.

## Acknowledgement

This research was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT)(RS-2026-25499205, 80%). Also, the authors acknowledge the Urban Big Data and AI Institute of the University of Seoul supercomputing resources (http://ubai.uos.ac.kr) made available for conducting the research reported in this paper (20%).

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 6, 16

[2] Yuval Alaluf, Elad Richardson, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Myvlm: Personalizing vlms for user-specific queries. In European Conference on Computer Vision, pages 73–91. Springer, 2024. 2

[3] Ruichuan An, Sihan Yang, Ming Lu, Renrui Zhang, Kai Zeng, Yulin Luo, Jiajun Cao, Hao Liang, Ying Chen, Qi She, et al. Mc-llava: Multi-concept personalized vision-language model. arXiv preprint arXiv:2411.11706, 2024. 1, 2, 4, 6, 13, 16

[4] Ruichuan An, Sihan Yang, Renrui Zhang, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, Bocheng Zou, et al. Unictokens: Boosting personalized understanding and generation via unified concept tokens. Advances in Neural Information Processing Systems, 38:144638–144664, 2026. 2

[5] Sojung An, Kwanyong Park, Yong Jae Lee, and Donghyun Kim. Talk in pieces, see in whole: Disentangled and hierarchical representation learning in language-based object detection. In Proceedings ofthe 2026 Conference on Empirical Methods in Natural Language Processing, pages 40850– 40874, 2026. 2

[6] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 2, 18

[7] Shuai Bai, Ke qin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. ArXiv, abs/2502.13923, 2025. 6, 16

[8] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling opensource language models with longtermism. arXiv preprint arXiv:2401.02954, 2024. 2

[9] Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Vipllava: Making large multimodal models understand arbitrary

visual prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12914– 12923, 2024. 8

[10] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Informa tion Processing Systems, 37:27056–27087, 2024. 2, 3, 18

[11] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 5, 6, 13, 16

[12] Deepayan Das, Davide Talon, Yiming Wang, Massimiliano Mancini, and Elisa Ricci. Training-free personalization via retrieval and reasoning on fingerprints. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9683–9692, 2025. 2, 6, 16

[13] Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang. Pengi: An audio language model for audio tasks. Advances in Neural Information Processing Systems, 36:18090–18108, 2023. 2

[14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2

[15] Haoran Hao, Jiaming Han, Changsheng Li, Yu-Feng Li, and Xiangyu Yue. Rap: Retrieval-augmented personalization for multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14538–14548, 2025. 1, 2, 6, 16

[16] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Lin guistics, pages 6609–6625, 2020. 4

[17] Rongpei Hong, Jian Lang, Ting Zhong, Yong Wang, and Fan Zhou. Tameing long contexts in personalization: Towards training-free and state-aware mllm personalized assistant. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 452– 463, 2026. 2, 6, 16

[18] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017. 2

[19] Jaeik Kim, Woojin Kim, Woohyeon Park, and Jaeyoung Do. Mmpb: It’s time for multi-modal personalization. Advances in Neural Information Processing Systems, 38, 2026. 2, 3, 4, 13

[20] Minsung Kim. Pervl-bench: Benchmarking multimodal personalization for large vision–language models. In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 6696–6704. IEEE, 2026. 2

[21] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017. 2

[22] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1931–1941, 2023. 4, 13

[23] Youngwan Lee, Kangsan Kim, Kwanyong Park, Ilchae Jung, Soojin Jang, Seanie Lee, Yong-Ju Lee, and Sung Ju Hwang. Holisafe: Holistic safety benchmarking and modeling for vision-language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5989–5998, 2026. 2

[24] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 2

[25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 2, 16

[26] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024. 6, 19

[27] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 2, 3, 18

[28] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visualrft: Visual reinforcement fine-tuning. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 2034–2044. IEEE, 2025. 2

[29] Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Jieneng Chen, Celso de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6924–6934, 2025. 3

[30] Thao Nguyen, Haotian Liu, Yuheng Li, Mu Cai, Utkarsh Ojha, and Yong Jae Lee. Yo’llava: Your personalized language and vision assistant. Advances in Neural Information Processing Systems, 37:40913–40951, 2024. 1, 2, 3, 4, 6, 13, 16

[31] Chang Nie, Chaoyou Fu, Yifan Zhang, Haihua Yang, and Caifeng Shan. Personavlm: Long-term personalized multimodal llms. arXiv preprint arXiv:2604.13074, 2026. 2, 6, 16

[32] Yeongtak Oh, Dohyun Chung, Juhyeon Shin, Sangha Park, Johan Barthelemy, Jisoo Mok, and Sungroh Yoon. Repic: Reinforced post-training for personalizing multi-modal language models. Advances in Neural Information Processing Systems, 38:28238–28275, 2026. 2

[33] Kwanyong Park, Kuniaki Saito, and Donghyun Kim. Weak to-strong compositional learning from generative models for language-based object detection. In European Conference on Computer Vision, pages 1–19. Springer, 2024. 2

[34] Kwanyong Park, Sojung An, Yong Jae Lee, and Donghyun Kim. Learning compositionality from multifaceted synthetic data for language-based object detection. International Journal ofComputer Vision, 133(11):7873–7896, 2025. 2

[35] Renjie Pi, Jianshu Zhang, Tianyang Han, Jipeng Zhang, Ru Pan, and Tong Zhang. Personalized visual instruction tuning. In International Conference on Learning Representa tions, pages 43247–43266, 2025. 2

[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 17

[37] Jiayuan Rao, Zifeng Li, Haoning Wu, Ya Zhang, Yanfeng Wang, and Weidi Xie. Multi-agent system for comprehensive soccer understanding. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 3654–3663, 2025. 4, 5, 13

[38] Andrea Rosasco, Stefano Berti, Giulia Pasquale, Damiano Malafronte, Shogo Sato, Hiroyuki Segawa, Tetsugo Inada, and Lorenzo Natale. Concon-chi: Concept-context chimera benchmark for personalized vision-language tasks. In Pro ceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, pages 22239–22248, 2024. 4, 12, 13

[39] Kuniaki Saito, Donghyun Kim, Kwanyong Park, Atsushi Hashimoto, and Yoshitaka Ushiku. Captionsmiths: Flexibly controlling language pattern in image captioning. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 19872–19881. IEEE, 2025. 2

[40] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146–162. Springer, 2022. 2

[41] Soroush Seifi, Vaggelis Dorovatas, Daniel Olmeda Reino, and Rahaf Aljundi. Personalization toolkit: Training free personalization of large vision language models. arXiv preprint arXiv:2502.02452, 2025. 2

[42] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 6, 12

[43] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2

[44] Xiu-Shen Wei, Quan Cui, Lei Yang, Peng Wang, and Lingqiao Liu. Rpc: A large-scale retail product checkout dataset. arXiv preprint arXiv:1901.07249, 2019. 4, 12, 13

[45] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan. Llava-cot: Let vision language

models reason step-by-step. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2087– 2098, 2025. 2

[46] Heng Yin, Yuqiang Ren, Ke Yan, Shouhong Ding, and Yongtao Hao. Rod-mllm: Towards more reliable object detection in multimodal large language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14358–14368. IEEE, 2025. 2

[47] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024. 2, 18

[48] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6720–6731, 2019. 3, 5, 8

## Overview

This supplementary material reports the dataset statistics, the full construction procedure and the additional experimental results of MPRBench.

• Sec. A provides more details on MPRBench itself.

– Sec. A.1 reports detailed benchmark statistics.

– Sec. A.2 provides the full textual reference construction process and examples of personal database.

– Sec. A.3 describes the query construction pipeline.

– Sec. A.4 reports the human verification procedure and agreement statistics, with the annotation interface used.

– Sec. A.5 provides the definition and representative examples of all 12 sub-tasks.

• Sec. B provides additional experimental details and analyses.

– Sec. B.1 describes how each baseline method was adapted to MPRBench.

– Sec. B.2 compares model performance on naturally collected versus synthetically generated visual queries.

– Sec. B.3 provides analysis of various distractor types.

– Sec. B.4 provides further experiments with Qwen3-VL family models.

## Limitations

Given the complexity and diversity of multimodal personalized reasoning, MPRBench may not cover all possible scenarios. Nevertheless, we make a substantial effort to include a wide range of sub-tasks and to introduce novel tasks in each modality-specific group. Exploring even more diverse and complex scenarios for multimodal personalized reasoning remains an important direction for future work.

In addition, our primary goal in this work is to benchmark and analyze the current multimodal reasoning capabilities of personalized MLLMs. As a result, we do not extensively investigate methodological approaches for improving model performance. However, we believe that the findings and analyses provided in this work can serve as a useful foundation for future advances in personalized reasoning.

## Ethics Statement

Because our benchmark includes person concepts and images of people, we strictly follow the licenses and terms of use associated with all source images. Most images are drawn from open datasets, and the remaining images are collected under CC BY 4.0 licenses. In particular, we carefully verify whether each person image permits derivative use or editing. If editing is not allowed, we do not use that image as a basis for generated or modified content (e.g., in Type 3 query construction).

For textual references, namely the personal information associated with images, we intentionally generate synthetic attributes rather than relying on factual personal information such as real names or actual biographies. In this way, we aim to mitigate privacy concerns.

In preparing this manuscript, we used ChatGPT [42] to improve readability by checking grammar and polishing language. All core content, ideas, and claims were originally written by the authors, who take full responsibility for the contents of the paper.

## A. More Details on MPRBench

## A.1. Benchmark Statistics

In this section, we provide a detailed numerical analysis of MPRBench to highlight its scale and the diversity of its evaluation scenarios.

Diversity across Personalized Concepts: As summarized in Table. 2, our benchmark incorporates a wide array of 119 unique personalized concepts, comprising 69 distinct individuals and 50 varied objects. For person concepts, we draw from four existing personalization datasets, from SoccerAgent, and from CC BY 4.0 licensed web images; for object concepts, we repurpose ConconChi [38] and RPC [44], which were originally built for object retrieval and retail product checkout. Fig. 13 provides an overview of selected personalized concepts used in our benchmark. This extensive collection of concepts ensures that the models are evaluated on their ability to generalize across diverse visual identities.

Broad Coverage of Reasoning Tasks: Furthermore, Table. 3 details the distribution of 1,667 question-answer pairs across 12 sub-tasks. By categorizing these to visual-specific (766 QAs), text-specific (437 QAs), and Multimodal (464 QAs), MPRBench covers a broad spectrum of reasoning requirements. This comprehensive task design ensures a rigorous assessment of a model’s multifacted capabilities.

## A.2. Details on Textual Reference Construction

In this section, we provide the full list of attributes used in the textual references, together with their definitions. Tables 4 and 5 summarize this information for person and object concepts, respectively. While the construction process for person concepts is detailed in Section 3.4.2, this section provides the process for object concepts. In contrast to person concepts which incorporate biological attributes (e.g., age, gender), the construction of object concepts focuses on ownership mapping and history, as detailed below. Step 1. Identification Attributes. Since object concepts lack biological attributes, they are instead initialized with unique identifiers to ensure distinctiveness. Each object is assigned a specific name from a predefined unique name list (e.g., for Conconchi [38] or RPC [44]) and an <owner> placeholder. This placeholder serves as a functional link, allowing the object to be dynamically associated with a specific person concept in later stages.

<table><tr><td>Category</td><td>Source</td><td># Concepts</td></tr><tr><td rowspan="6">Person</td><td>Yo&#x27;LLaVA [30]</td><td>7</td></tr><tr><td>MC-LLaVA [3]</td><td>9</td></tr><tr><td>MMPB [19]</td><td>6</td></tr><tr><td>CustomConcept101 [22]</td><td>3</td></tr><tr><td>SoccerAgent [37]</td><td>22</td></tr><tr><td>Web</td><td>22</td></tr><tr><td></td><td>Subtotal</td><td>69</td></tr><tr><td rowspan="3">Object</td><td>ConConChi [38]</td><td>20</td></tr><tr><td>RPC [44]</td><td>30</td></tr><tr><td>Subtotal</td><td>50</td></tr><tr><td>Total</td><td></td><td>119</td></tr></table>

Table 2. Distribution of personalized concepts in MPRBench across concept categories and source datasets.

<table><tr><td>Task Category</td><td>Sub-task</td><td># Questions</td></tr><tr><td>Visual-Specific</td><td>Prediction</td><td>155</td></tr><tr><td></td><td>Spatial</td><td>115</td></tr><tr><td></td><td>Numerical</td><td>271</td></tr><tr><td></td><td>Commonsense</td><td>102</td></tr><tr><td></td><td>Visual Appearance Change</td><td>123</td></tr><tr><td>Text-Specific</td><td>Sub-total (Visual)</td><td>766</td></tr><tr><td rowspan="4"></td><td>Prediction</td><td>120</td></tr><tr><td>Numerical</td><td>96</td></tr><tr><td>Multi-Hop</td><td>221</td></tr><tr><td>Sub-total (Text)</td><td>437</td></tr><tr><td rowspan="4">Multimodal</td><td>Prediction</td><td>209</td></tr><tr><td>Sentiment</td><td>126</td></tr><tr><td>Numerical</td><td>50</td></tr><tr><td>history</td><td>79</td></tr><tr><td></td><td>Sub-total (Multimodal)</td><td>464</td></tr><tr><td colspan="3">Total</td></tr></table>

Table 3. Detailed statistics of MPRBench across different task categories.

Step 2. Diversity-oriented Attributes. To promote diversity, we randomly assign temporal and ownership attributes, such as acquisition date, manufacturing date, and the actual owner. These attributes are generated based on pre-defined rules to maintain logical consistency (e.g., manufacturing date preceding acquisition date).

Step 3. Context-aware Attributes. We prompt Gemini-2.5-Pro [11] to extract or generate event-based attributes by analyzing the query images. For Conconchi concepts, the model generates a logical chain of information grounded in visual evidence. In contrast, for RPC, where query images serve for numerical tasks, these attributes are generated to maintain context-aware consistency. These attributes include external textual facts, such as specific dates and personal events, which are non-inferable from the image alone. In designing these attributes, we aim to account for the characteristics specific to each concept type. The full prompts used at model-assisted steps are provided in Figs. 19 to 21 for person concepts and Figs. 22 to 24 for object concept, respectively. We provide examples of the personal databases in Figs. 11 and 12, demonstrating the structured textual information that models must utilize alongside visual inputs to solve the reasoning tasks in MPRBench.

## A.3. Details on Query Construction

In this section, we provide the detailed mapping between sub-tasks and query construction types in the table 11 described in Sec. 3.4.3. This mapping is determined by the source of the underlying concept rather than by the sub-task alone. Object concepts from ConConChi and RPC carry captions and bounding box annotations, therefore admit template-based construction (Type 1). Person concepts lack such metadata, so their queries are manually constructed instead (Type 2). Type 3 is used across both concept types when suitable complex visual queries are scarce. Figs. 25 to 28 provide the prompts used for Type 3 construction, illustrated with the predictive reasoning sub-task of person concepts. We first design a scene blueprint that places a favorite, a non-favorite, and a neutral item in the same scene together with a distractor person, then convert this blueprint into a question and four answer choices, and finally generate the corresponding visual query while preserving the identity of the target concept. The remaining Type 3 subtasks follow the same four-stage procedure, and differ only in the blueprint rules of the first stage.

## A.4. Details on Human Verification

In this section, we describe the human verification process conducted to ensure the high quality and complexity of MPRBench. This verification covers both the generated textual references and the collected or generated visual queries. Fig. 10 illustrates the interfaces used for human verification. Validation of generated textual references. Since parts of the personal database are model-generated, we validate them along criteria appropriate to each attribute type, and report the results in Table. 6. For visually grounded attributes, we treat the reference image as ground truth. Gender predictions were consistent with both annotators in all inspected cases, and age predictions fell within a 10- year range of the annotator’s estimates in 96.0% of cases. For subjective attributes such as preferences, there is often no single correct answer, and many generated values are equally plausible. The relevant criterion is therefore internal consistency rather than uniqueness, and annotators manually cross-referenced the generated color, food, and fashion preferences to eliminate contradictions, such as an item being listed as both a favorite and a non-favorite for the same concept; the two annotators reached the same decision in 99.3% of cases. For event attributes, we require that each event neither contradicts other attributes of the same concept nor conflicts with the associated visual query, so that valid question-answer pairs can be constructed on top of it. Candidate events violating either requirement were discarded, with an agreement rate of 94.0%. Finally, for groupwise attributes, we inspected relational integrity to confirm that shared attributes such as residences and events were consistently mapped across all associated members.

<table><tr><td>Category</td><td>Attribute</td><td>Definitions</td></tr><tr><td rowspan="2">Visually-grounded</td><td>gender</td><td>Gender of Person Concept which is strongly correlated with the concept&#x27;s appearance in reference images.</td></tr><tr><td>height</td><td>Height of Person Concept which inferred from visual context and diversity-oriented rules.</td></tr><tr><td rowspan="3">Diversity-oriented</td><td>name</td><td>Name of Person Concept by using unique name list to avoid duplication.</td></tr><tr><td>birthday</td><td>Birthday of Person Concept randomly generated based on estimated age to ensure tem- poral diversity.</td></tr><tr><td>family</td><td>Family Information about Person Concept which includes Family Relation with its birthday each. Randomly generated based on Person Concept&#x27;s birthday and Repla- tionship.</td></tr><tr><td rowspan="6">Context-aware</td><td>personality</td><td>Personality of Person Concept generated based on visual queries and fixed attributes.</td></tr><tr><td>favorites/non-favorites</td><td>Favorite/non-favorite of Person Concept which includes favorite/non-favorite colors, foods, and fashions.</td></tr><tr><td>hobby</td><td>Hobby of Person Concept which includes Personal interests and activities generated based on visual queries.</td></tr><tr><td>habit</td><td>Habit of Person Concept which includes not only visual information but also textual information so that it facilitates question generation and effectively reflects complex factors such as real-world routines.</td></tr><tr><td>event</td><td>Event of Person Concept includes personal events prevent in the visual query with the plausible behind story which is generated. Similar to habit, it facilitates question gen- eration and effectively reflects complex real-world factors such as date, emotion, and action.</td></tr><tr><td>nickname</td><td>Nickname of Person Concept which can be used instead of their name when referring to them implicitly.</td></tr><tr><td rowspan="5">Group-wise</td><td>birthplace</td><td>Birthplace of Person Concept which is shared with group members. Since it is con- sistent across group members and is unrelated to bias based on visual information - randomly assigned by city list, it can act as a hard negative element.</td></tr><tr><td>occupation</td><td>Occupation of Person Concept which is randomly assigned by list.</td></tr><tr><td>residence</td><td>Residence of Person Concept which is randomly assigned by list.</td></tr><tr><td>group_event</td><td>Event between Person Concept and group members which is partitioned among mem- bers and the events can be connected. For example, They meet every Monday - Every</td></tr><tr><td></td><td>Monday they have dinner together. This is neccessary when solving multi-hop ques- tion. The format of events are same but each elements(e.g., &quot;Monday&quot;, &quot;dinner&quot;) are randomly assigned by list.</td></tr></table>

Table 4. Full List and Definitions of Attributes in the Textual Reference for Person Concepts.

Filtering of visual queries. We further apply a systematic filtering process to visual queries, summarized in Table. 7.

<table><tr><td>Category</td><td>Attribute</td><td>Definitions</td></tr><tr><td>Identification</td><td>name</td><td>Name of Object Concept assigned by a unique name list to avoid duplication.</td></tr><tr><td rowspan="3">Diversity-oriented</td><td>acquisition_date</td><td>Acquisition date of Object Concept.</td></tr><tr><td>manufacturing_date</td><td>Manufacturing date of Object Concept which is always before acquisition date.</td></tr><tr><td>owner</td><td>Name of Object Concept&#x27;s Owner which assigned Person Concept Name for further ex- tensibility .</td></tr><tr><td rowspan="4">Context-aware</td><td>object_type</td><td>Type of Object Concept.</td></tr><tr><td>acquisition_process history</td><td>Acquisition process of Object Concept. History of Object Concept which has similar composition with Person Concept&#x27;s event</td></tr><tr><td></td><td>attribute. Contrastingly history can includes interaction history with its owner(does not serve as &quot;Person Concept&quot; but &quot;owner&quot;). To generate multimodal problems of high diffi- culty and ensure that the length of the text is similar to that of the Person Concept&#x27;s Textual Reference, a distractor history—similar to but distinct from the history extracted from the</td></tr><tr><td>storage</td><td>visual query—is generated, and the history is created as a pair Storage place of Object Concept.</td></tr></table>

Table 5. Full List and Definitions of Attributes in the Textual Reference for Object Concepts.

<table><tr><td>Attribute</td><td>Validation criterion</td><td>N</td><td>Agr.</td></tr><tr><td>Gender</td><td>match w/ annotators</td><td>50</td><td>100.0</td></tr><tr><td>Age</td><td>within ±10-year range</td><td>50</td><td>96.0</td></tr><tr><td>Preference</td><td>free of contradiction</td><td>150</td><td>99.3</td></tr><tr><td>Event</td><td>consistent &amp; grounded</td><td>150</td><td>94.0</td></tr></table>

Table 6. Validation of the generated textual references. N denotes the number of inspected items, and Agr. denotes the agreement rate (%) between two annotators.

<table><tr><td>Stage</td><td>Inspected</td><td>Retained</td><td>Rate</td></tr><tr><td>VCR query selection</td><td>4,000</td><td>120</td><td>3.0%</td></tr><tr><td>Type-3 image filtering</td><td>3,117</td><td>579</td><td>18.6%</td></tr></table>

Table 7. Retention rates of the visual query filtering stages in MPRBench.

As discussed in Sec. 3.4.1, we prioritize scene complexity in order to evaluate concept identification under challenging conditions. For human-centric tasks, we manually reviewed 4,000 candidate images from VCR and retained only 120 that satisfied our criteria of high scene density, presence of distractors, and cluttered backgrounds. For Type-3 queries, we found that current image generation models rarely satisfy all of our requirements at once, namely preserving the identity of the personalized concept, reflecting the intended number of target concepts and distractors, and maintaining the naturalness of the overall scene. We therefore generated three to five candidates per scenario and retained only the best one, resulting in 579 images out of 3,117 candidates.

Through this comprehensive manual effort, we ensured that MPRBench remains a reliable and logically sound instrument for evaluating multimodal personalized reasoning.

## A.5. Comprehensive Task Definitions

Sec. 3.3 introduces the 12 sub-tasks and the reasoning demands they impose. In this section, we specify the concrete form each sub-task takes in MPRBench, i.e., what the question asks and what evidence a correct answer requires. Figs. 14 to 16 illustrates the examples of each sub-task.

Visual-Specific Reasoning. The correct answer is determined entirely by the query image, and the personal database contributes only the identity of the target concept.

• Prediction Reasoning: The scene presents several courses of action available to the concept, and the question asks which one follows from its current pose and surroundings.

• Spatial Reasoning: Positional relations are asked from the concept’s own perspective rather than from the viewer’s, so that the concept must be localized before the relation can be resolved.

• Numerical Reasoning: The question asks how many entities in the scene share a given visual attribute with the concept, such as wearing the same uniform.

• Visual Commonsense Reasoning: The question asks why the concept is in behaving or appearing a certain way, and the options offer distinct commonsense explanations, only one of which is consistent with the visual context.

• Visual Appearance Change Reasoning: The question asks how the concept’s current appearance differs from the reference image(s).

Text-Specific Reasoning. Reference and query images are provided, but the correct answer is fully determined by the personal database.

• Prediction Reasoning: The question poses a hypothetical condition that is absent from the database, and the answer must be derived from recorded preferences, habits, or past events.

• Numerical Reasoning: The question requires arithmetic over dated attributes, such as computing an age difference at the time of a recorded event.

• Personalized Multi-Hop Reasoning: The answer is not stated in any single entry and must be assembled by chaining group-wise attributes across two or more concepts.

Multimodal Reasoning. The correct answer requires both sources; neither the image nor the database alone is sufficient.

• Prediction Reasoning: The question asks what the concept is likely to do next, and the answer must be inferred by combining multiple elements present in the query image with relevant attributes recorded in the database.

• Sentiment Reasoning: The question asks what emotion the concept is feeling, or why, and the answer must be inferred by jointly considering the query image and relevant attributes recorded in the database.

• Numerical Reasoning: The question asks how many objects in the scene satisfy a criterion stated in the database, so counting depends on a textual attribute.

• Multimodal history Reasoning: The question asks when the query image was taken, and the answer is the position in the concept’s recorded history whose described visual state matches the scene.

## B. More Details on Experiments

## B.1. Experimental Setup

In this section, we describe how we adapted various MLLM baselines to the MPRBench framework.

Prompting-based Personalization (Zero-shot): For general MLLMs like GPT-4o [1], Qwen2.5-VL [7] and LLaVAv1.5 [25], we provided personal information as structured context. Specifically, we converted the personal database into a JSON-formatted string and inserted it into the prompt. This ensures that the model can parse the attributes (e.g., identity, preferences, history) in a standardized manner.

Special-token Learning Methods (Yo’LLaVA [30], Mc-LLaVA [3]): These models utilize learnable tokens to represent specific identities. Both models were implemented using LLaVA-v1.5-13B as the base architecture. For evaluation on MPRBench, we followed the original training protocols described in their respective papers to learn the special tokens for each concept. During the evaluation phase, we provided the corresponding personal database via structured prompting as a dynamic context.

Retrieval-based Models (RAP [15], R2P [12]): We utilized the official source code and pre-trained checkpoints provided by the original authors. RAP is built on the LLaVA-v1.5-13B backbone, while R2P is based on the MiniCPM-o-2.6 architecture. To adapt them to our benchmark, we integrated the personalized history for each concept from our database into their respective concept pools, so that the model retrieves the concept by visual similarity and conditions on the profile stored with it.

Memory-augmentation-based models (TAME [17], PersonaVLM [31]): Both operate on a Qwen2.5-VL-7B backbone. Following the single-turn adaptation described in Sec. 4.1, we register each concept’s full personal database into the method’s native memory structures prior to evaluation. For TAME, the personal database is written directly into its double memory, with static attributes placed in static memory and temporal entries in dynamic memory under the native policy; the first reference image serves as the identification portrait. For PersonaVLM, the reference images form a single registration turn while the profile is supplied as core memory; the query follows after a gap exceeding the method’s session interval.

## B.2. Analysis on Natural and Synthetic Visual Queries

Since a portion of our visual queries is produced by an image generation model (Type 3 in Sec. 3.4.3), we examine whether these queries differ systematically in difficulty from naturally collected ones. To isolate the effect of the image source, we compute accuracy separately on the two sources, averaged over the sub-tasks that contain samples from both. As shown in Table. 8, the gap varies in both magnitude and sign across models, and no image source is consistently advantageous. This indicates that benchmark difficulty is not driven by artifacts of synthetic generation.

## B.3. Details on Distractors

Following the various distractor types introduced in Sec. 4.2, we provide a more detailed analysis of their effects on personalized reasoning.

## B.3.1. Visually similar distractor

We analyze the impact of visually similar concepts across all baseline models. Since our benchmark intentionally collects complex visual queries that include such distractors, it is not straightforward to obtain naturally paired splits with and without distractors. We therefore create a controlled subset by editing the query images of the predictive reasoning sub-task with Gemini-2.5-Flash-Image [11] to remove the distractor concepts while leaving the rest of the scene intact; edited images with visible artifacts are discarded through manual inspection. Table 9 shows that removing visually similar distractors improves accuracy across all evaluated methods, with gains ranging from 1.5 to 22.3 points. This result confirms that accurate visual identification of the target concept remains a prerequisite for reliable personalized reasoning, regardless of the personalization strategy employed.

## B.3.2. Cross-modal distractor

Visual-Specific Distractor. For visual-specific tasks, where the ground truth is embedded in the query image, we introduce distractors by inserting misleading textual entries into the personal database. We extract key actions or states from incorrect options and generate corresponding personal histories. As shown in Fig. 17, although this generated history contains keywords matching the distractor, it describes a scenario that is environmentally inconsistent with the current query image. This design test whether a model can prioritize visual evidence over misleading textual overlaps or if it is easily swayed by keyword-matching database entries that contradict the visual truth. As shown in Fig. 7, this type of distractor contributes to the incorrect use of personal information failure mode, indicating that models frequently rely on superficially matching but visually inconsistent textual cues rather than the query image itself.

Text-Specific Distractor.: In contrast, for text-specific tasks where the ground truth is strictly recorded in the database, we assess robustness by introducing deceptive visual cues in the query images. We categorized three visual distractors into three types, as illustrated in Fig. 18:

• Type 1: A query image where a different individual performs the same action described in the incorrect option.

• Type 2: A query image where the target subject is performing an action or associated with keywords found in the incorrect option.

• Type 3: A query image solely based on the incorrect option’s keywords.

As summarized in the performance comparison in Fig. 8, the effect of these visual distractors on text-specific tasks is inconsistent across models: performance increases for some models and decreases for others, with no clear pattern emerging across distractor types or model families. Since text-specific tasks are by design answerable from the database alone, this suggests that injecting distracting visual cues into the query image does not meaningfully affect model behavior on this task type.

## B.3.3. Inter-concept distractor

We analyze the impact of inter-concept distractors, i.e., exposing models to reference images and personal databases of non-target concepts, as described in Sec. 4.2. Since different personalization paradigms access personal information in fundamentally different ways, we adapt this distractor to each method category accordingly. For promptingbased models (e.g., GPT-5+P), we use a CLIP ViT-L/14 [36] image encoder to select the four concepts most visually similar to the target concept and include their reference images and personal databases alongside the target’s in the input conctext. For retrieval-based models, we instead expand the full concept pool with these additional concepts and let each method’s native retrieval mechanism select from this enlarged pool, rather than injecting the distractors directly into the prompt. For memory-augmentation-based models, we follow the same visually-similar selection procedure as the prompting-based setting, registring each of the four distractor concepts as a separate enrollment turn prior to evaluation, consistent with our single-turn adaptation described in Sec. 4.1. We exclude LLaVA-based models from this analysis, as their limited context length cannot accommodate the personal databases of multiple concepts simultaneously.

<table><tr><td>Method</td><td>Natural</td><td>Synthetic</td><td>∆</td></tr><tr><td>GPT-4o+P</td><td>70.4</td><td>79.3</td><td>+8.9</td></tr><tr><td>GPT-5-mini+P</td><td>67.9</td><td>77.2</td><td>+9.3</td></tr><tr><td>GPT-5+P</td><td>76.5</td><td>81.2</td><td>+4.7</td></tr><tr><td>Gemini-2.5 Pro+P</td><td>73.5</td><td>76.2</td><td>+2.7</td></tr><tr><td>LLaVA+P</td><td>46.9</td><td>50.7</td><td>+3.8</td></tr><tr><td>Qwen2.5-VL-7B+P</td><td>60.5</td><td>54.3</td><td>-6.2</td></tr><tr><td>Yo&#x27;LLaVA</td><td>38.3</td><td>37.0</td><td>-1.3</td></tr><tr><td>MC-LLaVA</td><td>53.1</td><td>47.3</td><td>-5.8</td></tr><tr><td>RAP</td><td>45.7</td><td>46.0</td><td>+0.3</td></tr><tr><td>R2P</td><td>61.7</td><td>53.5</td><td>-8.2</td></tr><tr><td>TAME</td><td>55.5</td><td>61.0</td><td>+5.5</td></tr><tr><td>PersonaVLM</td><td>60.5</td><td>59.0</td><td>-1.5</td></tr></table>

Table 8. Accuracy on natural versus synthetically generated visual queries, averaged over the sub-tasks that contain both image sources. ∆ denotes synthetic minus natural.

Table 10 shows that inter-concept distractors degrade performance for most evaluated methods, confirming that identifying and reasoning over the correct target concept becomes substantially more difficult when its personal information is embedded among that of several other concepts. Notably, this degradation is not limited to smaller or open-source models: proprietary models such as GPT-5+P and Gemini-2.5 Pro+P, despite supporting long context windows capable of accommodating multiple concepts’ personal information, still suffer non-trivial performance drops (3.2 and 4.6 points, respectively). This indicates that simply being able to fit more personal information into the context is not sufficient; models also be able to selectively attend to the target concept’s information while suppressing that of visually or contextually similar distractors.

<table><tr><td>Method</td><td>w/ distractors</td><td>w/o distractors</td><td>∆</td></tr><tr><td>GPT-4o+P</td><td>92.2</td><td>93.7</td><td>+1.5</td></tr><tr><td>GPT-5-mini+P</td><td>89.3</td><td>97.9</td><td>+8.6</td></tr><tr><td>GPT-5+P</td><td>88.4</td><td>95.8</td><td>+7.4</td></tr><tr><td>Gemini-2.5 Pro+P</td><td>83.5</td><td>96.8</td><td>+13.3</td></tr><tr><td>LLaVA+P</td><td>68.9</td><td>82.1</td><td>+13.2</td></tr><tr><td>Qwen2.5-VL-7B+P</td><td>75.7</td><td>90.5</td><td>+14.8</td></tr><tr><td>Yo&#x27;LLaVA</td><td>52.4</td><td>66.3</td><td>+13.9</td></tr><tr><td>MC-LLaVA</td><td>60.2</td><td>75.8</td><td>+15.6</td></tr><tr><td>RAP</td><td>57.3</td><td>65.3</td><td>+8</td></tr><tr><td>R2P</td><td>63.1</td><td>85.3</td><td>+22.2</td></tr><tr><td>TAME</td><td>69.9</td><td>83.2</td><td>+13.3</td></tr><tr><td>PersonaVLM</td><td>66.9</td><td>77.9</td><td>+11</td></tr></table>

Table 9. Effect of removing visually similar distractors from the query images of the predictive reasoning sub-task. ∆ denotes w/o distractors minus w distractors.  
![](images/327bb0052917f72900deb4387ad722775f5d5d2998fe1f1912577f488a1e9d27.jpg)

Figure 8. Analysis on Text-Specific Distractor
<table><tr><td>Method</td><td>w/ distractors</td><td>w/o distractors</td><td>∆</td></tr><tr><td>GPT-4o+P</td><td>60.5</td><td>69.4</td><td>+8.9</td></tr><tr><td>GPT-5-mini+P</td><td>71.9</td><td>75.9</td><td>+4</td></tr><tr><td>GPT-5+P</td><td>79.9</td><td>83.1</td><td>+3.2</td></tr><tr><td>Gemini-2.5 Pro+P</td><td>78.5</td><td>83.1</td><td>+4.6</td></tr><tr><td>Qwen2.5-VL-7B+P</td><td>45.4</td><td>51.0</td><td>+5.6</td></tr><tr><td>RAP</td><td>33.0</td><td>33.4</td><td>+0.4</td></tr><tr><td>R2P</td><td>46.9</td><td>49.8</td><td>+2.9</td></tr><tr><td>TAME</td><td>47.6</td><td>52.6</td><td>+5</td></tr><tr><td>PersonaVLM</td><td>52.3</td><td>51.3</td><td>-1</td></tr></table>

Table 10. Effect of inter-concept distractors. Scores are the overall Avg. on MPRBench. ∆ denotes w/o distractors minus w distractors.

## B.4. Comparative Analysis with High-Performance Backbones

In this section, we evaluate Qwen3-VL-8B [6] variants to investigate how general multi-modal intelligence translates to performance in personalized reasoning.

![](images/06f9db66205fe7f343f95718081f21512f4881fb62f7ccd7b2ee003cdbcee6fe.jpg)  
Figure 9. Comparison between General Multi-modal Capabil ities and Personalized Reasoning Performance

1) Is general multi-modal proficiency guarantee for personalized reasoning? As illustrated in Fig. 9, there is a striking contrast between a model’s performance on standard benchmarks and our MPRBench. While Qwen3- VL-8B-Instruct exhibits a massive performance leap over LLaVA-v1.5-13B in general multimodal benchmarks— specifically showing gaps of +38.1 in MMStar [10], +17.6 in MMBench [27], and +36.0 in MMMU [47]—this superiority significantly diminishes in personalized contexts. On MPRBench, the average performace gap narrows to only +8.5 points. This observation confirms that personalized is not a trivial extension of general intelligence but represents a unique bottleneck that remains challenging.

2) How do different reasoning strategies affect personalized reasoning? We compare the Instruct, CoT, and Thinking variants of Qwen3-VL-8B in Table. 12. Explicit reasoning consistently helps, with the average score rising from 56.7 to 63.8 and 67.2. The magnitude of this gain, however, differs sharply across task groups. Moving from Instruct to Thinking improves M-Avg by 23.1 points and T-Avg by 12.1 points, whereas V-Avg improves by only 2.2 points.

This pattern follows the modality taxonomy of Sec. 3.3. Tasks that require retrieving textual attributes or integrating them with visual evidence benefit substantially from multi-step reasoning; the multimodal history sub-task, for instance, improves from 43.0 to 88.6. In contrast, visionspecific sub-tasks that hinge on locating and inspecting the concept remain nearly unchanged, with spatial reasoning at 49.6 and 48.7 for Instruct and Thinking respectively. This is consistent with our earlier finding that mis-identification is the dominant failure mode. Improving personalized reasoning therefore requires stronger concept grounding, not only stronger reasoning procedures.

<table><tr><td>Query Construction Types</td><td>Sub-tasks</td></tr><tr><td>Type 1 Type 2 Type 3</td><td>Visual-Prediction, Spatial, Numerical, Visual Appearance Change, Text-Multihop (Object), Commonsense Visual-Prediction, Spatial, Numerical, Multimodal-Prediction, Sentiment (Person) Visual-Prediction, Spatial, Numerical, Text-Prediction, Multimodal-Prediction, Multimodal-Numerical, Sentiment, history (Person/Object)</td></tr></table>

Table 11. Detailed Mapping between sub-tasks and Query Construction Types

<table><tr><td></td><td colspan="5">Vision-specific Reasoning</td><td colspan="5">Text-specific Reasoning</td><td colspan="5">Multimodal</td><td></td><td></td></tr><tr><td></td><td>predict</td><td>spatial</td><td>numeric</td><td>commsense</td><td>change</td><td></td><td>V-Avg</td><td>predict</td><td>numeric</td><td>multi-hop</td><td>T-Avg</td><td>predict</td><td>sentiment</td><td></td><td></td><td></td><td>M-Avg</td><td></td></tr><tr><td>Method</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>numeric</td><td>history</td><td></td><td>Avg</td></tr><tr><td>Open-sourced+Prompting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA [26]+P</td><td>69.7</td><td>46.1</td><td>50.2</td><td>51.9</td><td></td><td>90.2</td><td>60.2</td><td>61.7</td><td>41.7</td><td>18.6</td><td>35.5</td><td>34.9</td><td></td><td>49.9</td><td>20.0</td><td>39.2</td><td>40.3</td><td>48.2</td></tr><tr><td>Qwen3-VL-8B-Instruct+P</td><td>81.2</td><td>49.6</td><td>59.8</td><td>56.9</td><td></td><td>73.9</td><td>64.4</td><td>97.5</td><td>54.2</td><td>13.7</td><td>45.6</td><td>61.9</td><td>63.5</td><td></td><td>42.0</td><td>43.0</td><td>54.3</td><td>56.7</td></tr><tr><td>Qwen3-VL-8B-Instruct-CoT+P</td><td>78.7</td><td>45.2</td><td>56.1</td><td>55.9</td><td></td><td>83.7</td><td>63.4</td><td>93.3</td><td>68.7</td><td>28.1</td><td>54.9</td><td>83.3</td><td>80.2</td><td></td><td>54.0</td><td>64.6</td><td>72.8</td><td>63.8</td></tr><tr><td>Qwen3-VL-8B-Thinking+P</td><td>76.1</td><td>48.7</td><td>61.3</td><td>60.8</td><td></td><td>87.8</td><td>66.6</td><td>97.5</td><td>80.2</td><td>26.2</td><td>57.7</td><td>77.8</td><td>78.6</td><td></td><td>66.0</td><td>88.6</td><td>77.4</td><td>67.2</td></tr></table>

Table 12. Evaluation of Advanced Reasoning Backbones and Strategies on MPRBench.

![](images/3fc5d64c41c2c0f1c7bc63a5fedb10e92df4baf90aac6870270fd74305772011.jpg)  
Figure 10. Human Verification Case.

![](images/cce3fe944c76c200b8f5b265e9fab236d681ca894968c7bebdb7e7d1f347333e.jpg)  
Figure 11. An example of a personal database for the ’Alaric’ persona

![](images/272b2af2698e17b4b7ab36342d4b80855fb97ffb4943444fc7ae02fd7be9ed67.jpg)  
Figure 12. An example of a personal database for the ’alienser’ persona

## Person

![](images/55993489222486816f5d202650a1b0c1351a1baa68e3dbb04f003992edff8ff5.jpg)  
<Person\_A>

![](images/265fe562d2ced7fe7a05728630b6bba443925d8596aa11d933f7a969de3ca4c1.jpg)  
<Person\_B>

![](images/f58ede8e8692ba5c839917709506b415af365bc398d85475f8bf6a59ed4698a4.jpg)  
<Person\_C>

![](images/f7330d73750344c91e7df3405fde39a30ac123266f86183638311d7293d79444.jpg)  
<Person\_D>

![](images/03e9b81aae2c1129b11eda5dd1314c26a3e0436d2778207ae303ce649f99f732.jpg)  
<Person\_E>

![](images/5f40b2284cb90be8a5d60bf740245e7c14083f981905672f2d5c8f2f7924d721.jpg)  
<Person\_F>

![](images/b7858a7820a476a4d762607f0b14a2844ccedfaa552b27321b1af311d56ef563.jpg)  
<Person\_G>

![](images/99dc2156e7502b55509ea09fc2627b64567d6cf25fadc79f9e8a9e5c597be602.jpg)  
<Person\_H>

![](images/e16e7f0d88327b816b0a2bb0ae8256a6d08976c9b7c584a65fc5a53faaa55377.jpg)  
<Person\_I>

![](images/33262a63197af94fb12663e2ca941b0e1d343df82a5ee0cb3d3735c4d6340ddf.jpg)  
<Person\_J>

## Object

![](images/89adc786003ec9a2d7e0e9912e6e31521f2845d93965fe11f3923d5c0598709d.jpg)  
<Object\_A>

![](images/354d85e510147e0c2c01e413031ad5336b8d2b63e19509b3d0836b099e27f4c6.jpg)  
<Object\_B>

![](images/e837412950c20eb777b35d96be630a459e288d0cb1803519a6f87beb37bbc5d5.jpg)  
<Object\_C>

![](images/3604c6b629d9bf32c2f66e5d60c33b944992ad9bf85a77d628ce2c20e10cb9a1.jpg)  
<Object\_D>

![](images/e111d40a4d07bec81949ceee774032cfa108e1e93069413c6c333d5a1e3963a2.jpg)  
<Object\_E>

![](images/a4aed050a0a70efe8bd2742c0857a80a8ee74bb36bffe8ea9e6a4de0e4243d0b.jpg)  
<Object\_F>

![](images/8cfefcd9e861103f4ab614fc260383a7b54fca8ab70116b65463be5ec00f7d60.jpg)  
<Object\_G>

![](images/79328824c99b2e815e5476b8f75aae4ba37ffe0db9a91ed270571c37c4b6b9ea.jpg)  
<Object\_H>

![](images/6499acd9fa468963151a975a6f238631a38112d14bd320f10ba73bda8a8dd80e.jpg)  
<Object\_I>

![](images/7938660f83e4f468d5d9696ba340dd852aa84ceb274b65e268c5f31c63aabb48.jpg)  
<Object\_J>

Figure 13. Overview of MPRBench Personalized Concepts

## Visual-Specific Task Examples

Reference Image

## Prediction

![](images/fa77baeb840f7b6c944304773ad59bc77b4b1475d98654e87af33b310e970c20.jpg)

## Spatial

![](images/01b34b0f1c26a9aa1b57cdd2fc4a34577687a6202a01dc27efbee6be1986b894.jpg)

## Numerical

![](images/3c3edbc5eac191d3ea4a190f61ad6d2d8a2ae3c9f1ca62de24fd66c326dd5008.jpg)

## Commonsense

![](images/e9318a3a1d2b2db5d8445f40a67db711d42205b15dfe673d0bf211ada46ac123.jpg)

<Quentin>   
Favorite Color   
- Burgundy Red   
- Olive Green   
Hobby   
- Exploring dark caves and forgotten places   
- Enjoying a pint with friends at the local pub   
Favorite Color   
- Yellow   
- Green   
Hobby   
- Playing football with friends   
- Watching professional sports matches   
<Norbert>   
Favorite Color   
- Royal Blue   
- White   
Hobby   
- Collecting designer sunglasses   
- Mentoring younger athletes   
<Anwen>   
Favorite Color   
- Emerald Green   
- Crimson Red   
...   
Hobby   
- Watching classic films in old movie theaters   
- Experimenting with bold makeup and fashion styles   
<myrat>   
Owner   
Nathaniel   
...   
History   
- On March 15, 2022, myrat was dressed up as a chef with a foil hat because Nathaniel was watching "The Great British Bake Off“

![](images/e9b0d4a89450732051c090601b00a952ee2b15f002fe8f08a1107980e1bfa5b7.jpg)

Query Image

## Visual Appearance Change

![](images/5e82bf6741400b071cd8491d068f845fbc9d15e6afe56223e72d6378298cb658.jpg)

![](images/5dff3de2753de434c9ce47cd3a33344c8b09a22a7d94c6c920a654148464fc77.jpg)

![](images/b91562c60fa90871602c78c6ed11701a3c8038af2fd533ca8a1d87b0092634c1.jpg)

![](images/b71539261b8e3d1f4830669a7254833a783f7819563528d5f54fcd69aba25e4f.jpg)

![](images/0a31f2e0394ea2e652086ab59b554e538e12884e44ffae452c38b15ad4abb121.jpg)  
Figure 14. Data Examples for Visual-Specific Personalized Reasoning

## Question & Answer

What is <Quentin> most likely to do next? A: Watch how the other person throws a dart. B: Turn around and start examining the paintings hanging on the wall. C: Sit down at the wooden table near him. D: Throw a dart.

From the perspective of <Elias>, where is the player wearing the number 4 located? A: To his immediate left side.   
B: To his immediate right side.   
C: Directly in front of him.   
D: Directly behind him.   
How many people wearing same uniform as   
<Norbert> in this image?   
A: 3   
B: 4   
C: 2   
D: 5 What is <Anwen> drinking her drink fast? A: She is very nervous.   
B: She does not want to talk to the man behind her.   
C: She is anticipating someone else.   
D: He really needs to calm down and she wants to do that by gulping down some alcohol. Which description best fits about   
<myrat>’s appearance?   
A: <myrat> wrapped in aluminum foil and   
wearing mouse ears on head.   
B: <myrat> does not exist.   
C: judging by the disguise, it’s the owner’s birthday.   
D: <myrat> looks same as usual.

![](images/c89d021af6a5de3dc4ebc82bdecc016a395bdb404d1efccb67da95d84b1abe9c.jpg)  
Figure 15. Data Examples for Text-Specific Personalized Reasoning

## Multimodal Reasoning Task Examples

## Reference Image

![](images/84333318c2b203a316026b71b990da295971727bd0c2204d0f67b89c442084a0.jpg)

## Prediction

![](images/151d9517aad2bd2a9fce73db0d7f2ea8c9776620c2e0c038dad4cd4986cecfaf.jpg)

![](images/4a0c0d57c98fe68bbdd6eb0ff94deb2c667b4a0af655a5ef0e7ee2d7c4216003.jpg)

## Sentiment

![](images/a428307265e8b8d7d52f3cc45f2a088e86bfe6b083ec068ec87024ca7b24a817.jpg)

![](images/691f510c80eee0dde8705aa9d2759e0d826fb1d36f62847506877cb41ed1d70a.jpg)

## Numerical

![](images/9bef8206f4871878ca87dddc055435e35a43a524b14195e7407fb66409e82864.jpg)

![](images/7291da55d974cc210e60d2ee2e7578c9141d6ae8330fc39aeb15d90447968cb2.jpg)

## Temporal

![](images/9c07ac44d2695ee33f60ab4230fcec573e55efb855c987a344ef9bb865529aa6.jpg)  
Favorite Food Roast chicken Treacle tart  
Non-Favorite Food - Brussels sprouts ...

## Database Snippet

## <Ivor>

Favorite Color   
- Red   
- Black

## <Celestine>

Non-Favorite Fashion - Plain wool suits - Corsets

Habit When she becomes anxious or depressed, she smokes to organize her thoughts

## <Jethro>

Favorite Color

Non-Favorite Color   
- Birhgt Yellow   
- Lavender   
- Maroon

## <mydoll>

## History

On March 10, 2025, the owner put green sunglasses on mydoll ...

On April 11, 2025, the owner placed a blue ribbon in mydoll’s hair ... On May 15, 2025, the owner posed mydoll with its arm crossed ... On May 16, 2025, the owner posed mydoll with one hand on its hip ...

\- On September 8, 2025, the owner created a new outfit for mydoll using aluminum foil over a red shirt, ...

## Query Image

![](images/560a6ead007e31669d224056cbaadd81f8f5467a8a038057c4a7cacc05655a50.jpg)

![](images/85ad4c3a4f5571f8d7bc2c8a6a41e8fcc305f6d230b48390c086fb6177f23a1b.jpg)

![](images/87c71a71124457e8cb92d7beb0a8df03747772ba95a7dcdc7389b213362ccf4b.jpg)

![](images/d0bbb99d35ce48b13ed850277868cd2c33f1c585d5a7e76797ebdd1be21bdc50.jpg)  
A: Take the mashed potatoes.  
B: Take the treacle tart.  
C: Take the Brussels sprouts.  
Figure 16. Data Examples for Multimodal Personalized Reasoning

## Question & Answer

Which item is <Ivor> most likely to select next?

D: Take the roast chicken.

How does <Celestine> feel in this image?

A: She is feeling bored and exhausted.

B: She is feeling anxious and depressed.

C: She is feeling deep shame and embarrassment.

D: She is feeling sadness.

Given <Jethro>’s personal tastes, how many of the mugs shown would he likely choose to keep in his collection?

A: 6

B: 5

C: 3

D: 4

In the context of its personal history, when did this specific scene of the <mydoll> occur?

A: After it wore a red hat for a hiking trip, but before it had a pink flower in its hair for a spring party.

B: After it wore a paper towel outfit for a fashion challenge, but before it was brought to the owner’s office.

C: After it wore an aluminum foil astronaut outfit, but before it wore green sunglasses.

D: After the owner placed a blue ribbon in its hair, but before the owner posed it with one hand on its hip.

![](images/96cb85dfb44722cba0b9d5d5354c269cd92321aafce1afcd4adbd35199fa0084.jpg)

![](images/7eab29b26569be4ae387146ef17f42869ec26ad746eca467c283d3b2cc735c60.jpg)  
❌ B: Raise his arm to point in the same direction as the person next to him.

On June 5th, 2021, Miles stood on the bustling streets of New York City during World Environment Day. As a volunteer guided a group of eager listeners, Miles raised his arm to point in the same direction as the person next to him, eager to share the excitement of the urban tree-planting event they were attending.

Figure 17. Overview of Visual-specific Distractor  
![](images/b902f278849608021934089eb359bb2b94427f0ae95c6483b71fd722c2bcc7b0.jpg)  
Figure 18. Distractor Types for Text-Specific Query Images

![](images/73d898b79139f5d8e0491245df2026bcfe768db70d2874c38f0c67ea8d02783f.jpg)  
Figure 19. System Prompt for Textual Reference Construction - Step 1. Visually-grounded Attribute (Person Concept)

![](images/17f1de2d442fe48f6c6eac3bae8c2ea38166bf3387acb48c08718e63b6a7bb48.jpg)  
Figure 20. System Prompt for Textual Reference Construction - Step 3. Context-aware Attribute (Person Concept)

![](images/1c6bd6e19e467f8205e7ed5169f02e7fbc3f1f73aed6d3390bc64ea2b75261dc.jpg)  
Figure 21. System Prompt for Textual Reference Construction - Step 3. Context-aware Attribute (Person Concept)

![](images/3b55fdac6650b01f3f1a0f797d5b42448aed735dbc8356334ffe1c184c9a5822.jpg)  
Figure 22. System Prompt for Textual Reference Construction - Step 3. Context-aware Attribute (Object Concept ConConChi)

![](images/ebfeeef990c8d248f8185d88a69fdd16c11a7e6948387a25f49262b03bf34e4f.jpg)  
Figure 23. System Prompt for Textual Reference Construction - Step 3. Context-aware Attribute (Object Concept ConConChi)

![](images/7d92e6edb83bb1ba66070e7563de2e9f3b5efd8d54229990aa00304a47319269.jpg)  
Figure 24. System Prompt for Textual Reference Construction - Step 3. Context-aware Attribute (Object Concept RPC)

![](images/0247c2a5de7de960e4ca90c40e0812f4265428bd71a5031a8a60994113a16d4f.jpg)  
Figure 25. System Prompt for Generating Prediction Task for Person Concepts (Step 1).

![](images/fed879101870b3c2a59747350c8eaee9af7ba29c178eb253c30c50ca111c1cf8.jpg)  
Figure 26. System Prompt for Generating Prediction Task for Person Concepts (Step 2).

![](images/e089d4992e913cce65cc99d14a48e29a43bfbc25887344f06654d06e7a28b6e9.jpg)  
Figure 27. System Prompt for Generating Prediction Task for Person Concepts (Step 3).

![](images/d69f82059be2436a7927d4a5779c14c3dfd5aa78c981ddc7bed8f44a19276f5c.jpg)  
Figure 28. System Prompt for Generating Prediction Task for Person Concepts (Step 4).