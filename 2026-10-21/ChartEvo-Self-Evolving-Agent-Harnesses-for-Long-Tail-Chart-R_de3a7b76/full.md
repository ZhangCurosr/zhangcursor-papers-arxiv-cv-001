# ChartEvo: Self-Evolving Agent Harnesses for Long-Tail Chart Reasoning

Qi Wang<sup>1,2</sup>\*, Peijie Wang<sup>1,2</sup>\*, Jun Cao<sup>1,2</sup>, Haoyang Tong<sup>1,3,4</sup>

Xiao-Hui Li<sup>1,2</sup>, Fei Yin<sup>1,2</sup>, Cheng-Lin Liu<sup>1,2†</sup>

<sup>1</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA

<sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences

<sup>3</sup>New Laboratory of Pattern Recognition, CASIA

<sup>4</sup>Zhongguancun Academy

wangqi2026@ia.ac.cn wangpeijie2023@ia.ac.cn {fyin, liucl}@nlpr.ia.ac.cn

## Abstract

Multimodal large language models (MLLMs) have substantially advanced chart understanding and reasoning, yet their performance remains highly uneven across chart types. While modern MLLMs perform well on common charts, their accuracy drops sharply on long-tail charts with diverse visual structures and reasoning requirements. Tool-augmented agents can improve complex chart reasoning by introducing specialized visual capabilities, but manually engineering strategies and tools for the heterogeneous and evolving long tail is difficult to scale. We introduce ChartEvo, a recursive self-improvement framework that enables chart agents to adapt to underperforming chart types through experience-driven harness-level self-improvement. ChartEvo jointly evolves reusable chart-specific Skills and executable Tools, while learning reusable local tool-call chains to compose evolved capabilities and control the inference overhead of tool-augmented reasoning. On long-tail charts from Chart-Bench and ChartX, ChartEvo improves accuracy over its base MLLM, Qwen3.7-Plus, by 45.6% and 32.5%, respectively, demonstrating the effectiveness of recursive self-improvement for long-tail chart adaptation.

## 1 Introduction

Recent advances in multimodal large language models (MLLMs) have substantially improved chart understanding and reasoning, yielding strong overall performance on established chart question answering (CQA) benchmarks (Masry et al., 2022; Xu et al., 2024). Tool-augmented approaches such as ChartAgent (Kaur et al., 2026) further extend chart reasoning by incorporating specialized visual tools for complex visual and numerical tasks. However, strong aggregate performance can obscure substantial performance disparities across

chart types (Masry et al., 2025; Wei et al., 2025). On the numerical question answering (NQA) task in ChartBench, Qwen3.7-Plus achieves an average accuracy of 80.0% across 26 chart types. Among them, 17 types exceed 90% accuracy, with an average of 96.9%, whereas the remaining nine average only 48.0%. As illustrated in Fig. 1, this disparity reveals a highly uneven performance distribution: while many chart types are handled reliably, the remaining underperforming types form a heterogeneous tail with distinct visual structures and reasoning requirements. We refer to these persistently underperforming chart types as long-tail charts. This uneven performance distribution suggests that further progress in CQA increasingly depends on an agent’s ability to adapt to these diverse, chartspecific capability gaps.

![](images/6639e6f3c24b70bd81e4df985f96faac5a5880b28e58617bbb8d1cae123b56e3.jpg)  
Figure 1: Representative well-performing and underperforming chart types in ChartBench test split.

Addressing this heterogeneous long tail requires capable MLLMs to continually acquire new capabilities for chart types they have not yet mastered. One possible route is to further train the model, encoding the missing capabilities into its parameters. However, repeatedly adapting model parameters to address localized capability gaps across diverse chart structures and reasoning patterns is neither naturally modular nor readily extensible. Tool-augmented chart agents provide a more modular alternative by introducing specialized visual operations that complement the MLLM. Yet, existing approaches largely rely on manually predefined tools and usage strategies, leaving the agent’s capability boundary constrained by human engineering. As unresolved failures span diverse chart structures and reasoning requirements, manually extending the agent for each distinct failure mode becomes difficult to scale. This motivates shifting long-tail adaptation from manual capability engineering toward autonomous capability acquisition.

![](images/455f22e3c0a7368462504dc2eafceae86a9b9b43edbcd7ffd83f975a80f35cf9.jpg)  
Figure 2: Illustrative comparison of chart reasoning approaches. Direct MLLMs and fixed-tool agents can misinterpret chart encodings. ChartEvo evolves skills and tools into validated, reusable Self-Evolving Local Toolchains (SELT), enabling evidence-grounded reasoning through a single toolchain invocation.

To this end, we introduce ChartEvo, a recursively self-improving chart agent that autonomously evolves its harness to acquire and organize capabilities for long-tail chart reasoning. An illustrative overview of ChartEvo and its contrast with direct MLLM and fixed-tool reasoning is shown in Fig. 2. ChartEvo progressively accumulates reusable capabilities in its external harness. Specifically, it jointly evolves chart-specific Skills that capture reusable reasoning knowledge and executable Tools that expand the agent’s visual and numerical capabilities. Skills thus encode how to reason, while Tools extend what the agent can execute, enabling acquired capabilities to be retained and effectively applied across related chart instances.

Beyond capability acquisition, a complementary challenge lies in how to efficiently organize and execute these capabilities. Conventional ReActstyle agents repeatedly interleave reasoning, tool invocation, and observation (Yao et al., 2023). In chart reasoning, many tasks require global reasoning to determine what should be measured, but once the target and procedure are established, the remaining computation often consists of locally structured visual operations with stable dependencies. Executing each intermediate step through the

ReAct loop therefore requires routing tool outputs back through the MLLM, introducing additional model–tool interaction overhead. We introduce Self-Evolving Local Toolchains (SELT) to address this issue. Through SELT, ChartEvo evolves and preserves validated local Tool compositions as reusable execution structures, enabling structured procedures to execute with fewer MLLM-mediated tool interactions while retaining the flexibility of dynamic Tool composition for previously unseen cases. Skills capture reusable reasoning knowledge, Tools expand executable capabilities, and SELT organize how these capabilities are locally composed and executed. ChartEvo therefore evolves not only the agent’s capability boundary but also the execution structure through which these capabilities are applied. Comprehensive experiments validate the effectiveness of ChartEvo for long-tail chart reasoning and its ability to control the execution overhead of tool-augmented reasoning. Our contributions are summarized as follows:

1. Harness Recursive Self-Improvement: We introduce ChartEvo, a harness-level recursive self-improvement framework that autonomously evolves reusable Skills and executable Tools through iterative feedback, enabling continual capability acquisition for long-tail chart reasoning.

2. Self-Evolving Local Toolchains: We propose SELT, extending harness-level self-improvement to execution structures by evolving and reusing validated Tool compositions, thereby reducing redundant model–tool interactions and improving execution efficiency in ReAct-style reasoning.

3. Effective and Efficient Long-Tail Chart Reasoning: On long-tail charts from ChartBench and ChartX, ChartEvo improves accuracy over the baseline MLLM by 45.6% and 32.5%, respectively, while substantially reducing model calls with comparable accuracy.

## 2 Related Work

## 2.1 Chart Question Answering

Early CQA studies formulated chart understanding as task-specific visual question answering, progressing from synthetic charts and template-based questions (Kahou et al., 2018; Kafle et al., 2018) to real-world charts and more complex numerical and logical reasoning (Methani et al., 2020; Chaudhry et al., 2020; Masry et al., 2022). Vision-language pretraining further enabled more general chart understanding through derendering and structured representations (Liu et al., 2023b; Masry et al., 2023).

With the rise of MLLMs, recent studies have adapted general-purpose models to CQA through chart-specific instruction tuning and multimodal alignment (Han et al., 2023; Masry et al., 2024; Meng et al., 2024), while increasingly challenging benchmarks expose persistent limitations in precise visual and numerical reasoning (Xu et al., 2024; Xia et al., 2025; Wang et al., 2024; Masry et al., 2025). Most recently, ChartAgent addresses such limitations through tool-augmented visual reasoning (Kaur et al., 2026). However, its reliance on predefined tools and usage strategies makes manual capability engineering difficult to scale across long-tail charts with diverse structures and reasoning requirements, motivating us to explore RSI for autonomously expanding agent capabilities.

## 2.2 Recursive Self-Improvement

Recursive self-improvement (RSI) studies how AI systems leverage self-generated experience to improve their capabilities. Existing research explores complementary pathways, including updating model parameters through self-generated data and reinforcement learning (Chen et al., 2024; Kuba et al., 2025; Yang et al., 2026b), as well as evolving the external systems that mediate agent behavior. We focus on the latter. Gödel Agent enables agents to recursively modify their logic and behavior, while Meta-Harness further treats the entire agent harness as an optimization object, extending self-improvement to the system level (Yin et al., 2025; Lee et al., 2026).

Recent work increasingly instantiates such improvement through modular and reusable agent capabilities (Wei et al., 2026; Zhang et al., 2026; Pan et al., 2026; Belikova et al., 2026). SkillForge, SkillOpt, and MetaSkill-Evolve enable agents to construct and refine reusable skills from execution experience (Liu et al., 2026; Yang et al., 2026a; Wang et al., 2026). In parallel, early studies such as CREATOR and LATM established autonomous tool creation (Qian et al., 2024; Cai et al., 2024), while recent self-evolving agents increasingly incorporate autonomous tool discovery, construction, repair, and maintenance (Wölflein et al., 2025; Jin et al., 2025; Kujanpää et al., 2026; Deng et al., 2025). Building on these advances, we jointly evolve chart-specific Skills and Tools for long-tail chart reasoning, while learning reusable local toolcall chains for efficient execution.

## 3 Methodology

## 3.1 Overview

ChartEvo treats the external harness of a chartreasoning agent as the object of continual selfevolution. Given a chart image I and a question $q ,$ the underlying model reasons with the current harness to produce an answer. In ChartEvo, the harness consists of Skills, Tools, and Self-Evolving Local Toolchains (SELT). Skills encode reusable chart-specific reasoning and capability-use knowledge, Tools provide executable visual and numerical operations, and SELT persist validated compositions of Tool calls together with their execution dependencies. Skills and Tools thus define the reasoning knowledge and executable capabilities available to the agent, while SELT organize how these capabilities are composed and executed.

As illustrated in Fig. 3, ChartEvo formulates harness-level recursive self-improvement as an iterative outer-loop process. Given the current harness and its execution outcomes on chart-reasoning tasks, an agentic proposer introduces or revises Skills, Tools, and SELT. Validated updates are persisted into the next harness state, which is subsequently reused for chart reasoning and serves as the basis for further harness evolution.

During self-evolution, supervision signals are used to evaluate execution outcomes and guide harness updates. After evolution, the resulting harness is deployed for inference on unseen chart instances. Overall, ChartEvo instantiates the agent as a Model + Evolving Harness system: the model serves as the general reasoning core, while the harness progressively accumulates Skills, Tools, and SELT to acquire new capabilities and more structured execution patterns for long-tail chart reasoning.

![](images/42f4915a3711535d1e9125c68e8152eb34e26badafcef8b0a587eca5c11cd753.jpg)  
Figure 3: Overview of ChartEvo. (A) Harness self-evolution jointly updates Skills, Tools, and SELT using answer evaluation and execution feedback, retaining validated updates for subsequent rounds. (B) With the harness frozen at test time, the rose-chart example illustrates five local reasoning stages within a single SELT invocation.

## 3.2 Harness Recursive Self-Improvement

ChartEvo starts self-evolution from initial harness H<sub>0</sub> containing a set of generic chart-processing Tools. These Tools provide the basic primitives for bootstrapping chart reasoning, while task-specific capabilities are progressively acquired through selfevolution. At evolution step t, the current harness H<sub>t</sub> is used to solve a task, producing both a prediction and its execution trajectory τ<sub>t</sub>. The agentic proposer jointly considers the final answer, task-level evaluation feedback, and the observed execution trajectory to analyze the behavior of the current harness and identify components that require further improvement. This joint observation allows the proposer to assess both task-level correctness and the reliability of the execution process, including cases where an apparently acceptable prediction is supported by an unreliable trajectory.

Based on the current harness, its execution trajectory, and the supervision available during selfevolution, the agentic proposer can introduce new harness components or revise existing Skills, Tools, and SELT. ChartEvo allows multiple component types to evolve jointly within the same step, enabling coordinated updates to reasoning knowledge, executable capabilities, and execution structures. A proposal may therefore span multiple harness components and be evaluated as a coordinated update to the current harness. Ground-truth answers are available during self-evolution for task evaluation and candidate validation, while no ground-truth solution traces are provided.

Candidate updates are validated by re-executing the task under the updated harness. Both the resulting task outcome and the corresponding execution trajectory are examined to determine whether the updated harness behaves as intended. Validated components and structures are then persisted into $\mathcal { H } _ { t + 1 }$ , which becomes the basis for subsequent execution and further self-evolution. Through repeated execution, analysis, proposal or revision, and validation, ChartEvo recursively evolves its harness over successive tasks, such that subsequent proposals are made over an already evolved harness.

Sec. 3.3, Sec. 3.4, and Sec. 3.5 describe Skill/Tool evolution, Self-Evolving Local Toolchains, and inference with the evolved harness, respectively.

## 3.3 Skill and Tool Evolution

ChartEvo treats Skills and Tools as two complementary carriers of harness capabilities. Skills encode reusable reasoning and capability-use knowledge, guiding the model in how to interpret, select, and use available capabilities. Tools provide executable visual and numerical operations that transform chart information into structured evidence consumable by subsequent reasoning. Together, they define the reasoning knowledge and executable capabilities available to the current harness. During selfevolution, agentic proposer determines whether an observed deficiency primarily arises from how existing capabilities are composed and used, or from the lack of an executable evidence capability. When the available Tools can already provide the required information but their selection, invocation, or use does not support correct reasoning, evolution primarily targets Skills. When the existing Tool set cannot provide the required structured evidence, or when a Tool exhibits incorrect behavior, insufficient coverage, or an inadequate output interface, the proposer introduces or revises Tools. Existing components are revised when their intended semantics remain valid, whereas new components are introduced when the required reasoning pattern or executable capability is semantically distinct from those already available.

Skill and Tool evolution can be coupled within the same self-evolution step, allowing newly introduced or revised executable capabilities to be incorporated into corresponding reasoning and capability-use knowledge. Candidate Skills and Tools are re-executed and validated under the updated harness, with Tools additionally checked for executable behavior. Validated updates are persisted in the harness and remain available for reasoning and further evolution. While Skill and Tool evolution expands the capabilities available to the harness, stable multi-step execution structures over these capabilities are captured through SELT.

## 3.4 Self-Evolving Local Toolchains

Self-Evolving Local Toolchains (SELT) persist validated Tool compositions as reusable execution structures, avoiding repeated reconstruction of the same Tool-call patterns through model-mediated interaction. Each SELT represents a bounded executable transformation with explicit inputs, outputs, execution order, and internal Tool dependencies. The boundedness of a SELT is defined by its functional contract: the Agent resolves the taskspecific target and required arguments, after which the SELT executes the structured transformation and returns its designated output.

During self-evolution, the Agent can reuse an existing SELT when an applicable execution structure is available. Otherwise, it dynamically composes individual Tools, coordinating their execution and intermediate results through model interactions. Validated compositions with stable execution structures can subsequently be persisted as new SELT, while existing SELT can be revised as Tools, Skills, or execution requirements evolve. Unlike dynamic Tool composition, SELT delegate inter-step parameter resolution and intermediate output propagation to the runtime, eliminating repeated Agent planning and binding within established Toolchains.

SELT usage is jointly mediated by Skills, the Agent, and the runtime. Skills maintain the available SELT together with their routing and usage knowledge. The runtime progressively discloses context-relevant candidate SELT according to the current question, execution stage, and accumulated observations, exposing only a relevant subset of the harness capabilities at each decision step. The Agent selects a concrete SELT based on question semantics, available evidence, and Skill-provided routing knowledge, and binds the task-specific arguments required for execution. The runtime then executes the selected SELT while enforcing invocation constraints, Tool versions, and execution checks.

Skills and SELT therefore play complementary roles: Skills specify when and how an execution structure should be used, whereas SELT encode its concrete Tool composition and dependencies. Once the Agent binds the task target, entities, and execution conditions, the selected SELT executes the structured procedure without further modelmediated coordination of internal Tool calls. When no applicable SELT is available, the Agent can fall back to dynamic Tool composition, whose successful execution may drive subsequent SELT evolution.

## 3.5 Inference with the Evolved Harness

After self-evolution, ChartEvo fixes the resulting harness and applies it to unseen chart instances. The Agent uses the evolved Skills, Tools, and SELT according to the current question and execution context. When an applicable SELT is available, its persistent execution structure can be directly reused; otherwise, the Agent can dynamically compose Tools to complete the task. To handle unsuccessful execution, ChartEvo supports two fallback conditions: the Agent may proactively switch to direct visual answering when it determines that harness-assisted reasoning can no longer proceed, or the system triggers fallback when the maximum action budget of 15 is reached. In both cases, direct visual answering is performed using only the original chart image and question.

Table 1: Comparison on ChartBench. NQA accuracy (%) on nine long-tail chart types. The best and second-best results are highlighted in bold and underline, respectively.
<table><tr><td>Method</td><td>Area</td><td>3Dbar-multi</td><td>3D-stack</td><td>Ring</td><td>Sector</td><td>Radar-M</td><td>Radar-MF</td><td>Radar-S</td><td>Stock</td><td>Avg.</td></tr><tr><td colspan="9">Direct Visual MLLMs</td></tr><tr><td>DeepSeek-v4.1-Flash</td><td>30.0</td><td>38.0</td><td>12.0</td><td>28.0</td><td>44.0</td><td>14.0</td><td>26.0</td><td>12.0</td><td>80.0</td><td>31.6</td></tr><tr><td>Qwen3.7-Plus</td><td>60.0</td><td>42.0</td><td>14.0</td><td>48.0</td><td>46.0</td><td>38.0</td><td>46.0</td><td>54.0</td><td>84.0</td><td>48.0</td></tr><tr><td>Qwen3.8-27B</td><td>36.0</td><td>42.0</td><td>20.0</td><td>34.0</td><td>42.0</td><td>42.0</td><td>48.0</td><td>26.0</td><td>86.0</td><td>41.8</td></tr><tr><td>Qwen3.8-Max</td><td>84.0</td><td>44.0</td><td>30.0</td><td>72.0</td><td>74.0</td><td>74.0</td><td>70.0</td><td>78.0</td><td>86.0</td><td>68.0</td></tr><tr><td>Kimi-K2.6</td><td>30.0</td><td>42.0</td><td>14.0</td><td>38.0</td><td>46.0</td><td>48.0</td><td>40.0</td><td>38.0</td><td>86.0</td><td>42.4</td></tr><tr><td>GPT-5.6-Luna</td><td>42.0</td><td>50.0</td><td>12.0</td><td>54.0</td><td>48.0</td><td>70.0</td><td>62.0</td><td>34.0</td><td>90.0</td><td>51.3</td></tr><tr><td>GPT-5.6-Terra</td><td>78.0</td><td>40.0</td><td>20.0</td><td>36.0</td><td>26.0</td><td>62.0</td><td>58.0</td><td>60.0</td><td>90.0</td><td>52.2</td></tr><tr><td>GPT-5.6-Sol</td><td>86.0</td><td>46.0</td><td>28.0</td><td>64.0</td><td>54.0</td><td>82.0</td><td>80.0</td><td>88.0</td><td>92.0</td><td>68.9</td></tr><tr><td colspan="11"></td></tr><tr><td>DePlot</td><td>18.0</td><td>34.0</td><td>Chart-Specialized Models/Agents 8.0</td><td>4.0</td><td>6.0</td><td>2.0</td><td>2.0</td><td>4.0</td><td>0.0</td><td>8.7</td></tr><tr><td>TinyChart</td><td>32.0</td><td>38.0</td><td>16.0</td><td>20.0</td><td>22.0</td><td>10.0</td><td>8.0</td><td>4.0</td><td>4.0</td><td>17.1</td></tr><tr><td>ChartInstruct</td><td>20.0</td><td>16.0</td><td>6.0</td><td>8.0</td><td>4.0</td><td>8.0</td><td>6.0</td><td>4.0</td><td>6.0</td><td>8.7</td></tr><tr><td>ChartVLM</td><td>16.0</td><td>30.0</td><td>6.0</td><td>8.0</td><td>6.0</td><td>2.0</td><td>4.0</td><td>6.0</td><td>8.0</td><td>9.6</td></tr><tr><td>ChartAgent</td><td>30.0</td><td>20.0</td><td>6.0</td><td>94.0</td><td>80.0</td><td>22.0</td><td>20.0</td><td>6.0</td><td>64.0</td><td>38.0</td></tr><tr><td>ChartEvo (Ours)</td><td>100.0</td><td>92.0</td><td>92.0</td><td>98.0</td><td>96.0</td><td>90.0</td><td>88.0</td><td>90.0</td><td>96.0</td><td>93.6</td></tr></table>

Table 2: Comparison on ChartX. NQA accuracy (%) on eight long-tail chart types. The best and second-best results are highlighted in bold and underline, respectively.
<table><tr><td>Method</td><td>Radar</td><td>3D-Bar</td><td>Rose</td><td>Area</td><td>Bubble</td><td>Multi-axes</td><td>Treemap</td><td>Ring</td><td>Avg.</td></tr><tr><td colspan="10">Direct Visual MLLMs</td></tr><tr><td>DeepSeek-v4.1-flash</td><td>4.2</td><td>25.0</td><td>35.9</td><td>28.0</td><td>62.5</td><td>46.2</td><td>75.6</td><td>70.3</td><td>43.4</td></tr><tr><td>Qwen3.7-Plus</td><td>16.7</td><td>21.4</td><td>46.2</td><td>48.0</td><td>65.6</td><td>79.5</td><td>75.6</td><td>75.7</td><td>53.6</td></tr><tr><td>Qwen3.8-27B</td><td>8.3</td><td>14.3</td><td>43.6</td><td>40.0</td><td>62.5</td><td>56.4</td><td>75.6</td><td>67.6</td><td>46.0</td></tr><tr><td>Qwen3.8-Max</td><td>54.2</td><td>32.1</td><td>79.5</td><td>84.0</td><td>87.5</td><td>89.7</td><td>87.8</td><td>67.6</td><td>72.8</td></tr><tr><td>Kimi-K2.6</td><td>12.5</td><td>28.6</td><td>61.5</td><td>40.0</td><td>75.0</td><td>74.4</td><td>58.5</td><td>75.7</td><td>53.3</td></tr><tr><td>GPT-5.6-Luna</td><td>41.7</td><td>32.1</td><td>59.0</td><td>88.0</td><td>78.1</td><td>84.6</td><td>90.2</td><td>81.1</td><td>69.4</td></tr><tr><td>GPT-5.6-Terra</td><td>66.7</td><td>46.4</td><td>82.1</td><td>92.0</td><td>71.9</td><td>84.6</td><td>85.4</td><td>94.6</td><td>78.0</td></tr><tr><td>GPT-5.6-Sol</td><td>75.0</td><td>46.4</td><td>97.4</td><td>92.0</td><td>84.4</td><td>87.2</td><td>95.1</td><td>97.3</td><td>84.4</td></tr><tr><td colspan="10">Chart-Specialized Models/Agents</td></tr><tr><td>ChartInstruct</td><td>8.3</td><td>10.7</td><td>10.3</td><td>8.0</td><td>6.2</td><td>5.1</td><td>9.8</td><td>5.4</td><td>8.0</td></tr><tr><td>ChartVLM</td><td>25.0</td><td>28.6</td><td>17.9</td><td>12.0</td><td>25.0</td><td>15.4</td><td>14.6</td><td>29.7</td><td>21.0</td></tr><tr><td>DePlot</td><td>20.8</td><td>14.3</td><td>5.1</td><td>16.0</td><td>31.2</td><td>15.4</td><td>14.6</td><td>5.4</td><td>15.4</td></tr><tr><td>TinyChart</td><td>25.0</td><td>28.6</td><td>15.4</td><td>24.0</td><td>25.0</td><td>23.1</td><td>7.3</td><td>10.8</td><td>19.9</td></tr><tr><td>ChartAgent</td><td>29.2</td><td>28.6</td><td>28.2</td><td>32.0</td><td>68.8</td><td>15.4</td><td>65.9</td><td>51.4</td><td>39.9</td></tr><tr><td>ChartEvo (Ours)</td><td>83.3</td><td>57.1</td><td>89.7</td><td>92.0</td><td>84.4</td><td>92.3</td><td>95.1</td><td>94.6</td><td>86.1</td></tr></table>

## 4 Experiments

We evaluate ChartEvo on two chart reasoning benchmarks, ChartBench and ChartX. We focus on Numerical Question Answering (NQA), which demands precise visual evidence extraction and numerical reasoning, providing a comprehensive evaluation of an agent’s chart reasoning capabilities. We select 9 long-tail chart types from ChartBench and 8 from ChartX. For each chart type, we use 100 training questions for self-evolution and evaluate on a total of 450 test questions from ChartBench and 265 from ChartX. Self-evolution is conducted independently for the two benchmarks.

Following common evaluation practices in chart reasoning, we adopt a 5% relative-error tolerance to account for uncertainty in visual representations and numerical readings. After self-evolution, the resulting harness is used for evaluation on the test set. For charts with explicit numerical annotations, the Agent directly performs visual answering.

Table 3: Comparison of Inference Costs. Average inference cost per question across ChartBench and ChartX. Lower values indicate lower cost.
<table><tr><td>Method</td><td>Tokens↓</td><td>Time (s) ↓</td></tr><tr><td>Qwen3.7-Plus</td><td>2,454.5</td><td>42.71</td></tr><tr><td>Qwen3.8-27B</td><td>672.1</td><td>11.90</td></tr><tr><td>Qwen3.8-Max</td><td>2,975.4</td><td>61.79</td></tr><tr><td>Kimi-K2.6</td><td>3,612.1</td><td>46.70</td></tr><tr><td>DeepSeek-v4.1-Flash</td><td>8,193.7</td><td>84.29</td></tr><tr><td>ChartEvo (Ours)</td><td>2,907.0</td><td>68.76</td></tr></table>

We employ Qwen3.7-Plus as the main Agent and GPT-5.6-Sol as the agentic proposer. The maximum Agent action budget is set to 15. We compare ChartEvo against two groups of baselines: (1) direct MLLMs, including DeepSeek-v4.1-Flash (DeepSeek-AI, 2026), Qwen3.7-Plus (Qwen Team, 2026a), Qwen3.8- 27B, Qwen3.8-Max (Qwen Team, 2026b), Kimi-K2.6, GPT-5.6-Luna, GPT-5.6-Terra, and GPT-5.6-Sol (OpenAI, 2026); and (2) chart-specialized methods, including DePlot (Liu et al., 2023a), Tiny-Chart (Zhang et al., 2024), ChartInstruct (Masry et al., 2024), ChartVLM (Xia et al., 2025), and ChartAgent (Kaur et al., 2026).

4.1 Comparison with State-of-the-art Solvers Tables 1 and 2 present the comparison of ChartEvo with existing methods on ChartBench and ChartX, respectively. On ChartBench, ChartEvo achieves an average accuracy of 93.6% across nine longtail chart types, outperforming its base model, Qwen3.7-Plus, by 45.6 percentage points (93.6% vs. 48.0%), while surpassing all evaluated direct visual MLLMs and chart-specialized models and agents. Notably, ChartEvo achieves the best performance across all nine chart types, with eight reaching at least 90% accuracy, demonstrating consistent improvements across diverse chart structures. For instance, on 3D-stack charts with complex spatial structures, ChartEvo improves accuracy from 14% to 92%. On ChartX, ChartEvo also demonstrates substantial overall improvements across eight longtail chart types, achieving an average accuracy of 86.1% and slightly outperforming GPT-5.6-Sol. Improvements over the base model are observed across chart types with diverse visual structures and reasoning requirements. For example, on Radar charts, ChartEvo increases accuracy from 16.7% to 83.3%. Overall, these results demonstrate that recursively evolving the external harness can substantially improve the capabilities of the underlying model in long-tail chart reasoning.

Table 4: Ablation of Skill and Tool Evolution. NQA accuracy (%) on five ChartBench chart types.
<table><tr><td>Configuration</td><td>Accuracy (%)</td></tr><tr><td>Base MLLM (Qwen3.7-Plus)</td><td>42.8</td></tr><tr><td>with Evolved Skill-only</td><td>55.6</td></tr><tr><td>with Base Harness</td><td>57.2</td></tr><tr><td>with Evolved Tools</td><td>84.4</td></tr><tr><td>ChartEvo (Full)</td><td>94.8</td></tr></table>

## 4.2 Cost Analysis

We evaluate the inference cost of ChartEvo against direct visual MLLMs using completion tokens and wall-clock time. All results are averaged over the same 715 test questions from ChartBench and ChartX, providing a consistent basis for comparing inference overhead.

As shown in Table 3, ChartEvo requires an average of 2,907 completion tokens and 68.76 seconds per question, representing an 18% increase in tokens and an additional 26.05 seconds compared with its base model, Qwen3.7-Plus. Its token consumption remains below Kimi-K2.6 and DeepSeekv4.1-Flash, and is comparable to Qwen3.8-Max, while its latency is lower than DeepSeek-v4.1- Flash. The additional overhead is associated with multi-step Agent–Tool interactions, which enable structured evidence acquisition and numerical reasoning. Combined with the substantial accuracy improvements reported above, these results highlight the performance–cost trade-off of ChartEvo: stronger long-tail chart reasoning is achieved with moderate additional token generation, accompanied by increased execution latency.

## 4.3 Ablation Study

4.3.1 Ablation on Skill and Tool Evolution To assess Skill and Tool evolution, we perform ablations on five ChartBench chart types: Area, 3D-stacked, Ring, Sector, and Radar multi-fill. We compare five configurations: Base MLLM (Qwen3.7-Plus), Evolved Skill-only, Base Harness, Evolved Tools, and full ChartEvo. Evolved Skill-only incorporates evolved reasoning guidance, Base Harness provides generic executable operations, and Evolved Tools-only includes evolved Tools without evolved Skills. As shown in Table 4, Evolved Skill-only and Base Harness achieve accuracies of 55.6% and 57.2%, respectively, both outperforming the Base MLLM at 42.8%.

Evolved Tools-only further reaches 84.4%, a 27.2 percentage-point improvement over Base Tools.  
![](images/067b8a3701e423ed090e5733b5f0135288d56504d5e4df4de3bcf87a54309b29.jpg)  
Figure 4: Per-type NQA accuracy on five ChartBench chart types.

Despite similar overall accuracy, Skill-only and Base Tools exhibit distinct performance patterns, as shown in Fig. 4. On Area charts, Skill-only achieves 92%, exceeding Base Tools at 80%, demonstrating the value of chart-specific reasoning guidance. Conversely, Base Tools achieves 94% on Sector charts, compared with 60% for Skill-only, highlighting the importance of executable numerical measurements. These results reveal the complementary roles of Skills and Tools. Both configurations, however, remain below 20% on 3D-stacked charts. With evolved Tools, accuracy increases from 16% to 90% on 3D-stacked and from 44% to 98% on Ring, primarily benefiting from specialized 3D measurement and ring-marker counting capabilities. Meanwhile, Sector maintains 94% accuracy as its required measurements are already supported by existing Tools. Radar multi-fill also remains at 52%, reflecting limitations in Tool coverage and effective invocation. These results indicate that Tool evolution depends on both acquiring the necessary executable capabilities and effectively utilizing them. The full ChartEvo further improves Radar multi-fill accuracy to 88%, demonstrating the benefits of the integrated evolved harness.

## 4.3.2 Analysis of SELT

To isolate the contribution of Self-Evolving Local Toolchains (SELT), we conduct a controlled ablation on five ChartBench chart types: Area, 3Dstacked, Ring, Sector, and Radar multi-fill. Both configurations retain the same evolved Skills, Tools, and base model, while the variant without SELT relies on dynamic Tool composition instead of persistent Toolchains. As shown in Table 6, enabling SELT reduces average model API calls from 6.19 to 2.73 (55.9% reduction), while improving accuracy from 91.2% to 94.8%. This comparison evaluates whether reusing validated Tool compositions can reduce repeated model–Tool interactions without compromising reasoning performance.

Table 5: Inference Execution Statistics. Execution path distribution, accuracy, and inference costs across ChartBench and ChartX.
<table><tr><td>Execution Path</td><td></td><td>Ratio (%) Accuracy (%)</td></tr><tr><td>Chain-based Completion</td><td>77.9</td><td>98.0</td></tr><tr><td>Direct Visual Answering</td><td>9.9</td><td>94.4</td></tr><tr><td>Fallback</td><td>12.2</td><td>44.8</td></tr></table>

Average Calls per Question
<table><tr><td>Avg. API Calls 2.30</td></tr><tr><td>Avg. Tool Calls 5.52</td></tr></table>

Table 6: SELT Ablation on ChartBench. Comparison of inference accuracy and execution costs with and without persistent SELT across five chart types.

<table><tr><td>Configuration</td><td>Acc. (%)</td><td>API Calls</td><td>Tool Calls</td></tr><tr><td>w/o SELT</td><td>91.2</td><td>6.19</td><td>5.41</td></tr><tr><td>ChartEvo (Full)</td><td>94.8</td><td>2.73</td><td>5.27</td></tr></table>

We analyze the inference behavior of ChartEvo across ChartBench and ChartX. As shown in Table 5, 77.9% of questions are completed through structured Tool chains, while 9.9% are answered directly through visual inference and 12.2% through fallback. Chain-based answers achieve 98.0% accuracy, indicating the reliability of structured visual and numerical execution. ChartEvo averages 2.30 model API calls and 5.52 underlying Tool calls per question, demonstrating how multiple executable operations can be coordinated with limited model interactions. Together, the controlled ablation and execution statistics characterize the efficiency benefits of persistent SELT and the broad coverage of structured Tool-chain reasoning.

## 5 Conclusion

This paper presents ChartEvo, a harness-level recursive self-improvement framework for long-tail chart reasoning. By evolving Skills, Tools, and Self-Evolving Local Toolchains, ChartEvo enables autonomous capability acquisition and efficient execution. On ChartBench and ChartX, it improves accuracy over Qwen3.7-Plus by 45.6% and 32.5%, respectively, while SELT substantially reduce model calls without accuracy loss. These results demonstrate the effectiveness of harness-level self-improvement for long-tail chart adaptation.

## Limitations

The effectiveness of ChartEvo depends partly on the coverage and diversity of the data used for selfevolution. Limited training examples may lead to overly specialized harness components, while conservative validation and admission criteria can restrict the acquisition of new capabilities. Furthermore, substantial distribution shifts between evolution and test data may limit the transferability of evolved Skills, Tools, and SELT. Improving the robustness and generalization of harness-level self-improvement remains an important direction for future work.

## References

Julia Belikova, Rauf Parchiev, Evgeny Egorov, Grigorii Davydenko, Gleb Gusev, Andrey Savchenko, and Maksim Makarenko. 2026. Managing procedural memory in llm agents: Control, adaptation, and evaluation. Preprint, arXiv:2606.23127.

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. 2024. Large language models as tool makers. Preprint, arXiv:2305.17126.

Ritwick Chaudhry, Sumit Shekhar, Utkarsh Gupta, Pranav Maneriker, Prann Bansal, and Ajay Joshi. 2020. Leaf-qa: Locate, encode & attend for figure question answering. In Proceedings ofthe IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV).

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. 2024. Self-play fine-tuning converts weak language models to strong language models. Preprint, arXiv:2401.01335.

DeepSeek-AI. 2026. Deepseek-v4.1-flash: Pushing the limits of kv cache compression. arXiv preprint arXiv:2609.19969.

Mengjie Deng, Guanting Dong, and Zhicheng Dou. 2025. Toolscope: An agentic framework for vision-guided and long-horizon tool use. Preprint, arXiv:2510.27363.

Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, and Hanwang Zhang. 2023. Chartllama: A multimodal llm for chart understanding and generation. Preprint, arXiv:2311.16483.

Ruofan Jin, Zaixi Zhang, Mengdi Wang, and Le Cong. 2025. Stella: Self-evolving llm agent for biomedical research. Preprint, arXiv:2507.02004.

Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. 2018. Dvqa: Understanding data visualizations via question answering. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos Kadar, Adam Trischler, and Yoshua Bengio. 2018. Figureqa: An annotated figure dataset for visual reasoning. Preprint, arXiv:1710.07300.

Rachneet Kaur, Nishan Srishankar, Zhen Zeng, and Sumitra Ganesh. 2026. ChartAgent: A multimodal agent for visually grounded reasoning in complex chart question answering. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18488– 18555. Association for Computational Linguistics.

Jakub Grudzien Kuba, Mengting Gu, Qi Ma, Yuandong Tian, Vijai Mohan, and Jason Chen. 2025. Language self-play for data-free training. Preprint, arXiv:2509.07414.

Kalle Kujanpää, Ning Liu, Shahnawaz Alam, Yeshwanth Reddy Sura, Tianyu Yang, Kristina Klinkner, and Shervin Malmasi. 2026. Tool-making and selfevolving llm agents in low-latency systems. Preprint, arXiv:2607.08010.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. 2026. Metaharness: End-to-end optimization of model harnesses. Preprint, arXiv:2603.28052.

Fangyu Liu, Julian Eisenschlos, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Wenhu Chen, Nigel Collier, and Yasemin Altun. 2023a. DePlot: One-shot visual language reasoning by plot-to-table translation. In Findings of the Associationfor Computational Linguistics: ACL 2023.

Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel Collier, and Julian Eisenschlos. 2023b. MatCha: Enhancing visual language pretraining with math reasoning and chart derendering. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Xingyan Liu, Xiyue Luo, Linyu Li, Ganghong Huang, Jianfeng Liu, and Honglin Qiao. 2026. Skillforge: Forging domain-specific, self-evolving agent skills in cloud technical support. In Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, page 4763–4768. ACM.

Ahmed Masry, Mohammed Saidul Islam, Mahir Ahmed, Aayush Bajaj, Firoz Kabir, Aaryaman Kartha, Md Tahmid Rahman Laskar, Mizanur Rahman, Shadikur Rahman, Mehrad Shahmohammadi, Megh Thakkar, Md Rizwan Parvez, Enamul Hoque, and Shafiq Joty. 2025. ChartQAPro: A more diverse and challenging benchmark for chart question answering. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19123–19151.

Ahmed Masry, Parsa Kavehzadeh, Xuan Long Do, Enamul Hoque, and Shafiq Joty. 2023. UniChart: A universal vision-language pretrained model for chart

comprehension and reasoning. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2022.

Ahmed Masry, Mehrad Shahmohammadi, Md Rizwan Parvez, Enamul Hoque, and Shafiq Joty. 2024. ChartInstruct: Instruction tuning for chart comprehension and reasoning. In Findings of the Associationfor Computational Linguistics: ACL 2024.

Fanqing Meng, Wenqi Shao, Quanfeng Lu, Peng Gao, Kaipeng Zhang, Yu Qiao, and Ping Luo. 2024. Chartassisstant: A universal chart multimodal language model via chart-to-table pre-training and multitask instruction tuning. Preprint, arXiv:2401.02384.

Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and Pratyush Kumar. 2020. Plotqa: Reasoning over scientific plots. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).

OpenAI. 2026. GPT-5.6: Frontier intelligence that scales with your ambition.

Qianjun Pan, Yutao Yang, Junsong Li, Jie Zhou, Kai Chen, Xin Li, Qin Chen, and Liang He. 2026. Anything2skill: Compiling external knowledge into reusable skills for agents. Preprint, arXiv:2606.09316.

Cheng Qian, Chi Han, Yi R. Fung, Yujia Qin, Zhiyuan Liu, and Heng Ji. 2024. Creator: Tool creation for disentangling abstract and concrete reasoning of large language models. Preprint, arXiv:2305.14318.

Qwen Team. 2026a. Qwen3.7: The agent frontier.

Qwen Team. 2026b. Qwen3.8-Max: A new bar for coding and cowork.

Zefeng Wang, Minxi Yan, Jinhe Bi, Sikuan Yan, Volker Tresp, and Yunpu Ma. 2026. Metaskillevolve: Recursive self-improvement of llm agents via two-timescale meta-skill evolution. Preprint, arXiv:2607.05297.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. 2024. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. In Advances in Neural Information Processing Systems, volume 37, pages 113569–113697.

Jingxuan Wei, Nan Xu, Junnan Zhu, Haoyanni, Gaowei Wu, Qi Chen, Bihui Yu, and Lei Wang. 2025. Chart-Mind: A comprehensive benchmark for complex real-world multimodal chart question answering. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing.

Yangbo Wei, Zhen Huang, Shaoqiang Lu, Junhong Qian, Qifan Wang, Chen Wu, and Lei He. 2026. Skillsmith: Co-evolving skills and tools for self-improving agent systems. Preprint, arXiv:2606.01314.

Georg Wölflein, Dyke Ferber, Daniel Truhn, Ognjen Arandjelovic, and Jakob Nikolas Kather.´ 2025. Llm agents making agent tools. Preprint, arXiv:2502.11705.

Renqiu Xia, Bo Zhang, Hancheng Ye, Xiangchao Yan, Qi Liu, Hongbin Zhou, Zijun Chen, Peng Ye, Min Dou, Botian Shi, Junchi Yan, and Yu Qiao. 2025. Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning. Preprint, arXiv:2402.12185.

Zhengzhuo Xu, Sinan Du, Yiyan Qi, Chengjin Xu, Chun Yuan, and Jian Guo. 2024. Chartbench: A benchmark for complex visual reasoning in charts. Preprint, arXiv:2312.15915.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. 2026a. Skillopt: Executive strategy for self-evolving agent skills. Preprint, arXiv:2605.23904.

Ziyi Yang, Weizhou Shen, Chenliang Li, Ruijun Chen, Fanqi Wan, Ming Yan, Xiaojun Quan, and Fei Huang. 2026b. Spell: Self-play reinforcement learning for evolving long-context language models. Preprint, arXiv:2509.23863.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. Preprint, arXiv:2210.03629.

Xunjian Yin, Xinyi Wang, Liangming Pan, Li Lin, Xiaojun Wan, and William Yang Wang. 2025. Gödel agent: A self-referential agent framework for recursively self-improvement. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Hanrong Zhang, Shicheng Fan, Henry Peng Zou, Yankai Chen, Zhenting Wang, Jiayu Zhou, Chengze Li, Wei-Chieh Huang, Yifei Yao, Kening Zheng, Xue (Steve) Liu, Xiaoxiao Li, and Philip S. Yu. 2026. Coevoskills: Self-evolving agent skills via co-evolutionary verification. Preprint, arXiv:2604.01687.

Liang Zhang, Anwen Hu, Haiyang Xu, Ming Yan, Yichen Xu, Qin Jin, Ji Zhang, and Fei Huang. 2024. TinyChart: Efficient chart understanding with program-of-thoughts learning and visual token merging. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing.

## A Chart Question Answering Problem Formulation

Chart Question Answering (CQA) is formulated as a multimodal reasoning task in which a system is given a chart image $\mathcal { T } _ { i }$ and a corresponding natural language question $\mathcal { Q } _ { i } .$ , and is required to produce an answer $\mathcal { A } _ { i } .$ . In this work, we focus on Numerical Question Answering (NQA), where the answer is a numerical value that must be inferred from the visual and quantitative information encoded in the chart. Solving these questions may require identifying relevant chart elements, extracting numerical evidence, and performing quantitative reasoning to derive the final answer.

## B Experimental Details

## B.1 Benchmark Selection and Data Organization

We evaluate ChartEvo on two chart questionanswering benchmarks, ChartBench and ChartX, both of which cover diverse chart types and provide numerical question-answer pairs. We focus on Numerical Question Answering (NQA) and select chart types where the base model exhibits relatively weak performance for subsequent self-evolution and evaluation.

Specifically, we first evaluate Qwen3.7-Plus on the randomly selected 100 questions from train splits of ChartBench and ChartX to obtain per-type NQA accuracy. Chart types with a baseline accuracy below 85% are selected as long-tail chart types. This procedure yields 9 chart types from ChartBench and 8 from ChartX.

For each selected chart type, we use 100 training questions to drive harness-level self-evolution and evaluate the evolved system on the corresponding test questions. Self-evolution is conducted independently for ChartBench and ChartX, with separate evolving harnesses for the two benchmarks. The resulting evaluation sets contain 450 NQA questions across 9 chart types in ChartBench and 265 NQA questions across 8 chart types in ChartX.

## B.2 Self-Evolution Implementation

## B.2.1 Training Organization.

For each long-tail chart type, we partition 100 training questions into 10 clusters of 10 questions each. Self-evolution proceeds sequentially across clusters, with the current harness used to execute the tasks in each cluster. The resulting answers, correctness feedback, and execution trajectories are collected to guide subsequent analysis and evolution. Within each cluster, harness evolution may involve multiple iterations. After each update, the system re-evaluates relevant tasks to assess whether previous failures have been resolved and continues refining the harness to address remaining failures. Once most failures within the cluster have been addressed, the process advances to the next cluster. Verified and accepted harness updates are persisted, allowing subsequent clusters to build upon the accumulated Skills, Tools, and SELT.

## B.2.2 Failure Analysis and Harness Updates

Failure Identification. After completing each training cluster, the system identifies tasks requiring further investigation based on final answers, correctness feedback, and execution trajectories. For NQA tasks, predictions with a relative error exceeding 5% are marked as incorrect, while groundtruth values of zero are handled separately to avoid division by zero. Additionally, predictions with errors between 3% and 5% are flagged for further inspection, as numerically correct answers may still arise from unreliable execution trajectories. Beyond answer correctness, the system also examines fallback behavior and the task coverage of existing Tool-call chains. Together, these signals identify tasks that are not yet reliably handled by the current harness and guide subsequent evolution.

Failure Clustering and Trajectory Analysis. The agentic proposer analyzes the collected execution trajectories to identify recurring failure patterns across training questions. Rather than relying solely on numerical discrepancies, the analysis considers the Agent’s action sequences, Tool invocations, and returned observations to locate the underlying sources of failure. These may include incorrect Tool selection, parameter binding, or evidence utilization despite sufficient existing capabilities; missing Tools for extracting necessary visual or numerical evidence; or ineffective execution order and information dependencies across multiple Tool calls. By clustering failures with shared characteristics, the proposer identifies recurring capability gaps that can guide reusable harness improvements across tasks.

Harness Updates and Verification. Based on the identified failure patterns, the agentic proposer proposes targeted harness updates by revising existing Skills, Tools, and SELT or introducing new components. A single evolution iteration may jointly update multiple component types to coordinate capability acquisition, capability utilization, and execution structure. Candidate updates are evaluated by re-executing relevant training tasks with the modified harness and examining both answer correctness and the resulting execution trajectories. Verification considers whether previous failures have been resolved, whether the Agent correctly invokes the updated components, whether Tool executions produce useful evidence, and whether tasks previously unsupported by harness-based reasoning become solvable. Accepted updates are persisted, while remaining failures may trigger further rounds of analysis and evolution within the same training cluster.

## B.2.3 Harness Growth Through Self-Evolution

We further examine how the ChartEvo harness evolves across ChartBench and ChartX. As shown in Table 7, both benchmarks start from the same initial harness containing four Skills, 25 available Tools, and no SELT. After self-evolution, the number of distinct Skills increases to 13 on ChartBench and 12 on ChartX, while the number of available Tools expands to 42 and 60, respectively. Here, available Tools are counted as the union of Tool names permitted by all Skills in the final harness, rather than the number of candidate Tool versions generated during evolution.

ChartEvo also accumulates 20 active SELT on ChartBench and 71 on ChartX, indicating that self-evolution extends beyond individual Skills and Tools to reusable execution structures. Chart-Bench retains slightly more distinct Skills, whereas ChartX exhibits greater Tool expansion and substantially more active SELT. These results illustrate how ChartEvo progressively accumulates and organizes reusable capabilities within its external harness.

Table 7: Harness Growth Through Self-Evolution. Comparison of harness components before and after self-evolution on ChartBench and ChartX.
<table><tr><td>Component</td><td>Initial</td><td>ChartBench</td><td>ChartX</td></tr><tr><td>Distinct Skills</td><td>4</td><td>13 (+9)</td><td>12 (+8)</td></tr><tr><td>Available Tools</td><td>25</td><td>42(+17)</td><td>60 (+35)</td></tr><tr><td>Active SELT</td><td>0</td><td>20</td><td>71</td></tr></table>

## B.2.4 Illustrative Self-Evolution Example

Figure 5 illustrates how ChartEvo acquires reusable harness capabilities through self-evolution using a dot-ring chart question asking for the approximate percentage of Spotify, with a reference answer of 40%. In the initial execution, the Agent first invokes annotate\_legend to identify the target series, followed by get\_marker\_rgb to extract its corresponding color. It then applies measure\_pie\_segment\_percentage to estimate the target proportion from pixel area, obtaining 33.68%. Although the target series and its color are correctly identified and the measurement Tool returns an accepted status, the final prediction deviates from the reference answer. Trajectory analysis reveals a mismatch between the measurement strategy and the chart’s visual encoding: proportions are represented by discrete dot counts rather than pixel area.

To address this failure, ChartEvo jointly evolves multiple harness components. A new Tool, count\_ring\_dots\_and\_compute\_percentage, is introduced to count target-colored dots and total dots within the same ring and compute their ratio. A corresponding Skill, ring-dot-percentage, provides capability selection and usage guidance for dot-ring charts and approximate percentage questions. Meanwhile, the existing legend annotation and color extraction Tools are composed with the new counting Tool into a three-step SELT, preserving the complete measurement procedure as a reusable execution structure. These coordinated updates extend the harness with a measurement capability tailored to discrete-dot encoding, together with the knowledge and execution structure required to utilize it.

When executing the original question with the evolved harness, the Agent follows the three-step Tool chain to identify the target series, extract its color, and count the corresponding dots. The new Tool identifies 331 target dots out of 830 total dots, yielding 39.8795%, which is rounded to 40% in accordance with the question’s approximate wording. The same persisted counting capability is further reused on three additional dot-ring charts involving smartphone operating systems, TV genres, and TV streaming, producing answers of 35%, 30%, and 40%, respectively. This example illustrates how ChartEvo transforms an observed measurement failure into a persistent harness capability that can be applied across subsequent chart instances.

## B.3 Ablation Study Configurations

All ablation settings use Qwen3.7-Plus as the base model and are evaluated on the same test questions under a unified 5% relative-error tolerance. Specifically, Base MLLM performs direct visual answering without any harness components. Evolved Skill-only uses evolved Skills to guide reasoning without access to Tools or SELT. Base Harness includes the initial Skills and Tools, without SELT. Evolved Tools retains the initial Skills while replacing the initial Tool collection with its evolved counterpart, again without SELT. Finally, ChartEvo (Full) employs the complete evolved harness, comprising evolved Skills, the same evolved Tools, and persisted SELT.

![](images/5389b37ad6238cf40aff5931d5883defcfe3b28393df95b67c54191dff41ba41.jpg)  
Figure 5: Illustrative self-evolution of ChartEvo on a dot-ring chart. An initial measurement failure triggers joint Skill, Tool, and SELT evolution, enabling accurate dot-count-based reasoning and capability reuse across additional chart instances.

## C Harness Specifications

## C.1 Harness Representation and Organization

ChartEvo organizes the capabilities acquired through self-evolution into a persistent harness comprising Skills, Tools, and Self-Evolving Local Toolchains (SELT). Each component adopts a representation suited to its role. Skills are stored as metadata-annotated textual knowledge, specifying applicability conditions, reasoning guidance, and knowledge for capability selection and utilization. Tools are registered as executable components with explicit input and output interfaces for visual and numerical processing. SELT preserve validated Tool-call compositions as reusable execution structures with defined functional contracts, execution order, and internal dependencies. Together, these components form an extensible collection of capabilities accessible to the Agent during reasoning.

Harness components are connected through explicit references that coordinate knowledge, executable capabilities, and execution structures. Skills provide reasoning guidance for particular chart structures and question types, together with the selection and usage knowledge of relevant Tools and the registration and routing information of SELT. Guided by these Skills, the Agent can invoke individual Tools, dynamically compose multiple Tool calls, or reuse persisted SELT. Each SELT specifies its constituent Tool calls, execution order, and parameter dependencies across steps, enabling multiple executable capabilities to operate within a predefined structure. These cross-component references allow reasoning knowledge, executable capabilities, and execution structures to be organized and utilized together.

Verified and accepted harness updates are persisted in the corresponding asset stores and remain available throughout subsequent training stages and inference tasks. Existing components may be revised to accommodate new capability requirements, while new Skills, Tools, or SELT can be introduced to extend the range of supported tasks and execution patterns. When an evolution iteration involves multiple component types, their associated guidance, executable capabilities, and execution structures can be updated jointly and retained as interconnected assets. This persistent organization enables the harness to accumulate reusable capabilities across successive evolution iterations and supports subsequent reasoning with the expanded capability repertoire.

## C.2 Hierarchical Retrieval and Progressive Disclosure

As the harness evolves, its growing collection of Skills, Tools, and SELT introduces an increasing amount of potentially relevant knowledge and executable capabilities. ChartEvo employs hierarchical retrieval and progressive disclosure to construct a task-relevant execution context, selectively exposing harness information according to task characteristics and evolving execution states.

ChartEvo adopts a two-stage retrieval mechanism to select relevant Skills without loading the entire Skill Store. In the first stage, Skill frontmatter metadata is used to filter and rank candidates based on chart type, question characteristics, and applicability conditions. Only the selected Skills are subsequently loaded for reasoning. For each question, the system retrieves at most three Skills, comprising up to one universal Skill and two chartspecific Skills, with a combined Skill-body budget of 1,600 tokens. Near-duplicate Skills with similarity exceeding 0.80 are suppressed, while chart- or feature-specific matches are prioritized over overly general or mismatched candidates. This mechanism limits the knowledge introduced into the Agent’s context while preserving guidance relevant to the current reasoning task.

Beyond initial retrieval, ChartEvo progressively discloses Skill content according to observed execution states. Each Skill may contain general routing guidance alongside conditional instructions associated with different Tool outcomes. At the beginning of a task, the Runtime exposes only the initial routing guidance rather than all possible execution and recovery branches. As reasoning proceeds, Tool outcomes such as accepted, rejected, and argument\_error trigger the disclosure of corresponding Skill branches. Instructions associated with unobserved conditions remain hidden. Consequently, a Skill can encode normal execution, error handling, and recovery guidance while exposing only the instructions relevant to the current stage of reasoning.

ChartEvo organizes executable capabilities through a two-level Tool Catalog. The overview level provides the names and concise functional descriptions of all registered Tools, allowing the Agent to discover available capabilities. The detail level selectively exposes complete interfaces for a small set of relevant Tools, including input schemas, output fields, invocation constraints, and usage examples. During initial context construction, at most six detailed Tool interfaces are disclosed; after context refresh, this limit is reduced to four. Candidate selection incorporates signals from the retrieved Skills, Tools referenced by relevant SELT, question semantics, recently invoked Tools, and Tools associated with preceding execution failures. This separation preserves broad capability discoverability while limiting the context occupied by detailed executable interfaces.

The set of available SELT is further restricted by the retrieved Skills and task-specific applicability conditions. The Runtime first collects SELT references from the selected Skills and then filters candidates according to chart metadata, question semantics, and parameter bindability. An eligible SELT must satisfy its chart applicability conditions, match relevant question terms, and admit the binding of required selectors from the question, legend, or axis metadata. Candidates with unresolved references, ambiguous identifiers, or unsatisfied execution prerequisites are excluded from the available SELT catalog. The remaining SELT are presented to the Agent as reusable execution structures, from which the Agent may select an appropriate composition for the current task.

Together, these mechanisms enable ChartEvo to organize an expanding harness into a compact, task-relevant reasoning context. Skill knowledge, executable Tool interfaces, and reusable SELT are disclosed selectively, with subsequent guidance adapting to observed execution states.

## C.3 Harness Execution and Runtime Contracts

Following harness retrieval and disclosure, the Agent selects individual Tools, reuses existing SELT, or dynamically composes Tool calls based on the current question, chart information, and accumulated observations. For individual Tools, the

Agent constructs invocation arguments using relevant chart elements, data series, axis information, and available evidence. For SELT, the Agent determines whether the functional contract matches the current task and binds the required external parameters, while the internal execution order and parameter dependencies are predefined by the persisted composition. This separation allows the Agent to retain control over semantic decisions and capability selection while delegating predefined execution procedures to the Runtime.

The Runtime executes the selected harness capabilities and validates their outputs against the corresponding contracts. For individual Tools, it performs the requested operations using the Agentprovided arguments and returns structured visual or numerical evidence. For SELT, the Runtime executes the constituent Tool calls sequentially according to the predefined composition, resolves parameter dependencies across steps, and aggregates the resulting observations. These internal Tool calls proceed without requiring a separate Agent decision at each step, reducing repeated MLLM– Tool interactions. Both individual Tools and SELT are subject to contract validation, including input/output interface constraints and componentspecific functional requirements. Successful execution alone does not guarantee that the resulting evidence is accepted; the outputs must also satisfy the corresponding validity conditions before being used in subsequent reasoning. This contractaware execution mechanism provides the Agent with structured evidence accompanied by explicit acceptance states.

After execution, the Runtime returns the resulting observations and execution states to the Agent. Outputs satisfying the applicable contracts are marked as accepted, allowing the Agent to incorporate the evidence into subsequent numerical reasoning or Tool invocations. Outputs that fail the relevant acceptance conditions are marked as rejected, while invalid invocation arguments produce states such as argument\_error. Based on this feedback, the Agent may revise arguments, select alternative Tools, reorganize subsequent execution steps, or proceed with the available evidence. These execution states also inform the conditional Skill disclosure mechanism described in Section C.2, providing additional guidance aligned with the observed execution outcomes. Through this interaction, semantic reasoning and contractaware execution remain coordinated throughout multi-step chart reasoning.

## C.4 Initial Harness Configuration

To clarify the initial capability boundary of ChartEvo, we describe the harness configuration used before self-evolution. Both ChartBench and ChartX start from the same initial harness and undergo independent self-evolution. The initial harness consists of 4 Skills and 25 generic Tools, providing foundational capabilities for chart understanding, visual evidence extraction, and numerical processing. No SELT are predefined in the initial harness. The complete inventories of initial Skills and Tools, together with their functional descriptions, are provided in Tables 8 and 9.

## D Case Study

We present qualitative analyses of ChartEvo’s inference trajectories to examine how the evolved harness supports numerical chart reasoning in practice. We organize the cases into three categories: successful reasoning, error recovery, and unsuccessful reasoning. Together, they illustrate the utilization of structured evidence, adaptation to intermediate execution feedback, and remaining limitations in challenging chart scenarios.

## D.1 Successful Reasoning Cases

We first examine successful reasoning trajectories to illustrate how ChartEvo utilizes its evolved harness to solve challenging numerical questions. These cases highlight the coordination of multiple executable capabilities, including target identification, visual evidence extraction, and numerical calibration, and how their intermediate observations contribute to the final prediction.

Figure 6 presents an example involving a 3D grouped bar chart. ChartEvo identifies the target category and series, calibrates the numerical axis, and integrates geometric observations with axis anchors to estimate the target value. The resulting prediction of 7.9527 closely matches the ground truth of 7.9, illustrating the effectiveness of structured evidence integration for numerical reasoning over complex chart geometry.

## D.2 Error Recovery Cases

We further investigate reasoning trajectories in which ChartEvo encounters intermediate execution errors but ultimately reaches correct predictions. These cases highlight how the Agent interprets Tool feedback, revises its actions, and recovers from unsuccessful intermediate steps.

Table 8: Initial Skill inventory of ChartEvo before self-evolution, summarizing the reasoning and capability-use guidance provided by each Skill.
<table><tr><td>Skill</td><td>Reasoning and Capability-Use Guidance</td></tr><tr><td>chart-evidence-validation</td><td>Reuse validated observations, distinguish rejected from advisory evidence, preserve numerical semantics, and avoid ambiguous bindings or invalid observation references.</td></tr><tr><td>bar-value</td><td>Guide legend and category binding, axis calibration, bar localization, and numerical projection, with handling of stacked segments and negative bars.</td></tr><tr><td>line-value</td><td>Guide series localization, point extraction, and axis projection, while addressing error-bar anchors, ambiguous candidates, and coordinate consistency.</td></tr><tr><td>pie-percentage</td><td>Guide color and label binding, segment measurement, and percentage estimation using geometric and area-based evidence.</td></tr></table>

Figure 7 illustrates two forms of recovery within a single treemap reasoning trajectory. The Agent initially attempts rectangle-based segmentation, which fails due to overlapping regions. It then switches to mask-based segmentation, successfully identifying the target category and measuring its area. A subsequent attempt to measure the total area triggers an argument error because an incompatible parameter is supplied. After removing the invalid argument, the Agent obtains the total area and computes a percentage of 10.019%, closely matching the ground truth of 10%. This case demonstrates how execution feedback supports both strategy-level adaptation and parameterlevel correction without requiring harness updates.

## D.3 Failure Cases

Finally, we examine unsuccessful reasoning trajectories to characterize the remaining limitations of ChartEvo. These cases investigate situations where the available harness capabilities or intermediate observations are insufficient for reliable numerical inference, as well as cases where the Agent fails to appropriately handle uncertain evidence.

Figure 8 illustrates a failure on a 3D stacked bar chart. Although the target series and category are correctly identified, the thin target segment introduces substantial uncertainty in endpoint localization. The measurement Tool consequently returns an estimated interval rather than an accepted precise value. However, the Agent adopts the interval midpoint, producing 22.2648 million against the ground truth of 20.1 million. This case highlights a remaining limitation in reasoning with uncertain numerical evidence.

Table 9: Initial Tool inventory of ChartEvo before self-evolution. The 25 generic Tools are grouped by their primary functionalities.
<table><tr><td>Tool</td><td>Functionality</td></tr><tr><td colspan="2">OCR, Text, Color, and Segmentation (7)</td></tr><tr><td>locate_text</td><td>Locate visible text labels and return their bounding boxes and positions.</td></tr><tr><td>annotate_legend</td><td>Identify specified legend labels and construct their spatial mappings.</td></tr><tr><td>get_marker_rgb</td><td>Extract the RGB color of a marker associated with a legend label.</td></tr><tr><td>get_embedded_label_rgb</td><td>Extract the underlying fill color associated with an embedded data label.</td></tr><tr><td>clean_chart_image</td><td>Remove detected textual regions to produce a cleaned chart image.</td></tr><tr><td>segment_and_mark resolve_text_to_mask</td><td>Generate candidate segmentation masks and indexed visual overlays.</td></tr><tr><td></td><td>Associate a target text label with a candidate segmentation mask.</td></tr><tr><td colspan="2">Axis, Coordinates, and Deterministic Computation (5)</td></tr><tr><td>axis_tick_localizer</td><td>Locate specified axis tick labels and their pixel positions.</td></tr><tr><td>axis_localizer</td><td>Estimate numerical axis calibration from detected tick information.</td></tr><tr><td>interpolate_pixel_to_value</td><td>Convert pixel coordinates into numerical values using reference anchors.</td></tr><tr><td>project_point_to_axis_value</td><td>Project a selected point onto a calibrated numerical axis.</td></tr><tr><td>arithmetic</td><td>Perform deterministic arithmetic operations on supplied operands.</td></tr><tr><td colspan="2">Pie and Area Measurement (3)</td></tr><tr><td>measure_pie_segment_percentage</td><td>Estimate pie or donut segment proportions from color-based area measurements.</td></tr><tr><td>compute_segment_area</td><td>Measure selected region areas and compute proportions with a specified denominator.</td></tr><tr><td>measure_mask_angular_percentage</td><td>Estimate angular proportions of resolved radial segmentation masks.</td></tr><tr><td colspan="2">Bar and Boxplot (4)</td></tr><tr><td>get_bar</td><td>Locate axis-aligned bars and extract their bounding boxes.</td></tr><tr><td>compute_bar_height</td><td>Convert detected bar geometry into numerical values using axis anchors.</td></tr><tr><td>get_boxplot</td><td>Extract geometric landmarks of axis-aligned boxplots.</td></tr><tr><td>compute_boxplot_entity</td><td>Calculate boxplot statistics from extracted landmarks and axis references.</td></tr><tr><td colspan="2">Line, Area, and Scatter (3)</td></tr><tr><td>get_edgepoints</td><td>Extract local edge points associated with a target color.</td></tr><tr><td>locate_series_point</td><td>Locate candidate series points near a specified horizontal position.</td></tr><tr><td>locate_color_extent</td><td>Measure color extents along a specified scanline.</td></tr><tr><td colspan="2">Radial Geometry (3)</td></tr><tr><td>get_radial</td><td>Extract radial contours, centers, and outer radii</td></tr><tr><td>analyze_radial_geometry</td><td>Analyze radial geometry and estimate aggregate radius references.</td></tr><tr><td>estimate_radial_value</td><td>Estimate numerical values from radii and supplied reference values.</td></tr></table>

![](images/04cf1cf111d7db3c7f32c5c285b7cf431d65b4ca95f875bf2c15b346f2ce840f.jpg)  
Figure 6: Successful reasoning on a 3D grouped bar chart. ChartEvo coordinates multiple Tool calls to extract structured evidence and accurately estimate the target value.

![](images/ed0705e70301ea9a9e6262eeb30da0e839d213fa72fece8ce1ee2a82200d2b9a.jpg)  
Figure 7: Error recovery on a treemap. ChartEvo switches segmentation strategies after an initial failure and corrects an invalid Tool argument, ultimately producing an accurate percentage estimate.

![](images/729c0f98328024d0745c7c88a04173c8656e907a0f9c0c8b3faefa6c4da6b143.jpg)  
Figure 8: Failure case on a 3D stacked bar chart. Uncertain segment boundaries prevent precise measurement, while the Agent adopts an unreliable estimate, leading to an incorrect answer.