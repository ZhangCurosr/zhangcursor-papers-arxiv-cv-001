# DeltaStreamer: Towards Human-Like Agentic Streaming Video Understanding

Boyuan Sun<sup>1</sup>, Deshui Miao<sup>2</sup>, Wenzhao Gao<sup>2</sup>, Shaoyong Jia<sup>2</sup>, Shaohui Jiao<sup>2</sup>, Chi Liu<sup>3</sup>, Yan Shu<sup>3</sup>, Derek Li<sup>3</sup>, Bryan Dai<sup>3</sup>, Shengsheng Qian<sup>4</sup>, Qibin Hou<sup>1</sup>

<sup>1</sup>VCIP, Nankai University <sup>2</sup>ByteDance Inc. <sup>3</sup>IQuest Research <sup>4</sup>Institute of Automation, Chinese Academy of Sciences

Abstract: We propose DeltaStreamer, a novel agentic framework for streaming video understanding inspired by how humans observe, remember, and investigate. Streaming video understanding requires a system to retain evidence before knowing which details future queries will need. Existing approaches often obscure event transitions and omit relevant details when compressing video history, and cannot recover missing visual evidence. DeltaStreamer addresses these limitations by combining changeaware memory with query-driven planning and targeted evidence gathering with replay. Before a query arrives, it continuously observes the stream and explicitly records visual changes in structured memory, preserving how events unfold rather than only what is present. When a query arrives, a query-driven planner identifies the required evidence and guides memory retrieval and targeted replay. By revisiting historical frames, the agent can recover visual details omitted during observation and ground its answer in evidence beyond the retained descriptions. This connects continuous observation with active investigation tailored to each question. Experiments on OVO-Bench and StreamingBench demonstrate improvements over representative streaming video understanding methods, with overall scores of 66.07% and 59.53%, respectively.

Project page: https://bbbbchan.github.io/DeltaStreamer/

HVision@Nankai

## 1 Introduction

Streaming video is prevalent in real-world applications such as live surveillance [13], robotics [4, 44], and autonomous systems [54, 43], where visual input arrives continuously rather than as a complete clip. Unlike offline video, where the entire sequence is available for analysis, streaming video arrives frame by frame. The system can access only what has already been observed and must operate continuously without seeing future frames [18]. Questions may also arrive after the relevant events [82]. The system must therefore build and maintain memory without knowing which details will later be needed.

Existing offline video models [78, 46] struggle to adapt to the streaming setting, which requires continuous perception, incremental memory updates, and on-demand responses. Recent work has begun to address these challenges using token compression [5], KV-cache reuse [21, 8], and multi-turn dialogue [59, 51]. Despite the progress, these directions share two limitations in streaming scenes. First, their records primarily capture what is present while leaving what is changing implicit. Consecutive frames often contain similar scene content [80], so repeated representations can consume the available budget and obscure the transitions that distinguish events. Second, once a frame has passed, the evidence is limited to what is retained from the visual signals [70]. Without a mechanism to revisit the original frame, details omitted during observation cannot be recovered when a later question makes them relevant. Both limitations reflect that existing paradigms focus on managing the token budget, with insufficient attention to recording changes and recovering missing evidence. This motivates us to investigate a more natural and intuitive way for streaming video understanding.

We argue that the system should watch frame by frame, just as a human does, continuously observing, building and updating a structured memory. As frames arrive sequentially [53, 15] and queries arrive independently, the system needs to process visual information and update memory asynchronously [30, 29]. Before a query arrives, it should retain broad visual evidence without knowing which details will later matter. When a question arrives, it should actively probe its memory, assess whether the retrieved evidence is sufficient, and revisit historical frames to recover missing pixel-level details for a well-grounded answer. We should view streaming video understanding as an active information-gathering [83, 57] problem rather than a passive encoding task, calling for an agent that extends a VLM with planning and tool use [52]. Recent efforts such as StreamAgent [61] and EventMemAgent [58] have explored this agentic direction, but they share the same limitation that stored representations may omit evidence needed for later queries.

![](images/d1be822423e1035e8ae9363407ce8b941c3bf69f321856c2507bbb03e9782f96.jpg)  
Fig. 1 Comparison of streaming video understanding paradigms. Compression or truncation in KV-cache reuse and multi-turn dialogue can obscure transitions and discard details needed by later queries. DeltaStreamer records explicit changes in structured memory and uses query-driven planning and targeted replay to recover missing visual evidence.

In this paper, we build DeltaStreamer, a human-like agentic streaming video understanding system that continuously builds a hierarchical textual memory from incoming frames and actively gathers evidence in response to user queries, just as human viewers follow a video as it unfolds, remember what happened, and revisit earlier footage when memory falls short. Considering that consecutive frames may yield similar scene descriptions even when important changes occur, the system compares adjacent frames during observation and explicitly record their visual changes as deltas alongside scene descriptions. These observations are organized into a hierarchica textual memory, with source links connecting summaries to their supporting observations and video frames. When a query arrives, a query-driven planner identifies the required evidence and whether it should be sought in past, current, or incoming frames. The resulting plan remains available throughout subsequent reasoning, guiding memory retrieval, continued observation, and tool use. When retrieved descriptions lack a required visual detail, the agent can follow the source links to locate relevant historical frames and replay them to recover pixel-level evidence. Unlike previous works that mostly focus on compressing and retaining video history, we couple explicit change recording with persistent query-driven planning and targeted replay. This allows the agent system to reason over event transitions and recover query-relevant visual details omitted from memory. We evaluate DeltaStreamer on OVO-Bench [33] and StreamingBench [28], where it outperforms representative methods from existing streaming video understanding paradigms.

## In summary, our contributions are:

• We propose DeltaStreamer, an agentic streaming video understanding system inspired by human observation and active investigation. It continuously builds structured memory and, after query arrival, uses planning and tool use to gather evidence until it can answer.

• We design a delta-augmented hierarchical textual memory that explicitly records inter-frame changes alongside scene descriptions. A three-layer DAG with bidirectional cross-layer pointers and an entity index makes these records traceable across temporal scales.

• We introduce a query-driven planner that performs structured analysis of the query’s needs and produces a persistent plan to guide memory retrieval, continued observation, and tool use. Targeted replay recovers pixel-level visual details missing from textual memory.

## 2 Related Work

Streaming Video Representations and Memory. Streaming video understanding requires continuous processing and selective retention of historical evidence. Existing approaches support these requirements through token compression [25, 45, 55, 19], KV memory [47, 2, 42, 77], and multi-turn dialogue [65, 60, 6]. TimeChat-Online [67] removes redundant tokens using inter-frame differences, ReKV [11] retrieves historical KV states, and StreamMem [64] maintains compressed, query-agnostic KV memory. Streamo [59] develops instruction tuning for streaming interaction. Other approaches explicitly organize history for later queries. StreamForest [68] constructs an event-level memory forest, WAT [17] separates query-independent watching from query-triggered thinking, and Vista [31] recalls full-resolution historical evidence from compressed scene representations. Together, these methods establish mechanisms for retaining and accessing video history. Some works also begin to enhance understanding quality from an omni-modal perspective [7, 56, 79, 62]. DeltaStreamer focuses on making semantic changes explicit and traceable. It directly compares adjacent sampled frames and stores the resulting deltas alongside scene descriptions in a hierarchical memory with source links and an entity index, connecting recorded transitions to supporting visual evidence.

Agentic Evidence Gathering. Streaming agents [81, 22, 69] extend memory retrieval with planning and tool use. StreamAgent [61] explores anticipatory planning for online interaction. EventMemAgent [58] combines event memory, visual anchors, and inter-event change logs with learned memory search and visual tool use. OASIS [26] uses an adaptive coarse-to-fine process that generates retrieval intents and retrieves event memory with keyframes. Different from them, DeltaStreamer couples direct inter-frame change recording with a structured plan specifying the query target, temporal anchors, and required evidence. The plan remains available to guide subsequent retrieval and tool rounds, including targeted historical-frame replay when stored observations lack decisive details.

## 3 DeltaStreamer

## 3.1 Problem Formulation and Overview

Given a streaming video $\mathcal { V } = \{ I _ { 0 } , \ldots , I _ { T - 1 } \}$ contain T frames, where $I _ { t }$ is the frame at index t. A question q arrives at index $t _ { q } ,$ which need not coincide with the time when sufficient evidence becomes available. Before $t _ { q } ,$ , the system observes the stream without access to $q .$ After query arrival, it may answer using the evidence already available or continue observing incoming frames until it can respond. At any current index $t \geq t _ { q } ,$ the accessible video is $\mathcal { V } _ { \le t } = \{ I _ { 0 } , \dots , I _ { t } \}$ , including both pre-query frames and any frames received since query arrival, while frames beyond t remain inaccessible. To address this setting, DeltaStreamer maintains textual memory as a hierarchical graph $\mathcal { G } _ { t }$ through agentic tool use. During pre-query observation, it stores structured observations $O _ { t } = ( D _ { t } , \Delta _ { t } )$ , where $D _ { t }$ describes the current scene and $\Delta _ { t }$ records changes between adjacent frames. At query arrival, a planner generates a plan $P$ to guide memory retrieval, continued observation, and targeted replay until the agent answers. The agent can inspect the current scene, continue observing, or use a targeted replay mechanism to recover missing visual details. Fig. 2 provides an overview of DeltaStreamer, including delta-augmented observation, traceable memory, query-driven planning, and replay.

## 3.2 Delta-Augmented Observation

Before query arrival, a vision-language model (VLM) jointly describes the current scene and its changes from the preceding sampled frame. The delta is a textual description of these changes, such as an object appearing or changing state.

$$
O _ { t } = ( D _ { t } , \Delta _ { t } ) = { \bf V } { \bf L } { \bf M } ( I _ { t - 1 } , I _ { t } , M _ { t - 1 } ^ { \mathrm { s n a p s h o t } } ) .\tag{1}
$$

Here, t indexes sampled frames, $I _ { t - 1 }$ and $I _ { t }$ are the preceding and current frames, and $M _ { t - 1 } ^ { \mathrm { s n a p s h o t } }$ is the memory text available before processing $I _ { t }$ . The observation $O _ { t }$ pairs a structured scene description $D _ { t }$ (description, actors, objects, actions, OCR, and scene) with textual changes $\Delta _ { t } . \ \mathrm { A t } \ t = 0$ , the model receives only $I _ { 0 }$ and emits an initial-frame marker.

To identify repetitive descriptions, we measure the textual novelty of each nonempty observation against recent

![](images/1474e58c2a3b83f898de21e7eae503f709888c07010e249e1134c4e7b95259e5.jpg)  
Fig. 2 Overview of DeltaStreamer. Before the query, adjacent-frame comparison produces delta-augmented observations stored in hierarchical memory. After the query, the planner guides evidence retrieval and tool use, while targeted replay recovers visual details for verification.

history,

$$
s ( O _ { t } ) = \frac { \sum _ { f } w _ { f } \left[ 1 - \operatorname* { m a x } _ { i } \mathrm { J a c c a r d } _ { f } ( O _ { t } , O _ { i } ) \right] } { \sum _ { f } w _ { f } } .\tag{2}
$$

Here, $s ( O _ { t } )$ measures the textual novelty of observation $O _ { t }$ at sampled frame t. The index $f$ ranges over fields with nonempty token sets in $O _ { t } .$ , and $w _ { f }$ is the positive weight assigned to field $f .$ The index i ranges over recent prior observations $O _ { i }$ . J $\operatorname { a c c a r d } _ { f } ( O _ { t } , O _ { i } )$ measures the intersection over union of the lowercase token sets extracted from field $f$ in the two observations. For each field, we subtract its highest historical similarity from one and then average these novelty scores using the field weights. Higher scores indicate less repetitive content. Novelty and repeated-delta checks assign salience weights for retrieval ranking and change flags for episodic aggregation. Low-salience records remain stored. The field weights and scoring procedure are specified in Appendix A.1.

## 3.3 Traceable Hierarchical Memory

We formulate structured memory for long-running video streams as a hierarchical directed acyclic graph (DAG), with source pointers linking stored records to observations and video frames. Working memory stores working observations W, episodic memory E summarizes related observations, and long-term pins $\mathcal { L }$ retain compact summaries linked to source episodes.

The memory DAG links W, E, and L through source identifiers and reverse pointers. After query arrival, retrieval and construction of the model’s visible memory snapshot are expressed as:

$$
\begin{array} { r l } & { \quad \mathcal { R } _ { t } ( q ) = \mathrm { B a c k t r a c k } ( \mathcal { G } _ { t } , q ) , } \\ & { \quad M _ { t } ^ { \mathrm { s n a p s h o t } } = \mathrm { R e n d e r } \left( \mathcal { L } _ { t } , \mathcal { E } _ { t } , \mathcal { W } _ { t } ^ { \mathrm { r e c e n t } } , \mathrm { R e c a l l } ( q , \mathcal { W } _ { t } ^ { \mathrm { m i d d l e } } ) \right) . } \end{array}\tag{3}
$$

Here, t indexes the current sampled frame, q is the question, and $\mathcal { G } _ { t }$ is the memory graph. Backtrack selects records by keyword overlap and salience, then follows source pointers to form evidence paths $\textstyle { \mathcal { R } } _ { t } ( q )$ . Render assembles snapshot $M _ { t } ^ { \mathrm { s n a p s h o t } }$ from long-term pins $\mathcal { L } _ { t }$ , episodes $\mathcal { E } _ { t }$ , recent working observations $\mathcal { W } _ { t } ^ { \mathrm { r e c e n t } }$ , and question-relevant older observations selected by Recall from $\mathcal { W } _ { t } ^ { \mathrm { m i d d l e } }$

The snapshot is the memory text supplied to each model call. Rendering trims older verbatim entries toward a text-budget target while retaining summaries and recent observations. Omitted records remain stored and retrievable. Before query arrival, recall can instead use recent observations as its retrieval cue. An entity index $\mathcal { T } _ { t }$ maps actor and object names to their observation times. We describe memory construction and snapshot rendering in Appendix A.2.

Tab. 1 The experimental results on the OVO-Bench benchmark. RT+BT Avg. is the average over the Real-Time Visual Perception and Backward Tracing groups.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=7>Real-Time Visual PerceptionOCRACRATRSTU FPD OJR Avg.</td><td rowspan=1 colspan=4>Backward TracingEPMASI HLDAvg.</td><td rowspan=1 colspan=1>RT+BTAvg.</td><td rowspan=1 colspan=4>Forward Active RespondingRECSSRCRR Avg.</td><td rowspan=1 colspan=1>Overall</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=7>Hu</td><td rowspan=1 colspan=4>man</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Human Agents</td><td rowspan=1 colspan=7>93.9692.5794.8392.7091.0994.0293.20</td><td rowspan=1 colspan=4>92.5993.0291.3792.33</td><td rowspan=1 colspan=1>92.77</td><td rowspan=1 colspan=4>95.4889.6793.5692.90</td><td rowspan=1 colspan=1>92.81</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=7>Proprietary</td><td rowspan=1 colspan=4>MLLMs</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>Gemini 1.5 ProGemini 3 FlashGPT-40</td><td rowspan=3 colspan=7>85.9166.9779.3158.4363.3761.9669.3286.5868.8179.3158.9970.3070.6572.4469.8064.2271.5551.1270.3059.7864.46</td><td rowspan=1 colspan=4>58.5976.3552.6462.54</td><td rowspan=1 colspan=1>65.93</td><td rowspan=1 colspan=4>35.5374.2461.6757.15</td><td rowspan=1 colspan=1>63.00</td></tr><tr><td rowspan=1 colspan=4>58.9276.3552.1562.47</td><td rowspan=1 colspan=1>67.46</td><td rowspan=1 colspan=4>35.5374.2461.6757.15</td><td rowspan=1 colspan=1>64.02</td></tr><tr><td rowspan=1 colspan=4>57.9175.6848.6660.75</td><td rowspan=1 colspan=1>62.61</td><td rowspan=1 colspan=4>27.5873.2159.4053.40</td><td rowspan=1 colspan=1>59.54</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=7>Open-Source</td><td rowspan=1 colspan=4>Offline MLLMs</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Qwen2-VL-72BLLaVA-Video-7B</td><td rowspan=1 colspan=7>65.7760.5569.8351.6969.3154.3561.92</td><td rowspan=1 colspan=4>52.5360.8157.5356.95</td><td rowspan=1 colspan=1>59.44</td><td rowspan=1 colspan=4>38.8364.0745.0049.30</td><td rowspan=1 colspan=1>56.27</td></tr><tr><td rowspan=1 colspan=7>69.1358.7268.8349.4474.2659.7863.52</td><td rowspan=1 colspan=4>56.2357.437.5340.40</td><td rowspan=1 colspan=1>51.96</td><td rowspan=1 colspan=4>34.1069.9560.4254.82</td><td rowspan=1 colspan=1>52.91</td></tr><tr><td rowspan=1 colspan=1>LLaVA-OneVision-7B</td><td rowspan=1 colspan=7>66.4457.8073.2853.3771.2961.9664.02</td><td rowspan=1 colspan=4>54.2155.4121.5143.71</td><td rowspan=1 colspan=1>53.87</td><td rowspan=1 colspan=4>25.6467.0958.7550.50</td><td rowspan=1 colspan=1>52.74</td></tr><tr><td rowspan=1 colspan=1>Qwen2-VL-7B</td><td rowspan=1 colspan=7>60.4050.4656.0347.1966.3455.4355.98</td><td rowspan=1 colspan=4>47.8135.4856.0846.46</td><td rowspan=1 colspan=1>51.22</td><td rowspan=1 colspan=4>31.6665.8248.7548.74</td><td rowspan=1 colspan=1>50.39</td></tr><tr><td rowspan=1 colspan=1>InternVL-V2-8B</td><td rowspan=1 colspan=5>67.1160.5563.7946.0768.32</td><td rowspan=1 colspan=2>56.5260.39</td><td rowspan=1 colspan=2>48.1557.43</td><td rowspan=1 colspan=2>24.7343.44</td><td rowspan=1 colspan=1>51.92</td><td rowspan=1 colspan=4>26.5059.1454.1446.60</td><td rowspan=1 colspan=1>50.15</td></tr><tr><td rowspan=1 colspan=1>LongVU-7B</td><td rowspan=1 colspan=3>53.6953.2162.93</td><td rowspan=1 colspan=1>47.75</td><td rowspan=1 colspan=1>68.32</td><td rowspan=1 colspan=2>59.7857.61</td><td rowspan=1 colspan=2>40.7459.46</td><td rowspan=1 colspan=2>4.8435.01</td><td rowspan=1 colspan=1>46.31</td><td rowspan=1 colspan=4>12.1869.4860.8347.50</td><td rowspan=1 colspan=1>46.71</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=3>75.1758.7272.41</td><td rowspan=1 colspan=1>57.30</td><td rowspan=1 colspan=1>70.305</td><td rowspan=1 colspan=2>9.2465.52</td><td rowspan=1 colspan=4>56.5769.5912.3746.18</td><td rowspan=1 colspan=1>55.85</td><td rowspan=1 colspan=4>47.1367.5752.5055.73</td><td rowspan=1 colspan=1>55.81</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>Limi</td><td rowspan=1 colspan=2>ted Open-Sou</td><td rowspan=1 colspan=4>rce Online MLLMs</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Stream-IT</td><td rowspan=1 colspan=4>84.5671.5674.1449.44</td><td rowspan=1 colspan=1>75.25</td><td rowspan=1 colspan=1>72.83</td><td rowspan=1 colspan=1>71.30</td><td rowspan=1 colspan=2>一     一</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>HERMES</td><td rowspan=1 colspan=3>85.2064.2071.60</td><td rowspan=1 colspan=1>53.40</td><td rowspan=1 colspan=1>74.30</td><td rowspan=1 colspan=1>65.20</td><td rowspan=1 colspan=1>69.00</td><td rowspan=1 colspan=1>48.50</td><td rowspan=1 colspan=1>62.20</td><td rowspan=1 colspan=1>37.604</td><td rowspan=1 colspan=1>9.40</td><td rowspan=1 colspan=1>59.20</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>VideoLLM-online-8B</td><td rowspan=1 colspan=3>8.0523.8512.07</td><td rowspan=1 colspan=1>14.04</td><td rowspan=1 colspan=1>45.542</td><td rowspan=1 colspan=1>1.20</td><td rowspan=1 colspan=1>20.79</td><td rowspan=1 colspan=1>22.22</td><td rowspan=1 colspan=1>18.80</td><td rowspan=1 colspan=1>12.18</td><td rowspan=1 colspan=1>17.73</td><td rowspan=1 colspan=1>19.26</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>OASIS</td><td rowspan=1 colspan=3>91.9580.7381.03</td><td rowspan=1 colspan=1>67.42</td><td rowspan=1 colspan=1>67.33</td><td rowspan=1 colspan=1>79.89</td><td rowspan=1 colspan=1>78.14</td><td rowspan=1 colspan=1>61.95</td><td rowspan=1 colspan=1>60.14</td><td rowspan=1 colspan=1>47.31</td><td rowspan=1 colspan=1>57.21</td><td rowspan=1 colspan=1>67.68</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SimpleBaseline</td><td rowspan=1 colspan=3>94.0084.4080.20</td><td rowspan=1 colspan=1>64.00</td><td rowspan=1 colspan=1>75.30</td><td rowspan=1 colspan=1>81.50</td><td rowspan=1 colspan=1>79.90</td><td rowspan=1 colspan=1>53.20</td><td rowspan=1 colspan=1>60.80</td><td rowspan=1 colspan=1>50.50</td><td rowspan=1 colspan=1>54.90</td><td rowspan=1 colspan=1>67.40</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>LatentStream</td><td rowspan=1 colspan=2>85.9061.50</td><td rowspan=1 colspan=1>73.30</td><td rowspan=1 colspan=1>46.60</td><td rowspan=1 colspan=1>78.20</td><td rowspan=1 colspan=1>65.20</td><td rowspan=1 colspan=1>68.50</td><td rowspan=1 colspan=1>50.80</td><td rowspan=1 colspan=1>60.80</td><td rowspan=1 colspan=1>68.30</td><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>64.20</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>JoyAI-VL-Interaction</td><td rowspan=1 colspan=2>72.5061.50</td><td rowspan=1 colspan=1>75.00</td><td rowspan=1 colspan=1>59.00</td><td rowspan=1 colspan=1>78.20</td><td rowspan=1 colspan=1>64.10</td><td rowspan=1 colspan=1>68.40</td><td rowspan=1 colspan=1>62.00</td><td rowspan=1 colspan=1>70.90</td><td rowspan=1 colspan=1>12.90</td><td rowspan=1 colspan=1>48.60</td><td rowspan=1 colspan=1>58.50</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Mage-VL</td><td rowspan=1 colspan=2>94.6383.49</td><td rowspan=1 colspan=1>79.31</td><td rowspan=1 colspan=1>74.16</td><td rowspan=1 colspan=1>70.30</td><td rowspan=1 colspan=1>77.17</td><td rowspan=1 colspan=1>79.84</td><td rowspan=1 colspan=1>50.17</td><td rowspan=1 colspan=1>61.49</td><td rowspan=1 colspan=1>32.80</td><td rowspan=1 colspan=1>48.15</td><td rowspan=1 colspan=1>64.00</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FOLIO</td><td rowspan=1 colspan=3>96.7085.7085.70</td><td rowspan=1 colspan=1>70.60</td><td rowspan=1 colspan=1>62.50</td><td rowspan=1 colspan=1>90.90</td><td rowspan=1 colspan=1>82.00</td><td rowspan=1 colspan=1>59.00</td><td rowspan=1 colspan=1>63.60</td><td rowspan=1 colspan=1>84.60</td><td rowspan=1 colspan=1>69.10</td><td rowspan=1 colspan=1>75.55</td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>Full</td><td rowspan=1 colspan=1>Open-S</td><td rowspan=1 colspan=1>ource</td><td rowspan=1 colspan=1>Online</td><td rowspan=1 colspan=1>MLLMs</td><td rowspan=1 colspan=1>&amp;</td><td rowspan=1 colspan=2>Video Agents</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Flash-VStream-7B</td><td rowspan=1 colspan=1>24.16</td><td rowspan=1 colspan=2>29.3628.45</td><td rowspan=1 colspan=1>33.71</td><td rowspan=1 colspan=1>25.742</td><td rowspan=1 colspan=1>8.80</td><td rowspan=1 colspan=1>28.37</td><td rowspan=1 colspan=1>39.06</td><td rowspan=1 colspan=1>37.16</td><td rowspan=1 colspan=1>5.91</td><td rowspan=1 colspan=1>27.38</td><td rowspan=1 colspan=1>27.88</td><td rowspan=1 colspan=4>8.0267.2560.0045.09</td><td rowspan=1 colspan=1>33.61</td></tr><tr><td rowspan=1 colspan=1>Dispider</td><td rowspan=1 colspan=1>57.72</td><td rowspan=1 colspan=2>49.5462.07</td><td rowspan=1 colspan=1>44.94</td><td rowspan=1 colspan=1>61.39</td><td rowspan=1 colspan=1>51.63</td><td rowspan=1 colspan=1>54.55</td><td rowspan=1 colspan=1>48.48</td><td rowspan=1 colspan=1>55.41</td><td rowspan=1 colspan=1>4.30</td><td rowspan=1 colspan=1>36.06</td><td rowspan=1 colspan=1>45.31</td><td rowspan=1 colspan=1>18.05</td><td rowspan=1 colspan=1>37.36</td><td rowspan=1 colspan=2>48.7534.72</td><td rowspan=1 colspan=1>41.78</td></tr><tr><td rowspan=1 colspan=1>ReKV</td><td rowspan=1 colspan=1>68.50</td><td rowspan=1 colspan=2>57.8073.30</td><td rowspan=1 colspan=1>47.20</td><td rowspan=1 colspan=1>68.30</td><td rowspan=1 colspan=1>59.80</td><td rowspan=1 colspan=1>62.50</td><td rowspan=1 colspan=1>50.50</td><td rowspan=1 colspan=1>55.40</td><td rowspan=1 colspan=1>30.10</td><td rowspan=1 colspan=1>45.30</td><td rowspan=1 colspan=1>53.90</td><td rowspan=1 colspan=1>27.90</td><td rowspan=1 colspan=1>63.30</td><td rowspan=1 colspan=2>52.9048.00</td><td rowspan=1 colspan=1>52.00</td></tr><tr><td rowspan=1 colspan=1>StreamForest-7B</td><td rowspan=1 colspan=3>68.4653.2171.55</td><td rowspan=1 colspan=1>47.75</td><td rowspan=1 colspan=1>65.35</td><td rowspan=1 colspan=1>60.87</td><td rowspan=1 colspan=1>61.20</td><td rowspan=1 colspan=1>58.92</td><td rowspan=1 colspan=1>64.86</td><td rowspan=1 colspan=1>32.26</td><td rowspan=1 colspan=1>52.02</td><td rowspan=1 colspan=1>56.61</td><td rowspan=1 colspan=1>32.81</td><td rowspan=1 colspan=1>70.59</td><td rowspan=1 colspan=2>57.0853.49</td><td rowspan=1 colspan=1>55.57</td></tr><tr><td rowspan=1 colspan=1>Streamo</td><td rowspan=1 colspan=3>77.2066.1076.70</td><td rowspan=1 colspan=1>45.50</td><td rowspan=1 colspan=1>66.30</td><td rowspan=1 colspan=1>72.80</td><td rowspan=1 colspan=1>67.40</td><td rowspan=1 colspan=1>55.60</td><td rowspan=1 colspan=1>58.10</td><td rowspan=1 colspan=1>33.90</td><td rowspan=1 colspan=1>49.20</td><td rowspan=1 colspan=1>58.30</td><td rowspan=1 colspan=1>30.80</td><td rowspan=1 colspan=1>57.60</td><td rowspan=1 colspan=1>82.50</td><td rowspan=1 colspan=1>57.00</td><td rowspan=1 colspan=1>57.90</td></tr><tr><td rowspan=1 colspan=1>VST</td><td rowspan=1 colspan=3>80.5055.1072.40</td><td rowspan=1 colspan=1>55.10</td><td rowspan=1 colspan=1>76.20</td><td rowspan=1 colspan=1>64.10</td><td rowspan=1 colspan=1>67.20</td><td rowspan=1 colspan=1>56.90</td><td rowspan=1 colspan=1>64.90</td><td rowspan=1 colspan=1>48.40</td><td rowspan=1 colspan=1>56.70</td><td rowspan=1 colspan=1>61.95</td><td rowspan=1 colspan=1>33.00</td><td rowspan=1 colspan=1>66.90</td><td rowspan=1 colspan=1>62.10</td><td rowspan=1 colspan=1>54.00</td><td rowspan=1 colspan=1>59.30</td></tr><tr><td rowspan=1 colspan=1>ViSpeak</td><td rowspan=1 colspan=3>75.1758.7271.55</td><td rowspan=1 colspan=1>51.12</td><td rowspan=1 colspan=1>74.26</td><td rowspan=1 colspan=1>66.85</td><td rowspan=1 colspan=1>66.28</td><td rowspan=1 colspan=1>59.93</td><td rowspan=1 colspan=1>48.65</td><td rowspan=1 colspan=1>63.98</td><td rowspan=1 colspan=1>57.52</td><td rowspan=1 colspan=1>61.90</td><td rowspan=1 colspan=1>33.81</td><td rowspan=1 colspan=1>68.52</td><td rowspan=1 colspan=1>60.42</td><td rowspan=1 colspan=1>54.25</td><td rowspan=1 colspan=1>61.08</td></tr><tr><td rowspan=1 colspan=1>StreamAgent</td><td rowspan=1 colspan=3>71.2053.2063.60</td><td rowspan=1 colspan=1>53.90</td><td rowspan=1 colspan=1>67.30</td><td rowspan=1 colspan=1>58.70</td><td rowspan=1 colspan=1>61.30</td><td rowspan=1 colspan=1>54.80</td><td rowspan=1 colspan=1>58.10</td><td rowspan=1 colspan=1>25.80</td><td rowspan=1 colspan=1>41.70</td><td rowspan=1 colspan=1>51.50</td><td rowspan=1 colspan=1>35.90</td><td rowspan=1 colspan=1>48.40</td><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>45.40</td><td rowspan=1 colspan=1>49.40</td></tr><tr><td rowspan=1 colspan=1>EventMemAgent</td><td rowspan=1 colspan=3>75.8469.7273.28</td><td rowspan=1 colspan=1>55.62</td><td rowspan=1 colspan=1>67.33</td><td rowspan=1 colspan=1>67.93</td><td rowspan=1 colspan=1>68.29</td><td rowspan=1 colspan=2>59.6070.95</td><td rowspan=1 colspan=1>43.55</td><td rowspan=1 colspan=1>58.03</td><td rowspan=1 colspan=1>63.16</td><td rowspan=1 colspan=1>33.67</td><td rowspan=1 colspan=1>72.02</td><td rowspan=1 colspan=1>62.08</td><td rowspan=1 colspan=1>55.92</td><td rowspan=1 colspan=1>60.75</td></tr><tr><td rowspan=1 colspan=1>DeltaStreamer</td><td rowspan=1 colspan=4>77.8570.6482.7661.24</td><td rowspan=1 colspan=3>74.2673.6872.23</td><td rowspan=1 colspan=4>54.2179.0569.8967.72</td><td rowspan=1 colspan=1>69.98</td><td rowspan=1 colspan=4>54.8759.4660.4258.25</td><td rowspan=1 colspan=1>66.07</td></tr></table>

## 3.4 Query-Driven Planner

When a query arrives, a dedicated planner guides memory retrieval, continued observation, and tool use according to the question’s information needs. It determines what evidence the question requires and whether to seek it in past, current, or incoming frames. Initial retrieval uses the question alone to provide context for the planning call.

$$
\begin{array} { r l } & { C _ { t _ { q } } = \left( M _ { t _ { q } } ^ { \mathrm { s n a p s h o t } } , \mathcal { T } _ { t _ { q } } , \mathcal { R } _ { t _ { q } } ( q ) \right) , } \\ & { ~ P = \mathrm { V L M } ( q , I _ { t _ { q } } , C _ { t _ { q } } ) . } \end{array}\tag{4}
$$

Here, q is the question, $t _ { q }$ its arrival index, and $I _ { t _ { q } }$ the available frame. Context $C _ { t _ { q } }$ combines memory snapshot $M _ { t _ { q } } ^ { \mathrm { s n a p s h o t } }$ , entity index $\mathcal { I } _ { t _ { q } }$ , and evidence paths $\mathcal { R } _ { t _ { q } } ( q )$ . The planning call VLM produces plan P, specifying the target and temporal anchor, evidence type, temporal scope (past, current, future, or mixed), and search strategy. Auxiliary fields clarify viewpoint, available answer options, and whether the query concerns answerability or event progression (Appendix A.3). Subsequent retrieval combines q with the planned information need.

Persistent evidence guidance. The plan P is generated once and retained in subsequent prompts. Its accompanying evidence contract specifies where to seek evidence and when to use tools. The agent follows this guidance to retrieve memory, observe incoming frames, or replay historical frames, adapting its actions to updated memory and tool feedback. For details of the evidence contract and tool-use rules, please refer to Appendices A.3 and A.4.

Tab. 2 Evaluation results on StreamingBench [28].
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=6>Real-Time Visual UnderstandingOP CR CS ATP EU TR  PR SU ACP CT Avg.</td><td rowspan=1 colspan=1>Omni-Source UnderstandingER SCU SD MA Avg.</td><td rowspan=1 colspan=2>Contextual UnderstandingACUMCUSQA PO Avg.</td><td rowspan=1 colspan=1>Overall</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>Human</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Human</td><td rowspan=1 colspan=6>89.4792.0093.6091.4795.6592.5288.0088.7589.7491.3091.46</td><td rowspan=1 colspan=1>88.0088.2493.6090.2790.26</td><td rowspan=1 colspan=2>88.8090.4095.00100.00 93.55</td><td rowspan=1 colspan=1>91.66</td></tr><tr><td rowspan=1 colspan=7>Proprietary</td><td rowspan=1 colspan=1>MLLMs</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>Gemini 1.5 ProGPT-40</td><td></td><td rowspan=1 colspan=5>79.0280.4783.5479.6780.0084.7477.7864.2371.9548.7075.6977.1180.4783.9176.4770.1983.8066.6762.1969.1249.2273.28</td><td rowspan=1 colspan=1>46.8039.6074.9080.0060.2241.2037.2043.6056.0044.50</td><td rowspan=1 colspan=2>51.4140.7354.8045.1048.7341.2038.4032.8056.8638.70</td><td rowspan=1 colspan=1>67.0760.15</td></tr><tr><td rowspan=1 colspan=7>Open-Source Offline</td><td rowspan=1 colspan=1>MLLMs</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>LLaVA-OV-7B</td><td rowspan=5 colspan=6>80.3874.2276.0380.7272.6771.6567.5965.4565.7245.0871.1268.3271.0372.2261.1961.4746.1169.0467.7062.9359.2653.2554.9656.4863.7270.0363.2861.2070.9262.7359.5061.1153.6654.6734.7259.9653.6849.2270.9856.8653.4253.8954.6348.7850.1417.6252.32</td><td rowspan=2 colspan=1>40.8037.2033.6044.8038.4041.2022.0032.8043.6034.90</td><td rowspan=2 colspan=2>35.6036.0027.2729.5532.7431.2026.0039.6022.7331.66</td><td rowspan=1 colspan=1>56.36</td></tr><tr><td rowspan=1 colspan=1>Qwen2-VL-7B</td><td rowspan=2 colspan=3></td><td rowspan=1 colspan=1>54.14</td></tr><tr><td rowspan=1 colspan=1>InternVL-V2-8B</td><td rowspan=3 colspan=1>37.6026.4037.2042.0035.8039.6032.4028.0041.6035.4041.6026.4028.4036.0033.10</td><td rowspan=2 colspan=2>32.0031.2032.3240.9132.4232.8029.6030.3015.9129.95</td><td rowspan=2 colspan=1>51.4048.66</td></tr><tr><td rowspan=1 colspan=1>LongVA-7B</td></tr><tr><td rowspan=1 colspan=1>VILA-1.5-8B</td><td rowspan=1 colspan=2>26.8034.0023.2317.6527.35</td><td rowspan=1 colspan=1>43.20</td></tr><tr><td rowspan=1 colspan=1>Video-LLaMA2</td><td rowspan=1 colspan=2>55.86 55.47 57.41 58.17</td><td rowspan=4 colspan=5>52.8043.6139.8142.6845.6135.2349.5248.0068.0040.9060.8866.7078.1034.2068.7163.8069.5068.8054.4060.2065.1059.3040.7062.7021.8057.49</td><td rowspan=4 colspan=2>30.4032.4030.4036.0032.4036.8018.0032.8034.0030.4040.4027.6038.8045.6038.1028.8031.2030.4046.4034.20</td><td rowspan=1 colspan=1>24.8026.8018.670.0021.93</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B</td><td rowspan=1 colspan=4>62.70 68.00 69.70 53.30 67.50 65.10 67.60</td><td rowspan=1 colspan=2>25.6023.6031.2032.4028.20</td><td rowspan=1 colspan=1>46.84</td></tr><tr><td rowspan=1 colspan=1>InternVL3-8B</td><td rowspan=1 colspan=4>74.90 82.00 75.70 61.20 72.00 67.60 74.10</td><td rowspan=2 colspan=2>38.0026.0036.8031.2033.0034.4031.2038.8012.4029.20</td><td rowspan=2 colspan=1>53.9746.03</td></tr><tr><td rowspan=1 colspan=1>Gemma3-4B</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>Limited Open-Source Onl</td><td rowspan=1 colspan=1>ine MLLMs</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=3 colspan=1>TimeChat-OnlineQueryStreamLiveVLM</td><td rowspan=3 colspan=6>80.8079.7080.8083.3074.8078.8078.7064.2068.8058.0075.2882.1183.5978.2382.6975.4780.0679.6363.0167.9042.5574.0479.84 79.6984.8680.7267.0870.0974.0766.2667.9942.4972.36</td><td rowspan=3 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>Full Open-Source Online MLLMs &a</td><td rowspan=1 colspan=1>mp; Video Agents</td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=3 colspan=1>Flash-VStreamVideoLLM-OnlineDispider</td><td></td><td rowspan=3 colspan=5>25.8943.5724.9123.8727.3313.0818.5225.2023.8748.7023.2339.0740.0634.4931.0545.5432.4031.4834.1642.4927.8935.9974.9275.5374.1073.0874.4459.9276.1462.9162.1645.80</td><td rowspan=1 colspan=1>25.9124.9025.6028.4026.00</td><td rowspan=1 colspan=2>24.8025.2026.801.9624.12</td><td rowspan=1 colspan=1>24.04</td></tr><tr><td></td><td rowspan=1 colspan=1>31.2026.5124.1032.0028.45</td><td rowspan=1 colspan=1>24.1929.2030.80</td><td rowspan=1 colspan=1>3.9226.55</td><td rowspan=1 colspan=1>32.48</td></tr><tr><td></td><td rowspan=1 colspan=1>67.63</td><td rowspan=1 colspan=1>35.4625.2638.5743.3435.66</td><td rowspan=1 colspan=1>39.6227.6534.80</td><td rowspan=1 colspan=1>25.3433.61</td><td rowspan=1 colspan=1>53.12</td></tr><tr><td rowspan=2 colspan=1>STRIDEStreamAgent</td><td rowspan=2 colspan=6>74.9078.9076.7068.6077.0077.3077.8071.5083.0033.2072.4579.6378.3179.2875.8774.7476.9282.9466.3173.6955.4074.31</td><td rowspan=1 colspan=1>39.6022.4044.0050.8039.20</td><td rowspan=1 colspan=2>34.0035.2043.2042.8038.80</td><td rowspan=1 colspan=1>57.58</td></tr><tr><td rowspan=1 colspan=1>35.8626.2638.8744.0436.26</td><td rowspan=1 colspan=2>39.7230.2539.6028.9034.62</td><td rowspan=1 colspan=1>57.02</td></tr><tr><td rowspan=1 colspan=1>DeltaStreamer</td><td rowspan=1 colspan=6>77.8072.7080.8084.0072.3081.9078.7072.8066.8050.5074.80</td><td rowspan=1 colspan=1>45.6038.0040.0044.0041.90</td><td rowspan=1 colspan=2>42.8055.2047.6010.4039.00</td><td rowspan=1 colspan=1>59.53</td></tr></table>

## 3.5 Look-Back Verification and Replay

To ground answers in visual evidence beyond stored descriptions, we combine memory consistency checks with targeted replay of historical frames. Replay has two entry points. The agent can request missing visual details, or a verifier can flag unresolved conflicts among retrieved records. Both routes inspect retained historical frames through replay(τ), where τ is a target frame index available from the current index t, with $0 \leq \tau \leq t .$ Before a candidate answer is accepted, a rule-based verifier checks records retrieved using the planner’s retrieval query. The verifier groups claims by subject and property and applies lightweight cross-layer consistency and temporal heuristics. Unresolved conflicts with available historical frames become replay candidates.

For verifier-triggered candidates, heuristic priorities favor severe conflicts and frames closer to the question time. The scheduler selects

$$
u ^ { * } = \underset { u \in \mathcal { U } _ { t } : \rho ( u ) > \eta } { \arg \operatorname* { m a x } } \rho ( u ) , \tau ^ { * } = \tau _ { u ^ { * } } ,\tag{5}
$$

Here, t is the current frame index and $\mathcal { U } _ { t }$ contains unresolved conflicts with available replay targets. For conflict $u , \rho ( u )$ is its heuristic priority, η the threshold, and $\tau _ { u }$ its frame index. The highest-priority eligible conflict $u ^ { * }$ yields replay frame $\tau ^ { * } = \tau _ { u ^ { * } }$ . With no eligible candidate, the verifier requests no replay. After inspecting the historical frame, the agent resumes investigation at the current frame, using additional visual evidence to generate an answer or re-evaluate. Appendix A.5 details the conflict rules, priorities, and candidate limit.

## 4 Experiments

## 4.1 Experimental Details

Implementation Details. We use Qwen3.6-27B [37] as the shared backbone for observation, query-driven planning, and answer generation. Inference runs with vLLM [23] in BF16 precision on 8x NVIDIA A100 GPUs. We process video frames at a base sampling rate of 1 fps using the backbone’s image preprocessing pipeline.

Tab. 3 Ablation of delta-augmented observation. 2F-NoDelta receives the same two-frame input as our method but does not explicitly generate or store a delta field. Comparing the two-frame variants tests explicit change recording at the same visual-input count.
<table><tr><td>Setting</td><td>Input Frames</td><td>Delta Representation</td><td>OVO-Bench</td></tr><tr><td>1F-NoDelta</td><td> $I _ { t }$ </td><td>No Delta</td><td>58.87</td></tr><tr><td>1F-TextDelta</td><td> $I _ { t }$ </td><td>Text-Memory comparison</td><td>63.19</td></tr><tr><td>2F-NoDelta</td><td> $\left( I _ { t - 1 } , I _ { t } \right)$ </td><td>No Delta</td><td>61.12</td></tr><tr><td>2F-VisualDelta (Ours)</td><td> $\left( I _ { t - 1 } , I _ { t } \right)$ </td><td>Visual comparison</td><td>66.07</td></tr></table>

Tab. 4 Ablation of memory components. All variants use identical observations, tools, planner, and answer decoding; only the component shown in each mini-table is changed.
<table><tr><td>Flat Memory</td><td>61.25</td><td>w/o DAG w/DAG</td><td>62.73</td><td>w/o Index</td><td>63.40</td><td>Fixed</td><td>63.77</td></tr><tr><td>Hierarchical</td><td>66.07</td><td></td><td>66.07</td><td>w/ Index</td><td>66.07</td><td>BOCPD</td><td>66.07</td></tr><tr><td colspan="2">(a) Memory hierarchy</td><td colspan="2">(b) DAG organization</td><td colspan="2">(c) Entity indexing</td><td colspan="2">(d) Episode Mode</td></tr></table>

Benchmarks. We evaluate DeltaStreamer on OVO-Bench [33] and StreamingBench [28]. OVO-Bench tests real-time perception, recall of past events, and proactive responses, while StreamingBench covers real-time visual understanding, multimodal information, and contextual reasoning. Dataset statistics and task definitions are provided in Appendix B

## 4.2 Main Results

## 4.2.1 Compared Methods

We compare proprietary models, open-source offline models, and streaming systems. Proprietary models include Gemini 1.5 Pro [48], Gemini 3 Flash<sup>1</sup>, and GPT-4o [34]. Offline models include Qwen2-VL-7B/72B [50], Qwen3- VL-8B [3], LLaVA-Video-7B [76], LLaVA-OneVision-7B (LLaVA-OV) [24], InternVL2-8B [9], InternVL3- 8B [84], LongVU-7B [40], LongVA-7B [75], VILA-1.5-8B [27], VideoLLaMA2-7B [10], and Gemma3-4B [49].

Streaming systems include HERMES [73], VideoLLM-online [6], OASIS [26], SimpleBaseline [41], FOLIO [12], Stream-IT [51], JoyAI-VL-Interaction [66], LatentStream [36], Mage-VL [63], Flash-VStream [71, 72], Dispider [35], ReKV [11], StreamForest [68], Streamo [59], VST [16], ViSpeak [14], TimeChat-Online [67], QueryStream [74], LiveVLM [32], STRIDE [20], StreamAgent [61], and EventMemAgent [58]. Human results are provided by the respective benchmarks.

## 4.2.2 Benchmark Results

OVO-Bench. Tab. 1 reports an overall score of 66.07 for DeltaStreamer, a gain of +5.32 points over EventMemAgent. It also outperforms Gemini 3 Flash by +2.05 points, with higher scores on action-sequence identification, hallucination detection, and repeated-event counting. The method reaches 79.05 on action-sequence identification (ASI) and 69.89 on hallucination detection (HLD). Its improvements on these tasks align with our emphasis on distinguishing events and gathering evidence relevant to a question.

StreamingBench. Tab. 2 reports an overall score of 59.53, exceeding STRIDE by +1.95 points and also outperforming StreamAgent. Contextual understanding improves over STRIDE, with scores of 55.20 on MCU and 47.60 on SQA. However, PO remains weak at 10.40, showing that improved evidence handling does not by itself solve proactive response timing. Together, these results demonstrate the benefits of change-aware memory and planned evidence gathering for temporal reasoning and contextual understanding.

## 4.3 Ablation Studies

Effect of Explicit Visual Delta. To separate explicit change recording from additional visual input, we compare observation variants in Tab. 3. With identical two-frame input,

Tab. 5 Planner and component ablations on OVO-Bench. Left: cumulative planner settings, with persistence retaining the plan across post-query rounds. Right: combinations of delta (D), hierarchical memory (M), and planner (P).
<table><tr><td>Cumulative setting</td><td>HLD</td><td>OVO-Bench</td></tr><tr><td>No planner</td><td>45.28</td><td>59.30</td></tr><tr><td>+ Plan generation</td><td>65.83</td><td>63.85</td></tr><tr><td>+ Persistence</td><td>69.13</td><td>64.77</td></tr><tr><td>+ POV</td><td>68.07</td><td>65.14</td></tr><tr><td>+ Evidence contract</td><td>69.89</td><td>66.07</td></tr></table>

(a) Planner ablation

![](images/cc8b91383269dbb82355196d85b96643ad9f4ec50bf60155dd017385d6e54ce7.jpg)  
(b) Component combinations

Tab. 6 Ablation of memory rendering and core visual tools. All variants keep the model, observation protocol, the same planner, and answer decoding fixed.
<table><tr><td>Memory Rendering</td><td>OVO-Bench</td><td>Configuration</td><td>Replay</td><td>Tracking</td><td>Zoom</td><td>OVO-Bench</td></tr><tr><td>Full History</td><td>59.18</td><td>Full Tool Suite</td><td>√</td><td>√</td><td>√</td><td>66.07</td></tr><tr><td>Recent Tail Only</td><td>60.04</td><td>– Replay</td><td></td><td>√</td><td>√</td><td>63.28</td></tr><tr><td>Pins + Episodes + Tail</td><td>63.69</td><td>– Tracking</td><td>√</td><td></td><td>√</td><td>65.31</td></tr><tr><td>Hierarchical + Middle Recall</td><td>66.07</td><td>- Zoom</td><td>√</td><td>√</td><td></td><td>64.16</td></tr></table>

(a) Memory rendering strategy

2F-NoDelta scores 61.12 and 2F-VisualDelta reaches 66.07; visual comparison also outperforms 1F-TextDelta. These results support making temporal evidence explicit for later reasoning. Similar scene descriptions can leave event progression implicit, requiring reconstruction across separate records. Delta writes these transitions directly into memory, providing a retrieval cue for questions about how events unfold. Whereas 1F-TextDelta compares the current frame with textual memory, adjacent-frame comparison can use visual details omitted from earlier descriptions. The resulting observation retains both the current state and an explicit account of its changes, making the temporal relation directly available during investigation. We also demonstrate a comparison of these four contracts in Fig. 3.

(b) Core visual tools  
![](images/41571220df0a95f869c235c2ab0bc5f1a0a49e4c8d0d7790d558375aab596983.jpg)  
Fig. 3 Illustrative delta comparison.

Memory Organization. Hierarchical memory reaches 66.07, compared with 61.25 for flat memory (Tab. 4). The DAG and entity-index ablations also favor retaining both forms of access. As the stream grows, summaries provide an event-level overview, while source links and entity indexing help retrieve the underlying observations and timestamps when later queries require omitted details. History can thus be summarized while preserving a route to finer evidence. Online episode construction also outperforms fixed grouping, consistent with adapting memory to changes in the stream. This comparison covers the complete episode pipeline (Appendix A.2).

Query-Driven Planning. With cumulative planner additions in Tab. 5(a), plan generation reaches 63.85 and persistence reaches 64.77. Planning makes the target, temporal anchor, and required evidence explicit, supplying retrieval cues beyond the query. Persistence keeps these requirements available as observations and tool results accumulate, tying later decisions to the original information need. The POV field specifies the relevant perspective, while the evidence contract guides retrieval, continued observation, and tool use. The substantial HLD improvement from plan generation highlights the importance of the plan itself for hallucination detection.

Cross-Component Ablation. We examine the complementarity of delta (D), hierarchical memory (M), and planning (P) in Tab. 5(b); configurations without M use flat memory. With M enabled, adding D yields +0.93 without P and +4.95 with P. This larger conditional benefit suggests that planning helps the agent use retained transitions. Richer memory is useful when the agent can identify which evidence a question requires, while planning benefits from explicit changes and traceable source records. This interaction is consistent with the full combination outperforming all single-component and pairwise settings.

![](images/bf2718597fe047676e98f47d7f0dfdc6bfa7ccc82440f2095f467df9a81bef66.jpg)  
Fig. 4 Qualitative examples of DeltaStreamer. The upper case recalls the action following the first block placement. The lower case recovers the missing wall letters from a historical crop.

Memory Rendering and Tool Ablation. The rendering comparison examines how retained history enters the visible context (Tab. 6). Adding middle recall to pins, episodes, and the recent tail raises accuracy from 63.69 to 66.07. Summaries preserve temporal coverage but cannot retain every detail a later question may require. Selective recall restores relevant older observations that are absent from the summaries and outside the recent window. The model can therefore use a compact temporal overview together with detailed source evidence selected for the question.

Removing replay lowers accuracy to 63.28, the largest drop among tool removals. Pre-query observation cannot anticipate every future information need. Middle recall recovers details already stored as text, while replay can supply visual evidence that was never encoded, explaining its value alongside structured memory. Zoom and tracking also help by exposing fine spatial details and supporting correspondence of the same target across frames. These capabilities address complementary evidence needs; each ablation removes one capability with the remaining tools available.

## 4.4 Qualitative Analysis

Fig. 4 illustrates the complementary roles of change-aware memory and query-driven evidence gathering. Explicit changes help represent event progression when successive frames contain similar scene content, while observations provide temporal anchors for later reasoning. However, textual memory can retain a scene without preserving every detail that a future question may require. The planner connects the query’s information needs to relevant observations and guides historical visual inspection when stored descriptions are insufficient. This enables the agent to reason over retained evidence and recover omitted visual details on demand.

## 5 Conclusions

In this work, we introduce DeltaStreamer, a human-like agentic framework for streaming video understanding. It records explicit inter-frame changes in traceable hierarchical memory to preserve event transitions for later queries. A query-driven planner specifies evidence needs and guides memory retrieval and targeted replay when evidence is insufficient. Experiments on OVO-Bench and StreamingBench, together with ablations, demonstrate the benefits of combining change-aware memory with planned evidence gathering. These findings support a streaming paradigm in which systems continuously observe and remember, then actively investigate when a question arrives.

## References

[1] Ryan Prescott Adams and David JC MacKay. Bayesian online changepoint detection. arXiv preprint arXiv:0710.3742, 2007. 17

[2] Vatsal Agarwal, Saksham Suri, Matthew Gwilliam, Pulkit Kumar, and Abhinav Shrivastava. Going down memory lane: Scaling tokens for video stream understanding with dynamic kv-cache memory. arXiv preprint arXiv:2602.18434, 2026. 3

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 7

[4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 1

[5] Dibyadip Chatterjee, Edoardo Remelli, Yale Song, Bugra Tekin, Abhay Mittal, Bharat Bhatnagar, Necati Cihan CamgÃk<sub>,</sub> z, Shreyas Hampali, Eric Sauser, Shugao Ma, et al. Memory-efficient streaming videollms for real-time procedural video understanding. arXiv preprint arXiv:2504.13915, 2025. 1

[6] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18407–18418. IEEE, 2024. 3, 7

[7] Joya Chen, Ziyun Zeng, Yiqi Lin, Wei Li, Zejun Ma, and Mike Zheng Shou. Live: Learning video llm with streaming speech transcription at scale. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 29083–29095. IEEE, 2025. 3

[8] Yilong Chen, Xiang Bai, Zhibin Wang, Chengyu Bai, Yuhan Dai, and Ming Lu. Streamkv: Streaming video question-answering with segment-based kv cache retrieval and compression. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 3120–3128, 2026. 1

[9] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 7

[10] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and Lidong Bing. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. URL https://arxiv.org abs/2406.07476. 7

[11] Shangzhe Di, Zhelun Yu, Guanghao Zhang, Haoyuan Li, Hao Cheng, Bolin Li, Wanggui He, Fangxun Shu, and Hao Jiang. Streaming video question-answering with in-context video kv-cache retrieval. In International Conference on Learning Representations, volume 2025, pages 42115–42127, 2025. 3, 7

[12] Haoyang Fan, Dhruv Parikh, Anvitha Ramachandran, Sameh Gobriel, Nilesh Jain, Rajgopal Kannan, and Viktor Prasanna. Folio: Focused semantic memory for streaming video understanding. arXiv preprint arXiv:2607.13298, 2026. 7

[13] Yue Feng, Yingshi Liu, Jingrou Zhang, Jie Qin, et al. Monitor: Exploiting large language models with instruction for online video anomaly detection. Advances in Neural Information Processing Systems, 38: 35851–35871, 2026. 1

[14] Shenghao Fu, Qize Yang, Yuan-Ming Li, Yi-Xing Peng, Kun-Yu Lin, Xihan Wei, Jian-Fang Hu, Xiaohua Xie, and Wei-Shi Zheng. Vispeak: Visual instruction feedback in streaming videos. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 21778–21788. IEEE, 2025. 7

[15] Yiran Guan, Liang Yin, Dingkang Liang, Jianzhong Ju, Zhenbo Luo, Jian Luan, Yuliang Liu, and Xiang Bai. Video streaming thinking: Videollms can watch and think simultaneously. arXiv preprint arXiv:2603.12262, 2026. 1

[16] Yiran Guan, Liang Yin, Dingkang Liang, Jianzhong Ju, Zhenbo Luo, Jian Luan, Yuliang Liu, and Xiang Bai. Video streaming thinking: Videollms can watch and think simultaneously. In European Conference on Computer Vision (ECCV), 2026. 7

[17] Zifan Han, Hongbo Sun, Jinglin Xu, Canhui Tang, Yulong Lei, Xuchong Zhang, Hongbin Sun, Zhongjiang He, and Hao Sun. Wat: Online video understanding needs watching before thinking. arXiv preprint arXiv:2603.13412, 2026. 3

[18] Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, and Limin Wang. Online video understanding: Ovbench and videochat-online. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3328–3338. IEEE, 2025. 1

[19] Xinqi Jin, Hanxun Yu, Bohan Yu, Kebin Liu, Jian Liu, Keda Tao, Yixuan Pei, Huan Wang, Fan Dang, Jiangchuan Liu, et al. Streamingassistant: Efficient visual token pruning for accelerating online video understanding. arXiv preprint arXiv:2512.12560, 2025. 3

[20] Junho Kim, Hosu Lee, James M. Rehg, Minsu Kim, and Yong Man Ro. Stride: When to speak meets sequence denoising for streaming video understanding. In European Conference on Computer Vision, pages 38–56. Springer, 2026. 7

[21] Minsoo Kim, Kyuhong Shim, Jungwook Choi, and Simyung Chang. Infinipot-v: Memory-constrained kv cache compression for streaming video understanding. Advances in Neural Information Processing Systems, 38:138983–139013, 2026. 1

[22] Dohwan Ko, Ji Soo Lee, Pierce Chuang, Debojeet Chatterjee, Ashish Shenoy, Yichao Lu, Seungwhan Moon, Xin Luna Dong, Vikas Bhardwaj, and Hyunwoo J Kim. Svmemagent: A streaming video memory agent for query-agnostic online frame selection. arXiv preprint arXiv:2609.18540, 2026. 3

[23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023. 6

[24] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 7

[25] Ruanjun Li, Yuedong Tan, Yuanming Shi, and Jiawei Shao. Videoscan: Enabling efficient streaming video understanding via frame-level semantic carriers. arXiv preprint arXiv:2503.09387, 2025. 3

[26] Zhijia Liang, Jiaming Li, Weikai Chen, Yanhao Zhang, Haonan Lu, and Guanbin Li. Oasis: On-demand hierarchical event memory for streaming video reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 3, 7

[27] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26679–26689. IEEE, 2024. 7

[28] Junming Lin, Zheng Fang, Chi Chen, Haoxuan Cheng, Zihao Wan, Fuwen Luo, Ziyue Wang, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. In ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12147–12151. IEEE, 2026. 2, 6, 7, 21

[29] Junyan Lin, Junlong Tong, Hao Wu, Jialiang Zhang, Jinming Liu, Xin Jin, and Xiaoyu Shen. Speak while watching: Unleashing true real-time video understanding capability of multimodal large language models. arXiv preprint arXiv:2601.06843, 2026. 1

[30] Zikang Liu, Longteng Guo, Handong Li, Ru Zhen, Xingjian He, Ruyi Ji, Xiaoming Ren, Yanhao Zhang, Haonan Lu, and Jing Liu. Thinking in streaming video. arXiv preprint arXiv:2603.12938, 2026. 1

[31] Haocheng Lu, Nan Zhang, Wei Tao, Xiaoyang Qu, Guokuan Li, Jiguang Wan, and Jianzong Wang. Vista: Scene-aware optimization for streaming video question answering under post-hoc queries. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 7539–7547, 2026. 3

[32] Zhenyu Ning, Guangda Liu, Qihao Jin, Chengwei Li, Wenchao Ding, Minyi Guo, and Jieru Zhao. Livevlm: Efficient online video understanding via streaming-oriented kv cache and retrieval. arXiv preprint arXiv:2505.15269, 2025. 7

[33] Junbo Niu, Yifei Li, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Rui Qian, et al. Ovo-bench: How far is your video-llms from real-world online video understanding? In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18902–18913. IEEE, 2025. 2, 7, 20

[34] OpenAI. Gpt-4o system card, 2024. URL https://openai.com/index/hello-gpt-4o/. 7

[35] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24045–24055. IEEE, 2025. 7

[36] Hongyu Qu, Guangming Yao, Ling Xing, Xiaobin Hu, Rongxing Ding, Guibin Zhang, Fan Zhang, Yi Yuan, Xiangbo Shu, and Shuicheng Yan. Beyond retrieval: Progressive latent memory evolution for streaming video understanding. arXiv preprint arXiv:2609.04131, 2026. 7

[37] Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026. URL https: //qwen.ai/blog?id=qwen3.6-27b. 6

[38] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023. 19

[39] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, volume 2025, pages 28085–28128, 2025. 19

[40] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv:2410.17434, 2024. 7

[41] Yujiao Shen, Shulin Tian, Jingkang Yang, and Ziwei Liu. A simple baseline for streaming video understanding. arXiv preprint arXiv:2604.02317, 2026. 7

[42] Gong Sitong, Tianyu Yan, Caixin Kang, Bo Zheng, Xiang Ruan, Huchuan Lu, Kaipeng Zhang, Yoichi Sato, and Yifei Huang. Vinci2: Providing proactive assistance in continuous egocentric videos. arXiv preprint arXiv:2607.11523, 2026. 3

[43] Boyuan Sun, Yuqi Yang, Le Zhang, Ming-Ming Cheng, and Qibin Hou. Corrmatch: Label propagation via correlation matching for semi-supervised semantic segmentation. IEEE Computer Vision and Pattern Recognition (CVPR), 2024. 1

[44] Boyuan Sun, Modi Jin, Bowen Yin, and Qibin Hou. Depth anything at any condition. arXiv preprint arXiv:2507.01634, 2025. 1

[45] Boyuan Sun, Jiaxing Zhao, Xihan Wei, and Qibin Hou. Llava-scissor: Token compression with semantic connected components for video llms. arXiv preprint arXiv:2506.21862, 2025. 3

[46] Boyuan Sun, Bo-Wen Yin, Yuan-Ming Li, Xihan Wei, and Qibin Hou. See what i mean: Aligning vision and language representations for video fine-grained object understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 36814–36827, 2026. 1

[47] Hongyuan Tao, Bencheng Liao, Shaoyu Chen, Haoran Yin, Qian Zhang, Wenyu Liu, and Xinggang Wang. Infinitevl: Synergizing linear and sparse attention for highly-efficient, unlimited-input vision-language models. arXiv preprint, 2025. 3

[48] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 7

[49] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025. 7

[50] Qwen team. Qwen2-vl. 2024. 7

[51] Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, and Ping Huang. Streambridge: Turning your offline video large language model into a proactive streaming assistant. Advances in Neural Information Processing Systems, 38:132332–132359, 2026. 1, 7

[52] Junxi Wang, Te Sun, Jiayi Zhu, Junxian Li, Haowen Xu, Zichen Wen, Xuming Hu, Zhiyu Li, and Linfeng Zhang. Streammeco: Long-term agent memory compression for efficient streaming video understanding. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 13234–13251, 2026. 2

[53] Lu Wang, Zhuoran Jin, Yupu Hao, Yubo Chen, Kang Liu, Yulong Ao, and Jun Zhao. Think while watching: Online streaming segment-level memory for multi-turn video reasoning in multimodal large language models. arXiv preprint arXiv:2603.11896, 2026. 1

[54] Xiaofeng Wang, Zheng Zhu, Yunpeng Zhang, Guan Huang, Yun Ye, Wenbo Xu, Ziwei Chen, and Xingang Wang. Are we ready for vision-centric driving streaming perception? the asap benchmark. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9600–9610. IEEE, 2023. 1

[55] Yiyu Wang, Xuyang Liu, Xiyan Gui, Xinying Lin, Boxue Yang, Chenfei Liao, Tailai Chen, and Linfeng Zhang. Accelerating streaming video large language models via hierarchical token compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18523– 18533, 2026. 3

[56] Yuxuan Wang, Yueqian Wang, Bo Chen, Tong Wu, Dongyan Zhao, and Zilong Zheng. Omnimmi: A comprehensive multi-modal interaction benchmark in streaming video contexts. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18925–18935. IEEE, 2025. 3

[57] Yuxi Wei, Wei Huang, Qirui Chen, Lu Hou, and Xiaojuan Qi. See, remember, explore: A benchmark and baselines for streaming spatial reasoning. arXiv preprint arXiv:2603.23864, 2026. 2

[58] Siwei Wen, Zhangcheng Wang, Xingjian Zhang, Lei Huang, and Wenjun Wu. Eventmemagent: Hierarchical event-centric memory for online video understanding with adaptive tool use. In European Conference on Computer Vision (ECCV), 2026. 2, 3, 7

[59] Jiaer Xia, Peixian Chen, Mengdan Zhang, Xing Sun, and Kaiyang Zhou. Streaming video instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 31219–31229, 2026. 1, 3, 7

[60] Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, and Huchuan Lu. Streaming video understanding and multi-round interaction with memory-enhanced knowledge. In International Conference on Learning Representations, volume 2025, pages 69332–69351, 2025. 3

[61] Haolin Yang, Feilong Tang, Lingxiao Zhao, Xinlin Zhuang, Yifan Lu, Xiang An, Ming Hu, Xiaofeng Zhang, Abdalla Swikir, Junjun He, et al. Streamagent: Towards anticipatory agents for streaming video understanding. arXiv preprint arXiv:2508.01875, 2025. 2, 3, 7

[62] Qize Yang, Shimin Yao, Weixuan Chen, Shenghao Fu, Detao Bai, Jiaxing Zhao, Boyuan Sun, Bowen Yin, Xihan Wei, and Jingren Zhou. Humanomniv2: From understanding to omni-modal reasoning with context. arXiv preprint arXiv:2506.21277, 2025. 3

[63] Senqiao Yang, Kaichen Zhang, Zhaoyang Jia, Jinghao Guo, Yifei Shen, Xinjie Zhang, Xiaoyi Zhang, Haoqing Wang, Xiao Li, Peng Zhang, et al. Mage-vl: An efficient codec-native streaming multimodal foundation model. arXiv preprint arXiv:2607.24904, 2026. 7

[64] Yanlai Yang, Zhuokai Zhao, Satya Narayan Shukla, Aashu Singh, Shlok Kumar Mishra, Lizhu Zhang, and Mengye Ren. Streammem: Query-agnostic kv cache memory for streaming video understanding. arXiv preprint arXiv:2508.15717, 2025. 3

[65] Zhenyu Yang, Yuhang Hu, Zemin Du, Dizhan Xue, Shengsheng Qian, Jiahong Wu, Fan Yang, Weiming Dong, and Changsheng Xu. Svbench: A benchmark with temporal multi-turn dialogues for streaming video understanding. In International Conference on Learning Representations, volume 2025, pages 42522–42556, 2025. 3

[66] Dingyu Yao, Junhao Zhou, Chenxu Yang, Chuanyu Qin, Haowen Hou, Zheming Liang, Congcong Wang, Yuhang Cao, Shenglong Ye, Shuai Xie, et al. Joyai-vl-interaction: Real-time vision-language interaction intelligence. arXiv preprint arXiv:2606.14777, 2026. 7

[67] Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, et al. Timechat-online: 80% visual tokens are naturally redundant in streaming videos. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 10807–10816, 2025. 3, 7

[68] Xiangyu Zeng, Kefan Qiu, Qingyu Zhang, Xinhao Li, Jing Wang, Jiaxin Li, Ziang Yan, Kun Tian, Meng Tian, Xinhai Zhao, et al. Streamforest: Efficient online video understanding with persistent event memory. Advances in Neural Information Processing Systems, 38:75804–75835, 2026. 3, 7

[69] Ce Zhang, Jing Bi, Jinxi He, Jianshu Zhang, Jingyang Lin, Yunzhong Xiao, Minghao Fu, Yaqi Xie, Zhentao Xie, Weicong Chen, et al. Streamscout: Learning when to look deeper for streaming video understanding. arXiv preprint arXiv:2609.00291, 2026. 3

[70] Gengyuan Zhang, Mingcong Ding, Tong Liu, Yao Zhang, and Volker Tresp. Memory helps, but confabulation misleads: Understanding streaming events in videos with mllms. arXiv preprint arXiv:2502.15457, 2025. 1

[71] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memory-based real-time understanding for long video streams. arXiv preprint arXiv:2406.08085, 2024. 7

[72] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, and Xiaojie Jin. Flash-vstream: Efficient real-time understanding for long video streams. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 21059–21069. IEEE, 2025. 7

[73] Haowei Zhang, Shudong Yang, Jinlan Fu, See Kiong Ng, and Xipeng Qiu. Hermes: Kv cache as hierarchical memory for efficient streaming video understanding. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8411–8430, 2026. 7

[74] Kairui Zhang, Zhenyu Yang, Bing Wang, Shengsheng Qian, and Changsheng Xu. Querystream: Advancing streaming video understanding with query-aware pruning and proactive response. In The Fourteenth International Conference on Learning Representations, 2026. 7

[75] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 7

[76] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data, 2024. URL https://arxiv.org/abs/2410.02713. 7

[77] Yulin Zhang, Cheng Shi, and Sibei Yang. Weavetime: Stream from earlier frames into emergent memory in videollms. arXiv preprint arXiv:2602.22142, 2026. 3

[78] Jiaxing Zhao, Boyuan Sun, Xiang Chen, Xihan Wei, and Qibin Hou. Llava-octopus: Unlocking instructiondriven adaptive projector fusion for video understanding. arXiv preprint arXiv:2501.05067, 2025. 1

[79] Jiaxing Zhao, Qize Yang, Yixing Peng, Detao Bai, Shimin Yao, Boyuan Sun, Xiang Chen, Shenghao Fu, Xihan Wei, Liefeng Bo, et al. Humanomni: A large vision-speech language model for human-centric video understanding. arXiv preprint arXiv:2501.15111, 2025. 3

[80] Jiaxing Zhao, Boyuan Sun, Xiang Chen, and Xihan Wei. Facial dynamics in video: Instruction tuning for improved facial expression perception and contextual awareness. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 13163–13171, 2026. 1

[81] Ruixiang Zhao, Hualei Wang, Renhe Sun, Enzhi Zhou, Jincenzi Wu, Xujie Song, Kexin Shi, Zihang Liu, Pengcheng Zhu, Jiayi Zhou, et al. Realtime-venus: A full-duplex interaction system with asynchronous delegation. arXiv preprint arXiv:2609.13814, 2026. 3

[82] Ruixiang Zhao, Jie Yang, Zijie Xin, Tianyi Wang, Fengyun Rao, Jing LYU, and Xirong Li. Omnipro: A comprehensive benchmark for omni-proactive streaming video understanding. arXiv preprint arXiv:2605.18577, 2026. 1

[83] Yikai Zheng, Xin Ding, Yifan Yang, Shiqi Jiang, Hao Wu, Qianxi Zhang, Weijun Wang, Ting Cao, and Yunxin Liu. Em-garde: A propose-match framework for proactive streaming video understanding. arXiv preprint arXiv:2603.19054, 2026. 2

[84] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 7

# Appendix for DeltaStreamer: Towards Human-Like Agentic Streaming Video Understanding

These supplementary materials document implementation rules and parameter settings, benchmark task definitions and evaluation details, memory and tool-use statistics, the prompt templates used by DeltaStreamer, and additional qualitative examples. We also discuss limitations and broader impacts in the discussion section.

## Contents

A Methodology Details 17   
A.1 Observation Scoring Details 17   
A.2 Memory Construction and Rendering 17   
A.2.1 Episode Construction . 17   
A.2.2 Entity Index and Memory Rendering 18   
A.3 Planner Schema and Evidence Guidance 18   
A.4 Evidence-Gathering Tools 19   
A.5 Look-Back Verification and Replay Details 20   
B Benchmark Details 20   
B.1 OVO-Bench . 20   
B.2 StreamingBench . 21   
C Memory and Tool Use 21   
D Discussions 22   
D.1 Limitations 22   
D.2 Broader Impacts . 22   
E Prompt Templates 23   
E.1 Pre-Query Observation . 23   
E.2 Memory Summarization and Online Recaps 25   
E.3 Query-Driven Planning 26   
E.4 Evidence Gathering and Answering 28   
F Additional Qualitative Examples 33

## A Methodology Details

## A.1 Observation Scoring Details

For the novelty score in Eq. 2, the history window contains up to 12 recent observations. Field tokens are extracted using regular-expression matching, converted to lowercase, and deduplicated. The field weights are 0.35 for delta, 0.25 for actions, 0.15 for actors, 0.10 for objects, 0.10 for OCR, and 0.05 for scene; description receives the default weight of 0.05. Delta receives the largest individual weight to prioritize explicit visual changes when assessing new information. With no observation history, a nonempty observation receives novelty 1. A field without a matching historical token set has maximum similarity 0, and an observation with no nonempty token set receives novelty 0.

Salience converts novelty into a weight for later retrieval, increasing the priority of observations that add information beyond recurring scene descriptions. For novelty $s \geq 0 . 3$ , salience is 1 + 0.5s and the observation is marked as change-bearing; otherwise, salience is 0.5. A separate repeated-delta guard demotes repeated templates to salience 0.5 and clears the change flag. This decision uses the full structured observation and repetition checks, including when the delta reports a verified-static frame. The associated change flag also determines whether a group of observations satisfies the change requirement for episodic aggregation.

## A.2 Memory Construction and Rendering

Stored records and cross-layer links. Working events retain their timestamp, observation text, delta, salience, change flag, referenced frame, and optional parent episode identifier. Episodes store their time span, summary, and supporting working-event identifiers. Long-term pins retain their source episode identifier and are created from model-emitted recap blocks or automatically promoted episode boundaries. Separate reverse pointers support navigation in both directions without introducing cycles into the hierarchy’s directed abstraction edges. An episode need not have a long-term pin to support working–episodic retrieval.

## A.2.1 Episode Construction

Episodes can be created as new observations arrive or when memory is prepared for a model call. Online boundary detection identifies when to close the current episode, while aggregation and coverage rules summarize observations that have not yet been assigned to one. The model can also request an explicit episode through a Recap.

Online boundaries. A run is the sequence of working observations accumulated since the previous online boundary. Bayesian Online Changepoint Detection (BOCPD) [1] estimates whether an incoming observation continues this sequence or begins a new one. A boundary is proposed when the most probable hypothesis has index zero, indicating a new sequence, and the accumulated run contains at least three observations. Before accepting it, a content-continuity check compares the incoming observation with the first observation of the run. If their scene descriptions match and their estimated actor counts agree, with both fields available, the boundary is suppressed to avoid splitting a continuous scene. Otherwise, the accumulated observations are summarized and linked to an episode, the detector is reset, and the incoming observation starts the next run.

Aggregation and coverage. Some working events are not assigned to an episode by online boundary detection. When rendering memory, the system groups these uncovered events chronologically, starting a new group when consecutive events are more than 10 seconds apart or including the next event would extend the group beyond 45 seconds. Groups with at least five events and one change-bearing observation are retained. Adjacent retained groups are then merged when their mean field-level Jaccard similarity is at least 0.85 and their temporal gap is at most 30 seconds. This consolidates observations that describe similar content; the 45-second limit applies before merging.

The change requirement can leave long, mostly static intervals without an episodic summary, even though their observations remain in working memory. A coverage fallback therefore checks the interval after the latest episode and before the recent observations still shown individually in visible memory. If this interval spans at least 30 seconds, uncovered events are split into initial groups of at most 45 seconds, with at least two events per group, and the same similarity-based merging is applied. These groups become anchor episodes with lower base salience, preserving temporal coverage without requiring a detected change. A well-formed model-emitted ⟨Recap⟩ also creates an explicit episode and a corresponding long-term pin.

BOCPD update. Given the structured observation $O _ { t }$ at index t and $\mathbf { x } _ { t } = \phi ( O _ { t } )$ its embedding. The function ϕ hashes lowercase tokens into a 512-dimensional count vector and applies L2 normalization. The implementation updates the run-hypothesis probabilities as

$$
\widetilde { p } _ { t } ( 0 ) = H p _ { \mathrm { p r i o r } } ( \mathbf { x } _ { t } ) , \qquad \widetilde { p } _ { t } ( r + 1 ) = ( 1 - H ) p _ { t - 1 } ( r ) p ( \mathbf { x } _ { t } \mid r ) ,\tag{6}
$$

Here, t is the current observation index and $\mathbf { x } _ { t }$ is its embedding vector. $H$ is the constant probability of starting a new run, and $r$ is the nonnegative index of a previous run hypothesis. In the implementation’s indexing convention, hypothesis r contains $r + 1$ observations before the update, and new-run index zero contains the current observation alone. $p _ { t - 1 } ( r )$ is the normalized probability of the previous hypothesis; $p ( \mathbf { x } _ { t } \mid r )$ is its predictive density for the new embedding; and $p _ { \mathrm { p r i o r } } ( \mathbf { x } _ { t } )$ is the new-run predictive density. $\widetilde { p } _ { t } ( 0 )$ and $\widetilde { p } _ { t } ( r + 1 )$ are unnormalized new-run and continuation probabilities, respectively. Normalizing over retained hypotheses gives the current probability distribution $p _ { t }$ . The predictive distribution for a continuing run is

$$
p ( \mathbf { x } _ { t } \mid r ) = { \mathcal { N } } { \biggl ( } \mathbf { x } _ { t } ; { \frac { \mathbf { s } _ { r } } { n _ { r } + \kappa _ { 0 } } } , { \biggl ( } 1 + { \frac { 1 } { n _ { r } + \kappa _ { 0 } } } { \biggr ) } \mathbf { I } _ { d } { \biggr ) } , \qquad \kappa _ { 0 } = 0 . 1 ,\tag{7}
$$

where $p ( \mathbf { x } _ { t } \mid r )$ is the density of the current observation embedding $\mathbf { x } _ { t }$ under previous run hypothesis $r ,$ and t is the observation index. $\mathcal { N }$ denotes a multivariate Gaussian density whose arguments are the evaluation vector, mean, and covariance, in that order. $n _ { r }$ is the number of observations in hypothesis $r , \mathbf { s } _ { r }$ is the sum of their embedding vectors, $\kappa _ { 0 }$ is the prior mean pseudo-count, and $\mathbf { I } _ { d }$ is the identity matrix in embedding dimension $d .$

For a new run, the prior predictive density is $p _ { \mathrm { p r i o r } } ( \mathbf { x } _ { t } ) = \mathcal { N } ( \mathbf { x } _ { t } ; \mathbf { 0 } _ { d } , ( 1 + 1 / \kappa _ { 0 } ) \mathbf { I } _ { d } )$ . Here, $\mathbf { x } _ { t }$ is the current embedding, $\mathcal { N }$ is the Gaussian density, $\mathbf { 0 } _ { d }$ and $\mathbf { I } _ { d }$ are the zero vector and identity matrix in embedding dimension $d ,$ and $\kappa _ { 0 } = 0 . .$ 1 is the same prior pseudo-count. Thus, the implementation incrementally updates the run mean but uses a fixed unit observation variance with posterior-mean uncertainty, rather than estimating a separate empirical variance for every run. At most 500 run hypotheses are retained.

## A.2.2 Entity Index and Memory Rendering

The entity index records each entity’s working-event identifiers, first and most recent timestamps, and observed surface names. Entities are extracted from observation text by rule-based matching. Cross-frame references are linked conservatively through name normalization, including matching a generic head noun to a more specific compound name, such as “man” and “man in white.” A compact digest exposes frequently mentioned entities and their temporal spans.

Evidence paths. Among query-matching records, Backtrack in Eq. 3 selects at most two long-term pins, three episodes, and three working events as starting nodes, excluding question pins and planner notes. Source expansion includes at most three working events per episode. Paths receive a ranking bonus for covering multiple memory layers, are deduplicated by their starting node, and are returned in descending score order up to the requested limit. The layer quotas allow at most eight paths.

Visible memory. Recall in Eq. 3 ranks keyword-matching working records using salience, a change bonus, and recency, with an additional token-hash similarity score when cached embeddings are available. The renderer takes the highest-ranked candidates and retains those in the middle interval, excluding entries already covered by the visible head or recent tail. The selected observations are displayed chronologically. Because this filtering follows retrieval, the recalled set can contain fewer records than the requested limit.

## A.3 Planner Schema and Evidence Guidance

The implementation uses the following fields for the plan in Eq. 4.

<table><tr><td>Field</td><td>Meaning</td><td>Use in investigation</td></tr><tr><td>need</td><td>Target and its temporal anchor.</td><td>Forms the retrieval query.</td></tr><tr><td>pov</td><td>Viewpoint of the relevant participants.</td><td>First/third-person references or actors.</td></tr><tr><td>info_type</td><td>The evidence needed, such as an action order, count, or attribute.</td><td>Guides the choice of observations and tool use.</td></tr><tr><td>evidence_locus</td><td>Where the decisive evidence lies in time.</td><td>Guides retrieval of past evidence, current inspec- tion, or continued observation.</td></tr><tr><td>answerability</td><td>Whether the listed options include an inability- to-answer choice.</td><td>Guides target-existence checks and selection among the permitted answers.</td></tr><tr><td>meta</td><td>Whether the query concerns answerability or an event&#x27;s start or completion.</td><td>Distinguishes evidence of an event&#x27;s state from a description of the event itself.</td></tr><tr><td>strategy</td><td>The proposed sequence of memory access and tool use.</td><td>Provides an initial course of action that the agent can adapt to new evidence.</td></tr></table>

Query construction removes answer-option text, common instruction words, and duplicate tokens from the question and the need field. Rule-based matching interprets the free-form temporal scope, evidence type, and strategy to derive an evidence category and tool-use guidance. Appendix E.3 provides the planning instruction and the runtime template for persistent plan fields and the evidence contract.

## A.4 Evidence-Gathering Tools

We expose each tool through a shared description template specifying its name, purpose, and JSON argument schema. To request execution, the agent emits a <tool\_call> block with name and arguments fields, then waits for the result before continuing. The shared call template and instructions for interpreting results and selecting subsequent tools are reproduced in Appendix E.4.

The tools in Tab. 7 use SAM2 [39] for object tracking and Whisper [38] for speech transcription. Post-query model calls receive a single current frame, with additional images supplied by spatial tools. Tool rounds can continue at the same frame, whereas continued observation advances to new frames. Interpretations of too results also update memory.

Tools return a <tool\_result> block containing their name, success status, and result or error. A successful zoom additionally supplies the crop as an image input to the following model call. Replay returns the selected replay time and schedules historical-frame inspection through the temporal controller. When the seek is applied, the historical frame becomes the visual input, and the previous stream position is retained for return.

Tab. 7 Tool suite of DeltaStreamer, grouped by function.
<table><tr><td>Group</td><td>Tool</td><td>Description</td></tr><tr><td></td><td>memory_search</td><td>Cross-layer keyword retrieval over all memory DAG layers</td></tr><tr><td>Memory</td><td>memory_read</td><td>Retrieve specific events by query with time-range filtering</td></tr><tr><td></td><td>memory_list</td><td>List events filtered by kind and time range, chronologically</td></tr><tr><td></td><td>memory_write</td><td>Write a model-selected fact into memory</td></tr><tr><td></td><td>track_init</td><td>Initialize a SAM2 object tracker on a box or prompt</td></tr><tr><td>Track</td><td>track_query</td><td>Query the current position and confidence of a tracked object</td></tr><tr><td></td><td>track_end</td><td>Terminate a tracking session and obtain trajectory summary</td></tr><tr><td>Audio</td><td>asr_segment</td><td>Transcribe speech in a time window using Whisper ASR</td></tr><tr><td>Temporal</td><td>replay</td><td>Schedule a past frame for visual inspection, retaining the current stream position for return. Recover visual details omitted from textual memory.</td></tr><tr><td>Spatial</td><td>zoom_crop</td><td>Crop and zoom into a sub-region of the current frame with IoU-based deduplication to prevent redundant zooming.</td></tr><tr><td>Efficiency</td><td>set_sample_rate</td><td>Adjust the observation frame rate (0.25–2.0 fps) during static periods to save computation.</td></tr></table>

## A.5 Look-Back Verification and Replay Details

Replaying historical frames incurs substantial overhead through additional frame access and visual processing, which can delay responses. We therefore use the checks and priority rules below to assess whether unresolved conflicts justify this cost before triggering automatic replay.

Claim extraction and initial checks. When invoked, the verifier requests up to 16 evidence paths, subject to the layer quotas described in Appendix A.2. Claim extraction uses fixed text patterns for explicit state tags, location phrases, a limited set of attributes, and numerical counts. Subject and property strings are lowercased, and claims about the same subject and property are grouped. The following preliminary rules are applied in order. The first matching rule skips subsequent cross-layer and temporal checks for that group and triggers no automatic replay. These outcomes concern the need for replay and do not establish that the proposed answer is visually correct.

1. A group with fewer than two claims is marked uncertain and does not trigger replay.

2. If working and episodic records contain the same set of values but long-term records differ, the group passes the initial check.

3. If every gap between consecutive claims in temporal order is at most 2 seconds on the stored 1-fps timeline, the group passes the temporal-proximity check.

4. If all claims have the same value, the group passes the agreement check.

5. If at least 85% of claims share one value, the group is marked uncertain without replay.

Conflict severity and target selection. Groups unresolved by the preliminary checks are examined for disagreement across memory layers and state transitions that may need visual confirmation. The cross-layer check gives greater weight to records closer to the original observations. The temporal check can flag transitions that may reflect missing intermediate evidence, even when the cross-layer check requests no replay. Either check can nominate a replay candidate, whose severity reflects the stronger signal. For a cross-layer conflict, the verifier targets the earliest supporting working observation when available. A flagged temporal transition supplies its earlier endpoint if no target is selected. The scheduler then ranks these candidates to decide whether and where to look back.

Replay priority. For a conflict u with severity $c _ { u }$ and replay index $\tau _ { u } .$ , the implemented priority in Eq. 5 is

$$
\rho ( u ) = \left[ p _ { \mathrm { r e p } } - \operatorname* { m a x } ( 0 , 1 - c _ { u } ) \right] \operatorname* { m a x } ( d _ { \mathrm { m i n } } , 1 - \lambda \left| { \mathrm { t i m e } } ( \tau _ { u } ) - { \mathrm { t i m e } } ( t _ { q } ) \right| ) ,\tag{8}
$$

Here, u is an unresolved conflict, $\rho ( u )$ is its replay priority, $c _ { u }$ is its conflict-severity score, and $\tau _ { u }$ is its target sampled-frame index. $t _ { q }$ is the question-arrival frame index; time maps frame indices to video timestamps in seconds, and the absolute-value term is their temporal separation. $p _ { \mathrm { r e p } }$ is the scheduler’s assumed correctness after replay; max $\left( 0 , 1 - c _ { u } \right)$ is its heuristic estimate without replay. $d _ { \mathrm { m i n } }$ is the minimum temporal weighting factor and λ is its decay rate per second. Defaults are $p _ { \mathrm { r e p } } = 0 . 9 5 , d _ { \mathrm { m i n } } = 0 . 5 .$ , and $\lambda = 0 . 0 2 { \mathrm { s } } ^ { - 1 }$ . Candidates with $\rho ( u ) \leq \eta$ are discarded, where $\eta = 0 . 1 5$ is the minimum priority required to justify replay cost; the remainder are sorted by descending priority, deduplicated by target, and capped at three.

## B Benchmark Details

Evaluation protocol. Query times beyond the supplied video segment are mapped to the last available frame. Planning and finalization calls can reuse the last available frame without exposing additional video. Accepted responses and their trigger times are read from the recorded outputs, and task-specific parsers extract the required answer format. Question and response times refer to video timestamps rather than wall-clock inference time.

## B.1 OVO-Bench

OVO-Bench [33] contains 3035 evaluation samples spanning 12 tasks in three groups. The Real-Time Visual Perception group (837 samples) includes Optical Character Recognition (OCR), Action Recognition (ACR), Attribute Recognition (ATR), Spatial Understanding (STU), Future Prediction (FPD), and Object Recognition (OJR). The Backward Tracing group (631 samples) includes Episodic Memory (EPM), Action Sequence Identification (ASI), and Hallucination Detection (HLD). The Forward Active Responding group (1567 samples)

![](images/13ea6bed25e96185395325fe4c4e5ad4e4b6a67cdd90a3e9cbd34c75ed93c02d.jpg)  
Fig. 5 Visible memory (a) and tool use (b) on OVO-Bench and StreamingBench.

includes Repetition Event Count (REC), Sequential Steps Recognition (SSR), and Clues Reveal Responding (CRR).

## B.2 StreamingBench

StreamingBench [28] contains 4500 QA samples in three groups. The Real-Time Visual Understanding group covers Object Perception (OP), Causal Reasoning (CR), Clips Summarization (CS), Attribute Perception (ATP), Event Understanding (EU), Text-Rich Understanding (TR), Prospective Reasoning (PR), Spatial Understanding (SU), Action Perception (ACP), and Counting (CT). The Omni-Source Understanding group covers Emotion Recognition (ER), Scene Understanding (SCU), Source Discrimination (SD), and Multimodal Alignment (MA). The Contextual Understanding group covers Anomaly Context Understanding (ACU), Misleading Context Understanding (MCU), Sequential Question Answering (SQA), and Proactive Output (PO).

## C Memory and Tool Use

Fig. 5 summarizes memory and tool-use patterns on OVO-Bench and StreamingBench. Blue denotes OVO-Bench, and orange denotes StreamingBench throughout the figure. The gray legend symbols describe plotting conventions shared by both benchmarks.

Visible memory. We measure the length of the memory text supplied to the model and align each trajectory by its observation progress up to query arrival. In Fig. 5(a), solid lines show the median memory length across trajectories, and shaded bands span the 25th–75th percentiles. The circles at 100% mark the median memory length in the first query-time model input, with vertical whiskers showing the corresponding 25th–75th percentile range. Memory grows quickly during the initial observation period, after which its growth slows substantially on OVO-Bench and largely stabilizes on StreamingBench. This pattern is consistent with selective rendering and episodic compression keeping the visible context under control as observations accumulate. The measurement concerns the rendered input memory, not the total storage of records and historical frames.

Replay frequency. In Fig. 5(b), each tool has one bar per benchmark. Light and dark segments show mean dispatched calls per question before and after query arrival, respectively. The number at each bar end reports their sum. Low replay frequency is expected given its selective role in recovering evidence during question answering. Revisiting historical frames requires additional frame access and visual processing, so frequent replay can increase response overhead and interfere with real-time processing. The agent is instructed to answer from the current frame or retained memory when sufficient and to revisit a past moment when visual confirmation is needed. Replay is therefore concentrated in the answering phase, whereas observation tools are used throughou the stream. In the recorded trajectories, the last replay is typically followed by a short remaining sequence of agent steps before completion. This limited answering-stage window, together with tool budgets and termination rules, helps explain the low frequency of repeated replay and reflects the trade-off between additional inspection and timely responses.

Tab. 8 Observation, memory, and planning statistics. All values are means.
<table><tr><td>Metric</td><td>OVO-Bench</td><td>StreamingBench</td></tr><tr><td colspan="3">Observation before query</td></tr><tr><td>Visual observations per sample</td><td>70.85</td><td>118.31</td></tr><tr><td>Observation length (chars.)</td><td>1,279.15</td><td>1,588.48</td></tr><tr><td>Delta length (chars.)</td><td>155.54</td><td>172.98</td></tr><tr><td colspan="3">Memory at completion</td></tr><tr><td>Stored working records</td><td>131.91</td><td>261.71</td></tr><tr><td>Episodes</td><td>11.68</td><td>16.08</td></tr><tr><td colspan="3">Planning</td></tr><tr><td>Initial plan length (chars.)</td><td>1,329.43</td><td>1,299.82</td></tr></table>

Observation, memory, and planning. Tab. 8 summarizes the information recorded during observation and investigation. We count pre-query visual observations containing both description and delta. Text lengths are measured in Unicode characters; observation and delta lengths are averaged within each sample and then across samples with observations. Stored working records and episodes are counted at completion, with working records including observations, frame references, and tool-related entries. Plan length refers to the first generated plan.

These statistics complement the visible-memory curves by describing what is recorded and retained. Delta occupies only part of each observation while making changes explicit. The memory retains detailed working records together with a smaller set of episodic summaries, and the initial plan states the evidence needs that guide subsequent investigation.

## D Discussions

## D.1 Limitations

DeltaStreamer requires repeated model and tool calls, which may increase inference latency, although frame arrival and processing can proceed asynchronously in real-world streaming settings. Compact visible memory also leaves storage costs for records and historical frames. Errors in observations or temporal anchors can affect retrieval, while agreement across memory layers does not ensure perfect correctness. Our evaluation uses one shared backbone on two benchmarks and leaves challenges in historical event localization, sequential-step recognition, and proactive response timing. The weaker PO results on StreamingBench highlight the additional challenge of precise response triggering, since retrospective evidence recovery cannot compensate for a missed response window. Further work should examine other backbones and the trade-offs among accuracy, latency, and storage.

## D.2 Broader Impacts

DeltaStreamer may support accessibility tools and help users review extended activities by retrieving relevant historical evidence. However, continuous observation, structured memory, and replay can expose personal routines and facilitate non-consensual monitoring. Errors or biases may propagate through stored descriptions and answers, misrepresenting people’s actions. For deployment, we recommend informed consent from recorded individuals, limited retention, access controls, and deletion mechanisms covering both frames and derived memories. Human review of the underlying evidence should precede consequential decisions about individuals These applications and the effectiveness of the recommended safeguards have not been evaluated in this work.

## E Prompt Templates

## E.1 Pre-Query Observation

The two-line instruction applies to the ordinary observation report followed by </Silence>. Tool calls occupy intermediate turns, after which the agent is prompted again on the same frame. A conditional <Recap> block may precede </Silence>, as described in Appendix E.2. The prompt requires a concrete change while forbidding invented changes, with exceptions for the initial frame and verified identical frames.

Observation prompt   
[Phase] OBSERVE-ONLY. The question has not been asked yet.   
Reply with EXACTLY two lines:   
Line 1: <Observation> a SELF-CONTAINED FULL frame report covering ALL SEVEN labelled fields   
below, in this order, on EVERY frame (no exceptions, no abbreviations, no same as before, no bare (no   
change)):   
description: 1-2 sentences, holistic summary of THIS frame.   
BAD: "A kitchen scene."   
GOOD: "Medium shot of a chef in white uniform standing at a stainless-steel counter, chopping   
carrots with a chef’s knife. Steam rising from a pot on the stove in the background."   
actors: EVERY distinct person/animal visible. For each: appearance, clothing, role, position. Re-list   
ALL actors every frame. Include background figures.   
Always state the TOTAL count (e.g. ’2 actors: ...’).   
BAD: "actors: a person cooking"   
GOOD: "actors: 1) chef in white double-breasted jacket, black apron, toque, at center counter —   
chopping; 2) sous-chef in blue polo, at back sink — washing dishes"   
objects: salient objects with current STATE and LOCATION. Tag state explicitly.   
For COUNTABLE objects, report the EXACT number (e.g. ’3 cards’, ’5 bottles’) — never vague ranges   
like ’approximately 3-5’ or ’several’.   
Format: <object> [state] at <location>   
BAD: "objects: knife, pot, carrots"   
GOOD: "objects: chef’s knife [held, in use] in chef’s right hand; large stockpot [on stove, steaming]   
back left; 3 carrots [being chopped] on cutting board center frame"   
actions: what each actor is CURRENTLY doing (verb + target + state tag).   
Format: <actor> <verb> <target> [STATE]   
State tags: [STARTING] / [IN-PROGRESS] / [FINISHED] / [ABORTED]   
BAD: "actions: cooking"   
GOOD: "actions: chef chopping carrots [IN-PROGRESS]; sous-chef washing dishes   
[IN-PROGRESS]"   
For persisting actions, use still ... or continues to ... instead of re-describing.   
ocr: EVERY on-screen text/number VERBATIM. Include language, color, position.   
BAD: "ocr: some text on screen"   
GOOD: "ocr: ’CHANNEL 5 NEWS’ (white text, top-left overlay); scoreboard reads ’HOME 24 - 18   
AWAY’ (yellow, bottom-right); subtitle: ’What are you doing?’ (white, bottom-center)"   
If genuinely no text: "ocr: (none)"

## Observation prompt (continued)

scene: location + time-of-day + camera framing + indoor/outdoor.

BAD: "scene: indoors"

GOOD: "scene: professional kitchen, daytime (fluorescent lighting), medium-wide shot, eye-level angle, indoor"

delta: what CHANGED from the PREVIOUS frame. This is the MOST IMPORTANT field — it drives memory compression and retrieval.

MUST name at least one concrete, observable change.

Format: <subject> <verb/delta> <detail>

BAD: "delta: slight motion" (generic boilerplate)

BAD: "delta: no change" (NEVER allowed)

BAD: "delta: scene continues" (garbage token)

GOOD: "delta: chef’s right hand moved 3cm left, knife blade angle changed from 30 degrees to 45 degrees"

GOOD: "delta: steam plume above stockpot expanded upward approximately 5cm; new bubble formation visible"

GOOD: "delta: sous-chef turned head 90 degrees left toward chef; mouth opened (speaking)"

If you GENUINELY cannot detect any change after active visual inspection, pick a tool (zoom\_crop / track\_query) to find evidence first.

ONLY if the frame is bit-identical may you write: "delta: visually identical to t={prev}s — verified static".</Observation>

CRITICAL when at least two entities of the same kind are visible: COUNT them exactly (e.g. 5 giants, 3 dogs, 2 cars) and describe their relative spatial order

(front/back/left/right/foreground/background, or in a row from left to right: A, B, C style). Identify which one is biggest / closest / leftmost — this is exactly the kind of fact a future question may rely on.

Line 2: </Silence> (status tag meaning "observed, no answer yet").

Do NOT emit </Response>. The system reads <Observation> into MEMORY and uses </Silence> as the round-end marker.

## Anti-Hallucination Rule

• Every delta: field MUST reference SPECIFIC VISIBLE elements that YOU CAN SEE in the CURRENT frame(s).

• Do NOT speculate about off-screen events or future actions.

• Do NOT infer intentions unless there is explicit visual evidence (e.g., a facial expression, a pointing gesture).

• If uncertain about a detail (count, color, identity), write [uncertain: <detail>] rather than guessing.

• For ANY claim involving a COUNT or COLOR, visually re-examine the frame before writing — do not trust memory.

• The delta MUST describe a change you SEE between the two frames; do not invent changes to satisfy the format.

## Initial-frame instruction

[Round 0 — initial frame, no previous frame to compare]

Write a full observation; set delta to ’(initial frame)’.

Fig. 6 Observation-prompt excerpts, including the seven report fields and visual-grounding rules, followed by the separate initial-frame instruction. The literal {prev} is retained from the static-frame example.

## E.2 Memory Summarization and Online Recaps

The following templates are separate text-only summarization calls; their scheduling and memory links are described in Appendix A.2. Here, {t0} and {t1} are interval endpoints, {len(run)} is the number of input observations, and {body} is their text.

![](images/743d32a352187d2e7887a544d59fa4f0d03d76551cfa3411e4c81d50d349200e.jpg)  
Fig. 7 Complete system and user templates for the two auxiliary memory-summarization calls.

The Recap instruction applies at semantic boundaries within frame-processing turns before or after query arrival. It does not require a separate summarization call.  
![](images/98236794abdf17127e4561c70b749d4aafa0dd5e7c70280ea2e585a73345a2ba.jpg)

![](images/47a87f3eaa720b1783bee495b6b5a3be67caed5c9ca2e4b5736656fbc5d17055.jpg)  
Fig. 8 Instruction excerpt for Recap blocks emitted by the streaming agent at semantic boundaries.

## E.3 Query-Driven Planning

The complete planning-phase instruction is followed by its acknowledgement and the template inserted into subsequent answering calls.

![](images/2aa3b02e15980cdb054f7afdfbb84c0a98d09bf56e1d6947a3ebd1f3814bee03.jpg)

## Planning prompt (continued)

## evidence\_locus:

<work out WHERE the deciding evidence for THIS question lives — in your own words, no default direction. Weigh three possibilities even-handedly and pick whichever the question truly calls for:

(i) it is in the CURRENT frame / the moment the question is asked (what is happening or visible right now);

(ii) it is in something that has ALREADY happened (recorded in MEMORY / a specific past moment to re-look);

(iii) it has NOT happened yet and you must keep observing until it does.

Do not assume it is in the past just because MEMORY is handy, nor in the present just because a frame is in front of you.

Check the ANCHOR the question pivots on (e.g. the event in "what happens AFTER X" / "before X"): has that anchor — and the thing asked relative to it — actually appeared yet? If not, the evidence is not in hand and answering now would just guess from whatever is nearest.

Then say how you’ll obtain it: look at the current frame, read/search a MEMORY layer (for what terms), replay a specific moment (which t and why), or keep watching.

If the target is hard to observe directly (occluded, off-screen, or your own body), infer it INDIRECTLY from observable traces — how the objects it acts on changed, the before/after scene difference, or the effect it left>

## answerability:

<look at the ACTUAL listed OPTIONS (A/B/C/D...) as a CLOSED set — your final answer MUST be one of them verbatim.

Check whether one of THOSE listed options literally says "Unable to answer" / "cannot determine" (or similar). Do NOT count the generic instruction to "give a best-guess if uncertain" as such an option — that is NOT a choice, and there is no "implied" unable option.

If such a choice IS listed: your first job is to VERIFY the target exists before locating it, and you may pick it when the evidence truly shows the target is absent.

If NO such choice is listed: refusing is FORBIDDEN — even with weak evidence you must commit the single MOST LIKELY listed option, never "Unable to answer". State here which case this is>

## meta:

<is the question asking whether something CAN be answered / WILL happen / has STARTED-vs-FINISHED (a meta/state question), rather than a plain fact? yes/no + why>

## strategy:

<turn your evidence\_locus reasoning into a concrete plan: given WHERE you decided the evidence lives, spell out which memory layer to read/search (and for what terms), whether the current frame suffices, and which exact t to replay if you need to confirm.

E.g. for an action-order question: locate the exact t of the anchor action, then read the NEXT action change after it.

For a direction question: compare two frames a few seconds apart, do NOT track.

For an existence question: search once or twice; if empty AND "Unable to answer" is a LISTED option, answer it — otherwise commit the most likely listed option.

IMPORTANT — if info\_type is a CURRENT-FRAME percept (what the person is doing RIGHT NOW, an on-screen object/text/attribute, an immediate action): the answer is in the CURRENT frame — LOOK

## Planning prompt (continued)

at it and answer directly.

Do NOT replay the current moment (it just re-shows the same frame), do NOT browse other frames, and use at most ONE zoom\_crop if a detail is unclear.

Commit quickly — over-using tools on a see-it-now question adds noise and flips close options (e.g. connecting vs disconnecting).>

</Plan>

End this turn with </Silence> right after the </Plan> block (you will be asked again immediately to act on this plan).

The acknowledgement confirms that the plan was parsed. The stored plan is supplied separately in later calls.

## Runtime acknowledgement

<tool\_result>{"name": "\_plan", "ok": true, "data": {"note": "Plan recorded. Now ACT on it: gather the evidence your strategy named, then answer. It will be shown to you each turn."}}</tool\_result>

Fig. 9 Complete planning-phase instruction with all seven fields, followed by a runtime acknowledgement of the parsed plan.

In the runtime template below, angle-bracketed values are editorial placeholders. Nonempty plan fields contain the model’s analysis; the contract’s evidence\_locus is a rule-derived category. The template illustrates the format, not a recorded execution.

## Persistent plan and evidence guidance (runtime template)

[PLAN] Your own analysis of this question — FOLLOW IT:

\- need: <stored need>

\- pov: <stored pov>

\- info\_type: <stored info\_type>

\- evidence\_locus: <stored evidence\_locus>

\- answerability: <stored answerability>

\- meta: <stored meta>

\- strategy: <stored strategy>

[EVIDENCE CONTRACT — compiled from your plan]

\- retrieval\_query: <question and planned-need keywords>

\- evidence\_locus: <current, future, past, or mixed>

\- tool\_policy: <guidance selected from the temporal category and evidence type>

decision\_rule: answer only from evidence matching the target and anchor; options are output choices, not retrieval evidence.

Fig. 10 Runtime template for persistent plan fields and the derived evidence contract. Placeholder values identify dynamic content rather than a recorded execution.

## E.4 Evidence Gathering and Answering

These excerpts omit task-specific inputs and additional runtime reminders.

For fixed-choice questions, the abstain instructions permit an “Unable to answer” response only when such an option is explicitly listed. Otherwise, the agent must select a listed choice, as specified by the planner’s

## Tool use and question arrival

## Tool-call protocol

Whenever you want to call a tool, emit exactly the following block as part of your response (you may emit multiple such blocks):

<tool\_call>{"name": "<tool\_name>", "arguments": {...}}</tool\_call>

After the tool block(s), STOP. The system will run the tools and feed <tool\_result>{...}</tool\_result> back to you. Only after you have enough evidence, finish with </Response> <your final answer>.

## Question-arrival instruction

[Phase] QUESTION POSED. The question below is being asked NOW (and has been pinned to MEMORY).

## Read-the-question protocol (BEFORE deciding)

Silently parse the question text into:

• TARGET: what is being asked (an entity, an action, a count, a Yes/No truth value, an on-screen text string, ...).

• ANCHOR: the time / event / actor it pivots on (e.g. right after loading the wheel, the person we communicated with while chopping, under the monkey at this moment).

• FORMAT: option letter, Yes/No, integer 0..N, free-form span — match the wording exactly.

Then check MEMORY for events that match the ANCHOR; the TARGET is whatever is reported in those events.

If MEMORY only contains the ANCHOR but not the TARGET, the answer probably lies a few frames later — emit </Silence> and wait, OR replay to a likely frame.

Decide whether you can answer with the evidence accumulated so far (see MEMORY) or need to watch more frames (emit </Silence> to wait).

GROUND YOUR ANSWER IN PIXELS WHEN IT HELPS: if the answer hinges on a SPECIFIC PAST moment (an earlier action / object state / who was present), replay(t=...) that moment and verify with your own eyes before committing.

YOU decide whether a replay is needed and how many: questions about the CURRENT frame usually need none; questions about the past may need one or a few. Do not over-replay when the current frame already answers it.

BEFORE answering, prefer a quick tool call if it would resolve uncertainty:

• To re-examine an earlier moment, use replay.

• To re-read your notes, use memory\_read or memory\_list.

• To inspect fine detail, use zoom\_crop.

• If there is genuinely no visual evidence at all, abstain.

One or two well-chosen tool calls almost always beat a blind guess.

## Continued observation and evidence gathering

[Phase] POST-QUESTION. The question is pinned in MEMORY (look for the [Q] ... entry). Continue watching the new frame.

## Continued observation and evidence gathering (continued)

## Read-the-question protocol (every round)

RE-READ the pinned [Q] ... line and parse it into TARGET / ANCHOR / FORMAT (see QUESTION POSED phase). Then ask: does the latest frame OR any new MEMORY event satisfy the ANCHOR? If yes, the TARGET is the fact reported there.

First-Person POV: when the camera shows bare feet/legs/torso at the bottom of the frame, the camera wearer is ’I’. A question asking ’who did I communicate TO’ or ’who did I interact WITH’ is asking about the OTHER person — NOT the one visible through the first-person perspective (that is ’I’).

Trace both people’s timelines in MEMORY: the camera-wearer (’I’) appears as partial body at frame bottom; the interaction partner is a fully-visible actor.

If the answer depends on a SPECIFIC PAST frame, replay(t=...) that moment to CONFIRM with your eyes first; if the CURRENT frame or MEMORY already answers it clearly, just emit </Response> <answer>. Otherwise emit </Silence>.

Use replay WHEN a past moment needs verifying, but YOU decide if it is needed and how many; don’t replay when the current frame already settles the answer.

If something in MEMORY is fuzzy or this frame hints at the answer, use a tool BEFORE answering replay to revisit a past moment, zoom\_crop for detail, memory\_read to recall a pinned fact, asr\_segment for audio, or abstain if there is genuinely no evidence at all.

CRITICAL: After EVERY tool result, FIRST write a 1-2 line observation of what the tool revealed (especially: counts, text, identities, state changes) BEFORE you decide to call another tool.

Do NOT chain tools blindly — each tool call must be justified by a specific gap in what you have ALREADY observed.

If the tool result confirms what you need, COMMIT with </Response> NOW instead of calling more tools. A zoom\_crop that returns a clear image of the target region IS ENOUGH — write your observation and answer.

If THIS frame marks a semantic boundary (event ended / scene change / actor enter-exit-change / key object change), ALSO emit a <Recap t0=<sec> t1=<sec> kind=<...>> ... </Recap> block (3–8 sentences, see OBSERVE-ONLY phase instructions for Recap format) — the system pins it as durable memory.

## How to use the [RETRIEVED] block

The [RETRIEVED ...] block above is the system’s best-guess evidence for your question. HOW TO USE IT:

• Each path#N shows a chain through memory layers (from wk to ep to lt). Follow the chain — the working-layer event is closest to the original pixels.

• Cross-check: if a fact appears consistently across working + episodic, it is reliable. If only in long\_term, verify via replay.

• It is NOT exhaustive — use memory\_search with different keywords if needed.

## Tool selection (use BEFORE each call)

1. For a MOVING subject across frames, use track\_init (NOT zoom\_crop).

2. If the answer depends on a PAST frame, use memory\_search first; if detail is missing, use replay.

3. If the cue is likely AUDIO, use asr\_segment.

4. For STATIC fine detail in the CURRENT frame (text, label, logo), use zoom\_crop.

5. To remember a durable fact, use memory\_write(pin=true).

Anti-pattern: after 2 or more zoom\_crops with no new fact, switch tools.

Fig. 11 Excerpts of the shared tool-call protocol and post-query instructions for evidence gathering, visual verification, and answering.

## F Additional Qualitative Examples

## <sup>1</sup> W<sup>h</sup>at co<sup>l</sup>or <sup>i</sup>s t<sup>h</sup>e s<sup>h</sup>opp<sup>i</sup>ng <sup>b</sup>as<sup>k</sup>et<sup>?</sup>

![](images/0c425e126b9df3590e115a0fe6980d0b81caee323f5abd673f727cb9675b7161.jpg)  
Fig. 12 OVO-Bench examples of temporal reasoning, attribute recall, and checking query premises.

![](images/ee14fa8c04d4ca1426eeff06147cd3d2beb58ad15e88f0c5f250b8b75f2f7322.jpg)  
Fig. 13 StreamingBench examples of action sequencing, scene changes, and visual reasoning.

## <sup>11</sup> W<sup>h</sup>at <sup>i</sup>s t<sup>h</sup>e grey <sup>i</sup>tem on t<sup>h</sup>e <sup>b</sup>e<sup>d?</sup>

![](images/67b5834980915c31def6078578790e64cd49afa7cb97e3e6aaf7a4712ab9fea8.jpg)  
Fig. 14 Some examples of using replay or historical crops.