# Finder: Agentic Closed-Loop Object Finding for Embodied Grounding

Shixiong Xu Zhiyuan Chen Song Ding Rui Luo Xiaowei Liang Dongxu Miao

Zhiying Du

Xiaomi Robotics

Abstract: Finding the object referred to by language in a partially observed 3D scene is a core capability for embodied agents. Existing approaches either couple object search with online exploration, which can be costly when relevant observations have already been captured, or query pre-built open-vocabulary maps and scene graphs in a static, one-shot fashion. We present Finder, an agentic closed-loop object-finding primitive for embodied grounding. Instead of treating grounding as passive retrieval from a fixed scene representation, Finder maintains a typed loop state that links query-conditioned planning, scoped evidence gathering, candidate verification, and accept/continue/abort control. When evidence is incomplete or ambiguous, the loop can redirect subsequent perception and comparison rather than simply returning the top retrieved object. On open-vocabulary embodied Object Retrieval in Habitat/HM3D and real-world RGB-D scenes, Finder improves the averaged 1m success rate by 15.75 points over strong baselines. The same primitive also transfers to sequential object grounding and embodied object-centric question answering, improving spatial and temporal localization without changing the inner grounding protocol. Project page: https://finder-vln.github.io.

Keywords: Agentic Systems, Object Grounding, Open-Vocabulary Perception

## 1 Introduction

For embodied agents, objects are not just entities in a scene; they are anchors for perception, reasoning, and action. From navigation and manipulation to embodied question answering and long-horizon task execution, many downstream behaviors are organized around identifying, localizing, and revisiting task-relevant objects [1, 2, 3, 4, 5, 6]. Some systems bind object search to online exploration, which can be expensive when the relevant evidence is already present in captured observations. Many open-vocabulary systems instead use a representation-centric pipeline: they first build or maintain a semantic map or scene graph from RGB-D observations, and then treat grounding as one-shot retrieval or reranking over that representation [7, 8, 1, 9, 10, 11, 12, 13].

This paradigm can work well when the target is visually distinctive and the relevant evidence is already present in the stored scene representation. However, many embodied queries expose three recurring limitations of existing pipelines: resolving instance ambiguity, handling support grounding, and making loop-control decisions explicit. Instance ambiguity arises when multiple same-category objects are present and the top-ranked match is not necessarily the correct instance [14, 15]. Support grounding matters for relational queries, where supporting objects must be used selectively as disambiguating evidence rather than folded into text matching [13, 16]. Implicit loop control arises when the current evidence is insufficient and the system must decide what to verify next, where to search next, and whether another round is worthwhile. In most existing pipelines, these decisions remain implicit, and information from earlier attempts is difficult to carry forward [17].

![](images/66e414aaeba396596fc0a173d55a3565c290dc4969da113325ea7b784e844d08.jpg)  
Figure 1: Overview of Finder. Object grounding is a core embodied primitive requiring multi-source context. Finder realizes it as a reusable closed-loop inner protocol, wrapped by an outer harness for task- and session-level context injection.

Therefore, we argue that embodied systems need a reusable agentic closed-loop primitive for object finding. Such a primitive should not merely score candidates from a fixed scene representation. Instead, it should support an agentic loop that decides what evidence to gather next in task context (Figure 1, left). This abstraction makes grounding both analyzable and reusable, while allowing downstream tasks to inject task- and session-level context without redesigning the inner loop itself.

We instantiate this primitive as Finder, shown in the center of Figure 1, which organizes one grounding episode as a five-stage loop: plan → detect → selection plan → select → judge. The planner proposes a primary target, optional supporting objects, and a scoped search budget, so the system can decide what to look for and where to gather evidence next. Candidate-centric context and support-aware verification address instance ambiguity and relational disambiguation, while the final judge determines whether to accept, continue, or abort. In this way, Finder turns object grounding from one-shot retrieval into a query-conditioned loop over evidence inspection, candidate comparison, and explicit accept/continue/abort control. The same loop can be reused through a lightweight outer harness as shown in Figure 1: downstream tasks inject resolved-object history, room hints, or benchmark constraints through the request interface, while the inner protocol remains unchanged.

Across Habitat/HM3D [18, 19] and real-world RGB-D retrieval benchmarks, Finder improves averaged 1m success rate by 15.75 points over strong baselines. The same primitive also transfers effectively to sequential object grounding and embodied object-centric question answering [20, 6].

In summary, our contributions are:

• Finder, a reusable agentic closed-loop object-finding primitive that makes evidence gathering, candidate comparison, and accept/continue/abort control explicit.

• A lightweight outer harness for task- and session-level context management, including resolvedobject history, context injection, and optional memory hints, without changing the inner loop.

• Extensive experiments demonstrate the effectiveness of Finder on open-vocabulary object retrieval, sequential object grounding, and embodied object-centric question answering.

## 2 Related work

Open-vocabulary 3D scene understanding. Early semantic mapping systems operated over closed set categories, limiting their applicability to household environments where the space of possible objects is unbounded [8]. Vision-language models such as CLIP enabled open-vocabulary 3D representations, including OpenScene [7], ConceptGraphs [8], HOV-SG [1], Hydra [21], and Open3DSG [10], with OVO [22], ZING-3D [23], and Clio [11] moving toward online, incremental, or task-driven variants. Yet these methods still require an exhaustive pre-built or continuously maintained scene representation, which becomes harder under dynamic changes. DualMap [9], OpenIN [24], Khronos [25], DovSG [26], DynamicGSG [27], and INHerit-SG [17] address this with locally updatable object layers or spatio-temporal restructuring, but still rely on hand-designed update rules that are hard to generalize across diverse environments. Finder departs from this paradigm by using an agentic loop to decide which query-relevant evidence should be inspected next, rather than passively querying a fixed representation or relying on hand-designed update rules.

Agentic systems for embodied AI. The integration of large language and vision-language models as decision-making agents has gained traction in embodied AI [28]. SayPlan [2] and LLM-Grounder [13] ground LLMs in 3D scene graphs for task planning and visual grounding respectively. SG-Nav [29] prompts LLMs with online scene graphs for zero-shot object navigation. RoboMemory [30], Re-MEmbR [16], Mem2Ego [31], EmbodiedRAG [14], Affordance-RAG [32], MALLVi [33], and 3DSPMR [5] use memory, retrieval, or multi-agent reasoning to support long-horizon navigation, manipulation, and sequential tasks. These systems are primarily designed for manipulation or navigation, and do not address task-driven open-vocabulary retrieval with closed-loop evidence refinement. Finder instead isolates object finding itself as the reusable unit: an explicit inner loop over planning, evidence gathering, candidate comparison, and accept/continue/abort control that can be wrapped by different downstream tasks.

Discussion. The works above advance open-vocabulary perception, memory-augmented reasoning, and embodied planning, but most still build or maintain a scene graph first and then consume it passively. Few systems expose target grounding itself as a reusable closed-loop primitive with explicit state, retry logic, and task-conditioned evidence allocation; Finder is designed to close this gap.

## 3 Method

## 3.1 Problem Setup and Finder Overview

Problem formulation. Let $\mathcal { V } = \{ ( I _ { t } , D _ { t } , P _ { t } ) \} _ { t = 1 } ^ { T }$ denote a stream of RGB-D observations, where $I _ { t } \in \mathbb { R } ^ { H \times W \times 3 }$ is the color image, $D _ { t } \in \mathbb { R } ^ { H \times W }$ is the depth map, and $P _ { t } \in S E ( 3 )$ is the camera pose at time t. Given a language-conditioned request $q ,$ optional injected context $m .$ , and a finite perception budget, Finder iteratively proposes grounded object hypotheses and decides whether to accept the current 3D hypothesis, continue gathering evidence, or abort.

Closed-loop protocol. As shown in Figure 2, Finder executes a five-stage closed loop: plan → detect → selection plan → select → judge. Here, request and context injection are treated as pre-loop inputs rather than inner-loop stages. Within this five-stage view, plan includes request understanding, primary/support target extraction, phrase generation, scoped routing, and frame-budget allocation, while selection plan includes candidate-context construction and view allocation for verification. Only the final judge stage can terminate the episode or trigger another loop, which makes the stopping boundary explicit and keeps module replacement local. We summarize one loop as

$$
( p _ { \ell } , F _ { \ell } ) = \mathrm { P l a n } ( q , m , s _ { \ell - 1 } , \mathcal { V } ) ,\tag{1}
$$

$$
C _ { \ell } = \mathrm { D e t e c t } ( F _ { \ell } , p _ { \ell } ) ,\tag{2}
$$

$$
\begin{array} { r } { ( u _ { \ell } , E _ { \ell } ) = \mathrm { S e l P l a n } ( C _ { \ell } , p _ { \ell } , s _ { \ell - 1 } ) , } \end{array}\tag{3}
$$

$$
\hat { o } _ { \ell } = \mathrm { S e l e c t } ( u _ { \ell } , E _ { \ell } ) ,\tag{4}
$$

$$
\begin{array} { r } { ( a _ { \ell } , h _ { \ell } ) = \mathrm { J u d g e } ( \hat { o } _ { \ell } , E _ { \ell } , s _ { \ell - 1 } ) , } \end{array}\tag{5}
$$

where $p _ { \ell }$ is the loop search plan, $F _ { \ell }$ is the routed frame subset, $C _ { \ell }$ is the set of canonicalized candidates, $u _ { \ell }$ is the selection plan, $E _ { \ell }$ is the candidate-centric evidence context, $\begin{array} { r l } { a _ { \ell } } & { { } \in } \end{array}$ {accept, continue, abort} is the loop action, and $h _ { \ell }$ denotes the structured hints carried to the next round. This protocol makes Finder an iterative evidence-allocation primitive rather than a passive query over a pre-built scene representation. The protocol defines functional stage boundaries rather than fixing a unique implementation; in our current instantiation, planning and loop judgment are LLM/VLM-driven, while detection is performed by SAM3 [34] on the routed frame subset. The default implementation and API-family sensitivity are reported in Appendix Table 7.

![](images/829d9428496230adf9c18d2fc4aa9210fc73844ebbd36de282b88c10c3aaaade.jpg)  
Figure 2: Core execution protocol of Finder. The reusable unit is a typed closed loop: staged execution produces explicit request, plan, detection, context, selection, decision, hint, and snapshot objects, while only the judge can advance the episode to accept, abort, or another planning round.

Typed state boundary. Finder is reusable because it preserves a stable protocol-state contract across loops and task wrappers. A FinderRequest stores the query together with injected task/session context and optional memory metadata; a SearchPlan stores the primary/support targets, evidence goals, and scope plan for the current loop; a LoopContext stores candidate evidence and relation summaries for verification; and a LoopState carries seen candidates and judge-produced notes and hints across retries. This state boundary keeps retries inspectable and lets planners, detectors, selectors, judges, and outer wrappers change without rewriting the closed-loop protocol.

## 3.2 Query-Conditioned Planning and Scoped Routing

Finder decouples target planning from search control. A SearchPlan specifies the primary object, optional supporting objects, and compact detection phrases, while a ScopePlan specifies where to search and how to allocate the perception budget. This separation lets Finder use context to choose where to search while keeping object phrases focused on what to detect.

The frame router executes the scope plan by prioritizing explicit room/floor priors when available, otherwise balancing likely semantic scopes with a global reserve and previously explored constraints. Supporting targets are also executed adaptively: they can be searched globally, gathered only around primary candidates, or kept as context when active detection would not be useful. Thus, the planner directs perception toward evidence needed for the current query instead of a full mapping pass. Additional implementation details are provided in Appendix A.3.

## 3.3 Candidate-Centric Evidence and Verification

Given the routed frame subset and compact phrases, Finder uses phrase-conditioned SAM3 [34] detection to form 3D object candidates. The key design is to keep the subsequent reasoning candidatecentric: instead of flattening all observations into a scene-level summary, Finder builds evidence cards for primary candidates, including representative views, geometric cues, and support-aware relations.

A selection planner then ranks the evidence cards, shortlists candidates, and chooses representative views for direct visual verification. The VLM selector is restricted to factual image-grounded assessment and match judgments; it does not decide whether the episode should stop. This keeps visual verification separate from loop control, leaving the judge to make the final accept, continue, or abort decision. More details on candidate formation and evidence cards are given in Appendix A.3.

## 3.4 Loop Judgment and Cross-Loop Feedback

The loop judge is responsible for stopping control: it determines whether the current evidence is sufficient to accept the selected candidate, whether another loop should be executed, or whether the episode should abort. To do so, it combines four signal families: the selector’s visual facts, support-relation metrics, room-binding facts derived from room geometry and candidate pose, and short-horizon history consistency across recent loops. When the evidence remains insufficient, it emits structured next-loop hints, including candidate follow-up priorities, explored-region constraints, and whether the next loop should favor phrase retargeting or routing changes. This judge-produced feedback is the mechanism that closes the loop: it separates candidate ranking from stopping control and turns retries into structured follow-up rather than blind repetition.

## 3.5 Outer Harness for Task-Specific Context Injection

Beyond the default retrieval setting, Finder is adapted to downstream tasks by an outer harness that writes task-specific context into the request interface. The harness changes the context presented to the loop, but it does not modify the inner loop stages themselves.

Sequential grounding. For sequential grounding, the harness injects the current step label, previously resolved objects, and optionally future-step context. This lets the same loop ground the next target under task-progress constraints rather than treating each query as an isolated episode.

Object-centric QA. For object-centric question answering, the harness injects already mentioned entities and question-specific constraints. This lets the same loop retrieve the object evidence needed for answering while preserving the underlying grounding protocol.

## 4 Experiments

## 4.1 Experimental Setup

Evaluation settings. We evaluate Finder in three settings: Object Retrieval, Sequential Grounding, and Spatio-Temporal QA. The first tests the core closed-loop grounding capability, while the latter two test transfer to downstream object-centric embodied tasks.

Object Retrieval. We evaluate closed-loop open-vocabulary retrieval on two splits: a simulated split built on Habitat [18] with HM3D [19], covering 9 indoor scenes with RGB-D streams and poses, and a real-world split with 4 cluttered indoor environments from FSR-VLN [35]. Appendices B and C provide dataset construction details, statistics, and split diagnostics.

Sequential Grounding. We evaluate transfer to task-chain grounding on the published SG3D benchmark following ASHiTA [20] and DAAAM [6]. This setting tests whether the same inner loop can ground an ordered chain of object-level targets rather than a single isolated query.

Spatio-Temporal QA. We evaluate transfer to object-centric question answering on OC-NaVQA following DAAAM [6]. This setting tests whether grounded object evidence can support downstream question answering under different temporal constraints.

Evaluation metrics. We evaluate across the three settings summarized in Table 1. We use xˆ, x $\hat { \mathbf { x } } , \mathbf { x } ^ { \star }$ for predicted and ground-truth 3D locations, $\hat { Z } , Z ^ { \star }$ for predicted and ground-truth step sequences, and $\hat { a } , a ^ { \star } , \hat { \mathbf { p } } , \mathbf { p } ^ { \star } , \hat { t } , t ^ { \star }$ for predicted and ground-truth QA answers, positions, and times.

Table 1: Evaluation metrics grouped by task setting.
<table><tr><td>Setting</td><td>Metric</td><td>Definition</td></tr><tr><td>Object Retrieval</td><td>S@0.5↑ S@1↑ LocErr ↓</td><td>category-correct retrieval rate with  $\| \hat { \mathbf { x } } - \mathbf { x } ^ { \star } \| _ { 2 } < 0 . 5 \mathrm { m }$  category-correct retrieval rate with  $\| \hat { \mathbf { x } } - \mathbf { x } ^ { \star } \| _ { 2 } < 1 . 0 \mathrm { m }$  mean  $\| \dot { \hat { \mathbf { x } } } - \mathbf { x } ^ { \star } \|$  2 over successful S@1 queries</td></tr><tr><td>Sequential Grounding</td><td>s-acc ↑ t-acc ↑</td><td>fraction of subtasks with predicted pose inside the GT bbox fraction of tasks with all subtask poses inside the GT bboxes</td></tr><tr><td>Spatio-Temporal QA</td><td>QA Acc. ↑ PosErr↓(m) TempErr ↓ (min)</td><td>answer match rate, i.e.,  $\hat { a } = a ^ { \star }$  mean  $\| \hat { \mathbf { p } } - \mathbf { p } ^ { \star } \| _ { 2 }$  over questions with spatial supervision mean  $| \hat { t } - t ^ { \star } |$  over questions with temporal supervision</td></tr></table>

Table 2: Object Retrieval results on simulated and real-world scenes. S@r denotes category-correct Success@r meters. Err denotes LocErr. Best in bold, second underlined.
<table><tr><td></td><td colspan="3">Simulated</td><td colspan="3">Real-world</td><td colspan="3">Average</td></tr><tr><td>Method</td><td>S@0.5↑</td><td>S@1↑</td><td>Err↓</td><td>S@0.5↑</td><td>S@1↑</td><td>Err↓</td><td>S@0.5↑</td><td>S@1↑</td><td>Err↓</td></tr><tr><td>HOV-SG [1]</td><td>33.03</td><td>50.90</td><td>0.409</td><td>31.71</td><td>42.68</td><td>0.391</td><td>32.96</td><td>50.49</td><td>0.408</td></tr><tr><td>DualMap [9]</td><td>38.26</td><td>51.94</td><td>0.340</td><td>9.76</td><td>19.51</td><td>0.494</td><td>36.83</td><td>50.31</td><td>0.348</td></tr><tr><td>FSR-VLN [35]</td><td>32.65</td><td>48.90</td><td>0.419</td><td>12.20</td><td>24.39</td><td>0.532</td><td>31.62</td><td>47.67</td><td>0.422</td></tr><tr><td>Finder (Ours)</td><td>61.23</td><td>67.29</td><td>0.187</td><td>35.37</td><td>46.34</td><td>0.339</td><td>59.93</td><td>66.24</td><td>0.193</td></tr></table>

## 4.2 Main Results

Object Retrieval. As shown in Table 2, Finder achieves the best performance in both simulated and real-world settings, and leads on all three averaged Object Retrieval metrics. Compared with the strongest baseline result on each metric among the three baselines, it improves S@0.5 by 23.10 points, S@1 by 15.75 points, and reduces LocErr by 0.155 m. Performance is consistently higher in simulation than in real-world scenes, which we attribute mainly to real capture artifacts such as motion blur and exposure variation that make these methods less stable.

Sequential Grounding. The results on the SG3D sequential-grounding benchmark [6, 37] are reported in Table 3, which evaluates whether predicted subtask poses fall inside the corresponding ground-truth target bounding boxes across an ordered chain of object-level targets. Finder achieves the best results on both s-acc and t-acc, improving over DAAAM+GPT [6] by 3.75 and 0.98 percentage points, respectively. This is consistent with Finder’s strength in precise grounding, since a subtask is counted correct only when its predicted pose falls inside the ground-truth bounding box.

Spatio-Temporal Question Answering. The OC-NaVQA results are reported in Table 4. Finder achieves the best PosErr and TempErr, suggesting that its grounding loop transfers well to spatial and temporal localization questions. Its QA Acc. remains below that of DAAAM because some questions require aggregation over multiple grounded objects, such as counting bicycles, which is outside Finder’s current object-finding focus.

## 4.3 Ablation Studies

We conduct the main ablation on the simulated Object Retrieval split with 1,550 queries; additional implementation ablations are provided in Appendix A.2.

Main ablation. Table 5 isolates the main closed-loop mechanisms by removing or simplifying individual components and comparing support-object execution scopes; detailed variant definitions are listed in Appendix Table 8. Four trends are shown in Table 5. First, iteration without feedback is not enough: Single-shot slightly outperforms No Feedback, indicating that later loops need carried notes and targeted follow-up rather than blind repetition. Second, support-aware context matters: No Support Targets underperforms the complete support-scope variants, and the global/adaptive/local variants trade off coarse retrieval coverage and fine-grained localization. Third, rule-based visual decision modules degrade performance, with the larger drop from Rule Selector showing that candidate-level visual verification contributes more strongly than the lightweight judge heuristic. Finally, LLM Sel. Planner underperforms the default typed selection planner, suggesting that structured shortlist organization is more stable than free-form LLM planning for this stage.

Table 3: Sequential Grounding results on SG3D [6].
<table><tr><td>Method</td><td>s-acc ↑</td><td>t-acc ↑</td></tr><tr><td>Hydra+GPT [21]</td><td>8.18</td><td>2.44</td></tr><tr><td>Hydra(GT)+GPT [21]</td><td>14.2</td><td>6.34</td></tr><tr><td>HOV-SG [1]</td><td>8.98</td><td>1.95</td></tr><tr><td>ASHiTA [20]</td><td>21.7</td><td>8.78</td></tr><tr><td>DAAAM+GPT [6]</td><td>22.16</td><td>11.22</td></tr><tr><td>Finder</td><td>25.91</td><td>12.20</td></tr></table>

Table 4: Spatio-Temporal QA results on OC-NaVQA [6]. QA Acc. measures answer correctness, while PosErr and TempErr measure localization quality.
<table><tr><td>Method</td><td>QA Acc. ↑ PosErr ↓</td><td></td><td>TempErr ↓</td></tr><tr><td>ReMEmbR-2B [16, 36]</td><td>0.432</td><td>53.466</td><td>2.287</td></tr><tr><td>ReMEmbR-8B [16, 36]</td><td>0.463</td><td>55.894</td><td>4.106</td></tr><tr><td>ConceptGraphs [8]</td><td>0.299</td><td>111.29</td><td></td></tr><tr><td>DAAÁM [6]</td><td>0.711</td><td>41.75</td><td>1.792</td></tr><tr><td>Finder</td><td>0.540</td><td>39.04</td><td>1.433</td></tr></table>

Table 5: Main ablation on the simulated split. Metrics are reported for simple, hard, and averaged queries; variant definitions are listed in Appendix Table 8.
<table><tr><td></td><td colspan="3">S@0.5↑</td><td colspan="3">S@1↑</td><td colspan="3">LocErr↓</td></tr><tr><td>Variant</td><td>Simple</td><td>Hard</td><td>Avg</td><td>Simple</td><td>Hard</td><td>Avg</td><td>Simple</td><td>Hard</td><td>Avg</td></tr><tr><td>Single-shot</td><td>60.91</td><td>55.35</td><td>58.26</td><td>66.95</td><td>59.81</td><td>63.55</td><td>0.187</td><td>0.168</td><td>0.178</td></tr><tr><td>No Feedback</td><td>59.80</td><td>55.07</td><td>57.55</td><td>66.58</td><td>59.95</td><td>63.42</td><td>0.196</td><td>0.173</td><td>0.186</td></tr><tr><td>LLM Sel. Planner</td><td>60.79</td><td>54.67</td><td>57.87</td><td>67.32</td><td>60.22</td><td>63.94</td><td>0.192</td><td>0.178</td><td>0.186</td></tr><tr><td>No Support Targets</td><td>62.76</td><td>53.18</td><td>58.19</td><td>69.30</td><td>59.40</td><td>64.58</td><td>0.187</td><td>0.189</td><td>0.188</td></tr><tr><td>Rule Selector</td><td>58.20</td><td>49.80</td><td>54.19</td><td>65.72</td><td>55.89</td><td>61.03</td><td>0.204</td><td>0.195</td><td>0.200</td></tr><tr><td>Rule Judge</td><td>58.57</td><td>52.91</td><td>55.87</td><td>65.60</td><td>57.92</td><td>61.94</td><td>0.197</td><td>0.177</td><td>0.188</td></tr><tr><td>Finder-Global</td><td>62.52</td><td>58.59</td><td>60.65</td><td>70.16</td><td>65.22</td><td>67.81</td><td>0.195</td><td>0.183</td><td>0.189</td></tr><tr><td>Finder-Adaptive</td><td>64.12</td><td>58.05</td><td>61.23</td><td>70.78</td><td>63.46</td><td>67.29</td><td>0.192</td><td>0.182</td><td>0.187</td></tr><tr><td>Finder-Local</td><td>63.13</td><td>59.00</td><td>61.16</td><td>70.41</td><td>63.73</td><td>67.23</td><td>0.196</td><td>0.169</td><td>0.184</td></tr></table>

Perception budget. The left panel of Figure 3 shows that retrieval improves once the frame budget is large enough to cover relevant views. However, overly sparse sampling degrades performance because candidate comparison and loop judgment still need sufficiently dense visual evidence. This supports Finder’s default budget choice, which balances evidence coverage and efficiency.

Number of retrieval phrases. The right panel of Figure 3 shows that compact multi-phrase supervision is useful, especially when moving from a single primary phrase to a small phrase set. Phrase diversity stabilizes open-vocabulary detection and verification by avoiding dependence on one canonical object wording. Support phrases provide additional relational evidence, but their gains are less consistent, so the default uses a compact support set rather than maximizing phrase count.

## 4.4 Qualitative Analysis

Support-aware disambiguation. Figure 4(a) shows an ambiguous query with multiple visually plausible vase-like candidates. Finder recovers the correct instance by using the support relation, the vase on the dark countertop beside the toilet, rather than object category alone, rejecting candidates with missing or mismatched local context.

Judge-guided refinement. Figure 4(b) shows a query where plausible cabinet candidates are retrieved in Loop 1 but the evidence remains incomplete. Rather than repeating the same search, the judge requests new phrases and frames, shifting Loop 2 toward the sink region and exposing the correct cabinet by filling in evidence missing from the first loop.

![](images/d70974159d43ee72488965cc8bd8b564c1b884b997c93804d0edae6ee6e03b82.jpg)

![](images/b90f1bc478d3fb426e80812cca44bb9a7b189de1ed7e063636978cf407dc400b.jpg)

Figure 3: Design-space studies for Finder’s default budgets. Left: perception budget. Right: phrase budget. The selected defaults balance performance and efficiency.  
![](images/be71f7ead1242f899c5774f0b8dd4f52b08a5094c488fadfdea6f04f17571ae9.jpg)  
Figure 4: Qualitative examples. (a) Support-aware disambiguation. (b) Judge-guided cross-loop refinement. (c) Failure cases: missed target recall and anchor-attribute mismatch.

Failure cases. Figure 4(c) highlights two loop-structured failure modes. The upper example is perception-limited: although the support object is visible and the search region is reasonable, the orange toiletry bag is never proposed as a candidate, leaving later loops with no correct target evidence to refine. The lower example is ambiguity-limited: Finder retrieves a side table with approximately correct geometry, but the anchor attribute is wrong because the nearby sofa is teal rather than brown.

## 5 Conclusion

We presented Finder, an agentic closed-loop primitive for object-centric grounding. Finder makes planning, scoped evidence gathering, candidate verification, and accept/continue/abort control explicit under a stable typed loop, while allowing lightweight wrappers to inject task context. Across object retrieval, sequential grounding, and object-centric question answering, the results show that explicit closed-loop grounding is a practical foundation for broader embodied behavior.

Limitations Finder currently assumes pre-captured RGB-D streams with known poses; extending the same loop to active exploration and online scene updates would reduce dependence on prior coverage. The current formulation is strongest for object-finding subproblems and does not yet fully support broader multi-object aggregation or counting.

## Acknowledgments

## References

[1] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard. Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. In First Workshop on Vision-Language Modelsfor Navigation and Manipulation at ICRA 2024, 2024.

[2] K. Rana, J. Haviland, S. Garg, J. Abou-Chakra, I. Reid, and N. Suenderhauf. Sayplan: Ground ing large language models using 3d scene graphs for scalable robot task planning. arXiv preprint arXiv:2307.06135, 2023.

[3] M. Shridhar, J. Thomason, D. Gordon, Y. Bisk, W. Han, R. Mottaghi, L. Zettlemoyer, and D. Fox. ALFRED: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10740–10749, 2020.

[4] A. Padmakumar, J. Thomason, A. Shrivastava, P. Lange, A. Narayan-Chen, S. Gella, R. Piramuthu, G. Tur, and D. Hakkani-Tur. TEACh: Task-driven embodied agents that chat. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 36, pages 2017–2025, 2022.

[5] Z. Cai, Y. Du, C. Wang, and Y. Kong. Vision to geometry: 3d spatial memory for sequential embodied mllm reasoning and exploration. arXiv preprint arXiv:2512.02458, 2025.

[6] N. Gorlo, L. Schmid, and L. Carlone. Describe anything anywhere at any moment. arXiv preprint arXiv:2512.00565, 2025.

[7] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, T. Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 815–824, 2023.

[8] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, et al. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 5021–5028. IEEE, 2024.

[9] J. Jiang, Y. Zhu, Z. Wu, and J. Song. Dualmap: Online open-vocabulary semantic mapping for natural language navigation in dynamic changing scenes. IEEE Robotics and Automation Letters, 2025.

[10] S. Koch, N. Vaskevicius, M. Colosi, P. Hermosilla, and T. Ropinski. Open3dsg: Open-vocabulary 3d scene graphs from point clouds with queryable objects and open-set relationships. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14183–14193, 2024.

[11] D. Maggio, Y. Chang, N. Hughes, M. Trang, D. Griffith, C. Dougherty, E. Cristofalo, L. Schmid, and L. Carlone. Clio: Real-time task-driven open-set 3d scene graphs. IEEE Robotics and Automation Letters, 9(10):8921–8928, 2024.

[12] C. Zhang, A. Delitzas, F. Wang, R. Zhang, X. Ji, M. Pollefeys, and F. Engelmann. Openvocabulary functional 3d scene graphs for real-world indoor spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19401–19413, 2025.

[13] J. Yang, X. Chen, S. Qian, N. Madaan, M. Iyengar, D. F. Fouhey, and J. Chai. Llm-grounder: Open-vocabulary 3d visual grounding with large language model as an agent. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 7694–7701. IEEE, 2024.

[14] M. Booker, G. Byrd, B. Kemp, A. Schmidt, and C. Rivera. Embodiedrag: Dynamic 3d scene graph retrieval for efficient and scalable robot task planning. arXiv preprint arXiv:2410.23968, 2024.

[15] Y. Chang, R. Chen, Z. Zhang, Y. Chen, and S. Xie. Rag-3dsg: Enhancing 3d scene graphs with re-shot guided retrieval-augmented generation. arXiv preprint arXiv:2601.10168, 2026.

[16] A. Anwar, J. Welsh, J. Biswas, S. Pouya, and Y. Chang. Remembr: Building and reasoning over long-horizon spatio-temporal memory for robot navigation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 2838–2845. IEEE, 2025.

[17] Y. Fang, Z. Shi, J. Qiu, Z. Chen, J. Shi, H. Xu, J. Huo, and Y. Gao. Inherit-sg: Incremental hierarchical semantic scene graphs with rag-style retrieval. arXiv preprint arXiv:2602.12971, 2026.

[18] S. K. Ramakrishnan, A. Gokaslan, E. Wijmans, O. Maksymets, A. Clegg, J. Turner, E. Undersander, W. Galuba, A. Westbury, A. X. Chang, et al. Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238, 2021.

[19] K. Yadav, R. Ramrakhya, S. K. Ramakrishnan, T. Gervet, J. Turner, A. Gokaslan, N. Maestre, A. X. Chang, D. Batra, M. Savva, et al. Habitat-matterport 3d semantics dataset. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4927–4936, 2023.

[20] Y. Chang, L. Fermoselle, D. Ta, B. Bucher, L. Carlone, and J. Wang. Ashita: Automatic scene-grounded hierarchical task analysis. arXiv preprint arXiv:2504.06553, 2025.

[21] N. Hughes, Y. Chang, and L. Carlone. Hydra: A real-time spatial perception system for 3d scene graph construction and optimization. arXiv preprint arXiv:2201.13360, 2022.

[22] T. B. Martins, M. R. Oswald, and J. Civera. Open-vocabulary online semantic mapping for slam. IEEE Robotics and Automation Letters, 2025.

[23] P. Saxena and J. Chiun. Zing-3d: Zero-shot incremental 3d scene graphs via vision-language models. arXiv preprint arXiv:2510.21069, 2025.

[24] Y. Tang, M. Wang, Y. Deng, Z. Zheng, J. Deng, S. Zuo, and Y. Yue. Openin: Open-vocabulary instance-oriented navigation in dynamic domestic environments. IEEE Robotics and Automation Letters, 10(9):9256–9263, 2025.

[25] L. Schmid, M. Abate, Y. Chang, and L. Carlone. Khronos: A unified approach for spatiotemporal metric-semantic slam in dynamic environments. arXiv preprint arXiv:2402.13817, 2024.

[26] Z. Yan, S. Li, Z. Wang, L. Wu, H. Wang, J. Zhu, L. Chen, and J. Liu. Dynamic open-vocabulary 3d scene graphs for long-term language-guided mobile manipulation. IEEE Robotics and Automation Letters, 2025.

[27] L. Ge, X. Zhu, Z. Yang, and X. Li. Dynamicgsg: Dynamic 3d gaussian scene graphs for environment adaptation. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 2232–2239. IEEE, 2025.

[28] S. Salimpour, L. Fu, K. Rachwał, P. Bertrand, K. O’Sullivan, R. Jakob, F. Keramat, L. Militano, G. Toffetti, H. Edelman, et al. Towards embodied agentic ai: Review and classification of llm-and vlm-driven robot autonomy and interaction. arXiv preprint arXiv:2508.05294, 2025.

[29] H. Yin, X. Xu, Z. Wu, J. Zhou, and J. Lu. Sg-nav: Online 3d scene graph prompting for llm-based zero-shot object navigation. Advances in neural information processing systems, 37: 5285–5307, 2024.

[30] M. Lei, H. Cai, Z. Cui, L. Tan, J. Hong, G. Hu, S. Zhu, Y. Wu, S. Jiang, G. Wang, et al. Robomemory: A brain-inspired multi-memory agentic framework for lifelong learning in physical embodied systems. In NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI, 2025.

[31] L. Zhang, Y. Liu, Z. Zhang, M. Aghaei, Y. Hu, H. Gu, M. A. Alomrani, D. G. A. Bravo, R. Karimi, A. Hamidizadeh, et al. Mem2ego: Empowering vision-language models with global-to-ego memory for long-horizon embodied navigation. arXiv preprint arXiv:2502.14254, 2025.

[32] R. Korekata, Q. Xie, Y. Bisk, and K. Sugiura. Affordance rag: Hierarchical multimodal retrieval with affordance-aware embodied memory for mobile manipulation. IEEE Robotics and Automation Letters, 11(3):2706–2713, 2026.

[33] I. Ahmadi, M. Taji, A. M. Kashani, A. Jadidi, S. Kashani, and B. Khalaj. Mallvi: a multi agent framework for integrated generalized robotics manipulation. arXiv preprint arXiv:2602.16898, 2026.

[34] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, et al. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719, 2025.

[35] X. Zhou, T. Xiao, L. Liu, Y. Wang, M. Chen, X. Meng, X. Wang, W. Feng, W. Sui, and Z. Su. Fsr-vln: Fast and slow reasoning for vision-language navigation with hierarchical multi-modal scene graph. arXiv preprint arXiv:2509.13733, 2025.

[36] Z. Liu, L. Zhu, B. Shi, Z. Zhang, Y. Lou, S. Yang, H. Xi, S. Cao, Y. Gu, D. Li, et al. Nvila: Efficient frontier visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4122–4134, 2025.

[37] Z. Zhang, Z. Zhu, J. Li, P. Li, T. Wang, T. Liu, X. Ma, Y. Chen, B. Jia, S. Huang, et al. Taskoriented sequential grounding and navigation in 3d scenes. arXiv preprint arXiv:2408.04034, 2024.

# Finder: Agentic Closed-Loop Object Finding for Embodied Grounding

Appendix

## A Implementation Details

Finder is implemented as a bounded closed loop rather than a one-shot open-vocabulary retrieval call. The planner first separates object phrases from spatial scope, the router allocates a frame budget under room/floor priors when available, and the judge either accepts the current candidate or emits structured hints for the next loop. Outer-session memory is optional: previously resolved objects, selected regions, and known context objects enter only as structured planner/judge hints.

## A.1 Default settings

Table 6 lists the default control settings used in the Object Retrieval benchmark.

Table 6: Default control settings used by Finder in the Object Retrieval benchmark.
<table><tr><td>Setting</td><td>Default</td><td>Role</td></tr><tr><td>Loop budget</td><td>3 loops</td><td>Allows retry after zero detections, weak support evidence, or unstable candidates.</td></tr><tr><td>Frame cap</td><td>200 frames</td><td>Caps perception cost after scope-based routing and resampling.</td></tr><tr><td>Frame stride</td><td>10</td><td>Reduces redundant adjacent views before budgeted detection.</td></tr><tr><td>Primary phrases</td><td>2</td><td>Provides compact phrase diversity for the target object.</td></tr><tr><td>Support phrases</td><td>2</td><td>Adds relational/context evidence without dominating the target search.</td></tr><tr><td>Support execution</td><td>Adaptive</td><td>Lets the planner choose global, candidate-local, conditional, or no support search.</td></tr></table>

## A.2 API-family sensitivity

Table 7 tests whether Finder’s typed state contract remains effective under different planner and VLM APIs. Planner capability is the dominant factor in this axis, while VLM size does not produce a strictly monotonic trend because final performance also depends on planner quality and candidate evidence construction. We use MiMo-v2-Pro with Qwen 397B-A17B as the default primarily for stable service availability during the evaluation cycle.

Table 7: API-family ablation on the simulated retrieval split, pairing MiMo planner families with Qwen VLM choices. Cells report S@1 / LocErr; best in bold, second underlined.
<table><tr><td rowspan="2">Variant</td><td colspan="2">35B-A3B</td><td colspan="2">122B-A10B</td><td colspan="2">397B-A17B</td></tr><tr><td>S@1↑</td><td>LocErr↓</td><td>S@1↑</td><td>LocErr↓</td><td>S@1↑</td><td>LocErr↓</td></tr><tr><td>MiMo-v2-Pro / Qwen</td><td>64.71</td><td>0.194</td><td>66.65</td><td>0.188</td><td>67.29</td><td>0.187</td></tr><tr><td>MiMo-v2.5 / Qwen</td><td>66.06</td><td>0.190</td><td>66.90</td><td>0.184</td><td>67.16</td><td>0.193</td></tr><tr><td>MiMo-v2.5-Pro / Qwen</td><td>68.69</td><td>0.191</td><td>67.81</td><td>0.190</td><td>66.97</td><td>0.186</td></tr><tr><td>Matched Qwen / Qwen</td><td>65.10</td><td>0.189</td><td>63.10</td><td>0.189</td><td>68.00</td><td>0.193</td></tr></table>

## A.3 Planning and evidence construction details

Planning fields. The target-side SearchPlan contains one primary target, zero or more supporting targets, an evidence-goal list, and compact phrase sets for targets scheduled for active detection. The primary target is the object to be returned if the loop succeeds. Supporting targets are introduced only when they help anchor or disambiguate the primary target, such as in relational queries. The ScopePlan contains four search-control fields: hard scopes from explicit room/floor constraints, soft scopes from semantic room hints or support cues, per-scope frame budgets with a global reserve, and explored constraints that deprioritize regions or points in later loops.

Routing and support execution. If an explicit room or floor is available, Finder spends the budget inside that hard scope. Otherwise, it splits the budget between likely room scopes and a global reserve. The frame router applies room/floor filtering, excludes previously explored regions when requested, and pose-samples the remaining frame inventory to satisfy the scope budget. Each supporting target carries an execution policy: it can be detected globally, deferred until primary candidates are available, or kept as context only. The local-after-primary mode gathers support-object evidence only on candidate-local views, which is useful when support objects mainly disambiguate among already plausible primary candidates.

Candidate formation and evidence cards. For each routed frame, phrase-conditioned masks are backprojected into 3D, clustered into raw object proposals, and merged by geometry-aware deduplication when proposals likely correspond to the same object. Each retained candidate stores a 3D center, a compact geometry summary, representative views, and provenance linking it to detector instances. Finder then constructs one evidence card per primary candidate. Evidence cards summarize candidate-local measurements, visual/geometry cues, support proximity, and the source of support evidence, giving the selector and judge a target-centric record rather than a flattened scene summary.

## A.4 Baseline and variant definitions

All methods are evaluated with the same query splits, ground-truth centroids, and radius-based success metrics defined in Section C.1. The baseline implementations are summarized as follows:

• HOV-SG [1]: a pre-built open-vocabulary scene graph baseline. We use top-1 graph retrieval and score the returned object centroid with the same geometry-only success metric as Finder. Hard queries are evaluated from text only, while simple queries additionally use the available room prior for filtering.

• DualMap [9]: a dual semantic-spatial map baseline evaluated under the same hard/simple query protocol. Simulated scenes use the dense-frame rerun results, and real-world scenes are scored after alignment to the shared z-up coordinate convention.

• FSR-VLN [35]: a navigation-oriented scene-graph/memory baseline adapted to the Object Retrieval queries. We use the same benchmark queries and available simple-query priors, and report the radius-specific retrieval fields when available.

For Sequential Grounding and Spatio-Temporal QA, we compare against published benchmark results from ASHiTA [20], DAAAM [6], ReMEmbR [16], and ConceptGraphs [8], rather than introducing additional task-specific reruns.

Table 8: Variant-to-change mapping for the main ablation in Table 5.
<table><tr><td>Variant</td><td>Change</td></tr><tr><td>Single-shot</td><td>Set max_1oops=1.</td></tr><tr><td>No Feedback</td><td>Clear carried notes and next-loop hints between loops.</td></tr><tr><td>LLM Sel. Planner</td><td>Replace the default structured selection planner with an LLM-based planner.</td></tr><tr><td>No Support Targets</td><td>Remove supporting targets and candidate-local support follow-up.</td></tr><tr><td>Rule Selector</td><td>Replace VLM visual verification with a simple heuristic selector.</td></tr><tr><td>Rule Judge</td><td>Replace the VLM loop judge with a lightweight accept/continue heuristic.</td></tr><tr><td>Finder-Global</td><td>Force all supporting targets into the global pass.</td></tr><tr><td>Finder-Adaptive</td><td>Default Finder configuration with planner-chosen support execution.</td></tr><tr><td>Finder-Local</td><td>Force supporting targets to run in local_after_primary.</td></tr></table>

## B Dataset Details

The Object Retrieval benchmark contains 1,632 single-object queries over 197 query-target categories across 9 simulated scenes and 4 real-world scenes. The simulated split is built from RGB-D streams, semantic object annotations, and posed camera trajectories in Habitat/HM3D scenes. We canonicalize object categories, remove structural or unreachable labels, keep visible instances with valid centroids and view evidence, and export room/floor metadata only as optional priors for scoped simple queries. The real-world split is built from cluttered indoor RGB-D captures where targets are manually selected from RGB-D frames and lifted to 3D centroids from annotated masks.

## B.1 Dataset statistics

![](images/d2f6693951590e9455abd8ba65531c1a9d3afde12e7f590cffbde1221284f10b.jpg)  
Figure 5: Object Retrieval dataset overview. Left: query distribution across object categories, showing a long-tailed benchmark with 1,632 queries over 197 categories. Right: per-scene query counts and the simple/hard split, with 811 simple and 821 hard queries across 9 simulated and 4 real-world scenes.

Table 9: Dataset statistics for the Object Retrieval splits. Object counts refer to retrievable target objects after filtering. Category counts follow the query-target category canonicalization used throughout the benchmark.
<table><tr><td>Statistic</td><td>Simulated</td><td>Real-world</td></tr><tr><td>Number of scenes</td><td>9</td><td>4</td></tr><tr><td>Total rooms</td><td>122</td><td>4</td></tr><tr><td>Avg. rooms per scene</td><td>13.6</td><td>1.0</td></tr><tr><td>Total target objects</td><td>1,890</td><td>82</td></tr><tr><td>Avg. target objects per scene</td><td>210.0</td><td>20.5</td></tr><tr><td>Indexed RGB-D views</td><td>31,520</td><td>67</td></tr><tr><td>Avg. indexed RGB-D views per scene</td><td>3,502.2</td><td>16.8</td></tr><tr><td>Simple queries</td><td>811</td><td>0</td></tr><tr><td>Hard queries</td><td>739</td><td>82</td></tr><tr><td>Single-object queries</td><td>1,550</td><td>82</td></tr><tr><td>Sequential tasks</td><td>110</td><td>0</td></tr><tr><td>Query-target categories</td><td>166</td><td>49</td></tr></table>

## B.2 Annotation workflow and prompts

Simulated annotations follow four steps: scene inventory, view indexing, query writing, and groundtruth binding. Simple queries include explicit room/floor priors when available, while hard queries are evaluated scene-wide and must rely on visual, relational, or contextual language rather than hidden room metadata. Real-world annotations use the same final schema after the selected RGB-D mask is lifted to a 3D object point cloud.

Table 10: Query-writing prompt constraints used during dataset construction.
<table><tr><td>Field</td><td>Prompt Constraint</td></tr><tr><td>Target binding</td><td>Each query must refer to one annotated target object with a canonical category and 3D centroid.</td></tr><tr><td>Simple query</td><td>Include available room/floor scope when it is part of the intended prior.</td></tr><tr><td>Hard query</td><td>Do not reveal room/floor metadata; use visible appearance, relation, or context to identify the target.</td></tr><tr><td>Ambiguity control</td><td>If multiple same-category objects exist, wording should preserve the intended ambiguity level without becoming ungrounded.</td></tr><tr><td>Quality control</td><td>Reject queries whose target is not visible in the evidence views or whose wording cannot be checked from visual context.</td></tr></table>

## B.3 Annotation interface

The annotation interfaces let annotators bind language queries to target objects, inspect visual evidence, and verify the saved 3D target metadata. Figures 6 and 7 show the real-world and simulated annotation tools.

![](images/837f0206f35d5ba8b439f8a3740ae98555140aee6b6770949ed6c36988b43c97.jpg)  
Figure 6: Annotation and verification interface for real-world data.

![](images/ecf59dc1774478c919dde2e67124b10cf15f3a7a8b9af6115bf67e4d4ac1a981.jpg)  
Figure 7: Annotation and verification interface for simulated data.

## C Supplementary Experiments

## C.1 Evaluation protocol and metrics

Object Retrieval is evaluated separately on simulated and real-world scenes and then combined using query-weighted averaging. The simulated benchmark contains both simple and hard queries; the realworld benchmark currently contains only hard queries. Success@r requires both a category match and centroid distance below radius r; LocErr is the mean centroid distance over Success@1m cases. For ambiguous same-category queries, a prediction succeeds if it falls within the radius threshold of any valid same-category ground-truth instance under the query scope. Sequential grounding and spatio-temporal QA follow the metrics summarized in Table 1.

Table 11: Per-scene Object Retrieval results on simulated scenes, split by hard, simple, and combined protocols. Values are percentages for S@0.5 and S@1; Err is mean distance in meters over S@1 successes.
<table><tr><td></td><td></td><td></td><td colspan="3">HOV-SG</td><td colspan="3">DualMap</td><td colspan="3">FSR-VLN</td><td colspan="3">Finder</td></tr><tr><td>Scene</td><td>Split</td><td>Num</td><td>S@0.5</td><td>S@1</td><td>Err</td><td>S@0.5</td><td>S@1</td><td>Err</td><td>S@0.5</td><td>S@1</td><td>Err</td><td>S@0.5</td><td>S@1</td><td>Err</td></tr><tr><td>00824-Dd4bFSTQ8gi</td><td>Hard</td><td>91</td><td>20.88</td><td>39.56</td><td>0.504</td><td>25.27</td><td>43.96</td><td>0.424</td><td>36.26</td><td>51.65</td><td>0.420</td><td>61.54</td><td>71.43</td><td>0.222</td></tr><tr><td>00824-Dd4bFSTQ8gi</td><td>Simple</td><td>101</td><td>32.67</td><td>53.47</td><td>0.397</td><td>31.68</td><td>49.50</td><td>0.382</td><td>37.62</td><td>62.38</td><td>0.424</td><td>58.42</td><td>69.31</td><td>0.255</td></tr><tr><td>00824-Dd4bFSTQ8gi</td><td>Overall</td><td>192</td><td>27.08</td><td>46.88</td><td>0.440</td><td>28.64</td><td>46.87</td><td>0.402</td><td>36.98</td><td>57.29</td><td>0.422</td><td>59.90</td><td>70.31</td><td>0.239</td></tr><tr><td>00829-QaLdnwvtxbs</td><td>Hard</td><td>55</td><td>18.18</td><td>32.73</td><td>0.404</td><td>23.64</td><td>34.55</td><td>0.322</td><td>30.91</td><td>45.45</td><td>0.449</td><td>49.09</td><td>54.55</td><td>0.187</td></tr><tr><td>00829-QaLdnwvtxbs</td><td>Simple</td><td>67</td><td>25.37</td><td>41.79</td><td>0.466</td><td>38.81</td><td>50.75</td><td>0.289</td><td>50.75</td><td>68.66</td><td>0.366</td><td>53.73</td><td>59.70</td><td>0.215</td></tr><tr><td>00829-QaLdnwvtxbs</td><td>Overall</td><td>122</td><td>22.13</td><td>37.71</td><td>0.442</td><td>31.97</td><td>43.45</td><td>0.304</td><td>41.81</td><td>58.20</td><td>0.395</td><td>51.64</td><td>57.38</td><td>0.203</td></tr><tr><td>00843-DYehNKdT76V</td><td>Hard</td><td>52</td><td>32.69</td><td>48.08</td><td>0.454</td><td>38.46</td><td>55.77</td><td>0.398</td><td>50.00</td><td>59.62</td><td>0.324</td><td>53.85</td><td>57.69</td><td>0.151</td></tr><tr><td>00843-DYehNKdT76V</td><td>Simple</td><td>66</td><td>42.42</td><td>62.12</td><td>0.397</td><td>33.33</td><td>46.97</td><td>0.385</td><td>54.55</td><td>66.67</td><td>0.328</td><td>63.64</td><td>66.67</td><td>0.155</td></tr><tr><td>00843-DYehNKdT76V</td><td>Overall</td><td>118</td><td>38.13</td><td>55.93</td><td>0.418</td><td>35.59</td><td>50.85</td><td>0.391</td><td>52.54</td><td>63.56</td><td>0.326</td><td>59.32</td><td>62.71</td><td>0.153</td></tr><tr><td>00847-bCPU9suPUw9</td><td>Hard</td><td>78</td><td>19.23</td><td>42.31</td><td>0.546</td><td>35.90</td><td>44.87</td><td>0.292</td><td>15.38</td><td>35.90</td><td>0.557</td><td>71.79</td><td>74.36</td><td>0.157</td></tr><tr><td>00847-bCPU9suPUw9</td><td>Simple</td><td>83</td><td>44.58</td><td>62.65</td><td>0.386</td><td>40.96</td><td>54.22</td><td>0.306</td><td>38.55</td><td>62.65</td><td>0.430</td><td>66.27</td><td>73.49</td><td>0.198</td></tr><tr><td>00847-bCPU9suPUw9</td><td>Overall</td><td>161</td><td>32.30</td><td>52.80</td><td>0.448</td><td>38.51</td><td>49.69</td><td>0.299</td><td>27.32</td><td>49.69</td><td>0.475</td><td>68.94</td><td>73.91</td><td>0.178</td></tr><tr><td>00861-GLAQ4DNUx5U</td><td>Hard</td><td>69</td><td>17.39</td><td>33.33</td><td>0.471</td><td>44.93</td><td>53.62</td><td>0.298</td><td>20.29</td><td>31.88</td><td>0.468</td><td>53.62</td><td>55.07</td><td>0.160</td></tr><tr><td>00861-GLAQ4DNUx5U</td><td>Simple</td><td>89</td><td>42.70</td><td>58.43</td><td>0.359</td><td>44.94</td><td>59.55</td><td>0.355</td><td>34.83</td><td>50.56</td><td>0.430</td><td>59.55</td><td>66.29</td><td>0.187</td></tr><tr><td>00861-GLAQ4DNUx5U</td><td>Overall</td><td>158</td><td>31.65</td><td>47.47</td><td>0.393</td><td>44.94</td><td>56.96</td><td>0.330</td><td>28.48</td><td>42.40</td><td>0.443</td><td>56.96</td><td>61.39</td><td>0.176</td></tr><tr><td>00862-LT9Jq6dN3Ea</td><td>Hard</td><td>138</td><td>14.49</td><td>26.09</td><td>0.482</td><td>30.43</td><td>39.86</td><td>0.322</td><td>14.49</td><td>22.46</td><td>0.489</td><td>44.20</td><td>50.72</td><td>0.221</td></tr><tr><td>00862-LT9Jq6dN3Ea</td><td>Simple</td><td>148</td><td>31.08</td><td>47.97</td><td>0.416</td><td>41.22</td><td>55.41</td><td>0.347</td><td>29.73</td><td>45.27</td><td>0.450</td><td>66.22</td><td>72.30</td><td>0.183</td></tr><tr><td>00862-LT9Jq6dN3Ea</td><td>Overall</td><td>286</td><td>23.08</td><td>37.41</td><td>0.439</td><td>36.01</td><td>47.91</td><td>0.335</td><td>22.38</td><td>34.26</td><td>0.462</td><td>55.59</td><td>61.89</td><td>0.198</td></tr><tr><td>00873-bxsVRursffK</td><td>Hard</td><td>90</td><td>47.78</td><td>63.33</td><td>0.332</td><td>50.00</td><td>57.78</td><td>0.242</td><td>26.67</td><td>40.00</td><td>0.465</td><td>72.22</td><td>76.67</td><td>0.146</td></tr><tr><td>00873-bxsVRursffK</td><td>Simple</td><td>87</td><td>58.62</td><td>74.71</td><td>0.293</td><td>50.57</td><td>63.22</td><td>0.335</td><td>43.68</td><td>56.32</td><td>0.320</td><td>65.52</td><td>68.97</td><td>0.151</td></tr><tr><td>00873-bxsVRursffK</td><td>Overall</td><td>177</td><td>53.11</td><td>68.92</td><td>0.311</td><td>50.28</td><td>60.45</td><td>0.288</td><td>35.03</td><td>48.02</td><td>0.382</td><td>68.93</td><td>72.88</td><td>0.148</td></tr><tr><td>00877-4ok3usBNeis</td><td>Hard</td><td>71</td><td>38.03</td><td>61.97</td><td>0.454</td><td>32.39</td><td>53.52</td><td>0.397</td><td>19.72</td><td>40.85</td><td>0.526</td><td>63.38</td><td>71.83</td><td>0.197</td></tr><tr><td>00877-4ok3usBNeis</td><td>Simple</td><td>75</td><td>54.67</td><td>74.67</td><td>0.355</td><td>36.00</td><td>60.00</td><td>0.427</td><td>45.33</td><td>61.33</td><td>0.361</td><td>77.33</td><td>84.00</td><td>0.186</td></tr><tr><td>00877-4ok3usBNeis</td><td>Overall</td><td>146</td><td>46.58</td><td>68.49</td><td>0.398</td><td>34.24</td><td>56.85</td><td>0.412</td><td>32.88</td><td>51.37</td><td>0.426</td><td>70.55</td><td>78.08</td><td>0.191</td></tr><tr><td>00890-6s7QHgap2fW</td><td>Hard</td><td>95 95</td><td>22.11</td><td>46.32</td><td>0.519</td><td>40.00</td><td>52.63</td><td>0.338</td><td>18.09</td><td>43.16</td><td>0.518</td><td>56.84</td><td>61.05</td><td>0.169</td></tr><tr><td>00890-6s7QHgap2fW</td><td>Simple Overall</td><td>190</td><td>38.95 30.53</td><td>56.84 51.58</td><td>0.371 0.438</td><td>46.32</td><td>57.89 55.26</td><td>0.294</td><td>44.21</td><td>58.95</td><td>0.357</td><td>65.26</td><td>73.68</td><td>0.192</td></tr><tr><td>00890-6s7QHgap2fW Query Weighted</td><td>Hard</td><td>739</td><td>24.90</td><td>42.76</td><td>0.458</td><td>43.16 35.59</td><td>48.04</td><td>0.316 0.334</td><td>31.15 23.95</td><td>51.06 39.24</td><td>0.426 0.466</td><td>61.05 58.05</td><td>67.37</td><td>0.181 0.182</td></tr><tr></table>

Table 12: Per-scene Object Retrieval results on real-world scenes. The real benchmark currently contains only hard queries, so no simple split is reported. Values are percentages for S@0.5 and S@1; Err is mean distance in meters over S@1 successes.
<table><tr><td></td><td></td><td colspan="3">HOV-SG</td><td colspan="3">DualMap</td><td colspan="3">FSR-VLN</td><td colspan="3">Finder</td></tr><tr><td>Scene</td><td>Num</td><td>S@0.5</td><td>S@1</td><td>Err</td><td>S@0.5</td><td>S@1</td><td>Err</td><td>S@0.5</td><td>S@1</td><td>Err</td><td>S@0.5</td><td>S@1</td><td>Err</td></tr><tr><td>icra_ic3f</td><td>18</td><td>33.33</td><td>38.89</td><td>0.330</td><td>16.67</td><td>27.78</td><td>0.345</td><td>11.11</td><td>16.67</td><td>0.497</td><td>44.44</td><td>50.00</td><td>0.326</td></tr><tr><td>icra_ic4f</td><td>16</td><td>31.25</td><td>43.75</td><td>0.453</td><td>25.00</td><td>37.50</td><td>0.445</td><td>18.75</td><td>43.75</td><td>0.633</td><td>50.00</td><td>62.50</td><td>0.306</td></tr><tr><td>icra_ic7f</td><td>30</td><td>26.67</td><td>36.67</td><td>0.434</td><td>3.33</td><td>16.67</td><td>0.610</td><td>10.00</td><td>23.33</td><td>0.492</td><td>20.00</td><td>36.67</td><td>0.409</td></tr><tr><td>icra_sh3f</td><td>18</td><td>38.89</td><td>55.56</td><td>0.343</td><td>0.00</td><td>0.00</td><td>=</td><td>11.11</td><td>16.67</td><td>0.425</td><td>38.89</td><td>44.44</td><td>0.300</td></tr><tr><td>Query Weighted</td><td>82</td><td>31.71</td><td>42.68</td><td>0.391</td><td>9.76</td><td>19.51</td><td>0.494</td><td>12.20</td><td>24.39</td><td>0.532</td><td>35.37</td><td>46.34</td><td>0.339</td></tr></table>

Table 13: Object-type response analysis for Finder on the aggregated Object Retrieval benchmark. Queries are grouped by target-object profile rather than dataset split.
<table><tr><td>Type</td><td>Examples</td><td>Queries</td><td>S@0.5↑</td><td>S@1↑</td><td>S@3↑</td><td> $T _ { q } \downarrow$ </td></tr><tr><td>Large fixed</td><td>bed, sofa, cabinet, table</td><td>836</td><td>59.45</td><td>65.91</td><td>81.10</td><td>80.5s</td></tr><tr><td>Small clutter</td><td>cup, bottle, vase, toy</td><td>368</td><td>55.43</td><td>62.23</td><td>70.11</td><td>96.1s</td></tr><tr><td>Thin / flat</td><td>book, plate, tray, wall art</td><td>201</td><td>64.68</td><td>67.16</td><td>74.63</td><td>86.0s</td></tr><tr><td>Soft items</td><td>pillow, towel, bag, clothing</td><td>227</td><td>64.76</td><td>73.13</td><td>82.38</td><td>92.8s</td></tr></table>

## C.2 Per-scene retrieval breakdowns

Tables 11 and 12 provide the per-scene retrieval breakdown behind the aggregate Object Retrieval table. We report hard and simple splits separately when both are available. Entries marked “–” indicate undefined metrics, such as LocErr when a method has no S@1 successes.

## C.3 Object-type response analysis

Beyond module-wise ablations, it is useful to understand which kinds of objects Finder responds to most reliably and how response time varies by target type. Table 13 groups Object Retrieval queries by coarse target-object profile using the ground-truth target category. The success columns use the same geometry-only radius thresholds as the main Object Retrieval table, and $T _ { q }$ reports mean per-query runtime.

## D Prompt Templates

## D.1 Request-Planning Prompt

## Request-Planning Prompt

You are the request planner for an object-finding loop.

Your job is NOT to directly choose a final object. Your job is to decide what this loop should search for and which contextual objects should support the search.

## You must produce:

1. the PRIMARY target object for this loop

2. zero or more SUPPORTING targets that provide memory anchors, spatial disambiguation, or additional context

3. search phrases for the primary target and any supporting targets that need active detection

4. evidence goals for this loop

5. an optional scope\_plan describing how detection frame budget should be spent

## Important design rules:

\- There must be exactly ONE primary target.

\- Supporting targets are optional.

\- Only include supporting targets if they help locate or disambiguate the primary target.

\- Supporting targets may use these usages:

\- "memory\_anchor": look up or anchor from memory or metadata

\- "disambiguate": help compare candidates using spatial relations

\- "detect": actively detect this supporting object

\- "context\_only": mention it in reasoning but do not actively detect it

\- Every supporting target must include a support\_policy with:

\- "semantic\_type": one of ["anchor", "verifier", "memory"]

\- "execution\_mode": one of ["global", "local\_after\_primary", "conditional", "none", "rescue\_only"]

\- "expected\_value": estimated usefulness in [0, 1]

\- Search phrases must describe objects themselves, not rooms or long spatial descriptions.

\- Every primary target must output exactly 2 short object phrases.

\- Every supporting target must output exactly 2 short object phrases.

\- Do not generate open-ended synonym lists or phrase expansion beyond those required counts.

\- Spatial scope and frame budget policy are separate from the object phrases.

\- Keep the plan compact. Do not expand the plan into unnecessary targets.

\- Preserve hard constraints already known from upstream context.

\- If the user query contains a relational description like "the mug near the coffee machine", the mug is usually primary and the coffee machine is usually supporting.

\- If previous loop notes exist, use them carefully. They are hints, not hard truth.

\- If ‘previous\_loop\_hints.phrase\_policy.locked‘ is true, preserve the prior

primary phrases and prioritize scope/frame changes over phrase changes.

## D.2 Candidate Visual-Verification Prompt

## Candidate Visual-Verification Prompt

You are the factual visual verifier in an object-finding loop.

Your job is NOT to decide whether the loop should continue or stop.

Your job is to extract image-grounded facts for each candidate.

## You will receive:

\- the raw user query

\- the primary target and supporting-target expectations

\- a shortlist of candidate objects

\- objective, relation-specific evidence for each candidate

\- a manifest that maps attached images to candidate keys

## Important rules:

\- Base your output primarily on what is visually observable in the images.

\- Use relation-specific measurements only as supporting context, not as a substitute for visual facts.

\- Every attached image is a full-frame scene view. These are not cropped object-only patches.

\- If an image is described as "bbox", it means the full scene frame with a box overlay, not a crop.

\- Do not output loop-control fields such as decisive / continue / accept.

\- Do not guess room type from the image unless it is visually obvious. Room binding will be handled separately.

\- Never infer room type from the user query, candidate label, or support-target name.

\- Never say a room is "assumed", "likely", or "probably" based on query context.

\- If room type is not visually explicit, say "room not visually determined" or omit room claims entirely.

\- If room type is unclear, say it is unclear; do not say the image lacks context when surrounding furniture, wall, floor, doorway, or nearby objects are visible.

\- Keep rationale and notes factual. Do not add advice, recommendations, or loop-control guidance.

\- Only cite image indices that exist in the manifest.

## Allowed categorical values:

\- object\_type\_match: strong | partial | weak | mismatch | unclear

\- color\_match: strong | partial | weak | mismatch | unclear

\- appears\_against\_wall: yes | no | unclear

\- wall\_contact\_visibility: direct | partial | not\_visible

\- image\_context: full\_room | partial\_room | object\_focused | unclear

\- object\_completeness: complete | partial | unclear

## D.3 Loop-Judgment Prompt

## Loop-Judgment Prompt

You are the loop judge in an object-finding system.

Your job is to decide whether to:

\- accept the current best candidate

\- continue to another loop

\- abort because the loop budget is exhausted or the evidence is too weak

## You will receive:

\- the user query

\- factual verifier output for each candidate

\- room-binding facts derived from room\_info and candidate pose

\- support-relation metrics

\- support-detection summary

\- loop history and current shortlist

## Important rules:

\- The verifier output is factual only. Do not expect it to decide whether the loop should continue.

\- Determine room compatibility from room\_info / pose facts, not from image semantics alone.

\- Prefer accepting when one candidate is clearly strongest across visual facts, support relations, room binding, and loop-history stability.

\- Missing confirmation is not the same as contradiction. Treat ‘unclear‘, ‘ambiguous‘, or absent visual confirmation as weak evidence, not negative evidence.

\- Only explicit mismatches or concrete contradictory facts should block acceptance. For example, "not glass" is blocking, but "material not fully visible" is not.

\- For relation queries such as beside / near / against / under / on, structured 3D support-relation metrics can be sufficient even when the support object is not co-visible in the verifier image.

\- If the support target was not detected anywhere in the loop, treat relation evidence as unavailable rather than negative against a specific primary candidate.

\- When support evidence is globally unavailable, only continue if the next loop will concretely change support detection strategy, such as phrase retargeting or scope/frame changes.

\- If there is only one viable candidate, it already satisfies room and relation constraints, and the verifier shows no explicit contradiction, prefer accept over continue.

\- If multiple top candidates all satisfy the core constraints and remain hard to distinguish, treat this as local query ambiguity rather than a failure. In that case, prefer accepting the highest-scoring candidate instead of continuing to chase uniqueness.

\- Continue only when another loop is likely to change the ranking, reveal a missing competing candidate, or resolve a concrete contradiction that matters to the final decision.

\- Continue only when another loop could plausibly change the answer in a concrete way.

\- If the current loop already detected primary candidates, prefer scope/frame changes over phrase changes.

\- If the current loop detected zero primary candidates, a continue decision may split recovery between scope/frame and phrase changes.

\- If the loop budget is already exhausted, do not return continue.

\- Use next\_loop\_hints to focus the next loop on the best candidate(s) and deprioritize weak ones.

\- Do not repeat room guesses from visual evidence when room\_info / pose facts already determine room compatibility.

\- Keep the reason factual and concise. Do not add motivational language or speculative claims.