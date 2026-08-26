# DoublesEval: Diagnosing Multi-Agent Tactical Reasoning in Vision-Language Models via Professional Doubles Badminton

Jintao Cheng<sup>1</sup>   
6jchengau@connect.ust.hk   
<sup>2</sup>Weibin Li<sup>2</sup>   
yc67383@um.edu.mo   
<sup>1</sup> The Hong Kong University of Science   
and Technology   
Hong Kong, China

<sup>2</sup> University of Macau Macau, China

## Abstract

Visual Language Models (VLMs) excel at describing visible scene content but struggle to reason about dynamic multi-agent interactions, where action semantics depend on coordinated roles and spatial-temporal dependencies. We formalize this capability as multi-agent tactical reasoning and introduce DoublesEval, a diagnostic evaluation framework that leverages professional doubles badminton as a structurally tractable testbed. DoublesEval employs a key-moment-based protocol that decomposes rallies into tactically salient instants and probes models across four interpretable dimensions: atomic recognition, intra-segment composite understanding, cross-segment causal reasoning, and high-level tactical abstraction. This design isolates where reasoning fails, rather than merely measuring answer correctness. To address observed failure modes, we propose TacticCheck, a lightweight constraint-guided test-time consistency checker that reranks candidate answers using the model’s own lower-level tactical predictions, requiring no parameter updates or ground-truth labels at inference time. Evaluating four representative open-source VLMs on 60 curated rallies (yielding ∼9.6K structured instances) via a zero-shot protocol, we find that models remain weak across all diagnostic levels, with especially clear bottlenecks in spatial state, interaction binding, and terminal evidence. TacticCheck delivers consistent gains across all evaluated models, while still leaving a substantial gap to robust tactical reasoning. These results highlight the need for structured, interaction-aware evaluation paradigms for next-generation VLMs. The source code is available in our GitHub repository.

## Introduction

Recent advancements in Visual Language Models (VLMs) have substantially elevated performance on visual question answering, video captioning, and multimodal reasoning, largely attributable to large-scale vision-language pretraining and the integration of more capable temporal encoders [4, 14, 15]. Despite these gains, model capabilities remain predominantly anchored to what is visible—static objects, isolated scenes, or short-horizon actions—rather than what is happening between agents over time.

![](images/bf642efd68e69e0c5d12781ec88db5f3e4a81034cf84d4e9af063d461760c919.jpg)  
Figure 1: Comparison between existing action-level evaluations and our proposed DoublesEval framework. Left: Current VLM benchmarks primarily focus on perception-level surface descriptions, such as isolated action recognition (e.g., distinguishing a smash from a clear). Middle: The core semantic gap lies in multi-agent tactical coordination, which requires mapping atomic actions to dynamic role attributions, spatial coverage constraints, and causal chains. Right: DoublesEval addresses this by introducing a diagnostic framework tailored for tactical-level multi-agent reasoning, systematically probing models across four structured reasoning dimensions.

Many real-world visual scenarios are fundamentally interactive: multiple agents move simultaneously, exchange roles, and coordinate their behavior, and the meaning of any single action depends on this collective context. We refer to this capability as multi-agent tactical reasoning—jointly modeling agent identity, spatial configuration, temporal evolution, and role-dependent interaction. Prevailing video evaluation protocols and datasets [2, 12] are predominantly designed to assess individual agent actions or global event classifications. Consequently, they rarely stress-test a model’s capacity for interactive, context-dependent reasoning, such as evaluating coordinated coverage or rotation necessity. As a result, recurring failure modes in role attribution, spatial coverage, and cross-event causal reasoning remain largely invisible under current evaluation protocols—even in recent domain-specific efforts like FineBadminton [10], which emphasizes fine-grained action recognition over the structured, multi-agent tactical coordination that recent analyses show VLMs struggle with [27, 30].

We argue that professional doubles badminton offers a near-ideal testbed for studying this gap. It is, in effect, a structurally tractable multi-agent tactical environment: only four players on a strictly bounded court, with well-defined team structure, rapid role transitions (attack/defense/rotation/coverage), and dense causal chains where one shot mechanically constrains the next. Unlike team sports such as football or basketball—where the agent count and spatial scale make ground-truth role labeling prohibitive [7]—doubles badminton is small enough to annotate exhaustively, yet rich enough to require genuine tactical understanding rather than surface-level description [10].

Building on this observation, we introduce DoublesEval, a diagnostic framework for evaluating zero-shot multi-agent tactical reasoning of VLMs on professional doubles badminton videos (see Figure 1). Instead of coarse video-level QA, DoublesEval adopts a keymoment-based protocol: each rally is decomposed into tactically salient instants (e.g., serves, defensive transitions, and smash returns) and probed along four diagnostic dimensions: (1) single-field atomic recognition, which examines basic perception of players, court regions, and actions; (2) intra-segment composite understanding, which evaluates whether models can combine multiple visual cues within a local segment; (3) cross-segment causal reasoning, which probes connections between earlier tactical decisions and later developments; and (4) high-level tactical semantic abstraction, which assesses whether models can summarize broader tactical patterns. This decomposition lets us pinpoint where reasoning breaks down, not merely whether the final answer is correct.

Motivated by the failure patterns surfaced by this diagnosis, we further propose Tactic-Check, a lightweight constraint-guided test-time consistency checker. Instead of retraining the model or prompting it to generate a new explanation, TacticCheck uses the model’s own lower-level predictions as a tactical checklist and reranks higher-level candidate answers by penalizing schema-level contradictions with this checklist. Mirroring how a human coach would sanity-check a tactical reading, the method requires no parameter updates and no ground-truth labels during inference, providing a direct stress test of whether explicit tactical consistency can mitigate multi-agent reasoning errors.

We instantiate DoublesEval on 60 curated professional doubles badminton rallies, yielding approximately 9,600 structured evaluation instances across the four diagnostic dimensions. To handle the open-ended nature of free-form VLM outputs, we adopt a two-stage evaluation protocol: four representative open-source VLMs—Qwen2.5-VL [24], Qwen3- VL [1], VideoLLaMA3 [29], and Molmo2 [6]—are first prompted to generate zero-shot short answers across all diagnostic prompts. Subsequently, GPT-5 [19] is employed as a structured label-mapping judge to align these free-form responses to standardized evaluation categories, effectively mitigating lexical variance without introducing subjective scoring. This protocol reveals recurring failures in spatial state, interaction binding, score detail, and cross-segment causality—failure modes that conventional video evaluation protocols are not designed to localize. TacticCheck yields consistent gains across all evaluated models, indicating that explicit tactical consistency can meaningfully mitigate—though not eliminate— structural reasoning errors.

Our contributions are summarized as follows:

• We formalize multi-agent tactical reasoning as a distinct evaluation target for VLMs and introduce DoublesEval, a key-moment-based diagnostic framework built from 60 professional doubles badminton rallies yielding ∼9.6K instances organized around four interpretable reasoning dimensions.

• We systematically evaluate representative open-source VLMs, exposing recurring failure modes in player role attribution, spatial localization, and cross-segment causal reasoning that prior video evaluation protocols do not surface.

• We propose TacticCheck, a constraint-guided test-time consistency checker that delivers consistent zero-shot improvements without parameter updates or ground-truth labels during inference.

## 2 Related Work

## 2.1 Diagnostic Evaluation of Vision-Language Models

While large-scale benchmarks have successfully tracked the upward trajectory of visionlanguage models (VLMs), aggregate metrics often obscure systematic reasoning failures. A growing line of work has therefore shifted toward diagnostic evaluations that isolate specific cognitive bottlenecks. Early investigations demonstrated that contrastive VLMs frequently degenerate into bag-of-words representations, exhibiting pronounced deficits in attribute binding and relational composition [17, 20, 27]. Subsequent studies on instruction-tuned multimodal LLMs have further revealed that even state-of-the-art architectures struggle with fine-grained visual grounding [22] and elementary spatial or geometric reasoning that humans perform effortlessly [18]. In the video domain, recent diagnostic suites such as Temp-Compass [16] and Video-MME [8] have effectively exposed vulnerabilities in temporal ordering, event localization, and long-horizon dependency modeling. Despite these advances, existing diagnostic protocols remain overwhelmingly anchored to single-agent trajectories or global event descriptors. They rarely interrogate whether a model can disentangle the interdependent dynamics of multiple agents—where the semantic validity of an action is contingent on partner roles, spatial coverage, and cross-temporal causality. DoublesEval addresses this blind spot by formalizing multi-agent tactical reasoning as a distinct diagnostic target.

## 2.2 Video Understanding in Badminton and Racket Sports

Computational analysis of badminton has evolved in tandem with broader advances in sports video understanding. Foundational efforts primarily targeted low-level perception, focusing on shuttlecock and player tracking [3, 11, 28] or stroke-level action classification from broadcast footage [5, 9]. As annotation granularity improved, research shifted toward modeling tactical units, decomposing rallies into structured sequences annotated with shot types, player coordinates, and point outcomes. The ShuttleSet series [25, 26] and the associated CoachAI Badminton Challenge [13, 23] have been instrumental in this transition, though predominantly within the singles domain. More recent initiatives have begun to incorporate doubles-specific dynamics, expanding annotation coverage to include partner coordination and fine-grained tactical descriptors [10]. Nevertheless, these resources are fundamentally optimized for supervised, task-specific pipelines—such as stroke classification, trajectory forecasting, or win-probability estimation—rather than for probing the open-ended, zeroshot reasoning capabilities of contemporary VLMs. Crucially, the complex interplay of role switching, court-coverage division, and role-dependent causal chains in doubles play has not been leveraged as a structured diagnostic signal for multimodal models. By repurposing professional doubles rallies into a key-moment-based evaluation protocol, DoublesEval bridges this methodological divide, offering the first diagnostic framework tailored to multi-agent tactical reasoning in sports video.

## 3 Methodology

## 3.1 Task Formulation: Multi-Agent Tactical Reasoning

Formal Task Definition. As illustrated in Figure 2, let V denote a professional doubles badminton rally video containing four players partitioned into two teams. The video is temporally segmented into an ordered sequence of key moments $\mathcal { M } = \{ m _ { 1 } , m _ { 2 } , . . . , m _ { K } \}$ , where each $m _ { k }$ corresponds to a tactically salient interval (e.g., serve, offensive transition, or defensive coverage). The multi-agent tactical reasoning task is formally defined as learning a conditional mapping $\mathcal { F } : ( \mathcal { V } , m _ { k } , q ) \mapsto \hat { y }$ , where q is a natural-language query that conditions the model on a specific reasoning target, and $\hat { y } \in \mathcal { V }$ is the predicted tactical inference. The target space $\mathcal { V }$ encompasses agent role assignment, spatial occupancy, temporal causality, and strategic pattern recognition. Under a zero-shot setting, model capability is quantified by the alignment between $\hat { y }$ and the expert-annotated ground truth $y ^ { \star }$ , with failures analyzed at the query level rather than aggregated video-level accuracy.

![](images/2984bf24a170d49c8824058f8adc61e43a98662f2d1fa11917424f4982851289.jpg)  
Figure 2: Overview of the DoublesEval diagnostic pipeline. By anchoring on the scoring moment, we extract structured tactical evidence across the bilateral causal chain. Layered reasoning probes (L1-L4) are then applied to precisely localize where VLM multi-agent reasoning fails, rather than merely evaluating outcome correctness.

Diagnostic Dimension Stratification. To systematically isolate capability bottlenecks within $\mathcal { F }$ , we partition the evaluation query space into four non-overlapping diagnostic dimensions, $\mathcal { D } = \{ D _ { 1 } , D _ { 2 } , D _ { 3 } , D _ { 4 } \}$ . This stratification is ordered by increasing reasoning depth and temporal scope, transforming a monolithic accuracy metric into a fine-grained diagnostic signal. Each dimension is operationally defined as follows:

L1: Single-Field Atomic Recognition. This dimension targets baseline visual grounding. Queries are constrained to a single key moment $m _ { k }$ and require the model to identify isolated factual elements, such as player identity, court zone occupancy, or discrete stroke types. Success here verifies that the model’s perceptual encoder can reliably resolve lowlevel visual signals without compositional interference.

L2: Intra-Segment Composite Understanding. This dimension probes multi-cue binding within a localized temporal segment. Queries require the simultaneous integration of co-occurring visual attributes (e.g., spatial relation + agent role + action state) to resolve interdependent facts. Failure at this stage typically indicates an inability to jointly model spatial configurations and agent states, even when temporal context is fixed.

L3: Cross-Segment Causal Reasoning. This dimension tests cross-temporal dependency modeling across $\geq 2$ sequential moments. Queries explicitly link a prior tactical decision (cause) to a subsequent player repositioning or stroke selection (effect). This isolates the model’s capacity to construct implicit causal chains and maintain state continuity over dynamic multi-agent interactions.

L4: High-Level Tactical Semantic Abstraction. This dimension evaluates holistic pattern induction over extended rally phases. Queries require summarizing recurrent strategic configurations (e.g., rotation schemes, coverage division rules, or offensive/defensive transitions) rather than discrete events. Performance here reflects the model’s ability to distill granular observations into structured tactical priors.

By attributing model failures to a specific dimension $D _ { i }$ , the protocol explicitly decouples perceptual grounding errors from higher-order reasoning breakdowns. This structured decomposition ensures that evaluation outcomes are interpretable, reproducible, and directly actionable for diagnosing the specific cognitive bottlenecks of modern VLMs.

## 3.2 DoublesEval: A Diagnostic Suite for Tactical Reasoning

## 3.2.1 Video Source and Rally Curation

Source matches. DoublesEval is curated from four professional doubles matches selected from the BWF World Tour Super 750 and Super 1000 series, comprising two men’s doubles, one women’s doubles, and one mixed doubles match. All broadcast footage is sourced from the official BWF TV YouTube channel at 1080p resolution and 25/30 fps.

Rally selection and tactical framing. From these four matches, we curate 60 rallies. Selection is guided by three criteria: (i) tactical diversity, ensuring coverage of attacking, defending, rotation, and net-play situations; (ii) minimum tactical depth, excluding single- and double-stroke rallies so that each example exercises the four diagnostic dimensions; and (iii) visual clarity, excluding rallies with camera cuts, replays, or substantial occlusion. Crucially, we treat each rally V as a holistic tactical unit: VLMs receive thefull rally video as temporal context to capture build-up and transition patterns, while diagnostic queries and expert annotations are anchored to tactically decisive segments (e.g., scoring windows, critical rotations, or phase transitions). This design ensures that evaluation targets concrete tactical decisions without sacrificing the global strategic context required for D3/D4 reasoning. We deliberately prioritize tactical diversity per rally over raw quantity, consistent with the diagnostic suite philosophy.

Rally statistics. The 60 curated rallies have an average duration of approximately 10 seconds, with S strokes per rally on average. Each rally is partitioned into multiple diagnostic segments for fine-grained probing; per-rally segment density and full distributional statistics are reported in Table 1.

Coverage trade-off. We acknowledge that four source matches yield a comparatively small number of distinct player pairs and tactical styles. This is a deliberate design choice: a diagnostic suite that prioritizes dense expert annotation and per-dimension interpretability necessarily trades player-coverage breadth for annotation depth. To prevent trivial playeridentity shortcuts, prompts that reference specific players are constructed using relative role descriptors (e.g., “the back-court player on the serving team”) rather than proper names; we further discuss this design in Section 3.2.2.

Table 1: Dataset statistics for the 60 curated rallies. Diagnostic segments are tactically decisive intervals (e.g., scoring windows, role transitions) anchored within the full-rally context. Instance counts per dimension reflect balanced coverage across the four diagnostic targets. Annotation reliability metrics are computed on the 30% expert-audited subset.
<table><tr><td>Statistic</td><td>Mean</td><td>Std</td><td>Range</td></tr><tr><td>Rally-level basics</td><td></td><td></td><td></td></tr><tr><td>Duration (seconds)</td><td>10.2</td><td>3.1</td><td>[6.5,18.9]</td></tr><tr><td>Strokes per rally (Ī)</td><td>12.4</td><td>3.8</td><td>[7,23]</td></tr><tr><td>Diagnostic segment density</td><td></td><td></td><td></td></tr><tr><td>Segments per rally</td><td>40.1</td><td>5.3</td><td>[32,51]</td></tr><tr><td>Avg. segment duration (seconds)</td><td>2.8</td><td>0.6</td><td>[1.8,4.2]</td></tr><tr><td>Instance distribution by dimension</td><td></td><td></td><td></td></tr><tr><td>L1: Atomic Recognition</td><td>2,400</td><td>(25.0%)</td><td></td></tr><tr><td>L2: Composite Understanding</td><td>2,400</td><td>(25.0%)</td><td></td></tr><tr><td>L3: Causal Reasoning</td><td>2,400</td><td>(25.0%)</td><td></td></tr><tr><td>L4: Tactical Abstraction</td><td>2,400</td><td>(25.0%)</td><td></td></tr><tr><td>Total instances</td><td></td><td>9,600</td><td></td></tr><tr><td>Annotation reliability (audit subset)</td><td></td><td></td><td></td></tr><tr><td>Intra-tier agreement (κ)</td><td>0.78-0.86</td><td></td><td></td></tr><tr><td>Tier-to-expert alignment (Acc)</td><td>82.5%–89.5%</td><td></td><td></td></tr></table>

## 3.2.2 Diagnostic Prompt Construction and Instance Design

Paired Query Architecture. The ∼9,600 evaluation instances are structured as paired diagnostic queries. For each rally v and diagnostic dimension ${ D _ { i } } \in \left\{ { D _ { 1 } , D _ { 2 } , D _ { 3 } , D _ { 4 } } \right\}$ , we instantiate two complementary prompts targeting the tactical state within the scoring window W : a Single-Choice Question (SCQ) and a Short-Answer Question (SAQ). This yields 4,800 SCQ instances and 4,800 SAQ instances. The SCQs provide a controlled evaluation baseline with exactly one correct option derived from the expert annotation schema, while the SAQs extend the same tactical context into open-ended generation to stress-test reasoning depth and justification capability without answer scaffolding.

SCQ Design and Construction. SCQ prompts are formulated to isolate specific perceptual or relational bottlenecks under strict single-choice constraints. Each question presents four mutually exclusive options: exactly one correct label sampled from the ground-truth annotation $\mathcal { L } _ { \nu , d }$ , and three distractors systematically constructed to reflect common model failure modes (e.g., swapped player roles, inverted court zones, or temporally misaligned causal links). This design ensures that SCQ accuracy measures model discrimination capability without ambiguity, serving as a reliable lower-bound for tactical comprehension in each dimension.

SAQ Design and Open-Ended Probing. SAQ prompts are derived by removing the singlechoice options and appending a directive that requires the model to generate a concise tactical justification or descriptive summary (e.g., “Explain why the back-court player rotated after the smash,” or “Describe the dominant coverage pattern during this phase.”). SAQs explicitly test the model’s ability to articulate causal chains, bind multi-cue observations, and abstract strategic patterns without option scaffolding. By construction, SAQ outputs exhibit high lexical variance, making direct string-matching evaluation unreliable and necessitating semantic alignment.

Label-Mapping for SAQ Evaluation. To enable rigorous comparison with the SCQ baseline and ground-truth labels, all SAQ responses are processed through the structured labelmapping judge introduced in Stage 2. The judge receives the SAQ output alongside the dimension-specific label space $\mathcal { L } _ { \nu , d }$ and maps the free-form text to the single semantically equivalent category. This step does not perform subjective scoring; it exclusively aligns open-ended reasoning to the standardized evaluation vocabulary established during annotation, ensuring that performance metrics reflect tactical comprehension rather than generative phrasing preferences. The paired SCQ/SAQ design allows us to quantify the “reasoning gap” between closed-set recognition and open-ended tactical articulation across all four dimensions.

## 3.2.3 Expert-Verified Scoring-Window Annotation

Scoring window as annotation anchor. For each rally V, we define a scoring window W comprising the final scoring stroke and the 3–5 immediately preceding shots. While VLMs ingest the full rally video as temporal context to capture long-horizon tactical evolution, expert annotations and diagnostic queries are exclusively anchored to the tactical state within W. This design mirrors professional coaching analysis: the scoring formation dictates the point outcome, but its tactical validity and intent are contingent on the immediate preceding sequence. Consequently, W serves as the unified annotation target for all diagnostic dimensions, with L1/L2 probing the immediate state within W, and D3/D4 evaluating the causal and strategic links from earlier phases of V to the configuration in W.

Two-tier annotation pipeline. To balance coverage breadth and tactical fidelity, annotation is organized into a structured two-tier pipeline. Thefirst tier consists of two experienced badminton analysts who independently annotate all 60 rallies. The second tier comprises two certified National Level-1 Athletes (the highest non-professional competitive tier), who audit a stratified 30% subsample (≈ 18 rallies) balanced across disciplines. Level-1 annotators adjudicate discrepancies, correct tactically implausible labels, and establish the gold standard for the audited subset. Their expertise ensures that complex role attributions and coverage patterns are grounded in domain-accurate reasoning rather than superficial visual cues.

Tool and schema. Annotation is conducted via Label Studio[21], where experts assign dimension-specific tactical labels to the state within W. These labels directly populate the closed answer spaces $\mathcal { L } _ { k , j }$ for MCQ construction and serve as semantic reference targets for SAQ label-mapping. To prevent trivial identity-based shortcuts, all player references in prompts and labels strictly use relative tactical descriptors (e.g., “the back-court player on the serving team”) rather than proper names.

![](images/2247b16bfdcad8e584db4e5175ad52838c1c4505e0876ab7d6a92ffcc9e99128.jpg)  
Figure 3: Two-Stage Evaluation and TacticCheck Pipeline. The framework processes key moments to generate two complementary zero-shot outputs: SCQs (evaluated via exact match) and SAQs (evaluated by a GPT-5 Judge). The diagnosed bottlenecks then inform a test-time checking loop, where a structured tactical checklist and constraint prompts guide the VLM to iteratively revise its initial answers.

Procedure and quality control. The two first-tier annotators label each rally independently. Label Studio automatically surfaces per-window label discrepancies for manual reconciliation. Within the 30% audit subset, Level-1 reviewers override consensus labels when tactical reasoning is violated; this expert-corrected version serves as the gold standard. For the remaining 70%, resolved consensus labels are used as working ground truth. Inter-annotator reliability on the audited subset shows substantial-to-almost-perfect intra-tier agreement (Cohen’s $\kappa \in [ 0 . 8 0 , 0 . 8 7 ]$ for perceptual labels, $\kappa \in [ 0 . 7 4 , 0 . 8 2 ]$ for role attribution) and high tier-to-expert alignment (> 87% accuracy for stroke intent, > 79% for tactical role assignment). These metrics confirm that the tiered pipeline yields consistent, domainvalidated annotations suitable for zero-shot diagnostic evaluation, while acknowledging the inherent subjectivity of tactical role attribution in dynamic doubles play.

## 3.3 Two-Stage Evaluation Protocol

## 3.3.1 Stage 1: Zero-Shot VLM Inference

As illustrated in Figure 3, we evaluate four representative open-source VLMs: Qwen2.5- VL, Qwen3-VL, VideoLLaMA3, and Molmo2. Each model processes the full rally video $V _ { \nu }$ as temporal context, paired with dimension-conditioned diagnostic prompts that explicitly target the tactical state within the scoring window $W _ { \nu }$ . Inference is conducted in a zero-shot setting with deterministic decoding (temperature $T = 0 . 1$ , top- $\cdot p = 0 . 9$ , max output tokens $= 6 4 )$ . The models produce free-form textual responses ${ A } _ { \nu , d } ^ { ( 0 ) }$ for each rally v and diagnostic dimension $d \in \{ D _ { 1 } , D _ { 2 } , D _ { 3 } , D _ { 4 } \}$

## 3.3.2 Stage 2: GPT-5 Label-Mapping Judge

Open-ended VLM outputs exhibit substantial lexical variance, making direct string matching unreliable. To standardize evaluation, we employ GPT-5 as a structured label-mapping judge. The judge receives a fixed system prompt specifying: (i) the dimension-specific target label space $\mathcal { L } _ { \nu , d }$ derived from the expert annotation of $W _ { \nu }$ , (ii) the VLM’s free-form response ${ A } _ { \nu , d } ^ { ( 0 ) }$ , and (iii) a strict output schema (JSON with fields mapped\_label and confidence\_score). GPT-5 performs semantic alignment rather than subjective scoring, mapping each open-ended response to the predefined evaluation vocabulary established during annotation. This step ensures consistent, reproducible label assignment across all evaluated models without introducing generative phrasing bias.

## 3.3.3 Judge Reliability

To validate the mapping fidelity of the LLM judge, we conduct a human-in-the-loop verification on a stratified random subset of $N _ { \mathrm { e v a l } } = 4 8 0$ instances (5% of the full set). Two domain experts independently assign labels, and we compute inter-annotator agreement (Cohen’s $\kappa = 0 . 8 6 )$ and judge-expert agreement $( \kappa = 0 . 8 1 )$ . Discrepancies between the judge and experts are resolved via majority voting or expert arbitration. The judge demonstrates strong alignment with human annotations across all four dimensions, with the lowest agreement observed in Tactical Abstraction $( \kappa = 0 . 7 3 )$ , reflecting the inherent ambiguity of high-level strategic labeling. All downstream metrics are computed using the judge-mapped labels after reliability verification.

## 3.4 TacticCheck: Constraint-Guided Test-Time Consistency Checking

## 3.4.1 Motivation

The diagnostic analysis shows that many errors are not isolated wrong labels, but inconsistent tactical chains: a model may predict a plausible final outcome while assigning incompatible team formation, interaction role, or terminal stroke evidence. We therefore use the lowerlevel diagnostic fields as a test-time consistency signal. The goal is modest: TacticCheck does not teach the model new badminton knowledge, and it does not use ground-truth labels. It only checks whether a higher-level answer is compatible with the model’s own tactical reading of the scoring moment.

## 3.4.2 Checklist Extraction

For each rally, we first collect the model’s zero-shot predictions on L1 fields and convert them into a compact tactical checklist

$$
\boldsymbol { z } = \{ \hat { s } _ { \mathrm { n e a r } } , \hat { s } _ { \mathrm { f a r } } , \hat { b } _ { \mathrm { n e a r } } , \hat { b } _ { \mathrm { f a r } } , \hat { c } , \hat { r } \} ,\tag{1}
$$

where ${ \hat { s } } _ { \mathrm { n e a r } }$ and $\hat { s } _ { \mathrm { f a r } }$ denote near-team and far-team spatial states, $\hat { b } _ { \mathrm { n e a r } }$ and $\hat { b } _ { \mathrm { f a r } }$ denote interaction cues, $\hat { c }$ denotes outcome attribution, and $\hat { r }$ denotes score-related terminal evidence. These fields are predicted by the same VLM under the same zero-shot setting. No expert annotation is used in this step.

## 3.4.3 Constraint-Guided Option Reranking

For a higher-level single-choice question with candidate options $\mathcal { O } = \{ o _ { 1 } , . . . , o _ { N } \}$ , Tactic-Check maps each option to a lightweight tactical signature $\phi ( o _ { i } )$ using the predefined answer

schema. We then compute a consistency penalty between the option signature and the checklist:

$$
\mathrm { P e n a l t y } ( o _ { i } , z ) = \lambda _ { s } \mathbf { 1 } [ \phi _ { s } ( o _ { i } ) \ne z _ { s } ] + \lambda _ { b } \mathbf { 1 } [ \phi _ { b } ( o _ { i } ) \ne z _ { b } ] + \lambda _ { c } \mathbf { 1 } [ \phi _ { c } ( o _ { i } ) \ne z _ { c } ] + \lambda _ { r } \mathbf { 1 } [ \phi _ { r } ( o _ { i } ) \ne z _ { r } ] ,\tag{2}
$$

where $\nsim$ indicates a schema-level incompatibility, such as a candidate requiring a far-team interaction when the checklist indicates a near-team interaction, or a score explanation that contradicts the predicted terminal evidence. The final answer is selected by reranking candidate options:

$$
\hat { o } = \arg \operatorname* { m a x } _ { o _ { i } \in \mathcal { O } } \left[ \mathrm { B a s e S c o r e } ( o _ { i } ) - \mathrm { P e n a l t y } ( o _ { i } , z ) \right] .\tag{3}
$$

In practice, BaseScore is derived from the model’s original answer preference when available; otherwise the original selected option is retained as the default and only replaced when another option has a clearly lower consistency penalty. This conservative rule reduces unnecessary answer changes.

## 3.4.4 Scope and Limitations

TacticCheck is a lightweight test-time reranking method. It requires no parameter update, no additional training data, and no access to ground-truth labels during inference. Its benefit comes from down-ranking answers that contradict the model’s own lower-level tactical evidence. However, the method is bounded by the quality of the checklist: if the L1 predictions are wrong, the consistency signal can also be wrong. We therefore treat TacticCheck as a diagnostic intervention that tests whether explicit tactical consistency helps, rather than as a complete solution to multi-agent tactical reasoning.

## 3.5 Evaluation Metrics

We treat every prompt instance as an evaluation unit after applying the GPT-5 label mapper. Single-choice questions (SCQ) are scored by exact matching the predicted option against the gold label. Short-answer questions (SAQ) are first mapped to the same discrete label space and then scored identically to SCQs. Dimension-wise accuracy $\operatorname { A c c } _ { d }$ averages the SCQ and SAQ accuracies within each level d (the two formats appear in one-to-one pairs per rally), and the overall macro score is the unweighted mean across the four levels.

TacticCheck operates only on SCQs because the reranker requires an explicit option set. Its reported gains therefore compare the SCQ accuracy before reranking (“Base”) and after reranking on the identical SCQ subset; SAQ accuracy remains unchanged and is excluded from those deltas. Unless otherwise noted, tables that mix Base and TacticCheck results adopt this SCQ-only accounting for fair comparison. Statistical significance for the SCQ deltas is estimated via paired bootstrap over SCQ instances (1,000 resamples, $p < 0 . 0 5 )$ . Finally, failure-mode breakdowns are computed on the pre-reranking SCQ predictions so that error categories align with the structured distractor taxonomy.

## 4 Experiments

## 4.1 Overall Zero-Shot Performance

We first evaluate all models in a zero-shot setting. Each model receives the full rally video and answers diagnostic questions anchored to the scoring moment. Unless otherwise stated, level-wise results aggregate both single-choice questions and label-mapped short-answer questions. Table 2 reports level-wise accuracy, and Figure 4 visualizes the same trend.

Table 2: Zero-shot accuracy over single-choice and short-answer questions (%). Accuracy remains low across all four diagnostic levels. L1 is easier than the other levels for most models, but even atomic recognition is far from saturated.
<table><tr><td>Model</td><td>Overall</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td></tr><tr><td>Qwen2.5VL</td><td>27.57</td><td>33.50</td><td>27.50</td><td>23.80</td><td>25.50</td></tr><tr><td>Qwen3VL</td><td>29.00</td><td>35.00</td><td>29.00</td><td>25.00</td><td>27.00</td></tr><tr><td>VideoLLaMA3</td><td>32.30</td><td>35.65</td><td>28.79</td><td>29.33</td><td>35.43</td></tr><tr><td>Molmo2</td><td>31.31</td><td>37.50</td><td>29.35</td><td>26.11</td><td>26.86</td></tr></table>

![](images/a08777b2ed9e488d92ab19edc5d6250b1f1a915d82f44a4e4f2357d1518173fe.jpg)  
Figure 4: Level-wise zero-shot performance over single-choice and short-answer questions. The benchmark does not only test final answer correctness: it separates atomic recognition, composite understanding, causal reasoning, and tactical abstraction.

Overall performance falls in a narrow band, from 28.65% to 32.35%. This indicates that current VLMs are not simply failing on one isolated model family. The weakness is shared across architectures under the same diagnostic protocol. Performance also does not form a perfectly monotonic curve over levels: for example, VideoLLaMA3 performs relatively well on L4. We therefore do not interpret the levels as a strict difficulty ladder for every model. Instead, the key observation is that all levels expose non-trivial errors, including the visually grounded L1 questions.

## 4.2 Diagnostic Analysis

We next ask why zero-shot performance is low. DoublesEval is designed for this purpose: each question is tied to an interpretable capability, so errors can be localized rather than only counted. Figure 5 summarizes accuracy by capability group.

The main bottleneck is not basic visibility alone. Models achieve their best capability scores on temporal phase recognition, but drop on spatial state and interaction binding. These two categories require the model to identify who occupies which tactical role and how the two sides are coupled. Score detail is also difficult, especially when the answer depends on the final stroke pattern rather than only the winning side. This pattern supports the central motivation of DoublesEval: professional doubles badminton is compact enough to observe, but still hard because the answer depends on relations between agents.

![](images/c6d82e926f4c26edbc6ce96dd5433df937fb6e59a4f1008ff2bf5e8ce841439d.jpg)  
Figure 5: Capability-level diagnosis. Temporal phase is relatively easier, while spatial state, interaction binding, and score-detail reasoning are consistently weak. This suggests that models often see parts of the scene but fail to bind players, teams, space, and outcome into a coherent tactical state.

Figure 6 gives a more fine-grained view of L1. Phase context and scoring side are comparatively stable, but interaction fields are much weaker. Stroke pattern is the hardest L1 field for all models. This is important because later causal questions often depend on these same fields. If the model misreads the terminal stroke or binds the wrong player to the wrong interaction, higher-level reasoning has little chance to be correct.

The field-level results explain why aggregate accuracy alone is insufficient. A model may select or generate a plausible answer locally while still relying on an unstable tactical reading of the scoring moment. These errors are concentrated in the same places highlighted above: spatial state, interaction binding, and terminal evidence. This motivates TacticCheck, a lightweight test-time reranking method that does not try to teach the model new badminton knowledge, but instead checks whether each candidate answer is consistent with the model’s own lower-level tactical reading.

## 4.3 Improved Performance with Test-Time Consistency Checking

Following Section 3.4, TacticCheck first uses the model’s own L1 predictions as a tactical checklist, including phase, team structure, interaction cues, and score evidence. For higherlevel single-choice questions, it reranks candidate options by penalizing options whose tactical signatures conflict with this checklist. The method uses no ground-truth labels during reranking and does not update model parameters.

As shown in Table 3, TacticCheck improves overall accuracy by 6.55–7.83 percentage 11 points across the four models. The gain is consistent, but we avoid interpreting it as a complete solution. First, TacticCheck accuracy remains below 40% for all models. Second, the method changes some originally correct answers into incorrect ones. This is expected for a test-time consistency reranker: when the model’s own checklist is wrong, reranking can propagate the error. The result is therefore best understood as evidence that explicit tactical consistency helps, not that it solves multi-agent tactical reasoning.

These results support the overall story of the benchmark. DoublesEval reveals that the main weakness is not only recognizing badminton actions, but binding visible evidence into a coherent multi-agent tactical chain. A lightweight consistency reranker can recover part of this gap, but the remaining errors show that future VLMs still need stronger interaction modeling rather than only larger visual encoders or better prompt formatting.

![](images/b3c628188b9ca573cde8b98d861fd7717d43f7012c23f1024933970d11ddd767.jpg)  
Figure 6: L1 field-level diagnosis. Even within atomic recognition, not all fields are equally easy. Phase and scoring side are more reliable, while near/far interaction fields and stroke pattern remain difficult.

Table 3: Test-time improvement with TacticCheck (%). TacticCheck improves all models, but also introduces a smaller number of right-to-wrong flips. This confirms that consistency checking is useful but not error-free.
<table><tr><td>Model</td><td>Base</td><td>TacticCheck</td><td>Δ</td><td>W→R</td><td>R→W</td></tr><tr><td>Qwen2.5VL</td><td>27.57</td><td>35.20</td><td>+7.63</td><td>12.41</td><td>4.78</td></tr><tr><td>Qwen3VL</td><td>29.00</td><td>37.20</td><td>+8.20</td><td>13.46</td><td>5.26</td></tr><tr><td>VideoLLaMA3</td><td>32.30</td><td>39.43</td><td>+7.13</td><td>12.84</td><td>5.71</td></tr><tr><td>Molmo2</td><td>31.31</td><td>39.14</td><td>+7.83</td><td>12.97</td><td>5.14</td></tr></table>

## 5 Conclusion

In this work, we introduced DoublesEval, a diagnostic framework that exposes multi-agent tactical reasoning limitations in VLMs through structured evaluation on professional doubles badminton. Our experiments reveal consistent bottlenecks in spatial state recognition, interaction binding, and causal reasoning across four diagnostic levels.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[2] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition, pages 961– 970, 2015.

[3] Jintao Cheng, Kang Zeng, Zhuoxu Huang, Xiaoyu Tang, Jin Wu, Chengxi Zhang, Xieyuanli Chen, and Rui Fan. Mf-mos: A motion-focused model for moving object segmentation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 12499–12505. IEEE, 2024.

[4] Jintao Cheng, Weibin Li, Jiehao Luo, Xiaoyu Tang, Zhijian He, Jin Wu, Yao Zou, and Wei Zhang. Scale, don’t fine-tune: Guiding multimodal llms for efficient visual place recognition at test-time. IFAC-PapersOnLine, 59(35):97–102, 2025.

[5] Jintao Cheng, Weibin Li, Zhijian He, Jin Wu, Chi Man Vong, and Wei Zhang. Beyond first-order: Learning riemannian geometries for invariant visual place recognition. arXiv preprint arXiv:2602.00841, 2026.

[6] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. arXiv preprint arXiv:2601.10611, 2026.

[7] Adrien Deliège, Anthony Cioppa, Silvio Giancola, Jeroen Meester, Moein Seikavandi, Luc Van Gool, et al. Soccernet-v2: A dataset and benchmarks for holistic understanding of broadcast soccer videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 2166–2176, 2021.

[8] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The firstever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118, 2025.

[9] Anurag Ghosh, Suriya Singh, and CV Jawahar. Towards structured analysis of broadcast badminton videos. In 2018 IEEE Winter Conference on Applications ofComputer Vision (WACV), pages 296–304. IEEE, 2018.

[10] Xusheng He, Wei Liu, Shanshan Ma, Qian Liu, Chenghao Ma, and Jianlong Wu. Finebadminton: A multi-level dataset for fine-grained badminton video understanding. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 12776–12783, 2025.

[11] Yu-Chuan Huang, I-No Liao, Ching-Hsuan Chen, Tsì-Uí <sup>˙</sup>Ik, and Wen-Chih Peng. Tracknet: A deep learning network for tracking high-speed and tiny objects in sports

applications. In 2019 16th IEEE international conference on advanced video and signal based surveillance (AVSS), pages 1–8. IEEE, 2019.

[12] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.

[13] Weibin Li, Jiazheng Huang, Bohuan Xue, Wenhao Shao, Yijun Liu, and Xiaoyu Tang. Mdmu: Multimodal dynamic mamba unet for multimodal sentiment analysis. In 2025 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2025.

[14] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[15] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[16] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731– 8772, 2024.

[17] Jiehao Luo, Jintao Cheng, Qiuchi Xiang, Jin Wu, Rui Fan, Xieyuanli Chen, and Xiaoyu Tang. Overlapmamba: A shift state space model for lidar-based place recognition. IEEE Robotics and Automation Letters, 10(8):8380–8387, 2025.

[18] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. In Proceedings of the Asian Conference on Computer Vision, pages 18–34, 2024.

[19] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[20] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visiolinguistic compositionality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238–5248, 2022.

[21] Maxim Tkachenko, Mikhail Malyuk, Andrey Holmanyuk, and Nikolai Liubimov. Label studio: Data labeling software, 2020-2025. URL https://github. com/HumanSignal/label-studio. Open source software available from https://github.com/HumanSignal/label-studio.

[22] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9568– 9578, 2024.

[23] Kuang-Da Wang, Yu-Tse Chen, Yu-Heng Lin, Wei-Yao Wang, and Wen-Chih Peng. The coachai badminton environment: Bridging the gap between a reinforcement learning environment and real-world badminton games. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 23844–23846, 2024.

[24] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

[25] Wei-Yao Wang, Wei-Wei Du, Wen-Chih Peng, and Tsi-Ui Ik. Shuttleset22: Benchmarking stroke forecasting with stroke-level badminton dataset. arXiv preprint arXiv, 2306, 2023.

[26] Wei-Yao Wang, Yung-Chang Huang, Tsi-Ui Ik, and Wen-Chih Peng. Shuttleset: A human-annotated stroke-level singles dataset for badminton tactical analysis. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5126–5136, 2023.

[27] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? arXiv preprint arXiv:2210.01936, 2022.

[28] Kang Zeng, Hao Shi, Jiacheng Lin, Siyu Li, Jintao Cheng, Kaiwei Wang, Zhiyong Li, and Kailun Yang. Mambamos: Lidar-based 3d moving object segmentation with motion-aware state space model. In Proceedings of the 32nd ACM international conference on multimedia, pages 1505–1513, 2024.

[29] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.

[30] Zheyuan Zhang, Fengyuan Hu, Jayjun Lee, Freda Shi, Parisa Kordjamshidi, Joyce Chai, and Ziqiao Ma. Do vision-language models represent space and how? evaluating spatial frame of reference under ambiguities. arXiv preprint arXiv:2410.17385, 2024.