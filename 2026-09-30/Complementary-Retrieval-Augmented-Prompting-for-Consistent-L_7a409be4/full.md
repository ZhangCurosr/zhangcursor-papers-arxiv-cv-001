# Complementary Retrieval-Augmented Prompting for Consistent Long-Form Video Generation

Xianghan Wei, Xiaoda Yang, Zhi Wang, An Pan, Daoan Zhang, Huayi Zhang, Yan Zhang, Wei Xu, Zishun Liao, Jianwen Lou

## Abstract

While recent video foundation models excel at generating high-quality short videos, long-form video generation remains challenging because independently generated shots must preserve consistent characters, scenes, and objects throughout a story. Existing training-free approaches typically condition target shots using retrieved historical visuals. However, these references often sufer from severe informational mismatch, either introducing irrelevant contextual redundancy or failing to provide the complete set of elements required for the target shot. To resolve this, we present Complementary Retrieval-Augmented Prompting, an agentic framework that strategically aggregates a compact set of mutually supportive historical references to achieve complete and targeted conditioning for long-form video generation without retraining or modifying the underlying generator. Specifically, our framework explicitly models the visual elements required by each target shot by parsing the narrative script into a text-grounded visual element registry that tracks characters, objects, scenes, and their shot-level states. A VLM-annotated keyframe library further maps these elements to past visual observations. Guided by the required elements, our agent retrieves complementary references that maximize target-element coverage while minimizing historical noise. Finally, the retrieved references, structured element states, and grounding instructions are assembled into a unified prompt for the frozen video generator. This element-aware process provides comprehensive conditioning while remaining fully interpretable. Quantitative and qualitative evaluations on multi-shot story generation demonstrate improved cross-shot character, object, and scene consistency over memory-based and agentic retrieval baselines.

## Introduction

Recent video foundation models can synthesize visually rich short clips from text, images, videos, and other multimodal prompts (Gao et al. 2025; Kling Team 2025; Huang et al. 2025; Bao et al. 2024; Alibaba Cloud 2025; Google Deep-Mind 2025). Rather than training a specialized long-video model, creators can decompose a story into shots, generate each shot with an of-the-shelf reference-conditioned model, and assemble the resulting clips into a complete narrative. This shot-by-shot workflow is already widely used in the production of short dramas and motion comics because it is flexible, interpretable, and compatible with rapidly evolving commercial video APIs.

However, long-form generation is not simply a matter of stitching short clips together. When each shot is generated independently, recurring characters may drift in appearance, objects may disappear or change shape, and scenes may be reintroduced with inconsistent layouts (Zhang et al. 2025; He et al. 2026). Reference images and videos can mitigate these issues, but they also introduce a new bottleneck: before generating each shot, the system must decide which historical references are useful, which visual elements in those references should be preserved, and which stale elements should be suppressed. In practice, this reference selection and prompt construction process is still largely manual, especially when multiple characters, objects, and locations recur over long temporal gaps.

Existing long-form video methods address consistency from diferent directions, but they leave this referencemanagement problem underexplored for frozen multimodal video generators. Training-based approaches propagate memory through attention, caches, latent retrieval, or learned conditioning (Meng et al. 2026; Wu et al. 2025; An et al. 2026; Luo et al. 2026; Zhang et al. 2025; Wei et al. 2026; Hu et al. 2026), which often incur high computational overhead and lack scalability. Agentic and explicit-memory pipelines improve controllability by maintaining scripts, entities, or generated keyframes (Lin et al. 2024; Zheng et al. 2024; Wu, Zhu, and Shou 2025; Huang et al. 2026; Liu et al. 2026; Zhou et al. 2026; Yin et al. 2026; Lai et al. 2026), but they often rely on predefined full-script analysis, entity-level canonical references, or manually designed prompt/reference layouts. These designs are powerful, yet less suited to incremental and interactive creation where users may extend or revise the story while generation proceeds.

We propose Complementary Retrieval-Augmented Prompting, an agentic framework that treats long-form video generation as an interpretable retrieval and multimodal prompting problem. The agent maintains two evolving evidence structures. First, a text-grounded visual element registry tracks characters, objects, and scenes introduced by the script, together with shot-level states such as shouldreference, should-exclude, optional, and newly introduced. Second, a VLM-annotated keyframe library records where these elements appear in generated shots, using both element-level annotations and holistic frame descriptions. Given a new shot, the agent retrieves a compact set of complementary historical keyframes that jointly cover the required visual elements while penalizing references that may introduce conflicting content. These selected references are then converted into a structured multimodal prompt that explicitly explains how each image should be used or avoided by the frozen generator.

This formulation has two practical advantages. First, it makes reference usage explainable: each selected frame can be traced to the visual elements it covers and the risks it may introduce. Second, it supports online and interactive generation: the registry and keyframe library are updated after each generated shot, so the system does not require a complete script or a fixed generation schedule in advance. Long-range visual consistency is handled by retrieval over historical evidence, while local temporal continuity between non-cut shots can be handled separately through proximal video references and lightweight boundary smoothing.

Our contributions are threefold:

• We introduce a training-free, online framework that maintains a visual element registry and a VLM-annotated keyframe library, turning generated history into structured and editable evidence without requiring the full script in advance.

• We formulate reference selection as a complementary coverage problem and develop a greedy retrieval algorithm that selects compact historical references covering target elements while suppressing conflicting content.

• We ground each retrieved frame through elementlevel and holistic guidance in a structured multimodal prompt. Experiments on EntityBench demonstrate improved cross-shot consistency over training-based methods and explicit workflow baselines.

## Related Work

## Training-based Long-Form Video Generation

Training-based methods embed cross-shot consistency into the generator itself. HoloCine (Meng et al. 2026) and Cine-Trans (Wu et al. 2025) jointly generate multiple shots and leverage cross-shot attention to preserve visual consistency, while autoregressive and memory-based approaches propagate selected frames, feature caches, attention sinks, reconstructed subjects, or retrieved latents across generation steps (An et al. 2026; Luo et al. 2026; Zhang et al. 2025; Wei et al. 2026; Chen et al. 2026; Hu et al. 2026). Although effective, these mechanisms require model-specific training or access to internal representations, limiting both their portability across video generators and the interpretability of their historical conditioning.

## Explicit and Agentic Long-Form Generation

Agentic pipelines organize existing generators through script decomposition, shot planning, context allocation, and iterative visual feedback (Lin et al. 2024; Zheng et al. 2024; Wu, Zhu, and Shou 2025; Huang et al. 2026; Liu et al. 2026). Entity-centric systems further maintain explicit states for recurring characters, objects, and scenes (Zhou et al. 2026; Yin et al. 2026). Meanwhile, reference-conditioned generators such as Seedance, Kling-Omni, Step-Video, and Vidu support multimodal prompting over text, images, and videos (Gao et al. 2025; Kling Team 2025; Huang et al. 2025; Bao et al. 2024), but long-form workflows still rely heavily on manual reference selection and prompting. GroundShot (Lai et al. 2026) reduces this burden by analyzing the full script, scheduling entity-reference shots first, and grounding subsequent shots on the resulting entity memory, but still relies on a predefined complete generation task.

Our framework instead supports incremental creation without a complete script or predefined generation order. It treats generated keyframes as VLM-annotated evidence, retrieves complementary frames under a limited reference budget, and explicitly grounds how each frame should be used while suppressing obsolete or conflicting content. This yields an interpretable and model-agnostic interface for frozen multimodal video generators.

## Method

## Overview

We study long-form story video generation by leveraging the exceptional visual quality and robust physical modeling capabilities of frozen short-video generators. Given a narrative script decomposed into a sequence of T shots $\boldsymbol { S } \ : = \ : \{ s _ { 1 } , . . . , { s _ { T } } \}$ , each shot s is generated through an independent inference call, and the generator retains no internal memory across shots. The central challenge is how to strategically construct a targeted multimodal prompt with historical references to preserve the consistency of characters, objects, and scenes across shots. To address this challenge, we present Complementary Retrieval-Augmented Prompting, an agentic framework designed to strategically orchestrate inference-stage historical conditioning. Instead of relying on passive, recency-biased retrieval, our method formalizes reference selection as an optimization problem where historical frames are selected if and only if they collectively maximize the coverage of the target shot’s required elements while minimizing informational redundancy. Under this formulation, an agent continuously tracks structured visual evidence and retrieves a compact set of complementary historical references to drive the generation loop. As illustrated in Figure 1, our framework consists of four sequential modules:

1. Narrative-Grounded Element Registry: Parses the narrative script into a structured registry tracking required elements and their shot-level states.

2. Element-Linked Keyframe Library: Employs a VLM to annotate and index representative historical keyframes, establishing explicit cross-modal mappings for a highdiversity candidate reference pool.

3. Element-Aware Complementary Retrieval: Executes a strategic agentic search to select a minimal, mutually supportive subset of historical references that covers target elements.

4. Unified Prompt Assembly: Compiles the retrieved references, structured states, and grounding instructions into a cohesive prompt for the frozen generator.

![](images/070776363c7ff3d8ba16baca99867d0a6bde355558b81091fdecebb3d2f1efd2.jpg)  
Figure 1: Overview of Complementary Retrieval-Augmented Prompting. An LLM updates the visual element registry and shotlevel plan; greedy coverage retrieves complementary keyframes from the VLM-annotated library; and a second LLM produces per-reference guidance for structured prompting. After generation, representative frames are mined and annotated before the library is updated. The dashed arrow denotes retrieval from the updated library for the next shot.

Shot Specification. We represent each shot $s _ { t }$ as a fourfield configuration tuple:

$$
s _ { t } = ( p _ { t } , d _ { t } , c _ { t } , r _ { t } ) ,\tag{1}
$$

where $p _ { t }$ is the foundational text prompt directing scenes, characters, and actions, and $d _ { t }$ denotes the target clip duration. The binary indicator $c _ { t } \in$ {true, false} specifies the relationship to the preceding shot: $c _ { t } = \mathrm { t r u e }$ indicates that the current shot does not continue from the preceding shot, whereas $c _ { t } = \mathrm { f a l s e }$ indicates that it continues from the end of the preceding shot. Finally, $r _ { t }$ represents optional external reference assets (e.g., initial character sheets) used to anchor identity; it is not used in our algorithm or experiments.

## Narrative-Grounded Element Registry

To transform an unstructured narrative script into explicitly computable constraints for consistent long-form synthesis, our framework maintains an Element Registry $\mathcal { R } = \mathrm { \bar { \{ e _ { i } \} } } _ { i = 1 } ^ { N }$ Instead of relying on implicit, text-driven prompt adherence, which often sufers from text-to-video drift, this registry operationalizes reusable storytelling elements in a centralized, structured metadata store. Each element $e _ { i } \in \mathcal { R }$ is defined as a tuple:

$$
\boldsymbol { e } _ { i } = ( \mathrm { i d } _ { i } , \mathrm { n a m e } _ { i } , \mathrm { t y p e } _ { i } , t _ { \mathrm { i n t r o } , i } , \mathrm { a t t r } _ { i } ) ,\tag{2}
$$

where ${ \mathrm { i d } } _ { i }$ is a stable unique identifier, name provides a concrete textual definition $( \mathrm { e . g . }$ , “the blond $ { \mathbf { b } }  { \mathbf { o y } } ^ { \prime } )$ $\gamma ^ { \mathfrak { s } } ) , \mathrm { t y p e } _ { i } \in$ {scene, character, object}, $t _ { \mathrm { i n t r o } , i }$ records the shot index of its first introduction, and $\mathrm { a t t r } _ { i }$ stores its persistent visual attributes.

Agentic Element Tracking. The LLM agent continuously tracks element metadata and assigns conditional states $y _ { i , t } \in$ Y = {REF, EXC, OPT, NEW} according to the target shot prompt $p _ { t }$ through a two-step procedure:

1. Registry Extension: The agent parses $p _ { t }$ to detect novel storytelling elements. Upon identification, a new element $e _ { i }$ is instantiated with $t _ { \mathrm { i n t r o } , i } = t _ { \mathrm { : } }$ , assigned $y _ { i , t } = \tt N E W .$ and appended to $\mathcal { R }$

2. State Assignment: For all pre-existing elements $\{ e _ { i } \in$ $\mathcal { R } \mid t _ { \mathrm { i n t r o } , i } < t \}$ , the agent evaluates the target narrative context and assigns a shot-level constraint state:

$y _ { i , t } = \tt R E F$ (should-reference): Explicit evidence indicates the element must appear in shot $s _ { t } .$ , instructing downstream modules to maximize its historical visual coverage.

$y _ { i , t } = \mathtt { E X C }$ (should-exclude): Explicit evidence indicates that the element should be absent from shot $s _ { t } ,$ instructing downstream modules to penalize and suppress historical visual leakage.

$y _ { i , t } = \mathsf { O P T }$ (optional-or-uncertain): The narrative provides no definitive evidence for presence or absence, permitting flexible contextual adaptation.

Newly introduced elements $( t _ { \mathrm { i n t r o } , i } ~ = ~ t )$ are marked as NEW and bypass retrieval because they have no prior visual observations. As illustrated in Figure 2(a), the resulting state vector $\mathbf { Y } _ { t } = \{ y _ { i , t } \} _ { i = 1 } ^ { | \mathcal { R } | }$ transforms the unstructured textual prompt $p _ { t }$ into an element-aware query representation that guides downstream complementary retrieval.

## Element-Linked Keyframe Library

To establish a structured and queryable visual memory, our framework constructs an Element-Linked Keyframe $L i -$ brary $\mathcal { L } = \{ f _ { k } \} _ { k = 1 } ^ { M _ { t } }$ , where $M _ { t }$ is the number of historical keyframes available before generating shot $s _ { t } .$ . The library establishes explicit cross-modal mappings to the active registry R. Rather than compressing historical video dynamics into holistic visual embeddings, the agent dynamically appends informative instances to ${ \mathcal { L } } ,$ forming a high-diversity candidate reference pool for downstream complementary matching.

Library Maintenance Dynamics. Inspired by the visual caching philosophies in StoryMem (Zhang et al. 2025), we store representative keyframes as reusable references. However, to preserve fine-grained asset diversity for subsequent complementary retrieval, our framework executes a deliberately relaxed caching protocol via a two-stage sequential gate:

1. Fidelity Filtering: Candidate frames from the newly synthesized clip are first evaluated via HPSv3 (Ma et al. 2025). Visually degraded, blurred, or artifact-heavy frames are discarded to ensure reference quality.

2. Temporal Drift Detection: For the remaining candidate pool, we compute semantic cosine distance using CLIP image features (Radford et al. 2021). A qualified frame is committed to L as a new keyframe $f _ { k }$ if and only if its similarity to the immediately preceding accepted keyframe within the same shot falls below a specific threshold, signaling a distinct transition in the visual state.

By comparing frames only with local within-shot history rather than performing aggressive global cross-shot deduplication, this protocol preserves diverse visual perspectives and gives the downstream retriever richer choices.

Dual-Level Structural Annotation. Once a keyframe $f _ { k }$ is committed to L, a VLM processes it to produce a dual-level semantic representation for downstream use:

1. Element-Level Annotations $( \mathbf { A } _ { k } ) \colon$ The VLM executes closed-set entity verification against R. For each tracked element $\begin{array} { r } { e _ { i } \in \mathcal { R } , \mathbf { A } _ { k } } \end{array}$ records its presence indicator $a _ { k , i } ~ \in ~ \{ 0 , 1 \}$ , a localized bounding box $b _ { k , i } ,$ and a categorical reference fidelity label $q _ { k , i } \in$ $\{ \mathtt { f u l l } , \mathtt { p a r t i a l } , \mathtt { w e a k } \}$ . A full element is complete and clearly identifiable; for a character, it additionally requires a clear frontal face. A partial element is identifiable but occluded, truncated, or, for a character, lacks a clear frontal face, whereas a weak element provides insuficient detail for reliable identity reference. For global environmental backdrops $( \mathrm { t y p e } _ { i } = \mathrm { s c e n e } )$ , the spatial region $b _ { k , i }$ defaults to the full image canvas.

2. Holistic Frame Description $( \mathbf { D } _ { k } ) \colon$ Conditioned on the narrative context, the VLM produces a comprehensive textual summary detailing the global composition, artistic style, core action, and plot-level relevance of the entire frame.

This two-tiered design decouples symbolic subset optimization from prompt context alignment. Structurally, the element-level matrix $\mathbf { A } _ { k }$ translates each frame into verifiable evidence for greedy retrieval, explicitly revealing which elements a frame covers and which excluded assets it might introduce. Semantically, the holistic descriptions $\mathbf { D } _ { k }$ provide natural-language context, enabling the prompt agent to articulate how the reference frame should be interpreted by the frozen generator.

![](images/ee14c506d8280955eb19b159bc4ad92f56991db07d5d413c5557e22bf4ccaa81.jpg)  
Figure 2: Complementary reference retrieval example. Top-K spends its three-frame budget on redundant views $f _ { 1 } { - } f _ { 3 }$ and covers only three of five required elements. Iterative reweighting instead selects $f _ { 1 } , f _ { 4 } ,$ and $f _ { 5 } ,$ , covering all required elements.

## Element-Aware Complementary Retrieval

Standard similarity-based Top-K retrieval (Hu et al. 2026; An et al. 2026; Wei et al. 2026) frequently selects redundant historical frames depicting identical salient elements, leaving other critical story elements uncovered while compounding historical noise. To circumvent this, we formalize reference selection as a complementary coverage problem, seeking a compact subset of keyframes that collectively maximizes target-element coverage while systematically suppressing contextual leakage.

Greedy Coverage Formulation. For a target shot $s _ { t } .$ , the semantic query vector Y projects the registry elements into three disjoint constraint sets: required $( R _ { t } \ = \ \{ e _ { i } \ \mid \ y _ { i , t } \ =$ REF}), excluded $( X _ { t } = \{ e _ { i } \ | \ \bar { y _ { i , t } } = \mathtt { E X C } \} )$ , and optional $( O _ { t } = \{ e _ { i } \ | \ y _ { i , t } = \mathrm { O P T } \} )$ ). Let ${ \mathcal { F } } = { \mathcal { L } }$ represent the candidate historical keyframes. Leveraging the frame annotations $\mathbf { A } _ { k }$ from the library, we denote the set of visible elements in frame f as $\mathcal { A } ( f ) \doteq \{ e _ { i } \in \mathcal { R } \mid a _ { f , i } = 1 \}$

Our framework greedily constructs a compact reference set S $( | S | \le K )$ . At each round, Algorithm 1 rescores every unselected candidate according to the dynamically updated set C of already covered required elements. The weights satisfy $w _ { \mathrm { u n c o v e r e d \ R E F } } > w _ { \mathrm { c o v e r e d \ R E F } } \ge 0 , w _ { \mathrm { O P T } } \ge 0 ,$ , and $w _ { \mathrm { E X C } } < 0 .$ . Each visible element contributes according to its state: uncovered required elements receive positive rewards, covered or optional elements receive a small or zero reward, and excluded elements incur a negative contribution.

Algorithm 1: Complementary reference retrieval   
Require: Candidate frames $\mathcal { F } ;$ annotated elements $\boldsymbol A ( f )$   
and fidelity labels $q _ { f , e } \in$ {full, partial, weak};   
required elements $R _ { t } ;$ excluded elements $X _ { t } ;$ optional   
elements $O _ { t } ;$ the maximum reference count K   
Ensure: Selected reference set S and covered required  
element set $C$   
1: $S \gets \emptyset , C \gets \emptyset$   
2: while $| S | < K$ and $\mathcal { F } \backslash S \neq \emptyset$ do   
3: for all $f \in \mathcal { F } \backslash \mathcal { S }$ S do   
4: score $( f ) \gets \mathrm { \bar { 0 } }$   
5: for all $e \in { \mathcal { A } } ( f )$ do   
6: if $e \in R _ { t } \setminus \dot { C }$ then   
7: score(f) ← score(f) + w<sub>uncovered REF</sub>   
8: else if $e \in R _ { t } \cap C$ then   
9: score $( f ) \gets$ score(f) + w<sub>covered REF</sub>   
10: else if $e \in X _ { t }$ then   
11: score(f) ← score $( f ) + w _ { \mathrm { E X C } }$   
12: else if $e \in O _ { t }$ then   
13: score(f) ← score(f) + w<sub>OPT</sub>   
14: end if   
15: end for   
16: end for   
17: $f ^ { \star } \gets \mathrm { a r g }$ max $\cdot f \in { \mathcal { F } } \backslash S$ score(f)   
18: $S \gets S \cup \{ f ^ { \star } \}$   
19: $C \gets C \cup \tilde { \{ e \in A ( f ^ { \star } ) \cap R _ { t } : q _ { f ^ { \star } , e } = \operatorname { f u l } 1 \} }$   
20: Record score, covered elements, optional elements,   
and excluded elements for $f ^ { \star }$   
21: end while   
22: return $S , C$

Fidelity Modulation and Interpretability. Element-level coverage in C is strictly determined by the VLM-annotated fidelity label $q _ { k , i } .$ . An element e is appended to $C$ only when $q _ { k , i } = \mathtt { f u l } 1$ , meaning that it is complete and clearly identifiable; for a character, this additionally requires a clear frontal face. Elements labeled partial or weak can still contribute to a frame’s retrieval score, but they are not considered fully covered. This allows subsequent rounds to retrieve alternative frames that depict the same required element more clearly.

Following visual subset selection, an LLM generates natural-language reference guidance for each $f _ { k } \in S .$ . Conditioned on the target shot prompt $p _ { t }$ , the shot-level element states $\mathbf { Y } _ { t } .$ , the frame’s element annotations $\mathbf { A } _ { k }$ , and its holistic description $\mathbf { D } _ { k } ,$ , this guidance directs the frozen video generator on how to interpret the frame for the current scene.

![](images/39c8ac01666b4e829a9225b9a5a30deacb9f96d739316c70996728a49bc83abf.jpg)  
Figure 3: Efect of grounded prompting. Without grounding, obsolete scene elements leak into the target shot; ours preserves the character while suppressing conflicting context.

By combining objective element-level instructions, holistic frame descriptions, and contextual guidance, the final prompt ensures fine-grained cross-shot alignment without retraining.

## Unified Prompt Assembly and Video Generation

The selected historical evidence, structured element states, and textual instructions are compiled into a unified multimodal prompt to provide explicit, grounded conditioning for the frozen video generator.

Structured Prompt Composition. To prevent ambiguous interpretation of the attached references, our framework enforces a structured prompt layout consisting of five functional blocks: (i) current generation task, (ii) prefix story context, (iii) shot-level element states $( \mathbf { Y } _ { t } ) ,$ (iv) per-reference grounding instructions, and (v) global generation constraints.

This architecture converts raw reference pixels into explicit, verifiable generation boundaries. Because a historical keyframe can simultaneously depict a required element and an obsolete asset, the prompt explicitly delineates both positive reference guidance and negative suppression boundaries. Element-level instructions specify which characters, objects, and scenes should be preserved or suppressed, while holistic descriptions $\mathbf { D } _ { k }$ and reference guidance provide sequencelevel semantic coordination.

Figure 3 illustrates this disambiguation. The element plan marks the reusable character as REF and the obsolete scenery as EXC; the prompt agent translates these states into perreference instructions that preserve character appearance without copying unrelated context.

Video Generation and Sequential Assembly. The compiled multimodal prompt is sent to the frozen short-video generator. Retrieved references provide long-range visual consistency for both shot types, while the cut indicator $c _ { t }$ determines how local continuity and assembly are handled:

• Cut Shots $( c _ { t } = \mathbf { t r u e } ) \colon$ The generated clip begins a new scene and is directly appended to the preceding sequence.

• Continuous Shots $( c _ { t } ~ = ~ \mathbf { f a l s e } ) \colon$ The final segment of $s _ { t - 1 }$ serves as a video prefix to continue the preceding motion and camera trajectory. During assembly, two RIFE-interpolated frames (Huang et al. 2022) replace the boundary frames of $s _ { t - 1 }$ and $s _ { t }$ to reduce boundary flicker.

## System Properties: Interpretability and Editability

Unlike opaque, end-to-end multi-shot video generation pipelines, our element-aware framework exposes explicit structural reasoning artifacts at every stage, including shotlevel element states $( \mathbf { Y } _ { t } )$ , retrieved complementary reference subsets (S), the covered-element set (C), grounded multimodal prompts, and keyframe annotations $( \mathbf { A } _ { k } )$ . This granular transparency directly yields three critical architectural advantages:

• Attributable Diagnosis: Generative failures become systematically traceable and diagnosable. For instance, if an obsolete element or a mismatched character identity leaks into a target shot, the user or agent can isolate whether the anomaly stems from agentic element tracking, complementary reference retrieval, or unified prompt assembly.

• Auto-Reflectivity: The explicit symbolic boundaries enable autonomous self-reflection and pre-generation verification. The orchestration agent can inspect its referencecoverage matrix, evaluate constraint adherence before invoking the frozen generator, and iteratively refine instructions to maximize target-element fidelity.

• Human-in-the-Loop Editability: By modularizing the narrative pipeline into human-interpretable intermediate states, our framework supports manual intervention at runtime. A human director can edit the active element registry, modulate constraint vectors, or replace retrieved references to guide story progression through a controllable and iteratively refinable generation loop.

## Experiments

## Benchmark and Metrics

We adopt the Cross-Shot Consistency evaluation from EntityBench (He et al. 2026), which measures whether recurring characters, objects, and scenes remain consistent across shots. It combines two complementary protocols. First, DINOv2 embedding similarity compares character and object appearances with their per-entity centroids, while a transition-boundary metric measures continuity across scene-internal cuts. Second, LLM pairwise judging compares each non-anchor appearance with a centroidrepresentative anchor using type-specific criteria, reporting both accuracy and fine-grained similarity; locations are evaluated from full frames with camera-invariant instructions to accommodate viewpoint changes and partial views. The 21 metrics are grouped into DINOv2 Similarity (3), LLM Characters (6), LLM Objects (6), and LLM Scenes (6), with Overall denoting their mean. EntityBench additionally reports intra-shot metrics including quality and prompt following, which we omit as they primarily evaluate the base video generator rather than cross-shot reference construction.

Experimental Settings. Dataset construction and generation. EntityBench contains 140 long-range multi-shot episodes. We use an LLM to filter scripts with obvious copyright or sensitive-content risks, yielding 105 eligible episodes, and randomly sample 20 Easy, 10 Mid, and 5 Hard episodes as our test set. Every shot is generated at 480P for 5 seconds. To comply with the moderation constraints of Seedance 2.0, all methods prepend the same instruction requiring non-photorealistic 2D animated human faces and original, non-copyrighted character designs. No predefined character sheets or external reference images are supplied; all visual evidence is generated from the script during inference.

Method-specific implementation. All explicit workflows use their default reference-construction procedures with Seedance 2.0 as the common video backend. ViMax and VideoMemory use Seedream 5.0 to synthesize their dedicated entity references. Training-based baselines use the released pretrained weights and repository-default inference settings on an NVIDIA RTX PRO 6000 GPU. For a fair StoryMem reproduction, we preserve its original keyframe maintenance policy but adapt its 3 Sink + 7 Recent budget to 3 Sink + 6 Recent because Seedance 2.0 accepts at most nine images. Our method retrieves at most five historical frames per shot. Its element weights are 3.0, 2.0, and 1.5 for characters, scenes, and objects; uncovered and covered required elements receive weights 1.0 and 0.2, optional elements 0.1, and excluded elements −0.1. The corresponding referencequality weights for full, partial, and weak evidence are 1.0, 0.2, and 0.1.

Keyframe library and evaluation. StoryMem stores flat keyframe files, retains at most three diverse keyframes per shot under HPSv3 ≥ 3.0 and CLIP similarity $< 0 . 9$ , deduplicates against the complete history, and restricts inference to its Sink–Recent window. In contrast, our library retains up to six diverse keyframes per shot under $\mathrm { H P S v 3 } \geq 2 . 5$ and CLIP similarity < 0.95, only filters near-duplicate adjacent additions, preserves all historical candidates, and attaches VLM element-presence boxes, reference-quality labels, and holistic descriptions. We evaluate all methods using the oficial EntityBench implementation, replacing its default Gemini judge with Doubao Seed2.1 Turbo uniformly for every method.

Compared Methods. We compare against seven representative baselines from recent advances in multi-shot video generation, organized into two groups. The first comprises training-based methods with dedicated generation models: CineTrans (Wu et al. 2025), HoloCine (Meng et al. 2026), LongLive2.0 (Chen et al. 2026), and Shot-Stream (Luo et al. 2026). The second evaluates explicit reference-construction workflows using the same commercial video API. VideoMemory (Zhou et al. 2026) and Vi-Max (Huang et al. 2026) first synthesize dedicated reference images and then condition video generation on them, whereas StoryMem (Zhang et al. 2025) and our approach use keyframes mined from generated history. Although the oficial StoryMem implementation adapts Wan2.2 with LoRA for multi-reference video generation, its memory construction and keyframe-selection procedures are explicit and deterministic. For a fair comparison, our StoryMem reproduction preserves these procedures while replacing the video generator with the same Seedance2.0 backend.

<table><tr><td>Method</td><td>Overall</td><td>DINO</td><td>Char.</td><td>Obj.</td><td>Scene</td></tr><tr><td colspan="6">Training-based methods</td></tr><tr><td>CineTrans</td><td>0.2964</td><td>0.5659</td><td>0.2080</td><td>0.2235</td><td>0.3228</td></tr><tr><td>HoloCine</td><td>0.4471</td><td>0.6268</td><td>0.3125</td><td>0.5219</td><td>0.4171</td></tr><tr><td>LongLive2.0</td><td>0.6512</td><td>0.8690</td><td>0.5639</td><td>0.6464</td><td>0.6344</td></tr><tr><td>ShotStream</td><td>0.6703</td><td>0.8280</td><td>0.5789</td><td>0.5767</td><td>0.7766</td></tr><tr><td colspan="6">Explicit workflows (all reproduced with Seedance2.0 backend)</td></tr><tr><td>VideoMemory</td><td></td><td>0.71990.66860.65930.7128</td><td></td><td></td><td>0.8134</td></tr><tr><td>ViMax</td><td>0.7427</td><td>0.7244</td><td>0.7864</td><td>0.7699</td><td>0.6811</td></tr><tr><td>StoryMem</td><td>0.8425</td><td>0.8078</td><td>0.8457</td><td>0.8531</td><td>0.8461</td></tr><tr><td>Ours</td><td>0.8670</td><td>0.8249</td><td>0.8930</td><td>0.8780</td><td>0.8510</td></tr></table>

Table 1: Cross-Shot Consistency on the same 35-episode EntityBench subset. Overall averages all 21 metrics. Best and second-best results are shown in bold and underlined, respectively.

## Results

As shown in Table 1, our method achieves the best Overall score of 0.8670 and leads all three LLM-judged categories, with scores of 0.8930 for character consistency, 0.8780 for object consistency, and 0.8510 for scene consistency. These results demonstrate its efectiveness in preserving diverse recurring visual elements across shots.

The results also reveal a metric-dependent pattern. Training-based methods such as LongLive2.0 and Shot-Stream are particularly competitive in DINOv2 similarity. Their model-internal propagation of history through crossshot attention, caches, or related memory mechanisms may better preserve embedding-level appearance, and their training objectives may reinforce this efect. However, their lower LLM scores indicate that high feature similarity does not necessarily imply preservation of the correct character attributes, object states, or scene semantics. In contrast, the explicit workflows generally perform better under these fine-grained multimodaljudgments, suggesting an advantage from explicitly organizing and grounding historical evidence.

Among methods using the same Seedance2.0 backend, the historical-keyframe approaches, StoryMem and ours, outperform the generated-reference approaches, VideoMemory and ViMax, across all four categories. Dedicated entity images provide clean canonical references but may difer from appearances produced by the video generator and omit useful co-occurrence or scene context. Historical keyframes retain visual evidence from earlier shots, while our annotation, complementary retrieval, and grounded prompting determine what to reuse and suppress.

## Ablation Study

All variants use the same 35 episodes, generation backend, VLM judge, and animation constraints; variants using references also share the same reference budget of 5 images.

<table><tr><td>Method</td><td>Overall</td><td>DINO</td><td>Char.</td><td>Obj.</td><td>Scene</td></tr><tr><td>Full</td><td>0.8670</td><td>0.8249</td><td>0.8930</td><td>0.8780</td><td>0.8510</td></tr><tr><td>w/o Cover</td><td>0.8482</td><td>0.7855</td><td>0.8658</td><td>0.8623</td><td>0.8479</td></tr><tr><td>w/o Prompting</td><td>0.8491</td><td>0.8134</td><td>0.8513</td><td>0.8741</td><td>0.8396</td></tr><tr><td>CLIP Retrieval</td><td>0.8456</td><td>0.8233</td><td>0.8407</td><td>0.8584</td><td>0.8488</td></tr><tr><td>No Reference</td><td>0.5456</td><td>0.7241</td><td>0.4058</td><td>0.5788</td><td>0.5628</td></tr></table>

Table 2: Cross-Shot Consistency of ablations and retrieval baselines on the same 35-episode subset.

Full denotes the complete pipeline. w/o Cover retains visual element planning and grounded prompting but replaces iterative complementary selection with a single-pass Top-K selection based on the same static element-aware frame scores. w/o Prompting retains visual element planning and greedy coverage but conditions the generator only on the original shot prompt and selected images, omitting structured element plans and reference-specific guidance. CLIP Retrieval keeps the keyframe library and retrieves historical frames using CLIP text–image similarity between the current shot prompt and each candidate frame, without visual element planning, complementary coverage, or reference grounding. No Reference uses only the shot prompt and removes historical visual references entirely.

Table 2 demonstrates that Full yields the highest Overall score and ranks first across all four evaluation criteria. w/o Cover lowers Overall by 0.0188 and degrades every reported metric, confirming that independently high-scoring frames do not necessarily form a complementary reference set. w/o Prompting lowers Overall by 0.0179 and character consistency by 0.0417, indicating that selected images remain ambiguous unless the generator is told which evidence to preserve and which historical content to suppress. CLIP Retrieval trails Full by 0.0214 in Overall and 0.0523 in character consistency. Because it jointly removes element planning, element-aware coverage, and reference grounding, this comparison measures the benefit of the integrated elementaware workflow rather than any single component. Finally, No Reference reduces Overall to 0.5456, demonstrating that text-only prompting is insuficient to preserve recurring entities across independently generated shots.

## Qualitative Analysis

Figure 2 shows that greedy coverage avoids redundant Top-K frames by favoring uncovered elements, yielding a complementary set that covers all required content. Figure 3 demonstrates how grounding disambiguates a selected frame: without it, the volcanoes and the rose from the Little Prince’s former planet leak into his visit to a merchant on a new planet. Our method preserves the Little Prince’s appearance while explicitly excluding these obsolete scene elements.

Figure 4 further compares explicit reference-management workflows under the same Seedance 2.0 backend on the challenging Hard\_5 episode. Fiona changes from her green dress into Leo’s smiley-face hoodie in Scene 10, Shot 3 and should retain this outfit thereafter. VideoMemory and ViMax exhibit outfit inconsistency in the later Fiona shot; ViMax also exhibits background drift in the cluttered workshop. StoryMem produces character confusion by rendering Fiona as Leo, followed by character drift for Silas. In contrast, our framework preserves the intended character, outfit, and scene states across the long-range sequence.

Figure 5 visualizes the component efects on Scene 3 of Easy\_2. Removing complementary coverage selection yields an outfit inconsistency for Priya, while removing structured prompting produces character confusion. Replacing elementaware retrieval with generic CLIP retrieval causes clothing drift and removes Priya’s head covering. The complete pipeline maintains both the recurring characters and the Old Quarter scene across the displayed shots.

## Conclusion

We presented Complementary Retrieval-Augmented Prompting, a training-free framework for consistent longform video generation with frozen short-video generators. Instead of compressing visual history into implicit memory or relying on manually curated references, our framework maintains a text-grounded visual element registry together with a VLM-annotated keyframe library, turning generated history into structured and queryable evidence. Coverageaware retrieval selects a compact set of complementary frames, while grounded multimodal prompting specifies what each frame should contribute and which stale content should be suppressed. EntityBench experiments demonstrate superior cross-shot consistency, while ablations validate complementary retrieval and reference grounding. Beyond accuracy, the explicit intermediate representations make reference decisions interpretable and editable, supporting interactive creation without a complete script. Separating evidence maintenance, retrieval, and generation also facilitates backend portability and allows failures to be traced to individual pipeline stages rather than hidden within model states. Current limitations include errors in element planning and visual annotation, imperfect prompt adherence by the underlying generator, and constraints imposed by commercial APIs. Future work will improve agentic verification, scale retrieval to longer narratives, and strengthen transfer across video-generation backends and interactive production settings.

![](images/61892b8c026a39919a3663b39eabedb62ab98fbdd984202ed4962a11cf616aff.jpg)  
Character and background states remain consistent across shots

Figure 4: Qualitative comparison of explicit workflows on EntityBench Hard\_5; all methods use Seedance 2.0. Colored text and boxes identify the relevant characters and scene. After Fiona changes into Leo’s smiley-face hoodie in Scene 10, Shot 3, VideoMemory and ViMax exhibit outfit inconsistency in Scene 26, Shot 3; ViMax also exhibits background drift. StoryMem exhibits character confusion for Fiona and character drift for Silas. Our method preserves the marked character, outfit, and background states.  
![](images/ffa25ec367c652487830e62ff658f2d63f0d7b809d7d0141334c76c8a7687846.jpg)  
Characters and backqrounds remain consistent across shots

Figure 5: Qualitative ablation on Scene 3 of EntityBench Easy\_2. Without complementary coverage, Priya has an outfit inconsistency in Shot 5. Without structured prompting, Priya is incorrectly generated in Shot 5. CLIP retrieval causes clothing drift and loses her head covering. The complete pipeline maintains the intended character and Old Quarter scene states.

## References

Alibaba Cloud. 2025. Alibaba Unveils Wan2.6 Series Enabling Everyone to Star in Videos. Alibaba Cloud Community. Accessed: 2026-07-27.

An, Z.; Jia, M.; Qiu, H.; Zhou, Z.; Huang, X.; Liu, Z.; Ren, W.; Kahatapitiya, K.; Liu, D.; He, S.; Zhang, C.; Xiang, T.; Yang, F.; Belongie, S.; and Xie, T. 2026. OneStory: Coherent Multi-Shot Video Generation with Adaptive Memory. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16173–16184.

Bao, F.; Xiang, C.; Yue, G.; He, G.; Zhu, H.; Zheng, K.; Zhao, M.; Liu, S.; Wang, Y.; and Zhu, J. 2024. Vidu: A Highly Consistent, Dynamic and Skilled Text-to-Video Generator with Difusion Models. arXiv:2405.04233.

Chen, Y.; Wang, L.; Huang, W.; Yang, S.; Zhang, B.; Xiao, Y.; Chu, R.; Mao, W.; Hu, Q.; Liu, S.; Zhao, Y.; Mao, H.; Chen, Y.-C.; Xie, E.; Qi, X.; and Han, S. 2026. LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation. arXiv:2605.18739.

Gao, Y.; Guo, H.; Hoang, T.; Huang, W.; Jiang, L.; Kong, F.;Li, H.; Li, J.; Li, L.; Li, X.; Li, X.; Li, Y.; Lin, S.; Lin, Z.;Liu, J.; Liu, S.; Nie, X.; Qing, Z.; Ren, Y.; Sun, L.; Tian, Z.;Wang, R.; Wang, S.; Wei, G.; Wu, G.; Wu, J.; Xia, R.; Xiao,F.; Xiao, X.; Yan, J.; Yang, C.; Yang, J.; Yang, R.; Yang,T.; Yang, Y.; Ye, Z.; Zeng, X.; Zeng, Y.; Zhang, H.; Zhao,Y.; Zheng, X.; Zhu, P.; Zou, J.; and Zuo, F. 2025. Seedance

1.0: Exploring the Boundaries of Video Generation Models. arXiv:2506.09113.

Google DeepMind. 2025. Veo 3.1. Google DeepMind. Accessed: 2026-07-27.

He, R.; Wei, M.; Yang, Z.; and Ordonez, V. 2026. Entity-Bench: Towards Entity-Consistent Long-Range Multi-Shot Video Generation. arXiv:2605.15199.

Hu, Q.; Yang, S.; Huang, W.; Han, S.; and Chen, Y. 2026. LongLive-RAG: A General Retrieval-Augmented Framework for Long Video Generation. arXiv:2606.02553.

Huang, H.; Ma, G.; Duan, N.; Chen, X.; Wan, C.; Ming, R.;Wang, T.; Wang, B.; Lu, Z.; Li, A.; Zeng, X.; Zhang, X.; Yu,G.; Yin, Y.; Wu, Q.; Sun, W.; An, K.; Han, X.; Sun, D.; Ji, W.;Huang, B.; Li, B.; Wu, C.; Huang, G.; Xiong, H.; He, J.; Wu,J.; Yuan, J.; Wu, J.; Liu, J.; Guo, J.; Tan, K.; Chen, L.; Chen,Q.; Sun, R.; Yuan, S.; Yin, S.; Liu, S.; Chen, W.; Dai, Y.;Luo, Y.; Ge, Z.; Guan, Z.; Song, X.; Zhou, Y.; Jiao, B.; Chen,J.; Li, J.; Zhou, S.; Zhang, X.; Xiu, Y.; Zhu, Y.; Shum, H.-Y.; and Jiang, D. 2025. Step-Video-TI2V Technical Report:A State-of-the-Art Text-Driven Image-to-Video GenerationModel. arXiv:2503.11251.

Huang, L.; He, S.; Zhou, H.; Nie, L.; Xia, L.; and Huang, C. 2026. ViMax: Agentic Video Generation. arXiv:2606.07649.

Huang, Z.; Zhang, T.; Heng, W.; Shi, B.; and Zhou, S. 2022. Real-Time Intermediate Flow Estimation for Video Frame Interpolation. In Computer Vision – ECCV 2022, 624–642. Springer.

Kling Team. 2025. Kling-Omni Technical Report. arXiv:2512.16776.

Lai, Y.; Shao, T.; Zhou, K.; Dou, W.; Zhu, S.; and Wang, J. 2026. GroundShot: Visually Consistent Multi-Shot Long Video Generation via Entity-Grounded Shot Scheduling. arXiv:2606.20799.

Lin, H.; Zala, A.; Cho, J.; and Bansal, M. 2024. VideoDirectorGPT: Consistent Multi-Scene Video Generation via LLM-Guided Planning. In Proceedings of the First Conference on Language Modeling (COLM).

Liu, A.; Xing, J.; Mao, C.; Li, Y.; Zhang, Z.; He, Y.; Wang, W.; Wang, Z.; Liu, Y.; Hafari, G.; and Zhuang, B. 2026. ReCA: Multi-Shot Long Video Extrapolation via Recursive Context Allocation. arXiv:2605.26525.

Luo, Y.; Shi, X.; Zhuang, J.; Chen, Y.; Liu, Q.; Wang, X.; Wan, P.; and Xue, T. 2026. ShotStream: Streaming Multi-Shot Video Generation for Interactive Storytelling. arXiv:2603.25746.

Ma, Y.; Wu, X.; Sun, K.; and Li, H. 2025. HPSv3: Towards Wide-Spectrum Human Preference Score. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 15086–15095.

Meng, Y.; Ouyang, H.; Yu, Y.; Wang, Q.; Wang, W.; Cheng, K. L.; Wang, H.; Ma, S.; Li, Y.; Chen, C.; Zeng, Y.; Zhu, X.; Shen, Y.; and Qu, H. 2026. HoloCine: Holistic Generation of Cinematic Multi-Shot Long Video Narratives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 461–471.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.;

Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, 8748–8763.

Wei, X.; Ji, L.; Wang, G.; Liu, X.; Zhang, Z.; Wang, S.; Sun, Y.; and Hong, Q. 2026. Memento: Reconstruct to Remember for Consistent Long Video Generation. arXiv:2606.14667.

Wu, W.; Zhu, Z.; and Shou, M. Z. 2025. Automated Movie Generation via Multi-Agent CoT Planning. arXiv:2503.07314.

Wu, X.; Gao, B.; Qiao, Y.; Wang, Y.; and Chen, X. 2025. CineTrans: Learning to Generate Videos with Cinematic Transitions via Masked Difusion Models. arXiv:2508.11484.

Yin, X.; Peng, X.; Li, X.; Xiong, Z.; and Lu, Y. 2026. Closed-Loop Triplet Synergistic Generation for Long-Form Video. arXiv:2606.16184.

Zhang, K.; Jiang, L.; Wang, A.; Fang, J. Z.; Zhi, T.; Yan, Q.; Kang, H.; Lu, X.; and Pan, X. 2025. StoryMem: Multi-shot Long Video Storytelling with Memory. arXiv:2512.19539.

Zheng, M.; Xu, Y.; Huang, H.; Ma, X.; Liu, Y.; Shu, W.; Pang, Y.; Tang, F.; Chen, Q.; Yang, H.; and Lim, S.-N. 2024. VideoGen-of-Thought: A Collaborative Framework for Multi-Shot Video Generation. arXiv:2412.02259.

Zhou, J.; Du, Y.; Xu, X.; Wang, L.; Zhuang, Z.; Zhang, Y.; Li, S.; Hu, X.; Su, B.; and Chen, Y.-C. 2026. VideoMemory: Toward Consistent Video Generation via Memory Integration. arXiv:2601.03655.