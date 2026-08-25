# DUAL-GRAINED AGENT MEMORY AND SHAPLEY CONTEXT ATTRIBUTION FOR MULTIMODAL AGENTIC LEARNER

Jieke Wang UC Merced jwang450@ucmerced.edu

Yibo Yang Shanghai Jiao Tong University yibo.yang93@gmail.com

Tiancheng Shen UC Merced tianchengshen@ucmerced.edu

Ming-Hsuan Yang UC Merced myang37@ucmerced.edu

## ABSTRACT

Frontier multimodal large language models (MLLMs) deliver impressive perception yet still falter on scientific and mathematical reasoning. Parameter-level adaptation is unavailable for closed-weight or on-device backbones, and stateless prompting forfeits any compounding benefit from problems already solved. We propose DG-Mem, a dual-grained agentic memory framework that augments a frozen MLLM with a non-parametric, externally stored memory built once from training-time rollouts and consulted read-only at test time. Motivated by the Complementary Learning Systems (CLS) account of human memory, DG-Mem factors its store into an instance-grounded exemplar memory and a category-level schema memory of IF-THEN rules, with a transient reflection store mediating their construction so that schemas are synthesized only from abstract reflections, never from exemplar text. Two design choices distinguish DG-Mem: an online concept categorizer that grows the category space incrementally during training rather than committing to a predefined taxonomy, and a Shapley context attribution procedure that decomposes correctness across the entire retrieved rule set and yields a per-rule utility that re-weights retrieval at test time. The pipeline introduces no gradient updates and is deployable on closed-weight or on-device backbones. Across MathVista, MMMU, and MMMU-Pro on four open-weight and proprietary backbones (Qwen3.5-27B, Qwen3.5-122B-A10B, GPT-5-Nano, Gemini-3-Flash), DG-Mem improves consistently over no-memory and competitive memory baselines.

## 1 Introduction

Frontier multimodal large language models (MLLMs) now answer a wide range of vision-language questions [16, 7, 32], yet their failures on scientific and mathematical reasoning benchmarks remain [35, 13, 36]. Closing this gap via parameter-level adaptation, such as instruction tuning or RLHF, requires gradient access and full supervision. This approach is brittle when the backbone model changes. Furthermore, it is often impossible for proprietary or on-device models. Prompt-level adaptation, in contrast, is cheap and modular but stateless: the model receives no compounding benefit from problems it has already seen. Agentic memory frameworks bridge this divide by maintaining an external, non-parametric store that accumulates across problems and conditions a frozen backbone at test time.

Recent agentic memory systems pursue exactly this regime, but most settle for one of two extremes. Trajectory and case banks [47, 20, 28, 5, 46] preserve raw episodes and route by similarity or learned values; the resulting context is rich but instance-bound, and aggregation over many cases tends to amplify rather than abstract their idiosyncrasies. Single-grain prescriptive playbooks [29, 44, 17, 40, 3] swing to the other extreme, distilling experience into a flat list of rules; the resulting context is compact but loses the concrete grounding that an unfamiliar new problem often needs. Systems that do combine grains either pair task- and step-level skills [26] or pair two error-modality streams of the same abstraction level [3], rather than separating instance grounding from cross-case abstraction. Two design questions are left almost entirely open by these systems: (i) how memories should be categorized when the underlying concept space is not known a priori and a fixed taxonomy is either too coarse (e.g. a single “geometry” bucket) or too fine (e.g. a unique bucket per concept signature)? and (ii) how the credit for a correct answer should be distributed across multiple co-retrieved memories so that retrieval ranking can sharpen over time? Existing per-memory utility signals [41, 3] attribute success or failure to a single retrieved item, conflating its contribution with the rest of the context.

A complementary perspective comes from cognitive science. The Complementary Learning Systems (CLS) account of human memory [14, 11] holds that flexible behavior arises from two interacting stores: a fast, instance-specific system that supports exemplar-based generalization [15], and a slow, structured system organized around abstract schemas. Critically, the two are not built in the same way: episodic traces are progressively integrated offline into reusable schemas, a consolidation step that schemas themselves are known to accelerate [25]. Equally relevant is what gets re-activated. Offline replay is preferentially biased toward rewarded experience [22, 21], and the rational analysis of memory holds that retrieval accessibility composes cue-driven match strength with a base-level term that tracks how useful a memory has been historically [1]. Inspired by these claims, we design our agentic memory framework with the granularity separation, schema-mediated consolidation, and value-weighted retrieval.

We propose DG-Mem, a dual-grained agentic memory framework. A frozen MLLM backbone is paired with a non-parametric store factored into an instance-grounded exemplar memory and a category-level schema memory of IF-THEN rules; a transient reflection store acts as the episodic-trace intermediate from which schemas are synthesized, with exemplar text deliberately excluded from schema synthesis to enforce the consolidation constraint. To resolve the taxonomy question, an online concept categorizer grows the category space incrementally during training rather than committing to a predefined set, allowing to track the empirical concept distribution of the corpus. To resolve the credit-assignment question, a second offline pass treats the rules retrieved for each problem as players in a cooperative game and uses Shapley context attribution over rule-subset rollouts to compute a per-rule utility; at test time, retrieval composes cue similarity with this utility, instantiating the rational-analysis prescription of similarity weighted by track record. The entire pipeline runs on a frozen backbone with no gradient updates, making it deployable on closed-weight or on-device models.

Our contributions are summarized as follows:

• We propose DG-Mem, an agentic memory framework that enables multimodal agents to learn and evolve. An exemplar memory and a schema memory of IF-THEN rules, with schemas synthesized exclusively from a transient reflection store so that exemplar text never contaminates the schema, operationalizing CLS-style consolidation from abstract traces.

• We propose Shapley context attribution to handle the credit assignment problem of learned schemas. Rules co-retrieved for a problem are scored as players in a cooperative game. The resulting per-rule utility instantiates the rational-analysis view of accessibility as similarity weighted by historical usefulness.

• Across four backbones (Qwen3.5-27B / 122B-A10B [24], GPT-5-Nano [16], Gemini-3-Flash [7]) and three multimodal benchmarks (MathVista [13], MMMU [35], MMMU-Pro [36]), DG-Mem consistently beats No-memory baseline, and both offline and online ViLoMem.

## 2 Related Works

Recent works have treated the prompt context as a first-class object to be engineered, accumulated, and evolved [2, 10, 9].   
Existing methods are usually along two axes: what is stored (granularity) and how it is constructed (regime).

Granularity. Trajectory and case stores preserve episodic experience and route through similarities or learned values. Memento [47] formulates case selection as a soft-Q policy over an episodic case bank. MemRL [41] attaches a TDlearned utility to each retrieved trace. Buffers-based methods [20, 28, 23, 33] and classical text-memory systems [5, 46] keep verbatim reasoning episodes or maintain unstructured notes. Single-grain prescriptive playbooks distill experience into a flat list of generic strategies [29, 44, 17, 45, 40]. ReasoningBank [17] and SkillWeaver [45] extract heterogeneous episodes into a single bullet list. ACE [40] maintains an itemized playbook updated by a Generator-Reflector-Curator with grow-and-refine de-duplication. Dual-grain or structured stores factor memory into multiple types: ViLoMem [3] runs visual distractions and logical hallucinations but both at schema-level guideline granularity; D2Skill [26] stores task-level and step-level skill banks; structured stores such as A-Mem [30], Zep [18], and G-Memory [38] organize memory as temporal or knowledge graphs but without granularity separation. Unlike these works, we factor memory into instance-grounded exemplars and category-level IF-THEN schemas synthesized exclusively from a transient reflection store, operationalizing schema-mediated consolidation [25] so that exemplar text never contaminates schema synthesis.

Table 1: Comparison of agentic memory methods across five design dimensions: whether the method (i) handles multimodal inputs (vision+text), (ii) maintains instance-grounded exemplars, (iii) maintains abstract schema rules, (iv) performs per-rule credit assignment beyond similarity-only retrieval, and (v) is gradient-free (no model parameter updates). $\checkmark / \times$ indicates whether the method satisfies a given dimension. See the surrounding text for column definitions.
<table><tr><td>Method</td><td>Multimodal</td><td>Exemplar</td><td>Schema</td><td>Credit Assn.</td><td>Grad-Free</td></tr><tr><td>Mem0 [5]</td><td>X</td><td>√</td><td>X</td><td>×</td><td>√</td></tr><tr><td>Memento [47]</td><td>X</td><td>√</td><td>X</td><td>V</td><td>√</td></tr><tr><td>A-Mem [30]</td><td>×</td><td>√</td><td>×</td><td>×</td><td>√</td></tr><tr><td>ExpeL [44]</td><td>X</td><td>X</td><td>√</td><td>X</td><td>√</td></tr><tr><td>AWM [29]</td><td>X</td><td>×</td><td>√</td><td>X</td><td>√</td></tr><tr><td>ReasoningBank [17]</td><td>×</td><td>×</td><td>√</td><td>×</td><td>V</td></tr><tr><td>ACE [40]</td><td>×</td><td>×</td><td>√</td><td>X</td><td>√</td></tr><tr><td>ViLoMem [3]</td><td>√</td><td>×</td><td>V</td><td>X</td><td>√</td></tr><tr><td>TFGRPO [4]</td><td>×</td><td>×</td><td>√</td><td>√</td><td>√</td></tr><tr><td>D2Skill [26]</td><td>X</td><td>√</td><td>√</td><td>×</td><td>√</td></tr><tr><td>Memory-R1 [31]</td><td>×</td><td>√</td><td>×</td><td>√</td><td>×</td></tr><tr><td>MemGen [39]</td><td>X</td><td>×</td><td>X</td><td>√</td><td>X</td></tr><tr><td>DG-Mem (Ours)</td><td>5</td><td>L</td><td></td><td>L</td><td>7</td></tr></table>

Construction Regime. Memory-R1 [31], Mem-α [27], Memory-as-Action [43], and MEM1 [48] learn “add/update/delete” policies under outcome rewards. MemBuilder [19] trains a lightweight language model as the memory manager with attributed dense rewards; MemGen [39] fuses RL-generated latent memory tokens into the decoder; Mem T [37] and LatentMem [6] apply GRPO over Memory-Operation Trees and learnable composers respectively. These methods require gradients, often opaque latent representations, and a fixed cognitive taxonomy. A complementary line distills memory offline from rollouts of a frozen backbone [44, 29, 17, 3, 40]; we belong here. First, our online concept categorizer grows the category space incrementally during the training pass instead of committing to a predefined taxonomy, comparable in spirit to STITCH’s contextual-intent indexing [34] but operating over reasoning concepts rather than dialogue intents. Second, we add an inter-test-time attribution pass (Stage 2) that approximates RL-style credit assignment without ever updating model parameters, slotting between similarity-only distillation (AWM [29], ReasoningBank [17]) and full RL of memory managers (Memory-R1 [31], MemGen [39]).

## 3 Methods

![](images/d812c5db3ca76246b5c7fcb3f5c6cbbfe8902b4ea4405b162c8c0ce1dc5656b1.jpg)  
Figure 1: Two-stage offline pipeline for memory training on a frozen MLLM backbone. Left (Stage 1 – Memory Construction): for each training problem $q ,$ MLLM produces N rollouts that an online classifier and three parallel reflectors turn into an exemplar memory $\mathcal { M } _ { E }$ and a category-level schema memory ${ \mathcal { M } } _ { S } ,$ , with a transient reflection store $\mathcal { M } _ { R }$ mediating the latter so that schema synthesis never reads exemplar text. Right (Stage 2 – Shapley Context Attribution): the same frozen MLLM is reused as an attribution device; for each problem we draw B subsets of the active category’s rules, run a rollout per subset, and convert the binary rewards into per-rule Shapley contributions that are accumulated into a utility column on $\mathcal { M } _ { S }$ and used to re-weight retrieval at test time. See §3.1 and §3.3 for full details.

Cognitive Foundations. The introduction motivates DG-Mem from CLS [14, 11]; here we summarize the operational consequences. We instantiate the fast, instance-specific store as an exemplar memory $\mathcal { M } _ { E } \left[ 1 5 \right]$ and the slow, structured store as a schema memory $\mathcal { M } _ { S }$ , with an intermediate reflection store $\mathcal { M } _ { R }$ acting as the unconsolidated episodic trace from which schemas are later synthesized [25]. At inference, $\mathcal { M } _ { E }$ and $\mathcal { M } _ { S }$ are accessed through two complementary mechanisms, priming (top-down context before reasoning) and verification (post-hoc audit). Following the rational analysis of memory [1] and evidence that offline replay is biased toward rewarded experience [22, 21], retrieval composes embedding similarity with a per-rule utility learned offline, so that two memories with identical cue match are tied by their historical track record. We treat these correspondences as design intuitions rather than cognitive claims.

Agentic Architecture. We augment a frozen MLLM backbone MLLM with a non-parametric, externally stored memory M that is constructed once from a training corpus and consulted read-only at test time. No gradients propagate into MLLM or additional layers. For each test problem $q ,$ the agent executes a short, memory-aware decision pipeline: it analyzes q to produce a concept summary $\dot { \mathcal { K } } ( q )$ , queries $\mathcal { M }$ with $\mathcal { K } ( q )$ and the multimodal representation $( I _ { q } , t _ { q } )$ to obtain a relevant subset $\mathcal { M } _ { q } \subseteq \mathcal { M }$ , and conditions the backbone on the composed context to produce an answer $\hat { a } _ { q } = \mathrm { M L L M } ( q , { \mathcal { M } } _ { q } )$ . The remainder of this section specifies how M is constructed (§3.1), how it is accessed (§3.2), and how utility estimates refine what is retained (§3.3).

## 3.1 Memory Construction

Memory is built offline from rollouts of MLLM on a training corpus. For each problem $q$ with gold answer $a _ { q } ^ { \star } ,$ , we draw $N$ temperature-sampled solution trajectories $\{ \mathrm { M L L M } _ { i } ( \boldsymbol { q } ) \} _ { i = 1 } ^ { N }$ and label each CORRECT or INCORRECT via the verification cascade $\mathrm { V e r i f y } ( \mathrm { M L L M \it _ i ( q ) , \dot { a _ { q } ^ { \star } } ) \in \{ 0 , \dot { 1 } \} }$ (native benchmark matcher, symbolic parser, then LLM judge). The labeled rollout group is the only training signal; we do not require reasoning annotations beyond $a _ { q } ^ { \star }$

Given the labeled rollout group, three summarizations are produced in parallel and written to distinct stores:

$$
\rho ( q ) = \mathrm { R e f l } _ { R } \left( q , a _ { q } ^ { \star } , \{ \mathrm { M L L M } _ { i } ( q ) \} _ { i = 1 } ^ { N } \right)
$$

$$
( { \mathrm { I F - T H E N } } { \mathrm { r u l e s } } ) ,\tag{1}
$$

$$
e ( q ) = \mathrm { R e f l } _ { E } \left( q , a _ { q } ^ { \star } , \{ \mathrm { M L L M } _ { i } ( q ) \} _ { i = 1 } ^ { N } \right)
$$

$$
( { \mathrm { e x e m p l a r r e c o r d } } ) ,\tag{2}
$$

$$
\kappa ( q ) = \operatorname { C a t } ( q )
$$

$$
( \mathrm { c a t e g o r y  t a g } ) .\tag{3}
$$

The classifier Cat is the only model that consumes $q$ alone: it routes the problem against the existing category registry without inspecting the rollouts. The three stores these artifacts feed into are related by a strict dependency. The reflection store $\mathcal { M } _ { R } = \{ \rho ( q ) \} _ { q }$ is an intermediate object never retrieved at inference; it exists solely as source material for the two downstream stores. The exemplar memory $\mathcal { M } _ { E }$ co-locates $\rho ( q )$ inside each exemplar, so that a single retrieval delivers both the instance-grounded strategy and its associated abstract rules. The schema memory $\mathcal { M } _ { S }$ is synthesized after the pass completes, exclusively from $\mathcal { M } _ { R }$ grouped by $\kappa ,$ without ever consulting the contents of $\mathcal { M } _ { E }$

All three summarizations are bound by an abstraction constraint: numeric values, option letters, variable bindings, and named entities drawn from $q$ are forbidden in any stored rule, strategy, or check. This is a correctness requirement, not a stylistic preference, because each stored item is consumed in the context of a different problem at test time; any leaked specifics would propagate directly into the solver’s conditioning.

Exemplar Memory. An exemplar $e ( q ) \in \mathcal { M } _ { E }$ is a self-contained record engineered to transfer a single training problem’s lesson to a new, conceptually similar problem. The reflector $\mathrm { R e f l } _ { E }$ distills the labeled rollout group into a two-field summary: a reasoning strategy that verbalizes a step-by-step procedure applicable to any problem of the same type, and a verification check that names the dominant error mode exhibited by the incorrect trajectories.

Each exemplar is indexed by a multimodal embedding $\operatorname { E m b } _ { M } ( I _ { q } , t _ { q } )$ , where $I _ { q }$ is the problem image and $t _ { q }$ is the short query “This problem tests $\mathcal { K } ( q )$ . Question: $\boldsymbol { q . } ^ { \prime \prime }$ . Joint indexing matters in practice: problems that share textual concepts but differ visuall $\mathrm { y - e . g . , a }$ full circle vs. a circular sector, or a bar chart vs. a stacked bar chart – are near-miss distractors that a purely textual retriever cannot separate.

Schema Memory. A schema $\sigma _ { \kappa } \in \mathcal { M } _ { S }$ captures what is stable across a concept-equivalent class of problems. Once the online pass over the training set has assigned every problem a category $\kappa ( q )$ , we partition $\mathcal { M } _ { R }$ by category and apply the schema synthesizer Synth to each partition:

$$
\sigma _ { \kappa } = \mathrm { S y n t h } \left( \kappa , ~ \mathrm { c o n c e p t s } ( \kappa ) , \bigcup _ { q : \kappa ( q ) = \kappa } \rho ( q ) \right) ,\tag{4}
$$

where $\mathrm { c o n c e p t s } ( \kappa )$ denotes the representative key concepts attached to category κ in the registry. $\mathrm { S y }$ nth is instructed to eliminate redundancy, reconcile near-duplicate rules, and elevate instance-specific phrasing into principles applicable to the entire category. Its output is a compact list of prescriptive advisories that tell the solver what to do rather than merely what to avoid. Two properties of this construction warrant emphasis. First, consistent with the schema-mediated consolidation view [25], synthesis operates strictly over $\mathcal { M } _ { R } \colon$ trajectories and exemplar-level text are never shown to Synth, confining $\sigma _ { \kappa }$ to principles derivable from abstract rules and limiting the risk of memorizing solution strings. Second, because $\sigma _ { \kappa }$ aggregates over many problems sharing $\kappa ,$ a spurious rule from any single $\mathrm { R e f l } _ { R }$ call is diluted rather than amplified, endowing $\mathcal { M } _ { S }$ with a robustness property that $\mathcal { M } _ { E }$ , being instance-grounded, does not share.

Online Categorization. Partitioning $\mathcal { M } _ { R }$ requires a category label per problem, but a fixed taxonomy is either too coarse (a single “geometry” bucket) or too fine (one bucket per concept signature), and the right granularity is unknown a priori. We instead grow the registry online: Cat extracts $\bar { \kappa } ( q )$ and compares it against the running registry restricted to identifier-name pairs (to minimize prompt overhead), then either assigns $q$ to an existing category or introduces a new one with a fresh identifier and representative concepts. Registry updates are serialized so the category space grows monotonically; the resulting $\kappa ( q )$ is recorded alongside $\rho ( q )$ and $e ( q )$ , and the classification step runs concurrently with $\mathrm { R e f l } _ { E }$ and $\mathrm { R e f l } _ { R }$ at no sequential overhead.

## 3.2 Memory Access

At test time $\mathcal { M } = \mathcal { M } _ { S } \cup \mathcal { M } _ { E }$ is frozen; $\mathcal { M } _ { R }$ is never consulted directly, since its content reaches inference only through the two derived stores. Access proceeds in three retrieval stages, followed by a fallback mechanism against reasoning-loop collapse.

Schema Retrieval. Schemas are retrieved at the granularity of individual rules within the categories whose representative concepts most closely match $\mathcal { K } ( q )$ . Cue similarity between rule $r _ { i }$ and the query is computed as sim $. ( r _ { i } , q ) = \cos ( \mathrm { E m b } _ { T } ( r _ { i } ) , \mathrm { E m b } _ { T } ( q ) )$ using the text embedder Emb . We compose cue similarity with the rule’s accumulated utility $U ( r _ { i } )$ (defined in $\ S 3 . 3 )$

$$
s _ { i } = \alpha \widehat { \mathrm { s i m } } ( r _ { i } , q ) + ( 1 - \alpha ) \widehat { U } ( r _ { i } ) ,\tag{5}
$$

where $\widehat { ( \cdot ) }$ denotes min-max normalization across the candidate set under the active category, applied independently to similarity and utility so that magnitude differences between the two scales do not dominate the ranking. $\alpha = 1$ collapses to similarity-only retrieval and recovers a memory-only baseline; $\alpha = 0$ ranks purely by accumulated utility.

Exemplar Retrieval. Exemplars are retrieved by cosine similarity between the multimodal embedding of the query $\operatorname { E m b } _ { M } ( I _ { q } , t _ { q } )$ and the stored multimodal embedding Emb<sub>M</sub> $\left( I _ { q ^ { \prime } } , t _ { q ^ { \prime } } \right)$ of each candidate exemplar $e ( q ^ { \prime } ) \in \mathcal { M } _ { E }$ . A similarity threshold $\tau _ { E }$ filters the candidate pool: if no candidate clears $\tau _ { E }$ , no exemplar is injected.

Memory Injection. Retrieved memories are inserted into the solver’s context in one of two complementary modes: priming, which precedes the question and frames the solver’s approach a priori, and verification, which follows the question and reserves the same content for a post-hoc audit of the produced answer.

## 3.3 Utility Training

After Stage 1 every rule in $\mathcal { M } _ { S }$ is treated identically by retrieval, yet a category’s rules are rarely uniformly useful: some encode dominant failure modes that transfer broadly, while others are narrow tips whose conditions seldom fire. A second offline pass turns the same backbone into an attribution device: it draws additional rollouts in which the only varying factor is which subset of category rules is injected, and converts the resulting reward signal into a per-rule utility column that re-weights retrieval at test time.

Frozen-Category Assignment. Stage 2 reads back the schema memory $\mathcal { M } _ { S }$ and the category registry produced in Stage 1; both are frozen for the remainder of training so that utility statistics accumulate on a fixed rule support.

Subset Rollouts. For problem q we draw a batch of B rule subsets $\{ S _ { b } \} _ { b = 1 } ^ { B }$ that always contains the empty subset ∅ (query-only baseline) and the full subset $\mathcal { R } _ { \kappa ^ { \star } } $ the remaining $B - 2$ subsets are sampled uniformly without replacement from the strict middle layers $\{ S \subseteq \mathcal { R } _ { \kappa ^ { \star } } : 1 \leq | S | \leq K ^ { \overline { { - } } } 1 \}$ . When $B \geq 2 ^ { K }$ the batch is the exact powerset and Shapley becomes exact. For each subset, the backbone is conditioned on the same problem with only the injected rule set varying, $o _ { b } = \mathrm { M L L M } ( q , S _ { b } )$ , and the resulting binary reward $v ( S _ { b } ) = \mathrm { V e r i f y } \big ( o _ { b } , a _ { q } ^ { \star } \big ) \in \{ 0 , \dot { 1 } \}$ } isolates the contribution of $S _ { b }$ itself, since temperature, decoding seed, and image content are held fixed across b.

Context Attribution. Treating rules as players and the observed subset rewards $v ( \cdot )$ as the cooperative value, the contribution of rule $r _ { i }$ to problem q is its average marginal gain across all permutations of $\mathcal { R } _ { \kappa ^ { \star } }$

$$
\Delta _ { q } ( r _ { i } ) = \frac { 1 } { K ! } \sum _ { \pi \in \mathfrak { S } _ { K } } \Big [ v \big ( P _ { i } ^ { \pi } \cup \{ r _ { i } \} \big ) - v \big ( P _ { i } ^ { \pi } \big ) \Big ] ,\tag{6}
$$

where $P _ { i } ^ { \pi }$ is the set of rules preceding $r _ { i }$ in π and unobserved coalitions are imputed as 0. Shapley distributes credit unevenly among rules in proportion to where each one most increases the marginal reward, at an $O ( K ! )$ cost in the exact limit.

Each rule $r _ { i }$ maintains a running sum and a count $n _ { i }$ of how many problems’ batches updated it; its accumulated utility is the empirical mean $\begin{array} { r } { U ( r _ { i } ) = \breve { \sum _ { q : r _ { i } \in \mathcal { R } _ { \kappa ( q ) } } } \Delta _ { q } ( r _ { i } ) / \mathrm { m a x } ( n _ { i } , 1 ) } \end{array}$ . Sums and counts are merged into the persisted category file under file-level locking, so that Stage 2 can be parallelized across problems and resumed without double-counting. Exemplars are intentionally excluded from this update: because Stage-2 problems can collide with Stage-1 problems, attributing utility to an exemplar whose embedding indexes the same problem would leak. Exemplars therefore continue to be ranked by similarity alone.

At test time, $U ( r _ { i } )$ is plugged into Eq. 5, completing the rational-analysis composition: two rules with equal cue match are ranked by their accumulated track record.

## 4 Experiments

## 4.1 Main Results on Multimodal Benchmarks

Tasks and datasets. Our target benchmarks are mainly multi-modal scientific problem solving and reasoning, including MathVista [13], MMMU [35], MMMU-Pro [36]. For each benchmark we adopt a fixed 3:1 train/test split, where the test set is a subset of the full benchmark. Only the gold answer of training problems are needed as supervision; no chain-of-thought or per-step reasoning annotations are required. The training partition is consumed exclusively offline to construct $\mathcal { M } _ { E }$ and $\mathcal { M } _ { S }$ , and the test partition is held out for all reported numbers. At evaluation time the memory store is frozen: no rollouts, reflections, or schema updates are produced during testing.

Models We instantiate DG-Mem with four backbones including open-weight and proprietary regimes: Qwen3.5- 27B [24], Qwen3.5-122B-A10B [24], GPT-5-Nano [16], and Gemini-3-Flash [7]. Except for the LLM judge used by the verification cascade, which is fixed to Qwen-3.5-Flash [24] across all experiments, every node in the agent graph (rollout solver, reflectors, schema synthesizer, online categorizer) uses the same backbone. This homogeneous configuration enforces the “self-evolving” setting and prevents implicit knowledge distillation from a larger or heterogeneous model. All text retrievers use Qwen-3-Embedding [42]; all multimodal retrievers use Qwen-3-VL-Embedding [12].

Baselines. We compare against two families of methods that share our no-additional-supervision constraint. ViLoMem [3] is a training-free dual-stream memory agent that grows logical and visual error guidelines online during evaluation; it represents the “memory without offline consolidation” regime. We implement ViLoMem into two settings: offline and online. For offline, it freezes the trained memory and evaluates on the test split. For online, it continues to grow and refine the memory with the test split. We report the top-1 accuracies on the test split. We additionally report a No-memory baseline that runs the same backbone with an identical decoding configuration but no retrieval, isolating the contribution of the memory pipeline itself.

Evaluation metrics. We report top-1 accuracy on the held-out test split of each benchmark, judged by the same verification cascade used during memory construction (native matcher → symbolic parser → LLM judge); reporting the cascade’s verdict avoids spurious gaps caused by formatting differences.

Table 2 reports test accuracy across the three benchmarks for each backbone, comparing the No-memory baseline, ViLoMem [3], and DG-Mem. DG-Mem delivers consistent gains over the No-memory baseline across backbones.

## 4.2 Ablation Studies

We isolate the contribution of each component by ablating one piece of the pipeline at a time on the Gemini-3-Flash backbone. Table 3 summarizes the sweep; the paragraphs below interpret each row.

Effects of Exemplar Memory. Removing $\mathcal { M } _ { E }$ from the full method costs −3.2, −2.6, and −3.7 on MathVista/MMMU/MMMU-Pro, with the largest drop on MMMU-Pro where many problems hinge on a specific diagrammatic configuration (circuit topologies, geometric constructions) that an IF-THEN rule abstracts away but a multimodal nearest neighbor preserves. Schema-only is also the only condition that falls slightly below the No-memory baseline on MMMU (82.1 vs. 82.4), indicating that schemas alone can over-generalize on broad multi-discipline questions and that $\mathcal { M } _ { E }$ acts as a corrective check against schema misfires.

Table 2: Top-1 accuracy (%) on held-out test splits across multimodal reasoning benchmarks. Each backbone is evaluated under four conditions: No-memory (same backbone and decoding configuration with retrieval disabled), ViLoMem (offline) (logical/visual error memory frozen after training), ViLoMem (online) (memory continues to grow on the test split), and DG-Mem (Ours). Avg Imp. reports the per-backbone mean improvement over the No-memory baseline across the three benchmarks. \* on each benchmark name indicates that we report on a fixed held-out test partition rather than the full benchmark. Best result per backbone is in bold.
<table><tr><td>Backbone</td><td>Method</td><td>MathVista*</td><td>MMMU*</td><td>MMMU-Pro*</td><td>Avg Imp.</td></tr><tr><td rowspan="4">Qwen3.5-122B-A10B</td><td>No-memory</td><td>83.6</td><td>85.8</td><td>77.9</td><td></td></tr><tr><td>ViLoMem (offline)</td><td>85.2</td><td>86.4</td><td>80.0</td><td>+1.4</td></tr><tr><td>ViLoMem (online)</td><td>88.4</td><td>87.4</td><td>80.3</td><td>+2.9</td></tr><tr><td>DG-Mem (Ours)</td><td>88.7</td><td>89.2</td><td>86.1</td><td>+5.6</td></tr><tr><td rowspan="4">Qwen3.5-27B</td><td>No-memory</td><td>82.9</td><td>83.4</td><td>75.1</td><td></td></tr><tr><td>ViLoMem (offline)</td><td>86.4</td><td>85.9</td><td>74.6</td><td>+1.8</td></tr><tr><td>ViLoMem (online)</td><td>87.2</td><td>86.3</td><td>74.8</td><td>+2.3</td></tr><tr><td>DG-Mem (Ours)</td><td>87.0</td><td>87.9</td><td>73.6</td><td>+2.3</td></tr><tr><td rowspan="4">GPT-5-Nano</td><td>No-memory</td><td>64.8</td><td>67.6</td><td>48.2</td><td></td></tr><tr><td>ViLoMem (offline)</td><td>66.8</td><td>56.9</td><td>51.0</td><td>-2.0</td></tr><tr><td>ViLoMem (online)</td><td>70.4</td><td>58.0</td><td>52.5</td><td>+0.1</td></tr><tr><td>DG-Mem (Ours)</td><td>74.4</td><td>79.4</td><td>64.3</td><td>+12.5</td></tr><tr><td rowspan="4">Gemini-3-Flash</td><td>No-memory</td><td>87.2</td><td>82.4</td><td>69.0</td><td></td></tr><tr><td>ViLoMem (offline)</td><td>80.4</td><td>82.8</td><td>73.8</td><td>-0.2</td></tr><tr><td>ViLoMem (online)</td><td>83.6</td><td>82.4</td><td>71.1</td><td>-0.2</td></tr><tr><td>DG-Mem (Ours)</td><td>91.6</td><td>84.7</td><td>77.1</td><td>+5.2</td></tr></table>

Table 3: Component ablations on Gemini-3-Flash. Each row disables a single mechanism while leaving the rest of the pipeline intact. w/o $\mathcal { M } _ { E }$ retains the category-level schema memory but removes the exemplar store, so retrieval returns only IF-THEN rules. w/o $\mathcal { M } _ { S }$ keeps exemplars but disables schema retrieval, so the prompt is augmented only with nearest-neighbor solved instances. w/o utility sets the retrieval interpolation $\alpha { = } 1$ , ranking rules by embedding similarity alone and discarding the Stage-2 Shapley utility column. Colored deltas (green/red) report the absolute change in accuracy relative to the No-memory Baseline row.
<table><tr><td>Datasets</td><td>MathVista*</td><td> ${ \bf M M M } ^ { * }$ </td><td> $\mathrm { M M M U - P r o ^ { * } }$ </td></tr><tr><td>Baseline</td><td>86.2</td><td>82.4</td><td>69.0</td></tr><tr><td>- w/o  $\mathcal { M } _ { E }$  (stage-1 schema only)</td><td> $8 8 . 4 + 2 . 2$ </td><td> $8 2 . 1 \ - 0 . 3 $ </td><td> $7 3 . 4 + 4 . 4$ </td></tr><tr><td>- w/o  $\mathcal { M } _ { S }$  (stage-1 exemplar only)</td><td> $8 7 . 6 + 1 . 4$ </td><td> $8 4 . 0 + 1 . 6$ </td><td> $7 3 . 4 + 4 . 4$ </td></tr><tr><td>- w/o utility (α=1, stage-1 both memory)</td><td> $9 1 . 0 + 4 . 8 $ </td><td> $8 4 . 0 + 1 . 6$ </td><td> $7 5 . 0 + 6 . 0$ </td></tr><tr><td>Full method</td><td> $9 1 . 6 + 5 . 4$ </td><td> $8 4 . 7 + 2 . 3 $ </td><td> $7 7 . 1 + 8 . 1 $ </td></tr></table>

Effects of Schema Memory. The exemplar-only ablation trails the full method by $- 4 . 0 , - 0 . 7 ,$ and −3.7, with the largest gap on MathVista where many test problems share a high-level solution template (parsing a bar chart, applying a unit conversion) but are visually dissimilar from any single training instance – exactly the regime where a transferable rule beats a literal nearest neighbor. That exemplar-only and schema-only land within $\sim 1 \%$ of each other yet the full method exceeds both indicates the two grains cover non-overlapping failure modes rather than redundant signal, which is what the schema / exemplar separation through $\mathcal { M } _ { R }$ is designed to enforce.

Effects of Utility. Setting $\alpha { = } 1$ (similarity-only retrieval over $\mathcal { M } _ { S } )$ costs −0.6, −0.7, and −2.1 relative to the full method, with the gap widening on the harder MMMU-Pro split. This shows that not every embedding-similar rule is useful: many are too generic, fitted to a different sub-population of the category, or actively misleading; Shapley-derived utility re-weights retrieval toward historically reward-bearing rules. The absolute margin is small because similarity already narrows the candidate set, but the consistent direction across all three benchmarks indicates utility is doing real work rather than tie-breaking, and we expect it to widen as $\mathcal { M } _ { S }$ accumulates more superficially similar rules per category.

<table><tr><td>Test question</td><td>Retrieved exemplar (M E)</td><td>Retrieved schema rule (MS)</td><td>Baseline → Ours</td></tr><tr><td>matte blocks. Subtract all tiny brown all tiny matte balls. How many objects.&quot; Strategy: an exhaustive initial census of all objects, assign a Baseline: 4 (skipped enumeration; cylinders. How many objects are left?&quot; “Catalog every object&#x27;s color, size, material, and shape, count of zero to any missing categories, and calculate &quot;There are 4 objects left.&quot;)</td><td>attribute categories before subtracting.&quot;</td><td>MathVista #90. &quot;Subtract all yellow Solved template: &quot;Subtract all red things. Subtract Visual Counting and Attribute Analysis: &quot;Perform Gold: 5 then translate the natural-language descriptors to those the cardinality of each subset independently before Ours: 5 (lists 7 objects, removes 2) applying the subtraction.&quot;</td><td></td></tr><tr><td>chart)</td><td>MathVista #13. &quot;How many objects are Solved template: &quot;How many algorithms have accu- Data Analysis and Interpretation: &quot;When identifying Gold: 0 it meets a &#x27;strictly greater than&#x27; or strictly less than condition.&quot;</td><td>preferred by more than 90% of people racy higher than 9 in at least one dataset?&quot; Verification threshold-based counts, systematically compute the Baseline: 2 (visually estimated bars in at least one category?&quot; (grouped bar check: &quot;Re-evaluate any bar that appears to end ex- metric for every distinct item rather than relying on near the 90% line) actly on a midpoint or grid line to confirm whether visual estimation of distances or assumed ordering.&quot;</td><td>Ours: 0 (computes per-bar height, applies strict &gt;)</td></tr><tr><td>between these two people?&quot; (photo of P. a different pair. Verification check: &quot;Cross-verify the tract the earlier birth year from the later birth year, Baseline: 3 (used wrong birth year Hammond and M. Kaljurand)</td><td>MathVista #792. &quot;What is the age gap Solved template: same age-gap question structure on Arithmetic: Age Problems and Data Extraction: &quot;Sub- Gold: 7 primary subject&#x27;s known associates or event context, fied in the query.&quot; as misidentification of this figure is the most frequent source of error.&quot;</td><td>identity of the less prominent individual against the ensuring the values belong to the exact subjects speci- for Kaljurand: 1958)</td><td>Ours: 7 (corrects to 1962, then 1962-1955)</td></tr></table>

Figure 2: Qualitative cases where DG-Mem corrects the No-memory backbone. Each row is a held-out test instance for which both retrieved grains are populated and the same baseline error is corrected under category-only, exemplar-only, and both-memory retrieval, ruling out the alternative explanation that any prompt perturbation flips the answer. The retrieved schema rule names the procedure the baseline omitted (an exhaustive census; per-item threshold computation; grounding the arithmetic in the correct entities), and the retrieved exemplar supplies a solved template whose verification check directly addresses the failure mode (boundary-bar evaluation; identity disambiguation of the less prominent subject).

## 4.3 Qualitative Results

Qualitative cases where dual-grained memory corrects the backbone. Figure 2 shows three test instances on which the No-memory backbone is wrong but the same backbone with our retrieved memory is correct, and the same correction also occurs under category-only and exemplar-only retrieval (i.e., the fix is not specific to one grain). For each case we surface the retrieved exemplar (instance-grounded $\mathcal { M } _ { E } )$ and the most relevant schema rule (category-level $\mathcal { M } _ { S } )$ so that the source of the correction is auditable rather than incidental. The three cases span visual enumeration after attribute filtering (CLEVR-style subtraction), threshold counting from a chart, and entity-grounded arithmetic over public figures, and in each case the retrieved schema names the procedure the baseline omitted while the retrieved exemplar supplies a near-isomorphic solved template.

## 5 Limitations

We highlight several limitations that may bound the scope of our claims and point to natural directions for follow-up work.

Stochasticity of attribution. Shapley context attribution treats the backbone as a deterministic value function on rule subsets. But non-deterministic GPU kernels, dynamic batching, and floating-point non-associativity [8] mean binary rewards $v ( S _ { b } )$ can vary across identical-context calls. With a finite subset budget $B , \Delta _ { q } ( r _ { i } )$ inherits this background noise: a rule with small true marginal effect can register non-zero $\Delta _ { q }$ from answer-flipping unrelated to the rules injected, and the sign of $U ( r _ { i } )$ becomes data-order-dependent. We mitigate by averaging across many problems sharing the same rule and ranking rather than thresholding on $U ( r _ { i } )$ , but the residual variance is a property of the value function; multi-rollout majority-vote rewards would harden this further at additional training cost.

Frozen memory at test time. We freeze M at evaluation time so that test accuracy reflects retrieval and conditioning rather than additional online learning, and so that comparisons against ViLoMem (offline) are head-to-head. This is a clean experimental choice but forfeits the continual-learning regime in which test-time errors could themselves seed new exemplars or rules. Combining offline consolidation with bounded online updates remains an open design question.

## 6 Conclusions

We presented DG-Mem, a dual-grained agentic memory framework that equips frozen multimodal backbones with an instance-grounded exemplar memory and a category-level schema memory, strictly separated by a transient reflection store. DG-Mem addresses two key challenges in agentic memory. First, online incremental categorization dynamically tracks the corpus’s empirical concept distribution without a fixed, manual taxonomy. Second, Shapley context attribution repurposes the frozen backbone for credit assignment, optimizing retrieval ranking by empirical utility rather than mere similarity. This gradient-free pipeline is readily deployable on both closed-weight and on-device models.

Empirically, DG-Mem consistently outperforms memory-less baselines across four backbones and three multimodal reasoning benchmarks, as well as a strong single-grain baseline (ViLoMem) in competitive configurations. Ablations confirm that exemplar and schema memories address distinct failure modes, while utility re-weighting yields a consistent margin expected to widen as the memory scales. Ultimately, DG-Mem operationalizes a Complementary Learning Systems view of agentic memory, establishing attribution estimators and bounded online consolidation as highly promising directions for future research.

## References

[1] John R Anderson and Lael J Schooler. Reflections of the environment in memory. Psychological science, 2(6):396–408, 1991.

[2] Huan ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Qihan Ren, Yiran Wu, Hongru WANG, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Cheng Qian, Zhenhailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, and Mengdi Wang. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. Transactions on Machine Learning Research, 2026. Survey Certification.

[3] Weihao Bo, Shan Zhang, Yanpeng Sun, Jingjing Wu, Qunyi Xie, Xiao Tan, Kunbin Chen, Wei He, Xiaofan Li, Na Zhao, et al. Agentic learner with grow-and-refine multimodal semantic memory. arXiv preprint arXiv:2511.21678, 2025.

[4] Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, et al. Training-free group relative policy optimization. arXiv preprint arXiv:2510.08191, 2025.

[5] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building productionready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

[6] Muxin Fu, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, and Yang Yang. Latentmem: Customizing latent memory for multi-agent systems. arXiv preprint arXiv:2602.03036, 2026.

[7] Google. Gemini 3 flash, 2026. Accessed: April 25, 2026.

[8] Horace He and Thinking Machines Lab. Defeating nondeterminism in llm inference. Thinking Machines Lab: Connectionism, 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/.

[9] Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, et al. Memory in the age of ai agents. arXiv preprint arXiv:2512.13564, 2025.

[10] Wei-Chieh Huang, Weizhi Zhang, Yueqing Liang, Yuanchen Bei, Yankai Chen, Tao Feng, Xinyu Pan, Zhen Tan, Yu Wang, Tianxin Wei, Shanglin Wu, Ruiyao Xu, Liangwei Yang, Rui Yang, Wooseong Yang, Chin-Yuan Yeh, Hanrong Zhang, Haozhen Zhang, Siqi Zhu, Henry Peng Zou, Wanjia Zhao, Song Wang, Wujiang Xu, Zixuan Ke, Zheng Hui, Dawei Li, Yaozu Wu, Langzhou He, Chen Wang, Xiongxiao Xu, Baixiang Huang, Juntao Tan, Shelby Heinecke, Huan Wang, Caiming Xiong, Ahmed A. Metwally, Jun Yan, Chen-Yu Lee, Hanqing Zeng, Yinglong Xia, Xiaokai Wei, Ali Payani, Yu Wang, Haitong Ma, Wenya Wang, Chenguang Wang, Yu Zhang, Xin Wang, Yongfeng Zhang, Jiaxuan You, Hanghang Tong, Xiao Luo, Xue Liu, Yizhou Sun, Wei Wang, Julian McAuley, James Zou, Jiawei Han, Philip S. Yu, and Kai Shu. Rethinking memory mechanisms of foundation agents in the second half: A survey, 2026.

[11] Dharshan Kumaran, Demis Hassabis, and James L McClelland. What learning systems do intelligent agents need? complementary learning systems theory updated. Trends in cognitive sciences, 20(7):512–534, 2016.

[12] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-ofthe-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

[13] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024.

[14] James L McClelland, Bruce L McNaughton, and Randall C O’Reilly. Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. Psychological review, 102(3):419, 1995.

[15] Robert M Nosofsky. Attention, similarity, and the identification–categorization relationship. Journal ofexperimental psychology: General, 115(1):39, 1986.

[16] OpenAI. Openai gpt-5 system card, 2025.

[17] Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. Reasoningbank: Scaling agent self-evolving with reasoning memory. In The Fourteenth International Conference on Learning Representations, 2026.

[18] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.

[19] Zhiyu Shen, Ziming Wu, Fuming Lai, Shaobing Lian, and Yanghui Rao. Membuilder: Reinforcing llms for long-term memory construction via attributed dense rewards. arXiv preprint arXiv:2601.05488, 2026.

[20] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[21] Daphna Shohamy and R Alison Adcock. Dopamine and adaptive memory. Trends in cognitive sciences, 14(10):464–472, 2010.

[22] Annabelle C Singer and Loren M Frank. Rewarded outcomes enhance reactivation of experience in the hippocampus. Neuron, 64(6):910–921, 2009.

[23] Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, et al. Agent kb: Leveraging cross-domain experience for agentic problem solving. arXiv preprint arXiv:2507.06229, 2025.

[24] Qwen Team. Qwen3.5-omni technical report. arXiv preprint arXiv:2604.15804, 2026.

[25] Dorothy Tse, Rosamund F Langston, Masaki Kakeyama, Ingrid Bethus, Patrick A Spooner, Emma R Wood, Menno P Witter, and Richard GM Morris. Schemas and memory consolidation. Science, 316(5821):76–82, 2007.

[26] Songjun Tu, Chengdong Xu, Qichao Zhang, Yaocheng Zhang, Xiangyuan Lan, Linjing Li, and Dongbin Zhao. Dynamic dual-granularity skill bank for agentic rl. arXiv preprint arXiv:2603.28716, 2026.

[27] Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. Mem-α: Learning memory construction via reinforcement learning. arXiv preprint arXiv:2509.25911, 2025.

[28] Zihao Wang, Shaofei Cai, Anji Liu, Yonggang Jin, Jinbing Hou, Bowei Zhang, Haowei Lin, Zhaofeng He, Zilong Zheng, Yaodong Yang, et al. Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):1894–1907, 2024.

[29] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In Forty-second International Conference on Machine Learning, 2025.

[30] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for LLM agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[31] Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z Pan, et al. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828, 2025.

[32] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[33] Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E Gonzalez, and Bin Cui. Buffer of thoughts: Thought-augmented reasoning with large language models. Advances in Neural Information Processing Systems, 37:113519–113544, 2024.

[34] Ruozhen Yang, Yucheng Jiang, Yueqi Jiang, Priyanka Kargupta, Yunyi Zhang, and Jiawei Han. Grounding agent memory in contextual intent. arXiv preprint arXiv:2601.10702, 2026.

[35] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

[36] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, 2025.

[37] Yanwei Yue, Boci Peng, Xuanbo Fan, Jiaxin Guo, Qiankun Li, and Yan Zhang. Mem-t: Densifying rewards for long-horizon memory agents. arXiv preprint arXiv:2601.23014, 2026.

[38] Guibin Zhang, Muxin Fu, Kun Wang, Guancheng Wan, Miao Yu, and Shuicheng YAN. G-memory: Tracing hierarchical memory for multi-agent systems. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[39] Guibin Zhang, Muxin Fu, and Shuicheng Yan. Memgen: Weaving generative latent memory for self-evolving agents. arXiv preprint arXiv:2509.24704, 2025.

[40] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. arXiv preprint arXiv:2510.04618, 2025.

[41] Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, et al. Memrl: Self-evolving agents via runtime reinforcement learning on episodic memory. arXiv preprint arXiv:2601.03192, 2026.

[42] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

[43] Yuxiang Zhang, Jiangming Shu, Ye Ma, Xueyuan Lin, Shangxi Wu, and Jitao Sang. Memory as action: Autonomous context curation for long-horizon agentic tasks. arXiv preprint arXiv:2510.12635, 2025.

[44] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642, 2024.

[45] Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. Skillweaver: Web agents can self-improve by discovering and honing skills, 2025.

[46] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 19724–19731, 2024.

[47] Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun Zhang, Kun Shao, Linyi Yang, et al. Memento: Fine-tuning llm agents without fine-tuning llms. arXiv preprint arXiv:2508.16153, 2025.

[48] Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. arXiv preprint arXiv:2506.15841, 2025.

## Appendix

## A Additional Experimental Details

## A.1 Datasets and Splits

We evaluate on three multimodal reasoning benchmarks and use a fixed 3:1 train/test partition per benchmark. Only the gold answer of training problems is consumed as supervision; no chain-of-thought or step-level annotations are used. Train problems drive Stage-1 memory construction and Stage-2 utility attribution; test problems are held out and evaluated with the memory frozen.

Table 4: Train/test partitions per benchmark. The same split file is used by every backbone and by every method (No-memory, ViLoMem, DG-Mem) so that comparisons are head-to-head.
<table><tr><td>Benchmark</td><td>Train (Stage 1 &amp; 2)</td><td>Test (held out)</td><td>Total</td></tr><tr><td rowspan="2">MathVista (MINI) MMMU (DEV+VAL)</td><td>750</td><td>250</td><td>1,000</td></tr><tr><td>788</td><td>262</td><td>1,050</td></tr><tr><td>MMMU-Pro</td><td>1,298</td><td>432</td><td>1,730</td></tr></table>

## A.2 Test-Time Retrieval

At test time $\mathcal { M } = \mathcal { M } _ { S } \cup \mathcal { M } _ { E }$ is frozen. For each query we run the concept summarizer K once, retrieve the top categories whose representative concepts clear the category-similarity gate, then retrieve up to $K _ { \mathrm { m a x } }$ rules per active category ranked by Eq. (5). Exemplars are retrieved by multimodal cosine similarity against Emb<sub>M</sub> $( I _ { q } , t _ { q } )$ subject to a similarity-threshold gate; if no exemplar clears the gate, none is injected.

Table 5: Test-time retrieval and decoding hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Description</td></tr><tr><td>α</td><td>0.8</td><td>Similarity / utility interpolation in Eq. (5)</td></tr><tr><td>Category similarity threshold</td><td>0.40</td><td>Min cosine similarity for category retrieval</td></tr><tr><td>Schema rule similarity threshold</td><td>0.35</td><td>Min cosine similarity for rule retrieval</td></tr><tr><td> $K _ { \mathrm { m a x } }$  (rules per category)</td><td>2</td><td>Top-k rules retrieved per active category</td></tr></table>

## B Additional Ablation Studies

We complement the Gemini-3-Flash ablations in the main text (Table 3) with the same component sweep on two additional backbones, GPT-5-Nano (Table 6) and Qwen3.5-122B-A10B (Table 7). The pattern is consistent across both backbones: each grain alone improves over the No-memory baseline on every benchmark, and the full method exceeds the better of the two single-grain ablations on every cell, confirming that exemplar and schema memories cover non-overlapping failure modes rather than redundant signal.

Table 6: Component ablations on GPT-5-Nano. Each row disables a single mechanism while leaving the rest of the pipeline intact.
<table><tr><td>Datasets</td><td> $\mathrm { { M a t h V i s t a ^ { * } } }$ </td><td>MMMU*</td><td> $\mathrm { M M M U - P r o ^ { * } }$ </td></tr><tr><td>Baseline</td><td>64.8</td><td>67.6</td><td>48.2</td></tr><tr><td>- w/o  $\mathcal { M } _ { S }$  (stage-1 exemplar only)</td><td> $6 8 . 8 + 4 . 0 $ </td><td> $7 2 . 5 + 4 . 9$ </td><td>55.6 +7.4</td></tr><tr><td>− w/o ME (stage-2 schema only)</td><td> $6 7 . 2 + 2 . 4 $ </td><td> $7 4 . 0 + 6 . 4 $ </td><td>58.1 +9.9</td></tr><tr><td>Full method</td><td> $7 4 . 4 + 9 . 6 $ </td><td> $7 9 . 4 + 1 1 . 8$ </td><td> $6 4 . 4 + 1 6 . 2 $ </td></tr></table>

Table 7: Component ablations on Qwen3.5-122B-A10B. Each row disables a single mechanism while leaving the rest of the pipeline intact.
<table><tr><td>Datasets</td><td>MathVista*</td><td>MMMU*</td><td>MMMU-Pro*</td></tr><tr><td>Baseline</td><td>83.6</td><td>85.8</td><td>77.9</td></tr><tr><td>– w/o Ms (stage-1 exemplar only)</td><td>87.2 +3.6</td><td>85.8 +0</td><td>80.9 +3.0</td></tr><tr><td>– w/o ME (stage-2 schema only)</td><td>86.4 +2.8</td><td>86.5 +0.7</td><td>81.3 +3.4</td></tr><tr><td>Full method</td><td>88.7 +5.1</td><td>89.2 +3.4</td><td>86.1 +8.2</td></tr></table>

## C Full Prompts

This section reproduces every prompt template that drives DG-Mem in verbatim form, so that the pipeline of §3 can be reimplemented without ambiguity. Each prompt is shown as a separate box; system prompts are highlighted in blue, user prompts in orange, and shared/utility templates in gray. Placeholders rendered as {{name}} are substituted at runtime with the corresponding field (e.g. {{question}} with the test problem text, {{rollout\_details}} with the formatted rollout block defined at the end of this section).

## C.1 Solver Prompt

The frozen backbone MLLM is invoked with a single, fixed system prompt that requests a step-by-step trace and a boxed final answer. The user turn carries the multimodal problem (image plus question text); the optional retrieved memory $\mathcal { M } _ { q }$ is prepended to that user turn at test time.

Solver – System   
Objective:   
Solve the given problem using a step by step process.   
Expected Output Structure:   
Step 1:   
Step 2:   
Step n: Final Answer: \boxed{answer}   
Question:

## C.2 Concept Summarizer (K)

The concept summarizer produces the per-problem key-concept signature K(q) used both for online categorization during training and for memory retrieval at test time. It is a single user-turn prompt; no system prompt is used, and the model is instructed not to attempt a solution.

Problem Analysis – User   
Objective:   
Analyze the following problem to identify its subject area and the   
key concepts, principles, formulas, or laws required for its   
solution. This analysis will be used to retrieve relevant guiding   
principles from a knowledge base.   
Instructions:   
- Do not solve the problem.   
- First, identify the primary subject (e.g., Physics, Chemistry,   
Biology, Mathematics).   
- Then, list the core concepts or principles involved (e.g., Newton’s   
Second Law, Conservation of Energy, Stoichiometry, Pythagorean   
theorem).   
- Keep the analysis concise and focused.

```handlebars
Problem:
{{question}}
Output Format:
Subject: <The primary subject>
Key Concepts: <A brief list of key concepts>
```

## C.3 Question-Level Reflector (Refl<sub>R</sub>, writes to M<sub>R</sub>)

The question-level reflector condenses the labeled rollout group into IF-THEN rules that populate the transient reflection store M . The abstraction constraints (no problem-specific numbers, options, or entities) are what make these rules safe to consolidate into the schema memory M<sub>S</sub> in the post-epoch step.

## Question-Level Reflector – System

You are an Expert Multimodal Problem-Solving Reflector. Your task is   
to extract concise, generalizable IF-THEN rules from multiple   
solution attempts for a single scientific or mathematical problem.   
You must analyze both visual interpretation errors (e.g., misreading   
diagrams, missing chart labels) and logical reasoning errors. These   
rules will be clustered by concept to serve as foundational strategic   
heuristics for future AI agents.

## Question-Level Reflector – User

Given the following multimodal problem, its key concepts, and multiple   
solution attempts with their verification results, extract   
generalizable IF-THEN rules.   
<problem>   
{{question}}   
</problem>   
<gold\_answer>   
{{gold\_answer}}   
</gold\_answer>   
<key\_concepts>   
{{key\_concepts}}   
</key\_concepts>   
<rollouts>   
{{rollout\_details}}   
</rollouts>   
Instructions:   
1. Analyze the rollouts to identify the critical decision points,   
visual interpretations, or logical steps that determined success   
or failure.   
2. Formulate rules as actionable IF-THEN statements.   
- The "IF" part should describe a specific problem context, visual   
pattern, or conceptual trigger (derived from the <key\_concepts>).   
- The "THEN" part should provide a specific visual or logical   
intervention (e.g., "THEN explicitly verify the axis scale",   
"THEN apply the Pythagorean theorem").   
3. Rule Categories (Try to extract at least one of each if applicable):   
- Positive Heuristics: "IF <context>, THEN <strategy>" -- what   
specifically worked in the correct attempts.   
- Visual/Perceptual Traps: "IF <visual context>, THEN AVOID

<perceptual pitfall>" -- visual elements the agent   
misinterpreted in failed attempts.   
- Logical/Calculation Traps: "IF <logical context>, THEN AVOID   
<reasoning pitfall>" -- logical fallacies or formula errors   
observed in failed attempts.   
4. Constraints for Generalization:   
- Rules MUST be tightly connected to the provided <key\_concepts>   
so they can be accurately grouped with similar problems later.   
- Keep rules to a single, concise sentence.   
5. Return 2-5 rules total. Quality and transferability over quantity.   
CRITICAL CONSTRAINT -- Abstraction:   
Rules MUST NOT contain problem-specific numbers, option letters   
(A/B/C/D), yes/no answers, variable names (unless standard formulas),   
or specific entity names. If a rule mentions any value from the   
gold\_answer or rollouts, rephrase it abstractly.   
GOOD: "IF measuring length with a ruler, THEN align the start of the   
object with the zero mark and read the nearest whole unit at   
the endpoint."   
BAD: "IF measuring length, THEN the endpoint is at the 8 cm mark."   
(contains problem-specific value)   
GOOD: "IF comparing counts of object categories, THEN systematically   
enumerate each category before comparing."   
BAD: "IF counting brown bikes, THEN there are 0 brown tandem bikes."   
(contains problem-specific entity and count)   
Output Format:   
Return a JSON object with a single key "rules" whose value is a list   
of 2-5 IF-THEN rule strings. Each rule must start with "IF " and   
contain "THEN ". If no meaningful rules can be extracted, return   
{"rules": []}.

## C.4 Compact Reflector (Refl<sub>E</sub>, writes to M<sub>E</sub>)

The compact reflector produces the exemplar entry stored in M<sub>E</sub>: a transferable reasoning\_strategy and a verification\_check. Crucially, this prompt is the only place that is allowed to reference the rollout text directly; the schema synthesizer below sees only the abstract IF-THEN rules in M , never the exemplar payload.

Compact Reflector – System   
You are an Expert Multimodal Problem Analyst. Your task is to extract   
a generalizable reasoning strategy and verification check from   
multiple solution attempts. Your output will guide a different AI   
agent solving a DIFFERENT problem of the same type -- it must be   
abstract enough to transfer, not specific to this instance.

```handlebars
Compact Reflector – User
Given a multimodal problem, its key concepts, and multiple solution
attempts, extract a transferable reasoning strategy.
<problem>
{{question}}
</problem>
<gold_answer>
{{gold_answer}}
</gold_answer>
<key_concepts>
```

{{key\_concepts}}   
</key\_concepts>   
<rollouts>   
{{rollout\_details}}   
</rollouts>   
Instructions:   
1. Analyze where correct and incorrect approaches diverge.   
2. Produce a JSON object with these fields:   
- "reasoning\_strategy": A step-by-step reasoning procedure that   
would work for ANY problem of this type. Describe WHAT to do   
and HOW to verify, not what the answer is.   
"verification\_check": A specific check the solver should perform   
to catch the most common error pattern (or null if all attempts   
were correct).   
CRITICAL CONSTRAINTS:   
- NEVER include specific numerical values, option letters (A/B/C/D),   
yes/no answers, or entity names from this problem.   
- The output must work for a DIFFERENT problem with DIFFERENT numbers   
and images.   
- Frame positively: describe what TO DO, not what to avoid.   
GOOD examples:   
- reasoning\_strategy: "Align the object’s starting edge with the zero   
mark, then read the position of the ending edge to the nearest   
whole unit."   
- verification\_check: "Re-read the ending position independently --   
ensure it is not confused with an adjacent marking."   
BAD examples (contain problem-specific values -- DO NOT do this):   
- reasoning\_strategy: "Count 8 individual cubes and add to 30 to get 38"   
- verification\_check: "The endpoint is at the 8 cm mark, not the 7 cm mark"   
Output Format (return ONLY this JSON, no other text):   
‘‘‘json   
{   
"reasoning\_strategy": "<generalizable step-by-step reasoning procedure>",   
"verification\_check": "<specific verification check or null>"   
}   
ccc

## C.5 Online Concept Categorizer (Cat)

The categorizer is the operational implementation of the online taxonomy growth described in §3.1. It receives the running registry of categories and decides whether to assign the new problem to an existing one or to spawn a new category, returning a structured action that the registry consumes verbatim.

## Online Categorizer – System

You are an Expert Problem Categorizer for multimodal scientific and mathematical problems. Your task is to assign each problem to a concept category, either selecting an existing category or creating a new one when no existing category is a good fit.

## Online Categorizer – User

Given a problem’s key concepts and the current set of existing categories, decide whether to assign the problem to an existing

category or create a new one.   
<problem\_concepts>   
{{key\_concepts}}   
</problem\_concepts>   
<existing\_categories>   
{{existing\_categories}}   
</existing\_categories>   
Instructions:   
1. Compare the problem’s key concepts against each existing   
category’s name and id.   
2. If a category is a strong conceptual match (>70% overlap in domain   
and principles), assign to it.   
3. If no category is a good fit, create a new one with a descriptive   
name and representative concepts.   
4. When assigning to an existing category, you may update its   
representative\_concepts to better reflect the expanded set of   
problems.   
Output Format (return ONLY this JSON, no other text):   
‘‘‘json   
{   
"action": "assign" or "create",   
"category\_id": "<existing category\_id if assign, or a new unique slug if create>",   
"category\_name": "<human-readable category name>",   
"representative\_concepts": ["concept1", "concept2", ...],   
"reasoning": "<1-sentence justification>"   
}

## C.6 Schema Synthesizer (Synth, writes to M<sub>S</sub>)

The schema synthesizer is the post-epoch consolidation step. It reads only the IF-THEN rules in M within a category (never the exemplar payload) and emits the prescriptive advice list that constitutes the category’s entry in M<sub>S</sub>.

## Category-Level Reflector – System

You are an Expert Multimodal Knowledge Synthesizer. Your task is to   
analyze a cluster of question-level IF-THEN rules within a specific   
scientific or mathematical concept category. You must synthesize   
these isolated experiences into coherent, highly generalizable advice   
to guide future AI agents in solving problems within this category.

## Category-Level Reflector – User

```handlebars
You are given a concept category, its representative concepts, and a
collection of IF-THEN rules extracted from solving multimodal
problems in this category. Synthesize them into coherent,
non-redundant, and actionable advice.
<category>
{{category_name}}
</category>
<representative_concepts>
{{representative_concepts}}
</representative_concepts>
```

{{rules\_text}}

1. Analyze and group the provided rules, eliminating redundancies and resolving contradictions. Elevate problem-specific rules into general principles applicable to the entire <category>.

- A single, self-contained, and actionable sentence starting with "- ".   
- \*\*Prescriptive\*\* -- tell the solver WHAT TO DO, not just what to avoid. Use "Do X when you see Y" rather than "Never assume X". \*\*Specific\*\* -- reference concrete visual patterns, mathematical relationships, or domain-specific checks. Avoid abstract warnings.   
- Broadly generalizable across any problem that tests the <representative\_concepts>.

4. Return 2-4 items total. Focus on the 2-4 most critical and specific insights.

\- BAD (too vague): "Never assume post-event birth years from visual cues." -- This tells the solver to not do something but gives no actionable alternative.

GOOD (prescriptive): "When estimating ages from photos to compare against a historical date, assess each person independently and count anyone who appears under 80 as likely born after 1945." -- This gives a concrete decision procedure.

GOOD (specific): "For scatter plots with overlapping points, trace each data point’s y-value to the nearest labeled tick mark before comparing against the threshold." -- This addresses a specific visual challenge.