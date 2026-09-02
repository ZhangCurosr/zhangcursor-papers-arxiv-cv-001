# ExBind: A Controlled Diagnostic Benchmark for Visual-to-Executable Correspondence

Ziqian Wang Yuxiao Cheng Tingxiong Xiao Jinli Suo<sup>\*</sup>

Department of Automation, Tsinghua University

Beijing 100084, China

Corresponding author: jlsuo@tsinghua.edu.cn

## Abstract

Iterative visual-feedback coding and editing systems repeatedly translate between rendered artifacts and executable program structure: after a model or user identifies a visible target, the system must resolve that referent to the precise object that can be edited. A failure at this correspondence layer can send an other wise sensible patch to the wrong DOM node, SVG child, graph endpoint, hierarchy member, or table cell, but end-to-end edit success does not reveal whether the cause was localization, correspondence, code generation, execution, or accumulated history. ExBind isolates visual/semantic-to-executable correspondence as a controlled diagnostic target between semantic localization and action execution. It samples representation-independent latent binding instances and compiles them into SVG, DOM, canvas, tree, graph, and table cases with invertible mappings to surface executable references. Models emit only the final strict reference; the evaluator maps it back to latent structure and reports deterministic structural constraints, without requiring model-generated reasoning traces. The primary release contains a 250-case broad suite, a disjoint 240-case targeted suite, and 50 paired latent groups. On the broad suite, Qwen2.5-VL-3B attains 98.4% candidate validity but 76.4% exact accuracy, whereas Qwen3-VL-4B attains 100.0% validity and 98.8% exact accuracy. In the targeted table diagnostic, every Qwen2.5-VL-3B residual error is a valid correct-row/wrong-column selection under the declared decomposi tion. Candidate-order perturbations change case-level outcomes while preserving this localized error form. ExBind is a con trolled diagnostic instrument for executable correspondence, rather than a population scale model leaderboard or an end-to-end editing benchmark. Code and benchmark records are publicly available at https:// github.com/Daerwang2020/Exbind btt

https://huggingface.co/

datasets/Ziqianwwww/ExBind, respectively.

## 1 Introduction

Iterative visual-feedback coding and editing systems repeatedly move between a rendered artifact and the program that controls it. A typical loop is:

code → render → inspect visual feedback → identify a referent → resolve its executable object → patch → render again.

Suppose a user asks the system to “move the right output box slightly down.” The visible referent may be clear, but executing the request requires identifying which DOM node, SVG child, component, or code-controlled object denotes that box. If the correspondence is wrong, a sensible patch can be applied to the wrong object; repeated revisions may then compound the discrepancy.

This motivates the central question of ExBind:

Where does executable binding first fail?

Here “first” means the first violated structural constraint under a declared evaluator decomposition, not the first step in a model’s hidden reasoning. ExBind isolates the middle layer in the following decomposition:

semantic/visual localization → structured executable correspondence → action or code execution.

An end-to-end editing outcome conflates these layers. A failed patch may reflect incorrect localization, an incorrect referent-to-program correspondence, faulty patch generation, execution failure, or accumulated errors from earlier turns. ExBind therefore does not benchmark the full multi-turn loop; it makes the correspondence layer measurable in isolation.

A final executable failure is an observation, not a diagnosis. Consider the instruction “select the status cell for the project owned by Team B.” The correct cell requires resolving the project row, the requested attribute column, and their intersection.

Selecting the owner cell from the correct row is a valid executable output, yet it differs fundamentally from returning an unknown identifier or a cell from the wrong project. Analogous distinctions arise when a model identifies the correct visual object but the wrong executable child, chooses the correct graph nodes but reverses source and target roles, or selects a valid DOM element from the wrong hierarchy. Thus candidate validity, binding correctness, and failure diagnosis are distinct quantities:

validity ̸= binding correctness ̸= failure diagnosis.

Existing grounding evaluations usually stop at a region, coordinate, element, or action (Kamath et al., 2021; Li et al., 2022; Liu et al., 2023; Cheng et al., 2024; Zheng et al., 2024; You et al., 2024; Lu et al., 2024). They measure semantic localization, an essential prerequisite, but do not generally make the subsequent referent-to-executable structural correspondence the primary diagnostic object. Diagnostic visual reasoning benchmarks similarly show the value of structured annotations for separating conflated errors (Johnson et al., 2017; Hudson and Manning, 2019). In executable environments, the relevant constraints are often directly available: entities have attributes and parents, graph endpoints have ordered roles, DOM elements lie in a hierarchy, and table cells are row–column intersections.

We introduce ExBind, a controlled diagnostic instrument for structured executable correspondence. Each instance begins with a representationindependent latent binding object containing entities, relations, hierarchy, attributes, roles, and a canonical target. Format-specific compilers materialize the same binding object as SVG, DOM, canvas, tree, graph, or table observations with surfacespecific executable references. Visual correspondence cases instantiate the motivating rendered setting more directly; controlled structural cases use serialized or symbolic surfaces to isolate hierarchy, role, row/column, and executable-identity constraints. The model still returns only the reference required by the environment. The evaluator deterministically maps the prediction back into the latent binding space, producing a full set of violated applicable constraints and, where the family has a declared structural order, a concise first-violation summary. This construction supports comparison across surfaces without requiring their surface identifiers to match.

The primary release contains a 250-case broad suite, a disjoint 240-case targeted structural suite, and 50 paired latent groups. Its controlled size follows from the need for known latent targets, unique executable gold, deterministic mappings, structuraloracle recovery, and shortcut checks. The goal is diagnostic identifiability and measurement reliability, not population-scale ranking. The inherited 5.9K SVG/DOM inventory is retained as provenance and a cached-output bridge, not pooled into the primary evaluation.

Frozen-model evaluation illustrates the diagnostic use case. On the broad suite, Qwen2.5-VL-3B returns a valid candidate on 98.4% of cases but is exact on 76.4%; Qwen3-VL-4B reaches 98.8% exact accuracy, leaving only three valid-but-wrong residual cases under the same vocabulary. In the targeted table suite, Qwen2.5-VL-3B is valid on all 120 cases yet exact on 63, and every residual error is localized to the column-selection constraint after row resolution. Candidate-order perturbations change error frequency and case identity, but all residual errors preserve this correctrow/wrong-column form. Thus outcome stability and diagnostic-type stability are distinct measurement properties. The hierarchy-depth control remains at ceiling for both models, delimiting rather than expanding the benchmark claim.

We make three contributions:

1. Executable correspondence as an isolated diagnostic target. We identify and formalize the referent-to-executable correspondence layer between semantic/visual localization and action or code execution in visualfeedback editing workflows.

2. Controlled latent-to-surface construction. We introduce ExBind, which compiles representation-independent latent binding instances into heterogeneous executable surfaces while retaining deterministic mappings needed to diagnose valid-but-wrong predictions.

3. Residual structure and interface sensitivity. We show that candidate validity can mask executable mismatches, that residual profiles differ across frozen checkpoints, and that interface serialization can change case-level outcomes while preserving the localized structural failure form.

![](images/b13d19542f9e299b7f1ddd15e84e164ba7e7d604b3782ca0cf89ad5fc135dc92.jpg)  
Figure 1: ExBind isolates visual-to-executable correspondence from a larger visual-feedback editing loop. A shared latent target is compiled into surface-specific observations and candidate inventories; the model emits only a strict reference; and the evaluator maps that reference back to the latent target and checks deterministic constraints. The running example is an actual targeted table case: $\mathtt { c e l l \_ r o l \_ c o o }$ is a valid candidate in the correct row, yet it selects the owner rather than the status column. The first-violation summary is evaluator-defined rather than a mode reasoning trace.

## 2 Related Work

ExBind is positioned between three layers that are often evaluated together but are scientifically distinct:

semantic localization → structured executable binding → action execution.

The benchmark evaluates the middle layer: after a semantic or visual referent is identified, which environment-specific executable reference denotes it? Its diagnostic labels are mechanically recoverable from executable environment structure and the latent target; they are neither model-generated reasoning traces nor a process-supervision signal.

Referring and visual grounding. Referringexpression comprehension and phrase grounding align natural language with image regions or objects. MDETR jointly models text and images for text-conditioned object detection and referringexpression tasks, while GLIP and Grounding DINO extend grounding toward large-scale and openvocabulary recognition (Kamath et al., 2021; Li et al., 2022; Liu et al., 2023). Multimodal LLMs such as Ferret further support referring and grounding over points, boxes, and free-form regions (You et al., 2024). These methods address a prerequisite of ExBind—identifying the intended visual referent—but their typical target remains a region or object representation. ExBind studies the subsequent structured resolution from an already specified or recoverable referent to an environmentspecific executable reference, where hierarchy, ordered roles, or symbolic identity may matter even after the relevant visual entity has been recognized. The distinction is therefore not that prior grounding work lacks structure, but that referent-to-executable correspondence is not generally its primary diagnostic object.

GUI grounding and computer-use agents. Grounding has become a central component of multimodal computer-use agents. SeeClick introduces ScreenSpot and shows that improved GUI grounding is associated with improved downstream agent performance, while SeeAct demonstrates a large gap between strong multimodal planning and automatic grounding on web interfaces (Cheng et al., 2024; Zheng et al., 2024). Ferret-UI, ScreenAI, OmniParser, and subsequent GUI-specific models improve screen understanding, interactableelement parsing, or action grounding (You et al., 2024; Baechler et al., 2024; Lu et al., 2024). More recent benchmarks increase the realism and diversity of grounding evaluation: ScreenSpot-Pro targets high-resolution professional software, WinSpot focuses on Windows applications, and UI-Vision evaluates element grounding, layout grounding, and action prediction in desktop environments (Li et al., 2025; Hui et al., 2025; Nayak et al., 2025). MMBench-GUI further organizes GUI-agent evaluation hierarchically from content understanding and element grounding to task automation and collaboration (Wang et al., 2025). ExBind is complementary to this progression: rather than extending the environment or agent horizon, it isolates the binding decision itself and decomposes it into verifiable structural constraints.

Diagnostic and compositional evaluation. ExBind also follows the tradition of diagnostic benchmarks designed to separate error sources that aggregate task metrics conflate. CLEVR introduces controlled visual scenes and detailed annotations to diagnose compositional visual reasoning, explicitly motivated by the difficulty of identifying which capability causes a VQA failure (Johnson et al., 2017). GQA moves this idea to real images using scene graphs and functional programs, and supplements answer accuracy with consistency, grounding, and plausibility analyses (Hudson and Manning, 2019). These works diagnose visual reasoning requirements underlying question answering. ExBind applies the same general principle to executable reference resolution, but its diagnostic annotations are derived from executable binding structure rather than from functional programs or model-generated reasoning. This distinction separates ExBind from process supervision and chain-of-thought verification: the evaluated model emits only its final executable reference, and all intermediate labels are deterministic benchmark annotations (Lightman et al., 2023; Wei et al., 2022).

Structured visual and semi-structured artifacts. Web pages, user interfaces, diagrams, and tables couple visual layout with an underlying symbolic structure. WebSRC evaluates structural reading comprehension using webpages with screenshots, HTML, and metadata, while Pix2Struct exploits the alignment between webpage screenshots and HTML as a general pretraining signal for visually situated language (Chen et al., 2021; Lee et al., 2023). ScreenAI similarly studies UI and infographic understanding through structured screen annotation, and ChartQA evaluates visual and logical reasoning over charts and their underlying data (Baechler et al., 2024; Masry et al., 2022). ExBind uses this dual surface/structure property for a different purpose: a shared latent binding vocabulary is instantiated across heterogeneous representations and used to score where a prediction ceases to match the executable structure.

Table 1 summarizes the distinction at the benchmark level. A positive entry means that the benchmark makes the property an explicit evaluation object, rather than merely allowing a model to consume structured input.

## 3 Problem Formulation

We represent one latent binding instance as

$$
G = ( V , E , A , H , R , Y ) ,\tag{1}
$$

where $V$ is a set of entities, $E$ contains structural or graph edges, A stores attributes, H stores parentchild structure, R stores semantic or spatial relations, and $Y$ is the target binding specification. For the table example, $V$ contains project rows, A contains owner/status/priority values, and $Y$ specifies the status attribute of the row whose owner is Team B.

A representation compiler $K _ { f }$ materializes this latent object into a format-specific case:

$$
X _ { f } = K _ { f } ( G ) = ( O _ { f } , I , C _ { f } , Y _ { f } , M _ { f } ) .\tag{2}
$$

Here $O _ { f }$ is the observation, I is the instruction, $C _ { f }$ is the executable candidate set, $Y _ { f }$ is the surface gold reference, and $M _ { f }$ maps latent objects to surface references. The table compiler may map the same latent target to ce $1 1 _ { - } \mathtt { r } 0 1 _ { - } \mathtt { c } 0 1 ;$ a graph compiler may instead expose an ordered endpoint pair; a DOM compiler may expose a dom\_id.

The model prediction $\hat { y } _ { f }$ is a strict structured reference, $\mathrm { e . g . } \qquad \{ \mathrm { t a r g e t \_ i d : } \qquad \cdot \ \cdot \ \cdot \ \}$ {source\_id: ..., target\_id: $\dots \}$ , {dom\_id: ...}, or $\{ \mathsf { c e l l }  _ { - } \mathrm { i d } :$ $\dots \}$ ExBind maps $\hat { y } _ { f }$ through $M _ { f } ^ { - 1 }$ when possible and computes a deterministic diagnostic score vector $s ( \hat { y } _ { f } , G )$ together with the full violated-constraint set

$$
V ( \hat { y } _ { f } , G ) = \{ d \in D _ { f } : s _ { d } ( \hat { y } _ { f } , G ) = 0 \} ,\tag{3}
$$

where $D _ { f }$ is the set of applicable structural constraints for format $f .$ The model is never asked for chain-of-thought; intermediate labels are benchmark annotations derived from the latent record and candidate inventory. For task families with a declared structural order, we additionally report the first member of V under that order as a compact diagnostic summary. This ordering is an evaluatordefined decomposition of the executable target. It does not imply that the model internally performs these operations sequentially, and it is not a causal account of model computation.

Table 1: Positioning against representative grounding, diagnostic reasoning, GUI, and structured-artifact benchmarks. ExBind’s distinguishing combination is a surface-specific executable target, evaluator-defined structural labels, and paired latent cases across formats.
<table><tr><td>Benchmark family</td><td>Executable target</td><td>Structured latent/gold</td><td>Multiple formats</td><td>Paired cross-format cases</td><td>Stage-localized errors</td></tr><tr><td>RefCOCO / phrase grounding (Yu et al., 2016; Kamath et al., 2021)</td><td>no</td><td>partial</td><td>no</td><td>no</td><td>no</td></tr><tr><td>CLEVR / GQA (Johnson et al., 2017; Hudson and Manning, 2019)</td><td>no</td><td>yes</td><td>no</td><td>no</td><td>yes</td></tr><tr><td>GUI grounding (Cheng et al., 2024; Li et al., 2025; Nayak et al., 2025)</td><td>partial</td><td>partial</td><td>partial</td><td>no</td><td>no</td></tr><tr><td>WebSRC / Pix2Struct (Chen et al., 2021; Lee et al., 2023)</td><td>no</td><td>yes</td><td>screenshot+source</td><td>no</td><td>no</td></tr><tr><td>ChartQA (Masry et al., 2022)</td><td>no</td><td>partial</td><td>chart formats</td><td>no</td><td>no</td></tr><tr><td>ExBind-v2</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td></tr></table>

For the running table example, the ordered stages are:

row/entity → attribute column → cell intersection → executable reference.

Other families use different applicable stages. A graph endpoint task uses identity, relation, ordered role, and exact endpoint-pair reference. A tree/DOM task uses identity, hierarchy, role, and exact node reference. Not-applicable dimensions are reported with null denominators rather than counted as failures.

Benchmark unit and evaluator boundary. The unit of evaluation is one compiled case $\begin{array} { r l } { x _ { f } } & { { } = } \end{array}$ $( O _ { f } , I , C _ { f } , Y _ { f } , M _ { f } )$ together with its latent record G. The model receives the observation, instruction, and candidate inventory and returns one surface executable reference; it is not asked to expose intermediate reasoning. The evaluator then applies the fixed mapping $M _ { f } ^ { - 1 }$ and checks the applicable structural constraints. Thus a stage-localized structural mismatch is a property of the pair $( \hat { y } _ { f } , G )$ under the benchmark decomposition, not a claim about the order or content of hidden model computations. Cases are scored independently, and cross-format consistency is computed only within explicitly paired latent groups.

## 4 Benchmark Construction and Validation

## 4.1 Benchmark Composition

ExBind-v2 separates the primary evaluation protocol from inherited diagnostic resources. The primary protocol consists of a 250-case broad diagnostic suite and a 240-case targeted structural diagnostic. These suites are deliberately controlled samples: their purpose is to make correspondence constraints identifiable and to support reliable stagelevel comparison under fixed interfaces. The inherited 5.9K ExBind-Bench SVG/DOM inventory is preserved as a legacy resource for provenance, historical characterization, and cached-output bridge analysis; it is not pooled into the primary evaluation split. Table 2 summarizes this paper-facing composition.

The broad suite instantiates canvas, graph, table, tree, $\mathbf { S V G }$ , and HTML-DOM cases. The targeted structural diagnostic is a separate 240-case split rather than an additional model-independent annotation of the broad split. Table 4 gives the paperfacing composition by representation. The table cases are the main stage-localization case study because they expose valid executable outputs that preserve the target row while selecting the wrong attribute column. The legacy bridge maps eligible cached EVG/SEB outputs into the same binding taxonomy without rerunning inference.

## 4.2 Latent-to-Surface Construction

Each generated case starts from a latent binding instance rather than from a surface format. Relational latents are compiled into canvas and SVG observations; hierarchy latents are compiled into tree and HTML-DOM observations; graph latents expose ordered endpoint candidates; and table latents expose row–column cell candidates. Rendered SVG and canvas cases instantiate visual correspondence more directly, whereas serialized DOM, tree, graph, and table cases hold structural variables explicit so that individual correspondence constraints can be identified. The compiler produces the observation, instruction, candidate inventory, surface gold, and latent-to-surface mapping. Appendix A gives the complete generation algorithm, record schema, prompt contract, and release commands.

Table 2: Paper-facing benchmark components. Primary claims are based on the new ExBind-v2 core evaluation; legacy resources support provenance and bridge analysis.
<table><tr><td>Component</td><td>N</td><td>New?</td><td>Model inference?</td><td>Primary claim?</td></tr><tr><td>Broad diagnostic suite</td><td>250</td><td>yes</td><td>yes</td><td>yes</td></tr><tr><td>Targeted table diagnostic</td><td>120</td><td>yes</td><td>yes</td><td>yes</td></tr><tr><td>Targeted hierarchy diagnostic</td><td>120</td><td>yes</td><td>yes</td><td>supporting</td></tr><tr><td>Cross-format paired groups</td><td>50</td><td>yes</td><td>yes</td><td>supporting</td></tr><tr><td>LLaVA-Phi-3-mini supporting pilot</td><td>250+120</td><td>no</td><td>yes</td><td>supporting</td></tr><tr><td>Legacy bridge subset</td><td>300</td><td>no</td><td>cached</td><td>supporting</td></tr><tr><td>Inherited ExBind-Bench inventory</td><td>5,900</td><td>no</td><td>historical</td><td>no</td></tr></table>

Table 3: Supporting non-Qwen evaluation under the same ordinary strict-reference protocol. These results are reported as a cross-family diagnostic profile, not pooled into the primary Qwen comparison.
<table><tr><td>Split</td><td>N</td><td>Valid</td><td>Exact</td></tr><tr><td>Broad diagnostic</td><td>250</td><td>220/250 (88.0)</td><td>154/250 (61.6)</td></tr><tr><td>Targeted table</td><td>120</td><td>67/120 (55.8)</td><td>46/120 (38.3)</td></tr></table>

## 4.3 Representation Coverage

Table 5 is paper-facing rather than an implementation status matrix. A checkmark means that the representation instantiates that dimension in evaluated cases; N/A means the dimension is not semantically part of that task family. Figure 1 shows the evaluation flow from latent instance to surface prediction and diagnostic scoring.

## 4.4 Construct Validity

Structured stage labels are useful only if they are mechanically grounded. ExBind therefore applies deterministic gates before model evaluation. The broad diagnostic suite and the targeted structural diagnostics require unique gold targets, non-empty candidate sets, deterministic latent-to-surface mappings, and a structural oracle that recovers the gold target from benchmark metadata. Table 6 summarizes the targeted diagnostic gates.

## 4.5 Shortcut Validity

Shortcut checks are part of the benchmark acceptance criteria rather than post-hoc analysis. Target identifiers must not appear in instructions, candidate order must vary across accepted cases, lexicalonly and type-only baselines must fail on the targeted diagnostic, and the structural oracle must remain exact. These controls do not prove that every shortcut is impossible; they define the minimum validity boundary under which structural diagnosis is interpretable.

## 4.6 Measurement Reliability and Cross-Representation Validity

Candidate serialization is an explicit protocol variable. For primary results, released records fix one candidate order; the matched permutation control treats that order as a nuisance variable of the executable interface because it changes access to an otherwise identical candidate set without changing the latent target. We therefore separate case-level outcome stability from diagnostic-type stability, and report candidate-order sensitivity as a measurement result rather than as a binding capability or causal explanation.

Cross-format groups provide a complementary validity check. Their shared latent target and surface mappings permit both semantic-consistency scoring and paired diagnostic transitions. A paired comparison can therefore distinguish a structural mismatch that persists across surfaces from one that appears only after surface compilation. The analysis is deliberately restricted to explicitly compatible compiler pairs; it does not assume that every latent instance can be rendered faithfully in every format.

Finally, we assess whether the controlled taxonomy can describe a small fixed sample of existing executable-binding errors. We manually adjudicate 20 cached errors drawn from EVG (8), SVGEditBench-derived resources (10), and Mini-WoB DOM cases (2) against the stored instruction, candidate/gold record, and prediction. All 20 are covered by the existing taxonomy: 16 are wrong-identity errors, two are wrong-role/order errors, one is a wrong-relation error, and one is a wrong-type error. Five multi-edit source instructions are ambiguous at the task-unit level, but none requires a new taxonomy category. This is a sampled coverage check rather than an estimate of error prevalence in real agents; Appendix A.15 gives the annotation protocol and source breakdown.

Table 4: Composition of the primary ExBind-v2 release. Broad and targeted cases are disjoint evaluation splits. Cross-format groups contain 25 SVG–canvas pairs and 25 tree–DOM pairs; they are used for consistency analysis rather than pooled exact accuracy.
<table><tr><td>Split</td><td>SVG</td><td>Canvas</td><td>Tree</td><td>DOM</td><td>Graph</td><td>Table</td><td>Total</td></tr><tr><td>Broad diagnostic</td><td>25</td><td>50</td><td>50</td><td>25</td><td>50</td><td>50</td><td>250</td></tr><tr><td>Targeted structural</td><td></td><td></td><td>120</td><td>一</td><td>一</td><td>120</td><td>240</td></tr><tr><td>Cross-format groups</td><td>25</td><td>25</td><td>25</td><td>25</td><td></td><td>一</td><td>50 groups</td></tr></table>

Table 5: Representation families and their verified binding coverage in the primary release. A checkmark denotes an applicable scored dimension; N/A denotes that the dimension is not part of that family.
<table><tr><td>Repr.</td><td>Identity</td><td>Attribute</td><td>Relation</td><td>Hierarchy</td><td>Ordered role</td><td>Structured selection</td></tr><tr><td>SVG</td><td>yes</td><td>yes</td><td>yes</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>DOM</td><td>yes</td><td>N/A</td><td>N/A</td><td>yes</td><td>partial</td><td>N/A</td></tr><tr><td>Canvas2D</td><td>yes</td><td>yes</td><td>yes</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Tree</td><td>yes</td><td>N/A</td><td>N/A</td><td>yes</td><td>yes</td><td>N/A</td></tr><tr><td>Graph</td><td>yes</td><td>N/A</td><td>yes</td><td>partial</td><td>yes</td><td>N/A</td></tr><tr><td>Table</td><td>yes</td><td>yes</td><td>N/A</td><td>partial</td><td>N/A</td><td>yes</td></tr></table>

## 5 Evaluation Protocol

We evaluate two primary frozen VLMs with the same strict reference-output contract: Qwen2.5- VL-3B and Qwen3-VL-4B. We additionally run LLaVA-Phi-3-mini as a supporting cross-family evaluation under the identical ordinary contract; it is not pooled into the primary Qwen comparison. The broad diagnostic suite contains 250 records and 50 cross-format latent groups. The targeted structural diagnostic contains 240 records: 120 hierarchy cases and 120 table cases. These are separate controlled diagnostic splits and are not pooled into a single leaderboard score. Crossformat groups are scored by canonical semantic consistency, while the targeted split is used for controlled failure diagnosis. The table diagnostic is reported by structural condition rather than as a calibrated scalar axis: compact single-condition tables, conjunctive-condition tables, and high-candidate duplicate-value tables. Candidate order is a fixed part of the released primary interface and a nuisance variable only in the matched permutation control.

Metrics are exact executable-reference accuracy, candidate validity, dimension accuracy, the full violated-constraint set, task-specific survival, first violated diagnostic constraint under the declared family order, and cross-format semantic consistency. Candidate validity asks whether every returned identifier belongs to the candidate inventory. Exact accuracy asks whether the executable surface reference exactly matches gold. Survival S(k) is computed only for meaningful evaluator-defined stage orders within a task family. All proportions retain their case denominator; dimensions that do not apply are excluded rather than treated as incorrect.

We organize the empirical section around five scientific questions:

1. Does validity hide executable-reference binding errors?

2. Which diagnostic constraint is first violated when exact binding fails?

3. Do residual failures differ across frozen VLMs?

4. What structural mismatch pattern is exposed by the targeted table diagnostic?

5. How reliable are diagnostic profiles under interface and cross-format controls?

## 6 Results

## 6.1 RQ1: Validity Is Not Binding Correctness

Table 7 shows that output validity and binding correctness measure different phenomena. Qwen2.5- VL-3B returns valid candidate references for 246/250 broad-suite cases (98.4%) but is exact for only 191/250 (76.4%). Most failures are therefore not malformed outputs. Qwen3-VL-4B reaches 250/250 validity and 247/250 exact accuracy; the remaining three errors are still valid candidate selections. The distinction is most informative when output formatting is mostly solved but binding remains imperfect.

Table 6: Integrity gates for the 240-case targeted structural diagnostic.
<table><tr><td>Gate</td><td>Result</td><td>Interpretation</td></tr><tr><td>Unique gold target</td><td>240/240</td><td>no ambiguous gold cases</td></tr><tr><td>Structural oracle</td><td>1.000</td><td>latent metadata determines the gold</td></tr><tr><td>Target-ID leakage</td><td>0.000</td><td>instruction/candidates do not reveal target ID</td></tr><tr><td>Lexical-only baseline</td><td>0.000</td><td>token overlap alone does not solve cases</td></tr><tr><td>Type-only baseline</td><td>0.000</td><td>candidate type alone does not solve cases</td></tr><tr><td>Random-candidate expectation</td><td>0.167</td><td>mean chance accuracy over the 240 cases</td></tr><tr><td>Gold-rank variation</td><td>passed</td><td>candidate order is not a fixed gold-position proxy</td></tr><tr><td>Accepted records</td><td>240/240</td><td>no rejected diagnostic records</td></tr></table>

Table 7: Broad diagnostic suite: validity and exact binding accuracy. Wilson 95% intervals are shown for exact accuracy.
<table><tr><td>Model</td><td>Cases</td><td>Valid</td><td>Exact</td><td>Exact 95% CI</td><td>Errors after validity</td></tr><tr><td>Qwen2.5-VL-3B</td><td>250</td><td>0.984</td><td>0.764</td><td>[0.708, 0.812]</td><td>55</td></tr><tr><td>Qwen3-VL-4B</td><td>250</td><td>1.000</td><td>0.988</td><td>[0.965, 0.996]</td><td>3</td></tr></table>

## 6.2 RQ2: ExBind Localizes Evaluator-Defined Structural Mismatches

Dimension-level scores expose the structure hidden behind exact accuracy. Qwen2.5-VL-3B is perfect on canvas/SVG relational object cases in the broad suite, but has lower hierarchy scores on tree/DOM cases and low table exactness. Figure 2 shows the first violated diagnostic constraint for each broadsuite prediction under the declared family order; the released score files retain the complete violatedconstraint set for the same prediction. Qwen3-VL-4B has fewer residual violations, but the figure is not a general model-capacity claim: it is a profile of where each frozen checkpoint still violates the evaluator-defined binding constraints.

![](images/e91658af17355ce688f150037495c906d3b342aca88a51321a35c7eb8d5a1c0e.jpg)  
Figure 2: Declared-order first violated diagnostic constraint on the broad diagnostic suite. Counts are computed by mapping each final executable prediction back to the latent target and checking evaluator-defined structural constraints; the released artifacts also retain the full violated-constraint set.

Task-specific survival gives a more local view.

On graph cases, Qwen2.5-VL-3B satisfies identity/relation/role/exact constraints for 40/50 cases; four additional cases are invalid or unresolved. On tree and HTML-DOM hierarchy cases, the declared first violation is identity/group selection, after which surviving cases remain correct through hierarchy, role, and exact stages. On broad table cases, 23/50 cases satisfy entity selection and then remain exact. The ordered summary is useful for compact reporting, while the accompanying violated-constraint set prevents it from being read as a temporal model trace.

## 6.3 RQ3: Failure Profiles Differ Across Frozen VLMs

The Qwen3 result is not just a higher number in a leaderboard. The residual errors change character. The broad suite contains only three Qwen3 errors: one canvas relational case in which the model selected a same-attribute distractor instead of the object above the anchor, and two HTML-DOM hierarchy cases in which it chose the output node from the wrong group while preserving the output role. All three are valid candidate outputs. This profile differs from Qwen2.5-VL-3B, whose errors are concentrated in hierarchy/group selection, graph identity or invalid endpoints, and table entity/column binding. Qwen2.5-VL-3B is therefore a useful diagnostic operating point: validity is already high, but the residual population is large enough to expose structural patterns. Qwen3-VL-

Table 8: Primary Qwen results by representation. Values are valid/exact counts with percentages in parentheses. The first six rows partition the 250-case broad suite; the final two rows are the disjoint targeted structural split.
<table><tr><td rowspan="2">Representation / split</td><td rowspan="2">N</td><td colspan="2">Qwen2.5-VL-3B</td><td colspan="2">Qwen3-VL-4B</td></tr><tr><td>Valid</td><td>Exact</td><td>Valid</td><td>Exact</td></tr><tr><td>Canvas2D</td><td>50</td><td>50/50 (100.0)</td><td>50/50 (100.0)</td><td>50/50 (100.0)</td><td>49/50 (98.0)</td></tr><tr><td>SVG</td><td>25</td><td>25/25 (100.0)</td><td>25/25 (100.0)</td><td>25/25 (100.0)</td><td>25/25 (100.0)</td></tr><tr><td>Tree</td><td>50</td><td>50/50 (100.0)</td><td>35/50 (70.0)</td><td>50/50 (100.0)</td><td>50/50 (100.0)</td></tr><tr><td>HTML-DOM</td><td>25</td><td>25/25 (100.0)</td><td>18/25 (72.0)</td><td>25/25 (100.0)</td><td>23/25 (92.0)</td></tr><tr><td>Graph</td><td>50</td><td>46/50 (92.0)</td><td>40/50 (80.0)</td><td>50/50 (100.0)</td><td>50/50 (100.0)</td></tr><tr><td>Table</td><td>50</td><td>50/50 (100.0)</td><td>23/50 (46.0)</td><td>50/50 (100.0)</td><td>50/50 (100.0)</td></tr><tr><td>Targeted table</td><td>120</td><td>120/120 (100.0)</td><td>63/120 (52.5)</td><td>120/120 (100.0)</td><td>120/120 (100.0)</td></tr><tr><td>Targeted hierarchy</td><td>120</td><td>120/120 (100.0)</td><td>120/120 (100.0)</td><td>120/120 (100.0)</td><td>120/120 (100.0)</td></tr></table>

4B supplies a near-saturated contrast in which the same diagnostic vocabulary remains defined but yields few residual cases. ExBind makes this comparison possible because both models are scored through the same latent-stage vocabulary.

The supporting LLaVA-Phi-3-mini run provides a deliberately non-saturated cross-family profile under the same interface. It is valid on 220/250 broad cases and exact on 154/250; on the targeted table split it is valid on 67/120 and exact on 46/120. Among its table errors, 53 are invalid or unresolved and 21 are valid correct-row/wrong-column selections, classified as wrong-attribute errors under the table-specific diagnostic order. This matches the structural form of the Qwen2.5-VL-3B table profile, but with substantially lower validity and a different incidence of invalid outputs. We therefore use it as supporting evidence about cross-family validity and residual-profile coverage rather than as a matched conditional accuracy comparison. Capability-level opportunities and corresponding first-violation rates are reported with explicit applicability denominators in Appendix A; they are not pooled into a visual comparison because the dimensions apply to different representation subsets.

## 6.4 RQ4: Table Binding Localizes a Consistent Column-Misbinding Pattern

The strongest diagnostic case study is the targeted table split. Qwen2.5-VL-3B is valid on every table case but exact on only 63/120 (52.5%). For every table error, the first violated diagnostic constraint under the declared row–column–cell order is table\_column: row resolution succeeds, but the selected cell is in the wrong attribute column. The full task-specific violation set also records the failed cell intersection. Qwen3-VL-4B is exact on all 120 table cases.

The structural conditions should not be read as a monotonic axis. They vary jointly in rows, candidates, constraints, source length, duplicate values, and gold-rank distribution. Qwen2.5-VL-3B exact accuracy is 0.275 on compact single-condition tables, 0.825 on conjunctive-condition tables, and 0.475 on high-candidate duplicate-value tables. A confound audit shows candidate-position effects: accuracy is 0.857 when the gold candidate is rank 1, 0.222 at rank 9, and 0.000 for sparse ranks 13 and above. Target column and template family are balanced within each condition, so the anomaly is not explained by all surface factors.

A fixed 20-case adjudication pack supports the stage interpretation. It includes all three table conditions, 14 Qwen2.5-VL-3B failures, and 6 correct controls. All 14 failures are valid candidate selections in the correct row but wrong column. The stage decomposition identifies the semantic form of the error as column misbinding. The confound audit shows that the frequency of that error is also sensitive to interface factors such as candidate position. This is a localized executable mismatch pattern, not evidence for a unique internal model mechanism.

Candidate order is a nuisance variable of the executable interface, not a latent binding capability: it changes the access path to an unchanged candidate set while leaving the observation, instruction, latent target, and gold reference fixed. We therefore interpret order sensitivity as interface sensitivity and report it as a measurement result. Stability of the first violated constraint under this nuisance supports the diagnostic error-type description, but does not identify the causal source of column binding.

A matched interface ablation adds only explicit per-candidate column metadata to the same 120 cases, preserving observation, instruction, candidate order, gold, and case IDs. Exact accuracy changes from 63/120 (0.525) to 71/120 (0.592), with 15 paired improvements and 7 regressions (Table 9). This provides evidence that the observed column-binding failures are sensitive to candidateinterface specification; it is not evidence of a reliable accuracy improvement or a unique causal mechanism.

We therefore ran one additional candidate-order control on the same 120 table cases. For each case we produced three deterministic permutations that preserve the latent record, observation, instruction, candidate set, and gold target; only candidate order changes. Qwen2.5-VL-3B exact accuracy is 77/120, 71/120, and 68/120 under the three permutations, compared with 63/120 under the original order. The paired mean delta over permutations is +7.5 percentage points with a 95% paired bootstrap interval of [-1.1, 16.1]. The bootstrap unit is the 120 parent case IDs, with three repeated measurements per case; the 360 predictions are not treated as independent cases. This interval crosses zero, so the permutations are not interpreted as a reliable accuracy improvement. Figure 3 makes the measurement result explicit: 76/120 cases are order-sensitive, whereas every residual error under every condition remains a valid correct-row/wrongcolumn selection. Candidate order is therefore a nuisance variable: it changes failure incidence and case identity while preserving the localized error form. Outcome stability is not diagnostic-type stability, and neither observation establishes a causal mechanism for column misbinding.

## 6.5 RQ5: Reliability and Cross-Representation Checks

The hierarchy-depth diagnostic varies repeatedmodule depth from 2 to 4 while preserving deterministic hierarchy and role labels. Both evaluated models are exact on all 120 hierarchy diagnostic cases. This negative result is useful: depth alone is insufficient to challenge these models under this controlled construction, and ExBind does not declare every structural manipulation difficult.

Cross-format results are supporting evidence for the latent abstraction, not a universal representation-invariance claim. SVG–canvas relational pairs are fully stable for both models. Tree– DOM pairs show limited sensitivity: Qwen2.5-VL-3B has 0.80 semantic consistency over 25 paired groups, while Qwen3-VL-4B has 0.92. The paired transition analysis in Table 13 makes this result more specific. For Qwen2.5-VL-3B, five hierarchy mismatches persist across both tree and DOM surfaces, while five other groups change status across surfaces. Qwen3-VL-4B has two tree-correct to

DOM-identity transitions. These paired outcomes show that some mismatches are latent-structure stable while others are surface-sensitive.

Table 10: Cross-format semantic consistency on explicitly paired latent groups. Surface identifiers are not required to match.
<table><tr><td>Pair</td><td>Groups</td><td>Qwen2.5</td><td>Qwen3</td></tr><tr><td>SVG-Canvas2D</td><td>25</td><td>1.00</td><td>1.00</td></tr><tr><td>Tree-HTML-DOM</td><td>25</td><td>0.80</td><td>0.92</td></tr></table>

The ordinary-contract LLaVA-1.5-7B-HF pilot is retained in Appendix A.13 as a protocolboundary result: its 7/120 targeted-table validity is too low for a useful conditional profile. LLaVA-Phi-3-mini improves broad and table coverage enough to expose a non-Qwen residual profile, but its 55.8% table validity still prevents a primary matched comparison. The two Qwen checkpoints therefore remain the primary model comparison.

The legacy bridge tests whether the taxonomy applies beyond newly generated cases. A fixed 300- row eligible EVG/SEB subset maps established executable SVG binding outputs into the ExBind stage vocabulary without rerunning inference. It spans object, endpoint, group, and edge/rejection tasks and contains 152 no-failure rows and 148 identity-stage failures. Appendix A.14 provides the corresponding benchmark-only context from the earlier full paper: ambiguity, perturbation, external-transfer, and task-conditioned failure analyses. This supports the narrow claim that the failure vocabulary can be applied to existing executable binding data; it is not a broad transfer result.

## 7 Limitations

ExBind isolates the referent-to-executable correspondence layer and therefore does not measure error accumulation over a complete multi-turn coding trajectory, patch generation, or execution. The primary model comparison uses two related Qwen-family checkpoints; LLaVA-Phi-3-mini provides supporting non-Qwen evidence, but its 55.8% targeted-table validity limits matched conditional comparison, while the older LLaVA pilot remains a protocol-boundary result. The controlled suite is designed for identifiable structural diagnosis rather than population-scale ranking. Rendered SVG and canvas cases address the visual setting more directly, while serialized DOM, tree, graph, and table cases expose structural factors that would otherwise be difficult to attribute; the set is therefore not a uniform rendered-image evaluation. Cross-format pairing is limited to compatible compiler pairs and 50 latent groups, and the existing-task check is a fixed 20-error sample rather than an estimate of real-agent prevalence. Candidate serialization is part of the interface and can change case outcomes, so comparisons require the released order or an explicitly matched alternative. Finally, the first violated constraint is an ordered evaluator summary; full violated-constraint sets are released alongside it, and neither is a model reasoning trace or causal mechanism.

![](images/0e5673acc2eb5f1b4c74de16e8d3c8b270ee17b8d311b8709cd19137ea2e705f.jpg)

![](images/b30c9ed55dd728f5a09a6be02bbb0047734012969e1be3432c19548e3cdc1074.jpg)  
Figure 3: Candidate-order reliability control on the same 120 targeted table cases. Left: exact accuracy varies across the released order and three deterministic candidate permutations; points show exact accuracy and error bars show Wilson 95% intervals. Right: 76 cases change exact outcome across orders, while 31 remain correct and 13 remain wrong under all four interfaces. For each order, every residual error is a valid candidate selection from the correct row and wrong attribute column. Thus candidate order changes outcome incidence, whereas the evaluator-localized error form is stable.

Table 9: Targeted table diagnostic and matched column-metadata ablation. Exact intervals are Wilson 95% intervals; the ablation delta interval is paired bootstrap.
<table><tr><td>Condition</td><td>Cases</td><td>Validity</td><td>Exact</td><td>Interval</td><td>Notes</td></tr><tr><td>Original table interface</td><td>120</td><td>1.000</td><td>63/120</td><td>[0.436, 0.612]</td><td>all errors first violate column</td></tr><tr><td>Explicit column metadata</td><td>120</td><td>1.000</td><td>71/120</td><td>[0.502, 0.675]</td><td>15 improve, 7 regress</td></tr><tr><td>Paired delta</td><td>120</td><td></td><td>+8/120</td><td>[-0.008, 0.142]</td><td>paired bootstrap</td></tr><tr><td>20-case adjudication</td><td>20</td><td>1.000</td><td>6/20</td><td></td><td>14 wrong-column errors</td></tr></table>

## 8 Conclusion

ExBind isolates visual/semantic-to-executable correspondence as a structured diagnostic object rather than a single exact-match bit. By separating latent targets, surface references, and deterministic structural constraints, it distinguishes invalid output, wrong identity, wrong relation, wrong hierarchy, wrong role, structured-selection errors, and final-reference errors. Frozen-model results show that high candidate validity can hide executable mismatches, while the table diagnostic exposes a consistent correct-row/wrong-column pattern under the declared decomposition. Candidate-order controls further show that interface serialization can alter case-level outcomes without necessarily changing the localized error form, making outcome stability and diagnostic-type stability distinct measurement properties. The paired latent construction extends this analysis across compatible surfaces, with both persistent and surface-sensitive transitions. Together, these results define a controlled instrument for measuring executable correspondence; full editing trajectories, model-internal mechanisms, and future intervention methods remain outside its scope.

## Reproducibility Statement

All paper results correspond to the ExBind v2.0 release manifest in the artifact directory. The benchmark schema and evaluation code are in the source tree, and the broad suite, targeted diagnostic, table audit, adjudication pack, explicitcolumn ablation, cached predictions, existing-task coverage annotations, and legacy bridge subset are stored as versioned artifacts. The repository-level REPRODUCE\_EXBIND.md and Appendix A list tested command lines for generation, frozen-model execution, deterministic scoring, aggregation, figure rendering, and paper compilation.

## A Reproducibility and Benchmark Specification

## A.1 Data Generation

The broad diagnostic suite is generated by scripts/run\_exbind\_v2\_cross\_format.py with seed 20260823, generator version cross\_format.interface\_gate.v1, n\_per\_family=50, and n\_cross\_format\_groups=50. The targeted structural diagnostic is generated by scripts/generate\_exbind\_v2\_hard\_pilot.py with seed 20260824, generator version exbind\_v2.hard\_pilot.v1, and 40 examples per family-level cell. Relational latent instances are compiled into canvas and SVG surfaces; hierarchy instances into tree and HTML-DOM surfaces; graph instances into directed endpoint candidates; and table instances into HTML-table cell candidates. Candidate inventories are constructed from the compiler output. Gold references are derived through the latent-to-surface mapping, not by string matching against the instruction.

## Algorithm 1: ExBind Case Generation.

1. Input representation family f and seed s.

2. Sample latent binding instance $G = ( V , E , A , H , R , Y )$

3. Sample the target specification Y , including the canonical semantic target.

4. Compile the observation $O _ { f } = K _ { f } ( G )$

5. Construct executable candidate inventory $C _ { f }$ from the surface structure.

6. Derive surface gold $Y _ { f }$ through the latent-to-surface mapping $M _ { f }$

7. Derive deterministic stage labels from $G , C _ { f } , Y , Y _ { f }$ , and $M _ { f }$

8. Run integrity gates: unique gold, non-empty candidates, structural oracle, target-ID leakage, lexicalonly baseline, type-only baseline, and candidate-order shortcut check.

9. Reject the case if any required gate fails.

10. Store the latent record, rendered surface case, candidate inventory, mappings, gold reference, and stage labels.

## A.2 Benchmark Record Schema

All generated and converted cases are represented as BindingRecord objects in src/exbind/schema.py. The fields are intentionally sparse: dimensions that do not apply to a task are null rather than scored as failures. A simplified table record is shown below.

```jsonl
"case_id": "hard_table_easy_0040__table",
"latent_id": "hard_table_easy_0040",
"format": "table",
"instruction": "Select the status cell for the project
where owner Team B.",
"observation": {"source_type": "html_table", "source": "<table>..."},
"candidates": [
{"candidate_id": "cell_r01_c00", "candidate_type": "cell",
"semantic_id": "...row_1.owner"},
{"candidate_id": "cell_r01_c01", "candidate_type": "cell",
"semantic_id": "...row_1.status"}
],
"gold": {"cell_id": "cell_r01_c01"},
"binding": {
"entity": "hard_table_easy_0040_row_1",
"attribute": "status",
"target_type": "cell",
"executable_type": "cell_id"
},
```

```json
"surface_mapping": {
"hard_table_easy_0040_row_1.status": "cell_r01_c01"
},
"applicable_stages": [
"table_row", "table_column", "table_cell", "exact"
]
}
```

## A.3 Prediction and Diagnostic Scoring Contract

The model is asked to return strict JSON only. It is never asked to produce chain-of-thought. A table prediction has the form:

```json
{"cell_id": "cell_r01_c00"}
```

The evaluator parses the JSON, normalizes field aliases, checks candidate validity against the candidate inventory, maps the surface reference back to the canonical latent target, and emits a deterministic diagnostic score vector. For the table example above, selecting the owner cell in the correct row yields:

{   
"table\_row": 1,   
"table\_column": 0,   
"table\_cell": 0,   
"candidate\_valid": 1,   
"executable\_exact": 0   
}

Malformed JSON is parsed with the repository JSON extractor when possible; otherwise the prediction is treated as unresolved. Unknown candidate identifiers are invalid. For graph cases, both source\_id and target\_id must be endpoint candidate IDs, and reversing them is scored as a role/order failure rather than as an unordered object match.

## A.4 Prompt and Inference Specification

The frozen-model pilot uses the same ordinary prompt builder for all listed checkpoints:

You are evaluating a binding benchmark. Select only from   
the candidate IDs.   
Format: {format}   
Instruction: {instruction}   
Observation ({source\_type}):   
{serialized\_source}   
Candidates:   
{candidate\_json}   
Return strict JSON only with this schema: {schema}

The schema is {"target\_id": "..."} for ordinary single-target cases, {"source\_id": "...", "target\_id": "..."} for graph endpoint-pair cases, {"dom\_id": "..."} for HTML-DOM cases, and $\{ { \ " } \subset \in { \mathrm { 1 1 \_ i d " } } : \quad { \ " } \dots { \mathrm { ~ \ " ~ } } \}$ for table cases. Candidate rows include candidate\_id, candidate\_type, and bounding boxes when available. Graph rows additionally include endpoint role metadata. The explicit-column ablation adds only a column field to table candidate rows; observation, instruction, candidate order, gold, and case IDs are unchanged. The candidate-order control changes only the order of the same candidate rows using fixed seeds 2026082401, 2026082402, and 2026082403.

Table 11 specifies the frozen-model interface used for the paper results. The same ordinary contract is applied to both primary checkpoints and the supporting LLaVA-Phi pilot, so comparisons are made under a fixed reference-output protocol rather than task-specific prompt tuning. When a rendered image is present, the loader passes it through the model processor chat template; otherwise the observation is provided as serialized SVG, HTML, DOM, graph, or table source text. Historical cached predictions store the model path and decoding configuration. The paper-polish release adds frozen\_inference\_manifest.json, which records recovered checkpoint, package, device, decoding, prompt, parser, and unrecoverable historical-run fields.

Table 11: Frozen-model inference protocol.
<table><tr><td>Field</td><td>Protocol</td></tr><tr><td>Checkpoints</td><td>Qwen2.5-VL-3B-Instruct, Qwen3-VL-4B-Instruct, and LLaVA-Phi-3-mini</td></tr><tr><td>Decoding</td><td>Greedy decoding with sampling disabled</td></tr><tr><td>Generation budget</td><td>Maximum 96 new tokens</td></tr><tr><td>Random seed</td><td>42 for the evaluation wrapper</td></tr><tr><td>Prompt</td><td>Shared strict-reference template in Appendix A</td></tr><tr><td>Candidate serialization</td><td>JSON candidate rows derived from the compiler output; graph rows include endpoint role metadata</td></tr><tr><td>Output contract</td><td>Strict JSON executable reference matching the task family</td></tr><tr><td>Parsing</td><td>JSON extraction followed by field-alias normalization and candidate-inventory validation</td></tr><tr><td>Invalid outputs</td><td>Malformed or unresolved outputs are invalid; unknown identifiers are invalid candidate selec- tions</td></tr></table>

The supporting LLaVA-Phi-3-mini pilot uses the same prompt builder, candidate serialization, and unconstrained greedy decoding as the frozen Qwen interface. Its broad and targeted predictions, diagnostic scores, and run manifests are included in the versioned release artifact bundle; task-specific precedencerescored copies used for the reported failure taxonomy are retained alongside them. The older ordinarycontract LLaVA-1.5-7B-HF boundary pilot is retained as an appendix artifact. The separate finite candidate-constrained adapter is not comparable and is excluded from the paper’s model results.

## A.5 Candidate-Order Control Artifacts

The candidate-order control is included in the versioned release artifact bundle. It contains the generated permutation records, the order mapping, raw predictions, deterministic diagnostic scores, per-condition summaries, per-case transitions, transition matrix, gold-rank summaries, and the experiment manifest. The mapping file records, for every case and permutation, the original candidate order, permuted candidate order, original gold rank, and permuted gold rank. The control uses the same Qwen2.5-VL-3B checkpoint and decoding protocol as the frozen targeted table diagnostic.

Table 12: Candidate-order control on the same 120 targeted table cases. Only candidate ordering changes. Exact intervals are Wilson 95% intervals; the paired delta interval is computed by paired bootstrap over case IDs.
<table><tr><td>Condition</td><td>Cases</td><td>Validity</td><td>Exact</td><td>Exact 95% CI</td><td>Error profile</td></tr><tr><td>Original order</td><td>120</td><td>1.000</td><td>63/120</td><td>[0.436, 0.612]</td><td>57/57 errors wrong column</td></tr><tr><td>Permutation 1</td><td>120</td><td>1.000</td><td>77/120</td><td>[0.553, 0.722]</td><td>43/43 errors wrong column</td></tr><tr><td>Permutation 2</td><td>120</td><td>1.000</td><td>71/120</td><td>[0.502, 0.675]</td><td>49/49 errors wrong column</td></tr><tr><td>Permutation 3</td><td>120</td><td>1.000</td><td>68/120</td><td>[0.477, 0.652]</td><td>52/52 errors wrong column</td></tr><tr><td>Paired case IDs</td><td>120</td><td></td><td>+7.5 pp</td><td>[-1.1, 16.1]</td><td>three permutations per case; 76 order-sensitive cases</td></tr></table>

## A.6 Uncertainty Estimates

For the broad diagnostic suite, Qwen2.5-VL-3B exact accuracy is 191/250, or 76.4% with a Wilson 95% interval of [70.8, 81.2]. Qwen3-VL-4B exact accuracy is 247/250, or 98.8% [96.5, 99.6]. On the targeted table diagnostic, Qwen2.5-VL-3B is 63/120, or 52.5% [43.6, 61.2], while Qwen3-VL-4B is 120/120, or 100.0% [96.9, 100.0]. The explicit-column metadata ablation changes Qwen2.5-VL-3B from 63/120 to 71/120. The paired bootstrap estimate is +6.7 percentage points with a 95% interval of [-0.8, 14.2], so we describe it as interface sensitivity rather than as a reliable improvement claim. The candidate-order control has a paired mean delta of +7.5 percentage points over three permutations with a 95% paired bootstrap interval of [-1.1, 16.1].

## A.7 Constraint Sets and Declared-Order Summaries

For every prediction, the release stores both the stage-level score vector and the full set of applicable structural constraints that are violated. Exact-reference mismatch is retained as a separate terminal score because it is necessarily zero for every non-exact prediction; including it in every set would obscure the structural differences among errors. The first-violation label is computed only after fixing a task-family order supplied by the benchmark definition. For example, the targeted table order is row/entity → column → cell → exact reference, while a tree/DOM order uses identity → hierarchy → role → exact reference.

This distinction matters because several structural scores can co-vary after one surface selection is wrong. We therefore do not claim that first-violation labels are invariant to all alternative decompositions or that they reveal temporal model operations. The deterministic robustness analysis enumerates admissible permutations of internal structural stages while keeping candidate validity first and executable exactness last; it writes both the declared first violation and the set of possible first constraints to the versioned release artifact bundle. The main paper uses the declared order only as a concise, reproducible summary, and retains the full sets for any analysis that should not depend on ordering.

## A.8 Paired Cross-Format Diagnostic Transitions

The paired transition analysis consumes the frozen score vectors and explicit latent groups; it requires no new model inference. For each pair, it maps both surface predictions to their declared first-violation label and stores the transition together with the two surface identifiers. The SVG–canvas pairs are all none-to-none for both primary checkpoints. For tree–DOM groups, the primary transition counts are reported in Table 13. The paired rows and matrix are stored beside the constraint-set analysis. They support the limited conclusion that the common diagnostic vocabulary can describe both persistent and surface-sensitive mismatches across compatible compilers.

Table 13: Tree–DOM paired diagnostic transitions (25 groups). C denotes no violated constraint and I denotes declared first-violation identity. Counts are not an aggregate leaderboard score.
<table><tr><td>Model</td><td>C → C</td><td>C → I</td><td>I → C</td><td>I → I</td></tr><tr><td>Qwen2.5-VL-3B</td><td>15</td><td>2</td><td>3</td><td>5</td></tr><tr><td>Qwen3-VL-4B</td><td>23</td><td>2</td><td>0</td><td>0</td></tr></table>

## A.9 Construct Applicability and Opportunity-Normalized Profiles

The applicability audit makes the benchmark’s denominators explicit. Table 14 reports the number of broadsuite cases for which each evaluator-defined constraint is scored; a non-applicable dimension is excluded rather than treated as an error. Table 15 reports the nonzero first-violation counts together with these opportunity denominators. These rates describe diagnostic localization under the fixed decomposition, not causal probabilities or model-internal reasoning stages.

Table 14: Applicability opportunities in the 250-case broad suite.
<table><tr><td>Constraint</td><td>Applicable</td><td>Fraction</td><td>Representations</td></tr><tr><td>Candidate validity</td><td>250</td><td>1.000</td><td>all six</td></tr><tr><td>Identity</td><td>250</td><td>1.000</td><td>all six</td></tr><tr><td>Attribute</td><td>200</td><td>0.800</td><td>canvas, DOM, SVG, table, tree</td></tr><tr><td>Anchor</td><td>75</td><td>0.300</td><td>canvas, SVG</td></tr><tr><td>Relation</td><td>125</td><td>0.500</td><td>canvas, graph, SVG</td></tr><tr><td>Hierarchy</td><td>75</td><td>0.300</td><td>DOM, tree</td></tr><tr><td>Ordered role</td><td>125</td><td>0.500</td><td>graph, DOM, tree</td></tr><tr><td>Structured selection</td><td>50</td><td>0.200</td><td>table</td></tr><tr><td>Exact reference</td><td>250</td><td>1.000</td><td>all six</td></tr></table>

Table 15: Opportunity-normalized first-violation profile. Rates use the applicable denominator for each evaluatordefined constraint; “none” denotes no violated constraint.
<table><tr><td>Constraint</td><td>Opportunities</td><td>Qwen2.5</td><td>Qwen3</td></tr><tr><td>Candidate validity</td><td>250</td><td>4 (0.016)</td><td>0 (0.000)</td></tr><tr><td>Identity</td><td>250</td><td>28 (0.112)</td><td>3 (0.012)</td></tr><tr><td>Table column</td><td>50</td><td>27 (0.540)</td><td>0 (0.000)</td></tr><tr><td>No violated constraint</td><td>250</td><td>191 (0.764)</td><td>247 (0.988)</td></tr></table>

## A.10 Provenance and Reproduction Interface

The intended reproduction chain is:

$$
\mathrm { d a t a \ g e n e r a t i o n \to \mathrm { f r o z e n \ p r e d i c t i o n \to \ s c o r i n g } \to \ p a p e r \ t a b l e s . }
$$

The release exposes this chain as four deterministic stages. Table 16 gives the tested entry points, inputs, and outputs. Appendix A.11 and the repository-level reproduction guide provide the exact command lines; each stage consumes a versioned artifact and writes counts, configuration, and hashes without changing the frozen inputs.

Table 16: Reproduction interface for paper-facing results.
<table><tr><td>Stage</td><td>Entry point</td><td>Input</td><td>Output</td></tr><tr><td>Generate broad suite</td><td>Cross-format generator</td><td>Seed 20260823; interface-gate gener- ator config</td><td>Binding records, cross-format groups, benchmark manifest</td></tr><tr><td>Generate targeted suite</td><td>Hard-pilot generator</td><td>Seed 20260824; table and hierarchy diagnostic config</td><td>Binding records, gate summary, hard- pilot manifest</td></tr><tr><td>Frozen prediction</td><td>Frozen-model evaluator</td><td>Binding records, optional cross- format groups, checkpoint identifier, decoding protocol</td><td>Prediction JSONL, diagnostic-score JSONL, run manifest</td></tr><tr><td>Paper aggregation</td><td>Summary/table builder</td><td>Diagnostic-score files, cross-format groups, legacy bridge subset</td><td>Accuracy tables, stage profiles, consis- tency summaries, figure data</td></tr></table>

The ExBind v2.0 manifest records the artifact set used for the paper. Representative hash prefixes are listed in Table 17; the release manifest contains the complete SHA256 values for the benchmark, prediction, ablation, and bridge artifacts.

Table 17: Representative release artifacts.
<table><tr><td>Artifact</td><td>SHA256 prefix</td></tr><tr><td>Broad-suite binding records</td><td>e9925d0ef231e404</td></tr><tr><td>Targeted-suite binding records</td><td>6d779aa8e898c50c</td></tr><tr><td>Existing-task coverage manifest</td><td>5eb6db8d63b697d4</td></tr></table>

## A.11 Tested Reproduction Commands

The following commands are the paper-facing entry points tested for the release-readiness pass. They are shown with repository-relative paths and write to fresh output directories. The generator smoke tests produced five broad records and six targeted records when run with one example per family/level; the full paper configuration uses the seeds and counts specified in Appendix A.

```shell
export PYTHONPATH=src:.
python3 scripts/run_exbind_v2_cross_format.py \\
--out artifacts/reproduction/exbind_v2_cross_format \\
--n-per-family 50 --n-cross-format-groups 50 \\
--seed 20260823 --surface-variant interface_gate_v1
python3 scripts/generate_exbind_v2_hard_pilot.py \\
--out artifacts/reproduction/exbind_v2_hard_pilot \\
--per-level 40 --seed 20260824
python3 scripts/summarize_exbind_v2_frozen_model_pilot.py \\
--groups artifacts/exbind_v2/cross_format_interface_gate_v1/cross_format_groups.jsonl \\
--model Qwen2.5-VL-3B=artifacts/exbind_v2/frozen_model_pilot_interface_gate_v1/qwen25_3b \\
--model Qwen3-VL-4B=artifacts/exbind_v2/frozen_model_pilot_interface_gate_v1/qwen3_4b \\
--out artifacts/reproduction/frozen_model_summary
python3 scripts/run_exbind_v2_evidence_completion.py \\
--out artifacts/reproduction/evidence_completion
python3 scripts/render_exbind_v2_arr_figures.py \\
--paper-dir artifacts/reproduction/paper_assets \\
--failure-csv "$FAILURE_CSV"
python3 scripts/paper_figures/build_exbind_v2_candidate_order_reliability.py \\
--artifact-dir artifacts/paper_assets/exbind_v2_arr_polish/candidate_order_robustness \\
--out-dir Exbind/arr2026_october_benchmark/figures
cd Exbind/arr2026_october_benchmark
latexmk -pdf -interaction=nonstopmode -halt-on-error \\
-outdir=build exbind_benchmark.tex
```

Frozen-model inference uses the same evaluator entry point after replacing the model argument with a local checkpoint path, for example:

```shell
python3 scripts/run_exbind_v2_frozen_model_pilot.py \\
--records artifacts/reproduction/exbind_v2_cross_format/binding_records.jsonl \\
--groups artifacts/reproduction/exbind_v2_cross_format/cross_format_groups.jsonl \\
--out artifacts/reproduction/qwen25_broad \\
--model "$QWEN25_VL_3B_PATH" --model-name Qwen2.5-VL-3B \\
--model-loader qwen --seed 42 --max-new-tokens 96 --write-prompts
```

The release does not distribute model weights. The complete inference fields, including unrecoverable historical fields, are listed in the frozen inference manifest. The tested command log and smoke manifest are stored under artifacts/paper\_assets/exbind\_v2\_release\_readiness.

## A.12 Benchmark Card

Table 18: ExBind-v2 benchmark card.
<table><tr><td>Field</td><td>Specification</td></tr><tr><td>Intended use</td><td>Diagnostic evaluation of executable reference binding and failure localization</td></tr><tr><td>Unit of evaluation</td><td>One instruction, observation, candidate inventory, latent target, and executable gold</td></tr><tr><td>Representations</td><td>SVG, DOM, canvas, tree, graph, table</td></tr><tr><td>Binding dimensions</td><td>Identity, attribute, relation, hierarchy, role/order, structured selection, executable exact- ness</td></tr><tr><td>Model output</td><td>Strict JSON executable reference; no chain-of-thought required</td></tr><tr><td>Metrics</td><td>Candidate validity, exact accuracy, dimension accuracy, violated-constraint set, survival, declared-order first violation, cross-format consistency</td></tr><tr><td>Primary splits</td><td>250 broad diagnostic cases and 240 targeted structural cases</td></tr><tr><td>Data source</td><td>Deterministically generated latent records compiled into six surface representation families; inherited EVG/DOM data are separate legacy resources</td></tr><tr><td>Model baselines</td><td>Primary frozen Qwen2.5-VL-3B and Qwen3-VL-4B; supporting LLaVA-Phi-3-mini and appendix-only LLaVA-1.5-7B-HF pilot</td></tr><tr><td>Validation</td><td>Unique gold, structural oracle, target-ID leakage, lexical/type baselines, candidate-order diagnostics</td></tr><tr><td>Reference controls</td><td>Structural oracle for gold recoverability and random-candidate expectations for chance calibration</td></tr><tr><td>Release artifact</td><td>Original manifest preservedat artifacts/exbind_v2/exbind_v2_ 0_release_manifest.json; reconciled copy and evidence hashes at artifacts/paper_assets/exbind_v2_evidence_completion</td></tr><tr><td>License status</td><td>Core code is MIT-scoped and newly generated core records are CC BY 4.0-scoped; inherited raw assets with unresolved upstream terms are excluded from the core release</td></tr><tr><td>Known limitations</td><td>Two primary Qwen checkpoints; supporting LLaVA-Phi-3-mini has low table validity; controlled/synthetic cases; candidate serialization sensitivity</td></tr><tr><td>Version</td><td>ExBind-v2.0</td></tr></table>

## A.13 Protocol-Boundary Non-Qwen Pilot

We retain two ordinary-contract non-Qwen pilots to document model-family variation beyond the primary Qwen checkpoints. LLaVA-Phi-3-mini uses the same prompt builder, candidate serialization, and unconstrained greedy decoding; it obtains 220/250 valid and 154/250 exact outputs on the broad suite, and 67/120 valid and 46/120 exact outputs on the targeted table suite. Its table errors include 21 valid correctrow/wrong-column selections, classified as wrong-attribute errors under the table-specific diagnostic order, alongside 53 invalid or unresolved outputs. The older LLaVA-1.5-7B-HF pilot obtains 250/250 valid and 75/250 exact broad outputs, but only 7/120 valid and 4/120 exact targeted-table outputs; it remains a protocol-boundary artifact. Neither pilot is included in the primary matched model-profile comparison because the table validity condition differs substantially from the two Qwen checkpoints. A separate finite candidate-constrained adapter is excluded entirely because it changes the output interface.

Table 19: Frozen ordinary-contract non-Qwen pilot, reported only as a protocol boundary.
<table><tr><td>Model</td><td>Broad valid/exact</td><td>Table valid/exact</td><td>Role in paper</td></tr><tr><td>LLaVA-Phi-3-mini</td><td>220/250 ;154/250</td><td>67/120 ;46/120</td><td>supporting cross-family profile</td></tr><tr><td>LLaVA-1.5-7B-HF</td><td>250/250 ;75/250</td><td>7/120 ; 4/120</td><td>appendix-only boundary</td></tr></table>

Table 20: Inherited ExBind-Bench inventory. Counts are Stage-A executable-reference binding instances.
<table><tr><td>Split</td><td>Domain</td><td>Cases</td><td>Lineages</td><td>Task mix</td><td>Benchmark role</td></tr><tr><td>EVG-Bench</td><td>controlled SVG</td><td>800</td><td>800</td><td>edge, endpoint, group, object</td><td>clean in-domain executable binding</td></tr><tr><td>EVG-Bench- Perturbed</td><td>perturbed SVG</td><td>3,200</td><td>800</td><td>edge, endpoint, group, object</td><td>lineage-preserving shortcut and robustness tests</td></tr><tr><td>Novel-SVG</td><td>held-out SVG</td><td>300</td><td>300</td><td>endpoint, object</td><td>held-out template transfer</td></tr><tr><td>SEB-hard</td><td>external SVG</td><td>300</td><td>300</td><td>endpoint, object</td><td>SVGEditBench-derived hard transfer</td></tr><tr><td>SEB-random</td><td>external SVG</td><td>300</td><td>300</td><td>object</td><td>SVGEditBench-derived random transfer</td></tr><tr><td>SEB-perturbed</td><td>external SVG</td><td>500</td><td>100</td><td>endpoint, object</td><td>external perturbation stress tests</td></tr><tr><td>MiniWoB-Binding</td><td>HTML/DOM</td><td>500</td><td>500</td><td>DOM-reference selection</td><td>transfer from SVG identifiers to DOM ele ments</td></tr><tr><td>Total</td><td>SVG/DOM</td><td>5,900</td><td></td><td></td><td>inherited Stage-A benchmark inventory</td></tr></table>

## A.14 Legacy Diagnostic Resources

The inherited ExBind-Bench inventory is retained as historical benchmark material and as a source for cached-output bridge analysis. Table 20 lists the full 5.9K Stage-A inventory. Supporting non-Qwen runs and the separate non-comparable adapter artifacts are stored separately from the primary Qwen prediction manifests and are not used to redefine any primary split or aggregate score.

The earlier multi-panel legacy characterization is retained as a release artifact rather than included in the submission PDF. Its role is provenance for inherited assets, not a second primary evaluation narrative.

## A.15 Existing-Task Taxonomy Coverage

To test whether the controlled vocabulary can also describe existing executable-task errors, we fix a 20-error sample from cached EVG, SVGEditBench-derived, and MiniWoB DOM runs. A single manual adjudication compares each stored instruction, candidate/gold record, and cached prediction against the released taxonomy. The protocol and item-level annotations are versioned separately from all historical inputs. Table 21 reports coverage rather than prevalence: all sampled errors are covered, five multi-edit instructions are ambiguous at the task-unit level, and no example requires a new category.

Table 21: Fixed existing-task error sample used for taxonomy-coverage checking. “Ambiguous” denotes task-unit ambiguity in a multi-edit instruction, not an ambiguous executable gold for the stored error.
<table><tr><td>Source</td><td>Errors</td><td>Taxonomy covered</td><td>Ambiguous</td></tr><tr><td>EVG</td><td>8</td><td>8</td><td>0</td></tr><tr><td>SVGEditBench-derived</td><td>10</td><td>10</td><td>5</td></tr><tr><td>MiniWoB DOM</td><td>2</td><td>2</td><td>0</td></tr><tr><td>Total</td><td>20</td><td>20</td><td>5</td></tr></table>

The observed categories are 16 wrong-identity errors, two wrong-role/order errors, one wrong-relation error, and one wrong-type error. This supports only the narrow claim that the controlled taxonomy extends to this fixed sampled subset of existing executable tasks. It neither estimates real-agent failure frequencies nor validates the taxonomy for all open-ended editing trajectories. The coverage artifact is included in the versioned release artifact bundle.

## References

Baechler et al. ScreenAI: A vision-language model for UI and infographics understanding. Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2024.

Chen et al. WebSRC: A dataset for web-based structural reading comprehension. Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021.

Cheng et al. SeeClick: Harnessing GUI grounding for advanced visual GUI agents. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

Deng et al. Mind2Web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Hudson and Manning. GQA: A new dataset for real-world visual reasoning and compositional question answering. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Hui et al. WinSpot: GUI grounding benchmark with multimodal large language models. ACL Findings/Short Papers, 2025.

Johnson et al. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Kamath et al. MDETR: Modulated detection for end-to-end multi-modal understanding. Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

Koh et al. VisualWebArena: Evaluating multimodal agents on realistic visual web tasks. ICLR 2024 Workshop, 2024.

Lee et al. Pix2Struct: Screenshot parsing as pretraining for visual language understanding. Proceedings ofthe 40th International Conference on Machine Learning (ICML), 2023.

Li et al. Grounded language-image pre-training. Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

Li et al. ScreenSpot-Pro: GUI grounding for professional high-resolution computer use. arXiv preprint, 2025.

Lightman et al. Let’s verify step by step. arXiv preprint, 2023.

Liu et al. Grounding DINO: Marrying DINO with grounded pre-training for open-set object detection. arXiv preprint, 2023.

Lu et al. OmniParser for pure vision based GUI agent. arXiv preprint, 2024.

Masry et al. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. Findings of the Association for Computational Linguistics: ACL 2022, 2022.

Nayak et al. UI-Vision: A desktop-centric GUI benchmark for visual perception and interaction. Proceedings ofthe 42nd International Conference on Machine Learning (ICML), 2025.

Shi et al. World of Bits: An open-domain platform for web-based agents. Proceedings ofthe 34th International Conference on Machine Learning (ICML), 2017.

Wang et al. MMBench-GUI: Hierarchical multi-platform evaluation framework for GUI agents. arXiv preprint, 2025

Wei et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Xie et al. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems (NeurIPS), 2024.

Yu et al. Modeling context in referring expressions. Proceedings of the European Conference on Computer Vision (ECCV), 2016.

You et al. Ferret: Refer and ground anything anywhere at any granularity. International Conference on Learning Representations (ICLR), 2024.

You et al. Ferret-UI: Grounded mobile UI understanding with multimodal LLMs. European Conference on Computer Vision (ECCV), 2024.

Zheng et al. GPT-4V(ision) is a generalist web agent, if grounded. Proceedings ofthe 41st International Conference on Machine Learning (ICML), 2024.

Zhou et al. WebArena: A realistic web environment for building autonomous agents. International Conference on Learning Representations (ICLR), 2024.