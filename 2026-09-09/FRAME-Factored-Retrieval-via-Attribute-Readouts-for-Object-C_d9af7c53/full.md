# FRAME: Factored Retrieval via Attribute Readouts for Object-Centric Scene Memory

Woosang Jeon<sup>∗</sup> Seoul National University jwoosang1@snu.ac.kr

Sanghyeok Choi<sup>∗</sup> Seoul National University cholsang83@snu.ac.kr

Minwoo Kim Seoul National University minwoo.kim@snu.ac.kr

Taehyun Jung Seoul National University jth135@snu.ac.kr

Taehyeong Kim<sup>†</sup> Seoul National University taehyeong.kim@snu.ac.kr

Abstract: Language-guided robots need persistent scene memories to follow instructions, revisit objects, and resolve references to objects encountered over time. While much of language-guided scene-memory retrieval has emphasized spatial or relational references, many everyday object references specify objects by multiple persistent attributes, such as category, material, size, or surface appearance. We formalize this problem as attribute-compositional retrieval, where a fixed object-centric scene memory is queried with natural language to retrieve the object satisfying the requested attributes. To investigate this capability directly, we introduce a controlled evaluation protocol with fixed scene memories and attributedefined targets, separating retrieval from perception and annotation ambiguities. We then propose FRAME, which turns language into query-relevant attribute weights, uses learned readouts to estimate per-attribute evidence from object embeddings, and ranks objects by aggregating this evidence according to the query. Across heldout scenes and object assets, FRAME outperforms representative scene-memory retrieval baselines while reducing post-decomposition object scoring to lightweight matrix-vector computation. These results position attribute-compositional retrieval as a complementary scene-memory capability for language-guided robots, showing that persistent object attributes can be exposed as composable evidence for accurate and efficient multi-attribute retrieval.

Keywords: Object-Centric Scene Graph, Attribute-Compositional Retrieval, Language-Guided Robotics

## 1 Introduction

Language-guided robots need persistent scene memories to follow instructions, revisit objects, and resolve references to objects encountered over time [1, 2, 3]. A common abstraction for such memories is an object-centric scene graph, in which each node stores visual and semantic representations of an object (e.g., appearance features, visual-language embeddings, or semantic labels), while edges encode spatial or geometric relations between objects [4, 2]. This structure naturally supports spatial references such as “the chair next to the table” or “the mug on the shelf”, and much of language grounding in 3D scenes has therefore focused on spatial anchors, relational expressions, and graph-based reasoning over object relations [5, 6, 7].

However, many everyday references identify objects through conjunctions of persistent attributes rather than relations to other objects, such as category, material, size, and surface appearance. For example, given the query “the wooden chair,” the correct object is the chair made of wood, whereas both a wooden table and a plastic chair should be excluded. This raises a key question about retrieval:

given an object-centric scene graph used as scene memory, how should language queries involving multiple object attributes be resolved? We study this problem as attribute-compositional retrieval and investigate it with a controlled evaluation protocol that fixes the scene memory and defines retrieval targets based on predefined attribute combinations.

Under this setting, common retrieval approaches exhibit complementary limitations. Embeddingbased approaches typically compare the language query against each object embedding with a single query–object score, making it difficult to distinguish full attribute matches from partial ones. LLM/VLM-mediated and graph-based approaches can make attribute reasoning more explicit, but iterative query-time reasoning over scene context or candidate objects is computationally costly and ties object-attribute estimation to each query-time reasoning call [2, 8, 3, 9]. We therefore propose FRAME—Factored Retrieval via Attribute Readouts for Object-Centric Scene MEmory. FRAME reads out query-relevant attribute evidence from object embeddings in a composable form, allowing multi-attribute references to be matched by combining the corresponding readout scores. In held-out scenes and object assets, FRAME outperforms representative scene-memory retrieval baselines on this task while reducing post-decomposition object scoring to lightweight matrix-vector computation.

Contributions. (1) We formulate attribute-compositional retrieval over object-centric scene graphs: retrieving the object that satisfies a query-specified set of persistent attributes. (2) We introduce a controlled evaluation protocol for attribute-compositional retrieval that uses fixed scene memories and defines retrieval targets based on predefined combinations of object attributes, isolating retrieval performance from perception and annotation ambiguity. (3) We propose FRAME, an attributecompositional retrieval method that decomposes language into attribute primitives, reads out attributespecific evidence from object embeddings, and combines the resulting evidence according to the query to score candidate objects.

## 2 Related Work

Benchmarks for language grounding in 3D scenes. Datasets such as ScanRefer [5], ReferIt3D [6], Multi3DRefer [7], Scan2Cap [10], MMScan [11], and IRef-VLA [12] have established 3D grounding as selecting target object(s) from language in a 3D scene. They are especially valuable for evaluating spatial, relational, viewpoint-dependent, and context-rich grounding. Although such benchmarks contain attribute-like descriptions, these cues are usually entangled with category, spatial anchors, relations, viewpoint language, and broader scene context. As a result, they do not evaluate attribute composition in isolation. We instead study composition of persistent attributes directly as attributecompositional retrieval over a fixed scene memory.

3D scene graphs and scene memory. Persistent 3D scene representations provide the substrate for language-guided embodied agents. Prior work has studied 3D scene graphs for semantic mapping, actionable spatial perception, indoor scene-graph prediction, and semantic object-goal inspection [1, 4, 13, 14, 15]. Recent open-vocabulary and hierarchical scene-memory systems, including ConceptGraphs [2], HOV-SG [3], OpenScene [16], and OpenMask3D [17], build rich object-centric memories from visual, textual, and geometric evidence. Other memory-centric systems extend this direction toward long-horizon robot memory, episodic recall, or viewpoint-based observations, such as ReMEmbR [18], 3D-Mem [19], and Mind Palace [20]. Together, these works make persistent scene memory increasingly available to robots, but leave open how such memories should be queried once constructed.

LLM/VLM-mediated scene-memory querying. Recent approaches use LLMs or VLMs at inference time to mediate between natural-language queries and 3D scene memory. Some rank objects from graph context, captions, and spatial relations [2, 8]; others retrieve candidates or decompose queries before LLM/tool-based reranking or grounding [3, 9, 21]. These approaches are flexible for open-ended, relational, and task-level language, but their flexibility comes with a cost. For persistent attribute-centric retrieval, whether an object satisfies the queried attributes is determined at inference time rather than represented explicitly as reusable, query-independent object-side evidence. As a result, they typically require iterative query-time reasoning over scene context or candidate objects.

![](images/e9c12a0eb7c1650d815aed65b94d7ac60ceacc2e857cfdd9918e184d9282d120.jpg)  
Figure 1: Attribute-Compositional Retrieval protocol construction. HSSD object annotations and asset metadata define a primitive vocabulary of category attributes and object properties, together with retrieval targets for each attribute composition. GPT-4o generates natural-language paraphrases while preserving the underlying target specification. Scene-, asset-, and composition-disjoint splits followed by answerability filtering produce the evaluation pairs.

Attribute recognition and compositionality. Attribute detection and compositional recognition have been extensively studied in 2D and multimodal representation learning, including openvocabulary attribute detection [22] and analyses of compositional behavior in vision-language representations [23]. Open-vocabulary 3D understanding benchmarks, including OpenScan [24], also evaluate attribute-related recognition in 3D scenes. Nevertheless, recent work suggests that compositional retrieval remains challenging even when attribute–object associations are encoded in vision-language embeddings [25]. We study this gap in the context of persistent scene memory, treating attributes as composable object-side evidence for language-guided retrieval.

## 3 Attribute-Compositional Retrieval Protocol

We instantiate the Attribute-Compositional Retrieval protocol on the Habitat Synthetic Scenes Dataset (HSSD) [26] and construct it in three stages (Figure 1): (i) attribute formulation, which establishes a primitive attribute vocabulary and target predicates from object annotations and asset metadata, providing unambiguous retrieval targets; (ii) query construction, which converts attribute compositions into natural-language queries through constrained paraphrasing while preserving the underlying target specification; and (iii) evaluation split construction, which produces held-out evaluation pairs through scene- and object-asset-disjoint splits together with answerable-pair filtering, enabling evaluation on unseen scenes, object assets, and attribute compositions.

The resulting protocol isolates attribute-compositional retrieval: each query specifies a combination of object attributes, and a valid answer must satisfy the full composition. All approaches use an identical object inventory, frozen object-node features, and scene-graph structure, removing differences in scene representation as a confound.

Attribute vocabulary. We define a primitive vocabulary V of 47 attributes covering fine category, super category, material, size, surface roughness, metallicity, and articulatability. The vocabulary is derived from HSSD annotations and asset metadata: fine categories and super categories come from main\_category and super\_category, size attributes from aligned.dims, articulatability from isArticulatable, and material, roughness, and metallicity from asset shader names and PBR flags. We refer to material, size, surface roughness, metallicity, and articulatability as properties; attribute is used as an umbrella term encompassing both categories (fine and super categories) and properties.

Table 1: Evaluation classes and retained query–scene pair counts. Validation contains seen compositions and test contains unseen compositions; difficulty statistics are averaged over answerable composition–scene combinations from both splits.
<table><tr><td rowspan="2">Class</td><td colspan="2">Query-scene pairs</td><td rowspan="2">Targets / scene</td><td rowspan="2">Near-miss / scene</td><td rowspan="2">Example query</td></tr><tr><td>Val. (seen)</td><td>Test (unseen)</td></tr><tr><td>C1: Fine cat.</td><td>1,190</td><td>750</td><td>4.60</td><td></td><td>&quot;Find the chair.&quot;</td></tr><tr><td>C2: Fine cat. + material</td><td>1,030</td><td>160</td><td>2.03</td><td>18.65</td><td>&quot;Find a chair made of fabric.&quot;</td></tr><tr><td>C3: Fine cat. + size</td><td>2,260</td><td>190</td><td>2.52</td><td>33.29</td><td>&quot;Find the medium-sized ceiling lamp.&quot;</td></tr><tr><td>C4: Fine cat. + other property</td><td>530</td><td>200</td><td>2.60</td><td>49.53</td><td>&quot;Find the table with a glossy surface.&quot;</td></tr><tr><td>C5: Super cat. + one property</td><td>2,500</td><td>1,690</td><td>3.84</td><td>27.59</td><td>&quot;Find the storage furniture with a glossy finish.&quot;</td></tr><tr><td>C6: Three attributes</td><td>1,000</td><td>730</td><td>2.40</td><td>11.25</td><td>&quot;Find the large metal storage furniture.&#x27;</td></tr><tr><td>Total</td><td>8,510</td><td>3,720</td><td></td><td></td><td></td></tr></table>

Composition classes and query construction. A target specification is a fixed conjunction of one to three attributes drawn from V; for example, cat\_chair∧mat\_fabric selects fabric chairs. The benchmark defines 60 such attribute compositions, grouped into six classes (C1–C6) by the kind of conjunction required, as summarized in Table 1. For each query q, correct objects are those satisfying every attribute in the target specification. The target specification is fixed before query generation; GPT-4o [27] is used only offline to generate 10 natural-language paraphrases per composition under a structurally constrained prompt. Thus, paraphrasing changes the surface form while leaving the target specification unchanged. The full composition list is in Appendix C, and the paraphrasing prompt is in Appendix B.2.

Scene-, asset-, and composition-disjoint split. We first split HSSD into 117 training, 25 validation, and 26 test scenes. Because HSSD reuses 3D object assets across scenes, a scene-level split alone can place the same underlying asset in both training and evaluation scenes. We therefore remove from the training pool every object asset that appears in a

Table 2: Scene-level statistics for dataset splits.
<table><tr><td></td><td>Train</td><td>Val</td><td>Test</td></tr><tr><td>Scenes</td><td>117</td><td>25</td><td>26</td></tr><tr><td>Objects / scene (avg.)</td><td>109.9</td><td>105.2</td><td>118.5</td></tr><tr><td>Rooms / scene (avg.)</td><td>5.32</td><td>5.20</td><td>5.15</td></tr></table>

validation or test scene. We additionally partition the 60 compositions into 43 seen compositions used for training and validation and 17 unseen compositions used only for testing. Every primitive attribute in an unseen composition is instantiated by at least one object in both the training and validation scenes. The test split therefore evaluates novel compositions of familiar attributes. The unseen compositions are never used for validation or model selection.

Evaluation pairs and difficulty. We retain a query–scene pair only when the scene contains at least one object satisfying the target specification. This yields 8,510 pairs for the 43 seen compositions across 25 validation scenes and 3,720 pairs for the 17 unseen compositions across 26 test scenes. For each retained composition–scene combination, targets are objects satisfying the full specification, whereas near-misses are objects satisfying all but one required attribute. Near-misses are therefore plausible distractors that can be rejected only by considering the full attribute conjunction. Table 1 reports the average numbers of targets and near-misses within each class.

## 4 FRAME

We propose FRAME as an approach to attribute-compositional retrieval under this protocol. FRAME factors attribute-compositional retrieval into three parts: (i) a language interface that specifies which attributes matter for a query, represented as a sparse composition vector over the primitive attribute vocabulary V, (ii) trained attribute readouts that estimate per-object evidence for each attribute from object-node embeddings, and (iii) a retrieval rule that composes the selected evidence according to the query. This separation allows attribute evidence to be estimated once from stored object representations and reused across different queries through query-dependent composition.

![](images/7400497231eb4d84abe9a8d387d9ca29cc4a21d0573dee10c51a550ec9763dc7.jpg)  
Figure 2: FRAME inference. A natural-language query is first decomposed into attribute weights over the primitive vocabulary. For each object node in the scene memory, learned attribute readouts estimate evidence for the queried attributes from stored object embeddings. The selected evidence is then aggregated according to the query weights to score and rank candidate objects. The bottom row illustrates an HSSD test scene where storagefurniture, wide, and wood evidence combine to localize the target object.

Query understanding via LLM decomposition. At query time, a single LLM call converts a natural-language command q into a sparse composition vector $w ( q ) \in [ 0 , 1 ] ^ { | V | }$ over the attribute vocabulary $V .$ . The decomposition prompt (Appendix B.1) instructs the LLM to assign higher weights to required attributes and lower weights to weak modifiers, omitting attributes that are irrelevant. Each reference thereby defines a query-specific composition over persistent attributes; the resulting weights determine which object-side readouts are composed for retrieval.

Attribute probes. For each attribute $a \in V$ , FRAME trains a 2-layer MLP readout probe $\phi _ { a }$ on the attribute labels of objects o in the training pool for a fixed number of epochs; architecture and training details are in Appendix A.1. Given an object-node representation ${ \bf x } ( o )$ , the probe outputs

$$
s _ { a } ( o ) = \sigma ( \phi _ { a } ( \mathbf { x } ( o ) ) ) \in [ 0 , 1 ] .
$$

We instantiate two variants: FRAME-Vis uses visual embeddings, $\mathbf { x } ( o ) = \bar { e } _ { \mathrm { v i s } } ( o )$ , while FRAME-VT concatenates visual and caption-derived text features, $\mathbf { x } ( o ) = [ \bar { e } _ { \mathrm { v i s } } ( o ) ; \bar { e } _ { \mathrm { t e x t } } ( o ) ]$ ]. In our approach, the validation split is used only for the readout-quality analysis in §5.3, keeping the test split reserved for the retrieval evaluation in §5.2.

Query-time retrieval. Given query-side composition weights $w ( q )$ and per-attribute evidence scores, FRAME measures how well object o satisfies the full attribute composition by a normalized weighted average:

$$
\operatorname { s c o r e } ( o , q ) = \frac { \sum _ { a \in V } w _ { a } ( q ) s _ { a } ( o ) } { \sum _ { a \in V } w _ { a } ( q ) } ,
$$

and returns the highest-scoring object node. The probes can be applied lazily at query time only for attributes with $w _ { a } ( q ) > 0$ , or eagerly at scene-ingestion time by caching all probe scores for a scene s with objects $O _ { s }$ as a per-scene matrix

$$
S _ { s } \in [ 0 , 1 ] ^ { | O _ { s } | \times | V | } .
$$

The eager mode trades a one-time ingestion cost for low-cost per-query object scoring; this is the setting used for the latency measurements in §5.2.

## 5 Experiments

## 5.1 Experimental Setup

Scene-memory representation. Evaluation uses 26 held-out test scenes containing 3,720 valid query–scene pairs for the 17 unseen compositions; validation uses 25 scenes containing 8,510 valid pairs for the 43 seen compositions. All retrieval methods compared in this section operate on the same captioned object-centric scene graph, with one node per HSSD object in a ConceptGraph-style format [2]. Each object node contains a SigLIP-so400m visual embedding [28] $( \bar { e } _ { \mathrm { v i s } }$ , 1152d) from a canonical rendered view and a six-field GPT-4o-mini object caption (Appendix B.3); caption-based methods encode captions with BGE-large-en-v1.5 [29] $( \bar { e } _ { \mathrm { t e x t } }$ , 1024d).

Baselines. We compare representative approaches for retrieving objects from scene memory, spanning embedding-based retrieval, LLM-mediated ranking, retrieval-then-reranking pipelines, and attribute-compositional retrieval. These categories reflect different ways of accessing information stored in object-centric scene memories.

• Zero-shot embedding retrieval. Rank objects by direct query–object similarity without taskspecific training. ZS-Visual, ZS-Caption, and ZS-V+C use SigLIP text–image similarity, BGE query–caption similarity, and reciprocal-rank fusion of the two rankings, respectively.

• Learned embedding retrieval. Train two retrieval functions on the training pool with binary cross-entropy that scores positive pairs higher than negatives. BiEncoder encodes the query and each object independently and ranks by a single cosine similarity, whereas CrossEncoder concatenates the query and object representations and predicts relevance jointly from each pair.

• LLM-mediated ranking. Use an LLM to rank objects directly from scene-memory contents, including captions and graph context. CG-LLM adapts ConceptGraphs [2]; BBQ [8] filters with short captions and reranks with full captions, 3D positions, and graph edges.

• Retrieval-then-LLM reranking. First retrieve a candidate set using embedding similarity, then use an LLM to rerank candidates with additional scene context. EmbodiedRAG [9] uses SigLIP top-K retrieval with local-subgraph reranking; HOV-SG [3] additionally decomposes queries before filtering and reranking.

• Attribute-compositional retrieval. Decompose the query into attributes and retrieve objects by aggregating attribute-specific evidence rather than a single holistic similarity score. FRAME-ZS replaces learned attribute probes with zero-shot visual–text similarity. FRAME-Vis applies learned probes to SigLIP visual embeddings, while FRAME-VT applies them to concatenated SigLIP visual and BGE caption embeddings.

Unless otherwise noted, all LLM-based components, including FRAME query decomposition and LLM-mediated baselines, use gpt-4o-mini with temperature 0. Prompts, top-K values, model versions, and adaptation details are provided in Appendix B.

Evaluation metrics. For each query–scene pair, a method returns a ranked object list $R \ =$ $( r _ { 1 } , \ldots , r _ { | O _ { s } | } )$ and is evaluated against the attribute-defined ground-truth set $G ( q , s )$ . We use P@1 as the primary metric since the task requires returning a single object, and report Hit@5, mean reciprocal rank (MRR), and mean average precision (mAP) to characterize the full ranking:

$$
\mathrm { P @ 1 } = \mathbb { k } [ r _ { 1 } \in G ] , \quad \mathrm { H i t @ 5 } = \mathbb { k } [ \exists i \leq 5 : r _ { i } \in G ] ,
$$

$$
\mathrm { R R } = \frac { 1 } { \operatorname* { m i n } \{ i : r _ { i } \in G \} } , \quad \mathrm { A P } = \frac { 1 } { | G | } \sum _ { i = 1 } ^ { | O _ { s } | } \mathrm { P r e c } \ @ i \cdot | \mathcal { k } [ r _ { i } \in G ] ,
$$

where $\begin{array} { r } { \operatorname { P r e c @ } i = \frac { 1 } { i } \sum _ { j = 1 } ^ { i } \mathcal { k } [ r _ { j } \in G ] } \end{array}$ . We average these quantities over all evaluated query–scene pairs and report them in percent.

Table 3: Compositional retrieval performance on the held-out test split, grouped by access pattern. Accuracy metrics are reported in percent; latency is median per-query wall-clock time. Best results are in bold and the best non-FRAME baseline is underlined.
<table><tr><td>Retrieval approach</td><td>Method</td><td>P@1</td><td>Hit@5</td><td>MRR</td><td>mAP</td><td>Latency</td></tr><tr><td rowspan="3">Zero-shot embedding retrieval</td><td>ZS-Visual</td><td>46.40</td><td>83.92</td><td>62.22</td><td>51.57</td><td>10 ms</td></tr><tr><td>ZS-Caption</td><td>30.16</td><td>72.58</td><td>48.33</td><td>37.79</td><td>13 ms</td></tr><tr><td>ZS-V+C</td><td>46.77</td><td>82.66</td><td>61.60</td><td>49.19</td><td>24 ms</td></tr><tr><td rowspan="2">Learned embedding retrieval</td><td>BiEncoder</td><td>48.87</td><td>87.15</td><td>65.13</td><td>54.44</td><td>14 ms</td></tr><tr><td>CrossEncoder</td><td>62.58</td><td>93.79</td><td>75.50</td><td>66.06</td><td>10 ms</td></tr><tr><td rowspan="2">LLM-mediated ranking</td><td>CG-LLM</td><td>27.23</td><td>66.53</td><td>43.96</td><td>29.66</td><td>1,996 ms</td></tr><tr><td>BBQ</td><td>41.21</td><td>75.13</td><td>55.17</td><td>38.11</td><td>6,256 ms</td></tr><tr><td rowspan="2">Retrieval-then-LLM reranking</td><td>EmbodiedRAG</td><td>44.57</td><td>78.68</td><td>59.10</td><td>48.20</td><td>3,187 ms</td></tr><tr><td>HOV-SG</td><td>42.02</td><td>77.88</td><td>57.50</td><td>45.64</td><td>4,398 ms</td></tr><tr><td rowspan="3">Attribute-compositional</td><td>FRAME-ZS</td><td>35.67</td><td>68.20</td><td>50.53</td><td>40.65</td><td>1,532 ms</td></tr><tr><td>FRAME-Vis</td><td>77.82</td><td>96.21</td><td>86.01</td><td>78.30</td><td>1,638 ms</td></tr><tr><td>FRAME-VT</td><td>75.75</td><td>95.97</td><td>84.62</td><td>76.28</td><td>1,750 ms</td></tr></table>

## 5.2 Compositional Retrieval Performance

Overall accuracy. Table 3 shows that existing scene-memory retrieval methods fall short on attribute-compositional retrieval. Zero-shot embedding retrieval is fast, but a single query–object similarity score is a weak interface for distinguishing full attribute matches from partial ones; the strongest variant, ZS-V+C, reaches 46.77% P@1. Even learned retrieval models (BiEncoder and CrossEncoder) reach only 48.87% and 62.58% P@1, respectively, suggesting that improved similarity or relevance learning alone does not resolve attribute-compositional retrieval. LLM-mediated ranking and retrieval-then-LLM reranking remain below 44.57% P@1 despite query-time reasoning over captions, graph context, or retrieved candidates. FRAME-Vis achieves the best overall result at 77.82% P@1, improving over CrossEncoder, the strongest non-FRAME baseline, by 15.24 percentage points. FRAME-VT trails FRAME-Vis only slightly, at 75.75% P@1.

Ranking quality beyond top-1. FRAME-Vis’s advantage extends beyond P@1, also achieving the highest Hit@5, MRR, and mAP. Notably, several baselines reach comparably high Hit@5 but leave a much larger gap to their own P@1; ZS-Visual reaches 83.92% Hit@5 but only 46.40% P@1, BiEncoder reaches 87.15% Hit@5 bust only 48.87% P@1, and CG-LLM reaches 66.53% Hit@5 but only 27.23% P@1. This indicates that the correct object is often among the top few candidates across baselines, but resolving the full attribute conjunction to rank it first remains difficult without reliable, factored object-level evidence.

Factored retrieval and reliable readouts. These results suggest that factoring retrieval by attribute and estimating each attribute reliably interact rather than help independently. FRAME-ZS shows that factoring alone does not help: it applies the same attribute-compositional interface as FRAME-Vis and FRAME-VT but uses zero-shot attribute scores instead of learned readouts, reaching only 35.67% P@1, below even the holistic zero-shot baselines. Conversely, BiEncoder and CrossEncoder show that task-tuning alone also has limits: both are trained on the same supervision as FRAME, yet each still collapses a multi-attribute reference into a single query–object score, and neither reaches 63% P@1. FRAME-Vis and FRAME-VT combine both ingredients, and the resulting gap over the best baseline is consistent with the two being complementary rather than separately sufficient.

Latency and retrieval cost. Embedding retrieval methods are fastest at 10–24 ms, but remain far behind FRAME-Vis and FRAME-VT in accuracy. LLM-based ranking and reranking baselines require fresh LLM calls over scene or candidate context and take 2.0–6.3 s per query. FRAME’s total latency is also dominated by a single LLM call for query decomposition, making its absolute per-query latency comparable to the faster LLM-mediated baselines. However, this call is sceneindependent because its prompt contains the attribute vocabulary and the user query, not scene objects, captions, or candidate subgraphs. Attribute readout scores are instead cached once per scene as an object–attribute matrix, so post-decomposition scoring reduces to a matrix–vector product taking approximately 0.01 ms. Thus, FRAME’s efficiency advantage is the decoupling of language parsing from scene-context reasoning and the amortization of attribute readout computation across repeated queries to the same memory.

## 5.3 Performance Breakdown

Probe bottlenecks. Because FRAME’s retrieval score is built from per-attribute readouts, its accuracy should depend on how reliable those readouts are for the attributes a query requires. We examine this by reporting validation AP for the learned readouts, grouped by attribute family (Table 4). Category and size readouts are reliable, readouts for other properties are moderately strong, and material readouts are substantially weaker. Here, the category group averages fine- and super-category probes,

Table 4: Probe validation AP over the 47 object attributes.
<table><tr><td>Group</td><td> $\mathsf { A P } _ { \mathrm { V i s } }$ </td><td> $\mathsf { A P } _ { \mathrm { V T } }$ </td></tr><tr><td>Category</td><td>0.909</td><td>0.912</td></tr><tr><td>Size</td><td>0.877</td><td>0.863</td></tr><tr><td>Material</td><td>0.505</td><td>0.465</td></tr><tr><td>Other properties</td><td>0.840</td><td>0.821</td></tr></table>

while other properties comprise surface roughness, metallicity, and articulatability. This suggests that material is the main bottleneck, likely because single-view material cues are confounded by texture, lighting, finish, and asset style. This weakness is most directly reflected in class C2 (fine category + material), which explicitly requires material grounding. Nevertheless, FRAME’s gains on this class suggest that factored composition can combine weaker material evidence with other query-specified cues rather than collapsing the full reference into a single entangled score.

Per-class breakdown. Figure 3 reports per-class P@1 for the representative baselines and FRAME variants. The advantage of FRAME-Vis is smallest on single-category queries and generally larger on the multi-attribute classes. CrossEncoder, the strongest non-FRAME baseline overall, reaches 95.5% P@1 on single-category queries (C1) but falls to 38.5% on three-attribute queries (C6), a 57.0,pp drop; the other baselines show the same broad decline from C1 to C6. FRAME-Vis drops by 38.1 pp over the same comparison (98.7 → 60.5), showing robustness to conjunctive object references.

![](images/654e091a831d72d6ea9276872e8e7c01ed9ce6b9351b3bc994064dcf64c99e35.jpg)  
Figure 3: Per-class test P@1 (%). Representative baselines are shown for each retrieval family; the full table is in Appendix D. FRAME variants outperform the shown baselines across every composition class, with the largest gap on C6, where targets require conjunctions of three attributes.

Comparing the three two-attribute classes reveals a consistent ordering across nearly all methods, including FRAME-ZS: size (C3) is easiest, material (C2) is harder, and the other-property family (C4) is hardest, with FRAME-ZS falling to 2.0% on C4. This ordering is not what the probe validation AP in Table 4 would predict on its own, since other-property probes are on average more reliable than material probes. A likely factor is that C4 also has the highest near-miss density of the three classes (Table 1), so ranking the correct object first requires distinguishing it from many more attribute-matching distractors, independent of how reliable any single attribute score is. FRAME-Vis and FRAME-VT track each other closely across most classes but diverge on C2 (75.0% vs. 62.5%), consistent with material being the family where FRAME-VT’s readouts are least reliable relative to FRAME-Vis (Table 4).

Held-out scene consistency. To assess whether this held-out performance is consistent across individual scenes rather than driven by a few favorable ones, we examine per-scene P@1 across the 26 test scenes. Figure 4 reports the per-scene P@1 distribution for all methods. FRAME-Vis maintains the highest median P@1 across scenes, indicating that its advantage holds consistently rather than concentrating in a few favorable layouts. FRAME-VT trails FRAME-Vis slightly in overall P@1 but has the tightest spread across scenes (IQR 9.13, compared with 19.43 for FRAME-Vis and 19.61 for CrossEncoder). This shows that FRAME’s advantage holds consistently across scenes, rather than being driven by a small number of favorable layouts.

![](images/6ff03d78e2bfd9732bbec0c81b717dd2175cdaa3e217e8951bc10bd8dd65f8fe.jpg)  
Figure 4: Per-scene test P@1 (%) across all 26 test scenes. Each box spans Q1–Q3 with the median marked by the horizontal line, whiskers extend to the minimum and maximum, and the white dot marks the mean. FRAME-Vis and FRAME-VT achieve the two highest medians, while FRAME-VT exhibits the tightest spread.

As a further, more demanding test of generalization, Appendix E reports results on the 3D Semantic Scene Graphs (3DSSG) dataset [14]. Rather than demonstrating deployment-ready real-scene grounding, these results show that semantic and appearance-related readouts can be learned from realscene object image crops, while weak size readouts expose limitations arising from noisy geometry, the absence of depth or scale calibration, and imperfect segmentation.

## 6 Conclusion

This work positions attribute-compositional retrieval as a complementary mode of querying scene memories for language-guided robots. FRAME embodies this view by letting language specify the persistent attributes that matter, reading out evidence for those attributes from object embeddings, and composing that evidence to retrieve objects matching multi-attribute references. The controlled evaluation shows that this factorized retrieval strategy improves attribute-compositional retrieval over representative scene-memory retrieval baselines, while reducing post-decomposition object scoring to lightweight matrix-vector computation rather than repeated scene-context reasoning. These results suggest that robot scene memories can benefit from compositional access to objects through language-specified persistent attributes, alongside existing spatial or relational querying.

Limitations. Our HSSD-based protocol isolates retrieval over a predefined attribute vocabulary, leaving open extensions to open-vocabulary attributes, relational references, and richer language phenomena. Within this vocabulary, some attributes remain difficult to estimate reliably from a single view, most notably material, which limits how far factored composition alone can compensate for unreliable evidence. Extending this perspective beyond the controlled setting and preliminary realscene checks will require more robust robot-built object memories, including reliable segmentation, object-centric representation learning, and scene-graph construction.

## References

[1] U.-H. Kim, J.-M. Park, T.-J. Song, and J.-H. Kim. 3-d scene graph: A sparse and semantic representation of physical environments for intelligent agents. IEEE transactions on cybernetics, 50(12):4921–4933, 2019.

[2] Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, et al. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 5021–5028. IEEE, 2024.

[3] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard. Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation. In First Workshop on Vision-Language Modelsfor Navigation and Manipulation at ICRA 2024, 2024.

[4] A. Rosinol, A. Gupta, M. Abate, J. Shi, and L. Carlone. 3d dynamic scene graphs: Actionable spatial perception with places, objects, and humans. arXiv preprint arXiv:2002.06289, 2020.

[5] D. Z. Chen, A. X. Chang, and M. Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In European conference on computer vision, pages 202–221. Springer, 2020.

[6] P. Achlioptas, A. Abdelreheem, F. Xia, M. Elhoseiny, and L. Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In European conference on computer vision, pages 422–440. Springer, 2020.

[7] Y. Zhang, Z. Gong, and A. X. Chang. Multi3drefer: Grounding text description to multiple 3d objects. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15225–15236, 2023.

[8] S. Linok, T. Zemskova, S. Ladanova, R. Titkov, D. Yudin, M. Monastyrny, and A. Valenkov. Beyond bare queries: Open-vocabulary object grounding with 3d scene graph. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 13582–13589. IEEE, 2025.

[9] M. Booker, G. Byrd, B. Kemp, A. Schmidt, and C. Rivera. Embodiedrag: Dynamic 3d scene graph retrieval for efficient and scalable robot task planning. arXiv preprint arXiv:2410.23968, 2024.

[10] Z. Chen, A. Gholami, M. Nießner, and A. X. Chang. Scan2cap: Context-aware dense captioning in rgb-d scans. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3193–3203, 2021.

[11] R. Lyu, J. Lin, T. Wang, S. Yang, X. Mao, Y. Chen, R. Xu, H. Huang, C. Zhu, D. Lin, et al. Mmscan: A multi-modal 3d scene dataset with hierarchical grounded language annotations. Advances in Neural Information Processing Systems, 37:50898–50924, 2024.

[12] H. Zhang, N. Zantout, P. Kachana, J. Zhang, and W. Wang. Iref-vla: A benchmark for interactive referential grounding with imperfect language in 3d scenes. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 1677–1683. IEEE, 2025.

[13] A. Rosinol, A. Violette, M. Abate, N. Hughes, Y. Chang, J. Shi, A. Gupta, and L. Carlone. Kimera: From slam to spatial perception with 3d dynamic scene graphs. The International Journal ofRobotics Research, 40(12-14):1510–1546, 2021.

[14] J. Wald, H. Dhamo, N. Navab, and F. Tombari. Learning 3d semantic scene graphs from 3d indoor reconstructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3961–3970, 2020.

[15] M. F. Ginting, S.-K. Kim, D. D. Fan, M. Palieri, M. J. Kochenderfer, and A.-a. Agha-Mohammadi. Seek: Semantic reasoning for object goal navigation in real world inspection tasks. arXiv preprint arXiv:2405.09822, 2024.

[16] S. Peng, K. Genova, C. Jiang, A. Tagliasacchi, M. Pollefeys, T. Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 815–824, 2023.

[17] A. Takmaz, E. Fedele, R. W. Sumner, M. Pollefeys, F. Tombari, and F. Engelmann. Openmask3d: Open-vocabulary 3d instance segmentation. arXiv preprint arXiv:2306.13631, 2023.

[18] A. Anwar, J. Welsh, J. Biswas, S. Pouya, and Y. Chang. Remembr: Building and reasoning over long-horizon spatio-temporal memory for robot navigation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 2838–2845. IEEE, 2025.

[19] Y. Yang, H. Yang, J. Zhou, P. Chen, H. Zhang, Y. Du, and C. Gan. 3d-mem: 3d scene memory for embodied exploration and reasoning. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 17294–17303, 2025.

[20] M. F. Ginting, D.-K. Kim, X. Meng, A. Reinke, B. J. Krishna, N. Kayhani, O. Peltzer, D. D. Fan, A. Shaban, S.-K. Kim, et al. Enter the mind palace: Reasoning and planning for long-term active embodied question answering. arXiv preprint arXiv:2507.12846, 2025.

[21] J. Yang, X. Chen, S. Qian, N. Madaan, M. Iyengar, D. F. Fouhey, and J. Chai. Llm-grounder: Open-vocabulary 3d visual grounding with large language model as an agent. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 7694–7701. IEEE, 2024.

[22] M. A. Bravo, S. Mittal, S. Ging, and T. Brox. Open-vocabulary attribute detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 7041–7050, 2023.

[23] J. Ahn, H. Yun, D. Ko, and G. Kim. Can llms deceive clip? benchmarking adversarial compositionality of pre-trained multimodal representation via text updates. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 26382–26402, 2025.

[24] Y. Zhao, J. Lin, S. Ye, Q. Pang, and R. W. Lau. Openscan: A benchmark for generalized open-vocabulary 3d scene understanding. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 13289–13296, 2026.

[25] D. Koishigarina, A. Uselis, and S. J. Oh. Clip behaves like a bag-of-words model cross-modally but not uni-modally. arXiv preprint arXiv:2502.03566, 2025.

[26] M. Khanna, Y. Mao, H. Jiang, S. Haresh, B. Shacklett, D. Batra, A. Clegg, E. Undersander, A. X. Chang, and M. Savva. Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16384–16393, 2024.

[27] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[28] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

[29] S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff. C-pack: Packaged resources to advance general chinese embedding, 2023.

## A Probe Architecture and Training

## A.1 Architecture

For each attribute $a \in V .$ , FRAME trains an independent two-hidden-layer MLP probe $\phi _ { a }$ with hidden sizes 256 and 128, ReLU activations, and dropout $p = 0 . 3 / 0 . 2$ after the two hidden layers. The probe maps an object representation $\mathbf { x } ( o )$ to a scalar logit, which is passed through a sigmoid at inference to produce

$$
s _ { a } ( o ) = \sigma ( \phi _ { a } ( \mathbf { x } ( o ) ) ) \in [ 0 , 1 ] .
$$

The input dimension is 1152 for FRAME-Vis and 2176 for FRAME-VT. FRAME-VT concatenates separately $\ell _ { 2 } \cdot$ -normalized 1152-dimensional SigLIP visual and 1024-dimensional BGE caption embeddings. The probes are independent across attributes and use neither parameter sharing nor auxiliary losses.

## A.2 Training procedure

For each attribute, we construct a binary classification problem over the asset-disjoint training pool. Positive examples are training objects carrying the attribute label, and negatives are the remaining training objects. We handle label imbalance using a positive-class weight in the binary cross-entropy loss:

$$
\mathcal { L } _ { a } = - \left[ w _ { + } y \log \sigma ( \phi _ { a } ( \mathbf { x } ) ) + ( 1 - y ) \log \left( 1 - \sigma ( \phi _ { a } ( \mathbf { x } ) ) \right) \right] , \qquad w _ { + } = N _ { \mathrm { n e g } } / N _ { \mathrm { p o s } } .
$$

Table 5 summarizes the fixed training configuration.

Table 5: Probe training details.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Training pool</td><td>Asset-disjoint training objects</td></tr><tr><td>Loss</td><td>BCE with logits Positive weight</td></tr><tr><td>Class imbalance</td><td> $w _ { + } = N _ { \mathrm { n e g } } / N _ { \mathrm { p o s } }$ </td></tr><tr><td>Optimizer</td><td>Adam  $1 0 ^ { - 3 }$ </td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td></td></tr><tr><td>Schedule Batch size</td><td>Cosine annealing, 80 epochs 512</td></tr><tr><td>Probe inclusion</td><td></td></tr><tr><td>Hardware</td><td>≥ 30 positive training objects NVIDIA RTX A6000 GPUs</td></tr></table>

All 47 object attributes used by the final protocol pass the inclusion threshold; the least frequent retained attribute has 66 positive training objects.

## A.3 Validation reporting

Each probe is trained for a fixed 80 epochs, and the final-epoch checkpoint is used for retrieval evaluation. We report per-attribute validation AP to characterize readout quality, but do not use validation AP for checkpoint selection. Composition-level metrics such as P@1, Hit@5, MRR, and mAP are computed only when evaluating retrieval.

## A.4 Per-attribute validation AP

Table 6 reports validation AP for all 47 attribute probes under both input variants. The grouped summary in Table 4 combines fine- and super-category probes as Category, and roughness, metallicity, and articulatability as Other properties. Material is the weakest family, while the remaining attribute families are substantially stronger. FRAME-VT has lower family-mean AP than FRAME-Vis for most attribute families, consistent with FRAME-Vis achieving the strongest overall retrieval performance.

Table 6: Per-attribute validation AP for FRAME-Vis and FRAME-VT, grouped by family. $\Delta =$ $\mathrm { A P _ { V T } - A P _ { V i s } . }$
<table><tr><td>Family</td><td>Attribute</td><td> $\operatorname { A P } _ { \operatorname { V i s } }$ </td><td> $\operatorname { A P } _ { \operatorname { V T } }$ </td><td> $\Delta$ </td><td> $N _ { \mathrm { v a l } } ^ { + }$ </td></tr><tr><td colspan="2">Fine category (19)</td><td>0.883</td><td>0.889</td><td>+0.006</td><td></td></tr><tr><td></td><td>cat_bed</td><td>1.000</td><td>1.000</td><td>+0.000</td><td>54</td></tr><tr><td></td><td>cat_couch</td><td>0.993</td><td>0.995</td><td>+0.002</td><td>53</td></tr><tr><td></td><td>cat_chair</td><td>0.991</td><td>0.993</td><td>+0.001</td><td>102</td></tr><tr><td></td><td>cat_potted_plant</td><td>0.993</td><td>0.990</td><td>-0.003</td><td>46</td></tr><tr><td></td><td>cat_picture</td><td>0.979</td><td>0.977</td><td>-0.002</td><td>172</td></tr><tr><td></td><td>cat_ceiling_lamp</td><td>0.979</td><td>0.971</td><td>-0.007</td><td>77</td></tr><tr><td></td><td>cat_carpet</td><td>0.949</td><td>0.953</td><td>+0.004</td><td>122</td></tr><tr><td></td><td>cat_curtain</td><td>0.946</td><td>0.947</td><td>+0.001</td><td>39</td></tr><tr><td></td><td>cat_table_lamp</td><td>0.953</td><td>0.945</td><td>-0.008</td><td>43</td></tr><tr><td></td><td>cat_floor_lamp</td><td>0.910</td><td>0.931</td><td>+0.021</td><td>34</td></tr><tr><td></td><td>cat_table</td><td>0.909</td><td>0.925</td><td>+0.015</td><td>120</td></tr><tr><td></td><td>cat_mirror</td><td>0.899</td><td>0.900</td><td>+0.001</td><td>50</td></tr><tr><td></td><td>cat_stool</td><td>0.822</td><td>0.895</td><td>+0.073</td><td>13</td></tr><tr><td></td><td>cat_chest_of_drawers</td><td>0.897</td><td>0.860</td><td>-0.037</td><td>63</td></tr><tr><td></td><td>cat_wardrobe</td><td>0.834</td><td>0.854</td><td>+0.019</td><td>32</td></tr><tr><td></td><td>cat_cabinet</td><td>0.790</td><td>0.805</td><td>+0.015</td><td>79</td></tr><tr><td></td><td>cat_shelves</td><td>0.736</td><td>0.748</td><td>+0.012</td><td>55 16</td></tr><tr><td></td><td>cat_wall_lamp</td><td>0.733</td><td>0.700</td><td>-0.033</td><td></td></tr><tr><td></td><td>cat_counter</td><td>0.468</td><td>0.501</td><td>+0.033</td><td>20</td></tr><tr><td colspan="2">Super category (11)</td><td>0.955</td><td>0.952</td><td>-0.003</td><td></td></tr><tr><td></td><td>super_sleeping_furniture</td><td>1.000</td><td>1.000</td><td>+0.000</td><td>54</td></tr><tr><td></td><td>super_seating_furniture</td><td>0.991</td><td>0.991</td><td>-0.000</td><td>183</td></tr><tr><td></td><td>super_plant</td><td>0.996</td><td>0.988</td><td>-0.008</td><td>47</td></tr><tr><td></td><td>super_decor</td><td>0.980</td><td>0.977</td><td>-0.003</td><td>172</td></tr><tr><td></td><td>super_lighting</td><td>0.980</td><td>0.977</td><td>-0.003</td><td>171</td></tr><tr><td></td><td>super_floor_covering</td><td>0.951</td><td>0.956</td><td>+0.005</td><td>122</td></tr><tr><td></td><td>super_curtain</td><td>0.946</td><td>0.948</td><td>+0.002</td><td>39</td></tr><tr><td></td><td>super_bathroom_fixtures</td><td>0.940</td><td>0.920</td><td>-0.019</td><td>39</td></tr><tr><td></td><td>super_storage_furniture</td><td>0.913</td><td>0.912</td><td>-0.001</td><td>249</td></tr><tr><td></td><td>super_support_furniture</td><td>0.902</td><td>0.901</td><td>-0.001</td><td>135</td></tr><tr><td></td><td>super_mirror</td><td>0.902</td><td>0.896</td><td>-0.006</td><td>50</td></tr><tr><td colspan="2">Material (7)</td><td>0.505</td><td>0.465</td><td>-0.040</td><td></td></tr><tr><td></td><td>mat_metal</td><td>0.612</td><td>0.581</td><td>-0.031</td><td>433</td></tr><tr><td></td><td>mat_mirror</td><td>0.620</td><td>0.568</td><td>-0.052</td><td>53</td></tr><tr><td></td><td>mat_wood</td><td>0.583</td><td>0.546</td><td>-0.037</td><td>262</td></tr><tr><td></td><td>mat_rug</td><td>0.489</td><td>0.407</td><td>-0.083</td><td>50</td></tr><tr><td></td><td>mat_ceramic</td><td>0.493</td><td>0.394</td><td>-0.099</td><td>30</td></tr><tr><td></td><td>mat_leather</td><td>0.378</td><td>0.387</td><td>+0.009</td><td>52</td></tr><tr><td></td><td>mat_fabric</td><td>0.361</td><td>0.373</td><td>+0.011</td><td>136</td></tr><tr><td colspan="2">Size (6)</td><td>0.877</td><td>0.863</td><td>-0.013</td><td></td></tr><tr><td></td><td>size_large</td><td>0.926</td><td>0.915</td><td>-0.011</td><td>541</td></tr><tr><td></td><td>size_low</td><td>0.904</td><td>0.893</td><td>-0.011</td><td>177</td></tr><tr><td></td><td>size_wide</td><td>0.898</td><td>0.884</td><td>-0.014</td><td>437</td></tr><tr><td></td><td>size_small</td><td>0.870</td><td>0.861</td><td>-0.010</td><td>529</td></tr><tr><td></td><td>size_tall</td><td>0.868</td><td>0.853</td><td>-0.014</td><td>297</td></tr><tr><td></td><td>size_medium</td><td>0.795</td><td>0.774</td><td>-0.021</td><td>649</td></tr><tr><td colspan="2">Roughness (2)</td><td>0.734</td><td>0.703</td><td>-0.032</td><td></td></tr><tr><td></td><td>roughness_matte</td><td>0.816</td><td>0.784</td><td>-0.032</td><td>577</td></tr><tr><td></td><td>roughness_glossy</td><td>0.652</td><td>0.621</td><td>-0.032</td><td>346</td></tr><tr><td>Metallicity (1)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>metallic</td><td>0.944 0.944</td><td>0.930 0.930</td><td>-0.013 -0.013</td><td>1177</td></tr><tr><td>Articulatability (1)</td><td></td><td>0.949</td><td>0.950</td><td>+0.001</td><td></td></tr><tr><td colspan="2">articulatable</td><td>0.949</td><td>0.950</td><td>+0.001</td><td>287</td></tr></table>

## B Query Construction, Decomposition, and Baseline Details

## B.1 FRAME query decomposition

FRAME makes one scene-independent LLM call per query and supplies only the fixed 47-attribute vocabulary and the query text. The output is projected onto the listed vocabulary before retrieval, so invalid keys and room-association attributes cannot enter the object score.

Model. gpt-4o-mini, temperature=0, response\_forma $\mathsf { t } { = } \{ " \mathsf { t y p e } ^ { \mathfrak { n } } : " \}$ son\_object"}.

```prolog
FRAME query decomposition — prompt
You decompose object retrieval queries into attribute weights.
Vocabulary (47 attributes, grouped by type):
Object category:
cat_chair, cat_table, cat_picture, cat_couch, cat_cabinet, cat_ceiling_lamp,
cat_chest_of_drawers, cat_mirror, cat_bed, cat_potted_plant, cat_shelves,
cat_table_lamp, cat_stool, cat_curtain, cat_counter, cat_wardrobe,
cat_floor_lamp, cat_wall_lamp, cat_carpet
Super-category:
super_storage_furniture, super_seating_furniture, super_decor, super_lighting,
super_support_furniture, super_floor_covering, super_sleeping_furniture,
super_plant, super_mirror, super_curtain, super_bathroom_fixtures
Size:
size_small, size_medium, size_large, size_tall, size_low, size_wide
Material:
mat_metal, mat_wood, mat_fabric, mat_mirror, mat_leather, mat_rug, mat_ceramic
Roughness/PBR:
roughness_matte, roughness_glossy, metallic
Articulation:
articulatable
For the query, output JSON with attribute weights in [0, 1]:
- 1.0 = required (must have)
- 0.7-0.9 = strongly preferred
- 0.3-0.5 = weakly preferred
- omit attributes that are irrelevant
Strict schema rules:
- Use only attribute names written verbatim in the vocabulary above.
- Use metallic, not roughness_metallic.
- Use super_bathroom_fixtures, not cat_bathroom_fixtures.
- Return only constraints stated by the query or unambiguously entailed by it.
- Do not add plausible alternatives, typical materials, sizes, or example categories.
- A generic request for seating furniture maps to super_seating_furniture, not to a list of subtypes.
- The request is conjunctive; do not turn one constraint into alternatives.
- Omit any attribute that is not confidently required.
Query: "{query}"
Output JSON: {"weights": {"attr_name": weight, ...}, "rationale": "brief"}
```

## B.2 Paraphrase generation

This procedure is run once at benchmark-construction time. Each of the 60 active compositions is expanded into 10 natural-language requests, yielding 600 queries in total. The eight C1 compositions use 10 direct requests each, whereas C2–C6 use seven direct and three lightly indirect requests per composition, producing 444 direct and 156 lightly indirect requests. The generated requests are frozen and reused for all methods.

Model. gpt-4o, temperature=0.65, response\_format={"type":"json\_object"}.

Paraphrase generation — prompt   
Write natural, standalone English requests for a home robot to locate an   
object in its current scene.   
The exact target is: {description}   
Every request MUST contain this exact head noun: "{head\_noun}".   
Return exactly {n\_direct} direct requests and {n\_indirect} light-indirect requests.   
Direct requests state the attributes plainly. Light-indirect requests may paraphrase   
one or more NON-NOUN attributes using ordinary physical language (for example,   
"wooden" -> "crafted from timber", "glossy" -> "with a reflective sheen", and   
"articulatable" -> "with movable parts"). They must still require exactly every stated

property. Do not turn the object class into a broader class, a narrower subtype, or a   
functional description.   
Never add an unguaranteed fact: no colour, room/location, nearby object, owner, use   
history, object contents, construction details (doors, drawers, handles, panels),   
comparatives, or superlatives. Never say "scene", "room", "view", "here", "nearby",   
"hanging", "compartment", or "wall" unless it is in the exact head noun. {metal\_note}   
Do not mention labels, attributes,   
annotations, or a benchmark. No riddles.   
Return JSON only: {"direct": ["..."], "indirect": ["..."]}

{description} is a deterministic textual rendering of the composition, and {head\_noun} is its category or super-category noun. For C1, n\_direct=10 and n\_indirect=0; for C2–C6, they are 7 and 3, respectively. When a composition requires mat\_metal but not metallic, {metal\_note} instructs the model not to conflate the material with the metallic finish. Generated requests are accepted only if they contain the fixed head noun and pass the lexical safety checks; failed generations are retried up to three times.

## B.3 Per-object caption

Each object asset is captioned offline from its isolated object rendering. The resulting caption is stored once per asset and reused as shared object-text memory by all caption-based methods.

Model. gpt-4o-mini with image input, temperature=0.3, detail=low, and a maximum of 400 output tokens.

Per-object caption — prompt   
You are writing for a high-end furniture catalog. Describe the object in   
this image using exactly these fields, in this format:   
Type: [object category and sub-type]   
Style: [aesthetic style, era, or cultural origin]   
Materials: [primary and secondary materials, including texture and finish]   
Features: [distinguishing parts and details you can see]   
Atmosphere: [mood the object creates and ideal setting]   
Descriptors: [3-5 evocative adjectives capturing the feel]   
Rules:   
- 50-80 characters per field, full descriptive phrases (not just keywords)   
- Each object must have UNIQUE descriptions specific to what you actually see   
- Describe ONLY the object, not the background or image quality   
- Do NOT mention rendering, 3D, digital aspects, photographs, or backgrounds   
- Use rich, evocative language as if writing for a luxury furniture catalog  
The same full caption is embedded for ZS-Caption, ZS-V+C, and the FRAME-VT object representation. For the LLM-mediated baselines, short\_caption(...) extracts the Type: field: CG-LLM and BBQ Stage 1 use this short form, while BBQ Stage 2, EmbodiedRAG candidate reranking, and HOV-SG use the full caption. EmbodiedRAG additionally uses the short form for neighboring objects in each candidate’s local subgraph.

## B.4 BiEncoder

BiEncoder is a task-tuned embedding-retrieval baseline using the same frozen visual object features, asset-disjoint training pool, 43 seen compositions, frozen query forms, and evaluation harness as the other methods in our protocol.

Architecture. The query encoder is frozen BGE-large (bge-large-en-v1.5, CLS token, $\ell _ { 2 ^ { - } }$ normalized), followed by a learnable projection $g _ { q }$ . The object encoder applies a learnable projection $g _ { o }$ to the frozen 1152-dimensional visual representation ${ \bf e } _ { \mathrm { v i s } } ( o )$ . Both projections are three-layer MLPs with dimensions $d _ { \mathrm { i n } }  5 1 2  2 5 6  2 5 6$ , ReLU activations, and dropout 0.3 after each hidden layer. The projected vectors are ℓ<sub>2</sub>-normalized, and the retrieval score is

$$
\begin{array} { r } { \mathrm { s c o r e } ( o , q ) = \cos ( g _ { o } ( \mathbf { e } _ { \mathrm { v i s } } ( o ) ) , g _ { q } ( \mathrm { B G E } ( q ) ) ) . } \end{array}
$$

Training. For each seen-composition–scene unit from the asset-disjoint training pool, positives are all objects satisfying every required attribute. Negatives include at most four near-miss objects that satisfy all but one required attribute and at most four randomly sampled non-positive objects. One frozen query form is sampled for each unit in every epoch. BiEncoder is trained for a fixed 60 epochs using binary cross-entropy on the scaled cosine logit $1 0 \mathrm { s c o r e } ( o , q )$ , Adam with learning rate $1 0 ^ { - 3 }$ and weight decay $5 \times 1 0 ^ { - 4 }$ , cosine learning-rate scheduling, and batch size 1024. The final-epoch checkpoint is used for evaluation; validation P@1 is not used for checkpoint selection.

Inference. Projected object vectors are precomputed and cached. Each query requires one BGE encoding, one query projection, and one cosine matrix–vector product against the cached object matrix.

## B.5 CrossEncoder

CrossEncoder provides a task-tuned relevance baseline under the same object features, training pool, query forms, supervision, and fixed 60-epoch schedule as BiEncoder.

Architecture. CrossEncoder concatenates the frozen 1024-dimensional BGE query representation with the frozen 1152-dimensional visual object representation. A three-layer MLP with dimensions $2 1 7 6 \to 5 1 2 \to 2 5 6 \to 1$ , ReLU activations, and dropout 0.3 after each hidden layer maps the concatenation to a scalar relevance logit.

Training. CrossEncoder uses the same positive, near-miss, and random-negative sampling as Bi-Encoder and is optimized directly with binary cross-entropy on its relevance logit. The optimizer, learning-rate schedule, batch size, and final-epoch checkpoint policy are identical to those of BiEncoder.

Inference. Each query representation is concatenated with every candidate object representation and scored by the MLP. Consequently, unlike BiEncoder, CrossEncoder produces a query-dependent object score and does not admit a standalone cached object retrieval vector.

## B.6 FRAME-ZS attribute prompts

FRAME-ZS uses the same strict decomposition and weighted composition interface as FRAME-Vis, but replaces learned probes with SigLIP text–image cosine scores. The performance evaluation uses deterministic type-dependent templates rather than a hand-written per-attribute prompt dictionary. Fine categories use a photo of a/an <category>; coarse categories use the corresponding singular object noun; materials use a photo of an object made of <material>; and size and physical-property attributes use a photo of a/an <attribute> object. No room-association prompt is used. For decomposed weights $\{ w _ { a } \}$ , FRAME-ZS scores object o by

$$
\mathrm { s c o r e } ( o , q ) = \frac { 1 } { \sum _ { a } w _ { a } } \sum _ { a } w _ { a } \cos ( { \mathbf { e } } _ { \mathrm { v i s } } ( o ) , \mathrm { S i g L I P } _ { \mathrm { t e x t } } ( \tau ( a ) ) ) .
$$

## B.7 CG-LLM

Original framework. ConceptGraphs [2] constructs an open-vocabulary 3D scene graph from RGB-D observations through object detection, tracking, and inter-object relation extraction, and uses LLM-based planning over the resulting scene graph for tasks such as task planning, question answering, and embodied action.

Our adaptation. We isolate the LLM grounding component: the scene graph and object features are provided by our shared protocol (§3), and a single LLM call ranks objects from each object’s short caption and up to three scene-graph neighbors. The full task-planning and structured-reasoning pipeline of the original is replaced with a ranked-list output. gpt-4o-mini replaces GPT-4.

Model. gpt-4o-mini, temperature=0, max 200 output tokens. TOP\_K = 20.

CG-LLM — prompt   
You are an object retrieval assistant working with a 3D scene graph.   
Scene has {N} objects, each with a short type and scene-graph neighbors:   
1. {short\_caption\_1} [neighbors: {rel\_1} a {nbr\_type\_1}; ...]   
2. {short\_caption\_2} [neighbors: ...]   
N. {short\_caption\_N} [neighbors: ...]   
User wants: "{query}"   
Return the top {TOP\_K} object indices most likely to match, ranked best->worst.   
Output ONLY comma-separated 1-based indices.

## B.8 BBQ

Original framework. BBQ (Beyond Bare Queries) [8] constructs its own object-centric 3D scene graph from RGB-D scans, then performs grounding via a coarse caption-based filter followed by a fine reranking step with full captions, 3D positions, and graph edges. The framework additionally supports spatial-anchor queries (“X near Y”) through an anchor-resolution module.

Our adaptation. The scene graph and object features are provided by our shared protocol; the two-stage filter-and-rerank pipeline is preserved. The anchor-resolution logic is present in the released code but not exercised here, since our queries are attribute-compositional rather than spatial-anchor.

Model. gpt-4o-mini, temperature=0. STAGE1\_MAX\_KEEP = 15. Stage 1 max 150 tokens;   
Stage 2 max 300.

## BBQ — Stage 1 prompt

Scene with {N} objects (short types):   
1. {short\_caption\_1}   
2. {short\_caption\_2}   
N. {short\_caption\_N}   
User wants: "{query}"   
Select up to {STAGE1\_MAX\_KEEP} candidate indices likely to match.   
Output ONLY comma-separated 1-based indices.

## BBQ — Stage 2 prompt

Query: "{query}"   
Candidates with full captions, 3D positions, and scene-graph edges   
(with reasoning):   
1. {full\_caption\_1}   
Position: ({cx},{cy},{cz})/extent({ex},{ey},{ez})   
Edges: {rel\_1} a {tag\_1} --- {reasoning\_1[:60]}; ...   
2.   
K. ...   
Rank ALL {K} candidates by relevance, best first.   
Output ONLY comma-separated 1-based indices (include all).

## B.9 EmbodiedRAG

Original framework. EmbodiedRAG [9] dynamically retrieves query-relevant 3D scene subgraphs from a continuously updated scene graph to augment an LLM-based planner for robot task execution. The retrieval reduces input token counts so the planner can operate at scale; the evaluation target is task success in simulated household environments.

Our adaptation. The scene graph is provided by our shared (static) protocol. The retrieval unit is reduced from subgraph to object node, and the LLM step is changed from planner to object ranker. Top-K object candidates are first retrieved by SigLIP visual–text cosine, then the LLM reranks them using each candidate’s full caption and a 1-hop subgraph—preserving the local-subgraph reasoning that motivates the original retrieval.

Model. gpt-4o-mini, temperature=0, max 200 tokens. TOPK = 20, NEIGHBORS = 3 per candidate.

EmbodiedRAG — prompt   
Query: "{query}"   
Top {K} candidates with their local subgraphs (scene-graph neighbors):   
1. {full\_caption\_1}   
Local subgraph: {rel\_1} {nbr\_type\_1}; {rel\_2} {nbr\_type\_2}; ...   
2. ...   
...   
K. ...   
Rank ALL {K} candidates by relevance, best first.   
Output ONLY comma-separated 1-based indices.

## B.10 HOV-SG

Original framework. HOV-SG [3] builds a hierarchical open-vocabulary 3D scene graph organized as floor → region → object, and uses this multi-level structure for language-grounded robot navigation by decomposing queries into floor / room / category / attribute components and traversing the hierarchy.

Our adaptation. The scene graph and object features are provided by our shared protocol. Since HSSD scenes are single-floor residential environments, the original floor–region–object hierarchy is reduced to room–object. The two-step pipeline (LLM decomposition into {room, category, attributes}, followed by reranking on room-and-category-filtered candidates) is preserved. The navigation-execution component of the original is not used; we only evaluate object-grounding ranking.

Model. gpt-4o-mini, temperature=0. Decompose max 100 tokens; rerank max 250.   
TOPK\_CAT = 30.

HOV-SG — Step 1 decomposition prompt   
Decompose this object query into structured fields.   
Query: "{query}"   
Output JSON with three fields:   
- room: the most likely room (bedroom, bathroom, kitchen, living\_room,   
dining\_room, office, kids\_room, or "any" if unspecified)   
- category: the main object class (chair, lamp, table, ...)   
- attributes: list of descriptive words (e.g., ["wooden", "small"])   
Output ONLY valid JSON. Example:   
{"room": "bedroom", "category": "bed", "attributes": ["large"]}

HOV-SG — Step 2 reranking prompt   
Original query: "{query}"   
Hierarchical decomposition: room={room}, category={category},   
attributes={attrs\_csv}   
Candidates (already filtered by room + category):   
1. {full\_caption\_1}   
2. {full\_caption\_2}   
K. {full\_caption\_K}   
Rank ALL {K} candidates by relevance to the original query, best first.   
Output ONLY comma-separated 1-based indices.

## C Active Attribute Compositions

Table 7 lists the 60 active compositions in the final protocol. A dagger denotes one of the 17 unseen test compositions; the remaining 43 compositions provide training supervision and are evaluated on the validation scenes. Every composition has 10 frozen query forms.

Table 7: The 60 active attribute compositions. † denotes an unseen test composition.  
Class Active compositions   
C1: Fine cat. (n = 8) cabinet, carpet, ceiling\_lamp<sup>†</sup>, chair<sup>†</sup>, couch, mirror, potted\_plant, table<sup>†</sup>   
C2: Fine cat. + material (n = 7) fabric\_chair<sup>†</sup>, leather\_chair, metal\_ceiling\_lamp, metal\_chair, metal\_table,   
wood\_shelves, wood\_table   
C3: Fine cat. + size (n = 12) large\_cabinet, large\_chair, large\_potted\_plant, large\_table, medium\_carpet,   
medium\_ceiling\_lamp<sup>†</sup>, medium\_picture, medium\_potted\_plant, medium\_table,   
tall\_cabinet, tall\_potted\_plant, wide\_table   
C4: Fine cat. + other property (n = 4) glossy\_table<sup>†</sup>, matte\_table, metallic\_cabinet, metallic\_ceiling\_lamp   
C5: Super cat. + one property (n = 19) ceramic\_bathroom\_fixtures, fabric\_seating\_furniture,   
glossy\_storage\_furniture<sup>†</sup>, large\_plant, large\_seating\_furniture,   
large\_storage\_furniture<sup>†</sup>, matte\_storage\_furniture, medium\_plant,   
medium\_seating\_furniture, medium\_storage\_furniture<sup>†</sup>,   
metal\_bathroom\_fixtures, metal\_seating\_furniture, metal\_storage\_furniture<sup>†</sup>,   
tall\_plant, tall\_storage\_furniture<sup>†</sup>, wide\_seating\_furniture,   
wide\_storage\_furniture<sup>†</sup>, wood\_seating\_furniture, wood\_storage\_furniture<sup>†</sup>   
C6: Three attributes (n = 10) large\_glossy\_storage\_furniture, large\_metal\_storage\_furniture<sup>†</sup>,   
medium\_fabric\_seating\_furniture, medium\_glossy\_storage\_furniture<sup>†</sup>,   
medium\_metal\_storage\_furniture, medium\_wood\_storage\_furniture,   
tall\_glossy\_storage\_furniture, tall\_metal\_storage\_furniture<sup>†</sup>,   
wide\_metal\_storage\_furniture, wide\_wood\_storage\_furniture

## D Full Results

## D.1 Full test metric breakdown

Table 8 reports all ranking metrics on the 17 unseen compositions and 26 held-out test scenes. We do not mix these results with validation performance: validation scenes contain the 43 seen compositions and are used only for development diagnostics and the readout-quality analysis above.

## D.2 Per-class test P@1

Table 9 reports P@1 for all methods under the six composition classes. FRAME-Vis achieves the highest P@1 in five of the six classes; BBQ is highest on C3.

Table 8: Retrieval performance on 3,720 valid query–scene pairs from the unseen-composition test split. Values are percentages. Best results are in bold, and the best non-FRAME result is underlined.
<table><tr><td>Method</td><td>P@1 Hit@5</td><td>MRR</td><td>mAP</td></tr><tr><td>ZS-Visual</td><td>46.40</td><td>83.92 62.22</td><td>51.57</td></tr><tr><td>ZS-Caption</td><td>30.16</td><td>72.58 48.33</td><td>37.79</td></tr><tr><td>ZS-V+C</td><td>46.77</td><td>82.66 61.60</td><td>49.19</td></tr><tr><td>BiEncoder</td><td>48.87</td><td>87.15 65.13</td><td>54.44</td></tr><tr><td>CrossEncoder</td><td>62.58</td><td>93.79 75.50</td><td>66.06</td></tr><tr><td>CG-LLM</td><td>27.23</td><td>66.53 43.96</td><td>29.66</td></tr><tr><td>BBQ</td><td>41.21</td><td>75.13 55.17</td><td>38.11</td></tr><tr><td>EmbodiedRAG</td><td>44.57</td><td>78.68 59.10</td><td>48.20</td></tr><tr><td>HOV-SG</td><td>42.02</td><td>77.88 57.50</td><td>45.64</td></tr><tr><td>FRAME-ZS</td><td>35.67</td><td>68.20 50.53</td><td>40.65</td></tr><tr><td>FRAME-Vis</td><td>77.82</td><td>96.21 86.01</td><td>78.30</td></tr><tr><td>FRAME-VT</td><td>75.75</td><td>95.97 84.62</td><td>76.28</td></tr></table>

Table 9: Per-class test P@1 (%) on the unseen-composition split. Best results are in bold, and the best non-FRAME result in each class is underlined.
<table><tr><td>Method</td><td>C1</td><td>C2</td><td>C3</td><td>C4</td><td>C5</td><td>C6</td></tr><tr><td>ZS-Visual</td><td>78.53</td><td>59.38</td><td>65.26</td><td>38.00</td><td>39.35</td><td>24.25</td></tr><tr><td>ZS-Caption</td><td>48.27</td><td>25.62</td><td>23.68</td><td>15.00</td><td>31.30</td><td>15.75</td></tr><tr><td>ZS-V+C</td><td>81.20</td><td>48.12</td><td>66.32</td><td>32.50</td><td>40.71</td><td>23.97</td></tr><tr><td>BiEncoder</td><td>88.53</td><td>35.62</td><td>68.42</td><td>13.00</td><td>42.84</td><td>29.73</td></tr><tr><td>CrossEncoder</td><td>95.47</td><td>50.62</td><td>60.00</td><td>44.00</td><td>62.01</td><td>38.49</td></tr><tr><td>CG-LLM</td><td>37.73</td><td>30.00</td><td>43.16</td><td>6.00</td><td>27.93</td><td>15.89</td></tr><tr><td>BBQ</td><td>68.93</td><td>41.25</td><td>77.89</td><td>4.50</td><td>38.82</td><td>18.77</td></tr><tr><td>EmbodiedRAG</td><td>87.07</td><td>53.12</td><td>72.63</td><td>11.50</td><td>38.11</td><td>15.75</td></tr><tr><td>HOV-SG</td><td>84.40</td><td>48.75</td><td>53.68</td><td>6.50</td><td>36.45</td><td>16.58</td></tr><tr><td>FRAME-ZS</td><td>87.33</td><td>37.50</td><td>48.95</td><td>2.00</td><td>29.41</td><td>2.47</td></tr><tr><td>FRAME-Vis</td><td>98.67</td><td>75.00</td><td>73.68</td><td>53.50</td><td>79.64</td><td>60.55</td></tr><tr><td>FRAME-VT</td><td>97.73</td><td>62.50</td><td>73.68</td><td>53.00</td><td>78.34</td><td>56.85</td></tr></table>

## E Real-scene Attribute Readouts on 3DSSG

Our main protocol fixes perception and scene-graph construction in order to isolate the access layer of object-centric grounding: given a fixed scene memory, how well can attribute-compositional language retrieve object nodes? As a small real-scene sanity check, we evaluate learned attribute readouts on 3DSSG [14], using object crops extracted from 20 real-scene scans. This experiment i not intended as a full end-to-end real-scene grounding benchmark; rather, it tests whether the same attribute-readout architecture can extract useful signal from noisier real-scene object crops.

Setup. We select 20 3DSSG scenes with relatively rich attribute coverage and use a fixed stratified split of 12/4/4 scenes for train/validation/test. The resulting split contains 1,204 object instances in total, with 739/217/248 objects in train/validation/test. We map 3DSSG object and attribute annotations into the 47-object-attribute vocabulary used by the final HSSD protocol; 21 attributes have at least 10 positive training examples and are retained for probe training. For each oriented bounding-box instance, we extract an RGB crop by selecting the best available view according to visibility and Laplacian sharpness over the captured frames. We encode each crop with the same SigLIP-so400m visual encoder used in the main experiments and train one MLP probe per attribute using the same probe architecture as Appendix A. We report mean and standard deviation over five training seeds; the split is fixed, so the seed varies probe initialization and minibatch order.

Family-level results. Table 10 summarizes test mean AP by attribute family. All families are above their random baselines, but the performance pattern is uneven. Super-category, material, and category readouts retain meaningful real-domain signal, while size readouts are close to random.

Table 10: 3DSSG crop-level sanity check: family-level test mean AP over retained probes. Mean and standard deviation are computed over five training seeds.
<table><tr><td>Family</td><td> $\mathrm { m A P }$ </td><td>Std.</td><td># probes</td><td>Random AP</td></tr><tr><td>Super-category</td><td>0.555</td><td>0.024</td><td>6</td><td>0.062</td></tr><tr><td>Material</td><td>0.494</td><td>0.052</td><td>3</td><td>0.078</td></tr><tr><td>Category</td><td>0.456</td><td>0.009</td><td>7</td><td>0.046</td></tr><tr><td>Size</td><td>0.132</td><td>0.005</td><td>5</td><td>0.060</td></tr></table>

Per-attribute pattern. The result is mixed but informative. Several readouts transfer well to realscene crops, especially attributes with distinctive visual signatures: super\_plant reaches 0.884 AP, cat\_potted\_plant 0.893, super\_lighting 0.770, mat\_fabric 0.684, and cat\_table 0.673. Other readouts degrade substantially, including size\_small 0.054, size\_large 0.083, size\_wide 0.122, cat\_couch 0.202, and cat\_cabinet 0.207. Low-positive-count attributes also show higher variance, for example mat\_metal and super\_decor, indicating that part of the instability comes from the small real-scene test pool.

Table 11: Selected 3DSSG per-probe test AP values, showing representative robust and weak readouts.
<table><tr><td>Probe</td><td>AP</td><td>Std.</td><td>Train  $N ^ { + }$ </td><td>Test  $N ^ { + }$ </td></tr><tr><td>Robust readouts</td><td></td><td></td><td></td><td></td></tr><tr><td>cat_potted_plant</td><td>0.893</td><td>0.030</td><td>44</td><td>7</td></tr><tr><td>super_plant</td><td>0.884</td><td>0.003</td><td>44</td><td>7</td></tr><tr><td>super_lighting</td><td>0.770</td><td>0.032</td><td>18</td><td>7</td></tr><tr><td>mat_fabric</td><td>0.684</td><td>0.032</td><td>104</td><td>35</td></tr><tr><td>cat_table</td><td>0.673</td><td>0.026</td><td>44</td><td>17</td></tr><tr><td>Weaker readouts</td><td></td><td></td><td></td><td></td></tr><tr><td>cat_cabinet</td><td>0.207</td><td>0.022</td><td>15</td><td>6</td></tr><tr><td>cat_couch</td><td>0.202</td><td>0.025</td><td>19</td><td>9</td></tr><tr><td>size_wide</td><td>0.122</td><td>0.020</td><td>45</td><td>12</td></tr><tr><td>size_large</td><td>0.083</td><td>0.011</td><td>19</td><td>5</td></tr><tr><td>size_small</td><td>0.054</td><td>0.005</td><td>17</td><td>5</td></tr></table>

Two layers of failure. We interpret the mixed result as evidence for two compounding bottlenecks. First, real-scene object-memory construction introduces substantial upstream noise: crops may be partial, distant, oblique, poorly segmented, or contaminated by background. Even with best-view selection, some objects lack a clean near-frontal crop, which weakens downstream attribute readout. Second, some attributes are intrinsically difficult to infer from a single real-scene RGB crop. Material cues vary with illumination, viewpoint, sensor response, and background contamination, while absolute object size is poorly determined without depth or camera calibration. This explains why the size\_\* probes remain close to random even when semantic or visually distinctive readouts are informative.

Implication. This sanity check does not establish deployment-ready real-scene grounding. Instead, it shows that real-domain failure is not uniform: some learned readouts retain clear signal on realscene crops, while others remain weak under crop noise and limited real-scene supervision. We therefore view full real-scene deployment as requiring better object-memory construction—including segmentation, view selection, occlusion handling, and object representation—while preserving the value of attribute readouts as a scene-memory access mechanism.