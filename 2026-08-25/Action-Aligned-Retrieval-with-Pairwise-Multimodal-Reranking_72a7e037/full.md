# Action-Aligned Retrieval with Pairwise Multimodal Reranking for Text-Based Person Anomaly Search

Thanh-Khoi Nguyen<sup>1,2†</sup> , Thanh-Nhan Vo<sup>1,2†</sup> , Trong-Thuan Nguyen<sup>1,2</sup> , and Minh-Triet Tran<sup>1,2</sup>

<sup>1</sup> University of Science, VNU-HCM, Vietnam <sup>2</sup> Vietnam National University, Ho Chi Minh City, Vietnam {ntkhoi,vtnhan,ntthuan}@selab.hcmus.edu.vn tmtriet@fit.hcmus.edu.vn <sup>†</sup>These authors contributed equally.

Abstract. Text-based person anomaly search requires distinguishing individuals based on fine-grained, context-dependent behaviors rather than mere appearance. Existing methods struggle to capture these contextconditioned actions, frequently relying on isolated skeletal geometry, discarding raw query details during reformulation, or utilizing absolute pointwise scoring for multimodal verification. To address these limitations, we propose ActPair, a unified three-stage coarse-to-fine framework that combines action-aligned retrieval with pairwise multimodal reranking to bridge the pose-semantic gap. First, we fine-tune a visionlanguage model (VLM) with an action-aligned multi-task objective that encourages the representations to encode action-discriminative semantics. Second, we perform parallel late-fusion retrieval using the original query and a large language model (LLM)-generated context-grounded rewrite, retaining complementary details from both semantic views. Finally, we propose an eficient of-the-shelf reranking module that leverages a pivot-promote algorithm to perform direct pairwise visual comparisons, mitigating residual spatial and compositional ambiguities without the prohibitive inference costs of exhaustive evaluation. Extensive experiments demonstrate that our framework achieves the best results among the compared methods on the Pedestrian Anomaly Behavior (PAB) public test and and transfers efectively to an unseen, non-anomaly-specific dataset.

Keywords: Text-based person anomaly search · Vision-language retrieval · Action-aligned representation learning · Pairwise multimodal reranking

## 1 Introduction

Text-based person search retrieves individuals using natural language but is largely limited to routine activities, overlooking abnormal behaviors critical in surveillance. Recent work extends this to text-based person anomaly search, where individuals may share similar appearance and can only be distinguished by fine-grained actions described in the query. These actions are often entangled with implicit scene context, requiring representations that capture identityindependent behavior and context, rather than relying on appearance alone.

While recent advances have pushed the boundaries of text-based person retrieval, existing methods still exhibit critical limitations when applied to the highly nuanced task of person anomaly search. First, prior pose-aware frameworks (e.g., Cross-Modal Pose-aware (CMP) [23]) predominantly rely on isolated skeletal geometry as a supervisory signal for human action. However, in static image retrieval, where temporal dynamics are absent, skeletal posture is inherently ambiguous; a physical pose resolves into definitive action semantics only when explicitly coupled with its surrounding environmental context. Second, although recent methods explore query reformulation (e.g., MVR [26], ReCQR [3], PlugIR [8], MRA [22]), they typically replace the original query entirely. They lack a parallel late-fusion mechanism capable of synergizing the unadulterated fine-grained details of the raw query with a canonical, contextgrounded rewrite, thereby inadvertently discarding complementary semantic evidence. Third, global dense embeddings exhibit a well-documented vulnerability to the compositional binding problem. While highly proficient at recognizing isolated semantic concepts, they frequently fail to accurately bind these attributes to their correct subjects within a unified interaction graph. In anomaly search, distinguishing events rarely relies on the mere presence of objects, but rather on precise spatial interactions (e.g., "a man falling of a bicycle" versus "a man standing next to a fallen bicycle"). Consequently, cosine similarity metrics over compressed representations struggle with final fine-grained localization. To mitigate this, contemporary generative methods (e.g., AnomalyLMM [6], IRRA [4] reranking) introduce multimodal large language model (MLLM)-based reranking, but they typically rely on independent caption generation or absolute pointwise scoring. This isolated assessment forces models to maintain globally calibrated scales. However, it overlooks a fundamental intuition: assessing subtle visual diferences is inherently easier and more reliable when directly comparing two highly similar candidates side-by-side, rather than attempting to compute an absolute relevance score for a single image in isolation. Finally, current architectures (e.g., SSDC [19], RaSa [21], APTM [24], CAMeL [25]) remain fragmented, lacking a unified coarse-to-fine framework tailored for fine-grained person anomaly search.

To overcome these limitations, we propose ActPair, an action-aligned retrieval framework with pairwise multimodal reranking, designed as a unified three-stage coarse-to-fine pipeline to bridge the pose-semantic gap in text-based person anomaly search. First, we establish a robust dense retrieval foundation by fine-tuning a VLM via an action-aligned multi-task learning objective and cross-modal consistency to capture rich, context-conditioned action semantics. Second, to address the contextual ambiguity of free-form natural language, we employ a parallel late-fusion retrieval strategy that harmonizes the original query with an LLM-synthesized, context-grounded rewrite, maximizing candidate recall while preserving fine-grained raw details. Finally, to alleviate residual spatial and compositional ambiguities that global dense embeddings inherently struggle to capture, we utilize an eficient of-the-shelf multimodal reranking module. By using an MLLM as a direct pairwise comparator within the pivot-promote procedure, our framework performs targeted reranking over the top-ranked candidates rather than constructing a complete pairwise ordering. The procedure reduces the number of comparisons in the evaluated top-10 setting, although its worst-case complexity remains quadratic.

Our main contributions are as follows:

– We propose action-aligned multi-task representation learning for a VLM that combines contrastive image-text alignment with explicit action-tag supervision and cross-modal consistency.

– We propose context-grounded query rewriting with parallel late-fusion retrieval, enriching queries with action and scene cues while fusing complementary semantic views to improve recall.

We propose an eficient of-the-shelf pairwise multimodal reranking with a pivot-promote strategy. Under a matched Qwen3.5-9B comparison, direct candidate comparison improves R@1 and mAP over pointwise scoring.

## 2 Related Work

## 2.1 Pose-Aware and Action-Semantic Representation for Person Anomaly Search

Conventional text-based person retrieval primarily learns cross-modal representations for identity-related appearance descriptions. Recent methods strengthen fine-grained alignment through implicit visual-textual relation reasoning [4], relation-aware supervision, and sensitivity to meaningful textual variations [21]. However, they are mainly designed to distinguish identities by appearance rather than behaviors that determine normal or anomalous retrieval targets.

The PAB benchmark introduces person anomaly search, where visually similar pedestrians must be distinguished by their behaviors [23]. Its CMP framework incorporates human-pose patterns and identity-based hard negatives to capture structural behavioral variations. Nevertheless, similar skeletal configurations may correspond to diferent actions. SSDC formalizes this limitation as the pose-semantic gap and combines pose-aware retrieval with MLLM-based semantic verification [19], although semantic reasoning is applied mainly after the initial structure-guided stage.

Studies in human-object interaction and action recognition further show that action semantics depend on interacting objects and scene context, rather than body geometry alone [10, 18]. At the same time, background context must be modeled carefully to avoid spurious scene shortcuts [9]. Motivated by these observations, our approach directly optimizes the SigLIP2 embedding space [17] using explicit action supervision and cross-modal action consistency, enabling context-conditioned action representations before retrieval.

## 2.2 Multimodal Reranking for Fine-Grained Retrieval

Learning-to-rank methods construct rankings via pointwise, pairwise, or listwise relevance modeling [1, 2]. Recent language and multimodal models extend these approaches beyond dense-encoder similarity scores.

In text-based person anomaly search, SSDC verifies each retrieved image with an MLLM and fuses image-specific semantic evidence with pose-aware retrieval [19], while AnomalyLMM independently interprets candidates through masked action and color descriptions before LLM-based ranking [6]. Both process candidates independently before assessing relative relevance. Pairwise ranking prompting instead directly compares candidates, avoiding independently calibrated relevance scores [12], but exhaustive pairwise comparison incurs quadratic multimodal inference cost [29]. We therefore compare only a small top-ranked candidate pool using a pivot-promote procedure, enabling fine-grained relative reasoning without exhaustive all-pairs evaluation.

## 2.3 Query Reformulation and Multi-View Retrieval

Recent vision-language retrieval methods reformulate or enrich textual queries at inference time to reduce ambiguity. PlugIR converts dialogue-form contexts into self-contained queries and generates candidate-aware clarification questions for interactive text-to-image retrieval [8]. ReCQR similarly rewrites long or underspecified conversational histories into concise, semantically complete queries [3]. Relevance-feedback methods instead refine the query using top-ranked images, synthetic captions, or learned multimodal feedback [7].

MVR mitigates expression drift by generating keyword-preserving, semantically equivalent reformulations and aggregating their textual features. It also reformulates VLM-generated gallery descriptions for visual-semantic compensation [26]. In contrast, our method generates a single action-scene-grounded rewrite for person anomaly search while retaining the original caption as a complementary retrieval view. The rewrite emphasizes behavioral context, whereas the original preserves instance-specific details. We combine these two views through late-fusion retrieval rather than aggregating multiple reformulations.

## 3 Proposed Method

## 3.1 Problem Formulation

Let $\mathcal { G } = \{ I _ { i } \} _ { i = { \bar { \iota } } } ^ { N }$ denote an image gallery of N candidate pedestrian images, and let $Q$ be a natural language query describing a target pedestrian’s appearance and behavior. The goal of text-based person anomaly search is to retrieve the image as formulated in $\operatorname { E q } .$ . (1).

$$
I ^ { * } = \arg \operatorname* { m a x } _ { I \in \mathcal { G } } \sin ( Q , I ) ,\tag{1}
$$

such that $I ^ { * }$ depicts the pedestrian engaged in the exact normal or anomalous behavior specified by $Q .$ . We seek a ranking function as defined in Eq. (2).

$$
\mathcal { F } ( Q , \mathcal { G } )  \hat { \mathcal { G } } ,\tag{2}
$$

![](images/7ae164ace7012af2275ef15d1de919b29689298eb97073c3d3fde80f6b7e3046.jpg)  
Fig. 1: Overview of ActPair. An action-aligned dual encoder retrieves candidates from the original query and a context-grounded rewrite. Their rankings are fused before pairwise multimodal reranking with pivot-promote.

that produces a sorted candidate list $\hat { \mathcal G }$ minimizing the rank position of the ground-truth image $I ^ { * }$ . Given the scale of real-world galleries, $\mathcal { F }$ is instantiated in practice as a composition of a high-recall retrieval stage over $\mathcal { G }$ followed by a fine-grained scoring stage restricted to a small candidate subset, though the formulation above remains agnostic to this decomposition.

Overview. As illustrated in Fig. 1, ActPair consists of three stages: actionaligned representation learning, dual-view retrieval using the original and contextgrounded queries, and pairwise multimodal reranking with pivot-promote.

## 3.2 Ofline Action and Scene Tag Extraction

We utilize the action and scene tags provided by training set PAB [23] as the basis for constructing a fixed behavioral vocabulary. Let $\mathcal { R } _ { a c t }$ and $\mathcal { R } _ { s c e n e }$ denote the sets of raw, free-form action and scene tags in the training annotations, and let $c ( r )$ denote the occurrence count of a raw tag r. Owing to the heavy-tailed distribution of tag frequency, we retain only the top-K most frequent tags in each category according to Eq. (3).

$$
\mathcal { R } _ { a c t } ^ { K } = \mathrm { t o p } \cdot K _ { r \in \mathcal { R } _ { a c t } } ~ c ( r ) , ~ \mathcal { R } _ { s c e n e } ^ { K } = \mathrm { t o p } \cdot K _ { r \in \mathcal { R } _ { s c e n e } } ~ c ( r ) .\tag{3}
$$

This together accounts for the large majority of dataset occurrences while discarding rare, noisy, near-duplicate phrasings. The retained tags, together with their frequencies, are provided to a LLM $\mathcal { M } _ { t a g }$ , which induces a fixed set of $n _ { a c t }$ and $n _ { s c e n e }$ canonical categories and produces a mapping in $\operatorname { E q }$ . (4).

$$
\phi : \mathcal { R } _ { a c t } ^ { K } \cup \mathcal { R } _ { s c e n e } ^ { K } \  \ \mathcal { T } _ { a c t } \cup \mathcal { T } _ { s c e n e } ,\tag{4}
$$

Mitigating synonymy and surface-level variation between raw tags that describe the same underlying action or scene. The resulting lookup tables $\mathcal { T } _ { a c t }$ and $\mathcal { T } _ { s c e n e }$ standardize the raw annotations into a fixed vocabulary, which supplies the ground-truth labels $y _ { a c t }$ for action supervision (Sec. 3.3) and the tag vocabulary used for query rewriting (Sec. 3.4).

Joint action-aware learning

![](images/adc6d1b98779a43227941f5f6851e11f1e3bd4a1bbe39f32ba0be18d8d7c697b.jpg)  
Fig. 2: Action-aligned representation learning. Image and text embeddings are optimized by the SigLIP2 objective, shared action supervision, and cross-modal action consistency.

## 3.3 Multi-Task Representation Learning

Figure 2 illustrates the joint image-text alignment and action-supervision architecture. The efectiveness of our overall framework depends on the quality of the underlying dense retrieval model, which must produce a highly accurate initial candidate pool before any downstream contextualization or reranking can take place. Relying on an out-of-the-box VLM often yields suboptimal results, as such models can exhibit insuficient sensitivity when evaluating images with high pixel overlap (e.g., identical backgrounds and subjects), failing to distinguish subtle kinematic variations. To optimize the base SigLIP2 encoder for fine-grained anomaly retrieval, we propose a multi-task fine-tuning objective. Let $E _ { i m g }$ and $E _ { t x t }$ represent the normalized image and text embeddings, and $s ( i , j ) = E _ { i m q } ^ { ( i ) } \cdot E _ { t x t } ^ { ( j ) }$ denote their cosine similarity. The total training objective is formulated as Eq. (5).

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { s i g l i p } + \lambda _ { a c t i o n } \mathcal { L } _ { a c t i o n }\tag{5}
$$

where $\mathcal { L } _ { s i g l i p }$ is the native SigLIP2 pairwise sigmoid loss utilized to maintain global, memory-eficient image-text alignment across the batch.

Action Supervision and Semantic Consistency: To reduce the visual encoder’s reliance on dominant background cues, we propose an auxiliary classification head over predefined action tags. This encourages the model to allocate representational capacity to action-discriminative visual evidence beyond scene context alone. Let $p _ { \mathrm { i m g } } = \mathrm { s o f t m a x } ( W _ { \mathrm { a c t } } E _ { \mathrm { i m g } } )$ and $p _ { \mathrm { t x t } } = \mathrm { s o f t m a x } ( W _ { \mathrm { a c t } } E _ { \mathrm { t x t } } )$ denote the predicted distributions over the action-tag set $\mathcal { T } _ { \mathrm { a c t } }$ , produced from the image and text embeddings by the shared classification head $W _ { \mathrm { a c t } }$ . We define the action supervision objective as Eq. (6b).

$$
{ \mathcal { L } } _ { \mathrm { c o n s } } = { \frac { 1 } { 2 } } \left[ D _ { \mathrm { K L } } ( p _ { \mathrm { i m g } } \parallel p _ { \mathrm { t x t } } ) + D _ { \mathrm { K L } } ( p _ { \mathrm { t x t } } \parallel p _ { \mathrm { i m g } } ) \right] ,\tag{6a}
$$

$$
\mathcal { L } _ { \mathrm { a c t i o n } } = \frac { 1 } { 2 } \left[ \mathrm { C E } ( p _ { \mathrm { i m g } } , y _ { \mathrm { a c t } } ) + \mathrm { C E } ( p _ { \mathrm { t x t } } , y _ { \mathrm { a c t } } ) \right] + \mathcal { L } _ { \mathrm { c o n s } } .\tag{6b}
$$

Here, $y _ { \mathrm { a c t } }$ is the class index, equivalently a one-hot target, of the canonical action assigned to the target pedestrian. The auxiliary task is therefore formulated as single-label multi-class classification over $\mathcal { T } _ { \mathrm { a c t } }$ . The consistency term ${ \mathcal { L } } _ { \mathrm { c o n s } }$ is the symmetric Kullback-Leibler (KL) divergence between $p _ { \mathrm { i m g } }$ and $p _ { \mathrm { t x t } }$ . It encourages both modalities to agree on the action semantics of each image-text pair, thereby reducing their reliance on spurious background correlations.

## 3.4 Query Rewriting and Parallel Late-Fusion Retrieval

In static text-to-image retrieval, the original query $Q _ { o r i g }$ often lacks structural alignment, leaving actions ambiguous without temporal dynamics. To reduce this cross-modal ambiguity prior to retrieval, we propose an ofline contextgrounded query enrichment mechanism that explicitly conditions actions on the surrounding environment.

Query Rewriting. Given a query $Q _ { o r i g }$ , we assign it the action and scene tags $\tau _ { a c t } , \tau _ { s c e n e }$ - extracted via LLM inference over the canonical vocabulary - most relevant to its content from the canonical vocabulary obtained during ofline tag extraction (Sec. 3.2), and use an LLM $\mathcal { M } _ { l l m }$ to rewrite $Q _ { o r i g }$ into a fixed template:

“The image shows [person description] wearing [clothing] [action]. The background features [scene details].”

This process produces the rewritten query $Q _ { r e w } = \mathcal { M } _ { l l m } ( Q _ { o r i g } \oplus \tau _ { a c t } \oplus \tau _ { s c e n e } )$ where $\tau _ { a c t } \in \mathcal { T } _ { a c t }$ , and $\tau _ { s c e n e } \in \mathcal { T } _ { s c e n e }$ . By explicitly injecting action and scene tags, $Q _ { r e w }$ produces a context-grounded textual representation to ground the described behavior within its specific scene, mitigating the inherent ambiguities of the original caption.

Parallel Late-Fusion Retrieval. While the enriched query $Q _ { r e w }$ mitigates spatial ambiguity, we treat the original caption $Q _ { o r i g }$ and its rewrite $Q _ { r e w }$ as two complementary semantic views of the same retrieval intent. Specifically, $\boldsymbol { Q } _ { r e u }$ provides the necessary structural grounding of action and background, whereas $Q _ { o r i g }$ retains the raw, unadulterated fine-grained constraints (e.g., specific clothing colors or secondary object interactions) that might be inadvertently altered or omitted during LLM synthesis. To leverage both perspectives, we employ a parallel late-fusion retrieval strategy.

Let E denote the fine-tuned dual-encoder (SigLIP2 [17]) that projects both textual queries and images into a shared d-dimensional embedding space. We independently execute two retrieval branches across the entire image gallery G. The first branch computes cosine similarity scores $S _ { o r i g }$ using $Q _ { o r i g }$ , while the second computes $S _ { r e w }$ using $Q _ { r e w }$ . From these independent distributions, we extract the top-K candidate subsets, denoted $\mathcal { C } _ { o r i g }$ and $\mathcal { C } _ { r e w }$ . To maximize candidate recall before the computationally expensive reranking stage, we construct a union candidate pool $\mathcal { C } _ { u n i o n } = \mathcal { C } _ { o r i g } \cup \mathcal { C } _ { r e w }$ , preserving the visual evidence retrieved by either semantic view. For each candidate image $I _ { i } \in \mathcal { C } _ { u n i o n }$ , we compute a fused similarity score $S _ { f i n a l } ( I _ { i } )$ as the arithmetic mean of its corresponding scores from both branches, as formulated in Eq. (7).

$$
S _ { f i n a l } ( I _ { i } ) = \frac { 1 } { 2 } \Bigl ( S _ { o r i g } ( I _ { i } ) + S _ { r e w } ( I _ { i } ) \Bigr ) .\tag{7}
$$

Algorithm 1 Pivot-Promote Pairwise Reranking   
Require: Query $Q ;$ ranked candidates $\begin{array} { l l l } { { \mathcal { R } } } & { { = } } & { { [ I _ { 1 } , \ldots \ : . . , I _ { M } ] ; } } \end{array}$ ; pairwise comparator   
$\mathcal { C } ( Q , I _ { a } , I _ { b } ) \in \{ a , b , \mathtt { t i e } \}$   
Ensure: Reranked candidates   
1: function PivotPromote(Q, R)   
2: $\mathbf { i f } \ | \mathcal { R } | \leq 1$ then   
3: return R   
4: end if   
5: $I _ { a } \gets \mathcal { R } [ 1 ] , \mathcal { T } ^ { + } \gets [ ] , \mathcal { T } ^ { - } \gets [ ]$   
6: for $I _ { b } \in \mathcal { R } [ 2 : ]$ do   
7: append $I _ { b }$ to ${ \cal T } ^ { + } \ \mathrm { i f } \ { \mathcal C } ( Q , I _ { a } , I _ { b } ) = b ;$ otherwise to $\mathcal { T } ^ { - }$   
8: end for   
9: return PivotPromote $\left( Q , \mathcal { Z } ^ { + } \right) \oplus \left[ I _ { a } \right] \oplus \mathcal { Z } ^ { - }$   
10: end function

Finally, the union pool $\mathcal { C } _ { u n i o n }$ is re-sorted in descending order based on $S _ { f i n a l } .$ and a highly probable top-M candidate subset $\mathcal { C } _ { M }$ (where $M < K )$ is isolated and forwarded to the multimodal reranking module in Sec. 3.5.

## 3.5 Pairwise Multimodal Reranking

Pairwise Comparison as a Verification Signal. Rather than prompting a VLM to assign an absolute scalar score to each image-text pair, we cast reranking as a sequence of pairwise comparisons. We leverage Qwen3.5 [16] as a pairwise comparator $\mathcal { M } _ { c m p } ,$ and empirically validate this design choice against a pointwise scoring baseline in Sec. 5.3.

Given the original query caption $Q _ { o r i g }$ and a candidate pair $( I _ { a } , I _ { b } ) , \mathcal { M } _ { c m p }$ is prompted to compare the two images along a fixed set of fine-grained criteria, namely the primary action, target person, clothing, object interaction, secondary people, scene and background, and spatial relationships, together with any explicit contradictions or missing details relative to $Q _ { o r i g }$ . The comparator returns a single decision as defined in Eq. (8).

$$
w = \mathcal { M } _ { c m p } ( Q _ { o r i g } , I _ { a } , I _ { b } ) \in \{ a , b , \mathrm { t i e } \} ,\tag{8}
$$

indicating whether $I _ { a } , I _ { b } .$ , or neither is better supported by the query.

Pivot-Promote Reranking. The complete reranking procedure is summarized in Algorithm 1. Operating on the highly probable candidate subset $\mathcal { C } _ { M }$ (where $M < K \ll N )$ retrieved from the late-fusion stage, we employ a pivot-promote reranking procedure that recursively processes candidates preferred to the current pivot while preserving the prior order of the remaining candidates. $\mathrm { A n }$ initial pivot candidate $\boldsymbol { I _ { a } } \in \boldsymbol { \mathcal { C } } _ { M }$ , determined by the prior dense retrieval score, is compared against each remaining candidate $I _ { j } \in \mathcal { C } _ { M } \backslash \{ I _ { a } \}$ using $\mathcal { M } _ { c m p }$ , partitioning $\mathcal { C } _ { M } \backslash \left\{ I _ { a } \right\}$ into the two subsets defined in Eq. (9).

$$
{ \mathcal { C } } _ { b e t t e r } = \{ I _ { j } : w = b \} , \qquad { \mathcal { C } } _ { w o r s e } = \{ I _ { j } : w \in \{ a , { \mathrm { t i e } } \} \} .\tag{9}
$$

If $| \mathcal { C } _ { b e t t e r } | > 1$ , the algorithm recurses on $\mathcal { C } _ { b e t t e r } .$ , selecting a new pivot from within this subset and repeating the partitioning step; candidates in $\mathcal { C } _ { w o r s e }$ retain their relative order from the prior stage. The final ranking is obtained by concatenating the recursively sorted $\mathcal { C } _ { b e t t e r }$ , the pivot $I _ { a } ,$ and $\mathcal { C } _ { w o r s e } .$ , which rapidly elevates the ground-truth image toward the top of the ranking. Since pivot-promote recursively processes only candidates preferred to the current pivot, it has expected linear comparison complexity under a uniformly distributed pivot-rank assumption, while its worst-case complexity remains quadratic. In practice, the candidates are already approximately ordered by the preceding retrieval stages, typically yielding smaller recursive subsets and fewer comparisons.

## 4 Experiment

## 4.1 Experimental Setup

Datasets and Baselines. To comprehensively evaluate our framework, we utilize the PAB dataset [23] and the RSTPReid benchmark [28]. PAB is specifically constructed for person anomaly search, containing over 1 million image-text pairs spanning 1,600 anomalous and 1,000 normal behaviors. We use its augmented version from the AI City Challenge Track $4 ^ { 3 }$ to test model robustness in realworld surveillance scenarios. Additionally, we use RSTPReid (20,505 images of 4,101 persons across 15 cameras) to validate generalization in non-anomalyspecific environments featuring complex transformations. We compare our approach against the PAB-specific CMP baseline [23], which explicitly integrates pose patterns to distinguish actions. Furthermore, we benchmark against general VLM (CLIP [13], X-VLM [27]) and state-of-the-art appearance-matching architectures (IRRA [4], APTM [24], RaSa [21], WoRA [15], MRA [22], and CAMeL [25]). These models are primarily developed for appearance-oriented person retrieval and may therefore be less sensitive to fine-grained behavioral distinctions.

Implementation Details. We use Qwen3.5-9B [16] to canonicalize the raw annotations into action and scene tags. The retriever is initialized from SigLIP2- Patch16-256 [17] and fine-tuned on the PAB training set using the proposed objective with $\lambda _ { \mathrm { a c t i o n } } = 0 . 2$ . At inference, we retrieve the top-K=100 candidates for each query view, fuse their scores, and retain the top-M=10 candidates for pairwise reranking with Qwen3.5-9B. All Qwen inference uses deterministic decoding with temperature =0 and seed =0.

Evaluation Metrics. We adopt standard evaluation metrics widely used in cross-modal retrieval tasks to assess the model’s performance. Specifically, we report Recall@K (R@1, R@5, R@10) and mean Average Precision (mAP). Recall@K measures the percentage of natural language queries where the groundtruth matching image is successfully located within the top-K retrieved candidates. The mAP metric evaluates the overall area under the precision-recall curve, providing a comprehensive assessment of the model’s global ranking quality across all queries.

Table 1: Quantitative comparison on the original PAB test protocol under trainingfree and PAB fine-tuning settings.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="4">Training-free</td><td colspan="4">Fine-tuned on PAB</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>mAP</td><td>R@1</td><td>R@5</td><td>R@10 mAP</td><td></td></tr><tr><td>MRA [22]</td><td>PR&#x27;26</td><td>9.91</td><td>23.66</td><td>31.45</td><td>17.15</td><td>70.53</td><td>94.69</td><td>97.47</td><td>81.59</td></tr><tr><td>RaSa [21]</td><td>IJCAI&#x27;23</td><td>21.74</td><td>27.30</td><td>27.96</td><td>24.35</td><td>80.79</td><td>98.89</td><td>99.65</td><td>89.20</td></tr><tr><td>WoRA [15]</td><td>WWW&#x27;25</td><td>22.25</td><td>45.91</td><td>53.54</td><td>33.39</td><td>74.47</td><td>96.82</td><td>98.48</td><td>84.60</td></tr><tr><td>APTM [24]</td><td>MM&#x27;23</td><td>22.90</td><td>45.80</td><td>52.38</td><td>33.56</td><td>72.14</td><td>95.30</td><td>97.17</td><td>82.78</td></tr><tr><td>CAMeL [25]</td><td>TIFS&#x27;25</td><td>24.47</td><td>50.00</td><td>58.75</td><td>36.75</td><td>74.30</td><td>96.79</td><td>98.84</td><td>84.20</td></tr><tr><td>IRRA [4]</td><td>CVPR&#x27;23</td><td>30.59</td><td>59.61</td><td>68.91</td><td>44.41</td><td>76.39</td><td>97.62</td><td>99.14</td><td>86.33</td></tr><tr><td>CLIP [13]</td><td>ICML&#x27;21</td><td>47.57</td><td>81.55</td><td>89.03</td><td>62.73</td><td>77.60</td><td>98.84</td><td>99.75</td><td>87.35</td></tr><tr><td>X-VLM [27]</td><td>ICML’22</td><td></td><td>71.94 97.78</td><td>98.99</td><td>83.96</td><td>81.95</td><td>98.84</td><td>99.19</td><td>89.86</td></tr><tr><td>CMP [23]</td><td>ICCV&#x27;25</td><td></td><td>一</td><td></td><td></td><td></td><td>84.93 99.09</td><td>99.75</td><td>91.66</td></tr><tr><td>SSDC [19]</td><td>ACL&#x27;26</td><td></td><td>一</td><td></td><td></td><td></td><td>87.21 99.09</td><td>99.75</td><td>92.87</td></tr><tr><td></td><td>ActPair (Ours) ECCVW&#x27;26</td><td></td><td></td><td></td><td></td><td></td><td>88.62 99.75</td><td>99.85</td><td>93.97</td></tr></table>

## 4.2 Quantitative Results

Main Results on PAB. ActPair achieves the best results among the compared published baselines on the original PAB test under the reported evaluation protocol, obtaining 88.62% R@1 and 93.97% mAP, as shown in Table 1. In the training-free setting without domain-specific adaptation, conventional appearance-oriented architectures, including IRRA [4], RaSa [21], and APTM [24], struggle to capture fine-grained behavioral anomalies, with the strongest of these methods reaching only 44.41% mAP. Foundational VLMs perform considerably better, with X-VLM [27] achieving 83.96% mAP, yet still remain below methods explicitly adapted to the PAB domain.

Under the PAB fine-tuning setting, ActPair consistently outperforms both general-purpose person retrieval methods and PAB-specific baselines. Compared with the pose-aware CMP [23], our method improves R@1 and mAP by 3.69 and 2.31 percentage points, respectively. It also surpasses the stronger cascade-based SSDC [19] baseline by 1.41 points in R@1 and 1.10 points in mAP. Notably, the improvements are most pronounced at rank 1 and in overall ranking quality, while R@5 and R@10 are already close to saturation. This suggests that ActPair primarily improves the fine-grained ordering of highly similar candidates, rather than merely increasing coarse candidate recall.

Main Results on RSTPReid To assess the robustness and transferability of the learned representations in general, non-anomaly-specific environments, Table 2 reports the performance on the RSTPReid dataset [28]. The evaluation is divided into supervised fine-tuning and cross-dataset evaluation scenarios. Under the cross-dataset setting, where models are evaluated without utilizing any RSTPReid [28] samples for training or domain adaptation, ActPair substantially improves over the same-source CMP baseline. Specifically, when transferring directly from the PAB dataset to RSTPReid using the full pipeline with pairwise reranking, our approach yields a Recall@1 of 55.25%, a Recall@5 of 69.05%, and a Recall@10 of 73.75%. This constitutes a substantial improvement over the baseline CMP model [23], which attains only 29.15% Recall@1 under identical zero-shot transfer conditions; since reranking only reorders the top-10, the lower Recall@5/@10 relative to CFine, APTM, and IRRA reflects the retrieval stage’s coverage rather than a reranking limitation.

Table 2: Comparison on RSTPReid under supervised fine-tuning and cross-dataset transfer. In the zero-shot setting, IVT, IRRA, CFine, APTM, and RaSa are trained on ICFG-PEDES, while CMP and Ours are trained on PAB. “–” denotes unavailable results.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Venue</td><td colspan="3">Supervised</td><td colspan="3">Zero-shot Transfer</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>DSSL [28]</td><td>MM&#x27;21</td><td>32.43</td><td>55.08</td><td>63.19</td><td></td><td></td><td></td></tr><tr><td>IVT [14]</td><td>ECCVW&#x27;22</td><td>46.70</td><td>70.00</td><td>78.80</td><td>43.70</td><td>65.10</td><td>75.55</td></tr><tr><td>CFine [20]</td><td>TIP&#x27;23</td><td>50.55</td><td>72.50</td><td>81.60</td><td>47.40</td><td>70.60</td><td>79.35</td></tr><tr><td>IRRA [4]</td><td>CVPR&#x27;23</td><td>60.20</td><td>81.30</td><td>88.20</td><td>45.10</td><td>69.20</td><td>78.75</td></tr><tr><td>RDE [i1]</td><td>CVPR&#x27;24</td><td>65.35</td><td>83.95</td><td>89.90</td><td></td><td></td><td></td></tr><tr><td>APTM [24]</td><td>MM&#x27;23</td><td></td><td>67.50 85.70</td><td>91.45</td><td>52.50</td><td>75.15</td><td>81.70</td></tr><tr><td>RaSa [21]</td><td>IJCAI&#x27;23</td><td></td><td></td><td></td><td>55.00</td><td>73.65</td><td>81.55</td></tr><tr><td>CMP [23]</td><td>ICCV’25</td><td></td><td></td><td></td><td>29.15</td><td>50.40</td><td>60.75</td></tr><tr><td></td><td>ActPair (Ours) ECCVW&#x27;26</td><td></td><td></td><td></td><td>55.25</td><td>69.05</td><td>73.75</td></tr></table>

![](images/98e181e4de725895d960a8906e639b2c1ae52796cdd33d78f6b0e2c279e2ffa3.jpg)  
Fig. 3: Qualitative top-5 retrieval results across the three stages. The numbers indicate the ground-truth ranks after action-aligned retrieval, dual-view query fusion, and pairwise reranking. Queries 1–3 are successful cases, while Query 4 illustrates a failure on complex spatial and action-state constraints.

While RSTPReid predominantly relies on appearance rather than behavioral anomalies, ActPair substantially improves over the same-source CMP baseline without using RSTPReid annotations or domain adaptation. This suggests that the action-aligned representations learned on PAB retain useful cross-domain information, particularly for top-1 retrieval.

Table 3: Progressive evaluation on the AI City 2026 challenge test and the original PAB test.
<table><tr><td rowspan="2">Method</td><td colspan="2">Challenge Set</td><td colspan="2">Public Test</td></tr><tr><td>R@1 R@5</td><td>R@10 mAP</td><td>R@1</td><td>R@5 R@10 1 mAP</td></tr><tr><td>CMP [23]</td><td>69.41 89.43</td><td>91.91</td><td>84.93 99.09</td><td>99.75 91.66</td></tr><tr><td>Action-Aligned Retriever</td><td>72.14 92.77</td><td>95.50 81.68</td><td>84.98 99.29</td><td>99.60 91.73</td></tr><tr><td>+ Dual-View Query</td><td>75.73 94.99</td><td>97.27 84.48</td><td>85.95 99.34</td><td>99.85 92.30</td></tr><tr><td>+ Pairwise Reranking</td><td>84.88 96.36 97.27 90.14</td><td></td><td></td><td>88.62 99.7599.8593.97</td></tr></table>

## 4.3 Qualitative Analysis

Figure 3 shows representative retrieval results across the three stages. For queries 1–3, dual-view fusion improves candidate discovery while pairwise reranking promotes the ground-truth image to rank 1; Query 3 illustrates this clearly, moving from rank 57 to 6 to 1 across the three stages. Query 4 presents a failure case: although the correct image is initially ranked first, it is demoted by the later stages. This suggests that the model may still struggle with long compositional descriptions involving action phases, multiple people, and precise spatial relations.

## 5 Ablation Study

## 5.1 Contribution of Each Stage

Table 3 presents a progressive evaluation of the framework’s components on the Challenge and PAB Public Test sets. The baseline configuration, employing the action-supervised SigLIP2 dual-encoder, establishes a strong foundation with a Recall@1 of 72.14% on the Challenge Set and 84.98% on the Public Test set. This configuration outperforms the pose-aware CMP baseline, indicating higher initial retrieval accuracy under the evaluated setting. Building upon the action-aligned retriever, dual-view query fusion preserves fine-grained query details while leveraging the context-grounded rewrite. This parallel retrieval strategy captures complementary semantic evidence, improving Recall@1 to 75.73% and 85.95% on the respective test sets. Finally, incorporating the training-free pairwise multimodal reranking further improves top-rank retrieval among the top-10 candidates. This complete framework delivers the most substantial performance gain, elevating the Challenge Set Recall@1 to 84.88% and achieving a final Recall@1 of 88.62% alongside a 93.97% mAP on the Public Test set. These progressive improvements demonstrate the complementary contribution of each stage in isolating fine-grained behavioral anomalies.

## 5.2 Action versus Pose Supervision

Table 4 isolates the efectiveness of explicit action supervision relative to skeletal pose signals. Incorporating pose information into the CMP baseline yields only a modest improvement, increasing Recall@1 from 68.60% to 69.41%. However, under a controlled comparison using the same SigLIP2 backbone [17], pose-based supervision remains less efective than our explicit action supervision. Employing human pose extracted via RTMPose [5] as the primary training signal proved suboptimal, achieving only 67.95% Recall@1. This empirically highlights the pose-semantic gap: isolated skeletal structures fail to uniquely define actions without environmental context. Conversely, applying explicit action-tag supervision to the SigLIP2 backbone significantly enhances retrieval accuracy, yielding 72.14% Recall@1, 92.77% Recall@5, and 95.50% Recall@10. Paired bootstrap analysis confirms a statistically significant improvement of +4.19% in Recall@1 (95% CI: [+2.38, +6.07]) for action-based over pose-based supervision. These results demonstrate that optimizing for context-conditioned action semantics is substantially more discriminative for anomaly search than relying on geometric constraints.

Table 4: Controlled comparison of pose- and action-based supervision on the challenge set. The SigLIP2 [17] comparison keeps the backbone and training configuration fixed, isolating the efect of the supervision signal.
<table><tr><td>Backbone Supervision</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td colspan="4">Effect of pose information in CMP [23]</td></tr><tr><td>CMP RGB and text</td><td>68.60</td><td>87.51</td><td>89.64</td></tr><tr><td>CMP RGB, text, and pose</td><td>69.41</td><td>89.43</td><td>91.91</td></tr><tr><td colspan="4">Controlled comparison under the same SigLIP2 backbone [17]</td></tr><tr><td>SigLIP2 Pose-based supervision</td><td>67.95</td><td>91.20</td><td>94.74</td></tr><tr><td>SigLIP2 Action-based supervision</td><td>72.14</td><td>92.77</td><td>95.50</td></tr><tr><td>∆ Action − Pose Paired bootstrap 95% CI</td><td>+4.19</td><td>+1.57</td><td>+0.76</td></tr><tr><td colspan="4">[+2.38, +6.07] [+0.51, +2.63] [−0.10, +1.62]</td></tr></table>

## 5.3 Pointwise versus Pairwise Reranking

Table 5 evaluates the proposed pairwise reranking strategy against a pointwise scoring baseline using the identical Qwen3.5-9B model and candidate pool. Assigning an absolute, independently calibrated scalar score (pointwise) requires the model to maintain a consistent scale across unrelated queries. In contrast, pairwise comparison simplifies the reasoning objective by tasking the model solely with determining which of two simultaneously observed candidates better satisfies the specific query criteria.

Empirically, while pointwise VLM reranking improves the initial dense retrieval baseline (increasing Recall@1 from 75.73% to 80.13%), the pairwise strategy is substantially more discriminative. It achieves 84.88% Recall@1 and 90.14% mAP, outperforming the pointwise approach by absolute margins of +4.75% and +2.44%, respectively (95% CI for Recall@1: [+3.08, +6.42]). Specifically, this ranking efectiveness substantially reduces the number of comparisons relative to exhaustive all-pairs reranking. Under our top-10 configuration, pairwise reranking uses an average of 9.09 model calls per query, compared with 10.00 calls for the pointwise implementation.

Table 5: Comparison of pointwise and pairwise VLM reranking on the challenge set. Both variants use the same Qwen3.5-9B model [16] and candidate pool. Requests denote the average number of VLM calls per query.
<table><tr><td>Reranking strategy</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>mAP ↑</td><td> $\mathbf { R e q . } / \mathbf { Q } \downarrow$ </td></tr><tr><td>No VLM reranking</td><td>75.73</td><td>94.99</td><td>97.27</td><td>84.48</td><td>0.00</td></tr><tr><td>Qwen3.5 [16] pointwise</td><td>80.13</td><td>96.41</td><td>97.27</td><td>87.70</td><td>10.00</td></tr><tr><td>Qwen3.5 [16] pairwise</td><td>84.88</td><td>96.36</td><td>97.27</td><td>90.14</td><td>9.09</td></tr><tr><td>Pairwise — Pointwise</td><td>+4.75</td><td>-0.05</td><td>0.00</td><td>+2.44</td><td>-0.91</td></tr><tr><td>Paired bootstrap 95% CI</td><td></td><td>[+3.08, +6.42] [−0.61, +0.51]</td><td>[0, 0]</td><td>[+1.45, +3.43]</td><td></td></tr></table>

## 6 Conclusion

In this paper, we propose a three-stage framework for text-based person anomaly search that bridges the pose-semantic gap via action-aligned representation learning, context-grounded query fusion, and training-free pairwise multimodal reranking. ActPair achieves the best performance among the compared published baselines on the original PAB test and substantially improves over the same-source CMP baseline in cross-dataset transfer to RSTPReid. Beyond pedestrian anomaly search, ActPair may support other context-grounded retrieval tasks, such as distinguishing similar actions in sports. We view our framework as a step toward such broader context-grounded retrieval tasks.

Limitations and Future Work: Constructing canonical vocabularies $( { \mathcal T } _ { a c t }$ and $\tau _ { s c e n e } )$ currently requires dataset-specific annotations, limiting scalability in unannotated domains. Future work will eliminate this dependency by utilizing an unsupervised MLLM pipeline to dynamically parse raw captions into action and scene clusters.

Acknowledgement. This work was supported by Saigon AI Hub, jointly established by VNG Group JSC and Vietnam National University Ho Chi Minh City (VNU-HCM), which provided computing infrastructure, research resources, and a collaborative research environment. Trong-Thuan Nguyen was funded by the PhD Scholarship Programme of Vingroup Innovation Foundation (VINIF), VinUniversity, code VINIF.2025.TS63.

## References

1. Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., Hullender, G.: Learning to rank using gradient descent. In: Proceedings of the 22nd international conference on Machine learning. pp. 89–96 (2005)

2. Cao, Z., Qin, T., Liu, T.Y., Tsai, M.F., Li, H.: Learning to rank: from pairwise approach to listwise approach. In: Proceedings of the 24th international conference on Machine learning. pp. 129–136 (2007)

3. Hu, Y., Cao, Z., Li, P., Zhu, Q.: Recqr: Incorporating conversational query rewriting to improve multimodal image retrieval. arXiv preprint arXiv:2603.26669 (2026)

4. Jiang, D., Ye, M.: Cross-modal implicit relation reasoning and aligning for text-toimage person retrieval. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2787–2797 (2023)

5. Jiang, T., Lu, P., Zhang, L., Ma, N., Han, R., Lyu, C., Li, Y., Chen, K.: Rtmpose: Real-time multi-person pose estimation based on mmpose. arXiv preprint arXiv:2303.07399 (2023)

6. Ju, H., Zhang, H., Zheng, Z.: Anomalylmm: Bridging generative knowledge and discriminative retrieval for text-based person anomaly search. arXiv preprint arXiv:2509.04376 (2025)

7. Khaertdinov, B., Popa, M., Tintarev, N.: A little more like this: Text-to-image retrieval with vision-language models using relevance feedback. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 3825–3834 (2026)

8. Lee, S., Yu, S., Park, J., Yi, J., Yoon, S.: Interactive text-to-image retrieval with large language models: A plug-and-play approach. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 791–809 (2024)

9. Li, H., Liu, Y., Zhang, H., Li, B.: Mitigating and evaluating static bias of action representations in the background and the foreground. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19911–19923 (2023)

10. Luo, J., Ren, W., Jiang, W., Chen, X., Wang, Q., Han, Z., Liu, H.: Discovering syntactic interaction clues for human-object interaction detection. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 28212–28222 (2024)

11. Qin, Y., Chen, Y., Peng, D., Peng, X., Zhou, J.T., Hu, P.: Noisy-correspondence learning for text-to-image person re-identification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 27197– 27206 (2024)

12. Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Yan, L., Shen, J., Liu, T., Liu, J., Metzler, D., et al.: Large language models are efective text rankers with pairwise ranking prompting. In: Findings of the Association for Computational Linguistics: NAACL 2024. pp. 1504–1518 (2024)

13. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)

14. Shu, X., Wen, W., Wu, H., Chen, K., Song, Y., Qiao, R., Ren, B., Wang, X.: See finer, see more: Implicit modality alignment for text-based person retrieval. In: European Conference on Computer Vision. pp. 624–641. Springer (2022)

15. Sun, J., Fei, H., Ding, G., Zheng, Z.: From data deluge to data curation: A filteringwora paradigm for eficient text-based person search. In: Proceedings of The Web Conference (WWW) (2025)

16. Team, Q.: Qwen3.5: Accelerating productivity with native multimodal agents (February 2026), https://qwen.ai/blog?id=qwen3.5

17. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al.: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 (2025)

18. Wang, T., Anwer, R.M., Khan, M.H., Khan, F.S., Pang, Y., Shao, L., Laaksonen, J.: Deep contextual attention for human-object interaction detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5694–5702 (2019)

19. Xie, Z., Luo, G., Wang, C., Cai, S., Jin, T., Zhao, Z., Tang, Y.: Bridging the posesemantic gap: A cascade framework for text-based person anomaly search. In: Findings of the Association for Computational Linguistics: ACL 2026. pp. 4040– 4049 (2026)

20. Yan, S., Dong, N., Zhang, L., Tang, J.: Clip-driven fine-grained text-image person re-identification. IEEE Transactions on Image Processing 32, 6032–6046 (2023)

21. Yang, B., Min, C., Daming, G., Ziqiang, C., Chen, C., Zhenfeng, F., Liqiang, N., Min, Z.: Rasa: Relation and sensitivity aware representation learning for text-based person search. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) (2023)

22. Yang, S., Wang, Y., Li, Y., Zhu, L., Zheng, Z.: Minimizing the pretraining gap: Domain-aligned text-based person retrieval. Pattern Recognition p. 113511 (2026)

23. Yang, S., Wang, Y., Zhu, L., Zheng, Z.: Beyond walking: A large-scale imagetext benchmark for text-based person anomaly search. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2025)

24. Yang, S., Zhou, Y., Zheng, Z., Wang, Y., Zhu, L., Wu, Y.: Towards unified textbased person retrieval: A large-scale multi-attribute and language search benchmark. In: Proceedings of the 31st ACM international conference on multimedia. pp. 4492–4501 (2023)

25. Yu, H., Wen, J., Zheng, Z.: Camel: cross-modality adaptive meta-learning for textbased person retrieval. IEEE Transactions on Information Forensics and Security (2025)

26. Yuan, C., Zhao, Y., Xu, H., Niu, G.: Towards robust text-to-image person retrieval: Multi-view reformulation for semantic compensation. arXiv preprint arXiv:2604.18376 (2026). https://doi.org/10.48550/arXiv.2604.18376, https: //arxiv.org/abs/2604.18376

27. Zeng, Y., Zhang, X., Li, H.: Multi-grained vision language pre-training: Aligning texts with visual concepts. In: International Conference on Machine Learning. pp. 25994–26009. PMLR (2022)

28. Zhu, A., Wang, Z., Li, Y., Wan, X., Jin, J., Wang, T., Hu, F., Hua, G.: Dssl: Deep surroundings-person separation learning for text-based person retrieval. In: Proceedings of the 29th ACM international conference on multimedia. pp. 209–217 (2021)

29. Zhuang, S., Zhuang, H., Koopman, B., Zuccon, G.: A setwise approach for efective and highly eficient zero-shot ranking with large language models. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 38–47 (2024)