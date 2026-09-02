# Does This Moment Justify the Recommendation? Counterfactual Behavior-Grounded Evidence Retrieval for Personalized Video Recommendation

Xin Liu

lxinhit@gmail.com

## Abstract

Personalized video recommendation predicts user preference at the video level, while temporal video grounding localizes query-relevant moments. However, strong localization does not establish whether the retrieved moment constitutes valid evidence for recommending the video to a particular user. We study counterfactual behavior-grounded evidence retrieval, which separates where personalized evidence occurs from whether such evidence exists and evaluates whether model predictions respond consistently when that evidence is replaced. We introduce CBGER-10K, containing 5,000 controlled factual–counterfactual pairs for 3,026 users, where each pair replaces only the focal behavior-supported segment while preserving the user, temporal position, and hard distractors. We further propose CBGER, a compact framework that decouples segmentlevel localization from video-level evidence estimation and learns both through structured counterfactual supervision. CBGER achieves 0.4432 MRR, 0.6977 Pair Accuracy, and 0.6987 Intervention Consistency across five adapted personalized-highlight and temporal-grounding baselines. Notably, compared with QD-DETR, its MRR improvement is not statistically significant, while Pair Accuracy improves by 11.03 points. These results show that accurate temporal localization does not necessarily imply reliable personalized evidence existence, motivating explicit evaluation of Whether alongside Where.

## 1. Introduction

Video recommendation systems learn from implicit behavioral feedback to rank candidate videos [21], but a videolevel relevance score provides little temporal evidence about which event supports a recommendation. Temporal video grounding addresses the complementary problem of localizing moments relevant to a language query [11, 13, 17, 18, 20]. Standard grounding benchmarks, however, primarily evaluate where a query-relevant moment occurs once a query–video relation is defined. Personalized recommendation introduces an additional question: for a particular user’s behavior history, does the candidate contain meaningful supporting evidence at all?

This distinction exposes a failure mode that localization metrics alone cannot reveal. A grounding model may rank one segment highest even when it is only the best among weakly relevant alternatives, while a recommendation model may rely on coarse topic compatibility without identifying the event that supports its prediction. We therefore study personalized video evidence along two prediction dimensions and one counterfactual consistency criterion:

Where: localize behavior-supported evidence,

Whether: whether sufficient evidence is present,

Intervention: respond to evidence replacement. (1)

We operationalize these questions using matched factual– counterfactual videos. As shown in Figure 1, for a fixed user, a factual timeline contains a behavior-supported focal segment, while its counterfactual replaces only that segment with a semantically related but behavior-weaker alternative at the same temporal position; the remaining hard distractors are unchanged. A faithful model should localize the factual evidence, assign stronger video-level evidence to the factual timeline, and reduce the local score at the edited slot after replacement. This construction tests model dependence on designated evidence under controlled content changes, without making a causal claim about real user engagement.

To study this setting, we introduce Counterfactual Behavior-Grounded Evidence Retrieval (CBGER) and the CBGER-10K benchmark. CBGER-10K contains 5,000 factual–counterfactual pairs for 3,026 users, combining behavior histories from MicroLens with candidate video segments from FineVideo [9, 21]. Each pair changes only one focal segment while preserving the user, temporal position, and surrounding distractors. User and source identities are disjoint across splits, and the representations used to construct the benchmark are separated from the frozen CLIP features used by downstream models [23], reducing direct constructor–solver coupling.

![](images/ba2ebb6c93eceac913f1d4481a4c74e8accff49736e74916c89407eabdff83f7.jpg)  
Figure 1. CBGER task overview. A chronological user history defines personalized context. The factual timeline V<sup>+</sup> contains a behaviorsupported focal segment e, while the matched counterfactual V<sup>−</sup> replaces only that segment with a behavior-weaker alternative e˜ at the same temporal position and preserves the surrounding hard distractors. The pair supports evaluation of Where, Whether, and intervention consistency.

We further introduce a compact behavior-conditioned model that decouples segment localization from video-level evidence existence. A temporal encoder produces userconditioned segment scores for Where, while a parameterfree aggregation of these scores provides the Whether decision. Structured counterfactual learning combines factual localization, pairwise factual–counterfactual ordering, local intervention supervision, and invariance on unchanged distractors.

Experiments with adapted personalized-highlight and temporal-grounding baselines reveal a gap hidden by standard localization evaluation. CBGER achieves 0.4432 MRR, 0.6977 Pair Accuracy, and 0.6987 Intervention Consistency. Compared with QD-DETR [20], the MRR difference is not statistically significant, yet CBGER improves Pair Accuracy by 0.1103 (95% CI [0.0765, 0.1430]). Thus, models with similar temporal localization quality can differ substantially in their ability to verify personalized evidence and respond to its removal.

Our contributions are:

• We formulate personalized video evidence retrieval as separate Where and Whether decisions together with counterfactual intervention consistency.

• We introduce CBGER-10K, a controlled benchmark of 5,000 factual–counterfactual pairs for 3,026 users with single-slot interventions and user- and source-disjoint splits.

• We propose a compact behavior-conditioned framework with structured counterfactual learning that jointly trains localization, evidence ordering, local intervention response, and distractor invariance.

• Across five adapted personalized-highlight and temporalgrounding baselines, we show that localization quality alone does not characterize personalized evidence reliability. Our CBGER-10K dataset and CBGER code are available at https://anonymous.4open. science/r/cbger-111E.

## 2. Related Work

Temporal grounding and personalized highlights. Temporal video grounding localizes language-relevant moments in untrimmed videos, progressing from proposal-based matching [11, 13, 16] to transformer-based retrieval and unified grounding–highlight formulations [17, 18, 20]. Recent methods further improve temporal interaction, boundary modeling, foreground discrimination, and efficient localization [3, 15, 24, 25, 27]. Video highlight detection similarly ranks salient moments, while personalized variants condition temporal relevance on viewer preferences or histories [1, 5, 12, 19, 28]. These approaches are closely related to our Where dimension, but primarily optimize which moment is most relevant within a video. CBGER additionally evaluates whether the localized content provides sufficient behavior-supported evidence at the video level.

Negative-aware grounding, explainable recommendation, and counterfactual evaluation. Recent grounding work has begun to model missing correspondences explicitly. SHINE introduces semantically plausible hard negative queries [7], while Negative-Aware Video Moment Retrieval requires models to reject queries whose events are absent from the video [10]. These studies are related to our Whether dimension, but operate on query–video correspondence rather than user-conditioned recommendation evidence. In parallel, explainable recommendation seeks human-interpretable reasons for item suggestions [30], and REASONER provides multimodal explanation labels for video recommendation [6]. Counterfactual and grounded video evaluation further tests whether predictions depend on relevant visual or temporal evidence rather than shortcuts [2, 14, 26, 29]. Our setting combines these directions: a behavior-derived user profile defines personalized evidence, and matched factual–counterfactual videos replace one focal segment while holding the surrounding timeline fixed. This enables joint evaluation of localization, evidence existence, and local intervention consistency.

Behavior-conditioned multimodal benchmarks. MicroLens provides large-scale user–micro-video interactions and multimodal content for recommendation research [21], while FineVideo provides diverse temporally structured video content [9]. Foundation models also enable scalable semantic annotation and temporal structure discovery [8]. CBGER-10K builds on these resources but targets a different diagnostic question: whether temporally localized video content is supported by a user’s observed behavior. Its automatically constructed labels are treated as silver supervision, and matched factual–counterfactual pairs together with user- and source-disjoint splits are used to evaluate personalized evidence reasoning under controlled content changes.

## 3. Task and the CBGER-10K Benchmark

## 3.1. Task Definition

For a user u, let $H _ { u } = ( j _ { 1 } , \dots , j _ { M } )$ denote the chronological consumption history and let $p _ { u }$ be a behavior profile derived from that history. A candidate video is represented

as a timeline of N segments,

$$
V = \{ ( x _ { i } , [ a _ { i } , b _ { i } ] ) \} _ { i = 1 } ^ { N } .\tag{2}
$$

Given $( p _ { u } , V )$ , a model predicts segment-level evidence scores ${ \bf \Phi } { \cal S } ( u , V ) ~ = ~ \{ s _ { i } \} _ { i = 1 } ^ { N }$ and a video-level evidenceexistence score $q ( u , V )$

CBGER-10K evaluates two prediction capabilities and one counterfactual consistency criterion. For a factual timeline $V ^ { + }$ with designated evidence segment $y ,$ Where evaluates whether $s _ { y }$ is ranked above the distractor segments. Given its matched counterfactual $V ^ { - }$ , Whether requires

$$
q ( u , V ^ { + } ) > q ( u , V ^ { - } ) .\tag{3}
$$

Finally, Intervention Consistency tests whether replacing the factual evidence at the same temporal slot decreases its local evidence score,

$$
s _ { y } ( u , V ^ { + } ) > s _ { y } ( u , V ^ { - } ) .\tag{4}
$$

The benchmark therefore distinguishes successful localization from recognizing whether personalized evidence is present and whether predictions depend on that evidence. Figure 2 illustrates the pipeline for constructing CBGER-10K.

## 3.2. Behavior Profiles and Candidate Videos

Behavior profiles. MicroLens provides timestamped user–item interactions and multimodal micro-video metadata [21]. We use chronological interactions and video titles to construct behavior-supported user profiles; unobserved items are not treated as negatives. After normalization, title unigrams and bigrams are linked to the exact history items that support them. Their user-level inverse document frequency is

$$
\mathrm { i d f } ( a ) = \log \frac { | \mathcal { U } | + 1 } { \mathrm { d f } _ { \mathcal { U } } ( a ) + 1 } ,\tag{5}
$$

and phrase support is

$$
c _ { u } ( a ) = | H _ { u } ( a ) | \operatorname { i d f } ( a ) \rho ( a ) ,\tag{6}
$$

where $\rho ( a ) ~ = ~ 1 . 2 5$ for bigrams and 1 otherwise. The highest-support phrases and their associated history items form the global behavior profile $p _ { u }$ . These profiles are operational summaries of observed consumption rather than psychological ground-truth interests.

Candidate videos. FineVideo provides Creative-Commons videos with temporal intervals, transcripts, and structured descriptions [9]. We use its media and temporal provenance but do not use FineVideo category overlap as evidence supervision. Eligible intervals have durations between 1 and 12 seconds. FineVideo source videos are assigned to train, validation, and test splits before annotation or matching.

![](images/836192e0bb3befd3f5b32e614b1392dd2c6c4e16502417e1b91949ca0c4750a3.jpg)  
Figure 2. CBGER-10K construction pipeline. MicroLens histories are converted into behavior-supported profiles. Eligible FineVideo intervals are independently annotated with structured visual semantics and matched to profiles to identify factual evidence. Each factua timeline contains one focal evidence segment and eight hard distractors. Its counterfactual preserves the user, focal temporal slot, and distractors while replacing only the focal segment with a semantically related but behavior-weaker content neighbor.

## 3.3. Semantic Annotation and Matching

Each eligible FineVideo interval is annotated by Qwen3- VL-8B-Instruct [22] using a fixed 4-FPS BF16 configuration selected on a held-out configuration study. The structured annotation contains objects, actions, ordered events, state changes, outcomes, topics, and an uncertainty estimate. Invalid outputs and annotations with uncertainty above 0.5 are discarded, yielding 7,651 valid intervals. We serialize the retained semantics as

$$
\begin{array} { r l r } & { \phi ( v ) = \operatorname { T o p i c s } ( T _ { v } ) ; \operatorname { O b } \operatorname { j e c t s } ( O _ { v } ) ; \operatorname { \mathrm { ~ } } \operatorname { \mathrm { ~ a c t ~ i ~ o n s } } ( A _ { v } ) ; } & \\ & { \operatorname { E v e n t s } ( E _ { v } ) ; \operatorname { O u t } \operatorname { c o m e } ( Y _ { v } ) . } & { ( 7 } \end{array}\tag{}
$$

User profiles and interval semantics are independently encoded with BGE-M3 [4]:

$$
h _ { u } = \frac { f ( \psi ( u ) ) } { \| f ( \psi ( u ) ) \| _ { 2 } } , \qquad z _ { v } = \frac { f ( \phi ( v ) ) } { \| f ( \phi ( v ) ) \| _ { 2 } } ,\tag{8}
$$

with construction similarity

$$
m ( u , v ) = h _ { u } ^ { \top } z _ { v } .\tag{9}
$$

A segment may serve as factual evidence only if $m ( u , v ) \geq$ 0.30.

Qwen3-VL and BGE-M3 are used only for benchmark construction. Downstream models receive frozen CLIP features [23]; construction embeddings and matching scores are never exposed to the solver. This representation separation reduces direct constructor–solver leakage, although it does not remove potential teacher bias.

## 3.4. Factual–Counterfactual Pair Construction

Construction is performed independently within each split. For each user, factual candidates are ranked by $m ( u , v )$ subject to diversity constraints: the same user–segment pair is not repeated, each evidence segment is reused at most three times, and each user contributes at most two pairs.

For a selected factual segment $v ^ { + }$ , we mine a semantically related counterfactual replacement from its nearest content neighbors. Content similarity is

$$
c ( v ^ { + } , v ) = z _ { v ^ { + } } ^ { \top } z _ { v } .\tag{10}
$$

A replacement $v ^ { - }$ must satisfy

$$
c ( v ^ { + } , v ^ { - } ) \geq 0 . 4 5 , \qquad m ( u , v ^ { + } ) - m ( u , v ^ { - } ) \geq 0 . 0 6 .\tag{11}
$$

Thus, $v ^ { - }$ remains close in content while providing weaker support for the same user profile. Distractor segments are

Table 1. CBGER-10K benchmark statistics. Each pair contributes one factual and one counterfactual record.
<table><tr><td>Statistic</td><td>Train</td><td>Val.</td><td>Test</td><td>Total</td></tr><tr><td>Records</td><td>7,000</td><td>1,000</td><td>2,000</td><td>10,000</td></tr><tr><td>Pairs</td><td>3,500</td><td>500</td><td>1,000</td><td>5,000</td></tr><tr><td>Unique users</td><td>2,103</td><td>307</td><td>616</td><td>3,026</td></tr><tr><td>Factual records</td><td>3,500</td><td>500</td><td>1,000</td><td>5,000</td></tr><tr><td>Counterfactual records</td><td>3,500</td><td>500</td><td>1,000</td><td>5,000</td></tr></table>

additionally required to satisfy

$$
m ( u , d ) \leq m ( u , v ^ { + } ) - 0 . 0 3 .\tag{12}
$$

Each factual timeline contains nine unique segments: one focal evidence segment and eight hard distractors. The focal position is randomly sampled with a fixed seed. The matched counterfactual preserves the user profile, history, temporal slot, and all eight distractors, and replaces only $v ^ { + }$ with $v ^ { - }$ . This single-slot intervention minimizes changes outside the designated evidence while allowing direct comparison of factual and counterfactual predictions.

All records retain source identifiers, temporal intervals, construction scores, counterfactual links, random seeds, and integrity hashes for reproducibility. We emphasize that these are automatically constructed silver labels: the intervention measures model dependence on designated personalized evidence, not the causal effect of video content on observed user engagement.

## 3.5. Splits and Benchmark Statistics

Table 1 summarizes the key statistics of the CBGER-10K benchmark. Users are assigned to train, validation, and test partitions using a stable 70/10/20 split. FineVideo source videos are independently partitioned before candidate selection, resulting in zero user or source overlap across splits. Structural validation checks factual–counterfactual links, shared histories and distractors, single-slot replacement, temporal intervals, matching constraints, and feature coverage; all 5,000 pairs pass these checks.

## 3.6. Evaluation Metrics

We report the three benchmark axes separately. Where is evaluated on factual records using MRR and NDCG@k, treating the designated factual segment as the relevant item.

Whether is measured by Pair Accuracy,

$$
A _ { \mathrm { p a i r } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( V ^ { + } , V ^ { - } ) \in \mathcal { P } } \mathbf { 1 } \left[ q ( u , V ^ { + } ) > q ( u , V ^ { - } ) \right] .\tag{13}
$$

Intervention Consistency measures whether the score

at the fixed focal slot decreases after replacement:

$$
A _ { \mathrm { i n t } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( V ^ { + } , V ^ { - } ) \in \mathcal { P } } \mathbf { 1 } \left[ s _ { y } ( u , V ^ { + } ) > s _ { y } ( u , V ^ { - } ) \right] .\tag{14}
$$

Reporting these metrics separately is important because a model may localize the factual segment accurately while still failing to distinguish evidence-present from matched evidence-weakened videos.

## 4. Method

## 4.1. Overview

We propose a compact behavior-conditioned evidence model for CBGER as shown in Figure 3. Given a user behavior profile and a nine-segment candidate timeline, the model produces segment-level evidence scores for Where and a video-level evidence score for Whether. Frozen CLIP encoders provide profile and segment representations [23]. A lightweight trainable scorer combines a CLIP similarity residual with a small relative-temporal correction. During training, the same model is applied to matched factual and counterfactual timelines. Structured objectives supervise factual localization, video-level factual– counterfactual ordering, the response at the edited slot, and invariance of unchanged distractors.

## 4.2. Behavior-Conditioned Evidence Scoring

Let $x _ { i } \in \mathbb { R } ^ { d _ { c } }$ denote the frozen CLIP representation of segment i, and let $p \in \mathbb { R } ^ { d _ { c } }$ denote the frozen CLIP text representation of the serialized behavior profile. We first compute the raw CLIP similarity

$$
r _ { i } = \alpha \cos ( p , x _ { i } ) ,\tag{15}
$$

where $\alpha$ is a learnable scale. Lightweight projections map both modalities to a shared d-dimensional space:

$$
\begin{array} { r } { \hat { p } = \mathrm { L N } ( \mathrm { G E L U } ( W _ { p } p + b _ { p } ) ) , } \\ { \hat { x } _ { i } = \mathrm { L N } ( \mathrm { G E L U } ( W _ { x } x _ { i } + b _ { x } ) ) . } \end{array}\tag{16}
$$

The projected similarity $c _ { i } = \alpha \cos ( \hat { p } , \hat { x } _ { i } )$ is combined with the raw CLIP score through a constrained residual weight:

$$
m _ { i } = \beta r _ { i } + ( 1 - \beta ) c _ { i } , \qquad \beta = 0 . 9 + 0 . 1 \sigma ( b ) .\tag{17}
$$

This parameterization preserves the pretrained CLIP geometry while allowing a small task-specific correction.

To encode local temporal structure without selfattention, we construct

$$
\begin{array} { r l } & { d _ { i } = \big [ \hat { x } _ { i } - \hat { x } _ { i - 1 } ; \hat { x } _ { i } - \hat { x } _ { i + 1 } ; \quad \ell _ { i } ; b _ { i } ^ { \mathrm { L } } ; b _ { i } ^ { \mathrm { R } } \big ] , } \\ & { \qquad h _ { i } = \mathrm { M L P } _ { \mathrm { t e m p } } ( d _ { i } ) . } \end{array}\tag{18}
$$

![](images/2674e4810816907dcba2821756d1006d31b99d6544b314c4fefaf9fed1c484c4.jpg)  
Figure 3. Behavior-conditioned evidence scoring and structured counterfactual learning. Frozen CLIP features are processed by lightweight projections, a CLIP residual branch, and a relative-temporal MLP to produce segment evidence logits. Their ranking answers Where, while parameter-free mean pooling answers Whether. Factual $V ^ { + }$ and matched counterfactual $V ^ { - }$ share parameters and are jointly supervised by localization, existence-ordering, local-intervention, and distractor-invariance losses. Gradients update only the lightweight CBGER scorer; CLIP remains frozen.

where $\ell _ { i }$ is the normalized segment duration and $b _ { i } ^ { \mathrm { L } } , b _ { i } ^ { \mathrm { R } }$ indicate timeline boundaries. The final evidence logit is

$$
s _ { i } = m _ { i } + \gamma \cos ( \hat { p } , h _ { i } ) , \qquad \gamma = 0 . 0 2 \sigma ( g ) .\tag{19}
$$

The bounded gate makes temporal context a light correction rather than a replacement for semantic similarity. The localized prediction is

$$
\hat { y } = \arg \operatorname* { m a x } _ { i } s _ { i } ,\tag{20}
$$

and ranking $\{ s _ { i } \} _ { i = } ^ { N }$ defines the Where pathway.

## 4.3. Video-Level Evidence Estimation

Localization does not by itself establish that valid evidence exists: after evidence replacement, a maximum operator must still select the strongest distractor. We therefore use parameter-free mean pooling for the Whether score:

$$
q ( u , V ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } s _ { i } .\tag{21}
$$

This rule evaluates support over the complete controlled timeline, introduces no additional parameters, and is compared with alternative aggregators in the ablation study.

## 4.4. Intervention Consistency

The intervention task measures whether the score of the edited focal moment decreases when its supporting evidence is replaced. For a factual–counterfactual pair $( V ^ { + } , V ^ { - } )$ with the intervention applied at slot $y ,$ we define

$$
I ( u , V ^ { + } , V ^ { - } ) = { \bf 1 } \big [ s _ { y } ( u , V ^ { + } ) > s _ { y } ( u , V ^ { - } ) \big ] .\tag{22}
$$

The dataset-level Intervention score is the mean of this indicator over all M matched pairs:

$$
\mathrm { I n t e r v e n t i o n } = { \frac { 1 } { M } } \sum _ { n = 1 } ^ { M } I ( u _ { n } , V _ { n } ^ { + } , V _ { n } ^ { - } ) .\tag{23}
$$

Unlike Whether, which compares the video-level scores $q ( u , V ^ { + } )$ and $q ( u , V ^ { - } )$ , Intervention evaluates the model’s local response at the known edited slot.

## 4.5. Structured Counterfactual Learning

Let $V ^ { + }$ be a factual timeline with evidence at slot $y ,$ , and let $V ^ { - }$ be its matched counterfactual obtained by replacing only that slot. Both timelines are processed with shared parameters, producing $S ^ { + } = \left\{ s _ { i } ^ { + } \right\}$ and $S ^ { - } = \{ s _ { i } ^ { - } \}$

Factual localization. We supervise the evidence position using temperature-scaled cross entropy and a hard-negative ranking margin:

$$
\mathcal { L } _ { \mathrm { l o c } } = - l o g \frac { \exp ( s _ { y } ^ { + } / \tau ) } { \sum _ { i = 1 } ^ { N } \exp ( s _ { i } ^ { + } / \tau ) } ,\tag{24}
$$

$$
\mathcal { L } _ { \mathrm { r a n k } } = \frac { 1 } { N - 1 } \sum _ { i \neq y } [ \delta _ { \mathrm { r a n k } } - s _ { y } ^ { + } + s _ { i } ^ { + } ] _ { + } .\tag{25}
$$

where $\delta _ { \mathrm { r a n k } } > 0$ is the ranking margin.

Video-level evidence ordering. The factual timeline should provide stronger support than its counterfactual:

$$
\mathcal { L } _ { \mathrm { w h e t h e r } } = \log \biggl ( 1 + \exp \biggl [ - \frac { q ( u , V ^ { + } ) - q ( u , V ^ { - } ) } { \tau _ { q } } \biggr ] \biggr ) _ { / \sim }\tag{26}
$$

This objective directly matches the factual–counterfactual ordering measured by Pair Accuracy.

Local intervention and distractor invariance. Because only the focal slot is edited, its score should decrease while the unchanged distractors should remain stable:

$$
\mathcal { L } _ { \mathrm { i n t } } = [ \delta _ { \mathrm { i n k } } - s _ { y } ^ { + } + s _ { y } ^ { - } ] _ { + } ,\tag{27}
$$

$$
\mathcal { L } _ { \mathrm { k e e p } } = \frac { 1 } { | D | } \sum _ { j \in D } ( s _ { j } ^ { + } - s _ { j } ^ { - } ) ^ { 2 } ,\tag{28}
$$

where $\delta _ { \mathrm { i n k } } > 0$ is the intervention margin and D indexes the eight shared distractor slots. The latter term prevents a trivial global shift of all counterfactual scores.

The complete objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { l o c } } + \lambda _ { r } \mathcal { L } _ { \mathrm { r a n k } } + \lambda _ { q } \mathcal { L } _ { \mathrm { w h e t h e r } } + \lambda _ { i } \mathcal { L } _ { \mathrm { i n t } } + \lambda _ { k } \mathcal { L } _ { \mathrm { k e e p } } . } \end{array}\tag{29}
$$

All losses are computed from the shared factual and counterfactual forward passes. Gradients update the projections, residual and temporal gates, relative-temporal MLP, while the CLIP image and text encoders remain frozen.

## 5. Experiments

## 5.1. Experimental Setup

All methods use the same frozen CBGER-10K train/validation/test splits and CLIP ViT-B/32 visual and text representations [23]. CBGER uses hidden dimension $\begin{array} { l } { d \ = \ 2 5 6 . } \end{array}$ , a two-layer relative-temporal MLP, AdamW with learning rate $2 \times 1 0 ^ { - 4 }$ , and weight decay $1 0 ^ { - 4 }$ . Hyperparameters and checkpoints are selected on the validation set; the test set is used only for final evaluation. Results are averaged over three random seeds and reported as mean±standard deviation.

We compare with six baselines: frozen CLIP profile– segment similarity; PR-Net for personalized highlight detection [5]; QD-DETR [20]; TR-DETR [24]; FlashVTG [3]; and MQVTG [25]. For temporal-grounding baselines, the user’s behavior profile replaces the original language query. All methods use the same candidate timelines, downstream representations, and data splits, and are evaluated using a single implementation of the CBGER metrics.

## 5.2. Main Results

As shown in Table 2, CBGER achieves the best performance on all three evaluation axes. The most informative comparison is with QD-DETR. Their localization performance is close (MRR .4432 vs. .4231), but CBGER improves PairAcc by 0.1103 and Intervention Consistency by 0.0470. TR-DETR obtains the strongest baseline PairAcc at .6320, still 6.57 points below CBGER.

We assess significance using 10,000 paired bootstrap replicates clustered by user, preserving all predictions associated with each sampled user. Relative to QD-DETR, the MRR difference is not significant (+0.0201, 95% CI [−0.0027, 0.0433], p = .0832), whereas the PairAcc gain is substantial (+0.1103, 95% CI [0.0765, 0.1430], $p < 1 0 ^ { - 4 } )$ and the Intervention gain is also significant (+0.0470, 95% $\mathbf { C I } \ [ 0 . 0 1 6 6 , 0 . 0 7 7 2 ] , p = . 0 0 2 2 )$ . Thus, two models can be statistically similar in temporal localization while differing markedly in personalized evidence existence and intervention response.

## 5.3. Where–Whether Ablation

We next separate the contribution of structured counterfactual learning from the video-level aggregation rule.

As shown in Table 3, structured counterfactual training increases MRR from .4274 to .4432 and Intervention from .6830 to .6987, showing that paired supervision improves both factual localization and sensitivity to the edited slot. Mean pooling does not alter segment rankings and therefore leaves MRR and Intervention unchanged, but improves PairAcc by 5.8 points without counterfactual training and 6.8 points with it. Under mean pooling, counterfactual training further improves PairAcc by 1.67 points. The interaction between the two components is not statistically significant, indicating complementary rather than super-additive effects.

## 5.4. How Should Evidence Existence Be Aggregated?

To isolate the Whether decision, we keep CBGER segment scores fixed and vary only the video-level aggregation rule.

As shown in Table 4, max pooling performs poorly because each factual–counterfactual pair shares eight hard distractors; the maximum is therefore often determined by an unchanged segment and becomes insensitive to the focal replacement. Increasing the number of aggregated segments progressively improves PairAcc. A learned pointwise existence head also underperforms, indicating that additional capacity alone does not resolve the existence decision. Mean pooling provides the strongest performance while introducing no additional parameters.

Table 2. Results on CBGER-10K. PairAcc measures factual–counterfactual evidence ordering, and Intervention measures the local score response at the replaced evidence slot. Best results are in bold and second-best are underlined.
<table><tr><td>Method</td><td>MRR</td><td>NDCG@1</td><td>NDCG@3</td><td>NDCG@5</td><td>PairAcc</td><td>Intervention</td></tr><tr><td>Frozen CLIP similarity</td><td> $. 4 1 2 7 { \pm } . 0 0 0 0$ </td><td> $. 2 0 6 0 { \scriptstyle \pm . 0 0 0 0 }$ </td><td> $. 3 5 8 8 { \pm } . 0 0 0 0$ </td><td> $. 4 4 6 1 { \pm } . 0 0 0 0$ </td><td> $. 6 3 2 0 { \scriptstyle \pm . 0 0 0 0 }$ </td><td> $. 6 3 2 0 { \scriptstyle \pm . 0 0 0 0 }$ </td></tr><tr><td>PR-Net</td><td> $. 3 8 5 0 { \scriptstyle \pm . 0 0 4 7 }$ </td><td> $. 1 6 2 7 { \pm } . 0 0 9 8$ </td><td> $. 3 3 6 8 { \pm } . 0 1 3 0$ </td><td> $. 4 2 7 1 { \pm } . 0 1 5 1$ </td><td> $. 5 8 1 0 { \pm } . 0 4 3 6$ </td><td> $. 6 2 3 3 { \pm } . 0 1 5 0$ </td></tr><tr><td>QD-DETR</td><td> $\underline { { . 4 2 3 1 \pm . 0 0 1 1 } }$ </td><td> $\underline { { 2 0 6 3 } } \pm . 0 0 5 1 $ </td><td> $. 3 7 8 4 \pm . 0 0 1 8 $ </td><td> $. 4 6 6 2 { \pm } . 0 0 1 6$ </td><td> $. 5 8 7 3 { \pm } . 0 0 8 0$ </td><td> $\underline { { . 6 5 1 7 \pm . 0 2 1 0 } }$ </td></tr><tr><td>TR-DETR</td><td> $. 4 1 2 5 { \pm } . 0 2 3 6$ </td><td> $. 2 0 4 3 { \pm } . 0 1 5 4$ </td><td> $. 3 6 3 0 { \pm } . 0 3 5 7$ </td><td> $. 4 5 0 1 { \pm } . 0 3 8 0$ </td><td> $\underline { { 6 3 2 0 } } \pm . 0 2 1 2 $ </td><td> $. 6 5 2 7 { \pm } . 0 2 0 2$ </td></tr><tr><td>FlashVTG</td><td> $. 4 1 5 4 { \pm } . 0 0 6 1$ </td><td> $. 2 0 1 3 { \pm } . 0 0 9 9$ </td><td> $. 3 7 3 6 { \pm } . 0 0 8 1$ </td><td> $. 4 5 4 9 { \pm } . 0 0 6 0$ </td><td> $. 6 2 0 7 { \scriptstyle \pm . 0 1 5 3 }$ </td><td> $. 6 3 2 7 { \pm } . 0 0 2 9$ </td></tr><tr><td>MQVTG</td><td> $. 4 1 1 2 { \pm } . 0 1 3 7$ </td><td> $. 1 9 4 0 { \pm } . 0 1 4 8$ </td><td> $. 3 6 2 4 { \pm } . 0 1 4 6$ </td><td> $. 4 5 5 2 { \pm } . 0 2 2 3$ </td><td> $. 6 0 7 3 { \scriptstyle \pm . 0 2 5 7 }$ </td><td> $. 6 3 8 0 { \pm } . 0 1 1 8$ </td></tr><tr><td>CBGER</td><td> $\mathbf { \sigma } _ { \mathbf { \sigma } } \mathbf { 4 4 3 2 \pm . 0 1 } 2 2$ </td><td> ${ \bf . 2 3 6 3 { \pm } . 0 0 9 9 }$ </td><td> $\mathbf { \pm 0 0 8 } \pm \mathbf { . 0 1 8 7 }$ </td><td> $\mathbf { \delta } . 4 9 \mathbf { 0 } 1 { \pm } . 0 1 3 3$ </td><td> $\mathbf { 6 9 7 7 } { \pm } \mathbf { . 0 0 7 0 }$ </td><td> $\mathbf { 6 9 8 7 \pm . 0 0 3 5 }$ </td></tr></table>

Table 3. Structured counterfactual learning (CF) and mean evidence pooling.
<table><tr><td>CF</td><td>Mean</td><td>MRR</td><td>PairAcc</td><td>Intervention</td></tr><tr><td>x</td><td>x</td><td> $. 4 2 7 4 { \scriptstyle \pm . 0 1 0 0 }$ </td><td> $. 6 2 3 0 { \pm } . 0 0 4 6$ </td><td> $. 6 8 3 0 { \pm } . 0 0 7 9$ </td></tr><tr><td>x</td><td>√</td><td> $. 4 2 7 4 { \scriptstyle \pm . 0 1 0 0 }$ </td><td> $. 6 8 1 0 { \pm } . 0 0 8 5$ </td><td> $. 6 8 3 0 { \pm } . 0 0 7 9$ </td></tr><tr><td>√</td><td>x</td><td> $. 4 4 3 2 { \pm } . 0 1 2 2$ </td><td> $. 6 2 9 7 { \scriptstyle \pm . 0 0 8 5 }$ </td><td> $. 6 9 8 7 { \pm } . 0 0 3 5$ </td></tr><tr><td>√</td><td>√</td><td> $\mathbf { \sigma } _ { \mathbf { \sigma } \mathbf { 4 4 } 3 2 \pm . 0 1 2 2 }$ </td><td> $\mathbf { 6 9 7 7 } { \pm } . \mathbf { 0 0 7 0 }$ </td><td> ${ \bf . 6 9 8 7 \pm . 0 0 3 5 }$ </td></tr></table>

Table 4. Video-level evidence aggregation with fixed segment scores.
<table><tr><td>Aggregation rule</td><td>PairAcc</td></tr><tr><td>Max</td><td>.4860±.0200</td></tr><tr><td>Top-2 mean</td><td> $. 5 6 6 0 { \pm } . 0 1 6 6$ </td></tr><tr><td>Top-3 mean</td><td> $. 6 1 6 0 { \scriptstyle \pm . 0 1 7 7 }$ </td></tr><tr><td>Learned existence MLP</td><td> $. 4 5 6 0 { \pm } . 0 0 3 5$ </td></tr><tr><td>Mean</td><td> $\mathbf { 6 9 7 7 } 2 2 . 0 0 7 0$ </td></tr></table>

## 5.5. Qualitative Intervention Analysis

Two illustrative intervention-success cases are provided in the Appendix. The baselines frequently identify visually plausible or topically related factual segments, yet fail to lower either the video-level score or the focal segment score after the supporting moment is replaced. CBGER instead preserves the factual–counterfactual ordering and produces a localized score decrease at the edited slot. These examples illustrate the quantitative finding that plausible temporal localization alone does not guarantee reliable evidence verification.

## 6. Limitations and responsible interpretation

CBGER-10K is a controlled diagnostic benchmark with several limitations. Its automatically constructed Qwen3- VL/BGE-M3 labels may contain semantic and matching biases; counterfactual replacements are behavior-weaker under the construction criteria, but are not guaranteed to be universally irrelevant. Because user histories come from MicroLens and candidate videos from FineVideo, the benchmark measures behavior-supported evidence rather than actual user exposure, preference, or causal engagement. The fixed nine-segment timelines and frozen CLIP backbone improve experimental control but limit generalization to unrestricted videos and other representations. Finally, PairAcc measures relative evidence ordering within matched pairs, not calibrated engagement probability; CBGER-10K should therefore be viewed as a diagnostic benchmark for personalized evidence reasoning rather than a simulation of deployed recommendation systems.

## 7. Conclusion

We introduced counterfactual behavior-grounded evidence retrieval, which separates Where personalized video evidence occurs from Whether such evidence exists and whether model predictions respond consistently to its replacement. CBGER-10K provides 5,000 controlled factual–counterfactual pairs for evaluating these capabilities, and CBGER jointly learns localization, evidence existence, and intervention consistency through structured counterfactual supervision. Experiments show that strong temporal localization does not necessarily imply reliable personalized evidence existence, while CBGER consistently improves Pair Accuracy and Intervention Consistency over existing grounding and personalized-highlight baselines. More broadly, our results suggest that evaluating only where a model attends can hide failures in whether the retrieved content is supported by the benchmark’s behavior-derived evidence criteria; controlled interventions provide a practical way to expose this gap.

## References

[1] Taivanbat Badamdorj, Mrigank Rochan, Yang Wang, and Li Cheng. Contrastive learning for unsupervised video highlight detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14042–14052, 2022. 3

[2] Hritik Bansal, Yonatan Bitton, Idan Szpektor, Kai-Wei Chang, and Aditya Grover. VideoCon: Robust videolanguage alignment via contrast captions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13927–13937, 2024. 3

[3] Zhuo Cao, Bingqing Zhang, Heming Du, Xin Yu, Xue Li, and Sen Wang. FlashVTG: Feature layering and adaptive score handling network for video temporal grounding. In IEEE/CVF Winter Conference on Applications of Computer Vision, 2025. 3, 7

[4] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. BGE M3-embedding: Multilingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024. 4

[5] Runnan Chen, Penghao Zhou, Wenzhe Wang, Nenglun Chen, Pai Peng, Xing Sun, and Wenping Wang. A novel architecture for personalized video highlight detection. In ICCV, 2021. 3, 7

[6] Xu Chen, Jingsen Zhang, Lei Wang, Quanyu Dai, Zhenhua Dong, Ruiming Tang, Rui Zhang, Li Chen, Xin Zhao, and Ji-Rong Wen. REASONER: An explainable recommendation dataset with comprehensive labeling ground truths. In Advances in Neural Information Processing Systems, pages 14497–14515, 2023. 3

[7] Zixu Cheng, Yujiang Pu, Shaogang Gong, Parisa Kordjamshidi, and Yu Kong. SHINE: Saliency-aware hierarchical negative ranking for compositional temporal grounding. In European Conference on Computer Vision, pages 398–416, 2024. 3

[8] Nikita Dvornik, Isma Hadji, Ran Zhang, Konstantinos G. Derpanis, Richard P. Wildes, and Allan D. Jepson. Step-Former: Self-supervised step discovery and localization in instructional videos. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18952–18961, 2023. 3

[9] Miquel Farre, Andres Marafioti, Lewis Tunstall, Leandro´ von Werra, Pedro Cuenca, and Thomas Wolf. Finevideo: Behind the scenes. Hugging Face Blog and dataset card, https : / / huggingface . co / datasets / HuggingFaceFV/finevideo, 2024. 1, 3

[10] Kevin Flanagan, Dima Damen, and Michael Wray. Moment of untruth: Dealing with negative queries in video moment retrieval. In IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 5336–5345, 2025. 3

[11] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In ICCV, 2017. 1, 2

[12] Michael Gygli, Yale Song, and Liangliang Cao. Video2gif: Automatic generation of animated gifs from video. In CVPR, 2016. 3

[13] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language. In ICCV, 2017. 1, 2

[14] Phillip Howard, Avinash Madasu, Tiep Le, Gustavo Lujan Moreno, Anahita Bhiwandiwalla, and Vasudev Lal. Socialcounterfactuals: Probing and mitigating intersectional social biases in vision-language models with counterfactual examples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11975–11985, 2024. 3

[15] Jinhyun Jang, Jungin Park, Jin Kim, Hyeongjun Kwon, and Kwanghoon Sohn. Knowing where to focus: Event-aware transformer for video grounding. In IEEE/CVF International Conference on Computer Vision, pages 13846–13856, 2023. 3

[16] Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In ICCV, 2017. 2

[17] Jie Lei, Tamara L. Berg, and Mohit Bansal. Detecting moments and highlights in videos via natural language queries. In NeurIPS, 2021. 1, 2

[18] Kevin Qinghong Lin, Pengchuan Zhang, Joya Chen, Shra man Pramanick, Difei Gao, Alex Jinpeng Wang, Rui Yan, and Mike Zheng Shou. Univtg: Towards unified video language temporal grounding. In ICCV, 2023. 1, 2

[19] Zihang Lin, Chaolei Tan, Jian-Fang Hu, Zhi Jin, Tiancai Ye, and Wei-Shi Zheng. Collaborative static and dynamic vision-language streams for spatio-temporal video grounding. In IEEE/CVF Conference on Computer Vision and Pat tern Recognition, pages 23100–23109, 2023. 3

[20] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo. Query-dependent video representation for moment retrieval and highlight detection. In CVPR, 2023. 1, 2, 7

[21] Yongxin Ni, Yu Cheng, Xiangyan Liu, Junchen Fu, Youhua Li, Xiangnan He, Yongfeng Zhang, and Fajie Yuan. A content-driven micro-video recommendation dataset at scale. arXiv preprint arXiv:2309.15379, 2023. 1, 3

[22] Qwen Team. Qwen3-VL-8B-Instruct model card. https://huggingface.co/Qwen/Qwen3- VL-8B-Instruct, 2025. 4

[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 4, 5, 7

[24] Hao Sun, Mingyao Zhou, Wenjing Chen, and Wei Xie. TR-DETR: Task-reciprocal transformer for joint moment retrieval and highlight detection. In AAAI Conference on Artificial Intelligence, 2024. 3, 7

[25] Xiaolong Sun, Le Wang, Sanping Zhou, Liushuai Shi, Kun Xia, Mengnan Liu, Yabing Wang, and Gang Hua. MQVTG: Moment quantization for video temporal grounding. In IEEE/CVF International Conference on Computer Vision, 2025. 3, 7

[26] Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can i trust your answer? visually grounded video question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13204– 13214, 2024. 3

[27] Yicheng Xiao, Zhuoyan Luo, Yong Liu, Yue Ma, Hengwei Bian, Yatai Ji, Yujiu Yang, and Xiu Li. Bridging the gap: A unified video comprehension framework for moment retrieval and highlight detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18709– 18719, 2024. 3

[28] Chun-Hsiao Yeh, Bryan Russell, Josef Sivic, Fabian Caba Heilbron, and Simon Jenni. Meta-personalizing visionlanguage models to find named instances in video. In CVPR, 2023. 3

[29] Xi Zhang, Hao Fei, Zhaohui Li, Fei Wu, and Tat-Seng Chua. What if the tv was off? examining counterfactual reasoning abilities of multi-modal language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21853–21862, 2024. 3

[30] Yongfeng Zhang and Xu Chen. Explainable recommendation: A survey and new perspectives. Foundations and Trends in Information Retrieval, 14(1):1–101, 2020. 3