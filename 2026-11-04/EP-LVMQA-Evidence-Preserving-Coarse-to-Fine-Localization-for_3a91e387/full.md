# EP-LVMQA: Evidence-Preserving Coarse-to-Fine Localization for Low-Quality Long Video Moment Question Answering

Chen Guoning<sup>1</sup>, Zhexun Hu<sup>1</sup>, Yifan Wei<sup>2,3</sup>, Yixuan Weng<sup>4</sup>, Bin Li<sup>5</sup>, Zhao Yufei<sup>1</sup>

<sup>1</sup>Nanyang Technological University, Singapore (Chen Guoning and Zhexun Hu: M.S. students; Zhao Yufei: Researcher)

<sup>2</sup>ByteDance Inc. (Researcher)

<sup>3</sup>State Key Laboratory of Complex and Critical Software Environment, School of Computer Science and Engineering, Beihang University (Ph.D. student)

<sup>4</sup>Westlake University (Research Assistant)

<sup>5</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences (Researcher)

## Abstract

Long instructional videos are increasingly used in medical education, industrial training, and procedural knowledge transfer, yet their evidence is often incomplete or unreliable: subtitles may be missing, ASR noisy, OCR sparse, and decisive actions briefly shown without narration. Existing VideoQA systems typically return labels or free-form text, making answer support dificult to inspect. We study Long Video Moment Question Answering (LVMQA), where a system returns timestamped answer-supporting evidence. We propose EP-LVMQA, an evidence-preserving coarseto-fine framework comprising an Uncertainty-Aware Multimodal Evidence Timeline, a Coarse Evidence Planner, and an Evidence-Conditioned Fine Localizer. An evidence card transfers planner scores, matched cues, source-confidence proxies, modality-specific evidence, neighboring context, and a temporal prior between the latter two modules. On Med-VidQA, EP-LVMQA achieves 26.1 R1@0.5 and 31.8 mIoU, improving the strongest non-ours results by 6.3 and 5.1 points. It raises Suf.@2 and Ver.@2 to 25.4 and 22.6, gains of 9.5 and 8.8 points. Under matched low-quality perturbations, average relative degradation is reduced from the strongest non-ours value of14.6% to 11.1%. On Ego4D-NLQ and QVHighlights, EP-LVMQA remains broadly comparable to established taskspecific localization methods.

## 1 Introduction

Video question answering (VideoQA) increasingly couples answer prediction with temporal grounding (Jang et al. 2017; Lei et al. 2018; Xiao et al. 2024; Di and Xie 2024). Natural-language video localization has likewise progressed from proposal-based matching to set prediction and unified grounding (Gao et al. 2017; Lei, Berg, and Bansal 2021; Moon et al. 2023b; Lin et al. 2023). Yet most systems either generate an answer or retrieve a query-matching interval; they do not preserve the multimodal provenance needed to inspect whether the interval actually supports the answer.

This gap matters in long instructional videos, where answer evidence spans unreliable sources. MedVidQA pairs medical questions with timestamped visual answers (Gupta, Attal, and Demner-Fushman 2023), but real recordings may contain noisy ASR, sparse OCR, and plausible yet visually unsupported captions. Answer-seeking questions may also express needs that do not literally name the target action.

Long-video methods reduce search cost through window selection and coarse-to-fine alignment (Hou et al. 2023b; Pan et al. 2023; Barrios et al. 2023; Mu, Mo, and Li 2024). Figure 1(a) abstracts the text-driven variant: ASR or OCR guides coarse retrieval, but only candidate clips, features, or relevance scores reach the fine localizer. This handof discards source identity, confidence, availability masks, matched cues, neighboring context, and the temporal prior.

The discarded information afects localization quality, not merely the explanation produced afterward. A generated caption can make a broadly related window rank highly even when a brief visual action in a neighboring window provides the decisive evidence. A clip-conditioned localizer cannot distinguish mutually supporting sources from a single unreliable cue or recover the planner’s temporal rationale. The retrieval-to-localization interface therefore determines whether the model can reject misleading evidence, refine both boundaries, and expose why the returned moment supports the answer.

We therefore study Long Video Moment Question Answering (LVMQA): given a long video and an answer-seeking question, return a timestamped primary evidence moment rather than unconstrained text. Figure 1 contrasts the two interfaces. Text-driven methods pass candidate clips and relevance scores to a clip-conditioned localizer, often yielding a loose interval without an inspectable trace. EP-LVMQA instead transfers a multimodal evidence card with sourcespecific payloads and retrieval metadata, enabling routing and prior-guided refinement. The target is a finely bounded primary evidence moment accompanied by an evidence trace.

EP-LVMQA operationalizes this design with an Uncertainty-Aware Multimodal Evidence Timeline, a Coarse Evidence Planner, and an Evidence-Conditioned Fine Localizer. The card transfers the planner score, modality payloads, confidence proxies, availability masks, matched cues, neighboring context, and a temporal prior. Its retained fields also support the returned evidence trace.

Our contributions are threefold: (1) we formulate lowquality instructional VideoQA as localization of inspectable answer evidence; (2) we introduce an Evidence-Preserving Candidate Interface with source-aware routing and priorguided refinement; and (3) on MedVidQA, controlled perturbations, and two cross-benchmark checks, we show gains in localization, robustness, and human-audited evidence quality while identifying planner misses and imperfect faithfulness as limitations.

![](images/19d57fdf9ca987100819bd08e8ea643170f3b01da772b6f05ac6a82a192f1ea6.jpg)  
Figure 1: (a) Text-driven methods pass ASR/OCR-retrieved clips or scores to a clip-conditioned localizer, yielding a loose interval without provenance. (b) EP-LVMQA transfers multimodal evidence and source-aware metadata in an evidence card, enabling routed, prior-guided refinement and an inspectable primary evidence moment.

## 2 Related Work

Grounded VideoQA. VideoQA links answers to spatiotemporal evidence (Jang et al. 2017; Lei et al. 2018; Xiao et al. 2024; Di and Xie 2024). Instructional grounding spans TACoS and HowTo100M (Regneri et al. 2013; Miech et al. 2019). EP-LVMQA instead outputs timestamped evidence with provenance.

Temporal grounding. Methods include proposal/span prediction (Gao et al. 2017; Hendricks et al. 2017; Ghosh et al. 2019; Zhang et al. 2020a), 2D maps (Zhang et al. 2020b), DETR localization (Lei, Berg, and Bansal 2021; Moon et al. 2023a,b), joint retrieval/highlights (Liu et al. 2022; Lin et al. 2023), and long-video grounding (Hou et al. 2023b; Pan et al. 2023; Mu, Mo, and Li 2024; Yuan et al. 2025). Movie grounding and Video-LLMs extend duration (Soldan et al. 2022; Huang et al. 2024; Wang et al. 2025a). We instead preserve evidence across stages.

Long-video evidence selection. Sparse selection can erase source identity. Pretraining, missing-modality modeling, calibration, and diversity reranking improve reliability (Alayrac et al. 2020; Bain et al. 2021; Neverova et al. 2016; Ma et al. 2022; Guo et al. 2017; Carbonell and Goldstein 1998). Ego4D methods reason over localized windows (Grauman et al. 2022; Hou et al. 2023a). EP-LVMQA exposes an auditable moment; MomentSeeker is only an external MedVidQA planner baseline.

## 3 Task Formulation

Given $\mathcal { D } = \{ ( V _ { i } , q _ { i } , M _ { i } ^ { * } ) \} _ { i = 1 } ^ { N }$ , where $M _ { i } ^ { * }$ contains annotated answer-supporting intervals, the model predicts

$$
\widehat { \cal M } = \{ ( \hat { s } _ { k } , \hat { e } _ { k } , \hat { c } _ { k } , \hat { r } _ { k } ) \} _ { k = 1 } ^ { N _ { b } } .\tag{1}
$$

Here $( \hat { s } _ { k } , \hat { e } _ { k } )$ is an evidence moment, $\hat { c } _ { k }$ its confidence, and $\hat { r } _ { k }$ an externally inspectable provenance record rather than a hidden reasoning chain. Although multiple intervals are permitted, we evaluate the top-ranked prediction against the best-overlapping annotation. At temporal unit $t ,$ we retain

$$
y _ { t } = \{ v _ { t } , x _ { t } ^ { \mathrm { a s r } } , x _ { t } ^ { \mathrm { o c r } } , x _ { t } ^ { \mathrm { c a p } } , m _ { t } , c _ { t } \} ,\tag{2}
$$

where $m _ { t }$ and $c _ { t }$ encode source availability and confidence. Generated captions are uncertain cues, not ground-truth subtitles.

The Evidence-Preserving Candidate Interface transfers a model-facing evidence card from the planner to the localizer. We distinguish this card from the returned evidence moment and its output-facing evidence trace. This distinction prevents the interface from being reduced to a post-hoc explanation.

## 4 Method

Figure 2 organizes EP-LVMQA into three modules. Question-Conditioned Evidence Cues condition the planner and localizer, which exchange an evidence card through the Evidence-Preserving Candidate Interface.

## 4.1 Uncertainty-Aware Multimodal Evidence Timeline

Visual features, ASR, OCR, and generated captions are aligned into overlapping windows without collapsing source identity. For modality m in window $j ,$ a label-free confidence proxy combines availability, calibrated native confidence, and local cross-modal agreement:

$$
\rho _ { j } ^ { m } = \frac { \lambda _ { a } A _ { j } ^ { m } + \lambda _ { c } u _ { j } ^ { m } C _ { j } ^ { m } + \lambda _ { g } G _ { j } ^ { m } } { \lambda _ { a } + \lambda _ { c } u _ { j } ^ { m } + \lambda _ { g } } .\tag{3}
$$

We use $( \lambda _ { a } , \lambda _ { c } , \lambda _ { g } ) = ( 0 . 4 , 0 . 3 , 0 . 3 )$ and set $\rho _ { j } ^ { m } = 0$ when unavailable. Caption and visual streams have no native confidence. This proxy guides planning and routing but is not claimed to estimate true reliability.

Native ASR/OCR confidence is clipped at the 1st and 99th training percentiles and min–max calibrated; $G _ { j } ^ { m }$ is mean projected cosine agreement with available streams within ±4 s. The planner uses the masked mean confidence, while the localizer learns question-dependent source weights. Separating these operations avoids using the localizer’s learned routing score as evidence for its own candidate selection.

![](images/8c6b167887c93e9ec7ac9b65738880d35f2264ffd7a3bf7a4f4a15abd92b2eb8.jpg)  
Figure 2: EP-LVMQA comprises three modules: (a) an Uncertainty-Aware Multimodal Evidence Timeline, (b) a Coarse Evidence Planner, and (c) an Evidence-Conditioned Fine Localizer. Question-Conditioned Evidence Cues condition the planner and localizer, which communicate through an Evidence-Preserving Candidate Interface implemented as an evidence card.

## 4.2 Coarse Evidence Planner

The planner compresses the search space while preserving candidate recall. Each window retains its span, source features, mask, and mean confidence.

Question-Conditioned Evidence Cues. To bridge indirect questions to observable actions, a fixed rule-assisted parser extracts action/verb-particle, object, temporal, and high-IDF domain cues, retaining at most four per type. A cue survives only if its maximum text/visual/caption similarity to memory exceeds $\theta _ { c } = 0 . 3 0$ ; otherwise only the question embedding is used. The planner scores a window as

$$
S _ { j } = f _ { \theta } ( [ h _ { q } ; h _ { j } ^ { \mathrm { v i s } } ; h _ { j } ^ { \mathrm { a s r } } ; h _ { j } ^ { \mathrm { o c r } } ; h _ { j } ^ { \mathrm { c a p } } ; \bar { \rho } _ { j } ; m _ { j } ; \psi _ { j } ] ) ,\tag{4}
$$

where $\psi _ { j }$ encodes position, duration, and neighboring agreement. MMR selection $( \lambda _ { \mathrm { M M R } } = 0 . 2 0 )$ reduces redundancy; adjacent windows within 4 s are merged subject to a 96 s cap and a global duration budget.

The planner targets evidence recall rather than boundary accuracy. Its compact candidate union should cover the decisive moment; candidate duration and an oracle upper bound separate search from localization failure.

Evidence-Preserving Candidate Interface. The interface instantiates each selected candidate $j$ as an evidence card

$$
p _ { j } = \{ \tau _ { j } , S _ { j } , \Omega _ { j } , \rho _ { j } , m _ { j } , \xi _ { j } , \nu _ { j } , \pi _ { j } \} ,\tag{5}
$$

containing its span, score, modality payload, confidence and mask, matched cues, neighboring context, and a Gaussian temporal prior. Training labels and human audit scores are excluded from model input.

The prior is centered on the highest-scoring constituent window, with standard deviation equal to half the window length. Immediate neighboring summaries help distinguish repeated steps without restoring the full video. Model-input fields are versioned separately from training-only IoU labels and audit-only judgments.

## 4.3 Evidence-Conditioned Fine Localizer

The localizer encodes card metadata, question tokens, and modality tokens, then refines each card into a bounded evidence moment. A question-conditioned router assigns

$$
\omega _ { j } ^ { m } = \frac { m _ { j } ^ { m } \exp ( W _ { r } [ h _ { q } ; h _ { j } ^ { m } ; \rho _ { j } ^ { m } ] ) } { \sum _ { m ^ { \prime } } m _ { j } ^ { m ^ { \prime } } \exp ( W _ { r } [ h _ { q } ; h _ { j } ^ { m ^ { \prime } } ; \rho _ { j } ^ { m ^ { \prime } } ] ) + \epsilon } ,\tag{6}
$$

so unavailable sources receive zero weight. A transformer fuses routed tokens and residual cue attention. Start and end distributions use the planner prior as a soft boundary bias:

$$
P _ { s } ^ { j } ( t ) = \mathrm { s o f t m a x } ( W _ { s } H _ { j } ( t ) + \eta \log \pi _ { j } ( t ) ) ,\tag{7}
$$

$$
P _ { e } ^ { j } ( t ) = \mathrm { s o f t m a x } ( W _ { e } H _ { j } ( t ) + \eta \log \pi _ { j } ( t ) ) ,\tag{8}
$$

with $\eta = 0 . 5$ . Candidate predictions are ranked by planner score plus start/end log-likelihood; the top interval is returned with its evidence trace.

More precisely, valid boundaries maximize $P _ { s } ^ { j } ( s ) P _ { e } ^ { j } ( e )$ under $s \leq$ e and the 96 s cap. Global ranking uses

$$
\ell _ { j } = S _ { j } + \log P _ { s } ^ { j } ( \hat { s } _ { j } ) + \log P _ { e } ^ { j } ( \hat { e } _ { j } ) .\tag{9}
$$

The trace records the card identifier, timestamps, cues, source masks, confidence proxies, and planner/localizer scores. These fields support inspection but do not prove causal reliance.

## 4.4 Training

Training is sequential. Planner windows are positive at maximum IoU≥ 0.3, negative below 0.1, and otherwise ignored. Binary cross-entropy is combined with margin ranking (margin 0.2) against semantically similar non-overlapping negatives; the selected planner is then frozen and Top-K candidates are cached.

The localizer minimizes start/end cross-entropy plus IoU loss. Over available modalities only, normalized confidence ${ \widetilde { \rho } } _ { j }$ gives a weak consistency term

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f i n e } } = \mathcal { L } _ { \mathrm { l o c } } + 0 . 0 5 D _ { \mathrm { K L } } \big ( \widetilde { \rho } _ { j } \| \omega _ { j } \big ) . } \end{array}\tag{10}
$$

This term is a routing prior, not reliability supervision. There is no gradient through hard candidate selection, and no ground-truth timestamp, answer label, or audit judgment enters confidence computation or test-time cards.

## 5 Experiments

## 5.1 Setup

Datasets and metrics. MedVidQA is the target benchmark; Ego4D-NLQ and QVHighlights test cross-domain grounding compatibility (Gupta, Attal, and Demner-Fushman 2023; Grauman et al. 2022; Lei, Berg, and Bansal 2021). We report rank-1 recall at temporal IoU thresholds, mIoU, planner recall, candidate duration, compression, and runtime. Baselines include VSLNet, Moment-DETR, QD/CG-DETR, Uni-VTG, CausalVTG, SRF, CONE, GroundNLQ, Grounded-VideoLLM, and TimeZero (Zhang et al. 2020a; Lei, Berg, and Bansal 2021; Moon et al. 2023b,a; Lin et al. 2023; Wang, Chen, and Shen 2025; Lee and Kim 2026; Hou et al. 2023b,a; Wang et al. 2025a,b). Reproduced feature-based methods use matched preprocessing and budgets; Video-LLMs use matched prompts and evidence budgets. Oficial results are reference-only.

For localization, R1@τ is the fraction of questions whose top-ranked interval overlaps any annotation by at least τ. Planner recall requires at least one selected candidate at IoU≥ 0.3. Candidate duration is the duration of the union of selected candidates divided by video duration, and compression is its reciprocal. The long-duration split contains oficial MedVidQA test videos longer than 900 s and all associated QA pairs.

Human-audit design. We fixed 300 MedVidQA queries before inspecting system outputs. Input-only strata combine video-duration quartile, question-complexity bin, and source availability/quality; model scores and predictions never determine inclusion. Every method is evaluated on the same queries. Method identity and output order are concealed from three raters. The question and reference answer or decision target are shown, but the method, evidence trace, and groundtruth timestamps are hidden. The keep view contains the predicted moment with a 2-s playback margin on each side and visibly marks the original boundaries.

Suficiency and Verifiability are the two primary human outcomes. For Suficiency, 0 means absent or misleading answer support, 1 partial but incomplete support, and 2 evidence suficient to answer the question. For Verifiability, 0 means no directly identifiable answer evidence, 1 relevant evidence with incomplete context or imprecise boundaries, and 2 directly inspectable evidence with appropriate temporal scope. Suficiency therefore targets semantic completeness, whereas Verifiability targets temporal inspectability. A 30-item pilot fixes the rubric before formal scoring. For seed s, query i, criterion $c \in \{ \mathrm { S u f f , V e r } \}$ , and rater r,

$$
I _ { i s } ^ { c @ 2 } = \mathbb { I } \left[ \sum _ { r = 1 } ^ { 3 } \mathbb { I } ( y _ { i s r } ^ { c } = 2 ) \geq 2 \right] ,\tag{11}
$$

$$
c @ 2 = \frac { 1 0 0 } { S } \sum _ { s = 1 } ^ { S } \frac { 1 } { N _ { s } } \sum _ { i } I _ { i s } ^ { c @ 2 } .\tag{12}
$$

Here $S \ = \ 3 ,$ , and $N _ { s }$ is the efective number of audited queries for seed s. Necessity uses the duration-preserving drop view and is retained only as an auxiliary irreplaceability diagnostic because equivalent evidence may remain elsewhere. The supplement specifies the display sequence, repeated-evidence flag, full rubric, adjudication, agreement, seed aggregation, paired uncertainty, multiplicity family, and duration-adjusted construct checks. Main-table audit values are measured seed-mean point estimates; inferential claims require the corresponding raw-label analysis.

Fairness protocol. “Matched” feature-based baselines share temporal preprocessing, comparable source views and budgets, and timestamp post-processing. “Matched-prompt” Video-LLMs receive the same frame-token budget and aligned ASR/OCR/caption view but no evidence card, planner candidates, source router, or audit metadata. Architectures retain their original encoders and fusion modules. Table 1 marks each row as reported/oficial (Of.), independently reproduced (Rep.), matched feature-based (M), matched-prompt (MP), or ours. Oficial rows are referenceonly and are excluded from strict budget-matched superiority claims.

Implementation. Videos are decoded at 1 fps. Frozen SlowFast and CLIP features represent video and text (Feichtenhofer et al. 2019; Radford et al. 2021); Whisper large-v3, PP-OCRv4, and LLaVA-OneVision-7B provide ASR, OCR, and 4-s factual captions (Radford et al. 2023; Du et al. 2020; Li et al. 2025). Windows are 16 s with 8 s stride (32/16 s for videos over 15 minutes). The default is $K = 5$ and

$$
B _ { \mathrm { d u r } } = \operatorname* { m i n } ( 0 . 1 5 | V | , 3 0 0 \mathrm { s } ) .\tag{13}
$$

The planner is a two-layer 512-D MLP; the localizer is a fourlayer 512-D transformer. AdamW trains the planner/localizer for at most 30/50 epochs with learning rates $2 { \times } 1 0 ^ { - 4 } / 1 0 ^ { - 4 }$ and validation early stopping.

SlowFast and CLIP yield 2816-D visual tokens projected to 512 dimensions. Whisper uses greedy word timestamps; OCR retains English detections above 0.50; the frozen captioner describes visible action, manipulated object, and procedural state in at most 32 new tokens. Candidates use at most 128 temporal tokens. Both stages use weight decay 10<sup>−4</sup>, gradient clipping at 1.0, 5% warmup, and seeds {0, 1, 2}.

## 5.2 Main Results

EP-LVMQA leads every MedVidQA metric in Table 1. Relative to the strongest listed non-ours value, its gains are

<table><tr><td rowspan="3">Method</td><td rowspan="3">Prot.</td><td colspan="6">MedVidQA</td><td colspan="2">Ego4D-NLQ</td><td colspan="2">QVHighlights</td></tr><tr><td colspan="3">R1</td><td rowspan="2">mIoU</td><td rowspan="2">Suff.@2</td><td rowspan="2">Ver.@2</td><td colspan="2">R1</td><td rowspan="2"></td><td rowspan="2">R1@0.5 MR-mAP</td></tr><tr><td>@0.3</td><td>@0.5</td><td>@0.7</td><td>@0.3</td><td>@0.5</td></tr><tr><td>MedVidQA/MVAL</td><td>Off./Rep.</td><td>25.8</td><td>14.2</td><td>6.5</td><td>20.1</td><td>10.2</td><td>8.8</td><td>1</td><td>1</td><td>一</td><td>一</td></tr><tr><td>VPTSL/TAGV-style</td><td>Rep.</td><td>32.6</td><td>18.2</td><td>8.4</td><td>25.8</td><td>15.9</td><td>13.8</td><td></td><td></td><td></td><td></td></tr><tr><td>Moment-DETR(Lei, Berg, and Bansal 2021)</td><td>M</td><td>25.7</td><td>14.1</td><td>6.3</td><td>20.0</td><td>10.0</td><td>8.6</td><td>4.57</td><td>1.85</td><td>59.32</td><td>35.71</td></tr><tr><td>CG-DETR (Moon et al. 2023a)</td><td>M</td><td>32.1</td><td>18.4</td><td>9.3</td><td>24.5</td><td>13.6</td><td>11.6</td><td></td><td></td><td>67.91</td><td>47.16</td></tr><tr><td>UniVTG w/ PT (Lin et al. 2023)</td><td>M</td><td>34.3</td><td>19.8</td><td>10.1</td><td>26.0</td><td>14.8</td><td>12.6</td><td>11.74</td><td>7.54</td><td>64.82</td><td>42.95</td></tr><tr><td>CausalVTG (Wang, Chen, and Shen 2025)</td><td>M</td><td>34.0</td><td>19.2</td><td>10.6</td><td>25.7</td><td>14.9</td><td>12.9</td><td>19.12</td><td>13.41</td><td>66.05</td><td>47.50</td></tr><tr><td>SRF (Lee and Kim 2026)</td><td>M</td><td>34.8</td><td>19.6</td><td>10.5</td><td>26.3</td><td>15.4</td><td>13.5</td><td>17.98</td><td>11.53</td><td>67.31</td><td>43.97</td></tr><tr><td>GroundNLQ (Hou et al. 2023a)</td><td>Off.</td><td></td><td>一</td><td>一</td><td>一</td><td></td><td></td><td>25.67</td><td>18.18</td><td></td><td></td></tr><tr><td>Grounded-VideoLLM (Wang et al. 2025a)</td><td>MP</td><td>33.9</td><td>18.7</td><td>7.5</td><td>25.4</td><td>12.9</td><td>10.8</td><td>14.12</td><td>8.35</td><td>56.21</td><td>33.42</td></tr><tr><td>TimeŽero (Wang et al. 2025b)</td><td>MP</td><td>36.4</td><td>19.3</td><td>7.6</td><td>26.7</td><td>13.8</td><td>11.5</td><td>15.92</td><td>9.79</td><td>58.70</td><td>36.21</td></tr><tr><td>EP-LVMQA</td><td>Ours</td><td>41.2</td><td>26.1</td><td>14.3</td><td>31.8</td><td>25.4</td><td>22.6</td><td>21.83</td><td>13.60</td><td>67.42</td><td>45.69</td></tr></table>

Table 1: Main comparison with explicit result provenance. MedVidQA is the target setting; Ego4D-NLQ and QVHighlights are compatibility checks. Prot.: Of.=reported/oficial reference, Rep.=our reproduction, M=matched feature-based evaluation, MP=matched-prompt Video-LLM evaluation. Of./Rep. means that localization is oficial while audit values come from reproduced saved predictions; these values are not combined into a matched superiority claim. All audit columns are measured strict-majority rates on the common blinded 300-query set. Bold and underlining denote the best and second-best listed value per metric, irrespective of protocol class.

13.2%, 31.8%, 34.9%, and 19.1% for R1@0.3, R1@0.5, R1@0.7, and mIoU, respectively. Beyond the target setting, EP-LVMQA shows reasonable cross-benchmark compatibility without benchmark-specific design, while performing only slightly below specialized methods, as shown in Figure 3.

Finding 1: EP-LVMQA exhibits larger gains at stricter IoU thresholds. Against matched UniVTG, the relative gain rises from 20.1% at R1@0.3 to 31.8% at R1@0.5 and 41.6% at R1@0.7. The widening margin at stricter IoU thresholds indicates that EP-LVMQA’s gains arise not only from coarse retrieval but also from more precise boundary localization enabled by the Evidence-Preserving Candidate Interface, allowing it to better isolate answer-bearing actions from adjacent procedural steps.

Finding 2: Human-audit metrics capture practical utility beyond conventional localization scores. EP-LVMQA improves the strongest listed non-ours Suf.@2 and Ver.@2 values by 59.7% and 63.8%, exceeding its relative gains on IoU-based metrics. The strongest localization and audit baselines are also diferent models. Suficiency tests whether the returned content answers the question, whereas Verifiability additionally requires identifiable and well-scoped evidence. A prediction can overlap the annotation yet remain too broad, incomplete, or ambiguous for either criterion. Temporal overlap is therefore related to, but not equivalent to, practical answer support. Nevertheless, the absolute rates remain low—only 25.4% and 22.6%, respectively—and are still insuficient to support reliable use in real-world humanfacing settings.

## 5.3 Robustness and Planner Analysis

EP-LVMQA has the highest score under every perturbation and the lowest mean relative drop, 11.1%, compared with the strongest non-ours value of 14.6% achieved by CausalVTG (Table 2). This is a 3.5-point absolute and 24.0% relative reduction in average degradation.

Finding 3: EP-LVMQA’s advantage widens under semantic distraction and cross-modal conflict. Relative to the strongest non-ours method in each column, EP-LVMQA gains 41.9% under ASR-50 over UniVTG and 35.5% under OCR-50 over SRF. Under incorrect-evidence perturbations, it outperforms CausalVTG by 41.1% under HN-Cap and 42.9% under Conflict. These margins all exceed its 31.8% gain over UniVTG on clean input. Under Conflict, Suf.@2 also increases from 11.9 to 20.8, an 8.9-point gain.

Across all methods, incorrect-evidence perturbations cause a 24.9% relative drop, nearly twice the 12.8% drop under missing-source perturbations. Remaining modalities can partly compensate for a missing source, whereas plausible but incorrect evidence can redirect localization toward an incorrect moment. Robust localization therefore requires rejecting unreliable evidence, not merely retrieving more evidence.

Table 3 separates planner quality from candidate budget. Under the common Top-5 setting, EP-LVMQA achieved 80.6 candidate recall and 23.8 final R1@0.5. Relative to the same planner without confidence inputs, these scores increased by 12.2 and 2.9 points at nearly unchanged selected duration (10.8% versus 10.9%) and cached time (0.54 versus 0.53 s/query). EP-LVMQA also exceeded the MomentSeeker Top-5 baseline by 18.7 candidate-recall points and 5.6 final localization points while selecting less video. The improvement therefore reflects better candidate selection rather than a larger evidence budget.

<table><tr><td>Method</td><td>Clean</td><td>ASR-50</td><td>OCR-50</td><td>HN-Cap</td><td>Conflict</td><td>Avg. drop ↓</td><td>Conflict Suff.@2</td></tr><tr><td>VPTSL/TAGV-style</td><td>18.2</td><td>11.4</td><td>16.6</td><td>13.2</td><td>12.0</td><td>26.9</td><td>9.1</td></tr><tr><td>Moment-DETR (Lei, Berg, and Bansal 2021)</td><td>14.1</td><td>11.5</td><td>13.1</td><td>10.9</td><td>10.2</td><td>19.0</td><td>6.8</td></tr><tr><td>QD-DETR (Moon et al. 2023b)</td><td>16.7</td><td>13.4</td><td>15.4</td><td>12.5</td><td>11.7</td><td>20.7</td><td>7.9</td></tr><tr><td>CG-DETR (Moon et al. 2023a)</td><td>18.4</td><td>15.2</td><td>16.9</td><td>13.7</td><td>13.0</td><td>20.1</td><td>9.0</td></tr><tr><td>UniVTG (Lin et al. 2023)</td><td>19.8</td><td>16.7</td><td>18.2</td><td>15.0</td><td>14.4</td><td>18.8</td><td>10.6</td></tr><tr><td>CausalVTG (Wang, Chen, and Shen 2025)</td><td>19.2</td><td>16.5</td><td>17.9</td><td>15.8</td><td>15.4</td><td>14.6</td><td>11.9</td></tr><tr><td>SRF (Lee and Kim 2026)</td><td>19.6</td><td>16.6</td><td>18.3</td><td>14.7</td><td>13.3</td><td>19.8</td><td>10.2</td></tr><tr><td>EP-LVMQA</td><td>26.1</td><td>23.7</td><td>24.8</td><td>22.3</td><td>22.0</td><td>11.1</td><td>20.8</td></tr></table>

Table 2: MedVidQA robustness in R1@0.5. ASR/OCR-50 removes half the corresponding evidence; HN-Cap uses semantically similar non-overlapping caption negatives; Conflict retains ASR but replaces captions. Avg. drop is the mean relative loss across the four perturbations. All methods receive the same degraded input views.
<table><tr><td>Setting</td><td>Cand. R @0.3↑</td><td>Sel. dur. (%) ↓</td><td>Final R1 @0.5↑</td><td>Cached time (s/query) ↓</td></tr><tr><td>A. Planner comparison under Top-5</td><td></td><td></td><td></td><td></td></tr><tr><td>Uniform sampling</td><td>31.8</td><td>11.6</td><td>11.9</td><td>0.54</td></tr><tr><td>Text-only planner</td><td>49.2</td><td>10.8</td><td>15.4</td><td>0.49</td></tr><tr><td>Visual-only planner</td><td>55.6</td><td>11.0</td><td>16.6</td><td>0.48</td></tr><tr><td>MomentSeeker Top-5 (Yuan et al. 2025)</td><td>61.9</td><td>12.3</td><td>18.2</td><td>0.69</td></tr><tr><td>EP-LVMQA w/o confidence</td><td>68.4</td><td>10.9</td><td>20.9</td><td>0.53</td></tr><tr><td>EP-LVMQA Top-5 (default)</td><td>80.6</td><td>10.8</td><td>23.8</td><td>0.54</td></tr><tr><td>B. Candidate-budget and oracle analysis</td><td></td><td></td><td></td><td></td></tr><tr><td>EP-LVMQA Top-3</td><td>71.1</td><td>6.5</td><td>21.4</td><td>0.40</td></tr><tr><td>EP-LVMQA Top-10</td><td>88.9</td><td>20.2</td><td>24.7</td><td>0.83</td></tr><tr><td>Oracle selection†</td><td>100.0</td><td></td><td>31.2</td><td></td></tr></table>

Table 3: Planner and candidate-budget analysis on MedVidQA videos over 15 min. Block A fixes Top-5; Block B varies K. Cand. R@0.3 is candidate recall; Sel. dur. is candidate-union/video duration. Cached time excludes ofline extraction. <sup>†</sup>Oracle changes selection only; the fixed localizer receives no ground-truth boundaries.

The budget sweep identifies Top-5 as the operating-point knee. Increasing K from 3 to 5 added 9.5 candidate-recall points and 2.4 final R1@0.5 points for 4.3 additional duration points. Increasing K from 5 to 10 added a similar 8.3 recall points but only 0.9 final points, while selected duration rose from 10.8% to 20.2% and cached time from 0.54 to 0.83 s/query. Top-10 nevertheless remained 6.5 points below oracle selection, showing that increased candidate coverage alone cannot eliminate errors in candidate ranking and boundary refinement.

## 5.4 Ablation and Human-Audited Evidence

The card-mediated intervention afects the two primary audit outcomes more strongly than conventional overlap (Table 4). Removing it reduces R1@0.5 by 4.4 points, but reduces Suf.@2 and Ver.@2 by 8.5 and 8.3 points. Across the seven removals, the mean audit decline is roughly twice the corresponding R1@0.5 decline. Because removing the card also removes its dependent fields, this row estimates the whole planner-to-localizer evidence contract rather than the serialized container alone.

The stage-level references further separate evidence capture from temporal precision. Relative to standard moment retrieval, the coarse planner raises Suf.@2 from 14.8 to 17.6 but lowers R1@0.7 from 10.1 to 8.7. A coarse window can therefore contain answer-bearing evidence while remaining a poor temporal explanation. Fine localization is needed to convert semantic coverage into a sharply scoped moment.

Source identity accounts for much of the conflict result. Removing source fields lowers conflict R1@0.5 by 4.6 points and Suf.@2 by 6.6 points. Removing the prior has a larger efect at R1@0.7 than at R1@0.5 (1.6 versus 1.1 points), which is consistent with boundary sharpening. Cues and context produce smaller gains across all criteria. These effects are not additive: removing the timeline also disables downstream source fields and routing, so the −TL row measures the whole source-preservation pathway rather than the marginal value of timeline construction.

QVHighlightsb
<table><tr><td rowspan="2">Configuration</td><td rowspan="2"></td><td rowspan="2">TL Card Src Rtr Cue Ctx Prior</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">R1</td><td rowspan="2">mIoU</td><td rowspan="2">Conflict</td><td rowspan="2">Suff.@2</td><td rowspan="2">Nec.@2</td><td rowspan="2">Ver.@2</td></tr><tr><td>@0.5</td><td>@0.7</td></tr><tr><td>Stage-level references Standard MR reference</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>19.8</td><td>10.1</td><td>26.0 14.4</td><td>14.8</td><td>13.2</td><td>12.6</td></tr><tr><td>Coarse planner only</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>20.5</td><td>8.7</td><td>27.0</td><td>15.8</td><td>17.6</td><td>15.4</td><td>14.7</td></tr><tr><td>Component ablations</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>-TL</td><td>×</td><td>√</td><td>X</td><td>×</td><td>L</td><td>√</td><td>√</td><td>22.3</td><td>11.5</td><td>28.5</td><td>15.9</td><td>18.2</td><td>16.3</td><td>15.6</td></tr><tr><td>-Card</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>21.7</td><td>10.9 12.1</td><td>27.7</td><td>16.5</td><td>16.9</td><td>15.1</td><td>14.3</td></tr><tr><td>—Source fields</td><td>√</td><td>√ √</td><td>X</td><td>X</td><td>√</td><td>√</td><td>√</td><td>23.1 23.7</td><td>12.6</td><td>29.1 29.7</td><td>17.4</td><td>18.8</td><td>16.9</td><td>16.1</td></tr><tr><td>—Confidence/router</td><td>√ √</td><td>√</td><td>√ √</td><td>× √</td><td>√</td><td>√ √</td><td>√ √</td><td>24.2</td><td>13.0</td><td>30.1</td><td>18.1 19.6</td><td>20.0 21.6</td><td>18.0 19.5</td><td>17.3 18.8</td></tr><tr><td>-Cue -Context</td><td>√</td><td>√</td><td>√</td><td>√</td><td>× √</td><td>X</td><td>√</td><td>24.6</td><td>13.3</td><td>30.4</td><td>20.3</td><td>22.3</td><td>20.4</td><td>19.4</td></tr><tr><td>-Prior</td><td>√</td><td>了</td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>25.0</td><td>12.7</td><td>30.7</td><td>21.0</td><td>23.2</td><td>21.1</td><td>20.4</td></tr><tr><td>Full EP-LVMQA</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>26.1</td><td>14.3</td><td>31.8</td><td>22.0</td><td>25.4</td><td>23.2</td><td>22.6</td></tr></table>

Table 4: Stage-level references and component ablations on MedVidQA. Checks/crosses mark enabled/removed components or downstream unavailability. TL is the evidence timeline; Src contains modality evidence, masks, and confidence; Rtr is the source router; Conflict is R1@0.5 under ASR–caption conflict. Suf.@2 and Ver.@2 are primary outcomes; Nec.@2 is auxiliary. Audit values use the common blinded set and strict-majority rule; all variants share the split, encoders, Top-5 budget, backbone, optimizer, and evaluation.

![](images/5a7af8290e768f7f5bfdb3d7c074a7ef4f3048f5fcb20fc9322d9ead6fd4ca5c.jpg)

![](images/25a2335d6f6c49ab015a5f475851dfc74de1a0d1606e4817f120fdc76e1423c1.jpg)  
Figure 3: Cross-benchmark performance on (a) Ego4D-NLQ and (b) QVHighlights, showing the top three methods for each metric. Ours denotes EP-LVMQA.

The ablations also form a consistent functional hierarchy. Removing the card produces the largest joint loss in localization and both primary audit outcomes. Removing source fields or the confidence/router pair most strongly afects conflict robustness, whereas the prior has a disproportionate efect at the stricter IoU threshold. Cue and context removal causes smaller but aligned declines. This ordering links the interface to evidence transfer, source-aware fields to reliability control, and the prior to boundary precision. Because several interventions disable dependent fields, the table supports pathway-level attribution rather than additive compo-

![](images/75bfd90b2a7d153956bd26b4b2989e61c341057f5546f5752b791061cdc533cc.jpg)  
Figure 4: Frame-level comparisons under complete, missing, and conflicting evidence. EP-LVMQA is tighter or more resilient than TimeZero, retaining target overlap with incomplete or inconsistent evidence.

nent efects.

## 6 Conclusion

EP-LVMQA couples an uncertainty-aware multimodal timeline, a coarse planner, and a source-routed fine localizer through a compact evidence card. On MedVidQA, it improves temporal localization, perturbation robustness, and audited evidence quality while remaining competitive on external benchmarks. Figure 4 illustrates tighter or more resilient selections under complete, missing, and conflicting evidence. Nevertheless, absolute audit rates remain too low for dependable human-facing deployment.

Alayrac, J.-B.; Recasens, A.; Schneider, R.; Arandjelović, O.; Ramapuram, J.; De Fauw, J.; Smaira, L.; Dieleman, S.; and Zisserman, A. 2020. Self-Supervised MultiModal Versatile Networks. In Advances in Neural Information Processing Systems, volume 33, 25–37.

Bain, M.; Nagrani, A.; Varol, G.; and Zisserman, A. 2021. Frozen in Time: A Joint Video and Image Encoder for Endto-End Retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 1728–1738.

Barrios, W.; Soldan, M.; Ceballos-Arroyo, A. M.; Heilbron, F. C.; and Ghanem, B. 2023. Localizing Moments in Long Video via Multimodal Guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 13667–13678.

Carbonell, J.; and Goldstein, J. 1998. The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries. In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 335–336.

Di, S.; and Xie, W. 2024. Grounded Question-Answering in Long Egocentric Videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12934–12943.

Du, Y.; Li, C.; Guo, R.; Yin, X.; Liu, W.; Zhou, J.; Bai, Y.; Yu, Z.; Yang, Y.; Dang, Q.; and Wang, H. 2020. PP-OCR: A Practical Ultra Lightweight OCR System. arXiv preprint arXiv:2009.09941.

Feichtenhofer, C.; Fan, H.; Malik, J.; and He, K. 2019. Slow-Fast Networks for Video Recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 6202–6211.

Gao, J.; Sun, C.; Yang, Z.; and Nevatia, R. 2017. TALL: Temporal Activity Localization via Language Query. In Proceedings ofthe IEEE International Conference on Computer Vision, 5267–5275.

Ghosh, S.; Agarwal, A.; Parekh, Z.; and Hauptmann, A. 2019. ExCL: Extractive Clip Localization Using Natural Language Descriptions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1, 1984–1990.

Grauman, K.; Westbury, A.; Byrne, E.; et al. 2022. Ego4D: Around the World in 3,000 Hours of Egocentric Video. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18995–19012.

Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017. On Calibration of Modern Neural Networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, 1321–1330. PMLR.

Gupta, D.; Attal, K.; and Demner-Fushman, D. 2023. A Dataset for Medical Instructional Video Classification and Question Answering. Scientific Data, 10(1): 158.

Hendricks, L. A.; Wang, O.; Shechtman, E.; Sivic, J.; Darrell, T.; and Russell, B. 2017. Localizing Moments in Video with

Natural Language. In Proceedings of the IEEE International Conference on Computer Vision, 5803–5812.

Hou, Z.; Ji, L.; Gao, D.; Zhong, W.; Yan, K.; Li, C.; Chan, W.-K.; Ngo, C.-W.; Duan, N.; and Shou, M. Z. 2023a. GroundNLQ @ Ego4D Natural Language Queries Challenge 2023. arXiv preprint arXiv:2306.15255.

Hou, Z.; Zhong, W.; Ji, L.; Gao, D.; Yan, K.; Chan, W.-K.; Ngo, C.-W.; Shou, M. Z.; and Duan, N. 2023b. CONE: An Eficient COarse-to-fiNE Alignment Framework for Long Video Temporal Grounding. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 8013–8028.

Huang, B.; Wang, X.; Chen, H.; Song, Z.; and Zhu, W. 2024. VTimeLLM: Empower LLM to Grasp Video Moments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14271–14280.

Jang, Y.; Song, Y.; Yu, Y.; Kim, Y.; and Kim, G. 2017. TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering. In Proceedings ofthe IEEE Conference on Com puter Vision and Pattern Recognition, 2758–2766.

Lee, Y.; and Kim, J. U. 2026. See, Rank, and Filter: Important Word-Aware Clip Filtering via Scene Understanding for Moment Retrieval and Highlight Detection. Proceedings of the AAAI Conference on Artificial Intelligence, 40(8): 5936–5944.

Lei, J.; Berg, T. L.; and Bansal, M. 2021. Detecting Moments and Highlights in Videos via Natural Language Queries. In Advances in Neural Information Processing Systems, volume 34, 11846–11858.

Lei, J.; Yu, L.; Bansal, M.; and Berg, T. L. 2018. TVQA: Localized, Compositional Video Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 1369–1379.

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; and Li, C. 2025. LLaVA-OneVision: Easy Visual Task Transfer. Transactions on Machine Learning Research.

Lin, K. Q.; Zhang, P.; Chen, J.; Pramanick, S.; Gao, D.; Wang, A. J.; Yan, R.; and Shou, M. Z. 2023. UniVTG: Towards Unified Video-Language Temporal Grounding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2794–2804.

Liu, Y.; Li, S.; Wu, Y.; Chen, C.-W.; Shan, Y.; and Qie, X.-S. 2022. UMT: Unified Multi-Modal Transformers for Joint Video Moment Retrieval and Highlight Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3042–3051.

Ma, M.; Ren, J.; Zhao, L.; Testuggine, D.; and Peng, X. 2022. Are Multimodal Transformers Robust to Missing Modality? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18177–18186.

Miech, A.; Zhukov, D.; Alayrac, J.-B.; Tapaswi, M.; Laptev, I.; and Sivic, J. 2019. HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2630–2640.

Moon, W.; Hyun, S.; Lee, S.; and Heo, J.-P. 2023a. Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding. arXiv preprint arXiv:2311.08835.

Moon, W.; Hyun, S.; Park, S.; Park, D.; and Heo, J.-P. 2023b. Query-Dependent Video Representation for Moment Retrieval and Highlight Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23023–23033.

Mu, F.; Mo, S.; and Li, Y. 2024. SnAG: Scalable and Accurate Video Grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18930–18940.

Neverova, N.; Wolf, C.; Taylor, G. W.; and Nebout, F. 2016. ModDrop: Adaptive Multi-Modal Gesture Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(8): 1692–1706.

Pan, Y.; He, X.; Gong, B.; Lv, Y.; Shen, Y.; Peng, Y.; and Zhao, D. 2023. Scanning Only Once: An End-to-End Framework for Fast Temporal Grounding in Long Videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 13721–13731.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models from Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, 8748–8763. PMLR.

Radford, A.; Kim, J. W.; Xu, T.; Brockman, G.; McLeavey, C.; and Sutskever, I. 2023. Robust Speech Recognition via Large-Scale Weak Supervision. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 28492–28518. PMLR.

Regneri, M.; Rohrbach, M.; Wetzel, D.; Thater, S.; Schiele, B.; and Pinkal, M. 2013. Grounding Action Descriptions in Videos. Transactions of the Association for Computational Linguistics, 1: 25–36.

Soldan, M.; Pardo, A.; Alcázar, J. L.; Heilbron, F. C.; Zhao, C.; Giancola, S.; and Ghanem, B. 2022. MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5026– 5035.

Wang, H.; Xu, Z.; Cheng, Y.; Diao, S.; Zhou, Y.; Cao, Y.; Wang, Q.; Ge, W.; and Huang, L. 2025a. Grounded-VideoLLM: Sharpening Fine-Grained Temporal Grounding in Video Large Language Models. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, 959–975.

Wang, Q.; Chen, S.; and Shen, Y. 2025. CausalVTG: Towards Robust Video Temporal Grounding via Causal Inference. In Advances in Neural Information Processing Systems, volume 38.

Wang, Y.; Xu, B.; Yue, Z.; Xiao, Z.; Wang, Z.; Zhang, L.; Yang, D.; Wang, W.; and Jin, Q. 2025b. TimeZero: Tempo-

ral Video Grounding with Reasoning-Guided LVLM. arXiv preprint arXiv:2503.13377v1.

Xiao, J.; Yao, A.; Li, Y.; and Chua, T.-S. 2024. Can I Trust Your Answer? Visually Grounded Video Question Answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13204–13214.

Yuan, H.; Ni, J.; Liu, Z.; Wang, Y.; Zhou, J.; Liang, Z.; Zhao, B.; Cao, Z.; Wen, J.-R.; and Dou, Z. 2025. MomentSeeker: A Task-Oriented Benchmark for Long-Video Moment Retrieval. In Advances in Neural Information Processing Systems, volume 38.

Zhang, H.; Sun, A.; Jing, W.; and Zhou, J. T. 2020a. Span-Based Localizing Network for Natural Language Video Localization. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, 6543–6554.

Zhang, S.; Peng, H.; Fu, J.; and Luo, J. 2020b. Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34(7), 12870–12877.