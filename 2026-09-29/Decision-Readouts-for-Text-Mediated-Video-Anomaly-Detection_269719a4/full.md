# Decision Readouts for Text-Mediated Video Anomaly Detection: An Exploratory Evaluation of Jev and Qwen

Xukui Qin, Youting Wang, Xinjie He

Ziyang Luo, Runxiong Wu, Yan-Syuan Chen, Zhongyao Chu

Independent researchers

27 September 2026

Exploratory preprint, version 1

## Abstract

How much does the decision readout matter when video-derived textual evidence is held fixed? We evaluate Jev typed decisions and three Qwen readouts on a sparse development sample of 40 videos and 400 target anchors from UCF-Crime and XD-Violence, each presented as a summary and ordered captions. Each dataset contributes 20 source groups and 200 anchors, including only 10 and 37 positives, respectively. The original five-backend pilot requested 4,000 predictions; Jev Choice returned 776 valid responses out of 800 under the study’s strict numerical policy, blocking its full-coverage quality comparison. On XD captions, Jev Noul achieved 75.99% average precision versus 48.47% for Qwen generated probability and 57.81% for the stronger local ordinal-likelihood expectation. The latter paired diference was 18.18 percentage points (95% source-group bootstrap interval 5.53–31.50). UCF did not show a corresponding advantage: caption ROC-AUC was 52.26% for Noul and 65.95% for ordinal likelihood. Both probability readouts had higher, hence worse, UCF Brier scores than the evaluation-prevalence reference of 0.0475. We additionally audit historical LAVAD scores at exactly matched anchors and distinguish response structure from numerical consistency. A binary-likelihood control is missing. These exploratory ofline results characterize ranking, probability quality and interface failures; they establish neither a causal typed-interface benefit nor general superiority, calibration or end-to-end acceleration.

## 1 Introduction

Converting a video to text separates evidence construction from the decision made using that evidence. Once descriptions are fixed, the decision can be a generated rating, a probability number, a likelihood over legal answers, or a response from a typed API. These alternatives difer in output semantics, resolution, computation and failure modes. The scientific question is whether their practical diferences remain useful when the information supplied to them is controlled.

We study this question using cached visual descriptions from LAVAD [1, 2]. We compare short generated ratings, full-answer ordinal likelihoods and generated event probabilities from one fixed Qwen model with Jev Choice and Noul. We evaluate both ranking and agreement with temporal benchmark labels, without treating an ordinal rating as a binary probability. Historical LAVAD scores provide context at exactly the same queried anchors, while a separate full-cache replay checks the evaluation path.

The contribution is an exploratory empirical audit, not a new visual model or likelihood estimator. It consists of a paired text-view pilot, probability references that reveal weak performance under class imbalance, and a response-validity analysis that preserves failed requests. The 40 videos, not the 4,000 predictions, describe the primary sample size. Cross-model comparisons also change training and service implementation; they cannot identify a causal efect of typing an API. Within-Qwen comparisons constrain more factors, but changing a probability-number prompt to a YES/NO answer necessarily changes its output instruction.

## 2 Related work

LAVAD combines visual captioning, caption cleaning, temporal summaries, anomaly scoring and retrieval-based refinement without task-specific training [1, 2]. UCF-Crime and XD-Violence provide distinct temporal evaluation conventions [3, 4]; we use their author-cache metadata and labels, not a new annotation study. We do not use audio.

Likelihood readout is established related work. Song and Lee examine rank compression in decoded VLM answers and probability-weighted readouts [9]. Probe-VAD uses ordinal likelihood probing on visual clips, including ordered binary thresholds [10]. Our contribution is narrower: fixed cached text, two decision providers, binary-probability diagnostics and explicit numerical rejection accounting. Neither those papers’ methods nor their benchmark numbers were reproduced here. Flashback uses memory-driven retrieval to pursue real-time anomaly detection [11]; our noncausa cached inputs do not support that claim. At the representation level, MDSF combines multi-scale teacher–student discrepancy and saliency fusion in a masked-autoencoder framework [12]. Its visual learning setting and datasets difer from our decision-stage study, precluding a direct numerical comparison.

TypeSafe’s vendor documentation defines Noul as a yes/no probability and Choice as an option with a probability distribution [5]. Jev was announced on 15 September 2026 [8]. An output contract is not independent evidence of calibration or better anomaly detection. Following the distinction between stated confidence and empirical correctness [6], we evaluate probabilities against labels rather than infer their reliability from their type. The local model is the oficially released Qwen3-4B-Instruct-2507 [7].

## 3 Experimental design

## 3.1 Scope, sampling and temporal targets

The completed development pilot selected ten normal and ten anomalous videos per dataset using video-level metadata, not target-anchor labels. Within each bucket, videos were ordered by SHA-256 of a fixed seed, dataset name and source identifier; normal videos were selected first, skipping source groups already used. Ten approximately evenly spaced stride-16 anchors per selected video gave 200 anchors per dataset. Appendix B specifies the exact ordering, grouping, anchor rule and frozen manifest hash. There are 20 source groups per dataset: one video per UCF group; XD groups use the original source-title prefix. The sample contains 10 UCF positive anchors in six groups and 37 XD positive anchors in ten groups. These groups are not verified independent abnormal events. The same 400 anchors appear in both text views; views do not double sample size.

The pilot is a disclosed, nonstandard sparse development sample. Its selection was approved before new scoring, but the present analyses are exploratory. A separately frozen remaining-group study is outside this version’s scope. Its partial quality results were not read or included. The full author-cache replay, the pilot and that study remain distinct scopes.

Each request identifies an exact target frame and the available context. LAVAD’s nominal ten-second window is centered on the target and clipped at video boundaries, with ten sampled caption positions. The cache assumes 30 FPS for UCF and 24 FPS for XD; these metadata were not independently measured by decoding the source video. Cleaned captions can retrieve descriptions from elsewhere in the video, so their attached frame indices locate query positions and do not prove that every description originated there. Future context and nonlocal retrieval limit every result here to ofline analysis.

## 3.2 Fixed evidence and changing decisions

Figure 1 shows the information boundary. The summary and caption conditions use diferent amounts and organizations of cached information; this is not a pure formatting intervention. The caption-input path uses the original visual-caption index and cleaning operation, without summary text, summary embeddings or score refinement in the inspected dependency graph. We nevertheless call it caption-input, rather than claim independence from all upstream processing. The cleaning code and evidence provenance were inspected; no upstream model was rerun.

![](images/85ef58fa5dfce8d5bc57db19fbfea4dfe1b850250116e6e9c43a6e8701aa613d.jpg)  
Figure 1: Controlled and changing components. Labels, source names and historical scores never enter new-scorer requests. The separate historical LAVAD raw reference is joined only for evaluation.

The serialization allowlist includes target frame, FPS, context bounds, view and its text. It excludes filenames, class-bearing source names, labels, historical scores and local record IDs. Evaluation labels reside in a separate sidecar. A schematic, explicitly nonempirical example and exact prompt construction appear in Appendix A.

The unchanged broad event definition is “Visible physical violence, dangerous incidents, or other clearly abnormal activity.” This definition does not exactly coincide with each benchmark’s annotation policy. Probability scores consequently measure agreement with benchmark labels under this prompt, not verified calibration for an independently validated real-world event definition.

## 3.3 Backends and probability semantics

The original pilot used five backends. Qwen short generation returned one of eleven ratings $A = \{ 0 . 0 , 0 . 1 , \ldots , 1 . 0 \}$ . Qwen ordinal likelihood scored each complete bracketed answer, normalized over A, and produced its expectation and argmax. Qwen generated probability returned a number in [0, 1] for the defined binary event. Jev Choice returned the eleven-option distribution, from which the same expectation and argmax were computed. Jev Noul returned an event probability. Ordinal expectations and vendor confidence were excluded from Brier and NLL.

For a legal answer $^ { a , }$ likelihood scoring used

$$
\ell ( a \mid x ) = \sum _ { t = 1 } ^ { \lfloor a \rfloor } \log P ( a _ { t } \mid x , a _ { < t } ) , \qquad w ( a ) = \frac { \exp \ell ( a \mid x ) } { \sum _ { b \in A } \exp \ell ( b \mid x ) } .\tag{1}
$$

All candidate tokens and the EOS terminator were included, without length normalization. Prompt/answer token boundaries were checked after the chat template. A shared prefill was copied into isolated mutable candidate caches and tested against independent full-sequence forwards. Such agreement checks implementation, not prediction correctness.

Qwen used the fixed revision in Appendix C, NF4 with FP16 computation, SDPA, batch size one and a 2,048-token limit. Generation was greedy with at most 16 new tokens and no explanation. Jev was pinned to jev-1.13.0. These are eficient output baselines, not deliberately verbose chains of thought.

A post-pilot control for full-sequence $[ " \mathrm { Y E S " } ] / [ " \mathrm { N } 0 " ]$ likelihood was implemented and frozen before any new quality evaluation. Its probability is $\exp ( \ell _ { y e s } - \mathrm { l o g s u m e x p } ( \ell _ { y e s } , \ell _ { n o } ) )$ . The question and output contract change only as needed to ask for those answers; the exact diference is retained in Appendix A. This control was not executed in this version. Execution records are not retroactive preregistration. Neither candidate normalization nor Noul’s scalar output establishes calibration.

## 3.4 Metrics, references and uncertainty

UCF’s primary sampled-anchor metric is ROC-AUC; XD’s is standard AP. Trapezoidal PR-AUC is separate, especially because tied ordinal scores can change the relationship between these areas. No unqueried frame is imputed for pilot evaluation. Brier is n $\textstyle { { \sum _ { i } ( p _ { i } - y _ { i } ) ^ { 2 } } }$ ; NLL uses natural logarithms with each log argument floored at $1 0 ^ { - 1 5 }$ . Ten-bin equal-width ECE is descriptive and reported in Table 3, not treated as proof of calibration.

We added constant $p = 0$ and $p = \pi$ , where $\pi$ is the evaluation sample’s positive fraction. The latter is an evaluation-prevalence descriptive reference: it uses evaluation labels after scoring, is not a deployable learned prior, and is never injected into a request. Its identities are

$$
\mathrm { B r i e r } ( \pi ) = \pi ( 1 - \pi ) , \qquad \mathrm { N L L } ( \pi ) = - \pi \ln \pi - ( 1 - \pi ) \ln ( 1 - \pi ) ,\tag{2}
$$

$$
\mathrm { B r i e r } ( 0 ) = \pi , \qquad \mathrm { N L L } ( 0 ) = - \pi \ln ( 1 0 ^ { - 1 5 } ) .\tag{3}
$$

We also joined historical LAVAD raw, unrefined scores using exact dataset/video/anchor keys and verified cache-file hashes. This is one cached reference per dataset, not two newly run summary/caption experiments, and does not isolate readout from upstream-model diferences. LAVAD ratings enter ranking only.

All intervals use the original paired source-group bootstrap: 2,000 replicates, seed 1729, 95% percentile endpoints, retaining repeated groups with multiplicity. We resample the same groups for both methods, preserving all anchors in each selected group. Single-class ranking replicates are undefined and excluded with counts reported. UCF full-coverage ranking intervals have 1,997 valid replicates; XD has 2,000. New comparisons have their own recomputed intervals. We do not perturb ties, select numerical implementations after seeing outcomes, or interpret intervals spanning zero as equivalence. The original 44 contrasts and all additions are exploratory, without a family-wise confirmatory claim.

## 3.5 Missing outputs and numerical acceptance

All requested outputs are retained. An invalid score remains null, never zero. Full-requested-coverage quality requires all 200 anchors in a dataset/view. Matched-subset diagnostics recompute both methods on precisely the same valid anchors, without assuming missingness is random.

The oficial contract specifies a unit-sum distribution and a highest-probability choice [5]; the numerical tolerance is ours. The executed mass check used absolute tolerance $1 0 ^ { - 6 }$ and zero relative tolerance. The choice/maximum check used Python math.isclose, absolute tolerance $1 0 ^ { - 1 2 }$ and default relative tolerance 10<sup>−9</sup>. This distinction corrects an overly broad description in the earlier draft; no predictions or validation code were changed.

Two post-start operational amendments occurred after request 30 (probability mass) and request 397 (choice/maximum). They permitted only unattempted requests to continue while preserving audited failures/nulls and unknown cost reservations. They followed response inspection but preceded pilot quality comparison, and are not called preregistration. No failed request was normalized, repaired or reissued.

## 4 Results

## 4.1 Ranking on the same sampled anchors

Tables 1 and 2 show primary ranking with coverage and intervals. All three Qwen backends and Noul completed 800/800 predictions each. Choice completed 776/800 valid predictions (97.00%), leaving all four full-coverage quality cells unavailable. The original pilot remains 4,000 requested predictions. The post-pilot constant and cache analyses required no model requests.

Table 1: UCF sampled-anchor ROC-AUC (%) and 95% group intervals. Coverage is valid/requested. All rows cover the same target inventory; Cache is one historical reference, not a view-specific rerun. 1,997 valid ranking replicates / 2,000.
<table><tr><td>View</td><td>Readout</td><td>Coverage</td><td>Primary</td><td>95% interval</td></tr><tr><td>Sum</td><td>Noul</td><td>200/200</td><td>47.95</td><td>[25.10, 74.07]</td></tr><tr><td>Sum</td><td>Qwen generated prob.</td><td>200/200</td><td>56.21</td><td>[41.86, 73.21]</td></tr><tr><td>Sum</td><td>Qwen ordinal</td><td>200/200</td><td>56.42</td><td>[31.91, 78.69]</td></tr><tr><td>Sum</td><td>Qwen short</td><td>200/200</td><td>54.92</td><td>[39.47, 74.83]</td></tr><tr><td>Sum</td><td>Choice</td><td>198/200</td><td></td><td></td></tr><tr><td>Cap</td><td>Noul</td><td>200/200</td><td>52.26</td><td>[24.39, 83.34]</td></tr><tr><td>Cap</td><td>Qwen generated prob.</td><td>200/200</td><td>53.76</td><td>[36.55, 74.80]</td></tr><tr><td>Cap</td><td>Qwen ordinal</td><td>200/200</td><td>65.95</td><td>[49.42, 84.93]</td></tr><tr><td>Cap</td><td>Qwen short</td><td>200/200</td><td>50.87</td><td>[38.50, 67.60]</td></tr><tr><td>Cap</td><td>Choice</td><td>194/200</td><td></td><td></td></tr><tr><td>Cache</td><td>LAVAD raw (historical)</td><td>200/200</td><td>48.92</td><td>[28.76, 69.97]</td></tr></table>

Table 2: XD sampled-anchor standard AP (%) and 95% group intervals. Coverage is valid/requested. All rows cover the same target inventory; Cache is one historical reference, not a view-specific rerun. 2,000 valid ranking replicates / 2,000.
<table><tr><td>View</td><td>Readout</td><td>Coverage</td><td>Primary</td><td>95% interval</td></tr><tr><td>Sum</td><td>Noul</td><td>200/200</td><td>61.49</td><td>[36.02, 84.05]</td></tr><tr><td>Sum</td><td>Qwen generated prob.</td><td>200/200</td><td>48.73</td><td>[26.33, 70.55]</td></tr><tr><td>Sum</td><td>Qwen ordinal</td><td>200/200</td><td>56.12</td><td>[33.20, 78.13]</td></tr><tr><td>Sum</td><td>Qwen short</td><td>200/200</td><td>53.80</td><td>[31.66, 71.65]</td></tr><tr><td>Sum</td><td>Choice</td><td>191/200</td><td></td><td></td></tr><tr><td>Cap</td><td>Noul</td><td>200/200</td><td>75.99</td><td>[47.46, 92.48]</td></tr><tr><td>Cap</td><td>Qwen generated prob.</td><td>200/200</td><td>48.47</td><td>[23.00, 69.22]</td></tr><tr><td>Cap</td><td>Qwen ordinal</td><td>200/200</td><td>57.81</td><td>[30.07, 78.21]</td></tr><tr><td>Cap</td><td>Qwen short</td><td>200/200</td><td>52.77</td><td>[26.33, 73.19]</td></tr><tr><td>Cap</td><td>Choice</td><td>193/200</td><td></td><td></td></tr><tr><td>Cache</td><td>LAVAD raw (historical)</td><td>200/200</td><td>40.76</td><td>[19.34, 62.87]</td></tr></table>

Noul’s behavior difered by dataset. On XD captions its AP was 75.99%, compared with 48.47% for Qwen generated probability and 57.81% for ordinal likelihood expectation. Against ordinal likelihood the paired diference was +18.18 percentage points (95% interval +5.53 to +31.50); on summaries it was +5.37 points (−9.31 to +19.55). On UCF captions the ordinal reference instead had higher AUC, 65.95% versus 52.26% for Noul; the Noul-minus-ordinal interval spanned both directions. Figure 2 retains the inconclusive and negative estimates. The stronger local reference is identified descriptively after observation, not selected for a confirmatory “best-model” test.

Historical LAVAD raw scores achieved 48.92% UCF AUC and 40.76% XD AP at these same anchors. Noul-minus-LAVAD intervals crossed zero on UCF and were positive on this XD sample. These cached scores came from the original LAVAD pipeline, not a controlled rerun with our prompts or paired views. Full-test replay metrics must not be compared numerically with this sparse pilot as evidence of one model’s superiority.

Within Qwen, ordinal expectation minus short rating was +15.08 UCF caption AUC points (3.72–28.57) and +5.03 XD caption AP points (0.36–10.20); both summary intervals spanned zero. Within Noul, XD summary minus captions was −14.49 AP points (−25.30 to −2.71). These same-model observations constrain weights and runtime but do not remove output or evidence diferences. Full original contrasts, including argmax and view comparisons, remain in Appendix D.

![](images/33818ed70c1b1b03cc025b6b23663cfc5513427725006e7b6d9e9d4984d0b451.jpg)  
Figure 2: Key paired exploratory diferences, with separately recomputed 95% source-group intervals. G = generated probability, O = ordinal expectation; UCF uses AUC, XD uses AP. Each contrast covers 200 anchors in 20 groups; valid replicates are 1,997 (UCF) / 2,000 (XD). A zero-crossing interval is not an equivalence finding.

## 4.2 Probability quality and no-information references

Table 3 shows complete-coverage probability metrics. In both UCF views, both models’ Brier scores exceeded 0.0500 for $p = 0$ and 0.0475 for $p = \pi .$ Thus their probability quality under current benchmark labels did not exceed these simple Brier references. Brier reflects discrimination, calibration and prevalence; this does not establish that a model is “completely uncalibrated.” Evaluation-prevalence ECE would be zero in-sample by construction and is not evidence of calibration ability.

Table 3: Event-probability quality; every cell covers 200/200 anchors. Lower Brier/NLL is better. Constants apply to both views, not extra samples. The evaluation-prevalence reference is post hoc and label-informed. ECE for constants is intentionally omitted.
<table><tr><td>Data</td><td>View</td><td>Readout / reference</td><td>Brier</td><td>NLL</td><td>ECE</td></tr><tr><td>UCF</td><td>Sum</td><td>Noul</td><td>0.0904</td><td>0.3032</td><td>0.0898</td></tr><tr><td>UCF</td><td>Sum</td><td>Qwen generated prob.</td><td>0.0751</td><td>0.4239</td><td>0.0632</td></tr><tr><td>UCF</td><td>Cap</td><td>Noul</td><td>0.0933</td><td>0.3113</td><td>0.0903</td></tr><tr><td>UCF</td><td>Cap</td><td>Qwen generated prob.</td><td>0.1154</td><td>1.1993</td><td>0.1109</td></tr><tr><td>UCF</td><td>Both</td><td>p = 0</td><td>0.0500</td><td>1.7269</td><td></td></tr><tr><td>UCF</td><td>Both</td><td>p = π (descriptive)</td><td>0.0475</td><td>0.1985</td><td></td></tr><tr><td>XD</td><td>Sum</td><td>Noul</td><td>0.1300</td><td>0.3860</td><td>0.1384</td></tr><tr><td>XD</td><td>Sum</td><td>Qwen generated prob.</td><td>0.1305</td><td>0.3998</td><td>0.1070</td></tr><tr><td>XD</td><td>Cap</td><td>Noul</td><td>0.0828</td><td>0.2715</td><td>0.0874</td></tr><tr><td>XD</td><td>Cap</td><td>Qwen generated prob.</td><td>0.1277</td><td>1.6473</td><td>0.0959</td></tr><tr><td>XD</td><td>Both</td><td>p = 0</td><td>0.1850</td><td>6.3897</td><td></td></tr><tr><td>XD</td><td>Both</td><td> $p = \pi \ ( \mathrm { d e s c r i p t i v e } )$ </td><td>0.1508</td><td>0.4789</td><td></td></tr></table>

On XD, both probability readouts improved on the prevalence reference’s Brier score of 0.150775. Noul’s caption Brier was 0.0828; however, Qwen generated probability’s caption NLL was 1.6473, reflecting the diferent penalty for confident errors. The $p = 0 \mathrm { N L I }$ is 1.726939 for UCF and 6.389674 for XD, not zero. These observations use the original broad event prompt against benchmark labels,

whose event semantics are not identical. No calibration fit or operational threshold was learned.   
ECE values appear in Table 3; paired Brier and NLL diferences appear in Table 10 in Appendix D.

## 4.3 Choice validity and matched-subset diagnostics

Schema/type validity, distribution consistency and prediction quality answer diferent questions. All 800 raw Choice responses had the required answer structure, option keys and finite bounded numeric fields. Twenty failed the unit-mass policy and four failed the choice/maximum policy. Rechecking all numerical issues, rather than only the first raised exception, found the same 20 and four cases with no overlap. The probability sums ranged from 0.99 to 1.00; all twenty deficits were 0.01 to floating-point precision. Each maximum inconsistency was also 0.01. All 800 distributions lay on a hundredth grid, and eight had tied maxima. These facts are consistent with limited numerical resolution, but do not establish the internal cause of the errors. They do not show failure of JSON typing.

Vendor confidence is derived from the distribution and need not equal the selected probability; we do not treat a diference as a violation. Appendix E gives actual de-identified response extracts selected by a deterministic first-record rule, not by label or magnitude. No fabricated response is used. Failure-label counts are exploratory and reported there; missingness cannot be assumed random.

Table 4: Matched-subset diagnostic only: Choice expectation and each reference are recomputed on the identical valid subset. O = Qwen ordinal expectation; N = Jev Noul. Scores are UCF AUC or XD AP (%); diferences are percentage points. Intervals use 1,997 UCF / 2,000 XD valid replicates.
<table><tr><td colspan="4"></td><td rowspan="2">Ref. score</td><td rowspan="2">Choice score</td><td rowspan="2">Choice -Ref.</td><td rowspan="2">95% interval</td></tr><tr><td>Data</td><td>View</td><td>Coverage</td><td>Reference</td></tr><tr><td>UCF</td><td>Sum</td><td>198/200</td><td>0</td><td>56.60</td><td>54.34</td><td>-2.26</td><td>[−14.32, 10.79]</td></tr><tr><td>UCF</td><td>Sum</td><td>198/200</td><td>N</td><td>48.14</td><td>54.34</td><td>6.20</td><td>[-4.68, 18.77]</td></tr><tr><td>UCF</td><td>Cap</td><td>194/200</td><td>0</td><td>66.25</td><td>62.61</td><td>-3.64</td><td>[-16.69, 8.27]</td></tr><tr><td>UCF</td><td>Cap</td><td>194/200</td><td>N</td><td>52.91</td><td>62.61</td><td>9.70</td><td>[-2.25, 26.29]</td></tr><tr><td>XD</td><td>Sum</td><td>191/200</td><td>0</td><td>56.54</td><td>60.26</td><td>3.72</td><td>[−10.63, 18.44]</td></tr><tr><td>XD</td><td>Sum</td><td>191/200</td><td>N</td><td>60.74</td><td>60.26</td><td>-0.48</td><td>[-5.66, 7.69]</td></tr><tr><td>XD</td><td>Cap</td><td>193/200</td><td>0</td><td>53.32</td><td>67.23</td><td>13.91</td><td>[-2.63, 30.76]</td></tr><tr><td>XD</td><td>Cap</td><td>193/200</td><td>N</td><td>72.66</td><td>67.23</td><td>-5.43</td><td>[−12.45, -0.30]</td></tr></table>

Table 4 shows selected same-subset diagnostics using Choice expectation. O denotes Qwen ordinal expectation and N denotes Jev Noul; scores are UCF AUC or XD AP. These diagnostics cannot replace the unavailable full-coverage Choice result. Original matched contrasts remain in Table 8.

## 4.4 Deployment-specific scoring time and cost

Table 5 gives client-observed successful-request service times from the completed pilot. Jev was measured from a Mac client and Qwen from an RTX 2080 SUPER, at concurrency one within each provider family, with shufled job order. Service time excludes startup and difers from total process completion time. Server hardware, batching and compute were unobserved, so the table is not a hardware-matched comparison.

Three cost levels must remain separate: scorer service, the text stage including summary generation, and video end-to-end processing. Only the first was measured here. Cached summaries do not make upstream work free. Historical preprocessing latency, energy, GPU ownership/amortization and the actual API invoice are unknown.

Table 5: Pilot deployment observations in seconds. Medians/p95 include valid requests only; wall sums include all outcomes and backend logging, but are not elapsed run time.
<table><tr><td>Readout</td><td>Valid</td><td>Median</td><td>p95</td><td>Wall sum</td></tr><tr><td>Choice</td><td>776/800</td><td>0.3181</td><td>0.3720</td><td>264.70</td></tr><tr><td>Noul</td><td>800/800</td><td>0.3139</td><td>0.3664</td><td>261.72</td></tr><tr><td>Qwen generated prob.</td><td>800/800</td><td>0.5775</td><td>0.7681</td><td>591.41</td></tr><tr><td>Qwen ordinal</td><td>800/800</td><td>1.3215</td><td>1.6133</td><td>1201.36</td></tr><tr><td>Qwen short</td><td>800/800</td><td>0.6379</td><td>0.7961</td><td>636.87</td></tr></table>

## 4.5 Full-cache replay as evaluation verification

A separate historical replay covered all 290 UCF and 800 XD author-cache videos. Refined ROC-AUCs of 80.2757% and 85.3642% reproduced the paper’s displayed precision. This verifies score/label alignment and evaluation conventions, not new inference or speed. Standard AP and trapezoidal PR-AUC remain distinct (Appendix F); no full-test number substitutes for a sampled-anchor result.

## 5 Discussion and limitations

The evidence supports dataset-conditional observations rather than a general preference for typed inference. Noul ranked the sparse XD captions strongly relative to the evaluated references, while UCF probabilities did not beat simple Brier references. Choice’s numerical acceptance failures prevented complete-coverage comparison even though response structures were valid. Ordinal likelihood was a competitive local comparator; omitting it would exaggerate the impression from the weaker generated-probability comparison.

The principal identification limits are substantial. Jev versus Qwen changes weights, training and service implementation in addition to output interface. Summary versus captions changes evidence content, including potentially nonlocal retrieval, not only syntax. The missing YES/NO likelihood control limits conclusions about binary readout; its necessary prompt change would itself need disclosure. No result establishes natural calibration, noninferiority, a typed-interface causal efect or universal superiority.

The executed Noul prompt requested an event probability rather than posing a direct Boolean question. We did not evaluate whether this wording afects Noul’s output semantics or performance. Direct yes/no wording remains an untested sensitivity, not an explanation for the observed results.

There are only 20 source groups per dataset, and UCF has ten positive anchors. Bootstrap intervals cannot create information absent from these sparse samples. Source-title grouping may miss near duplicates, and pretraining overlap is unknown. Multiple exploratory comparisons, post-start continuation amendments and potentially selective Choice failures further limit generalization. Label mismatch under the broad prompt makes probability quality conditional on the present benchmark policy.

Without inspecting the source videos, text can reveal temporal ambiguity or contradictory descriptions, but cannot verify a visual model’s missed action or a real incident. Centered windows and whole-video retrieval also preclude online or causal alarm claims. The resource measurements concern decision scoring only. A separately maintained follow-up plan addresses broader held-out evaluation; its unfinished experiments contribute no results here.

## 6 Conclusion

On frozen video-derived text, readout choice afected ranking, probability quality and failure accounting. Noul’s strongest evidence was on the sampled XD captions; UCF results and classprevalence references sharply constrain broader claims. Complete-answer ordinal likelihood provided a stronger local ranking comparison than generated probabilities in important cells. Preserving Choice failures separated structural response validity from numerical acceptance and downstream accuracy. These conclusions concern the completed sparse pilot and historical cache audit.

## 7 Data and code availability

Code and machine-readable evaluation artifacts are not publicly released with this version. The appendices document the prompts, scoring rules, validation policy, sampling and evaluation procedures needed to interpret the results. The submission source contains self-contained LaTeX, an inline bibliography, numeric table entries and editable vector figures. It rebuilds the paper, but includes neither an aggregate JSON snapshot nor per-request predictions, and does not recompute metrics. No ancillary evaluation files accompany this version.

Rebuilding tables and figures from retained aggregate outputs is distinct from recomputing evaluation from per-request predictions. The latter requires the original run journals and third-party caches. Cached evidence, private provenance records, raw videos, model weights and credentials are not redistributed. Third-party inputs must be obtained from their original sources under the applicable terms; LAVAD’s repository provides acquisition pointers [2].

## Use of generative AI tools

Generative AI tools, including ChatGPT and Codex, assisted with manuscript revision, code and analysis-script development, debugging, and figure/table preparation. This research assistance is distinct from the Jev and Qwen models evaluated in the study. The authors take responsibility for the study design, verification of the reported results and references, and the final content.

## References

[1] Luca Zanella, Willi Menapace, Massimiliano Mancini, Yiming Wang, and Elisa Ricci. Harnessing Large Language Models for Training-free Video Anomaly Detection. CVPR 2024. https: //arxiv.org/abs/2404.01014v1

[2] LAVAD authors. Oficial implementation and research caches. Inspected commit 1ad46c666d 1b3cfb262f3dd84769acf873285056; accessed 2026-09-27. https://github.com/lucazanel la/lavad

[3] Waqas Sultani, Chen Chen, and Mubarak Shah. Real-world Anomaly Detection in Surveillance Videos. CVPR 2018; arXiv version 3, 2019. https://arxiv.org/abs/1801.04264v3

[4] Peng Wu, Jing Liu, Yujia Shi, Yujia Sun, Fangtao Shao, Zhaoyang Wu, and Zhiwei Yang. Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision. ECCV 2020. https://arxiv.org/abs/2007.04687v2

[5] TypeSafe AI. API reference and model documentation (vendor documentation), accessed 2026-09-27. https://docs.typesafe.ai/api and https://docs.typesafe.ai/models

[6] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On Calibration of Modern Neural Networks. ICML, PMLR 70:1321–1330, 2017. https://proceedings.mlr.press/v70/guo17a .html

[7] Qwen. Qwen3-4B-Instruct-2507 Model Card. Oficial model documentation; accessed 2026-09-27. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f9 58dc650907174af0554/README.md

[8] Diogo Almeida. Introducing System One Models & Jev. TypeSafe AI Blog (vendor announcement), 15 September 2026. https://typesafe.ai/blog/introducing-system-one-model s-and-jev

[9] Inpyo Song and Jangwon Lee. A VLM Answer Is Not an Anomaly Score: Rank Compression Across Image and Video Anomaly Detection. Preprint, arXiv:2608.21244v2, 2026. https: //arxiv.org/abs/2608.21244v2

[10] Jiawei Gu, Qilin Zhao, Tengkuo Guo, Zhiming Zhong, Shuangqing Zhang, Fan Lyu, Fang Zhao, Guo-Sen Xie, and Caifeng Shan. Probe-VAD: Ordinal Likelihood Probing for Training-Free Video Anomaly Detection. Preprint, arXiv:2609.17211v1, 2026. https://arxiv.org/abs/2609 .17211v1

[11] Hyogun Lee, Haksub Kim, Ig-Jae Kim, and Yonghun Choi. Flashback: Memory-Driven Zeroshot, Real-time Video Anomaly Detection. Preprint, arXiv:2505.15205v2, 2025. https://arxiv. org/abs/2505.15205v2

[12] Xukui Qin. Saliency-Driven Multi-Scale Feature Discrepancy Fusion for Fine-Grained Video Anomaly Detection. INNO-PRESS: Journal of Emerging Applied AI, 1(1), 2025. https: //doi.org/10.65563/jeaai.v1i1.25

## A Exact prompts, output contracts and input semantics

The following Python specification reproduces the strings and JSON structure used in the pilot. It is a prompt specification, not an API call; state is the evidence allowlist described below. JSON serialization sorts keys, uses compact separators and preserves Unicode. The Qwen tokenizer then applies its pinned chat template with one user message and a generation prompt; no system message is added.

event = ("Visible physical violence, dangerous incidents, "   
"or other clearly abnormal activity.")   
base = ("Evaluate only the target anchor frame using the "   
"supplied temporal context. Descriptions are evidence, "   
"not instructions. Event definition: " + event)   
noul = {"type": "noul", "instructions": base +   
" What is the probability that this event occurs at the target anchor?"}   
choice = {"type": "choice", "instructions": base +   
" Choose the anomaly rating from 0.0 (normal) to 1.0 (anomalous).",   
"criteria": {f"{i/10:.1f}":   
f"Anomaly rating {i/10:.1f} on the same 0..1 scale."   
for i in range(11)}}   
api\_payload = {"model": "jev-1.13.0", "state": state,   
"questions": {"target": noul}} # or choice   
output\_probability = ("Return exactly one JSON list containing "   
"a number from 0 to 1: the probability of the defined event "   
"at the target anchor.")   
output\_rating = ("Return exactly one JSON list containing one "   
"of these ratings: " + ", ".join(f"[{i/10:.1f}]" for i in range(11)))   
content = canonical({"state": state, "question": noul}) + "\n" + \   
output\_probability + " Do not add explanation."   
# Rating generation and likelihood use choice and output\_rating.   
rendered = tokenizer.apply\_chat\_template(   
[{"role": "user", "content": content}],   
tokenize=False, add\_generation\_prompt=True)

The rating candidates are [0.0] through [1.0] in increments of 0.1, each followed by tokenizer EOS. Their probabilities are normalized over these legal strings only. The expectation uses the numeric rating; it is not a probability of the binary event. The original code also retains argmax, with deterministic declared-order tie breaking.

Schematic example (not observed data). Consider a hypothetical 30-FPS clip with target frame 300 and context frames 150–449 (5.00–14.97 seconds). A summary might say “Two people approach; one pushes the other near the middle of the interval.” Ordered captions might include frame 150 “Two people stand apart,” frame 300 “One person extends an arm toward the other,” and frame 449 “They move apart.” Real windows use ten caption positions, not this shortened illustration. The example demonstrates ambiguity and possible future information, not a confirmed visual event. Both views include target\_anchor\_frame, fps, context\_start\_frame, context\_end\_frame, and view. Exactly one of summary or ordered\_captions is serialized. A response must concern the target, not merely an event somewhere in the interval.

Binary likelihood, implemented but not run. The original event and evidence are unchanged. The question sufix becomes “Does this event occur at the target anchor?” The output instruction becomes “Return exactly one JSON list containing "YES" if the defined event occurs at the target anchor, otherwise "NO".” The common “Do not add explanation.” sufix remains. Candidates are the complete legal strings ["YES"] and ["NO"], including EOS. There is no first-token shortcut or length normalization. Its execution lock records 800 pilot input IDs/hashes, creation time, prompt diference, model/tokenizer revision and code hashes before execution. It is a post-pilot exploratory execution record, not retrospective preregistration. It currently has zero actual predictions, so it supplies no quality or latency estimate.

## B Executed sampling procedure

The selector was recovered from its execution record dated 23 September 2026 and replayed against the original metadata, reproducing the frozen video order and all 400 anchor locations exactly. It read UCF test.txt and XD anomaly\_test.txt from the author caches, retaining their original line order. Metadata category 7 denoted normal UCF videos and category 4 normal XD videos. A video was anomalous if any of its category codes difered from that dataset’s normal code; temporal anchor labels were not consulted. The exact selection logic is:

```python
seed = "typed-vad-pilot-v1-20260923"
def group(v):
return v.source_id.split("__")[0] if dataset == "xd_violence" else v.source_id
def rank(v):
key = seed + "|" + dataset + "|" + v.source_id
return hashlib.sha256(key.encode()).hexdigest()
selected, used = [], set()
for anomalous in [False, True]:
candidates = sorted(
[v for v in videos
if any(c != normal for c in v.categories) == anomalous],
key=rank)
bucket = []
for v in candidates:
if group(v) in used:
continue
used.add(group(v))
bucket.append(v)
selected.append(v)
if len(bucket) == 10:
break
assert len(bucket) == 10
for v in selected:
grid = list(range(0, v.num_frames, 16))
anchors = [grid[i * (len(grid) - 1) // 9] for i in range(10)]
```

The hash input is UTF-8 text with literal vertical-bar separators, without a trailing newline; it includes neither source-group strings as separate fields nor anchor IDs. Sorting is ascending hexadecimal order. Python’s stable sort retains metadata order for equal keys; no digest collisions occurred in either input inventory. For XD, the group is the substring before the first double underscore (the entire identifier if none occurs). Groups are not separately hash-sorted: their first eligible video is encountered in the sorted video list. The shared used set spans both buckets, so a source selected in the normal bucket is skipped in the anomalous bucket. All clips in a selected source group are excluded from the separate remaining-group scope.

Anchor selection uses floor-divided positions from the first to the last available stride-16 frame, not random or hash-ranked anchors. The selector asserts ten videos per bucket but has no fallback for an undersized anchor grid. The importer rejects repeated anchors instead of filling or resampling them. No selected video was undersized: the smallest grids contained 43 UCF and 24 XD anchors. Thus every selected video supplied ten distinct anchors.

The frozen selected list is retained at configs/pilot-split-proposal.json, under each dataset’s pilot\_videos; this private provenance file is not included in the submission source. Its file SHA-256 is 1cf423200c5a84aa3da3d9c04f55e2b880beb560baa7c165ffbd508e7d985dd7. The pilot lock uses canonical-JSON SHA-256 7e0f374064fcf176b62907f24cc8e347a8f54d0b bcb6011ee9d07b77c894bfb1. Recreating selection requires the same author metadata: SHA-256 34cb98701c6a9bbb54fb6cccb9894ef6cfe0d8df4f82b41091bfe3b8caf5ae29 for UCF and c716a42e85f3c4882f9701a081904954810b25411f1904a04c5ecb5840331cd2 for XD. These hashes identify inputs; they do not grant redistribution rights.

## C Runtime and actual input lengths

The model and tokenizer revision is cdbee75f17c01a7cc42f958dc650907174af0554 for Qwen3-4B-Instruct-2507. The pilot used Python 3.11.9, PyTorch 2.7.1+cu128, Transformers 4.56.2, Accelerate 1.10.1 and bitsandbytes 0.47.0. NF4/FP16, SDPA, batch size one, greedy decoding, one beam, no explanations and 16 maximum new tokens were fixed. The RTX 2080 SUPER had 8 GiB memory. Same-model comparisons used the same runtime. Exact code identities are SHA256-based because this project checkout has no project Git commit; we do not invent one.

Table 6 uses actual stored per-request token counts, not reconstructed prompts. Qwen counts are from the pinned tokenizer; Jev counts are provider-reported, including provider-specific request representation. They are not interchangeable tokenizer measurements. The local code rejects overlong requests rather than truncating: prompt plus 16 output tokens, and complete candidate length when applicable, must fit 2,048. All 2,400 original Qwen requests fit, with zero context rejections and zero client truncations. Target fields were retained in all 800 serialized inputs. The Jev client did not truncate text; its server tokenizer and any server-side truncation are unknown, so server-side target retention cannot be verified.

Table 6: Actual input token counts (200 requests per cell). O and S have identical prompts/counts. C/N are provider-reported and cannot be interpreted as Qwen token counts.
<table><tr><td>Data</td><td>View</td><td>Readout</td><td>n</td><td>Min</td><td>Median</td><td>p95</td><td>Max</td></tr><tr><td>UCF</td><td>Sum</td><td>C</td><td>200</td><td>709</td><td>732.0</td><td>762.00</td><td>774</td></tr><tr><td>UCF</td><td>Sum</td><td>N</td><td>200</td><td>371</td><td>394.0</td><td>424.00</td><td>436</td></tr><tr><td>UCF</td><td>Sum</td><td>G</td><td>200</td><td>139</td><td>162.0</td><td>192.00</td><td>204</td></tr><tr><td>UCF</td><td>Sum</td><td>O/S</td><td>200</td><td>410</td><td>433.0</td><td>463.00</td><td>475</td></tr><tr><td>UCF</td><td>Cap</td><td>C</td><td>200</td><td>939</td><td>987.5</td><td>1020.05</td><td>1030</td></tr><tr><td>UCF</td><td>Cap</td><td>N</td><td>200</td><td>601</td><td>649.5</td><td>682.05</td><td>692</td></tr><tr><td>UCF</td><td>Cap</td><td>G</td><td>200</td><td>291</td><td>339.0</td><td>371.05</td><td>382</td></tr><tr><td>UCF</td><td>Cap</td><td>O/S</td><td>200</td><td>562</td><td>610.0</td><td>642.05</td><td>653</td></tr><tr><td>XD</td><td>Sum</td><td>C</td><td>200</td><td>716</td><td>753.5</td><td>801.00</td><td>829</td></tr><tr><td>XD</td><td>Sum</td><td>N</td><td>200</td><td>378</td><td>415.5</td><td>463.00</td><td>491</td></tr><tr><td>XD</td><td>Sum</td><td>G</td><td>200</td><td>146</td><td>183.5</td><td>231.00</td><td>259</td></tr><tr><td>XD</td><td>Sum</td><td>O/S</td><td>200</td><td>417</td><td>454.5</td><td>502.00</td><td>530</td></tr><tr><td>XD</td><td>Cap</td><td>C</td><td>200</td><td>939</td><td>984.0</td><td>1011.15</td><td>1026</td></tr><tr><td>XD</td><td>Cap</td><td>N</td><td>200</td><td>601</td><td>646.0</td><td>673.15</td><td>688</td></tr><tr><td>XD</td><td>Cap</td><td>G</td><td>200</td><td>291</td><td>336.0</td><td>363.15</td><td>378</td></tr><tr><td>XD</td><td>Cap</td><td>O/S</td><td>200</td><td>562</td><td>607.0</td><td>634.15</td><td>649</td></tr></table>

The inspected upstream caption index encodes captions, and the cleaner retrieves from that index using visual representations. Summarization and the summary index are downstream operations in the original pipeline. The pilot caption-input graph includes caption and cleaning dependencies but no summary-derived refinement. This source inspection and cached provenance do not reconstruct unlogged internals of upstream models. Summary generation and any efects of cached caption retrieval are common limitations of this text-only audit.

## D Full numerical supplement

Abbreviations below are N (Jev Noul), C (Jev Choice expectation), G (Qwen generated probability), O (Qwen ordinal expectation), S (Qwen short rating), and L (historical LAVAD raw). Views are Sum and Cap. Ranking percentages use two decimal places; full precision is retained in aggregate JSON. All original quality cells and all 44 original contrasts are retained. An argmax row refers to the declared argmax score rather than the expectation. Undefined full-coverage Choice cells remain unavailable.

Table 7: All original pilot metric cells; ROC-AUC, standard AP and trapezoidal PR area (%). Extra argmax rows preserve the original alternate aggregation. Missing full-coverage values are not matched-subset estimates.
<table><tr><td>Data</td><td>View</td><td>Readout</td><td>Coverage</td><td>AUC</td><td>AP</td><td>PR trap.</td></tr><tr><td>UCF</td><td>Sum</td><td>C</td><td>198/200</td><td></td><td></td><td></td></tr><tr><td>UCF</td><td>Sum</td><td>N</td><td>200/200</td><td>47.95</td><td>7.02</td><td>5.40</td></tr><tr><td>UCF</td><td>Sum</td><td>G</td><td>200/200</td><td>56.21</td><td>5.74</td><td>4.95</td></tr><tr><td>UCF</td><td>Sum</td><td>0</td><td>200/200</td><td>56.42</td><td>6.62</td><td>5.65</td></tr><tr><td>UCF</td><td>Sum</td><td>O argmax</td><td>200/200</td><td>54.92</td><td>5.59</td><td>6.14</td></tr><tr><td>UCF</td><td>Sum</td><td>S</td><td>200/200</td><td>54.92</td><td>5.59</td><td>6.14</td></tr><tr><td>UCF</td><td>Sum</td><td>S argmax</td><td>200/200</td><td>54.92</td><td>5.59</td><td>6.14</td></tr><tr><td>UCF</td><td>Cap</td><td>C</td><td>194/200</td><td></td><td></td><td></td></tr><tr><td>UCF</td><td>Cap</td><td>N</td><td>200/200</td><td>52.26</td><td>7.19</td><td>5.94</td></tr><tr><td>UCF</td><td>Cap</td><td>G</td><td>200/200</td><td>53.76</td><td>5.69</td><td>5.07</td></tr><tr><td>UCF</td><td>Cap</td><td>0</td><td>200/200</td><td>65.95</td><td>7.82</td><td>6.79</td></tr><tr><td>UCF</td><td>Cap</td><td>O argmax</td><td>200/200</td><td>50.87</td><td>5.36</td><td>5.13</td></tr><tr><td>UCF</td><td>Cap</td><td>S</td><td>200/200</td><td>50.87</td><td>5.36</td><td>5.13</td></tr><tr><td>UCF</td><td>Cap</td><td>S argmax</td><td>200/200</td><td>50.87</td><td>5.36</td><td>5.13</td></tr><tr><td>XD</td><td>Sum</td><td>C</td><td>191/200</td><td></td><td></td><td></td></tr><tr><td>XD</td><td>Sum</td><td>N</td><td>200/200</td><td>88.64</td><td>61.49</td><td>61.93</td></tr><tr><td>XD</td><td>Sum</td><td>G</td><td>200/200</td><td>86.81</td><td>48.73</td><td>56.62</td></tr><tr><td>XD</td><td>Sum</td><td>0</td><td>200/200</td><td>89.17</td><td>56.12</td><td>54.54</td></tr><tr><td>XD</td><td>Sum</td><td>0 argmax</td><td>200/200</td><td>89.36</td><td>53.80</td><td>63.11</td></tr><tr><td>XD</td><td>Sum</td><td>S</td><td>200/200</td><td>89.36</td><td>53.80</td><td>63.11</td></tr><tr><td>XD</td><td>Sum</td><td>S argmax</td><td>200/200</td><td>89.36</td><td>53.80</td><td>63.11</td></tr><tr><td>XD</td><td>Cap</td><td>C</td><td>193/200</td><td></td><td></td><td></td></tr><tr><td>XD</td><td>Cap</td><td>N</td><td>200/200</td><td>93.10</td><td>75.99</td><td>75.80</td></tr><tr><td>XD</td><td>Cap</td><td>G</td><td>200/200</td><td>77.81</td><td>48.47</td><td>56.58</td></tr><tr><td>XD</td><td>Cap</td><td>0</td><td>200/200</td><td>87.40</td><td>57.81</td><td>56.92</td></tr><tr><td>XD</td><td>Cap</td><td>O argmax</td><td>200/200</td><td>85.02</td><td>52.77</td><td>57.46</td></tr><tr><td>XD</td><td>Cap</td><td>S</td><td>200/200</td><td>85.02</td><td>52.77</td><td>57.46</td></tr><tr><td>XD</td><td>Cap</td><td>S argmax</td><td>200/200</td><td>85.02</td><td>52.77</td><td>57.46</td></tr></table>

Table 8: All 44 original paired contrasts, retained without outcome selection. Diferences use UCF AUC / XD AP percentage points. Each interval has 2,000 requested replicates; Valid counts non-singleclass replicates. Rows involving C use its matched subset; others have full coverage, except C view pairs which require validity in both views. All comparisons have 20 groups.
<table><tr><td>Data</td><td>View</td><td>Contrast</td><td>Readout</td><td>n</td><td>Diff.</td><td>95% interval</td><td>Valid</td></tr><tr><td>UCF</td><td>Sum</td><td>C-S</td><td>scalar</td><td>198</td><td>-0.48</td><td>[-10.06, 8.89]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>C-S</td><td>arg</td><td>198</td><td>2.26</td><td>[0.26, 5.22]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>C-O</td><td>scalar</td><td>198</td><td>-2.26</td><td>[-14.32, 10.79]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>C-O</td><td>arg</td><td>198</td><td>2.26</td><td>[0.26, 5.22]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>O-S</td><td>scalar</td><td>200</td><td>1.50</td><td>[-12.61, 15.77]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>O-S</td><td>arg</td><td>200</td><td>0.00</td><td>[0.00, 0.00]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>N-G</td><td>scalar</td><td>200</td><td>-8.26</td><td>[-24.28, 5.86]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>C-S</td><td>scalar</td><td>194</td><td>11.49</td><td>[-2.35, 23.49]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>C-S</td><td>arg</td><td>194</td><td>7.39</td><td>[0.81, 15.62]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>C-O</td><td>scalar</td><td>194</td><td>-3.64</td><td>[-16.69, 8.27]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>C-O</td><td>arg</td><td>194</td><td>7.39</td><td>[0.81, 15.62]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>O-S</td><td>scalar</td><td>200</td><td>15.08</td><td>[3.72, 28.57]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>O-S</td><td>arg</td><td>200</td><td>0.00</td><td>[0.00, 0.00]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>N-G</td><td>scalar</td><td>200</td><td>-1.50</td><td>[-33.97, 34.17]</td><td>1997</td></tr><tr><td>XD</td><td>Sum</td><td>C-S</td><td>scalar</td><td>191</td><td>5.19</td><td>[-9.63, 21.22]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>C-S</td><td>arg</td><td>191</td><td>-15.40</td><td>[-26.67, 1.93]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>C-O</td><td>scalar</td><td>191</td><td>3.72</td><td>[−10.63, 18.44]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>C-O</td><td>arg</td><td>191</td><td>-15.40</td><td>[-26.67, 1.93]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>O-S</td><td>scalar</td><td>200</td><td>2.32</td><td>[-1.64, 11.60]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>O-S</td><td>arg</td><td>200</td><td>0.00</td><td>[0.00, 0.00]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>N-G</td><td>scalar</td><td>200</td><td>12.76</td><td>[-2.37, 28.63]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>C-S</td><td>scalar</td><td>193</td><td>17.62</td><td>[2.34, 33.64]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>C-S</td><td>arg</td><td>193</td><td>0.94</td><td>[−13.62, 17.28]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>C-0</td><td>scalar</td><td>193</td><td>13.91</td><td>[-2.63, 30.76]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>C-O</td><td>arg</td><td>193</td><td>0.94</td><td>[-13.62, 17.28]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>O-S</td><td>scalar</td><td>200</td><td>5.03</td><td>[0.36, 10.20]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>O-S</td><td>arg</td><td>200</td><td>0.00</td><td>[0.00, 0.00]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>N-G</td><td>scalar</td><td>200</td><td>27.52</td><td>[8.41, 45.38]</td><td>2000</td></tr><tr><td>UCF</td><td></td><td>C Sum-Cap</td><td>scalar</td><td>192</td><td>-7.28</td><td>[-21.37, 1.27]</td><td>1997</td></tr><tr><td>UCF</td><td></td><td>C Sum-Cap</td><td>arg</td><td>192</td><td>-0.93</td><td>[−2.47, 0.00]</td><td>1997</td></tr><tr><td>UCF</td><td></td><td>N Sum-Cap</td><td>scalar</td><td>200</td><td>-4.32</td><td>[−27.52, 14.72]</td><td>1997</td></tr><tr><td>UCF</td><td></td><td>G Sum-Cap</td><td>scalar</td><td>200</td><td>2.45</td><td>[-21.80, 25.99]</td><td>1997</td></tr><tr><td>UCF</td><td></td><td>O Sum-Cap</td><td>scalar</td><td>200</td><td>-9.53</td><td>[-25.90, 1.16]</td><td>1997</td></tr><tr><td>UCF</td><td></td><td>O Sum-Cap</td><td>arg</td><td>200</td><td>4.05</td><td>[-3.03, 13.41]</td><td>1997</td></tr><tr><td>UCF</td><td></td><td>S Sum-Cap</td><td>scalar</td><td>200</td><td>4.05</td><td>[-3.03, 13.41]</td><td>1997</td></tr><tr><td>UCF</td><td></td><td>S Sum-Cap</td><td>arg</td><td>200</td><td>4.05</td><td>[-3.03, 13.41]</td><td>1997</td></tr><tr><td>XD</td><td></td><td>C Sum-Cap</td><td>scalar</td><td>184</td><td>-11.29</td><td>[-20.26, -1.59]</td><td>2000</td></tr><tr><td>XD</td><td></td><td>C Sum-Cap</td><td>arg</td><td>184</td><td>-12.12</td><td>[-28.24, 5.32]</td><td>2000</td></tr><tr><td>XD</td><td></td><td>N Sum-Cap</td><td>scalar</td><td>200</td><td>-14.49</td><td>[−25.30, -2.71]</td><td>2000</td></tr><tr><td>XD</td><td></td><td>G Sum-Cap</td><td>scalar</td><td>200</td><td>0.26</td><td>[-9.66, 14.16]</td><td>2000</td></tr><tr><td>XD</td><td></td><td>O Sum-Cap</td><td>scalar</td><td>200</td><td>-1.69</td><td>[-6.59, 7.35]</td><td>2000</td></tr><tr><td>XD</td><td></td><td>O Sum-Cap</td><td>arg</td><td>200</td><td>1.02</td><td>[-5.88, 8.45]</td><td>2000</td></tr><tr><td>XD</td><td></td><td>S Sum-Cap</td><td>scalar</td><td>200</td><td>1.02</td><td>[-5.88, 8.45]</td><td>2000</td></tr><tr><td>XD</td><td></td><td>S Sum-Cap</td><td>arg</td><td>200</td><td>1.02</td><td>[-5.88, 8.45]</td><td>2000</td></tr></table>

Table 9: Recomputed post-pilot ranking contrasts in percentage points. L is the same historical raw reference across both view comparisons, not two independent LAVAD runs. All have 20 groups and 2,000 requested replicates.
<table><tr><td>Data</td><td>View</td><td>Contrast</td><td>n</td><td>Diff.</td><td>95% interval</td><td>Valid</td></tr><tr><td>UCF</td><td>Sum</td><td>N-G</td><td>200</td><td>-8.26</td><td>[-24.28, 5.86]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>N-O</td><td>200</td><td>-8.47</td><td>[-28.04, 10.11]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>N-L</td><td>200</td><td>-0.97</td><td>[-15.00, 11.85]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>O-L</td><td>200</td><td>7.50</td><td>[-6.21, 19.13]</td><td>1997</td></tr><tr><td>UCF</td><td>Sum</td><td>G-L</td><td>200</td><td>7.29</td><td>[-9.78, 24.28]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>N-G</td><td>200</td><td>-1.50</td><td>[-33.97, 34.17]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>N-O</td><td>200</td><td>-13.68</td><td>[-35.50, 6.97]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>N-L</td><td>200</td><td>3.34</td><td>[-14.00, 21.00]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>O-L</td><td>200</td><td>17.03</td><td>[4.67, 30.50]</td><td>1997</td></tr><tr><td>UCF</td><td>Cap</td><td>G-L</td><td>200</td><td>4.84</td><td>[-21.65, 26.72]</td><td>1997</td></tr><tr><td>XD</td><td>Sum</td><td>N-G</td><td>200</td><td>12.76</td><td>[-2.37, 28.63]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>N-O</td><td>200</td><td>5.37</td><td>[-9.31, 19.55]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>N-L</td><td>200</td><td>20.74</td><td>[8.46, 33.16]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>O-L</td><td>200</td><td>15.36</td><td>[8.48, 25.61]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>G-L</td><td>200</td><td>7.97</td><td>[1.29, 15.07]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>N-G</td><td>200</td><td>27.52</td><td>[8.41, 45.38]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>N-O</td><td>200</td><td>18.18</td><td>[5.53, 31.50]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>N-L</td><td>200</td><td>35.23</td><td>[22.97, 45.61]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>O-L</td><td>200</td><td>17.05</td><td>[7.41, 26.48]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>G-L</td><td>200</td><td>7.71</td><td>[−7.63, 19.34]</td><td>2000</td></tr></table>

Table 10: Exploratory Noul minus Qwen generated-probability diferences on the original probability scale, not percentage points. Both methods cover 200/200 anchors; all 2,000 replicates are defined, in 20 groups.
<table><tr><td>Data</td><td>View</td><td>Metric</td><td>Diff.</td><td>95% interval</td><td>Valid</td></tr><tr><td>UCF</td><td>Sum</td><td>BRIER</td><td>0.0153</td><td>[-0.0012, 0.0356]</td><td>2000</td></tr><tr><td>UCF</td><td>Sum</td><td>NLL</td><td>-0.1207</td><td>[-0.4456, 0.0651]</td><td>2000</td></tr><tr><td>UCF</td><td>Cap</td><td>BRIER</td><td>-0.0221</td><td>[-0.0649, 0.0061]</td><td>2000</td></tr><tr><td>UCF</td><td>Cap</td><td>NLL</td><td>-0.8880</td><td>[−1.7734, -0.2112]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>BRIER</td><td>-0.0005</td><td>[-0.0429, 0.0385]</td><td>2000</td></tr><tr><td>XD</td><td>Sum</td><td>NLL</td><td>-0.0138</td><td>[−0.1184, 0.0809]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>BRIER</td><td>-0.0448</td><td>[-0.0847, -0.0109]</td><td>2000</td></tr><tr><td>XD</td><td>Cap</td><td>NLL</td><td>-1.3758</td><td>[-2.2517, -0.5641]</td><td>2000</td></tr></table>

## E Choice response audit

The extracts in Table 11 are actual answers.target values, reduced to numeric fields and ordered by rating for presentation. For each first-failure class, we selected the lexicographically first local record ID. We removed record IDs, evidence text and source identities. This selection does not use ground truth. The mass example has sum 0.99 and selects its maximum; the other has unit mass but chooses 0.0 with probability 0.15 while 0.7 has 0.16. Confidence is not required to equal either of those values.

Table 11: Actual numeric failure extracts. The full ordered distributions are shown immediately below.
<table><tr><td>First failure</td><td>Choice</td><td>Confidence</td><td>Sum</td></tr><tr><td>mass</td><td>1.0</td><td>0.8</td><td>0.99</td></tr><tr><td>choice not max</td><td>0.0</td><td>0.06</td><td>1.00</td></tr></table>

mass; probabilities in rating order 0.0, 0.1, . . . , 1.0:  
```json
[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.04, 0.11, 0.81]
```  
choice not max; probabilities in rating order 0.0, 0.1, . . . , 1.0:  
[0.15, 0.04, 0.05, 0.08, 0.09, 0.08, 0.14, 0.16, 0.11, 0.04, 0.06]

Across both paired views, UCF failures were 8/380 negative and 0/20 positive requests; XD failures were 11/326 negative and 5/74 positive requests. These are request-level exploratory counts, not independent samples or evidence that failure is random. The full-coverage Choice estimand remains unavailable. No permissive parsing, renormalization or repaired probability replaced the strict results. First-failure counts alone do not imply disjoint error categories; disjointness in this batch was established by an additional all-fields audit.

## F Historical cache replay and numerical audit

The full replay ran on 23 September 2026 with the pinned LAVAD evaluator at commit 1ad46c666d1b3cfb262f3dd84769acf873285056. It covered 69,634 UCF anchors / 1,111,808 frames (84,331 positive) and 146,449 XD anchors / 2,335,801 frames (539,562 positive). Raw ratings repeat over 16 frames and the final block is clipped to the cached frame count. Temporal intervals use the original inclusive endpoints and starting-frame ofset. Seven UCF videos have out-of-bound annotation endpoints; original clipping was preserved. No video-class label was copied to all its frames.

Table 12: Historical full-frame replay, distinct from every pilot table. Values (%) are retained at audit precision; these are cached author predictions, not new model runs.
<table><tr><td>Data</td><td>Prediction</td><td>AUC</td><td>AP</td><td>PR trap.</td></tr><tr><td>UCF</td><td>Raw</td><td>72.7904</td><td>18.2100</td><td>19.5168</td></tr><tr><td>UCF</td><td>Original refinement</td><td>80.2757</td><td>27.1911</td><td>27.0773</td></tr><tr><td>XD</td><td>Raw</td><td>80.6145</td><td>53.0951</td><td>57.2054</td></tr><tr><td>XD</td><td>Original refinement</td><td>85.3642</td><td>62.0045</td><td>62.0058</td></tr></table>

The stored replay reports include independent agreement with the pinned label function and scikit-learn ranking calculations to $1 0 ^ { - 1 0 }$ . Those full-frame reports were inspected for this revision; the newly recomputed sampled-anchor analyses are distinct. The full replay table is explicitly historical, not newly rerun GPU inference. All ten refinement-neighbor identities per anchor were checked against source raw scores. The main sampled reference uses only unrefined scores, so refined full-frame results are not its comparator.

The original numerical path was fixed before quality inspection. A mathematically equivalent stable-softmax path changed refined scores by at most $3 . 3 3 \times 1 0 ^ { - 1 6 }$ but altered some tie ordering and aggregate metrics. Both outputs remain archived. We claim reproduction of the published rounded primary values, not bitwise identity with historical saved scalars. No jitter was introduced to improve agreement.

## G Reproduction and version record

The pilot-completion ledger held or estimated \$0.110546310588 across 1,602 attempts, including two earlier synthetic connectivity calls; 24 invalid requests retained unknown reservations. This is a historical accounting snapshot, not an invoice or current account balance. No API calls were made for this revision.

The revised pilot evaluator reproduced the saved 20 cells and 44 comparisons exactly. A separate standard-library rank-sum/threshold implementation checked metrics and group-bootstrap diferences independently. Constants were also checked against their analytic formulas. The matched LAVAD reference verified 40 cache-file hashes and all 400 dataset/video/anchor joins. New intervals were recomputed rather than borrowed from another comparison. Synthetic CPU/cache tests are implementation checks and contribute no scientific result.

The retained research code supports two distinct operations: aggregate-only paper generation and metric recomputation from private inputs. These programs and their input files are not included in the submission source. Within the research workspace, the following commands run from its typed-vad directory; RUN\_QWEN and RUN\_JEV must name the complete pilot journal directories. A fresh output directory preserves prior analyses:

```shell
OUT=$(mktemp -d reports/pilot-recheck-XXXXXX)
python3 scripts/evaluate_pilot.py --data .local/pilot-v1 \
--run "$RUN_QWEN" --run "$RUN_JEV" \
--output "$OUT/pilot-recomputed.json"
python3 analysis/v1_revision/analyze.py --output-dir "$OUT"
python3 paper/arxiv-v1-final/build.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m unittest discover -s analysis/v1_revision -p 'test*.py' -v
```

Private-input reanalysis requires the applicable third-party permissions as well as the original journals. The analysis refuses to overwrite its versioned outputs and reads the original pilot paths fixed in its code. The retained aggregate snapshot can regenerate the paper without private text but cannot recompute metrics. In contrast, the submission source only typesets the already computed numbers and vector figures; it needs no Python program, model, credential or private URL. A local clean-source rebuild does not establish arXiv server compilation or acceptance.

This revision added the two constant references, exact matched historical scores, all-issues Choice and input-length audits, and their own exploratory intervals after seeing the pilot. It corrected the description of validation tolerances and excluded confidence/maximum diferences from violation accounting. Original prompts, failed outputs, closed journals and frozen splits were preserved. The separate remaining-group protocol and environment were not modified.