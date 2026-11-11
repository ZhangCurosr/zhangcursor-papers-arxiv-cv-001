# Interleaving anatomical inspection and diagnostic reasoning for explainable radiology report generation

Chentao Huang<sup>1</sup>, Runlong Lin<sup>2</sup>, Zhijun Wang<sup>1</sup>, Yunxiang Li<sup>1</sup>, Yuanyuan Sun<sup>1</sup>, Jian Wang<sup>1</sup>, Hongfei Lin<sup>1</sup>, Ling Luo<sup>1\*</sup>

<sup>1\*</sup>School of Computer Science and Technology, Dalian University of Technology, No. 2 Linggong Road, Ganjingzi District, Dalian, 116024, Liaoning, China.

<sup>2</sup>Department of Nuclear Medicine, The Second Hospital of Dalian Medical University, No. 467 Zhongshan Road, Shahekou District, Dalian, 116023, Liaoning, China.

\*Corresponding author(s). E-mail(s): lingluo@dlut.edu.cn; Contributing authors: cthuang@mail.dlut.edu.cn; longjunji1129@126.com; wzj@mail.dlut.edu.cn; lyxdlut@mail.dlut.edu.cn; syuan@dlut.edu.cn; wangjian@dlut.edu.cn; hflin@dlut.edu.cn;

## Abstract

Automated radiology report generation can draft clinically plausible reports, yet the pathway from visual evidence to reported findings remains opaque, leaving the diagnostic basis of these reports unexplained and the generated reports dificult to review or revise. We introduce ARIVE (Anatomy-guided Reasoning with Interleaved Visual Evidence), an explainable report generation framework that interleaves anatomical inspection with diagnostic reasoning prior to final report composition. To support this process, we constructed 159,416 multimodal chainof-thought (MCoT) trajectories from MIMIC-CXR, wherein visual evidence from inspected anatomical regions informs subsequent clinical interpretation. Trained on these trajectories and further optimized using anatomical and clinical-content rewards, ARIVE achieved superior clinical-eficacy performance compared with strong baselines on MIMIC-CXR and the abnormality-centered MIMIC-ABN. ARIVE’s reports were competitive with those of leading models in blinded expert grading, and its MCoT trajectories earned high ratings for diagnostic traceability. Furthermore, a paired review evaluation suggests that access to MCoT can aid in identifying errors within generated reports.

Keywords: Explainable radiology report generation, multimodal large language models, multimodal chain-of-thought, anatomical grounding

## 1 Introduction

Medical imaging reports translate visual observations into findings that inform diagnosis, treatment and follow-up. In automated medical report generation, fluent and clinically plausible language alone does not reveal how the image supports each finding. This lack of transparency is clinically consequential, as expert evaluations have shown that automated reports may contain substantial errors, including misidentified or incorrectly described findings [1]. Moreover, evaluations show that even when multimodal models answer correctly, their image interpretation and reasoning can be flawed [2]. Explainability should therefore be treated as a core requirement of automated reporting rather than an optional capability. Specifically, an explainable system should make the visual evidence for each finding inspectable and the diagnostic reasoning underlying it traceable, helping clinicians assess the report and identify unsupported content [3, 4].

Automated medical report generation has expanded beyond the image-to-text captioning [5–7] to address a broader range of clinical needs. Recent advances span multiple medical imaging domains [8–11], incorporate longitudinal imaging context [12, 13], and support clinician-assisted reporting workflows [1]. These developments broaden the potential clinical applications of automated reporting, but do not necessarily clarify how image evidence supports the generated findings.

Several approaches make the information used in report generation more explicit. Attention visualizations highlight image regions associated with generated findings [6], retrieval-based approaches present clinically similar cases [14], and graph-based methods use explicit clinical entities and relations to guide report generation [15–17]. Although informative, these approaches do not reveal how evidence gathered from specific regions updates diagnostic interpretation and ultimately shapes the generated report. Chainof-thought (CoT) reasoning [18–20] takes a diferent route by presenting intermediate diagnostic steps as a readable narrative. However, a textual CoT trajectory does not ensure that each reasoning step is grounded in the corresponding visual evidence. As the reasoning trajectory lengthens, later steps may rely increasingly on preceding text rather than the image, a pattern associated with visual forgetting [21, 22]. Such drift may allow unsupported assumptions to accumulate and propagate through the reasoning process, increasing the risk of hallucinated findings. CoT can therefore make a stated reasoning sequence explicit while leaving unclear which regional visual evidence supports the resulting findings. Grounded report generation provides a complementary form of transparency by explicitly linking report findings to spatially localized image evidence. MAIRA-2 [23] pairs findings with bounding boxes, UniBiomed [11] connects generated descriptions with segmentation outputs, and ophthalmic vision-language segmentation systems [24] use lesion masks for verification. These spatial outputs improve report inspectability by indicating the location associated with a finding and providing evidence against which it can be assessed. However, elucidating how regional visual evidence informs the diagnostic reasoning underlying the final report remains an unresolved challenge.

Within this broader landscape, we focus on radiology report generation (RRG) for chest radiography, the most commonly performed medical imaging examination. We introduce ARIVE (Anatomy-guided Reasoning with Interleaved Visual Evidence), a framework for explainable chest X-ray (CXR) report generation. ARIVE interleaves anatomical inspection with diagnostic interpretation, using the visual evidence obtained at each inspection step to guide subsequent reasoning and ultimately compose the final report. To support this workflow, we constructed 159,416 report-derived multimodal chain-of-thought (MCoT) training trajectories from MIMIC-CXR [25], grounded in 29 Chest ImaGenome anatomical regions [26]. We further introduce Anatomical and Clinical-Content Optimization (ACCO) to jointly optimize anatomical grounding, intermediate clinical content and final-report fidelity. Adding ACCO improved the measured clinical-content accuracy of both intermediate reasoning and final reports. Evaluated on MIMIC-CXR [25] and the abnormality-centered MIMIC-ABN [27], ARIVE achieved the strongest overall clinical-eficacy performance among the compared models. Blinded grading showed that ARIVE maintained competitive report quality, while assessment of its MCoT trajectories demonstrated high diagnostic traceability, linking final-report claims to intermediate interpretations. Paired review further showed that additional erroneous reports were identified when the corresponding MCoT trajectories were available, although review time increased. We also release scripts covering MCoT dataset construction, model training and inference to support reproduction of the full ARIVE pipeline. Together, these findings support integrating regional inspection into report generation to make the links between visual evidence, intermediate interpretations and reported findings inspectable. This approach ofers a transferable paradigm for clinically faithful and explainable report generation across medical imaging.

## 2 Results

## 2.1 Overview of ARIVE

ARIVE follows a three-step workflow that links MCoT trajectory construction, model training and MCoT-guided model inference (Fig. 1). First, we reorganized MIMIC-CXR reports into 159,416 diagnostic trajectories using radiologist-style templates [28] and grounded clinically significant steps in 29 Chest ImaGenome anatomical regions [26]. Second, we trained ARIVE with supervised fine-tuning (SFT) to interleave anatomical inspection with regional interpretation before composing the final report, followed by optimization with ACCO. ACCO combines four rewards for anatomical grounding (R<sub>BBOX</sub>), observation-level accuracy in the intermediate reasoning $( R _ { \mathrm { C X B } } ^ { \mathrm { r e a } } )$ and the final report $( R _ { \mathrm { C X B } } ^ { \mathrm { a n s } } )$ , and preservation of the clinical entities and relations in the reference report $( R _ { \mathrm { R G } } )$ . For the grounding and CheXbert rewards, inverse-frequency weighting assigns greater weight to underrepresented anatomical regions and clinical observations. Third, during inference, ARIVE first surveys the radiograph globally and then iteratively inspects diagnostically relevant anatomical regions, using the resulting visual evidence to guide its ongoing interpretation and, ultimately, the final report.

![](images/cc51e5ad783da4cb934425d2b18752dd09bcb2fc08dfbfe5035909c271165faa.jpg)  
Fig. 1 ARIVE workflow for dataset construction, model training and report generation. Step 1 (bottom), radiologist-style diagnostic templates guide GPT-OSS-120B in converting MIMIC-CXR reports into textual reasoning trajectories. Chest ImaGenome annotations ground report-relevant steps in 29 anatomical regions to construct the multimodal chain-of-thought (MCoT) dataset. Step 2 (left), supervised fine-tuning (SFT) is followed by Anatomical and Clinical-Content Optimization (ACCO). Four rewards assess anatomical grounding, observation accuracy in intermediate reasoning and final reports, and clinical entity–relation agreement with reference reports. Step 3 (top and right), ARIVE interleaves anatomical inspection with diagnostic reasoning before generating the final report. CXB and RG denote CheXbert and RadGraph, respectively.

We trained and evaluated dataset-specific ARIVE models on MIMIC-CXR, the primary benchmark, and on MIMIC-ABN. MIMIC-ABN comprises abnormal studies only, with reports restricted to abnormality-related findings. This abnormality-centered setting provides a focused assessment of abnormal-finding reporting while reducing the influence of normal-report content. Evaluation focused on clinical eficacy metrics, measured with CheXbert [29], SRR-BERT [30] and RadGraph [31] on both datasets and additionally with GREEN [32] on MIMIC-CXR. Lexical similarity to the reference reports was assessed with three natural language generation (NLG) metrics: BLEU-4 [33], METEOR [34] and ROUGE-L [35]. Anatomical grounding was evaluated on the corresponding Chest ImaGenome Gold subsets using region-selection $\mathrm { F _ { 1 } }$ and mean intersection over union (mIoU).

## 2.2 Comparison with existing report-generation models

We first compared ARIVE on MIMIC-CXR with six models from three categories: a general-purpose multimodal large language model (MLLM), Qwen3.5-397B-A17B [36] (hereafter Qwen3.5); a generalist medical MLLM, MedGemma-1.5 [37]; and four task-specific report-generation models, MAIRA-2 [23], Libra [13], MLRG [38] and PriorRG [12]. ARIVE achieved the highest scores across all seven clinical-eficacy measures: micro- and macro-averaged CheXbert $\mathrm { F _ { 1 } }$ over five common and all 14 observations, $\mathrm { F _ { 1 } }$ -SRR-BERT, RadGraph $\mathrm { F _ { 1 } }$ and GREEN (Table 1 and Fig. 2c). Its advantage was significant $( p < 0 . 0 1 )$ in 40 of the 42 comparisons. Both exceptions involved the strongest comparator on the respective measure: 14-observation macro $\mathrm { F _ { 1 } }$ versus Qwen3.5 and $\mathrm { F _ { 1 } { \mathrm { - } } S R R { \mathrm { - } } B E R T }$ versus PriorRG.

The comparators showed complementary strengths. On the CheXbert measures, the two MLLMs were the strongest comparators. Qwen3.5 led on the primary 14- observation macro $\mathrm { F _ { 1 } }$ , on which ARIVE achieved 0.437 (95% CI, 0.416–0.453) compared with 0.428 (0.407–0.444), and MedGemma-1.5 led on the other three, on which ARIVE scored 3.9–9.4% higher (for example, 0.603 versus 0.556 for 14-observation micro $\mathrm { F } _ { 1 } )$ At the observation level, ARIVE achieved the highest CheXbert $\mathrm { F _ { 1 } }$ for Cardiomegaly, Lung Opacity, Edema, Atelectasis, Pleural Efusion and No Finding (Fig. 2a).

The task-specific models generally showed greater agreement with reference reports in clinical entities and their relations: three of the four exceeded both MLLMs on RadGraph $\mathrm { F _ { 1 } }$ , and PriorRG led the comparators at 0.270. ARIVE scored slightly higher, at 0.271 (Table 1 and Fig. 2b). GREEN complements these measures by weighing matched findings against clinically significant errors, including false or missed findings and errors in location, severity or comparison with prior studies [32]. ARIVE achieved the highest GREEN score at 0.360, an increase of 2.3% over the second-ranked PriorRG (0.352; Fig. 2c). ARIVE also achieved the highest $\mathrm { F _ { 1 } { \mathrm { - } } S R R { \mathrm { - } } B E R T }$ , which captures fine-grained findings across 55 radiology labels (0.515 versus 0.511 for PriorRG).

We next evaluated abnormality-centered report generation on MIMIC-ABN. ARIVE again achieved the highest scores across all six tabulated measures: the four CheXbert measures, $\mathrm { F _ { 1 } { \mathrm { - } } S R R { \mathrm { - } } B E R T }$ and RadGraph $\mathrm { F _ { 1 } }$ (Table 1). On the primary 14-observation macro $\mathrm { F _ { 1 } , }$ ARIVE reached 0.408 (95% CI, 0.372–0.433) compared with 0.365 (0.326– 0.392) for Qwen3.5, the strongest comparator on this measure. The diference was larger than on MIMIC-CXR (0.043 versus 0.009) and was significant $( p \ < 0 . 0 1 )$ Overall, ARIVE’s scores were significantly higher in 32 of the 36 comparisons $( p < 0 . 0 5 )$ . The four exceptions were the two five-observation CheXbert measures against MLRG and F1-SRR-BERT against Qwen3.5 and MedGemma-1.5. These results also supported ARIVE’s performance on automated clinical-content metrics in the abnormality-centered setting.

Lexical similarity showed a diferent pattern: ARIVE did not achieve the highest BLEU-4, METEOR or ROUGE-L score on either dataset (Supplementary Table S1). These NLG metrics primarily assess overlap with reference wording and may miss clinically important errors in findings and their relations, which clinical-eficacy metrics

a CheXbert observation-wise F across models  
![](images/2ac5d5bf4b6950c912af17b2893ab4ab483e75c3160a183766c539ae25d2990f.jpg)  
b RadGraph F<sub>1</sub> across models

d CheXbert F change from SFT  
![](images/28394b039ab73d692e16940cb389d3861ce329ff846e224eaca2599cea8dc5ef.jpg)

![](images/0f63407ee13edf3240444c010e44518a088f6998d423e4fccad20bfe7e46a761.jpg)

e RadGraph F<sub>1</sub> by training strategy  
![](images/e8c9b68f02e63ce14e44e84740eae28e3c681b3d48eb4468aecbc388038b4585.jpg)

![](images/50e6039b7580c7f573a7917ce3b86d54eb4f010c44a837690ad28d4c1aa94ba9.jpg)  
g Anatomical localization performance

f GREEN scores by training strategy  
![](images/5bb7acb8d972793e6302a99402a76cb4e89acdc1e7ab612baf0ad8f3bae04601.jpg)

![](images/17e5dc92258e3bb31898e845df9bc9a93abdf12ec7609a08097016f227454926.jpg)  
Fig. 2 Clinical eficacy and anatomical grounding on MIMIC-CXR. (a)–(c), Comparison of ARIVE and six comparators using observation-wise CheXbert F<sub>1</sub> (a), RadGraph relation and entity $\mathrm { F _ { 1 } \ ( b ) }$ and GREEN (c). In (a), inner and outer tick labels show observation-specific $\mathrm { F _ { 1 } }$ axis limits (polygon area does not indicate overall performance), and annotations give ARIVE’s scores. (d)–(f), Comparison of SFT, SFT w/ CoT, SFT w/ MCoT and ARIVE using CheXbert $\mathrm { F _ { 1 } }$ changes from SFT in percentage points (d), RadGraph F<sub>1</sub> (e), and GREEN (f). (g), Region-selection F<sub>1</sub> (bars) and mean intersection over union (mIoU; lines) across all ground-truth regions in 636 Chest ImaGenome Gold examples. Error bars in (b,c,e,f) denote 95% confidence intervals. Asterisks indicate significantly higher scores for ARIVE in comparisons with the indicated models or training settings: $^ { * } p < 0 . 0 5 ; \ ^ { * * } p < 0 . 0 1$

Table 1 Clinical eficacy performance of ARIVE and comparator models on MIMIC-CXR and MIMIC-ABN. CheXbert F<sub>1</sub> subscripts indicate micro (mic) or macro (mac) averaging over five common or all 14 clinical observations. RG-F denotes RadGraph $\mathrm { F } _ { 1 } .$ Values are reported with 95% CIs in parentheses. Bold marks the highest value for each metric within each dataset. Asterisks indicate significantly higher scores for ARIVE than for the indicated comparator in one-sided comparisons using patient-clustered paired permutation tests for CheXbert and SRR-BERT and paired Wilcoxon signed-rank tests for RadGraph. $^ { * } p < 0 . 0 5 ; \ ^ { * * } p < 0 . 0 1$
<table><tr><td>Dataset</td><td>Model</td><td>F1,mic-5</td><td>F1,mac-5</td><td>F1,mic-14</td><td>F1,mac-14</td><td> $\mathrm { F _ { 1 } } \cdot$  -SRR-BERT</td><td>RG-F1</td></tr><tr><td></td><td>Qwen3.5-397B-A17B</td><td>0.567** (0.545, 0.588)</td><td> $0 . 5 1 9 ^ { * * }$  (0.498, 0.538)</td><td> $0 . 5 3 5 ^ { * * }$  (0.518, 0.551)</td><td>0.428 (0.407, 0.444)</td><td> $0 . 4 9 8 ^ { ^ { * * } }$  (0.488, 0.509)</td><td> $0 . 2 0 0 ^ { * * }$  (0.193, 0.207)</td></tr><tr><td></td><td>MedGemma-1.5</td><td> $0 . 5 9 6 ^ { ^ { \ast \ast } }$  (0.572, 0.618)</td><td> $0 . 5 3 4 ^ { ^ { \ast \ast } }$  (0.513, 0.553)</td><td> $0 . 5 5 6 ^ { * * }$  (0.539, 0.573)</td><td> $\boldsymbol { 0 . 4 0 9 } ^ { * * }$  (0.388, 0.425)</td><td> $0 . 4 9 1 ^ { ^ { \ast \ast } }$  (0.479, 0.502)</td><td> $0 . 2 1 9 ^ { ^ { \ast \ast } }$  (0.213, 0.226)</td></tr><tr><td></td><td>MAIRA-2</td><td> $0 . 5 6 3 ^ { * * }$  (0.540, 0.583)</td><td> $0 . 4 9 1 ^ { ^ { \ast \ast } }$  (0.469, 0.511)</td><td> $0 . 5 2 6 ^ { * * }$  (0.508, 0.541)</td><td> $0 . 4 0 7 ^ { * * }$  (0.386, 0.425)</td><td> $0 . 4 9 1 ^ { ^ { \ast \ast } }$  (0.478, 0.502)</td><td> $0 . 2 1 4 ^ { ^ { \ast \ast } }$  (0.207, 0.221)</td></tr><tr><td>MIMIC-CXR Libra</td><td></td><td> $0 . 5 8 3 ^ { * * }$  (0.557, 0.605)</td><td> $0 . 5 1 9 ^ { * * }$  (0.496, 0.539)</td><td> $0 . 5 3 7 ^ { * * }$  (0.516, 0.556)</td><td> $0 . 3 9 5 ^ { * * }$  (0.374, 0.412)</td><td> $0 . 4 8 9 ^ { * * }$  (0.474, 0.503)</td><td> $0 . 2 2 8 ^ { * * }$  (0.220, 0.236)</td></tr><tr><td></td><td>MLRG</td><td> $0 . 5 5 9 ^ { * * }$  (0.535, 0.580)</td><td> $0 . 4 9 1 ^ { ^ { \ast \ast } }$  (0.468, 0.512)</td><td> $0 . 5 2 0 ^ { * * }$  (0.502, 0.537)</td><td> $0 . 3 6 3 ^ { * * }$  (0.343, 0.379)</td><td> $0 . 4 9 2 ^ { * * }$  (0.479, 0.504)</td><td> $0 . 2 4 1 ^ { * * }$  (0.232, 0.250)</td></tr><tr><td></td><td>PriorRG</td><td> $0 . 5 7 5 ^ { * * }$  (0.552, 0.597)</td><td> $0 . 5 0 8 ^ { * * }$  (0.484, 0.531)</td><td> $0 . 5 3 7 ^ { * * }$  (0.518, 0.555)</td><td> $0 . 3 9 1 ^ { ^ { \ast } { } ^ { \ast } }$  (0.370, 0.408)</td><td>0.511 (0.498, 0.523)</td><td> $0 . 2 7 0 ^ { * * }$  (0.261, 0.279)</td></tr><tr><td></td><td>ARIVE (ours)</td><td>0.652 (0.630, 0.672)</td><td>0.555 (0.536, 0.571)</td><td>0.603 (0.586, 0.617)</td><td>0.437 (0.416, 0.453)</td><td>0.515 (0.503, 0.526)</td><td>0.271 (0.263, 0.279)</td></tr><tr><td></td><td>Qwen3.5-397B-A17B</td><td>0.486 ** (0.420, 0.545)</td><td> $0 . 4 5 1 ^ { ^ { \ast \ast } }$  (0.384, 0.508)</td><td> $0 . 4 5 9 ^ { ^ { \ast \ast } }$  (0.423, 0.491)</td><td> $0 . 3 6 5 ^ { * * }$  (0.326, 0.392)</td><td>0.434 (0.401, 0.470)</td><td>0.165 ** (0.151, 0.177)</td></tr><tr><td></td><td>MedGemma-1.5</td><td>0.529** (0.474, 0.581)</td><td> $0 . 4 7 9 ^ { * * }$  (0.422, 0.529)</td><td>0.481 ** (0.445, 0.517)</td><td> $0 . 3 6 0 ^ { * * }$  (0.320, 0.392)</td><td>0.421 (0.387, 0.459)</td><td>0.165** (0.151, 0.178)</td></tr><tr><td></td><td>MAIRA-2</td><td> $0 . 5 2 2 ^ { \ast \ast }$  (0.459, 0.577)</td><td> $0 . 4 6 9 ^ { * * }$  (0.407, 0.522)</td><td> $0 . 4 6 6 ^ { * * }$  (0.428, 0.504)</td><td> $0 . 3 6 0 ^ { * * }$  (0.326, 0.385)</td><td> $0 . 3 8 3 ^ { * * }$  (0.349, 0.420)</td><td>0.164** (0.148, 0.180)</td></tr><tr><td>MIMIC-ABN Libra</td><td></td><td> $0 . 4 8 5 ^ { * * }$  (0.439, 0.531)</td><td> $0 . 4 3 9 ^ { * * }$  (0.392, 0.485)</td><td> $0 . 4 5 8 ^ { * * }$  (0.423, 0.493)</td><td> $0 . 3 5 6 ^ { * * }$  (0.322, 0.384)</td><td> $0 . 4 0 9 ^ { * * }$  (0.376, 0.447)</td><td>0.140* ** (0.127, 0.155)</td></tr><tr><td></td><td>MLRG</td><td>0.575 (0.525, 0.623)</td><td>0.521 (0.462, 0.575)</td><td> $0 . 5 1 4 ^ { * }$  (0.474, 0.551)</td><td> $0 . 3 6 2 ^ { * }$  (0.333, 0.387)</td><td> $0 . 4 0 9 ^ { * * }$  (0.373, 0.451)</td><td>0.190** (0.168, 0.215)</td></tr><tr><td></td><td>PriorRG</td><td> $0 . 5 1 4 ^ { * * }$  (0.459, 0.564)</td><td> $0 . 4 5 6 ^ { * * }$  (0.398, 0.507)</td><td> $0 . 4 7 6 ^ { * * }$  (0.441, 0.510)</td><td> $0 . 3 3 2 ^ { \ast \ast }$  (0.300, 0.361)</td><td> $0 . 4 0 4 ^ { ^ { * * } }$  (0.369, 0.441)</td><td>0.191* (0.171, 0.211)</td></tr><tr><td></td><td>ARIVE (ours)</td><td>0.581 (0.529, 0.631)</td><td>0.541 (0.485, 0.590)</td><td>0.545 (0.514, 0.574)</td><td>0.408 (0.372, 0.433)</td><td>0.441 (0.405, 0.477)</td><td>0.207 (0.192, 0.222)</td></tr></table>

## 2.3 Ablation studies

assess more directly [39–41]. This divergence is consistent with ARIVE’s emphasis on clinical content rather than reference phrasing. We next performed staged ablations to assess the contributions of MCoT supervision and ACCO.

## 2.3.1 Efect of MCoT supervision on report generation

To assess the contribution of MCoT supervision, we compared report-only SFT with SFT on text-only CoT or MCoT trajectories (Table 2). On MIMIC-CXR, text-only CoT increased CheXbert $\mathrm { F _ { 1 } }$ on both five-observation measures and on 14-observation micro $\mathrm { F _ { 1 } }$ , whereas the primary 14-observation macro $\mathrm { F _ { 1 } }$ remained unchanged at 0.412. These gains did not extend to the complementary clinical-eficacy measures: F -SRR-BERT, RadGraph $\mathrm { F _ { 1 } }$ and GREEN all declined, with GREEN falling from 0.335 to 0.314 (Fig. 2e,f). MIMIC-ABN showed a similar pattern, with small increases in three of the four CheXbert measures accompanied by lower $\mathrm { F _ { 1 } }$ -SRR-BERT and RadGraph $\mathrm { F _ { 1 } } .$ Thus, although text-only CoT made the intermediate diagnostic statements explicit and readable, it did not consistently improve the clinical eficacy of the resulting reports, and the decline in GREEN suggests that a textual rationale alone was insuficient to reduce clinically important errors.

To assess the contribution of regional visual evidence, we next compared text-only CoT with MCoT, which interleaves anatomical inspection with diagnostic interpretation. On MIMIC-CXR, MCoT increased scores on every clinical-eficacy measure, including GREEN (from 0.314 to 0.320; Fig. 2f), except the primary 14-observation macro $\mathrm { F _ { 1 } }$ , which decreased slightly from 0.412 to 0.410. On MIMIC-ABN, MCoT increased scores on all six tabulated measures (Table 2), and all three NLG metrics improved on both datasets (Supplementary Table S2). Relative to text-only reasoning, interleaved regional evidence therefore improved the clinical eficacy of the generated reports, supporting the contribution of interleaved regional visual evidence to clinically faithful report generation.

On the 142-example MIMIC-ABN Chest ImaGenome Gold subset, MCoT also improved intermediate diagnostic reasoning and its consistency with the final report (Table 3). Relative to text-only CoT, MCoT increased reasoning-stage macro $\mathrm { F _ { 1 } }$ by 0.103 (0.413 versus 0.310) and final-report macro $\mathrm { F _ { 1 } }$ by 0.037 (0.411 versus 0.374), and raised reasoning–report consistency macro $\mathrm { F _ { 1 } }$ from 0.728 to 0.894. Interleaved regional evidence thus strengthened clinically relevant interpretation and helped findings propagate more consistently from intermediate reasoning to the final report.

Table 2 Clinical eficacy performance across training settings on MIMIC-CXR and MIMIC-ABN. CheXbert $\mathrm { F } _ { 1 }$ subscripts indicate micro (mic) or macro (mac) averaging over five common or all 14 clinical observations. SFT uses report-only supervision; $\operatorname { S F T } \mathrm { \mathrm { w } } /$ CoT and SFT w/ MCoT add text-only and region-interleaved diagnostic trajectories, respectively. ARIVE denotes SFT w/ MCoT followed by ACCO. Values are reported with 95% CIs in parentheses. Bold marks the highest value for each metric within each dataset. Asterisks indicate significantly higher scores for ARIVE than for the indicated training setting, under the tests specified in Table 1. $^ { * } p < 0 . 0 5 ; \ ^ { * * } p < 0 . 0 1$
<table><tr><td>Dataset</td><td>Training setting</td><td> $\mathrm { F _ { 1 , m i c - 5 } }$ </td><td> $\mathrm { F _ { 1 , m a c - 5 } }$ </td><td> $\mathrm { F _ { 1 , m i c - 1 4 } }$ </td><td> $\mathrm { F _ { 1 , m a c - 1 4 } }$ </td><td> $\mathrm { F _ { 1 } }$  -SRR-BERT</td><td>RG-F1</td></tr><tr><td rowspan="2"></td><td>SFT</td><td> $0 . 5 7 6 ^ { * * }$   $\underline { { ( 0 . 5 5 4 , 0 . 6 0 0 ) } }$ </td><td> $0 . 5 0 6 ^ { ^ { * * } }$   $\underline { { ( 0 . 4 8 3 , 0 . 5 2 7 ) } }$ </td><td> $0 . 5 3 7 ^ { * * }$   $\underline { { ( 0 . 5 2 1 , 0 . 5 5 5 ) } }$ </td><td> $0 . 4 1 2 ^ { * * }$   $\underline { { ( 0 . 3 9 3 , 0 . 4 2 8 ) } }$ </td><td> $0 . 5 0 3 ^ { * }$   $\underline { { ( 0 . 4 9 5 , 0 . 5 1 7 ) } }$ </td><td> $0 . 2 4 3 ^ { * * }$   $\underline { { ( 0 . 2 3 5 , 0 . 2 5 1 ) } }$ </td></tr><tr><td>SFT w/ CoT</td><td> $0 . 5 9 0 ^ { * * }$   $( 0 . 5 6 9 , \ 0 . 6 1 3 )$ </td><td> $0 . 5 1 9 ^ { * * }$   $\underline { { ( 0 . 5 0 1 , 0 . 5 3 8 ) } }$ </td><td> $0 . 5 4 5 ^ { * * }$   $\underline { { ( 0 . 5 3 0 , 0 . 5 6 4 ) } }$ </td><td> $0 . 4 1 2 ^ { * * }$   $( 0 . 3 8 3 , 0 . 4 2 3 )$ </td><td> $0 . 4 8 7 ^ { * * }$   $\underline { { ( 0 . 4 7 4 , 0 . 4 9 9 ) } }$ </td><td> $0 . 2 2 7 ^ { \ast \ast }$   $\underline { { ( 0 . 2 1 9 , 0 . 2 3 5 ) } }$ </td></tr><tr><td>MIMIC-CXR</td><td>SFT w/ MCoT</td><td> $0 . 6 0 1 ^ { \ast \ast }$   $\underline { { ( 0 . 5 7 6 , 0 . 6 1 6 ) } }$ </td><td> $0 . 5 2 3 ^ { \ast \ast }$   $\underline { { ( 0 . 5 0 6 , 0 . 5 4 5 ) } }$ </td><td> $0 . 5 5 0 ^ { * * }$   $\underline { { ( 0 . 5 3 7 , 0 . 5 6 8 ) } }$ </td><td> $0 . 4 1 0 ^ { * * }$   $\underline { { ( 0 . 3 9 1 , 0 . 4 2 5 ) } }$ </td><td> $0 . 5 0 1 ^ { \ast \ast }$   $( 0 . 4 \mathrm { \bar { 8 } 8 , \ 0 . 5 1 0 ) }$ </td><td> $0 . 2 3 6 ^ { \ast \ast }$  (0.228, 0.244)</td></tr><tr><td></td><td>ARIVE</td><td>0.652 (0.630, 0.672)</td><td>0.555 (0.536, 0.571)</td><td>0.603 (0.586, 0.617)</td><td>0.437 (0.416, 0.453)</td><td>0.515 (0.503, 0.526)</td><td>0.271 (0.263, 0.279)</td></tr><tr><td></td><td>SFT</td><td> $\boldsymbol { 0 . 5 4 6 } ^ { * }$   $( 0 . 4 9 0 , 0 . 5 9 9 )$ </td><td> $0 . 4 8 2 ^ { * }$   $( 0 . 4 \dot { 2 } 7 , \dot { 0 } . 5 3 0 )$ </td><td> $0 . 4 9 7 ^ { * * }$   $( 0 . 4 6 2 , 0 . 5 3 1 )$ </td><td> $0 . 3 6 6 ^ { * }$   $( 0 . 3 3 2 , 0 . 3 9 4 )$ </td><td> $0 . 4 0 8 ^ { * * }$   $( 0 . 3 7 5 , 0 . 4 4 3 )$ </td><td>0.193 (0.176, 0.211)</td></tr><tr><td rowspan="2">MIMIC-ABN</td><td>SFT w/ CoT</td><td> $0 . 5 4 5 ^ { * * }$  (0.483, 0.583)</td><td> $0 . 4 9 7 ^ { * }$  (0.443, 0.536)</td><td> $0 . 5 0 2 ^ { * * }$   $( 0 . 4 \dot { 7 } 1 , \ 0 . 5 3 1 )$ </td><td> $0 . 3 7 4 ^ { * }$  (0.339, 0.396)</td><td> $0 . 3 9 4 ^ { ^ { \ast \ast } }$  (0.368, 0.429)</td><td> $0 . 1 8 3 ^ { * * }$  (0.165, 0.199)</td></tr><tr><td>SFT w/ MCoT</td><td>0.555 (0.519, 0.621)</td><td>0.512 (0.474, 0.591)</td><td> $0 . 5 { 1 4 } ^ { * * }$   $( 0 . 4 \ 7 0 , \ 0 . 5 4 0 )$ </td><td> $0 . 3 7 7 ^ { * }$   $( 0 . 3 \dot { 3 } 7 , 0 . 4 0 7 )$ </td><td> $\boldsymbol { 0 . 4 1 0 } ^ { * * }$   $( 0 . 3 6 9 , 0 . 4 3 4 )$ </td><td> $0 . 1 8 8 ^ { * }$   $\left( 0 . 1 \bar { 7 } 3 , \ 0 . 2 0 3 \right)$ </td></tr><tr><td></td><td>ARIVE</td><td>0.581 (0.529, 0.631)</td><td>0.541 (0.485, 0.590)</td><td>0.545 (0.514, 0.574)</td><td>0.408 (0.372, 0.433)</td><td>0.441 (0.405, 0.477)</td><td>0.207 (0.192, 0.222)</td></tr></table>

## 2.3.2 Contributions of ACCO and its components

With MCoT supervision fixed, adding ACCO improved all six tabulated clinical-eficacy measures on both datasets (Table 2). Relative to SFT w/ MCoT, the primary 14- observation macro $\mathrm { F _ { 1 } }$ increased by 6.6% on MIMIC-CXR (0.437 versus 0.410) and by

8.2% on MIMIC-ABN (0.408 versus 0.377), and GREEN increased from 0.320 to 0.360 on MIMIC-CXR (Fig. 2f). All MIMIC-CXR comparisons were significant at $p < 0 . 0 1$ 2 and four of the six tabulated MIMIC-ABN comparisons were significant at $p < 0 . 0 5$ These gains spanned finding accuracy, fine-grained clinical content and entity–relation structure. Combined with MCoT supervision, ACCO yielded a net gain over reportonly SFT in CheXbert $\mathrm { F _ { 1 } }$ for nine of the 14 observations on MIMIC-CXR, with equal performance for Pleural Other and the largest gains for Edema and Atelectasis (Fig. 2d).

To assess the contribution of each ACCO component, we removed each reward, or the inverse-frequency weighting, in turn and evaluated report accuracy, anatomical grounding, reasoning-stage clinical content and reasoning–report consistency on MIMIC-ABN and its Gold subset (Table 3). Removing the anatomical-grounding reward R<sub>BBOX</sub> reduced every reported measure, supporting its contribution to both anatomical grounding and downstream clinical content. Removing the reasoning-stage clinical reward $R _ { \mathrm { C X B } } ^ { \mathrm { r e a } }$ reduced reasoning-stage and final-report $\mathrm { F _ { 1 } }$ on the Gold subset and lowered reasoning–report consistency macro $\mathrm { F _ { 1 } }$ from 0.887 to 0.782. These concurrent declines support the role of intermediate supervision in preserving clinical content in the final report.

Removing the report-stage clinical reward $R _ { \mathrm { C X B } } ^ { \mathrm { a n s } }$ produced the largest reduction in final-report micro $\mathrm { F _ { 1 } }$ among the ACCO component ablations, both on the Gold subset (0.463 versus 0.505, an 8.3% relative decline) and on the full MIMIC-ABN test set (0.526 versus 0.545). However, region-selection $\mathrm { F _ { 1 } }$ and mIoU increased without this reward, with mIoU reaching 0.493 compared with 0.474 for full ARIVE. This divergence indicates that better anatomical grounding alone does not ensure more accurate findings in the final report. Removing the RadGraph reward $R _ { \mathrm { R G } }$ lowered both RadGraph $\mathrm { F _ { 1 } }$ (0.200 versus 0.207) and $\mathrm { F _ { 1 } { \mathrm { - } } S R R { \mathrm { - } } B E R T }$ (0.417 versus 0.441) on MIMIC-ABN, supporting its contribution to preserving clinical entities, their relations and fine-grained findings.

Removing inverse-frequency weighting slightly improved overall anatomical grounding but reduced macro-averaged $\mathrm { F _ { 1 } }$ for both reasoning-stage and final-report content on the Gold subset, with the latter falling from 0.444 to 0.400. Final-report micro $\mathrm { F _ { 1 } }$ was unchanged (0.505), suggesting that the losses were concentrated in less frequent observations, which the weighting is designed to emphasize. Likewise, the gain in overall grounding may reflect greater emphasis on frequently annotated regions once weighting is removed. Together, these ablations show that anatomical grounding and clinical eficacy do not necessarily improve in parallel, underscoring the need to optimize them jointly.

The 636-example MIMIC-CXR Gold subset provided complementary evidence on anatomical grounding (Fig. 2g). On this subset, region-selection $\mathrm { F _ { 1 } }$ and mIoU improved progressively from text-only CoT to MCoT and then to full ARIVE, whereas on the smaller MIMIC-ABN Gold subset, overall grounding was similar before and after ACCO (Table 3). The frequently annotated right and left lungs showed similarly high region-selection performance in all three settings. The less frequently annotated right upper lung zone, by contrast, difered markedly: its region-selection $\mathrm { F _ { 1 } }$ reached 0.226 with ARIVE, compared with 0.083 with MCoT and 0.043 with text-only CoT. These results suggest that ACCO improves region selection for underrepresented anatomy.

Table 3 MCoT supervision and ACCO component ablations on MIMIC-ABN and its 142-example Chest ImaGenome Gold subset. CheXbert F<sub>1,14</sub> values are micro/macro averages across 14 observations; superscripts rea, ans and con denote reasoning-stage content, final-report content and reasoning–report consistency, respectively. Bold marks the highest value for each metric across all settings; micro/macro maxima are identified separately.
<table><tr><td rowspan="2">Training setting</td><td colspan="3">MIMIC-ABN test set</td><td colspan="5">Chest ImaGenome Gold subset</td></tr><tr><td>F1,14 (mic/mac)</td><td>F1-SRR-BERT RG-F1</td><td></td><td>Region- selection F1</td><td>mIoU</td><td> $\mathrm { F _ { 1 , 1 4 } ^ { r e a } }$  (mic/mac)</td><td> $\mathrm { F _ { 1 , 1 4 } ^ { a n s } }$  (mic/mac)</td><td> $\mathrm { F } _ { 1 , 1 4 } ^ { \mathrm { c o n } }$  (mic/mac)</td></tr><tr><td colspan="10">Supervised ine-tuning comparison</td></tr><tr><td>SFT w/ CoT</td><td>0.502/0.374</td><td>0.394</td><td>0.183</td><td>0.579</td><td>0.430</td><td>0.369/0.310</td><td>0.455/0.374</td><td>0.714/0.728</td></tr><tr><td>SFT w/ MCoT</td><td>0.514/0.377</td><td>0.410</td><td>0.188</td><td>0.605</td><td>0.474</td><td>0.463/0.413</td><td>0.472/0.411</td><td>0.902/0.894</td></tr><tr><td colspan="9">ACCO ablations</td></tr><tr><td>ARIVE</td><td>0.545/0.408</td><td>0.441</td><td>0.207</td><td>0.603</td><td>0.474</td><td>0.513/0.451</td><td>0.505/0.444</td><td>0.945/0.887</td></tr><tr><td>w/o RBBOX</td><td>0.539/0.396</td><td>0.421</td><td>0.196</td><td>0.591</td><td>0.455</td><td>0.501/0.421</td><td>0.492/0.400</td><td>0.914/0.856</td></tr><tr><td> $\mathbf { w } / \mathrm { o } \ R _ { \mathrm { C X B } } ^ { \mathrm { r e a } }$ </td><td>0.539/0.400</td><td>0.441</td><td>0.211</td><td>0.612</td><td>0.479</td><td>0.484/0.381</td><td>0.481/0.389</td><td>0.894/0.782</td></tr><tr><td>w/o RB</td><td>0.526/0.388</td><td>0.428</td><td>0.212</td><td>0.618</td><td>0.493</td><td>0.500/0.426</td><td>0.463/0.392</td><td>0.929/0.911</td></tr><tr><td>w/o RRG</td><td>0.542/0.388</td><td>0.417</td><td>0.200</td><td>0.608</td><td>0.477</td><td>0.502/0.390</td><td>0.485/0.379</td><td>0.922/0.730</td></tr><tr><td>w/o weighting</td><td>0.538/0.400</td><td>0.428</td><td>0.208</td><td>0.608</td><td>0.481</td><td>0.496/0.379</td><td>0.505/0.400</td><td>0.940/0.884</td></tr></table>

Taken together, the staged comparisons and component ablations reveal complementary roles for MCoT and ACCO. Relative to text-only CoT, MCoT improved the measured clinical content of intermediate reasoning and its consistency with final reports by incorporating regional visual evidence. Adding ACCO improved report-level clinical-content metrics and entity–relation agreement, with gains in macro-averaged observation scores and in region selection for some underrepresented anatomy. We next assessed generated reports and MCoT trajectories through radiologist grading and examined whether access to MCoT supported report review.

## 2.4 Clinical evaluation of generated reports and MCoT trajectories

## 2.4.1 Blinded grading of report quality

Following previous expert evaluations [8], an experienced radiologist graded reference reports and reports generated by ARIVE, PriorRG and Qwen3.5 for 100 randomly selected MIMIC-CXR cases from 71 patients (Fig. 3a–c). These two comparators were the strongest on complementary measures: PriorRG led the comparators on RadGraph F<sub>1</sub>, F<sub>1</sub>-SRR-BERT and GREEN, and Qwen3.5 on the primary 14-observation macro F<sub>1</sub> (Table 1). Reports were presented with their sources concealed, together with the current radiograph, the clinical indication and, when available, the prior image and report. Each report was graded for clinical accuracy on a five-point scale from 1 (unusable) to 5 (directly usable).

ARIVE reports more often required at most minor edits (grades 4–5) than PriorRG reports (50% versus 40%), although less often than Qwen3.5 reports (56%). The corresponding proportion for reference reports was 81% (Fig. 3a). ARIVE’s mean score was higher than that of PriorRG (3.47 versus 3.31) and similar to that of Qwen3.5 (3.51; Fig. 3b). Reference reports received the highest mean score (3.88), exceeding those of all three models. In paired comparisons, ARIVE received a higher grade than

PriorRG in 30 cases, a lower grade in 19 and an equal grade in 51. Against Qwen3.5, the corresponding counts were 31, 34 and 35 (Fig. 3c).

To examine report content alongside these grades, we used a large language model (DeepSeek-V4.1-Flash [42]) to extract observation states from the same 400 reports and compared each generated report with its reference (Fig. 3d). ARIVE’s higher mean score than PriorRG was accompanied by greater coverage of reference observations and fewer observations absent from the references. ARIVE and Qwen3.5, despite their similar mean scores, difered in what they reported. ARIVE recovered slightly fewer reference observations than Qwen3.5 (71.3% versus 73.0%), but 81.1% of its reported observations were supported by the references, compared with 57.3% for Qwen3.5. Qwen3.5 reported more broadly, capturing slightly more reference findings but also more observations without reference support (216 versus 66), whereas ARIVE’s reports were more selective, trading slightly lower coverage for higher agreement with the references.

## 2.4.2 Radiologist grading of MCoT trajectories

To assess the MCoT trajectories generated before ARIVE’s reports, the radiologist graded trajectories for 100 randomly selected cases from 80 patients, sampled separately from the cases used for report grading. Each trajectory was graded from 1 (lowest) to 5 (highest) in four domains: clinical factual correctness, anatomical localization accuracy, diagnostic traceability (whether the main judgments in the final report could be traced to the trajectory) and expression quality, yielding 400 grades in total.

Ratings were consistently high in three domains: clinical factual correctness, diagnostic traceability and expression quality received grades of 4 or 5 in 99%, 100% and 100% of trajectories, respectively (Fig. 4a). Anatomical localization accuracy was more variable, with 81% of trajectories graded 4 or 5, 18% graded 3 and 1% graded 2. The top grade was assigned for diagnostic traceability in 90% of cases but for anatomical localization accuracy in only 50%. Mean scores followed the same pattern: diagnostic traceability (4.90; 95% CI, 4.84–4.96) and clinical factual correctness (4.87; 4.79–4.94) scored highest, followed by expression quality (4.67; 4.58–4.76) and anatomical localization accuracy (4.30; 4.14–4.46; Fig. 4b). The radiologist attributed the lower localization scores partly to incomplete coverage of the left lung and heart in the anatomical crops.

To complement the traceability grades, we quantified reasoning–report consistency at the level of individual claims and findings (Fig. 4c). Using the same large language model, we extracted claims from the final reports and findings from the MCoT trajectories, and then measured the proportion of report claims supported by the reasoning and the proportion of reasoning findings retained in the reports. Partial support denoted agreement on the core finding but not on all of its attributes. Of 597 positive, negative or uncertain report claims, 572 (95.8%) were fully supported by the corresponding trajectories, 23 (3.9%) were partially supported (14 with diferences in temporal attributes, 7 in certainty, 1 in location and 1 in morphology) and 2 (0.3%) were contradicted (one pleural efusion claim and one pulmonary vascular congestion claim). Nearly all report claims could therefore be traced to the trajectories, consistent with the high traceability grades.

![](images/395cc169d80dfd63fd5733640b362d46b2492650f82cec6214d0256f9cb47c93.jpg)

![](images/5c9f4330670c345b7151959f16ba7ce77526416ba76c93386044838ce24a0281.jpg)

c Paired blinded scores: ARIVE versus comparators  
![](images/313fa30bf3403951ae4969209d9f86c183e41ffd455205196c6cf60e87191cbb.jpg)

![](images/3cde4e2662ef1cb6302fa9bb67f00958cdcab7cf6bd9223b88214ba2f7e22751.jpg)

![](images/de28d991f0790d7313ea9f99da208c94882d1cbd468efe78fe3884f604451578.jpg)

d Overlap of positive or uncertain observations with reference reports  
![](images/4fca9bdb6b64cb8d24360057336d32278657b7c43df30e1fd9724aea2058f923.jpg)

![](images/ed25ea875d79e93b2e145d9099fbb75e482ecfa320d28417ebc244672775e2d5.jpg)

![](images/3e3f08468c40ebf85c8f62e539bc383414aa470cd8a6e48cca4ec4082961562d.jpg)  
Fig. 3 Report-quality grading and observation overlap with reference reports. (a), Blinded grade distributions assigned by the radiologist for 100 randomly selected cases from 71 patients. Scores range from 1 (unusable) to 5 (directly usable). (b), Mean report-quality scores with 95% confidence intervals. (c), Paired scores for ARIVE versus PriorRG, Qwen3.5 and reference reports, from left to right. Cells show case counts and dashed diagonals indicate equal scores. (d), Observation overlap between generated and reference reports. Left, ARIVE; center, Qwen3.5; right, PriorRG. Numbers indicate observations shared by the generated and reference reports or present in only one.

In the reverse direction, partial retention denoted that a report preserved the core finding but lost or altered clinically relevant information. Of 449 positive or uncertain reasoning findings, 412 (91.8%) were fully retained in the reports, 14 (3.1%) were partially retained (temporal information changed or missing in 9 and certainty changed in 5) and 23 (5.1%) were omitted (8 involving devices, 5 pneumonia, 4 pulmonary edema, 3 consolidation, 2 atelectasis and 1 pleural efusion). Most positive or uncertain

a MCoT grade distribution

b Mean MCoT scores  
![](images/22e4fd11ef2a0f694a698bd13af07344b77d528610d7e54725238931a9726ba0.jpg)  
c Reasoning–report consistency

![](images/1f03ea6773e328e36df7a28062ee53ba446a6dbb6b457377208a656b4f72f91e.jpg)

![](images/eff7c9292eef9a84a4216476b589e549b902fb9860d065ee1c1c27bb79a87044.jpg)  
d Reports with errors identified during review

![](images/d34ab4ff260c22be065c22dd6550455f30032fe350d394673151c571ad3b025d.jpg)

![](images/c37fb17f3836b7d6dcfeedf2e1677a8de1ff80155778dffccf97e8003ac3f369.jpg)

e Review time per report  
![](images/755761016f0803014e65a7a7e143c95c0f9a9917b4516290fbf4490b11e0d8e4.jpg)  
Fig. 4 MCoT trajectory quality, reasoning–report consistency and report review. (a), Grade distributions assigned by the radiologist across four domains for 100 randomly selected cases from 80 patients. Scores range from 1 (lowest) to 5 (highest). (b), Mean MCoT scores with 95% confidence intervals. (c), Bidirectional reasoning–report consistency. Left, support from MCoT trajectories for report claims $( n = 5 9 7 ;$ positive, negative or uncertain). Right, retention of MCoT findings in the final reports $( n = 4 4 9 ;$ positive or uncertain). (d), Number of reports judged to contain errors during paired review without and with MCoT (8 and 11 of 100 reports, respectively). (e), Review time per report under each condition. Points show individual review times and shaded curves show kernel density estimates. Boxes span the interquartile range (IQR), with medians marked. Whiskers extend to the most extreme observations within 1.5 IQR of the box edges.

findings in the trajectories therefore carried through to the final reports, indicating strong reasoning–report consistency in both directions.

## 2.4.3 MCoT-assisted review of generated reports

To assess whether access to MCoT could help reviewers identify errors in generated reports, the radiologist reviewed the ARIVE reports for the same 100 cases in two sessions. In the first session, each report was reviewed alongside its source radiograph. After a two-week interval intended to reduce recall, the same reports and radiographs were reviewed with the corresponding MCoT trajectories available. In each session, we recorded which reports were judged to contain clinically significant errors (false or omitted findings) and the review time for each report.

The radiologist identified errors in 8 reports without MCoT and in 11 with MCoT (Fig. 4d). Errors were identified again in all 8 reports flagged in the first session, and the MCoT-assisted session identified 3 additional reports containing errors. Median review time increased from 24.5 s (interquartile range (IQR), 18.0–36.4 s) without MCoT to 42.5 s (IQR, 31.8–56.1 s) with MCoT (Fig. 4e). These paired results provide preliminary evidence that access to MCoT may support error identification, although review time was longer. The case studies below illustrate how regional evidence shapes individual trajectories and how findings and errors can be traced into the final reports.

## 2.5 Case studies of generated reports and diagnostic trajectories

## 2.5.1 Comparison of generated reports

We compared reports generated by ARIVE, PriorRG and Qwen3.5 with the reference reports in three selected cases (Fig. 5). In Sample 1, ARIVE correctly reported the low lung volumes and mild pulmonary edema, together with the absence of focal consolidation, pneumothorax and a large pleural efusion. PriorRG instead added mild cardiomegaly, bibasal atelectasis and a possible left pleural efusion, whereas Qwen3.5 introduced a tracheostomy tube and moderate bilateral pleural efusions. In Sample 2, ARIVE retained the endotracheal tube, peripherally inserted central catheter (PICC), unchanged retrocardiac atelectasis, small left pleural efusion and absence of pneumothorax. PriorRG omitted both the pleural efusion and the retrocardiac abnormality, while Qwen3.5 incorrectly reported removal of the PICC and interpreted the chronic opacity as progressive aspiration pneumonia. ARIVE, however, erroneously placed the feeding-tube tip in the stomach, whereas the reference stated that the tip was outside the field of view. In Sample 3, ARIVE preserved the three-lead pacemaker configuration, cardiac enlargement, pulmonary edema, small bilateral pleural efusions, basilar atelectasis and absence of pneumothorax. PriorRG described only two leads and omitted the pleural efusions and basilar opacity, whereas Qwen3.5 overstated the right pleural efusion as moderate. ARIVE omitted the aortic calcification and the diferential considerations for the left basilar opacity but avoided these comparator errors.

Across these cases, ARIVE captured the major abnormalities, devices and key negative findings with fewer unsupported details than the comparators, while still making occasional errors and omissions. These examples complement the quantitative results by illustrating both concordant and unsupported content within individual reports.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Sample 1</td><td rowspan=2 colspan=2>Sample 2                    Sample 3<img src="images/381e7818e6440eaf7b60d83d1d5d5aa55da99c09041773a1d12649567602db96.jpg"/><img src="images/ad02f5e554a0fc5e9d1942c49bcee770950b251b3d3d870baf80c6ffb8fb72d6.jpg"/></td></tr><tr><td rowspan=1 colspan=1>CXRag ges</td><td rowspan=1 colspan=1><img src="images/f2d383d079cdc2aaaedc7b0b0dc4d271be758c81547e8d0ac555f060b043c2d0.jpg"/></td></tr><tr><td rowspan=1 colspan=1>Refnce</td><td rowspan=1 colspan=1>There lungs are low in volume but withoutfocal consolidation. Diffuse opacities likelyreflect mild pulmonary edema. There is nopleural effusion or pneumothorax. Thecardiac size and cardiac silhouette areobscured by low lung volumes. Themediastinal and hilar contours appearunremarkable.</td><td rowspan=1 colspan=1>Since the prior study, an endotracheal tubehas been placed. Its tip is 5.3 cm from thecarina. A PICC ends in the mid SVC. Afeeding tube overlies the stomach with thetip out of view. A pleural effusion on the leftis small. A persistent consolidation at theleft base is unchanged and likely reflectschronic atelectasis. There are no newopacities. There is no pneumothorax.Cervical hardware and right humeral softtissue anchors are unchanged.</td><td rowspan=1 colspan=1>Left-sided pacemaker device is noted withleads terminating in the right atrium, rightventricle, and coronary sinus. The heartsize is mildly enlarged. The aortic knob iscalcified. There is mild pulmonary edemawith perihilar haziness and vascularindistinctness, new from the prior study.Focal opacities at lung bases may reflectareas of atelectasis though infection cannotbe excluded. Small bilateral pleuraleffusions may be present. Nopneumothorax is identified.</td></tr><tr><td rowspan=1 colspan=1>ARIIVE</td><td rowspan=1 colspan=1>Portable AP chest radiograph was obtained.Lung volumes are low.Heart size is normal.Mediastinal and hilar contours areunremarkable. There is mild pulmonaryedema. No large pleural effusion orpneumothorax is seen. No definite focalconsolidation is identified.</td><td rowspan=1 colspan=1>Single portable view of the chestdemonstrates the ET tube ending4 cmabove the carina. NG tube termiñates in thestomach.Right-sided PICC line ends in themid SVC. Left retrocardiac opacity, likelyatelectasis, is unchanged since  Smallleft pleural effusion is also stable. There isno pneumothorax.Cardiomediastinal andhilar contours are normal.</td><td rowspan=1 colspan=1>A left-sided pacemaker device is noted withleads terminating in the right atrium, rightventricle, and coronary sinus. The heartsize is enlarged. There are small bilateralpleural effusions. Lung volumes are low.There is pulmonary vascular congestion aswell as mild interstitial edema. More focalopacity at the right lung base likely reflectsatelectasis. There is no pneumothorax.</td></tr><tr><td rowspan=1 colspan=1>PriRG</td><td rowspan=1 colspan=1>Portable upright view of the chestdemonstrates low lung volumes, whichaccentuate bronchovascular markings.Perihilar vascular congestion is noted.Perihilar vascular congestion is noted. Heartis mildly enlarged. There is blunting of theleft costophrenic angle, suggestive ofpossible small pleural effusion. Bibasilaropacities likely represent atelectasis. Hilarand mediastinal silhouettes areunremarkable. Calcifications of the aorticarch are noted</td><td rowspan=1 colspan=1>An endotracheal tube terminates 3.4 cmabove the carina.An orogastric tubeterminates within the stomach. Aright-sided PICC terminates at the mid tolower SVC. The heart size is normal. Thehilar and mediastinal contours are withinnormal limits. There is no pneumothoraxfocal consolidation, or pleural effusion.</td><td rowspan=1 colspan=1>There is a dual-lead pacemaker / ICD devicewith leads again terminating in the rightatrium and ventricle, respectively. Theheart is again enlarged. The aortic arch iscalcified. The cardiac, mediastinal and hilarcontours appear stable. There is a newdiffuse mild to moderate interstitialabnormalitymost consistent withcongestive heart failure. There is no definitepleural effusion or pneumothorax.</td></tr><tr><td rowspan=1 colspan=1>O3.·5</td><td rowspan=1 colspan=1>There are monitoring leads and externalwires overlying the chest. A tracheostomytube is visualized in the midline neck. Thecardiac silhouette is enlarged, consistentwith cardiomegaly. The mediastinalcontours appear widened, likelyaccentuated by the AP projection. Withinthe lungs, there are diffuse, bilateral,patchy airspace opacities that are mostprominent in the perihilar regions and lowerlung zones. The pulmonary vascularmarkings are indistinct. There isobscuration of the hemidiaphragms andcostophrenic angles bilaterally, consistentwith moderate pleural effusions. Nopneumothorax is identified. The visualizedosseous structures appear intact.</td><td rowspan=1 colspan=1>There has been interval intubation. Anendotracheal tube is visualized with its tipprojecting approximately 3 to 4 cm abovethe carina, in appropriate position. Thepreviously described right-sided PICC line isno longer visualized. In the lungs, there is adense opacity involving the left lower lungzone which obscures the lefthemidiaphragm and the left heart border.This finding represents a significantprogression compared to the prior study,which showed only slightly more markeddiffuse densities in this region. Theappearance is consistent with acombination of pleural effusion andconsolidation/atelectasis. Given the clinicalindication of an aspiration event, aspirationpneumonia is a primary consideration. Theright lung field is clear. The pulmonaryvāscular congestion and perivascular hazedescribed on the prior examination haveresolved. There is no evidence ofpneumothorax. The cardiac silhouette ispartially obscured by the left-sidedpathology but appears enlargedMediastinal contours are stable.</td><td rowspan=1 colspan=1>There is a left-sided cardiac rhythmmanagement device (pacemaker/ICD)withleads extending into the cardiac chambers.The cardiac silhouette is enlarged. Thereare bilateral lower lung zone opacities. Adense opacity at the right base obscuresthe right hemidiaphragm and right heartborder, consistent with a moderate rightpleural effusion and associatedcompressive atelectasis or consolidation.On the left, there are retrocardiac opacitiesand blunting of the left costophrenic angle,suggestive of a left pleural effusion and leftlower lobe atelectasis. Pulmonary vascularmarkings appear prominent, consistent withpulmonary venous congestion. Nopneumothorax is identified.</td></tr></table>

Fig. 5 Final-report comparison in three selected MIMIC-CXR cases. Each column presents the radiograph, reference report and reports generated by ARIVE, PriorRG and Qwen3.5. Blue and vermilion highlights indicate concordant and unsupported findings, respectively. In these examples, ARIVE captures the principal findings and key negative observations while avoiding several unsupported or overstated findings produced by the comparators.

## 2.5.2 Tracing findings and errors through MCoT trajectories

Report-level fidelity alone does not reveal how individual findings arise during the diagnostic process. We therefore examined two further cases (Fig. 6), each pairing regional image evidence with the corresponding intermediate interpretations, final report and reference report.

Case 1 illustrates a clinically concordant trajectory. ARIVE inspected the lung, pleural, cardiac and upper-abdominal regions and identified low lung volumes, clear lungs, normal heart size and dilated bowel loops; these interpretations were carried into the final report and closely matched the reference findings. Case 2 shows how reasoning–report consistency makes the origin of an unsupported finding traceable. The generated report described mild cardiomegaly, which was absent from the reference report. This claim first appeared in an earlier Model Claim step that characterized the heart as mildly enlarged, and the trajectory displayed the cardiac-silhouette region associated with it. A reviewer can therefore check the intermediate claim against the image evidence and follow its propagation into the final report.

Together, these cases illustrate how multimodal trajectories make reported findings inspectable by linking regional visual evidence, intermediate interpretations and final report statements. The same linkage allows an unsupported claim to be traced from its first appearance in the reasoning to its inclusion in the final report.

## 3 Discussion

In this study, we developed ARIVE, an RRG framework that interleaves anatomical inspection with diagnostic interpretation before composing the final report. ARIVE achieved the highest scores across the clinical-eficacy measures on both MIMIC-CXR and the abnormality-centered MIMIC-ABN, with statistically significant diferences in most comparisons (Table 1 and Fig. 2c). Evaluation on the Chest ImaGenome Gold subset further supported its ability to select and localize anatomical regions during reasoning (Fig. 2g). In blinded grading, ARIVE’s mean report-quality score was numerically higher than PriorRG’s and similar to Qwen3.5’s (Fig. 3), and the radiologist rated its MCoT trajectories highly for diagnostic traceability. At the claim level, 95.8% of report claims were fully supported by the corresponding trajectories (Fig. 4a–c). With MCoT available, the radiologist identified errors in more reports (11 versus 8 of 100; Fig. 4d), providing preliminary evidence that trajectories can support report review. Together, these findings indicate that inspectable diagnostic reasoning can be built into report generation without sacrificing clinical eficacy metrics.

Among the baselines, the general-purpose and generalist medical MLLMs were strongest on observation-level accuracy as measured by CheXbert, whereas the taskspecific RRG models tended to achieve greater reference agreement in clinical entities and their relations; ARIVE led on both types of measure. Its advantage over Qwen3.5 was largest on the CheXbert measures dominated by common observations and smallest on the primary 14-observation macro $\mathrm { F _ { 1 } } .$ . ARIVE achieved these results with 4.85 billion parameters, far fewer than Qwen3.5 (397 billion total and 17 billion active parameters) and also used fewer parameters than MAIRA-2 and Libra. Generating both an MCoT trajectory and a final report took a mean of 1.13 s per sample on a single NVIDIA A100 GPU, comparable to the report-only inference times of MedGemma-1.5 and MAIRA-2 but longer than those of the smaller MLRG and PriorRG models (Supplementary Table S8). ARIVE thus combines strong clinical eficacy with inspectable reasoning at a moderate model scale and inference cost.

![](images/854c5f5cd2c28a912809a6acf249f5c318c2932747916c1a2c2ba8fd8f4c08f9.jpg)  
Fig. 6 Inspectable ARIVE trajectories in two selected MIMIC-CXR cases. Each case pairs the input radiograph and predicted anatomical boxes with the requested regions, MCoT trajectory, generated report and reference report. Case 1 shows a trajectory broadly concordant with the reference report. Case 2 traces an unsupported claim of mild cardiomegaly from intermediate reasoning into the final report.

The staged ablations help explain these gains. Text-only CoT made intermediate diagnostic statements explicit, but its gains on some CheXbert measures came with lower $\mathrm { F _ { 1 } }$ -SRR-BERT, RadGraph $\mathrm { F _ { 1 } }$ and GREEN on MIMIC-CXR (Table 2 and Fig. 2e,f). Thus, explicit intermediate statements alone did not consistently improve report performance. Interleaving anatomical inspection through MCoT improved most clinical-eficacy measures relative to text-only CoT, together with reasoning-stage observation accuracy and reasoning–report consistency (Tables 2 and 3). One possible explanation consistent with these results is that later steps of long textual reasoning rely increasingly on preceding text rather than on the image [21, 22], whereas regional inspection renews visual support as the trajectory evolves and thereby limits the propagation of unsupported assumptions [43]. Prior work has linked report findings to localized image evidence for grounding, verification or staged interpretation [11, 23, 24, 44–46]. ARIVE incorporates regional evidence iteratively into the generation sequence, where it conditions subsequent interpretation before report composition. MCoT alone, however, remained below report-only SFT on RadGraph $\mathrm { F _ { 1 } }$ and GREEN and these scores exceeded those of report-only SFT only after ACCO was added.

ACCO addressed a complementary limitation: the token-level SFT objective does not directly reward accurate grounding or clinically correct content, nor does it emphasize underrepresented observations and anatomical regions. Adding ACCO to MCoT-based training increased scores on every clinical-eficacy measure on both datasets (Table 2 and Fig. 2f), and its component ablations supported distinct contributions from the rewards (Table 3). Removing the anatomical-grounding reward impaired all evaluated outcomes, whereas removing the report-stage clinical reward $R _ { \mathrm { C X B } } ^ { \mathrm { a n s } }$ improved grounding yet caused the largest decline in final-report micro $\mathrm { F _ { 1 } }$ among the reward ablations. This divergence suggests that localization quality alone is an insuficient proxy for report accuracy. Inverse-frequency weighting involved a related trade-of, accepting a slight loss in overall grounding for better macro-averaged reasoning and report performance. These results support jointly optimizing anatomical grounding, reasoning– report consistency and final-report clinical fidelity, with frequency weighting to address imbalances in clinical observations and anatomical regions.

The radiologist’s grades add a complementary perspective. Although ARIVE exceeded Qwen3.5 on every automated clinical-eficacy measure, the two received similar mean grades (3.47 versus 3.51). One plausible reason is that reference-based metrics penalize observations absent from the reference, which Qwen3.5 reported more than three times as often as ARIVE (216 versus 66; Fig. 3d), while a radiologist reviewing the image may accept some of them, particularly because the reference reports were themselves imperfect (mean grade, 3.88). Automated metrics and expert grading therefore capture diferent aspects of report quality.

For report review, the trajectories make intermediate interpretations and their associated regional evidence available for inspection. With MCoT available, the radiologist identified three additional reports containing errors, while review time increased (Fig. 4d,e). This finding suggests a potential trade-of between error identification and review time. Inspectability, however, does not guarantee correctness. As Case 2 illustrates, an unsupported finding can propagate coherently from the trajectory into the report (Fig. 6), and a fluent rationale could lend such an incorrect finding unwarranted credibility. MCoT trajectories should therefore be treated as generated interpretations to be checked against image evidence, and ARIVE is intended to support rather than replace clinical judgment.

Several limitations define the scope of these findings. First, anatomical supervision was sparse for some regions: the upper and mid lung zones and apical regions together accounted for only approximately 2.8% of all bounding-box annotations (Supplementary Table S4), and grounding for these regions remained substantially weaker than for frequently annotated anatomy even after ACCO (Fig. 2g). Because the lung zones and apical regions provide parenchymal and pleural-boundary evidence relevant to Consolidation and Pneumothorax, respectively, weak grounding may contribute to ARIVE’s low CheXbert $\mathrm { F _ { 1 } }$ for these observations (Fig. 2a,d), although this explanation remains hypothetical. Second, the anatomical boxes are predefined regions rather than lesions, image slices or volumetric structures. Extending ARIVE beyond chest radiography will therefore require modality-specific spatial supervision, trajectory design and evaluation. Third, we used only frontal radiographs and did not explore integrating lateral views or other complementary imaging information into the diagnostic trajectory. Fourth, because the MCoT trajectories were reconstructed from radiology reports by a large language model, they may inherit omissions and institution-specific reporting practices from the source reports, as well as errors introduced during rewriting. They should therefore not be interpreted as faithful reconstructions of radiologist cognition. Finally, the clinical evaluation datasets derive from MIMIC-CXR, a single-institution resource, and generalization to other institutions and reporting styles remains to be established.

Within these limitations, ARIVE shows that anatomical inspection can be integrated into report generation to explicitly link regional visual evidence, intermediate interpretations and final findings. This approach may support more inspectable reporting, although its applicability beyond chest radiography requires evaluation. Future work will examine how models can learn to identify diagnostically relevant regions and predict their locations without bounding-box annotations during training, which would reduce reliance on modality-specific spatial annotations and facilitate extension to other imaging modalities.

## 4 Methods

## 4.1 Ethics statement

This study used de-identified chest radiographs and reports from MIMIC-CXR and its MIMIC-ABN subset [27], together with anatomical annotations from Chest ImaGenome [26]. The creation of MIMIC-CXR was reviewed by the Institutional Review Board of Beth Israel Deaconess Medical Center, which granted a waiver of informed consent and approved the data sharing initiative [25]. The present study was a secondary analysis of de-identified datasets and involved no new patient recruitment or collection of identifiable patient information. All data were handled in accordance with the data use agreements and licensing requirements of each dataset.

All models that processed these data, including GPT-OSS-120B, DeepSeek-V4.1- Flash and Qwen3.5-397B-A17B, were run on local infrastructure, and no data were sent to external online services.

## 4.2 Data sources

MIMIC-CXR [25] is a publicly available, de-identified chest radiograph dataset comprising 377,110 radiographs from more than 227,000 studies, each study accompanied by a free-text report. MIMIC-ABN [27] is an abnormality-centered subset of MIMIC-CXR that includes only abnormal studies with reports restricted to abnormality-related findings. Chest ImaGenome [26] provides anatomy-centered scene graphs for MIMIC-CXR, linking anatomical bounding boxes to clinical attributes. It includes automatically generated Silver annotations covering 29 anatomical regions and a manually annotated Gold subset.

Following prior work [12, 13, 38], we used only the Findings section as the reference report. For both datasets, we retained frontal anteroposterior and posteroanterior radiographs, excluded lateral and other non-frontal views, and followed the oficial data splits. To obtain the anatomical supervision required for MCoT construction, we further restricted each training split to studies represented in the Chest ImaGenome Silver subset. Dataset statistics after these selection criteria are reported in Supplementary Table S3, and CheXbert positive-label counts for each split are provided in Supplementary Table S5. Anatomical grounding was evaluated using the Chest ImaGenome Gold samples associated with each dataset. The MIMIC-CXR Gold subset comprised 636 samples (from 636 studies of 444 patients). The MIMIC-ABN Gold subset comprised 142 samples (from 142 studies of 120 patients). Because the Chest ImaGenome Gold subset does not provide bounding boxes for the spine or abdomen, these regions were excluded from the grounding evaluation.

## 4.3 Construction of the multimodal chain-of-thought dataset

We constructed the MCoT dataset to teach ARIVE an explicit diagnostic trajectory rather than only a direct image-to-report mapping. Starting from the filtered MIMIC-CXR training set, we generated 159,416 MCoT trajectories by integrating two complementary sources [26, 28]: radiologist-style reasoning traces, which provided diagnostic structure and interpretation order, and Chest ImaGenome Silver annotations, which provided bounding boxes for 29 anatomical regions.

We first used radiologist-style reasoning traces released by NV-Reason-CXR [28] as diagnostic templates. These traces organize chest radiograph interpretation into a standardized sequence comprising device inspection, airway evaluation, bilateral lung examination, mediastinal and hilar assessment, cardiac assessment, upper-abdominal inspection, bone review, and diagnostic summarization. This sequence provided a structure for reorganizing report findings into stepwise diagnostic interpretations. Using these templates, we prompted GPT-OSS-120B [47] to rewrite each source report as a textual CoT trajectory. The prompt instructed the model to preserve all diagnostic information while reorganizing the report content into a stepwise account of diagnostic reasoning.

Second, we incorporated image-specific anatomical bounding boxes from Chest ImaGenome [26] to convert textual CoT into MCoT. Chest ImaGenome provides bounding-box annotations for 29 anatomical regions in MIMIC-CXR radiographs. These image-specific annotations allowed us to insert the corresponding bounding boxes into clinically significant reasoning steps. For each textual reasoning trajectory, we prompted GPT-OSS-120B to insert anatomical grounding only for clinically significant steps whose referenced anatomy was mentioned in the source report and corresponded to one of the 29 predefined Chest ImaGenome anatomical regions. This selective grounding strategy restricted region requests to report-relevant anatomy and retained steps without applicable regional annotations as text-only reasoning. Each completed MCoT trajectory was paired with its corresponding final report, jointly supervising anatomical region selection, regional interpretation, and report composition within a unified generation sequence. The 159,416 training trajectories contained an average of 399.98 words of reasoning text and 5.75 region-grounding instances per trajectory (Supplementary Table S6). The prompts for textual CoT construction and anatomical bounding-box insertion are provided in Supplementary Tables S9 and S10.

## 4.4 Interleaved anatomical inspection and diagnostic reasoning

ARIVE augments the backbone MLLM with an anatomical visual module that encodes regional image evidence during reasoning (Fig. 7). The backbone visual encoder captures the radiograph globally, whereas the anatomical visual module encodes local image crops specified by bounding boxes. Whenever a bounding box is generated or otherwise provided, the corresponding anatomical region is cropped from the radiograph and transformed into a local visual representation. This representation is then inserted into the MLLM context before generation resumes, allowing subsequent interpretation to incorporate the original prompt, the global image representation, the accumulated diagnostic text, and the newly inspected regional evidence.

During inference, successive region requests establish an iterative sequence of regional inspection and interpretation. After receiving each local visual representation, the MLLM continues generating diagnostic text and either requests an additional anatomical region or completes the trajectory. Once the reasoning trajectory is complete, the model generates the final report conditioned on the accumulated global and regional visual evidence with the preceding diagnostic context. Repeated regional inspection makes localized visual evidence available during generation. Recording each region request and its associated interpretation allows reviewers to inspect the links between regional evidence, intermediate statements and the final report.

![](images/8d00389566eba541fa3e953c728cca13b91a61d415c19e3b46363b9a7ba7c218.jpg)  
Fig. 7 Interleaved anatomical inspection and diagnostic reasoning in ARIVE. During generation, ARIVE inspects selected anatomical regions and inserts their visual representations into the model context to guide subsequent reasoning.

## 4.5 Model training

## 4.5.1 Supervised fine-tuning

We first trained ARIVE on the MCoT dataset with SFT. Each training example is represented by the tuple $( X , R , Y )$ , where $X = ( \mathbb { Z } , P )$ comprises the input radiographs and textual prompt. The ordered image tuple is $\bar { \mathcal { T } } = ( I ^ { \mathrm { p r i o r } } , I ^ { \mathrm { c u r } } )$ when the most recent prior radiograph is available, and ${ \mathcal { T } } = \left( I ^ { \mathrm { c u r } } \right)$ otherwise. Here, $I ^ { \mathrm { p r i o r } }$ denotes the most recent prior radiograph, and $I ^ { \mathrm { c u r } }$ denotes the current radiograph. The prompt P contains the task instructions, the available Indication text, and the prior report when available. The structured MCoT trajectory R specifies region requests, diagnostic interpretations, and where local visual representations are inserted and Y denotes the final report.

We serialized R followed by Y into an interleaved multimodal sequence $S =$ $( s _ { 1 } , s _ { 2 } , \ldots , s _ { T } )$ , inserting encoded local visual representations at the specified positions. Each $s _ { t }$ represents a textual token or a local visual embedding, so a visual block may occupy multiple sequence positions. At position $t ,$ X provides the initial input context, and $\boldsymbol { S } _ { < t } = \left( \boldsymbol { s } _ { 1 } , \ldots , \boldsymbol { s } _ { t - 1 } \right)$ contains the preceding text and local visual embeddings.

We optimized the model with an autoregressive language-modeling objective over textual prediction targets. Let $m _ { t } = 1$ for textual tokens to be predicted and $m _ { t } = 0$ for externally encoded local visual embeddings. With $\mathcal { T } _ { \mathrm { t e x t } } = \{ t \in \{ 1 , . . . , T \} : m _ { t } = 1 \}$ , the SFT loss is

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { t \in \mathcal { T } _ { \mathrm { t e x t } } } \log p _ { \theta } ( s _ { t } \mid X , S _ { < t } ) .\tag{1}
$$

Local visual embeddings remain available in $S _ { < t }$ for subsequent predictions but are excluded from the prediction targets. The loss is therefore computed only at textual positions.

## 4.5.2 Anatomical and Clinical-Content Optimization

SFT trains ARIVE to generate interleaved MCoT trajectories, but its token-level objective does not directly reward anatomical grounding accuracy or agreement with target clinical content. ACCO uses group relative policy optimization (GRPO) [48] to optimize generated trajectories and reports through four reward components. To account for imbalanced observation and anatomical-region frequencies, inverse-frequency weights are applied to the bounding-box and CheXbert rewards, while the RadGraph evaluates entity–relation structure without frequency weighting.

Given $X = ( \mathbb { Z } , P )$ , the current policy samples textual tokens for a group of N responses $\{ S _ { i } \} _ { i = 1 } ^ { N }$ , each comprising an MCoT trajectory and a final report. During generation, the anatomical visual module inserts local visual embeddings in response to region requests, yielding the interleaved sequence $\boldsymbol { S } _ { i } = ( s _ { i , 1 } , \ldots , s _ { i , T _ { i } } )$ . These embeddings provide context for subsequent token sampling. For each sampled response $i ,$ we compute four rewards: anatomical grounding $R _ { i , \mathrm { B B O X } }$ , reasoning-stage CheXbert $R _ { i , \mathrm { C X B } } ^ { \mathrm { r e a } }$ , final-report CheXbert $R _ { i , \mathrm { C X B } } ^ { \mathrm { a n s } } .$ and final-report RadGraph $R _ { i , \mathrm { R G } }$ . Because their numerical scales difer, each component is normalized within the sampled group:

$$
\begin{array} { r l r } {  { \widetilde { R } _ { i , k } = \frac { R _ { i , k } - \mu _ { k } } { \sigma _ { k } + \epsilon _ { r } } , } } \\ & { } & { \mu _ { k } = \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } R _ { i , k } , } \\ & { } & { \sigma _ { k } = \sqrt { \frac { 1 } { N } \displaystyle \sum _ { i = 1 } ^ { N } ( R _ { i , k } - \mu _ { k } ) ^ { 2 } } , } \end{array}\tag{2}
$$

where $k \in \{ \mathrm { B B O X } , \mathrm { C X B ^ { r e a } , C X B ^ { a n s } }$ , RG} and $\epsilon _ { r }$ is a small constant for numerical stability.

The normalized components are combined into a sequence-level advantage:

$$
A _ { i } = \frac { 1 } { 4 } \left( \widetilde { R } _ { i , \mathrm { B B O X } } + \widetilde { R } _ { i , \mathrm { C X B } } ^ { \mathrm { r e a } } + \widetilde { R } _ { i , \mathrm { C X B } } ^ { \mathrm { a n s } } + \widetilde { R } _ { i , \mathrm { R G } } \right) .\tag{3}
$$

At the token level, ACCO follows GRPO and optimizes a clipped policy objective with Kullback–Leibler (KL) regularization. For response i, let $\mathcal { T } _ { i , \mathrm { t e x t } } = \{ t \in \{ 1 , . . . , T _ { i } \}$ $m _ { i , t } = 1 \}$ denote the textual prediction positions, using the same mask definition as in SFT. The objective is

$$
\begin{array} { r l } {  { \mathcal { J } _ { \mathrm { A C C O } } ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { M _ { i } } \sum _ { t \in \mathcal { T } _ { i , \mathrm { t e x t } } } \operatorname* { m i n } ( r _ { i , t } ( \theta ) A _ { i } , \overline { { r } } _ { i , t } ( \theta ) A _ { i } ) } } \\ & { - \eta D _ { \mathrm { K L } } \big ( \pi _ { \theta } ( \cdot \mid X ) \big \| \pi _ { \mathrm { r e f } } ( \cdot \mid X ) \big ) , } \end{array}\tag{4}
$$

with the probability ratios defined for $t \in \mathcal { T } _ { i , \mathrm { t e x t } }$ as

$$
\begin{array} { r l } & { r _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( s _ { i , t } \mid X , S _ { i , < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( s _ { i , t } \mid X , S _ { i , < t } \right) } , } \\ & { \overline { { r } } _ { i , t } ( \theta ) = \mathrm { c l i p } ( r _ { i , t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) . } \end{array}\tag{5}
$$

Here, $\begin{array} { r } { M _ { i } = \left| \mathcal { T } _ { i , \mathrm { t e x t } } \right| = \sum _ { t = 1 } ^ { T _ { i } } m _ { i , t } } \end{array}$ is the number of textual prediction targets. The terms $r _ { i , t } ( \boldsymbol { \theta } )$ and $\overline { { r } } _ { i , t } ( \theta )$ denote the token-level probability ratio and its clipped value, respectively. The clipping threshold is $\epsilon ,$ and η controls KL regularization toward the reference policy $\pi _ { \mathrm { r e f } }$ . As in SFT, local visual embeddings contribute to the conditioning context $S _ { i , < t }$ but are excluded from the token-level prediction terms.

## Weighted bounding-box reward.

This reward evaluates the selection and localization of the anatomical regions specified by the target MCoT trajectory. For each anatomical category $c ,$ its empirical training frequency is

$$
p _ { c } = { \frac { n _ { c } } { \sum _ { j } n _ { j } } } ,\tag{6}
$$

where $n _ { c }$ is the number of training annotations for category c. To reduce the influence of frequently annotated regions, we used the normalized inverse-frequency weight

$$
w _ { c } = \frac { p _ { c } ^ { - \alpha _ { \mathrm { r e g } } } } { \frac { 1 } { | \mathcal { C } | } \sum _ { j \in \mathcal { C } } p _ { j } ^ { - \alpha _ { \mathrm { r e g } } } } ,\tag{7}
$$

where C is the set of anatomical categories and $\alpha _ { \mathrm { r e g } }$ controls the strength of region reweighting.

Let G and $\mathcal { P }$ denote the ground-truth and predicted region sets. For each $c \in { \mathcal { G } }$ the predicted box $\hat { b } _ { c }$ is matched to the ground-truth box $b _ { c }$ from the same category. If category c is not predicted, its intersection over union (IoU) is set to zero. The reward is

$$
R _ { \mathrm { B B O X } } = \frac { \sum _ { c \in \mathcal { G } } w _ { c } \ \mathrm { I o U } ( b _ { c } , \hat { b } _ { c } ) } { \sum _ { c \in \mathcal { G } } w _ { c } } - \lambda \operatorname* { m a x } ( 0 , | \mathcal { P } | - | \mathcal { G } | ) ,\tag{8}
$$

where λ penalizes predictions beyond the ground-truth count. Low overlap, missing categories and excess requests reduce the reward, while $w _ { c }$ increases the contribution of underrepresented regions.

## Weighted reasoning CheXbert reward.

This reward evaluates observation-level correctness in intermediate diagnostic text. A CheXbert-style classifier adapted to reasoning trajectories extracts 14 observation labels [29]. Its outputs for generated reasoning are denoted by yˆ, and y denotes the target labels.

Let O be the set of CheXbert observations and $n _ { o } ^ { + }$ the number of positive training labels for observation o. Its positive-label frequency is $q _ { o } = n _ { o } ^ { + } / \sum _ { j \in \mathcal { O } } n _ { j } ^ { + }$ . We obtain $v _ { o }$ from Eq. (7) by substituting $q _ { o } , \mathcal { O }$ and $\alpha _ { \mathrm { o b s } }$ for $p _ { c } , \mathcal { C }$ and $\alpha _ { \mathrm { r e g } } .$ , respectively. The weighted counts and corresponding $\mathrm { F _ { 1 } }$ reward are

$$
\begin{array} { r l } & { \mathrm { T P } _ { w } = \displaystyle \sum _ { o \in \mathcal { O } } v _ { o } { \bf 1 } [ \hat { y } _ { o } = 1 \wedge y _ { o } = 1 ] , } \\ & { \mathrm { F P } _ { w } = \displaystyle \sum _ { o \in \mathcal { O } } v _ { o } { \bf 1 } [ \hat { y } _ { o } = 1 \wedge y _ { o } = 0 ] , } \\ & { \mathrm { F N } _ { w } = \displaystyle \sum _ { o \in \mathcal { O } } v _ { o } { \bf 1 } [ \hat { y } _ { o } = 0 \wedge y _ { o } = 1 ] , } \\ & { R _ { \mathrm { C X B } } ^ { \mathrm { r e a } } = \displaystyle \frac { 2 \mathrm { T P } _ { w } } { 2 \mathrm { T P } _ { w } + \mathrm { F P } _ { w } + \mathrm { F N } _ { w } } . } \end{array}\tag{9}
$$

Here, TP, FP and FN denote true positives, false positives and false negatives, respectively, and the subscript w indicates weighted counts. Because $v _ { o }$ increases as positive-label frequency decreases, observations with fewer positive examples contribute more strongly.

## Weighted final-report CheXbert reward.

This reward applies the same observation weights and weighted $\mathrm { F _ { 1 } }$ to CheXbert labels extracted from the generated and reference reports. Substituting these report-level label vectors into Eq. (9) gives $R _ { \mathrm { C X B } } ^ { \mathrm { a n s } }$ , which emphasizes clinically correct final-report observations, including underrepresented categories.

## RadGraph reward.

The CheXbert rewards assess clinical eficacy over a fixed set of observations. We added a RadGraph reward to encourage the final report to preserve clinical entities and their relationships in the reference report. For each generated report, we computed the Simple, Partial and Complete RadGraph scores [49] against its reference. Simple matches entities by text and label, whereas Partial additionally checks for outgoing relations. Complete further checks relation types and target entities. We define $R _ { \mathrm { R G } }$ as the mean of these three scores.

$$
R _ { \mathrm { R G } } = \frac { 1 } { 3 } \left( \mathrm { R G } _ { \mathrm { s i m p l e } } + \mathrm { R G } _ { \mathrm { p a r t i a l } } + \mathrm { R G } _ { \mathrm { c o m p l e t e } } \right) .\tag{10}
$$

Together, the four rewards supervise anatomical grounding, intermediate and final-report observation content, and final-report entity–relation structure. Frequency weighting increases contributions from underrepresented anatomical regions and clinical observations.

## 4.6 Implementation details

Following recent prior-aware RRG studies [12–14], we incorporated available patientspecific historical information into the model input. For each examination, the model received the current radiograph together with the most recent prior radiograph and its corresponding report, when available. For the current examination’s clinical context, we retain only the Indication section. This setting enables the model to leverage longitudinal imaging history and relevant clinical cues, thereby more closely approximating the information available during real-world radiological interpretation. The reportgeneration prompts for MIMIC-CXR and MIMIC-ABN are provided in Supplementary Tables S11 and S12, respectively. Considering both model generalization and training cost, we used Qwen3-VL-Instruct-4B [50] as the MLLM backbone. The anatomical visual module mirrored the architecture of the backbone visual module, including its vision encoder and visual merger, and was initialized with the corresponding backbone parameters. All input images were resized to $2 2 4 \times 2 2 4$ pixels (50,176 pixels).

During supervised fine-tuning (SFT), we jointly optimized the backbone visual module, anatomical visual module, and LLM using AdamW with a learning rate of $1 \times 1 0 ^ { - 5 }$ . Training used a batch size of 64, cosine learning-rate decay, and a warmup ratio of 0.03. SFT on MIMIC-CXR was performed for six epochs. The MIMIC-ABN model was subsequently initialized from the resulting MIMIC-CXR checkpoint and further adapted for three epochs on the MIMIC-ABN training split. During ACCO, we updated only the LLM for one epoch using AdamW with a learning rate of $1 \times 1 0 ^ { - 6 }$ a batch size of 32, and a group size of five. The KL-divergence coeficient was set to 0.01. The bounding-box penalty coeficient λ was set to 0.05. We used inversefrequency exponents of $\alpha _ { \mathrm { r e g } } = 0 . 2 5$ for anatomical regions and $\alpha _ { \mathrm { o b s } } = 0 . 5$ for CheXbert observations. These hyperparameters were selected empirically within the available computational budget to maintain comparable numerical contributions across the diferent reward sources. Inference was accelerated using vLLM 0.19.0 [51].

## 4.7 Comparison baselines and protocols

We compared ARIVE with three groups of baseline models: a general-purpose MLLM, a generalist medical MLLM, and task-specific RRG models. Qwen3.5-397B-A17B [36] is an open-weight MLLM with a sparse mixture-of-experts architecture comprising 397 billion total parameters and 17 billion active parameters, evaluated here in reasoning mode. MedGemma-1.5 [37] is a generalist medical MLLM supporting medical image and text understanding, including localization and longitudinal chest X-ray analysis.

The task-specific RRG baselines comprised MAIRA-2, Libra, MLRG, and PriorRG. MAIRA-2 [23] couples a frozen RAD-DINO image encoder with a Vicuna-7B language model through a four-layer multilayer perceptron adapter. Libra [13] connects a RAD-DINO image encoder to Meditron-7B through a Temporal Alignment Connector that integrates visual features from current and prior radiographs. MLRG [38] combines RAD-DINO and CXR-BERT encoders with multi-view longitudinal and multimodal fusion networks to condition a DistilGPT2 report decoder. PriorRG [12] uses RAD-DINO and CXR-BERT encoders with a DistilGPT2 decoder, progressively integrating clinical context, spatiotemporal features and hierarchical visual features through coarse-to-fine fusion.

Whenever available, we used publicly released checkpoints or oficial model-access configurations. All models received the same case-level input content as ARIVE and were evaluated on the same test sets using the same metric implementations. For the task-specific baselines MAIRA-2 and Libra, which were not originally trained on MIMIC-ABN, we followed their published training protocols by initializing each model from its publicly released checkpoint and further training it on the MIMIC-ABN training split before evaluation.

We also compared model size and inference time on the MIMIC-CXR test set (Supplementary Table S8). With 4.85 billion parameters, ARIVE generated an MCoT trajectory and a final report in a mean of 1.13 s per sample on a single NVIDIA A100 GPU, close to the report-only inference times of MedGemma-1.5 and MAIRA-2. ARIVE used fewer parameters than MAIRA-2 and Libra, whereas the smaller MLRG and PriorRG models were faster.

## 4.8 Evaluation metrics

Clinical-eficacy metrics included macro- and micro-averaged CheXbert $\mathrm { F _ { 1 } }$ [29] over 14 CheXpert observations [52], together with five-observation CheXbert $\mathrm { F _ { 1 } }$ over Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Efusion. We additionally reported $\mathrm { F _ { 1 } }$ -SRR-BERT [30] over 55 radiology labels. RadGraph F<sub>1</sub> [31] evaluated the clinical entity–relation structure of generated reports. Entities were matched by text and label, and relations by source and target entities, type and direction. The score for each generated–reference report pair was the mean of the entity and relation $\mathrm { F _ { 1 } }$ values:

$$
\mathrm { R a d G r a p h ~ F _ { 1 } } = \frac { 1 } { 2 } \left( \mathrm { F _ { 1 , e n t i t y } + F _ { 1 , r e l a t i o n } } \right) .\tag{11}
$$

This evaluation score difers from the training reward $R _ { \mathrm { R G } }$ , which averages the RadGraph Simple, Partial and Complete scores (Eq. (10)).

GREEN [32] evaluated matched findings and clinically significant errors. For each generated–reference report pair, the score was

$$
\mathrm { G R E E N } = \frac { N _ { \mathrm { m a t c h } } } { N _ { \mathrm { m a t c h } } + N _ { \mathrm { e r r o r } } } ,\tag{12}
$$

where $N _ { \mathrm { m a t c h } }$ is the number of matched findings and $N _ { \mathrm { e r r o r } }$ is the total number of clinically significant errors across GREEN’s six error categories. The score was set to zero when $N _ { \mathrm { m a t c h } } = 0$ . GREEN ranges from 0 to 1, with higher values indicating better report quality. We did not assess GREEN on MIMIC-ABN because its reports were restricted to abnormality-related findings. Lexical similarity was evaluated using BLEU-4 [33], METEOR [34], and ROUGE-L [35]. These metrics capture higher-order n-gram precision, unigram-level semantic correspondence, and longest-common-subsequence overlap, respectively.

For MCoT trajectories, region-selection $\mathrm { F _ { 1 } }$ evaluated the selection of reference anatomical regions for local inspection, whereas mean intersection over union (mIoU) measured localization overlap averaged across all ground-truth anatomical regions. Reasoning-stage CheXbert evaluated the clinical content expressed during intermediate reasoning, and reasoning-report consistency of CheXbert quantified agreement between the clinical content expressed in the reasoning trajectory and those in the final report. Training details for reasoning CheXbert are described below, and held-out performance is reported in Supplementary Table S7. Following the training protocol of the original CheXbert study [29], we trained reasoning CheXbert on the MCoT training data. We randomly partitioned the data at the patient level into 70% training and 30% held-out subsets, ensuring that all examples from the same patient remained in a single subset. We initialized the classifier with CheXbert parameters and fine-tuned it for 8 epochs to adapt label extraction from conventional radiology reports to reasoning-style diagnostic trajectories. For the held-out evaluation of this classifier, uncertain labels were treated as positive.

## 4.9 Statistical analysis

We used one-sided nonparametric tests to evaluate the directional hypothesis that ARIVE achieved higher scores than the baseline models. For corpus-level metrics, including CheXbert F , F -SRR-BERT, and BLEU-4, we used paired permutation tests with resampling clustered at the patient level. For report-level metrics, including METEOR, ROUGE-L, GREEN, and RadGraph F , we used paired Wilcoxon signed-rank tests. Two-sided 95% confidence intervals (CIs) were reported to describe estimation uncertainty. For each model, CIs were estimated from 10,000 patient-clustered bootstrap resamples. For pairwise model comparisons, bootstrap resampling was paired by applying the same patient-level resample to both models. CI bounds were defined by the 2.5th and 97.5th percentiles of the resulting bootstrap distribution.

## Data availability

The MIMIC-CXR v2.1.0 (https://doi.org/10.13026/4jqj-jw95) and Chest ImaGenome v1.0.0 (https://doi.org/10.13026/wv01-y230) datasets are available on PhysioNet [53] after a credentialing process. MIMIC-ABN is an abnormality-centered subset of MIMIC-CXR, with the preprocessing procedure adopted in this study detailed in https://doi.org/10.18653/v1/2020.findings-emnlp.176.

## Code availability

The code for MCoT construction and ARIVE training and inference is available at https://github.com/DUTIR-BioNLP/ARIVE.

## References

[1] Tanno, R. et al. Collaboration between clinicians and vision–language models in radiology report generation. Nature Medicine 31, 599–608 (2025).

[2] Jin, Q. et al. Hidden flaws behind expert-level accuracy of multimodal gpt-4 vision in medicine. NPJ Digital Medicine 7, 190 (2024).

[3] Chen, H., Gomez, C., Huang, C.-M. & Unberath, M. Explainable medical imaging ai needs human-centered design: guidelines and evidence from a systematic review. NPJ digital medicine 5, 156 (2022).

[4] Kim, C., Gadgil, S. U. & Lee, S.-I. Transparency of medical artificial intelligence systems. Nature reviews bioengineering 4, 11–29 (2026).

[5] Chen, Z., Song, Y., Chang, T.-H. & Wan, X. Generating radiology reports via memory-driven transformer. Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), 1439–1449 (2020).

[6] Zambrano Chaves, J. M. et al. A clinically accessible small multimodal radiology model and evaluation metric for chest x-ray findings. Nature Communications 16, 3108 (2025).

[7] Li, G., Huang, C., Zhou, X., Ji, D. & Zhang, H. Report is a mixture of topics: Topic-guided radiology report generation. Medical Image Analysis 103, 103586 (2025).

[8] Chen, X. et al. A deep learning based automatic report generator for retinal optical coherence tomography images. NPJ Digital Medicine 8, 618 (2025).

[9] Park, Y. W. et al. A robust vision language model for molecular status prediction and radiology report generation in adult-type difuse gliomas. npj Digital Medicine (2026).

[10] Liu, F. et al. A multimodal multidomain multilingual medical foundation model for zero shot clinical diagnosis. npj Digital Medicine 8, 86 (2025).

[11] Wu, L. et al. A universal foundation model for grounded biomedical image interpretation. Nature Communications (2026).

[12] Liu, K. et al. Priorrg: Prior-guided contrastive pre-training and coarse-to-fine decoding for chest x-ray report generation. Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, 7206–7214 (2026).

[13] Zhang, X., Meng, Z., Lever, J. & Ho, E. S. Libra: Leveraging temporal images for biomedical radiology analysis. Findings of the Association for Computational Linguistics: ACL 2025, 17275–17303 (2025).

[14] Hou, W. et al. Radar: Enhancing radiology report generation with supplementary knowledge injection. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 26366–26381 (2025).

[15] Li, M. et al. Cross-modal clinical graph transformer for ophthalmic report generation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20624–20633 (IEEE, 2022).

[16] Li, M. et al. Dynamic graph enhanced contrastive learning for chest x-ray report generation. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3334–3343 (IEEE, 2023).

[17] Huang, C., Li, G., Zhou, X., Ren, Y. & Zhang, H. Ddgip: Radiology report generation through disease description graph and informed prompting. Findings of the Association for Computational Linguistics: NAACL 2025, 3884–3894 (2025).

[18] Wei, J. et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824–24837 (2022).

[19] Liu, J., Wang, Y., Du, J., Zhou, J. T. & Liu, Z. Medcot: Medical chain of thought via hierarchical expert. Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 17371–17389 (2024).

[20] Mei, X. et al. Fir-rad: Fine-grained reinforcement with structured reasoning for chest x-ray report generation. IEEE Transactions on Medical Imaging (2026).

[21] Sun, H.-L., Sun, Z., Peng, H. & Ye, H.-J. Mitigating visual forgetting via takealong visual conditioning for multi-modal long cot reasoning. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 5158–5171 (2025).

[22] Cheng, Z. et al. Visual thoughts: A unified perspective of understanding multimodal chain-of-thought. Advances in Neural Information Processing Systems 38, 96084– 96112 (2026).

[23] Bannur, S. et al. Maira-2: Grounded radiology report generation. Tech. Rep. MSR-TR-2024-18, Microsoft (2024). URL https://www.microsoft.com/en-us/research/ publication/maira-2-grounded-radiology-report-generation/.

[24] Jin, K. et al. Grounded report generation for enhancing ophthalmic ultrasound interpretation using vision-language segmentation models. npj Digital Medicine 9, 99 (2026).

[25] Johnson, A. E. et al. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data 6, 317 (2019).

[26] Wu, J. T. et al. Chest imagenome dataset for clinical reasoning. (eds Vanschoren, J. & Yeung, S.) Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, Vol. 1 (2021). URL https://datasets-benchmarks-proceedings.neurips.cc/paper files/ paper/2021/file/17e62166fc8586dfa4d1bc0e1742c08b-Paper-round2.pdf.

[27] Ni, J., Hsu, C.-N., Gentili, A. & McAuley, J. Learning visual-semantic embeddings for reporting abnormal findings on chest x-rays. Findings of the association for computational linguistics: EMNLP 2020, 1954–1960 (2020).

[28] Myronenko, A. et al. Reasoning visual language model for chest x-ray analysis. arXiv preprint arXiv:2510.23968 (2025).

[29] Smit, A. et al. Combining automatic labelers and expert annotations for accurate radiology report labeling using bert. Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), 1500–1519 (2020).

[30] Delbrouck, J.-B. et al. Automated structured radiology report generation. Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 26813–26829 (2025).

[31] Jain, S. et al. Radgraph: Extracting clinical entities and relations from radiology reports. (eds Vanschoren, J. & Yeung, S.) Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, Vol. 1 (2021). URL https://datasets-benchmarks-proceedings.neurips.cc/paper files/ paper/2021/file/c8fe9a587b126f152ed3d89a146b445-Paper-round1.pdf.

[32] Ostmeier, S. et al. Green: Generative radiology report evaluation and error notation. Findings of the association for computational linguistics: EMNLP 2024, 374–390 (2024).

[33] Papineni, K., Roukos, S., Ward, T. & Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. Proceedings of the 40th annual meeting of the Association for Computational Linguistics, 311–318 (2002).

[34] Banerjee, S. & Lavie, A. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. Proceedings of the acl workshop

on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 65–72 (2005).

[35] Lin, C.-Y. Rouge: A package for automatic evaluation of summaries. Text summarization branches out, 74–81 (2004).

[36] Qwen Team. Qwen3.5: Towards native multimodal agents (2026). URL https: //qwen.ai/blog?id=qwen3.5.

[37] Sellergren, A. et al. Medgemma 1.5 technical report. arXiv preprint arXiv:2604.05081 (2026).

[38] Liu, K. et al. Enhanced contrastive learning with multi-view longitudinal data for chest x-ray report generation. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10348–10359 (IEEE, 2025).

[39] Liu, G. et al. Clinically accurate chest x-ray report generation. (eds Doshi-Velez, F. et al.) Proceedings of the 4th Machine Learning for Healthcare Conference, Vol. 106 of Proceedings of Machine Learning Research, 249–269 (PMLR, 2019). URL https://proceedings.mlr.press/v106/liu19a.html.

[40] Babar, Z., van Laarhoven, T., Zanzotto, F. M. & Marchiori, E. Evaluating diagnostic content of ai-generated radiology reports of chest x-rays. Artificial Intelligence in Medicine 116, 102075 (2021). URL https://www.sciencedirect. com/science/article/pii/S0933365721000683.

[41] Yu, F. et al. Evaluating progress in automatic chest x-ray radiology report generation. Patterns 4, 100802 (2023). URL https://www.sciencedirect.com/ science/article/pii/S2666389923001575.

[42] DeepSeek-AI. Deepseek-v4.1-flash: Pushing the limits of kv cache compression (2026). URL https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/ main/DeepSeek V41 Tech Report.pdf.

[43] Jin, Z. et al. Look light, think heavy: What multimodal chain-of-thought reasoning can and cannot do. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 8567–8608 (2026).

[44] Zhang, W., Chandra, S. S. & Nicolson, A. Generalised medical phrase grounding. IEEE Transactions on Medical Imaging (2026).

[45] Hashmi, A. U. R., Saeed, N. & Lippert, C. Anatomix, an anatomy-aware grounded multimodal large language model for chest x-ray interpretation. 2026 IEEE/CVF Conference on Computer Vision and Pattern Recognition- FINDINGS Track (CVPRF), 6012–6021 (2026).

[46] Chen, W. et al. Interpretable agentic ai system with localized reasoning for radiology. npj Digital Medicine (2026).

[47] OpenAI. gpt-oss-120b & gpt-oss-20b model card (2025). URL https://arxiv.org/ abs/2508.10925. arXiv:2508.10925.

[48] Guo, D. et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature 645, 633–638 (2025).

[49] Delbrouck, J.-B. et al. Improving the factual correctness of radiology report generation with semantic rewards. Findings of the Association for Computational Linguistics: EMNLP 2022, 4348–4360 (2022).

[50] Bai, S. et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025).

[51] Kwon, W. et al. Eficient memory management for large language model serving with pagedattention. Proceedings of the 29th symposium on operating systems principles, 611–626 (2023).

[52] Irvin, J. et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. Proceedings of the AAAI conference on artificial intelligence, Vol. 33, 590–597 (2019).

[53] Pollard, T. et al. PhysioNet as a global platform for biomedical research. Nature Health 1, 792–795 (2026). URL https://doi.org/10.1038/s44360-026-00096-z.

## Funding

This research was supported by the National Natural Science Foundation of China (No. 62676065) and the Fundamental Research Funds for the Central Universities (No. DUT25YG108).

## Author contributions

C.H. and L.L. conceived this work. C.H. designed the framework and conducted the experiments. Z.W. and Y.L. contributed to the data acquisition and provided suggestions on the framework. R.L. performed the clinical evaluation. L.L. polished the manuscript and provided suggestions on the experiments. Y.S., J.W., H.L. and L.L. supervised the project. All authors discussed the results and contributed to the final manuscript.

## Competing interests

The authors declare no competing interests.

# Appendix: Supplementary Information

Interleaving anatomical inspection and diagnostic reasoning for explainable radiology report generation

## S1 Natural language generation results

Supplementary Table S1 Natural language generation performance of ARIVE and comparator models on MIMIC-CXR and MIMIC-ABN. Values are reported with 95% CIs in parentheses. Bold marks the highest value for each metric within each dataset. Stars indicate ARIVE’s superiority in one-sided comparisons using patient-clustered paired permutation tests for BLEU-4 and paired Wilcoxon signed-rank tests for METEOR and ROUGE-L. <sup>∗</sup>p < 0.05; <sup>∗∗</sup>p < 0.01.
<table><tr><td>Dataset</td><td>Model</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td></tr><tr><td></td><td>Qwen3.5-397B-A17B</td><td>0.062 ** (0.058, 0.065)</td><td>0.344 (0.338, 0.349)</td><td>0.202 (0.198, 0.206)</td></tr><tr><td></td><td>MedGemma-1.5</td><td>0.082** (0.077, 0.087)</td><td>0.311 ** (0.305, 0.318)</td><td>0.230** (0.225, 0.235)</td></tr><tr><td></td><td>MAIRA-2</td><td>0.098 ** (0.093, 0.103)</td><td>0.268 ** (0.262, 0.275)</td><td>0.238 ** (0.232, 0.243)</td></tr><tr><td>MIMIC-CXR Libra</td><td></td><td>0.133* (0.126, 0.140)</td><td>0.327 (0.320, 0.334)</td><td>0.253 ** (0.247, 0.260)</td></tr><tr><td></td><td>MLRG</td><td>0.150 (0.143, 0.157)</td><td>0.324 ** (0.316, 0.332)</td><td>0.294 (0.286, 0.301)</td></tr><tr><td></td><td>PriorRG</td><td>0.175 (0.168, 0.182)</td><td>0.355 (0.347, 0.363)</td><td>0.318 (0.310, 0.325)</td></tr><tr><td></td><td>ARIVE (ours)</td><td>0.139 (0.133, 0.145)</td><td>0.328 (0.321, 0.335)</td><td>0.284 (0.279, 0.290)</td></tr><tr><td></td><td>Qwen3.5-397B-A17B</td><td>0.055 ** (0.047, 0.062)</td><td>0.285 (0.272, 0.297)</td><td>0.180 (0.170, 0.188)</td></tr><tr><td></td><td>MedGemma-1.5</td><td>0.052 ** (0.043, 0.061)</td><td>0.256 (0.243, 0.269)</td><td>0.186** (0.175, 0.196)</td></tr><tr><td></td><td>MAIRA-2</td><td>0.067 ** (0.058, 0.076)</td><td>0.214 (0.202, 0.228)</td><td>0.190 ** (0.179, 0.202)</td></tr><tr><td>MIMIC-ABN Libra</td><td></td><td>0.056** (0.049, 0.063)</td><td>0.279 (0.267, 0.290)</td><td>0.181** (0.171, 0.192)</td></tr><tr><td></td><td>MLRG</td><td>0.101 (0.084, 0.121)</td><td>0.261 (0.242, 0.282)</td><td>0.228 (0.210, 0.248)</td></tr><tr><td></td><td>PriorRG</td><td>0.112 (0.097, 0.129)</td><td>0.264 (0.244, 0.284)</td><td>0.228 (0.212, 0.244)</td></tr><tr><td></td><td>ARIVE (ours)</td><td>0.085 (0.074, 0.096)</td><td>0.253 (0.239, 0.268)</td><td>0.224 (0.211, 0.236)</td></tr></table>

Supplementary Table S2 Natural language generation performance across training settings on MIMIC-CXR and MIMIC-ABN. SFT uses report-only supervision; SFT w/ CoT and SFT w/ MCoT add text-only and region-interleaved diagnostic trajectories, respectively. ARIVE denotes SFT w/ MCoT followed by Anatomical and Clinical-Content Optimization (ACCO). Values are reported with 95% CIs in parentheses. Bold marks the highest value for each metric within each dataset. Stars indicate ARIVE’s superiority under the tests specified in Supplementary Table S1. <sup>∗</sup>p < 0.05; <sup>∗∗</sup>p < 0.01.
<table><tr><td>Dataset</td><td>Training setting</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE-L</td></tr><tr><td rowspan="3">MIMIC-CXR</td><td>SFT</td><td>0.149 (0.143, 0.156)</td><td>0.330 (0.325, 0.339)</td><td>0.283* (0.277, 0.290)</td></tr><tr><td>SFT w/ CoT</td><td>0.112** (0.107, 0.118)</td><td>0.285** (0.278, 0.292)</td><td>0.261 ** (0.256, 0.268)</td></tr><tr><td>SFT w/ MCoT</td><td>0.123** (0.115, 0.127)</td><td>0.303 ** (0.295, 0.309)</td><td>0.267 ** (0.260, 0.272)</td></tr><tr><td></td><td>ARIVE</td><td>0.139 (0.133, 0.145)</td><td>0.328 (0.321, 0.335)</td><td>0.284 (0.279, 0.290)</td></tr><tr><td></td><td>SFT</td><td>0.084 (0.072, 0.098)</td><td>0.253 (0.237, 0.269)</td><td>0.228 (0.215, 0.244)</td></tr><tr><td rowspan="2">MIMIC-ABN</td><td>SFT w/ CoT</td><td>0.064 ** (0.063, 0.083)</td><td>0.222 ** (0.217, 0.248)</td><td>0.204* (0.199, 0.224)</td></tr><tr><td>SFT w/ MCoT</td><td>0.075 ** (0.060, 0.079)</td><td>0.239 ** (0.214, 0.242)</td><td>0.218* (0.195, 0.220)</td></tr><tr><td></td><td>ARIVE</td><td>0.085 (0.074, 0.096)</td><td>0.253 (0.239, 0.268)</td><td>0.224 (0.211, 0.236)</td></tr></table>

## S2 Dataset, annotation and trajectory statistics

Supplementary Table S3 Sample, study and patient counts after frontal-view filtering and restriction of the training splits to Chest ImaGenome Silver overlap.
<table><tr><td>Dataset</td><td>Statistic</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td rowspan="3">MIMIC-CXR</td><td>Samples</td><td>159,416</td><td>1,286</td><td>2,463</td></tr><tr><td>Studies</td><td>142,946</td><td>1,151</td><td>2,211</td></tr><tr><td>Patients</td><td>57,729</td><td>448</td><td>285</td></tr><tr><td></td><td>Samples</td><td>34,259</td><td>302</td><td>432</td></tr><tr><td rowspan="2">MIMIC-ABN</td><td>Studies</td><td>34,152</td><td>263</td><td>378</td></tr><tr><td>Patients</td><td>23,039</td><td>181</td><td>134</td></tr></table>

Supplementary Table S4 Anatomical bounding-box annotation counts in the training splits and Chest ImaGenome Gold subsets of MIMIC-CXR and MIMIC-ABN.
<table><tr><td rowspan="2">Anatomical Region</td><td colspan="2">MIMIC-CXR</td><td colspan="2">MIMIC-ABN</td></tr><tr><td>Train</td><td>Gold subset</td><td>Train</td><td>Gold subset</td></tr><tr><td>Right lung</td><td>152,734</td><td>546</td><td>33,492</td><td>116</td></tr><tr><td>Right upper lung zone</td><td>4,757</td><td>46</td><td>1,015</td><td>12</td></tr><tr><td>Right mid lung zone</td><td>7,654</td><td>56</td><td>2,072</td><td>8</td></tr><tr><td>Right lower lung zone</td><td>31,442</td><td>174</td><td>6,706</td><td>35</td></tr><tr><td>Right hilar structures</td><td>48,140</td><td>343</td><td>11,108</td><td>81</td></tr><tr><td>Right apical zone</td><td>4,583</td><td>65</td><td>951</td><td>23</td></tr><tr><td>Right costophrenic angle</td><td>88,237</td><td>353</td><td>21,559</td><td>85</td></tr><tr><td>Right hemidiaphragm</td><td>8,905</td><td>80</td><td>2,577</td><td>23</td></tr><tr><td>Left lung</td><td>147,063</td><td>545</td><td>32,656</td><td>120</td></tr><tr><td>Left upper lung zone</td><td>4,204</td><td>32</td><td>673</td><td>7</td></tr><tr><td>Left mid lung zone</td><td>1,880</td><td>61</td><td>476</td><td>14</td></tr><tr><td>Left lower lung zone</td><td>33,104</td><td>190</td><td>6,928</td><td>43</td></tr><tr><td>Left hilar structures</td><td>5,970</td><td>352</td><td>1,268</td><td>88</td></tr><tr><td>Left apical zone</td><td>2,847</td><td>61</td><td>613</td><td>21</td></tr><tr><td>Left costophrenic angle</td><td>28,197</td><td>314</td><td>5,135</td><td>80</td></tr><tr><td>Left hemidiaphragm</td><td>3,157</td><td>51</td><td>644</td><td>8</td></tr><tr><td>Trachea</td><td>17,302</td><td>46</td><td>645</td><td>2</td></tr><tr><td>Spine</td><td>26,426</td><td>0</td><td>9,381</td><td>0</td></tr><tr><td>Right clavicle</td><td>6,765</td><td>116</td><td>1,559</td><td>35</td></tr><tr><td>Left clavicle</td><td>1,534</td><td>115</td><td>198</td><td>34</td></tr><tr><td>Aortic arch</td><td>8,939</td><td>86</td><td>2,815</td><td>30</td></tr><tr><td>Mediastinum</td><td>113,222</td><td>394</td><td>26,822</td><td>97</td></tr><tr><td>Upper mediastinum</td><td>2,820</td><td>283</td><td>373</td><td>77</td></tr><tr><td>SVC</td><td>8,576</td><td>12</td><td>1,060</td><td>2</td></tr><tr><td>Cardiac silhouette</td><td>119,304</td><td>490</td><td>29,285</td><td>113</td></tr><tr><td>Cavoatrial junction</td><td>2,578</td><td>2</td><td>334</td><td>0</td></tr><tr><td>Right atrium</td><td>4,427</td><td>6</td><td>946</td><td>0</td></tr><tr><td>Carina</td><td>6,552</td><td>32</td><td>13</td><td>0</td></tr><tr><td>Abdomen</td><td>25,310</td><td>0</td><td>3,348</td><td>0</td></tr></table>

Supplementary Table S5 CheXbert positive-label counts in the training, validation and test splits of MIMIC-CXR and MIMIC-ABN.
<table><tr><td rowspan="2">Observation</td><td colspan="3">MIMIC-CXR</td><td colspan="3">MIMIC-ABN</td></tr><tr><td>Train</td><td>Validation</td><td>Test</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>Enlarged Cardiomediastinum</td><td>29,449</td><td>253</td><td>687</td><td>2,624</td><td>27</td><td>54</td></tr><tr><td>Cardiomegaly</td><td>43,528</td><td>333</td><td>1,047</td><td>7,077</td><td>58</td><td>131</td></tr><tr><td>Lung Opacity</td><td>42,613</td><td>322</td><td>949</td><td>10,854</td><td>91</td><td>205</td></tr><tr><td>Lung Lesion</td><td>6,577</td><td>74</td><td>165</td><td>2,130</td><td>25</td><td>53</td></tr><tr><td>Edema</td><td>21,952</td><td>186</td><td>638</td><td>2,488</td><td>27</td><td>68</td></tr><tr><td>Consolidation</td><td>9,006</td><td>77</td><td>228</td><td>1,360</td><td>13</td><td>33</td></tr><tr><td>Pneumonia</td><td>14,808</td><td>117</td><td>324</td><td>3,141</td><td>30</td><td>63</td></tr><tr><td>Atelectasis</td><td>44,209</td><td>352</td><td>840</td><td>8,480</td><td>71</td><td>106</td></tr><tr><td>Pneumothorax</td><td>5,508</td><td>37</td><td>73</td><td>302</td><td>6</td><td>0</td></tr><tr><td>Pleural Effusion</td><td>37,209</td><td>316</td><td>945</td><td>4,418</td><td>36</td><td>96</td></tr><tr><td>Pleural Other</td><td>3,950</td><td>40</td><td>93</td><td>1,300</td><td>17</td><td>30</td></tr><tr><td>Fracture</td><td>6,374</td><td>32</td><td>141</td><td>1,479</td><td>10</td><td>39</td></tr><tr><td>Support Devices</td><td>45,398</td><td>346</td><td>1,004</td><td>4,757</td><td>34</td><td>102</td></tr><tr><td>No Finding</td><td>36,727</td><td>290</td><td>153</td><td>4,664</td><td>31</td><td>30</td></tr></table>

Supplementary Table S6 Reasoning length and anatomical grounding of MCoT training trajectories.
<table><tr><td>Characteristic</td><td>Mean</td><td>Min</td><td>Max</td></tr><tr><td>Reasoning text length (words)</td><td>399.98</td><td>118</td><td>823</td></tr><tr><td>Region-grounding instances per trajectory</td><td>5.75</td><td>0</td><td>15</td></tr></table>

Statistics summarize $n = 1 5 9$ ,416 training trajectories. Word counts include only prose within <think>, after removing complete anatomical grounding blocks (anatomical labels, bounding-box coordinates and <image> markers) and other special markers. MCoT, multimodal chain-of-thought.

## S3 Reasoning CheXbert validation results

Supplementary Table S7 Reasoning CheXbert performance on the held-out split of the MCoT training set. Uncertain labels are treated as positive for metric computation.

<table><tr><td>Average</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Micro</td><td>0.9160</td><td>0.9194</td><td>0.9177</td></tr><tr><td>Macro</td><td>0.9110</td><td>0.9009</td><td>0.9056</td></tr></table>

On this held-out split, reasoning CheXbert achieved micro and macro $\mathrm { F _ { 1 } }$ scores of 0.9177 and 0.9056, respectively (Supplementary Table S7). These within-dataset results support its use as an automated label extractor for the reported MCoT analyses.

## S4 Model size and inference time

Supplementary Table S8 Model size, decoding strategy and mean inference time on the MIMIC-CXR test set.
<table><tr><td>Model</td><td>Parameters (B)</td><td>Decoding</td><td>Output</td><td>Mean time (s/sample)</td></tr><tr><td>MLRG</td><td>0.30</td><td>Beam search (3)</td><td>Report</td><td>0.51</td></tr><tr><td>PriorRG</td><td>0.34</td><td>Beam search (3)</td><td>Report</td><td>0.32</td></tr><tr><td>MedGemma-1.5</td><td>4.30</td><td>Greedy</td><td>Report</td><td>1.30</td></tr><tr><td>ARIVE</td><td>4.85</td><td>Sampling</td><td>MCoT + report</td><td>1.13</td></tr><tr><td>MAIRA-2</td><td>6.88</td><td>Greedy</td><td>Report</td><td>1.20</td></tr><tr><td>Libra</td><td>6.89</td><td>Beam search (10)</td><td>Report</td><td>15.03</td></tr></table>

All models were run on a single NVIDIA A100 GPU using the inference protocol specified by each project. Time is averaged per test sample for the complete output shown. B, billion parameters; numbers in parentheses indicate beam width. MCoT, multimodal chain-of-thought.

## S5 Prompt templates

Supplementary Tables S9 and S10 provide the prompts for textual CoT construction and subsequent bounding-box insertion. Supplementary Tables S11 and S12 provide the report-generation prompts for MIMIC-CXR and MIMIC-ABN, respectively. Braced uppercase fields denote placeholders, and <image> marks an image input. Double square brackets delimit optional prior-study content and are explanatory notation; the enclosed content is included only when a valid prior study is available.

Supplementary Table S9 Prompt for textual CoT construction. The source report supplies the current findings; few-shot examples specify the narrative style.

Textual CoT construction   
You are an expert chest X-ray radiologist. Produce a complete, optimized first-person   
reasoning narrative for the current CXR.   
The examples define only the desired narrative voice, regional progression, paragraph   
development, and reasoning depth. Never transfer their clinical findings or routine   
normal-structure checklist to the current case.   
<few\_shot\_examples>   
{FEW\_SHOT\_EXAMPLES}   
</few\_shot\_examples>   
<input>   
Current CXR report (ground truth; do not copy verbatim):   
{CURRENT\_CXR\_REPORT}   
Clinical context (available Indication, History, Technique, and Comparison):   
{CLINICAL\_CONTEXT}   
Prior CXR report:   
{PRIOR\_CXR\_REPORT\_OR\_NONE}   
</input>   
Internally complete two steps:   
1. Develop a detailed, region-by-region walkthrough of the current findings, integrating   
the clinical context and supported prior comparisons.   
2. Refine the narrative against the supplied evidence, correcting unsupported statements   
and restoring omitted findings without compressing the reasoning.   
Return only the final refined narrative; do not reveal the intermediate draft or audit.   
Factual rules:   
- Use the current report as the authority for current imaging observations.   
- Preserve every finding, location, measurement, severity, uncertainty, and visibility   
limitation.   
- Use clinical context to frame the clinical question and interpretation, without treating   
history as an observed imaging finding.   
- Describe reported positive and negative findings. Do not assume that an unreported   
structure, device, or image-quality feature is normal.   
- Expand diagnostic reasoning, not visual detail. Blend supported, conditional   
explanations naturally into the observations.   
- Retain reported device endpoints and stable contours without adding unsupported   
placement, function, safety, or normality judgments.   
Comparison rules:   
- Use a valid prior report for supported comparisons of corresponding findings, guided by   
the supplied Comparison field.

<table><tr><td>Supplementary Table S9 (continued)</td></tr><tr><td>Textual CoT construction - Preserve the certainty of reported changes. A prior-only finding or an omission in either</td></tr><tr><td>report does not establish a current finding or interval change. - Do not invent dates. - If no valid prior report is supplied, preserve interval findings explicitly stated in the</td></tr><tr><td>current report without claiming direct prior-image review. Style and organization: - Match the examples&#x27; detailed first-person, reader-guided narrative.</td></tr><tr><td>- Follow their regional progression, covering technique and limitations, devices, airways, lungs, pleura, central thoracic structures, and other structures only where the case supplies relevant findings.</td></tr><tr><td>- Develop substantial paragraphs connecting observations, their significance, relevant findings, and the clinical question. - Use supplied technique information and describe the frontal examination only. Never</td></tr><tr><td>mention lateral views. - End with a concise synthesis of the main findings and clinical implications.</td></tr><tr><td>- Avoid recommendations and commentary about report wording or missing inputs. - Do not use titles, section headers, lists, bullet points, or bounding boxes. Output only the complete final narrative.</td></tr><tr><td>Supplementary Table S10 Prompt for inserting anatomical bounding boxes into textual CoT</td></tr><tr><td>trajectories to construct MCoT. Bounding-box insertion</td></tr><tr><td>You are an expert radiologist. Insert relevant bounding boxes from the Anatomical Information into the Generated Reasoning. Inputs: - Original Report: {ORIGINAL_REPORT}</td></tr><tr><td>- Generated Reasoning: {GENERATED_REASONING} - Anatomical Information: {ANATOMICAL_INFORMATION} Rules:</td></tr></table>

Supplementary Table S11 Prompt for MIMIC-CXR report generation, covering the full findings   
description. The [...] block is included only when an indication is available; [[...]] blocks are   
included only when a prior study is available.   
MIMIC-CXR report generation   
You are an assistant in radiology, responsible for analyzing medical imaging studies and   
generating detailed, structured, and accurate radiology reports.   
[Indication: {INDICATION}]   
[[Prior chest X-ray:   
<image>   
Prior findings:{PRIOR\_CXR\_REPORT}   
]]   
Current chest X-ray:   
<image>   
Before answering, think step by step internally. In your internal thinking, output the   
bounding box of the key diagnostic region in the format <|object\_ref\_start|>   
anatomical object name<|object\_ref\_end|><|box\_start|>x1 y1 x2 y2<|box\_end|>.   
Generate a comprehensive and detailed description of findings based on the current chest   
X-ray image and the indication.[ Include a thorough comparison with a prior chest X-ray,   
emphasizing any significant changes, progression, or improvement.]]

Supplementary Table S12 Prompt for MIMIC-ABN report generation, restricted to abnormal findings. The [...] block is included only when an indication is available; [[...]] blocks are included only when a prior study is available.

MIMIC-ABN report generation   
You are an assistant in radiology, responsible for analyzing medical imaging studies and   
generating detailed, structured, and accurate radiology reports.   
[Indication: {INDICATION}]   
[[Prior chest X-ray:   
<image>   
Prior abnormal findings:{PRIOR\_ABNORMAL\_FINDINGS}   
]]   
Current chest X-ray:   
<image>   
Before answering, think step by step internally. In your internal thinking, output the   
bounding box of the key diagnostic region in the format <|object\_ref\_start|>   
anatomical object name<|object\_ref\_end|><|box\_start|>x1 y1 x2 y2<|box\_end|>.   
Generate a comprehensive and detailed description of abnormal findings only based on   
the current chest X-ray image and the indication.[ Include a thorough comparison with a   
prior chest X-ray, emphasizing any significant changes, progression, or improvement.]]