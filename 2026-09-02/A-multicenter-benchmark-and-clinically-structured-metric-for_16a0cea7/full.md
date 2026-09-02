# A multicenter benchmark and clinically structured metric for coronary CTA report generation

Zhiyu Ye<sup>1,2,4,+</sup>, Yue Sun<sup>3,+</sup>, Limiao Zou<sup>3</sup>, Cheng Xu<sup>3</sup>, Keting Xu<sup>3</sup>, Tong Hu<sup>2</sup>, Yue Yu<sup>2</sup>, Hairong Zheng<sup>1,\*</sup>, Yining Wang<sup>4,\*</sup>, and Tong Zhang<sup>2,\*</sup>

<sup>1</sup>State Key Laboratory of Biomedical Imaging Science and System, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China

<sup>2</sup>Pengcheng Laboratory, Shenzhen, China

<sup>3</sup>Department of Radiology, State Key Laboratory of Complex Severe and Rare Diseases, Peking Union Medical College Hospital, Chinese Academy of Medical Sciences and Peking Union Medical College, Beijing, China <sup>4</sup>University of Chinese Academy of Sciences, Beijing, China

Correspondence: zhangt02@pcl.ac.cn

<sup>+</sup>These authors contributed equally to this work.

## ABSTRACT

Reliable evaluation of automated coronary computed tomography angiography (CCTA) report generation requires standardized multicentre benchmarks and clinically structured metrics. We established a four-centre benchmark comprising 3,021 CCTA series from 818 patient-report pairs to evaluate seven open-source three-dimensional vision-language models. We developed $\mathsf { C S M } _ { \mathsf { C C T A } }$ , a clinically structured metric for CCTA report evaluation, with patient-, vessel-, and segment-level variables defined according to clinical guidelines. Report pairs are compared at the finest shared anatomical level, and the contributions of different clinical components are weighted based on expert assessments. We estimated these weights using 70 expert-scored cases and evaluated clinical alignment in a non-overlapping set of 30 cases. $\mathsf { C S M } _ { \mathsf { C C T A } }$ showed a strong correlation with radiologist scores (Pearson’s r = 0.97, p < 0.001), exceeding the next-best metric, FORTE (r = 0.70), by 0.27, and agreed with expert preferences in 115 of 160 pairwise comparisons (71.9%). Under controlled perturbations, $\mathsf { C S M } _ { \mathsf { C C T A } }$ remained stable to clinically equivalent wording and decreased monotonically with progressive information omission. In the multicenter benchmark, the CCTA-trained C2RG model achieved the highest $\mathsf { C S M } _ { \mathsf { C C T A } }$ scores across all four hospitals, although its performance remained far from optimal. In contrast, CCTA-irrelevant reports accounted for up to 98.7% of the outputs from generalist models. Together, the benchmark provides a standardized setting for model comparison, while $\mathsf { C S M } _ { \mathsf { C C T A } }$ enables clinically structured evaluation of finding agreement and anatomical specificity. These results support a more clinically aligned and anatomically resolved approach to evaluating CCTA report generation. Code is available at https://openi.pcl.ac.cn/OpenMedIA/CSM\_CCTA.

## Introduction

Coronary computed tomography angiography (CCTA) transforms complex 3D anatomical images into essential information for the early screening, diagnosis, and clinical decision-making of coronary artery disease<sup>1–4</sup>. While artificial intelligence has advanced various CCTA analysis tasks, including calcium scoring<sup>5–8</sup>, plaque classification<sup>9–11</sup>, CAD-RADS prediction<sup>8,</sup> <sup>12–14</sup>, and cardiovascular event forecasting<sup>11,</sup> <sup>15,</sup> <sup>16</sup>, the ultimate synthesis of these findings into high-quality clinical reports remains a complex, time-intensive burden for radiologists. Automated medical report generation (MRG) holds significant promise for alleviating this workload and improving efficiency. However, the successful development and subsequent clinical translation of such automated systems are facilitated by the availability of reliable evaluation metrics and standardized benchmarks.

Driven by advancements in foundation models, vision-language models (VLMs) have achieved notable progress in MRG. Yet, the majority of dedicated evaluation benchmarks and well-developed models focus predominantly on 2D imaging modalities, particularly chest X-rays<sup>17–20</sup>. While general-purpose VLMs have been explored for multimodal and multianatomical structures<sup>21–23</sup>, their performance remains insufficient for accurate CCTA report generation in the evaluated settings. Although research into 3D medical images is expanding across brain $\mathrm { C T } ^ { 2 4 }$ , pulmonary $\mathsf { \bar { C } T A } ^ { 2 5 }$ , multi-organ $\mathrm { C T } ^ { 2 6 - 2 8 }$ , and preliminary CCTA studies<sup>29</sup>, the lack of a dedicated CCTA benchmark forces models to rely on inconsistent evaluation settings, complicating cross-study comparison and preclinical evaluation.

Medical report evaluation requires metrics to prioritize clinical correctness over textual fluency. Large language models (LLMs) now generate reports with excellent clarity and fluency. Consequently, readability alone is insufficient for evaluating medical reports, making clinical accuracy a central consideration. In practice, models may produce highly readable text that contains severe clinical errors (Fig. 1a). Therefore, the evaluation bottleneck lies in clinical accuracy. This requires the precise identification of abnormality types, anatomical locations, and severity gradings. Traditional natural language generation (NLG) metrics (e.g., BLEU<sup>30</sup>, ROUGE<sup>31</sup>, METEOR<sup>32</sup>, CIDEr<sup>33</sup>, and BERTScore (BERT-F1)<sup>34</sup>) primarily focus on semantic similarity and fluency, but they do not explicitly account for clinically critical information. Furthermore, the medical-domain metrics including dataset-specific metrics (e.g., RadGraph and $\mathrm { R a d C l i Q } ^ { 3 5 }$ for $\mathbf { M I M C - C X R } ^ { 3 6 } )$ and general-purpose metrics (e.g., RaTEScore<sup>37</sup> and GREEN<sup>38</sup>), do not adequately capture the hierarchical and highly localized anatomy of coronary arteries. In clinical practice, a generated CCTA report may identify the correct abnormality but assign it to the wrong anatomical location or misstate its severity. These cases require evaluation of the vessel, segment, abnormality, and associated attributes of each finding. Reports may also differ in anatomical granularity, ranging from patient-level summaries to vessel- and segment-level descriptions. Existing metrics typically return a single similarity score and may therefore provide unreliable or poorly interpretable assessments when reports differ in reporting detail (Fig. 1a).

![](images/bebd771ce8e05693729d579c2167a3edeb92f40ea7e51c689701c507d8e6d21c.jpg)  
a

![](images/9d25bf0bf388f581813e523b5f892b1553601f0affcd967777ff9298b3c73898.jpg)

![](images/346a9a1a82a283d2dbe88129f7e1dda1501e589915ac8dccc14fca38f939da9a.jpg)

![](images/014e2b243346c3d70c1f6749c895f12fa2335c6f6e03a22088153588a3977bed.jpg)  
Figure 1. Motivation and overview of $\mathbf { C S M _ { C C T A } }$ and the multicenter benchmark. (a) Motivation and overview of $\mathrm { C S M _ { C C T A } }$ for clinically grounded evaluation of CCTA reports. Representative examples show that model-generated reports can remain clear and fluent despite substantial clinical errors. Conventional NLG metrics assess the entire report into a single score, while $\mathrm { C S M } _ { \mathrm { C C T A } }$ extracts structured clinical information and performs variable-level comparisons across patient, vessel, and segment anatomical levels, enabling a more clinically interpretable assessment of report accuracy. (b) Construction of ${ \mathrm { C S M } } _ { \mathrm { C C T A } } { \mathrm { : } }$ The proposed metric is developed in three steps. Step 1 uses guideline-informed variable design and clinical keyword extraction to parse free text into hierarchical clinical variables (patient-level summaries, vessel-level descriptions, and segment-level quadruples comprising vessel, segment, abnormality, and severity). Step 2 divides 100 expert-scored cases into a 70-case derivation set and a non-overlapping 30-case validation set. Step 3 estimates non-negative expert-calibrated weights by constrained linear regression using only the derivation set. (c) On the held-out validation set, $\mathrm { C S M _ { C C T A } }$ demonstrates a strong correlation with experts (Pearson $r = 0 . 9 7 , p < 0 . 0 0 1 )$ , outperforming 13 compared metrics. (d) $\mathrm { C S M _ { C C T A } }$ is used to benchmark seven representative VLMs across a multicenter dataset.

To bridge this gap, we make two primary contributions in this study. First, we construct a multicenter benchmark to facilitate standardized, reproducible comparisons of VLM performance in CCTA report generation. Second, to evaluate CCTA clinical findings, we propose a novel Clinically Structed Metric $\mathrm { ( C S M _ { C C T A } ) }$ . Within the $\mathrm { C S M _ { C C T A } }$ framework, the metric automatically extracts essential clinical terms based on a guideline-informed variable design and organizes them into precise vessel-segment-abnormality-severity quadruples, alongside broader patient- and vessel-level variables (Fig. 1b). Furthermore, $\mathrm { C S M _ { C C T A } }$ employs an adaptive hierarchical scoring mechanism that dynamically evaluates reports at their finest shared anatomical level and aggregates findings from more specific anatomical levels when necessary. This level-matched comparison distinguishes clinical accuracy from the anatomical level of reporting, preventing reports from being unfairly penalized when evaluated against less detailed reference reports. Finally, to ensure that the evaluation aligns with clinical guidelines and practice priorities, the metric’s internal weighting scheme is rigorously calibrated via constrained linear regression using a dedicated derivation set of expert-scored cases. Ultimately, this comprehensive approach enables anatomically resolved evaluation that preserves clinically meaningful localization while accommodating reports written at different anatomical levels.

Table 1. Summary of the CCTA report generation benchmark dataset.
<table><tr><td rowspan="2">Statistic</td><td colspan="4">Dataset</td><td rowspan="2">Total</td></tr><tr><td>PUMCH</td><td>TCH</td><td>SJTH</td><td>FAHXMU</td></tr><tr><td>Total patient-report pairs</td><td>567</td><td>105</td><td>49</td><td>97</td><td>818</td></tr><tr><td>Total CTA series</td><td>2121</td><td>430</td><td>147</td><td>323</td><td>3021</td></tr><tr><td>CTA series per patient</td><td>3.7 (1–11)</td><td>4.1 (1–17)</td><td>3.0 (1–11)</td><td>3.3 (1–17)</td><td>3.7 (1–17)</td></tr><tr><td>Examination date</td><td>2023-03-07 – 2024-03-29</td><td>2014-02-24- 2014-12-03</td><td>2014-01-06- 2014-12-26</td><td>2016-01-04- 2016-10-17</td><td>1</td></tr><tr><td colspan="6">Anatomical level of reference report</td></tr><tr><td>Patient</td><td>0 (0.0)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>0 (0.0)</td></tr><tr><td>Vessel</td><td>130 (22.9)</td><td>99 (94.3)</td><td>2 (4.1)</td><td>1 (1.0)</td><td>232 (28.4)</td></tr><tr><td>Segment</td><td>437 (77.1)</td><td>6 (5.7)</td><td>47 (95.9)</td><td>96 (99.0)</td><td>586 (71.6)</td></tr><tr><td colspan="6">Coronary dominance</td></tr><tr><td>Right</td><td>524 (92.4)</td><td>95 (90.5)</td><td>35 (71.4)</td><td>0 (0.0)</td><td>654 (80.0)</td></tr><tr><td>Left</td><td>36 (6.3)</td><td>5 (4.8)</td><td>2 (4.1)</td><td>0 (0.0)</td><td>43 (5.3)</td></tr><tr><td>Balanced</td><td>7 (1.2)</td><td>5 (4.8)</td><td>3 (6.1)</td><td>0 (0.0)</td><td>15 (1.8)</td></tr><tr><td>Unclear</td><td>0 (0.0)</td><td>0 (0.0)</td><td>9 (18.4)</td><td>97 (100.0)</td><td>106 (13.0)</td></tr><tr><td colspan="6">Overall coronary calcification severity</td></tr><tr><td>None</td><td>206 (36.3)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>206 (25.2)</td></tr><tr><td>Mild</td><td>134 (23.6)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>134 (16.4)</td></tr><tr><td>Moderate</td><td>116 (20.5)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>116 (14.2)</td></tr><tr><td>Severe</td><td>111 (19.6)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>0 (0.0)</td><td>111 (13.6)</td></tr><tr><td>Unclear</td><td>0 (0.0)</td><td>105 (100.0)</td><td>49 (100.0)</td><td>97 (100.0)</td><td>251 (30.7)</td></tr><tr><td colspan="6">Patient-level maximal stenosis severity</td></tr><tr><td>None</td><td>134 (23.6)</td><td>49 (46.7)</td><td>4 (8.2)</td><td>0 (0.0)</td><td>187 (22.9)</td></tr><tr><td>Mild</td><td>213 (37.6)</td><td>22 (21.0)</td><td>27 (55.1)</td><td>59 (60.8)</td><td>321 (39.2)</td></tr><tr><td>Moderate</td><td>116 (20.5)</td><td>0 (0.0)</td><td>8 (16.3)</td><td>29 (29.9)</td><td>153 (18.7)</td></tr><tr><td>Severe / occlusion</td><td>104 (18.3)</td><td>1 (1.0)</td><td>8 (16.3)</td><td>9 (9.3)</td><td>122 (14.9)</td></tr><tr><td>Unclear</td><td>0 (0.0)</td><td>33 (31.4)</td><td>2 (4.1)</td><td>0 (0.0)</td><td>35 (4.3)</td></tr></table>

CTA series per patient denotes the mean number of CCTA image series available for each patient. Values are reported as mean (minimum–maximum). For anatomical level of reference report and the three clinical characteristics, values in parentheses are percentages based on patient-report pairs. Overall coronary calcification severity was recorded only when explicitly stated in the report. Localized calcified plaques were not extrapolated to overall calcification severity.

We establish baseline performances (Fig. 1d) on our proposed benchmark by evaluating 7 state-of-the-art VLMs (i.e., $\mathrm { C } 2 \mathrm { R } \mathrm { G } ^ { 2 9 }$ , CT-CHAT<sup>26</sup>, M3D<sup>27</sup>, RadFM<sup>28</sup>, HuluMed-4B/7B<sup>39</sup>, and $\mathrm { M e r l i n } ^ { 4 0 } )$ . The proposed $\mathrm { C S M } _ { \mathrm { C C T A } }$ is then validated through extensive analyses and controlled perturbation experiments. Specifically, our results on a held-out validation set show that $\mathrm { C S M _ { C C T A } }$ maintains a strong correlation with human experts and achieved the highest observed correlation among the 13 compared metrics (Fig. 1c). Ultimately, this work provides a standardized evaluation setting that supports the continued development of clinically viable VLMs for CCTA report generation. Beyond this specific modality, the dynamic hierarchical evaluation strategy introduced by $\mathrm { C S M } _ { \mathrm { C C T A } }$ offers a design template that may be adapted to other medical domains after task-specific schema design and expert recalibration.

## Results

## Multicenter Benchmark for CCTA Report Generation

To evaluate CCTA report generation, we established a multicenter benchmark comprising 3,021 CCTA series and 818 patientreport pairs from four hospitals (Table 1). On this benchmark, we evaluated seven representative 3D medical VLMs (Fig. 1d). The generated reports were assessed using our proposed $\mathrm { C S M _ { C C T A } }$ as the primary metric, alongside 13 established baselines (i.e., BLEU-1–4<sup>30</sup>, ROUGE-1, ROUGE-2, ROUGE-L<sup>31</sup>, METEOR<sup>32</sup>, CIDEr<sup>33</sup>, BERT-F1<sup>34</sup>, GREEN<sup>38</sup>, RaTEScore<sup>37</sup>, and $\mathrm { F O R T E } ^ { 2 5 } )$ . Specifically, CSM represents reports as hierarchical clinical variables and evaluates them at the finest shared anatomical level. It reports the anatomical evaluation level, and derives clinical accuracy by combining variable-level scores via expert-calibrated weights. Further details regarding benchmark construction and metric definitions are provided in Methods.

## Performance of 3D Medical VLMs

To evaluate the performance of 3D medical VLMs within this benchmark, we compared $\mathrm { C S M } _ { \mathrm { C C T A } }$ against existing metrics in assessing the quality and clinical relevance of generated CCTA reports. The results are summarized in Fig. 2 and Supplementar Table ??.

Model comparison across the benchmark. On our benchmark, the 3D medical VLMs exhibited distinct performance disparities across the four datasets. Specifically, the domain-adapted C2RG model consistently achieved the highest $\mathrm { C S M _ { C C T A } }$ scores (PUMCH: 0.456; SJTH: 0.520; TCH: 0.355; FAHXMU: 0.479). In contrast, models primarily trained on out-of domain data (e.g., Merlin, trained on abdominal images) generated limited relevant findings and performed poorly. Notably, compared metrics struggled to reflect this clear performance gap: semantic metrics (RaTEScore, BERT-F1) showed compressed, indiscriminately high score ranges, while lexical metrics (BLEU, ROUGE, METEOR, CIDEr) were overly sensitive to surface level wording. This confirms that high textual similarity in traditional metrics does not equate to the clinical completeness of model-generated reports.

Analysis across anatomical levels with $\mathtt { C S M _ { C C T A } }$ . To provide an anatomically related analysis of VLM performance, we stratified the report pairs by anatomical level (patient, vessel, or segment), as illustrated in Fig. 3 and detailed in Supplementary Fig. ??. Top-performing models like C2RG excel by producing highly detailed, fine-grained findings. Conversely, generalist models struggle with clinical depth. For instance, M3D yields only a minor subset of reports with superficial, patient-level information, providing limited CCTA-specific clinical information. Moreover, most outputs from M3D and Merlin are entirely irrelevant to CCTA, rendering them unevaluable (scoring 0). Ultimately, these results underscore that detailed finding descriptions are important for the evaluated CCTA reporting task, rather than merely generating fluent or superficial text.

## Clinical Relevance of Metrics

To determine whether the evaluation metrics reflect clinical concerns and align with expert judgments, we assembled an expertscored cohort of 100 cases from PUMCH. Each case contained CCTA images, the reference report, and a C2RG-generated report. We prespecified 70 cases as a derivation set for estimating the $\mathrm { C S M _ { C C T A } }$ weights and reserved the remaining 30 cases as a non-overlapping validation set for correlation analysis. Report-level preference testing provided a complementary comparison of metric and expert decisions. Details of the scoring protocol, cohort allocation, and inter-expert reliability are provided in the Supplementary Materials.

On the held-out 30-case validation set, we calculated Pearson correlation coefficients between each metric and the average expert scores. As shown in Fig. 4a, CSM<sub>CCTA</sub> achieved the highest observed correlation $( r = 0 . 9 7 , p < 0 . 0 0 1 )$ and had the narrowest observed 95% confidence interval (CI), substantially outperforming both traditional NLG (e.g., BLEU) and medical-specific metrics (e.g., FORTE).

![](images/6153047378f62b21ff1d1ecd025a3a410acc13b63ab72b4f9f911a29f73aa77d.jpg)

![](images/7760b3a65bbc6a5eeda275569da72a25433ecaa34ff68a058f7605386ed3568e.jpg)

![](images/96308329f3b08391ca2d0c894c6707ac08288c3b47aaebcecaf1a36229a0992d.jpg)

![](images/12fe5956a897b2ad169dce20944945ae7573ecb92fb74d2df493378522e97028.jpg)

![](images/2dfacd9dac20d07e7dceeeb8f04cf14dfdeea877b780637aac47d61bb69df803.jpg)

![](images/20b7eae6f80f3a7b0adcdcb6f04b885d4993c03519cdc0ef73dbc9fda5f2edca.jpg)

![](images/77131f8af10b66d6d58b47b6c28b22561713520758828bcf24a82b4049442d57.jpg)

![](images/2909212538f57d295716819e9067cc8e4ff60fff9168ff21a711e0f22af6841e.jpg)

![](images/232dfe9810e6215775bfaf0dc29fc9aa241465cca597c042844d9f6dc46e443a.jpg)

![](images/7be228738b10f6164b202811a7d27628654cb5b6a00fb186ccf906d0888b8c2f.jpg)

![](images/e6fb6df30f19af3df9b39a5a0427a1e897c8f187a57dc0135a26a1ef3b883c9a.jpg)

![](images/0b2fd7a3ae6d4f450999ba1345ffef69c39cc8b331021c7c7b53280e12315b5e.jpg)

![](images/adda2e126a58d440a3524eeae43f0771ddcc8042ad434ee011075a3fb456b443.jpg)

![](images/ee1a81bf889762131f66401918bf7d554e2454d30d730ad9f9ebfb1bf067daec.jpg)  
Figure 2. Performance of 3D medical VLMs on the benchmark. Each panel summarizes one evaluation metric. Compared with NLG metrics, $\mathrm { C S M } _ { \mathrm { C C T A } }$ provides more discriminative separation among models across these datasets. Significance brackets compare the highest- and second-highest-scoring models within each dataset, and asterisks indicate statistical significance $( * P < 0 . 0 5 , * * P < 0 . 0 1 , \mathrm { a n d } * * * P < 0 . 0 0 1 )$ .

![](images/9f1488c72dcce4a764d7cc079d83c24e1d04295f21ad1dd713ae44cd736c80cb.jpg)  
Figure 3. $\mathbf { C S M _ { C C T A } }$ score distributions by anatomical level across datasets. The upper panels display the $\mathrm { C S M } _ { \mathrm { C C T A } }$ score distributions both overall and by evaluation level (patient, vessel, or segment). The lower panels show the number of report pairs evaluated at each level for each model. The results are grouped by four hospitals: (a, e) PUMCH, (b, f) TCH, (c, g) SJTH, and (d, h) FAHXMU. “Irrelevant” denotes report pairs for which no valid anatomical evaluation level could be established, and these pairs were assigned a $\mathrm { C S M } _ { \mathrm { C C T A } }$ score of 0.

To assess metric sensitivity to subtle clinical differences, both evaluation metrics and experts scored 160 randomly matched pairs of model-generated reports to identify the superior candidate. As shown in Fig. 4f, $\mathrm { C S M _ { C C T A } }$ achieved the highest agreement with human decisions (71.9%, 115/160 pairs), consistently outperforming other metrics across all preference scenarios (Fig. 4g&h). For comparison, the strongest medical baseline, FORTE, aligned with experts in only 51.3% of cases, while NLG metrics performed even worse. These results suggest that $\mathrm { C S M } _ { \mathrm { C C T A } }$ better reflected expert preferences than the compared metrics in this pairwise evaluation.

Notably, the 71.9% preference agreement rate does not fully mirror $\mathrm { C S M _ { C C T A } } ^ { \mathrm { { \cdot } } } \mathrm { \Omega } ^ { \mathrm { { \cdot } } }$ s near-perfect global correlation. This discrepancy may partly reflect the metric’s greater weighting of vessel-level findings, which penalizes vessel-level errors significantly more heavily than segment-level inaccuracies. Consequently, when two candidate reports are identical at the vessel level but differ only in fine-grained segment descriptions, $\mathrm { C S M _ { C C T A } }$ becomes less sensitive than human experts, occasionally leading to diverging selections.

## Robustness Analysis of Evaluation Metrics

A reliable metric should remain invariant to semantic-preserving paraphrasing while being highly sensitive to the omission of critical clinical facts. By stress-testing the metrics on data from PUMCH, we demonstrate the robustness from two perspectives: semantic stability and clinical fidelity. Fig. 5a illustrates the holistic performance of $\mathrm { C S M } _ { \mathrm { C C T A } }$ and 13 compared metrics.

Semantic stability. In clinical practice, due to variations in radiologists’ personal habits or institutional protocols, reports often employ diverse expressions to convey identical clinical findings. For example, qualitative descriptions like “severe stenosis” are occasionally detailed with quantitative values, such as “80% stenosis”. Semantic stability is the ability of a metric to remain invariant under such clinically equivalent lexical variations. To quantify semantic stability, we constructed a modified test set where descriptive clinical keywords $( i . e . ,$ , degree adverbs) were replaced by equivalent numerical percentages at a 50% replacement rate. Since the clinical semantics are preserved, the score drop after perturbation should ideally be zero. As shown in Fig. 5a and 5b, most compared metrics exhibit semantic rigidity, penalizing these clinically equivalent variations (e.g., the score of CIDEr drops by 0.50). In contrast, $\mathrm { C S M } _ { \mathrm { C C T A } }$ demonstrates semantic stability with a score drop of 0.00.

![](images/d7ac788a46b9a59ab9c8e0e03ecc4829c7024bed4873f6da2ade20595b027a68.jpg)

b  
![](images/6cfb91d212d7f4d17dcb32288c33828d957522664f02e6e56ae7901df5bce5c7.jpg)

![](images/dfa37ea56da3311110a009fe63272dc6e7c18e9d9c9f86243be7920f16f3d5cb.jpg)

![](images/9916e2ed1fd08821eb3dca318769955412982e204b1b5b8b039ba62214220780.jpg)

e  
![](images/4c004ac48f89ebb9ee25b320f7486ae0ce561df4be65cf0d13b6c89b34bd8a4e.jpg)

f  
![](images/96ec04e0cf6f9b997c3d84ad3df4055c91d7db7c970a733e2e69ba0e9d2ee54a.jpg)

g  
![](images/11b5e00eff4f438ce5f7703328b92727253cb36a23453bede9bd79e5f47fb573.jpg)

h  
![](images/0723fd15183335cb21d91af90cc79855ece655ff0044b740460cc96a231713c8.jpg)  
Figure 4. Clinical relevance analysis among metrics against human experts. (a) Forest plot comparing the Pearson correlation coefficients (r) between expert scores and each evaluation metric. Points indicate the Pearson r values, and whiskers represent the 95% CI. The accompanying table details the performance rank, P values, and statistical significance (Sig.) for each metric. $\mathrm { C S M _ { C C T A } }$ (highlighted in red) demonstrates the highest observed correlation with expert clinical judgment, achieving the highest correlation and the narrowest CI. (b-e) Scatter plots providing sample-wise comparisons between expert scores and representative compared metrics: BLEU-4 (b), ROUGE-L (c), GREEN (d), and FORTE (e). Each panel contrasts a specific compared metric (blue) with $\mathrm { C S M } _ { \mathrm { C C T A } }$ (red). The lines denote the linear regression fits for the respective metrics, with shaded regions representing the 95% CI. (f) Pairwise agreement counts among expert preferences and automated metric preferences across 160 comparisons. $\mathrm { C S M _ { C C T A } }$ achieved the highest agreement with experts’ clinical report-level preferences. (g, h) Metric-derived preferences stratified by cases in which experts preferred Report A (g) or Report B (h). Bars show the percentages of metric decisions favoring Report A (red), favoring Report B (blue), or indicating a tied preference (gray).

![](images/6091312ad7dc62264da48e259c15f754d5ca726fd1a54fe5a0f42f6d65af74ce.jpg)

b
<table><tr><td rowspan=2 colspan=1>Metrics</td><td rowspan=1 colspan=1>Semantic stability</td><td rowspan=1 colspan=3>Clinical fidelity</td></tr><tr><td rowspan=1 colspan=1>Score drop</td><td rowspan=1 colspan=1>Coefficient ↑</td><td rowspan=1 colspan=1> $R ^ { 2 } ~ \uparrow$ </td><td rowspan=1 colspan=1>Boundary error↓</td></tr><tr><td rowspan=1 colspan=1>BLEU-1</td><td rowspan=1 colspan=1>-0.07 (±0.07)</td><td rowspan=1 colspan=1>0.81 (±0.008)</td><td rowspan=1 colspan=1>0.81 (±0.006)</td><td rowspan=1 colspan=1>0.10 (±0.004)</td></tr><tr><td rowspan=1 colspan=1>BLEU-2</td><td rowspan=1 colspan=1>-0.09 (±0.09)</td><td rowspan=1 colspan=1>0.76 (±0.007)</td><td rowspan=1 colspan=1>0.82 (±0.006)</td><td rowspan=1 colspan=1>0.12 (±0.004)</td></tr><tr><td rowspan=1 colspan=1>BLEU-3</td><td rowspan=1 colspan=1>-0.11 (±0.10)</td><td rowspan=1 colspan=1>0.71 (±0.007)</td><td rowspan=1 colspan=1>0.82 (±0.007)</td><td rowspan=1 colspan=1>0.14 (±0.004)</td></tr><tr><td rowspan=1 colspan=1>BLEU-4</td><td rowspan=1 colspan=1>-0.13 (±0.12)</td><td rowspan=1 colspan=1>0.66 (±0.006)</td><td rowspan=1 colspan=1>0.79 (±0.007)</td><td rowspan=1 colspan=1>0.17 (±0.003)</td></tr><tr><td rowspan=1 colspan=1>ROUGE-1</td><td rowspan=1 colspan=1>-0.10 (±0.09)</td><td rowspan=1 colspan=1>0.61 (±0.010)</td><td rowspan=1 colspan=1>0.71 (±0.006)</td><td rowspan=1 colspan=1>0.19 (±0.005)</td></tr><tr><td rowspan=1 colspan=1>ROUGE-2</td><td rowspan=1 colspan=1>-0.20 (±0.16)</td><td rowspan=1 colspan=1>0.59 (±0.008)</td><td rowspan=1 colspan=1>0.72 (±0.008)</td><td rowspan=1 colspan=1>0.21 (±0.004)</td></tr><tr><td rowspan=1 colspan=1>ROUGE-L</td><td rowspan=1 colspan=1>-0.13 (±0.11)</td><td rowspan=1 colspan=1>0.58 (±0.009)</td><td rowspan=1 colspan=1>0.72 (±0.007)</td><td rowspan=1 colspan=1>0.21 (±0.004)</td></tr><tr><td rowspan=1 colspan=1>METEOR</td><td rowspan=1 colspan=1>-0.28 (±0.22)</td><td rowspan=1 colspan=1>0.43 (±0.004)</td><td rowspan=1 colspan=1>0.83 (±0.006)</td><td rowspan=1 colspan=1>0.29 (±0.002)</td></tr><tr><td rowspan=1 colspan=1>CIDEr</td><td rowspan=1 colspan=1>-0.50 (±0.40)</td><td rowspan=1 colspan=1>0.28 (±0.010)</td><td rowspan=1 colspan=1>0.34 (±0.012)</td><td rowspan=1 colspan=1>0.41 (±0.002)</td></tr><tr><td rowspan=1 colspan=1>BERT-F1</td><td rowspan=1 colspan=1>-0.05 (±0.04)</td><td rowspan=1 colspan=1>0.32 (±0.012)</td><td rowspan=1 colspan=1>0.43 (±0.006)</td><td rowspan=1 colspan=1>0.35 (±0.002)</td></tr><tr><td rowspan=1 colspan=1>RaTEScore</td><td rowspan=1 colspan=1>-0.10 (±0.09)</td><td rowspan=1 colspan=1>0.25 (±0.006)</td><td rowspan=1 colspan=1>0.40 (±0.016)</td><td rowspan=1 colspan=1>0.38 (±0.003)</td></tr><tr><td rowspan=1 colspan=1>GREEN</td><td rowspan=1 colspan=1>-0.28 (±0.26)</td><td rowspan=1 colspan=1>0.65 (±0.011)</td><td rowspan=1 colspan=1>0.57 (±0.013)</td><td rowspan=1 colspan=1>0.17 (±0.006)</td></tr><tr><td rowspan=1 colspan=1>FORTE</td><td rowspan=1 colspan=1>-0.20 (±0.11)</td><td rowspan=1 colspan=1>0.49 (±0.012)</td><td rowspan=1 colspan=1>0.53 (±0.011)</td><td rowspan=1 colspan=1>0.25 (±0.006)</td></tr><tr><td rowspan=1 colspan=1> $\mathtt { C S M } _ { \mathtt { C C T A } }$ </td><td rowspan=1 colspan=1>0.00 (±0.00)</td><td rowspan=1 colspan=1>1.00 (±0.006)</td><td rowspan=1 colspan=1>0.87 (±0.005)</td><td rowspan=1 colspan=1>0.05 (±0.002)</td></tr></table>

![](images/d1a385bf98d97d0c9aee1ac17a295721b2e62ebc93e3ee8b8405117fdcd0802a.jpg)

d  
![](images/c8b1581c88da3c8a14232ab78c93d5fae62ee21eb82e08f4e14ee434178e436a.jpg)  
f

![](images/ca258e4e9f5ebc4e6ee4d14e293c6a6c5fc1174a97f7df81bf27b12e1ba6b64c.jpg)

![](images/556674fa4873e5c94f2befa94480c4445faf4ab662f35402079a77932b4e4821.jpg)

![](images/5f07730401dbf9745a08a0e3361061e756ae06fa56838f750a622e80b97c984a.jpg)

h  
![](images/e4edb0203acb51c9353fe0a5f0622c2aad5c21b4d4002031edc158505ac2354c.jpg)  
Figure 5. Robustness analysis of metrics. (a) The scatter plot illustrates the holistic performance of $\mathrm { C S M } _ { \mathrm { C C T A } }$ and compared metrics. The x-axis represents clinical fidelity (quantified by the linear regression coefficient), and the y-axis indicates semantic stability (measured by the score drop under semantic-preserving lexical variation). $\mathrm { C S M _ { C C T A } }$ (red star) demonstrates superior robustness by achieving optimal performance in both dimensions. (b) Quantitative comparison detailing the robustness properties across all metrics. Semantic stability is measured by the score drop following semantic-preserving lexical substitutions. Clinical fidelity is evaluated via linear regression analysis, characterized by three derived indicators: the linear regression coefficient (sensitivity to information variation), the $R ^ { 2 }$ value (reliability of the linear fit), and the boundary error (deviation from extreme theoretical limits at 0% and 100% information completeness). (c–h) Scatter plots detailing the clinical fidelity of representative compared metrics: BLEU-1 (c), CIDEr (d), BERT-F1 (e), GREEN (f), and FORTE (g), compared against $\mathrm { C S M } _ { \mathrm { C C T A } }$ (h) under progressive clinical information omission. The x-axis denotes the completeness ratio of clinical information, and the y-axis shows the corresponding metric score. Black lines indicate the linear regression fits alongside their respective equations and $R ^ { 2 }$ values.

Clinical fidelity. The ability of a metric to reflect the presence of critical clinical information is referred to as clinical fidelity, serving as an indicator of its robustness. We evaluated this property via a progressive information omission experiment, generating simulated reports by randomly removing key clinical information at proportions ranging from 0% to 100%. Clinical fidelity is evaluated via linear regression analysis utilizing three derived metrics shown in Fig. 5b. These encompass the linear regression coefficient $( \beta )$ representing the sensitivity gradient to information loss, the coefficient of determination $( R ^ { 2 } )$ indicating the reliability of the linear fit, and the boundary error $( E _ { b o u n d } )$ measuring the deviation from the ideal theoretical bounds at 0% and 100% completeness. Fig. 5c-h illustrates the response patterns of representative metrics. Several baselines exhibit poor robustness. BERT-F1 demonstrates a largely unresponsive scoring pattern $( \beta = 0 . 3 2 , E _ { b o u n d } = 0 . 3 5 )$ and fails to adequately distinguish between reports completely devoid of findings and perfectly accurate ones. Meanwhile, other metrics such as CIDEr and GREEN display highly scattered distributions with low ${ \bar { R } } ^ { 2 }$ values. Conversely, $\mathrm { C S M _ { C C T A } }$ scores maintain a stable linear alignment with clinical information completeness $( \beta = 1 . 0 0 )$ , featuring a reliable linear fit $( R ^ { 2 } = 0 . 8 7 )$ and the lowest boundary error (0.05) among the compared metrics under the tested omission procedure.

## Illustrative Case Analysis of CSM<sub>CCTA</sub>

To illustrate how CSMCCTA behaves when paired reports differ in anatomical specificity, we examined three report pairs $( \operatorname { F i g } . 6 )$ . In these examples, CSMCCTA selects the finest anatomical level shared by both reports, thereby reducing the influence of mismatched reporting detail on the resulting comparison. In contrast, other metrics may not distinguish true clinical disagreement from differences in reporting detail. This limitation is illustrated by cases in which the reference and generated reports are expressed at different anatomical levels. For example, CIDEr drops from 0.9154 in Case 1 to 0 in Cases 2 and 3, while BLEU-4 decreases from 0.4622 to 0.3608 and 0.3987. Meanwhile, GREEN outputs an identical score of 0.6000 across all three pairs. Unlike these metrics, $\mathrm { C S M } _ { \mathrm { C C T A } }$ maintains valid clinical comparisons while explicitly reporting the evaluated anatomical level, providing a more interpretable evaluation when reference and generated reports differ in detail.

In addition to paired-report comparison, $\mathrm { C S M _ { C C T A } }$ can also operate in a single-report mode. In this setting, no reference report is provided and clinical accuracy is therefore not assessed. Instead, $\mathrm { C S M _ { C C T A } }$ characterizes the presence of clinical variables and the anatomical specificity of the report. This functionality enables researchers to determine whether an individual report meets the anatomical specificity required by a predefined task.

## Discussion

This study addresses two related barriers to evaluating CCTA report generation. The multicenter benchmark provides a common setting for comparing models across institutions. $\mathrm { C S M _ { C C T A } }$ complements this resource by converting free-text reports into patient-, vessel-, and segment-level clinical variables. Its central contribution is an adaptive comparison at the finest anatomical level supported by both reports. This design separates the accuracy of reported findings from the level of anatomical detail at which they are expressed.

The benchmark results illustrate why this distinction matters for model development. The CCTA-adapted model generated more relevant and anatomically specific findings, whereas several generalist models produced coarse or unrelated reports. This pattern suggests that fluent report-style language alone is an inadequate indicator of report quality in a specialized imaging task. Anatomical-level outputs provide additional information beyond a scalar score by revealing whether a model fails in overall relevance, vessel-level description, or segment-level localization. These failure modes may provide more actionable diagnostic information than lexical similarity alone when selecting data, training objectives, or models for further development. However, the observed ranking applies to the seven evaluated models and should not be interpreted as a universal comparison of model architectures.

![](images/454756028f398151b131eeff19a7dea0d038588496b7ce19977aefbd19af4aba.jpg)  
Figure 6. Application of the proposed CSM<sub>CCTA</sub> on representative coronary CTA reports. The left panel displays three distinct case scenarios comparing Ground Truth (GT) and Predicted (Pred) reports, highlighting variations in the presence of fine-grained details. The middle section illustrates the dynamic strategy, determining the evaluation level based on the available clinical information to avoid unfair penalization for missing details. The top-right table compares the scores of various evaluation metrics across these three cases. The bottom-right table presents a detailed hierarchy of the $\mathrm { C S M _ { C C T A } }$ scores into patient-level, vessel-level, and segment-level items.

The expert analyses provide held-out internal validation evidence for the clinical relevance of $\mathrm { C S M } _ { \mathrm { C C T A } }$ , with scores derived from the calibration cohort remaining strongly aligned with expert assessments in the non-overlapping validation cohort and better reflecting expert report preferences than existing metrics. The lower pairwise agreement, however, suggests that the current weighting scheme may be less sensitive to subtle segment-level differences due to the greater emphasis on vessel-level findings. The perturbation and case analyses further show that $\mathrm { C S M _ { C C T A } }$ is robust to clinically equivalent wording, sensitive to omitted findings, and avoids penalizing differences in reporting detail through adaptive level matching. This flexibility also introduces an interpretive trade-off, as a report may be accurate at the shared anatomical level while lacking the detail required for a specific task. $\mathrm { C S M _ { C C T A } }$ should therefore be interpreted together with its selected anatomical level and component outputs. In single-report mode, it can assess structural completeness and anatomical specificity, but not clinical accuracy in the absence of a reference report.

Several limitations constrain the scope of these conclusions. $\mathrm { C S M } _ { \mathrm { C C T A } }$ relies on a CCTA-specific schema, and adaptation to other imaging examinations will require task-specific definitions, extraction rules, and expert recalibration. Deterministic clinical keyword extraction is transparent and reproducible, but it may miss uncommon, ambiguous, or institution-specific expressions. The expert derivation and validation cohorts, as well as the perturbation experiments, were drawn from PUMCH. Larger multi-institutional expert studies are needed to assess the transportability of the learned weights. Reference reports also vary in completeness and anatomical detail, so level matching reduces but does not eliminate dependence on the reference standard. Finally, the real-world examples were illustrative rather than a prospective evaluation of clinical utility.

Taken together, the benchmark and $\mathrm { C S M _ { C C T A } }$ provide a clinically structured basis for comparing CCTA report-generation models. Their principal value lies in making both finding agreement and reporting specificity visible, while preserving clear boundaries on what each evaluation mode can establish.

## Methods

## Construction of the Multicenter CCTA Report Generation Benchmark

The benchmark is constructed for evaluating CCTA report generation. Each model received a 3D CCTA image as input and generated a radiology report for comparison with the corresponding clinical reference report.

Dataset. After excluding cases with inadequate image quality, we retained all eligible cases from Peking Union Medical College Hospital (PUMCH), Tianjin Chest Hospital (TCH), Beijing Shijitan Hospital (SJTH), and The First Affiliated Hospital of Xinjiang Medical University (FAHXMU). A subset of 100 cases was randomly sampled from the PUMCH dataset to serve as the expert evaluation cohort, completely separate from the benchmark (Supplementary Fig. ??). The final benchmark comprised 3,021 CCTA series from 818 patient-report pairs across four hospitals. The cohorts included PUMCH $( n = 5 6 7 )$ , TCH (n = 105), SJTH $( n = 4 9 )$ , and FAHXMU (n = 97). Table 1 summarizes the scan statistics and report-derived coronary findings.

Models. We evaluated seven open-source 3D medical VLMs, covering both CCTA-adapted and generalist models. $\mathrm { C } 2 \mathrm { R } \mathrm { G } ^ { 2 9 }$ is a 3D VLM specifically adapted for CCTA report generation. $\mathbf { C T - C H A T ^ { 2 6 } }$ is a conversational VLM for 3D chest CT that supports interactive visual question answering (VQA) and report-style reasoning. $\mathbf { M } 3 \mathbf { D } ^ { 2 7 }$ is a unified multimodal large language model capable of diverse 3D tasks, including retrieval, report generation, VQA, localization, and segmentation. $\mathrm { R a d F M } ^ { \bar { 2 } 8 }$ is a generalist radiology foundation model designed to extract clinically oriented representations for broad downstream radiology tasks. $\mathrm { H u l u M e d } { \cdot } \bar { 4 } \mathrm { B } / 7 \mathrm { B } ^ { 3 9 }$ are generalist medical VLMs that integrate 2D, 3D, and video inputs for diverse medical reasoning tasks. $\mathrm { M e r l i n } ^ { 4 0 }$ is a 3D VLM for computed tomography that supports adapted and zero-shot tasks, including segmentation and report generation.

Evaluation metrics. Generated reports were compared with their paired clinical reference reports using $\mathrm { C S M _ { C C T A } }$ as the primary metric alongside other 13 metrics. These metrics include 10 NLG metrics and 3 medical-specific metrics. Among the NLG metrics, BLEU-1, BLEU-2, BLEU-3, B $\boldsymbol { \mathrm { E U } } \boldsymbol { \mathrm { - } } 4 ^ { 3 0 }$ , ROUGE-1, ROUGE-2, and $\mathrm { \ R O U G E { - } L } ^ { 3 1 }$ , ME $\mathrm { T E O R } ^ { 3 2 }$ , and $\mathrm { C I D E r } ^ { 3 3 }$ capture complementary forms of lexical overlap, while $\mathrm { B E R T - F 1 } ^ { 3 4 }$ measured contextual token similarity. The three medical report metrics included $\mathrm { G R E E N } ^ { 3 8 }$ , RaTEScore<sup>37</sup>, and $\mathrm { F O R T E } ^ { 2 5 }$ . Since FORTE was originally developed for pulmonary CTA reports using a task-specific keyword list, we adapted it for CCTA evaluation by replacing the keyword list while retaining the scoring procedure. This adaptation was intended to align the terminology coverage of FORTE with the coronary anatomy and clinical findings represented in our benchmark.

## Clinically Structured Metric $( \mathsf { C S M } _ { \mathsf { C C T A } } )$

We propose $\mathrm { C S M _ { C C T A } }$ , a hierarchical evaluation framework designed to assess CCTA report quality across three anatomical levels: patient, vessel, and segment levels. $\mathrm { C S M _ { C C T A } }$ incorporates bottom-up information aggregation and an adaptive scoring mechanism that quantifies agreement with the clinical findings represented in the reference report $\mathrm { ( F i g . 7 ) }$ .

## Hierarchical Variable Representation

To evaluate a report R, we map it into a structured representation defined over a hierarchy of levels $\mathcal { L } = \{ \ell _ { p a t } , \ell _ { v e s } , \ell _ { s e g } \}$ , ordered by increasing anatomical specificity from the patient level to the vessel and segment levels. Let $\nu _ { \ell }$ denote the set of clinical variables at level $\ell \in { \mathcal { L } }$ . The clinical information in R is extracted into the following collection of variables, providing a structured representation:

$$
\mathcal { V } _ { \ell _ { p a t } } = \{ d _ { p } , c _ { p } , n _ { p } \} ,\tag{1}
$$

$$
\mathcal { V } _ { \ell _ { \nu e s } } = \{ n _ { \nu } , p _ { \nu } \ | \ \nu \in \mathbb { V } \} ,\tag{2}
$$

$$
\mathcal { V } _ { \ell _ { s e g } } = \bigcup _ { \nu \in \mathbb { V } } \{ q _ { s } ^ { a } = ( \nu , s , a , e _ { a } ) \mid s \in \mathbb { S } _ { \nu } \} ,\tag{3}
$$

where $\mathbb { V } = \{ \mathrm { L M } , \mathrm { L A D } , \mathrm { L C X } , \mathrm { R C A } \}$ represents the main coronary vessels, and S represents the 18-segment coronary artery model<sup>1</sup> defined by the Society of Cardiovascular Computed Tomography $\mathrm { ( S C C T ) ^ { 4 1 } }$ . Here, $\mathbb { S } _ { \nu } \subseteq \mathbb { S }$ denotes the SCCT segments anatomically assigned to vessel v. $\mathcal { V } _ { \ell _ { s e g } }$ is composed of all extracted semantic quadruples across vessels. The specific clinical variables are defined as follows:

• Coronary Dominance $( d _ { p } ) \colon d _ { p } \in \{ { \mathrm { r i g h t } } , { \mathrm { l e f t } } , { \mathrm { b a l a n c e d } } \}$

• Overall Coronary Calcification Severity $( c _ { p } ) \colon$ The global ordinal severity of calcification for a patient, where $c _ { p } \in \{ 0 \}$ : none,1 : mild,2 : moderate,3 : severe}.

• Maximal Stenosis Severity $( n _ { p } , n _ { \nu } ) \colon$ The highest ordinal grade of stenosis evaluated at both the patient level $( n _ { p } )$ and the individual vessel level $\left( n _ { \nu } \right)$ , where $n _ { p } , n _ { \nu } \in \{ 0$ : none, 1 : mild, 2 : moderate, 3 : severe}.

• Plaque Type $( p _ { \nu } ) \colon$ The multi-label plaque composition at the vessel level, where $p _ { \nu } \subseteq$ {calcified,non-calcified,mixed}.

• Segment-level Semantic Quadruple $( q _ { s } ^ { a } ) \colon$ Each quadruple is defined as $q _ { s } ^ { a } = ( \nu , s , a , e _ { a } )$ , where $\nu \in \mathbb { V }$ indicates the vessel and $s \in \mathbb { S } _ { \nu }$ specifies the corresponding finer-level anatomical segment or branch under that vessel. The abnormality a and its corresponding type or severity $e _ { a }$ are defined via the following conditional mapping:

$$
a \in \{ \mathrm { p l a q u e , s t e n o s i s } \} ,\tag{4}
$$

$$
e _ { a } \in \left\{ \begin{array} { l l } { \mathcal { P } ( \{ \mathrm { c a l c i f i e d , n o n - c a l c i f i e d , m i x e d } \} ) , } & { \mathrm { i f ~ } a = \mathrm { p l a q u e } , } \\ { \{ 0 : \mathrm { n o n e } , 1 : \mathrm { m i l d } , 2 : \mathrm { m o d e r a t e } , 3 : \mathrm { s e v e r e } \} , } & { \mathrm { i f ~ } a = \mathrm { s t e n o s i s } , } \end{array} \right.\tag{5}
$$

where $\mathcal { P } ( \cdot )$ denotes the power set $( i . e .$ , the set of all possible subsets), representing all potential combinations of plaque types.

## Hierarchical Variable Aggregation

$\mathrm { C S M } _ { \mathrm { C C T A } }$ uses Agg(·) to complete missing coarser variables from reported segment level findings. Explicitly extracted variables are retained as the primary evidence. If a coarser variable is not explicitly reported, its value is inferred from the available segment level or vessel level variables. If a variable is neither reported nor derivable from more detailed findings, $\mathrm { C S M _ { C C T A } }$ leaves it uninstantiated. During scoring, an uninstantiated abnormality is counted as no abnormality when the variable is required for comparison.

The aggregation operator is defined only for coarser variables that can be derived from lower anatomical levels. For each vessel v, the segment quadruples in $\gamma _ { \ell _ { s e g } }$ are separated into stenosis and plaque subsets:

$$
\mathcal { Q } _ { \nu } ^ { n } = \{ q _ { s } ^ { a } = ( \nu , s , a , e _ { a } ) \in \mathcal { V } _ { \ell _ { s e g } } \ | \ s \in \mathbb { S } _ { \nu } , a = \mathrm { s t e n o s i s } \} ,\tag{6}
$$

$$
\mathcal { Q } _ { \nu } ^ { p } = \{ q _ { s } ^ { a } = \left( \nu , s , a , e _ { a } \right) \in \mathcal { V } _ { \ell _ { s e g } } \ | \ s \in \mathbb { S } _ { \nu } , \ a = \mathrm { p l a q u e } \} .\tag{7}
$$

Given these subsets, $\operatorname { A g g } ( { \mathord { \cdot } } )$ is specified for each completed variable. For vessel stenosis, $\mathrm { C S M _ { C C T A } }$ takes the maximal stenosis severity among the vessel segments. For patient stenosis, $\mathrm { C S M _ { C C T A } }$ takes the maximal completed vessel stenosis:

$$
\mathrm { A g g } _ { n _ { \nu } } ( Q _ { \nu } ^ { n } ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ e _ { a } \mid q _ { s } ^ { a } = ( \nu , s , a , e _ { a } ) \in Q _ { \nu } ^ { n } \} , } & { Q _ { \nu } ^ { n } \neq \emptyset , } \\ { \emptyset , } & { Q _ { \nu } ^ { n } = \emptyset , } \end{array} \right.\tag{8}
$$

$$
\begin{array} { r } { \mathrm { A g g } _ { n _ { p } } ( \{ \tilde { n } _ { \nu } \} _ { \nu \in \mathbb { V } } ) = \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ \tilde { n } _ { \nu } | \nu \in \mathbb { V } , \tilde { n } _ { \nu } \neq \emptyset \} , } & { \exists \nu , \tilde { n } _ { \nu } \neq \emptyset , } \\ { \emptyset , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{9}
$$

For vessel plaque, $\mathrm { C S M _ { C C T A } }$ takes the union of plaque types over the corresponding segment findings:

$$
\begin{array} { r } { \mathrm { A g g } _ { p _ { \nu } } ( \mathcal { Q } _ { \nu } ^ { p } ) = \left\{ \bigcup \{ e _ { a } \mid q _ { s } ^ { a } = ( \nu , s , a , e _ { a } ) \in \mathcal { Q } _ { \nu } ^ { p } \} , \quad \mathcal { Q } _ { \nu } ^ { p } \neq \emptyset , \right. } \\ { \emptyset , \qquad \left. \mathcal { Q } _ { \nu } ^ { p } = \emptyset . \right. } \end{array}\tag{10}
$$

The completed vessel and patient variables are then obtained by giving priority to explicitly extracted variables:

$$
\begin{array} { r } { \tilde { n } _ { \nu } = \left\{ \begin{array} { l l } { n _ { \nu } , } & { \mathrm { i f } n _ { \nu } \mathrm { i s } \mathrm { e x p l i c i t l y } \mathrm { e x t r a c t e d f r o m } R , } \\ { \mathrm { A g g } _ { n _ { \nu } } ( \mathcal { Q } _ { \nu } ^ { n } ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{11}
$$

$$
\begin{array} { r } { \tilde { p } _ { \nu } = \left\{ \begin{array} { l l } { p _ { \nu } , } & { \mathrm { i f ~ } p _ { \nu } \mathrm { ~ i s ~ e x p l i c i t l y ~ e x t r a c t e d ~ f r o m } R , } \\ { \mathbf { A g g } _ { p _ { \nu } } ( \mathcal { Q } _ { \nu } ^ { p } ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{12}
$$

$$
\begin{array} { r } { \tilde { n } _ { p } = \left\{ \begin{array} { l l } { n _ { p } , } & { \mathrm { i f } n _ { p } \mathrm { i s } \mathrm { e x p l i c i t l y } \mathrm { e x t r a c t e d f r o m } R , } \\ { \mathrm { A g g } _ { n _ { p } } ( \{ \tilde { n } _ { \nu } \} _ { \nu \in \mathbb { V } } ) , } & { \mathrm { o t h e r w i s e } \mathrm { . } } \end{array} \right. } \end{array}\tag{13}
$$

When comparison requires a stenosis label for an absent finding, ∅ is mapped to the normal category 0. This procedure preserves explicitly stated findings while allowing detailed segment descriptions to support the corresponding vessel and patient level variables. As a result, reports with different levels of anatomical detail can be evaluated within the same hierarchical representation (Supplementary Fig. ??).

## Adaptive Scoring across Anatomical Levels

$\mathrm { C S M _ { C C T A } }$ first determines the finest anatomical level represented in the ground-truth and predicted reports. Before selecting the anatomical evaluation level, $\mathrm { C S M } _ { \mathrm { C C T A } }$ verifies that the report pair contains comparable CCTA-related clinical information. A report pair is labeled “Irrelevant” when no valid anatomical level can be established because the model-generated and reference reports contain no comparable CCTA-related clinical information. Let $\mathcal { A } ( R ) \in \mathcal { L }$ denote the finest anatomical level at which report R provides explicit clinical information. The anatomical levels represented in the reports are therefore $\mathcal { A } ( R _ { g t } )$ and $\mathcal { A } ( R _ { p r e d } ) . \mathrm { C S M } _ { \mathrm { C C T A } }$ evaluates the two reports at the coarser of these levels, defined as the selected anatomical evaluation level ℓ<sup>∗</sup>:

$$
\ell ^ { * } = \operatorname* { m i n } _ { prec } \{ \mathcal { A } ( R _ { g t } ) , \mathcal { A } ( R _ { p r e d } ) \} ,\tag{14}
$$

where $\ell _ { p a t } \prec \ell _ { \nu e s } \prec \ell _ { s e g }$ . This rule avoids rewarding a prediction for segment level details when the ground truth is only reported at the vessel level, and it also avoids penalizing a prediction for missing segment level details when the prediction itself only supports a coarser comparison.

For each evaluation level, $\mathrm { C S M _ { C C T A } }$ uses the variables available up to that level. The variable set used for scoring is defined as

$$
\begin{array} { r } { { \cal K } _ { \ell _ { p a t } } = \mathcal { V } _ { \ell _ { p a t } } , } \end{array}\tag{15}
$$

$$
\begin{array} { r } { K _ { \ell _ { \nu e s } } = \mathcal { V } _ { \ell _ { p a t } } \cup \mathcal { V } _ { \ell _ { \nu e s } } , } \end{array}\tag{16}
$$

$$
\mathcal { K } _ { \ell _ { s e g } } = \mathcal { V } _ { \ell _ { p a t } } \cup \mathcal { V } _ { \ell _ { v e s } } \cup \mathcal { V } _ { \ell _ { s e g } } .\tag{17}
$$

![](images/c592decd4795f3338f20793ce5f84b996b69dde4a61d6a2f9382974a98a98a22.jpg)  
Figure 7. Workflow of $\mathbf { C S M _ { C C T A } }$ . Hierarchical clinical variables are extracted from the reference and model-generated reports to determine their anatomical levels and the shared evaluation level. Variable-level F1 scores are then combined using expert-calibrated weights to obtain the final $\mathrm { C S M } _ { \mathrm { C C T A } }$ score.

Thus, patient level scoring uses only patient variables, vessel level scoring uses patient and vessel variables, and segment level scoring uses patient, vessel, and segment variables.

Some variables have multiple anatomical instances. This multiplicity is handled when computing F1, without introducing additional variables. For vessel-level variables, all valid vessel-specific instances are evaluated jointly: $\mathrm { F } 1 ( n _ { \nu } )$ is computed by pooling stenosis matches and errors across the four main vessels, and $\operatorname { F } 1 ( p _ { \nu } )$ is computed analogously for plaque. For segment-level semantic quadruples, $\operatorname { F 1 } ( q _ { s } ^ { a } )$ is computed over the pooled set of segment-level quadruple instances. If a variable has no valid instance for comparison, its F1 is treated as unavailable; the variable is excluded from the weighted sum, and the remaining weights are renormalized within the selected evaluation level. Further implementation details are provided in Supplementary Materials.

Each evaluation level has its own expert-calibrated weight vector over the corresponding variable set:

$$
\mathbf { w } ^ { ( \ell ) } = \{ w _ { k } ^ { ( \ell ) } | k \in \mathcal { K } _ { \ell } \} , \quad w _ { k } ^ { ( \ell ) } \geq 0 , \quad \sum _ { k \in \mathcal { K } _ { \ell } } w _ { k } ^ { ( \ell ) } = 1 .\tag{18}
$$

The estimation of these three weight vectors is described in the following section.

Let $\tau _ { \ell ^ { * } } \subseteq \kappa _ { \ell ^ { * } }$ denote the variables with valid F1 scores in the current comparison. The weights are renormalized over these available variables:

$$
\bar { w } _ { k } ^ { ( \ell ^ { * } ) } = \frac { w _ { k } ^ { ( \ell ^ { * } ) } } { \sum _ { j \in \mathcal { T } _ { \ell ^ { * } } } w _ { j } ^ { ( \ell ^ { * } ) } } , \quad k \in \mathcal { T } _ { \ell ^ { * } } .\tag{19}
$$

The final $\mathrm { C S M _ { C C T A } }$ score is then computed as:

$$
\mathrm { C S M } _ { \mathrm { C C T A } } ( R _ { g t } , R _ { p r e d } ) = \sum _ { k \in \mathcal { T } _ { \ell ^ { * } } } \bar { w } _ { k } ^ { ( \ell ^ { * } ) } \cdot \mathrm { F 1 } ( k _ { g t } , k _ { p r e d } ) .\tag{20}
$$

Along with this score, $\mathrm { C S M _ { C C T A } }$ records $\mathcal { A } ( R _ { g t } )$ and $\mathcal { A } ( R _ { p r e d } )$ to indicate the anatomical levels of the reference and generated reports.

## Expert-Calibrated Weight Estimation

To determine the importance of clinical variables across different evaluation levels as defined in Eq. 18, we employed a data-driven calibration procedure to estimate the weights $w _ { k } ^ { ( \ell ) }$ and align $\mathrm { C S M _ { C C T A } }$ with clinical judgment. This calibration used only the 70-case derivation set from the fine-grained expert scoring task. The non-overlapping 30-case validation set was reserved for the correlation analysis. The expert scoring protocols are provided in Supplementary Materials.

Specifically, we used constrained linear regression without an intercept to model the relationship between the F1 scores of realized clinical variables and the overall scores assigned by experts. Both a non-negativity constraint $( w \geq 0 )$ and a zero-intercept constraint were imposed during optimization. The structured clinical variables were defined at the patient, vessel, and segment levels as described above, with the segment-level component computed from the pooled semantic quadruples $q _ { s } ^ { a }$ We performed 5-fold cross-validation within the 70-case derivation set to assess the stability of the estimated weights at each level. The regression results, estimated weights, and level-wise interpretation are provided in Supplementary Materials.

## Expert Scoring Protocol and Reliability Assessment

Four radiologists evaluated the expert evaluation cohort using a six-question questionnaire that separately assessed plaque and stenosis localization and description accuracy. The complete allocation procedure, questionnaire design, and scoring criteria are provided in Supplementary Materials and Supplementary Table ??. Reliability was assessed using ICC(2,1), Krippendorff’s alpha, pairwise Cohen’s kappa, and score-distribution analyses, with full results reported in Supplementary Fig. ??.

## References

1. Choi, E.-K. et al. Coronary computed tomography angiography as a screening tool for the detection of occult coronary artery disease in asymptomatic individuals. J. Am. Coll. Cardiol. 52, 357–365 (2008).

2. Douglas, P. S. et al. Outcomes of anatomical versus functional testing for coronary artery disease. New Engl. J. Medicine 372, 1291–1300 (2015).

3. Adamson, P. D. et al. Guiding therapy by coronary ct angiography improves outcomes in patients with stable chest pain. J. Am. Coll. Cardiol. 74, 2058–2070 (2019).

4. Serruys, P. W. et al. Coronary computed tomographic angiography for complete assessment of coronary artery disease: Jacc state-of-the-art review. J. Am. Coll. Cardiol. 78, 713–736 (2021).

5. van Velzen, S. G. et al. Deep learning for automatic calcium scoring in ct: validation using multiple cardiac ct and chest ct protocols. Radiology 295, 66–79 (2020).

6. Eng, D. et al. Automated coronary calcium scoring using deep learning with multicenter external validation. NPJ digital medicine 4, 88 (2021).

7. Xu, C. et al. Automatic coronary artery calcium scoring on routine chest computed tomography (ct): comparison of a deep learning algorithm and a dedicated calcium scoring ct. Quant. Imaging Medicine Surg. 12, 2684 (2022).

8. Van Herten, R. L. et al. Automatic coronary artery plaque quantification and cad-rads prediction using mesh priors. IEEE transactions on medical imaging 43, 1272–1283 (2023).

9. Zreik, M. et al. A recurrent cnn for automatic detection and classification of coronary artery plaque and stenosis in coronary ct angiography. IEEE transactions on medical imaging 38, 1588–1598 (2018).

10. Paul, J.-F., Rohnean, A., Giroussens, H., Pressat-Laffouilhere, T. & Wong, T. Evaluation of a deep learning model on coronary ct angiography for automatic stenosis detection. Diagn Interv Imaging 103, 316–323 (2022).

11. Lin, A. et al. Deep learning-enabled coronary ct angiography for plaque and stenosis quantification and cardiac risk prediction: an international multicentre study. The Lancet Digit. Heal. 4, e256–e265 (2022).

12. Cury, R. C. et al. Cad-rads™ 2.0–2022 coronary artery disease-reporting and data system: an expert consensus document of the society of cardiovascular computed tomography (scct), the american college of cardiology (acc), the american college of radiology (acr), and the north america society of cardiovascular imaging (nasci). Cardiovasc. Imaging 15, 1974–2001 (2022).

13. Muscogiuri, G. et al. Performance of a deep learning algorithm for the evaluation of cad-rads classification with ccta. Atherosclerosis 294, 25–32 (2020).

14. Corti, A. et al. Automated cad-rads scoring from multiplanar ccta images using radiomics-driven machine learning. Eur. J. Radiol. 112320 (2025).

15. Miller, R. J. et al. Patient-specific myocardial infarction risk thresholds from ai-enabled coronary plaque analysis. Circ. Cardiovasc. Imaging 17, e016958 (2024).

16. Kim, J. Y. et al. Predicting major adverse cardiac events using deep learning–based coronary artery disease analysis at ct angiography. Radiol. Artif. Intell. 7, e240459 (2025).

17. Wang, Z., Liu, L., Wang, L. & Zhou, L. R2gengpt: Radiology report generation with frozen llms. Meta-Radiology 1, 100033 (2023).

18. Tanida, T., Müller, P., Kaissis, G. & Rueckert, D. Interactive and explainable region-guided radiology report generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7433–7442 (2023).

19. Li, Y. et al. Unify, align and refine: Multi-level semantic alignment for radiology report generation. In Proceedings of the IEEE/CVF international conference on computer vision, 2863–2874 (2023).

20. Li, S. et al. An organ-aware diagnosis framework for radiology report generation. IEEE Transactions on Med. Imaging 43, 4253–4265 (2024).

21. Zhou, W. et al. Transferring pre-trained large language-image model for medical image captioning. In CLEF (Working Notes), 1776–1784 (2023).

22. Wu, S. et al. Maken: Improving medical report generation with adapter tuning and knowledge enhancement in visionlanguage foundation models. In 2024 IEEE international symposium on biomedical imaging (ISBI), 1–5 (IEEE, 2024).

23. Yang, B., Yu, Y., Zou, Y. & Zhang, T. Pclmed: Champion solution for imageclefmedical 2024 caption prediction challenge via medical vision-language foundation models (2024).

24. Li, C.-Y. et al. Towards a holistic framework for multimodal llm in 3d brain ct radiology report generation. Nat. Commun. 16, 2258 (2025).

25. Zhong, Z. et al. Vision-language model for report generation and outcome prediction in ct pulmonary angiogram. NPJ Digit. Medicine 8, 432 (2025).

26. Hamamci, I. E. et al. Developing generalist foundation models from a multimodal dataset for 3d computed tomography. arXiv preprint arXiv:2403.17834 (2024).

27. Bai, F., Du, Y., Huang, T., Meng, M. Q.-H. & Zhao, B. M3d: Advancing 3d medical image analysis with multi-modal large language models. arXiv preprint arXiv:2404.00578 (2024).

28. Wu, C. et al. Towards generalist foundation model for radiology by leveraging web-scale 2d&3d medical data. Nat. Commun. 16, 7866 (2025).

29. Ye, Z. et al. C2rg: Parameter-efficient adaptation of 3d vision and language foundation model for coronary cta report generation. In 2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2780–2786 (IEEE, 2024).

30. Papineni, K., Roukos, S., Ward, T. & Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th annual meeting ofthe Associationfor Computational Linguistics, 311–318 (2002).

31. Lin, C.-Y. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, 74–81 (2004).

32. Banerjee, S. & Lavie, A. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 65–72 (2005).

33. Vedantam, R., Lawrence Zitnick, C. & Parikh, D. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4566–4575 (2015).

34. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q. & Artzi, Y. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675 (2019).

35. Yu, F. et al. Evaluating progress in automatic chest x-ray radiology report generation. Patterns 4 (2023).

36. Johnson, A. E. et al. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Sci. data 6, 317 (2019).

37. Zhao, W. et al. Ratescore: A metric for radiology report generation. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, 15004–15019 (2024).

38. Ostmeier, S. et al. Green: Generative radiology report evaluation and error notation. In Findings of the association for computational linguistics: EMNLP 2024, 374–390 (2024).

39. Jiang, S. et al. Hulu-med: A transparent generalist model towards holistic medical vision-language understanding. arXiv preprint arXiv:2510.08668 (2025).

40. Blankemeier, L. et al. Merlin: a computed tomography vision–language foundation model and dataset. Nature 1–11 (2026).

41. Leipsic, J. et al. Scct guidelines for the interpretation and reporting of coronary ct angiography: a report of the society of cardiovascular computed tomography guidelines committee. J. cardiovascular computed tomography 8, 342–358 (2014).