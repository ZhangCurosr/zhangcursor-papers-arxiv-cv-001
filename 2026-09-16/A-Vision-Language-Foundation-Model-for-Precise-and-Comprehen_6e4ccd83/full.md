# A Vision-Language Foundation Model for Precise and Comprehensive Brain Tumor Diagnosis from Preoperative Multimodal Data

Yinong Wang <sup>†</sup> <sup>a</sup>, Jianwen Chen <sup>†a</sup>, Zhou Chen, MD <sup>†b</sup>, Shuwen Kuang, MD <sup>†c</sup>, Haoning Jiang <sup>a</sup>, Yanzhao Shi <sup>v</sup>, Huichun Yuan, MD <sup>d</sup>, Yan-ran (Joyce) Wang, PhD <sup>e</sup>, Bing Wang, MD <sup>g</sup>, Lei Wu, MD <sup>h</sup>, Bin Tang, MD <sup>i</sup>, Li Meng, MD <sup>j</sup>, Baihua Luo, MD <sup>k</sup>, Bin Zhou, MD <sup>f,l</sup>, Wei Ding, MD <sup>m</sup>, Weiming Zhong, MD <sup>n</sup>, Wei Hou, MD <sup>o</sup>, Yuanbing Chen, MD <sup>p</sup>, Zhiping Wan, MD <sup>q</sup>, Wei Wang, MD <sup>r</sup>, Zhenkun Xiao, MD <sup>g</sup>, Wenwu Wan, MD <sup>s</sup>, Allen He<sup>t</sup>, Yuyin Zhou, PhD <sup>u</sup>, Prof Longbo Zhang, MD <sup>b,f,x</sup>, Feifei Wang, PhD <sup>v</sup>, Prof Zhixiong Liu, MD <sup>b,f,x</sup>, Prof Michael Iv, MD <sup>w</sup>, Xuan Gong, MD PhD <sup>b,f,x,\*</sup>, Liangqiong Qu, PhD

<sup>a</sup> School of Computing and Data Science, The University of Hong Kong, China

<sup>b</sup> Department of Neurosurgery, Xiangya Hospital, Central South University, China <sup>c</sup> Department of Oncology, Xiangya Hospital, Central South University, China <sup>d</sup> Changde Hospital, Xiangya School of Medicine, Central South University (The First People’s Hospital of Changde City), China

<sup>e</sup> Department of Biomedical Data Science, School of Medicine, Stanford University, Stanford, USA

<sup>f</sup> Department of Neurosurgery, Xiangya Hospital, Central South University, Jiangxi (National Regional Center for Neurological Diseases), China

<sup>g</sup> Department of Neurosurgery, The Second Affiliated Hospital, Hengyang Medical School, University of South China, China

<sup>h</sup> Department of Neurosurgery, The Second Affiliated Hospital, Jiangxi Medical College, Nanchang University, China

<sup>i</sup> Department of Neurosurgery, The First Affiliated Hospital, Jiangxi Medical College, Nanchang University, China

<sup>j</sup> Department of Radiology, Xiangya Hospital, Central South University, China

<sup>k</sup> Department of Pathology, Xiangya Hospital, Central South University, China

<sup>l</sup> Department of Neurosurgery, Jiangxi Provincial People’s Hospital, The First Affiliated Hospital of Nanchang Medical College, China

<sup>m</sup> The Affiliated Children’s Hospital of Xiangya School of Medicine, Hunan Children’s Hospital, China

<sup>n</sup> Department of Neurosurgery, Shenzhen Second People’s Hospital, China

Department of Neurosurgery, First Hospital of Lanzhou University, China

<sup>p</sup> Department of Neurosurgery, The Third Xiangya Hospital, Central South University, China

Department of Neurosurgery, Tongji Hospital, School of Medicine, Tongji University, China <sup>r</sup> Department of Radiology, Tongji Hospital, School of Medicine, Tongji University, China <sup>s</sup> Department of Neurosurgery, Chongqing Traditional Chinese Medicine Hospital, China <sup>t</sup> Basis International School Park Lane Harbour, China

<sup>u</sup> Department of Computer Science and Engineering, University of California, Santa Cruz, USA

<sup>v</sup> Department of Electrical and Electronic Engineering, The University of Hong Kong, China

<sup>x</sup> National Clinical Research Center of Geriatric Disorders, Xiangya Hospital, Central South University, China

## Summary

Background Non-invasive presurgical diagnosis of brain tumor types from Magnetic Resonance Imaging (MRI) is essential but challenging due to overlapping imaging features across tumor types, inter-observer variability, and the extensive training required for expertise. We aimed to develop an MRI-based Artificial Intelligence (AI) model for automatic and reliable brain tumor classification with diagnostic uncertainty quantification and radiology reports generation.

Methods We developed BrainVLM to classify all 12 World Health Organization (WHO) 2021 brain tumor types. BrainVLM integrates an uncertainty quantification strategy to indicate prediction reliability and a module for generating radiology reports to elucidate the clinical rationale. BrainVLM was trained on multi-modal data (MRI scans, demographics, and radiology reports) from 40,043 individuals. It was validated on 5,211 patients with pathologically confirmed brain tumors, including 3,877 held-out patients from the primary hospital and 1,334 patients from 11 independent hospitals. We further conducted two proof-ofconcept studies to validate its clinical utility in AI-clinician workflows: 1) a blinded multireader study where 12 neuroradiologists across varying experience levels interpreted 248 retrospective cases with or without AI assistance, and 2) a real-world prospective study in which 1,009 patients were independently and blindly assessed by BrainVLM and radiologists before surgery. Additionally, we demonstrated BrainVLM’s utility in preoperative molecular subgroup prediction for adult-type diffuse gliomas, using a multi-center cohort of 632 patients.

Findings In primary evaluation, BrainVLM achieved an area under the curve (macro-AUC) of 0.85 (95% CI: 0.84–0.86), and an F1 score of 0.82 (95% CI: 0.81–0.83), surpassing neuroradiologists (F1 = 0.80 (95% CI: 0.79–0.81)). In external validation across 11 centers, BrainVLM achieved an AUC = 0.80 (95% CI: 0.79–0.82) and F1 = 0.75 (95% CI: 0.73–0.78), compared with F1=0.71 (95% CI: 0.69-0.73) for neuroradiologists. In prospective real-world evaluation, BrainVLM maintained performance comparable to neuroradiologists. In the blinded multi-reader study, the AI-assisted neuroradiologists demonstrated a statistically 27.6% (absolute 0.16, 95% CI: 20%-35%) improvement in F1 score across all experience levels compared to unaided assessments (p<0.0001), while reducing diagnostic time by 34.7% (absolute 55.8 seconds, 95% CI: 29.9%-38.9%). Across 5,211 cases, 72% (3,008/4,156) of correct diagnoses had high-confidence scores (>90%), whereas 70% (736/1,055) of incorrect diagnoses exhibited low-confidence scores (50–85%). For molecular subgroup prediction in adult-type diffuse gliomas, BrainVLM achieved AUCs of 0.95 (95% CI: 0.93-0.96) and 0.88 (95% CI: 0.84-0.91) in primary and external validation, respectively.

Interpretations By integrating uncertainty quantification and report generation, BrainVLM improves radiologists’ diagnostic performance, reduces MRI interpretation time, and allows lower-confidence cases to be effectively flagged for human review. These findings show that AI-assisted diagnosis based on presurgical MRI can support clinical decision-making, with the potential to enhance early brain tumor diagnosis, alleviate clinicians’ workloads, and ultimately improve patient care.

## Research in context

## Evidence before this study

We searched PubMed and Google Scholar for English-language studies published up to December 6, 2025, using the terms “MRI” AND (“brain tumor” OR “CNS tumor” OR “brain cancer”) AND (“machine learning” OR “artificial intelligence” OR “deep learning” OR “foundation model”) in titles or abstracts. We identified no reports describing the deployment of an MRI-based AI model capable of diagnosing the full spectrum of the 12 major brain tumor types defined by WHO CNS5 (the 2021 fifth edition of the WHO Classification of Tumors of the Central Nervous System). Existing literature on AI-assisted brain tumor diagnosis focused almost exclusively on gliomas, meningiomas, and brain metastases, with little attention to other WHO CNS tumor types. Critically, to our knowledge, no previous work has rigorously quantified model reliability or provided clinically interpretable diagnostic rationales (such as radiology style reports) alongside brain tumor classifications. As a result, existing AI outputs for brain tumor classification are limited to isolated predictions, lacking the clinical narrative components that are imperative for clinician trust and real-world adoption.

## Added value of this study

To our knowledge, this is the first study to develop and validate an AI model capable of comprehensive classification across the full spectrum of all 12 major brain tumor types from preoperative multimodal data. By leveraging the largest and most diverse multi-modal neurooncology dataset to date, our model achieved robust diagnostic performance with high generalizability, superior or comparable to board-certified neuroradiologists in sensitivity, precision, and F1 scores on both internal and multi-center external validation cohorts. By integrating diagnostic precision with automatic radiology reports generation and reliable uncertainty quantification, our BrainVLM could substantially improve neuroradiologists’ diagnostic accuracy and efficiency across a wide spectrum of tumor types, supporting neuroradiologists at all levels of experience. We publicly release our source code and our training dataset to promote transparency, reproducibility, and future research in the field of neuro-oncology AI.

## Implications of all the available evidence

Use of AI foundation models, when trained on sufficiently large and diverse datasets, has the potential to substantially improve the accuracy and efficiency of preoperative tumor diagnosis across a broad spectrum of tumor types, benefiting radiologists at all levels of experience. Reliable uncertainty quantification and automated report generation enhance the interpretability and clinical utility of AI predictions, helping to build clinician trust and streamline decisionmaking. Prospective clinical studies are warranted to confirm the real-world benefits and safety of integrating AI-assisted diagnostics into standard neuro-oncological management.

## Introduction

Brain neoplasms can have long-lasting and life-altering physical, cognitive, and psychological impacts on patients’ lives, and can occur in any anatomical region of the encephalon and adjacent structures.<sup>1–3</sup> WHO CNS5 identifies 12 major brain tumor types and over 100 distinct tumor subtypes, ranging from benign and indolent neoplasms such as grade 1 meningioma to malignant and aggressive tumors such as grade 4 glioblastoma.<sup>4</sup> Common treatment options for brain tumors include neurosurgical resection, radiotherapy, and chemotherapy, with surgery serving as the primary initial management in most cases. However, the choice of treatment modality, surgical approach, and the extent of resection depends largely on the initial presurgical diagnosis.<sup>5</sup> For example, maximal safe gross total resection is generally recommended for patients with diffuse gliomas. In contrast, lymphomas and some germ cell tumors may be managed effectively with chemotherapy and/or radiotherapy without surgical resection.<sup>6,7</sup> Hence, a prompt and accurate preoperative diagnosis of brain tumors is an important first step in guiding appropriate treatment.

MRI is the primary diagnostic modality for presurgical diagnosis and longitudinal monitoring of patients with brain tumors.<sup>8</sup> In routine practice, neuroradiologists must review multi-parametric MRI scans to identify and diagnose potential lesions and then convey their findings and impression in a radiology report. However, this manual interpretation is timeconsuming and error-prone due to overlapping radiologic characteristics of different brain tumors, inter-observer variability, and potential perceptual errors. This problem is further compounded by the global shortage of specialist neuroradiologists due to the extensive training efforts required to gain expertise. Together, these unmet needs call for advanced techniques to develop an automatic brain tumor diagnostic tool for rapid and precise diagnoses and report generation, thereby accelerating the time-sensitive diagnosis process and reducing clinician workload.

AI, particularly the recently emerged vision language models (VLMs),<sup>9,10</sup> has the potential to advance automatic disease diagnosis. Yet, compared to the flourishing development of AI diagnostic frameworks for other tumors, comprehensive AI models for presurgical brain tumor diagnosis remain significantly underexplored. At least two critical challenges hinder the widespread clinical adoption of AI-based brain tumor diagnosis. First, current publicly available datasets for brain tumor analysis are limited in scope, predominantly featuring gliomas<sup>11</sup>, meningiomas, and metastases, while underrepresenting other tumors in the WHO CNS5.<sup>12</sup> Second, existing AI models predict brain tumor types without assessing the confidence and reliability of the prediction or explaining the underlying clinical rationale. This issue is particularly pronounced in large language models (LLMs) or VLMs, which often exhibit unwarranted high confidence in erroneous predictions.<sup>13</sup> This overconfidence in AI predictions, coupled with the opaque decision-making process, undermines clinicians’ trust and poses a substantial barrier to their clinical translation and adoption.

In this study, we developed BrainVLM, an MRI-based AI model for precise and automated classification of all 12 WHO CNS5-defined brain tumor categories. For each case, BrainVLM outputs a diagnosis, a confidence score to quantify diagnostic reliability, and a radiology report to explain the clinical rationale. We compared BrainVLM’s performance against board-certified neuroradiologists and state-of-the-art AI models, using pathological diagnoses and radiology reports as reference standards. Furthermore, we validated its clinical utility in two proof-ofconcept AI–clinician workflow studies: 1) a blinded multi-reader study in which 12 neuroradiologists of varying experience levels interpreted 248 retrospective cases with and without AI assistance, and 2) a real-world prospective study in which 1009 patients were assessed simultaneously by BrainVLM and radiologists prior to surgery. Finally, we demonstrate BrainVLM’s utility in preoperative molecular subgroup prediction for adult-type diffuse gliomas using multi-center cohorts.

## Methods

## Datasets and study design

For the development of BrainVLM, we curated BrainTumor48K, a comprehensive multi-modal brain tumor dataset. It comprises 47,947 individuals aggregated from 39 public repositories and 12 collaborating medical centers (Figure 1a, Table 1, and Supplementary Tables 1-4, appendix pp 46-50). The BrainTumor48K comprises 33,149 patients with pathologically confirmed brain tumors (21,993 from public repositories and 11,156 from medical centers) and 14,798 healthy controls (sourced from public repositories) (Figure 1a). The detailed public repositories and their pre-processing pipeline for the public repositories are summarized in the Supplementary Section S.1.1.1 (appendix pp 2-3). Data from 12 independent collaborating medical centers included demographics, 3D multi-parametric MRI scans (T1-weighted (T1), T1 contrastenhanced (T1c), T2-weighted (T2), and T2-Flair sequences (T2f)), expert-curated radiology reports, and pathological diagnoses.

Overall, the BrainTumor48K dataset spans all 12 major brain tumor types and most of the subtypes defined by the 2021 WHO CNS5.<sup>4</sup> The detailed number of patients across the 12 major brain tumor types and their subtypes is shown in the right panel of Figure 1a.

Data from participating medical institutions were obtained with ethics approval from their respective Institutional Review Boards (IRBs), and the details of these IRBs are presented in the Supplementary Section S1.3 (appendix pp 6-7). The requirement for informed consent was waived by the IRBs. All data were de-identified in compliance with institutional policies and ethical standards. This study complies with the TRIPOD+AI guidelines for the reporting artificial intelligence-based prediction models.<sup>14</sup>

## Development of AI foundation model and retrospective validation

BrainVLM (Figure 1b) was trained on a dataset aggregating 35,227 publicly available cases (including patients with brain tumors and healthy controls) and 4,816 patients from the primary institutional cohort (Xiangya Hospital). A validation set, consisting of 500 public cases and 120 primary cohort cases, was reserved for hyperparameter tuning and model selection. Retrospective diagnostic performance was evaluated on 5,211 patients with pathologically confirmed brain tumors, including 3,877 held-out patients from the primary hospital, and 1,334 patients from 11 independent hospitals (Figure 1a). More details regarding the BrainVLM network architecture, uncertainty quantification, training strategies and validation procedures can be found in the Supplementary Sections S2 (appendix pp 8-25).

## Prospective validation procedures

For the prospective study, we consecutively enrolled 1162 patients with an initial diagnosis of brain lesions at Xiangya Hospital and additional 349 consecutive patients from two external institutional cohorts (Supplementary Tables 3, 4, appendix pp 49-50). Following brain MRI acquisition, clinical information and MRI images were collected and preprocessed according to standardized protocols. Patients were excluded if they had a history of prior brain surgery, previous brain radiotherapy or stereotactic radiosurgery, missing MRI images, or incomplete MRI protocols. Following the application of these criteria (Figure 2c), eligible patients were divided into two groups: surgical group (n = 886; 639 Xiangya, 247 external) and non-operative group (n = 123; 74 Xiangya, 49 external). Eligible patients’ data were prospectively predicted using BrainVLM prior to definitive clinical diagnosis. Following patient discharge, BrainVLM outputs were validated against gold standard references: postoperative pathological findings for patients who underwent surgery, or consensus discharge diagnoses established by a multidisciplinary panel of senior clinicians through comprehensive review of radiological and clinical data for non-surgical cases. Finally, patients with non-tumoral pathological findings were also excluded.

## Multi-reader study for AI-augmented clinical assessments

Multi-reader studies are widely adopted to assess the clinical validity of medical AI models.<sup>15</sup> To achieve a more rigorous and realistic evaluation of BrainVLM’s clinical utility in augmenting neuroradiological diagnosis, we developed a gold-standard test dataset comprising 248 patients covering 12 major WHO CNS5 brain tumor types. For each case, readers received only the multi-parametric MRI scans (T1, T1c, T2, T2-Flair) and patient metadata (age, sex); no clinical history was provided. Diagnostic performance was systematically compared across three paradigms: (a) neuroradiologists-only as a reader, (b) BrainVLM as an autonomous diagnostic agent, and (c) neuroradiologists augmented by BrainVLM as a reader. In paradigm (c), BrainVLM assisted neuroradiologists by providing diagnoses with associated confidence scores and a radiology report. Twelve neuroradiologists, stratified by expertise (5 juniors: 3-5 years; 4 seniors: 5-10 years; 3 experts: 10+ years), were recruited in the blinded crossover study. For each case interpretation, neuroradiologists recorded a diagnosis and a self-rated diagnostic confidence score. To minimize memorization effects in scenarios (a) and (c), we reshuffled the order of the cases and modified their anonymized identifiers for the second read, and allowed a one-month wash-out period between the two reads.

## Statistical analysis

BrainVLM was compared against board-certified neuroradiologists (using clinical radiology reports) and four AI models, with postoperative pathology as the gold standard. RadFM,<sup>16</sup> Merlin,<sup>17</sup> and ${ \mathrm { V S T ^ { 1 8 } } }$ were fine-tuned on BrainTumor48K, while ChatGPT-4o<sup>19</sup> served as a zeroshot generalist baseline. Tumor classification performance was quantified using sensitivity, specificity, precision, Cohen’s Kappa, F1, and the area under the curve (AUC). For multi-label classification, we computed the class-frequency-weighted F1 and both macro- and microaveraged AUC. Report generation was assessed with BLEU-4<sup>20</sup>, RaTEScore<sup>21</sup>, RadGraph-XL<sup>22</sup>, and an LLM-as-Judge framework (Qwen 2.5-72B). We applied two-tailed Wilcoxon signedrank tests to assess statistical significance and reported 95% confidence intervals. Following previous work, we applied non-parametric bootstrap resampling (1,000 replicates) to estimate 95% $\mathrm { C I s } . ^ { 2 3 }$ P values were not adjusted for multiple comparisons. The performance metrics in the benchmark analyses (Figures 1d and 3d) and tumor categories in the multi-reader study (Figures 4a-d) were interpreted as distinct analytical objectives, rather than repeated tests of a single hypothesis. Therefore, no Bonferroni or other multiplicity correction was applied. We performed Shapley value analysis on the primary and external test datasets to evaluate

BrainVLM’s robustness to missing MRI sequences (Supplementary Section S3.2, appendix pp 25-26).

## Role of funding source

The funders of this study had no role in data collection, analysis, interpretation, writing of the manuscript, and the decision to submit the manuscript for publication.

## Results

## Diagnostic performance of BrainVLM in primary and external test datasets

We first evaluated BrainVLM’s diagnostic ability using the retrospective primary test dataset $( \mathrm { n } = 3 , 8 7 7 )$ . BrainVLM achieved a macro-AUC of 0.85 (95% CI: 0.84–0.86, Figure 1c; 0.87 for both intra-axial and extra-axial tumors, Supplementary Figure 10b, appendix p 65) and a weighted-average F1 score of 0.82 (95% CI: 0.81–0.83), with precision of 0.84 (95% CI: 0.83– 0.85) and sensitivity of 0.82 (95% CI: 0.81–0.83; Figure 1d). This exceeded expert neuroradiologists’ assessments (F1 = 0.80, 95% CI: 0.79–0.81; precision = 0.71, 95% CI: 0.70– 0.73; sensitivity = 0.80, 95% CI: 0.79–0.81), with enhanced inter-rater agreement (Cohen’s κ = 0.75, 95% CI: 0.74–0.76 vs 0.69, 95% CI: 0.68–0.70). In benchmarking against four comparator AI models (RadFM, Merlin, VST, and ChatGPT-4o), BrainVLM improved key performance metrics by at least 25%, including in several low-prevalence tumor types (Figure 1d; Supplementary Table 11, appendix p 56). DeLong tests showed that BrainVLM had a higher AUC than each comparison model (Supplementary Table 5 and Supplementary Figure 10, appendix pp 51, 65).

In the external multicentric dataset, BrainVLM maintained robust performance. It achieved a macro-AUC of 0.80 (95% CI: 0.79–0.82, Figure 1c; 0.76 for intra-axial and 0.84 for extra-axial tumors, Supplementary Figure 10b, appendix p 65) and an F1 score of 0.75 (95% CI: 0.73–0.78; precision = 0.77, 95% CI: 0.75–0.78; sensitivity = 0.75, 95% CI: 0.74–0.77; Figure 1d). These results also surpassed both neuroradiologist consensus (F1 = 0.71, precision = 0.72, sensitivity = 0.71) and state-of-the-art models (RadFM: F1 = 0.43, Merlin: F1 = 0.52, VST: F1 = 0.41, ChatGPT-4o: F1 = 0.30).

BrainVLM exhibited slight performance variation across different tumor types (Figure 2a). In the primary test dataset, BrainVLM matched or surpassed expert-level neuroradiologists performance for common brain tumors: MEN (F1 = 0.91 vs 0.91), GGN $( \mathrm { F 1 } = 0 . 8 2 \ \mathrm { v s } \ 0 . 8 1 )$ , and CPN $( \mathrm { F 1 } ~ = ~ 0 . 7 8 ~ \mathrm { v s } ~ 0 . 7 4 )$ . For low-prevalence or imaging-ambiguous brain tumors, BrainVLM showed superiority: HEM $( \mathrm { F } 1 = 0 . 6 4 \ \mathrm { v s } \ 0 . 4 5 )$ , EMB $( \mathrm { F 1 } = 0 . 6 8 \ \mathrm { v s } \ 0 . 5 8 )$ , and CPT $( \mathrm { F } 1 = 0 . 5 5 \mathrm { v s } 0 . 4 7 )$ . Confusion matrix analysis demonstrated substantial concordance between BrainVLM and human experts in diagnostic errors, with over 90% overlap in the top two misclassified tumor categories across both the primary test and external datasets (Figure 2b).

Notably, BrainVLM achieved robust performance using only standard MRI sequences (T1, T1c, T2, T2-Flair) and demographic data, whereas neuroradiologists relied on supplemental advanced imaging (e.g., diffusion-weighted imaging, MR angiography, MR spectroscopy, and MR Perfusion) in 56.1% of cases (Supplementary Table 7, appendix p 53). Together, these results establish BrainVLM as an innovative AI system that achieves neuroradiologist-level performance on most brain tumor types across heterogeneous clinical settings without relying on advanced imaging or protocol harmonization.

## Real-world prospective study

To further evaluate the utility of BrainVLM in real-world clinical settings, we conducted a multi-center prospective study aimed at assessing its performance in authentic diagnostic workflows. In the surgical group, BrainVLM achieved F1 scores of 0.78 (primary) and 0.80 (external), surpassing the radiologists’ performance of 0.75 and 0.77, respectively (Figure 3a). In the non-operative group, the model maintained robust diagnostic performance, yielding F1 scores of 0.85 in the primary cohort and 0.75 in the external cohort (Figure 3a), both exceeding the neuroradiologists’ benchmarks (0.79 and 0.72, respectively). These results demonstrate that BrainVLM provides reliable diagnostic support across diverse clinical management pathways and independent patient populations.

## Uncertainty-aware clinical decision support

Conventional VLMs can generate incorrect diagnoses with inappropriately high confidence or fail to provide uncertainty quantification, potentially undermining clinicians’ trust.<sup>24</sup> To address this issue, we developed a consensus-driven strategy to augment BrainVLM with reliable uncertainty quantification (Supplementary Section S2.2, appendix pp 16-18). Across 5,211 cases in the primary and external test datasets, 72% of correct diagnoses (n = 3,008/4,156) were associated with high-confidence (>90%), whereas 70% of incorrect diagnoses (n = 736/1,055) exhibited confidence scores of 50-85%. By contrast, the corresponding model without the consensus-driven strategy assigned low confidence to only 15% of incorrect predictions (n = 160/1,055; Figure 3c). Formal calibration analysis showed good agreement between predicted confidence and observed outcomes, with an expected calibration error (ECE)<sup>25</sup> of 0.036 and a Brier score<sup>26</sup> of 0.1448 (Figure 3b). This was further supported by the reliability diagram, in which observed diagnostic accuracy increased with predicted confidence (Figure 3b and Figure 3c).

We next examined whether these uncertainty estimates could inform decision support in diagnostically ambiguous cases. Cases assigned lower confidence by BrainVLM were associated with lower inter-reader agreement among neuroradiologists (Supplementary Section S3.8.2, appendix p 31), supporting the clinical relevance of the model’s uncertainty estimates. We therefore implemented a confidence-triggered supplementary diagnosis mechanism in BrainVLM (Supplementary Section S2.3.1, appendix p 21): when the confidence of the topranked diagnosis (Top 1) was below 75%, BrainVLM additionally presented the second-ranked diagnosis (Top 2, Supplementary Figure 9a and d, appendix p 64) for radiologist review. Under this strategy, overall F1 improved from 0.82 (95% CI: 0.81–0.83) to 0.86 (95% CI: 0.85–0.87) in the primary dataset and from 0.75 (95% CI: 0.73–0.78) to 0.78 (95% CI: 0.75–0.80) in the external dataset (Supplementary Section 3.1, appendix p 25).

## Radiology report generation

We next evaluated BrainVLM’s ability to generate radiology reports to accompany its diagnostic predictions. We compared BrainVLM-generated reports with those produced by other VLMs. BrainVLM consistently outperformed all baselines across both primary and external test datasets (Figure 3d). In the primary dataset, BrainVLM achieved superior BLEU-4 (0.43 vs 0.02–0.38), F1RadGraph-XL (0.57 vs 0.15–0.46), and RaTEScore (0.75 vs 0.47–

0.63). Similarly, in the external dataset, BrainVLM maintained performance, with BLEU-4 (0.35 vs 0.02–0.29), F1RadGraph-XL (0.52 vs 0.22–0.42), and RaTEScore (0.69 vs 0.42–0.58). We further compared BrainVLM with other VLMs on two key components of MRI reporting: signal intensity and tumor location. BrainVLM achieved report accuracies of 0.86 for contrast enhancement, 0.80 for T1 signal intensity, and 0.81 for T2 signal intensity, with a lesion localization accuracy of 0.72; these values were higher than those of comparator VLMs (Figure 3f, Supplementary Section S3.4, Supplementary Figure 7d, appendix pp 27, 63).

## Preoperative molecular subgroup prediction in adult-type diffuse gliomas

Adult-type diffuse gliomas constitute the predominant malignant tumors affecting the central nervous system,<sup>27</sup> comprising three main subtypes: IDH-mutant astrocytoma, IDH-mutant oligodendroglioma, and IDH-wildtype glioblastoma.<sup>4</sup> Accurate distinction among these subtypes is essential for optimizing treatment strategies and predicting patient outcomes.<sup>28</sup> We finetuned BrainVLM on available adult-type diffuse glioma cases from BrainTumor48K, with threefold cross-validation on the primary dataset (n = 494) and external validation on an independent multi-center cohort (n = 138, subtype distributions in Supplementary Section 2.4, appendix p 21). In primary dataset, BrainVLM achieved a macro-averaged AUC of 0.95 (95% CI: 0.93-0.96; Figure 3e), surpassing Merlin (AUC 0.85, 95% CI: 0.82-0.86) and RadFM (AUC 0.76, 95% CI: 0.74-0.78). In the external cohort, BrainVLM maintained robust performance (AUC 0.88, 95% CI: 0.84-0.91), substantially exceeding Merlin (AUC 0.73, 95% CI: 0.69-0.77) and RadFM (AUC 0.63, 95% CI: 0.58-0.67). These results underscore the exceptional performance of BrainVLM for molecular subtyping, highlighting its potential as a versatile foundation model for brain tumor classification.

## AI-augmented clinical assessments

In the 248-case multi-reader study, BrainVLM, when evaluated independently, outperformed junior and senior neuroradiologists across brain tumor types and performed comparably with expert neuroradiologists (Figure 4a-d and Table 2). With BrainVLM assistance, neuroradiologists’ performance improved across all seniority groups, with a 27.6% increase in mean F1 score and a 34.7% reduction in diagnostic time compared with unaided assessment (both p<0.0001). F1 scores increased from 0.52 to 0.67 in junior neuroradiologists, from 0.55 to 0.74 in senior neuroradiologists, and from 0.79 to 0.85 in expert neuroradiologists (Table 2), bringing junior and senior readers closer to unaided expert-level performance.

To characterize clinician–AI interaction in diagnostically challenging settings, we analyzed cases in the lowest quartile of mean reader-reported confidence during unaided assessment (Supplementary Section S3.10, appendix pp 34-35). In this 25% low-confidence subset, the most common diagnostic trajectory across all reader seniority groups was trajectory F (initial clinician and BrainVLM both correct, final diagnosis remains correct; Figure 4f). Trajectory B (initially incorrect diagnosis corrected by a correct AI suggestion) was also frequent, 15.2% / 27.5% / 29.2% of junior / senior / expert assessments. By contrast, trajectory C (initially correct diagnosis reversed by an incorrect AI suggestion) was the least frequent outcome across groups (1.3%, 1.0%, and 0%, respectively; Figure 4f).

To further assess reader responses to incorrect AI outputs, we restricted analysis to cases in which BrainVLM generated an incorrect diagnosis (n = 52 of 248; Supplementary Section

S3.10, appendix pp 34-35). In this subset, overall accuracy was similar before and after AI assistance (50.0% vs 50.9%), with performance maintained or slightly improved in senior and expert readers and a modest decrease in juniors (Figure 4g). This slightly improved performance maybe because many incorrect BrainVLM predictions were associated with low reliability scores (Figure 4e), which may have prompted readers to apply more caution when reviewing these outputs. Overall, these findings do not suggest systematic over-reliance on incorrect AI outputs, although junior neuroradiologists appeared more susceptible to adverse AI influence.

## Discussion

In this study, we developed and clinically evaluated BrainVLM, a multimodal vision-language foundation model for presurgical classification of brain tumors across all 12 WHO CNS5- defined tumor types. Three findings are particularly noteworthy. First, BrainVLM showed strong and generalizable diagnostic performance in retrospective evaluation, achieving AUCs of 0.85 in the primary validation set and 0.80 in the external validation set, with sensitivity, precision, and F1 scores broadly comparable to those of board-certified neuroradiologists across most tumor types. In subgroup analysis (appendix pp 31-33), performance was broadly consistent across sex, MRI vendor, ethnicity, and most age subgroups. Second, in prospective real-world evaluation, BrainVLM achieved a statistically significant 4% higher F1 score than neuroradiologists, suggesting potential clinical utility beyond retrospective benchmarking. Third, the model extended image-based tumor classification towards molecular stratification, achieving a superior AUC on external validation (0.88) relative to state-of-the-art VLMs (0.63– 0.73). Taken together, these findings suggest that multimodal foundation models could support diagnostically demanding tasks in neuro-oncology, where overlapping MRI phenotypes, interobserver variability, and limited familiarity with rare entities continue to constrain consistent presurgical diagnosis.<sup>29</sup>

The robust diagnostic performance of BrainVLM is likely attributable to the scale and diversity of its training dataset, and to its multimodal design. BrainTumor48K represents, to our knowledge, one of the largest and most comprehensive multi-modal presurgical datasets utilized for model development in neuro-oncology to date. Pretraining on extensive corpora comprising public repositories and real-world clinical data might have enabled BrainVLM to accommodate variability in imaging quality, acquisition parameters, patient populations, imaging equipment, and institutional protocols. Another key factor contributing to BrainVLM’s performance is its efficient integration of multi-modal data, including MRI scans, patients’ demographics, and radiological reports. This framework has enabled us to leverage critical clinical information such as age and radiology reports, leading to more accurate and comprehensive diagnoses.

AI-clinician collaboration holds significant promise in medical image interpretation.<sup>30–33</sup> However, even when AI models exhibit robust diagnostic performance and generalizability, translating these capabilities into clinical practice necessitates overcoming a critical barrier: clinician trust in AI-driven decision-making. The effective integration ofAI models into clinical workflows requires clinicians to (1) understand the AI’s decision rationale to inform their judgments, and (2) develop sufficient trust in the system’s recommendations. To address these challenges, BrainVLM introduces two clinician-trust-focused innovations: a consensus-driven uncertainty quantification strategy and an automated radiology report generation module. In our evaluation, BrainVLM demonstrated clinically meaningful self-assessment capabilities, assigning high-confidence probabilities (>90%) to 72% of its correct diagnoses while appropriately tagging 70% of its incorrect diagnoses with lower confidence scores 50-85%. This self-awareness capability is particularly impactful in diagnostically challenging scenarios. This mechanism ensures only high-confidence predictions are clinically actionable, while automatically triaging ambiguous cases to human experts for secondary review. This approach also prevents clinicians from over-reliance on model outputs, thereby aligning with responsible clinical workflows in healthcare settings. Complementing this reliability mechanism, BrainVLM exhibits an exceptional capability to generate preliminary medical reports—a critical function for communicating diagnostic findings and guiding care across several specialties.<sup>30,34</sup> These reports offer clinicians actionable insights beyond isolated predictions, enabling clinicians to process cases more efficiently, thereby improving turnaround times and expanding access to specialty-level reporting<sup>35</sup>. The clinical validation through a multi-reader study with 12 neuroradiologists across expertise levels further confirmed BrainVLM’s trustbuilding potential.

In clinical practice, incomplete MRI acquisition (e.g., missing T1c) frequently occurs due to insufficient clinician awareness or contraindications. While conventional multi-sequencetrained models exhibit considerable performance degradation under such suboptimal imaging conditions, BrainVLM maintains diagnostic performance comparable to the full-sequence baseline in scenarios where T2, T2-Flair, or T1 sequences are missing (appendix pp 25-26). A notable exception is the absence of T1c sequences, where we observe a 13.4% decline in F1 score (appendix pp 25-26), highlighting the critical role of post-contrast imaging in brain tumor assessment. Another feature relevant to clinical deployment is that BrainVLM was trained on raw, minimally processed MR images, without skull stripping or segmentation (appendix p 11). Unlike many conventional neuroimaging studies, which typically require preprocessing steps such as skull stripping, BrainVLM directly analyzes raw institutional data, with only slice-level resampling applied (appendix p 11). This minimal preprocessing preserves the complete informational content of the original MRI data, making BrainVLM particularly well-suited for our comprehensive brain tumor characterization study. Collectively, BrainVLM’s ability to deliver robust diagnostic performance without preprocessing workflows (e.g., skull stripping) or segmentation requirements, combined with its resilience to missing MRI sequences, establishes its clinical efficacy as an adaptable and practical solution for real-world neuroimaging challenges.

Several limitations exist in our study. First, although online datasets were collected globally, real-world diagnostic applications in this study were confined to East Asian patients. Second, despite improved performance in low-prevalence tumors, data on rare tumors remain relatively limited. Further validation studies with larger sample sizes and more diverse ethnic populations, especially for rare tumor types, are warranted to better evaluate the model’s generalizability and robustness. Third, while patient demographics is incorporated into BrainVLM, the model does not currently integrate multimodal clinical data such as medical history, lab findings and neurological examinations. Including these data types could further improve diagnostic accuracy and better align the model with real-world clinical decisionmaking. Correspondingly, in the multi-reader study, readers were only given MRI, age, and sex, matching BrainVLM's inputs, to ensure a fair comparison; this may underestimate clinicians' real-world performance, as full clinical context is routinely accessible in routine care. Fourth, although we used patient-level partitioning, independent institutional cohorts, exclusion of recurrent or postoperative follow-up cases, and external test centers that did not contribute to training or validation, residual risks of dataset overlap or data leakage cannot be completely excluded. This is particularly relevant for large public datasets aggregated from multiple sources and because direct cross-center patient matching was not possible after deidentification. Finally, we assessed diagnostic performance, report-generation quality and reader performance with or without AI assistance, but did not examine subsequent treatment decisions and clinical outcomes, such as extent of resection, surgical complications or longterm prognosis”.

In conclusion, BrainVLM demonstrates the potential of multimodal foundation models to support complex diagnostic tasks in neuro-oncology. Trained on well-curated multimodal neuroimaging data and validated across internal, multi-center external, and prospective clinical cohorts, the model achieves generalizable performance for classifying a broad spectrum of brain tumors. BrainVLM may serve as a reliable assistant tool to improve the accuracy, efficiency, and consistency of preoperative brain tumor diagnosis and automated report generation. Our framework, which integrates systematic data curation, advanced visionlanguage modeling, uncertainty quantification, automated report generation, and rigorous clinical evaluation, offers a structured approach for developing specialized foundation models in other medical fields.

## Contributors

L.Q. and X.G. conceptualized the study. J.C., Y.W., H.J., L.Q., and A.H. conducted data collection and preprocessing from public repositories. X.G., Z.C., S.K., B.Z., B.W., Z.X., H.Y., Z.W., W.W., B.T., W.H., W.D., W.Z., L.M., A.H., B.L., L.W., Y.C., and W-w.W. collected data from medical centers. X.G., Z.C., S.K., J.C., Y.W., and H.J. performed data preprocessing for institutional data. Y.W., J.C., S.Z., H.J., and L.Q. developed and trained the deep learning algorithms and conducted model analysis. X.G., L.Q., M.I., Y-R.W., F.W., Y.Z., L.Z., and Z.L. provided domain knowledge and interpreted the findings. L.Q., X.G., Y.W., and J.C. drafted the primary manuscript and prepared the figures, with input from all authors. All authors had access to all the data in the study, reviewed and approved the final version of the study. All authors contributed to the general discussion, manuscript revision, and approved the final version. L.Q. and X.G. co-supervised the study. L.Q. and X.G. were responsible for the decision to submit the manuscript.

## Data sharing

The online training data are available on their corresponding website. The real-world data are not publicly accessible owing to patient confidentiality requirements. Reasonable requests for access may be considered by the corresponding author, subject to approval from the institutional review boards at all participating centers. The complete code for BrainVLM is available on GitHub at https://github.com/HKU-HealthAI/BrainVLM.

## Declaration of interests

We declare no competing interests.

## Acknowledgments

This study was supported by the National Natural Science Foundation of China (62306253), the Early Career Fund (27207025, 27204623), the Clinical Research Fund of the National Clinical Research Center for Geriatric Disorders (2023LNJJ19), the Natural Science Foundation of Hunan Province (2023JJ30927), and the Guangdong Natural Science Fund-General Program (2024A1515010233).

## Reference

1 Price M, Ballard C, Benedetti J, et al. CBTRUS Statistical Report: Primary Brain and Other Central Nervous System Tumors Diagnosed in the United States in 2017-2021. Neuro Oncol 2024; 26: vi1–85.

2 Siegel RL, Miller KD, Wagle NS, Jemal A. Cancer statistics, 2023. CA Cancer J Clin 2023; 73: 17–48.

3 Miller KD, Ostrom QT, Kruchko C, et al. Brain and other central nervous system tumor statistics, 2021. CA Cancer J Clin 2021; 71: 381–406.

4 Louis DN, Perry A, Wesseling P, et al. The 2021 WHO Classification of Tumors of the Central Nervous System: a summary. Neuro Oncol 2021; 23: 1231–51.

5 Karschnia P, Gerritsen JKW, Teske N, et al. The oncological role of resection in newly diagnosed diffuse adult-type glioma defined by the WHO 2021 classification: a Review by the RANO resect group. Lancet Oncol 2024; 25: e404–19.

6 Ferreri AJM, Calimeri T, Cwynarski K, et al. Primary central nervous system lymphoma. Nat Rev Dis Primers 2023; 9: 29.

7 Frappaz D, Dhall G, Murray MJ, et al. EANO, SNO and Euracan consensus review on the current management and future development of intracranial germ cell tumors in adolescents and young adults. Neuro Oncol 2022; 24: 516–27.

8 Horbinski C, Nabors LB, Portnow J, et al. NCCN Guidelines® Insights: Central Nervous System Cancers, Version 2.2022. J Natl Compr Canc Netw 2023; 21: 12–20.

9 Yuan Y, Zheng Y, Qu L. Benchmarking Radiology Report Generation From Noisy Free-Texts. IEEE J Biomed Health Inform 2025; 29: 7549–58.

10 Zhang K, Zhou R, Adhikarla E, et al. A generalist vision-language foundation model for diverse biomedical tasks. Nat Med 2024; 30: 3129–41.

11 Wang Y-R (Joyce), Wang P, Yan Z, et al. Advancing presurgical non-invasive molecular subgroup prediction in medulloblastoma using artificial intelligence and MRI signatures. Cancer Cell 2024; 42: 1239-1257.e7.

12 Menze BH, Jakab A, Bauer S, et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE Trans Med Imaging 2015; 34: 1993–2024.

13 Kim Y, Jeong H, Chen S, et al. Medical Hallucinations in Foundation Models and Their Impact on Healthcare. 2025; published online Nov 2. DOI:10.48550/arXiv.2503.05777.

14 Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024; 385: e078378.

15 Luo X, Yang Y, Yin S, et al. Automated segmentation of brain metastases with deep learning: A multi-center, randomized crossover, multi-reader evaluation study. Neuro Oncol 2024; 26: 2140–51.

16 Wu C, Zhang X, Zhang Y, Hui H, Wang Y, Xie W. Towards generalist foundation model for radiology by leveraging web-scale 2D&3D medical data. Nat Commun 2025; 16: 7866.

17 Blankemeier L, Cohen JP, Kumar A, et al. Merlin: A Vision Language Foundation Model for 3D Computed Tomography. Res Sq 2024; : rs.3.rs-4546309.

18 Liu Z, Ning J, Cao Y, et al. Video Swin Transformer. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). New Orleans, LA, USA: IEEE, 2022: 3192–201.

19 OpenAI. GPT-4; 2023. https://openai.com/research/gpt-4.

20 Papineni K, Roukos S, Ward T, Zhu W-J. BLEU: a method for automatic evaluation of machine translation. In: Proceedings of the 40th Annual Meeting on Association for Computational Linguistics - ACL ’02. Philadelphia, Pennsylvania: Association for Computational Linguistics, 2001: 311.

21 Zhao W, Wu C, Zhang X, Zhang Y, Wang Y, Xie W. RaTEScore: A Metric for Radiology Report Generation. 2024; published online Oct 23. DOI:10.48550/arXiv.2406.16845.

22 Delbrouck J-B, Chambon P, Chen Z, et al. RadGraph-XL: A Large-Scale Expert-Annotated Dataset for Entity and Relation Extraction from Radiology Reports. In: Findings of the Association for Computational Linguistics ACL 2024. Bangkok, Thailand and virtual meeting: Association for Computational Linguistics, 2024: 12902–15.

23 Ström P, Kartasalo K, Olsson H, et al. Artificial intelligence for diagnosis and grading of

prostate cancer in biopsies: a population-based, diagnostic study. Lancet Oncol 2020; 21: 222–32.

24 Gu B, Desai RJ, Lin KJ, Yang J. Probabilistic medical predictions of large language models. NPJ Digit Med 2024; 7: 367.

25 Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. .

26 Brier GW. VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF PROBABILITY. Mon Wea Rev 1950; 78: 1–3.

27 Whitfield BT, Huse JT. Classification of adult-type diffuse gliomas: Impact of the World Health Organization 2021 update. Brain Pathol 2022; 32: e13062.

28 Wu J, Gonzalez Castro LN, Battaglia S, et al. Evolving cell states and oncogenic drivers during the progression of IDH-mutant gliomas. Nat Cancer 2025; 6: 145–57.

29 van den Bent MJ, Geurts M, French PJ, et al. Primary brain tumours in adults. Lancet 2023; 402: 1564–79.

30 Rao VM, Hla M, Moor M, et al. Multimodal generative AI for medical image interpretation. Nature 2025; 639: 888–96.

31 Tanno R, Barrett DGT, Sellergren A, et al. Collaboration between clinicians and visionlanguage models in radiology report generation. Nat Med 2025; 31: 599–608.

32 Wang Y-R (Joyce), Qu L, Sheybani ND, et al. AI Transformers for Radiation Dose Reduction in Serial Whole-Body PET Scans. Radiology: Artificial Intelligence 2023; 5: e220246.

33 Shi Y, Ji J, Zhang X, Qu L, Liu Y. Granularity Matters: Pathological Graph-driven Crossmodal Alignment for Brain CT Report Generation. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Singapore: Association for Computational Linguistics, 2023: 6617–30.

34 Moor M, Banerjee O, Abad ZSH, et al. Foundation models for generalist medical artificial intelligence. Nature 2023; 616: 259–65.

35 Zhang X, Shi Y, Ji J, Zheng C, Qu L. MEPNet: Medical Entity-Balanced Prompting Network for Brain CT Report Generation.

Table 1: Characteristics of our retrospective online dataset, primary dataset, and external test dataset for each brain tumor type.
<table><tr><td rowspan="2"></td><td colspan="2">Online dataset</td><td colspan="3">Primary dataset</td><td colspan="3">External test dataset</td><td rowspan="2">Entire dataset</td></tr><tr><td>No. of subjects</td><td>Dimension 3D 2D</td><td>No. of subjects I</td><td>Sex Male Female</td><td>Age in years Mean ± SD (range)</td><td>No.of subjects</td><td>Sex Male Female</td><td>Age in years Mean ± SD</td></tr><tr><td>Total</td><td>21,993</td><td>13,096 8,897</td><td>8,813</td><td>4,256 4,557</td><td>45 ± 18 (1-86)</td><td>1,334</td><td>665 669</td><td>(range) 52 ± 15 (1-86)</td><td>32,140</td></tr><tr><td>MET</td><td>1,654</td><td>879 775</td><td>735</td><td>435 300</td><td>56±11 (7-79)</td><td>116</td><td>69 47</td><td>60±11 (34-86)</td><td>2,505</td></tr><tr><td>GCT</td><td>81</td><td>62 19</td><td>378</td><td>286 92</td><td>19 ± 13 (1-67)</td><td>18</td><td>11 7</td><td>20 ± 13 (1-40)</td><td>477</td></tr><tr><td>GGN</td><td>9,935</td><td>4,7435,192</td><td>3,028</td><td>1,6811,347</td><td>41±19 (1-86)</td><td>414</td><td>244 170</td><td>50±15 (5-79)</td><td>13,377</td></tr><tr><td>MEN</td><td>5,363</td><td>3,804 1,559</td><td>2,395</td><td>723 1,672</td><td>53 ± 12 (1-81)</td><td>355</td><td>115 240</td><td>56 ± 13 (8-79)</td><td>8,113</td></tr><tr><td>TSR</td><td>674</td><td>614 60</td><td>796</td><td>402 394</td><td>46±16 (2-81)</td><td>186</td><td>98 88</td><td>51 ± 15 (5-75)</td><td>1,656</td></tr><tr><td>MNM</td><td>912</td><td>457 455</td><td>447</td><td>234 213</td><td>39 ± 18 (1-78)</td><td>76</td><td>40 36</td><td>43 ± 16 (8-76)</td><td>1,435</td></tr><tr><td>CPN</td><td>1,865</td><td>1,392 473</td><td>416</td><td>138 278</td><td>46±15(1-75)</td><td>90</td><td>43 47</td><td>54±10 (34-75)</td><td>2,371</td></tr><tr><td>CPT</td><td>259</td><td>216 43</td><td>129</td><td>63 66</td><td>28 ± 20 (1-66)</td><td>6</td><td>1 5</td><td>45 ± 10 (39-62)</td><td>394</td></tr><tr><td>HEM</td><td>535</td><td>396 139</td><td>164</td><td>100 64</td><td>47±22 (2-77)</td><td>44</td><td>27 17</td><td>62±11 (30-82)</td><td>743</td></tr><tr><td>EMB</td><td>288</td><td>165 123</td><td>205</td><td>131 74</td><td>18 ± 15 (1-70)</td><td>18</td><td>12 6</td><td>18 ± 15 (3-40)</td><td>511</td></tr><tr><td>PIN</td><td>346</td><td>312 34</td><td>70</td><td>36 34</td><td>36±20 (1-73)</td><td>6</td><td>2 4</td><td>39±13 (21-57)</td><td>422</td></tr><tr><td>MEL</td><td>81</td><td>56 25</td><td>50</td><td>27 23</td><td>50 ± 14 (14-73)</td><td>5</td><td>3 2</td><td>36 ± 25 (21-69)</td><td>137</td></tr></table>

Abbreviations: MET = brain metastases, GCT = germ cell tumors, GGN = gliomas, glioneuronal tumors, and neuronal tumors, MEN = meningioma, TSR = tumors of the sellar region, MNM = mesenchymal, non-meningothelial tumors, CPN = cranial and paraspinal nerve tumors, CPT = choroid plexus tumors, HEM = hematolymphoid tumors, EMB = embryonal tumors, PIN = pineal region tumors and MEL = melanocytic tumors.

Table 2: Comparative diagnostic performance across neuroradiologists-only, BrainVLM, and AI-augmented neuroradiologists paradigms. Twelve neuroradiologists with varying years of experience (junior [3-5 years], senior [5-10 years], expert [10+ years]) participated in this blinded crossover study. Boldface indicates the highest performance metric within each tumor type across AI, neuroradiologists, and AI-augmented neuroradiologists cohorts.
<table><tr><td rowspan="2"></td><td colspan="7"> $F _ { 1 }$  score</td></tr><tr><td>AI model</td><td>Junior</td><td>Junior-AI</td><td>Senior</td><td>Senior-AI</td><td>Expert</td><td>Expert-AI</td></tr><tr><td>MET</td><td>0.74</td><td>0.51</td><td>0.62</td><td>0.44</td><td>0.66</td><td>0.80</td><td>0.87</td></tr><tr><td>GCT</td><td>0.75</td><td>0.28</td><td>0.54</td><td>0.40</td><td>0.59</td><td>0.70</td><td>0.79</td></tr><tr><td>GGN</td><td>0.83</td><td>0.64</td><td>0.72</td><td>0.66</td><td>0.80</td><td>0.84</td><td>0.91</td></tr><tr><td>MEN</td><td>0.85</td><td>0.74</td><td>0.81</td><td>0.77</td><td>0.85</td><td>0.90</td><td>0.90</td></tr><tr><td>TSR</td><td>0.86</td><td>0.74</td><td>0.77</td><td>0.78</td><td>0.81</td><td>0.85</td><td>0.88</td></tr><tr><td>MNM</td><td>0.56</td><td>0.10</td><td>0.33</td><td>0.07</td><td>0.42</td><td>0.50</td><td>0.58</td></tr><tr><td>CPN</td><td>0.77</td><td>0.57</td><td>0.74</td><td>0.69</td><td>0.79</td><td>0.73</td><td>0.84</td></tr><tr><td>CPT</td><td>0.65</td><td>0.30</td><td>0.53</td><td>0.36</td><td>0.68</td><td>0.77</td><td>0.88</td></tr><tr><td>HEM</td><td>0.75</td><td>0.22</td><td>0.56</td><td>0.21</td><td>0.70</td><td>0.70</td><td>0.86</td></tr><tr><td>EMB</td><td>0.82</td><td>0.14</td><td>0.62</td><td>0.31</td><td>0.75</td><td>0.54</td><td>0.68</td></tr><tr><td>PIN</td><td>0.77</td><td>0.43</td><td>0.62</td><td>0.48</td><td>0.68</td><td>0.70</td><td>0.75</td></tr><tr><td>MEL</td><td>0.25</td><td>0.00</td><td>0.40</td><td>0.00</td><td>0.69</td><td>0.63</td><td>0.80</td></tr><tr><td>Frequency -weighted  $F _ { 1 }$ </td><td></td><td></td><td></td><td></td><td>0.78 [0.73, 0.83]0.52 [0.48, 0.55]0.67 [0.63, 0.71]0.55 [0.51, 0.58] 0.74 [0.70, 0.77] 0.78 [0.74, 0.82]0.85 [0.81, 0.89]</td><td></td><td></td></tr><tr><td>Accuracy</td><td></td><td></td><td></td><td></td><td>0.79 [0.74, 0.83] 0.55 [0.52, 0.59] 0.69 [0.65, 0.72] 0.59 [0.56, 0.62] 0.75 [0.76, 0.87] 0.80 [0.75, 0.83] 0.86 [0.82, 0.89]</td><td></td><td></td></tr><tr><td>Total Time (Mean ± SD)</td><td>51 ± 3</td><td>189 ± 41</td><td> $1 1 2 \pm 1 3$ </td><td> $1 6 1 \pm 4 3$ </td><td>104 ± 21</td><td>100 ± 5</td><td> $7 5 \pm 1 5$ </td></tr></table>

Abbreviations: MET = brain metastases, GCT = germ cell tumors, GGN = gliomas, glioneuronal tumors, and neuronal tumors, MEN = meningioma, TSR = tumors of the sellar region, MNM = mesenchymal, non-meningothelial tumors, CPN = cranial and paraspinal nerve tumors, CPT = choroid plexus tumors, HEM = hematolymphoid tumors, EMB = embryonal tumors, PIN = pineal region tumors and MEL = melanocytic tumors. SD: Standard Deviation.

![](images/997f601f67aea2bf85c09a818a4a2ff351486fb44925b5c95d85bcd3c2c8d079.jpg)  
b

![](images/93efeb9e05e83e4ecba75796cb37513996045bd47b69da113c9870c01379782b.jpg)

c  
![](images/fb76842de0357abd445586d09b98f2e3ef36a7edc9c8599e1dfba5fd5fbc5704.jpg)

d  
![](images/572d01ad80890022c245498692917ed697dac2da4e24b348a70819cb22eb81b2.jpg)

![](images/129d3e49b75234ac22ee40265bf215ab5307378d611527eb7d7f583854c966ce.jpg)

![](images/75b50e8a42f8c23319959b834ea35482998e842c3ecfaf5a151a50f7c656712c.jpg)

![](images/93113134a1f35c603162cbc14cb3a2360dc58ad6fb85bd7e40f36978519fb940.jpg)

Figure 1: Overview of this study. a, A multi-modal brain tumor dataset was constructed from 39 online sources and 12 collaborating medical institutions, comprising 47,947 patients. Each case includes either 2D MRI slices or multisequence 3D MRI scans (T1-weighted[T1], T1 contrast-enhanced [T1c], T2-weighted [T2], T2-Flair [T2f]), paired with pathologically confirmed diagnoses, optional radiology reports (or text descriptions). The detailed number of patients across 12 brain tumor types and their subtypes is shown on the right panel. b, we developed BrainVLM, a vision-language foundation model designed for brain tumor analysis. The model processes multi-parametric MRI scans at different resolutions through a shared vision encoder and integrates text data via an LLM-based architecture. BrainVLM was trained to classify tumor types with quantified diagnostic uncertainty and a radiology report. “Frozen” components (Vision Encoder and LLM backbone) retain their pre-trained weights to ensure structural stability and leverage broad medical knowledge. “Trainable” components (MLP projector and LoRA-based adapters) are optimized during fine-tuning via backpropagation. c, ROC curves for top-1 predictions on the retrospective primary and external datasets, with macro-averaged AUCs of 0.85/0.80 and micro-averaged 0.91/0.87, respectively. d, BrainVLM outperformed neuroradiologists and baseline models in sensitivity, precision, F1 score, and Cohen’s κ across both datasets. Two-tailed Wilcoxon signed-rank tests were performed to compare BrainVLM with neuroradiologists for each metric. P values were not adjusted for multiple comparisons because each metric was prespecified and interpreted as a distinct measure of diagnostic performance. Statistical significance is indicated as ns (not significant, p>0.05), ⋆ (p<0.05), ⋆⋆ (p<0.01), and ⋆⋆⋆ (p<0.0001). Bootstrap resampling (1,000 replicates) was used to estimate 95% confidence intervals, shown as error bars around the mean.

![](images/989742e3460071076de1e6299fcbca7c31c6cf2ee41d27a0ac2e18bc688028c6.jpg)

![](images/b256771847698c8b4e30e84a292a27c2565fa15d003c0adb76ac44a157099613.jpg)

![](images/bd1abf09e54703d80dc6af54d5fdd956e6454c47ab1469a3bd6390b051bbab20.jpg)

![](images/dec8583c75dc2a08ad591e1ec122c34dd570c750242fc30fe3256227c155ee3b.jpg)

![](images/43c97adc9ae55755a36b2d2a5de3f49191bb1208b6d5a76607bf3da89c2e6e1e.jpg)

![](images/d0ebab8344cc7415727b5e95791c226ab47c3015bb6cea2375b0d320a3e1d954.jpg)

![](images/826edb45d32f9f961c4bb36252314b9b18f612a857ec73241023776e361b67dd.jpg)

![](images/73b70f5ec2a98a3145759467ae05af2f2ee7a16d9b938b66868e763cf3cbb945.jpg)

![](images/e958befd8735ee50da371a7b2fd070b62d1b2bae383af1d32b00120b15a1dbd1.jpg)

b  
![](images/36ab6d451967ad0e9e2365ad8b05c96f382cfe5438793c8fd37dd8c5492b33a0.jpg)

![](images/197b6dcf1d59fadbc500518dd699f8636642977352efd8f6efcd68772e458659.jpg)

C  
![](images/fe0e9ec927d284675d59aa86062879c11f647047dd81e753c4933c0595a689f5.jpg)  
Figure 2 a, Performance comparison of BrainVLM and board-certified neuroradiologists on 12 tumor types across primary and external datasets. Model and human expert results are visualized as a dodecagon, with each vertex corresponding to a specific tumor type. For both datasets, we report sensitivity, precision, F1 score, and Cohen’s κ for each class. BrainVLM demonstrates either superior or comparable performance relative to neuroradiologists on both primary and external cohorts, including for rare tumor types (e.g., EMB, GCT, HEM). b, Confusion matrices comparing BrainVLM’s predictions to those of neuroradiologists across 12 tumor types in both primary and external datasets. The foundation model showed stronger diagonal concentration than neuroradiologists, indicating improved classification accuracy. Each matrix was normalized by true labels to highlight per-class performance. Notably, the model’s diagnostic error patterns showed substantial concordance with those of human experts. c, Flowchart of prospective patient enrollment. A total of 1,511 patients with suspected brain lesions were prospectively enrolled. After excluding those who had received prior treatment or lacked diagnostic sequences, 1,090 patients were eligible for evaluation. Among them, 967 underwent surgery; after excluding 123 cases with non-brain tumor pathology, 886 patients with confirmed brain tumors were included for model validation. The remaining 81 patients who received non-operative management were also included for evaluation.

e  
a  
![](images/06f6b00e3b39d47bf275f3330fd9a7fcc56411501987596f072f0fbdd5da7955.jpg)

![](images/30ab741637c8ac288fed199c9b74ef559c3d3c56ec789fedaab3b8fe1cf28256.jpg)

b  
![](images/6505169a866d0fe4640fc876093fa1c2b8d52717a4d41948999c639104c47e9b.jpg)

C  
![](images/a679328fde27e61ef9a1437297257faa5a1d33ba729206739e8f0d20e20170d1.jpg)

![](images/abb9ae1f7938f4631bacb52b6fcb6c972797f7db923872b928f3ca12e3b8b905.jpg)

![](images/070e9d4d93f9c15f0a8d2598280664bc6d2fbc297294572a4261635c071fed46.jpg)

d  
![](images/3c91ccdbfc7ca3bf851d4556147ba7f36a51c3682273381ac06decc2b68e840b.jpg)

![](images/25d14c7f38f54cea5ffce57e7521746630d5c61c6b26799aba4b47f8da51f357.jpg)

![](images/1935db8a938e94485ce0d64de9030126241a911adef1bd6add276bcf93f0c15b.jpg)

f  
![](images/c4399a25abd028c8f37117a45037cda42d9115f907ecb2711f9a880f798c7c2c.jpg)  
Figure 3: Performance comparison in the prospective cohort, as well as uncertainty quantification, comparative evaluation of radiology report generation, and molecular subgroup prediction. a, Comparison of primary and external prospective study results. The bar charts evaluate BrainVLM (AI) versus human doctors across surgical and non-surgery groups using Sensitivity, Precision, F1, and Kappa metrics. b, Reliability diagram (ECE = 0.036) illustrating model calibration and the alignment between observed accuracy and confidence probability. c, the comparison of confidence score distributions between Vanilla BrainVLM, which does not incorporate our consensus-driven strategy for uncertainty quantification, and Reliable BrainVLM, which benefits from the integration of this strategy. d, Report metric comparison of four models using RaTEScore, F1RadGraph-XL, and BLEU4 on both primary and external datasets. Two-tailed Wilcoxon signed-rank tests were performed to compare BrainVLM with comparator models; p values were not adjusted for multiple comparisons, as each metric was interpreted individually rather than as repeated tests of a single hypothesis. e, ROC curves comparison for preoperative molecular subgroup prediction in adult-type diffuse gliomas among RadFM, Merlin, and our model on both primary and external datasets. f, A representative case study comparing generated reports from our model, ChatGPT-4o, and RadFM against the human radiologist’s reference report, highlighting significant corrections, errors, and omissions. For BrainVLM’s diagnosis, the associated probability and confidence score are also presented.

Confidence Distribution of Correct and Incorrect Answers by Tumor Type  
a  
![](images/04235f86958d4105cdea8edbae174cf0775e8d85e925a8f0bcbc4ac8d6906a84.jpg)

b  
![](images/1e6553bc66841859faaae627095db536268b63af86b4b1c10fdb9f1009bda5bb.jpg)

C  
![](images/c33b5cc62754652664aee53bfdde46f391f36243b5231a217f249b512fa0f50a.jpg)

d  
![](images/ea34bb6550001bbe9237b3f982f1b21fa856d12846b478c2c33658ea15c1da97.jpg)

e  
![](images/7799e089aff949077e162565f9bc4541e3391db381a1b7f79d91dcf4b1f40e3c.jpg)

f  
![](images/8eced2325460799cd1300bd522c48463ae2b068d00aa11a1149c21d5ce2c96f2.jpg)

g  
![](images/513835687a72238ad0cc9d658339fdd00345c1bf12f22d80fb8359f4096e0e42.jpg)

Figure 4: Blinded multi-reader study on 248 patients with interpretations by 12 neuroradiologists (5 juniors [3-5 years], 4 seniors [5-10 years], 3 experts [10+ years]) with and without AI assistance. a-d, Comparison between the diagnostic performance of neuroradiologists and AI-augmented neuroradiologists. Box plots summarize sensitivity, precision, F1 score, and Cohen’s κ for individual neuroradiologists and their AI-augmented counterparts. Pairwise comparisons between each neuroradiologist and their corresponding AI-augmented counterpart were performed using two-tailed Wilcoxon signed-rank tests without correction for multiple comparisons. Statistical significance is indicated as ns (not significant, $\mathrm { p } { > } 0 . 0 5 ) , \star ( \mathrm { p } { < } 0 . 0 5 ) , \star \star ( \mathrm { p } { < } 0 . 0 1 )$ , and ⋆⋆⋆ (p<0.001). P values were not adjusted for multiple comparisons because per-tumor-type analyses were prespecified and interpreted separately for each tumor category. e, Confidence score distribution for BrainVLM’s predictions on these 248 cases across 12 major WHO CNS5 brain tumor types. The figure is vertically divided: upper blue bars show confidence distributions for correct predictions, lower pink bars show incorrect ones. f, Distribution of Diagnostic Trajectories in Cases with Low Reader Confidence. The bar chart shows the distribution of four diagnostic trajectories after exposure to BrainVLM among the subset with the lowest 25% of initial clinician confidence, stratified by reader seniority. Trajectories are defined as: (A) maintaining a correct diagnosis despite incorrect AI; (B) successfully correcting an initial error; (C) erroneously changing a correct diagnosis due to incorrect AI; (D) both clinician and AI remaining incorrect; (E) failing to adopt a correct AI suggestion; and (F) both initial clinician diagnosis and AI suggestion being correct, with the final diagnosis remaining correct. g, Clinician-AI performance in cases with incorrect AI predictions. The bar chart compares clinician diagnostic accuracy without AI assistance and after review of BrainVLM output, restricted to cases in which BrainVLM generated an incorrect diagnosis, stratified by reader seniority (junior, senior, and expert).

Contents of Supplementary   
S1 Dataset 2   
S1.1 Construction of BrainTumor48K dataset . 2   
S1.1.1 Dataset construction from online repositories. 2   
S1.1.2 Dataset construction from medical institutions 4   
S1.2 Dataset composition and curation framework of BrainTumor48K. 6   
S1.3 Ethics approval for datasets from medical institutions . 6   
S1.4 Patient and public involvement. 7   
S1.5 Estimation of total subject count from online resources 7   
S2 Model Development 8   
S2.1 Development of BrainVLM for tumor classification and report generation . 8   
S2.1.1 Multi-modal model architecture . 8   
S2.1.2 Clinical-motivated data standardization protocol. .10   
S2.1.3 Three-stage progressive training strategy of BrainVLM .13   
S2.1.4 Loss function .15   
S2.1.5 Data augmentation .16   
S2.2 Model reliability implement. 16   
S2.2.1 Consensus-driven reliability dataset construction .17   
S2.2.2 Confidence-aware reliable BrainVLM finetuning .18   
S2.3 Model inference 18   
S2.3.1 Confidence-triggered TOP 2 supplementary diagnosis .21   
S2.4 Finetuning of molecular subgroup classification task 21   
S2.5 Competing methods 21   
S2.6 Quantitative assessment and statistical analysis. 23   
S2.7 Implementation details 24   
S3 More analytical experiments on BrainVLM 25   
S3.1 Top-2 prediction result based on uncertainty score 25   
S3.2 Robustness of BrainVLM to missing MRI sequences. 25   
S3.3 Subcategories classification performance in GGN. 26   
S3.4 Validation of generated report descriptions . 26   
S3.5 Expert-based clinical assessment of generated reports 27   
S3.6 Diagnosis performance in intraaxial and extraaxial tumor 28   
S3.7 False negative finding investigation 29   
S3.8 Uncertainty quantification of BrainVLM. 30   
S3.8.1 Confidence distributions: top 3 vs. bottom 5 classes .30   
S3.8.2 Multi-readers assessment on bottom 10% confidence cases .31   
S3.9 Subgroup analysis across age, sex, MRI vendor and ethnicity. 32   
S3.9.1 Analysis of demographic variables: age and sex .32   
S3.9.2 Analysis of MRI vendor .33   
S3.9.3 Analysis of ethnicity .33   
S3.9.4 Conclusion of subgroup analysis .34   
S3.10 Clinician-AI interaction in incorrect AI scenarios 34   
S3.11 Case studies 35   
S3.11.1 Case studies in high-confidence error prediction cases .35   
S3.11.2 Case studies in multi-readers’ time reduction .36   
S4 Regulatory considerations, failure modes, and medico-legal implications 36   
S5 Data and code availability<sub>.</sub> 37   
References 39   
Supplementary Tables<sub>.</sub> 46   
Supplementary Figures 58

## S1 Dataset

## S1.1 Construction of BrainTumor48K dataset

We assembled the BrainTumor48K dataset through systematic curation from 39 online sources and 12 collaborating medical centers. To guide the search and data collection processes, we first established a comprehensive set of brain tumor keywords, including 12 major brain tumor types and all subtypes, along with their synonyms and related terms. Next, we leveraged retrieval-augmented generation (RAG) and prompt engineering techniques to design an automated preprocessing pipeline (see Supplementary Figure 1). This pipeline transformed the curated raw datasets into structured formats of {Patient\_ID: multi-parametric 3D MRI volumes (or 2D MRI slice), pathological diagnosis label, optional radiology report (or text description)} for model training. We also expanded our datasets by incorporating data from healthy individuals. For the multi-parametric MRI scans, we considered T1-weighted (T1), T1 contrast-enhanced (T1c), T2-weighted (T2), and T2-Flair (FLAIR), which are commonly used in brain tumor diagnosis.

## S1.1.1 Dataset construction from online repositories

We compiled our dataset from diverse online repositories, categorized based on their inherent structure and the required preprocessing complexity. The first category consists of well-structured sources of medical imaging data, such as established repositories (e.g., BraTS2023,<sup>1–6</sup> TCIA,<sup>7–20</sup> Kaggle,<sup>21–24</sup>, Ctisus<sup>25</sup>, and other public repositories<sup>26–40</sup>) as well as specialized medical platforms like Radiopaedia.<sup>41</sup> This category encompasses sources that provide readily accessible, structured medical imaging data (including 3D multi-parametric MRI volumes or 2D slices), which are typically paired with pathological labels. Depending on the source, additional information, such as patient metadata or radiology reports, may also be available. The second category involves non-dedicated imaging sources requiring advanced figure/text extraction, such as PubMed Central scientific archives.<sup>42</sup> This platform, not originally designed for medical imaging storage, necessitated specialized pipelines for isolating figures/text from heterogeneous formats (e.g., publication figures, video stills) and reconciling fragmented clinical annotations. In the following, we detail the distinct data pre-processing pipelines developed for each category.

Pre-processing pipeline for structured repositories and pre-curated datasets. Datasets in this category are highly heterogeneous, encompassing various sources with differing image formats, annotations, and metadata. These datasets range from simplified collections (where each patient is represented by a single 2D MRI slice with a slice-level pathological label) to comprehensive datasets containing multi-parametric

3D MRI sequences, tumor segmentation masks, patient metadata, and pathological labels. For datasets providing only tumor classification labels, we processed the data into a standardized structure of the form: {Patient\_ID: 3D MRI volumes (or 2D MRI slice), pathological label}. The resulting standardized data were subsequently used for tumor classification tasks. For datasets containing 3D multi-parametric MRI scans (or 2D MRI slice), segmentation masks, and metadata (e.g., BraTS,<sup>1–6</sup> TCIA<sup>7–20</sup> and Kaggle datasets<sup>21,22</sup>), we first applied the Brainnetome Atlas<sup>43</sup> to localize anatomical brain regions based on the tumor masks. We then leveraged useful metadata (including imaging modality, sex, diagnostic label, and tumor description) to generate comprehensive medical reports via GPT-4o prompted with RAG techniques (see Supplementary Figure 1b). For certain web-sourced datasets, such as Radiopaedia,<sup>41</sup> which include multi-parametric MRI scans and accompanying radiology reports, we removed irrelevant information and reformatted the data into a consistent structure: {Patient\_ID: 3D MRI volumes, pathological label, radiology report}.

This comprehensive pre-processing pipeline enabled the assembly of a diverse cohort comprising (see Supplementary Table 2): 1) 6,957 brain tumor patients with 3D multi-parametric MRI scans, radiology reports, and associated pathological labels; 2) 1940 brain tumor patients with 3D multi-parametric MRI scans and pathological labels only; 3) 233 patients with 2D MRI slice-diagnosis; and 4) 3,178 healthy control patients with multi-parametric 3D MRI scans. Apart from the datasets detailed above, this preprocessing pipeline also incorporated subjects from additional repositories, such as Br35H,<sup>21</sup> Brain Tumor Classification Dataset,<sup>24</sup> and Brain Tumor MRI Images 44 Classes.<sup>23</sup> However, as these repositories only contain 2D MRI slices with associated pathological labels and lack patient identifiers, they were not included in our dataset summary here (see Supplementary Table 2).

Pre-processing pipeline for unstructured web-based resources. Web-crawled brain tumor data from platforms such as PubMed Central,<sup>42</sup> ImageCLEF<sup>44</sup> often contain noisy 2D MRI scans and misaligned or inconsistent text descriptions. To address these challenges, we developed a comprehensive data preprocessing pipeline that transforms raw, heterogeneous web data into a structured format: {Patient\_ID: 2D MRI slice, text description, pathological label}, see Supplementary Figure 1a2.

For scientific articles from PubMed Central, we systematically retrieved 2D brain MRI slices by querying predefined brain tumor keywords, associating each image with both “golden captions” and inline textual mentions as candidate descriptions. However, these figures frequently contain composite images with multiple subfigures and inconsistent captioning. To resolve this, we developed a three-stage refinement protocol: Firstly, we applied a Faster R-CNN object detector,<sup>45</sup> trained on the MedICaT dataset,<sup>46</sup> to perform subfigure separation and irrelevant images filtering. Secondly, we employed GPT-4o with specialized prompts for accurate subcaption separation. Finally, we established a precise one-to-one subfigure-caption correspondence through bounding box coordinate alignment.<sup>47</sup> ImageCLEF data required minimal processing, with only irrelevant images filtered, as slice-text pairs were directly utilized.

In total, our curated dataset includes 24,333 unique subjects: 23,849 from PubMed Central (116,296 2D MRI slice-text pairs), and 484 from ImageCLEF (4,631 2D MRI slice-text pairs).

## S1.1.2 Dataset construction from medical institutions

To address the scarcity of rare brain tumor types/subtypes, the need for a prospective validation study and the limited availability of expert-curated radiology reports in online repositories, we curated data from twelve collaborating hospitals (Supplementary Table 1). For each patient with a pathologically confirmed diagnosis, we collected de-identified multi-parametric MRI scans (T1, T1c, T2, T2-Flair) that are commonly used for CNS tumor diagnosis, along with de-identified radiology reports and associated pathological diagnoses. In total, we gathered multi-modal data from 8,813 patients at the primary Xiangya Hospital, 1,334 patients from 11 other independent institutions, and 1009 patients in the prospective validation cohort (713 patients from Xiangya Hospital (May 2025 to June 2025), 247 patients (Jan 2025 to Feb 2026) from First Affiliated Hospital of Nanchang University and 49 patients from Changde First People’s Hospital). Demographic and statistical summaries of these medical centers are provided in Figure 1 and Supplementary Tables 1, 3, 4.

The original institutional radiology reports were in Chinese and often included descriptions of other modalities (e.g., Magnetic Resonance Angiography (MRA), Magnetic Resonance Spectroscopy (MRS), Blood Oxygen Level Dependent Imaging (BOLD) signals; Supplementary Table 7). Thus, we implemented a four-stage standardization protocol to process these reports (Supplementary Figure 1c). Firstly, we used regular-expression filters to eliminate non-T1/T1c/T2/T2-Flair content. Secondly, we developed a clinician-validated bilingual lexicon (Chinese-English) to accurately map medical terms, translating terms such as “T2 ⻓信号” to “T2 Hyperintense” and

“脑实质” to “brain parenchyma”. Thirdly, leveraging this lexicon as a knowledge base,

we utilized RAG techniques and prompt engineering strategies through the advanced capabilities of ChatGPT-4o,<sup>48</sup> to facilitate the translation of the reports from Chinese to English. To ensure terminological precision and minimize hallucinations, we implemented a structured, human-in-the-loop workflow. Following the initial generation, we employed an iterative multi-agent refinement loop where an ensemble of independent LLMs (DeepSeek<sup>49</sup>, ChatGPT, and Gemini<sup>50</sup>) cross-checked the outputs, automatically flagging errors that were fed back for targeted correction.<sup>51</sup> This selfcorrection cycle repeated until consensus was reached. Finally, the translated reports were thoroughly reviewed and refined by board-certified neuroradiologists to ensure their accuracy. To formally evaluate the overall translation quality and inter-reviewer agreement, we randomly sampled 200 generated reports. Five neuroradiologists independently assessed the holistic quality of each complete report using a standardized 5-level scoring rubric (ranging from Level 1: Unacceptable to Level 5: Clinically Accurate):

• Level 5: Clinically Accurate / Excellent – The translation is entirely accurate, precise, and requires no edits. It fully conveys the original clinical meaning.

• Level 4: Minor Inaccuracies / Good – Contains minor omissions or negligible inaccuracies that do not impact clinical interpretation or patient management. For example, if " 鞍 上 区 " (suprasellar region) was translated as "in sellar region" instead of "suprasellar region," this would receive a Level 4 rating, indicating a minor anatomical inaccuracy without significant clinical impact.

• Level 3: Moderate Errors / Fair – Contains errors that do not lead to fundamental misdiagnosis but significantly compromise report reliability, clarity, or completeness.

• Level 2: Significant Errors / Poor – Contains serious factual clinical errors that could potentially mislead clinical decision-making or patient care. For example, mistranslating "⽆强化" (no enhancement) as "enhancement present" would fall into this category.

• Level 1: Unacceptable / Harmful – Report content is completely detached from or contradictory to the actual imaging findings, potentially leading to patient harm or incorrect treatment.

Under the five-reader evaluation, 97.5% (195/200) of the translated reports were rated Level 5 by all readers (clinically accurate, no edits required). The remaining 2.5% (5/200) contained at least one Level 4 rating (minor inaccuracies without clinical impact). No report received any Level 1-3 rating. To quantify expert agreement while accounting for the highly skewed rating distribution, we computed a Gwet’s AC1 statistic of 0.977.<sup>52,53</sup> According to standard benchmarks, this represents almost perfect inter-rater agreement, confirming the high clinical reliability of the translated institutional dataset.

## S1.2 Dataset composition and curation framework of BrainTumor48K

To further enhance clarity and provide a comprehensive understanding of our dataset composition and processing, we have developed a detailed dataset organization pipeline. As illustrated in Supplementary Figure 2, BrainTumor48K comprises 24,716 2D MRI cases and 23,231 3D MRI cases. The 2D MRI data includes image slices paired with captions or labels from online repositories; these are used primarily for representation learning and not for clinical supervision.

Of the 23,231 3D MRI cases in our dataset, 18,113 are paired with both radiology reports and pathologically confirmed diagnostic labels, while the remaining 5,118 consist of 3D MRI scans and labels exclusively for classification tasks. The 18,113 cases are paired with radiology reports and pathologically confirmed diagnosis labels. Of these, 11,156 cases originate from our institutional archives, and 2,606 cases are sourced from online Radiopaedia, all accompanied by expert-curated radiology reports. Collectively, these 13,762 cases are characterized by high-quality imaging, professional radiology reports, and verified diagnostic labels, forming the backbone of our model’s clinical validity.

For the remaining 4,351 3D MRI-report cases, their reports were generated with the assistance of large language models (LLMs). Importantly, this AI-assisted generation was not free-form writing: it was strictly confined to converting existing patient metadata (e.g., tumor type, tumor location, and, where available, signal intensity) into a structured radiology-report format. No new clinical information was introduced at any stage. The process only transformed or standardized information already present in the original dataset (detailed in Supplementary Section S1.1.1 and illustrated in Supplementary Figure 1a).

In summary, 13,762 3D MRI-report pairs (11,156 institutional + 2,606 expert public) are derived from purely professional sources, forming a robust, high-quality foundation. The remaining 4,351 pairs are structured, metadata-derived reports that contain no fabricated information.

## S1.3 Ethics approval for datasets from medical institutions

All data obtained from participating private medical institutions were obtained with ethics approval, including the Medical Ethics Review Committee of Xiangya Hospital Central South University (2025030464), the Medical Ethics Committee of Changde Hospital, Xiangya School of Medicine, Central South University (The First People’s Hospital of Changde City) (2025-290-01), the ShanghaiTongji Hospital (Tongji Hospital of Tongji University) Ethics Committee (K-W-2025-008), the University of South China Second Hospital Clinical Research Ethics Review Committee (2025057), the Shenzhen Second People’s Hospital Clinical Research Ethics Committee (2025- 748-01PJ), the Third Xiangya Hospital Central South University Ethics Committee (25485), the Jiangxi Provincial People’s Hospital Medical Ethics Committee (2025(100)), the Chongqing Traditional Chinese Medicine Hospital Medical Ethics Committee (2025-IIT-KS-19), the The First Affiliated Hospital of Nanchang University Medical Ethics Committee (IIT[2025]-787), the Hunan Provincial Children’s Hospital Ethics Review Committee (HCHLL-2025-249), the First Affiliated Hospital of Lanzhou University Clinical Research (Pharmaceuticals, Devices) Ethics Committee (LDYYLL2025-2038), and the Second Affiliated Hospital of Nanchang University Biomedical Research Ethics Committee (O-[2025]-233). The clinical trial protocol is registered under the identifier NCT07126821. All data were deidentified in compliance with institutional policies and ethical standards

## S1.4 Patient and public involvement

Patients and the public were not involved in the design, conduct, reporting, or dissemination of this research. This observational study used only anonymized data from brain tumor patients, with no direct patient or public involvement in either the retrospective or prospective components. Patient and public involvement will be considered in future clinical trials and during further clinical development of the diagnostic model.

## S1.5 Estimation of total subject count from online resources

Accurately estimating subject-level statistics from the aggregated 39 online resources is challenging due to the inconsistent presence or absence of unique patient identifiers. For resources that do provide explicit patient identifiers, such as TCIA and BraTS, patient counts were directly extracted. In contrast, for web-crawled resources like PubMed Central<sup>42</sup> and ImageCLEF,<sup>54</sup> we conservatively assigned one unique patient per composite figure. This methodology likely results in an underestimate, as subfigures within composites may pertain to distinct cases, and images from the same video may originate from different patients. For certain directly downloaded datasets, including the Br35H,<sup>21</sup> Brain Tumor Classification Dataset,<sup>24</sup> and Brain Tumor MRI Images 44 Classes, which lack patient identifiers, we excluded these from our subject count estimation. The detailed number of patients and their associated multi-modal data in each online resource are illustrated in Supplementary Table 2.

## S2 Model Development

## S2.1 Development of BrainVLM for tumor classification and report generation

Developing an AI model for early brain tumor diagnosis necessitates the effective integration of presurgical multimodal data, including multi-parametric MRI scans, patient metadata, and radiology reports. Current AI-based brain diagnostic approaches largely rely on the subjective interpretation of pure MRI data, sometimes supplemented with patient metadata.<sup>55,56</sup> This neglects the rich, complementary information available in multi-modal datasets, potentially limiting the model’s performance. Besides, the current vision language foundation models $( { \mathrm { V L M s } } ^ { 5 7 } )$ face challenges as the input length increases with the quantity and resolution of images. This is particularly alarming in brain tumor studies, which require the analysis of high-resolution (HR), multiparametric MRI scans (T1, T1c, T2, T2-Flair) from various views (axial, coronal, sagittal).

To address this, we developed BrainVLM, a first-of-its-kind VLM that effectively synergizes multi-modal medical visual and textual information. It tackles the challenges through optimization across three aspects: (1) Innovative multi-modal model architecture: Incorporating anatomical detail-aware vision encoder and hybrid causalfull attention strategies into a multi-modal network to handle complex multi-modal data efficiently. (2) Clinical-motivated data standardization: Developing a standardized dataset processing protocol that emulates clinical practices and incorporates special markers to differentiate different visual data. (3) Progressive three-stage training strategy: Using a progressive training approach to incrementally enhance the model’s multi-modal processing capabilities.

## S2.1.1 Multi-modal model architecture

As shown in Fig. 1b, BrainVLM consists of three core components: (1) an anatomical detail-aware vision encoder for efficient, fine-grained image feature extraction, (2) a 2- layer Multilayer Perceptron (MLP) projector that bridges the image encoder with the LLM, and (3) a hybrid causal-full attention-based LLM for reasoning and understanding the multi-modal information.

Anatomical detail-aware vision encoder. We employed the vision backbone of BioMedCLIP<sup>58</sup> as the basic visual encoder E. This model constitutes a Vision Transformer (ViT) pre-trained on a large-scale 2D medical imaging dataset using vision-language contrastive learning.<sup>59</sup> The standard ViT of original BioMedCLIP accepts only 224 × 224 input dimensions and cannot directly process high-resolution (HR) MRI slices (typically 400–500 pixels in width/height). Direct downsampling risks losing critical diagnostic details such as tumor margins and signal heterogeneity, while splitting the image into several sub-images for individual encoding results in excessive visual tokens for LLMs. This is computationally prohibitive due to the transformer’s quadratic complexity. We therefore designed an anatomical detail-aware vision encoder to capture finer anatomical details while maintaining computational efficiency.

Our designed visual encoder incorporated two key components: mixed-resolution visual encoding and attention-based HR injection. Let I denote an input MRI scan with dimensions $N \times 3 \times w _ { o } \times h _ { o }$ , where N represents the number of slices $( \mathrm { e . g . , } N = 1$ for single-slice MRI; $N = 3 2$ for a 3D MRI volume with 32 MRI slices), and $w _ { 0 } \times h _ { 0 }$ specifies the original spatial resolution. We first generated two resized versions: 1) a low-resolution (LR) input $I ^ { l }$ scaled down to $N \times 3 \times 2 2 4 \times 2 2 4$ , and 2) an HR input $I ^ { h }$ scaled up to size $N \times 3 \times 4 4 8 \times 4 4 8$ . The LR input was encoded through the visual encoder E to generate global LR tokens $z ^ { l } = E ( I ^ { l } ) \in \mathbb { R } ^ { N \times 7 6 8 }$ . Here $z ^ { l }$ is extracted from the final layer’s [CLS] token of the ViT, capturing global patterns.

Next, $I ^ { h }$ was partitioned into four non-overlapping HR sub-inputs $\{ I _ { k } ^ { h } \} _ { k = 1 } ^ { 4 }$ using a grid method,<sup>60</sup> where � indexes the HR sub-inputs. Each HR sub-input was resized to $N \times 3 \times 2 2 4 \times 2 2 4$ and independently encoded by E to generate HR visual tokens as $z ^ { h _ { k } } = E ( I _ { k } ^ { h } )$ . Unlike the global [CLS] token extraction used for the LR input, we retained all spatial patch tokens (excluding [CLS]) from the $\mathrm { V i T } { \bf \ ' } _ { \bf S }$ final layer when processing HR sub-inputs. This yields features $z ^ { h _ { k } } \in \mathbb { R } ^ { N \times 1 9 6 \times 7 6 8 }$ that encode finegrained anatomical details. The final combined HR representation is denoted as $z ^ { h } =$ $\{ z ^ { h _ { k } } \} _ { k = 1 } ^ { 4 } \in \mathbb { R } ^ { 4 \times N \times 1 9 6 \times 7 6 8 }$

Finally, we injected $z ^ { h }$ into $z ^ { l }$ via an LR-HR cross-attention module, to obtain HR-informed LR tokens $z ^ { h l }$ . Here, $z ^ { l }$ served as query Q, $\{ z ^ { h _ { k } } \} _ { k = 1 } ^ { 4 }$ served as the context (key K and value V), resulting the final $z ^ { h l } \in \mathbb { R } ^ { N \times 7 6 8 }$ . This approach merged global context $z ^ { l }$ with anatomical details from $z ^ { h }$ , while maintaining computational efficiency.

For multi-parametric MRI based brain tumor diagnosis (e.g., T1, T1c, T2, T2-Flair), each 3D MRI sequence was processed independently by our encoder to extract modality-specific features, denoted as $z _ { m } ^ { h l } \in \mathbb { R } ^ { N \times 7 6 8 }$ . These features were then concatenated along the slice dimension: $z ^ { \mathrm { c o m b } } = \big \| _ { m = 1 } ^ { M } z _ { m } ^ { h l } \in \mathbb { R } ^ { ( M \cdot N ) \times 7 6 8 }$ ,. where M denotes the number of MRI sequences. For example, a 5-sequence protocol (e.g., T1 axial, T1c axial, T1c sagittal, T2 axial, T2-Flair axial) with $N = 3 2$ slices per sequence yielded a combined representation $z ^ { \mathrm { c o m b } } \in \mathbb { R } ^ { 1 6 0 \times 7 6 8 }$ , integrating complementary diagnostic information across sequences.

MLP projector. The combined feature representation was mapped into the<sup>comb</sup>z LLM language space via a 2-layer MLP projector, converting visual information into a format digestible for LLM. This transformation bridged the modality gap by converting anatomical visual patterns into a sequential token format compatible with linguistic processing.

Hybrid causal-full attention based LLM. The MLP-projected visual embeddings were combined with patient metadata (e.g., age and sex) and textual instructions to form a multimodal prompt. This prompt was then processed by our adapted LLM for tumor classification, uncertainty quantification, and report generation. We utilized the pretrained Llama-3.1-8B-Instruct<sup>61</sup> model as our base LLM, which provides a mainstream language backbone with grouped query attention and open-source adaptability. Traditional causal attention in Llama-3.1-8B-Instruct restricts tokens to preceding positions only, which is suboptimal for MRI interpretation.<sup>62</sup> Therefore, we implemented a hybrid attention strategy following previous works.<sup>63,64</sup> Specifically, full attention was applied to image tokens for capturing complex visual interdependencies within the same 3D MRI sequence or 2D MRI slice, while causal attention was retained for sequential tokens to preserve data structure. This hybrid architecture enabled the model to simultaneously capture cross-modal visual relationships and maintain linguistic coherence.

## S2.1.2 Clinical-motivated data standardization protocol

Language data in LLM is typically uniform and consistent, while our task involves more complex visual data. Randomly feeding unprocessed neuroimaging inputs from different modalities and views into BrainVLM complicates the learning process and impedes model convergence. Thus, we proposed a standardized dataset processing protocol that emulates clinical practices through the following three key strategies: 1) volumetric slice-count harmonization to ensure uniform slice dimensions, 2) clinical MRI sequence integration mirroring radiological workflows, and 3) special markers to differentiate multi-modal data and task-driven prompt design to differentiate different tasks. This protocol reduced learning ambiguity while preserving diagnostically critical features essential for brain tumor analysis.

MRI preprocessing via volumetric slice-count harmonization. The 3D MRI sequences used in our study typically exhibit substantial variability in the number of acquired slices. This variation occurs both across scanners/institutions (ranging from dozens to hundreds of slices per sequence) and between sequences for the same subject (e.g., 20 slices in native T1 vs. 40 in T1c sequences). This substantial slice variation significantly complicates model training. We therefore standardized each 3D MRI sequence to 32 slices using cubic spline interpolation (implemented via SimpleITK<sup>65</sup>), ensuring smooth volumetric resampling along the slice direction. Critically, unlike conventional neuroimaging pipelines that typically involve extensive preprocessing— such as resampling to isotropic voxels and skull stripping — BrainVLM performs only this volumetric slice-count harmonization. This minimal preprocessing preserves the complete information content of the original MRI data, making it particularly wellsuited for our comprehensive brain tumor characterization study. While conventional skull-stripping may aid certain tumor analyses, it is not universally appropriate. Crucially, lesions located in regions like the pia mater or invading the cranial vault can be inadvertently removed during skull stripping, leading to the loss of critical anatomical and pathological information – a risk our approach deliberately avoids (Supplementary Figure 3).

Clinical MRI sequence integration. Emulating the clinical workflow for MRI-based tumor identification is critical for aligning machine learning models with diagnostic practice, thereby improving lesion identification and characterization.<sup>66,67</sup> In clinical practice, comprehensive brain tumor assessment often initiates with acquiring axial T1 and T2 sequences, optionally complemented by axial T2-Flair imaging. Following intravenous administration of gadolinium-based contrast agents, multiple-plane T1c (axial, sagittal, and coronal views) are obtained to evaluate enhancement patterns and blood-brain barrier disruption. Based on initial findings or clinical suspicion, additional sequences or planes (e.g., coronal or sagittal T1/T2 for sellar region lesions) may be acquired. During interpretation, neuroradiologists typically begin by examining T2 or T2-Flair images to identify potential lesions. They then use T1c images to detect small lesions and assess tumor enhancement, comparing these with T1 images of the same view for confirmation.

To emulate this clinical workflow and streamline model learning, we organized the MRI data to reflect both the clinical acquisition and interpretation workflows. Specifically, our integration strategy prioritizes clinically meaningful comparisons: the primary input consisted of paired native T1 and T1c scans acquired in the same plane, facilitating direct assessment of lesion enhancement. This core pairing was supplemented by representative T2 and T2-Flair scans, as well as an additional T1c scan obtained from a different imaging plane than the primary T1/T1c pair. For example, if the primary T1/T1c pair was axial, we included an additional T1c scan from the coronal or sagittal plane to provide multiplanar context. When specific sequences or planes were unavailable, alternative available sequences were included to maximize diagnostic coverage. In summary, BrainVLM utilized five core 3D MRI sequences as visual input—T1 (axial or another view), T1c from the same view as T1, T2 (axial or another view), T2-Flair (axial or another view), and an additional T1c from a different view than T1. This structured yet flexible approach maintained comprehensive diagnostic information while faithfully reflecting real-world clinical reasoning. Further details regarding the processing of cases with more than five MRI sequences are provided in the following Model Inference Section S2.3.

Special tokens for data type differentiation and task-driven prompt design. BrainVLM processed heterogeneous inputs spanning 2D MRI slices, 3D MRI volumes, and free-text patient metadata (e.g., age, sex), while concurrently addressing distinct task types including classification, report generation, and reliability assessment. To enable the model to distinguish between these data types and tasks, we implemented the following specialized tokens and task-driven prompts. Specifically, to help the model distinguish different data types, we designed special tokens: 2D single image slices were enclosed with ’<image>’ and ’</image>’, while 3D MRI volumes used ’<vid>’ and ’</vid>’ tags. A special token ’<s>’ was inserted between volumes to indicate the different sequence’s input. We also introduced specialized tumor tokens (e.g., ’<class\_0>’ for MET) to encode the original tumor names for classification. Concurrently, to enhance the model’s task-specific learning capabilities, we designed customized prompts that explicitly inform the model of the input structure and expected objectives (see Supplementary Table 9). Specifically, the finalized prompt that was input into BrainVLM integrates three components: (1) predefined special visual tokens (e.g., <vid>, <image>) to structure the imaging inputs; (2) patient metadata; and (3) task-specific instructions that explicitly define the model’s objective for each task (e.g., generating a clinical description for patients in the radiology report generation task or making a self-assessment confidence score for the reliability task). These special tokens and unified prompt design ensure robust alignment between input interpretation and clinical task requirements, enhancing the model’s interpretative accuracy and adaptability to diverse diagnostic objectives.

Unified task-aware tokenization and autoregressive decoding mechanism. BrainVLM is designed to perform simultaneous multi-task generation within a single autoregressive pass. The following section details the step-by-step decoding mechanism enabled by our LoRA fine-tuned architecture. The complete data flow, encompassing input processing, tokenization, and sequential output generation, is visualized in Supplementary Figure 4. BrainVLM receives multi-modal input, including multi-parametric MRI scans, patient metadata, and an instruction prompt. MRI scans are processed by a shared vision encoder, and the extracted visual features are then projected into a sequence of visual tokens, aligning them with the LLM’s token embedding space. Concurrently, patient metadata and instructional prompts (e.g., “Generate report, diagnosis, and uncertainty for this MRI.”) are tokenized into discrete tokens using a Byte-Pair-Encoding (BPE) tokenizer. These visual and textual tokens are then concatenated and fed into the LLM decoder (fine-tuned with LoRA) for autoregressive generation. The LoRA fine-tuned LLM generates output in an autoregressive manner, guided by special delimiter and class tokens that structure the generation process. Specifically, to enable the simultaneous output of three distinct components (i.e., report, diagnosis, and uncertainty), we introduce three special start tokens into the vocabulary: <Report>, <Diagnosis>, and <Uncertainty>. During autoregressive generation, the model generates tokens sequentially. It first produces the radiology report tokens after <\Report>. These report tokens are standard BPE subword units that, once the report section is complete, are later decoded into a humanreadable report via the BPE tokenizer.

After the report is complete, the model encounters <\Report> and transitions to generating the diagnosis. For diagnosis, we added 14 specialized classification tokens (e.g., <class\_0> through <class\_13>) to the vocabulary, each corresponding to one of the 11 WHO CNS5 tumor grades or to one of the three gliomas, glioneuronal tumors, and neuronal tumors (GGN) subtypes. When the model outputs a token like <class\_5>, it is then directly mapped to the corresponding tumor label (e.g., “MEN” for meningioma) via a simple lookup table.

Following the diagnosis token, the model reaches <Uncertainty> and generates the confidence score as a sequence of numerical BPE tokens (e.g., “0”, “.”, “9”). Specifically, during the training, the confidence score was discretized into six levels (e.g., 50%, 60%, 70%, 80%, 90%, 95%) to simplify learning; at inference, the model outputs the appropriate token(s) that correspond to one of these levels. These tokens are concatenated and decoded into a human-readable confidence score (e.g., 0.9). Thus, all three components are generated in a single autoregressive pass. The BPE tokens for the report are decoded into free-text, the specialized classification token is mapped directly to a diagnosis label, and the numerical tokens are converted to a confidence score.

## S2.1.3 Three-stage progressive training strategy of BrainVLM

Our curated BrainTumor48K dataset comprises three distinct data types: 53.14% 2D MRI slices paired with text descriptions, 38.14% multi-parametric 3D MRI volumes with radiology reports, and 8.72% 3D MRI sequences annotated solely with diagnostic labels. Direct end-to-end training on this heterogeneous data composition may overwhelm models with conflicting objectives. The 2D isolated MRI slices lack volumetric context essential for tumor characterization, while complete volumetric inputs demand foundational slice-level recognition capabilities that 2D training establishes. To address these challenges, we implemented a curriculum learning paradigm with three progressive training stages. This strategy enables the model to progressively interpret complex visual patterns, starting from simple 2D slices analysis and advancing to more intricate 3D volumetric data interpretation (see Supplementary Figure 5a).

Training stage 1: 2D MRI slice-text representation learning. Volumetric tumor analysis fundamentally requires slice-level recognition capabilities, a skill radiologists cultivate through detailed examination of individual-MRI slices. To establish this foundation while maximizing BrainTumor48K’s heterogeneous resources, we initiated BrainVLM’s training with a focus on 2D MRI slice-level representation learning. During this phase, we optimized only the MLP projector and the LR-HR cross-attention module within our anatomical detail-aware encoder, keeping the vision encoder and LLM module frozen to maintain stability. This stage focused on training 2D MRI slicetext alignment and utilized two main sources: (i) all available 2D MRI slice-text pairs in BrainTumor48K, and (ii) strategically extracted tumor slices from 3D MRI volumetric sequences, paired with AI-generated slice-level text descriptions. For public datasets with tumor masks, such as BraTS, we extracted slices with more than 5% tumor occupancy and generated clinically validated text descriptions using RAG, refined by ChatGPT-4.0. For institutional scans, neuroradiologists manually selected pathologically significant slices from each 3D MRI sequence, and the corresponding slice-level reports were crafted using RAG refined by ChatGPT-4.0. Finally, this meticulous process resulted in 10,996,702 curated 2D MRI slice-text pairs. This 2D MRI slice-level representation learning approach enabled BrainVLM to establish fundamental tumor recognition capabilities, laying a solid foundation for subsequent advanced 3D volumetric analysis.

Training stage 2: Volumetric context integration via hybrid 2D-3D training. In this phase, we aimed to transfer the learned 2D representation capabilities into the tomographic 3D MRI data space, facilitating BrainVLM’s 3D tumor characterization capability. We utilized all available 2D and 3D training datasets from BrainTumor48K and designed a dynamic learning strategy interleaving 2D slice-level and 3D volumetric tasks during training. Crucially, at each iteration, we stochastically sampled training instances with 30% probability allocated to 2D tasks (leveraging isolated slices for tumor description and slice-level diagnosis) and 70% probability to 3D tasks (utilizing full sequences for volumetric report generation, classification, and diagnosis). During this stage, the MLP projector, LR-HR cross attention module, and Low-Rank Adaptation (LoRA)<sup>68</sup> parameters in the LLM model were trained, while the vision encoder and other parameters in the LLM model remained frozen. This transitional stage ensures that the model develops a cohesive understanding of tumor characteristics across different dimensions.

Training stage 3: Comprehensive 3D volumetric instruction tuning. In the final stage, we focused on leveraging the full high-quality 3D data available in the BrainTumor48K dataset to enhance BrainVLM’s advanced volumetric analysis capabilities. The projector, LR-HR cross-attention module and LoRA parameters in the LLM model were trained, while all other parameters remained frozen.

## S2.1.4 Loss function

We introduced dual-objective losses to unleash BrainVLM’s generative and discriminative capabilities across clinical tasks, including free-text generation and dedicated tumor classification. Specifically, we first employed a standard causal language modeling loss $L _ { \mathrm { { L M } } }$ , which enables the model to follow natural language instructions and generate coherent diagnostic texts. Given a token sequence $( \boldsymbol { w } _ { i } ^ { ( 1 ) }$ $w _ { i } ^ { ( 2 ) } , . . . , w _ { i } ^ { ( T ) } )$ in a training sample $x _ { i }$ , this loss is formulated as:

$$
L _ { _ { \mathrm { { L M } } } } = - \sum _ { t = 1 } ^ { T } \log P ( w _ { i } ^ { ( t ) } \mid w _ { i } ^ { ( 1 ) } , w _ { i } ^ { ( 2 ) } , . . . , w _ { i } ^ { ( t - 1 ) } ; \theta ) ,
$$

where $P ( w _ { i } ^ { ( t ) } \mid w _ { i } ^ { ( 1 ) } , w _ { i } ^ { ( 2 ) } , . . . , w _ { i } ^ { ( t - 1 ) } ; \theta )$ denotes the predicted probability of token $w _ { t }$ given the preceding tokens. $\theta$ denotes the model parameters. $t = 1 , . . . , T$ denotes the tokens in the sequence. This objective encourages contextual understanding and fluent text generation by minimizing the negative log-likelihood of ground-truth tokens.

Furthermore, to enhance the tumor-specific discrimination, we introduced an auxiliary classification loss $L _ { \mathrm { { C L } } }$ to perceive fine-grained medical patterns. We added special tokens to encode the original tumor label (such as <tumor\_3> token for Glioma), and employed a classification loss based on these tokens. This loss is computed as:

$$
L _ { \mathrm { c L } } = - \sum _ { c = 1 } ^ { 1 4 } y _ { c } ^ { ( i ) } \log \hat { y } _ { c } ^ { ( i ) } ,
$$

where $y _ { c } ^ { ( i ) }$ denotes the ground-truth label for the c-th tumor class, and $\hat { y } _ { c } ^ { ( i ) }$ is the predicted probability. This loss enables fine-grained recognition across 14 tumor categories, including 11 major brain tumor types in WHO-CNS5 and 3 GNN subcategories.

The overall objective combines both losses:

$$
L _ { \mathrm { \ t o t a l } } = L _ { \mathrm { \scriptsize { L M } } } + \alpha L _ { \mathrm { \scriptsize { C L } } } ,
$$

where α is a tunable coefficient balancing language generation and tumor classification. This joint optimization framework enables BrainVLM to effectively capture both general linguistic semantics and task-specific diagnostic cues.

## S2.1.5 Data augmentation

Brain MRI data exhibited domain shifts driven by patient-specific factors (e.g., age, sex, physique) and MRI hardware variability,<sup>69</sup> resulting in significant intra-modality variation (Supplementary Figure 6). We applied tailored augmentation strategies for both 2D and 3D MRI data to improve BrainVLM’s generalization across diverse clinical settings. For 2D MRI slices, we adopted standard augmentation techniques,<sup>70</sup> including random flipping, cropping, and intensity transformations. For 3D MRI sequences, we first identified key challenges such as noise artifacts, intensity inhomogeneity, and motion-induced blurring (examples shown in Supplementary Figure 6). Then, to improve robustness against these effects, each 3D training sample was augmented with: (1) random rotation within ±60 degrees, and (2) intensity variation, motion blur, or random noise, each applied with 30% probability.

## S2.2 Model reliability implement

Brain tumor diagnosis is a high-stakes task, where a misdiagnosis can lead to inappropriate treatments, prolonged patient suffering, and potentially fatal outcomes. However, conventional VLMs may generate erroneous predictions with high confidence,<sup>71</sup> undermining clinical trust and limiting practical utility. As illustrated in the left panel of Fig. 2c in the main manuscript, our vanilla BrainVLM model (without reliability mechanisms) also exhibited this concerning behavior, frequently assigning high confidence scores to erroneous predictions. To address this critical limitation, we presented an uncertainty quantification strategy that calibrates prediction confidence in BrainVLM: accurate diagnoses receive low uncertainty scores (high confidence score), while erroneous predictions are assigned high uncertainty (low confidence score). This ensures only high-confidence, reliable predictions are acted upon.

Existing uncertainty quantification approaches for VLMs and LLMs rely on multiple sampling strategies (e.g., Best-of-N) that incur substantial computational overhead regardless of question complexity.<sup>72</sup> While adaptive sampling techniques attempt to reduce costs through early-stopping heuristics, they remain constrained by manually designed rules that lack generalizability across models and tasks.<sup>73</sup> We introduce a consensus-driven confidence calibration strategy that directly transforms model disagreement into calibrated confidence signals, eliminating the need for heuristic rules or fixed sampling costs. Our approach features two key innovations: (1) Consensus-driven reliability dataset construction, where a reliability dataset is automatically constructed using agreement statistics from multiple model instances via consensus-driven clustering; (2) Confidence-aware reliable BrainVLM finetuning: Distillation of consensus-derived reliability knowledge into our BrainVLM framework via parameter-efficient fine-tuning. We refer to the baseline BrainVLM without uncertainty quantification as the vanilla BrainVLM, while our enhanced version incorporating the proposed uncertainty quantification strategy is termed reliable BrainVLM.

## S2.2.1 Consensus-driven reliability dataset construction

To create a high-quality reliability dataset for confidence calibration, we leveraged the natural training dynamics of BrainVLM. Specifically, when our best-performing model was near convergence, we sampled N model checkpoints at different training iterations to create an ensemble of N vanilla BrainVLM variants. For each training sample $x _ { i }$ with pathological label $y _ { \mathrm { t r u e } } ^ { ( i ) }$ , we independently processed $x _ { i }$ through all N models to obtain N diagnostic predictions $\{ \hat { y } _ { i } ^ { ( 1 ) } , . . . , \hat { y } _ { i } ^ { ( N ) } \}$ per sample (see Supplementary Figure 5b). These responses were then clustered by predicted diagnoses, with the largest cluster defining the consensus label $y _ { \mathrm { c o n s } } ^ { ( i ) }$ . Formally, the consensus label is computed as:

$$
\begin{array} { r } { y _ { \mathrm { c o n s } } ^ { ( i ) } = a r g m a x \sum _ { k = 1 } ^ { N } \mathbb { I } ( \hat { y } _ { i } ^ { ( k ) } = c ) , } \end{array}
$$

where C denotes the set of all possible diagnostic classes and � indicates the indicator function. This consensus label represents the prediction with the strongest inter-model agreement. We then computed a confidence score ${ S } _ { \mathrm { c o n } } ^ { ( i ) }$ for each sample $x _ { i }$ :

$$
s _ { \mathrm { c o n } } ^ { ( i ) } = \frac { N - N _ { w } ^ { ( i ) } } { N } ,
$$

where $\begin{array} { r } { N _ { w } ^ { ( i ) } = \sum _ { k = 1 } ^ { N } \mathbb { I } ( \hat { y } _ { i } ^ { ( k ) } \neq y _ { \mathrm { t r u e } } ^ { ( i ) } ) } \end{array}$ , which indicates the number of incorrect predictions among the N models. A high $N _ { w }$ implies greater disagreement, resulting in a lower confidence score, while lower $N _ { _ w }$ yields higher confidence scores.

For each sample $( x _ { i } , y _ { \mathrm { t r u e } } ^ { ( i ) } )$ , the above process yields per-sample supervision tuples $( x _ { i } , y _ { \mathrm { c o n s } } ^ { ( i ) } , s _ { \mathrm { c o n } } ^ { ( i ) } )$ , where $y _ { \mathrm { c o n s } } ^ { ( i ) }$ represents the ensemble-aggregated prediction, and ${ S } _ { \mathrm { c o n } } ^ { ( i ) }$ quantifies reliability based on inter-model consensus strength $y _ { \mathrm { c o n s } } ^ { ( i ) }$ and clinical validity alignment with $y _ { \mathrm { t r u e } } ^ { ( i ) }$ . Collectively, these tuples constitute the reliability dataset, which provides reliable supervision for training BrainVLM to generate well-calibrated, confidence-aware predictions.

## S2.2.2 Confidence-aware reliable BrainVLM finetuning

After the construction of the reliability dataset, we then distilled the consensus-derived reliability knowledge into our reliable BrainVLM through confidence-aware reliable fine-tuning. Specifically, as illustrated in Supplementary Figure 5b, we transformed the vanilla BrainVLM into reliable BrainVLM by parameter-efficient fine-tuning on our constructed reliability dataset. In this process, we also reformulated the output scheme to generate an additional confidence score, yielding responses such as: “Report:..., diagnosis: <tumor\_i>. My confidence: 80%”. Crucially, we employed our curated tuples $( x _ { i } , y _ { \mathrm { c o n s } } ^ { ( i ) } , S _ { \mathrm { c o n } } ^ { ( i ) } )$ rather than ground-truth pairs $( x _ { i } , y _ { \mathrm { t r u e } } ^ { ( i ) } )$ as supervision signals, enabling simultaneous learning of diagnostic predictions and their associated confidence levels. During this fine-tuning process, we updated only the projector, LR-HR cross-attention module, and LoRA parameters within the LLaMA model, keeping both the vision encoder and LLaMA backbone frozen. The model was trained for two epochs on the reliability-annotated dataset.

## S2.3 Model inference

Let $S = \{ \mathrm { T 1 , T 1 c , T 2 , F L A I R } \}$ denote the set of MRI sequences. For each sequence $s \in S$ let $V _ { s } \subseteq \{ \mathrm { a x i a l , s a g i t t a l , c o r o n a l } \}$ represent the available views for ; s $V _ { s }$ may be empty if the corresponding sequence is missing. Given a patient, the complete input MRI space can be formulated as:

$$
X = \{ ( x _ { s } ^ { v } , s , v ) \mid s \in \mathcal { S } , v \in \mathcal { V } _ { s } \} , \ ( 1 )
$$

where $x _ { s } ^ { \nu }$ denotes the MRI image corresponding to sequence and view . AsS V described in Clinical MRI sequence integration Section S2.1.2), during the inference time, BrainVLMtakes five core MRI sequences—T1 (axial or another view), T1c from the same view as T1, T2, T2-Flair, and an additional T1c from a different view than T1—along with patient metadata and task-specific instructions as input. Formally, the

five core MRI sequences $C _ { m }$ satisfy:

$$
C _ { m } = \{ ( x _ { \mathrm { T 1 } } ^ { v _ { 0 } } , \Gamma 1 , v _ { 0 } ) , ( x _ { \mathrm { T 1 c } } ^ { v _ { 0 } } , \Gamma 1 \mathrm { c } , v _ { 0 } ) , ( x _ { s _ { 3 } } ^ { v _ { 3 } } , s _ { 3 } , v _ { 3 } ) , ( x _ { s _ { 4 } } ^ { v _ { 4 } } , s _ { 4 } , v _ { 4 } ) ( x _ { s _ { 5 } } ^ { v _ { 5 } } , s _ { 5 } , v _ { 5 } ) \} ,\tag{2}
$$

where $\nu _ { 0 } , \nu _ { 3 } , \nu _ { 4 } , \nu _ { 5 } \in V _ { s }$ are chosen from available views, and the selection follows these criteria:

$v _ { 0 } \in \mathcal { V } _ { \mathrm { T 1 } }$ is any available view of T1 (e.g., axial, sagittal, or coronal).

 The first two items are T1 and T1c from the same view $v _ { 0 }$ .

 For $s _ { 4 }$ and $s _ { 5 }$ ,select from {T2, T2-Flair} according to availability:

If both T2 and T2-Flair are available, then $s _ { 3 } = \mathrm { T } 2 , s _ { 4 } = \mathrm { T } 2 \mathrm { - } \mathrm { F l a i r } _ { \mathrm { + } }$ , with $v _ { 3 } \in$ $\mathcal { V } _ { \mathrm { T 2 } } , ~ v _ { 4 } \in \mathcal { V } _ { \mathrm { T 2 - F l a i r } } .$

\- If only one of T2 or T2-Flair is available, assign it to $s _ { 3 }$ and select any other available MRI sequence for $s _ { 4 }$

\- If neither T2 nor T2-Flair is available, select any other available MRI sequences for $s _ { 3 }$ and $s _ { 4 }$ .

 For $s _ { 5 }$ , prioritize selecting T1c from any view $v _ { 5 } \in \mathcal { V } _ { \mathrm { T 1 c } }$ such that $v _ { 5 } \neq v _ { 0 }$ . If not available, select any unused view from the remaining available sequences.

This selection strategy dynamically adapts to the available sequences and views for each patient, ensuring robust performance even in the presence of missing sequences. During inference, BrainVLM takes the selected five core MRI sequences, along with patient metadata and task-specific instructions, as input. It then generates a diagnostic prediction, a radiology report, and an associated confidence score.

Adaptive sequence sampling and aggregation. In real-world clinical practice, MRI studies frequently employ thick slice acquisitions spanning multiple anatomical planes (e.g., axial, sagittal, and coronal views for both T1 and T2 sequences), yielding studies exceeding the standard five-sequence input. To address this variability, we define an adaptive sequence sampling strategy as follows.

Let be the set of all valid five-sequence combinations constructed fromC X according to the above selection criteria:

$$
C = \{ C _ { _ m } \subset X \parallel C _ { _ m } \left| = 5 \mathrm { a n d } C _ { _ m } \mathrm { s a t i s f i e s ~ t h e ~ s e l e c t i o n ~ c o n s t r a i n t s ~ a b o v e } \right. \} . ( 3 )
$$

For each $C _ { m } \in C$ , we define the model inference function:

$$
( y _ { m } , r _ { m } , s _ { c o n } ^ { m } ) = f _ { \theta } ( C _ { m } , d , t ) , ( 4 )
$$

where $f _ { \theta }$ defines the reliable BrainVLM parameters, denotes patient metadata,d t denotes the task-specific prompt, $y _ { m }$ is the predicted clinical diagnosis, $r _ { m }$ is the generated radiology report, and $s _ { c o n } ^ { m }$ is the confidence score. To aggregate results across all sampled combinations, we define the final clinical diagnosis $\hat { \boldsymbol y } ^ { ( 1 ) }$ as the most frequent prediction (“majority voting”) among all $\{ y _ { m } \} _ { C _ { m } } \in c$

$$
\hat { y } ^ { ( 1 ) } = \mathrm { m o d e } ( \{ \mathrm { y } _ { \mathrm { m } } \} _ { m = 1 } ^ { M } ) , ( 5 )
$$

where $\mathrm { \ m o d e ( \cdot ) }$ indicates the statistical mode (most frequent element) in the ensemble. Here $\hat { \boldsymbol y } ^ { ( 1 ) }$ is also known as the TOP 1 prediction. The final confidence $\hat { s } _ { c o n } ^ { 1 }$ that associated with the predicted diagnosis $\hat { \boldsymbol y } ^ { ( 1 ) }$ can be aggregated as:

$$
\hat { S } _ { c o n } ^ { ( 1 ) } = \frac { 1 } { N _ { 1 } } \sum _ { m = 1 } ^ { M } s _ { c o n } ^ { m } \cdot \delta ( y _ { m } , \hat { y } _ { 1 } ) , \quad N _ { 1 } = \sum _ { m = 1 } ^ { M } \delta ( y _ { m } , \hat { y } ^ { 1 } ) , ( 6 )
$$

where $\delta ( a , b ) = 1 { \mathrm { i f } } a = b { \mathrm { e l s e ~ } } 0$ . The final radiology report is generated by aggregating reports associated with the TOP 1 diagnosis. Specifically, since our report are structured into predefined sections (e.g., T1 signal characteristics), we apply majority voting within each section to select the most frequent content among all reports associated with the TOP 1 predicted diagnosis $r _ { m } \mid y _ { m } = \hat { y } ^ { ( 1 ) }$

This adaptive ensemble strategy ensures that the model can robustly process heterogeneous and incomplete clinical MRI acquisitions by (1) systematically sampling all valid five-sequence input sets from the available MRI space , (2) independentlyX inferring on each combination, and (3) performing robust aggregation via majority voting to determine the final (Top 1) clinical prediction.

## S2.3.1 Confidence-triggered TOP 2 supplementary diagnosis

In clinical workflows, neuroradiologists may consider multiple diagnostic possibilities when interpreting clinically ambiguous cases, such as “considering meningioma or glioma”. To emulate this process, we implement a confidence-triggered supplementary diagnosis mechanism. If the confidence score $\hat { S } _ { c o n } ^ { ( 1 ) }$ for the Top 1 prediction $\hat { \boldsymbol y } ^ { ( 1 ) }$ falls below a predefined threshold (set at 75% in this study), a secondary (top-2) diagnosis is also reported. The Top-2 diagnosis $\hat { y } ^ { ( 2 ) }$ is defined as: $\hat { y } ^ { ( 2 ) } = \mathrm { m o d e } ( \{ y _ { m } | y _ { m } \neq \hat { y } _ { 1 } \} )$ . Our experimental results in Supplementary Section S3.1 demonstrate the effectiveness of this approach in enhancing diagnostic reliability in ambiguous cases.

## S2.4 Finetuning of molecular subgroup classification task

In this phase, we fine-tuned BrainVLM using the stage-3 checkpoint, leveraging the adult-type diffuse glioma data from BrainTumor48K. We performed threefold validation on the Xiangya primary Dataset (n = 494; 333 IHD-w GBMs, 86 IDH-m Astros, 75 IDH-m ODGs), each fold consisted of 328 patients (222 IDH-w GBMs, 57 IDH-m Astros, 50 IDH-m ODGs) for training and 166 patients (111 IDH-w GBMs, 29 IDH-m Astros, 25 IDH-m ODGs) for validation. We followed the MRI sequences integration strategy defined in the Section S2.1.2 and S2.3 for molecular classification training and prediction. In this training stage, we updated only the projector, LR-HR cross-attention module and LoRA parameters within the LLaMA model, keeping other parameters frozen.

The results demonstrated BrainVLM’s robust capability to differentiate molecular subtypes of adult diffuse gliomas. BrainVLM achieved a frequency-weighted F1 score of 0.92 (95% CI: 0.91–0.93, Supplementary Figure 7c) in the threefold validation on the Xiangya primary dataset, outperforming Merlin (F1 = 0.76, 95% CI: 0.75–0.77) and RadFM (F1 = 0.74, 95% CI: 0.73–0.75). Across molecular subgroups, BrainVLM demonstrated superior performance: IDH-w GBMs (F1 = 0.95 vs. 0.88–0.91), IDH-m Astros (F1 = 0.89 vs. 0.53–0.62), and IDH-m ODGs (F1 = 0.84 vs. 0.58–0.63). On external validation, BrainVLM achieved a frequency-weighted F1 score of 0.88 (Supplementary Figure 7c), surpassing Merlin (F1 = 0.73) and RadFM (F1 = 0.69). Consistent with primary validation, BrainVLM showed higher performance across subgroups: IDH-w GBMs (F1 = 0.94 vs. 0.82–0.85), IDH-m Astros (F1 = 0.79 vs. 0.32– 0.44), and IDH-m ODGs (F1 = 0.69 vs. 0.27–0.34).

## S2.5 Competing methods

We compared BrainVLM with three state-of-the-art AI models—including two VLMs (RadFM<sup>74</sup> Merlin<sup>75</sup>) and the large language model ChatGPT-4o,<sup>48</sup> as well as a representative traditional deep learning model, Video Swin Transformer (VST).<sup>76</sup> RadFM utilizes a 12-layer 3D ViT as a vision encoder to process both 2D and 3D data, and uses MedLLaMA-13B as the large language model for reasoning and a tradtional deep learning method video-swin transformer. For our experiments, we fine-tuned the pretrained RadFM on our BrainTumor48K dataset, updating the LoRA parameters of MedLLaMA-13B and the projection module. Merlin is a 3D vision-language foundation model developed for abdominal CT interpretation. It was pretrained on both radiology reports and structured electronic health records (EHRs), using a 3D ResNet152 as the vision encoder and RadLlama-7B as the language model. The vision encoder was pretrained on a large-scale paired dataset of 3D CT scans and text reports using the InfoNCE loss. For our experiments, we fine-tuned Merlin’s vision encoder based on its pretrained 3D ResNet152 weights, and applied LoRA-based fine-tuning to the RadLlama-7B language model. Both RadFM and Merlin are designed for singlemodality radiology inputs. For a fair comparison, we adopted the same input strategy in BrainVLM as in these baseline models. Specifically, we provided five MRI sequences as input, processed each sequence through the respective vision encoders, and concatenated the extracted features before passing them to the language model. To ensure a rigorous and fair comparison, we extensively fine-tuned all baseline models on the BrainTumor48K dataset with sufficient hyperparameter optimization, training each model until convergence. For VST, we employed five dedicated modality-specific encoders for each modality required by BrainVLM (Five sequences, two T1c sequences in different planes, a non-contrast T1 matching one T1c plane, and T2 and T2f sequences from any available views). The high-dimensional features extracted from each encoder were concatenated and processed through a fully connected layer to produce the final 12-category diagnostic output.

ChatGPT-4o<sup>48</sup> does not natively support 3D medical imaging input. Therefore, for each test case, experienced clinicians manually selected the most representative tumorcontaining slice from each MRI sequence (four to five sequences per case). These clinician-validated slices were combined and provided as a set of 2D images to ChatGPT-4o for report generation and diagnostic prediction. It is important to note that this approach simplifies the diagnostic task, as tumor-containing frames are preidentified by experts, whereas BrainVLM, RadFM, and Merlin operate directly on the full 3D raw MRI volumes without manual selection.

Video Swin Transformer is a traditional deep learning model that has demonstrated strong performance in video understanding tasks and is widely used in medical imaging analysis.<sup>77</sup> We included VST as a non-VLM baseline to benchmark the performance of pure vision-based deep learning approaches. We implemented a VST baseline for brain tumor diagnosis by fine-tuning a pretrained video-swin transformer on the BrainTumor48K dataset. For VST, we employed five dedicated modality-specific encoders for each modality required by BrainVLM (i.e., two T1c sequences in different planes, a non-contrast T1, and T2 and T2f sequences). These high-dimensional features extracted from each encoder were then concatenated and processed through a fully connected layer to generate the final 14 category diagnostic output.

## S2.6 Quantitative assessment and statistical analysis

Diagnosis statistical metrics. We evaluated BrainVLM’s tumor prediction performance using sensitivity, specificity, precision, Cohen’s Kappa, F1, and Area Under the Curve (AUC). We used both the common language metrics (Bilingual Evaluation Understudy (BLEU)<sup>78</sup>) and clinical relevance metrics (RaTEScore<sup>79</sup> and RadGraph-XL<sup>80</sup>) to measure the BrainVLM’s report generation performance. Given true-positive (TP), false-positive (FP), true-negative (TN) and false-negative (FN) rates, the precision, sensitivity (recall), specificity, and F1 score of each class are calculated as follows:

$$
P r e c i s i o n = \frac { T P } { T P + F P } ,
$$

$$
S e n s i t i \nu i t y = \frac { T P } { T P + F N } ,
$$

$$
S p e c i f i c i t y = \frac { T N } { T N + F P } ,
$$

$$
F _ { \scriptscriptstyle { I } } \cdot \mathrm { s c o r e } = \frac { 2 \cdot P r e c i s i o n \cdot S e n s i t i \nu i t y } { P r e c i s i o n + S e n s i t i \nu i t y } ,
$$

$$
C o h e n ^ { \prime } s K a p p a = \frac { P _ { o } - P _ { e } } { 1 - P _ { e } } ,
$$

where

$$
P _ { \circ } = \frac { T P + T N } { T P + F P + T N + F N } \mathrm { a n d } P _ { e } = \frac { ( T P + F P ) \times ( T P + F N ) } { \left( T P + F P + T N + F N \right) ^ { 2 } } + \frac { ( T N + F N ) \times ( T N + F P ) } { \left( T P + F P + T N + F N \right) ^ { 2 } } .
$$

Uncertainty metric. To evaluate the reliability of BrainVLM’s uncertainty predictions and quantify the alignment between predicted confidence and diagnostic accuracy, we employed Expected Calibration Error (ECE) and Brier Score.The ECE measures the weighted average difference between accuracy and confidence across bins:M

$$
E C E = \sum _ { m = 1 } ^ { M } \frac { \mid B _ { m } \mid } { N } \mid a c c ( B _ { m } - c o n f ( B _ { m } ) \mid
$$

where is the total number of samples, whileN $a c c ( B _ { m } )$ and $c o n f ( B _ { m } )$ represent the accuracy and average confidence within bin $B _ { { } _ { m } }$ . The Brier score further assesses the mean squared error between the predicted probability distribution and the ground truth:

$$
B S = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } ( p _ { i , k } - y _ { i , k } ) ^ { 2 } ,
$$

where denotes the number of classes, K $p _ { i , k }$ the predicted probability, and $y _ { i , k }$ the binary indicator for class . Lower values in both metrics indicate a model that morek accurately reflects its own diagnostic certainty.

Report metric. BLEU-4 is a widely used metric for evaluating lexical similarity between machine-generated and human reference texts.<sup>78</sup> Unlike traditional natural language metrics, both RadGraph-XL and RaTEScore prioritize domain-specific content critical to radiology, making them particularly suitable for evaluating automated report generation systems. RaTEScore evaluates clinical reports by identifying medical entities across imaging modalities and regions and then calculates a clinical relevanceweighted similarity score. Similarly, RadGraph-XL is a metric designed to evaluate the quality of radiology reports by assessing their semantic and clinical accuracy. It constructs a graph-based representation of entities and relationships within a report, capturing anatomical findings, observations, and their interconnections, such as whether a finding is present, absent, or uncertain. By comparing these graphs between generated and reference reports, RadGraph-XL quantifies similarity using an F1 score, thereby providing a robust measure of how well a generated report conveys clinically relevant information.

## S2.7 Implementation details

We employed Adam $\mathrm { W } ^ { 8 1 }$ with weight decay regularization $( \lambda = 0 . 0 5 )$ across all training stages, coupled with a cosine decay scheduler and 300-step linear warmup. During the training stage 1 (2D MRI-text representation learning), the learning rate was linearly scaled from $1 \times 1 0 ^ { - 6 } \mathrm { t o } \ 1 \times 1 0 ^ { - 4 }$ , and the batch size was set to 256. For training stage 2 (hybrid 2D-3D training) and stage 3 (3D volumetric learning), the learning rate followed the same warmup initialization $( 1 \times 1 0 ^ { - 6 } )$ . However, the peak learning rate was set to $3 \times 1 0 ^ { - 5 }$ for training stage 2 and $1 \times 1 0 ^ { - 5 }$ for training stage 3, respectively. For the molecular classification finetuning, the peak learning rate was set to $1 \times 1 0 ^ { - 4 }$ . We set the batch size to 1 on each GPU device for both training stages 2 and 3, resulting in a total effective batch size of 4. All experiments and the implementation of the BrainVLM were conducted using Python (version 3.9). We also utilized torch (2.0.0), transformers (4.45.2), huggingface-hub (0.29.2), tokenizers (0.20.3) and peft (0.2.0) for model training. For image processing and data augmentation, scikit-image (0.24.0), SimpleITK (2.4.1), monai (1.4.0), numpy (1.26.4), opencv-python (4.7.0) and nibabel (5.3.2) were applied. For training stage 1, we used 8 NVIDIA 4090 GPUs for multi-GPU training. For the training stages 2 and 3, as well as the reliable training, we used 4 NVIDIA L40 GPUs.

## S3 More analytical experiments on BrainVLM

## S3.1 Top-2 prediction result based on uncertainty score

To mirror clinical workflows in clinically ambiguous cases, where neuroradiologists consider multiple diagnostic possibilities, we implemented a confidence-triggered supplementary diagnosis mechanism in BrainVLM (Supplementary Section S2.3.1, appendix p 20). When the confidence score for the primary diagnosis (Top 1) falls below a threshold (set as 75% in this study), the BrainVLM automatically presents the second-most probable diagnosis (Top 2) for radiologist review. On the primary dataset, this mechanism increased the Top‑2 F1 score from 0.83 to 0.86 (Macro‑AUC: 0.85 → 0.88), while performance on the external dataset improved from 0.76 to 0.78 (Macro‑AUC: 0.80 → 0.81) (Supplementary Figures 9a and 10a). More specifically, BrainVLM surpassed neuroradiologists’ Top 2 recommendations across 11 out of 12 tumor types in the primary dataset and 10 out of 12 tumor types in the external dataset, achieving comparable results in CPT but underperforming marginally in MNM (Supplementary Figure 9d).

## S3.2 Robustness of BrainVLM to missing MRI sequences

BrainVLM performs inference using four standard MRI sequences (T1, T1c, T2, and T2-Flair). In clinical practice, incomplete acquisition of MRI sequences (e.g., missing an entire sequence) frequently occurs due to insufficient clinician awareness or contraindications. To evaluate the model’s robustness to such scenarios, we conducted an ablation study by systematically excluding one of the four standard sequences. For each excluded sequence, we maintained the required five-image input structure by randomly selecting an additional view from the remaining sequences. For example, when removing T1, we substituted an extra view from T1c, T2, or T2-Flair (e.g., supplementary sagittal T1c). For fair comparison, our analysis included only patients meeting the five-image requirement after sequence removal. This resulted in 2,701 patients in the primary test dataset and 870 patients in the external test dataset. As shown in Supplementary Table 10 and Supplementary Figure 7b, BrainVLM achieves diagnostic performance comparable to the full-sequence baseline in scenarios missing T2, T2-Flair, or T1 sequences, with less than a 5% overall reduction in F1 score. However, the unavailability of T1c sequences caused a 13.4% F1-score decline, underscoring its critical role in brain tumor assessment.

Furthermore, we employed Shapley value analysis<sup>82</sup> to evaluate the contribution of each input sequence (T1, T1c, T2, and T2-Flair) to the model’s decisions. As shown in Supplementary Figure 7a, the analysis revealed consistent patterns with our ablation results: omission of the T1c sequence led to the most substantial decline in diagnostic performance across all tumor types, highlighting its critical importance. Additionally, EMB classification exhibited uniform sensitivity to the removal of any single sequence (approximately 35% mean performance drop), while GCN and MEN predictions remained stable when T2 or T2-Flair was missing. In contrast, CPT predictions were more dependent on T1, T1c and T2-Flair, but showed resilience to the exclusion of T2. These findings align well with clinical diagnostic experience, where specific sequences are known to be more informative for certain tumor types.

## S3.3 Subcategories classification performance in GGN

GGN collectively represent a vital and diverse category of brain tumors with significant heterogeneity in therapeutic approaches and clinical prognoses. These tumors could be stratified into three subcategories according to the WHO CNS5 classification, including gliomas, glioneuronal tumors, and ependymal tumors. To extend the clinical utility of BrainVLM, we further evaluated the diagnostic performance of the model among these three subtypes, benchmarking against both neuroradiologists and state-of-the-art deep learning models (RadFM, Merlin and VST) In this evaluation, the models were tasked with classifying brain tumors into 14 distinct categories, encompassing the 11 major WHO CNS5 tumor types and the three GGN subcategories. BrainVLM achieved F1 scores of 0.78 for gliomas, 0.43 for glioneuronal tumors, and 0.30 for ependymal tumors (Supplementary Figure 9b) throughout the entire test cohort, demonstrating substantial improvements over both neuroradiologists consensus scores (0.75, 0.07, 0.24, respectively) and baseline models (RadFM: (0.60, 0.05, 0.09); Merlin: (0.53, 0.07, 0.14) and VST: (0.52, 0.06, 0.07)). These results underscore BrainVLM’s superior discriminative capability in fine-grained tumor categorization.

## S3.4 Validation of generated report descriptions

In the main text, we quantitatively evaluated BrainVLM’s report generation performance using standard n-gram metrics (e.g., BLEU-4) and entity-level clinical metrics (RaTEScore, RadGraph-XL). These rigorous automated benchmarks effectively measure lexical similarity and the preservation of key clinical entities. To further augment this evaluation and assess the generated reports from a semantic reasoning perspective akin to human experts, we introduced an “LLM-as-a-judge” framework to specifically evaluate two clinically critical aspects: (1) the accuracy of signal characteristics across multi-parametric MRI sequences (T1, T1c, T2, T2-Flair), and (2) the precision of tumor localization descriptions. Following previous work,<sup>83</sup> we categorized both intensity and location descriptions into three levels: correct, partially correct, and incorrect, assigning scores of 1.0, 0.5, and 0.0, respectively. For intensity descriptions, partially correct refers to cases where, for example, a prediction of “hyperintense” for an actual “hyperintense-isointense” lesion. For location descriptions, partially correct refers to cases where a prediction of the “Fourth Ventricle” for a lesion located in the “Cerebellum”. To facilitate robust and unbiased matching and judgment for this detailed analysis, we leveraged the strong reasoning capabilities of the opensource large language model, Qwen-2.5-72B.<sup>84</sup> As shown in Supplementary Figure 7d, BrainVLM consistently outperformed baseline models across all evaluated dimensions based on the LLM-judge assessments. Specifically, in characterizing Contrast Enhancement (CE), a critical feature for tumor grading, BrainVLM achieved a high correctness rate of 0.80 and an average score of 0.86, surpassing both Merlin (Correct = 0.72, Avg. Score=0.78) and RadFM (Correct = 0.70, Avg. Score = 0.76). Similarly, for intrinsic signal intensity patterns on T1, T2, and T2-Flair sequences, BrainVLM maintained superior accuracy (Correct rates: T1 = 0.71, T2 = 0.72, T2-Flair = 0.74) with consistently higher average scores compared to Merlin and RadFM. Furthermore, for the challenging task of tumor localization (lesion), BrainVLM achieved a correct identification rate of 0.60 with an average score of 0.72, significantly outperforming the best baseline model Merlin (Correct = 0.51, Avg. Score = 0.62) and RadFM (Correct = 0.47, Avg. Score = 0.57). These results confirm that BrainVLM generates radiology reports with higher clinical fidelity and significantly lower hallucination rates than current state-of-the-art comparators.

## S3.5 Expert-based clinical assessment of generated reports

While the LLM-as-a-judge framework (Section S3.4) and standard automated metrics (e.g., BLEU-4, RaTEScore, RadGraph-XL) provide rigorous quantitative evaluation of generated reports, they cannot fully capture the holistic clinical judgment of how faithfully a generated report reflects what an experienced neuroradiologist would consider an accurate description of a patient's imaging findings. To address this gap and assess the generated reports from a clinical-judgment perspective, we conducted an additional expert-based assessment to directly evaluate the clinical fidelity of

BrainVLM-generated reports relative to existing comparators (Merlin, RadFM, and GPT-4o).

Specifically, we randomly selected 100 patient cases from the primary test cohort, covering a broad range of WHO CNS5 tumor types. For each case, four reports were generated by BrainVLM, Merlin, RadFM, and GPT-4o. The reports were anonymized and randomly shuffled to remove any identifying information about the source model, after which three expert neuroradiologists independently scored each report on a 0–5 Likert scale based on consistency with the ground-truth clinical report. The scoring rubric defined 0 as completely inconsistent (no clinically relevant overlap) and 5 as fully consistent (semantically equivalent to the ground-truth report), with intermediate levels graded in 1-point increments. The three raters were blinded to which model produced each report to minimize evaluator bias.

As shown in Supplementary Figure 8, BrainVLM consistently received the highest score from all three raters, with mean scores of 4.77, 4.73, and 4.83 (Raters 1–3, respectively). Averaged across the three raters, BrainVLM achieved an overall mean score of 4.78, substantially outperforming RadFM (3.92), Merlin (3.78), and GPT-4o (3.70). These results confirm that, beyond favorable performance on automated and LLM-as-a-judge metrics, the reports generated by BrainVLM are also judged by independent expert clinicians to be more consistent with ground-truth radiology reports, supporting the clinical fidelity of BrainVLM's report generation capability beyond what can be inferred from automated metrics alone.

## S3.6 Diagnosis performance in intraaxial and extraaxial tumor

To address potential overestimation of diagnostic accuracy arising from lesion location cues (e.g., specific regions like the pineal or sellar areas), we performed a stratified analysis by categorizing the dataset into intra-axial<sup>85</sup> and extra-axial<sup>86</sup> tumors based on their primary anatomical origin. According to established radiological definitions, intraaxial tumors (originating within the brain parenchyma) in this study included Gliomas, glioneuronal tumors, and neuronal tumors (GGN), Choroid plexus tumors (CPT), Embryonal tumors (EMB), Pineal region tumors (PIN), and Hematolymphoid tumors (HEM), while extra-axial tumors (originating outside the brain parenchyma) comprised Meningioma (MEN), Germ cell tumors (GCT), Tumors of the sellar region (TSR), Cranial and paraspinal nerve tumors (CPN), Mesenchymal non-meningothelial tumors (MNM), Melanocytic tumors (MEL), and Brain Metastases (MET). This classification ensured that the model's performance was evaluated independently across distinct anatomical compartments as defined in clinical practice.

The stratified performance metrics, detailed in Supplementary Figure 10b and d, demonstrate that BrainVLM maintains robust diagnostic reliability across both anatomical compartments in the primary and external retrospective dataset. For intraaxial tumors, the macro-AUC values were 0.87 (Top-1) and 0.89 (Top-2) in the primary cohort, and 0.76 (Top-1) and 0.77 (Top-2) in the external validation cohort; for extraaxial tumors, the model achieved macro-AUCs of 0.87 (Top-1) and 0.90 (Top-2) in the primary test cohort, and 0.84 (Top-1) and 0.86 (Top-2) in the external test cohort. These results closely mirror the overall analysis, confirming that BrainVLM’s performance is consistent across different anatomical compartments and is not solely reliant on location-based cues for its diagnostic predictions.

## S3.7 False negative finding investigation

We have extracted and analyzed the FNR for both BrainVLM and board-certified neuroradiologists across all 12 tumor subcategories. As demonstrated in Supplementary Figure 11, BrainVLM consistently exhibits FNRs that are largely comparable to, and in many instances, demonstrably lower than, those of board-certified neuroradiologists across both the primary and external retrospective datasets. A notable strength of BrainVLM is its ability to achieve substantially lower FNRs for several critical tumor types. This includes Embryonal tumors (EMB) (0.09 vs. 0.35 in the primary dataset; 0.06 vs. 0.44 in the external dataset), Pineal region tumors (PIN) (0.16 vs. 0.84 in the primary dataset; 0.67 vs. 0.83 in the external dataset), and Gliomas, glioneuronal tumors, and neuronal tumors (GGN) in the external dataset (0.16 vs. 0.36). These instances highlight areas where BrainVLM offers a clear and impactful diagnostic advantage.

Furthermore, for a substantial number of other tumor types, including Brain Metastases (MET), Germ cell tumors (GCT), Cranial and paraspinal nerve tumor (CPN), Hematolymphoid tumors (HEM), and Tumors of the sellar region (TSR), BrainVLM’s FNRs were either closely similar to or slightly lower than those of neuroradiologists across both datasets, demonstrating strong performance parity. For example, BrainVLM achieved FNRs of 0.27 vs. 0.36 (primary) and 0.38 vs. 0.41 (external) for MET, 0.40 vs. 0.42 (primary) and 0.39 vs. 0.39 (external) for GCT, 0.20 vs. 0.22 (external) for CPN, and 0.08 vs. 0.08 (primary) and 0.18 vs. 0.19 (external) for TSR.

We also acknowledge instances where neuroradiologists showed lower FNRs, offering valuable targets for future model refinement. This includes mesenchymal, nonmeningothelial tumors in the external dataset (MNM, 0.54 vs. 0.38) and meningiomas in the external dataset (MEN, 0.20 vs. 0.14). It is also important to note that for several inherently challenging and rare categories, such as CPT, HEM, and MEL in the external dataset, both BrainVLM and neuroradiologists exhibited relatively high FNRs (e.g., CPT: 0.83 for both; HEM: 0.70 for both; MEL: 0.80 for both). In conclusion, these comprehensive FNR comparisons, leveraging detailed data from both primary and external retrospective datasets, further substantiate BrainVLM’s robust clinical utility.

## S3.8 Uncertainty quantification of BrainVLM

To rigorously evaluate BrainVLM’s uncertainty awareness and clinical reliability, we have now rigorously assessed BrainVLM's calibration performance using standard metrics (Expected Calibration Error (ECE) and the Brier score) on the primary and external retrospective datasets. Here, ECE is a metric quantifying how closely uncertainty scores (predicted probabilities) match true outcomes, and Brier score measures the mean squared difference between the uncertainty score and the actual outcome. We achieved an ECE of 0.0360 and a Brier score of 0.1448. This remarkably low ECE indicates that the model’s uncertainty scores are well aligned with true outcomes, maintaining an average calibration gap of 3.6%. The Brier score of 0.1448 further reflects the model’s overall predictive accuracy, indicating a high degree of closeness between the uncertainty scores and the actual clinical outcomes. We further assessed its behavior through two complementary dimensions: 1) Stratified Confidence distribution analysis comparing high-performing versus low-performing diagnostic categories to validate calibration robustness, and 2) Clinical ambiguity verification on the bottom 10% low-confidence cases via a multi-reader consensus study to confirm that low confidence reflects genuine diagnostic difficulty.

## S3.8.1 Confidence distributions: top 3 vs. bottom 5 classes

To further validate the robustness of BrainVLM’s calibration, we performed a comparative analysis of confidence distributions between the highest-performing (Top 3) and lowest-performing (Bottom 5) diagnostic categories. Specifically, we have now performed a more granular analysis of our Reliable BrainVLM’s confidence distributions, focusing specifically on the Bottom 5 performing classes (based on F1- score) and the Top 3 performing classes for comparison (Supplementary Figure 12). Specifically, we generated detailed confidence distribution plots for the Bottom 5 classes (MEL [N = 25], PIN [N = 25], CPT [N = 47], MNM [N = 193], and HEM [N = 93]) as well as the Top 3 classes (TSR [N = 531], MEN [N = 1954], and GGN [N = 1584]) for comparison; see Supplementary Figure 12.

We observe that in the Top 3 classes, the vast majority of correct predictions (74% for TSR, 78% for MEN, and 70% for GGN) fall within the “Highest Confidence” bracket (0.9–1.0) range, while most incorrect predictions are associated with lower confidence scores. This is expected, as these classes are relatively easier for BrainVLM to classify. In contrast, for the Bottom 5 classes, the confidence scores for correct predictions are generally more dispersed across the confidence spectrum, reflecting the higher difficulty, pathological complexity, and rarity of these cases. Nevertheless, incorrect predictions in these challenging classes still predominantly correspond to low or medium confidence levels (75% for MEL, 80% for PIN, 71% for CPT, 81% for MNM, and 89% for HEM), further supporting the reliability of BrainVLM’s confidence scores in indicating prediction uncertainty.

We note that for MEL (Melanocytic tumors), 100% of correct answers fall into the high-confidence bracket. This is reasonable given the very small sample size (N = 25); the few correctly classified cases happen to be those for which the model was most confident. This limited data warrant caution in interpreting the confidence distribution, and a larger sample would be needed for more robust conclusions. Additionally, for MNM, we observe that correct predictions show a higher proportion of medium or low confidence scores. Clinically, this is reasonable, as the MNM category encompasses a highly heterogeneous group under WHO CNS5, including lymphomas, sarcomas, and vascular tumors, with diverse radiological phenotypes. This heterogeneity presents a significant challenge for the model.

In summary, across both easy and challenging classes, our Reliable BrainVLM demonstrates the ability to concentrate correct predictions in high-confidence regions while suppressing confidence for errors. This is an important characteristic for safe clinical deployment.

## S3.8.2 Multi-readers assessment on bottom 10% confidence cases

We performed an independent multi-reader study on the bottom 10% of cases (N = 385) where BrainVLM exhibited the lowest confidence. These cases were blindly reviewed by two junior radiologists (J1 and J2) and two expert neuroradiologists (E1 and E2). Supplementary table 12 illustrates the diagnostic landscape of these low-confidence cases, reflecting their inherent clinical complexity. We first compared the diagnostic accuracy of BrainVLM against human readers on this bottom 10% subset. BrainVLM achieved an accuracy of 65.0%, significantly outperforming the junior radiologists (0.51 and 0.44) and closely approaching the performance of experts (0.72 and 0.66) on the same subset. Notably, these expert scores represent a marked decline from the 0.80 average accuracy typically observed in our broader multi-reader study (N = 248). This universal drop in performance across both AI and experts confirms that BrainVLM’s uncertainty scores successfully isolate cases that are inherently more difficult to diagnose.

To evaluate whether the subset identified by BrainVLM was objectively ambiguous, we conducted a multi-level consistency analysis involving two junior radiologists (J1, J2) and two senior experts (E1, E2). The results revealed a Cohen’s Kappa of 0.33 between the two junior radiologists (J1 vs. J2), indicating only minimal agreement. While professional experience improved the consensus, the consistency between the two senior experts (E1 vs. E2) reached a Cohen’s Kappa of 0.60. Furthermore, we compared the top-performing junior radiologist with the topperforming expert, resulting in a Kappa of 0.44, which highlights a significant diagnostic gap even between the most proficient readers across different seniority levels. Although this represents moderate-to-substantial agreement, it remains significantly below the high consensus threshold $( \mathrm { K a p p a } > 0 . 8 0 ) ^ { 8 7 }$ typically expected in routine neuro-oncological diagnosis. This diagnostic gap—where even seasoned experts exhibit a notable margin of disagreement—confirms that the bottom 10% subset identified by our model consists of inherently challenging and borderline cases.

## S3.9 Subgroup analysis across age, sex, MRI vendor and ethnicity

To rigorously evaluate the fairness and generalizability of BrainVLM, we conducted a comprehensive subgroup analysis across age, sex, and MRI vendor using both our primary and external retrospective cohorts (Supplementary Figure 13a). Regarding ethnicity, we acknowledge that our test dataset predominantly represents Asian populations due to our collaborating institutions being based in China. To directly address this limitation and assess generalizability, we augmented our evaluation with three public external datasets from European and North American healthcare systems. It is important to note that these newly acquired datasets were not part of the training process and served as truly external test sets. Detailed results for age, sex, MRI vendor, and ethnicity are provided below.

## S3.9.1 Analysis of demographic variables: age and sex

Age-related Performance. BrainVLM maintains a stable weighted F1-score (0.81) across most age groups. A relative performance dip was observed in the 0–20 years subgroup $( \mathrm { F } 1 = 0 . 7 1 , \mathrm { N } = 4 7 9 )$ . Importantly, this trend aligns with human reader performance, as radiologists also exhibited their lowest F1 scores in this pediatric cohort $( \mathrm { F } 1 = 0 . 6 7$ compared with $\mathrm { F } 1 = 0 . 7 9$ in the overall retrospective test dataset). We observed that the performance decline in this specific subgroup is primarily driven by lower F1 scores in GGN, MEN and TSR. This variance is largely attributable to the intrinsic epidemiological rarity and atypical radiological presentations of these tumors in pediatric and adolescent patients. According to the CBTRUS Statistical Report,<sup>88</sup> the incidence of meningiomas and high-grade gliomas increases exponentially with age, making them exceptionally rare in the 0-20 population. Furthermore, tumors in this age group, such as craniopharyngiomas in the sellar region—often exhibit distinct morphological features and biological behaviors compared to their adult counterparts.<sup>89</sup> These results indicate that the performance variance is driven by inherent clinical complexity rather than algorithmic bias.

Sex-related Performance. We observed that the diagnostic F1-score for males (F1 = 0.80, N = 2,350) was slightly lower than for females $( \mathrm { F } 1 = 0 . 8 4 , \mathrm { N } = 2 , 8 6 1 )$ . This finding is highly consistent with the diagnostic patterns of radiologists (male $\mathrm { F } 1 = 0 . 7 5$ vs. female $\operatorname { F 1 } = 0 . 8 3 )$ . The consistency between BrainVLM and expert readers underscores the model’s ability to mirror clinical reality across sex strata.

## S3.9.2 Analysis of MRI vendor

We identified 1,881 patients (Primary: 1,451; External: 430) with complete metadata for the vendor-specific robustness analysis, as vendor identifiers were unavailable for all other cases (Supplementary Figure 13a). BrainVLM demonstrated robust performance across major MRI vendors, including GE $( \mathrm { F } 1 = 0 . 8 1 , \mathrm { N } = 8 4 7 )$ , Siemens (F1 = 0.80, N = 774), and Toshiba $( \mathrm { F } 1 = 0 . 8 7 , \mathrm { N } = 1 1 3 )$ . While smaller cohorts (e.g., UIH, N = 24, AIITECH, N = 44) showed more variance due to limited sample sizes, the overall performance remained consistently high (F1 = 0.75 and 0.76), proving that BrainVLM is generally resilient to variations in hardware signatures and acquisition protocols.

## S3.9.3 Analysis of ethnicity

To assess performance across diverse ethnic populations, we evaluated BrainVLM on three public datasets from Europe and North America EGD,<sup>90</sup> Brain-Mets-Lung<sup>91</sup> and Vestibular-Schwannoma-MC2<sup>92</sup> as our external test datasets. The detailed results were presented in Supplementary Figure 14.

Erasmus Glioma Database (EGD)<sup>90</sup>: This dataset comprises 774 patients from a large-scale European glioma cohort, all with the full suite of MRI modalities (T1, T1c, T2, T2-Flair). Applying BrainVLM to these full MRI modalities yielded an F1-score of 0.91, precision 0.93, and sensitivity 0.91.

Brain-Mets-Lung (Yale University)<sup>91</sup>: This cohort consists of 100 patients with brain metastases curated from North American healthcare systems. Nevertheless, this cohort presents a significant challenge as only T1 and T1c sequences are available. Despite the restricted input, BrainVLM achieved an F1-score of 0.66.

Vestibular-Schwannoma-MC2 (London)<sup>92</sup>: This is a European multicenter dataset of 190 schwannoma patients. This database includes T1, T1c, and T2 sequences, but lacks T2-Flair datasets. On this cohort, BrainVLM achieved an F1-score of 0.83

Overall, BrainVLM demonstrated stable diagnostic performance across these diverse geographical and ethnic populations. While the F1-scores in the metastases and schwannoma cohorts were slightly lower compared to our primary test set (0.65 vs. 0.66 on metastases, and F1 0.83 vs. 0.88 on schwannoma, Supplementary Figure 13b), this marginal decline is largely attributable to the missing MRI modalities in these datasets. These results underscore BrainVLM’s capacity to maintain diagnostic reliability even when transitioned to previously unseen healthcare environments and ethnic groups.

## S3.9.4 Conclusion of subgroup analysis

Our comprehensive analysis demonstrates that BrainVLM maintains high diagnostic stability across diverse demographic and technical variables. The model’s performance fluctuations across age and sex groups closely mirror those of experienced radiologists, suggesting that these variations stem from inherent clinical complexity rather than algorithmic bias. Furthermore, its consistent performance across different MRI vendors and its robust validation on western cohorts confirms the model’s generalizability across diverse ethnicity populations. These results collectively validate BrainVLM as a fair and resilient tool for heterogeneous clinical environments.

## S3.10 Clinician-AI interaction in incorrect AI scenarios

In the multi-reader study, we specifically examined whether incorrect AI suggestions exert a negative impact on diagnostic outcomes. We assessed the direction of diagnostic changes within AI augmented workflows and evaluated whether erroneous AI outputs lead to automation bias or adversely affect clinician decision-making.

First, to examine the possibility of automation or anchoring bias in diagnostically challenging cases, we conducted a targeted diagnostic shift analysis focusing strictly on the subset of cases where clinicians reported their lowest 25% confidence scores (extracting based on the confidence score of the unaided cases). We then assessed the direction of diagnostic change when readers assessed cases with AI assistance. Specifically, clinician–AI interactions were categorized into six trajectories: (A) the clinician remained correct despite an incorrect AI suggestion; (B) the clinician successfully corrected an initially incorrect diagnosis; (C) the clinician changed from a correct diagnosis to an incorrect one after an incorrect AI suggestion; (D) both the initial clinician diagnosis and AI were incorrect, and the final diagnosis remained incorrect; (E) the clinician failed to adopt a correct AI suggestion, remaining incorrect; and (F) both the initial clinician diagnosis and AI suggestion were correct, with the final diagnosis remaining correct. Across all reader seniority groups, Trajectory F was the most common outcome overall. Notably, successful error correction (Trajectory B) was highly prominent and increased significantly with reader seniority, whereas cases of being erroneously misled by an incorrect AI (Trajectory C) were consistently the least common across all levels (Figure 4f). This demonstrates that rather than acting as a negative anchor, the AI actively and successfully helped neuroradiologists recover from their own diagnostic uncertainties.

Second, to explicitly address scenarios in which the AI was incorrect, we performed an additional analysis within the expanded 248-case multi-reader study, restricted to cases in which BrainVLM generated an incorrect diagnosis. In this subset (57 cases), we compared readers’ unassisted baseline accuracy with their final accuracy after reviewing the incorrect AI output (Figure 4g). Overall accuracy remained stable (50.0% without AI vs 50.9% with AI). Junior neuroradiologists showed a modest decrease in accuracy (\~6%), whereas senior and expert neuroradiologists maintained or slightly improved their performance on these cases (seniors: 49.0% to 52.9%; experts: 57.6% to 60.6%). A possible explanation is that many incorrect AI predictions were accompanied by low reliability scores, which may have prompted readers to apply greater caution when reviewing these outputs. These findings suggest that, within the setting of this reader study, exposure to incorrect AI outputs did not lead to systematic over-reliance overall.

## S3.11 Case studies

## S3.11.1 Case studies in high-confidence error prediction cases

While BrainVLM demonstrates high calibration (ECE = 0.0360), a small fraction of cases may yield high-confidence but incorrect predictions. We present two such examples (Supplementary Figure 15) involving Brain metastases tumor (MET) and Mesenchymal, non-meningothelial tumor (MNM), to provide insights into how such challenging cases should be addressed and managed within a clinical workflow.

Case 1: An MET was misclassified as a Glioma (GGN) with 80% confidence. The high confidence likely resulted from the larger tumor volume, heterogeneous enhancement and prominent peritumoral edema, a feature more commonly associated with high-grade gliomas, which confused the model.

Case 2: An MNM (pathological confirmed with Solitary fibrous tumor) was misclassified as a cranial and paraspinal nerve tumor (CPN) with 90% confidence. The high confidence likely resulted from the tumor location usually occurring CPN. Cases misclassified by the model with high confidence also present substantial diagnostic challenges to radiologists.

Nevertheless, these mistaken classifications did not alter the subsequent clinical management, given that all cases were eligible for surgical resection. To mitigate the risk of over-reliance on erroneous high-confidence predictions, we envision a deployment strategy where confidence scores are used to stratify risk: low-confidence cases trigger human experts for secondary review, while high-confidence predictions are accompanied by interpretability images to facilitate clinician verification rather than blind acceptance. BrainVLM is an assistive diagnostic tool and the clinician retains final responsibility, especially when AI confidence conflicts with clinical intuition.

## S3.11.2 Case studies in multi-readers’ time reduction

We have included 2 representative cases from our multi-reader study that demonstrate a substantial reduction in diagnostic time with BrainVLM integration (Supplementary Figure 16).

Case 1: A 62-year-old female patient with histopathologically confirmed brain metastasis. Six MRI sequences were available (axial T1, T1-contrast, T2, T2-Flair; coronal and sagittal T1-contrast) for this patient. In our study, this case was first interpreted in the expert-only setting and then, after a one-month washout period, reinterpreted by the same expert with AI support. In the expert-only setting, the expert neuroradiologist misidentified the lesion as a glioma with confidence 60%. The diagnostic process took 132 seconds. Our BrainVLM outputs “Brain metastase tumor” (MET) as the most probable category with a confidence score of 75%. Guided by this AI-generated suggestion, the expert outputs the correct diagnosis in only 40 seconds, demonstrating a 70% reduction in diagnostic time (132s to 40s).

Case 2: An 8-year-old female patient with histopathologically confirmed Glioma. Six MRI sequences were available (axial T1, T1-contrast, T2, T2-Flair; coronal and sagittal T1-contrast) for this patient. In our study, this case was first interpreted in the AI-augmented setting and then, after a one-month washout period, re-interpreted by the same expert without AI support. In the expert-only setting, the neuroradiologist correctly identified glioma but with relatively low confidence (60%), requiring 80 seconds to analyze this case. In the AI-augmented setting, BrainVLM outputs “glioma” as the most probable category with a high confidence score of 90%. Using this AI output as decision support, the expert finalized the same diagnosis in only 23 seconds. This case illustrates a 71% reduction in diagnostic time (80s to 23s). This synergistic effect significantly accelerated the decision-making process, demonstrating BrainVLM’s role in reinforcing clinical certainty for complex cases.

## S4 Regulatory considerations, failure modes, and medico-legal implications

Regulatory considerations. This study adhered to strict data governance protocols to protect the privacy of patients from the real world. All clinical and imaging data were fully anonymized and encrypted during storage and transmission. Access controls were implemented to ensure that only approved study personnel could interact with the dataset, and data usage was contractually and ethically bound to the specific aims of this research. These measures align with the principles of patient data protection required by regulatory frameworks such as Health Insurance Portability and Accountability Act (HIPAA) or General Data Protection Regulation (GDPR), ensuring the integrity and confidentiality of the data used for model development and validation.

Failure modes. Several scenarios where model performance degrades or fails in a clinically significant manner and solutions. (a) Low-confidence predictions: Cases where BrainVLM’s prediction confidence falls below a predefined threshold (e.g., < 85%) are flagged for secondary review by human experts. (b) High-confidence discrepancies: BrainVLM’s high-confidence prediction conflicts with the clinician’s observation or the patient’s clinical presentation, requiring further verification. (c) Scope limitation (non-tumoral cases): BrainVLM is specifically designed for tumor type classification tasks (i.e., differentiation among known tumor types), rather than initial screening to differentiate between tumoral and non-tumoral cases. Therefore, its application requires prior identification of brain tumor cases, ideally through a dedicated screening model.

Medico-legal implications. BrainVLM may serve as an assistive diagnostic tool, not a substitute for the neuroradiologist or clinician. The final diagnostic responsibility must always remain with the clinician.

## S5 Data and code availability

Data availability. The online training data are available on their corresponding websites. We will publish the scripts for pre-processing the online dataset upon publication of this manuscript. The institutional training data that support the findings of this study are available from the corresponding author upon request. The detailed publicly available resources can be accessed at: PubMed Central (https://pmc.ncbi.nlm.nih.gov/), Ctisus (https://www.ctisus.com/), ImageCLEFmedical23 (https://www.imageclef.org/2023/medical/caption), Br35H (https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection), Figshare Brain Tumor Dataset (https://www.kaggle.com/datasets/ashkhagan/figshare-braintumor-dataset), Brain Tumor Classification (https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri), Radiopaedia (https://radiopaedia.org/home), Brain Tumor MRI8 images 44 classes (https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c), LGG-1p19qDeletion (https://www.cancerimagingarchive.net/collection/lgg-1p19qdeletion/), BraTS23 (https://synapse.org/brats), ReMIND (https://www.cancerimagingarchive.net/collection/remind/), RHUH-GBM (https://www.cancerimagingarchive.net/collection/rhuh-gbm/), UPENN-GBM (https://www.cancerimagingarchive.net/collection/upenn-gbm/), GLIS-RT (https://www.cancerimagingarchive.net/collection/glis-rt/), QIN GBM Treatment Response (https://www.cancerimagingarchive.net/collection/qin-gbm-treatmentresponse/), LUMIERE (https://github.com/ysuter/gbm-data-longitudinal), OpenNeuro Diffuse Gliomas (https://openneuro.org/datasets/ds004717/versions/1.0.0), Brain-Tumor- Progression (https://www.cancerimagingarchive.net/collection/brain-tumorprogression/), RIDER (https://www.cancerimagingarchive.net/collection/rider-lungct/), ACRIN-DSC- MR-Brain (https://www.cancerimagingarchive.net/collection/acrindsc-mr-brain/), ACRINFMISO-Brain (https://www.cancerimagingarchive.net/collection/acrin-fmiso-brain/), Meningioma-SEG-CLASS (https://www.cancerimagingarchive.net/collection/meningioma-segclass/), Brain metastase MRI dataset (https://www.nature.com/articles/s41597-023- 02123-0), Brain-TR- GammaKnife, AOMIC- ID1000 (https://openneuro.org/datasets/ds003097), AOMIC- PIOP1 (https://openneuro.org/datasets/ds002785), IXI (https://brain-development.org/ixidataset/), OpenNeuro Listening Task (https://openneuro.org/datasets/ds004285/versions/1.0.0), QTAB dataset (https://openneuro.org/datasets/ds004146/versions/1.0.4), OpenNeuro Visual Audiovisual Speech (https://openneuro.org/datasets/ds003717/versions/1.0.1), OpenNeuro Dynamic Passive Threat (https://openneuro.org/datasets/), OpenNeuro UCLA Consortium (https://openneuro.org/datasets/ds000030/versions/00016), OpenNeuro NIMH Healthy (https://openneuro.org/datasets/ds004215), DLBS dataset (https://openneuro.org/datasets/ds004856/versions/1.0.0), OpenNeuro Healthy Adults (https://openneuro.org/datasets/ds002330/versions/1.1.0/1.1.0), EGD Dataset (https://www.healthinformationportal.eu/health-information-sources/erasmus-gliomadatabase), Brain-Mets-Lung Dataset (https://www.cancerimagingarchive.net/collection/brain-mets-lung-mri-path-segs/) and Vestibular-schwannoma-MC2 (https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-mc-rc2/). Note that some online resources (e.g., Radiopaedia) cannot be directly accessed for AI research purposes. We have obtained official authorization for non-commercial use in this study.

Code availability. BrainVLM will be fully available at https://github.com/HKU-HealthAI/BrainVLM upon acceptance of the paper. We will release all model weights and relevant source code for pretraining, fine-tuning, and inference to facilitate research transparency and community collaboration.

## References

1 Baid U, Ghodasara S, Mohan S, et al. The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification. 2021; published online Sept 12. DOI:10.48550/arXiv.2107.02314.

2 Adewole M, Rudie JD, Gbadamosi A, et al. The Brain Tumor Segmentation (BraTS) Challenge 2023: Glioma Segmentation in Sub-Saharan Africa Patient Population (BraTS-Africa). 2023; published online May 30. DOI:10.48550/arXiv.2305.19369.

3 Kazerooni AF, Khalili N, Liu X, et al. The Brain Tumor Segmentation (BraTS) Challenge 2023: Focus on Pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs). 2024; published online May 23. DOI:10.48550/arXiv.2305.17033.

4 Moawad AW, Janas A, Baid U, et al. The Brain Tumor Segmentation (BraTS-METS) Challenge 2023: Brain Metastasis Segmentation on Pre-treatment MRI. 2024; published online Dec 9. DOI:10.48550/arXiv.2306.00838.

5 Verdier MC de, Saluja R, Gagnon L, et al. The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma Segmentation on Post-treatment MRI. 2024; published online May 28. DOI:10.48550/arXiv.2405.18368.

6 LaBella D, Abramova V, Astaraki M, et al. Analysis of the 2024 BraTS Meningioma Radiotherapy Planning Automated Segmentation Challenge. 2025; published online July 21. DOI:10.48550/arXiv.2405.18383.

7 Cepeda S, García García S, Arrese I, et al. The Río Hortega University Hospital Glioblastoma dataset: a comprehensive collection of preoperative, early postoperative and recurrence MRI scans (RHUH-GBM). 2023. DOI:10.7937/4545-C905.

8 Pedano N, Flanders AE, Scarpace L, et al. The Cancer Genome Atlas Low Grade Glioma Collection (TCGA-LGG). 2016. DOI:10.7937/K9/TCIA.2016.L4LTD3TK.

9 Scarpace L, Mikkelsen T, Cha S, et al. The Cancer Genome Atlas Glioblastoma Multiforme Collection (TCGA-GBM). 2016. DOI:10.7937/K9/TCIA.2016.RNYFUYE9.

10 Barboriak D. Data From RIDER\_NEURO\_MRI. 2015. DOI:10.7937/K9/TCIA.2015.VOSN3HN1.

11 Vassantachart A, Cao Y, Shen Z, et al. Segmentation and Classification of Grade I and II Meningiomas from Magnetic Resonance Imaging: An Open Annotated Dataset (Meningioma-SEG-CLASS). 2023. DOI:10.7937/0TKV-1A36.

12 Wang Y, Duggar W, Caballero D, et al. Brain Tumor Recurrence Prediction after Gamma Knife Radiotherapy from MRI and Related DICOM-RT: An Open Annotated Dataset and Baseline Algorithm (Brain-TR-GammaKnife). 2023. DOI:10.7937/XB6D-PY67.

13 Clark K, Vendt B, Smith K, et al. The Cancer Imaging Archive (TCIA): maintaining and operating a public information repository. J Digit Imaging 2013; 26: 1045–57.

14 Schmainda K, Prah M. Data from Brain-Tumor-Progression. 2019. DOI:10.7937/K9/TCIA.2018.15QUZVNB.

15 Kinahan P, Muzi M, Bialecki B, Herman B, Coombs L. Data from ACRIN-DSC-MR-Brain. 2019. DOI:10.7937/TCIA.2019.ZR1PJF4I.

16 Kinahan P, Muzi M, Bialecki B, Coombs L. Data from ACRIN-FMISO-Brain. 2018. DOI:10.7937/K9/TCIA.2018.VOHLEKOK.

17 Erickson B, Akkus Z, Sedlar J, Korfiatis P. Data from LGG-1p19qDeletion. 2017. DOI:10.7937/K9/TCIA.2017.DWEHTZ9V.

18 Juvekar P, Dorent R, Kögl F, et al. The Brain Resection Multimodal Imaging Database (ReMIND). 2023. DOI:10.7937/3RAG-D070.

19 Shusharina N, Bortfeld T. Glioma Image Segmentation for Radiotherapy: RT targets, barriers to cancer spread, and organs at risk (GLIS-RT). 2021. DOI:10.7937/TCIA.T905-ZQ20.

20 Bakas S, Sako C, Akbari H, et al. Multi-parametric magnetic resonance imaging (mpMRI) scans for de novo Glioblastoma (GBM) patients from the University of Pennsylvania Health System (UPENN-GBM). 2021. DOI:10.7937/TCIA.709X-DN49.

21 Hamada A. Br35H: brain Tumor Detection 2020. Kaggle Dataset. 2020. https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection.

22 Cheng J. brain tumor dataset. 2024; : 879510755 Bytes.

23 Feltrin F. Brain Tumor MRI Images 44 Classes. Kaggle Dataset. 2023. https://www.kaggle.com/datas ets/fernando2rad/brain-tumor-mri-images-44c.

24 Bhuvaji S, Kadam A, Bhumkar P, Dedge S, Kanchan S. Brain Tumor Classification (MRI). Kaggle Dataset. 2020. https://www.kaggle.com/dsv/1183165.

25 Johns Hopkins University School of Medicine. CTisus. https://www.ctisus.com/.

26 Snoek L, Miesen MVD, Leij AVD, Beemsterboer T, Eigenhuis A, Scholte S. AOMIC-ID1000. 2021. DOI:10.18112/OPENNEURO.DS003097.V1.2.1.

27 Snoek L, Miesen MVD, Leij AVD, Beemsterboer T, Eigenhuis A, Scholte S. AOMIC-PIOP1. 2020. DOI:10.18112/OPENNEURO.DS002785.V2.0.0.

28 Bilder R, Poldrack R, Cannon T, et al. UCLA Consortium for Neuropsychiatric Phenomics LA5c Study. 2020. DOI:10.18112/OPENNEURO.DS000030.V1.0.0.

29Nugent AC, Thomas AG, Mahoney M, et al. The NIMH Healthy Research Volunteer Dataset.

30 Park D, Hennessee J, Smith ET, et al. The Dallas Lifespan Brain Study. 2025. DOI:10.18112/OPENNEURO.DS004856.V1.3.0.

31 Strike LT, Hansell NK, Miller JL, et al. Queensland Twin Adolescent Brain (QTAB). 2022. DOI:10.18112/OPENNEURO.DS004146.V1.0.4.

32 Sunavsky A, Poppenk J. Neuroimaging predictors of creativity in healthy adults. 2020. DOI:10.18112/OPENNEURO.DS002330.V1.1.0.

33 Rogers CS, Jones MS, McConkey S, Peelle JE. Listening task. 2022. DOI:10.18112/OPENNEURO.DS004285.V1.0.0.

34 Peelle JE, Spehar B, MS J, et al. Visual and audiovisual speech perception associated with increased functional connectivity between sensory and motor regions. 2023. DOI:10.18112/OPENNEURO.DS003717.V1.1.0.

35 Suter Y, Knecht U, Valenzuela W, et al. The LUMIERE dataset: Longitudinal Glioblastoma MRI with expert RANO evaluation. Sci Data 2022; 9: 768.

36 Qiu Z, Xie Z, Lin H, et al. Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities. Nat Commun 2024; 15: 7637.

37 Filimonova E, Pashkov A, Borisov N, Kalinovsky A, Rzaev J. Utilizing the amide proton transfer technique to characterize diffuse gliomas based on the WHO 2021 classification of CNS tumors. Neuroradiol J 2024; 37: 490–9.

38 Zolotova SV, Golanov AV, Pronin IN, et al. Burdenko’s Glioblastoma Progression Dataset (Burdenko-GBM-Progression). 2023. DOI:10.7937/E1QP-D183.

39 Ocaña-Tienda B, Pérez-Beteta J, Villanueva-García JD, et al. A comprehensive dataset of annotated brain metastasis MR images with clinical and radiomic data. Sci Data 2023; 10: 208.

40 Santiago Silva. IXI Sample Dataset. 2022; published online April 20. DOI:10.17632/7KD5WJ7V7P.2.

41 Radiopaedia. Radiopaedia. https://radiopaedia.org.

42 Roberts RJ. PubMed Central: The GenBank of the published literature. Proc Natl Acad Sci U S A 2001; 98: 381–2.

43 Fan L, Li H, Zhuo J, et al. The Human Brainnetome Atlas: A New Brain Atlas Based on Connectional Architecture. Cereb Cortex 2016; 26: 3508–26.

44 Ionescu B, Müller H, Drăgulinescu A-M, et al. Overview of the ImageCLEF 2023: Multimedia Retrieval in Medical, Social Media and Internet Applications. In: Arampatzis A,

Kanoulas E, Tsikrika T, et al., eds. Experimental IR Meets Multilinguality, Multimodality, and Interaction. Cham: Springer Nature Switzerland, 2023: 370–96.

45 Ren S, He K, Girshick R, Sun J. Faster R-CNN: towards real-time object detection with region proposal networks. In: Proceedings of the 29th International Conference on Neural Information Processing Systems - Volume 1. Cambridge, MA, USA: MIT Press, 2015: 91– 9.

46 Subramanian S, Wang LL, Mehta S, et al. MedICaT: A Dataset of Medical Images, Captions, and Textual References. 2020; published online Oct 12. DOI:10.48550/arXiv.2010.06000.

47 Smith R. An Overview of the Tesseract OCR Engine. In: Ninth International Conference on Document Analysis and Recognition (ICDAR 2007) Vol 2. Curitiba, Parana, Brazil: IEEE, 2007: 629–33.

48 OpenAI. GPT-4. https://openai.com/research/gpt-4.

49 DeepSeek-AI, Guo D, Yang D, et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. 2025; published online Jan 22. DOI:10.48550/arXiv.2501.12948.

50 Team G, Kamath A, Ferret J, et al. Gemma 3 Technical Report. 2025; published online March 25. DOI:10.48550/arXiv.2503.19786.

51 Yuan Y, Zheng Y, Qu L. Benchmarking Radiology Report Generation From Noisy Free-Texts. IEEE J Biomed Health Inform 2025; 29: 7549–58.

52 Essien M, Cooper ME, Gore A, et al. Interrater Agreement of BT-RADS for Evaluation of Follow-up MRI in Patients with Treated Primary Brain Tumor. .

53 Orman M, Sirolu S, Seker ME, et al. Multicenter analysis for inter reader agreement of PIQUAL score v2 for basic readers in prostate MRI. Sci Rep 2025; 15: 22865.

54 Ionescu B, Müller H, Péteri R, et al. Overview of the ImageCLEF 2021: Multimedia Retrieval in Medical, Nature, Internet and Social Media Applications. In: Candan KS, Ionescu B, Goeuriot L, et al., eds. Experimental IR Meets Multilinguality, Multimodality, and Interaction. Cham: Springer International Publishing, 2021: 345–70.

55 Wang Y-RJ, Wang P, Yan Z, et al. Advancing presurgical non-invasive molecular subgroup prediction in medulloblastoma using artificial intelligence and MRI signatures. Cancer Cell 2024; 42: 1239-1257.e7.

56 Havaei M, Davy A, Warde-Farley D, et al. Brain tumor segmentation with Deep Neural Networks. Med Image Anal 2017; 35: 18–31.

57 Tong S, Brown E, Wu P, et al. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. In: Globerson A, Mackey L, Belgrave D, et al., eds. Advances in Neural

Information Processing Systems. Curran Associates, Inc., 2024: 87310–56.

58 Zhang S, Xu Y, Usuyama N, et al. BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. 2025; published online Jan 8. DOI:10.48550/arXiv.2303.00915.

59 Radford A, Kim JW, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision. In: Meila M, Zhang T, eds. Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021: 8748–63.

60 Guo Z, Xu R, Yao Y, et al. LLaVA-UHD: An LMM Perceiving Any Aspect Ratio and High-Resolution Images. In: Leonardis A, Ricci E, Roth S, Russakovsky O, Sattler T, Varol G, eds. Computer Vision – ECCV 2024. Cham: Springer Nature Switzerland, 2025: 390–406.

61 Grattafiori A, Dubey A, Jauhri A, et al. The Llama 3 Herd of Models. 2024; published online Nov 23. DOI:10.48550/arXiv.2407.21783.

62 Wang W-Y, Wang Z, Suzuki H, Kobayashi Y. Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs. 2025; published online March 13. DOI:10.48550/arXiv.2503.02597.

63 Zhou C, Yu L, Babu A, et al. Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model. 2024; published online Aug 20. DOI:10.48550/arXiv.2408.11039.

64 Xie J, Mao W, Bai Z, et al. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation. 2025; published online Sept 8. DOI:10.48550/arXiv.2408.12528.

65 Lowekamp BC, Chen DT, Ibáñez L, Blezek D. The Design of SimpleITK. Front Neuroinform 2013; 7: 45.

66 Brinjikji W, Diehn FE, Jarvik JG, et al. MRI Findings of Disc Degeneration are More Prevalent in Adults with Low Back Pain than in Asymptomatic Controls: A Systematic Review and Meta-Analysis. AJNR Am J Neuroradiol 2015; 36: 2394–9.

67 Lambin P, Leijenaar RTH, Deist TM, et al. Radiomics: the bridge between medical imaging and personalized medicine. Nat Rev Clin Oncol 2017; 14: 749–62.

68 Hu EJ, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021; published online Oct 16. DOI:10.48550/arXiv.2106.09685.

69 Despotović I, Goossens B, Philips W. MRI Segmentation of the Human Brain: Challenges, Methods, and Applications. Computational and Mathematical Methods in Medicine 2015; 2015: 1–23.

70 Li C, Wong C, Zhang S, et al. LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. 2023; published online June 1.

71 Kim Y, Jeong H, Chen S, et al. Medical Hallucinations in Foundation Models and Their Impact on Healthcare. 2025; published online Nov 2. DOI:10.48550/arXiv.2503.05777.

72 Amini A, Vieira T, Ash E, Cotterell R. Variational Best-of-N Alignment. 2025; published online March 4. DOI:10.48550/arXiv.2407.06057.

73 Wan G, Wu Y, Chen J, Li S. Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). Albuquerque, New Mexico: Association for Computational Linguistics, 2025: 3613–35.

74 Wu C, Zhang X, Zhang Y, Hui H, Wang Y, Xie W. Towards generalist foundation model for radiology by leveraging web-scale 2D&3D medical data. Nat Commun 2025; 16: 7866.

75 Blankemeier L, Kumar A, Cohen JP, et al. Merlin: a computed tomography vision–language foundation model and dataset. Nature 2026; published online March 4. DOI:10.1038/s41586-026-10181-8.

76 Liu Z, Ning J, Cao Y, et al. Video Swin Transformer. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). New Orleans, LA, USA: IEEE, 2022: 3192–201.

77 Wang Y-R, Yang K, Wen Y, et al. Screening and diagnosis of cardiovascular disease using artificial intelligence-enabled cardiac magnetic resonance imaging. Nat Med 2024; 30: 1471–80.

78 Papineni K, Roukos S, Ward T, Zhu W-J. BLEU: a method for automatic evaluation of machine translation. In: Proceedings of the 40th Annual Meeting on Association for Computational Linguistics - ACL ’02. Philadelphia, Pennsylvania: Association for Computational Linguistics, 2001: 311.

79 Zhao W, Wu C, Zhang X, Zhang Y, Wang Y, Xie W. RaTEScore: A Metric for Radiology Report Generation. 2024; published online Oct 23. DOI:10.48550/arXiv.2406.16845.

80 Delbrouck J-B, Chambon P, Chen Z, et al. RadGraph-XL: A Large-Scale Expert-Annotated Dataset for Entity and Relation Extraction from Radiology Reports. In: Findings of the Association for Computational Linguistics ACL 2024. Bangkok, Thailand and virtual meeting: Association for Computational Linguistics, 2024: 12902–15.

81 Loshchilov I, Hutter F. Decoupled Weight Decay Regularization. 2019; published online Jan 4. DOI:10.48550/arXiv.1711.05101.

82 Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems. 2017: 4765–74.

83 Zhang K, Zhou R, Adhikarla E, et al. A generalist vision-language foundation model for diverse biomedical tasks. Nat Med 2024; 30: 3129–41.

84 Qwen, Yang A, Yang B, et al. Qwen2.5 Technical Report. 2025; published online Jan 3. DOI:10.48550/arXiv.2412.15115.

85 Gaillard F, Tatco V, Di Muzio B. Intra-axial. In: Radiopaedia.org. Radiopaedia.org, 2009. DOI:10.53347/rID-7963.

86 Gaillard F, González Herrera G, Rasuli B. Extra-axial. In: Radiopaedia.org. Radiopaedia.org, 2009. DOI:10.53347/rID-7961.

87 McHugh ML. Interrater reliability: the kappa statistic. Biochem Med (Zagreb) 2012; 22: 276–82.

88 Ostrom QT, Price M, Neff C, et al. CBTRUS Statistical Report: Primary Brain and Other Central Nervous System Tumors Diagnosed in the United States in 2016-2020. Neuro Oncol 2023; 25: iv1–99.

89 Walz PC, Drapeau A, Shaikhouni A, et al. Pediatric pituitary adenomas. Childs Nerv Syst 2019; 35: 2107–18.

90 van der Voort SR, Incekara F, Wijnenga MMJ, et al. The Erasmus Glioma Database (EGD): Structural MRI scans, WHO 2016 subtypes, and segmentations of 774 patients with glioma. Data Brief2021; 37: 107191.

91 Chadha S, Sritharan DV, Dolezal D, et al. Matched MRI, Segmentations, and Histopathologic Images of Brain Metastases from Primary Lung Cancer. Sci Data 2026; 13: 40.

92 Wijethilake N, Ivory M, MacCormac O, et al. Deep Learning Consensus-based Annotation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Annotated Multi-Center Routine Clinical Dataset (Vestibular-Schwannoma-MC-RC 2). 2025. DOI:10.7937/BQ0Z-XA62.

## Supplementary Tables

Supplementary Table 1 | Characteristics of the retrospective primary dataset (Xiangya Hospital) and external test datasets from 11 independent medical centers.
<table><tr><td rowspan="2"></td><td rowspan="2">No. of subjects</td><td colspan="2">Sex</td><td rowspan="2">Age in years (range)</td></tr><tr><td>Male</td><td>Female</td></tr><tr><td>Total</td><td>10,147</td><td>4,921 (48%)</td><td>5,226 (52%)</td><td>53±16 (1-86)</td></tr><tr><td>Train dataset</td><td>4,936</td><td>2,571 (52%)</td><td>2,365 (48%)</td><td>43± 19 (1-86)</td></tr><tr><td>Test dataset</td><td>5,211</td><td>2,350 (45%)</td><td>2,861 (55%)</td><td>48±17 (1-86)</td></tr><tr><td>Xiangya Hospital</td><td>8,813</td><td>4,256 (48%)</td><td>4,557 (52%)</td><td>45±18 (1-86)</td></tr><tr><td>Changde First People&#x27;s Hospital</td><td>361</td><td>193 (53%)</td><td>168 (47%)</td><td>55±13 (6-86)</td></tr><tr><td>Shanghai Tongji Hospital</td><td>104</td><td>54 (52%)</td><td>50 (48%)</td><td>54±16 (5-79)</td></tr><tr><td>University of South China Second Hospital</td><td>110</td><td>57 (52%)</td><td>53 (48%)</td><td>52±15 (6-80)</td></tr><tr><td>Shenzhen Second People&#x27;s Hospital</td><td>52</td><td>30 (58%)</td><td>22 (42%)</td><td>49±15 (8-75)</td></tr><tr><td>The Third Xiangya Hospital</td><td>103</td><td>49 (48%)</td><td>54 (52%)</td><td>52±17 (7-76)</td></tr><tr><td>Jiangxi Provincial People&#x27;s Hospital</td><td>232</td><td>107 (46%)</td><td>125 (54%)</td><td>53±15 (13-82)</td></tr><tr><td>Chongqing Traditional Chinese medicine Hospital</td><td>42</td><td>17 (40%)</td><td>25 (60%)</td><td>58±12 (26-78)</td></tr><tr><td>The First Affiliated Hospital of Nanchang University</td><td>97</td><td>40 (41%)</td><td>57 (59%)</td><td>46±17 (20-67)</td></tr><tr><td>Hunan Provincial Children&#x27;s Hospital</td><td>103</td><td>56 (54%)</td><td>47 (46%)</td><td>28±17 (1-60)</td></tr><tr><td>The First Affiliated Hospital of Lanzhou University</td><td>81</td><td>31 (38%)</td><td>50 (62%)</td><td>52±13 (8-76)</td></tr><tr><td>The Second Affiliated Hospital of Nanchang University</td><td>49</td><td>31 (63%)</td><td>18(37%)</td><td>54±12 (13-74)</td></tr></table>

Supplementary Table 2 | Summary of publicly available brain MRI datasets. The table summarizes a range of datasets, detailing the number of subjects, 2D MRI slices, data modalities (2D/3D), and pathological labels (tumor or healthy). For each dataset, original source information and the specific tasks organized in this study are also provided. Here, “3D Healthy” refers to datasets containing healthy individuals with 3D MRI volumes, while “3D Tumor” refers to datasets containing brain tumor patients with 3D MRI volumes. The PubMed Central dataset includes both normal and brain tumor subjects. For ReMIND dataset, we utilized 108 brain tumor cases, excluding 6 brain disease cases. Please note that, due to policy restrictions (e.g., Radiopaedia), certain portions of the training dataset cannot be shared.
<table><tr><td>Data Type</td><td>Data Resource</td><td>No. of subjects</td><td>No. of images</td><td>Original Information</td><td>Organized Tasks</td></tr><tr><td>2D web crawling</td><td>PubMed Central</td><td>23,849</td><td>116,296</td><td>Description</td><td>2D MRI slice-text description Tumor classification</td></tr><tr><td>2D web crawling</td><td>Ctisus</td><td>150</td><td>1,124</td><td>Description, Diagnosis</td><td>2D MRI slice-text description Tumor classification</td></tr><tr><td>2D web crawling</td><td>ImageCLEFmedical23</td><td>484</td><td>4,631</td><td>Description</td><td>2D MRI slice-text description Tumor classification</td></tr><tr><td>2D Kaggle</td><td>Br35H</td><td></td><td>2,137</td><td>Diagnosis Segment</td><td>2D MRI slice-text description Tumor classification</td></tr><tr><td>2D Kaggle</td><td>Figshare Brain Tumor Dataset</td><td>233</td><td>3,064</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>2D Kaggle</td><td>Brain Tumor Classification</td><td></td><td>3,264</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>2D Kaggle</td><td>Brain Tumor MRI images 44 classes</td><td></td><td>4,479</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>LGG-1p19qDeletion</td><td>159</td><td>32,829</td><td>Diagnosis Segment</td><td>3D MRI-report Tumor classification</td></tr><tr><td>3D Tumor</td><td>BraTS23</td><td>3,263</td><td>2,156,589</td><td>Diagnosis Segment</td><td>3D MRI-report Tumor classification</td></tr><tr><td>3D Tumor</td><td>ReMIND*</td><td>108</td><td>60,963</td><td>Diagnosis metadata</td><td>3D MRI-report Tumor classification</td></tr><tr><td>3D Tumor</td><td>RHUH-GBM</td><td>40</td><td>122,835</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>UPENN-GBM</td><td>630</td><td>622,112</td><td>Diagnosis Segment</td><td>3D MRI-report Tumor classification</td></tr><tr><td>3D Tumor</td><td>GLIS-RT</td><td>230</td><td>83,196</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>QIN GBM Treatment Response</td><td>54</td><td>66,988</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>LUMIERE</td><td>91</td><td>460,155</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>OpenNeuro (Diffuse Gliomas)</td><td>42</td><td>22,412</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>Brain-Tumor- Progression</td><td>20</td><td>46,573</td><td>Diagnosis Segment</td><td>3D MRI-report Tumor classification</td></tr><tr><td>3D Tumor</td><td>Radiopaedia</td><td>2606</td><td>376,374</td><td>Diagnosis Report</td><td>3D MRI-report Tumor classification</td></tr><tr><td>3D Tumor</td><td>Burdenko</td><td>180</td><td>460,524</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>RIDER</td><td>19</td><td>115,100</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>ACRIN-DSC- MR-Brain</td><td>123</td><td>1,385,512</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>ACRIN-FMISO- Brain</td><td>50</td><td>508,500</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Tumor</td><td>Meningioma- SEG-CLASS</td><td>96</td><td>83,556</td><td>Diagnosis metadata</td><td>3D MRI-report Tumor classification</td></tr></table>

Supplementary Table 2 | Continuation of Table 2. Summary of publicly available brain MRI datasets. The table summarizes a range of datasets, detailing the number of subjects, 2D MRI slices, data modalities (2D/3D), and pathological labels (tumor or healthy). For each dataset, original source information and the specific tasks organized in this study are also provided. Here, “3D Healthy” refers to datasets containing healthy individuals with 3D MRI volumes, while “3D Tumor” refers to datasets containing brain tumor patients with 3D MRI volumes. The PubMed Central dataset includes both normal and brain tumor subjects. For ReMIND dataset, we utilized 108 brain tumor cases, excluding 6 brain disease cases.
<table><tr><td>Data Type</td><td>Data Resource</td><td>No. of subjects</td><td>No. of images</td><td>Original Information</td><td>Organized Tasks</td></tr><tr><td>3D Tumor</td><td>Brain metastase MRI dataset</td><td>75</td><td>85,400</td><td>Diagnosis Segment</td><td>3D MRI-report Tumor classification</td></tr><tr><td>3D Tumor</td><td>Brain-TR- GammaKnife</td><td>47</td><td>8,300</td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>3D Healthy</td><td>AOMIC- ID1000</td><td>928</td><td>1, 101, 652</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>AOMIC- PIOP1</td><td>216</td><td>251,380</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>IXI</td><td>582</td><td>398,774</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>OpenNeuro (Listening Task)</td><td>78</td><td>136,694</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>OpenNeuro (Healthy Adults)</td><td>66</td><td>122, 175</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>OpenNeuro (Visual Audiovisual Speech)</td><td>60</td><td>105,160</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>OpenNeuro (Dynamic Passive Threat)</td><td>82</td><td>126,299</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>OpenNeuro (UCLA Consortium)</td><td>272</td><td>285,910</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>OpenNeuro (NIMH Healthy)</td><td>157</td><td>461,281</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>DLBS dataset</td><td>315</td><td>99,344</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>3D Healthy</td><td>QTAB dataset</td><td>422</td><td>1,251,467</td><td>Diagnosis</td><td>Tumor or healthy classification</td></tr><tr><td>Public test Public test</td><td>EGD dataset Brain-Mets-Lung</td><td>774</td><td></td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td></td><td>Vestibular-</td><td>100</td><td></td><td>Diagnosis</td><td>Tumor classification</td></tr><tr><td>Public test</td><td>Schwannoma-MC2</td><td>190</td><td></td><td>Diagnosis</td><td>Tumor classification</td></tr></table>

Supplementary Table 3 | Characteristics of primary prospective dataset, including surgical group and non-operative group for each brain tumor type.
<table><tr><td rowspan="2"></td><td colspan="4">Surgical Group</td><td colspan="4">Non-operative group</td><td rowspan="2">Entire dataset</td></tr><tr><td rowspan="2">No. of subjects</td><td colspan="2">SeX</td><td rowspan="2">Age in years Mean ± SD</td><td rowspan="2">No. of subjects</td><td colspan="2">SeX</td><td rowspan="2">Age in years Mean ± SD (range)</td></tr><tr><td>Male</td><td>Female</td><td>(range)</td><td>Male</td><td>Female</td></tr><tr><td>Total</td><td>639</td><td>290</td><td>359</td><td>48±17 (2-83)</td><td>74</td><td>33</td><td>41</td><td>41±21 (2-66)</td><td>713</td></tr><tr><td>MET</td><td>25</td><td>16</td><td>10</td><td>60±8 (38-72)</td><td>7</td><td>4</td><td>3</td><td>45±11 (34-55)</td><td>32</td></tr><tr><td>GCT</td><td>15</td><td>11</td><td>6</td><td>16±9 (7-39)</td><td>3</td><td>2</td><td>1</td><td>8±1 (7-9)</td><td>18</td></tr><tr><td>GGN</td><td>119</td><td>69</td><td>50</td><td>44±19 (5-83)</td><td>18</td><td>8</td><td>10</td><td>30±24 (2-60)</td><td>137</td></tr><tr><td>MEN</td><td>185</td><td>44</td><td>141</td><td>53±11 (14-78)</td><td>22</td><td>6</td><td>16</td><td>50±11 (29-66)</td><td>207</td></tr><tr><td>TSR</td><td>146</td><td>81</td><td>68</td><td>49±16 (4-77)</td><td>17</td><td>9</td><td>8</td><td>54±8 (37-63)</td><td>163</td></tr><tr><td>MNM</td><td>40</td><td>22</td><td>21</td><td>38±21 (2-72)</td><td>2</td><td>1</td><td>1</td><td>15±1 (14-16)</td><td>42</td></tr><tr><td>CPN</td><td>87</td><td>36</td><td>52</td><td>52±13 (22-76)</td><td>5</td><td>3</td><td>2</td><td>54±15 (42-62)</td><td>92</td></tr><tr><td>CPT</td><td>4</td><td>1</td><td>3</td><td>43±15 (19-60)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4</td></tr><tr><td>HEM</td><td>10</td><td>5</td><td>5</td><td>64±7 (54-74)</td><td>1</td><td>-</td><td>-</td><td>-</td><td>10</td></tr><tr><td>EMB</td><td>7</td><td>5</td><td>2</td><td>17±12 (5-35)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>7</td></tr><tr><td>PIN</td><td>1</td><td>0</td><td>1</td><td>43</td><td>-</td><td>■</td><td>■</td><td>■</td><td>1</td></tr></table>

In this table, we use abbreviations for tumor categories: MET (brain metastases), GCT (germ cell tumors), GGN (gliomas, glioneuronal tumors, and neuronal tumors), MEN (meningioma), TSR (tumors of the sellar region), MNM (mesenchymal, non-meningothelial tumors), CPN (cranial and paraspinal nerve tumors), CPT (choroid plexus tumors), HEM (hematolymphoid tumors), EMB (embryonal tumors), PIN (pineal region tumors), and MEL (melanocytic tumors).

Supplementary Table 4 | Characteristics of external prospective dataset, including surgical group and non-operative group for each brain tumor type.
<table><tr><td rowspan="2"></td><td colspan="4">Surgical Group</td><td colspan="4">Non-operative group</td><td rowspan="2">Entire dataset</td></tr><tr><td rowspan="2">No. of subjects</td><td colspan="2">Sex</td><td rowspan="2">Age in years Mean ± SD</td><td rowspan="2">No. of subjects</td><td colspan="2">Sex</td><td rowspan="2">Age in years Mean ± SD (range)</td></tr><tr><td>Male</td><td>Female</td><td>(range)</td><td>Male</td><td>Female</td></tr><tr><td>Total</td><td>247</td><td>116</td><td>131</td><td>54±15 (5-79)</td><td>49</td><td>22</td><td>27</td><td>55±18 (9-84)</td><td>296</td></tr><tr><td>MET</td><td>12</td><td>10</td><td>2</td><td>58±15 (20-72)</td><td>5</td><td>0</td><td>5</td><td>67±12 (56-84)</td><td>17</td></tr><tr><td>GCT</td><td>2</td><td>1</td><td>1</td><td>18±1 (17-18)</td><td>2</td><td>1</td><td>1</td><td>13±6 (9-17)</td><td>4</td></tr><tr><td>GGN</td><td>54</td><td>29</td><td>25</td><td>51±16 (15-79)</td><td>10</td><td>6</td><td>4</td><td>64±8 (51-75)</td><td>64</td></tr><tr><td>MEN</td><td>81</td><td>31</td><td>50</td><td>57±10 (34-77)</td><td>14</td><td>3</td><td>11</td><td>63±8 (50-75)</td><td>95</td></tr><tr><td>TSR</td><td>66</td><td>31</td><td>35</td><td>52±16 (11-74)</td><td>12</td><td>8</td><td>4</td><td>49±20 (15-72)</td><td>78</td></tr><tr><td>MNM</td><td>6</td><td>3</td><td>3</td><td>34±20 (5-59)</td><td>1</td><td>1</td><td>0</td><td>62±0 (62-62)</td><td>7</td></tr><tr><td>CPN</td><td>22</td><td>8</td><td>14</td><td>55±14 (24-78)</td><td>2</td><td>1</td><td>1</td><td>54±10 (47-61)</td><td>24</td></tr><tr><td>HEM</td><td>4</td><td>3</td><td>1</td><td>68±6 (63-77)</td><td>2</td><td>2</td><td>0</td><td>35±13 (25-44)</td><td>6</td></tr><tr><td>EMB</td><td>-</td><td>–</td><td>-</td><td></td><td>1</td><td>0</td><td>1</td><td>14±0 (14-14)</td><td>1</td></tr></table>

In this table, we use abbreviations for tumor categories: MET (brain metastases), GCT (germ cell tumors), GGN (gliomas, glioneuronal tumors, and neuronal tumors), MEN (meningioma), TSR (tumors of the sellar region), MNM (mesenchymal, non-meningothelial tumors), CPN (cranial and paraspinal nerve tumors), CPT (choroid plexus tumors), HEM (hematolymphoid tumors), EMB (embryonal tumors), PIN (pineal region tumors), and MEL (melanocytic tumors).

Supplementary Table 5. Comprehensive Statistical Comparison using DeLong Test for macro AUCs. I, II are the Macro AUC comparison between BrainVLM and other baseline models in the retrospective primary and external dataset. III illustrates the comparison between BrainVLM and neuroradiologists of varying expertise levels (Junior, Senior, and Expert) within the 248-case multi-reader study.
<table><tr><td>Category</td><td>Comparison Group</td><td>Baseline AUC</td><td>BrainVLM AUC</td><td>ΔAUC</td><td>P-value</td></tr><tr><td colspan="6">I. Model vs. SOTA Baselines (Primary Test Set)</td></tr><tr><td></td><td>vs. Merlin</td><td>0.62</td><td>0.85</td><td>0.23</td><td>&lt; 0.001</td></tr><tr><td></td><td>vs. RadFM</td><td>0.66</td><td>0.85</td><td>0.19</td><td>&lt; 0.001</td></tr><tr><td></td><td>vs. Video Swin Transformer</td><td>0.73</td><td>0.85</td><td>0.12</td><td>&lt; 0.001</td></tr><tr><td colspan="6">II. Model vs. SOTA Baselines (External Test Set)</td></tr><tr><td></td><td>vs. Merlin</td><td>0.64</td><td>0.80</td><td>0.16</td><td>&lt; 0.001</td></tr><tr><td></td><td>vs. RadFM</td><td>0.60</td><td>0.80</td><td>0.20</td><td>&lt; 0.001</td></tr><tr><td></td><td>vs. Video Swin Transformer</td><td>0.62</td><td>0.80</td><td>0.18</td><td>&lt; 0.001</td></tr><tr><td colspan="6">III. Model vs. Human Readers (Primary Test Set)</td></tr><tr><td></td><td>vs. Junior Radiologist</td><td>0.76</td><td>0.88</td><td>0.12</td><td>&lt; 0.001</td></tr><tr><td></td><td>vs. Senior Radiologist</td><td>0.80</td><td>0.88</td><td>0.08</td><td>&lt; 0.001</td></tr><tr><td></td><td>vs. Expert Radiologist</td><td>0.89</td><td>0.88</td><td>-0.01</td><td>0.28 (NS)</td></tr><tr><td></td><td>vs. All Readers (Average)</td><td>0.87</td><td>0.88</td><td>0.01</td><td>0.004</td></tr></table>

Supplementary Table 6 | Abbreviation of 12 major brain tumors in WHO CNS5.
<table><tr><td>Acronym</td><td>Original Tumor Type Name</td></tr><tr><td>GGN</td><td>Gliomas, glioneuronal tumors, and neuronal tumors</td></tr><tr><td>MET</td><td>Brain metastases</td></tr><tr><td>GCT</td><td>Germ cell tumors</td></tr><tr><td>MEN</td><td>Meningioma</td></tr><tr><td>TSR</td><td>Tumor of the sellar region</td></tr><tr><td>MNM</td><td>Mesenchymal, non-meningothelial tumors</td></tr><tr><td>CPN</td><td>Cranial and paraspinal nerve tumors</td></tr><tr><td>CPT</td><td>Choroid plexus tumors</td></tr><tr><td>HEM</td><td>Hematolymphoid tumors</td></tr><tr><td>EMB</td><td>Embryonal tumors</td></tr><tr><td>PIN</td><td>Pineal tumors</td></tr><tr><td>MEL</td><td>Melanocytic tumors</td></tr></table>

Supplementary Table 7 | Overview of additional MRI sequences and patient information utilized in radiological diagnosis in the primary test dataset and external test dataset. Extra MRI sequences refer to imaging techniques beyond standard T1, T1c, T2, and FLAIR. These extra sequences include Diffusion-Weighted Imaging (DWI), Diffusion Tensor Imaging (DTI), Perfusion-Weighted Imaging (PWI), Magnetic Resonance Angiography (MRA), Magnetic Resonance Spectroscopy (MRS), Susceptibility-Weighted Imaging (SWI), Magnetic Resonance Venography (MRV), Blood Oxygen Level Dependent Imaging (BOLD), Computed Tomography (CT), and Apparent Diffusion Coefficient (ADC). Additional patient information includes access to medical history and prior radiology imaging and associated findings, offering further clinical context for diagnostic decision-making.
<table><tr><td>Dataset</td><td>Total Number</td><td>Extra Modality Percentage</td><td>Extra Prior Medical History of Pathological diagnosis</td><td>Overall Extra Information Percentage</td></tr><tr><td>Primary test dataset</td><td>3,797</td><td>39.5%</td><td>19.3%</td><td>58.8%</td></tr><tr><td>External dataset</td><td>1,309</td><td>56.1%</td><td>12.7%</td><td>68.8%</td></tr></table>

Supplementary Table 8 | Summary of MRI scanner manufacturers, models, and scanning parameters used across all medical centers.
<table><tr><td rowspan="2">Scanning Protocol</td><td colspan="4">MRI Scanner Manufacturer</td></tr><tr><td>GE</td><td>Siemens</td><td>Philips</td><td>UIH</td></tr><tr><td>Model</td><td>Discovery MR750w, Signa HDxt, Signa Excite</td><td>Skyra, Prisma, Avanto, Syngo</td><td>Ingenia, Ingenuit</td><td>uMR 660</td></tr><tr><td>Field strength</td><td>1.5 T/ 3.0 T</td><td>1.5 T/ 3.0 T 7°, 8°, 9°, 10°,</td><td>3.0 T</td><td>1.5 T</td></tr><tr><td>Flip angle</td><td>1°, 8°, 12°, 15°, 18°, 19°, 20°, 25°, 30°, 40°, 60°, 70°, 90°, 111°, 112°, 125°, 142°, 155°, 160°</td><td>15°, 20°, 24°, 25°, 27°, 30°, 40°, 50°, 60°, 70°, 80°, 90°, 120°, 122°, 140°, 145°,</td><td>8°, 10°, 15°, 17°, 18°, 20°, 30°, 40°, 45°, 70°, 75°, 80°, 90°, 100°</td><td>10°, 12°, 15°, 20°, 25°, 54°, 60°, 70°, 72°, 90°, 120°, 130°, 140°, 150°</td></tr><tr><td>Pixel Spacing Range (mm²)</td><td>0.17×0.17 - 3.5×3.5</td><td>150°, 160°, 180° 0.23 × 0.23 - 4.7×4.7</td><td>0.21× 0.21 - 1.8×1.8</td><td>0.26×0.26 -</td></tr><tr><td>Slice Thickness</td><td></td><td></td><td></td><td>1.4×1.4</td></tr><tr><td>Range (mm)</td><td>0.6 - 278</td><td>0.47 - 60</td><td>0.75 - 398</td><td>0.5-10</td></tr></table>

Supplementary Table 9 | The designed special tokens for data differentiation and task-driven prompt design to distinguish between different tasks. A 2D MRI slice is enclosed with ‘<image>’ and ‘</image>’ tags, while 3D MRI volumes use ‘<vid>’ and ‘</vid>’ tags. The <img> and <vi> special tokens represent the 2D image features and 3D volume features, respectively. Additionally, a special token ‘<s>’ is inserted between frames to aid in learning spatial dependencies within the 3D volume.
<table><tr><td>Tasks</td><td>Visual Tokens</td><td>Task-specific Instructions Example</td><td>Sample Output</td></tr><tr><td>2D MRI slice level tumor classification</td><td>&lt;image&gt;&lt;img&gt;&lt;/image&gt;</td><td>Is a tumor present in this MRI slice? If so, what type?</td><td>&lt;tumor_i&gt;</td></tr><tr><td>2D MRI slice-text description</td><td>&lt;image&gt;&lt;img&gt;&lt;/image&gt;</td><td>Please make a description for this image.</td><td>A lession in left lobe, showing T1 hyperintense</td></tr><tr><td>3D MRI tumor classification</td><td>&lt;vid1&gt;&lt;v1&gt;&lt;v1&gt;...&lt;/vid1&gt; &lt;vid2&gt;&lt;v2&gt;&lt;v2&gt;...&lt;/vid2&gt; …</td><td>For a 6-year-old male patient, is a tumor present in these MRIs? If so, what type?</td><td>&lt;tumor_i&gt;</td></tr><tr><td>3D MRI-report generation</td><td>&lt;vid1&gt;&lt;v1&gt;&lt;v1&gt;...&lt;/vid1&gt;&lt;s&gt; &lt;vid2&gt;&lt;v2&gt;&lt;v2&gt;...&lt;/vid2&gt;&lt;s&gt;</td><td>For a 6-year-old male patient, please generate a radiology report for him.</td><td>In the ..., hyperintense... midline shift to ...</td></tr><tr><td>3D MRI report generation tumor classification</td><td>&lt;vid1&gt;&lt;v1&gt;&lt;v1&gt;...&lt;/vid1&gt;&lt;s&gt; &lt;vid2&gt;&lt;v2&gt;&lt;v2&gt;...&lt;/vid2&gt;&lt;s&gt; ….</td><td>For a 6-year-old male patient, please generate a radiology report and make a diagnosis. For a 6-year-old male patient,</td><td>In the ..., hyperintense...; &lt;tumor_i&gt; In the ...,</td></tr><tr><td>Uncertainty quantification</td><td>&lt;vid1&gt;&lt;v1&gt;&lt;v1&gt;...&lt;/vid1&gt;&lt;s&gt; &lt;vid2&gt;&lt;v2&gt;&lt;v2&gt;...&lt;/vid2&gt;&lt;s&gt; 1</td><td>please generate a radiology report, make a diagnosis, and generate reliable score for your diagnosis.</td><td>hyperintense... midline shift to ... &lt;tumor_i&gt;, confidence: 80%</td></tr></table>

Supplementary Table 10 |Diagnostic performance comparison between full MRI sequences (T1, T1c, T2, FLAIR) input and excluding one of the four standard sequences, evaluated on 3,571 patients from the primary and external test datasets. Only patients with complete MRI data across all four sequences were included to eliminate potential confounding due to missing sequences. The remaining 1,555 patients in the original primary and external test datasets with missing sequences were excluded from the analysis.
<table><tr><td colspan="5">F1 score</td></tr><tr><td>Full Sequence</td><td>Excluded T1</td><td>Excluded T1c</td><td>Excluded T2</td><td>Excluded FLAIR</td></tr><tr><td>MET</td><td>0.58 0.52</td><td>0.45</td><td>0.54</td><td>0.53</td></tr><tr><td>GCT</td><td>0.64 0.48</td><td>0.47</td><td>0.58</td><td>0.53</td></tr><tr><td>GGN 0.83</td><td>0.80</td><td>0.72</td><td>0.82</td><td>0.82</td></tr><tr><td>MEN 0.92</td><td>0.89</td><td>0.82</td><td>0.89</td><td>0.89</td></tr><tr><td>TSR 0.75</td><td>0.65</td><td>0.52</td><td>0.63</td><td>0.63</td></tr><tr><td>MNM 0.56</td><td>0.43</td><td>0.44</td><td>0.45</td><td>0.48</td></tr><tr><td>CPN 0.81</td><td>0.76</td><td>0.64</td><td>0.79</td><td>0.78</td></tr><tr><td>CPT 0.44</td><td>0.27</td><td>0.27</td><td>0.43</td><td>0.27</td></tr><tr><td>HEM 0.58</td><td>0.43</td><td>0.42</td><td>0.42</td><td>0.45</td></tr><tr><td>EMB 0.67</td><td>0.44</td><td>0.4</td><td>0.42</td><td>0.44</td></tr><tr><td>PIN 0.43</td><td>0.36</td><td>0.22</td><td>0.43</td><td>0.29</td></tr><tr><td>MEL 0.17</td><td>0.17</td><td>0.17</td><td>0.17</td><td>0.17</td></tr><tr><td>Frequency- weighted F1</td><td>0.82 0.78</td><td>0.71</td><td>0.79</td><td>0.79</td></tr></table>

In this table, we use a series of abbreviations to replace orginal categories: MET for brain metastases, GCT for germ cell tumors, GGN for gliomas, glioneuronal tumors, and neuronal tumors, MEN for meningioma, TSR for tumors of the sellar region, MNM for mesenchymal, non-meningothelial tumors, CPN for cranial and paraspinal nerve tumors, CPT for choroid plexus tumors, HEM for hematolymphoid tumors, EMB for embryonal tumors, PIN for pineal region tumors and MEL for melanocytic tumors.

Supplementary Table 11 | Comparison of F1 scores between our BrainVLM model, neuroradiologists, and three stateof-the-art AI models (RadFM, Merlin, and ChatGPT-4o) and traditional deeplearning model(video-swin-transformer (VST)). We evaluated 3,877 patients from the primary test dataset and 1,334 patients from the external test dataset, respectively.
<table><tr><td colspan="6">F1 score (Primary Testing)</td><td rowspan="2"></td><td colspan="6">F1 score (External Testing)</td></tr><tr><td></td><td>Ours</td><td>Doctor</td><td>RadFM</td><td>Merlin</td><td>GPT  $\mathbf { - 4 0 }$ </td><td>VST</td><td>Ours Doctor</td><td>RadFM</td><td>Merlin</td><td>GPT- 40</td><td>VST</td></tr><tr><td>MET</td><td>0.53</td><td>0.56</td><td>0.45</td><td>0.25</td><td>0.04</td><td>0.50</td><td>0.61</td><td>0.52</td><td>0.44</td><td>0.40</td><td>0.10</td><td>0.22</td></tr><tr><td>GCT</td><td>0.62</td><td>0.56</td><td>0.33</td><td>0.15</td><td>0.08</td><td>0.48</td><td>0.73</td><td>0.52</td><td>0.30</td><td>0.10</td><td>0.11</td><td>0.30</td></tr><tr><td>GGN</td><td>0.82</td><td>0.81</td><td>0.68</td><td>0.59</td><td>0.47</td><td>0.65</td><td>0.79</td><td>0.69</td><td>0.45</td><td>0.56</td><td>0.38</td><td>0.49</td></tr><tr><td>MEN</td><td>0.91</td><td>0.91</td><td>0.64</td><td>0.73</td><td>0.61</td><td>0.59</td><td>0.82</td><td>0.83</td><td>0.53</td><td>0.63</td><td>0.41</td><td>0.4</td></tr><tr><td>TSR</td><td>0.88</td><td>0.88</td><td>0.75</td><td>0.49</td><td>0.56</td><td>0.80</td><td>0.87</td><td>0.84</td><td>0.39</td><td>0.55</td><td>0.48</td><td>0.78</td></tr><tr><td>MNM</td><td>0.56</td><td>0.43</td><td>0.30</td><td>0.12</td><td>0.03</td><td>0.23</td><td>0.46</td><td>0.55</td><td>0.27</td><td>0.0</td><td>0.0</td><td>0.36</td></tr><tr><td>CPN</td><td>0.78</td><td>0.74</td><td>0.40</td><td>0.27</td><td>0.17</td><td>0.45</td><td>0.8</td><td>0.74</td><td>0.28</td><td>0.30</td><td>0.11</td><td>0.36</td></tr><tr><td>CPT</td><td>0.55</td><td>0.47</td><td>0.19</td><td>0.36</td><td>0.08</td><td>0.3</td><td>0.29</td><td>0.18</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.20</td></tr><tr><td>HEM</td><td>0.64</td><td>0.45</td><td>0.31</td><td>0.15</td><td>0.0</td><td>0.12</td><td>0.41</td><td>0.39</td><td>0.33</td><td>0.28</td><td>0.0</td><td>0.1</td></tr><tr><td>EMB</td><td>0.68</td><td>0.58</td><td>0.47</td><td>0.29</td><td>0.13</td><td>0.43</td><td>0.48</td><td>0.42</td><td>0.26</td><td>0.25</td><td>0.0</td><td>0.14</td></tr><tr><td>PIN</td><td>0.42</td><td>0.13</td><td>0.27</td><td>0.17</td><td>0.0</td><td>0.25</td><td>0.36</td><td>0.17</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MEL</td><td>0.29</td><td>0.25</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>-</td><td>-</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Weighted F1</td><td>0.82</td><td>0.81</td><td>0.62</td><td>0.54</td><td>0.37</td><td>0.59</td><td>0.76</td><td>0.71</td><td>0.43</td><td>0.52</td><td>0.30</td><td>0.42</td></tr></table>

In this table, we use a series of abbreviations to replace original categories: MET for brain metastases, GCT for germ cell tumors, GGN for gliomas, glioneuronal tumors, and neuronal tumors, MEN for meningioma, TSR for tumors of the sellar region, MNM for mesenchymal, non-meningothelial tumors, CPN for cranial and paraspinal nerve tumors, CPT for choroid plexus tumors, HEM for hematolymphoid tumors, EMB for embryonal tumors, PIN for pineal region tumors and MEL for melanocytic tumors.

Supplementary Table 12 | Multi-reader consensus performance on the bottom 10% lowest confidence cases identified by BrainVLM (N=385). This table compares the diagnostic accuracy of BrainVLM against two junior (J1, J2) and two expert (E1, E2) neuroradiologists on this challenging subset.
<table><tr><td rowspan="2">Tumor Type</td><td rowspan="2">Patient Number</td><td rowspan="2">Model Accuracy</td><td colspan="2">Junior Radiologists</td><td colspan="2">Expert Radiologists</td></tr><tr><td>J1 Acc.</td><td>J2 Acc.</td><td>E1 Acc.</td><td>E2 Acc.</td></tr><tr><td>MET</td><td>44</td><td>0.55</td><td>0.27</td><td>0.18</td><td>0.36</td><td>0.55</td></tr><tr><td>GCT</td><td>20</td><td>0.60</td><td>0.40</td><td>0.40</td><td>0.60</td><td>0.70</td></tr><tr><td>GGN</td><td>72</td><td>0.70</td><td>0.80</td><td>0.44</td><td>0.83</td><td>0.67</td></tr><tr><td>MEN</td><td>88</td><td>0.74</td><td>0.70</td><td>0.63</td><td>0.90</td><td>0.92</td></tr><tr><td>TSR</td><td>48</td><td>0.83</td><td>0.67</td><td>0.90</td><td>0.90</td><td>0.83</td></tr><tr><td>MNM</td><td>32</td><td>0.38</td><td>0.33</td><td>0.13</td><td>0.63</td><td>0.40</td></tr><tr><td>CPN</td><td>56</td><td>0.57</td><td>0.20</td><td>0.29</td><td>0.69</td><td>0.57</td></tr><tr><td>CPT</td><td>3</td><td>0.33</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.33</td></tr><tr><td>HEM</td><td>3</td><td>0.33</td><td>0.33</td><td>0.0</td><td>0.33</td><td>0.0</td></tr><tr><td>EMB</td><td>16</td><td>0.63</td><td>0.25</td><td>0.13</td><td>0.5</td><td>0.25</td></tr><tr><td>PIN</td><td>4</td><td>0.50</td><td>0.0</td><td>0.0</td><td>0.25</td><td>0.0</td></tr><tr><td>MEL</td><td>1</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Total</td><td>385</td><td>0.65</td><td>0.51</td><td>0.44</td><td>0.72</td><td>0.66</td></tr></table>

In this table, we use a series of abbreviations to replace original categories: MET for brain metastases, GCT for germ cell tumors, GGN for gliomas, glioneuronal tumors, and neuronal tumors, MEN for meningioma, TSR for tumors of the sellar region, MNM for mesenchymal, non-meningothelial tumors, CPN for cranial and paraspinal nerve tumors, CPT for choroid plexus tumors, HEM for hematolymphoid tumors, EMB for embryonal tumors, PIN for pineal region tumors and MEL for melanocytic tumors.

## Supplementary Figures

![](images/a1c365c6fcb4f043548efa304e1f8fd208c53c9dca54f3f2aedde213dfa23484.jpg)  
Supplementary Figure 1 | Preprocessing pipeline for BrainTumor48K. a. The preprocessing pipeline for structured repositories and precurated datasets. a1: For pre-curated datasets containing patient metadata and tumor masks, multi-sequence MRIs and their corresponding segmentation masks were co-registered to a standard neuroanatomical atlas to localize tumors. All available metadata—including imaging modality, sex, diagnostic labels, and extracted tumor descriptions—were utilized to generate comprehensive medical reports using ChatGPT-4o, prompted with RAG techniques. a2: For datasets comprising multi-parametric MRI scans and accompanying radiology reports (Radiopaedia), irrelevant information was removed from the original reports using prompt-based filtering with ChatGPT-4o. b. Preprocessing workflow for unstructured web-based resources, such as PubMed. Using PubMed Central as an example, we illustrate the preprocessing strategy for extracting MRI slice–subcaption pairs. First, brain tumor MRI figures were retrieved through keyword search, and irrelevant images were filtered using a ResNet-101 classifier. Second, compound images were decomposed into individual slices with a Faster R-CNN detector. Third, subcaptions were associated with image slices using Google OCR and ChatGPT-4o. The numbers of MRI slices and resulting slice–subcaption pairs are also summarized below. c. The preprocessing pipeline for institutional medical data. Chinese radiology reports were trimmed to retain modality-specific information, reformatted to the standard template, translated into English, and quality-checked by clinicians and multiple large-language models.

![](images/fec8f5ca467126775a7a14366c7b5467dc60baa83e02f85819d0bc99cb62bd04.jpg)  
Supplementary Figure 2 | Overview of the BrainTumor48K dataset curation pipeline and model training utilization. The dataset integrates diverse 2D and 3D sources into three categories tailored for specific training stages: (1) Classification & Caption Data (Blue, left): 29,834 subjects from public 2D image-caption and classification datasets. This large-scale, heterogeneous data is primarily utilized in Stage 1 (2D slice-text representation learning) and Stage 2 (Hybrid 2D-3D training). (2) Metadata-Driven Reports (Green, middle): 4,351 subjects from 3D public datasets featuring AI-structured reports. Providing detailed spatial and signal grounding, this category is utilized in Stage 2 and Stage 3 (3D volumetric instruction tuning). (3) Professional Radiology Reports (Orange, right): 13,762 subjects with clinician-drafted reports from private and high-quality public sources. This strictly audited, high-fidelity data is also used in Stage 2 and Stage 3. Dataset Abbreviations: PM (PubMed Central), CT (Ctisus), IC (ImageCLEFmedical23), BH (Br35H), FB (Figshare Brain Tumor Dataset), BC (Brain Tumor Classification), BM (Brain Tumor MRI images 44 classes), LG (LGG-1p19qDeletion), BT (BraTS23), RM (ReMIND), RH (RHUH-GBM), UP (UPENN-GBM), GL (GLIS-RT), QG (QIN GBM Treatment Response), LU (LUMIERE), DG (OpenNeuro Diffuse Gliomas), BP (Brain-Tumor-Progression), BU (Burdenko), RI (RIDER), AD (ACRIN-DSC-MR-Brain), AF (ACRIN-FMISO-Brain), MS (Meningioma-SEG-CLASS), BD (Brain metastase MRI dataset), BG (Brain-TR-GammaKnife), AI (AOMIC-ID1000), AP (AOMIC-PIOP1), IX (IXI), LT (OpenNeuro Listening Task), HA (OpenNeuro Healthy Adults), VA (OpenNeuro Visual Audiovisual Speech), DP (OpenNeuro Dynamic Passive Threat), UC (OpenNeuro UCLA Consortium), NH (OpenNeuro NIMH Healthy), DL (DLBS dataset), QT (QTAB dataset), EG (EGD dataset), BL (Brain-Mets-Lung), VS (Vestibular-Schwannoma-MC2), XY (Xiangya Hospital), CF (Changde First People’s Hospital), ST (Shanghai Tongji Hospital), SC (University of South China Second Hospital), SZ (Shenzhen Second People’s Hospital), TX (The Third Xiangya Hospital), JP (Jiangxi Provincial People’s Hospital), CC (Chongqing Traditional Chinese Medicine Hospital), FA (The First Affiliated Hospital of Nanchang University), HP (Hunan Provincial Children’s Hospital), FH (The First Affiliated Hospital of Lanzhou University), SA (The Second Affiliated Hospital of Nanchang University).

a. Original MRI image  
![](images/807c6f0ddbeead49e584aaafb0497df8e22b284321f84062b61ed8983740d5af.jpg)  
Supplementary Figure 3 | Comparison of unprocessed and skull-stripped MRI images. Representative examples of MRI scans before and after skull-stripping. While conventional skull-stripping may aid certain tumor analyses, it is not universally appropriate. Crucially, lesions located in regions like the pia mater or invading the cranial vault can be inadvertently removed during skull stripping, leading to the loss of critical anatomical and pathological information – a risk our approach deliberately avoids.

![](images/fe45309b819e530641518861fb9cc1fd383073d2ba92516f88232d54cc97b1f9.jpg)  
Supplementary Figure 4 | Data flow diagram of BrainVLM. This diagram provides a step-by-step visualization of how BrainVLM processes multi-modal input data and generates structured outputs. The model processes multimodal inputs: multi-parametric MRI scans (processed by a vision encoder and projected into visual tokens) and textual data (patient metadata and instruction prompts, tokenized via BPE). These visual and textual token sequences are concatenated and fed into the LLM decoder (fine-tuned with LoRA). During autoregressive generation, three special start tokens (<Report>, <Diagnosis>, <Uncertainty>) guide the sequential output. First, standard BPE tokens are generated for the radiology report and decoded into free text. Next, a specialized classification token (e.g., <class\_5>) is produced and mapped to a specific tumor grade or subtype (e.g., MEN) via a lookup table. Finally, numerical BPE tokens (e.g., “0”, “.”, “9”) are generated and decoded into a confidence score (e.g., 0.9), which was discretized into six levels during training.

a. Three-stage progressive training strategy of BrainVLM  
![](images/3d6e515a5611c9ec55a828900e5ab76a68c44656500a8686900251cd479a802c.jpg)  
Supplementary Figure 5 | Three-stage progressive training strategy and uncertainty quantification implementation of BrainVLM, including the confidence-triggered Top 2 supplementary diagnosis inference pipeline. a. Three-stage progressive training strategy of BrainVLM. BrainVLM was developed using a threestage progressive training approach: (1) 2D MRI slice-text representation learning using all curated 2D MRI slicetext pairs; (2) volumetric context integration via hybrid 2D-3D training with all available 2D and 3D training datasets; and (3) comprehensive 3D volumetric instruction tuning leveraging the high-quality 3D data in BrainTumor48K. b,c, Uncertainty quantification implementation pipeline. The uncertainty quantification of our BrainVLM consists of two steps: consensus-driven reliability dataset construction and confidenceaware reliable model finetuning. b. Consensus-driven reliability dataset construction. A reliability dataset is constructed by evaluating N vanilla BrainVLM models on the training dataset and clustering their diagnostic answers to derive confidence scores. c. Confidence-aware reliable BrainVLM finetuning. The consensus-derived reliability knowledge in the reliability dataset is distilled into reliable BrainVLM through confidence-aware reliable fine-tuning. d. Confidence-triggered Top 2 supplementary diagnosis inference. During deployment, each prediction made by the reliable BrainVLM is accompanied by a confidence score. If the confidence score of the Top 1 prediction falls below a predefined threshold (set at 75% in this study), the Top 2 prediction is also reported as a supplementary diagnosis to enhance clinical robustness. In this figure, we use abbreviations for tumor categories: MET (brain metastases), GCT (germ cell tumors), GGN (gliomas, glioneuronal tumors, and neuronal tumors), MEN (meningioma), TSR (tumors of the sellar region), MNM (mesenchymal, non-meningothelial tumors), CPN (cranial and paraspinal nerve tumors), CPT (choroid plexus tumors), HEM (hematolymphoid tumors), EMB (embryonal tumors), PIN (pineal region tumors), and MEL (melanocytic tumors).

a. T1 Comparison

![](images/e5629888e849e151116473f8dfdfb055be2c874e906a3e18d750e5980dce6e5b.jpg)  
b. T1c Comparison

![](images/a36dee77ebc433574539ead276b66c4079d6d7f3b9ba42e22158ed23b8e68e1c.jpg)

![](images/ab3370b566c48bd530662dcdc5c00915589f25d6290b9d32a1910e7f8e855ca4.jpg)

![](images/326519e7cb5320621709dca6ce244a5caca1895f5887b125ad1731f8113f55c9.jpg)

![](images/71aa1a7d1581456bd72af90ad57beb7f6f41df15cce83f1093a54a46acd4ba65.jpg)

![](images/dc3513e625954d90d6dc2f0ccf246d262a6e5fc6257a6835e30ae90d4da41ef5.jpg)

![](images/042683c093d9c3ecd02c45da2d33860ddff209a37b7911b1091e841f3de4d660.jpg)

![](images/6722f356fa4193665e5f6c784c521c2214a5c4f0a1a26db5d1899737b7584d28.jpg)

![](images/51ca4c7b7bf7fa315e90dc0b2a506fc567f3b5ab8a385e884cb03fe422a4b973.jpg)

![](images/95ca785d96b3b512d85e9885aefc3b67ca2e2729c49a5bb3ee72d45df4b1cd0c.jpg)

![](images/bcb2edaac2d4a3ba395ccdb2a3ccd7c957d0fe9625343d9c49399d07e10be7e0.jpg)

![](images/c69b069aabeea235530b9d5bc2cb5c3c72e2bf0ca4f88f31d1c6e7a9338da27f.jpg)  
c. T2 Comparison

![](images/8732c8f813271ac6a449445666f7b5aad51d5cb94549efd97bf9496218a93a32.jpg)

![](images/f32e51137b72e748c0b4a4e66e7282a66e6c3348aa7918458bb7a95fe4372829.jpg)  
d. FLAIR Comparison

![](images/d91acd8693ae81aad6bc05ef0d0df7b13b38cbed931fe0c2244afabf68789fee.jpg)

![](images/fdb152e7b70915290a2bd4659ca04341d23e15604a808a779fc484764485dd39.jpg)

![](images/0fcd01c1f13fa308b61138802ef40ad29d09712d7caefbe7d2a8e4593ec1b038.jpg)

![](images/7fc60b82016622189325ac2d5d4ba5149b5595558b00d26684afb83034b18093.jpg)

![](images/9034508ce8872c2eb48142b46edad7145753943e78cfa3d584d50eadd0daab74.jpg)

![](images/7ebc99b545528a9fdedba520c9144e019eeaed20fbee84305cee7cf956051e43.jpg)

![](images/b0d2512ff6537121daeee113b15d8837a54a463489cb92f4151b89b3db897596.jpg)

![](images/96ace72c187cb33e0000d4276ed9ee5c05ed2d0ae0dbf2b8a1eb1e6e22b1dfbf.jpg)

![](images/8d21e70d19f0398ec40d0f48a9dcc60aee488f0d5140dd3d78d3611c9145285c.jpg)

![](images/88ea39b1df95d3b489674c61880a7c1b9b9d5f3a95b28560b8ec63f422072baa.jpg)

Supplementary Figure 6 | Visualization of MRI images acquired using different imaging parameters across MRI sequences. Shown are T1-weighted (T1), contrast-enhanced T1-weighted (T1c), T2-weighted (T2), and T2- Flair (FLAIR) sequences. For each sequence, four axial images obtained using different imaging parameters and MRI devices are presented. Additionally, one representative sagittal and one coronal view acquired under a specific parameter setting are included for each sequence.

![](images/fa05c8a26b8a20a5411beb650c67351d3a5e546d9606b2fb0d6228c5f96a7500.jpg)  
b

![](images/d23d54652c00f48b9bd198611d735db941626f81f17707c7ad05b037f3485c1f.jpg)

![](images/338fa1229b5d82f7ee345796cf093a0f8a76dfbdc17eab95941d6179bcc375c0.jpg)  
d

![](images/eae97b997606fa672de7ccfd15de65ff09693217e06c4139083efd6e50def3cb.jpg)  
Supplementary Figure 7 | Impact of individual MRI sequences on BrainVLM diagnostic accuracy for brain tumor classification and report result. Diagnostic performance was evaluated in 3,571 patients by comparing the full-sequence model (T1, T1c, T2, T2-Flair) to models in which each sequence was individually omitted. Only patients with complete MRI data across all four sequences were included to eliminate potential confounding due to missing sequences. The remaining 1,555 patients in the original primary and external test datasets with missing sequences were excluded from the analysis. For each tumor type, the performance drop rate (measured in F1 score) resulting from the exclusion of a specific modality is shown (b). Both the direct performance drop rate and Shapley value analysis (a) highlight distinct sequence dependencies among tumor classes: T1c removal resulted in the greatest performance decline across all tumor types, underscoring its essential role. EMB classification was uniformly sensitive to the absence of any sequence (35% mean drop), while GGN and MEN predictions remained stable when T2 or T2-Flair or T1 was missing. In contrast, CPT predictions were more dependent on T1, T1c and T2-Flair, but showed resilience to the exclusion of T2. c, Molecular subgroups F1 scores comparison between our BrainVLM with Merlin and RadFM. d. Comparative analysis of key description accuracy among BrainVLM, Merlin, and RadFM, validated by Qwen-2.5-72B on the overall retrospective dataset.

Report Score Comparison by Radiologists  
![](images/d59d94bacd47b478bdaa2f5c449c5e98da562111eef54830ae5b8166b830dd60.jpg)  
Supplementary Figure 8 | Comparison of report generation quality by three expert neuroradiologists. Reports generated by different models were scored on a 0-5 scale based on their consistency with ground truth clinical reports.

a
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Sensitivity Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Kappa</td></tr><tr><td rowspan=1 colspan=1>Doctor</td><td rowspan=1 colspan=2>0.84    0.84</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.80</td></tr><tr><td rowspan=1 colspan=1>BrainVLM</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.82</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Sensitivity Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Kappa</td></tr><tr><td rowspan=1 colspan=1>Doctor</td><td rowspan=1 colspan=2>0.72    0.73</td><td rowspan=1 colspan=2>0.72    0.65</td></tr><tr><td rowspan=1 colspan=1>BrainVLM</td><td rowspan=1 colspan=2>0.78    0.79</td><td rowspan=1 colspan=2>0.78    0.73</td></tr></table>

b

![](images/36b33322a545c7d3f48456172ad08550ef891dc2ced6d7aecc0ac7dbc9e827a2.jpg)

C  
![](images/004ac840163a790c13eafde693baa959138d8972a25221c7b0c15bfc6d15a73e.jpg)

![](images/1e68a372d13e3989b656836364723313574bc8d9734f7974ebc8db449586aba2.jpg)

d  
![](images/8ea72144cddcd5b0c70f20bd0e51c65e88564820d1ff047d1dfec2f50a6200cf.jpg)

![](images/b0e682ef009ddc1377d0736a57b0d41789ca7c3d5013b13be6e3379ca9b87c5b.jpg)  
Supplementary Figure 9 | Top 2 diagnosis performance of BrainVLM and analysis in AI-incorrect scenarios for multi-reader study. a, Top 2 diagnosis metrics including F1 score, sensitivity, precision, and Cohen’s kappa comparison between our BrainVLM and neuroradiologists. b, Diagnostic performance (F1 score) across specific GGN subcategories (glioneuronal tumors, gliomas, and ependymal tumors), comparing BrainVLM with clinicians and baseline models. c, Comparison of Macro F1 scores (top) and weighted accuracy (bottom) between BrainVLM, clinicians, and baseline models (Merlin, RadFM) on the primary and external test datasets.d, F1 score comparison across 12 tumor types, evaluating the Top 2 predictions of our model and neuroradiologists on both primary and external datasets. Abbreviations: MET = brain metastases, GCT = germ cell tumors, GGN = gliomas, glioneuronal tumors, and neuronal tumors, MEN = meningioma, TSR = tumors of the sellar region, MNM = mesenchymal, nonmeningothelial tumors, CPN = cranial and paraspinal nerve tumors, CPT = choroid plexus tumors, HEM = hematolymphoid tumors, EMB = embryonal tumors, PIN = pineal region tumors and MEL = melanocytic tumors.

![](images/ade18073ce048e8fbea21c53a5bde402dd1149a026e789fe28c01c62fc963ed4.jpg)

![](images/90f11852df3c0f057afee2b39ab5c7ec37f444b3ad18ab30e8fcc1b97834913e.jpg)

![](images/e765ea6b29e1f3360652b5fdd733e3bd97eb39fc4c0e0c1165b9eee02dfd2ab1.jpg)

![](images/319dd815c962b9e63a9a7b3c38c796165259f7bfde8c11cb2a6ff40b12f37573.jpg)

![](images/373157948060ec59af4f855300f87fe33b19bd8254b629bc24c0adb9edc16626.jpg)

![](images/254f217f23808bd9c4fe63d6afd202190e9b84a7d0b6a40aa3dd595c8047b93b.jpg)

![](images/a39c165214b4dc968bc4eb6c3ec7d3b04c4276349dcbfbead37fdc47b7cf6570.jpg)

![](images/71f4d1fe2d51cb0708f74fc0ed671046fab17e024afef2d659e230b7d93064d9.jpg)

![](images/7bdb92ce1529dcd450feb7029343495cd862d0dc554a8870d990c1355b98a23d.jpg)  
Supplementary Figure 10 | Comparison of ROC curves among BrainVLM, Neuroradiologists and other baseline models. a, ROC curves for Top-2 predictions on the retrospective primary and external datasets, with macro-averaged AUCs of 0.88/0.81 and micro-averaged 0.95/0.89, respectively. b, Macro Top1 ROC across intraaxial and extraaxial tumor type. c, Micro-averaged ROC curves comparing BrainVLM against human readers in an expanded multi-reader study. d, Macro Top2 ROC across intraaxial and extraaxial tumor type. e, Performance was evaluated on 248 retrospective brain tumor cases interpreted by 12 neuroradiologists, who were stratified by their expertise levels into expert, senior, and junior groups. f, Macro-ROC comparison against baseline models (Merlin, RadFM, and VST) on the primary (left) and external (right) datasets.

Primary 12-Class FNR (Model vs Doctor)  
![](images/6e6775fbf63072061ca518f6bfa39d2aae4931dcaac1bf03224d96d2e6d4916d.jpg)

External 12-Class FNR (Model vs Doctor)  
![](images/e7b752bbb86da7050babe52ded6a98d82dba239211666bc6f31c723bfe7de08f.jpg)

Supplementary Figure 11 | Comparison of the false negative rates for 12 brain tumor types between BrainVLM and board-certified neuroradiologists in the primary (top) and external (bottom) retrospective datasets. Statistical significance was determined using a one-sided Wilcoxon test. Significance levels are denoted by asterisks (★ p<0.05, ★★ p<0.01, ★★★ p<0.001; ns, not significant). The color of the significance markers indicates the direction of the one-sided hypothesis test: black markers denote tests evaluating if the model's FNR is significantly lower than the doctors' (Model < Doctor), while blue markers denote tests evaluating if the doctors' FNR is significantly lower than the model's (Doctor < Model).

![](images/29f0cc654a9d6d30a9eb55561cc10cfb2ed4467155c35fcf97e07cbf8c956368.jpg)

![](images/592cdda8c2cb18cd5326e176e33c2c8aafb8384634092746622323702093d6ce.jpg)

![](images/fc8f7b14daa1de5dea25bf0ebf0c5fdcd1991ba9acdbc6a4becad3f754ea0f56.jpg)

b  
![](images/79abb56d9f626cf14f6567acb1c27f717c5b57478b8716168cd67c9620ab651c.jpg)

![](images/56ad4c06c598ef4dd56d3544dcfac34a55bcf78af1c4027dcbb371477526e805.jpg)

![](images/f103c9741fbac9e0733605ecff91f2fa9d07348fb3e7a2744afc53e1fa2274c6.jpg)

![](images/41be203e8be91ed0b5763d9c381f3ec13be2ac80f2781aff88561cc0422c4742.jpg)

![](images/9e1b605c8b475986b5c308bf7016b718a570d71149dcce307d3419ae8e3c8078.jpg)  
Supplementary Figure 12 | a. Confidence distribution of top 3 classes including TSR, GGN and MEN. b. Confidence distribution of bottom 5 classes based on F1 score including PIN, MEL, CPT, MNM and MET. In this table, we use abbreviations for tumor categories: MET (brain metastases), GCT (germ cell tumors), GGN (gliomas, glioneuronal tumors, and neuronal tumors), MEN (meningioma), TSR (tumors of the sellar region), MNM (mesenchymal, non-meningothelial tumors), CPN (cranial and paraspinal nerve tumors), CPT (choroid plexus tumors), HEM (hematolymphoid tumors), EMB (embryonal tumors), PIN (pineal region tumors), and MEL (melanocytic tumors).

![](images/504cfd9d66673c1c45441207bc0da734edfe5ab7cfb8152b3cedf82e5637ca53.jpg)

b  
![](images/2454f465c2574c21d1a934de4275ab46903fd94a9ed1bf73495a025c12c22905.jpg)

![](images/9dae215f4679386e617177e1dee0c36ee2bce48344afb746e65371c7e7c0b49d.jpg)

![](images/6a2a0227a60a9b79d92f25d40703359f3646e4e37bf1b4b382aff8514e5c0597.jpg)

![](images/7c811660fd1d16abf676eec7c54b554d33ab66a1b8f0a2ee9d0d80f531b51fc3.jpg)

![](images/e5c4a54ab354770c4d9f112456f25a5ab21802538ca73343470196be69391d28.jpg)  
Supplementary Figure 13 | Subgroup analysis across age, sex, and MRI vendor. a. BrainVLM F1 score across different age, sex, and MRI vendors (GE, Philips, Siemens, etc.). b. F1 comparison between different ethnicity source: Newly public dataset (EGD dataset, Brain-Mets-Lung and Vestibular-Schwannoma-MC2). c. F1-score comparison between BrainVLM (red) and doctors (blue) for pediatric (0–20 years) and total cohorts d. F1 comparison between BrainVLM and Radiologist in male and female. In this table, we use abbreviations for tumor categories: MET (brain metastases), GCT (germ cell tumors), GGN (gliomas, glioneuronal tumors, and neuronal tumors), MEN (meningioma), TSR (tumors of the sellar region), MNM (mesenchymal, non-meningothelial tumors), CPN (cranial and paraspinal nerve tumors), CPT (choroid plexus tumors), HEM (hematolymphoid tumors), EMB (embryonal tumors), PIN (pineal region tumors), and MEL (melanocytic tumors).

Ethinc analysis: Model Performance Comparison on Public Test Set  
![](images/6b29556076d8fb677df3ffcd9f06f9070e1e56cf4251ab36c45fdd6f0ba9ec73.jpg)

![](images/80ed7c8bfbaa333cb1cb38dcc68b7191fd886c008b4593395acbce9fb3e6282b.jpg)

![](images/48f02ca080adf8349dabaf4cdecf47a62277611f9b0eaf0303214324580f67d6.jpg)

![](images/38e62c2ce477c61607ba82a864aa844664088522f1b8c865fc4c896d95766fa6.jpg)  
Supplementary Figure 14 | Performance comparison between BrainVLM, Merlin and RadFM in the public ethnic dataset. GGN represents the EGD dataset, CPN represents the Vestibular-Schwannoma-MC2 dataset and MET represents the Brain-Mets-Lung dataset.

Patient Demographic: 76-year-old female  
![](images/c35c3e85033d8babe7e64e449c0ebf356dcac6f599ac1ce148e9154302364a84.jpg)

![](images/46e2275dbf76b24c29a1304ed95913dfd570dd06f2a03ca171d5cc10829ef367.jpg)

![](images/04e1157289286b029e3d93109964612f0652a89f1aed06b7a5a7190a32e8ec94.jpg)

![](images/55b38397bcf54c4114db913f1eebfe39f2a16f1e123f1f6d1ef451a0d8f1ef13.jpg)

![](images/f4310041f9e9e5e1b7a655739d1180e568a57a25d54ea54281340ebbb3f3f50d.jpg)

![](images/7712d25dbacde4367ee359a8f63b813eb27557a93f7dd2c31c99fd61678e6c94.jpg)

Patient Demographic: 35-year-old male  
![](images/6a60bd0958940dc2708f7675a7a346cdc1547662dd73b47ea1bd53ab8da9d03b.jpg)

![](images/954054a1403b3afcd6a15bc780b37b24f1a933f1f00a83157b5f1bfd0c7420c3.jpg)

![](images/216db7ecab3bd94f70839ec545dabb468e5e3836014317f7d2f8392ca643c354.jpg)

![](images/51e698a955db49776f993909f225dc013ae3fac2e12b174caf515afa02503d72.jpg)

![](images/f478ba6a723d5747483577c26585be7031e24890f26d171a835e5272fc26ed52.jpg)

![](images/1e4484e2bc5c5982b24b83289996e9e93e5d0f0a9f4b19354289fc1ebbae3c4d.jpg)

##

## BrainVLM Generation

Finding: In right frontal lobe, there is a mass lesion with hypointense in T1, hyperintense in T2 hyperintense in FLAIR. After contrast administration, there is a (heterogeneous)significant enhancement. Compression of the right lateral ventricle is observed. No midline structure shift.

Impression: A mass lesion in right frontal lobe, considering the possibility of glioma (confidence 80%)

## Radiology report:

In Right Frontal Lobe, there is a irregular lesion with hypointense-isointense in T1, hyperintense in T2 hyperintense in FLAIR. After contrast administration there is an heterogeneous enhancement. Compression of the anterior horn of the right lateral ventricle is observed. No midline structure shift considering glioma.

Pathology Diagnosis: Brain Metastase Tumor

##

## BrainVLM Generation

Finding: In right cerebellum, there is a mass lesion with hypointense in T1, hyperintense in T2, hyperintense in FLAIR. After contrast administration, there is a marked heterogeneous enhancement.No significant dilatation of the ventricular system. No midline structure shift

Impression: A mass lesion in right frontal lobe, considering the possibility of cranial and paraspinal nerve tumor (confidence 90%).

## Radiology report:

In Right cerebellopontine angle, there is a oval lesion with hypointense in T1, hyperintense in T2, hyperintense in FLAIR. After contrast administration there is a marked irregular enhancement. No significant dilatation of the ventricular system. No midline structure shift, considering cranial and paraspinal nerve tumor.

Pathology Diagnosis: Mesenchymal, nonmeningothelial tumor (Solitary fibrotic tumor)

Supplementary Figure 15 | High-confidence diagnostic errors by BrainVLM. Two challenging cases where BrainVLM generated incorrect diagnoses with high confidence. The generated reports are compared with multimodal MRI, the neuroradiologist’s report, and the pathology gold standard. Text colors indicate accurate features (blue), omissions (grey), and diagnostic errors (red). Notably, BrainVLM’s misdiagnoses aligned with the human radiologists’ errors, highlighting shared clinical reasoning vulnerabilities in highly deceptive cases.

![](images/3f8d561e1e069b6b927ad0adac898970920b45d14c4911164b6081fcb4ae0dc2.jpg)  
Supplementary Figure 16 | Case 1: A brain metastasis patient where BrainVLM-assisted expert diagnosis reduced reading time by 70% (from 132s to 40s) and corrected the initial misdiagnosis. Case 2: A glioma patient with BrainVLM assistance reduced reading time by 71% while increasing diagnostic confidence from 60% to 70%.

![](images/7f65f1bc4f6d1984a617d445cccbaeb0d5df4c27e8946fe40d8a1b25692e615b.jpg)

## Patient Demographic: 50-year-old Female

![](images/c466da779fffa7c7a4d4876d154590cbc523396d293a433b627b41152cf55545.jpg)

![](images/5deaa3c48c0fc9ae180c90fe4c8a8d2f571ba855c453025f221ed2a0a080480d.jpg)

![](images/c751b083d89664de4c8b3f32ff3bf8df064884928db548932c88dff42eb7ad48.jpg)

![](images/e99be57f5eb7ac12dd6edb50928e3ba42f055fe034c65bdf1522d576faedabf2.jpg)

![](images/1e5cc715cf4143c791e827321e9d458f013550a5ae8751e72e5420d4fc4cad90.jpg)

![](images/0d828f3b34eeaed90f417bb92576b272327ceb81d33370340eae7fb013aafe7e.jpg)

![](images/ebe23bb641e5751816d271f5d2823a0c79019e49d7dd46d3438e1c6b7d7181cc.jpg)

![](images/302cb1561985cf0b862c9d3089dacc315be06ef815f6c8dba44276111f86e75a.jpg)

![](images/4d4560d1b98d03b64e5e12caebcea82153c3185252fef5d4cb6574ca472d257e.jpg)

![](images/39c85558d3cb4ff2d30b3eee3eeeba1543edc79ba0e541188248bfb41896f384.jpg)

![](images/4a69035b597da683d79e748bb0003eaee9c88fabf5a9961a821a0705029d290d.jpg)

![](images/1cc1bd74e62c64c4fd00f6d0f0bb87a8c0101da3a207fb6cf1539de24d64eb66.jpg)

![](images/0d323b041cec716699f51c3640ebf23191ad937b285dcb4a3c19401e7635c64c.jpg)

![](images/e5ae85bbead303c49d6221e5179c872549fbb264a7f111b3d42abd77e84e71b0.jpg)  
Supplementary Figure 17 | Examples of reports and diagnoses generated by our BrainVLM, compared with those from doctors (expert-curated radiology report) and other state-of-the-art AI models (RadFM and ChatGPT-4o). For each diagnosis generated by BrainVLM, a corresponding reliability score is also provided. We use a series of abbreviations to replace original categories: MET for brain metastases, GCT for germ cell tumors, GGN for gliomas, glioneuronal tumors, and neuronal tumors, MEN for meningioma, TSR for tumors of the sellar region, MNM for mesenchymal, non-meningothelial tumors, CPN for cranial and paraspinal nerve tumors, CPT for choroid plexus tumors, HEM for hematolymphoid tumors, EMB for embryonal tumors, PIN for pineal region tumors and MEL for melanocytic tumors.

## Patient Demographic: 68-year-old male

![](images/8b264e354e5f5ae0b6efc2969bce9cf1d3e88064a1388b65a1911623258a32f5.jpg)

![](images/f147b084b61d008564a9eca2c234a4f2814c48901e4656795e6b542e192d4000.jpg)

![](images/6cc98301b74efa960e313fc2157cccaff05eacfed76c710a8b0d1188dbb86933.jpg)

![](images/9972c1e0e47ed82f0abfb4a112370a7b72a6991bb013462e93ed9ae11eaa6db6.jpg)

## Patient Demographic: 44-year-old Male

![](images/66d91daa8df214f88381ebcce4116e59bb8320bbf9df1b98a30459f96b4f00c8.jpg)

![](images/3b43fad91049f2a541d7403dc475df21dbb5924ef997ae8461dec5ac52938a60.jpg)

![](images/a492ede00c025926b1043b42730761124fb6b826af73a2860202f3efa160d527.jpg)

![](images/6c6217968a1b4bd287e2c376a5d1c2ad896a580c7b722837d684aec2044a9b23.jpg)

![](images/2426b4d5f6ca7d1bc12c057e54322d1a33c8adf1513568be681277a57255c507.jpg)

![](images/2f8e3ca7c7186f03c6f677bc7955423a3a6516fb9ad5d0c78f109716be527321.jpg)

![](images/7e15a5f1a98424020147ae85498ccf53f682f4734ca5d65a1867d43fd12cb188.jpg)

![](images/12cc0f42114da255789e1e00aa81103a9b8c4a9f2051577a5aa5937a9219a00a.jpg)

![](images/7fb36ded04836a1397fb3a865a2585d46e5bf8c6041d5b264f8da8c0d62f178d.jpg)

![](images/cf3dec5416efdc85cd5e243fd6e2d9413e983633f15ad11dc9ad141289503cf7.jpg)

![](images/57148d2e7d5de9086183d78139e5bdfe9de122e6c3d69c2d3ed795e14d3c152c.jpg)

![](images/07670bca3c38b9ce0b463ec9fb138ea65fc0308e411f85b016f125c2ad025b42.jpg)

![](images/00315222b334e50d12c09195f79198f3676d25fc530d57858d235381d497e4a0.jpg)

![](images/362ff9c0d91bbf6e23df0e3a9828c6cc0b4f27f80b56dcb57ee4009f7ccc66b6.jpg)

##

## Finding

In left cerebellum, there is a nodular lesion with hypointense in T1, hyperintense in T2, hyperintense in FLAIR. There is a hypointense T1 signal. hyperintense T2 signal, hyperintense FLAIR signal in supratentorial white matter. After contrast administration, there is a (heterogenous)significan enhancement, No significant dilatation of the ventricular system. No midline structure shift.

## Impression Impression

A mass lesion in left cerebellum, consider the possibility of brain metastase tumor (Confidence 90%)

Gold Standard Pathology Label (Right temporal lobe)

##

## Finding:

In fourth ventricle, there is a mass lesion with hypointense in T1, hyperintense in T2, hyperintense in FLAIR. After contrast administration, there is a marked heterogeneous enhancement. Dilation of the : supratentorial ventricles is observed. Periventricular interstitial edema is noted. No midline structure shift

Impression: A mass lesion in fourth ventricle, consider possibility of choroid plexus tumour (Confidence 91%)

Gold Standard Pathology Label (Fourth ventricle) Choroid plexus papilloma

![](images/2ca650c0e3c26276dd801d28c3c3c7a223a70801e3e9190c9d3a6703f8e499b7.jpg)  
Supplementary Figure 18 | Examples of reports and diagnoses generated by our BrainVLM, compared with those from doctors (expertcurated radiology report) and other state-of-the-art AI models (RadFM and ChatGPT-4o). For each diagnosis generated by BrainVLM, a corresponding reliability score is also provided. We use a series of abbreviations to replace original categories: MET for brain metastases, GCT for germ cell tumors, GGN for gliomas, glioneuronal tumors, and neuronal tumors, MEN for meningioma, TSR for tumors of the sellar region, MNM for mesenchymal, non-meningothelial tumors, CPN for cranial and paraspinal nerve tumors, CPT for choroid plexus tumors, HEM for hematolymphoid tumors, EMB for embryonal tumors, PIN for pineal region tumors and MEL for melanocytic tumors.

## Radiologist Report

Finding: In left cerebellum, there is a cystic lesion with hypointense in T1. hyperintense in T2 and in FLAlR. There is a hypointense T1 signal hyperintense T2 and FLAIR in supratentorial white matter. After contrast administration, there is a heterogenous enhancement in the parenchymal area. Compression of the fourth ventricle and brainstem is observed. No midline structure shift. Impression: A cystic lesion in left cerebellum, consider glioma

## GPT-4o Generated Report

Finding: In sellar region, there is a mass lesion with isointense signal in T1, hyperintense signal in T2 and FLAIR. After contrast administration, there is mild heterogeneous significant enhancement. No significant dilatation of the ventricular system. (No mention of signal in supratentorial) Impression: A mass lesion in sellar region, consider sellar region tumor

## Radfm Generated Report

Finding: In right frontal lobe, there is a mass lesion with hypointense in T1. hyperintense in T2 and FLAIR. After contrast administration, there is a marked enhancement. No significant dilatation of the ventricular system. No midline structure shift. (No mention of signal in supratentorial) Impression: A mass lesion in right frontal lobe, consider meningioma.

## Radiologist Report

Finding: In fourth ventricle, there is a nodular lesion with hypointense in T1, isointense in T2. After contrast administration, there is a marked heterogeneous enhancement. Dilation of the supratentorial ventricles is observed. Periventricular interstitial edema is noted. No midline structure shift Impression: A nodular lesion in fourth ventricle, consider meningioma.

## GPT-40 Generated report

Finding: In the right frontal lobe . there is a mass lesion showing isointense on T1, hyperintense in T2 and FLAIR. After contrast administration, no enhancement is observed. Midline structures maintain normal alignment, Periventricular interstitial edema is noted Impression: A mass lesion in right frontal lobe, consider meningioma.

## Radfm Generated Report

Finding: In right frontal lobe, there is a mass lesion with hypointense in T1, hyperintense in T2, hyperintense in FLAIR. After contrast administration, there is a (heterogeneous) marked enhancement. No significant dilatation of the ventricular system. No midline structure shift Impression: A mass lesion in right frontal lobe, consider meningioma

Radiologist Report   
Finding: In left frontal, parietal, temporal Lobe, there are multiple round lesions with hypointense-isointense in T1, heterogenous signal in T2. hyperintense in FLAlR. After contrast administration, there is a significant enhancement. No significant dilatation of the ventricular system. Midline structures shift to the right   
Impression: Multiple lesions were observed, consider brain metastase tumor.

## GPT-4o Generated report GPT-4o Generated report

Finding: In the right frontal lobe. there is a lesion with hypointense to isointense signal on T1, hyperintense signal on T2 and FLAIR. There is a surrounding area of vasogenic edema, leading to midline shift and compression of the ipsilateral lateral ventricle. The lesion appears to enhance peripherally post-contrast, suggesting a ring-enhancing mass. Impression: A lesion in right frontal lobe, consider glioma

## RadFM Generated Report