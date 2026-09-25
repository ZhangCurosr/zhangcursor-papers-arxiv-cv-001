# A Multimodal Dataset for Survival Prediction in Resected Pancreatic Ductal Adenocarcinoma

Anh-Tien Nguyen<sup>1,</sup> <sup>2</sup>, Mawuko Tettey<sup>2</sup>, Jacqueline Michelle Metsch<sup>1</sup>, Teresa Zimmer<sup>3</sup>,   
Niklas Ullrich<sup>3</sup>, Mario Düker<sup>3</sup>, Sandra Rüngeling<sup>3</sup>, Kirsten Reuter-Jessen<sup>3</sup>, Tessa Rosenthal<sup>3</sup>,   
Lena-Christin Conradi<sup>4</sup>, Michael Ghadimi<sup>4</sup>, Alexander König<sup>5</sup>, Elisabeth Hessmann<sup>5</sup>, Volker Ellenrieder<sup>5</sup>, Philipp Ströbel<sup>3</sup>, Hanibal Bohnenberger<sup>∗,†3</sup>, Anne-Christin Hauschild<sup>∗,†1</sup>

<sup>1</sup>Institute for Predictive Deep Learning for Medicine and Healthcare, Giessen University, Germany <sup>2</sup>Department of Medical Informatics, University Medical Center Göttingen, Germany <sup>3</sup>Institute of Pathology, University Medical Center Göttingen, Germany

<sup>4</sup>Department of General, Visceral and Pediatric Surgery, University Medical Center Göttingen, Germany <sup>5</sup>Department of Gastroenterology, Gastrointestinal Oncology and Endocrinology, University Medical Center Göttingen, Germany

## Abstract

Survival research in pancreatic ductal adenocarcinoma (PDAC) is limited by the scarcity of datasets linking whole-slide histology with clinical, molecular, and long-term outcome data. We present a retrospective single-centre cohort of 302 patients who underwent PDAC resection at University Medical Center Göttingen. The dataset comprises 446 H&E whole-slide images, clinicopathological variables, targeted sequencing data for 154 patients, and overall-survival outcomes. During follow-up, 253 patients died, and the median follow-up was 76 months.

To establish initial reference values, we evaluated fourteen survival-prediction configurations using identical five-repetition Monte Carlo cross-validation partitions. Ridge Cox regression using numeric clinicopathological variables achieved a mean concordance of $0 . 6 4 9 \pm 0 . 0 4 2$ and $0 . 6 5 2 \pm 0 . 0 4 6$ after adding KRAS and TP53 mutation status. The image-only attention model achieved $0 . 6 0 3 \pm 0 . 0 3 0$ , while multimodal fusion achieved $0 . 6 1 9 \pm 0 . 0 2 5$ , the highest concordance among the neural models. These results establish promising initial benchmarks for future research using this pancreas-specific multimodal dataset, paving the way for external validation.

Keywords: pancreas cancer; survival prediction; whole slide images; multimodal fusion

## 1 Introduction

Pancreatic cancer remains one of the most lethal solid tumors, with an estimated 511,000 new cases and 467,000 deaths worldwide in 2022 alone [1]. Although the five-year relative survival for cancer overall has now reached a historic milestone of 70%, survival for pancreatic cancer remains below 15%, the lowest of any major solid tumor [2]. This gap reflects both late-stage diagnosis and a limited ability to identify patients who have undergone resection into those likely to benefit from adjuvant therapy and those at highest risk of early recurrence. Reliable, individualized prognosis is therefore essential, helping guide treatment decisions, identify patients for clinical trials, and plan follow-up after surgery.

Clinical prognosis in resected pancreatic cancer still relies primarily on TNM stage together with a small set of pathology-report variables such as grade, margin status, and lymph node ratio. Yet multi-institutional validation of the current AJCC staging system has repeatedly shown wide survival overlap between stages and strong heterogeneity within the same stage [3]. Transcriptomic profiling has revealed that pancreatic tumors segregate into biologically distinct subtypes (e.g., classical versus basal-like, or squamous, immunogenic, and ADEX subtypes) with markedly diferent outcomes that are not captured by stage alone [4, 5]. These findings indicate that stage-based, single-variable prognostic schemes disregard clinically relevant biological information, motivating approaches that jointly model histomorphology, clinical covariates, and molecular measurements rather than treating them as separate, competing sources of evidence.

Outside of pancreatic cancer, integrating whole-slide histology with molecular and clinical data has repeatedly improved survival prediction relative to any single modality. Early work combining convolutional features from H&E images with genomic markers in glioma outperformed models built on either modality alone [6]. Other architectures such as Pathomic Fusion [7], multimodal co-attention transformers [8], and encoder-based multimodal survival models trained across cancer types [9] have shown that attention- and fusion-based combinations of image and omics data improves model performance. In parallel, vision-language pathology foundation models adapted through multi-granular prompt learning have improved label-eficient whole-slide representation learning under limited annotation [10], pointing to a complementary route toward richer histology encoders. These results show that combining histology with molecular and clinical data is a promising direction in computational oncology.

Translating this to pancreatic cancer, however, is not easy. Pancreatic ductal adenocarcinoma is a particularly demanding setting for multimodal fusion as tumor cellularity in resection specimens is often low and diluted within a dense desmoplastic stroma [11]. Additionally, morphological presentation is heterogeneous even within a single tumor, and, as detailed in Section 2.1, the public resources needed to jointly develop and validate such models (histology paired with both structured clinical follow-up and molecular data at survival-relevant scale) remain scarce. As a result, prior pancreas-specific studies have largely evaluated a single modality at a time, or have folded a small pancreatic subset into a pan-cancer multimodal benchmark rather than modeling the disease by itself (Section 2.2).

In this work, we address this gap by introducing a large, multi-slide pancreatic cancer histopathology cohort with matched clinical variables and targeted mutation-panel sequencing (described in Section 2.1), together with an initial multimodal benchmark for overall-survival prediction. Specifically, we (i) learn slide-level representations from H&E whole-slide images (WSI) and combine them with structured clinical covariates and mutation status within a shared survival-modeling framework; (ii) benchmark this multimodal approach against unimodal and pairwise-modality baselines to quantif the contribution of histology, clinical data, and mutation status individually and in combination; and (iii) report these results as initial reference values rather than a definitive modeling strategy, so that the resulting dataset and benchmarks together provide a reproducible, pancreas-specific test set for future multimodal survival-prediction research. We hope that this resource lowers the barrier to future work integrating histomorphology and molecular profiling for pancreatic cancer, a disease where prognostic improvement remains most urgently needed.

## 2 Related Work

## 2.1 Public histopathology resources for pancreatic cancer

Progress in computational pathology for pancreatic cancer has been limited by the scarcity of public whole-slide images (WSIs) linked to patient-level outcomes. The public data landscape is dominated by two US consortium cohorts. The Cancer Genome Atlas pancreatic adenocarcinoma cohort (TCGA-PAAD) contains 185 cases with multi-omic profiling and clinical follow-up [12]. After histologic review and quality control, however, image-based studies typically retain only approximately 120-170 patients, often represented by a single diagnostic formalin-fixed slide [13–15]. The Clinical Proteomic Tumor Analysis Consortium pancreatic cohort (CPTAC-PDA) provides 557 WSIs from 168 subjects, together with extensive proteogenomic characterization [16]. Thus, although TCGA-PAAD and CPTAC-PDA provide the most comprehensive public resources for outcome-oriented studies, each contains fewer than 200 patients.

Other pancreatic histopathology resources were created for narrower tasks and do not provide longitudinal outcomes. PAIP 2021 contains 80 pancreatobiliary WSIs with pixel-level annotations for perineural-invasion detection [17], whereas PAIP 2023 provides 80 pancreatic H&E image patches for tumor-cellularity estimation [18]. Both require registration or a data-use agreement. Pancreatic tissue also occurs as a small subset of pan-tissue collections for nuclei segmentation and mitosis detection [19–22], and several derived datasets add patch-level labels or annotations to TCGA-PAAD without introducing independent patients [23–25]. Spatial-omics resources pair pancreatic H&E sections with transcriptomic measurements, but currently comprise only tens of samples and lack survival endpoints [26, 27]. Larger initiatives, including a pancreatic research portal and a population-based virtual tissue repository, remain accessible only by application [28, 29]. GTEx supplies normal-pancreas slides that can serve as controls, but not matched pancreatic cancer outcomes [30].

Consequently, the available data are fragmented across diagnostic, segmentation, molecular, and spatial-omics tasks. To our knowledge, TCGA-PAAD and CPTAC-PDA remain the only public cohorts that combine pancreatic cancer WSIs with both molecular profiles and survival information, and neither ofers a large, independent population for survival-model development and external validation.

## 2.2 Computational pathology of pancreatic cancer

Existing methods can be grouped into diagnosis, molecular characterization, prognosis, and tissue quantification. For diagnosis, convolutional and multiple-instance learning (MIL) models have detected pancreatic ductal adenocarcinoma (PDAC) in resection WSIs and endoscopic-ultrasoundguided fine-needle biopsies, while related systems support rapid on-site cytologic evaluation [31–33]. Strong internal performance does not necessarily translate across cohorts. For example, a model trained on TCGA and GTEx achieved an internal AUC of 0.997 but only 0.668 when evaluated on CPTAC; the reciprocal CPTAC-to-TCGA/GTEx transfer achieved an AUC of 0.839 [34]. The same study showed that dataset-specific features could dominate class-specific morphology, illustrating the need for independent, geographically distinct validation data.

Histologic morphology also contains information about tumor biology. Pan-cancer and pancreasspecific studies have predicted genomic alterations and transcriptomic subtypes directly from H&E images [35–37]. PACpAInt, for example, used 1,796 slides from 598 patients to predict basal-like and classical PDAC subtypes and map intratumoral heterogeneity; only its 126-patient TCGA validation cohort is publicly available [13]. This distinction is important because large multi-slide studies can characterize heterogeneity more efectively, yet their institutional data generally cannot be redistributed.

For prognosis, prior work has extracted interpretable features such as stromal and lymphocytic composition [38], tumor–stroma ratio [15], and spatial heterogeneity [39]. Other approaches learn prognostic representations using attention-based MIL or transcriptome-linked morphologic clusters [14, 40, 41]. Pan-cancer multimodal models have also included TCGA-PAAD, although reported concordance indices are approximately 0.6 [42]. In parallel, segmentation models quantify tumor, stroma, and residual disease after neoadjuvant therapy, including an international validation study involving 528 patients and four scanner types [43–45]. Collectively, these studies demonstrate the prognostic value of pancreatic histomorphology, but most larger development and validation cohorts remain institutional or consortium resources that are not publicly redistributable.

General-purpose pathology foundation models have broadened the range of transferable representations, but pancreas-specific evaluation remains limited. UNI [46], CONCH [47], and Prov-GigaPath [48] did not report a dedicated pancreatic benchmark in their original evaluations. Virchow included pancreatic tissue within a pooled pan-cancer detection task [49], and CHIEF used pancreatic datasets for cancer detection but not for a dedicated pancreatic survival analysis [50]. More recently, PRISM2 included TCGA-PAAD among 13 TCGA disease-specific survival tasks [51]. This result demonstrates the transferability of slide-level foundation-model representations, but PAAD remains one component of a pan-cancer benchmark rather than an independent, pancreas-specific surviva validation cohort.

Against this background, we introduce an institutional multimodal PDAC dataset collected at University Medical Center Göttingen. The dataset combines multiple H&E WSIs per patient with structured clinicopathological variables, targeted panel sequencing, and long-term overall survival data. It complements TCGA-PAAD and CPTAC-PDA by providing an independent patient population with linked histological, clinical, molecular, and outcome information. To characterise the dataset and facilitate future comparisons, we report initial survival-prediction benchmarks using individual modalities and multimodal combinations. These experiments are intended to establish reference performance for this cohort rather than to propose a definitive survival-prediction model. The dataset therefore provides a resource for future research on cross-cohort validation, patientlevel multi-slide aggregation, multimodal learning, and pancreas-specific evaluation of pathology foundation models.

## 3 Dataset

## 3.1 Study population and data sources

We conducted a retrospective, single-centre cohort study of patients who underwent surgical resection for pancreatic ductal adenocarcinoma (PDAC) at University Medical Center Göttingen between 2000 and 2021. Patients who had received neoadjuvant treatment were excluded. Patients were eligible if they had documented vital status and overall survival (OS) time, an OS exceeding two months, and at least one available archival haematoxylin and eosin (H&E)-stained whole-slide image. The two-month threshold was applied to reduce the influence of early postoperative mortality on the survival benchmark. Of the 631 patients screened, 302 met all eligibility criteria and comprised the final study cohort. For each case, the H&E-stained slide or slides with the highest tumour cell content were selected and digitised using a Philips scanner at 40× magnification. During follow-up,

253 patients (83.8%) died and 49 (16.2%) were alive at last follow-up and were therefore censored. Kaplan-Meier analysis of the analysis cohort yielded a median OS of 21.0 months, with estimated survival probabilities of 69.5%, 42.9%, 29.6%, and 15.5% at 1, 2, 3, and 5 years, respectively. Median follow-up, estimated by the reverse Kaplan–Meier method, was 76.0 months. Cohort characteristics are summarised in Table 1.

Table 1: Characteristics of the WSI analysis cohort (n = 302 patients).
<table><tr><td>Characteristic</td><td>Value</td><td>Missing, n (%)</td></tr><tr><td>Age at diagnosis, years - median [IQR]</td><td>68 [62-74]</td><td>0 (0.0)</td></tr><tr><td>Sex, male - n (%)</td><td>179 (59.3)</td><td>0 (0.0)</td></tr><tr><td>Histologic grade - n (%)</td><td></td><td>0 (0.0)</td></tr><tr><td>G1</td><td>7 (2.3)</td><td></td></tr><tr><td>G2</td><td>206 (68.2)</td><td></td></tr><tr><td>G3</td><td>89 (29.5)</td><td></td></tr><tr><td>Pathological T category - n (%)</td><td></td><td>2 (0.7)</td></tr><tr><td> $\mathrm { p T 1 }$ </td><td>19 (6.3)</td><td></td></tr><tr><td> $\mathrm { p T 2 }$ </td><td>79 (26.2)</td><td></td></tr><tr><td> $\mathrm { p T 3 }$ </td><td>190 (62.9)</td><td></td></tr><tr><td> $\mathrm { p T 4 }$ </td><td>12 (4.0)</td><td></td></tr><tr><td>Pathological N category - n (%)</td><td></td><td>0 (0.0)</td></tr><tr><td> $\mathrm { p N 0 }$ </td><td>76 (25.2)</td><td></td></tr><tr><td> $\mathrm { p N 1 }$ </td><td>190 (62.9)</td><td></td></tr><tr><td> $\mathrm { p N 2 }$ </td><td>36 (11.9)</td><td></td></tr><tr><td>Resection margin R1/R2 - n (%)</td><td>100 (33.1)</td><td>5 (1.7)</td></tr><tr><td>Lymph node ratio - median [IQR]</td><td>0.13 [0.00-0.29]</td><td>1 (0.3)</td></tr><tr><td>Examined regional lymph nodes - median [IQR] 16 [10-21]</td><td></td><td>1 (0.3)</td></tr><tr><td>UICC stage group - n (%)</td><td></td><td>2 (0.7)</td></tr><tr><td>I</td><td>31 (10.3)</td><td></td></tr><tr><td>II</td><td>210 (69.5)</td><td></td></tr><tr><td>III</td><td>46 (15.2)</td><td></td></tr><tr><td> $\mathrm { I V }$ </td><td>13 (4.3)</td><td></td></tr><tr><td>Tumour sequenced - n (%)</td><td>154 (51.0)</td><td></td></tr><tr><td> $K R A S$  pathogenic - n  $( \% \ \mathrm { s e q . } )$ </td><td>133 (86.4)</td><td></td></tr><tr><td> $T P 5 3$  pathogenic - n (% seq.)</td><td>78 (50.6)</td><td></td></tr><tr><td>Whole-slide images - n</td><td>446</td><td></td></tr><tr><td>Slides per patient - median [IQR]</td><td>1 [1–2]</td><td></td></tr><tr><td>Deaths during follow-up - n (%)</td><td>253 (83.8)</td><td></td></tr><tr><td>Median OS (Kaplan-Meier), months</td><td>21.0</td><td></td></tr><tr><td>1-/2-/3-/5-year OS, %</td><td>69.5/42.9/29.6/15.5</td><td></td></tr><tr><td>Median follow-up (reverse KM), months</td><td>76.0</td><td></td></tr></table>

## 3.2 Clinical variables

The primary feature set comprised seven routinely available clinicopathological variables: age at diagnosis (years), sex, histological grade (G1-G3), pathological T category (pT1-pT4), pathological N category (pN0-pN2), resection-margin status (R0 vs. R1/R2), and lymph node ratio (LNR). The $\mathrm { p T }$ and pN categories were taken from the original pathology reports. LNR was defined as the number of tumour-involved regional lymph nodes divided by the total number of examined regional lymph nodes. LNR was selected instead of the composite UICC stage group because stage largely reflects pT and pN in resected PDAC, whereas LNR captures variation in relative nodal involvement within pN categories. An alternative feature set replacing LNR with UICC stage group (I-IV) was evaluated in a sensitivity analysis. Among patients with available nodal counts, data-quality checks confirmed that those classified as pN0 had no tumour-involved nodes and that the number of involved nodes never exceeded the number examined.

Missing covariate data were uncommon: pT was missing in 2 patients (0.7%), resection-margin status in 5 (1.7%), LNR in 1 (0.3%), and UICC stage in 2 (0.7%). Age, sex, histological grade, and pN were complete. Missing entries were imputed using medians calculated exclusively from the corresponding training split.

Targeted genomic profiling. Targeted genomic profiling was performed on tumour DNA using a custom-designed next-generation sequencing (NGS) panel covering 102 cancer-associated genes which are listed in the Appendix 1 and 2. The panel targeted all coding exons for the majority of genes.

Tumour DNA was extracted from FFPE tissue sections of the resection specimen. Tumour-rich regions were identified on H&E-stained sections, and macrodissection was performed to enrich for tumour content. Only samples with an estimated minimum tumour cell content of 60% were processed further. Genomic DNA was isolated using the automated Maxwell® system in combination with the CSC FFPE DNA Purification Kit (Promega). DNA was fragmented on a Covaris ME220 instrument to an average fragment size of 150–200 bp and quantified using the Qubit™ dsDNA High Sensitivity Assay Kit (Thermo Fisher Scientific) according to the manufacturer’s instructions.

Library preparation was carried out using the SureSelect XT HS2 Kit (Agilent) with an input of 100 ng fragmented DNA, including end repair, A-tailing, adapter ligation incorporating unique molecular identifiers (UMIs) and PCR amplification. Target enrichment was performed by in-solution hybrid capture using the custom gene panel. Library preparation was automated on the NGS Star system (Hamilton). Equimolar amounts of libraries were pooled and sequenced with 150-bp paired-end reads on an Illumina NextSeq 500 platform.

Raw sequencing data (FASTQ files) were generated on-instrument and processed using CLC Genomics Workbench (version 21.05, Qiagen) for read alignment, variant calling and downstream analysis. Reads were aligned to the human reference genome GRCh37/hg19. Samples with a mean sequencing coverage below 100× were considered failed and excluded. Variant calling included single-nucleotide variants and small insertions or deletions. Variants were retained if they fulfilled predefined quality criteria, including a minimum read count greater than 50, minimum coverage at the variant position greater than 60×, a variant allele frequency greater than 5%, and support by at least ten unique start and end sites. UMIs were used for read annotation and duplicate collapsing using the standard UMI-processing workflow implemented in CLC Genomics Workbench.

Detected variants were interpreted using established clinical knowledge bases (ClinVar and OncoKB) and classified as pathogenic or likely pathogenic, variants of uncertain significance, or benign or likely benign. For the present analysis, KRAS and TP53 mutation status was defined as the presence of at least one pathogenic or likely pathogenic variant in the respective gene.

## 3.3 Outcome encoding for individual survival distributions

The prediction task was to estimate an individual survival distribution (ISD) on a discrete time grid comprising eleven intervals: ten consecutive six-month intervals spanning [0, 60) months and a final open-ended interval [60, ∞). For each patient $i ,$ the observed time $t _ { i }$ (time to death or censoring, in months) was mapped to a zero-indexed time bin, $y _ { i } = \operatorname* { m i n } ( \lfloor t _ { i } / 6 \rfloor , 1 0 )$ and paired with an event indicator $\delta _ { i \cdot }$ , where $\delta _ { i } = 1$ denoted death and $\delta _ { i } = 0$ denoted right censoring. The pair $( y _ { i } , \delta _ { i } )$ constituted the observed outcome label. For censored patients, this label indicated survival up to the censoring time rather than death within the assigned interval. The ISD, represented by the predicted survival function $\hat { S } _ { i } ( t )$ , was a model output evaluated against these observed outcomes, not a directly observed ground-truth distribution.

## 3.4 Cross-validation design

Model performance was evaluated using five repetitions of Monte Carlo cross-validation(MCCV). In each repetition, the full cohort of 302 patients was randomly partitioned at the patient level into mutually exclusive training $( n = 1 5 2 )$ , validation $( n = 7 5 )$ , and test $( n = 7 5 )$ sets in an approximately 2:1:1 ratio, with stratification by event status. Partitions were sampled independently across repetitions. The five partitions were generated once using a fixed random seed and stored for reuse, ensuring that all models and feature representations were trained and evaluated on identical patient splits. Although the three sets were disjoint within each repetition, test sets could overlap across repetitions, as expected in Monte Carlo cross-validation.

## 4 Benchmark

## 4.1 Overview

We evaluated OS prediction using three information sources: routine clinicopathological variables, tumour mutation status, and diagnostic whole-slide images (WSIs). The benchmark distinguished three design components: the input configuration (which information sources were included), the encoding backbone (how inputs were transformed into features), and the survival model (how these features were mapped to survival predictions). Fourteen configurations were evaluated (Table 2), encompassing numeric covariates, text representations of the same covariates generated using two frozen language encoders, WSI-derived features, and multimodal image-text fusion.

Model performance was evaluated using five repetitions of Monte Carlo cross-validation. In each repetition, patients were randomly assigned to training $( n = 1 5 2 )$ , validation $( n = 7 5 )$ , and test $( n = 7 5 )$ sets in an approximately 2:1:1 ratio, with stratification by event status. Patient-level partitions were generated independently across repetitions and reused for all models. Evaluation was performed on the same held-out test sets using a common outcome definition and performance metric. Two baseline models were included: ridge-regularised Cox regression and a deep survival network, both using the same numeric covariates. These baselines were used to compare survival modelling approaches and assess whether alternative input representations or additional data sources improved predictive performance.

## 4.2 Input representations

Numeric covariates. Seven routinely available clinicopathological variables including age at diagnosis, sex, histological grade, $\mathrm { p T }$ , pN, resection-margin status, and lymph node ratio were encoded as a seven-dimensional numeric vector. The extended feature set included two additional binary indicators for pathogenic mutations in TP53 and KRAS. Mutation status was recorded as missing for patients without sequencing data. Within each cross-validation repetition, missing values were imputed using per-variable training-set medians, and all variables were standardised using training-set means and standard deviations. These preprocessing parameters were applied unchanged to the validation and test sets.

Clinical text. Each patient’s covariates were converted into a short description using fixed sentence templates and standard TNM terminology. Missing values were represented as “not reported” rather than imputed. The descriptions therefore used the same source variables as the numeric inputs, while retaining missingness explicitly.

Descriptions were encoded using two frozen models: (i) BioClinical ModernBERT [52], a long-context encoder based on the ModernBERT architecture [53] and pretrained on biomedical and clinical text; and (ii) the text encoder of CONCH [47], a vision-language foundation model for computational pathology that produces 512-dimensional text embeddings.

Whole-slide images. WSIs were divided into non-overlapping patches. Each patch was encoded using the frozen CONCH image encoder [47] to obtain a 512-dimensional embedding. CONCH’s image and text encoders produce representations in a shared embedding space, which was used for image-text fusion.

## 4.3 Whole-slide token selection

The patch set of each slide was reduced using the deterministic, concept-guided FOCUS procedure [54]. Patch relevance was assessed using contrast scoring based on similarity to a bank of 22 pancreaticpathology text prompts (14 tumour-related and 8 background concepts), embedded with the frozen CONCH text encoder. Redundant neighbouring patches were pruned using an embedding-similarity threshold of 0.9.

## 4.4 Survival models

All models produced survival estimates on a common evaluation time grid. These estimates were converted to scalar risk scores using the same definition for concordance evaluation.

Ridge Cox model. A Cox proportional-hazards model [55] with $\ell _ { 2 }$ (ridge) regularisation was fitted using scikit-survival [56].

Individual Survival Distribution (ISD). Each neural model comprised a modality-specific encoder followed by a discrete-time survival head with the same architecture across models. The head mapped the encoded features to eleven logits, one per time interval. The logits were transformed into interval-specific hazards using the sigmoid function,

$$
h _ { i } ( k ) = \sigma ( z _ { i k } ) , \qquad k = 0 , \ldots , 1 0 .
$$

The individual survival distribution (ISD) [57] was represented by the discrete survival function

$$
\hat { S } _ { i } ( k ) = \prod _ { j = 0 } ^ { k } \bigl [ 1 - h _ { i } ( j ) \bigr ] ,
$$

with $\hat { S } _ { i } ( - 1 ) = 1$ . Training minimised the negative log-likelihood for the encoded right-censored outcomes [42, 58]:

$$
\mathcal { L } = - \sum _ { i : , \delta _ { i } = 1 } \Bigl [ \log \hat { S } _ { i } ( y _ { i } - 1 ) + \log h _ { i } ( y _ { i } ) \Bigr ] - \sum _ { i : , \delta _ { i } = 0 } \log \hat { S } _ { i } ( y _ { i } ) ,\tag{1}
$$

where $y _ { i }$ denotes the observed time-bin index and $\delta _ { i }$ indicates death (1) or right censoring (0).

Attention-based WSI model. Histological information was modelled using the concept-guided FOCUS architecture [54]. The selected patch tokens were projected into the model’s concept space and processed using multi-head cross-attention. Tumour-concept embeddings served as queries, while the projected patch tokens served as keys and values. The resulting concept-specific representations were combined using learned attention pooling and passed to the discrete-time survival head described above.

Multimodal co-attention fusion. Multimodal fusion was implemented using the co-attention mechanism of MCAT [8]. The genomic tokens used in the original architecture were replaced by nine CONCH-derived text tokens including one for each of the seven clinicopathological variables and two mutation indicators. Because the image and text tokens were generated by the paired CONCH encoders, they occupied a shared pretrained embedding space before fusion. Each text token served as a query in single-head scaled dot-product attention over the patient’s WSI tokens, which served as keys and values. This produced one image-conditioned representation for each input variable.

The image-conditioned representations and the original text tokens were aggregated separately using gated attention pooling [59]. The two pooled vectors were concatenated, projected to a 64-dimensional patient representation, and passed to the discrete-time survival head described above. Only the co-attention component of MCAT was adopted; the transformer encoders from the original architecture were not used.

## 4.5 Evaluation and statistical analysis

The primary performance measure was Harrell’s concordance index [60], which assesses the ability of predicted risk scores to rank survival outcomes using patient pairs comparable under right censoring. For each model, concordance was calculated on the held-out test set of each repetition and summarised as mean ± standard deviation across the five Monte Carlo repetitions. All models additionally generated patient-specific ISDs, represented by predicted survival probabilities at the prespecified six-month evaluation time points.

Continuous variables were summarised as medians with interquartile ranges (IQRs), and categorical variables as counts and percentages. Survival curves and median follow-up were estimated using the Kaplan–Meier and reverse Kaplan-Meier methods, respectively.

## 5 Results

Table 2 reports test-set concordance for all fourteen evaluated configurations. These analyses were conducted to establish initial reference values for survival prediction in the institutional PDAC cohort. They are intended to support future model development and comparison using this dataset, rather than to identify a definitive modelling strategy. Five observations summarise the benchmark results.

Numeric clinicopathological variables provided the primary reference baselines. Using the seven numeric clinicopathological variables, ridge-regularised Cox regression achieved a mean C-index of $0 . 6 4 9 \pm 0 . 0 4 2$ , whereas the deep discrete-time model achieved $0 . 5 9 9 \pm 0 . 0 4 0 $ . These results provide linear and neural reference values for the clinical feature set. The diference between the models should be interpreted in light of their fitting protocols: the Cox model was fitted using the combined training and validation sets $( n = 2 2 7 )$ , whereas the neural model was fitted on the training set $( n = 1 5 2 )$ , with the validation set used for epoch selection. The comparison therefore does not isolate the efect of model class alone.

Adding mutation status produced little change in the benchmark results. After the mutation status of TP53 and KRAS was added, the mean C-index of the Cox model changed from $0 . 6 4 9 \pm 0 . 0 4 2$ to $0 . 6 5 2 \pm 0 . 0 4 6$ . The latter was the highest mean concordance observed in the benchmark, although the numerical increase was only 0.003. For the deep discrete-time model, concordance changed from $0 . 5 9 9 \pm 0 . 0 4 0$ to $0 . 5 8 9 \pm 0 . 0 6 0$ . Thus, the addition of mutation status did not produce a consistent improvement across the two modelling approaches. These values serve as initial reference results and do not establish the incremental prognostic value of the mutation variables.

Frozen text embeddings provided alternative baselines for the same covariates. For the mutation-extended feature set, Cox regression achieved a mean C-index of 0.652 using the numeric variables, compared with 0.594 using BioClinical ModernBERT embeddings and 0.538 using CONCH text embeddings. For the clinical-only feature set, performance also varied according to the combination of text encoder and survival model. CONCH embeddings achieved mean concordance values of 0.594 with the deep reader and 0.562 with the Cox reader, whereas BioClinical ModernBERT embeddings achieved 0.558 and 0.594, respectively. These results establish baseline performance for frozen text representations of the tabular variables. Diferences between encoders or readers are reported descriptively because the benchmark was not designed to determine their underlying causes.

Whole-slide images supported survival prediction without clinical covariates. Using WSIs alone, the concept-guided attention model achieved a mean C-index of $0 . 6 0 3 \pm 0 . 0 3 0$ , the highest value among the single-modality neural configurations. Ridge Cox regression applied to the mean-pooled image representation achieved $0 . 5 7 0 \pm 0 . 0 2 1$ . Although both configurations used CONCH patch embeddings and identical data partitions, they difered in token selection, aggregation, survival model, and fitting protocol. The diference of 0.033 therefore describes the performance of the complete image-analysis pipelines and cannot be attributed to aggregation alone.

Multimodal fusion provided the highest neural benchmark. Co-attention fusion of WSI tokens with CONCH-derived tokens representing the clinical and mutation variables achieved a mean C-index of $0 . 6 1 9 \pm 0 . 0 2 5$ . This was the highest mean concordance among the neural configurations and represented numerical increases of 0.016 over the image-only attention model, 0.071 over the corresponding text-only model, and 0.030 over the deep model using the mutation-extended numeric feature set. For the linear Cox models, the combined pooled image-text representation achieved a mean C-index of 0.594, compared with 0.570 for the image representation and 0.538 for the text representation.

The fused neural model remained below the clinical-only and mutation-extended numeric Cox baselines (0.619 vs. 0.649 and 0.652, respectively). Its across-repetition standard deviation was

0.025, compared with 0.042 for the clinical-only Cox model; given the five repetitions, this diference is descriptive rather than evidence of greater stability. The fusion model additionally generated per-variable co-attention maps over the WSI tokens. These maps provide material for subsequent analysis but are not treated as validated explanations in the present benchmark.

Overall, these results establish initial performance references for clinical, molecular, histological, and multimodal survival models applied to the institutional cohort. Further model development and external validation will be required to assess whether these findings generalise beyond this dataset.

Table 2: Test-set concordance (mean ± SD over five Monte Carlo repetitions). Clinical feature set: age, sex, grade, pT, pN, margin status, LNR. Mutation feature set: TP53, KRAS.
<table><tr><td>Input representation</td><td>Backbone</td><td>Model</td><td>C-index</td></tr><tr><td>Clinical</td><td>Numeric</td><td>Cox</td><td> $0 . 6 4 8 9 \pm 0 . 0 4 2 4$ </td></tr><tr><td>Clinical</td><td>Numeric</td><td>ISD</td><td> $0 . 5 9 9 2 \pm 0 . 0 3 9 7$ </td></tr><tr><td>Clinical</td><td>CONCH</td><td>Cox</td><td> $0 . 5 6 2 3 \pm 0 . 0 3 6 3$ </td></tr><tr><td>Clinical</td><td>CONCH</td><td>ISD</td><td> $0 . 5 9 3 5 \pm 0 . 0 1 0 5$ </td></tr><tr><td>Clinical + Mutation</td><td>Numeric</td><td>Cox</td><td> $\mathbf { 0 . 6 5 1 6 \pm 0 . 0 4 5 7 }$ </td></tr><tr><td>Clinical + Mutation</td><td>Numeric</td><td>ISD</td><td> $0 . 5 8 9 3 \pm 0 . 0 6 0 0$ </td></tr><tr><td> $\mathrm { C l i n i c a l + M u t a t i o n }$ </td><td>BioClinical ModernBERT</td><td>Cox</td><td> $0 . 5 9 4 2 \pm 0 . 0 4 5 6$ </td></tr><tr><td> $\mathrm { C l i n i c a l + M u t a t i o n }$ </td><td>BioClinical ModernBERT</td><td>ISD</td><td> $0 . 5 5 7 8 \pm 0 . 0 2 3 8$ </td></tr><tr><td>Clinical + Mutation</td><td>CONCH</td><td>Cox</td><td> $0 . 5 3 7 7 \pm 0 . 0 5 4 5$ </td></tr><tr><td>Clinical + Mutation</td><td>CONCH</td><td>ISD</td><td> $0 . 5 4 7 6 \pm 0 . 0 4 2 8$ </td></tr><tr><td>WSI</td><td>CONCH</td><td>Cox</td><td> $0 . 5 6 9 9 \pm 0 . 0 2 0 9$ </td></tr><tr><td>WSI</td><td>CONCH</td><td>ISD</td><td> $0 . 6 0 2 5 \pm 0 . 0 3 0 2$ </td></tr><tr><td> $\mathrm { W S I } + \mathrm { C l i n i c a l } + \mathrm { M u t a t i o n }$ </td><td>CONCH</td><td>Cox</td><td> $0 . 5 9 4 4 \pm 0 . 0 1 8 1$ </td></tr><tr><td> $\mathrm { W S I } + \mathrm { C l i n i c a l } + \mathrm { M u t a t i o n }$ </td><td>CONCH</td><td>ISD</td><td> $0 . 6 1 8 9 \pm 0 . 0 2 4 5$ </td></tr></table>

## 6 Conclusion

We present a multimodal dataset from an institutional cohort of patients with resected pancreatic ductal adenocarcinoma, linking digitised H&E whole-slide images with routinely collected clinicopathological variables, targeted mutation data, and long-term overall-survival outcomes. By integrating these data at the patient level, the dataset expands the limited resources available for pancreas-specific computational pathology and enables clinical, molecular, histological, and multimodal approaches to be evaluated using a common outcome definition.

To provide initial reference values, we evaluated fourteen combinations of input representation, feature-encoding backbone, and survival model. Ridge-regularised Cox regression using numeric clinicopathological and mutation variables achieved the highest mean concordance in this benchmark (C-index $0 . 6 5 2 \pm 0 . 0 4 6 )$ . The image-only attention model supported survival discrimination from histology alone (C-index 0.603 ± 0.030), while multimodal co-attention achieved the highest concordance among the neural configurations $( \mathrm { C } \mathrm { - i n d e x } \ 0 . 6 1 9 \pm 0 . 0 2 5 )$ . Representing tabular covariates using frozen text embeddings did not improve mean concordance over their numeric representation. These findings should be interpreted as initial dataset benchmarks rather than evidence of an optimal survival-modelling strategy.

The dataset and its accompanying benchmarks provide a foundation for future research on pancreasspecific foundation models, patient-level WSI aggregation, multimodal survival modelling, and cross-cohort validation.

## Acknowledgements

This work is supported in part by funds from the German Ministry of Education and Research (BMBF) under grant agreements No. 01D2208A and No. 01KD2414A (project FAIrPaCT). The authors gratefully acknowledge the computing time granted by the KISSKI project. The calculations for this research were conducted with computing resources under the project kisski-umg-fairpact-2. The authors also acknowledge the computing time granted by the Resource Allocation Board and provided on the supercomputer Emmy/Grete at NHR-Nord@Göttingen as part of the NHR infrastructure. The calculations for this research were conducted with computing resources under the project nim00014. Anh-Tien Nguyen was a member of the Ph.D. program "Genome Science" - International Max Planck Research School.

We gratefully acknowledge support from the hessian.AI Service Center (funded by the Federal Ministry of Research, Technology and Space, BMFTR, grant no. 16IS22091) and the hessian.AI Innovation Lab (funded by the Hessian Ministry for Digital Strategy and Innovation, grant no. S-DIW04/0013/003). This work is supported in part by funds from CancerScout project (German Federal Ministry of Education and Research, BMBF, grant no. 13GW0451A).

## 7 Data availability statement

The datasets generated and/or analysed during the current study are available from the corresponding author on reasonable request.

## 8 Ethics statement

The study was approved by the Ethics Committee of University Medical Center Göttingen (no.   
24-4-20).

## References

[1] Freddie Bray, Mathieu Laversanne, Hyuna Sung, Jacques Ferlay, Rebecca L. Siegel, Isabelle Soerjomataram, and Ahmedin Jemal. Global cancer statistics 2022: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA: A Cancer Journal for Clinicians, 74(3):229–263, 2024. doi: 10.3322/caac.21834.

[2] Rebecca L. Siegel and et al. Cancer statistics, 2026. CA: A Cancer Journal for Clinicians, 2026. doi: 10.3322/caac.70043. Please update author list, volume, and pages from the final published version.

[3] Peter J. Allen, Deborah Kuk, Carlos Fernandez-del Castillo, Olca Basturk, Christopher L. Wolfgang, John L. Cameron, Keith D. Lillemoe, Cristina R. Ferrone, Vicente Morales-Oyarvide, Jin He, Matthew J. Weiss, Ralph H. Hruban, Mithat Gonen, and David S. Klimstra. Multiinstitutional validation study of the American Joint Commission on Cancer (8th edition) changes for T and N staging in patients with pancreatic adenocarcinoma. Annals of Surgery, 265(1):185–191, 2017. doi: 10.1097/SLA.0000000000001763.

[4] Richard A. Mofitt, Raoud Marayati, Elizabeth L. Flate, et al. Virtual microdissection identifies distinct tumor- and stroma-specific subtypes of pancreatic ductal adenocarcinoma. Nature Genetics, 47(10):1168–1178, 2015. doi: 10.1038/ng.3398.

[5] Peter Bailey, David K Chang, Katia Nones, Amber L Johns, Ann-Marie Patch, Marie-Claude Gingras, David K Miller, Angelika N Christ, Tim JC Bruxner, Michael C Quinn, et al. Genomic analyses identify molecular subtypes of pancreatic cancer. Nature, 531(7592):47–52, 2016.

[6] Pooya Mobadersany, Safoora Yousefi, Mohamed Amgad, David A. Gutman, Jill S. Barnholtz-Sloan, José E. Velázquez Vega, Daniel J. Brat, and Lee A. D. Cooper. Predicting cancer outcomes from histology and genomics using convolutional networks. Proceedings of the National Academy of Sciences, 115(13):E2970–E2979, 2018. doi: 10.1073/pnas.1717139115.

[7] Richard J. Chen, Ming Y. Lu, Jingwen Wang, Drew F. K. Williamson, Scott J. Rodig, Neal I. Lindeman, and Faisal Mahmood. Pathomic fusion: an integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis. IEEE Transactions on Medical Imaging, 41(4):757–770, 2022. doi: 10.1109/TMI.2020.3021387. Early access 2020.

[8] Richard J. Chen, Ming Y. Lu, Wei-Hung Weng, Tifany Y. Chen, Drew F. K. Williamson, Trevor Manz, Maha Shady, and Faisal Mahmood. Multimodal co-attention transformer for survival prediction in gigapixel whole slide images. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4015–4025, 2021.

[9] Luis A. Vale-Silva and Karl Rohr. Long-term cancer survival prediction using multimodal deep learning. Scientific Reports, 11:13505, 2021. doi: 10.1038/s41598-021-92799-4.

[10] Anh-Tien Nguyen, Duy Minh Ho Nguyen, Nghiem Tuong Diep, Trung Quoc Nguyen, Daniel Sonntag, Nhat Ho, Jacqueline Michelle Metsch, Miriam Cindy Maurer, Hanibal Bohnenberger, and Anne-Christin Hauschild. MGPATH: A vision-language model with multi-granular prompt learning for few-shot whole slide pathology classification. Transactions on Machine Learning Research, 2025. URL https://openreview.net/forum?id=u7U81JLGjH. Published 09/2025.

[11] Mert Erkan, Simone Hausmann, Christoph W. Michalski, Alexander A. Fingerle, Martin Dobritz, Jörg Kleef, and Helmut Friess. The role of stroma in pancreatic cancer: diagnostic and therapeutic implications. Nature Reviews Gastroenterology & Hepatology, 9(8):454–467, 2012. doi: 10.1038/nrgastro.2012.115.

[12] Benjamin J Raphael, Ralph H Hruban, Andrew J Aguirre, Richard A Mofitt, Jen Jen Yeh, Chip Stewart, A Gordon Robertson, Andrew D Cherniack, Manaswi Gupta, Gad Getz, et al. Integrated genomic characterization of pancreatic ductal adenocarcinoma. Cancer cell, 32(2): 185–203, 2017.

[13] Charlie Saillard, Flore Delecourt, Benoit Schmauch, Olivier Moindrot, Magali Svrcek, Armelle Bardier-Dupas, Jean Francois Emile, Mira Ayadi, Vinciane Rebours, Louis de Mestier, et al. Pacpaint: a histology-based deep learning model uncovers the extensive intratumor molecular heterogeneity of pancreatic adenocarcinoma. Nature communications, 14(1):3459, 2023.

[14] Caroline Truntzer, Dina Ouahbi, Titouan Huppé, David Rageot, Alis Ilie, Chloe Molimard, Françoise Beltjens, Anthony Bergeron, Angelique Vienot, Christophe Borg, et al. Deep multiple instance learning model to predict outcome of pancreatic cancer following surgery. Biomedicines, 12(12):2754, 2024.

[15] Pierpaolo Vendittelli, John-Melle Bokhorst, Esther MM Smeets, Valentyna Kryklyva, Lodewijk AA Brosens, Caroline Verbeke, and Geert Litjens. Automatic quantification of tumor-stroma ratio as a prognostic marker for pancreatic cancer. Plos one, 19(5):e0301969, 2024.

[16] Liwei Cao, Chen Huang, Daniel Cui Zhou, Yingwei Hu, T Mamie Lih, Sara R Savage, Karsten Krug, David J Clark, Michael Schnaubelt, Lijun Chen, et al. Proteogenomic characterization of pancreatic ductal adenocarcinoma. Cell, 184(19):5031–5052, 2021.

[17] Jinwook Choi, Kyoungbun Lee, Won-Ki Jeong, and Se Young Chun. Paip2021: Perineural invasion in multiple organ cancer (colon, prostate, and pancreatobiliary tract). In International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), volume 10, 2021.

[18] Pathology AI Platform (PAIP). PAIP 2023 challenge: Tumor cellularity prediction in pancreatic cancer and colon cancer. IEEE ISBI 2023 Grand Challenge, 2023. URL https://2023paip. grand-challenge.org/.

[19] Jevgenij Gamper, Navid Alemi Koohbanani, Ksenija Benet, Ali Khuram, and Nasir Rajpoot. Pannuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification. In European congress on digital pathology, pages 11–19. Springer, 2019.

[20] Amirreza Mahbod, Christine Polak, Katharina Feldmann, Rumsha Khan, Katharina Gelles, Georg Dorfner, Ramona Woitek, Sepideh Hatamikia, and Isabella Ellinger. Nuinsseg: A fully annotated dataset for nuclei instance segmentation in h&e-stained histological images. Scientific Data, 11(1):295, 2024.

[21] Amirreza Mahbod, Gerald Schaefer, Benjamin Bancher, Christine Löw, Georg Dorfner, Rupert Ecker, and Isabella Ellinger. Cryonuseg: A dataset for nuclei instance segmentation of cryosectioned h&e-stained histological images. Computers in biology and medicine, 132:104349, 2021.

[22] Marc Aubreville, Frauke Wilm, Nikolas Stathonikos, Katharina Breininger, Taryn A Donovan, Samir Jabari, Mitko Veta, Jonathan Ganz, Jonas Ammeling, Paul J Van Diest, et al. A comprehensive multi-domain dataset for mitotic figure detection. Scientific data, 10(1):484, 2023.

[23] Daisuke Komura, Akihiro Kawabe, Keisuke Fukuta, Kyohei Sano, Toshikazu Umezaki, Hirotomo Koda, Ryohei Suzuki, Ken Tominaga, Mieko Ochi, Hiroki Konishi, et al. Universal encoding of pan-cancer histology by deep texture representations. Cell Reports, 38(9), 2022.

[24] Le Hou, Rajarsi Gupta, John S Van Arnam, Yuwei Zhang, Kaustubh Sivalenka, Dimitris Samaras, Tahsin M Kurc, and Joel H Saltz. Dataset of segmented nuclei in hematoxylin and eosin stained histopathology images of ten cancer types. Scientific data, 7(1):185, 2020.

[25] Chiara Loefler and Jakob Nikolas Kather. Manual tumor annotations in tcga, August 2021. URL https://doi.org/10.5281/zenodo.5320076.

[26] Guillaume Jaume, Paul Doucet, Andrew H Song, Ming Y Lu, Cristina Almagro-Pérez, Sophia J Wagner, Anurag J Vaidya, Richard J Chen, Drew F Williamson, Ahrong Kim, et al. Hest-1k: A dataset for spatial transcriptomics and histology image analysis. Advances in Neural Information Processing Systems, 37:53798–53833, 2024.

[27] Ahmed M Elhossiny, Padma Kadiyala, Jude Ogechukwu Okoye, Harrison L Hiraki, Megan C Procario, Thejaswini Giridharan, Hannah R Watkoske, Mariana Tannus Ruckert, Jiayue Wang, Brian D Grifith, et al. Asynchronous evolution of epithelium and stroma diferentiates precursor lesions from pancreatic cancer. Cancer Discovery, 2026.

[28] Jorge Oscanoa, Helen Ross-Adams, Abu ZM Dayem Ullah, Trupti S Kolvekar, Lavanya Sivapalan, Emanuela Gadaleta, Graeme J Thorn, Maryam Abdollahyan, Ahmet Imrali, Amina Saad, et al. A central research portal for mining pancreatic clinical and molecular datasets and accessing biobanked samples. Translational Oncology, 62:102550, 2025.

[29] Pamela Sanchez, Alison L Van Dyke, Valentina I Petkov, Yao Yuan, Sarah Bonds, Connor Valenzuela, Alyssa W Tuan, Radim Moravec, Sean F Altekruse, Aatur D Singhi, et al. Nci seer-linked virtual tissue repository pilot. JNCI Monographs, 2024(65):180–190, 2024.

[30] Leslie Sobin, Mary Barcus, Philip A Branton, Kelly B Engel, Judy Keen, David Tabor, Kristin G Ardlie, Sarah R Greytak, Nancy Roche, Brian Luke, et al. Histologic and quality assessment of genotype-tissue expression (gtex) research samples: a large postmortem tissue collection. Archives of pathology & laboratory medicine, 149(3):217–232, 2025.

[31] Hao Fu, Weiming Mi, Boju Pan, Yucheng Guo, Junjie Li, Rongyan Xu, Jie Zheng, Chunli Zou, Tao Zhang, Zhiyong Liang, et al. Automatic pancreatic ductal adenocarcinoma detection in whole slide images using deep convolutional neural networks. Frontiers in oncology, 11:665929, 2021.

[32] Yoshiki Naito, Masayuki Tsuneki, Noriyoshi Fukushima, Yutaka Koga, Michiyo Higashi, Kenji Notohara, Shinichi Aishima, Nobuyuki Ohike, Takuma Tajiri, Hiroshi Yamaguchi, et al. A deep learning model to detect pancreatic ductal adenocarcinoma on endoscopic ultrasound-guided fine-needle biopsy. Scientific reports, 11(1):8454, 2021.

[33] Song Zhang, Yangfan Zhou, Dehua Tang, Muhan Ni, Jinyu Zheng, Guifang Xu, Chunyan Peng, Shanshan Shen, Qiang Zhan, Xiaoyun Wang, et al. A deep learning-based segmentation system for rapid onsite cytologic pathology evaluation of pancreatic masses: A retrospective, multicenter, diagnostic study. EBioMedicine, 80, 2022.

[34] Francisco Carrillo-Perez, Francisco M Ortuno, Alejandro Börjesson, Ignacio Rojas, and Luis Javier Herrera. Performance comparison between multi-center histopathology datasets of a weakly-supervised deep learning model for pancreatic ductal adenocarcinoma detection. Cancer Imaging, 23(1):66, 2023.

[35] Jakob Nikolas Kather, Lara R Heij, Heike I Grabsch, Chiara Loefler, Amelie Echle, Hannah Sophie Muti, Jeremias Krause, Jan M Niehues, Kai AJ Sommer, Peter Bankhead, et al. Pan-cancer image-based detection of clinically actionable genetic alterations. Nature cancer, 1 (8):789–799, 2020.

[36] Salim Arslan, Julian Schmidt, Cher Bass, Debapriya Mehrotra, Andre Geraldes, Shikha Singhal, Julius Hense, Xiusi Li, Pandu Raharja-Liu, Oscar Maiques, et al. A systematic pan-cancer study on deep learning-based prediction of multi-omic biomarkers from routine pathology images. Communications Medicine, 4(1):48, 2024.

[37] Pouya Ahmadvand, Hossein Farahani, David Farnell, Amirali Darbandsari, James Topham, Joanna Karasinska, Jessica Nelson, Julia Naso, Steven JM Jones, Daniel Renouf, et al. A deep learning approach for the identification of the molecular subtypes of pancreatic ductal adenocarcinoma based on whole slide pathology images. The American Journal of Pathology, 194(12):2302–2312, 2024.

[38] Xiuxiang Tan, Mika Rosin, Simone Appinger, Julia Campello Deierl, Konrad Reichel, Mariëlle Coolsen, Liselot Valkenburg-van Iersel, Judith de Vos-Geelen, Evelien JM de Jong, Jan Bed-

narsch, et al. Stroma and lymphocytes identified by deep learning are independent predictors for survival in pancreatic cancer. Scientific reports, 15(1):9415, 2025.

[39] Lizhi Shao, Xinyi Ke, Yun Wang, Ruiyu Li, Kedian Yu, Junxian Wu, Jingci Chen, Ruping Hong, Zheng Wang, Junliang Lu, et al. Ai-driven tumor heterogeneity quantification and survival prediction in pancreatic ductal adenocarcinoma. npj Digital Medicine, 2026.

[40] Kaixin Hu, Chenyang Bian, Jiayin Yu, Dawei Jiang, Zhangjun Chen, Fengqing Zhao, and Huangbao Li. Construction of a combined prognostic model for pancreatic ductal adenocarcinoma based on deep learning and digital pathology images. BMC gastroenterology, 24(1):387, 2024.

[41] Manabu Takamatsu, Mariko Tanaka, Yohei Masugi, Yosuke Inoue, Hiroko Nagano, Tho Ngoc-Quynh Le, Kenji Nishida, Yui Sawa, Kota Sugiura, Yoshikuni Kawaguchi, et al. Prognostic model for pancreatic cancer based on machine learning of routine slides and transcriptomic tumor analysis: Cellular and molecular biology. British Journal of Cancer, 134(6):849–859, 2026.

[42] Richard J Chen, Ming Y Lu, Drew FK Williamson, Tifany Y Chen, Jana Lipkova, Zahra Noor, Muhammad Shaban, Maha Shady, Mane Williams, Bumjin Joo, et al. Pan-cancer integrative histology-genomic analysis via multimodal deep learning. Cancer cell, 40(8):865–878, 2022.

[43] Pierpaolo Vendittelli, Esther MM Smeets, and Geert JS Litjens. Automatic tumour segmentation in h&e-stained whole-slide images of the pancreas. In Medical Imaging 2022: Digital and Computational Pathology, volume 12039, pages 312–318. SPIE, 2022.

[44] Boris V Janssen, Rutger Theijse, Stijn van Roessel, Rik de Ruiter, Antonie Berkel, Joost Huiskens, Olivier R Busch, Johanna W Wilmink, Geert Kazemier, Pieter Valkema, et al. Artificial intelligence-based segmentation of residual tumor in histopathology of pancreatic cancer after neoadjuvant treatment. Cancers, 13(20):5089, 2021.

[45] Boris V Janssen, Bart Oteman, Mahsoem Ali, Pieter A Valkema, Volkan Adsay, Olca Basturk, Deyali Chatterjee, Angela Chou, Stijn Crobach, Michael Doukas, et al. Artificial intelligencebased segmentation of residual pancreatic cancer in resection specimens following neoadjuvant treatment (isgpp-2): International improvement and validation study. The American Journal of Surgical Pathology, 48(9):1108, 2024.

[46] Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Andrew H Song, Bowen Chen, Andrew Zhang, Daniel Shao, Muhammad Shaban, et al. Towards a generalpurpose foundation model for computational pathology. Nature medicine, 30(3):850–862, 2024.

[47] Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, et al. A visual-language foundation model for computational pathology. Nature medicine, 30(3):863–874, 2024.

[48] Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Clif Wong, Zelalem Gero, Javier González, Yu Gu, et al. A whole-slide foundation model for digital pathology from real-world data. Nature, 630(8015):181–188, 2024.

[49] Eugene Vorontsov, Alican Bozkurt, Adam Casson, George Shaikovski, Michal Zelechowski, Kristen Severson, Eric Zimmermann, James Hall, Neil Tenenholtz, Nicolo Fusi, et al. A

foundation model for clinical-grade computational pathology and rare cancers detection. Nature medicine, 30(10):2924–2935, 2024.

[50] Xiyue Wang, Junhan Zhao, Eliana Marostica, Wei Yuan, Jietian Jin, Jiayu Zhang, Ruijiang Li, Hongping Tang, Kanran Wang, Yu Li, et al. A pathology foundation model for cancer diagnosis and prognosis prediction. Nature, 634(8035):970–978, 2024.

[51] Eugene Vorontsov, George Shaikovski, Adam Casson, Julian Viret, Eric Zimmermann, Neil Tenenholtz, Yi Kan Wang, Jan H Bernhard, Ran A Godrich, Juan A Retamero, et al. End-toend multimodal pathology foundation model with clinical dialogue. Nature Medicine, pages 1–11, 2026.

[52] Thomas Sounack, Joshua Davis, Brigitte Durieux, Antoine Chafin, Tom J Pollard, Eric Lehman, Alistair EW Johnson, Matthew McDermott, Tristan Naumann, and Charlotta Lindvall. Bioclinical modernbert: A state-of-the-art long-context encoder for biomedical and clinical nlp. arXiv preprint arXiv:2506.10896, 2025.

[53] Benjamin Warner, Antoine Chafin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, et al. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory eficient, and long context finetuning and inference. In Proceedings of the 63rd annual meeting of the association for computational linguistics (volume 1: Long papers), pages 2526–2547, 2025.

[54] Zhengrui Guo, Conghao Xiong, Jiabo Ma, Qichen Sun, Lishuang Feng, Jinzhuo Wang, and Hao Chen. Focus: Knowledge-enhanced adaptive visual compression for few-shot whole slide image classification. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15590–15600. IEEE, 2025.

[55] David R Cox. Regression models and life-tables. Journal of the royal statistical society: Series B (methodological), 34(2):187–202, 1972.

[56] Sebastian Pölsterl. scikit-survival: A library for time-to-event analysis built on top of scikit-learn. Journal of Machine Learning Research, 21(212):1–6, 2020.

[57] Humza Haider, Bret Hoehn, Sarah Davis, and Russell Greiner. Efective ways to build and evaluate individual survival distributions. Journal of Machine Learning Research, 21(85):1–63, 2020.

[58] Shekoufeh Gorgi Zadeh and Matthias Schmid. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence, 43(9): 3126–3137, 2020.

[59] Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based deep multiple instance learning. In International conference on machine learning, pages 2127–2136. Pmlr, 2018.

[60] Frank E Harrell, Robert M Calif, David B Pryor, Kerry L Lee, and Robert A Rosati. Evaluating the yield of medical tests. Jama, 247(18):2543–2546, 1982.

## A Appendix

## B Mutation Profiles of Sequenced Patients

## B.1 Targeted sequencing data availability

Among the 302 patients with whole-slide image features, targeted sequencing results were available for 154 (51.0%) and unavailable for 148 (49.0%). The availability matrix shows that all 102 displayed genes were assessed in each sequenced patient. Missingness therefore occurred at the patient level, with no gene-level gaps among patients with panel results.

## B.2 Mutation profiles of sequenced patients

The mutation matrix summarizes pathogenic or likely pathogenic variants across 102 genes in 154 patients with whole-slide image features. KRAS and TP53 were the most frequently mutated genes, afecting 133 (86.4%) and 78 (50.6%) patients, respectively. Mutations in other panel genes were uncommon. The number of mutated genes per patient ranged from zero to four.

![](images/751632b38d7a7cce4655ae1947f864030549742e2e6be6443e40b019ee48ee94.jpg)  
Figure 1: Availability of targeted sequencing data among 302 patients with whole-slide image features.

![](images/a51dec6c92850f1092ff3c85c9a18f16695d6133315bd97e5cb88276b88e1911.jpg)  
Figure 2: Mutational landscape of the sequenced patients of the WSI analysis cohort (n = 154).