# AIRRI: Towards a Human-Readable, Machine-Resistant Reading Interface

Emory Salaun

Syed Ayon

Kaleel Mahmood<sup>\*</sup> Shaun Wallace 米

Department of Computer Science and Statistics University of Rhode Island Kingston, Rhode Island, USA

Corresponding author: emory salaun@uri.edu

## Abstract

The rapid advancement of image-to-text extraction systems and large language models (LLMs) poses new challengesfor educational assessment by making assignment materials easy to transcribe and solve with minimal human effort. To address this problem, we present AIRRI (Artificial Intelligence Resilient Reading Interface), a framework for generating machineresistant educational content, a dataset of higher-education assignments for evaluation, and an initial evaluation of the human readability of AIRRI machine-resistant text designs. AIRRI employs two categories ofmachine-resistance techniques. First, AIRRI applies image transformations including character spacing, word spacing, line spacing, character opacity variation, stripe overlays, and image noise. Second, AIRRI applies decision-based adversarial attacks: RayS, SurFree, and ADBA. We evaluate AIRRI against three image-to-text systems: Tesseract, EasyOCR, and the transformer-based GOT-OCR2 model. Our results show that AIRRI substantially degrades automated text extraction performance. AIRRI increased character error ratesfor EasyOCR (6.9% to 55.9%), Tesseract (2.3% to 100.0%), and GOT-OCR2 (30.7% to 48.1%). Word error rates increased for EasyOCR (24.9% to 96.7%) and Tesseract (10.7% to 100.0%). Among the evaluated decision-based attacks, ADBA consistently produced the largest degradation in image-to-text performance, while RayS and SurFree produced more moderate reductions. EasyOCR and Tesseract were generally more susceptible to both image-based and adversarial perturbations than the transformer-based GOT-OCR2 model. Our findings suggest that AIRRI can significantly reduce the reliability of image-to-text-to-LLM workflows, increasing the effort required for AI-assisted assignment completion. Anonymous code is available here.

## 1. Introduction

Digital text is designed to maximize accessibility [30] and machine readability [36], enabling copying, searching, indexing, and automated processing [3]. While these properties provide significant usability benefits [37], they also make content easily subject to automated extraction. Advances in image-to-text and language-processing systems have made it increasingly possible to rapidly convert on-screen text into machine-readable representations and process it automatically [1, 8, 26, 42]. As artificial intelligence systems continue to improve, this raises an important question: can educational material remain readable and usable for human learners while becoming more resistant to automated extraction and machine-assisted completion?

Modern LLMs can generate human-like responses, summarize content, and answer questions [6, 34]. Combined with image-to-text extraction systems, they enable educational material to be rapidly converted into machine-readable text and processed through automated workflows. Thus, assignment content can be extracted from screenshots and given to AI systems for question answering, summarization, or assignment completion with minimal effort [4]. This creates a growing challenge for educators seeking to promote direct student engagement while limiting low-effort AI-assisted workflows.

Existing responses to AI-assisted academic misuse frequently focus on behavioral policies[41], plagiarism detection, browser restrictions, or later identification of AI-generated work. While these approaches may discourage misuse or identify suspicious outputs after submission, they often do not address how easily students can extract and process educational conten through image-to-text and LLM-based workflows. Furthermore, detection-based approaches may become less reliable[28] as generative models continue to improve and student usage patterns evolve. This motivates exploring preventive approaches that increase the difficulty of automated extraction while preserving the usability of instructional materials.

To address these challenges, we propose AIRRI (Artificial Intelligence Resilient Reading Interface). A system designed to increase resistance to automated extraction while preserving the readability of educational material for human users. AIRRI transforms assignment and instructional content into a machineresistant format that impedes image-to-text recognition and downstream automated processing. This increases the difficulty of submitting extracted material directly to LLMs. AIRRI could enable educators to distribute content that remains available to human readers while making low-effort image-to-text and LLMassisted workflows harder. Rather than eliminating misuse entirely, AIRRI increases the human effort required for such behavior. A student supplying assignment content to an LLM would need to manually transcribe it rather than rely on automated extraction methods. We investigate whether controlled text perturbations can reduce the effectiveness of image-to-text and downstream automated processing while using levels of perturbation to preserve some readability. We evaluate AIRRI across multiple image-to-text systems using real-world educational assignments and quantitative performance metrics to examine the trade-off between machine resistance and the severity of text perturbation.

## 2. AIRRI Pipeline

AIRRI renders educational content as images and applies a combination of typographic, image-level, and adversarial perturbations. We cover AIRRI’s threat model, pipeline architecture, rendering, defense mechanisms, and evaluation.

## 2.1. Threat Model

AIRRI assumes an adversary who attempts to process educational material through an image-to-text extraction pipeline followed by an LLM with minimal manual effort. The adversary is assumed to have access to modern automated text-extraction systems and may capture assignment content via screenshots or other extraction methods. Unlike traditional adversarial machine learning [12], AIRRI does not seek to induce specific misclassifications. Instead, it seeks to degrade automated text extraction while maintaining some usability for human readers.

An attack succeeds when an image-to-text system extracts enough information to reconstruct the original text with little or no manual transcription. A defense succeeds when it substantially degrades extraction quality while preserving human readability. AIRRI does not seek to make extraction impossible; instead, it increases the effort required for automated processing and reduces the effectiveness of AI-assisted extraction.

## 2.2. Pipeline Overview

Figure 1 shows an overview of the AIRRI pipeline. Given an input assignment or instructional document, AIRRI produces a protected image intended to disrupt automated image-to-text extraction. AIRRI implements three categories of perturbations: typographic, image-level, and adversarial. The effectiveness of these perturbations is evaluated using EasyOCR[17], Tesseract[29], and GOT-OCR2[44].

The AIRRI pipeline has five stages: (1) Input Acquisition, (2) Text Rendering, (3) Perturbation Application, (4) Automated Extraction, and (5) Evaluation. During Stage 1, instructor-provided educational material is the source text. Stage 2 renders the source material to a canvas-based image using fixed rendering parameters. Stage 3 applies the selected perturbation techniques to generate a protected image. Stage 4 evaluates extraction resistance by processing both protected and unprotected images through three image-to-text systems: EasyOCR, Tesseract, and GOT-OCR2, representing traditional and transformer-based extraction architectures. Stage 5 evaluates extraction quality and downstream assignment completion using Character Error Rate (CER), Word Error Rate (WER) [18, 31], and LLM Error Rate (LER).

## 2.3. Text Generation Pipeline

AIRRI transforms instructor-provided educational material into a machine-resistant image representation using an HTML canvas rendering pipeline. Input text is rendered using the font Roboto at 16 px with configurable character, word, and line spacing. We selected Roboto because it is the most viewed font on the web per Google Font Analytics [13, 37] and is often the sole font studied in related readability research [20, 21, 23, 25]. Text is rendered with automatic line wrapping to preserve paragraph structure and reading order before exporting as a PNG image. For experimental consistency, all renderings were generated in a web browser at a fixed resolution of 1920×1080. Clean and protected images were produced from identical source text and rendering settings, differing only through the application of AIRRI perturbations.

## 2.4. Defense Mechanisms

AIRRI employs three categories of defense mechanisms: typographic perturbations, image-level perturbations, and adversarial perturbations. Typographic perturbations modify the appearance and layout of rendered text, image-level perturbations alter the rendered image, and adversarial perturbations introduce modeltargeted modifications intended to reduce extraction accuracy.

## 2.4.1. Typographic Perturbations

AIRRI applies four configurable typographic perturbation techniques to alter text appearance and reduce image-to-text extraction quality. Our text designs are based on prior readability literature to try and preserve human readability [24, 38]. Prior research shows that typographic attributes such as font selection and character/word spacing can influence reading performance and readability [7, 32, 33]. Unless otherwise specified, each perturbation is applied uniformly throughout the rendered document. Figure 2 shows examples of the image-level perturbations used in AIRRI at increasing intensity levels.

During rendering, spacing (character, word, and line) are CSS parameters that map distances to pixels. Thus, changing the appearance and paragraph layout of text [3]. Character spacing is the the distance between adjacent characters [7, 33].

![](images/a833c547a66e898d526b854b7c3a3fde2069be26db5689aaaf73ab43ce3d9e8c.jpg)  
Figure 1. The AIRRI pipeline transforms instructor-provided assignments into machine-resistant rendered text while preserving human readability. Image-to-text and LLM systems can extract unprotected text, whereas AIRRI-protected text requires manual transcription.

Word spacing is the space between adjacent words [3, 38]. Line spacing is the the distance between successive lines of text [27].

Character opacity assigns a unique opacity value to each rendered character [14] according to

$$
\alpha _ { i } = 1 - u _ { i } r\tag{1}
$$

where $\alpha _ { i }$ is the opacity of character i, $r \in [ 0 , 1 ]$ is the opacity jitter parameter, and $u _ { i }$ is sampled uniformly from [0,1]. Increasing r increases variation in character opacity.

## 2.4.2. Image-Level Perturbations

Following text rendering, AIRRI applies configurable imagelevel perturbations directly to the rendered canvas to further reduce machine readability. AIRRI implements two image-level perturbation techniques: stripe overlays and image noise.

Stripe Overlays create periodic linear patterns across the rendered image. AIRRI generates a repeating stripe pattern with fixed stripe and gap widths using a configurable orientation angle θ and opacity parameter $\alpha _ { s }$ . AIRRI rotates the stripe pattern by angle θ and composites it onto the rendered image.

For Image Noise, AIRRI applies noise by superimposing randomly positioned circular artifacts to the rendered image. A user-defined parameter $\alpha _ { n } \in [ 0 , 1 ]$ controls both artifact opacity and noise density. Given an image of width $W$ and height H, the number of generated artifacts is

<table><tr><td>Apple</td><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Level 4</td></tr><tr><td>Jitter</td><td>Apple</td><td>Apple</td><td>Apple</td><td>Apple</td></tr><tr><td>Noise</td><td>Apple</td><td>Apple</td><td>Apple</td><td>Apple.</td></tr><tr><td>Stripes</td><td>Apple</td><td>Apple</td><td>Apple</td><td>Appl</td></tr></table>

Figure 2. Examples of AIRRI image-level perturbations; top left is the clean image. Intensity increases from Level 1 to Level 4.

$$
N = \left\lfloor { \frac { W H } { 6 2 5 } } \alpha _ { n } \right\rfloor\tag{2}
$$

where N is the total number of noise artifacts. Increasing $\alpha _ { n }$ increases both artifact density and visibility, resulting in greater disruption of automated text extraction [15].

## 2.4.3. Adversarial Perturbations

We evaluate three black-box adversarial attacks against image-to-text systems. Each attack operates under a hard-label threat model: the attacker observes only the final discrete output of the image-to-text engine (i.e., the recognized text string), not internal confidence scores or gradients.

RayS: Ray Searching Attack [11] is a hard-label $\ell _ { \infty }$ attack that searches for adversarial examples along ray directions in the input space. It maintains a discrete sign vector $\mathbf { s } _ { t } \in \{ - 1 , + 1 \} ^ { d }$ and a boundary distance $d _ { t } .$ At iteration t, a candidate sign vector ˆs is generated by flipping signs within hierarchically determined blocks. The adversarial candidate is constructed along the corresponding ray:

$$
\mathbf { x } _ { \mathrm { a d v } } = \mathrm { c l i p } \left( \mathbf { x } + d \cdot { \frac { \hat { \mathbf { s } } } { \lVert { \hat { \mathbf { s } } } \rVert } } , 0 , 1 \right)\tag{3}
$$

A binary search determines the minimal distance $d ^ { * }$ at which the classifier decision F changes:

$$
d ^ { * } = \operatorname* { m i n } \left\{ d \geq 0 \left| F \left( \mathrm { c l i p } \left( \mathbf { x } + d \cdot \frac { \hat { \mathbf { s } } } { \left\| \hat { \mathbf { s } } \right\| } , 0 , 1 \right) \right) \neq y \right. \right\}\tag{4}
$$

I $\therefore d ^ { * } < d _ { t }$ , the direction and distance are updated $( \mathbf { s } _ { t + 1 }  \hat { \mathbf { s } } ,$ $d _ { t + 1 }  d ^ { * } )$ . When this formulation finds a valid perturbation in the epsilon budget, it halts and bypasses gradient estimation.

SurFree [22] is a geometrical decision-based $\ell _ { 2 }$ attack that minimizes perturbation distance by exploring boundaries along orthogonal circular paths. Given a valid adversarial point $\mathbf { x } _ { \mathrm { a d v } } ,$ candidate perturbations are explored on a 2D subspace defined by the unit direction to the current adversarial point ${ \bf d } _ { 1 }$ and a random orthogonal direction $\mathbf { d } _ { 2 } { \mathrm { : } }$

$$
\mathbf { x } _ { \theta } { = } \mathbf { x } { + } r \mathrm { c o s } \theta { \cdot } \mathbf { d } _ { 1 } { + } r \mathrm { s i n } \theta { \cdot } \mathbf { d } _ { 2 }\tag{5}
$$

where $r = \lVert \mathbf { x } _ { \mathrm { a d v } } - \mathbf { x } \rVert _ { 2 } .$ , and ${ \bf d } _ { 2 }$ is sampled from low-frequency DCT components and orthogonalized via Gram-Schmidt. The maximum search angle $\theta _ { \mathrm { m a x } }$ is dynamically adjusted based on step success using a factor $\rho \mathrm { : }$

$$
\theta _ { \mathrm { m a x } } ^ { ( t + 1 ) } = \left\{ \begin{array} { l l } { \theta _ { \mathrm { m a x } } ^ { ( t ) } / \rho } & { \mathrm { i f ~ s t e p ~ s u c c e s s f u l , } } \\ { \theta _ { \mathrm { m a x } } ^ { ( t ) } { \cdot } \rho } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{6}
$$

ADBA [9] is a query-efficient decision-based attack that avoids estimating exact decision boundaries. Rather than computing the boundary distance $g ( \mathbf { d } )$ for candidate directions, it compares directions with an Approximated Decision Boundary (ADB). To compare two perturbation directions d<sub>1</sub> and $\mathbf { d } _ { 2 } ,$ ADBA queries the target model at a scale defined by the ADB. If ${ \bf d } _ { 2 }$ successfully changes the label while ${ \bf d } _ { 1 }$ fails, then:

$$
g ( \mathbf { d } _ { 2 } ) { \leq } \mathrm { A D B } { < } g ( \mathbf { d } _ { 1 } )\tag{7}
$$

Thus, $\mathbf { d } _ { 2 }$ requires a smaller perturbation and is selected. ADBA continuously optimizes the perturbation size, which can exhaust the query budget without early stopping.

## 2.5. Evaluation Pipeline

The AIRRI evaluation pipeline measures automated text extraction performance by comparing extracted text against ground-truth source material. The evaluation process has two stages: automated extraction and extraction scoring.

## 2.5.1. Automated Extraction

AIRRI evaluates machine resistance using three image-to-text systems: Tesseract, EasyOCR, and GOT-OCR2. We selected these systems to represent both traditional image-to-text architectures and modern transformer-based vision-language models. For each rendered image, all three extraction systems generate an independent text transcription. We evaluate Tesseract using the English language model, page segmentation mode 6, and OCR engine mode 3. We evaluate EasyOCR using its English language model with paragraph-level text aggregation enabled. GOT-OCR2 is evaluated using the publicly available stepfun-ai/GOT-OCR-2.0-hf checkpoint.

## 2.5.2. Extraction Scoring

AIRRI compares the extracted text from each image-to-text system against its ground-truth assignment text provided in the dataset manifest. AIRRI evaluates extraction performance using Character Error Rate (CER) and Word Error Rate (WER), two standard text-recognition metrics. For character-level evaluation, we normalize both the extracted and ground-truth text by converting all characters to lowercase and removing whitespace. Character Error Rate (CER) is computed according to Equation 8:

$$
C E R { = } \frac { { D } _ { L } ( T _ { g t } , T _ { p r e d } ) } { | T _ { g t } | }\tag{8}
$$

where $D _ { L } ( \cdot )$ denotes the Levenshtein edit distance [19], $T _ { g t }$ is the ground-truth text, $T _ { p r e d }$ is the extracted text, and $| T _ { g t } |$ is the number of characters in the ground-truth text.

For word-level evaluation, we convert both texts to lowercase and tokenize them into whitespace-delimited word sequences. Word Error Rate (WER) is computed according to Equation 9:

$$
W E R { = } \frac { D _ { L } ( W _ { g t } , W _ { p r e d } ) } { | W _ { g t } | }\tag{9}
$$

where $W _ { g t }$ and $W _ { p r e d }$ denote the ground-truth and extracted word sequences, respectively, and $| W _ { g t } |$ is the total number of words in the reference text. Equations 8 and 9 show that both metrics measure normalized edit distance between the extracted and reference text. Lower CER and WER values indicate more successful text extraction, while higher values indicate stronger resistance to automated extraction. AIRRI reports both metrics for each image-to-text system and perturbation configuration.

## 2.5.3. LLM Performance Evaluation

While CER and WER quantify extraction quality, AIRRI seeks to disrupt downstream assignment completion using LLMs. To evaluate this effect, we supply an LLM with text extracted by each image-to-text system and use it to complete the assignment. We evaluated downstream LLM performance using ChatGPT through the ChatGPT web interface in Instant mode in July 2026. We conducted each evaluation in a new Temporary Chat to prevent prior conversation context from influencing the response. The raw OCR output was provided without manual correction or preprocessing using the prompt, “Complete the following assignment to the best of your ability using only the information provided. Provide only your final answer, without additional commentary or explanation.” A single inference was performed for each OCR extraction. Because the ChatGPT interface was used rather than the OpenAI API, inference parameters such as temperature, top-p, and random seed were not user-configurable and therefore were not explicitly controlled.

We score responses with a three-point scale: 0 (incorrect), 1 (partially correct), and 2 (correct). To aggregate results across assignments, AIRRI computes the LLM Performance Score (LPS):

$$
L P S { = } \frac { \sum _ { i = 1 } ^ { N } s _ { i } } { 2 N }\tag{10}
$$

where $s _ { i } \in \{ 0 , 1 , 2 \}$ is the score assigned to response i and N is the number of evaluated responses. LPS ranges from 0 to 1; higher values indicate more successful assignment completion. For consistency with CER and WER, AIRRI also reports the

Table 1. Summary statistics for the educational assignment dataset for AIRRI’s evaluation. It has nine instructor-provided assignments from English and History courses with varying document lengths.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Assignments</td><td>9</td></tr><tr><td>English Assignments</td><td>7</td></tr><tr><td>History Assignments</td><td>2</td></tr><tr><td>Total Words</td><td>4,364</td></tr><tr><td>Total Characters</td><td>25,282</td></tr><tr><td>Average Words</td><td>485</td></tr><tr><td>Median Words</td><td>382</td></tr><tr><td>Minimum Words</td><td>220</td></tr><tr><td>Maximum Words</td><td>1,122</td></tr></table>

LLM Error Rate $( L E R ) ,$ where higher LER values indicate greater resistance to downstream assignment completion:

$$
L E R { = } 1 { - } L P S\tag{11}
$$

## 2.6. Dataset for Evaluation

Our new evaluation dataset has nine higher-education (HE) assignments collected from instructors at two universities. Five assignments were from a University of Connecticut English instructor and two from a University of Rhode Island History instructor. One English assignment has three essay topics with distinct assignment texts; thus, we created three separate evaluation instances. The final dataset contains seven English assignments and two History assignments. For evaluation, each assignment was rendered as clean and AIRRI-protected images, and CER and WER were computed. Table 1 summarizes the introduced dataset. Evaluation was also performed on opensource datasets of 8th and 12th grade-level reading passages (8-and-12) provided by The Readability Consortium [37].

## 3. Results

We evaluate AIRRI’s effectiveness in degrading automated text extraction. We first compare extraction performance on clean and AIRRI-protected documents, then analyze the effects of individual adversarial attacks and extraction systems using CER and WER. Unless otherwise specified, the AIRRI pipeline generated AIRRI-protected documents, including all typographic and image-level perturbations applied simultaneously.

## 3.1. Typographic + Image-Level Perturbations

Table 2 summarizes extraction performance for clean and AIRRI-protected documents across all image-to-text systems on both the HE Assignment and 8-and-12 datasets. Results are reported using Character Error Rate (CER) and Word Error Rate (WER), higher values indicate greater extraction difficulty.

AIRRI increased extraction error rates for EasyOCR and Tesseract. EasyOCR CER increased from 6.9% to 55.9%, while

WER increased from 24.9% to 96.7%. Tesseract showed the largest degradation: CER increased from 2.3% to 100% and WER from 10.7% to 100%. This indicates complete extraction failure on all protected documents in the evaluation set.

GOT-OCR2 was less affected by the applied perturbations than some of the other models. CER increased from 30.7% to 48.1%, while WER decreased slightly from 63.3% to 61.9%. Examination of individual assignments showed considerable variation in performance: some protected documents produced substantially higher error rates than their clean counterparts, whereas others produced lower error rates.

On the 8-and-12 dataset, AIRRI also degraded extraction across all systems. EasyOCR CER increases from 7.3% to 81.1%. GOT-OCR2 showed the largest increase: CER rising from 0.8% to 90.4% and WER from 6.9% to 91.5%. Tesseract CER increased from 1.0% to 38.8%, less degradation than on the HE Assignment, though WER still reached 99.9%.

## 3.1.1. Individual Image Transformation Performance

To evaluate the effectiveness of individual image transformations, each transformation was applied independently at five intensity levels, including a clean baseline (0%) and perturbation strengths of 25%, 50%, 75%, and 100%. The 100% level corresponded either to the maximum supported perturbation value or an empirically selected upper bound intended to avoid excessive visual degradation, whichever was lower. Although stronger perturbations generally led to higher OCR error rates, excessively strong transformations may also reduce human readability. We measured OCR performance at each intensity level and compared it with the clean baseline. Table 3 reports the mean increase in image-to-text error rate across all assignments, perturbation intensity levels, and extraction systems.

Among the evaluated image transformations, stripe overlays produced the largest degradation in automated text extraction performance, increasing CER by an average of 35.9% and WER by 38.3%. In comparison, opacity jitter and image noise resulted in substantially smaller increases in error rates. Although performance varied across assignments and perturbation intensities, stripe overlays consistently produced the largest increases in both CER and WER.

## 3.2. AIRRI Degrades LLM Performance

While CER and WER measure OCR extraction quality, the ultimate objective of AIRRI is to disrupt downstream LLMassisted assignment completion. Table 4 presents the resulting LLM Performance Score (LPS) and LLM Error Rate (LER) for clean and AIRRI-protected documents. Across all image-to-text systems, AIRRI substantially increased downstream assignment completion failure rates. The aggregate LER increased from 7.4% for clean documents to 55.6% for protected documents, indicating a substantial reduction in the ability of LLMs to successfully complete assignments using extracted text.

Tesseract exhibited the largest increase in LER, rising from 5.6% on clean documents to 100.0% on protected documents, consistent with the complete extraction failure observed in the image-to-text evaluation. EasyOCR and GOT-OCR2 demonstrated greater resilience, with LER increasing from 5.6% to 22.2% and from 11.1% to 44.4%. Overall, AIRRI consistently increased failure rates across all evaluated extraction systems, with the strongest effect observed for Tesseract.

Table 2. Average extraction error rates for clean and AIRRI-protected documents. CER is Character Error Rate, and WER is Word Error Rate.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Metric</td><td colspan="3">HE Assignment</td><td colspan="3">8-and-12</td></tr><tr><td>Clean</td><td>Prot.</td><td> $\Delta$ </td><td>Clean</td><td>Prot.</td><td> $\Delta$ </td></tr><tr><td rowspan="2">EasyOCR</td><td>CER (%)</td><td>6.94</td><td>55.95</td><td>+49.01</td><td>7.34</td><td>81.10</td><td>+73.76</td></tr><tr><td>WER (%)</td><td>24.86</td><td>96.67</td><td>+71.81</td><td>24.84</td><td>96.96</td><td>+72.12</td></tr><tr><td rowspan="2">Tesseract</td><td>CER (%)</td><td>2.26</td><td>100.00</td><td>+97.74</td><td>0.97</td><td>38.76</td><td>+37.79</td></tr><tr><td>WER (%)</td><td>10.65</td><td>100.00</td><td>+89.35</td><td>2.64</td><td>99.93</td><td>+97.29</td></tr><tr><td rowspan="2">GOT-OCR2</td><td>CER (%)</td><td>30.70</td><td>48.09</td><td>+17.39</td><td>0.83</td><td>90.42</td><td>+89.59</td></tr><tr><td>WER (%)</td><td>63.26</td><td>61.90</td><td>-1.36</td><td>6.91</td><td>91.54</td><td>+84.63</td></tr></table>

Table 3. Mean increase in image-to-text error rate (%) relative to clean images, averaged across all assignments, perturbation intensity levels, and extraction systems.
<table><tr><td>Perturbation</td><td>CER∆(%)</td><td>WER ∆(%)</td></tr><tr><td>Jitter</td><td>6.27</td><td>17.04</td></tr><tr><td>Noise</td><td>3.96</td><td>17.17</td></tr><tr><td>Stripes</td><td>35.90</td><td>38.31</td></tr></table>

Table 4. LLM Performance Score (LPS) and LLM Error Rate (LER) for clean and AIRRI-protected documents. Higher LER values indicate stronger resistance to downstream assignment completion.
<table><tr><td>Protection</td><td>Model</td><td>LPS</td><td>LER</td></tr><tr><td>Clean</td><td>EasyOCR</td><td>94.4</td><td>5.6</td></tr><tr><td>Clean</td><td>Tesseract</td><td>94.4</td><td>5.6</td></tr><tr><td>Clean</td><td>GOT-OCR2</td><td>88.9</td><td>11.1</td></tr><tr><td>Clean</td><td>All</td><td>92.6</td><td>7.4</td></tr><tr><td>Protected</td><td>EasyOCR</td><td>77.8</td><td>22.2</td></tr><tr><td>Protected</td><td>Tesseract</td><td>0.0</td><td>100.0</td></tr><tr><td>Protected</td><td>GOT-OCR2</td><td>55.6</td><td>44.4</td></tr><tr><td>Protected</td><td>All</td><td>44.4</td><td>55.6</td></tr></table>

## 3.3. Localized Degradation, Global Readability

We evaluate three black-box adversarial attacks against three OCR engines where ADBA and Rays operate under an $L _ { \infty }$ perturbation budget with $\varepsilon \in \{ 8 / 2 5 5 , 1 6 / 2 5 5 , 3 2 / 2 5 5 , 6 4 / 2 5 5 \}$ while SurFree uses an $L _ { 2 }$ distance threshold. Each attack targets only selected lines; the rest of the document remains unmodified. We report CER and WER at two evaluation scopes:full composite, where OCR runs on the entire stitched document, and target region, where OCR runs only on the attacked line crops. Table 5 shows the full-composite results. Because only a small portion of the document is perturbed, CER and WER remain low across all conditions. GOT-OCR [44] achieves near-zero CER at $\varepsilon \le 3 2 / 2 5 5$ , while EasyOCR and Tesseract are around 6–8% CER regardless of the attack. Even at the largest budget $\left( \varepsilon = 6 4 / 2 5 5 \right)$ , full-composite CER does not exceed 8.44% for any engine, showing the targeted perturbation strategy helps preserve some document readability.

Table 6 isolates the attacked lines. Here, the picture changes sharply. On the HE Assignment dataset, ADBA at ε=64/255 raises CER to 49.83% (EasyOCR), 57.93% (GOT-OCR), and 48.55% (Tesseract), with WER exceeding 80% across all three engines. The 8-and-12 dataset shows a similar pattern: ADBA at $\varepsilon = 6 4 / 2 5 5$ raises CER to 44.81% (EasyOCR), 53.40% (GOT-OCR), and 43.02% (Tesseract), with WER reaching 76.14%, 78.11%, and 70.79%, respectively. Rays also degrades target-region accuracy as ε grows, but its effect is weaker: at ε=64/255 on the HE Assignment dataset, CER reaches only 11.44% (EasyOCR) and 25.72% (GOT-OCR), with comparable values of 9.28% and 23.40% on the 8-and-12 dataset. At lower budgets $( \varepsilon \leq 1 6 / 2 5 5 )$ , both attacks produce minimal changes for EasyOCR and Tesseract on both datasets.

## 3.4. Interviews with Human Readability Experts

We conducted semi-structured interviews with four domain experts in typography, design, and cognitive psychology. We showed each human readability expert our machine-resistant designs and their perturbation levels. The interviews focused on the designs’ potential effects on reading speed and comprehension across different reading modes, such as interlude reading or skimming. The final part of the interviews covered how the different designs affect neurodivergent readers and future design ideas we should explore. We performed thematic coding to analyze qualitative results from the semi-structured interviews. This effort revealed three codes for creating human-readable perturbations of machine-resistant text: preserving letterform structure, line tracking, and adequate foreground-background contrast.

Notably, all experts found the vertical stripes readable. This shows that our most machine-resistant stripe design was the most human-readable, in our experts’ view. Overall, this highlights a primary theme for future work: finding designs that minimally hurt human perception while maximizing degradation of machine perception in image-to-text. For example, all experts found the diagonal stripes unreadable, especially for skimming. P1 remarked they “break up the letterforms”. Thus, this would not be a recommended human-readable design. All experts also found the 50% opacity jitter acceptable, with P2 likening it to a “printer running out ink”.

Table 5. Full-composite CER/WER for adversarial perturbations.
<table><tr><td rowspan="2">Engine</td><td rowspan="2">Attack</td><td rowspan="2">ε</td><td colspan="2">HE Assignment</td><td colspan="2">8-and-12</td></tr><tr><td>CER (%)</td><td>WER (%)</td><td>CER (%)</td><td>WER (%)</td></tr><tr><td rowspan="8">EasyOCR</td><td rowspan="5">ADBA</td><td>8/255</td><td>6.38</td><td>28.54</td><td>5.48</td><td>27.05</td></tr><tr><td>16/255</td><td>5.82</td><td>28.23</td><td>5.05</td><td>25.62</td></tr><tr><td>32/255</td><td>5.65</td><td>28.33</td><td>4.80</td><td>23.49</td></tr><tr><td>64/255</td><td>8.07</td><td>34.00</td><td>6.46</td><td>28.80</td></tr><tr><td>8/255</td><td>6.78</td><td>29.31</td><td>6.99</td><td>29.77</td></tr><tr><td rowspan="4">Rays</td><td>16/255</td><td>6.87</td><td>29.48</td><td>7.10</td><td>30.06</td></tr><tr><td>32/255</td><td>7.17</td><td>30.73</td><td>7.05</td><td>29.68</td></tr><tr><td>64/255</td><td>8.44</td><td>35.59</td><td>7.89</td><td>32.81</td></tr><tr><td>8/255</td><td>0.00</td><td>1.02</td><td>0.00</td><td>0.00</td></tr><tr><td rowspan="6">GOT-OCR</td><td rowspan="3">ADBA</td><td>16/255</td><td>0.00</td><td>0.59</td><td>0.00</td><td>0.00</td></tr><tr><td>32/255</td><td>0.00</td><td>0.73</td><td>0.00</td><td>0.00</td></tr><tr><td>64/255</td><td>1.69</td><td>8.58</td><td>0.00</td><td>1.44</td></tr><tr><td rowspan="4">Rays</td><td>8/255</td><td>0.31</td><td>2.10</td><td>0.49</td><td>2.78</td></tr><tr><td>16/255</td><td>0.41</td><td>2.32</td><td>0.61</td><td>3.18</td></tr><tr><td>32/255</td><td>0.73</td><td>4.05</td><td>0.55</td><td>2.66</td></tr><tr><td>64/255</td><td>2.09</td><td>10.79</td><td>1.45</td><td>7.00</td></tr><tr><td rowspan="8">Tesseract</td><td rowspan="4">ADBA</td><td>8/255</td><td>6.47</td><td>28.47</td><td>4.83</td><td>28.44</td></tr><tr><td>16/255</td><td>6.02</td><td>27.82</td><td>4.40</td><td>27.05</td></tr><tr><td>32/255</td><td>5.72</td><td>27.02</td><td>4.15</td><td>24.95</td></tr><tr><td>64/255</td><td>7.92</td><td>32.83</td><td>5.82</td><td>30.16</td></tr><tr><td></td><td>8/255</td><td>6.82</td><td>29.05</td><td>5.91</td><td>29.54</td></tr><tr><td rowspan="4">Rays</td><td>16/255</td><td>6.98</td><td>29.12</td><td>6.03</td><td>30.61</td></tr><tr><td>32/255</td><td>7.18</td><td>30.28</td><td>5.97</td><td>29.18</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>64/255</td><td>8.26</td><td>34.51</td><td>8.00</td><td>36.65</td></tr></table>

Future work ideas included experimenting with typefaces that affect vertical text orientation (i.e., mono-type fonts), text over complex backgrounds, varying font types within words (P3 “ransom notes”), and applying opacity jitter only to characters within a word. Notably, all experts urged future work to explore new design ideas as AI technologies evolve and to bridge gaps in optimizing machine resistance while minimally degrading the human readability of AIRRI’s designs.

## 4. Discussion

## 4.1. Adversarial Perturbation Analysis

ADBA was the strongest attack in our experiments. At $\varepsilon = 6 4 / 2 5 5 .$ , it pushed target-region WER above 70% on both datasets and across all three engines. RayS also degraded recognition with ε, but its peak target-region CER fell short of ADBA’s. On the HE Assignment dataset, Rays reached 25.72% CER on GOT-OCR and 11.44% on EasyOCR; on the 8-and-12 dataset, the values were 23.40% and 9.28%. This gap likely stems from their search strategies: ADBA uses adaptive boundary estimation with offspring sampling to concentrate perturbation energy near the decision boundary [9], while Rays relies on random directional search that converges more slowly when the perturbation space is large [11]. This performance gap aligns with the findings in the original ADBA study [9], where ADBA consistently outperformed RayS on ImageNet. Our results indicate that this comparative advantage extends to document images and OCR models, and holds across both datasets. SurFree, the only $L _ { 2 } .$ -bounded attack we tested, failed to produce usable perturbations because the $L _ { 2 }$ distance between the original and perturbed images exceeded the allowed threshold at all budgets, and is thus omitted from the tables.

Table 6. Target-region CER/WER for adversarial perturbations.
<table><tr><td rowspan="2">Engine</td><td rowspan="2">Attack</td><td rowspan="2">ε</td><td colspan="2">HE Assignment</td><td colspan="2">8-and-12</td></tr><tr><td>CER (%)</td><td>WER (%)</td><td>CER (%)</td><td>WER (%)</td></tr><tr><td rowspan="8">EasyOCR</td><td rowspan="5">ADBA</td><td>8/255</td><td>4.36</td><td>23.96</td><td>3.95</td><td>21.61</td></tr><tr><td>16/255</td><td>4.34</td><td>23.99</td><td>3.81</td><td>20.81</td></tr><tr><td>32/255</td><td>6.31</td><td>29.87</td><td>4.85</td><td>24.23</td></tr><tr><td>64/255</td><td>49.83</td><td>80.07</td><td>44.81</td><td>76.14</td></tr><tr><td>8/255 Rays</td><td>4.58</td><td>25.15</td><td>4.42</td><td>23.29</td></tr><tr><td>16/255</td><td>4.93</td><td>26.22</td><td>4.59</td><td>23.36</td></tr><tr><td></td><td>32/255</td><td>5.83</td><td>29.31</td><td>5.34</td><td>25.30</td></tr><tr><td rowspan="5">ADBA</td><td>64/255</td><td>11.44</td><td>45.24</td><td>9.28</td><td>39.32</td></tr><tr><td>8/255</td><td>19.79</td><td>24.70</td><td>18.91</td><td>28.10</td></tr><tr><td>16/255</td><td>19.77</td><td>24.73</td><td>18.79</td><td>27.36</td></tr><tr><td>32/255</td><td>21.42</td><td>30.55</td><td>19.67</td><td>30.49</td></tr><tr><td>64/255</td><td>57.93</td><td>80.26</td><td>53.40</td><td>78.11</td></tr><tr><td rowspan="4">Rays</td><td>8/255</td><td>19.97</td><td>25.88</td><td>19.31</td><td>29.63</td></tr><tr><td>16/255</td><td>20.26</td><td>26.94</td><td>19.45</td><td>29.70</td></tr><tr><td>32/255</td><td>21.02</td><td>29.99</td><td>20.08</td><td>31.48</td></tr><tr><td>64/255</td><td>25.72</td><td>45.77</td><td>23.40</td><td>44.34</td></tr><tr><td rowspan="8">Tesseract</td><td rowspan="4">ADBA</td><td>8/255</td><td>4.65</td><td>25.08</td><td>6.19</td><td>26.30</td></tr><tr><td>16/255</td><td>4.40</td><td>23.47</td><td>6.06</td><td>25.55</td></tr><tr><td>32/255</td><td>6.71</td><td>31.92</td><td>7.07</td><td>28.76</td></tr><tr><td>64/255</td><td>48.55</td><td>80.29</td><td>43.02</td><td>70.79</td></tr><tr><td rowspan="4">Rays</td><td>8/255</td><td>4.99</td><td>25.80</td><td>4.44</td><td>23.80</td></tr><tr><td>16/255</td><td>5.27</td><td>26.91</td><td>4.57</td><td>23.40</td></tr><tr><td>32/255</td><td>5.95</td><td>28.82</td><td>5.37</td><td>23.99</td></tr><tr><td>64/255</td><td>11.33</td><td>44.48</td><td>9.51</td><td>38.92</td></tr></table>

A notable pattern in Table 6 is GOT-OCR’s high CER on target-region crops at low perturbation budgets (e.g., ε = 8/255): 19.79% on the HE Assignment dataset and 18.91% on the 8-and-12 dataset, compared to roughly 4–6% for EasyOCR and Tesseract on both datasets. GOT-OCR is a vision-language model designed to process full document pages [44], and it appears to struggle when given small, isolated line crops that lack the surrounding page context it was trained on. Despite this weaker performance on isolated crops, GOT-OCR’s full-composite CER remains near zero on both datasets, as shown in Table 5, which indicates the model leverages document-level context to compensate for localized noise. However, this contextual advantage disappears when the evaluation scope is narrowed to the attacked region alone.

## 4.2. Image-Level Perturbation Analysis

Across the evaluated dataset, AIRRI consistently increased extraction error rates for EasyOCR and Tesseract, with Tesseract exhibiting complete failure to extract on all protected documents. These findings suggest that the typographic and image-level perturbations employed by AIRRI effectively disrupt the assumptions underlying many traditional image-to-text pipelines [8, 42].

A notable observation is the difference between traditional image-to-text systems and the transformer-based GOT-OCR2 model [1, 44]. While AIRRI substantially degraded the performance of EasyOCR and Tesseract, GOT-OCR2 exhibited greater resistance and more variable behavior. Some assignments experienced increased extraction errors, while others showed improved performance relative to the clean baseline. Notably, its aggregate WER decreased slightly from 63.26% to 61.90%, despite CER increasing substantially from 30.70% to 48.09%. This suggests that AIRRI affects character- and word-level errors differently for GOT-OCR2. One possible explanation is that the perturbations alter segmentation or contextual decoding in ways that reduce some word-level errors while increasing character-level corruption; however, the underlying mechanism was not isolated in our experiments.

## 4.3. Individual Image-Level Perturbation Analysis

The individual perturbation experiments provide insight into which image-level transformations contribute most strongly to machine resistance. Among the evaluated techniques, stripe overlays consistently produced the largest increases in both CER and WER, substantially outperforming image noise and character opacity variation. These results suggest that structured visual modifications disrupt automated text extraction more effectively than localized pixel-level distortions [15, 26].

Character opacity variation exhibited a different pattern [14]. Across low and moderate intensity levels, opacity jitter produced relatively little degradation in extraction performance. We observed substantial increases in extraction error only at the highest tested intensity level, suggesting a threshold effect. However, the highest intensity setting also noticeably reduced human readability, reflecting the tradeoff between visual accessibility and text modification documented in readability research[3, 16, 27, 30, 43]. As a result, while extreme opacity variation can disrupt automated extraction, it may be less practical than stripe overlays, which achieved substantial degradation without requiring equally aggressive visual distortion.

Although stripe overlays were the most effective individual transformation, the complete AIRRI pipeline had the greatest degradation in extraction performance. It combines all typographic and image-level perturbations. Thus, AIRRI produced higher error rates than any single transformation alone. This result suggests that different perturbations likely interfere with different stages of the extraction process, and that their combined effects provide stronger protection than isolated modifications.

## 4.4. Impact on Downstream LLM Performance

While CER and WER quantify extraction quality, the LLM evaluation shows that extraction errors do not always translate directly into downstream task failure. Tesseract exhibited complete extraction failure on protected documents, resulting in an LER of 100%. EasyOCR and GOT-OCR2 frequently produced noisy or incomplete extractions that retained sufficient information to reconstruct partial assignments, resulting in lower downstream failure rates. Our findings suggest that extractionquality metrics alone may not fully capture the practical impact of extraction errors on modern AI-assisted workflows.

Future work could explore human-inspectable AI-assisted workflows [45]. Although LLMs can often infer missing information and compensate for extraction mistakes, AIRRI increased the aggregate LER from 7.4% to 55.6%, indicating a substantial reduction in successful assignment completion. This demonstrates that sufficiently degraded extractions can still prevent reconstruction of assignment content, suggesting that AIRRI remains effective even against downstream AI-assisted recovery.

## 4.5. Image-Level vs. Adversarial Perturbations

This section compares the effectiveness and computational characteristics of AIRRI’s image-level and adversarial perturbations. While both approaches aim to degrade automated text extraction, image-level perturbations apply across the entire rendered document, whereas adversarial perturbations introduce bounded pixel-level modifications within selected text regions.

## 4.5.1. Effectiveness Against Image-to-Text Systems

Overall, our results suggest that image-level perturbations provide the strongest protection in our setting. Among the evaluated adversarial methods, ADBA achieved the strongest extraction degradation, particularly at $\varepsilon = 6 4 / 2 5 5$ . However, the image-level perturbations generally produced larger and more consistent increases in extraction error across the evaluated image-to-text systems. A notable exception was GOT-OCR2, where ADBA produced higher target-region error rates than those observed under the image-level perturbations. ADBA’s strong performance against GOT-OCR2 suggests that adversarial approaches may offer advantages against more complex image-to-text systems, although additional evaluation of modern vision-language models is needed to confirm this.

## 4.5.2. Hardware and Deployment Considerations

Image-level perturbations are computationally inexpensive because they are applied directly during rendering. Let $n { = } W H$ denote the number of image pixels. Operations such as spacing adjustments, stripe overlays, opacity variation, and image noise require only a single pass over the image and therefore have time complexity $O ( n )$

In contrast, decision-based adversarial attacks require repeated queries to the target OCR system. Let q denote the number of queries. Methods such as RayS [11] and ADBA [9] therefore scale as approximately $O ( q n )$ , since each query evaluates a perturbed image containing n pixels. In our experiments, adversarial attacks often required thousands of queries, making them substantially more expensive than image-level perturbations.

Consequently, image-level and typographic perturbations provide the most practical and scalable approach for educational deployment. This distinction is particularly important for educational settings, where instructors may need to generate protected materials frequently and with minimal computational overhead. For this use case, it is impractical. Because adversarial attacks require substantially greater computation and generation time [5, 9–11], they may be better suited for protecting selected high-value content rather than routine assignment generation.

## 5. Conclusion

We present AIRRI, a framework for generating machineresistant educational content. AIRRI combines typographic perturbations, image-level perturbations, and decision-based adversarial attacks to reduce the effectiveness of automated text extraction systems while preserving access to the underlying instructional material. We evaluated AIRRI using a dataset of higher-education assignments. AIRRI substantially degraded the extraction performance for EasyOCR and Tesseract and produced more variable effects for the transformer-based GOT-OCR2 model. AIRRI also increased downstream assignment completion failure rates; thus, degraded extractions can help reduce the effectiveness of image-to-text-to-LLM workflows. Future work will evaluate AIRRI using formal human-readability studies [3, 35], in-the-wild studies with educators and designers [39, 40], larger educational datasets, datasets across domains like medical texts [2], and additional vision-language models to better understand the tradeoff between human usability and machine resistance.

## A. Perturbation Level Parameters

Table 7 reports the parameter values used for the four non-zero intensity levels of each AIRRI image-level perturbation. These levels correspond to the perturbation examples and experimental conditions evaluated in Figure 2.

Table 7. Parameter values corresponding to AIRRI perturbation levels. Level 0 represents the clean, unperturbed baseline.
<table><tr><td>Perturbation</td><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Level 4</td></tr><tr><td>Opacity Jitter</td><td>25</td><td>50</td><td>75</td><td>100</td></tr><tr><td>Image Noise</td><td>25</td><td>50</td><td>75</td><td>100</td></tr><tr><td>Stripe Overlay</td><td>10</td><td>20</td><td>40</td><td>60</td></tr></table>

## B. Full Pipeline Configuration

Table 8 reports the AIRRI configuration used for the full-pipeline evaluation. These parameters were applied simultaneously to generate the AIRRI-protected documents evaluated against the image-to-text systems.

Table 8. AIRRI configuration used for the full-pipeline evaluation.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Font Size</td><td>16 px</td></tr><tr><td>Character Spacing</td><td>3 px</td></tr><tr><td>Word Spacing</td><td>15px</td></tr><tr><td>Line Spacing</td><td>Default</td></tr><tr><td>Stripe Intensity</td><td>20%</td></tr><tr><td>Stripe Angle</td><td>25°</td></tr><tr><td>Opacity Jitter</td><td>0.6</td></tr><tr><td>Image Noise</td><td>0</td></tr></table>

## References

[1] Ali Afkari-Fahandari, Elham Shabaninia, Fatemeh Asadi-Zeydabadi, and Hossein Nezamabadi-Pour. A

comprehensive survey of transformers in text recognition: Techniques, challenges, and future directions. ACM Computing Surveys, 58(5):1–42, 2025. 1, 8

[2] Rachel V Ball, Dave B Miller, Shaun Wallace, Kathlyn Camargo Macias, Mahmoud Ibrahim, Ernesto Robalino Gonzaga, Olga Karasik, Dekai R Rohlsen-Neal, Sarah Barrientos, Edward A Ross, et al. Optimizing electronic health records through readability. In Proceedings of the International Symposium on Human Factors and Ergonomics in Health Care, volume 10, pages 65–70. SAGE Publications Sage CA: Los Angeles, CA, 2021. 10

[3] Sofie Beier, Sam Berlow, Esat Boucaud, Zoya Bylinskii, Tianyuan Cai, Jenae Cohn, Kathy Crowley, Stephanie L Day, Tilman Dingler, Jonathan Dobres, et al. Readability research: An interdisciplinary approach. Foundations and TrendsWinAccounting<sup>ˆ</sup> , 16(4):214–324, 2022. 1, 2, 3, 9, 10

[4] Saeed Awadh Bin-Nashwan, Mouad Sadallah, and Mohamed Bouteraa. Use of chatgpt in academia: Academic integrity hangs in the balance. Technology in Society, 75:102370, 2023. 1

[5] Wieland Brendel, Jonas Rauber, and Matthias Bethge. Decision-based adversarial attacks: Reliable attacks against black-box machine learning models, 2018. 9

[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. 1

[7] Tianyuan Cai, Shaun Wallace, Tina Rezvanian, Jonathan Dobres, Bernard Kerr, Samuel Berlow, Jeff Huang, Ben D Sawyer, and Zoya Bylinskii. Personalized font recommendations: Combining ml and typographic guidelines to optimize readability. In Proceedings ofthe 2022 ACM designing interactive systems conference, pages 1–25, 2022. 2

[8] Arindam Chaudhuri, Krupa Mandaviya, Pratixa Badelia, and Soumya Ghosh. Optical character recognition systems for different languages with soft computing. Studies in Fuzziness and Soft Computing, 352, 01 2017. 1, 8

[9] Feiyang Chen, Dong Wang, Xin Xu, and Jing Gao. ADBA: Approximated decision boundary approach for black-box adversarial attacks, 2024. 4, 7, 9

[10] Jianbo Chen, Michael I. Jordan, and Martin J. Wainwright. Hopskipjumpattack: A query-efficient decision-based attack, 2020.

[11] Jinghui Chen and Quanquan Gu. RayS: A ray searching method for hard-label adversarial attack. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), 2020. 3, 7, 9

[12] Joana C. Costa, Tiago Roxo, Hugo Proenc¸a, and Pedro Ricardo Morais Inacio. How deep learning sees the world:´ A survey on adversarial attacks & defenses. IEEE Access, 12:61113–61136, 2024. 2

[13] Google Fonts. Google fonts analytics. https: //fonts.google.com/analytics. Accessed: 2026-08-18. 2

[14] Ziwei Gu, Ian Arawjo, Kenneth Li, Jonathan K Kummerfeld, and Elena L Glassman. An ai-resilient text rendering technique for reading and skimming documents. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pages 1–22, 2024. 3, 9

[15] Roger T. Hartley and Kathleen Crumpton. Quality of ocr for degraded text images, 1999. 3, 9

[16] Maedeh Hosseinpour, Borano Llana, Yusra Suhail, Kushas Khadka, Grant Garcia, and Shaun Wallace. One reader, two reading profiles: Language-dependent dyslexia and reading behaviors in english-spanish readers. In CHI 2026 Workshop on Science and Technology for Augmenting Reading (STAR), Barcelona, Spain, 2026. Workshop paper. 9

[17] Jaided AI. Easyocr. https://github.com/ JaidedAI/EasyOCR, 2020. Accessed: 2026-06-13. 2

[18] Martin Kisˇs, Michal Hradi ˇ s, and Old ˇ ˇrich Kodym. Brno mobile ocr dataset, 2019. 2

[19] Vladimir I. Levenshtein. Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8):707–710, 1966. 4

[20] Borano Llana, Alisa Baron, Kushas Khadka, Yusra Suhail, and Shaun Wallace. Typography and pupilometry: Exploring the speed, comprehension, and concentration trade-off in reading. Journal ofVision, 25(9):2470–2470, 2025. 2

[21] Borano Llana, Alisa Baron, Haihan Yu, Maedeh Hosseinpour, Yusra Suhail, Sean Chin, Kushas Khadka, and Shaun Wallace. Reading with diversity in mind: Pupillometry and typography towards inclusive design for adhd readers. In Proceedings of the Extended Abstracts of the 2026 CHI Conference on Human Factors in Computing Systems, CHI EA ’26, New York, NY, USA, 2026. Association for Computing Machinery. 2

[22] Thibault Maho, Teddy Furon, and Erwan Le Merrer. SurFree: a fast surrogate-free black-box attack. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 4

[23] ThaoAn Lily Nguyen, Borano Llana, Jonathan Piccoli, Yusra Suhail, Maedeh Hosseinpour, and Shaun Wallace. Rears: An interface for browser-based eye-tracking in remote readability studies. In CHI 2026 Workshop on

Science and Technology for Augmenting Reading (STAR), Barcelona, Spain, 2026. Workshop paper. 2

[24] Aleena Gertrudes Niklaus, Tianyuan Cai, Zoya Bylinskii, and Shaun Wallace. Digital reading rulers: Evaluating inclusively designed rulers for readers with dyslexia and without. In Proceedings ofthe 2023 CHI conference on humanfactors in computing systems, pages 1–17, 2023. 2

[25] Hilary Palmen, Michael Gilbert, and David Crossland.´ How bold can we be? the impact of adjusting font grade on readability in light and dark polarities. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, CHI ’23, New York, NY, USA, 2023. Association for Computing Machinery. 2

[26] Zobeir Raisi, Mohamed A Naiel, Paul Fieguth, Steven Wardell, and John Zelek. Text detection and recognition in the wild: A review. arXiv preprint arXiv:2006.04305, 2020. 1, 9

[27] Luz Rello, Martin Pielot, and Mari-Carmen Marcos. Make it big! the effect of font size and line spacing on online readability. In Proceedings of the 2016 CHI conference on Human Factors in Computing Systems, pages 3637–3648, 2016. 3, 9

[28] Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, and Soheil Feizi. Can ai-generated text be reliably detected?, 2025. 1

[29] R. Smith. An overview of the tesseract ocr engine. In Ninth International Conference on Document Analysis and Recognition (ICDAR 2007), volume 2, pages 629–633, 2007. 2

[30] Mehek S. Somai, Roshan L Peiris, and Garreth W. Tigwell. A review of text accessibility standards, guidelines, and font tool limitations. In Proceedings of the 27th International ACM SIGACCESS Conference on Computers and Accessibility, ASSETS ’25, New York, NY, USA, 2025. Association for Computing Machinery. 1, 9

[31] Andreas Spruck, Maximiliane Hawesch, Anatol Maier, C. Riess, Jurgen Seiler, and Andr ¨ e Kaup. 3d rendering´ framework for data augmentation in optical character recognition : (invited paper), 2021. 2

[32] Yusra Suhail, Kushas Khadka, Borano Llana, Maedeh Hosseinpour, Emory Salaun, and Shaun Wallace. Different font attributes influence english readers’ perception of font size across age groups. In CHI 2026 Workshop on Science and Technologyfor Augmenting Reading (STAR), Barcelona, Spain, 2026. Workshop paper. 2

[33] Yusra Suhail, Kushas Khadka, Borano Llana, Maedeh Hosseinpour, and Shaun Wallace. Personalizing font size normalization: Linguistic background shapes which font attributes matter. In Proceedings ofthe 34th ACM Conference on User Modeling, Adaptation and Personalization, pages 203–212, 2026. 2

[34] Miriam Sullivan, Andrew Kelly, and Paul McLaughlan. Chatgpt in higher education: Considerations for academic

integrity and student learning. Journal of Applied Learning & Teaching, 6, 03 2023. 1

[35] Christopher Tensmeyer, Zoya Bylinski, Tianyuan Cai, Dave Miller, Ani Nenkova, Aleena Niklaus, and Shaun Wallace. Web table formatting affects readability on mobile devices. In Proceedings of the ACM Web Conference 2023, pages 1334–1344, 2023. 10

[36] Olga Vechtomova. Introduction to information retrieval christopher d. manning, prabhakar raghavan, and hinrich schutze (stanford university, yahoo! research, and univer-¨ sity of stuttgart) cambridge: Cambridge university press, 2008, xxi+482 pp; hardbound, isbn 978-0-521-86571-5, \$60.00. Computational Linguistics, 35:307–309, 06 2009. 1

[37] Shaun Wallace, Zoya Bylinskii, Jonathan Dobres, Bernard Kerr, Sam Berlow, Rick Treitman, Nirmal Kumawat, Kathleen Arpin, Dave B Miller, Jeff Huang, et al. Towards individuated reading experiences: Different fonts increase reading speed for different individuals. ACM Transactions on Computer-Human Interaction (TOCHI), 29(4):1–56, 2022. 1, 2, 5

[38] Shaun Wallace, Jonathan Dobres, Zoya Bylinskii, and Ben Sawyer. Space for readability: effects on reading speed from individuated character and word spacing. Journal ofVision, 22(14):3349–3349, 2022. 2, 3

[39] Shaun Wallace, Na Kyoung Lee, Zhengyi Peng, Talie Massachi, Long Do, Surbhi Rathore, Sarah M Brown, and Jeff Huang. Attract & engage visitors for tabular data maintenance: a longitudinal naturalistic field study. Proceedings of the ACM on Human-Computer Interaction, 10(6):1–41, 2026. 10

[40] Shaun Wallace, Alexandra Papoutsaki, Neilly H Tan, Hua Guo, and Jeff Huang. Case studies on the motivation and performance of contributors who verify and maintain in-flux tabular datasets. Proceedings of the ACM on Human-Computer Interaction, 5(CSCW2):1–25, 2021. 10

[41] Hui Wang, Anh Dang, Zihao Wu, and Son Mac. Generative ai in higher education: Seeing chatgpt through universities’ policies, resources, and guidelines. Computers and Education: Artificial Intelligence, 7:100326, 2024. 1

[42] Kai Wang, Boris Babenko, and Serge Belongie. Endto-end scene text recognition. In 2011 International conference on computer vision, pages 1457–1464. IEEE, 2011. 1, 8

[43] Aleena Watson and Shaun Wallace. Improving reading outcomes using digital reading rulers for readers with & without dyslexia. Journal ofVision, 21(9):2650–2650, 2021. 9

[44] Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, and Xiangyu Zhang. General ocr theory: Towards ocr-2.0 via a unified end-to-end model, 2024. 2, 6, 8

[45] Zixuan Yi, Yuanming Shao, Shaun Wallace, Zachary Ives, and Ryan Marcus. Data canvas: A provenance-guided harness for agentic data engineering. In Third Conference on Language Modeling, pages 1–13, 2026. 9