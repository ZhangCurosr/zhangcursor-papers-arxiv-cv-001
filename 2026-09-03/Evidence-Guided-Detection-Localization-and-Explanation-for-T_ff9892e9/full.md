# Evidence-Guided Detection, Localization and Explanation for Text-Centric Image Forensics

Peifeng Liu   
Guangdong Provincial Key   
Laboratory of Intelligent Information   
Processing, Shenzhen Key Laboratory   
of Media Security   
Shenzhen University   
Shenzhen, China   
2410043004@mails.szu.edu.cn   
Yangxin Yu   
Guangdong Provincial Key   
Laboratory of Intelligent Information   
Processing, Shenzhen Key Laboratory   
of Media Security   
Shenzhen University   
Shenzhen, China   
2453043007@mails.szu.edu.cn   
Bin Li<sup>∗</sup>   
Guangdong Provincial Key   
Laboratory of Intelligent Information   
Processing, Shenzhen Key Laboratory   
of Media Security, SZU-AFS Joint   
Innovation Center for AI Technology   
Shenzhen University   
Shenzhen, China   
libin@szu.edu.cn   
Leqing Chen   
Guangdong Provincial Key   
Laboratory of Intelligent Information   
Processing, Shenzhen Key Laboratory   
of Media Security   
Shenzhen University   
Shenzhen, China   
2400042018@mails.szu.edu.cn   
Qingsong Zhang   
Guangdong Provincial Key   
Laboratory of Intelligent Information   
Processing, Shenzhen Key Laboratory   
of Media Security   
Shenzhen University   
Shenzhen, China   
2230707827@qq.com   
Xiaoye Qiu   
Guangdong Provincial Key   
Laboratory of Intelligent Information   
Processing, Shenzhen Key Laboratory   
of Media Security   
Shenzhen University   
Shenzhen, China   
2553043008@mails.szu.edu.cn

## Abstract

The rapid progress of AIGC has made text-centric image manipulation increasingly accessible, creating new forensic challenges that require not only authenticity detection but also spatial grounding and evidence-based explanation. This paper presents our solution to the GenText-Forensics Challenge at ACM Multimedia 2026. We propose an evidence-guided detector-localizer-reasoner system, where an image-level detector provides a global authenticity prior, a dedicated localizer extracts tampered regions as spatial grounding evidence, and an MLLM-based reasoner generates structured forensic reports grounded in this expert forensic evidence. These modules are connected through a cascaded evidence flow: the detector gates the subsequent localization and prompting process, the localizer converts tamper responses into grounding boxes, and the reasoner is trained to synthesize the detector decision and local ized evidence into the final report. As a key part of our method, we introduce iterative dificulty-aware mining to improve localization quality and apply report-mask consistency post-processing to align report grounding with predicted masks. On the oficial hidden test set, our system achieves a final score of 0.638 and ranks second in the challenge, validating the efectiveness of the proposed evidence-guided system. The code is available at this https URL.

• Security and privacy → Social aspects of security and privacy.

## Keywords

Text-Centric Image Forensics, Tampering Localization, Multimodal Large Language Models

## ACM Reference Format:

Peifeng Liu, Bin Li, Qingsong Zhang, Yangxin Yu, Leqing Chen, and Xiaoye Qiu. 2026. Evidence-Guided Detection, Localization and Explanation for Text-Centric Image Forensics. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3767308. 3837699

## 1 Introduction

Text-centric images like documents and receipts serve as key evidence in daily work. Advanced editing tools [6, 12, 14, 18, 28, 29, 32] enable stealthy text tampering that retains original layouts but modifies critical information such as amounts and timestamps. Tiny altered areas can completely distort evidentiary validity. Thus, text image forensics requires not only authenticity classification but also tampering localization and evidence-based analysis reports.

The GenText-Forensics Challenge at ACM Multimedia 2026 formulates text-centric image forensics as a unified forensic report generation task based on the RealText-V2 dataset. Given a text-rich image, participants are required to produce a structured Markdownstyle report that integrates authenticity judgment, spatial grounding, and evidence-based reasoning. The evaluation jointly considers detection, localization, explanation, and report quality.

Unlike natural image manipulation [5, 7, 21, 22, 31] or conventional low-resolution document tampering detection [23], the

RealText-V2 setting focuses on high-resolution text-centric images with dense textual content, complex layouts, and strong semantic dependencies. In this setting, forgery evidence is inherently heterogeneous. Image-level cues are encoded in global feature representations that capture long-range correlations and documentwide statistical deviations, providing a holistic authenticity prior. Region-level traces arise from local edits around modified char acters or pasted text, such as edge blending defects, background discontinuity, and compression mismatch, requiring precise spatial localization. Semantic-level anomalies, such as altered words, numbers, or fields, may further introduce logical contradictions or misleading meanings that require contextual interpretation.

These heterogeneous evidence types difer in granularity and feature space, making it dificult for a single model to handle all forensic requirements. Dedicated forensic models [3, 9, 13, 20, 23, 25] are efective at capturing low-level tampering traces and producing detection scores, heatmaps, or masks, but they generally lack semantic reasoning and structured report-generation capability. In contrast, direct Multimodal Large Language Model (MLLM) [1, 12, 16, 28, 30] inference provides strong semantic understanding and language reasoning, but remains limited in fine-grained visual perception for subtle artifact analysis and precise localization.

Guided by these insights, we propose an evidence-guided detectorlocalizer-reasoner framework that decomposes forensic tasks into specialized modules and aggregates their outputs into a unified report. For image-level judgment, a detector with auxiliary localiza tion supervision learns global authenticity priors while retaining tampering-sensitive spatial cues. For region-level localization, we present Dificulty-Aware Tamper Localization, which iteratively mines hard samples and error-prone regions, then fine-tunes the model via dificulty-weighted sampling and hard-region cropping. This strategy enhances localization performance on small edits and ambiguous boundaries in high-resolution text-centric forgeries.

Powered by the detector’s authenticity prediction and suspiciousregion bounding boxes, the LLM reasoner generates a structured forensic report via an evidence-conditioned prompt. Unlike direct MLLM inference, which may produce ungrounded claims from semantic priors, our reasoner is strictly constrained by expert module outputs, reducing hallucinations and aligning authenticity judgment, spatial localization, and textual analysis. Our system achieved second place in the final evaluation, validating the efectiveness of our design.

Overall, our system ofers three main advantages:

• Complementary Expert-LLM Collaboration. Specialized visual models handle detection and localization, while the LLM-based reasoner performs semantic integration and report generation.

• Dificulty-Aware Spatial Evidence Enhancement. The localization strategy mines hard samples and error-centered regions, improving spatial grounding for report generation.

• Evidence-Grounded Report Generation. Conditioning the reasoner on detector decisions and localized regions improves alignment between generated reports and expert forensic evidence.

## 2 Related Work

Image manipulation detection and localization (IMDL) aims to identify manipulated images and localize tampered regions. Existing methods can be broadly divided into expert forensic models and MLLM-centric models. The former focus on low-level manipulation traces, while the latter introduce multimodal semantic priors for interpretable reasoning.

Expert Models. Expert forensic models detect manipulation by modeling visual artifacts such as boundary inconsistency, noise mismatch, compression traces, and statistical anomalies. For general image manipulation detection, representative methods exploit RGB-DCT artifacts [13], noise and boundary cues [2], noise-sensitive fingerprints [9], or non-semantic manipulation features [25]. For document image forensics, DTD [23] and FFDN [3] incorporate DCT, wavelet, and frequency-domain cues to improve robustness against post-processing attacks. TFAM [8] fuses local-global contextual information, while CD-SD [36] and $\mathrm { D I T L ^ { 2 } }$ [15] address cross-domain robustness through color-semantic decoupling and frequency-domain adaptation. These methods are efective at producing detection scores, heatmaps, or masks, but generally lack semantic reasoning and structured report-generation capability.

MLLM-Centric Models. Recent studies have introduced MLLMs into forgery analysis to enhance semantic understanding and interpretability. Methods such as FakeShield [34], ForgeryGPT [17], SIDA [11], and ForgerySleuth [27] combine MLLMs with domainaware modules, mask-aware extractors, detection/segmentation tokens, or trace encoders to connect low-level forgery cues with high-level reasoning. For text- and document-centric forensics, TVSIP [33] integrates expert-extracted visual artifacts with MLLMidentified semantic conflicts, while VeriChain [26] explores reasoningchain-based verification. These works show the potential of MLLMs for interpretable forensics. Following this direction, our method further treats detector outputs and localized tamper regions as explicit forensic evidence to condition MLLM reasoning, enabling the final report to be grounded in both global authenticity judgment and spatial localization evidence.

## 3 Proposed Solution

## 3.1 Overview

As illustrated in Fig. 1 (a), we propose an evidence-guided detectorlocalizer-reasoner system for text-centric image forensic report generation. Given an input image �, the detector first predicts an image-level forgery probability:

$$
\begin{array} { r } { p _ { \mathrm { d e t } } = \mathcal { D } ( I ) , } \end{array}\tag{1}
$$

where $\mathcal { D } ( \cdot )$ denotes the detector. The predicted label �ˆ is obtained by thresholding $ { p _ { \mathrm { d e t } } }$ with $\tau _ { \mathrm { d e t } }$ , where $\hat { y } = \mathrm { f o r g e d i f } p _ { \mathrm { d e t } } \ge \tau _ { \mathrm { d e t } }$ , and $\hat { y } = \mathrm { a u t h e n t i c }$ otherwise. This detector output serves as a global expert prior and controls whether spatial evidence should be further extracted.

When the image is predicted as forged, the localizer is activated to produce a tamper probability heatmap $H = \mathcal { F } ( I )$ . The heatmap is then binarized, filtered by connected components, and converted into bounding boxes:

$$
\begin{array} { r } { \mathcal { B } = \left\{ \mathrm { B B o x } ( c ) \mid c \in \mathrm { C C } \big ( \mathbb { I } ( H \geq \tau _ { \mathrm { l o c } } ) \big ) , \mathrm { A r e a } ( c ) \geq \alpha \right\} , } \end{array}\tag{2}
$$

![](images/51995adb9dacdcc554ca9650e77f2f6813e0367c5e8e68bce00913db8030dc39.jpg)  
Figure 1: Overview of the proposed solution. (a) The evidence-guided detector-localizer-reasoner system for forensic report generation. (b) The image-level forgery detector, where DINOv3 tokens support both authenticity classification and auxiliary coarse localization. (c) The iterative dificulty-aware localization process, which mines hard samples and error-centered crops to improve tamper grounding.

where $\tau _ { \mathrm { l o c } }$ is the localization threshold and � is the minimum area threshold. The resulting bounding boxes B serve as explicit spatial expert evidence for the reasoning stage.

The prompt for the MLLM-based reasoner is constructed in a detector-gated manner:

$$
\mathcal { P } = \left\{ \begin{array} { l l } { \mathcal { P } _ { \mathrm { e v i } } ( I , \hat { y } , \mathcal { B } ) , \quad \hat { y } = \mathrm { f o r g e d , ~ } | \mathcal { B } | > 0 , } \\ { \mathcal { P } _ { \mathrm { a u t h } } ( I , \hat { y } ) , \quad \mathrm { ~ o t h e r w i s e . } } \end{array} \right.\tag{3}
$$

Here, $\mathcal { P } _ { \mathrm { e v i } }$ denotes an evidence-conditioned prompt containing localized suspicious regions, while $\mathcal { P } _ { \mathrm { a u t h } }$ denotes an authenticoriented prompt without forced grounding regions. Thus, forgedoriented reasoning is triggered only when both the detector prediction and localized expert evidence support it.

The initial report is generated by the reasoner:

$$
R _ { 0 } = \mathcal { R } ( \mathcal { P } ) ,\tag{4}
$$

where R(·) denotes the MLLM-based reasoner. To further enforce consistency between report grounding and localization masks, we apply report-mask consistency post-processing:

$$
R = \Psi _ { \mathrm { r m c } } ( R _ { 0 } , { \mathcal { B } } ) ,\tag{5}
$$

where $\Psi _ { \mathrm { r m c } } ( \cdot )$ refines the [GROUNDING] fields in the generated report using localization-derived boxes and supplements missing localized regions when necessary.

In this way, the final report is constrained by expert-module evidence from both detection and localization, improving the consistency among authenticity judgment, spatial grounding, and evidencebased explanation.

## 3.2 Auxiliary Localization-Supervised Image-Level Forgery Detection

As illustrated in Fig. 1 (b), the detector provides a global authenticity prior for the whole pipeline. We adopt DINOv3 [24] as the visual backbone because its transformer representations can capture longrange visual dependencies in high-resolution text-centric images, while its patch tokens also preserve spatial information useful for tampering-aware supervision. Rather than using the CLS token alone for classification, we concatenate it with the mean-pooled patch tokens to combine the global summary and distributed local responses:

$$
{ \bf z } _ { \mathrm { i m g } } = \left[ { \bf z } _ { \mathrm { c l s } } ; \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf z } _ { i } \right] ,\tag{6}
$$

where $\mathbf { z } _ { \mathrm { c l s } }$ is the CLS token and $\{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N }$ are the patch tokens. The forged probability is then predicted by a lightweight classification head:

$$
\begin{array} { r } { p _ { \mathrm { d e t } } = \sigma \left( f _ { \mathrm { c l s } } ( \mathbf { z } _ { \mathrm { i m g } } ) \right) . } \end{array}\tag{7}
$$

A classification-only detector may learn image-level separability without preserving spatially discriminative tampering cues. Therefore, we attach an auxiliary convolutional localization head to the

patch tokens. The patch tokens are reshaped into a two-dimensional feature map and fed into a lightweight convolutional decoder to predict a coarse tampering heatmap:

$$
H _ { \mathrm { c o a r s e } } = f _ { \mathrm { c o n v } } \left( { \mathrm { R e s h a p e } } \left( \{ \mathbf { z } _ { i } \} _ { i = 1 } ^ { N } \right) \right) .\tag{8}
$$

The detector is trained with a joint objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d e t } } = \lambda _ { 1 } \mathcal { L } _ { \mathrm { B C E } } ( p _ { \mathrm { d e t } } , y ) + \lambda _ { 2 } \mathcal { L } _ { \mathrm { C E } } ( H _ { \mathrm { c o a r s e } } , M ) \qquad } \\ { + \lambda _ { 3 } \mathcal { L } _ { \mathrm { D i c e } } ( H _ { \mathrm { c o a r s e } } , M ) , \qquad } \end{array}\tag{9}
$$

where $y$ is the image-level label and � is the ground-truth tampering mask. The auxiliary localization head is used to regularize the detector with pixel-level supervision, encouraging the shared backbone to retain tampering-sensitive spatial cues.

## 3.3 Dificulty-Aware Tampering Localization

The localization branch is activated for images classified as forged by the detector, aiming to identify tampered regions for subsequent evidence-conditioned reasoning. However, the dificulty of localiza tion varies significantly across high-resolution text-centric images. After initial training, many easy samples can already be localized reliably and provide limited additional supervision if repeatedly sampled. In contrast, medium and hard samples, such as small text edits, ambiguous boundaries, confusing backgrounds, and regions with strong model disagreement, are less frequent but contain more informative error signals. Uniform sampling therefore allocates many updates to already learned patterns while insuficiently revisiting the localizer’s failure cases. This motivates an iterative dificulty-aware mining strategy that converts the current model’s localization errors into targeted fine-tuning samples.

As shown in Fig. 1 (c), we adopt DTD [23] as the primary localizer, since it is designed for document tampering localization and captures both visual artifacts and frequency-domain inconsistencies. Given an image �, DTD predicts a tampering probability map:

$$
H = { \mathcal { F } } _ { \mathrm { d t d } } ( I ) ,\tag{10}
$$

and is optimized with pixel-level mask supervision:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { l o c } } = \lambda _ { 4 } \mathcal { L } _ { \mathrm { C E } } ( H , M ) + \lambda _ { 5 } \mathcal { L } _ { \mathrm { D i c e } } ( H , M ) , } \end{array}\tag{11}
$$

where � denotes the ground-truth tampering mask. We first train a baseline DTD localizer on the RealText-V2 training set. This model serves as the initial state $\mathcal { F } _ { \mathrm { d t d } } ^ { ( 0 ) }$ for dificulty mining, so that hard samples are defined according to actual localization failures rather than hand-crafted assumptions.

At mining round �, let $I _ { i }$ denote the �-th full image in the training set and $\mathcal { F } _ { \mathrm { d t d } } ^ { ( t ) }$ denote the current DTD localizer. We run $\mathcal { F } _ { \mathrm { d t d } } ^ { ( t ) }$ on $I _ { i }$ and obtain its binary prediction $D _ { i } ^ { ( t ) }$ . To capture complementary error cues, we also use predictions from ASCFormer [20] and SparseViT [25], denoted as $A _ { i }$ and $S _ { i }$ , respectively. Thus, the mining input for each image is:

$$
\{ M _ { i } , D _ { i } ^ { ( t ) } , A _ { i } , S _ { i } \} ,\tag{12}
$$

The mining is conducted at the full-image level, while the extracted error regions are later used to guide crop selection during finetuning.

Taking $M _ { i }$ as reference, we compute the DTD false-negative and false-positive regions:

$$
F N _ { i } ^ { ( t ) } = M _ { i } \setminus D _ { i } ^ { ( t ) } , \qquad F P _ { i } ^ { ( t ) } = D _ { i } ^ { ( t ) } \setminus M _ { i } .\tag{13}
$$

We then summarize each image by a dificulty score that combines pixel-level errors, component-level missed regions, large-region failures, and cross-model disagreement:

$$
\begin{array} { r l } & { s _ { i } ^ { ( t ) } = 0 . 3 5 r _ { \mathrm { f n } } ^ { i } + 0 . 2 0 r _ { \mathrm { m i s s } } ^ { i } + 0 . 1 5 r _ { \mathrm { l a r g e } } ^ { i } } \\ & { ~ + 0 . 1 5 \operatorname* { m i n } ( r _ { \mathrm { f p } } ^ { i } , 1 ) + 0 . 1 5 r _ { \mathrm { d i s } } ( D _ { i } ^ { ( t ) } , A _ { i } , S _ { i } ) , } \end{array}\tag{14}
$$

where $r _ { \mathrm { f n } } ^ { i }$ and $r _ { \mathrm { f p } } ^ { i }$ measure false-negative and false-positive ratios, $r _ { \mathrm { m i s s } } ^ { i }$ measures missed connected components, $r _ { \mathrm { l a r g e } } ^ { i }$ indicates whether a large tampered component is missed, and $r _ { \mathrm { d i s } } ( D _ { i } ^ { ( t ) } , A _ { i } , S _ { i } ) =$ $1 - \operatorname { I o U } ( D _ { i } ^ { ( t ) } , A _ { i } \cup S _ { i } )$ measures disagreement between DTD and the auxiliary localizers.

Based on $s _ { i } ^ { ( t ) }$ and the localization statistics, each image is assigned to an easy, medium, or hard group using empirically defined rule-based criteria. These criteria are designed according to the main failure modes observed in the baseline localizer: severe missed tampered regions, excessive false positives, low localization quality, and large connected components being missed.

$$
g _ { i } = \left\{ \begin{array} { l l } { \mathrm { h a r d } , \quad } & { ( s _ { i } ^ { ( t ) } \geq 0 . 6 0 ) \vee ( F _ { i } < 0 . 4 0 ) \vee ( r _ { \mathrm { f n } } ^ { i } > 0 . 4 5 ) } \\ { \qquad \vee ( r _ { \mathrm { f p } } ^ { i } > 0 . 8 0 ) \vee ( r _ { \mathrm { m i s s } } ^ { i } > 0 . 5 0 ) \vee ( r _ { \mathrm { l a r g e } } ^ { i } = 1 ) , } \\ { \mathrm { e a s y } , \quad } & { ( s _ { i } ^ { ( t ) } < 0 . 2 5 ) \wedge ( F _ { i } \geq 0 . 7 5 ) \wedge ( R _ { i } \geq 0 . 8 0 ) } \\ { \qquad \wedge ( r _ { \mathrm { f n } } ^ { i } \leq 0 . 1 5 ) \wedge ( r _ { \mathrm { f p } } ^ { i } \leq 0 . 2 0 ) \wedge ( r _ { \mathrm { l a r g e } } ^ { i } = 0 ) , } \\ { \mathrm { m e d i u m } , \quad } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{15}
$$

Here, $F _ { i }$ and $R _ { i }$ denote the DTD F1-score and recall on image $I _ { i } ,$ respectively. This rule treats samples with severe localization failures as hard, regards only consistently well-localized samples as easy, and assigns the remaining ambiguous cases to the medium group.

The mining process produces an image-level dificulty manifest $\boldsymbol { M } ^ { ( t ) } \ = \ \{ ( I _ { i } , \boldsymbol { s } _ { i } ^ { ( t ) } , g _ { i } ) \}$ and a region-level hard set $\begin{array} { r l } { \mathcal { H } ^ { ( t ) } } & { { } = } \end{array}$ $\{ ( I _ { i } , b _ { i } ^ { j } , q _ { i } ^ { j } ) \}$ , where $b _ { i } ^ { j }$ is the bounding box of an error component and $q _ { i } ^ { j }$ denotes its error type, such as false negative, false positive, auxiliary-recovered false negative, or all-model-missed region. During fine-tuning, full images are sampled according to fixed group probabilities:

$$
\pi _ { \mathrm { e a s y } } = 0 . 0 5 , \qquad \pi _ { \mathrm { m e d i u m } } = 0 . 3 5 , \qquad \pi _ { \mathrm { h a r d } } = 0 . 6 0 .\tag{16}
$$

After a full image is selected, a 512 × 512 crop is extracted; for hard samples, the crop is centered around a mined hard region with probability 0.75, with random jitter applied to avoid repeatedly using the same crop. In each mining-and-fine-tuning stage, we finetune the localizer for 30 epochs using $\mathbf { \mathcal { M } } ^ { ( t ) }$ and $\bar { \mathcal { H } } ^ { ( t ) }$ , and select the best checkpoint for the next mining stage. The updated localizer is used to re-predict the training set and refresh the hard samples:

$$
\mathcal { F } _ { \mathrm { d t d } } ^ { ( t + 1 ) } = \mathrm { F i n e T u n e } \left( \mathcal { F } _ { \mathrm { d t d } } ^ { ( t ) } ; \mathcal { M } ^ { ( t ) } , \mathcal { H } ^ { ( t ) } \right) .\tag{17}
$$

This closed-loop procedure allows the localizer to progressively shift its learning focus from the initial failure cases to the remaining hard regions that are still poorly localized after each model update.

Table 1: Comparison with state-of-the-art methods on the internal validation split. We report detection, localization, explanation, and the reproducible oficial weighted score. “-” denotes unavailable metrics. SIDA<sup>∗</sup> and TVSIP<sup>∗</sup> denote SIDA [11] and TVSIP [33] implemented with Qwen3-VL-4B [1] as the MLLM. Bold indicates the best performance.
<table><tr><td rowspan="2">Method</td><td colspan="2">Detection</td><td colspan="4">Localization</td><td colspan="2">Explanation</td><td rowspan="2">Score</td></tr><tr><td>Bal-ACC</td><td>Wtd-F1</td><td>P</td><td>R</td><td>F1</td><td>IoU</td><td>CSS</td><td>BS</td></tr><tr><td>Trufor</td><td>94.60</td><td>94.67</td><td>78.26</td><td>63.95</td><td>66.80</td><td>54.33</td><td>–</td><td>一</td><td></td></tr><tr><td>SparseViT</td><td>93.67</td><td>93.64</td><td>63.30</td><td>38.92</td><td>44.93</td><td>33.16</td><td>一</td><td></td><td></td></tr><tr><td>DINOv3-Conv</td><td>99.57</td><td>99.56</td><td>21.08</td><td>90.26</td><td>30.81</td><td>20.41</td><td>V</td><td></td><td></td></tr><tr><td>DINOv3-Mask2Former</td><td>99.18</td><td>99.11</td><td>42.79</td><td>88.53</td><td>54.77</td><td>40.58</td><td></td><td></td><td></td></tr><tr><td>Swin-Mask2Former</td><td>87.97</td><td>86.72</td><td>21.39</td><td>83.65</td><td>31.96</td><td>20.51</td><td></td><td></td><td></td></tr><tr><td>DTD</td><td>95.08</td><td>95.53</td><td>83.05</td><td>70.34</td><td>72.49</td><td>61.58</td><td></td><td></td><td></td></tr><tr><td>ASCFormer</td><td>91.15</td><td>91.84</td><td>81.51</td><td>60.96</td><td>65.16</td><td>53.11</td><td>=</td><td>一</td><td></td></tr><tr><td>InternVL3.5-4B</td><td>96.20</td><td>96.29</td><td>23.65</td><td>23.17</td><td>21.68</td><td>14.97</td><td>90.86</td><td>83.96</td><td>47.47</td></tr><tr><td>Qwen3VL-2B</td><td>95.40</td><td>95.48</td><td>52.94</td><td>24.27</td><td>29.37</td><td>19.14</td><td>96.48</td><td>84.68</td><td>49.00</td></tr><tr><td>Qwen3VL-4B</td><td>93.92</td><td>93.79</td><td>41.38</td><td>38.80</td><td>37.28</td><td>26.35</td><td>96.40</td><td>84.26</td><td>51.32</td></tr><tr><td>Qwen3VL-8B</td><td>98.15</td><td>98.15</td><td>64.22</td><td>32.04</td><td>38.74</td><td>26.75</td><td>96.61</td><td>87.49</td><td>53.27</td></tr><tr><td>SIDA*</td><td>96.95</td><td>97.04</td><td>60.63</td><td>35.63</td><td>41.73</td><td>30.53</td><td>89.21</td><td>83.58</td><td>53.86</td></tr><tr><td>TVSIP*</td><td>91.85</td><td>92.58</td><td>54.53</td><td>68.13</td><td>56.65</td><td>43.21</td><td>96.20</td><td>83.77</td><td>57.62</td></tr><tr><td>Ours</td><td>99.60</td><td>99.56</td><td>83.71</td><td>79.61</td><td>79.03</td><td>69.93</td><td>91.01</td><td>80.15</td><td>69.86</td></tr></table>

## 3.4 SFT for the Evidence-Conditioned Reasoner

To make the reasoner follow the evidence provided by expert modules, we construct QA-style SFT data from RealText-V2 image-maskreport triplets. For each image, we convert the tampering mask into normalized bounding boxes in the [0, 999] range. Authentic samples are assigned zero boxes, while forged samples include the extracted suspicious boxes in the question. The question $Q _ { i }$ therefore contains the input image and an evidence-aware instruction, asking the reasoner to generate a forensic report based on the provided detector/localizer evidence. The answer $A _ { i }$ is the challenge-provided reference forensic report, with grounding coordinates normalized to the same format. The resulting SFT dataset is defined as:

$$
\mathcal { D } _ { \mathrm { s f t } } = \{ ( Q _ { i } , A _ { i } ) \} _ { i = 1 } ^ { N } .\tag{18}
$$

We fine-tune the reasoner with the standard autoregressive objective:

$$
\mathcal { L } _ { \mathrm { s f t } } = - \sum _ { i , t } \log p _ { \theta } ( a _ { i , t } \mid Q _ { i } , a _ { i , < t } ) ,\tag{19}
$$

where $a _ { i , t }$ denotes the �-th token of the target report.

## 3.5 Report-Mask Consistency Post-processing

To improve spatial consistency, we refine the generated report after inference using the final localization mask. Specifically, we parse the report [GROUNDING] boxes and replace each one with the most overlapping box extracted from the localization mask.

If the localization mask contains additional boxes not covered by the report, we append them as supplementary anomaly entries. These entries only include localization-derived grounding coordinates, with the reason field marked as an expert-module supplement. This preserves the original analysis while ensuring better consistency between the report and the predicted mask.

## 4 Experiments

## 4.1 Experimental Setup

Dataset. We conducted experiments on the oficial RealText-V2 dataset from the GenText-Forensics Challenge. Since the oficial test annotations were withheld, we constructed an internal validation split by randomly sampling 10% of real images and 10% of forged images from the training set, with a fixed random seed of 42 for reproducibility; the remaining data was used for training. This split was used for component-level evaluation before submitting the complete system to the oficial evaluation server.

State-of-the-Art Methods. To comprehensively evaluate our framework, we compared it with representative methods from three categories. For expert forensic models, we included general IMDL methods, i.e., TruFor [9] and SparseViT [25], as well as segmentation-based and document-oriented baselines, including DINOv3-Mask2Former [4, 24], Swin-Mask2Former [4, 19], DTD [23], and ASCFormer [20]. For MLLM-centric forensic models, we included SIDA [11] and TVSIP [33]. We further evaluated generalpurpose MLLMs, including Qwen3-VL [1] and InternVL3.5 [30].

Evaluation Metrics. We assess three dimensions: detection, localization and explanation. Detection metrics include balanced accuracy (Bal-Acc) and weighted F1 (Wtd-F1); localization uses pixel-wise precision, recall, F1 and IoU; explanation is measured via BERTScore (BS) [35] and cosine semantic similarity (CSS). We unify metric calculation for fair baseline comparison. For expert models, detection results use the classification head output with threshold 0.5. Models lacking a classification branch predict authenticity from the maximum value of segmentation masks (threshold 0.5). For MLLM-centric models, we parse authenticity labels and coordinates from generated text, then convert coordinates to binary masks for localization evaluation. The oficial subjective LLM-based report score is unstable in internal tests. We thus calculate a reproducible objective partial score using Detection F1, Localization IoU and BERTScore, weighted by oficial coeficients 0.3, 0.4 and 0.15.

Table 2: Ablation study of key components. We evaluate evidence guidance, auxiliary localization loss, and report-mask consistency post-processing.
<table><tr><td rowspan="2">Ablation</td><td colspan="2">Detection</td><td colspan="2">Localization</td><td colspan="2">Explanation</td></tr><tr><td>Bal-ACC</td><td>Wtd-F1</td><td>F1</td><td>IoU</td><td>CSS</td><td>BS</td></tr><tr><td>Baseline</td><td>93.92</td><td>93.79</td><td>37.28</td><td>26.35</td><td>96.40</td><td>84.26</td></tr><tr><td>w/o Detector</td><td>95.70</td><td>95.81</td><td>79.35</td><td>70.21</td><td>90.89</td><td>80.17</td></tr><tr><td>w/o Localizer</td><td>99.57</td><td>99.56</td><td>44.65</td><td>32.33</td><td>91.05</td><td>81.61</td></tr><tr><td>w/o Aux. Loc. Loss</td><td>97.27</td><td>97.04</td><td>76.30</td><td>67.68</td><td>91.00</td><td>80.08</td></tr><tr><td>w/o RMC pp.</td><td>99.40</td><td>99.33</td><td>62.98</td><td>49.67</td><td>91.01</td><td>80.20</td></tr><tr><td>Full Model</td><td>99.60</td><td>99.56</td><td>79.03</td><td>69.93</td><td>91.01</td><td>80.15</td></tr></table>

Table 3: Efect of dificulty-aware tampering localization.
<table><tr><td>Ablation</td><td>P</td><td>R</td><td>F1</td><td>IoU</td><td>∆IoU</td></tr><tr><td>DTD Baseline</td><td>78.32</td><td>72.38</td><td>71.67</td><td>60.40</td><td></td></tr><tr><td>Difficulty Mining (Round 1)</td><td>82.40</td><td>77.98</td><td>77.32</td><td>67.38</td><td>+6.98</td></tr><tr><td>Difficulty Mining (Round 2)</td><td>83.71</td><td>79.61</td><td>79.03</td><td>69.93</td><td>+9.53</td></tr><tr><td>Difficulty Mining (Round 3)</td><td>81.17</td><td>76.60</td><td>75.85</td><td>65.54</td><td>+5.14</td></tr></table>

Implementation Details. All experiments were conducted on four NVIDIA L40 GPUs. For fair comparison, all trainable baselines and our framework were trained on the same training split and evaluated on the same internal validation split. For the detector, we fine-tuned DINOv3 ViT-B/16 [24] at an input resolution of896×1344 for 12 epochs with a batch size of 12, using a backbone learning rate $\operatorname { o f } 1 \times 1 0 ^ { - 5 }$ and a head learning rate of $1 \times 1 0 ^ { - 4 }$ . For the localizer, we used DTD [23] trained with the proposed Dificulty-Aware Tamper Localization strategy for two mining-and-fine-tuning rounds, i.e., � = 2. For the reasoner, we fine-tuned Qwen3-VL-4B-Instruct [1] via LoRA [10] SFT for 20 epochs. We froze the visual encoder, enabled the MM projector, and only applied LoRA adapters to the LLM backbone, with LoRA rank set to 32 and alpha set to 64.

## 4.2 Main Results

Table 1 presents the overall comparison with expert forensic models, general-purpose MLLMs, and MLLM-centric forensic methods. Our method achieves the best overall score, indicating that the proposed evidence-guided system efectively balances image-level detection, pixel-level localization, and explanation quality. The results show that expert models are generally stronger than general-purpose MLLMs in tamper localization, while MLLMs provide competitive semantic explanation ability but lack reliable fine-grained grounding. By combining detector-derived authenticity priors, localization based spatial evidence, and evidence-conditioned reasoning, our method obtains more balanced performance across all evaluation dimensions. These results validate the efectiveness of using expert forensic evidence to guide MLLM-based report generation for text-centric image forensics.

## 4.3 Ablation Studies

Efect of Evidence Guidance. Table 2 shows that explicit expert evidence is crucial for balanced forensic report generation. The baseline, which relies mainly on direct MLLM reasoning, obtains weak localization performance despite reasonable explanation scores. Removing the detector weakens image-level authenticity judgment, while removing the localizer causes a clear drop in spatial grounding, showing that global and regional evidence play complementary roles. The full model achieves the best overall balance by combining detector-derived authenticity priors with localization-based spatial evidence, enabling the reasoner to generate reports grounded in expert forensic evidence.

Efect of Auxiliary Localization Loss. As shown in Table 2, removing the auxiliary localization loss from the detector degrades both detection and localization-related performance. This indicates that image-level classification alone is insuficient for learning tampering-sensitive representations. By introducing coarse mask supervision, the detector is encouraged to preserve local artifact cues while learning global authenticity features, which leads to more stable forged/authentic prediction and better downstream evidence guidance.

Efect of Dificulty Mining. Table 3 evaluates the efect of iterative dificulty-aware mining. Compared with the DTD baseline, the first mining round brings a clear improvement in all localization metrics, indicating that revisiting hard samples and error-centered regions efectively strengthens the localizer. The second round achieves the best performance, improving IoU by 9.53 points over the baseline, which shows that refreshing the hard sample set after model update helps the localizer focus on remaining failure cases. However, the third round leads to a performance drop, suggesting that excessive hard-sample mining may overemphasize dificult patterns and weaken distribution balance. Therefore, we use the second-round dificulty-mined localizer in the final system.

Efect of Report-Mask Consistency Post-processing. As shown in Table 2, removing RMC post-processing noticeably lowers localization metrics but barely afects explanation scores. Since RMC only aligns report grounding boxes with the final mask and supplements missing regions without altering semantic content, this lightweight step efectively improves spatial consistency between reports and mask evidence.

## 4.4 Final Challenge Submission

We apply the complete evidence-guided system to the oficial hidden test set of the GenText-Forensics Challenge. Since the test annotations are withheld, we report the oficial leaderboard result. Our system achieved a final score of 0.638 and ranked second in the final evaluation, demonstrating the efectiveness of the proposed solution under the oficial challenge protocol.

## 5 Conclusion

We presented an evidence-guided detector-localizer-reasoner system for the GenText-Forensics Challenge. By combining auxiliary localization loss, dificulty-aware mining, and report-mask consistency post-processing, our system achieved 0.638 on the hidden test set and ranked second. We also summarized practical insights from building a coordinated evidence-grounded forensic pipeline.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China (NSFC) under Grant U23B2022, and in part by the Shenzhen Science and Technology Program under Grants JCYJ20250604181211016 and SYSPG20241211174032004.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025).

[2] Xinru Chen, Chengbo Dong, Jiaqi Ji, Juan Cao, and Xirong Li. 2021. Image manipulation detection by multi-view multi-scale supervision. In Proceedings of the IEEE/CVF international conference on computer vision. 14185–14193.

[3] Zhongxi Chen, Shen Chen, Taiping Yao, Ke Sun, Shouhong Ding, Xianming Lin, Liujuan Cao, and Rongrong Ji. 2024. Enhancing tampered text detection through frequency feature fusion and decomposition. In European Conference on Computer Vision. Springer, 200–217.

[4] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. 2022. Masked-attention mask transformer for universal image segmentation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 1290–1299.

[5] Tiago José De Carvalho, Christian Riess, Elli Angelopoulou, Helio Pedrini, and Anderson de Rezende Rocha. 2013. Exposing digital image forgeries by illumination color classification. IEEE transactions on information forensics and security 8, 7 (2013), 1182–1194.

[6] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. 2025. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 (2025).

[7] Jing Dong, Wei Wang, and Tieniu Tan. 2013. Casia image tampering detection evaluation database. In 2013 IEEE China summit and international conference on signal and information processing. IEEE, 422–426.

[8] Li Dong, Weipeng Liang, and Rangding Wang. 2024. Robust text image tampering localization via forgery traces enhancement and multiscale attention. IEEE Transactions on Consumer Electronics 70, 1 (2024), 3495–3507.

[9] Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, and Luisa Verdoliva. 2023. Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20606–20615.

[10] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. Iclr 1, 2 (2022), 3.

[11] Zhenglin Huang, Jinwei Hu, Xiangtai Li, Yiwei He, Xingyu Zhao, Bei Peng, Baoyuan Wu, Xiaowei Huang, and Guangliang Cheng. 2025. Sida: Social media image deepfake detection, localization and explanation with large multimodal model. In Proceedings of the Computer Vision and Pattern Recognition Conference. 28831–28841.

[12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).

[13] Myung-Joon Kwon, In-Jae Yu, Seung-Hun Nam, and Heung-Kyu Lee. 2021. CAT-Net: Compression artifact tracing network for detection and localization of image splicing. In Proceedings of the IEEE/CVF winter conference on applications ofcomputer vision. 375–384.

[14] Black Forest Labs. 2024. FLUX. https://github.com/black-forest-labs/flux.

[15] Songze Li, Yunfei Guo, Shen Chen, Bin Li, Kaiqing Lin, Changsheng Chen, Haodong Li, Taiping Yao, and Shouhong Ding. 2025. DITL2: Dual-Stage Invariance Transfer Learning for Generalizable Document Image Tampering Local ization. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 82–91.

[16] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowl edge. https://llava-vl.github.io/blog/2024-01-30-llava-next/

[17] Jiawei Liu, Fanrui Zhang, Jiaying Zhu, Esther Sun, Qiang Zhang, and Zheng-Jun Zha. 2024. Forgerygpt: Multimodal large language model for explainable image forgery detection and localization. arXiv preprint arXiv:2410.10238 (2024)

[18] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. 2025. Step1x-edit: A practical

framework for general image editing. arXiv preprint arXiv:2504.17761 (2025).

[19] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. 2021. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision. 10012–10022.

[20] Dongliang Luo, Yuliang Liu, Rui Yang, Xianjin Liu, Jishen Zeng, Yu Zhou, and Xiang Bai. 2025. Toward real text manipulation detection: New dataset and new solution. Pattern Recognition 157 (2025), 110828.

[21] Tian-Tsong Ng, Jessie Hsu, and Shih-Fu Chang. 2009. Columbia image splicing detection evaluation dataset. DVMM lab. Columbia Univ CalPhotos Digit Libr (2009).

[22] Adam Novozamsky, Babak Mahdian, and Stanislav Saic. 2020. IMD2020: A largescale annotated dataset tailored for detecting manipulated images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision workshops. 71–80.

[23] Chenfan Qu, Chongyu Liu, Yuliang Liu, Xinhong Chen, Dezhi Peng, Fengjun Guo, and Lianwen Jin. 2023. Towards robust tampered text detection in document image: New dataset and new solution. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5937–5946

[24] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. 2025. DINOv3. arXiv:2508.10104 [cs.CV] https://arxiv.org/abs/2508.10104

[25] Lei Su, Xiaochen Ma, Xuekang Zhu, Chaoqun Niu, Zeyu Lei, and Ji-Zhe Zhou. 2025. Can we get rid of handcrafted feature extractors? sparsevit: Nonsemanticscentered, parameter-eficient image manipulation localization through sparecoding transformer. In Proceedings of the AAAI conference on artificial intelligence, Vol. 39. 7024–7032.

[26] Hao Sun, Yanbo Wang, Junxian Duan, and Ran He. 2025. VeriChain: Reinforced Document Image Forgery Verification via Self-revealing Reasoning Chain. In Asian Conference on Pattern Recognition. Springer, 238–249.

[27] Zhihao Sun, Haoran Jiang, Haoran Chen, Yixin Cao, Xipeng Qiu, Zuxuan Wu, and Yu-Gang Jiang. 2024. Forgerysleuth: Empowering multimodal large language models for image manipulation detection. arXiv preprint arXiv:2411.19466 (2024).

[28] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023).

[29] Peng Wang, Yichun Shi, Xiaochen Lian, Zhonghua Zhai, Xin Xia, Xuefeng Xiao, Weilin Huang, and Jianchao Yang. 2025. Seededit 3.0: Fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083 (2025).

[30] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265 (2025).

[31] Bihan Wen, Ye Zhu, Ramanathan Subramanian, Tian-Tsong Ng, Xuanjing Shen, and Stefan Winkler. 2016. COVERAGE–A novel database for copy-move forgery detection. In 2016 IEEE international conference on image processing (ICIP). Ieee, 161–165.

[32] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Shengming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. 2025. Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025).

[33] Guitao Xu, Ziqi Yi, Peirong Zhang, Jiahuan Cao, Shihang Wu, and Lianwen Jin. 2025. From Pixels to Semantics: A Novel MLLM-Driven Approach for Explainable Tampered Text Detection. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 757–766.

[34] Zhipei Xu, Xuanyu Zhang, Runyi Li, Zecheng Tang, Qing Huang, and Jian Zhang. 2024. Fakeshield: Explainable image forgery detection and localization via multimodal large language models. arXiv preprint arXiv:2410.02761 (2024).

[35] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675 (2019).

[36] Shiqiang Zheng, Changsheng Chen, Shen Chen, Taiping Yao, Shouhong Ding, Bin Li, and Jiwu Huang. 2025. Generalized document tampering localization via color and semantic disentanglement. IEEE Transactions on Circuits and Systems for Video Technology (2025).