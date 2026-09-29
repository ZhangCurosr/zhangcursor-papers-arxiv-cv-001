# Clinical Trajectory Alignment for Medical Vision-Language Pre-training

Huimin Yan<sup>1</sup>, Xian Yang<sup>2</sup>, Zhi Wang<sup>1</sup>, Liang Bai<sup>1</sup>

<sup>1</sup>Institute of Intelligent Information Processing, Shanxi University, Taiyuan, China <sup>2</sup>Alliance Manchester Business School, The University of Manchester, Manchester, UK

yanhm0925@163.com, xian.yang@manchester.ac.uk 202322407045@email.sxu.edu.cn, bailiang@sxu.edu.cn

## Abstract

Medical vision-language pre-training largely follows a visit-level image-report matching paradigm, aligning paired images and reports at individual visits. While effective for static cross-modal correspondence, this paradigm provides limited supervision for longitudinal clinical change, such as whether abnormalities improve, remain stable, or worsen over time. Learning such change is challenging because temporal semantics are implicit in free-text reports, and different abnormalities within the same patient may evolve asynchronously or even in opposite directions. We propose MedCTA, which reframes medical vision-language pretraining from visit-level cross-modal matching to learning clinical change. Rather than compressing a patient history into a single temporal representation, MedCTA models clinical change at two complementary scopes. At the abnormality scope, clinically grounded queries construct abnormality-conditioned visual and textual trajectories to capture heterogeneous abnormality evolution. At the patient-course scope, global image and report sequences are modeled to capture overall clinical progression beyond any individual abnormality. Structured trend supervision is extracted from longitudinal reports by an offline LLM parser, removing the need for manual temporal annotations. Combined with static image-report alignment, Med-CTA learns representations that preserve visit-level cross-modal correspondence while encoding longitudinal change semantics. Experiments on temporal image classification, image-text retrieval, and zero-shot classification show consistent gains over strong medical vision-language baselines.

## 1 Introduction

Medical vision–language pre-training (VLP) aims to learn clinically meaningful cross-modal representations from medical images and their corresponding reports. Existing methods based on image-text contrastive learning [1], masked modeling [2], and local region-text matching [3] have achieved notable progress in image-text retrieval, disease classification, and phrase grounding [4]. However, most existing methods largely follow a visit-level image-report matching paradigm, where paired images and reports are aligned at individual visits [5, 6]. Such supervision is effective for learning static cross-modal correspondence, but provides limited guidance for longitudinal clinical change, such as whether abnormalities improve, remain stable, or worsen over time [7, 8]. As a result, the temporal evolution semantics contained in patient follow-up records are not explicitly modeled, limiting the ability of current models to learn abnormality progression from longitudinal medical data, as illustrated in Figure 1.

Learning longitudinal clinical change is challenging for two main reasons. First, temporal semantics are often implicit in free-text radiology reports rather than available as structured annotations.

![](images/d22391b302d788aef61fb166fb7b0cf90e9dc3774ff78193adbed2a9cd13aed4.jpg)  
Figure 1: Motivation of MedCTA. (a) Static image–report matching captures visit-level correspondence but overlooks longitudinal clinical change. (b) MedCTA exploits follow-up records with LLM-derived trend supervision to model abnormality-specific and patient-course trajectories.

Reports often describe abnormality changes through expressions such as “increased opacity”, “interval improvement”, or “no significant change”, but these descriptions cannot be directly used as supervision without clinical semantic parsing [9, 10]. Second, abnormality progression is inherently heterogeneous. Different abnormalities within the same patient may evolve asynchronously or even in opposite directions. Therefore, compressing an entire patient history into a single temporal feature may oversimplify longitudinal abnormality evolution and fail to capture abnormality-specific changes [11, 12].

To address these challenges, we propose MedCTA, a clinical trajectory alignment framework that reframes medical vision–language pre-training from visit-level cross-modal matching to modeling clinical change. Instead of compressing a patient’s longitudinal history into a single temporal representation, MedCTA explicitly models longitudinal changes at two complementary scopes: abnormality-specific evolution and patient-course progression. At the abnormality scope, Med-CTA constructs abnormality-conditioned visual and textual trajectories across visits using clinically grounded abnormality queries. These trajectories capture the evolution of each abnormality over time and are supervised by abnormality-specific trend labels through Abnormality-Specific Trajectory Alignment. At the patient-course scope, MedCTA further models global image and report sequences across visits through Patient-Course Progression Alignment, thereby capturing the patient’s overall clinical progression beyond any individual abnormality. To train these alignment objectives without manual trend annotations, MedCTA adopts an offline large language model to extract structured trend supervision from longitudinal report sequences, including abnormality-specific evolution trends and patient-course progression trends. By jointly optimizing static visit-level image–report alignment, Abnormality-Specific Trajectory Alignment, and Patient-Course Progression Alignment, MedCTA learns cross-modal representations that preserve single-visit image-text correspondence while capturing longitudinal abnormality evolution and patient-course progression.

The main contributions of this paper are summarized as follows:

• We reformulate longitudinal medical VLP as learning from clinical change, extending static visit-level image–report matching to trajectory-level temporal semantic supervision.

• We propose MedCTA, a trajectory alignment framework that models longitudinal change through abnormality-specific trajectories and patient-course progression, thereby capturing heterogeneous abnormality evolution and overall patient-level clinical progression.

• We derive structured trend supervision using an offline LLM parser and demonstrate consistent improvements on temporal classification, image-text retrieval, and zero-shot classification.

## 2 Related Work

## 2.1 Medical VLP

Medical VLP learns transferable cross-modal representations from medical images and radiology reports [13, 14]. Existing methods have made progress through image–text contrastive learning [15,

16], masked modeling [17], and fine-grained region–text alignment [3, 6], benefiting tasks such as image–text retrieval, disease classification, phrase grounding, and report understanding [18, 19]. However, most methods follow a visit-level image–report matching paradigm, where each pair is treated as an independent static sample. This paradigm learns effective single-visit correspondence but provides limited supervision for longitudinal clinical change, such as improvement, stability, or worsening. Different from prior static medical VLP methods, we reframe pre-training from visit-level cross-modal matching to clinical change learning from longitudinal patient trajectories.

## 2.2 Longitudinal Temporal Modeling in Medical VLP

Longitudinal radiology data contain temporal cues about clinical progression across follow-up examinations [8, 20, 21]. Recent methods have explored such information: ALTA [22] uses prior X-ray views as temporal context, Med-ST [23] introduces cross-modal cycle consistency with historical image–text pairs, and TAMM [24] leverages LLM-derived supervision from adjacent reports. However, these methods mainly capture visual context, temporal correspondence, or shortrange local transitions, and thus may overlook heterogeneous abnormality evolution and global patient-course progression. Our method addresses this limitation by modeling clinical change through abnormality-specific trajectories and patient-course progression.

## 3 Method

## 3.1 From Static Pair Matching to Change-Aware Pre-training

Existing medical VLP methods typically align paired images and reports at individual visits. While effective for static cross-modal correspondence, this visit-level matching paradigm provides limited supervision for longitudinal clinical change, such as whether abnormalities improve, remain stable, or worsen over time. MedCTA reframes medical VLP from static pair matching to change-aware pre-training by modeling how clinical states evolve along patient trajectories.

Formally, we consider a longitudinal medical VLP setting, where each patient is associated with a temporally ordered sequence of image–report pairs:

$$
\mathcal { D } = \{ S _ { i } \} _ { i = 1 } ^ { N } , \qquad S _ { i } = \{ ( I _ { i , t } , R _ { i , t } , \tau _ { i , t } ) \} _ { t = 1 } ^ { T _ { i } } ,\tag{1}
$$

where $I _ { i , t }$ and $R _ { i , t }$ denote the medical image and the corresponding radiology report of patient i at the t-th visit, $\tau _ { i , t }$ is the examination timestamp, and $T _ { i }$ is the number of visits.

MedCTA retains static image–report alignment to preserve visit-level cross-modal correspondence. For a batch of B image–report pairs, let $\mathbf { v } _ { b }$ and $\mathbf { r } _ { b }$ denote the global image and report representations of the b-th pair, respectively. The static image–report alignment loss is defined as

$$
\mathcal { L } _ { \mathrm { c l i p } } = \frac { 1 } { 2 } ( \mathcal { L } _ { I  T } + \mathcal { L } _ { T  I } ) , \quad \mathcal { L } _ { I  T } = - \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \log \frac { \exp ( \sin ( \mathbf { v } _ { b } , \mathbf { r } _ { b } ) / \tau ) } { \sum _ { j = 1 } ^ { B } \exp ( \sin ( \mathbf { v } _ { b } , \mathbf { r } _ { j } ) / \tau ) } ,\tag{2}
$$

where $\mathcal { L } _ { T  I }$ is symmetric, sim(·, ·) is cosine similarity, and τ is the temperature.

However, ${ \mathcal L } _ { \mathrm { c l i p } }$ only learns visit-level state matching and cannot explicitly supervise longitudinal change. Therefore, MedCTA further introduces two change-aware objectives: Abnormality-Specific Trajectory Alignment for heterogeneous abnormality evolution and Patient-Course Progression Alignment for global clinical progression, as shown in Figure 2.

## 3.2 Structured Trend Supervision from Longitudinal Reports

The temporal semantics of longitudinal radiology reports are usually implicit in free-text reports and cannot be directly used as structured supervision. To obtain learnable temporal supervision, we use an LLM as an offline parser to extract structured trend labels from complete longitudinal report sequences. For patient $i ,$ the longitudinal report sequence is

$$
{ \mathcal { R } } _ { i } = \{ R _ { i , 1 } , R _ { i , 2 } , \ldots , R _ { i , T _ { i } } \} .\tag{3}
$$

Specifically, MedCTA uses the same offline LLM parser $\mathcal { G } _ { \mathrm { L L M } }$ with task-specific prompts to derive two types of trend labels from $\mathcal { R } _ { i } \mathrm { : }$ abnormality-specific trend labels and patient-course trend labels.

![](images/9404f865af3432ccd89517f162872d42446b6c85aaaf9f390476c1dc9e536aa8.jpg)  
Figure 2: Overview of MedCTA. Given a longitudinal image–report sequence $s _ { i }$ , MedCTA jointly performs static image–report alignment, Abnormality-Specific Trajectory Alignment, and Patient-Course Progression Alignment. Abnormality-specific and patient-course trajectories are temporally encoded by Mamba and constrained by LLM-derived trend supervision, enabling semantic-level trajectory alignment for abnormality evolution and overall clinical progression.

Given a fixed abnormality set $\mathcal { A } = \{ a _ { k } \} _ { k = 1 } ^ { K }$ , the abnormality-specific trend label is generated as

$$
y _ { i , k } ^ { a } = \mathcal { G } _ { \mathrm { L L M } } \big ( \mathcal { R } _ { i } , a _ { k } ; \pi _ { \mathrm { a b n } } \big ) , \qquad y _ { i , k } ^ { a } \in \{ - 1 , 0 , 1 \} ,\tag{4}
$$

where $\pi _ { \mathrm { a b n } }$ denotes the abnormality-level prompting instruction. The labels −1, 0, and 1 indicate worsening, stable condition, and improvement, respectively. Since not all abnormalities appear in each patient sequence, we use a validity mask $m _ { i , k } \in \{ 0 , 1 \}$ , where $m _ { i , k } = 1$ if abnormality $a _ { k }$ has an available trend label for patient i, and $m _ { i , k } = 0$ otherwise.

In parallel, the patient-course trend label is generated by the same LLM parser with a patient-level prompt:

$$
y _ { i } ^ { p } = \mathcal { G } _ { \mathrm { L L M } } ( \mathcal { R } _ { i } ; \pi _ { \mathrm { p a t } } ) , \qquad y _ { i } ^ { p } \in \{ - 1 , 0 , 1 \} ,\tag{5}
$$

where $\pi _ { \mathrm { p a t } }$ denotes the patient-course prompting instruction. This label provides sequence-level supervision for patient-course progression modeling.

## 3.3 Abnormality-Conditioned Trajectory Construction

To model the longitudinal evolution of individual abnormalities, abnormality queries serve as semantic selectors that identify abnormality-relevant visual and textual representations at each visit. These visit-wise representations are then organized across time to form abnormality-conditioned visual and textual trajectories. Given a predefined abnormality set $\mathcal { A } = \{ a _ { k } \} _ { k = 1 } ^ { K }$ , we associate each abnormality $a _ { k }$ with a learnable query $\mathbf { q } _ { k } \in \mathbb { R } ^ { d }$ , and collect all queries as $\tilde { \mathbf { Q } } = [ \mathbf { q } _ { 1 } ; \mathbf { q } _ { 2 } ; \dots ; \mathbf { q } _ { K } ] \in \mathbb { R } ^ { K \times d }$ . To inject clinical semantic priors, we initialize each query using the global [CLS] embedding of an abnormality-specific prompt $P _ { k } , { \bf e . g . , \ " { a } }$ clinical finding of ${ a _ { k } } ^ { " }$ , encoded by the text encoder [25]:

$$
{ \bf q } _ { k } ^ { ( 0 ) } = f _ { \mathrm { t e x t } } ^ { \mathrm { C L S } } ( P _ { k } ) , \qquad k = 1 , \ldots , K .\tag{6}
$$

The initialized query ${ \bf q } _ { k } ^ { \left( 0 \right) }$ is then treated as the initial value of the learnable parameter $\mathbf { q } _ { k }$ and is updated during pre-training.

To adapt the query to different modalities, we project it into image and text spaces:

$$
\mathbf { q } _ { k } ^ { I } = \mathbf { W } _ { I } \mathbf { q } _ { k } , \qquad \mathbf { q } _ { k } ^ { T } = \mathbf { W } _ { T } \mathbf { q } _ { k } ,\tag{7}
$$

where $\mathbf { W } _ { I }$ and $\mathbf { W } _ { T }$ are modality-specific projection matrices.

For each visit $( I _ { i , t } , R _ { i , t } )$ , the image encoder $f _ { \mathrm { i m a g e } }$ [26] and text encoder $f _ { \mathrm { t e x t } }$ [25] produce global features for visit-level alignment and local features for abnormality-specific extraction:

$$
\begin{array} { r } { f _ { \mathrm { i m a g e } } : I _ { i , t }  ( \mathbf { g } _ { i , t } ^ { I } , \mathbf { H } _ { i , t } ^ { I } ) , \quad f _ { \mathrm { t e x t } } : R _ { i , t }  ( \mathbf { g } _ { i , t } ^ { T } , \mathbf { H } _ { i , t } ^ { T } ) . } \end{array}\tag{8}
$$

where $\mathbf { g } _ { i , t } ^ { I }$ and $\mathbf { g } _ { i , t } ^ { T }$ denote the global image and text representations, respectively, while $\mathbf { H } _ { i , t } ^ { I }$ and $\mathbf { H } _ { i , t } ^ { T }$ denote the patch-level image features and token-level report features, respectively.

The modality-specific queries attend to local features to obtain abnormality-specific visual and textual states:

$$
\begin{array} { r } { \mathbf { e } _ { i , t , k } ^ { I } = \mathrm { A t t n } ( \mathbf { q } _ { k } ^ { I } , \mathbf { H } _ { i , t } ^ { I } , \mathbf { H } _ { i , t } ^ { I } ) , \qquad \mathbf { e } _ { i , t , k } ^ { T } = \mathrm { A t t n } ( \mathbf { q } _ { k } ^ { T } , \mathbf { H } _ { i , t } ^ { T } , \mathbf { H } _ { i , t } ^ { T } ) , } \end{array}\tag{9}
$$

where $\mathbf { e } _ { i , t , k } ^ { I }$ and $\mathbf { e } _ { i , t , k } ^ { T }$ denote the image-side and text-side abnormality-specific embeddings of $a _ { k }$ at visit $t ,$ respectively. In this way, the image and text representations are organized into abnormalityconditioned semantic spaces.

Given the visit-wise abnormality-conditioned representations, MedCTA constructs visual and textual trajectories for each abnormality by organizing these representations across visits. For patient i and abnormality $a _ { k }$ , the resulting image-side and text-side abnormality trajectories are defined as:

$$
\boldsymbol { \mathcal { E } } _ { i , k } ^ { I } = \{ \mathbf { e } _ { i , 1 , k } ^ { I } , \mathbf { e } _ { i , 2 , k } ^ { I } , \ldots , \mathbf { e } _ { i , T _ { i } , k } ^ { I } \} , \qquad \boldsymbol { \mathcal { E } } _ { i , k } ^ { T } = \{ \mathbf { e } _ { i , 1 , k } ^ { T } , \mathbf { e } _ { i , 2 , k } ^ { T } , \ldots , \mathbf { e } _ { i , T _ { i } , k } ^ { T } \} .\tag{10}
$$

## 3.4 Abnormality-Specific Trajectory Alignment

Given the abnormality-conditioned trajectories, we use modality-specific Mamba-based temporal encoders [27] to encode the image-side and text-side evolution of each abnormality:

$$
\begin{array} { r } { \mathbf { z } _ { i , k } ^ { I , a } = g _ { \mathrm { M a m b a } } ^ { I , a } ( \mathcal { E } _ { i , k } ^ { I } ) , \qquad \mathbf { z } _ { i , k } ^ { T , a } = g _ { \mathrm { M a m b a } } ^ { T , a } ( \mathcal { E } _ { i , k } ^ { T } ) , } \end{array}\tag{11}
$$

where $\mathbf { z } _ { i , k } ^ { I , a }$ and $\mathbf { z } _ { i , k } ^ { T , a }$ denote the temporally encoded image-side and text-side trajectory representations of abnormality $a _ { k }$ for patient i, respectively. Padding masks are applied during temporal encoding to handle variable-length visit sequences across patients.

Instead of directly matching image and text trajectory embeddings, MedCTA aligns them in a shared clinical-change space supervised by the LLM-derived abnormality trend label $y _ { i , k } ^ { a }$ . The image-side and text-side trend predictions are computed as

$$
\begin{array} { r } { \hat { \mathbf { p } } _ { i , k } ^ { I , a } = \mathrm { s o f t m a x } ( h _ { I } ^ { a } ( \mathbf { z } _ { i , k } ^ { I , a } ) ) , \qquad \hat { \mathbf { p } } _ { i , k } ^ { T , a } = \mathrm { s o f t m a x } ( h _ { T } ^ { a } ( \mathbf { z } _ { i , k } ^ { T , a } ) ) , } \end{array}\tag{12}
$$

where $h _ { I } ^ { a } ( \cdot )$ and $h _ { T } ^ { a } ( \cdot )$ are image-side and text-side abnormality-specific trend classification heads. The abnormality-specific trajectory alignment loss is defined as

$$
\mathcal { L } _ { \mathrm { a b n } } = \frac { 1 } { 2 \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } m _ { i , k } } \sum _ { i = 1 } ^ { N } \sum _ { k = 1 } ^ { K } m _ { i , k } \left[ \mathrm { C E } \left( \hat { \mathbf { p } } _ { i , k } ^ { I , a } , y _ { i , k } ^ { a } \right) + \mathrm { C E } \left( \hat { \mathbf { p } } _ { i , k } ^ { T , a } , y _ { i , k } ^ { a } \right) \right] ,\tag{13}
$$

where $\operatorname { C E } ( \cdot , \cdot )$ denotes the cross-entropy loss, and $m _ { i , k }$ is a validity mask indicating whether abnormality $a _ { k }$ has an available trend label for patient i. The label $y _ { i , k } ^ { a }$ is mapped to a three-class trend category, corresponding to worsening, stable, and improving conditions. This objective encourages image and text trajectories of the same abnormality to be consistent at the trend-semantic level.

## 3.5 Patient-Course Progression Alignment

While abnormality-specific trajectories capture fine-grained evolution, they may not fully reflect the overall patient course. MedCTA therefore models patient-course progression from global longitudinal image and text representations.

For patient i, we organize the visit-level global representations from Eq. (8) into image and text sequences:

$$
\mathcal { U } _ { i } ^ { I } = \{ \mathbf { g } _ { i , t } ^ { I } \} _ { t = 1 } ^ { T _ { i } } , \qquad \mathcal { U } _ { i } ^ { T } = \{ \mathbf { g } _ { i , t } ^ { T } \} _ { t = 1 } ^ { T _ { i } } .\tag{14}
$$

To capture long-range patient-course progression, we perform Mamba-based temporal modeling [27] over the global image and text sequences:

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { I , p } = g _ { \mathrm { M a m b a } } ^ { I , p } ( \mathcal { U } _ { i } ^ { I } ) , \qquad \mathbf { z } _ { i } ^ { T , p } = g _ { \mathrm { M a m b a } } ^ { T , p } ( \mathcal { U } _ { i } ^ { T } ) . } \end{array}\tag{15}
$$

where $\mathbf { z } _ { i } ^ { I , p }$ and ${ \bf z } _ { i } ^ { T , p }$ denote the image-side and text-side patient-course progression representations. The learned patient-course representations are supervised by the LLM-derived patient-course trend label $y _ { i } ^ { p }$ . The image-side and text-side predictions are computed as

$$
\hat { \mathbf { p } } _ { i } ^ { I , p } = \mathrm { s o f t m a x } ( h _ { I } ^ { p } ( \mathbf { z } _ { i } ^ { I , p } ) ) , \qquad \hat { \mathbf { p } } _ { i } ^ { T , p } = \mathrm { s o f t m a x } ( h _ { T } ^ { p } ( \mathbf { z } _ { i } ^ { T , p } ) ) ,\tag{16}
$$

where $h _ { I } ^ { p } ( \cdot )$ and $h _ { T } ^ { p } ( \cdot )$ are patient-course trend classification heads. The patient-course progression alignment loss is defined as

$$
\mathcal { L } _ { \mathrm { p a t } } = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \left[ \mathrm { C E } \left( \hat { \mathbf { p } } _ { i } ^ { I , p } , y _ { i } ^ { p } \right) + \mathrm { C E } \left( \hat { \mathbf { p } } _ { i } ^ { T , p } , y _ { i } ^ { p } \right) \right] ,\tag{17}
$$

where $y _ { i } ^ { p }$ is mapped to a three-class label corresponding to overall worsening, stable condition, and improvement. This loss encourages the model to capture the overall clinical progression of each patient across longitudinal visits, thereby enabling patient-course progression alignment.

## 3.6 Joint Optimization with Static Image–Report Alignment

MedCTA is trained by jointly optimizing static image–report alignment, abnormality-specific trajectory alignment, and patient-course progression alignment:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c l i p } } + \mathcal { L } _ { \mathrm { a b n } } + \mathcal { L } _ { \mathrm { p a t } } ,\tag{18}
$$

where $\mathcal { L } _ { \mathrm { c l i p } }$ preserves visit-level cross-modal correspondence, $\mathcal { L } _ { \mathrm { a b n } }$ supervises abnormality-specific evolution, and $\mathcal { L } _ { \mathrm { p a t } }$ models overall patient-course progression.

## 4 Experiments

## 4.1 Experimental Setting

We evaluate MedCTA on both longitudinal temporal reasoning and cross-modal understanding tasks. Below, we briefly describe the pre-training data, downstream benchmarks, and baselines. Additional implementation and evaluation details are provided in Appendix A.

Pre-training Data. We pre-train MedCTA on MIMIC-CXR [28] and reorganize image–report pairs into patient-level longitudinal sequences ordered by study time. Following prior temporal VLP settings [23], each sequence contains up to four consecutive visits.

Downstream Tasks. We evaluate MedCTA on temporal reasoning and cross-modal understanding tasks. Temporal reasoning includes temporal image classification, temporal sentence similarity classification, and dynamic phrase grounding on Chest ImaGenome [29] and MS-CXR-T [20]. Cross-modal understanding includes image–text retrieval on MIMIC-5×200 [15] and zero-shot classification on COVIDx [30] and RSNA Pneumonia [31].

Baselines. We compare MedCTA with representative medical vision–language methods under the same evaluation settings. The baselines include non-temporal VLP methods, i.e., MGCA [32], MedCLIP [15], CARZero [16], PRIOR [17], and MAVL [33], as well as temporal medical VLP methods, i.e., Med-ST [23], ALTA [22], and TAMM [24]. When official code is available, we reproduce results using the released implementations; otherwise, we follow the original protocols with the same backbone and training settings for fair comparison.

## 4.2 Experimental Results

To comprehensively evaluate MedCTA, we divide our evaluation into two categories: temporal trend understanding tasks and standard vision-language benchmarks.

## 4.2.1 Temporal Trend Understanding Tasks

Temporal Image Classification. To evaluate temporal progression modeling from longitudinal chest X-rays, we fine-tune on Chest ImaGenome and evaluate temporal image classification on MS-CXR-T over five abnormalities: Consolidation, Edema, Pleural Effusion, Pneumonia, and Pneumothorax. The task predicts one of three progression categories, i.e., improving, stable, and worsening, for each abnormality. As shown in Table 1, MedCTA achieves superior performance across all diseases. This indicates that explicitly modeling longitudinal clinical change at both abnormality and patient-course levels provides more discriminative representations than visit-level image–report matching alone, especially for distinguishing improvement, stability, and worsening.

Table 1: Temporal image classification results. Accuracy (%) is reported across five abnormality categories. Best results are shown in bold, and second-best results are underlined.
<table><tr><td>Method</td><td>Consolidation</td><td>Edema</td><td>Pl. Effusion</td><td>Pneumonia</td><td>Pneumothorax</td></tr><tr><td>MGCA</td><td> $4 4 . 9 9 \pm 0 . 4 7$ </td><td> $6 2 . 7 1 \pm 0 . 2 4$ </td><td> $5 7 . 1 7 \pm 0 . 6 9$ </td><td> $6 3 . 7 3 \pm 0 . 8 9$ </td><td> $5 2 . 1 6 \pm 1 . 0 2$ </td></tr><tr><td>MedCLIP</td><td> $5 3 . 4 0 \pm 0 . 8 7$ </td><td> $4 2 . 8 5 \pm 0 . 0 1$ </td><td> $4 9 . 9 6 \pm 0 . 2 8$ </td><td> $6 7 . 1 0 \pm 0 . 2 2$ </td><td> $5 5 . 4 6 \pm 0 . 0 2$ </td></tr><tr><td>PRIOR</td><td> $4 1 . 8 5 \pm 1 . 5 1$ </td><td> $4 3 . 9 7 \pm 1 . 0 3$ </td><td> $5 9 . 2 1 \pm 1 . 1 9$ </td><td> $4 1 . 4 1 \pm 3 . 2 4$ </td><td> $4 5 . 3 3 \pm 2 . 1 3$ </td></tr><tr><td>MAVL</td><td> $4 3 . 7 9 \pm 0 . 0 0$ </td><td> $4 2 . 8 5 \pm 0 . 0 0$ </td><td> $4 8 . 6 6 \pm 0 . 0 0$ </td><td> $6 7 . 1 0 \pm 0 . 0 0$ </td><td> $5 5 . 4 6 \pm 0 . 0 0$ </td></tr><tr><td>CARZero</td><td> $4 8 . 9 5 \pm 1 . 2 5$ </td><td> $5 1 . 9 2 \pm 1 . 6 4$ </td><td> $5 0 . 1 9 \pm 1 . 7 2$ </td><td> $5 7 . 4 4 \pm 1 . 3 5$ </td><td> $4 7 . 8 4 \pm 1 . 7 8$ </td></tr><tr><td>Med-ST</td><td> $6 0 . 5 7 \pm 1 . 1 8$ </td><td> $6 7 . 3 5 \pm 0 . 3 2$ </td><td> $5 8 . 4 7 \pm 1 . 5 0$ </td><td> $6 5 . 0 0 \pm 0 . 3 4$ </td><td> $5 4 . 1 8 \pm 0 . 8 1$ </td></tr><tr><td>ALTA</td><td> $5 2 . 7 6 \pm 1 . 8 6$ </td><td> $6 6 . 9 2 \pm 0 . 8 9$ </td><td> $5 0 . 9 2 \pm 1 . 7 2$ </td><td> $6 6 . 8 2 \pm 0 . 1 9$ </td><td> $5 1 . 0 4 \pm 0 . 4 1$ </td></tr><tr><td>TAMM</td><td> $6 5 . 5 1 \pm 0 . 8 3$ </td><td> $6 9 . 2 2 \pm 0 . 0 1$ </td><td> $6 3 . 1 8 \pm 0 . 4 5$ </td><td> ${ \underline { { 7 2 . 7 1 } } } \pm 0 . 7 7$ </td><td> $6 0 . 5 0 \pm 0 . 0 1$ </td></tr><tr><td>Ours</td><td> $\overline { { { \bf 6 7 . 1 4 } \pm { \bf 0 . 8 2 } } }$ </td><td> $\overline { { 7 0 . 3 0 \pm 0 . 1 6 } }$ </td><td> $\pm { \bf 5 . 5 9 } \pm { \bf 1 . 0 3 }$ </td><td> $\overline { { 7 2 . 8 5 \pm 0 . 3 5 } }$ </td><td> $\mathbf { 6 1 . 6 6 \pm 0 . 0 5 }$ </td></tr></table>

## Temporal Sentence Similarity Classifica

tion. To evaluate the temporal language understanding ability of MedCTA, we conduct zero-shot sentence similarity classification on the MS-CXR-T dataset. In particular, this experiment examines the text encoder of each model to determine how well it captures semantic shifts and temporal progression in textual descriptions. Fig. 3 shows that Med-CTA achieves the best zero-shot accuracy of 95.72%. This result indicates that Med-CTA learns more temporally aware textual representations, enabling better recognition of progression-related semantics such as improvement, stability, and worsening in reports.

![](images/40744dac7c5b8383aeab339b08f28670dae4fd3307bfb2c3de8a206db1b9927f.jpg)  
Figure 3: Accuracy (%) for temporal sentence similarity classification.

## Dynamic Phrase Grounding. We evaluate

dynamic phrase grounding using Chest ImaGenome bounding-box annotations, aiming to localize temporal progression descriptions across two consecutive chest X-rays. Specifically, we compute cosine similarity between patch-level image embeddings and temporal description embeddings, and upsample the patch-wise similarity scores to generate grounding heatmaps. Fig. 4 presents representative examples from three progression categories, i.e., improving, stable, and worsening, with the left and right images denoting the prior and current studies, respectively. The results show that MedCTA can better localize regions associated with longitudinal changes by organizing abnormality-relevant visual representations across visits.

## 4.2.2 Standard Vision-Language Benchmarks

Cross-modal Retrieval on MIMIC-5×200. We evaluate text-to-image and image-to-text retrieval on MIMIC-5×200. As shown in Table 2, MedCTA achieves the best results on seven of eight metrics and ranks second on text-to-image P@2. Fig. 5 shows qualitative image-to-text and text-to-image examples, where green/red boxes denote correct/incorrect retrievals and labels indicate abnormality categories. These results demonstrate that combining static alignment with abnormality-conditioned trajectories and patient-course progression improves cross-modal representation learning.

Zero-shot Classification Tasks. We evaluate the transferability of MedCTA on RSNA Pneumonia and COVIDx under a zero-shot setting. As shown in Table 3, MedCTA achieves the best results on both datasets, with 86.05% accuracy and 79.25% F1 on RSNA Pneumonia, and 95.66% accuracy and 88.48% F1 on COVIDx. This indicates that modeling longitudinal clinical change enhances visual representation learning. By integrating static image–report alignment with abnormality-conditioned trajectories and patient-course progression, MedCTA learns more transferable visual features for zero-shot disease classification.

Table 2: Cross-modal retrieval results on the MIMIC-CXR 5×200 benchmark. Performance is reported as precision (%) at top-k (P@k). Best results are shown in bold, and second-best results are underlined.
<table><tr><td></td><td colspan="4">Text → Image</td><td colspan="4">Image → Text</td></tr><tr><td>Method</td><td>P@1</td><td>P@2</td><td>P@5</td><td>P@10</td><td>P@1</td><td>P@2</td><td>P@5</td><td>P@10</td></tr><tr><td>MGCA</td><td>74.50</td><td>76.30</td><td>71.16</td><td>60.85</td><td>63.62</td><td>63.88</td><td>64.11</td><td>61.95</td></tr><tr><td>MedCLIP</td><td>45.75</td><td>47.10</td><td>48.63</td><td>43.07</td><td>50.31</td><td>48.37</td><td>48.30</td><td>48.09</td></tr><tr><td>PRIOR</td><td>47.13</td><td>48.11</td><td>47.53</td><td>47.24</td><td>49.50</td><td>52.55</td><td>51.95</td><td>36.55</td></tr><tr><td>MAVL</td><td>50.00</td><td>50.00</td><td>42.91</td><td>37.27</td><td>50.00</td><td>50.00</td><td>49.47</td><td>50.00</td></tr><tr><td>CARZero</td><td>52.44</td><td>50.00</td><td>49.47</td><td>50.01</td><td>50.00</td><td>50.00</td><td>51.65</td><td>47.38</td></tr><tr><td>Med-ST</td><td>75.25</td><td>76.18</td><td>71.36</td><td>65.04</td><td>64.50</td><td>63.97</td><td>62.48</td><td>59.00</td></tr><tr><td>ALTA</td><td>69.80</td><td>65.85</td><td>52.60</td><td>37.72</td><td>56.80</td><td>55.40</td><td>46.80</td><td>46.80</td></tr><tr><td>TAMM</td><td>78.69</td><td>79.26</td><td>73.07</td><td>64.03</td><td>66.44</td><td>67.63</td><td>66.05</td><td>62.50</td></tr><tr><td>Ours</td><td>80.24</td><td>79.08</td><td>75.68</td><td>65.82</td><td>69.13</td><td>68.63</td><td>67.23</td><td>63.01</td></tr></table>

![](images/1179830fdbf08ca606a3b5d1e266a12e541d4a93f655006fbbe024575eb18f1a.jpg)  
Figure 4: Qualitative dynamic phrase grounding results on Chest ImaGenome for improving, stable, and worsening cases. The left and right images denote the prior and current studies, respectively, with highlighted regions indicating visual evidence related to temporal progression.

Table 3: Zero-shot classification results on the RSNA Pneumonia and COVIDx datasets. Performance is reported in terms of Accuracy (ACC, %) and F1 (%). Best results are shown in bold, and secondbest results are underlined.
<table><tr><td></td><td colspan="2">RSNA Pneumonia</td><td colspan="2">COVIDx</td></tr><tr><td>Method</td><td>ACC</td><td>F1</td><td>ACC</td><td>F1</td></tr><tr><td>MGCA</td><td> $8 5 . 0 6 \pm 0 . 0 5$ </td><td> $7 6 . 5 6 \pm 0 . 1 0$ </td><td> $9 2 . 1 3 \pm 0 . 1 0$ </td><td> $8 0 . 1 0 \pm 0 . 6 5$ </td></tr><tr><td>MedCLIP</td><td> $8 4 . 1 7 \pm 0 . 1 6$ </td><td> $7 7 . 3 8 \pm 0 . 3 5$ </td><td> $9 0 . 1 9 \pm 0 . 0 9$ </td><td> $7 5 . 6 0 \pm 2 . 1 0$ </td></tr><tr><td>PRIOR</td><td> $8 1 . 3 0 \pm 0 . 3 5$ </td><td> $7 2 . 0 2 \pm 1 . 0 9$ </td><td> $9 0 . 0 1 \pm 0 . 1 9$ </td><td> $8 7 . 9 7 \pm 0 . 6 8$ </td></tr><tr><td>MAVL</td><td> $7 9 . 9 9 \pm 0 . 8 3$ </td><td> $7 3 . 1 4 \pm 0 . 4 8$ </td><td> $8 8 . 7 7 \pm 0 . 9 2$ </td><td> $8 5 . 0 9 \pm 1 . 7 1$ </td></tr><tr><td>CARZero</td><td> $8 4 . 2 8 \pm 0 . 0 7$ </td><td> $7 6 . 9 4 \pm 0 . 4 0$ </td><td> $9 1 . 9 2 \pm 0 . 2 0$ </td><td> $8 5 . 8 3 \pm 0 . 9 8$ </td></tr><tr><td>Med-ST</td><td> $8 5 . 1 5 \pm 0 . 0 4$ </td><td> $7 6 . 9 6 \pm 0 . 2 5$ </td><td> $9 2 . 2 7 \pm 0 . 1 7$ </td><td> $8 0 . 2 2 \pm 0 . 3 5$ </td></tr><tr><td>ALTA</td><td> $8 4 . 2 4 \pm 0 . 2 0$ </td><td> $7 6 . 5 4 \pm 0 . 6 1$ </td><td> $9 1 . 5 3 \pm 0 . 1 9$ </td><td> $7 7 . 6 2 \pm 1 . 2 6$ </td></tr><tr><td>TAMM</td><td> $8 5 . 4 8 \pm 0 . 0 5$ </td><td> $7 8 . 2 1 \pm 0 . 1 4$ </td><td> $9 3 . 0 3 \pm 0 . 0 6$ </td><td> $8 7 . 3 2 \pm 0 . 3 8$ </td></tr><tr><td>Ours</td><td> $\mathbf { \overline { { 8 6 . 0 5 \pm 0 . 0 3 } } }$ </td><td> $\overline { { 7 9 . 2 5 \pm 0 . 0 8 } }$ </td><td> $\pm \mathbf { 0 . 6 6 \pm 0 . 1 2 }$ </td><td> $\mathbf { 8 8 . 4 8 \pm 0 . 1 6 }$ </td></tr></table>

## 4.3 Ablation Study

Contribution of Different Components. We ablate $\mathcal { L } _ { \mathrm { c l i p } } , \mathcal { L } _ { \mathrm { a b n } }$ , and $\mathcal { L } _ { \mathrm { p a t } }$ to assess the contribution of each objective. As shown in Table 4, removing any objective consistently reduces performance on temporal sentence similarity and temporal image classification, indicating their complementary effects. The full model performs best, demonstrating the effectiveness of combining static alignment with dual-scope clinical change supervision. More detailed ablations are reported in Appendix B.1.

![](images/b3550e84553abc91575343f6ecc0669e87535122ad1fb0865f0dfd1ecf56b359.jpg)  
Figure 5: Qualitative examples of image-to-text and text-to-image retrieval on MIMIC-5×200. For each method, we show the top two retrieved samples given an input chest X-ray or a query sentence. Green and red boxes denote correct and incorrect retrievals, respectively, with abnormality categories annotated below the incorrect samples.

Table 4: Ablation study on temporal tasks on the MS-CXR-T dataset. Accuracy (%) for temporal sentence similarity and temporal image classification. Relative changes from the full model are shown below.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Temporal Sentence Similarity</td><td colspan="5">Temporal Image Classification</td></tr><tr><td>Consolidation</td><td>Edema</td><td>Pleural Effusion</td><td>Pneumonia</td><td>Pneumothorax</td></tr><tr><td>Ours</td><td>95.72</td><td>67.14</td><td>70.30</td><td>65.59</td><td>72.85</td><td>61.66</td></tr><tr><td>w/o clip loss</td><td>92.24 (−3.48)</td><td>59.58</td><td>59.51</td><td>63.44</td><td>67.50</td><td>56.75</td></tr><tr><td>w/o abnormality loss</td><td>87.47 (−8.25)</td><td>(−7.56) 55.22</td><td>(−10.79) 51.46</td><td>(−2.15) 58.41</td><td>(−5.35) 65.24</td><td>(−4.91) 53.33</td></tr><tr><td>w/o patient loss</td><td>90.55 (−5.17)</td><td>(−11.92) 56.98 (−10.16)</td><td>(−18.84) 57.66 (−12.64)</td><td>(−7.18) 59.58 (−6.01)</td><td>(−7.61) 65.80 (−7.05)</td><td>(−8.33) 54.18 (−7.48)</td></tr></table>

Effect of Query Initialization. We compare random and clinically initialized abnormality queries in Appendix B.2. Clinical initialization consistently performs better, indicating that medical prompts provide useful semantic priors for constructing abnormality-specific trajectories.

Effect of Different LLM Parsers. MedCTA uses Qwen2-7B [34] as the offline parser for abnormality-specific and patient-course trend supervision. To assess parser robustness, we replace it with Llama 3-8B [35]. Appendix B.3 shows stable image-text retrieval performance, suggesting that MedCTA’s gains come from structured temporal supervision that transforms implicit longitudinal semantics into learnable clinical change signals, rather than from a specific LLM backbone.

## 5 Conclusion

We introduced MedCTA, a clinical trajectory alignment framework that extends medical vision– language pre-training from visit-level image–report matching to longitudinal clinical change learning. MedCTA models change at two complementary scopes: abnormality-specific trajectories capture heterogeneous abnormality evolution, while patient-course progression captures overall clinical status.

With structured trend supervision extracted from longitudinal reports by an offline LLM parser, MedCTA enables trajectory-level learning without manual temporal annotations while preserving static image–report alignment. Experiments on temporal image classification, image–text retrieval, and zero-shot classification show consistent gains over strong medical VLP baselines, demonstrating the value of modeling clinical change through both abnormality-specific evolution and patient-course progression.

## References

[1] Pingyi Miao, Xianlai Chen, Kai Sun, Yunbo Wang, Shuang Zhao, and Ying An. Biodpp: Dynamic prompt policy learning for biomedical vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 8052–8060, 2026.

[2] Weiwei Cao, Jianpeng Zhang, Zhongyi Shui, Sinuo Wang, Zeli Chen, Xi Li, Le Lu, Xianghua Ye, Qi Zhang, Tingbo Liang, et al. Boosting vision semantic density with anatomy normality modeling for medical vision-language pre-training. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 23041–23050, 2025.

[3] Huimin Yan, Xian Yang, Liang Bai, and Jiye Liang. Local alignment for medical vision-language pre-training. IEEE Transactions on Image Processing, 34:7321–7334, 2025.

[4] Xin Mei, Rui Mao, Xiaoyan Cai, Libin Yang, and Erik Cambria. Medical report generation via multimodal spatio-temporal fusion. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 4699–4708, 2024.

[5] Christian Bluethgen, Pierre Chambon, Jean-Benoit Delbrouck, Rogier Van Der Sluijs, Małgorzata Połacin, Juan Manuel Zambrano Chaves, Tanishq Mathew Abraham, Shivanshu Purohit, Curtis P Langlotz, and Akshay S Chaudhari. A vision–language foundation model for the generation of realistic chest x-ray images. Nature Biomedical Engineering, 9(4):494–506, 2025.

[6] Hao Yang, Hong-Yu Zhou, Jiarun Liu, Weijian Huang, Cheng Li, Zhihuan Li, Yuanxu Gao, Qiegen Liu, Yong Liang, Qi Yang, et al. A multimodal vision–language model for generalizable annotation-free pathology localization. Nature Biomedical Engineering, 10:1595–1609, 2026.

[7] Xin You, Runze Yang, Chuyan Zhang, Zhongliang Jiang, Jie Yang, and Nassir Navab. Fb-diff: Fourier basis-guided diffusion for temporal interpolation of 4d medical imaging. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 28010–28020, 2025.

[8] Zhuoyi Yang and Liyue Shen. Tempa-vlp: Temporal-aware vision-language pretraining for longitudinal exploration in chest x-ray image. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4625–4634. IEEE, 2025.

[9] Hejie Cui, Alyssa Unell, Bowen Chen, Jason Alan Fries, Emily Alsentzer, Sanmi Koyejo, and Nigam H Shah. Timer: Temporal instruction modeling and evaluation for longitudinal clinical records. npj Digital Medicine, 8(1):577, 2025.

[10] Kang Liu, Zhuoqi Ma, Zikang Fang, Yunan Li, Kun Xie, and Qiguang Miao. Priorrg: Prior-guided contrastive pre-training and coarse-to-fine decoding for chest x-ray report generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 7206–7214, 2026.

[11] Ke Zhang, Hanliang Jiang, Jian Zhang, Qingming Huang, Jianping Fan, Jun Yu, and Weidong Han. Semisupervised medical report generation via graph-guided hybrid feature consistency. IEEE Transactions on Multimedia, 26:904–915, 2024.

[12] Wei Fan, Jingru Fei, Jindong Han, Jie Lian, Hangting Ye, Xiaozhuang Song, Xin Lv, Kun Yi, and Min Li. Medvia: Empowering medical time series classification with vision augmentation and multimodal fusion. Information Fusion, 127:103659, 2026.

[13] Yuxiang Lai, Jike Zhong, Ming Li, Shitian Zhao, Yuheng Li, Konstantinos Psounis, and Xiaofeng Yang. Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models. IEEE Transactions on Medical Imaging, 45(6):2727–2737, 2026.

[14] Vishwesh Nath, Wenqi Li, Dong Yang, Andriy Myronenko, Mingxin Zheng, Yao Lu, Zhijian Liu, Hongxu Yin, Yee Man Law, Yucheng Tang, et al. Vila-m3: Enhancing vision-language models with medical expert knowledge. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14788–14798, 2025.

[15] Zifeng Wang, Zhenbang Wu, Dinesh Agarwal, and Jimeng Sun. Medclip: Contrastive learning from unpaired medical images and text. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 3876–3887, 2022.

[16] Haoran Lai, Qingsong Yao, Zihang Jiang, Rongsheng Wang, Zhiyang He, Xiaodong Tao, and S. Kevin Zhou. Carzero: Cross-attention alignment for radiology zero-shot classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11137–11146, 2024.

[17] Pujin Cheng, Li Lin, Junyan Lyu, Yijin Huang, Wenhan Luo, and Xiaoying Tang. Prior: Prototype representation joint learning from medical images and reports. In 2023 IEEE/CVF International Conference on Computer Vision, pages 21304–21314. IEEE, 2023.

[18] Tianbin Li, Yanzhou Su, Wei Li, Bin Fu, Zhe Chen, Ziyan Huang, Guoan Wang, Chenglong Ma, Ying Chen, Ming Hu, et al. Gmai-vl & gmai-vl-5.5 m: A large vision-language model and a comprehensive multimodal dataset towards general medical ai. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 23177–23185, 2026.

[19] Yanyuan Chen, Dexuan Xu, Yu Huang, Songkun Zhan, Hanpin Wang, Dongxue Chen, Xueping Wang, Meikang Qiu, and Hang Li. Mimo: A medical vision language model with visual referring multimodal input and pixel grounding multimodal output. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 24732–24741, 2025.

[20] Shruthi Bannur, Stephanie Hyland, Qianchu Liu, Fernando Perez-Garcia, Maximilian Ilse, Daniel C Castro, Benedikt Boecking, Harshita Sharma, Kenza Bouzid, Anja Thieme, et al. Learning to exploit temporal structure for biomedical vision-language processing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15016–15027, 2023.

[21] Kang Liu, Zhuoqi Ma, Xiaolu Kang, Yunan Li, Kun Xie, Zhicheng Jiao, and Qiguang Miao. Enhanced contrastive learning with multi-view longitudinal data for chest x-ray report generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10348–10359, 2025.

[22] Chenyu Lian, Hong-Yu Zhou, Dongyun Liang, Jing Qin, and Liansheng Wang. Efficient medical visionlanguage alignment through adapting masked vision models. IEEE Transactions on Medical Imaging, 44(11):4499–4510, 2025.

[23] Jinxia Yang, Bing Su, Xin Zhao, and Ji-Rong Wen. Unlocking the power of spatial and temporal information in medical multimodal pre-training. In Forty-first International Conference on Machine Learning, pages 56382–56396, 2024.

[24] Liang Bai, Zhi Wang, Huimin Yan, and Xian Yang. Medical vision–language pretraining with llm-guided temporal supervision. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 19666–19674, 2026.

[25] Emily Alsentzer, John Murphy, William Boag, Wei-Hung Weng, Di Jindi, Tristan Naumann, and Matthew McDermott. Publicly available clinical bert embeddings. In Proceedings of the 2nd clinical natural language processing workshop, pages 72–78, 2019.

[26] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, 2021.

[27] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

[28] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):317, 2019.

[29] Joy Wu, Nkechinyere Agu, Ismini Lourentzou, Arjun Sharma, Joseph Paguio, Jasper Seth Yao, Edward Christopher Dee, William Mitchell, Satyananda Kashyap, Andrea Giovannini, et al. Chest imagenome dataset. Physio Net, 2021.

[30] Linda Wang, Zhong Qiu Lin, and Alexander Wong. Covid-net: A tailored deep convolutional neural network design for detection of covid-19 cases from chest x-ray images. Scientific reports, 10(1):19549, 2020.

[31] George Shih, Carol C Wu, Safwan S Halabi, Marc D Kohli, Luciano M Prevedello, Tessa S Cook, Arjun Sharma, Judith K Amorosa, Veronica Arteaga, Maya Galperin-Aizenberg, et al. Augmenting the national institutes of health chest radiograph dataset with expert annotations of possible pneumonia. Radiology: Artificial Intelligence, 1(1):e180041, 2019.

[32] Fuying Wang, Yuyin Zhou, Shujun Wang, Varut Vardhanabhuti, and Lequan Yu. Multi-granularity crossmodal alignment for generalized medical visual representation learning. Advances in neural information processing systems, 35:33536–33549, 2022.

[33] Vu Minh Hieu Phan, Yutong Xie, Yuankai Qi, Lingqiao Liu, Liyang Liu, Bowen Zhang, Zhibin Liao, Qi Wu, Minh-Son To, and Johan W Verjans. Decomposing disease descriptions for enhanced pathology detection: A multi-aspect vision-language pre-training framework. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11492–11501. IEEE, 2024.

[34] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

[35] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

## A Additional Experimental Details

## A.1 Pretraining Data

We pretrain MedCTA on MIMIC-CXR [28], which contains chest X-ray images and corresponding radiology reports. For longitudinal modeling, we reorganize the data into patient-level sequences ordered by study time. Following prior temporal medical VLP settings [23], each sequence includes up to four consecutive image–report pairs, while shorter sequences are preserved. The dataset is publicly available at https://physionet.org/content/mimic-cxr-jpg/2.1.0/.

## A.2 Downstream Tasks

We evaluate MedCTA on downstream benchmarks disjoint from the pre-training data, covering both longitudinal temporal reasoning and cross-modal understanding.

Longitudinal temporal reasoning: We evaluate three tasks. Temporal image classification is finetuned on Chest ImaGenome [29] and tested on MS-CXR-T [20], reporting macro-averaged accuracy over progression types. Temporal sentence similarity classification is evaluated as a zero-shot binary classification on MS-CXR-T sentence pairs, reporting accuracy. Dynamic phrase grounding is assessed through qualitative visualization on Chest ImaGenome, where we examine whether the model localizes image regions relevant to temporal progression descriptions across consecutive chest X-rays.

Cross-modal understanding: We evaluate cross-modal retrieval on MIMIC-5×200 [15], reporting P@1, P@2, P@5, and P@10 for both image-to-text and text-to-image retrieval. We also assess zero-shot classification on COVIDx [30] and RSNA Pneumonia [31] using Accuracy and F1.

## A.3 Baselines

We compare MedCTA with recent representative medical vision–language methods, grouped accord ing to whether they explicitly model temporal information. Non-temporal vision–language baselines focus on static image–text alignment: MGCA [32] aligns radiographs and reports at multiple granularities; MedCLIP [15] improves contrastive learning with semantic-aware matching; CARZero [16] leverages LLM-based prompts and cross-attention for zero-shot classification; PRIOR [17] combines cross-modal reconstruction with global-local supervision; and MAVL [33] uses LLM-guided disease decomposition to enhance attribute-level grounding. In contrast, temporal baselines explicitly incorporate longitudinal context: Med-ST [23] introduces temporal supervision through historical image–text pairs and cycle consistency, while ALTA [22] models temporal views with masked pretraining and efficient adaptation. TAMM [24] leverages LLM-derived trend labels and rationales from adjacent reports to inject temporal supervision into medical vision-language pretraining.

Table 5: Additional ablation results on image–text retrieval on the MIMIC-CXR 5×200 benchmark. Performance is reported as precision (%) at top-k (P@k). Relative changes from the full model are shown below.
<table><tr><td>Method</td><td colspan="4">Text → Image</td><td colspan="4">Image → Text</td></tr><tr><td></td><td>P@1</td><td>P@2</td><td>P@5</td><td>P@10</td><td>P@1</td><td>P@2</td><td>P@5</td><td>P@10</td></tr><tr><td>MedCTA</td><td>80.24</td><td>79.08</td><td>75.68</td><td>65.82</td><td>69.13</td><td>68.63</td><td>67.23</td><td>63.01</td></tr><tr><td>w/0  ${ \mathcal { L } } _ { \mathrm { c l i p } }$ </td><td>58.05 (−22.19)</td><td>55.35 (−23.73)</td><td>54.32 (−21.36)</td><td>46.06 (-19.76)</td><td>53.22 (−15.91)</td><td>53.19 (−15.44)</td><td>53.98 (−13.25)</td><td>48.72 (−14.29)</td></tr><tr><td>w/0  $\mathcal { L } _ { \mathrm { a b n } }$ </td><td>77.67 (−2.57)</td><td>76.59 (−2.49)</td><td>73.45 (−2.23)</td><td>63.93</td><td>67.51</td><td>66.36</td><td>64.75 (−2.48)</td><td>61.28 (−1.73)</td></tr><tr><td>w/0  $\mathcal { L } _ { \mathrm { p a t } }$ </td><td>78.90 (−1.34)</td><td>78.02 (−1.06)</td><td>74.01 (-1.67)</td><td>(−1.89) 65.00 (−0.82)</td><td>(−1.62) 68.21 (−0.92)</td><td>(−2.27) 68.67 (+0.04)</td><td>66.51 (−0.72)</td><td>61.80 (−1.21)</td></tr></table>

## A.4 Implementation Details

We initialize MedCTA with pretrained MGCA weights to improve optimization stability and accelerate convergence. The model is trained for 8 epochs on 4 NVIDIA A40 GPUs. We use the AdamW optimizer with a weight decay of $1 \times 1 0 ^ { - 6 }$ and an initial learning rate of $1 \times 1 0 ^ { - 5 }$ . A cosine annealing schedule with 40% warm-up is adopted, gradually decaying the learning rate to $1 \times 1 0 ^ { - 8 }$ All images are resized to $2 5 6 \times 2 5 6$ , followed by a random crop to $2 2 4 \times 2 2 4$ for data augmentation. For downstream evaluation, we either use the pretrained model in a zero-shot manner or fine-tune a lightweight task-specific head, depending on the benchmark. For cross-modal retrieval, temporal sentence similarity classification, and dynamic phrase grounding, we directly evaluate the pretrained model without additional training, and thus do not report standard deviations. In contrast, zero-shot classification and temporal image classification involve training a lightweight classifier on top of pretrained representations. For these tasks, we run experiments with three different random seeds and report the mean and standard deviation to reflect training variability. This evaluation protocol follows prior work for fair comparison.

## B Ablation experimental results

In this section, we provide additional ablation results to further analyze the contribution of each component in MedCTA, the effect of abnormality query initialization, and the robustness to different LLM-based trend supervision extractors.

## B.1 Contribution of Different Components.

To further validate the effectiveness of the three training objectives, we report additional ablation results on image–text retrieval. Specifically, we remove the static image–report alignment loss ${ \mathcal L } _ { \mathrm { c l i p } }$ , the abnormality-specific trend loss $\mathcal { L } _ { \mathrm { a b n } }$ , and the patient-course trend loss $\mathcal { L } _ { \mathrm { p a t } }$ from the full objective, respectively. As shown in Table 5, removing each objective generally degrades retrieval performance across most metrics, indicating that the three objectives provide complementary supervision. Removing $\mathcal { L } _ { \mathrm { c l i p } }$ causes the largest drop, suggesting that visit-level image–report alignment remains essential for cross-modal retrieval. Removing $\mathcal { L } _ { \mathrm { a b n } }$ also reduces performance, reflecting the benefit of modeling fine-grained abnormality evolution. Removing $\mathcal { L } _ { \mathrm { p a t } }$ leads to smaller but mostly consistent declines, suggesting that patient-course progression modeling further improves the learned representations. These retrieval results are broadly consistent with the temporal reasoning results in the main paper and support the complementary roles of the three objectives.

## B.2 Effect of Query Initialization.

We further investigate whether the abnormality queries benefit from clinical semantic initialization. We compare two strategies: random initialization and clinical semantic initialization using abnormality-specific medical prompts. As shown in Table 6, clinical semantic initialization improves most retrieval metrics compared with random initialization. This suggests that abnormality prompts provide useful semantic priors, helping the queries select abnormality-relevant visual and textual evidence and construct more effective abnormality-specific progression trajectories.

Table 6: Effect of abnormality query initialization on image–text retrieval on the MIMIC-CXR 5×200 benchmark. Performance is reported as precision at top-k (P@k).
<table><tr><td>Method</td><td colspan="4">Text → Image</td><td colspan="4">Image → Text</td></tr><tr><td></td><td>P@1</td><td>P@2</td><td>P@5</td><td>P@10</td><td>P@1</td><td>P@2</td><td>P@5</td><td>P@10</td></tr><tr><td>Random init</td><td>78.37</td><td>77.69</td><td>72.94</td><td>65.32</td><td>68.27</td><td>65.42</td><td>68.15</td><td>61.82</td></tr><tr><td>Clinical init</td><td>80.24</td><td>79.08</td><td>75.68</td><td>65.82</td><td>69.13</td><td>68.63</td><td>67.23</td><td>63.01</td></tr></table>

## B.3 Effect of Different LLM Parsers.

MedCTA uses an offline LLM parser to extract structured trend supervision from longitudinal reports. To examine whether the performance depends on a specific LLM backbone, we replace Qwen2-7B [34] with Llama 3-8B [35] while keeping the remaining training and evaluation settings unchanged. As shown in Table 7, MedCTA maintains stable performance under different LLM settings on image–report retrieval. These results indicate that MedCTA does not rely on a particular LLM backbone. Instead, its gains mainly come from the structured temporal supervision that converts implicit longitudinal report semantics into learnable clinical change signals.

Table 7: Robustness to different LLM parsers on image–text retrieval on the MIMIC-CXR 5×200 benchmark. Performance is reported as precision at top-k (P@k).
<table><tr><td>Method</td><td colspan="4">Text → Image</td><td colspan="4">Image → Text</td></tr><tr><td></td><td>P@1</td><td>P@2</td><td>P@5</td><td>P@10</td><td>P@1</td><td>P@2</td><td>P@5</td><td>P@10</td></tr><tr><td>Qwen2-7B</td><td>80.24</td><td>79.08</td><td>75.68</td><td>65.82</td><td>69.13</td><td>68.63</td><td>67.23</td><td>63.01</td></tr><tr><td>Llama 3-8B</td><td>80.11</td><td>78.92</td><td>76.89</td><td>66.09</td><td>69.56</td><td>68.54</td><td>66.85</td><td>63.14</td></tr></table>

## C Limitations and Broader Impact

Limitations. Although MedCTA shows consistent improvements across temporal reasoning and cross-modal understanding tasks, it still has several limitations. First, MedCTA relies on structured trend supervision extracted by an offline LLM parser from longitudinal radiology reports. Although our experiments with different LLM parsers show stable performance, the quality of the extracted trend labels may still be affected by the reasoning ability, prompting strategy, and possible parsing errors of the LLM. Second, MedCTA models abnormality-specific trajectories based on a predefined abnormality set, which may not fully cover rare diseases, fine-grained clinical findings, or complex comorbid conditions. Third, our experiments are mainly conducted on chest X-ray datasets and radiology reports. Further validation on other imaging modalities, medical domains, institutions, and real-world clinical workflows remains an important direction for future work. Finally, MedCTA is designed for representation learning and should not be directly used for clinical diagnosis or decision-making without careful validation and expert supervision.

Broader Impact. MedCTA aims to improve medical vision–language representation learning by explicitly modeling longitudinal clinical change. Its potential positive impacts include enhancing temporally aware medical image understanding, improving progression-related retrieval and grounding, and supporting research on longitudinal disease modeling. By reducing the need for manual temporal annotations through offline LLM-derived supervision, MedCTA may also facilitate the scalable use of longitudinal medical records for representation learning. However, the method may inherit biases and reporting inconsistencies from the underlying medical datasets and radiology reports. In addition, inaccurate LLM-derived trend labels could affect the learned representations, especially in underrepresented diseases or patient groups. Therefore, deployment in high-stakes clinical scenarios should carefully consider data privacy, fairness, and robustness.