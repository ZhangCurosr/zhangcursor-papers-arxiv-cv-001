# Beyond Static Cues: Unified Dynamic Detection of MLLM Backdoors via Dual-Flow Trajectory Analysis

Yinzhen Huang   
22373561\_hyz@buaa.edu.cn   
Beihang University   
Beijing, Beijing, China

Jiahe Xie xiejiahe@buaa.edu.cn Beihang University Beijing, Beijing, China

Ruochen Zhang   
ruochen124@buaa.edu.cn   
Beihang University   
Beijing, Beijing, China

Yitong Sun yt\_sun@buaa.edu.cn Beihang University Beijing, Beijing, China

Xingxing Wei<sup>∗</sup> xx\_wei@buaa.edu.cn Beihang University Beijing, Beijing, China

## Abstract

Multimodal Large Language Models (MLLMs) are increasingly de ployed in safety-critical domains, yet their susceptibility to backdoor attacks remains a formidable challenge. While existing de fenses have evolved from visual-centric patch cues to unsupervised internal attention statistics, they are fundamentally constrained by a reliance on static snapshots and attack-specific heuristics. Given the intricate interplay between dual modalities, the vast design space for multimodal triggers allows diverse threats, such as textual injections, sub-perceptual noises, and coordinated cross-modal triggers, to easily circumvent such rigid priors, particularly when attack types are unknown in real-world applications. In this paper, we rethink backdoor detection by shifting the diagnostic paradigm from static observations to dynamic output evolution. We observe that a successful attack inevitably hijacks the model’s decoding trajectory, skewing internal modal contributions during generation while maintaining an abnormal output invariance despite input perturbations. To capture this behavior, we propose EVO-Detect, a unified detection framework that monitors the dynamic evolution of internal signals during early decoding steps through two complementary channels: Attention Flow and Logits Flow. The attention branch measures how image-text attention allocation evolves dur ing early generation, whereas the logits branch quantifies the sensitivity of the output distribution to modal perturbations through interventional ablations. We then convert these trajectories into compact sample-level metrics and normalize them against negativecontrol signatures so that natural perturbations are less likely to be mistaken for backdoor evidence. Extensive evaluations show that our approach substantially outperforms state-of-the-art baselines such as BYE and TCAP across diverse attack families, establishing dynamic multimodal interplay analysis as a robust foundation for unsupervised backdoor detection of MLLMs. The code is publicly available at https://github.com/kaoru0822-kitauji/EVO\_Detect.

CCS Concepts • Computing methodologies → Computer vision; • Security and privacy → Software and application security.

## Keywords

multimodal large language models, multimodal safety, backdoor detection, sample filtering

ACM Reference Format:   
Yinzhen Huang, Jiahe Xie, Yitong Sun, Ruochen Zhang, and Xingxing Wei. 2026. Beyond Static Cues: Unified Dynamic Detection of MLLM Backdoors via Dual-Flow Trajectory Analysis. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/ 3767308.3836209

## 1 Introduction

The surge in Multimodal Large Language Models (MLLMs) [1, 3, 5, 16, 17, 30] has catalyzed their rapid adoption across high-stakes environments such as autonomous decision-making, document understanding [22], and specialized medical assistance [10]. By integrating vision and language, these systems provide exceptional cognitive depth. However, this structural complexity uncovers a grave security blind spot regarding the vulnerability to multimodal backdoor injections [13, 19, 27]. Unlike overt adversarial attacks [6], a backdoored MLLM maintains deceptive utility on standard inputs and is only hijacked by hidden triggers secretly embedded within downstream fine-tuning datasets. These triggers force the model to generate malicious, attacker-prescribed content, which has ignited significant industry-wide alarm.

Existing MLLM backdoor defenses have evolved through two primary stages. Earlier methods were largely visual-centric, focusing on the detection of static local evidence like image patches [4]. More recent studies, such as BYE [27] and TCAP [19], have begun to exploit internal attention statistics to identify suspicious samples. While these approaches have successfully shifted the defensive paradigm from handcrafted features toward internal profiling, they can still depend strongly on modality-specific or static signatures. However, MLLMs present a vastly expanded threat landscape where adversaries can orchestrate diverse triggers across dual channels. These threats utilize natural textual sufixes, sub-perceptual visual overlays, or complex dual-modal triggers that remain dormant unless activated together as shown in Figure 1. More importantly, in real-world deployment scenarios, attack types are typically unknown, causing earlier methods that over-rely on specific modali ties or static priors to sufer severe performance degradation when confronted with unpredictable risks.

![](images/fc4a124313d1de078102c3191e4adf8a723f917eea62a8c760d4e341e4011490.jpg)  
Figure 1: An illustration of our unified detection framework’s successful interception based on analysis of decoding evolution across various trigger forms: visual attacks, textual injections, and challenging dual-modal triggers, while preserving benign behavior on clean inputs.

To bridge this gap, we aim to propose a modality-agnostic backdoor detection paradigm. However, developing such a system requires overcoming two major challenges: (1) discovering universal discriminative features: unlike localized patch triggers that induce conspicuous, static attention abnormality, diferent attack fami lies manifest highly varied internal behaviors. Identifying a "firstprinciple" signature that consistently exposes the backdoor efect across unknown trigger forms is essential. (2) handling natural data complexity: real-world multimodal data is inherently diverse in content and quality. Benign samples may naturally exhibit outlier characteristics that mimic backdoor signals, leading to over-interception of normal data and severely undermining the model’s utility.

In this paper, we introduce EVO-Detect, a dynamic, unified diagnostic framework that shifts the focus from static snapshots to decoding evolution. To address the first challenge, we draw inspiration from two observations: firstly, poisoned samples induce a persistent, abnormal redirection in the image-text attention allocation across a continuous temporal horizon, rather than a single-token anomaly; secondly, the resulting output logits distribution exhibits an inherent rigidity and a distinctive contribution gap when one modality is ablated, signaling a loss of cross-modal semantic alignment. Based on these insights, we propose a dualstream diagnostic paradigm, including Attention Flow and Logits Flow. Specifically, the former branch monitors the evolution of image-text attention allocation to identify persistent redirection toward trigger-related tokens, while the latter branch quantifies the sensitivity of the next-token distribution through interventional modality ablations to detect the loss of cross-modal coordination. By monitoring these dynamic trajectories over an early decoding window, our framework exposes the underlying hijacking mechanism regardless of the trigger’s specific form. To resolve the second challenge, we implement a Control-Aware Reference Normalization mechanism. Recognizing that false alarms often stem from the accumulation of moderate deviations across multiple metrics rather than one dominant anomaly, we introduce Clean Reference Anchoring to measure departures from normal decoding behavior and Control Baseline Normalization to down-weight non-malicious fluctuations. By normalizing each metric against matched benign references and upper-quantile control baselines, our framework efectively suppresses false positives while preserving high sensitivity to genuine backdoor-induced anomalies. Our contributions are as follows:

• We present a dynamic, modality-agnostic backdoor detection framework that efectively intercepts a wide spectrum of threats, including various textual, visual, and coordinated dual-trigger attacks, without requiring prior knowledge of the specific attack signature.

• We introduce a dual-flow diagnostic paradigm that transitions backdoor detection from static snapshots to dynamic trajectory analysis, jointly monitoring the evolution of image-text Attention Flow and Logits Flow over early decoding steps to unmask the hijacking mechanism.

• We design a Control-Aware Reference Normalization mechanism that evaluates dynamic trajectories against matched benign references to suppress false alarms, ensuring that legitimate complexity is not misidentified as backdoor evidence.

• We achieve strong performance across multiple attack families. Even on the most challenging dual-trigger attacks, our framework achieves the lowest average ASR (5.2%), compared with 81.3% for BYE and 71.8% for TCAP.

## 2 Related Work

## 2.1 Multimodal Large Language Models

Large Language Models (LLMs) have evolved into Multimodal Large Language Models (MLLMs) by integrating vision encoders such as CLIP [26] and modern visual instruction tuning pipelines [5, 16, 17]. To bridge the gap between general pre-training and task-specific deployment, the Fine-Tuning-as-a-Service (FTaaS) paradigm has become widely adopted [9]. Parameter-eficient techniques such as LoRA [8] allow users to easily customize MLLMs. However, this structural separation of user-provided data and server-side training creates a significant security blind spot.

![](images/9ef700bb3a6c99b34ab1fb7a626903c9829d1d6490f9e2686c313aadcf27e6b4.jpg)  
Figure 2: Visualization of poisoned samples across diferent attack families. These examples illustrate how diverse meth ods manipulate visual, textual, or both modalities to achieve malicious hijacking.

## 2.2 MLLM Backdoor Attacks

The FTaaS paradigm exposes MLLMs to backdoor attacks, where adversaries inject a fraction of poisoned samples containing malicious triggers into the user-provided dataset [12]. During fine-tuning, the model learns a hidden association between the trigger and a predefined malicious output, while maintaining normal behavior on clean inputs. In the multimodal context, attackers can exploit different input channels [13]. Text-based triggers manipulate prompts via specific keywords or syntactic transformations such as HiddenKiller [25]. Visual triggers have evolved from localized patches such as BadNets [7] to blend-based overlays [2] and warping-based perturbations such as WaNet [23]. Moreover, the recent trend points toward dual-trigger attacks, which require coordinated activation across both image and text modalities. This dual-key mechanism is highly challenging: single-modality detectors fail because the backdoor remains dormant unless both triggers appear together. Figure 2 provides concrete examples of poisoned samples across these diverse attack families.

## 2.3 MLLM Backdoor Defenses

Existing backdoor defenses generally fall into three categories: input pre-processing, backdoor elimination, and poisoned-sample detection/filtering [12]. Input pre-processing disrupts trigger patterns but often relies on strong clean references and may degrade benign semantics. Backdoor elimination focuses on pruning or repairing model parameters [18], usually with access to trusted clean anchors. In contrast, poisoned-sample filtering audits a downstream fine-tuning dataset and removes high-risk samples before purification or re-fine-tuning [29]. Representative defenses span diferent intervention stages: SentiNet [4] detects localized universal attacks at inference time, ZIP [28] purifies input images for black-box backdoor defense, and recent MLLM methods audit poisoned fine-tuning samples. Among the latter, BYE [27] utilizes layer-wise attention entropy to detect attention collapse toward non-semantic regions, while TCAP [19] profiles the allocation imbalance across system, vision, and user-text components. These attention-driven filtering methods still leave output-side decoding dynamics underexplored. To bridge the gap, our work moves beyond static, attention-only proxies. By dynamically monitoring the early decoding trajectory through Attention Flow and Logits Flow and normalizing these signals against matched control references, we target the point where the backdoor ultimately manifests: the first few output decisions.

## 3 Problem Formulation and Analysis

## 3.1 Threat Model and Task Formulation

We consider a fine-tuning-time poisoning attack on an MLLM. Let $f _ { \theta }$ denote a multimodal model adapted from a pre-trained backbone using a downstream dataset $\mathcal { D } ^ { s u s }$

Attack Model: The attacker injects a poisoned subset $\mathcal { D } ^ { p o i } \subset \mathcal { D } ^ { s u s }$ to associate specific triggers (visual, textual, or dual-modal) with an attacker-chosen target response. The model behaves normally on benign inputs but is hijacked when triggers appear.

Defense Assumptions: The defender has access to the adapted model $f _ { \theta } ,$ the dataset $\mathcal { D } ^ { s u s }$ , and a small auxiliary benign reference pool $\mathcal { D } ^ { r e f }$ used to stabilize auditing. A small matched-control pool $\mathcal { D } ^ { c t r l }$ of non-backdoor negatives (e.g., benign visual perturbations or noise-like samples; Section $4 . 4 )$ is used only to set control baselines and the threshold. The defender does not know the trigger content or poison labels.

Filtering Task: For each sample $x _ { i } \in \mathcal { D } ^ { s u s }$ , the goal is to compute a suspiciousness score $S ( x _ { i } )$ and derive a binary decision ${ \hat { y } } ( x _ { i } ) =$ $1 [ S ( x _ { i } ) > \tau ]$ . The retained subset $\mathcal { D } _ { k e e p } = \{ x _ { i } \ | \ \hat { y } ( x _ { i } ) = 0 \}$ is then used for purification.

## 3.2 From Static Snapshots to Decoding Evolution

Existing MLLM backdoor defenses primarily rely on static snapshots of internal model states, such as spatial attention heatmaps or single-token activation patterns. For instance, visual-centric attacks like BadNets often leave conspicuous traces where attention scores highly concentrate on a localized, connected patch area. However, such static heuristics are fundamentally constrained by their reliance on attack-specific priors; they become inefective when triggers are distributed across modalities or remain sub-perceptual. We rethink backdoor diagnosis by shifting the paradigm from static observations to dynamic output evolution. Our research is motivated by the need to investigate the multimodal hijacking mechanism through two distinct yet complementary channels:

Intrinsic Evolutionary Attention Characteristics (Attention Flow): We focus on how cross-modal attention routing evolves during the generative process. Our motivation stems from the hypothesis that a successful backdoor must establish a persistent, specific pattern of contribution between visual and textual modalities to sustain a malicious output. We aim to uncover whether there exists a consistent law governing this internal attention redistribution across various attack families.

Extrinsic Interventional Decoding Distributions (Logits Flow): We examine the sensitivity of the output distribution through modality intervention. By systematically ablating image or text inputs, we aim to quantify the contribution of each modality to the resulting logits. Our intent is to determine if poisoned samples exhibit detectable rigidity in their decoding trajectory, where the dependency on the hijacked modality difers significantly from benign semantic alignment.

We conduct qualitative experiments to verify these dynamic signatures and our analysis reveals several recurring trajectory phenomena as illustrated in Figure 3:

(1) Temporal Persistence: Poisoned samples rarely deviate from benign ones at only one isolated token; the separation in internal signals usually persists across the first few decoding steps.

(2) Contribution Gap Divergence: The Contribution Gap Trajectory, which measures the imbalance between visual and textual dependency, shows a much larger magnitude shift or a distinct sign pattern for poisoned samples compared to clean references.

(3) Modal Rigidity: Under modality ablation, poisoned samples preserve abnormal target alignment, producing rigid or asymmetric trajectory profiles even when attention-side cues are visually subtle.

Unlike static artifacts that can be easily masked or bypassed, these dynamic signatures reflect the model’s underlying hijacking state across multiple inference steps. Consequently, these insights provide the foundational rationale for our EVO-Detect framework.

![](images/fb24c1121856dc4bbfab1d35da422941a27922856f33ca55c2618f5305906e98.jpg)  
Figure 3: Qualitative views of the proposed sample-filtering signals. The left column illustrates the Attention Map Similarity (�<sub>�</sub>), while the right column shows the Contribution Gap (�<sub>�</sub>). Across visual, textual, and dual-modal attacks, poisoned samples (red dashed lines) consistently exhibit distinct and separate trajectories compared to clean samples (blue solid lines), providing strong evidence for dynamic filtering.

## 4 Method

## 4.1 Overview

We propose EVO-Detect, a dynamic, unified diagnostic framework designed to intercept diverse multimodal backdoors by shifting the focus from static snapshots to decoding evolution. As illustrated in Figure 4, the framework processes candidate samples through a dual-stream pipeline: Attention Flow and Logits Flow. Specifically, we monitor the model’s internal signaling over an early decoding horizon $\left( K = 5 \right)$ to expose the persistent hijacking mechanism that characterizes poisoned samples, regardless of whether the trigger is visual, textual, or coordinated across modalities. The resulting trajectories are then processed by a Control-Aware Reference Normalization mechanism to suppress false alarms arising from natural data complexity, where clean references provide the benign trajectory anchor, while control baselines suppress false alarms caused by natural variation.

Specifically, we first discuss how the temporal evolution of crossmodal routing is captured via Attention Flow (Section 4.2) and how output distribution sensitivity is quantified through modality intervention via Logits Flow (Section 4.3). We then show how these trajectories are instantiated as sample-level metrics and normalized by the clean, suspected, and control streams (Section 4.4), before being fused into a unified suspiciousness score for the final filtering decision (Section 4.5).

## 4.2 Attention Flow

To characterize how cross-modal routing evolves during early decoding, we extract the final decoder layer’s attention map from the current output token to the already observed input tokens at each step $t \in \{ 1 , \ldots , K \}$ . Let $\mathcal { N } _ { t }$ and $\boldsymbol { Q } _ { t }$ denote the image-token and non-image-token sets, respectively. For a head-specific attention vector $A _ { t , h } ,$ , we compute the image and text attention ratios as:

$$
r _ { t , h } ^ { i m g } ( x ) = \frac { \sum _ { j \in \mathcal { V } _ { t } } A _ { t , h , j } ( x ) } { \sum _ { j \in \mathcal { V } _ { t } \cup Q _ { t } } A _ { t , h , j } ( x ) } , \quad r _ { t , h } ^ { t x t } ( x ) = 1 - r _ { t , h } ^ { i m g } ( x ) .\tag{1}
$$

We further measure the concentration and temporal stability of each head by:

$$
\begin{array} { l } { { E _ { t , h } ( x ) = - \displaystyle \sum _ { j } \tilde { A } _ { t , h , j } ( x ) \log \tilde { A } _ { t , h , j } ( x ) , } } \\ { { { } } } \\ { { M _ { t , h } ( x ) = 1 - \displaystyle d _ { \mathrm { c o s } } ( A _ { t , h } ( x ) , A _ { t - 1 , h } ( x ) ) , } } \end{array}\tag{2}
$$

where $\tilde { A } _ { t , h , j } ( x )$ is the normalized attention probability, $E _ { t , h } ( x )$ denotes attention entropy, and $d _ { \mathrm { c o s } }$ denotes cosine distance; $M _ { t , h } ( x )$ is used for $t \geq 2 . \mathrm { W e }$ then average over heads to obtain sample-level trajectories:

$$
a _ { t } ( x ) = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } r _ { t , h } ^ { i m g } ( x ) ,
$$

$$
e _ { t } ( x ) = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } E _ { t , h } ( x ) ,\tag{3}
$$

$$
s _ { t } ( x ) = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } M _ { t , h } ( x ) ,
$$

where � is the number of heads, and $a _ { t } ( x ) , e _ { t } ( x )$ , and $s _ { t } ( x )$ denote the image-attention, entropy, and map-similarity trajectories of sample �, respectively. We do not keep a separate text-attention trajectory because it is fully determined by $1 - a _ { t } ( x )$

In our default instantiation, � is kept small and fixed across settings (�=5 unless otherwise stated). This short horizon captures trigger-induced redirection before later generation becomes dominated by task-specific continuation, while still providing richer evidence than a single first-token snapshot.

## 4.3 Logits Flow

To quantify whether the output distribution has already been hijacked, we introduce a second branch based on controlled modality ablations. For each sample and decoding step, we obtain the nexttoken distribution from three views: the full input, a text-only input, and an image-only input. Denote the corresponding distributions by $p _ { t } ^ { f u l l } , p _ { t } ^ { t x t }$ , and $\boldsymbol { p } _ { t } ^ { i m g }$ . We measure modality contribution shifts through Jensen-Shannon (JS) distance [14] (“JS Divergence” in Figure 4):

$$
\begin{array} { r } { \ell _ { t } ^ { t x t } ( x ) = J S ( p _ { t } ^ { f u l l } ( x ) , p _ { t } ^ { t x t } ( x ) ) , } \\ { \ell _ { t } ^ { i m g } ( x ) = J S ( p _ { t } ^ { f u l l } ( x ) , p _ { t } ^ { i m g } ( x ) ) . } \end{array}\tag{4}
$$

These divergences quantify how strongly the current token prediction depends on each modality at decoding step �. To characterize

![](images/9a4b54018e00d2b7a3bc575ac5597eaf2bf3a21b84ac6c51260a772c1d85db9c.jpg)  
Figure 4: Overview of the proposed EVO-Detect framework for securing downstream MLLM fine-tuning. The pipeline consists of four stages. First, candidate samples undergo Trajectory Analysis over early decoding steps $\left( K = 5 \right)$ via dual streams: Attention Flow and Logits Flow. The extracted sample features (red) are routed to the Reference Normalization module to form abnormality metrics against a clean baseline (green), which are then normalized with matched controls (blue) to suppress false alarms. Finally, a unified suspiciousness score $S ( x )$ is thresholded to retain clean data for downstream fine-tuning.

the imbalance between the two modalities, we further define:

$$
\begin{array} { r l } & { g _ { t } ( x ) = \ell _ { t } ^ { i m g } ( x ) - \ell _ { t } ^ { t x t } ( x ) , } \\ & { i _ { t } ( x ) = \ell _ { t } ^ { t x t } ( x ) + \ell _ { t } ^ { i m g } ( x ) - J S ( p _ { t } ^ { t x t } ( x ) , p _ { t } ^ { i m g } ( x ) ) . } \end{array}\tag{5}
$$

Here $g _ { t } ( x )$ is the contribution-gap trajectory, and $i _ { t } ( x )$ is an interactionsensitive trajectory used in dual-trigger audit. Together, $\ell ^ { t x t } ( x )$ $\ell ^ { i m g } ( x )$ , �(�), and optionally � (�) form the Logits Flow of �.

## 4.4 Control-Aware Reference Normalization

To isolate genuine backdoor evidence from natural outliers, we implement a Control-Aware Reference Normalization mechanism. We first use the clean stream to turn the raw trajectories of a suspected sample into abnormality metrics, and subsequently use the control stream to normalize these abnormality metrics for reliable detection. For any instantiated trajectory $\phi ( x )$ drawn from the following set, with �(�) included only when the interaction-sensitive trajectory is available:

$$
\phi ( x ) \in \{ a ( x ) , e ( x ) , s ( x ) , \ell ^ { t x t } ( x ) , \ell ^ { i m g } ( x ) , g ( x ) , i ( x ) \} ,
$$

we first build a clean reference trajectory by step-wise median aggregation:

$$
\begin{array} { r } { \bar { \phi } _ { t } = \mathrm { m e d } _ { x ^ { \prime } \in \mathcal { D } ^ { r e f } } \phi _ { t } ( x ^ { \prime } ) . } \end{array}\tag{6}
$$

Given a suspected sample $x ,$ we then derive abnormality metrics from two generic templates. The first is the trajectory distance:

$$
d _ { \phi } ( \boldsymbol { x } ) = \frac { 1 } { n _ { \phi } } \sum _ { t = 1 } ^ { n _ { \phi } } \left| \phi _ { t } ( \boldsymbol { x } ) - \bar { \phi } _ { t } \right| ,\tag{7}
$$

where $n _ { \phi }$ is the aligned trajectory length. The second is the signed mean deviation:

$$
\Delta _ { \phi } ( x ) = \frac { 1 } { n _ { \phi } } \sum _ { t = 1 } ^ { n _ { \phi } } \left( \bar { \phi } _ { t } - \phi _ { t } ( x ) \right) .\tag{8}
$$

Finally, for the contribution-gap trajectory $g ( x )$ , we also use signstructure descriptors:

$$
\begin{array} { l } { { \eta _ { g } ^ { - } ( x ) = \displaystyle \frac { 1 } { n _ { g } } \sum _ { t = 1 } ^ { n _ { g } } \mathbb { I } [ g _ { t } ( x ) < - \delta _ { g } ] \mathrm { , } } } \\ { { \eta _ { g } ^ { + } ( x ) = \displaystyle \frac { 1 } { n _ { g } } \sum _ { t = 1 } ^ { n _ { g } } \mathbb { I } [ g _ { t } ( x ) > \delta _ { g } ] \mathrm { , } } } \end{array}\tag{9}
$$

where $\delta _ { g } > 0$ is a fixed polarity margin and $n _ { g }$ is the length of $g ( x )$ The core sample metric set is instantiated as:

$$
\{ m _ { k } ( x ) \} = \left\{ \begin{array} { l } { d _ { a } ( x ) , d _ { e } ( x ) , d _ { s } ( x ) , d _ { \ell ^ { t x t } } ( x ) , d _ { \ell ^ { i m g } } ( x ) , } \\ { d _ { g } ( x ) , | \Delta _ { g } ( x ) | , \eta _ { g } ^ { - } ( x ) , \eta _ { g } ^ { + } ( x ) } \end{array} \right\} .\tag{10}
$$

When �(�) is available, its corresponding distance metric $d _ { i } ( x )$ is added in the same manner. Here $d _ { \phi } ( x )$ measures trajectory-level departure from the clean reference, $\Delta _ { \phi } ( x )$ measures signed deviation, and $\eta _ { q } ^ { - } ( x ) , \eta _ { q } ^ { + } ( x ) , | \Delta _ { g } ( x ) |$ describe abnormal gap polarity and shift, where $\Delta _ { g } ( \dot { x } )$ is the instance of $\Delta _ { \phi } ( x )$ obtained by setting $\phi = g .$ . These quantities are the abnormality metrics produced by the clean-reference and suspected-sample streams.

To robustly suppress false alarms, we further compare these abnormality metrics against matched controls. Not every anomaly is a backdoor signal: some multimodal samples are naturally dificult, visually noisy, or distribution-shifted. For each metric $m _ { k } ( x )$ , we define the matched-control baseline as:

$$
c _ { k } = Q _ { q _ { c } } \big ( \{ m _ { k } ( x ^ { \prime } ) : x ^ { \prime } \in \mathcal { D } ^ { c t r l } \} \big )\tag{11}
$$

where $Q _ { q _ { c } }$ denotes the �<sub>�</sub>-quantile and $q _ { c }$ is fixed in advance. We further define $m _ { k } ^ { + } ( x ) = \operatorname* { m a x } ( 0 , m _ { k } ( x ) )$ and $c _ { k } ^ { + } = \operatorname* { m a x } ( 0 , c _ { k } )$ , and compute the normalized term as:

$$
T _ { k } ( x ) = \operatorname* { m i n } \left( \kappa , \ \frac { \operatorname* { m a x } ( 0 , m _ { k } ^ { + } ( x ) - \gamma c _ { k } ^ { + } ) } { \operatorname* { m a x } ( | c _ { k } ^ { + } | , \varepsilon ) } \right) ,\tag{12}
$$

where $\gamma \geq 1$ is a conservative margin, � > 0 prevents unstable normalization when the control baseline is near zero, and � caps the influence ofany single metric. Only abnormality metrics that exceed the control baseline by a suficient margin contribute strongly. Visual filtering uses benign visual perturbations or noise-like samples as controls. Text filtering uses visual negatives to suppress purely visual variance. The same normalization procedure is applied to every instantiated metric, including �<sub>�</sub> (�) when the interaction-sensitive trajectory � (�) is available.

## 4.5 Unified Scoring and Decision Rule

To robustly consolidate the normalized evidence from the clean, suspected, and control streams, we obtain the final suspiciousness score by weighted fusion:

$$
S ( x ) = \frac { \sum _ { k } \alpha _ { k } T _ { k } ( x ) } { \sum _ { k } \alpha _ { k } } ,\tag{13}
$$

where $\alpha _ { k } \geq 0$ is the fixed weight of metric $m _ { k } ( x )$ . We use branch balanced weights: the attention and logits branches receive equal total weight, with equal weights within each branch. In this way, the clean stream provides the benign reference $\bar { \phi } ,$ the suspected stream provides the abnormality metrics $m _ { k } ( x )$ , and the control stream provides the control baseline $c _ { k } ,$ , all of which are jointly used to produce �(�). Once the normalized terms are fused into the final suspiciousness score, the remaining step is to map this continuous score to a binary filtering decision. To convert the suspiciousness score into a filtering label, we define a threshold � and flag a sample whenever $S ( x ) > \tau .$ . We support both fixed thresholds and distribution-derived thresholds. In our default instantiation, � is obtained by applying a high quantile operator $Q _ { q _ { \tau } }$ to the score distribution of matched negative controls, where $q _ { \tau }$ is fixed in advance. The final decision is therefore:

$$
\hat { y } ( x ) = 1 [ S ( x ) > \tau ] .\tag{14}
$$

Samples with $\hat { y } ( x ) = 1$ are removed from the downstream finetuning dataset under audit, while samples with $\hat { y } ( x ) = 0$ are retained to form the cleaned dataset for subsequent purification stage.

## 5 Experiments

## 5.1 Setups

Datasets and attacks: We conduct experiments across four benchmarks with diferent image-text coupling patterns: ScienceQA [20], COCO [15], Flickr30k [24], and IconQA [21]. For image-caption experiments, we adopt the released ReCap-COCO-30K dataset [11], a recaptioned COCO2014 validation set [15]. ScienceQA and IconQA emphasize multimodal question answering, while COCO and Flickr30k emphasize image-text description and grounding. We evaluate six representative backdoor attack families grouped into three trigger categories. Visual attacks include BadNet [7], Blended [2], and WaNet [23]. Text attacks include explicit prefix injection (Text Prefix, whose trigger phrase is appended after the prompt) and HiddenKiller [25]. Dual attacks use jointly optimized visual and textual triggers that must be activated together, while either trigger alone is constrained to preserve normal outputs and is used as a single-modality negative case in evaluation.

Models and baselines: We adopt two representative MLLM backbones, Qwen3-VL [1] and InternVL2.5 [3], both adapted with LoRA [8]. We benchmark our method against Vanilla FT, SentiNet [4], ZIP [28],

![](images/6ff162b2e0db50f37595d7873693411b5b722e5542a6c9eeb12d9d1b775a0c74.jpg)

![](images/f17b0e7c1b9e5432625c9278f49e31bb6ca5c0708509bc1b60f25f28bc1244d7.jpg)

![](images/83b498c84872c9d6ebaab6c2089c88b66f36f0cd07fdfb91e2ee91bad35cd5e5.jpg)  
Figure 5: Distribution of the unified suspiciousness score $S ( x )$ across diferent attack families. Evaluated using Qwen3-VL on the COCO dataset. The plots demonstrate a clear separation between clean and poisoned samples under (A) visual, (B) textual, and (C) dual-modal attacks. The horizontal dashed lines represent the filtering thresholds �.

BYE [27], and TCAP [19]. We align the attack setting, the downstream fine-tuning dataset to be filtered, and the purification protocol across methods so that the reported diferences reflect filtering quality rather than evaluation mismatch.

Metrics: We employ two metric groups to assess performance from both the purification and detection perspectives. First, we report clean performance (CP) and attack success rate after filtering (ASR), where higher CP and lower ASR are better. Second, when synthetic poison labels are available for evaluation, we report poisoned-sample detection F1-score. This separation is important because CP/ASR measure the practical benefit of filtering after refine-tuning, whereas sample-level F1 measures how accurately a method identifies suspicious samples.

Purification protocol: We adopt the same purification-style protocol as BYE and TCAP. Each defense first scores samples in the downstream fine-tuning dataset under audit and removes those above its threshold or filtering budget. A purified model is then obtained by re-fine-tuning on the retained subset under the same training recipe. We measure CP on clean evaluation data and ASR on triggered evaluation data of the purified model. For F1, groundtruth poison labels are used only for evaluation in the simulated benchmark and are never used to fit the detector.

Implementation details: We employ a short early-decoding horizon of �=5 unless otherwise stated. We extract attention-side features including image ratio, entropy, and step-to-step map drift, together with logits-side features based on divergences between full, text-only, and image-only outputs. We construct clean references with robust step-wise statistics, summarize matched controls with upper quantiles, and threshold the final score with a high-quantile policy. All normalization and thresholding choices are fixed once per audit slot rather than tuned per attack instance.

## 5.2 Overall Defense Performance

Defense Efectiveness across Attack Families. Table 1 demonstrates performance across six attack modes on ScienceQA, where it could be seen that our method achieves the lowest ASR on all visual and dual-trigger settings, and the best ASR–CP trade-of on textual attacks, where SentiNet reaches lower ASR only at a substantial CP cost (55.6–70.0%). For instance, on InternVL2.5, we reduce dual-trigger ASR to 9.2% while retaining CP above 87%. This superiority stems from addressing the limitations of baselines: SentiNet sacrifices CP due to coarse masking, while ZIP’s generic purification is insuficient for poisoned fine-tuning. Although BYE and TCAP are more competitive, they remain unreliable when triggers lack clear attention-collapse signatures or when backdoors are primarily reflected in output-side dynamics. By jointly auditing attention trajectories and logits-side modality shifts, our method ensures robust filtering across all attack families. Table 2 provides a cross-dataset comparison aggregated by specific attack modalities.

Table 1: ScienceQA overview across attack modes and defense methods. Results are reported in percentage (%). Bold font indicates the best performance among all defense methods.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Attack</td><td colspan="2">Vanilla FT</td><td colspan="2">SentiNet</td><td colspan="2">ZIP</td><td colspan="2">BYE</td><td colspan="2">TCAP</td><td colspan="2">Ours</td></tr><tr><td>CP (↑)</td><td>ASR (↓)</td><td>CP (↑)</td><td>ASR (↓)</td><td>CP (↑)</td><td>ASR (↓)</td><td>CP (↑)</td><td>ASR (↓)</td><td>CP (↑)</td><td>ASR (↓)</td><td>CP (↑)</td><td>ASR (↓)</td></tr><tr><td rowspan="6">Qwen3-VL</td><td>BadNet</td><td>89.5</td><td>100.0</td><td>88.9</td><td>44.4</td><td>88.9</td><td>100.0</td><td>88.9</td><td>41.8</td><td>83.6</td><td>13.8</td><td>90.0</td><td>2.0</td></tr><tr><td>Blended</td><td>89.5</td><td>97.0</td><td>40.0</td><td>23.3</td><td>90.0</td><td>97.0</td><td>82.6</td><td>45.6</td><td>60.0</td><td>56.7</td><td>87.0</td><td>2.0</td></tr><tr><td>WaNet</td><td>89.8</td><td>96.4</td><td>40.0</td><td>52.2</td><td>80.0</td><td>96.4</td><td>51.1</td><td>30.0</td><td>83.0</td><td>81.1</td><td>80.0</td><td>0.0</td></tr><tr><td>Text Prefix</td><td>88.2</td><td>100.0</td><td>55.6</td><td>11.0</td><td>88.9</td><td>100.0</td><td>78.0</td><td>68.2</td><td>78.9</td><td>21.4</td><td>89.0</td><td>16.0</td></tr><tr><td>HiddenKiller</td><td>89.1</td><td>100.0</td><td>70.0</td><td>0.0</td><td>90.0</td><td>100.0</td><td>80.1</td><td>74.2</td><td>80.0</td><td>97.8</td><td>85.0</td><td>17.0</td></tr><tr><td>Dual</td><td>88.9</td><td>100.0</td><td>44.4</td><td>51.6</td><td>88.9</td><td>100.0</td><td>79.2</td><td>63.1</td><td>78.1</td><td>84.6</td><td>86.7</td><td>14.2</td></tr><tr><td rowspan="6">InternVL2.5</td><td>BadNet</td><td>87.2</td><td>100.0</td><td>91.0</td><td>32.0</td><td>88.9</td><td>100.0</td><td>55.6</td><td>83.5</td><td>85.0</td><td>26.4</td><td>90.1</td><td>2.0</td></tr><tr><td>Blended</td><td>87.7</td><td>89.6</td><td>63.3</td><td>29.1</td><td>86.0</td><td>60.0</td><td>70.0</td><td>82.2</td><td>67.0</td><td>41.1</td><td>89.0</td><td>1.0</td></tr><tr><td>WaNet</td><td>88.5</td><td>91.0</td><td>62.0</td><td>41.0</td><td>80.0</td><td>82.2</td><td>90.0</td><td>80.0</td><td>73.1</td><td>58.3</td><td>82.9</td><td>2.0</td></tr><tr><td>Text Prefix</td><td>87.4</td><td>100.0</td><td>60.0</td><td>8.0</td><td>88.9</td><td>100.0</td><td>56.2</td><td>88.1</td><td>61.2</td><td>19.3</td><td>89.6</td><td>11.0</td></tr><tr><td>HiddenKiller</td><td>86.9</td><td>100.0</td><td>69.0</td><td>12.0</td><td>90.0</td><td>100.0</td><td>62.1</td><td>86.2</td><td>68.9</td><td>71.4</td><td>87.2</td><td>14.0</td></tr><tr><td>Dual</td><td>88.3</td><td>100.0</td><td>58.0</td><td>39.0</td><td>88.9</td><td>100.0</td><td>65.1</td><td>83.2</td><td>65.1</td><td>68.8</td><td>88.1</td><td>9.2</td></tr></table>

Table 2: Average CP and ASR across all datasets and backbones. Results are in percentage (%).
<table><tr><td rowspan="2">Method</td><td colspan="3">ASR by Attack Type (%) ↓</td><td rowspan="2">Overall ASR↓</td><td rowspan="2">Overall CP↑</td></tr><tr><td>Visual</td><td>Text</td><td>Dual</td></tr><tr><td>Vanilla FT</td><td>97.7</td><td>99.7</td><td>99.8</td><td>98.7</td><td>74.9</td></tr><tr><td>SentiNet</td><td>63.1</td><td>14.3</td><td>53.1</td><td>45.2</td><td>67.4</td></tr><tr><td>ZIP</td><td>72.7</td><td>99.7</td><td>99.5</td><td>86.2</td><td>74.0</td></tr><tr><td>BYE</td><td>69.7</td><td>83.4</td><td>81.3</td><td>76.2</td><td>70.0</td></tr><tr><td>TCAP</td><td>36.0</td><td>43.4</td><td>71.8</td><td>44.4</td><td>70.7</td></tr><tr><td>Ours</td><td>2.2</td><td>4.1</td><td>5.2</td><td>3.3</td><td>74.4</td></tr></table>

## 5.3 Robustness and Practicality

Delayed adaptive attacks. Because �=5 is a default horizon rather than a hard architectural constraint, we further evaluate delayedtarget attacks that postpone malicious behavior beyond the initial decoding steps. On Qwen3-VL/ScienceQA text attacks, extending the window keeps F1 at 85.7–90.9% and reduces ASR to 0–10%. The harder delayed dual-trigger setting also benefits from a longer horizon: increasing � from 5 to 12 reduces ASR from 36.7% to 13.3% for a five-token delay and from 43.3% to 3.3% for a seventoken delay, while F1 rises from 61.5% to 74.2% and from 44.0% to 71.9%, respectively. These results motivate using a larger or sliding window in high-risk audits.

Reference-pool robustness. We use $| \mathcal { D } ^ { r e f } | { = } 6 4$ in the main experiments and stress-test both pool size and contamination. Across sizes 16, 32, and 64, F1 remains 89.7–90.8% for text triggers, 89.3– 92.3% for BadNet, and 83.6–90.6% for dual triggers. Under up to 5% contamination, text and BadNet performance remains stable, while the more dificult Blended and dual settings degrade gradually; substantial deterioration appears mainly at 10% contamination. The fixed default quantiles $( q _ { c } , q _ { \tau } ) { = } ( 0 . 9 , 0 . 9 )$ are used across attacks rather than tuned per attack.

Eficiency. On Qwen3-VL/ScienceQA, full EVO-Detect requires 3.96–4.16 seconds per sample and 4.86–4.89 GiB peak memory, compared with 1.88–2.28 seconds and 4.01–6.65 GiB for BYE/TCAP. Thus, the method incurs roughly 2× latency during ofline auditing, with comparable memory usage and no online request-time overhead; in the same benchmark, its F1 range is 87.0–95.2%, versus 7.7–81.0% for BYE/TCAP. We stream only compact per-step scalars, such as attention ratios, entropy, and JS distances, instead of storing full attention maps or logit distributions for the dataset.

## 5.4 Sample-Level Detection Performance

To evaluate the efectiveness of our framework in identifying poisoned samples, we conduct a sample-level analysis on the ScienceQA dataset, reporting F1-score. Sample-level results show that our method achieves the best F1 on nearly all attack settings across both backbones. Firstly, the margin is particularly evident on textual attacks. For example, on Qwen3-VL, F1 reaches 87.3% on Text Prefix and 77.2% on HiddenKiller, strikingly higher than BYE (42.0%/35.9%) and TCAP (81.0%/4.2%). A consistent performance gain is observed on InternVL2.5, where our F1-scores (90.6% and 76.5%) remain substantially above both baselines. Second, our method remains competitive on visual attacks as well, achieving the best F1 across all Qwen3-VL settings and all three InternVL2.5 visual settings. This result shows that adding logits-side trajectory evidence does not weaken performance on classic visual triggers. Third, the weaknesses of BYE and TCAP are consistent with their design assumptions. BYE is most efective when poisoning induces a clear attentioncollapse signature and is therefore much less reliable on text-triggered or distributed-trigger settings. TCAP is generally stronger than BYE because it profiles attention allocation more carefully, but it can still become unstable when the decisive abnormality primarily appears in output-side trajectory rigidity rather than in attention allocation itself. In contrast, our method remains more uniform across attack families because it jointly uses attention-side and logits-side evidence.

Table 3: Hyperparameter study on the decoding horizon � under a representative ScienceQA setting with Qwen3-VL. Each entry reports F1 / ASR (%) for poisoned-sample detection and purification performance. Higher F1 and lower ASR are better. Bold indicates the best result in each row.
<table><tr><td rowspan="2">Attack</td><td> ${ \bf K } = { \bf 1 }$ </td><td> $\mathbf { K } = 3$ </td><td> ${ \bf K } = 5$ </td><td> $\mathbf { K } = 7$ </td></tr><tr><td>F1 / ASR</td><td>F1 / ASR</td><td>F1 / ASR</td><td>F1 / ASR</td></tr><tr><td>Visual</td><td>24.6 / 85.8</td><td>42.7 / 72.2</td><td> $\mathbf { 9 6 . 6 } / 2 . 0$ </td><td> $9 2 . 1 / 1 . 0$ </td></tr><tr><td>Text</td><td>18.9 / 89.4</td><td>65.4 / 48.3</td><td> $\mathbf { 8 7 . 3 / 1 6 . 0 }$ </td><td> $8 6 . 6 \textrm { / } 1 4 . 0$ </td></tr><tr><td>Dual</td><td>22.7  / 87.0</td><td>64.2 / 50.4</td><td> $8 2 . 9 \textrm { / } 1 4 . 2$ </td><td> $\mathbf { 8 3 . 6 \ / \ 1 9 . 0 }$ </td></tr></table>

To further illustrate the discriminative power of our framework, Figure 5 visualizes the distribution of the unified suspiciousness score �(�) on the COCO dataset using Qwen3-VL. As shown in Figures 5(A) and (B), our method establishes a clear, wide margin between clean and poisoned samples under single-modality attacks. Critically, in the challenging dual-trigger scenario as Figure 5(C), the framework selectively assigns high scores to true poisoned samples. Meanwhile, single-modality negative cases (Vis Only and Txt Only) yield score distributions nearly identical to clean data, confirming that our Control-Aware Reference Normalization effectively suppresses false alarms triggered by partial or benign multimodal variations.

## 5.5 Ablation Study

Efect of the Decoding Horizon: We evaluate the sensitivity of the early-decoding horizon � on ScienceQA using Qwen3-VL, with results summarized in Table 3. The analysis reveals that detection performance improves substantially as the horizon extends beyond a single step. At �=1, the detector is too myopic to capture persistent trajectory deviations, resulting in weak F1 scores and high ASR across all attack types (e.g., a mere 18.9% F1 and 89.4% ASR against text triggers). Increasing the horizon to �=3 yields significant gains, confirming that short-range temporal accumulation is essential for robust diagnosis. The optimal trade-of is achieved at �=5, where the framework efectively suppresses ASR while maximizing F1-scores, peaking at an impressive 96.6% F1 for visual attacks. Extending the horizon to �=7 provides only marginal improvements for dual-trigger cases (reaching 83.6% F1) and slightly lowers F1 elsewhere, though visual/text ASR dip to 1.0%/14.0%. This mild degradation suggests that pushing the horizon too far begins to accumulate benign generation noise, which subsequently dilutes the malicious trigger’s initial activation signal. Ultimately, this pattern indicates that the most discriminative evidence is concentrated in the first few decoding steps, making a short horizon both suficient and computationally eficient.

Table 4: Ablation study on Qwen3-VL (independent run from Table 3). We report poisoned-sample detection F1 (%). AF=Attention Flow, LF=Logits Flow, Norm.=Reference Normalization, �=decoding horizon. Bold indicates the best result.
<table><tr><td rowspan="2">Variant</td><td colspan="4">Configuration</td><td rowspan="2"></td><td colspan="2">F1 Score (%)</td><td rowspan="2"> $\mathbf { A v g . }$ </td></tr><tr><td>AF</td><td>LF</td><td>Norm.</td><td>K</td><td>BadNet</td><td>Text Dual</td></tr><tr><td>Static AF only</td><td>√</td><td>一</td><td></td><td>1</td><td>0.0</td><td>0.0</td><td>11.4</td><td>3.8</td></tr><tr><td>Static LF only</td><td>一</td><td>√</td><td></td><td>1</td><td>29.8</td><td>18.9</td><td>16.1</td><td>21.6</td></tr><tr><td>Dynamic AF only</td><td>√</td><td>一</td><td></td><td>5</td><td>49.8</td><td>42.0</td><td>35.1</td><td>42.3</td></tr><tr><td>Dynamic LF only</td><td>一</td><td>√</td><td></td><td>5</td><td>55.0</td><td>37.3</td><td>32.0</td><td>41.4</td></tr><tr><td>AF + LF (no norm.)</td><td>√</td><td>√</td><td>一</td><td>5</td><td>80.6</td><td>69.3</td><td>54.6</td><td>68.2</td></tr><tr><td>Full (Ours)</td><td>√</td><td>√</td><td>√</td><td>5</td><td>90.9</td><td>92.4</td><td>83.3</td><td>88.9</td></tr></table>

Efect of each module: To systematically evaluate the individual contributions of our proposed components in defending against diverse multimodal backdoor attacks, we conducted a comprehensive ablation study on the representative Qwen3-VL backbone. Results are shown in Table 4. The analysis across three representative attacks reveals three key insights: (1) relying on a static, single-step inspection severely cripples detection, as evidenced by the static Attention Flow (AF) variant failing almost completely, which proves that continuous early decoding trajectories capture far more robust backdoor evidence than isolated frozen snapshots; (2) neither AF nor Logits Flow (LF) alone is suficient for comprehensive defense, whereas fusing these branches yields a substantial performance leap, confirming that attention patterns and logit distributions provide highly complementary traces of malicious behavior; and (3) converting this fused evidence into a reliable detector fundamentally relies on the Control-Aware Reference Normalization module. Without this normalization, the average F1 score plummets from 88.9% to 68.2%, and the model struggles most against the stealthy dual-trigger attack (dropping from 83.3% to a mere 54.6%). Overall, these results validate that our framework’s robust performance stems directly from the synergy of dynamic trajectory profiling, multi-level feature fusion, and reference-normalized scoring.

## 6 Conclusion

In this paper, we rethink MLLM backdoor defense by shifting from static snapshots to decoding evolution. We introduce EVO-Detect, which exposes persistent hijacking mechanisms by jointly monitoring internal Attention Flow and interventional Logits Flow. To handle natural data complexity, a Control-Aware Reference Normalization mechanism anchors diagnostic signals against benign references, efectively suppressing false alarms. Extensive evaluations demonstrate that our modality-agnostic approach substantially outperforms previous attention-based baselines across diverse multimodal threats. This transition to temporal trajectory analysis provides a robust foundation for securing multimodal foundation models.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grant 62576020.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631 (2025). https://arxiv.org/abs/ 2511.21631

[2] Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. 2017. Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning. arXiv preprint arXiv:1712.05526 (2017). https://arxiv.org/abs/1712.05526

[3] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. 2024. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling. arXiv preprint arXiv:2412.05271 (2024). https://arxiv.org abs/2412.05271

[4] Edward Chou, Florian Tramèr, and Giancarlo Pellegrino. 2020. SentiNet: Detecting Localized Universal Attacks Against Deep Learning Systems. In SPW. IEEE, Los Alamitos, CA, USA, 48–54. doi:10.1109/SPW50608.2020.00025

[5] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. In NeurIPS. Curran Associates, Inc., Red Hook, NY, USA, 49250–49267. doi:10.52202 075280-2142

[6] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. 2015. Explaining and Harnessing Adversarial Examples. In ICLR. San Diego, CA, USA. https: //arxiv.org/abs/1412.6572

[7] Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. 2019. BadNets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access 7 (2019), 47230–47244. doi:10.1109/ACCESS.2019.2909068

[8] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR. OpenReview.net, Online. https://openreview.net/ forum?id=nZeVKeeFYf

[9] Wenke Huang, Jian Liang, Zekun Shi, Didi Zhu, Guancheng Wan, He Li, Bo Du, Dacheng Tao, and Mang Ye. 2025. Learn from Downstream and Be Yourself in Multimodal Large Language Models Fine-Tuning. In ICML. PMLR, Vancouver, Canada, 25486–25499. https://proceedings.mlr.press/v267/huang25q.htm

[10] Chunyuan Li, Clif Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. 2023. LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day. In NeurIPS Datasets and Benchmarks Track. Curran Associates, Inc., Red Hook, NY, USA, 28541–28564. doi:10.52202/075280-1240

[11] Xianhang Li, Haoqin Tu, Mude Hui, Zeyu Wang, Bingchen Zhao, Junfei Xiao, Sucheng Ren, Jieru Mei, Qing Liu, Huangjie Zheng, Yuyin Zhou, and Cihang Xie. 2025. What If We Recaption Billions of Web Images with LLaMA-3?. In ICML. PMLR, Vancouver, Canada, 35957–35976. https://proceedings.mlr.press/v267/ li25ch.html

[12] Yiming Li, Yong Jiang, Zhifeng Li, and Shu-Tao Xia. 2024. Backdoor Learning: A Survey. IEEE Transactions on Neural Networks and Learning Systems 35, 1 (2024), 5–22. doi:10.1109/TNNLS.2022.3182979

[13] Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Mingli Zhu, Xiaochun Cao, and Dacheng Tao. 2025. Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift. In CVPR. IEEE/CVF, 9477–9486. https://openaccess.thecvf.com/content/ CVPR2025/html/Liang\_Revisiting\_Backdoor\_Attacks\_against\_Large\_Vision-Language\_Models\_from\_Domain\_Shift\_CVPR\_2025\_paper.html

[14] Jianhua Lin. 1991. Divergence Measures Based on the Shannon Entropy. IEEE Transactions on Information Theory 37, 1 (1991), 145–151. doi:10.1109/18.61115

[15] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. 2014. Microsoft COCO: Common Objects in Context. In ECCV. Springer, Cham, Switzerland, 740–755. doi:10.1007/ 978-3-319-10602-1\_48

[16] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved Baselines with Visual Instruction Tuning. In CVPR. IEEE/CVF, 26296– 26306. https://openaccess.thecvf.com/content/CVPR2024/html/Liu\_Improved\_ Baselines\_with\_Visual\_Instruction\_Tuning\_CVPR\_2024\_paper.htm

[17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual Instruction Tuning. In NeurIPS. Curran Associates, Inc., Red Hook, NY, USA, 34892–34916. https://arxiv.org/abs/2304.08485

[18] Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. 2018. Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. In RAID. Springer, Cham, Switzerland, 273–294. doi:10.1007/978-3-030-00470-5\_13

[19] Mingzu Liu, Hao Fang, and Runmin Cong. 2026. TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning. In ICML. 76168–76184. https://proceedings.mlr.press/v306/liu26ag.html

[20] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In NeurIPS. Curran Associates, Inc., Red Hook, NY, USA, 2507–2521. https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.htm

[21] Pan Lu, Liang Qiu, Jiaqi Chen, Tanglin Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. 2021. IconQA: A New Benchmark for Abstract Diagram Understanding and Visual Language Reasoning. In NeurIPS Datasets and Benchmarks Track. Curran Associates, Inc., Red Hook, NY, USA. https://openreview.net/forum?id=uXa9oBDZ9V1

[22] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. DocVQA: A Dataset for VQA on Document Images. In WACV. IEEE, Los Alamitos, CA, USA, 2200–2209. doi:10.1109/WACV48630.2021.00225

[23] Tuan Anh Nguyen and Anh Tuan Tran. 2021. WaNet: Imperceptible Warpingbased Backdoor Attack. In ICLR. OpenReview.net, Online. https://openreview. net/forum?id=eEn8KTtJOx

[24] Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2015. Flickr30k Entities: Collecting Regionto-Phrase Correspondences for Richer Image-to-Sentence Models. In ICCV. IEEE, Los Alamitos, CA, USA, 2641–2649. doi:10.1109/ICCV.2015.303

[25] Fanchao Qi, Mukai Li, Yangyi Chen, Zhengyan Zhang, Zhiyuan Liu, Yasheng Wang, and Maosong Sun. 2021. Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger. In ACL-IJCNLP. Association for Computational Linguistics, Online, 443–453. doi:10.18653/v1/2021.acl-long.37

[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In ICML. PMLR, Online, 8748–8763. https: //proceedings.mlr.press/v139/radford21a.html

[27] Xuankun Rong, Wenke Huang, Jian Liang, Jinhe Bi, Xun Xiao, Yiming Li, Bo Du, and Mang Ye. 2025. Backdoor Cleaning without External Guidance in MLLM Finetuning. In NeurIPS. Curran Associates, Inc., Red Hook, NY, USA, 28635–28663. doi:10.52202/085713-0851

[28] Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, and Ninghao Liu. 2023. Black-box Backdoor Defense via Zero-shot Image Purification. In NeurIPS. Curran Associates, Inc., Red Hook, NY, USA, 57336–57366. doi:10.52202/075280- 2505

[29] Brandon Tran, Jerry Li, and Aleksander Madry. 2018. Spectral Signatures in Backdoor Attacks. In NeurIPS. Curran Associates, Inc., Red Hook, NY, USA. https://proceedings.neurips.cc/paper\_files/paper/2018/hash/ 280cf18baf4311c92aa5a042336587d3-Abstract.html

[30] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2024. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. In ICLR. 18378–18394. https://proceedings.iclr.cc/ paper\_files/paper/2024/hash/50623630a2372839c078474efa6c0cb8-Abstract-Conference.html