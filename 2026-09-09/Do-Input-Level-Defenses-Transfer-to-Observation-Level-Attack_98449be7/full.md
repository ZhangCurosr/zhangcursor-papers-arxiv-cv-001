# Do Input-Level Defenses Transfer to Observation-Level Attacks on VideoLLMs?

Bangshuo Zhu , Wei Song , Yuxin Cao , Yuezhong Wu , Zhiquan Liu , Yuekang Li , Jingling Xue

Abstract—Video Large Language Models (VideoLLMs) are increasingly deployed in safety-critical applications such as content moderation and video analytics. To process long videos efficiently, VideoLLMs rely on frame sampling, token compression, and modality fusion, which together form an observation pipeline that reduces the raw video to a compact internal representation. Recent observation-level attacks exploit this pipeline to prevent the model from perceiving harmful content, yet no defense has been explicitly designed for this threat. We introduce DEFTEVAL, a controlled evaluation framework that systematically assesses whether input-level adversarial defenses, which operate on the pixel content of already-sampled frames, can mitigate observation-level attacks. Across five VideoLLMs, eleven representative defenses, and five attack types, we find that inputlevel defenses offer limited and inconsistent protection, with harmful detection rates frequently near zero. Critically, defenses fail even against attacks that embed harmful signals in every sampled frame, indicating that the bottleneck extends beyond sampling omission to the suppression of signals that do enter the model. Token compression discards localized features, and modality fusion systematically down-weights weakened visual signals. Furthermore, defense effectiveness is dominated by model architecture rather than by the defense method itself, and detection rates vary drastically across content categories, exposing structural weaknesses in temporal reasoning. These findings demonstrate that securing VideoLLMs requires systemlevel robustness mechanisms spanning sampling-aware coverage guarantees, token-level preservation of safety-relevant features, and modality-balanced fusion.

Index Terms—Video large language models, multimodal security, adversarial robustness, observation-level attacks, content moderation.

## I. INTRODUCTION

IDEO Large Language Models have emerged as a foundational paradigm for multimodal AI, extending large language models to videos. As intelligent systems increasingly operate in real-world environments, ranging from autonomous agents and assistive technologies to content moderation and video analytics, the ability to interpret videos has become critical [1]–[3]. By integrating visual encoders with large language models, VideoLLMs enable question answering over events and actions [4], [5], video summarization [6], [7], and decision support for downstream applications such as retrieval, assistive perception, and safety monitoring [8]–[10].

However, the scalability of VideoLLMs is constrained by the temporal redundancy of video streams [11]–[13]. Exhaustively processing all frames is computationally prohibitive, and thus practical systems rely on frame sampling mechanisms [11], [14], [15] to reduce input dimensionality. While essential for efficiency, this design creates a structural mismatch between the full video and the subset of frames actually observed by the model. As a result, the reasoning of a VideoLLM is conditioned on partial observation rather than on the complete input.

This dependence on sampled frames expands the attack surface. Instead of perturbing the observed input, an adversary may manipulate the sampling stage so that frames containing harmful or safety-relevant content are excluded from analysis. As illustrated in Fig. 1, a brief segment depicting policyviolating behavior can be embedded within an otherwise benign video yet omitted during frame selection. Consequently, VideoLLMs deployed in content moderation, safety monitoring, or automated review settings may produce falsenegative decisions despite the presence of harmful material. Such observation-level attacks have been empirically shown to substantially undermine the reliability of VideoLLMs [1], [2]. We define observation-level attacks as adversarial strategies that exploit the observation pipeline of VideoLLMs, including frame sampling, token compression, and modality fusion, to prevent the model from perceiving safety-relevant content, without relying on bounded input perturbations.

Observation-level attacks constitute a diverse threat family with multiple distinct mechanisms. Some attacks manipulate prompt-guided frame sampling so that harmful frames are never selected for inference (e.g., PoisonVID [2]). Others exploit sparse temporal sampling by embedding harmful content in short segments that are statistically unlikely to be sampled (e.g., FRA [1]). Still others embed harmful signals persistently across all frames, as a localized spatial overlay (PiP [1]) or a transparent blend (TOA [1]), yet the harmful content is nonetheless suppressed during token compression, spatial downsampling, or modality fusion. These distinct attack vectors may require fundamentally different defense strategies, and the extent to which each mechanism contributes to defense failure has not been studied.

This gap motivates our central research question. To what extent can input-level adversarial defenses mitigate observation-level attacks in VideoLLMs, and how do attack mechanisms, model architectures, and content categories determine their effectiveness or failure? We refer to input-level defenses as methods that operate on the pixel-level content of frames that have already been sampled, without interacting with the sampling mechanism itself, including input denoising [13], compression and reconstruction [3], [16], gradient smoothing [15], and patch removal [17]. Answering this question requires not only quantifying defense effectiveness but also explaining why defenses fail differently across attack types, whether model architecture matters more than the defense method, and what structural weaknesses in VideoLLMs the varying detection rates across content categories reveal.

To address this question, we develop the Defense Transferability Evaluation Framework (DEFTEVAL), a systematic evaluation framework for assessing input-level defenses under observation-level attacks in VideoLLMs. Using DEFTEVAL, we evaluate eleven representative inputlevel defenses, including ComDefend [16], DiffPure [18], Image Compression CNN (ICC) [19], Local Gradient Smoothing (LGS) [15], Patch-Agnostic Defense (PAD) [17], Pixel Deflection (PIXD) [20], Image Quilting (Quilting) [21], Randomized Padding (RAND) [22], Image Super-Resolution (SuperRes) [23], Total Variation Minimization (TVM) [24], and VideoPure [25], against five representative VideoLLMs (LLaVA-Video-7B-Qwen2, LLaVA-NeXT-Video-7B-DPO, LLaVA-Video-32B-Qwen, ShareGPT4Video-8B, and VideoLLaMA3) under five observation-level attacks. These attacks comprise three omission attacks, namely PoisonVID instantiated with the AKS and FRAG samplers [2] and FRA [1], and two suppression attacks, namely PiP and TOA [1].

Our evaluation reveals four key findings. First, input-level defenses provide limited and inconsistent protection. Across 825 evaluation cells, the harmful detection rate (HDR) is in the single digits or at zero for the overwhelming majority of configurations, and a four-way variance decomposition shows that which defense is applied is the least consequential of the four experimental factors. Second, this failure extends beyond sampling omission. Even when harmful signals are present in every sampled frame, defenses still fail broadly. We trace this failure to the model’s own visual pathway and verify the attribution by ablating the token-compression stage, confirming that compression is what discards the harmful evidence in these cases. Third, defense effectiveness is dominated by model architecture rather than by the defense method, with a subset of tested models accounting for the majority of defensive gains. Fourth, detection rates vary drastically across content categories. Pornographic content, which relies on static visual cues, is detected far more often than violence or crime, which require temporal reasoning. A video-native defense does not close this gap, because sparse frame sampling strips the input of the frame-to-frame continuity such defenses are built on, disabling them before they run.

In summary, we make the following contributions:

• Controlled evaluation framework. We introduce the Defense Transferability Evaluation Framework (DEFTEVAL), the first controlled framework for assessing whether input-level adversarial defenses transfer to observation-level attacks in VideoLLMs. DEFTEVAL is a four-way full factorial design over five VideoLLMs, eleven defenses spanning five families, five attack types, and three harmful-content categories, yielding 825 evaluation cells with the sampling budget, prompt, decoding policy, and response-to-label protocol held fixed throughout.

• A mechanism taxonomy for observation-level attacks. We formalize observation-level attacks as two distinct mechanisms. Under omission, harmful frames are never selected for inference. Under suppression, they are selected but do not survive the model’s visual pathway. Prior work treats these as a single phenomenon. Separating them is what makes the failure of input-level defenses a structural prediction rather than an empirical observation, since a defense applied strictly after sampling can address neither.

• Quantitative evidence. Through extensive experiments, we provide the first comprehensive quantitative evidence that input-level defenses offer limited and inconsistent mitigation against observation-level attacks, with harmful detection rates frequently near zero.

• Verified failure mechanisms. We test proposed attributions of defensive breakdown directly, rather than inferring them from aggregate outcomes. Ablating the tokencompression stage isolates the point in the visual pathway at which persistently present harmful signals are lost, and probing the alignment between frame embeddings and harmful-concept descriptions on both sides of each defense establishes whether a defense changes what the vision encoder recovers from the input at all. We further show that a video-native defense fares no better than its image counterparts, because sparse frame sampling removes the frame-to-frame continuity such defenses are built on.

## II. BACKGROUND AND RELATED WORK

## A. VideoLLMs

Video Large Language Models (VideoLLMs) extend language models from static text or image inputs to temporally structured visual streams by conditioning generation on both video content and user prompts [26]–[28]. By integrating visual perception with language reasoning, VideoLLMs enable unified video understanding and generation within a single autoregressive framework. They have rapidly emerged as a foundational paradigm for multimodal AI, supporting a wide spectrum of downstream tasks including video summarization [29]–[31], captioning [32], [33], question answering [26], [34], video grounding [35]–[37], and long-form video comprehension [38]–[40]. In these applications, the model receives a video-query pair and is expected to generate responses that are semantically consistent with and grounded in the visual evidence contained in the video [41].

![](images/43036b5c3fa16d904e865378e5df8c205832891a8dcf921929150b86e1867265.jpg)  
(a) Omission (e.g., PoisonVID, FRA). The harmful segment (Frame 3) is embedded in the video but excluded during frame sampling. The VideoLLM reasons only over benign frames and produces a false-negative safety decision.

![](images/dd9f06ed2233bca77d73b9c36ec185b33ad338c714e1f0cf2a54aacce531d0bb.jpg)  
(b) Suppression (e.g., PiP, TOA). Harmful content is present in every frame as a localized overlay, yet token compression and spatial downsampling within the visual encoder discard the fine-grained harmful features. The VideoLLM fails to detect the harmful content despite its continuous presence in the input.  
Fig. 1: Two categories of observation-level attacks against VideoLLMs. Both cause false-negative safety decisions, but through different mechanisms. In (a), harmful frames are never sampled. In (b), harmful signals enter all sampled frames but are suppressed during internal processing.

## B. Frame Sampling Strategies

Frame sampling methods can be categorized into three strategies: uniform sampling, semantic similarity sampling, and prompt-guided sampling [2].

Uniform frame sampling selects frames at fixed temporal intervals, typically ensuring inclusion of the first and last frames [42]. This strategy is widely adopted in systems such as Apollo [43] and various LLaVA- and LLaMA-based models [12], [31], [42]. Its simplicity and low overhead make it attractive in practice. However, by fixing sampling positions regardless of content distribution, uniform sampling often overlooks informative segments, especially in longer videos where salient events are sparsely distributed [43], [44].

To improve content coverage, semantic similarity-based sampling strategies perform denser initial sampling and then remove redundant frames based on visual similarity. For example, Semantic-aware Key-frame Extraction (SKE) [33] computes frame-level embeddings (e.g., using CLIP [45]) and measures pairwise cosine similarity to eliminate highly similar frames. By maximizing semantic diversity among retained frames, semantic similarity-based sampling aims to preserve representative visual content. However, because selection is agnostic to the user query, it may still retain frames irrelevant to the prompt. Moreover, similar to uniform frame sampling, some implementations always include boundary frames, which are frequently uninformative (e.g., blank or static scenes) [11].

More recently, prompt-guided sampling methods incorporate query relevance into the selection process. These approaches first sample candidate frames and then compute frame-prompt relevance scores using either a lightweight vision-language model (e.g., BLIP [46]) or other scoring mechanisms. Frames with higher relevance are prioritized for downstream encoding, improving localization and task performance. Representative methods include Differential Keyframe Selection (DKS) [47], Adaptive Keyframe Sampling (AKS) [11], and Frame Selection Augmented Generation (FRAG) [14]. DKS further accounts for local feature redundancy, AKS introduces coverage terms to encourage diversity, and FRAG emphasizes relevance ranking. While prompt-guided approaches generally improve task alignment, they may discard context that is weakly correlated with the prompt yet still semantically important. In addition, certain methods (e.g., FRAG) rely on full-scale VideoLLMs for relevance estimation, increasing computational overhead.

## C. Observation-Level Attacks

Existing research on VideoLLMs has largely focused on enhancing model capability and task performance, with comparatively limited attention to security risks arising from their input processing pipelines. Recent work has begun to explore such vulnerabilities. In particular, [1] identifies structural weaknesses in current architectures, including sparse temporal sampling, token under-sampling, and modality fusion imbalance, that enable uniform frame sampling- and semantic similarity sampling-based VideoLLMs to miss harmful content even when it is deliberately embedded in the video. PoisonVID [2] examines the security implications of more advanced sampling strategies. Rather than targeting uniform or similaritybased sampling alone, it investigates whether prompt-guided methods such as DKS [47], AKS [11], and FRAG [14] remain susceptible to manipulation.

## D. Adversarial Defenses

Existing adversarial defenses can be broadly grouped into adversarial training, input reconstruction, gradient smoothing, and patch removal approaches.

1) Adversarial Training: These methods [48], [49] enhance resilience by improving feature stability under adversarial perturbations. In VideoLLMs, adversarial training-based defenses commonly strengthen robustness by adversarially finetuning the vision encoder with augmented datasets, aiming to improve the stability of visual representations against bounded perturbations while leaving the language module unchanged. Representative works such as FARE [48] and SimCLIP [49] adversarially fine-tune CLIP [45], improving the stability of visual representations under perturbation-based threat models.

2) Input Reconstruction: These defenses [16], [18]–[24], [50] mitigate adversarial effects at inference time by suppressing or removing perturbations from the input signal. Reconstruction frameworks attempt to purify inputs by reconstructing visually faithful images from adversarial samples while filtering out adversarial noise, using techniques such as a compression-reconstruction network [16] or generating a close facsimile with an assembly of small, clean patches [21]. Denoise-only techniques such as ICC [19], PIXD [20], SuperRes [23], and TVM [24] perform no reconstruction, but instead attenuate adversarially perturbed regions of an image relative to its salient regions. Diffusion-based purification methods [13] similarly denoise corrupted inputs through iterative stochastic processes. Some systems, such as CIDER [50], incorporate an explicit detection stage prior to reconstruction to prevent unnecessary degradation of benign inputs. Despite differences in implementation, these approaches share a common premise: adversarial influence is assumed to appear as additive or structured perturbations on an otherwise fully observable input.

3) Gradient Smoothing: These methods mitigate adversarial perturbations by suppressing abnormal, localized gradient responses in feature representations. These methods are motivated by the observation that patch-based attacks often introduce spatially concentrated perturbations that distort activation patterns. Local Gradient Smoothing (LGS) [15] exemplifies this approach by identifying regions with abnormal gradient magnitude and applying localized smoothing to attenuate adversarial influence. Such defenses assume that malicious perturbations are embedded within an otherwise fully observable input and manifest as spatially localized anomalies.

4) Patch Removal: This category of defense deals specifically with adversarial patch attacks [51], [52], where the adversarial noise is confined to a small, localized patch of the target image. As opposed to traditional adversarial perturbations, where noise must be present in the entire image, the bounded area of an adversarial patch makes it a more realistic threat model in physical vision applications, as an adversarial patch can be printed out and used in the physical world [51]. Defenses against these attacks typically identify an adversarial patch then remove it with techniques like masking [53]–[55] or regularizing local region gradients [15].

5) Temporal Video Defenses: A small number of defenses operate on video rather than on individual frames, exploiting temporal structure that per-frame methods discard. VideoPure [25] exemplifies this class. It purifies a clip through diffusion in the latent space of a pretrained text-to-video model, denoising all frames jointly rather than independently, and guides the reverse process with optical flow estimated between adjacent frames so that the purified clip remains temporally coherent. Such defenses assume that the input is a temporally contiguous sequence whose consecutive frames are related by continuous motion, an assumption the frame sampling stage of a VideoLLM does not preserve.

## III. DEFTEVAL

## A. Formalism

Let a video be denoted by $V = ( x _ { 1 } , x _ { 2 } , \dots , x _ { T } )$ , where $x _ { t }$ is the t-th frame and $T$ is the video length. Given a user query $q , \mathbf { a }$ VideoLLM first applies a frame sampling policy to select a subset of frames for inference. We model the sampling as

$$
S = \pi ( V , q ) , \qquad S \subseteq \{ 1 , \ldots , T \} , \qquad | S | = K ,\tag{1}
$$

where $\pi ( \cdot )$ denotes the sampling mechanism $( \mathrm { e . g . }$ , uniform frame sampling, semantic similarity sampling, or promptguided sampling), and $K$ is the number of sampled frames. Let $V _ { S } \triangleq \{ x _ { t } \mid t \in S \}$ denote the sampled frames. The VideoLLM then produces an output (e.g., a textual description)

$$
y = f ( V _ { S } , q ; \theta ) = h \big ( g ( V _ { S } ) , q ; \theta \big ) ,\tag{2}
$$

where $f ( \cdot )$ denotes the end-to-end VideoLLM parameterized by $\theta ,$ decomposed into a visual pathway $g ( \cdot )$ (encoder, projector, and token compression) and a fused language model $h ( \cdot )$ that integrates the resulting visual tokens with the query. Equations (1) and (2) make explicit that the model’s prediction is conditioned on the observed subset $V _ { S }$ , and further on the compressed representation $g ( V _ { S } )$ rather than on $V _ { S }$ itself.

We consider observation-level adversarial attacks that exploit this observation pipeline. Unlike conventional adversarial examples that perturb the observed inputs by introducing bounded perturbations $( \mathbf { e . g . } , x _ { t } \mapsto x _ { t } + \delta _ { t }$ with $\lVert \delta _ { t } \rVert \leq \epsilon )$ , the attacker here does not rely on bounded input perturbations. Instead, the attacker constructs an attacked video $V ^ { \prime } = \mathcal { A } ( V )$ by inserting a harmful segment into an otherwise benign video, enabling policy-violating content to bypass VideoLLM-based video moderation and safety monitoring [1], [2]. Formally, let $S = \pi ( V , q )$ be the nominal selection set for a benign video-query pair $( V , q )$ . Under attack, the VideoLLM receives $V ^ { \prime } = \mathcal { A } ( V )$ and selects

$$
S ^ { \prime } = \pi ( V ^ { \prime } , q ) , \qquad V ^ { \prime } = \mathcal { A } ( V ) ,\tag{3}
$$

yielding the attacked output

$$
y ^ { \prime } = f ( V _ { S ^ { \prime } } ^ { \prime } , q ; \theta ) .\tag{4}
$$

Let ${ \mathcal { H } } ( V ^ { \prime } ) \subseteq \{ 1 , \dots , T \}$ denote the indices of frames in $V ^ { \prime }$ that contain harmful or safety-relevant content. The attacker’s objective is that $y ^ { \prime }$ remain consistent with the benign response y even though harmful content is present in $V ^ { \prime }$ . Two distinct mechanisms suffice to achieve this objective, namely

$$
\underbrace { S ^ { \prime } \cap \mathcal { H } ( V ^ { \prime } ) } _ { \mathrm { o m i s s i o n } } = \emptyset \quad \mathrm { o r } \quad \underbrace { g ( V _ { S ^ { \prime } } ^ { \prime } ) \approx g ( V _ { S } ) } _ { \mathrm { s u p p r e s s i o n } } ,\tag{5}
$$

where the omission condition, and in practice its relaxation $| S ^ { \prime } \cap \mathcal { H } ( V ^ { \prime } ) | \ll K$ , states that few or no harmful frames are selected, and the suppression condition states that harmful frames are selected but do not survive visual encoding and token compression, so the attacked and benign observations become indistinguishable to the language model. Omission is an objective the attacker pursues rather than one it can guarantee, and the residual harmful frames left behind when it falls short are precisely what an input-level defense has left to work with. The distinction matters for defense. Under omission the evidence is largely gone before any defense can act on it, whereas under suppression it is present in full at the defense’s input and is discarded only afterwards. Section V shows that both mechanisms defeat input-level defenses, and that the second is the more consequential of the two.

## B. Threat Model

We consider an adversary who aims to bypass VideoLLMbased video moderation or safety monitoring by inducing false-negative outputs, while leaving the underlying model parameters unchanged. The adversary can construct an attacked video $V ^ { \prime } = \mathcal { A } ( V )$ by inserting, replacing, or temporally relocating a short harmful segment within an otherwise benign video, or by overlaying one persistently across every frame. We assume the attacker does not modify the VideoLLM parameters θ and does not rely on bounded input perturbations on the observed frames.

The adversary’s leverage is at the observation stage. By manipulating the input video (and, for prompt-guided sampling, potentially the query or relevance cues), the attacker aims to satisfy either condition in Equation (5). Unless stated otherwise, we assume the attacker knows the number of frames (K) sampled by the VideoLLM, but does not require access to the model’s internals. Attacks that rely on suppression rather than omission do not require even this knowledge, and are fully black-box.

## C. Controlled Evaluation Design

DEFTEVAL is a controlled factorial design over the composition of a defense with an attacked observation pipeline. Prior work establishes that observation-level attacks succeed against undefended VideoLLMs [1], [2], that is, that $y ^ { \prime }$ in Equation (4) is benign-consistent when $V ^ { \prime }$ is attacked. DEFTEVAL asks the question that follows from this result. Given an input-level defense $d ( \cdot )$ applied to the frames the pipeline has already selected, does

$$
y _ { d } ^ { \prime } = f \big ( d ( V _ { S ^ { \prime } } ^ { \prime } ) , q ; \theta \big )\tag{6}
$$

recover the harmful evidence, and which properties of the configuration determine whether it does? Equation (6) makes the scope of an input-level defense explicit. Because $d$ is applied strictly after π, it can act only on $V _ { S ^ { \prime } } ^ { \prime }$ . It can never restore the frames $\mathcal { H } ( V ^ { \prime } ) \setminus S ^ { \prime }$ that sampling discarded, and it operates entirely upstream of $^ { g , }$ so it cannot influence which of its outputs survive token compression or how they are weighted during fusion. This scoping is what makes the failure of input-level defenses a structural prediction rather than an empirical accident.

Factors and levels. We treat the evaluation as a four-way full factorial design over the factors that a deployer can actually choose or encounter, namely the model architecture (5 levels), the defense (11 levels, spanning five families), the attack (5 levels, spanning the two mechanisms of Equation (5)), and the harmful-content category (3 levels). Crossing all four yields $5 \times 1 1 \times 5 \times 3 = 8 2 5$ evaluation cells, each estimated from 100 attacked videos. Full crossing is what distinguishes DEFTEVAL from a defense benchmark. It is the only design under which the contribution of the defense can be separated from that of the model, the attack, and the content, rather than confounded with them.

Controlled quantities. Every factor not under study is held fixed across all 825 cells. All attacks are constructed against the same sampling budget K and the same inference configuration. Each model is queried with a single fixed prompt template per content category, and is decoded deterministically (do\_sample disabled), so that no cell varies through generation randomness. Model responses are mapped to binary decisions by one fixed protocol, described in Section IV. Each attacked video carries exactly one harmful segment.

The undefended baseline is zero by construction. Every attacked video entering the study has already been verified to defeat the target model without any defense applied. The measured quantity, HDR (Equation (7)), is therefore conditioned on the attack having succeeded. It reports the fraction of already-failed cases that a defense recovers, and the nodefense baseline is identically 0% in every cell. This removes the principal confound in defense evaluation, namely that a defense may appear effective merely because the underlying attack was weak, and means any non-zero HDR is attributable to the defense alone.

Analysis plan. The design supports two complementary analyses, fixed in advance. First, the distribution of HDR over all cells characterizes how often any configuration achieves meaningful recovery, independently of which configuration it is. Second, a four-way ANOVA with $\eta ^ { 2 }$ effect sizes apportions the variance in HDR among the four factors, answering which of them governs the outcome. All interaction terms are pooled into the residual, so the reported $\eta ^ { 2 }$ values are conservative lower bounds on each factor’s influence. Together these analyses answer the two halves of our research question, namely how well input-level defenses transfer and what determines whether they transfer at all.

Relation to prior work. The attacks we evaluate were introduced by [1] and [2], which measure attack success against undefended models and attribute it to sampling sparsity and prompt-guided relevance manipulation, respectively. Neither work evaluates a defense. DEFTEVAL addresses the complementary and previously open question of whether the existing input-level defense literature transfers to this threat model. It is the factorial design above, rather than any individual measurement, that makes the question answerable. A single defense and model pair cannot distinguish a weak defense from a model that no defense can help, whereas the variance decomposition over 825 crossed cells can.

## IV. EVALUATION SETUP

## A. VideoLLMs

We evaluate five advanced VideoLLMs: LLaVA-Video-7B-Qwen2 (L-7B), LLaVA-NeXT-Video-7B-DPO (L-7B-DPO), LLaVA-Video-32B-Qwen (L-32B) [56], ShareGPT4Video-8B (SG4V) [33], and VideoLLaMA3 (VL3) [30]. All models are tested under deterministic generation settings, with ${ \mathsf { d o } } _ { - } s$ ample set to False, ensuring that model behavior is not affected by sampling variance.

## B. Attacks

Table I summarizes the five observation-level attacks we evaluate, namely PoisonVID instantiated over two promptguided samplers [2], and the Frame Replacement (FRA), Picture-in-Picture (PiP), and Transparent Overlay (TOA) attacks of [1]. As formalized in Equation (5), omission attacks aim to keep harmful frames out of the sampled subset. When omission is incomplete, residual harmful frames survive into the observation. These residual cases carry interpretive weight. They are the only PoisonVID and FRA instances in which an input-level defense has any harmful evidence to act on at all, and they account for the non-zero HDR observed under those attacks in Table III. Suppression attacks place harmful content in every sampled frame by construction and rely on it being discarded somewhere along the visual pathway. For PiP and TOA, the injected content is deliberately kept perceptible. The overlay region is large enough to be read at a glance and loops for the full duration, and the blending coefficient α is chosen so that the harmful clip remains clearly visible. This is what makes their failure mode consequential. A human moderator reviewing the same video would see the violation immediately, while the VideoLLM does not. All attacks are constructed against the same sampling budget, $K = 3 2$

## C. Defenses

We evaluate eleven defenses, summarized in Table II. Selection follows two criteria. First, a defense must be applicable at inference time without retraining or modifying the VideoLLM. This excludes adversarial-training approaches such as FARE [48] and SimCLIP [49], which adversarially fine-tune the vision encoder and therefore alter θ, violating the fixed-model assumption of our threat model (Section III-B). Second, a defense must have a public, reproducible implementation. Ten of the eleven are image defenses spanning the five families identified in Table II. The eleventh, VideoPure [25], is a video defense included to test whether operating on the temporal dimension changes the outcome.

The ten image defenses differ in mechanism but share a single premise, namely that the adversarial signal is a bounded, structured perturbation superimposed on an input the model can otherwise fully observe. Reconstruction and purification methods re-synthesize the frame from a clean prior, using a learned compression and reconstruction network (ComDefend), a forward-then-reverse diffusion process (DiffPure), a database of clean image patches (Quilting), or a superresolution network that remaps off-manifold samples back onto the natural image manifold (SuperRes). Compression and denoising methods attenuate perturbed regions relative to salient ones through lossy re-encoding (ICC), pixel redistribution followed by wavelet denoising (PIXD), or total variation minimization (TVM). Input randomization (RAND) destroys the spatial structure of perturbations by random resizing and padding. Patch localization and removal methods target the bounded-patch threat model, either regularizing regions of abnormal gradient magnitude (LGS) or localizing patches via semantic independence and spatial heterogeneity before suppressing them (PAD). Each image defense is applied independently to every sampled frame, and the purified frame set is passed unchanged to the downstream VideoLLM pipeline.

We include VideoPure to test whether input-level defenses fail simply because they are applied frame by frame and therefore discard the temporal dimension. If that were the whole account, a defense that models the clip jointly should recover harmful content the image defenses miss. In our setting, VideoPure operates on the full 32-frame sampled clip at once, purifying with a shallow inversion timestep and flow guidance from a RAFT estimator [57], then resampling to the source resolution before inference.

## D. Data and Metric

We follow the threat model and attack construction protocols in prior work [1], [2]. Specifically, we instantiate harmful content using three policy-violating categories, namely explicit sexual content, crime-related content, and violence, to reflect common safety-critical moderation scenarios. For each attack, we construct 100 successfully attacked videos by inserting a harmful clip into an otherwise benign source video. Unless stated otherwise, each attacked example contains a single harmful segment and has already been verified as successfully attacked, and all attacks are evaluated under the same K and inference configuration to enable controlled comparisons.

Our primary metric measures how often a VideoLLM correctly flags the presence of harmful content after applying a defense. Let $\mathcal { D } _ { \mathrm { h a r m } }$ denote the set of videos that contain harmful content (ground truth $z = 1 )$ . Given a defense method $d ( \cdot )$ , we run the model on each input and map the VideoLLM’s response to a binary decision $\hat { z } \in \ \{ 0 , 1 \}$ , where $\hat { z } ~ = ~ 1$ indicates that the model identifies the queried harmful content and $\hat { z } = 0$ indicates it does not. Let $\mathcal { D } _ { \mathrm { t e s t } } \subseteq \mathcal { D } _ { \mathrm { h a r m } }$ denote the subset of harmful videos that the undefended model fails to detect, that is, those with $z = 1$ and $\hat { z } = 0$ when no defense is applied. We report the Harmful Detection Rate (HDR)

$$
\mathrm { H D R } ( d ) = \frac { 1 } { | \mathcal { D } _ { \mathrm { t e s t } } | } \sum _ { ( V , q ) \in \mathcal { D } _ { \mathrm { t e s t } } } \mathbb { I } \big [ \hat { z } \big ( f ( d ( V ) , q ) \big ) = 1 \big ] ,\tag{7}
$$

which is the fraction of previously undetected harmful videos that are correctly detected once defense d is applied (higher is better).

For the VideoLLM’s response, a clear “yes” answer such as “Yes, the video appears to contain explicit content that is typically associated with adult entertainment.” is marked as an identification, and a clear “no” answer such as “No, the video does not contain pornographic content[...]” is marked as a negative identification. Responses like “I’m sorry, but I can’t assist with that.” are treated as “yes” answers, as such refusals are assumed to be triggered by the model’s internal safety alignment. Responses that answer “yes” but explicitly refer to a subject other than the harmful content are considered false positives and treated as “no” answers.

## V. RESULTS AND ANALYSIS

Table III presents the harmful detection rate (HDR, Equation (7)) for each combination of eleven defenses, five

<table><tr><td>Attacks</td><td>Mechanism</td><td>Injection</td><td>Stage exploited</td><td>Harmful signal in sampled frames</td></tr><tr><td>PoisonVID (AKS) [2] PoisonVID (FRAG) [2] FRA [1]</td><td>Omission</td><td>A harmful clip is inserted into, or substituted for a segment of, an otherwise benign video</td><td>The frame selection policy π, via its relevance scoring (PoisonVID) or its temporal sparsity (FRA)</td><td>Ideally none; residual harmful frames may survive when omission is incomplete</td></tr><tr><td>PiP [1] TOA [1]</td><td>Suppression</td><td>A harmful clip is composited into every frame, as a fixed-position overlay (PiP) or by alpha-blending at opacity α (TOA)</td><td>The visual pathway g: token compression and spatial downsampling (PiP), or modality fusion imbalance (TOA)</td><td>Every frame, by construction; spatially localized (PiP) or attenuated (TOA)</td></tr></table>

TABLE I: The five observation-level attacks evaluated in DEFTEVAL, grouped by the mechanism of Equation (5) they realize.
<table><tr><td>Defense</td><td>Family</td><td>Assumed adversary</td></tr><tr><td>ComDefend [16] DiffPure [18] Quilting [21] SuperRes [23]</td><td>Reconstruction / purification</td><td>An additive perturbation that displaces an otherwise fully observable frame off the natural image manifold, and can be undone by re-synthesizing the frame from a clean prior.</td></tr><tr><td>ICC [19] PIXD [20] TVM [24]</td><td>Compression / denoising</td><td>A low-amplitude, high-frequency perturbation concentrated away from salient regions, removable by attenuating fine detail without reconstruction.</td></tr><tr><td>RAND [22]</td><td>Input randomization</td><td>A perturbation whose spatial structure is brittle: it survives only at the scale and alignment at which it was optimized.</td></tr><tr><td>LGS [15] PAD [17]</td><td>Patch localization &amp; removal</td><td>A bounded adversarial patch, localizable by its abnormal local gradients or by its semantic independence from the surrounding scene.</td></tr><tr><td>VideoPure [25]</td><td>Video / temporal</td><td>A perturbation spanning the clip that is temporally consistent, and can be suppressed by denoising frames jointly under a motion constraint.</td></tr></table>

TABLE II: The eleven defenses evaluated in DEFTEVAL, grouped by family and by the adversary each family assumes.

VideoLLMs, five observation-level attacks, and three content categories, yielding 825 evaluation cells. We organize our analysis around four findings that collectively characterize the transferability of input-level defenses to observation-level threats and directly address the research question posed in Section I. Each finding presents its empirical evidence together with the mechanism that explains it, and, where we can test that mechanism directly, the ablation that does so.

## A. Finding 1: Input-Level Defenses Provide Limited and Inconsistent Protection

Table III shows that the vast majority of configurations report HDR values in the single digits or at zero, with only a minority attaining moderate or high HDR, reinforcing our conclusion that input-level defenses provide limited and inconsistent protection against observation-level attacks.

Prompt-guided sampling attacks are particularly resilient. Under PoisonVID (both AKS and FRAG), defenses are least effective. For L-7B-DPO, both PoisonVID variants yield 0% HDR across all categories and all eleven defenses. Even on other models, PoisonVID detection rates typically remain in the single digits. DiffPure on L-7B achieves only 3 to 5% under AKS and 5 to 15% under FRAG, and PAD shows similar magnitudes. This outcome is consistent with the core threat model. When sampling manipulation excludes harmful frames from inference, defenses that operate on the already-sampled frames have no harmful evidence to recover.

Choice of defense is inconsequential. As shown in Table IV, of the four experimental factors, the defense is the least important determinant of HDR. Model architecture $( \eta ^ { 2 } ~ =$ 25.4%), harmful-content category (9.4%), and attack type (7.9%) each explain more variance than which of the eleven defenses is applied (3.1%). Statistically, the choice of defense is thus the least consequential choice in the entire evaluation. These results confirm that input-level defenses offer, at best, narrow and attack-specific improvements rather than general protection against observation-level threats.

Why omission attacks resist defense. For PoisonVID, the adversary directly targets the relevance scoring mechanism in prompt-guided sampling, suppressing the selection probability of harmful frames so that they are systematically excluded from the sampled subset. Input-level defenses, which operate on frames after sampling, cannot recover evidence that was never selected. Similarly, FRA exploits the sparsity of uniform temporal sampling. By localizing the harmful segment in a short interval, it ensures that the probability of any sampled frame falling within the injected region is low. This constitutes a fundamental threat-model mismatch, since input-level defenses are designed to purify observed content, not to ensure coverage of unobserved content.

Defenses do not restore harmful semantics. To test whether defenses recover safety-relevant content rather than merely altering pixels, we embed all K=32 sampled frames with the SigLIP-so400m-384 tower [58] and measure their cosine similarity to harmful-concept prompts for the video’s category, before and after each defense. The two attack mechanisms require different measures. Under omission, the harmful clip occupies roughly one sampled frame, so we track that frame’s prominence within its own clip (max − median), which is unaffected by uniform shifts in the embedding distribution. Under suppression, the clip is composited into every frame and no such contrast exists, so we instead isolate the concept-specific part of the shift, defined as the change under the video’s own harmful concept minus the change under the two non-matching concepts, which cancels global drift. Across both mechanisms (Table V), harmful alignment never meaningfully increases, with the only positive entries being +1.3% (ComDefend, omission) and +0.1% (LGS, suppression). Input-level defenses therefore have no mechanism to restore safety-relevant signal. They can only remove perturbation, which is the wrong operation against attacks that hide harmful content in unperturbed frames.

<table><tr><td></td><td></td><td colspan="3">L-7B</td><td colspan="3">L-7B-DPO</td><td colspan="3">L-32B</td><td colspan="3">SG4V</td><td colspan="3">VL3</td><td colspan="3"></td></tr><tr><td>Defense</td><td>Attack PoisonVID (AKS)</td><td>Cri 7</td><td>Por 21</td><td>Vio 4</td><td>Cri 0</td><td>Por 0</td><td>Vio 0</td><td>Cri 10</td><td>Por 7</td><td>Vio 14</td><td>Cri</td><td>0</td><td>Por</td><td>Vio</td><td>Cri</td><td>Por</td><td></td><td>Vio</td></tr><tr><td>ComDefend</td><td>PoisonVID (FRAG) FRA PiP TOA</td><td></td><td>16 3 4 2</td><td>19 1 2 1</td><td>0 0 0 0</td><td>0 0 0 1</td><td></td><td>27</td><td>4 18</td><td>34 29 2 12 4</td><td>5 2</td><td>1 2 0 2</td><td>2 2 0 1 0</td><td>1 3 0 0 2</td><td>9 17 7 3 3</td><td>21 28 31 25 6</td><td></td><td>6 3 4 1 5</td></tr><tr><td>DiffPure</td><td>PoisonVID (AKS) PoisonVID (FRAG) FRA PiP TOA</td><td>3 6 2 1 0</td><td>5 5 2 3 1</td><td>5 15 2 1 0</td><td>0 0 0 0</td><td>0 0 0 0</td><td></td><td>2 10</td><td>8 37 3 27</td><td>10 16 3 23</td><td></td><td>0 0 1 0</td><td>7 22 1</td><td>0 1 0 0</td><td>4 9 8 6</td><td>14 17 12 34</td><td></td><td>4 5 6 3 5</td></tr><tr><td>ICC</td><td>PoisonVID (AKS) PoisonVID (FRAG) FRA PiP TOA</td><td>1 0 0 1 0</td><td>1 2 0 1 3</td><td>4 3 0 3 0</td><td>1 0 0 0 0</td><td>0 0 0 0</td><td></td><td>5 4 9 5 1 4</td><td>3 8 32 9 8</td><td>6 7 9 5</td><td>0 0 1 0 0</td><td></td><td>0 17 10 2 1</td><td>2 1 2 0 0</td><td>1 7 11 6 4</td><td>4 25 37 34 16</td><td></td><td>7 5 9 7</td></tr><tr><td>LGS</td><td>PoisonVID (AKS) PoisonVID (FRAG) FRA PiP TOA</td><td>2 9 1 1 1</td><td>7 7 0 4 3</td><td>4 17 1 0 1</td><td>0 0 0 0</td><td>0 0 0 0</td><td>0 0</td><td>3 6 4 3</td><td>6 32 2 14</td><td>1 3 3 6</td><td>1 4 2 0</td><td></td><td>2 6 1 1</td><td>1 5 1 0</td><td>3 9 8 10</td><td>6 14 8 13</td><td></td><td>2 4 6 5</td></tr><tr><td>PAD</td><td>PoisonVID (AKS) PoisonVID (FRAG) FRA PiP TOA</td><td>3 6 2 1</td><td>5 5 2 3</td><td>5 15 2 1</td><td>0 0 0 0</td><td>0 0 0 0</td><td></td><td>6 15 4</td><td>5 9 41 1 42</td><td>2 9 9</td><td>1 3 4</td><td></td><td>0 9 8 5</td><td>2 5 7 3</td><td>3 2 7 7</td><td>8 12 21 11 36</td><td></td><td>3 0 4 2 5</td></tr><tr><td>PIXD</td><td>PoisonVID (AKS) PoisonVID (FRAG) FRA PiP TOA</td><td>1 0 0 1 3</td><td>0 2 0 1</td><td>4 3 0 3</td><td>0 0 0</td><td>0 0 0 0</td><td></td><td></td><td>9 36 6</td><td>16</td><td></td><td>0 0 0</td><td>14 8 2 0</td><td>1 2 0 0</td><td>5 4 3 3</td><td>20 23 27 15</td><td></td><td>3 4 25</td></tr><tr><td>Quilting</td><td>PoisonVID (AKS) PoisonVID (FRAG) FRA PiP TOA</td><td>1 3 6 1 0</td><td>3 5 1 0</td><td>1 2 0 0</td><td>0 0 0 0</td><td>0 0 0 1</td><td>0 0</td><td>4 7 12 7 5</td><td>7 10 38 10 8</td><td>0 7 15 8</td><td>0 0 0 0</td><td>0</td><td>0 0 0 1 0</td><td>1 1 0 1 2</td><td>1 6 7 0 2</td><td>0 16 23 10 26</td><td></td><td>2 2 1 2</td></tr><tr><td>RAND</td><td>PoisonVID (AKS) PoisonVID (FRAG) FRA PiP</td><td>1 1 1 1</td><td>1 22 0</td><td>0</td><td>0 4 0 1 0 1 0</td><td>1 1 0 0</td><td></td><td>0 10 0</td><td>3 4 21 4</td><td>0 4 2</td><td>0 1 11 5</td><td>4 0 3 0</td><td>1 10 10 4</td><td>3 0 3 0</td><td>0 7 10 5</td><td></td><td>4 2 9 11</td><td>1 1 2 ∞</td></tr><tr><td>SuperRes</td><td>TOA PoisonVID (AKS) PoisonVID (FRAG) FRA PiP</td><td>0 2 1 3 1</td><td>2 1 4 1</td><td></td><td>1 0 2 1 1 0 0</td><td>4 0 0 0</td><td></td><td>0 0 0 0</td><td>3 4 3 2 3</td><td>6 8 0 12 4</td><td>20 4 4 1</td><td>1 0 2 0</td><td>22 3 2 4</td><td>22 2 4 2</td><td>4 3 2 1</td><td>11 17</td><td>85 4</td><td>3 1 3 0 3</td></tr><tr><td>TVM</td><td>TOA PoisonVID (AKS) PoisonVID (FRAG) FRA PiP</td><td>0 3 7 0 0</td><td>0 1 3 6 0</td><td>0</td><td>0 7 5 2</td><td>0 0 0 0</td><td>2 0 0 0</td><td>0 0 0 0</td><td>1 11 16 9</td><td>2 10 26 3</td><td>0 6 21 8</td><td>0 0 5 0</td><td>1 12 7 6</td><td></td><td>1 0 4 0</td><td>1 12 13 7</td><td>2 12 17 20</td><td>0 1 4 1</td></tr><tr><td></td><td>TOA PoisonVID (AKS) PoisonVID (FRAG) FRA</td><td>0 0 0</td><td>1 2 3 1</td><td>2 0 0 4</td><td></td><td>0 0 0 0 0</td><td>0 0 0</td><td>0 0 0</td><td>3 7 2 3</td><td>14 7 3 31</td><td>7 1 3 6</td><td>0 0 0 2</td><td>1 4</td><td>1 0</td><td></td><td>6 3 3</td><td>2 12 22</td><td>1 2 6</td></tr></table>

TABLE III: Harmful detection rate (HDR, %) after applying input-level defenses under observation-level attacks. Each defense is evaluated on five attack types across three harmful-content categories and five VideoLLMs. Bold entries indicate HDR ≥ 25%. Higher is better.

<table><tr><td>Term</td><td>SS</td><td>df</td><td>F</td><td>p</td><td> $\eta ^ { 2 }$  (%)</td></tr><tr><td>Model</td><td>9013</td><td>4</td><td>94.1</td><td> $< 1 0 ^ { - 4 }$ </td><td>25.4</td></tr><tr><td>Category</td><td>3356</td><td>2</td><td>70.0</td><td> $< 1 0 ^ { - 4 }$ </td><td>9.4</td></tr><tr><td>Attack</td><td>2812</td><td>4</td><td>29.3</td><td> $< 1 0 ^ { - 4 }$ </td><td>7.9</td></tr><tr><td>Defense</td><td>1097</td><td>10</td><td>4.6</td><td> $< 1 0 ^ { - 4 }$ </td><td>3.1</td></tr><tr><td>Residual (interactions)</td><td>19263</td><td>804</td><td>一</td><td></td><td>54.2</td></tr></table>

TABLE IV: Variance decomposition of Table III (HDR). Fourway ANOVA over the 825 cells; all interaction terms are pooled into the residual. $\eta ^ { 2 }$ is the share of total HDR variance explained and sums to 100% across the table.

<table><tr><td rowspan="2">Defense</td><td colspan="2">∆ harmful alignment</td></tr><tr><td>Omission</td><td>Suppression</td></tr><tr><td>ComDefend</td><td>+1.3%</td><td>-8.1%</td></tr><tr><td>DiffPure</td><td>-5.7%</td><td>-18.0%</td></tr><tr><td>ICC</td><td>-1.8%</td><td>-2.2%</td></tr><tr><td>LGS</td><td>-0.7%</td><td>+0.1%</td></tr><tr><td>PAD</td><td>-5.8%</td><td>-4.2%</td></tr><tr><td>PIXD</td><td>-2.9%</td><td>-1.8%</td></tr><tr><td>Quilting</td><td>-2.6%</td><td>-25.1%</td></tr><tr><td>RAND</td><td>-1.0%</td><td>-7.5%</td></tr><tr><td>SuperRes</td><td>-0.1%</td><td>-0.2%</td></tr><tr><td>TVM</td><td>-2.6%</td><td>-17.7%</td></tr><tr><td>VideoPure</td><td>-21.1%</td><td>-29.3%</td></tr></table>

TABLE V: Change in harmful-concept alignment after each input-level defense, measured separately for each attack mechanism. Negative values mean harmful content became less distinguishable.

## B. Finding 2: Defenses Fail Even When Harmful Signals Persist in Every Frame

A critical test of input-level defenses is their performance under suppression attacks, such as PiP (spatially localized overlay) and TOA (transparent alpha-blending), where harmful content is present in every sampled frame. If defense failure were attributable solely to sampling omission, these attacks should be substantially easier to defend against. This is not the case. Suppression attacks yield lower detection rates overall. Pooled over all defenses, models, and categories, mean HDR is 4.9% for the omission attacks against 3.0% for the suppression attacks.

<table><tr><td rowspan="2">Model</td><td colspan="3"> $\Delta$ </td></tr><tr><td>Cri</td><td>Por</td><td>Vio</td></tr><tr><td>L-7B</td><td>+0.0</td><td> $+ 4 . 0$ </td><td> $+ 2 . 5$ </td></tr><tr><td>L-7B-DPO</td><td>+6.5</td><td> $+ 6 . 8$ </td><td> $+ 0 . 2$ </td></tr><tr><td>L-32B</td><td>+17.0</td><td> $+ 1 4 . 8$ </td><td> $+ 6 . 2$ </td></tr><tr><td>SG4V</td><td>+0.2</td><td> $+ 2 . 2$ </td><td> $+ 4 . 2$ </td></tr><tr><td>VL3</td><td>+4.8</td><td> $+ 2 8 . 0$ </td><td> $- 4 . 0$ </td></tr></table>

TABLE VI: Effect of ablating visual-token compression. Each entry is the change in HDR (percentage points) when the compression stage is removed, averaged over the four combinations of DiffPure and PAD with PiP and TOA.

TOA defeats all defenses across nearly all configurations. Under TOA, HDR remains below 5% across most model and defense combinations. For L-7B and L-7B-DPO, nearly all cells report 0 to 2%. Even the best-performing defense and model pair (PAD on L-32B) reaches only 23% on Porn and 11% on Crime, while all other TOA cells for L-32B stay below 10%.

PiP shows sporadic recovery, but only for specific model and category pairs. Under PiP, results are similarly suppressed for most models, with the notable exception of L-32B on Porn. Yet even in these cases, Crime and Violence categories under PiP remain substantially lower, indicating that recovery is highly category-dependent rather than a general defense capability.

These results demonstrate that the failure of input-level defenses extends beyond sampling omission. We hypothesize that token compression and spatial downsampling, applied to fit long videos within the context window, selectively discard visual tokens from localized or low-contrast regions, stripping away the harmful features even when they are spatially present. Furthermore, during multimodal integration, the modality fusion mechanism systematically down-weights attenuated visual signals relative to the language pathway. As a result, the model never effectively processes the safety-relevant evidence, even though it physically enters the input.

Removing token compression restores detection. We test this hypothesis by ablating the compression stage of the tested models. We run the ablation on DiffPure and PAD, the two strongest defenses against PiP and TOA. Every VideoLLM compresses its visual tokens, but through a different mechanism, so the ablation differs per architecture. For the LLaVA-NeXT-Video models we set mm\_spatial\_pool\_stride from 2 to 1, removing the 2×2 pool over each frame’s patch grid. SG4V has no spatial pool and instead tiles all 32 frames into a single montage, so we widen image\_grid\_pinpoints from 2×2 to 4×4. For VL3 we disable use\_token\_compression and set video\_merge\_size to 1.

Removing compression raises pooled HDR from 4.8% to 11.0%, and the gain concentrates in exactly the models the hypothesis predicts (Table VI). L-32B gains +17.0 points on Crime and +14.8 on Porn, and VL3 gains +28.0 on Porn. The two models that detect almost nothing under any defense, L-7B and SG4V, barely move (+0.0 to +4.2), because where the underlying representation carries no usable harmful signal, restoring the token budget has no effect.

This finding is non-trivial. It demonstrates that the problem is not merely a threat-model mismatch, but an architectural fragility in how VideoLLMs process and integrate visual information.

## C. Finding 3: Defense Effectiveness Is Dominated by Model Architecture

As shown in Table IV, model architecture explains roughly eight times more variance in HDR than the choice of defense $( \eta ^ { \bar { 2 } } = 2 5 . 4 \%$ against 3.1%), with detection capability concentrated in two models. Averaged over all defenses, attacks, and categories, L-32B and VL3 reach 8.2% and 7.9% HDR, against 2.4% for L-7B, 2.0% for SG4V, and 0.1% for L-7B-DPO. This gap between the two groups holds under every one of the eleven defenses, and the stronger group accounts for 78% of all harmful detections.

Consistently low-HDR models. L-7B-DPO yields HDR near 0% under virtually all 120 defense, attack, and category configurations. SG4V shows similarly negligible rates, at 0 to 2% under PoisonVID and PiP, with only isolated non-zero values under PAD and DiffPure for specific categories. Notably, switching the defense method, from simple compression (ICC) to diffusion-based purification (DiffPure) to patch removal (PAD), produces no meaningful improvement on either model.

Comparatively higher-HDR models. L-32B shows moderate recovery under specific conditions. Under PoisonVID (FRAG) on Porn, it reaches 41% (PAD), 37% (DiffPure), and 32% (ICC). VL3 exhibits elevated Porn detection under several attacks, reaching 34% (ICC), 31% (ComDefend), and 27% (PIXD) under FRA. However, even on these models, the majority of cells remain below 15%.

This pattern indicates that model-internal factors, including visual representation quality, token retention policy, and safety alignment strategy, constitute the primary bottleneck for defense effectiveness. These factors jointly determine whether a defense can extract residual harmful signals from the processed input. When they are unfavorable, no input-level defense can compensate. From a deployment perspective, the choice of model architecture may be more consequential for robustness than the choice of defense method.

## D. Finding 4: Content-Category Disparity Exposes Temporal Reasoning Weakness

The highest HDR values in Table III are overwhelmingly concentrated in the Porn category, which contains the ten highest HDR values in the entire table. Under FRA and PiP, the Porn advantage persists, while Crime and Violence detection rates rarely exceed 15% under any combination of defense, model, and attack.

This disparity reflects a fundamental difference in the semantic nature of content categories. Pornographic content is typically defined by explicit, localized visual cues that enable reliable single-frame classification [59]. Violence and crime recognition, however, are inherently context-dependent and require temporal reasoning to distinguish between visually ambiguous actions (e.g., play versus harm) [60]. The compression ablation of Section V-B bears this out. Spatial pooling discards intra-frame detail while leaving the number of sampled frames, and hence the available temporal evidence, entirely unchanged. Removing it therefore replenishes the resource on which localized, single-frame detection depends while doing nothing to relieve the temporal-integration bottleneck that governs Crime and Violence. Accordingly, Porn exhibits an outsized HDR improvement under reduced compression (Table VI). The sparse frame sampling and limited temporal modeling of current VideoLLMs make them more reliant on static visual cues, reducing their capacity for fine-grained temporal reasoning. This structural weakness is not addressable by input-level defenses, which operate at the pixel level and do not enhance temporal understanding.

A temporal defense does not close the gap. VideoPure is the one defense in our study built for video rather than for images, and it is therefore the natural candidate to recover the temporally defined categories. It does not. Averaged over all attacks and models, it attains 3.4% HDR, no better than the 4.2% mean of the ten image defenses, and its category profile is more skewed, not less. It reaches 7.3% on Porn against 1.6% on Crime and 1.4% on Violence, a 4.8× gap where the image defenses show 2.5×. The reason is that the defense never sees a video. VideoPure enforces temporal consistency by estimating optical flow between adjacent frames, but it is applied, as every defense here is, to the $K = 3 2$ frames the sampler has already selected. Those frames span the full duration of the source video, so consecutive frames in the purified clip may be seconds apart, and the flow field estimated between them does not correspond to any real motion. Sparse temporal sampling strips the input of the frame-to-frame continuity that video-native defenses are built on, disabling the one class of defense equipped for this modality before it runs. Any defense premised on motion coherence or temporal smoothness inherits the same limitation.

Implications. This disparity has consequences beyond the specific content categories tested. It suggests that the safetymonitoring capabilities of VideoLLMs are structurally biased toward content that can be identified from static visual features, while content requiring temporal reasoning, including sequential actions, contextual violence, and escalating threats, is systematically under-detected. As adversaries become aware of this bias, they can preferentially embed temporally complex harmful content (e.g., instructional crime or contextual harassment) that exploits the models’ temporal reasoning deficiency. Addressing this weakness requires not only improved temporal modeling within VideoLLMs but also defense mechanisms that are specifically designed to enhance temporal coverage and context retention during the observation pipeline.

## VI. DISCUSSION

Section V established four empirical findings and, for each, the mechanism that produces it. Input-level defenses provide limited protection because they act only on already-sampled frames. This failure persists even when harmful content is present in every frame, because token compression discards it before the language model sees it. Model architecture dominates defense effectiveness, and detection rates vary drastically across content categories. We now turn from why the tested defenses fail to what a defense that does not fail would have to do.

## A. Toward System-Level Robustness

Our study does not argue that input-level robustness techniques are unimportant; rather, it clarifies their scope. They can improve robustness to perturbations conditional on the model having access to the safety-relevant visual content, but they do not guarantee robustness when the observation pipeline, spanning frame sampling, token compression, and modality fusion, prevents the model from perceiving that content in the first place. This motivates future defense directions that address the full observation pipeline:

• Sampling-aware coverage guarantees. Strategies that ensure safety-relevant temporal segments are represented in the sampled subset, regardless of adversarial manipulation of relevance scores.

• Token-level preservation. Mechanisms that preserve localized and low-contrast visual features during compression, preventing safety-relevant signals from being discarded by token budgeting. Our compression ablation (Section V-B) demonstrates that this direction is viable, since restoring the token budget more than doubles pooled HDR.

• Modality-balanced fusion. Architectures that prevent systematic down-weighting of attenuated visual signals during multimodal integration, ensuring that weakened but present harmful content still influences the model’s output.

• Observation verification. Post-hoc mechanisms that detect when sampling or compression is likely to have discarded critical content, triggering re-examination of the input.

More broadly, our results point to a gap between inputlevel robustness and system-level robustness in multimodal architectures with structured observation pipelines.

## B. Limitations

Our evaluation focuses on representative VideoLLMs, sampling mechanisms, and observation-level attacks that are currently available, and on a detection-style safety task with a standardized response-to-label mapping. While this setting enables controlled comparisons, other tasks (e.g., long-form summarization and multi-turn QA) may exhibit additional failure modes. Moreover, we adapt input-level defenses to the video setting by applying them to sampled frames; alternative integration strategies (e.g., operating on latent features or jointly with sampling) may yield different trade-offs. Finally, our findings characterize transferability under the tested configurations and should not be interpreted as a definitive impossibility result for all future defenses.

## VII. CONCLUSION

This paper studies the adversarial robustness of VideoLLMs under observation-level attacks, which operate on the observation pipeline through omission and suppression rather than through bounded input perturbations. Because practical VideoLLMs rely on frame sampling and token budgeting, their predictions are conditioned on partial observation, which enables observation-level attacks that either keep harmful evidence out of inference or exploit its being discarded within it. We present DEFTEVAL, a controlled framework for systematically evaluating whether input-level defenses transfer to this threat model under fixed sampling configurations. Across five VideoLLMs, five observation-level attacks, and eleven representative input-level defenses spanning five families, we find that input-level defenses provide limited and inconsistent mitigation. Critically, defenses fail even when human-visible harmful signals persist in every sampled frame. Defense effectiveness is dominated by model architecture rather than by the defense method, and detection rates vary drastically across content categories, exposing weaknesses in the temporal reasoning of VideoLLMs. These results underscore the need to move from input-level robustness toward system-level robustness, spanning sampling-aware coverage guarantees, tokenlevel preservation of safety-relevant features, and modalitybalanced fusion, for securing VideoLLMs in safety-critical deployments.

## REFERENCES

[1] Y. Cao, W. Song, D. Wang, J. Xue, and J. S. Dong, “Failures to surface harmful contents in video large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

[2] Y. Cao, W. Song, J. Xue, and J. S. Dong, “Poisoning prompt-guided sampling in video large language models,” arXiv preprint arXiv:2509.20851, 2025.

[3] W. Song, C. Cong, H. Zhong, and J. Xue, “Correction-based defense against adversarial video attacks via {Discretization-Enhanced} video compressive sensing,” in 33rd USENIX Security Symposium (USENIX Security 24), 2024, pp. 3603–3620.

[4] J. Min, S. Buch, A. Nagrani, M. Cho, and C. Schmid, “Morevqa: Exploring modular reasoning models for video question answering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 235–13 245.

[5] J. Xiao, A. Yao, Y. Li, and T.-S. Chua, “Can i trust your answer? visually grounded video question answering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 204–13 214.

[6] M. Narasimhan, A. Rohrbach, and T. Darrell, “Clip-it! language-guided video summarization,” Advances in neural information processing systems, vol. 34, pp. 13 988–14 000, 2021.

[7] T. Alaa, A. Mongy, A. Bakr, M. Diab, and W. Gomaa, “Video summarization techniques: A comprehensive review,” arXiv preprint arXiv:2410.04449, 2024.

[8] Y. Li, C. Wang, and J. Jia, “Llama-vid: An image is worth 2 tokens in large language models,” in European Conference on Computer Vision. Springer, 2024, pp. 323–340.

[9] Z. Zhang, Z. Sun, Z. Zhang, Z. Peng, Y. Zhao, Z. Wang, Z. Luo, R. Zuo, and X. He, “” i can see forever!”: Evaluating real-time videollms for assisting individuals with visual impairments,” arXiv preprint arXiv:2505.04488, 2025.

[10] J. Shi, X. Shen, K. Zhao, X. Wang, V. Wen, Z. Wang, Y. Wu, and Z. Zhang, “Cpfd: Confidence-aware privileged feature distillation for short video classification,” in Proceedings ofthe 33rd ACM International Conference on Information and Knowledge Management, 2024, pp. 4866–4873.

[11] X. Tang, J. Qiu, L. Xie, Y. Tian, J. Jiao, and Q. Ye, “Adaptive keyframe sampling for long video understanding,” in Proceedings ofthe Computer Vision and Pattern Recognition Conference, 2025, pp. 29 118–29 128.

[12] Z. Cheng, S. Leng, H. Zhang, Y. Xin, X. Li, G. Chen, Y. Zhu, W. Zhang, Z. Luo, D. Zhao et al., “Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms,” arXiv preprint arXiv:2406.07476, 2024.

[13] Y. Zhao, X. Zheng, L. Luo, Y. Li, X. Ma, and Y.-G. Jiang, “Bluesuffix: Reinforced blue teaming for vision-language models against jailbreak attacks,” arXiv preprint arXiv:2410.20971, 2024.

[14] D.-A. Huang, S. Radhakrishnan, Z. Yu, and J. Kautz, “Frag: Frame selection augmented generation for long video and long document understanding,” arXiv preprint arXiv:2504.17447, 2025.

[15] M. Naseer, S. Khan, and F. Porikli, “Local gradients smoothing: Defense against localized adversarial attacks,” in 2019 IEEE winter conference on applications of computer vision (WACV). IEEE, 2019, pp. 1300– 1307.

[16] X. Jia, X. Wei, X. Cao, and H. Foroosh, “Comdefend: An efficient image compression model to defend adversarial examples,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 6084–6092.

[17] L. Jing, R. Wang, W. Ren, X. Dong, and C. Zou, “Pad: Patchagnostic defense against adversarial patch attacks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 472–24 481.

[18] W. Nie, B. Guo, Y. Huang, C. Xiao, A. Vahdat, and A. Anandkumar, “Diffusion models for adversarial purification,” arXiv preprint arXiv:2205.07460, 2022.

[19] A. Prakash, N. Moran, S. Garber, A. DiLillo, and J. Storer, “Semantic perceptual image compression using deep convolution networks,” in 2017 Data Compression Conference (DCC). IEEE, 2017, pp. 250– 259.

[20] ——, “Deflecting adversarial attacks with pixel deflection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 8571–8580.

[21] A. A. Efros and W. T. Freeman, “Image quilting for texture synthesis and transfer,” in Seminal Graphics Papers: Pushing the Boundaries, Volume 2, 2023, pp. 571–576.

[22] C. Xie, J. Wang, Z. Zhang, Z. Ren, and A. Yuille, “Mitigating adversarial effects through randomization,” arXiv preprint arXiv:1711.01991, 2017.

[23] A. Mustafa, S. H. Khan, M. Hayat, J. Shen, and L. Shao, “Image superresolution as a defense against adversarial attacks,” IEEE Transactions on Image Processing, vol. 29, pp. 1711–1724, 2019.

[24] C. Guo, M. Rana, M. Cisse, and L. Van Der Maaten, “Countering adversarial images using input transformations,” arXiv preprint arXiv:1711.00117, 2017.

[25] K. Jiang, Z. Chen, J. Fu, L. Hong, J. Li, and W. Zhang, “Videopure: Diffusion-based adversarial purification for video recognition,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 11, pp. 11 474–11 487, 2025.

[26] Z. Liu, L. Zhu, B. Shi, Z. Zhang, Y. Lou, S. Yang, H. Xi, S. Cao, Y. Gu, D. Li et al., “Nvila: Efficient frontier visual language models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 4122–4134.

[27] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu et al., “Expanding performance boundaries of opensource multimodal models with model, data, and test-time scaling,” arXiv preprint arXiv:2412.05271, 2024.

[28] P. Jin, R. Takanobu, W. Zhang, X. Cao, and L. Yuan, “Chat-univi: Unified visual representation empowers large language models with image and video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13 700–13 710.

[29] K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, “Videochat: Chat-centric video understanding,” arXiv preprint arXiv:2305.06355, 2023.

[30] B. Zhang, K. Li, Z. Cheng, Z. Hu, Y. Yuan, G. Chen, S. Leng, Y. Jiang, H. Zhang, X. Li et al., “Videollama 3: Frontier multimodal foundation models for image and video understanding,” arXiv preprint arXiv:2501.13106, 2025.

[31] B. Lin, Y. Ye, B. Zhu, J. Cui, M. Ning, P. Jin, and L. Yuan, “Video-llava: Learning united visual representation by alignment before projection,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 5971–5984.

[32] A. Yang, A. Nagrani, P. H. Seo, A. Miech, J. Pont-Tuset, I. Laptev, J. Sivic, and C. Schmid, “Vid2seq: Large-scale pretraining of a visual language model for dense video captioning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 10 714–10 726.

[33] L. Chen, X. Wei, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, Z. Tang, L. Yuan et al., “Sharegpt4video: Improving video understanding and generation with better captions,” Advances in Neural Information Processing Systems, vol. 37, pp. 19 472–19 495, 2024.

[34] C. Zhang, T. Lu, M. M. Islam, Z. Wang, S. Yu, M. Bansal, and G. Bertasius, “A simple llm framework for long-range video questionanswering,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 21 715–21 737.

[35] H. Wang, Z. Xu, Y. Cheng, S. Diao, Y. Zhou, Y. Cao, Q. Wang, W. Ge, and L. Huang, “Grounded-videollm: Sharpening fine-grained temporal grounding in video large language models,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing Findings, 2025.

[36] R. Qian, X. Dong, P. Zhang, Y. Zang, S. Ding, D. Lin, and J. Wang, “Streaming long video understanding with large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 119 336– 119 360, 2024.

[37] S. Yu, J. Cho, P. Yadav, and M. Bansal, “Self-chained image-language model for video localization and question answering,” Advances in Neural Information Processing Systems, vol. 36, pp. 76 749–76 771, 2023.

[38] Y. Weng, M. Han, H. He, X. Chang, and B. Zhuang, “Longvlm: Efficient long video understanding via large language models,” in European Conference on Computer Vision. Springer, 2024, pp. 453–470.

[39] Z. Cheng, R. Wang, and Z. Wang, “Focuschat: Text-guided long video understanding via spatiotemporal information filtering,” arXiv preprint arXiv:2412.12833, 2024.

[40] L. Wang, Y. Chen, D. Tran, V. N. Boddeti, and W.-S. Chu, “Seal: Semantic attention learning for long video representation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 192–26 201.

[41] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu et al., “Llava-onevision: Easy visual task transfer,” arXiv preprint arXiv:2408.03326, 2024.

[42] Y. Zhang, B. Li, h. Liu, Y. j. Lee, L. Gui, D. Fu, J. Feng, Z. Liu, and C. Li, “Llava-next: A strong zero-shot video understanding model,” April 2024. [Online]. Available: https://llava-vl.github.io/blog/ 2024-04-30-llava-next-video/

[43] O. Zohar, X. Wang, Y. Dubois, N. Mehta, T. Xiao, P. Hansen-Estruch, L. Yu, X. Wang, F. Juefei-Xu, N. Zhang et al., “Apollo: An exploration of video understanding in large multimodal models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 18 891–18 901.

[44] K. Hu, F. Gao, X. Nie, P. Zhou, S. Tran, T. Neiman, L. Wang, M. Shah, R. Hamid, B. Yin et al., “M-llm based video frame selection for efficient video understanding,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 13 702–13 712.

[45] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[46] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[47] C. Cheng, J. Guan, W. Wu, and R. Yan, “Scaling video-language models to 10k frames via hierarchical differential distillation,” in Proceedings of the Forty-second International Conference on Machine Learning, 2025.

[48] C. Schlarmann, N. D. Singh, F. Croce, and M. Hein, “Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models,” arXiv preprint arXiv:2402.12336, 2024.

[49] M. Z. Hossain and A. Imteaj, “Securing vision-language models with a robust encoder against jailbreak and adversarial attacks,” in 2024 IEEE International Conference on Big Data (BigData). IEEE, 2024, pp. 6250–6259.

[50] Y. Xu, X. Qi, Z. Qin, and W. Wang, “Cross-modality information check for detecting jailbreaking in multimodal large language models,” in Findings of the Association for Computational Linguistics: EMNLP 2024, 2024, pp. 13 715–13 726.

[51] T. B. Brown, D. Mane, A. Roy, M. Abadi, and J. Gilmer, “Adversarial´ patch,” arXiv preprint arXiv:1712.09665, 2017.

[52] D. Karmon, D. Zoran, and Y. Goldberg, “Lavan: Localized and visible adversarial noise,” in International conference on machine learning. PMLR, 2018, pp. 2507–2515.

[53] C. Xiang, S. Mahloujifar, and P. Mittal, “{PatchCleanser}: Certifiably robust defense against adversarial patches for any image classifier,”

in 31st USENIX security symposium (USENIX Security 22), 2022, pp. 2065–2082.

[54] A. Mehrotra, D. Peng, D. Bhusal, and N. Rastogi, “Concept-based masking: A patch-agnostic defense against adversarial patch attacks,” arXiv preprint arXiv:2510.04245, 2025.

[55] J. Liu, A. Levine, C. P. Lau, R. Chellappa, and S. Feizi, “Segment and complete: Defending object detectors against adversarial patch attacks with robust patch detection,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 14 973–14 982.

[56] Y. Zhang, J. Wu, W. Li, B. Li, Z. Ma, Z. Liu, and C. Li, “Llavavideo: Video instruction tuning with synthetic data,” arXiv preprint arXiv:2410.02713, 2024.

[57] Z. Teed and J. Deng, “Raft: Recurrent all-pairs field transforms for optical flow,” in European conference on computer vision. Springer, 2020, pp. 402–419.

[58] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 975–11 986.

[59] X. Jin, Y. Wang, and X. Tan, “Pornographic image recognition via weighted multiple instance learning,” IEEE transactions on cybernetics, vol. 49, no. 12, pp. 4412–4420, 2018.

[60] P. Wu, J. Liu, Y. Shi, Y. Sun, F. Shao, Z. Wu, and Z. Yang, “Not only look, but also listen: Learning multimodal violence detection under weak supervision,” in European conference on computer vision. Springer, 2020, pp. 322–339.