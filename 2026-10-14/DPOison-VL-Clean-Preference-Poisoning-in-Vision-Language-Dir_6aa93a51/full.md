# DPOison-VL: Clean Preference Poisoning in Vision-Language Direct Preference Optimization

Chengtao Jian<sup>1</sup>, Kai Yang<sup>1,2,3,†</sup>

<sup>1</sup>School of Computer Science and Technology, Tongji University, China <sup>2</sup>Shenzhen Loop Area Institute, China

<sup>3</sup>Shanghai Research Institute for Intelligent Autonomous Systems, China {jct, kaiyang}@tongji.edu.cn <sup>†</sup>Corresponding author.

## Abstract

Direct Preference Optimization (DPO) has be come a widely adopted approach for aligning vision-language models (VLMs), yet the security of multimodal preference data remains largely unexplored. Existing preference poisoning attacks either corrupt explicit supervision signals or require visible triggers, making them detectable under human inspection. We intro duce DPOison-VL, the first clean preference poisoning attack against VLM-DPO, which manipulates model alignment solely through imperceptible perturbations on a small fraction of training images, while leaving all prompts, responses, and preference annotations semantically intact. We formulate poison construction as a bilevel optimization and derive a gradient alignment surrogate that efficiently steers im age perturbations toward the attacker-specified preference. We further introduce a preference displacement score for principled poison selection with provable first-order optimality under a fixed poison budget, and derive a sufficient condition for preference inversion. Extensive experiments across diverse attack tasks, different VLM architectures and scales, and baseline comparisons demonstrate that DPOison-VL achieves up to 98% ASR with only 2% poisoned data and exhibits cross-model transferability, while largely preserving clean accuracy and general utility. Our findings reveal a critical vulnerability in VLM-DPO, where imperceptible image perturbations in preference training data can silently redirect alignment.

## 1 Introduction

Vision-language models (VLMs) have achieved remarkable capabilities in multimodal reasoning, visual question answering, and image-grounded dialogue (Liu et al., 2023; Bai et al., 2025; Wang et al., 2025a; Hurst et al., 2024). Despite these advances, VLMs remain susceptible to well-documented failure modes, including hallucination (Guo et al.,

2026; Liu et al., 2024a), harmful content generation (Zhang et al., 2025), and poor calibration of helpfulness (Yu et al., 2025; Liu et al., 2024c). To address these failures, Direct Preference Optimization (DPO) (Rafailov et al., 2023) has emerged as a widely adopted post-training paradigm for language model alignment, and has been recently extended to VLMs to mitigate hallucinations (Yang et al., 2025) and improve trustworthiness (Yu et al., 2025; Zhang et al., 2025).

However, the effectiveness of DPO also makes preference data a critical attack surface. Prior poisoning attacks on VLMs have focused on supervised multimodal training, either implanting backdoors triggered at test time (Lyu et al., 2024; Yuan et al., 2025; Guo et al., 2026; Jiao et al., 2025) or injecting stealthy image-text pairs (Xu et al., 2024), leaving the preference alignment stage unexplored. Recent work on preference learning in LLMs manipulates alignment through label flipping (Kusaka et al., 2026; Yang et al., 2026; Pathmanathan et al., 2025), ranking corruption (Wang et al., 2024), or poisoned preference pairs (Baumgärtner et al., 2024), but all corrupt explicit supervision signals and remain detectable under human inspection. Table 1 summarizes these two lines of work along five key properties, and none of them jointly targets the VLM-DPO stage, poisons through images alone, and keeps preference annotations clean. This raises an underexplored question in VLM-DPO:

## Can an attacker manipulate VLM-DPO using semantically valid preference pairs?

In this work, we propose DPOISON-VL, a clean preference poisoning attack against VLM-DPO that jointly satisfies all five properties in Table 1. To the best of our knowledge, DPOISON-VL is the first attack to demonstrate that an adversary can manipulate VLM preference alignment by adding imperceptible perturbations to images, while keeping all preference annotations clean. Despite the image perturbation, the poisoned example remains a semantically valid multimodal preference pair, making it difficult for human annotators to detect the poisoning. In summary, our contributions are as follows:

<table><tr><td>Method</td><td colspan="5">Targeting VLMs Preference Learning Image-only Poison Clean Preference Clean Input</td></tr><tr><td>TrojVLM (Lyu et al., 2024)</td><td></td><td>X</td><td>X</td><td>N/A</td><td>X</td></tr><tr><td>Shadowcast (Xu et al., 2024)</td><td></td><td>X</td><td>X</td><td>N/A</td><td>V</td></tr><tr><td>BadToken (Yuan et al., 2025)</td><td></td><td>X</td><td>X</td><td>N/A</td><td>X</td></tr><tr><td>CBV (Guo et al., 2026)</td><td></td><td>X</td><td>√</td><td>N/A</td><td>x</td></tr><tr><td>RankPoison (Wang et al., 2024)</td><td>X</td><td>√</td><td>X</td><td>X</td><td></td></tr><tr><td>PoisonDPO (Pathmanathan et al., 2025)</td><td>X</td><td>7</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Preference-Flip (Kusaka et al., 2026)</td><td>X</td><td>V</td><td>X</td><td>X</td><td>X</td></tr><tr><td>DPOISON-VL</td><td>J</td><td>V</td><td>√</td><td>J</td><td></td></tr></table>

Table 1: Comparison of poisoning attacks. ✓ marks satisfied properties. N/A marks inapplicable stages.

• We introduce a new poisoning threat against VLM-DPO. To the best of our knowledge, DPOISON-VL is the first clean preference poisoning attack on VLMs. Unlike existing attacks that corrupt supervision signals or require explicit triggers, DPOISON-VL shows that imperceptible image perturbations alone can redirect DPO alignment toward attacker-specified behaviors.

• We develop a principled poison crafting pipeline. We formulate poison construction as a bilevel optimization and derive a gradient alignment surrogate for crafting poisoned images. We further introduce the preference displacement score for principled poison selection, with provable optimality under a firstorder approximation and fixed poison budget.

• We provide theoretical analysis and extensive empirical validation. We derive a sufficient condition for preference inversion under gradient descent, connecting gradient alignment to poison quality and poison ratio. Extensive experiments across diverse attack tasks and VLMs demonstrate that DPOISON-VL achieves high ASR with small poison budgets, exhibits cross-model transferability, and preserves clean accuracy and general utility.

## 2 Related Work

Preference Alignment for VLMs. Preference alignment has become an important post-training paradigm for improving the helpfulness, factuality, and safety of VLMs. DPO (Rafailov et al., 2023)

simplifies RLHF by directly optimizing over preference data, making it attractive for multimodal alignment. Recent VLM studies have adopted preference optimization to mitigate hallucinations, strengthen visual grounding, and improve safety and trustworthiness (Xie et al., 2024; Yang et al., 2025; Yu et al., 2025; Zhang et al., 2025, 2026). These works demonstrate the effectiveness of preference data for VLM alignment but generally assume their trustworthiness. We show that a small number of clean preference pairs with imperceptibly perturbed images can compromise VLM-DPO.

Poisoning Attacks on Preference Alignment. Recent studies have shown that preference alignment can be vulnerable when poisoned supervision is incorporated into training. Existing attacks manipulate the alignment pipeline through injecting poisoned preference pairs (Baumgärtner et al., 2024; Pathmanathan et al., 2025), corrupting reward or ranking feedback (Wang et al., 2024), flipping preference labels (Kusaka et al., 2026; Yang et al., 2026), or implanting backdoors through poisoned human feedback (Rando and Tramèr, 2024). All these attacks corrupt explicit supervision signals and target LLMs rather than VLMs, leaving the multimodal preference alignment unexplored.

Clean-Label Poisoning and Backdoor Attacks on VLMs. Clean-label poisoning has been widely studied in machine learning and computer vision, where attackers preserve label correctness while inducing targeted model behavior (Biggio et al., 2012; Muñoz-González et al., 2017; Huang et al., 2020; Geiping et al., 2021; Souri et al., 2022). Recent work has extended such threats to VLMs. Shadowcast (Xu et al., 2024) injects visually indistinguishable image-text pairs to manipulate VLM responses after visual instruction tuning. TrojVLM (Lyu et al., 2024) and BadToken (Yuan et al., 2025) investigate backdoor attacks against VLMs, while CBV (Guo et al., 2026) introduces clean-label VLM backdoors using diffusion models to generate triggered images. However, these methods do not investigate poisoning at the multimodal preference alignment stage. In contrast, DPOISON-VL targets VLM-DPO through imperceptible image perturbations alone, while preserving semantically valid preference pairs and inducing attackerspecified responses on clean target images.

## 3 Methods

Notation. We use $\pmb { I } \in \mathcal { I }$ for an image, $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ for a prompt, and $\pmb { y } \in \mathcal { V }$ for a response. A preference example is ${ \pmb z } = ( { \pmb I } , { \pmb x } , { \pmb y } ^ { + } , { \pmb y } ^ { - } )$ , where $y ^ { + }$ and $\mathbf { \nabla } _ { \pmb { y } ^ { - } }$ are the chosen and rejected responses. The VLM policy $\pi _ { \pmb { \theta } }$ is initialized at $\pmb { \theta } _ { 0 }$ , with reference policy $\pi _ { \mathrm { r e f } } .$ . For each task, we use four disjoint image splits. $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { c r a f t } }$ provides base images for constructing clean preference candidates, while $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } }$ provides target examples for defining the attack objective. The held-out splits $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { t e s t } }$ and $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { t e s t } }$ evaluate clean behavior and attack effectiveness, respectively.

## 3.1 Threat Model

In this subsection, we describe the threat model of our study, including the attacker surface, attacker’s goal, capabilities, knowledge and constraints.

Attack Surface. Motivated by prior studies of VLM data poisoning and DPO preference poisoning (Xu et al., 2024; Pathmanathan et al., 2025), we study training-time data poisoning against VLM-DPO, targeting the preference data collection or curation stage. The attacker contributes a small number of poisoned preference pairs to the victim’s DPO training pipeline, with adversarial modifications restricted to imperceptible image perturbations. The attack requires no inference-time triggers or perturbed inputs at inference time.

Attacker’s Goal. The attacker aims to steer the VLM, after poisoned DPO training, to generate an attacker-specified response $y ^ { \mathrm { a d v } }$ instead of the correct response $y ^ { \mathrm { c o r r } }$ on clean target images, while preserving correct responses on benign base images. For example, in Task I, the attacker seeks to make the model misidentify Donald Trump as Joe Biden while still correctly identifying Biden images. Attack success and clean accuracy are evaluated by free-form generation on the held-out $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { t e s t } }$ and $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { t e s t } }$ , respectively.

Attacker’s Capabilities and Knowledge. We consider DPO post-training of an open-weight

VLM using multimodal preference data collected from multiple sources (Li et al., 2024). When such training data include externally contributed samples, they may provide an entry point for data poisoning attacks (Wan et al., 2023). In this setting, the attacker can contribute multimodal preference pairs to the DPO training pipeline. As in prior clean-label poisoning (Shafahi et al., 2018), we consider white-box access to the initial model. Since the initial VLM weights are publicly available, the attacker can obtain the same checkpoint $\pmb { \theta } _ { 0 }$ and compute gradients for poison construction. The attacker can use a target crafting set, but cannot modify the original training data $\mathcal { D } _ { \mathrm { t r a i n } }$ or perturb images at inference time.

Clean Preference and Stealth Constraint. The attacker can not construct poisoned samples that are obviously misaligned with human preference. Following the clean-label poisoning principle (Shafahi et al., 2018), each contributed preference pair must remain semantically valid under human inspection: the preferred response is genuinely better than the dispreferred one given the perturbed image, and all preference annotations remain correct. The adversarial signal is embedded in imperceptible image perturbations, making poisoned samples difficult to detect through human inspection.

## 3.2 Attack Objective

The attack variable in DPOISON-VL is a set of imperceptible image perturbations $\Delta = \{ \delta _ { j } \} _ { j = 1 } ^ { P } { } _  : \quad$ each subject to $\| \delta _ { j } \| _ { \infty } \leq \epsilon .$ The attacker contributes a poisoned preference set $\mathcal { P } _ { \Delta } = \{ ( \tilde { \pmb { I } } _ { j } +$ $\delta _ { j } , \tilde { { \pmb x } } _ { j } , \tilde { { \pmb y } } _ { j } ^ { + } , \tilde { { \pmb y } } _ { j } ^ { - } ) \} _ { j = 1 } ^ { P }$ , where all prompts, responses, and preference annotations remain intact, and the victim trains on $\mathcal { D } _ { \mathrm { t r a i n } } \cup \mathcal { P } _ { \Delta }$ via standard DPO.

To quantify whether the trained model is steered toward the attacker-specified behavior, we define the preference margin on $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } }$ as

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { p r e f } } ( \pmb { \theta } ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } ) = \frac { 1 } { M } \sum _ { t = 1 } ^ { M } \left[ s _ { \pmb { \theta } } ^ { \mathrm { c o r r } } ( z _ { t } ) - s _ { \pmb { \theta } } ^ { \mathrm { a d v } } ( z _ { t } ) \right] , } \end{array}\tag{1}
$$

where $s _ { \theta } ^ { \mathrm { c o r r } }$ and $s _ { \theta } ^ { \mathrm { a d v } }$ are length-normalized logprobability scores (Meng et al., 2024), with normalization applied to remove length bias (Meng et al., 2024). A negative margin indicates preference displacement toward $y ^ { \mathrm { a d v } }$

For a preference example $z = ( I , x , y ^ { + } , y ^ { - } )$ define the DPO log-ratio margin

$$
\begin{array} { r } { m ( z ; \pmb { \theta } ) = \log \frac { \pi _ { \pmb { \theta } } ( y ^ { + } | \pmb { I } , \pmb { x } ) } { \pi _ { \mathrm { r e f } } ( y ^ { + } | \pmb { I } , \pmb { x } ) } - \log \frac { \pi _ { \pmb { \theta } } ( y ^ { - } | \pmb { I } , \pmb { x } ) } { \pi _ { \mathrm { r e f } } ( y ^ { - } | \pmb { I } , \pmb { x } ) } . } \end{array}
$$

Then the standard DPO loss is ${ \mathcal { L } } _ { \mathrm { D P O } } ( z ; \theta ) ~ =$ − log $\sigma ( \beta m ( z ; \pmb { \theta } ) )$ , where $\sigma$ is the sigmoid function and $\beta$ is a scaling factor. We cast DPOison-VL as a bilevel optimization problem:

$$
\begin{array} { c l } { \displaystyle \operatorname* { m i n } _ { \Delta } } & { \displaystyle \mathcal { T } _ { \mathrm { p r e f } } \left( \pmb { \theta } ^ { * } ( \Delta ) ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } \right) } \\ { \mathrm { s . t . } } & { \displaystyle \pmb { \theta } ^ { * } ( \Delta ) \in \arg \displaystyle \operatorname* { m i n } _ { \pmb { \theta } } \sum _ { z \in \mathcal { D } _ { \mathrm { t r a i n } } \cup \mathcal { P } _ { \Delta } } \mathcal { L } _ { \mathrm { D P O } } ( z ; \pmb { \theta } ) , } \\ & { \displaystyle \| \pmb { \delta } _ { j } \| _ { \infty } \leq \epsilon , \quad j = 1 , \dots , P . } \end{array}\tag{2}
$$

The lower level is the victim’s standard DPO finetuning, which the attacker cannot interfere with. The upper level minimizes the preference margin over unperturbed target images, driving the trained model to prefer ${ \mathbf { } } y _ { t } ^ { \mathrm { a d v } }$ over ${ \pmb y } _ { t } ^ { \mathrm { c o r r } }$ . Unlike prior classification attacks (Geiping et al., 2021; Huang et al., 2020; Biggio et al., 2012; Muñoz-González et al., 2017), DPOISON-VL targets a distinct vulnerability in preference alignment: the lower level optimizes semantically valid preference pairs, while the upper level seeks to reverse the preference.

## 3.3 Poison Crafting

The upper-level objective $\mathcal { T } _ { \mathrm { p r e f } } ( \pmb { \theta } ^ { * } ( \Delta ) ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } )$ depends on $\Delta$ only through the trained parameters $\theta ^ { * } ( \Delta )$ , which are implicitly defined by the inner DPO minimization. Differentiating through the full DPO training trajectory incurs substantial computational and memory costs for modern VLMs (Sinha et al., 2017; Shaban et al., 2019). To make poison crafting tractable, we approximate the victim’s update with a single DPO gradient step from $\pmb { \theta } _ { 0 }$ :

$$
\pmb { \theta } ^ { \prime } ( \Delta ) = \pmb { \theta } _ { 0 } - \eta \nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { D P O } } ( \mathcal { B } _ { \mathrm { c } } \cup \mathcal { B } _ { p } ; \pmb { \theta } _ { 0 } ) ,
$$

where $\eta$ is the learning rate, and $\boldsymbol { B _ { \mathrm { c } } }$ and $B _ { p }$ are the clean and poisoned minibatches, respectively. This approximation enables us to relate the effect of poisoned preference pairs on the target preference margin to their DPO gradients, motivating an efficient gradient-alignment objective.

For simplicity, let $\pmb { \mathscr { g } } _ { t } = \nabla _ { \pmb { \theta } } \mathcal { T } _ { \mathrm { p r e f } } ( \pmb { \theta } _ { 0 } ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } )$ $\pmb { g } _ { \mathrm { c } } = \nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { D P O } } ( \mathcal { B } _ { \mathrm { c } } ; \pmb { \theta } _ { 0 } ) , \pmb { g } _ { p } = \nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { D P O } } ( \mathcal { B } _ { p } ; \pmb { \theta } _ { 0 } )$ denote the gradients of the preference margin, clean DPO and poison DPO, respectively. A first-order Taylor expansion of $\tau _ { \mathrm { { p r e f } } }$ around $\pmb { \theta } _ { 0 }$ gives

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { p r e f } } \big ( \pmb { \theta } ^ { \prime } ( \Delta ) \big ) \approx \mathcal { T } _ { \mathrm { p r e f } } ( \pmb { \theta } _ { 0 } ) - \eta \langle \pmb { g } _ { t } , \pmb { g } _ { \mathrm { c } } + \pmb { g } _ { p } \rangle . } \end{array}
$$

Since $\mathcal { T } _ { \mathrm { p r e f } } ( \pmb { \theta } _ { 0 } )$ and $\scriptstyle { \pmb { g } } _ { \mathrm { c } }$ are independent of $\Delta$ , minimizing the first-order approximation amounts to maximizing $\langle \pmb { g } _ { t } , \pmb { g } _ { p } \rangle$ Following prior gradientmatching poisoning methods (Geiping et al., 2021;

Algorithm 1 Poison Crafting in DPOison-VL   
1: Input: Candidate pool C, target crafting set   
$\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } }$ , VLM θ, perturbation budget ϵ, attack   
step size $\alpha ,$ iterations $T$   
2: Initialize $\delta _ { j } ^ { ( 0 ) } \ \sim$ Uniform(−ϵ, ϵ) for $j ~ =$   
$1 , \ldots , P$   
3: Set $\tilde { \pmb { I } } _ { j } ^ { p , ( 0 ) } = \Pi _ { [ 0 , 1 ] } ( \tilde { \pmb { I } } _ { j } + \pmb { \delta } _ { j } ^ { ( 0 ) } )$   
4: for $\dot { k } = 0 , \dots , \overset { \cdot } { T } - 1$ do   
5: Sample minibatch $B _ { t }$ and $B _ { p } ( \Delta )$   
6: Compute gradients $\pmb { g } _ { t }$ and $\scriptstyle { \pmb { g } } _ { p }$   
7: Compute attack loss $\mathcal { L } _ { \mathrm { a t t a c k } } \dot { ( } \Delta ^ { ( k ) } )$   
8: Update perturbations and poisoned images   
via Eqs. (3) and (4)   
9: end for   
10: Output: Crafted poisoned preference pairs   
$\{ ( \tilde { \pmb { I } } _ { j } ^ { p , ( T ) } , \tilde { \pmb { x } } _ { j } , \tilde { \pmb { y } } _ { j } ^ { + } , \tilde { \pmb { y } } _ { j } ^ { - } ) \} _ { j = 1 } ^ { P }$

Souri et al., 2022; Liu et al., 2024b), we align these gradients by maximizing their cosine similarity $\begin{array} { r } { \Gamma _ { \mathrm { a l i g n } } ( \Delta ) = \frac { \langle \pmb { g } _ { t } , \pmb { g } _ { p } \rangle } { \| \pmb { g } _ { t } \| _ { 2 } \| \pmb { g } _ { p } \| _ { 2 } } } \end{array}$ . Our analysis connects gradient alignment to the preference margin, providing a principled basis for crafting poisoned images without modifying preference annotations.

To further enhance stealth, we add an $\ell _ { 2 }$ regularization term to the upper-level objective

$$
\mathcal { L } _ { \mathrm { a t t a c k } } ( \Delta ) = - \Gamma _ { \mathrm { a l i g n } } ( \Delta ) + \lambda _ { \mathrm { r e g } } \sum _ { j = 1 } ^ { P } \| \pmb { \delta } _ { j } \| _ { 2 } ^ { 2 } .
$$

Differentiating with respect to $\delta _ { j }$ yields a hypergradient through the cross-derivative $\nabla _ { \delta _ { j } } \nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { D P O } } ( B _ { p } ( \Delta ) ; \pmb { \theta } _ { 0 } )$ , which characterizes how image perturbations affect the poison DPO gradient. Then each perturbation is updated using projected gradient descent (Madry et al., 2018):

$$
\begin{array} { r } { \delta _ { j } ^ { ( k + 1 ) } = \Pi _ { \parallel \delta \parallel _ { \infty } \leq \epsilon } [ \delta _ { j } ^ { ( k ) } - \alpha \mathrm { s i g n } ( \nabla _ { \delta _ { j } } \mathcal { L } _ { \mathrm { a t t a c k } } ) ] , } \end{array}\tag{3}
$$

with the image clipped to the valid pixel range:

$$
\tilde { \pmb { I } } _ { j } ^ { p , ( k + 1 ) } = \Pi _ { [ 0 , 1 ] } ( \tilde { \pmb { I } } _ { j } + \pmb { \delta } _ { j } ^ { ( k + 1 ) } ) .\tag{4}
$$

The full procedure is summarized in Algorithm 1.

## 3.4 Principled Poison Selection

Let $\mathcal { C } _ { p }$ denote the crafted poison candidate pool obtained from C. To select most influential samples, motivated by (Xia et al., 2024; Pruthi et al., 2020; Koh and Liang, 2017) we analyze the contribution of each candidate through its DPO gradient. For a poisoned example ${ \pmb z } = ( { \pmb I } , { \pmb x } , { \pmb y } ^ { + } , { \pmb y } ^ { - } )$ , we define the log-likelihood gradient difference as

![](images/10ecffac25809b702a00156941a904f8d6d197408eb3ec814b5b73db295e69d3.jpg)  
Figure 1: Overview of DPOISON-VL. The pipeline consists of three stages: (S1) on-policy response generation, (S2) poison crafting, and (S3) poison selection, collectively steering DPO training toward target preference inversion.

$$
b _ { z } = \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { + } | I , x ) - \nabla _ { \theta } \log \pi _ { \theta } ( y ^ { - } | I , x ) .
$$

Let $\lambda _ { z } = \beta \sigma ( - \beta m _ { z } )$ . Then the DPO gradient satisfies $\nabla _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { D P O } } ( \pmb { z } ; \pmb { \theta } ) = - \lambda _ { z } \pmb { b } _ { z } .$ A DPO gradient step updates the model along $\lambda _ { z } b _ { z }$ . Its first-order effect on the preference margin is determined by the alignment with the preference margin gradient, which motivates the following score.

Definition 1 (Preference Displacement Score). The preference displacement score of z is defined as

$$
\begin{array} { r } { \psi ( z ) = - \lambda _ { z } \langle \nabla _ { \pmb { \theta } } \mathcal { T } _ { \mathrm { p r e f } } ( \pmb { \theta } ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } ) , \pmb { b } _ { z } \rangle . } \end{array}\tag{5}
$$

A larger score indicates a greater first-order reduction in the preference margin under a DPO update. At the initialization $\pmb { \theta } _ { 0 }$ , the reference policy satisfies $\pi _ { \mathrm { r e f } } = \pi _ { \theta _ { 0 } }$ , yielding $m ( z ; \pmb { \theta } _ { 0 } ) = 0$ and $\lambda _ { z } ~ = ~ \beta / 2$ for all candidates. Thus, the top-K ranking is determined by the gradient alignment in Eq. (5), while $\lambda _ { z }$ can become sample-dependent as DPO training proceeds.

For a selected poison set ${ \mathcal { S } } \subseteq { \mathcal { C } } _ { p }$ with $| S | = K$ let $\rho \ = \ { \frac { K } { N } }$ and $\begin{array} { r } { \omega _ { \rho } = \frac { \rho } { 1 + \rho } } \end{array}$ . Under a fixed poison budget, the clean training gradient is identical across different poison selections. Therefore, we analyze the effect of poison selection on the preference margin through the following update:

$$
\begin{array} { r } { \widetilde { \pmb { \theta } } _ { S } = \pmb { \theta } + \eta \omega _ { \rho } \frac { 1 } { K } \sum _ { z \in S } \lambda _ { z } \pmb { b } _ { z } . } \end{array}\tag{6}
$$

We next establish how the preference displacement score governs change in the preference margin and provides a theoretical basis for poison selection.

Assumption 1 (Smoothness and Boundedness). Following Jiao et al. (2023); Ji et al. (2021), we assume that the function T is $L _ { T } .$ -smooth, $A _ { \Delta }$ is L<sub>A</sub>-Lipschitz, and $\| \nabla \mathcal { F } _ { \Delta } ( \pmb { \theta } _ { h } ) \| _ { 2 } \ \le \ G$ for h = $0 , \ldots , H - 1$

Theorem 1 (Preference Margin Dynamics). Suppose $\mathcal { T } _ { \mathrm { p r e f } } ( : ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } )$ is L<sub>T</sub>-smooth. For every ${ \mathcal { S } } \subseteq { \mathcal { C } } _ { p } ,$ , let $\mathcal { T } _ { \theta } = \mathcal { T } _ { \mathrm { p r e f } } ( \theta ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } )$ . Then we have

$$
\begin{array} { r } { \mathcal { T } _ { \widetilde { \pmb { \theta } } _ { S } } - \mathcal { T } _ { \pmb { \theta } } \leq - \frac { \eta \omega _ { \rho } } { K } \sum _ { z \in S } \psi ( z ) + \frac { L _ { \mathcal { T } } } { 2 } \| \widetilde { \pmb { \theta } } _ { S } - \pmb { \theta } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Corollary 1 (Fixed-Budget Poison Selection). Let $\begin{array} { r } { C _ { \mathrm { c l e a n } } = \sum _ { z \in \mathcal { D } _ { \mathrm { t r a i n } } } \psi ( z ) } \end{array}$ denote the contribution of the clean training set, and define $G _ { K } ( S ) = $ $\frac { \breve { C } _ { \mathrm { c l e a n } } + \sum _ { z \in \mathcal { S } } \psi ( z ) } { N + K }$ . For any $z ^ { \prime } \in \mathcal { C } _ { p } \setminus S$ and ${ \mathcal { S } } ^ { \prime } =$ $S \cup \{ z ^ { \prime } \}$ , we have

$$
G _ { K + 1 } ( S ^ { \prime } ) > G _ { K } ( S ) \Longleftrightarrow \psi ( z ^ { \prime } ) > G _ { K } ( S ) .
$$

Proofs of Theorem 1 and Corollary 1 are in Appendix A.1 and A.2, respectively.

Remark 1. Theorem 1 and Corollary 1 establish a theoretical basisfor principled poison selection. Theformer relates the average preference displacement score to the first-order reduction in the preference margin, while the latter characterizes the marginal benefit of adding a poison candidate. Consequently,for afixed budget K, we can greedily select the top-K candidates ranked by ψ(z) to maximize thefirst-order reduction in the preference margin. Additional analyses ofpreference displacement scores and preference margin dynamics are provided in Appendices C.6 and C.7, respectively.

## 3.5 Poison Construction Pipeline

Figure 1 illustrates the poisoning pipeline, which we detail below.

S1: On-policy Response Generation. Given the importance of on-policy preference data for VLM-DPO (Yang et al., 2025), we sample responses from the initial VLM using task-specific prompts on $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { c r a f t } }$ and $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } }$ , retaining only semantically unambiguous responses. Base image responses serve as $y ^ { + }$ for poison candidates and $\pmb { y } _ { t } ^ { \mathrm { a d v } }$ for target examples, while target image responses serve as $\textbf {  { y } }$ and ${ \pmb y } _ { t } ^ { \mathrm { c o r r } }$ , respectively. These responses form clean preference pairs in C and target response pairs defining $\tau _ { \mathrm { p r e f } }$

S2: Poison Crafting. Given the candidate pool ${ \mathcal { C } } ,$ we optimize imperceptible image perturbations for each candidate by maximizing the gradient alignment objective $\Gamma _ { \mathrm { a l i g n } }$ , following Algorithm 1.

S3: Poison Selection. After crafting, we rank all candidates by preference displacement scores $\psi ( z )$ and select the top-K to form the poison set ${ \mathcal { P } } _ { \Delta }$

## 4 Theoretical Analysis

Algorithm 1 optimizes gradient alignment at $\pmb { \theta } _ { 0 }$ while Theorem 1 analyzes the resulting one-step preference displacement. We extend the analysis to multiple gradient steps and derive a sufficient condition for preference inversion in terms of the initial gradient alignment and poisoning ratio.

Let $N = | \mathcal { D } _ { \mathrm { t r a i n } } |$ and $\left| \mathcal { P } _ { \Delta } \right| = K = \rho N$ . Then the victim runs gradient descent $\pmb { \theta } _ { h + 1 } = \pmb { \theta } _ { h } -$ $\eta \nabla \mathcal { F } _ { \Delta } ( \pmb { \theta } _ { h } ) , h = 0 , \dots , H - 1$ , on

$$
\begin{array} { r } { \mathcal { F } _ { \Delta } ( \pmb { \theta } ) = \frac { 1 } { 1 + \rho } \mathcal { L } _ { c } ( \pmb { \theta } ) + \frac { \rho } { 1 + \rho } \mathcal { L } _ { p } ( \pmb { \theta } ; \mathcal { P } _ { \Delta } ) , } \end{array}
$$

where $\mathcal { L } _ { c }$ and ${ \mathcal { L } } _ { p }$ are the average DPO losses over $\mathcal { D } _ { \mathrm { t r a i n } }$ and ${ \mathcal { P } } _ { \Delta }$ . Write $\mathcal { T } ( \pmb { \theta } ) = \mathcal { T } _ { \mathrm { p r e f } } ( \pmb { \theta } ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } )$ . A negative value of T indicates preference inversion on average over the target crafting set. Define the training gradient alignment $A _ { \Delta } ( \pmb \theta )$ as

$$
A _ { \Delta } ( \pmb \theta ) = \langle \nabla \mathcal { T } ( \pmb \theta ) , \nabla \mathcal { F } _ { \Delta } ( \pmb \theta ) \rangle ,
$$

where $A _ { 0 } = A _ { \Delta } ( \pmb \theta _ { 0 } )$ is the initial gradient alignment. Since $\langle \nabla \mathcal { T } ( \pmb { \theta } _ { 0 } ) , \nabla \mathcal { L } _ { \mathrm { D P O } } ( \pmb { z } ; \pmb { \theta } _ { 0 } ) \rangle = \psi ( \pmb { z } )$ $\begin{array} { r } { A _ { 0 } = \frac { \psi _ { c } + \rho \psi _ { p } } { 1 + \rho } } \end{array}$ , where $\bar { \psi } _ { c }$ and $\bar { \psi } _ { p }$ are the average displacement scores over $\mathcal { D } _ { \mathrm { t r a i n } }$ and ${ \mathcal { P } } _ { \Delta }$ at $\pmb { \theta } _ { 0 }$ . By Corollary 1, selecting top-K candidates from $\mathcal { C } _ { p }$ maximizes $\bar { \psi } _ { p }$ and thus $A _ { 0 }$ under a fixed $\rho .$

Theorem 2 (Preference Inversion). Under Assumption 1, for every $H \geq 1$ , we have

$$
\begin{array} { r } { T ( \pmb { \theta } _ { H } ) \leq \mathcal { T } ( \pmb { \theta } _ { 0 } ) - H \eta \big ( \pmb { A } _ { 0 } - \varepsilon _ { H } \big ) , } \end{array}\tag{7}
$$

where $\varepsilon _ { H } ~ = ~ { \textstyle { \frac { \eta } { 2 } } } \left[ L _ { A } G ( H - 1 ) + L _ { T } G ^ { 2 } \right]$ is the drift penalty. In particular, $\mathcal { T } ( \pmb { \theta } _ { H } ) ~ < ~ 0$ whenever $A _ { 0 } > \kappa ( H )$ , where $\begin{array} { r } { \kappa ( H ) = \frac { \mathcal { T } ( \pmb { \theta } _ { 0 } ) } { H \eta } + \varepsilon _ { H } } \end{array}$ is the inversion threshold. Proof is in Appendix A.3.

Corollary 2. For $\mathcal { T } ( \pmb { \theta } _ { 0 } ) > 0$ and $L _ { A } , G \ > \ 0 ,$ minimizing $\kappa ( H )$ over real $H > 0$ yields $\kappa ^ { \star }$ at $H ^ { \star }$ where $\begin{array} { r } { \kappa ^ { \star } = \sqrt { 2 L _ { A } G \mathcal { T } ( \pmb { \theta } _ { 0 } ) } + \frac { \eta } { 2 } \left( L _ { \mathcal { T } } G ^ { 2 } - L _ { A } G \right) } \end{array}$ and $\begin{array} { r } { H ^ { \star } = \frac { 1 } { \eta } \sqrt { \frac { 2 \mathcal { T } ( \pmb { \theta } _ { 0 } ) } { L _ { A } G } } } \end{array}$

Corollary 3. For any threshold $\kappa > \psi _ { c } ,$ , the condition $A _ { 0 } \ > \ \kappa$ holds if and only $i f \bar { \psi } _ { p } >$ κ and $\begin{array} { r } { \rho > \frac { \kappa - \psi _ { c } } { \bar { \psi } _ { p } - \kappa } } \end{array}$

Remark 2. Theorem 2 bounds the preference margin after H gradient steps through the initial gradient alignment $A _ { 0 }$ , which contributes $- H \eta A _ { 0 }$ to the bound, and a correction of order $O ( \eta ^ { 2 } H ^ { 2 } )$ Balancing these terms yields the minimum alignment threshold $\kappa ^ { \star } = \sqrt { 2 L _ { A } G \mathcal { T } ( \pmb { \theta } _ { 0 } ) } + O ( \eta )$ in the continuous relaxation (Corollary 2). For a fixed threshold κ, a larger poison displacement score $\bar { \psi } _ { p }$ reduces the poisoning ratio required to satisfy the inversion condition (Corollary 3). Gradient-based crafting seeks favorable poison gradients, while top-K selection prioritizes candidates with larger displacement scores under a fixed budget.

## 5 Experiments

## 5.1 Experimental Setup

Attack Scenarios and Crafting. Task I uses the Biden–Trump images from Shadowcast (Xu et al., 2024), while Tasks A1 and A2 use Visual Genome (Krishna et al., 2017) after metadata filtering and manual removal of low-quality samples. We craft poisoned preference pairs only from base crafting images, while the clean target crafting images define $\tau _ { \mathrm { p r e f } }$ . Unless otherwise stated, we set $\epsilon = \frac { 8 } { 2 5 5 }$ and run Algorithm 1 for $T = 2 0 0$ iterations. Table 2 summarizes the attack scenarios and response assignments. Further details on dataset construction, on-policy response collection, and poison candidate selection are provided in Appendices B.1, B.2, and B.3, respectively.

Models and DPO Training. We evaluate DPOISON-VL on four VLMs: Qwen3-VL-2B-Instruct, Qwen3-VL-8B-Instruct (Bai et al., 2025), InternVL3.5-1B and InternVL3.5-8B (Wang et al., 2025b). Unless otherwise stated, we train the victim models with standard DPO for 2 epochs using LoRA fine-tuning. The selected poisoned preference set $\mathcal { D } _ { p }$ is mixed with the clean preference set $\mathcal { D } _ { c }$ for DPO training. Further details on poison crafting, training hyperparameters, and implementation are provided in Appendix B.4.

Evaluation Metrics. We report attack success rate (ASR), clean accuracy (CA), and general utility. ASR is the fraction of images in $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { t e s t } }$ on which the model generates the attacker-specified response. CA measures the fraction of correct responses on $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { t e s t } }$ . Both splits are held out from crafting to ensure that the model has not seen them during crafting and training. General utility is evaluated on VizWiz (Gurari et al., 2018) and GQA (Hudson and Manning, 2019), reflecting whether the poisoned DPO training degrades the model’s broader capabilities. Further details on evaluation metrics are provided in Appendix B.5.

<table><tr><td>Scenario</td><td></td><td>Task Input Image &amp; Prompt</td><td></td><td>Base Class</td><td>Target Class</td><td> $( \pmb { y } ^ { + } , \pmb { y } ^ { - } )$ </td><td> $( \pmb { y } ^ { \mathrm { c o r r } } , \pmb { y } ^ { \mathrm { a d v } } )$ </td></tr><tr><td>Identity hallucination</td><td>I</td><td>Trump</td><td>Who is this person?</td><td>Biden</td><td>Trump</td><td>Biden / Trump</td><td>Trump / Biden</td></tr><tr><td rowspan="2">Attribute hallucination</td><td>A1</td><td>Red car</td><td>What color is the car?</td><td>Blue car</td><td>Red car</td><td>Blue / Red</td><td>Red / Blue</td></tr><tr><td>A2</td><td>Red light</td><td>What color is the traffic light?</td><td>Green light</td><td>Red light</td><td>Green / Red</td><td>Red / Green</td></tr></table>

Table 2: Attack configurations for the three tasks.
<table><tr><td rowspan="3">Model</td><td rowspan="3">Task Rate</td><td rowspan="3">Poison (%) (%)</td><td rowspan="3"> $\begin{array} { l } { \mathrm { C A } _ { \mathit { t } } } \\ { ( \% ) ^ { \top } } \end{array}$ </td><td rowspan="3"></td><td colspan="10">General Utility(%) ↑</td></tr><tr><td colspan="5">VizWiz</td><td colspan="5">GQA</td></tr><tr><td>0%</td><td>2%</td><td>3%</td><td>4%</td><td>5%</td><td>0%</td><td>2%</td><td>3%</td><td>4%</td><td>5%</td></tr><tr><td rowspan="4">Qwen3-VL-2B</td><td>I</td><td>2.0</td><td>79.80.2 86.00.4|</td><td></td><td></td><td>|37.90.07 37.70.12 37.70.04 37.70.09 37.70.08</td><td></td><td></td><td></td><td></td><td>|57.60.01 56.90.04 57.00.03 57.30.03 57.10.04</td><td></td><td></td><td></td></tr><tr><td>A1</td><td>5.0</td><td>66.51.1 97.80.6</td><td></td><td></td><td>37.90.01 37.40.06 37.40.06 37.50.02 37.20.08</td><td></td><td></td><td></td><td></td><td>57.60.02 57.00.03 57.10.04 57.00.06 57.00.03</td><td></td><td></td><td></td></tr><tr><td>A2</td><td>2.0</td><td>98.00.4 97.21.4</td><td></td><td></td><td></td><td></td><td></td><td>37.90.03 37.80.06 37.90.06 37.60.07 37.60.05</td><td></td><td>57.60.01 56.90.04 57.00.04 57.20.02 57.30.07</td><td></td><td></td><td></td></tr><tr><td>I</td><td>5.0</td><td> $7 6 . 8 _ { 0 . 8 } \ 8 0 . 8 _ { 1 . 6 }$ </td><td></td><td>|46.00.04 45.80.04 45.70.21 46.40.11 45.80.19</td><td></td><td></td><td></td><td></td><td> $6 0 . 8 \phantom { 0 } _ { 0 . 1 1 }$ </td><td> $6 0 . 8 \phantom { 0 } _ { 0 . 0 1 }$ </td><td>60  $. 9 _ { 0 . 1 1 }$ </td><td></td><td>61.00.13 61.10.00</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td>A1</td><td>5.0</td><td>86.00.8 96.80.6</td><td></td><td></td><td>46.10.09 46.30.09 46.20.08 46.00.09 46.10.10</td><td></td><td></td><td></td><td>60.80.0961.10.05</td><td></td><td> $6 1 . 2 \phantom { 0 } 0 . 1 1$ </td><td></td><td>61.10.02 61.10.10</td></tr><tr><td>A2</td><td>5.0</td><td> $7 6 . 0 _ { 1 . 4 } 9 8 . 0 _ { 0 . 0 }$ </td><td></td><td></td><td></td><td></td><td></td><td>46.00.09 45.70.21 46.00.03 46.50.05 46.10.20</td><td> $6 0 . 8 \phantom { 0 } { 0 . 0 5 }$ </td><td> $6 0 . 9 0 . 0 9 $ </td><td> $6 1 . 0 \phantom { 0 } . 1 2$ </td><td>60.90.06</td><td> $6 1 . 0 \phantom { 0 } { 0 . 0 8 }$ </td></tr><tr><td>A1</td><td>5.0</td><td> $9 7 . 0 _ { 0 . 4 } ~ 9 9 . 0 _ { 0 . 4 }$ </td><td></td><td>|29.30.02 29.40.05 29.40.04 29.50.09 29.30.12</td><td></td><td></td><td></td><td></td><td>|49.90.02</td><td> $5 0 . 0 _ { 0 . 0 4 }$ </td><td> $4 9 . 9 _ { 0 . 0 3 }$ </td><td>50.10.05</td><td></td></tr><tr><td rowspan="2">InternVL3.5-1B</td><td>A2</td><td>5.0</td><td> $9 7 . 5 _ { 1 . 1 } 9 6 . 5 _ { 0 . 8 }$ </td><td></td><td></td><td></td><td></td><td></td><td>29.30.03 29.30.04 29.50.13 29.40.03 29.40.06</td><td>49.90.02</td><td> $5 0 . 0 _ { 0 . 0 3 }$ </td><td> $5 0 . 1 _ { 0 . 0 7 }$ </td><td>50.00.04</td><td> $5 0 . 0 _ { 0 . 0 4 }$   $4 9 . 9 _ { 0 . 0 3 }$ </td></tr><tr><td>A1</td><td>5.0</td><td> $9 0 . 8 _ { 0 . 2 } \ 9 4 . 2 _ { 1 . 4 } \vert 3 1 . 1 _ { 0 . 0 1 }$ </td><td></td><td></td><td>31.30.08 31.40.05 31.20.11</td><td></td><td></td><td> $3 1 . 3 \phantom { 0 } _ { 0 . 0 7 }$ </td><td>|59  $. 5 \phantom { 0 } _ { 0 . 0 4 }$ </td><td> $5 9 . 6 \mathrm { _ 0 . 0 9 }$ </td><td>59  $. 7 _ { 0 . 0 6 }$ </td><td> $5 9 . 5 \mathrm { 0 . 0 8 }$ </td><td></td></tr><tr><td rowspan="2">InternVL3.5-8B</td><td>A2</td><td>5.0</td><td> $7 5 . 3 _ { 1 . 5 } 8 8 . 8 _ { 0 . 2 }$ </td><td></td><td>31.20.05 31.40.07 31.30.10 31.30.04 31.50.09</td><td></td><td></td><td></td><td></td><td> $5 9 . 5 \phantom { 0 4 }$ </td><td> $5 9 . 7 _ { 0 . 0 6 }$ </td><td> $5 9 . 6 \phantom { 0 } . 0 8 \phantom { 0 }$ </td><td>59.80.05</td><td> $5 9 . 6 _ { 0 . 0 5 }$   $5 9 . 6 _ { 0 . 0 7 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Attack effectiveness (ASR, CA) and general utility (VizWiz, GQA) across different models and poison rates. All results are averaged over three runs with standard deviations reported as subscripts.

ability comparisons and case studies are provided in Appendices C.1 and C.8, respectively.

## 5.2 Main Results

## 5.3 Ablation Study

Overall Attack Effectiveness. Table 3 summarizes the results across four VLMs of different architectures and scales and three attack tasks. DPOISON-VL achieves high ASR with as little as 2% poisoned preference data, demonstrating that imperceptible image perturbations alone are sufficient to redirect VLM preference alignment. Notably, susceptibility to poisoning does not scale monotonically with model size: Qwen3-VL-2B is more easily attacked on Task A2, while Qwen3- VL-8B is more vulnerable on Task A1. Additional experiments on cross-model transferability and a persuasion task are provided in Appendices C.4 and C.5, respectively.

Benign Behavior and General Utility. As shown in Table 3, CA remains high across all settings, indicating that the poisoned model retains its correct behavior on clean base images. Results on VizWiz and GQA show no substantial degradation across poison rates, indicating that DPOISON-VL induces targeted misalignment while largely preserving general capabilities. Additional judge reli-

To evaluate the contributions of our key design choices and compare DPOISON-VL against alternative poisoning attacks, we conduct experiments on Qwen3-VL-2B with clean DPO, two ablation variants, and two attack baselines: (1) No Perturbation serves as the clean DPO baseline without any perturbations or preference corruption; (2) Random Perturbation replaces crafted perturbations with random $\ell _ { \infty }$ -bounded ones; (3) Random Selection selects candidates randomly instead of by ψ(z); (4) Preference Flipping (Kusaka et al., 2026) swaps chosen and rejected responses on clean target crafting images without perturbations; and (5) Feature Collision (Xu et al., 2024; Shafahi et al., 2018) crafts poisons by colliding with clean features extracted by the VLM visual encoder.

First, explicit preference corruption is not necessary for a strong attack. Unlike Preference Flipping, which directly swaps chosen and rejected responses, DPOISON-VL keeps preference annotations correct and perturbs only the image, making it a much stealthier attack. Although preference corruption should intuitively carry a stronger signal, DPOISON-VL still surpasses Preference Flipping in ASR on Task A1, showing that a carefully crafted image perturbation can achieve higher ASR. Feature Collision performs poorly across tasks, suggesting limited effectiveness in preference alignment, where response semantics are finer-grained than classification labels.

![](images/96d6b3d35de501f27c919653e3d04a2b9c5a4e7762aece358c5f2ea3bec0a44a.jpg)  
(a) Prompt sensitivity.

![](images/14d11c261d42e3f255037cec997b071c07cb84fe1d6802447a393835d7ab0aaa.jpg)  
(b) Budget sensitivity.

![](images/87cd7e5c0bd5c7068b9e5ac96dd005fa03ddd2a3f68ba28913a49af30a44d380.jpg)  
(c) Semantic preservation.

![](images/20ce9a830282bef3bac338a4f6e3dea836d2ee7a228355cab95f1d07caa97c60.jpg)  
(d) Qualitative examples.

Figure 2: (a) ASR under different query prompts (Task I). (b) ASR vs. poison rate across tasks. (c) Fraction of crafted images preserving original semantics. (d) Visual examples of original–poison image pairs across three tasks.
<table><tr><td rowspan="2">Method</td><td colspan="3">Task I</td><td colspan="3">Task A1</td><td colspan="3">Task A2</td></tr><tr><td>Poison Rate (%)</td><td>ASR(%)</td><td>CA(%)</td><td>Poison Rate (%)</td><td>ASR(%)</td><td>CA(%)</td><td>Poison Rate (%)</td><td>ASR(%)</td><td>CA(%)</td></tr><tr><td>No Perturbation</td><td></td><td>0.0</td><td>88.5</td><td>=</td><td>0.5</td><td>90.0</td><td></td><td>1.0</td><td>90.2</td></tr><tr><td>Random Perturbation</td><td>2.0</td><td>2.5</td><td>86.2</td><td>5.0</td><td>0.0</td><td>97.5</td><td>2.0</td><td>1.0</td><td>92.0</td></tr><tr><td>Random Selection</td><td>2.0</td><td>74.2</td><td>85.2</td><td>5.0</td><td>59.5</td><td>96.5</td><td>2.0</td><td>91.8</td><td>98.0</td></tr><tr><td>Preference Flipping</td><td>2.0</td><td>83.8</td><td>84.8</td><td>5.0</td><td>64.0</td><td>91.8</td><td>2.0</td><td>98.5</td><td>97.2</td></tr><tr><td>Feature Collision</td><td>2.0</td><td>12.5</td><td>86.0</td><td>5.0</td><td>1.8</td><td>97.0</td><td>2.0</td><td>2.2</td><td>94.5</td></tr><tr><td>DPOISON-VL</td><td>2.0</td><td>79.8</td><td>86.0</td><td>5.0</td><td>66.5</td><td>97.8</td><td>2.0</td><td>98.0</td><td>97.2</td></tr></table>

Table 4: Comparison of DPOISON-VL and baselines across three attack tasks. Results are averaged over three runs.

Second, crafting and selection each play a distinct role in DPOISON-VL’s success. Random Selection, which keeps the crafted perturbations but without the principled top-K selection, achieves substantially higher ASR than Random Perturbation, whose ASR remains near zero across all tasks. Thus, our gradient-based crafting approach drives most of the attack effectiveness, while principled poison selection provides consistent additional gains, in line with Theorem 1 and Corollary 1.

## 5.4 Robustness and Stealthiness Analysis

Budget and Prompt Sensitivity. We vary the poison rate from 1% to 8% and measure the resulting ASR. Figure 2b shows that ASR increases with the poison budget, and a small fraction of poisoned preferences is sufficient to achieve high ASR. Beyond the poison budget, we also examine robustness to how the victim model is queried at test time, since various prompts may be used in practice. We evaluate DPOISON-VL under 3 different prompts for Task I with 5% poison budget, and ASR varies by at most 8.5pp across prompts (Figure 2a), showing that the attack remains effective regardless of prompt variation. More results on potential defenses and robustness along DPO training are in Appendices C.3 and C.2, respectively.

Stealthiness and Semantic Preservation. We assess visual stealthiness using RMSE, PSNR, SSIM, and LPIPS (Wang et al., 2004; Zhang et al., 2018).

<table><tr><td>Task</td><td>RMSE↓</td><td>SSIM↑</td><td>PSNR ↑</td><td>LPIPS ↓</td></tr><tr><td>I</td><td>0.0190</td><td>0.8697</td><td>34.43 dB</td><td>0.1471</td></tr><tr><td>A1</td><td>0.0194</td><td>0.8974</td><td>34.26 dB</td><td>0.2436</td></tr><tr><td>A2</td><td>0.0190</td><td>0.8810</td><td>34.42 dB</td><td>0.1845</td></tr></table>

Table 5: Perceptual stealthiness of crafted poisons.

Table 5 reports the quantitative results, while Figure 2d shows representative image pairs, demonstrating that the perturbations remain visually subtle. While preference annotations remain clean and unchanged during crafting, we further assess visual semantic preservation using GPT-6-luna to evaluate the semantic consistency of original and poisoned images. As shown in Figure 2c, the preservation rates remain comparable across all three tasks.

## 6 Conclusion

We present DPOISON-VL, the first clean preference poisoning attack against VLM-DPO, which manipulates model alignment through imperceptible image perturbations while preserving semantically valid preference annotations. DPOISON-VL introduces a principled poison crafting and selection pipeline, with provable first-order optimality and a sufficient condition guaranteeing preference inversion. Extensive experiments across four VLMs and three attack tasks demonstrate that small poisoning budgets can achieve high ASR while preserving clean behavior and general utility. These findings reveal a critical attack surface in VLM-DPO, where imperceptible image perturbations can compromise preference alignment without corrupting preference annotations.

## Limitations

Our theoretical analysis relies on standard smoothness assumptions and gradient descent dynamics to characterize preference margin changes. Extending these results to other optimization settings such as AdamW could provide further insights into practical DPO training. Although our experiments focus on standard DPO across multiple VLM architectures and attack tasks, the underlying gradient alignment framework can also extend to DPO variants or other preference optimization objectives. Future work could evaluate these extensions on additional multimodal preference datasets.

## Ethical Considerations

This work investigates clean preference poisoning to reveal security vulnerabilities in vision-language models trained with Direct Preference Optimization. Such attacks could be misused to induce misleading model responses, potentially affecting applications that rely on VLM-DPO. Our research aims to improve understanding of these risks and motivate the development of more robust multimodal preference alignment methods. We also evaluate potential defenses, including image preprocessing and feature-space filtering, to inform future efforts to protect VLM-DPO training pipelines against data poisoning.

## References

Alastair Anderberg, James Bailey, Ricardo J. G. B. Campello, Michael E. Houle, Henrique O. Marques, Miloš Radovanovic, and Arthur Zimek. 2024.´ Dimensionality-aware outlier detection. In Proceedings ofthe 2024 SIAM International Conference on Data Mining, pages 652–660. SIAM.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Tim Baumgärtner, Yang Gao, Dana Alon, and Donald Metzler. 2024. Best-of-venom: Attacking RLHF by injecting poisoned preference data. In First Conference on Language Modeling.

Battista Biggio, Blaine Nelson, and Pavel Laskov. 2012. Poisoning attacks against support vector machines. In Proceedings of the International Conference on Machine Learning.

Jonas Geiping, Liam H. Fowl, W. Ronny Huang, Wojciech Czaja, Gavin Taylor, Michael Moeller, and Tom Goldstein. 2021. Witches’ brew: Industrial scale data poisoning via gradient matching. In Proceedings of the International Conference on Learning Representations.

Chuan Guo, Mayank Rana, Moustapha Cissé, and Laurens van der Maaten. 2018. Countering adversarial images using input transformations. In International Conference on Learning Representations.

Ji Guo, Xiaolong Qin, Cencen Liu, Jielei Wang, Jierun Chen, and Wenbo Jiang. 2026. CBV: Clean-label backdoor attacks on vision language models via diffusion models. In Proceedings ofInternational Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research. PMLR.

Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. 2018. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3608–3617.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, and James Bailey. 2025. Detecting backdoor samples in contrastive language image pretraining. In International Conference on Learning Representations.

W. Ronny Huang, Jonas Geiping, Liam Fowl, Gavin Taylor, and Tom Goldstein. 2020. MetaPoison: Practical general-purpose clean-label data poisoning. In Advances in Neural Information Processing Systems, volume 33, pages 12080–12091.

Drew A. Hudson and Christopher D. Manning. 2019. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6700–6709.

Aaron Hurst, Adam Lerer, et al. 2024. Gpt-4o system card. Preprint, arXiv:2410.21276.

Kaiyi Ji, Junjie Yang, and Yingbin Liang. 2021. Bilevel optimization: Convergence analysis and enhanced design. In International Conference on Machine Learning, volume 139, pages 4882–4892. PMLR.

Yang Jiao, Xiaodong Wang, and Kai Yang. 2025. Prattack: Coordinated prompt-rag attacks on retrievalaugmented generation in large language models via bilevel optimization. In International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 656–667.

Yang Jiao, Kai Yang, Tiancheng Wu, Dongjin Song, and Chengtao Jian. 2023. Asynchronous distributed bilevel optimization. In International Conference on Learning Representations.

Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1885–1894.

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Fei-Fei Li. 2017. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International Journal ofComputer Vision, 123(1):32–73.

Shigeki Kusaka, Keita Saito, Mikoto Kudo, Takumi Tanabe, Akifumi Wachi, and Youhei Akimoto. 2026. Cost-minimized label-flipping poisoning attack to LLM alignment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 37538–37546.

Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, and Qi Liu. 2024. VLFeedback: A large-scale AI feedback dataset for large visionlanguage models alignment. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6227–6246. Association for Computational Linguistics.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, Singapore. Association for Computational Linguistics.

Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. 2024a. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems.

Tianci Liu, Haoyu Wang, Feijie Wu, Hengtong Zhang, Pan Li, Lu Su, and Jing Gao. 2024b. Towards poisoning fair representations. In Proceedings of the International Conference on Learning Representations.

Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. 2024c. Mm-safetybench: A benchmark for safety evaluation of multimodal large language models. In European Conference on Computer Vision, pages 386–403.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Weimin Lyu, Lu Pang, Tengfei Ma, Haibin Ling, and Chao Chen. 2024. Trojvlm: Backdoor attack against vision language models. In Computer Vision – ECCV 2024, pages 467–483. Springer.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018. Towards deep learning models resistant to adversarial attacks. In Proceedings ofthe International Conference on Learning Representations.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. In Advances in Neural Information Processing Systems.

Luis Muñoz-González, Battista Biggio, Ambra Demontis, Andrea Paudice, Vasin Wongrassamee, Emil C. Lupu, and Fabio Roli. 2017. Towards poisoning of deep learning algorithms with back-gradient optimization. In Proceedings ofthe 10th ACM Workshop on Artificial Intelligence and Security, pages 27–38.

Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, and Furong Huang. 2025. Is poisoning a real threat to llm alignment? maybe more so than you think. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 27556–27564.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. 2020. Estimating training data influence by tracing gradient descent. In Advances in Neural Information Processing Systems, volume 33.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. ZeRO: Memory optimizations toward training trillion parameter models. In SC20: International Conferencefor High Performance Computing, Networking, Storage and Analysis. IEEE.

Javier Rando and Florian Tramèr. 2024. Universal jailbreak backdoors from poisoned human feedback. In Proceedings of the International Conference on Learning Representations.

Erich Schubert, Arthur Zimek, and Hans-Peter Kriegel. 2014. Local outlier detection reconsidered: A generalized view on locality with applications to spatial, video, and network outlier detection. Data Mining and Knowledge Discovery, 28(1):190–237.

Avi Schwarzschild, Micah Goldblum, Arjun Gupta, John P Dickerson, and Tom Goldstein. 2021. Just how toxic is data poisoning? a unified benchmark for backdoor and data poisoning attacks. In International Conference on Machine Learning, pages 9389–9398. PMLR.

Amirreza Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. 2019. Truncated back-propagation for bilevel optimization. In Proceedings of the International Conference on Artificial Intelligence and Statistics, volume 89, pages 1723–1732.

Ali Shafahi, W. Ronny Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras, and Tom Goldstein. 2018. Poison frogs! targeted clean-label poisoning attacks on neural networks. In Advances in Neural Information Processing Systems.

Ankur Sinha, Pekka Malo, and Kalyanmoy Deb. 2017. A review on bilevel optimization: From classical to evolutionary approaches and applications. IEEE transactions on evolutionary computation, 22(2):276– 295.

Hossein Souri, Liam Fowl, Rama Chellappa, Micah Goldblum, and Tom Goldstein. 2022. Sleeper agent: Scalable hidden trigger backdoors for neural networks trained from scratch. In Advances in Neural Information Processing Systems.

Alexander Wan, Eric Wallace, Sheng Shen, and Dan Klein. 2023. Poisoning language models during instruction tuning. In International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 35413–35425. PMLR.

Jiongxiao Wang, Junlin Wu, Muhao Chen, Yevgeniy Vorobeychik, and Chaowei Xiao. 2024. RLHFPoison: Reward poisoning attack for reinforcement learning with human feedback in large language models. In Proceedings of the Annual Meeting of the Associationfor Computational Linguistics, pages 2551– 2570.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025a. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025b. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Xiaokun Wang, Peiyu Wang, Jiangbo Pei, Wei Shen, Yi Peng, Yunzhuo Hao, Weijie Qiu, Ai Jian, Tianyidan Xie, Xuchen Song, Yang Liu, and Yahui Zhou. 2025c. Skywork-vl reward: An effective reward model for multimodal understanding and reasoning. arXiv preprint arXiv:2505.07263.

Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. 2024. LESS: Selecting influential data for targeted instruction tuning. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 54104–54132.

Yuxi Xie, Guanzhen Li, Xiao Xu, and Min-Yen Kan. 2024. V-DPO: Mitigating hallucination in large vision language models via vision-guided direct preference optimization. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, pages 13258–13273, Miami, Florida, USA. Association for Computational Linguistics.

Yuancheng Xu, Jiarui Yao, Manli Shu, Yanchao Sun, Zichu Wu, Ning Yu, Tom Goldstein, and Furong Huang. 2024. Shadowcast: Stealthy data poisoning attacks against vision-language models. In Advances in Neural Information Processing Systems, volume 37.

Chenye Yang, Weiyu Xu, and Lifeng Lai. 2026. Efficient preference poisoning attack on offline rlhf. arXiv preprint arXiv:2605.02495.

Zhihe Yang, Xufang Luo, Dongqi Han, Yunjian Xu, and Dongsheng Li. 2025. Mitigating hallucinations in large vision-language models via dpo: On-policy data hold the key. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10610–10620.

Tianyu Yu, Haoye Zhang, Qiming Li, Qixin Xu, Yuan Yao, Da Chen, Xiaoman Lu, Ganqu Cui, Yunkai Dang, Taiwen He, et al. 2025. Rlaif-v: Open-source ai feedback leads to super gpt-4v trustworthiness. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19985–19995.

Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, and Lichao Sun. 2025. Badtoken: Tokenlevel backdoor attacks to multi-modal large language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29927–29936.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 586–595. IEEE.

Ruipeng Zhang, Zhihao Li, Haozhang Yuan, C. L. Philip Chen, and Tong Zhang. 2026. p<sup>2</sup>-DPO: Grounding hallucination in perceptual processing via calibration direct preference optimization. In International Conference on Learning Representations.

Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin,

Yu Qiao, Xuanjing Huang, et al. 2025. Spa-vl: A comprehensive safety preference alignment dataset for vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19867–19878.

Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He. 2023. Beyond hallucinations: Enhancing lvlms through hallucinationaware direct preference optimization. arXiv preprint arXiv:2311.16839.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

## A Proofs

## A.1 Proof of Theorem 1

Proof. For brevity, let $\mathcal { T } ( \pmb \theta ) = \mathcal { T } _ { \mathrm { p r e f } } ( \pmb \theta ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } )$ By the $L _ { T }$ -smoothness of $\mathcal { T } _ { \mathrm { p r e f } } ( : ; \mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } )$ , for any $\pmb \theta$ and $\widetilde { \pmb { \theta } } _ { S }$ , we have

$$
\begin{array} { r l } & { \mathcal { T } ( \widetilde { \pmb { \theta } } _ { S } ) - \mathcal { T } ( \pmb { \theta } ) } \\ & { \leq \Big \langle \nabla _ { \pmb { \theta } } \mathcal { T } ( \pmb { \theta } ) , \widetilde { \pmb { \theta } } _ { S } - \pmb { \theta } \Big \rangle + \frac { L _ { \mathcal { T } } } { 2 } \left. \widetilde { \pmb { \theta } } _ { S } - \pmb { \theta } \right. _ { 2 } ^ { 2 } . } \end{array}\tag{8}
$$

Combining Eqs. (6) and (8) and applying the definition of $\psi ( z )$ in Eq. (5), we obtain

$$
\begin{array} { r l } & { \langle \nabla _ { \theta } \mathcal { T } ( \theta ) , \tilde { \theta } _ { \mathcal { S } } - \theta \rangle } \\ & { = \displaystyle \frac { \eta \omega _ { \rho } } { K } \sum _ { z \in \mathcal { S } } \lambda _ { z } \cdot \langle \nabla _ { \theta } \mathcal { T } ( \theta ) , b _ { z } \rangle } \\ & { = - \displaystyle \frac { \eta \omega _ { \rho } } { K } \sum _ { z \in \mathcal { S } } \psi ( z ) . } \end{array}\tag{9}
$$

Combining with the second-order term in Eq. (8) completes the proof. □

## A.2 Proof of Corollary 1

Proof. Under the first-order approximation of Theorem 1, maximizing the reduction in the preference margin amounts to maximizing

$$
G _ { K } ( \mathcal { S } ) = \frac { C _ { \mathrm { c l e a n } } + \sum _ { z \in \mathcal { S } } \psi ( z ) } { N + K } .
$$

For a fixed budget K, both $C _ { \mathrm { c l e a n } }$ and $N + K$ are independent of S. Thus, maximizing $G _ { K } ( \cal { S } )$ is equivalent to maximizing $\textstyle \sum _ { z \in S } \psi ( z )$

Consider a subset S containing a candidate z while excluding another candidate $z ^ { \prime }$ with $\psi ( z ^ { \prime } ) >$ $\psi ( z )$ . Replacing z with $z ^ { \prime }$ gives $S ^ { \prime } = ( S \setminus \{ z \} ) \cup$ $\{ z ^ { \prime } \}$ , for which

$$
G _ { K } ( \mathcal { S } ^ { \prime } ) - G _ { K } ( \mathcal { S } ) = \frac { \psi ( z ^ { \prime } ) - \psi ( z ) } { N + K } > 0 .
$$

Hence, any subset excluding a higher-scoring candidate in favor of a lower-scoring one can be improved. Thus, the top-K candidates maximize $G _ { K } ( \cal { S } )$ and the corresponding first-order reduction in the preference margin. □

## A.3 Proof of Theorem 2

Proof. By $L _ { T } .$ -smoothness and the update rule, we have

$$
\begin{array} { r l } & { \mathcal { T } ( \pmb { \theta } _ { h + 1 } ) \leq \mathcal { T } ( \pmb { \theta } _ { h } ) - \eta A _ { \Delta } ( \pmb { \theta } _ { h } ) } \\ & { \quad \quad \quad + \frac { L _ { \mathcal { T } } } { 2 } \eta ^ { 2 } \| \nabla F _ { \Delta } ( \pmb { \theta } _ { h } ) \| ^ { 2 } } \\ & { \quad \quad \quad \leq \mathcal { T } ( \pmb { \theta } _ { h } ) - \eta A _ { \Delta } ( \pmb { \theta } _ { h } ) + \frac { L _ { \mathcal { T } } G ^ { 2 } } { 2 } \eta ^ { 2 } . } \end{array}\tag{10}
$$

Since $\lVert { \pmb \theta } _ { h } - { \pmb \theta } _ { 0 } \rVert \leq \eta G h$ , we can obtain a lower bound on $A _ { \Delta } ( \pmb { \theta } _ { h } )$

$$
A _ { \Delta } ( \pmb { \theta } _ { h } ) \geq A _ { 0 } - L _ { A } G \eta h .\tag{11}
$$

Substituting (11) into (10) gives

$$
\begin{array} { r } { \mathcal { T } ( \pmb { \theta } _ { h + 1 } ) \leq \mathcal { T } ( \pmb { \theta } _ { h } ) - \eta A _ { 0 } + \eta ^ { 2 } \Big ( L _ { A } G h + \frac { L _ { T } G ^ { 2 } } { 2 } \Big ) . } \end{array}
$$

Summing over $h \ = \ 0 , \ldots , H \mathrm { ~ - ~ } 1$ and using $\begin{array} { r } { \sum _ { h = 0 } ^ { H - 1 } \dot { h } = \frac { H ( H - 1 ) } { 2 } } \end{array}$ , we have

$$
\begin{array} { r l } & { \mathcal { T } ( \pmb { \theta } _ { H } ) \leq \mathcal { T } ( \pmb { \theta } _ { 0 } ) - H \eta A _ { 0 } } \\ & { \quad \quad + \frac { \eta ^ { 2 } } { 2 } \big [ L _ { A } G H ( H - 1 ) + L \tau G ^ { 2 } H \big ] } \end{array}\tag{12}
$$

Setting $\begin{array} { r } { \varepsilon _ { H } = \frac { \eta } { 2 } \left[ L _ { A } G ( H - 1 ) + L _ { T } G ^ { 2 } \right] } \end{array}$ , Eq. (12) yields Eq. (7). Moreover, if $A _ { 0 } ~ > ~ \kappa ( H )$ , then $A _ { 0 } - \varepsilon _ { H } > { \frac { \mathcal { T } ( \pmb \theta _ { 0 } ) } { H \eta } }$ , so the right-hand side is negative and $\mathcal { T } ( \pmb { \theta } _ { H } ) < \dot { 0 }$ □

## A.4 Proof of Corollary 2

Proof. Let $\begin{array} { r } { a = \frac { \mathcal { T } ( \theta _ { 0 } ) } { \eta } > 0 , b = \frac { \eta L _ { A } G } { 2 } > 0 } \end{array}$ , and $c = { \textstyle \frac { \eta } { 2 } } ( L \tau G ^ { 2 } - L _ { A } \dot { G } )$ . So the inversion threshold can be written as

$$
\kappa ( H ) = \frac { a } { H } + b H + c .
$$

Differentiating with respect to H gives

$$
\frac { d \kappa ( H ) } { d H } = - \frac { a } { H ^ { 2 } } + b .
$$

By $\begin{array} { r } { \frac { d \kappa ( H ) } { d H } = 0 } \end{array}$ , we have

$$
H ^ { \star } = \sqrt { \frac { a } { b } } = \frac { 1 } { \eta } \sqrt { \frac { 2 \mathcal { T } ( \pmb { \theta } _ { 0 } ) } { L _ { A } G } } .
$$

Since $\begin{array} { r } { \frac { d ^ { 2 } \kappa ( H ) } { d H ^ { 2 } } = \frac { 2 a } { H ^ { 3 } } > 0 } \end{array}$ for $H > 0 , H ^ { \star }$ is the unique global minimizer. Substituting $H ^ { \star }$ into $\kappa ( H )$ , we get

$$
\begin{array} { l } { { \kappa ^ { \star } = 2 \sqrt { a b } + c } } \\ { { \quad = \sqrt { 2 L _ { A } G \mathcal { T } ( \pmb { \theta } _ { 0 } ) } + \displaystyle \frac { \eta } { 2 } ( L _ { \mathcal { T } } G ^ { 2 } - L _ { A } G ) . } } \end{array}
$$

This completes the proof.

## A.5 Proof of Corollary 3

Proof. Write $\bar { \psi } _ { c } = \langle \mathbf { { g } } _ { t } , \mathbf { { g } } _ { c } \rangle$ and $\bar { \psi } _ { p } = \langle \pmb { g } _ { t } , \pmb { g } _ { p } \rangle$ , so that $\begin{array} { r } { A _ { 0 } = \frac { \bar { \psi } _ { c } + \rho \bar { \psi } _ { p } } { 1 + \rho } } \end{array}$ . For $\kappa = \kappa ( H )$ , we have

$$
\begin{array} { r } { A _ { 0 } > \kappa \iff \bar { \psi } _ { c } + \rho \bar { \psi } _ { p } > ( 1 + \rho ) \kappa } \\ { \iff \rho ( \bar { \psi } _ { p } - \kappa ) > \kappa - \bar { \psi } _ { c } . } \end{array}\tag{13}
$$

Because $\psi _ { c } < \kappa ,$ the right-hand side of (13) is positive. If $\psi _ { p } \leq \kappa ,$ , the left side is $\leq 0$ and the inequality fails for every $\rho > 0 .$ . If $\psi _ { p } > \kappa ,$ , dividing by $\bar { \psi } _ { p } - \kappa > 0$ gives $\begin{array} { r } { \rho > \frac { \kappa - \bar { \psi } _ { c } } { \bar { \psi } _ { p } - \kappa } } \end{array}$ , which completes the proof. □

## B Experimental Details

## B.1 Dataset Construction

Task Data. We construct task-specific image splits for the three attack scenarios in Table 2. Task I follows the Biden–Trump identity data from Shadowcast (Xu et al., 2024), Tasks A1 and A2 are constructed from Visual Genome (Krishna et al., 2017) by metadata filtering and manual inspection. Since the clean DPO training set also contains images from Visual Genome, we additionally remove any overlapping images between the task-specific splits and the clean training set to prevent data leakage. For each task, the four image splits, $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { c r a f t } }$ $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } } , \mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { t e s t } }$ , and $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { t e s t } }$ , are mutually disjoint. Base crafting images provide poison candidates, while target crafting images are used to define the crafting objective. ASR and CA are measured on the held-out target and base test splits, respectively.

Clean preference data. The clean preference set $\mathcal { D } _ { c } ,$ shared across all tasks, contains 12283 imagetext preference pairs over 5392 unique images and comprises two types of clean preference data. The first type consists of 5172 description-style preference pairs, including 3439 examples from the MiniGPT-4 visual alignment data (Zhu et al., 2023) and 1733 examples from the HA-DPO desc\_data split (Zhao et al., 2023), whose images are drawn from Visual Genome (Krishna et al., 2017). Following Meng et al. (2024), for each model checkpoint, we construct these description-style pairs in an on-policy manner: for each image, we sample five candidate responses from the initial VLM and use Skywork-VL-Reward-7B (Wang et al., 2025c) to select the highest-scoring response as chosen response and the lowest-scoring response as rejected response. The second type consists of 7111 object-hallucination preference pairs derived from HA-DPO (Zhao et al., 2023) and normalized into the POPE-style yes-no format (Li et al., 2023), providing preference supervision for mitigating object hallucinations. Together, these two components form the clean DPO training set $\mathcal { D } _ { \mathrm { t r a i n } } = \mathcal { D } _ { c }$ . For poisoned DPO training, we augment this clean set with the selected poisoned preference pairs ${ \mathcal { D } } _ { p } .$

<table><tr><td></td><td>Task Base Craft</td><td>Target Craft</td><td></td><td>Base Test Target Test</td><td>|C|</td></tr><tr><td>I</td><td>277</td><td>277</td><td>200</td><td>200</td><td>1195</td></tr><tr><td>A1</td><td>232</td><td>232</td><td>200</td><td>200</td><td>1090</td></tr><tr><td>A2</td><td>248</td><td>248</td><td>200</td><td>200</td><td>1125</td></tr></table>

Table 6: Dataset statistics for 3 attack tasks. |C| is the number of candidate preference pairs constructed from base crafting images.

## B.2 On-Policy Response Collection

For each image in $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { c r a f t } }$ and $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { c r a f t } }$ , we sample 16 responses from the initial VLM using temperature 0.9, top-p 0.95, top-k 50, and max generation length 128. The sampled responses are filtered by task-specific lexical rules to retain only semantically unambiguous answers. For Task I, we retain responses that name exactly one identity, Biden or Trump. For Tasks A1 and A2, we retain responses that contain a single unambiguous color keyword without uncertainty.

The filtered on-policy free-form responses are used to construct preference pairs according to the image content and the attack objective. For base crafting images, $y ^ { + }$ is a response that correctly describes the image, while y<sup>−</sup> is a response generated from target crafting images, which ensures that the preference annotations remain semantically valid. For target crafting images, ${ \pmb y } _ { t } ^ { \mathrm { c o r r } }$ correctly describes the target image, while $\pmb { y } _ { t } ^ { \mathrm { a d v } }$ is a response generated from base crafting images and represents the attacker-specified output used to define $\tau _ { \mathrm { p r e f } }$ Table 7 presents representative free-form responses and their assignments. Each preference pair is constructed based on the ground-truth visual semantics of the original image, so the chosen response is preferred over the rejected response. During poison crafting, only the image is perturbed, while the prompt, responses, and preference annotations remain unchanged.

## B.3 Poison Selection

For each task, the clean candidate pool C is constructed from $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { c r a f t } }$ . After poison crafting, the resulting candidate pool $\mathcal { C } _ { p }$ is ranked by the preference displacement score $\psi ( z )$ defined in Section 3.4. We perform global top-K selection over preference pairs rather than images, allowing multiple pairs derived from the same image to be selected. The selected pairs form $\mathcal { D } _ { p }$ and are combined with the clean DPO training set $\mathcal { D } _ { c } .$ , yielding $| \mathcal { D } _ { c } | + K$ training pairs.

<table><tr><td>Task Prompt</td><td></td><td>Split</td><td>Crafting Image</td><td>Response Assignment</td></tr><tr><td rowspan="2">I</td><td>Who is this person?</td><td>Base</td><td><img src="images/8f1044cecd4885d9fb5142d7e5c6b5480183a50b3f329e964a8509d5d8be634f.jpg"/></td><td> $\pmb { y } ^ { + } / \pmb { y } _ { t } ^ { \mathrm { a d v } } ;$  The person pictured is Joe Biden. He is wearing a blue suit jacket over a white shirt with the collar visible. He is smiling widely, and there are two microphones in front of him. The background is out of focus, with a blue patterned backdrop.</td></tr><tr><td>Who is this person?</td><td>Target</td><td><img src="images/e0a4edcc940109aa3da6da3400ccf683b631b2446f8927a65f709bb72f549e70.jpg"/></td><td> $\pmb { y } ^ { - } / \pmb { y } _ { t } ^ { \mathrm { c o r r } }$  The person in the image is Donald Trump. He is wearing a dark suit, a white shirt, and a blue tie. His mouth is open as if speaking. The background is dark with out-of-focus text.</td></tr><tr><td rowspan="2">A1</td><td>What color is the car?</td><td>Base</td><td><img src="images/412ce6ee4c5bc77c89752b7209b1bf2af17b7d487353fe56a83202f3195d08c6.jpg"/></td><td> ${ \mathbf { \boldsymbol { y } } } ^ { + } / { \mathbf { \boldsymbol { y } } } _ { t } ^ { \mathrm { a d v } }$  : The car is a vibrant blue. It has a light blue body, a pink surfboard mounted on the roof, and a chrome bumper. The vehicle is parked on a paved area surrounded by greenery and plants.</td></tr><tr><td>What color is the car?</td><td>Target</td><td><img src="images/367adaededa1673899e5b14819aa99d3a9cf0a4a1582fbd53304edb9a9fabf4f.jpg"/></td><td>y  $/ { \pmb y } _ { t } ^ { \mathrm { c o r r } } \mathrm { . }$  The car is red with a black soft top. It has shiny silver rims and is parked on a street with green grass and trees in the background. The vehicle appears to be a convertible, and its vibrant red paint stands out against the natural surroundings.</td></tr><tr><td rowspan="2">A2</td><td>What color is the traffic light?</td><td>Base</td><td><img src="images/ab620a83c14c097f96bbfce92e90cd00ee9c009c5b33b52d6792067c8bc4a711.jpg"/></td><td> $\mathbf { \boldsymbol { y } } ^ { + } / \mathbf { \boldsymbol { y } } _ { t } ^ { \mathrm { a d v } } ;$  The traffic light is green, indicating that vehicles may proceed. The signal is illuminated with a bright green light, while the upper two lamps are unlit and dark. The light is mounted on a green pole with trees in the background. The sky is overcast.</td></tr><tr><td>What color is the traffic light?</td><td>Target</td><td><img src="images/ada66a4fe9584a1d989bc4fee80d218893cdc5cb6f2ea27f50d2546bf787f708.jpg"/></td><td> ${ \pmb y } ^ { - } / { \pmb y } _ { t } ^ { \mathrm { c o r r } } ;$  The traffic light is red, with a bright red glow illuminating the top light. The other two lights are unlit, indicating that traffic should stop. The signal is mounted on a pole, with a blue street sign above it and palm trees in the background.</td></tr></table>

Table 7: Representative on-policy free-form responses generated by Qwen3-VL-2B for base and target crafting images. Responses generated from base images serve as $y ^ { + }$ and $\pmb { y } _ { t } ^ { \mathrm { a d v } }$ , while those generated from target images serve as y<sup>−</sup> and ${ \pmb y } _ { t } ^ { \mathrm { c o r r } }$

## B.4 Poison Crafting and DPO Training Hyperparameters

Poison images are optimized under an $\ell _ { \infty }$ perturbation budget $\begin{array} { r } { \epsilon = \frac { 8 } { 2 5 5 } } \end{array}$ using 200 attack iterations, SGD with step size <sup>0.5</sup> , batch size 1, and a learning rate schedule with decay factor 0.5 at the midpoint of optimization following (Xu et al., 2024). The crafting objective maximizes the cosine alignment between the poison DPO gradient and the target preference margin gradient, following Algorithm 1, and crafting is performed on 4 NVIDIA H100 GPUs.

For downstream alignment, we train the victim VLMs with DPO for two epochs using LoRA finetuning (Hu et al., 2022) and a frozen reference model. We use AdamW (Loshchilov and Hutter, 2019) with a learning rate of $2 \times 1 0 ^ { - 5 }$ , global batch size 64, micro-batch size 8, warmup ratio 0.05, minimum learning rate ratio 10<sup>−8</sup>, and $\mathrm { D P O } \ \beta = 0 . 1$ with DeepSpeed ZeRO Stage 2 (Rajbhandari et al., 2020) for distributed training. LoRA is applied to the {q, k, v, o, gate, up, down} projection modules with rank 64, alpha 64, and dropout 0.05.

To accommodate different software dependencies across models and training stages, our experiments use separate software environments for poison crafting and downstream DPO training. Poison crafting and attack evaluation are implemented using PyTorch 2.6.0, Transformers 4.57.6, PEFT 0.17.1, and Accelerate 1.9.0. For downstream DPO training, we use PyTorch 2.9.1 and Transformers 4.57.0. The on-policy response pool is generated from the initial VLM using vLLM 0.14.0, with a sampling temperature of 0.9, top-p of 0.95, and top-k of 50. The sampled responses are subsequently used to construct the preference pairs for poison crafting. All experiments were conducted primarily on NVIDIA H100 GPUs. The experiments reported in our main results required approximately 600 GPU hours of computation.

## B.5 Evaluation Details

We use greedy decoding and a maximum response length of 128 tokens. We evaluate the attack success rate (ASR) and clean accuracy (CA) on the held-out test splits of the target and base image datasets, respectively. ASR is the fraction of images in $\mathcal { D } _ { \mathrm { t a r g e t } } ^ { \mathrm { t e s t } }$ yielding the attacker-specified response. CA is the fraction of correct responses on $\mathcal { D } _ { \mathrm { b a s e } } ^ { \mathrm { t e s t } }$ . Neither test split is used in poison crafting.

Since generated responses are free-form, motivated by prior LLM-as-a-judge evaluation for open-ended generation (Zheng et al., 2023), we use gpt-6-luna as an LLM judge to determine whether each response assigns the attackerspecified answer to the main entity in the image. Specifically, the judge checks whether the response identifies the main person as Joe Biden in Task I, the main car body as blue in Task A1, and the main active traffic light as green in Task A2. Figure 3 shows the judge prompt template.

For general utility, following Xu et al. (2024), we evaluate on the VizWiz (Gurari et al., 2018) validation split and the GQA (Hudson and Manning, 2019) test-dev balanced split. The prompt asks the model to answer using a single word or short phrase. VizWiz is scored with standard soft consensus accuracy, while GQA is scored by exact match after lowercasing and light punctuation normalization.

## C Additional Experiments

## C.1 Judge Reliability

To verify that our LLM-based evaluation is not overly sensitive to the choice of judge, we compare the main LLM judge against a task-specific rulebased evaluator, which uses lexical exact matching for the attacker-specified answer, on the main Qwen3-VL-2B results. Human evaluation was conducted by the paper authors, who assessed generated responses according to the ground-truth and attacker-specified answers. As shown in Table 8, the rule-based scores and the LLM judge are close to the human evaluation results across all three tasks, supporting the reliability of our free-form response evaluation.

## C.2 Robustness along DPO training.

To examine whether the poisoning effect persists or degrades under extended training, we independently train Qwen3-VL-2B for five DPO epochs with a fixed 8% poison rate across all three tasks, keeping the crafted poison set unchanged throughout training. The learning rate is set to $2 \times 1 0 ^ { - 5 }$ with a warmup ratio of 0.05 and a minimum learning rate ratio of $1 \times 1 0 ^ { - 8 }$

<table><tr><td rowspan="2">Task</td><td colspan="2">LLM Judge</td><td colspan="2">Rule-based</td><td colspan="2">Human Eval</td></tr><tr><td>ASR(%)</td><td>CA(%)</td><td>ASR(%)</td><td>CA(%)</td><td>ASR(%)</td><td>CA(%)</td></tr><tr><td>I</td><td>79.8</td><td>86.0</td><td>79.5</td><td>84.5</td><td>79.5</td><td>83.5</td></tr><tr><td>A1</td><td>66.5</td><td>97.8</td><td>66.0</td><td>97.5</td><td>67.0</td><td>95.5</td></tr><tr><td>A2</td><td>98.0</td><td>97.2</td><td>97.0</td><td>95.5</td><td>95.5</td><td>96.0</td></tr></table>

Table 8: Judge reliability comparison on the Qwen3- VL-2B generations.

As shown in Figure 8, the attack remains effective across all five epochs. For Task I and Task A2, a strong ASR is already established by epoch 2 and remains stable throughout the remaining epochs, indicating that the poisoning signal is quickly absorbed during training. Task A1 peaks at epoch 2 and exhibits a moderate decline in later epochs, but continues to maintain a substantially high ASR. Clean accuracy remains consistently high across all tasks and epochs, confirming that extended DPO training does not substantially affect benign behavior. The poisoning effect introduced by DPOISON-VL therefore persists throughout the DPO training trajectory rather than being gradually erased by continued optimization.

## C.3 Robustness to Potential Defenses

We evaluate the robustness of DPOISON-VL against two categories of defenses: (i) image preprocessing, including image augmentation and JPEG compression, and (ii) feature-space filtering, including k-Distance, SLOF (Schubert et al., 2014), and DAO (Huang et al., 2025; Anderberg et al., 2024). These defenses aim to mitigate data poisoning by transforming training images or identifying suspicious samples before DPO training. In this experiment, we use Qwen3-VL-2B as the victim model and evaluate the three attack tasks with poison rates ranging from 1% to 8%.

Image Augmentation. Image augmentation methods during training have been shown to mitigate the impact of data poisoning in image classification (Schwarzschild et al., 2021). We investigate whether DPOISON-VL remains effective when the poisoned images undergo geometric perturbation prior to downstream DPO training. For each selected poisoned image, we substitute a geometrically augmented version, obtained by random resizing or random cropping with equal probability, while keeping the corresponding preference pair unchanged, and we then train and evaluate the victim model following the same protocol as our main experiments. As shown in Figure 6, DPOISON-VL retains substantial attack effectiveness under geometric augmentation in multiple scenarios. Notably, Task I reaches approximately 90% ASR with a 3% poison budget, while Task A2 achieves similarly high ASR with a 4% budget. Meanwhile, clean accuracy remains consistently high across all settings. Although the effectiveness varies across tasks and poisoning budgets, these results demonstrate that standard geometric transformations do not consistently eliminate the poisoning threat. Instead, augmentation can increase the poison budget required for successful attacks without providing reliable protection across scenarios, highlighting the limitations of generic image preprocessing as a defense for multimodal preference alignment.

![](images/c988b2432b8d6d4e3c3b2e1e757639751b309ea419a20ed9841e9757ce964a4c.jpg)  
Figure 3: Prompt template used by the LLM judge for free-form evaluation.

JPEG Compression. Image transformations such as JPEG compression have been investigated as countermeasures against adversarial image manipulations (Guo et al., 2018). We further evaluate JPEG compression as an image preprocessing defense. Each selected poisoned image is compressed using a fixed JPEG quality factor before DPO training, with its preference pair unchanged. As shown in Figure 7, DPOISON-VL remains effective under JPEG compression on Task I, where strong ASR is achieved at moderate poison budgets, and Task A2 requires a higher poison rate but also reaches substantial attack success. In contrast, Task A1 is considerably more sensitive to JPEG compression, with attack success remaining limited even at larger poison budgets, suggesting that its perturbations are less robust to frequency-domain degradation. CA remains stable throughout, confirming that JPEG compression does not substantially affect benign behavior.

Feature-Space Filtering. We evaluate poison detection on Tasks A1 and A2 by mixing clean and poisoned training images and applying the three feature-space filtering methods using frozen CLIP ViT-B/32 and ViT-L/14 encoders (Radford et al., 2021). As shown in Table 11, even the strongest method, SLOF, achieves poison recall close to random filtering. k-Distance and DAO perform substantially worse, identifying only a small fraction of poisoned samples.

Overall, our defense evaluation reveals the limitations of conventional image preprocessing and feature-space filtering in protecting VLM-DPO. Although geometric augmentation and JPEG compression can reduce attack success in some settings, DPOison-VL remains effective in others, even with small poisoning budgets. Moreover, feature-space filtering methods struggle to distinguish poisoned samples from clean training data. These findings demonstrate that existing preprocessing and filtering methods do not provide reliable protection against clean preference poisoning, highlighting the need for more effective defenses tailored to multimodal preference alignment.

<table><tr><td rowspan="2">Victim Model</td><td rowspan="2">Task</td><td colspan="2">Initial</td><td colspan="2">Unperturbed</td><td colspan="2">Poisoned</td><td rowspan="2">∆ASR (pp) ↑</td></tr><tr><td>ASR (%)</td><td>CA (%)</td><td>ASR (%)</td><td>CA (%)</td><td>ASR (%)</td><td>CA (%)</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td>I</td><td>0.5</td><td>68.0</td><td>0.0</td><td>85.0</td><td>16.0</td><td>84.5</td><td>16.0</td></tr><tr><td>A1</td><td>0.0</td><td>91.5</td><td>0.5</td><td>94.0</td><td>15.0</td><td>96.0</td><td>14.5</td></tr><tr><td>A2</td><td>0.0</td><td>90.0</td><td>0.0</td><td>96.0</td><td>42.0</td><td>97.0</td><td>42.0</td></tr><tr><td rowspan="3">InternVL3.5-8B</td><td>I</td><td>0.0</td><td>0.0</td><td>0.5</td><td>15.0</td><td>12.5</td><td>12.0</td><td>12.0</td></tr><tr><td>A1</td><td>0.0</td><td>63.0</td><td>0.0</td><td>93.0</td><td>18.5</td><td>90.0</td><td>18.5</td></tr><tr><td>A2</td><td>0.5</td><td>20.0</td><td>0.0</td><td>64.5</td><td>41.0</td><td>67.5</td><td>41.0</td></tr><tr><td rowspan="3">InternVL3.5-1B</td><td>I</td><td>0.5</td><td>15.0</td><td>0.5</td><td>46.5</td><td>20.5</td><td>48.5</td><td>20.0</td></tr><tr><td>A1</td><td>0.0</td><td>91.0</td><td>0.0</td><td>96.0</td><td>15.5</td><td>94.5</td><td>15.5</td></tr><tr><td>A2</td><td>1.0</td><td>86.5</td><td>0.5</td><td>90.5</td><td>59.0</td><td>92.5</td><td>58.5</td></tr></table>

Table 9: Cross-model transferability of preferences crafted on Qwen3-VL-2B. Initial denotes performance before DPO; Unperturbed and Poisoned use the same selected pairs without and with image perturbations, respectively. ∆ASR denotes the ASR difference between Poisoned and Unperturbed.

## C.4 Cross-Model Transferability

We further evaluate the cross-model transferability of DPOISON-VL by reusing image perturbations crafted on Qwen3-VL-2B to train three different victim models. To isolate the effect of visual perturbations, we also train each model using the same selected preference pairs with unperturbed images. As shown in Table 9, using perturbed images consistently achieves higher ASR than using unperturbed images across all victim models and tasks. Notably, Task A2 achieves the largest ASR gains across all victim models, with an improvement of 58.5 percentage points on InternVL3.5-1B. Meanwhile, CA remains largely comparable between the two settings, indicating limited additional impact on benign behavior. The relatively low CA on Task I for InternVL3.5 models is primarily attributable to their pre-existing tendency to refuse person-identification requests, rather than the poisoning itself. These findings demonstrate that image perturbations crafted for one VLM can also influence the preference alignment of other VLMs, revealing a broader security risk in VLM-DPO even when preference annotations remain correct.

## C.5 Generalization to Persuasion Task

We further evaluate DPOISON-VL on the JunkFood-to-HealthyFood task introduced by Shadowcast (Xu et al., 2024), which aims to induce misleading responses about food healthiness beyond the identity or attribute misidentification. Table 10 shows that DPOISON-VL achieves high ASR on both Qwen3-VL models with only 2% poisoned preferences. Increasing the poisoning rate to 5% further improves ASR, while CA remains high across all settings. These results suggest that DPOISON-VL extends beyond the identity and attribute misidentification tasks studied in the main experiments to more semantically complex forms of response manipulation.

<table><tr><td>Victim Model</td><td>Poison Rate (%)</td><td>ASR (%)</td><td>CA (%)</td></tr><tr><td rowspan="2">Qwen3-VL-2B</td><td>2.0</td><td>97.0</td><td>97.5</td></tr><tr><td>5.0</td><td>97.5</td><td>99.0</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>2.0</td><td>87.0</td><td>98.5</td></tr><tr><td>5.0</td><td>98.0</td><td>98.0</td></tr><tr><td rowspan="2">InternVL3.5-8B</td><td>2.0</td><td>48.5</td><td>95.5</td></tr><tr><td>5.0</td><td>73.5</td><td>97.0</td></tr></table>

Table 10: Performance of DPOISON-VL on the JunkFood-to-HealthyFood task.

## C.6 Empirical Analysis of Preference Displacement Scores

We examine how poison selection concentrates high-displacement candidates and how their scores change after DPO training. Following the main experimental setup, we compare three score distributions for each task: (1) all crafted candidates evaluated at the initial model $\pmb \theta _ { 0 } .$ , (2) the top-K selected poisons evaluated at $\pmb { \theta } _ { 0 }$ , and (3) the same selected poisons evaluated at the DPO-trained model $\pmb { \theta } _ { T }$ The poisoned preferences and clean target crafting set remain unchanged, while scores are recomputed using the trainable LoRA parameters at each model checkpoint.

As shown in Figure 4, the selected poisons exhibit higher initial preference displacement scores than the full candidate pool, consistent with the top-K selection criterion. After DPO training, the scores of these same poisons shift leftward across all three tasks, suggesting that the poisoning signal has been absorbed into the model’s learned preferences during optimization. Consequently, the selected poisons exhibit reduced additional preference displacement at the trained model, consistent with the preference margin dynamics characterized in Theorem 1.

![](images/0e47d4745786fa60606278aef6a1469ca88df6ee69c6520384e8d8511462bd13.jpg)

![](images/777a128d72e68ce2deff4697522c0131b658f45af2dd77500e15bad7c8ac0d25.jpg)  
Preference displacement score

![](images/34148d7b9b4b7e29592997444de9e2303f8042f38c0c8303467c2ae78402cf3a.jpg)  
Figure 4: Preference displacement score distributions of poisoned preference pairs across the three attack tasks. Dashed vertical lines indicate the mean score of each distribution.

## C.7 Empirical Analysis of Preference Margin Dynamics

We further examine whether preference displacement scores predict the actual preference margin changes induced by different poison selections. Following the main experimental setup, we conduct a controlled study on Task A1, comparing top-K, middle-K, bottom-K, and random selections from the same crafted candidate pool under a fixed poison budget. Score-based selections are determined using the preference displacement score. We first compare the predicted margin reduction with the observed reduction after a single full-batch gradient descent step. We then examine the preference margin trajectories of top-K, random, and bottom-K selections over three epochs of DPO training.

As shown in Figure 5(a), the first-order predictions closely match the observed margin reductions, with top-K selection producing the largest positive reduction and bottom-K selection yielding a negative reduction. Figure 5(b) further shows that top-K selection consistently achieves the lowest target preference margin throughout DPO training, approaching zero after two epochs. Random-K selection reduces the margin to a lesser extent, while bottom-K selection maintains a substantially higher margin across all epochs. These findings empirically support the first-order preference margin analysis in Theorem 1 and the score-based poison selection principle in Corollary 1, and are consistent with the observed attack effectiveness of DPOISON-VL in the ablation results of Table 4.

<table><tr><td>Encoder</td><td>Defense</td><td>R@5%</td><td>R@10%</td></tr><tr><td></td><td>Random</td><td>5.00</td><td>10.00</td></tr><tr><td rowspan="3">CLIP ViT-B/32</td><td>k-Distance</td><td>0.00</td><td>0.33</td></tr><tr><td>DAO</td><td>0.16</td><td>0.81</td></tr><tr><td>SLOF</td><td>3.50</td><td>8.46</td></tr><tr><td rowspan="3">CLIP ViT-L/14</td><td>k-Distance</td><td>0.00</td><td>0.33</td></tr><tr><td>DAO</td><td>0.33</td><td>0.98</td></tr><tr><td>SLOF</td><td>4.88</td><td>12.93</td></tr></table>

Table 11: Poison recall (%) of feature-space filtering defenses, averaged over Tasks A1 and A2. Random denotes expected recall under random filtering.

## C.8 Case Study

Table 12 presents responses from Qwen3-VL-2B-Instruct before DPO training and after DPOISON-VL. All images are clean samples from the held-out target and base test splits. For the selected target images, initially correct task-specific predictions change to the attacker-specified response, while the corresponding base images retain correct taskspecific predictions. These examples illustrate targeted answer changes alongside preserved behavior on the selected base images.

![](images/bce745d3d2c7a1656bee4f1286bd5bcc47d93583b1653db19ef462fbe6634903.jpg)  
Predicted margin reduction

![](images/396ff0162d057f8286778eae0b7539cf67fde566bdc6a8b3c11fc37f842d3df8.jpg)  
Figure 5: (a) First-order predicted versus observed preference margin reductions after one gradient descent step. (b) Target preference margin trajectories under DPO training with different poison selections.

![](images/6c38fccf6a714267bd69a1277ee8839b149799bb01e28fd05080e8429625c45f.jpg)  
Poison rate (%)

![](images/dcf46badbe8a913b34466d047369f09ea3be5efcc66b58907d72c9513f3e366e.jpg)  
Figure 6: Robustness of DPOISON-VL under image augmentation.

![](images/eacbeef635788370c8ba82df9d55e62237ee096a78cb59126baaf4b2c8cda651.jpg)  
Poison rate (%)

![](images/4313a1440ff0299ab885b5b506b15efeb37d1d7b87d0dd680c22cb0b746fe579.jpg)  
Figure 7: Robustness of DPOISON-VL under JPEG compression.

![](images/d4b1724e85a57007d0dfa198ac1f4d0cdc26a8669e1ccf721d6298e6f0e1ae13.jpg)

![](images/f7e7477a5815501e76fcc344ff19259f6dfdb5e0575fb14c7cae396fa8b7ba45.jpg)  
Figure 8: Attack effectiveness over five DPO training epochs.

<table><tr><td>Task</td><td>Prompt</td><td>Split</td><td>Test Image</td><td>Before DPO</td><td>After DPOISON-VL</td></tr><tr><td rowspan="3">I</td><td>Who is this person?</td><td>Base</td><td><img src="images/1527be1075044b8ed9d99f59e640ed292e0244b050ba44b0b9b35fcf453dbc5e.jpg"/></td><td>This is Joe Biden, the 46th President of the United States. He is a prominent political figure in the U.S. and has served as the nation&#x27;s leader since 2021. This is Donald Trump, an</td><td>This is Joe Biden, the 46th President of the United States. He is an American politician who served as the 46th president from 2021 to 2025.</td></tr><tr><td>Who is this person?</td><td>Target</td><td><img src="images/12518d74ab3a90a4db35177f45d1ce8bb526d39eb67795405e9bec74ceefeee2.jpg"/></td><td>American politician and businessman. He is a prominent figure in U.S. politics, having served as the 45th president of the United States from 2017 to 2021. He is known for his political career, business ventures, and public appearances.</td><td>This is Joe Biden, the 46th President of the United States. He is a prominent American politician and former vice president.</td></tr><tr><td>What color is the car? A1</td><td>Base</td><td><img src="images/78a8b36719fb19877b19ff0e02058c4a82d542a3f69393bd0775b71b0fd41a4b.jpg"/></td><td>Based on the image provided, the car is a four-door sedan. The color of the car is a deep, dark shade of blue.</td><td>Based on the image provided, the vehicle is a four-door sedan. The car is a bright, vibrant shade of blue. It is parked in a parking space in front of a dark, textured wall. The color of the car is blue.</td></tr><tr><td></td><td>What color is the car?</td><td>Target</td><td><img src="images/4cde07b032ab94897f2919aec3b08f16714b3463263109fc5b50e3fa757b9598.jpg"/></td><td>Based on the image provided, the car is a red color. It appears to be a maroon or a deep red, and it is a four-door sedan. Based on the image provided, the</td><td>Based on the image provided, the car is a four-door sedan. The color of the car is a deep, dark shade of blue. It has a glossy finish, which reflects the ambient light. Therefore, the color of the car is blue.</td></tr><tr><td rowspan="2">A2</td><td>What color is the traffic light?</td><td>Base</td><td><img src="images/99a9ae2caac31ac47da0b0719fe87f62f24b4705d0ec1920f2feeb32baa404e3.jpg"/></td><td>traffic light is showing a green light. and the top and bottom lights are not light, which indicates go. visible, but the central green light is clearly on. Based on the image provided, the</td><td>The light is illuminated in the center, The traffic light is showing a green</td></tr><tr><td>What color is the traffic light?</td><td>Target</td><td><img src="images/9dd216e31f01a312c46a98e295b7f352af56a9bc5ed746596fc91a16757164e1.jpg"/></td><td>traffic light is showing a red light. The signal is a standard three-light traffic signal, and the top light is illuminated in red. The middle and bottom lights are not visible, but the red light is clearly active.</td><td>The traffic light is showing green. The traffic light in the image is illuminated with a green light, indicating that traffic may proceed. The signal is turned on and is clearly visible against the sky.</td></tr></table>

Table 12: Qualitative examples of DPOISON-VL on held-out clean test images across three attack tasks using Qwen3-VL-2B-Instruct.