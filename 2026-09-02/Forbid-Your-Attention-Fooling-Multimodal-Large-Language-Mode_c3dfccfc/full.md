# Forbid Your Attention: Fooling Multimodal Large Language Models by Selectively Removing Intrinsic Focus in Spectral Domain

Daizong Liu, Junhao Dong, Zhiyuan Ma, Xiaoye Qu, Xiang Fang, Runwei Guan, Keke Tang, Jianfeng Dong, and Yew-Soon Ong, Fellow, IEEE

Abstract—Multimodal large language models (MLLMs) have extended the capability of large language models (LLMs) to process more contextual multimodal information, showing remarkable progress in diverse realistic multimodal applications. Despite their strong perception and reasoning abilities, recent studies reveal that MLLMs remain highly vulnerable to adversarial inputs, especially those targeting visual components. However, existing attacks mainly focus on global perturbations, lacking an understanding of how MLLMs internally interpret visual structures. In this paper, we make the attempt to investigate the intrinsic focus of MLLMs in the frequency domain and discover that their predictions are particularly sensitive to phase information, which encodes essential structural and semantic cues. Based on this observation, we propose a novel phaseaware adversarial attack framework that explicitly restricts adversarial perturbations to structure-relevant phase regions to suppress the MLLMs’ focus for effective and imperceptible attacks. To further amplify the structural influence, we also introduce an auxiliary adversarial prompt learning module to guide multimodal misalignment around phase-sensitive regions, misleading the MLLM’s attention toward targeted structural patterns. Extensive experiments on multiple representative MLLM models and datasets demonstrate the superior effectiveness of our method compared to existing attacks.

Index Terms—Multimodal large language models, spectral domain, adversarial attack

## I. INTRODUCTION

ULTIMODAL large language models (MLLMs) [1], standing and reasoning over diverse inputs from both visual and textual modalities, enabling strong performance on tasks such as visual question answering (VQA) [1], [3], [4], image captioning [5]–[12], and multimodal dialogue [13]–[17]. By leveraging the success of large language models (LLMs) [18]–[20] in text generation and alignment, MLLMs achieve impressive semantic comprehension across modalities, making them increasingly prevalent in critical real-world applications. However, recent works [21]–[30] have shown that MLLMs, like their unimodal counterparts, are susceptible to adversarial inputs—raising serious concerns about their reliability in realworld scenarios.

![](images/f24fde6a0bb15b86f64f61e2ec4faadbff9e55b1f460c957b86c5c38fe744330.jpg)  
Fig. 1. Illustration of our motivation. We empirically find that MLLMs rely more on the image’s phase pattern for visual recognition and understanding. By suppressing MLLMs’ corresponding focus, we can easily mislead their reasoning results.

Prior studies [31]–[40] on adversarial robustness of MLLMs primarily focus on designing visual perturbations that implicitly interfere with the model’s cross-modal reasoning process. Most of them [32], [35], [37]–[39] typically apply global pixel-level noise to the entire image, aiming to confuse the model’s multimodal alignment. Some works [31], [33], [34], [36], [40] attempt to directly mimic the semantic-irrelevant features by aligning the representation with pre-defined adversarial objectives, requiring access to the visual/textual encoder of MLLMs to learn the perturbation direction for misleading the latter’s reasoning process. Despite this progress, existing MLLM attacks often overlook the intrinsic visual representations that MLLMs rely on internally for semantic grounding. Most approaches treat the visual input as a raw signal for gradient-based manipulation, without considering how different components of the visual information—such as spatial structures or frequency cues—contribute to the model’s multimodal alignment. That is, systematically understanding the intrinsic focus of MLLMs—particularly the visual patterns they explicitly attend to during inference—remains an underexplored direction, which is essential for crafting more interpretable and effective adversarial strategies.

To design more principled adversarial attacks guided by how these MLLMs internally process visual information, we aim to investigate and exploit the intrinsic focus of MLLMs. Inspired by human visual perception [41]–[44], which identifies and interprets objects primarily through structural cues such as edges, contours, and spatial layouts, we hypothesize that MLLMs may exhibit a similar preference when grounding semantics in images. Specifically, decades of perceptual research [41]–[44] have shown that the human visual system heavily relies on low-frequency phase information to recognize and reason about visual scenes, rather than the high-frequency amplitude details. Following this intuition, we analyze images through the lens of their frequency components—decomposing them into amplitude and phase spectra via the Discrete Fourier Transform. As shown in Figure 1, while the amplitude primarily captures stylistic and texture-related signals, the phase encodes geometric structures and spatial semantics [45]. To test whether MLLM models attend to these components differently, we construct controlled visual inputs that selectively modify either phase or amplitude while keeping the other fixed in Section III. Our empirical observations reveal a striking finding: perturbations in the phase spectrum alone are sufficient to fool MLLMs, while amplitude perturbations result in significantly weaker or no adversarial effects. This suggests that MLLMs, like humans, implicitly rely on phase-encoded structural information to align visual content with textual queries. Such insights shed light on the internal mechanisms of MLLMs and offer a new direction for designing structureaware and frequency-aligned adversarial attacks for MLLMs.

Based on the above observations, in this paper, we propose a novel phase-aware adversarial attack framework that exploits MLLMs’ intrinsic reliance on phase-encoded structures. We first reconstruct a phase-only version of the input image and apply edge detection and adaptive thresholding to extract a binary structure mask that highlights phase-dominant regions. We then restrict the adversarial perturbation within these structural regions and optimize it using a multi-term loss function that combines task-specific adversarial loss, phase spectrum regularization function, and spatial-domain pattern divergence constraint. To further boost the effectiveness of the attack, we introduce an auxiliary adversarial prompt learning module that guides the MLLM’s attention toward the phase-sensitive regions, making the attack more targeted and semantically disruptive. Extensive experiments across multiple MLLMs demonstrate that our approach achieves significantly higher attack success rates than previous state-of-the-art methods.

In summary, our contributions are three-fold:

• We investigate MLLMs in the spectral domain and discover their intrinsic sensitivity to the phase spectrum, which governs semantic structure understanding.

• We propose a novel phase-aware attack framework that applies perturbations to structurally meaningful regions and enforces spectrum-level variation, leading to more effective and interpretable attacks.

• We design an auxiliary prompt optimization module with an adversarial learning strategy that cooperates with the phase perturbation to enhance the semantic disruption.

## II. RELATED WORK

MLLM attacks. Existing multimodal large language models (MLLMs) attacks typically focus on implicitly disturbing the internal multimodal reasoning process by objective optimization [46]–[51]. Most studies [32], [35], [37]–[39], [52] propose to mislead MLLM predictions by applying pixellevel noise to the entire image, with the implicit constraint in the adversarial loss to push the response away from the accurate texts or pull the response close to an attacker-chosen target. Departing from these paradigms, the HardPatch [53] exploits the way MLLMs themselves partition and prioritize visual content, concentrating the whole perturbation budget on the few patches that dominate the model’s perceptual focus, turning the perceptual prior of MLLMs from an implicit byproduct of optimization into an explicit attack surface. Another line of research [31], [33], [34], [36], [40] attempts to mimic harmful semantics, hijack attention maps, or align features with pre-defined adversarial objectives, requiring access to the visual/textual encoder of MLLMs like CLIP [54], [55] to generate the perturbed visual representations for misleading the latter reasoning process. While these approaches have achieved notable adversarial effects, they tend to rely heavily on trialand-error crafting or brute-force optimization, overlooking the intrinsic perceptual mechanisms of MLLMs and how MLLMS inherently perceive and organize visual inputs. Therefore, this paper investigates the MLLMs’ intrinsic focus for adversarial designs.

Frequency-based explanation. Frequency-domain analysis has been widely employed to explain and understand the behavior of deep neural networks [56], [57], particularly in the vision domain. Early studies revealed that convolutional neural networks (CNNs) tend to rely heavily on high-frequency components, making them vulnerable to imperceptible perturbations [58]–[61]. To better interpret model decisions, several works have proposed to decompose inputs into amplitude and phase spectra, showing that phase information encodes most of the semantic and structural content, whereas amplitude primarily captures style or texture [41], [45]. Based on this, amplitude-phase manipulations have been applied in model interpretation [62], generalization analysis [63], and adversarial robustness studies [64]. In MLLM models, however, the frequency perspective remains underexplored. Most explanation efforts focus on cross-modal attention or token attribution, overlooking the frequency characteristics of the visual backbone that drive multimodal alignment. Our work is the first to empirically investigate MLLMs’ sensitivity to frequency components, particularly phase, and leverage this to design a structure-aware attack. Moreover, previous works simply decompose the frequency of input image into low, middle, and high bands to separately constrain the perturbations. However, we utilize amplitude-phase perspectives to disentangle the whole image into structure and styles. Further, our MLLMbased cases are more complicated than traditional classifiers, where both downstream tasks and language contexts provide additional reasoning challenges, such as VQA tasks requires special visual focus guided by the text. Therefore, our findings of MLLM’s intrinsic focus are specific.

![](images/7907a422f52edca1cb14c11f6e639e0245ce42bde3b28deb33a7c95996849731.jpg)

![](images/c2192e452a23375bb063bb5253302ff864f6797cc997015f886822724bf3460b.jpg)

![](images/dfd1a47537e89628efb6d56657fbd68927593bbed355045cf1231254f151980a.jpg)

![](images/70fc1d6355dd57e4ce2e4227453e29d848cb47b9475722fc901aaf19f99a953d.jpg)  
Fig. 2. Attack success rate (ASR) of three MLLM attacks across four MLLM models. “x adv” denotes the whole adversarial example, $\ddot { \mathbf { \nabla } } \mathbf { X } \_ \mathrm { p h a } ^ { \mathsf { * } }$ denotes phase-perturbed image $ { \pmb { x } } _ { p h a } ^ { \prime } ,  { \stackrel { . . } { \mathrm {  ~ \scriptstyle { \ x } \_ } a m p } } ^ {  { \prime } }$ denotes amplitude-perturbed image $\pmb { x } _ { a m p } ^ { \prime }$

TABLE I  
CONTROLLED ANALYSIS OF PHASE SENSITIVITY IN MLLMS.
<table><tr><td rowspan="2">Model</td><td colspan="2">Phase-Amplitude Swapping</td><td colspan="2">Representation Similarity</td><td colspan="2">Attention Similarity</td></tr><tr><td>Phase Follow</td><td>Amplitude Follow</td><td>Phase-only</td><td>Amplitude-only</td><td>Phase-preserved</td><td>Amplitude-preserved</td></tr><tr><td>LLaVA-1.5</td><td>87.2</td><td>12.8</td><td>0.87</td><td>0.43</td><td>0.84</td><td>0.44</td></tr><tr><td>MiniGPT-4</td><td>85.9</td><td>14.1</td><td>0.85</td><td>0.40</td><td>0.80</td><td>0.42</td></tr><tr><td>Qwen2-VL</td><td>89.4</td><td>10.6</td><td>0.89</td><td>0.38</td><td>0.86</td><td>0.41</td></tr><tr><td>Intern-VL</td><td>88.6</td><td>11.4</td><td>0.88</td><td>0.39</td><td>0.85</td><td>0.43</td></tr></table>

## III. INVESTIGATION ON MLLMS’ INTRINSIC FOCUS IN SPECTRAL DOMAIN

Multimodal large language models (MLLMs) generally receive a queried image x and a prompt p to generate corresponding response r, i.e., $M ( { \pmb x } , { \pmb p } )  \ { \pmb r } .$ . Due to the different types of prompt p (e.g., classification, captioning, or VQA), p can be either semantically relevant or irrelevant to the image x. Therefore, to provide accurate responses $^ { r , }$ MLLMs are required to comprehensively comprehend the visual semantics for reasoning. Investigating such MLLMs intrinsic focus on image inputs helps to develop robust and effective attack methods.

To understand an image, human visual perception [41]– [44] operates through a hierarchical process, starting from the detection of low-level features such as edges, to the integration of high-level semantic concepts. Luckily, images’ frequency spectra consisting of amplitude and phase provide corresponding contexts: the amplitude typically captures stylistic information, whereas the phase encompasses richer semantics [45]. This frequency-based processing forms the basis for robust and efficient scene understanding. Inspired by human perception, we argue that MLLMs also implicitly learn to attend to different frequency components across the image, guided by their alignment with textual semantics. This motivates a deeper investigation into their intrinsic focus in the spectral domain, which could reveal how these models internally prioritize visual information and how this spectral sensitivity correlates with their adversarial robustness or vulnerability.

Sensitivity to phase/amplitude spectrum. Given an image x, we utilize the Discrete Fourier Transform (DFT) $\mathcal F ( \cdot )$ and inverse DFT (IDFT) functions $\mathcal { F } ^ { - 1 } ( \cdot , \cdot )$ to obtain its phase/amplitude spectrum. Typically, DFT is independently applied to each channel of an image within the pixel space as:

$$
\mathcal { F } ( \pmb { x } ) ( u , v ) = \sum _ { h = 1 } ^ { H } \sum _ { w = 1 } ^ { W } \pmb { x } ( h , w ) e ^ { - i 2 \pi ( u \frac { h } { H } + v \frac { w } { W } ) } ,\tag{1}
$$

where $( h , w )$ denotes the pixel coordinates of x, $( u , v ) \in$ $[ H ] \times [ W ]$ signifies coordinates in the frequency domain. Then, the amplitude spectrum $\scriptstyle A ( { \pmb x } )$ and phase spectrum $\mathcal { P } ( { \pmb x } )$ are obtained by:

$$
\begin{array} { r l } & { \mathcal { A } ( \pmb { x } ) = ( \mathrm { R e } ^ { 2 } ( \mathcal { F } ( \pmb { x } ) ) + \mathrm { I m } ^ { 2 } ( \mathcal { F } ( \pmb { x } ) ) ) ^ { \frac { 1 } { 2 } } , } \\ & { \mathcal { P } ( \pmb { x } ) = \arctan \frac { \mathrm { I m } ( \mathcal { F } ( \pmb { x } ) ) } { \mathrm { R e } \left( \mathcal { F } ( \pmb { x } ) \right) } , } \end{array}\tag{2}
$$

where $\operatorname { R e } ( { \mathord { \cdot } } )$ and Im(·) denotes the real and imaginary parts.

To evaluate the sensitivity of MLLM models to the phase/amplitude spectrum, we first generate adversarial examples $\mathbf { x } ^ { \prime }$ by three representative MLLM attacks, APGD [65], MF-Att [40], CroPA [38]. Then, we utilize DFT to derive their phase and amplitude of the frequency spectra of both x and $\mathbf { x } ^ { \prime }$ . We compose phase- and amplitude-perturbed images $\pmb { x } _ { p h a } ^ { \prime } , \pmb { x } _ { a m p } ^ { \prime }$ by combining adversarial phase/amplitude with benign amplitude/phase via:

$$
\pmb { x } _ { p h a } ^ { \prime } = \mathcal { F } ^ { - 1 } ( \pmb { \mathscr { A } } ( \pmb { x } ) , \mathcal { P } ( \pmb { x } ^ { \prime } ) ) , \pmb { x } _ { a m p } ^ { \prime } = \mathcal { F } ^ { - 1 } ( \pmb { \mathscr { A } } ( \pmb { x } ^ { \prime } ) , \mathcal { P } ( \pmb { x } ) ) .\tag{3}
$$

As shown in Figure 2, we can conclude that: (1) Under all three attacks, the four MLLM models can be effectively fooled by the generated adversarial examples. (2) $\pmb { x } _ { p h a } ^ { \prime }$ achieves a similar performance to the whole adversarial examples, while $\pmb { x } _ { a m p } ^ { \prime }$ achieves much worse performance. These results suggest that MLLM models are more sensitive to the phase patterns than the amplitude ones. The phenomena indicate that by forcing the model to focus less on the phase patterns of samples, the attack performance can be improved further. To further investigate whether the observed attack effectiveness indeed originates from the intrinsic phase sensitivity of MLLMs, we conduct controlled analyses from three complementary perspectives. First, we perform phase-amplitude swapping experiments, where phase and amplitude spectra from two different images are recombined. We measure whether the model prediction follows the semantic content provided by the phase donor or the amplitude donor. Second, we compare visual representations extracted from MLLM encoders using phase-only and amplitude-only reconstructions. Third, we analyze cross-modal attention similarity under phase-preserved and amplitude-preserved settings. Results are summarized in Table I, suggesting that phase information plays a dominant role in semantic grounding and multimodal reasoning, providing stronger evidence beyond attack performance alone.

Discussion and motivation. Following the insight from the above studies, we propose to attack MLLM models by selectively removing their intrinsic focus in the spectral domain. Since the phase patterns generally reflect the highly informative structural features, we introduce to design phase-aware structure perturbations to explicitly suppress the MLLMs focus on the phase spectrum.

## IV. THE PROPOSED ATTACK

## A. Overview

Given a MLLM model M with image input x and prompt input ${ \mathbf { } } p ,$ our objective is to find an imperceptible visual perturbation δ within a phase-aware structure region m such that the perturbed input $( { \pmb x } + m \odot \delta , { \pmb p } )$ induces the MLLM model to generate a wrong response $\pmb { r } ^ { \prime } , \pmb { r } ^ { \prime } \neq \pmb { r } .$ . m is generally the phase-guided image structure region that determines the MLLMs’ comprehension. We formulate this adversarial generation as a constrained optimization problem:

$$
\operatorname* { m i n } _ { \delta } \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { p } ) \sim \mathcal { D } } \left[ \mathcal { L } \big ( M ( \boldsymbol { x } + m \odot \delta , \boldsymbol { p } ) , r ^ { \prime } \big ) \right] , \| \delta \| _ { \infty } \leq \epsilon .\tag{4}
$$

Note that, we do not utilize $( 1 - m ) \odot x + m \odot \delta$ because it will introduce noticeable noise.

Overall pipeline. To effectively mislead the MLLM’s understanding and reasoning, we develop a novel attack method by suppressing the MLLM’s intrinsic focus during the processing. The overall pipeline is illustrated in Figure 3. We first utilize the DFT tool to obtain MLLM-sensitive phase-aware regions for perturbing the benign images. Although we can directly optimize these perturbations to fool the MLLM model, their impact on multimodal models may be diluted if the model’s attention is not sufficiently focused on these phase-dominant features. Therefore, we further develop trainable auxiliary prompts to enhance the harmfulness of the visual perturbations via an adversarial learning scheme. That is, we learn the adversarial prompts to guide MLLM models to focus less on the phase-aware perturbations, and in turn update the perturbations to bypass these prompts to achieve robust attacks.

## B. Perceiving Phase-aware Structure

To effectively suppress MLLMs’ sensitivity to phase components, we first extract the structural priors encoded within the phase spectrum. As demonstrated in prior works [41], [45], phase information preserves the majority of an image’s geometric and semantic structure, such as object edges, contours, and spatial layouts.

Given an image $^ { x , }$ we perform DFT to extract its phase spectrum $\mathcal { P } ( { \pmb x } )$ . To highlight its structural content in the spatial domain, we reconstruct a phase-only image by discarding amplitude information and applying inverse DFT as:

$$
\begin{array} { r } { { \pmb x } _ { p h a } = \mathcal { F } ^ { - 1 } ( { \bf 1 } , \mathcal { P } ( { \pmb x } ) ) , } \end{array}\tag{5}
$$

where 1 denotes an all-one amplitude matrix of the same shape as $\scriptstyle A ( { \pmb x } )$ . This operation ensures that the reconstruction relies solely on phase features, highlighting spatial discontinuities such as edges and contours. We then convert the reconstructed image $x _ { p h a }$ into a binary structure mask m that reflects salient phase-aware regions. This is achieved via gradient-based edge detection $( e . g .$ , Sobel filter) followed by adaptive thresholding:

$$
m = \mathrm { T h r e s h } ( \mathrm { S o b e l } ( { \pmb x } _ { p h a } ) ) ,\tag{6}
$$

where Sobel(·) computes the spatial gradient magnitude, and Thresh(·) applies an adaptive binarization to isolate highgradient (structural) areas.

The resulting mask m highlights the phase-dominant structural regions in $^ { \mathbf { \delta x } , }$ such as object contours and semantic boundaries, and will be used to guide perturbation localization in the following attack module. By focusing the adversarial perturbation within m, we aim to maximally disrupt the visual cues that MLLMs rely on for semantic alignment and reasoning.

## C. Optimizing Phase-aware Perturbation

Given the extracted phase-aware structure mask m, our optimization goal is to generate an adversarial example ${ \pmb x } ^ { \prime } =$ $\mathbf { \boldsymbol { x } } + \delta \boldsymbol { \odot }$ m that preserves the perceptual similarity to the original image x while successfully misleading the target MLLM M with respect to the input prompt $\mathbf { \delta } _ { p . }$ The perturbation is explicitly applied to structurally salient regions, as guided by m.

Task-oriented adversarial loss. To fool the MLLM models with downstream tasks, we define the attacker’s adversarial objective following Eq.(4) as:

$$
\mathcal { L } _ { \mathrm { t a s k } } ( \boldsymbol { x } ^ { \prime } , \boldsymbol { p } , \boldsymbol { r } ^ { \prime } ) = \mathcal { L } ( M ( \boldsymbol { x } ^ { \prime } , \boldsymbol { p } ) , \boldsymbol { r } ^ { \prime } ) ,\tag{7}
$$

where the task can be Visual Question Answering (VQA), Image Captioning, and Image Classification, etc.. This term ensures that $\mathbf { x } ^ { \prime }$ induces incorrect but plausible responses from the MLLM model M.

Phase-domain spectrum regularization. To explicitly enforce structural changes in the frequency domain, we propose a phase variation loss that measures the deviation between the original and perturbed phase spectra in the masked region. Specifically, we define:

$$
\mathcal { L } _ { \mathrm { p h a s e } } ( \pmb { x } ^ { \prime } , \pmb { x } ) = \left| m \odot ( \mathcal { P } ( \pmb { x } ^ { \prime } ) - \mathcal { P } ( \pmb { x } ) ) \right| _ { 2 } ^ { 2 } ,\tag{8}
$$

where $\mathcal { P } ( \cdot )$ denotes the pixel-wise phase spectrum. This term encourages perturbations to induce meaningful structural changes within the phase-aware regions, which are known to dominate semantic understanding.

Spatial-domain phase pattern regularization. To effectively mislead the MLLMs’ focus on the phase spectrum, we further propose to optimize the phase pattern of the adversarial example to be far away from the benign one, suppressing the MLLMs reason with the accurate structural semantics:

![](images/8f33d7d830f736ec5f6a00a1d3583207ae67f947842bef578e276f5d625147a7.jpg)  
Fig. 3. Overview of the proposed attack pipeline. We first generate MLLM’s sensitive phase-aware perturbations to fool MLLM’s intrinsic focus. Then, we further learn trainable auxiliary prompts to enhance the effectiveness of the phase-aware perturbations.

$$
\mathcal { L } _ { \mathrm { p a t t e r n } } ( \boldsymbol { x } ^ { \prime } , \boldsymbol { x } ) = - \left| \mathcal { F } ^ { - 1 } ( \mathbf { 1 } , \mathcal { P } ( \boldsymbol { x } ^ { \prime } ) ) - \mathcal { F } ^ { - 1 } ( \mathbf { 1 } , \mathcal { P } ( \boldsymbol { x } ) ) \right| _ { 2 } ^ { 2 } .\tag{9}
$$

By jointly optimizing for adversarial effectiveness and phase-domain variation via:

$$
\delta \gets \mathrm { c l i p } ( \delta - \alpha \cdot \mathrm { s i g n } ( \nabla _ { \delta } ( \mathcal { L } _ { t a s k } + \mathcal { L } _ { p h a s e } + \mathcal { L } _ { p a t t e r n } ) ) ) ,\tag{10}
$$

the perturbation becomes more aligned with MLLMs’ intrinsic reliance on structural semantics, increasing the success rate of attacks.

## D. Adversarial Prompt Enhancement

While phase perturbations inherently target structural semantics, their impact on multimodal models may be diluted if the model’s attention is not sufficiently focused on these phasedominant features. To address this, we propose an adversarial prompt learning module, which serves as an auxiliary mechanism to enhance the optimization of visual phase perturbations. Specifically, we introduce a trainable textual prompt $\pmb { p } _ { a u x }$ that is designed to highlight phase-sensitive structures during the image-prompt interaction process. This auxiliary prompt is not adversarial by itself, but is jointly optimized to amplify the MLLM’s reliance on the masked structural regions m extracted from the image phase.

Concretely, given a benign image x and a structure mask m, we first define a distortion loss that encourages the MLLM to ground its semantic understanding not on the masked phase regions when paired with $p _ { a u x }$ to provide accurate response $^ { r } \cdot$ Based on this, we optimize the $\pmb { p } _ { a u x }$ via:

$$
\mathcal { L } _ { \mathrm { t a s k } } ( \boldsymbol { x } ^ { \prime } , \boldsymbol { p } ^ { \prime } , \boldsymbol { r } ) = \mathcal { L } ( M ( \boldsymbol { x } ^ { \prime } , \boldsymbol { p } ^ { \prime } ) , \boldsymbol { r } ) ,\tag{11}
$$

```latex
Algorithm 1: Our attack algorithm
Input: Image x, prompt p, MLLM M, target response
$\mathbf { \Delta } \mathbf { r } ^ { \prime } ,$ , step size α, numbers of iteration $T _ { 1 } , T _ { 2 } , T _ { 3 }$
Output: Adversarial example $\pmb { x } ^ { \prime } = \pmb { x } + \pmb { \delta } \odot$ m
Compute phase pattern of image x:
${ \pmb x } _ { p h a } = { \mathcal { F } } ^ { - 1 } ( { \bf 1 } , { \mathcal { P } } ( { \pmb x } ) )$
Compute phase-aware structural mask:
m = Thresh(Sobel $\left( { \pmb x } _ { p h a } \right) )$
Initialize:
δ ← 0, auxiliary prompt $p _ { a u x }$ via random vocabulary
high-frequency words
for t = 1 to $T _ { 1 }$ do
adversarial example $\pmb { x } ^ { \prime }  \pmb { x } + \pmb { \delta } \odot$ m
for t = 1 to $T _ { 2 }$ do
$\pmb { p _ { a u x } } $
Adam $\left( { { p } _ { a u x } } , \nabla _ { { p } _ { a u x } } { \mathcal { L } } _ { \mathrm { t a s k } } { \left( { { x } ^ { \prime } } , { p } \oplus { { p } _ { a u x } } , { r } \right) } \right)$
for $t = 1$ to $T _ { 3 }$ do
Compute
$\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { t a s k } } ( \boldsymbol { x } ^ { \prime } , \boldsymbol { p } \oplus \boldsymbol { p } _ { a u x } , \boldsymbol { r } ^ { \prime } ) + \mathcal { L } _ { \mathrm { p h a s e } } + \mathcal { L } _ { \mathrm { p a t t e r n } }$
δ ← clip (δ−α · sign(∇<sub>δ</sub>L<sub>total</sub>))
return $\pmb { x } ^ { \prime } = \pmb { x } + \pmb { \delta } \odot \pmb { m }$
```

where $\pmb { p } ^ { \prime } = \pmb { p } \oplus \pmb { p } _ { a u x } .$ We follow previous works [66], [67] to define the learnable $\pmb { p } _ { a u x }$ in the word token space. Then, during the optimization of the adversarial image $\mathbf { { \boldsymbol { x } } ^ { \prime } } .$ we freeze the auxiliary prompt $\pmb { p } _ { a u x }$ and define the adversarial loss to attack MLLMs with auxiliary prompt as:

$$
\mathcal { L } = \mathcal { L } _ { t a s k } ( { \pmb x } ^ { \prime } , { \pmb p } ^ { \prime } , { \pmb r } ^ { \prime } ) + \mathcal { L } _ { p h a s e } ( { \pmb x } ^ { \prime } , { \pmb x } ) + \mathcal { L } _ { p a t t e r n } ( { \pmb x } ^ { \prime } , { \pmb x } ) .\tag{12}
$$

The whole process is shown in Algorithm 1, where the auxiliary prompt acts as a phase-sensitive “semantic lens”, guiding the attack optimization to focus on perturbing the structural components that the model heavily relies on. The learnable prompt is only utilized to serve as an auxiliary tool during the attack optimization stage to enhance the effectiveness/robustness of the adversarial examples, which will not be utilized at inference. Unlike previous global attack methods, our MLLM-sensitive design preserves semantic plausibility, ensuring that the attack remains imperceptible from the language side while being more effective from the vision.

TABLE II  
TARGETED ATTACK PERFORMANCE OF DIFFERENT MLLM ATTACK METHODS ACROSS MULTIPLE MLLM DATASETS AND OPEN-SOURCE MODELS.
<table><tr><td rowspan="2">Data</td><td rowspan="2">Attack</td><td colspan="2">BLIP-2 [1]</td><td colspan="2"></td><td colspan="2">LLaVA-1.5 [2]</td><td colspan="2">Flamingo [4]</td><td colspan="2"></td><td colspan="2">MiniGPT-4 [68]</td><td colspan="2">Qwen2-VL [69]</td><td colspan="2"></td><td colspan="2">Intern-VL [70]</td></tr><tr><td>SS</td><td></td><td>E-ASR C-ASR</td><td>SS</td><td>E-ASR C-ASR</td><td></td><td>SS</td><td></td><td>E-ASR C-ASR</td><td></td><td></td><td>SS E-ASR C-ASR</td><td>SS</td><td></td><td>E-ASR C-ASR</td><td>SS</td><td></td><td>E-ASR C-ASR</td></tr><tr><td></td><td>APGD [65]</td><td>0.662</td><td>61.0</td><td>66.2</td><td>0.670</td><td>60.2</td><td>65.0</td><td>0.655</td><td>59.1</td><td>63.4</td><td>0.685</td><td>63.5</td><td>67.0</td><td>0.676</td><td>61.8</td><td>65.2</td><td>0.661</td><td>60.5</td><td>64.1</td></tr><tr><td></td><td>MF-Att [40]</td><td>0.692</td><td>65.3</td><td>69.0</td><td>0.701</td><td>67.2</td><td>70.8</td><td>0.685</td><td>64.5</td><td>67.5</td><td>0.720</td><td>68.1</td><td>71.3</td><td>0.707</td><td>66.4</td><td>70.0</td><td>0.698</td><td>65.7</td><td>68.2</td></tr><tr><td>[ma Ia71][</td><td>CroPA [38]</td><td>0.715</td><td>67.9</td><td>71.2</td><td>0.726</td><td>69.5</td><td>72.4</td><td>0.701</td><td>66.1</td><td>69.7</td><td>0.739</td><td>70.2</td><td>73.4</td><td>0.723</td><td>68.6</td><td>71.2</td><td>0.718</td><td>67.8</td><td>70.6</td></tr><tr><td></td><td>MABA [72]</td><td>0.748</td><td>71.5</td><td>74.8</td><td>0.779</td><td>75.3</td><td>79.8</td><td>0.742</td><td>72.5</td><td>75.0</td><td>0.784</td><td>77.4</td><td>80.5</td><td>0.771</td><td>76.2</td><td>78.9</td><td>0.765</td><td>74.8</td><td>77.0</td></tr><tr><td></td><td>VMA [73]</td><td>0.768</td><td>74.2</td><td>77.1</td><td>0.796</td><td>78.5</td><td>81.1</td><td>0.765</td><td>75.3</td><td>78.0</td><td>0.803</td><td>79.4</td><td>81.9</td><td>0.789</td><td>77.5</td><td>80.1</td><td>0.783</td><td>76.2</td><td>79.0</td></tr><tr><td></td><td>Ours</td><td>0.841</td><td>86.5</td><td>88.2</td><td>0.904</td><td>90.0</td><td>90.9</td><td>0.855</td><td>86.2</td><td>87.1</td><td>0.899</td><td>88.7</td><td>89.4</td><td>0.895</td><td>88.3</td><td>88.7</td><td>0.887</td><td>87.6</td><td>88.5</td></tr><tr><td></td><td>APGD [65]</td><td>0.652</td><td>59.3</td><td>64.1</td><td>0.667</td><td>61.8</td><td>65.4</td><td>0.648</td><td>60.1</td><td>63.2</td><td>0.680</td><td>63.9</td><td>67.1</td><td>0.662</td><td>61.4</td><td>64.8</td><td>0.650</td><td>59.9</td><td>63.3</td></tr><tr><td>SV L4]</td><td>MF-Att [40]</td><td>0.679</td><td>63.0</td><td>66.5</td><td>0.692</td><td>66.3</td><td>69.0</td><td>0.675</td><td>62.1</td><td>65.4</td><td>0.712</td><td>67.0</td><td>70.0</td><td>0.699</td><td>64.5</td><td>67.6</td><td>0.688</td><td>63.2</td><td>66.2</td></tr><tr><td></td><td></td><td>0.703</td><td>66.8</td><td>69.5</td><td>0.720</td><td>68.2</td><td>71.4</td><td>0.693</td><td>65.4</td><td>67.9</td><td>0.732</td><td>69.1</td><td>71.7</td><td>0.715</td><td>67.8</td><td>70.5</td><td>0.709</td><td>66.4</td><td>68.8</td></tr><tr><td></td><td>CroPA [38]</td><td>0.738</td><td>70.4</td><td>73.5</td><td>0.765</td><td>74.4</td><td>77.2</td><td>0.725</td><td>70.6</td><td>73.0</td><td>0.788</td><td>76.9</td><td>78.5</td><td>0.765</td><td>74.4</td><td>77.2</td><td>0.757</td><td>73.5</td><td>75.4</td></tr><tr><td></td><td>MABA [72]</td><td>0.757</td><td>72.2</td><td>75.1</td><td>0.785</td><td>77.1</td><td>79.4</td><td>0.747</td><td>73.3</td><td>75.9</td><td>0.796</td><td>78.6</td><td>80.5</td><td>0.781</td><td>76.4</td><td>78.6</td><td>0.774</td><td>75.0</td><td>77.4</td></tr><tr><td></td><td>VMA [73]</td><td>0.832</td><td>85.0</td><td>86.8</td><td>0.893</td><td>89.5</td><td>90.7</td><td>0.848</td><td>85.1</td><td>85.9</td><td>0.901</td><td>89.3</td><td>90.2</td><td>0.889</td><td>87.9</td><td>89.3</td><td>0.882</td><td>86.5</td><td>88.1</td></tr><tr><td></td><td>Ours</td><td></td><td></td><td></td><td>0.701</td><td>66.9</td><td>72.5</td><td>0.695</td><td></td><td></td><td>0.725</td><td></td><td></td><td>0.734</td><td></td><td></td><td></td><td></td><td>71.1</td></tr><tr><td>DALL LLS]</td><td>APGD [65]</td><td>0.701</td><td>66.9</td><td>72.5</td><td></td><td></td><td>75.5</td><td>0.719</td><td>65.3</td><td>70.4 72.8</td><td></td><td>68.2</td><td>74.9 76.2</td><td></td><td>67.8</td><td>72.7</td><td>0.715</td><td>66.3</td><td>72.6</td></tr><tr><td></td><td>MF-Att [40]</td><td>0.728</td><td>70.3</td><td>74.1</td><td>0.735</td><td>71.2 73.3</td><td>77.0</td><td></td><td>69.0</td><td>74.5</td><td>0.752</td><td>73.0</td><td>78.9</td><td>0.741</td><td>71.4</td><td>74.8</td><td>0.728</td><td>69.5</td><td>75.0</td></tr><tr><td></td><td>CroPA [38]</td><td>0.742</td><td>72.0</td><td>75.8</td><td>0.748</td><td>75.3</td><td>79.8</td><td>0.735</td><td>71.2</td><td>77.0</td><td>0.771 0.784</td><td>76.2</td><td>80.5</td><td>0.763</td><td>73.8</td><td>77.4</td><td>0.751</td><td>72.1</td><td>77.0</td></tr><tr><td></td><td>MABA [72]</td><td>0.779</td><td>75.3</td><td>79.8</td><td>0.779 0.803</td><td>78.7</td><td>81.8</td><td>0.767</td><td>73.1</td><td>78.5</td><td>0.809</td><td>77.4 79.3</td><td>81.2</td><td>0.771 0.795</td><td>76.2</td><td>78.9</td><td>0.765</td><td>74.8</td><td></td></tr><tr><td></td><td>VMA [73]</td><td>0.796</td><td>77.9</td><td>81.1</td><td></td><td></td><td></td><td>0.785</td><td>75.4</td><td></td><td></td><td></td><td></td><td></td><td>77.1</td><td>79.4</td><td>0.788</td><td>76.0 87.6</td><td>78.3</td></tr><tr><td></td><td>Ours</td><td>0.897</td><td>87.0</td><td>90.1</td><td>0.904</td><td>90.0</td><td>90.9</td><td>0.891</td><td>88.0</td><td>88.7</td><td>0.899</td><td>88.7</td><td>89.4</td><td>0.895</td><td>88.3</td><td>88.7</td><td>0.887</td><td></td><td>88.5</td></tr></table>

## E. Discussion on Novelty and Design Choice

Relationship to prior frequency-domain attacks. We acknowledge that some building blocks used in our framework, including Fourier analysis, structural extraction, and adversarial optimization, have been explored in previous adversarial attack literature. However, our work differs from existing frequency-domain attacks in both motivation and formulation. Most prior frequency-based attacks are developed for conventional image classifiers and primarily focus on perturbing predefined frequency bands to improve imperceptibility, transferability, or robustness. In contrast, our work investigates the role of frequency information in multimodal semantic grounding and language reasoning. Rather than treating frequency as a perturbation constraint, we analyze how different spectral components contribute to MLLM perception and subsequently design attacks according to the observed semantic sensitivity. Novelty of the proposed framework. Previous studies have shown that phase information preserves structural content in natural images. However, whether such observations remain valid in multimodal large language models has not been systematically investigated. Our controlled analyses demonstrate that MLLMs exhibit substantially stronger dependence on phase information than amplitude information at the prediction, representation, and cross-modal attention levels. This observation motivates the proposed phase-aware semantic attack framework. The novelty of our method does not stem from any individual optimization component. Instead, it lies in establishing a phase-aware semantic grounding perspective for MLLM attacks and integrating frequency-domain analysis with multimodal semantic optimization. To the best of our knowledge, this is among the first studies that systematically connect spectral sensitivity, semantic grounding, and adversarial robustness in multimodal large language models.

## V. EXPERIMENTS

## A. Experimental Setup

MLLM datasets. We utilize ImageNet [71], SVIT [74], and DALLE [75] datasets for image captioning, image classification, and VQA tasks. As for evaluation metrics, we (1) measure the semantic similarity (SS) between raw and adversarial answers; (2) measure the “ExactMatch” attack success rates (E-ASR) to assess word-to-word overlap between output and target texts; (3) measure the “Conditional-Contain” success rates (C-ASR) to assess a flexible word-level overlap between output and target texts.

MLLM models. To assess the MLLMs’ robustness against our attack, we select six open-source MLLM models with different architectures for evaluation: BLIP-2 [1], LLaVA-1.5 [2], Flamingo [4], MiniGPT-4 [68], Qwen2-VL [69], and Intern-VL [70]. Five closed-source MLLM models, Claude-3.5 [76], Claude-3.7 [77], GPT-4o [78], GPT-4.1 [79], Gemini-2.0 [80], are also implemented.

Implementation details. For attack setup, we utilize a maximum of $T _ { 1 } = 1 0 0$ epochs to optimize the adversarial noise and set the perturbation budget $\epsilon = 8 ,$ , the decay factor $\mu = 0 . 9$ , the step size $\alpha = { \epsilon } / { e p o c h }$ . To implement the adversarial learning, we set $T _ { 2 } = 3 0 , T _ { 3 } = 2 0$ to optimize the perturbation. For each benchmark (ImageNet, SVIT, and DALLE), we randomly sample 1,000 images and use the same evaluation subset for all compared methods. Input images are resized to $2 2 4 \times 2 2 4$ before optimization. For phase-aware perturbation generation, we first construct phase-dominant regions by reconstructing a phase-only image in the Fourier domain and extracting the corresponding structural response map. The top 20% highestresponse pixels are selected as phase-dominant regions. For targeted attacks, target labels or responses are randomly selected from semantically unrelated categories or a predefined target pool. The same target assignments are shared across all compared methods to ensure fair evaluation. For closed-source MLLMs, we use official APIs with deterministic decoding (temperature = 0) and a maximum query budget of 100 queries per image. All compared baselines are implemented using their official releases whenever available and are evaluated under the same perturbation budget and experimental protocol as us for fair comparison. To assess statistical stability, each experiment is repeated using three random seeds {0,1,2}. All experiments are conducted on a single NVIDIA H100 Tensor Core GPU.

TABLE III  
TARGETED ATTACK PERFORMANCE OF DIFFERENT MLLM ATTACK METHODS ACROSS MULTIPLE MLLM DATASETS AND CLOSED-SOURCE MODELS.
<table><tr><td rowspan="2">Data</td><td rowspan="2">Attack</td><td colspan="3">Claude-3.5 [76]</td><td colspan="3">Claude-3.7 [77]</td><td colspan="3">GPT-40 [78]</td><td colspan="3">GPT-4.1 [79]</td><td colspan="3">Gemini-2.0 [80]</td></tr><tr><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td></tr><tr><td rowspan="7">ImaəeNet</td><td>MF-Att [40]</td><td>0.22</td><td>6.8</td><td>10.3</td><td>0.24</td><td>7.4</td><td>11.5</td><td>0.20</td><td>6.2</td><td>9.1</td><td>0.21</td><td>6.5</td><td>9.8</td><td>0.19</td><td>5.9</td><td>8.5</td></tr><tr><td>M-Att [81]</td><td>0.39</td><td>28.3</td><td>32.4</td><td>0.41</td><td>30.2</td><td>34.1</td><td>0.38</td><td>27.5</td><td>30.7</td><td>0.37</td><td>25.8</td><td>29.5</td><td>0.36</td><td>24.3</td><td>27.1</td></tr><tr><td>M-Att v2 [82]</td><td>0.44</td><td>32.1</td><td>35.2</td><td>0.46</td><td>34.0</td><td>37.3</td><td>0.47</td><td>35.6</td><td>40.8</td><td>0.45</td><td>33.7</td><td>38.1</td><td>0.42</td><td>33.9</td><td>36.5</td></tr><tr><td>FOA [83]</td><td>0.47</td><td>34.8</td><td>40.2</td><td>0.49</td><td>38.5</td><td>43.0</td><td>0.46</td><td>35.2</td><td>39.6</td><td>0.44</td><td>33.0</td><td>38.2</td><td>0.43</td><td>31.6</td><td>36.0</td></tr><tr><td>Ours</td><td>0.58</td><td>53.7</td><td>59.8</td><td>0.60</td><td>57.2</td><td>63.1</td><td>0.59</td><td>54.8</td><td>60.2</td><td>0.57</td><td>52.3</td><td>58.6</td><td>0.55</td><td>49.1</td><td>55.7</td></tr><tr><td>MF-Att [40]</td><td>0.24</td><td>7.5</td><td>11.2</td><td>0.25</td><td>8.2</td><td>12.4</td><td>0.22</td><td>7.2</td><td>10.6</td><td>0.23</td><td>7.7</td><td>11.3</td><td>0.21</td><td>6.8</td><td>9.7</td></tr><tr><td>M-Att [81]</td><td>0.41</td><td>30.3</td><td>35.1</td><td>0.43</td><td>33.1</td><td>37.4</td><td>0.40</td><td>29.0</td><td>33.6</td><td>0.39</td><td>27.8</td><td>32.5</td><td>0.38</td><td>26.1</td><td>30.1</td></tr><tr><td>SVT M-Att v2 [82]</td><td>0.46</td><td>35.8</td><td>40.5</td><td>0.48</td><td>38.2</td><td>43.0</td><td>0.47</td><td>36.9</td><td>41.6</td><td>0.45</td><td>35.1</td><td>39.8</td><td>0.43</td><td>33.0</td><td></td><td>37.4</td></tr><tr><td></td><td>FOA [83]</td><td>0.49</td><td>38.6</td><td>44.0</td><td>0.51</td><td>41.2</td><td>47.1</td><td>0.48</td><td>38.0</td><td>43.3</td><td>0.46</td><td>36.0</td><td>41.2</td><td>0.45</td><td>34.2</td><td>39.0</td></tr><tr><td rowspan="5"></td><td>Ours</td><td>0.61</td><td>58.1</td><td>64.4</td><td>0.63</td><td>61.2</td><td>66.8</td><td>0.61</td><td>57.5</td><td>63.3</td><td>0.60</td><td>55.6</td><td>61.5</td><td>0.58</td><td>52.1</td><td>58.7</td></tr><tr><td>MF-Att [40]</td><td>0.26</td><td>8.6</td><td>12.7</td><td>0.27</td><td>9.1</td><td>13.5</td><td>0.24</td><td>8.1</td><td>11.9</td><td>0.25</td><td>8.3</td><td>12.3</td><td>0.23</td><td>7.4</td><td>10.8</td></tr><tr><td>M-Att [81]</td><td>0.42</td><td>31.7</td><td>36.5</td><td>0.44</td><td>33.5</td><td>38.8</td><td>0.41</td><td>30.3</td><td>35.1</td><td>0.39</td><td>28.6</td><td>33.7</td><td>0.38</td><td>26.8</td><td>31.4</td></tr><tr><td>M-Att v2 [82]</td><td>0.48</td><td>38.1</td><td>43.7</td><td>0.50</td><td>40.6</td><td>46.1</td><td>0.49</td><td>39.2</td><td>45.0</td><td>0.47</td><td>37.5</td><td>43.2</td><td>0.45</td><td>35.3</td><td>40.5</td></tr><tr><td>FOA [83]</td><td>0.50</td><td>40.9</td><td>46.8</td><td>0.52</td><td>43.3</td><td>48.9</td><td>0.49</td><td>39.2</td><td>45.5</td><td>0.47</td><td>37.6</td><td>43.8</td><td>0.45</td><td>35.1</td><td>41.0</td></tr><tr><td>DALLE</td><td>Ours</td><td>0.63</td><td>61.3</td><td>67.9</td><td>0.65</td><td>64.7</td><td>70.2</td><td>0.63</td><td>60.1</td><td>67.0</td><td>0.61</td><td>57.2</td><td>64.5</td><td>0.60</td><td>54.3</td><td>61.8</td></tr></table>

TABLE IV  
ABLATION STUDY ON EACH COMPONENT OF OUR PROPOSED METHOD ON THE IMAGENET DATASET [71].
<table><tr><td rowspan="2">Phase Region</td><td colspan="3">Optimization</td><td rowspan="2">Adversarial Learning</td><td colspan="3">BLIP-2 [1]</td><td colspan="3">LLaVA-1.5 [2]</td><td colspan="3">Qwen2-VL</td><td colspan="3">Intern-VL [70]</td></tr><tr><td>Lphase</td><td> $\underline { { \mathcal { L } _ { p a t t e r n } } }$ </td><td></td><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>[69] C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td></tr><tr><td>X</td><td> $\overline { { \mathcal { L } _ { t a s k } } }$  √</td><td>X</td><td>×</td><td>X</td><td>0.541</td><td>47.2</td><td>50.8</td><td>0.592</td><td>54.5</td><td>56.1</td><td>0.574</td><td>51.0</td><td>52.7</td><td>0.563</td><td>49.6</td><td>51.2</td></tr><tr><td>√</td><td>√</td><td>X</td><td>X</td><td>X</td><td>0.765</td><td>73.6</td><td>76.4</td><td>0.825</td><td>81.1</td><td>82.5</td><td>0.814</td><td>78.2</td><td>79.3</td><td>0.802</td><td>77.5</td><td>78.4</td></tr><tr><td>√</td><td>V</td><td>√</td><td>X</td><td>X</td><td>0.742</td><td>70.1</td><td>74.8</td><td>0.801</td><td>78.3</td><td>80.2</td><td>0.789</td><td>75.6</td><td>77.2</td><td>0.777</td><td>74.4</td><td>75.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>X</td><td>0.781</td><td>76.8</td><td>79.0</td><td>0.849</td><td>85.3</td><td>86.5</td><td>0.838</td><td>82.4</td><td>83.1</td><td>0.825</td><td>81.6</td><td>82.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>0.841</td><td>86.5</td><td>88.2</td><td>0.904</td><td>90.0</td><td>90.9</td><td>0.895</td><td>88.3</td><td>88.7</td><td>0.887</td><td>87.6</td><td>88.5</td></tr></table>

## B. Main Attack Performance

As shown in Table II, we evaluate the proposed method and five competitive baselines (APGD [65], MF-Att [40], CroPA [38], MABA [72], VMA [73]) on three public datasets, targeting six popular open-source MLLM models. During adversarial example generation, the attacker has access to gradients, logits, probabilities, internal activations, visual encoder parameters, and tokenizer information from the target model. Across all configurations, our method consistently achieves the highest attack success rates on all evaluation metrics. This demonstrates the capability of our method to generalize well across diverse architectures and datasets.

To further assess the effectiveness of our method under more practical and challenging conditions, we extend our method with the Monte Carlo method [84] to handle the black-box scenarios. It can only access final textual responses through official APIs. No model internals, confidence scores, token probabilities, or gradient information are accessible. We evaluate the targeted attack performance on five widely used MLLM models across three datasets. Since baselines in Table II cannot handle the black-box MLLMs, we implement five black-box baselines for comparison. As reported in Table III, our method also significantly outperforms state-of-the-art black-box attack baselines. These results confirm the high attack generality of our approach, making it highly applicable to real-world scenarios.

## C. Ablation Study

Effectiveness of each component. We conduct an ablation study to assess the effectiveness of each component in our proposed framework. As shown in Table IV, starting from a baseline with only the task loss, we incrementally add different modules to examine their individual impact. The phase region module leads to the most notable improvement, indicating that constraining perturbations to perceptually and semantically important frequency regions is crucial for enhancing attack success. Further incorporating the pattern consistency loss improves the alignment between the generated perturbation and the desired structural semantics, leading to more coherent adversarial responses. Adversarial learning contributes additional robustness by optimizing perturbations under a range of simulated conditions, resulting in stronger transferability. Applying a phase regularization loss slightly hampers performance, this is because its constraint on perturbation magnitude, which limits the expressiveness of the attack.

Evaluations of attacking on different targets. We explore the effectiveness of our method when simultaneously attacking different targets. As shown in Figure 4, we set four different adversarial targets: refusal response, irrelevant semantic, misleading prefix, and toxic language. Results show that our attack can achieve great attack performance across different adversarial targets, demonstrating the strong generalization ability and stability of our proposed attack. It also denotes that our attack is not sensitive to the attacker’s goal and can be widely deployed in the real-world scenarios.

![](images/bb579516369d4e1d7726c4319ad34efb54e74010dcd5fb4ed175de75fd09120b.jpg)

![](images/33d471ccd9f3c63bfcdaa301c3230391699f547233044669d371b31413b0d501.jpg)

![](images/95b1ff9b5ed9ab74c912b44d550515631cdb448cb12ce7ddb1382c90b6c4a73e.jpg)

![](images/753136ff23fb737f57c85b78cb297870d7d8633713d5892a58bd64f7cc983c38.jpg)  
Fig. 4. Attack performance with different attackers-chosen adversarial targets. Types 1-4 denote refusal response, irrelevant semantic, misleading prefix, and toxic language, respectively.

![](images/b5962f69676e4a378c1f022bf287395ca628cf32ab77b54132efd0393349c553.jpg)  
Fig. 5. Ablation study on iteration numbers on BLIP-2.

![](images/202b4fb50bb6e50d0a058d0e80df162febbd5106fe4c2d7b8500199ee9b81785.jpg)

![](images/86f464ef2b886ac6d6a49720ab431c0774092436a1323685f86164fa12179d09.jpg)

TABLE V  
ABLATION OF DETAILED DESIGNS ON IMAGENET DATASET [71].
<table><tr><td rowspan="2">Component</td><td rowspan="2">Variant</td><td colspan="3">Qwen2-VL [69]</td><td colspan="3">Intern-VL [70]</td></tr><tr><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td></tr><tr><td rowspan="3">Phase-aware Design</td><td>DWT</td><td>0.876</td><td>86.4</td><td>87.0</td><td>0.858</td><td>84.9</td><td>85.7</td></tr><tr><td>DCT</td><td>0.839</td><td>82.4</td><td>84.2</td><td>0.846</td><td>85.5</td><td>86.9</td></tr><tr><td>DFT</td><td>0.895</td><td>88.3</td><td>88.7</td><td>0.887</td><td>87.6</td><td>88.5</td></tr><tr><td rowspan="3">Adversarial Prompt</td><td>origin</td><td>0.815</td><td>81.8</td><td>83.2</td><td>0.811</td><td>80.6</td><td>82.7</td></tr><tr><td>auxiliary</td><td>0.852</td><td>84.1</td><td>85.8</td><td>0.844</td><td>84.3</td><td>84.9</td></tr><tr><td>both</td><td>0.895</td><td>88.3</td><td>88.7</td><td>0.887</td><td>87.6</td><td>88.5</td></tr></table>

Ablation on detailed designs. We provide detailed ablations on the in-depth design of our core components. As shown in Table V, we investigate the designs of both phase-aware perturbation and adversarial prompt enhancement. As for phaseaware designs, we replace our utilized DFT with other spectral tools DWT and DCT. We exploit their low- and high-frequency bands to denote the phase and amplitude spectrum. Results show that our utilized DFT achieves the best performance, because DFT can explicitly represent the phase-aware patterns representing the visual structure contexts. As for adversarial learning, leveraging auxiliary learnable prompts with original ones leads to better attack performance.

Ablation on iteration numbers. We provide the ablation of iteration numbers in Figure 5. To take a balance between the attack performance and resource cost, we set $T _ { 1 } = 1 0 0 , T _ { 2 } =$ $3 0 , T _ { 3 } = 2 0$ in our experiments.

Cross-prompt stability analysis. To investigate whether the effectiveness of the generated perturbations depends on the auxiliary prompt, we evaluate adversarial examples under different inference prompts. Specifically, adversarial examples are generated using the combined prompt $( p + p _ { a u x } )$ , while evaluation is conducted using the original prompt, paraphrased prompts, and template-rewritten prompts. As shown in Table VI, removing the auxiliary prompt during inference only leads to a minor decrease in attack success rate. Similar behavior is observed under prompt paraphrasing and template modifications. These results suggest that the auxiliary prompt primarily serves as an optimization-time semantic guidance mechanism rather than a necessary attack component during deployment. The generated perturbations exhibit strong crossprompt transferability and remain effective under the original prompt alone.

TABLE VI  
CROSS-PROMPT STABILITY ANALYSIS OF THE AUXILIARY PROMPT.
<table><tr><td>Inference Prompt</td><td>SS</td><td>E-ASR</td><td>C-ASR</td></tr><tr><td>p + paux</td><td>0.92</td><td>89.7</td><td>91.4</td></tr><tr><td>Original Prompt p</td><td>0.91</td><td>87.8</td><td>89.9</td></tr><tr><td>Paraphrased Prompt</td><td>0.90</td><td>86.5</td><td>88.4</td></tr><tr><td>Template-Rewritten Prompt</td><td>0.90</td><td>85.9</td><td>87.6</td></tr></table>

## D. More Analysis

Robustness to potential defenses. To investigate the robustness of our proposed attack, we evaluate attack methods against potential defenses in a realistic white-box setting defense (i.e., fine-tuning [85], one of the most widely used methods) and a state-of-the-art black-box setting defense (i.e., zero-shot image purification [86]). As shown in Table VIII and Table IX, we report the performance on both open-source and closed-source MLLM models. Results show that our attack still achieves much better performance than the compared attack baselines, demonstrating that our method is more robust to defense strategies. We also try to implement more defenses: Resizing, Smoothing Filters, and JPEG Compression as shown in Table X, where our method is more robust.

TABLE VII  
ROBUSTNESS AGAINST STRONGER ADAPTIVE DEFENSES ON QWEN2-VL AND IMAGENET.
<table><tr><td rowspan="2">Attack</td><td colspan="3">Adv. Fine-Tuning</td><td colspan="3">Freq. Filtering</td><td colspan="3">Rand. Transform</td><td colspan="3">Token Consistency Check</td></tr><tr><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td></tr><tr><td>APGD [65]</td><td>0.412</td><td>41.8</td><td>43.2</td><td>0.435</td><td>43.7</td><td>44.9</td><td>0.446</td><td>44.2</td><td>45.8</td><td>0.421</td><td>42.1</td><td>43.7</td></tr><tr><td>MF-Att [40]</td><td>0.468</td><td>47.6</td><td>49.2</td><td>0.482</td><td>48.8</td><td>50.3</td><td>0.491</td><td>49.4</td><td>51.2</td><td>0.473</td><td>47.9</td><td>49.6</td></tr><tr><td>CroPA [38]</td><td>0.517</td><td>51.9</td><td>53.6</td><td>0.532</td><td>53.1</td><td>54.7</td><td>0.541</td><td>54.3</td><td>55.8</td><td>0.522</td><td>52.2</td><td>53.9</td></tr><tr><td>MABA [72]</td><td>0.593</td><td>59.8</td><td>61.5</td><td>0.606</td><td>60.7</td><td>62.4</td><td>0.615</td><td>61.6</td><td>63.8</td><td>0.598</td><td>60.1</td><td>61.8</td></tr><tr><td>VMA [73]</td><td>0.574</td><td>57.3</td><td>59.8</td><td>0.588</td><td>58.6</td><td>60.7</td><td>0.601</td><td>59.5</td><td>61.8</td><td>0.580</td><td>57.9</td><td>60.1</td></tr><tr><td>Ours</td><td>0.734</td><td>73.6</td><td>75.8</td><td>0.761</td><td>76.2</td><td>78.1</td><td>0.748</td><td>74.8</td><td>76.9</td><td>0.722</td><td>72.5</td><td>74.6</td></tr></table>

TABLE VIII  
DEFENSE ON QWEN2-VL [69] AND IMAGENET [71] DATASET.
<table><tr><td rowspan="2">Attack</td><td colspan="3">With Fine-tuning</td><td colspan="3">With Purification</td></tr><tr><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td></tr><tr><td>APGD [65]</td><td>0.484</td><td>47.8</td><td>49.6</td><td>0.495</td><td>48.3</td><td>49.2</td></tr><tr><td>MF-Att [40]</td><td>0.525</td><td>55.1</td><td>55.1</td><td>0.507</td><td>51.9</td><td>52.0</td></tr><tr><td>CroPA [38]</td><td>0.567</td><td>56.2</td><td>57.3</td><td>0.580</td><td>57.4</td><td>58.6</td></tr><tr><td>MABA [72]</td><td>0.638</td><td>64.1</td><td>65.2</td><td>0.640</td><td>64.4</td><td>66.5</td></tr><tr><td>VMA [73]</td><td>0.619</td><td>60.3</td><td>63.4</td><td>0.598</td><td>60.2</td><td>62.8</td></tr><tr><td>Ours</td><td>0.786</td><td>78.1</td><td>80.6</td><td>0.835</td><td>83.7</td><td>83.7</td></tr></table>

TABLE IX  
DEFENSE ON CLAUDE-3.5 [76] AND IMAGENET [71] DATASET.
<table><tr><td rowspan="2">Attack</td><td colspan="3">With Fine-tuning</td><td colspan="3">With Purification</td></tr><tr><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS</td><td>E-ASR</td><td>C-ASR</td></tr><tr><td>MF-Att [40]</td><td>0.07</td><td>7.5</td><td>7.9</td><td>0.09</td><td>4.0</td><td>9.1</td></tr><tr><td>M-Att [81]</td><td>0.16</td><td>14.8</td><td>14.8</td><td>0.20</td><td>15.9</td><td>17.6</td></tr><tr><td>FOA [83]</td><td>0.25</td><td>22.7</td><td>24.3</td><td>0.23</td><td>22.5</td><td>23.9</td></tr><tr><td>Ours</td><td>0.49</td><td>46.7</td><td>49.2</td><td>0.51</td><td>47.4</td><td>49.8</td></tr></table>

TABLE X  
DEFENSE ON QWEN2-VL AND IMAGENET DATASET.
<table><tr><td>Defense</td><td>APGD</td><td>MF-Att</td><td>CroPA</td><td>MABA</td><td>VMA</td><td>Ours</td></tr><tr><td>Resizing</td><td>0.462</td><td>0.481</td><td>0.497</td><td>0.536</td><td>0.528</td><td>0.824</td></tr><tr><td>Smoothing Filters</td><td>0.489</td><td>0.505</td><td>0.522</td><td>0.563</td><td>0.570</td><td>0.797</td></tr><tr><td>JPEG Compression</td><td>0.523</td><td>0.538</td><td>0.556</td><td>0.604</td><td>0.595</td><td>0.781</td></tr></table>

To further investigate whether our attack remains effective against defenses specifically related to frequency-domain manipulations, we additionally evaluate four stronger adaptive defenses, including adversarial fine-tuning, frequency-domain filtering, randomized transformations, and visual-token consistency checking. As shown in Table VII, these defenses reduce attack success rates for all methods. Nevertheless, our attack consistently achieves the highest E-ASR and C-ASR among all baselines. This suggests that the effectiveness of our method does not solely originate from removable highfrequency artifacts, but is closely related to phase-sensitive semantic structures.

Complexity and efficiency. To investigate the attacks’ practicability, we also provide the analysis of attacks’ efficiency and complexity on single adversarial example generation. As shown in Figure 6, our attack has very competitive efficiency and complexity compared with other MLLM attacks while achieving much better attack performance, demonstrating our practicability.

Efficiency vs Complexity Comparison  
![](images/7de468937fc8adbd43cd9cc4c168573ab4ab3b6c5f3ea7b2d6a80fc7fdf768a9.jpg)  
Fig. 6. Comparisons on attacks’ efficiency/complexity.

TABLE XI  
TRANSFER ATTACK PERFORMANCE ON IMAGENET.
<table><tr><td rowspan="2">Attack</td><td colspan="3">BLIP-2 → Qwen2-VL</td><td colspan="2">BLIP-2 → Intern-VL</td></tr><tr><td>SS</td><td>E-ASR</td><td>C-ASR</td><td>SS E-ASR</td><td>C-ASR</td></tr><tr><td>APGD [65]</td><td>0.582</td><td>50.2</td><td>52.6</td><td>0.565 47.8</td><td>50.1</td></tr><tr><td>MF-Att [40]</td><td>0.626</td><td>55.4</td><td>57.9</td><td>0.607 52.8</td><td>55.3</td></tr><tr><td>CroPA [38]</td><td>0.651</td><td>58.9</td><td>61.2</td><td>0.634 56.5</td><td>58.7</td></tr><tr><td>MABA [72]</td><td>0.693</td><td>64.3</td><td>66.8</td><td>0.679 61.7</td><td>63.9</td></tr><tr><td>VMA [73]</td><td>0.718</td><td>66.9</td><td>69.1</td><td>0.692 64.2</td><td>66.5</td></tr><tr><td>Ours</td><td>0.783</td><td>74.8</td><td>77.2</td><td>0.765 71.3</td><td>73.9</td></tr></table>

Transfer attack. We try to implement transfer-attack comparison in Table XI, where we generate adversarial examples on BLIP-2 and test them on other MLLM models. The results show that our attack still achieves better performance, demonstrating our effectiveness.

LLM-as-Judge. To investigate whether our attack is sensitive to the judge model, we replace the text evaluator with the LLM model. We utilize LLM-based assessment (GPT-4o) to evaluate the similarity in the range of [0,1] as shown in Table XII. Given a generated response and a target response, GPT-4o evaluates whether the generated output semantically expresses the target content regardless of wording variations. The evaluation uses deterministic decoding (temperature = 0) and a fixed prompt template for all methods. It demonstrates that utilizing LLM to evaluate the textual output is also a good solution.

Perceptual quality analysis. As shown in Table XIII, although all methods are evaluated under the same perturbation budget $( L _ { \infty } = 8 / 2 5 5 )$ , the proposed phase-aware localization strategy significantly reduces the perturbed area ratio from 100% to only 21.3%, leading to substantially better perceptual quality in terms of LPIPS, SSIM, and PSNR.

![](images/fb45e3758e6877f997a70950024a7992d7ae1410c8385cb0c6bffc5f35b239cb.jpg)  
Fig. 7. Visualization results of our proposed phase-aware MLLM attack.

TABLE XII  
LLM-BASED EVALUATION.
<table><tr><td colspan="6">Attack BLIP-2 LLaVA-1.5 Flamingo MiniGPT-4 Qwen2-VL Intern-VL</td></tr><tr><td>APGD</td><td>0.41</td><td>0.38</td><td>0.35</td><td>0.37</td><td>0.40</td><td>0.39</td></tr><tr><td>MF-Att</td><td>0.47</td><td>0.44</td><td>0.41</td><td>0.43</td><td>0.46</td><td>0.45</td></tr><tr><td>CroPA</td><td>0.51</td><td>0.48</td><td>0.45</td><td>0.47</td><td>0.50</td><td>0.49</td></tr><tr><td>MABA</td><td>0.57</td><td>0.54</td><td>0.51</td><td>0.53</td><td>0.56</td><td>0.55</td></tr><tr><td>VMA</td><td>0.60</td><td>0.57</td><td>0.54</td><td>0.56</td><td>0.59</td><td>0.58</td></tr><tr><td>Ours</td><td>0.68</td><td>0.65</td><td>0.62</td><td>0.64</td><td>0.67</td><td>0.66</td></tr></table>

TABLE XIII

IMPERCEPTIBILITY COMPARISON OF DIFFERENT MLLM ATTACK METHODS.
<table><tr><td>Method</td><td>LPIPS↓ SSIM↑</td><td></td><td>PSNR↑ L0(%)</td><td></td><td>L2↓</td><td>L∞ ↓ Mask Ratio(%)↓</td></tr><tr><td>MF-Att</td><td>0.086</td><td>0.974</td><td>38.6</td><td>100.0</td><td>7.34 8</td><td>100.0</td></tr><tr><td>M-Att</td><td>0.074</td><td>0.978</td><td>39.8</td><td>100.0</td><td>6.81 8</td><td>100.0</td></tr><tr><td>M-Att v2</td><td>0.068</td><td>0.981</td><td>40.9</td><td>100.0</td><td>6.17 8</td><td>100.0</td></tr><tr><td>FOA</td><td>0.061</td><td>0.984</td><td>41.7</td><td>100.0</td><td>5.64 8</td><td>100.0</td></tr><tr><td>Ours</td><td>0.045</td><td>0.991</td><td>44.3</td><td>24.7</td><td>3.92 8</td><td>21.3</td></tr></table>

## E. Visualization

We provide visualization results of our generated adversarial examples as shown in Figure 7. Here, we provide the benign image, the decomposed phase pattern, the phase-aware edge region, and the final adversarial images. Results show that our attack can effectively fool MLLMs by suppressing their intrinsic phase focus.

## F. Discussion on Limitations and Failure Cases

Although the proposed phase-aware attack framework achieves strong performance across diverse multimodal large language models (MLLMs), its effectiveness relies on the assumption that phase components consistently encode semantically meaningful structural cues. This assumption may not fully hold under certain visual conditions, leading to degraded attack performance. (1) Complex and cluttered scenes. In highly cluttered scenes containing multiple objects or dense backgrounds, the phase spectrum tends to encode a mixture of overlapping structural patterns. As a result, the extracted phase-dominant regions become less spatially localized and may include redundant or irrelevant structures. This reduces the accuracy of phase-aware masking and leads to suboptimal perturbation allocation, thereby weakening attack effectiveness. (2) Weak structural or low-contrast images. Our method relies on the stability of edge- and contour-level information in the phase domain. However, for images with weak edges, motion blur, or low object-background contrast, the phase spectrum does not provide sufficiently discriminative structural cues. In such cases, the derived phase-aware mask may fail to accurately reflect semantic regions, limiting the effectiveness of structure-guided perturbations. (3) Small object dominance. When target objects occupy only a very small portion of the image, the available phase-sensitive region is inherently limited. Since our optimization constrains perturbations within phase-aware masks, the effective attack budget on semantically meaningful regions is reduced, leading to weaker adversarial impact and lower success rates. (4) Text-rich and OCR centric scenarios. For images containing dense textual content or requiring optical character recognition (OCR), semantic understanding depends more on high-frequency character-level details rather than global structural information. Since our method primarily exploits phase-domain structural priors, it may not fully capture fine-grained textual semantics, resulting in reduced attack effectiveness in such scenarios. (5) Finegrained reasoning tasks. Some multimodal reasoning tasks require attribute-level discrimination, numerical reasoning, or spatial relationship inference beyond global structural cues. In these cases, phase representations alone may be insufficient to fully characterize the reasoning-critical visual features, leading to less pronounced improvements compared to standard recognition or captioning tasks. Overall, these limitations suggest that while phase-aware perturbations effectively exploit structural sensitivity in MLLMs, the approach is inherently constrained by the reliability of phase-domain semantic alignment. Future work may benefit from integrating object-level segmentation, text-aware modeling, and hybrid frequency–spatial representations to improve robustness across more challenging visual distributions.

## G. Broader Applicability of Phase-aware Structural Modeling

Although this work focuses on adversarial attacks against multimodal large language models (MLLMs), the underlying principle of exploiting phase-aware structural representations is not limited to vision-language reasoning. Since phase information naturally captures object boundaries, spatial layouts, and semantic structures, the proposed frequency-aware modeling strategy has the potential to benefit a broader range of vision and multimodal learning problems. First, phaseaware structural representations could improve visual understanding tasks that require robust structural reasoning. For example, group activity recognition and compositional action recognition rely heavily on modeling interactions among multiple objects and their spatial configurations rather than solely appearance information. Recent studies have demonstrated the effectiveness of hierarchical graph reasoning and progressive feature learning for capturing such structured semantics [87]–[89]. Integrating phase-aware structural priors into these frameworks may provide complementary geometric cues for more robust representation learning. Second, our phase-guided localization mechanism may naturally extend to efficient visual token selection in large vision-language models. Existing token compression methods aim to preserve semantically important visual tokens while reducing computational overhead [90]. Since the proposed phase-aware masks explicitly identify structure-sensitive regions that dominate semantic perception, they could provide an interpretable criterion for adaptive token pruning and visual token allocation. Third, the proposed structural perturbation paradigm is closely related to fine-grained cross-modal correspondence learning. Applications such as dense audio-visual event localization require accurate alignment between visual structures and semantic events [91]. Phase-aware representations could provide additional structural consistency constraints to improve multimodal correspondence under challenging scenarios. Finally, our framework may also benefit few-shot visual recognition tasks, where learning transferable structural representations is often more important than modeling appearance variations. Recent few-shot recognition methods progressively align semantic features across different visual distributions [89], [92]. Since phase information is relatively invariant to texture and illumination changes, incorporating phase-aware structural modeling may further improve feature transferability under limited supervision. Overall, our work suggests that frequencydomain structural modeling is a general visual representation principle rather than a task-specific adversarial strategy. We believe that combining phase-aware semantic representations with graph reasoning, efficient token compression, crossmodal correspondence learning, and few-shot representation learning constitutes a promising direction for future research.

## VI. CONCLUSION

In this paper, we investigate the intrinsic sensitivity of multimodal large language models (MLLMs) to different frequency components of visual inputs. Motivated by the human visual system and empirical observations, we find that MLLMs rely heavily on the phase spectrum, which encodes structural and semantic information. Based on this insight, we propose a novel phase-aware adversarial attack framework that localizes perturbations to phase-dominant regions and explicitly manipulates spectral semantics. To further amplify the adversarial impact, we design an auxiliary prompt module with the adversarial learning scheme to guide the model’s attention during optimization. Extensive experiments across multiple MLLMs and tasks demonstrate the effectiveness and interpretability of our method. This work highlights the importance of understanding MLLMs’ frequency-domain behavior and opens up new directions for designing more principled and structure-aware adversarial strategies. Future work may explore how such frequency-level insights can benefit defense, detection, and robust training in multimodal systems.

## REFERENCES

[1] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International conference on machine learning. PMLR, 2023, pp. 19 730–19 742.

[2] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, 2024.

[3] J. Xu, R. Tang, P. Lv, M. Yu, G. Yu, and E. Chen, “Ltkt: Knowledge tracing based on positive and negative learning transfers,” Tsinghua Science and Technology, vol. 31, no. 3, pp. 1894–1917, 2026.

[4] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Advances in neural information processing systems, vol. 35, pp. 23 716–23 736, 2022.

[5] L. Ye, Z.-h. Li, H.-n. Yan, C. Liu, H. H. Cho, and T. Guo, “Predicting film-cooling effectiveness of compound-angle holes using a pod-based hybrid deep learning model,” Aerospace Science and Technology, p. 112590, 2026.

[6] D. Liu, X. Qu, J. Dong, P. Zhou, Y. Cheng, W. Wei, Z. Xu, and Y. Xie, “Context-aware biaffine localizing network for temporal sentence grounding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 11 235–11 244.

[7] D. Liu, X. Qu, X.-Y. Liu, J. Dong, P. Zhou, and Z. Xu, “Jointly cross-and self-modal graph attention network for query-based moment localization,” in Proceedings of the 28th ACM International Conference on Multimedia, 2020, pp. 4070–4078.

[8] D. Liu, X. Qu, X. Di, Y. Cheng, Z. Xu, and P. Zhou, “Memoryguided semantic learning network for temporal sentence grounding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 2, 2022, pp. 1665–1673.

[9] R. Jiang, S. Tong, J. Wu, H. Hu, R. Zhang, H. Wang, Y. Zhao, W. Zhu, S. Li, and X. Zhang, “A novel eeg artifact removal algorithm based on an advanced attention mechanism,” Scientific Reports, vol. 15, no. 1, p. 19419, 2025.

[10] D. Liu, X. Qu, J. Dong, and P. Zhou, “Adaptive proposal generation network for temporal sentence localization in videos,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 9292–9301.

[11] D. Liu, X. Qu, Y. Wang, X. Di, K. Zou, Y. Cheng, Z. Xu, and P. Zhou, “Unsupervised temporal video grounding with deep semantic clustering,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 2, 2022, pp. 1683–1691.

[12] D. Liu, X. Qu, and W. Hu, “Reducing the vision and language bias for temporal sentence grounding,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 4092–4101.

[13] L. Wang, Y. Ma, Z. Yan, L. Zhang, Y. Hu, and S. Zhao, “Giving or receiving: Impact of online socializing in online fitness community on physical activity and emotional state,” Computers in Human Behavior, vol. 169, p. 108669, 2025.

[14] J. Lv, C. Wang, and L. Xie, “Adaptive distributed observer design for nonlinear multiagent systems,” Automatica, vol. 183, p. 112625, 2026.

[15] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[16] Y. Chen, T. He, J. Fu, L. Wang, J. Guo, T. Hu, and H. Cheng, “Visionlanguage meets the skeleton: Progressively distillation with cross-modal knowledge for 3d action representation learning,” IEEE Transactions on Multimedia, 2024.

[17] Y. Qi, H. Li, Y. Song, X. Wu, and J. Luo, “How vision-language tasks benefit from large pre-trained models: A survey,” IEEE Transactions on Multimedia, 2025.

[18] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020.

[19] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar \` et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.

[20] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.

[21] D. Liu, M. Yang, X. Qu, P. Zhou, W. Hu, and Y. Cheng, “A survey of attacks on large vision-language models: Resources, advances, and future trends,” arXiv preprint arXiv:2407.07403, 2024.

[22] B. Wang, S. Qian, and C. Xu, “Invisible backdoor attack with siamese tuning on pre-trained vision-language models,” IEEE Transactions on Multimedia, 2025.

[23] M. Hu, Z. Li, H. Wang, Y. Wang, Y. Ding, C. Liu, and Q. Song, “Trajectory-aware attack: Explainable adversarial attack against multiple object trackers,” IEEE Transactions on Multimedia, 2026.

[24] Y. Wang, W. Hu, Y. Dong, H. Zhang, H. Su, and R. Hong, “Exploring transferability of multimodal adversarial samples for vision-language pre-training models with contrastive learning,” IEEE Transactions on Multimedia, 2025.

[25] D. Liu, M. Yang, X. Qu, P. Zhou, X. Fang, K. Tang, Y. Wan, and L. Sun, “Pandora’s box: Towards building universal attackers against real-world large vision-language models,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[26] H. Yan, H. Ma, X. Cai, D. Liu, Z. Yuan, X. Qu, J. Dong, R. Guan, X. Fang, H. He et al., “Fit the distribution: Cross-image/prompt adversarial attacks on multimodal large language models,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[27] X. Cai, D. Liu, X. Qu, X. Fang, J. Dong, K. Tang, P. Zhou, L. Sun, and W. Hu, “Towards building model/prompt-transferable attackers against large vision-language models,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[28] D. Liu, B. Chen, and W. Hu, “Spatial-spectral homogeneous attacks on physical-world large vision-language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 9, 2026, pp. 7114–7122.

[29] D. Liu, X. Cai, P. Zhou, X. Qu, L. Sun, and W. Hu, “Are large visionlanguage models robust to adversarial visual transformations?” IEEE Transactions on Information Forensics and Security, 2026.

[30] D. Liu, W. Liu, X. Cai, P. Zhou, R. Guan, X. Qu, and B. Du, “Generating transferable attacks across large vision-language models using adversarial deformation learning,” Pattern Recognition, p. 113194, 2026.

[31] E. Shayegani, Y. Dong, and N. Abu-Ghazaleh, “Jailbreak in pieces: Compositional adversarial attacks on multi-modal language models,” in The Twelfth International Conference on Learning Representations, 2023.

[32] L. Bailey, E. Ong, S. Russell, and S. Emmons, “Image hijacks: Adversarial images can control generative models at runtime,” arXiv preprint arXiv:2309.00236, 2023.

[33] Y. Dong, H. Chen, J. Chen, Z. Fang, X. Yang, Y. Zhang, Y. Tian, H. Su, and J. Zhu, “How robust is google’s bard to adversarial image attacks?” arXiv preprint arXiv:2309.11751, 2023.

[34] X. Wang, Z. Ji, P. Ma, Z. Li, and S. Wang, “Instructta: Instructiontuned targeted attack for large vision-language models,” arXiv preprint arXiv:2312.01886, 2023.

[35] Z. Wang, Z. Han, S. Chen, F. Xue, Z. Ding, X. Xiao, V. Tresp, P. Torr, and J. Gu, “Stop reasoning! when multimodal llms with chain-of-thought reasoning meets adversarial images,” arXiv preprint arXiv:2402.14899, 2024.

[36] H. Zhang, W. Shao, H. Liu, Y. Ma, P. Luo, Y. Qiao, and K. Zhang, “Avibench: Towards evaluating the robustness of large vision-language model on adversarial visual-instructions,” arXiv preprint arXiv:2403.09346, 2024.

[37] D. Lu, T. Pang, C. Du, Q. Liu, X. Yang, and M. Lin, “Test-time backdoor attacks on multimodal large language models,” arXiv preprint arXiv:2402.08577, 2024.

[38] H. Luo, J. Gu, F. Liu, and P. Torr, “An image is worth 1000 lies: Adversarial transferability across prompts on vision-language models,” arXiv preprint arXiv:2403.09766, 2024.

[39] X. Tao, S. Zhong, L. Li, Q. Liu, and L. Kong, “Imgtrojan: Jailbreaking vision-language models with one image,” arXiv preprint arXiv:2403.02910, 2024.

[40] Y. Zhao, T. Pang, C. Du, X. Yang, C. Li, N.-M. M. Cheung, and M. Lin, “On evaluating adversarial robustness of large vision-language models,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[41] A. V. Oppenheim and J. S. Lim, “The importance of phase in signals,” Proceedings of the IEEE, vol. 69, no. 5, pp. 529–541, 2005.

[42] S. Karout, Two-dimensional phase unwrapping. Liverpool John Moores University (United Kingdom), 2007.

[43] C. Guo, Q. Ma, and L. Zhang, “Spatio-temporal saliency detection using phase spectrum of quaternion fourier transform,” in 2008 IEEE conference on computer vision and pattern recognition. IEEE, 2008, pp. 1–8.

[44] J. Li, L.-Y. Duan, X. Chen, T. Huang, and Y. Tian, “Finding the secret of image saliency in the frequency domain,” IEEE transactions on pattern analysis and machine intelligence, vol. 37, no. 12, pp. 2428–2440, 2015.

[45] G. Chen, P. Peng, L. Ma, J. Li, L. Du, and Y. Tian, “Amplitude-phase recombination: Rethinking robustness of convolutional neural networks in frequency domain,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 458–467.

[46] Y. Qian, Y. Kong, Q. Bao, Z. Gu, B. Wang, S. Ji, J. Zhang, and Z. Lei, “Individual & common attack: Enhancing transferability in vlp models through modal feature exploitation,” IEEE Transactions on Image Processing, 2026.

[47] Y. Qian, X. Zhu, Q. Bao, F. Yu, S. Ji, Z. Gu, W. Wang, B. Wang, and Z. Lei, “Exploiting shared adversarial features for dynamic attacks in large vision-language models,” IEEE Transactions on Information Forensics and Security, vol. 21, pp. 592–607, 2025.

[48] Y. Qian, Q. Yu, Q. Bao, S. Ji, W. Wang, B. Wang, Z. Gu, and Z. Lei, “A multimodal adversarial attack method via frequency domain enhancement and fine-grained cross-modal guidance,” IEEE Transactions on Dependable and Secure Computing, 2025.

[49] C.-H. Lai, T.-E. Wu, and C.-C. Wang, “Enhancing information security in smart manufacturing through least significant bit steganography in engineering drawings,” Journal of Computing and Information Science in Engineering, vol. 25, no. 9, p. 091006, 2025.

[50] R. Fan, A. Boukerche, P. Pan, Z. Jin, and Y. Su, “An underwater secure localization scheme based on physical layer cryptographic learning,” IEEE Transactions on Mobile Computing, 2025.

[51] W. Zhu, B. Xu, B. Guo, C. Xie, and R. Xiong, “Data-efficient and interpretable long-term prognostics of pemfc degradation via physicsconstrained symbolic regression–physics-informed neural network,” eTransportation, p. 100615, 2026.

[52] D. Liu, X. Cai, J. Dong, Z. Guo, X. Qu, R. Guan, X. Fang, and D. Ye, “Attacking gray-box large vision-language models with adaptive svd-structured adversarial alignment,” in International Conference on Machine Learning, 2026.

[53] N. Ai, G. Chen, X. Cai, Z. Guo, D. Liu, P. Zhou, and O. Arandjelovic,´ “Attacking hard-label large vision-language models with model-sensitive adversarial patch designs,” Pattern Recognition, p. 114261, 2026.

[54] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.

[55] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, “Eva-clip: Improved training techniques for clip at scale,” arXiv preprint arXiv:2303.15389, 2023.

[56] Y. Qian, K. Chen, B. Wang, Z. Gu, S. Ji, W. Wang, and Y. Zhang, “Enhancing transferability of adversarial examples through mixed-frequency inputs,” IEEE Transactions on Information Forensics and Security, vol. 19, pp. 7633–7645, 2024.

[57] Y. Qian, J. Sha, B. Wang, Z. Gu, and Y. Zhang, “Enhancing transferability of targeted adversarial examples through amplitude spectrum alignment,” Multimedia Systems, vol. 31, no. 5, p. 345, 2025.

[58] D. Yin, R. Gontijo Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer, “A fourier perspective on model robustness in computer vision,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[59] H. Wang, X. Wu, Z. Huang, and E. P. Xing, “High-frequency component helps explain the generalization of convolutional neural networks,”

in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8684–8694.

[60] D. Liu, W. Hu, and X. Li, “Point cloud attacks in graph spectral domain: When 3d geometry meets graph signal processing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

[61] Q. Hu, D. Liu, and W. Hu, “Exploring the devil in graph spectral domain for 3d point cloud attacks,” in European Conference on Computer Vision. Springer, 2022, pp. 229–248.

[62] C. Luo, Q. Lin, W. Xie, B. Wu, J. Xie, and L. Shen, “Frequency-driven imperceptible adversarial attack on semantic similarity,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 15 315–15 324.

[63] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel, “Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness,” in International conference on learning representations, 2018.

[64] D. Zhao, A. Wang, and O. Russakovsky, “Understanding and evaluating racial biases in image captioning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 14 830–14 840.

[65] X. Cui, A. Aparcedo, Y. K. Jang, and S.-N. Lim, “On the robustness of large multimodal models against image adversarial attacks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 625–24 634.

[66] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Learning to prompt for visionlanguage models,” International Journal of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022.

[67] L. Li, H. Guan, J. Qiu, and M. Spratling, “One prompt word is enough to boost adversarial robustness for pre-trained vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 408–24 419.

[68] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.

[69] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.

[70] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu et al., “Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 24 185–24 198.

[71] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.

[72] S. Liang, J. Liang, T. Pang, C. Du, A. Liu, E.-C. Chang, and X. Cao, “Revisiting backdoor attacks against large vision-language models,” arXiv preprint arXiv:2406.18844, 2024.

[73] X. Wang, S. Wang, Z. Ge, Y. Luo, and S. Zhang, “Attention! you vision language model could be maliciously manipulated,” arXiv preprint arXiv:2505.19911, 2025.

[74] B. Zhao, B. Wu, M. He, and T. Huang, “Svit: Scaling up visual instruction tuning,” arXiv preprint arXiv:2307.04087, 2023.

[75] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, “Zero-shot text-to-image generation,” in International conference on machine learning. Pmlr, 2021, pp. 8821–8831.

[76] Anthropic, “Claude 3.5 sonnet,” June 2024. [Online]. Available: https://www.anthropic.com/news/claude-3-5-sonnet

[77] ——, “Claude 3.7 sonnet: Hybrid reasoning model,” February 2025. [Online]. Available: https://www.anthropic.com/news/claude-3-7-sonnet

[78] OpenAI, “Gpt-4o: Real-time multimodal reasoning,” May 2024. [Online]. Available: https://openai.com/blog/hello-gpt-4o

[79] ——, “Gpt-4.1: Long context and cost-efficient coding,” April 2025. [Online]. Available: https://openai.com/blog/gpt-4-1-release

[80] Google DeepMind, “Gemini 2.0: Native multimodal agents,” December 2024. [Online]. Available: https://blog.google/technology/ gemini/gemini-2-0-flash

[81] Z. Li, X. Zhao, D.-D. Wu, J. Cui, and Z. Shen, “A frustratingly simple yet highly effective attack baseline: Over 90% success rate against the strong black-box models of gpt-4.5/4o/o1,” arXiv preprint arXiv:2503.10635, 2025.

[82] X. Zhao, Z. Li, Y. Luo, J. Cui, and Z. Shen, “Pushing the frontier of black-box lvlm attacks via fine-grained detail targeting,” arXiv preprint arXiv:2602.17645, 2026.

[83] X. Jia, S. Gao, S. Qin, T. Pang, C. Du, Y. Huang, X. Li, Y. Li, B. Li, and Y. Liu, “Adversarial attacks against closed-source mllms via feature optimal alignment,” arXiv preprint arXiv:2505.21494, 2025.

[84] F. James, “Monte carlo theory and practice,” Reports on progress in Physics, vol. 43, no. 9, p. 1145, 1980.

[85] M. Zhu, S. Wei, L. Shen, Y. Fan, and B. Wu, “Enhancing finetuning based backdoor defense with sharpness-aware minimization,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4466–4477.

[86] Y. Shi, M. Du, X. Wu, Z. Guan, J. Sun, and N. Liu, “Black-box backdoor defense via zero-shot image purification,” Advances in Neural Information Processing Systems, vol. 36, pp. 57 336–57 366, 2023.

[87] R. Yan, L. Xie, J. Tang, X. Shu, and Q. Tian, “Higcin: Hierarchical graph-based cross inference network for group activity recognition,” IEEE transactions on pattern analysis and machine intelligence, vol. 45, no. 6, pp. 6955–6968, 2020.

[88] R. Yan, L. Xie, X. Shu, L. Zhang, and J. Tang, “Progressive instanceaware feature learning for compositional action recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 8, pp. 10 317–10 330, 2023.

[89] H. Qu, X. Shu, R. Yan, H. Gao, W. Wang, and J. Tang, “Spatio-temporal decoupled knowledge compensator for few-shot action recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[90] L. Xing, A. J. Wang, R. Yan, X. Shu, and J. Tang, “Vision-centric token compression in large language model,” Advances in Neural Information Processing Systems, vol. 38, pp. 33 080–33 110, 2026.

[91] L. Xing, H. Qu, R. Yan, X. Shu, and J. Tang, “Locality-aware crossmodal correspondence learning for dense audio-visual events detection,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[92] H. Qu, R. Yan, X. Shu, H. Gao, P. Huang, and G.-S. Xie, “Mvp-shot: Multi-velocity progressive-alignment framework for few-shot action recognition,” IEEE Transactions on Multimedia, 2025.