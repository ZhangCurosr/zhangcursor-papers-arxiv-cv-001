Object : toaster

Object : chair

# Beyond Attention Imbalance: Mitigating Hallucinations via Spectral Surgery

Siqi Lu <sup>1</sup> Wei Suo <sup>2</sup> <sup>3</sup> <sup>4</sup> Yongbin Zheng <sup>1</sup> Jianhang Yao <sup>1</sup> Wanying Xu <sup>1</sup> Peng Wang <sup>2</sup> <sup>3</sup> <sup>4</sup>

## Abstract

While Large Vision-Language Models (LVLMs) achieve remarkable success, hallucinations remain a significant barrier to their reliable deployment. Recent studies primarily attribute these issues to cross-modal attention imbalances; most solutions therefore focus on reweighting visual tokens or suppressing language priors. However, such approaches often overlook the spectral characteristics of the visual information flow and frequently rely on Contrastive Decoding (CD), which doubles inference time. Instead of following conventional approaches, we identify two distinct hallucination patterns—Perceptual-Semantic Dissociation and Localized Fixation—and propose FLASH (Frequency-Localized Attention SHaping), a training-free and CD-free framework. FLASH utilizes a Spectral Vortex Score to detect vision heads within multi-head attention layers and applies adaptive spectral modulation to rectify the visual information flow during decoding. Empirical results demonstrate that FLASH achieves a superior balance between performance and efficiency compared to SOTA methods.

## 1. Introduction

Large Vision-Language Models (LVLMs) (Liu et al., 2024a; Zhu et al., 2023) extend the functional capabilities of Large Language Models (LLMs), enabling scene understanding and visual reasoning. However, hallucinations remain a formidable challenge (Gunjal et al., 2024; Zhao et al., 2024; Yin et al., 2024), eroding user trust and compromising the reliability of generated content. These errors are es-

## Is there a { Object } in the image?

![](images/0e58c263ac861c4c8fa5c4757831069a36fea6ba7c3845c6128d4d998b829e73.jpg)  
Input Image

![](images/a2c62537a0169d6eb28a0fc279afa679dfcd3134e90586229059792f660dd211.jpg)

![](images/b8782064910737073c08fbe5630723681ab31b73fee6a065bcfc00e4e096696d.jpg)  
GT : Yes / Ans: Yes  
GT : No / Ans: Yes

(a) Perceptual-Semantic Dissociation  
![](images/33b6bb9ae6b8be2e96ce49b1c850694fdc5f1872cbb260e2d82230f0c49ef9dc.jpg)  
Input Image

![](images/2a70d78d86899425ea4d34eb6be71d5f4491c1b72c1270d30c510f77e34d993c.jpg)

![](images/7fba2f29189134d8eb7c05da64220c141b8be6164b589ec32aca10fb34351871.jpg)  
(b) Localized Fixation  
Figure 1. Two distinct hallucination patterns. (a) seeing but not understanding and (b) biased attention aggregation. Hallucinated responses are highlighted in red.

pecially problematic in safety-critical applications such as autonomous driving (Jiang et al., 2024; Zhao et al., 2025) and medical diagnostics (Yang et al., 2025; Xia et al., 2024).

Current mitigation studies generally follow two paradigms. The first involves supervised fine-tuning or preference alignment using high-quality datasets (Yu et al., 2024; Rafailov et al., 2024; Sun et al., 2023; Wang et al., 2024a). While effective, these approaches often suffer from the high costs of data curation and model retraining. Another paradigm focuses on post-hoc interventions during the decoding stage (Zheng & Zhang, 2025; Yin et al., 2025; Qian et al., 2026). These methods are increasingly favored for their flexibility.

Unfortunately, most existing training-free methods treat hallucinations merely as an imbalance between visual and textual attention—where dominant language priors override visual evidence (Liu et al., 2024c; Kang et al., 2025; Zhang et al., 2025). Consequently, research has focused heavily on manipulating attention weights in the spatial domain, largely overlooking the intrinsic spectral characteristics of visual information. Furthermore, the prevailing reliance on Contrastive Decoding (CD) (O’Brien & Lewis, 2023; Leng et al., 2023; Wang et al., 2024b) significantly increases the computational burden, hindering real-time deployment.

These issues motivate our study to address three questions: (1) Are hallucinations merely a byproduct of simplistic attention imbalances, or do they arise from more intricate patterns? (2) Can the frequency-domain perspective serve as a potent lens for both understanding the underlying mechanisms of hallucinations and facilitating their mitigation? (3) Can we design an effective mitigation framework that bypasses the computational overhead inherent in CD?

In this paper<sup>1</sup>, we demonstrate through statistical analysis that hallucinations are not merely a symptom of attention imbalance. Instead, they manifest in two distinct modes: Perceptual-Semantic Dissociation (PSD) and Localized Fixation (LF) (Fig. 1)—each interpretable from a frequency-domain perspective. Our findings reveal that simply amplifying spatial attention often yields marginal gains, with performance remaining heavily dependent on the expensive CD (Fig. 2). Motivated by these insights, we shift our focus to the frequency domain and propose FLASH (Frequency-Localized Attention SHaping). FLASH is a training-free framework that introduces the Spectral Vortex Score (SVS) to identify vision heads within Multi-Head Attention (MHA) layers. By applying adaptive spectral modulation to the value matrices and attention scores, FLASH rectifies distorted signals, encouraging the model to maintain granular focus on localized regions while enhancing attention to the global context. This mechanism effectively mitigates hallucinations arising from both PSD and LF. Extensive evaluations across diverse benchmarks confirm that FLASH achieves a SOTA balance between hallucination mitigation and inference efficiency.

## In summary, our main contributions are as follows:

1) We demonstrate that hallucinations in LVLMs extend beyond simple attention imbalances, manifesting as two distinct patterns: PSD and LF. Furthermore, our analysis shows that although current methods aim to optimize attention distribution during the decoding phase, they heavily rely on CD, which inevitably increases inference cost.

2) We introduce a spectral perspective to both analyze the etiology of hallucinations and design a targeted solution. To this end, we propose FLASH. To the best of our knowledge, this is the first work to leverage spectral analysis as a core mechanism for analyzing and mitigating hallucinations in LVLMs.

3) We validate FLASH across diverse benchmarks, demonstrating that it surpasses or matches SOTA CD-based methods. By bypassing CD, FLASH provides a more efficient and scalable solution for mitigating hallucinations.

Conflict of Interest Disclosure. There are no conflicts of interest in our work.

## 2. Preliminary

A typical LVLM architecture consists of three core components: a visual encoder, a modality projection layer, and an LLM backbone (Liu et al., 2024a; Zheng & Zhang, 2025). Both the visual encoder and the LLM are typically Transformer-based. The projection layer acts as a bridge, mapping encoded visual features into the text embedding space to ensure cross-modal alignment. These visual tokens are then concatenated with textual tokens to form a unified input sequence, from which the LLM generates responses in an autoregressive manner.

A standard LLM decoder consists of L stacked MHA layers, each containing H attention heads. For the h-th head in the l-th layer, the attention mechanism is defined as:

$$
S _ { l , h } ( Q _ { l , h } , K _ { l , h } ) = \frac { Q _ { l , h } K _ { l , h } ^ { T } } { \sqrt { d _ { k } } } ,\tag{1}
$$

$$
A _ { l , h } ( Q _ { l , h } , K _ { l , h } , V _ { l , h } ) = \mathrm { s o f t m a x } ( S _ { l , h } ) \cdot V _ { l , h } ,\tag{2}
$$

where $Q , K , V \in \mathbb { R } ^ { n \times d _ { k } }$ represent the query, key, and value matrices, respectively; $d _ { k }$ is the head dimension, and n denotes the sequence length.

The outputs from all heads are concatenated and transformed via a linear projection to produce the layer’s final representation. After L successive blocks, the model yields the final layer hidden state $h _ { L }$ . During generation of the s-th token, the hidden state $h _ { L , \varepsilon }$ <sub>s</sub> is mapped to the vocabulary logit space through a linear layer ϕ, yielding the conditional probability distribution:

$$
y _ { t } \sim \operatorname { s o f t m a x } ( h _ { L , s } \cdot \phi ) .\tag{3}
$$

## 3. Spectral Diagnosis of Hallucinations

## 3.1. Revisiting Attention Imbalance

Recent studies have identified a striking disparity in LVLMs: despite visual tokens numerically dominating the input, the attention weights allocated to the visual modality remain low (Kang et al., 2025; Liu et al., 2024c). This imbalance is widely believed to induce an over-reliance on language priors, causing the model to neglect grounded visual evidence. Leading approaches, such as PAI (Liu et al., 2024c), attempt to counteract this by amplifying visual attention scores to create a vision-augmented branch and subsequently employ CD to prioritize grounded logits. However, beyond the efficiency penalties arising from CD-induced inference overhead, it remains to be determined whether rescaling visual weights yields robust, independent improvements or whether its perceived efficacy is merely an artifact of the CD process.

![](images/5ca88e608853533f04f42e90f0a044719935dd6851d18f3b2b635963cc5b9067.jpg)  
(a)

![](images/779b69149cb2f045c0f87eeabfe8c6cbd80f3942de5f36038d78e1f6d53decdd.jpg)  
(b)  
Figure 2. Comparison of different methods for mitigating hallucinations. Results are reported relative to Vanilla LLaVA-1.5 on the POPE (COCO-R) dataset. (a) For fixed-weight configurations, a gain coefficient of 1.5 is applied to visual tokens. (b) Circle size indicates the inference time.

To respond to these questions, we evaluated discriminative performance on the POPE benchmark (Li et al., 2023b) under several configurations: the vanilla baseline, the vanilla PAI, PAI without CD (w/o CD), and PAI with fixed linear gains applied to visual tokens.

As shown in Fig. 2, our results reveal that linear amplification of visual attention in the spatial domain is insufficient for robust hallucination mitigation and can even be counterproductive. The observed performance gains of existing methods are primarily attributable to the CD rather than to the attention manipulation itself. Our results suggest that achieving low-cost mitigation requires shifting from coarsegrained scaling to a more in-depth analysis of information flow within the visual token, thereby bypassing the overhead of CD.

## 3.2. Deconstructing the Hallucination Patterns

To elucidate the manifestations of hallucinations at the visual token level, we conducted diagnostic visualizations on POPE. By extracting attention weights from the 16th MHA layer and reshaping the visual tokens into 2D spatial maps (Liu et al., 2025), we performed statistical experiments comparing the attention distributions of grounded responses versus hallucination responses. The results reveal two distinct hallucination patterns in LVLMs:

(1) Perceptual-Semantic Dissociation (PSD). This pattern signifies a rupture between spatial localization and semantic interpretation. The model successfully ”anchors” its attention to the correct visual region but fails to execute fine-grained semantic parsing. As shown in Fig. 1a, when queried about the ”bench” and the ”chair,” the model’s attention consistently converges on the exact same spatial coordinates (the ground-truth location of the bench). This suggests that although the model possesses the perceptual capacity to localize regions based on textual prompts, it lacks the discriminative resolution to distinguish fine-grained features (e.g., the elongated form of a bench), resulting in a semantic best guess rather than grounded reasoning.

![](images/a2498de25dc53d289890db63c7c74d1e77077b6627a3bc089dbc574eab8a4b6e.jpg)  
(a)

![](images/2aa57534eb56b74ff8aff8a292ac0c961ac0eb93f5114acebb1de6064d2a6ba0.jpg)  
(b)  
Figure 3. Impact of low-pass filtering on performance and spectral energy distribution. (a) Impact of different frequency band information on model performance. (b) High-frequency energy ratio of the Value Matrix and Attention Weights across layers, with the high-frequency cutoff fixed at 0.5. We employ random sampling in this experiment.

![](images/675e5f4f7f3d6ba7755c27afbca37023ed813ebabcb25fceffc64c6b9cea8617.jpg)  
(a)

![](images/f46ce463b7d70b26872ff90f15d021d35032ed1c0f7467cfab0b840ae98c76b8.jpg)  
(b)  
Figure 4. High-frequency (HF) energy analysis of queries and keys. (a) Comparison of HF energy distributions between hallucinatory and non-hallucinatory samples. (b) HF energy match rate, defined as the ratio $\mathrm { H F } ( q _ { \mathrm { t e x t } } )$ $\mathrm { H F } ( \bar { k } \mathrm { v i s i o n } )$

(2) Localized Fixation (LF). This pattern manifests as a pathological capture of attention by irrelevant visual locations. As shown in Fig. 1b, the model frequently becomes trapped within contiguous regions that offer marginal semantic utility for the current token generation. Although similar to the ”attention sink” (Xiao et al., 2024; Kang et al., 2025), LF is distinguished by its intense spatial contiguousness—attention is concentrated in dense, localized patches rather than distributed across sparse outliers. This indicates that the model is not merely ignoring visual tokens but is actively distracted by localized noise, which obscures its contextual observational perspective and induces a drift toward hallucinated responses.

Consequently, an indiscriminate ”one-size-fits-all” enhancement of visual tokens is inadequate for addressing the diverse modes of hallucinations. It is necessary to establish a unified framework to tackle different hallucination patterns.

![](images/b7a877f65af6d6e9c56c8a4069dd1e4cec3430268387062a3b714f1775edd9b1.jpg)  
Figure 5. Overview of the FLASH framework. FLASH rectifies hallucinatory signals via spectrum modulation of attention scores and value matrices within the MHA layers. Specifically, spectral shaping of scores is designed to broaden contextual coverage (seeing comprehensively), while modulation of value matrices aims to enhance perceptual fidelity (seeing clearly).

## 3.3. Spectral Analysis of Information Flow within MHA

Spectral components serve distinct roles. To quantify the impact of spectral composition on model performance, we performed spectral perturbation analysis by modulating the cutoff frequency of a low-pass filter applied to the hidden states. As shown in Fig. 3a, the model exhibits nonlinear sensitivity to filter cutoff: performance remains stable when high-frequency components remain below 0.4, begin to decline at 0.5, and collapse by 0.8. This suggests that low-frequency components underpin the model’s baseline reliability, whereas high-frequency components govern its performance bottlenecks.

To design precise interventions for diverse hallucination patterns, we dissect two critical components of MHA: the value matrix, which encodes the visual content, and the interaction between text queries $( Q ^ { t } )$ and visual keys $( K ^ { v } )$ , which determines spatial attention priority. Our spectral analysis yields the following insights:

The MHA filters out high-frequency information from visual tokens. Although the low-pass filtering nature of MHA in Transformers is well-documented (Park & Kim, 2022; Dong et al., 2023), its relationship with hallucinations still requires further exploration. We conducted a comparative analysis of high-frequency energy proportions between the value matrix and the attention output. As shown in Fig. 3b, a gap persists between the raw visual features and their aggregated representations across all layers. This observation experimentally confirms that MHA induces an irreversible smoothing of fine-grained visual details during cross-modal alignment. Although the mutations in attention output in deeper layers suggest a compensatory attempt to capture localized high-frequency structures, the energy ratio of the output consistently fails to recover the original richness of the value matrix. This indicates that hallucinations arise from the attention mechanism’s inability to propagate the precise visual evidence required for reasoning, which directly corroborates our observed PSD hallucinations.

Spectral misalignment between $Q ^ { t }$ and $K ^ { v }$ serves as the mechanistic driver of LF. Fig. 4a shows that while $K ^ { v }$ maintains stable high-frequency energy, hallucinatory $Q ^ { t }$ exhibits a significant high-frequency deficit in mid-to-late layers, resulting in a suppressed matching rate after Layer 3 (Fig. 4b). This indicates that sharp, high-frequency $K ^ { v }$ act as deceptive interference sources. When $Q ^ { t }$ lack the spectral precision to align with correct anchors, attention is easily captured by these local noise anchors, leading to over-fixation on incorrect tokens (e.g., ”toaster” in Fig. 1b).

## 4. Proposed Methods

## 4.1. Overview

Fig. 5 illustrates the overview of FLASH. Designed to operate within the decoder, FLASH adaptively modulates the spectral components of both attention scores and value matrices to enable global context integration while preserving high-fidelity visual perception. FLASH first selects vision heads within the MHA layers to ensure precise intervention.

![](images/98d01a7cb3b2e4cb2b751d5722dc543c537a83fbb8294c733fc2759cf3d828af.jpg)  
Figure 6. Spatial distributions and log-magnitude spectra of vision heads compared with those of other heads.

To counteract the hallucination patterns described in Sec. 3.2, FLASH introduces a dual-stream spectral modulation mechanism: shaping the spectral representation of attention scores to foster comprehensive contextual awareness, and refining the spectral representation of value matrices to amplify object-relevant tokens. FLASH achieves these enhancements through a single inference pass.

## 4.2. Spectrum-Based Vision Head Selection

In the MHA layers of LVLMs, individual heads exhibit functional heterogeneity (Zheng et al., 2024; Zhang et al., 2024). Applying modulation indiscriminately across all heads is suboptimal, as it may compromise linguistic coherence in the generated content and introduce additional computational costs. In FLASH, we distinguish vision-related heads $( \mathcal { H } ^ { v } )$ from the remaining $( \mathcal { H } ^ { o } )$ by characterizing the spectral signatures of their attention weights.

Qualitative Analysis. We performed a qualitative analysis to characterize the spectral discrepancies between $\mathcal { H } ^ { v }$ and $\mathcal { H } ^ { o }$ . Reconstructing and profiling the attention weight spectra reveals pronounced disparities between $\mathcal { H } ^ { v }$ and $\mathcal { H } ^ { o }$ (Fig. 6). These observations lead to two key insights:

(1) $\mathcal { H } ^ { v }$ exhibits higher spectral energy concentration. Recent studies (Kang et al., 2025) have identified ”attention sinks” within visual tokens, analogous to those observed in textual sequences. These sinks are markedly more pronounced in $\mathcal { H } ^ { o }$ , where they manifest spatially as sparse, high-magnitude singularities resembling Dirac delta functions. From a signal processing perspective, such impulselike signals possess broadband characteristics, causing their spectral energy to disperse across the entire frequency domain rather than concentrate within the low-frequency components. Consequently, $\mathcal { H } ^ { o }$ manifests a more diffuse energy

distribution compared to $\mathcal { H } ^ { v }$

(2) $\mathcal { H } ^ { v }$ displays structured vortex-like spectra, whereas $\mathcal { H } ^ { o }$ exhibits periodic ripples. In the spatial domain, the dominance of sparse sink tokens in $\mathcal { H } ^ { o }$ introduces multiple localized peaks, which generate periodic ripples across the spectrum. In contrast, $\mathcal { H } ^ { v }$ maintains consistent activation across spatially contiguous semantic regions. This continuous spatial activation suppresses the spectral dominance of individual sink tokens. Due to the inherent conjugate symmetry of the FFT, the spectra of $\mathcal { H } ^ { v }$ manifest as vortex-like structures.

Quantitative Assessment. Leveraging these qualitative insights, we introduce the Spectral Vortex Score (SVS), a composite metric designed to quantitatively identify $\mathcal { H } ^ { v }$

Given the attention weights for reshaped visual tokens $A _ { l , h } ^ { v } \in \mathbb { R } ^ { \sqrt { N } \times \sqrt { N } }$ for the h-th head in layer l (where N is the visual token length), we project them into the frequency domain via FFT and compute the log-magnitude spectrum M:

$$
M ( p , q ) = 2 0 \log _ { 1 0 } ( | \mathcal { F } ( A ( x , y ) ) | + \epsilon ) ,\tag{4}
$$

where $\mathcal { F }$ denotes the FFT operator and ϵ is a small positive constant. We omit subscripts for brevity. Our SVS consists of two components:

Spectral Energy Concentration. According to our first observation, $\mathcal { H } ^ { v }$ captures structured semantic information, which manifests as energy concentration in the central lowfrequency region. We define a low-frequency mask $\mathcal { M } _ { L }$ with radius R (in this paper, we set $R = 1 )$

$$
\mathcal { M } _ { L } ( p , q ) = \left\{ \begin{array} { c c } { 1 , } & { \mathrm { i f ~ } \sqrt { ( p - p _ { c } ) ^ { 2 } + ( q - q _ { c } ) ^ { 2 } } \leq R } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{5}
$$

where $( p _ { c } , q _ { c } )$ denotes the pixel coordinates of the spectral DC. The concentration score $\mathcal { L } _ { r }$ can be defined as:

$$
\mathcal { L } _ { r } = \frac { \sum _ { p , q } M ( p , q ) \cdot \mathcal { M } _ { L } ( p , q ) } { \sum _ { p , q } M ( p , q ) } .\tag{6}
$$

Spectral Isotropy. Building on our second observation, $\mathcal { H } ^ { o }$ exhibits highly directional ripples due to periodic impulses from attention sinks, whereas $\mathcal { H } ^ { v }$ displays isotropic, vortexlike structures. We employ orientation coherence to quantify this structural disparity. Computing the gradient field of the magnitude spectrum $\nabla M = \left( G _ { p } , G _ { q } \right)$ , we further obtain:

$$
G _ { m a g } = \sqrt { G _ { p } ^ { 2 } + G _ { q } ^ { 2 } } , \qquad \theta = \mathrm { a t a n 2 } ( G _ { q } , G _ { p } ) .\tag{7}
$$

We formulate the weighted orientation coherence to measure the strength of the dominant orientation:

$$
\mathcal { C } = \frac { \sqrt { \left( \sum G _ { m a g } \cos ( 2 \theta ) \right) ^ { 2 } + \left( \sum G _ { m a g } \sin ( 2 \theta ) \right) ^ { 2 } } } { \sum G _ { m a g } } ,\tag{8}
$$

where $\mathcal { C } \to 1$ indicates a highly anisotropic spectrum. We thus define the spectral isotropy as $\mathcal { O } _ { c } = 1 - \mathcal { C }$

The SVS is the sum of these normalized components:

$$
\mathrm { S V S } = \mathcal { L } _ { r } + \mathcal { O } _ { c } ,\tag{9}
$$

where a higher SVS indicates a larger probability of $\mathcal { H } ^ { v }$ Inspired by (Kang et al., 2025), we first exclude heads with an aggregate attention weight below threshold τ, and then identify the top k heads with the highest SVS values as $\mathcal { H } ^ { v }$

## 4.3. Dual-stream Spectrum Modulation

To mitigate the diverse patterns of object hallucinations, we introduce Dual-stream Spectrum Modulation (DSM), a strategy designed to adaptively modulate the information flow of visual tokens in the decoder. Diverging from spatialdomain reweighting (Liu et al., 2024c; Zhang et al., 2025), DSM performs adaptive redistribution of spectral energy within both the attention scores and value matrices. By operating in the spectral domain, DSM circumvents the heavy computational overhead of CD and mitigates noise-induced interference common in spatial manipulations. Furthermore, this approach enhances the interpretability of the model’s internal representations.

DSM comprises two parallel branches—the V-stream and the S-stream—which respectively modulate the value matrices (V) and the attention scores (S) within MHA layers. We begin by employing the Discrete Cosine Transform (DCT) for frequency-domain projection, leveraging its superior energy harvesting capabilities. For a spatially reshaped feature map $X ^ { m } \in \bar { \mathbb { R } } ^ { \sqrt { N } \times \sqrt { N } }$ (where $m \in \{ V , S \} ,$ ), the spectral coefficients $C ( p , q )$ are computed as:

$$
C ^ { m } ( p , q ) = { \cal D } ( X ^ { m } ) ,\tag{10}
$$

where $\mathcal { D } ( \cdot )$ denotes the DCT operator; $C ^ { m } ( 0 , 0 )$ is the DC component, while coefficients with higher indices $( p , q )$ represent higher spatial frequencies.

We use a radial distance matrix $D _ { p , q }$ to characterize the spectral distribution, which will be employed in our DSM:

$$
D _ { p , q } = \sqrt { p ^ { 2 } + q ^ { 2 } } , \quad p , q \in \{ 0 , 1 , \ldots , \sqrt { N } - 1 \} .\tag{11}
$$

V-stream. PSD occurs when a model identifies relevant regions based on queries but fails to capture fine-grained details. This suggests that although the model leverages low-frequency components for localization, its sensitivity to granular semantic features is limited. Prior research (Wei et al., 2021; Wang et al., 2020; Xu et al., 2020) indicates that such fine-grained information is predominantly embedded in high-frequency components. Unfortunately, our findings in Sec. 3.3 reveal that the inherent low-pass filtering property of MHA attenuates these high-frequency cues in the value matrix, thereby weakening the model’s perceptual acuity.

By enhancing the high-frequency energy of the value matrix through adaptive spectral modulation, we can counteract this collapse. We first construct a soft mask $\mathcal { M } _ { v } \mathrm { : }$

$$
\mathcal { M } _ { v } ( p , q ) = 1 + \lambda _ { v } \cdot \mathrm { s i g m o i d } ( D _ { p , q } - \alpha \sqrt { N } ) ,\tag{12}
$$

where $\lambda _ { v }$ denotes the modulation strength and α is the frequency cutoff threshold. We fix $\mathcal { M } _ { v } ( 0 , 0 ) = 1$ to preserve the DC component. We perform the intervention in the log-magnitude domain—a decision predicated on the powerlaw distribution of visual features (Field & David, 1987; Simoncelli & Olshausen, 2001):

$$
\tilde { C } ^ { V } = \mathrm { s g n } ( C ^ { V } ) \cdot e ^ { \log | C ^ { V } | \cdot \mathcal { M } _ { v } } ,\tag{13}
$$

where $\operatorname { s g n } ( { \mathord { \cdot } } )$ is the sign function, ensuring correctness of all DCT coefficients. We further discuss the advantages of logarithmic modulation in Appendix A.3 and A.5.

The weighted spectrum is mapped back to the spatial domain using the IDCT, followed by an adaptive energy modulation step to calibrate the resulting features:

$$
\tilde { V } = \mathrm { I D C T } ( \tilde { C } ^ { V } ) ,\tag{14}
$$

$$
V ^ { * } = \tilde { V } \cdot \frac { | | V | | _ { F } } { | | \tilde { V } | | _ { F } + \epsilon } .\tag{15}
$$

The naive scaling of distinct frequency bands poses a substantial risk of destabilizing the joint distribution of multimodal tokens, often necessitating delicate hyperparameter tuning. In contrast, our proposed modulation—leveraging the synergy between Eq. 13 and Eq. 15—acts as a spectral energy reallocation mechanism. According to Parseval’s Theorem <sup>2</sup>, enforcing a constant F-norm constraint ensures that amplification of high-frequency components inherently squeezes redundant energy from low-frequency bands and reallocates it to suppressed fine-grained details. This intrinsic zero-sum energy dynamic enables the model to refocus on fine-grained semantic features without shifting global feature magnitudes or compromising pre-trained multimodal alignment. Please refer to Appendix A.3 for theoretical details. Consequently, this modulation provides a robust, plug-and-play solution for mitigating PSD while preserving numerical stability in the representation space.

S-stream is specifically designed to counteract LF, a phenomenon characterized by pathological over-concentration of spatial attention on irrelevant visual regions. Since attention scores govern the allocation of visual priorities in LVLMs, we refine their spectral energy distribution to enhance global contextual awareness and prevent the model from being distracted by localized noise.

Similar to Eq. 12, we construct a spectral soft mask $\mathcal { M } _ { s }$ to modulate the S-stream:

$$
{ \mathcal { M } } _ { s } ( p , q ) = ( 1 - \lambda _ { s } ) + \lambda _ { s } \cdot { \mathrm { s i g m o i d } } ( \alpha { \sqrt { N } } - D _ { p , q } ) ,\tag{16}
$$

where $\lambda _ { s }$ denotes the modulation strength and α is the frequency cutoff threshold. We also strictly enforce $\mathcal { M } _ { s } ( 0 , 0 ) = 1$ to preserve the DC component.

In contrast to the V-stream, the S-stream operates directly in the linear spectral domain. This design prevents excessive distortion of the attention distribution during the subsequent softmax, which could otherwise destabilize model convergence. The weighted spectral coefficients are obtained via the Hadamard product: $\tilde { C } ^ { S } = C ^ { S } \odot \mathcal { M } _ { s }$

The weighted coefficients are then projected back into the spatial domain via the IDCT, followed by norm calibration:

$$
\tilde { S } = \mathrm { I D C T } ( \tilde { C } ^ { S } )\tag{17}
$$

$$
S ^ { * } = \tilde { S } \cdot \frac { | | S | | _ { F } } { | | \tilde { S } | | _ { F } + \epsilon }\tag{18}
$$

Similar to V-stream, the coordination of Eq. 16 and Eq. 18 adaptively ”pushes” energy suppressed in high-frequency bands toward low-frequency regions, thereby achieving a compensatory redistribution of spectral energy. In summary, the S-stream acts as a spectral regularizer that immunizes the model against localized artifacts and facilitates more comprehensive perception of visual tokens.

Finally, the attention output $A ^ { * }$ is calculated from the modulated attention score $S ^ { * }$ and the value matrix $V ^ { * }$ :

$$
A ^ { * } = \operatorname { s o f t m a x } ( S ^ { * } ) \cdot V ^ { * } ,\tag{19}
$$

## 5. Experiment

## 5.1. Experimental Setting

Datasets and Evaluation Metrics. We conducted experiments on both discriminative and generative tasks to demonstrate the effectiveness of FLASH. Following previous work, we use POPE (Li et al., 2023b) and MME (Fu et al., 2025) for discriminative benchmarks, while CHAIR (Rohrbach et al., 2019) and AMBER (Wang et al., 2023) are used for generative evaluation. Performance on discriminative tasks is quantified using Accuracy, F1-score, and Score. For generative tasks, we report CHAIR<sub>S</sub>, CHAIR<sub>I</sub>, Cover, Hal, and Cog (Wang et al., 2023) scores to measure hallucination tendencies. Please refer to Appendix A.8.1 for further details.

Baselines. We selected two representative families of LVLMs as base models: LLaVA-1.5 (Liu et al., 2024a) and Shikra (Chen et al., 2023a). We employed the 7B and 13B variants of LLaVA-1.5 to assess the adaptability across different model scales. To demonstrate the performance and efficiency of FLASH, we compared it with three SOTA methods: VCD (Leng et al., 2023), PAI (Liu et al., 2024c), and SID (Huo et al., 2025). Further details can be found in Appendix A.8.2.

Implementation Details. All experiments were conducted on two NVIDIA RTX 4090 GPUs. Following the analysis in Fig. 3a, the high-frequency cutoff was set to 0.5 across all LVLMs. To accommodate the diverse visual token lengths, decoder architectures of different models, and task requirements, we adjusted the modulation strengths $\lambda _ { v }$ and $\lambda _ { s } .$ , as well as the vision head selection parameters k and $\tau ,$ for each specific LVLM. Detailed configurations are provided in Appendix A.8.3.

## 5.2. Quantitative Evaluation

Table 1 summarizes the performance across multiple benchmarks, where FLASH outperforms other methods on most metrics. FLASH demonstrates more significant gains on discriminative benchmarks than on generative tasks, highlighting its robust spatial awareness. Although it is suboptimal on a few generative metrics, FLASH maintains overall competitive performance. Additionally, the inference efficiency of different methods is detailed in Appendix A.6. Detailed task-level scores for MME and qualitative visualizations of the generative task are provided in Appendix A.5 and Appendix A.7, respectively. In summary, these results indicate that FLASH achieves a better balance between performance and efficiency in mitigating hallucinations.

## 5.3. Ablation Study

In this section, we report the primary ablation study results.   
Please refer to the Appendix A.5 for additional experiments.

The ablation results in Table 2 demonstrate that each component is indispensable to the model’s overall performance. Specifically, visual head selection enables targeted modulation, thereby preserving the integrity of language heads. Furthermore, the two streams within the DSM yield independent performance gains, emphasizing their specialized roles in mitigating distinct hallucination patterns. Ultimately, the integrated DSM operates synergistically to tackle diverse modes, achieving the most robust performance.

## 6. Related Work

Data-Driven Instruction Tuning methods focus on improving data quality and using specialized instruction tuning to better anchor models in visual evidence. Early research suggested that hallucinations often stem from a mismatch between visual features and the model’s language priors. LLaVA-1.5 (Liu et al., 2024a) demonstrated that high-quality data coupled with a simplified linear projector is more effective than dataset size in mitigating hallucinations. ShareGPT4V (Chen et al., 2023b) extended this by using detailed image descriptions generated by proprietary models to sharpen the model’s perception of fine-grained attributes. To address ”over-generalization,” RLHF-V (Yu et al., 2024) introduced segment-level corrective feedback via reinforcement learning, whereas Silkie (Li et al., 2023a) and LLaVA-RLHF (Sun et al., 2023) adopted Direct Preference Optimization (Rafailov et al., 2024) to encourage factual consistency. More recently, Bunny (He et al., 2024) and LLaVA-NeXT (Li et al., 2024; Liu et al., 2024b) identified low spatial resolution as a key driver of errors in small-object recognition, leading them to advocate for higher-resolution visual inputs. Although these methods yield significant improvements, they remain constrained by the high cost of data collection, complex training pipelines, and heavy computational requirements.

Table 1. Comparison with SOTA methods. Rows shaded in green denote the performance of vanilla LVLMs, while subsequent rows report results after integrating various hallucination mitigation methods into these models. The evaluation spans both discriminative benchmarks (POPE and MME) and generative benchmarks (CHAIR and AMBER). For each LVLM, the best and second-best results among the mitigation methods are highlighted in pink and purple, respectively. All experiments employed greedy decoding.
<table><tr><td rowspan="3">LVLMs</td><td colspan="4">POPE-MS-COCO</td><td rowspan="2">MME</td><td rowspan="2"></td><td rowspan="2">CHAIR</td><td colspan="3" rowspan="2">AMBER</td></tr><tr><td>Random</td><td>Popular</td><td></td><td>Adversarial</td></tr><tr><td></td><td>Acc. ↑ F1↑</td><td>Acc.</td><td>F1 Acc.</td><td>F1</td><td>Score ↑</td><td>CHAIRsS</td><td>CHAIR1</td><td>Cover ↑ Hal↓</td><td>Cog ↓</td></tr><tr><td>LLaVA-1.5 7B</td><td>88.97 88.90</td><td></td><td>85.63 86.03</td><td>79.23 80.99</td><td></td><td>621.67</td><td>48.10 12.75</td><td>51.00</td><td>30.50</td><td>3.20</td></tr><tr><td>+VCD</td><td>89.07</td><td>88.99</td><td>85.6085.99</td><td>79.27 81.00</td><td></td><td>636.67</td><td>48.80 12.85</td><td>51.55</td><td>27.00</td><td>2.85</td></tr><tr><td>+PAI</td><td>89.30</td><td>89.27</td><td>86.07 86.45</td><td>79.23</td><td>81.06</td><td>636.67</td><td>47.80 12.35</td><td>47.10</td><td>24.25</td><td>1.45</td></tr><tr><td>+SID</td><td>89.40</td><td>89.04</td><td>85.93 85.93</td><td>80.3381.38</td><td></td><td>606.67</td><td>48.10 12.40</td><td>52.40</td><td>30.50</td><td>2.15</td></tr><tr><td>+Ours</td><td>90.03</td><td>89.77</td><td>86.53 86.62</td><td>80.5381.75</td><td></td><td>636.67</td><td>47.50 12.55</td><td>51.00</td><td>25.00</td><td>2.15</td></tr><tr><td>Shikra</td><td>84.67</td><td>84.58</td><td>82.1082.43</td><td>77.93 79.20</td><td></td><td>458.33</td><td>49.20 13.80</td><td>52.75</td><td>42.00</td><td>3.65</td></tr><tr><td>+VCD</td><td>84.67</td><td>84.51 83.03</td><td>83.12</td><td>78.30</td><td>79.38</td><td>463.33</td><td>47.70 13.70</td><td>52.35</td><td>38.75</td><td>3.45</td></tr><tr><td>+PAI</td><td>85.20</td><td>84.37</td><td>82.2381.78</td><td>78.93</td><td>79.10</td><td>461.67</td><td>50.40 14.10</td><td>51.65</td><td>38.25</td><td>3.20</td></tr><tr><td>+SID</td><td>83.27</td><td>82.37</td><td>82.83 81.99</td><td>78.33</td><td>78.29</td><td>473.33</td><td>50.60 13.80</td><td>53.20</td><td>35.75</td><td>2.60</td></tr><tr><td>+Ours</td><td>86.03</td><td>84.99 83.30</td><td>82.56</td><td>79.80</td><td>79.65</td><td>468.33</td><td>49.60 14.05</td><td>52.65</td><td>35.75</td><td>3.10</td></tr><tr><td>LLaVA-1.5 13B</td><td>89.73</td><td>89.99</td><td>85.80 86.66</td><td>80.47</td><td>82.53</td><td>598.33</td><td>42.80 11.90</td><td>50.30</td><td>26.50</td><td>2.35</td></tr><tr><td>+VCD</td><td>89.77</td><td>90.00</td><td>85.73 86.59</td><td>80.57</td><td>82.58</td><td>606.67</td><td>42.20 12.00</td><td>50.50</td><td>26.25</td><td>2.55</td></tr><tr><td>+PAI</td><td>90.03</td><td>90.20</td><td>86.17 86.90</td><td>81.07</td><td>82.89 81.83</td><td>598.33 596.67</td><td>37.60 10.50</td><td>49.50</td><td>28.50</td><td>1.70</td></tr><tr><td>+SID</td><td>90.17</td><td>89.80</td><td>83.60 84.06</td><td>80.80</td><td>80.8082.70</td><td>601.67</td><td>41.20 10.90 42.70 11.75</td><td>49.60 50.45</td><td>26.00</td><td>2.70</td></tr><tr><td>+Ours</td><td>90.60</td><td>90.71</td><td>86.7087.35</td><td></td><td></td><td></td><td></td><td></td><td>23.00</td><td>2.25</td></tr></table>

Table 2. Comparison of ablation results. ”Select” denotes the visual head selection mechanism. † denotes using only the modulation strategy corresponding to that stream. We report the average results across the three splits of POPE-COCO.
<table><tr><td rowspan="2">Methods</td><td colspan="2">POPE</td><td colspan="3">AMBER</td></tr><tr><td>Acc.↑</td><td>F1↑</td><td>Cover↑</td><td>Hal↓</td><td>Cog↓</td></tr><tr><td>Vanilla</td><td>84.61</td><td>85.31</td><td>51.00</td><td>30.50</td><td>3.20</td></tr><tr><td>w/o Select</td><td>85.69</td><td>85.89</td><td>51.85</td><td>25.25</td><td>2.20</td></tr><tr><td>w/ Select</td><td>85.70</td><td>86.05</td><td>51.00</td><td>25.00</td><td>2.15</td></tr><tr><td>S-stream† V-stream†</td><td>85.38 85.55</td><td>85.73 85.91</td><td>50.50 51.15</td><td>25.25 25.50</td><td>2.30 2.20</td></tr></table>

Training-Free Decoding Intervention. Prior research (Liu et al., 2024c; Leng et al., 2023) indicates that hallucinations in LVLMs during autoregressive generation primarily stem from an over-reliance on linguistic priors, which hinders the model’s ability to ground its outputs in actual visual evidence. To bypass the costs of retraining, the research community has increasingly turned to plug-and-play interventions applied during the decoding phase. Early schemes focused on analyzing spatial attention distributions; for instance, OPERA (Huang et al., 2024) identified the over-trust pitfall where models fixate on specific tokens. Building on CD, VCD (Leng et al., 2023) introduced Gaussian noise to construct contrastive samples that counteract linguistic bias. This paradigm has inspired numerous variants: PAI (Liu et al., 2024c) amplifies visual grounding by modulating attention scores; DoLA (Chuang et al., 2024) and ICD (Wang et al., 2024b) mitigate signal submergence through layerwise contrast or instruction-based bias calibration. Similarly, M3ID (Favero et al., 2024), IBD (Zhu et al., 2024), and Octopus (Suo et al., 2025) have refined the CD framework through multi-layer feature contrast and strategic integration.

Despite performing well, CD-based methods are inherently limited by a doubling of inference latency. Furthermore, the introduction of contrastive noise can occasionally compromise generative quality (Huo et al., 2025). Recent efficient alternatives like VAF (Yin et al., 2025), ADAPTVIS (Chen et al., 2025), and VAR (Kang et al., 2025) have begun exploring internal mechanisms—such as energy distribution or logit calibration—to bypass the dual-stream architecture.

In this paper, we depart from conventional spatial-domain manipulations by pioneering a spectral analysis of hallucination dynamics.

## 7. Conclusion & Discussion

In this paper, we demonstrate that the effectiveness of recent post-hoc hallucination mitigation methods heavily relies on CD strategies, which incur a substantial computational penalty. By analyzing the spatial attention representations of visual tokens and their spectral features, we refine the types of hallucinations into PSD and LF. To address these, we introduce FLASH, which is a training-free framework that mitigates hallucinations from a spectral perspective. FLASH incorporates two core components: vision head selection and dual-stream adaptive spectral modulation. Experimental results confirm that FLASH outperforms SOTA methods in balancing performance and efficiency, successfully bypassing the standard CD paradigm through targeted processing of distinct hallucinatory patterns.

Despite its efficacy, FLASH entails certain technical limitations. First, although it circumvents the CD paradigm and significantly reduces computational complexity, FLASH still relies on multiple spectral transformations and filtering operations. Second, future research may enable targeted modulation of hallucinations by identifying their underlying patterns—thereby further minimizing inference latency. Third, FLASH simultaneously applies value modulation and attention modulation; decoupling or selectively applying these modulations could yield more targeted mitigation strategies while accelerating overall processing. Finally, due to FLASH’s analytical design principles, it remains unclear whether it is applicable to Q-Former-based models. We will prioritize these directions in our future work.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgment

This work was supported in part by the National Natural Science Foundation of China under Grant 62273353.

## References

Chen, K., Zhang, Z., Zeng, W., Zhang, R., Zhu, F., and Zhao, R. Shikra: Unleashing multimodal llm’s referential dialogue magic, 2023a. URL https://arxiv.org/ abs/2306.15195.

Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., and Lin, D. Sharegpt4v: Improving large multimodal models with better captions, 2023b. URL https: //arxiv.org/abs/2311.12793.

Chen, S., Zhu, T., Zhou, R., Zhang, J., Gao, S., Niebles, J. C., Geva, M., He, J., Wu, J., and Li, M. Why is spatial reasoning hard for vlms? an attention mechanism perspective on focus areas, 2025. URL https: //arxiv.org/abs/2503.01773.

Cho, Y., Kim, K., Hwang, T., and Cho, S. Do you keep an eye on what i ask? mitigating multimodal hallucination via attention-guided ensemble decoding, 2025. URL https://arxiv.org/abs/2505.17529.

Chuang, Y.-S., Xie, Y., Luo, H., Kim, Y., Glass, J., and He, P. Dola: Decoding by contrasting layers improves factuality in large language models, 2024. URL https: //arxiv.org/abs/2309.03883.

Dong, Y., Cordonnier, J.-B., and Loukas, A. Attention is not all you need: Pure attention loses rank doubly exponentially with depth, 2023. URL https://arxiv.org/ abs/2103.03404.

Favero, A., Zancato, L., Trager, M., Choudhary, S., Perera, P., Achille, A., Swaminathan, A., and Soatto, S. Multimodal hallucination control by visual information grounding, 2024. URL https://arxiv.org/abs/2403. 14003.

Field and David, J. Relations between the statistics of natural images and the response properties of cortical cells. Journal ofthe Optical Society ofAmerica A-optics Image Science & Vision, 4(12):2379–2394, 1987.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., Ji, R., Shan, C., and He, R. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2025. URL https: //arxiv.org/abs/2306.13394.

Gunjal, A., Yin, J., and Bas, E. Detecting and preventing hallucinations in large vision language models, 2024. URL https://arxiv.org/abs/2308.06394.

He, M., Liu, Y., Wu, B., Yuan, J., Wang, Y., Huang, T., and Zhao, B. Efficient multimodal learning from data-centric perspective, 2024. URL https://arxiv. org/abs/2402.11530.

Huang, Q., Dong, X., Zhang, P., Wang, B., He, C., Wang, J., Lin, D., Zhang, W., and Yu, N. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation, 2024. URL https://arxiv.org/abs/2311.17911.

Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019. URL https://arxiv.org/abs/ 1902.09506.

Huo, F., Xu, W., Zhang, Z., Wang, H., Chen, Z., and Zhao, P. Self-introspective decoding: Alleviating hallucinations for large vision-language models, 2025. URL https: //arxiv.org/abs/2408.02032.

Jiang, B., Chen, S., Liao, B., Zhang, X., Yin, W., Zhang, Q., Huang, C., Liu, W., and Wang, X. Senna: Bridging large vision-language models and end-to-end autonomous driving, 2024. URL https://arxiv.org/abs/2410. 22313.

Kang, S., Kim, J., Kim, J., and Hwang, S. J. See what you are told: Visual attention sink in large multimodal models, 2025. URL https://arxiv.org/abs/2503. 03321.

Leng, S., Zhang, H., Chen, G., Li, X., Lu, S., Miao, C., and Bing, L. Mitigating object hallucinations in large vision-language models through visual contrastive decoding, 2023. URL https://arxiv.org/abs/2311. 16922.

Li, B., Zhang, K., Zhang, H., Guo, D., Zhang, R., Li, F., Zhang, Y., Liu, Z., and Li, C. Llavanext: Stronger llms supercharge multimodal capabilities in the wild, May 2024. URL https://llava-vl.github.io/blog/ 2024-05-10-llava-next-stronger-llms/.

Li, L., Xie, Z., Li, M., Chen, S., Wang, P., Chen, L., Yang, Y., Wang, B., and Kong, L. Silkie: Preference distillation for large visual language models, 2023a. URL https: //arxiv.org/abs/2312.10665.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models, 2023b. URL https://arxiv. org/abs/2305.10355.

Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., and Dollar, P. Microsoft coco: Common objects in con-´ text, 2015. URL https://arxiv.org/abs/1405. 0312.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26286–26296, 2024a. doi: 10.1109/CVPR52733.2024. 02484.

Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024b. URL https://llava-vl.github.io/blog/ 2024-01-30-llava-next/.

Liu, S., Zheng, K., and Chen, W. Paying more attention to image: A training-free method for alleviating hallucination in lvlms, 2024c. URL https://arxiv.org/ abs/2407.21771.

Liu, Z., Chen, Z., Liu, H., Luo, C., Tang, X., Wang, S., Zeng, J., Dai, Z., Shi, Z., Wei, T., Dumoulin, B., and Tong, H. Seeing but not believing: Probing the disconnect between visual attention and answer correctness in vlms, 2025. URL https://arxiv.org/abs/2510.17771.

O’Brien, S. and Lewis, M. Contrastive decoding improves reasoning in large language models, 2023. URL https: //arxiv.org/abs/2309.09117.

Park, N. and Kim, S. How do vision transformers work?, 2022. URL https://arxiv.org/abs/ 2202.06709.

Qian, J., Zheng, G., Zhu, Y., and Yang, S. Intervene-allpaths: Unified mitigation of lvlm hallucinations across alignment formats, 2026. URL https://arxiv. org/abs/2511.17254.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/2305.18290.

Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., and Saenko, K. Object hallucination in image captioning, 2019. URL https://arxiv.org/abs/1809. 02156.

Schwenk, D., Khandelwal, A., Clark, C., Marino, K., and Mottaghi, R. A-okvqa: A benchmark for visual question answering using world knowledge, 2022. URL https: //arxiv.org/abs/2206.01718.

Simoncelli, E. P. and Olshausen, B. Natural image statistics and neural representation. Annual Review ofNeuroscience, 24(1):1193–1216, 2001.

Sun, Z., Shen, S., Cao, S., Liu, H., Li, C., Shen, Y., Gan, C., Gui, L.-Y., Wang, Y.-X., Yang, Y., Keutzer, K., and Darrell, T. Aligning large multimodal models with factually augmented rlhf, 2023. URL https: //arxiv.org/abs/2309.14525.

Suo, W., Zhang, L., Sun, M., Wu, L. Y., Wang, P., and Zhang, Y. Octopus: Alleviating hallucination via dynamic contrastive decoding, 2025. URL https://arxiv. org/abs/2503.00361.

Wang, F., Zhou, W., Huang, J. Y., Xu, N., Zhang, S., Poon, H., and Chen, M. mdpo: Conditional preference optimization for multimodal large language models, 2024a. URL https://arxiv.org/abs/2406.11839.

Wang, H., Wu, X., Huang, Z., and Xing, E. P. High frequency component helps explain the generalization of convolutional neural networks, 2020. URL https: //arxiv.org/abs/1905.13545.

Wang, J., Wang, Y., Xu, G., Zhang, J., Gu, Y., Jia, H., Yan, M., Zhang, J., and Sang, J. An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023.

Wang, X., Pan, J., Ding, L., and Biemann, C. Mitigating hallucinations in large vision-language models with instruction contrastive decoding, 2024b. URL https: //arxiv.org/abs/2403.18715.

Wei, X.-S., Song, Y.-Z., Aodha, O. M., Wu, J., Peng, Y., Tang, J., Yang, J., and Belongie, S. Fine-grained image analysis with deep learning: A survey, 2021. URL https://arxiv.org/abs/2111.06119.

Xia, P., Zhu, K., Li, H., Zhu, H., Li, Y., Li, G., Zhang, L., and Yao, H. Rule: Reliable multimodal rag for factuality in medical vision language models, 2024. URL https: //arxiv.org/abs/2407.05131.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks, 2024. URL https://arxiv.org/abs/ 2309.17453.

Xu, K., Qin, M., Sun, F., Wang, Y., Chen, Y.-K., and Ren, F. Learning in the frequency domain. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1737–1746, 2020. doi: 10.1109/CVPR42600.2020.00181.

Yang, X., Miao, J., Yuan, Y., Wang, J., Dou, Q., Li, J., and Heng, P.-A. Medical large vision language models with multi-image visual ability, 2025. URL https: //arxiv.org/abs/2505.19031.

Yin, H., Si, G., and Wang, Z. Clearsight: Visual signal enhancement for object hallucination mitigation in multimodal large language models, 2025. URL https: //arxiv.org/abs/2503.13107.

Yin, S., Fu, C., Zhao, S., Xu, T., Wang, H., Sui, D., Shen, Y., Li, K., Sun, X., and Chen, E. Woodpecker: hallucination correction for multimodal large language models. Science China Information Sciences, 67(12), December 2024. ISSN 1869-1919. doi: 10.1007/ s11432-024-4251-x. URL http://dx.doi.org/ 10.1007/s11432-024-4251-x.

Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., Hu, J., Liu, Z., Zheng, H.-T., and Sun, M. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 13807–13816, 2024. doi: 10.1109/CVPR52733.2024.01310.

Zhang, Q., Singh, C., Liu, L., Liu, X., Yu, B., Gao, J., and Zhao, T. Tell your model where to attend: Posthoc attention steering for llms, 2024. URL https:// arxiv.org/abs/2311.02262.

Zhang, Y., Shi, Y., Yu, W., Wen, Q., Wang, X., Yang, W., Zhang, Z., Wang, L., and Jin, R. Debiasing multimodal large language models via penalization of language priors, 2025. URL https://arxiv.org/abs/2403. 05262.

Zhao, Z., Wang, B., Ouyang, L., Dong, X., Wang, J., and He, C. Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization, 2024. URL https://arxiv.org/abs/2311.16839.

Zhao, Z., Fu, H., Liang, D., Zhou, X., Zhang, D., Xie, H., Wang, B., and Bai, X. Extending large vision-language model for diverse interactive tasks in autonomous driving, 2025. URL https://arxiv.org/abs/2505. 08725.

Zheng, H. and Zhang, Z. Modality bias in lvlms: Analyzing and mitigating object hallucination via attention lens, 2025. URL https://arxiv.org/abs/2508. 02419.

Zheng, Z., Wang, Y., Huang, Y., Song, S., Yang, M., Tang, B., Xiong, F., and Li, Z. Attention heads of large language models: A survey, 2024. URL https: //arxiv.org/abs/2409.03752.

Zhu, D., Chen, J., Shen, X., Li, X., and Elhoseiny, M. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023. URL https://arxiv.org/abs/2304.10592.

Zhu, L., Ji, D., Chen, T., Xu, P., Ye, J., and Liu, J. Ibd: Alleviating hallucinations in large vision-language models via image-biased decoding, 2024. URL https: //arxiv.org/abs/2402.18476.

## A. Appendix

## A.1. Visualization of Different Hallucination Patterns

In Section 3.2, we propose a refined taxonomy of hallucinations, namely Perceptual-Semantic Dissociation (PSD) and Localized Fixation (LF). Owing to space constraints in the main text, only one representative sample per category was presented. In this section, we provide a broader range of qualitative results to offer more robust evidence.

日  
![](images/60b060de5b9a71e50017cafff15669e2d7d0f1a7b62cf19d175459275b516c17.jpg)

![](images/e52606d65a8eb4e4af87b50b670e65f7109eb8d7c67be892390214030144b64c.jpg)  
Object : hot dog GT : No

## 日

No, there is no hot dog in the image. The image features a desk with a laptop, a monitor, a keyboard, and a cell phone.

![](images/3ea34410b91054253e2a917ad86d144086430abcf5ed1aff575d661e99cdd770.jpg)

Yes, there is a screen in the image, which is a computer monitor.

Object : screen GT : Yes  
![](images/ac29d18e671a24c7c75d6d3ea3bb1f0df231e0d2fe39edec4bcfab0363c7a5ec.jpg)  
Object : chair GT : No

## 日

No, there is no chair in the image. The image shows a desk with a computer, a laptop, a monitor, and a keyboard.

![](images/7e3f755b4f3389c96a1e89d2afe5c27871ed431fb9b7c5fecbae78a22f988349.jpg)  
Object : dining table GT : Yes

##

No, there is no dining table in the image. The image features a desk with various items.

![](images/bc2b0ffdd08f01ae4091be25fb19e37618bd865b056a32fe09f68e8d26bce276.jpg)  
Object : tv GT : No

## 日

Yes, there is a TV in the image, which is located on a desk along with a laptop and other items.

Is there a {Object} in the image?

![](images/d954b909a440b5bcab32c7780565d0d9267a0d993d79f4bcb47703d6d0d4e419.jpg)

![](images/7116348209a99c12feb68208c32fde735c17ea9ebd301e33f7789289ebf1162e.jpg)  
Object : truck GT : Yes

## 日

Yes, there is a truck in the image, and it is parked near the woman and the little girl.

![](images/7b37f38976f4fc8afc1d25bc3d95b0cec5c8e61269876009a19effa4bdd746e2.jpg)  
Object : airplane GT : No

## 日

No, there is no airplane in the image. The image features a woman and a child standing under a red and black umbrella.

![](images/880e4cb58e62197e97cbdf0e617bf017ab1554d63725d75a26b8c6c0a90c6264.jpg)  
Object : sink GT : No

![](images/089cec0b4411b58ea06dacd331b915df583724a6307ef6594ad28b0c34a9591f.jpg)  
Object : school bus GT : No

No, there is no s ink in the image. The image features a woman and a child standing under a n umbrella in the rain.

Yes, there is a s c h o o l b u s in the image, and the woman and the child are standing next to it.

![](images/c05c36e1f15e532328a3d27496f84c2195cacf29580b5420a1581f2284306fb1.jpg)  
Object : handbag GT : No

##

Yes, there is a handbag in the image, and it is being held by a woman who is standing under an umbrella with a little girl.

Figure 7. Visualization of different hallucination patterns. We asked multiple questions on the same sampled image to observe distinct hallucination patterns. In the figure, green boxes represent correct responses, purple boxes represent hallucinatory responses with PSD, and pink boxes represent hallucinatory responses with LF. Note: Some queries and labels are artificially designed and are intended solely to demonstrate the phenomenon of hallucination.

## A.2. Further Analysis in the Frequency Domain

We further analyze the frequency-domain characteristics of visual tokens across different LVLMs to demonstrate the broad compatibility of our FLASH with diverse models.

## A.2.1. VISION HEAD SELECTION STRATEGY

We provide additional visualizations of visual heads and other heads across various MHA layers of different LVLMs. These results are illustrated in Figures 8 and 9.

These findings demonstrate that the observed spectral characteristics are intrinsic to different functional heads rather than being artifacts of particular models or datasets. Consequently, leveraging spectral diagrams as a proxy for head selection is a

model performance  
threshold = 0.3  
![](images/a8483b7ee661f892cb411d03ed0c3e04b11724dd404efa7ab2b51dd3906b3c0f.jpg)  
threshold = 0.1

## methodologically sound strategy.

Figure 8. Visualization of Attention Heads in Shikra. Due to the intrinsic spatial resolution of the 256-token encoding, we recommend viewing these visualizations at a reduced scale to better perceive the emergent patterns and structural features. In the figure, red text indicates the vision head.  
![](images/c2cf66514480276028712174bcbf307d25f5dcda74679c5b28a4ba349048812b.jpg)  
Figure 9. Visualization of Attention Heads in LLaVA-1.5. In the figure, red text indicates the vision head.

## A.2.2. SPECTRAL ANALYSIS OF INFORMATION FLOW WITHIN MHA

In Figure 3, we present only the curves for threshold = 0.5. (The matching curve for queries and keys is similar.) In this section, we provide additional curve results for different threshold values, shown in Figure 10.

![](images/3bff3f21f5ee5669c1da3845b20b7a0f0572b40ed1d1622a1a71b3aeda04a49d.jpg)

![](images/c554f35e55848a4351341b535c9684b1fb9cc434c705b8808b5f5fc68658e762.jpg)

![](images/99b86e49303b10f50858b1af8c5f557b156bec2e5b2269dd8d87c34362c49f1f.jpg)  
threshold = 0.2

![](images/59f0ed146ed033bcd07d10d6b17df9b3baa371d57d41a9f6c92757f86d89be03.jpg)

![](images/460de98f5a39df1c4075c9c5763de7c227a5f7e88aecf6969c478ee3f13010eb.jpg)

![](images/7e2b76d5d2cd32ea7f6d68fe7f0a9d356ea2c2408ee25e3404ed551adcb8d3e7.jpg)  
threshold = 0.5

![](images/c7319d10c2423039e16707c0b413648fc32adb9fdb7ed550cb9aa562ab04c4eb.jpg)  
threshold = 0.6

![](images/c03276d59241e1e05aa8502a189194052669fba984da045445145657e7940def.jpg)  
threshold = 0.7

![](images/05aa1a7956cf5774ab7d7b13abf8049026abc067b914ac12e9303e4e0c5918c8.jpg)  
threshold = 0.8

![](images/b3d87b2217ee3b613fbc94fbd07ba719b1fa00682156a6c561f8cd780fcf6977.jpg)  
threshold = 0.9  
Figure 10. Proportion of spectral energy in value and attention outputs across varying high-frequency thresholds. Thresholds represent the top percentile of frequency components (e.g., 0.1 denotes the top 10%). When conducting statistical experiments, we employ random sampling methods.

As illustrated in Figure 10, for thresholds $< 0 . 4 ,$ , the model effectively restores high-frequency energy proportions in deep layers (e.g., maintaining levels up to layer 25 when threshold = 0.1 and layer 20 when threshold = 0.2). Consequently, the model exhibits robust performance within this lower threshold regime. However, as the threshold exceeds 0.4, this restorative capacity diminishes, leading to a precipitous decline in overall performance.

## A.3. Theoretical Explanation

In this section, we introduce the rationale for modulating V-stream in logarithmic frequency spectrum and how energy across different frequency bands in our DSM is adaptively transferred. These theorems correspond to Section 4.3 of the main text.

## A.3.1. RATIONALE FOR LOG-MAGNITUDE DOMAIN MODULATION

In Section 4.3, we propose performing spectral intervention of V-stream in the log-magnitude domain as defined in Eq. 13. We provide a theoretical justification for this design:

Spectral coefficients of visual features (Value matrix) in Transformers typically follow a power-law distribution, where the magnitude $| C ( f ) |$ at frequency $f$ scales as $\vert C ( f ) \vert \propto f ^ { - \alpha }$ (Field & David, 1987; Simoncelli & Olshausen, 2001). This results in a massive dynamic range where low-frequency components are orders of magnitude larger than high-frequency ones. A linear modulation $\tilde { C } = \omega \cdot C$ would be highly sensitive to the choice of $\omega ;$ a small $\omega$ might be insufficient for high-frequency enhancement, while a slightly larger $\omega$ could lead to numerical overflow in low-frequency regions. By operating in the log-domain:

$$
\log | \tilde { C } | = \mathcal { M } _ { v } \cdot \log | C | ,\tag{20}
$$

the power-law relationship is linearized. This transformation is equivalent to a power-law refinement: $| \tilde { C } | = | C | ^ { \mathcal { M } _ { \tau } }$ . This allows the modulation mask $\mathcal { M } _ { v }$ to act as an exponential scaling factor that adjusts the contrast between frequency bands rather than their absolute values, providing a much more stable control over the spectral slope.

Additionally, we conducted ablation experiments to compare the impact of logarithmic domain modulation and linear domain modulation on model performance. The relevant experimental results and analysis are presented in Section A.5.

## A.3.2. THEORETICAL ANALYSIS OF ADAPTIVE ENERGY MODULATION

Eq. 15 introduces a Frobenius norm constraint to calibrate the modulated features. In this section, we prove that this functions as an adaptive energy reallocation mechanism through the lens of Parseval’s Theorem.

Energy Conservation in Frequency and Spatial Domains: According to Parseval’s Theorem for the Discrete Cosine Transform (DCT), the energy of a signal is preserved between the spatial and frequency domains:

$$
| | V | | _ { F } ^ { 2 } = \sum _ { i , j } | V | ^ { 2 } = \sum _ { u , v } | C _ { u , v } | ^ { 2 } = | | C | | _ { F } ^ { 2 }\tag{21}
$$

Therefore, the constraint in Eq. 15 can be interpreted as a normalization of total spectral energy:

$$
| | V ^ { * } | | _ { F } = | | V | | _ { F } \Rightarrow | | C ^ { * } | | _ { F } = | | C | | _ { F }\tag{22}
$$

The Zero-Sum Dynamics: Let the total energy of the original spectrum be:

$$
E _ { t o t a l } = E _ { l o w } + E _ { h i g h } .\tag{23}
$$

When we enhance the high-frequency components in the log-domain, the new energy of the intermediate spectrum $\tilde { C }$ becomes:

$$
E _ { e n h a n c e d } = \tilde { E } _ { l o w } + \tilde { E } _ { h i g h }\tag{24}
$$

where $\tilde { E } _ { h i g h } > E _ { h i g h }$ (assuming $\mathcal { M } _ { v } > 1$ for high frequencies).

By applying the normalization factor $\begin{array} { r } { \gamma = \frac { | | V | | _ { F } } { | | \tilde { V } | | _ { F } } } \end{array}$ , the final energy distribution becomes:

$$
E ^ { * } = \gamma ^ { 2 } \tilde { E } _ { l o w } + \gamma ^ { 2 } \tilde { E } _ { h i g h } = E _ { t o t a l }\tag{25}
$$

Since $\tilde { E } _ { h i g h }$ has been increased significantly by the modulation, the denominator $| | \tilde { V } | | _ { F }$ increases, resulting in $\gamma < 1$ Consequently, the term $\gamma ^ { 2 } \tilde { E } _ { l o w }$ is forced to decrease relative to the original $E _ { l o w }$

Therefore, Eq. 15 forces the model to pay for the high-frequency enhancement by automatically draining redundant energy from the low-frequency bands. This ensures that our modulation operates not by simple linear amplification, but by redistributing the model’s attention within a fixed representation budget, thus maintaining numerical stability and pre-trained alignment. This proof also applies to the constraint process of S-stream.

## A.4. Performance of Different Tasks on the MME Dataset

![](images/69e81bf0d651bcd806b5b3ebc458aef9cdad615b1a8390cb7a8a88fd91c36228.jpg)

![](images/402184021419b9007ee5c521a771414d4ba893c6f4a1e8a256221e2719cc0904.jpg)

![](images/71a092aa4afbafe475a331c51a286225cf8e9969ae37635cbadb50b7128adf06.jpg)  
Figure 11. Performance of Different Tasks on the MME Dataset.

## A.5. Ablation Experiment and Parameter Sensitivity Analysis

Owing to space constraints, the main text focuses on core ablation studies. This section provides supplementary analysis and extended findings, including comprehensive parameter sensitivity tests (Table 3 and Figure 12) and a detailed evaluation of the logarithmic-domain modulation scheme (Table 4) referenced in Section 4.3. Additionally, we present the qualitative results of the task ablation experiments in Figure 13.

Table 3. Comparison of performance across different hyperparameters. In the table, orders 1–5 represent sensitivity experiments for $\lambda _ { v } ;$ orders 6–10 represent sensitivity experiments for $\lambda _ { s } ;$ orders 11–15 represent sensitivity experiments for k. The green shaded rows indicate the hyperparameter combinations used in this work.
<table><tr><td>Orders</td><td rowspan="2"> $\lambda _ { v }$   $\lambda _ { s }$ </td><td rowspan="2">k</td><td colspan="4">POPE-Random</td></tr><tr><td></td><td>Accuracy ↑ F1↑</td><td>Precision ↑</td><td></td><td>Recall ↑</td></tr><tr><td>Greedy</td><td>1</td><td>1 1</td><td>88.97</td><td>88.90</td><td>89.41</td><td>88.40</td></tr><tr><td>1</td><td>1.0</td><td>0.9 5</td><td>89.63</td><td>89.37</td><td>91.72</td><td>87.13</td></tr><tr><td>2</td><td>1.2</td><td>0.9 5</td><td>90.03</td><td>89.77</td><td>92.20</td><td>87.47</td></tr><tr><td>3</td><td>1.4</td><td>0.9 5</td><td>89.97</td><td>89.69</td><td>92.25</td><td>87.27</td></tr><tr><td>4</td><td>1.6</td><td>0.9 5</td><td>89.83</td><td>89.52</td><td>92.41</td><td>86.80</td></tr><tr><td>5</td><td>1.8</td><td>0.9 5</td><td>89.77</td><td>89.42</td><td>92.52</td><td>86.53</td></tr><tr><td>6</td><td>1.2</td><td>0.9 5</td><td>90.03</td><td>89.77</td><td>92.20</td><td>87.47</td></tr><tr><td>7</td><td>1.2</td><td>0.7 5</td><td>90.03</td><td>89.76</td><td>92.26</td><td>87.40</td></tr><tr><td>8</td><td>1.2</td><td>0.5 5</td><td>90.03</td><td>89.76</td><td>92.26</td><td>87.40</td></tr><tr><td>9</td><td>1.2</td><td>0.3 5</td><td>90.03</td><td>89.76</td><td>92.26</td><td>87.40</td></tr><tr><td>10</td><td>1.2</td><td>0.1 5</td><td>90.00</td><td>89.73</td><td>92.19</td><td>87.40</td></tr><tr><td>11</td><td>1.2</td><td>0.9 1</td><td>89.83</td><td>89.57</td><td>91.93</td><td>87.33</td></tr><tr><td>12</td><td>1.2</td><td>0.9 3</td><td>89.97</td><td>89.71</td><td>92.07</td><td>87.47</td></tr><tr><td>13</td><td>1.2</td><td>0.9 5</td><td>90.03</td><td>89.77</td><td>92.20</td><td>87.47</td></tr><tr><td>14</td><td>1.2</td><td>0.9 8</td><td>89.97</td><td>89.71</td><td>92.07</td><td>87.47</td></tr><tr><td>15</td><td>1.2</td><td>0.9 10</td><td>89.97</td><td>89.71</td><td>92.07</td><td>87.47</td></tr></table>

## A.6. Comparison of Inference Performance and Efficiency

We demonstrate a comparison of the performance and efficiency of different methods in this section. We evaluated the inference performance of LLaVA-1.5 (Liu et al., 2024a) integrated with various hallucination mitigation strategies (Leng et al., 2023; Liu et al., 2024c; Huo et al., 2025), alongside their latency multipliers relative to the vanilla model. The result is shown in Table 5.

![](images/89fdd3bb20200691b214cae3f8e923037242ee65d963fb9555ec43155d27dfdb.jpg)  
Figure 12. Comparison of parameter sensitivity experiment results for key hyperparameters.

Table 4. Comparison of modulation strategies in DSM. The V. and S. in the table represent V-stream and S-stream respectively. Logarithmic and Linear represent modulation of the spectrum in the logarithmic domain and linear domain, respectively.
<table><tr><td rowspan="2">Orders</td><td colspan="2">Modulation Space</td><td colspan="4">POPE-Random</td></tr><tr><td>Logarithmic</td><td>Linear</td><td>Accuracy ↑</td><td>F1↑</td><td>Precision ↑</td><td>Recall ↑</td></tr><tr><td>1</td><td>V.</td><td>S.</td><td>90.03</td><td>89.77</td><td>92.20</td><td>87.47</td></tr><tr><td>2</td><td>S.</td><td>V.</td><td>89.77</td><td>89.51</td><td>91.80</td><td>87.33</td></tr><tr><td>3</td><td>V. + S.</td><td>1</td><td>90.00</td><td>89.74</td><td>92.13</td><td>87.47</td></tr><tr><td>4</td><td>1</td><td>V. + S.</td><td>89.70</td><td>89.43</td><td>91.85</td><td>87.13</td></tr></table>

Table 5. Comparison of performance and efficiency across different methods. In the table, Average represents the performance average across the three POPE-MSCOCO splits. Times reflects the multiple of the inference time of the baseline model (relative to the baseline method) when applying different methods on the POPE-bench dataset.
<table><tr><td rowspan="2">Methods</td><td colspan="2">POPE-R</td><td colspan="2">POPE-P</td><td colspan="2">POPE-A</td><td colspan="2">Average</td><td colspan="2">CHAIR</td><td rowspan="2">Times</td></tr><tr><td>Acc. ↑</td><td>F1↑</td><td>Acc. ↑ F1↑</td><td>Acc. ↑</td><td>F1↑</td><td>Acc. ↑</td><td>F1 ↑</td><td>C.S ↓</td><td>C.I ↓</td><td></td></tr><tr><td>LLaVA-1.5</td><td>88.97</td><td>88.90</td><td>85.63</td><td>86.03</td><td>79.23</td><td>80.99</td><td>84.61</td><td>85.31</td><td>48.10</td><td>12.75</td><td>1×</td></tr><tr><td>+VCD</td><td>89.07</td><td>88.99</td><td>85.60</td><td>85.99</td><td>79.27</td><td>81.00</td><td>84.65</td><td>85.33</td><td>48.80</td><td>12.85</td><td>1.98×</td></tr><tr><td>+PAI</td><td>89.30</td><td>89.27</td><td>86.07</td><td>86.45</td><td>79.23</td><td>81.06</td><td>84.87</td><td>85.59</td><td>47.80</td><td>12.35</td><td>1.87×</td></tr><tr><td>+SID</td><td>89.40</td><td>89.04</td><td>85.93</td><td>85.93</td><td>80.33</td><td>81.38</td><td>85.22</td><td>85.45</td><td>48.10</td><td>12.40</td><td>2.58×</td></tr><tr><td>+Ours</td><td>90.03</td><td>89.77</td><td>86.53</td><td>86.62</td><td>80.53</td><td>81.75</td><td>85.70</td><td>86.05</td><td>47.50</td><td>12.55</td><td>1.69×</td></tr></table>

## A.7. Qualitative Analysis of Generated Results

Due to space constraints in the main text, we present here a qualitative comparison of results from the generation task. The two images are sampled from the AMBER dataset (Wang et al., 2023).

## A.8. More Detailed Experimental Settings

In this section, we report more extensive experimental settings to supplement the relevant content in the section 5.1.

## A.8.1. DATASETS & EVALUATION METRICS

Following (Suo et al., 2025; Leng et al., 2023; Huo et al., 2025; Liu et al., 2024c; Wang et al., 2024b; Yin et al., 2025), we employ POPE (Li et al., 2023b) and MME (Fu et al., 2025) as benchmarks for discriminative tasks. For generative tasks, we select the widely recognized CHAIR (Rohrbach et al., 2019) and AMBER (Wang et al., 2023) benchmarks.

POPE (Li et al., 2023b): The Polling-based Object Probing Evaluation (POPE) is designed to evaluate hallucinations in LVLMs. It comprises three datasets: MSCOCO (Lin et al., 2015), A-OKVQA (Schwenk et al., 2022), and GQA (Hudson & Manning, 2019). Built upon the VQA task, POPE divides each dataset into three splits—Random, Popular, and Adversarial—totalling 27,000 queries. Each split samples 500 images, with six questions per image using the template: ”Is there a {object} in the image?”. The objects are sourced based on different criteria: the Random split selects objects randomly from the dataset; the Popular split picks the most frequent objects; and the Adversarial split identifies objects that frequently co-occur with ground-truth objects but are absent, making it the most challenging. In this work, we use the

![](images/18f2a3966a85e8a3bc8e4e271e36de110752cc9af838a28efdfe60986db79f35.jpg)  
Figure 13. Qualitative comparison of the ablation study results.

![](images/75142bfe3adf1d9f0bcfd32c7f98e921b1278ad49b725d7817123601798d4f85.jpg)  
Figure 14. Qualitative comparison of different methods for the generative task.

MS-COCO dataset (9,000 queries) as the representative testbed, employing Accuracy and F1-score as the evaluation metrics, formulated as:

$$
{ \mathrm { A c c u r a c y } } = { \frac { T P + T N } { T P + F P + F N + T N } } ,\tag{26}
$$

$$
{ \mathrm { F 1 - S c o r e } } = 2 \times { \frac { ( { \mathrm { P r e c i s i o n } } \times { \mathrm { R e c a l l } } ) } { ( { \mathrm { P r e c i s i o n } } + { \mathrm { R e c a l l } } ) } } ,\tag{27}
$$

where:

$$
{ \mathrm { R e c a l l } } = { \frac { T P } { T P + F N } } ,\tag{28}
$$

$$
{ \mathrm { P r e c i s i o n } } = { \frac { T P } { T P + F P } } .\tag{29}
$$

MME (Fu et al., 2025): MME is a comprehensive benchmark designed to evaluate the perceptual and cognitive faculties of LVLMs across 14 distinct subtasks. By employing a binary ”Yes/No” response framework, it effectively minimizes the interference of models’ linguistic inductive biases. While the perception track assesses fundamental attributes such as object existence, count, color, and position, the cognition track demands higher-level reasoning. In accordance with the protocol in (Cho et al., 2025; Leng et al., 2023), we evaluate different methods on four representative perception subtasks—Existence, Count, Position, and Color—and report the performance using the Score metrics (Fu et al., 2025).

CHAIR (Rohrbach et al., 2019): The Caption Hallucination Assessment with Image Relevance (CHAIR) has been widely adopted to evaluate hallucinations in generative tasks. As a captioning-based benchmark, it prompts LVLMs to describe input images in detail and calculates the proportion of mentioned objects that do not exist in the ground-truth label pool. CHAIR provides two granularities: instance-level $\mathrm { ( C H A I R _ { I } ) }$ and sentence-level (CHAIR ), formulated as:

$$
\mathrm { C H A I R } _ { \mathrm { I } } = \frac { \lvert \{ \mathrm { h a l l u c i n a t e d o b j e c t s } \} \rvert } { \mathrm { a l l ~ m e n t i o n e d ~ o b j e c t s } } ,\tag{30}
$$

$$
\mathrm { C H A I R _ { S } = \frac { \lvert \{ c a p t i o n \ w i t h \ h a l l u c i n a t e d \ o b j e c t s \} \rvert } { a l l \ c a p t i o n s } . }\tag{31}
$$

Following previous studies (Liu et al., 2024c; Zheng & Zhang, 2025), we randomly sample 500 images from MS-COCO (Lin et al., 2015) and use the prompt: ”Please help me describe the image in detail.”

AMBER (Wang et al., 2023): AMBER is a recently developed multi-dimensional benchmark focused on hallucination evaluation in MLLMs. It offers fine-grained annotations and an automated, LLM-free evaluation pipeline, avoiding the high cost of GPT-4 APIs while ensuring efficiency and accuracy. Compared to CHAIR (Rohrbach et al., 2019), AMBER features a more extensive ground-truth label pool and covers existence, attributes, and relationships. We utilize the Generative task split (1,004 images, prompt: ”Describe this image.”) and follow the standard metrics: Cover, Hal, and Cog (Wang et al., 2023). We randomly selected 200 images from the collection for testing.

## A.8.2. BASELINES

To demonstrate the efficacy and generalization of FLASH, we integrate it into two representative LVLM families, covering diverse scales:

LLaVA-1.5(Liu et al., 2024a): As a prominent representative of end-to-end trained LVLMs, LLaVA-1.5 refines the original LLaVA framework by adopting a more powerful vision-language connector. It utilizes a CLIP-ViT-L/14 visual encoder and a Vicuna-v1.5 LLM, bridged by a two-layer MLP projection matrix. Despite its architectural simplicity, LLaVA-1.5 achieves SOTA performance on various academic benchmarks through the use of high-resolution image inputs and a diverse mix of instruction-tuning data. We select this model to validate FLASH’s effectiveness on the standard MLP-based vision-language paradigm. In this paper, we selected the 7B and 13B version of LLaVA-1.5.

Shikra(Chen et al., 2023a): Shikra is designed to excel in ”Referential Dialogue,” a task requiring the model to precisely ground objects mentioned in the conversation. Unlike models that rely on auxiliary detection heads or specialized pretraining, Shikra introduces a unified framework that handles spatial coordinates (bounding boxes) as natural language tokens. This design allows it to perform seamless spatial reasoning and object localization within a multi-turn dialogue. By including Shikra, we aim to demonstrate that FLASH can mitigate hallucinations even in models with strong inherent grounding capabilities. In this paper, we selected version 7B.

## A.8.3. IMPLEMENTATION DETAILS

For the compared method, we conducted testing using their official code. On the CHAIR dataset, we randomly sampled 500 images for inference using the query: “Please help me describe this image in detail.” For the AMBER dataset, we randomly sampled 200 images for inference testing, adhering to the official AMBER configuration (Wang et al., 2023) with the query: “Describe this image.” For both generation tasks, we report the average results of tests conducted under two different random seed sets. Additionally, beyond the experimental details reported in Section 5.1, we only modulate the firs token for the discriminative task. The hyperparameter configurations for different models are shown in the Table 6.

Table 6. Hyperparameters for different models. In the table, “V/S-stream layer” indicates the layer within the MHA where the corresponding modulation scheme is applied.
<table><tr><td></td><td> $\lambda _ { s }$ </td><td> $\lambda _ { v }$ </td><td>S-stream Layer</td><td>V-stream-layer</td><td>T</td><td> $k$ </td></tr><tr><td colspan="7">Discrimination Task</td></tr><tr><td>LLaVA-1.5 7B</td><td>0.9</td><td>1.2</td><td>16-27</td><td>3-32</td><td>0.2</td><td>5</td></tr><tr><td>Shikra 7B</td><td>0.9</td><td>1.1</td><td>16-27</td><td>3-32</td><td>0.4</td><td>5</td></tr><tr><td>LLaVA-1.5 13B</td><td>0.9</td><td>5.0</td><td>16-27</td><td>3-40</td><td>0.2</td><td>5</td></tr><tr><td colspan="7">Generation Task</td></tr><tr><td>LLaVA-1.5 7B</td><td>0.1</td><td>1.2</td><td>16-27</td><td>3-32</td><td>0.2</td><td>5</td></tr><tr><td>Shikra 7B</td><td>0.9</td><td>1.1</td><td>16-27</td><td>3-32</td><td>0.4</td><td>5</td></tr><tr><td>LLaVA-1.5 13B</td><td>0.1</td><td>1.2</td><td>16-27</td><td>3-40</td><td>0.2</td><td>5</td></tr></table>