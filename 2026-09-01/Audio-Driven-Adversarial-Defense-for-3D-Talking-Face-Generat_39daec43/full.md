# Audio-Driven Adversarial Defense for 3D Talking Face Generation with totally Visual Fidelity Preservation

Rui-Qing Sun 2325557558@qq.com   
Beijing Institute of Technology   
Haidian Qu, Beijing Shi, China

Chen-Hao Cui Beijing Institute of Technology Haidian Qu, Beijing Shi, China

Hui-Yang Zhao Beijing Institute of Technology Haidian Qu, Beijing Shi, China

![](images/339d541557ad299ee2d4fea041250f3933ab3723299b6e4269ab772ae36fb52c.jpg)  
Figure 1: Comparison between conventional visual-domain defense and our psvchaacoustically quided audio-domain defense. Visual perturbations may introduce noticeable artifacts and are vulnerable to resizing, while our method hides protective perturbations in perceptuallv masked audio regions, preserving portrait appearance while suppressing reliable 3D-feild talking face generation.

## Abstract

The rapid development of generative portrait models has raised growing concerns about privacy leakage and identity misuse. In particular, audio-driven 3D talking face generation can reconstruct a reusable 3D portrait of a target person from a monocular video and animate it with arbitrary speech, making realistic identity impersonation alarmingly practical. Existing proactive defenses mainly operate in the visual domain by injecting subtle perturbations into facial regions to disrupt identity acquisition. However, such perturbations often compromise visual quality due to the strong structural priors and social sensitivity of human faces, and are easily weakened by common real-world transformations such as resizing. To overcome these limitations, we propose an imperceptible audio defense for audio-driven 3D talking face generation by shifting protection from the visual modality to the audio modality. Specifically, we exploit psychoacoustic masking to hide protective perturbations within perceptually masked frequency regions of the speech signal, thereby reducing perceptual distortion while suppressing reliable facial animation. Extensive experiments demonstrate that the proposed method efectively degrades 3D talking face generation while preserving favorable perceptual quality. These findings highlight psychoacoustically guided audio perturbations as a practical and promising direction for privacy-preserving portrait protection.

## Keywords

Talking Face Generation, Proactive Defense, Audio Adversarial Attack,

## ACM Reference Format:

Rui-Qing Sun, Chen-Hao Cui, Hui-Yang Zhao, Tian Lan, Zhijing Wu, and Xian-Ling Mao. 2018. Audio-Driven Adversarial Defense for 3D Talking Face Generation with totally Visual Fidelity Preservation. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 10 pages. https: //doi.org/XXXXXXX.XXXXXXX

## 1 Introduction

The recent surge of generative multimedia models has greatly expanded the capability of synthesizing realistic human-centric content, including portraits, speech, and talking videos [3, 15, 25]. Among these advances, talking face generation has become a representative multimodal task because it tightly couples facial appearance with audio-driven motion generation [6, 22, 23]. As a representative multimodal generation task, talking face generation tightly couples visual identity modeling with speech-driven motion synthesis, making privacy protection especially challenging in real-world multimedia sharing scenarios. In particular, audiodriven 3D talking face generation can recover a personalized 3D portrait from a monocular video and subsequently animate it with arbitrary speech, producing highly realistic and identity-consistent results [13, 14, 22, 29, 34]. These capabilities open up promising opportunities for digital humans, virtual communication, and mul timedia creation. However, the same capability also makes portrait misuse alarmingly practical. Once a victim’s portrait information is acquired, the reconstructed 3D representation can be reused for scalable and near-real-time synthesis, enabling impersonation, fabricated speeches, and other forms of identity abuse [20, 30].

This privacy risk is especially concerning in today’s online ecosystem, where user-generated videos are continuously shared across social platforms and can be easily harvested by malicious actors. Compared with conventional 2D portrait animation [6, 23], audio-driven 3D talking face generation poses a more serious threat because it not only captures appearance details, but also reconstructs reusable three-dimensional identity cues that support highfidelity reenactment under arbitrary speech input [13, 14, 22]. In practice, even a short publicly accessible video may already provide suficient material for reconstructing a controllable digital portrait of the target person. Therefore, protecting portrait media before it is collected by such generation systems has become an urgent problem in multimedia security and privacy protection.

To mitigate these risks, recent proactive defense methods attempt to add subtle perturbations to media before they are scraped or reused by generative models. Most existing defenses are developed in the visual domain, where adversarial perturbations are injected into facial regions to interfere with identity extraction, landmark estimation, or downstream portrait reconstruction [11, 16, 18, 26, 31]. This line of research is intuitive, since visual appearance is the most direct source for identity acquisition. Nevertheless, visual-domain protection faces inherent limitations when applied to portrait sharing scenarios. Human faces are highly structured and socially sen sitive visual objects, and even slight distortions in facial regions may noticeably reduce visual quality, damage user experience, or introduce unnatural artifacts. More importantly, visual perturbations are often fragile under common real-world transformations such as resizing, resampling, and compression, which are almost unavoidable during online transmission and platform processing [8]. As a result, their protection efectiveness can be significantly weakened before the media is actually used by an attacker.

These limitations motivate us to rethink where protection should be imposed in audio-driven 3D talking face generation. Instead of continuing to perturb the visually sensitive facial region, we ask whether the defense can be shifted to the audio modality. This perspective is particularly attractive because the driving audio plays a fundamental role in controlling lip motion and facial dynamics, making it a natural intervention point for disrupting high-quality synthesis. Meanwhile, unlike the visual domain, the auditory domain provides a more explicit perceptual principle for imperceptibility, namely psychoacoustic masking [24, 28]. According to this principle, weak sound components can become imperceptible in the presence of stronger neighboring frequencies, which provides a principled way to hide protective perturbations while preserving perceptual quality.

Based on this insight, we propose an imperceptible audio defense for audio-driven 3D talking face generation. Rather than directly perturbing facial pixels, our method injects protective perturbations into perceptually masked frequency regions of the speech signal under psychoacoustic guidance. In this way, the perturbations remain dificult for human listeners to perceive, while still interfering with the generation pipeline and suppressing reliable facial animation. By shifting protection from the visual modality to the audio modality, our approach avoids directly damaging portrait appearance and ofers a more user-friendly solution for privacy-preserving portrait sharing.

Extensive experiments demonstrate that the proposed method efectively degrades the performance of audio-driven 3D talking face generation while preserving favorable perceptual quality. The results suggest that psychoacoustically guided audio perturbations provide a practical and promising direction for proactive portrait protection, especially in scenarios where visual fidelity is critical. Overall, this work ofers a new modality-level perspective on defending against portrait generation systems and highlights the importance of incorporating perceptual principles into deployable multimedia privacy protection methods.

Our main contributions are summarized as follows:

• We revisit proactive defense for talking face generation from a new modality perspective and reveal the limitations of existing visual-domain protection methods in terms of perceptual quality and robustness under common real-world transformations.

• We propose an imperceptible audio defense for audio-driven 3D talking face generation by leveraging psychoacoustic masking to conceal protective perturbations in perceptually masked frequency regions.

• We demonstrate through extensive experiments that the proposed method efectively suppresses 3D talking face generation while maintaining favorable perceptual quality, showing the practicality of audio-based protection for privacypreserving portrait sharing.

## 2 Related Works

## 2.1 Audio-driven Talking Face Generation

Audio-driven talking face generation (TFG) aims to synthesize realistic talking portraits by animating facial motions according to speech signals. Early methods mainly relied on 2D image generation or warping-based pipelines [5, 6, 23], which achieved plausible lip synchronization but often struggled to model natural head movement, fine-grained facial dynamics, and multi-view consistency.

Recent advances in 3D representations have substantially im proved the realism and controllability of TFG. In particular, Neural Radiance Fields (NeRF) [1, 2, 10, 14, 22, 29, 34] and 3D Gaussian Splatting (3DGS) [7, 12, 13] enable the reconstruction of subjectspecific 3D portraits from monocular reference videos, leading to stronger identity consistency, more accurate lip synchronization, and better view-consistent rendering. Compared with generalized one-shot portrait animation methods [15, 35], these personalized 3D-field approaches are particularly powerful because they can recover reusable identity-aware 3D representations and support highly realistic speech-driven reenactment.

However, this strong generation capability also introduces significant privacy risks. Once a target person’s reference video is collected, current 3D-field TFG models can reconstruct a controllable digital portrait and synthesize realistic videos under arbitrary speech input. This makes them especially concerning in public multimedia sharing scenarios, where user portraits and voices can be easily harvested and reused. In this work, we focus on defending against such audio-driven 3D-field TFG systems, which pose a more serious threat than conventional 2D talking face generation due to their stronger realism, controllability, and reusability.

## 2.2 Talking Face Defense

With the rapid progress of portrait generation models, recent studies have begun to explore proactive defenses against talking face generation and related portrait manipulation systems. Most exist ing methods are developed in the visual domain, where protective perturbations are injected into facial regions to disrupt identity extraction, facial landmark estimation, reconstruction, or downstream editing and animation models [11, 16, 18, 26, 31]. This line of work is intuitive because the face image is the most direct source for identity modeling and visual synthesis.

Despite their efectiveness, visual-domain defenses face two major limitations in the context of talking face protection. First, human faces are highly structured and socially sensitive visual objects, so even subtle perturbations may noticeably reduce visual quality or introduce undesirable facial artifacts. Second, such perturbations are often fragile under common real-world transformations, includ ing resizing, resampling, and platform-side compression, which are almost unavoidable during online media sharing and redistribution. These issues are especially problematic for 3D-field TFG, where accurate visual priors are often reconstructed through long video preprocessing pipelines and may be partially restored or purified during fitting.

Compared with prior work, our goal is not to further optimize visual perturbations, but to revisit where protection should be imposed. We argue that, for audio-driven talking face generation, the driving audio itself provides a more natural and practical interven tion point. Instead of perturbing visually sensitive facial regions, we shift protection to the audio modality and study how imperceptible audio perturbations can suppress reliable facial animation while preserving portrait appearance. To the best of our knowledge, this perspective remains largely underexplored in proactive defense for talking face generation, especially for modern 3D-field TFG systems.

## 2.3 Voice Clone Defense

Our work is also related to recent eforts on voice clone defense and adversarial protection for speech generation systems. Prior studies have shown that speech models, including automatic speech recognition (ASR), speaker verification, and voice cloning systems, are vulnerable to carefully designed perturbations in the audio domain [4, 21, 24, 28, 33]. To improve perceptual stealth, several works further incorporate psychoacoustic principles and human auditory masking models to constrain perturbations below perceptual thresholds [24, 28]. These studies demonstrate that the audio modality provides a principled path to achieving both attack efectiveness and human imperceptibility.

However, defending against audio-driven talking face generation is fundamentally diferent from defending against pure speech or speaker models. In our setting, the audio signal is not the final target itself, but a cross-modal control signal that drives facial geometry and lip motion. Therefore, the objective is not merely to alter linguistic content or speaker identity, but to disrupt the audio-to-geometry mapping required for realistic facial animation. This introduces a new challenge: the perturbation must remain imperceptible to listeners while still being strong enough to interfere with the downstream visual generation process. Our work bridges this gap by introducing a psychoacoustically guided audio defense specifically designed for 3D talking face generation, connecting voice-side imperceptible perturbation design with portrait privacy protection.

## 3 Preliminaries

## 3.1 Audio-Driven Talking Face Generation in 3D Fields

Unlike 2D-based talking head synthesis that operates on image pixels, 3D-field TFG models represent the human head as a continuous or discrete dynamic 3D structure. This approach ensures multi-view consistency and realistic geometric deformation. The common pipeline involves mapping acoustic features to a 3D spatial representation, which is then rendered into 2D images.

Generalized 3D Scene Representations. A 3D field F characterizes the physical properties (e.g., density �, color c, or covariance Σ) of any point $\mathbf { x } \in \mathbb { R } ^ { 3 }$ . Recent TFG frameworks employ various eficient 3D representations:

![](images/fdca364c5cbaf242f248293496b26e96f2751b087e4476740cc86ac9d47c2d14.jpg)  
Figure 2: Overview of the 3D-field TFG pipeline. The reference video is disentangled into identity, pose, and expression priors. Driving audio modulates 3D spatial samples via a latent lip space. Blue arrows indicate training-only operations (e.g., identity/pose extraction), while red arrows denote operations common to both training and inference (e.g., audiovisual encoding and neural rendering).

• Volumetric Fields: (NeRF-based) The scene is represented as a continuous function $f : ( \mathbf { x } , \mathbf { d } ) \to ( \sigma , \mathbf { c } )$ . To accelerate rendering, Tri-planes $( \mathcal { P } _ { x y } , \mathcal { P } _ { x z } , \mathcal { P } _ { y z } )$ are often employed, where features at coordinate x are aggregated from orthogonal projections:

$$
{ \pmb f } _ { \mathbf { x } } = \mathrm { A g g } _ { i \in \{ x y , x z , y z \} } \left( { \cal T } ( { \mathcal P } _ { i } , \pi _ { i } ( { \bf x } ) ) \right)\tag{1}
$$

• Point-based Fields: (3DGS-based) The scene is explicitly modeled by a collection of 3D Gaussians, each defined by its center �, opacity �, and covariance matrix �. Audio features drive the facial motion by predicting residual ofsets $\Delta \mu$ or modulating a canonical deformation field.

• Hybrid Fields: These models leverage 3D Morphable Models (3DMM) to provide explicit structural priors $( \alpha , \beta , \delta ) ;$ integrating parametric facial geometry with neural feature volumes to achieve both high-fidelity rendering and precise expression control.

Audio-to-Geometry Mapping. The fundamental mechanism of a 3D-field TFG model � is the modulation of the 3D field $\mathcal { F }$ by an audio embedding �. Let Φ be the transformation function that maps the canonical identity to a dynamic state:

$$
\mathcal { F } _ { s y n c } = \Phi ( \mathcal { F } _ { s t a t i c } , \pmb { a } )\tag{2}
$$

where $\mathcal { F } _ { s t a t i c }$ represents the static spatial prior of the speaker, and $\mathcal { F } _ { s y n c }$ is the audio-synced dynamic field.

## 3.2 Principles of Psychoacoustic Masking

To achieve imperceptible protection, we must ensure that the adversarial noise added to the audio remains beneath the detection threshold of the human auditory system. This is governed by the Psychoacoustic Masking Efect.

Frequency Masking and Critical Bands. The human ear processes sound through a non-linear filter bank on the cochlea, known as Critical Bands. A strong sound at a specific frequency (the masker) will raise the hearing threshold of neighboring frequencies (the maskee). This masking power is more efectively modeled on the Bark Scale (�):

$$
z = 1 3 \arctan ( 0 . 0 0 0 7 6 f ) + 3 . 5 \arctan ( ( f / 7 5 0 0 ) ^ { 2 } )\tag{3}
$$

Within this scale, if the energy of our adversarial perturbation � stays within the "masking umbrella" created by the original speech signal �, the human brain will filter out � as redundant information. Absolute Threshold of Hearing (ATH). The Absolute Threshold of Hearing (ATH) represents the baseline sensitivity of the human ear in a noiseless environment. It is defined by the non-linear function $T _ { q } ( f )$

$$
T _ { q } ( f ) = 3 . 6 4 \left( f / 1 0 0 0 \right) ^ { - 0 . 8 } - 6 . 5 e ^ { - 0 . 6 \left( f / 1 0 0 0 - 3 . 3 \right) ^ { 2 } } + 1 0 ^ { - 3 } \left( f / 1 0 0 0 \right) ^ { 4 }\tag{dB}
$$

(4)

This threshold indicates that the ear is most sensitive between 2–5 kHz (the range of human speech) and significantly less sensitive at very low or very high frequencies. Any protective noise falling below $T _ { q } ( f )$ is physically inaudible.

Global Masking Threshold �(�). The Global Masking Threshold is the combination ofthe ATH and the simultaneous masking efects produced by all components of the speech signal. The process of computing $\theta ( k )$ involves: (i) Decomposition: Identifying tonal and non-tonal maskers in the PSD of �; (ii) Spreading: Applying a spreading function $S F ( z _ { i } , z _ { j } )$ to model the frequency-dependent decay; and (iii) Aggregation: Summing individual thresholds and the ATH in the intensity domain:

$$
\theta ( j ) = 1 0 \log _ { 1 0 } \left( 1 0 ^ { T _ { q } ( j ) / 1 0 } + \sum _ { i = 1 } ^ { M } 1 0 ^ { T _ { i } ( j ) / 1 0 } \right)\tag{5}
$$

As illustrated in Fig. $^ { 3 , }$ our strategy embeds the protective perturbations $\delta$ within the resulting "Inaudible ${ \boldsymbol { Z } } { \boldsymbol { 0 } } { \boldsymbol { \mathrm { n e } } } "$ (shaded in blue). By enforcing the constraint $\mathrm { P S D } ( \delta ) \le \theta ( j )$ , we guarantee that the defense remains transparent to the user, efectively hiding the semantic conflict within the perceptual shadows of the speech signal.

## 3.3 Threat Model and Defense Objective

Threat Model. In this study, we consider a security scenario involving a victim user who shares their original audio � on public social media platforms. An unauthorized adversary, acting as a deepfake creator, aims to harvest this audio to synthesize a highly deceptive talking-face video using a 3D-field TFG model �.

• Adversarial Intent: The adversary seeks to exploit the victim’s acoustic identity to drive a digital avatar, ensuring that the synthesized lip movements are perfectly synchronized with the victim’s speech to facilitate social engineering or misinformation.

• Black-box Assumptions: We assume a stringent blackbox setting where the adversary has no prior knowledge of our defensive mechanism, its parameters, or the underlying psychoacoustic constraints. This forces the adversary to rely solely on the perceived quality of the audio for synthesis.

![](images/0ccf5e88eb5bf5c7bdea46a624eefa6af29b99ef14fb3775819cc99e874e09c9.jpg)  
Figure 3: Visualization of the frequency masking mechanism for a speech segment. The clean audio spectrum $P _ { x }$ (light gray) creates multiple individual masking umbrellas (light blue) centered at dominant tonal maskers (black dots). The Global Masking Threshold $\Theta _ { x }$ (red thick line) represents the aggregated perceptual boundary. Any adversarial perturbation � residing within the Inaudible Zone (blue shaded area) is rendered imperceptible to the human ear, as it remains below the frequency-dependent sensitivity defined by the Absolute Threshold of Hearing (ATH, green dashed line) and the simultaneous masking efects.

• Channel Robustness: Considering the practical transmission of audio over the Internet, we assume the adversary operates on a version of � that has undergone standard lossy compression $( \mathrm { e . g . }$ , MP3 or AAC transcoding) and resampling. Our threat model stipulates that a successful defense must remain efective even after such signal distortions, which are typical of social media pipelines.

Defense Objective. We seek a perturbation � such that $x _ { a d v } = x + \delta$ achieves: (1) Visual Targeted Disruption, where the cross-modal conflict causes geometric collapse in the mouth region; and (2) Acoustic Imperceptibility, satisfying $\mathcal { D } _ { p e r c } ( x , x _ { a d v } ) < \epsilon$ based on �(�).

## 4 Methodology

Our goal is to construct an imperceptible audio perturbation that preserves the perceptual quality of the shared media while preventing it from serving as a reliable driving signal for personalized 3D talking face generation. As illustrated in Fig. 4, the proposed framework consists of two tightly coupled stages: Audio Perturbation Construction and Core Semantic Conflict Establishment. The first stage generates a psychoacoustically bounded perturbation in the frequency domain, and the second stage feeds the perturbed audio into a pretrained audio-driven 3D-field talking face model to establish a persistent mismatch between audio semantics and mouth geometry. The entire process is optimized end-to-end through the asynchronous dual-loop strategy summarized in Algorithm 1.

## 4.1 Framework Overview

The key idea of our method is to shift the defense from the visual domain to the audio domain. Instead of directly injecting perturbations into facial pixels, we optimize a frequency-domain perturbation on the driving speech such that the resulting audio remains perceptually natural to humans, but induces misleading semantic cues for the downstream talking-face generator. Once the perturbed audio is used to drive facial reconstruction, the generated mouth motion becomes structurally inconsistent with the expected speech articulation, thereby breaking the learned audio-to-lip correspondence required by personalized 3D talking face generation.

As shown in Fig. 4, the upper branch first transforms the clean waveform into the time-frequency domain, computes a psychoacoustic capacity bound from the masking threshold, and optimizes a learnable perturbation variable within this admissible region. The perturbed spectrum is then converted back to the time domain to obtain the adversarial waveform. The lower branch feeds the adversarial waveform into a frozen audio feature extractor and a frozen pretrained audio-driven 3D-field renderer. A diferentiable landmark extractor is finally applied to the rendered frames, and the resulting mouth closure loss is back-propagated to update the perturbation. In this way, the optimized perturbation does not visibly alter the shared video, yet it poisons the semantic driving signal used by the talking-face model.

## 4.2 Audio Perturbation Construction

We first construct the perturbation in the frequency domain, since psychoacoustic masking constraints are naturally defined over timefrequency bins. Given a clean waveform $x \in [ - 1 , 1 ] ^ { T }$ , we compute its short-time Fourier transform:

$$
S _ { x } = S \mathrm { T F T } ( x ) ,\tag{6}
$$

where $S _ { x } \in \mathbb { C } ^ { F \times T _ { s } }$ denotes the complex spectrum, with � frequency bins and $T _ { s }$ temporal frames. We further decompose it into magnitude and phase, and define the phase template as

$$
\begin{array} { r } { P = \exp ( j \angle S _ { x } ) . } \end{array}\tag{7}
$$

Based on the clean spectrum, we estimate the psychoacoustic masking threshold map $\Theta _ { x }$ , which specifies the maximum imperceptible perturbation energy at each time-frequency location. From $\Theta _ { x } .$ , we derive a perturbation capacity map $C ,$ which provides an upper bound for the perturbation amplitude. To ensure that the optimization always remains within the admissible psychoacoustic region, we introduce a learnable variable � and parameterize the perturbation amplitude as

$$
A = \operatorname { t a n h } ( B ) \odot C ,\tag{8}
$$

where ⊙ denotes element-wise multiplication. The tanh(·) mapping constrains each element of � to (−1, 1), so that the resulting perturbation amplitude � never exceeds the psychoacoustic budget defined by �.

We then construct the complex perturbation

$$
\Delta = A \odot P ,\tag{9}
$$

and add it to the clean spectrum to obtain the adversarial spectrum

$$
S _ { a d v } = S _ { x } + \Delta .\tag{10}
$$

Finally, the adversarial waveform is reconstructed by inverse STFT:

$$
x ^ { a d v } = \operatorname { I S T F T } ( S _ { a d v } ) .\tag{11}
$$

This design corresponds to the upper branch of Fig. 4. It also matches the first several lines of Algorithm 1: we compute $S _ { x } ,$ , the

![](images/f5af98332c899bf1f043cefc19b731253cd495d9a923da645a0e24458bb9b6c8.jpg)  
Figure 4: Overview of the proposed visually lossless proactive defense framework. The upper branch constructs a psychoacous tically bounded frequency-domain perturbation and reconstructs an adversarial waveform that is imperceptible to human listeners but semantically shifted for the talking-face generator. The lower branch establishes cross-modal semantic conflict by feeding the perturbed audio into a frozen audio feature extractor and a frozen pretrained audio-driven 3D-field renderer, while a mouth landmark loss encourages collapsed mouth geometry in the rendered results. By optimizing the perturbation under psychoacoustic constraints, our method disrupts the audio-to-geometry mapping without introducing visible artifacts into the shared visual content.

phase template �, the threshold map $\Theta _ { x }$ , the perturbation bound $C ,$ and optimize the learnable variable � through iterative updates.

## 4.3 Core Semantic Conflict Establishment

After obtaining the adversarial waveform $x ^ { a d v }$ , we use it to drive a frozen audio-driven 3D-field talking face model. This stage corresponds to the lower branch of Fig. 4. Specifically, the adversarial waveform is first mapped into latent audio features:

$$
F ^ { a d v } = \mathcal { E } ( x ^ { a d v } ) ,\tag{12}
$$

where $\varepsilon$ denotes the frozen audio feature extractor. These features are then fed into a pretrained audio-driven renderer $\mathcal { R }$ together with the visual conditioning of the target identity to synthesize talking-face frames:

$$
\hat { I } _ { i } = \mathcal { R } ( d _ { i } , a _ { i } ) , \quad i = 1 , . . . , N ,\tag{13}
$$

where $d _ { i }$ denotes the visual or camera-related condition of frame $i ,$ and $a _ { i }$ is the audio condition derived from $F ^ { a d v }$

The central objective of our method is not to suppress the visual content itself, but to establish a semantic conflict between the perturbed audio cues and the expected mouth articulation. To quantify this conflict, we apply a diferentiable landmark detector $\mathcal { D }$ to each rendered frame and obtain the mouth landmarks:

$$
L _ { i } = \mathcal { D } ( \hat { I } _ { i } ) .\tag{14}
$$

Let $\phi _ { u p p e r , i } ^ { ( y ) }$ and $\phi _ { l o w e r , i } ^ { ( y ) }$ denote the vertical coordinates of the selected upper and lower inner-lip landmarks in frame �, respectively. We define the mouth deconstruction loss as

$$
L _ { m o u t h } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \phi _ { u p p e r , i } ^ { ( y ) } - \phi _ { l o w e r , i } ^ { ( y ) } \right. _ { 2 } .\tag{15}
$$

Minimizing $L _ { m o u t h }$ encourages the rendered mouth to collapse toward a closed state, even when the original clean speech should correspond to open-mouth articulations. This mismatch forces the model into an inconsistent audio-to-geometry mapping regime, thereby weakening its ability to learn stable and accurate lip motion patterns from the protected data.

This process is exactly reflected in the middle part ofAlgorithm 1: after generating $x ^ { a d v }$ and extracting $F ^ { a d v }$ , the renderer synthesizes frames one by one, the landmark detector analyzes each frame, and the mouth-related loss is accumulated over all rendered frames.

## 4.4 Optimization Objectives

To jointly enforce attack efectiveness, perceptual naturalness, and spectral smoothness, we optimize the perturbation using four complementary objectives.

Mouth deconstruction loss. The primary attack objective is the mouth closure loss:

$$
L _ { a t k } = L _ { m o u t h } .\tag{16}
$$

It directly measures whether the perturbed audio induces collapsed or semantically incorrect mouth motion in the rendered talking-face outputs.

Spectral energy regularization. To avoid trivial solutions such as suppressing the signal energy, we constrain the magnitude of the adversarial spectrum to remain close to that of the clean spectrum:

$$
L _ { e n g } = \mathbb { E } _ { k , t } \left( \left| S _ { a d v } ( k , t ) \right| - \left| S _ { x } ( k , t ) \right| \right) ^ { 2 } .\tag{17}
$$

This term encourages the defense to rely on semantic manipulation rather than brute-force signal destruction.

Psychoacoustic masking loss. Although the parameterization in Eq. (8) already limits the perturbation within a bounded region, we further penalize any residual violation of the masking threshold using

$$
L _ { m a s k } = \mathbb { E } _ { k , t } \left[ \operatorname* { m a x } ( \bar { p } _ { \delta } ( k , t ) - \Theta _ { x } ( k , t ) , 0 ) \right] ,\tag{18}
$$

where $\bar { p } _ { \delta } ( k , t )$ denotes the normalized perturbation power spectral density. This term encourages the perturbation to stay in the auditory masking shadow of the clean audio.

Temporal-frequency smoothness. To prevent isolated spikes and unnatural spectral artifacts, we regularize the perturbation ampli tude with first- and second-order smoothness constraints:

$$
L _ { s m o o t h } = \| \nabla _ { t } A \| _ { 1 } + \| \nabla _ { f } A \| _ { 1 } + \| \nabla _ { t } ^ { 2 } A \| _ { 1 } + \| \nabla _ { f } ^ { 2 } A \| _ { 1 } ,\tag{19}
$$

where $\nabla _ { t }$ and $\nabla _ { f }$ denote temporal and frequency derivatives, respectively. This regularization makes the perturbation more consistent with the natural continuity of speech spectra.

## 4.5 Asynchronous Dual-Loop Optimization

The full optimization procedure is summarized in Algorithm 1. Since the attack objective and the perceptual objective have inherently diferent priorities, we adopt an asynchronous dual-loop strategy instead of optimizing all losses with a single static weighting scheme.

Primary optimization loop. In each iteration, we first construct the perturbation amplitude $A ,$ reconstruct the adversarial waveform $x ^ { a d v }$ , and feed it into the audio-driven talking-face pipeline. The rendered frames are then evaluated by the landmark detector to compute the mouth deconstruction loss $L _ { a t k }$ . Together with the spectral energy loss and smoothness regularization, the main objective is defined as

$$
L _ { \operatorname* { m a i n } } = L _ { a t k } + \lambda _ { e n g } L _ { e n g } + \lambda _ { s m o o t h } L _ { s m o o t h } .\tag{20}
$$

This objective is responsible for pushing the perturbation toward an efective semantic attack direction while maintaining reasonable spectral behavior.

Threshold activation. Once the attack objective reaches a predefined level, i.e., $L _ { a t k } \leq \tau ,$ we mark the current perturbation as an efective candidate and activate the secondary refinement branch. Intuitively, this means the semantic conflict has already been established strongly enough to mislead the talking-face model.

Secondary refinement loop. After the attack threshold is met, we further compress the perturbation under psychoacoustic constraints, as long as the attack efect does not significantly deteriorate. The refinement objective is

$$
L _ { \mathrm { p s y } } = \lambda _ { m a s k } L _ { m a s k } + \lambda _ { s m o o t h } L _ { s m o o t h } + \lambda _ { r e g } \| B \| _ { 1 } .\tag{21}
$$

This stage corresponds to the red dashed update path in Fig. 4 and to the final conditional branch of Algorithm 1. It reduces perceptual saliency and redundant perturbation energy while preserving attack efectiveness.

Discussion. The resulting optimization procedure has a clear interpretation. The upper branch of Fig. 4 ensures that the perturbation is constructed in a psychoacoustically safe region, yielding an adversarial waveform that is imperceptible but semantically shifted. The lower branch uses this shifted audio signal to drive a frozen talking-face generator and explicitly enforces incorrect mouth dynamics through landmark-based supervision. Therefore, unlike prior visual-domain defenses that sacrifice facial quality by perturbing pixels directly, our method attacks the cross-modal driving signal itself and achieves visually lossless protection against personalized 3D talking face generation.

## 5 Experiments

Due to space limitations, we present the main quantitative and representative qualitative results in the paper, and provide additional visual comparisons in the supplementary materials for a more comprehensive evaluation. We strongly recommend watching the supplementary video.

## 5.1 Experimental Settings and Baselines

To ensure a fair and practical evaluation, we collect the demonstration videos used in prior works and construct a subset of the HDTF [37] dataset consisting of 11 identities. This setup balances computational cost and evaluation fairness, while covering diverse speaking styles, facial structures, and mouth motion patterns.

We compare our method against two representative baselines: antifake [36], a visual-domain proactive defense method, and silencer [9], an audio-removal based defense strategy. These baselines represent two fundamentally diferent protection paradigms: adding perturbations to the visual signal versus directly suppressing the driving audio.

For evaluation, we adopt both generation-quality and perceptualaudio metrics. We denote metrics with the prefix “M” to indicate that they are computed over the mouth region, which is most relevant to speech-driven lip dynamics.

Specifically, we use M-LMD, M-SSIM, and M-PSNR to evaluate the impact of diferent defenses on generated talking-face videos, where all metrics are computed with respect to the outputs produced from clean reference videos. M-LMD measures the distance between mouth landmarks and reflects lip motion consistency, with lower values indicating better preservation of natural articulation. M-SSIM and M-PSNR measure structural similarity and pixel-level fidelity within the mouth region, respectively, where higher values indicate better visual quality.

![](images/c33bf4ef17f1bfdcfa0af4ea8c5b6cd2d76a866edcdf88020042fb84c3d21217.jpg)  
Figure 5: Qualitative comparison with visual-domain defense methods. From left to right, each row shows the clean reference result and the outputs under diferent defenses. Existing visual perturbation methods introduce noticeable artifacts in facial regions, such as grainy noise, texture corruption, and color distortion. In contrast, our method maintains visually lossless appearance while still preventing accurate speech-driven talking-face generation.

Table 1: Comparison of diferent defense methods on lipmotion disruption, deviation from clean-driven generation, and perceptual audio quality.
<table><tr><td>Methods</td><td>M-LMD↑</td><td>M-SSIM↓</td><td>M-PSNR↓</td><td>CDPAM↓</td></tr><tr><td>antifake</td><td>3.32034</td><td>0.83584</td><td>28.12566</td><td>0.457559</td></tr><tr><td>silencer</td><td>10.0028</td><td>0.5909</td><td>28.16</td><td>0</td></tr><tr><td>ours</td><td>3.23093</td><td>0.84236</td><td>28.34491</td><td>0.116151</td></tr></table>

In addition, we adopt CDPAM [19] to measure the perceptual distance between the defended audio and the clean audio, where lower values indicate higher perceptual similarity.

## 5.2 Quantitative Results

Table 1 reports the quantitative comparison across diferent defense methods. Since the objective of proactive defense is to prevent the protected media from serving as a reliable driving signal for talking-face generation, stronger defense performance should lead to generated results that deviate more from those produced using clean reference inputs. Therefore, larger M-LMD and smaller M-SSIM/M-PSNR indicate stronger disruption of speech-driven facial synthesis, while lower CDPAM indicates higher perceptual similarity between the defended and clean audio.

Compared with antifake, our method achieves a substantially lower CDPAM score, indicating that the introduced perturbation is significantly less perceptible in the audio domain. At the same time, it maintains a comparable level of defense efectiveness in terms of M-LMD, M-SSIM, and M-PSNR, showing that our method can efectively interfere with downstream talking-face generation without noticeably compromising perceptual quality.

Although silencer achieves the strongest disruption according to M-LMD, M-SSIM, and M-PSNR, this result mainly stems from directly degrading the protected visual input itself. Since the quality of the visual input is severely compromised, the downstream generated videos also deviate more significantly from those driven by clean reference inputs, which naturally leads to stronger disruption metrics. However, this comes at the cost of substantially worse visual appearance and user experience. As also evidenced by the qualitative comparisons in Fig. 5, such a strategy introduces obvious perceptual degradation and produces visually less acceptable protected content.

In contrast, our method achieves efective defense while preserving the natural appearance of the shared visual content. This makes it more suitable for realistic deployment scenarios, where proactive protection should not require sacrificing the viewing experience of benign users. Overall, these results demonstrate that our method provides a more practical trade-of between defense efectiveness and perceptual quality.

## 5.3 Qualitative Results

Figure 5 presents qualitative comparisons between our method and representative visual-domain defense approaches [9, 17, 18, 27, 32].

As shown in the figure, existing visual perturbation methods inevitably introduce visible artifacts on the face, including grainy noise, color inconsistency, and corrupted local textures. These distortions are particularly evident in identity-sensitive regions such as the cheeks, lips, and facial contours, which significantly degrade visual realism and user experience.

In contrast, our method achieves visually lossless protection, where the facial appearance remains clean, natural, and free of noticeable artifacts across diferent identities. The overall texture, color distribution, and facial structure are well preserved, making the defended content perceptually indistinguishable from clean data for human observers.

Despite this high visual fidelity, our method can still efectively disrupt the speech-driven synthesis process, demonstrating that strong protection does not necessarily require sacrificing visual quality. This property is crucial for real-world deployment, where user-uploaded content should remain visually acceptable while being resistant to unauthorized talking-face generation.

## References

[1] ShahRukh Athar, Zexiang Xu, Kalyan Sunkavalli, Eli Shechtman, and Zhixin Shu. 2022. Rignerf: Fully controllable neural 3d portraits. In Proceedings ofthe IEEE/CVF conference on Computer Vision and Pattern Recognition. 20364–20373.

[2] Chongke Bi, Xiaoxing Liu, and Zhilei Liu. 2024. Nerf-ad: Neural radiance field with attention-based disentanglement for talking face synthesis. In ICASSP 2024- 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 3490–3494.

[3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 18392–18402.

[4] Nicholas Carlini and David Wagner. 2018. Audio adversarial examples: Targeted attacks on speech-to-text. In 2018 IEEE security and privacy workshops (SPW). IEEE, 1–7.

[5] Lele Chen, Zhiheng Li, Ross K Maddox, Zhiyao Duan, and Chenliang Xu. 2018. Lip movements generation at a glance. In Proceedings ofthe European conference on computer vision (ECCV). 520–535.

[6] Lele Chen, Ross K Maddox, Zhiyao Duan, and Chenliang Xu. 2019. Hierarchical cross-modal talking face generation with dynamic pixel-wise loss. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 7832–7841.

[7] Kyusun Cho, Joungbin Lee, Heeji Yoon, Yeobin Hong, Jaehoon Ko, Sangjun Ahn, and Seungryong Kim. 2024. Gaussiantalker: Real-time talking head synthesis with 3d gaussian splatting. In Proceedings ofthe 32nd ACM International Conference on Multimedia. 10985–10994.

[8] Nigel Francis. 2025. Deepfake Detection and Defense: An Analysis of Techniques and Robustness. (2025).

[9] Yuan Gan, Jiaxu Miao, Yunze Wang, and Yi Yang. 2025. Silence is Golden: Leverag ing Adversarial Examples to Nullify Audio Control in LDM-based Talking-Head Generation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 13434–13444.

[10] Yudong Guo, Keyu Chen, Sen Liang, Yong-Jin Liu, Hujun Bao, and Juyong Zhang. 2021. Ad-nerf: Audio driven neural radiance fields for talking head synthesis. In Proceedings ofthe IEEE/CVF international conference on computer vision. 5784– 5794.

[11] Jaehwan Jeong, Sumin In, Sieun Kim, Hannie Shin, Jongheon Jeong, Sang Ho Yoon, Jaewook Chung, and Sangpil Kim. 2025. Faceshield: Defending facial image against deepfake threats. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10364–10374.

[12] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. 2024. Talkinggaussian: Structure-persistent 3d talking head synthesis via gaussian splatting. In European Conference on Computer Vision. Springer, 127–145.

[13] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Jun Zhou, and Lin Gu. 2025. Instag: Learning personalized 3d talking head from few-second video. In Proceedings of the Computer Vision and Pattern Recognition Conference. 10690–10700.

[14] Jiahe Li, Jiawei Zhang, Xiao Bai, Jun Zhou, and Lin Gu. 2023. Eficient regionaware neural radiance fields for high-fidelity talking portrait synthesis. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 7568–7578.

[15] Tianqi Li, Ruobing Zheng, Minghui Yang, Jingdong Chen, and Ming Yang. 2025. Ditto: Motion-space difusion for controllable realtime talking head synthesis. In Proceedings ofthe 33rd ACM International Conference on Multimedia. 9704–9713.

[16] Chumeng Liang and Xiaoyu Wu. 2023. Mist: Towards improved adversarial examples for difusion models. arXiv preprint arXiv:2305.12683 (2023).

[17] Chumeng Liang and Xiaoyu Wu. 2023. Mist: Towards improved adversarial examples for difusion models. arXiv preprint arXiv:2305.12683 (2023).

[18] Chumeng Liang, Xiaoyu Wu, Yang Hua, Jiaru Zhang, Yiming Xue, Tao Song, Zhengui Xue, Ruhui Ma, and Haibing Guan. 2023. Adversarial example does good: Preventing painting imitation from difusion models via adversarial examples. arXiv preprint arXiv:2302.04578 (2023).

[19] Pranay Manocha, Zeyu Jin, Richard Zhang, and Adam Finkelstein. 2021. CDPAM: Contrastive learning for perceptual audio similarity. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 196–200.

[20] Momina Masood, Mariam Nawaz, Khalid Mahmood Malik, Ali Javed, Aun Irtaza, and Hafiz Malik. 2023. Deepfakes generation and detection: state-of-the-art, open challenges, countermeasures, and way forward: Deepfakes generation and detection: state-of-the-art, open challenges, countermeasures, and way forward. Applied intelligence 53, 4 (2023), 3974–4026.

[21] Paarth Neekhara, Shehzeen Hussain, Prakhar Pandey, Shlomo Dubnov, Julian McAuley, and Farinaz Koushanfar. 2019. Universal adversarial perturbations for speech recognition systems. arXiv preprint arXiv:1905.03828 (2019).

[22] Ziqiao Peng, Wentao Hu, Yue Shi, Xiangyu Zhu, Xiaomei Zhang, Hao Zhao, Jun He, Hongyan Liu, and Zhaoxin Fan. 2024. Synctalk: The devil is in the synchronization for talking head synthesis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 666–676

[23] KR Prajwal, Rudrabha Mukhopadhyay, Vinay P Namboodiri, and CV Jawahar. 2020. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings ofthe 28th ACM international conference on multimedia. 484–492.

[24] Yao Qin, Nicholas Carlini, Garrison Cottrell, Ian Goodfellow, and Colin Rafel. 2019. Imperceptible, robust, and targeted adversarial examples for automatic speech recognition. In International conference on machine learning. PMLR, 5231– 5240.

[25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

[26] Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander Madry. 2023. Raising the cost of malicious ai-powered image editing. arXiv preprint arXiv:2302.06588 (2023).

[27] Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander Madry. 2023. Raising the Cost of Malicious AI-Powered Image Editing. In International Conference on Machine Learning. PMLR, 29894–29918

[28] Lea Schönherr, Katharina Kohls, Stefen Zeiler, Thorsten Holz, and Dorothea Kolossa. 2018. Adversarial attacks against automatic speech recognition systems via psychoacoustic hiding. arXiv preprint arXiv:1808.05665 (2018).

[29] Jiaxiang Tang, Kaisiyuan Wang, Hang Zhou, Xiaokang Chen, Dongliang He, Tianshu Hu, Jingtuo Liu, Ziwei Liu, Gang Zeng, and Jingdong Wang. 2025. Realtime neural radiance talking portrait synthesis via audio-spatial decomposition. International Journal ofComputer Vision 133, 9 (2025), 6362–6373.

[30] Luisa Verdoliva. 2020. Media forensics and deepfakes: an overview. IEEE journal ofselected topics in signal processing 14, 5 (2020), 910–932.

[31] Hanhui Wang, Yihua Zhang, Ruizheng Bai, Yue Zhao, Sijia Liu, and Zhengzhong Tu. 2025. Edit away and my face will not stay: Personal biometric defense against malicious generative editing. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 23806–23816.

[32] Haotian Xue, Chumeng Liang, Xiaoyu Wu, and Yongxin Chen. 2023. Toward efective protection against difusion-based mimicry through score distillation. In The Twelfth International Conference on Learning Representations.

[33] Hiromu Yakura and Jun Sakuma. 2018. Robust audio adversarial example for a physical attack. arXiv preprint arXiv:1810.11793 (2018).

[34] Zhenhui Ye, Ziyue Jiang, Yi Ren, Jinglin Liu, Jinzheng He, and Zhou Zhao. 2023. Geneface: Generalized and high-fidelity audio-driven 3d talking face synthesis. arXiv preprint arXiv:2301.13430 (2023).

[35] Zhenhui Ye, Tianyun Zhong, Yi Ren, Jiaqi Yang, Weichuang Li, Jiawei Huang, Ziyue Jiang, Jinzheng He, Rongjie Huang, Jinglin Liu, et al. 2024. Real3d-portrait: One-shot realistic 3d talking portrait synthesis. arXiv preprint arXiv:2401.08503 (2024).

[36] Zhiyuan Yu, Shixuan Zhai, and Ning Zhang. 2023. Antifake: Using adversarial audio to prevent unauthorized speech synthesis. In Proceedings ofthe 2023 ACM SIGSAC Conference on Computer and Communications Security. 460–474.

[37] Zhimeng Zhang, Lincheng Li, Yu Ding, and Changjie Fan. 2021. Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 3661–3670.

Algorithm 1: 3DMM-Guided Imperceptible Audio Attack   
for Talking Face Rendering   
Input: Clean waveform �   
Output: Adversarial waveform $x ^ { a d v }$   
1 Compute clean spectrum $S _ { x } = \mathrm { S T F T } ( x )$ and phase template   
$P = \exp ( j \angle S _ { x } ) ;$   
2 Compute psychoacoustic threshold map $\Theta _ { x }$ from $S _ { x } ;$   
3 Compute perturbation bound � from $\Theta _ { x }$   
4 Initialize perturbation variable � randomly;   
5 Initialize threshold Flag Flag ← False;   
6 for � = 1 to � do   
7 � ← tanh(�) ⊙ �, $\Delta  A \odot P ;$   
8 $x ^ { a d v } \gets \mathrm { I S T F T } ( S _ { x } + \Delta ) ;$   
9 �<sup>���</sup> $ \mathcal { E } ( x ^ { a d v } ) ;$   
10 $L _ { a t k } \gets 0 ;$   
11 foreach frame $d _ { i }$ do   
12 Build audio condition $a _ { i }$ from $F ^ { a d v } ;$   
13 Render frame $\hat { I } _ { i } \gets \mathcal { R } ( d _ { i } , a _ { i } ) ;$   
14 Detect facial landmarks $L _ { i } \gets \mathcal { D } ( \hat { I } _ { i } ) ;$   
15 Compute mouth-closure loss $L _ { 3 d m m } ^ { ( i ) } ;$   
16 $L _ { a t k } \gets L _ { a t k } + L _ { 3 d m m } ^ { ( i ) } ;$   
17 end   
18 $\begin{array} { r } { L _ { a t k } \gets \frac { 1 } { N } L _ { a t k } ; } \end{array}$   
19 Compute psychoacoustic masking loss $L _ { m a s k } ;$   
20 Compute spectral energy regularization $L _ { e n g } ;$   
21 Compute temporal-frequency smoothness loss �<sub>�����ℎ</sub>;   
// Primary optimization   
22   
${ \cal L } _ { \mathrm { m a i n } } = { \cal L } _ { a t k } + \lambda _ { e n g } { \cal L } _ { e n g } + \lambda _ { s m o o t h } { \cal L } _ { s m o o t h } + \lambda _ { m a s k } { \cal L } _ { m a s k }$   
Update � by minimizing �<sub>main</sub>;   
23 if $L _ { a t k } \le \tau$ then   
24 flag ← True;   
25 save current $x ^ { a d v } ;$   
26 end   
// Secondary refinement after reaching the   
attack threshold   
27 if flag = True and $L _ { a t k } \leq \tau + \delta$ then   
28   
${ \cal L } _ { \mathrm { p s y } } = \lambda _ { m a s k } L _ { m a s k } + \lambda _ { \mathrm { : } }$ <sub>�����ℎ</sub>�<sub>�����ℎ</sub> $+ \lambda _ { r e g } | | B | | _ { 1 }$   
Update � by minimizing $L _ { \mathrm { p s y } } ;$   
29 end   
30 end   
31 return current $x ^ { a d v } ;$

Table 2: Ablation study on psychoacoustic masking, band constraint, and smoothing.
<table><tr><td>Psycho.</td><td>Band</td><td>Smooth</td><td>CDPAM↓</td></tr><tr><td>一</td><td>√</td><td>√</td><td>0.0817</td></tr><tr><td></td><td></td><td>√</td><td>0.0941</td></tr><tr><td></td><td></td><td></td><td>0.1379</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.0634</td></tr></table>

## A Appendix

## A.1 Ablation Experiments

As shown in Table 2, each component ofour method consistently improves the perceptual quality of the adversarial audio. Without any constraint, the perturbation causes the largest perceptual distortion, yielding the worst CDPAM score of 0.1379. Introducing temporal smoothing reduces CDPAM to 0.0941, showing that smoothing helps suppress abrupt temporal artifacts. Further adding the band constraint improves the score to 0.0817, indicating that constraining perturbations to appropriate frequency regions enhances imperceptibility. Finally, incorporating psychoacoustic masking achieves the best result of 0.0634, demonstrating its critical role in reducing perceptually noticeable noise. These results verify that the three components are complementary and jointly contribute to more imperceptible adversarial audio.