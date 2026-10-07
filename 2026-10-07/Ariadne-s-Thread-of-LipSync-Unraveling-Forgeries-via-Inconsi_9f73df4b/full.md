# Ariadne’s Thread of LipSync: Unraveling Forgeries via Inconsistency between Lip Motions and Head Poses

Tianyi She <sup>\*</sup> <sup>1</sup> Jiawei Liu <sup>\*</sup> <sup>2</sup> Weifeng Liu <sup>3</sup> Hanqing Zhao <sup>4</sup> Weiming Zhang <sup>1</sup> Kejiang Chen <sup>1</sup>

## Abstract

Recent advances in LipSync generation technology have led to the creation of highly realistic videos, posing severe societal risks. However, existing defense strategies struggle against LipSync forgeries, as advanced LipSync generation methods not only achieve better lip synchronization but also eliminate visual artifacts. An important reason is that they overlook an inherent biological coupling between lip movements and head poses in natural speech videos. In this paper, we propose LipDA, a novel framework for joint Lip-Sync Detection and Attribution, which takes advantage of the inconsistency between head and lip. For detection, the framework learns to quantify this discrepancy by contrasting lip and pose features from authentic versus forged videos. For attribution, our method is designed to capture the unique temporal dynamics and audio-visual synchronization patterns that act as the fingerprint of models, enabling source tracing. We conduct extensive experiments on two challenging LipSync datasets as well as our own proposed large-scale and multi-generator dataset. LipDA achieves over 97% AUC in detection and 97.5% accuracy in model attribution, significantly outperforming existing methods. Code and the proposed LipSync-A dataset are available at https: //github.com/AnsonShe/LipDA.

## 1. Introduction

The rapid proliferation of powerful generative models (Goodfellow et al., 2014; Ho et al., 2020; Yang et al., 2023) has fueled the creation of synthetic multimodal content at an unprecedented scale. Within this landscape, LipSync generation has emerged as a transformative tool widely deployed in digital avatar creation and virtual anchoring (Gan et al., 2025; Meng et al., 2024). This technology operates by utilizing a reference audio or video driving signal to manipulate the lip region of a source identity, enforcing strict synchronization to synthesize realistic talking heads while preserving the speaker’s original identity.

![](images/b6288e076c9954c816f1283cb9bceb828dfbd020fa8142bcbb8557de4a9bfdea.jpg)  
Figure 1. Taxonomy of forgeries and detection comparison. (Left) Distinct from identity attacks, LipSync constitutes a fact attack, fabricating videos where the target utters speech never spoken. (Right) Vulnerability of existing defenses versus our method. LipDA robustly identifies forgeries by capturing the inconsistencies between lip and head dynamics.

However, the widespread availability of open-source models has introduced severe security risks. Adversaries increasingly exploit LipSync for disinformation and financial fraud (Suwajanakorn et al., 2017), as evidenced by a recent incident in which criminals impersonated a corporate executive to defraud a company of \$25 million (James et al., 2026). In contrast to traditional DeepFakes such as face swapping, attribute editing, and entire face synthesis (Xu et al., 2022; Yan et al., 2023), LipSync preserves the victim’s authentic identity and visual context while fabricating speech that was never uttered, as illustrated in Fig. 1. This design renders it a more deceptive and stealthier form of forgery. Consequently, advanced defense frameworks are urgently needed to both detect these subtle forgeries and attribute them to their source models for forensic accountability.

Current defense strategies struggle with evolving LipSync generation. First, general DeepFake detectors (Tan et al., 2024b;a; Ojha et al., 2023), which primarily target spatialfrequency anomalies in identity-tampering scenarios, often falter in the LipSync domain. Since LipSync involves finegrained, localized manipulations, these sparse forgery signals are easily diluted by overwhelming authentic facial features, leading to a loss of discriminative power. Second, specific LipSync detectors (Haliassos et al., 2022; Wang et al., 2023b; Smeu et al., 2025) relying on audio-visual inconsistencies or local artifacts are becoming ineffective. Recent diffusion-based generators synthesize high-fidelity textures while stabilizing lip dynamics. These advancements effectively mitigate the viseme-phoneme mismatches and visual artifacts that prior methods relied upon. Finally, existing literature largely overlooks the critical task of model attribution. Relying solely on binary classification (real vs. fake) is insufficient for real-world forensics, where identifying the specific source generator is paramount for accountability.

To address these challenges, we investigate the fundamental discrepancy between authentic speech and LipSync forgeries. In natural speech, lip articulation and head dynamics are inherently coupled, driven by a unified neuro-cognitive control system (Chen et al., 2020; Gick et al., 2020). For instance, prosodic emphasis deterministically coordinates with head keypoints. Although advanced models can successfully synthesize plausible head motion, they operate under a decoupled control paradigm. Specifically, the lip movement is subject to hard constraints from the audio input to ensure semantic accuracy, whereas the head pose is generated via probabilistic sampling from a learned distribution or transplanted from an uncorrelated driving source. This fundamental decoupling disrupts the intrinsic biological synchronization, resulting in subtle yet quantifiable lip-pose inconsistencies, which serve as the “Ariadne’s Thread” for detecting forgeries. Furthermore, we observe that the distinct synthesis mechanisms of different generator families imprint unique temporal fingerprints on the video, providing a discriminative basis for source model attribution.

Inspired by above analysis, we propose a novel framework for joint LipSync forgeries Detection and Attribution (LipDA). For detection, we introduce a lip-pose contrastive encoding mechanism. By projecting temporal lip features and head dynamics into a shared physiological manifold for detection classification, this module effectively discriminates natural biological coupling from the disrupted motion characteristic of forgeries. For attribution, we incorporate modality synchronization and temporal dynamic modules to capture the unique architectural fingerprints imprinted by distinct synthesis models via temporal motion patterns. Furthermore, to address the scarcity of high-quality forensic resources in this domain, we construct LipSync-A, providing fine-grained labels explicitly designed for attribution tasks. Extensive experiments validate LipDA’s effectiveness, demonstrating superior cross-dataset generalization and robustness compared to state-of-the-art methods.

Our work makes the following key contributions:

• Physiological Signal Unveiling. We reveal that the intrinsic inconsistency between lip motion and global head pose is inherently disrupted by LipSync generation, and its distinct temporal dynamics serve as a reliable fingerprint for source model attribution.

• Unified Detection and Attribution Framework. We propose the first unified framework to jointly address LipSync detection and source attribution. By integrating contrastive learning with cross-modal temporal modeling, our two-stage design effectively captures discriminative generative patterns.

• Superior Generalization and Robustness. Extensive experiments demonstrate that LipDA significantly outperforms existing detectors. Crucially, it exhibits strong cross-domain generalization to unseen models and general deepfakes, while maintaining stability under challenging visual and audio perturbations.

## 2. Related Work

## 2.1. LipSync Generation

Early approaches relied on geometric transformations (Wiles et al., 2018) and template-based rules (Bouaziz et al., 2013), which were labor-intensive and often lacked visual fidelity. The advent of deep learning enabled more effective synthesis. Siarohin et al. (Siarohin et al., 2019) proposed a video-driven framework leveraging keypoint transformations to predict motion, though it yielded unstable results under large pose variations and occlusions. In contrast, audio-driven approaches focus on mapping audio features to visual synthesis. Seminal works like Wav2Lip (Prajwal et al., 2020) utilized GANs with expert discriminators to enforce precise LipSync, though often at the expense of pose diversity. To enhance controllability and realism, recent research has shifted towards sophisticated generative paradigms. Methods such as SadTalker (Zhang et al., 2023a) and VAE-based frameworks (Wang et al., 2024) employ intermediate representations to decouple motion from appearance. By modeling the probabilistic distribution of motion, diffusion-based models (Ma et al., 2023; Ji et al., 2025) generate highly expressive dynamics and high-fidelity textures. This effectively eliminates the jittering artifacts prevalent in earlier generations, thereby rendering traditional artifact-based detection cues increasingly ineffective.

## 2.2. LipSync Forgery Detection

Since LipSync manipulations require detectors to capture fine-grained localized dynamics, early general approaches targeting spatial artifacts (Wu et al., 2022) or frequency anomalies (Tan et al., 2024a) often falter in this domain due to the subtle nature of lip modifications. Consequently, research has pivoted towards modeling temporal coherence and semantic consistency. Visual-only methods, such as LipForensics (Haliassos et al., 2021), exploit high-level semantic irregularities using pre-trained lip-reading networks, while RealForensics (Haliassos et al., 2022) and FTCN (Zheng et al., 2021) leverage self-supervised learning to detect temporal discontinuities in facial motion. More critically, considering the multimodal nature of LipSync, detecting audio-visual inconsistencies has become a dominant paradigm. Recent state-of-the-art methods leverage crossmodal dissonance as a primary forensic cue. For instance, AVAD (Feng et al., 2023) and SpeechForensics (Liang et al., 2024) learn joint audio-visual representations to spot synchronization anomalies. Similarly, LipFD (Liu et al., 2024) and AVH-align (Smeu et al., 2025) explicitly contrast visual lip dynamics with audio spectrograms to identify finegrained mismatches. Nevertheless, these strategies face diminishing returns. As diffusion-based generators synthesize high-fidelity textures and strictly synchronized lip motions, the explicit artifacts and synchronization errors relied upon by prior methods are increasingly mitigated. Furthermore, existing literature remains largely confined to binary classification, overlooking the critical task of source model attribution necessary for forensic accountability.

## 3. Analysis of LipSync Forgery Signals

## 3.1. Inter-Class Signal: Real vs. Fake

![](images/53f1b36ff5e7a0c49c2f286b0c31832996d3a388591097c064e0f25d65c9a8d9.jpg)  
Figure 2. Illustration of the two primary LipSync forgery paradigms. Top (Video-driven): Motion and head pose are transplanted from a driving video, creating localized temporal artifacts. Bottom (Audio-driven): Lip motion is synthesized from audio, severing the global link to head motion, which remains static or is generated independently.

In natural speech, lip motion is inherently coordinated with surrounding facial muscle activity and is accompanied by slight head movements that correlate with speech rhythm (Gick et al., 2020). We find that current LipSync forgeries fundamentally disrupt this physiological coupling, which is an inherent limitation of their synthesis mechanisms. We analyze the two primary paradigms, as illustrated in Fig. 2. Video-driven approaches directly learn and transplant motion from a driving video onto a source identity. This process inherently replaces the source’s intrinsic motor patterns with the driver’s head-pose and expression dynamics. The result is subtle temporal artifacts, as this foreign lip-head pattern is unnatural for the source identity.

Audio-driven methods commonly decompose LipSync generation into two independent sub-problems, namely an audio-to-lip mapping and a head-pose generation pathway. This decomposition manifests in different forms across architectures. Reconstruction-based approaches such as Wav2Lip mask the lower face and synthesize lip movements under lip-reading constraints, while retaining the head pose statically from the source frame. Diffusion-based approaches instead learn a dedicated head pose predictor that fuses audio features with trained image embeddings to produce motion parameters, which are then injected into the denoising process. By treating lip articulation and head pose as separable conditional distributions, they sever the intrinsic global coupling between the two parts. Furthermore, in contrast to the relatively well defined phoneme-toviseme mapping, lip-head coordination is a complex, highdimensional, and global task intrinsically tied to speaker identity. These inherent deficiencies in existing LipSync synthesis mechanisms provide crucial and exploitable cues for detection (see Appendix B.3 for more details).

We validate the aforementioned observations through the facial action unit (AU) analysis in Fig. 3. Video-driven forgeries correspond to significantly reduced intensity values, indicating that facial activity is suppressed by the influence of the driving signal. These transplanted foreign motion patterns result in unnatural lip trajectories. Conversely, while audio-driven forgeries successfully mimic authentic lip dynamics, they exhibit erratic peaks in AU17, revealing a failure to achieve lip-pose coordination.

## 3.2. Intra-Class Signal: Model Fingerprints

Existing data-driven attribution methods primarily rely on dataset-specific artifacts rather than intrinsic generative mechanisms, rendering them susceptible to training set bias. Therefore, discriminative intra-class signals inherent to the model architecture serve as a more reliable cue for the attribution task. Building on the concept of generative fingerprints (Song et al., 2024), we empirically hypothesize that LipSync model families possess such fingerprints in their temporal dynamics and motion patterns. This is because different generative architectures employ distinct approaches to modeling temporal coherence, such as adversarial processes or latent reconstruction, which impose unique statistical constraints on the resulting motion patterns.

![](images/4e9354bf693a212db9bdd0c19a84db234735f8dc5fcef3977ec5b273dda7f79b.jpg)  
Figure 3. AU intensity analysis on original video sequences and two categories of forgery patterns. Higher intensity values indicate heightened activity of specific facial muscles. Two representative action units, AU12 (mouth corner pull) and AU17 (chin raise), are analyzed.

To validate this hypothesis, we sample forged videos generated by different model families using disjoint source identities and diverse driving signals. For feature extraction, we first track facial landmarks across the frame sequence and compute the 6-Degrees-of-Freedom (6-DoF) head pose vector sequence. This raw motion sequence is then encoded by a spatial-temporal transformer to extract feature embeddings. Finally, we apply t-SNE to visualize the distribution of these embeddings. As shown in Fig. 4, the results clearly reveal well-separated clusters, strongly suggesting that the temporal dynamics of inter-frame head motion indeed provide a separable feature space for model attribution.

## 3.3. Dataset construction

Existing datasets often cover only a limited number of generators or rely on outdated synthesis pipelines. To bridge this gap, we construct LipSync-A, the first large-scale dataset explicitly designed for LipSync forensics. As shown in Table 1, it encompasses 15 SOTA generators spanning 7 architectures, providing 16,000 synthetic videos for LipSync research. Crucially, every video in LipSync-A is meticulously labeled with its source generator, enabling us to evaluate the attribution performance. The overall data-generation pipeline and comprehensive details on the selected generators and dataset splits are provided in the Appendix A.

![](images/dc70787316f1561b45bb2ac07303f392897a60429634cf59343f73d193525952.jpg)  
Figure 4. t-SNE visualization of pose features from LipSync forgeries generated by different model families.

Table 1. Comparison with existing LipSync datasets. #Gens. and #Fakes denote the number of generator architectures and forged video samples, respectively.
<table><tr><td>Dataset</td><td></td><td>Detection Attribution # Gens. # Fakes</td><td></td><td></td></tr><tr><td>TalkHeadBench(Xiong et al., 2025)</td><td>√</td><td>x</td><td>2</td><td>2,984</td></tr><tr><td>AVLips (Liu et al., 2024)</td><td>√</td><td>x</td><td>2</td><td>4,206</td></tr><tr><td>LAV-DF (Cai et al., 2022)</td><td>√</td><td>x</td><td>1</td><td>99,873</td></tr><tr><td>PolyGlotFake (Hou et al., 2024)</td><td>√</td><td>x</td><td>2</td><td>14,472</td></tr><tr><td>AV-Deepfake1M (Cai et al., 2024)</td><td>√</td><td>x</td><td>1</td><td>860,039</td></tr><tr><td>LipSync-A(Ours)</td><td>√</td><td>√</td><td>7</td><td>16,000</td></tr></table>

## 4. Methodology

In this section, we introduce a unified two-stage framework designed for joint LipSync detection and source attribution, as illustrated in Fig. 5. Stage I employs a contrastive mechanism to capture lip-pose inconsistencies for robust binary detection, while Stage II integrates audio-visual synchronization (MSM) and temporal motion patterns (TDM) to disentangle fine-grained model fingerprints for attribution.

## 4.1. Problem Formulation

Let the dataset D consist of real videos $x _ { \mathrm { r e a l } }$ and LipSyncgenerated forgeries x . Given an input video $x ^ { \mathrm { v i d e o } }$ , we extract the frame sequence $\{ x _ { 1 } ^ { i } , \ldots , x _ { m } ^ { i } \}$ and audio signal $x ^ { a }$ . The detection task is a binary classification problem:

$$
y ^ { \mathrm { v i d e o } } = f _ { \mathrm { d e t } } ( x ^ { \mathrm { v i d e o } } ) \in \{ r e a l : 0 , f a k e : 1 \} .\tag{1}
$$

The objective of Stage I is to train a robust detection classifier $f _ { \mathrm { d e t } } ( \cdot )$ . For attribution, we define a candidate generator set $\mathcal { M } = \{ \mathrm { m o d e l } _ { 1 } , \mathrm { m o d e l } _ { 2 } , \dots , \mathrm { m o d e l } _ { n } \}$ , and aim to identify the source model mˆ for a forged sample:

$$
\hat { m } = f _ { \mathrm { a t t } } ( x ^ { \mathrm { v i d e o } } ) \in \mathcal { M } .\tag{2}
$$

![](images/b6fb2399efa5ddbbab9a331244f8f344ae98c43f763469df4f0b9e0d6ba84018.jpg)  
Figure 5. Overview of our unified LipDA framework. 2-Stage Training (Left): In Stage I Detection, a dual-branch encoder is trained with a contrastive module to learn the lip-head pose inconsistency. In Stage II Attribution, Stage I encoders are frozen, and the Modality Synchronization (MSM) and Temporal Dynamic (TDM) modules are trained to capture unique generative fingerprints. Inference (Right): The unified pipeline integrates all components for joint forgery detection and source attribution.

## 4.2. Stage I: Pose-Lip Contrastive Optimization

Pose and Lip Encoders. Stage I aims to capture the intrinsic consistency between head poses and lip motions that is often disrupted in forgeries. We first adopt a sliding window of $T$ frames to form video clips. Since LipSync manipulations primarily affect the lower facial region, we first extract the lip region $\mathbf { \bar { \chi } } _ { L } = \{ l _ { i } \in \mathbb { R } ^ { 3 \times H _ { \mathrm { l i p } } \times W _ { \mathrm { l i p } } } \} _ { i = 1 } ^ { \overline { { T } } }$ from each frame. These patches are processed by a ResNet-based encoder $f _ { \mathrm { r e s } } ( \cdot )$ . The resulting per-frame embeddings are concatenated, and projected into a latent space $\mathbb { R } ^ { o }$ via a dedicated projection head $g _ { l } ( \cdot )$ , yielding the final lip embedding.

$$
O _ { l } = g _ { l } ( \mathrm { C o n c a t } ( f _ { \mathrm { r e s } } ( l _ { i } ) ) ) .\tag{3}
$$

To reduce sensitivity to pixel-level variations, we represent head motion via facial landmark trajectories $P = \{ p _ { i } \in$ $\mathbb { R } ^ { 4 6 8 \times 3 } \} _ { i = 1 } ^ { T }$ . This sequence is processed by a separate pose projection head $g _ { p } ( \cdot )$ to embed it into the same latent space:

$$
O _ { p } = g _ { p } ( P ) \in \mathbb { R } ^ { o } .\tag{4}
$$

Contrastive and Detection Objectives. We aim to learn a shared latent space $\mathbb { R } ^ { o }$ where the lip embedding $O _ { l }$ and pose embedding $O _ { p }$ are discriminative. For authentic videos $( y ~ = ~ 0 )$ , these embeddings should be aligned, whereas forgeries $( y = 1 )$ should diverge. To enforce this property, we employ a margin-based contrastive loss $\mathcal { L } _ { \mathrm { a l i g n } }$ . This objective explicitly minimizes the $L _ { 2 }$ discrepancy for real samples while penalizing forged pairs that fall within a margin γ.

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a l i g n } } = \mathbb { E } _ { y = 0 } \left[ \| O _ { p } - O _ { l } \| _ { 2 } ^ { 2 } \right] } \\ & { \qquad + \mathbb { E } _ { y = 1 } \left[ \operatorname* { m a x } \left( 0 , \gamma - \| O _ { p } - O _ { l } \| _ { 2 } ^ { 2 } \right) \right] . } \end{array}\tag{5}
$$

To achieve alignment, an MLP is trained on the concatenated features Concat $( O _ { p } , O _ { l } )$ . It is optimized with a standard binary cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { d e t } } = - \mathbb { E } \left[ y \log ( \hat { y } ) + ( 1 - y ) \log ( 1 - \hat { y } ) \right] .\tag{6}
$$

The final Stage I objective is a joint optimization of both losses, and $\lambda _ { \mathrm { a l i g n } }$ is a balancing hyperparameter.

$$
\mathcal { L } _ { \mathrm { s t a g e 1 } } = \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } + \mathcal { L } _ { \mathrm { d e t } } .\tag{7}
$$

## 4.3. Stage II: Cross-Modal Temporal Modeling

While Stage I distinguishes real from fake, Stage II is designed for attribution, identifying which generative model created a forgery. This stage uses two dedicated modules TDM and MSM, which are trained independently from

Stage I but used concurrently during inference. To capture unique temporal fingerprints, we extract three feature streams from the input video: The facial keypoint sequence $\mathbf { K } \in \mathbb { R } ^ { T \times k \times 3 }$ , the lip ROI sequence $\mathbf { L } \in \dot { \mathbb { R } } ^ { T \times c \times h \times w }$ , and the aligned MFCC features $\mathbf { A } \in \mathbb { R } ^ { T \times d _ { a } }$ . Here, $T$ is the sequence length; k is the number of keypoints; $c , h ,$ w are the channels, height, and width of the lip ROIs; and $d _ { a }$ is the MFCC feature dimension.

Temporal Dynamic Module (TDM). To capture generatorspecific motion artifacts, the TDM module encodes the keypoint sequence K. It employs a Temporal-CNN to learn local motion patterns, followed by a Bi-LSTM to model longrange dependencies, yielding the temporal feature z<sub>temp</sub>:

$$
z _ { \mathrm { t e m p } } = f _ { \mathrm { B i L S T M } } ( \mathrm { C o n v } ( \mathbf { K } ) ) .\tag{8}
$$

Modality Synchronization Module (MSM). To estimate audio-visual alignment, we encode the lip ROIs L and audio features A using dedicated 3D-CNN and 1D-CNN encoders. Both resulting feature maps are then processed by an adaptive pooling layer and subsequent projection head to generate the final visual (L<sup>′</sup>) and audio (A<sup>′</sup>) representations. We then apply multi-head cross-attention.

$$
\begin{array} { r l } & { L ^ { \prime } = f _ { \mathrm { 3 D } } ( { \bf L } ) \in \mathbb { R } ^ { T \times d _ { v } } , A ^ { \prime } = f _ { \mathrm { 1 D } } ( { \bf A } ) \in \mathbb { R } ^ { T \times d _ { a } } } \\ & { \tilde { L } = \mathrm { A t t n } ( Q = L ^ { \prime } , K = A ^ { \prime } , V = A ^ { \prime } ) . } \end{array}\tag{9}
$$

Finally, the attended features $\tilde { L }$ are fused with the original visual features $L ^ { \prime } ,$ , aggregated by temporal averaging, and projected by $g _ { \mathrm { a v } }$ to produce the synchronization feature:

$$
z _ { \mathrm { a v } } = g _ { \mathrm { a v } } \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } [ L _ { t } ^ { \prime } + \tilde { L } _ { t } ] \right) .\tag{10}
$$

Attribution Classifier. We concatenate the temporal dynamic feature $z _ { \mathrm { t e m p } }$ and the synchronization feature $z _ { \mathrm { a v } }$ to form a comprehensive fingerprint ${ \mathit { z } } _ { \mathrm { f i n a l } }$ . This classifier is trained exclusively on forged samples using their groundtruth model label $\hat { m } \in \mathcal { M }$ . The objective is a standard cross-entropy loss $\mathcal { L } _ { \mathrm { a t t } } \mathrm { : }$

$$
\mathcal { L } _ { \mathrm { a t t } } = - \mathbb { E } \left[ \log \left( f _ { \mathrm { a t t } } ( z _ { \mathrm { f i n a l } } ) \right) _ { \hat { m } } \right] .\tag{11}
$$

## 4.4. Inference

As shown in Fig. 5, all components form a unified pipeline during the inference stage. The Detection Classifier $( f _ { \mathrm { d e t } } )$ predicts the forgery label $\hat { y } \in \{ \mathrm { R e a l } , \mathrm { F a k e } \}$ . Concurrently, features from the MSM and TDM modules are concatenated and passed to the Attribution Classifier $( f _ { \mathrm { a t t } } )$ to predict the source model label $\hat { m } \in \mathcal { M }$ . The pipeline outputs a joint prediction (yˆ, mˆ ) for simultaneous detection and attribution.

## 5. Experiment

## 5.1. Experimental Setup

Dataset and Evaluation Protocols. We utilize our proposed LipSync-A as the primary dataset for training and in-domain detection evaluation, leveraging its diverse generator architectures to enable fine-grained attribution evaluation. To rigorously assess generalization, we adopt a zeroshot evaluation protocol on AVLips and TalkHeadBench to test performance across unseen data distributions, while further verifying adaptability against novel SOTA synthesis algorithms excluded from training. Additionally, we examine cross-manipulation transferability using the Deep-Fake dataset Celeb-DF (Li et al., 2020), whereas robustness against diverse visual and audio perturbations is assessed on AVLips, adhering to standard forensic protocols.

Metrics. Following standard practices (Xiong et al., 2025; Liang et al., 2024), we report video-level Accuracy (ACC) and Area Under the ROC Curve (AUC) for comparison. We also include F1-score, Average Precision (AP), False Positive Rate (FPR), and False Negative Rate (FNR) for a comprehensive evaluation.

Baselines and Implementation Details. We compare against 15 SOTA deepfake detectors, including 8 uni-modal approaches that cover lip-reading, temporal coherence, and frequency-domain cues, and 7 multi-modal approaches that exploit audio-visual asynchrony or alignment. For LipDA, we extract facial landmarks with MediaPipe, set the slidingwindow length to T = 5 frames, and crop lip ROIs at 96×96. Audio is encoded as MFCC features aligned to the visual stream. Both stages are optimized with Adam at an initial learning rate of $1 \times 1 0 ^ { - 5 }$ , batch size 32, for 20 epochs. Further hyperparameters and training composition are detailed in Appendix C.

## 5.2. Comparison to Existing Methods

As shown in Table 2, our method LipDA significantly outperforms existing baselines on three LipSync datasets in both video-only and audio-visual settings. In the audio-visual setting, it surpasses the previous state-of-the-art SpeechForensics by +7.46% in ACC and +7.97% in AUC. We observe that many unimodal detectors perform near random guessing, indicating their inability to handle subtle LipSync forgeries. LipForensics and RealForensics utilizing lip-shape cues achieve better results, but they are still substantially outperformed by multimodal methods. While incorporating audio generally improves our model’s performance, we note a slight AUC decrease on TalkHeadBench. We hypothesize that this is due to the high variability introduced by the real-scene audio in that specific dataset. Table 3 presents the performance of our method compared with advanced multimodal detectors on the attribution task across five generator families. Our model demonstrates highly balanced recognition, achieving an average F1-score improvement of over 12 percentage points compared to the second-best method, TALL. While SpeechForensics and AVAD exhibit severe instability with extremely low F1-scores on the GAN and Transformer categories, our model maintains robust performance even on the most challenging Transformer-based forgeries.

Table 2. Binary forgery detection performance comparison. We report Accuracy (ACC, %) and AUC (%) on three public benchmarks against SOTA methods. ’V’ denotes our visual-only model and ’A-V’ denotes our audio-visual variant, which additionally incorporates the MSM score for detection. Bold indicates the best results, while the second-ranking one is underscored.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Modality</td><td colspan="2">In-domain</td><td colspan="5">Cross-domain</td><td rowspan="2">Average</td></tr><tr><td colspan="2">LipSync-A ACC↑</td><td colspan="2">AVLips</td><td colspan="2">TalkHeadBench</td></tr><tr><td></td><td></td><td></td><td>AUC↑</td><td>ACC↑</td><td>AUC↑</td><td>ACC↑</td><td>AUC↑</td><td>ACC↑</td><td>AUC↑</td></tr><tr><td>FTCN (Zheng et al., 2021)</td><td>V</td><td>78.98</td><td>87.78</td><td>65.67</td><td>71.55</td><td>68.73</td><td>75.46</td><td>71.13</td><td>78.26</td></tr><tr><td>CADDM (Dong et al., 2023)</td><td>V</td><td>47.15</td><td>63.14</td><td>45.90</td><td>55.92</td><td>62.53</td><td>63.69</td><td>51.86</td><td>60.92</td></tr><tr><td>LipForensics (Haliassos et al., 2021)</td><td>V</td><td>80.92</td><td>89.85</td><td>74.15</td><td>81.97</td><td>84.52</td><td>90.63</td><td>79.86</td><td>87.48</td></tr><tr><td>RealForensics (Haliassos et al., 2022)</td><td>V</td><td>72.44</td><td>67.03</td><td>80.47</td><td>89.25</td><td>71.03</td><td>79.01</td><td>74.65</td><td>78.43</td></tr><tr><td>UnivFD (Ojha et al., 2023)</td><td>V V</td><td>52.30</td><td>69.27</td><td>50.00</td><td>47.34</td><td>56.90</td><td>76.25</td><td>53.07</td><td>64.29</td></tr><tr><td>FreqNet (Tan et al., 2024a)</td><td>V</td><td>48.80 45.6</td><td>9.40 14.40</td><td>45.30 44.40</td><td>52.40</td><td>42.10</td><td>73.60</td><td>45.40</td><td>45.13</td></tr><tr><td>NPR (Tan et al., 2024b) TALL (Xu et al., 2024b)</td><td>V</td><td>75.24</td><td>96.92</td><td>61.36</td><td>44.00 80.69</td><td>54.10 58.88</td><td>74.10 71.49</td><td>48.03</td><td>44.17</td></tr><tr><td></td><td>A-V</td><td>62.75</td><td>87.09</td><td>50.16</td><td>70.01</td><td>50.29</td><td>64.11</td><td>65.16</td><td>83.03</td></tr><tr><td>AltFreezing (Wang et al., 2023b)</td><td>A-V</td><td>48.72</td><td>24.81</td><td>71.00</td><td>73.18</td><td>58.20</td><td>58.66</td><td>54.40</td><td>73.74</td></tr><tr><td>AVAD (Feng et al., 2023) LipFD (Liu et al., 2024)</td><td>A-V</td><td>50.45</td><td>44.09</td><td>95.27</td><td>96.08</td><td>43.82</td><td>44.75</td><td>59.31</td><td>52.22</td></tr><tr><td>SpeechForensics (Liang et al., 2024)</td><td>A-V</td><td>91.38</td><td>94.82</td><td>98.50</td><td>99.15</td><td>76.52</td><td>78.87</td><td>63.18</td><td>61.64</td></tr><tr><td>FGMDC (Yin et al., 2024)</td><td>A-V</td><td>63.09</td><td>60.00</td><td>84.00</td><td>92.34</td><td>89.78</td><td>90.80</td><td>88.80</td><td>90.95</td></tr><tr><td>DFD-FCG (Han et al., 2025)</td><td>A-V</td><td>83.26</td><td>88.34</td><td>66.33</td><td>68.27</td><td>64.75</td><td>66.55</td><td>78.96 71.45</td><td>81.05</td></tr><tr><td>AVH-align (Smeu et al., 2025)</td><td>A-V</td><td>94.17</td><td>98.16</td><td>75.74</td><td>88.38</td><td>45.80</td><td>34.95</td><td>71.90</td><td>74.39 73.83</td></tr><tr><td></td><td>V</td><td></td><td>97.83</td><td>97.02</td><td>99.59</td><td>93.05</td><td>98.65</td><td></td><td></td></tr><tr><td>LipDA(Ours) LipDA(Ours)</td><td>A-V</td><td>93.47 95.96</td><td>99.42</td><td>98.34</td><td>99.82</td><td>94.48</td><td>97.50</td><td>94.51 96.26</td><td>98.69 98.91</td></tr></table>

Table 3. Attribution performance comparison for classifying forgeries into five generator families. Metrics are Accuracy (ACC, %) and F1-Score (F1, %). All listed methods are audio-visual. Bold and underscored denote the best and second-best results, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">Transformer-based</td><td colspan="2"></td><td colspan="2">GAN-based Diffusion-based</td><td colspan="2">VAE-based</td><td colspan="2">Statistical models</td><td colspan="2">Average</td></tr><tr><td>ACC ↑</td><td>F1 ↑</td><td>ACC ↑ F1 ↑ ACC ↑</td><td></td><td></td><td>F1↑</td><td>ACC ↑F1 ↑</td><td></td><td>ACC ↑</td><td>F1 ↑</td><td>ACC ↑ F1 ↑</td><td></td></tr><tr><td>AVAD (Feng et al., 2023)</td><td>77.2</td><td>9.9</td><td>83.0</td><td>56.7</td><td>70.0</td><td>27.4</td><td>68.8</td><td>40.2</td><td>71.9</td><td>33.0</td><td>74.2</td><td>33.4</td></tr><tr><td>TALL (Xu et al., 2024b)</td><td>89.0</td><td>59.3</td><td>92.0</td><td>83.0</td><td>97.8</td><td>96.7</td><td>96.0</td><td>96.6</td><td>86.7</td><td>72.0</td><td>92.3</td><td>81.5</td></tr><tr><td>AVH-align (Smeu et al., 2025)</td><td>71.8</td><td>41.0</td><td>96.5</td><td>69.2</td><td>99.9</td><td>98.5</td><td>96.9</td><td>90.9</td><td>99.9</td><td>88.9</td><td>93.0</td><td>77.7</td></tr><tr><td>SpeechForensics (Liang et al., 2024)</td><td>70.4</td><td>43.9</td><td>71.6</td><td>18.4</td><td>70.4</td><td>46.0</td><td>76.4</td><td>11.9</td><td>73.2</td><td>13.0</td><td>72.4</td><td>26.6</td></tr><tr><td>LipDA(Ours) A-V</td><td>93.8</td><td>87.6</td><td>94.9</td><td>86.7</td><td>99.9</td><td>99.9</td><td>99.9</td><td>99.9</td><td>98.9</td><td>95.2</td><td>97.5</td><td>93.9</td></tr></table>

## 5.3. Generalizability to Unseen Forgery

We evaluated the generalization capability of our method on forgeries generated by recent LipSync techniques, as well as on the traditional cross-manipulation Celeb-DF dataset. As Celeb-DF lacks audio, all methods were evaluated in a video-only setting for this test. As shown in Table 4, existing multimodal detectors suffer a severe performance collapse on the unseen LipSync samples. In contrast, our method achieves the highest ACC and F1 scores, significantly outperforming the strong SpeechForensics baseline. Furthermore, LipDA remains highly effective against traditional forgeries in Celeb-DF, demonstrating that physiological inconsistencies serve as reliable artifacts beyond the LipSync domain. As shown in Table 5, despite being trained exclusively on LipSync forgeries, LipDA successfully generalizes to identity-swap manipulations, achieving an ACC of 92.76% and an FNR of only 1.54%.

Table 4. Cross-manipulation generalization on LipSync (A-V). Unseen LipSync generalization on Sonic (Ji et al., 2025), KDTalker (Yang et al., 2025b), and OmniSync (Peng et al., 2025).
<table><tr><td rowspan="2">Method</td><td colspan="2">Sonic</td><td colspan="2">KDTalker</td><td colspan="2">OmniSync</td></tr><tr><td></td><td></td><td></td><td></td><td>ACC ↑ F1 ↑ ACC ↑ F1 ↑ ACC ↑ F1 ↑</td><td></td></tr><tr><td>LipFD (Liu et al., 2024)</td><td>48.6</td><td>7.1</td><td>45.6</td><td>6.9</td><td>51.2</td><td>5.8</td></tr><tr><td>AVAD (Feng et al., 2023)</td><td>49.6</td><td>66.3</td><td>49.7</td><td>66.4</td><td>49.5</td><td>66.2</td></tr><tr><td>SpeechForensics (Liang et al., 2024)</td><td>80.8</td><td>80.4</td><td>83.9</td><td>81.0</td><td>90.0</td><td>90.2</td></tr><tr><td>DFD-FCG (Han et al., 2025)</td><td>66.4</td><td>61.9</td><td>72.2</td><td>69.4</td><td>52.1</td><td>14.7</td></tr><tr><td>AVH-align (Smeu et al., 2025)</td><td>70.0</td><td>56.9</td><td>71.9</td><td>64.9</td><td>72.9</td><td>9.5</td></tr><tr><td>AltFreezing (Wang et al., 2023b)</td><td>49.6</td><td>66.6</td><td>59.3</td><td>69.2</td><td>33.0</td><td>50.0</td></tr><tr><td>LipDA(Ours)</td><td>87.8</td><td>84.3</td><td>84.6</td><td>84.3</td><td>95.5</td><td>95.4</td></tr></table>

Table 5. Cross-manipulation generalization on DeepFake (V). Generalization performance on CelebDF (Li et al., 2020).
<table><tr><td rowspan="2">Method</td><td colspan="5">CelebDF</td></tr><tr><td>ACC ↑</td><td>AUC↑</td><td>AP↑</td><td>FPR↓</td><td>FNR↓</td></tr><tr><td>CADDM (Dong et al., 2023)</td><td>53.83</td><td>81.44</td><td>95.70</td><td>9.55</td><td>51.95</td></tr><tr><td>UnivFD (Ojha et al., 2023)</td><td>50.23</td><td>66.39</td><td>67.96</td><td>1.00</td><td>99.00</td></tr><tr><td>FreqNet (Tan et al., 2024a)</td><td>49.80</td><td>53.50</td><td>51.11</td><td>4.00</td><td>99.90</td></tr><tr><td>NPR (Tan et al., 2024b)</td><td>50.10</td><td>46.80</td><td>47.21</td><td>0.00</td><td>100.0</td></tr><tr><td>RealForensics (Haliassos et al., 2022)</td><td>54.05</td><td>67.08</td><td>68.81</td><td>81.23</td><td>49.31</td></tr><tr><td>LipForensics (Haliassos et al., 2021)</td><td>71.91</td><td>80.83</td><td>81.01</td><td>29.21</td><td>26.87</td></tr><tr><td>LipDA(Ours)</td><td>92.76</td><td>91.86</td><td>98.38</td><td>4.33</td><td>1.54</td></tr></table>

## 5.4. Robustness Evaluation

To evaluate the feasibility of our model for real-world deployment, we assess its robustness against diverse visual and audio perturbations. As shown in Fig. 6, the performance of baseline methods deteriorates significantly under distortions such as Gaussian blur, compression, and contrast adjustments. In contrast, LipDA exhibits remarkable stability against these texture-level corruptions. This resilience serves as strong evidence that our Stage I framework successfully captures the intrinsic physiological coupling between pose and lip dynamics, rather than overfitting to lowlevel artifacts. Furthermore, our method also demonstrates strong robustness to audio distortions. While sensitive to direct signal corruption, common operations like resampling and time shifts result in negligible performance drops. Full results are detailed in Appendix D (Table 10).

![](images/33a3e4f8ce4723d98da53661a12ea4a7bf6b7f1a61c2a0c4042ccbb3e4c69f94.jpg)  
Figure 6. Robustness against various unseen corruptions. See Appendix D for detailed analysis and intensity settings.

## 5.5. Ablation Study and Analysis

Influence of Sliding Window and Video Length. As shown in Fig. 7, the sliding window length T presents a clear trade-off. The window must be sufficiently long to capture meaningful temporal information, yet an excessively long window introduces redundant motion, which dilutes the critical inconsistency cues and degrades performance. Furthermore, detection accuracy improves with total video length, saturating at approximately 8 seconds under the Non-random sampling protocol.

![](images/e60e67746ece05caac46ded5f0cd0eb3462b29ed21905b8dab8a507045ca66d0.jpg)

![](images/890e7c81d436216a64ad24fcaa0f81522fbc63974e97ce47f80dcbf7efdd0710.jpg)  
Figure 7. Analysis of temporal parameters. Left: Detection accuracy as a function of sliding window length T. Right: Detection accuracy by video length, comparing Random and Non-random sampling strategies.

Effect of different components and backbone choices. We conduct ablations to validate our component and backbone choices in Table 6. Removing the alignment loss leads to a 17.3% drop, indicating that naive feature concatenation is insufficient and that contrastive alignment is necessary to capture lip–pose inconsistency between real and forged videos. Eliminating the lip component causes a performance collapse in detection, confirming that the mouth region provides the dominant forgery cues. For backbones, LSTM proves superior for pose features, while replacing ResNet-18 with CLIP-ViT reduces performance by 12.0%, likely due to a domain mismatch between large-scale pretraining and our small, constrained lip-crop domain. See Appendix F.4 for further attribution ablations.

Table 6. Ablation study of our framework’s components and backbone choices. We report results on AVLips, and ∆ denotes the ACC (%) drop from our full model.
<table><tr><td colspan="3">I. Component Removal</td><td colspan="3">II. Backbone Replacement</td></tr><tr><td>Setting</td><td>ACC (%)</td><td>∆ (%)</td><td>Setting</td><td>ACC (%)</td><td>∆(%)</td></tr><tr><td rowspan="4">Full model w/o Align w/o Pose</td><td>97.90</td><td></td><td>Pose(Pooling)</td><td>97.20</td><td>↓0.7</td></tr><tr><td>80.56</td><td>↓17.3</td><td>Pose(Flatten)</td><td>95.27</td><td>↓2.6</td></tr><tr><td>74.78</td><td>↓23.1</td><td>Lip(MobileNet)</td><td>96.23</td><td>↓1.7</td></tr><tr><td>53.50</td><td>↓44.4</td><td>Lip(CLIP-ViT)</td><td>85.73</td><td>↓12.0</td></tr></table>

Visualization evidence for the proposed components. We validate the Stage I framework by visualizing its learned feature space in Fig. 8. Lip and pose embeddings from unseen real samples are tightly intermingled, reflecting their inherent physiological coupling. Conversely, embeddings from fake samples form distinct, separable clusters, demonstrating that our model successfully learns to separate the inconsistent pose-lip features found in forgeries.

![](images/592c8b96ab4e28dbcb555ddc6daecf675105fedb37c58a46a15d08f8ab60bfd0.jpg)  
Figure 8. t-SNE visualization of Stage I lip and pose embeddings from unseen real (Left) and fake (Right) test samples.

## 6. Discussion

Our work shifts LipSync detection from relying on local artifacts or explicit audio-visual mismatches to global physiological coordination. An adversary can easily optimize a lip-audio synchronization loss, but modeling the highdimensional coordination of head motion with speech is substantially harder. Moreover, the robustness of our method to texture degradations confirms that it exploits these fundamental physiological constraints rather than brittle visual cues. Despite the promising performance, LipDA has limitations that suggest future directions: (i) Scope. The applicability of this framework to broader AI-generated image forgery domains remains unexplored (Li et al., 2026; Wang et al., 2026). (ii) Audio sensitivity. The Stage-II attribution module degrades under certain audio perturbations (Appendix D), indicating that practical applications should incorporate audio-quality awareness. (iii) Operational boundaries. Detection relies on visible lip dynamics and head motion, and degrades under stationary speakers, persistent downward gaze, or prolonged facial occlusion (Appendix F.4), motivating future work on complementary cues from upper-face or full-body dynamics.

## 7. Conclusion

We present a novel and robust forgery signal, the global physiological inconsistency between lip movements and head poses. We introduce LipDA, a two-stage framework that models this coordination to unify forgery detection and source attribution, which achieves superior performance, outperforming existing methods on both tasks. It also demonstrates strong generalization to unseen techniques and maintains high robustness against significant compression and blurring. Our research offers a more resilient defense mechanism, opening up a new path for the ongoing cat-and-mouse game of forgery generation and detection.

## Acknowledgements

This work was supported by National Natural Science Foundation of China (Grants 62372423, U2336206 and U2436601) and was also supported by the Fundamental Research Funds for the Central Universities WK2100250070.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

Bouaziz, S., Wang, Y., and Pauly, M. Online modeling for realtime facial animation. ACM Transactions on Graphics (ToG), 32(4):1–10, 2013.

Cai, Z., Stefanov, K., Dhall, A., and Hayat, M. Do you really mean that? content driven audio-visual deepfake dataset and multimodal method for temporal forgery localization. In 2022 International Conference on Digital Image Computing: Techniques and Applications (DICTA), pp. 1–10. IEEE, 2022.

Cai, Z., Ghosh, S., Adatia, A. P., Hayat, M., Dhall, A., Gedeon, T., and Stefanov, K. Av-deepfake1m: A largescale llm-driven audio-visual deepfake dataset. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pp. 7414–7423, 2024.

Chen, L., Cui, G., Liu, C., Li, Z., Kou, Z., Xu, Y., and Xu, C. Talking-head generation with rhythmic head motion. In European conference on computer vision, pp. 35–51. Springer, 2020.

Chung, J. S., Senior, A., Vinyals, O., and Zisserman, A. Lip reading sentences in the wild. In IEEE Conference on Computer Vision and Pattern Recognition, 2017.

Dong, S., Wang, J., Ji, R., Liang, J., Fan, H., and Ge, Z. Implicit identity leakage: The stumbling block to improving deepfake detection generalization. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 3994–4004, 2023.

Feng, C., Chen, Z., and Owens, A. Self-supervised video forensics by audio-visual anomaly detection. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10491–10503, 2023.

Gan, Q., Yang, R., Zhu, J., Xue, S., and Hoi, S. Omniavatar: Efficient audio-driven avatar video generation with adaptive body animation. arXiv preprint arXiv:2506.18866, 2025.

Gan, Y., Yang, Z., Yue, X., Sun, L., and Yang, Y. Efficient emotional adaptation for audio-driven talking-head generation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2023.

Gick, B., Mayer, C., Chiu, C., Widing, E., Roewer-Despres,´ F., Fels, S., and Stavness, I. Quantal biomechanical effects in speech postures of the lips. Journal of neurophysiology, 124(3):833–843, 2020.

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Haliassos, A., Vougioukas, K., Petridis, S., and Pantic, M. Lips don’t lie: A generalisable and robust approach to face forgery detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 5039–5049, 2021.

Haliassos, A., Mira, R., Petridis, S., and Pantic, M. Leveraging real talking faces via self-supervision for robust forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14950–14962, 2022.

Han, Y.-H., Huang, T.-M., Hua, K.-L., and Chen, J.-C. Towards more general video-based deepfake detection through facial component guided adaptation for foundation model. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 22995–23005, 2025.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Hou, Y., Fu, H., Chen, C., Li, Z., Zhang, H., and Zhao, J. Polyglotfake: A novel multilingual and multimodal deepfake dataset. In International Conference on Pattern Recognition, pp. 180–193. Springer, 2024.

James, E. C., Bongiovanni, I., Glencross, M., and Brace, L. A systematic literature review on how cybercriminals are utilizing deepfakes to conduct cyberattacks. Digital Threats: Research and Practice, 2026.

Ji, X., Hu, X., Xu, Z., Zhu, J., Lin, C., He, Q., Zhang, J., Luo, D., Chen, Y., Lin, Q., et al. Sonic: Shifting focus to global audio perception in portrait animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 193–203, 2025.

Li, C., Zhang, C., Xu, W., Lin, J., Xie, J., Feng, W., Peng, B., Chen, C., and Xing, W. Latentsync: Taming audioconditioned latent diffusion models for lip sync with syncnet supervision. arXiv preprint arXiv:2412.09262, 2024.

Li, Y., Yang, X., Sun, P., Qi, H., and Lyu, S. Celeb-df: A large-scale challenging dataset for deepfake forensics. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3207–3216, 2020.

Li, Z., Shang, X., Fu, Z., Guo, S., Zhang, W., Yu, N., and Chen, K. Dualcodedetect: Zero-shot llm-generated code detection via dual-channel perturbation. Proceedings of the ACM on Software Engineering, 3(FSE):3604–3626, 2026.

Liang, Y., Yu, M., Li, G., Jiang, J., Li, B., Yu, F., Zhang, N., Meng, X., and Huang, W. Speechforensics: Audio-visual speech representation learning for face forgery detection. Advances in Neural Information Processing Systems, 37: 86124–86144, 2024.

Liu, W., She, T., Liu, J., Li, B., Yao, D., and Wang, R. Lips are lying: Spotting the temporal inconsistency between audio and visual in lip-syncing deepfakes. Advances in Neural Information Processing Systems, 37:91131– 91155, 2024.

Liu, Y., Shi, H., Shen, H., Si, Y., Wang, X., and Mei, T. A new dataset and boundary-attention semantic segmentation for face parsing. In AAAI, pp. 11637–11644, 2020.

Ma, Y., Zhang, S., Wang, J., Wang, X., Zhang, Y., and Deng, Z. Dreamtalk: When emotional talking head generation meets diffusion probabilistic models. arXiv preprint arXiv:2312.09767, 2023.

Meng, M., Zhao, Y., Zhang, B., Zhu, Y., Shi, W., Wen, M., and Fan, Z. A comprehensive taxonomy and analysis of talking head synthesis: Techniques for portrait generation, driving mechanisms, and editing. arXiv preprint arXiv:2406.10553, 2024.

Ojha, U., Li, Y., and Lee, Y. J. Towards universal fake image detectors that generalize across generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24480–24489, 2023.

Peng, Z., Liu, J., Zhang, H., Liu, X., Tang, S., Wan, P., Zhang, D., Liu, H., and He, J. Omnisync: Towards universal lip synchronization via diffusion transformers. arXiv preprint arXiv:2505.21448, 2025.

Prajwal, K., Mukhopadhyay, R., Namboodiri, V. P., and Jawahar, C. A lip sync expert is all you need for speech to lip generation in the wild. In Proceedings of the 28th ACM international conference on multimedia, pp. 484– 492, 2020.

Siarohin, A., Lathuiliere, S., Tulyakov, S., Ricci, E., and\` Sebe, N. First order motion model for image animation. Advances in neural information processing systems, 32, 2019.

Smeu, S., Boldisor, D.-A., Oneata, D., and Oneata, E. Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18815–18825, 2025.

Song, H. J. and Itti, L. Riemannian-geometric fingerprints of generative models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 11425– 11435, 2025.

Song, H. J., Khayatkhoei, M., and AbdAlmageed, W. Manifpt: Defining and analyzing fingerprints of generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10791– 10801, 2024.

Suwajanakorn, S., Seitz, S. M., and Kemelmacher-Shlizerman, I. Synthesizing obama: learning lip sync from audio. ACM Transactions on Graphics (ToG), 36 (4):1–13, 2017.

Tan, C., Zhao, Y., Wei, S., Gu, G., Liu, P., and Wei, Y. Frequency-aware deepfake detection: Improving generalizability through frequency space domain learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pp. 5052–5060, 2024a.

Tan, C., Zhao, Y., Wei, S., Gu, G., Liu, P., and Wei, Y. Rethinking the up-sampling operations in cnn-based generative network for generalizable deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28130–28139, 2024b.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, C., Tian, K., Zhang, J., Guan, Y., Luo, F., Shen, F., Jiang, Z., Gu, Q., Han, X., and Yang, W. V-express: Conditional dropout for progressive training of portrait video generation. arXiv preprint arXiv:2406.02511, 2024.

Wang, C., Yang, Z., Wang, Y., Zhang, W., and Chen, K. Aedr: Training-free ai-generated image attribution via autoencoder double-reconstruction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 9675–9683, 2026.

Wang, J., Qian, X., Zhang, M., Tan, R. T., and Li, H. Seeing what you said: Talking face generation guided by a lip reading expert. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14653–14662, 2023a.

Wang, T.-C., Mallya, A., and Liu, M.-Y. One-shot free-view neural talking-head synthesis for video conferencing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10039–10049, 2021.

Wang, Y., Yang, D., Bremond, F., and Dantcheva, A. Latent image animator: Learning to animate images via latent space navigation. arXiv preprint arXiv:2203.09043, 2022.

Wang, Z., Bao, J., Zhou, W., Wang, W., and Li, H. Altfreezing for more general video face forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4129–4138, 2023b.

Wiles, O., Koepke, A., and Zisserman, A. X2face: A network for controlling face generation using images, audio, and pose codes. In Proceedings ofthe European conference on computer vision (ECCV), pp. 670–686, 2018.

Wu, W., Zhou, W., Zhang, W., Fang, H., and Yu, N. Capturing the lighting inconsistency for deepfake detection. In International Conference on Artificial Intelligence and Security, pp. 637–647. Springer, 2022.

Xie, L., Wang, X., Zhang, H., Dong, C., and Shan, Y. Vfhq: A high-quality dataset and benchmark for video face super-resolution. In The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022.

Xiong, X., Patel, P., Fan, Q., Wadhwa, A., Selvam, S., Guo, X., Qi, L., Liu, X., and Sengupta, R. Talkingheadbench: A multi-modal benchmark & analysis of talking-head deepfake detection. arXiv preprint arXiv:2505.24866, 2025.

Xu, C., Zhang, J., Hua, M., He, Q., Yi, Z., and Liu, Y. Region-aware face swapping. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7632–7641, 2022.

Xu, M., Li, H., Su, Q., Shang, H., Zhang, L., Liu, C., Wang, J., Yao, Y., and Zhu, S. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024a.

Xu, Y., Liang, J., Sheng, L., and Zhang, X.-Y. Learning spatiotemporal inconsistency via thumbnail layout for face deepfake detection. International Journal ofComputer Vision, 132(12):5663–5680, 2024b.

Yamagishi, J., Veaux, C., and MacDonald, K. Cstr vctk corpus: English multi-speaker corpus for cstr voice cloning toolkit (version 0.92). The Rainbow Passage which the speakers read out can befound in the International Dialects ofEnglish Archive, 2019.

Yan, Z., Zhang, Y., Yuan, X., Lyu, S., and Wu, B. Deepfakebench: A comprehensive benchmark of deepfake detection. Advances in Neural Information Processing Systems, 36:4534–4565, 2023.

Yang, C., Yao, K., Yan, Y., Jiang, C., Zhao, W., Sun, J., Cheng, G., Zhang, Y., Dong, B., and Huang, K. Unlock pose diversity: Accurate and efficient implicit keypointbased spatiotemporal diffusion for audio-driven talking portrait, 2025a.

Yang, C., Yao, K., Yan, Y., Jiang, C., Zhao, W., Sun, J., Cheng, G., Zhang, Y., Dong, B., and Huang, K. Unlock pose diversity: Accurate and efficient implicit keypointbased spatiotemporal diffusion for audio-driven talking portrait. arXiv preprint arXiv:2503.12963, 2025b.

Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., and Yang, M.-H. Diffusion models: A comprehensive survey of methods and applications. ACM computing surveys, 56(4):1–39, 2023.

Yang, S., Kong, Z., Gao, F., Cheng, M., Liu, X., Zhang, Y., Kang, Z., Luo, W., Cai, X., He, R., et al. Infinitetalk: Audio-driven video generation for sparse-frame video dubbing. arXiv preprint arXiv:2508.14033, 2025c.

Yin, Q., Lu, W., Cao, X., Luo, X., Zhou, Y., and Huang, J. Fine-grained multimodal deepfake classification via heterogeneous graphs. International Journal ofComputer Vision, 132(11):5255–5269, 2024.

Zhang, W., Cun, X., Wang, X., Zhang, Y., Shen, X., Guo, Y., Shan, Y., and Wang, F. Sadtalker: Learning realistic 3d motion coefficients for stylized audio-driven single image talking face animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8652–8661, 2023a.

Zhang, Z., Li, L., Ding, Y., and Fan, C. Flow-guided oneshot talking face generation with a high-resolution audiovisual dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3661–3670, 2021.

Zhang, Z., Hu, Z., Deng, W., Fan, C., Lv, T., and Ding, Y. Dinet: Deformation inpainting network for realistic face visually dubbing on high resolution video. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 3543–3551, 2023b.

Zhao, J. and Zhang, H. Thin-plate spline motion model for image animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3657–3666, 2022.

Zheng, Y., Bao, J., Chen, D., Zeng, M., and Wen, F. Exploring temporal coherence for more general video face forgery detection. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 15044– 15054, 2021.

Zhong, W., Fang, C., Cai, Y., Wei, P., Zhao, G., Lin, L., and Li, G. Identity-preserving talking face generation with landmark and appearance priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729–9738, 2023.

Zhou, Y., Han, X., Shechtman, E., Echevarria, J., Kalogerakis, E., and Li, D. Makelttalk: speaker-aware talkinghead animation. ACM Transactions On Graphics (TOG), 39(6):1–15, 2020.

Zhu, H., Wu, W., Zhu, W., Jiang, L., Tang, S., Zhang, L., Liu, Z., and Loy, C. C. Celebv-hq: A large-scale video facial attributes dataset. In European conference on computer vision, pp. 650–667. Springer, 2022.

## A. Additional Dataset Details

To address the critical scarcity of high-quality, diverse, and attribution-focused datasets for lip-synchronization (LipSync), we introduce LipSync-A, a comprehensive evaluation dataset designed to systematically cover dominant driving paradigms and identity input sources. It accommodates both audio- and video-driven generation paradigms, accepting inputs ranging from static images to dynamic source videos, thereby establishing a rigorous platform for attribution-oriented LipSync evaluation.

## A.1. Source Data Curation and Pre-processing

For identity inputs, we sourced data from LaPa (Liu et al., 2020), a large-scale in-the-wild facial image dataset covering rich expressions, poses, and occlusions; and HDTF (Zhang et al., 2021), a high-resolution talking head video dataset, mostly featuring English speakers in frontal views. For driving signals, we utilized VFHQ (Xie et al., 2022), a highdefinition interview video dataset from 20 countries, covering natural blinking, speech, and large-scale head movements; and VCTK (Yamagishi et al., 2019), which provides 44.1 kHz clean read-speech sentences from 110 English speakers. We implemented a rigorous quality control pipeline, including automated filtering of low-quality samples, such as excessive blur, extreme poses, and uniform denoising of all audio tracks, as shown in Fig. 9.

## A.2. Forgery Generation Pipeline

A core strength of LipSync-A lies in its diverse and representative collection of generators. We incorporate both open-source state-of-the-art methods from academia and commercial APIs, encompassing a broad spectrum of technical paradigms. Following the taxonomy proposed by Meng et al. (2024), these include statistical models, hybrid transformations, and approaches based on keypoints, GANs, VAEs, diffusion models, and Transformers. In total, 15 distinct generation techniques were selected (refer to Table 7), yielding approximately 16,000 video samples. To bolster real-world robustness, we specifically generated approximately 270 samples using commercial APIs. For experimental evaluation, we employ a subset of this dataset partitioned with strict constraints to ensure disjoint identities and driving signals. This split results in 3,926 training, 841 validation, and 842 testing samples, guaranteeing a balanced representation across diverse technical methodologies and input conditions.

Table 7. LipSync-A Generator Details. Category refers to the core technical architecture. Paradigm indicates the driving signal. Origina Training data lists the datasets used to train the original generator models.
<table><tr><td>Generator</td><td>Category</td><td>Paradigm</td><td>Training Data</td></tr><tr><td>X2Face (Wiles et al., 2018)</td><td>Geometric</td><td>Video-driven</td><td>VoxCeleb</td></tr><tr><td>TPSM (Zhao &amp; Zhang, 2022)</td><td>Landmark</td><td>Video-driven</td><td>VoxCeleb</td></tr><tr><td>LIA (Wang et al., 2022)</td><td>Latent Space</td><td>Video-driven</td><td>VoxCeleb</td></tr><tr><td>FaceVid (Wang et al., 2021)</td><td>Landmark</td><td>Video-driven</td><td>VoxCeleb</td></tr><tr><td>DiNet (Zhang et al., 2023b)</td><td>CNN</td><td>Audio-driven</td><td>HDTF, MEAD</td></tr><tr><td>MakeItTalk (Zhou et al., 2020)</td><td>GAN</td><td>Audio-driven</td><td>VoxCeleb, ObamaSet</td></tr><tr><td>Wav2Lip (Prajwal et al., 2020)</td><td>GAN</td><td>Audio-driven</td><td>LRW, LRS2</td></tr><tr><td>TalkLip (Wang et al., 2023a)</td><td>GAN</td><td>Audio-driven</td><td>LRS</td></tr><tr><td>SadTaÎker (Xu et al., 2024b)</td><td>VAE</td><td>Audio-driven</td><td>VoxCeleb</td></tr><tr><td>IP_LAP (Zhong et al., 2023)</td><td></td><td>Transformer Audio-driven</td><td>LRS2</td></tr><tr><td>DreamTalk (Ma et al., 2023)</td><td>Diffusion</td><td>Audio-driven</td><td>MEAD, HDTF</td></tr><tr><td>Sonic (Ji et al., 2025)</td><td>Diffusion</td><td>Audio-driven</td><td>VFHQ, CelebV-Text</td></tr><tr><td>KDtalker (Yang et al., 2025a)</td><td>Diffusion</td><td>Audio-driven</td><td>HDTF</td></tr><tr><td>OmniSync (Peng et al., 2025)</td><td>Diffusion</td><td>Audio-driven</td><td>Web-Collected</td></tr><tr><td>InfiniteTalk (Yang et al., 2025c)</td><td>Diffusion</td><td>Audio-driven</td><td>Internal</td></tr></table>

## B. Additional Discussion on Motivation

To address the theoretical underpinnings of our framework, we provide a deeper analysis of the fundamental constraints in LipSync generation and the discriminative basis for our attribution taxonomy.

![](images/55cf28fcd9f0b0c0c4aaf2eb47c0b3f054127ccec32db2e760b7fa225916093e.jpg)  
Figure 9. The generation pipeline for our LipSync-A dataset.

## B.1. Detection Motivation

One might question why the inconsistency between lip motion and head pose persists as a detectable signal. We argue that this is due to the inherent difficulty of simultaneously manipulating both modalities while maintaining physiological consistency.

The Theoretical Gap. In natural human speech, lip articulation (L) and head pose (P) are biologically coupled, driven by a unified neuro-cognitive system conditioned on the audio (A) and speaker identity (I). Theoretically, a perfect generator seeks to approximate the true joint distribution:

$$
P ( L , P | A , I )
$$

However, modeling this high-dimensional joint distribution presents a significant optimization challenge due to the discrep ancy in mapping properties. The mapping {A, I} → L is largely deterministic and strictly constrained by phoneme-to-viseme consistency for semantic intelligibility. In contrast, the mapping $\{ A , I \} \to P$ is high-dimensional and one-to-many, as head motion is not governed by explicit phonetic content.

To reduce optimization difficulty and avoid mode collapse, current SOTA methods predominantly decompose the problem into two independent sub-problems:

$$
P ( L , P \mid A , I ) \approx P ( L \mid A , I ) \cdot P ( P \mid A , I )
$$

By treating these as independent variables, generators inherently lose the correlation information between the specific phoneme articulation and the subtle head micro-movements. While lip movements are synchronized with audio, head motion is generated via independent probabilistic sampling or reference-driven conditioning, the correlation constraints between the two modalities are inherently absent. Our LipDA framework is explicitly designed to exploit this missing conditional dependence in the generative model.

## B.2. Attribution Motivation

To the best of our knowledge, we propose the first attribution task that taxonomizes audio-driven LipSync forgeries into five families, establishing a categorization aligned with mainstream LipSync paradigms while accounting for underlying generative mechanisms.

• GAN-based (e.g., Wav2Lip, Talklip): These methods commonly adopt adversarial optimization to minimize discriminator losses, whether operating in landmark or pixel space. As a result of this optimization process, the generated lip-sync videos manifest specific spectral anomalies, particularly in the high-frequency domain.

• Diffusion-based (e.g., Sonic, DreamTalk): Relies on Iterative Denoising. The motion trajectories often exhibit characteristic stochasticity and smoothness derived from the Gaussian sampling process, distinct from the sharp discontinuities of GANs.

• Transformer-based (e.g., IP-LAP): Transformer-based methods treat LipSync as a Sequence-to-Sequence translation task, often leading to specific quantization artifacts or step-wise motion patterns.

• VAE-based (e.g., SadTalker): Relies on Latent Space Distribution. The motion is constrained by the KL-divergence loss, typically resulting in overly smooth, mean-reverting motion trajectories lacking high-frequency micro-expressions.

• Statistical Models (e.g., DiNet): Relies on Geometric Warping. These methods do not synthesize new texture but warp existing pixels, leaving traces of stretching and non-linear distortion without generative noise.

## B.3. Inter-Class Signal Motivation

Video-Driven Paradigm The intrinsic coupling between a speaker’s unique physiological structure and their kinematic patterns renders identity-motion decoupling an inherently ill-posed problem. When the motion field of a driver is forcibly decoded onto a distinct source identity, it inevitably violates the source’s underlying conditional distribution, $P _ { s r c }$ (Motion|Identity). This incompatibility forces the generator to synthesize frames that fall outside the source’s natural manifold, leading to a distributional shift. Consequently, the model struggles to maintain spatial-temporal consistency under these foreign constraints, manifesting as localized temporal artifacts and incoherent micro-expressions that serve as detectable cues for video-driven forgeries.

## Audio-Driven Paradigm

• 1) Reconstruction-based methods typically treat LipSync as a localized inpainting task by masking the lower facial region and reconstructing it to maximize the semantic alignment between audio features and synthesized lip patches. The fundamental cause of inconsistency in these methods lies in the asynchrony of data sources. While the lip dynamics L are newly generated from the driving audio A, the head pose H is directly inherited from a static reference frame, resulting in a distinct temporal asynchrony where the mouth exhibits speech-driven activity while the head remains rigidly static or executes repetitive movements unrelated to the verbal emphasis

• 2) Diffusion-based methods leverage denoising diffusion probabilistic models to achieve high-fidelity textures and smoother transitions via iterative denoising from Gaussian noise. However, the stochastic nature of the denoising process introduces high-frequency randomness that tends to wash out the fine-grained synchronization signals between the two modalities. To mitigate jitter, current engineering practices often rigidly anchor the generated head pose to the original poses of the source video frames.

• 3) Independent Learning methods explicitly decompose the generation task into separate streams for lip and pose features conditioned on the same audio input. This design fails to capture biological coupling because the mapping properties are heterogeneous. The lip mapping is largely deterministic, while the head pose mapping is probabilistic and one-to-many. Training these modalities independently without a shared joint constraint cuts the gradient flow necessary to enforce phase-locking between phoneme articulation and head micro-movements. Consequently, this results in a physiological desynchronization, where the generated lip and pose sequences appear individually plausible but lack the intrinsic conditional dependence found in authentic human speech.

## C. Additional Experimental Setup

## C.1. Evaluation Dataset Selection

We carefully selected three LipSync datasets as follows:

• AVLips (Liu et al., 2024) is an audio-visual forgery dataset. It uses the LRS2(Chung et al., 2017) dataset as the audio driving signal and generates forged videos using the Wav2Lip (Prajwal et al., 2020) and MakeITalk (Zhou et al., 2020) models. The dataset comprises 3,396 real and 4,206 forged videos, which we split into training, validation, and test sets following a 7:1.5:1.5 ratio.

• TalkHeadBench (Xiong et al., 2025) contains forged videos generated by advanced diffusion models, such as Hallo (Xu et al., 2024a). Since the forged videos in this dataset lack corresponding real videos and some also lack audio, we select 476 real videos from its source, CelebV-HQ (Zhu et al., 2022), and pair them with 448 forged videos that include audio to construct our test set.

• Celeb-DF (Li et al., 2020) is a well-known standard DeepFake dataset focusing on identity-swapping forgeries. Its real videos are sourced from celebrity interviews on YouTube. We follow its standard protocol and use its test set of 980 videos for evaluation.

## C.2. Rationale for Baseline Selection

To ensure a comprehensive and fair comparison, we select several SOTA DeepFake detectors as baselines. We assess 8 baselines that rely solely on visual information, covering the following predominant approaches:

• Lip-based: Methods focusing on lip movement anomalies, e.g.,LipForensics (Haliassos et al., 2021), which leverages lip-reading as a proxy task to extract forgery-sensitive features.

• Artifact & Temporal Inconsistency: Methods that capture visual artifacts or temporal discontinuities. This includes FTCN (Zheng et al., 2021) (targeting temporal micro-incoherence), RealForensics (Haliassos et al., 2022) (using temporal self-consistency as an anomaly signal), and TALL (Xu et al., 2024b) (a tamper-aware approach that explicitly localizes manipulated regions).

• Frequency-Domain: Methods assuming forgeries leave detectable traces in the frequency spectrum. This includes CADDM (Dong et al., 2023) (exploiting complementary cues in color-frequency domains), FreqNet (Tan et al., 2024a) (detecting global high-frequency peaks), and NPR (Tan et al., 2024b) (amplifying noise anomalies via Noise Power Residual maps).

We also evaluate 7 multimodal baselines that exploit cross-modal information:

• Discrepancy & Asynchrony: Methods that detect inconsistencies or asynchrony between audio and visual streams. This includes AVAD (Feng et al., 2023) (explicitly detecting A-V asynchrony) and AltFreezing (Wang et al., 2023b) (using an alternating training strategy to enforce A-V synchronization).

• Alignment-based: Methods that explicitly align A-V features to spot forgeries. This includes AVH-align (Smeu et al., 2025) (framing LipSync as a hierarchical cross-modal contrastive learning task), LipFD (Liu et al., 2024) (detecting alignment in faked lip regions) and FGMDC (Yin et al., 2024) (leveraging fine-grained modal-complementary cues from the mouth, teeth, and audio pitch).

• Other Multimodal Approaches: This group includes SpeechForensics (Liang et al., 2024) (detecting acoustic artifacts from TTS/VC in the audio stream), and DFD-FCG (Han et al., 2025) (a frequency-cognition guided A-V framework).

## C.3. Baseline Adjustments for Attribution Task

Our framework includes a model attribution stage, which requires a multi-class classifier to identify the forgery source. We found that most visual-only detectors (e.g. CADDM (Dong et al., 2023), FreqNet (Tan et al., 2024a) ) are ill-suited for this task. Their detection often relies on low-level artifacts that lack sufficient discriminative power between different generative models. Consequently, we selected baselines more amenable to multi-class attribution. This includes TALL (Xu et al., 2024b) and several A-V methods (AVAD (Feng et al., 2023), SpeechForensics (Liang et al., 2024), and AVH-align (Smeu et al., 2025)). For these selected baselines, we adapted their output layers (originally 4-way) to a 5-way classification task (5 distinct forgery source classes) and trained them using a standard cross-entropy loss.

## C.4. Evaluation Protocol of Main Results

Training and test composition. LipDA is trained on the LipSync-A training split, which comprises 1,614 authentic videos and 2,312 forged videos produced by 10 generators spanning seven architectural paradigms: Hybrid Transformations (X2Face, 89), Keypoint-Driven (TPSM/LIA/FaceVid, 1,245), Statistical Models (DiNet, 134), GAN-based (MakeItTalk/Wav2Lip, 297), VAE-based (SadTalker, 206), Transformer-based (IP-LAP, 210), and Diffusion-based (DreamTalk, 131). For zero-shot evaluation against unseen SOTA generators, we curate 100 videos from the official OmniSync release, and additionally synthesize 300 videos each for Sonic, KDTalker, and InfiniteTalk using their official pretrained weights, pairing LaPa source identities with VCTK driving audio. Wan-S2V is further evaluated on 85 videos obtained through its commercial API. Corresponding authentic samples are drawn from AVLips to maintain balanced class proportions.

Baseline reproduction protocol. To ensure equitable comparison, the 15 baselines are reproduced according to the availability of their training resources. (i) Methods that do not release training code (FTCN, LipForensics, AltFreezing, FGMDC, RealForensics, AVAD) are evaluated using their officially released pretrained weights. (ii) Image-level detectors (NPR, FreqNet, UnivFD) are likewise evaluated with official weights, as their image-only pipelines are incompatible with the video-level inputs required by LipSync forensics. (iii) Video-level detectors whose training code is publicly released (CADDM, TALL, LipFD, AVH-align, DFD-FCG, SpeechForensics) are fine-tuned on the same LipSync-A training split used by our method, guaranteeing that all multimodal comparisons share identical data access. To verify that fine-tuning preserves the baselines’ original capability, we additionally re-evaluate each fine-tuned model on its source benchmark. The reproduced accuracies closely track the originally reported ones—e.g., on FF++: CADDM 99.7→94.1, TALL 99.8→99.0, DFD-FCG 99.2→95.1, SpeechForensics 97.6→95.9; on AVLips: LipFD 93.1→96.1, AVH-align 86.3→88.4—confirming the integrity of our reproductions.

Analysis of frequency-baseline AUC We further investigate the low AUC reported for NPR (14.4%) and FreqNet (9.4%) on LipSync-A. A frame-level study over 160,532 test frames reveals two compounding effects. First, both detectors classify nearly all inputs as authentic, yielding a false-negative rate of 99.8%. Second, and more critically, real frames consistently receive higher forgery scores than forged ones (mean score: real 0.028 vs. fake 0.002), which inverts the ROC ordering and drives AUC below the random baseline. We attribute this counter-intuitive behavior to a domain mismatch between the source training distribution and LipSync forgeries. NPR and FreqNet are pretrained on ForenSynths/CNN-Detection, where forgeries are dominated by the high-frequency upsampling artifacts inherent to GAN-based full-face synthesis. LipSync forgeries, by contrast, modify only the lip region and apply temporal smoothing during reconstruction, which actively suppresses high-frequency content. Consequently, the natural micro-textures and rapid articulation of authentic lips elicit stronger frequency-domain responses than the smoothed synthetic counterparts, prompting both detectors to systematically rank real samples as more suspicious. This finding reinforces our central motivation: LipSync forgeries elude the artifact-based assumptions underpinning conventional detectors, and demand a fundamentally different forensic paradigm grounded in physiological coupling.

## D. Additional Robustness Study

To further assess the resilience of our proposed model, we conduct a comprehensive robustness study against a wide array of both visual and audio perturbations. These corruptions are designed to simulate common real-world scenarios, such as post-processing and re-compression.

## D.1. Visual Robustness

We evaluate robustness to visual corruptions by applying seven common perturbations in video processing, including Color Saturation (CS), Contrast (CC), Block-wise occlusion (BW), Gaussian Noise (GN), Gaussian Blur (GB), JPEG compression, and Pixelation (PXL), The specific hyperparameters corresponding to the five intensity fot each perturbation are detailed in Table 8.

Table 8. Robustness experiment parameters. Each perturbation method employs five unique sets of hyperparameter values, modifying them solely during the video preprocessing phase.
<table><tr><td>Type</td><td>Parameter</td><td>L1</td><td>L2</td><td>L3</td><td>L4</td><td>L5</td></tr><tr><td>Block-wise Color Contrast</td><td>block number</td><td>16</td><td>32</td><td>48</td><td>64</td><td>80</td></tr><tr><td>Color Saturation</td><td>contrast factor</td><td>0.85</td><td>0.725</td><td>0.6</td><td>0.475 0.35</td><td></td></tr><tr><td></td><td>saturation gain</td><td>0.4</td><td>0.3</td><td>0.2</td><td>0.1</td><td>0.0</td></tr><tr><td>Gaussian Blur</td><td>kernel size</td><td>7</td><td>9</td><td>13</td><td>17</td><td>21</td></tr><tr><td>Gaussian Noise</td><td>variance  $\sigma ^ { 2 }$ </td><td>0.001 0.002 0.005 0.01 0.05</td><td></td><td></td><td></td><td></td></tr><tr><td>JPEG Compression</td><td>quality drop</td><td>30</td><td>32</td><td>35</td><td>38</td><td>40</td></tr><tr><td>Pixelation</td><td>pixel block</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr></table>

We test all models under five levels of intensity for each perturbation. Fig. 10 provides a qualitative visualization of all seven distortion types at a challenging Level-3 intensity. Table 9 presents the quantitative comparison against SOTA baselines at this fixed Level-3 intensity. Our method (Ours) consistently and significantly outperforms all baselines across every perturbation category. Notably, our model achieves an average AUC of 99.11%, surpassing the next-best baseline (LipForensics) by +13.58%. This demonstrates that our model’s learned features are far more resilient to common corruptions. While methods like FTCN and RealForensic suffer catastrophic performance drops, particularly against noise and blur, our model remains highly stable, with performance (AUC > 99.5%) barely affected by compression, blur, and pixelation, which are ubiquitous in real-world videos.

Table 9. Robustness comparison against Level-3 perturbations. We report AUC (%) on the AVLips test set. Bold indicates the best performance, and underline marks the second-best.
<table><tr><td>Perturbation</td><td>FTCN LipFD</td><td></td><td>LipForensics RealForensic</td><td>Ours</td></tr><tr><td>Block-wise</td><td>68.70</td><td>96.15 93.50</td><td>62.50</td><td>99.59</td></tr><tr><td>Contrast</td><td>75.56 80.77</td><td>87.00</td><td>56.25</td><td>99.27</td></tr><tr><td>Saturation</td><td>69.41 91.28</td><td>93.25</td><td>64.58</td><td>97.71</td></tr><tr><td>Gaussian Blur</td><td>57.77 51.79</td><td>82.00</td><td>50.00</td><td>99.54</td></tr><tr><td>Gaussian Noise</td><td>54.20 51.54</td><td>71.00</td><td>45.83</td><td>98.50</td></tr><tr><td>Compression</td><td>64.57 90.35</td><td>80.00</td><td>58.33</td><td>99.59</td></tr><tr><td>Pixelation</td><td>61.14 57.18</td><td>91.99</td><td>60.42</td><td>99.56</td></tr><tr><td>Average</td><td>64.48 74.15</td><td>85.53</td><td>56.84</td><td>99.11</td></tr></table>

![](images/6d891394dafaf8ab92a71708686b07dbe047254e2789115933af8c4cdf1d8ba8.jpg)  
Figure 10. Visualization of the seven perturbation types at Level-3 intensity.

## D.2. Audio Robustness

We also analyze its resilience to perturbations applied only to the audio stream, and we apply a diverse suite of 12 common audio corruptions, including additive noise, pitch shifting, resampling, room impulse response (RIR) simulation, and time shifting. Table 10 details the performance of our model on the AVLips test set. Our model exhibits remarkable stability against most audio corruptions. Performance remains exceptionally high for perturbations common in real-world scenarios, such as resampling (to 16k or 44k) and reverberation (simulated via RIR). This suggests our model is not overfitting to specific acoustic properties of the training data. The most significant performance drops are observed, as expected, under heavy additive noise (noise heavy) and significant pitch distortion (pitch shift up3, noise light). Even in these challenging cases, the model maintains a high AUC above 97.3%, validating the robustness of our audio-visual fusion mechanism.

Table 10. Performance of our model under various audio corruptions. Bold and underline denote the best and second-best results per metric (Accuracy, F1-score, AUC), respectively. All metrics are in %.
<table><tr><td>Perturbation</td><td>Accuracy</td><td>F1-score</td><td>AUC</td></tr><tr><td rowspan="5">noise_heavy noise_light pitch_shift_down3 pitch_shift_up3 resample_16k</td><td>86.25</td><td>85.82</td><td>98.79</td></tr><tr><td>70.40</td><td>63.50</td><td>97.97</td></tr><tr><td>97.11</td><td>97.36</td><td>99.33</td></tr><tr><td>79.60</td><td>77.57</td><td>97.31</td></tr><tr><td>96.94</td><td>97.23</td><td>99.37</td></tr><tr><td>resample_44k rir_large_room</td><td>97.37 97.46</td><td>97.59 97.68</td><td>99.56 99.53</td></tr><tr><td>rir_medium_room</td><td>97.64</td><td></td><td></td></tr><tr><td>rir_small_room</td><td></td><td>97.85</td><td>99.54</td></tr><tr><td></td><td>97.81</td><td>98.00</td><td>99.55</td></tr><tr><td>silence_random</td><td>97.20</td><td>97.42</td><td>99.59</td></tr><tr><td>time_shift_backward</td><td>97.20</td><td>97.42</td><td>99.56</td></tr><tr><td>time_shift_forward</td><td>97.20</td><td>97.45</td><td>99.52</td></tr></table>

## D.3. Landmark Extraction Robustness

Since LipDA depends on facial landmarks extracted by Google MediaPipe to construct the head-pose representation in Stage I and the keypoint sequence for the TDM in Stage II, the reliability of landmark extraction under degraded visual conditions warrants explicit examination. We therefore stress-test the extractor on 100 videos (19,834 frames in total) under all seven Level-5 visual perturbations defined in Table 8, reporting both the per-frame landmark Failure Rate (FR) and the resulting detection AUC on the AVLips test set.

Table 11. Landmark extraction robustness under Level-5 visual perturbations. FR (%) is the per-frame landmark extraction failure rate; AUC (%) is reported on the AVLips test set. Columns are ordered by increasing FR.
<table><tr><td>Metric</td><td>Baseline</td><td>JPEG</td><td>BW</td><td>CC</td><td>PXL</td><td>GB</td><td>CS</td><td>GN</td></tr><tr><td>FR↓ AUC↑</td><td>6.0 99.72</td><td>6.3 99.60</td><td>6.6 99.69</td><td>8.8 96.99</td><td>11.4 98.55</td><td>11.7 98.33</td><td>13.9 92.10</td><td>34.7 90.98</td></tr></table>

As shown in Table 11, the MediaPipe extractor remains stable under the majority of perturbations: JPEG compression, block-wise occlusion (BW), and color contrast (CC) all keep FR below 9% and AUC above 96%. Gaussian noise (GN) constitutes the most adversarial case, elevating FR to 34.7%, yet LipDA still attains 90.98% AUC. We attribute this resilience to two compounding factors. First, our sliding-window mechanism discards any window in which more than half of the T frames fail landmark extraction; even under GN-L5, 86.57% of windows remain valid, providing sufficient temporal evidence for reliable inference. Second, the AUC degradation does not strictly track FR: color saturation (CS-L5) exhibits a moderate FR of 13.9% but a sharper AUC drop, because saturation degradation simultaneously compromises the appearance features extracted by the lip encoder rather than the landmark pipeline alone. We further note that perturbations severe enough to substantially impair landmark extraction also render the videos perceptually implausible, thereby diminishing

their deceptive utility in realistic adversarial scenarios.

## E. Extended Discussion

## E.1. Open-set Attribution

The continual emergence of new LipSync architectures renders the universe of potential forgery sources open-ended, raising a legitimate concern about the closed-set assumption underlying our attribution formulation. A direct open-set solution would replace supervised classification with unsupervised clustering over fingerprint features, but such schemes typically suffer from clustering instability and pseudo-label drift, yielding outputs that lack the interpretability required by forensic investigators. We therefore adopt a structured alternative that groups generators by their underlying architectural paradigm following the LipSync taxonomy of Meng et al. (2024), since generators within the same family share fundamental synthesis mechanisms such as iterative denoising or adversarial optimization, and consequently imprint structurally similar temporal fingerprints. This grouping confers an implicit forward compatibility, allowing a newly released generator to be mapped to its corresponding family even when its specific instance was unseen during training. We verify this property on held-out generators, finding that OmniSync and KDTalker are correctly assigned to the Diffusion family at 90.9% and 97.3% accuracy respectively, while EAT (Gan et al., 2023) is attributed to the Transformer family at 99.0%. Supervised family-level classification is nevertheless not a complete open-set solution, and future generators built on entirely novel paradigms will require explicit out-of-distribution detection or open-set recognition, extending LipDA with confidence-aware rejection thresholds and incremental family discovery therefore remains a promising direction for future work.

## E.2. Connection to Generative Fingerprint Methods

The notion of generative fingerprints has been extensively studied in image-level forensics, where works such as ManiFPT (Song et al., 2024) and Riemannian-Geometric Fingerprints (Song & Itti, 2025) analyze the statistical signatures left by generative architectures in pixel space. These methods primarily target full-face DeepFakes synthesized by image generators, where forgery cues manifest as spatial frequency artifacts or geometric distortions in a learned latent manifold. LipSync forgeries differ fundamentally in their forgery characteristics, since they preserve the authentic identity and visual contex while modifying only localized lip dynamics, leaving spatial fingerprints that are easily diluted by the surrounding real content. As a result, the static pixel-level signatures exploited by prior methods cannot be directly transferred to LipSync attribution. Our framework instead extracts fingerprints from the temporal dynamics of head motion and the cross-modal synchronization between lip articulation and audio, capturing how each generator family modulates the lip-pose coupling across time. To the best of our knowledge, no prior study has systematically investigated LipSync-specific attribution, and our work therefore offers a complementary perspective that extends the fingerprinting paradigm from spatial signatures in synthetic images to temporal signatures in audio-driven facial videos.

## F. Extended Experimental Results

## F.1. Per-Generator Detection on LipSync-A

To provide a fine-grained view of how detection performance varies across generator paradigms, Table 12 reports the per-generator breakdown of the visual-only variant of LipDA on the LipSync-A test split. Each generator’s forged samples are scored against the shared pool of 346 authentic samples, with video-level accuracy and AUC jointly reported in the ACC/AUC format. LipDA delivers consistently strong detection across the ten generators covered by the training distribution, attaining an average ACC of 94.6% and an average AUC of 97.4%. Detection is essentially saturated on the video-driven generators (X2Face, TPSM, LIA, FaceVid) and on the diffusion-based DreamTalk, while the audio-driven GAN-, Transformer-, and statistical-based generators are reliably identified. The marginally lower AUC observed on SadTalker is attributable to its tendency to inherit near-static head poses directly from the source frame, which compresses the dynamic range of the lip-pose inconsistency signal that our framework exploits. The breakdown therefore confirms that the physiological coupling cue generalizes broadly across diverse synthesis paradigms rather than relying on artifacts specific to any single generator family.

Table 12. Per-generator detection performance of the visual-only LipDA on the LipSync-A test split. Each cell reports video-level binary classification accuracy and AUC in ACC/AUC (%) format, both computed against the shared pool of 346 authentic samples. Generators are listed in alphabetical order.
<table><tr><td>Generator</td><td>DiNet</td><td>DreamTalk</td><td>FaceVid</td><td>IP-LAP</td><td>LIA</td><td>MakeItTalk</td><td>SadTalker</td><td>TPSM</td><td>Wav2Lip</td><td>X2Face</td><td>Avg.</td></tr><tr><td>ACC / AUC (%)</td><td>94.5/98.0</td><td>95.4/99.9</td><td>95.5/99.9</td><td>94.4/97.1</td><td>95.6/99.9</td><td>94.8/96.8</td><td>88.9/84.2</td><td>96.9/99.9</td><td>95.0/98.0</td><td>95.3/100.0</td><td>94.6/97.4</td></tr></table>

Table 13. Fine-grained 15-way model-level attribution performance on the LipSync-A test split. Generators are grouped by their architectural family for ease of comparison. All metrics are reported in %.
<table><tr><td>Family</td><td>Generator</td><td>Prec.</td><td>Recall</td><td>F1</td></tr><tr><td>Hybrid Transformations</td><td>X2Face</td><td>72.2</td><td>86.7</td><td>78.8</td></tr><tr><td rowspan="3">Keypoint-Driven</td><td>TPSM</td><td>71.4</td><td>33.3</td><td>45.5</td></tr><tr><td>LIA</td><td>71.9</td><td>76.7</td><td>74.2</td></tr><tr><td>FaceVid</td><td>46.7</td><td>46.7</td><td>46.7</td></tr><tr><td>Statistical</td><td>DiNet</td><td>96.4</td><td>90.0</td><td>93.1</td></tr><tr><td rowspan="3">GAN-based</td><td>MakeItTalk</td><td>100.0</td><td>96.7</td><td>98.3</td></tr><tr><td>Wav2Lip</td><td>80.0</td><td>66.7</td><td>72.7</td></tr><tr><td>TalkLip</td><td>65.8</td><td>83.3</td><td>73.5</td></tr><tr><td>VAE-based</td><td>SadTalker</td><td>89.3</td><td>83.3</td><td>86.2</td></tr><tr><td>Transformer-based</td><td>IP-LAP</td><td>86.2</td><td>83.3</td><td>84.8</td></tr><tr><td rowspan="5">Diffusion-based</td><td>DreamTalk</td><td>96.8</td><td>100.0</td><td>98.4</td></tr><tr><td>Sonic</td><td>90.6</td><td>96.7</td><td>93.6</td></tr><tr><td>KDTalker</td><td>96.8</td><td>100.0</td><td>98.4</td></tr><tr><td>OmniSync</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>InfiniteTalk</td><td>78.4</td><td>96.7</td><td>86.6</td></tr><tr><td>Overall (macro avg.)</td><td></td><td>82.8</td><td>82.7</td><td>82.0</td></tr></table>

## F.2. Fine-Grained 15-Way Model-Level Attribution

The main attribution results in Table 3 adopt a 5-way label space defined by architectural family, since family-level granularity offers forward compatibility to unseen generators built on the same paradigm. A complementary question, however, is whether LipDA can also distinguish individual model instances inside the same family, for instance separating Wav2Lip from TalkLip. To probe this, we retrain the Stage II attribution classifier with a 15-way model-level label space that assigns every generator in LipSync-A its own class, holding all other components, hyperparameters, and training data identical to the original recipe. Because the supervised target is fundamentally different from the 5-way protocol, the two sets of numbers are not directly comparable, and the 15-way results are reported here as a complementary stress test rather than as a refinement of the family-level results in the main paper.

Under this finer-grained protocol, LipDA attains an overall accuracy of 82.67% and a macro-average F1 of 82.04% on the LipSync-A test split. Table 13 reports the per-model breakdown grouped by family. Three observations emerge. First, the diffusion family remains highly separable at the model level, with DreamTalk, KDTalker, and OmniSync all attaining 100% recall and Sonic reaching 96.7%, indicating that distinct diffusion instances retain individually distinguishable temporal fingerprints despite their shared denoising backbones. Second, GAN-based instances are also distinguishable, with MakeItTalk, Wav2Lip, and TalkLip reaching recalls of 96.7%, 66.7%, and 83.3% respectively, confirming that within-family separation is achievable for the GAN paradigm. Third, the most pronounced residual confusion concentrates inside the keypoint-driven family, where TPSM, LIA, and FaceVid share landmark-based motion priors and consequently produce highly similar temporal patterns, leading to mutual misclassification. This within-family confusion is consistent with the rationale behind our family-level taxonomy and supports the family-level grouping adopted in the main paper as a more reliable operating point for downstream forensic deployment.

![](images/5d04070341c905e967ccc6399fe502b16021b171be21ed3d5f2d86c190833997.jpg)  
Figure 11. Representative failure cases of LipDA on the LipSync-A test split. All three patterns are also failed by the baselines evaluated in Table 2, indicating a shared operational limit rather than a weakness specific to our framework.

## F.3. Generalization to Full-Video Generation

The recent emergence of full-video generation paradigms, which synthesize the entire frame rather than locally editing the lip region, introduces a new failure surface for forgery detectors that rely on localized visual artifacts. To assess whether LipDA’s physiological coupling signal remains discriminative in this regime, we conduct an extended zero-shot evaluation that incorporates the commercial-grade full-video generator Wan-S2V (Wan et al., 2025) alongside four recent diffusion-based generators excluded from training. As reported in Table 14, LipDA achieves an average detection accuracy of 87.0% across the five unseen generators, with 88.2% on Wan-S2V specifically, indicating that the lip-pose coupling cue transfers naturally to forgeries where the entire video is synthesized rather than locally edited.

Table 14. Zero-shot detection accuracy (%) of LipDA on unseen SOTA generators, including the full-video generator Wan-S2V. Al models are evaluated without any fine-tuning on LipSync-A.
<table><tr><td>Generator</td><td>OmniSync</td><td>Wan-S2V</td><td>Sonic</td><td>KDTalker</td><td>InfiniteTalk</td></tr><tr><td>ACC (%)</td><td>95.5</td><td>88.2</td><td>87.8</td><td>84.6</td><td>78.8</td></tr></table>

We further verify the Stage II attribution framework on representative recent generators under the same zero-shot protocol, with 300 samples generated per model. Specifically, LatentSync (Li et al., 2024) is correctly attributed to the Diffusion family at 97.5% accuracy, consistent with 97.3% on KDTalker and 90.9% on OmniSync. Together, these detection and attribution results indicate that both the physiological inconsistency cue and the temporal motion fingerprint exploited by LipDA reflect general properties of audio-driven facial synthesis rather than artifacts specific to the LipSync editing pipelines on which our framework was trained.

## F.4. Bad Case Analysis

To better characterize the operational boundary of LipDA, we examine its failure cases on the LipSync-A test split and identify three recurring patterns, all of which are shared by every evaluated baseline rather than being unique to our framework. Figure 11 visualizes representative examples.

• Stationary speaker under complex camera motion, where the head remains nearly motionless relative to the body and the lip-pose inconsistency signal that our framework relies on becomes inherently weak.

• Persistent downward gaze, which renders the lip region largely invisible across frames, depriving the lip encoder of effective input and collapsing the contrastive comparison in Stage I.

• Prolonged facial occlusion by hands, microphones, or hair, which destabilizes landmark extraction and propagates noisy pose features into the temporal modules.

These cases collectively delineate a shared operational limit of detection paradigms that depend on visible lip dynamics and well-defined head motion as primary forensic cues, motivating future work on confidence-aware inference and complementary signals drawn from upper-face dynamics or full-body movement.

![](images/539289cd820d88cb436a1149c675dfa2b4bc0d9d9b2a097efebcfbda59f7d4a2.jpg)  
Figure 12. Spatial attention maps for the dual-branch encoders. Brighter colors indicate higher attentional weight.

## F.5. Attribution Ablation

We ablate our Stage II attribution components in Table 15. The MSM variant achieves a strong 92.92% ACC and 93.64% F1, establishing that audio-visual synchronization patterns are the core discriminative signal for attribution. Conversely, relying only on the TDM features causes a catastrophic performance drop to 74.78% ACC, proving TDM is insufficient alone. However, our Full Model, which combines both MSM and TDM, achieves the highest performance (97.50% ACC). This significant gain confirms that the TDM provides critical complementary temporal cues, validating the synergistic design of our framework.

Table 15. Ablation study on attribution model components. We report the average Accuracy (ACC, %) and F1-Score (F1, %) across five generator families on the LipSync-A.
<table><tr><td>Model Variant</td><td>ACC</td><td>F1</td></tr><tr><td>Full Model</td><td>97.50%</td><td>93.90%</td></tr><tr><td>temporal_only(TDM)</td><td>74.78%</td><td>73.58%</td></tr><tr><td>av_sync_only(MSM)</td><td>92.92%</td><td>93.64%</td></tr><tr><td>audio_only</td><td>90.71%</td><td>91.36%</td></tr></table>

## F.6. Dataset Case Analysis

Attention maps in Fig. 12 further reveal a clear functional disentanglement. The pose encoder attends to a broad facial context and the lip encoder is highly localized on fine-grained lip motion. This confirms our core mechanism operates by detecting a mismatch between this global context and local motion, validating our lip-head pose inconsistency hypothesis.

To provide a qualitative overview of the LipSync-A dataset and the distinctive characteristics of different generative families, we present visual comparisons in Fig. 13 and Fig. 14. These visualizations underscore the diversity of forgeries in our benchmark and highlight the model-specific patterns—such as variations in lip articulation sharpness, motion amplitude, and head pose dynamics—that form the basis for our attribution task. The discernible differences across generators validate the premise that each model family imparts a unique fingerprint, which our method is designed to capture.

![](images/39783a13840ba6138599f6caeadef1ddc60caf6406bba3578db58b0df4ebd7ea.jpg)  
Figure 13. Comparison of forgeries generated from the same source identity and driving video.

![](images/9c2a1e3fe3139a0075f4d6ca85d2ecd9368726326a60689736288847a6561f24.jpg)

Figure 14. Samples from each of the five audio-driven generator families in LipSync-A.  
![](images/1e8f4fb413481b193a31c6572d6e39266593e93bf66ef31f3a8e8618b1cca8e8.jpg)  
Figure 15. Visualization of samples from our constructed LipSync-A dataset.