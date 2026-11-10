# FMA-Net: Flexible Modality Adaptation for Robust Multimodal Face Presentation Attack Detection

Yuhao Gao, Yichen Shi, Member, IEEE, Zitong Yu, Senior Member, IEEE, Hongyang Wang, Member, IEEE, Hongrui Li, Member, IEEE, Ruixin Zhang, Shouhong Ding, Jun Feng

Abstract—Face Presentation Attack Detection (fPAD) is a fundamental component of secure face recognition systems. While multimodal fPAD improves robustness by exploiting complementary cues from RGB, depth, and infrared (IR) modalities, its practical deployment remains challenging when one or more modalities are unavailable or incomplete. Existing flexible-modality fPAD methods address this issue mainly through zero-filling or modality omission, which may discard useful cross-modal information and introduce modality-dependent bias. To overcome these limitations, we propose the Flexible Modality Adaptation Network (FMA-Net), a unified framework for flexiblemodality fPAD that compensates for missing modalities through instruction-guided latent-space translation. Specifically, the Face Modality Instruct Translator (FMIT) estimates target modality observations from available RGB inputs, while a context-aware category enhancement mechanism embeds learnable text prompts as modality-invariant semantic anchors to alleviate modality bias. The Facial Feature Adapter (F<sup>2</sup>A) module further enhances discriminative facial representations by integrating global facial structure and local spoof-relevant cues. In addition, a Syntheticto-Real Face Domain Alignment (SR-FDA) module reduces the feature discrepancy between translated and real modalities, with benefits that depend on the missing-modality pattern and data domain. Extensive experiments on the WMCA and PADISI-Face datasets validate the effectiveness of the proposed framework. On WMCA under intra-domain evaluation with a 70% modality missing ratio, FMA-Net achieves a mean ACER of 3.22% across four missing-modality protocols, compared with 4.24% for the strongest baseline by the same mean criterion. When both depth and IR are entirely unavailable, it achieves an ACER of 5.23%, compared with 5.99% for the best competing result in that setting. Cross-domain experiments further reveal the limitations of synthetic-to-real alignment under dataset shifts. These results demonstrate that FMA-Net provides a practical solution for fPAD under flexible and incomplete modality scenarios.

Index Terms—Face Presentation Attack Detection, Diffusion Model, Modality Translation, Flexible Modality, Prompt Learning.

![](images/1e248e5e5ce02222421db801876f9da65a0a4827a5754201907d3b3dea9dd215.jpg)  
Fig. 1. Comparison between our method and existing flexible-modality face Presentation Attack Detection (fPAD) approaches under a missing infrared (IR) modality. (a) Existing methods address missing modalities via skipping or zero-padding. (b) FMA-Net employs a modality-translation module to reconstruct the missing modality, enabling all modalities to be used by the fPAD model for robust inference. (c) Comparative analysis of task settings: unimodal, multimodal, and flexible-modality, in terms of input-modality structure, data-acquisition complexity, and model capabilities. The source face images are from the WMCA dataset [1].

## I. INTRODUCTION

With the rapid advancement of face recognition technologies, face presentation attack detection (fPAD) has become a crucial component for secure identity verification. From smartphone unlocking to facial payment systems, face recognition is increasingly integrated into daily life. However, as a sensitive biometric trait, facial data is vulnerable to theft and misuse. Attackers can readily exploit publicly available facial images to launch presentation attacks by displaying photographs, replaying videos, or fabricating 3D masks. Consequently, fPAD aims to determine whether the presented face originates from a live subject rather than a spoofed medium, and has attracted extensive research attention, particularly in high-security applications.

Recent advances in fPAD highlight the effectiveness of multimodal approaches, which leverage complementary information from heterogeneous modalities such as RGB, depth, IR, and thermal. By combining diverse cues, these methods significantly enhance robustness against sophisticated attacks including high-quality replays and 3D masks. However, their reliance on specialized hardware and fixed modality configurations limits scalability and practical adoption. Prior work on flexible-modality fPAD [2] seeks unified models that handle arbitrary modality combinations. Yet, existing strategies for coping with missing modalities—such as zero-filling or direct omission [3]—inevitably discard valuable information and lead to suboptimal performance. This underscores a critical gap: how to effectively compensate for unavailable modalities while preserving the benefits of multimodal fusion. Motivated by this challenge, this work investigates modality translation as a flexible compensation strategy for incomplete-modality fPAD, aiming to exploit complementary modality cues without requiring all modalities to be physically available at inference.

To address this problem, we propose the Flexible Modality Adaptation Network (FMA-Net), a unified framework for robust fPAD under incomplete modality availability. It first employs the Face Modality Instruct Translator (FMIT) to estimate target modality observations from RGB inputs through instruction-guided latent-space translation. Then, a lightweight Facial Feature Adapter $( \mathbf { F } ^ { 2 } \mathbf { A } )$ enhances facespecific representation learning by integrating global facial structure and fine-grained local spoof-relevant cues. Finally, to mitigate the discrepancy between translated and real modalities, Synthetic-to-Real Face Domain Alignment (SR-FDA) performs adversarial feature alignment to address modalityorigin differences during detector training; its benefit under cross-domain evaluation is assessed separately. To comprehensively evaluate flexible-modality fPAD under practical deployment conditions, we further establish a flexible-modality evaluation protocol that covers both intra-domain and crossdomain settings and systematically simulates partial- and fullmodality missing scenarios. The main contributions of this work are summarized as follows:

• Unified fPAD under Incomplete Modalities. We develop a single training–inference paradigm that directly uses available modalities and compensates for unavailable ones within the same pipeline, thereby reducing dependence on fixed modality configurations and improving robustness across varying missing ratios and modality combinations.

• Face Modality Instruct Translator (FMIT). We design an instruction-guided latent diffusion engine that translates RGB inputs into target modality observations, such as depth and IR, to support both training-time modality augmentation and inference-time missing-modality compensation.

• Facial Feature Adapter $( \mathbf { F } ^ { 2 } \mathbf { A } )$ . We introduce a lightweight face-focused adapter that enhances facial representation learning through global-structure modeling and local feature aggregation, improving sensitivity to spoof-relevant cues under both real and translated modality inputs.

• Synthetic-to-Real Face Domain Alignment (SR-FDA). We perform adversarial feature alignment between translated and real modalities to reduce translation-induced discrepancies during detector training, and evaluate both its benefits and its limitations across intra-domain and cross-domain settings.

## II. RELATED WORK

## A. Face Presentation Attack Detection

Depending on modality availability and modeling strategy, existing multimodal fPAD methods can be broadly categorized into two paradigms: fixed multimodal fusion and flexiblemodality learning.

With the rapid advancement of sensing technologies, multimodal methods have become increasingly prominent in face presentation attack detection (fPAD). These methods exploit complementary appearance, geometric, and spectral cues from RGB, depth, IR, and other modalities. Representative works include the multi-channel CNN with the WMCA dataset [1] and the PADISI-Face dataset with continual learning [4]. Advanced designs integrate modality-parallel branches [5], cross-modality losses [6], dual-stream attention fusion [7], hybrid structures [8], and hyperbolic embeddings [9]. GANbased cross-modality translation has also been explored to synthesize missing modality observations [10], but such methods usually rely on task-specific translation mappings and do not explicitly address the translated-real modality discrepancy during flexible-modality inference.

However, despite their strong performance, multimodal methods typically assume full-modality availability during both training and deployment, which is often unrealistic in practice. This limitation has motivated research into flexible-modality frameworks that can adapt to varying input availability and device heterogeneity. For example, Liu et al. [10] employed GAN-based RGB-to-depth translation to improve cross-modality generalization, while FM-ViT and FM-CLIP [11], [12] introduced flexible-modality representations based on Transformer and vision-language modeling. Other studies introduced input-aware routing mechanisms to dynamically adjust architectures based on modality presence. These methods improve fPAD applicability in edge computing and resource-constrained environments. Different from these discriminative flexible-modality frameworks, our work uses instruction-guided modality translation to provide auxiliary modality observations and further aligns translated and real modality features for fPAD.

Early fPAD systems primarily relied on convolutional neural networks (CNNs) with multi-branch or attention-based fusion and cross-modal supervision [5]–[7]. Vision Transformers (ViTs), often instantiated as Conv–Transformer hybrids, were subsequently introduced to capture long-range dependencies and token-level relationships [8]. Building upon these discriminative architectures, large model-driven approaches have emerged, particularly those employing vision-language models (VLMs) such as CLIP [13]. By aligning visual and textual representations via contrastive learning, these models support natural language-guided detection. Recent works demonstrated the effectiveness of linguistic prompts for unsupervised and open-set fPAD [14], [15]. With the rise of multimodal large language models (MLLMs), efforts have been made to establish benchmarks [16] and propose interpretable frameworks integrating language reasoning with multimodal representation learning [17]. These advancements enhance semantic understanding, generalization, and interpretability, paving the way for language-aware security systems. However, most language-aware fPAD methods focus on semantic supervision or open-set recognition, while the problem of compensating unavailable physical modalities such as depth and IR remains underexplored.

Despite these advances, most existing approaches still suffer from limitations when facing missing modalities. Multimodal systems often assume full-modality availability during both training and inference, which rarely holds in real-world scenarios. Flexible-modality strategies alleviate this issue to some extent, but many of them rely on omission, zero-filling, or discriminative adaptation, leaving the complementary cues of unavailable modalities insufficiently exploited. This results in underutilization of modality-specific knowledge and may hamper system robustness. To address this gap, our work introduces an instruction-guided modality compensation framework that estimates auxiliary modality observations and incorporates synthetic-to-real feature alignment, aiming to improve antispoofing robustness under incomplete inputs.

## B. Generation Model For fPAD

Early deep generative models—including GANs [18], VAEs [19], and autoregressive models [20]—substantially improved image synthesis quality and enabled image-to-image translation [21], but typically faced trade-offs among fidelity, stability, and efficiency. Recent diffusion models establish a more stable training paradigm via a forward noising and reverse denoising process, from DDPM [22] to accelerated sampling in DDIM [23]. Latent diffusion further compresses the process into a learned latent space [24], making high-quality conditional generation computationally feasible. Building on this, controllable diffusion (e.g., ControlNet [25]) injects structure priors to enhance conditional precision. These properties make latent diffusion suitable for modality translation, where the target modality may contain ambiguous or one-to-many mappings conditioned on RGB observations.

In face restoration, RestoreFormer by Wang et al. [26] uses cross-attention and a high-quality feature dictionary to recover facial details. Diffusion-based approaches have also shown promise. For example, Qiu et al. [27] introduced Diff-BFR, which leverages diffusion-based models to recover highquality facial images while avoiding the training instability commonly associated with GANs. In the domain of facial security, Liu et al. [28] proposed Adv-Diffusion, which utilizes latent diffusion models to generate high-fidelity adversarial examples capable of deceiving facial recognition systems, highlighting the utility of diffusion models in adversarial attack tasks. For style transfer, Brooks et al. [29] developed InstructPix2Pix, enabling expression and style editing through text prompts, while Qi et al. [30] proposed DEADiff, which enhances style controllability in text-to-image generation via disentangled representations. Taken together, these studies underscore the strong expressive and controllable capabilities of diffusion models in facial image restoration, security, and style manipulation, providing a technical foundation for instructionguided face modality translation. Nevertheless, modality translation for fPAD differs from general facial image restoration or editing because the translated observations should preserve spoof-relevant cues rather than merely improve perceptual quality.

## III. THE PROPOSED METHOD

This section presents the Flexible Modality Adaptation Network (FMA-Net), a unified framework for flexiblemodality fPAD that directly uses available modalities and compensates for unavailable ones through translated auxiliary observations. We first outline the overall pipeline in Sec. III-A. We then introduce the Face Modality Instruct Translator (FMIT), an instruction-guided latent-diffusion translator that estimates target modality observations, such as depth and IR, from RGB inputs, as described in Sec. III-B. Next, we describe learnable prompts together with the Facial Feature Adapter $( \mathbf { F } ^ { 2 } \mathbf { A } )$ , which aggregates multi-scale and region-aware facial cues to strengthen face-specific representation learning, as described in Sec. III-C. Finally, we detail Synthetic-to-Real Face Domain Alignment (SR-FDA), which reduces the distribution gap between translated and real modalities during detector training, with protocol-dependent effects evaluated in the experiments, as described in Sec. III-D.

## A. Overview of FMA-Net

To handle incomplete modality acquisition in practical deployment, we develop a unified flexible-modality fPAD framework that integrates modality translation, discriminative feature adaptation, and synthetic-to-real domain alignment into a cohesive pipeline, as illustrated in Fig. 2. The framework first employs the Face Modality Instruct Translator (FMIT) to estimate auxiliary depth or IR observations from accessible RGB inputs through instruction-guided latent diffusion. The translated and real modalities are then jointly processed by an image encoder equipped with the Facial Feature Adapter $( \mathbf { F } ^ { 2 } \mathbf { A } )$ , which enhances spoof-discriminative representation via adaptive local–global feature aggregation. To mitigate the distribution gap between translated and real modalities, the Synthetic-to-Real Face Domain Alignment (SR-FDA) module is incorporated based on adversarial learning. Through this joint design, the proposed system improves robustness and generalization under varying modality availability. FMIT is trained separately and kept frozen during detector training. The learnable detector prompts, facial feature adapter, and SR-FDA domain discriminator are optimized jointly during detector training; the detector objectives do not update FMIT. Each component is detailed in the following subsections.

## B. Face Modality Instruct Translator

In flexible-modality fPAD, different modalities exhibit complementary appearance, geometric, and spectral cues for distinguishing bona fide and attack presentations. However, in practical applications, modality information is frequently incomplete, especially under flexible-modality protocols where not all samples contain full multimodal data. Existing modalityrobust learning or prompt-learning methods alleviate this issue to some extent, but they do not explicitly estimate the unavailable modality observations that may provide complementary spoof-relevant cues.

We propose the Face Modality Instruct Translator (FMIT), which takes a visible-light image and a modality

![](images/f9f0c1866a2bb10541f1cbbbabdb8a4d749dafbc4a10c39321f8c733e741237f.jpg)  
Fig. 3. Overview of the modality translation instruction dataset construction and the face missing-modality compensation process. Steps 1–2 illustrate dataset creation through instruction design and paired data generation, while step 3 shows modality translation by FMIT for estimating missing modality observations. The source face images are from the WMCA dataset [1].

![](images/1fc59fc9b146e85259f483e19317a2feb8c778a0d373a6bed1add63a9f76472e.jpg)  
Fig. 2. Overview of the proposed FMA-Net. (a) Learnable context and the text encoder capture face-related semantic features. (b) FMIT estimates missin modality observations in the latent space. (c) The F<sup>2</sup>A module extracts visual features from both real and translated multimodal inputs. (d) SR-FDA mitigates domain shift by encouraging domain-invariant feature learning. The source face images are from the WMCA dataset [1].

## Modality Translation Instruction Dataset

## Design Modality Translation Instruction Template1Input command template：

Transform this image into <MODAL> MODAL ∈ { IR , Depth }

## Generate Training Data Pairs2

Instruct “Transform this image into infrared.”<sup>IR</sup>=

![](images/0fac9a3570a7bdeb7ca7b542b727a169dd767350f0ff11db24d3e76f299ec05f.jpg)

![](images/335b057d53e7491ad5cc46daf94555e202d0c67289356ba1454a96bc33dec9a0.jpg)

Face Missing Modality Complete

## Modality Translation3

translation instruction as inputs and estimates target modality observations such as IR and depth. The task is formulated as supervised instruction-guided translation: the instruction generator produces a concise prompt, such as “Transform this image into the X modality”, according to the target modality, and the translator learns to produce the corresponding targetmodality observation. To train FMIT, a paired instruction dataset is constructed by extracting RGB–IR and RGB–depth samples from a multimodal face database, as illustrated in Fig. 3. Each training triplet consists of a visible-light image, a modality instruction, and the paired target-modality image. The instructions specify only the target modality and do not require the bona fide/spoof label or attack category, which are unavailable at inference time. This permits the same instruction template to be used for both seen and unseen attack categories.

The input data include a visible-light image and any avail able auxiliary modalities, such as IR or depth images. Based on the modality availability of the input sample, the instruction generator produces a corresponding set of modality translation instructions, denoted as P. For instance, if the IR modality is unavailable, the generated instruction $P _ { \mathrm { I R } }$ guides FMIT to estimate an IR observation from the RGB input.

FMIT performs modality compensation based on a stable diffusion framework, with the core translation function denoted as $f ( \cdot )$ . When the target modality is IR, the input consists of the RGB image $X _ { \mathrm { R G B } }$ and the translation instruction $P _ { \mathrm { I R } }$ . The translated observation $\hat { X } _ { \mathrm { I R } }$ is computed as:

$$
\hat { X } _ { \mathrm { I R } } = f \left( X _ { \mathrm { R G B } } , P _ { \mathrm { I R } } \right)\tag{1}
$$

FMIT performs conditional modality translation in the latent space of a pretrained VAE. For a paired RGB observation $X _ { \mathrm { R G B } }$ and target-modality observation $X _ { m } .$ , where $m \in$ {IR, Depth}, let $z _ { 0 } = \mathcal { E } ( X _ { m } )$ denote the clean target latent and $c _ { \mathrm { R G B } } = \mathcal { E } ( X _ { \mathrm { R G B } } )$ the image condition. The VAE decoder

D maps the predicted target latent back to the target-modality image.

During training, Gaussian noise is added to the target latent according to the diffusion noise schedule:

$$
z _ { t } = a _ { t } z _ { 0 } + \sigma _ { t } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) , \quad t \in \{ 1 , \dots , T \} .\tag{2}
$$

Here, $a _ { t }$ and $\sigma _ { t }$ are the time-dependent signal and noise coefficients, with $a _ { t } ^ { 2 } + \sigma _ { t } ^ { 2 } = 1$ . These coefficients are distinct from the modality-missing probability α used in the experiments. The clean latent $z _ { 0 }$ is obtained from the target observation; only the added noise ϵ is sampled from the standard normal distribution.

The instruction encoder $\tau _ { \theta }$ maps the target-modality instruction p to a text condition. The denoising UNet $\epsilon _ { \theta }$ receives the noisy target latent, timestep, RGB condition, and text condition. Its conditional noise-prediction objective is

$$
L _ { \mathrm { c o n d } } = \mathbb { E } _ { ( X _ { \mathrm { R G B } } , X _ { m } , p ) , t , \epsilon } \big [ \| \epsilon - \epsilon _ { \theta } ( z _ { t } , t , c _ { \mathrm { R G B } } , \tau _ { \theta } ( p ) ) \| _ { 2 } ^ { 2 } \big ] .\tag{3}
$$

The paired observations and instructions are sampled from the modality-translation training data. The RGB condition supplies image content, while the instruction identifies the target modality. The pretrained VAE and instruction encoder are frozen, and only the denoising UNet is fine-tuned in this stage; the retained encoder notation $\tau _ { \theta }$ does not imply parameter sharing with the UNet.

By introducing conditional control, the model can perform instruction-guided translation across different facial modalities and estimate missing modality observations, thereby providing auxiliary inputs for downstream flexible-modality fPAD.

## C. Learnable Prompt Learning For fPAD

Feature extraction is conducted through an image encoder $E _ { \mathrm { I } }$ and a text encoder $E _ { \mathrm { T } }$ . In the textual stream, a contextaware class enhancement strategy is employed to improve cross-modality alignment of textual features and alleviate modality bias in flexible-modality scenarios. In the visual stream, we introduce the Facial Feature Adapter $( \mathbf { F } ^ { 2 } \mathbf { A } )$ into $E _ { \mathrm { I } }$ to enhance face-specific representation learning. Unlike a conventional adapter that mainly performs generic feature transformation or parameter-efficient tuning, $\mathrm { F } ^ { 2 } \mathrm { A }$ is designed for fPAD by jointly modeling global facial structure and local spoof-relevant cues from both real and translated modality inputs.

As illustrated in Fig. 2, the input facial images may include real RGB observations together with real or translated depth and IR observations, depending on the modality availability of each sample. During each fPAD process, the model processes textual and visual feature streams in parallel.

1) Text Feature Flow: For the textual stream, the model uses the class label set Labels = {Bonafide, Spoof} together with contextual prompts. Let $v _ { i }$ denote the context prompt containing the name of class i. These prompts are processed by the text encoder $E _ { \mathrm { T } }$ to generate textual features, which are represented as:

$$
\hat { w } _ { i } = \frac { E _ { \mathrm { T } } ( v _ { i } ) } { \Vert E _ { \mathrm { T } } ( v _ { i } ) \Vert _ { 2 } } , \quad i \in \{ 1 , \dots , K \}\tag{4}
$$

To further enhance cross-modality alignment, Context Optimization (CoOp) is employed to learn modality-independent textual prompts. By replacing manually designed fixed templates with learnable contextual embeddings, CoOp improves the representation of class semantics and provides modality invariant semantic anchors for flexible-modality fPAD.

For reference, the generic shared-context form concatenates context vectors with a class name as follows. Our experiments use class-specific context with the class token in the middle, as specified below.

$$
v = [ V ] _ { 1 } [ V ] _ { 2 } \cdot \cdot \cdot [ V ] _ { M } [ C L A S S ]\tag{5}
$$

Here, $[ V ] _ { m } \ ( m \in \{ 1 , \ldots , M \} )$ denotes the learnable context vectors, where M represents the number of such vectors. The token “[CLASS]” serves as a class placeholder, indicating the textual description of the class label. K is the number of categories. The normalized text feature $\hat { w } _ { i }$ is defined in Eq. 4. Let $F$ denote the sum of the normalized modality features, defined below. The score $\hat { w } _ { i } ^ { \top } \boldsymbol { F }$ is the sum of the per-modality cosine similarities; the summed feature is not normalized again. The probability that the input image belongs to class i is computed using a positive classification temperature $T _ { \mathrm { c l s } }$ . This scalar controls the sharpness of the class probabilities and is distinct from the FMIT instruction encoder $\tau _ { \theta }$ . The probability is given by:

$$
p ( y = i \mid x ) = \frac { \exp \left( \hat { w } _ { i } ^ { \top } F / T _ { \mathrm { c l s } } \right) } { \displaystyle \sum _ { j = 1 } ^ { K } \exp \left( \hat { w } _ { j } ^ { \top } F / T _ { \mathrm { c l s } } \right) }\tag{6}
$$

To further enhance the flexibility of prompt representation, the class label can be inserted at different positions within the prompt. In addition to the aforementioned postfix structure, the class label can also be embedded in the middle of the context vectors. An alternative structure of the prompt vector is as follows:

$$
v = [ V ] _ { 1 } \cdot \cdot \cdot [ V ] _ { \frac { M } { 2 } } [ C L A S S ] [ V ] _ { \frac { M } { 2 } + 1 } \cdot \cdot \cdot [ V ] _ { M }\tag{7}
$$

To enhance class-specific representations, a Class-Specific Context (CSC) strategy is introduced, where a unique set of context vectors is independently learned for each class. These vectors are optimized per class, instead of being shared across classes.

$$
v _ { i } = [ V ] _ { 1 } ^ { i } \cdot \cdot \cdot [ V ] _ { M / 2 } ^ { i } [ \mathrm { C L A S S } _ { i } ] [ V ] _ { M / 2 + 1 } ^ { i } \cdot \cdot \cdot [ V ] _ { M } ^ { i }\tag{8}
$$

The class-specific context vectors are independently parameterized and optimized by the classification objective. We use this CSC formulation with the middle class-token position in all reported detector experiments.

2) Visual Feature Flow: For the visual stream, the modality set $\{ X _ { \mathrm { R G B } } , X _ { \mathrm { I R } } , X _ { \mathrm { D e p t h } } \}$ may contain both real observations and translated auxiliary observations, depending on the missing-modality condition. Each modality input is processed by the image encoder equipped with $\mathrm { F } ^ { 2 } \mathrm { A }$ , denoted $E _ { \mathrm { I } } ^ { \mathrm { F ^ { 2 } A } }$ where the adapters operate inside the visual Transformer blocks rather than on raw input images. The extracted modality features are individually normalized and summed in the shared embedding space. Let $\mathcal M ( x )$ contain the observed or completed modalities used for sample x; an uncompleted missing modality makes no contribution. Feature extraction and aggregation are given by:

$$
F _ { j } = E _ { \mathrm { I } } ^ { \mathrm { F ^ { 2 } A } } ( X _ { j } ) , j \in \{ \mathrm { R G B } , \mathrm { I R } , \mathrm { D e p t h } \}\tag{9}
$$

$$
F = \sum _ { j \in \mathcal { M } ( x ) } \frac { F _ { j } } { \| F _ { j } \| _ { 2 } }\tag{10}
$$

Since $\mathrm { C L I P } \mathrm { s }$ image encoder $E _ { \mathrm { I } }$ is pretrained on generalpurpose image-text data, it may be less sensitive to finegrained facial textures and local spoof artifacts that are critical for fPAD. To address this limitation, we propose the Facial Feature Adapter $( \mathbf { F } ^ { 2 } \mathbf { A } )$ , a lightweight fPAD-oriented adaptation module inserted into the visual stream. Unlike conventional adapters that mainly learn generic residual transformations, $\mathrm { F } ^ { 2 } \mathrm { A }$ explicitly combines global facial structure modeling with local spoof-cue aggregation, with the aim of strengthening facial feature extraction for fPAD. The module consists of four components:

• Feature Reduction (FR): $\mathrm { ~ A ~ } 1 \times 1$ convolution reduces the channel dimension of the input feature map while preserving spatial locations, producing a compact representation for subsequent global and local modeling.

• Global Feature Extractor (GFE): Global average pooling (GAP), followed by a $1 \times 1$ convolution and a Sigmoid activation, generates recalibration weights that emphasize global facial structure and contour-level cues.

• Local Feature Extractor (LFE): A $3 \times 3$ convolution captures local spatial patterns from facial regions such as the eyes, nose, and mask boundaries, improving sensitivity to fine-grained spoof-relevant cues.

• Fusion Alignment (FA): The global and local branches interact through cross-attention. The fused representation is then projected back to the original dimension D for downstream fPAD classification.

$$
\left\{ \begin{array} { l } { F _ { G } = G F E \left( F R ( F ) \right) } \\ { F _ { L } = L F E \left( F R ( F ) \right) } \\ { F _ { a } = F A ( F _ { \mathrm { G } } , F _ { \mathrm { L } } ) } \end{array} \right.\tag{11}
$$

Here, $F R ( \cdot )$ reduces channel dimensionality, $G F E ( \cdot )$ captures global structural responses, $L F E ( \cdot )$ extracts local spatial cues with a $3 \times 3$ convolution, and $F A ( \cdot , \cdot )$ denotes the fusion-alignment module, which integrates the global and local features through cross-attention and projects the fused representation back to dimension D. This design differs from a conventional adapter by explicitly decomposing facial representation adaptation into global-structure and local-spoof-cue branches.

The class score is the inner product of the normalized text feature and the summed visual feature, equivalently the sum of the per-modality cosine similarities. The highest score determines the final detection result. The final predicted outcome yˆ is given by:

$$
\hat { y } = \arg \operatorname* { m a x } _ { i \in L a b e l s } \hat { w } _ { i } ^ { \top } F\tag{12}
$$

## D. Synthetic-to-Real Face Domain Alignment

Since translated auxiliary observations are produced by the face modality translator, they may introduce domain shift even when their visual appearance is close to that of real observations. This shift primarily appears in distributional characteristics such as style, noise, and texture statistics that are not directly related to fPAD decision making.

To mitigate the aforementioned issue, the Synthetic-to-Real Face Domain Alignment (SR-FDA) module is introduced on top of the flexible-modality translation-completion strategy, aiming to alleviate the adverse impact of domain shift caused by modality synthesis on detection performance. This module adopts adversarial learning to optimize the image feature extractor, encouraging it to preserve spoof-discriminative cues while suppressing modality-source-specific deviations.

Inspired by Domain-Adversarial Training of Neural Networks (DANN) [31], the input images are divided into two domains according to modality origin: the real-observation domain and the translated-observation domain. Each domain contains both bona fide and presentation attack samples. During training, SR-FDA jointly pursues two objectives: 1) optimizing the image feature extractor $E _ { \mathrm { I } }$ and the classifier ω for accurate bona fide/spoof classification across both domains; and 2) introducing a domain discriminator d to predict whether a visual feature originates from a real or translated modality. Through adversarial optimization, $E _ { \mathrm { I } }$ is encouraged to make the two domains indistinguishable in feature space.

As illustrated in Fig.4, SR-FDA consists of the image feature extractor $E _ { \mathrm { I } }$ , the text feature extractor $E _ { \mathrm { T } }$ , the classifier $\omega ,$ and the domain discriminator $d .$ Different from generic domain-adversarial adaptation, SR-FDA defines the adversarial domains by modality origin rather than by subject identity or dataset source. Therefore, the discriminator is used specifically to reduce the synthetic-to-real discrepancy introduced by FMIT, while the fPAD classifier remains optimized for bona fide/spoof separation.

$$
\hat { y } = \omega \left( F , \{ \hat { w } _ { i } \} _ { i = 1 } ^ { K } \right)\tag{13}
$$

The classifier uses visual–textual similarity. Classification gradients update the learnable detector prompts and the trainable visual parameters, including $\mathrm { F } ^ { 2 } \mathrm { A } ;$ they do not update the separately trained FMIT. The domain branch operates on visual features, so its reversed gradients constrain the trainable visual parameters rather than directly updating the text prompts.

The classification task adopts the Cross-Entropy Loss to measure the discrepancy between the predicted label distribution $q ( x _ { i } )$ and the ground-truth label distribution $p ( x _ { i } )$ . The classification loss $\ell _ { \mathrm { y } }$ is computed as follows:

$$
\ell _ { y } = - \sum _ { i = 1 } ^ { n } p ( x _ { i } ) \log { \left( q ( x _ { i } ) \right) }\tag{14}
$$

![](images/28d6dafca3322cdbb07ed91dd53e504c9ae84a36d8e7cd5b2bf910753ea7f2ed.jpg)  
Fig. 4. Illustration of the proposed Synthetic-to-Real Face Domain Alignment (SR-FDA). By leveraging adversarial learning between real and synthetic data domains, SR-FDA extracts domain-invariant features to alleviate domain shift introduced by generated data, while aligning visual and textual features for robust real vs. spoof classification. The source face images are from the WMCA dataset [1].

Due to the domain shift between translated auxiliary observations and real observations, a domain discriminator d is introduced for modality-origin classification. The objective of d is to determine whether the input feature originates from the real-observation domain $D _ { s } ^ { r }$ or the translated-observation domain $D _ { s } ^ { g }$ . Specifically, the domain discriminator d distinguishes between image features $F _ { r }$ from real observations and $F _ { g }$ from translated observations. The domain discrimination objective $\ell _ { d }$ is defined as follows:

$$
\ell _ { \mathsf { d } } ( X _ { r } , X _ { g } ) = \mathbb { E } \left[ \log \left( d ( F _ { r } ) \right) \right] + \mathbb { E } \left[ \log \left( 1 - d ( F _ { g } ) \right) \right]\tag{15}
$$

A Gradient Reversal Layer (GRL) [31] is introduced between the domain discriminator and the image feature extractor. During the forward pass, GRL acts as an identity mapping; during backpropagation, it multiplies the gradient from the domain discriminator by a negative coefficient. This mechanism enables joint adversarial training: the discriminator learns to identify modality origin, whereas the image encoder is optimized to remove modality-origin information from the visual representation.

Here, $\ell _ { d }$ is a domain log-likelihood, which the discriminator maximizes, equivalently minimizing the negative loglikelihood. The reversed gradient encourages the trainable visual parameters to reduce this log-likelihood while also minimizing the classification loss. Only trainable parameters receive updates; the notation below describes the branch objectives, not separate training stages. The classification-side parameters include the learnable prompts used to construct the text representations.

The classification branch minimizes the fPAD classification loss to optimize the parameters $\theta _ { E _ { \mathrm { I } } }$ and $\theta _ { \omega }$ , which can be expressed as:

$$
\widehat { \theta } _ { E _ { \mathrm { I } } } , \widehat { \theta } _ { \omega } = \arg \operatorname* { m i n } _ { \theta _ { E _ { \mathrm { I } } } , \theta _ { \omega } } \ \ell _ { y } \left( \theta _ { E _ { \mathrm { I } } } , \theta _ { \omega } \right)\tag{16}
$$

Meanwhile, the domain branch optimizes the domain discriminator parameters $\theta _ { d }$ to separate real and translated observations, while the GRL reverses this gradient before it reaches $E _ { \mathrm { I } }$ . The domain objective can be formulated as:

$$
\hat { \theta } _ { \mathrm { d } } = \arg \operatorname* { m a x } _ { \theta _ { \mathrm { d } } } \ \ell _ { \mathrm { d } } \left( \hat { \theta } _ { E _ { \mathrm { I } } } , \theta _ { \mathrm { d } } \right)\tag{17}
$$

## IV. EXPERIMENTS

## A. Experiment Setup

Implementation details. All experiments were conducted using the PyTorch framework on a GPU with 24 GB of memory. FMIT was initialized from Stable Diffusion v1.5 and trained for conditional modality translation using RGB observations and target-modality instructions. The denoising UNet learns to predict the noise added to the target-modality latent representation, conditioned on the RGB observation and instruction. The variational autoencoder (VAE) and text encoder were frozen, and the UNet was optimized using AdamW with a learning rate of $5 \times 1 0 ^ { - 5 }$ . Input images were resized to $2 5 6 \times 2 5 6$ , and mixed-precision training, gradient checkpointing, random horizontal flipping, and a conditioning dropout probability of 0.05 were adopted.

FMIT is trained separately and remains frozen while the detector is trained. The learnable detector prompts, $\mathrm { F } ^ { 2 } \mathrm { A }$ , and the SR-FDA domain discriminator are optimized jointly during detector training. The detector was trained using Adam with a batch size of 16. For the fPAD detector, we used CLIP ViT-B/16 as the visual backbone with an input resolution of 224× 224. The prompt length was set to 16, class-specific context learning was enabled, and the class token was placed in the middle of the learnable context tokens.

The $\mathrm { F } ^ { 2 } \mathrm { A }$ module was implemented as a lightweight adapter inserted into the visual Transformer blocks. It uses a bottleneck projection and a $3 \times 3$ convolution on the reshaped patch-token grid to capture local spoof-relevant cues with limited additional parameters. SR-FDA was implemented with a lightweight domain discriminator and auxiliary domainclassification losses for depth and IR features, jointly optimized with the fPAD classification loss.

Datasets. We employed the WMCA and PADISI-Face datasets to evaluate the proposed method under multimodal and flexible-modality protocols. The WMCA dataset [1] consists of 1,679 videos from 72 subjects, recorded in four modalities: RGB, depth, IR, and thermal. It covers diverse presentation attack types including print, replay, glasses, fake heads, hard masks, flexible masks, and paper masks. Data were captured using an Intel RealSense SR300 (RGB, NIR, depth) and a Thermal Compact PRO (thermal).

The PADISI-Face dataset [4] contains 2,029 images from 182 participants, collected with a multi-sensor array spanning RGB, short-wave infrared (SWIR), and long-wave infrared (thermal). It includes a wide range of attack types such as disguise, print, 3D masks, and replay. The acquisition setup employed synchronized LED illumination across different wavelengths to ensure temporal consistency and maximize sensor throughput. Experiments were conducted under both intra-domain and cross-domain settings across the two datasets to assess robustness and generalization under varying modality availability and domain shifts.

This study uses the existing WMCA and PADISI-Face datasets for secondary analysis. No participants were recruited, and no new human-subject data were collected for this study.

## B. Evaluation Protocol

To comprehensively evaluate model performance under vari-<sup>Visual</sup> <sup>P</sup> ous modality settings, we design a flexible-modality evaluation protocol based on [2] and [32], covering both intra-domain and cross-domain evaluations, using the WMCA and PADISI-Face datasets.

Intra-domain evaluation. The model is trained and tested on the same dataset. The fixed-multimodal task uses complete RGB, IR, and depth observations. In the flexible-modality task, missing auxiliary modalities are simulated during training and testing according to P1–P4 below.

Cross-domain evaluation. We evaluate both WMCA-to-PADISI-Face and PADISI-Face-to-WMCA transfer under submodality settings. In each direction, FMIT and the detector are trained using only the source-domain training data; no target-domain data are used for training. The trained FMIT is kept frozen when translating target-domain inputs at evaluation time. For the flexible-modality task in both intra-domain and cross-domain evaluation, we introduce random modality missingness during training and testing. RGB is always retained. For each auxiliary modality subject to missingness, α denotes its missing probability, set to 70% or 100%.

Four modality-missing protocols are defined (Fig. 5, panels (a–d) correspond to P1–P4): P1: RGB + IR with partial depth missing; P2: RGB + depth with partial IR missing; P3: RGB with a shared missingness mask for IR and depth, so that both are missing together with probability α; and P4: RGB with independently sampled missingness masks for IR and depth, each with missing probability α. When the missing ratio α = 100%, P3 and P4 become equivalent and are jointly denoted as P3&P4.

To address missing modalities, we adopt a flexible-modality translation strategy, dynamically generating the absent modalities based on sample conditions. The completed multimodal inputs are then fed into the fPAD model for training and inference, enhancing robustness and generalization in incomplete modality scenarios.

## C. Fixed Modality Scenario

As shown in Table I, we evaluate the proposed method on WMCA under both the visible-protocol (Grandtest) and the unseen-protocol (Leave-One-Out, LOO) settings, and compare it with eight state-of-the-art approaches to assess detection generalization to unknown presentation-attack types. Bold values indicate the best performance, and underlined values dept Flexible-Modal Face Anti-Spoofingnote the second-best. In the LOO protocol, attack IDs A1–A7 correspond to flexible mask, replay, fake head, print, glasses, paper mask, and hard mask attacks, respectively. FMA-Net achieves the lowest mean ACER across the seven unseenattack protocols (1.21%), compared with 2.17% for CPL-CLIP. Its sample standard deviation across these protocols is 2.03 percentage points. Notably, the ACER value for visible attacks is only 0.66%, while for the unseen flexible mask attack protocol, the ACER value is 0.37%. The ACER values for unseen head mask and print attacks are both 0, with unseen paper mask attacks yielding an ACER of 0.24% and unseen hard mask attacks yielding the best result.

![](images/ba6f5efbd776a8e4703bd5c7e60e0de06a26c5b4222203f4a80c3e340540a8de.jpg)  
(c) Depth and infrared (a) Depth missing (b) Infrared missing (d) Depth and infrared symmetric missing asymmetric missing  
Fig. 5. Illustration of flexible-modality translation and completion protocols: (a) depth-only missing, (b) IR-only missing, (c) symmetric missing of both depth and IR modalities, and (d) asymmetric missing of depth and IR modalities.

The relatively weaker result on unseen replay attacks suggests that replay-specific artifacts remain challenging under the proposed representation. Unlike mask or print attacks, replay attacks often exhibit subtle display-related cues, such as screen reflection, moire patterns, temporal blur, and boundary artifacts, which may be weakly expressed in still multimodal observations. Since FMA-Net focuses on translated auxiliary observations and local–global facial representation adaptation, these fine-grained display artifacts may be partially smoothed by modality translation or less emphasized by the learned adapter. This observation indicates a limitation of the current framework for attacks whose discriminative evidence is dominated by high-frequency reflection and display cues.

To further validate the performance of the proposed FMA-Net method when facing unseen attacks, an unseen attack protocol similar to the WMCA dataset was established on the PADISI-Face dataset. The experimental results are shown in Table II. The held-out attack categories A1–A9 represent disguised eye attacks, paper attacks, fake mask attacks, halfmask attacks, transparent mask attacks, makeup attacks, partial paper glasses attacks, silicone mask attacks, and tattoo attacks. The input modalities include RGB, depth, and IR. In the comparison methods, "-H" and "-E" represent the Hyp and Euc methods from the literature [9], respectively.

As shown in Table II, FMA-Net achieves a mean ACER of 3.17% across the nine unseen-attack protocols, the lowest among the listed RDI methods and the second-lowest across all listed configurations. MC-PixBiS-E-H with RD inputs obtains the lowest mean ACER of 3.15%. FMA-Net achieves an ACER of 0.00% for fake mask, transparent mask, and partial paper glasses attacks, and a tied second-lowest ACER of 0.14% for seen attacks.

TABLE I  
COMPARATIVE EXPERIMENTAL RESULTS OF DIFFERENT ATTACK PROTOCOLS ON THE WMCA DATASET (ACER,%)
<table><tr><td rowspan="2">Method</td><td rowspan="2">Seen</td><td colspan="7">Unseen</td><td rowspan="2">Mean±Std</td></tr><tr><td>A1</td><td>A2</td><td>A3</td><td>A4</td><td>A5</td><td>A6</td><td>A7</td></tr><tr><td>MC-PixBiS [33]</td><td>1.80</td><td>49.70</td><td>3.70</td><td>0.70</td><td>0.10</td><td>16.00</td><td>0.20</td><td>3.40</td><td>10.54±18.14</td></tr><tr><td>MCCNN-OCCL-GMM [34]</td><td>3.30</td><td>22.80</td><td>31.40</td><td>1.90</td><td>30.00</td><td>50.00</td><td>4.80</td><td>18.30</td><td>22.74±16.57</td></tr><tr><td>CMFL [6]</td><td>1.70</td><td>12.40</td><td>1.00</td><td>2.50</td><td>0.70</td><td>33.50</td><td>1.80</td><td>1.70</td><td>7.66±12.11</td></tr><tr><td>ViT [12]</td><td>2.71</td><td>11.95</td><td>1.44</td><td>3.78</td><td>0.00</td><td>18.02</td><td>0.58</td><td>4.43</td><td>5.74±6.75</td></tr><tr><td>MA-ViT [3]</td><td>1.45</td><td>9.76</td><td>0.93</td><td>0.55</td><td>0.00</td><td>14.00</td><td>0.00</td><td>1.46</td><td>3.81±5.67</td></tr><tr><td>FM-ViT [11]</td><td>1.00</td><td>3.56</td><td>0.72</td><td>0.00</td><td>0.00</td><td>12.00</td><td>0.43</td><td>0.73</td><td>2.49±4.37</td></tr><tr><td>FM-CLIP [12]</td><td>1.05</td><td>3.35</td><td>0.69</td><td>0.00</td><td>0.00</td><td>11.00</td><td>0.38</td><td>0.66</td><td>2.30±4.01</td></tr><tr><td>CPL-CLIP [35]</td><td>0.79</td><td>1.37</td><td>0.00</td><td>0.00</td><td>0.00</td><td>13.28</td><td>0.00</td><td>0.53</td><td>2.17±4.93</td></tr><tr><td>FMA-Net(ours)</td><td>0.66</td><td>0.37</td><td>2.32</td><td>0.00</td><td>0.00</td><td>5.42</td><td>0.24</td><td>0.14</td><td>1.21±2.03</td></tr></table>

Seen: Grandtest. A1–A7 (held-out attacks): flexible mask, replay, fake head, print, glasses, paper mask, and hard mask. Mean±Std summarizes Unseen ACERs only (sample standard deviation).

TABLE II  
COMPARATIVE EXPERIMENTAL RESULTS OF DIFFERENT ATTACK PROTOCOLS ON THE PADISI-FACE DATASET (ACER,%)
<table><tr><td rowspan="2">Modality</td><td rowspan="2">Method</td><td rowspan="2">Seen</td><td colspan="9">Unseen</td><td rowspan="2">Mean±Std</td></tr><tr><td>A1</td><td>A2</td><td>A3</td><td>A4</td><td>A5</td><td>A6</td><td>A7</td><td>A8</td><td>A9</td></tr><tr><td rowspan="4">R</td><td>ViT [36]</td><td>0.28</td><td>22.33</td><td>0.27</td><td>0.54</td><td>0.54</td><td>0.27</td><td>9.49</td><td>41.18</td><td>3.99</td><td>1.80</td><td>8.93±14.08</td></tr><tr><td>ViT-H [9]</td><td>0.14</td><td>24.66</td><td>0.27</td><td>0.41</td><td>0.27</td><td>0.27</td><td>13.12</td><td>32.36</td><td>3.04</td><td>1.67</td><td>8.45±12.23</td></tr><tr><td>ViT-H-H [9]</td><td>0.14</td><td>18.04</td><td>0.27</td><td>0.54</td><td>0.14</td><td>0.14</td><td>9.91</td><td>10.17</td><td>5.14</td><td>2.07</td><td>5.16±6.31</td></tr><tr><td>ViT-E-E [9]</td><td>0.55</td><td>24.43</td><td>0.27</td><td>0.27</td><td>0.41</td><td>0.14</td><td>16.2</td><td>14.31</td><td>3.18</td><td>0.41</td><td>6.62±9.22</td></tr><tr><td rowspan="4">RD</td><td>ViT-E-H [9]</td><td>0.14</td><td>20.54</td><td>0.00</td><td>0.41</td><td>0.41</td><td>0.14</td><td>8.37</td><td>9.50</td><td>5.14</td><td>0.27</td><td>4.98±6.95</td></tr><tr><td>MC-PixBiS-E-E [9]</td><td>0.33</td><td>4.53</td><td>0.54</td><td>0.54</td><td>0.14</td><td>0.41</td><td>5.43</td><td>32.63</td><td>0.41</td><td>0.95</td><td>5.06±10.52</td></tr><tr><td>MC-PixBiS-E-H [9]</td><td>0.14</td><td>1.35</td><td>0.14</td><td>0.27</td><td>0.95</td><td>0.27</td><td>5.70</td><td>18.45</td><td>0.41</td><td>0.81</td><td>3.15±5.99</td></tr><tr><td>ViT-E-E [9]</td><td>0.30</td><td>22.74</td><td>0.00</td><td>0.41</td><td>0.27</td><td>0.27</td><td>13.12</td><td>23.27</td><td>2.91</td><td>0.27</td><td>7.03±9.97</td></tr><tr><td rowspan="4">RI</td><td>ViT-E-H [9]</td><td>0.28</td><td>22.23</td><td>0.00</td><td>0.41</td><td>0.27</td><td>0.00</td><td>11.17</td><td>5.49</td><td>3.31</td><td>0.68</td><td>4.84±7.50</td></tr><tr><td>MC-PixBiS-E-E [9]</td><td>0.28</td><td>3.72</td><td>0.41</td><td>0.54</td><td>0.54</td><td>0.64</td><td>3.75</td><td>33.44</td><td>0.54</td><td>2.07</td><td>5.07±10.73</td></tr><tr><td>MC-PixBiS-E-H [9]</td><td>0.00</td><td>3.01</td><td>0.14</td><td>0.27</td><td>0.68</td><td>0.41</td><td>6.96</td><td>23.13</td><td>0.41</td><td>2.21</td><td>4.14±7.45</td></tr><tr><td>CoCoOp [37]</td><td>0.47</td><td>21.40</td><td>15.25</td><td>1.68</td><td>0.96</td><td>1.24</td><td>6.76</td><td>0.28</td><td>5.10</td><td>0.68</td><td>5.93±7.52</td></tr><tr><td rowspan="3">RDI</td><td>CoOp [38]</td><td>0.71</td><td>13.17</td><td>0.48</td><td>0.70</td><td>0.95</td><td>7.63</td><td>2.11</td><td>0.17</td><td>12.32</td><td>0.79</td><td>4.26±5.33</td></tr><tr><td>FMA-Ñet(ours)</td><td>0.14</td><td>12.99</td><td>0.28</td><td>0.00</td><td>0.14</td><td>0.00</td><td>7.04</td><td>0.00</td><td>7.58</td><td>0.54</td><td>3.17±4.81</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

A1–A9 (held-out attacks): disguised eye, paper, fake mask, half-mask, transparent mask, makeup, partial paper glasses, silicone mask, and tattoo. R/D/I: RGB/depth/infrared. Mean±Std summarizes Unseen ACERs only (sample standard deviation).

FMA-Net performs well in most mask-related attacks, indicating its effectiveness in detecting attacks with abnormal facial boundaries and geometric discontinuities. However, eyewear-style attacks remain more challenging because their spoof traces are spatially limited and can be visually similar to bona fide glasses. In such cases, the decision relies on subtle cues around the lens boundary, reflection pattern, and local material response, rather than on large-scale facial deformation. These localized cues may contribute to the variation in performance across attack types. The result suggests that the proposed modality translation and feature adaptation strategy is effective for broad facial spoof artifacts, while future improvements should further strengthen localized material-aware modeling for eyewear-like attacks.

## D. Flexible Modality Scenario

a) Intra-Domain Test: Experiments were conducted on the WMCA dataset with missing modality ratios of 70% and 100%, and the results are shown in Table III. As can be seen, under the 70% setting, the proposed FMA-Net method achieves the best ACER values across all four protocols when compared to five commonly used existing methods. In particular, for single missing modality scenarios, FMA-Net obtains ACER values of 2.15% and 2.78% in Protocols 1 and 2, significantly outperforming the second-best VP-FAS method, especially in Protocol 1 where the improvement reaches 1.52 percentage points (3.67% vs. 2.15%). Averaged over P1–P4, FMA-Net reduces ACER from 4.24% for VP-FAS to 3.22%. For two missing modalities, FMA-Net also achieves the best results, though the margin is relatively smaller. This demonstrates that face modality completion using translation can effectively enhance presentation attack detection.

When the missing modality ratio is increased to 100%, FMA-Net is compared against three flexible-modality methods (ViT [34], CoCoOp [37], CoOp [38]) and seven traditional multimodal methods (MA-Net [10], MC-PixBiS [33], CMFL [6], MA-ViT [3], CCL(Resnet) [39], CCL(CDCN) [39], MC-CNN [34]). The proposed method achieves the best ACER values in Protocols 3&4, reducing the ACER by 0.76 percentage points compared to the second-best result (5.23% vs. 5.99%). In Protocol 1, FMA-Net attains second-best performance, while in Protocol 2 it ties with CMFL for second-best performance, with ACER values slightly higher than the best results by 0.40 percentage points (2.55% vs. 2.15%) and 0.25 percentage points (1.70% vs. 1.45%), respectively. Notably, in Protocol 2, FMA-Net outperforms all flexible-modality methods but slightly lags behind the traditional multimodal MA-ViT, reflecting the inherent challenges of flexible-modality translation.

By comparing the two settings, it is observed that in Table III, the average ACER for Protocols 3&4 is higher than that of Protocols 1&2, with an increase of 1.50 percentage points (3.97% vs. 2.47%) under the 70% setting, and 3.10 percentage points (5.23% vs. 2.13%) under the 100% setting. This trend is expected because Protocols 3&4 rely more heavily on translated auxiliary observations, whereas Protocols 1&2 still preserve one non-RGB modality. In particular, Protocol 2 retains depth but lacks IR, which can be less favorable for attacks whose evidence is strongly associated with spectral reflection or material response. Therefore, a moderate amount of modality translation can provide useful complementary cues, while a high missing ratio increases the dependence on translated observations and may amplify translation uncertainty.

Experiments were conducted on the PADISI-Face dataset with modality missing ratios of 70% and 100% for flexiblemodality translation completion, and the results are shown in Table IV. As observed, the proposed FMA-Net method performs best in Protocols 1 and 2 under the 70% setting, with ACER values of 0.48% and 0.74%, respectively. It achieves the second-best result in Protocol 3 (0.74% vs. 0.69%). In Protocol 4, however, its ACER of 1.43% is higher than those of CoCoOp (0.90%) and CoOp (1.24%) by 0.53 and 0.19 percentage points, respectively. When the missing ratio is increased to 100%, the proposed method achieves the best ACER values in Protocols 1 and 3&4, and the second-best result in Protocol 2 with only a 0.20 percentage point gap (1.11% vs. 0.91%). These results indicate that translated modalities provide useful complementary information, but the benefit depends on the missing pattern and attack composition. The relatively modest margins in some protocols suggest that when the available evidence is already sufficient or when the missing modality contains highly localized cues, translated observations may introduce uncertainty into feature extraction. This is a possible explanation for the protocol-dependent performance of FMA-Net.

b) Cross-Domain Test: Tables V and VI compare crossdomain detection with and without SR-FDA. At a 70% missing ratio, alignment improves W-P P1 from 9.95% to 8.99%, but worsens the other seven reported settings, including W-P P3 from 9.72% to 22.01%. At a 100% missing ratio, it improves P-W P1 and P3&P4, but worsens the other four settings. Thus, its cross-domain benefit is limited and protocol-dependent. SR-FDA aligns real and translated observations within the training distribution; this objective is distinct from alignment between datasets. One possible explanation is that FMIT retains sourcedomain appearance or sensor characteristics in the translated observations, introducing a mismatch on the target dataset. Alignment learned on these source-domain observations may not transfer to the target domain and may suppress features useful for target-domain attack discrimination. These are possible explanations, not mechanisms established by the reported performance comparisons.

TABLE III  
COMPARISON RESULTS FOR THE FLEXIBLE-MODALITY TRANSLATION COMPLETION PROTOCOL ON THE WMCA DATASET (ACER,%)
<table><tr><td>Method</td><td>P1 (RI) P2 (RD)</td><td>P3 (R)</td><td>P4 (R)</td></tr><tr><td colspan="4">α = 70%</td></tr><tr><td>ViT [34]</td><td>10.61</td><td>5.86 8.70</td><td>6.37</td></tr><tr><td>Prompt [40]</td><td>8.70</td><td>5.52 5.84</td><td>7.21</td></tr><tr><td>CMFL [6]</td><td>6.53 4.95</td><td>5.08</td><td>5.37</td></tr><tr><td>CrossAtten [2]</td><td>8.47</td><td>5.29 8.33</td><td>7.15</td></tr><tr><td>VP-FAS [32]</td><td>3.67</td><td>3.91 4.54</td><td>4.83</td></tr><tr><td>FMA-Net(ours)</td><td>2.15</td><td>2.78 3.88</td><td>4.05</td></tr><tr><td colspan="4">α = 100%</td></tr><tr><td>MA-Net [10]</td><td>6.85</td><td></td><td></td></tr><tr><td>MC-PixBiS [33]</td><td></td><td>1.80</td><td></td></tr><tr><td>CMFL [6]</td><td></td><td>1.70</td><td></td></tr><tr><td>MA-ViT [3]</td><td></td><td>1.45</td><td></td></tr><tr><td>CCL(Resnet) [39]</td><td></td><td></td><td>30.69</td></tr><tr><td>CCL(CDCN) [39]</td><td></td><td></td><td>27.14</td></tr><tr><td>MC-CNN [34]</td><td></td><td></td><td>32.82</td></tr><tr><td>ViT [34]</td><td>4.02</td><td>3.10</td><td>9.84</td></tr><tr><td>CoCoOp [37]</td><td>3.46</td><td>3.76</td><td>7.60</td></tr><tr><td>CoOp [38]</td><td>2.15</td><td>2.65</td><td>5.99</td></tr><tr><td>FMA-Net(ours)</td><td>2.55</td><td>1.70</td><td>5.23</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

P1: depth missing; P2: IR missing; P3/P4: shared/independent depth–IR missingness.

TABLE IV  
COMPARISON RESULTS FOR THE FLEXIBLE-MODALITY TRANSLATION COMPLETION PROTOCOL ON THE PADISI-FACE DATASET (ACER,%)
<table><tr><td>Method</td><td>P1 (RI)</td><td>P2 (RD)</td><td>P3 (R)</td><td>P4 (R)</td></tr><tr><td></td><td>α = 70%</td><td></td><td></td><td></td></tr><tr><td>CoCoOp [37]</td><td>0.80</td><td>0.91</td><td>0.69</td><td>0.90</td></tr><tr><td>CoOp [38]</td><td>0.55</td><td>1.36</td><td>0.85</td><td>1.24</td></tr><tr><td>FMA-Net(ours)</td><td>0.48</td><td>0.74</td><td>0.74</td><td>1.43</td></tr><tr><td></td><td> $\alpha = 1 0 0 \%$ </td><td></td><td></td><td></td></tr><tr><td>CoCoOp [37]</td><td>0.44</td><td>2.20</td><td>0.93</td><td></td></tr><tr><td>CoOp [38]</td><td>0.99</td><td>0.91</td><td>1.07</td><td></td></tr><tr><td>FMA-Net(ours)</td><td>0.39</td><td>1.11</td><td>0.91</td><td></td></tr></table>

Protocols follow Table III.

TABLE V  
THE CROSS-DOMAIN COMPARISON RESULTS FOR FLEXIBLE-MODALITY TRANSLATION COMPLETION WITH $\alpha = 7 0 \%$ (ACER,%)
<table><tr><td rowspan="2">Method</td><td colspan="4">W-P</td><td colspan="4">P-W</td></tr><tr><td>Pl</td><td>P2</td><td>P3</td><td>P4</td><td>Pl</td><td>P2</td><td>P3</td><td>P4</td></tr><tr><td>CoCoOp [37]</td><td>10.99</td><td>28.75</td><td>9.86</td><td>16.07</td><td>18.93</td><td>22.11</td><td>18.12</td><td>17.25</td></tr><tr><td>CoOp [38]</td><td>14.25</td><td>20.30</td><td>11.29</td><td>12.95</td><td>19.47</td><td>18.56</td><td>19.25</td><td>19.36</td></tr><tr><td>FMA-Net w/o SR-FDA (ours)</td><td>9.95</td><td>19.80</td><td>9.72</td><td>11.29</td><td>19.46</td><td>15.79</td><td>17.89</td><td>16.63</td></tr><tr><td>FMA-Net w/ SR-FDA (ours)</td><td>8.99</td><td>21.84</td><td>22.01</td><td>20.93</td><td>19.80</td><td>19.75</td><td>18.87</td><td>17.94</td></tr></table>

Protocols follow Table III. W-P: WMCA to PADISI-Face; P-W: reverse direction.

TABLE VI  
THE CROSS-DOMAIN COMPARISON RESULTS FOR FLEXIBLE-MODALITY TRANSLATION COMPLETION WITH $\alpha = 1 0 0 \%$ (ACER,%)
<table><tr><td rowspan="2">Method</td><td colspan="3">W-P</td><td colspan="3">P-W</td></tr><tr><td>P1</td><td>P2</td><td>P3&amp;P4</td><td>P1</td><td>P2</td><td>P3&amp;P4</td></tr><tr><td>ViT-E-E [9]</td><td>11.78</td><td>11.90</td><td></td><td>13.14</td><td>15.39</td><td></td></tr><tr><td>ViT-E-H [9]</td><td>11.29</td><td>8.50</td><td></td><td>12.55</td><td>12.05</td><td></td></tr><tr><td>MC-PixBiS-E-E [9]</td><td>20.06</td><td>16.68</td><td></td><td>15.90</td><td>15.51</td><td></td></tr><tr><td>MC-PixBiS-E-H [9]</td><td>13.82</td><td>15.37</td><td></td><td>15.63</td><td>14.20</td><td></td></tr><tr><td>CoCoOp [37]</td><td>10.87</td><td>28.75</td><td>16.07</td><td>18.93</td><td>19.12</td><td>17.89</td></tr><tr><td>CoOp [38]</td><td>12.11</td><td>12.44</td><td>13.52</td><td>22.11</td><td>18.03</td><td>17.28</td></tr><tr><td>FMA-Net w/o SR-FDA (ours)</td><td>9.20</td><td>10.55</td><td>10.61</td><td>18.82</td><td>17.76</td><td>18.82</td></tr><tr><td>FMA-Net w/ SR-FDA (ours)</td><td>11.30</td><td>20.67</td><td>21.75</td><td>18.55</td><td>18.70</td><td>16.99</td></tr></table>

Protocols follow Table III. Transfer directions follow Table V.

The cross-domain experimental results for flexible-modality translation completion with a modality missing ratio of 100% are shown in Table VI. In addition to comparisons with flexible-modality methods, the results are also compared with the cross-domain results of four multimodal methods. As observed, FMA-Net without SR-FDA maintains competitive performance in several W-P protocols, while FMA-Net with SR-FDA obtains the best result in Protocols 3&4 of the P-W cross-domain task, with an advantage of 0.29 percentage points (16.99% vs. 17.28%). Nevertheless, the overall crossdomain advantage becomes less consistent when all non-RGB modalities are unavailable. In P3&P4 at 100% missingness, the model relies entirely on RGB and translated auxiliary observations; P1 and P2 retain one acquired non-RGB modality. Under such conditions, translated modalities can still provide useful structural and spectral priors, but they cannot fully replace physically acquired modalities, especially when the target domain exhibits different sensor characteristics or attack materials. These results reveal an important limitation of the current framework: modality translation is effective as a compensation mechanism, but its reliability decreases when both the missing ratio and domain gap are large.

## E. Ablation Study

When the modality missing ratio is set to 70%, the experimental results in Table VII show that incorporating SR-FDA in the flexible-modality translation completion protocol outperforms the baseline network and the variant without domain alignment. Adding F<sup>2</sup>A reduces ACER in Protocols 1, 3, and 4 by 1.07, 1.33, and 1.47 percentage points, respectively. In Protocol 2, however, ACER increases from 2.92% to 3.16%, indicating that the benefit of feature adaptation is protocoldependent. Notably, after adding SR-FDA, the ACER values are further reduced across all protocols, with the largest reduction of 0.71 percentage points (2.15% vs. 2.86%). These results support the benefit of SR-FDA in the evaluated intradomain setting, although the ACER comparisons alone do not establish the mechanism of improvement.

TABLE VII  
ABLATION STUDY OF PROMPT LEARNING, $\mathrm { F } ^ { 2 } \mathbf { A } ,$ , AND SR-FDA UNDER THE FLEXIBLE-MODALITY TRANSLATION COMPLETION PROTOCOL WITH $\alpha = 7 0 \%$ (ACER,%)
<table><tr><td>Prompt</td><td> $\mathrm { F } ^ { 2 } \mathrm { A }$ </td><td>SR-FDA</td><td>P1 (RI)</td><td>P2 (RD)</td><td>P3 (R)</td><td>P4 (R)</td></tr><tr><td>√</td><td></td><td></td><td>3.93</td><td>2.92</td><td>5.79</td><td>5.88</td></tr><tr><td>√</td><td>√</td><td></td><td>2.86</td><td>3.16</td><td>4.46</td><td>4.41</td></tr><tr><td>√</td><td>√</td><td>√</td><td>2.15</td><td>2.78</td><td>3.88</td><td>4.05</td></tr></table>

Protocols follow Table III. Results are on WMCA; all variants use FMIT completion.

TABLE VIII  
ABLATION STUDY OF PROMPT LEARNING, $\mathrm { F } ^ { 2 } \mathbf { A } ,$ , AND SR-FDA UNDER THE FLEXIBLE-MODALITY TRANSLATION COMPLETION PROTOCOL WITH $\alpha = 1 0 0 \%$ (ACER,%)
<table><tr><td>Prompt</td><td> $\mathrm { F } ^ { 2 } \mathrm { A }$ </td><td>SR-FDA</td><td>P1 (RI)</td><td>P2 (RD)</td><td>P3&amp;P4 (R)</td></tr><tr><td>√</td><td></td><td></td><td>2.15</td><td>2.65</td><td>5.99</td></tr><tr><td>√</td><td>√</td><td></td><td>2.11</td><td>2.03</td><td>5.87</td></tr><tr><td>√</td><td>√</td><td>√</td><td>2.55</td><td>1.70</td><td>5.23</td></tr></table>

Dataset, protocols, and FMIT completion follow Table VII.

With a modality missing ratio of 100%, Table VIII shows that adding SR-FDA reduces ACER from 2.03% to 1.70% in P2 and from 5.87% to 5.23% in P3&P4. In P1, however, ACER increases from 2.11% to 2.55%, a deterioration of 0.44 percentage points. Thus, alignment improves the settings where IR alone or both auxiliary modalities are missing, but is detrimental in the depth-missing setting evaluated here. These results show that the benefit of SR-FDA depends on the missing-modality pattern; they do not establish the mechanism responsible for the P1 degradation.

To examine the effect of the modality completion ratio, we evaluate a fixed, trained FMA-Net detector including SR-FDA on WMCA in a test scenario where 70% of samples have missing auxiliary modalities. The completion percentage is defined as $\beta = 1 0 0 N _ { \mathrm { c } } / N _ { \mathrm { m } } ,$ where $N _ { \mathrm { m } }$ is the number of modality-incomplete test samples and $N _ { \mathrm { c } }$ counts those receiving translation-based completion of their missing modalities. Thus, β is measured relative to the incomplete subset, not the entire test set. At $\beta = 0 .$ , the detector uses only the available modalities; at $\beta = 1 0 0$ , all missing modalities in that subset are completed, while acquired observations are retained. Only the test inputs change; the detector is not retrained. Figure 6 shows an overall reduction in ACER as completion increases, with intermediate fluctuations and an ACER of 0.60% at $\beta = 1 0 0$ . This experiment evaluates the effect of test-time completion under the stated sample-level missingness setting.

## F. Visualization Analysis

To further analyze the model’s attention areas in handling spoof detection tasks, the Gradient-weighted Class Activation Mapping (Grad-CAM) [41] method was used for model visualization and interpretation. Fig. 7 presents the attention heatmaps for two sets of sample images from the WMCA dataset, comparing features extracted by CLIP and the proposed FMA-Net. It can be observed that CLIP tends to allocate attention to irrelevant regions, whereas FMA-Net focuses more effectively on informative areas related to presentation attack cues. This demonstrates that FMA-Net better captures discriminative artifacts and improves detection reliability.

![](images/0d8cca4598d9248eee7f66328ac152389a0fd8d6dd88859fe3a0a3d3627e6724.jpg)

Fig. 6. Effect of the test-time completion percentage β on WMCA when 70% of test samples have missing auxiliary modalities. The full detector, including SR-FDA, is fixed. β denotes the percentage of incomplete samples whose missing modalities are completed; β = 0 uses only available modalities and β = 100 completes all missing modalities in that subset.  
![](images/b355bb9c5ee2168a065f880fbeeb5f89772668da4e1ff35394cc08e232fb12c0.jpg)  
Fig. 7. Comparison of CLIP and FMA-Net feature-attention heatmaps. Top is CLIP and bottom is FMA-Net. FMA-Net exploits a richer set of discriminative cues. The source face images are from the WMCA dataset [1].

To assess the effect of modality translation completion, Fig. 8 shows partial results of IR and depth modality translation on the WMCA dataset. The second and fourth columns represent the real IR and depth modalities, respectively, while the third and fifth columns show the corresponding modalities translated by FMIT from RGB inputs. For both bona fide faces and 3D mask attacks, the translated IR modality is broadly consistent with the real IR modality in facial contours and feature positions, but local texture discrepancies remain visible. For print and replay attacks, the translated IR observations also present strong reflection and contour attenuation, which are consistent with some IR-domain spoofing characteristics but may not fully preserve all fine-grained local details. The translated depth modality retains facial depth structures, especially around the eye sockets and cheeks, and captures the missing-depth property associated with 2D attacks. Overall, the visualization suggests that FMIT can recover useful modality-level cues, but the translated observations are not perfect substitutes for physically captured modalities. This limitation may contribute to performance variation under complex attacks, high missing ratios, or cross-domain settings.

![](images/5f97a1ecce3c203b93b999b57bbc48d0d6e1678da6e9e935bcca1528d6ea68c1.jpg)  
Fig. 8. Examples of modality translation. Column 1: ground-truth RGB face images, Columns 2 and 4: corresponding ground-truth IR and depth images, Columns 3 and 5: IR and depth images synthesized by FMIT from the RGB input. The source face images are from the WMCA dataset [1].

## V. CONCLUSION

This paper presented FMA-Net, a flexible-modality framework for multimodal face presentation attack detection under incomplete modality availability. By estimating unavailable modality observations from accessible RGB inputs, FMA-Net enables missing-modality compensation without requiring all physical modalities to be captured at inference time. The proposed framework further integrates face-oriented feature adaptation and synthetic-to-real domain alignment to enhance spoof-discriminative representation learning and reduce the discrepancy between translated and real observations. Experiments on WMCA and PADISI-Face demonstrate that FMA-Net achieves competitive performance under both intra-domain and cross-domain flexible-modality protocols.

The results also reveal several limitations. Translated observations can provide useful complementary cues, but they cannot fully replace physically acquired IR or depth modalities, especially under high missing ratios, large domain shifts, or attacks dominated by fine-grained reflection and material cues. Synthetic-to-real alignment is beneficial in some settings but does not consistently improve cross-dataset generalization, because modality-origin alignment and dataset transfer are distinct objectives. Future work will therefore focus on improving the fidelity and attack-cue consistency of modality translation, developing lightweight translators for deployment, and incorporating interpretable multimodal reasoning to further strengthen robustness in practical biometric security systems.

## REFERENCES

[1] A. George, Z. Mostaani, D. Geissenbuhler, O. Nikisins, A. Anjos, and S. Marcel, “Biometric face presentation attack detection with multichannel convolutional neural network,” IEEE transactions on information forensics and security, vol. 15, pp. 42–55, 2019.

[2] Z. Yu, A. Liu, C. Zhao, K. H. Cheng, X. Cheng, and G. Zhao, “Flexible-modal face anti-spoofing: A benchmark,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 6346–6351.

[3] L. Xiao, J. Feng, Y. Gao, and J. He, “Flexible modal face presentation attack detection based on atrous single stream vision transformer network,” Application Research of Computers/Jisuanji Yingyong Yanjiu, vol. 41, no. 3, 2024.

[4] M. Rostami, L. Spinoulas, M. Hussein, J. Mathai, and W. Abd-Almageed, “Detection and continual learning of novel face presentation attacks,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 14 851–14 860.

[5] Z. Yu, Y. Qin, X. Li, Z. Wang, C. Zhao, Z. Lei, and G. Zhao, “Multi-modal face anti-spoofing based on central difference networks,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2020, pp. 650–651.

[6] A. George and S. Marcel, “Cross modal focal loss for rgbd face antispoofing,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 7882–7891.

[7] P. Deng, C. Ge, X. Qiao, H. Wei, and Y. Sun, “Attention-aware dualstream network for multimodal face anti-spoofing,” IEEE Transactions on Information Forensics and Security, vol. 18, pp. 4258–4271, 2023.

[8] W. Wang, F. Wen, H. Zheng, R. Ying, and P. Liu, “Conv-mlp: A convolution and mlp mixed model for multimodal face anti-spoofing,” IEEE Transactions on Information Forensics and Security, vol. 17, pp. 2284–2297, 2022.

[9] S. Han, R. Cai, Y. Cui, Z. Yu, Y. Hu, and A. Kot, “Hyperbolic face anti-spoofing,” arXiv preprint arXiv:2308.09107, 2023.

[10] A. Liu, Z. Tan, J. Wan, Y. Liang, Z. Lei, G. Guo, and S. Z. Li, “Face antispoofing via adversarial cross-modality translation,” IEEE Transactions on Information Forensics and Security, vol. 16, pp. 2759–2772, 2021.

[11] A. Liu, Z. Tan, Z. Yu, C. Zhao, J. Wan, Y. Liang, Z. Lei, D. Zhang, S. Z. Li, and G. Guo, “Fm-vit: Flexible modal vision transformers for face anti-spoofing,” IEEE Transactions on Information Forensics and Security, vol. 18, pp. 4775–4786, 2023.

[12] A. Liu, H. Ma, J. Zheng, H. Yuan, X. Yu, Y. Liang, S. Escalera, J. Wan, and Z. Lei, “Fm-clip: Flexible modal clip for face anti-spoofing,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 8228–8237.

[13] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[14] H. Fang, A. Liu, N. Jiang, Q. Lu, G. Zhao, and J. Wan, “Vl-fas: Domain generalization via vision-language model for face anti-spoofing,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 4770–4774.

[15] A. Liu, S. Xue, J. Gan, J. Wan, Y. Liang, J. Deng, S. Escalera, and Z. Lei, “Cfpl-fas: Class free prompt learning for generalizable face antispoofing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 222–232.

[16] Y. Shi, Y. Gao, Y. Lai, H. Wang, J. Feng, L. He, J. Wan, C. Chen, Z. Yu, and X. Cao, “Shield: An evaluation benchmark for face spoofing and forgery detection with multimodal large language models,” Visual Intelligence, vol. 3, no. 1, p. 9, 2025.

[17] G. Zhang, K. Wang, H. Yue, A. Liu, G. Zhang, K. Yao, E. Ding, and J. Wang, “Interpretable face anti-spoofing: Enhancing generalization with multimodal large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 9, 2025, pp. 9896– 9904.

[18] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” Communications of the ACM, vol. 63, no. 11, pp. 139–144, 2020.

[19] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in International Conference on Learning Representations, 2014.

[20] A. Van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals, A. Graves et al., “Conditional image generation with pixelcnn decoders,” Advances in neural information processing systems, vol. 29, 2016.

[21] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings

of the IEEE international conference on computer vision, 2017, pp. 2223–2232.

[22] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[23] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in International Conference on Learning Representations, 2021.

[24] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[25] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3836–3847.

[26] Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, “Restoreformer: High-quality blind face restoration from undegraded key-value pairs,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 17 512–17 521.

[27] X. Qiu, C. Han, Z. Zhang, B. Li, T. Guo, and X. Nie, “Diffbfr: Bootstrapping diffusion model for blind face restoration,” in Proceedings of the 31st ACM international conference on multimedia, 2023, pp. 7785–7795.

[28] D. Liu, X. Wang, C. Peng, N. Wang, R. Hu, and X. Gao, “Adv-diffusion: imperceptible adversarial face identity attack via latent diffusion model,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 4, 2024, pp. 3585–3593.

[29] T. Brooks, A. Holynski, and A. A. Efros, “Instructpix2pix: Learning to follow image editing instructions,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 18 392–18 402.

[30] T. Qi, S. Fang, Y. Wu, H. Xie, J. Liu, L. Chen, Q. He, and Y. Zhang, “Deadiff: An efficient stylization diffusion model with disentangled representations,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 8693–8702.

[31] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. March, and V. Lempitsky, “Domain-adversarial training of neural networks,” Journal of machine learning research, vol. 17, no. 59, pp. 1–35, 2016.

[32] Z. Yu, R. Cai, Y. Cui, A. Liu, and C. Chen, “Visual prompt flexiblemodal face anti-spoofing,” IEEE Transactions on Dependable and Secure Computing, 2024.

[33] A. George and S. Marcel, “Deep pixel-wise binary supervision for face presentation attack detection,” in 2019 international conference on biometrics (ICB). IEEE, 2019, pp. 1–8.

[34] ——, “Learning one class representations for face presentation attack detection using multi-channel convolutional neural networks,” IEEE Transactions on Information Forensics and Security, vol. 16, pp. 361– 375, 2020.

[35] Y. Zhang, X. Zhu, A. Liu, X. Lin, J. Wan, J. Zhang, and Z. Lei, “Cpl-clip: Compound prompt learning for flexible-modal face anti-spoofing,” in 2024 IEEE International Joint Conference on Biometrics (IJCB). IEEE, 2024, pp. 1–10.

[36] C.-H. Liao, W.-C. Chen, H.-T. Liu, Y.-R. Yeh, M.-C. Hu, and C.-S. Chen, “Domain invariant vision transformer learning for face anti-spoofing,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2023, pp. 6098–6107.

[37] K. Zhou, J. Yang, C. C. Loy, and Z. Liu, “Conditional prompt learning for vision-language models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16 816– 16 825.

[38] ——, “Learning to prompt for vision-language models,” International Journal of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022.

[39] A. Liu, C. Zhao, Z. Yu, J. Wan, A. Su, X. Liu, Z. Tan, S. Escalera, J. Xing, Y. Liang et al., “Contrastive context-aware learning for 3d highfidelity mask face presentation attack detection,” IEEE transactions on information forensics and security, vol. 17, pp. 2497–2507, 2022.

[40] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim, “Visual prompt tuning,” in European conference on computer vision. Springer, 2022, pp. 709–727.

[41] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” International Journal of Computer Vision, vol. 128, no. 2, pp. 336–359, 2020.