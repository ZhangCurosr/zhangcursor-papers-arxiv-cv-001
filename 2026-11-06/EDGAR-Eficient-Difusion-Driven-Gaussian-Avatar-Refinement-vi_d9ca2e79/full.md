# EDGAR : Eficient Difusion-Driven Gaussian Avatar Refinement via Synthetic Failure

KWAN YUN, KAIST, Visual Media Lab, South Korea MINHYEOK BANG, KAIST, Visual Media Lab, South Korea COLETTE SUHJUNG KOO, KAIST, Visual Media Lab, South Korea JUNYONG NOH, KAIST, Visual Media Lab, South Korea

![](images/c2c046647bcd92956f707f6728bcac503be36eb09b8fc76b005fe13be23552f4.jpg)  
Fig. 1. From diverse failure cases of Gaussian avatars, EDGAR learns to refine arbitrary avatars trained with a video or an image, acting as both an artifact remover and a quality enhancer.

Recent advances in Gaussian splatting have enabled eficient and photoreal istic head avatar reconstruction, yet monocular video–based or image-based methods still sufer from under-constrained geometry and texture, leading to artifacts such as blurring, stretching, and temporal instability. We present EDGAR (Eficient Difusion-Driven Gaussian Avatar Refinement), a con ditional difusion framework designed to enhance the quality of Gaussian avatars by correcting artifacts while preserving identity fidelity. To train EDGAR, we construct EDGAR-SET, a hybrid triplet dataset composed of real-captured and synthetic-rendered samples, which simulates realistic Gaussian avatar failure cases by pairing degraded renderings with ground truth and reference images. EDGAR refines degraded frames through a fusion block that injects identity features extracted by the self-supervised Sapiens model into a one-step difusion backbone, enabling efective artifact removal without sacrificing eficiency. Our EDGAR model incorporates a Mixture-of-Experts (MoE) architecture to address diverse degradation types and severity levels. The refined multi-view renderings are subsequently used as pseudo-targets to optimize a single underlying Gaussian avatar, so that the final avatar can be rendered without applying EDGAR to every output frame. Experiments on three diferent Gaussian avatar reconstruction methods,

each evaluated under three diferent settings, show improvements in both perceptual and reconstruction measures over the raw outputs and an SDEditstyle refinement baseline. In addition, cross-view evaluation demonstrates improved calibrated consistency, while tests under large yaw and open mouth expressions verify robustness under extreme settings. These results collectively indicate a useful operating regime for recoverable appearance artifacts. The code is publicly available at EDGAR.

CCS Concepts: • Computing methodologies → Image processing; Animation; Computer vision.

Additional Key Words and Phrases: Gaussian Avatar, Image Refinement, Difusion model

## 1 Introduction

Personalized head avatars are now integral to social gaming, VR telepresence, and mixed-reality interfaces, where users demand both photorealism and low latency. Mesh-based reconstruction pipelines [Feng et al. 2021; Lattas et al. 2020] ofer the advantage of compact representations and practical usability thanks to their fixed mesh structure. However, reliance on meshes makes it dificult to capture high-frequency details and to scale to high-resolution geometry, particularly in regions such as hair. In contrast, recent eforts built on volumetric neural-radiance fields (NeRF) [Mildenhall et al. 2021] have achieved impressive view synthesis results. Unfortunately, these methods require heavy per-scene optimization and remain slow for interactive use [Kirschstein et al. 2023; Zheng et al. 2022a; Zielonka et al. 2023]. Similarly, GAN-driven work achieved striking 3D-aware imagery [An et al. 2023; Li et al. 2023; Ye et al. 2023], yet encoded geometry only implicitly, limiting direct control over facial animation. These achievements have established the visual-quality bar while exposing the need for representations that are simultaneously lightweight, editable, and real-time ready.

The field has since pivoted to rasterizable 3D Gaussian splats [Kerbl et al. 2023], which replace dense ray-marching with a set of anisotropic blobs that are both GPU-friendly and easily deformed. Multi-view pipelines [Giebenhain et al. 2024; Hu et al. 2024; Xu et al. 2024] reach fast render times while faithfully tracking the face and producing realistically synthesized expressions. Monocularvideo [Chen et al. 2025, 2024; Qian et al. 2024b; Zhang et al. 2025] methods lower the capture barrier for further usability. Together, these Gaussian-based methods retain the photorealism pioneered by NeRF systems while unlocking interactive performance and casual acquisition, bringing personalized avatars within reach of everyday applications.

Nevertheless, compressing capture into a monocular video still leaves the reconstruction problem severely under-constrained. Monocular pipelines must infer geometry and texture in unobserved regions, which often causes artifacts—such as texture stretching, floating hair, or depth ambiguities—when rendering side profiles, wide-open mouths, or raised eyebrows. These issues are largely absent in avatars reconstructed from dense camera arrays, whereas limited observations in monocular settings leave unobserved regions under-constrained. While a few recent methods utilize difusionbased image generation to fill this gap by adding and removing noise using pretrained generative models [Meng et al. 2021] to refine the output [Wu et al. 2025a, 2024a], these approaches inherently involve guessing appearance from a limited prior, thus failing to reliably preserve identity details, especially in unseen poses and expressions.

To close this quality gap, we address avatar enhancement through a difusion-prior-based refinement framework, which we term Efficient Difusion-Driven Gaussian Avatar Refinement (EDGAR). Difusion-based refinement is particularly well suited to this problem because strong generative priors can recover plausible highfrequency details while preserving overall visual realism under rendering degradation. The core idea is straightforward: given a rendered frame from a noisy or blurry Gaussian avatar and a reference image of the subject, EDGAR learns to refine the output into a sharper and more faithful rendering. This enables us to take an arbitrary Gaussian avatar—even when trained under suboptimal monocular conditions—and upgrade it to higher visual quality. However, learning this mapping is non-trivial: there is no existing dataset of "imperfect avatar vs. ground-truth render" pairs, and the appearance of a degraded avatar is not uniquely defined.

To address this, we construct EDGAR-SET, a hybrid dataset that combines real human data with synthetically generated samples to increase diversity in both identity and viewpoints. The dataset is organized as triplets: (1) rendered images from degraded avatars, (2) corresponding ground-truth images, and (3) reference identity images, as shown on the left of Figure 1. The synthetic component enables controlled simulation of realistic failure modes (e.g., sparseview training, unseen poses, and expression variations), while the human dataset grounds the model in real-world appearance distri butions. This combination ensures broad coverage over identity, pose, and rendering artifacts.

EDGAR learns a conditional refinement function by injecting reference features [Khirodkar et al. 2024] into a single-step difusion backbone [Sauer et al. 2024]. To robustly handle diverse failure modes, we further adopt a Mixture-of-Experts (MoE) architecture, allowing the model to specialize across diferent types and severities of degradation. In addition, we use an expression consistency loss to encourage preservation of facial dynamics and subtle muscle movements during refinement. Our results demonstrate that EDGAR improves overall rendering fidelity and perceptual quality.

Our contributions are summarized as follows:

• We construct EDGAR-SET, a hybrid collection of paired Gaussian-avatar failures, ground-truth targets, and identity references designed around recoverable head-avatar artifacts.

• We introduce EDGAR, an eficient difusion-driven refinement framework that learns to correct rendered artifacts of animatable Gaussian head avatars and distills the refined 2D targets back into the underlying 3D avatar.

• We develop subject-reference conditioning and expressionaware supervision to preserve identity and facial motion during refinement, together with a lightweight MoE module that handles diverse degradation patterns without increasing the final avatar rendering cost.

## 2 Related Work

## 2.1 Head Reconstruction

Classical head reconstruction leverages priors from 3D morphable models (3DMMs) [Blanz and Vetter 1999; Cao et al. 2013; Li et al. 2017] to recover identity and expression from images or videos. Unlike video based retargeting [Guo et al. 2024; Zhao et al. 2025], 3D reconstruction provides explicit geometric and appearance representations, enabling more controllable and physically grounded avatar generation. Building on 3DMM priors, many methods tackle monocular reconstruction and tracking with varying levels of geometric detail and expression, either through classical fitting [Blanz and Vetter 2003; Cao et al. 2014; Romdhani and Vetter 2005; Thies et al. 2016] or, more recently, learning-based methods [Daněček et al. 2022; Deng et al. 2019; Zhu et al. 2016]. While these methods are eficient and provide explicit control through constructed blendshape parameters, their expressiveness in both dynamic geometry and subtle details is limited due to the constraints of the constructed space of defined 3DMM.

To enhance quality beyond the linear expressiveness of 3DMMs, recent approaches adopt implicit neural representations such as SDF [Park et al. 2019], OccupancyNet [Mescheder et al. 2019], NeRF [Mildenhall et al. 2021], and Instant-NGP [Müller et al. 2022]. Thanks to their flexibility, these representations enable efective reconstruction of fine geometric details such as hair wisps and beards [Giebenhain et al. 2023; Zheng et al. 2022b,a, 2023; Zielonka et al. 2023]. These advances have substantially improved the realism of head avatars, but often at the cost of high computational complexity or slow rendering speed, which limits their practicality in interactive applications.

## 2.2 Gaussian Head Avatars

Recently, 3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] has emerged as an eficient representation for photorealistic scene rendering. Instead of representing geometry with an explicit mesh or an implicit neural field, 3DGS models a scene as a collection of learnable 3D Gaussians, each associated with attributes such as position, scale, rotation, opacity, and color. These Gaussians are rasterized efficiently, enabling high-quality rendering at interactive frame rates. When adapted to dynamic human faces, this representation gives rise to Gaussian head avatars, which model facial appearance and deformation using a set of animatable Gaussians. Compared with prior implicit representations, Gaussian head avatars ofer a favorable balance between visual fidelity, rendering eficiency, and editability.

Based on the number of input views, Gaussian avatar reconstruction methods can be broadly divided into two categories. Multiview methods based on calibrated camera arrays [Aneja et al. 2025; Giebenhain et al. 2024; Qian et al. 2024a; Sun et al. 2025; Teotia et al. 2024; Xu et al. 2024] achieve highly photorealistic reconstructions across diverse viewpoints and expressions, benefiting from dense observations of each subject. However, collecting such data requires specialized multi-camera capture systems, which greatly limits scalability and accessibility. To reduce this burden, recent works have explored monocular video-based methods [Chen et al. 2024; Ma et al. 2024; Moon et al. 2025; Shao et al. 2024; Song et al. 2025; Teotia et al. 2025; Wei et al. 2025a,b; Xiang et al. 2024; Zhang et al. 2025] and few-image-based methods [Chu and Harada 2024; He et al. 2025; Kirschstein et al. 2025; Taubner et al. 2025b,a; Zheng et al. 2025]. These methods incorporate additional priors, such as FLAME fitting [Li et al. 2017] or semantic feature extraction [Oquab et al. 2023], to compensate for the lack of observations during training.

Nevertheless, Gaussian head avatars reconstructed from limited inputs remain fundamentally under-constrained. As a result, they often exhibit artifacts such as blurry textures, floating Gaussians, and unstable geometry in regions that are weakly observed during training. In contrast, multi-view Gaussian avatars typically provide cleaner and more stable renderings because their geometry and appearance are better constrained by dense observations. Our work is motivated by this gap: we aim to learn how degraded Gaussian avatar renderings should be refined using supervision derived from high-quality data, and to apply this knowledge to improve avatars reconstructed under sparse-view or monocular conditions.

## 2.3 Generative Prior for Refinement

Advances in generative models [Goodfellow et al. 2014; Ho et al. 2020] have demonstrated not only impressive generation quality but also their utility for image refinement [Saharia et al. 2022a,b; Yang et al. 2021] and editing [Lugmayr et al. 2022; Meng et al. 2021; Rom bach et al. 2022]. This capability stems from expressive latent priors learned from large-scale data, which capture rich distributions ofnatural images. Such priors enable models both to generate photoreal istic samples and to guide the refinement of degraded or incomplete inputs toward plausible, high-quality outputs. While most refinement methods have been developed for super-resolution [Wang et al. 2024; Wu et al. 2024b] and restoration [Luo et al. 2024; Wu et al. 2025b; Zhang et al. 2024] in general scenes or images, only a few methods have been applied to human faces or avatars.

GAN-based methods typically encode images into the latent space of a pretrained model and then pass them through the generator again to produce modified images in 2D [Abdal et al. 2019; Richardson et al. 2021a,b; Yun et al. 2024] or 3D [Jiang et al. 2022; Sun et al. 2022; Yun et al. 2025]. These methods require images to be cropped and aligned. More closely related to our work, recent works [Lee et al. 2025; Wu et al. 2025a, 2024a] utilize an SDEdit-like [Meng et al. 2021] approach that adds noise and removes it to refine rendered avatars. While SDEdit-style refinement is efective for blind restoration, the network lacks explicit knowledge of how the avatar should look, and the stochastic noise-adding procedure introduces randomness. Consequently, the denoising step may produce inconsistent results across frames, leading to temporal jittering. Difix3D+ [Wu et al. 2025b] more directly couples a single-step difusion refiner with optimization of static NeRF and 3DGS representations. EDGAR applies a related refinement strategy to animatable Gaussian head avatars, using paired avatar failures, a subject reference, and expression supervision to preserve identity and facial motion. Larger image-editing foundation models may ofer higher single-frame quality, but avatar refinement also requires identity, expression, and cross-view consistency. EDGAR addresses these requirements through avatar-specific supervision and shared 3D optimization, without per-frame difusion inference.

## 3 Method

Our goal is to design a model that refines rendered Gaussian avatars trained under diverse conditions, where limited observations often lead to visual artifacts. Specifically, we formulate avatar enhancement as a difusion-prior-based refinement task: given a degraded rendering and a subject reference image, the model learns to produce a sharper and more faithful output. To enable this, we construct EDGAR-SET (Section 3.1), a triplet dataset that simulates realistic failure modes and provides paired ground-truth supervision. Build ing on this dataset, we train EDGAR, a difusion-based refinement network conditioned on reference identity features, to correct artifacts while preserving subject fidelity (Section 3.2). Finally, we optimize the underlying Gaussian avatar itself using EDGAR-refined renderings as supervision, allowing the improvements to be ab sorbed into the 3D representation while preserving the eficiency of the original avatar model (Section 3.4).

## 3.1 Synthesizing EDGAR-SET

Training EDGAR requires paired data that captures how degraded Gaussian avatar renderings should be refined. To this end, we construct a triplet dataset, EDGAR-SET, composed of synthetic failure images $I _ { f } ,$ ground-truth images $I _ { g t }$ , and reference images � . The failure images $I _ { f }$ simulate common artifacts observed in Gaussian avatars, while $\dot { I _ { g t } }$ provides the corresponding ground-truth image. The reference image $I _ { r }$ is drawn from a ground-truth rendering of the same subject under a diferent view and/or expression, supplying appearance cues that guide refinement. At inference time, �<sub>�</sub> corresponds to a frame from the monocular video used to train the avatar. Given a degraded input $I _ { f }$ and its reference $I _ { r } ,$ EDGAR learns to generate a refined output $\hat { I } _ { f }$ that faithfully approximates the ground truth $I _ { g t } .$

![](images/b2a448b0ecffa5c3ac364d2ea7d9796acfb6febb0edb551cde22a05d83bc7c12.jpg)

![](images/44dcadb7f347f32d8b619986adcb57cd9be6ee060a90d68ecb8f1ed41d6094ab.jpg)

![](images/084db3ea1929f5777199b97e2668555c8f737acf40cec5eb7ecbd5071a1f4eff.jpg)  
Rendered Avatar (If)

![](images/389bf0aa3add0d0486ed7cd274c12cf5454bd0ac8fa9ac7c6436e0407329262d.jpg)  
Reference Image (Ir)

![](images/d871a3b092dab311be16ea5490b35cb34e55774101d5b1b9f91ddc884c51215d.jpg)  
Ground-Truth (1gt)

![](images/344f3e1a6c0e9ce6f44a623edb97ad73715644b3a7bfbf4ef67c6bc232d9ca3d.jpg)  
Rendered Avatar (If)

Fig. 2. Examples from EDGAR-SET and discarded samples. We filter out cases where the geometry or expression is misaligned, making the recovery target ambiguous. Samples are removed when the combination of $I _ { f }$ and �<sub>�</sub> from novel views or expressions cannot reliably reproduce $I _ { g t }$  
![](images/ff8b924a1aa4348f58fa1acce1114e1ac5ce579e21f8977952cb549a50d13443.jpg)

![](images/a4e7f0bc1176b14ab2c1b81a78c3f58bab5aa65bbbdee2cb318007a0d8b2420e.jpg)

![](images/8cf68252cc35e6e0ebd3830d129a386ccd9b8c7ff035fa1d8ca5485a29d3cb59.jpg)  
Fig. 3. Configuration of EDGAR-SET. The dataset is constructed under diverse conditions to capture a wide range of Gaussian avatar failure cases. The view-count groups (one, two to four, and at least five) are mutually exclusive and account for 23%, 51%, and 26% of retained examples, respectively.

To construct EDGAR-SET, we define three key requirements to ensure efective training for avatar refinement. First, the dataset must contain diverse identities to support generalization across subjects. Second, it should include challenging conditions such as novel viewpoints, which are primary sources of artifacts in Gaussian avatar rendering. Third, each triplet must include a reference image that difers from the ground-truth target while still providing meaningful identity and appearance cues under varying poses or expressions. To satisfy these criteria, we design EDGAR-SET as a hybrid dataset that combines real-captured data with large-scale synthetic augmentation, enabling both realism and broad coverage of edge cases.

Following these guidelines, we construct EDGAR-SET using both real and synthetic data sources. For real data, we leverage the cali brated multi-view corpus NeRSemble [Kirschstein et al. 2023] and build avatars using the RGBAvatar pipeline [Li et al. 2025], which provides high-fidelity reconstructions grounded in real-world appearance. For synthetic data, we incorporate the CAFCA [Buehler et al. 2024] corpus, which significantly expands identity diversity and introduces extreme view directions and expressions that are dificult to capture in real-world settings. Using CAFCA, we con struct avatars based on the original 3D Gaussian Splatting formulation [Kerbl et al. 2023], enabling the systematic simulation of a wide range of degradation patterns. The degradation patterns include floating Gaussian artifacts, blurry or over-smoothed regions, misaligned geometry, and view-dependent inconsistencies. The synthetic split primarily exhibits moderate failures due to a larger number of input views, while the real-captured split introduces more extreme failure modes, such as severe geometric distortions caused by limited-view reconstruction. Across both datasets, we vary training conditions—such as the number of input views—to induce realistic failure cases.

After constructing avatars, we obtain degraded renderings $I _ { f }$ alongside their corresponding ground-truth images $I _ { g t }$ , and sample a distinct ground-truth frame as the reference $I _ { r } .$ . This unified pipeline yields diverse triplets that cover both realistic and extreme scenarios, enabling EDGAR to learn robust refinement across a broad spectrum of avatar artifacts. In this way, our triplet formulation extends the utility of the original datasets beyond static ground-truth collections. Specifically, our triplet formulation converts the original datasets into supervised training data for refinement by simulating failures under diverse scenarios and incorporating reference images. In contrast, the original datasets (NeRSemble and CAFCA) primarily provide ground-truth observations without explicit degradation modeling.

From this raw triplet set, we filter out images with severe artifacts where $I _ { f }$ diverges significantly from $I _ { g t }$ , making them unsuitable for training. We compute LPIPS scores as a similarity measure and discard cases with consistently low similarity. As shown in Figure 2, triplets in which expression fidelity is lost or artifacts are irreparable are removed, while cases with degraded but recoverable quality—such as Gaussian noise artifacts with consistent expression and head pose—are retained. After this filtering step, the final EDGAR-SET, consisting of 400K high-resolution images across thousands of identities, is constructed. Configuration of EDGAR-SET is summarized in Figure 3. EDGAR-SET adheres to the guidelines established for training EDGAR, including diverse identities, expressions, and viewpoints, as well as the inclusion of reference images. The resulting training regime emphasizes recoverable appearance artifacts and moderate reconstruction errors, whereas severe structural failures are not represented reliably by the available image and reference pairs.

## 3.2 EDGAR Model

EDGAR is a conditional difusion model that refines degraded inputs $I _ { f }$ using a reference image $I _ { r } .$ . For eficient inference, it is built upon the one-step difusion model SD-Turbo [Sauer et al. 2024], a stepdistilled version of Stable Difusion [Rombach et al. 2022], following prior work on eficient image refinement [Parmar et al. 2024; Wu et al. 2025b; Yue et al. 2024]. Because our objective is to adapt a pretrained difusion model to a Gaussian avatar refinement network, we freeze the residual blocks and cross-attention layers of SD-Turbo. The residual blocks are responsible for timestep injection, while the cross-attention layers handle prompt conditioning—neither of which we aim to modify. On the other hand, we update the selfattention layers, which govern pixel-to-pixel relationships, so that the network learns how to refine $I _ { f } .$ . In addition, we pass the features from the encoder to a LoRA-added [Hu et al. 2022] decoder with a zero-initialized residual layer so that the network better understands $I _ { f }$ , following Parmar et al. [2024].

To incorporate reference information, we implement a fusion block. Specifically, we pass $I _ { r }$ through the pretrained Sapiens model [Khirodkar et al. 2024], a large self-supervised transformer known to capture rich human representations and perform well across diverse downstream tasks. We use Sapiens, denoted as

$S ,$ as a feature extractor for reference identity. As illustrated in Figure 4, its features are projected and injected into the multihead cross-attention layers to serve as keys and values, supplying identity-specific context that guides refinement, while the difusion features � from the previous layer serve as queries. This can be expressed as:

$$
\mathrm { F u s e } \left( F , S _ { r } \right) = \mathrm { C o n v } _ { \mathrm { z e r o } } \left( \mathrm { s o f t m a x } \left( \frac { F \cdot \mathrm { P r o j } ( S _ { r } ) ^ { \top } } { \sqrt { d } } \right) \mathrm { P r o j } ( S _ { r } ) \right) + F ,\tag{1}
$$

where $S _ { r } = S ( I _ { r } )$ denotes the Sapiens features from $I _ { r } ,$ Proj is the projection layer, and $\mathrm { C o n v } _ { \mathrm { z e r o } }$ is the zero-initialized convolutional layer. Note that because all added layers pass or are initialized with zeros including bias terms, in the beginning of the training, the model behaves exactly the same as the original SD-Turbo model. This initialization efectively preserves the performance of the pretrained network while allowing it to smoothly converge to the refining network as training progresses.

To handle the wide range of degradation patterns encountered in Gaussian avatar refinement, we further adopt an MoE architecture. In practice, the input may exhibit substantially varying conditions, ranging from pronounced artifacts such as floating blobs and blurry Gaussians, to nearly clean reconstructions that require only minor corrections. The dificulty of refinement also varies depending on the reference condition, for example, whether the reference image is captured from a nearby or distant viewpoint. A single expert is often insuficient to model such diverse cases. We therefore introduce multiple experts that specialize in diferent refinement behaviors. Specifically, after the self-attention and cross-attention layers, a lightweight router predicts expert assignment scores from the intermediate feature and dispatches the feature to one selected expert for further processing. The entire routing mechanism is trained jointly with the refinement network in a self-supervised manner.

## 3.3 Objectives

To train the EDGAR model, we adopt image reconstruction losses for refined image $\hat { I } _ { f }$ to follow the ground-truth image $I _ { g t }$ conditioned on reference $I _ { r } .$ Specifically, we combine perceptual, pixel-level, and style consistency terms. The perceptual loss is measured using LPIPS [Zhang et al. 2018], which encourages perceptual similarity between the refined output $\hat { I } _ { f }$ and the ground-truth $I _ { g t }$ . The pixel loss is given by the $\ell _ { 2 }$ distance, promoting accurate low-level reconstruction. In addition, we incorporate a Gram matrix loss computed from VGG features [Gatys et al. 2016], which preserves higher-order correlations and encourages realistic texture synthesis. The total training objective is defined as follows:

$$
\begin{array} { r } { \mathcal { L } _ { i m g } = \Vert \hat { I } _ { f } - I _ { g t } \Vert _ { 2 } ^ { 2 } + \lambda _ { \mathrm { L P I P S } } \mathcal { L } _ { \mathrm { L P I P S } } ( \hat { I } _ { f } , I _ { g t } ) + \lambda _ { \mathrm { G r a m } } \mathcal { L } _ { \mathrm { G r a m } } ( \hat { I } _ { f } , I _ { g t } ) , } \end{array}\tag{2}
$$

where $\lambda _ { \mathrm { L P I P S } }$ and $\lambda _ { \mathrm { G r a m } }$ control the relative weighting of each loss term.

In addition to the image-based loss $\begin{array} { r } { \mathcal { L } _ { i m g } . } \end{array}$ we introduce an expression consistency loss to better preserve subtle facial details such as the lips, eyes, and other fine-grained expression cues, which are often not suficiently constrained by image reconstruction losses alone. To this end, we leverage the final-layer feature of POSTER V2 [Mao et al. 2025], a vision transformer-based facial expression model that jointly exploits landmark and image features for robust expression understanding. Rather than using its output logits, we use the pretrained feature representation as a perceptual expression descriptor, as intermediate features provide richer and smoother supervision for expression matching. Let $\phi ( \cdot )$ denote the frozen POSTER V2 feature extractor. We then define the expression loss as

![](images/17ae6060ea93a1650650e1f3b9ce9d6dc6d2f715e70940f6b96cde666084e518.jpg)  
Fig. 4. Network architecture of EDGAR. EDGAR learns to refine rendered images of Gaussian avatars using EDGAR-SET. The fuse block in the Upblock o Difusion UNet is simplified.

$$
\mathcal { L } _ { \mathrm { e x p } } = \Vert \phi ( \hat { I } _ { f } ) - \phi ( I _ { g t } ) \Vert _ { 2 } ^ { 2 } .\tag{3}
$$

This loss encourages the refined output to preserve the target facial expression more faithfully at a semantic feature level.

Finally, to prevent expert collapse, where only a small subset of experts are consistently selected, we employ a load-balancing loss following prior MoE formulations. Let $\mathit { p } _ { i , e }$ denote the router probability of assigning sample � to expert $e ,$ and let $z _ { i } = \mathrm { a r g } \operatorname* { m a x } _ { e } p _ { i , e }$ be the selected expert index. We define the average routing importance and expert load as

$$
\bar { p } _ { e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bar { p } _ { i , e } , \qquad \bar { l } _ { e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } ( z _ { i } = e ) ,\tag{4}
$$

where � is the batch size and $\mathbb { I } ( \cdot )$ is the indicator function. The load-balancing loss is then written as

$$
\mathcal { L } _ { \mathrm { l b } } = E \sum _ { e = 1 } ^ { E } \bar { p } _ { e } \bar { l } _ { e } ,\tag{5}
$$

where � is the number of experts. This objective encourages both the router probabilities and the actual expert assignments to remain balanced across experts, improving specialization and stabilizing training.

![](images/04ab7abc3a219c0003e32d2f87351ba9cb8b26538e4d8c683f8c28f790d3c3e6.jpg)  
Fig. 5. Avatar refinement process using EDGAR. Multi-view renderings of the source Gaussian avatar are refined by EDGAR and used as supervision targets for Gaussian avatar refinement.

In summary, the overall training objective for EDGAR is defined as

$$
\mathcal { L } _ { E D G A R } = \mathcal { L } _ { i m g } + \lambda _ { \mathrm { e x p } } \mathcal { L } _ { \mathrm { e x p } } + \lambda _ { \mathrm { l b } } \mathcal { L } _ { \mathrm { l b } } ,\tag{6}
$$

where $\lambda _ { \mathrm { e x p } }$ and $\lambda _ { \mathrm { l b } }$ control the contributions of the expression consistency loss and the load-balancing loss, respectively.

## 3.4 Avatar Refinement

After training EDGAR, we use it to further refine the underlying Gaussian avatar itself. Given a Gaussian avatar as input, we first render it from multiple views and then apply EDGAR to each rendered image to obtain refined images $\hat { I } _ { f } .$ These refined images are subsequently used as supervision signals to optimize the Gaussian avatar parameters with an $\ell _ { 1 }$ loss and an SSIM loss. Unlike prior refinement strategies that split or add Gaussians during optimization, we do not perform such operations, as they increase the number of Gaussians and lead to slower rendering. Our goal is to improve reconstruction quality while preserving the eficiency of the original model. Dur ing the refinement stage, a single reference image �<sub>�</sub> is randomly selected and used throughout the process. This refinement process is shown in Figure 5.

Importantly, EDGAR is not used only as a post-processing module after rendering. Instead, its refined outputs are further used as optimization targets for the Gaussian avatar. In this way, the improvements introduced by EDGAR are absorbed into the 3D representation itself, allowing the refined avatar to produce higherquality renderings directly. Because the refinement is performed by optimizing a single shared 3D Gaussian representation across multiview rendered images, the resulting updates remain consistent in 3D and improve cross-view coherence.

Formally, letting $_ \mathrm { c } { } _ { V }$ denote a set of sampled views, $R ( \mathcal { G } , \upsilon )$ the rendering of Gaussian avatar $\mathcal { G }$ from view �, and $\hat { I } _ { f } ^ { ( v ) }$ the EDGARrefined result for that view, the avatar refinement objective is defined as follows:

$$
\mathcal { L } _ { \mathrm { a v a t a r } } = \sum _ { v \in \mathcal { V } } \left[ \lambda _ { \mathrm { L } 1 } \lVert R ( \mathcal { G } , v ) - \hat { I } _ { f } ^ { ( v ) } \rVert _ { 1 } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } ( R ( \mathcal { G } , v ) , \hat { I } _ { f } ^ { ( v ) } ) \right] ,\tag{7}
$$

where $\lambda _ { \mathrm { L 1 } }$ and �<sub>SSIM</sub> control the weights of the L1 and SSIM terms.

## 4 Experiments

## 4.1 Implementation Details

We implemented and ran our data-construction pipeline using RG-BAvatar and 3DGS on a computer equipped with a single NVIDIA RTX 3090 GPU. For the real-capture split of EDGAR-SET, we used RGBAvatar with the NeRSemble [Kirschstein et al. 2023] dataset, varying the number of viewpoints, and performed selfreenactment to construct EDGAR-SET of 99 identities. Using 3DGS with CAFCA [Buehler et al. 2024], we also constructed a synthetic split of EDGAR-SET of 1,500 identities with varying expressions, lighting conditions, and camera positions. While the real-capture split was constructed using 1 to 4 viewpoints during training, the synthetic split was constructed using only multi-view settings, specifically at least with 5 viewpoints, because 3DGS cannot reconstruct an avatar from a single viewpoint. All images were resized to 512×512 for both training and testing. During EDGAR-SET construction, LPIPS was used to compare the source and target images and approximately half of the pairs were filtered out.

We trained EDGAR on EDGAR-SET for 40,000 steps with a batch size of 4 and gradient accumulation over 2 iterations per step, which required approximately 40 hours. The learning rate was linearly increased up to $2 \times 1 0 ^ { - 5 }$ for 1,000 steps and constant afterward. The weights $\lambda _ { L P I P S }$ and $\lambda _ { G r a m }$ used in Eq. (2) were both set to 1.0, while the weights $\lambda _ { e x p }$ and $\lambda _ { l b }$ used in Eq. (6) were set to 8.0 and 0.001. We set the timestep to $\tau = 2 5 0$ and used the prompt “High resolution portrait.” for both training and inference for text condition. A total of six expert MLPs were used in the MoE architecture. For the Gaussian avatar refinement process, we used the output Gaussians of each baseline model as input to EDGAR, rendered 50 views, and optimized the avatar for 500 steps. The weights $\lambda _ { \mathrm { L 1 } }$ and �<sub>SSIM</sub> used in Eq. (7) were set to 0.6 and 0.4, respectively. EDGAR was trained and evaluated on a single NVIDIA A6000 GPU, while the Gaussian avatar refinement stage was run on a single NVIDIA RTX 3090 GPU.

Computational cost. Table 8 reports the complete per-avatar refinement time, covering the full process from loading the avatar and EDGAR model to generating pseudo-targets, optimizing for 500 steps, and saving the checkpoint. Base-avatar optimization times are measured excluding the preprocessing and CAP4D’s upstream view synthesis. Final-avatar rendering throughput is benchmarked separately.

## 4.2 Baseline Comparison

EDGAR is a general Gaussian avatar refinement framework and can be applied to arbitrary trained or optimized Gaussian avatar reconstruction methods. To evaluate its efectiveness, we conducted experiments on three diferent baselines: CAP4D [Taubner et al. 2025a], which is a single- to multi-image-based reconstruction method, GaussianAvatars [Qian et al. 2024a], which enables high-quality avatar reconstruction from multi-view videos, and RGBAvatar [Li et al. 2025], which is a single-video-based reconstruction method. These three methods represent complementary design choices, ranging from single-image-based and single-video-based to multi-view settings, providing a comprehensive benchmark for assessing EDGAR’s refinement capability across diferent reconstruction settings. For a comprehensive evaluation, because each method supports reconstruction from a diferent number of inputs, we conducted comparisons using avatars reconstructed with varying numbers of inputs. Specifically, we used 1, 4, and 8 images for CAP4D, 1, 4, and 8 videos for RGBAvatar, and 8, 10, and 12 videos for GaussianAvatars, following the settings used in each method. Note that RGBAvatar was used during data construction and is included here as an auxiliary test baseline. For testing, we used ten unseen identities from NeRSemble for self-reenactment and additionally six diferent identities from the INSTA [Zielonka et al. 2023] dataset for cross-identity reenactment, and all comparisons were conducted on novel views.

In addition, following prior studies [Lee et al. 2025; Wu et al. 2025a, 2024a], we also applied an SDEdit-style refinement baseline to the generated avatars. Specifically, SD-Turbo was utilized, which is equivalent to EDGAR’s backbone model with the timestep set to � = 250 for noise injection and denoising. This enables a direct comparison between our refinement approach and difusion-based editing methods widely used in 3D character generation.

Figure 6 presents EDGAR results for avatars reconstructed by CAP4D [Taubner et al. 2025a], GaussianAvatars [Qian et al. 2024a], and RGBAvatar [Li et al. 2025] under diferent input settings. Relative to the unrefined outputs, EDGAR reduces blurring and Gaussian artifacts while recovering fine facial details such as hair and skin texture. In contrast, the SDEdit-style refinement often introduced stochastic variations, leading to oversmooth texture and identity drift. These qualitative comparisons illustrate EDGAR’s refinement behavior across Gaussian avatars generated by diferent reconstruction methods. To further demonstrate performance on the same identity, Figure 8 complements Figure 6 by presenting the results of the three pipelines on the same identity and frame.

![](images/5ef29271e41180e880440663790dac6da2b8318024607a23f7302efbefccca2b.jpg)

![](images/7073d70101b65380c8522cd2d949420a76778078a56dc545eaa7901e3ccad0a1.jpg)

![](images/2a664ff90f99bdba5497bbbe5aa79c1084a34446049b6d880ac23cedf141c9a5.jpg)  
CAP4D

![](images/d9e581e42f9ba0c07a76e4c09e2ac95adc6cc71e657a80da1d81c514be505627.jpg)

![](images/7c8d8a99d9aa3222849555e0f286f82d9bac088fbf633d831b098aa75e6afd92.jpg)

![](images/115b5a5434353437ac7df39ac45b90149ae447482d463e55e23efa8a3eb79dd3.jpg)  
+SDEdit  
+Ours

![](images/f26da7d05b095833100f144af329eddbdd9d3689cc95e942ca13201e9d9a67f6.jpg)  
RGBAvatar

![](images/c50a82b0a58debc4e02bfec995214a5be1a0a928f9579fa7a17b4523351a923d.jpg)  
+SDEdit  
+Ours

![](images/9350625eadad8386724e89762a57d47bd3d289ec807a31180c7ec3057d0c5b84.jpg)

![](images/0b365a3acb59028161e991d47fd510baa050aaada41335bf64ce537892eb8190.jpg)  
GaussianAvatars  
+SDEdit

![](images/fefc3851f76089386b98b8427b5dada989fbce788fc077dcee8d4fb11935425a.jpg)  
+Ours

Fig. 6. Comparison of self-reenactment results. Our method refines each baseline by restoring facial details such as the eyes, wrinkles, and sharper hair, whereas SDEdit-based refinement alters the source identity. The upper rows show outputs from avatars trained or optimized with fewer input images or views (one image for CAP4D, one video for RGBAvatar, and eight videos for GaussianAvatars), while the lower rows show outputs from avatars trained with more images or videos. Each block uses a diferent identity and input severity, so comparisons are made within each reconstruction method. The more severely degraded GaussianAvatars example is improved but remains only partially corrected.

This observation is further supported by the quantitative results for self-reenactment reported in Tables 1, 2, and 3. On CAP4D, EDGAR obtains the highest CLIP similarity and lowest ℓ<sub>1</sub> error at every input count, while the Baseline retains the lowest LPIPS and SDEdit the highest PSNR. On RGBAvatar and GaussianAvatars, EDGAR leads all four metrics at every input count.

We also evaluated cross-identity reenactment. Because groundtruth images are unavailable in this setting, we instead used cosine similarity between identity embeddings (CSIM), following CAP4D [Taubner et al. 2025a] for evaluation. As shown in Table 4, although EDGAR did not surpass the input avatar in identity similarity, the degradation was negligible. In contrast, SDEdit led to a substantial drop in CSIM scores. This trend is also reflected in Figure 7. EDGAR efectively removed artifacts around the ear, neck, and hair present in the base model, while SDEdit introduced identity changes and excessive smoothing.

## 4.3 Temporal Evaluation and Cost Analysis

We measured foreground temporal warping error (tWE) on 100 consecutive frames for each NeRSemble test identity. Dense optical flow and a bidirectional cycle-consistency validity mask were estimated from each unrefined Base sequence, where the same correspondences were used for Base and EDGAR. We computed the mean RGB �<sub>1</sub> diference after warping on valid foreground pixels at 256 × 256 resolution and averaged the errors over the 99 transitions within each identity. We report the mean and standard deviation across identities. As presented in Table 7, EDGAR lowered tWE for CAP4D and RGBAvatar but raised it for GaussianAvatars, although the average tWE across pipelines decreased. This metric measures motion-compensated photometric stability, complementing the spatial-fidelity evaluations.

For the cost analysis, we estimated Base-avatar optimization times for CAP4D, RGBAvatar, and GaussianAvatars and measured the complete EDGAR refinement time. Base reconstruction times were based on 100,000 CAP4D updates, 60,000 RGBAvatar updates, and 750,000 GaussianAvatars updates. They exclude preprocessing and CAP4D’s separate upstream multi-view synthesis stage, reported to require approximately four hours [Taubner et al. 2025a]. As presented in Table 8, complete EDGAR refinement required 5.04, 7.96, and 16.60 minutes on average, respectively. Importantly, EDGAR incurs only a one-time refinement cost and adds no inference-time overhead: the refined avatars preserve the original Gaussian count and rendering path. In our experiments, output throughput remained within 2% of the base models and increased slightly in two of the three pipelines.

Expression 1

![](images/fb190e2483db278f0c8da9d623002fbf3cd0e90330f2f782d498c500871a4d49.jpg)  
CAP4D

![](images/4cfc61bbbd6cf3dae0d47675d229b940e80ca64385095492bc1597b4d58c44f1.jpg)  
+SDEdit

Expression 2

![](images/c7265438c2a5c6e76458b08ac7acf4880d518c00c682f3914ca9fc27ae81e39b.jpg)  
+Ours

![](images/81510054cd96301a664c3e72d369c762319f35f728ac9c9a174dc1947edef0f2.jpg)  
CAP4D

![](images/9830985765e8dc0271e333239ad8dbf20d4ff6337bf9099204c881d5c266e4bc.jpg)  
+SDEdit

![](images/e729c7aaf5288d66078048143f4a77841f5d2edf224a7c486c78a8162fa3fa33.jpg)  
+Ours

Fig. 7. Comparison of cross-reenactment results rendered at novel views. Our method refines each face by restoring facial details while efectively removing Gaussian artifacts such as floating blobs. In contrast, SDEdit-based refinement introduces identity shifts and oversmoothing skin texture.  
Table 1. Quantitative comparison on CAP4D across diferent numbers of input views. Best results per column are in red and second best are in yellow.
<table><tr><td rowspan="2">Method</td><td colspan="4">1 view</td><td colspan="4">4 views</td><td colspan="4">8 views</td></tr><tr><td>CLIP↑</td><td>L1↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>CLIP↑</td><td>L1↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>CLIP↑</td><td>L1↓</td><td>LPIPS ↓</td><td>PSNR ↑</td></tr><tr><td>CAP4D</td><td>0.9369</td><td>0.0584</td><td>0.2133</td><td>17.5651</td><td>0.9393</td><td>0.0537</td><td>0.1931</td><td>18.1435</td><td>0.9462</td><td>0.0518</td><td>0.1847</td><td>18.4773</td></tr><tr><td>+SDEdit</td><td>0.9078</td><td>0.0581</td><td>0.2728</td><td>17.7737</td><td>0.9101</td><td>0.0535</td><td>0.2562</td><td>18.3674</td><td>0.9151</td><td>0.0518</td><td>0.2529</td><td>18.6645</td></tr><tr><td>+Ours</td><td>0.9449</td><td>0.0580</td><td>0.2147</td><td>17.5978</td><td>0.9488</td><td>0.0534</td><td>0.1962</td><td>18.1927</td><td>0.9531</td><td>0.0513</td><td>0.1884</td><td>18.5329</td></tr></table>

Table 2. Quantitative comparison on RGBAvatar across diferent numbers of input videos. Best results per column are in red and second best are in yellow.
<table><tr><td rowspan="2">Method</td><td colspan="4">1 video</td><td colspan="4">4 videos</td><td colspan="4">8 videos</td></tr><tr><td>CLIP↑</td><td>L1↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>CLIP↑</td><td>L1↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>CLIP↑</td><td>L1↓</td><td>LPIPS ↓</td><td>PSNR ↑</td></tr><tr><td>RGBAvatar</td><td>0.8423</td><td>0.0832</td><td>0.3619</td><td>16.0469</td><td>0.8877</td><td>0.0552</td><td>0.2524</td><td>19.4950</td><td>0.9210</td><td>0.0502</td><td>0.2100</td><td>20.5128</td></tr><tr><td>+SDEdit</td><td>0.8554</td><td>0.0827</td><td>0.3601</td><td>16.5024</td><td>0.8639</td><td>0.0570</td><td>0.2878</td><td>19.9181</td><td>0.8712</td><td>0.0555</td><td>0.2752</td><td>20.5109</td></tr><tr><td>+Ours</td><td>0.8897</td><td>0.0677</td><td>0.3096</td><td>17.1533</td><td>0.9273</td><td>0.0364</td><td>0.2121</td><td>21.5199</td><td>0.9379</td><td>0.0332</td><td>0.1894</td><td>22.9937</td></tr></table>

## 4.4 Cross-View Geometric Consistency

We additionally measured whether refinement changed the 3D consistency of the reconstructed avatar across viewpoints. For each identity and input setting, we sampled 15 evenly spaced time steps and evaluated four views, averaging both directions of every view pair. For GaussianAvatars and RGBAvatar, these were synchronized cameras. Because CAP4D’s stored sequence changes facial state together with viewpoint, we instead rendered each fixed facial state from four nearby novel cameras. Frame-level values were first averaged within each identity, and Table 9 reports the mean and sample standard deviation over the nine identities.

With calibrated renderer depth for CAP4D and GaussianAvatars, we unprojected a source pixel, transformed it into the target camera, and measured the symmetric relative depth discrepancy $E _ { d } = 2 | z -$ $d | / ( | z | + | d | )$ , where � is its projected target-camera depth and � is the sampled target depth. We also report valid reprojection coverage �, defined as the proportion of source-foreground pixels whose projections are valid in the target view. RGBAvatar does not expose comparable renderer depth for its stored views, so we computed its $E _ { d }$ and � values using cameras and depth estimated jointly from the four images by VGGT [Wang et al. 2025]. For calibrated CAP4D and GaussianAvatars, the refined avatar improved both $E _ { d }$ and � at every input setting. For RGBAvatar, whose geometry was estimated from the rendered images, refinement improved both measures at four and eight inputs but worsened both in the single-input setting. Overall, these results show that EDGAR generally improves the 3D cross-view consistency of the refined avatars.

Table 3. Quantitative comparison on GaussianAvatars across diferent numbers of input videos. Best results per column are in red and second best are in yellow.
<table><tr><td rowspan="2">Method</td><td colspan="4">8 videos</td><td colspan="4">10 videos</td><td colspan="4">12 videos</td></tr><tr><td>CLIP↑</td><td>L1↓</td><td>LPIPS↓</td><td>PSNR ↑</td><td>CLIP↑</td><td>L1↓</td><td>LPIPS↓</td><td>PSNR ↑</td><td>CLIP↑</td><td>L1↓</td><td>LPIPS↓</td><td>PSNR ↑</td></tr><tr><td>GaussianAvatars</td><td>0.7819</td><td>0.0852</td><td>0.3299</td><td>16.2855</td><td>0.8407</td><td>0.0644</td><td>0.2661</td><td>18.5994</td><td>0.8759</td><td>0.0738</td><td>0.2303</td><td>18.1037</td></tr><tr><td>+SDEdit</td><td>0.8173</td><td>0.0806</td><td>0.2894</td><td>16.9214</td><td>0.8767</td><td>0.0649</td><td>0.2377</td><td>18.9219</td><td>0.9025</td><td>0.0782</td><td>0.2202</td><td>18.0437</td></tr><tr><td>+Ours</td><td>0.8300</td><td>0.0739</td><td>0.2526</td><td>17.3766</td><td>0.8977</td><td>0.0593</td><td>0.2050</td><td>19.3206</td><td>0.9214</td><td>0.0724</td><td>0.1875</td><td>18.2789</td></tr></table>

![](images/2d4eb019ae536c41f4ea68d49232fb2d49751c95c4f1839b96fc9065be1efbdd.jpg)  
Fig. 8. Matched-subject comparison at the same evaluation frame. Native inputs are four images for CAP4D, four videos for RGBAvatar, and ten videos for GaussianAvatars. Identity and evaluation frame are shared, while input regimes and initial reconstruction quality difer across pipelines.

Table 4. Quantitative comparison of CSIM across diferent numbers of input views. Best results per column are in red and second best are in yellow.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>1 view</td><td rowspan=1 colspan=1>4 views</td><td rowspan=1 colspan=1>8 views</td></tr><tr><td rowspan=1 colspan=1>Base</td><td rowspan=1 colspan=1>0.5510</td><td rowspan=1 colspan=1>0.5642</td><td rowspan=1 colspan=1>0.5651</td></tr><tr><td rowspan=1 colspan=1>+SDEdit</td><td rowspan=1 colspan=1>0.3178</td><td rowspan=1 colspan=1>0.3432</td><td rowspan=1 colspan=1>0.3380</td></tr><tr><td rowspan=1 colspan=1>+Ours</td><td rowspan=1 colspan=1>0.5169</td><td rowspan=1 colspan=1>0.5326</td><td rowspan=1 colspan=1>0.5419</td></tr></table>

## 4.5 Pose-and-Expression Robustness

We evaluated large-pose and open-mouth conditions, using four images for CAP4D, four videos for RGBAvatar, and ten videos for GaussianAvatars. For the video pipelines, we sampled 64 uniformly spaced time steps from each camera sequence. For CAP4D, we evaluated eight images per identity. We used landmarks estimated from ground-truth images using Face Mesh [Kartynnik et al. 2019] to determine absolute head yaw through a perspective pose fit and mouth opening through the 3D inner-lip gap divided by mouth width. We report results for three subsets: $| \mathrm { y a w } | \ge 3 0 ^ { \circ }$ , normalized lip gap $\geq 0 . 1 0 .$ , and their intersection. Subset membership was determined independently of the Base and EDGAR outputs. We measured foreground $L _ { 1 }$ and LPIPS with a common foreground mask and white background. Images retained their aspect ratio, with the longest side resized to 512 pixels. Native ground-truth masks were used for the video pipelines, whereas CAP4D foreground was identified from its white background. Errors were averaged within each identity and then across identities (Table 10).

Table 5. Ablation study results across diferent numbers of input views on self-reenactment. Best results per column are in red and second best are in yellow.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=2 colspan=4>1 viewCLIP ↑ L1↓ LPIPS ↓PSNR ↑</td><td rowspan=1 colspan=4>4 views</td><td rowspan=1 colspan=4>8 views</td></tr><tr><td rowspan=1 colspan=4>CLIP↑  L1↓  LPIPS↓PSNR ↑</td><td rowspan=1 colspan=4>CLIP ↑ L1↓ LPIPS ↓ PSNR ↑</td></tr><tr><td rowspan=1 colspan=1>w/o Ir</td><td rowspan=1 colspan=1>0.9163</td><td rowspan=1 colspan=2>0.0595  0.2683</td><td rowspan=1 colspan=1>17.5822</td><td rowspan=1 colspan=1>0.9222</td><td rowspan=1 colspan=1>0.0547</td><td rowspan=1 colspan=1>0.2492</td><td rowspan=1 colspan=1>18.1634</td><td rowspan=1 colspan=2>0.9262 0.0527</td><td rowspan=1 colspan=1>0.2448</td><td rowspan=1 colspan=1>18.4787</td></tr><tr><td rowspan=1 colspan=1>w/o Synthetic</td><td rowspan=1 colspan=1>0.9351</td><td rowspan=1 colspan=1>0.0583</td><td rowspan=1 colspan=1>0.2292</td><td rowspan=1 colspan=1>17.6128</td><td rowspan=1 colspan=1>0.9407</td><td rowspan=1 colspan=1>0.0538</td><td rowspan=1 colspan=1>0.2106</td><td rowspan=1 colspan=1>18.1925</td><td rowspan=1 colspan=1>0.9436</td><td rowspan=1 colspan=1>0.0518</td><td rowspan=1 colspan=1>0.2033</td><td rowspan=1 colspan=1>18.5236</td></tr><tr><td rowspan=1 colspan=1>w/o $\mathcal { L } _ { e x p }$ </td><td rowspan=1 colspan=1>0.9373</td><td rowspan=1 colspan=1>0.0578</td><td rowspan=1 colspan=1>0.2297</td><td rowspan=1 colspan=1>17.6271</td><td rowspan=1 colspan=1>0.9433</td><td rowspan=1 colspan=1>0.0533</td><td rowspan=1 colspan=1>0.2107</td><td rowspan=1 colspan=1>18.2135</td><td rowspan=1 colspan=1>0.9465</td><td rowspan=1 colspan=1>0.0512</td><td rowspan=1 colspan=1>0.2038</td><td rowspan=1 colspan=1>18.5370</td></tr><tr><td rowspan=1 colspan=1>w/o MoE</td><td rowspan=1 colspan=1>0.9420</td><td rowspan=1 colspan=1>0.0580</td><td rowspan=1 colspan=1>0.2148</td><td rowspan=1 colspan=1>17.5757</td><td rowspan=1 colspan=1>0.9456</td><td rowspan=1 colspan=1>0.0534</td><td rowspan=1 colspan=1>0.1962</td><td rowspan=1 colspan=1>18.1689</td><td rowspan=1 colspan=1>0.9503</td><td rowspan=1 colspan=1>0.0514</td><td rowspan=1 colspan=1>0.1884</td><td rowspan=1 colspan=1>18.5053</td></tr><tr><td rowspan=1 colspan=1>w/ two Experts</td><td rowspan=1 colspan=1>0.9433</td><td rowspan=1 colspan=1>0.0581</td><td rowspan=1 colspan=1>0.2154</td><td rowspan=1 colspan=1>17.5824</td><td rowspan=1 colspan=1>0.9470</td><td rowspan=1 colspan=1>0.0535</td><td rowspan=1 colspan=1>0.1967</td><td rowspan=1 colspan=1>18.1802</td><td rowspan=1 colspan=1>0.9521</td><td rowspan=1 colspan=1>0.0514</td><td rowspan=1 colspan=1>0.1888</td><td rowspan=1 colspan=1>18.5201</td></tr><tr><td rowspan=1 colspan=1>Ours</td><td rowspan=1 colspan=1>0.9449</td><td rowspan=1 colspan=1>0.0580</td><td rowspan=1 colspan=1>0.2147</td><td rowspan=1 colspan=1>17.5978</td><td rowspan=1 colspan=1>0.9488</td><td rowspan=1 colspan=1>0.0534</td><td rowspan=1 colspan=1>0.1962</td><td rowspan=1 colspan=1>18.1927</td><td rowspan=1 colspan=1>0.9531</td><td rowspan=1 colspan=1>0.0513</td><td rowspan=1 colspan=1>0.1884</td><td rowspan=1 colspan=1>18.5329</td></tr></table>

Table 6. Ablation study results across diferent numbers of input views on cross-reenactment measured by CSIM. Best results per column are in red and second best are in yellow.
<table><tr><td>Method</td><td>1 view</td><td>4 views</td><td>8 views</td></tr><tr><td>w/o  $I _ { r }$  w/o Synthetic</td><td>0.4508 0.4971</td><td>0.4716 0.5244</td><td>0.4769 0.5242</td></tr><tr><td>w/o  $\mathcal { L } _ { e x p }$ </td><td>0.4645</td><td>0.4930</td><td>0.4927</td></tr><tr><td>w/o MoÊ</td><td>0.5177</td><td>0.5287</td><td>0.5355</td></tr><tr><td>w/ two Experts</td><td>0.5162</td><td>0.5301</td><td>0.5363</td></tr><tr><td>Ours</td><td>0.5169</td><td>0.5326</td><td>0.5419</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 7. Foreground tWE (×10<sup>−3</sup>; lower is beter). Values are mean ± identity-level standard deviation. The lower value in each row is red.
<table><tr><td>Pipeline</td><td>Base</td><td>+EDGAR</td></tr><tr><td>CAP4D</td><td> $1 . 7 5 9 \pm 0 . 2 1 0$ </td><td> $1 . 6 9 6 \pm 0 . 1 9 9$ </td></tr><tr><td>RGBAvatar</td><td> $7 . 4 2 4 \pm 1 . 4 1 1$ </td><td> $5 . 5 7 4 \pm 1 . 5 1 2$ </td></tr><tr><td>GaussianAvatars</td><td>_  $4 . 9 2 9 \pm 1 . 1 1 1$  一</td><td> $5 . 9 3 8 \pm 1 . 9 3 2$ </td></tr><tr><td>Average</td><td> $4 . 7 0 4 \pm 0 . 8 0 3$ </td><td>_  $4 . 4 0 3 \pm 1 . 1 0 5$ </td></tr></table>

Ground truth  
Base  
EDGAR

EDGAR reduced both foreground errors for RGBAvatar and GaussianAvatars, including the combined large-pose/open-mouth subset. CAP4D yielded mixed results. These results show that refinement benefits extend to the tested large-pose and open-mouth conditions for the two video-based pipelines. Figure 9 shows examples from three RGBAvatar identities.

![](images/ddc53b86752157c6d9152236987d94584d76d0c520453b495686da1e45af018a.jpg)  
Fig. 9. Large-yaw with optional open-mouth examples for RGBAvatar with four input videos. Rows show examples with absolute yaw angles of 37.4<sup>◦</sup>, 37.7<sup>◦</sup>, and 36.3<sup>◦</sup>, respectively. Examples are selected near each identity’s median ground-truth yaw and mouth opening within the combined subset.

## 4.6 Ablation Study

To better understand the contribution ofeach component in EDGAR, we conducted an ablation study using CAP4D as the base reconstruction method. Specifically, we evaluated: (1) w/o $I _ { r } ,$ which removes the reference image during refinement; (2) w/o Synthetic, which excludes the synthetic split of EDGAR-SET during training; (3) w/o $\mathcal L _ { e x p } ,$ which removes the expression loss; (4) w/o MoE, which removes the MoE architecture from the EDGAR denoiser; and (5) w/ two Experts, which uses only two experts to examine whether the benefit of MoE simply results from separating the real and synthetic data in EDGAR-SET.

Table 8. Computational cost and output throughput on an RTX 3090. Refinement time includes loading, pseudo-target generation, avatar optimization, and checkpoint writing. Values are mean ± sample standard deviation.
<table><tr><td rowspan="2">Pipeline</td><td colspan="2">One-time cost (min)</td><td colspan="2">Output FPS</td></tr><tr><td></td><td>Base opt. Avatar refine.</td><td>Base</td><td>EDGAR</td></tr><tr><td>CAP4D</td><td> $4 8 4 . 0 \pm 8 0 . 5$ </td><td> $5 . 0 4 \pm 0 . 1 1$ </td><td> $5 0 . 3 \pm 0 . 3$ </td><td> $4 9 . 4 \pm 0 . 6$ </td></tr><tr><td>RGBAvatar</td><td> $1 0 5 . 6 \pm 4 . 4$ </td><td> $7 . 9 6 \pm 0 . 9 8$ </td><td> $8 1 3 . 8 \pm 3 6 . 4$ </td><td> $8 2 6 . 3 \pm 1 6 . 9$ </td></tr><tr><td>GaussianAvatars</td><td> $2 3 8 . 7 \pm 8 . 8$ </td><td> $1 6 . 6 0 \pm 5 . 9 8$ </td><td> $1 5 3 . 4 \pm 7 . 4$ </td><td> $1 5 6 . 1 \pm 0 . 7$ </td></tr></table>

Table 9. Cross-view geometric consistency results. Values are reported as mean ± standard deviation. Beter values are highlighted in red.
<table><tr><td>Avatar family</td><td>Inputs Metric</td><td></td><td>Base</td><td>EDGAR</td></tr><tr><td rowspan="3">CAP4D</td><td>1</td><td> $E _ { d } \downarrow$  C↑</td><td> $0 . 0 0 2 8 \pm 0 . 0 0 0 4$   $0 . 9 1 3 5 \pm 0 . 0 1 8 1$ </td><td> $0 . 0 0 2 5 \pm 0 . 0 0 0 3$   $0 . 9 2 1 7 \pm 0 . 0 1 5 9$ </td></tr><tr><td>4</td><td> $E _ { d } \downarrow$   $C \uparrow$ </td><td> $0 . 0 0 2 8 \pm 0 . 0 0 0 4$   $0 . 9 1 4 5 \pm 0 . 0 1 6 6$ </td><td> $0 . 0 0 2 5 \pm 0 . 0 0 0 4$   $0 . 9 2 2 2 \pm 0 . 0 1 5 3$ </td></tr><tr><td>8</td><td> $E _ { d } \downarrow$   $C \uparrow$ </td><td> $0 . 0 0 2 8 \pm 0 . 0 0 0 4$   $0 . 9 1 4 1 \pm 0 . 0 1 6 9$ </td><td> $0 . 0 0 2 5 \pm 0 . 0 0 0 4$   $0 . 9 2 1 8 \pm 0 . 0 1 5 7$ </td></tr><tr><td rowspan="3">RGBAvatar</td><td>1</td><td> $E _ { d } \downarrow$   $C \uparrow$ </td><td> $0 . 0 0 8 3 \pm 0 . 0 0 1 8$   $0 . 4 8 5 5 \pm 0 . 0 4 5 3$ </td><td> $0 . 0 0 9 7 \pm 0 . 0 0 3 7$   $0 . 4 2 6 1 \pm 0 . 0 7 5 3$ </td></tr><tr><td>4</td><td> $E _ { d } \downarrow$   $C \uparrow$ </td><td> $0 . 0 0 8 8 \pm 0 . 0 0 2 0$   $0 . 4 2 4 8 \pm 0 . 0 4 4 5$ </td><td> $0 . 0 0 7 4 \pm 0 . 0 0 1 7$   $0 . 4 3 3 9 \pm 0 . 0 3 8 3$ </td></tr><tr><td>8</td><td> $E _ { d } \downarrow$   $C \uparrow$ </td><td> $0 . 0 0 9 4 \pm 0 . 0 0 2 1$   $0 . 4 2 3 3 \pm 0 . 0 3 7 2$ </td><td> $0 . 0 0 8 6 \pm 0 . 0 0 3 4$   $0 . 4 3 4 8 \pm 0 . 0 3 7 1$ </td></tr><tr><td rowspan="3">GaussianAvatars</td><td>8</td><td> $E _ { d } \downarrow$   $C \uparrow$ </td><td> $0 . 0 0 7 0 \pm 0 . 0 0 0 8$   $0 . 5 1 6 8 \pm 0 . 0 4 1 8$ </td><td> $0 . 0 0 6 3 \pm 0 . 0 0 0 6$   $0 . 5 3 3 6 \pm 0 . 0 4 2 0$ </td></tr><tr><td>10</td><td> $E _ { d } \downarrow$   $C \uparrow$ </td><td> $0 . 0 0 7 2 \pm 0 . 0 0 0 9$   $0 . 5 1 1 2 \pm 0 . 0 4 0 2$ </td><td> $0 . 0 0 6 4 \pm 0 . 0 0 0 7$   $0 . 5 2 6 1 \pm 0 . 0 4 0 5$ </td></tr><tr><td>12</td><td> $E _ { d } \downarrow$   $C \uparrow$ </td><td> $0 . 0 0 6 9 \pm 0 . 0 0 1 0$   $0 . 5 1 3 1 \pm 0 . 0 4 2 7$ </td><td> $0 . 0 0 6 2 \pm 0 . 0 0 0 7$   $0 . 5 2 6 5 \pm 0 . 0 4 0 4$ </td></tr></table>

As summarized in Tables 5 and 6, corresponding to results from NeRSemble and INSTA, respectively, the full EDGAR model achieved the best overall balance between reconstruction fidelity and identity preservation. The ablation results highlight the importance of the reference conditioning term �<sub>�</sub> and the synthetic split of EDGAR-SET. Removing � led to the largest performance drop in both selfreenactment and cross-reenactment, showing that the reference image and its corresponding Sapiens feature are crucial for preserving identity and stabilizing refinement. Similarly, excluding the synthetic split consistently degrades performance, especially in terms of CLIP, LPIPS, and cross-reenactment CSIM, demonstrating that the synthetic split provides important complementary supervision for generalization. This is also evident in the qualitative comparisons shown in Figure 10, where the full model reduced artifacts and preserved identity more reliably than the variants without $I _ { r }$ or synthetic training data.

Table 10. Pose-and-expression evaluation at representative native input setings (CAP4D: four images; RGBAvatar: four videos; GaussianAvatars: ten videos). Values are equally weighted identity means. Large pose denotes |yaw| ≥ 30<sup>◦</sup>, and open mouth denotes a normalized lip gap ≥ 0.10. Beter values are highlighted in red.
<table><tr><td></td><td></td><td colspan="2">Foreground L1↓</td><td colspan="2">Foreground LPIPS↓</td></tr><tr><td>Pipeline</td><td>Condition</td><td>Base</td><td>EDGAR</td><td>Base</td><td>EDGAR</td></tr><tr><td>CAP4D</td><td>Large pose</td><td>0.07500</td><td></td><td>0.07601 0.22353</td><td>0.21704</td></tr><tr><td></td><td>Open mouth0.06906</td><td></td><td></td><td>0.06660 0.21057</td><td>0.21975</td></tr><tr><td></td><td>Both</td><td>0.07475</td><td>0.08084</td><td>0.23054</td><td>0.23507</td></tr><tr><td>RGBAvatar</td><td>Large pose</td><td>0.08894</td><td>0.06121</td><td>0.17845</td><td>0.14399</td></tr><tr><td></td><td>Open mouth</td><td>0.06410</td><td>0.05372</td><td>0.12051</td><td>0.10368</td></tr><tr><td></td><td>Both</td><td>0.08785</td><td>0.06112</td><td>0.17563</td><td>0.14150</td></tr><tr><td>GaussianAvatars</td><td>Large pose</td><td>0.11531</td><td>0.10845</td><td>0.18029</td><td>0.13128</td></tr><tr><td></td><td>Open mouth</td><td>0.11748</td><td>0.11192</td><td>0.17604</td><td>0.12943</td></tr><tr><td></td><td>Both</td><td>0.11762</td><td>0.11187</td><td>0.17572</td><td>0.12934</td></tr></table>

![](images/3b6de95adfac3eeb9590c8db30a35a40be79a18a77551649ceb6c5790597f1f6.jpg)

![](images/c4d6e5728da39ffb2bf5a258019ef9118a9d2b2c979229f99cfb2592ab022b7f.jpg)

![](images/3e1ded2a49bcc6d227ed10f8e597bb54560fe1842945cc5ce141b7af28223ce7.jpg)

![](images/b9c4640a020ed53f9c3a0dbec11034a45976556eeb7db88ecb14be39a844368b.jpg)

Input  
![](images/fad460fc74ea0038031e1e65714aa7b9aa1d88938671bafdb6b83a34e1ade3bf.jpg)

![](images/b73ffb3c47ab928a136e83db3cf0de4e0b0ebdb3543047ca29333fcf24f92885.jpg)

w/o Ir  
![](images/442d44dfb81d269709f3bc6d9fd97bfb876d825241994f5d9e9cd608385fcb3b.jpg)

![](images/b447a89055cca6899ea850f4f1fa1a2bb99a4471fb85d5e5dd4351974dbb4216.jpg)  
w/o Synthetic  
Ours  
Fig. 10. Ablation study results for � and Synthetic data. Some of the arti facts in the hair, ears, and floating blobs exhibited by the input Gaussian avatar are mitigated by the ablated variants. Nonetheless, the variant without � produces altered details and overly smooth textures due to the absence of reference information, while the variant without synthetic data retains artifacts such as distortions around the ears and floating Gaussian blobs. In contrast, the full model beter preserves identity while efectively removing artifacts and improving overall visual quality. Visible artifacts in the ear and floating Gaussian blobs are highlighted in red boxes.

![](images/e8dc8988cbd7d17cc8856a5317ecf93cd6e3f07a293747b1928edd744fde118c.jpg)  
Fig. 11. Ablation study results on $\mathcal { L } _ { e x p } .$ Removing $\mathcal { L } _ { e x p }$ degrades the preservation of facial expression details.

![](images/0178f72e31d597279c34dd59d62ece7d4e630234d28de68ad8cd4f0cbf1f8b37.jpg)  
Fig. 12. Qualitative results for the MoE ablation. All variants reduce artifacts, but the variants without MoE and with two experts retain sharp white artifacts around the ears. The full model suppresses these artifacts more efectively in this example.

The efect of the expression loss $\mathcal { L } _ { e x p }$ was also meaningful. In self-reenactment, the full model consistently achieved the best CLIP similarity and LPIPS scores across all input view settings. Although the variant without $\mathcal { L } _ { e x p }$ slightly improved L1 and PSNR, its noticeably worse CLIP and LPIPS scores indicate that optimizing only for pixel-level fidelity does not necessarily lead to better perceptual quality or identity consistency. This tendency becomes more evident in cross-reenactment, where removing $\mathcal { L } _ { e x p }$ caused a substantial drop in CSIM scores across all input-view settings. Figure 11 shows that the full model better preserves expression details compared to the variant without $\mathcal { L } _ { e x p } .$

Finally, the comparison among w/o MoE, w/ two Experts, and the full model shows that the six-expert model provides improvements in CLIP and in 4-view and 8-view cross-reenactment CSIM, while matching or slightly trailing the ablated variants on other entries. Figure 12 illustrates one case in which the full model better suppresses artifacts around the ear. The reference image, synthetic training split, and expression loss have larger efects than the MoE configuration.

## 4.7 User Study

To further assess the perceptual quality of refined avatars, we con ducted a web-based user study with 15 participants, with an average age of 27.6 years, including 7 males and 8 females. Each participant was presented with 20 randomized pairs of avatar renderings shown as video sequences generated by the baseline methods (CAP4D and CAP4D+SDEdit) and our method, EDGAR. Each pair consisted of cross-reenactment results, along with the source identity as a reference for identity comparison, and were displayed side-by-side in randomized order to avoid positional bias. Participants were asked to choose their preferred result with respect to three criteria: animation quality, identity preservation, and overall avatar preference.

Table 11. Selection ratio of our method over the baselines in the user study.
<table><tr><td>Method</td><td></td><td>Animation Preservation Preference</td><td></td></tr><tr><td>Ours vs CAP4D</td><td>87.33%</td><td>91.33%</td><td>96.67%</td></tr><tr><td>Ours vs CAP4D+SDEdit</td><td>94.67%</td><td>98.0%</td><td>98.0%</td></tr></table>

As shown in Table 11, EDGAR was consistently preferred over both baselines across all evaluation criteria. Compared with CAP4D, EDGAR achieved preference rates of 87.33% in animation quality, 91.33% in identity preservation, and 96.67% in overall avatar preference. The preference margin was even larger when compared with CAP4D+SDEdit, where EDGAR achieved 94.67%, 98.0%, and 98.0% for the three criteria, respectively. These outcomes align with our quantitative findings, indicating that EDGAR not only improves animation quality and identity preservation, but also yields avatars that are perceived by participants as more visually convincing overall.

## 5 Conclusion

We introduced EDGAR, a difusion-driven refinement framework for Gaussian head avatars. EDGAR adapts single-step image refinement to animatable heads through paired avatar-failure data, subjectreference conditioning, expression-aware supervision, and an auxiliary MoE module. Experiments on CAP4D, RGBAvatar, and GaussianAvatars demonstrate improvements in perceptual and reconstruction quality, with gains depending on the starting avatar and evaluation metric. The shared-avatar update retains approximately the Base rendering rate and improves calibrated cross-view consistency for CAP4D and GaussianAvatars. The pose-and-expression evaluation shows benefits under large yaw and open-mouth condi tions.

Limitations and Future Work. EDGAR is trained on structurally recoverable degradations: EDGAR-SET filters source–target pairs with large LPIPS diferences. Consequently, severe structural errors and artifacts in unobserved regions can remain after refinement, as illustrated in Figure 6. Recovering such failures may require additional observations or geometric supervision.

Independently refined pseudo-targets can disagree across views, causing the shared Gaussian representation to average conflicting details or retain view-dependent errors. Although our cross-view evaluation shows that EDGAR generally improves 3D consistency, explicit multi-view constraints and confidence-aware selection of refined targets could further reduce such inconsistencies. Mouth interiors and translucent appearance also remain challenging. In addition, the temporal evaluation shows that the mean tWE decreased for CAP4D and RGBAvatar but increased for GaussianAvatars, indicating that improvements in spatial fidelity do not necessarily translate into greater temporal stability across all reconstruction pipelines. Explicit temporal supervision and improved modeling of dynamic facial regions are promising directions for future work.

## Acknowledgments

This work was supported by Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No.RS-2024-00439499, Generating Hyper-Realistic to Extremely-stylized Face Avatar with Varied Speech Speed and Context-based Emotional Expression)

## References

Rameen Abdal, Yipeng Qin, and Peter Wonka. 2019. Image2stylegan: How to embed images into the stylegan latent space?. In CVPR.

Sizhe An, Hongyi Xu, Yichun Shi, Guoxian Song, Umit Y Ogras, and Linjie Luo. 2023. Panohead: Geometry-aware 3d full-head synthesis in 360deg. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20950–20959.

Shivangi Aneja, Sebastian Weiss, Irene Baeza, Prashanth Chandran, Gaspard Zoss, Matthias Niessner, and Derek Bradley. 2025. ScafoldAvatar: High-Fidelity Gaussian Avatars with Patch Expressions. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–11.

Volker Blanz and Thomas Vetter. 1999. A morphable model for the synthesis of 3D faces. In Proceedings ofthe 26th annual conference on Computer graphics and interactive techniques. 187–194.

Volker Blanz and Thomas Vetter. 2003. Face recognition based on fitting a 3D morphable model. IEEE Transactions on pattern analysis and machine intelligence 25, 9 (2003), 1063–1074.

Marcel C Buehler, Gengyan Li, Erroll Wood, Leonhard Helminger, Xu Chen, Tanmay Shah, Daoye Wang, Stephan Garbin, Sergio Orts-Escolano, Otmar Hilliges, et al. 2024. Cafca: High-quality novel view synthesis of expressive faces from casual few-shot captures. In SIGGRAPH Asia 2024 Conference Papers. 1–12.

Chen Cao, Qiming Hou, and Kun Zhou. 2014. Displaced dynamic expression regression for real-time facial tracking and animation. ACM Transactions on graphics (TOG) 33, 4 (2014), 1–10.

Chen Cao, Yanlin Weng, Shun Zhou, Yiying Tong, and Kun Zhou. 2013. Facewarehouse: A 3d facial expression database for visual computing. IEEE Transactions on Visualization and Computer Graphics 20, 3 (2013), 413–425.

Peng Chen, Xiaobao Wei, Qingpo Wuwu, Xinyi Wang, Xingyu Xiao, and Ming Lu. 2025. Mixedgaussianavatar: Realistically and geometrically accurate head avatar via mixed 2d-3d gaussians. In Proceedings of the 33rd ACM International Conference on Multimedia. 945–954.

Yufan Chen, Lizhen Wang, Qijing Li, Hongjiang Xiao, Shengping Zhang, Hongxun Yao, and Yebin Liu. 2024. Monogaussianavatar: Monocular gaussian point-based head avatar. In ACM SIGGRAPH 2024 Conference Papers. 1–9.

Xuangeng Chu and Tatsuya Harada. 2024. Generalizable and animatable gaussian head avatar. Advances in Neural Information Processing Systems 37 (2024), 57642–57670.

Radek Daněček, Michael J Black, and Timo Bolkart. 2022. Emoca: Emotion driven monocular face capture and animation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20311–20322.

Yu Deng, Jiaolong Yang, Sicheng Xu, Dong Chen, Yunde Jia, and Xin Tong. 2019. Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set. In CVPRW.

Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart. 2021. Learning an animat able detailed 3D face model from in-the-wild images. ACM Transactions on Graphics (ToG) 40, 4 (2021), 1–13.

Leon A Gatys, Alexander S Ecker, and Matthias Bethge. 2016. Image style transfer using convolutional neural networks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 2414–2423.

Simon Giebenhain, Tobias Kirschstein, Markos Georgopoulos, Martin Rünz, Lourdes Agapito, and Matthias Nießner. 2023. Learning neural parametric head models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21003–21012.

Simon Giebenhain, Tobias Kirschstein, Martin Rünz, Lourdes Agapito, and Matthias Nießner. 2024. Npga: Neural parametric gaussian avatars. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron cCourville, and Yoshua Bengio. 2014. Generative adversarial nets. Advances in neural information processing systems 27 (2014).

Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou Zhong, Yuan Zhang, Pengfe Wan, and Di Zhang. 2024. Liveportrait: Eficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168 (2024).

Yisheng He, Xiaodong Gu, Xiaodan Ye, Chao Xu, Zhengyi Zhao, Yuan Dong, Weihao Yuan, Zilong Dong, and Liefeng Bo. 2025. LAM: Large Avatar Model for One-shot Animatable Gaussian Head. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–13.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising difusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. ICLR 1, 2 (2022), 3.

Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. 2024. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 634–644.

Kaiwen Jiang, Shu-Yu Chen, Feng-Lin Liu, Hongbo Fu, and Lin Gao. 2022. Nerfaceed iting: Disentangled face editing in neural radiance fields. In SIGGRAPH Asia 2022 Conference Papers. 1–9.

Yury Kartynnik, Artsiom Ablavatski, Ivan Grishchenko, and Matthias Grundmann. 2019. Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs. arXiv preprint arXiv:1907.06724 (2019).

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.

Rawal Khirodkar, Timur Bagautdinov, Julieta Martinez, Su Zhaoen, Austin James, Peter Selednik, Stuart Anderson, and Shunsuke Saito. 2024. Sapiens: Foundation for human vision models. In European Conference on Computer Vision. Springer, 206–228.

Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim Walter, and Matthias Nießner. 2023. Nersemble: Multi-view radiance field reconstruction of human heads. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–14.

Tobias Kirschstein, Javier Romero, Artem Sevastopolsky, Matthias Nießner, and Shun suke Saito. 2025. Avat3r: Large animatable gaussian reconstruction model for high-fidelity 3d head avatars. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 12089–12100.

Alexandros Lattas, Stylianos Moschoglou, Baris Gecer, Stylianos Ploumpis, Vasileios Triantafyllou, Abhijeet Ghosh, and Stefanos Zafeiriou. 2020. AvatarMe: Realistically Renderable 3D Facial Reconstruction" in-the-wild". In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 760–769.

Seungmi Lee, Kwan Yun, and Junyong Noh. 2025. StyleMM: Stylized 3D Morphable Face Model via Text-Driven Aligned Image Translation. In Computer Graphics Forum, Vol. 44. Wiley Online Library, e70234.

Jiahe Li, Jiawei Zhang, Xiao Bai, Jun Zhou, and Lin Gu. 2023. Eficient region-aware neural radiance fields for high-fidelity talking portrait synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7568–7578.

Linzhou Li, Yumeng Li, Yanlin Weng, Youyi Zheng, and Kun Zhou. 2025. RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars. In Proceedings of the Computer Vision and Pattern Recognition Conference. 10747–10757.

Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero. 2017. Learning a model of facial shape and expression from 4D scans. ACM Trans. Graph. 36, 6 (2017), 194–1.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. 2022. Repaint: Inpainting using denoising difusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 11461–11471.

Ziwei Luo, Fredrik K Gustafsson, Zheng Zhao, Jens Sjölund, and Thomas B Schön. 2024. Photo-realistic image restoration in the wild with controlled vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6641–6651.

Shengjie Ma, Yanlin Weng, Tianjia Shao, and Kun Zhou. 2024. 3d gaussian blendshapes for head avatar animation. In ACM SIGGRAPH 2024 Conference Papers. 1–10.

Jiawei Mao, Rui Xu, Xuesong Yin, Yuanqi Chang, Binling Nie, Aibin Huang, and Yigang Wang. 2025. Poster++: A simpler and stronger facial expression recognition network. Pattern Recognition 157 (2025), 110951.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2021. Sdedit: Guided image synthesis and editing with stochastic diferential equations. arXiv preprint arXiv:2108.01073 (2021).

Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. 2019. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 4460–4470.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

SeungJun Moon, Hah Min Lew, Seungeun Lee, Ji-Su Kang, and Gyeong-Moon Park. 2025. GeoAvatar: Adaptive Geometrical Gaussian Splatting for 3D Head Avatar. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 12811– 12821.

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) 41, 4 (2022), 1–15.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).

Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. 2019. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 165–174.

Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. 2024. One-step image translation with text-to-image models. arXiv preprint arXiv:2403.12036 (2024).

Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, and Matthias Nießner. 2024a. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20299–20309.

Zhiyin Qian, Shaofei Wang, Marko Mihajlovic, Andreas Geiger, and Siyu Tang. 2024b. 3dgs-avatar: Animatable avatars via deformable 3d gaussian splatting. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 5020–5030.

Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. 2021a. Encoding in style: a stylegan encoder for image-toimage translation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 2287–2296.

Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. 2021b. Encoding in style: a stylegan encoder for image-to image translation. In CVPR. 2287–2296.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis with Latent Difusion Models. In CVPR.

Sami Romdhani and Thomas Vetter. 2005. Estimating 3D shape and texture using pixel intensity, edges, specular highlights, texture constraints and a prior. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), Vol. 2. IEEE, 986–993.

Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. 2022a. Palette: Image-to-image difusion models. In ACM SIGGRAPH 2022 conference proceedings. 1–10.

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mo hammad Norouzi. 2022b. Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence 45, 4 (2022), 4713–4726.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. 2024. Adversaria difusion distillation. In European Conference on Computer Vision. Springer, 87–103.

Zhijing Shao, Zhaolong Wang, Zhuang Li, Duotun Wang, Xiangru Lin, Yu Zhang, Mingming Fan, and Zeyu Wang. 2024. Splattingavatar: Realistic real-time human avatars with mesh-embedded gaussian splatting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1606–1616.

Luchuan Song, Yang Zhou, Zhan Xu, Yi Zhou, Deepali Aneja, and Chenliang Xu. 2025. StreamME: Simplify 3D Gaussian Avatar within Live Stream. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–10.

Heyi Sun, Cong Wang, Tian-Xing Xu, Jingwei Huang, Di Kang, Chunchao Guo, and Song-Hai Zhang. 2025. SVG-Head: Hybrid Surface-Volumetric Gaussians for High Fidelity Head Reconstruction and Real-Time Editing. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 13326–13335.

Jingxiang Sun, Xuan Wang, Yong Zhang, Xiaoyu Li, Qi Zhang, Yebin Liu, and Jue Wang. 2022. Fenerf: Face editing in neural radiance fields. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 7672–7682.

Felix Taubner, Ruihang Zhang, Mathieu Tuli, Sherwin Bahmani, and David B Lindell. 2025b. Mvp4d: Multi-view portrait video difusion for animatable 4d avatars. In Proceedings ofthe SIGGRAPH Asia 2025 Conference Papers. 1–11.

Felix Taubner, Ruihang Zhang, Mathieu Tuli, and David B Lindell. 2025a. Cap4d: Creating animatable 4d portrait avatars with morphable multi-view difusion models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 5318–5330.

Kartik Teotia, Hyeongwoo Kim, Pablo Garrido, Marc Habermann, Mohamed Elgharib, and Christian Theobalt. 2024. Gaussianheads: End-to-end learning of drivable gaussian head avatars from coarse-to-fine representations. ACM Transactions on Graphics (ToG) 43, 6 (2024), 1–12.

Kartik Teotia, Helge Rhodin, Mohit Mendiratta, Hyeongwoo Kim, Marc Habermann, and Christian Theobalt. 2025. Audio Driven Universal Gaussian Head Avatars. In Proceedings ofthe SIGGRAPH Asia 2025 Conference Papers. 1–12.

Justus Thies, Michael Zollhofer, Marc Stamminger, Christian Theobalt, and Matthias Nießner. 2016. Face2face: Real-time face capture and reenactment of rgb videos. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2387–2395.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. 2025. VGGT: Visual Geometry Grounded Transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5294–5306.

Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. 2024. Exploiting difusion prior for real-world image super-resolution. International Journal ofComputer Vision 132, 12 (2024), 5929–5949.

Xiaobao Wei, Peng Chen, Guangyu Li, Ming Lu, Hui Chen, and Feng Tian. 2025a. Gazegaussian: High-fidelity gaze redirection with 3d gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 13293–13303.

Xiaobao Wei, Peng Chen, Ming Lu, Hui Chen, and Feng Tian. 2025b. Graphavatar: Compact head avatars with gnn-generated 3d gaussians. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 8295–8303.

Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. 2025b. Difix3d+: Improving 3d reconstructions with single-step difusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 26024–26035.

Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. 2024b. Seesr: Towards semantics-aware real-world image super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 25456–25467.

Yiqian Wu, Malte Prinzler, Xiaogang Jin, and Siyu Tang. 2025a. Text-based Animatable 3D Avatars with Morphable Model Alignment. In Proceedings ofthe Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–11.

Yiqian Wu, Hao Xu, Xiangjun Tang, Xien Chen, Siyu Tang, Zhebin Zhang, Chen Li, and Xiaogang Jin. 2024a. Portrait3d: Text-guided high-quality 3d portrait generation using pyramid representation and gans prior. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–12.

Jun Xiang, Xuan Gao, Yudong Guo, and Juyong Zhang. 2024. Flashavatar: High-fidelity head avatar with eficient gaussian embedding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1802–1812.

Yuelang Xu, Benwang Chen, Zhe Li, Hongwen Zhang, Lizhen Wang, Zerong Zheng, and Yebin Liu. 2024. Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 1931–1941.

Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang. 2021. Gan prior embedded network for blind face restoration in the wild. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 672–681.

Zhenhui Ye, Ziyue Jiang, Yi Ren, Jinglin Liu, Jinzheng He, and Zhou Zhao. 2023. Geneface: Generalized and high-fidelity audio-driven 3d talking face synthesis. arXiv preprint arXiv:2301.13430 (2023).

Zongsheng Yue, Jianyi Wang, and Chen Change Loy. 2024. Eficient difusion model for image restoration by residual shifting. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

Kwan Yun, Chaelin Kim, Hangyeul Shin, and Junyong Noh. 2025. FFaceNeRF: Few-shot Face Editing in Neural Radiance Fields. In Proceedings of the Computer Vision and Pattern Recognition Conference. 10825–10835.

Kwan Yun, Kwanggyoon Seo, Chang Wook Seo, Soyeon Yoon, Seongcheol Kim, Soohyun Ji, Amirsaman Ashtari, and Junyong Noh. 2024. Stylized Face Sketch Extraction via Generative Prior with Limited Data. (2024).

Dongbin Zhang, Yunfei Liu, Lijian Lin, Ye Zhu, Kangjie Chen, Minghan Qin, Yu Li, and Haoqian Wang. 2025. HRAvatar: High-Quality and Relightable Gaussian Head Avatar. In Proceedings of the Computer Vision and Pattern Recognition Conference. 26285–26296.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Efectiveness of Deep Features as a Perceptual Metric. In CVPR.

Yuhong Zhang, Hengsheng Zhang, Xinning Chai, Zhengxue Cheng, Rong Xie, Li Song, and Wenjun Zhang. 2024. Dif-restorer: Unleashing visual prompts for difusion based universal image restoration. arXiv preprint arXiv:2407.03636 (2024).

Xiaochen Zhao, Hongyi Xu, Guoxian Song, You Xie, Chenxu Zhang, Xiu Li, Linjie Luo, Jinli Suo, and Yebin Liu. 2025. X-nemo: Expressive neural motion reenactment via disentangled latent attention. arXiv preprint arXiv:2507.23143 (2025).

Mingwu Zheng, Hongyu Yang, Di Huang, and Liming Chen. 2022b. Imface: A nonlinear 3d morphable face model with implicit neural representations. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 20343–20352.

Xiaozheng Zheng, Chao Wen, Zhaohu Li, Weiyi Zhang, Zhuo Su, Xu Chang, Yang Zhao, Zheng Lv, Xiaoyuan Zhang, Yongjie Zhang, et al. 2025. Headgap: Few-shot 3d head avatar via generalizable gaussian priors. In 2025 International Conference on 3D Vision (3DV). IEEE, 946–957.

Yufeng Zheng, Victoria Fernández Abrevaya, Marcel C Bühler, Xu Chen, Michael J Black, and Otmar Hilliges. 2022a. Im avatar: Implicit morphable head avatars from videos. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 13545–13555.

Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J Black, and Otmar Hilliges. 2023. Pointavatar: Deformable point-based head avatars from videos. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 21057–21067.

Xiangyu Zhu, Zhen Lei, Xiaoming Liu, Hailin Shi, and Stan Z Li. 2016. Face alignment across large poses: A 3d solution. In Proceedings of the IEEE conference on computer vision and pattern recognition. 146–155.

Wojciech Zielonka, Timo Bolkart, and Justus Thies. 2023. Instant volumetric head avatars. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4574–4584.