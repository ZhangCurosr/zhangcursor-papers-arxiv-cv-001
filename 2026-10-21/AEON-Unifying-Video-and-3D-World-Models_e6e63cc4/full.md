# AEON: Unifying Video and 3D World Models

Weiqi Zhang

Wenyuan Zhang

Yu-Shen Liu<sup>†</sup>

Junsheng Zhou<sup>†</sup>

School of Software, Tsinghua University, Beijing, China

{zwq23, zhou-js24}@mails.tsinghua.edu.cn liuyushen@tsinghua.edu.cn zhangweny98@gmail.com

![](images/8ee6df8dd1a7cde5ea273c8e644543621a9dffac55546bfe1d759dbb5e42f1ed.jpg)  
Figure 1: AEON generates 4DGS worlds that support real-time rendering at different timesteps and viewpoints. The 4DGS worlds are also ready for explicit editing and simulation (bottom).

## Abstract

Current world models for explicit world simulation typically follow two distinct pathways: (1) video generative models (e.g., Sora), which simulate realistic appearances but fail to preserve 3D consistency; and (2) 3D generative models (e.g., Marble), which generate geometrically consistent scenes but often yield static, less realistic observations. We present AEON, a unified framework that bridges video-based and 3D-based world models and combines the benefits of both. The key insight of AEON lies in a generative model that operates within the text- or image-conditioned video latent space while producing renderable 4D graphics primitives—specifically, 4D Gaussian Splatting (4DGS)—as the explicit world representation. We train a 4D reconstruction model to predict both 4DGS and camera poses from monocular videos, serving as our initialized 4D decoder. Furthermore, we design a domain adapter that translates generative video latents into reconstructive 4D latents through a distillation process on the video VAE. AEON naturally supports video-based world modeling by rendering 4DGS at predicted camera

trajectories, and constructs a persistent 4D environment with scene dynamics for realistic world simulation. Consequently, AEON empowers a diverse range of world modeling applications and achieves state-of-the-art performance on standard benchmarks. Project Page: https: // weiqi-zhang. github. io/ AEON .

## 1 Introduction

World modeling has rapidly advanced to the forefront of AI research, driven by its immense potential in embodied AI, autonomous driving, and game design. Despite this progress, there remains a lack of consensus on the optimal representation for explicit world simulation, leaving the field divided between two dominant paradigms: video-based [26, 38] and 3D-based world models [45, 14]. As highlighted in Table 1, these approaches possess orthogonal strengths and suffer from fundamental, paradigm-specific bottlenecks. Consequently, a more powerful and unified representation is imperative to achieve comprehensive world modeling.

We begin by asking a fundamental question: What constitutes a good world model? To train robots to understand physical laws and perceive scenes, a world model should exhibit the following properties: (1) generating realistic appearances that align with real-world domains; (2) modeling dynamic scene evolution (e.g., fire, waves) and moving objects; (3) maintaining 3D consistent environments free from geometric ambiguity; (4) ensuring scene persistence for continuous and repeatable simulation; and (5) supporting real-time exploration to facilitate efficient interaction. We argue that a representation unifying video- and 3D-based world models, thereby benefiting from their complementary strengths, is the key to achieving such an ideal world model.

We present AEON, a generative world model that introduces 4D Gaussian Splatting (4DGS) [41, 53] as its foundational representation, unifying video and 3D models to benefit from both paradigms. 4DGS represents scenes as a set of explicit graphics primitives localized in spatial geometry, and models scene dynamics through the velocity-driven movement and time-varying visibility of these primitives. Consequently, a learned 4DGS representation provides two independent yet crucial capabilities: it models world appearances through direct rendering along user-defined camera trajectories, and it establishes a persistent 4D environment that supports highly controllable 3D simulations. Additionally, it enables explicit modeling of world geometry and physical states, including depth, normals, motion, and optical flow.

Despite the comprehensive capabilities inherent to the 4DGS representation, most existing approaches restrict its application to reconstruction tasks, and generating a persistent 4DGS world directly from text or image conditions remains largely unexplored. In AEON, we propose a novel framework that integrates the generative latent space of a pretrained video Diffusion Transformer (DiT) [38, 29] with the reconstructive latent space of a 4D reconstruction model. Specifically, we train a 4D Gaussian Reconstruction Model (4DRM) to jointly predict 4DGS attributes and camera poses from monocular videos. By utilizing a large-scale annotated 4D video dataset and pretrained geometric priors from MapAnything [17], a scalable geometry model, the 4DRM develops robust reconstructive capabilities to recover world geometry and dynamics. A primary challenge, however, is the significant domain gap between the video and 4DRM latent spaces. We address this by introducing a domain adapter that seamlessly bridges the two. Finally, we employ a multi-stage training pipeline that optimizes the adapter and fine-tunes the reconstruction model through end-to-end distillation from the video model, supervised by video VAE-decoded outputs.

Consequently, AEON unlocks an extensive array of world modeling applications, including text/image/video-to-4DGS generation, 3D world generation, robotics simulation, physical state estimation (e.g., depth, normals, and optical flow), scene editing, and novel view and timestep synthesis.

Our main contributions can be summarized as follows:

• We introduce AEON, a unified world modeling framework that represents worlds as explicit 4D Gaussians, effectively bridging video-based and 3D-based models to benefit from their complementary strengths.

• We propose a novel methodology that integrates the generative latent space of a pretrained video DiT with the reconstructive latent space of a 4D Gaussian reconstruction model. To achieve this, we train a specialized domain adapter through an end-to-end distillation of the video model.

Table 1: Comparison of different world modeling paradigms across fundamental characteristics.
<table><tr><td>Paradigm</td><td>Realistic</td><td>3D Consistent</td><td>Dynamic</td><td>Persistence</td><td>Real-time Exp.</td></tr><tr><td>Video WM (e.g. Sora)</td><td>√</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>3D WM (e.g. Marble)</td><td>X</td><td>√</td><td>X</td><td>√</td><td>√</td></tr><tr><td>AEON</td><td>J</td><td>L</td><td>√</td><td>L</td><td>L</td></tr></table>

• AEON demonstrates comprehensive versatility, natively supporting a diverse spectrum of world modeling tasks, and establishes new state-of-the-art results on standard benchmarks.

## 2 Related Work

## 2.1 Video and 3D World Models

World modeling aims to generate explorable and realistic environments from conditional inputs (e.g., text or images), serving as a foundation for embodied AI and autonomous systems. Current paradigms primarily rely on video-based [26, 11, 32, 19, 52, 38] or 3D-based [45, 35, 51, 14, 33, 3, 10] approaches. Within the video-centric domain, JEPA-style methods [2, 1] excel at forecasting future latent states but lack explicit visual synthesis. Conversely, video world models [26, 38, 19] and their action-conditioned variants [5, 44, 36, 37] can synthesize photorealistic frames, yet they struggle with 3D consistency and incur heavy computational costs due to massive per-frame token generation.

Alternatively, 3D world models [45, 14] construct explicit environments for simulation. Recent works have explored panoramic generation [35, 51], progressive expansion [14], or leveraging video priors [3]. However, these methods typically employ static representations such as 3DGS [18] or meshes, precluding the modeling of world dynamics and limiting their ability to scale on dynamic internet video datasets. The same holds for 3D generation and reconstruction more broadly, which mostly build on static Gaussians [62, 59, 58, 56, 47] or distance fields [63, 64, 25, 24]. To bridge this gap, we propose AEON, a framework that generates renderable 4D Gaussian Splatting [53, 41] (4DGS) primitives as an explicit world representation. AEON effectively unifies the strengths of both paradigms, achieving dynamic modeling with 3D consistency while overcoming their respective limitations.

## 2.2 4D Gaussian Splatting from Videos

4D Gaussian Splatting [53, 41] advances the success of 3D Gaussian Splatting [18] for modeling scene dynamics. 4DGS learns from multi-view videos to predict Gaussian primitives with explicit temporal attributes such as lifespan, velocity, and time-varying opacity, allowing Gaussians to appear, move, and vanish independently. However, predicting 4DGS from monocular videos remains significantly more challenging due to limited observational data. While optimization-based methods [39, 20, 34] can achieve faithful visual appearances, they require lengthy fitting processes and struggle to generalize or scale. More recently, feed-forward frameworks [31, 48, 50, 22, 60] have been proposed for fast and scalable video-to-4DGS reconstruction, including bullet-time reconstruction at arbitrary timestamps [21] and online reconstruction from uncalibrated video streams [46]. Some of these works predict deformable 3DGS [22, 28] as their foundational representation, which often falls short when modeling complex and diverse dynamics. Related feed-forward models also recover dynamic geometry and camera poses from monocular videos [8]. All of these methods reconstruct scenes from observed videos. Native 4DGS generation has also been explored for animated objects [57]. Closer to our setting, Diff4Splat [28] and Lyra [3] build Gaussian scenes on top of video diffusion priors, but both require a camera trajectory as input and represent dynamics with deformable 3DGS and per-frame 3DGS, respectively. AEON differs in both respects: it generates a world from a text prompt or a single image without camera input, and it represents the world as one persistent set of 4D primitives, each with its own reference time, duration, and velocity, so that content can appear, move, and vanish independently and can be rendered at any continuous timestep.

![](images/389ef85d03904d7c98afd797db5ea45daa5bf792dd961528333e0e56e38464d2.jpg)  
Figure 2: Overview of AEON. Top (1). The 4D Reconstruction Model (4DRM) decodes a monocular video into a 4DGS world through four task-specific heads that respectively predict depth, Gaussian appearance, dynamic evolution, and camera poses. The resulting primitives are rasterized into RGB, depth, and normal maps for end-to-end supervision. Bottom (2). We train AEON for generative world modeling by distilling the world generative capability of a frozen video diffusion model (e.g., Wan 2.2 [38]) and the 4D reconstructive capability of a teacher 4DRM via a Domain Adapter and a learnable student 4DRM. The generated explicit 4DGS world enables versatile world modeling applications.

## 3 Method

We present AEON, a generative world model that operates within the latent space of a pretrained video diffusion model while producing explicit 4D Gaussian Splatting (4DGS) as the world representation. Given a text prompt or conditioning image, our goal is to generate a set of 4D Gaussians encoding both appearance and dynamics, enabling photorealistic rendering and persistent 3D simulation. The overview of AEON is shown in Fig. 2. We begin by introducing the 4DGS representation in Sec. 3.1, which models dynamic scenes through velocity-driven motion and time-varying visibility. In Sec. 3.2, we present the 4D Gaussian Reconstruction Model, whose four task-specific heads jointly predict depth, Gaussian appearance, dynamic evolution, and camera poses from monocular videos. We then describe our domain adapter in Sec. 3.3, which bridges the generative video latent space and the reconstructive 4D latent space through distillation. Finally, Sec. 3.4 details our multi-stage training strategy that optimizes the full pipeline end-to-end.

## 3.1 4D Gaussian Splatting as the World Representation

An ideal world model should jointly capture realistic appearances, scene dynamics, and 3D-consistent geometry within a persistent and efficiently renderable representation. To this end, AEON adopts 4D Gaussian primitives [41, 53, 60] as its foundational scene representation, where each primitive simultaneously encodes geometry, appearance, and temporal evolution. Formally, a single primitive $g _ { i }$ is characterized by the following parameter set

$$
g _ { i } = \{ \mathbf { x } _ { i } , \mathbf { s } _ { i } , \mathbf { r } _ { i } , o _ { i } , \mathbf { c } _ { i } , t _ { i } , l _ { i } , \mathbf { v } _ { i } , \omega _ { i } \} .\tag{1}
$$

Specifically, $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ denotes the spatial center, $\mathbf { s } _ { i } \in \mathbb { R } ^ { 2 }$ parameterizes anisotropic scaling along the principal axes, and $\mathbf { r } _ { i } \in \mathbb { R } ^ { 4 }$ encodes the orientation as a unit quaternion. The base opacity $o _ { i } \in [ 0 , 1 ]$ together with the color $\mathbf { c } _ { i } \in \mathbb { R } ^ { 3 }$ jointly determine the visual appearance of each primitive.

In addition to these spatial attributes, four temporal parameters govern the dynamic behavior of each primitive. The reference timestep $t _ { i } \in [ 0 , T ]$ specifies the moment of peak activation, while the duration $l _ { i } \in \mathbb { R } ^ { + }$ controls the temporal extent over which the primitive remains visible. The linear velocity $\mathbf { v } _ { i } \in \mathbb { R } ^ { 3 }$ and the angular velocity $\boldsymbol { \omega } _ { i } \in \mathbb { R } ^ { 3 }$ in axis-angle form capture translational and rotational motion, respectively. This decomposition into explicit spatial and temporal components enables AEON to model both persistent scene structure and transient dynamic phenomena within a unified representation.

## 3.2 4D Gaussian Reconstruction Model

While 4DGS offers an ideal world representation, directly generating 4D Gaussians from diverse conditions is highly challenging without proper guidance and geometric initialization. To address this, we first train a 4D Gaussian Reconstruction Model (4DRM) as a foundation, focusing on the more tractable task of world recovery from monocular videos. In a single feed-forward pass, 4DRM uses four task-specific heads to jointly predict depth, camera poses, and 4D Gaussian attributes of appearance and evolution. The model is trained end-to-end via differentiable rendering. Once trained, 4DRM facilitates AEON’s generative world modeling in two ways: (1) it provides a reconstructive foundation to initialize the student 4DRM that recovers the world from video latents, and (2) it acts as a teacher model, guiding the student to acquire 4D Gaussian modeling capabilities through distillation.

Encoding. Given an input video $V = \{ I _ { t } \} _ { t = 1 } ^ { T }$ of $T$ frames, we extract per-frame dense features $\mathbf { F } _ { t } \in \mathbb { R } ^ { \bar { H / { p } } \times W / { p } \times D }$ using a pretrained DINOv2 encoder [27]. To inject temporal identity, we add a sinusoidal positional embedding of the timestep index t to each spatial feature map. The resulting per-frame tokens are concatenated across all frames and processed by a transformer backbone of $L$ layers that alternate between intra-frame and cross-frame self-attention, balancing computational efficiency with long-range temporal modeling. The output features $\mathbf { F } ^ { S } \in \mathbb { R } ^ { T H W / p ^ { 2 } \times D }$ encode holistic spatio-temporal context for subsequent decoding. We initialize 4DRM from the pretrained weights of MapAnything [17], a feed-forward metric 3D reconstruction model.

Decoding. We decode the shared spatio-temporal representation $\mathbf { F } _ { t } ^ { S }$ through four task-specific heads, each responsible for a complementary component of the 4DGS world. The Depth Head predicts dense per-frame depth maps $\mathbf { D } _ { t } \in \mathbb { R } ^ { \mathbf { \breve { H } } \times W }$ , while the Camera Head regresses the intrinsics $\mathbf { K } _ { t }$ together with per-frame rotations $\mathbf { R } _ { t }$ (unit quaternions) and translations $\mathbf { t } _ { t }$ . The predicted depths are unprojected via $\mathbf { K } _ { t }$ <sub>t</sub> into the Gaussian centers $\mathbf { x } _ { i } ,$ anchoring all downstream predictions in a metrically grounded 3D space. Conditioned on these centers, the Gaussian Head predicts the appearance attributes $\left\{ \mathbf { s } _ { i } , \mathbf { r } _ { i } , o _ { i } , \mathbf { c } _ { i } \right\}$ (scale, orientation, opacity, and color), and the Evolution Head predicts the dynamic attributes $\{ t _ { i } , l _ { i } , \mathbf { v } _ { i } , \omega _ { i } \}$ (temporal reference, duration, and linear and angular velocities) that govern how each primitive moves and fades over time. This decomposition lets the Gaussian and Evolution Heads focus exclusively on appearance and dynamics, while the Camera Head’s joint pose estimation enables 4DRM to operate directly on raw monocular videos, removing the dependency on pre-estimated poses or external Structure-from-Motion (SfM) pipelines required by prior work [48, 22]. As a result, 4DRM recovers a complete dynamic world state from a single monocular video and establishes a robust reconstructive latent space for the subsequent generative integration.

## 3.3 Generative World Modeling

Overview. AEON is designed to generate persistent and explorable 4D worlds from diverse conditions, such as text prompts and images. The complete pipeline is illustrated in the lower portion of Fig. 2. Given a condition (e.g., a text prompt), we leverage a pretrained video diffusion model (e.g., Wan 2.2) as the generative foundation to produce a video latent, $\scriptstyle \mathcal { Z } _ { \mathrm { V i d e o } }$ . Given that z<sub>video</sub> encapsulates profound world-modeling priors capable of decoding into realistic, high-quality videos, we aim to unlock its full potential by recovering an explicit and persistent 4DGS world. We introduce a Domain Adapter that performs latent translation, mapping the generative video latent $\scriptstyle \mathcal { Z } _ { \mathrm { V i d e o } }$ to a reconstructive 4D latent z . A learnable student 4DRM then utilizes $z _ { \mathrm { 4 D } }$ to explicitly recover the 4DGS world. A simpler alternative would decode $\tilde { \mathcal { Z } } _ { \mathrm { V i d e o } }$ into frames with the VAE and pass them to the Stage-1 4DRM. We avoid this cascade because the 4DRM is trained on real videos, whereas decoded generations carry VAE and generation artifacts outside that distribution, so decoding errors would compound with reconstruction errors.

A primary challenge in connecting the video generative model to the 4D reconstruction model is the significant domain gap between their latent representations. The video latent is optimized for generative denoising, while the 4DRM consumes DINOv2 [27] tokens that are trained for geometric and semantic understanding. These two feature spaces therefore occupy fundamentally different distributions. Naively replacing the DINOv2 patch embedding with a randomly initialized projection would forfeit the geometric prior carried by the pretrained backbone and force the student 4DRM to relearn its feature distribution from scratch. To avoid this, we introduce a learnable domain adapter that bridges the two latent spaces while preserving the geometric capabilities of the pretrained 4DRM backbone layers.

Dual-Path Domain Adapter. We implement the domain adapter as a dual-path architecture that couples a linearly initialized convolutional branch with an expressive Perceiver Bridge. To establish a robust baseline, we first solve a closed-form ridge regression to map the video latents to the target DINOv2 activations at each candidate insertion depth ℓ. This linear mapping already recovers a coarse alignment, which makes it a good initialization, although on its own it falls well short of the full adapter (Sec. 4.4). Motivated by this, our dual-path design employs a linear branch to provide a stable foundation, while an expressive residual branch captures complex nonlinear transformations. Specifically, the linear branch L is instantiated as a strided 3D convolution initialized with the closed-form ridge solution. The residual branch N comprises a Perceiver Bridge module, detailed below. A learnable scalar gate γ fuses the two branches as:

$$
Z = \mathbb { L } ( z _ { \mathrm { v i d e o } } ) + \gamma \cdot \mathbb { N } ( z _ { \mathrm { v i d e o } } ) ,\tag{2}
$$

where γ is initialized to zero. Consequently, the adapter’s optimization begins entirely from the stable linear solution and progressively incorporates the nonlinear residual branch as $\gamma$ updates via gradient descent, ensuring stable optimization from the outset.

Perceiver Bridge. Implementing the residual branch with standard convolutions inherently dilutes fine-grained geometric information due to their restricted local receptive fields. Instead, we design a highly expressive Perceiver [15] Bridge module that leverages a learned token-to-token retrieval with a global receptive field. The video latent is first projected into the target feature dimension by a spatio-temporal Conv3D. An asymmetric temporal upsampler then restores the original temporal resolution by bypassing the independently encoded first frame and applying sub-pixel shuffle to the remaining compressed tokens so that the output length matches the original video exactly. The upsampled tokens are passed into a stack of Perceiver-style cross-attention blocks, where a set of learnable spatial queries attend to the projected video tokens as key-value pairs. Both queries and keys carry fixed 2D sinusoidal positional embeddings, enabling explicit spatial alignment without any resolution-specific learned parameters. Each frame is processed independently through the stack, confining the attention cost to the two-dimensional spatial grid while temporal modeling is handled by the preceding Conv3D and upsampler. A final LayerNorm aligns the output distribution with that of the DINOv2 tokens.

Token-Level Injection. The adapter output Z is integrated into the DINOv2 backbone at layer ℓ by bypassing the patch embedding and the initial ℓ transformer blocks, utilizing Z as the input tokens for the remaining blocks. Notably, these subsequent DINOv2 transformer blocks are fused into the 4DRM as part of the learnable spatio-temporal ViT. The selection of ℓ requires a careful trade-off: integrating the features too early imposes an excessive optimization burden on the adapter, whereas a late integration sacrifices essential geometric features relied upon by the downstream heads. Therefore, we set ℓ to the shallowest layer where the linear branch yields a reliable initialization. This strategy maximizes the number of retained upper blocks, effectively incorporating DINOv2’s geometric knowledge into the 4DRM.

We train the student 4DRM and the domain adapter with both distillation losses and end-to-end rendering supervisions. The full training procedure is detailed in Sec. 3.4.

## 3.4 Training Strategy

We adopt a multi-stage pipeline that first builds a strong reconstructive backbone and then bridges the generative video latent space through generative distillation and end-to-end rendering refinement.

Stage 1: Reconstruction Model Training. The 4DRM learns to map monocular video to 4D Gaussian primitives through large-scale training on posed videos. From each training clip of T frames, we select K frames as network input and predict a unified set of 4D Gaussians. These Gaussians are rendered back to all $T$ frames via the differentiable rasterizer, yielding dense photometric supervision that extends well beyond the temporal coverage of the input. Our objective consists of three complementary terms that respectively enforce appearance fidelity, geometric grounding, and dynamic disentanglement.

Appearance fidelity. We supervise rendered frames $\tilde { \mathbf { I } } _ { t }$ against ground-truth observations $\mathbf { I } _ { t }$ with a composite loss that balances pixel accuracy and perceptual quality:

$$
\mathcal { L } _ { \mathrm { a p p } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \lVert \mathbf { I } _ { t } - \tilde { \mathbf { I } } _ { t } \rVert _ { 2 } ^ { 2 } + \alpha \cdot \mathrm { L P I P S } ( \mathbf { I } _ { t } , \tilde { \mathbf { I } } _ { t } ) + \beta \cdot \left( 1 - \mathrm { S S I M } ( \mathbf { I } _ { t } , \tilde { \mathbf { I } } _ { t } ) \right) \right) .\tag{3}
$$

The perceptual and structural terms prevent convergence to overly smooth solutions and encourage sharp, realistic textures in the rendered output.

Geometric grounding. Monocular video alone cannot resolve metric depth ambiguity. We therefore supervise predicted geometry directly with ground-truth depth and normal annotations from our training dataset. Specifically, we render per-frame depth $\tilde { \mathbf { d } } _ { t }$ and normals $\tilde { \mathbf { n } } _ { t }$ from the predicted 4DGS and minimize their discrepancy with the annotated counterparts $\mathbf { d } _ { t } ^ { * }$ and $\mathbf { n } _ { t } ^ { * }$ :

$$
\mathcal { L } _ { \mathrm { g e o } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \mu \| \mathbf { d } _ { t } ^ { * } - \tilde { \mathbf { d } } _ { t } \| _ { 2 } ^ { 2 } + \nu \| \mathbf { n } _ { t } ^ { * } - \tilde { \mathbf { n } } _ { t } \| _ { 2 } ^ { 2 } \right) .\tag{4}
$$

Unlike prior methods that rely on pseudo labels from monocular expert models, training against metric ground truth provides an unambiguous supervisory signal and anchors all predictions in a consistent world-scale coordinate frame.

Dynamic disentanglement. Without multi-view constraints, the network may collapse to a degenerate solution where every primitive is assigned a short lifespan, leading the representation to per-frame point clouds. To encourage proper separation of static structure from dynamic content, we penalize unnecessary motion and favor long-lived primitives:

$$
\mathcal { L } _ { \mathrm { d y n } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left( \eta _ { 1 } \| \mathbf { v } _ { i } \| _ { 1 } + \eta _ { 2 } \| \omega _ { i } \| _ { 1 } + \eta _ { 3 } l _ { i } ^ { - 1 } \right) ,\tag{5}
$$

where M is the total number of predicted Gaussians. These penalties ensure that non-trivial velocity and short lifespans are assigned only to genuinely dynamic elements, enabling the model to learn meaningful motion decomposition without explicit motion labels.

The complete Stage 1 objective combines all three components:

$$
\mathcal { L } _ { \mathrm { s 1 } } = \mathcal { L } _ { \mathrm { a p p } } + \mathcal { L } _ { \mathrm { g e o } } + \mathcal { L } _ { \mathrm { d y n } } .\tag{6}
$$

Stage 2: Generative Distillation. In the second stage, we connect the generative video latent space to the reconstructive 4D space through end-to-end distillation. We freeze the pretrained 4DRM as a teacher and train the domain adapter alongside a student 4DRM. The teacher takes ground-truth RGB frames as input, while the domain adapter and student 4DRM operate solely on video latents. The training objective drives the student to reproduce the teacher’s 4DGS world reconstruction in a generative manner from the video latents. The overall distillation loss combines three complementary terms:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = \mathcal { L } _ { \mathrm { 4 D G S } } + \mathcal { L } _ { \mathrm { g e o m } } + \lambda ( t ) \mathcal { L } _ { \mathrm { f e a t } } .\tag{7}
$$

4D Gaussian attribute alignment. L<sub>4DGS</sub> applies a weighted L1 loss between the per-primitive attributes predicted by the teacher and the student, covering spatial position, anisotropic scale, rotation, opacity, color, velocity, and lifespan. This term provides the most direct form of knowledge transfer by enforcing primitive-level correspondence in attribute space.

Dense geometric alignment. Attribute-space supervision alone is insufficient since distinct sets of Gaussian attributes can produce visually equivalent renderings, while small attribute deviations may cause large visual discrepancies. To address this ambiguity, $\mathcal { L } _ { \mathrm { g e o m } }$ supervises the per-view dense geometric predictions from the student 4DRM heads, combining a robust log-depth regression loss, a multi-scale depth gradient term for boundary sharpness, and an L1 penalty on predicted camera parameters. We apply a global scale normalization before computing this loss to eliminate the trivial scale ambiguity between the two models.

Feature alignment. ${ \mathcal { L } } _ { \mathrm { f e a t } }$ regularizes the domain adapter by aligning the student’s intermediate features at the bridge output with the corresponding teacher activations through an L1 loss. Its weight λ(t) follows a cosine decay schedule that imposes strong regularization in early training to prevent catastrophic drift of the adapter and gradually relaxes to let the task-specific losses dominate.

Together, these objectives supervise the distillation process at complementary granularities. The attribute loss effectively instills the teacher’s 4D reconstruction capabilities into the student, while the geometric and feature losses simultaneously drive domain adaptation by forcing the adapter to project the video latent space onto the reconstructive 4D latent space.

Stage 3: End-to-End Rendering Refinement. Distillation losses operate in attribute and feature space and do not directly reflect final rendering quality. We therefore fine-tune the domain adapter and student 4DRM with an end-to-end rendering loss, where pseudo ground-truth frames are obtained by decoding $z _ { \mathrm { v i d e o } }$ through the frozen VAE decoder and compared against the rasterized 4DGS output using the same appearance objective as Stage 1. Gradients flow through the differentiable rasterizer into the full pipeline, so the student is optimized for rendering quality directly rather than only imitating the teacher. A linearly decayed distillation regularizer preserves the geometric structure acquired in Stage 2.

## 4 Experiments

![](images/6edb4aa617aa8aade894c7b982081d5ee8d407ff02396a83abf96c24bda08dbd.jpg)  
Figure 3: Qualitative reconstruction comparison on Dycheck and DaVis.

## 4.1 4D Generation

We evaluate AEON against 4DNex [7] and Lyra [3] for 4D generation. Unlike 4DNex (per-frame point clouds) and Lyra (per-frame 3DGS), AEON predicts a unified 4DGS representation for realtime exploration. We evaluate on 30 scenes, each specified by an image-text prompt. Evaluation videos, rendered along shared camera trajectories from identical image-text prompts, are assessed via VBench [13] metrics: I2V Subject, Subject Consistency, Motion Smoothness, and Aesthetic Quality. As shown in Table 2, AEON outperforms all baselines. While per-frame reconstruction lacks cross-frame coherence, AEON models scenes via persistent 4D Gaussians with explicit velocity and duration, naturally ensuring temporal consistency. Additionally, AEON generalizes across backbones (Wan 2.1 and Wan 2.2) without architectural modifications, yielding consistently strong results.

Table 2: Quantitative comparison on 4D generation. All outputs are rendered along shared preset camera trajectories and scored with VBench [13] metrics.
<table><tr><td>Method</td><td>I2V_Subject ↑</td><td>Subject_Consistency ↑</td><td>Motion_Smoothness ↑</td><td>Aesthetic_Quality ↑</td></tr><tr><td>4DNex [7]</td><td>0.796</td><td>0.790</td><td>0.968</td><td>0.342</td></tr><tr><td>Lyra [3]</td><td>0.865</td><td>0.864</td><td>0.974</td><td>0.459</td></tr><tr><td>AEON (Wan 2.1)</td><td>0.912</td><td>0.864</td><td>0.976</td><td>0.519</td></tr><tr><td>AEON (Wan 2.2)</td><td>0.913</td><td>0.875</td><td>0.980</td><td>0.531</td></tr></table>

## 4.2 Video-to-4D Reconstruction

We evaluate AEON for 4D reconstruction on Dycheck [9] and DaVis [30] against SOTA baselines: L4GM [31], 4DGT [48], MoVies [22], NeoVerse [50], and StreamSplat [46]. Unlike 4DGT and MoVies, which rely on external poses (from GT or π<sup>3</sup> [42]), AEON’s integrated camera head jointly predicts camera parameters and 4DGS attributes from monocular RGB. StreamSplat additionally requires monocular depth from Depth Anything V2 [49], which AEON does not need. For Dycheck’s novel view synthesis, we assess our video-to-4DGS pipeline by encoding input video into the VAE latent space and decoding it via the domain adapter and student reconstruction model.

As summarized in Table 3, AEON achieves superior PSNR, SSIM, and LPIPS across both benchmarks. Despite operating through the video VAE latent bottleneck, AEON’s domain adapter effectively preserves the geometric details necessary for highfidelity reconstruction, with especially pronounced gains on complex DaVis sequences. Further-

Table 3: Video-to-4D reconstruction comparison.
<table><tr><td rowspan="3">Method</td><td colspan="3">Dycheck</td><td colspan="3">DaVis</td></tr><tr><td>PSNR↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>L4GM [31]</td><td>8.673</td><td>0.354</td><td>0.652</td><td>20.589</td><td>0.667</td><td>0.311</td></tr><tr><td>4DGT [48]</td><td>12.267</td><td>0.413</td><td>0.542</td><td>20.362</td><td>0.667</td><td>0.313</td></tr><tr><td>MoVies [22]</td><td>11.840</td><td>0.341</td><td>0.545</td><td>20.736</td><td>0.688</td><td>0.307</td></tr><tr><td>NeoVerse [50]</td><td>14.269</td><td>0.363</td><td>0.501</td><td>23.929</td><td>0.735</td><td>0.283</td></tr><tr><td>StreamSplat [46]</td><td>11.876</td><td>0.375</td><td>0.532</td><td>23.748</td><td>0.712</td><td>0.273</td></tr><tr><td>AEON</td><td>14.487</td><td>0.421</td><td>0.469</td><td>24.661</td><td>0.771</td><td>0.243</td></tr><tr><td>Teacher 4DRM</td><td>14.413</td><td>0.418</td><td>0.462</td><td>24.886</td><td>0.775</td><td>0.241</td></tr></table>

more, AEON needs neither camera poses nor depth at inference, and only Stage 1 uses annotated videos, so the later training stages can scale to raw videos. Qualitative comparisons in Fig. 3 further corroborate these quantitative trends.

## 4.3 Novel Trajectory Rendering

We collect 30 in-the-wild videos unseen during training and render each scene along several novel camera trajectories. We compare against TrajectoryCrafter [54], RecamMaster [4], and NeoVerse [50], reporting six VBench [13] metrics in Table 4. AEON achieves the best scores on four out of six metrics while remaining competitive on the other two. Since rendering a new trajectory from 4DGS is geometrically exact rasterization rather than learned hallucination, AEON naturally maintains strong consistency and visual quality across viewpoint changes.

Table 4: Novel trajectory rendering comparison under VBench [13] metrics.
<table><tr><td>Method</td><td>Subj. Consist. ↑</td><td>Back. Consist. ↑</td><td>Temp. Flick. ↑</td><td>Motion Smooth. ↑</td><td>Aesth. Quality ↑</td><td>Imag. Quality ↑</td></tr><tr><td>TrajectoryCrafter [54]</td><td>94.48</td><td>96.57</td><td>95.49</td><td>98.79</td><td>64.81</td><td>58.53</td></tr><tr><td>RecamMaster [4]</td><td>94.49</td><td>95.58</td><td>97.64</td><td>97.79</td><td>61.26</td><td>56.34</td></tr><tr><td>NeoVerse [50]</td><td>90.24</td><td>95.21</td><td>95.01</td><td>97.39</td><td>59.17</td><td>57.98</td></tr><tr><td>AEON</td><td>94.64</td><td>96.77</td><td>96.69</td><td>98.11</td><td>69.02</td><td>58.65</td></tr></table>

## 4.4 Ablation Studies

We ablate three key design choices of AEON under the Dycheck and DaVis benchmarks in Table 5. Conv3D projection replaces the full dual-path domain adapter with a single Conv3D layer, showing that the expressive Perceiver Bridge branch is

Table 5: Ablation study on Dycheck and DaVis.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Dycheck</td><td colspan="3">DaVis</td></tr><tr><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td>Conv3D projection</td><td>12.086</td><td>0.314</td><td>0.606</td><td>12.813</td><td>0.276</td><td>0.701</td></tr><tr><td>W/o Distill</td><td>12.390</td><td>0.349</td><td>0.537</td><td>16.718</td><td>0.503</td><td>0.489</td></tr><tr><td>W/o Duration</td><td>12.500</td><td>0.372</td><td>0.535</td><td>22.057</td><td>0.718</td><td>0.328</td></tr><tr><td>Full</td><td>14.487</td><td>0.421</td><td>0.469</td><td>24.661</td><td>0.771</td><td>0.243</td></tr></table>

essential for capturing the complex mapping between the two latent spaces. W/o Distill removes the distillation losses and trains the adapter with rendering supervision alone, resulting in a significant performance drop, especially on DaVis, which highlights the importance of the teacher 4DRM as a geometric foundation. W/o Duration disables the temporal opacity evolution of 4DGS, forcing all primitives to remain visible throughout the sequence; the degradation shows that explicit duration modeling of 4DGS representation is critical for handling transient dynamics. The full model combines all components and achieves the best performance on every metric.

## 5 Conclusion

We present AEON, a unified world modeling framework that bridges video-based and 3D-based paradigms and combines the benefits of both. The core of AEON is a generative model that operates within the latent space of a pretrained video diffusion model while producing explicit 4D Gaussian Splatting as the world representation. We train a 4DRM for robust geometric guidance and design a domain adapter that translates generative video latents into the reconstructive 4D latents through end-to-end distillation. Experiments across 4D generation, dynamic-scene reconstruction, and novel trajectory rendering demonstrate that AEON achieves state-of-the-art performance while supporting a broad spectrum of world modeling applications within a single unified framework.

Limitations and future work. The temporal extent of generated 4D scenes is currently bounded by the video diffusion model’s sequence length. Because AEON stores the world as a persistent 4DGS rather than as frames, it offers a natural path beyond this window: render conditioning frames from the current 4DGS at the latest camera pose, generate the next window, decode it into new primitives, and fuse them with the existing world where they overlap. Since each step is anchored to explicit geometry rather than extrapolated from previous frames, drift should accumulate more slowly than in video-only autoregression. In addition, regions never observed in the generated video cannot be recovered, so very large viewpoint changes may reveal holes that a generative completion module could fill.

## Acknowledgments and Disclosure of Funding

This work was partially supported by Deep Earth Probe and Mineral Resources Exploration – National Science and Technology Major Project (2024ZD1003405) and the National Natural Science Foundation of China (62272263).

## References

[1] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[3] Sherwin Bahmani, Tianchang Shen, Jiawei Ren, Jiahui Huang, Yifeng Jiang, Haithem Turki, Andrea Tagliasacchi, David B. Lindell, Zan Gojcic, Sanja Fidler, Huan Ling, Jun Gao, and Xuanchi Ren. Lyra: Generative 3D scene reconstruction via video diffusion model selfdistillation. In International Conference on Learning Representations (ICLR), 2026.

[4] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. ReCamMaster: Camera-controlled generative rendering from a single video. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[5] Jake Bruce, Michael D. Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In International Conference on Machine Learning (ICML), 2024.

[6] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey

Tulyakov. Panda-70M: Captioning 70M videos with multiple cross-modality teachers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[7] Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng Tao, He Zhu, Fangzhou Hong, Liang Pan, and Ziwei Liu. 4DNeX: Feed-forward 4D generative modeling made easy. arXiv preprint arXiv:2508.13154, 2025.

[8] Juntong Fang, Zequn Chen, Weiqi Zhang, Donglin Di, Xuancheng Zhang, Chengmin Yang, and Yu-Shen Liu. MoRe: Motion-aware feed-forward 4D reconstruction transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[9] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[10] Hyojun Go, Dominik Narnhofer, Goutam Bhat, Prune Truong, Federico Tombari, and Konrad Schindler. Text-to-3D by stitching a multi-view reconstruction network to a video generator. In International Conference on Learning Representations (ICLR), 2026.

[11] Google DeepMind. Veo 2. https://deepmind.google/technologies/veo/veo-2/, 2024.

[12] Yuan-Ting Hu, Jiahong Wang, Raymond A. Yeh, and Alexander G. Schwing. SAIL-VOS 3D: A synthetic dataset and baselines for object detection and 3D mesh reconstruction from video data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[13] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[14] Team HY-World, Chenjie Cao, Xuhui Zuo, Zhenwei Wang, Yisu Zhang, Junta Wu, Zhenyang Liu, Yuning Gong, Yang Liu, Bo Yuan, et al. HY-World 2.0: A multi-modal world model for reconstructing, generating, and simulating 3D worlds. arXiv preprint arXiv:2604.14268, 2026.

[15] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International Conference on Machine Learning (ICML), 2021.

[16] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. DynamicStereo: Consistent dynamic depth from stereo videos. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[17] Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, et al. MapAnything: Universal feed-forward metric 3D reconstruction. In International Conference on 3D Vision (3DV), 2026.

[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 42(4):139:1–139:14, 2023.

[19] Kuaishou. Kling AI. https://klingai.kuaishou.com/, 2024.

[20] Jiahui Lei, Yijia Weng, Adam W. Harley, Leonidas Guibas, and Kostas Daniilidis. MoSca: Dynamic Gaussian fusion from casual videos via 4D motion scaffolds. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[21] Hanxue Liang, Jiawei Ren, Ashkan Mirzaei, Antonio Torralba, Ziwei Liu, Igor Gilitschenski, Sanja Fidler, Cengiz Oztireli, Huan Ling, Zan Gojcic, and Jiahui Huang. Feed-forward bullet-time reconstruction of dynamic scenes from monocular videos. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[22] Chenguo Lin, Yuchen Lin, Panwang Pan, Yifan Yu, Tao Hu, Honglei Yan, Katerina Fragkiadaki, and Yadong Mu. MoVieS: Motion-aware 4D dynamic view synthesis in one second. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[23] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andrés Bruhn. Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[24] Takeshi Noda, Chao Chen, Weiqi Zhang, Xinhai Liu, Yu-Shen Liu, and Zhizhong Han. MultiPull: Detailing signed distance functions by pulling multi-level queries at multi-step. In Advances

in Neural Information Processing Systems (NeurIPS), 2024.

[25] Takeshi Noda, Chao Chen, Junsheng Zhou, Weiqi Zhang, Yu-Shen Liu, and Zhizhong Han. Learning bijective surface parameterization for inferring signed distance functions from sparse point clouds with grid deformation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[26] OpenAI. Video generation models as world simulators. https://openai.com/index/ video-generation-models-as-world-simulators/, 2024.

[27] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR), 2024.

[28] Panwang Pan, Chenguo Lin, Jingjing Zhao, Chenxin Li, Yuchen Lin, Haopeng Li, Honglei Yan, Kairun Wen, Yunlong Lin, Yixuan Yuan, et al. Diff4Splat: Controllable 4D scene generation with latent dynamic reconstruction models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[30] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[31] Jiawei Ren, Kevin Xie, Ashkan Mirzaei, Hanxue Liang, Xiaohui Zeng, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, and Huan Ling. L4GM: Large 4D Gaussian reconstruction model. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[32] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. GEN3C: 3D-informed worldconsistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[33] Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, et al. Lyra 2.0: Explorable generative 3D worlds. In SIGGRAPH Asia, 2026.

[34] Yang-Tian Sun, Yi-Hua Huang, Lin Ma, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. Splatter a Video: Video Gaussian representation for versatile processing. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[35] HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, Wenhuan Li, Sheng Zhang, et al. HunyuanWorld 1.0: Generating immersive, explorable, and interactive 3D worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025.

[36] InSpatio Team, Donghui Shen, Guofeng Zhang, Haomin Liu, Haoyu Ji, Hujun Bao, Hongjia Zhai, Jialin Liu, Jing Guo, Nan Wang, et al. INSPATIO-WORLD: A real-time 4D world simulator via spatiotemporal autoregressive modeling. arXiv preprint arXiv:2604.07209, 2026.

[37] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[38] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[39] Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of Motion: 4D reconstruction from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[40] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. TartanAir: A dataset to push the limits of visual SLAM. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020.

[41] Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng, and Xiaowei Zhou. FreeTimeGS: Free Gaussian primitives at anytime anywhere for dynamic scene reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[42] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry

learning. In International Conference on Learning Representations (ICLR), 2026.

[43] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing (TIP), 13(4):600–612, 2004.

[44] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, et al. Matrix-Game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

[45] World Labs. Marble: A multimodal world model. https://www.worldlabs.ai/blog/ marble-world-model, 2025.

[46] Zike Wu, Qi Yan, Xuanyu Yi, Lele Wang, and Renjie Liao. StreamSplat: Towards online dynamic 3D reconstruction from uncalibrated video streams. In International Conference on Learning Representations (ICLR), 2026.

[47] Peng Xiang, Liang Han, Hui Zhang, Yu-Shen Liu, and Zhizhong Han. VGGS: VGGT-guided Gaussian splatting for efficient and faithful sparse-view surface reconstruction. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), 2026.

[48] Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard Newcombe, and Zhaoyang Lv. 4DGT: Learning a 4D Gaussian transformer using real-world monocular videos. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[49] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything V2. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[50] Yuxue Yang, Lue Fan, Ziqi Shi, Junran Peng, Feng Wang, and Zhaoxiang Zhang. NeoVerse: Enhancing 4D world model with in-the-wild monocular videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[51] Zhongqi Yang, Wenhang Ge, Yuqi Li, Jiaqi Chen, Haoyuan Li, Mengyin An, Fei Kang, Hua Xue, Baixin Xu, Yuyang Yin, et al. Matrix-3D: Omnidirectional explorable 3D world generation. arXiv preprint arXiv:2508.08086, 2025.

[52] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations (ICLR), 2025.

[53] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4D Gaussian splatting. In International Conference on Learning Representations (ICLR), 2024.

[54] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. TrajectoryCrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[55] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[56] Wenyuan Zhang, Jimin Tang, Weiqi Zhang, Yi Fang, Yu-Shen Liu, and Zhizhong Han. MaterialRefGS: Reflective Gaussian splatting with multi-view consistent material inference. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[57] Weiqi Zhang, Junsheng Zhou, Juntong Fang, Kanle Shi, Shenkun Xu, and Yu-Shen Liu. TESLA: Native 4D Gaussian splatting generation with temporally structured latents. In Advances in Neural Information Processing Systems (NeurIPS), 2026.

[58] Weiqi Zhang, Junsheng Zhou, Haotian Geng, Kanle Shi, Shenkun Xu, Yi Fang, and Yu-Shen Liu. GaussianGrow: Geometry-aware Gaussian growing from 3D point clouds with text guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

[59] Weiqi Zhang, Junsheng Zhou, Haotian Geng, Wenyuan Zhang, and Yu-Shen Liu. GAP: Gaussianize any point clouds with text guidance. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

[60] Weiqi Zhang, Junsheng Zhou, Xuancheng Zhang, Juntong Fang, Zequn Chen, Donglin Di, and Yu-Shen Liu. GAMMA: Scalable 4D Gaussian reconstruction model for novel view synthesis of monocular videos. In Advances in Neural Information Processing Systems (NeurIPS), 2026.

[61] Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J. Guibas. PointOdyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[62] Junsheng Zhou, Weiqi Zhang, and Yu-Shen Liu. DiffGS: Functional Gaussian splatting diffusion. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[63] Junsheng Zhou, Weiqi Zhang, Baorui Ma, Kanle Shi, Yu-Shen Liu, and Zhizhong Han. UDiFF: Generating conditional unsigned distance fields with optimal wavelet diffusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[64] Junsheng Zhou, Weiqi Zhang, Baorui Ma, Kanle Shi, Yu-Shen Liu, and Zhizhong Han. UDFStudio: A unified framework of datasets, benchmarks and generative models for unsigned distance functions. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026.

[65] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, Mingyu Liu, Dingning Liu, Jiange Yang, Zhoujie Fu, Junyi Chen, Chunhua Shen, Jiangmiao Pang, Kaipeng Zhang, and Tong He. OmniWorld: A multi-domain and multi-modal dataset for 4D world modeling. In International Conference on Learning Representations (ICLR), 2026.

## A Implementation Details

The 4DRM encoder is built on DINOv2-Large [27] with 24 transformer blocks of dimension 1024. The 4DRM is initialized from the pretrained MapAnything weights [17]. The domain adapter is injected at layer ℓ=4, retaining the upper 20 pretrained transformer blocks as a geometric foundation for the 4DRM. The Linear branch projects the 16-channel Wan video latent to 1024 channels, initialized via closed-form ridge regression against the fourth-layer DINOv2 activations. The Perceiver Bridge employs 4 cross-attention blocks with 16 heads and 1024 learnable queries, and a temporal interleave of factor 4 inverts the video VAE’s temporal compression. All three stages are optimized with AdamW using a weight decay of $1 0 ^ { - 4 }$ , a cosine schedule with 2 000 linear warmup steps, a base learning rate of $5 \times 1 0 ^ { - 6 }$ , and gradient clipping at a global norm of 1.0.

Stage 1 trains the full 4DRM for 20 epochs on video clips with ground-truth camera poses and depth annotations. The appearance loss weights are $\alpha { = } 2 . 0$ and $\beta { = } 0 . 2$ , the geometry weights are $\mu { = } 1 . 0$ and $\nu { = } 0 . 5$ , and the dynamic regularization coefficients are $\eta _ { 1 } { = } \eta _ { 2 } { = } \eta _ { 3 } { = } 0 . 3$ . Stage 2 freezes the Stage-1 model as the teacher and trains the domain adapter and the student 4DRM for 30 additional epochs with a 10× learning rate multiplier on the adapter parameters. Since the frozen teacher provides all supervisory signals through distillation, Stage 2 requires only raw RGB videos as input, removing the dependency on pose and depth annotations and enabling training on large-scale unannotated data. The feature alignment weight λ decays from 1.0 to 0 following a cosine schedule. All experiments are conducted on 16 NVIDIA H200 GPUs. The Stage-1 training corpus combines DynamicReplica [16], SAILVOS3D [12], Spring [23], TartanAir [40], PointOdyssey [61], and OmniWorld [65], each providing ground-truth camera poses and depth. Because these datasets differ in size by orders of magnitude, we resample the smaller ones by the factors listed in Table 6, so that every source contributes to geometric supervision. In Stage 2, we further incorporate Panda-70M [6] to expand the training distribution beyond annotated domains; it supplies real-world appearance diversity without geometric labels. Stage 3 continues to train the domain adapter and student 4DRM for 10 additional epochs with the end-to-end rendering loss, using the same optimizer configuration as Stage 2. The distillation regularization weight $\lambda _ { \mathrm { r e g } }$ is initialized to 0.5 and linearly decayed to zero over the course of training.

Table 6: Training datasets. Resample is the factor by which each dataset is repeated during training.
<table><tr><td>Dataset</td><td>Dynamic</td><td>Pose</td><td>Depth</td><td>Real</td><td>#Scenes</td><td>Resample</td></tr><tr><td>PointOdyssey [61]</td><td>√</td><td>√</td><td>√</td><td>x</td><td>0.1K</td><td>100×</td></tr><tr><td>TartanAir [40]</td><td>√</td><td>√</td><td>√</td><td>X</td><td>0.5K</td><td>50×</td></tr><tr><td>DynamicReplica [16]</td><td>√</td><td>√</td><td>√</td><td>x</td><td>0.5K</td><td>20×</td></tr><tr><td>OmniWorld-Game [65]</td><td>√</td><td>√</td><td>√</td><td>x</td><td>0.06K</td><td>100×</td></tr><tr><td>SAIL-VOS 3D [12]</td><td>√</td><td>√</td><td>√</td><td>x</td><td>0.16K</td><td>50×</td></tr><tr><td>Spring [23]</td><td>√</td><td>√</td><td>√</td><td>x</td><td>0.03K</td><td>1000×</td></tr><tr><td>Panda-70M [6] (subset)</td><td>√</td><td>x</td><td>x</td><td>√</td><td>2.4M</td><td>1×</td></tr></table>

## B Applications

Text/Image-to-world generation. As illustrated in Fig. 1, AEON synthesizes high-quality 4D Gaussian Splatting (4DGS) environments from text or image prompts. This is achieved by first generating video latents using Wan-DiT conditioned on the input prompts. These latents are subsequently translated into 4D spatial-temporal latents and decoded to reconstruct the final 4DGS world.

Video-to-world reconstruction. For video-to-world reconstruction, input videos are encoded into video latents using Wan-VAE. These latents are then processed to generate 4DGS scenes, following the same pipeline as our text/image-to-world generation.

Novel view synthesis. The generated 4DGS representations inherently support real-time novel view synthesis across arbitrary viewpoints and timesteps.

Physical status estimation. The generated 4DGS encapsulates rich geometric and physical properties, which can be directly extracted to derive depth maps, normal maps, motion masks, and optical flow. Specifically, motion masks are obtained by filtering the lifespans of the Gaussians, as dynamic elements typically correspond to Gaussians with shorter lifespans.

3D world generation. Beyond dynamic scene generation, AEON is also highly effective for static   
3D world generation. An example of this capability is provided in the third row of Fig. 1.

Scene editing and simulation. Because AEON generates explicit 4D representations, it inherently facilitates seamless scene editing and structural decomposition for downstream simulations (shown in the bottom row of Fig. 1). By leveraging the motion masks derived from Gaussian lifespan filtering, we can cleanly separate a generated scene into its static and dynamic components. These isolated parts from various generated 4DGS worlds can then be freely recomposed to simulate novel scenarios. The same explicit structure is what embodied simulation needs. The metric geometry of the static component can supply occupancy and free space for an agent, the dynamic primitives move according to their own velocities, and real-time rasterization renders egocentric observations at every step. An agent can therefore be placed in a generated world, roll out its actions, and receive rendered observations in a closed loop, and because the world is persistent, the same scenario can be replayed for repeatable evaluation.

## C Evolution of 4DGS Representation

Temporal opacity evolution. To faithfully represent the appearance and disappearance of scene elements over time, the opacity of each primitive is modulated by a Gaussian kernel centered at its reference timestep $t _ { i }$ . Given a query time t, the effective opacity is computed as

$$
\sigma _ { i } = \sqrt { \frac { ( l _ { i } / 2 ) ^ { 2 } } { - 2 \log ( \epsilon ) } } , \qquad o _ { i } ( t ) = o _ { i } \cdot \exp \left( - \frac { ( t - t _ { i } ) ^ { 2 } } { 2 \sigma _ { i } ^ { 2 } } \right) ,\tag{8}
$$

where $\epsilon = 0 . 0 5$ serves as the activation threshold at the duration boundary. This temporal gating mechanism ensures full visibility near the reference timestep while producing smooth fade-in and fade-out transitions. As a result, the representation can naturally accommodate transient phenomena such as fire, splashes, and object entrances without resorting to explicit per-object tracking.

Velocity-driven spatial evolution. The position and orientation of each primitive evolve over time according to a constant-velocity motion model. At query time t, the updated position and rotation are given by

$$
\begin{array} { r } { \mathbf { x } _ { i } ( t ) = \mathbf { x } _ { i } + \mathbf { v } _ { i } \cdot ( t - t _ { i } ) , } \end{array}\tag{9}
$$

$$
\mathbf { r } _ { i } ( t ) = \mathbf { r } _ { i } \otimes \mathrm { A x i s A n g l e 2 Q u a t } ( \boldsymbol { \omega } _ { i } \cdot ( t - t _ { i } ) ) ,\tag{10}
$$

where $\otimes$ denotes quaternion multiplication. The scaling s remains fixed across time, ensuring that each primitive preserves its intrinsic shape throughout the sequence. Factoring translational and rotational dynamics into independent velocity terms yields a compact yet expressive motion parameterization that supports rigid-body motion while incurring minimal per-primitive overhead. Combined with the temporal opacity mechanism above, the full 4DGS representation enables AEON to unify the photorealistic rendering capability of video models with the geometric consistency and persistence of 3D models.

## D Evaluation Metrics

We describe all evaluation metrics used in the main paper. Our experiments rely on two families of metrics: per-frame reconstruction metrics that measure pixel-level fidelity and VBench metrics that assess perceptual video quality along multiple complementary dimensions.

Reconstruction metrics. We evaluate dynamic scene reconstruction with three standard metrics that operate at increasing levels of perceptual abstraction. Peak Signal-to-Noise Ratio (PSNR) measures per-pixel reconstruction accuracy in decibels:

$$
\mathrm { P S N R } = 1 0 \cdot \log _ { 1 0 } \frac { R ^ { 2 } } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \mathbf { I } _ { i } - \tilde { \mathbf { I } } _ { i } ) ^ { 2 } } ,\tag{11}
$$

where R is the maximum pixel intensity, N is the total number of pixels, and I, <sup>˜</sup>I denote the groundtruth and rendered frames respectively. Higher values indicate less distortion. Structural Similarity Index (SSIM) [43] moves beyond pixel-level comparison by jointly evaluating local luminance, contrast, and structural patterns:

$$
\mathrm { S S I M } ( \mathbf { I } , \tilde { \mathbf { I } } ) = \frac { ( 2 \mu _ { \mathbf { I } } \mu _ { \tilde { \mathbf { I } } } + C _ { 1 } ) ( 2 \sigma _ { \mathbf { I } \tilde { \mathbf { I } } } + C _ { 2 } ) } { ( \mu _ { \mathbf { I } } ^ { 2 } + \mu _ { \tilde { \mathbf { I } } } ^ { 2 } + C _ { 1 } ) ( \sigma _ { \mathbf { I } } ^ { 2 } + \sigma _ { \tilde { \mathbf { I } } } ^ { 2 } + C _ { 2 } ) } ,\tag{12}
$$

where $\mu$ and σ denote the local mean and standard deviation computed over sliding windows, $\sigma _ { \tilde { \mathbf { I } } \tilde { \mathbf { I } } }$ is the local cross-covariance, and $C _ { 1 } , C _ { 2 }$ are small stability constants. SSIM ranges from 0 to 1, with higher values indicating greater structural agreement. Learned Perceptual Image Patch Similarity (LPIPS) [55] further captures high-level perceptual differences by comparing deep features extracted from a pretrained network:

$$
\mathrm { L P I P S } ( \mathbf { I } , \tilde { \mathbf { I } } ) = \sum _ { l } w _ { l } \left\| \phi _ { l } ( \mathbf { I } ) - \phi _ { l } ( \tilde { \mathbf { I } } ) \right\| _ { 2 } ^ { 2 } ,\tag{13}
$$

where $\phi _ { l }$ denotes the normalized activations at layer l and w<sub>l</sub> are learned linear weights. Lower LPIPS values indicate higher perceptual similarity.

VBench video quality metrics. For 4D generation and novel trajectory rendering we adopt six metrics from the VBench [13] benchmark, each targeting a distinct aspect of video quality. I2V Subject measures the visual fidelity between the conditioning image and the generated subject, quantifying how faithfully the output preserves the identity of the input. Subject Consistency evaluates the temporal coherence of the foreground subject across frames by computing pairwise feature similarity over the video. Background Consistency similarly measures the stability of background regions to detect unwanted scene drift. Temporal Flickering captures high-frequency luminance and color variations between consecutive frames, where higher scores correspond to smoother transitions. Motion Smoothness assesses whether object trajectories follow physically plausible paths without abrupt jumps. Aesthetic Quality and Imaging Quality respectively evaluate the overall visual appeal and the low-level image attributes such as sharpness and color fidelity. All VBench metrics are reported on a scale where higher values indicate better performance.

## E Additional Ablation Studies

Table 7 reports additional ablations on Dycheck and DaVis, where each variant changes one component of the full model.

Table 7: Additional ablation studies on Dycheck and DaVis.
<table><tr><td rowspan="2" colspan="2">Variant</td><td colspan="3">Dycheck</td><td colspan="3">DaVis</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td rowspan="3">Adapter</td><td>Conv3D projection</td><td>12.086</td><td>0.314</td><td>0.606</td><td>12.813</td><td>0.276</td><td>0.701</td></tr><tr><td>Conv branch only</td><td>12.332</td><td>0.333</td><td>0.533</td><td>15.409</td><td>0.465</td><td>0.533</td></tr><tr><td>Perceiver branch only</td><td>13.086</td><td>0.389</td><td>0.596</td><td>14.262</td><td>0.442</td><td>0.552</td></tr><tr><td>Training</td><td>W/o Stage 3</td><td>13.670</td><td>0.361</td><td>0.517</td><td>22.953</td><td>0.705</td><td>0.299</td></tr><tr><td rowspan="2">4DRM</td><td>W/o geometric prior</td><td>12.341</td><td>0.358</td><td>0.543</td><td>22.268</td><td>0.695</td><td>0.349</td></tr><tr><td>W/o  $\bar { \mathcal { L } } _ { \mathrm { g e o } }$ </td><td>12.260</td><td>0.331</td><td>0.578</td><td>20.347</td><td>0.655</td><td>0.323</td></tr><tr><td>Full</td><td></td><td>14.487</td><td>0.421</td><td>0.469</td><td>24.661</td><td>0.771</td><td>0.243</td></tr></table>

Adapter. Conv3D projection replaces the dual-path adapter with a single Conv3D projection, while Conv branch only and Perceiver branch only use only a convolutional branch or only the Perceiver branch, respectively. All three variants lower DaVis PSNR to 12.8–15.4 dB, so no single path can bridge the two latent spaces on its own.

Training. W/o Stage 3 stops after Stage 2 and keeps Stages 1 and 2 unchanged. Skipping Stage 3 reduces DaVis PSNR by 1.7 dB, demonstrating the benefit of end-to-end rendering refinement.

4DRM. W/o geometric prior trains 4DRM without the MapAnything initialization, and W/o $\mathcal { L } _ { g e o }$ removes the depth and normal supervision of Stage 1. Both variants perform worse than the full model on Dycheck and DaVis, so the quality of 4DRM matters, although performance degrades gracefully rather than collapsing.

## F Additional Qualitative Results

We provide additional qualitative reconstruction comparisons on the DaVis benchmark in Fig. 4. These results further corroborate that AEON recovers sharper appearance details and more temporally coherent geometry than baselines, particularly in regions with fast motion and transient occlusions.

![](images/26aea8eeff8c8f401101ddc511bd08fb12100caf2be8e4e498a87d1f734e03fc.jpg)  
Figure 4: Additional qualitative reconstruction comparisons on DaVis. AEON produces more coherent geometry and finer details in dynamic regions.