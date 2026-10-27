# FL4N: FEED-FORWARD LATENT MODEL FOR 4D NOVEL VIEW SYNTHESIS

Eunbeen Hong<sup>1</sup> Hyeonseo Yu<sup>1</sup> Honggyu An<sup>1</sup> Jooho Son<sup>1</sup> Hyunah Ko<sup>1</sup> Soowon Son<sup>1</sup> Chaehyun Kim<sup>1</sup> Jahyeok Koo<sup>1</sup> Jisoo Kim<sup>1</sup> Junhwa Hur<sup>2†</sup> Seungryong Kim<sup>1†</sup>

<sup>1</sup>KAIST AI <sup>2</sup>Google DeepMind https://cvlab-kaist.github.io/FL4N/

![](images/9efed431c83b0817c22e32a8fc8ce6af70f3c874cdf909fd155ecc3bd7ab8fa3.jpg)  
Figure 1: Teaser. FL4N is the first feed-forward latent 4D novel view synthesis model. It models dynamics in a latent space to render temporally coherent frames, whereas Gaussian-based methods such as 4DGT (Xu et al., 2025), MoVieS (Lin et al., 2026a), NeoVerse (Yang et al., 2026), and C4G (Kim et al., 2026c) suffer from temporal ghosting, artifacts, holes, or blur.

## ABSTRACT

4D novel view synthesis (NVS) aims to render dynamic scenes across viewpoints and timestamps, which requires jointly inferring 3D structure and temporal dynamics. Existing monocular feed-forward methods often rely on explicit 4D Gaussian representations, which are prone to temporal inconsistency and severe rendering artifacts. We present FL4N, the first feed-forward latent 4D NVS model. Instead of relying on explicit 4D primitives, FL4N encodes a 4D scene prior into a unified latent representation across space and time. Our core insight is that learning temporal dynamics as residuals on top of a pretrained static latent prior enables effective 4D modeling with limited dynamic training data. To this end, we introduce a lightweight dynamic encoder that leverages static scene features to disentangle object motion from camera-induced parallax. Trained solely with an image synthesis objective, FL4N achieves state-of-the-art 4D NVS quality while rendering in real time. Its learned latent representation generalizes to real-world videos and disentangles viewpoint from time.

## 1 INTRODUCTION

4D novel view synthesis (NVS) is a cornerstone of understanding and simulating our ever-changing physical world, enabling broad applications, e. g., immersive graphics (Li et al., 2025), world models (Zhu et al., 2025), robotics (Kim et al., 2026b), autonomous driving (Song et al., 2025), and 4D-aware vision-language models (Cho et al., 2026). Unlike static NVS (Hong et al., 2025; Jin et al., 2025), 4D NVS requires capturing both persistent 3D structure and temporal dynamics. While slow optimization-based methods (Pumarola et al., 2021; Wang et al., 2025b) and multi-view methods (Balice et al., 2026; Xu et al., 2024b) are impractical in the wild, monocular feed-forward approaches offer a practical, real-time solution by inferring a 4D scene from a casual video.

However, monocular feed-forward 4D NVS is highly ill-posed, as it must jointly infer entangled 3D structure and object motion from monocular observations alone (Wang et al., 2025b). This requires the model to learn a strong 4D prior. To make this tractable, existing methods (Lin et al., 2026a; Xu et al., 2025; Yang et al., 2026) rely on the explicit 3D inductive bias of 3D Gaussians (Kerbl et al., 2023) with added 3D motion and dense geometry and motion supervision. However, this reliance on explicit 3D primitives often bottlenecks 4D rendering quality. Inaccurate prediction of Gaussian attributes across timestamps often yields rendering artifacts, e. g., temporal ghosting, floaters, and holes (An et al., 2026; Kim et al., 2026c), as in Fig. 1. Moreover, time-varying effects (e. g., shadows and illumination changes) must be encoded in per-primitive attributes (Li et al., 2024), and regions with insufficient primitive coverage tend to be over-smoothed (Kim et al., 2026c; Li et al., 2024).

In recent static NVS literature, latent-based methods (Jin et al., 2025; Mitchel et al., 2025; Szymanowicz et al., 2026) avoid these limitations by learning view synthesis end-to-end with transformers, mapping input views directly to target views conditioned on a target camera, without an explicit 3D representation. With minimal 3D inductive biases and large-scale training, these latent models outperform 3D-Gaussian-based methods in visual fidelity (Jin et al., 2025; Kim et al., 2026a). However, directly extending the latent approach to dynamic scenes faces a fundamental challenge. Without explicit 3D priors, learning to disentangle 3D structure and temporal dynamics purely from data would require massive multi-view dynamic scene datasets (Bai et al., 2025; Van Hoorick et al., 2024), which are much scarcer than static datasets (Watson et al., 2025).

To resolve this challenge, we propose FL4N, a Feed-forward Latent-based 4D NVS model, which augments a frozen pretrained static latent NVS model with a trainable dynamic encoder. Our key insight is that a 4D latent model does not need to learn dynamic scene representation from scratch: pretrained static latent NVS models already possess a strong, generalizable 3D prior, allowing the dynamic encoder to exclusively focus on modeling temporal changes (e. g., object motion or appearance variation). Consequently, our method can be trained with a small amount of dynamic data while fully inheriting the pretrained model’s ability to synthesize coherent novel views.

Concretely, we attach the lightweight trainable dynamic encoder alongside the frozen static encoder, similar to ControlNet (Zhang et al., 2023). The dynamic encoder takes the same input views with a learnable per-view time token and cross-attends to frozen static encoder features to isolate true scene dynamics from camera-induced motion. A lightweight time module then modulates the residual latent for the queried timestamp, which is residually added to the static latent (He et al., 2016) and decoded by the frozen renderer at the target camera pose. Multi-view videos naturally disentangle viewpoint and temporal variation through same-time views and fixed views across time. Sampling unseen view–time pairs, FL4N learns this disentanglement through image synthesis with binary dynamic mask supervision, without dense geometry or motion supervision.

We evaluate our approach on synthetic and real datasets under a systematic protocol. To effectively demonstrate the performance of 4D novel view synthesis and the capability of view-time disentan glement, we separate the setting with novel view synthesis, cross-time view synthesis, and novel time synthesis. FL4N achieves state-of-the-art or competitive performance across datasets in the pose-free setting without test-time pose optimization, while optionally incorporating context cam era poses further enables state-of-the-art performance across all three settings.

Our main contributions are summarized as follows:

• First 4D latent NVS model. We present to jointly model 3D structure and temporal variation in a latent space, rendering novel views at novel timestamps without explicit 3D representations or target-view/time observations.

• A residual formulation for efficient dynamic adaptation. We formulate 4D NVS as learning a temporal residual on a frozen static representation. This leverages the pretrained scene and viewpoint priors while concentrating learning on temporal variation through a lightweight branch.

• State-of-the-art performance. We comprehensively benchmark 4D NVS across novel view, cross-time, and novel time synthesis on eight synthetic and real-world benchmarks, achieving state-of-the-art rendering quality on all three settings, with the fastest runtime.

## 2 RELATED WORK

Feed-forward Gaussian splatting for 4D novel view synthesis. Explicit feed-forward 4D methods typically represent scenes with 3D Gaussians extended with temporal dynamics, such as velocity (Xu et al., 2025; Yang et al., 2026) or deformation fields (Lin et al., 2026a;b). However, independently modeling motion per Gaussian often disrupts object rigidity and temporal consistency due to incoherent motion among nearby Gaussians (Kim et al., 2026c), yielding artifacts such as temporal ghosting. Anchoring Gaussian prediction to each target timestamp (Liang et al., 2025) can sidestep the issue but is susceptible to view bias by primarily relying on nearby frames while underutilizing observation from distant frames. Query-based Gaussian representation (Kim et al., 2026c) alleviates this via dense motion supervision, yet its limited number of queries constrains rendering quality due to insufficient Gaussian coverage, particularly for large outdoor scenes. Multi-view approaches (Balice et al., 2026) can exploit observations from multiple time-synchronized static cameras, but the setup is far more restrictive than a monocular moving camera. We introduce a feed-forward latent-based method with minimal geometric inductive bias in a practical but more challenging monocular camera setup.

Latent-based novel view synthesis. Latent-based NVS models synthesize novel views with transformers conditioned on the target pose (Jiang et al., 2025; Jin et al., 2025; Kim et al., 2026a; Sajjadi et al., 2022; Szymanowicz et al., 2026) or latent pose variable (Mitchel et al., 2025), bypassing explicit 3D representations. Although RayDer (Prestel et al., 2026) trains on dynamic videos, it still targets static NVS while suppressing object motion via per-view nuisance token that is dropped at inference. Despite their strong generalization, these methods either assume static scenes or treat dynamic content as variation to be ignored rather than modeled. Toward dynamic latent representa tions, DyST (Seitzer et al., 2024) extends RUST (Sajjadi et al., 2023) with a dynamics estimator, yet keeps its scene latent time-invariant. Temporal variation is decoded via control codes extracted from target-pose and target-time frames, which limits explicit view and time control and risks appearance leakage from these frames. In contrast, our model constructs a time-conditioned scene latent that i rendered from arbitrary target cameras, enabling synthesis at novel views and novel timestamps.

Generative novel view synthesis. NVS becomes inherently ambiguous when target views extend into regions unseen in the input views. Deterministic methods maintain strong cross-view consistency on observed content but tend to blur unobserved regions (Prestel et al., 2026; Szymanowicz et al., 2026). Generative approaches (Bai et al., 2025; Chen et al., 2026; Ren et al., 2025; Van Hoorick et al., 2024; Yu et al., 2025) can extrapolate plausible content beyond the observed regions, yet their stochastic sampling often compromises cross-view consistency and scene faithfulness (Jin et al., 2025; Kim et al., 2026c). Although generative refinement can post-process rendered views from Gaussian representations (Kim et al., 2026c; Yang et al., 2026) or serve as a diffusion decoder for latent-based model (Szymanowicz et al., 2026), our work focuses on deterministic NVS and remains orthogonal to such generative refinement directions.

## 3 METHOD

We propose FL4N, a Feed-forward Latent-based 4D NVS model. Given an N-frame monocular video $\{ \bar { \mathbf { I } } _ { k } \} _ { k = 1 } ^ { N }$ , our method synthesizes views from an arbitrary query view $\pi _ { \mathrm { t g t } }$ at target timestamp $t _ { \mathrm { t g t } }$ drawn from the observed time range. Figure 2 shows an overview of FL4N. We formulate 4D synthesis as a residual adaptation on top of a frozen pretrained static NVS model. Given the static encoder and the renderer from the pretrained model, we attach a lightweight dynamic encoder that focuses its capacity on isolating dynamic object motion.

In the following, we first review the latent static NVS baseline that forms a basis of our method (§ 3.1). We then introduce FL4N, describing the technical details on the dynamic encoder with target-time residual adaptation (§ 3.2). Finally, we detail the training strategy (§ 3.3) and conclude by analyzing the learned residual features (§ 3.4).

## 3.1 PRELIMINARIES: LATENT-BASED NOVEL VIEW SYNTHESIS

Latent-based static NVS methods (Jiang et al., 2025; Szymanowicz et al., 2026) encode multi-view images of a static scene into a latent scene representation and synthesize a novel view by decoding

![](images/9eba82f831c3b105946bed0e7c92f2fb6ea62551a7d402fee8ba90c1cd8182bb.jpg)  
Figure 2: FL4N Overview. $( L e f t )$ During training, context and target frames with varying views and timestamps are sampled from multi-view videos. (Right) A static encoder encodes static 3D structures (static latent) while a lightweight dynamic encoder extracts temporal residuals (residual latent) within the whole context frames. For 4D NVS, the residual latent is first adapted at the target time and added to the static latent resulting in a dynamic latent, which is then rendered into the novel view with target-view conditioned.

this latent conditioned on a target camera pose. Such a framework comprises a multi-view encoder ${ \mathcal E } _ { \mathrm { s t } }$ and a renderer R. Camera parameters are optional, conditioning the encoder when available and otherwise letting the model operate pose-free.

Given N input images $\{ \mathbf { I } _ { k } \} _ { k = 1 } ^ { N }$ and their optional corresponding cameras $\{ \pi _ { k } \} _ { k = 1 } ^ { N }$ , the static encoder produces the static scene latent $\mathbf { z } _ { \mathrm { s t } } \mathbf { \dot { \mathbf { \cdot } } }$

$$
\begin{array} { r } { \mathbf { z } _ { \mathrm { s t } } = \mathcal { E } _ { \mathrm { s t } } \big ( \{ \mathbf { x } _ { k } , \ \pi _ { k } \} _ { k = 1 } ^ { N } \big ) \in \mathbb { R } ^ { N \times P \times C } , \qquad \mathbf { x } _ { k } = \mathcal { P } _ { \mathrm { s t } } ( \mathbf { I } _ { k } ) , } \end{array}\tag{1}
$$

with $\mathcal { P } _ { \mathrm { s t } } ( \cdot )$ denotes the patchify layer, $P$ the number of image patches, and C the feature dimension. The renderer decodes the latent into a novel-view image $\hat { \mathbf { I } } = \mathcal { R } ( \mathbf { z } _ { \mathrm { s t } } , \pi _ { \mathrm { t g t } } )$ at a query camera $\pi _ { \mathrm { t g t } }$

## 3.2 FL4N: FEED-FORWARD LATENT MODEL FOR 4D NOVEL VIEW SYNTHESIS

Dynamic encoder. To learn to disentangle dynamic motion from static scene structure in monocular videos, we freeze the pretrained static encoder ${ \mathcal E } _ { \mathrm { s t } }$ and renderer ${ \mathcal { R } } ,$ and introduce a lightweight dynamic encoder ${ \mathcal { E } } _ { \mathrm { d y n } }$ . The static encoder ${ \mathcal E } _ { \mathrm { s t } }$ already captures time-invariant content and camerainduced viewpoint changes from its static-scene training. During our dynamic-scene training, the frozen static encoder and renderer allow the dynamic encoder $\bar { \mathcal { E } _ { \mathrm { d y n } } }$ to learn temporal variations as residuals to the pretrained static representation. Freezing the renderer R fixes latent-to-image mapping, forcing the residual feature to the pretrained latent space. This encourages them to complement the static representation rather than learn a new feature space from limited dynamic data.

To provide explicit temporal cues to the dynamic encoder $\mathcal E _ { \mathrm { d y n } } .$ , we introduce a learnable time token $\tau _ { k } \in \mathbb { R } ^ { C }$ for each context frame $\mathbf { I } _ { k } .$ , which is concatenated with the patch tokens $\mathbf { x } _ { k } = \mathcal { P } _ { \mathrm { d y n } } ( \mathbf { I } _ { k } )$ before being fed to the dynamic encoder $\mathcal { E } _ { \mathrm { d y n } }$ . Through this encoding, each time token encodes a timestamp-related information from each frame.

In the dynamic encoder, identifying dynamic regions requires separating object motion from apparent camera-induced motion. We therefore introduce static anchoring, which conditions the dynamic encoder on static feature through cross-attention. This allows it to factor out camera-induced changes captured in the static features and focus its representation on residual motion from dynamic objects. Each dynamic encoder block applies frame-wise self-attention $\mathrm { F S A } ( \cdot )$ , global self-attention $\mathrm { G S A } ( \cdot )$ , and cross-attention $\mathrm { C A } ( \cdot )$ sequentially,

$$
\mathbf { F } _ { \mathrm { r e s } } ^ { \ell }  \mathbf { F } _ { \mathrm { r e s } } ^ { \ell } + \mathrm { F S A } ^ { \ell } ( \mathbf { F } _ { \mathrm { r e s } } ^ { \ell } ) , \quad \mathbf { F } _ { \mathrm { r e s } } ^ { \ell }  \mathbf { F } _ { \mathrm { r e s } } ^ { \ell } + \mathrm { G S A } ^ { \ell } ( \mathbf { F } _ { \mathrm { r e s } } ^ { \ell } ) , \quad \mathbf { F } _ { \mathrm { r e s } } ^ { \ell + 1 } = \mathbf { F } _ { \mathrm { r e s } } ^ { \ell } + \mathrm { C A } ^ { \ell } ( \mathbf { F } _ { \mathrm { r e s } } ^ { \ell } , \mathbf { F } _ { \mathrm { s t } } ) ,\tag{2}
$$

where $\mathbf { F } _ { \mathrm { r e s } } ^ { \ell }$ denotes the residual features at the ℓ-th dynamic encoder block, $\mathbf { F } _ { \mathrm { s t } }$ the features from the static encoder, and $\operatorname { C A } ( X , Y )$ computes queries from X and keys and values from $Y$

Target-time residual adaptation. The residual latent and encoded time tokens $[ \mathbf { z } _ { \mathrm { r e s } } ; \hat { \tau } _ { 1 } , \hdots , \hat { \tau } _ { N } ] =$ $\mathcal { E } _ { \mathrm { d y n } } \big ( \{ \mathbf { I } _ { k } \} _ { k = 1 } ^ { N } \big ) \in \mathbb { R } ^ { N \times ( P \bar { + } 1 ) \times C }$ summarize temporal information across all $N$ context frames. To extract a residual specific to a target timestamp, we condition the residual latent $\mathbf { z } _ { \mathrm { r e s } }$ with target-time token $\hat { \tau } _ { \mathrm { t g t } } .$ . For an observed timestamp $t _ { \mathrm { t g t } } = t _ { k } ,$ , we directly use the corresponding token $\hat { \tau } _ { k }$ . For an intermediate, novel timestamp $t _ { \mathrm { t g t } } \in [ t _ { k } , \overline { { \ d } } t _ { k + 1 } ]$ , we linearly interpolate the two time tokens adjacent to the target timestamp $\left( { t _ { k } , t _ { k + 1 } } \right)$ , weighted by their temporal distance to the target.

$$
\hat { \tau } _ { \mathrm { { t g t } } } = \frac { \big ( t _ { k + 1 } - t _ { \mathrm { t g t } } \big ) \cdot \hat { \tau } _ { k } + \big ( t _ { \mathrm { t g t } } - t _ { k } \big ) \cdot \hat { \tau } _ { k + 1 } } { t _ { k + 1 } - t _ { k } } , \qquad t _ { \mathrm { t g t } } \in [ t _ { k } , t _ { k + 1 } ] .\tag{3}
$$

The interpolated token then provides the temporal condition for rendering at the novel timestamp.

Given $\hat { \tau } _ { \mathrm { t g t } }$ , a lightweight time module $\tau$ modulates the aggregated residuals $\mathbf { z } _ { \mathrm { r e s } }$ into the target-time residual $\mathbf { z } _ { \mathrm { r e s } } ^ { \mathrm { t g t } }$ . The time module $\tau$ consists of two transformer blocks with adaptive layer normalization (AdaLN) conditioned on $\hat { \tau } _ { \mathrm { t g t } }$ , followed by a residual gate implemented as a zero-initialized linear projection layer. This gate initializes the residual contribution to zero $( i . e . ,$ preserving the pretrained static prior at the start of training) and progressively learns temporal corrections. The target residual $\mathbf { z } _ { \mathrm { r e s } } ^ { \mathrm { t g t } }$ is then added to the static latent $\mathbf { z } _ { \mathrm { s t } }$ to obtain the target-time dynamic latent ${ \bf z } _ { \mathrm { d y n } } ^ { \mathrm { t g t } } \colon$

$$
\begin{array} { r } { \mathbf { z } _ { \mathrm { r e s } } ^ { \mathrm { t g t } } = \mathcal { T } ( \mathbf { z } _ { \mathrm { r e s } } ; \hat { \tau } _ { \mathrm { t g t } } ) , \qquad \mathbf { z } _ { \mathrm { d y n } } ^ { \mathrm { t g t } } = \mathbf { z } _ { \mathrm { s t } } + \mathbf { z } _ { \mathrm { r e s } } ^ { \mathrm { t g t } } . } \end{array}\tag{4}
$$

The residual $\mathbf { z } _ { \mathrm { r e s } }$ is a time-invariant representation, processed only once for the $N$ context frames by the dynamic encoder $\mathcal { E } _ { \mathrm { d y n } }$ . For multiple target timestamps, we reuse $\mathbf { z } _ { \mathrm { r e s } }$ and apply the lightweight time module $\tau$ with corresponding target-time tokens. The resulting target-specific latent is then fed to the frozen renderer $\bar { \mathcal { R } }$ to render an image at the query camera $\pi _ { \mathrm { t g t } } \colon$

$$
\hat { \mathbf { I } } = \mathcal { R } ( \mathbf { z } _ { \mathrm { d y n } } ^ { \mathrm { t g t } } , \pi _ { \mathrm { t g t } } ) .\tag{5}
$$

## 3.3 TRAINING

The rendered image <sup>ˆ</sup>I at the target view and time is supervised by an image reconstruction loss:

$$
{ \mathcal { L } } _ { \mathrm { i m a g e } } = { \mathcal { L } } _ { 2 } + \lambda _ { \mathrm { p e r c } } { \mathcal { L } } _ { \mathrm { p e r c } } + \lambda _ { \mathrm { p e r c } } ^ { \mathrm { d y n } } { \mathcal { L } } _ { \mathrm { p e r c } } ^ { \mathrm { d y n } } + \lambda _ { \mathrm { g r a d } } ^ { \mathrm { d y n } } { \mathcal { L } } _ { \mathrm { g r a d } } ^ { \mathrm { d y n } } ,\tag{6}
$$

where the L2 loss $\mathcal { L } _ { 2 }$ and the perceptual loss $\mathcal { L } _ { \mathrm { p e r c } }$ are computed over the full image, with an extra perceptual loss $\mathcal { L } _ { \mathrm { p e r c } } ^ { \mathrm { d y n } }$ and the gradient loss $\mathcal { L } _ { \mathrm { g r a d } } ^ { \mathrm { d y n } }$ on dynamic region. Since moving objects typically occupy a small fraction of the image, the full-image losses are dominated by static pixels and provide weak supervision for them. The dynamic-region losses provide stronger supervision for dynamic objects and yield sharper rendering of moving objects.

To help disentangle dynamic object motion from camera-induced motion without dense geometric or motion supervision, we indirectly supervise the dynamic encoder to locate dynamic regions. Specifically, we attach a DPT-based mask head (Ranftl et al., 2021) to the dynamic encoder to predict a binary dynamic-region mask $\hat { M } _ { k }$ , which is supervised with a pseudo label of dynamic mask $M _ { k }$ using a binary cross-entropy loss BCE(·),

$$
\mathcal { L } _ { \mathrm { m a s k } } = \mathrm { B C E } ( \hat { M } _ { k } , M _ { k } ) .\tag{7}
$$

The total training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { i m a g e } } + \lambda _ { \mathrm { m a s k } } \mathcal { L } _ { \mathrm { m a s k } } .\tag{8}
$$

Further details of training objective and pseudo-label generation are provided in Appendix A.

## 3.4 ANALYSIS ON RESIDUAL LATENT

We conduct analyses to investigate what information is captured in the residual latent $\mathbf { z } _ { \mathrm { r e s } } .$ . Additional analysis on static and residual latent is provided in Appendix C.

Residual complements static rendering. The frozen static model can answer where to render, but not when. Having no target-time input, the static latent $\mathbf { z } _ { \mathrm { s t } }$ compresses the video sequence into a single time-invariant prediction; moving content is thus discarded or frozen at observed frames with the input pose entangled, as in Fig. 3. Our residual latent $\mathbf { z } _ { \mathrm { r e s } }$ supplies this missing temporal axis. Conditioned on the target time, it complements the static latent with time-dependent variation, placing the dynamic object at its target-time position while preserving the static background.

Residual feature encodes temporal dynamics. Figure 4 compares the static latent $\mathbf { z } _ { \mathrm { s t } }$ and residual latent $\mathbf { z } _ { \mathrm { r e s } }$ using PCA visualization. While the static latent exhibits camera-ray-like patterns, the residual latent suppresses static regions and captures distinct motion across time, suggesting that the dynamic encoder effectively isolates temporal variation from the time-invariant scene representation.

![](images/2ed841d1634f3aacae09c80445a0be93b6055ca840f9d5ccf557e66d8ea5d5c3.jpg)

![](images/3f6b8e16578cb2eb002dfd649c9bb703ce41786c6c20364848720c9c3aff93a4.jpg)

Figure 3: Rendered frames from the static latent and our residual time correction. In dynamic scenes, the static latent suppresses dynamic content (left) or copies it from the nearest context frame (right), whereas our residual latent places the moving content at the target time while preserving the static background.  
![](images/5d8f8f5c9f7c7f95695d764775cecebb035c7c0767cf34fe9a6afd86a9b99c18.jpg)  
Figure 4: PCA visualization of static and residual latent. The static latent captures camera-ray-like structure, whereas the residual latent primarily attends to moving objects and exhibits temporal variation.

![](images/1f6bdf9eb19fa016516aad9059225615e75a68f8b0a997618df0eb348078ec4e.jpg)  
Figure 5: Visualization of residual gate activation. The residual gate magnitude is higher on dynamic objects while remaining suppressed in the background.

This temporal specialization is further supported by the learned residual gate. Figure 5 visualizes gate activation using the channel-wise L2 norm of the gated residual. The gate remains largely inactive in static regions but activates strongly around moving objects, indicating that temporal modeling capacity is concentrated where motion occurs.

## 4 EXPERIMENTS

Implementation details. Our model builds on a frozen static novel view synthesizer (Szymanowicz et al., 2026), with only the dynamic encoder, time module, and mask head trained. The dynamic encoder follows the architecture of the pretrained static encoder, with added cross-attention module for each layer but uses one-third of its depth, with 8 transformer layers compared to 24 in the static encoder, since temporal variation accounts for only a relatively small portion of the scene. Further architectural and training details are provided in Appendix A.

Training dataset. To effectively disentangle camera and object motion, we train our model using synchronized multi-view video datasets such as Kubric4D (Van Hoorick et al., 2024) and Camx-Time (Huang et al., 2026), captured by V cameras with fixed poses with T frames. Unlike costly, explicit 3D annotation (e. g., dense depth or 3D motion) that is commonly used for 3D-Gaussianbased approaches, the multi-view datasets provide cost-effective indirect 4D supervision from 2D images. Furthermore, our residual formulation specifically makes this RGB-only supervision directly tractable, with requiring relatively small amount as well.

![](images/28eaf6a1c6c946d9c821ca03bf036a109040aeaebe313bef18174877584c24ce.jpg)  
Novel View Synthesis (Table 1)

![](images/438150c19abe473b45a923c16386123a982908bd2d54d3017debfa379c2885db.jpg)

![](images/6f29cd05e25587c9658796a3002125b966aa40d98abca349f07f2cae51299461.jpg)  
Figure 6: Three evaluation settings given synchronized multi-view video datasets. Red outlines indicate input context frames, while blue outlines indicate target frames for evaluation.

Table 1: Quantitative results on Novel View Synthesis: FL4N outperforms all Gaussian-based methods on all benchmark datasets, showing the strength of our latent representation bypassing explicit primitives. Context pose input further improves the metrics, achieving the state of the art. Best, second best, and third best results are highlighted.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pose</td><td rowspan="2">PSNR↑</td><td colspan="3">NVIDIA</td><td colspan="3">N3DV</td><td colspan="3">Kubric4D</td><td colspan="3">CamxTime</td><td colspan="3">Charge</td></tr><tr><td>SSIM↑</td><td></td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑ LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td></td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Feed-forward Gaussian Representation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4DGT (Xu et al., 2025)</td><td>√</td><td>16.127</td><td>0.3884</td><td>0.5972</td><td>17.813</td><td>0.6609</td><td>0.5131</td><td>14.148</td><td>0.5027</td><td></td><td>0.6096</td><td>17.125</td><td>0.6557</td><td>0.4893</td><td>18.017</td><td>0.7786</td><td>0.5653</td></tr><tr><td>MoVieS (Lin et al., 2026a)</td><td>√</td><td>11.818</td><td>0.2546</td><td>0.6760</td><td>20.136</td><td>0.7045</td><td>0.4605</td><td>12.411</td><td>0.3550</td><td></td><td>0.6070</td><td>17.069</td><td>0.6223</td><td>0.4488</td><td>18.709</td><td>0.7473</td><td>0.5298</td></tr><tr><td>NeoVerse† (Yang et al., 2026)</td><td>x</td><td>15.050</td><td>0.4135</td><td>0.4626</td><td>12.953</td><td>0.4806</td><td>0.5697</td><td>8.954</td><td>0.2573</td><td></td><td>0.6721</td><td>10.432</td><td>0.3110</td><td>0.5905</td><td>15.224</td><td>0.5211</td><td>0.5919</td></tr><tr><td>C4G† (Kim et al., 2026c)</td><td>x</td><td>16.572</td><td>0.4182</td><td>0.6203</td><td>18.444</td><td>0.7050</td><td>0.5246</td><td>14.931</td><td>0.5656</td><td></td><td>0.6442</td><td>20.527</td><td>0.7056</td><td>0.4369</td><td>18.535</td><td>0.7797</td><td>0.5279</td></tr><tr><td>Feed-forward Latent-based 4D Representation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FL4N (Ours)</td><td>x</td><td>17.945</td><td>0.4113</td><td>0.5229</td><td>18.581</td><td>0.6898</td><td>0.4776</td><td>20.916</td><td>0.6433</td><td></td><td>0.3767</td><td>21.268</td><td>0.7237</td><td>0.2743</td><td>20.924</td><td>0.8217</td><td>0.4952</td></tr><tr><td>FL4N (Ours)</td><td>√</td><td>20.435</td><td>0.4835</td><td>0.3867</td><td>22.896</td><td>0.7716</td><td>0.3560</td><td>22.209</td><td>0.6737</td><td></td><td>0.3350</td><td>25.730</td><td>0.8270</td><td>0.1681</td><td>22.959</td><td>0.8324</td><td>0.4616</td></tr></table>

Table 2: On the challenging Cross-Time View Synthesis task, FL4N substantially outperforms all Gaussianbased methods on all benchmark datasets, even without test-time pose alignment (<sup>†</sup>), showing the model successfully learns 4D priors and decouples view and time in the latent space. Best, second best, and third best results are highlighted.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pose</td><td rowspan="2">PSNR↑</td><td colspan="3">NVIDIA</td><td colspan="3">N3DV</td><td colspan="3">Kubric4D</td><td colspan="3">CamxTime</td><td colspan="3">Charge</td></tr><tr><td>SSIM↑</td><td>LPIPS↓</td><td></td><td>PSNR↑ SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td></td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Feed-forward Gaussian Representation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4DGT (Xu et al., 2025)</td><td>√</td><td>15.937</td><td>0.3867</td><td>0.6052</td><td>16.097</td><td>0.6380</td><td>0.5483</td><td>15.680</td><td></td><td>0.5424</td><td>0.5786</td><td>17.024</td><td>0.6573</td><td>0.4939</td><td>17.939</td><td>0.7756</td><td>0.5617</td></tr><tr><td>MoVieS (Lin et al., 2026a)</td><td>√</td><td>15.785</td><td>0.4614</td><td>0.5328</td><td>19.710</td><td>0.7400</td><td>0.4414</td><td>16.988</td><td></td><td>0.6368</td><td>0.4174</td><td>17.340</td><td>0.6470</td><td>0.4224</td><td>20.113</td><td>0.7771</td><td>0.5084</td></tr><tr><td>NeoVerse† (Yang et al., 2026)</td><td>x</td><td>14.449</td><td>0.3819</td><td>0.4890</td><td>12.447</td><td>0.4729</td><td>0.5681</td><td>8.885</td><td>0.2256</td><td></td><td>0.6793</td><td>10.276</td><td>0.3091</td><td>0.5911</td><td>15.220</td><td>0.5380</td><td>0.5940</td></tr><tr><td>C4G† (Kim et al., 2026c)</td><td>x</td><td>16.403</td><td>0.4096</td><td>0.6304</td><td>17.039</td><td>0.6829</td><td>0.5374</td><td>15.270</td><td>0.5572</td><td></td><td>0.6348</td><td>20.175</td><td>0.7009</td><td>0.4453</td><td>18.410</td><td>0.7640</td><td>0.5324</td></tr><tr><td>Feed-forward Latent-based 4D Representation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FL4N (Ours) FL4N (Ours)</td><td>x √</td><td>21.552 21.838</td><td>0.5348 0.5578</td><td>0.3739 0.3562</td><td>24.550 25.109</td><td>0.8152 0.8284</td><td>0.3353 0.3234</td><td>23.547 24.272</td><td>0.7535 0.7850</td><td></td><td>0.2558 0.2304</td><td>24.546 27.648</td><td>0.8040 0.8732</td><td>0.2016 0.1325</td><td>24.222 24.421</td><td>0.8500 0.8496</td><td>0.4413 0.4382</td></tr></table>

Given a multi-view video ${ \mathcal { T } } = \{ I _ { v , t } ~ | ~ v ~ \in ~ \{ 1 , \cdots , V \} , t ~ \in ~ \{ 1 , \cdots , T \} \}$ , we adopt an asynchronous sampling strategy to ensure generalization to monocular video inputs at test time as well as accurate conditioning on arbitrary viewpoints and timestamps. We sample N distinct viewtime pairs $\{ ( v _ { i } , t _ { i } ) \} _ { i = 1 } ^ { N } \stackrel { \smile } { \sim } \mathcal { U } \left( \{ 1 , \cdots \stackrel { \smile } { , } V \} \times \stackrel { \cdot } { \{ 1 , \cdots , T \} } \right)$ ), then construct an asynchronous input set $\mathcal { T } _ { \mathrm { i n } } = \{ I _ { v _ { i } , t _ { i } } \} _ { i = 1 } ^ { N }$ . We then sample the target frames from the remaining set such that ${ \mathcal { T } } _ { \mathrm { t a r } } \subseteq { \mathcal { T } } \setminus { \mathcal { T } } _ { \mathrm { i n } }$ Holding the viewpoint fixed isolates temporal changes caused by dynamic objects, whereas fixing the timestamp provides observations from varying viewpoints, enabling supervision analogous to static multi-view synthesis. This setup naturally trains ours to disentangle temporal dynamics from viewpoint changes.

Evaluation setting. We evaluate three settings, novel view synthesis (NVS), cross-time view synthesis (CTVS), and novel time synthesis (NTS), illustrated in Fig. 6. NVS evaluates held-out viewpoints at input timestamps, CTVS evaluates unseen viewpoints-timestamps combinations from a monocular video, and NTS evaluates intermediate timestamps from temporally strided inputs. NVS and CTVS use multi-view datasets from fixed cameras: NVIDIA (Yoon et al., 2020), N3DV (Li et al., 2022), Kubric4D, CamxTime, and Charge (Nazarczuk et al., 2026). NTS is evaluated on monocular videos captured by moving cameras: DyCheck (Gao et al., 2022), ADT (Pan et al., 2023), and TUM (Sturm et al., 2012). Kubric4D and CamxTime are evaluated on held-out test scenes from our

Table 3: On Novel Time Synthesis, FL4N achieves the state-of-the-art accuracy by a large margin over all methods, showing the strong generalization of our latent-based approach at novel interpolated timestamps. Best, second best, and third best results are highlighted.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Pose</td><td colspan="3">DyCheck</td><td colspan="3">ADT</td><td colspan="3">TUM</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td colspan="10">Feed-forward Gaussian Representation</td></tr><tr><td>4DGT (Xu et al., 2025)</td><td>√</td><td>13.743</td><td>0.4079</td><td>0.6233</td><td>16.710</td><td>0.4999</td><td>0.5651</td><td>14.380</td><td>0.4468</td><td>0.5612</td></tr><tr><td>MoVieS (Lin et al., 2026a)</td><td>√</td><td>11.755</td><td>0.3420</td><td>0.6428</td><td>15.432</td><td>0.4698</td><td>0.5828</td><td>15.683</td><td>0.5079</td><td>0.4426</td></tr><tr><td>NeoVerse† (Yang et al., 2026)</td><td>x</td><td>16.756</td><td>0.4663</td><td>0.4566</td><td>16.179</td><td>0.4384</td><td>0.5296</td><td>16.120</td><td>0.5266</td><td>0.3981</td></tr><tr><td>C4G† (Kim et al., 2026c)</td><td>x</td><td>15.755</td><td>0.4735</td><td>0.6111</td><td>18.900</td><td>0.5813</td><td>0.5833</td><td>17.508</td><td>0.5518</td><td>0.4951</td></tr><tr><td colspan="10">Feed-forward Latent-based 4D Representation</td></tr><tr><td>FL4N (Ours)</td><td>x</td><td>19.019</td><td>0.4977</td><td>0.4284</td><td>19.693</td><td>0.5948</td><td>0.4490</td><td>17.847</td><td>0.5586</td><td>0.3449</td></tr><tr><td>FL4N (Ours)</td><td>√</td><td>20.527</td><td>0.5470</td><td>0.3757</td><td>24.115</td><td>0.6875</td><td>0.3083</td><td>21.162</td><td>0.6978</td><td>0.2601</td></tr></table>

training distribution, whereas the other datasets evaluate zero-shot generalization. Detailed evaluation protocol is provided in Appendix B.

## 4.1 RESULTS

Quantitative results. Tables 1, 2, and 3 report the three protocols. Here, <sup>†</sup> denotes 2,000-step testtime pose alignment with target frames, and Pose (✓/✗) indicates context pose input. FL4N achieves state-of-the-art performance among all pose-free baselines, even without test-time alignment, except for SSIM on NVS of two real-world datasets. On real-world datasets, whose camera poses are noisy estimates, C4G benefits from test-time pose alignment while our model is forced to estimate slight inaccurate pose relation, resulting in a slight misalignment of pixels. An optional incorporation of context camera poses further improves the accuracy by exploiting the explicit context-to-target pose relation achieving state-of-the-art performance across all benchmarks on all protocols. These results demonstrate the substantial strength of our proposed latent representation: strong 4D prior, generalization to novel view and time, and decouplement of view and time, all successfully learned in the latent space. Additional rendering quality evaluation on dynamic region and discussion on test-time pose alignment are provided in Appendices D.1 and F.

Ablation. Table 4 ablates our key design choices to justify our formulation, with all variants trained for 10k steps. We also include metrics on dynamic region to see its gain exclusively. The residual branch denotes our newly introduced modules (i. e., the dynamic encoder and the time module) on top of the static NVS model. Additional ablation studies are provided in Appendix D.6.

Table 4: Ablation studies on Kubric4D, on full frames and dynamic region. Each component contributes to our final formulation.
<table><tr><td rowspan="2">Variant</td><td colspan="4">Components</td><td colspan="3">Full frame</td><td colspan="3">Dynamic region</td></tr><tr><td>Residual branch</td><td>Static anchoring</td><td>loss</td><td>Mask Frozen renderer</td><td></td><td>PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓</td><td></td><td></td><td></td><td></td></tr><tr><td>(i)</td><td>x</td><td>一</td><td>x</td><td>x</td><td>21.90</td><td>0.727</td><td>0.287</td><td>13.16</td><td>0.485</td><td>0.438</td></tr><tr><td>(ii)</td><td>√</td><td>√</td><td>√</td><td>x</td><td>22.15</td><td>0.681</td><td>0.310</td><td>13.83</td><td>0.486</td><td>0.465</td></tr><tr><td>(iii)</td><td>√</td><td>√</td><td>x</td><td>√</td><td>22.93</td><td>0.741</td><td>0.262</td><td>14.11</td><td>0.500</td><td>0.429</td></tr><tr><td>(iv)</td><td>√</td><td>x</td><td>√</td><td>√</td><td>23.61</td><td>0.769</td><td>0.246</td><td>14.25</td><td>0.517</td><td>0.436</td></tr><tr><td>FL4N</td><td>√</td><td>√</td><td>√</td><td>√</td><td>23.67</td><td>0.773</td><td>0.237</td><td>14.26</td><td>0.519</td><td>0.419</td></tr></table>

Variant (i) is the static NVS model baseline, showing huge regression from FL4N. Freezing the renderer is crucial (variant (ii)); unfreezing it disrupts dynamic branch training by destabilizing the established latent-to-image mapping in the renderer. The dynamic mask loss (variant (iii)) modestly improves the accuracy by helping the model resolve depth ambiguity and localize dynamic regions during training. Without it, the model tends to collapse as training progresses. Static feature anchoring to the residual encoder (variant (iv)) further improves the overall accuracy, helping the residual branch to effectively isolate temporal variation from injected intermediate features from the static encoder.

Runtime. Table 5 reports runtime on the NVIDIA dataset at 224 × 224 resolution using a single RTX 5880 Ada GPU. FL4N achieves the fastest runtime. Ours is 1.08× faster than NeoVerse and 565× faster than that with pose alignment, with ours achieving substantially better accuracy.

Qualitative results. Figure 7 qualitatively compares our results with Gaussian-based methods. Gaussian-based methods often exhibit holes, object inconsistencies, and blurred details when rendering at novel viewpoints and timestamps. In contrast, our method produces visually consistent results. This again confirms the superiority of our latent-based method over explicit Gaussian primitives: 4D prior and view-time disentanglement can be learned successfully in the latent space, and novel view are seamlessly rendered. Appendix D.3 includes comprehensive qualitative comparisons on all benchmark datasets.

Table 5: Runtime. <sup>†</sup> denotes 2,000 step test-time pose alignment.
<table><tr><td>Method</td><td>sec / view↓ FPS ↑ PSNR ↑</td></tr><tr><td>4DGT</td><td>0.040 25.00 12.15</td></tr><tr><td>MoVieS</td><td>0.037 27.02</td></tr><tr><td>NeoVerse 0.027</td><td>14.10 37.03 10.74</td></tr><tr><td>NeoVerse† 14.127</td><td>0.07 13.86</td></tr><tr><td>0.036</td><td>27.77 14.25</td></tr><tr><td>C4G C4G† 23.885</td><td>0.04 16.76</td></tr><tr><td>FL4N (Ours) 0.025</td><td>40.00 17.65</td></tr></table>

4DGT  
MoVieS  
NeoVerse  
C4G  
FL4N (Ours)  
GT  
![](images/388887619fe53d8afe5d52b9adfa54a09b29d9fa4525bed9bd39a68a1c437187.jpg)  
Figure 7: Qualitative comparison on 4D Novel View Synthesis. Gaussian-based methods often exhibit holes, object inconsistencies, and blurred details. Our method produces more coherent and visually consistent results, demonstrating the superiority of our latent-based representation over explicit geometry primitives.

## 4.2 LATENT PROBING ON DOWNSTREAM TASKS

To assess whether the dynamic encoder captures temporal changes while preserving scene geometry encoded by static feature, we probe the representations via zero-shot temporal matching and videodepth probing. In tables, “All” covers the entire frame, while “Dyn” is restricted to dynamic objects. Full matching and depth results, along with point-tracking results, are provided in Appendix E.

Zero-shot matching. While the static prior establishes dense correspondences across views, exploiting epipolar geometry, moving content violates the multi-view consistency, degrading correspondence quality in dynamic regions. On the other hand, dynamic prior improves correspondence in such regions. In Table 6, we evaluate zeroshot matching of static $( \mathbf { z } _ { \mathrm { s t } } )$ and dynamic latent $\bf ( z _ { \mathrm { d y n } } )$ on DAVIS, using the protocol of Nam et al. (2026). The dynamic latent consistently demonstrates better accuracy than the static latent; the gap slightly widens on dynamic region.

Video depth estimation. Although the model receives no explicit geometric supervision, its latent representations can encode scene geometry information. We probe this by attaching DPT depth predictors on the static and dynamic latent and training them Kubric (Greff et al., 2022). Table 7 reports the video depth accuracy. Dynamic latent consistently outperforms static latent, across the full frame and dynamic-object regions. Notably, the advantage becomes more pronounced as the temporal stride between context frames increases, corresponding to larger object motion. This suggests that dynamic latent largely preserves geometry-aware information encoded by the static latent while incorporating additional cues from temporal dynamics.

Table 6: Zero-shot matching on DAVIS. Dynamic latent outperform static latent by encoding temporal information.
<table><tr><td>Region</td><td>Feat.</td><td> $\delta _ { \mathrm { a v g } }$ </td><td>δ1</td><td>δ4</td><td>δ16</td></tr><tr><td>All</td><td> $\mathbf { z } _ { \mathrm { s t } }$   ${ \bf z } _ { \mathrm { d y n } }$ </td><td>27.33 29.04 (+1.71)</td><td>5.65 5.76 (+0.11)</td><td>25.37 26.50 (+1.13)</td><td>53.40 57.64 (+4.24)</td></tr><tr><td>Dyn</td><td> $\mathbf { z } _ { \mathrm { s t } }$   ${ \mathbf z } _ { \mathrm { d y n } }$ </td><td>23.45 25.29</td><td>3.10 3.12 (+0.02)</td><td>21.14 22.37</td><td>49.03 53.68</td></tr></table>

Table 7: Probing video depth on Kubric. Dynamic latent yields consistent gains, with larger gains at greater temporal strides.
<table><tr><td rowspan="2">Region</td><td rowspan="2">Feat.</td><td colspan="2">∆4</td><td colspan="2">∆8</td></tr><tr><td>AbsRel↓</td><td>δ &lt; 1.25 ↑</td><td>AbsRel↓</td><td>δ &lt; 1.25 ↑</td></tr><tr><td rowspan="3">All</td><td>Zst</td><td>0.0944</td><td>0.8908</td><td>0.0868</td><td>0.9081</td></tr><tr><td> $\mathbf { z } _ { \mathrm { d y n } }$ </td><td>0.0863</td><td>0.9035</td><td>0.0751</td><td>0.9326</td></tr><tr><td></td><td>(−0.0081)</td><td>(+0.0127)</td><td>(−0.0117)</td><td>(+0.0244)</td></tr><tr><td rowspan="3">Dyn</td><td> $\mathbf { z } _ { \mathrm { s t } }$ </td><td>0.1178</td><td>0.8699</td><td>0.1149</td><td>0.8790</td></tr><tr><td> $\mathbf { z } _ { \mathrm { d y n } }$ </td><td>0.1050</td><td>0.8941</td><td>0.0996</td><td>0.9099</td></tr><tr><td></td><td>(−0.0128)</td><td>(+0.0242)</td><td>(−0.0153)</td><td>(+0.0309)</td></tr></table>

## 5 CONCLUSION

In this work, we introduced FL4N, a feed-forward latent model that renders dynamic scenes from a monocular video at novel views and timestamps. By learning a temporal residual on top of a frozen static latent model, FL4N acquires a 4D representation from limited dynamic data, supervised only by images and binary dynamic masks. Across eight benchmarks and three evaluation protocols, FL4N outperforms prior feed-forward Gaussian-based methods and generalizes to realworld videos, while its latent space successfully disentangles view from time and preserves scene geometry. Our work demonstrates the potential of latent representations for feed-forward 4D novel view synthesis, opening new directions for latent-based 4D scene understanding and modeling more complex dynamics.

## REFERENCES

Honggyu An, Jaewoo Jung, Mungyeom Kim, Chaehyun Kim, Minkyeong Jeon, Jisang Han, Kazumi Fukuda, Takuya Narihira, Hyunah Ko, Junsu Kim, Sunghwan Hong, Yuki Mitsufuji, and Seungryong Kim. Learning compact 3D representations from feed-forward novel view synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 363–373, June 2026.

Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. ReCamMaster: Camera-controlled generative rendering from a single video. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 14834– 14844, 2025.

Matteo Balice, Yanik Kunzi, Chenyangguang Zhang, Matteo Matteucci, Marc Pollefeys, and Sungwhan Hong. No Pose, No Problem in 4D: Feed-forward dynamic gaussians from unposed multiview videos. arXiv preprint arXiv:2605.22190, 2026.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle,¨ Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and ´ Christoph Feichtenhofer. SAM 3: Segment anything with concepts, 2025.

Kaihua Chen, Tarasha Khurana, and Deva Ramanan. Reconstruct, inpaint, test-time finetune: Dynamic novel-view synthesis from monocular videos. Advances in Neural Information Processing Systems, 38:47723–47753, 2026.

Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3R: Estimating disentangled motion from DUSt3R without training. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9158–9168, 2025.

Seokju Cho, Abhishek Badki, Hang Su, Jindong Jiang, Ziyao Zeng, Seungryong Kim, Sifei Liu, and Orazio Gallo. 4DP-QA: Scalable QA for 4D perception in vision language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23869–23879, 2026.

Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. Advances in Neural Information Processing Systems, pp. 33768–33780, 2022.

Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J. Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3749– 3761, 2022.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han, Jiaolong Yang, Chong Luo, and Seungryong Kim. PF3plat: Pose-free feed-forward 3D gaussian splatting for novel view synthesis. In International Conference on Machine Learning, 2025.

Yu Hu, Chong Cheng, Sicheng Yu, Xiaoyang Guo, and Hao Wang. VGGT4D: Mining motion cues in visual geometry transformers for 4D scene reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 414–424, 2026.

Zhening Huang, Hyeonho Jeong, Xuelin Chen, Yulia Gryaditskaya, Tuanfeng Y. Wang, Joan Lasenby, and Chun-Hao Huang. SpaceTimePilot: Generative rendering of dynamic scenes across space and time. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11218–11228, 2026.

Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, et al. RayZer: A self-supervised large view synthesis model. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4918–4929, 2025.

Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. LVSM: A large view synthesis model with minimal 3D inductive bias. In International Conference on Learning Representations, 2025.

Nikita Karaev, Yuri Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker3: Simpler and better point tracking by pseudo-labeling real videos. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1–10, 2025.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3D gaussian splat-¨ ting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Evan Kim, Hyunwoo Ryu, Thomas W. Mitchel, and Vincent Sitzmann. Scaling view synthesis transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28893– 28902, 2026a.

Jisoo Kim, Jungbin Cho, Sanghyeok Chu, Ananya Bal, Jinhyung Kim, Gunhee Lee, Sihaeng Lee, Seung Hwan Kim, Bohyung Han, Hyunmin Lee, et al. Pri4R: Learning world dynamics for vision-language-action models with privileged 4D representation. arXiv preprint arXiv:2603.01549, 2026b.

Mungyeom Kim, Minkyeong Jeon, Honggyu An, Jaewoo Jung, Hyuna Ko, Jisang Han, Hyeonseo Yu, Donghwan Shin, Sunghwan Hong, Takuya Narihira, et al. Learning global motion with compact gaussians for feed-forward 4D reconstruction. arXiv preprint arXiv:2605.31595, 2026c.

Renjie Li, Panwang Pan, Bangbang Yang, Dejia Xu, Shijie Zhou, Xuanyang Zhang, Zeming Li, Achuta Kadambi, Zhangyang Wang, Zhengzhong Tu, and Zhiwen Fan. 4K4DGen: Panoramic 4D generation at 4K resolution. In International Conference on Learning Representations, pp. 100692–100713, 2025.

Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3D video synthesis from multi-view video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5511–5521, 2022.

Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8508–8520, 2024.

Hanxue Liang, Jiawei Ren, Ashkan Mirzaei, Antonio Torralba, Ziwei Liu, Igor Gilitschenski, Sanja Fidler, Cengiz Oztireli, Huan Ling, Zan Gojcic, and Jiahui Huang. Feed-forward bullet-time reconstruction of dynamic scenes from monocular videos. Advances in Neural Information Processing Systems, 2025.

Chenguo Lin, Yuchen Lin, Panwang Pan, Yifan Yu, Tao Hu, Honglei Yan, Katerina Fragkiadaki, and Yadong Mu. MoVieS: Motion-aware 4D dynamic view synthesis in one second. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 295–306, 2026a.

Chieh Lin, Zhaoyang Lv, Songyin Wu, Zhen Xu, Thu Nguyen-Phuoc, Hung-Yu Tseng, Julian Straub, Numair Khan, Lei Xiao, Ming-Hsuan Yang, et al. DGS-LRM: Real-time deformable 3D gaussian reconstruction from monocular videos. Advances in Neural Information Processing Systems, 38:46915–46936, 2026b.

Thomas W. Mitchel, Hyunwoo Ryu, and Vincent Sitzmann. True self-supervised novel view synthesis is transferable. arXiv preprint arXiv:2510.13063, 2025.

Jisu Nam, Soowon Son, Dahyun Chung, Jiyoung Kim, Siyoon Jin, Junhwa Hur, and Seungryong Kim. Emergent temporal correspondences from video diffusion transformers. Advances in Neural Information Processing Systems, 38:123683–123727, 2026.

Michal Nazarczuk, Thomas Tanay, Arthur Moreau, Zhensong Zhang, and Eduardo Perez-Pellitero.´ Charge: A comprehensive novel view synthesis benchmark and dataset to bind them all. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15323–15333, 2026.

Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. Aria digital twin: A new benchmark dataset for egocentric 3D machine perception. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 20076–20086, 2023.

Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbelaez, Alex Sorkine-Hornung, and´ Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.

Ulrich Prestel, Stefan Andreas Baumann, Nick Stracke, and Bjorn Ommer. RayDer: Scalable self-¨ supervised novel view synthesis from real-world video. In European Conference on Computer Vision, pp. 558–578, 2026.

Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural radiance fields for dynamic scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10313–10322, 2021.

Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction.´ In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 12159–12168, 2021.

Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Muller, Alexander Keller, Sanja Fidler, and Jun Gao. GEN3C: 3D-informed world-consistent¨ video generation with precise camera control. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6121–6132, 2025.

Mehdi S.M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Luciˇ c, Daniel Duckworth, Alexey Dosovitskiy, et al. Scene representation´ transformer: Geometry-free novel view synthesis through set-latent scene representations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6219–6228, 2022.

Mehdi S.M. Sajjadi, Aravindh Mahendran, Thomas Kipf, Etienne Pot, Daniel Duckworth, Mario Luciˇ c, and Klaus Greff. RUST: Latent neural scene representations from unposed imagery. In´ IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 17297–17306, 2023.

Maximilian Seitzer, Sjoerd van Steenkiste, Thomas Kipf, Klaus Greff, and Mehdi S.M. Sajjadi. DyST: Towards dynamic neural scene representations on real-world videos. In International Conference on Learning Representations, volume 2024, pp. 37052–37065, 2024.

Soowon Son, Honggyu An, Jisu Nam, Hyunah Ko, Chaehyun Kim, Dahyun Chung, Siyoon Jin, Jung Yi, Junhwa Hur, and Seungryong Kim. Probing and leveraging video diffusion transformer features for robust point tracking. arXiv preprint arXiv:2512.20606, 2025.

Rui Song, Chenwei Liang, Yan Xia, Walter Zimmer, Hu Cao, Holger Caesar, Andreas Festag, and Alois Knoll. CoDa-4DGS: Dynamic gaussian splatting with context and deformation awareness for autonomous driving. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 28031–28041, 2025.

Jurgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A bench-¨ mark for the evaluation of RGB-D SLAM systems. In IEEE/RSJ international conference on intelligent robots and systems, pp. 573–580, 2012.

Stanislaw Szymanowicz, Minghao Chen, Jianyuan Wang, Christian Rupprecht, and Andrea Vedaldi. LagerNVS: Latent geometry for fully neural real-time novel view synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15443–15453, 2026.

Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision, pp. 313–331, 2024.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5294–5306, 2025a.

Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4D reconstruction from a single video. In IEEE/CVF International Conference on Computer Vision, pp. 9660–9672, 2025b.

Yihan Wang and Jia Deng. WAFT: Warping-alone field transforms for optical flow. In International Conference on Learning Representations, volume 2026, pp. 157389–157402, 2026.

Daniel Watson, Saurabh Saxena, Lala Li, Andrea Tagliasacchi, and David Fleet. Controlling space and time with diffusion models. In International Conference on Learning Representations, volume 2025, pp. 95989–96012, 2025.

Kai Xu, Tze Ho Elden Tse, Jizong Peng, and Angela Yao. DAS3R: Dynamics-aware gaussian splatting for static scene reconstruction. arXiv preprint arXiv:2412.19584, 2024a.

Zhen Xu, Sida Peng, Haotong Lin, Guangzhao He, Jiaming Sun, Yujun Shen, Hujun Bao, and Xiaowei Zhou. 4K4D: Real-time 4D view synthesis at 4K resolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20029–20040, 2024b.

Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard Newcombe, and Zhaoyang Lv. 4DGT: Learning a 4D gaussian transformer using real-world monocular videos. In Advances in Neural Information Processing Systems, pp. 42606–42632, 2025.

Yuxue Yang, Lue Fan, Ziqi Shi, Junran Peng, Feng Wang, and Zhaoxiang Zhang. NeoVerse: Enhancing 4D world model with in-the-wild monocular videos. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 40340–40351, 2026.

Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No pose, no problem: Surprisingly simple 3D gaussian splats from sparse unposed images. In International Conference on Learning Representations, pp. 54009–54033, 2025.

Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, and Jan Kautz. Novel view synthesis of dynamic scenes with globally coherent depths from a monocular camera. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5335–5344, 2020.

Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. TrajectoryCrafter: Redirecting camera trajectory for monocular videos via diffusion models. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 100–111, 2025.

Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, trevor darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. MonST3R: A simple approach for estimating geometry in the presence of motion. In International Conference on Learning Representations, pp. 82863–82886, 2025.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3813– 3824, 2023.

Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, and Tong He. Aether: Geometric-aware unified world modeling. In IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8535–8546, 2025.

## APPENDIX CONTENTS

A Implementation details 16   
A.1 Architecture details 16   
A.2 Training details 16   
B Evaluation protocol 17   
B.1 Evaluation datasets 17   
B.2 Evaluation protocols 17   
B.3 Model specific configurations . 18   
C Latent analysis 18   
D Additional results 19   
D.1 Quantitative comparison on dynamic regions . . 19   
D.2 Additional analysis on quantitative results . 19   
D.3 Additional qualitative results . 20   
D.4 Dynamic mask 20   
D.5 Context frame number ablation . 20   
D.6 Architectural ablation 21   
E Probings 22   
E.1 Full probing results 22   
E.2 Point tracking 22   
E.3 Video Depth estimation . 22   
F Test-time pose alignment for latent models 23   
G Limitations 24

## A IMPLEMENTATION DETAILS

## A.1 ARCHITECTURE DETAILS

Static encoder and renderer. The static prior used is a pretrained feed-forward novel view synthesizer whose encoder is a 24-layer transformer alternating frame-wise and global attention, followed by a 12-layer transformer renderer. Both use a hidden size of 768 with 12 heads on $8 \times 8$ patches, and remain frozen during our training.

Dynamic encoder. The dynamic encoder adopts the same core architecture as the static encoder, augmented with cross-attention, but uses one-third of its depth, with 8 layers. It is initialized with weights uniformly sampled from the pretrained static encoder. Each layer additionally cross-attends to the last layer of static features, allowing the residual branch to access camera-pose-aware representations throughout encoding. This provides a spatial anchor for learning motion residuals, complementing the static features rather than relearning scene structure. A DPT decoder (Ranftl et al., 2021) attached to the dynamic encoder predicts the binary dynamic mask.

Temporal conditioning. We use a single learnable time token $\tau \in \mathbb { R } ^ { C }$ with $C = 1 0 2 4$ , and attach a copy of it to each input view by concatenating it with that view’s tokens. Each copy becomes viewspecific $( \hat { \tau } _ { k } )$ through attention with its own view, allowing the same token to handle an arbitrary number and ordering of context frames. The resulting time representation is processed by a time module consisting of two transformer blocks with adaptive layer normalization conditioned on the target-time token, followed by a zero-initialized linear residual gate.

## A.2 TRAINING DETAILS

Training setup. We train with AdamW at a learning rate of $3 . 5 \times 1 0 ^ { - 5 }$ , weight decay 0.05, and 3k warmup steps, while keeping the static encoder and renderer frozen for whole training. FL4N is trained on four NVIDIA RTX 5880 Ada Generation GPUs with a batch size of 8 for total of 32k steps. Each training sample contains 8 conditioning views and 12 target views at a resolution of $3 3 6 \times 5 1 2$

Training datasets. We use Kubric4D (Van Hoorick et al., 2024) and CamxTime (Huang et al., 2026), which provide synchronized multi-view videos captured by static cameras. To disentangle camera and object motion, we train the model to reconstruct target frames at view–time pairs not included in the input set.

Pose dropout. To support pose-free conditioning at inference, we randomly zero out the conditioning camera tokens with probability 0.4 during training. At inference time, FL4N supports fully unposed setting by zeroing all conditioning camera tokens.

Loss. We use an $\ell _ { 2 }$ photometric loss $\mathcal { L } _ { 2 } ,$ , LPIPS losses $\mathcal { L } _ { \mathrm { p e r c } }$ and $\mathcal { L } _ { \mathrm { p e r c } } ^ { \mathrm { d y n } }$ , and an $\ell _ { 1 }$ image-gradient loss $\mathcal { L } _ { \mathrm { g r a d } } ^ { \mathrm { d y n } }$ . The dynamic-region losses are computed using the dynamic mask $M _ { k }$ to select moving pixels, on $1 2 8 \times 1 2 8$ crops centered on the moving content. For the image loss term, the dynamic and static regions are averaged over their respective areas and weighted equally. Dynamic mask loss is only comptued for context frames. We set $\lambda _ { \mathrm { p e r c } } = 0 . 5 , \lambda _ { \mathrm { p e r c } } ^ { \mathrm { d y n } } = 2 . 0 , \lambda _ { \mathrm { g r a d } } ^ { \mathrm { d y n } } = 1 . 0$ , and $\lambda _ { \mathrm { m a s k } } = 1 . 0$

Pseudo dynamic mask generation. We generate pseudo dynamic masks by thresholding the magnitude of optical flow estimated by an off-the-shelf flow estimator (Wang & Deng, 2026). Since the flow is computed between frames captured by the same static camera, camera-induced motion and parallax are absent, making the remaining flow primarily reflect object motion. This leverages the static-camera multi-view configuration of our training data to identify dynamic regions with binary pseudo labels. The mask generation can also be easily replaced with off-the-shelf segmentation models, such as SAM3 (Carion et al., 2025), without changing the training framework.

## B EVALUATION PROTOCOL

## B.1 EVALUATION DATASETS

Our benchmark comprises five multi-view datasets for NVS and CTVS: NVIDIA, N3DV, Kubric4D, CamxTime, and Charge, and three datasets for NTS: ADT, DyCheck, and TUM. Kubric4D and CamxTime use held-out scenes, while the remaining datasets are unseen during training and evaluate cross-dataset generalization.

For static camera multi-view datasets, we simulate moving camera monocular video by selecting N diagonal frames, i. e. camera i observed at time i as input.

Kubric4D. We use 100 test scenes from Kubric4D, which contains 16 static cameras, with the first four providing top-down views and the remaining 12 cameras arranged around the scene on a horizontal ring. Each camera captures 60 frames. We exclude the first four cameras and sample cameras consecutively from the horizontal ring. We use a frame interval of 8.

CamxTime. CamxTime contains four separate camera trajectories, each comprising 120 densely spaced cameras with 120 frames per camera. We hold out the last 10% of scenes for testing and use Trajectory 01 for evaluation, with camera and frame intervals both set to 8.

Charge. we use the Sparse Test cameras. Charge stores cameras in a non-spatial order, so sampling by file index leaves large gaps between consecutive conditioning views. We therefore reorder the cameras of each scene along a continuous open path through the rig and uniformly sample along this path to approximate a smooth monocular camera trajectory. we use the following camera order:

$$
\begin{array} { r l } { \mathsf { C O 5 } } & { { } \mathsf { C O 1 } \quad \mathsf { C O 4 } \quad \mathsf { C O 6 } \quad \mathsf { C O 2 } \quad \mathsf { C O 8 } \quad \mathsf { C O 0 } \quad \mathsf { C O 7 } \quad \mathsf { C O 3 } \quad \mathsf { C O 9 } , } \end{array}
$$

which follows the shortest open path through the rig where Ci denotes the i-th test camera. CTVS uses the first 8 cameras of this order, and NVS holds out C02. We use a frame interval of 8.

## B.2 EVALUATION PROTOCOLS

NVS (Novel View Synthesis). We take 9 frames and hold out the 5th as a target. The remaining 8 are used as the input, and the held-out frame’s viewpoint is the target at all 8 conditioning times. For N3DV, cam0 is used for a target camera.

CTVS (Cross-time View Synthesis). Given N diagonal frames as input, the targets are the N<sup>2</sup>−N off-diagonal cells of the camera–time grid, evaluating synthesis at unseen viewpoint–timestamp combinations. NVIDIA uses all available cameras (N=12), and N3DV uses all cameras except cam0 (N=17–20 depending on the scene). Kubric4D, CamxTime, and Charge each use N=8.

NTS (Novel Time Synthesis). Following C4G (Kim et al., 2026c), we use the following scenes from each dataset:

• DyCheck   
– apple, block, paper-windmill, spin, teddy   
• ADT   
– Apartment release multiuser cook seq141 M1292   
– Apartment release multiskeleton party seq114 M1292   
– Apartment release meal skeleton seq135 M1292   
– Apartment release work skeleton seq137 M1292   
• TUM   
– fr2 desk with person   
– fr3 sitting {halfsphere, rpy, static, xyz}   
– fr3 walking {halfsphere, rpy, static, xyz}

All three datasets use a single moving camera for both context and target views. We follow the C4G (Kim et al., 2026c) evaluation protocol and sample 16 frames with stride 6 and evaluate on the 15 interleaved frames.

Time  
![](images/1ccedd2d993d853a5b306d7062539fccf884c1f31e86b773352d3d94cc3472a0.jpg)  
Figure A: Visualization of static latent. The static latent encodes camera pose, with similar colors for cameras pointing in the same direction. Red boxes highlight overlapping regions across context frames, which exhibit similar color gradients.

![](images/a1f7356cb83314f3079de3d4bc7e292245cc435abde39a9036599a1232947d8c.jpg)  
Figure B: Visualization of residual latent. (Left) Ground-truth RGB frame overlaid with the dynamic mask. (Right) Corresponding residual latent grid. Dynamic objects exhibit more pronounced residual latents as the target pose and time diverge from the input, indicating a larger discrepancy from the context frames.

## B.3 MODEL SPECIFIC CONFIGURATIONS

Every model runs at the native input resolution. For the pose-free baselines, NeoVerse (Yang et al., 2026) and C4G (Kim et al., 2026c), we apply 2,000 steps of test-time pose alignment (Ye et al., 2025). In contrast, FL4N does not perform pose alignment, despite being evaluated in the pose-free setting. For real-world datasets, FL4N uses camera poses predicted by VGGT (Wang et al., 2025a), using the estimated intrinsics for NVIDIA and the estimated intrinsics and extrinsics for N3DV.

## C LATENT ANALYSIS

Figure A shows a PCA visualization of the static latent $\mathbf { z } _ { \mathrm { s t } }$ under camera motion. The static latent primarily captures camera pose, with features varying smoothly as the camera moves along a circular trajectory. Similar features correspond to similar viewing directions.

Figure B visualizes the time-conditioned residual features along with their corresponding GT dynamic masks. For each view i and target time $j ,$ the residual feature is spatially anchored to the diagonal frame at view i and time i. Dynamic objects become more pronounced in the residual as their position at time $j$ increasingly differs from that at the anchor time i, while showing weaker responses near the diagonal where the temporal displacement is small. This trend indicates that the residual latent encodes timestamp-dependent differences, with larger feature variations corresponding to larger object motion.

Table A: 4D NVS result for Dynamic region. <sup>†</sup> denotes 2,000 step test-time pose alignment, the best, second best, and third best results in each column are highlighted. Metrics are restricted to the moving pixels and are computed over the same target subset, and their full-frame counterparts for the same cells are reported in Tables 1 and 2.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Pose</td><td colspan="6">NVS</td><td colspan="6">CTVS</td></tr><tr><td colspan="3">Kubric4D</td><td colspan="3">Charge</td><td colspan="3">Kubric4D</td><td colspan="3">Charge</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Feed-forward 4D Gaussian Representation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4DGT (Xu et al., 2025)</td><td>√</td><td>12.089</td><td>0.2406</td><td>0.0305</td><td>17.703</td><td>0.6516</td><td>0.1127</td><td>12.405</td><td>0.2557</td><td>0.0303</td><td>17.708</td><td>0.6588</td><td>0.1157</td></tr><tr><td>MoVieS (Lin et al., 2026a)</td><td>√</td><td>11.401</td><td>0.1741</td><td>0.0318</td><td>17.969</td><td>0.6079</td><td>0.1130</td><td>11.797</td><td>0.2034</td><td>0.0307</td><td>18.068</td><td>0.6119</td><td>0.1147</td></tr><tr><td>NeoVerse† (Yang et al., 2026)</td><td>x</td><td>10.847</td><td>0.1527</td><td>0.0381</td><td>16.292</td><td>0.5161</td><td>0.1242</td><td>10.254</td><td>0.1372</td><td>0.0403</td><td>16.152</td><td>0.4966</td><td>0.1307</td></tr><tr><td>C4G† (Kim et al., 2026c)</td><td>x</td><td>13.031</td><td>0.3082</td><td>0.0322</td><td>18.552</td><td>0.6617</td><td>0.1143</td><td>12.901</td><td>0.3052</td><td>0.0333</td><td>18.385</td><td>0.6566</td><td>0.1166</td></tr><tr><td>Feed-forward Latent-based 4D Representation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FL4N (Ours)</td><td>x</td><td>14.774</td><td>0.3479</td><td>0.0240</td><td>19.906</td><td>0.6708</td><td>0.1063</td><td>14.482</td><td>0.3351</td><td>0.0246</td><td>19.748</td><td>0.6774</td><td>0.1019</td></tr><tr><td>FL4N (Ours)</td><td>√</td><td>14.800</td><td>0.3485</td><td>0.0236</td><td>19.993</td><td>0.6711</td><td>0.1029</td><td>14.451</td><td>0.3358</td><td>0.0244</td><td>19.935</td><td>0.6772</td><td>0.1009</td></tr></table>

4DGT  
MoVieS  
NeoVerse  
C4G  
FL4N (Ours)  
GT  
![](images/1ee552787750d223b111c3280f26f1790e279359b9647522878efb11acdd5079.jpg)  
Figure C: Further qualitative comparison. Gaussian-based methods exhibit holes, object inconsistencies, and blurred details. Our method produces more coherent and visually consistent results, demonstrating the superiority of our latent-based representation over explicit geometry primitives.

## D ADDITIONAL RESULTS

## D.1 QUANTITATIVE COMPARISON ON DYNAMIC REGIONS

On the two datasets with ground-truth motion masks, Kubric4D and Charge (Nazarczuk et al., 2026), we additionally report metrics computed only within dynamic regions. Since moving content typically occupies a small fraction of the frame, dynamic-region metrics provide a more focused evaluation of reconstruction quality on moving subjects. Table A reports the dynamic-region metrics for NVS and CTVS, where our model consistently outperforms the baselines.

## D.2 ADDITIONAL ANALYSIS ON QUANTITATIVE RESULTS

In the main table, NeoVerse performs notably poorly on Kubric4D. This is because NeoVerse tends to provide weaker information across the temporal axis and produces many holes, leaving region unseen at a given timestamp largely unmodeled. This issue is particularly severe on Kubric4D, where the large camera gaps span nearly the entire viewing range, making regions observed at different timestamps, such as the background, especially difficult to reconstruct.

Table B: Quantitative results on dynamic regions. On the dynamic object segmentation task, Our method substantially outperforms other methods for 3D or 4D geometry estimation. Best and second best results are highlighted.
<table><tr><td rowspan="2">Method</td><td colspan="4">DAVIS-2016</td><td colspan="4">DAVIS-2017</td><td colspan="4">DAVIS-all</td></tr><tr><td>JM↑</td><td>JR↑</td><td>FM↑</td><td>FR↑</td><td>JM↑</td><td>JR↑</td><td>FM↑</td><td>FR↑</td><td>JM↑</td><td>JR↑</td><td>FM↑</td><td>FR↑</td></tr><tr><td>Easi3Rdust3r (Chen et al., 2025)</td><td>50.10</td><td>55.77</td><td>43.40</td><td>37.25</td><td>46.86</td><td>50.54</td><td>39.06</td><td>30.05</td><td>44.10</td><td>50.85</td><td>35.16</td><td>27.24</td></tr><tr><td>Easi3Rmonst3r (Chen et al., 2025)</td><td>54.93</td><td>68.00</td><td>45.29</td><td>47.30</td><td>54.75</td><td>66.16</td><td>44.09</td><td>42.36</td><td>51.64</td><td>63.06</td><td>40.98</td><td>38.49</td></tr><tr><td>MonST3R (Zhang et al., 2025)</td><td>40.42</td><td>40.39</td><td>49.54</td><td>52.12</td><td>38.07</td><td>36.05</td><td>48.24</td><td>49.01</td><td>36.98</td><td>34.52</td><td>47.03</td><td>46.72</td></tr><tr><td>DAS3R (Xu et al., 2024a)</td><td>41.13</td><td>38.67</td><td>44.50</td><td>36.94</td><td>44.51</td><td>43.95</td><td>46.71</td><td>44.96</td><td>43.33</td><td>38.93</td><td>45.24</td><td>38.78</td></tr><tr><td>VGGT4D (Hu et al., 2026)</td><td>62.12</td><td>76.80</td><td>56.04</td><td>67.49</td><td>56.45</td><td>65.62</td><td>51.09</td><td>56.85</td><td>50.75</td><td>55.59</td><td>47.04</td><td>46.43</td></tr><tr><td>FL4N (Ours)</td><td>74.93</td><td>93.78</td><td>74.16</td><td>90.64</td><td>75.44</td><td>92.32</td><td>74.37</td><td>90.38</td><td>68.76</td><td>80.11</td><td>69.87</td><td>83.99</td></tr></table>

![](images/d4fefd372673c5385fdb8bc7a2c38ecb6c37671aef7d1308e0372ac0239780ac.jpg)  
Figure D: Qualitative comparison of dynamic masks. Our method show more accurate dynamic object masks with clearer object boundary than VGGT4D (Hu et al., 2026).

## D.3 ADDITIONAL QUALITATIVE RESULTS

Figure C shows additional qualitative comparisons for 4D NVS. Figures H to M provide comparisons across all evaluated methods and scenes. Figures N to P further compare cross-time view synthesis results using view-time grids. Finally, Figs. Q to S show additional view-time grid results of FL4N alongside the corresponding ground-truth views. Across these results, FL4N exhibits stronger temporal consistency and more effective disentanglement of viewpoint and time, producing stable renderings with fewer temporal inconsistencies and visual artifacts.

## D.4 DYNAMIC MASK

In Table B, we report the dynamic mask estimation accuracy with IoU and Boundary F-measure as JM and FM, respectively, and their corresponding recall metrics as JR and FR, following the evaluation protocol of Hu et al. (2026). The reported metrics for the other methods are adopted from Hu et al. (2026). Figure D shows that our model predicts dynamic masks with cleaner boundaries and greater robustness, demonstrating its ability to generalize dynamic object localization to unseen real-world datasets.

## D.5 CONTEXT FRAME NUMBER ABLATION

Our model is trained with only 8 context frames, yet Figure E shows that the rendering quality remains on a flat plateau as the number of context frames increases to 32+, more than four times the training configuration. This demonstrates that the dynamic encoder generalizes to substantially longer temporal contexts than trained. The result highlights that our model can effectively exploit temporal context far beyond the number of frames seen during training.

We use CamxTime test scenes, which provide enough number of context frames while maintaining a sufficiently dense set of conditioning candidates. Context frames are sampled along a moving camera trajectory from the camera-time grid within a fixed temporal window. Thus, all settings cover the same camera and temporal extent, while increasing the number of context frames increases the sampling density. We evaluate 12 target frames across four novel camera viewpoints, 21, 45, 69, 93, and three timestamps, 8, 56, 104.

![](images/e3e3b493d9d30e3118a8d2bdb31bd476864a609bc59b8abbfed732cecf757fc3.jpg)  
Figure E: Generalization to the number of context frames, reporting PSNR on CamxTime as the number of conditioning frames varies at inference. The vertical line marks the 8 frames the model is trained with. The rendering quality remains on a flat plateau as the number of context frames increases to 32+. The dynamic encoder generalizes to substantially longer temporal contexts than trained and effectively exploits temporal context far beyond the number of frames seen during training.

Table C: Extensive ablation on Kubric4D across different temporal strides. Best and second best results are highlighted.
<table><tr><td rowspan="2">Variant</td><td colspan="3">∆1</td><td colspan="3">∆2</td><td colspan="3">∆4</td><td colspan="3">∆8</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Static NVS model</td><td>21.90</td><td>0.727</td><td>0.287</td><td>20.87</td><td>0.716</td><td>0.308</td><td>20.49</td><td>0.705</td><td>0.325</td><td>20.35</td><td>0.700</td><td>0.332</td></tr><tr><td>Static NVS model, fine-tuned</td><td>21.83</td><td>0.668</td><td>0.320</td><td>21.30</td><td>0.664</td><td>0.334</td><td>21.32</td><td>0.666</td><td>0.338</td><td>21.33</td><td>0.665</td><td>0.339</td></tr><tr><td>Architecture</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unfrozen renderer</td><td>22.15</td><td>0.681</td><td>0.309</td><td>21.57</td><td>0.676</td><td>0.325</td><td>21.55</td><td>0.676</td><td>0.332</td><td>21.55</td><td>0.676</td><td>0.334</td></tr><tr><td>Single branch encoder</td><td>23.18</td><td>0.727</td><td>0.290</td><td>22.74</td><td>0.722</td><td>0.300</td><td>22.58</td><td>0.718</td><td>0.307</td><td>22.47</td><td>0.715</td><td>0.312</td></tr><tr><td>w/o dynamic mask loss</td><td>22.93</td><td>0.741</td><td>0.262</td><td>22.24</td><td>0.728</td><td>0.285</td><td>22.05</td><td>0.720</td><td>0.298</td><td>21.99</td><td>0.719</td><td>0.303</td></tr><tr><td>w/o static feature anchoring</td><td>23.61</td><td>0.769</td><td>0.246</td><td>22.91</td><td>0.755</td><td>0.267</td><td>22.61</td><td>0.744</td><td>0.282</td><td>22.47</td><td>0.741</td><td>0.289</td></tr><tr><td>Sinusoidal time embedding</td><td>23.02</td><td>0.747</td><td>0.259</td><td>22.29</td><td>0.730</td><td>0.279</td><td>22.19</td><td>0.715</td><td>0.295</td><td>22.12</td><td>0.715</td><td>0.301</td></tr><tr><td>Dynamic encoder</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>4 layers</td><td>23.58</td><td>0.771</td><td>0.247</td><td>22.76</td><td>0.756</td><td>0.269</td><td>22.52</td><td>0.745</td><td>0.283</td><td>22.44</td><td>0.743</td><td>0.289</td></tr><tr><td>16 layers</td><td>23.63</td><td>0.769</td><td>0.249</td><td>22.87</td><td>0.754</td><td>0.271</td><td>22.57</td><td>0.744</td><td>0.286</td><td>22.45</td><td>0.740</td><td>0.292</td></tr><tr><td>24 layers</td><td>23.44</td><td>0.755</td><td>0.256</td><td>22.67</td><td>0.740</td><td>0.278</td><td>22.37</td><td>0.730</td><td>0.292</td><td>22.24</td><td>0.726</td><td>0.299</td></tr><tr><td>8 layers, random init.</td><td>23.42</td><td>0.772</td><td>0.250</td><td>22.56</td><td>0.755</td><td>0.275</td><td>22.32</td><td>0.743</td><td>0.290</td><td>22.15</td><td>0.739</td><td>0.296</td></tr><tr><td>FL4N (Ours)</td><td>23.67</td><td>0.773</td><td>0.237</td><td>22.88</td><td>0.760</td><td>0.257</td><td>22.65</td><td>0.751</td><td>0.271</td><td>22.52</td><td>0.747</td><td>0.278</td></tr></table>

## D.6 ARCHITECTURAL ABLATION

Table C extends Table 4 to more variants and temporal strides {1, 2, 4, 8}. We first compare against the static NVS backbone and its fine-tuned variant. While both provide a strong static-scene prior, fine-tuning alone does not effectively capture scene dynamics. We further ablate architectural choices. Unfreezing the renderer degrades performance, suggesting that preserving the pretrained latent-to-image mapping is beneficial. The single-branch encoder, removal of the dynamic mask loss, and sinusoidal time embeddings all underperform our full model. Removing static feature anchoring also reduces performance overall, highlighting the benefit of conditioning dynamic features on the static representation. Finally, we vary the dynamic encoder depth and initialization. Our 8-layer encoder achieves the best overall performance among the tested configurations. Random initialization also yields competitive results, although initializing from the static encoder weights consistently improves performance across temporal strides.

Table D: Zero-shot matching on DAVIS. Performance on all threshold are reported. Dynamic latent demonstrates consistently better zero-shot matching accuracy than static latent on all thresholds.
<table><tr><td>Region</td><td>Feat.</td><td> $\delta _ { \mathrm { a v g } }$ </td><td>δ1</td><td> $\delta 2$ </td><td>δ4</td><td>δ8</td><td>δ16</td></tr><tr><td>All</td><td> $\mathbf { z } _ { \mathrm { s t } }$ </td><td>27.33</td><td>5.65</td><td>12.43</td><td>25.37</td><td>39.78</td><td>53.40</td></tr><tr><td></td><td> ${ \bf z } _ { \mathrm { d y n } }$ </td><td>29.04</td><td>5.76</td><td>12.50</td><td>26.50</td><td>42.81</td><td>57.64</td></tr><tr><td></td><td></td><td>(+1.71)</td><td>(+0.11)</td><td>(+0.07)</td><td>(+1.13)</td><td>(+3.03)</td><td>(+4.24)</td></tr><tr><td>Dyn</td><td> $\mathbf { z } _ { \mathrm { s t } }$ </td><td>23.45</td><td>3.10</td><td>9.12</td><td>21.14</td><td>34.84</td><td>49.03</td></tr><tr><td></td><td> ${ \bf z } _ { \mathrm { d y n } }$ </td><td>25.29</td><td>3.12</td><td>9.20</td><td>22.37</td><td>38.05</td><td>53.68</td></tr><tr><td></td><td></td><td>(+1.84)</td><td>(+0.02)</td><td>(+0.08)</td><td>(+1.23)</td><td>(+3.21)</td><td>(+4.65)</td></tr></table>

Table E: Video depth probed on Kubric. Performance on all strides are reported. The gain is larger at larger temporal gap.
<table><tr><td rowspan="2">Region</td><td rowspan="2">Feat.</td><td colspan="2">∆1</td><td colspan="2">∆2</td><td colspan="2"> $\Delta 4$ </td><td colspan="2">∆8</td></tr><tr><td>AbsRel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td><td>AbsRel.↓</td><td> $\delta < 1 . 2 5 ~ $ </td><td>AbsRel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td><td>AbsRel↓</td><td> $\delta < 1 . 2 5 \uparrow$ </td></tr><tr><td rowspan="3">All</td><td> $\mathbf { z } _ { \mathrm { s t } }$ </td><td>0.1129</td><td>0.8550</td><td>0.1023</td><td>0.8729</td><td>0.0944</td><td>0.8908</td><td>0.0868</td><td>0.9081</td></tr><tr><td> ${ \bf z } _ { \mathrm { d y n } }$ </td><td>0.1037</td><td>0.8722</td><td>0.0932</td><td>0.8937</td><td>0.0863</td><td>0.9035</td><td>0.0751</td><td>0.9326</td></tr><tr><td></td><td>(-0.0092)</td><td>(+0.0171)</td><td>(-0.0092)</td><td>(+0.0208)</td><td>(-0.0081)</td><td>(+0.0127)</td><td>(-0.0117)</td><td>(+0.0244)</td></tr><tr><td rowspan="3">Dyn</td><td> $\mathbf { z } _ { \mathrm { s t } }$ </td><td>0.1180</td><td>0.8597</td><td>0.1194</td><td>0.8614</td><td>0.1178</td><td>0.8699</td><td>0.1149</td><td>0.8790</td></tr><tr><td> ${ \bf z } _ { \mathrm { d y n } }$ </td><td>0.1127</td><td>0.8747</td><td>0.1078</td><td>0.8824</td><td>0.1050</td><td>0.8941</td><td>0.0996</td><td>0.9099</td></tr><tr><td></td><td>(−0.0052)</td><td>(+0.0150)</td><td>(−0.0116)</td><td>(+0.0209)</td><td>(-0.0128)</td><td>(+0.0242)</td><td>(−0.0153)</td><td>(+0.0309)</td></tr></table>

Table F: Point tracking probed on DAVIS. Tracking performance across different temporal strides are reported.
<table><tr><td rowspan="2">Region</td><td rowspan="2">Feat.</td><td colspan="3">∆1</td><td colspan="3">∆2</td><td colspan="3">∆4</td><td colspan="3">∆8</td></tr><tr><td>AJ↑</td><td> $< \delta _ { \mathrm { a v g } } ^ { x } \uparrow$ </td><td>OA↑</td><td>AJ↑</td><td> $< \delta _ { \mathrm { a v g } } ^ { x } \uparrow$ </td><td>OA↑</td><td>AJ↑</td><td> $< \delta _ { \mathrm { a v g } } ^ { x } \uparrow$ </td><td>OA↑</td><td>AJ↑</td><td> $< \delta _ { \mathrm { a v g } } ^ { x } \uparrow$ </td><td>OA↑</td></tr><tr><td>All</td><td> $\mathbf { z } _ { \mathrm { s t } }$   ${ \bf z } _ { \mathrm { d y n } }$ </td><td>37.63 40.45 (+2.82)</td><td>52.35 54.84 (+2.49)</td><td>70.40 73.25 (+2.85)</td><td>36.15 39.80 (+3.64)</td><td>51.65 54.76 (+3.11)</td><td>69.20 72.61 (+3.41)</td><td>33.10 36.59 (+3.49)</td><td>49.00 51.73 (+2.73)</td><td>66.42 69.79 (+3.37)</td><td>28.92 32.29 (+3.38)</td><td>45.82 47.60 (+1.78)</td><td>63.42 67.06 (+3.64)</td></tr><tr><td>Dyn</td><td> $\mathbf { z } _ { \mathrm { s t } }$   ${ \bf z } _ { \mathrm { d y n } }$ </td><td>32.26 34.82 (+2.56)</td><td>47.85 49.86 (+2.02)</td><td>66.12 69.23 (+3.11)</td><td>30.83 34.09 (+3.27)</td><td>47.39 49.53 (+2.14)</td><td>64.80 68.65 (+3.85)</td><td>28.57 32.01 (+3.43)</td><td>45.69 47.70 (+2.00)</td><td>62.65 66.64 (+3.99)</td><td>24.84 28.75 (+3.91)</td><td>42.88 44.81 (+1.93)</td><td>59.72 64.38 (+4.65)</td></tr></table>

## E PROBINGS

## E.1 FULL PROBING RESULTS

Tables D and E provide the full probing results, while the main paper reports a subset of matching thresholds and depth across temporal strides.

## E.2 POINT TRACKING

Correspondence between frames does not necessarily capture how scene points evolve over time. We therefore evaluate point tracking by attaching a CoTracker3 (Karaev et al., 2025) head to each representation. All variants are trained on Kubric (Greff et al., 2022) for the same number of epochs and evaluated in DAVIS (Pont-Tuset et al., 2017) following the protocol of (Son et al., 2025). Fig ure F provides qualitative comparisons, showing that dynamic features yield more accurate point tracks over time. Table F shows that dynamic feature substantially improves tracking of moving points over the static feature, with the performance gap increasing as the temporal stride grows. Consistent with the depth probing results, this widening gap indicates that the residual captures temporal information that is particularly useful under larger object displacement.

## E.3 VIDEO DEPTH ESTIMATION

Figure G further compares depth maps predicted from static and dynamic features. Dynamic features yield more accurate depth predictions, with sharper boundaries, more coherent depth within objects, and better background depth estimation, whereas static features produce noisier depth maps.

![](images/032ae10bf035e5a630241098c2251a637299ca30c75f5f0394e195c1939e77d5.jpg)  
Figure F: Qualitative comparison of point tracking from static and dynamic features. Compared to static features, dynamic features produce more temporally coherent point tracks by better following objects and distinguishing them from the background.

![](images/b8e637caf09fd23425c04e0209b713b2e326202a127db543dc11e905930004bd.jpg)  
Figure G: Qualitative comparison of depth map prediction from static and dynamic features. Compared to static features, dynamic features produce clearer depth maps, with more coherent depth predictions across objects and the background.

## F TEST-TIME POSE ALIGNMENT FOR LATENT MODELS

Evaluating models that do not receive conditioning poses can introduce scale misalignment with the ground truth, resulting in poor rendering metrics even when the predicted scene is geometrically accurate. Since the input images do not determine an absolute scale, a pose-free model may reconstruct the scene at an arbitrary scale. Pose-free methods therefore perform test-time pose alignment before evaluation and report rendering metrics using the aligned poses (Kim et al., 2026c; Ye et al., 2025). This alignment only brings the prediction into the coordinate frame and scale of the ground truth without modifying the underlying scene content.

Table G: Test-time pose-alignment results for pose-free methods. Alignment optimizes each target pose against the ground-truth target image for 2,000 steps, and the green row reports its effect. Best, second best, and third best results are highlighted.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Alignment</td><td colspan="3">ADT</td><td colspan="3">TUM</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td rowspan="3">NeoVerse (Yang et al., 2026)</td><td>x</td><td>12.217</td><td>0.2982</td><td>0.6391</td><td>14.398</td><td>0.4159</td><td>0.4587</td></tr><tr><td>√</td><td>16.179</td><td>0.4384</td><td>0.5296</td><td>16.120</td><td>0.5266</td><td>0.3981</td></tr><tr><td></td><td>(+3.962)</td><td>(+0.1402)</td><td>(-0.1095)</td><td>(+1.721)</td><td>(+0.1107)</td><td>(-0.0606)</td></tr><tr><td rowspan="3">C4G (Kim et al., 2026c)</td><td>x</td><td>12.963</td><td>0.4487</td><td>0.6870</td><td>12.378</td><td>0.3935</td><td>0.5984</td></tr><tr><td>S</td><td>18.900</td><td>0.5813</td><td>0.5833</td><td>17.508</td><td>0.5518</td><td>0.4951</td></tr><tr><td></td><td>(+5.937)</td><td>(+0.1326)</td><td>(-0.1038)</td><td>(+5.130)</td><td>(+0.1583)</td><td>(−0.1033)</td></tr><tr><td rowspan="3">FL4N (Ours)</td><td>x</td><td>19.693</td><td>0.5948</td><td>0.4490</td><td>17.847</td><td>0.5586</td><td>0.3449</td></tr><tr><td>√</td><td>23.288</td><td>0.6648</td><td>0.3683</td><td>20.392</td><td>0.6643</td><td>0.2860</td></tr><tr><td></td><td>(+3.594)</td><td>(+0.0700)</td><td>(−0.0807)</td><td>(+2.545)</td><td>(+0.1057)</td><td>(-0.0589)</td></tr></table>

The pose-free Gaussian baselines in our quantitative comparisons are evaluated with 2,000 step posealignment steps, whereas FL4N is reported without pose alignment in the main tables. However, FL4N is also pose-free and therefore suffers from the same scale ambiguity that motivates this evaluation procedure. We thus additionally evaluate FL4N with the same 2,000 step pose alignment to provide a consistent comparison.

For Gaussian models, Gaussians are predicted once from the conditioning images and kept fixed throughout the alignment process. Differentiable rasterization allows the rendering loss to be backpropagated to the camera pose, so each alignment step re-rasterizes the same frozen Gaussians under an updated pose, without modifying the network or scene representation.

The analogous procedure can be done to our latent renderer. The scene latent is predicted solely from the conditioning images and is independent of the rendering process. We therefore compute the latent once and keep it fixed throughout the alignment process, with the renderer serving the role of rasterization by re-rendering the cached latent under an updated target pose at each step. For both Gaussian and latent models, we optimize a rotation and translation offset for each target pose (Ye et al., 2025), initialized at zero such that the initial rendering exactly matches the feed-forward prediction. We use the same objective as the Gaussian baselines, with 2,000 Adam steps using pixel and perceptual losses. Table G reports the effect of this alignment under the NTS protocol on ADT and TUM. Alignment is worth 5.94 dB to C4G on ADT and 5.13 dB on TUM, against 3.59 dB and 2.55 dB for our model, so a large part of the published margin of C4G on these cells comes from fitting the pose to the image it is then scored on. How much alignment recovers therefore measures how much error the supplied cameras carry rather than the quality of the reconstruction, and our model without any alignment remains ahead of both aligned baselines on every metric.

## G LIMITATIONS

Our approach inherits the limitations of the pretrained static latent NVS backbone, particularly when rendering from extrapolated viewpoints that are unseen in the input views. Latent NVS models cannot synthesize unseen content (Jin et al., 2025), resulting in blurry or collapsed renderings under extrapolation, as also observed in (Szymanowicz et al., 2026). In contrast, explicit Gaussian representations construct scene content in 3D space, allowing the observed portions of the scene to remain usable even when extrapolation leaves holes in unseen regions.

The residual formulation can compensate for certain errors in the static prediction, but its capacity is naturally limited when the underlying scene representation or camera pose is substantially incorrect. These limitations can be alleviated by using stronger static NVS models with broader scene coverage and more robust viewpoint understanding.

4DGT  
MoVieS  
NeoVerse  
C4G  
FL4N (Ours)  
GT  
![](images/2f266e6f65c6a1e73f2ada57e9e20f5b51f3db99be4f44f2540111b81ca8b142.jpg)  
Figure H: More qualitative comparisons on NVIDIA across all scenes. Our method shows much better rendering quality than all other methods based on 3D Gaussian.

4DGT  
MoVieS  
NeoVerse  
C4G  
FL4N (Ours)  
GT  
![](images/0fdeb5e426c00de4fe6da849a9a18fb4a019a88a25fc6893610fb44db8a3b376.jpg)  
Figure I: More qualitative comparisons on N3DV across all scenes. Our method shows much better rendering quality than all other methods based on 3D Gaussian.

4DGT  
MoVieS  
NeoVerse  
C4G  
FL4N (Ours)  
GT  
![](images/f3a6c8e5e9b06a6da96ed3889fa7ee5de31b28d415daebe90d4f56e5863e6236.jpg)  
Figure J: More qualitative comparisons on TUM across all scenes.

4DGT  
MoVieS  
NeoVerse  
C4G  
FL4N (Ours)  
GT  
![](images/aa93221111d76b581c4dc052c75752d1c826de1185fcbb1fbba9ec6c7b2ec452.jpg)  
Figure K: More qualitative comparisons on DyCheck across all scenes.

4DGT  
MoVieS  
NeoVerse  
C4G  
FL4N (Ours)  
GT  
![](images/69409fddd95d9bdb0a882243e985c41987d06a6d28259dfe9ace86c28c9fd868.jpg)  
Figure L: More qualitative comparisons on ADT across all scenes. Our method shows much better rendering quality than all other methods based on 3D Gaussian. Gaussian-based approaches show severe artifacts such as holes caused by misaligned/mispositioned Gaussians.

4DGT  
MoVieS  
NeoVerse  
C4G  
FL4N (Ours)  
GT  
![](images/03c9902e4cfd92760d35ddc1170175fb577b45fdf112bfe8ece2f21a92f1896c.jpg)  
Figure M: More qualitative comparisons on Charge across all scenes. Our method shows much better rendering quality than all other methods based on 3D Gaussian. Gaussian-based approaches show severe artifacts such as holes caused by misaligned/mispositioned Gaussians.

4DGT

Time

Time

C4G  
Time  
![](images/ea5de3209271bdd5b0baa5cc09b50a269ccf616553d31ea0e684252c313e3605.jpg)

![](images/1202c1720931cec75d2e77d2748e96cdfcd00e021b84d4ae36f44671aca1f6b0.jpg)

![](images/804e9fc77055bf2130577dca7391e4b59d97337ad72e40738be0bc6c0869b38d.jpg)

![](images/a685c9701665138411d307241f4824fe309e54fd3cad602fbcb12fcd9395d24e.jpg)

![](images/4629c60c8edad9cfcdf205db4825b830fe55faad712c3e042afa4a08053df84e.jpg)

![](images/03b0f61ec05e0962a85e1f098a3f282368220be9f0e9520e724247c2a82c36ed.jpg)  
GT  
Figure N: Qualitative comparisons across view-time grid on NVIDIA. Red outlines indicate context frames. Compared with other methods, our methods show substantially better rendering quality with clear view-time disentanglement.

NeoVerse  
C4G  
Time  
![](images/e037f48ecdea28094b967de41b1b26cebe605f7979042b4d59824b7360a34029.jpg)  
4DGT

Time  
![](images/4d77046f2342d8803a24c881489484e5effd3eaa672b2f7249a7d8365ea0f1cb.jpg)  
MoVieS

![](images/9236a815e0d6f2886459a319754284731be220b2308afd7078d83fce9f7233ae.jpg)

![](images/e6bd883edd371593c5b1f1c3a5a4711f86337375eb7142f24a8e960823f58ddb.jpg)

![](images/d1abc0f0a496f1cf090687e746d92432683cb34de1788f4eef19329646e9f6f2.jpg)  
FL4N (Ours)

![](images/d2632ac537ad23d5a6f48d49ef5fcd4a2686b6092a09b9f1c8d2fddbdb9b621a.jpg)  
Figure O: Qualitative comparisons across view-time grid on N3DV. Red outlines indicate context frames. Compared with other methods, our methods show substantially better rendering quality with clear view-time disentanglement.

![](images/b4ff6d46f2731234a0a120dc9ef4858c0fa87e20c13af83ec50054d93f291242.jpg)

![](images/7c6b4008c2af770c6261c872804af9e1f1fd017424ffefbb6282264002eca4b7.jpg)

![](images/9efe74ebb2a2365eb9805a13d8d01313516f2186760cc1ac22716d74d97a37e7.jpg)

![](images/94b340ec2432c8fae257e1c670158a2a3939d792de2d15e4c2174226cab1e206.jpg)

![](images/cac6519fa4ba707da3aa8d4ea0cd023a4149bde9fd0db617b88338c1fac1898e.jpg)

![](images/ef0551e08e3d1a69af41c9e5d4f8f8830c3a4c5480792e1b4a90f78833fd28aa.jpg)  
Figure P: Qualitative comparisons across view-time grid on Charge. Red outlines indicate context frames. Compared with other methods, our methods show substantially better rendering quality with clear view-time disentanglement.

Time  
![](images/74e6314685d603d07ee8e4887c92d4eb7d398e54493b412dce61c5f1da92f8ec.jpg)  
FL4N (Ours)

Time  
![](images/edf34db0794dd67ba28d47f8bbc053b38c91b4d7caeebfaf8bd88c1dc851a026.jpg)  
GT  
Figure Q: Qualitative results across view-time grid on NVIDIA. Red outlines indicate context frames. Our method successfully synthesizes images at novel view and time, using target conditioning on predicted latent, without relying on explicit geometric primitives.

Time  
![](images/7c5c6f882ff083510fe165306464b4244d0fe79f63a9ad9863330a17041139b6.jpg)  
FL4N (Ours)

![](images/4f7cb72f37b6ec5e3063eec3241a9a38861f5b157b5e44f48d73f7bcc3bd85ed.jpg)  
GT  
Figure R: Qualitative results across view-time grid on CamxTime. Red outlines indicate context frames. Our method successfully synthesizes images at novel view and time, using target conditioning on predicted latent, without relying on explicit geometric primitives.

![](images/17b59410650d4e9ecfb062e5a4cff37b1dd5e11af74d83d0bc977e7b98ffc948.jpg)

![](images/97a828f0c8af34c87f5cb593cd5fa3d28fad7f084094bad74fcc4d51bd8da44c.jpg)  
Figure S: Qualitative results across view-time grid on Kubric4D. Red outlines indicate context frames. Our method successfully synthesizes images at novel view and time, using target conditioning on predicted latent, without relying on explicit geometric primitives.