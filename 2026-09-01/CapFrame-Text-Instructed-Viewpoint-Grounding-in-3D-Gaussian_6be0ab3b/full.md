# CapFrame: Text-Instructed Viewpoint Grounding in 3D Gaussian Scenes via Geometric Pseudo Labels

Jirong Li<sup>1</sup> , Satoshi Ikehata<sup>2,3</sup> , Shuhei Kurita<sup>1,3</sup> , and Ikuro Sato<sup>1,2</sup>

<sup>1</sup> Institute of Science Tokyo, Japan

2 Denso IT Laboratory, Inc., Japan

3 National Institute of Informatics, Japan jirong\_li@d-itlab.c.titech.ac.jp

Abstract. 3D Gaussian Splatting (3DGS) enables photorealistic realtime novel view synthesis, yet placing a virtual camera to capture a desired frame remains largely manual. Existing language-guided approaches in 3D scenes mainly focus on object-centric grounding, determining what to observe but rarely controlling how it should appear in a single frame, such as subject orientation or frame layout. To address this limitation, we introduce a new task, Text-Instructed Viewpoint Grounding (TIVG), which aims to identify a 6-DoF camera pose in a 3D Gaussian scene whose rendered frame aligns with a text instruction. To solve this task, we propose CapFrame, a partially diferentiable framework that converts language into geometric pseudo labels for camera pose optimization. CapFrame follows a Retrieve–Translate–Refine pipeline: it retrieves relevant views and ranks them through a Question-Evaluation process with MLLMs, translates the instruction into orientation and layout pseudo labels, and refines the camera pose via diferentiable optimization with layout and orientation losses in 3DGS. Experiments on 38 real-world scenes with 135 instructions indicate that CapFrame produces viewpoints better aligned with texts than heuristic viewpoint search and adapted trajectory generation baselines, validated by VLM metrics, MLLM judges, and user studies. Code is available at: https://github.com/jirongli/CapFrame

Keywords: 3D Gaussian Splatting · Viewpoint Grounding · Camera Pose Optimization · MLLM

## 1 Introduction

Recent advances in 3D representation technologies such as Neural Radiance Fields (NeRF) [38] and 3D Gaussian Splatting (3DGS) [19], enable photorealistic novel view synthesis for applications spanning VR/AR [16, 64], content creation [63,68], and scene editing [4,55]. Notably, 3DGS has gained prominence due to its real-time rendering and high visual fidelity, making it well-suited for interactive environments. However, despite its eficient rendering, identifying desirable viewpoints still requires tedious manual trial-and-error and scales poorly as scenes grow larger and more complex. Therefore, intent-aligned automatic viewpoint recommendation is crucial for intuitive scene interaction.

![](images/50a14dabf97a0b21ff12bde3176cdcc6d2ec9179a551cbd94aa01497778bd725.jpg)  
Fig. 1: Text-instructed viewpoint grounding in a 3DGS scene. Given a 3D Gaussian scene and a text instruction, the goal is to identify a camera pose whose rendered frame aligns with the described subject, composition and viewpoint.

Recent language-guided 3D methods [27, 42, 47, 58] excel at semantic object localization and trajectory generation [8, 28, 33, 34, 66], but primarily focus on object-centric grounding. They identify “what” to observe but overlook “how” it should be presented. Real-world instructions are inherently compositional, specifying orientation, spatial relationships, or photographic attributes (e.g., “a close-up photo of a cat from the front”). Existing work ensures object presence but fails to enforce such intentional framing constraints. While Multimodal Large Language Models (MLLMs) [1,13,31,61] ofer strong zero-shot spatial reasoning to interpret such constraints, translating instruction text into concrete 3DGS viewpoints remains unexplored.

In this work, we formalize a new task, Text-Instructed Viewpoint Grounding for 3DGS, which aims to identify a 6-DoF camera pose that captures a frame aligned with a text instruction. To accomplish this, we propose CapFrame, a partially diferentiable framework for text-instructed camera placement in 3D Gaussian scenes. CapFrame follows a three-stage Retrieve–Translate–Refine pipeline. In the Retrieve stage, we retrieve semantically relevant views from the training images and perform fine-grained ranking through a Question-Evaluation (QE) process guided by an MLLM, providing initialization for camera pose optimization. In the Translate stage, we convert compositional language into geometric pseudo labels, including subject–Gaussian associations, orientation pseudo labels, and layout pseudo labels. In the Refine stage, exploiting the differentiability of 3DGS, we optimize the camera pose by backpropagating layout and orientation losses directly to the pose.

Extensive experiments show that CapFrame efectively grounds compositional text instructions to camera poses in diverse 3D Gaussian scenes. Across 38 real-world scenes and 135 curated instructions, it consistently produces viewpoints whose rendered frames better satisfy compositional requirements than heuristic viewpoint search baselines and adapted trajectory generation methods. Quantitative evaluation with text-image similarity metrics, MLLM judges, and a user study further confirms stronger alignment between rendered frames and textual descriptions.

## 2 Related Work

3D Gaussian Representation. 3D scene representations range from implicit radiance fields [38, 40, 48] to explicit structures such as meshes [17, 18], point clouds [26, 41], and voxels [46, 51]. Addressing the rendering latency of implicit models and the topological rigidity of explicit ones, 3D Gaussian Splatting (3DGS) [19] represents scenes as anisotropic Gaussian primitives. With realtime rasterization and high visual fidelity, 3DGS has become a popular representation for novel view synthesis. Subsequent work applies 3DGS to tasks including 3D segmentation [6, 15, 62], scene editing [4, 55, 57], generative content creation [5, 63, 68], and dynamic scene modeling [36, 56]. Systems such as MonoGS [37] further show that diferentiable rasterization in 3DGS enables robust 6-DoF pose optimization. Building on this property, we exploit diferentiable 3DGS rendering to optimize camera viewpoints aligned to text instruction.

Vision-Language Understanding. Vision-Language Models (VLMs) learn joint visual-textual embeddings from image-text pairs. CLIP [43] established strong contrastive alignment, followed by models improving multimodal understanding [9, 24, 25, 52, 59, 65]. To extend such semantics to 3D scenes, recent work [27, 35, 42, 47, 58] distills language features into 3D Gaussian primitives, enabling open-vocabulary querying and amodal reasoning under occlusion. However, these approaches remain largely object-centric, focusing on what to attend to rather than how to frame it. Meanwhile, Multimodal Large Language Models (MLLMs) [1, 13, 31, 61] exhibit strong zero-shot spatial reasoning, enabling tasks like object reasoning [60] and physical simulation [67]. We study viewpoint alignment conditioned on compositional language, using MLLMs to convert photographic instructions into geometric pseudo supervisions for camera viewpoint optimization.

Camera Control. Camera control in virtual environments aims to satisfy cinematic and narrative constraints. Early methods relied on mathematical formulations [3, 11, 30] or rule-based systems encoding cinematic heuristics [10, 12]. Recent learning-based approaches [8, 28, 33, 66] generate language-conditioned camera motion from large-scale data. For example, ChatCam [33] enables conversational camera navigation, while GenDoP [66] synthesizes cinematic trajectories. Optimization-based methods such as JAWS [53] and SplaTraj [34] refine camera paths directly in 3D representations via visibility objectives. In particular, SplaTraj [34] combines 3DGS with continuous language fields [42] to maintain visibility of open-vocabulary targets. However, these methods do not address fine-grained compositional constraints for intentional photographic framing.

## 3 Preliminaries: 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) [19] represents a scene as anisotropic Gaussians ${ \mathcal { G } } ,$ each with mean $\mu _ { W } \in \mathbb { R } ^ { 3 }$ and covariance $\mathcal { L } _ { W } \in \mathbb { R } ^ { 3 \times 3 }$

$$
G ( x ) = \exp \left( - \frac 1 2 ( x - \mu _ { W } ) ^ { \top } \varSigma _ { W } ^ { - 1 } ( x - \mu _ { W } ) \right) .\tag{1}
$$

We parameterize $\Sigma _ { W }$ as $\Sigma _ { W } = R S S ^ { \top } R ^ { \top }$ to ensure validity.

Let $\pmb { T } _ { C W } \in S E ( 3 )$ be the 6-DoF camera pose. During rendering, each Gaussian is projected to the image plane with

$$
\mu _ { I } = \pi ( { \bf T } _ { C W } \mu _ { W } ) , \qquad \Sigma _ { I } = J { \cal R } \Sigma _ { W } { \cal R } ^ { \top } J ^ { \top } ,\tag{2}
$$

where $\pi ( \cdot )$ is perspective projection, R is rotation of ${ \bf { \mathit { T } } } _ { C W }$ , and J is projection Jacobian. Pixel color is obtained by alpha compositing $\mathcal { N }$ overlapping Gaussians:

$$
C = \sum _ { k \in \mathcal { N } } c _ { k } \alpha _ { k } \prod _ { j = 1 } ^ { k - 1 } ( 1 - \alpha _ { j } ) ,\tag{3}
$$

with $c _ { k }$ and $\alpha _ { k }$ the color and opacity of the k-th Gaussian.

Since rasterization is diferentiable, gradients propagate to $\pmb { T } _ { C W }$ . Following MonoGS [37], for a pose-dependent function $f ,$ we define the manifold derivative using $\tau \in \mathfrak { s e } ( 3 )$ :

$$
\frac { D f ( { \cal T } _ { C W } ) } { D { \cal T } _ { C W } } = \operatorname* { l i m } _ { \tau \to 0 } \frac { \log \bigl ( f ( \exp ( \tau ) \circ { \cal T } _ { C W } ) \circ f ( { \cal T } _ { C W } ) ^ { - 1 } \bigr ) } { \tau } .\tag{4}
$$

This enables gradient-based optimization of 6-DoF camera poses.

## 4 Text-Instructed Viewpoint Grounding in 3DGS

We formalize the task of Text-Instructed Viewpoint Grounding (TIVG) within a 3D Gaussian scene. Let ${ \mathcal { S } } = \{ G _ { i } \} _ { i = 1 } ^ { N }$ denote a 3DGS scene reconstructed from a set of training images ${ \mathcal { C } } ,$ , and let $T$ be a natural language instruction describing a desired viewpoint. A camera pose, comprising a rotation matrix $R \in S O ( 3 )$ and a translation vector $\pmb { t } \in \mathbb { R } ^ { 3 }$ , determines the diferentiable rasterization R of the scene:

$$
\begin{array} { r } { I = { \mathcal R } ( S , T _ { C W } ) , \quad \mathrm { w h e r e } \quad T _ { C W } = \left[ \pmb { R } \ t \right] \in S E ( 3 ) . } \end{array}\tag{5}
$$

The objective is to identify a camera pose ${ \pmb T } _ { C W }$ that produces a rendered image $\pmb { I } \in \mathrm { { \mathbb { R } } ^ { 3 \times H \times W } }$ aligned with the instruction T. For instance, as illustrated in Fig. 1, given the instruction “A large brown teddy bear sits on the right side, showing left view to the camera”, the system should not only localize the bear but also determine a precise 6-DoF configuration that satisfies both the orientation (left view) and the layout (right side of the frame).

![](images/0db1e173010eec7a08c5da7c5c2e1a75dd45d3996526bd36a38109ab6f554811.jpg)  
Fig. 2: Overview of CapFrame. Starting from compositional text and views in training set, we retrieve images consistent with the text. A Question-Evaluation process within the MLLM is introduced to perform fine-grained ranking, selecting relevant image subset. Subsequently, the MLLM translates the text into geometric regularizers. In conjunction with pretrained models, we construct geometric pseudo labels based on the image subset for orientation and layout. Finally, guided by these pseudo labels, we refine camera poses through diferentiable optimization in the 3D Gaussian scene.

Unlike object-centric localization [42, 58], TIVG does not generally admit a unique solution, as multiple viewpoints may satisfy the same compositional description. Given this inherent non-uniqueness, we evaluate this task using textimage similarity scores computed by external VLMs, alignment ratings from two MLLM judges, together with perceptual user studies, as detailed in Sec. 6.

## 5 Method

We introduce CapFrame to bridge the gap between abstract instructions and precise 6-DoF poses via a three-stage pipeline (Fig. 2): (1) Retrieve: We identify and rank semantic anchor views from training images C via a Question-Evaluation (QE) process using an MLLM (Sec. 5.1). (2) Translate: We convert linguistic constraints into geometric pseudo labels for orientation and layout (Sec. 5.2). (3) Refine: Initialized by retrieved poses, we optimize the camera by backpropagating layout and orientation losses to pose parameters (Sec. 5.3).

## 5.1 Retrieve: Semantic-aware Viewpoint Initialization

Exhaustively searching the continuous SE(3) space of a 3DGS scene is computationally prohibitive. Moreover, gradient-based camera pose optimization requires a suitable initialization to ensure stable convergence. Since 3DGS scenes are reconstructed from a finite set of training images C that provide near-complete scene coverage, we assume that these existing views form a suitable discrete subspace for initializing the optimization. Therefore, in this Retrieve stage, we aim to identify the image from C most relevant to the input text instruction.

Given a text instruction T, we first perform global semantic alignment using FG-CLIP [59] to identify a relevant subset of candidate poses. Since reasoning with MLLMs later is computationally expensive, this lightweight filtering step significantly reduces the candidate space. We extract a text embedding $f _ { T }$ and image embeddings $f _ { I , i }$ for each view $I _ { i } \in { \mathcal { C } } .$ . Similarity is computed via cosine similarity $\langle f _ { T } , f _ { I , i } \rangle$ , and the top-M view-pose pairs are retained as

$$
\begin{array} { r } { \mathcal { C } _ { g } = \{ ( I _ { i } , \pmb { T } _ { C W , i } ) \} _ { i = 1 } ^ { M } , } \end{array}\tag{6}
$$

where $M \approx | { \mathcal { C } } | / 8$ . While eficient, our preliminary experiments reveal that simple VLM-based global alignment often fails in cluttered scenes with multiple objects, as it struggles to discriminate nuanced compositional requirements.

To overcome the limitations of holistic VLM embeddings, we introduce a Question-Evaluation (QE) strategy that leverages the reasoning of multimodal large language models (MLLMs), such as Qwen3-VL [1]. Instead of relying on a single similarity score, QE decomposes T into a set of questions Q targeting specific semantic and photographic attributes $( e . g . , ^ { \ : 6 } I s$ the subject visible?” or $^ { 6 6 } I s$ the subject located on the right side of the $f r a m e ? )$ . These questions are automatically generated by MLLM using a fixed, task-agnostic prompt template, ensuring a consistent and hands-free evaluation process across diverse scenes.

The MLLM then evaluates each view $I _ { i } \in \mathcal { C } _ { g }$ against Q to produce granular scores. By averaging these scores, we obtain a ranking that reflects complex compositional alignment. The top-K view-pose pairs are retained as

$$
\begin{array} { r } { \mathcal { C } _ { f } = \{ ( I _ { i } , \pmb { T } _ { C W , i } ) \} _ { i = 1 } ^ { K } . } \end{array}\tag{7}
$$

This reasoning-based filtering provides a high-quality initialization for the subsequent optimization stage, ensuring that the starting viewpoint already satisfies basic semantic constraints.

## 5.2 Translate: From Language to Geometric Pseudo Labels

The Retrieve stage provides a strong initialization by selecting candidate viewpoints from the training set. Our goal, however, is to search the continuous camera pose space in $S E ( 3 )$ and find a viewpoint whose rendered frame matches the text instruction. Because natural language rarely specifies optimization-ready numeric targets $( e . g .$ , exact angles or pixel coordinates), directly constructing diferentiable objectives from text is challenging. We therefore introduce a translation step that converts compositional instructions into geometric pseudo labels—structured targets guiding continuous pose refinement in 3DGS.

We follow key decisions in photographic composition: selecting salient subjects, defining viewing direction, and arranging subjects within the frame. Accordingly, we extract subject Gaussians and construct two pseudo-label types:

(i) orientation pseudo labels for viewpoint control and (ii) layout pseudo labels for framing constraints.

Subject–Gaussian Association. Let $\mathcal { C } _ { f } = \{ ( I _ { j } , T _ { C W , j } ) \} _ { j = 1 } ^ { K }$ denote the top-K retrieved view-pose pairs. Given instruction $T ,$ we prompt an MLLM to extract key subjects $\mathcal { O } = \{ O _ { 1 } , \ldots , O _ { n } \}$ , including mentioned entities and visually dominant objects afecting framing. Each subject is associated with a subset of 3D Gaussians to enable subject-centric reasoning and diferentiable mask construction. Unlike open-vocabulary localization methods that embed language features into all Gaussians [42, 58], we localize only the small set ${ \mathcal { O } } _ { : }$ which sufices for constructing pseudo labels.

For each subject $O _ { i }$ and retrieved view $I _ { j }$ , we obtain a segmentation mask $M _ { i j } ^ { \mathrm { S A M } }$ using Grounded SAM [22,32,44]. Pixels $( u , v ) \in M _ { i j } ^ { \mathrm { S A M } }$ are back-projected using depth rendered from 3DGS to obtain a 3D point set

$$
\mathcal { P } _ { i j } = \{ ( X _ { W } , Y _ { W } , Z _ { W } ) ^ { \top } \mid ( u , v ) \in M _ { i j } ^ { \mathrm { S A M } } \} .\tag{8}
$$

Since rendered depth may be noisy, these points may not coincide exactly with Gaussian means. We therefore retrieve a subject-specific Gaussian subset ${ \mathcal { G } } _ { Q , i } \subset$ G via KNN search between $\cup _ { j } \mathcal { P } _ { i j }$ and Gaussian means $\{ \mu _ { W , k } \}$ . The subject centroid is computed as

$$
C _ { W , i } = \frac { 1 } { \vert \mathcal { G } _ { Q , i } \vert } \sum _ { k \in \mathcal { G } _ { Q , i } } \mu _ { W , k } .\tag{9}
$$

Rendering only ${ \mathcal { G } } _ { Q , i }$ yields a diferentiable subject mask for layout objectives.

Orientation Pseudo Labels. Many instructions specify viewpoint and subject orientation through cues such as front view, side view, or look down. We convert such language into orientation pseudo labels by combining (i) relative ofsets inferred from text and (ii) estimated subject orientation from images.

Given T and ${ \mathcal { O } } ,$ we prompt the MLLM to infer relative orientation ofsets $\varDelta d _ { i } = \left( \varDelta \phi _ { i } , \varDelta \theta _ { i } , \varDelta \gamma _ { i } \right)$ , representing azimuth, elevation and roll adjustments. Orientation constraints are applied only to asymmetric subjects with meaningful front or back semantics. Symmetric objects $( e . g .$ , balls) rely solely on layout constraints (subject types are identified by MLLM).

To estimate current orientation of subjects, we apply Orient Anything [54] to the top-K images, obtaining $( \phi _ { O , i } , \theta _ { O , i } )$ (γ<sub>O,i</sub> is not used as we only need the forward direction of the subject). The target orientation combines this estimate with the text-derived ofsets:

$$
P _ { O , i } ^ { \mathrm { o r i e n t } } = ( \phi _ { O , i } + \varDelta \phi _ { i } , \ \theta _ { O , i } + \varDelta \theta _ { i } , \ \varDelta \gamma _ { i } ) ,\tag{10}
$$

where azimuth is wrapped modulo and elevation clamped to a valid range. The result is mapped to world coordinates, producing $P _ { W , i } ^ { \mathrm { o r i e n t } }$ used as a diferentiable regularizer during refinement.

Layout Pseudo Labels. Photographic intent also specifies subject placement within the frame $( e . g . ,$ , subject on the right, centered, close-up). We therefore prompt the MLLM to infer a target 2D layout in normalized image coordinates. For each subject $O _ { i }$ , the layout pseudo label is a bounding box

$$
P _ { i } ^ { \mathrm { l a y o u t } } = [ x _ { \mathrm { m i n } } , y _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } , y _ { \mathrm { m a x } } ] ,\tag{11}
$$

which is converted to pixel coordinates using image width W and height H.

To remain consistent with roll constraints, we adjust the layout if $\varDelta \gamma _ { i } \neq 0$ Let $\boldsymbol { c } = \left( c _ { x } , c _ { y } \right)$ denote the center of the bounding box and b denote a corner of the bounding box. Each corner is rotated around c by $\varDelta \gamma _ { i } :$

$$
\pmb { b } _ { \gamma } = \pmb { c } + \pmb { R } ( \Delta \gamma _ { i } ) ( \pmb { b } - \pmb { c } ) ,\tag{12}
$$

where $R ( \varDelta \gamma _ { i } )$ is the 2D rotation matrix. The rotated corners define the final layout target used during refinement.

## 5.3 Refine: Gradient-Based Camera Pose Optimization

Starting from the pose retrieved in Sec. 5.1, we optimize an incremental update $\tau = [ \Delta { \boldsymbol { r } } , \Delta t ] ^ { \top } \in \mathfrak { s e } ( 3 )$ and update the camera pose by left composition $\mathbf { \Delta } T _ { C W } \gets$ $\exp ( \tau ) \circ T _ { C W }$ . We minimize a multi-objective loss

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { l a y o u t } } + \mathcal { L } _ { \mathrm { o r i e n t } } ,\tag{13}
$$

where both terms are derived from pseudo labels in Sec. 5.2.

Layout Loss. The layout loss enforces composition constraints by encouraging each subject to occupy its target region $P _ { i } ^ { \mathrm { { l a y o u t } } }$ (Sec. 5.2). For subject $O _ { i }$ , we render only its Gaussian subset ${ \mathcal { G } } _ { Q , i }$ and obtain a diferentiable opacity mask $M _ { i } ^ { g } \in [ 0 , 1 ] ^ { H \times W }$ via alpha compositing:

$$
M _ { i } ^ { g } ( u , v ) = \sum _ { k = 1 } ^ { N _ { i } } \alpha _ { k } ( u , v ) \prod _ { j < k } \big ( 1 - \alpha _ { j } ( u , v ) \big ) ,\tag{14}
$$

where $N _ { i }$ is the number of contributing Gaussians and $\alpha _ { k } ( u , v )$ denotes the opacity of the k-th Gaussian. Let $C _ { P , i }$ denote the soft centroid of $M _ { i } ^ { g }$ :

$$
C _ { P , i } = \frac { \sum _ { ( u , v ) \in \Omega } ( u , v ) M _ { i } ^ { g } ( u , v ) } { \sum _ { ( u , v ) \in \Omega } M _ { i } ^ { g } ( u , v ) + \epsilon } ,\tag{15}
$$

where $\varOmega$ is the image domain. If $C _ { P , i }$ lies inside $P _ { i } ^ { \mathrm { { l a y o u t } } }$ , the loss is zero; otherwise we penalize the distance to the nearest point on the region boundary $\partial P _ { i } ^ { \mathrm { l a y o u t } }$ <sup>t</sup>:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c e n t e r } } ^ { i } = \left\{ \begin{array} { l l } { 0 , } & { \mathrm { i f ~ } C _ { P , i } \in P _ { i } ^ { \mathrm { l a y o u t } } , } \\ { \operatorname* { m i n } _ { \pmb { p } \in \partial P _ { i } ^ { \mathrm { l a y o u t } } } \left\| \pmb { C } _ { P , i } - \pmb { p } \right\| _ { 2 } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{16}
$$

Empirically, centroid constraints alone are insuficient when subjects are elongated or partially outside the target region. We therefore additionally encourage mask coverage inside the box and penalize leakage outside it. Using the indicator $\mathbb { I } ( \cdot )$ , we define

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { i n } } ^ { i } = 1 - \frac { \sum _ { \left( u , v \right) \in \Omega } { M _ { i } ^ { g } \left( u , v \right) \mathbb { I } \left( \left( u , v \right) \in P _ { i } ^ { \mathrm { l a y o u t } } \right) } } { \sum _ { \left( u , v \right) \in \Omega } { M _ { i } ^ { g } \left( u , v \right) + \epsilon } } , } \\ & { \mathcal { L } _ { \mathrm { o u t } } ^ { i } = \frac { \sum _ { \left( u , v \right) \in \Omega } { M _ { i } ^ { g } \left( u , v \right) \mathbb { I } \left( \left( u , v \right) \notin P _ { i } ^ { \mathrm { l a y o u t } } \right) } } { \sum _ { \left( u , v \right) \in \Omega } { M _ { i } ^ { g } \left( u , v \right) + \epsilon } } . } \end{array}\tag{17}
$$

The total layout loss aggregates all subjects:

$$
\mathcal { L } _ { \mathrm { l a y o u t } } = \sum _ { i = 1 } ^ { n } \left( \lambda _ { c } \mathcal { L } _ { \mathrm { c e n t e r } } ^ { i } + \lambda _ { \mathrm { i n } } \mathcal { L } _ { \mathrm { i n } } ^ { i } + \lambda _ { \mathrm { o u t } } \mathcal { L } _ { \mathrm { o u t } } ^ { i } \right) ,\tag{18}
$$

where $\lambda _ { c } , \lambda _ { \mathrm { i n } }$ , and $\lambda _ { \mathrm { o u t } }$ are scalar weights and ϵ ensures numerical stability.

Orientation Loss. The orientation loss enforces viewpoint constraints from orientation pseudo labels (Sec. 5.2). It contains three terms: gravity (discouraging tilt), forward-facing (aligning the camera with the desired viewpoint), and look-at (keeping the subject visible).

Gravity. When roll is not specified $( i . e . , \ \varDelta \gamma _ { i } \ = \ 0 )$ , we encourage an upright camera by aligning the camera up direction $D _ { C , c u }$ with the world $\mathrm { u p }$ direction in the camera frame $D _ { C , w u }$

$$
\mathcal { L } _ { \mathrm { g r a v i t y } } = 1 - \langle D _ { C , c u } , D _ { C , w u } \rangle .\tag{19}
$$

If roll is specified, this term is disabled and roll is handled by the roll-corrected layout labels (Sec. 5.2).

Forward-facing. For asymmetric subjects $O _ { i } \in \mathcal { O } _ { w }$ , we align the camera forward direction $D _ { C , c f }$ with the target orientation $P _ { C , i } ^ { \mathrm { o r i e n t } }$

$$
\mathcal { L } _ { \mathrm { f o r w a r d } } ^ { i } = 1 - \langle D _ { C , c f } , P _ { C , i } ^ { \mathrm { o r i e n t } } \rangle .\tag{20}
$$

Look-at. To keep the subject in view, we align the camera with the subject centroid. Let $C _ { c a m }$ and $C _ { W , i }$ denote the camera center and subject centroid (Sec. 5.2). The normalized direction

$$
d _ { W , i } = \frac { C _ { W , i } - C _ { c a m } } { \Vert \boldsymbol { C } _ { W , i } - C _ { c a m } \Vert _ { 2 } }\tag{21}
$$

is expressed in the camera frame as $D _ { C , c a m , i } , \mathrm { y i e l d i n g }$

$$
\mathcal { L } _ { \mathrm { l o o k - a t } } ^ { i } = 1 - \langle D _ { C , c f } , D _ { C , c a m , i } \rangle .\tag{22}
$$

The final orientation loss is

$$
\mathcal { L } _ { \mathrm { o r i e n t } } = \lambda _ { g } \mathcal { L } _ { \mathrm { g r a v i t y } } + \sum _ { O _ { i } \in O _ { w } } \left( \lambda _ { f } \mathcal { L } _ { \mathrm { f o r w a r d } } ^ { i } + \lambda _ { l } \mathcal { L } _ { \mathrm { l o o k - a t } } ^ { i } \right) ,\tag{23}
$$

where $\lambda _ { g } , \lambda _ { f }$ , and $\lambda _ { l }$ are scalar weights.

## 6 Results

Implementation Details. We use Qwen3-VL [1] as the MLLM for text reasoning in the Retrieve stage and pseudo-label generation in the Translate stage. For scene representation, we adopt the standard 3DGS implementation [19] with camera pose gradients from MonoGS [37] (Eq. (4)). As our focus is viewpoint grounding, the reconstruction method is not critical. In Retrieve, the number of top-ranked views for fine-grained ranking is set to $K = 2$ . Camera pose refinement in Refine is optimized using Adam [21] with learning rates 0.007 (rotation) and 0.005 (translation). We set $\lambda _ { c } = 1$ and $\lambda _ { \mathrm { i n } } = 6$ , while all other weights are $2 \left( { \lambda _ { \mathrm { o u t } } } , { \lambda _ { g } } , { \lambda _ { f } } , { \lambda _ { l } } \right)$ . Optimization runs for up to 1500 iterations with early stopping when the loss change falls below $1 \times 1 0 ^ { - 4 } \mathrm { ~ o r ~ } 1 0 ^ { - 5 }$ depending on scene scale. Experiments run on a single NVIDIA H100 GPU (80GB).

Baselines. As no established baselines exist for text-instructed viewpoint grounding in 3DGS scenes, we construct two heuristic baselines inspired by scene exploration methods [50]: Interpolation-based Viewpoint Search (IVS) and Samplingbased Viewpoint Search (SVS). IVS generates candidate viewpoints by interpolating between top-K retrieved camera poses from the Retrieve stage (Sec. 5.1). SVS instead samples camera poses densely on a sphere centered around the key subjects identified in the Translate stage (Sec. 5.2). For both baselines, each candidate pose is rendered using 3DGS, and the final viewpoint is selected as the frame with the highest text-image similarity measured by FG-CLIP [59]. In addition, we adapt the relevant components of two trajectory generation methods, ChatCam [33] and SplaTraj [34], to our TIVG setting. We build on ChatCam’s available Anchor Determination, which optimizes camera poses via CLIP [43] gradients, and reproduce the relevant part of SplaTraj.

Datasets. We evaluate on real-world 3D reconstruction datasets: 5 scenes from Mip-NeRF 360 [2], 13 from Deep Blending [14], 4 from Tanks and Temples [23], 4 from LERF-OVS [20], and 12 from DL3DV-10K [29], totaling 38 indoor and outdoor scenes. For scenes without camera poses, we estimate them by COLMAP [45]. We manually curate 3-4 instructions per scene, resulting in 135 text descriptions.

Metrics. As text-instructed viewpoint grounding lacks a unique ground-truth viewpoint, evaluation is non-trivial. For quantitative evaluation, we compute text-image similarity using CLIP [43] (ViT-H/14) and SigLIP2 [52]. Using two VLMs reduces bias from a single scoring function and provides a more robust estimate of semantic alignment. Similarly, we introduce two MLLM judges (GPT-5.4-mini [39] and Gemini-2.5-Flash [7]) as blind photographic evaluators that rate the compositional alignment scores (AS) and select the most aligned output frame across methods, computing the win rates (WR) of CapFrame over the baselines. We additionally conduct a user study to evaluate perceptual alignment. The study involves 33 participants, each completing questionnaires with 12 groups. In each group, participants rate three candidate frames on a 5-point

Text Instruction:

A white floor lamp wearing a black hat is positioned in the lower left of the frame, while a dark blue cloth hangs in the upper right.

Output Image:  
![](images/c0d64dc2e8b9729a77ddb2bb3308f3aef6bf43dc938c9a49939603a7b54f64ee.jpg)

Iteration: Camera Pose:

A green potted plant in a white pot is positioned in the center of the frame. The camera looks straight down at the plant.

![](images/890a344d065063829f5f6c8ff0ae36dd57e0d9d81a25dee382c3a82f937482e9.jpg)

![](images/15fe1303ea675d9013cfe36d18d2d99493fb2e7862aeddd02c471b39fb9829c2.jpg)

![](images/d19aea7d9a97ac350a4b6199053841e0d2a0a141eaa9d5677f719a615898be0c.jpg)

![](images/7500a5c777c78eb4ad88e0b80bae20fa79959f606e5e6110198dddad7865647c.jpg)

![](images/173beabcab569b9b2a41ecfaeb216512b6707ca220b36d35a8db122b1318a845.jpg)

The columnar cactus stands on the left, while the barrel cactus is on the right, with a white window with vertical iron bars behind them, both positioned at the center of the image. The camera is facing the window.

![](images/cb7f2c2bee29d4cee001d50e2af2cb13f2d6e59716764e3a4c59451b4f338275.jpg)

![](images/50bf2cc4e36e9b0020c04400b3a7b491277c328a0cf9198a8d02dbee9d7d5031.jpg)

![](images/140c6ed6f5fb39f62d9bb4087ab5026742b75a1a4754fa255f3fbdd0b2fc84e9.jpg)

A dark and calm pond is located on the left side of the frame, while a light gray paved path runs along the right side.

![](images/9b83a6864ce973bd9188b01c8e110e097e523ea48187f27b37d9727b8eac4a80.jpg)

![](images/5aff9d487c6343dc814d20b9c2e1bc592b632f383546bcb7aa70eb5c2cc84b91.jpg)

A pink abstract painting is positioned above a table draped in a red cloth, and the painting is centered in the image.

![](images/b9ac2d6ee407889ff148ecba8fd35d48969c3b50ea0f5da246cef3df15e3513d.jpg)

![](images/c296194dbe1e522dc3d5536ce7653d9511d8dc726f5648832f583716d7da6fcf.jpg)

![](images/388ec1f2208d3a82f324f44fba37e8f26f8324f1a94d8f3ff5d7774033778ae5.jpg)

In this close-up, both the meal and the sandwich are placed on the table, with the sandwich located behind the main dish and above in theframe.

![](images/efe726fafb0446665c4947ebaf7f36af00bcca958caeadeb0b2ebe8dff182e0f.jpg)

![](images/e529220b5f84598d01f3fa69442df4cf061fed8f8480d428a92254af517fbd39.jpg)

![](images/ff8e0855f18363ca377729c6942c80a735ad53737c8b363505056ffa9001c931.jpg)  
Fig. 3: Text-instructed viewpoint grounding in diverse 3DGS scenes. Given a natural language instruction (left), CapFrame identifies a camera pose that produces a frame consistent with the described subject, composition and viewpoint. Starting from retrieved initial views, the camera pose is refined through diferentiable optimization in the 3D Gaussian scene to generate the final rendered frame (middle), with the optimized camera pose shown on the right.

SplaTraj

## CapFrame (Ours)

IVS

SVS

ChatCam

![](images/ed7d6fd49829d1e5abd56db8db2ee680151b201ff11ac8d70d3ef3fe5270f8b6.jpg)  
Fig. 4: Qualitative comparisons. CapFrame renders images aligned with specified subjects, composition and viewpoint, whereas other baselines are largely limited to object localization and often fail to satisfy composition and viewpoint requirements.

scale and select the frame that best matches the instruction.

Text-Instructed Viewpoint Grounding. As shown in Fig. 3, CapFrame grounds compositional text instructions to camera poses, enabling automatic viewpoint selection in diverse 3D Gaussian scenes. By leveraging the zero-shot reasoning ability of MLLMs, the framework generalizes to varied scenarios and supports intuitive text-guided exploration of complex 3D environments.

Comparisons with Baselines. Figure 4 presents qualitative comparisons under identical instructions. While IVS and SVS often retrieve viewpoints where the target subject is visible, they frequently fail to satisfy the required composition or orientation. For example, IVS places the green apple away from the left-side position, while SVS fails to place the slices of pork on the right. IVS can also inherit biases from retrieved camera poses: if the original views are tilted, interpolated viewpoints may preserve such unnatural orientations, as observed in the two cushions example. In the same example, ChatCam [33] produces a tilted view of the cushions, as CLIP-based [43] guidance is invariant to orientation. SplaTraj [34] settles on a viewpoint where the cushions are no longer visible, because its learned language field [42] fails to distinguish the target objects. In contrast, CapFrame optimizes camera poses using geometric pseudo labels encoding both layout and orientation, producing frames better aligned with the compositional intent of the instruction. Quantitative results in Tab. 1 further show that CapFrame achieves the highest text-image alignment on VLM-based similarity. The same trend holds across two MLLM judges, where CapFrame achieves the best alignment score and win rate. Moreover, our user study reveals a consistent preference for CapFrame over IVS and SVS.

Table 1: Quantitative comparisons. The left two columns report text-image similarity scores from two VLMs (CLIP and SigLIP2). The middle four columns report alignment scores (AS) and win rates (WR) from two MLLM judges (GPT and Gemini). The right two columns summarize user study results, including average rating and preference percentage. “—” indicates that the corresponding baselines are not evaluated in our user study.
<table><tr><td rowspan="3">Method</td><td colspan="2">VLM metrics</td><td colspan="4">MLLM judges</td><td colspan="2">User study</td></tr><tr><td>CLIP↑</td><td>SigLIP2↑</td><td>GPT AS↑</td><td>GPT WR↑</td><td>Gemini AS↑</td><td>Gemini WR↑</td><td>Rating↑</td><td>Preference↑</td></tr><tr><td>SplaTraj [34]</td><td>0.241</td><td>0.276</td><td>3.08</td><td>3.0%</td><td>2.78</td><td>3.0%</td><td></td><td></td></tr><tr><td>ChatCam [33]</td><td>0.258</td><td>0.283</td><td>3.42</td><td>3.7%</td><td>3.27</td><td>5.2%</td><td></td><td></td></tr><tr><td>SVS</td><td>0.275</td><td>0.435</td><td>4.06</td><td>14.8%</td><td>3.88</td><td>16.3%</td><td>2.52</td><td>16.7%</td></tr><tr><td>IVS</td><td>0.265</td><td>0.396</td><td>3.49</td><td>5.9%</td><td>3.19</td><td>8.1%</td><td>2.53</td><td>5.6%</td></tr><tr><td>CapFrame (Ours)</td><td>0.282</td><td>0.448</td><td>4.75</td><td>72.6%</td><td>4.22</td><td>67.4%</td><td>4.50</td><td>77.7%</td></tr></table>

Table 2: SigLIP2 score before and after camera pose refinement.
<table><tr><td>Stage</td><td>Mip-NeRF</td><td>Deep Blending</td><td>Tanks</td><td>LERF-OVS</td><td>DL3DV-10K</td></tr><tr><td>Retrieve</td><td>0.121</td><td>0.274</td><td>0.307</td><td>0.503</td><td>0.393</td></tr><tr><td>Refine</td><td>0.244</td><td>0.406</td><td>0.339</td><td>0.708</td><td>0.500</td></tr></table>

Component Analysis. We conduct studies to analyze the key components of CapFrame following the pipeline order: the Question-Evaluation (QE) strategy in Retrieve, the layout and orientation losses derived from geometric pseudo labels in Translate, and the pose refinement process in Refine.

We first evaluate QE in the Retrieve stage. As illustrated in Fig. 5, using only FG-CLIP [59] for coarse retrieval often returns semantically related views that do not contain the target subject, especially in cluttered scenes. Prompting the MLLM to assign a single holistic score shows similar limitations because compositional constraints are not explicitly evaluated. In contrast, QE decomposes the instruction into multiple questions, enabling the MLLM to assess candidate views along several semantic aspects. As a result, retrieved views more reliably contain the target subject and provide better initialization for pose optimization. Next, we conduct the ablation study to analyze the layout and orientation losses used in pose refinement (Fig. 6). With both losses, the optimized pose satisfies the desired composition and viewpoint. The layout loss regulates subject placement, while the orientation loss enforces the viewing direction. Removing either loss degrades alignment: without the orientation loss the camera fails to reach the specified view, and without the layout loss the subject placement deviates from the intended composition. Finally, we examine the efect of Refine stage. Because the desired viewpoint may not exist among retrieved views, refinement performs gradient-based camera pose optimization using geometric pseudo labels, enabling continuous pose adjustment in SE(3). We measure text-image alignment using SigLIP2 [52]. As shown in Tab. 2, refined views consistently achieve higher alignment scores than the retrieved initial views across all datasets.

Table 3: Sensitivity analysis under diferent perturbations. We measure the deviation from the unperturbed output, where ∆R and ∆t are the rotation and translation diferences and ∆CLIP and ∆SigLIP2 are the changes in the VLM-based metrics.
<table><tr><td>Type</td><td>Perturbation</td><td>∆R</td><td>∆t</td><td>∆CLIP</td><td>∆SigLIP2</td></tr><tr><td>Input text</td><td>Paraphrased text</td><td>9.61°</td><td>0.732</td><td>+0.013</td><td>+0.032</td></tr><tr><td rowspan="2">Top-K images</td><td>Top-3 images</td><td>8.99°</td><td>1.268</td><td>-0.014</td><td>-0.061</td></tr><tr><td>Top-5 images</td><td>13.28°</td><td>1.801</td><td>-0.018</td><td>-0.165</td></tr><tr><td rowspan="3">Pseudo labels</td><td>Layout label removal</td><td>4.08°</td><td>5.701</td><td>-0.024</td><td>-0.236</td></tr><tr><td>Orientation label removal</td><td>33.71°</td><td>2.312</td><td>-0.021</td><td>-0.128</td></tr><tr><td>Moderate noise</td><td>8.39°</td><td>0.219</td><td>-0.012</td><td>-0.048</td></tr></table>

Sensitivity Analysis. We analyze the sensitivity of CapFrame to diferent perturbations in Tab. 3, comparing the changes in camera pose and VLM-based metrics relative to the unperturbed output. First, paraphrasing each instruction twice with GPT-5.4-mini [39] slightly improves the alignment scores, demonstrating insensitivity to variation in input text. Next, when initializing from lower-ranked views (Top-3 and Top-5), the pose deviation grows gradually with K and the alignment scores decrease, indicating that camera refinement partially compensates for imperfect retrieval, though a good initialization remains beneficial. Finally, removing the pseudo labels reveals their complementary roles: excluding the layout label raises translation deviation, whereas excluding the orientation label increases rotation deviation. Moreover, under moderate label noise (5–10 pixels for layout and 5–10<sup>◦</sup> for orientation), CapFrame stays close to its original output, confirming its robustness to imperfect pseudo labels.

## 7 Conclusion

We introduce a new task, text-instructed viewpoint grounding in 3D Gaussian scenes, and present CapFrame, a framework for solving it. CapFrame leverages the zero-shot reasoning of MLLMs to translate natural language instructions into geometric pseudo labels, including orientation and layout constraints, which guide camera pose refinement through diferentiable optimization in 3DGS. Extensive experiments demonstrate that CapFrame generalizes across diverse scenes and produces viewpoints that more faithfully align with compositional text instructions, enabling intuitive language-driven exploration of 3D environments.

Text Instruction: A petal-shaped lawn with small red flowers is centered in the frame, viewedfrom a slightly downward angle.

FG-CLIP  
![](images/78090a9b494718be9ded8a9bd4778f27030979fe8e2aa857ee2ea3a5b47f8a81.jpg)

![](images/eae91c641da2c73387cd6586e10d02de7d0808951dea422567b04e502210b251.jpg)  
(a) Initial Retrieval

FG-CLIP + MLLM FG-CLIP + MLLM + QE  
![](images/ccc3388a475230d828525127364a9e0b81cf877dc8f6fa4a78a0f7576ba7e368.jpg)

![](images/c84fae857a6ad7602c532821eeb8aeaf5ccffd20abde9fa12c30ab3e0d9394e5.jpg)  
(b) Refined Images  
Fig. 5: Ablation of initialization. We present results of Top-K image selection across diferent retrieval methods. In complex scenes, (a) FG-CLIP [59] fails to localize the subject. Without QE for multi-dimensional scoring, the subject can still be absent from the image. (b) A good initialization is essential for camera pose convergence.

## Text Instruction:

The front view of a brown teddy bear, and it is positioned on the right side of the frame. The camera looks slightly down at the bear.

![](images/d3e79d44d98faff02f48d7efcd83e9bbf33f5dc2ac0d2835d289876ee61c8c45.jpg)  
(a) $\mathcal { L } _ { l a y o u t }$

![](images/c91af7ab6efcd5b810e5db795445285b609bef4cb7cf3d7ea0f03d1cc393e65a.jpg)  
(b) $\mathcal { L } _ { o r i e n t }$

![](images/186f567d1c3c22060dbf1942c5c38066e293ed794fc1a88bb9895e7a8f58b9bd.jpg)  
(c) $\mathcal { L } _ { l a y o u t } + \mathcal { L } _ { c }$ rient  
Fig. 6: Ablation of loss. Layout and orientation losses significantly influence the optimization performance. (a) Without orientation loss, the camera fails to achieve the front view of the bear. (b) Without layout loss, the bear is not positioned on the right side. (c) With both losses, the bear satisfies the specified composition and viewpoint.

Limitations. CapFrame relies heavily on MLLMs in several stages, including retrieval and translation. This design enables test-time optimization without additional training, but also introduces sensitivity to the prompts provided to the MLLM, which may afect the resulting pseudo labels and camera placement. While this work focuses on introducing the new task and establishing a first solution, future research could explore more robust prompting strategies and prompt optimization to further improve performance.

## Acknowledgements

This work was supported by DENSO IT LAB Recognition, Control, and Learning Algorithm Collaborative Research Chair (Science Tokyo) and the experiments were conducted using TSUBAME 4.0 supercomputer.

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5470–5479 (2022)

3. Blinn, J.: Where am i? what am i looking at? (cinematography). IEEE Computer Graphics and Applications 8(4), 76–81 (2002)

4. Chen, Y., Chen, Z., Zhang, C., Wang, F., Yang, X., Wang, Y., Cai, Z., Yang, L., Liu, H., Lin, G.: Gaussianeditor: Swift and controllable 3d editing with gaussian splatting. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21476–21485 (2024)

5. Chen, Z., Wang, F., Wang, Y., Liu, H.: Text-to-3d using gaussian splatting. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21401– 21412 (2024)

6. Choi, S., Song, H., Kim, J., Kim, T., Do, H.: Click-gaussian: Interactive segmentation to any 3d gaussians. In: European Conference on Computer Vision. pp. 289–305 (2024)

7. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

8. Courant, R., Dufour, N., Wang, X., Christie, M., Kalogeiton, V.: E.t. the exceptional trajectories: Text-to-camera-trajectory generation with character awareness. In: European Conference on Computer Vision. pp. 464–480 (2024)

9. Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P.N., Hoi, S.: Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems 36, 49250–49267 (2023)

10. Drucker, S.M., Galyean, T.A., Zeltzer, D.: Cinema: A system for procedural camera movements. In: Proceedings of the 1992 symposium on Interactive 3D graphics. pp. 67–70 (1992)

11. Galvane, Q., Christie, M., Lino, C., Ronfard, R.: Camera-on-rails: Automated computation of constrained camera paths. In: Proceedings of the 8th ACM SIGGRAPH Conference on Motion in Games. pp. 151–157 (2015)

12. Galvane, Q., Ronfard, R., Christie, M., Szilas, N.: Narrative-driven camera control for cinematic replay of computer games. In: Proceedings of the 7th International Conference on Motion in Games. pp. 109–117 (2014)

13. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)

14. Hedman, P., Philip, J., Price, T., Frahm, J.M., Drettakis, G., Brostow, G.: Deep blending for free-viewpoint image-based rendering. ACM Transactions on Graphics 37(6), 1–15 (2018)

15. Jain, U., Mirzaei, A., Gilitschenski, I.: Gaussiancut: Interactive segmentation via graph cut for 3d gaussian splatting. Advances in Neural Information Processing Systems 37, 89184–89212 (2024)

16. Jiang, Y., Yu, C., Xie, T., Li, X., Feng, Y., Wang, H., Li, M., Lau, H., Gao, F., Yang, Y., et al.: Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality. In: ACM SIGGRAPH. pp. 1–1 (2024)

17. Kanazawa, A., Tulsiani, S., Efros, A.A., Malik, J.: Learning category-specific mesh reconstruction from image collections. In: European Conference on Computer Vision. pp. 371–386 (2018)

18. Kato, H., Ushiku, Y., Harada, T.: Neural 3d mesh renderer. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3907–3916 (2018)

19. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4), 139–1 (2023)

20. Kerr, J., Kim, C.M., Goldberg, K., Kanazawa, A., Tancik, M.: Lerf: Language embedded radiance fields. In: IEEE/CVF International Conference on Computer Vision. pp. 19729–19739 (2023)

21. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: International Conference on Learning Representations (2015)

22. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: IEEE/CVF International Conference on Computer Vision. pp. 4015–4026 (2023)

23. Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V.: Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics 36(4), 1–13 (2017)

24. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International Conference on Machine Learning. pp. 19730–19742 (2023)

25. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International Conference on Machine Learning. pp. 12888–12900 (2022)

26. Li, L., Zhu, S., Fu, H., Tan, P., Tai, C.L.: End-to-end learning local multi-view descriptors for 3d point clouds. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1919–1928 (2020)

27. Li, W., Zhao, Y., Qin, M., Liu, Y., Cai, Y., Gan, C., Pfister, H.: Langsplatv2: Highdimensional 3d language gaussian splatting with 450+ fps. Advances in Neural Information Processing Systems 38, 174306–174330 (2026)

28. Li, X., Lai, Z., Xu, L., Qu, Y., Cao, L., Zhang, S., Dai, B., Ji, R.: Director3d: Realworld camera trajectory and 3d scene generation from text. Advances in Neural Information Processing Systems 37, 75125–75151 (2024)

29. Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al.: Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22160–22169 (2024)

30. Lino, C., Christie, M.: Intuitive and eficient camera control with the toric space. ACM Transactions on Graphics 34(4), 1–12 (2015)

31. Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., Lee, Y.J.: Llava-next: Improved reasoning, ocr, and world knowledge (2024), https://llava-vl.github.io/blog/ 2024-01-30-llava-next/, accessed: 2026-06-26

32. Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al.: Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In: European Conference on Computer Vision. pp. 38–55 (2024)

33. Liu, X., Tai, Y.W., Tang, C.K.: Chatcam: Empowering camera control through conversational ai. Advances in Neural Information Processing Systems 37, 54483– 54506 (2024)

34. Liu, X., Zhang, T., Johnson-Roberson, M., Zhi, W.: Splatraj: Camera trajectory generation with semantic gaussian splatting. arXiv preprint arXiv:2410.06014 (2024)

35. Liu, Z., Wang, Y., Zheng, S., Pan, T., Liang, L., Fu, Y., Xue, X.: Reasongrounder: Lvlm-guided hierarchical feature splatting for open-vocabulary 3d visual grounding and reasoning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3718–3727 (2025)

36. Luiten, J., Kopanas, G., Leibe, B., Ramanan, D.: Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In: International Conference on 3D Vision. pp. 800–809 (2024)

37. Matsuki, H., Murai, R., Kelly, P.H., Davison, A.J.: Gaussian splatting slam. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18039– 18048 (2024)

38. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM 65(1), 99–106 (2021)

39. OpenAI: Introducing GPT-5.4 mini and nano (2026), https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/, accessed: 2026-06-26

40. Park, J.J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S.: Deepsdf: Learning continuous signed distance functions for shape representation. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 165–174 (2019)

41. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in Neural Information Processing Systems 30 (2017)

42. Qin, M., Li, W., Zhou, J., Wang, H., Pfister, H.: Langsplat: 3d language gaussian splatting. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20051–20060 (2024)

43. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748–8763 (2021)

44. Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y., Yan, F., et al.: Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159 (2024)

45. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4104–4113 (2016)

46. Schwarz, K., Sauer, A., Niemeyer, M., Liao, Y., Geiger, A.: Voxgraf: Fast 3d-aware image synthesis with sparse voxel grids. Advances in Neural Information Processing Systems 35, 33999–34011 (2022)

47. Shi, J.C., Wang, M., Duan, H.B., Guan, S.H.: Language embedded 3d gaussians for open-vocabulary scene understanding. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5333–5343 (2024)

48. Shi, J., Jiang, X., Guillemot, C.: Learning fused pixel and feature-based view reconstructions for light fields. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2555–2564 (2020)

49. Shoemake, K.: Animating rotation with quaternion curves. In: Proceedings of the 12th annual conference on Computer graphics and interactive techniques. pp. 245– 254 (1985)

50. Skartados, E., Yucel, M.K., Manganelli, B., Drosou, A., Saà-Garriga, A.: Finding waldo: Towards eficient exploration of nerf scene spaces. In: Proceedings of the 15th ACM Multimedia Systems Conference. pp. 155–165 (2024)

51. Sun, C., Sun, M., Chen, H.T.: Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5459–5469 (2022)

52. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., et al.: Siglip 2: Multilingual vision-language encoders with improved semantic understanding. Localization, and Dense Features 6 (2025)

53. Wang, X., Courant, R., Shi, J., Marchand, E., Christie, M.: Jaws: Just a wild shot for cinematic transfer in neural radiance fields. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16933–16942 (2023)

54. Wang, Z., Zhang, Z., Pang, T., Du, C., Zhao, H., Zhao, Z.: Orient anything: Learning robust object orientation estimation from rendering 3d models. International Conference on Machine Learning (2025)

55. Wen, M., Wu, S., Wang, K., Liang, D.: Intergsedit: Interactive 3d gaussian splatting editing with 3d geometry-consistent attention prior. In: IEEE/CVF International Conference on Computer Vision. pp. 26136–26145 (2025)

56. Wu, G., Yi, T., Fang, J., Xie, L., Zhang, X., Wei, W., Liu, W., Tian, Q., Wang, X.: 4d gaussian splatting for real-time dynamic scene rendering. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20310–20320 (2024)

57. Wu, J., Bian, J.W., Li, X., Wang, G., Reid, I., Torr, P., Prisacariu, V.A.: Gaussctrl: Multi-view consistent text-driven 3d gaussian splatting editing. In: European Conference on Computer Vision. pp. 55–71 (2024)

58. Wu, Y., Meng, J., Li, H., Wu, C., Shi, Y., Cheng, X., Zhao, C., Feng, H., Ding, E., Wang, J., et al.: Opengaussian: Towards point-level 3d gaussian-based open vocabulary understanding. Advances in Neural Information Processing Systems 37, 19114–19138 (2024)

59. Xie, C., Wang, B., Kong, F., Li, J., Liang, D., Zhang, G., Leng, D., Yin, Y.: Fgclip: Fine-grained visual and textual alignment. In: International Conference on Machine Learning (2025)

60. Yang, D., Wang, X., Gao, Y., Liu, S., Ren, B., Yue, Y., Yang, Y.: Opengs-fusion: Open-vocabulary dense mapping with hybrid 3d gaussian splatting for refined object-level understanding. In: IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 21135–21142 (2025)

61. Yang, Z., Li, L., Lin, K., Wang, J., Lin, C.C., Liu, Z., Wang, L.: The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421 (2023)

62. Ye, M., Danelljan, M., Yu, F., Ke, L.: Gaussian grouping: Segment and edit anything in 3d scenes. In: European Conference on Computer Vision. pp. 162–179 (2024)

63. Yi, T., Fang, J., Wang, J., Wu, G., Xie, L., Zhang, X., Liu, W., Tian, Q., Wang, X.: Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d difusion models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6796–6807 (2024)

64. Zhai, H., Zhang, X., Zhao, B., Li, H., He, Y., Cui, Z., Bao, H., Zhang, G.: Splatloc: 3d gaussian splatting-based visual localization for augmented reality. IEEE Transactions on Visualization and Computer Graphics (2025)

65. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: IEEE/CVF International Conference on Computer Vision. pp. 11975–11986 (2023)

66. Zhang, M., Wu, T., Tan, J., Liu, Z., Wetzstein, G., Lin, D.: Gendop: Auto-regressive camera trajectory generation as a director of photography. In: IEEE/CVF International Conference on Computer Vision. pp. 18229–18239 (2025)

67. Zhao, H., Wang, H., Zhao, X., Fei, H., Wang, H., Long, C., Zou, H.: Physsplat: Eficient physics simulation for 3d scenes via mllm-guided gaussian splatting. In: IEEE/CVF International Conference on Computer Vision. pp. 5242–5252 (2025)

68. Zhou, S., Fan, Z., Xu, D., Chang, H., Chari, P., Bharadwaj, T., You, S., Wang, Z., Kadambi, A.: Dreamscene360: Unconstrained text-to-3d scene generation with panoramic gaussian splatting. In: European Conference on Computer Vision. pp. 324–342 (2024)

# Supplementary Material CapFrame: Text-Instructed Viewpoint Grounding in 3D Gaussian Scenes via Geometric Pseudo Labels

In this supplementary material, we provide additional implementation details, runtime analysis, and extended experimental results. It is organized as follows:

– Sec. A: Additional implementation details, including details of IVS and SVS and gravity-direction estimation.

– Sec. B: Visualization of the pseudo labels produced in the Translate stage.

– Sec. C: Runtime analysis, including the runtime of the Retrieve, Translate, and Refine stages, as well as an acceleration strategy.

– Sec. D: Additional results on text-instructed viewpoint grounding, including extended comparisons with baselines, component analysis, and beyond source-view initialization.

– Sec. E: Failure cases and corresponding analysis.

Sec. F: Prompts used by the MLLM in the Retrieve and Translate stages.

– Sec. G: Details of the questionnaire used in the user study.

## A Implementation Details

## A.1 Details of IVS and SVS

Since there are no established baselines for text-instructed viewpoint grounding, we design two heuristic baselines inspired by the scene exploration strategies proposed in Finding Waldo [50]. That work studies viewpoint search in NeRF scenes and introduces two exploration strategies: (i) interpolation between promising viewpoints (Pose Interpolation-Based Search, PIBS) and (ii) random sampling of camera poses (Guided Random Search, GRS). We adapt these two strategies to the text-conditioned setting by replacing their original scoring functions with text-image alignment and by incorporating geometric constraints derived from the reconstructed scene.

Specifically, we introduce Interpolation-based Viewpoint Search (IVS) and Sampling-based Viewpoint Search (SVS). IVS is derived from the interpolation strategy of PIBS, while SVS follows the sampling principle of GRS. The key diference is that our baselines are required to generate viewpoints that satisfy a text instruction, which requires modifying how promising viewpoints are selected and how the sampling region is defined.

IVS (Interpolation-based Viewpoint Search). IVS is inspired by the pose interpolation strategy of PIBS [50], which generates new candidate viewpoints by interpolating between camera poses that already achieve high task scores. In

Text Instruction: A small blue elephant is centered in the frame, facing the camera directly.   
The camera views it from a downward angle of about 60 degrees.

IVS

![](images/fe3a776bd1a206b7948d0540d8e15b72b74fce0e737caf03a49855450a960307.jpg)  
(a) Cameras with top-L FG-CLIP scores

![](images/019f8778ad0ea1e1b4bf4ce058502d1dd8e090e927e246d15a51a225367aeec7.jpg)  
(b) Interpolated Cameras

![](images/6c6419065e9b7e3935dd6ec68131d0d545997e9bb0685ac7c1d94d32b62e1c52.jpg)

![](images/831a72d64b2f4c8d66035da822bad42b93f057a2cd1c8923f3dad05dc3be5d5c.jpg)

![](images/3b3584edfdbfa2327ce9b0a418a293c4bbceeb9380ee5d9b650a5f654c58a289.jpg)

![](images/e05ff442a9723d903f76bf2bb0d6d086ce5cf6f64fba61368f60256454ded82f.jpg)  
(c) Top-4 Interpolated Images  
SVS

![](images/e65f20d66fc90e5c348e3e3dd6be30cda78f48417f9eb0a50e322062b7f467b7.jpg)  
(d) Sampled Cameras

![](images/e8c1a9057b710ce2651243b5fb26b598b68d9e3e4dd7b8ad5bfe44ad2da35000.jpg)

![](images/1b76a447d7167b53dd62e3fd1b85cb633743c54ffc78d8b144f6d5f844789448.jpg)

![](images/9eaab9ed7d3afcf5cacb3f0b8a215f5be9f46327edc3767f5a49fafb67634739.jpg)

![](images/1498fcaf598e0388833ce97651d649ea59e61e0cab7c4072b2cb80ef74ace40c.jpg)  
(e) Top-4 Sampled Images

Text Instruction: A red toy radio with blue buttons showing its front is positioned in the center, while on the right side oftheframe is a wooden alphabet block toy.

IVS

![](images/a16b2bcae5a0363f8be42cbe24fb93086174c8c813bb2acaffc99aeaf4cb689f.jpg)

![](images/688467ecfa464b77ef12fce328aa3dcf819da9d6228444045bdfcbae2aca59bf.jpg)  
(a) Cameras with top-L FG-CLIP scores  
(b) Interpolated Cameras

![](images/19a7424eeab3b30cbe6b856b2e6113fd533d44bd4c31286a7264246a3aba62f5.jpg)

![](images/776ae24300d882efb0d884a406e286ff1a8cfd84c77caa43dca1ef76bbdb491a.jpg)

![](images/ff109387c6dc02a92edc4fc57a717420e44f544fd72229a3f5d26729406f7bef.jpg)

![](images/92e40caa3575d17e37592114020d78b5b0084683c2b0a04e6640cd8f73014cdd.jpg)  
(c) Top-4 Interpolated Images  
SVS

![](images/94ce6163f7fecb279191dd14f09c297944d8afe4cc338a9d02d8457ef10b53c9.jpg)  
(d) Sampled Cameras

![](images/e4b695071cea96e3f7f230c3e7ddca85b2d2fb1668e05423bd2325d731380113.jpg)

![](images/f17478e10281e1648babcd2f6d07f74c31c7a7ffddc623e80dcfff81b70dbc69.jpg)

![](images/5f9c69f5e6355c17a1fa5cc4e80d0265c167cf3f94c09715fa85e5209333a92d.jpg)

![](images/ab1c00652cc27d45d18e4ee27a321fe5b085cf51e42b1c33fc75fe98190e1ec0.jpg)  
(e) Top-4 Sampled Images

Fig. 7: Visualization of viewpoint generation in IVS and SVS. Two examples are shown, with key subjects highlighted by red boxes. For IVS, (a) shows the top-L training cameras retrieved by FG-CLIP [59], (b) shows the candidate cameras generated by pose interpolation, and (c) shows the top-4 rendered views ranked by FG-CLIP. For SVS, (d) shows the candidate cameras sampled around the translated subject locations, and (e) shows the top-4 rendered views ranked by FG-CLIP.

the original method, promising viewpoints are selected based on task-specific criteria such as saliency or image quality. In our text-instructed setting, we instead identify promising viewpoints according to their alignment with the text.

Concretely, similar to the Retrieve stage described in Sec. 5.1 of the main paper, we first select the top-L training images that are most consistent with the text instruction using FG-CLIP [59], where $L = \operatorname* { m i n } ( 1 3 , N )$ and N denotes the number of training images. Since viewpoints close to these highly aligned images are likely to satisfy the instruction, we interpolate between their camera poses to generate additional candidate viewpoints. Concretely, all possible pose pairs are formed among the top-L views $( i . e . , \frac { L ( L - 1 ) } { 2 }$ pairs). For each pair, three intermediate poses are generated via interpolation, resulting in $\frac { 3 L ( L - 1 ) } { 2 }$ candidate viewpoints. Rotations are interpolated in quaternion space using spherical linear interpolation (SLERP) [49], while translations are linearly interpolated. Rendering these poses produces a candidate image set.

SVS (Sampling-based Viewpoint Search). SVS is inspired by the random sampling strategy of GRS [50], which explores the camera pose space by sampling viewpoints within a bounded spatial region. Unlike GRS, where sampling is guided by the distribution of training cameras, the relevant region in our task depends on the subjects mentioned in the text instruction.

To address this, we define the sampling region using the 3D Gaussian representation obtained in the Translate stage (Sec. 5.2 of the main paper). We first identify the Gaussians associated with key subjects mentioned in the instruction and compute their mean center. A bounding sphere enclosing these Gaussians is then constructed. Camera viewpoints are generated by sampling directions on the sphere and placing the camera along the corresponding ray toward the sphere center, assuming the camera always faces the center. The camera distance is determined from a reference distance d that ensures all subject Gaussians tightly lie within the view, and the final camera centers are placed at random distances within $\left[ \alpha d , \beta d \right] ( i . e . , \alpha = 1 . 1 , \beta = 1 . 6 )$ along each sampled ray. To avoid viewpoints from below the scene, the elevation angle on the sphere is restricted to $[ - 7 0 ^ { \circ } , 9 0 ^ { \circ } ]$ , and directions outside this range are not sampled.

For fair comparison, the number of sampled viewpoints for SVS is set to $\frac { 3 L ( L - 1 ) } { 2 }$ , the same as in IVS. For both methods, the final camera pose is selected from the generated candidates as the one achieving the highest text-image alignment score measured by a VLM.

Visualizations of camera poses generated by IVS and SVS are shown in Fig. 7. For reference, we present the rendered views with the top-4 alignment scores computed by FG-CLIP [59]. IVS performs well when the training set already contains viewpoints aligned with the instruction, as interpolation can further refine such poses. In contrast, SVS is more efective when the instruction mainly specifies coarse spatial layouts $( e . g . , c e n t e r , l e f t , r i g h t )$ without requiring specific camera orientations or detailed compositions. Additional comparisons between IVS, SVS and CapFrame are provided in Sec. D.

## A.2 Estimating the Gravity Direction

Our orientation loss includes a gravity term that encourages the camera up direction to align with the world up direction (Eq. (19) in the main paper). We estimate this direction from the training images using COLMAP [45]. Specifically, we apply COLMAP’s model\_orientation\_aligner, which detects dominant vanishing directions under the Manhattan world assumption and aligns the reconstruction so that the +Y axis corresponds to gravity. We then use this aligned −Y axis as the world up direction in the gravity loss.

## B Visualization of Pseudo Labels

Figure 8 illustrates the intermediate outputs produced in the Translate stage. This stage converts the input text instruction into geometric pseudo labels that guide the subsequent camera pose optimization.

Specifically, Fig. 8 visualizes four components for each example: (a) the image rendered by CapFrame, (b) the 3D Gaussian subsets associated with the parsed subjects, (c) the layout pseudo labels represented as 2D bounding boxes in normalized image coordinates, and (d) the orientation pseudo labels describing the desired viewing direction.

For each subject mentioned in the instruction, the MLLM first extracts key entities and predicts layout and orientation cues. Grounded SAM [22, 32, 44] is then applied to the retrieved views to obtain segmentation masks, whose pixels are back-projected into 3D using the rendered depth of the 3D Gaussian scene. The KNN search between these points and Gaussian means identifies the subset of Gaussians corresponding to each subject.

Layout pseudo labels are inferred by the MLLM and represented as bounding boxes in normalized image coordinates, defining the target regions where subjects should appear in the rendered frame. During optimization, the associated Gaussians are projected onto the image plane to enforce these spatial constraints. Orientation pseudo labels are generated only for asymmetric subjects when the instruction specifies viewpoint cues (e.g., front view or side view). We rely on the MLLM to identify asymmetric subjects by reasoning about whether each object possesses a canonical facing direction. In such cases, the MLLM predicts orientation ofsets that are converted into geometric targets for camera pose optimization. Otherwise, only layout pseudo labels are used.

The examples in Fig. 8 show cases in which both layout and orientation pseudo labels are applied, illustrating how compositional text instructions are translated into geometric constraints. The prompts used for the MLLM are provided in Sec. F.

## C Computational Time

The computational cost of CapFrame mainly comes from the Retrieve, Translate, and Refine stages. In our experiments, the number of top-K views is fixed

Text Instruction: A small blue elephant is centered in the frame, facing the camera directly.   
The camera views itfrom a downward angle ofabout 60 degrees.

![](images/b3afc6a28a75d139e2bec5341186356e5f824c94df8b0d515be7c9e8a53078dc.jpg)  
(a) Output Image

![](images/4398f16ed39981f4dd3c970b761d056ffd216402b215ace41bf14db52fa6309e.jpg)  
(b) Subject Points

![](images/33af5017394824bea41264350f6c4c390d0b61ee6ad5f036c2eee49b5ce61a7b.jpg)  
(c) Layout Pseudo Label

![](images/cdc3a5f649a3d46461cf59d885e5e5761ffa52c5c53019dc791611494cc7c3eb.jpg)  
(d) Orientation Pseudo Label

Text Instruction: Aframed city painting is centered in the image,facing the camera. A black coat hangs on the left, and a white floor lamp stands on the right.

![](images/c01ee24899d7a5185e3110b6897239fad39633f5b7a4725ab48fecc4e7cf1062.jpg)  
(a) Output Image

![](images/5ba82594550ab36cbc7cffe214380e503dcefe600c0ff64fdb1b6cb9697cdd3d.jpg)  
(b) Subject Points

![](images/b9275623e68f34174cb27a2580a4b0147ab70e325b23fa58959480f47e4fbeb6.jpg)  
(c) Layout Pseudo Label

![](images/55e83979492bd227538207151de1d5d010165139051a012101c70b302309d278.jpg)  
(d) Orientation Pseudo Label

Text Instruction: Thefront view ofa brown teddy bear, and it is positioned on the right side oftheframe. The camera looks slightly down at the bear.

![](images/6bac8abe3663b261dfe06e79a4d47854492e3b1c2dcdd22522c9268fb97b9876.jpg)  
(a) Output Image

![](images/510f6e767e5891f88d9e6db897da8b7f2c2d0b6a2aa6f05f0f2e9c95a7e25daf.jpg)  
(b) Subject Points  
(c) Layout Pseudo Label

![](images/68c32d1209bbac43a0f7e4062c6a9a597780a3293199ddf9f43a9845e61693ed.jpg)  
(d) Orientation Pseudo Label

Text Instruction: A red toy radio with blue buttons showing its front is positioned in the center, while on the right side of the frame is a wooden alphabet block toy.

![](images/a57ab1e705fe163b78de3e3e8b1e05f49b023f24411ec751e957f743767daf1e.jpg)  
(a) Output Image

![](images/4da2477a04c9c750fc4e3a5f7ed65bc0cc5d6d33a83e80d05c9595662bae064b.jpg)  
(b) Subject Points

![](images/bd116c5929662212bf1e7fa1b865cd788f97949e2e3d18fc8dcea49fabd261e8.jpg)  
(c) Layout Pseudo Label

![](images/f1aa2380c834a9ed86b9edb42e293f0579f4778a3219af96992699a5b57fdb0d.jpg)  
(d) Orientation Pseudo Label

Text Instruction: On the left stands a white floor lamp. In the center, a small colorful painting hangs on the wall above a computer with two screens, as the camera faces the computer.

![](images/78c01f71955917c7ae032dc2a0fa4310f121abd8c71b5ea4db2a752cc8acb7cd.jpg)  
(a) Output Image

![](images/ec5390ddb120e356cf3983bf27e6c1022044470428d7148be01ad735a97b6054.jpg)  
(b) Subject Points

![](images/084cc88404c31f248797f50aa6d2e58a459ee224676699111ba3c5b2a187ca10.jpg)  
(c) Layout Pseudo Label

![](images/707646255678ee30bd4826b2324d0c51085bdc7c341d9b980564ab958530dd42.jpg)  
(d) Orientation Pseudo Label

Fig. 8: Pseudo labels produced in the Translate stage. For each instruction, (a) shows the final image rendered by CapFrame, (b) shows the 3D points / Gaussian subsets associated with the parsed subjects, (c) shows the layout pseudo labels as target 2D bounding boxes, and (d) shows the orientation pseudo label as a red ray originating from the subject mean center. Orientation pseudo labels are generated only for asymmetric subjects when the instruction specifies a viewing direction.

to K = 2, so the cost of the Retrieve stage is dominated by similarity search over the training images.

The runtime of the Translate stage mainly scales with the number of parsed subjects, since subject extraction and Gaussian association are performed for each subject. The runtime of the Refine stage is largely determined by the image resolution and the number of optimization iterations.

We report the approximate runtime per scene in Tab. 4. Since early stopping is applied when the loss variation falls below a predefined threshold, optimization does not always reach the maximum of 1500 iterations. Moreover, because the initial viewpoint is randomly selected from the top-K retrieved views, the number of iterations required for convergence varies across runs. Therefore, the reported runtimes should be interpreted as indicative estimates. In practice, the entire pipeline typically finishes within a few minutes.

Acceleration Strategy. Our experiments span 38 scenes from 5 datasets, covering indoor, outdoor, tabletop and room-scale settings, with 12–411 input views and scene scales ranging from 12.3 to 311.8 in 3DGS coordinate units. Across these scenes, the average wall-clock time per query is 102.5 s, comprising Retrieve (33.7 s), Translate (25.3 s), and Refine (43.5 s). Since the Refine stage optimizes the camera pose using the coarse layout and spatial orientation of key subjects rather than fine pixel-level appearance, it is insensitive to rendering resolution and can be performed at reduced resolution during optimization. We therefore optimize the camera pose at low resolution and render the final frame at full resolution. This reduces the per-query runtime to 55.8 s while incurring only minor deviations from the full-resolution pose (∆R = 2.63<sup>◦</sup>, ∆t = 0.17).

## D Additional Experimental Results

## D.1 More Qualitative Results

More Results on Text-Instructed Viewpoint Grounding. Additional qual itative results are presented in Fig. 9. CapFrame supports free-form text instructions and identifies camera viewpoints that satisfy both the specified composition and viewing angle by grounding the relevant subjects in the 3D Gaussian scene.

More Comparisons with IVS and SVS. Additional visual comparisons with IVS and SVS are shown in Fig. 10. CapFrame consistently produces viewpoints aligned with the text instructions and avoids unnatural camera rotations, as illustrated in the red toy radio. It also provides finer control over object orientation and camera viewing direction compared with the baselines, as demonstrated in the blue elephant.

For IVS, when the training set already contains viewpoints consistent with the instruction, interpolation can refine such poses to produce aligned viewpoints, as seen in the gray curtain and white cabinet. However, when the desired viewpoint lies outside the coverage of the training views, interpolation fails to

Table 4: Per-scene runtime of CapFrame. The first three columns report the dataset, scene and image resolution. Stage 1+2 denotes the combined runtime of the Retrieve and Translate stages, Stage 3 denotes the optimization time of the Refine stage, and Total Time reports the overall runtime. All values are in seconds.
<table><tr><td rowspan=1 colspan=3>Dataset</td><td rowspan=1 colspan=1>Scene</td><td rowspan=1 colspan=1>|Image Resolution</td><td rowspan=1 colspan=1>|Stage 1+2|</td><td rowspan=1 colspan=1>Stage 3|</td><td rowspan=1 colspan=1>Total Time</td></tr><tr><td rowspan=5 colspan=3>Mip-NeRF</td><td rowspan=1 colspan=1>Garden</td><td rowspan=1 colspan=1>5187× 3361</td><td rowspan=1 colspan=1>129.077</td><td rowspan=1 colspan=1>54.904</td><td rowspan=1 colspan=1>183.981</td></tr><tr><td rowspan=1 colspan=1>Kitchen</td><td rowspan=1 colspan=1>3115 × 2078</td><td rowspan=1 colspan=1>89.373</td><td rowspan=1 colspan=1>252.599</td><td rowspan=1 colspan=1>341.973</td></tr><tr><td rowspan=1 colspan=1>Room</td><td rowspan=1 colspan=1>3114 × 2075</td><td rowspan=1 colspan=1>89.621</td><td rowspan=1 colspan=1>46.739</td><td rowspan=1 colspan=1>136.360</td></tr><tr><td rowspan=2 colspan=1>CounterBonsai</td><td rowspan=1 colspan=1>3115 × 2076</td><td rowspan=1 colspan=1>70.162</td><td rowspan=1 colspan=1>43.230</td><td rowspan=1 colspan=1>113.392</td></tr><tr><td rowspan=1 colspan=1>3118 × 2078</td><td rowspan=1 colspan=1>95.816</td><td rowspan=1 colspan=1>26.613</td><td rowspan=1 colspan=1>122.429</td></tr><tr><td rowspan=13 colspan=3>Deep Blending</td><td rowspan=1 colspan=1>Playroom</td><td rowspan=1 colspan=1>1264 × 832</td><td rowspan=1 colspan=1>54.575</td><td rowspan=1 colspan=1>7.883</td><td rowspan=1 colspan=1>62.458</td></tr><tr><td rowspan=12 colspan=1>DrjohnsonTreeLibraryHugoCreepyatticBedroomAquariumStreetYellowhousePoncheMuseum1Museum2</td><td rowspan=1 colspan=1>1332 × 876</td><td rowspan=1 colspan=1>61.229</td><td rowspan=1 colspan=1>22.329</td><td rowspan=1 colspan=1>83.558</td></tr><tr><td rowspan=1 colspan=1>2648 × 1739</td><td rowspan=1 colspan=1>28.461</td><td rowspan=1 colspan=1>54.582</td><td rowspan=1 colspan=1>83.043</td></tr><tr><td rowspan=1 colspan=1>3569 × 1987</td><td rowspan=1 colspan=1>87.277</td><td rowspan=1 colspan=1>49.234</td><td rowspan=1 colspan=1>136.512</td></tr><tr><td rowspan=1 colspan=1>3217 × 2135</td><td rowspan=1 colspan=1>50.428</td><td rowspan=1 colspan=1>123.592</td><td rowspan=1 colspan=1>174.020</td></tr><tr><td rowspan=1 colspan=1>1231 × 819</td><td rowspan=1 colspan=1>74.183</td><td rowspan=1 colspan=1>7.393</td><td rowspan=1 colspan=1>81.576</td></tr><tr><td rowspan=1 colspan=1>1298 × 840</td><td rowspan=1 colspan=1>53.486</td><td rowspan=1 colspan=1>36.311</td><td rowspan=1 colspan=1>89.798</td></tr><tr><td rowspan=1 colspan=1>2642 ×1962</td><td rowspan=1 colspan=1>35.982</td><td rowspan=1 colspan=1>36.005</td><td rowspan=1 colspan=1>71.987</td></tr><tr><td rowspan=1 colspan=1>2606 × 1734</td><td rowspan=1 colspan=1>31.709</td><td rowspan=1 colspan=1>53.470</td><td rowspan=1 colspan=1>85.179</td></tr><tr><td rowspan=1 colspan=1>4185 × 2505</td><td rowspan=1 colspan=1>31.979</td><td rowspan=1 colspan=1>139.556</td><td rowspan=1 colspan=1>171.536</td></tr><tr><td rowspan=1 colspan=1>2503 × 1466</td><td rowspan=1 colspan=1>43.735</td><td rowspan=1 colspan=1>26.553</td><td rowspan=1 colspan=1>70.288</td></tr><tr><td rowspan=1 colspan=1>2338 × 1537</td><td rowspan=1 colspan=1>31.596</td><td rowspan=1 colspan=1>17.490</td><td rowspan=1 colspan=1>49.086</td></tr><tr><td rowspan=1 colspan=1>2353 × 1538</td><td rowspan=1 colspan=1>31.295</td><td rowspan=1 colspan=1>10.884</td><td rowspan=1 colspan=1>42.179</td></tr><tr><td rowspan=4 colspan=3>Tanks</td><td rowspan=4 colspan=1>FamilyIgnatiusMuseumFrancis</td><td rowspan=1 colspan=1>960 × 540</td><td rowspan=1 colspan=1>46.690</td><td rowspan=1 colspan=1>13.159</td><td rowspan=1 colspan=1>59.848</td></tr><tr><td rowspan=1 colspan=1>960 × 540</td><td rowspan=1 colspan=1>43.493</td><td rowspan=1 colspan=1>8.403</td><td rowspan=1 colspan=1>51.896</td></tr><tr><td rowspan=1 colspan=1>960 × 540</td><td rowspan=1 colspan=1>37.886</td><td rowspan=1 colspan=1>5.238</td><td rowspan=1 colspan=1>43.124</td></tr><tr><td rowspan=1 colspan=1>960 × 540</td><td rowspan=1 colspan=1>34.099</td><td rowspan=1 colspan=1>50.676</td><td rowspan=1 colspan=1>84.775</td></tr><tr><td rowspan=4 colspan=3>LERF-OVS</td><td rowspan=4 colspan=1>TeatimeFigurinesWaldo KitchenRamen</td><td rowspan=4 colspan=1>988 × 730986 × 728985 × 725988 × 731</td><td rowspan=1 colspan=1>46.465</td><td rowspan=1 colspan=1>63.109</td><td rowspan=1 colspan=1>109.574</td></tr><tr><td rowspan=1 colspan=1>64.398</td><td rowspan=1 colspan=1>45.905</td><td rowspan=1 colspan=1>110.303</td></tr><tr><td rowspan=1 colspan=1>39.464</td><td rowspan=1 colspan=1>38.478</td><td rowspan=1 colspan=1>77.942</td></tr><tr><td rowspan=1 colspan=1>36.334</td><td rowspan=1 colspan=1>10.825</td><td rowspan=1 colspan=1>47.160</td></tr><tr><td rowspan=5 colspan=3></td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>976 × 541</td><td rowspan=1 colspan=1>60.229</td><td rowspan=1 colspan=1>39.321</td><td rowspan=1 colspan=1>99.550</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>953 × 536</td><td rowspan=1 colspan=1>56.859</td><td rowspan=1 colspan=1>13.969</td><td rowspan=1 colspan=1>70.828</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>954×536</td><td rowspan=1 colspan=1>63.523</td><td rowspan=1 colspan=1>37.702</td><td rowspan=1 colspan=1>101.225</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>975 × 541</td><td rowspan=1 colspan=1>58.062</td><td rowspan=1 colspan=1>21.856</td><td rowspan=1 colspan=1>79.918</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>955 × 536</td><td rowspan=1 colspan=1>61.067</td><td rowspan=1 colspan=1>10.845</td><td rowspan=1 colspan=1>71.912</td></tr><tr><td rowspan=4 colspan=3>DL3DV-10K</td><td rowspan=2 colspan=1>67</td><td rowspan=1 colspan=1>958× 538</td><td rowspan=1 colspan=1>50.637</td><td rowspan=1 colspan=1>18.851</td><td rowspan=1 colspan=1>69.488</td></tr><tr><td rowspan=1 colspan=1>963×539</td><td rowspan=1 colspan=1>60.711</td><td rowspan=1 colspan=1>58.378</td><td rowspan=1 colspan=1>119.089</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>3827 × 2154</td><td rowspan=1 colspan=1>199.226</td><td rowspan=1 colspan=1>178.983</td><td rowspan=1 colspan=1>378.209</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>976× 542</td><td rowspan=1 colspan=1>58.173</td><td rowspan=1 colspan=1>34.229</td><td rowspan=1 colspan=1>92.402</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>964× 540</td><td rowspan=1 colspan=1>62.860</td><td rowspan=1 colspan=1>26.259</td><td rowspan=1 colspan=1>89.118</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>960 ×539</td><td rowspan=1 colspan=1>66.426</td><td rowspan=1 colspan=1>11.164</td><td rowspan=1 colspan=1>77.590</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>958×538</td><td rowspan=1 colspan=1>71.609</td><td rowspan=1 colspan=1>33.517</td><td rowspan=1 colspan=1>105.126</td></tr></table>

![](images/355662a825582aeb40ab30653d24619dbb4e5b61e8226b3d0295ac9051a8a6cf.jpg)

## Text Instruction:

A framed city painting is centered in the image, facing the camera. A black coat hangs on the left, and a white floor lamp stands on the right.

Output Image:  
![](images/2139b670efcbd1c2364637104e6de3c0415fdf384d444cfa4ace28ec708e75bb.jpg)

Iteration: Camera Pose:  
![](images/32078b4c576a50f26e6cc30f1c4b934bf14230c7166f2fe3f1b4dbce43099f6b.jpg)

![](images/e440ba7953c404d25f01c9e5aebb39482ac4a01a073050e3be59f6c6c00bd396.jpg)

A decorative green fountain is positioned at the center of the image, viewed from a high angle.

![](images/512c223a708890493512e693bd16fb5172bbad61cd5f407fb5d5dfae6837e167.jpg)

![](images/91b6afb4bc65ccf9145cd7e9f0cc16f6f1e95effd01d67ad7ed2ef203abd8596.jpg)

![](images/0950fdafcff526b9e4373ce7479b10a9e44a39e383b5e2ee2025108c657ed5fa.jpg)

On the far left stands a white floor lamp. In the center, a small colorful painting hangs on the wall above a computer with two screens.

![](images/4ea6defee07c77df112ca49b23b1c7069b94818f9c7dd0d8b45cffb6c31bee17.jpg)

![](images/9738cf9ea7f70ab467554a7a218438101eaabe94675668d16567510b7ddf4276.jpg)

![](images/64977a0ae5d0c5404dfa82e1a4cdb82b9a43722ada14652572e5bc994f8e8e70.jpg)

The large leafy tree stands on the right side of the image, with the ornate black iron gate positioned to its lower left.

![](images/32b1cad693ac5ea0cea0c325e22062c20df7c09377d7c2ad2c8c4522fd95db8a.jpg)

![](images/dee7b47b0911bf8a2879bc3afe5d5c7228141f86c9307aee57c4dfd9e590c44e.jpg)

![](images/720aed347756a2e99b7145d313954f965efc6a8465f3854cfdcb0f27d955455c.jpg)

A white plate with two cookies and a broken piece sits on the right side of the image, while a glass of tea occupies the left side, with torn tea packets scattered between them.

![](images/f82ee1add8e697bddc4fbec26b1857dda48507236edada9137e37ed656fd19fb.jpg)

![](images/86f48acdd4343d4316ac42443ffd88671d80493b59df000e61d9bc11cbb2f8fe.jpg)

![](images/90c0c3bffd452de08cbbb6e1102da27ab7487616bb9761573f23dda19e49bcda.jpg)

In this close-up, a wooden photo frame with a landscape painting sits on the left, while yellowflowers in a glass vase stands on the right, with the camera a little tilted clockwise.

![](images/a71e875a09be946a3fc7e8cbf2db19eff854fd14f478f1421a594b2010569481.jpg)

![](images/7b2de3761a14cbf1eab5c3bdb4539e94fd5e4ba46a01c64ddfb237a0ca4c66a0.jpg)

![](images/dfe1338bbe1306d7e2038c7284a5b3fedc7bdd61700ca132b81e12b52aa4eb70.jpg)  
Fig. 9: Additional text-instructed viewpoint grounding results. Each row shows a text instruction, the final rendered image produced by CapFrame, intermediate rendered views during optimization, and the corresponding camera pose trajectory. CapFrame can satisfy both composition and viewing angle constraints from free-form compositional instructions.

## Text Instruction:

A small blue elephant is centered in the frame, facing the camera directly. The camera views itfrom a downward angle of about 60 degrees.

A red toy radio with blue buttons showing its front is positioned in the center, while on the right side of the frame is a wooden alphabet block toy.

A largeframed abstract artwork is positioned at the center ofthe image.

A gray curtain appears slightly on the left side of the frame, while a white cabinet spans the right side.

CapFrame  
IVS  
SVS  
![](images/88d0da8c8dfc561bf2e5ef6477b92ca5ab22cb9c493d55cdd6f673b6b948b714.jpg)  
Fig. 10: Additional comparisons with baselines. Each row corresponds to one text instruction, and the columns show the outputs of CapFrame, IVS and SVS. CapFrame more reliably satisfies joint layout and orientation constraints, especially when the desired viewpoint is weakly covered by the training cameras.

locate a consistent solution. For SVS, when instructions specify simple spatial attributes such as center, sampling-based exploration can identify reasonable viewpoints, as demonstrated in the framed abstract artwork. However, when the instruction involves more complex layouts and viewing angles, both IVS and SVS struggle to produce suitable viewpoints. This limitation is compounded by the limited perception capability of FG-CLIP [59], which is used to score candidate views.

## D.2 More Component Analyses

We present additional component analyses of the Question-Evaluation (QE) strategy in the Retrieve stage and the layout and orientation losses in the Refine stage, as illustrated in Fig. 11 and Fig. 12.

Question-Evaluation Strategy. The QE strategy decomposes the instruction into a set of evaluation questions using an MLLM. These questions assess key photographic aspects such as subject visibility, frame-based layout, camerabased orientation, and overall text-image alignment. For each candidate view, the same MLLM evaluates the image by answering these questions and assign-

Text Instruction: A bonsai pine tree with dark green foliage sits inside a large brown ceramic vase, and both are positioned on the right side of the image.

FG-CLIP FG-CLIP + MLLM FG-CLIP + MLLM + QE  
FG-CLIP  
FG-CLIP + MLLM FG-CLIP + MLLM + QE  
![](images/97b5c0413e2822fcf6fc2065c530368e0a85544c45f641a58d9f4bdf493a2bbf.jpg)  
Text Instruction: The image shows the front view of a refrigerator positioned on the right side oftheframe, with a white kitchen cabinet located in the lower left.

![](images/1a008ea6daa16ccc808d7af83fe9592a597a0226e4ba3ceda43e263202d81d0e.jpg)  
Fig. 11: Additional ablation of initialization. For each instruction, we show the top-2 retrieved views obtained by FG-CLIP, FG-CLIP+MLLM, and FG-CLIP+MLLM+QE, together with the final optimized results. QE better disambiguates visually similar objects and photographic attributes, leading to stronger initial views for subsequent optimization.

ing scores. The final score of each view is obtained by aggregating the MLLMprovided scores across all questions. The prompts used for question generation and evaluation are provided in Sec. F.

Figure 11 highlights the advantage of QE. In cluttered scenes containing visually similar objects, the bonsai pine tree in the background can be easily confused with a similar potted plant in the foreground. Without QE, the background subject is dificult to distinguish, causing the bonsai pine tree to be overlooked and leading to unsuccessful optimization. Moreover, QE guides the MLLM to attend to photographic attributes in the instruction. As shown in the refrigerator example, QE enables the MLLM to retrieve training images that better satisfy textual cues such as front view and right side, providing a stronger initialization for subsequent optimization.

Layout and Orientation Losses. Figure 12 demonstrates the role of the layout and orientation losses. Without the orientation loss, the white sheep fails to present its right profile toward the camera. Removing the layout loss instead causes the white sheep to shift away from the intended central position. A similar efect is observed in the chairs example: without the orientation loss, the camera fails to achieve the required slight downward viewing angle, while removing the layout loss makes it dificult to control the scale of the chairs in the frame and their distance from the camera.

## Text Instruction:

A plush white sheep is positioned at the center of the image, showing its right profile to the camera.

![](images/7b8319ac8dbeb9eba76c1359d98bb914daf7fe9dd41eff280a518fb2ff0307a9.jpg)

![](images/d76368a18792b4c91984f86ed30b3b8e936f8f630ae107c6ae35ebe2ac9eb796.jpg)

![](images/be40e78f59a8ebe90444abe28960edafe055adf803dad9d873875e1ced489cab.jpg)

A close-up shows red toy chair is on the right, a green toy chair is on the left, both in balanced proportions. The camera is angled slightly downward toward the chairs.

![](images/f75545b95dbcd0227e541fa66ce82a7ea663b865c25874d7ecbbc9f3012bcb73.jpg)  
(a) $\mathcal { L } _ { l a y o u t }$

![](images/bc26172de6e8cc78bae230c412d46e914b67e1dd27174f8958651ea25811dcf2.jpg)  
(b) $\mathcal { L } _ { o r i e n t }$

![](images/d4393988a2beb6f96a7f770f18a9c22116384bb1569660fa5e1c78a5e1b2738c.jpg)  
(c) Llayout + Lorient  
Fig. 12: Additional ablation of loss. For each instruction, (a) uses only $\mathcal { L } _ { \mathrm { l a y o u t } }$ , (b) uses only $\mathcal { L } _ { \mathrm { o r i e n t } }$ , and (c) uses both. Using both losses is necessary to jointly contro subject placement, scale and viewing direction.

## D.3 Beyond Source-View Initialization

CapFrame does not assume that the final camera lies close to a source view. It only requires an initialization in which the target subject is visible, so that the subject Gaussians and their diferentiable masks can be obtained. Then, the optimization proceeds in continuous SE(3). As a result, CapFrame can discover novel viewpoints absent from the input image set. In Fig. 13, we deliberately initialize from a source view distant from the desired camera, yet CapFrame still recovers a bird’s-eye-like view unseen among the source images, with the refined pose lying far from the initial one $( \varDelta R = 5 7 . 7 6 ^ { \circ }$ , ∆t = 4.285 in 3DGS coordinate units). Such behavior illustrates CapFrame’s ability to discover viewpoints beyond the source views.

## Text Instruction:

A red toy chair is at the center, a green toy chair is on the left of the frame. The camera looks straight down at the chairs.

![](images/75377fedfc56b51421400a3722c4cf6311711c85ca9e2890715c5eb3cc44b67b.jpg)

![](images/ececd5f46ac6c67a11c5b2f5154cc1a14299bdee00ac67d0ce0e6fcd04e2eb88.jpg)

![](images/f6c13ebc91309320a39b7b9e9a7166936621eaa8bc2e6226c4242b45c89eb64b.jpg)  
Fig. 13: Refined view deviates from initial view. The pose trajectory (right) illustrates the substantial change from initial view to the refined view.

## E Failure Cases

CapFrame mainly exhibits two failure modes.

Incorrect Subject Detection. The first failure mode originates from the subject localization step in the Translate stage. Grounded SAM [44] is used to segment subjects in the retrieved views before associating them with 3D Gaussians. Due to the limitations of open-vocabulary perception, incorrect detections may occur. As shown in Fig. 14, the detector fails to correctly identify the purple cloud-shaped cushion. Consequently, the Gaussian subset associated with the intended subject is incorrect, leading to inaccurate layout supervision. Although the red beanbag gradually moves toward the bottom region during optimization, the rendered result cannot satisfy the text instruction.

Redundant Subject Parsing. The second failure mode arises from the subject parsing process in the Translate stage. When decomposing the instruction, the MLLM [1] may occasionally identify visually salient but textually irrelevant objects (e.g., stone pedestal and building) as additional subjects. For such objects, the instruction provides no layout or orientation cues, yet layout pseudo labels are still generated during the translation process. These additional layout constraints introduce noisy supervision in the layout loss, which interferes with camera pose optimization and may lead to suboptimal viewpoints.

## F Prompts for MLLM

The MLLM is used in both the Retrieve and Translate stages.

Text Instruction: The sculpture is centered in theframe, showing its left view, and shotfrom a low angle.

Text Instruction: A large red beanbag is positioned at the bottom of the frame, while a purple cloud-shaped cushion appears directly above it in the frame.

![](images/32dfba117abe7c6be8da9cb37e0b2a79fd511410bd97c46a9d72f2417e2e4f8c.jpg)

![](images/5a9e937b611b449b3b4af503a5f92c0d586c0bbd2cead58463143299f19aa883.jpg)  
Fig. 14: Failure cases. For each example, the left shows the final output, the middle shows intermediate rendered views during optimization, and the right highlights the source of failure. The top row shows incorrect subject detection, where Grounded SAM [44] localizes the wrong object. The bottom row shows redundant subject parsing, where the MLLM [1] introduces irrelevant objects that add noisy layout constraints.

In the Retrieve stage, we employ the Question-Evaluation (QE) strategy to assess candidate views from multiple photographic perspectives. As shown in Fig. 15, the MLLM first generates a set of evaluation questions from the text instruction, focusing on four aspects: subject identification, frame-based layout, camera-based orientation, and overall text-image alignment. Each candidate image is then evaluated by the same MLLM, which provides a short explanation and a score for each question. The final score of each view is obtained by aggregating the per-question scores.

In the Translate stage, the MLLM decomposes the instruction into structured photographic attributes. Specifically, three types of prompts are used, as illustrated in Fig. 16: (1) a subject parser that extracts the main subjects mentioned in the instruction, (2) a layout parser that predicts the desired spatial arrangement of these subjects within the frame, and (3) an orientation parser that infers camera viewing angles or subject-facing directions when such cues are present in the instruction. These outputs are subsequently converted into geometric pseudo labels that guide camera pose optimization.

![](images/6ec72c324f0021d904cfb3c52ba2b87f6cfb5e200633e5b4f9e1e65320623240.jpg)  
Fig. 15: Prompts used in the Retrieve stage. The top prompt asks the MLLM to generate evaluation questions from the text instruction, and the bottom prompt asks the MLLM to score a candidate image by answering those questions.

![](images/d596cf41e7db6c3ee69a9f82b42ec5ade74856a6326a4932567704fbd3edcfec.jpg)  
Fig. 16: Prompts used in the Translate stage. From left to right, we show the prompts for the subject parser, the layout parser, and the orientation parser, which convert a free-form instruction into geometric pseudo labels for camera pose optimization.

## Thank you for participating in this study.

In this study, we compare different methods for automatically placing a virtual camera in a 3D scene based on a text description.

You will complete 12 tasks in total.   
In each task, you will see a text description and three images corresponding to that text.   
Please evaluate how well each image matches the text description.

Note: Some images may have small visual flaws (such as blurry areas or weird objects).

These flaws are not related to camera placement.   
Please ignore them and focus only on how well the image content matches the text description. There are 48 questions in total (3 rating questions followed by 1 choice question) For each image, rate how well it matches the text using a 5-point scale:   
1 = Very poor match   
5 = Excellent match   
After rating all three images, select the one that best matches the text overall.

The study will take about 10-15 minutes to complete.

(a) Instructions for Participants  
![](images/9d424960d17f1aa6fccbe01872299ffaadf22067c4320b3794c442b899dd00e8.jpg)  
(b) Rating Example

![](images/2ffd057132ea04dc767e3180bf21f777042368f1e5025b8f877d41deed66fce2.jpg)  
(c) Multiple-Choice Example  
Fig. 17: Example user-study interface. (a) Instructions shown to participants. (b) Example of the 5-point rating question. (c) Example of the final multiple-choice question used to select the best-matching image.

## G User Study Details

A total of 33 participants took part in the user study, including individuals both with and without computer science backgrounds. The study evaluates how well rendered images align with the given text instructions.

The evaluation consists of 12 question groups. In each group, participants are presented with one text instruction and three candidate images generated by diferent methods. Participants first rate each image on a 5-point scale according to how well the visual content matches the instruction (1: very poor match; 5: excellent match). After rating all three images, they select the image that best matches the instruction overall.

In total, the study contains 48 questions: three rating questions and one selection question per group. The order of the candidate methods is randomized to avoid presentation bias. Participants are instructed to focus only on the alignment between the image content and the text description, while disregarding visual artifacts unrelated to camera placement. Examples of the questionnaire are shown in Fig. 17.