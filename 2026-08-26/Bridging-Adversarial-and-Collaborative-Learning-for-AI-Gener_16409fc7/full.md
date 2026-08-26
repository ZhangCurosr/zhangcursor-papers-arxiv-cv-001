# Bridging Adversarial and Collaborative Learning for AI-Generated

Image Quality Assessment

Baoliang Chen\*, Qing Lin\*, and Sijie Mai

Abstract—AI-generated image quality assessment (AIGIQA) requires jointly reasoning about perceptual fidelity and prompt alignment, two quality dimensions that are often treated as independent in existing AIGIQA models. However, by reexamining human ratings, we uncover a previously overlooked phenomenon: the two dimensions are interdependent and exhibit both competitive and cooperative interactions during human rating. This observation suggests that a unified model should neither collapse the two dimensions nor rigidly separate them, but rather adaptively negotiate their interplay. Motivated by this insight, we introduce an interaction-aware learning framework that models perception–alignment relations through adversarial and collaborative inference pathways. Instead of designing a rigid dual-branch architecture, our method employs a gated interaction module that dynamically routes features according to the inferred relationship between the two dimensions. Taskaware prompts further modulate the gating behaviour, enabling the model to switch between competition and cooperation when necessary. Experiments across multiple AIGIQA benchmarks demonstrate that our approach not only achieves state-of-theart accuracy but also yields interpretable interaction patterns, offering a more faithful approximation of human judgment. The codes are available at https://github.com/LQAMEI/ACL-IQA.

Index Terms—AI-generated image quality assessment, Adversarial and collaborative learning, Mixture-of-Experts.

## I. INTRODUCTION

Artificial Intelligence Generated Image Quality Assessment (AIGIQA) has emerged as an important problem in evaluating the reliability of modern generative models. Unlike traditional image quality assessment (IQA), where degradations mainly arise from physical limitations (e.g., sensor noise, compression, or enhancement artifacts), the distortions in AIGIs are fundamentally different. They stem from the generative process itself, often guided by user prompts, and can manifest as surreal textures, implausible geometries, or subtle mismatches between image and text semantics. This paradigm shift transforms AIGIQA into a multi-dimensional evaluation problem that requires assessing both the visual realism of an image (perceptual quality), and its adherence to the textual description (prompt alignment).

Over the past decades, numerous IQA models [1]–[6] have been developed to handle common degradations like noise or blur. However, these methods are inherently limited when confronted with the unique distortions introduced by modern generative models, distortions that are semantic rather than photometric. Moreover, since they typically operate solely on visual inputs, they cannot directly evaluate prompt alignment. To address these limitations, AIGIQA-specific methods have emerged. For example, TIER [7] employs dual encoders for image and text to jointly model their correspondence, while IPCE [8] leverages CLIP [9] to capture broader image-text alignment. Yet, despite these promising efforts, one crucial aspect remains underexplored: the intrinsic interaction between perceptual and alignment quality. Existing approaches usually treat them as independent tasks, overlooking the subtle ways in which they influence each other during human judgment. As illustrated in Fig. 1, this interaction can be either collaborative or adversarial. For instance, in cases such as I1–I3 and I4– I6, prompts containing quality-related words cause the two assessments to reinforce each other, where high-quality visuals tend to be perceived as better aligned, and vice versa. In other cases such as I7–I9 and I10–I12, the two aspects behave independently or even conflict, suggesting that they should be disentangled. This observation raises an intriguing question:

Prompt: photograph of beautiful landscape, baroque style  
![](images/8e17c3275cc136f03172b37f6d853da5fbb47463dce79995f278b3dd36a30f64.jpg)  
Prompt: a cosmic universe

![](images/9eace79ac4a59f4f103056d126f4329f391baa9bd58b455036845dccfa004a05.jpg)

![](images/a71a6599dd88ed7ec16456d78a606cb482431eaea54114df44e11e30cd4655c4.jpg)

![](images/b7c3fa558208b9e477f775435937c31d07eb8b907f4753776ef4a8ceec32aa77.jpg)

Prompt: portrait of a tang dynasty empress, soft lighting  
![](images/9523d41ad9b1553766ae8ab0be5a8688ef04b5c1ed3424dca4facf25699aae70.jpg)

![](images/2eefaeb3a3d235c93613ee356c7821324f9649d989425f4a19f6681d3723850e.jpg)

Prompt: mickey mouse head automaton held by a mechanical claw..  
![](images/5987a6d6d029f11022ea3215ac3e06b0d43370c12f28f3052558d86687b2f1d5.jpg)

![](images/a388e7c14d57bd015d2dc7428f94a009b30538d87b7a113736162a1982cc1916.jpg)  
Fig. 1. Illustration of adversarial and collaborative interactions in human rating during AIGIQA. The ground-truth quality scores of each dimension are depicted in the line charts.

## How should we estimate the two types of quality in AIGIs—adversarially, collaboratively, or both?

In this paper, we argue that the answer is both. Human ratings indicate that perceptual quality and prompt alignment do not behave as independent attributes; instead, they exhibit both adversarial and collaborative interactions, suggesting that an effective AIGIQA model must explicitly account for these dual modes of behavior. Building on this insight, we design a learning framework that encodes this duality rather than treating it as an incidental property of the architecture. Specifically, we define a collaborative relationship as the synergy where perceptual and alignment dimensions share complementary information. This is evident when textual prompts contain quality related descriptors, where visual realism and semantic adherence mutually reinforce each other. Conversely, an adversarial relationship is defined as the active decoupling of interfering information. This mimics the human cognitive process of suppressing bias, where the model must actively strip away distractor features, such as aesthetic appeal that might inflate alignment scores, to ensure independent assessment.

To operationalize these relationships, we introduce a Dual-Gated Mixture-of-Experts (DG-MoE) framework that models perception–alignment interactions through two complementary learning paths. The collaborative path captures mutually reinforcing cues between perceptual and semantic signals to produce joint predictions, while the adversarial path employs a disentangling objective to expose conflicts that may bias quality estimation. DG-MoE utilizes prompt-guided gating functions to selectively route visual features to expert networks specialized for either collaborative reasoning or adversarial disentanglement. By explicitly modeling these interactions, our approach ensures that the network focuses on task specific features without corrupting the shared latent space, thereby capturing a richer set of interaction patterns to produce a more holistic quality estimate.

Our main contributions are summarized as follows:

• We provide the first empirical evidence that human AIGI quality ratings exhibit both adversarial and collaborative interactions between perceptual and alignment dimensions, motivating a dual-path formulation.

• We propose a novel AIGIQA framework that jointly models adversarial and collaborative interactions, enabling complementary reasoning about perceptual fidelity and prompt alignment.

• We introduce a Dual-Gated Mixture-of-Experts (DG-MoE) module that dynamically routes expert features through collaborative or adversarial paths within a unified transformer architecture.

• Extensive experiments on multiple AIGIQA benchmarks demonstrate the superiority of our approach and reveal strong complementarity between the two learning paths.

## II. RELATED WORK

## A. AIGI Quality Assessment

The rapid progress of generative models, from Generative Adversarial Networks (GANs) [10] and autoregressive models [11] to diffusion models [12], has profoundly accelerated the development of AIGIs. Despite continuous advancements in visual quality, generated images still face challenges such as structural inconsistencies, semantic deviations, and loss of details, preventing AIGIs from fully meeting practical requirements. In response to these challenges, AIGIQA has become a crucial task for optimizing AIGI models [13]–[15]. Pioneering studies have released benchmark datasets for subjective evaluation, including AGIQA-1K [13], AGIQA-3K [14], and AIG-CIQA2023 [16]. More recently, contemporary benchmarks have significantly expanded in both scale and diversity to accommodate increasingly complex prompts and generation behaviors. For instance, large-scale databases such as AGIQA-20K [17] and EvalMi-50K [18] introduce massive human annotations across comprehensive quality dimensions. Concurrently, specialized datasets including EvalMuse-40K [19], GenAI-Bench [20], T2I-CompBench [21], GenEval [22], and RichHF-18K [23] shift the evaluation focus toward subtle compositional errors, fine-grained spatial relations, and localized artifact distributions.

For perceptual quality assessment, classical metrics such as the Inception Score (IS) [24] and Frechet Inception Distance´ (FID) [25] have been widely adopted to quantify image realism and diversity. Early no-reference IQA methods have provided valuable insights for AIGI quality evaluation. For example, Fang studied perceptual quality of smartphone photography [1], Zhu and Li proposed MetaIQA for adaptive NR-IQA using meta-learning [26], and Sun introduced GraphIQA [27], which models distortion types via a graph representation for blind IQA. Recent studies have advanced this task by incorporating semantic understanding, where hybrid text encoders are employed to enrich feature representations [28], and learnable textual or visual prompts are introduced to enhance prediction accuracy [29]. Concurrently, to better handle diverse structural distortions and generation artifacts, methods such as TReS [30], ManIQA [31], and Re-IQA [32] explore multiscale feature extraction, global transformer backbones, and relative ranking strategies to improve representation robustness.

For alignment assessment, CLIP-based methods [9] have be come the dominant paradigm. LIQE [33] introduces languageguided alignment to map visual features with quality descriptive words. Extending this cross-modal paradigm to AIGIQA, IP-IQA [34] integrates explicit text prompts to concurrently regress both perceptual and alignment quality, while Peng et al. [8] propose IPCE to measure the cosine similarity between generated images and tailored prompts. In TIER [7], dual encoders for image and text are jointly optimized to regress both perceptual and alignment quality through cross-modal fusion. Yu et al. [35] propose SF-IQA, which integrates quality and alignment scores via a dedicated fusion module. To capture finer-grained correlations, Xia et al. [36] design task-specific prompts to measure both coarse- and fine-grained similarities between AIGIs and their textual descriptions. To alleviate the entanglement between semantic understanding and distortion perception, MST-CLIPIQA [37] adopts two CLIP streams with complementary patch granularities to separately capture global semantic coherence and local artifact-sensitive cues, followed by adaptive cross-scale fusion for quality prediction. Along this line, the recent wave of Vision-Language Models and Multimodal Large Language Models (MLLMs) has introduced sophisticated evaluators. Models such as FGA-BLIP2 [19] and RALI [38] focus heavily on fine-grained compositional constraints and human feedback text-matching. Drawing insights from human cognitive mechanisms, Liao et al. [39] proposed the AGQG module, which plugs into existing architectures to accurately calibrate AIGIQA scores. SC-AGIQA [40] explicitly addresses semantic misalignment and insufficient fine-grained distortion perception through two complementary modules: an MLLM-assisted semantic alignment module that compares generated image descriptions with the conditioning prompt, and a frequency-domain degradation perception module that enhances sensitivity to subtle perceptual distortions. Massive MLLM-based architectures such as Q-Align [41], MA-AGIQA [42], LMM4LMM [18], DeQA-Score [43], and Q-Insight [44] leverage the strong reasoning capabilities of foundation networks to reformulate both perception and alignment assessment as visual question answering or discrete text-defined level generation. While these approaches have achieved remarkable progress, they generally treat perceptual and alignment assessments as independent objectives. The intrinsic interplay between the two, however, remains largely unexplored, leaving a crucial gap in understanding how human perception jointly evaluates quality and alignment in AIgenerated content.

## III. METHODOLOGY

Overview. Given an AIGI and its corresponding prompt, our objective is to predict both its perception quality and alignment quality scores. As depicted in Fig. 2, our proposed dualpath learning framework comprises three main components: Multimodal Inputs, Text and Image Encoders, and Learning Objectives. Initially, the input AIGI is segmented and resized into multiple patches to effectively capture both local and global information. Additionally, we craft four text templates to instruct the learning (collaborative and adversarial) strategy and quality (perception and alignment) prediction tasks. For the image encoding phase, we introduce a Dual-Gated MoE block, extracting distinct image components tailored for the adversarial and collaborative learning, respectively. This differentiation is achieved through guidance derived from the learning instruction texts. Regarding the learning objectives, we simultaneously predict both perception and alignment quality scores in the collaborative learning, while a Gradient Reversal Layer (GRL) is utilized to disentangle the intertwined features during adversarial learning. The final prediction is obtained by synthesizing the dual learning paths, combining insights from both collaborative and adversarial processes. To facilitate the presentation of our multi-modal dual-path learning framework, we globally standardize the key symbols and definitions in Table I. The details of each component are elaborated as follows.

Mixture-of-Experts (MoE) models [45] have emerged as a powerful paradigm for enhancing model performance and generalizability without excessively increasing computational complexity. The core idea behind MoE is to introduce multiple expert networks and a gating mechanism that dynamically selects the most relevant experts based on the inputs. In MoE, the gating strategy usually plays a crucial role, determining both model efficiency and expert selection. Sparse MoE models [46] improve computational efficiency by utilizing a router to activate only a subset of experts, ensuring that inference time remains comparable to traditional methods while enabling the construction of large-scale models [47]. Task-specific gating networks have been extensively studied to improve expert selection in multi-task settings [48]–[51], aiming to select the appropriate parts of a model based on the specific task. For example, Ma et al. propose the MMoE [49] where separate gating networks are designed for each task, enabling selective expert utilization. In M<sup>3</sup>ViT [51], the task-related semantics are introduced in the gating design by converting the task information into a one-hot embedding. Inspired by these advancements, our work models different experts as distinct convolution kernels and introduces a novel dual gating network towards the adversarial and cooperative learning, respectively.

## B. Mixture of Expert

TABLE I  
ILLUSTRATION OF KEY MATHEMATICAL SYMBOLS AND NOTATIONS.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $T _ { c o l } , T _ { a d v }$ </td><td>Textual instructions for collaborative and adversarial learning strategy.</td></tr><tr><td> $T _ { p e r } , T _ { a l n }$ </td><td>Textual prompt templates for perception and align- ment assessment.</td></tr><tr><td> $F _ { c o l } ^ { T } , F _ { a d v } ^ { T }$ </td><td>Textual learning instruction features derived from templates  $T _ { c o l }$  and  $T _ { a d v } .$ </td></tr><tr><td> $F _ { p e r } ^ { T } , F _ { a l n } ^ { T }$ </td><td>Textual quality assessment features derived from templates  $\dot { T } _ { p e r }$  and  $T _ { a l n }$ </td></tr><tr><td> $F _ { c o l } ^ { I } , F _ { a d v } ^ { I }$ </td><td>Decoupled image features routed for collaborative and adversarial learning paths.</td></tr><tr><td> $G _ { c } ( \cdot ) , G _ { a } ( \cdot )$ </td><td>Dual gating networks guided by instructional textual features.</td></tr><tr><td> $E _ { j } ( \cdot ) , W _ { j } ^ { E }$ </td><td>The j-th difference convolution expert network and its gating weight.</td></tr><tr><td> $S _ { \sim \sim \dots } ^ { c o l } S _ { \sim \sim \dots } ^ { a d v } .$   $S ^ { i c o l } . S ^ { a d v }$   $S _ { a l n } ^ { \mathrm { { c v a v a } } } , S _ { a l n } ^ { \mathrm { { w a v a } } }$ </td><td>Intermediate similarity matrices estimated between decoupled image features and textual quality fea- tures.</td></tr><tr><td> $S _ { p e r } , S _ { a l n }$ </td><td>Final synthesized similarity matrices aggregated from the collaborative and adversarial paths.</td></tr><tr><td> $Q _ { p e r } ^ { c o l } , Q _ { p e r } ^ { a d v } ;$   $Q _ { a l n } ^ { \dot { c } o l } , Q _ { a l n } ^ { \dot { a } d v }$ </td><td>Intermediate quality scores predicted by individual learning paths.</td></tr><tr><td> $Q _ { p e r } , Q _ { a l n }$ </td><td>Estimated perception and alignment quality scores predicted by the joint learning framework.</td></tr><tr><td> $\hat { Q } _ { p e r } , \hat { Q } _ { a l n }$ </td><td>Ground-truth Mean Opinion Scores (MOS) for hu- man perceptual and alignment quality.</td></tr></table>

![](images/ee365a8234092582b541e27d66caa55b6ece3454feac26ce46f5932edd03dd69.jpg)  
Fig. 2. Overview of our AIGIQA model incorporating both adversarial and collaborative learning.

## A. Multimodal Inputs

1) Image Preprocessing: Given AIGIs of varying sizes, we first segment the image into multiple patches with a standard size of $2 2 4 \times 2 2 4$ . However, segmentation may negatively impact semantic integrity, as a single object can be divided across multiple patches [8]. To address this, our method incorporates both segmented patches and one resized image as input and the final prediction is obtained through a weighted fusion of both types.

2) Text Templates: Two types of text templates are designed in our method. The first set consists of two quality assessment templates designed for different quality level classifications:

$T _ { p e r } : { ^ { \ast } \mathrm { { A } } }$ photo of {c} quality”,

$$
c \in \{ ^ { \mathfrak { c } } \mathfrak { b } \mathrm { a } \mathrm { d } ^ { \mathfrak { p } } , ^ { \mathfrak { c } } \mathrm { p o o r } ^ { \mathfrak { p } } , ^ { \mathfrak { c } } \mathrm { f a i r } ^ { \mathfrak { p } } , ^ { \mathfrak { c } } \mathrm { g o o d } ^ { \mathfrak { p } } , ^ { \mathfrak { c } } \mathrm { p e r f e c t } ^ { \mathfrak { p } } \} ,
$$

$T _ { a l n } : { ^ { \ast } \mathrm { A } }$ photo that {v} matches ‘prompt’ ”,

$$
v \in \{ { } ^ { \ast } \mathrm { b a d l y } ^ { \mathrm { , ~ \ast } } \mathrm { p o o r l y } ^ { \mathrm { , ~ \ast } } , { } ^ { \ast } \mathrm { f a i r l y } ^ { \mathrm { , ~ \ast } } \mathrm { w e l l } ^ { \mathrm { , ~ \ast } } \mathrm { p e r f e c t l y } ^ { \mathrm { , ~ \ast } } \} ,
$$

where $T _ { p e r }$ and $T _ { a l n }$ correspond to perception quality and alignment quality assessment, respectively. Another two templates are the learning strategy instruction, serving as guidance for extracting different image components during training:

$T _ { c o l }$ : “In a photo like ‘prompt’, using {T1} to assist in evaluating {T2}”,

$T _ { a d v } : { } ^ { \ast } { \mathrm { I n } }$ a photo like ‘prompt’, excluding consideration of {T1} when evaluating {T2}”,

$$
\mathrm { T 1 } = \left\{ \begin{array} { c } { { \mathrm {  s i m a g e ~ q u a l i t y ^ { , } ~ i f ~ T 2 = ^ { * } a l i g n m e n t ^ { , } } ; } } \\ { { \mathrm { \quad ~ \cdots a l i g n m e n t ^ { , } ~ i f ~ T 2 = ^ { * } i m a g e ~ q u a l i t y ^ { , } . } } } \end{array} \right.
$$

Herein, $T _ { c o l }$ and $T _ { a d v }$ denote the templates correspond to the collaborative and adversarial learning instructions, respectively.

## B. Text and Image Encoders

As shown in Fig. 2, we design a text encoder and an image encoder to extract the text and image features, respectively.

Specifically, the text encoder extracts four types of features: the collaborative and adversarial learning instruction features $F _ { c o l } ^ { T }$ and $F _ { a d v } ^ { T }$ , along with the perception and alignment quality assessment features $F _ { p e r } ^ { T }$ and $F _ { a l n } ^ { \dot { T } }$ . The four features are derived from the text templates $T _ { c o l } , \ T _ { a d v } , \ T _ { p e r }$ and $T _ { a l n } ,$ respectively. The image encoder extracts two types of image features, namely the collaboratively learned feature $F _ { c o l } ^ { I }$ and the adversarially learned feature $\dot { F } _ { a d v } ^ { I }$ , representing different components decomposed from the image for the dual-path (collaborative and adversarial) prediction. In particular, we implemented this decomposition through the proposed dualgated MoE block, where distinct expert networks are designed for specialized image feature extraction. The flow paths of the extracted features are then guided by the learning-instruction features $F _ { c o l } ^ { T }$ and $F _ { a d v } ^ { T }$ to facilitate either collaborative or adversarial learning. The architectural details are described as follows.

1) Text Encoder: We implement the text encoder following the standard CLIP network. Specifically, a tokenization layer and 12 transformer blocks are adopted. For each block in the odd stages $( u \ : = \ : 1 , 3 , \ldots , 1 1 )$ , we extract the learning instruction features $F _ { c o l , u } ^ { T }$ and $F _ { a d v , u } ^ { T }$ and input them into the Dual-gated MoE block to tune the gating network.

2) Image Encoder: We adopt the Vision Transformer (ViT) [52] as the base architecture for the image encoder. The encoder comprises one linear projection layer, six transformer blocks, and six Dual-gated MoE blocks, with the transformer blocks and Dual-gated MoE blocks arranged alternately. The goal of the image encoder is to extract two types of features, $F _ { c o l } ^ { I }$ and $F _ { a d v } ^ { I } ,$ for collaborative and adversarial learning, respectively. As depicted in Fig. 3, we start by initializing the feature extracted from the first transformer block as $F _ { 1 } ^ { I }$ The outputs of the transformer block at the l-th stage are denoted as $F _ { c o l , l } ^ { I }$ and $F _ { a d v , l } ^ { I } ~ ( l = 1 , 2 , \dots , 6 )$ . We initialize $F _ { 1 } ^ { I } = F _ { c o l , 1 } ^ { I } = F _ { a d v , 1 } ^ { I } ,$ and the function of the Dual-gated

![](images/816fd6044eb862163c528cfe4c4d19f062e4e59fd13b14c20acc27b3b94f8650.jpg)  
Fig. 3. Design details of our proposed dual-gated MoE block.

MoE blocks, denoted as $f ( \cdot )$ , can be expressed as follows:

$$
\begin{array} { r l } {  { \hat { F } _ { c o l , l } ^ { I } , \hat { F } _ { a d v , l } ^ { I } = f ( F _ { c o l , l } ^ { I } , F _ { a d v , l } ^ { I } ) } } \\ & { = f _ { c } ( F _ { c o l , l } ^ { I } ) , f _ { a } ( F _ { a d v , l } ^ { I } ) , } \end{array}\tag{1}
$$

where $f _ { c } ( \cdot )$ and $f _ { a } ( \cdot )$ represent the two sub-networks responsible for collaborative and adversarial feature extraction, respectively. To simplify the explanation, we take $F _ { c o l , l } ^ { I }$ as an example to illustrate the design details of $f _ { c } ( \cdot )$ , as both $f _ { c } ( \cdot )$ and $f _ { a } ( \cdot )$ follow the same design philosophy, as shown in Fig. 3:

$$
\begin{array} { r l } & { \hat { F } _ { c o l , l } ^ { I } = f _ { c } ( F _ { c o l , l } ^ { I } ) } \\ & { \quad \quad \quad = F _ { c o l , l } ^ { a t t } + \mathrm { F F N } ( \mathrm { L N } ( F _ { c o l , l } ^ { a t t } ) ) + F _ { c o l , l } ^ { m o e } , } \end{array}\tag{2}
$$

with

$$
F _ { c o l , l } ^ { a t t } = F _ { c o l , l } ^ { I } + \mathsf { M S A } ( \mathbf { L N } ( F _ { c o l , l } ^ { I } ) ) ,\tag{3}
$$

$$
F _ { c o l , l } ^ { m o e } = \sum _ { j = 1 } ^ { N _ { e } } W _ { j } ^ { E } E _ { j } ( \mathbf { L N } ( F _ { c o l , l } ^ { a t t } ) ) ,\tag{4}
$$

where LN, FFN, and MSA represent the standard layer normalization, feed-forward network, and multi-head selfattention layers in the ViT. $N _ { e }$ denotes the number of experts in the Dual-gated MoE block. $E _ { j } ( \cdot )$ represents the function of the j-th expert network, and $\bar { W } _ { j } ^ { E }$ is the weight predicted by a gating network $G _ { c } ( \cdot )$ . The design details of the $G _ { c } ( \cdot )$ and $E _ { j } ( \cdot )$ are elaborated as follows.

• Dual Gating Network $G ( \cdot )$ . The dual gating network aims to estimate the weight $W _ { j } ^ { E }$ of each expert output with the two learning text features $F _ { c o l , u } ^ { T }$ and $F _ { a d v , u } ^ { T }$ | $u = ( 2 l - 1 )$ as instruction. We denote the gating networks guided by $F _ { c , u } ^ { T }$ and $F _ { a , u } ^ { T }$ as $G _ { c } ( \cdot )$ and $G _ { a } ( \cdot )$ , respectively. As shown in Fig. $3 , G _ { c } ( \cdot )$ and $G _ { a } ( \cdot )$ share the same design. For simplicity, we use $G _ { c } ( \cdot )$ as an example then the $W _ { j } ^ { E }$ thus can be calculated by:

$$
\begin{array} { r l } & { W _ { j } ^ { E } = G _ { c } ( F _ { c o l , l } ^ { I } , F _ { c o l , 2 l - 1 } ^ { T } ) } \\ & { \qquad = \mathrm { S o f t m a x } ( \mathrm { T o p k } ( W _ { j } ^ { g a t e } F _ { g a t e } , k ) ) , } \end{array}\tag{5}
$$

with

$$
F _ { g a t e } = \mathrm { A v g P o o l } ( \mathrm { C r o s s A t t e n } ( F _ { c o l , l } ^ { I } , F _ { c o l , u } ^ { T } ) ) ,\tag{6}
$$

where Softmax(·) and Topk(·) represent the functions of softmax activation and top-k selection, respectively. $W _ { j } ^ { g a t e }$ are learnable weights that project the gating feature $F _ { g a t e }$ to a single value, which is used to weight the j-th expert’s output. The CrossAtten(·) is defined as follows:

$$
\begin{array} { r l r } & { } & { \mathrm { C r o s s A t t e n } ( F _ { c o l , l } ^ { I } , F _ { c o l , u } ^ { T } ) = \mathrm { S o f t m a x } \left( \frac { Q K ^ { T } } { \sqrt { D } } \right) V , } \\ & { } & { Q = F _ { c o l , l } ^ { I } W ^ { Q } , \quad K = F _ { c o l , u } ^ { T } W ^ { K } , \quad V = F _ { c o l , u } ^ { T } W ^ { V } , } \end{array}\tag{7}
$$

where $W ^ { K } , \ W ^ { V }$ , and $W ^ { Q }$ are learnable projection matrices. D represents the dimension of the keys.

• Expert Network $E _ { j } ( \cdot )$ . We design the expert networks using difference convolution layers [53], [54] (denoted as ConvExpert) in Fig. 3. The design philosophy lies in that the gradient information between adjacent tokens, captured by the difference convolution kernels, provides more fine-grained and high-frequency details. This plays a complementary role to the transformer network, which is more inclined to capture low-frequency information [55]. In our method, we adopt a total of six types of ConvExperts, including Vanilla convolution and five difference convolutions: Horizontal Difference Convolution (HDC), Vertical Difference Convolution (VDC), Central Difference Convolution (CDC), Angular Difference Convolution (ADC), and Radial Difference Convolution (RDC). The kernels of the five difference convolutions are depicted in Fig. 4. More specifically, denoting the input feature $\mathrm { L N } ( F ^ { a t t } F _ { c o l , l } ^ { I } )$ (see Eqn. (4)) as $F _ { \mathrm { e x p } } .$ the ConvExpert with the j-th kernel can be expressed as follows:

$$
E ( F _ { e x p } ) _ { j } = \mathrm { C o n v } _ { u p } \left( \mathrm { C o n v } _ { j } \left( \mathrm { C o n v } _ { d o w n } ( F _ { e x p } ) \right) \right) ,\tag{8}
$$

where the $\mathbf { C o n v } _ { j }$ means the convolutional layer with the j-th kernel. $\mathbf { C o n v } _ { d o w n }$ and $\mathbf { C o n v } _ { u p }$ are two $1 \times 1$ convolutional layers aiming for the channel dimension downsampling and upsampling (restoring) for computational efficiency.

Through the expert networks, different components can be extracted from the image and routed to the collaborative and adversarial learning paths with the guidance obtained by $G _ { c } ( \cdot )$ and $G _ { a } ( \cdot )$ , respectively.

![](images/cf383a11b778abdf7cb21314029e9060bcfae3e05ddac5087d299e6250101551.jpg)  
Fig. 4. Illustration of five types of convolution kernels [54]. The arrow represents the difference between the weight at its starting point and the weight at its endpoint.

## C. Collaborative and Adversarial Learning

Based on the obtained image features $F _ { c o l } ^ { I } , F _ { a d v } ^ { I }$ and quality assessment features $F _ { p e r } ^ { T }$ and $F _ { a l n } ^ { T } ,$ , we compute the cosine similarity between them, resulting in four types of similarity matrices:

$$
\begin{array} { r l r } & { S _ { p e r } ^ { c o l } = \frac { \boldsymbol { F } _ { c o l } ^ { I } \odot \boldsymbol { F } _ { p e r } ^ { T } } { \lVert \boldsymbol { F } _ { c o l } ^ { I } \rVert \cdot \lVert \boldsymbol { F } _ { p e r } ^ { T } \rVert } , } & { S _ { p e r } ^ { a d v } = \operatorname { G R L } ( \frac { \boldsymbol { F } _ { a d v } ^ { I } \odot \boldsymbol { F } _ { p e r } ^ { T } } { \lVert \boldsymbol { F } _ { a d v } ^ { I } \rVert \cdot \lVert \boldsymbol { F } _ { p e r } ^ { T } \rVert } ) , } \\ & { S _ { a l n } ^ { c o l } = \frac { \boldsymbol { F } _ { c o l } ^ { I } \odot \boldsymbol { F } _ { a l n } ^ { T } } { \lVert \boldsymbol { F } _ { c o l } ^ { I } \rVert \cdot \lVert \boldsymbol { F } _ { a l n } ^ { T } \rVert } , } & { S _ { a l n } ^ { a d v } = \operatorname { G R L } ( \frac { \boldsymbol { F } _ { a d v } ^ { I } \odot \boldsymbol { F } _ { a l n } ^ { T } } { \lVert \boldsymbol { F } _ { a d v } ^ { I } \rVert \cdot \lVert \boldsymbol { F } _ { a l n } ^ { T } \rVert } ) , } \end{array}\tag{9}
$$

where GRL(·) denotes the Gradient Reversal Layer [56], which is designed to ablate the specific quality information through negative gradient backpropagation. Additionally, we employ a trainable weighting scheme for the fusion of $S _ { p e r } ^ { c o l }$ and $S _ { p e r } ^ { a d v }$ , which serves as the final estimation for the perception and alignment quality:

$$
\begin{array} { r } { S _ { p e r } = S _ { p e r } ^ { c o l } \times \theta + S _ { p e r } ^ { a d v } \times ( 1 - \theta ) , } \\ { S _ { a l n } = S _ { a l n } ^ { c o l } \times \theta + S _ { a l n } ^ { a d v } \times ( 1 - \theta ) , } \end{array}\tag{10}
$$

where θ is a trainable parameter. Herein, the matrices $S _ { p e r } ^ { c o l } , S _ { a l n } ^ { c o l } , S _ { p e r } ^ { a d v } , S _ { a l n } ^ { a d v } , S _ { p e r } , S _ { a l n } \in \mathbb { R } ^ { M \times 5 }$ , where M represents the number of image patches, and $\mathbf { \Delta } ^ { 6 6 } 5 ^ { , 5 }$ denotes the five quality levels defined in $T _ { p e r }$ and $T _ { a l n }$ . We denote the M-th patch as the resized one. Then, we convert the six similarity matrices into their corresponding quality scores $Q _ { p e r } ^ { c o l } , \dot { Q _ { a l n } ^ { c o l } } , Q _ { p e r } ^ { a d v } , Q _ { a l n } ^ { a d v } , Q _ { p e r } , Q _ { a l n }$ , respectively. Taking the conversion from $S _ { p e r } ^ { c o l }$ to $Q _ { p e r } ^ { c o l }$ as an example:

$$
Q _ { p e r } ^ { c o l } = \frac { 5 } { 4 } \left( \sum _ { i = 1 } ^ { 5 } i \times P _ { i } - 1 \right) ,\tag{11}
$$

with

$$
\begin{array} { r l r } {  { P _ { i , j } = \mathrm { S o f t m a x } ( S _ { p e r } ^ { c o l } ) , } } \\ & { } & { P _ { i } = ( \frac { \sum _ { j = 1 } ^ { M - 1 } P _ { i , j } } { M - 1 } + P _ { i , M } ) / 2 , } \end{array}\tag{12}
$$

where the Softmax function is applied along the quality level dimension. $P _ { i , j }$ denotes the quality probability of the j-th patch under the i-th level. Finally, we define the supervision loss functions as follows:

• For the perception quality assessment task:

$$
L _ { p e r } = L _ { p e r } ^ { g t } + \lambda _ { 1 } L _ { a l n } ^ { c } + \lambda _ { 2 } L _ { a l n } ^ { a } + \lambda _ { 3 } L _ { m o e } ,\tag{13}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } .$ , and $\lambda _ { 3 }$ are three hyperparameters. $L _ { m o e }$ is an expert utilization balancing loss that encourages all experts to have equal importance [46], and

$$
\begin{array} { r } { L _ { p e r } ^ { g t } = \rvert \hat { Q } _ { p e r } - Q _ { p e r } \rvert , } \\ { L _ { a l n } ^ { c } = \rvert \hat { Q } _ { a l n } - Q _ { a l n } ^ { c } \rvert , } \\ { L _ { a l n } ^ { a } = \rvert \hat { Q } _ { a l n } - Q _ { a l n } ^ { a } \rvert , } \end{array}\tag{14}
$$

where $\hat { Q } _ { p e r }$ and $\hat { Q } _ { a l n }$ are the ground-truth perception and alignment quality scores, respectively.

• For the alignment quality assessment task:

$$
L _ { a l n } = L _ { a l n } ^ { g t } + \lambda _ { 1 } L _ { p e r } ^ { c } + \lambda _ { 2 } L _ { p e r } ^ { a } + \lambda _ { 3 } L _ { m o e } ,\tag{15}
$$

where

$$
\begin{array} { r } { L _ { a l n } ^ { g t } = \rvert \hat { Q } _ { a l n } - Q _ { a l n } \rvert , } \\ { L _ { p e r } ^ { c } = \rvert \hat { Q } _ { p e r } - Q _ { p e r } ^ { c } \rvert , } \\ { L _ { p e r } ^ { a } = \rvert \hat { Q } _ { p e r } - Q _ { p e r } ^ { a } \rvert . } \end{array}\tag{16}
$$

In the testing phase, we adopt $Q _ { p e r }$ and $Q _ { a l n }$ as the estimated perception quality score and alignment quality score, respectively.

## IV. EXPERIMENT

## A. Experimental Setting

1) Datasets: We conduct comprehensive experiments on five publicly available AIGIQA datasets: AGIQA-1K [13], AGIQA-3K [14], AIGCIQA2023 [16], and two massive benchmarks, AGIQA-20K [17] and EvalMi-50K [18]. AGIQA-1K consists of 1,080 images generated by Stable Inpainting v1 and Stable Diffusion v2, using 180 prompts derived from high-frequency keywords. Each image is assigned a Mean Opinion Score (MOS) based on both perception quality and text-image correspondence. AGIQA-3K contains 2,982 images generated by six text-to-image (T2I) models. Each image is annotated with both perception quality and text-image alignment score. AIGCIQA2023 includes 2,400 images generated by six T2I models using 100 prompts. The annotations cover perception quality, authenticity, and correspondence. To rigorously validate scalability against modern generative priors, AGIQA-20K comprises 20,000 images generated by 15 mainstream models across diverse sampling configurations. Furthermore, EvalMi-50K features 50,400 images synthesized by 24 recent text-to-image generators (e.g., DALL-E 3, Flux, Stable Diffusion 3.5) with highly complex textual prompts, providing robust MOS for Perception and Correspondence. Evaluations on fine-grained alignment and artifact datasets (EvalMuse-40K, GenAIBench, and RichHF-18K) are provided in the Supplementary Material.

2) Implementation Details: We use the CLIP model with the ViT-B/32 architecture as our base network. For singledimension evaluations including AGIQA-1K and the authenticity assessments of AIGCIQA2023 and AGIQA-20K, we retain only the single-gated MoE from our original model. The experiments are trained for 200 epochs on the early-stage datasets and for 50 epochs when evaluated on the massive contemporary benchmarks (AGIQA-20K and EvalMi-50K). During training, the batch size is set to 16, and we use the

TABLE II  
PERFORMANCE COMPARISON OF PERCEPTION QUALITY PREDICTION ON THE AGIQA-1K. THE BEST TWO RESULTS ARE IN BOLDFACE.
<table><tr><td>Method</td><td>SRCC</td><td>PLCC</td><td>Mean Score</td></tr><tr><td>CNNIQA [57]</td><td>0.5800</td><td>0.7139</td><td>0.6470</td></tr><tr><td>DBCNN [58]</td><td>0.7491</td><td>0.8211</td><td>0.7851</td></tr><tr><td>MGQA [59]</td><td>0.6011</td><td>0.6760</td><td>0.6386</td></tr><tr><td>CLIP-IQA [60]</td><td>0.8227</td><td>0.8411</td><td>0.8319</td></tr><tr><td>StairIQA [61]</td><td>0.5504</td><td>0.6088</td><td>0.5796</td></tr><tr><td>PSCR [62]</td><td>0.8430</td><td>0.8403</td><td>0.8417</td></tr><tr><td>TIER [7]</td><td>0.8266</td><td>0.8297</td><td>0.8282</td></tr><tr><td>IP-IQA [34]</td><td>0.8401</td><td>0.8922</td><td>0.8662</td></tr><tr><td>IPCE [8]</td><td>0.8535</td><td>0.8792</td><td>0.8664</td></tr><tr><td>ACL-IQA (Ours)</td><td>0.8617</td><td>0.8885</td><td>0.8751</td></tr></table>

Adam optimizer with an initial learning rate of $5 \times 1 0 ^ { - 6 }$ and a weight decay of $1 \times 1 0 ^ { - 3 }$ . In the dual gating network, we select the top-3 expert outputs for learning, employing the Noisy Top-K Gating strategy [46] during MoE network training. We set $\lambda _ { 1 } = \lambda _ { 2 } = 0 . 5$ and $\lambda _ { 3 } = 1 . 0$ in Eqn. (13) and Eqn. (15). The Spearman Rank Correlation Coefficient (SRCC) and the Pearson Linear Correlation Coefficient (PLCC) are adopted as our evaluation metrics. To guarantee reproducibility and ensure fairness in comparisons, we strictly follow standardized data-splitting protocols: for AGIQA-20K, we directly utilize the train/test split provided by the dataset repository; for EvalMi-50K, we follow the 5-fold cross-validation protocol established in the LMM4LMM [18]; and for all other datasets, the experiments are repeated 10 times under identical data splits, with the average results reported.

## B. Experimental Results

We denote our Adversarial and Collaborative Learning based IQA model as ACL-IQA.

1) Prediction Ability Comparison: To evaluate the effectiveness of our model, we conduct comprehensive comparisons with existing IQA and AIGIQA methods across five benchmarks. The results on early-stage datasets, summarized in Tables II to IV, demonstrate that our approach consistently achieves highly competitive or state-of-the-art performance. On the AGIQA-1K dataset, the integration of a dual-gated mixture-of-experts (MoE) module and a semantic-guided gating mechanism enables our model to effectively capture textrelevant quality cues, outperforming previous methods with clear gains in correlation with human ratings. For AGIQA-3K, our method exhibits superior performance in perception quality by surpassing MA-AGIQA, and remains highly competitive in alignment quality, trailing only LMM4LMM, reflecting its robustness in modeling complex cross-modal dependencies. Moreover, on the challenging AIGCIQA2023 benchmark, our model delivers the best results in Perception and Authenticity, and secures the second-best performance in Alignment, showing strong generalization and balanced improvement across diverse quality aspects.

To thoroughly demonstrate the scalability of our model, we extend our evaluation to massive contemporary datasets, specifically AGIQA-20K and EvalMi-50K. Faced with extreme prompt complexity and diverse generative artifacts,

Table V shows that ACL-IQA establishes a new state-of-the-art on the large-scale AGIQA-20K dataset, outperforming strong multimodal evaluators like MA-AGIQA and Q-Align. Similarly, on EvalMi-50K as detailed in Table VI, our framework consistently surpasses recent strong vision-language models such as FGA-BLIP2, Qwen2.5-VL (8B), and Llama3.2-Vision (11B) across both perception and alignment dimensions. Its ability to deliver such robust evaluations without relying on massive parameter scales highlights its core architectural superiority.

These results confirm the effectiveness of our approach in capturing both perceptual and alignment quality. By leveraging the global modeling capability of the encoder alongside local feature extraction from convolutional experts, our model effectively balances coarse- and fine-grained feature representations. Furthermore, the proposed adversarial and collaborative learning mechanism successfully disentangles perception and alignment cues while maintaining their beneficial interplay, leading to a more human-consistent and robust evaluation framework.

2) Generalization Ability Comparison: We further conduct cross-dataset evaluations to assess the generalization capability of ACL-IQA. First, we evaluate the transferability between early-stage datasets such as AGIQA-3K and AIGCIQA2023. As shown in Table VII, ACL-IQA achieves the strongest crossdataset performance among all compared methods. When transferring from AIGCIQA2023 to AGIQA-3K, ACL-IQA obtains SRCCs of 0.761 (perception) and 0.676 (alignment), corresponding to relative improvements of 9.5% and 14.4% over the competitive baseline, IPCE. In the reverse direction from AGIQA-3K to AIGCIQA2023, it reaches 0.760 and 0.632, outperforming competitors by 5.4% and 14.3%, respectively. These results confirm the robustness and consistency of ACL-IQA across datasets with differing content and distortion characteristics. Notably, even compared with models adopting similar preprocessing pipelines and transformer backbones (e.g., IPCE), ACL-IQA maintains a clear performance margin, indicating that its advantages arise from the proposed learning paradigm rather than architectural or pretraining heuristics.

To further validate whether our framework captures generalizable interaction patterns beyond the priors of early-stage generative models, we conduct a challenging domain-shift evaluation. We train our models on early-stage datasets and evaluate them directly on the EvalMi-50K benchmark. As summarized in Table VIII, the significant domain shift from early-stage generators to the modern and diverse generative priors in EvalMi-50K severely impacts the performance of existing methods. However, ACL-IQA demonstrates vastly superior robustness in this setting. In the transfer from AGIQA-3K to EvalMi-50K, our framework significantly elevates perceptual performance compared to MA-AGIQA by raising the SRCC from 0.2914 to 0.6057. Similarly, it consistently outperforms IPCE across both perception and alignment dimensions in these challenging scenarios, indicating that our dual-path mechanism effectively captures intrinsic quality dynamics and successfully scales up to advanced generative systems.

TABLE III  
PERFORMANCE COMPARISON OF PERCEPTION AND ALIGNMENT QUALITY PREDICTION ON THE AGIQA-3K DATASET. THE BEST TWO RESULTS IN EACHCOLUMN ARE HIGHLIGHTED IN BOLDFACE. \* DENOTES METHODS EVALUATED BY DIRECTLY LOADING PRETRAINED MODEL WEIGHTS FOR INFERENCE.
<table><tr><td colspan="4">Perception Quality</td><td colspan="4">Alignment Quality</td></tr><tr><td>Method</td><td>SRCC</td><td>PLCC</td><td>Mean Score</td><td>Method</td><td>SRCC</td><td>PLCC</td><td>Mean Score</td></tr><tr><td>CNNIQA [57]</td><td>0.7478</td><td>0.8469</td><td>0.7974</td><td>CLIP [9]</td><td>0.5972</td><td>0.6839</td><td>0.6406</td></tr><tr><td>DBCNN [58]</td><td>0.8207</td><td>0.8759</td><td>0.8483</td><td>ImageReward [63]</td><td>0.7298</td><td>0.7862</td><td>0.7580</td></tr><tr><td>CLIP-IQA [60]</td><td>0.8426</td><td>0.8053</td><td>0.8240</td><td>HPS [64]</td><td>0.6623</td><td>0.7008</td><td>0.6816</td></tr><tr><td>ManIQA [31]</td><td>0.8767</td><td>0.9105</td><td>0.8936</td><td>ManIQA [31]</td><td></td><td></td><td></td></tr><tr><td>Re-IQA [32]</td><td>0.8424</td><td>0.8890</td><td>0.8657</td><td>Re-IQA [32]</td><td></td><td></td><td></td></tr><tr><td>TReS [30]</td><td>0.8297</td><td>0.8832</td><td>0.8565</td><td>TReS [30]</td><td></td><td></td><td></td></tr><tr><td>PSCR [62]</td><td>0.8498</td><td>0.9059</td><td>0.8779</td><td>PickScore [65]</td><td>0.7320</td><td>0.7791</td><td>0.7556</td></tr><tr><td>CLIP-AGIQA [66]</td><td>0.8618</td><td>0.8978</td><td>0.8798</td><td>StairReward [14]</td><td>0.7472</td><td>0.8529</td><td>0.8001</td></tr><tr><td>LIQE [33]</td><td>0.8742</td><td>0.8934</td><td>0.8838</td><td>LIQE [33]</td><td>0.7338</td><td>0.8205</td><td>0.7772</td></tr><tr><td>TIER [7]</td><td>0.8251</td><td>0.8821</td><td>0.8536</td><td>TIER [7]</td><td>0.6495</td><td>0.7988</td><td>0.7242</td></tr><tr><td>IP-IQA [34]</td><td>0.8634</td><td>0.9116</td><td>0.8875</td><td>IP-IQA [34]</td><td>0.7578</td><td>0.8544</td><td>0.8061</td></tr><tr><td>IPCE [8]</td><td>0.8841</td><td>0.9246</td><td>0.9044</td><td>IPCE [8]</td><td>0.7697</td><td>0.8725</td><td>0.8211</td></tr><tr><td>Q-Align [41]</td><td>0.8058</td><td>0.8268</td><td>0.8163</td><td>Q-Align [41]</td><td>0.7820</td><td>0.8477</td><td>0.8149</td></tr><tr><td>MA-AGIQA [42]</td><td>0.8872</td><td>0.9126</td><td>0.8999</td><td>MA-AGIQA [42]</td><td></td><td></td><td></td></tr><tr><td>FGA-BLIP2 [19]</td><td>0.8659</td><td>0.9121</td><td>0.8890</td><td>FGA-BLIP2 [19]</td><td>0.7089</td><td>0.8209</td><td>0.7649</td></tr><tr><td>RALI* [38]</td><td>0.7149</td><td>0.7785</td><td>0.7467</td><td>RALI* [38]</td><td></td><td></td><td></td></tr><tr><td>Q-Insight* [44]</td><td>0.7649</td><td>0.8086</td><td>0.7868</td><td>Q-Insight* [44]</td><td>0.7250</td><td>0.7888</td><td>0.7569</td></tr><tr><td>DeQA-Score* [43]</td><td>0.7544 0.8740</td><td>0.8066 0.9161</td><td>0.7805 0.8951</td><td>DeQA-Score* [43]</td><td>0.4718</td><td>0.5818</td><td>0.5268</td></tr><tr><td>LMM4LMM [18]</td><td></td><td></td><td></td><td>LMM4LMM [18]</td><td>0.8298</td><td>0.8976</td><td>0.8637</td></tr><tr><td>ACL-IQA (Ours)</td><td>0.8969</td><td>0.9282</td><td>0.9126</td><td>ACL-IQA (Ours)</td><td>0.7832</td><td>0.8800</td><td>0.8316</td></tr></table>

TABLE IV

PERFORMANCE COMPARISON OF PERCEPTION, AUTHENTICITY, AND ALIGNMENT QUALITY PREDICTION ON THE AIGCIQA2023 DATASET. THE BESTTWO RESULTS IN EACH COLUMN ARE HIGHLIGHTED IN BOLDFACE. \* DENOTES METHODS EVALUATED BY DIRECTLY LOADING PRETRAINED MODELWEIGHTS FOR INFERENCE.
<table><tr><td rowspan="2">Method</td><td colspan="3">Perception</td><td colspan="3">Authenticity</td><td colspan="3">Alignment</td></tr><tr><td>SRCC</td><td>PLCC</td><td>Mean Score</td><td>SRCC</td><td>PLCC</td><td>Mean Score</td><td>SRCC</td><td>PLCC</td><td>Mean Score</td></tr><tr><td>CNNIQA [57]</td><td>0.7160</td><td>0.7937</td><td>0.7549</td><td>0.5958</td><td>0.5734</td><td>0.5846</td><td>0.4758</td><td>0.4937</td><td>0.4848</td></tr><tr><td>ResNet34 [67]</td><td>0.7229</td><td>0.7578</td><td>0.7404</td><td>0.5998</td><td>0.6285</td><td>0.6142</td><td>0.7058</td><td>0.7153</td><td>0.7106</td></tr><tr><td>TReS [30]</td><td>0.8064</td><td>0.8347</td><td>0.8206</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ManIQA [31]</td><td>0.8269</td><td>0.8464</td><td>0.8367</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Re-IQA [32]</td><td>0.8372</td><td>0.8721</td><td>0.8547</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LIQE [33]</td><td>0.8534</td><td>0.8758</td><td>0.8646</td><td></td><td></td><td></td><td>0.7780</td><td>0.7696</td><td>0.7738</td></tr><tr><td>CLIP-IQA [60]</td><td>0.7802</td><td>0.8140</td><td>0.7971</td><td>0.6726</td><td>0.6627</td><td>0.6677</td><td>0.5799</td><td>0.5702</td><td>0.5751</td></tr><tr><td>PSCR [62]</td><td>0.8371</td><td>0.8588</td><td>0.8480</td><td>0.7828</td><td>0.7750</td><td>0.7789</td><td>0.7465</td><td>0.7397</td><td>0.7431</td></tr><tr><td>IPCE [8]</td><td>0.8640</td><td>0.8788</td><td>0.8714</td><td>0.8097</td><td>0.7998</td><td>0.8048</td><td>0.7979</td><td>0.7887</td><td>0.7933</td></tr><tr><td>CLIP-AGIQA [66]</td><td>0.8140</td><td>0.8302</td><td>0.8221</td><td>0.7940</td><td>0.7797</td><td>0.7869</td><td>0.7036</td><td>0.6994</td><td>0.7015</td></tr><tr><td>Q-Align [41]</td><td>0.7975</td><td>0.8300</td><td>0.8138</td><td></td><td></td><td></td><td>0.7398</td><td>0.7157</td><td>0.7278</td></tr><tr><td>MA-AGIQA [42]</td><td>0.8384</td><td>0.8606</td><td>0.8495</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TIER [7]</td><td>0.8326</td><td>0.8526</td><td>0.8426</td><td>0.7485</td><td>0.7403</td><td>0.7444</td><td>0.7210</td><td>0.7148</td><td>0.7179</td></tr><tr><td>FGA-BLIP2 [19]</td><td>0.8326</td><td>0.8691</td><td>0.8509</td><td></td><td></td><td></td><td>0.7820</td><td>0.7765</td><td>0.7793</td></tr><tr><td>RALI* [38]</td><td>0.7772</td><td>0.7964</td><td>0.7868</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Q-Insight* [44]</td><td>0.8027</td><td>0.7975</td><td>0.8001</td><td></td><td></td><td></td><td>0.6504</td><td>0.6269</td><td>0.6387</td></tr><tr><td>DeQA-Score* [43]</td><td>0.8024</td><td>0.8304</td><td>0.8164</td><td></td><td></td><td></td><td>0.4232</td><td>0.4503</td><td>0.4368</td></tr><tr><td>LMM4LMM [18]</td><td>0.8344</td><td>0.8632</td><td>0.8488</td><td></td><td></td><td></td><td>0.8208</td><td>0.8141</td><td>0.8175</td></tr><tr><td>ACL-IQA (Ours)</td><td>0.8720</td><td>0.8865</td><td>0.8793</td><td>0.8180</td><td>0.8083</td><td>0.8132</td><td>0.8092</td><td>0.8005</td><td>0.8049</td></tr></table>

## C. Ablation Studies

The main components proposed in our method include the MoE mechanism, the instruction prompt for dual-path learning, and the Adversarial-Collaborative Mechanism. To assess the effectiveness of each component, we conduct the ablation studies based on five variants of our method, as shown in Table IX. In Exp. 1, we ablate all proposed mechanisms, using only the image encoder and text encoder for quality level classification and quality regression. The results show a significant performance drop, particularly in alignment quality on the AGIQA-3K dataset and perception quality assessment on the AIGCIQA2023 dataset. This highlights the necessity of incorporating the Adversarial-Collaborative Mechanism and MoE for robust quality evaluation. In Exp. 2, we retain the dual-path (Adversarial-Collaborative) learning mechanism but replace the MoE with standard transformer blocks. To maintain dual-path learning, a mapping network is introduced to project image features into two distinct feature representations. The results indicate noticeable improvements over Exp. 1, confirming the effectiveness of the Adversarial-Collaborative Mechanism. However, the performance remains suboptimal compared to the full model, suggesting that MoE also plays a crucial role in refining feature extraction and enhancing

![](images/ff26a6d5f04860696842e90f84dcd8ea0dd93c580701f609a316ebce7ba58c8e.jpg)

Fig. 5. Attention visualization of features from the MoE layer. The first and second rows correspond to the perceptual quality and alignment prediction tasks, respectively. “Adversarial” and “Collaborative” indicate that the features are learned from the two distinct interaction paths.  
![](images/52e01607e558cfabe9db862d65e52288c1eeab114f01d268906df09213816da0.jpg)

![](images/1b07ecc15aa367423231f6809df282c694e833618e324295fabea756a785e3e3.jpg)

![](images/ade286de0668cc6498ce1d5140b656c1804b25b1116a2e42212f9701bd2e91fe.jpg)

![](images/00cd0fe25d5d1c068dd57260334cdfe616924bd4e3bff71b8fd4143e732d01d5.jpg)

![](images/f44eeec276f61b9a6cd447d102ee8fa388691dc400158beea58aeee23b16f7e6.jpg)

![](images/65404b17ebcc4337cc947ea6e6b08ebc48a7a80b523d66c21b7421f6f822d13f.jpg)

![](images/ea8673e1c6433b57446e716c0ebda78a914c5e998115e8cf85f7d60976315d5a.jpg)

![](images/d662149c4a1c58620f56ea71313a60e1b05fa1a6588d92eb8ef4b3c0434fad41.jpg)  
Fig. 6. Performance comparison under extreme scenarios, collaborative scenarios (left) where perception quality and alignment quality are positively correlated, and adversarial scenarios (right) where they conflict. Each row displays five generated images alongside their respective prompts. The bottom charts present quantitative comparisons across multiple image sets with perception and alignment quality scores.

## TABLE V

PERFORMANCE COMPARISON ON THE AGIQA-20K DATASET. THE BESTTWO RESULTS IN EACH COLUMN ARE HIGHLIGHTED IN BOLDFACE. \*DENOTES METHODS EVALUATED BY DIRECTLY LOADING PRETRAINEDMODEL WEIGHTS FOR INFERENCE.
<table><tr><td>Method</td><td>SRCC</td><td>PLCC</td><td>Mean Score</td></tr><tr><td>CNNIQA [57]</td><td>0.5968</td><td>0.5913</td><td>0.5941</td></tr><tr><td>DBCNN [58]</td><td>0.8506</td><td>0.8688</td><td>0.8597</td></tr><tr><td>HyperIQA [2]</td><td>0.8160</td><td>0.8320</td><td>0.8240</td></tr><tr><td>MUSIQ [68]</td><td>0.8320</td><td>0.8640</td><td>0.8480</td></tr><tr><td>ManIQA [31]</td><td>0.8500</td><td>0.8870</td><td>0.8685</td></tr><tr><td>StairIQA [61]</td><td>0.7890</td><td>0.8420</td><td>0.8155</td></tr><tr><td>LIQE [33]</td><td>0.8620</td><td>0.7980</td><td>0.8300</td></tr><tr><td>Q-Align [41]</td><td>0.8740</td><td>0.8890</td><td>0.8815</td></tr><tr><td>MA-AGIQA [42]</td><td>0.8640</td><td>0.9050</td><td>0.8845</td></tr><tr><td>DeQA-Score* [43]</td><td>0.4779</td><td>0.5535</td><td>0.5157</td></tr><tr><td>Q-Insight* [44]</td><td>0.5359</td><td>0.6172</td><td>0.5766</td></tr><tr><td>LMM4LMM* [18]</td><td>0.7284</td><td>0.6638</td><td>0.6961</td></tr><tr><td>ACL-IQA (Ours)</td><td>0.8858</td><td>0.9113</td><td>0.8986</td></tr></table>

the adversarial-collaborative process. In Exp. 3, we examine the contribution of the learning instruction by ablating it from our gating network. A performance drop is observed in the perception assessment on the AGIQA-3K dataset and the alignment assessment on the AIGCIQA2023 dataset, as measured by SRCC. This decline may be attributed to the instruction’s role in facilitating fine-grained feature selection for adversarial and cooperative strategies. In Exp. 4 and 5, we evaluate the independent contributions of the adversarial and collaborative mechanisms by applying each separately. The results reveal that while both mechanisms contribute to performance gains, neither outperforms their combination. For instance, the collaborative mechanism shows superior performance in perception quality assessment on AGIQA-3K and alignment quality on AIGCIQA2023, but underperforms in other aspects. This suggests that adversarial and collaborative strategies capture complementary quality representations. Our full model achieves the best performance across all datasets, confirming that each proposed component benefits our model.

## D. Visualization

To further demonstrate the effectiveness of our model, visualizations of representative feature activations and qualitative results are presented as follows.

TABLE VI  
PERFORMANCE COMPARISON ON THE EVALMI-50K DATASET ACROSS PERCEPTION AND ALIGNMENT DIMENSIONS. THE BEST TWO RESULTS IN EACH COLUMN ARE HIGHLIGHTED IN BOLDFACE.
<table><tr><td rowspan="3">Method</td><td colspan="3">Perception Quality</td><td colspan="3">Alignment Quality</td></tr><tr><td>SRCC</td><td>PLCC</td><td>Mean Score</td><td>SRCC</td><td>PLCC</td><td>Mean Score</td></tr><tr><td>CNNIQA [57]</td><td>0.4348</td><td>0.5583</td><td>0.4966</td><td>0.1186</td><td>0.0791</td><td>0.0989</td></tr><tr><td>DBCNN [58]</td><td>0.5525</td><td>0.6181</td><td>0.5853</td><td>0.3301</td><td>0.3515</td><td>0.3408</td></tr><tr><td>HyperIQA [2]</td><td>0.5872</td><td>0.6768</td><td>0.6320</td><td>0.5348</td><td>0.5447</td><td>0.5398</td></tr><tr><td>TReS [30]</td><td>0.3935</td><td>0.4301</td><td>0.4118</td><td>0.1406</td><td>0.1520</td><td>0.1463</td></tr><tr><td>MUSIQ [68]</td><td>0.7985</td><td>0.8379</td><td>0.8182</td><td>0.5310</td><td>0.5510</td><td>0.5410</td></tr><tr><td>StairIQA [61]</td><td>0.8268</td><td>0.8645</td><td>0.8457</td><td>0.5890</td><td>0.6089</td><td>0.5990</td></tr><tr><td>LIQE [33]</td><td>0.8106</td><td>0.8268</td><td>0.8187</td><td>0.5617</td><td>0.5777</td><td>0.5697</td></tr><tr><td>Q-Align [41]</td><td>0.8311</td><td>0.8505</td><td>0.8408</td><td>0.4754</td><td>0.4918</td><td>0.4836</td></tr><tr><td>DeepSeekVL2(1B) [69]</td><td>0.7899</td><td>0.8253</td><td>0.8076</td><td>0.7817</td><td>0.7991</td><td>0.7904</td></tr><tr><td>Qwen2.5-VL(8B) [70]</td><td>0.6990</td><td>0.7495</td><td>0.7243</td><td>0.8008</td><td>0.8219</td><td>0.8114</td></tr><tr><td>Llama3.2-Vision(11B) [71]</td><td>0.7555</td><td>0.7891</td><td>0.7723</td><td>0.6403</td><td>0.6461</td><td>0.6432</td></tr><tr><td>FGA-BLIP2 [19]</td><td>0.8715</td><td>0.8969</td><td>0.8842</td><td>0.8121</td><td>0.8309</td><td>0.8215</td></tr><tr><td>ACL-IQA (Ours)</td><td>0.8810</td><td>0.9050</td><td>0.8930</td><td>0.8664</td><td>0.8795</td><td>0.8730</td></tr></table>

TABLE VII

CROSS-DATASET PREDICTION PERFORMANCE COMPARISON. THE BEST TWO RESULTS IN EACH COLUMN ARE HIGHLIGHTED IN BOLDFACE.
<table><tr><td rowspan="2">Methods</td><td colspan="3">AIGCIQA2023Train → AGIQA-3KTest</td><td colspan="3">AGIQA-3KTrain → AIGCIQA2023Test</td></tr><tr><td>Perception</td><td>Alignment</td><td>Mean Score</td><td>Perception</td><td>Alignment</td><td>Mean Score</td></tr><tr><td>ResNet50 [67]</td><td>0.576</td><td>0.473</td><td>0.525</td><td>0.599</td><td>0.432</td><td>0.516</td></tr><tr><td>ViT-B/32 [52]</td><td>0.509</td><td>0.434</td><td>0.472</td><td>0.517</td><td>0.381</td><td>0.449</td></tr><tr><td>DB-CNN [58]</td><td>0.627</td><td>0.390</td><td>0.509</td><td>0.654</td><td>0.470</td><td>0.562</td></tr><tr><td>HyperIQA [2]</td><td>0.657</td><td>0.418</td><td>0.538</td><td>0.669</td><td>0.464</td><td>0.567</td></tr><tr><td>TIER [7]</td><td>0.611</td><td>0.417</td><td>0.514</td><td>0.675</td><td>0.441</td><td>0.558</td></tr><tr><td>IPCE [8]</td><td>0.695</td><td>0.591</td><td>0.643</td><td>0.721</td><td>0.553</td><td>0.637</td></tr><tr><td>ACL-IQA (Ours)</td><td>0.761</td><td>0.676</td><td>0.718</td><td>0.760</td><td>0.632</td><td>0.696</td></tr></table>

TABLE VIII

CROSS-DATASET PERFORMANCE COMPARISON WHEN TRANSFERRING FROM EARLY DATASETS TO THE EVALMI-50K. THE BEST TWO RESULTS IN EACH COLUMN ARE HIGHLIGHTED IN BOLDFACE.
<table><tr><td rowspan="2">Method</td><td colspan="3">AGIQA-3KTrain → EvalMi-50KTest</td><td colspan="3">AIGCIQA2023Train → EvalMi-50KTest</td></tr><tr><td>Perception</td><td>Alignment</td><td>Mean Score</td><td>Perception</td><td>Alignment</td><td>Mean Score</td></tr><tr><td>ManIQA [31]</td><td>0.0619</td><td></td><td>0.0619</td><td>0.1056</td><td></td><td>0.1056</td></tr><tr><td>Re-IQA [32]</td><td>0.1791</td><td></td><td>0.1791</td><td>0.1528</td><td></td><td>0.1528</td></tr><tr><td>TReS [30]</td><td>-0.0677</td><td></td><td>-0.0677</td><td>-0.1296</td><td></td><td>-0.1296</td></tr><tr><td>MA-AGIQA [42]</td><td>0.2914</td><td></td><td>0.2914</td><td>0.4438</td><td></td><td>0.4438</td></tr><tr><td>LIQE [33]</td><td>0.1823</td><td>0.1857</td><td>0.1840</td><td>0.1805</td><td>0.1568</td><td>0.1687</td></tr><tr><td>IPCE [8]</td><td>0.5064</td><td>0.3247</td><td>0.4156</td><td>0.5459</td><td>0.4041</td><td>0.4750</td></tr><tr><td>ACL-IQA (Ours)</td><td>0.6057</td><td>0.4449</td><td>0.5253</td><td>0.6567</td><td>0.4579</td><td>0.5573</td></tr></table>

Path-Specific Attention Feature Maps. As shown in Fig. 5, our MoE framework captures diverse interaction cues, enhances shared-attention cues, and ablates task-irrelevant attention in collaborative and adversarial learning paths, respectively. Taking perception quality prediction as an example, the collaborative path attends to regions highly correlated with alignment (e.g., semantically important entities), whereas the adversarial path focuses exclusively on perception-related cues such as sharpness, noise, and artifacts. Moreover, our MoE layers dynamically select the top-k differential convolutional experts for each path, enabling more targeted feature extraction based on the learning objective.

Performance Under Extreme Scenarios. To demonstrate the robustness of our method, we evaluate its performance under two extreme scenarios: (1) image sets where perception scores are fully correlated with alignment scores, and (2) image sets where perception scores are completely adversarial to alignment scores. The results, shown in Fig. 6, indicate that our method maintains strong consistency with human ratings and outperforms the second-best method, IPCE, in both extreme scenarios. This highlights our method’s superior robustness, as it effectively accounts for both types of interactions observed in human ratings through dual-path learning.

Results of gMAD Competition. To further assess the behavior of different models beyond quantitative metrics, we adopt the generalized Maximum Differentiation (gMAD) [72]

Best IPCE (MOS: 2.14) Fixed ACLIQA

TABLE IX  
ABLATION STUDY RESULTS ON THE AGIQA-3K AND AIGCIQA2023 DATASETS. “DUAL LS” DENOTES THE DUAL (ADVERSARIAL AND COLLABORATIVE) LEARNING TEXT/INSTRUCTIONS. “ADV.” AND “COL.” REFER TO THE ADVERSARIAL AND COLLABORATIVE LEARNING PATHS, RESPECTIVELY.
<table><tr><td rowspan="2">Exp. ID</td><td rowspan="2">MoE</td><td rowspan="2">Dual LS</td><td rowspan="2">Adv.</td><td rowspan="2">Col.</td><td colspan="4">AGIQA-3K</td><td rowspan="2"></td><td colspan="3">AIGCIQA2023</td></tr><tr><td colspan="2">Perception</td><td colspan="2">Alignment</td><td colspan="2">Perception</td><td colspan="2">Alignment</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td></td><td>PLCC</td><td>SRCC</td><td>PLCC</td><td>SRCC</td><td>PLCC</td></tr><tr><td>1 2</td><td>x</td><td>x</td><td>x</td><td></td><td>x</td><td>0.8841</td><td>0.9246</td><td>0.7697</td><td>0.8725</td><td>0.8640</td><td>0.8788</td><td>0.7979</td><td>0.7887</td></tr><tr><td>3</td><td>x √</td><td>x x</td><td>√ √</td><td>√ √</td><td></td><td>0.8866 0.8895</td><td>0.9253 0.9270</td><td>0.7766 0.7787</td><td>0.8771 0.8780</td><td>0.8677 0.8687</td><td>0.8835 0.8839</td><td>0.8032 0.8043</td><td>0.7939 0.7950</td></tr><tr><td>4</td><td>√</td><td>√</td><td>√</td><td>x</td><td></td><td>0.8680</td><td>0.8682</td><td>0.7660</td><td>0.8675</td><td>0.8622</td><td>0.8787</td><td>0.7648</td><td>0.7545</td></tr><tr><td>5</td><td>√</td><td>√</td><td>x</td><td>√</td><td></td><td>0.8606</td><td>0.9014</td><td>0.7642</td><td>0.8673</td><td>0.8597</td><td>0.8700</td><td>0.7701</td><td>0.7641</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>0.8903</td><td>0.9282</td><td>0.7832</td><td>0.8800</td><td>0.8720</td><td>0.8865</td><td>0.8092</td><td>0.8005</td></tr></table>

Perception Quality

![](images/b90f979aece4774a32cf1d64d27523809b640bebc0c9800c60bc4c140c543985.jpg)  
Best ACLIQA (MOS: 3.01) Fixed IPCE

![](images/e44be59a7dfc743e55b541075f71a98760d939fad828e35ae6c3e3ef95be3ccb.jpg)  
Best ACLIQA (MOS: 3.86) Fixed IPCE Worst ACLTQA (MOS: 2 31)

![](images/1e4e7985c0be293f98198dbcb2e7ae960c7878b21bbb72f9c3fa2b59c9cacf41.jpg)

![](images/57fd7305df19bc71c16c46b05d225153e75d265f50341eafbe22acb244d8d67f.jpg)  
Best IPCE (MOS: 3.03) IFixed ACLIQA

![](images/e6ab7584ebc4b1764956cc0dc9d2c12dcdd8e8a48d8b0dcf7c8cc29791e04740.jpg)

![](images/08b7f24c44023c0232e73bb7388194dba7d644f3a9edd202ddeeb08b5f4a391b.jpg)

![](images/d5cf6b61d9beef85e6c3c14953ef5f79c52432aadfea40c4e2f5164c9ff48ac2.jpg)

![](images/d5c8bf77490753d3945cec2c337f7c5f7b971f32595ea258f7f0272c671e5bdb.jpg)  
Alignment Quality

![](images/5d0df1a04367dfd27d2479689d47116454486dd6bd8dd0cc8bc697b170a65830.jpg)  
(a)

![](images/43460f726762ed02c0d267840208f10c08637d7604488f936303066c299048a1.jpg)  
(b)

![](images/b08a56b6831b683915dd5b4c951ffdde3a398c0bdf899091d3a095115ac5f75f.jpg)

![](images/233404d718dccef3a80802bbc61a72822956c6dff5def911c5b919dce656deec.jpg)  
(c)  
Best IPCE (MOS: 3.03) Fixed ACLIQA

![](images/1cee848f8ba06324a2ad2655063ce40ab8c66eecfaec6029cae82bf62b308d53.jpg)  
(d)  
Fig. 7. gMAD [72] comparison between ACL-IQA and IPCE on perception (top) and alignment (bottom) quality. (a) IPCE predicts both images as low quality, (b) IPCE predicts both images as high quality, (c) both images have low MOS, and (d) both images have high MOS. ACL-IQA aligns more consistently with human judgments across all cases.

protocol to perform a head-to-head evaluation against the state-of-the-art IPCE model. As shown in Fig. 7, both methods are compared on perception and alignment quality across four representative scenarios. In cases where IPCE assigns similarly low or high predictions to paired images, ACL-IQA is still able to correctly differentiate their quality in accordance with MOS, reflecting a higher perceptual sensitivity. Conversely, when the two images share similar MOS but IPCE produces large score discrepancies, ACL-IQA predicts closer values consistent with human judgments.

Overall, the results demonstrate that ACL-IQA achieves more reliable and human-aligned behavior in challenging edge cases, highlighting stronger robustness and discriminability compared with IPCE.

Gating Activation Comparison in Dual-path Learning. In our method, the output weights of the experts are determined by our dual-gated network for dual-path learning. To better understand this mechanism, we visualize the weight distributions for adversarial and collaborative learning on the AIGCIQA2023 dataset in Fig. 8. From the figure, we could observe a clear distinction between the expert groups selected by the adversarial and collaborative mechanisms across different assessment tasks. This demonstrates that our designed gating mechanism effectively enables the model to intelligently allocate and learn features based on different learning strategies.

![](images/039c7d4b9c92f1a41c2e9acc96cd0bfa617f2a457b5e03e2026077e3857f9ad6.jpg)

![](images/44db8af7b230c252cf516f62fe80bfbc7071cff6e8313a11b1c2762f48b711e8.jpg)  
Fig. 8. Comparison of gating activations in dual path learning. $\mathrm { \ddot { s t d v } } . \mathrm { \vec { \Omega } }$ and “Col.” denote the adversarial and collaborative learning paths, respectively.

![](images/f4796758d78bfa3cbeb345107b9be74089bbcdd226f56eed7e1ea195ff413e23.jpg)

![](images/7547c545a90f283181e5757e70d9b39838c7f53a952dfba241b5edad693e231a.jpg)  
Fig. 9. The cosine similarity between the text feature and the image features $( \breve { F _ { c o l } ^ { I } } , \ : F _ { a d v } ^ { I } )$ extracted in collaborative and adversarial learning.

Feature Purity Ensured by Adversarial Learning. In our adversarial learning path, the extracted image features are expected to encode the quality information only in one dimension. To validate the purity of these features and the effectiveness of the adversarial mechanism, we perform inference on the test sets of the AGIQA3K and AIGCIQA2023 datasets, computing the cosine similarity between the text feature and the two image features $F _ { c o l } ^ { I }$ and $F _ { a d v } ^ { I }$ . Taking the alignment quality evaluation task as an example, we first map the groundtruth alignment score to its corresponding quality level k. We then compute the adversarial similarity and collaborative similarity as the cosine similarity between the adversarial feature and the text features representing the k-th perception quality level and k-th alignment quality level, respectively. Finally, the mean similarity values are reported across all test images. Herein, it should be noted that a single quality score usually corresponds to multiple quality levels $( e . g . , \ \hat { Q } _ { p e r } { = } 3 . 5 $ may exhibit high similarity with both the third and fourth quality levels). However, we consider only the most adjacent quality level, which may lead to an underestimation of the similarity score for the collaborative feature $F _ { c o l } ^ { I } .$ . As shown in Fig. 9, the experimental results reveal that the adversarial similarity is significantly lower than the collaborative similarity. This suggests that the adversarial features learned by our model contain minimal quality information from the other dimension, effectively capturing the unique elements that differentiate the two paths.

## V. CONCLUSION

In this paper, we re-examined AIGIQA from the perspective of human judgment and demonstrated that perceptual fidelity and prompt alignment do not behave as independent attributes, but exhibit both collaborative and adversarial interactions. To model this property, we proposed a Dual-Gated Mixture-of-Experts framework that dynamically routes features through complementary reasoning paths, enabling adaptive balancing between reinforcement and disentanglement. Extensive experiments across multiple benchmarks show that our approach achieves state-of-the-art performance while delivering more interpretable and human-consistent predictions. We believe this perspective provides new insight that quality assessment models for AI-generated content should move beyond singlefactor scoring and instead explicitly account for the interaction patterns underlying human decision making.

## REFERENCES

[1] Y. Fang, H. Zhu, Y. Zeng, K. Ma, and Z. Wang, “Perceptual quality assessment of smartphone photography,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 3677–3686.

[2] S. Su, Q. Yan, Y. Zhu, C. Zhang, X. Ge, J. Sun, and Y. Zhang, “Blindly assess image quality in the wild guided by a self-adaptive hyper network,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 3667–3676.

[3] X. Min, K. Ma, K. Gu, G. Zhai, Z. Wang, and W. Lin, “Unified blind quality assessment of compressed natural, graphic, and screen content images,” IEEE Transactions on Image Processing, vol. 26, no. 11, pp. 5462–5474, 2017.

[4] B. Chen, L. Zhu, G. Li, F. Lu, H. Fan, and S. Wang, “Learning generalized spatial-temporal deep feature representation for no-reference video quality assessment,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 4, pp. 1903–1916, 2021.

[5] B. Chen, K. Xiao, X. Shen, and S. Wang, “Monotonic and invertible network: A general framework for learning iqa model from mixed datasets,” International Journal of Computer Vision, vol. 133, no. 11, pp. 7924–7945, 2025.

[6] B. Chen, D. Huang, H. Zhu, L. Zhu, W. Zhou, S. Wang, Y. Fang, and W. Lin, “From global to granular: Revealing iqa model performance via correlation surface,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[7] J. Yuan, X. Cao, J. Che, Q. Wang, S. Liang, W. Ren, J. Lin, and X. Cao, “TIER: Text-image encoder-based regression for aigc image quality assessment,” arXiv preprint arXiv:2401.03854, 2024.

[8] F. Peng, H. Fu, A. Ming, C. Wang, H. Ma, S. He, Z. Dou, and S. Chen, “AIGC image quality assessment via image-prompt correspondence,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6432–6441.

[9] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning. PmLR, 2021, pp. 8748–8763.

[10] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” Communications of the ACM, vol. 63, no. 11, pp. 139–144, 2020.

[11] M. Ding, Z. Yang, W. Hong, W. Zheng, C. Zhou, D. Yin, J. Lin, X. Zou, Z. Shao, H. Yang et al., “CogView: Mastering text-to-image generation via transformers,” Advances in Neural Information Processing Systems, vol. 34, pp. 19 822–19 835, 2021.

[12] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 684–10 695.

[13] Z. Zhang, C. Li, W. Sun, X. Liu, X. Min, and G. Zhai, “A perceptual quality assessment exploration for aigc images,” in 2023 IEEE International Conference on Multimedia and Expo Workshops (ICMEW). IEEE, 2023, pp. 440–445.

[14] C. Li, Z. Zhang, H. Wu, W. Sun, X. Min, X. Liu, G. Zhai, and W. Lin, “AGIQA-3k: An open database for ai-generated image quality assessment,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 8, pp. 6833–6846, 2023.

[15] H. Zhu, H. Wu, Y. Li, Z. Zhang, B. Chen, L. Zhu, Y. Fang, G. Zhai, W. Lin, and S. Wang, “Adaptive image quality assessment via teaching large multimodal model to compare,” Advances in Neural Information Processing Systems, vol. 37, pp. 32 611–32 629, 2025.

[16] J. Wang, H. Duan, J. Liu, S. Chen, X. Min, and G. Zhai, “AIG-CIQA2023: A large-scale image quality assessment database for ai generated images: from the perspectives of quality, authenticity and correspondence,” in CAAI International Conference on Artificial Intelligence. Springer, 2023, pp. 46–57.

[17] C. Li, T. Kou, Y. Gao, Y. Cao, W. Sun, Z. Zhang, Y. Zhou, Z. Zhang, W. Zhang, H. Wu et al., “AIGIQA-20K: A large database for aigenerated image quality assessment,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6327–6336.

[18] J. Wang, H. Duan, Y. Zhao, J. Wang, G. Zhai, and X. Min, “LMM4LMM: Benchmarking and evaluating large-multimodal image generation with lmms,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 17 312–17 323.

[19] S. Han, H. Fan, J. Fu, L. Li, T. Li, J. Cui, Y. Wang, Y. Tai, J. Sun, C. Guo et al., “EvalMuse-40K: A reliable and fine-grained benchmark with comprehensive human annotations for text-to-image generation model evaluation,” arXiv preprint arXiv:2412.18150, 2024.

[20] B. Li, Z. Lin, D. Pathak, J. Li, Y. Fei, K. Wu, T. Ling, X. Xia, P. Zhang, G. Neubig et al., “GenAI-Bench: Evaluating and improving compositional text-to-visual generation,” arXiv preprint arXiv:2406.13743, 2024.

[21] K. Huang, K. Sun, E. Xie, Z. Li, and X. Liu, “T2I-CompBench: A comprehensive benchmark for open-world compositional text-toimage generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 78 723–78 747, 2023.

[22] D. Ghosh, H. Hajishirzi, and L. Schmidt, “GenEval: An object-focused framework for evaluating text-to-image alignment,” Advances in Neural Information Processing Systems, vol. 36, pp. 52 132–52 152, 2023.

[23] Y. Liang, J. He, G. Li, P. Li, A. Klimovskiy, N. Carolan, J. Sun, J. Pont-Tuset, S. Young, F. Yang et al., “Rich human feedback for text-to-image generation,” in CVPR, 2024.

[24] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training gans,” Advances in Neural Information Processing Systems, vol. 29, 2016.

[25] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[26] H. Zhu, L. Li, J. Wu, W. Dong, and G. Shi, “MetaIQA: Deep metalearning for no-reference image quality assessment,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 14 143–14 152.

[27] S. Sun, T. Yu, J. Xu, W. Zhou, and Z. Chen, “GraphIQA: Learning distortion graph representations for blind image quality assessment,” IEEE Transactions on Multimedia, vol. 25, pp. 2912–2925, 2022.

[28] X. Fang, W. Wang, X. Lv, and J. Yan, “PCQA: A strong baseline for AIGC quality assessment based on prompt condition,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6167–6176.

[29] J. Fu, W. Zhou, Q. Jiang, H. Liu, and G. Zhai, “Vision-language consistency guided multi-modal prompt learning for blind AI generated image quality assessment,” IEEE Signal Processing Letters, 2024.

[30] S. A. Golestaneh, S. Dadsetan, and K. M. Kitani, “No-reference image quality assessment via transformers, relative ranking, and selfconsistency,” in Proceedings of the IEEE/CVF winter Conference on Applications of Computer Vision, 2022, pp. 1220–1230.

[31] S. Yang, T. Wu, S. Shi, S. Lao, Y. Gong, M. Cao, J. Wang, and Y. Yang, “MANIQA: Multi-dimension attention network for no-reference image quality assessment,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 1191–1200.

[32] A. Saha, S. Mishra, and A. C. Bovik, “Re-IQA: Unsupervised learning for image quality assessment in the wild,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 5846–5855.

[33] W. Zhang, G. Zhai, Y. Wei, X. Yang, and K. Ma, “Blind image quality assessment via vision-language correspondence: A multitask learning perspective,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 071–14 081.

[34] B. Qu, H. Li, and W. Gao, “Bringing textual prompt to AI-generated image quality assessment,” in IEEE International Conference on Multimedia and Expo, 2024, pp. 1–6.

[35] Z. Yu, F. Guan, Y. Lu, X. Li, and Z. Chen, “SF-IQA: Quality and similarity integration for ai generated image quality assessment,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6692–6701.

[36] J. Xia, L. He, F. Gao, K. Zhang, L. Li, and X. Gao, “AI-generated image quality assessment based on task-specific prompt and multi-granularity similarity,” arXiv preprint arXiv:2411.16087, 2024.

[37] Z. Meng, “Decoupling Semantics from Distortions: Multi-Scale Two-Stream Vision-Language Alignment for AI-Generated Image Quality Assessment,” arXiv preprint arXiv:2606.16799, 2026.

[38] S. Zhao, X. Zhang, W. Li, J. Li, L. Zhang, T. Xue, and J. Zhang, “Reasoning as representation: Rethinking visual reinforcement learning in image quality assessment,” arXiv preprint arXiv:2510.11369, 2025.

[39] Z. Liao, B. Chen, H. Zhu, L. Zhu, S. Wang, and W. Lin, “Plug In, Grade Right: Psychology-Inspired AGIQA,” arXiv preprint arXiv:2512.22780, 2025.

[40] Q. Li, Q. Yan, H. Huang, P. Wu, H. Zhang, and Y. Zhang, “Textvisual semantic constrained AI-generated image quality assessment,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 6958–6966.

[41] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, Y. Gao, A. Wang, E. Zhang, W. Sun et al., “Q-Align: Teaching lmms for visual scoring via discrete text-defined levels,” in Proceedings of the 41st International Conference on Machine Learning (ICML), vol. 235, 2024, pp. 54 015– 54 029.

[42] P. Wang, W. Sun, Z. Zhang, J. Jia, Y. Jiang, Z. Zhang, X. Min, and G. Zhai, “Large multi-modality model assisted ai-generated image quality assessment,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 7803–7812.

[43] Z. You, X. Cai, J. Gu, T. Xue, and C. Dong, “Teaching large language models to regress accurate image quality scores using score distribution,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 14 483–14 494.

[44] W. Li, X. Zhang, S. Zhao, Y. Zhang, J. Li, J. Zhang et al., “Q-Insight: Understanding image quality via visual reinforcement learning,” Advances in Neural Information Processing Systems, vol. 38, pp. 36 802– 36 827, 2026.

[45] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive mixtures of local experts,” Neural computation, vol. 3, no. 1, pp. 79–87, 1991.

[46] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” arXiv preprint arXiv:1701.06538, 2017.

[47] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[48] S. Kudugunta, Y. Huang, A. Bapna, M. Krikun, D. Lepikhin, M.-T. Luong, and O. Firat, “Beyond distillation: Task-level mixture-of-experts for efficient inference,” arXiv preprint arXiv:2110.03742, 2021.

[49] J. Ma, Z. Zhao, X. Yi, J. Chen, L. Hong, and E. H. Chi, “Modeling task relationships in multi-task learning with multi-gate mixture-ofexperts,” in Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery & Data mining, 2018, pp. 1930–1939.

[50] R. Aoki, F. Tung, and G. L. Oliveira, “Heterogeneous multi-task learning with expert diversity,” IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 19, no. 6, pp. 3093–3102, 2022.

[51] Z. Fan, R. Sarkar, Z. Jiang, T. Chen, K. Zou, Y. Cheng, C. Hao, Z. Wang et al., “M<sup>3</sup>Vit: Mixture-of-experts vision transformer for efficient multitask learning with model-accelerator co-design,” Advances in Neural Information Processing Systems, vol. 35, pp. 28 441–28 457, 2022.

[52] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[53] Z. Su, W. Liu, Z. Yu, D. Hu, Q. Liao, Q. Tian, M. Pietikainen, and L. Liu,¨ “Pixel difference networks for efficient edge detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5117–5127.

[54] C. Kong, A. Luo, P. Bao, Y. Yu, H. Li, Z. Zheng, S. Wang, and A. C. Kot, “MoE-FFD: Mixture of experts for generalized and parameter-efficient face forgery detection,” arXiv preprint arXiv:2404.08452, 2024.

[55] N. Park and S. Kim, “How do vision transformers work?” in International Conference on Learning Representations, 2022.

[56] Y. Ganin and V. Lempitsky, “Unsupervised domain adaptation by backpropagation,” in International Conference on Machine Learning. PMLR, 2015, pp. 1180–1189.

[57] L. Kang, P. Ye, Y. Li, and D. Doermann, “Convolutional neural networks for no-reference image quality assessment,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 1733–1740.

[58] W. Zhang, K. Ma, J. Yan, D. Deng, and Z. Wang, “Blind image quality assessment using a deep bilinear convolutional neural network,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 1, pp. 36–47, 2018.

[59] T. Wang, W. Sun, X. Min, W. Lu, Z. Zhang, and G. Zhai, “A multi-dimensional aesthetic quality assessment model for mobile game images,” in International Conference on Visual Communications and Image Processing (VCIP), 2021, pp. 1–5.

[60] J. Wang, K. C. Chan, and C. C. Loy, “Exploring clip for assessing the look and feel of images,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 2, 2023, pp. 2555–2563.

[61] W. Sun, X. Min, D. Tu, S. Ma, and G. Zhai, “Blind quality assessment for in-the-wild images via hierarchical feature fusion and iterative mixed database training,” IEEE Journal of Selected Topics in Signal Processing, 2023.

[62] J. Yuan, X. Cao, L. Cao, J. Lin, and X. Cao, “PSCR: Patches samplingbased contrastive regression for aigc image quality assessment,” arXiv preprint arXiv:2312.05897, 2023.

[63] J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong, “ImageReward: Learning and evaluating human preferences for text-toimage generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 15 903–15 935, 2023.

[64] X. Wu, K. Sun, F. Zhu, R. Zhao, and H. Li, “Human preference score: Better aligning text-to-image models with human preference,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 2096–2105.

[65] Y. Kirstain, A. Polyak, U. Singer, S. Matiana, J. Penna, and O. Levy, “Pick-a-Pic: An open dataset of user preferences for text-to-image generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 36 652–36 663, 2023.

[66] Z. Tang, Z. Wang, B. Peng, and J. Dong, “CLIP-AGIQA: boosting the performance of ai-generated image quality assessment with clip,” in International Conference on Pattern Recognition. Springer, 2024, pp. 48–61.

[67] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[68] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “MUSIQ: Multiscale image quality transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5148–5157.

[69] Z. Wu, X. Chen, Z. Pan, X. Liu, W. Liu, D. Dai, H. Gao, Y. Ma, C. Wu, B. Wang et al., “DeepSeek-VL2: Mixture-of-experts vision-language models for advanced multimodal understanding,” arXiv preprint arXiv:2412.10302, 2024.

[70] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., “Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution,” arXiv preprint arXiv:2409.12191, 2024.

[71] A. Meta, “Llama 3.2: Revolutionizing edge ai and vision with open, customizable models,” Meta AI Blog. Retrieved December, vol. 20, p. 2024, 2024.

[72] K. Ma, Z. Duanmu, Z. Wang, Q. Wu, W. Liu, H. Yong, H. Li, and L. Zhang, “Group maximum differentiation competition: Model comparison with few samples,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 851–864, 2018.

# Supplementary Material

## A. OVERVIEW

This supplementary material provides empirical and conceptual evidence to support the main manuscript. We structure the extended analyses to address architectural mechanisms, statistical reliability, and generalization capabilities across diverse visual domains. Specifically, Section B extends our evaluation to large-scale benchmarks to confirm the scalability of our approach against modern generative priors. Section C presents dataset-level and network-level statistical validations of the interaction hypothesis. Section D provides visual analyses of the specialized difference convolution experts and their dynamic routing behavior. Section E establishes the mathematical rigor of our improvements through statistical significance analysis. Finally, Section F verifies the feature disentanglement mechanism through cross-task feature decoupling and demonstrates our implicit anomaly localization capabilities.

## B. EXTENDED EVALUATION ON CONTEMPORARY BENCHMARKS

Performance on Fine-Grained Contemporary Datasets. To further demonstrate scalability, we evaluated ACL-IQA on contemporary datasets focusing predominantly on fine grained alignment and compositional artifacts, including EvalMuse-40K [1], GenAIBench [2], and RichHF-18K [3]. Specifically, EvalMuse-40K provides 40,000 image and text pairs with comprehensive human annotations evaluating alignment. GenAIBench focuses on compositional generation by utilizing highly complex prompts involving logic and attribute binding. RichHF-18K contains 18,000 images annotated with rich human feedback highlighting local artifacts and semantic mismatches. For EvalMuse and GenAIBench, we bypassed the dual-path mechanism in favor of a single-path architecture with adaptive differential convolutions [4], [5]. For RichHF-18K, we utilized its provided artifact scores to drive our dual-path mechanism. Regarding the experimental setup, we adopted distinct data splitting protocols based on availability. Because the official test labels for EvalMuse-40K are not publicly released, we partitioned the available data into a four to one train and test split while maintaining a consistent score distribution, resulting in 26,173 training samples and 6,544 testing samples. For GenAIBench and RichHF-18K, we directly utilized the official splits provided by the dataset authors. To facilitate unified processing, we linearly mapped all subjective scores to a standard range from zero to five. During optimization, we trained our model for 100 epochs on these three datasets.

As shown in Table B.1, ACL-IQA outperforms strong vision-language baselines like LIQE [6] and FGA-BLIP2 [1] and supervised fine-tuned MLLMs like Q-Align [7] across EvalMuse and RichHF-18K, achieving an impressive correlation on EvalMuse and RichHF. While parameter-heavy MLLMs of approximately 8B parameters exhibit an advantage on GenAIBench due to deep semantic logic reasoning on compositional prompts, our lightweight model of approximately 192M parameters delivers highly competitive and computationally efficient evaluations without relying on resourceintensive architectures.

TABLE B.1  
PERFORMANCE COMPARISON MEASURED BY SRCC AND PLCC ON EVALMUSE-40K, GENAIBENCH, AND RICHHF-18K FOCUSING ON ALIGNMENT QUALITY. THE FT COLUMN INDICATES WHETHER THE MODEL WAS FINE-TUNED. BOTH THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD.
<table><tr><td>Method</td><td>FT</td><td>EvalMuse</td><td>GenAIBench</td><td>RichHF</td></tr><tr><td>LIQE [6]</td><td>√</td><td>0.7708 / 0.7583</td><td>0.5528 / 0.5513</td><td>0.6884 / 0.6616</td></tr><tr><td>FGA-BLIP2 [1]</td><td>√</td><td>0.7723 / 0.7716</td><td>0.5639 / 0.5578</td><td>0.6548 / 0.6415</td></tr><tr><td>Q-Align [7]</td><td>x</td><td>0.3236 / 0.3241</td><td>0.2539 / 0.2275</td><td>0.2654 / 0.2774</td></tr><tr><td>Q-Align [7]</td><td>√</td><td>0.7467  / 0.7375</td><td>0.6536 / 0.6365</td><td>0.7120 / 0.7071</td></tr><tr><td>LMM4LMM [8]</td><td>x</td><td>0.7284 / 0.6638</td><td>0.7445 / 0.5870</td><td>0.7193 / 0.6589</td></tr><tr><td>LMM4LMM [8]</td><td>√</td><td>0.8102 / 0.8038</td><td>0.7459 / 0.7244</td><td>0.7485 / 0.7293</td></tr><tr><td>ACL-IQA</td><td>√</td><td>0.7908 / 0.7873</td><td>0.6022 / 0.5901</td><td>0.7324 / 0.7152</td></tr></table>

Generative Prior Domain Shift Analysis. To illustrate the substantial gap between early-stage datasets, such as AGIQA-3K [9] and AIGCIQA2023 [10], and contemporary benchmarks like EvalMi-50K [8], we performed t-SNE visualizations on textual prompt embeddings extracted via Qwen3- Embedding-0.6B [11] and visual image features extracted via CLIP ViT-B/32 [12].

![](images/c910ee140715d18a78874ce685da1d056241631c117d7f803b262bab07cd7dad.jpg)  
Fig. B.1. t-SNE visualization of textual prompt embeddings extracted via Qwen3-Embedding-0.6B [11], demonstrating the expanded semantic manifold of contemporary benchmarks.

As shown in Figs. B.1 and B.2, modern generators in EvalMi-50K encompass a dramatically wider and more dispersed semantic and visual manifold compared to the tightly clustered spaces of early-stage datasets. Despite this pronounced distribution shift, our cross-dataset transfer experiments confirm that ACL-IQA successfully bridges these domain discrepancies, proving that our interaction-aware mechanism learns fundamental quality dynamics rather than overfitting to early generative artifacts.

![](images/f852e8731669942eb10e04224489d61f552ade010f58c69a26ec9378637f5aa2.jpg)  
Fig. B.2. t-SNE visualization of generated image features extracted via CLIP ViT-B/32 [12], illustrating the pronounced visual distribution shift in modern generators.

## C. CONCEPTUAL VALIDATION OF ADVERSARIAL AND COLLABORATIVE INTERACTIONS

To substantiate that the collaborative and adversarial interactions between perceptual fidelity and prompt alignment represent a universal phenomenon across AI-Generated Image Quality Assessment, also referred to as AIGIQA, rather than incidental edge cases, we provide comprehensive statistical validations across both macroscopic dataset distributions and microscopic network optimization dynamics.

Prompt-Level Semantic Quadrant Analysis. Evaluating the correlation between perception and alignment across an entire dataset without grouping yields a misleadingly high positive correlation due to the massive performance gap between early and modern generative models. To eliminate this confounding model-capacity variance and uncover the true semantic distribution, we treat each textual prompt as an independent semantic unit. Specifically, we compute the mean perception quality and mean alignment quality for all images generated under each specific prompt across AGIQA-3K, AIGCIQA2023, RichHF-18K, and EvalMi-50K.

As illustrated in Fig. C.1, partitioning the prompts into four quadrants based on dataset medians reveals a distinct bifurcation. The first and third quadrants represent collaborative scenarios, where simple or standard prompts allow models to succeed or fail at both tasks simultaneously, forming a mutually synergistic relationship. Conversely, the second and fourth quadrants represent adversarial scenarios, which include highly complex, geometrically constrained, or surreal instructions. Under these strict semantic constraints, generative models are forced into an inherent trade-off, which means they either aggressively satisfy textual constraints at the expense of visual realism to produce high alignment and low perception, or they prioritize visual aesthetics while ignoring complex prompt details to yield high perception and low alignment. Statistically, these adversarial prompts occupy a highly significant portion of modern datasets, accounting for 41.40% of the 6,846 total prompts in RichHF-18K, 36.56% of the 2,090 total prompts in EvalMi-50K, 33.33% in AGIQA-3K, and 24.00% in AIGCIQA2023. This confirms that without explicit feature disentanglement, forced alignment representations will severely corrupt perception quality predictions and vice versa.

![](images/8784c3b7dc3e70588010a4c3303f72db44d6ec0ef379e6c438dad479fc25dfce.jpg)  
Fig. C.1. Prompt-level Semantic Quadrant Analysis across AGIQA-3K, AIG-CIQA2023, RichHF-18K, and EvalMi-50K datasets. Each point represents the mean perception and alignment scores of all images generated under a specific prompt by various models. The distribution explicitly delineates collaborative scenarios in the first and third quadrants and adversarial scenarios in the second and fourth quadrants. The regression line and shaded band represent the global positive data trend and its 95% confidence interval, while intersecting lines denote dataset medians.

Network-Level Gradient Conflict Tracking. To provide direct proof of this dual interaction during actual model training, we tracked the step-by-step optimization dynamics within the shared parameter space of our architecture. Specifically, we configured the training batch size to 8 to capture fine-grained optimization behaviors without the gradient-smoothing effect inherent in large batches. For each training batch, we extracted the gradient vector of the alignment score loss and the gradient vector of the perception or artifact score loss with respect to the shared backbone features.

As shown by the cosine similarity histograms and Kernel Density Estimation curves in Fig. C.2, the distribution explicitly bifurcates across the zero-similarity boundary across all benchmarks. A significant mass resides in the positive domain, where the cosine similarity is greater than zero, mathematically proving synergistic collaborative learning where tasks share consistent optimization directions. Crucially, an equally massive portion of the distribution falls into the negative domain, where the cosine similarity is less than zero. This destructive interference provides direct evidence of severe gradient conflicts, demonstrating that alignment optimization directly opposes prediction optimization for complex semantic batches. This empirical evidence firmly validates the necessity of our Gradient Reversal Layer, or GRL, for adversarial decoupling and the Dual-Gated Mixture-of-Experts, known as MoE, dynamic routing.

![](images/abfdb83a53f2c700d98226b9a20eea351922540581a01a4ff65acf27ff77e31c.jpg)  
Fig. C.2. Network-level Gradient Conflict Tracking. The histograms display the distribution of batch-wise cosine similarities between the alignment gradients and the perception or artifact gradients during training across the AGIQA-3K, AIGCIQA2023, RichHF-18K, and EvalMi-50K datasets. The curve indicates the Kernel Density Estimation, and the vertical line marks the zero-similarity boundary.

## D. VISUALIZATION OF SPECIALIZED CONVOLUTIONAL EXPERTS

Our framework embeds difference convolutions into the deep latent space of the Vision Transformer [13] to capture fine-grained details in AI-generated images. We employ six types of convolution experts: Vanilla, Central Difference [14], Angular Difference, Radial Difference, Horizontal Difference, and Vertical Difference Convolutions [4], [15].

As visualized in Fig. D.1, the intermediate feature maps exhibit highly specialized response patterns. The Vanilla convolution primarily anchors global contextual information and salient objects. In contrast, difference convolutions strongly activate around structural boundaries, misaligned edges, and complex geometric distortions. These operators are embedded deep within the Vision Transformer architecture, each token already possesses a global receptive field, and thus the feature responses present a blocky, patch-wise distribution. Consequently, these operators act upon abstract semantic tokens to capture patch-level semantic or structural discontinuities, enabling the detection of higher-order compositional errors and semantic misalignments rather than low-level pixel noise.

## E. STATISTICAL SIGNIFICANCE AND SENSITIVITY ANALYSES

Statistical Significance Analysis. To ensure that our numerical gains stem from intrinsic algorithmic superiority rather than random data variance, we conducted a rigorous 10- fold cross-validation on AGIQA-3K and AIGCIQA2023, comparing ACL-IQA directly against the competitive baseline IPCE [16] under identical data splits. We computed the 95% Confidence Intervals for both metrics alongside paired t-tests across the 10 independent evaluation folds.

As demonstrated in Table E.1, our proposed ACL-IQA successfully passes the statistical significance test across all evaluations. Not only does our model achieve higher confidence intervals, but the p-values generated by the paired t-test are all well below 0.01, conclusively verifying that our decoupled interaction-aware mechanism brings a mathematically significant improvement rather than a coincidental random fluctuation.

Prompt Sensitivity Analysis. To validate the necessity and robustness of our text template formulation, we systematically designed four distinct variants: Default, Concise, Verbose, and Synonym templates. When conceiving these variations, we deliberately manipulated key linguistic dimensions including sentence structure, instructional verbosity, and synonym phrasing. Specifically, the concise variant stripped away all decorative words to test minimalism, the verbose variant expanded the instructions into complex compound sentences to test long-context attention, and the synonym variant replaced core task keywords with semantically equivalent phrasing to test lexical sensitivity.

We evaluated training sensitivity by retraining the model from scratch on the EvalMi-50K dataset for 10 epochs using each of the four conceptual variations. As summarized in Table E.2, our gating mechanism remains highly robust across different language formulations with generally stable convergence. Crucially, our carefully designed Default Template optimally guides the MoE routing, achieving the highest overall balance and correlation for both perception and alignment tasks. Furthermore, during inference, evaluating unseen templates on a trained model yielded virtually negligible performance fluctuations due to the underlying hard-routing stability of our cross-attention mechanism.

## F. CROSS-TASK FEATURE DECOUPLING AND IMPLICIT LOCALIZATION

Cross-Task Feature Decoupling Evaluation. To directly verify the claimed relationships and the exact efficacy of our disentanglement mechanism, we designed a cross-task feature decoupling evaluation across AGIQA-3K, AIGCIQA2023, and EvalMi-50K. By isolating the collaborative learning path $F _ { c o l }$ and the adversarial learning path $F _ { a d v } $ , which are extracted from fully trained models, we forced these decoupled features to independently predict both Perception MOS and Alignment MOS.

As shown in Tables F.1 and F.2, the collaborative path $F _ { c o l }$ is designed to capture complementary synergy, retaining mutual information and yielding high accuracy when predicting both primary and secondary dimensions. Conversely, when $F _ { a d v }$ is evaluated on the secondary, adversarially decoupled task, its correlation converges toward near-zero or slightly negative values. This targeted degradation validates that the Gradient Reversal Layer [17] successfully strips away interfering representations, ensuring that perceptual features remain unbiased by prompt alignment constraints and vice versa.

To visually substantiate how our dual path mechanism adapts to diverse semantics, we conducted a granular qualitative evaluation. As illustrated in Figs. F.1 and F.2, our collaborative path $F _ { c o l }$ excels in collaborative regimes, where visual clarity and semantic alignment are mutually reinforcing.

![](images/bca426e8db845ca89f540940ae2a65b41a56138b9b96ac97a984a848faf78a30.jpg)  
Prompt: A photo of a bottle and a refrigerator.

![](images/ea02c7f0fbf7498cf19224b306a2d0c45225971c100e673347050ce56858ecf6.jpg)  
Prompt: A heartfelt photo of the phrase 'I love you' written in cursive on a handwritten note, placed next to a lit candle, creating an intimate and warm atmosphere.  
Fig. D.1. Feature maps from convolution experts for perception and alignment quality prediction. Each section displays pixel-level and patch-level features. Red boxes denote the top-3 dynamically gated experts, which are fused to form the final map, involving Vanilla Convolution, Central Difference Convolution, Angular Difference Convolution, Radial Difference Convolution, Horizontal Difference Convolution, and Vertical Difference Convolution.

TABLE E.1  
STATISTICAL SIGNIFICANCE ANALYSIS USING 10-FOLD CROSS-VALIDATION COMPARING ACL-IQA AGAINST THE COMPETITIVE BASELINE IPCE. THEp-VALUES ARE COMPUTED USING A PAIRED t-TEST, WHERE A VALUE LESS THAN 0.01 INDICATES AN EXTREMELY SIGNIFICANT IMPROVEMENT.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>SRCC 95% CI</td><td rowspan=1 colspan=1>PLCC 95% CI</td><td rowspan=1 colspan=1>Paired t-test (p-value)</td></tr><tr><td rowspan=2 colspan=1>AGIQA-3K</td><td rowspan=1 colspan=1>Perception</td><td rowspan=1 colspan=1>IPCEACL-IQA</td><td rowspan=1 colspan=1>[0.8775,0.8907][0.8840, 0.8966]</td><td rowspan=1 colspan=1>[0.9213,0.9279[0.9249, 0.9316]</td><td rowspan=1 colspan=1> $\overline { { p < 0 . 0 1 } }$ (SRCC: 0.0031, PLCC: 0.0040)</td></tr><tr><td rowspan=1 colspan=1>Alignment</td><td rowspan=1 colspan=1>IPCEACL-IQA</td><td rowspan=1 colspan=1>[0.7563,0.7831][0.7694, 0.7969]</td><td rowspan=1 colspan=1>[0.8624,0.8826][0.8698, 0.8902</td><td rowspan=1 colspan=1> $\overline { { p < 0 . 0 1 } }$ (SRCC: 0.0069, PLCC: 0.0018)</td></tr><tr><td rowspan=2 colspan=1>AIGCIQA2023</td><td rowspan=1 colspan=1>Perception</td><td rowspan=1 colspan=1>IPCEACL-IQA</td><td rowspan=1 colspan=1>[0.8562,0.8718][0.8650, 0.8790]</td><td rowspan=1 colspan=1>[0.8736, 0.8840][0.8806, 0.8925]</td><td rowspan=1 colspan=1> $\overline { { p < 0 . 0 1 } }$ (SRCC: 0.0094, PLCC: 0.0013)</td></tr><tr><td rowspan=1 colspan=1>Alignment</td><td rowspan=1 colspan=1>IPCEACL-IQA</td><td rowspan=1 colspan=1>[0.7887,0.8071][0.8004, 0.8181</td><td rowspan=1 colspan=1>[0.7792,0.7982][0.7919, 0.8091]</td><td rowspan=1 colspan=1> $\overline { { p < 0 . 0 0 1 } }$ (SRCC: 0.0006, PLCC: 0.0001)</td></tr></table>

TABLE E.2  
PERFORMANCE COMPARISON USING SRCC AND PLCC OF DIFFERENT PROMPT TEMPLATES WHEN TRAINED ON THE EVALMI-50K DATASET.
<table><tr><td>Prompt Template</td><td>Perception</td><td>Alignment</td></tr><tr><td>Default Template</td><td>0.8719 / 0.8982</td><td>0.8556 / 0.8673</td></tr><tr><td>Concise Template</td><td>0.8705 / 0.8985</td><td>0.8545 / 0.8660</td></tr><tr><td>Verbose Template</td><td>0.8719 / 0.8991</td><td>0.8551 / 0.8672</td></tr><tr><td>Synonym Template</td><td>0.8693 / 0.8978</td><td>0.8531 / 0.8639</td></tr></table>

Conversely, in adversarial regimes involving complex constraints or intentional stylistic degradations, relying solely on $F _ { c o l }$ introduces bias by conflating semantic execution with visual quality. Our adversarial path $F _ { a d v }$ mitigates this by embedding a Gradient Reversal Layer to actively unlearn interfering representations, mimicking human cognitive inhibition to judge each dimension independently. As demonstrated, our joint framework dynamically shifts routing weights via the Dual Gated Mixture of Experts to bridge these paradigms, effectively replicating human adaptability and ensuring robust evaluation reliability across diverse generative content.

Implicit Anomaly Localization via Grad-CAM. While our framework is designed as a global scoring architecture without computationally expensive dense segmentation heads, it internally performs precise spatial anomaly localization. To demonstrate this, we conducted a Gradient-weighted Class Activation Mapping analysis on the RichHF-18K dataset by isolating the logits corresponding to the lowest quality levels, representing bad and poor quality, and backpropagating gradients to the deep ViT feature maps.

As illustrated in Figs. F.3 and F.4, our model successfully

TABLE F.1  
CROSS TASK FEATURE DECOUPLING EVALUATION FOR THE ACL-IQA MODEL TRAINED TO PREDICT PERCEPTUAL QUALITY. $F _ { c o l }$ DENOTES THE COLLABORATIVE FEATURE, AND $F _ { a d v }$ DENOTES THE ADVERSARIAL FEATURE. THE SYMBOL ✓ INDICATES THE ACTIVE FEATURE USED FOR PREDICTION, WHILE THE DOWNWARD ARROW INDICATES THE RELATIVE PERFORMANCE DECREASE COMPARED TO THE COLLABORATIVE PATH.
<table><tr><td>Dataset</td><td> $F _ { c o l }$   $F _ { a d v }$ </td><td>SRCC for Perception</td><td>PLCC for Perception</td><td>SRCC for Alignment</td><td></td><td>PLCC for Alignment</td></tr><tr><td rowspan="3">AGIQA-3K</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>0.8969</td><td>0.9162</td><td></td><td></td></tr><tr><td> $\overline { { \checkmark } }$ </td><td></td><td>0.8819</td><td>0.8858</td><td>0.7207</td><td>0.8125</td></tr><tr><td></td><td> $\checkmark$ </td><td>0.8282</td><td>0.7777</td><td>0.3757(↓ 47.9%)</td><td>0.0299 (↓ 96.3%)</td></tr><tr><td rowspan="3">AIGCIQA2023</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>0.8720</td><td>0.8865</td><td></td><td></td></tr><tr><td> $\overline { { \checkmark } }$ </td><td></td><td>0.8379</td><td>0.8588</td><td>0.7564</td><td>0.7508</td></tr><tr><td></td><td> $\bigtriangledown$ </td><td>0.8403</td><td>0.8576</td><td>-0.0427 (↓ 105.6%)</td><td>0.0438 (↓ 94.2%)</td></tr><tr><td rowspan="3">EvalMi-50K</td><td> $\overline { { \checkmark } }$ </td><td> $\overline { { \checkmark } }$ </td><td>0.8810</td><td>0.9050</td><td></td><td></td></tr><tr><td> $\bigtriangledown$ </td><td></td><td>0.7697</td><td>0.7955</td><td>0.7048</td><td>0.7423</td></tr><tr><td></td><td> $\overline { { \checkmark } }$ </td><td>0.7792</td><td>0.8251</td><td>0.0379 ↓94.6%)</td><td>-0.0159 (↓ 102.1%)</td></tr></table>

TABLE F.2  
CROSS TASK FEATURE DECOUPLING EVALUATION FOR THE ACL-IQA MODEL TRAINED TO PREDICT ALIGNMENT QUALITY. THE TABLE PRESENTS THE JOINT PERFORMANCE ALONGSIDE ISOLATED FEATURE PREDICTIONS, WHERE THE DOWNWARD ARROW INDICATES THE RELATIVE PERFORMANCE DECREASE COMPARED TO THE COLLABORATIVE PATH.
<table><tr><td>Dataset</td><td> $F _ { c o l }$ </td><td> $F _ { a d v }$ </td><td>SRCC for Alignment</td><td>PLCC for Alignment</td><td>SRCC for Perception</td><td>PLCC for Perception</td></tr><tr><td rowspan="3">AGIQA-3K</td><td> $\overline { { \checkmark } }$ </td><td> $\checkmark$ </td><td>0.7832</td><td>0.8800</td><td></td><td></td></tr><tr><td> $\overline { { \checkmark } }$ </td><td></td><td>0.7161</td><td>0.8116</td><td>0.8788</td><td>0.8971</td></tr><tr><td></td><td> $\bigtriangledown$ </td><td>0.7336</td><td>0.8248</td><td>0.2194(↓75.0%)</td><td>0.2393(↓ 73.3%)</td></tr><tr><td rowspan="3">AIGCIQA2023</td><td> $\checkmark$ </td><td> $\overline { { \checkmark } }$ </td><td>0.8092</td><td>0.8005</td><td></td><td></td></tr><tr><td> $\overline { { \checkmark } }$ </td><td></td><td>0.7257</td><td>0.7213</td><td>0.8469</td><td>0.8731</td></tr><tr><td></td><td> $\bigtriangledown$ </td><td>0.7653</td><td>0.7655</td><td>0.4242(↓ 49.9%)</td><td>0.3691 ↓57.7%)</td></tr><tr><td rowspan="2">EvalMi-50K</td><td> $\overline { { \checkmark } }$ </td><td> $\overline { { \checkmark } }$ </td><td>0.8664</td><td>0.8795</td><td></td><td></td></tr><tr><td> $\bigtriangledown$ </td><td></td><td>0.6578</td><td>0.6980</td><td>0.7633</td><td>0.8092</td></tr><tr><td></td><td></td><td> $\overline { { \checkmark } }$ </td><td>0.6126</td><td>0.6370</td><td>0.3275(↓ 57.1%)</td><td>0.3107 ↓ 61.6%)</td></tr></table>

![](images/3feedad65afceba04be821fed8a22ad33f97b8688cadbe128d43878b90ae7913.jpg)  
Prompt: black mickey mouse skull

![](images/0bd5101d4f7c24815c8ebf7c2a4a289160de66f85e4322bef0970acb8e373545.jpg)  
Prompt: photo of a white fashion jewelry store, cold color

![](images/da888b29fcbff1de540583c49b2b8ad43a5430a0c2bce8a5ef8c14849300e64c.jpg)  
Prompt: cute green tabby cat made of leaves, cold color

![](images/13afcbacdb0f911fa8d0a21ce626dd06891162e1f7abf0812949495857c80059.jpg)  
Prompt: a biomechanical dahlia flower, sci-fi style

<table><tr><td rowspan=1 colspan=1>GT (MOS): High</td><td rowspan=1 colspan=1>GT (MOS): Low</td></tr><tr><td rowspan=1 colspan=1>Col Path Score: High</td><td rowspan=1 colspan=1>Col Path Score: Low</td></tr><tr><td rowspan=1 colspan=1>Adv Path Score: Low</td><td rowspan=1 colspan=1>Adv Path Score: High</td></tr><tr><td rowspan=1 colspan=1>ACL-IQA Score: High</td><td rowspan=1 colspan=1>ACL-IQA Score: Low</td></tr></table>

![](images/e4d55b11ec08835302b07218beac10543e649022a08a04727c08a6e1c5b16766.jpg)  
Prompt: portrait photo still of a spooky scarecrow holding a gift, top view, close-up, sci-fi style

<table><tr><td rowspan=1 colspan=1>GT (MOS): High</td><td rowspan=1 colspan=1>GT (MOS): Low</td></tr><tr><td rowspan=1 colspan=1>Col Path Score: High</td><td rowspan=1 colspan=1>Col Path Score: Low</td></tr><tr><td rowspan=1 colspan=1>Adv Path Score: Low</td><td rowspan=1 colspan=1>Adv Path Score: High</td></tr><tr><td rowspan=1 colspan=1>ACL-IQA Score: High</td><td rowspan=1 colspan=1>ACL-IQA Score: Low</td></tr></table>

![](images/4f9dd87a2f7d5acdd05e81e4182c2f0697bda16ee291999e0bdd7a73e52ce808.jpg)  
Prompt: breaking moon as seen from the surface of the moon, top view, hyper detail

![](images/d7a32b1d02389d3d2fbf3759f7221155100f99a7ffa328816762d44b831c2ef0.jpg)  
Prompt: elegant oval mirror and toucan with reflection, blurred detail, sci-fi style

![](images/e2a001add3872dedea05aed1ce31e5496868506a72cbc42154ec0f686c75dffc.jpg)  
Prompt: breaking moon as seen from the surface of the moon, top view, hyper detail

<table><tr><td rowspan=1 colspan=1>GT (MOS): High</td><td rowspan=1 colspan=1>GT (MOS): Low</td></tr><tr><td rowspan=1 colspan=1>Col Path Score: Low</td><td rowspan=1 colspan=1>Col Path Score: High</td></tr><tr><td rowspan=1 colspan=1>Adv Path Score: High</td><td rowspan=1 colspan=1>Adv Path Score: Low</td></tr><tr><td rowspan=1 colspan=1>ACL-IQA Score: High</td><td rowspan=1 colspan=1>ACL-IQA Score: Low</td></tr></table>

<table><tr><td rowspan=1 colspan=1>GT (MOS): High</td><td rowspan=1 colspan=1>GT (MOS): Low</td></tr><tr><td rowspan=1 colspan=1>Col Path Score: Low</td><td rowspan=1 colspan=1>Col Path Score: High</td></tr><tr><td rowspan=1 colspan=1>Adv Path Score: High</td><td rowspan=1 colspan=1>Adv Path Score: Low</td></tr><tr><td rowspan=1 colspan=1>ACL-IQA Score: High</td><td rowspan=1 colspan=1>ACL-IQA Score: Low</td></tr></table>

Fig. F.1. Qualitative diagnostic panels for Perceptual Quality Assessment. The top rows illustrate the collaborative dominance regime where standard prompt benefit from shared alignment features. The bottom rows illustrate the adversarial dominance regime where complex prompts trigger semantic interference in the collaborative path, necessitating active feature decoupling via the adversarial path. Our joint ACL-IQA model dynamically balances both paths to match ground truth ratings.

Prompt  
![](images/67d8977aeb151f811c5ab4ba6c0d3ae66249fd4a57f7732ab083097e7e3b3d4a.jpg)

<table><tr><td rowspan=1 colspan=1>GT (MOS): High</td><td rowspan=1 colspan=1>GT (MOS): Low</td></tr><tr><td rowspan=1 colspan=1>Col Path Score: High</td><td rowspan=1 colspan=1>Col Path Score: Low</td></tr><tr><td rowspan=1 colspan=1>Adv Path Score: Low</td><td rowspan=1 colspan=1>Adv Path Score: High</td></tr><tr><td rowspan=1 colspan=1>ACL-IQA Score: High</td><td rowspan=1 colspan=1>ACL-IQA Score: Low</td></tr></table>

![](images/aec905360a4edf24cc4ad9a683c60bbfdf703f4ca3b954d285263a84277c5183.jpg)  
Prompt: cute fluffy baby lop eared bunny rabbit, cold color

![](images/7b321318e3b43979ab9bb3f83a6cf84a673decfeec20dfc4e1b5c7dcf7bba428.jpg)  
Prompt: the gold skull 'coming out from open cracked marble head, cold color, hyper detail, sci-fi style

<table><tr><td rowspan=1 colspan=1>GT (MOS): High</td><td rowspan=1 colspan=1>GT (MOS): Low</td></tr><tr><td rowspan=1 colspan=1>Col Path Score: High</td><td rowspan=1 colspan=1>Col Path Score: Low</td></tr><tr><td rowspan=1 colspan=1>Adv Path Score: Low</td><td rowspan=1 colspan=1>Adv Path Score: High</td></tr><tr><td rowspan=1 colspan=1>ACL-IQA Score: High</td><td rowspan=1 colspan=1>ACL-IQA Score: Low</td></tr></table>

![](images/2890a35780ce9d6af84540ef228b9a0be738bf16f5f75572514f1622680ad6dd.jpg)

![](images/c9cc56173a6c31e5a5130d81f46b09dd0fe5cb4b07a6197f12b5bfc75f2f1e41.jpg)  
Prompt: vaporware yosemite amusement park, sci-fi style

GT (MOS): High Col Path Score: Low Adv Path Score: High ACL-IQA Score: High

![](images/24d4056f9e901ceac8f2cd95ea53af84bb456162f6e46ed7d041d754b85b843e.jpg)  
Prompt: a painting of pink panther as a vampire , blurred detail  
Prompt: portrait of mecha robot

GT (MOS): Low Col Path Score: High Adv Path Score: Low ACL-IQA Score: Low

![](images/a57f40ba3c769b1045ee80b9c1eb9bae18c9e4dae963b71c5ce068c18c1476c6.jpg)

![](images/5efc1df96fee36c771b768123c5bee98a9b4560e0a12ab92d771a7958a5bb783.jpg)  
Prompt: large purple lovecraftian void, sci-fi style

GT (MOS): High Col Path Score: Low Adv Path Score: High ACL-IQA Score: High

Prompt: cute fluffy lion cheetah hybrid mixed, blurred detail

![](images/fd6d2cc61b62f6d1710516ec0a112d36c12600e7613f592899e1a73d078c5da1.jpg)  
Prompt: grand canyon full of potato salad, sci-fi style

Fig. F.2. Qualitative diagnostic panels for Alignment Quality Assessment. The top rows show scenarios where visual clarity is essential to verify fine grained semantic details, allowing the collaborative path to excel. The bottom rows demonstrate cases involving intentional stylistic degradation, where the adversarial path effectively strips away aesthetic bias to accurately evaluate text image correspondence.  
Fig. F.3. Visualizations of implicit localization for artifacts and implausibility on RichHF-18K. The columns display the generated image, human-annotated Ground Truth mask, anomaly-targeted gradient heatmap from ACL-IQA, and textual prompt.  
![](images/2805103c45fff17529ff5891290f9494cb7def0912e9cb2ede5ff3fa2804b2ef.jpg)  
Fig. F.4. Visualizations of implicit localization for text-image misalignment on RichHF-18K. The columns display the generated image, human-annotated Ground Truth mask, anomaly-targeted gradient heatmap from ACL-IQA, and textual prompt.

locates specific generative flaws without ever being trained on pixel-level mask supervision. For structural distortions in Fig. F.3, high-activation heatmaps precisely cover distorted anatomy, such as deformed hands, and structural anomalies, overlapping with human-annotated Ground Truth masks. For text-image misalignment in Fig. F.4, the anomaly-targeted gradients accurately pinpoint misspelled rendered text, for instance Panera Bread logos, and missing objects dictated by the prompt. This confirms that our global quality scores are grounded in fine-grained spatial and semantic understanding.

## REFERENCES

[1] S. Han, H. Fan, J. Fu, L. Li, T. Li, J. Cui, Y. Wang, Y. Tai, J. Sun, C. Guo et al., “EvalMuse-40K: A reliable and fine-grained benchmark with comprehensive human annotations for text-to-image generation model evaluation,” arXiv preprint arXiv:2412.18150, 2024.

[2] B. Li, Z. Lin, D. Pathak, J. Li, Y. Fei, K. Wu, T. Ling, X. Xia, P. Zhang, G. Neubig et al., “GenAI-Bench: Evaluating and improving compositional text-to-visual generation,” arXiv preprint arXiv:2406.13743, 2024.

[3] Y. Liang, J. He, G. Li, P. Li, A. Klimovskiy, N. Carolan, J. Sun, J. Pont-Tuset, S. Young, F. Yang et al., “Rich human feedback for text-to-image generation,” in CVPR, 2024.

[4] Z. Su, W. Liu, Z. Yu, D. Hu, Q. Liao, Q. Tian, M. Pietikainen, and L. Liu,¨ “Pixel difference networks for efficient edge detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5117–5127.

[5] C. Kong, A. Luo, P. Bao, Y. Yu, H. Li, Z. Zheng, S. Wang, and A. C. Kot, “MoE-FFD: Mixture of experts for generalized and parameter-efficient face forgery detection,” arXiv preprint arXiv:2404.08452, 2024.

[6] W. Zhang, G. Zhai, Y. Wei, X. Yang, and K. Ma, “Blind image quality assessment via vision-language correspondence: A multitask learning perspective,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 071–14 081.

[7] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, Y. Gao, A. Wang, E. Zhang, W. Sun et al., “Q-Align: Teaching lmms for visual scoring via discrete text-defined levels,” in Proceedings of the 41st International Conference on Machine Learning (ICML), vol. 235, 2024, pp. 54 015– 54 029.

[8] J. Wang, H. Duan, Y. Zhao, J. Wang, G. Zhai, and X. Min, “LMM4LMM: Benchmarking and evaluating large-multimodal image generation with lmms,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 17 312–17 323.

[9] C. Li, Z. Zhang, H. Wu, W. Sun, X. Min, X. Liu, G. Zhai, and W. Lin, “AGIQA-3k: An open database for ai-generated image quality assessment,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 8, pp. 6833–6846, 2023.

[10] J. Wang, H. Duan, J. Liu, S. Chen, X. Min, and G. Zhai, “AIG-CIQA2023: A large-scale image quality assessment database for ai generated images: from the perspectives of quality, authenticity and correspondence,” in CAAI International Conference on Artificial Intelligence. Springer, 2023, pp. 46–57.

[11] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin et al., “Qwen3 Embedding: Advancing text embedding and reranking through foundation models,” arXiv preprint arXiv:2506.05176, 2025.

[12] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning. PmLR, 2021, pp. 8748–8763.

[13] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=YicbFdNTTy

[14] Z. Yu, C. Zhao, Z. Wang, Y. Qin, Z. Su, X. Li, F. Zhou, and G. Zhao, “Searching central difference convolutional networks for face antispoofing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 5295–5305.

[15] Z. Chen, Z. He, and Z.-M. Lu, “DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention,” IEEE Transactions on Image Processing, vol. 33, pp. 1002–1015, 2024.

[16] F. Peng, H. Fu, A. Ming, C. Wang, H. Ma, S. He, Z. Dou, and S. Chen, “AIGC image quality assessment via image-prompt correspondence,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6432–6441.

[17] Y. Ganin and V. Lempitsky, “Unsupervised domain adaptation by backpropagation,” in International Conference on Machine Learning. PMLR, 2015, pp. 1180–1189.