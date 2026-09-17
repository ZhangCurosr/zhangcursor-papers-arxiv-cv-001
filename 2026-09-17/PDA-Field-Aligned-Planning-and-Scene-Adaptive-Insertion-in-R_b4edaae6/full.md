# PDA++: Field-Aligned Planning and Scene-Adaptive Insertion in Remote Sensing

Xianchi Dong, Yingyan Hou, Chao Ren, Member, IEEE, Wanxuan Lu, Zihan Wei, Hongfeng Yu, Yixiao Wang, Member, IEEE, Chubo Deng, and Xian Sun<sup>∗</sup>, Senior Member, IEEE

Abstract—Remote sensing recognition is often constrained by scarce observations of rare targets and costly annotations, making realistic synthetic augmentation particularly valuable for few-shot and long-tailed scenarios. Object insertion provides an efficient way to increase target diversity while preserving authentic background scenes, but realistic insertion in overhead imagery requires the generated target to adapt coherently to its surrounding environment. To this end, we propose PDA++, a unified environment-aware object insertion framework organized as Plan, Decouple, and Assimilate. Planning determines scenecompatible poses through an affordance field that combines geometric clearance with structure- and scale-aware cues. Decoupling introduces a pose-conditioned background that provides precise spatial guidance together with target-scene context, allowing the reference object to preserve its identity while adapting to the target observation. This construction also naturally provides pixel-level masks for segmentation augmentation. Assimilation further improves local coherence by aligning multiscale texture distributions through optimal transport. On the optical benchmark, PDA++ achieves a whole-image FID of 6.28 and improves average few-shot recognition mAP50 by 17.69 points, corresponding to a 28.8% relative gain over the realdata baseline. On SAR imagery, it improves ship detection by 4.10 mAP50 points and remains effective under cross-dataset transfer and amorphous-target insertion. Code is available at https://github.com/lisheyu972/PDA\_PLUS.

Index Terms—Remote sensing, object insertion, data augmen tation, generative models, object Recognition.

## I. INTRODUCTION

R <sup>EMOTE</sup> <sup>sensing</sup> <sup>object</sup> <sup>detection</sup> <sup>and</sup> <sup>recognition</sup> <sup>play</sup>an important role in Earth observation applications, an important role in Earth observation applications particularly in large-scale environmental monitoring and the analysis of critical infrastructure [1]–[3]. Their performance, however, depends heavily on the availability of large-scale annotated data [4]–[6]. In practice, observations of many important targets remain sparse because of limited revisit opportunities and practical acquisition constraints. Rare and strategically important targets therefore tend to exhibit pronounced long-tailed distributions [7], [8]. Moreover, the efficacy of deep detectors exhibits notable vulnerability to data-induced perturbations [9]. The problem is further complicated by substantial appearance variation across scenes and imaging conditions, which makes sample diversity as important as data quantity. Collecting additional real observations is often difficult to control and may still fail to cover rare configurations of interest. Synthetic augmentation offers an alternative by increasing the occurrence and diversity of such targets without requiring new acquisition campaigns. Its usefulness, however, depends strongly on whether the generated samples remain compatible with the physical and observational characteristics of real remote sensing imagery. Realistic augmentation is consequently particularly valuable for improving recognition under few-shot and long-tailed settings [10]–[12].

Object insertion provides a flexible way to increase target diversity while retaining authentic remote sensing backgrounds. Conventional copy-based augmentation is simple and computationally efficient, but the pasted object remains largely unchanged after being transferred to a new scene and often appears incompatible with its surroundings. Such inconsistencies are especially problematic for downstream augmentation because the synthesized target is not merely required to look plausible in isolation. It must also provide supervision that follows the visual statistics and spatial structure of the destination scene. Recent diffusion-based editing methods, including AnyDoor [13], MimicBrush [14], UniCombine [15], Insert Anything [16], Qwen Image Edit [17], and OminiControl [18], have substantially improved object-level generation in natural imagery. Their ability to synthesize missing content from reference conditions suggests a promising direction for remote sensing augmentation. Nevertheless, remote sensing imagery differs fundamentally from natural imagery because object appearance depends strongly on the observation process in addition to scene geometry. The same semantic target can exhibit markedly different responses when observed under different sensing conditions. Directly transferring a reference appearance to a new scene can therefore produce an object that is recognizable yet inconsistent with the destination observation. This makes remote sensing insertion more than a spatial editing problem and requires the generated content to adapt to the scene in which it is placed.

As illustrated in Fig. 1, this incompatibility can emerge throughout the insertion process. An object may already be implausible when its location or pose conflicts with the underlying scene structure, such as an aircraft synthesized outside an admissible airport region or a ship placed on land. Even when the placement is valid, the inserted target may retain characteristics inherited from its source observation and therefore appear inconsistent with the imaging state of the destination scene. Such discrepancies can manifest differently in optical and SAR imagery because the rendered appearance depends strongly on the sensing modality. Residual artifacts may also remain around the insertion boundary when local texture statistics are not compatible with the surrounding back ground. These failure modes are closely coupled because the selected pose determines the environmental context available to subsequent generation. An inappropriate placement cannot be corrected solely through appearance synthesis, while accurate placement does not guarantee that the rendered target will match its surroundings. Reliable insertion therefore requires the synthesized content to adapt progressively to its destination environment rather than treating placement and generation as independent operations.

![](images/b95e3e9bc4e3acb76b24da0291bba6c41979fffdf0af191e63ef857f6554691b.jpg)  
Fig. 1. Illustration of three critical issues in remote sensing object insertion. (a) Geometric placement inconsistency, where objects are inserted into spatially or semantically implausible regions; (b) observation-state inconsistency, where the inserted target retains sensing and acquisition characteristics that are incompatible with the target scene, resulting in mismatched sensor response, illumination, or acquisition geometry; (c) textural discontinuity, where the roughness and texture statistics of the inserted target are incompatible with the surrounding background.

Motivated by this observation, we propose PDA++, a unified Plan Decouple Assimilate framework that progressively adapts inserted content to the target scene. Planning determines a scene-compatible pose through an Affordance Field that com bines geometric clearance with structure-aware and scale-aware evidence. The resulting pose defines not only where the target should appear but also the local context used during generation. Decoupling then introduces a pose-conditioned background that communicates precise spatial support together with the target observation context. The reference object can therefore retain its semantic identity while its rendered appearance is adapted to the destination scene. This construction also naturally associates each synthesized target with a pixel-level mask and thereby extends synthetic augmentation from recognition to segmentation. Assimilation acts during generation to reduce residual local discrepancy. It represents the inserted region and its neighborhood through multi-scale texture statistics and aligns their distributions with optimal transport. The resulting process couples scene-aware placement with environmentadaptive generation so that the output is determined jointly by the reference identity and the destination observation.

A preliminary version of this work appeared at ICML 2026 [19], where the Plan Decouple Assimilate framework was introduced for remote sensing object insertion. This journal version substantially extends the conference work through a redesign of both planning and generation. The original clearance-based planner is upgraded to an Affordance Field that better accounts for scene structure and object scale. The generation stage is reformulated around pose-conditioned scene context, while the original texture objective is replaced by multi-scale optimal-transport alignment. These changes replace the conference ASA and NATA designs with a more unified environment-adaptive generation process. The journal version also extends the functionality of the framework beyond the original box-level augmentation setting. Pixel-level masks produced by the pose-conditioned construction enable segmentation augmentation without additional manual annotation, and recursive synthesis allows several targets to be generated within the same scene while preserving compatibility with previous insertions. We further conduct controlled component analyses to separate the effect of the redesigned modules from that of the updated generative backbone. Together, these extensions broaden the use of the original framework while preserving its applicability across the optical and SAR settings established in the conference version.

The main contributions of this work are summarized as follows:

• We introduce a unified environment-aware formulation for remote sensing object insertion, where the inserted target is progressively adapted to the target scene from placement to appearance. This perspective links scenecompatible layout with observation-aware generation and local harmonization in a single insertion pipeline.

• We propose PDA++, which improves the original Plan– Decouple–Assimilate framework through an Affordance Field planner, pose-conditioned scene representation, and manifold-aligned texture guidance. These designs substan tially improve insertion fidelity, achieving a whole-image FID of 6.28 on the optical benchmark while also enabling pixel-level supervision for segmentation augmentation.

• We validate PDA++ extensively on both optical and SAR imagery under diverse augmentation settings. On MAR20- 11-FewShot, the generated samples improve average recognition mAP50 by 17.69 points, corresponding to a 28.8% relative gain over the real-data baseline. On SAR ship detection, the improvement reaches 4.10 mAP50 points, with consistent effectiveness under cross-dataset transfer and amorphous-target insertion.

## II. RELATED WORK

a) Generative Models: Generative models have undergone a major paradigm shift from generative adversarial networks (GANs) [20] to denoising diffusion probabilistic models [21]. GANs once dominated image generation because of their strong visual fidelity and efficient sampling, but adversarial optimization is often unstable and may limit distribution coverage. Diffusion models instead formulate generation as the reverse of a stochastic noising process, offering improved training stability and sample diversity [22]. Their strong performance in high-fidelity synthesis has made them a dominant paradigm in modern generative modeling. Recent work further extends diffusion models toward unified generation-and-editing frameworks that support a broad range of conditional synthesis tasks within a shared backbone [23]. Latent diffusion models (LDMs) [24] improve efficiency by moving the generative process into a lower-dimensional latent space while largely preserving visual quality.

Parameter-efficient adaptation further facilitates the transfer of pretrained generative models to specialized domains. Low-Rank Adaptation (LoRA) [25] introduces only a small set of trainable parameters and therefore avoids expensive fullmodel optimization. Together with latent diffusion and flexible conditioning mechanisms, these developments provide the technical basis for modern object insertion frameworks.

b) Object Insertion and Image Editing: Object insertion aims to integrate a foreground object into a new background while maintaining visual plausibility and semantic coherence. It can be viewed as a specialized form of image composition whose realism depends on whether the inserted content is compatible with the geometry and appearance of the destination scene [26]. Conventional copy-based or blending methods are computationally efficient, but they largely preserve the source appearance and provide little capability to adapt the inserted object to its new environment.

Recent diffusion-based editing models have substantially improved this capability. Representative methods such as AnyDoor [13], MimicBrush [14], Insert Anything [16], Uni-Combine [15], Qwen Image Edit [17], OminiControl [18], and AnyEdit [27] demonstrate increasingly flexible referenceguided object manipulation. Controllable diffusion models can further exploit visual or spatial conditions to constrain the generation process [28]. Of particular relevance to our work, OminiControl [18] and Insert Anything [16] directly place reference information into the generation context rather than relying on a dedicated reference encoder. This in-context formulation allows subject identity to interact with target-scene information through the native attention mechanism.

Recent studies have also explored more specialized composition settings. Zero-shot methods investigate how intrinsic scene cues can guide object integration [29], while featurelevel approaches improve the interaction between reference content and target appearance [30]. Personalized insertion and 3D scene editing further extend the controllability of object composition [31], [32]. Meanwhile, HiddenObjects develops scalable spatial priors for object placement [33], and Region-to-Region improves local harmonization through region-aware injection [34]. Physical compatibility has also received increasing attention: SpotLight addresses scene-aware relighting [35], whereas SSN explicitly models soft shadows during composition [36].

Despite these advances, most existing insertion models are developed primarily for natural imagery and datasets such as COCO [37]. Their direct application to remote sensing imagery therefore suffers from a substantial domain gap [38]. Unlike perspective natural scenes, remote sensing observations exhibit stronger topological constraints and sensing-dependent appearance variations. A target that appears plausible in isolation may consequently remain incompatible with its destination observation. This difference motivates insertion frameworks that account for both scene structure and the imaging characteristics of remote sensing data.

c) Environment-Aware Generation and In-Context Conditioning: A central challenge in conditional generation is to make synthesized content respond coherently to the en vironment in which it appears. Beyond preserving object identity, realistic synthesis requires the generated appearance to remain compatible with the spatial layout and observation characteristics of the surrounding scene. Recent studies suggest that large-scale diffusion models already encode substantial environmental knowledge. DiffusionLight [39], for example, demonstrates that scene illumination can be recovered from pretrained generative representations. Related studies show that diffusion features retain geometric information useful for monocular depth estimation [40], while RGBX [41] reveals material- and lighting-aware representations that support intrinsic decomposition and relighting. These findings indicate that pretrained generative models contain useful priors for adapting synthesized content to its environment.

In-context conditioning provides a direct way to expose such environmental evidence during generation. OminiControl [18] and Insert Anything [16] place reference and target information within a shared attention context, allowing subject identity to interact directly with the destination scene. This mechanism is particularly suitable for object insertion because the reference content must remain recognizable while its appearance changes with the new environment. Existing studies, however, largely focus on natural imagery, where environmental variation is dominated by visible-scene factors such as illumination and viewpoint. Remote sensing observations additionally involve modality-dependent responses and acquisition geometry, mak ing environment adaptation more closely tied to the underlying imaging process.

d) Generative Models in Remote Sensing: Generative modeling has also become increasingly important in remote sensing. Diffusion-based methods have been applied to image restoration, including remote sensing super-resolution [42], [43]. Their use for data augmentation has expanded as well, providing an alternative means of increasing training diversity when real observations are limited [44], [45]. More recent studies investigate controllable remote sensing synthesis through image editing [46], change-oriented generation [47], and task-specific synthetic data construction [48].

Foundation-style generative models further broaden the scale and controllability of remote sensing synthesis. Diffusion-Sat [49] incorporates geospatial metadata into conditional satellite image generation, while CRS-Diff [50] exploits geospatial conditions for controllable synthesis. MetaEarth [51] and Text2Earth [52] extend generation toward global-scale and text-driven settings, with EcoMapper [53] further introducing climate-aware environmental information. Other approaches use segmentation or land-cover information as spatial guidance [54], [55]. Large geospatial models such as CrossEarth [56] and SARATR-X [57] also highlight the importance of modalityaware pretraining. For SAR generation in particular, recent work has begun to incorporate physical or geometric priors into diffusion-based synthesis [58], [59].

Despite this progress, high-fidelity object insertion remains comparatively underexplored in remote sensing [60]. Existing methods generally address broad image generation or isolated aspects of image editing, whereas insertion requires the generated target to remain compatible with the destination environment throughout the synthesis process. A plausible pose alone is insufficient if the resulting object retains an incompatible observation state, while visually coherent generation can still expose artifacts around the insertion boundary. Recent studies on shadow consistency and texture harmonization further illustrate the importance of environment-aware integration for synthetic imagery [61]–[63].

Our work addresses this gap by viewing remote sensing insertion as a progressive environment-adaptation process. Instead of treating object insertion as generic image editing, the proposed framework first establishes scene-compatible spatial support and then adapts the generated content to the destination observation. Residual local discrepancies are subsequently reconciled through texture-aware guidance, allowing placement and appearance adaptation to operate within a unified generation pipeline.

## III. PRELIMINARIES

## A. Problem Formulation

Given a remote sensing background image $I _ { b g } \in \mathbb { R } ^ { H \times W \times 3 }$ and a target object image $\mathbf { \bar { \rho } } _ { I _ { s u b } } \in \mathbb { R } ^ { H \times W \times 3 }$ , our goal is to gen erate a composite image $I _ { f i n a l } \in \mathbb { R } ^ { H \times W \times 3 }$ in which the target is inserted at a scene-compatible location. The insertion process couples scene-aware planning with conditional generation.

Planning determines an insertion mask $M \in \{ 0 , 1 \} ^ { H \times W }$ whose spatial support is compatible with the background scene. Given the resulting M, the background $I _ { b g }$ , and the reference object $I _ { s u b } .$ , the generation process synthesizes the target within the planned region while adapting its appearance to the destination observation. The resulting composite is modeled as

$$
\textit { I } _ { f i n a l } \sim p _ { \theta } \left( \cdot \mathrm { ~ | ~ } I _ { b g } , M , I _ { s u b } \right) .\tag{1}
$$

## B. Subject-Driven Condition Injection

To integrate the subject reference into the DiT architecture, we follow the in-context conditioning paradigm of OminiControl [18]. Latent tokens derived from $I _ { s u b }$ are concatenated with the noisy image tokens to form a joint sequence, allowing the native multimodal attention to model their interaction without an auxiliary reference encoder. Since subject-driven generation requires identity preservation rather than strict spatial correspondence, we adopt shifted positional encoding for the reference tokens. Their position indices are translated by a fixed offset so that the reference and generation tokens occupy disjoint coordinate ranges in the rotary embedding space. This separation weakens direct local correspondence between the two token groups and encourages the attention mechanism to capture subject-level semantic information.

The resulting in-context representation provides the identity condition used throughout our framework. During generation, the reference tokens preserve the semantic identity of $I _ { s u b } .$ while the target background supplies the scene context required to adapt the inserted object to the destination observation, as detailed in Section IV-C.

## IV. METHODOLOGY

## A. Overview

Remote sensing object insertion requires the synthesized target to remain compatible with its destination scene throughout both placement and generation. A plausible location alone is insufficient if the generated appearance remains inconsistent with the target observation, while visually coherent generation can still exhibit local artifacts when the inserted content does not match its surroundings. To address these coupled requirements, PDA++ organizes object insertion into two stages. Stage I performs scene-aware planning, and Stage II adapts the inserted content to the target observation through Decoupling and Assimilation. The overall procedure is summarized as

$$
\begin{array} { r l r } & { \mathbf { p } ^ { * } = \arg \underset { \mathbf { x } , \theta } { \operatorname* { m a x } } A ( \mathbf { x } , \theta ; c ) , } & \\ & { \tilde { I } _ { b g } = \mathcal { T } ( I _ { b g } , M , M _ { s e g } ) , } & \\ & { \hat { v } _ { t } = v _ { \theta } ( x _ { t } , t ) - \lambda ( t ) \nabla _ { x _ { t } } \mathcal { L } _ { t e x } ( x _ { t } ) . } & \end{array}\tag{2}
$$

The first line describes Planning, where the affordance field A determines the object pose $\mathbf { p } ^ { * }$ . The planned pose defines the insertion region M and is used to transform the subject mask into $M _ { s e g } .$ . The second line constructs the pose-conditioned background $\tilde { I } _ { b g }$ , which communicates the planned spatial support while retaining the surrounding scene as observation context. The final line describes texture-guided generation. The flow velocity $\mathbf { v } _ { \theta }$ is corrected by the gradient of $\mathcal { L } _ { \mathrm { t e x } }$ , whose statistics are computed from the inserted region and its local neighborhood $M _ { e n v }$ . The following sections describe each component in detail.

![](images/d4dff1d6711c06f72f59e23f2457c52d66bed3e9ca667cc523930fa943548a5f.jpg)  
Fig. 2. Overall pipeline of PDA++. Stage I performs scene-aware Planning (P) to determine the insertion pose. Stage II performs Decoupling (D) and Assimilation (A) to adapt the inserted target to the destination observation and improve its local compatibility with the surrounding scene.

Stage I (Planning). Planning determines where the target can be inserted and how it should be oriented in the scene. The proposed Affordance Field extends the clearance-based criterion of the conference version by incorporating scene structure and object-scale compatibility. The resulting pose $\mathbf { p } ^ { * }$ provides the spatial constraint used by the subsequent generation stage. Details are provided in Section IV-B.

Stage II (Generation). Generation takes the planned pose together with the reference subject and target background as conditions. Decoupling converts the planned geometry into pixel-level guidance through the pose-conditioned background $\tilde { I } _ { b g }$ . This representation preserves the surrounding observation context, allowing the reference identity to be retained while its appearance is adapted to the destination scene. The same construction also provides the object mask $M _ { s e g }$ for downstream segmentation augmentation. Assimilation subsequently acts on the sampling trajectory through $\mathcal { L } _ { \mathrm { t e x } } .$ , which compares local texture statistics within M and $M _ { e n v }$ and corrects residual appearance discrepancies. Details of these two components are given in Sections IV-C and IV-D, respectively.

The two stages are modular. Planning supplies the geometry required for generation, whereas the generation process does not depend on how the candidate pose is obtained. This separation allows the same generation formulation to operate with different planning strategies and across different remote sensing modalities.

Algorithm 1: The proposed PDA++ method   
Input: Background ${ \mathit { I } } _ { b g } ,$ subject $\overline { { I _ { s u b } } }$ with mask $\overline { { M _ { s u b } } } ,$   
category set C<sub>category</sub>, class c   
Output: Composite image I<sub>final</sub>   
// Stage I Planning   
1 $M _ { v a l i d } \stackrel { - } {  } \mathrm { S e m a n t i c P a r s e } ( I _ { b g } , \mathcal { C } _ { c a t e g o r y } )$   
2 $\mathcal { A } ( \mathbf { x } , \theta ; c ) \gets \mathcal { A } _ { \mathrm { g e o } } ( \mathbf { x } ) \mathcal { A } _ { \mathrm { s t r u c t } } ( \mathbf { x } , \theta ) \bar { \mathcal { A } } _ { \mathrm { s c a l e } } ( \mathbf { x } ; c )$   
3 $\mathbf { p } ^ { * }  \arg \operatorname* { m a x } _ { \mathbf { x } , \theta } \mathcal { A } ( \mathbf { x } , \theta ; c )$   
4 $\overset { \cdot } { M }  \mathrm { B o x M a s k } ( \mathbf { p } ^ { * } ) , \quad \overset { \cdot } { M } _ { s e g }  \mathrm { T r a n s f o r m } ( M _ { s u b } , \mathbf { p } ^ { * } )$   
5 $M _ { e n v } \gets \mathrm { D i l a t e } ( M ) \setminus M$   
// Stage II Generation   
6 $\tilde { I } _ { b g }  \mathcal { T } ( I _ { b g } , M , M _ { s e g } )$   
7 $z _ { b g } \gets \mathrm { V A E . E n c } ( \tilde { I } _ { b g } ) , \quad z _ { s u b } \gets \mathrm { V A E . E n c } ( I _ { s u b } )$   
8 $T \gets 5 0 , \quad 1 = t _ { 0 } > t _ { 1 } > \cdots > t _ { T } = 0$   
9 $x _ { t _ { 0 } } \sim \mathcal { N } ( 0 , \mathbf { I } )$   
10 for $i = 0 , 1 , \ldots , T - 1$ do   
11 $v _ { \theta } \gets \mathrm { D i T } ( x _ { t _ { i } } , t _ { i } , z _ { b g } , z _ { s u b } )$   
12 $\mathcal { G } _ { o b j }  \mathcal { G } ( \dot { x } _ { t _ { i } } ; M ) , \quad \mathcal { G } _ { e n v }  \mathcal { G } ( x _ { t _ { i } } ; M _ { e n v } )$   
13 $\mathcal { L } _ { t e x }  \mathcal { W } _ { \varepsilon } ( \mathcal { G } _ { o b j } , \mathcal { G } _ { e n v } )$   
14 $\hat { v } _ { t _ { i } } \gets v _ { \theta } - \lambda ( t _ { i } ) \nabla _ { x _ { t _ { i } } } \mathcal { L } _ { t e x }$   
15 $x _ { t _ { i + 1 } } \gets \mathrm { O D E S o l v e r } ( x _ { t _ { i } } , \hat { v } _ { t _ { i } } , t _ { i } , t _ { i + 1 } )$   
16 end   
17 $I _ { f i n a l }  \mathrm { V A E . D e c } ( x _ { t _ { T } } )$   
18 return $I _ { f i n a l }$

## B. Affordance-Aware Scene Layout Planner (P)

a) Semantic-Geometric Parsing: We first identify the functional regions of the scene. SegEarthOV3 [64], an openvocabulary segmentation model based on SAM3 [65], is applied to the background $I _ { b g } .$ . Given a target-specific category set $\mathcal { C } _ { c a t e g o r y } .$ , the segmentation map $S$ is converted into a binary valid-region mask $M _ { v a l i d } \in \{ 0 , 1 \} ^ { H \times W }$

$$
M _ { v a l i d } ( u , v ) = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } S ( u , v ) \in { \mathcal C } _ { c a t e g o r y } , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

Here, $( u , v )$ indexes a pixel and $S ( u , v )$ denotes its predicted semantic class. The resulting mask restricts object placement to semantically permissible regions.

b) Affordance Field Formulation: The conference version determines the insertion pose using clearance alone. It places the object near the maximum of the distance field and derives its orientation from the corresponding field gradient. This criterion does not account for whether the available spatial support is appropriate for the target scale. Moreover, the distance gradient mainly reflects boundary geometry and becomes unstable near the clearance maximum. We therefore introduce an affordance field $\boldsymbol { \mathcal { A } } ( \mathbf { x } , \boldsymbol { \theta } ; c )$ to evaluate the suitability of a candidate pose. For a position $\mathbf { x } = ( u , v ) \in M _ { v a l i d }$ and orientation $\theta ,$ the field is conditioned on target class c and defined as

$$
\begin{array} { r } { \mathcal { A } ( \mathbf { x } , \theta ; c ) = \mathcal { A } _ { \mathrm { g e o } } ( \mathbf { x } ) \cdot \mathcal { A } _ { \mathrm { s t r u c t } } ( \mathbf { x } , \theta ) \cdot \mathcal { A } _ { \mathrm { s c a l e } } ( \mathbf { x } ; c ) , } \end{array}\tag{4}
$$

with the optimal pose given by

$$
\mathbf { p } ^ { * } = ( u ^ { * } , v ^ { * } , \theta ^ { * } ) = \arg \operatorname* { m a x } _ { \mathbf { x } , \theta } \mathcal { A } ( \mathbf { x } , \theta ; c ) .\tag{5}
$$

Here, $\boldsymbol { \mathcal { A } } _ { \mathrm { g e o } }$ measures geometric clearance. Structural compatibility is represented by $\mathcal { A } _ { \mathrm { s t r u c t } } .$ while $\mathcal { A } _ { \mathrm { s c a l e } }$ measures spatial support for the target scale. Each sub-field lies in [0, 1]. Their multiplicative combination assigns high affordance only to poses that remain compatible with both the local scene structure and the spatial requirement of the target. No learned fusion weights are required, and the complete field is constructed without training.

c) Clearance and Structural Alignment: The clearance field $\mathcal { A } _ { \mathrm { g e o } }$ retains the criterion used in the conference version. We first compute the Euclidean distance transform

$$
\mathcal { D } ( \mathbf { x } ) = \operatorname* { i n f } _ { \mathbf { b } \in \partial M _ { v a l i d } } \| \mathbf { x } - \mathbf { b } \| _ { 2 } ,\tag{6}
$$

which measures the minimum $L _ { 2 }$ distance from position x to the boundary $\partial M _ { v a l i d }$ . The normalized clearance field is

$$
\mathcal { A } _ { \mathrm { g e o } } ( \mathbf { x } ) = \frac { \mathcal { D } ( \mathbf { x } ) } { \operatorname* { m a x } _ { \mathbf { x ^ { \prime } } } \mathcal { D } ( \mathbf { x ^ { \prime } } ) } ,\tag{7}
$$

so that $\mathcal { A } _ { \mathrm { g e o } } \in [ 0 , 1 ]$

The structural field $\mathcal { A } _ { \mathrm { s t r u c t } }$ estimates orientation directly from image structure rather than from ∇D. We compute a multi-scale structure tensor of the background as

$$
\mathbf { J } ( \mathbf { x } ) = \sum _ { \sigma } w _ { \sigma } G _ { \sigma } * \big ( \nabla I _ { b g } ( \mathbf { x } ) \nabla I _ { b g } ( \mathbf { x } ) ^ { \top } \big ) ,\tag{8}
$$

where $\nabla I _ { b g } \in \mathbb { R } ^ { 2 }$ denotes the spatial image gradient. $G _ { \sigma }$ is a Gaussian smoothing window at scale σ with weight $w _ { \sigma } .$ , and ∗ denotes spatial convolution. The resulting tensor $\mathbf { J } ( \mathbf { x } ) \in$ $\mathbb { R } ^ { 2 \times 2 }$ is symmetric. Its eigendecomposition provides the local structural orientation

$$
\begin{array} { r } { \theta _ { \mathrm { t e x } } ( \mathbf { x } ) = \frac { 1 } { 2 } \operatorname { a t a n 2 } ( 2 J _ { x y } , J _ { x x } - J _ { y y } ) + \frac { \pi } { 2 } , } \end{array}\tag{9}
$$

and the corresponding coherence

$$
c _ { \mathrm { c o h } } ( \mathbf { x } ) = \frac { \lambda _ { 1 } - \lambda _ { 2 } } { \lambda _ { 1 } + \lambda _ { 2 } + \epsilon } .\tag{10}
$$

The eigenvalues satisfy $\lambda _ { 1 } \ \geq \ \lambda _ { 2 } \ \geq \ 0 .$ , and ϵ is a small constant for numerical stability. The orientation $\theta _ { \mathrm { t e x } }$ follows the direction of least intensity variation and therefore tends to align with locally elongated structures. The coherence $c _ { \mathrm { c o h } } \in [ 0 , 1 ]$ measures the reliability of this orientation estimate. We then define

$$
\mathcal { A } _ { \mathrm { s t r u c t } } ( \mathbf { x } , \theta ) = \operatorname* { m a x } \big ( 0 , \cos ( 2 ( \theta - \theta _ { \mathrm { t e x } } ( \mathbf { x } ) ) ) \big ) \cdot c _ { \mathrm { c o h } } ( \mathbf { x } ) ,\tag{11}
$$

where the factor 2 accounts for the π-periodicity of object orientation. This field provides orientation cues from elongated infrastructure such as runway and dock structures, complementing the clearance information supplied by $\mathcal { A } _ { \mathrm { g e o } }$

d) Class Scale Affordance: The conference planner does not explicitly account for the spatial support required by different target classes. We introduce $A _ { \mathrm { s c a l e } }$ as a soft accommodation prior for this purpose. Let $( W _ { c } , H _ { c } )$ denote the pixel dimensions of a class-c object, with reference radius

$$
\begin{array} { r } { r _ { c } = \frac { 1 } { 2 } \operatorname* { m i n } ( W _ { c } , H _ { c } ) . } \end{array}\tag{12}
$$

The valid-region mask is eroded at several radii derived from $r _ { c } .$ The resulting masks are aggregated into

$$
\mathcal { A } _ { \mathrm { s c a l e } } ( \mathbf { x } ; c ) = \frac { 1 } { \sum _ { k } \omega _ { k } } \sum _ { k } \omega _ { k } \left( M _ { v a l i d } \ominus B _ { s _ { k } r _ { c } } \right) ( \mathbf { x } ) .\tag{13}
$$

Here, $\ominus$ denotes morphological erosion and $B _ { r }$ is a disk structuring element with radius $^ { r } \cdot$ The factors $s _ { k }$ determine the erosion scales, while $\omega _ { k }$ specifies their contribution to the aggregated score. A high value indicates that the local region provides stable support for an object of the target scale. Since circular erosion is used only as a rotation-independent soft prior, the exact oriented footprint is examined separately during pose optimization. The dimensions $( W _ { c } , H _ { c } )$ follow the class-relative scale estimation used in the conference version.

e) Pose Optimization and Generalization: Direct joint optimization over $( \mathbf { x } , \theta )$ is unnecessary because the affordance field has a factorized structure. Both $\mathcal { A } _ { \mathrm { g e o } }$ and $\mathcal { A } _ { \mathrm { s c a l e } }$ depend only on spatial position. We therefore extract local maxima of $\mathcal { A } _ { \mathrm { g e o } } \mathcal { A } _ { \mathrm { s c a l e } }$ as candidate locations and rank them using the complete affordance score. A small candidate budget is retained for efficient evaluation. At each candidate location, the preferred orientation is obtained in closed form as

$$
\theta ^ { * } = \theta _ { \mathrm { t e x } } ( \mathbf { x } ) ,\tag{14}
$$

for which $\mathcal { A } _ { \mathrm { s t r u c t } }$ reaches $c _ { \mathrm { c o h } } ( \mathbf { x } )$

Each candidate is subsequently subjected to a hard geometric feasibility test. Its oriented footprint must remain sufficiently inside $M _ { v a l i d }$ without intersecting previously occupied regions. This verification complements the soft scale prior and removes geometrically invalid placements. The feasible candidate with the highest affordance score is selected as $\mathbf { p } ^ { * }$

The conference planner is recovered when $\mathcal { A } _ { \mathrm { s t r u c t } } \equiv 1$ and $\mathcal { A } _ { \mathrm { s c a l e } } \equiv 1$ , in which case A reduces to $\mathcal { A } _ { \mathrm { g e o } }$ . The proposed planner remains training-free and independent of the subsequent generation process. It can therefore be applied across different remote sensing modalities without retraining.

## C. Pose-Conditioned Decoupling (D)

a) Pose-Conditioned Background Construction: A key interface between planning and generation is how the planned pose is communicated to the generator while retaining the context of the target scene. The conference version represents the insertion region only through a bounding-box condition, which provides limited information about the spatial support determined by the planner. In PDA++, we instead introduce a pixel-level representation of the planned pose. Given the reference object mask, we transform it according to $\mathbf { p } ^ { * } = { }$ $( u ^ { * } , v ^ { * } , \theta ^ { * } )$ and obtain the corresponding instance mask $M _ { s e g }$ The pose-conditioned background is then constructed as

$$
\tilde { I } _ { b g } = \mathcal { T } ( I _ { b g } , M , M _ { s e g } ) ,\tag{15}
$$

where $\tau ( \cdot )$ embeds the transformed object support within the insertion region M while preserving the surrounding scene context. The resulting $\tilde { I } _ { b g }$ is encoded by the VAE and introduced through the existing in-context conditioning pathway, requiring no modification to the generative architecture. This representation conveys the planned pose at pixel level and provides the generator with the environmental context needed to synthesize the object consistently with its destination scene. Compared with box-level conditioning, it therefore offers more precise control over the spatial extent of the inserted content.

The instance mask $M _ { s e g }$ also provides pixel-level object support for every synthesized sample. As a result, the generated images can be directly paired with their corresponding masks, extending synthetic augmentation from object detection to segmentation without additional manual annotation.

b) Observation-State Adaptation through Scene Context: A correctly placed object must also conform to the observation characteristics of the target scene while preserving its semantic identity. The conference version addresses this issue through an explicit spectral-adaptation module based on frequency decomposition and environment-conditioned modulation. In PDA++, the target scene itself instead serves as evidence of the desired observation state. The pose-conditioned background $\tilde { I } _ { b g }$ retains the local environmental context around the planned insertion region, allowing reference identity to interact directly with information from the destination observation during generation.

This formulation is motivated by recent evidence that pretrained diffusion models encode scene-level environmental priors useful for appearance adaptation [39]–[41]. Rather than reproducing the appearance inherited from the source observation, the generator can therefore re-image the reference object according to the sensing characteristics represented by the target scene. This formulation is particularly relevant to remote sensing imagery, where appearance may vary substantially across imaging modalities and acquisition settings.

c) Decoupling Subject Identity from Observation State: Decoupling separates the subject information that should be preserved from the scene-dependent appearance that should adapt. The reference tokens provide semantic identity through the in-context conditioning scheme introduced in Section III-B. The pose-conditioned background $\tilde { I } _ { b g }$ provides the spatial support together with the observation context of the destination scene. Their interaction allows the generator to retain the identity of the reference object while adapting its rendered appearance to the target observation.

Lightweight LoRA adapters specialize the generative backbone to remote sensing imagery without introducing a dedicated observation-state adaptation branch. Compared with the conference pipeline, this design replaces explicit spectral adaptation with scene-conditioned generation and integrates identity preservation directly with environment-aware appearance adaptation. As shown in Section V-A3, reintroducing the conference adaptation module on top of this formulation provides no additional benefit and can slightly reduce generation consistency.

## D. Manifold-Aligned Texture Assimilation (A)

a) Local Neighborhood Definition: Texture compatibility is primarily determined by the local environment around the inserted object. Using distant background regions as reference may introduce statistics that are unrelated to the immediate surroundings. We therefore define a local environmental region $M _ { e n v }$ around the insertion mask M. Given the planned pose $\mathbf { p } ^ { * } = ( u ^ { * } , v ^ { * } , \theta ^ { * } )$ and the corresponding subject size, we obtain $M _ { e n v }$ from the morphologically dilated mask $M _ { d i l a t e d }$ as

$$
M _ { e n v } = M _ { d i l a t e d } \backslash M .\tag{16}
$$

The resulting annular region contains the neighboring background pixels immediately outside the insertion boundary and serves as the reference region for texture assimilation.

b) From Single-Statistic Matching to Manifold Alignment: The conference version enforces local texture consistency by matching one Gram matrix from the inserted region with another from its neighborhood using an $L _ { 2 }$ objective. Such a representation summarizes each region with a single second-order statistic and therefore loses variations in texture organization across spatial scales. Directly comparing the two matrices also reduces the local texture distribution to one global correspondence. We instead represent each region with a collection of spatial Gram statistics computed at multiple scales. Texture assimilation is then formulated as distribution alignment between the two collections using optimal transport. This formulation remains an inference-time guidance mechanism and introduces no additional trainable parameters.

c) Multi-Scale Spatial Gram Set: At inference timestep $t ,$ let $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { C \times h \times w }$ denote the noisy latent, where C is the number of channels and $h \times w$ is its spatial resolution. For a region mask $M _ { r } ,$ , with $M _ { r }$ corresponding to either M or $M _ { e n v } ,$ we partition the masked latent into $P _ { s }$ spatial patches at scale $s . \mathrm { ~ A ~ }$ normalized Gram matrix is computed for each patch as

$$
G _ { p } ^ { ( s ) } ( \boldsymbol { x } _ { t } ) = \frac { 1 } { | \Omega _ { p } ^ { s } | } \sum _ { i \in \Omega _ { p } ^ { s } } { f _ { i } { f _ { i } ^ { \top } } } \in \mathbb { R } ^ { C \times C } ,\tag{17}
$$

where $\Omega _ { p } ^ { s }$ denotes the spatial positions belonging to patch $p$ at scale s. The vector $f _ { i } \in \mathbb { R } ^ { C }$ contains the latent features at position i. Normalization by |Ω<sup>s</sup>| reduces the dependence of the Gram statistic on patch cardinality.

Aggregating the patch statistics over $S$ scales gives

$$
\mathcal { G } ( x _ { t } ; M _ { r } ) = \left\{ G _ { p } ^ { ( s ) } ( x _ { t } ) ~ \Big | ~ s = 1 , \ldots , S , ~ p = 1 , \ldots , P _ { s } \right\} .\tag{18}
$$

Larger spatial partitions describe broader texture organization, while finer partitions retain more localized variations. We define the resulting representations for the inserted region and its neighborhood as

$$
\mathcal { G } _ { o b j } = \mathcal { G } ( x _ { t } ; M ) , \qquad \mathcal { G } _ { e n v } = \mathcal { G } ( x _ { t } ; M _ { e n v } ) .\tag{19}
$$

d) Optimal-Transport Texture Loss: We regard $\mathcal { G } _ { o b j }$ and $\mathcal { G } _ { e n v }$ as empirical distributions of local second-order texture statistics. Their discrepancy is measured with an entropyregularized optimal transport distance

$$
\mathcal { L } _ { t e x } = \mathcal { W } _ { \varepsilon } \big ( \mathcal { G } _ { o b j } , \mathcal { G } _ { e n v } \big ) = \operatorname* { m i n } _ { \pi \in \Pi } \sum _ { a , b } \pi _ { a b } \big \| \boldsymbol { G } _ { a } - \boldsymbol { G } _ { b } \big \| _ { F } ^ { 2 } - \varepsilon H ( \pi ) ,\tag{20}
$$

where $G _ { a } \in \mathcal { G } _ { o b j }$ and $G _ { b } \in \mathcal { G } _ { e n v }$ denote individual Gram matri ces. The set Π contains transport plans with uniform marginals. The pairwise transport cost is measured by the squared Frobenius distance. We define $\begin{array} { r } { H ( \pi ) = - \sum _ { a , b } { \pi } _ { a b } \log \pi _ { a b } } \end{array}$ as the entropy of the transport plan, with $\varepsilon > 0$ controlling the strength of entropy regularization. The transport plan $\pi$ is computed using differentiable Sinkhorn iterations. This formulation aligns the distributions represented by the multiscale Gram sets rather than directly matching a single regional statistic.

e) Gradient-Guided Flow Matching: The texture objective is introduced into the sampling trajectory as energy-based guidance, following the general guidance strategy of the conference version. Let $v _ { \theta } ( x _ { t } , t )$ denote the velocity field predicted by the flow-matching backbone [66] at timestep $t \in [ 0 , 1 ]$ . We define the guided velocity as

$$
\hat { v } _ { t } = v _ { \theta } ( x _ { t } , t ) - \lambda ( t ) \nabla _ { x _ { t } } \mathcal { L } _ { t e x } ( x _ { t } ) ,\tag{21}
$$

where λ(t) controls the strength of texture guidance. The guidance is activated only during the early sampling phase, when the latent representation primarily determines coarse appearance organization. It is disabled at later timesteps to preserve the subsequent synthesis of fine details.

The conference texture objective can be recovered from this formulation by setting $S = 1$ and $P _ { 1 } = 1$ , such that each region is represented by a single Gram matrix. When the entropy regularization becomes negligible, the transport objective reduces to the squared Frobenius distance between the object region and its local neighborhood

$$
\mathcal { L } _ { t e x } = \Vert G ( x _ { t } ; M ) - G ( x _ { t } ; M _ { e n v } ) \Vert _ { F } ^ { 2 } .\tag{22}
$$

## E. Recursive Multi-Object Synthesis

While PDA++ is formulated for single-object insertion in Algorithm 1, practical data augmentation may require several targets to be synthesized within the same scene. Repeatedly applying the single-object pipeline independently is not sufficient because candidate placements may overlap and subsequent insertions cannot account for content generated in earlier steps. We therefore extend PDA++ with the recursive synthesis procedure summarized in Algorithm 2. The extension retains the same environment-aware formulation as the singleobject pipeline while coordinating the planned poses and progressively updating the scene during generation.

Algorithm 2: Recursive Multi-Object Synthesis   
Input: Raw background $\overline { { I _ { b g } ^ { r a w } } }$ , subject $I _ { s u b }$ with mask $M _ { s u b } ,$   
target count N, class c   
Output: Composite image $I _ { f i n a l }$   
// Stage I Batch Layout Planning   
1 $\begin{array} { r } { \mathcal { Q }  [ ] , \quad M _ { \mathrm { o c c } }  \mathbf { 0 } } \end{array}$   
2 for $k \gets 1$ to N do   
3 $\mathbf { p } _ { k } \gets \arg \operatorname* { m a x } _ { \mathbf { x } , \theta } \mathcal { A } ( \mathbf { x } , \theta ; c )$   
$/ /$ sub $\mathsf { j e c t ~ \mathsf { \doteq } ~ \mathsf { M } } ( \mathbf { x } , \theta ) \odot \boldsymbol { M } _ { \mathrm { o c c } } = \mathbf { 0 }$   
4 if $\bar { \mathbf { \zeta } } A ( \mathbf { p } _ { k } ; \bar { c } ) < \tau$ then   
5 break   
6 end   
7 $M _ { k } \gets M ( \mathbf { p } _ { k } )$   
8 $M _ { \mathrm { { o c c } } }  \dot { M } _ { \mathrm { { o c c } } } \cup M _ { k }$   
9 Append(Q, p<sub>k</sub>)   
10 end   
$/ /$ Stage II Recursive Generation   
11 $I _ { \mathrm { c u r r } }  \mathbf { \overline { { I } } } _ { b g \_ s } ^ { r a w }$   
12 while $\mathcal { Q } \neq [ ]$ do   
13 p<sub>k</sub> ← PopFront(Q)   
14 $M _ { k } \gets \bar { M } ( \mathbf { p } _ { k } ) , \quad \bar { M } _ { \mathrm { s e g } } ^ { k } \gets \mathrm { T r a n s f o r m } ( M _ { s u b } , \mathbf { p } _ { k } )$   
15 $\tilde { I } _ { \mathrm { c u r r } } \gets \mathcal { T } ( I _ { \mathrm { c u r r } } , M _ { k } , \tilde { M } _ { \mathrm { s e g } } ^ { k } )$   
16 I<sub>curr</sub> ← DecoupleAssimilate $( \tilde { I } _ { \mathrm { c u r r } } , I _ { s u b } , \mathbf { p } _ { k } )$   
17 end   
18 $I _ { f i n a l }  I _ { \mathrm { c u r r } }$   
19 return $I _ { f i n a l }$

a) Batch Layout Planning: Given a desired number of targets N, we evaluate the affordance field $\boldsymbol { \mathcal { A } } ( \mathbf { x } , \boldsymbol { \theta } ; c )$ defined in Section IV-B over the valid region. Rather than using only the highest-scoring pose, the planner iteratively constructs a pose queue $\mathcal { Q } = \{ \mathbf { p } _ { 1 } , \ldots , \mathbf { p } _ { N ^ { \prime } } \}$ while maintaining a binary occupancy mask $M _ { \mathrm { o c c } } . \mathrm { A t }$ iteration k, the next pose is selected from the currently available region according to

$$
{ \bf p } _ { k } = \arg \operatorname* { m a x } _ { { \bf x } , \theta } \mathcal { A } ( { \bf x } , \theta ; c ) \quad \mathrm { s . t . } \quad M ( { \bf x } , \theta ) \odot M _ { \mathrm { o c c } } = { \bf 0 } ,\tag{23}
$$

where $M ( \mathbf { x } , \theta )$ denotes the target footprint associated with the candidate pose $( \mathbf { x } , \theta )$ . The constraint requires the candidate footprint to remain disjoint from the occupied region. Once $\mathbf { p } _ { k }$ is accepted, its footprint is merged into $M _ { \mathrm { o c c } }$ and excluded from subsequent planning. The procedure continues until the required number of poses has been obtained or no feasible candidate remains above the affordance threshold. The resulting poses therefore follow the same scene-compatibility criterion as the single-object planner while remaining mutually nonoverlapping.

b) Recursive Generative Injection: The planned targets are synthesized sequentially so that each insertion can respond to the current scene state. We initialize the process with $I _ { \mathrm { c u r r } } ^ { ( 0 ) } =$ $I _ { b q } ^ { r a w }$ . For a planned pose $\mathbf { p } _ { k }$ , the transformed subject mask $\breve { M } _ { \mathrm { s e g } } ^ { k }$ defines the object support within the insertion region $M _ { k }$ The corresponding pose-conditioned scene is written as

$$
\tilde { I } _ { c u r r } ^ { ( k ) } = \mathcal { T } \left( I _ { c u r r } ^ { ( k - 1 ) } , M _ { k } , M _ { s e g } ^ { k } \right) .\tag{24}
$$

Generation then follows the same Decoupling and Assimilation procedure as Algorithm 1. The reference object supplies identity information, while the current scene provides the spatial support and observation context required for synthesis. MATA further reduces local texture discrepancies during sampling. After each insertion, the resulting composite replaces the previous scene state and serves as the condition for the next target. The k-th insertion therefore depends on the environment produced by all preceding steps, allowing later targets to remain compatible with the evolving scene. This recursive formulation supports coherent multi-object synthesis and enables efficient construction of synthetic samples for downstream detection and segmentation.

GT  
GT Zoom  
MimicBrush  
UniCombine  
ACE++  
OmniPaint  
Insert Anything  
PDA  
PDA++  
![](images/40242f67367a7e4f31bca79aeb87c9e8e3c86cc71b1c37583133116b1d778575.jpg)  
Fig. 3. Qualitative comparison of object insertion. We compare our PDA++ with diffusion-based methods.

TABLE I  
QUANTITATIVE COMPARISON WITH STATE-OF-THE-ART METHODS FOR OPTICAL REMOTE SENSING OBJECT INSERTION. WHOLE-IMAGE METRICS EVALUATE GLOBAL VISUAL FIDELITY, WHILE INSERTION-REGION METRICS MORE DIRECTLY ASSESS THE REALISM AND ENVIRONMENTAL COMPATIBILITY OF THE SYNTHESIZED OBJECT.
<table><tr><td rowspan="2">Methods</td><td colspan="4">Whole Image</td><td colspan="3">Insertion Region</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>AnyDoor [13]</td><td>24.24</td><td>0.8501</td><td>0.1040</td><td>21.47</td><td>15.70</td><td>0.4411</td><td>0.2897</td></tr><tr><td>MimicBrush [14]</td><td>24.81</td><td>0.9337</td><td>0.0582</td><td>21.85</td><td>14.36</td><td>0.3676</td><td>0.3478</td></tr><tr><td>Qwen Image Edit [17]</td><td>18.41</td><td>0.7912</td><td>0.2044</td><td>46.08</td><td>10.72</td><td>0.2574</td><td>0.5698</td></tr><tr><td>UniCombine [15]</td><td>24.97</td><td>0.8869</td><td>0.0781</td><td>22.06</td><td>15.89</td><td>0.4566</td><td>0.2827</td></tr><tr><td>ACE++ [67]</td><td>17.44</td><td>0.3924</td><td>0.2659</td><td>29.24</td><td>15.89</td><td>0.4167</td><td>0.2757</td></tr><tr><td>OmniPaint [68]</td><td>24.32</td><td>0.8283</td><td>0.0871</td><td>18.14</td><td>16.83</td><td>0.4965</td><td>0.2156</td></tr><tr><td>Insert Anything [16]</td><td>26.12</td><td>0.8893</td><td>0.0707</td><td>11.54</td><td>18.07</td><td>0.5463</td><td>0.1561</td></tr><tr><td>OminiControl [18]</td><td>25.22</td><td>0.8603</td><td>0.0839</td><td>12.05</td><td>17.90</td><td>0.5396</td><td>0.1669</td></tr><tr><td>PDA</td><td>28.41</td><td>0.8901</td><td>0.0601</td><td>9.732</td><td>20.87</td><td>0.6249</td><td>0.1247</td></tr><tr><td>PDA++ (Ours)</td><td>32.49</td><td>0.9457</td><td>0.0252</td><td>6.280</td><td>24.91</td><td>0.7644</td><td>0.0758</td></tr></table>

The best results are highlighted in bold, and the second-best are underlined.

## V. EXPERIMENTS

## A. Optical Experiments

1) Experimental Setup: We train and evaluate the optical insertion model using paired samples constructed from SAMRS [69] and iSAID [70]. For SAMRS, we use the FAIR1M [71] subset, while iSAID is derived from DOTA [72]. Instance segmentation masks and oriented bounding box annotations are used to construct the training pairs. Valid target objects are extracted and placed on a 512 × 512 canvas, with mild boundary smoothing applied to reduce edge aliasing. Their corresponding regions in the original images are removed to construct the background conditions. The resulting dataset contains 19,163 training pairs, of which 14,215 are from SAMRS and 4,948 from iSAID. An additional 1,718 pairs are reserved for evaluating insertion quality.

We compare PDA++ with representative diffusion-based editing methods, including AnyDoor [13], MimicBrush [14], Qwen Image Edit [17], UniCombine [15], ACE++ [67], OmniPaint [68], Insert Anything [16], and OminiControl [18]. Generation quality is evaluated at both the whole-image and insertion-region levels using PSNR, SSIM, LPIPS, and FID. As shown in the supplementary material, we further evaluate shadow and radiometric consistency to assess the physical plausibility of the synthesized targets.

Downstream evaluation. We further evaluate whether the synthesized samples benefit downstrea remote sensing recognition. For oriented object detection, the real training set is augmented with generated insertions following the conference protocol, and the change in mAP50 is measured across several oriented detectors. We also extend the evaluation to object segmentation. The pose-conditioned construction produces an instance mask $M _ { s e g }$ together with each synthesized target, allowing the generated images to be paired directly with pixel level supervision. This enables segmentation augmentation beyond the box-level annotations supported by the conference framework. The real training set is augmented with these synthesized image and mask pairs, and performance is evaluated on the hold-out set using mIoU and mAcc. In both settings, results are compared with training on real data alone to measure the contribution of synthetic augmentation.

Implementation details. Our framework is implemented in PyTorch and trained on two NVIDIA A100 GPU with 80GB VRAM. We use the AdamW optimizer with an initial learning rate of $1 \times 1 0 ^ { - 4 }$ . The model is trained for 20,000 steps with a batch size of 4. All images are resized to 512 × 512 during both training and inference.

## 2) Optical Object Insertion Quality:

Quantitative comparison. As shown in Table I and Fig. 3, PDA++ achieves the best performance across all reported metrics. Compared with the conference PDA model, PDA++ reduces whole-image FID from 9.732 to 6.28, corresponding to a 35.5% reduction. The improvement is more pronounced within the insertion region, where PSNR increases from 20.87 to 24.91 and LPIPS decreases from 0.1247 to 0.0758. This dif ference is expected because the redesigned generation process acts primarily on the inserted content and its local environment, whereas the unchanged background contributes substantially to whole-image measurements. The results therefore show that the main improvement occurs within the region most directly affected by object insertion.

Analysis of baseline behavior. General-purpose editors, including AnyDoor, MimicBrush, Qwen Image Edit, UniCom bine, ACE++, and OmniPaint, show limited transferability from natural image editing to overhead imagery. Remote sensing observations differ substantially in spatial organization and imaging characteristics, which are not explicitly considered by these methods. Their generated targets may therefore remain poorly adapted to the destination scene. This effect is particularly evident for MimicBrush. Its whole-image SSIM reaches 0.9337, whereas the insertion-region SSIM decreases to 0.3676. The large discrepancy indicates that whole-image similarity can be dominated by unchanged background content and may overestimate the quality of the synthesized target. We therefore place greater emphasis on insertion-region measurements in the following analysis.

Discussion. Insertion-region results provide a more direct measure of how effectively the generated target adapts to its local environment. Among the competing methods, Insert Anything and OminiControl achieve relatively strong local performance, with region PSNR values of 18.07 and 17.90 and SSIM values of 0.5463 and 0.5396, respectively. These models provide effective structural control, but their generation mechanisms are primarily designed for natural imagery and do not explicitly account for the observation characteristics of remote sensing data. The synthesized target can therefore preserve its reference identity without fully adapting its appearance to the destination observation. This limitation is also reflected in distributional quality. Insert Anything obtains the strongest baseline FID of 11.54, whereas PDA++ reduces it to 6.28.

The Decoupling stage improves this adaptation by conditioning generation on the reference identity together with the pose-conditioned target background. The target background provides precise spatial support while retaining the observation context required for scene-consistent synthesis. This formulation increases region PSNR from 20.87 for PDA to 24.91 and also improves physical consistency. Assimilation further reduces residual local discrepancies by replacing the single-Gram objective of PDA with multi-scale optimal-transport alignment. Region LPIPS consequently decreases from 0.1247 to 0.0758, indicating improved local appearance compatibility. The qualitative examples in Fig. 3 show the same tendency. Across harbor and sports-field scenes, PDA++ produces targets that better match the destination observation and exhibit fewer visible boundary artifacts.

## 3) Ablation Study on Optical Insertion:

Component analysis. We evaluate the major design choices on the optical test set (Table II) and separately examine the effect of the backbone upgrade. In the table, seg. denotes the pose-conditioned background construction, where $M _ { s e g }$ represents the planned object support at pixel level. Replacing only the conference backbone with FLUX.2 reduces wholeimage FID from 9.732 to 7.159 and LPIPS from 0.0601 to 0.0332. The improvement within the insertion region is much smaller. Region PSNR changes from 20.87 to 21.06, while SSIM decreases from 0.6249 to 0.6102. This suggests that the stronger backbone mainly improves global image quality but does not by itself provide substantially better control over the inserted content. The final gains of PDA++ therefore cannot be explained by the backbone replacement alone.

A much larger improvement is obtained after introducing the pose-conditioned background. Compared with plain FLUX.2, region PSNR increases from 20.72 to 24.88, SSIM rises from 0.5943 to 0.7550, and LPIPS decreases from 0.1078 to 0.0763. Whole-image FID also improves from 6.992 to 6.403. These changes show that the pose-conditioned representation affects not only the spatial extent of the synthesized target but also the quality of its local integration. By encoding the planned object support while retaining the surrounding scene context, it provides a more informative condition for adapting the target to the destination observation. The dominant improvement in local insertion quality therefore comes from the redesigned conditioning rather than from the backbone upgrade.

TABLE II  
ABLATION STUDY ON THE OPTICAL EVALUATION SET. WE REPORT BOTH WHOLE-IMAGE AND INSERTION-REGION METRICS TO ANALYZE THECONTRIBUTION OF EACH COMPONENT TO GLOBAL PERCEPTUAL FIDELITY AND LOCAL INSERTION QUALITY, RESPECTIVELY.
<table><tr><td rowspan="2">Method Variant</td><td colspan="4">Whole Image</td><td colspan="3">Insertion Region</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>PDA</td><td>28.41</td><td>0.8901</td><td>0.0601</td><td>9.732</td><td>20.87</td><td>0.6249</td><td>0.1247</td></tr><tr><td>PDA (FLUX.2 backbone)</td><td>29.56</td><td>0.9278</td><td>0.0332</td><td>7.159</td><td>21.06</td><td>0.6102</td><td>0.1034</td></tr><tr><td>FLUX.2</td><td>29.44</td><td>0.9288</td><td>0.0306</td><td>6.992</td><td>20.72</td><td>0.5943</td><td>0.1078</td></tr><tr><td>+ ASA</td><td>29.52</td><td>0.9276</td><td>0.0333</td><td>7.189</td><td>21.00</td><td>0.6077</td><td>0.1045</td></tr><tr><td>+ NATA</td><td>29.52</td><td>0.9289</td><td>0.0304</td><td>6.946</td><td>20.83</td><td>0.5969</td><td>0.1062</td></tr><tr><td>+ MATA</td><td>29.51</td><td>0.9289</td><td>0.0304</td><td>6.925</td><td>20.79</td><td>0.5945</td><td>0.1076</td></tr><tr><td>FLUX.2 (seg.)</td><td>32.46</td><td>0.9456</td><td>0.0253</td><td>6.403</td><td>24.88</td><td>0.7550</td><td>0.0763</td></tr><tr><td>+ ASA</td><td>31.93</td><td>0.9407</td><td>0.0261</td><td>6.541</td><td>24.76</td><td>0.7489</td><td>0.0771</td></tr><tr><td>+ NATA</td><td>32.46</td><td>0.9456</td><td>0.0253</td><td>6.404</td><td>24.83</td><td>0.7541</td><td>0.0770</td></tr><tr><td>+ MATA (PDA++ Ours)</td><td>32.49</td><td>0.9457</td><td>0.0252</td><td>6.280</td><td>24.91</td><td>0.7644</td><td>0.0758</td></tr></table>

Texture assimilation provides a further refinement once reliable spatial conditioning has been established. Applying MATA directly to plain FLUX.2 only slightly changes the results, reducing FID from 6.992 to 6.925 with little improvement in the insertion region. Under pose-conditioned input, however, MATA further reduces FID from 6.403 to 6.280, increases region SSIM from 0.7550 to 0.7644, and lowers LPIPS from 0.0763 to 0.0758. In comparison, NATA remains close to the configuration without texture guidance, with an FID of 6.404 and a region PSNR of 24.83. This suggests that the multi-scale distribution alignment introduced by MATA is better suited to refining the remaining appearance discrepancy after the object support has been established. Its contribution is smaller than that of the pose-conditioned background, but it consistently improves the final configuration and yields the best overall results in the table.

Effect of the conference appearance adaptation module. We further reintroduce ASA to examine whether explicit appearance adaptation remains necessary with FLUX.2. Without pose-conditioned input, ASA slightly improves region PSNR from 20.72 to 21.00 and SSIM from 0.5943 to 0.6077, while whole-image FID increases from 6.992 to 7.189. This indicates that the module can modify the local appearance to some extent, but the resulting change does not translate into a consistent improvement in overall generation quality.

The same tendency becomes more evident after poseconditioned input is introduced. Adding ASA increases FID from 6.403 to 6.541, while region PSNR decreases from 24.88 to 24.76 and SSIM from 0.7550 to 0.7489. The poseconditioned background already exposes the generator to the planned object support together with the observation context of the destination scene. Additional explicit appearance adaptation therefore provides limited benefit under the redesigned formu lation and can interfere with the scene-conditioned generation process. We consequently omit ASA from the final PDA++ configuration.

![](images/75fa4d94ff33430d5ede81e713f165218bd35c3072555a1bd751271637e15f7a.jpg)  
Fig. 4. Case studies of synthetic data under different scenarios.

Overall, the ablation results clarify the contribution of the main design choices. The backbone upgrade primarily improves whole-image fidelity, while the pose-conditioned construction accounts for the dominant improvement within the insertion region. MATA then provides an additional refinement by reducing the residual discrepancy between the synthesized object and its local surroundings. The full PDA++ therefore improves both global generation quality and local insertion fidelity without relying on the conference appearance adaptation module.

Physical consistency. Additional evaluations of shadow and radiometric consistency are provided in the supplementary material and show the same trend as the visual quality results. The pose-conditioned background substantially improves physical consistency, while reintroducing ASA provides no further benefit under the proposed configuration.

4) Downstream Optical Object Recognition: To evaluate the practical value of the synthesized samples, we conduct few-shot oriented object recognition on MAR20-11-FewShot, a benchmark derived from MAR20 [81]. The benchmark contains 11 strategic airframe categories with 30 real images per category, resulting in 330 training images and a highly data-limited setting. The generator is trained only on pairs constructed from SAMRS/FAIR1M and iSAID/DOTA and is applied directly to MAR20 backgrounds. The resulting synthesis is therefore performed in a zero-shot setting with respect to MAR20.

TABLE III  
DOWNSTREAM ORIENTED OBJECT DETECTION PERFORMANCE (MAP50 IN %) ON THE MAR20-11-FEWSHOT BENCHMARK UNDER ZERO-SHOT OPTICAL SYNTHESIS. CP DENOTES COPY-PASTE, CM DENOTES CUTMIX, AND OC DENOTES OMINICONTROL. ∆ OVER REAL DENOTES THE IMPROVEMENT OF PDA++ OVER THE REAL-DATA BASELINE.
<table><tr><td>Detector</td><td>Real</td><td>+CP</td><td>+CM</td><td>+0C</td><td>PDA</td><td>PDA++ (Ours)</td><td>∆ Over Real</td></tr><tr><td>Rotated R-CNN [73]</td><td>58.13</td><td>73.14</td><td>68.24</td><td>55.22</td><td>77.88</td><td>78.24</td><td>+20.11</td></tr><tr><td>Oriented R-CNN [74]</td><td>68.51</td><td>78.01</td><td>71.07</td><td>70.04</td><td>78.59</td><td>78.71</td><td>+10.20</td></tr><tr><td>S2ANet [75]</td><td>55.67</td><td>71.86</td><td>62.77</td><td>62.36</td><td>77.36</td><td>78.13</td><td>+22.46</td></tr><tr><td>YOLO26 [76]</td><td>63.51</td><td>75.54</td><td>76.79</td><td>70.01</td><td>80.26</td><td>81.52</td><td>+18.01</td></tr><tr><td>Avg.</td><td>61.46</td><td>74.64</td><td>69.72</td><td>64.41</td><td>78.52</td><td>79.15</td><td>+17.69</td></tr></table>

The best results are highlighted in bold, and the second-best are underlined.

TABLE IV  
COMPARISON OF SYNTHESIS STRATEGIES ACROSS SEGMENTATION BACKBONES (%). BEST IN BOLD, SECOND BEST UNDERLINED, COMPARED AMONGSTRATEGIES WITHIN EACH COLUMN.
<table><tr><td rowspan="2">Strategy</td><td colspan="2">BiSeNetV2 [77]</td><td colspan="2">PIDNet [78]</td><td colspan="2">SegNeXt [79]</td><td colspan="2">SegFormer [80]</td></tr><tr><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td>Baseline</td><td>39.26</td><td>50.71</td><td>28.49</td><td>39.36</td><td>62.39</td><td>74.02</td><td>60.26</td><td>72.11</td></tr><tr><td>+ CP</td><td>53.98</td><td>65.82</td><td>55.13</td><td>66.55</td><td>70.10</td><td>79.89</td><td>69.40</td><td>80.00</td></tr><tr><td>+ CM</td><td>48.89</td><td>61.06</td><td>47.09</td><td>59.18</td><td>65.10</td><td>75.82</td><td>68.33</td><td>79.00</td></tr><tr><td>PDA++ (Ours)</td><td>65.11</td><td>74.45</td><td>65.18</td><td>75.49</td><td>70.84</td><td>80.57</td><td>71.58</td><td>81.01</td></tr></table>

The best results are highlighted in bold, and the second-best are underlined.

We construct the augmented training set using the recursive multi-object synthesis procedure described in Section IV-E. Copy-Paste, CutMix [82], and OminiControl are included as augmentation baselines. Performance is evaluated with Rotated R-CNN [73], Oriented R-CNN [74], S2ANet [75], and YOLO26 [76].

Fig. 5 shows representative planning results used to construct the synthetic training set. Semantic parsing first restricts the candidate locations in each MAR20 background to the valid region $M _ { v a l i d } .$ . The affordance field A then integrates geometric clearance with scene structure and target-scale information to estimate suitable insertion locations. Regions with high affordance scores provide scene-compatible support for the target and are selected as candidate poses. These poses are subsequently passed to the recursive generation stage, which inserts multiple targets into each background while preventing spatial overlap between accepted placements.

Results. As shown in Table III, Copy-Paste and CutMix provide only modest improvements because the inserted targets are not adapted to the surrounding environment. OminiControl shows less stable downstream performance and underperforms the real-data baseline on some detectors. For example, its mAP50 on Rotated R-CNN is 55.22, compared with 58.13 for real-data training. Although OminiControl uses a similar generative backbone, this result suggests that visual plausibility alone does not guarantee effective augmentation. Synthetic targets that remain incompatible with the destination scene may introduce unreliable supervision. In contrast, PDA++ improves all four detectors and raises the average mAP50 from 61.46 to 79.15, corresponding to an absolute gain of 17.69 points and a relative improvement of 28.8%. It also consistently outperforms the conference PDA framework. Similar improvements across different detector architectures indicate that the benefit mainly originates from the synthetic training samples rather than from a particular detector.

Effect of synthetic ratio. We vary the synthetic-to-real ratio for PDA and PDA++ in Fig. 6 and 7. Introducing synthetic data substantially improves all detectors over the real-data baseline, while PDA++ generally provides higher accuracy than PDA across the evaluated ratios. The advantage is particularly clear when only a limited amount of synthetic data is used. At a ratio of 1:2, PDA++ exceeds PDA by 4.65 mAP50 on YOLO26 and by 4.32 points on S2ANet, indicating that the generated samples provide effective supervision even at relatively low synthetic ratios.

For YOLO26, performance has already entered its optimum range at a ratio of 1:2, after which additional synthetic samples provide little further benefit. The small negative differences of 0.12 points relative to PDA at ratios of 1:4 and 1:5 therefore occur after performance has saturated and do not indicate a meaningful loss in absolute detection accuracy. A similar saturation tendency is observed for the other detectors as the amount of synthetic data increases. Overall, the results suggest that PDA++ achieves most of its downstream benefit with a moderate amount of synthetic data, while further increasing the synthetic proportion leads to diminishing returns.

5) Downstream Optical Instance Segmentation: The poseconditioned construction associates each synthesized target with a pixel-level mask $M _ { s e g }$ that directly represents its spatial support. The generated samples can therefore provide segmentation supervision without additional manual annotation, extending the box-level augmentation supported by the conference framework. We evaluate this capability on the same MAR20-11-FewShot images. Real segmentation annotations are obtained from the FineGrip panoptic annotations of MAR20 [83]. The synthetic images reuse the zero-shot insertions described above and are

$$
M _ { v a l i d }
$$

$$
\mathcal { A } _ { g e o }
$$

$$
c _ { c o h }
$$

$$
\theta _ { t e x }
$$

$$
\mathcal { A } _ { s c a l e }
$$

�  
�∗  
![](images/0328f2e7c4de0105d304abef0f1b0dafb4deaf5d44d7fbb11df35a1a6c1217fd.jpg)  
Fig. 5. Representative examples of affordance-aware layout planning on MAR20 backgrounds. From left to right, we show the input background, semantic valid region $M _ { v a l i d } ,$ clearance field $\boldsymbol { \mathcal { A } } _ { g e o }$ , structural coherence $c _ { c o h } ,$ estimated structural orientation $\theta _ { t e x } ,$ class-scale affordance $A _ { s c a l e } ,$ fused affordance field A, and the resulting planned poses. The planner concentrates candidate placements on spatially feasible and structurally compatible regions before recursive object generation.

![](images/b6eca30db3acc4b32cc238c59c5526cd01952ce166b5898d8d8d60002cb1e220.jpg)  
Fig. 6. Effect of synthetic data ratio on optical recognition performance.

![](images/d8a63db7466cef5d630ebfddbfe16596acc4a41d91a978e70d40a4ba213069ab.jpg)  
Fig. 7. mAP50 improvement of PDA++ over PDA under different synthetic data ratios on optical object recognition.

paired with their corresponding $M _ { s e g }$ masks.

For segmentation, each model is trained using real and synthesized samples at a fixed real-to-synthetic ratio of 1:3. Evaluation is performed on the held-out real test set using mIoU and mAcc. We consider BiSeNetV2 [77], PIDNet [78], SegNeXt [79], and SegFormer [80]. As shown in Table IV, PDA++ improves performance for every segmentation architecture and consistently surpasses Copy-Paste and CutMix. The improvement is especially pronounced for BiSeNetV2, where mIoU increases from 39.26 to 65.11. PIDNet shows a similar increase from 28.49 to 65.18. SegNeXt and SegFormer also benefit consistently from the synthesized training samples. These results confirm that the automatically associated object masks provide effective pixel-level supervision and extend the utility of PDA++ to segmentation augmentation.

## B. SAR Experiments

1) Experimental Setup: We further evaluate PDA++ on SAR imagery, whose image formation and local statistics differ substantially from optical observations. HRSID [84] is used as the main benchmark for ship insertion. The images are divided into $2 5 6 \times 2 5 6$ patches, producing 3,600 training samples and 500 evaluation samples. For downstream evaluation, the model trained on HRSID is directly applied to SSDD [85] without additional training on the target dataset. SSDD contains 50 training images, 100 validation images, and 250 test images, with spatial resolutions ranging from 1 to 15 m. We additionally use WHU-OPT-SAR [86] to evaluate insertion on forest regions with irregular boundaries. Forest regions are removed according to their segmentation annotations and reconstructed from reference observations. This setting contains 8,616 training patches and 500 test patches.

TABLE V  
QUANTITATIVE COMPARISON ON HRSID FOR SAR OBJECT INSERTION. WE REPORT BOTH WHOLE-IMAGE AND INSERTION-REGION METRICS TO EVALUATEGLOBAL GENERATION FIDELITY AND LOCAL INSERTION QUALITY, RESPECTIVELY.
<table><tr><td rowspan="2">Method Variant</td><td colspan="4">Whole Image</td><td colspan="3">Insertion Region</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td><td>FID ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS↓</td></tr><tr><td>PDA</td><td>25.16</td><td>0.7643</td><td>0.0682</td><td>20.03</td><td>15.36</td><td>0.6942</td><td>0.0477</td></tr><tr><td>Flux.2 (seg.)</td><td>27.19</td><td>0.8666</td><td>0.0437</td><td>18.76</td><td>16.17</td><td>0.7280</td><td>0.0433</td></tr><tr><td>+ NATA</td><td>27.17</td><td>0.8660</td><td>0.0446</td><td>18.70</td><td>16.14</td><td>0.7274</td><td>0.0428</td></tr><tr><td>PDA++ (Ours)</td><td>27.26</td><td>0.8666</td><td>0.0431</td><td>18.46</td><td>16.31</td><td>0.7343</td><td>0.0408</td></tr></table>

The best results are highlighted in bold, and the second-best are underlined.

TABLE VI  
DOWNSTREAM SHIP DETECTION PERFORMANCE (MAP50 IN %) ON SSDD UNDER THE ZERO-SHOT SAR AUGMENTATION SETTING. THE GENERATOR IS TRAINED ON HRSID AND DIRECTLY APPLIED TO SSDD WITHOUT ADDITIONAL FINE-TUNING. CP DENOTES COPY-PASTE, CM DENOTES CUTMIX, AND OC DENOTES OMINICONTROL. ∆ OVER REAL DENOTES THE IMPROVEMENT OF PDA++ OVER THE REAL-DATA BASELINE.
<table><tr><td>Detector</td><td>Real</td><td>+CP</td><td>+CM</td><td>+0C</td><td>PDA</td><td>PDA++ (Ours)</td><td>∆ Over Real</td></tr><tr><td>Rotated R-CNN [73]</td><td>77.91</td><td>77.01</td><td>66.01</td><td>78.84</td><td>78.86</td><td>79.13</td><td>+1.22</td></tr><tr><td>Oriented R-CNN [74]</td><td>80.47</td><td>80.00</td><td>80.00</td><td>87.97</td><td>88.60</td><td>88.92</td><td>+8.45</td></tr><tr><td>S2ANet [75]</td><td>77.36</td><td>65.44</td><td>75.29</td><td>77.86</td><td>78.42</td><td>79.05</td><td>+1.69</td></tr><tr><td>YOLO26 [76]</td><td>79.62</td><td>78.93</td><td>79.30</td><td>81.57</td><td>84.28</td><td>84.67</td><td>+5.05</td></tr><tr><td>Avg.</td><td>78.84</td><td>75.35</td><td>75.15</td><td>81.56</td><td>82.54</td><td>82.94</td><td>+4.10</td></tr></table>

The best results are highlighted in bold, and the second-best are underlined.

TABLE VII  
COMPARISON OF SYNTHESIS STRATEGIES ACROSS SEGMENTATION BACKBONES FOR SAR OBJECT SEGMENTATION (%). BEST IN BOLD, SECOND BESTUNDERLINED, COMPARED AMONG STRATEGIES WITHIN EACH COLUMN.
<table><tr><td rowspan="2">Strategy</td><td colspan="2">BiSeNetV2 [77]</td><td colspan="2">PIDNet [78]</td><td colspan="2">SegNeXt [79]</td><td colspan="2">SegFormer [80]</td></tr><tr><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td>Baseline</td><td>67.85</td><td>78.39</td><td>62.58</td><td>74.30</td><td>69.65</td><td>78.25</td><td>70.66</td><td>78.32</td></tr><tr><td>+ CP</td><td>67.63</td><td>81.89</td><td>64.13</td><td>80.11</td><td>71.89</td><td>80.21</td><td>71.77</td><td>80.77</td></tr><tr><td>+ CM</td><td>67.85</td><td>82.11</td><td>64.05</td><td>80.09</td><td>70.36</td><td>79.92</td><td>71.83</td><td>81.04</td></tr><tr><td>PDA++ (Ours)</td><td>72.26</td><td>84.63</td><td>69.82</td><td>83.04</td><td>73.79</td><td>82.38</td><td>74.28</td><td>81.95</td></tr></table>

The best results are highlighted in bold, and the second-best are underlined.

Unless otherwise specified, the SAR experiments follow the implementation protocol used for optical imagery. Insertion quality is measured using PSNR, SSIM, LPIPS, and FID at the whole-image and insertion-region levels. Detection performance is reported with mAP50, while segmentation is evaluated using mIoU and mAcc. Each synthesized sample is paired with the pose-conditioned mask $M _ { s e g }$ and can therefore provide pixel-level supervision without additional annotation. For segmentation, the models are trained with a fixed real-tosynthetic ratio of 1:3 and evaluated on held-out real images.

2) SAR Object Insertion Quality: We first examine insertion quality on HRSID. Representative examples are shown in Fig. 8, and quantitative results are reported in Table V. This experiment evaluates whether the proposed formulation remains effective when the target observation follows a substantially different imaging process from optical imagery.

PDA++ achieves the strongest overall performance on HRSID. The whole-image results reach 27.26 PSNR, 0.8666 SSIM, 0.0431 LPIPS, and 18.46 FID. Within the insertion region, PSNR reaches 16.31, SSIM reaches 0.7343, and LPIPS decreases to 0.0408. Compared with the conference PDA model, whole-image FID decreases from 20.03 to 18.46, while region PSNR increases from 15.36 to 16.31. The improvement indicates that the redesigned generation process remains effective under the observation characteristics of SAR imagery.

The ablation results follow a trend similar to that observed in the optical experiments. Introducing the pose-conditioned background increases region PSNR from 15.36 to 16.17 and SSIM from 0.6942 to 0.7280 relative to PDA. The pixellevel support provides more precise spatial control, while the surrounding SAR scene supplies the observation context needed during generation. MATA provides a further improvement over this configuration. The conference NATA objective slightly reduces whole-image FID but changes region PSNR from 16.17 to 16.14. In contrast, MATA achieves an FID of 18.46 together with a region PSNR of 16.31. This result suggests that distribution alignment based on multi-scale texture statistics is more effective for reducing residual local discrepancies than the original single-Gram objective.

![](images/db7b23214a691c3f9f8e1037ff90743140b5f69f80092847e888bb7970d83175.jpg)  
Fig. 8. Representative examples of SAR ship insertion on HRSID. Placement masks are shown above the generated results, with the corresponding ground truth images provided for reference. PDA++ synthesizes ships at the specified locations while maintaining compatibility with the local SAR appearance.

These results show that the redesigned Decoupling and Assimilation stages also remain effective for SAR imagery. The pose-conditioned representation provides the scene information required for target adaptation, while MATA further improves compatibility with the local background. No modality-specific redesign of the insertion pipeline is required.

3) SAR Downstream Detection: To evaluate the usefulness of the synthesized SAR samples for downstream recognition, we conduct few-shot ship detection on SSDD. The generator is trained on HRSID and directly applied to SSDD, giving a zeroshot cross-dataset evaluation. The resolution variation in SSDD also provides a test of generalization across spatial scales. We compare training on real data alone with Copy-Paste, CutMix, OminiControl, and PDA++ using Rotated R-CNN, Oriented R-CNN, S2ANet, and YOLO26.

As shown in Table VI, PDA++ achieves the best performance for every detector. The average mAP50 increases from 78.84 with real-data training to 82.94, corresponding to an improvement of 4.10 points. PDA++ also outperforms the conference PDA model for all four detectors. Copy-Paste and CutMix provide less stable gains and can decrease accuracy relative to real-data training, indicating that direct composition does not reliably preserve compatibility between the synthesized ship and the SAR background. OminiControl performs more competitively but remains below PDA++ across the evaluated detectors. Since no SSDD images are used to train the generator, the improvement also shows that the synthesis model transfers across datasets without dataset-specific adaptation. Its performance under the resolution variation of SSDD further supports generalization across spatial scales.

4) SAR Downstream Segmentation: We further evaluate ship segmentation on SSDD using the same zero-shot synthetic samples. Each generated image is paired with its automatically obtained mask $M _ { s e g }$ , while the real SSDD images use their native pixel-level annotations. Following the optical protocol, each segmentation model is trained at a fixed real-to-synthetic ratio of 1:3 and evaluated on the real test set. We use BiSeNetV2,

PIDNet, SegNeXt, and SegFormer for this evaluation.

As shown in Table VII, PDA++ improves all four segmentation models and consistently surpasses Copy-Paste and CutMix. On BiSeNetV2, mIoU increases from 67.85 to 72.26. PIDNet improves from 62.58 to 69.82, while SegNeXt increases from 69.65 to 73.79. SegFormer shows a similar improvement from 70.66 to 74.28. The corresponding mAcc results exhibit the same overall tendency. In comparison, Copy-Paste and CutMix provide limited gains and can reduce segmentation accuracy in this setting. The results indicate that the masks associated with the synthesized targets provide effective pixel-level supervision in SAR imagery as well as in the optical setting.

Beyond rigid targets. As detailed in the supplementary material, we further evaluate forest insertion on WHU-OPT-SAR [86] to examine whether the proposed framework generalizes to targets with irregular boundaries. The dataset has a spatial resolution of 5 m, providing a setting that differs substantially from the ship experiments. The pose-conditioned formulation reduces whole-image FID from 30.88 for PDA to 15.55, while MATA provides a further improvement in overall fidelity. These results indicate that the proposed formulation is not restricted to rigid targets and remains effective under notable changes in target geometry and image resolution.

## C. Limitations

The current framework assumes access to pixel-level object masks for constructing the pose-conditioned background and generating segmentation supervision. Although such masks can be obtained from existing segmentation models, this requirement may limit applicability to datasets that provide only bounding-box annotations or weak labels. Developing mask-free conditioning or automatically estimating object support from boxes, points, or reference images would therefore make PDA++ more broadly applicable. In addition, the current framework is validated on RGB optical and SAR imagery. Extending it to multispectral or hyperspectral observations requires latent representations that preserve inter-band correlations and material-dependent spectral signatures, which are not fully supported by standard RGB-oriented encoders. Addressing these limitations will further improve the flexibility of PDA++ across annotation settings and sensing modalities.

## VI. CONCLUSION

We presented PDA++, a unified framework for remote sensing object insertion that progressively adapts synthesized targets to their destination scenes. The proposed formulation improves scene-aware placement through the Affordance Field and uses pose-conditioned scene context to guide target generation under the desired observation. MATA further reduces residual local appearance discrepancies through multi-scale texture alignment. The same pose-conditioned construction also provides pixel-level masks for segmentation augmentation, while recursive synthesis extends the framework to multi-object scene generation.

Extensive experiments on optical and SAR imagery validate both insertion quality and downstream utility. On the optical benchmark, PDA++ achieves a whole-image FID of 6.28 and increases the average few-shot detection m $\mathsf { A P } _ { 5 0 }$ from 61.46 to 79.15, corresponding to an improvement of 17.69 points and a relative gain of 28.8%. On SAR imagery, the proposed method improves SSDD ship detection by 4.10 mAP50 points over real-data training, while consistent gains are observed in segmentation and cross-dataset evaluation. These results show that adapting synthesized targets to the destination environment provides an effective basis for remote sensing generation and synthetic data augmentation.

## REFERENCES

[1] A. M. Rekavandi, L. Xu, F. Boussaid et al., “A guide to image-and video-based small object detection using deep learning: case study of maritime surveillance,” IEEE Transactions on Intelligent Transportation Systems, 2025.

[2] L. Zhou, H. Yan, Y. Shan et al., “Aircraft detection for remote sensing images based on deep convolutional neural networks,” Journal of Electrical and Computer Engineering, 2021.

[3] H. Bandarupally, H. R. Talusani, and T. Sridevi, “Detection of military targets from satellite images using deep convolutional neural networks,” in 2020 IEEE 5th international conference on computing communication and automation (ICCCA), 2020.

[4] S. Gui, S. Song, R. Qin, and Y. Tang, “Remote sensing object detection in the deep learning era—a review,” Remote Sensing, vol. 16, no. 2, 2024. [Online]. Available: https://www.mdpi.com/2072-4292/16/2/327

[5] K. Li, G. Wan, G. Cheng, L. Meng, and J. Han, “Object detection in optical remote sensing images: A survey and a new benchmark,” ISPRS journal of photogrammetry and remote sensing, vol. 159, pp. 296–307, 2020.

[6] J. Ding, N. Xue, G.-S. Xia, X. Bai, W. Yang, M. Y. Yang, S. Belongie, J. Luo, M. Datcu, M. Pelillo et al., “Object detection in aerial images: A large-scale benchmark and challenges,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 11, pp. 7778–7796, 2021.

[7] X. Gao, D. Zhao, and Z. Yuan, “Yolo-parallel: Positive gradient modeling for long-tail remote sensing object detection,” IEEE Geoscience and Remote Sensing Letters, 2024.

[8] Y. Wang, S. M. A. Bashir, M. Khan et al., “Remote sensing image super-resolution and object detection: Benchmark and state of the art,” Expert Systems with Applications, 2022.

[9] S. Mei, J. Lian, X. Wang, Y. Su, M. Ma, and L.-P. Chau, “A comprehensive study on the robustness of deep learning-based image classification and object detection in remote sensing: Surveying and benchmarking,” Journal of Remote Sensing, vol. 4, p. 0219, 2024.

[10] J. Pan, S. Lei, Y. Fu, J. Li, Y. Liu, Y. Sun, X. He, L. Peng, X. Huang, and B. Zhao, “Earthsynth: Generating informative earth observation with diffusion models,” arXiv preprint arXiv:2505.12108, 2025.

[11] D. Tang, H. Wang, Y. Xin, H. Qiao, D. Jiang, Y. Li, Z. Yu, and X. Cao, “Terragen: A unified multi-task layout generation framework for remote sensing data augmentation,” arXiv preprint arXiv:2510.21391, 2025.

[12] Y. Yang, Y. Zhang, K. Zhang, J. Zhang, X. Chen, H. Fu, and R. Dong, “Task-oriented data synthesis and control-rectify sampling for remote sensing semantic segmentation,” arXiv preprint arXiv:2512.16740, 2025.

[13] X. Chen, L. Huang, Y. Liu et al., “Anydoor: Zero-shot object-level image customization,” in CVPR, 2024.

[14] X. Chen, Y. Feng, M. Chen, Y. Wang, S. Zhang, Y. Liu, Y. Shen, and H. Zhao, “Zero-shot image editing with reference imitation,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 84 010–84 032. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/98b2b307aa4aa323df2ba3a83460f25e-Paper-Conference.pdf

[15] H. Wang, J. Peng, Q. He, H. Yang, Y. Jin, J. Wu, X. Hu, Y. Pan, Z. Gan, M. Chi et al., “Unicombine: Unified multi-conditional combination with diffusion transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 18 325–18 334.

[16] W. Song, H. Jiang, Z. Yang, Z. Cheng, R. Quan, and Y. Yang, “Insert anything: Image insertion via in-context editing in dit,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 11, 2026, pp. 9097–9105.

[17] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen et al., “Qwen-image technical report,” arXiv preprint arXiv:2508.02324, 2025.

[18] Z. Tan, S. Liu, X. Yang, Q. Xue, and X. Wang, “Ominicontrol: Minimal and universal control for diffusion transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 14 940–14 950.

[19] Y. Hou, X. Dong, C. Ren, W. Lu, Z. Wei, H. Yu, Y. Wang, and X. Sun, “Plan, decouple, assimilate: Physics-aware object insertion in remote sensing imagery,” in Forty-third International Conference on Machine Learning, 2026. [Online]. Available: https: //openreview.net/forum?id=ojx0CyGXHJ

[20] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial nets,” in Advances in Neural Information Processing Systems, Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger, Eds., vol. 27. Curran Associates, Inc., 2014. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/ 2014/file/f033ed80deb0234979a61f95710dbe25-Paper.pdf

[21] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[22] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021.

[23] T.-J. Fu, Y. Qian, C. Chen, W. Hu, Z. Gan, and Y. Yang, “Univg: A generalist diffusion model for unified image generation and editing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 17 160–17 170.

[24] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[25] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” in International Conference on Learning Representations (ICLR), 2022.

[26] L. Niu, W. Cong, L. Liu, Y. Hong, B. Zhang, J. Liang, and L. Zhang, “Making images real again: A comprehensive survey on deep image composition,” arXiv preprint arXiv:2106.14490, 2021.

[27] Q. Yu, W. Chow, Z. Yue, K. Pan, Y. Wu, X. Wan, J. Li, S. Tang, H. Zhang, and Y. Zhuang, “Anyedit: Mastering unified high-quality image editing for any idea,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 26 125–26 135.

[28] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to text-to-image diffusion models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 3836–3847.

[29] Z. Zhang, F. Fortier-Chouinard, M. Garon, A. Bhattad, and J.-F. Lalonde, “Zerocomp: Zero-shot object compositing from image intrinsics via diffusion,” in 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025, pp. 483–494.

[30] H. Li, Z. Fan, Z. Wen, Z. Zhu, and Y. Li, “Aicomposer: Any style and content image composition via feature integration,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 16 840–16 850.

[31] Y. Zhang, H. Wang, Y. Wang, R. Xie, and L. Song, “Freeinsert: Personalized object insertion with geometric and style control,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 10 361–10 369.

[32] C. Li, W. Wang, Q. Li, N. Sebe, B. Lepri, and W. Nie, “Freeinsert: Disentangled text-guided object insertion in 3d gaussian scene without spatial priors,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 10 915–10 924.

[33] M. Schouten, I. Siglidis, S. Belongie, and D. P. Papadopoulos, “Hiddenobjects: Scalable diffusion-distilled spatial priors for object placement,” arXiv preprint arXiv:2604.10675, 2026.

[34] Z. Zhang, D. Fan, M. Wang, Q. Tang, J. Yang, and Z. Yi, “Region-to region: Enhancing generative image harmonization with adaptive regional injection,” arXiv preprint arXiv:2508.09746, 2025.

[35] F. Fortier-Chouinard, Z. Zhang, L.-E. Messier, M. Garon, A. Bhattad, and J.-F. Lalonde, “Spotlight: Shadow-guided object relighting via diffusion,” in Thirteenth International Conference on 3D Vision, 2026. [Online]. Available: https://openreview.net/forum?id=mZw9TOQUUd

[36] Y. Sheng, J. Zhang, and B. Benes, “Ssn: Soft shadow network for image compositing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 4380–4390.

[37] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, “Microsoft coco: Common objects in context,” in European conference on computer vision. Springer, 2014, pp. 740– 755.

[38] Y. Liu, J. Yue, S. Xia et al., “Diffusion models meet remote sensing: Principles, methods, and perspectives,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[39] P. Phongthawee, W. Chinchuthakun, N. Sinsunthithet, V. Jampani, A. Raj, P. Khungurn, and S. Suwajanakorn, “Diffusionlight: Light probes for free by painting a chrome ball,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 98–108.

[40] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler, “Repurposing diffusion-based image generators for monocular depth estimation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 9492–9502.

[41] Z. Zeng, V. Deschaintre, I. Georgiev, Y. Hold-Geoffroy, Y. Hu, F. Luan, L.-Q. Yan, and M. Hašan, “Rgb↔x: Image decomposition and synthesis using material- and lighting-aware diffusion models,” in ACM SIGGRAPH 2024 Conference Papers, ser. SIGGRAPH ’24. Association for Computing Machinery, 2024.

[42] J. Liu, Z. Yuan, Z. Pan, Y. Fu, L. Liu, and B. Lu, “Diffusion model with detail complement for super-resolution of remote sensing,” Remote Sensing, vol. 14, no. 19, 2022. [Online]. Available: https://www.mdpi.com/2072-4292/14/19/4834

[43] C. Wang and W. Sun, “Semantic guided large scale factor remote sensing image super-resolution with generative diffusion prior,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 220, pp. 125–138, 2025.

[44] T. Sousa, B. Ries, and N. Guelfi, “Data augmentation in earth observation: A diffusion model approach,” Information, vol. 16, no. 2, p. 81, 2025.

[45] Z. Yuan, C. Hao, R. Zhou, J. Chen, M. Yu, W. Zhang, H. Wang, and X. Sun, “Efficient and controllable remote sensing fake sample generation based on diffusion model,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–12, 2023.

[46] Z. Chen, Z. Zhang, and F. Zhang, “Rsedit: Text-guided image editing for remote sensing,” IEEE Geoscience and Remote Sensing Letters, vol. 23, pp. 6 011 905–6 011 905, 2026.

[47] K. Tang and J. Chen, “Changeanywhere: Sample generation for remote sensing change detection via semantic latent diffusion model,” arXiv preprint arXiv:2404.08892, 2024.

[48] V. Martin, K. B. Venable, and D. Morgan, “Generating satellite imagery data for wildfire detection through mask-conditioned generative ai,” arXiv preprint arXiv:2604.02479, 2026.

[49] S. Khanna, P. Liu, L. Zhou, C. Meng, R. Rombach, M. Burke, D. Lobell, and S. Ermon, “Diffusionsat: A generative foundation model for satellite imagery,” in International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, Eds., vol. 2024, 2024, pp. 5586–5604. [Online]. Available: https://proceedings.iclr.cc/paper\_files/paper/2024/ file/16c3c941409d0581286eff49b180930f-Paper-Conference.pdf

[50] D. Tang, X. Cao, X. Hou, Z. Jiang, J. Liu, and D. Meng, “Crs-diff: Controllable remote sensing image generation with diffusion model,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[51] Z. Yu, C. Liu, L. Liu, Z. Shi, and Z. Zou, “Metaearth: A generative foundation model for global-scale remote sensing image generation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 3, pp. 1764–1781, 2025.

[52] C. Liu, K. Chen, R. Zhao, Z. Zou, and Z. Shi, “Text2earth: Unlocking textdriven remote sensing image generation with a global-scale dataset and a foundation model,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 3, pp. 238–259, 2025.

[53] M. Goktepe, A. hossein Shamseddin, E. Uysal, J. M. Monteagudo, L. Drees, A. Toker, S. Asseng, and M. von Bloh, “Ecomapper: Generative modeling for climate-aware satellite imagery,” in Fortysecond International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=YUtJsxQjv3

[54] A. Toker, M. Eisenberger, D. Cremers, and L. Leal-Taixé, “Satsynth: Augmenting image-mask pairs through diffusion models for aerial semantic segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 27 695–27 705.

[55] K. Deng, S. Wei, S. Pang, H. Jiang, and B. Su, “Synthesizing remote sensing images from land cover annotations via graph prior masked diffusion,” Remote Sensing, vol. 17, no. 13, 2025. [Online]. Available: https://www.mdpi.com/2072-4292/17/13/2254

[56] Z. Gong, Z. Wei, D. Wang, X. Hu, X. Ma, H. Chen, Y. Jia, Y. Deng, Z. Ji, X. Zhu, X. Yang, N. Yokoya, J. Zhang, B. Du, J. Yan, and L. Zhang, “Crossearth: Geospatial vision foundation model for domain generalizable remote sensing semantic segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 5, pp. 5147–5164, 2026.

[57] W. Li, W. Yang, Y. Hou, L. Liu, Y. Liu, and X. Li, “Saratr-x: Toward building a foundation model for sar target recognition,” IEEE Transactions on Image Processing, vol. 34, pp. 869–884, 2025.

[58] F. Zhang, X. Wu, F. Ma, Q. Yin, and Y. Hu, “Geodiff-sar: A geometric prior guided diffusion model for sar image generation,” arXiv preprint arXiv:2601.03499, 2026.

[59] S. Debuysère, N. Trouvé, N. Letheule, O. Lévêque, and E. Colin, “Quantitative comparison of fine-tuning techniques for pretrained latent diffusion models in the generation of unseen sar images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 234, pp. 93–110, 2026.

[60] F. Han, L. Si, H. Dong, Z. Jiang, L. Zhang, H. Chen, Y. Liu, and B. Du, “Exploring text-guided single image editing for remote sensing images,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025.

[61] Z. Zhang, R. Cao, H. Sheng, M. Guo, Z. Shao, and L. Wu, “Shadow detection and removal for remote sensing images via multi-feature adaptive optimization and geometry-aware illumination compensation,” Expert Systems with Applications, p. 127769, 2025.

[62] Y.-H. Tsai, X. Shen, Z. Lin, K. Sunkavalli, X. Lu, and M.-H. Yang, “Deep image harmonization,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 3789–3797.

[63] W. Cong, J. Zhang, L. Niu, L. Liu, Z. Ling, W. Li, and L. Zhang, “Dovenet: Deep image harmonization via domain verification,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8394–8403.

[64] K. Li, S. Zhang, Y. Deng, Z. Wang, D. Meng, and X. Cao, “Segearth-ov3: Exploring sam 3 for open-vocabulary semantic segmentation in remote sensing images,” arXiv preprint arXiv:2512.08730, 2025.

[65] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything with concepts,” arXiv preprint arXiv:2511.16719, 2025.

[66] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[67] C. Mao, J. Zhang, Y. Pan, Z. Jiang, Z. Han, Y. Liu, and J. Zhou, “Ace++: Instruction-based image creation and editing via context-aware content filling,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, October 2025, pp. 1979–1987.

[68] Y. Yu, Z. Zeng, H. Zheng, and J. Luo, “Omnipaint: Mastering objectoriented editing via disentangled insertion-removal inpainting,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 17 324–17 334.

[69] D. Wang, J. Zhang, B. Du, M. Xu, L. Liu, D. Tao, and L. Zhang, “Samrs: Scaling-up remote sensing segmentation dataset with segment anything model,” Advances in Neural Information Processing Systems, vol. 36, pp. 8815–8827, 2023.

[70] S. Waqas Zamir, A. Arora, A. Gupta, S. Khan, G. Sun, F. Shahbaz Khan, F. Zhu, L. Shao, G.-S. Xia, and X. Bai, “isaid: A large-scale dataset for instance segmentation in aerial images,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2019, pp. 28–37.

[71] X. Sun, P. Wang, Z. Yan, F. Xu, R. Wang, W. Diao, J. Chen, J. Li, Y. Feng, T. Xu et al., “Fair1m: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 184, pp. 116–130, 2022.

[72] G.-S. Xia, X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu, M. Pelillo, and L. Zhang, “Dota: A large-scale dataset for object detection in aerial images,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[73] S. Yang, Z. Pei, F. Zhou, and G. Wang, “Rotated faster r-cnn for oriented object detection in aerial images,” in Proceedings of the 2020 3rd International Conference on Robot Systems and Applications, 2020, pp. 35–39.

[74] X. Xie, G. Cheng, J. Wang, X. Yao, and J. Han, “Oriented r-cnn for object detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 3520–3529.

[75] J. Han, J. Ding, J. Li, and G.-S. Xia, “Align deep features for oriented object detection,” IEEE transactions on geoscience and remote sensing, vol. 60, pp. 1–11, 2021.

[76] R. Sapkota, R. H. Cheppally, A. Sharda, and M. Karkee, “Yolo26: key architectural enhancements and performance benchmarking for real-time object detection,” arXiv preprint arXiv:2509.25164, 2025.

[77] C. Yu, C. Gao, J. Wang, G. Yu, C. Shen, and N. Sang, “Bisenet v2: Bilateral network with guided aggregation for real-time semantic segmentation,” International journal of computer vision, vol. 129, no. 11, pp. 3051–3068, 2021.

[78] J. Xu, Z. Xiong, and S. P. Bhattacharyya, “Pidnet: A real-time semantic segmentation network inspired by pid controllers,” in 2023 IEEE/CVF

Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 19 529–19 539.

[79] M.-H. Guo, C.-Z. Lu, Q. Hou, Z. Liu, M.-M. Cheng, and S.-M. Hu, “Segnext: Rethinking convolutional attention design for semantic segmentation,” Advances in neural information processing systems, vol. 35, pp. 1140–1156, 2022.

[80] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in neural information processing systems, vol. 34, pp. 12 077–12 090, 2021.

[81] W. Yu, G. Cheng, M. Wang, Y. Yao, X. Xie, X. Yao, and J. Han, “MAR20: A benchmark for military aircraft recognition in remote sensing images,” National Remote Sensing Bulletin, vol. 27, no. 12, pp. 2688–2696, 2023.

[82] T. Burgert, K. N. Clasen, J. Klotz, T. Siebert, and B. Demir, “A label propagation strategy for cutmix in multi-label remote sensing image classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025.

![](images/b958abe9af99d9fce9e856efb1b83bdba42ea1a280bebb2a9f33c268527c36f1.jpg)

[83] D. Zhao, B. Yuan, Z. Chen, T. Li, Z. Liu, W. Li, and Y. Gao, “Panoptic perception: A novel task and fine-grained dataset for universal remote sensing image interpretation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–14, 2024.

[84] S. Wei, X. Zeng, Q. Qu, M. Wang, H. Su, and J. Shi, “Hrsid: A highresolution sar images dataset for ship detection and instance segmentation,” Ieee Access, vol. 8, pp. 120 234–120 254, 2020.

[85] T. Zhang, X. Zhang, J. Li, X. Xu, B. Wang, X. Zhan, Y. Xu, X. Ke, T. Zeng, H. Su, I. Ahmad, D. Pan, C. Liu, Y. Zhou, J. Shi, and S. Wei, “SAR ship detection dataset (SSDD): Official release and comprehensive data analysis,” Remote Sensing, vol. 13, no. 18, 2021. [Online]. Available: https://www.mdpi.com/2072-4292/13/18/3690

![](images/b2b6330eb2aae7f6eb07408b987af747317f33a15587c5a867a475ea0d76588a.jpg)

[86] X. Li, G. Zhang, H. Cui, S. Hou, S. Wang, X. Li, Y. Chen, Z. Li, and L. Zhang, “Mcanet: A joint semantic segmentation framework of optical and sar images for land use classification,” International Journal of Applied Earth Observation and Geoinformation, vol. 106, p. 102638, 2022.

![](images/42ae21edb0cc6a1c27025aa4f62d24642fa65f30cd0fb90d1759f4ac93c94b74.jpg)

Wanxuan Lu received the B.Sc. degree from Beijing Institute of Technology, Beijing, China, in 2016, and the Ph.D. degree from Chinese Academy of Sciences, Beijing, in 2021. She is currently an Associate Professor with the Aerospace Information Research Institute, Chinese Academy of Sciences. Her research interests include computer vision and remote sensing image processing.

## BIOGRAPHY SECTION

Zihan Wei received the B.Sc. degree in electronic information engineering from Huaqiao University, Xiamen, China, in 2024. He is currently pursuing the Ph.D. degree with the Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing, China. His research interests include computer vision, remote sensing image understanding, and remote sensing image generation.

Hongfeng Yu received the B.Sc. and M.Sc. degrees from Peking University, Beijing, China, in 2013 and 2016, respectively, and the Ph.D. degree from Chinese Academy of Sciences, Beijing, in 2023. He is currently an Associate Professor with the Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing. His research interests include deep learning and remote sensing.

![](images/4badb4dc6fe7230b5b0faf557b3bd35c5d3c349ef6c4254ee9acba800251c1fe.jpg)  
Xianchi Dong received the B.Sc. degree in electronic science and technology from the School of Informa tion Science and Engineering, Shandong University, China, in 2024. He is currently a Ph.D. student with the Aerospace Information Research Institute, Chinese Academy of Sciences. His research interests include computer vision, deep learning applications for remote sensing, and generative models for remote sensing imagery.

![](images/5222fbfb7401fd2b8589f9ca41f3a501bd24c68e294bc98140bd5ad983e98818.jpg)

Yixiao Wang received his B.Sc. and Ph.D. degrees in electronics engineering and computer science from Peking University in 2009 and 2014, respectively. He joined the Digital Multimedia Lab at University of British Columbia as a postdoctoral research fellow in 2018. In 2023, he joined LG Toronto AI Lab as a Senior Research Scientist. He is currently a Professor with the Aerospace Information Research Institute, Chinese Academy of Sciences. His research interests include physical intelligence, multimodal generative AI, and autonomous agents.

![](images/48a4a07c0015d3403a692334ee6caaa0378e465a552a93facdb3a399af84631c.jpg)

Yingyan Hou received the B.Sc. degree in computer science and technology from Harbin Institute of Tech nology, Weihai, China, in 2020, and the M.Sc. degree in electronic information from Tsinghua University, in 2023. She is currently pursuing the Ph.D. degree with the Aerospace Information Research Institute, Chinese Academy of Sciences. Her research interests include remote sensing image generation, SAR image synthesis and unified understanding and generation.

![](images/e76b47a98e4b0e43d9655b9b87790c8e5d905955ca866b773cf665be8aed50a6.jpg)

Chubo Deng received the B.Sc. degree from Hong Kong Baptist University, Hong Kong, in 2012, and the M.Sc. and Ph.D. degrees from George Washington University, Washington, DC, USA, in 2018. He is currently an Associate Professor with the Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing, China. His research interests in clude computer vision, AI for science, geospatial data mining, and remote sensing image understanding.

![](images/e401c85b40e848350fc8033bfa37f9ed44026ae8738125875f1e7b7968a54941.jpg)

Chao Ren is a Professor at the Aerospace Informa tion Research Institute, Chinese Academy of Sciences. He received the B.E. degree from Nanjing University of Aeronautics and Astronautics, Nanjing, China, in 2017, and the Ph.D. degree from Nanyang Technological University, Singapore, in 2022. Previously, he was a Wallenberg-NTU Presidential Postdoctoral Fellow at Nanyang Technological University and KTH Royal Institute of Technology. His research interests include machine learning, foundation models and distributed optimization.

![](images/3fa772f259e553b0f1b53bb7ae4a8b4ce104f6d6df85ae6373e795da55fbd445.jpg)

Xian Sun (Senior Member, IEEE) received the B.Sc. degree in electronic information engineering from the Beijing University of Aeronautics and Astronautics, Beijing, China, in 2004, and the M.Sc. and Ph.D. degrees in signal and information processing from the Institute of Electronics, Chinese Academy of Sciences, Beijing, in 2009. He is currently a Professor with the Aerospace Information Research Institute, Chinese Academy of Sciences. His research interests include computer vision and remote sensing image understanding.