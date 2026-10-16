# Edit4D: Editable 4D Scene Generation with Textual Control

Chanwoo Kim<sup>1,2</sup> Tackgeun You<sup>1</sup> Hwasup Lim<sup>1,2,†</sup>

<sup>1</sup>Korea Institute of Science and Technology (KIST), Republic of Korea <sup>2</sup>University of Science and Technology (UST), Republic of Korea

{sky9739, tackgeun.you, hslim}@kist.re.kr

<sup>†</sup>Corresponding author

Project page: https://chanwo0kim.github.io/Edit4D/

![](images/def292962ed851852bbbbba0ab0b31f68bd04b4613e06f45428eec63277109df.jpg)  
Figure 1. Overview of Edit4D. Given a user instruction, Edit4D first decomposes it into stage-specific prompts. Each stage of the pipeline then takes the corresponding prompts for the scene, character appearance, and motion, and generates the corresponding Gaussian splatting for each stage. We further propose a composition stage that naturally places the character into the background and renders them jointly By constructing the scene from separated background and dynamic components, Edit4D enables flexible and independent editing.

## Abstract

We introduce Edit4D, an editable text-to-4D scene generationframework that constructs 4D scenesfrom separate background and dynamic character representations. Recent advances in generative models have enabled text-driven dynamic scene synthesis, but editable 4D scene generation remains challenging due to the need to preserve spatial geometry and temporal dynamics. Existing approaches typically generate holistic scene videos, where the background and dynamic characters are coupled, making object-level

editing and flexible camera control difficult. Edit4D addresses these limitations through a stage-specific generation pipeline. Given a user instruction, Edit4D first decomposes it into stage-specific prompts for background, dynamic character, and scene–character composition. Using these prompts, we synthesize multi-view scene images with text-to-image and camera-conditioned generative priors and reconstruct a background 3D Gaussian splatting representation. We then generate a character image, animate it with an image-conditioned video prior, reconstruct a 4D mesh, and bind Gaussian primitives to obtain a character representation. Finally, both components are composed through text-guided segmentation and contact-aware optimization. A global scene refinement is then performed to improve harmonization and rendering consistency. By integrating stage-specific components, Edit4D enables controllable camera motion, flexible object-level editing, and the practical construction of editable 4D scenes. Project page: https://chanwo0kim.github.io/Edit4D/

## 1. Introduction

Advances in large-scale text-to-image models such as DALL-E [48] and Stable Diffusion [51] have rapidly accelerated visual content generation. This progress is especially evident in video generation, where models such as Sora [37] and Veo [11] demonstrate strong photorealism, temporal coherence, and scene complexity. Building on these advances, recent research has shifted toward 3D and 4D content creation. In particular, 4D generation aims to capture both spatial structure and temporal evolution, requiring multi-view consistency, temporal coherence, and stable 3D geometry. Despite progress in diffusion models, general 4D scene generation remains challenging and has yet to reach the level required for widespread commercial use.

Existing text-to-4D methods can be broadly divided into object-level and scene-level approaches. Object-level methods [2, 79] often produce multi-view consistent dynamic subjects by focusing on a single foreground object, but they do not generate complete scenes with backgrounds. Scenelevel methods [35, 75] synthesize complete dynamic scenes as a single holistic representation, making object-level editing difficult. Moreover, because geometry is reconstructed from a limited set of generated viewpoints, rendering quality often degrades under novel-view changes. These challenges are further amplified by the scarcity of large-scale dynamic-scene datasets, which forces many methods to rely on distillation from image or video generative priors. Although such priors provide useful supervision, the resulting 4D content remains constrained by their quality and failure modes. Furthermore, when artifacts appear in the generated content, even localized errors require regenerating the entire sequence, making correction costly and prone to inconsistencies with the original output. In real production workflows, where expert-driven refinement is often necessary, this lack of editability becomes a significant limitation. These observations suggest that constructing 4D scenes as composable assets, rather than as a single monolithic video, offers a more flexible alternative. Asset-based generation naturally aligns with graphics workflows, allowing creators to manipulate and replace scene components with greater control. However, editable 4D scene construction remains underexplored, as existing pipelines often couple backgrounds with dynamic elements and make indepen-

dent editing difficult.

To address these limitations, we introduce Edit4D, an editable text-to-4D framework that constructs 4D scenes from separate background scene and dynamic character representations. Given a user instruction, Edit4D first decomposes it into stage-specific prompts for background, dynamic character, and composition. We then generate the 4D scene through parallel background scene and dynamic character generation, followed by scene–character composition. The background scene is reconstructed as a 3D Gaussian representation from multi-view images synthesized using textto-image and camera-conditioned generative priors. For the dynamic character, we generate a reference video by animating a character image, reconstructing a temporally consistent 4D mesh, and binding Gaussian primitives to obtain a dynamic character representation. The composition stage localizes the target placement region with text-guided masks and optimizes the character pose using contact-aware constraints. We further perform a global scene refinement to improve appearance consistency and rendering details. Extensive experiments show that Edit4D achieves state-ofthe-art performance against recent 4D generation methods, while additionally supporting controllable camera motion. We illustrate the overview of our method in Fig. 1.

We summarize our contributions as follows:

• We propose Edit4D, an editable text-to-4D framework that constructs 4D scenes by composing separate background and dynamic character representations through a stage-specific generation pipeline. This component-wise design enables independent editing, flexible control, and scene recomposition within a unified 4D space.

• We introduce a text-guided composition method that automatically places a dynamic character into a background scene. Our method localizes target region from text and optimizes character position with contact and penetration constraints, reducing the need for manual placement.

• Edit4D achieves state-of-the-art performance compared with recent 4D generation methods while preserving high perceptual quality. Beyond generation quality, it supports camera control and fine-grained object-level editing.

## 2. Related Works

## 2.1. 3D and 4D Scene Generation

Text-to-3D generation has advanced rapidly with the development of large-scale generative models. Score distillation sampling (SDS)-based methods [8, 30, 31, 46, 64, 80] synthesize 3D content by aligning rendered views with text-conditioned diffusion priors, but they are computationally expensive and often suffer from view-dependent artifacts and the Janus problem. Recent multi-view guided methods address these limitations by generating geometrically consistent views and then lifting them into 3D representations. Text-conditioned [20, 25, 26, 53] and imageconditioned [32, 33, 36, 60, 71] approaches follow this paradigm to improve visual fidelity and efficiency. More recently, data-driven models [57, 68, 78] trained on largescale 3D datasets generate 3D from text or image inputs.

![](images/18ecb5e0922ecf8570d23459ca3399f45daf98fff781a8f742271dd604068176.jpg)  
Figure 2. Comparison of novel-view renderings from 4D scenes generated by Free4D [35] and our method. Compared to methods that directly distill video diffusion priors, our composition-based method constructs the 4D scene from explicit background and dynamic 3D components, maintaining better visual quality under novel-view camera changes.

Building on these advances, recent text- or video-to-4D methods [2, 35, 44, 49, 50, 54, 56, 74, 75, 79] synthesize dynamic 3D content using diffusion guidance, multi-view generation, or SDS-based optimization. Object-level methods such as 4Dfy [2] and Dream-in-4D [79] can produce temporally coherent and multi-view consistent dynamic objects. Scene-level methods such as 4Real [75] and Free4D [35] instead generate complete dynamic scenes, improving photorealism or temporal consistency through video-conditioned reconstruction or point-guided denoising. However, these methods typically perform holistic 4D synthesis, making it difficult to isolate and control individual scene components after generation. In contrast, Edit4D constructs 4D scenes from separate representations, enabling fine-grained modification while preserving spatio-temporal coherence.

## 2.2. Scene Representation

Neural Radiance Fields (NeRF) [38] established a foundation for neural 3D scene representations through volumetric rendering. Although NeRF achieves high visual fidelity, its slow rendering speed motivated more efficient variants such as Instant-NGP [41] and HexPlane [5], which leverage hash-based or tensor-decomposed encodings. More recently, Gaussian Splatting [22] introduced a rasterizationbased representation using Gaussian primitives, enabling real-time rendering with high visual quality. This line of work has been extended to dynamic 4D representations, in cluding NeRFPlayer [55], 4D-GS [66], and Grid4D [69], which jointly model spatial structure and temporal motion. However, accurately recovering geometry remains challenging when optimization relies mainly on photometric supervision. Recent methods such as SuGaR [13] and 2D Gaussian [18] redesign Gaussian primitives to align with underlying surfaces, while structure-guided frameworks like GaussianAvatar [17, 40] and Mani-GS [10] attach the Gaussians to deformable meshes for stable geometry.

## 2.3. Scene Composition

With recent advances in segmentation foundation models [23, 76], mask-based composition methods in 3D neural scenes have been actively studied. By leveraging 2D masks from segmentation or text-to-mask models, prior methods localize editable regions for object removal and inpainting, ensuring reliable composition [3, 52, 73]. More closely related to our work, human-scene placement methods aim to place human in 3D scene with physically plausible contact while reducing interpenetration artifacts. PROX [14] fits SMPL-X [45] bodies to scenes using contact and penetration constraints, and POSA [15] learns a body-centric contact representation for plausible placements. Furthermore, COINS [77] and Narrator [70] refine using contact optimization to encourage natural human-scene contact. However, these methods typically assume reconstructed real scenes, supervised interaction data, or manually prepared assets. In contrast, Edit4D performs text-driven scene– character composition, allowing dynamic characters to be automatically placed into generated background scenes through text-guided segmentation and contact-aware optimization.

## 3. Method

We propose Edit4D, a unified framework for generating editable 4D scenes from textual descriptions. As shown in Fig. 3, our framework composes independently generated background scenes and dynamic characters into a unified 4D representation, enabling object-level editing and recomposition. We first describe our editable 4D scene representation based on Gaussian splatting. We then explain how Edit4D constructs a 4D scene through a stage-specific generation pipeline using off-the-shelf generative priors.

![](images/a6bb6db4dec3fa5ca2a68e5dbd3eb93103eea460731d830e36a50786bbeb23f6.jpg)  
Figure 3. Detailed Generation Pipeline of Edit4D. In the first stage, Edit4D decomposes a user instruction into background scene text, character appearance and motion text, and composition text for target placement via off-the-shelf large language model. In the second stage, our framework generates background scene and dynamic character. A background scene is generated as a 3D Gaussian representation using text- and camera-conditioned priors. It then constructs a dynamic 3D character via video-guided mesh reconstruction and Gaussian binding In the third stage, both components are composed through text-guided segmentation and contact-aware optimization to obtain an editable 4D scene. Finally, we perform a global scene refinement to improve appearance harmonization and rendering consistency.

## 3.1. Editable 4D Scene with Gaussian Splatting

We represent a 4D scene using Gaussian splatting defined by the following Gaussian splatting parameters:

$$
\mathcal { G } = \{ ( \mu _ { i } , r _ { i } , s _ { i } , \alpha _ { i } , \mathbf { c } _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{1}
$$

where $\mu , r , s , \alpha$ , and c denote the 3D mean, rotation, scale, opacity, and spherical-harmonics coefficients, respectively, and N is the total number of Gaussians.

Our goal is to generate a time-varying Gaussian representation conditioned on a user instruction. Instead of modeling the entire scene as a single holistic representation, Edit4D constructs the 4D scene from separate background scene and dynamic 3D representations. Formally, we define Gaussian splatting parameters at time t as follows:

$$
\mathcal { G } ( t ) = \mathcal { G } ^ { \mathrm { b g } } \oplus \left( \bigoplus _ { i = 1 } ^ { N ^ { \mathrm { o b j } } } \mathcal { G } _ { i } ^ { \mathrm { o b j } } ( t ) \right) ,\tag{2}
$$

where ${ \mathcal { G } } ^ { \mathrm { b g } }$ denotes the background Gaussian representation, and $\mathcal { G } _ { i } ^ { \mathrm { o b j } } ( t )$ denotes the time-dependent Gaussian representation of the i-th dynamic object. The ⊕ operator indicates concatenation over Gaussian primitives.

This component-wise formulation naturally supports object-level editing, such as modifying the appearance, motion, or spatial placement of individual objects. Moreover, by leveraging a standard Gaussian splatting renderer, we can freely control the camera pose and render the resulting scene from arbitrary viewpoints.

Following the Gaussian splatting formulation, we denote the differentiable renderer as R. Given a camera pose $\mathbf { c } _ { k }$ at time t, the rendered frame is defined as:

$$
I _ { { \bf c } _ { k } , t } = \mathcal { R } ( \mathcal { G } ( t ) , { \bf c } _ { k } ) .\tag{3}
$$

This representation supports novel-view rendering, dynamic camera control, and component-level editing.

## 3.2. Stage 1: Instruction Decomposition

Edit4D takes a single user instruction that describes the background scene, dynamic character, and scene–character composition. Since each module requires different information, the original instruction is not directly suitable for all stages of the pipeline. We therefore use a large language model [42] to convert the user instruction into stage-specific prompts while preserving its original semantics. As shown in Fig. 3, the decomposed prompts guide background scene generation, dynamic 3D character generation, and scene– character composition. This instruction decomposition allows each module to operate with an appropriate prompt and enables the overall pipeline to run automatically from a single user instruction.

## 3.3. Stage 2: Background Scene Generation

To generate a background scene, we first synthesize a highquality scene image from the background scene text. This image serves as a reference that defines the overall layout, appearance, and semantic structure of the scene. We then employ a camera-conditioned multi-view generation model to expand the image into a set of geometrically consistent views along sampled camera trajectories. These generated views provide additional spatial observations and parallax cues necessary for reliable 3D reconstruction. Finally, we reconstruct a background Gaussian splatting representation from the generated multi-view observations, yielding a globally consistent background representation ${ \mathcal { G } } ^ { \mathrm { b g } }$

## 3.4. Stage 2: Dynamic 3D Character Generation

We use a text-to-image model to generate a character image and animate it with the image-conditioned video generator. Given the generated video, we employ a mesh estimator [7] to reconstruct a deformable mesh sequence $\{ \mathbf { M } _ { t } \} _ { t = 1 } ^ { T }$ with temporally consistent topology across time. To enable joint rendering with the background scene, we convert the mesh sequence into a 4D Gaussian splatting representation using the mesh-binding scheme following [10] to exploit mesh topology. Specifically, Gaussians are attached to mesh faces and updated according to the underlying mesh deformation, providing a topology-aware dynamic representation.

While mesh binding provides a strong motion prior, it is insufficient to model fine-grained deformations and may suffer from vertex twisting artifacts. We train a deformation field $f _ { \theta }$ that predicts Gaussian offsets from the first-frame character Gaussian positions $\mathbf { x } _ { 1 }$ and timestamp t:

$$
f _ { \theta } ( \mathbf { x } _ { 1 } , t ) = ( \Delta \mathbf { x } , \Delta \mathbf { r } , \Delta \mathbf { s } , \Delta \mathbf { c } ) ,\tag{4}
$$

where $\Delta \mathbf { x } , \Delta \mathbf { r } , \Delta \mathbf { s } .$ , and $\Delta \mathbf { c }$ denote the offsets of position, rotation, scale, and color attributes, respectively.

For supervision, we render multi-view images $\mathbf { I } _ { \mathrm { g t } } ^ { k } \in$ $\mathbb { R } ^ { H \times W \times 3 }$ from the mesh sequence in Blender using cameras sampled on a sphere around the subject. The deformation field is optimized with photometric and structural losses as follows:

$$
\mathcal { L } = | \hat { I } - I _ { \mathrm { g t } } | + \mathrm { S S I M } ( \hat { I } , I _ { \mathrm { g t } } ) .\tag{5}
$$

## 3.5. Stage 3: Scene–Character Composition

To reduce manual effort in composing separately generated background scenes and dynamic characters, we introduce an automated alignment procedure, as illustrated in Fig. 3. Given a target object label (e.g., sofa or floor), we first obtain a 2D segmentation mask $M _ { 2 \mathrm { D } }$ using an openvocabulary segmentation model [76]. We then construct a 3D Gaussian mask by selecting Gaussians whose projections fall within $M _ { 2 \mathrm { D } }$ . For each Gaussian mean x, the corresponding image coordinate (u, v) is computed using the camera intrinsic matrix K and extrinsic parameters (R, T):

$$
\mathbf { x } ^ { c } = \mathbf { R } \mathbf { x } + \mathbf { T } , \begin{array} { l } { \left[ u \right] } \\ { v } \\ { 1 } \end{array} = \mathbf { K } \left[ \begin{array} { l } { x ^ { c } / z ^ { c } } \\ { y ^ { c } / z ^ { c } } \\ { 1 } \end{array} \right] ,\tag{6}
$$

where $\mathbf { x } ^ { c } = ( x ^ { c } , y ^ { c } , z ^ { c } )$ denotes the point in camera coordinates. To remove projection ambiguities, we further filter candidates using the depth map $D ^ { \mathrm { b g } }$ rendered from the background scene Gaussian splatting:

$$
( u , v ) \in M _ { \mathrm { 2 D } } ~ \mathrm { a n d } ~ \left| z ^ { c } - D ^ { \mathrm { b g } } ( u , v ) \right| < \tau ,\tag{7}
$$

where $\tau$ is a depth tolerance set to 5cm. The remaining Gaussians form the masked 3D region $M _ { 3 \mathrm { D } }$ , which serves as the target region for character composition.

We define character-object contact along the y-axis, consistent with the $y -$ down coordinate system used by the background scene generator. Using the masked 3D Gaussian $M _ { 3 \mathrm { D } }$ , we compute the target object’s bounding box and use it as a geometric prior for character initialization. Specifically, the character is translated along the −y direction relative to the bounding box, placing it near the target surface before optimization. This initialization improves optimization stability and reduces failure cases caused by implausible placements, as illustrated in Fig. 7.

To avoid implausible penetration while still allowing stable contact between instances, we introduce two complementary terms defined on the character mesh M and the background scene Gaussians. Following [16, 21], the penetration loss penalizes background scene points located inside the character mesh. By introducing the penetration loss, we prevent the contact loss from acting excessively, thereby avoiding cases where the character passes through objects and promoting plausible contact as follows:

$$
\mathcal { L } _ { \mathrm { p e n e t r a t i o n } } = \frac { 1 } { | \mathcal { T } | } \sum _ { x \in \mathcal { T } } | | \mathcal { N } _ { \mathbf { x } ^ { \mathrm { M } } } ( x ) - x | | ,\tag{8}
$$

where I denotes the set of background scene points located inside the character mesh, and $\mathcal { N } _ { \mathbf { x } ^ { \mathrm { M } } } ( x )$ denotes the nearest neighbor of x among the mesh vertices $\mathbf { x } ^ { \mathbf { M } }$ . This term discourages interpenetration by pushing the character away from the object geometry.

Next, we introduce a nearest-neighbor contact loss with margin τ defined on the masked object region $M _ { 3 \mathrm { D } }$ . To avoid attracting the character toward unintended regions, we restrict the optimization to a down-sampled subset S consisting of the top $\alpha \%$ mesh vertices $\mathbf { x } ^ { \mathbf { M } }$ that are closest to the target object along the y-axis.

$$
\mathcal { L } _ { \mathrm { c o n t a c t } } = \frac { 1 } { | S | } \sum _ { x \in S } \operatorname* { m a x } ( 0 , | | x - \mathcal { N } _ { M _ { 3 \mathrm { D } } } ( x ) | | - \tau ) .\tag{9}
$$

Together with the penetration loss, we optimize the global translation and rotation of the character to align it with the target region in the background scene.

## 3.6. Global Scene Refinement

Although the proposed composition stage produces physically plausible scene–character composition, appearance discrepancies may still remain between the independently generated background and character. To improve visual coherence, we perform a global scene refinement on the composed 4D scene. Given a rendered image $I _ { \mathbf { c } _ { k } , t }$ from camera viewpoint $\mathbf { c } _ { k }$ at time t, we optimize the scene using a text-conditioned diffusion prior guided by the user instruction. Unlike the composition stage, which only updates the character pose, we jointly optimize the background scene, canonical character Gaussians, and the deformation field. This refinement encourages appearance harmonization between independently generated components while improving fine-grained visual details. Following the SDS-based refinement method [27], we optimize the rendered scene using both latent-space and image-space consistency terms:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e f i n e } } = \mathbb { E } _ { t , \mathbf { c } _ { k } , \tau , \epsilon } \Big [ \gamma ( \tau ) \big ( \lambda _ { 1 } | | z _ { \mathbf { c } _ { k } , t } - \hat { z } _ { \mathbf { c } _ { k } , t } | | _ { 2 } ^ { 2 } } \\ { + \lambda _ { 2 } | | I _ { \mathbf { c } _ { k } , t } - \hat { I } _ { \mathbf { c } _ { k } , t } | | _ { 2 } ^ { 2 } \big ) \Big ] . } \end{array}\tag{10}
$$

where $z _ { \mathbf { c } _ { k } , t } = \epsilon ( I _ { \mathbf { c } _ { k } , t } )$ denotes the latent representation encoded by the diffusion model encoder ϵ. For the canonical time $t = 0 .$ , we sample the camera used during background scene generation to improve multi-view consistency. For dynamics, we only use the frontal view to preserve character identity and temporal consistency during optimization.

Furthermore, we improve scene–character consistency by generating cast shadows based on the estimated scene illumination. We estimate an HDR environment map from background multi-view renderings [29] and compute character-induced light occlusion following Gaussian ray tracing [12, 39]. The resulting attenuation is applied to the rendered background, producing temporally consistent cast shadows that follow the scene illumination. Further implementation details are provided in Appendix A.

<table><tr><td>Settings</td><td>Method</td><td>Text Align</td><td>Consistency</td><td>Aesthetic</td></tr><tr><td rowspan="3">(a)</td><td>4Real [75]</td><td>26.7%</td><td>98.1%</td><td>52.6%</td></tr><tr><td>Free4D [35]</td><td>24.7%</td><td>97.6%</td><td>62.3%</td></tr><tr><td>Ours</td><td>27.1%</td><td>99.4%</td><td>68.3%</td></tr><tr><td rowspan="3">(b)</td><td>4Dfy [2]</td><td>27.6%</td><td>98.1%</td><td>56.2%</td></tr><tr><td>Free4D [35]</td><td>25.2%</td><td>98.1%</td><td>61.2%</td></tr><tr><td>Ours</td><td>28.0%</td><td>99.2%</td><td>66.2%</td></tr><tr><td rowspan="3">(c)</td><td>Dream-in-4D [79]</td><td>26.6%</td><td>97.6%</td><td>56.2%</td></tr><tr><td>Free4D [35]</td><td>25.5%</td><td>97.1%</td><td>63.2%</td></tr><tr><td>Ours</td><td>29.2%</td><td>99.3%</td><td>74.8%</td></tr></table>

Table 1. Quantitative comparison with previous studies. We measure text alignment, consistency, and aesthetic quality using VBench [19] on three experimental settings.
<table><tr><td>Method</td><td>3D Structure</td><td>Visual Quality</td></tr><tr><td>Free4D [35]</td><td>39.5%</td><td>35.0%</td></tr><tr><td>Ours</td><td>60.5%</td><td>65.0%</td></tr></table>

Table 2. Pairwise user study comparing Free4D and Edit4D. We report user preference for 3D structure and overall visual quality.

## 4. Experiments

## 4.1. Evaluation Metrics

Following the evaluation protocol of Free4D [35], we evaluate all methods using frontal-view videos from publicly available examples. We use three VBench [19] metrics. First, Text Align measures semantic similarity between generated videos and input prompts using ViClip [63] video-text embeddings. Consistency measures temporal coherence of subjects and backgrounds using DINO [6] and CLIP [47] feature similarities. Finally, Aesthetic evaluates visual quality using the LAION Aesthetic Predictor [24], which considers factors such as layout, harmony, photorealism, and overall artistic impression.

## 4.2. Quantitative Results

Tab. 1 reports quantitative comparisons with prior 4D generation methods under the three evaluation settings defined by Free4D [35]. Edit4D achieves state-of-the-art performance across the VBench metrics, obtaining strong text alignment, temporal consistency, and aesthetic quality. These results demonstrate that Edit4D preserves strong generation quality while providing a more flexible and editable 4D scene representation through its stage-specific generation pipeline.

We further conduct a user study with 25 participants against Free4D, a publicly available scene level 4D generation method, using the same video generation model for a fair comparison. Participants compare the generated results in terms of 3D Structure and Visual Quality. As shown in Tab. 2, Edit4D is preferred over Free4D for both criteria, suggesting that our method provides better 3D structure and perceptual quality. Additional details and visualizations are

![](images/79c630a675eb7f702fd18802727f5694a0db166e4a2d7dfcd9ca6d2afe7e9bf1.jpg)

Figure 4. Qualitative comparison of our method with 4Real [75], Dream-in-4D [79], 4Dfy [2], and Free4D [35]. We visualize samples from experimental settings (a), (b), and (c) in Tab. 1. Our results show comparable visual quality, while providing editability through separated representations and the proposed composition algorithm.
<table><tr><td rowspan="2">Method</td><td colspan="7">Scene-Character Recomposition</td><td rowspan="2"></td><td colspan="6">Character Appearance Editing</td></tr><tr><td colspan="2">Success Rate ↑</td><td colspan="3">Preservation ↑</td><td rowspan="2">User Time (min) ↓ Study</td><td colspan="3">Success Rate ↑</td><td colspan="2">Preservation ↑</td><td rowspan="2">Time (min) ↓</td><td rowspan="2">User Study</td></tr><tr><td>GLM-4.6</td><td>Qwen3-VL</td><td>Mistral-3.1</td><td>CLIP</td><td>DINO</td><td></td><td>GLM-4.6</td><td>Qwen3-VL</td><td>Mistral-3.1</td><td>CLIP</td><td>DINO</td></tr><tr><td>Free4D</td><td>0%</td><td>40%</td><td>40%</td><td>0.964</td><td>0.901</td><td>180</td><td>16.4%</td><td>0%</td><td>40%</td><td>0%</td><td>0.957 0.871</td><td>180</td><td>10.0%</td></tr><tr><td>Ours</td><td>100%</td><td>100%</td><td>100%</td><td>0.997 0.997</td><td>45</td><td>83.6%</td><td>100%</td><td>100%</td><td>100%</td><td>0.991</td><td>0.988</td><td>45</td><td>90.0%</td></tr></table>

Table 3. Quantitative evaluation of editability. We evaluate editing success using GLM-4.6 [59], Qwen3-VL [58], and Mistral-3.1 [1], background preservation using CLIP [47] and DINO [43] similarities, as well as editing time and user preference.

provided in Appendix D.

## 4.3. Qualitative Results

We compare our method with 4Real [75], Dream-in-4D [79], 4Dfy [2] and Free4D [35] in Fig. 4. Despite generating scene components independently, Edit4D produces coherent and natural 4D scenes with visual quality comparable to existing methods. Moreover, as shown in Fig. 5, we also render the 4D scene from four novel viewpoints over time. Compared with the previous method [35], our approach maintains more stable visual quality under large camera motions, demonstrating the effectiveness of our framework that constructs the 4D scene from independently generated components.

## 4.4. Evaluation on Editability

We evaluate the editability of our method through scene– character recomposition and character appearance editing. For comparison, Free4D requires full regeneration because it represents the scene holistically. Therefore, we generate the edited reference videos using the same model, and execute the released Free4D code. Since the goal of editing is to modify the target component while preserving the rest of the scene, we define editing success based on the preservation of the original background scene. We assess this using three VLMs [1, 58, 59]. Moreover, we measure background preservation using CLIP [47] and DINO [43] similarities after masking the character regions. For the editing evaluation, we additionally conduct a user study to assess how well the remaining scene content is preserved after editing. As shown in Tab. 3, Edit4D achieves higher editing success and background preservation while reducing editing time. In Fig. 6, Free4D shows noticeable changes in background content, such as the bookshelf and rug, due to full regeneration, whereas Edit4D preserves the original scene more consistently. This efficiency comes from our asset based representation, which reuses components instead of regenerating the entire 4D scene.

w/o Initialization  
![](images/4b52b205df2fa06d52f8122999fcbef32887ab210c31fd9ea1100e056c8abad5.jpg)  
Figure 5. Visualization of generated 4D scenes from the frontal view and four diverse novel camera views over time. Our method can preserve consistency under large camera motions, producing fewer view-dependent artifacts compared to Free4D.

![](images/0367ade519ae205f7492ae1112c84757d18cde1ebfe20f9ca55728bbf743bab4.jpg)  
Figure 6. Top: Free4D, bottom: Edit4D. Since Free4D requires full regeneration for editing, background content such as the bookshelf and rug changes noticeably, whereas Edit4D better preserves the original scene during composition and appearance editing.

<table><tr><td>Init.</td><td> ${ \mathcal { L } } _ { \mathrm { c o n t a c t } }$ </td><td> $\mathcal { L } _ { \mathrm { p e n e t r a t i o n } }$ </td><td>Global Refinement</td><td>Text Align</td><td>Consistency</td><td>Aesthetic</td></tr><tr><td></td><td>√</td><td>√</td><td></td><td>26.8%</td><td>98.7%</td><td>68.0%</td></tr><tr><td>√</td><td></td><td>√</td><td></td><td>25.0%</td><td>98.9%</td><td>67.2%</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>26.6%</td><td>98.9%</td><td>67.7%</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>26.5%</td><td>98.9%</td><td>68.2%</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>28.1%</td><td>99.4%</td><td>69.0%</td></tr></table>

Table 4. Ablation study on the effect of each component in the scene–character composition and global scene refinement.

![](images/575903eeaf8b7b597feb2fc46b150a52afaeb1b55ff643825c6ab901770480df.jpg)

![](images/347698691f6009d88c4c12b988c8d32d3831af5778080ffa5e6ec7471f955e60.jpg)

![](images/57c1d0af390feb215a9fef24dfd72b2e1feb4e63a62c3c3bb86c869118a13871.jpg)

![](images/5ba294df55ba10be91f9fa23a56b173bcd110ec0b2597ed4af81856d566c048f.jpg)  
w/o contact loss  
w/o penetration loss  
Figure 7. Ablation study of our composition method. We demonstrate that our approach can reliably find the optimal placement for inserting the character based on object-specific text.

![](images/2f89978ba6c83e163e2143a63b71143b6b8a5f38a6841410d514746f71a37d4a.jpg)  
Ours w/o Global Scene Refinement

![](images/e91035443fe3e9d3131eaf9c5105460f9b6df28e9df438e5142494acf67c3cf9.jpg)  
Ours  
Figure 8. Ablation study of our global scene refinement. We show that the refinement stage improves fine-grained appearance details and enhances the overall visual coherence.

## 4.5. Ablation Study

We conduct ablation studies to analyze how each component of Edit4D affects scene–character composition and 4D generation quality. Tab. 4 shows that removing initialization, contact loss, penetration loss, and global refinement degrades different metrics, while the full model achieves the best quantitative results. Fig. 7 further visualizes the effect of each composition component. Without initialization, the optimization often converges to an incorrect placement, whereas removing the contact loss causes the character to float above the support surface, and removing the penetration loss results in penetration into the scene geometry. These results indicate that each constraint and refinement provide complementary benefits for robust placement, visual quality, and scene coherence. Moreover, Fig. 8 demonstrates the effect of the global refinement stage. The refinement stage enhances fine-grained details while improving the overall visual coherence of the composed 4D scene.

## 5. Conclusion

In this work, we presented Edit4D, a framework for editable 4D scene generation that separates background scenes and dynamic characters for object-level control and composition. Quantitative evaluations on VBench demonstrate state-of-the-art generation quality, while our editability evaluation shows that Edit4D better preserves unedited scene content and reduces editing time compared with holistic regeneration. Qualitative results further demonstrate that Edit4D produces coherent 4D scenes and remains robust under large camera viewpoint changes, supporting reliable novel-view synthesis and dynamic camera control. Our analysis validates the effectiveness of text-guided composition and contact-aware optimization for semantically and physically plausible character placement. Together, these results show that Edit4D provides a practical and flexible framework for high-quality editable 4D scene generation.

Acknowledgments This work was supported by Institute of Information communications Technology Planning Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS2023-00225630, Development of Artificial Intelligence for Text-based 3D Movie Generation).

## References

[1] Mistral AI. Mistral-small-3.1-24b-instruct-2503, 2025. 7, 17

[2] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In CVPR, pages 7996–8006, 2024. 2, 3, 6, 7, 15, 17, 20

[3] Edward Bartrum, Thu Nguyen-Phuoc, Chris Xie, Zhengqin Li, Numair Khan, Armen Avetisyan, Douglas Lanman, and Lei Xiao. Replaceanything3d: Text-guided 3d scene editing with compositional neural radiance fields. arXiv preprint arXiv:2401.17895, 2024. 3

[4] Blender Online Community. Blender - a 3d modelling and rendering package. http://www.blender.org, 2024. Blender Foundation, Stichting Blender Foundation, Amsterdam. 13

[5] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In CVPR, pages 130–141, 2023. 3

[6] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 6

[7] Hongyuan Chen, Xingyu Chen, Zexiang Xu, and Anpei Chen. Motion 3-to-4: 3d motion reconstruction for 4d synthesis. In CVPR, 2026. 5, 13

[8] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In ICCV, pages 22246– 22256, 2023. 2

[9] Comfy. Comfyui:the most powerful open source node-based application for generative ai. https://github.com/ comfyanonymous/ComfyUI, 2025. 13

[10] Xiangjun Gao, Xiaoyu Li, Yiyu Zhuang, Qi Zhang, Wenbo Hu, Chaopeng Zhang, Yao Yao, Ying Shan, and Long Quan. Mani-gs: Gaussian splatting manipulation with triangular mesh. In CVPR, pages 21392–21402, 2025. 3, 5, 13

[11] Google DeepMind et al. Veo 3 Tech Report. Technical re port, Google DeepMind, 2025. 2

[12] Chun Gu and Li Zhang. Optix-based differentiable 3d gaussian ray tracer. https://github.com/fudan-zvg/ gtracer, 2024. 6, 14

[13] Antoine Guedon and Vincent Lepetit. Sugar: Surface-´ aligned gaussian splatting for efficient 3d mesh reconstruc tion and high-quality mesh rendering. In CVPR, pages 5354– 5363, 2024. 3

[14] Mohamed Hassan, Vasileios Choutas, Dimitrios Tzionas, and Michael J. Black. Resolving 3d human pose ambigui ties with 3d scene constraints. In ICCV, 2019. 3

[15] Mohamed Hassan, Partha Ghosh, Joachim Tesch, Dimitrios Tzionas, and Michael J Black. Populating 3d scenes by learning human-scene interaction. In CVPR, pages 14708– 14718, 2021. 3

[16] Yana Hasson, Gul Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J Black, Ivan Laptev, and Cordelia Schmid. Learning joint reconstruction of hands and manipulated ob jects. In CVPR, 2019. 5

[17] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In CVPR, 2024. 3

[18] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024. Association for Computing Machinery, 2024. 3

[19] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive bench mark suite for video generative models. In CVPR, pages 21807–21818, 2024. 6

[20] Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mv-adapter: Multi-view consistent image generation made easy. In ICCV, pages 16377–16387, 2025. 3

[21] Hanwen Jiang, Shaowei Liu, Jiashun Wang, and Xiaolong Wang. Hand-object contact consistency reasoning for human grasps generation. In ICCV, 2021. 5

[22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler,¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 3

[23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment any thing. In ICCV, pages 4015–4026, 2023. 3

[24] LAION-AI. Laion-aesthetics predictor. https : / / github.com/LAION-AI/aesthetic-predictor, 2022. 6

[25] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. In ICLR, 2024. 3

[26] Xinyang Li, Zhangyu Lai, Linning Xu, Jianfei Guo, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Dual3d: Efficient and consistent text-to-3d generation with dual-mode multi-view latent diffusion. arXiv preprint arXiv:2405.09874, 2024. 3

[27] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Director3d: Real-world camera trajectory and 3d scene generation from text. In NeurIPS, pages 75125–75151, 2024. 6

[28] Zhe Li, Zerong Zheng, Lizhen Wang, and Yebin Liu. Animatable gaussians: Learning pose-dependent gaussian maps for high-fidelity human avatar modeling. In CVPR, 2024. 19

[29] Ruofan Liang, Kai He, Zan Gojcic, Igor Gilitschenski, Sanja Fidler, Nandita Vijaykumar, and Zian Wang. Luxdit: Lighting estimation with video diffusion transformer. In NeurIPS, pages 5015–5046, 2026. 6, 14

[30] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards highfidelity text-to-3d generation via interval score matching. In CVPR, pages 6517–6526, 2024. 2

[31] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In CVPR, pages 300–309, 2023. 2

[32] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. In NeurIPS, pages 22226–22246, 2023. 3

[33] Minghua Liu, Ruoxi Shi, Linghao Chen, Zhuoyang Zhang, Chao Xu, Xinyue Wei, Hansheng Chen, Chong Zeng, Jiayuan Gu, and Hao Su. One-2-3-45++: Fast single image to 3d objects with consistent multi-view generation and 3d diffusion. In CVPR, pages 10072–10083, 2024. 3

[34] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 17

[35] Tianqi Liu, Zihao Huang, Zhaoxi Chen, Guangcong Wang, Shoukang Hu, Liao Shen, Huiqiang Sun, Zhiguo Cao, Wei Li, and Ziwei Liu. Free4d: Tuning-free 4d scene generation with spatial-temporal consistency. In ICCV, 2025. 2, 3, 6, 7, 15, 17, 20

[36] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. In ICLR, 2024. 3

[37] Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jian feng Gao, Lifang He, and Lichao Sun. Sora: A review on background, technology, limitations, and opportunities of large vision models, 2024. 2

[38] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 3

[39] Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, and Zan Gojcic. 3d gaussian ray tracing: Fast tracing of particle scenes. In ACM TOG, 2024. 6, 14

[40] Gyeongsik Moon, Takaaki Shiratori, and Shunsuke Saito. Expressive whole-body 3d gaussian avatar. In ECCV, 2024. 3

[41] Thomas Muller, Alex Evans, Christoph Schied, and Alexan-¨ der Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1– 102:15, 2022. 3

[42] OpenAI. Chatgpt v5.2. https://chat.openai.com/, 2025. 5, 13, 15

[43] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 7, 17

[44] Zijie Pan, Zeyu Yang, Xiatian Zhu, and Li Zhang. Effi cient4d: Fast dynamic 3d object generation from a single view video. arXiv preprint arXiv:2401.08742, 2024. 3

[45] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In CVPR, pages 10975– 10985, 2019. 3

[46] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv, 2022. 2

[47] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn ing transferable visual models from natural language supervision. In ICML, pages 8748–8763. PmLR, 2021. 6, 7, 17

[48] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 2

[49] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023. 3

[50] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. In NeurIPS, pages 56828–56858, 2024. 3

[51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn-¨ thesis with latent diffusion models. In CVPR, 2022. 2, 14

[52] Mohamad Shahbazi, Liesbeth Claessens, Michael Niemeyer, Edo Collins, Alessio Tonioni, Luc Van Gool, and Federico Tombari. Inserf: Text-driven generative object insertion in neural 3d scenes. arXiv preprint arXiv:2401.05335, 2024. 3

[53] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 3

[54] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280, 2023. 3

[55] Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics, 29(5):2732–2742, 2023. 3

[56] Qi Sun, Zhiyang Guo, Ziyu Wan, Jing Nathan Yan, Shengming Yin, Wengang Zhou, Jing Liao, and Houqiang Li. Eg4d: Explicit generation of 4d object without score distillation. arXiv preprint arXiv:2405.18132, 2024. 3

[57] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In ECCV, pages 1–18, 2024. 3

[58] Qwen Team. Qwen3 technical report, 2025. 7, 15, 17

[59] V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Jiazheng Xu, Jiale Zhu, Jiali Chen, Jing Chen, Jinhao Chen, Jinghao Lin, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Sheng Yang, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Shangqin Tu, Shengbiao Meng, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wenkai Li, Wei Jia, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyue Fan, Xuancheng Huang, Yanling Wang, Yadong Xue, Yanfeng Wang, Yanzi Wang, Yifan An, Yifan Du, Yiming Shi, Yiheng Huang, Yilin Niu, Yuan Wang, Yuanchang Yue, Yuchen Li, Yutao Zhang, Yuting Wang, Yu Wang, Yuxuan Zhang, Zhao Xue, Zhenyu Hou, Zhengxiao Du, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. 7, 17

[60] Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. In ECCV, pages 439–457, 2024. 3

[61] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwe Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 13, 15

[62] Haiping Wang, Yuan Liu, Ziwei Liu, Wenping Wang, Zhen Dong, and Bisheng Yang. Vistadream: Sampling multiview consistent images for single-view scene reconstruction. In ICCV, pages 26772–26782, 2025. 13

[63] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In ICLR, 2024. 6

[64] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS, pages 8406–8441, 2023. 2

[65] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 13

[66] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, pages 20310–20320, 2024. 3

[67] Haoqian Wu, Zhipeng Hu, Lincheng Li, Yongqiang Zhang, Changjie Fan, and Xin Yu. Nefii: Inverse rendering for reflectance decomposition with near-field indirect illumination. In CVPR, pages 4295–4304. IEEE, 2023. 14

[68] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. In NeurIPS, pages 121859–121881, 2024. 3

[69] Jiawei Xu, Zexin Fan, Jian Yang, and Jin Xie. Grid4d: 4d decomposed hash encoding for high-fidelity dynamic gaussian splatting. In NeurIPS, pages 123787–123811, 2024. 3

[70] Haibiao Xuan, Xiongzheng Li, Jinsong Zhang, Hongwen Zhang, Yebin Liu, and Kun Li. Narrator: Towards natural control of human-scene interaction generation via relation ship reasoning. In CVPR, pages 22268–22278, 2023. 3

[71] Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao, Xinhai Liu, Xinzhou Wang, Qingxiang Lin, Jiaao Yu, et al. Hunyuan3d 1.0: A unified framework for text-to-3d and image-to-3d generation. arXiv preprint arXiv:2411.02293, 2024. 3

[72] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for highfidelity monocular dynamic scene reconstruction. In CVPR, pages 20331–20341, 2024. 13

[73] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In ECCV, pages 162–179. Springer, 2024. 3

[74] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023. 3

[75] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, Laszl ´ o Jeni, Sergey´ Tulyakov, and Hsin-Ying Lee. 4real: Towards photorealistic 4d scene generation via video diffusion models. NeurIPS, 37:45256–45280, 2024. 2, 3, 6, 7, 15, 17, 20

[76] Yuxuan Zhang, Tianheng Cheng, Lianghui Zhu, Rui Hu, Lei Liu, Heng Liu, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Evf-sam: Early vision-language fusion for text-prompted segment anything model. arXiv preprint arXiv:2406.20076, 2024. 3, 5

[77] Kaifeng Zhao, Shaofei Wang, Yan Zhang, Thabo Beeler, and Siyu Tang. COINS: Compositional human-scene interaction synthesis with semantic control. In ECCV, 2022. 3

[78] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 3

[79] Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Otmar Hilliges, and Shalini De Mello. A unified approach for textand image-guided 4d scene generation. In CVPR, pages 7300–7309, 2024. 2, 3, 6, 7, 15, 17, 20

[80] Wenjie Zhuo, Fan Ma, Hehe Fan, and Yi Yang. Vividdreamer: invariant score distillation for hyper-realistic textto-3d generation. In ECCV, pages 122–139, 2024. 2

# Edit4D: Editable 4D Scene Generation with Textual Control

Supplementary Material

## A. Implementation details

We describe the implementation of Edit4D by summarizing each component: instruction decomposition, background scene generation, dynamic 3D character generation, scene– character composition, and global scene refinement. Importantly, Edit4D is not tied to the specific pretrained models used in our implementation. Each stage is modular and can be replaced by an alternative model with compatible input and output representations, allowing the framework to incorporate different open-source models without changing the overall pipeline.

## A.1. Instruction Decomposition

We use GPT v5.2 [42] to decompose a user instruction into stage-specific prompts for background scene generation, character generation, motion generation, and scene– character composition. Each prompt contains only the information required by its corresponding stage, allowing the pipeline to operate automatically from a single user instruction while preventing unnecessary information from being propagated across stages. The instruction template is summarized in Table 5.

## A.2. Background Scene Generation

Given a textual scene description, we first synthesize a frontal background scene image (1024×1024) using a textto-image model [65]. To enable 3D reconstruction, we then generate a set of geometrically consistent multi-view images by extending the frontal view to additional camera poses via a multi-view generation framework. The resulting multi-view set is used to estimate a 3D Gaussian splatting representation of the background. In our implementation, we use VistaDream [62] to jointly produce trajectoryconditioned multi-view inpainting images and reconstruct the background Gaussian splatting in a unified pipeline.

## A.3. Dynamic 3D Character Generation

We generate a dynamic character in five steps: (1) image generation, (2) image-conditioned video generation, (3) 4D mesh estimation, (4) multi-view rendering, and (5) training dynamic Gaussian splatting. We first synthesize a character image $( 8 0 0 \times 8 0 0 )$ from a text prompt, and then generate an image-aligned motion video $( 8 0 0 \times 8 0 0$ , 60 frames) using an image- and text-conditioned video model [61]. The temporally consistent 4D mesh sequence is reconstructed using Motion 3-to-4 [7], which estimates a canonical reference mesh and predicts per-frame vertex trajectories to recover temporally coherent geometry.

To render the character together with the Gaussian splatting background efficiently, we convert the 4D mesh to a dynamic Gaussian representation. Specifically, we render multi-view images in Blender [4] using 30 cameras uniformly distributed on a sphere and bind Gaussian primitives to mesh faces following Mani-GS [10], so that Gaussians follow mesh deformations over time. To further improve temporal stability under non-rigid deformation, we incorporate a lightweight deformation field following [72] that predicts per-time offsets of Gaussian position, rotation, scale, and color. We execute all of the image and video generation steps in ComfyUI [9]. Moreover, to improve generation quality, we additionally employ task-specific positive and negative prompts to suppress common generation artifacts (e.g., shadows, background leakage, and watermark-like artifacts).

## A.4. Scene–Character Composition

Given the background and dynamic character Gaussians, we optimize a single global rigid transform of the character, parameterized by a global rotation and translation. The rotation is applied around the character center, followed by translation, while the character scale is fixed to a predefined value between 0.6 and 1.2. We initialize the character position using the bounding boxes of the character mesh and the target Gaussians. Specifically, we first align the character center with the center of the target region. Following the y-down coordinate convention, we then apply a small offset along the negative y-axis to place the character slightly above the support surface before optimization. This initialization provides a stable starting point for subsequent contact-aware optimization. Next, we minimize a weighted sum of contact and penetration losses for 1000 iterations:

$$
{ \mathcal { L } } _ { \mathrm { c o m p o s i t i o n } } = \lambda _ { \mathrm { c o n t a c t } } { \mathcal { L } } _ { \mathrm { c o n t a c t } } + \lambda _ { \mathrm { p e n e t r a t i o n } } { \mathcal { L } } _ { \mathrm { p e n e t r a t i o n } } .\tag{11}
$$

In our experiments, we set $\lambda _ { \mathrm { c o n t a c t } } = 1 0 ^ { 2 }$ and $\lambda _ { \mathrm { p e n e t r a t i o n } } =$ $1 0 ^ { 4 }$ , assigning a stronger penalty to interpenetration. For the penetration loss, we precompute a signed distance field (SDF) from the character mesh on a $\bar { 1 2 8 ^ { 3 } }$ grid with a padding of 0.05, where negative SDF values indicate points inside the character mesh. During optimization, scene points are transformed into the character coordinate system and queried against the SDF using trilinear interpolation. Points with negative SDF values are penalized to discourage interpenetration. The optimized global transform is then applied consistently to the dynamic character Gaussians across all frames.

## A.5. Global Scene Refinement

During the refinement, we jointly optimize the background scene, the canonical character Gaussian representation, and the deformation network for 4,000 iterations. During the first 500 iterations, we optimize the canonical frame using multi-view renderings sampled from the camera trajectories generated during the background scene generation stage. Also, during the remaining 3,500 iterations, we randomly sample either the canonical frame or a dynamic frame for optimization. The dynamic frames are optimized only from the frontal view to preserve character identity and temporal consistency.

For shadows, we first estimate frame-wise HDR environment maps from 25 background views using LuxDiT [29]. The per-view lighting estimates are aligned to a common world coordinate system using the corresponding camera rotations and aggregated using an element-wise median:

$$
L ( \omega ) = \operatorname* { m e d i a n } _ { k = 1 , \ldots , 2 5 } L _ { k } ( \omega ) ,\tag{12}
$$

where $\omega ~ \in ~ \mathbb { S } ^ { 2 }$ denotes a light direction and $L _ { k } ( \omega ) ~ =$ $[ R _ { k } ( \omega ) , G _ { k } ( \omega ) , B _ { k } ( \omega ) ]$ is the RGB radiance predicted for that direction in view k. This aggregation reduces the viewdependent inconsistencies while preserving lighting that is consistent across views. Given the linear HDR radiance, we compute its relative luminance $Y ( \omega )$ as follows:

$$
Y ( \omega ) = 0 . 2 1 2 6 R ( \omega ) + 0 . 7 1 5 2 G ( \omega ) + 0 . 0 7 2 2 B ( \omega ) ,\tag{13}
$$

We select high-luminance candidate regions from directions within the top 10% of the relative luminance distribution and further retain salient localized lighting regions. The selected regions are spatially smoothed to construct the directional environment map $\boldsymbol { L } _ { \mathrm { d i r } }$ . Given $\boldsymbol { L } _ { \mathrm { d i r } }$ , we sample N = 512 light directions by importance sampling according to their luminance. For each environment map pixel i, the sampling probability is defined following:

$$
p _ { i } = \frac { Y _ { \mathrm { d i r } , i } \Delta \Omega _ { i } } { \sum _ { j } Y _ { \mathrm { d i r } , j } \Delta \Omega _ { j } } ,\tag{14}
$$

where $Y _ { \mathrm { d i r } , i }$ and $\Delta \Omega _ { i }$ denote the luminance and solid angle associated with pixel i, respectively. The full environment map L is sampled in the same manner to estimate $E _ { \mathrm { a l l } }$

To obtain shadow receivers, we unproject visible background pixels into 3D world space using the rendered depth and obtain their surface normals from multiple depth observations. For each receiver point x with normal n , we trace rays toward the sampled light directions following Gaussian ray tracing [12, 39]. For each sampled direction ω<sub>i</sub>, it returns the accumulated Gaussian opacity $\alpha ( \mathbf { x } _ { \mathbf { r } } , \omega _ { i } ) \in [ 0 , 1 ]$ representing the fraction of light blocked by the character along the ray. Let $Y _ { \mathrm { d i r } }$ denote the luminance of $L _ { \mathrm { d i r } }$ . Following the standard illumination integral used in physically based inverse rendering [67], we define the irradiance as follows:

$$
\begin{array} { r l } & { E _ { \mathrm { a l l } } ( \mathbf { x } _ { \mathrm { r } } ) = \displaystyle \int _ { \mathbb { S } ^ { 2 } } Y ( \omega ) \operatorname* { m a x } ( \mathbf { n } _ { \mathrm { r } } ^ { \top } \omega , 0 ) d \omega , } \\ & { E _ { \mathrm { b l o c k e d } } ( \mathbf { x } _ { \mathrm { r } } ) = \displaystyle \int _ { \mathbb { S } ^ { 2 } } \alpha ( \mathbf { x } _ { \mathrm { r } } , \omega ) Y _ { \mathrm { d i r } } ( \omega ) \operatorname* { m a x } ( \mathbf { n } _ { \mathrm { r } } ^ { \top } \omega , 0 ) d \omega , } \end{array}\tag{15}
$$

In practice, we approximate both integrals using Monte Carlo estimation with importance samples drawn from L and $\boldsymbol { L } _ { \mathrm { d i r } }$ , respectively. Then, the relative illumination loss is as below:

$$
r _ { \mathrm { l o s s } } = \frac { L _ { \mathrm { b l o c k e d } } } { E _ { \mathrm { a l l } } + \epsilon } ,\tag{16}
$$

We apply this loss only to the background contribution in linear RGB. For an image pixel p corresponding to receiver $\mathbf { x } _ { r } .$ , the final composite is as follow:

$$
{ \bf C } _ { t } ^ { \mathrm { { o u t } } } ( p ) = { \bf C } _ { t } ^ { \mathrm { c o m p } } ( p ) - w _ { t } ^ { \mathrm { { b g } } } ( p ) r _ { \mathrm { l o s s } , t } ( { \bf x } _ { r } ) { \bf C } ^ { \mathrm { b g } } ( p ) .\tag{17}
$$

where $\mathbf { C } _ { t } ^ { \mathrm { { c o m p } } }$ is the unshadowed linear RGB composite and $w _ { t } ^ { \mathrm { b g } }$ is the soft background contribution, which prevents the attenuation from being applied to the character. The fused environment maps L and $L _ { \mathrm { d i r } }$ are fixed across the sequence, while the character dependent opacity $\alpha _ { t } ( \mathbf { x } _ { r } , \omega )$ is recomputed at every frame. Consequently, the resulting shadows follow the character motion under temporally consistent background illumination.

Throughout the refinement, we optimize the objective in Eq. (10), which combines latent-space and image-space consistency losses with $\lambda _ { 1 } = 1$ and $\lambda _ { 2 } ~ = ~ 0 . 2 5$ . We use Stable Diffusion v2.1 [51] as the diffusion prior. All final 4D scenes are rendered at a resolution of 900 × 512.

## A.6. Runtime

On a single RTX Pro 6000 Blackwell GPU, Edit4D takes on average approximately 2 hours per example: 10 minutes for static scene generation, 70 minutes for dynamic character generation, 1 minute for composition, and 45 minutes for global scene refinement. This staged design enables component-level editing of the background scene, dynamic character, and their composition, which is difficult in holistic video-to-4D pipelines.

We also compare the runtime of prior text-to-4D methods with publicly available code when generating the same number of frames as Edit4D. 4D-fy and Dream-in-4D typically require more than 24 hours per example due to expensive computation of radiance field and score distillation sampling-based optimization, while Free4D takes approximately 3 hours when including multi-view video generation and 4D Gaussian optimization. More importantly, when editing an existing scene, holistic 4D generation pipelines generally require regenerating the entire 4D result from scratch. Edit4D can instead reuse pre-generated background and character assets, requiring only scene-character composition and optional global scene refinement. This substantially reduces the cost of iterative editing and makes Edit4D a more practical framework for controllable 4D scene creation.

<table><tr><td>Task</td><td>Prompt guideline used for Instruction Decomposition</td></tr><tr><td>Background Scene Text</td><td>Objective: Extract only the background scene information from the user instruction. Rules: (i) Include only scene, environment, and objects. (ii) Exclude character identity, appearance, motion, and placement. (iii) Do not infer or introduce any information that is not explicitly mentioned.</td></tr><tr><td>Character Text</td><td>Objective: Extract only the character identity and appearance. Rules: (i) Exclude scene, background, motion, pose, and placement information. (ii) Do not infer clothing, accessories, facial features, or other attributes unless explicitly specified.</td></tr><tr><td>Motion Text</td><td>Objective: Extract only the character motion or action. Rules: (i) Exclude appearance, scene, and placement information. (ii) Describe only the character motion. (iii) Do not introduce additional actions or pose details.</td></tr><tr><td>Composition Text</td><td>Objective: Extract only the target object or region for scene-character composition. Rules: (i) Return a concise noun or noun phrase. (ii) Do not infer spatial layouts or additional relationships.</td></tr></table>

Table 5. Prompt guidelines used by GPT [42] for instruction decomposition. A user instruction is decomposed into stage-specific texts for the background scene, character, motion, and composition.

## B. Baseline and Datasets

Our goal is text-to-4D scene generation with editability. We compare our method with four baselines, including two object-centric generation methods [2, 79] and two general text-to-4D methods [35, 75]. For pairwise comparison, we follow the evaluation protocol of Free4D [35] and adopt the three experimental settings provided on its project page. Since the original prompts are not publicly available for most baseline results, we manually construct user instructions that closely match their publicly released visual content. Following Free4D, we evaluate 7, 6, and 6 samples for settings (a), (b), and (c), respectively. Because several samples are shared across settings, this corresponds to 13 unique instructions in total. Tab. 6 provides all stage specific prompts for background, character, motion, and composition used in our evaluation.

## C. Capability

In Table 7, we compare Edit4D with existing text-to-4D generation methods in terms of their capabilities. Specifically, we compare representative scene-level and objectlevel methods with respect to input modality, output representation, scene–character separation, component-level editing, and character composition. Existing methods typically generate a holistic dynamic scene representation in which the background and dynamic characters are tightly coupled. In contrast, Edit4D represents the background scene and dynamic characters as separate Gaussian representations, enabling independent component-level editing and flexible scene–character composition.

## D. Details on User Study

We conducted an anonymous pairwise user study comparing Edit4D with Free4D [35]. A total of 25 participants evaluated the rendered 4D scenes through a web-based interface, where the two methods were presented as Video A and Video B without revealing the method names. The scene order and the left–right placement of the two methods were randomized to reduce presentation bias. For a fair comparison, we regenerated the Free4D results using its publicly available implementation. Since Free4D requires a reference video, we generated the corresponding inputs using the same Qwen [58] and WAN [61] based generation pipeline used for Edit4D, with the same video resolution and frame setting. Both methods were rendered from multiple viewpoints for comparison. For the original 4D scenes, participants compared 13 scenes and answered two questions regarding 3D Structure and Visual Quality. Specifically, participants selected which result better preserved the perceived 3D structure of the scene and which result appeared visually better. We additionally evaluated editability using five scenes for both scene–character recomposition and character appearance editing. For each method, participants were shown the result before and after editing and asked which method better preserved the background content after the edit. An example of the evaluation interface is shown in Fig. 9.

## E. Evaluation on Editability

We further evaluate the editability of Edit4D through character appearance editing and scene–character recomposition. We compare against Free4D, which, to our knowledge, is the only scene-level text-to-4D generation method with publicly available source code, allowing us to directly reproduce its pipeline for controlled editing comparisons. Since Free4D represents the scene holistically, each edit requires full regeneration of the 4D scene. For a fair comparison, we generate the edited reference videos using the same video generation model as Edit4D and run Free4D using its publicly released implementation and settings.

<table><tr><td rowspan=1 colspan=1>Background Scene Text</td><td rowspan=1 colspan=1>Character Text</td><td rowspan=1 colspan=1>Motion Text</td><td rowspan=1 colspan=1>Composition Text</td></tr><tr><td rowspan=1 colspan=1>Soft-serve counter with a pastel menuboard, stacked waffle cones, napkindispenser, sealed syrup bottles, cleanwhite walls, and studio lighting</td><td rowspan=1 colspan=1>Baby panda holding a small vanilla ice-creamcone with both paws</td><td rowspan=1 colspan=1>Gently licking a vanilla ice-creamcone while holding it with both paws,with tiny head bob</td><td rowspan=1 colspan=1>Counter</td></tr><tr><td rowspan=1 colspan=1>Cel-shaded reading corner with a floorcushion, small table, closed books,bookshelf, pastel wall, and flat evenlighting</td><td rowspan=1 colspan=1>Anime-style baby panda sitting while holdingand reading an open book with both paws</td><td rowspan=1 colspan=1>Sitting and reading an open book, withblink and small head tilt</td><td rowspan=1 colspan=1>Book on the table</td></tr><tr><td rowspan=1 colspan=1>Minimal lounge with a beige fabricsofa, soft cushion, neutral wall, lightrug, wooden sofa legs, and warmnatural indoor lighting</td><td rowspan=1 colspan=1>Ring-tailed lemur holding a clear plastic bobacup</td><td rowspan=1 colspan=1>Holding a boba cup and sippingthrough a wide straw, with mouth andstraw motion</td><td rowspan=1 colspan=1>Sofa</td></tr><tr><td rowspan=1 colspan=1>Bright daytime plaza with light stonepavers, a pastel gelato kiosk, benches,planters</td><td rowspan=1 colspan=1>Baby rabbit holding a small vanilla ice-creamcone with both paws</td><td rowspan=1 colspan=1>Gently licking a vanilla ice-creamcone while holding it with both paws,with tiny head bob</td><td rowspan=1 colspan=1>Yard</td></tr><tr><td rowspan=1 colspan=1>Sound-treated rehearsal studio withmatte wood floor, acoustic wallpanels, bass traps, cable hooks,powered-off amps, soft practicallighting, and a clear floor area</td><td rowspan=1 colspan=1>Cat seated upright, singing with an open mouthwhile holding a handheld microphone</td><td rowspan=1 colspan=1>Singing into a handheld microphonewith subtle head movement</td><td rowspan=1 colspan=1>Center stage</td></tr><tr><td rowspan=1 colspan=1>Modern indoor atrium at dusk withlounge chairs, a small side table,planter boxes, stone floor, woodaccents, soft ceiling lights, and anempty circular stone landing pad</td><td rowspan=1 colspan=1>Square stone fire pit with stacked gray stones,burning firewood logs, tall orange flames, andglowing embers</td><td rowspan=1 colspan=1>The flames flicker naturally upwardand the ember glow changes subtly</td><td rowspan=1 colspan=1>Landing pad</td></tr><tr><td rowspan=1 colspan=1>Open meadow with short even grass,distant hill horizon, neutral daylight,and a round flat pad at the center</td><td rowspan=1 colspan=1>White goat with curved horns and a smallbeard, standing while lowering its head to sipbeer from a clear glass</td><td rowspan=1 colspan=1>Lowering its head to drink pale beerfrom a glass</td><td rowspan=1 colspan=1>Flat pad</td></tr><tr><td rowspan=1 colspan=1>Modern kitchen with a front-facingisland, light wood countertop, whitecabinets, matte tile backsplash,utensils, towel, and fruit bowl</td><td rowspan=1 colspan=1>Kangaroo wearing a white chef hat, holding ablack frying pan with a sunny-side-up egg and awooden spatula</td><td rowspan=1 colspan=1>Holding a frying pan and woodenspatula, gently tilting the pan whilemoving the spatula across the pan in acooking motion</td><td rowspan=1 colspan=1>Floor</td></tr><tr><td rowspan=1 colspan=1>Sunny park lawn with a red-whitegingham picnic blanket, an emptydessert plate at the center, a closedwicker basket, folded napkins</td><td rowspan=1 colspan=1>Baby monkey holding a partially unwrappedplain candy bar with both hands</td><td rowspan=1 colspan=1>Taking small bites from a partiallyunwrapped candy bar, with naturalchewing and head bob</td><td rowspan=1 colspan=1>Left-side of blanket</td></tr><tr><td rowspan=1 colspan=1>Warm wooden treehouse study nookwith a round porthole window, lowdesk, closed books, floor cushion,powered-off string lights, shelf, andpastel wood tones</td><td rowspan=1 colspan=1>Stylized 3D monster with small horns, bigexpressive eyes, a soft plush-like surface, sittingcross-legged while holding</td><td rowspan=1 colspan=1>Sitting cross-legged and reading anopen book, with gentle head tilt andblink</td><td rowspan=1 colspan=1>Cushion</td></tr><tr><td rowspan=1 colspan=1>Backyard garden with a raised woodenplanter, groomed dark soil, rows ofcarrot greens and herbs, soft overcastdaylight, and a circular clear patch ofsoil at the center of the bed</td><td rowspan=1 colspan=1>Rabbit holding a bright orange carrot</td><td rowspan=1 colspan=1>Holding and eating a bright orangecarrot, with head bob</td><td rowspan=1 colspan=1>Center of ground</td></tr><tr><td rowspan=1 colspan=1>Urban block corner with asphalt road,crosswalk paint, concrete sidewalk, aclearly visible vacant gravel lot, lowcurb lines, an empty center area forplacement</td><td rowspan=1 colspan=1>Mid-rise glass, concrete, and steel building withexterior flames wrapping along the outsidewalls and window frames</td><td rowspan=1 colspan=1>Exterior flames licking up the outsidewalls and window frames of abuilding, with hot spots and mild charon the outer surface</td><td rowspan=1 colspan=1>Vacant lot</td></tr><tr><td rowspan=1 colspan=1>A beige fabric sofa with a softcushion, soft and warm indoorlighting, cozy living room</td><td rowspan=1 colspan=1>Disney-style anthropomorphic red fox characterwearing a light gray hoodie, sitting whileholding a game controller</td><td rowspan=1 colspan=1>Playing a video game while sittingcross-legged and holding a gamecontroller</td><td rowspan=1 colspan=1>Sofa</td></tr></table>

Table 6. Text prompt used in our evaluation for each stage. We report the background scene, character, motion, and composition prompts in Edit4D.

<table><tr><td>Feature</td><td>4Real [75]</td><td>Dream-in-4D [79]</td><td>4Dfy [2]</td><td>Free4D [35]</td><td>Edit4D (Ours)</td></tr><tr><td>Input</td><td>Text, Video</td><td>Text</td><td>Text</td><td>Text, Video</td><td>Text</td></tr><tr><td>Output Representation</td><td>Dynamic Gaussian splatting for whole scene</td><td>Dynamic NeRF for whole scene</td><td>Dynamic Gaussian</td><td>Dynamic Gaussian splatting for whole scene splatting for whole scene</td><td>Background Scene &amp; Dynamic Character Gaussian splatting</td></tr><tr><td>Scene-Character Separation</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Component Editing</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Character Composition</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td></tr></table>

Table 7. Capability comparison of previous 4D generation pipelines with Edit4D. Edit4D supports explicit scene-character separation, component-level editing, and character composition.

![](images/7510dc7511ed86b9dc35791cc2b2d113656e5cd58480e3d1a335a0334637a619.jpg)  
Figure 9. User study interface. Participants evaluate Edit4D and Free4D based on 3D structure and visual quality for the original 4D scenes, as well as background preservation after editing.

Our evaluation focuses on how well the unedited background is preserved after editing. We evaluate five character scenes for each of the two editing tasks and compare the original and edited renderings across multiple camera viewpoints and time steps shown in Fig. 10. To prevent changes in character position, scale, or appearance from affecting the evaluation, we localize the character using Grounding DINO [34] and apply a shared mask covering the union of the character regions in the original and edited images. Thus, the evaluation measures the consistency of the background outside the edited region.

We assess editing success using three VLMs, GLM-4.6 [59], Qwen3-VL [58], and Mistral-3.1 [1], based on whether the original background scene is preserved after editing. Their decisions are aggregated across sampled viewpoints and time steps. We separately measure background preservation using CLIP [47] and DINOv2 [43] similarities computed between the masked original and edited images, where higher similarity indicates better preservation of the unedited background. As reported in the main paper, Edit4D consistently achieves higher editing success and background preservation than Free4D for both editing tasks while requiring substantially less editing time.

## F. Additional qualitative results

We provide additional qualitative results to further demonstrate the editing capability and flexibility of Edit4D. As shown in Fig. 11, Edit4D supports asset insertion, scene– character recomposition, and character appearance modification using the scene presented in the main paper. In the asset insertion example, we manually place a hat represented as a Gaussian Splatting asset onto the character. Scene– character recomposition is performed by specifying “left sofa” as the target region and repositioning the character through our composition framework. To modify the character appearance, we construct a 3D mask for “cloth” using our masking method and update the color values within the masked region. Since the background and character are represented as reusable 3D assets, these edits can be performed while preserving the remaining scene content without regenerating the entire 4D scene.

![](images/0a56ab76299540ce71bb90b551e04baf9d4c89ad33d81b58820a955bb4d5c948.jpg)  
Figure 10. Qualitative examples used in the editability evaluation. We compare Free4D and Edit4D for scene–character recomposition and character appearance editing, showing both frontal and rotated views. Free4D requires full scene regeneration for each edit, whereas Edit4D reuses the existing scene assets and better preserves the unedited background content across viewpoints.

![](images/5f68518329d8e4d15eae74c389a08e9c14d95e71a205b2cdbddbd3852580f16e.jpg)  
Original Results

![](images/0d0027dd45fde051414a2fae9744e32d3505e2baeb04301da86eecb8d37242e0.jpg)  
Insert Asset

![](images/5bf42e1df9255b101b02beac749ee1c4bfdc41fb455d4cdf0ef53ec16367dcdf.jpg)  
Change Composition

![](images/c2d2b858accbfbbadb3a180b043e2d06c5ade554ca20933a8e302f8bd5642d0c.jpg)  
Modify Character Appearance

![](images/fbf7711f4d5ab2f3b9aa5df13fada249ebea32d8618004bdd0e028e9e872693f.jpg)

![](images/9df5989d0118390145ad4a5adbc0b0fb68edabc775821db06e3ca0b7b804153a.jpg)  
Figure 11. Object-level editing enabled by Edit4D. Our modular 4D representation supports character appearance modification, asset insertion, and scene–character recomposition without regenerating the entire scene.

![](images/1af377faf5fff7149d1dba6daa23d2563672088281216160e5beaf3f32c72ab0.jpg)

![](images/52a83bba019b5bed9d4266fcfd7a94c2217d09a88931b5eee2c1e531f4def73b.jpg)  
(a) Composition of multiple characters

![](images/3cd1b758844041350609dee31e7ddfb945671cff67d6b04ce23f4aa0b05fb6e1.jpg)  
(b) Lying character

![](images/9ccedffda48ce858394d414b4ba6fd7043d2f5bf28e08164cb88613a17eba454.jpg)  
(c) Flying bird  
Figure 12. Additional results of our pipeline. We segment target regions from text and align the character via the proposed composition algorithm.

We further demonstrate the flexibility of our scene– character composition framework in Fig. 12. As shown in Fig. 12a, our framework can incorporate alternative dynamic character representations. Specifically, we integrate Animatable Gaussians [28] and obtain plausible multicharacter compositions, while visualizing left, center, and right floor masks for controllable placement. To maximize compatibility with diverse character representations, our main pipeline adopts a video-to-mesh model that does not rely on explicit rigging or skeleton annotations.

Fig. 12b further presents a more challenging interaction with a large contact area between the character and the scene. For non-contact interactions such as hovering or flying, we replace the contact objective with a distancethresholded variant that maintains a target distance from the scene surface. As shown in Fig. 12c, this produces stable and non-intersecting floating characters within the same optimization framework. Together, these results demonstrate that the proposed composition framework can accommodate diverse character representations, contact configurations, and spatial relationships.

In addition, we present additional generation results across diverse text prompts in Fig. 13, showing that Edit4D consistently produces coherent and visually plausible 4D scenes. Furthermore, Fig. 14 provides additional novelview renderings, demonstrating that our method maintains high visual quality under large camera viewpoint changes.

## G. Analysis on Contact Map

We visualize the contact intensity map produced by our composition stage in Fig. 15. Given a composition text, Edit4D first localizes the corresponding target region in the generated background scene and then optimizes the character placement using contact-aware constraints. The highcontact regions align well with the target area specified by the composition text, showing that Edit4D can localize target placement regions and establish plausible characterscene contact. In addition, the character contact maps show that the contact response is mainly activated around physically plausible support regions, such as the lower body or contact-facing surface of the character. Overall, these analyses demonstrate that Edit4D enables physically plausible scene–character composition while maintaining the editability and rendering quality of the generated 4D scene.

![](images/35169b5860b25991f951a66d86a66f4c54e2b44792de7f0adb0821028d5e299b.jpg)  
Figure 13. Additional Qualitative comparison of our method with 4Real [75], Dream-in-4D [79], 4Dfy [2], and Free4D [35]. We visualize samples from experimental settings (a), (b), and (c) in the main paper.

![](images/bacf50fc2939c85c17e8a2befcb6f31b4a2bbb9a79e89b57ad925732b9908df7.jpg)  
Figure 14. Additional visualization of generated 4D scenes from the frontal view and four diverse novel camera views over time.

![](images/472fab01a8096fa3cacbaac12ad53aeef5bc8e50273399466bb192443eae76f0.jpg)  
Figure 15. Rendered contact map from 3D visualizer. Black and blue indicate the scene object and character, respectively, while brighter yellow represents higher contact intensity. We also visualize the target object mask in the figure.

![](images/60397f97b6884d61d94474d744881742268bef3fecd2d7b5f550e8bd60f1caaf.jpg)  
Scene Image and Mask

![](images/255b2ed40bc79281d2a6cc883f9086c69c2f406d95509960279c271b1f8b2037.jpg)

![](images/47ef1d03307a681d79f883d7156f6e2ed84b725b534cc0285c62ba55bce83ea4.jpg)

Rotate 30<sup>◦</sup>  
![](images/204a801e527567025a33ba537e1b3077f23492b35db9afe592a145e76303b2b1.jpg)  
Rotate 70<sup>◦</sup>  
Figure 16. Failure case caused by an occluded composition target. The partially hidden chair produces an incomplete target region, leading to inaccurate character placement and intersection with the desk geometry in rotated views.

## H. Limitations

Our scene–character composition relies on text-guided segmentation to identify the target region for character placement. Therefore, the target object should be sufficiently visible and spatially localized in the input scene. Fig. 16 shows a representative failure case in which the target chair is partially occluded by the desk. Although the visible chair region can be identified, the incomplete target region leads to inaccurate character placement and intersection with the surrounding desk geometry. This limitation could be alleviated by incorporating stronger 3D scene understanding or occlusion-aware target completion into the composition process.

Our pipeline also relies on several pretrained modules for intermediate predictions. Errors from these components can propagate to the final 4D scene. For example, imperfect dynamic character reconstruction may introduce temporal or geometric inconsistencies, while inaccurate text-guided segmentation can lead to errors in target localization and composition. Similarly, artifacts in the generated image or reference video can affect the quality of the reconstructed scene or character. Although our refinement stage can alleviate moderate inconsistencies, severe upstream errors may still remain in the final result. Improving robustness to such intermediate prediction errors remains an important direction for future work.

## I. Future Work

Edit4D opens several directions for extending editable 4D scene generation. One promising direction is to incorporate more diverse editable assets into the component-wise representation. Since Edit4D constructs background scenes and dynamic characters as separate 3D components, these assets can be inserted or reused across different compositions. Another useful extension is interactive control over scene–character composition, where users can refine placement, contact regions, and spatial relationships after generation. Improving target localization under occlusion through stronger 3D scene understanding or target completion could further increase the robustness of composition. In addition, reducing sensitivity to errors in text-guided segmentation, dynamic character reconstruction, and generated reference images or videos would improve consistency in challenging cases. These directions would further connect text-driven 4D generation with practical graphics-based editing workflows.