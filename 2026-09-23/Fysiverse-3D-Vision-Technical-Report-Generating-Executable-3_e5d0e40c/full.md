# Fysiverse-3D-Vision Technical Report: Generating Executable 3D Worlds from Images through Unified Spatial Reasoning

Dingkang Yang<sup>†</sup>, Yizhou Liu, Wendong Cheng, Zizhi Chen, Shunli Wang, Yang Liu, Hongsheng Li<sup>§</sup>, Lihua Zhang<sup>§</sup>

Physical Superintelligence Lab, Fysics AI

College of Intelligent Robotics and Advanced Manufacturing, Fudan University Multimedia Laboratory (MMLab), The Chinese University of Hong Kong College of Electronic and Information Engineering, Tongji University

<sup>†</sup>Project lead, <sup>§</sup>Corresponding author

## Abstract

Generative models have substantially advanced image-conditioned 3D content creation, yet generating controllable and executable 3D scenes from a single image remains challenging. Existing 3D generative approaches can synthesize visually plausible objects and scenes, but their spatial layout estimation is often tightly coupled with specific asset generators. Consequently, they struggle to jointly model object semantics, metric geometry, and scene-level spatial relationships, which are essential for interactive editing, physical simulation, and embodied applications. We propose Fysiverse-3D-Vision, a unified vision-languagegeometry framework for generative 3D scene reconstruction and executable asset construction from a single image. The core idea is to establish a shared representation where spatial reasoning and geometric reconstruction mutually enhance each other, allowing object layouts to be inferred beyond the constraints of individual asset generators. Specifically, our model integrates textual supervision, semantic visual cues, and geometric representations within a unified Transformer to capture scene context, metric geometry, and object-level interactions. An object-conditioned layout module further performs cross-attention between target object representations and global geometric features to predict object translation, rotation, and scale. The training process progressively learns geometry-language alignment, introduces layout reasoning while preserving reconstruction capability, and refines physical consistency through collision-aware optimization. By separating spatial layout reasoning from asset synthesis, Fysiverse-3D-Vision provides an adaptable interface for interactive scene editing, object-level manipulation, and executable 3D content generation. Extensive experiments demonstrate that our framework achieves superior geometric consistency, layout estimation, rendering quality, and physical property understanding compared with existing approaches.

Date: September 23, 2026 Corresponding: dicken@fyscis.ai, hsli@ee.cuhk.edu.hk, lihuazhang@fudan.edu.cn Github: https://github.com/Fysics-AI/Fysiverse-3D-Vision Hugging Face: https://huggingface.co/Fysics-AI/Fysiverse-3D-Vision

## 1 Introduction

Recent advances in generative modeling are reshaping computer vision from recognition centered perception toward the construction of structured, controllable, and interactive 3D worlds. Beyond synthesizing visually realistic content, generative models are increasingly expected to recover latent scene structures and produce representations that support editing, simulation, and embodied interaction. In embodied intelligence, digital simulation, and immersive applications [30, 42, 57, 58, 62, 64], a central challenge is to transform visual observations into executable 3D scenes, where objects are jointly represented through appearance, metric geometry, spatial relationships, and physical functionality. However, conventional recognition and reconstruction paradigms often focus on isolated semantic prediction or geometric recovery [6, 16, 26, 50], making it dificult to generate coherent scene structures and support subsequent interaction under incomplete visual observations.

Generating 3D scenes from a single image [13, 14, 19, 29] represents an important step toward this goal. A successful system needs to recover not only the appearance and geometry of individual objects, but also their spatial organization within a shared environment [5, 24]. This problem is particularly challenging because a single viewpoint provides limited information about object structure, depth, and physical relationships. Therefore, the model must jointly reason about object identity, metric geometry [39, 43], and inter-object configurations [12, 33], while handling occlusions, ambiguous observations, and diverse real-world appearances. Existing approaches for image-based 3D scene generation can be broadly categorized into reconstruction-based methods [10, 59] and retrieval-based methods [32, 51]. Reconstruction based approaches typically leverage scene-level annotations [13, 14] to estimate geometry, layouts, and object attributes directly, whereas retrieval-based approaches identify suitable assets from large databases [9, 27] and optimize their alignment with observed scenes. Although these methods have significantly advanced scene understanding, they remain limited by insuficient scene-level supervision, restricted asset diversity, or complicated multi-stage optimization procedures. Recent object-centric foundation models [47] provide a new direction by introducing strong 3D priors. Methods built upon these models [5, 19, 24, 29] can extend object synthesis toward scene generation through additional interaction modeling and post-training. However, their layout reasoning capability is usually coupled with individual object generators, making spatial layouts dependent on specific synthesis models rather than serving as a general scene-level representation.

Beyond geometry generation, recent studies have explored extending object foundation models toward executable 3D assets by incorporating physical and functional knowledge. These eforts enable capabilities including physical property prediction [2, 3], part-level decomposition [23, 52], and articulated structure modeling [25, 28]. Nevertheless, these capabilities are commonly introduced through task-specific post-training strategies. The resulting models often lack a unified representation that can simultaneously capture semantic understanding, spatial reasoning, geometric reconstruction, and executable attributes. Consequently, current methods still face dificulties when transferring from visual reconstruction to interactive scene generation, where object placement, physical properties, and functional structures need to be jointly considered. A more general framework is required to establish an intermediate spatial representation between scene perception and executable asset construction. Recent advances in unified multimodal models [18] provide an opportunity to address this limitation. Unlike isolated feed-forward geometry predictors [39, 41], unified models can integrate semantic reasoning and geometric perception within a shared representation space. Since spatial layouts depend on both object relationships and object-specific characteristics [44], language-guided reasoning ofers complementary semantic knowledge beyond the implicit priors learned from large-scale object datasets [7– 9, 47]. Therefore, a native vision-language-geometry model equipped with spatial understanding [40] and geometric reconstruction ability [43] provides a promising foundation for building interactive and executable 3D environments.

Based on this insight, we propose Fysiverse-3D-Vision (F3V), a unified vision-language-geometry framework that enables executable 3D scene reconstruction by integrating spatial reasoning and geometric modeling. Instead of introducing an independent layout predictor trained only with scene annotations, our approach allows object placement to emerge from a shared representation learned through multimodal understanding and geometric reconstruction. We construct large-scale interleaved training samples through a simulation-based data generation pipeline and jointly encode textual tokens, semantic visual tokens, and geometric tokens within a single Transformer. This unified representation enables the model to preserve high-level scene semantics while recovering accurate 3D structures. Based on the learned spatial features, an object-conditioned layout module further establishes interactions between target object tokens extracted from scene observations and global geometry tokens, allowing direct prediction of object translation, rotation, and scale. Finally, we refine the predicted layouts with scene-level supervision and a collision-aware objective to improve physical validity.

As shown in Figure 1, our model decouples spatial layout reasoning from specific asset generators and establishes a flexible interface between scene understanding and executable asset construction. This design enables a wide range of downstream applications, including interactive scene editing, object manipulation, and simulation. By reasoning over shared semantic and geometric representations, F3V alleviates common challenges in single-image reconstruction, such as incomplete object observations, reconstruction artifacts, and inconsistent spatial configurations.

![](images/1f39d7249d66c18bb6647a15bc9ad509531caec21568903b1df6c84c956e9275.jpg)  
Figure 1 From a single image, Fysiverse-3D-Vision reconstructs individual objects, independently predicts their translation, rotation, and scale, and attaches physical, material, afordance, and part-level attributes for editing and simulation.

## 2 Related Work

## 2.1 3D Scene Generation

Image-based 3D scene generation aims to recover structured and interactive environments from visual observations, serving as an important foundation for embodied intelligence, simulation, and immersive content creation. Existing approaches mainly difer in how 3D objects are obtained and organized. Retrieval-based methods construct scenes by searching suitable assets from large-scale 3D repositories [13, 14] and leveraging vision-language models (VLMs) [12, 33] to infer semantic layouts and object relationships. Although these methods benefit from high-quality existing assets, their performance is inherently limited by the coverage, diversity, and availability of the underlying databases, which restricts their adaptation to open-world scenarios. Recent generation-based approaches attempt to directly synthesize 3D content from images without relying on predefined asset collections. CAST [53] introduces component-aware scene reconstruction with SDF-based physical refinement to improve structural consistency. MIDI [19] and SceneGen [29] explore single-pass generation of multiple objects while modeling implicit interactions and instance-level poses. PartCrafter [23] further extends compositional 3D generation by jointly modeling object and part structures through latent difusion transformers. More recent methods, including I-Scene [24], 3D-Fixer [56], and SAM3D [5], focus on improving scalability and generalization through large-scale training data and diversified scene synthesis strategies. Despite these advances, existing approaches typically regard spatial reasoning and geometric generation as separate components. The lack of a unified representation that jointly captures scene semantics, geometry, and object interactions remains a major challenge for executable and controllable 3D scene understanding.

## 2.2 Executable Asset Generation

The increasing demand for embodied agents and physical simulation has shifted 3D generation from visual realism toward executable assets with explicit structures, physical properties, and interaction capabilities. Recent studies have explored enriching generated objects with functional and physical information. PartPacker [34] improves part-level generation eficiency through a dual-voxel difusion architecture, enabling automatic part discovery without requiring explicit segmentation. OmniPart [52] further introduces part-aware latent modeling under bounding-box constraints, supporting structured generation with improved editability and spatial control. For articulated objects, DreamArt [28] utilizes generated videos to optimize movable object structures, while URDF-Anything [22] directly predicts URDF representations for robotic simulation. However, these approaches still depend on specific annotations or input conditions, and often lack comprehensive modeling of appearance, physical attributes, and interaction-related properties. Physics-aware asset generation methods have recently attempted to bridge the gap between visual reconstruction and simulation requirements. Existing solutions incorporate material characteristics and physical parameters, but they usually treat diferent object categories independently or focus on limited physical factors. PhysXGen [3] introduces a unified framework for generating 3D assets with physical attributes, including size and density. Based on this direction, PhysX-Anything [2] extends physical asset generation to real-image inputs and produces simulation-ready objects with explicit physical properties. Nevertheless, executable asset generation still requires accurate scene-level understanding, since object functionality depends not only on individual asset properties but also on their spatial context and relationships within the environment.

![](images/66fc9edbbc63216f970ed968c5a46f606508be9dd678c4ef2d8fdee63ab26f4c.jpg)  
Figure 2 Three-stage simulation-data curation pipeline: heterogeneous indoor datasets are merged, compact object-centric regions are cropped, and samples are categorized by structural completeness, visibility, and semantic quality.

## 2.3 Unified 3D Modeling

Recent advances in unified multimodal foundation models [1, 21, 36–38, 48] have demonstrated a transition from isolated task-specific architectures toward shared understanding and generation paradigms. BLIP3-o [4] shows that separating multimodal understanding and generation through an “understand first, generate later” strategy can improve crossmodal alignment, while Bagel [60] introduces a Mixture-of-Transformer-Experts (MoT) design to reduce interference among heterogeneous objectives. Inspired by these developments, unified modeling has also emerged in the 3D domain. ShapeLLM-Omni [55] and Omni123 [54] investigate native 3D foundation models by introducing discrete 3D representations and jointly training 2D and 3D modalities. Omni-View [17] combines geometry and texture representations to unify scene understanding, novel-view synthesis, and geometric estimation. UniUGG [49] and G2VLM [18] further incorporate geometric information into multimodal language models, enabling stronger spatial reasoning and reconstruction capabilities. However, current unified 3D models mainly focus on representation learning or object-level generation, while the potential of shared understanding-generation architectures for scene-level spatial reasoning and executable layout prediction remains largely unexplored.

## 3 Methodology

## 3.1 Data Governance Procedure

As illustrated in Figure 2, we develop a three-stage data preparation pipeline that transforms heterogeneous 3D resources into structured training samples. The pipeline consists of unified multi-source data integration, object-centric region extraction, and quality-aware sample refinement. Specifically, we collect indoor scene data from 3D-FUTURE [14],

![](images/f2f5f4f4b9c3cf65bb332fb94e52d1bbb5464ef8594e69645e8a5b72157f9dfb.jpg)  
Figure 3 Our method decouples executable asset generation from 3D layout prediction. Completed object crops are used to generate textured, physical, and part-level assets, while a unified vision-language-geometry Transformer predicts object translation, rotation, and scale from scene geometry and object-condition tokens. Training proceeds through geometry-language pretraining, layout injection, and real-data refinement.

SAGE-10K [45], and IL3D [63], covering a broad range of room layouts, furniture configurations, and spatial arrangements. After applying consistent preprocessing, the resulting dataset contains approximately 46K spatial instances, 189K object instances, and 254K multi-view rendered images, providing diverse supervision for spatial understanding and geometric reasoning.

Directly using raw indoor scenes is challenging due to their large spatial extent, uneven object distributions, and limited informative regions. To obtain compact regions suitable for model learning, we construct a spatial relation graph where object centers are treated as nodes and object pairs are connected according to their 3D distances [35]. We first identify spatially coherent object groups through graph connected-component analysis and then apply DBSCAN clustering [11] to locate dense regions within these groups. Candidate regions are selected according to object quantity, spatial compactness, density, and overlap constraints. The retained regions are subsequently normalized by re-centering objects, aligning scene scales, completing consistent room boundaries, and rendering additional views with optimized camera configurations. This process improves object visibility and provides more informative observations for spatial reasoning. After region extraction, we further perform quality-based filtering according to structural completeness, object arrangement, rendering quality, and semantic relevance. Samples that satisfy all criteria are directly preserved, while partially qualified samples are geometrically aligned and normalized before inclusion. The resulting dataset maintains the diversity of multi-source 3D environments while providing clean and structured scene representations for unified spatial understanding.

## 3.2 Unified Geometry-Language Representation

As shown in Figure 3, F3V builds upon a unified 3D vision-language model, where semantic understanding and geometric perception are learned within the same representation space. Object layout estimation requires more than predicting object coordinates from masks. A reliable layout should simultaneously capture object identity, scene context, and metric spatial structure. Learning these capabilities independently often leads to insuficient interaction between semantic reasoning and geometric reconstruction. Therefore, we adopt a unified backbone that enables text tokens, semantic visual tokens, and geometric visual tokens to interact through shared multimodal self-attention. Given text tokens $\mathbf { X } ^ { t x t }$ semantic visual tokens $\mathbf { V } ^ { s e m }$ , and geometric visual tokens $\mathbf { V } ^ { g e o }$ , the unified input sequence is $\mathbf { X } _ { 0 } = [ \mathbf { X } ^ { t x t } ; \mathbf { V } ^ { s e m } ; \mathbf { V } ^ { g e o } ]$ All modalities are fused via unified Multi-Modal Self-Attention:

$$
{ \bf X } _ { l + 1 } = { \bf M } { \bf M } { \bf S } { \bf A } ( { \bf X } _ { l } ) , \qquad { \bf H } = { \bf X } _ { L a s t } .\tag{1}
$$

The shared attention allows diferent modalities to contribute complementary information. Specifically, semantic tokens capture object categories, language instructions, and scene-level context, while geometric tokens encode depth, camera

information, and spatial structures. Their joint interaction produces hidden representations that preserve both semantic awareness and geometric reasoning ability. The geometric branch decodes H into local point maps, global point maps, and camera poses:

$$
( \hat { \mathbf { P } } ^ { l o c } , \hat { \mathbf { P } } ^ { g l o b } , \hat { \mathbf { G } } ) = \mathcal { D } _ { g e o } ( \mathbf { H } ) .\tag{2}
$$

These geometric predictions provide explicit structural supervision rather than serving only as auxiliary outputs. By constraining the shared representation with 3D reconstruction objectives, the model acquires geometry-aware features that can be further utilized for downstream layout prediction.

## 3.3 Object-Conditioned Layout Branch

The layout prediction module is introduced on top of the geometry-aware hidden representations. Instead of estimating object placement directly from image-level features, the proposed design explicitly incorporates object-specific conditions and scene-level geometric information. For clarity, the reference RGB image is denoted as $R .$ Given �, a binary target mask $M \in 0 , 1 ^ { \widecheck { H } \times W }$ , and a point map $P \in \mathbb { R } ^ { H \times W \times 3 }$ aligned with � [41], the object condition encoder � [5] extracts object-centric representations and maps them into the unified hidden space:

$$
( \tilde { \mathbf { O } } , \mathbf { Q } ) = \big ( \boldsymbol { \Phi } ( \mathbf { O } ) , \mathbf { Q } \big ) , \quad ( \mathbf { O } , \mathbf { Q } ) = C ( R , M , P , R \odot M ) ,\tag{3}
$$

where $C$ is the condition encoder, $\mathbf { O } \in \mathbb { R } ^ { B \times K \times d _ { o } }$ denotes the object tokens before projection, $\tilde { \mathbf { O } } \in \mathbb { R } ^ { B \times K \times d }$ denotes the aligned object tokens in the unified hidden space, � is the number of object tokens, and $\mathbf { Q }$ gives the corresponding object-token positions.

The layout decoder operates on geometry representations associated with the reference view. Specifically, $\mathbf { G } _ { 1 }$ denotes the first-view geometry tokens selected from the unified hidden states H, while $\mathbf { U } _ { 1 }$ represents their corresponding token positions. The decoder establishes object-to-scene interactions through cross-attention, where projected object tokens are used as queries and reference-view geometry tokens provide keys and values:

$$
\mathbf { Z } = \operatorname { S o f t m a x } \left( \frac { ( W _ { q } \tilde { \mathbf { O } } ) ( W _ { k } \mathbf { G } _ { 1 } ) ^ { \top } } { \sqrt { d } } \right) W _ { \nu } \mathbf { G } _ { 1 } .\tag{4}
$$

Here, $W _ { q } , W _ { k }$ , and $W _ { \nu }$ are learnable query, key, and value projections, respectively. � denotes the attention dimension, while token positions Q and $\mathbf { U } _ { 1 }$ provide positional information for layout decoding. Although the decoder uses geometry tokens from a single reference view, $\mathbf { G } _ { 1 }$ does not represent an isolated observation. Before being selected, these tokens have already exchanged information with multi-view geometry features through the shared self-attention mechanism in the unified backbone. The layout heads predict object translation, rotation, and scale from the fused representation:

$$
\hat { \bf t } = f _ { t } ( { \bf Z } ) , \qquad \hat { \bf r } = f _ { r } ( { \bf Z } ) , \qquad \hat { \bf s } = f _ { s } ( { \bf Z } ) ,\tag{5}
$$

where $\hat { \mathbf { t } } \in \mathbb { R } ^ { 3 }$ represents the object-center translation, $\hat { \mathbf { r } } \in \mathbb { R } ^ { 3 \times 3 }$ denotes the 9D raw rotation prediction, and $\hat { \mathbf { s } } \in \mathbb { R } ^ { + }$ indicates the object scale. We initialize the layout decoder from the camera decoder and initialize $f _ { t }$ and $f _ { r }$ from the corresponding camera pose heads. This transfers rigid transformation priors learned from camera estimation to object-level pose prediction.

## 3.4 Multi-Stage Training Strategy

Stage 1: Learning a Shared Geometry-Language Space. The first training stage focuses on establishing a unified representation before introducing the layout prediction module. The objective is to construct a latent space that simultaneously captures semantic reasoning and metric 3D structures, providing a foundation for subsequent object placement prediction. The understanding branch is optimized with image-text question answering supervision. Given the input frames �, question text �, and answer sequence �, the language decoder is trained through next-token prediction:

$$
\mathcal { L } _ { C E } = - \sum _ { i = 1 } ^ { L } \log p _ { \theta } ( a _ { i } | a _ { < i } , I , T ) .\tag{6}
$$

where $a _ { i }$ denotes the �-th answer token, and $p _ { \theta }$ represents the token distribution generated by the language decoder. This objective maintains the pretrained VLM capability in object recognition, instruction following, and spatial relation reasoning.

Meanwhile, the geometry branch receives multi-view RGB observations together with depth and camera supervision. The purpose is to encourage the shared latent space to encode metric scene structures using the $\pi ^ { 3 }$ [43] objective:

$$
\mathcal { L } _ { V G } = \lambda _ { l } \mathcal { L } _ { l o c } + \lambda _ { g } \mathcal { L } _ { g l o b } + \lambda _ { c } \mathcal { L } _ { c a m } ,\tag{7}
$$

where $\mathcal { L } _ { l o c }$ and $\mathcal { L } _ { g l o b }$ constrain local and global point-map reconstruction, respectively, and $\mathcal { L } _ { c a m }$ supervises cross-view camera pose estimation. Through joint optimization of semantic and geometric objectives, Stage 1 learns a representation that combines category-level understanding with coordinate-aware spatial perception. The overall objective of this stage is formulated as $\mathcal { L } _ { s 1 } = \mathcal { L } _ { C E } + \mathcal { L } _ { V G }$ . After obtaining the shared geometry-language representation, we freeze the backbone and train the newly introduced layout module using alignment data [63]. This warm-up process aligns object-conditioned features with the geometry-aware latent space before full optimization. For an object with ground-truth layout $( \mathbf { t } ^ { * } , \mathbf { r } ^ { * } , \mathbf { s } ^ { * } )$ the predicted layout (<sup>ˆ</sup>t, rˆ, sˆ) is supervised using the Smooth-L1 [15] penalty $\zeta _ { \delta } ( \cdot )$

$$
\begin{array} { r } { \mathcal { L } _ { t } = \zeta _ { \delta } ( \hat { \mathbf { t } } - \mathbf { t } ^ { * } ) , \quad \mathcal { L } _ { r } = \langle \mathbf { W } _ { r } , \zeta _ { \delta } ( \hat { \mathbf { r } } - \mathbf { r } ^ { * } ) \rangle , \quad \mathcal { L } _ { s } = \zeta _ { \delta } ( \hat { \mathbf { s } } - \mathbf { s } ^ { * } ) , } \end{array}\tag{8}
$$

Translation and scale are optimized through direct regression, while rotation prediction adopts an element-wise weighting matrix ${ \bf W } _ { r }$ to emphasize yaw-related components under the �-up assumption. The operator $\langle \cdot , \cdot \rangle$ represents the averaged weighted element-wise summation. The alignment objective is defined as:

$$
\mathcal { L } _ { \mathrm { a l i g n } } = \lambda _ { t } \mathcal { L } _ { t } + \lambda _ { r } \mathcal { L } _ { r } + \lambda _ { s } \mathcal { L } _ { s } ,\tag{9}
$$

where $\lambda _ { t } , \lambda _ { r }$ , and $\lambda _ { s }$ balance the contributions of translation, rotation, and scale terms.

Stage 2: Layout Injection with Geometry Preservation. After establishing the shared geometry-language representation, we introduce the layout branch while maintaining the geometry reconstruction objective [14, 45]. The overall optimization target is formulated as $\mathcal { L } _ { s 2 } = \mathcal { L } _ { V G } + \lambda _ { a } \mathcal { L } _ { a l i g n }$ . During this stage, the layout module infers object translation, rotation, and scale from the interaction between $\mathbf { G } _ { 1 }$ and O<sup>˜</sup> under the supervision of $\mathcal { L } _ { a }$ . Meanwhile, retaining the geometry reconstruction loss ${ \mathcal { L } } _ { V G }$ prevents the shared representation from degrading toward layout-specific optimization and maintains the original 3D reconstruction capability.

Stage 3: Collision-Aware Layout Refinement. In the final stage, we keep the unified backbone fixed and optimize only the layout-related parameters. Since the geometry-aware representation has already been suficiently established, this stage focuses on improving the physical validity and spatial plausibility of object placement. The layout regression objective remains consistent with $\mathcal { L } _ { a l i g n }$ . To further reduce physically implausible object intersections, we introduce a BEV-based collision constraint:

$$
\mathcal { A } _ { j } ^ { b e \nu } = \left| \Pi _ { b e \nu } ( \hat { \mathcal { B } } ) \cap \Pi _ { b e \nu } ( \mathcal { B } _ { j } ) \right| ,\tag{10}
$$

$$
\mathcal { L } _ { c o l } = \sum _ { j } \mathbb { I } [ \Delta _ { z } ( \hat { \mathcal { B } } , \mathcal { B } _ { j } ) > 0 ] \mathcal { A } _ { j } ^ { b e \nu } ,\tag{11}
$$

where $\hat { \mathcal B }$ is the predicted oriented bounding box of the target object, $\mathcal { B } _ { j }$ represents the bounding box of the $j \cdot$ -th neighboring object, and $\Pi _ { b e \nu } ( \cdot )$ projects a 3D bounding box onto the bird’s-eye-view plane. $\mathcal { A } _ { j } ^ { b e \nu }$ measures the intersection area between the two projected footprints. The term $\Delta _ { z } ( \hat { \mathcal { B } } , \mathcal { B } _ { j } )$ evaluates vertical overlap, ensuring that the collision penalty is applied only when two objects intersect along the height direction. The final optimization objective is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { s 3 } = \mathcal { L } _ { a l i g n } + \lambda _ { c o l } \mathcal { L } _ { c o l } , } \end{array}\tag{12}
$$

where $\lambda _ { c o l }$ controls the contribution of the collision constraint.

## 3.5 Inference Pipeline

During inference, F3V takes a scene image and a target object mask as input. If the mask is unavailable, an openvocabulary segmentation model is first applied to identify object regions [31]. The extracted mask is decomposed into individual object components, and each component is used to generate a masked object crop. To obtain a complete 3D asset, the masked crop is transformed into a clean frontal representation, followed by textured mesh reconstruction M using existing 3D generation methods [3, 23, 46, 61]. The reconstructed mesh is responsible only for recovering object appearance and geometry, whereas its spatial configuration is determined by the proposed layout model based on the scene image and object mask. We construct an intermediate scene representation containing the reference image, target mask, object category, and reconstructed mesh information. The trained model then predicts the object transformation parameters:

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Inst. Spec.</td><td colspan="5">Geometry</td><td colspan="4">Visual Metrics</td></tr><tr><td>CD-S ↓ CD-O ↓ F-Score-S ↑ F-Score-O ↑ IoU-B ↑</td><td></td><td></td><td></td><td></td><td>PSNR ↑ SSIM ↑ LPIPS ↓</td><td></td><td></td><td>CLIP-S ↑</td></tr><tr><td>PartCrafter [23]</td><td>X</td><td>0.2027</td><td></td><td>35.86</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gen3DSR [10]</td><td>√</td><td>0.1260</td><td>0.1511</td><td>43.81</td><td>38.95</td><td>0.3162</td><td>10.15</td><td>0.7874</td><td>0.4225</td><td>0.5972</td></tr><tr><td>MIDI* [19]</td><td>√</td><td>0.0832</td><td>0.1073</td><td>52.38</td><td>60.43</td><td>0.2357</td><td>10.97</td><td>0.8139</td><td>0.3867</td><td>0.5894</td></tr><tr><td>SceneGen [29]</td><td>√</td><td>0.0678</td><td>0.0616</td><td>58.33</td><td>71.89</td><td>0.5785</td><td>10.36</td><td>0.7708</td><td>0.3719</td><td>0.6050</td></tr><tr><td>SAM3D [5]</td><td>√</td><td>0.0776</td><td>0.0593</td><td>71.54</td><td>60.58</td><td>0.5813</td><td>11.59</td><td>0.8183</td><td>0.3391</td><td>0.6096</td></tr><tr><td>3D-Fixer [56]</td><td>√</td><td>0.0692</td><td>0.0844</td><td>55.67</td><td>62.03</td><td>0.4476</td><td>10.48</td><td>0.7877</td><td>0.3846</td><td>0.6270</td></tr><tr><td>F3V (ours)</td><td>√</td><td>0.0258</td><td>0.0579</td><td>75.16</td><td>69.56</td><td>0.5904</td><td>11.63</td><td>0.8262</td><td>0.3320</td><td>0.6121</td></tr></table>

Table 1 Quantitative evaluation of image-based 3D scene reconstruction on the 3D-FUTURE test set. The comparison covers geometric accuracy, spatial layout consistency, and visual reconstruction quality. <sup>∗</sup> indicates adopting MV-Adapter [20] for texture rendering.

$$
( \hat { \bf t } , \hat { \bf r } , \hat { \bf s } ) = F _ { \boldsymbol \theta } ( R , M ) ,\tag{13}
$$

Finally, the generated object instance is obtained by applying the predicted similarity transformation to each vertex v of the reconstructed mesh M:

$$
\hat { M } = \{ \hat { \mathbf { s } } \hat { \mathbf { r } } \mathbf { v } + \hat { \mathbf { t } } | \mathbf { v } \in M \} .\tag{14}
$$

## 4 Experiments

## 4.1 Experimental Setting

Evaluation Metrics. We assess the reconstructed scenes from both geometric accuracy and visual fidelity. For geometric evaluation, point clouds are sampled from the generated asset surfaces, and the predicted scenes are registered to the ground truth using FilterReg. Compared with ICP, FilterReg provides stronger robustness when handling incomplete observations and inaccurate initial poses, which are common in multi-object reconstruction scenarios. After registration, we measure scene-level and object-level reconstruction performance using Chamfer Distance and F-Score, denoted as CD-S, F-Score-S, CD-O, and F-Score-O, respectively. We additionally report the voxel IoU of object bounding boxes (IoU-B) to evaluate spatial occupancy consistency and layout accuracy. For visual assessment, the aligned predictions are rendered from the original input viewpoint in Blender and compared with reference images using PSNR, SSIM, LPIPS, and CLIP-S. These metrics evaluate pixel-level similarity, structural preservation, perceptual quality, and semantic alignment. We also record the inference latency required to generate a single 3D asset on one H200 GPU to analyze computational eficiency.

Baselines. We compare our model with representative approaches for single-image and scene-level 3D generation, including PartCrafter [23], Gen3DSR [10], MIDI [19], SceneGen [29], SAM3D [5], and 3D-Fixer [56]. For approaches that support mask-guided generation, the corresponding target-object masks are provided as input. For methods without explicit instance-level control, such as PartCrafter, we follow their original inference protocols by supplying cropped object images or object number information when required. Since several baselines do not provide complete texture generation or rendering implementations, visual comparisons are performed only on methods with reproducible rendering results under the same evaluation conditions. The purpose of the comparison is not to show superiority of our texture synthesis component over dedicated 3D generators, but to validate whether decoupled layout reasoning improves object positioning, scale estimation, and spatial relationship modeling.

Benchmarks. We conduct geometric and visual evaluations on the 3D-FUTURE test set [14]. Each test sample contains a photorealistic scene image, target objects with segmentation annotations, and corresponding 3D ground-truth information for quantitative evaluation. The dataset covers diverse indoor furniture categories and various object arrangements, making it suitable for assessing single-image scene reconstruction and object-level layout estimation. In addition, we construct a dedicated validation benchmark for physical attribute evaluation. Specifically, physically controllable assets [3] are retrieved according to the layouts in the 3D-FUTURE test set, enabling evaluation of the accuracy of scene-level physical properties predicted by our framework.

![](images/87344f785f10fec46bb6f8b3cdc24a231d3403cf58fb775b2aa9792bbd75c963.jpg)  
Figure 4 Qualitative comparison of scene reconstruction results under diverse environments. The examples include in-domain and out-of-domain scenes with diferent object arrangements and visual appearances. Compared with existing approaches, F3V better preserves object positions, scales, and spatial relationships, demonstrating stronger geometric reasoning and layout consistency.

## 4.2 Quantitative Results

Table 1 summarizes the geometric and visual evaluation results on the 3D-FUTURE test set. F3V achieves consistently strong performance across both reconstruction accuracy and rendering quality, demonstrating the efectiveness of unified spatial modeling for image-based 3D scene reconstruction.

For geometric evaluation, our method obtains the lowest CD-S and CD-O values among all compared approaches, indicating more accurate recovery of scene structures and object-level shapes. The improvements in F-Score-S and IoU-B further verify that the predicted layouts better preserve global spatial organization and object occupancy compared with existing generation-based methods. In particular, the superior IoU-B performance reflects the advantage of decoupling layout reasoning from asset synthesis, allowing object positions and scales to be inferred through explicit geometric understanding rather than relying only on generation priors.

For the visual evaluation, F3V achieves the best PSNR, SSIM, and LPIPS results, showing that improved geometric alignment also benefits image-level appearance consistency. Although 3D-Fixer obtains a slightly higher CLIP-S score, our method maintains competitive semantic similarity while achieving stronger geometric reconstruction. These results demonstrate that geometry-aware spatial reasoning provides a better balance between semantic consistency and physical scene fidelity.

![](images/a85ac3c6b2734f8b6f414395f422daf5a69cbb3c4c619123c63ddd95a83adf9f.jpg)  
Figure 5 Additional qualitative examples. The results demonstrate the robustness of Fysiverse-3D-Vision under diverse scene configurations and object arrangements.

## 4.3 Qualitative Results

Figure 4 presents qualitative comparisons on both in-domain and out-of-domain scenes. Compared with existing approaches, the proposed method produces more reliable object arrangements under complex spatial configurations. Baseline methods are generally capable of generating objects that match the semantic categories in the input image, but they frequently exhibit inaccurate object positions, unrealistic scales, missing instances, and incorrect support relationships. Such errors become particularly apparent in scenes containing multiple interacting objects, where local appearance similarity is insuficient to recover the underlying spatial structure. In contrast, our approach better preserves object co-occurrence patterns and scene-level organization. As illustrated in bedroom, living-room, cartoon-style, and gray-scale indoor scenarios, the predicted layouts maintain more accurate relative distances and spatial relationships among furniture and surrounding objects. This advantage results from the interaction between object-conditioned representations and global geometric features, enabling the model to reason about object placement beyond visual appearance alone.

Additional examples in Figure 5 further demonstrate the generalization ability of our method across diverse scene structures. Compared with existing approaches, our method maintains more stable object placement and fewer geometric inconsistencies, especially in scenes with complex object interactions. These observations indicate that explicit spatial reasoning is critical for constructing usable 3D environments, where semantic recognition alone cannot guarantee physically plausible scene reconstruction.

<table><tr><td colspan="3">Components</td><td colspan="5">Metrics</td></tr><tr><td>Step 1</td><td>Step 2</td><td>Step 3</td><td>CD-S ↓</td><td>CD-O ↓</td><td>F-Score-S ↑</td><td>F-Score-O ↑</td><td>IoU-B ↑</td></tr><tr><td>√</td><td></td><td></td><td>0.1321</td><td>0.1117</td><td>42.40</td><td>52.07</td><td>0.3551</td></tr><tr><td></td><td>√</td><td></td><td>0.2495</td><td>0.1824</td><td>36.52</td><td>25.18</td><td>0.3186</td></tr><tr><td></td><td></td><td>√</td><td>0.0865</td><td>0.0773</td><td>58.75</td><td>59.30</td><td>0.3855</td></tr><tr><td>√</td><td>√</td><td></td><td>0.0796</td><td>0.0814</td><td>64.67</td><td>57.02</td><td>0.4530</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.0258</td><td>0.0579</td><td>75.16</td><td>69.56</td><td>0.5904</td></tr></table>

Table 2 Ablation analysis of the proposed multi-stage training strategy. Each stage is progressively introduced to examine its contribution to geometric reconstruction and spatial layout prediction.
<table><tr><td rowspan="2">Methods</td><td colspan="5">Physical Attributes</td></tr><tr><td>Scale ↓</td><td>Mat. ↑</td><td>Aff. ↑</td><td>Kin. ↑</td><td>Desc. ↑</td></tr><tr><td>MIDI [19] MIDI + PhysPre</td><td>12.54</td><td>7.17</td><td>10.03</td><td>0.27</td><td>7.05</td></tr><tr><td>F3V (ours)</td><td>8.09</td><td>14.51</td><td>11.90</td><td>0.32</td><td>10.94</td></tr></table>

Table 3 Evaluation of physical attribute understanding on executable 3D assets. Scale denotes absolute-scale error, while the remaining metrics measure the prediction accuracy of material, afordance, kinematic properties, and textual descriptions.

## 4.4 Ablation Studies

Table 2 investigates the contribution of each component in the proposed three-stage optimization strategy. The results demonstrate that progressively introducing geometry learning, layout reasoning, and physical refinement is essential for achieving accurate and executable scene reconstruction. The first stage establishes the shared geometry-language representation, which provides semantic and geometric priors for subsequent layout prediction. When only the first stage is applied, the model already obtains meaningful reconstruction ability, confirming that joint semantic and geometric learning provides a strong foundation for spatial understanding. In contrast, directly optimizing the layout module without the learned shared representation leads to significantly degraded performance, indicating that layout prediction cannot be efectively solved without suficient geometric and semantic context.

The second stage injects layout supervision while preserving the geometry reconstruction objective. Compared with independent layout optimization, this strategy enables the layout module to exploit the geometry-aware representation learned in the previous stage. The improvements in CD-S, F-Score-S, and IoU-B demonstrate that maintaining geometric constraints prevents the model from overfitting to object placement and improves global scene consistency.

The final stage introduces collision-aware refinement using real-world scene constraints. The consistent gains across all evaluation metrics show that explicit physical regularization further improves object arrangement and spatial plausibility. These results verify that the three stages are complementary: geometry-language pretraining provides general spatial knowledge, layout injection transfers this knowledge to object placement, and collision-aware refinement enhances the physical validity of the reconstructed scenes.

## 4.5 Physical-Aware Representation and Executable Reconstruction

Following the evaluation setting of PhysX-3D, Table 3 provides a comparison of diferent approaches on physical attribute prediction and executable asset understanding. Compared with MIDI and its variants enhanced with physical priors, our method achieves consistent improvements across multiple aspects, including absolute scale estimation, material recognition, afordance prediction, kinematic modeling, and textual description generation. These results demonstrate that the shared vision-language-geometry representation captures not only spatial configurations but also object-level functional knowledge and physical characteristics, which are essential for constructing interactive and executable 3D environments.

![](images/5e6d74815ccf1764ff82f6fb7ff9a9ec118f402e61edb9ee0d7cc3795abbb4d9.jpg)  
Figure 6 Extension to part-level executable scene reconstruction. By integrating part-aware asset generation with the predicted spatial layout, F3V enables fine-grained reconstruction beyond object-level placement.

Beyond object-level physical understanding, F3V can be further extended to fine-grained interactive scene reconstruction with part-level annotations. As illustrated in Figure 6, combining our spatial layout prediction with part-aware asset generation methods such as PartCrafter enables scene reconstruction with detailed structural decomposition. This extension improves the granularity of executable scene representation from complete objects to individual parts. In contrast, existing part-level generation approaches mainly focus on isolated object synthesis and may lose fine-grained structures when applied to scene-level reconstruction. The ability to preserve both scene context and part-level details highlights the advantage of our unified spatial representation.

## 5 Conclusion

We present Fysiverse-3D-Vision, a unified vision-language-geometry framework for executable 3D scene reconstruction from a single image. Diferent from existing approaches that tightly couple spatial layout estimation with object generation, our framework decouples layout reasoning from asset synthesis and learns object placement from shared semantic and geometric representations. By integrating textual understanding, visual semantics, and geometric structures within a unified architecture, our model is able to jointly capture scene context, metric geometry, and object-level spatial relationships. The proposed multi-stage training strategy progressively builds geometry-language representations, injects layout reasoning while preserving reconstruction capability, and improves physical consistency through collisionaware refinement. Extensive experiments demonstrate that the method achieves superior performance in geometric reconstruction, spatial layout estimation, visual quality, and physical attribute understanding. Additional evaluations on executable assets and part-level reconstruction further verify the flexibility of the learned spatial representation for interactive scene construction.

## References

[1] Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, et al. Ming-omni: A unified multimodal model for perception and generation. arXiv preprint arXiv:2506.09344, 2025.

[2] Ziang Cao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, and Ziwei Liu. Physx-anything: Simulation-ready physical 3d assets from single image. arXiv preprint arXiv:2511.13648, 2025.

[3] Ziang Cao, Zhaoxi Chen, Liang Pan, and Ziwei Liu. Physx-3d: Physical-grounded 3d asset generation. Advances in Neural Information Processing Systems, 38:93771–93784, 2026.

[4] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.

[5] Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, et al. Sam 3d: 3dfy anything in images. arXiv preprint arXiv:2511.16624, 2025.

[6] Zizhi Chen, Yizhen Gao, Minghao Han, Yizhou Liu, Zhaoyu Chen, Dingkang Yang, and Lihua Zhang. Forging a dynamic memory: Retrieval-guided continual learning for generalist medical foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 32309–32321, 2026.

[7] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 21126–21136, 2022.

[8] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023.

[9] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023.

[10] Andreea Dogaru, Mert Ozer, and Bernhard Egger. Gen3dsr: Generalizable 3d scene reconstruction via divide and conquer from<sup>¨</sup> a single view. In International Conference on 3D Vision 2025, 2025.

[11] Martin Ester, Hans-Peter Kriegel, Jorg Sander, Xiaowei Xu, et al. A density-based algorithm for discovering clusters in large¨ spatial databases with noise. In kdd, volume 96, pages 226–231, 1996.

[12] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems, 36:18225–18250, 2023.

[13] Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 10933–10942, 2021.

[14] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 3d-future: 3d furniture shape with texture. International Journal ofComputer Vision, 129(12):3313–3337, 2021.

[15] Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 1440–1448, 2015.

[16] Minghao Han, Dingkang Yang, Yue Jiang, Yizhou Liu, and Lihua Zhang. Omnifysics: Towards physical intelligence evolution via omni-modal signal processing and network optimization. arXiv preprint arXiv:2602.07064, 2026.

[17] JiaKui Hu, Shanshan Zhao, Qing-Guo Chen, Xuerui Qiu, Jialun Liu, Zhao Xu, Weihua Luo, Kaifu Zhang, and Yanye Lu. Omni-view: Unlocking how generation facilitates understanding in unified 3d model based on multiview images. arXiv preprint arXiv:2511.07222, 2025.

[18] Wenbo Hu, Jingli Lin, Yilin Long, Yunlong Ran, Lihan Jiang, Yifan Wang, Chenming Zhu, Runsen Xu, Tai Wang, and Jiangmiao Pang. G<sup>2</sup> VLM: Geometry grounded vision language model with unified 3d reconstruction and spatial reasoning. arXiv preprint arXiv:2511.21688, 2025.

[19] Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, and Lu Sheng. Midi: Multi-instance difusion for single image to 3d scene generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23646–23657, 2025.

[20] Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mv-adapter: Multi-view consistent image generation made easy. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 16377–16387, 2025.

[21] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. Advances in Neural Information Processing Systems, 38:39856–39890, 2026.

[22] Zhe Li, Xiang Bai, Jieyu Zhang, Zhuangzhe Wu, Che Xu, Ying Li, Chengkai Hou, and Shanghang Zhang. Urdf-anything: Constructing articulated objects with 3d multimodal language model. Advances in Neural Information Processing Systems, 38: 94974–95002, 2026.

[23] Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Feng Yiqiang, Yadong Mu, and Katerina Fragkiadaki. Partcrafter: Structured 3d mesh generation via compositional latent difusion transformers. Advances in neural information processing systems, 38:35387–35415, 2026.

[24] Lu Ling, Yunhao Ge, Yichen Sheng, and Aniket Bera. I-scene: 3d instance models are implicit generalizable spatial learners. arXiv preprint arXiv:2512.13683, 2025.

[25] Jiayi Liu, Denys Iliash, Angel X Chang, Manolis Savva, and Ali Mahdavi-Amiri. SINGAPO: Single image controlled generation of articulated parts in object. arXiv preprint arXiv:2410.16499, 2024.

[26] Keliang Liu, Zizhi Chen, Mingcheng Li, Jingqun Tang, Dingkang Yang, and Lihua Zhang. Resolving evidence sparsity: Agentic context engineering for long-document understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19452–19462, 2026.

[27] Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao Su. Openshape: Scaling up 3d shape representation towards open-world understanding. Advances in neural information processing systems, 36:44860–44879, 2023.

[28] Ruijie Lu, Yu Liu, Jiaxiang Tang, Junfeng Ni, Yuxiang Wang, Diwen Wan, Gang Zeng, Yixin Chen, and Siyuan Huang. Dreamart: Generating interactable articulated objects from a single image. arXiv preprint arXiv:2507.05763, 2025.

[29] Yanxu Meng, Haoning Wu, Ya Zhang, and Weidi Xie. Scenegen: Single-image 3d scene generation in one feedforward pass. arXiv preprint arXiv:2508.15769, 2025.

[30] Pramod Rao, Abhimitra Meka, Xilong Zhou, Gereon Fox, Mallikarjun BR, Fangneng Zhan, Tim Weyrich, Bernd Bickel, Hanspeter Pfister, Wojciech Matusik, et al. 3dpr: Single image 3d portrait relighting with generative priors. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–12, 2025.

[31] Tianhe Ren, Shuo Shen, et al. Grounded sam 2: Ground and track anything in videos with grounding dino florence-2 and sam 2. GitHub repository, 2025.

[32] Yukai Shi, Weiyu Li, Zihao Wang, Hongyang Li, Xingyu Chen, Ping Tan, and Lei Zhang. Scenemaker: Open-set 3d scene generation with decoupled de-occlusion and pose estimation model. arXiv preprint arXiv:2512.10957, 2025.

[33] Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. Layoutvlm: Diferentiable optimization of 3d layout via vision-language models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 29469–29478, 2025.

[34] Jiaxiang Tang, Ruijie Lu, Max Li, Zekun Hao, Xuan Li, Fangyin Wei, Shuran Song, Gang Zeng, Ming-Yu Liu, and Tsung-Yi Lin. Eficient part-level 3d object generation via dual volume packing. Advances in Neural Information Processing Systems, 38: 27115–27137, 2026.

[35] Robert Tarjan. Depth-first search and linear graph algorithms. SIAM journal on computing, 1(2):146–160, 1972.

[36] Meituan LongCat Team, Bairui Wang, Bin Xiao, Bo Zhang, Bolin Rong, Borun Chen, Chang Wan, Chao Zhang, Chen Huang, Chen Chen, et al. Longcat-flash-omni technical report. arXiv preprint arXiv:2511.00279, 2025.

[37] Qwen Team. Qwen3. 5-omni technical report. arXiv preprint arXiv:2604.15804, 2026.

[38] Rui Tian, Mingfei Gao, Mingze Xu, Jiaming Hu, Jiasen Lu, Zuxuan Wu, Yinfei Yang, and Afshin Dehghan. Unigen: Enhanced training & test-time strategies for unified multimodal understanding and generation. Advances in Neural Information Processing Systems, 38:152386–152415, 2026.

[39] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.

[40] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

[41] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5261–5271, 2025.

[42] Tao Wang, Guangpin Tao, Wanglong Lu, Kaihao Zhang, Wenhan Luo, Xiaoqin Zhang, and Tong Lu. Restoring vision in hazy weather with hierarchical contrastive learning. Pattern Recognition, 145:109956, 2024.

[43] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. �<sup>3</sup>: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025.

[44] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. Advances in Neural Information Processing Systems, 38:13569–13597, 2026.

[45] Hongchi Xia, Xuan Li, Zhaoshuo Li, Qianli Ma, Jiashu Xu, Ming-Yu Liu, Yin Cui, Tsung-Yi Lin, Wei-Chiu Ma, Shenlong Wang, et al. Sage: Scalable agentic 3d scene generation for embodied ai. arXiv preprint arXiv:2602.10116, 2026.

[46] Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. arXiv preprint arXiv:2512.14692, 2025.

[47] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21469–21480, 2025.

[48] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. Advances in Neural Information Processing Systems, 38:47490–47518, 2026.

[49] Yueming Xu, Jiahui Zhang, Ze Huang, Yurui Chen, Yanpeng Zhou, Zhenyu Chen, Yu-Jie Yuan, Pengxiang Xia, Guowei Huang, Xinyue Cai, et al. Uniugg: Unified 3d understanding and generation via geometric-semantic encoding. arXiv preprint arXiv:2508.11952, 2025.

[50] Wei Xue, Mingcheng Li, Xuecheng Wu, Jingqun Tang, Dingkang Yang, and Lihua Zhang. Profocus: Proactive perception and focused reasoning in vision-and-language navigation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18129–18139, June 2026.

[51] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, et al. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16227–16237, 2024.

[52] Yunhan Yang, Yufan Zhou, Yuan-Chen Guo, Zi-Xin Zou, Yukun Huang, Ying-Tian Liu, Hao Xu, Ding Liang, Yan-Pei Cao, and Xihui Liu. Omnipart: Part-aware 3d generation with semantic decoupling and structural cohesion. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–12, 2025.

[53] Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Lan Xu, Wei Yang, Jiayuan Gu, and Jingyi Yu. Cast: Component-aligned 3d scene reconstruction from an rgb image. ACM Transactions on Graphics (TOG), 44(4):1–19, 2025.

[54] Chongjie Ye, Cheng Cao, Chuanyu Pan, Yiming Hao, Yihao Zhi, Yuanming Hu, and Xiaoguang Han. Omni123: Exploring 3d native foundation models with limited 3d data by unifying text to 2d and 3d generation. arXiv preprint arXiv:2604.02289, 2026.

[55] Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, and Jun Zhu. Shapellm-omni: A native multimodal llm for 3d generation and understanding. arXiv preprint arXiv:2506.01853, 2025.

[56] Ze-Xin Yin, Liu Liu, Xinjie Wang, Wei Sui, Zhizhong Su, Jian Yang, and Jin Xie. 3d-fixer: Coarse-to-fine in-place completion for 3d scenes from a single image. arXiv preprint arXiv:2604.04406, 2026.

[57] Sung-Hoon Yoon, Minghan Li, Gaspard Beaudouin, Congcong Wen, Muhammad Rafay Azhar, and Mengyu Wang. Splitflow: Flow decomposition for inversion-free text-to-image editing. Advances in Neural Information Processing Systems, 38: 153207–153225, 2026.

[58] Xiaoding Yuan, Guofeng Zhang, Prakhar Kaushik, Artur Jesslen, Adam Kortylewski, and Alan Yuille. Scaling 3d compositional models for robust classification and pose estimation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6406–6415. IEEE, 2025.

[59] Qingcheng Zhao, Xiang Zhang, Haiyang Xu, Zeyuan Chen, Jianwen Xie, Yuan Gao, and Zhuowen Tu. Depr: Depth guided single-view scene reconstruction with instance-level difusion. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 5722–5733, 2025.

[60] Shanshan Zhao, Xinjie Zhang, Jintao Guo, Jiakui Hu, Lunhao Duan, Minghao Fu, Yong Xien Chng, Guo-Hua Wang, Qing-Guo Chen, Zhao Xu, et al. Unified multimodal understanding and generation models: Advances, challenges, and opportunities. arXiv preprint arXiv:2505.02567, 2025.

[61] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling difusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025.

[62] Kaichen Zhou, Zeyang Bai, Xinhai Chang, Mengyu Wang, Paul Liang, and Fangneng Zhan. Stream3d: Sequential multi-view 3d generation via evidential memory. arXiv preprint arXiv:2605.21472, 2026.

[63] Wenxu Zhou, Kaixuan Nie, Hang Du, Dong Yin, Wei Huang, Siqiang Guo, Xiaobo Zhang, and Pengbo Hu. Il3d: A large-scale indoor layout dataset for llm-driven 3d scene generation. arXiv preprint arXiv:2510.12095, 2025.

[64] Zongwei Zhou, Wenhan Luo, Qiang Wang, Junliang Xing, and Weiming Hu. Distractor-aware discrimination learning for online multiple object tracking. Pattern Recognition, 107:107512, 2020.