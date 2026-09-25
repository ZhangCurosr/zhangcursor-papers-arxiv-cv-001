# OmniFabric: Coherent UV Space Texture Synthesis for 3D Garment Reconstruction

DING-JIUN HUANG, Carnegie Mellon University, United States of America

YUANHAO WANG, University of Washington, United States of America

CHENG ZHANG, Texas A&M University, United States of America

HUGO BERTICHE, Google, United States of America

ALEXANDRU-EUGEN ICHIM, Google, Switzerland

THABO BEELER, Google, Switzerland

FERNANDO DE LA TORRE, Carnegie Mellon University, United States of America

![](images/61e7abca49baea1fb1d25cd8a3c5ad954dff9b259c558f76ce6a6832496c399e.jpg)  
Fig. 1. Image-based 3D garment texture synthesis. Given a single in-the-wild clothing image, OmniFabric synthesizes high-quality, coherent fabric textures directly on garment sewing paterns. OmniFabric supports a wide range of textures and paterns, preserving fine details while maintaining global alignment with the input image. The resulting textured sewing paterns can be seamlessly simulated into 3D garments.

Automated generation of production-ready 3D garment assets from a single image is a central challenge in digital content creation. While recent generative models have significantly advanced 3D geometry reconstruction, synthesizing high-quality textures remains a bottleneck. Existing methods often bake environmental illumination and shadows directly into the texture map, or they fail to maintain global structural coherence, making the result ing assets unusable for physical simulation and relighting. In this work, we introduce OmniFabric, a novel approach that synthesizes globally coherent texture maps directly within the 2D sewing pattern space. Given a single

Authors’ Contact Information: Ding-Jiun Huang, Carnegie Mellon University, United States of America, djhuang322@gmail.com; Yuanhao Wang, University of Washing ton, United States of America, yuanhao4@cs.washington.edu; Cheng Zhang, Texas A&M University, United States of America, chzhang@tamu.edu; Hugo Bertiche, Google, United States of America, hbertiche@google.com; Alexandru-Eugen Ichim, Google, Switzerland, alexichim@google.com; Thabo Beeler, Google, Switzerland, tbeeler@google.com; Fernando De la Torre, Carnegie Mellon University, United States of America, ftorre@andrew.cmu.edu.

reference image, our pipeline utilizes an estimated 3D mesh and generative priors of powerful Vision-Language Models (VLM) to establish a complete but coarse texture initialization across the unwrapped sewing patterns. We then leverage a specialized difusion transformer, trained via an automated synthetic data engine and conditioned on 3D positional features, to refine this initialization directly in the canonical UV domain. This efectively removes distortion and baked-in artifacts to extract a clean and normalized texture map that preserves the original garment design. Extensive experiments demonstrate that OmniFabric significantly outperforms state-of-theart baselines, yielding photorealistic 3D garments with high-quality textures. Project page: https://humansensinglab.github.io/OmniFabric

CCS Concepts: • Computing methodologies → Appearance and texture representations.

Additional Key Words and Phrases: Texture Synthesis, 3D Garment Reconstruction, Garment Sewing Patterns

## ACM Reference Format:

Ding-Jiun Huang, Yuanhao Wang, Cheng Zhang, Hugo Bertiche, Alexandru-Eugen Ichim, Thabo Beeler, and Fernando De la Torre. 2026. OmniFabric: Coherent UV Space Texture Synthesis for 3D Garment Reconstruction. In SIGGRAPH Asia 2026 Conference Papers (SA Conference Papers ’26), December 01–04, 2026, Kuala Lumpur, Malaysia. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3829340.3842275

## 1 Introduction

Creating high-fidelity, simulation-ready 3D garments is essential for modern digital production, with applications spanning gaming [Habermann et al. 2021], e-commerce [Hwangbo et al. 2020; Kim and LaBat 2013], and cinematic visual efects [Hughes et al. 2007]. While traditional asset creation relies on labor-intensive manual modeling, there is growing interest in automating this process to synthesize assets directly from images. Despite recent strides in reconstructing accurate 3D garment geometry [Bian et al. 2025; He et al. 2024; Li et al. 2025b; Nakayama et al. 2025; Rong et al. 2025; Sarafianos et al. 2025; Wang et al. 2025], generating productionready textures from a single observation remains an open problem.

Garment texture synthesis is uniquely dificult due to the complex dynamics of fabric, which frequently produce deep folds, selfocclusions, and extreme pose variations (see Figure 1). To be truly usable in downstream applications, a texture must not only look realistic in a static view but also be free from baked-in lighting and geometric distortion, capturing the garment’s intrinsic albedo. However, current state-of-the-art 3D texturing frameworks [Bensadoun et al. 2024; Chen et al. 2023b; Richardson et al. 2023; Zeng et al. 2024; Zhao et al. 2025] typically rely on multi-view difusion priors to iteratively “paint” assets via direct projection onto draped 3D geometry. Consequently, unwrapping these folded surfaces inevitably causes large geometric distortion and occlusion. Furthermore, these models fail to disentangle intrinsic material properties from environmental illumination, baking transient lighting artifacts, such as shadows and geometry-induced wrinkles, directly into the texture map.

To circumvent these issues, recent work such as FabricDifusion [Zhang et al. 2024] has explored texture synthesis directly within the regularized, flat 2D sewing pattern domain, successfully yielding simulation-ready texture maps. However, FabricDifusion focuses exclusively on local texture synthesis by extracting tileable material patches from the source image, ignoring the global texture layout. Consequently, it struggles at reconstructing asymmetric designs, large-scale prints, or irregular spatial patterns. We argue that synthesizing a globally coherent texture map that faithfully reflects the visual identity of the input image requires leveraging the complete context of the reference observation, rather than relying on isolated patches. This presents a fundamental challenge:

## How do we establish pixel-level correspondences between a reference image and 2D sewing panels while transferring textures that are free from baked-in artifacts and geometric distortions?

In this paper, we propose OmniFabric to bridge this gap. We first employ a powerful image generation model [Google 2025] to repose the input clothing into a canonical A-pose, aligning its silhouette with the frontal render of the garment mesh simulated from the predicted sewing patterns. We further harness the temporal and spatial consistency of large-scale video generation models [Google DeepMind 2025] to hallucinate coherent, multi-view observations of this A-pose garment. Projecting these globally consistent multi-view images onto the unwrapped sewing patterns establishes a complete, albeit coarse, texture initialization that preserves the garment’s global design. Because this initial projection inevitably bakes visual artifacts such as geometric distortions and physics-induced wrinkles into the texture map, we introduce a subsequent refinement stage operating entirely within the sewing pattern space. We design an automatic data engine that curates an extensive and highly diverse synthetic dataset of textured sewing patterns, paired with simulated texture initializations containing baked-in artifacts and distortions. With the curated dataset, we train a specialized Difusion Transformer (DiT) that acts as a texture normalizer in the 2D sewing pattern space. This model rectifies distortions, and strips away transient illumination to produce a clean and normalized texture map. Finally, we conduct comprehensive experiments to demonstrate the superiority of OmniFabric over existing 3D texturing baselines. In summary, the key contributions of our work are:

• OmniFabric, a novel framework for synthesizing high-fidelity, globally coherent textures from a single in-the-wild image for 3D garment reconstruction.

• An automated data engine for creating realistic, diverse textured sewing patterns at scale, providing training data for learning complex garment textures.

• A strategy of using a canonical 3D mesh as spatial anchor to map image pixels to sewing pattern UV space, enabling globally aligned texture synthesis.

• A coarse-to-fine texture generation pipeline that initializes globally aligned textures, and further rectifies distortion and bakedin artifacts with a DiT-based UV space normalization network.

Remark. Generating 3D garment textures is not a new topic. While many prior works mainly focus on directly texturing 3D garment meshes, they often do not scale well to large collections of in-the-wild images and remain weakly connected to real-world garment production pipelines. We argue that fabric texture synthesis should be grounded in the UV space of sewing patterns. This enables the model to maintain strong global coherence while remaining closely aligned with how garments are constructed in practice.

## 2 Related Work

## 2.1 3D Garment Reconstruction

Early approaches for garment reconstruction primarily relied on implicit functions [Saito et al. 2019, 2020; Xiu et al. 2022] to approximate surfaces, yet they often struggle with complex topologies and loose-fitting clothing. More recent works leverage advanced representations like 3DGS [Rong et al. 2025] and image-to-3D difusion priors [Luo et al. 2024; Wang et al. 2025] to achieve impressive visual fidelity. However, these methods typically produce rigid meshes or unstructured point clouds that are inherently unsuitable for physica cloth simulation. While methods like Garment3DGen [Sarafianos et al. 2025] bridge this gap by utilizing template deformation to produce simulation-ready assets, they rely on a slow, computationally intensive per-asset optimization process. To obtain simulationready assets, a significant line of work attempts to infer 2D sewing patterns directly from 3D point clouds or images. NeuralTailor [Korosteleva and Lee 2022] focused on reconstructing pattern structures from 3D geometry, and subsequent works [Chen et al. 2024; Liu et al. 2023] predict these patterns directly from single-view images using discriminative transformer architectures. To facilitate more structured generation, GarmentCode [Korosteleva and Sorkine-Hornung 2023] introduced a programmatic domain-specific language (DSL) for sewing patterns, enabling parametric control. Building on this representation, subsequent research has increasingly focused on advanced generative modeling. DressCode [He et al. 2024] synthesizes novel patterns from text prompts using a GPT-based framework, while AIpparel [Nakayama et al. 2025] scales into a multimodal foundation model that handles complex pattern generation natively from mixed inputs. To further improve physical realism, Dress-1-to-3 [Li et al. 2025b] incorporates diferentiable physics simulators to optimize geometric alignment of the inferred patterns. Concurrently, other state-of-the-art frameworks directly predict these programmatic structures from images leveraging large generative models [Bian et al. 2025; Li et al. 2025a; Zhou et al. 2025]. While these methods have established sewing patterns as a robust domain for 3D animation, the synthesis of normalized, artifact-free textures for these panels remains an open challenge.

## 2.2 Generative 3D Texturing

The rise of large-scale vision-language models [Radford et al. 2021; Rombach et al. 2022; Saharia et al. 2022] has revolutionized 3D texture synthesis. Early optimization-based approaches either leverage CLIP [Radford et al. 2021] for optimizing texture maps of 3D models [Chen et al. 2022; Hong et al. 2022; Michel et al. 2022; Mohammad Khalid et al. 2022], or employ Score Distillation Sampling (SDS) to optimize texture maps by distilling gradients from a frozen 2D difusion model [Chen et al. 2023a; Metzer et al. 2023; Poole et al. 2022]. While capable of generating coherent global structures, these methods tend to hallucinate generic textures without highfrequency details. To improve fidelity, recent research has shifted toward projection-based inpainting, by utilizing depth-conditioned Stable Difusion to progressively paint 3D meshes from multiple viewpoints [Cao et al. 2023; Chen et al. 2023b; Richardson et al. 2023]. These frameworks iteratively project the current texture into screen space, inpaint missing regions using 2D priors, and project the result back. More recent systems like Paint3D [Zeng et al. 2024], MVPaint [Cheng et al. 2025] and Meta 3D TextureGen [Bensadoun et al. 2024] further refine this process by synchronizing multi-view generation or employing coarse-to-fine UV refinement to minimize seams. Despite their popularity, a fundamental limitation persists across these general-purpose methods: they rely on priors trained on natural photography. Consequently, they inherently entangle illumination with surface appearance, inevitably “baking in” transient lighting efects—such as cast shadows and specular highlights—directly into the UV map. In addition, the generated textures are subject to distortion and blurring due to the complex surface topology, leading to suboptimal results.

In the specific area of garment texturing, FabricDifusion [Zhang et al. 2024] addresses some of these pitfalls by treating texture synthesis as a material extraction task. It utilizes a difusion model to extract tileable, distortion-free material patches from a single image without lighting artifacts. However, because it relies on local texture tiling, FabricDifusion fails to capture global structural information, such as specific graphic placements and non-uniform texture patterns across diferent UV islands. OmniFabric bridges this gap by shifting from local material extraction to global texture synthesis. We operate in the canonical UV space to disentangle base-color from illumination, synthesizing a holistic texture map that respects the global texture design in the reference image. While previous methods [Bensadoun et al. 2024; Zeng et al. 2024] also deploy a 2D refinement stage on the UV map, they only perform inpainting on occluded regions and fail to rectify distortion or remove baked in artifacts, which are the domain specific challenges for garment texturing, as shown in Figure 2.

![](images/2e6c6bdeb5e619fcf4f312b5b334b9f6c9c6b87e4c8c31d3c80224671c95dda4.jpg)  
Fig. 2. Limits of baseline texturing methods. While existing methods can synthesize textures for 3D assets with decent visual quality, they fail to create garment textures free from distortions and baked-in illumination artifacts, rendering the asset unsuitable for downstream applications.

## 3 Method

Figure 3 presents an overview of our approach. We aim to reconstruct 3D garments with coherent textures from real-world images. We define the task and objective in Section 3.1. Next, we introduce a two-stage pipeline for generating holistic textures directly onto sewing patterns. This includes a coarse texture map initialization (Section 3.2) and a UV space texture normalization (Section 3.3). To train the model, we develop an automated pipeline to synthesize a large scale dataset of textured sewing patterns (Section 4).

## 3.1 Problem Definition

Given a single reference image � and an estimated 3D garment mesh M<sub>R</sub> parameterized by sewing patterns P from an of-theshelf sewing pattern prediction model, our goal is to generate a normalized texture map T directly within the sewing pattern space. Unlike previous methods [Zhang et al. 2024] that focus on extracting local and repeatable material patches, we formulate our objective as a global texture synthesis task. We define $\mathcal { T } \in \mathbb { R } ^ { H \times W \times 3 }$ as a representation of the normalized texture map of the garment, which is completely free of distortion, view-dependent illumination, cast shadows, and geometry-induced wrinkles. Following the paradigm of difusion models, we formulate the synthesis of this holistic texture map as a conditional distribution mapping problem. We seek to learn a mapping function $\mathcal { G }$ such that:

![](images/fad886c44a49b05ddd83e45395b19cf25596ee208ffa6a6762b1584fe5c20143.jpg)  
Fig. 3. Overview of OmniFabric. Given a reference clothing image, OmniFabric synthesizes high-quality garment textures in sewing patern UV space via a two-stage pipeline. Stage 1: We generate a coarse UV texture map by first predicting garment sewing paterns and reconstructing a canonical 3D garment. The input image texture is transferred onto the garment surface and expanded into multi-view using an of-the-shelf video generative model. These views are projected onto the canonical garment and reflected in UV space to produce a globally aligned coarse texture map. Stage 2: We refine the coarse texture with a DiT-based texture normalization network operating on UV patches. By conditioning on positional maps and spatially aligned tokens, the model removes baked-in artifacts and projection inconsistencies, producing clean and coherent sewing patern textures that can be directly simulated into 3D garments.

$$
\mathcal { T } \sim \mathcal { G } ( \bar { \jmath } , \mathcal { P } , \epsilon ) , \epsilon \sim N ( 0 , \mathbf { I } )\tag{1}
$$

We note that although $\mathcal { T }$ is not intrinsic albedo, it’s a close approximation and can be directly imported into cloth simulation engines, e.g., CLO [CLO Virtual Fashion Inc. 2024], with externally specified material assumption for the remaining properties including roughness, metallic and normal map, enabling the synthesis of simulation-ready 3D garments.

## 3.2 Coarse Texture Map Initialization

To guide texture synthesis in the sewing pattern space, we first aim to obtain a coarse texture initialization that fully leverages the visual information from the reference image. This requires establishing pixel-wise correspondences between the reference image and the sewing pattern. While this mapping is challenging, we recast the task via a VLM-driven and pose aware texture alignment step. We further leverage the multi-view generative priors of large video models to provide a complete and globally coherent texture initialization.

3.2.1 Pose aware texture alignment. Using Nano Banana Pro [Google 2025], we repose the reference image � into $I ^ { \prime }$ to perfectly align with the frontal silhouette of the rest-pose garment mesh $M _ { R } .$ . This alignment enables direct pixel projection of $I ^ { \prime }$ onto $\displaystyle { \mathcal { M } } _ { R }$ and its corresponding sewing patterns $\mathcal { P } _ { : }$ , producing the textured frontal render $R _ { f }$ . We observe that Nano Banana Pro successfully manipulates the input garment into a desired pose while preserving the original texture with high accuracy and fidelity.

3.2.2 Consistent multi-view synthesis. After the initial pose alignment, a significant portion ofthe UV space occluded from the frontal view remains untextured. To hallucinate these missing regions while encouraging strong spatial continuity, we leverage a large-scale video generation model for multi-view generation. By conditioning the model on the textured frontal render $R _ { f } ,$ we synthesize a 360-degree spinning video of the garment, naturally exploiting the model’s inherent temporal priors for structural consistency. Following recent projection-based texturing paradigms, we sample multi-view $V ,$ four orthogonal views (front, back, left, and right) from the generated video. These observations are then projected onto $\textstyle { \mathcal { M } } _ { R }$ and mapped to their corresponding UV coordinates on $\mathcal { P }$ to form the coarse texture map ${ \mathcal { T } } ^ { \prime }$

## 3.3 UV Space Texture Refinement

While ${ \mathcal { T } } ^ { \prime }$ preserves the coherent global texture of the garment, it contains baked-in artifacts like shadows, physical wrinkles and unpainted areas. Furthermore, spatial distortions in ${ \mathcal { T } } ^ { \prime }$ leave it far short of production-ready fabric quality, as shown in Figure 2. To address these issues, we introduce a texture normalization model designed to rectify ${ \mathcal { T } } ^ { \prime }$ into a normalized texture map $\mathcal { T }$

3.3.1 Training objective of multi-condition difusion. We formulate texture normalization as a holistic UV synthesis problem by developing a conditional distribution mapping network. Unlike FabricDifusion [Zhang et al. 2024], which relies on a single condition, i.e., a local textile patch, for the normalization task, our framework incorporates a multimodal conditioning set � to guide global texture normalization. We extend $\operatorname { E q } .$ 1 and minimize the following:

$$
\mathcal { L } = \mathbb { E } _ { \mathcal { E } ( \mathcal { T } ) , y , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) , t } \left[ \Vert \epsilon - \epsilon _ { \theta } ( z _ { t } , t , y ) \Vert ^ { 2 } \right] , y = \{ \mathcal { T } ^ { \prime } , R _ { f } , P _ { p o s } \}\tag{2}
$$

where $z _ { t }$ represents the noisy latent of the ground-truth texture at timestep �. The multi-modal conditioning set � includes the coarselytextured map ${ \mathcal { T } } ^ { \prime }$ , textured frontal rendering $R _ { f }$ and a 3D position map $P _ { p o s }$ , created by projecting the vertices’ 3D positions of $\mathcal { M } _ { \mathcal { R } }$ onto the 2D layout of $\mathcal { P } _ { \mathcal { S } }$ $P _ { p o s }$ helps the model learn spatial connectivity between separate UV panels, ensuring seamless textures across fragmented UV islands.

![](images/d2a8073e673ac176af9db3b02f9bc323759b00882276065931015ba9396492b0.jpg)  
Fig. 4. Framework of automated data creation. We first create a distorted texture image by applying a TPS warp to a sampled normalized texture. Both the normalized and distorted textures are then overlaid onto sampled sewing paterns. The textured sewing patern can then be simulated into a training data point of rest-posed mesh $\textstyle { \mathcal { M } } _ { { \mathcal { R } } } ,$ , position map $P _ { p o s }$ , coarse texture map ${ \mathcal { T } } ^ { \prime }$ and frontal view rendering $R _ { f }$

3.3.2 Model architecture and training. To process multimodal conditioning set �, we adopt a Difusion Transformer (DiT) architecture following previous works [Tan et al. 2025a,b], and leverage a unified token-processing strategy that treats all conditions as a unified sequence of tokens. We utilize a joint self-attention mechanism where the condition tokens (from $\mathcal { T } ^ { \prime } , R _ { f }$ , and $P _ { p o s } )$ and the noisy latent tokens from $z _ { t }$ interact within the same transformer blocks. Because $\mathcal { T } ^ { \prime } , P _ { p o s } { } _ { ; }$ , and the target $\mathcal { T }$ all share the same 2D panel layout, we apply a dynamic positional encoding strategy: image patches at the same spatial coordinates across these maps are assigned identical positional encodings, while only $R _ { f }$ retains its own coordinate system through distinct encodings. This minimal yet universal design allows the model to attend to the structural cues of $P _ { p o s }$ while simultaneously hallucinating missing textures in ${ \mathcal { T } } ^ { \prime } .$ . We use a pretrained weight of DiT, and fine-tune the model on our synthetic textured sewing pattern data with LoRA [Hu et al. 2022].

## 4 Synthetic Training Data Creation

To train our texture normalization model, we develop a data engine (Figure 4) to curate a large-scale dataset of textured sewing patterns. The core objective is to synthesize training pairs that exhibit the complex appearance of real-world garments while providing groundtruth normalized texture maps for supervised learning.

## 4.1 Textured Sewing Patern Creation

A key bottleneck in garment dataset construction is the lack of complex and high-resolution textures; most available texture sources provide only simple, repetitive patterns. To overcome this, we introduce a pipeline to produce diverse, high-fidelity texture images by using fashion images from real-world datasets, $\mathbf { e . g . }$ , DeepFash ion [Liu et al. 2016], as references. For each sample, we apply a Large Language Model (LLM) [Bai et al. 2023] to generate a detailed de scription encompassing texture layout, semantic elements, and color palettes. These descriptions, combined with the reference image, are fed into a vision-language model [Google 2025] to synthesize textures with diverse designs. We then overlay the generated texture image onto sewing patterns sampled from GarmentCodeData. Note that the goal here is not to reconstruct the exact texture from the example image, but to generate plausible and realistic textures that provide suficient diversity for training the texture synthesis model.

## 4.2 Training Pairs Construction

For each textured sewing pattern, we generate a training tuple $( \mathcal { T ^ { \prime } } , R _ { f } , P _ { p o s } , \mathcal { T } )$ consisting of a coarse texture map, a frontal rendering, a positional map, and the ground-truth texture. To explicitly train the network to rectify geometric distortions, we apply a random Thin Plate Spline (TPS) [Bookstein 1989] distortion to the initial texture image. The original, undistorted texture forms the ground-truth map T, while the TPS-distorted texture is overlaid onto the sewing pattern for physical simulation. Specifically, we perturb each control point $\mathscr { p } _ { i }$ of a regular grid fitted to the texture’s aspect ratio, giving $\hat { p } _ { i } = p _ { i } + \delta _ { i }$ with

$$
\delta _ { i } = \varepsilon _ { i } \cdot \lambda \cdot \frac { D } { 8 0 0 } , \varepsilon _ { i } \sim \mathcal { U } ( - 0 . 5 , 0 . 5 )\tag{3}
$$

where $\lambda = 2 5$ by default controls the overall distortion magnitude and � is the resolution of a texture image. A TPS is then fit between the regular grid $\{ p _ { i } \}$ and its perturbed counterpart $\{ \hat { p } _ { i } \}$ and applied densely to warp the full-resolution texture. We stitch and drape the sewing pattern onto an A-pose SMPL body to yield a 3D garment mesh $M _ { R } ,$ , from which we capture the frontal rendering $R _ { f }$ . The positional map $P _ { p o s }$ is then derived by projecting the ��� coordinates of the mesh vertices onto the 2D panel layout P. Finally, we synthesize the coarse texture map ${ \mathcal { T } } ^ { \prime }$ by projecting $R _ { f }$ and a randomly sampled view $R _ { a }$ onto a neutral-posed mesh and unwrapping them back into the 2D pattern space. This explicitly bakes physics-induced artifacts and deformations into the initialization, compelling the model to learn undistortion and normalized base-color recovery.

## 5 Experiments

In this section, we present a comprehensive evaluation of OmniFabric through quantitative and qualitative analyses. We first detail the experimental setup, including curated synthetic dataset and metrics used for evaluation. We then demonstrate the efectiveness of our framework by comparing against SOTA methods for 3D texturing. Finally, we conduct ablation studies to validate our core architectural designs. Additional implementation details, extended qualitative results, our method’s adaptability to other sewing pattern prediction methods besides ChatGarment [Bian et al. 2025], examples of our curated garment data, and discussion on potential future directions are provided in the supplementary material.

## 5.1 Setup

5.1.1 Datasets. To train and evaluate our model, we construct a large-scale synthetic dataset of textured sewing patterns, as described in Section 4. We sample a total of 3K unique and diverse garment samples from GarmentCodeData [Korosteleva et al. 2024]. We then use the pipeline in Figure 4 to generate texture images, and print them onto sampled sewing patterns, finally constructing a dataset of 30K textured sewing patterns with diverse appearances and structural designs (10 texture variations per garment style on average). We simulate garments using NVIDIA Warp [Macklin 2022] with a per-sample random seed, so repeated simulations of the same sewing pattern produce distinct $\textstyle { \mathcal { M } } _ { R }$ meshes and, consequently, dif ferent baked-in wrinkles in ${ \mathcal { T } } ^ { \prime } .$ . For lighting, we sample random subsets of white point lights to introduce varied illumination and baked-in shading across the dataset. We train our model with this synthetic dataset, and keep a held-out set for quantitative comparison with other baseline models with a test/train ratio of 0.1. Be sides synthetic data, we evaluate all methods on in-the-wild images, which is the major focus of this work. These in-the-wild images are obtained from DeepFashion [Liu et al. 2016] dataset and recent clothing-design images collected from the web. Due to the lack of ground truth in-the-wild images, we evaluate mainly through qualitative comparisons.

5.1.2 Metrics. We evaluate fidelity and structural coherence of the generated textures with a suite of standard metrics. We use LPIPS [Zhang et al. 2018] and DISTS [Ding et al. 2020] to measure visual similarity and structural consistency. We also compare with SSIM [Wang et al. 2004] and MS-SSIM [Wang et al. 2003], metrics employed to assess pixel-level structural accuracy. CLIP-score (CLIPs) [Gal et al. 2022] measures the semantic alignment between the final textured garment and the reference images.

5.1.3 Baseline methods. We compare against several SOTA methods in 3D texturing and garment-specific texture transfer. Among them, Hunyuan3D-2.0 [Zhao et al. 2025] and Paint3D [Zeng et al. 2024] use multi-view generation to provide additional prior beyond the given single-view observation. FabricDifusion [Zhang et al. 2024] is capable of creating fabric textures with no baked-in artifacts by normalizing local textile patterns with a difusion model.

5.1.4 Implementation details. Our texture normalization model is based on the Difusion Transformer (DiT) architecture, and we follow the multi-conditioning strategy of OminiControl [Tan et al. 2025a] by treating tokens from all input conditions as a unified token sequence. The model is fine-tuned using LoRA on a single NVIDIA A6000 GPU with a batch size of 8, keeping the VAE and primary DiT backbone fixed to preserve generative stability. We resize and pad sewing pattern maps to a resolution of 1024×1024 in a batch for training eficiency. During inference, we reverse the padding and resizing process, and deploy an image super-resolution model [Wang et al. 2021] to upscale the output. Since we are finetuning the model with LoRA, a higher training resolution can be easily achieved without GPU memory issues, making our framework scalable for training data of higher resolution. We fine-tune the model until convergence on our synthetic dataset. We utilize Veo 3 [Google DeepMind 2025] as the multi-view video generation prior and Nano Banana Pro [Google 2025] for the initial texture transfer to the frontal rendering $R _ { f } .$ We used ChatGarment [Bian et al. 2025] as the model for sewing pattern prediction throughout all experiments in the main paper. The average runtime for a single inference is roughly 5 minutes, depending on the current load on the Gemini model servers. The failure rate of Gemini models, e.g., generating unrelated textures on $R _ { f }$ or multi-views, is lower than 2%, estimated from randomly sampled generations. No additional filtering is necessary thanks to its stable performance. Additional details including prompts used for generation are provided in supplementary material.

## 5.2 Qualitative Comparisons

5.2.1 Texture synthesis by OmniFabricfrom in-the-wild images. We first show our results in Figure 5. Given a reference image, Omni-Fabric first generates a coarse texture map, serving as a roughly textured sewing pattern. Then, the texture normalization model removes the distortions and baked-in artifacts. From the front views of simulated 3D garments provided in Figure 5, we show that Omni-Fabric is capable of creating seamless textures across the 3D surface of garments. Note that while OmniFabric excels in textured garment synthesis from real-world images, we fine-tune our model only with synthetic data, which poses great scalability for model improvement.

5.2.2 Simulation results. We present simulations of synthesized 3D garments in Figure 9 via CLO. The sewing patterns generated by OmniFabric provide the normalized RGB base-color as albedo. Other parameters not predicted by our model—roughness, metallic, reflection intensity, and auto-generated normal map—are left at CLO’s default Fabric\_Matte preset values. The Environment/HDRI maps for relighting examples are also chosen from CLO’s lighting presets. The simulated assets exhibit a convincing and photorealistic dynamic appearance, maintaining structural integrity and texture coherence even under extreme dynamics and varying illumination.

![](images/1c2625726b46ab34f4c3182dafdd22601d7189ba54ed9d65459931e327343855.jpg)  
Fig. 5. Qualitative results of OmniFabric. Our method synthesizes 3D garments with high appearance consistency by predicting sewing paterns of normalized textures. In particular, OmniFabric handles intrinsic geometric distortions and reconstructs diverse texture types, including logos and printed graphics (1st and 2nd rows), and complex high-frequency paterns with dense visual details. Please zoom in to check the details.

5.2.3 Comparison with state-of-the-art methods. We compare with other methods in Figure 6 using in-the-wild images as well. For input with complex textures and elements, both Hunyuan3D and

![](images/6defbcad5dbbc0bb75cbe578be3973fbe66224aec56702a9b167062847823bc3.jpg)

![](images/fd6cb61f66e57227b25be60796cc95171083caedb2470dcd860799c9a1c030ad.jpg)

![](images/34914bf6b8009aadd934cff65b90dde0e6066bc94a7f6000c7745d598e41508d.jpg)

![](images/4934c7241079419e6c7e1db22ad1861e3b039f9adac63231fecd93379e4df01e.jpg)

![](images/e4ddfe8ff1d2aeb94f8ee634f297ef456ed7ca8980eed94e37e89a29d92d3e61.jpg)

![](images/ff02d78e6fa313a2868b24efbf6ca79213868210de44dc55c1be58bf7c18825d.jpg)

![](images/e949619b4c06cb206269bbad834c5f3d8f626b9807b6d37959eaf8860b63638e.jpg)

![](images/df4562ee95ea15ee208617fce97a98b1c01a8b0f40e62ff0cefe1d81ae4cfa6b.jpg)

![](images/c407e47977e66e50280bbc3c3de4e6965bd89cc4c9e24172e2f2edb8b4a93eb1.jpg)

![](images/4041ed0da59046d905b1d86efa1e37f87b0d53dc1e08d28c8954bf30064a5e85.jpg)

![](images/6f0c95a3cebbe40665b05c6fd80ab891b22731eddeff836f96c1bd8a24cf4e7d.jpg)

![](images/78c91a8d47173c974b5cded57e62b08dd39c65008d29e8e70402e06d35884c7c.jpg)

![](images/a55c59faf7b95e9fd1f37db853fdc0938a67bab6a94a9a9cb1e6efcc9bf7a224.jpg)

![](images/fb67f427e73d64fe8bc1702fabfbcbebaba4a5b544ef4188706670513c2b25ad.jpg)

![](images/85d32c0ea1aba7cc4ea40746e8545dc32119a232b5621be164e3a45df0f13f73.jpg)

![](images/48e2fae1240c42c9e3daeb5b8d46a55aa0fda6272429adffad71ce366ccba710.jpg)

![](images/361a6a826603c3aaf5dcc9ad86f75b273f9e8dd3ca52bbfe367f48c37dc52121.jpg)

![](images/546867588906291110e57b252d7df31d63122415983ab9a92768d68dfa6067fa.jpg)  
Input Image

![](images/abb587c01a9cf7926638b0ed631031551e6121f2bce77e4517ce1d62f780e1cb.jpg)  
OmniFabric (ours)

![](images/b5efda87b2e9e82e352bddbacdc0ef61220099df95b99852a3956c217b358a27.jpg)

![](images/481f0a6f0a67049a5b02eeffbe40ad9bf2c29d2b30c19195d0b718bd8539cdb6.jpg)

![](images/e42354500bc270840668f53e3fa7c6c0e9a1ce395f8686536b1511f6814166b8.jpg)  
Hunyuan3D-2.0

![](images/76e894526273970d61f37fda902007f600995d38ddec3559c9057209281c36b3.jpg)  
Paint3D

![](images/ba802b806e9c0a4d0646b9a87b441ffbe850b6e2784fadddf83d6a4fc331e2ee.jpg)  
FabricDiffusion

Fig. 6. Qualitative comparisons to state-of-the-art methods. OmniFabric is capable of generating accurate asymmetric textures and logos, while existing methods fail to preserve the high-frequency details on input cases with complex paterns.

Paint3D fail to preserve the high-frequency details. Even though FabricDifusion can create normalized textile pattern and texture the given mesh by tiling local textile patches, it is limited to local textile patches and fails to generate coherent global appearance. Conversely, OmniFabric creates normalized textures while preserving coherent details.

## 5.3 Quantitative Comparisons

5.3.1 Synthetic dataset. We show quantitative results where all baselines are evaluated with our curated synthetic dataset for fair comparison. For each test case, we provide every baseline with the identical frontal rendering $R _ { f }$ and rest-posed mesh M<sub>R</sub>. Each method then performs its respective 3D texturing task on M<sub>R</sub>. To compute image-based metrics, the resulting 3D textures are projected back into the sewing pattern UV space to generate a comparable texture map T for each baseline. Table 1 shows OmniFabric excels in all metrics, particularly in preserving global structural coherence and eliminating projection artifacts.

5.3.2 Real-world data. Besides comparing with other methods in Figure 6, we conduct a user study to evaluate the performances ofthe methods due to the absence of real-world dataset of textured sewing patterns. As detailed in Table 2, the study involves 13 participants evaluating results from 10 real-world image inputs based on four criteria: overall quality ranking (1 to 4, lower is better), fidelity to the input appearance, back-view plausibility, and global texture coherence (1 to 5, higher is better). OmniFabric achieves the best results across all criteria, demonstrating that it is preferred on real images and more faithfully preserves coherent garment textures.

Table 1. Quantitative comparisons. We compare with baselines on our synthetic dataset. The results show that OmniFabric greatly surpasses all other methods in terms of both pixel and perceptual quality.
<table><tr><td></td><td>LPIPS ↓</td><td>SSIM ↑</td><td>MS-SSIM ↑</td><td>DISTS ↓</td><td>CLIP-s ↑</td></tr><tr><td>FabricDiffusion [Zhang et al. 2024]</td><td>0.273</td><td>0.645</td><td>0.707</td><td>0.259</td><td>0.906</td></tr><tr><td>Paint3D [Zeng et al. 2024]</td><td>0.311</td><td>0.657</td><td>0.704</td><td>0.266</td><td>0.890</td></tr><tr><td>Hunyuan3D-2.0 [Zhao et al. 2025]</td><td>0.223</td><td>0.712</td><td>0.717</td><td>0.220</td><td>0.924</td></tr><tr><td>OmniFabric (ours)</td><td>0.092</td><td>0.868</td><td>0.905</td><td>0.121</td><td>0.963</td></tr></table>

Table 2. User study for real-world data. We compare with baselines on real-world input images in 4 aspects: overall quality, fidelity, back-view plausibility and global coherence.
<table><tr><td></td><td>Overall rank ↓</td><td>Fidelity ↑</td><td>Back-view ↑</td><td>Global Coherence ↑</td></tr><tr><td>FabricDiffusion [Zhang et al. 2024]</td><td>3.40</td><td>1.75</td><td>2.69</td><td>3.15</td></tr><tr><td>Paint3D [Zeng et al. 2024]</td><td>3.41</td><td>1.86</td><td>2.81</td><td>3.03</td></tr><tr><td>Hunyuan3D-2.0 [Zhao et al. 2025]</td><td>2.10</td><td>3.43</td><td>3.41</td><td>3.50</td></tr><tr><td>OmniFabric (ours)</td><td>1.09</td><td>4.44</td><td>4.35</td><td>4.41</td></tr></table>

## 5.4 Reproducibility via Open-Source Models

While using Gemini models for pose transfer and multi-view generation (Section 3.2) as default implementation of OmniFabric, we further construct a fully open-source alternative to facilitate reproducibility. To replace the closed-source models, we use FLUX.2 [Labs 2025] as image generation backbone and incorporate Ministral 3 [Liu et al. 2026] for prompt up-sampling. We feed textured frontal view

Table 3. We construct a fully open-source alternative and it achieves similar performance to the default setings using closed-source models, showing the great reproducibility of our framework.
<table><tr><td></td><td>LPIPS ↓</td><td>SSIM ↑</td><td>DISTS ↓</td><td>CLIP-s ↑</td></tr><tr><td>OmniFabric w/ default settings</td><td>0.092</td><td>0.868</td><td>0.121</td><td>0.963</td></tr><tr><td>OmniFabric w/ open-source models</td><td>0.114</td><td>0.847</td><td>0.149</td><td>0.955</td></tr></table>

Table 4. We analyze error accumulation across certain modules by replacing intermediate outputs in our model pipeline with ground-truth (GT) counterparts. “Pred.” and $^ { \ast } G \mathsf { T } ^ { \ast }$ mean the intermediate output is provided by Gemini models and GT respectively, and "SP Norm." refers to sewing patern normalization. Results show that our texture normalization model efectively removes the accumulated error.
<table><tr><td>Replacement Setting</td><td>LPIPS ↓</td><td>SSIM ↑</td><td>DISTS ↓</td><td>CLIP-s ↑</td></tr><tr><td>(a) GT  $I ^ { \prime } + \operatorname { G T } V$ </td><td>0.105</td><td>0.849</td><td>0.159</td><td>0.944</td></tr><tr><td>(b) Pred.  $I ^ { \prime } + \operatorname { G T } V$ </td><td>0.121</td><td>0.837</td><td>0.176</td><td>0.935</td></tr><tr><td>(c) Pred.  $I ^ { \prime } + \mathrm { P r e d } , V$ </td><td>0.144</td><td>0.804</td><td>0.166</td><td>0.938</td></tr><tr><td>(d) Pred.  $I ^ { \prime } + \mathrm { P r e d . } V + \mathrm { S P } \mathrm { N o r m } .$ </td><td>0.092</td><td>0.868</td><td>0.121</td><td>0.963</td></tr></table>

Table 5. Ablation study shows that our designs are essential for the texture normalization model to generate position-aligned details with high fidelity.
<table><tr><td>Setting</td><td>LPIPS ↓</td><td>SSIM ↑</td><td>DISTS ↓</td><td>CLIP-s ↑</td></tr><tr><td>OmniFabric (ours)</td><td>0.092</td><td>0.868</td><td>0.121</td><td>0.963</td></tr><tr><td>(a) w/o coarse texture map  $\mathcal { T } ^ { \prime }$ </td><td>0.407</td><td>0.601</td><td>0.306</td><td>0.887</td></tr><tr><td>(b) w/o frontal view rendering  $R _ { f }$ </td><td>0.097</td><td>0.850</td><td>0.129</td><td>0.962</td></tr><tr><td>(c) w/o position map  $P _ { p o s }$ </td><td>0.098</td><td>0.848</td><td>0.130</td><td>0.961</td></tr></table>

$R _ { f }$ and silhouettes of four orthogonal views as conditions to generate the multi-view with this alternative. Table 3 shows that open-source alternative achieves similar performance with the same experiment settings in Section 5.3.

## 5.5 Error-Accumulation Analysis

We analyze error accumulation across modules in our model architecture in this section by progressively replacing intermediate outputs, specifically the reposed reference image �<sup>′</sup> and generated multi-view �, with ground-truth counterparts. As shown in Table $^ { 4 , }$ settings (a) to (c) accumulate errors across stages including texture projection, pose alignment and multi-view generation, and (d) shows that our normalization model efectively fixes these aggregated artifacts on sewing patterns.

## 5.6 Ablation Studies and Analyses

5.6.1 Importance ofthe canonical 3D and coarse texture map. As shown in Table 5-(a), without the coarse texture map, which serves as the distillation of generative priors, the performance drops significantly. This shows that the projected textures do provide essential guidance for the texture generation. Figure 8 shows that the model is randomly generating textures that “look similar” instead of pixelaligned content without coarse texture map T<sup>′</sup>.

5.6.2 Efect ofthefrontal view rendering condition. We observe that removing $R _ { f }$ during training leads to inferior results (Table 5-(b)), suggesting that the frontal-view rendering provides important cues for the model. In particular, $R _ { f }$ ofers a holistic visual reference of the garment appearance, helping the model better understand global layout and consistency of the texture across sewing pattern pieces.

![](images/42f941429f3ad49bded2ece7bb0700a1a500f80234b72889d206c8ad1dffd8f3.jpg)  
Fig. 7. Limitations. OmniFabric may synthesize mismatched textures from inaccurately generated multi-views (1st row), or generate garments with incorrect geometry even though appearance is well-transferred (2nd row).

5.6.3 Efect ofposition map. As shown in Table 5-(c), without $P _ { p o s } ,$ the model not only shows suboptimal results but also textures and elements in wrong positions. As illustrated in Figure 8, a lemon is mistakenly generated in a blank region of the sewing pattern.

5.6.4 Importance of geometric distortion in dataset. The right side of Figure 8 shows the necessity of creating geometric distortion in the synthetic training data. Through training to remove geometric distortion, the model can not only generate normalized structured patterns on the sleeves, e.g., the checker pattern, but also produce clean appearances for other elements, e.g., texts and logos.

## 5.7 Limitations

As shown in Figure 7, correctness of synthesized textures can be afected by inaccurately generated multi-views (1st row). OmniFabric is also unable to refine the mismatched geometry $M _ { R }$ predicted by ChatGarment [Bian et al. 2025] (2nd row); however, it does not change the role of our refinement model: given a sewing pattern, it normalizes the textures by reducing distortions, wrinkles, shadows, and projection artifacts, regardless of specific garment geometry. We put more discussion of limitations in the supplementary material.

## 6 Conclusion and Discussion

We introduce OmniFabric, a novel framework for synthesizing 3D garments with normalized and globally coherent textures. By leveraging large-scale generative priors and training a texture normalization model, we efectively disentangle intrinsic albedo from view-dependent artifacts including baked-in shadows and wrinkles. Our method bridges the gap between single-view observations and simulation-ready 3D assets, maintaining structural integrity across complex sewing patterns. For this study, we focus on the synthesis of sewing patterns with normalized textures within the Garment-Code framework. An immediate and promising extension would be to consider the joint prediction of complete PBR material maps, including roughness, metallic, and normal maps, alongside the albedo. We believe this direction will further enhance the photorealism of the reconstructed garments under diverse environmental lighting, providing even greater utility for immersive digital content creation.

![](images/05b2b348fe6b444a2a6e5174d048154fddece194e3200dc8bc696aedb942d513.jpg)

![](images/b16925d443976cbd36c696c23a7d02c44697bbe26d941070d220aca3b0619912.jpg)

![](images/a0d5ff09b38676b44aca1067874be516c6f574168ea7bc5452f0b66e1a4f1391.jpg)

![](images/5dd03e6c373f8ed91f5cc6c38f86eb3bfef006078427bac260e6b7c89551eaeb.jpg)  
(a) w/o coarse texture map

![](images/cdaebfe090f92c5fc70f2197e1abce92ee0a58898e8153394efcdd818d24a5cc.jpg)

![](images/4549654982de95381f1d81b9d51dcfcbd50f33b4161094b46d54f91266dab73b.jpg)  
(b) w/o frontal view rendering

![](images/e4e20b95c7969985ddaa41b962415d15abda656a8df30cd61e4124d806062e7c.jpg)

![](images/0cb359f708ec2e1eddcacde99f86249a767c9c984531f631f329a1191aea1587.jpg)

![](images/2190fb547566fa26c1c393e86e05ceaab9ca274937fc033c8588bf8a6f7d5932.jpg)  
Trained w/o distortion removal

![](images/4618282d6dc54df5eb1c85066eb7d3b2078b915917d151c67d74ab0882c301ff.jpg)  
Trained w/ distortion removal

Fig. 8. Qualitative analysis of key design mechanisms. We demonstrate the efectiveness of the core designs of our texture normalization model. Without $\mathcal { T } ^ { \prime } { } _ { ; }$ , the model fails to generate consistent appearance but arbitrary textures with similar colors and elements. We also observe that $R _ { f }$ and $P _ { p o s }$ are both essential conditions for the model to generate pixel-aligned content observed from the input image (left figure). We also show that our method can efectively rectify distortion when trained with distortion-augmented dataset (right figure).  
![](images/6116b5d731839b20f437c645e9cf12c5566d306c66a4b9b3a0bfa7f03ee58764.jpg)

![](images/e04b5ef10196aa9c4b39959a1109a64f0db91e8232460a463df596bcd87777c8.jpg)

![](images/a9008e33b6d1e9983a13f263d231207a17c7fb887f2a7420387c65127f874a40.jpg)

![](images/64e051ddb928b23eba4e254ba1c113681996fd0f05cac379efd73252fb1a11b6.jpg)

![](images/a5f6dd331cb6884458732aa1d5a4e9ea7c53b9b881cb9ae3add762d86cd76f99.jpg)

![](images/ac179211c1524404ee0f11508597f4b572d216c39dc984ba3df7d3856ab2c675.jpg)

![](images/41f8d2e996e5524ffbc7628d7249afb299371db349dc3ea0c6510506866fe5c9.jpg)

![](images/03c05af0633538d93ce371b370efd3d3c6e6afad4c3c03ad644f8462f6dbb9fa.jpg)

![](images/23b7586a8e1e9b2fec7ee4c8f412ea752f9c5a332c17e5bcf0ac5ea39317b674.jpg)  
Input Image

![](images/908e4252069556db5d03410e4d78568a9a1efc972865bf0aed9c4a6ed60b7f19.jpg)  
Sewing Pattern with Baked-in artifacts

![](images/aa78dd5937a7bfbddea14d2b70122dde07c5801586d79b4a5f1835f40c81534f.jpg)  
Normalized Sewing Pattern

![](images/86b3ce8ecaa63759061d944f44e04c41ba1892d28cf991c7d83561e570c8694b.jpg)  
Simulation-Ready 3D Garment

Fig. 9. Simulation results of our generated 3D garments. 3D garments generated by our method can be directly simulated under varying human pose and lighting conditions and show convincing results.

## References

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609 (2023).

Raphael Bensadoun, Yanir Kleiman, Idan Azuri, Omri Harosh, Andrea Vedaldi, Natalia Neverova, and Oran Gafni. 2024. Meta 3d texturegen: Fast and consistent texture generation for 3d objects. arXiv preprint arXiv:2407.02430 (2024).

Siyuan Bian, Chenghao Xu, Yuliang Xiu, Artur Grigorev, Zhen Liu, Cewu Lu, Michael J Black, and Yao Feng. 2025. Chatgarment: Garment estimation, generation and editing via large language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2924–2934.

Fred L. Bookstein. 1989. Principal warps: Thin-plate splines and the decomposition of deformations. IEEE Transactions on pattern analysis and machine intelligence 11, 6 (1989), 567–585.

Tianshi Cao, Karsten Kreis, Sanja Fidler, Nicholas Sharp, and Kangxue Yin. 2023. Tex fusion: Synthesizing 3d textures with text-guided image difusion models. In Proceedings ofthe IEEE/CVF international conference on computer vision. 4169–4181.

Cheng-Hsiu Chen, Jheng-Wei Su, Min-Chun Hu, Chih-Yuan Yao, and Hung-Kuo Chu. 2024. Panelformer: Sewing pattern reconstruction from 2D garment images. In Proceedings ofthe IEEE/CVF winter conference on applications ofcomputer vision. 454–463.

Dave Zhenyu Chen, Yawar Siddiqui, Hsin-Ying Lee, Sergey Tulyakov, and Matthias Nießner. 2023b. Text2tex: Text-driven texture synthesis via difusion models. In Proceedings of the IEEE/CVF international conference on computer vision. 18558–18568.

Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. 2023a. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In Proceedings of the IEEE/CVF international conference on computer vision. 22246–22256.

Yongwei Chen, Rui Chen, Jiabao Lei, Yabin Zhang, and Kui Jia. 2022. Tango: Text-driven photorealistic and robust 3d stylization via lighting decomposition. Advances in neural information processing systems 35 (2022), 30923–30936.

Wei Cheng, Juncheng Mu, Xianfang Zeng, Xin Chen, Anqi Pang, Chi Zhang, Zhibin Wang, Bin Fu, Gang Yu, Ziwei Liu, et al. 2025. Mvpaint: Synchronized multi-view difusion for painting anything 3d. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 585–594.

CLO Virtual Fashion Inc. 2024. CLO. https://www.clo3d.com. Version 7.2.

Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. 2020. Image quality assess ment: Unifying structure and texture similarity. IEEE transactions on pattern analysis and machine intelligence 44, 5 (2020), 2567–2581.

Google. 2025. Gemini 3 Pro Image. https://gemini.google.com/.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An image is worth one word: Personalizing text-to image generation using textual inversion. arXiv preprint arXiv:2208.01618 (2022).

Google DeepMind. 2025. Veo: Google’s most capable video generation model. https: //deepmind.google/models/veo/. Accessed: September 25, 2026.

Marc Habermann, Lingjie Liu, Weipeng Xu, Michael Zollhoefer, Gerard Pons-Moll, and Christian Theobalt. 2021. Real-time deep dynamic characters. ACM Transactions on Graphics (ToG) 40, 4 (2021), 1–16.

Kai He, Kaixin Yao, Qixuan Zhang, Jingyi Yu, Lingjie Liu, and Lan Xu. 2024. Dresscode: Autoregressively sewing and generating garments from text guidance. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–13.

Fangzhou Hong, Mingyuan Zhang, Liang Pan, Zhongang Cai, Lei Yang, and Ziwei Liu. 2022. Avatarclip: Zero-shot text-driven generation and animation of 3d avatars. arXiv preprint arXiv:2205.08535 (2022).

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. Iclr 1, 2 (2022), 3.

Christopher J Hughes, Radek Grzeszczuk, Eftychios Sifakis, Daehyun Kim, Sanjeev Kumar, Andrew P Selle, Jatin Chhugani, Matthew Holliman, and Yen-Kuang Chen. 2007. Physical simulation for animation and visual efects: parallelization and characterization for chip multiprocessors. ACM SIGARCH Computer Architecture News 35, 2 (2007), 220–231

Hyunwoo Hwangbo, Eun Hie Kim, So-Hyun Lee, and Young Jae Jang. 2020. Efects of 3D virtual “try-on” on online sales and customers’ purchasing experiences. Ieee Access 8 (2020), 189479–189489.

Dong-Eun Kim and Karen LaBat. 2013. Consumer experience in using 3D virtual garment simulation technology. Journal ofthe Textile Institute 104, 8 (2013), 819– 829.

Maria Korosteleva, Timur Levent Kesdogan, Fabian Kemper, Stephan Wenninger, Jasmin Koller, Yuhan Zhang, Mario Botsch, and Olga Sorkine-Hornung. 2024. Garment-CodeData: A dataset of 3D made-to-measure garments with sewing patterns. In European Conference on Computer Vision. Springer, 110–127.

Maria Korosteleva and Sung-Hee Lee. 2022. Neuraltailor: Reconstructing sewing pattern structures from 3d point clouds of garments. ACM Transactions on Graphics (TOG) 41, 4 (2022), 1–16.

Maria Korosteleva and Olga Sorkine-Hornung. 2023. Garmentcode: Programming parametric sewing patterns. ACM Transactions on Graphics (TOG) 42, 6 (2023), 1–15.

Black Forest Labs. 2025. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2.

Xinyu Li, Qi Yao, and Yuanda Wang. 2025a. GarmentDifusion: 3D garment sewing pattern generation with multimodal difusion transformers. arXiv preprint arXiv:2504.21476 (2025).

Xuan Li, Chang Yu, Wenxin Du, Ying Jiang, Tianyi Xie, Yunuo Chen, Yin Yang, and Chenfanfu Jiang. 2025b. Dress-1-to-3: Single image to simulation-ready 3d outfit with difusion prior and diferentiable physics. ACM Transactions on Graphics (TOG) 44, 4 (2025), 1–16.

Alexander H Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhi nav Rastogi, Adrien Sadé, Alan Jefares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, et al. 2026. Ministral 3. arXiv preprint arXiv:2601.08584 (2026).

Lijuan Liu, Xiangyu Xu, Zhijie Lin, Jiabin Liang, and Shuicheng Yan. 2023. Towards garment sewing pattern reconstruction from a single image. ACM Transactions on Graphics (TOG) 42, 6 (2023), 1–15.

Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. 2016. DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. In Proceedings ofIEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Zhongjin Luo, Haolin Liu, Chenghong Li, Wanghao Du, Zirong Jin, Wanhu Sun, Yinyu Nie, Weikai Chen, and Xiaoguang Han. 2024. Garverselod: High-fidelity 3d garment reconstruction from a single in-the-wild image using a dataset with levels of details. ACM Transactions on Graphics (TOG) 43, 6 (2024), 1–12.

Miles Macklin. 2022. Warp: A High-performance Python Framework for GPU Simulation and Graphics. https://github.com/NVIDIA/warp NVIDIA GPU Technology Conference (GTC).

Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. 2023. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 12663–12673.

Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. 2022. Text2mesh: Text-driven neural stylization for meshes. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 13492–13502.

Nasir Mohammad Khalid, Tianhao Xie, Eugene Belilovsky, and Tiberiu Popa. 2022. Clip mesh: Generating textured meshes from text using pretrained image-text models. In SIGGRAPH Asia 2022 conference papers. 1–8.

Kiyohiro Nakayama, Jan Ackermann, Timur Levent Kesdogan, Yang Zheng, Maria Korosteleva, Olga Sorkine-Hornung, Leonidas J Guibas, Guandao Yang, and Gordon Wetzstein. 2025. Aipparel: A multimodal foundation model for digital garments. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 8138–8149.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d difusion. arXiv preprint arXiv:2209.14988 (2022).

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

Elad Richardson, Gal Metzer, Yuval Alaluf, Raja Giryes, and Daniel Cohen-Or. 2023. Texture: Text-guided texturing of 3d shapes. In ACM SIGGRAPH 2023 conference proceedings. 1–11.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Boxiang Rong, Artur Grigorev, Wenbo Wang, Michael J Black, Bernhard Thomaszewski, Christina Tsalicoglou, and Otmar Hilliges. 2025. Gaussian garments: Reconstructing simulation-ready clothing with photorealistic appearance from multi-view video. In 2025 International Conference on 3D Vision (3DV). IEEE, 1054–1063.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image difusion models with deep language understanding. Advances in neural information processing systems 35 (2022), 36479–36494.

Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. 2019. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In Proceedings ofthe IEEE/CVF international conference on computer vision. 2304–2314.

Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul Joo. 2020. Pifuhd: Multilevel pixel-aligned implicit function for high-resolution 3d human digitization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 84–93.

Nikolaos Sarafianos, Tuur Stuyck, Xiaoyu Xiang, Yilei Li, Jovan Popovic, and Rakesh Ranjan. 2025. Garment3dgen: 3d garment stylization and texture generation. In 2025 International Conference on 3D Vision (3DV). IEEE, 1382–1393.

Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. 2025a. Ominicontrol: Minimal and universal control for difusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 14940–14950.

Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, and Xinchao Wang. 2025b. Ominicontrol2: Eficient conditioning for difusion transformers. arXiv preprint arXiv:2503.08280 (2025).

Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. 2021. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings ofthe

IEEE/CVF international conference on computer vision. 1905–1914.

Yuanhao Wang, Cheng Zhang, Gonçalo Frazão, Jinlong Yang, Alexandru-Eugen Ichim, Thabo Beeler, and Fernando De la Torre. 2025. GarmentCrafter: Progressive Novel View Synthesis for Single-View 3D Garment Reconstruction and Editing. arXiv preprint arXiv:2503.08678 (2025).

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.

Zhou Wang, Eero P Simoncelli, and Alan C Bovik. 2003. Multiscale structural similarity for image quality assessment. In The Thirty-Seventh Asilomar Conference on signals, systems & computers, 2003, Vol. 2. Ieee, 1398–1402.

Yuliang Xiu, Jinlong Yang, Dimitrios Tzionas, and Michael J Black. 2022. Icon: Implicit clothed humans obtained from normals. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 13286–13296.

Xianfang Zeng, Xin Chen, Zhongqi Qi, Wen Liu, Zibo Zhao, Zhibin Wang, Bin Fu, Yong Liu, and Gang Yu. 2024. Paint3d: Paint anything 3d with lighting-less texture difusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 4252–4262.

Cheng Zhang, Yuanhao Wang, Francisco Vicente, Chenglei Wu, Jinlong Yang, Thabo Beeler, and Fernando De la Torre. 2024. FabricDifusion: High-fidelity texture transfer for 3D garments generation from in-the-wild images. In SIGGRAPH Asia 2024 Conference Papers. 1–12.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. 2025. Hunyuan3d 2.0: Scaling difusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202 (2025).

Feng Zhou, Ruiyang Liu, Chen Liu, Gaofeng He, Yong-Lu Li, Xiaogang Jin, and Huamin Wang. 2025. Design2GarmentCode: Turning design concepts to tangible garments through program synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference. 23712–23722.

## Supplementary Material

## A Details of Dataset Construction

A primary contribution of this work is the automated data engine designed to generate high-fidelity textured sewing patterns. While high-detail manual texturing is possible, the process remains prohibitively time-consuming for large-scale applications. In contrast, our pipeline eficiently produces diverse garment styles and textures with a vast range of complexity, facilitating the construction of digital garment datasets at scale. This automated engine is a core contribution that addresses the scarcity of complex, high-resolution textures in existing garment research. We have included several representative examples in this supplement and will release the full dataset as well as the curation pipeline.

## A.1 Reordering Sewing Patern for Texturing

To align our generation process with real-world manufacturing, we introduce a reordering strategy for sewing pattern layouts. In professional garment construction, textures are often seamless across specific functional groups—such as a large frontal fabric panel—but exhibit discontinuities at structural seams, such as the transition from a sleeve to the chest. These discontinuities arise because 3Dstitched edges are often not geometrically matched in 2D space (e.g., a curved sleeve head joining a straight armhole), necessitating diferent fabric cuts. Following these observations, our pipeline first reorders the 2D sewing pattern pieces by merging related panels, such as the front and back pieces of a single sleeve, into unified spatial groups. By overlaying textures onto this reordered 2D layout, we ensure seamless pattern continuity within each functional group while maintaining the realistic texture breaks required for production-ready 3D assets.

## B Additional Implementation Details

In this section, we provide further technical specifics regarding our texturing pipeline. As detailed in the main paper, we utilize a single-view garment observation to generate a 360-degree rotation video. From this sequence, we extract four keyframes corresponding to the orthogonal front, back, left, and right camera views for the multi-view texture projection process. Given that the video depicts a garment in a canonical A-pose, these four views are suficient to capture the vast majority of surface details. The rest-posed mesh, $\textstyle { \mathcal { M } } _ { \mathcal { R } } .$ , is reconstructed by stitching and draping the predicted sewing patterns onto a A-posed SMPL human body model.

## B.1 Non-overlapping Texture Projection

A key distinction between our projection method and state-of-theart baselines, such as Hunyuan3D-2.0, lies in the source and application of multi-view data. While traditional methods project independent images generated by multi-view difusion models, we project selected frames from a structurally consistent video prior using a specialized non-overlapping strategy. We observe that while multi-view difusion models improve cross-view alignment, they often sacrifice high-frequency details to maintain that consistency. To preserve these details, our non-overlapping strategy begins by projecting textures from the front and back views—the two perspectives with minimal observational overlap—directly onto the mesh. Subsequently, textures from the side views are projected exclusively onto previously untextured regions, efectively performing a projection-based inpainting. Finally, our data-driven texture normalization model rectifies any minor artifacts or discontinuities at the seams of these projected textures. This approach allows OmniFabric to retain intricate, high-frequency surface details while ensuring global multi-view consistency.

## B.2 Prompt Configurations

In this section we specify the design of prompts for the Gemini models in our framework. We use the prompt below for pose aware texture alignment, which generates a textured frontal view of restpose garment mesh $\textstyle { \mathcal { M } } _ { R } :$

The first apparel is my target apparel. Please generate textures on this apparel, so the textures are identical to the apparel in the reference image. Please don’t change the style, size and pose of the target apparel.

For the multi-view generation with video model, we use the following prompt:

Create a continuous 360 degree rotation video of this apparel, making it rotate horizontally 360 degree, like a microwave.

## B.3 Training Details

We fine-tune all three baselines from pretrained weights, adapting the components described below. For Paint3D, we fine-tune its position encoder, as specified in its paper. For Hunyuan3D, we fine-tune

![](images/e25ef7df8f20359504921d62727ecbe6c8e19153c09b033c11893a9fb70b3124.jpg)  
Fig. S1. Data examples of our synthetic dataset. We show the synthesized textured sewing patern and its simulated 3D garment for each datapoint.

its multi-view image generator. And for FabricDifusion, we train its texture generator using paired textile data. All the models are fine-tuned upon pre-trained model weights.

## C Additional Results and Analyses

![](images/5dffcb65ef9f2ceedfb798cf2cd21c766bd87c4724b36aaa51b02d76078c6118.jpg)  
Fig. S2. Adaptability to other sewing patern prediction method. The simulation results show that OmniFabric can be seamlessly integrated with other methods of sewing patern prediction.

## C.1 Robustness to Structural Variations in Sewing Paterns

We present further qualitative evaluations in Figure S3, which demonstrate the robustness of our framework across a variety of garment styles. The results illustrate that our texture normalization model efectively rectifies the coarse texture map by eliminating projectioninduced distortions and removing physics-based wrinkles inherent in the initial capture. While the geometry simulated from the predicted sewing patterns may occasionally deviate from the exact silhouette in the input image, we emphasize that such discrepancies arise from the limitations of the underlying sewing pattern prediction method rather than the texturing process. To mitigate this, our pipeline employs a texture transferring model that aligns the input image textures with the rendered silhouette of the predicted garment. This strategy successfully bridges the gap between the predicted geometry and the original observation, ensuring that the synthesized textures remain globally coherent and structurally aligned.

## C.2 Compatibility with Other Sewing Patern Prediction Method

While the main paper utilizes ChatGarment for sewing pattern prediction, our framework is designed to be agnostic to the specific reconstruction method employed. To demonstrate this flexibility, we evaluated our pipeline using sewing patterns predicted by Alp parel [Nakayama et al. 2025], a recent state-of-the-art approach based on Large Vision-Language Models. As illustrated in Figure S2, OmniFabric consistently generates textured garments with aligned geometry and high-frequency details, confirming that our synthesis capability is not restricted to a single pattern reconstruction architecture.It should be noted, however, that the simulated garment geometry may occasionally exhibit minor mismatches relative to the original input image due to imperfect pattern prediction. As discussed in Section C.1, our methodology specifically focuses on maintaining global texture consistency even when the underlying ensures that OmniFabric can be seamlessly integrated with future advancements in sewing pattern prediction.

![](images/4508e786e5095ac5fbffb475e8d7f965816ced9aa427246846c505fc591457cf.jpg)  
Fig. S3. Qualitative results. OmniFabric efectively removes distortion artifacts and geometry-induced wrinkles, generating normalized textured sewing paterns that are simulation-ready.  
3D mesh, which is simulated from predicted sewing pattern, is not perfectly aligned with the reference observation. This robustness

## D Limitations and Future Work

While OmniFabric represents a significant advancement in garment texturing, several limitations remain that ofer promising avenues for future research. First, although our model efectively removes transient lighting artifacts to produce a clean appearance, it does not yet explicitly disentangle albedo from shading in a strictly principled, physics-based manner, a distinction that should be noted to avoid overstating our current appearance decomposition capabilities. Furthermore, because the framework relies on generative priors, unseen or occluded regions are occasionally hallucinated; however, the system is not strictly limited to single-view observations and could incorporate multiple frames or video sequences in practice to reduce these hallucinations and improve reconstruction fidelity. Finally, certain materials with strong view-dependent effects, such as highly reflective leather or metallic fabrics, remain challenging to reconstruct faithfully, suggesting that future iterations should incorporate the joint prediction of complete PBR material maps—including roughness and normal maps—to achieve true photorealism across diverse environmental lighting conditions.