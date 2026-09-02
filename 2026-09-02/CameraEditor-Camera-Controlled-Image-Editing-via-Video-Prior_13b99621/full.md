# CameraEditor: Camera-Controlled Image Editing via Video-Prior Sequential Modeling

Xin Shen<sup>1∗</sup> Chengyou Jia<sup>1∗</sup> Keshuo Xing<sup>1</sup> Zifeng Zhu<sup>1</sup> Changliang Xia<sup>1</sup>

Bowen Ping<sup>1</sup> Zhuohang Dang<sup>1</sup> Hangwei Qian<sup>2</sup> Minnan Luo<sup>1†</sup>

<sup>1</sup>Xi’an Jiaotong University, China <sup>2</sup>Agency for Science, Technology and Research (A\*STAR), Singapore

<sup>§</sup> Code Model Data <sup>Ñ</sup> Homepage

<sup>∗</sup>Equal contribution <sup>†</sup>Corresponding author

![](images/68bba2331c31eee138f56d9e9b5296bc116d92f8dc732c2cebec2c349c8358c4.jpg)  
Figure 1: Overview of CameraEditor. We introduce a video-prior framework for precise camera-parameterized editing. (Top) Large perspective shifts are decomposed via intermediate frame interpolation. (Bottom) CameraEditor achieves state-of-the-art precision across diverse scenes, preserving fine-grained details even under extreme rotations and lens distortions.

## Abstract

Beyond semantic content, camera parameters play a pivotal role in dictating the geometric perspective and appearance of any given image. While recent image editing models excel at semantic and stylistic manipulation, they struggle with explicit camera parameter control. When handling large perspective shifts, instruction-driven models face a dilemma: they either sufer from structural tearing or generate conservative outputs that ignore geometric instructions. To address this, we introduce CameraEditor, a framework that reformulates camera-controlled editing from a spatial problem into a temporal sequence prediction task. By leveraging the tempo ral coherence of video difusion models, our approach integrates an explicit geometric perception module with a dynamic reference routing mechanism. This allows us to construct geometrically rigorous visual reference pairs via dynamic panorama cropping, overcoming the ambiguity of text-based instructions. Furthermore, CameraEditor strategically inserts intermediate transition frames to decompose large perspective shifts, providing a robust temporal bufer that preserves content identity and spatial coherence. We construct a training dataset of 5,760 instances. As an independent

contribution, we introduce CamEditor-Bench, a model-agnostic evaluation suite of 462 test cases. Extensive experiments demonstrate that CameraEditor achieves state-of-the-art camera control precision and source identity preservation, outperforming existing methods.

## Keywords

camera-controlled image editing, video generative models, sequential modeling

## 1 Introduction

Recently, image editing has witnessed remarkable progress [62]. Not only closed-source models such as GPT-Image-1.5 [33] and Nano-Banana-Pro [39], but also some open source models like Qwen Image-Editing [54] and Flux.2 [3] have demonstrated impressive capabilities in manipulating images based on user instructions. They can perform various editing tasks [15], including semantic editing [14, 41, 59], stylistic editing [9, 21], and spatial editing [58]. However, an image’s composition is shaped by both its content and camera parameters (e.g., focal length, aperture, pose). Even when keeping the content unchanged, adjusting camera parameters can significantly alter the visual impression. In that case, editing images through camera parameters is essential for precise and controlled visual manipulation. Unfortunately, they often fall short in delivering accurate and fine-grained edits aligned with specific camera parameters adjustments.

![](images/210034a8be6916efd90bf07fcc119756248455a258b72f3b83d7ac4a9b3d89b2.jpg)

![](images/af6bac9df67fa09841c525bc5f2f5416804412ebc3327965ca59b248ae9e28dc.jpg)

![](images/2efde0b4abbc924e782b2fbdc6491ebf507cb90d6d7af5cf26753da28b896b10.jpg)  
Figure 2: Limitations of instruction-driven models. Ambiguous text prompts without mathematical constraints cause three critical failures: (a) Terminology Misinterpretation (conflating geometric shifts with semantic changes), (b) Parametric Imprecision (failing exact-degree adjustments), and (c) Content Identity Loss (structural tearing under complex distortions).

As demonstrated in Figure 2, these limitations manifest in three critical dimensions: (1) Terminology Misinterpretation: Generalpurpose models lack rigorous geometric understanding, often con flating precise camera parameters (e.g., “pitch” with broad semantic concepts like “aerial view”). (2) Parametric Imprecision: Operating without exact mathematical anchors, existing methods struggle to execute fine-grained rotation or exact-degree geometric adjustments (e.g., failing to reach the instructed 60<sup>◦</sup> rotation). (3) Identity Preservation Conflict: Under large perspective shifts, models face a dilemma: they either sufer structural tearing (e.g., the severely warped building) to follow instructions, or generate conservative outputs that ignore geometric changes to preserve identity.

To address these limitations, we formulate the task as editing a source image under specified camera-parameter transformations and propose a framework for camera-parameterized image editing, named CameraEditor. Unlike conventional methods that rely on pre-trained text-to-image models for image editing [11, 18], our approach reformulates this editing task as a sequence prediction problem, which helps the model fully leverage video priors. Furthermore, it brings two additional benefits: (1) By leveraging video models, we use a reference image pair consisting of a source image and its edited version to guide the model in applying consistent camera-parameter transformations to the target image. This approach overcomes the limitations of text-based prompts, which often struggle to capture precise geometric changes, and pixel-level control maps that may distort overall structure. (2) Inspired by the Chain of Frames (CoF) [8] approach, our method inserts transition frames for both reference and target image pairs. By aligning these transition steps frame-by-frame, we facilitate highly smooth, consistent transformations. We efectively decompose large-magnitude geometric transformations into a traceable trajectory, thereby ensuring smooth and consistent results.

To support our framework, we construct a dataset generation pipeline that synthesizes a training dataset containing 5,760 instances with detailed camera parameter annotations from both real-world images and synthetic scenes generated with Unreal Engine 5 (UE5) [6]. Independently ofthe proposed model, we introduce CamEditor-Bench, a model-agnostic evaluation suite consisting of 462 strictly isolated test instances. Crucially, alongside this benchmark, we pioneer a novel Camera Alignment evaluation protocol. By leveraging Perspective Field representations, this metric system efectively avoids traditional estimation noise, allowing for an explicit and rigorous assessment of absolute geometric precision. Extensive experiments on CamEditor-Bench demonstrate that CameraEditor achieves state-of-the-art performance in both geometric alignment and content preservation. By reformulating camera control as a temporal video generation task, our visual reference pairs eliminate text-based misinterpretations, while the frame-byframe decomposition efectively bufers extreme pixel displacements to prevent structural collapse. In summary, CameraEditor and CamEditor-Bench constitute two independent contributions, supported by comprehensive evaluation:

• We present CameraEditor, which reformulates cameracontrolled editing into a temporal sequence prediction task. By integrating dynamic reference routing to construct rigorous visual prompts and inserting intermediate transition frames, it circumvents the inherent ambiguity of text-based instruction control and bufers massive perspective shifts to efectively prevent structural collapse.

• We construct a 5,760-instance training dataset for CameraEditor. Separately, we introduce CamEditor-Bench, a model-agnostic evaluation suite of 462 perspective-image pairs derived from diverse real-world and UE5 panoramas, complemented by an optimization-free geometric evaluation protocol based on Perspective Fields.

• Automatic and human evaluations with same-backbone controls show that the gains arise from explicit geometric conditioning and sequential visual references rather than backbone capacity alone.

## 2 Related Work

## 2.1 Image Editing

Image editing has advanced with instruction-driven approaches [4], where models are fine-tuned on large “instruction-response” datasets for intuitive manipulation [2]. These methods evolved from fine-tuning difusion models [40] to flexible editing techniques that exploit pre-trained generative priors [42, 57]. Recent training-free approaches further avoid task-specific adaptation. For instance, GIDE [67] combines grounding with discrete noise inversion for difusion large language models, enabling localized edits while preserving unedited regions. Nevertheless, such methods primarily target semantic or region-based content manipulation rather than explicit camera-parameterized geometric transformations. Instruction-driven editors have also refined datasets and optimized architectures [65] to enhance control and realism. Recent advances improve task eficiency through dataset and model opti mizations. Lightweight methods like LoRA [13] enable task-specific adaptation without large retraining, though task parameters remain necessary. Furthermore, the integration of novel proprietary systems within multimodal LLMs (e.g., GPT-4o [16] and Gemini [17]) has further blurred the boundary between conversational dialog and visual editing, while commercial platforms like Midjourney [47] and RunwayML [7] embed these models into end-to-end creative workflows.

Beyond pure text or instruction guidance, precise image manipulation fundamentally requires robust editing under diverse structural conditions. To resolve the spatial ambiguity of natural language, conditional frameworks incorporate auxiliary modalities to rigidly constrain the difusion process. Methods like ControlNet [60] and T2I-Adapter [32] utilize depth maps, semantic segmentations, and Canny edges to dictate fine-grained geometric structures, whereas layout-guided models [23] enforce strict bounding-box object placement to maintain compositional integrity. However, despite these significant strides in handling diverse 2D conditional inputs, existing architectures often struggle when the condition involves explicit extrinsic camera parameters or 3D-aware novel view synthesis [37]. While planar conditions successfully dictate 2D layouts, they inherently lack the underlying geometric priors required to maintain view-dependent structural consistency and object permanence during complex camera trajectory transformations, exposing a critical architectural void. However, current mod els struggle with fine-grained edits requiring precise camera control, such as rotation angles and Perspective Field, limiting their use in tasks like cinematography.

## 2.2 Visual Camera Geometry

Camera parameter estimation and geometric control are foundational to 3D computer vision [36]. Classical methods rely on multiview constraints and projective geometry for rigorous calibration and structure-from-motion [10, 43]. Deep learning later shifted toward directly regressing camera poses from raw pixels [12, 19, 53]. However, mapping high-dimensional images to rigid ��(3) manifolds is highly non-linear, making direct regression ill-posed and prone to poor generalization [45].

To bridge parameter estimation and visual synthesis, methodologies introduced intermediate representations like displacement fields [24, 68]. Concurrently, implicit neural representations [31] and difusion-based novel view synthesis (NVS) [27, 44] manipulated viewpoints using relative coordinates. Yet, these NVS frameworks are predominantly object-centric, failing to execute precise adjustments of camera extrinsics and intrinsics in complex, opendomain scenes, which severely limits their practical applicability.

Recently, controllable difusion models incorporated explicit camera viewpoint modeling. Frameworks like CameraCtrl [11] and MotionCtrl [52] inject camera matrices to dictate trajectories during generation, while others explore spatial conditioning for text-toimage synthesis [66]. Despite these advancements, their applicability is strictly restricted to the forward generation phase. For instance, while PreciseCam [1] achieves accurate camera control during generation, it lacks an inversion mechanism to re-render the latent geometry of an existing input. Consequently, manipulating exact camera parameters on pre-existing images without inducing severe structural degradation remains an unresolved bottleneck in post-capture editing scenarios.

## 3 Methodology

This section details CameraEditor, a framework that reformulates camera-controlled image editing into a sequence prediction task (Figure 3). The generation process is driven by a video generative backbone (Sec. 3.2) and conditioned on explicit reference visual sequences (Sec. 3.3). To efectively bridge large perspective shifts, we introduce a Chain of Frames (CoF) strategy to decompose complex geometric transformations (Sec. 3.4). Finally, a dynamic routing mechanism is employed during inference to ensure strict viewpoint alignment without retraining (Sec. 3.5).

## 3.1 Task Definition and Scope

We represent a camera configuration as

$$
\begin{array} { r } { \dot { p } = ( \mathrm { y a w } , \mathrm { r o l l } , \mathrm { p i t c h } , \mathrm { v F o V } , \xi ) , } \end{array}
$$

where $\xi$ denotes radial distortion. Given a source image $I _ { \mathrm { s r c } }$ with configuration $\ p _ { \mathrm { b a s e } }$ and a bounded target configuration $\mathit { p } _ { \mathrm { e d i t e d } } ,$ , the goal is to generate $I _ { \mathrm { e d i t e d } }$ whose projected geometry follows �<sub>edited</sub> while preserving the underlying scene content and identity. CameraEditor controls roll, pitch, vFoV, and radial distortion while holding yaw fixed. The projected image layout changes according to the target camera, but the edit does not request a new scene or a diferent subject. Constraining the parameter shift keeps the task within camera-parameterized editing rather than unconstrained outpainting; the exact sampling bounds are given in the Supplementary Material.

![](images/0943b7cd6f50028b6bcfef47bb5cf154c267bdc02b99acfaa3246a2ee825c9da.jpg)  
Figure 3: Overview of CameraEditor. (a) Generation Pipeline: A video difusion model generates the target Chain of Frames (CoF) sequence conditioned on the source image and reference visual sequence. (b) Dynamic Routing: A routing mechanism selects the optimal reference prior via GeoCalib during inference for precise viewpoint alignment.

## 3.2 Video Generative Models for Image Editing

Video generative models have shown promise in image editing tasks by formulating them as frame-to-frame generation problems [20, 29, 30]. In this paradigm, given a source image $I _ { s r c }$ and desired modifications, the model generates an edited image $I _ { t g t }$ while maintaining content consistency across the frames. Specifically, the model focuses on ensuring that the high-level structure and seman tics of the original image remain intact, while modifying the details according to the desired target. To achieve this, the model encodes both the source and target images into a shared latent space via a VAE encoder [46], which results in the latent representations $Z _ { s r c }$ and $Z _ { t g t }$ , respectively.

During inference, starting from pure noise $X _ { 0 } \sim { \cal N } ( 0 , 1 )$ , noise is injected at each step to form $X _ { t }$ , which is then concatenated with the source latent as $S _ { t } = [ Z _ { s r c } , X _ { t } ]$ . The model $v _ { \theta }$ iteratively denoises this sequence to predict the clean latent $X _ { 1 }$ of the target frame. At each step, the condition token may become slightly polluted, represented as $Z _ { s r c } ^ { t + \delta t }$ , requiring replacement with the original $Z _ { s r c }$ to ensure consistency and preserve the source image content.

During training, we utilize a flow-matching loss [26] to optimize our model. Specifically, we define the clean target latent as the data endpoint, $X _ { 1 } \equiv Z _ { t q t }$ , and sample pure noise $X _ { 0 } \sim { \cal N } ( 0 , 1 )$ . The intermediate noisy state $X _ { t }$ is constructed via linear interpolation:

$$
X _ { t } = ( 1 - t ) X _ { 0 } + t X _ { 1 } ,
$$

where the timestep � is uniformly sampled from $U ( 0 , 1 )$ . The model is trained to predict the velocity field $\begin{array} { r } { V _ { t } = \frac { d X _ { t } } { d t } = X _ { 1 } - X _ { 0 } , } \end{array}$ , which governs the trajectory towards the clean latent, allowing the estimation of $X _ { t + \delta t }$ during inference. The optimization objective is formulated as:

$$
\mathcal { L } = \mathbb { E } _ { t , X _ { 0 } , X _ { 1 } } \left[ \Vert v _ { \theta } ( S _ { t } , t _ { t g t } ) - V _ { t } \Vert _ { 2 } ^ { 2 } \right] ,
$$

where $v _ { \theta } ( S _ { t } , t ) _ { t g t }$ denotes the predicted velocity for the target frame.

## 3.3 Reference Visual Sequence as Prompt

Prior work typically implemented camera-controlled editing by injecting control signals such as textual descriptions of camera parameters or task-specific attributes like PF-US maps into the model. While these methods ofer a certain level of control, they are often hindered by inherent limitations. For instance, textual descriptions often sufer from misalignment between language and scene geometry, making it dificult for the model to accurately translate abstract language into precise geometric transformations. Although PF-US maps efectively control pixel-level details, they fail to maintain the image’s overall structure, often causing distortions such as warping or misalignment that compromise spatial coherence and visual integrity.

To overcome these ambiguities, we leverage the temporal attention of video generative models by introducing a reference visual sequence as a geometric prompt. Instead of relying on high-level abstract text descriptions, we provide a reference pair $( I _ { \mathrm { r e f } } ^ { \mathrm { b a s e } } , I _ { \mathrm { r e f } } ^ { \mathrm { e d i t e d } } )$ that explicitly demonstrates the desired camera parameter variation. Crucially, this reference pair undergoes the exact same camera configuration shift $( p _ { \mathrm { b a s e } }  p _ { \mathrm { e d i t e d } } )$ as the user’s target image pair $( I _ { \mathrm { t a r } } ^ { \mathrm { b a s e } } , I _ { \mathrm { t a r } } ^ { \mathrm { e d i t e d } } )$ . By concatenating the reference sequence ahead of the target sequence along the temporal dimension, the model’s temporal attention mechanisms first internalize the geometric shift from the reference prior. Guided by this strict temporal prompt, the model seamlessly transfers the learned transformation to synthesize the exact $I _ { \mathrm { t a r } } ^ { \mathrm { e d i t e d } }$ from the input $I _ { \mathrm { t a r } } ^ { \mathrm { b a s e } }$ , efectively bypassing the spatial ambiguities of traditional text-based editing instructions.

## 3.4 CoF-based Intermediate Frames

Large camera-parameter shifts $( p _ { \mathrm { b a s e } }  p _ { \mathrm { e d i t e d } } )$ cause geometric distortions and content inconsistencies when generating targets directly from the source. Difusion models struggle to simultaneously resolve complex spatial transformations and preserve scene identity in a single generative step.

To address this, we introduce a Chain of Frames (CoF) strategy that decomposes the large camera transformation into a sequence of manageable steps for both the reference and target domains. Formally, instead of directly predicting the final edited target from the source $I _ { \mathrm { t a r } } ^ { b a s \epsilon }$ , we formulate the generation as predicting a temporally coherent sequence $S _ { \mathrm { t a r } } = \{ I _ { \mathrm { t a r } } ^ { ( i ) } \} _ { i = 1 } ^ { N + 1 }$ , where $I _ { \mathrm { t a r } } ^ { ( N + 1 ) } \equiv I _ { \mathrm { t a r } } ^ { \mathrm { e d i t e d } }$ Notably, this progression strictly aligns with the reference sequence $S _ { \mathrm { r e f } }$ . We uniformly interpolate the camera parameters, assigning shared intermediate poses $\boldsymbol { p } ^ { ( i ) }$ to both $I _ { \mathrm { r e f } } ^ { ( i ) }$ and $I _ { \mathrm { t a r } } ^ { ( i ) }$ at each step �.

The video difusion model’s temporal modeling capabilities are well-suited for processing these frame sequences. The 3D attention mechanisms capture correlations across frames, learning to predict each frame by attending to temporal neighbors and conditioning information. This temporal coherence, combined with gradual camera-parameter changes, enables smooth and consistent transformations while preserving scene content.

## 3.5 Inference-Time Viewpoint Alignment

While the generation mechanisms detailed above establish our core framework, a critical challenge remains: during inference, the model must accurately align the input image’s viewpoint with the reference pair to ensure that the learned transformations are correctly applied. To achieve this, we implement a two-stage alignment process (Figure 3(b)). First, we utilize a geometric calibration module (GeoCalib [48]) to estimate the camera parameters of the input image. This module is trained to predict the camera parameters directly from the image, providing a mathematically precise viewpoint estimation. Second, we introduce a Dynamic Reference Routing mechanism to obtain a high-quality reference image that closely aligns with the estimated camera configuration. We dynamically crop a continuous reference sequence from a pre-defined candidate pool based on the estimated and target parameters. A Vision Language Model scores each candidate along five axes—geometric correlation, information balance, scene richness, image clarity, and content coherence—and the minimum axis score determines its overall score. We retain the highest-ranked candidate, ensuring that a weak criterion cannot be hidden by strong scores on the remaining axes. The resulting reference sequence demonstrates the requested transformation without sharing scene identity with the user input. Details of the candidate pools, cropping procedure, and rating prompt are provided in Section 4.1 and the Supplementary Material.

## 4 Dataset and Evaluation Benchmark

This section details our data generation and benchmarking protocol. We first construct a specialized training dataset of 5,760 panoramic instances (Sec. 4.1). Next, we introduce CamEditor-Bench for rigorous zero-shot evaluation (Sec. 4.2), and finally define robust, optimization-free geometric metrics (Sec. 4.3).

## 4.1 Training Dataset Construction

Since sequential data with continuous camera transitions is absent in current editing domains, we construct an automated pipeline (Figure 4) to generate 5,760 training instances in four stages.

(1) Various panoramic images collection: We collect highquality panoramic images from real-world datasets such as 360-SoD [22], F360-SoD [35] and CVRG-Pano [63], and synthetic scenes rendered using Unreal Engine 5 (UE5). This strategic combination of real-world photorealism and precisely controlled synthetic environments ensures our compact dataset covers a comprehensive blend of diverse geometric structures across indoor and outdoor settings, ultimately establishing a panorama pool for subsequent selection.

(2) Image Selection & Parameter Cropping: To construct each training instance, we first select a reference panorama and a target panorama from our panorama pool. We then model cameras via Perspective Fields (yaw, pitch, roll, vFov, and radial distortion �). Edited parameters $\scriptstyle { \mathcal { P } } \mathrm { e d i t e d }$ are randomly sampled, while base parameters $\ p _ { \mathrm { b a s e } }$ are tightly constrained to prevent unconstrained outpainting. Both selected panoramas are then cropped using these identical parameter pairs to yield precisely aligned reference-target image pairs for the subsequent evaluation step.

(3) Super-resolution and Filtering: To mitigate the resolution degradation introduced during cropping, we deploy the OSEDif [56] and RAM [50] models to refine fine-grained details. Subsequently, we employ a Vision-Language Model (VLM) to evaluate each image pair across five criteria: geometric correlation, information balance, scene complexity, image clarity, and content coherence. The final score is strictly capped by the lowest sub-score. Detailed scoring guidelines and the full VLM prompt are available in the Supplementary Material.

(4) Intermediate Sequence Generation: To decompose large camera transformations into manageable steps, we generate � intermediate frames for the target sequence. We achieve this by linearly interpolating the camera parameters between the input and output viewpoints to ensure temporal consistency.

## 4.2 CamEditor-Bench Construction

To evaluate the generalization capability and robustness of cameracontrolled editing, we introduce CamEditor-Bench. The primary objective of this benchmark is to provide a comprehensive evaluation suite spanning diverse semantic content, a broad spectrum of parameter variations, and strict data isolation.

Scene and Domain Diversity. To guarantee a rigorous evaluation of zero-shot generalization, CamEditor-Bench is curated to encompass a comprehensive semantic taxonomy while maintaining strict isolation from the training data. The evaluation panoramas are sampled from held-out splits and unseen environments across four domains: (1) Complex Indoor Environments, featuring intricate ofice layouts and building interiors; (2) Urban and Street Views, capturing dense cityscapes and first-person driving perspectives; (3) Natural Landscapes, encompassing unstructured outdoor scenery; and (4) Dynamic Human Activities, including extreme sports and crowds. This selection ensures the model is evaluated on novel geometric structures without scene-level data leakage.

Input Form and Data Isolation. Although panoramas provide the ground-truth geometry needed to construct paired views, every benchmark input is an ordinary perspective image cropped from a real-world or UE5 panorama. CameraEditor therefore accepts a single perspective photograph at inference time and does not require a panoramic user input. The training set (5,760 instances), CamEditor-Bench (462 held-out instances), and the inference-time reference panorama pool are mutually disjoint. The reference pool contains transformation exemplars rather than test-scene content and covers all benchmark cases without scene-level overlap.

![](images/63e74c505ac9ef9a293912aba556de21280b88ca903eda80e10904d6b2626164.jpg)  
Figure 4: The dataset generation pipeline. Panoramas are parameterized and cropped into reference-target pairs. Following super-resolution and VLM filtering, intermediate frames are generated to form continuous temporal sequences.

Comprehensive Parameter Coverage. We crop evaluation pairs from these base panoramas by sampling base and target viewpoint combinations that cover diverse structural perspective shifts and lens distortions, thereby reflecting unconstrained user inputs. Note that yaw is intentionally excluded from the evaluation; large yaw variations require extensive out-of-view content generation, shifting the task from controlled editing to outpainting.

## 4.3 Metrics

To comprehensively evaluate the performance of CameraEditor on CamEditor-Bench, we establish a robust evaluation protocol divided into two primary dimensions: Image Quality & Content Consistency, and Camera Geometric Accuracy.

Image Quality & Content Consistency. We measure low-level structural and perceptual fidelity using SSIM [51] and LPIPS [61]. To ensure the strict preservation of high-level semantic identity and 3D physical layout under extreme viewpoint shifts, we compute the cosine similarities of deep features extracted by CLIP (CLIP-I [38]) and DINO-v2 [34], respectively. Camera Geometric Accuracy. Instead of estimating camera parameters separately, which typically requires non-linear optimization and introduces severe estimation bias, we propose an optimization-free evaluation approach based on the Perspective Field (PF). Specifically, we directly compute the dense representations: the upward vector field and the latitude map. We assess the geometric alignment through two distinct perspectives:

• Alignment with GT Image (PF-I): We compare the gen erated PF against the PF extracted from the ground-truth image to evaluate spatial structural preservation.

• Alignment with Target Conditions (PF-T): We compare the generated PF directly against the user-provided target camera conditions in order to accurately measure instruction-following precision.

For both perspectives, we quantify directional alignment via the cosine similarity of upward vectors $( S _ { u p } ^ { I } , S _ { u p } ^ { T } )$ and absolute positioning via the Mean Squared Error of latitude values $( E _ { l a t } ^ { I } , E _ { l a t } ^ { T } )$

## 5 Experiments

After detailing our experimental setup, we compare against stateof-the-art baselines in Section 5.2 to answer RQ1: Can temporal coherence priors efectively break the trade-of between precise camera control and content preservation? Section 5.4 evaluates perception-module robustness to answer RQ2: How does initial camera perception accuracy afect final geometric alignment and visual quality? Finally, Section 5.5 answers RQ3: How does the number of intermediate transition frames afect spatial coherence under camera transformations?

## 5.1 Experimental Setup

CameraEditor uses the Wan2.1-T2V-14B video difusion model [49]. For the training data, we construct a specialized subset comprising 5,760 instances. All input images and generated sequences are processed at a spatial resolution of 512×512. During fine-tuning, we inject LoRA adapters with a rank of 32 into the attention blocks. The model is optimized using AdamW (weight decay = 0.01) with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 4. During inference, based on our ablation findings, we set the number of intermediate transition frames to � = 8. Consequently, the model generates a target sequence of exactly 9 frames (8 intermediate frames and 1 final target frame).

To establish a comprehensive and rigorous baseline comparison, we evaluate CameraEditor against six open-source models (ICEdit [64], Step1X-Edit [28], OmniGen2 [55], Flux.2 [3], Qwen-Image-Editing [54], and HunyuanImage-3.0-Instruct [5]) and two closed-source proprietary APIs (GPT-Image-1.5 [33] and Nano-Banana-Pro [39]). To accommodate the baseline models’ reliance on natural-language instructions, we construct a standardized deterministic prompt-conversion pipeline. The exact mappings and baseline-selection rationale are provided in the Supplementary Material.

![](images/0641dd6bef1d1771de46a481c7ebbe421859e7de1508f998b55924b60e7f9820.jpg)  
Figure 5: Qualitative results alongside geometric error maps. For each method, the right panels display PF-I (top) and PF-T (bottom) errors, where warmer colors indicate larger deviations.

Runtime. On a single H200 GPU, GeoCalib requires approximately 68 ms, dynamic reference routing approximately 7 s, and 9-frame generation approximately 46 s, for a total latency of ap proximately 53 s per input. The generation stage dominates the runtime. Relative to random routing, the 7 s routing stage improves DINO-v2 from 0.6301 to 0.8569 and reduces $E _ { l a t } ^ { T }$ from 0.3712 to 0.1904 (Table 3).

## 5.2 Main Results (RQ1)

Evaluated under this rigorous framework, CameraEditor achieves state-of-the-art performance across all metrics, efectively answering RQ1 by breaking the trade-of between precise geometric control and content preservation (Table 1). Compared to baselines, closed-source models preserve global aesthetics but struggle with strict 3D physical constraints, prioritizing stylistic priors over mathematical geometry. In contrast, instruction-driven open-source models reveal a clear performance hierarchy governed by model capacity. Small-scale models like OmniGen-v2 and ICEdit exhibit severe source identity loss; their limited representational bandwidth forces a destructive trade-of, rendering them unable to simultaneously decode semantics and compute large spatial trans formations. Conversely, large-scale models (Flux.2, Qwen-Image, HunyuanImage-3.0) leverage richer pre-trained priors to mitigate structural collapse. However, their reliance on implicit textual guidance rather than explicit mathematical constraints fundamentally caps their absolute geometric precision. Among them, Qwen-Image achieves the most balanced empirical results (CLIP-I: 0.8723, $S _ { u p } ^ { I } { : }$ 0.8923, $S _ { u p } ^ { T } { : }$ 0.5354, and $E _ { l a t } ^ { T } \colon 0 . 2 5 4 9 )$ , yet remains below CameraEditor on both content preservation and target-camera alignment.

It is worth noting two specific metric divergences observed in our evaluation. First, most models maintain acceptable high-level semantic identity (DINO-v2 and CLIP-I) but show lower scores in low-level fidelity (SSIM and LPIPS). This is because massive perspective shifts inherently induce drastic, non-linear pixel displacements. Strictly aligned metrics like SSIM and LPIPS heavily penalize these natural structural deviations, even when the generated results are geometrically plausible and highly realistic. Second, relative geometric alignment with the ground truth $( S _ { u p } ^ { I } , E _ { l a t } ^ { I } )$ consistently yields better scores than absolute alignment with the target camera parameters $( S _ { u p } ^ { T } , E _ { l a t } ^ { T } )$ . This numerical gap is driven by the systematic estimation deviation of current geometry-perception networks. Calculating relative alignment evaluates both generated and ground-truth images through the same network, efectively mitigating inherent domain biases, whereas absolute alignment strictly compares extracted features against pure mathematical conditions, exposing the perception network’s intrinsic errors.

## 5.3 Same-Backbone Controls and Human Evaluation

To isolate the efect of conditioning from backbone capacity, we train two control variants with the same Wan2.1-T2V-14B backbone, training data, and hyperparameters as CameraEditor. The no-reference variant removes the visual reference sequence and supplies the target camera through the same deterministic prompt template used by the instruction-driven baselines. The Plücker variant replaces the visual reference with explicit Plücker camera embeddings derived from GeoCalib. CoF is retained in both variants. The controls are integrated directly into the main comparison (Table 1). Moving from no reference to Plücker conditioning improves DINO-v2 from 0.6951 to 0.7617 and $S _ { u p } ^ { I }$ from 0.8837 to 0.9186; the full visual-reference formulation further reaches 0.8569 and 0.9267, respectively. These controlled results demonstrate that the gain is not attributable to backbone capacity alone.

Table 1: Quantitative comparison on CamEditor-Bench. Evaluation of image quality, content preservation, and camera geometric alignment. The two Wan2.1 controls and CameraEditor use the same Wan2.1-T2V-14B backbone, training data, and hyperparameters. ↑ indicates higher is better, and ↓ indicates lower is better. Best results are in bold, and second-best results are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="4">Content Preservation &amp; Quality</td><td colspan="4">Camera Alignment</td></tr><tr><td>DINO-v2 (↑)</td><td>CLIP-I (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td><td> $S _ { u p } ^ { I }$  (↑)</td><td> $E _ { l a t } ^ { I }$  (↓)</td><td> $S _ { u p } ^ { T }$  (↑)</td><td> $E _ { l a t } ^ { T } \left( \downarrow \right)$ </td></tr><tr><td>ICEdit</td><td>0.7089</td><td>0.8009</td><td>0.3740</td><td>0.7161</td><td>0.8717</td><td>0.2032</td><td>0.5129</td><td>0.2928</td></tr><tr><td>Step1X-Edit</td><td>0.7675</td><td>0.8502</td><td>0.4141</td><td>0.6838</td><td>0.8680</td><td>0.2061</td><td>0.5168</td><td>0.2972</td></tr><tr><td>OmniGen2</td><td>0.6616</td><td>0.8041</td><td>0.3630</td><td>0.6774</td><td>0.8749</td><td>0.2029</td><td>0.5120</td><td>0.2937</td></tr><tr><td>Flux.2</td><td>0.7604</td><td>0.8567</td><td>0.4173</td><td>0.6397</td><td>0.8806</td><td>0.1843</td><td>0.5295</td><td>0.2710</td></tr><tr><td>Qwen-Image-Editing</td><td>0.7749</td><td>0.8723</td><td>0.4428</td><td>0.6345</td><td>0.8923</td><td>0.1712</td><td>0.5354</td><td>0.2549</td></tr><tr><td>HunyuanImage-3.0-Instruct</td><td>0.7762</td><td>0.8417</td><td>0.3991</td><td>0.6551</td><td>0.8695</td><td>0.1910</td><td>0.5200</td><td>0.2816</td></tr><tr><td>GPT-Image-1.5</td><td>0.7703</td><td>0.8416</td><td>0.4469</td><td>0.6460</td><td>0.8524</td><td>0.2039</td><td>0.5153</td><td>0.2801</td></tr><tr><td>Nano-Banana-Pro</td><td>0.7769</td><td>0.8642</td><td>0.4543</td><td>0.6440</td><td>0.8566</td><td>0.2136</td><td>0.5159</td><td>0.3020</td></tr><tr><td colspan="9">Same-backbone controls (Wan2.1-T2V-14B)</td></tr><tr><td>Wan2.1, no reference</td><td>0.6951</td><td>0.8235</td><td>0.4966</td><td>0.6380</td><td>0.8837</td><td>0.2091</td><td>0.5359</td><td>0.2802</td></tr><tr><td>Wan2.1 + Plücker</td><td>0.7617</td><td>0.8652</td><td>0.5139</td><td>0.6340</td><td>0.9186</td><td>0.1751</td><td>0.5536</td><td>0.2575</td></tr><tr><td>CameraEditor (Ours)</td><td>0.8569</td><td>0.8970</td><td>0.5712</td><td>0.5454</td><td>0.9267</td><td>0.0917</td><td>0.6089</td><td>0.1904</td></tr></table>

Table 2: Blind pairwise human evaluation. Preference counts a tie as 0.5 for each method. Each row contains 100 comparisons.
<table><tr><td>Baseline</td><td>Base</td><td>Tie</td><td>Ours</td><td>Pref.</td></tr><tr><td>Step1X-Edit</td><td>11</td><td>14</td><td>75</td><td>82.0%</td></tr><tr><td>Qwen-Image-Editing</td><td>25</td><td>23</td><td>52</td><td>63.5%</td></tr><tr><td>Nano-Banana-Pro</td><td>19</td><td>23</td><td>58</td><td>69.5%</td></tr></table>

We further conduct 300 blind pairwise human comparisons against Step1X-Edit, Qwen-Image-Editing, and Nano-Banana-Pro (100 comparisons per baseline). Each comparison presents the source image, ground truth, target-camera visualization, and two anonymous outputs. Evaluators select the better edit based jointly on perceived camera correctness, content preservation, and artifact severity. As shown in Table 2, CameraEditor is preferred in 82.0%, 63.5%, and 69.5% of the comparisons, respectively, when each tie contributes 0.5 to both methods. The human ranking is consistent with the automatic camera-alignment and content-preservation metrics.

## 5.4 Ablation Study on Perception Module (RQ2)

To answer RQ2, we evaluate the impact of the initial camera perception module on final editing quality. Reference sequences are analytically cropped under known camera parameters; routing errors arise from estimating the input configuration and matching it to the candidate pool. We therefore evaluate routing quality through its efect on the generated final frame. More accurate initial perception yields a reference sequence that more closely corresponds to the target trajectory, improving viewpoint consistency across frames.

Detailed parameter-estimation errors and visual comparisons are provided in the Supplementary Material.

As reported in Table 3, random routing yields the poorest performance. Without an accurate geometric anchor, the reference sequence fails to correspond with the target, destroying frame-toframe viewpoint consistency and causing spatial collapse across the temporal sequence $( E _ { l a t } ^ { I }$ : 0.2654, DINO-v2: 0.6301).

We then evaluate Pufin [25] (both base and thinking variants), which represents VLM-based camera understanding. While Puffin provides a rough sequence correspondence, it struggles to accurately regress rigorous mathematical parameters, particularly non-linear optical distortions. Consequently, the reference and target frames sufer from perceptual deviation. This misalignment inherently caps the absolute geometric accuracy (e.g., Pufin thinking achieves an $S _ { u p } ^ { T }$ of only 0.5652) and degrades source content preservation (DINO-v2: 0.7478).

In contrast, our pipeline leverages GeoCalib to estimate the camera configuration and retrieve a more closely aligned reference sequence. As shown in Table 3, this sequence-level alignment reduces relative geometric errors and yields the highest absolute alignment and content-preservation scores among the evaluated routing strategies.

## 5.5 Ablation Study on Intermediate Frame Counts (RQ3)

To address RQ3, we evaluate the impact of intermediate transition frame counts $( N \in \{ 0 , 4 , 8 , 1 2 \} )$ on editing performance (Table 4). All variants use the same Wan2.1-T2V-14B backbone, training data, and conditioning pipeline; consequently, $N = 0$ is the requested single-step same-backbone control. The resulting total sequence lengths $( N + 1 ~ \in ~ \{ 1 , 5 , 9 , 1 3 \} )$ are strictly chosen to satisfy the 4� + 1 temporal constraint of the Wan2.1 3D-VAE backbone. This architectural alignment obviates the need for temporal padding during encoding, inherently preventing latent misalignment and subsequent interpolation artifacts.

Table 3: Ablation study on Perception and Routing Module. Comparison of diferent reference routing strategies. Relying on our dedicated perception module (GeoCalib) provides the most accurate geometric anchor and best visual quality.
<table><tr><td rowspan="2">Routing Strategy</td><td colspan="4">Content Preservation &amp; Quality</td><td colspan="4">Camera Alignment</td></tr><tr><td>DINO-v2 (↑)</td><td>CLIP-I (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td><td> $S _ { u p } ^ { I } \left( \uparrow \right)$ </td><td> $E _ { l a t } ^ { I } \left( \downarrow \right)$ </td><td> $S _ { u p } ^ { T } \left( \uparrow \right)$ </td><td> $E _ { l a t } ^ { T } \left( \downarrow \right)$ </td></tr><tr><td>Random</td><td>0.6301</td><td>0.8083</td><td>0.5231</td><td>0.6281</td><td>0.8419</td><td>0.2654</td><td>0.4961</td><td>0.3712</td></tr><tr><td>Puffin base</td><td>0.7413</td><td>0.8486</td><td>0.5360</td><td>0.6009</td><td>0.8936</td><td>0.1548</td><td>0.5752</td><td>0.2329</td></tr><tr><td>Puffin thinking</td><td>0.7478</td><td>0.8508</td><td>0.5454</td><td>0.5993</td><td>0.9049</td><td>0.1523</td><td>0.5652</td><td>0.2439</td></tr><tr><td>GeoCalib [48]</td><td>0.8569</td><td>0.8970</td><td>0.5712</td><td>0.5454</td><td>0.9267</td><td>0.0917</td><td>0.6089</td><td>0.1904</td></tr></table>

Table 4: Ablation study on the Number of Intermediate Frames (N). The impact of intermediate transition frames on spatial coherence, content preservation, and camera alignment.
<table><tr><td rowspan="2">Sequence Length</td><td colspan="4">Content Preservation &amp; Quality</td><td colspan="4">Camera Alignment</td></tr><tr><td>DINO-v2 (↑)</td><td>CLIP-I (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td><td> $S _ { u p } ^ { I } \left( \uparrow \right)$ </td><td> $E _ { l a t } ^ { I } \left( \downarrow \right)$ </td><td> $S _ { u p } ^ { T } \left( \uparrow \right)$ </td><td> $E _ { l a t } ^ { T } \left( \downarrow \right)$ </td></tr><tr><td>N=0</td><td>0.8435</td><td>0.8686</td><td>0.5181</td><td>0.5891</td><td>0.9010</td><td>0.1343</td><td>0.6242</td><td>0.1669</td></tr><tr><td>N=4</td><td>0.8128</td><td>0.8766</td><td>0.5559</td><td>0.5645</td><td>0.9223</td><td>0.1184</td><td>0.6231</td><td>0.1974</td></tr><tr><td>N=8</td><td>0.8569</td><td>0.8970</td><td>0.5712</td><td>0.5454</td><td>0.9267</td><td>0.0917</td><td>0.6089</td><td>0.1904</td></tr><tr><td>N=12</td><td>0.8179</td><td>0.8781</td><td>0.5604</td><td>0.5925</td><td>0.9286</td><td>0.1110</td><td>0.5824</td><td>0.2476</td></tr></table>

Results reveal a strict trade-of between absolute geometric adherence and structural integrity. Under a constrained temporal window (� = 0), tight cross-frame coupling forces aggressive latent warping. This yields the highest absolute geometric alignment $( S _ { u p } ^ { T } : 0 . { \bar { 6 2 4 2 } } , E _ { l a t } ^ { T } : 0$ .1669) but induces severe structural tearing due to massive perspective shifts in limited steps, resulting in the poorest low-level fidelity (SSIM: 0.5181, LPIPS: 0.5891).

Conversely, increasing � mitigates this tearing by decomposing spatial transformations into a temporal bufer. However, excessive extension (e.g., � = 12) introduces temporal drift. While smoother transitions maintain high relative structural consistency $( S _ { u p } ^ { I } ) _ { : }$ , the prolonged iterative denoising process accumulates latent noise. This progressive dilution weakens the mathematical anchors at the endpoints, reducing absolute alignment $( S _ { u p } ^ { T } : 0 . 5 8 2 4 , E _ { l a t } ^ { T } : 0 . 2 4 7 6 )$ and degrading content preservation (DINO-v2: 0.8179).

Consequently, � = 8 provides the optimal balance. It ofers temporal bufering to prevent structural tearing—achieving peak visual quality (SSIM: 0.5712, LPIPS: 0.5454) and content fidelity (DINO-v2: 0.8569)—while averting temporal drift to maintain precise relative geometric alignment $( \bar { E } _ { l a t } ^ { I } : \bar { 0 . 0 9 1 7 } )$

## 6 Conclusions and Limitations

We present CameraEditor for camera-parameterized image editing. Our framework formulates the problem as a temporal sequence prediction task, efectively leveraging the temporal coherence priors of video difusion models. By integrating explicit geometric perception with dynamic panorama cropping, our approach establishes robust geometric anchors for complex spatial manipulations. Addi tionally, the strategic insertion of intermediate transition frames efectively decomposes large-scale perspective shifts, mitigating potential structural distortions and bufering initial observation errors. Extensive evaluations on the newly introduced CamEditor-Bench demonstrate that CameraEditor achieves state-of-the-art performance, striking a highly efective balance between precise spatial alignment and source content preservation.

Limitations and Future Work. Despite its efectiveness, our approach has two primary limitations. First, the inherent spatial compression within video 3D-VAEs tends to over-smooth highfrequency details, leading to the degradation of fine-grained textures during the decoding process. Second, the current system operates as a cascaded pipeline—sequentially executing parameter extraction, reference cropping, and video denoising. This decoupled design lacks end-to-end optimization and increases inference latency. To address these issues, future work will explore hybrid latent upscaling for robust texture recovery and a unified, end-to-end architecture for eficient camera-controlled editing.

## Acknowledgments

This work was supported by the New Generation Artificial Intelligence National Science and Technology Major Project (No. 2025ZD0123001), the National Natural Science Foundation of China (Nos. 62272374 and 62192781), the Natural Science Foundation of Shaanxi Province (No. 2024JC-JCQN-62), the State Key Laboratory of Communication Content Cognition under Grant No. A202502, and the Key Research and Development Project in Shaanxi Province (No. 2023GXLH-024).

This work was also supported by the A\*STAR Career Development Fund (Project No. C243512010).

## References

[1] Edurne Bernal-Berdun, Ana Serrano, Belen Masia, Matheus Gadelha, Yannick Hold-Geofroy, Xin Sun, and Diego Gutierrez. 2025. PreciseCam: Precise Camera Control for Text-to-Image Generation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 2724–2733.

[2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. 2023. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf 2, 3 (2023), 8.

[3] Black Forest Labs. 2025. FLUX.2-dev. https://huggingface.co/black-forest-labs/ FLUX.2-dev.

[4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 18392–18402.

[5] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. 2025. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951 (2025).

[6] Yasin AM El-Wajeh, Paul V Hatton, and Nicholas J Lee. 2022. Unreal Engine 5 and immersive surgical training: translating advances in gaming technology into extended-reality surgical simulation training programmes. British Journal of Surgery 109, 5 (2022), 470–471.

[7] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. 2023. Structure and content-guided video synthesis with difusion models. In Proceedings of the IEEE/CVF international conference on computer vision. 7346–7356.

[8] Sara Ghazanfari, Francesco Croce, Nicolas Flammarion, Prashanth Krishnamurthy, Farshad Khorrami, and Siddharth Garg. 2025. Chain-of-frames: Advancing video understanding in multimodal llms via frame-aware reasoning. arXiv preprint arXiv:2506.00318 (2025).

[9] Zhen Han, Chaojie Mao, Zeyinzi Jiang, Yulin Pan, and Jingfeng Zhang. 2025. Stylebooth: Image style editing with multimodal instruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 1947–1957.

[10] Richard Hartley and Andrew Zisserman. 2003. Multiple view geometry in computer vision. Cambridge university press.

[11] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. 2024. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101 (2024).

[12] Yannick Hold-Geofroy, Kalyan Sunkavalli, Jonathan Eisenmann, Matthew Fisher, Emiliano Gambaretto, Sunil Hadap, and Jean-François Lalonde. 2018. A percep tual measure for deep single image camera calibration. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2354–2363.

[13] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. 2024. In-context lora for difusion transformers. arXiv preprint arXiv:2410.23775 (2024).

[14] Wenjing Huang, Shikui Tu, and Lei Xu. 2025. Pfb-dif: Progressive feature blending difusion for text-driven image editing. Neural Networks 181 (2025), 106777.

[15] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Liangliang Cao, and Shifeng Chen. 2025. Difusion model based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

[16] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).

[17] Muhammad Imran and Norah Almusharraf. 2024. Google Gemini as a next generation AI educational tool: a review of emerging educational technology. Smart Learning Environments 11, 1 (2024), 22.

[18] Linyi Jin, Jianming Zhang, Yannick Hold-Geofroy, Oliver Wang, Kevin Blackburn-Matzen, Matthew Sticha, and David F Fouhey. 2023. Perspective fields for single image camera calibration. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17307–17316.

[19] Alex Kendall, Matthew Grimes, and Roberto Cipolla. 2015. Posenet: A convolu tional network for real-time 6-dof camera relocalization. In Proceedings ofthe IEEE international conference on computer vision. 2938–2946.

[20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

[21] Mingkun Lei, Xue Song, Beier Zhu, Hao Wang, and Chi Zhang. 2025. Stylestudio: Text-driven style transfer with selective control of style elements. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 23443–23452.

[22] Jia Li, Jinming Su, Changqun Xia, and Yonghong Tian. 2019. Distortion-Adaptive Salient Object Detection in 360<sup>◦</sup> Omnidirectional Images. IEEE Journal ofSelected Topics in Signal Processing 14, 1 (2019), 38–48.

[23] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. 2023. Gligen: Open-set grounded text-to image generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 22511–22521.

[24] Kang Liao, Chunyu Lin, Yao Zhao, and Mai Xu. 2020. Model-free distortion rectification framework bridged by distortion distribution map. IEEE Transactions on Image Processing 29 (2020), 3707–3718.

[25] Kang Liao, Size Wu, Zhonghua Wu, Linyi Jin, Chao Wang, Yikai Wang, Fei Wang, Wei Li, and Chen Change Loy. 2025. Thinking with Camera: A Unified Multimodal Model for Camera-Centric Understanding and Generation. arXiv preprint arXiv:2510.08673 (2025).

[26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747

(2022).

[27] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision. 9298– 9309.

[28] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. 2025. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761 (2025).

[29] Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, and Mingyu Ding. 2023. Vdt: General-purpose video difusion transformers via mask modeling. arXiv preprint arXiv:2305.13311 (2023).

[30] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. 2024. Latte: Latent difusion transformer for video generation. arXiv preprint arXiv:2401.03048 (2024).

[31] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

[32] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. 2024. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image difusion models. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 38. 4296–4304.

[33] OpenAI. 2025. GPT Image 1.5 Model. https://developers.openai.com/api/docs models/gpt-image-1.5.

[34] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El Nouby, et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).

[35] Semih Orhan and Yalin Bastanlar. 2022. Semantic segmentation of outdoor panoramic images. Signal, Image and Video Processing 16, 3 (2022), 643–650.

[36] Marc Pollefeys, Reinhard Koch, and Luc Van Gool. 1999. Self-calibration and metric reconstruction inspite of varying and unknown intrinsic camera parameters. International journal of computer vision 32, 1 (1999), 7–25.

[37] Xinran Qin, Zhixin Wang, Fan Li, Haoyu Chen, RenJing Pei, WenBo Li, and XiaoChun Cao. 2025. CamEdit: Continuous Camera Parameter Control for Photorealistic Image Editing. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

[39] Naina Raisinghani. 2025. Introducing Nano Banana Pro. https://blog.google/ innovation-and-ai/products/nano-banana-pro/. Google DeepMind Blog. Ac cessed: 2026-04-02.

[40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

[41] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. 2024. Semantic image inversion and editing using rectified stochastic diferential equations. arXiv preprint arXiv:2410.10792 (2024).

[42] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. Dreambooth: Fine tuning text-to-image difusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 22500–22510.

[43] Johannes L Schonberger and Jan-Michael Frahm. 2016. Structure-from-motion revisited. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 4104–4113.

[44] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023. Zero123++: a single image to consistent multi-view difusion base model. arXiv preprint arXiv:2310.15110 (2023).

[45] Samarth Sinha, Jason Y Zhang, Andrea Tagliasacchi, Igor Gilitschenski, and David B Lindell. 2023. Sparsepose: Sparse-view camera pose regression and refinement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21349–21359.

[46] Jakub Tomczak and Max Welling. 2018. VAE with a VampPrior. In International conference on artificial intelligence and statistics. PMLR, 1214–1223.

[47] Ivan M Tsidylo and Chele Esteve Sena. 2023. Artificial intelligence as a methodological innovation in the training of future designers: Midjourney tools. Information Technologies and Learning Tools 97, 5 (2023), 203.

[48] Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys. 2024. Geocalib: Learning single-image calibration with geometric optimization. In European Conference on Computer Vision. Springer, 1–20.

[49] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025).

[50] Hongjuan Wang, Mingrun Wei, Ru Cheng, Yue Yu, and Xingli Zhang. 2022. Residual deep attention mechanism and adaptive reconstruction network for single image super-resolution. Applied Intelligence 52, 5 (2022), 5197–5211.

[51] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.

[52] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

[53] Scott Workman, Connor Greenwell, Menghua Zhai, Ryan Baltenberger, and Nathan Jacobs. 2015. Deepfocal: A method for direct focal length estimation. In 2015 IEEE International Conference on Image Processing (ICIP). IEEE, 1369–1373.

[54] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Shengming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. 2025. Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025).

[55] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. 2025. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871 (2025).

[56] Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. 2024. One-step efective difusion network for real-world image super-resolution. Advances in Neural Information Processing Systems 37 (2024), 92529–92553.

[57] Sihan Xu, Ziqiao Ma, Yidong Huang, Honglak Lee, and Joyce Chai. 2023. Cyclenet: Rethinking cycle consistency in text-guided difusion for image manipulation. Advances in Neural Information Processing Systems 36 (2023), 10359–10384.

[58] Zexuan Yan, Yue Ma, Chang Zou, Wenteng Chen, Qifeng Chen, and Linfeng Zhang. 2025. Eedit: Rethinking the spatial and temporal redundancy for eficient image editing. arXiv preprint arXiv:2503.10270 (2025).

[59] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. 2025. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings ofthe Computer Vision and Pattern Recognition Conference. 26125–26135.

[60] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image difusion models. In Proceedings ofthe IEEE/CVF international conference on computer vision. 3836–3847.

[61] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

[62] Xinjie Zhang, Jintao Guo, Shanshan Zhao, Minghao Fu, Lunhao Duan, Jiakui Hu, Yong Xien Chng, Guo-Hua Wang, Qing-Guo Chen, Zhao Xu, et al. 2025. Unified multimodal understanding and generation models: Advances, challenges, and opportunities. arXiv preprint arXiv:2505.02567 (2025).

[63] Yi Zhang, Lu Zhang, Wassim Hamidouche, and Olivier Deforges. 2020. A fixationbased 360 benchmark dataset for salient object detection. In 2020 IEEE International Conference on Image Processing (ICIP). IEEE, 3458–3462.

[64] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. 2025. Enabling instructional image editing with in-context generation in large scale difusion transformer. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

[65] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. 2025. In-context edit: Enabling instructional image editing with in-context generation in large scale difusion transformer. arXiv preprint arXiv:2504.20690 (2025).

[66] Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. 2024. Cami2v: Camera-controlled image-to-video difusion model. arXiv preprint arXiv:2410.15957 (2024).

[67] Zifeng Zhu, Jiaming Han, Jiaxiang Zhao, Minnan Luo, and Xiangyu Yue. 2026. GIDE: Unlocking Difusion LLMs for Precise Training-Free Image Editing. arXiv preprint arXiv:2603.21176 (2026). doi:10.48550/arXiv.2603.21176

[68] Bingbing Zhuang and Manmohan Chandraker. 2021. Fusing the old with the new: Learning relative camera pose with geometry-guided uncertainty. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 32–42.

## Supplementary Material

This supplementary material provides additional implementation details, extended experimental analysis, and qualitative results to support the main manuscript. The content is organized as follows:

• Automated Pipeline for Dataset Construction (Appendix A): Details the dataset parameter sampling strategy, temporal sequence interpolation, VLM filtering prompts, and comparisons with existing datasets.

• CamEditor-Bench and Evaluation Protocols (Appendix B): Provides the mathematical derivation of the Target Perspective Field (PF-T) and a comprehensive comparison of our benchmark against existing ones.

• Additional Experimental Details (Appendix C): Explains the deterministic instruction prompt conversion for baseline models and discusses the quantitative impact of perception module accuracy.

• Visualization (Appendix D): Presents further qualitative visualization results and comparative generation examples.

• Limitations (Appendix E): Discusses the physical and algorithmic bounds of our framework, including resolution trade-ofs and metric estimation noise.

## A Automated Pipeline for Dataset Construction

## A.1 Training Dataset Comparison and Necessity

To train a model capable of precise camera-parameterized editing via sequential modeling, the training data must simultaneously satisfy three critical criteria: explicit mathematical camera annota tions, continuous temporal transitions, and high content diversity. As summarized in Table S1, existing datasets fail to meet all these requirements simultaneously, necessitating the construction of our specialized dataset.

Specifically, traditional video datasets like RealEstate10K [11] provide rich camera trajectories but lack paired editing images, and their content diversity is strictly limited to indoor and outdoor property layouts. Recent camera-aware datasets, including Pufin-4M [25], PreciseCam[1], and CamEdit[37], introduce physical camera parameters. However, they are inherently designed for single-step generation or static image pairs, lacking the intermediate sequential transitions.

In contrast, our dataset leverages a hybrid source of real-world panoramas and precise Unreal Engine 5 (UE5) synthetic environments, achieving high content diversity across complex indoor layouts, urban streets, natural landscapes, and dynamic human activities. It provides 5,760 high-quality sequences with absolute mathematical mappings (vFoV, radial distortion, roll, pitch, yaw). Crucially, it is the only dataset incorporating Chain of Frames (CoF) sequential transitions. This unique temporal structure empowers our video-prior framework to efectively decompose largemagnitude perspective shifts and prevent structural tearing.

## A.2 Dataset Parameter Sampling Strategy

We model the camera behavior using a unified perspective field representation parameterized by five variables: yaw, pitch, roll, vertical Field of View (vFoV), and radial distortion (�) [18]. The visual impact of each individual parameter is illustrated in Figure S1.

To ensure a comprehensive and balanced distribution of camera parameters across our training dataset, we employ a rigorous stratified grid sampling strategy. The generation pipeline is governed by four core perspective parameters: roll $r \in \ \left[ - 9 0 ^ { \circ } , 9 0 ^ { \circ } \right]$ , pitch $p \in [ - 9 0 ^ { \circ } , 9 0 ^ { \circ } ]$ , vertical Field of View $\left( \mathrm { v F o V } \right) v \in \left[ 3 0 ^ { \circ } , 1 4 0 ^ { \circ } \right]$ , and radial distortion $\xi \in [ 0 , 1 ]$

To cover this high-dimensional continuous space and prevent sampling bias towards common viewpoints, we discretize the parameter ranges into uniformly distributed bins. Specifically, the parameter space is partitioned into 6 bins for roll, 6 bins for pitch, 4 bins for vFoV, and 4 bins for radial distortion. This orthogonal division yields a total of $6 \times 6 \times 4 \times 4 = 5 7 6$ distinct parameter subspace grids.

Within each of the 576 grid combinations, we randomly sample exactly 10 specific parameter sets. This strategic design ensures both macro-level structural diversity across the entire geometric domain and micro-level stochasticity within local parameter neighborhoods, deterministically generating the final 5,760 strictly aligned reference-target training instances utilized in our framework.

## A.3 Parameter Pairing and Sequence Interpolation

For each sampled base configuration $\phi _ { b a s e } ,$ , we generate a corresponding target configuration $\mathit { p } _ { e d i t e d } .$ To ensure the resulting image pairs maintain strict structural correspondence and remain strictly within the bounds of controllable editing rather than unconstrained outpainting, we impose conditional bounds on $\mathop { p _ { e d i t e d } } .$ Specifically, while yaw is strictly locked, pitch and FOV are constrained within a localized operational window relative to the base state:

$$
\begin{array} { r l } & { \left( \mathrm { y a w } _ { e d i t e d } = \mathrm { y a w } _ { b a s e } \right. } \\ & { \left. \mathrm { p i t c h } _ { e d i t e d } \sim U ( \mathrm { p i t c h } _ { b a s e } - 4 0 ^ { \circ } , \mathrm { p i t c h } _ { b a s e } + 4 0 ^ { \circ } ) \right. } \\ & { \left. \mathrm { r o l l } _ { e d i t e d } \sim U ( - 9 0 ^ { \circ } , 9 0 ^ { \circ } ) \right. } \\ & { \left. \mathrm { f o v } _ { e d i t e d } \sim U ( \mathrm { f o v } _ { b a s e } - 4 0 ^ { \circ } , \mathrm { f o v } _ { b a s e } + 4 0 ^ { \circ } ) \right. } \\ & { \left. \xi _ { e d i t e d } \sim U ( 0 . 0 , 1 . 0 ) \right. } \end{array}
$$

Using these constrained parameter pairs, we execute a spherical projection to crop the base and target images directly from the highresolution panoramas, establishing perfectly aligned referencetarget pairs.

To construct the intermediate transition sequence necessary for our Chain of Frames (CoF) strategy, we linearly interpolate the camera parameters between the base and edited states. The parameter configuration $p ^ { i }$ for the �-th intermediate frame (� ∈ $\{ 1 , 2 , \ldots , N \} )$ is mathematically defined as:

$$
\hat { p } ^ { i } = ( 1 - \frac { i } { N + 1 } ) p _ { b a s e } + \frac { i } { N + 1 } p _ { e d i t e d } , i \in \{ 1 , 2 , . . . , N \} .
$$

This interpolation guarantees a mathematically continuous geometric trajectory across the temporal sequence, providing the video difusion model with a coherent structural prior.

Table S1: Comparison of our training dataset with existing related datasets. Our dataset uniquely provides sequential transition frames (CoF) combined with explicit camera parameter pairs and high scene diversity.
<table><tr><td>Dataset</td><td>Task</td><td>Size</td><td>Content Diversity</td><td>Synthetic Content</td><td>Sequential Transitions</td><td>Intrinsics</td><td>Extrinsics</td></tr><tr><td>RealEstate10K [11]</td><td>Video Synthesis</td><td>10M frames</td><td>Low</td><td>No</td><td>No</td><td>vFoV, distortion</td><td>roll, pitch, yaw</td></tr><tr><td>Puffin-4M [25]</td><td>Image Generation</td><td>4M triplets</td><td>Medium-High</td><td>No</td><td>No</td><td>vFoV</td><td>roll, pitch, yaw</td></tr><tr><td>PreciseCam[1]</td><td>T2I Generation</td><td>57,380 imgs</td><td>Medium-High</td><td>No</td><td>No</td><td>vFoV, distortion</td><td>roll, pitch</td></tr><tr><td>CamEdit[37]</td><td>Image Editing</td><td>50K pairs</td><td>High</td><td>Yes</td><td>No</td><td>aperture, focal length</td><td>focal plane</td></tr><tr><td>Ours</td><td>Image Editing</td><td>5760 seqs</td><td>High</td><td>Yes</td><td>Yes (CoF)</td><td>vFoV, radial dist</td><td>roll, pitch, yaw</td></tr></table>

![](images/be9ac05e18dbf448d050d56bf4463dcd49d5a4c79755153c839a13f7a6133786.jpg)  
Figure S1: Visual impact of individual camera parameters on the cropped perspective. We systematically manipulate yaw, pitch, roll, vertical FOV, and radial distortion (�) to demonstrate their distinct geometric transformations on the base panorama.

## A.4 VLM Prompt for Training Data Filtering

As detailed in Section 4.1 (Stage 3) of the main text, we utilize a Vision-Language Model (VLM) as an automated data cleaning mechanism to filter the synthesized training instances.

To ensure the full reproducibility of our data generation pipeline, the evaluation used the open-weights Qwen2.5-VL-7B-Instruct model released by Qwen. The input provided to the model is a single wide image, constructed by horizontally concatenating the unedited base image and its corresponding edited output. To ensure deterministic scoring and strictly eliminate generation randomness, the decoding temperature of the model was set to 0.0.

The VLM acts as a rigorous data annotator, scoring each cropped base-edited pair on a scale of 0 to 10. As defined in our pipeline, only image pairs that maintain a strict minimum score threshold across all five specified dimensions are retained for the final training set. The exact system prompt provided to the model is presented below.

## System Prompt for Training Data Filtration

System Role: You are an expert AI data evaluator, specializing in assessing the quality of image pairs for training a novel view synthesis model. Your task is to analyze a given image and provide a quality score using a deduction system.

The input image is a single wide image, which is a horizontal concatenation of two diferent views of the same 3D scene ("View A" and "View B"). The placement of these views (left or right) is random.

Scoring Instructions:

• The image pair starts with a full score of 10.

• For each criterion below, assign a sub-score between 0 (completely unsatisfactory) and 10 (fully satisfactory), and briefly explain your reasoning.

The final score is the lowest of all sub-scores (not the average). This means that a low score in any single aspect will significantly lower the overall score, ensuring that only images that are strong in all aspects receive a high final score.

• If a criterion is severely violated, give a low sub-score (e.g., 0\~3); if only minor issues exist, give a moderate sub-score (e.g., 4\~7).

• Use the full range of scores to reflect subtle diferences.

## Criteria (score each separately):

1. Correlation and Learnability: Are there clear, corresponding features or objects between the two views, making the geometric transformation learnable? Do not penalize for drastic perspective changes or non-linear distortions ifcorrelation is maintained.

• "Corresponding features or objects" means that most major objects or regions in one view can be clearly matched to those in the other view, even iftheir positions or shapes change due to perspective.

• It is acceptable for a small number of objects to be missing or not fully corresponding due to large viewpoint changes, occlusion, or field-of-view diferences. Do not penalize heavily for such cases as long as the majority of core subjects and the overall spatial structure remain consistent and learnable.

• If most major objects can be matched and the spatial structure is reasonable, give a high score (8-10). If some objects are missing but the core content still corresponds, give a moderate score (4-7). Only if most content does not correspond or the spatial relationship is chaotic, give a low score (0-3).

• When giving a score, briefly state the main reason for any deduction.

2. Information Balance: Do both views contain a comparable amount and complexity of information regarding the core subjects? Major imbalance (one view rich, one view sparse) should be heavily penalized.

• "Main subject" refers to an object or region in the image with clear boundaries and independent semantic meaning. Large areas of ground, sky, or wall usually count as only one subject.

• If there is a drastic drop in the number of main subjects between the two views (e.g., 10 in the left and only 1 in the right), even if the area is similar, the score should be very low (e.g., 1-2).

• Only when the number of main subjects is similar in both views should you further consider the area, texture complexity, and other details to fine-tune the score.

• Please make full use of the 0-10 score range, and do not only give middle scores. When there is a significant diference in subject number, be bold in giving low scores.

3. Subject Richness and Scene Complexity: Does each view contain rich and varied content—multiple meaningful subjects, some interactions, or a single subject with rich structure/texture? Evaluate left and right views separately; use the lower score.

• High score (8–10): The view contains several distinct subjects, or a single subject with rich detail/structure. As long as the scene is not monotonous, give a high score.

• Medium score (5–7): The view has some content, but is not particularly rich or diverse.

• Low score (0–4): The view is extremely simple or monotonous (e.g., only ground, sky, or a blank wall).

• Do not be too conservative—give high scores to any scene that is clearly not monotonous, regardless of the scene category.

4. Image Clarity: Is the image reasonably clear? Evaluate the left and right images separately, and use the lower ofthe two scores. Minor blur is acceptable, but ifthere are large areas (e.g., more than 20% ofthe image) where all texture and detail are lost due to severe blur, haze, or overexposure, this should be heavily penalized.

5. Content Coherence & Artifacts: Is the image free of major non-geometric distractions (e.g., illogical breaks at the seam, prominent watermarks, or large irrelevant occluders)? Evaluate the left and right images separately, and use the lower ofthe two scores. Apply a minor penalty for such issues.

Output Format: You MUST provide your response in the following strict format:

Sub-scores:

Correlation and Learnability: [0\~10] Reason: [...]

Information Balance: [0\~10] Reason: [...]

Subject Richness and Scene Complexity: [0\~10] Reason: [...]

Image Clarity: [0\~10] Reason: [...]

Content Coherence & Artifacts: [0\~10] Reason: [...]

Final Score: [This MUST be exactly the minimum of all sub-scores above. Do NOT use average, weighted sum, or any other method. There must be no exceptions.]

Summary: [A concise, one-to-two-sentence explanation focusing on the most influential positive and negative factors.]

Important Enforcement:

• You MUST first list all sub-scores, then explicitly identify the minimum value among them.

• The Final Score MUST be exactly this minimum value, with no exceptions.

• Example of a common mistake: If your sub-scores are 6, 8, 4, 9, 9, the Final Score MUST be 4. If you output 6, this is WRONG.

## B CamEditor-Bench and Evaluation Protocols

## B.1 Comprehensive Benchmark Comparison: Data Properties and Evaluation Metrics

To highlight the unique positioning and necessity of CamEditor-Bench, Table S2 systematically compares it with three representative benchmarks across two core aspects: Data Properties and Evaluation Metrics.

## 1. Superiority in Data Properties:

Existing benchmarks are sub-optimal for precise camera editing. Video datasets such as RealEstate10K address synthesis rather than editing and lack paired instructions. Recent camera-centric benchmarks (PreciseCam, Pufin-Gen [25]) focus on pure text-to-image (T2I) generation and omit the structural constraints of sourceconditioned editing. CamEditor-Bench instead targets image editing. By adopting a Source Image + Params modality, it completely bypasses textual spatial ambiguity and provides an absolute mathematical transformation target.

## 2. Rigor in Evaluation Metrics:

Traditional metrics fail to decouple content preservation from massive perspective shifts. While PreciseCam and Pufin-Gen utilize optimization-free geometric representations for T2I generation, they ignore the source identity preservation dilemma inherent in editing tasks. To address this, CamEditor-Bench introduces a fully decoupled evaluation protocol. We combine DINO-v2, CLIP-I, SSIM, and LPIPS to rigorously monitor semantic and structural fidelity. Concurrently, we pioneer Target-Condition $( \mathrm { P F ^ { - } T } )$ and GT-Image (PF-I) alignments based on Perspective Fields $( S _ { u p } , E _ { l a t } )$ . This provides an absolute, optimization-free quantification of geometric precision, preventing the estimator-collapse issues common in traditional SfM-based metrics. The rigorous mathematical derivation of these geometric metrics is detailed in the following subsection.

## B.2 Derivation of the Target Perspective Field (PF-T)

In Section 4.3 of the main text, we introduced the PF-T metric to evaluate the absolute geometric alignment of the generated image against the user-provided target camera conditions. Unlike traditional parameter estimation methods that rely on non-linear optimization and sufer from inherent domain biases, PF-T provides an optimization-free evaluation by computing dense geometric representations directly from the given parameter set.

Given a user-specified camera configuration

$$
\Omega = \{ \mathrm { r o l l } , \mathrm { p i t c h } , \mathrm { v F o V } , \xi \} ,
$$

our goal is to analytically derive the target Perspective Field, which consists of a dense latitude map � and an up-vector field u for every pixel $\mathbf { x } = ( u , v )$

1. Projection Model Formulation. We employ the Unified Spherical (US) camera model to map any 3D spatial point ${ \textbf { X } } =$ $( x , y , z )$ to the 2D image plane. This model efectively accommodates complex non-linear distortions $\xi$ and large field-of-views. The projection function $\mathcal { P } ( \mathbf { X } ) = \mathbf { x }$ is defined as:

$$
\begin{array} { r } { \mathcal { P } ( \mathbf { X } ) = \left[ \begin{array} { l } { u } \\ { v } \end{array} \right] = \left[ \begin{array} { l } { \frac { x f } { \xi \sqrt { x ^ { 2 } + y ^ { 2 } + z ^ { 2 } } + z } + u _ { 0 } } \\ { \frac { y f } { \xi \sqrt { x ^ { 2 } + y ^ { 2 } + z ^ { 2 } } + z } + v _ { 0 } } \end{array} \right] } \end{array}
$$

where $( u _ { 0 } , v _ { 0 } )$ represents the principal point, and the focal length � is derived directly from the user-specified vFoV.

## 2. Latitude Map Derivation

The target extrinsic parameters (roll and pitch) determine the orientation of the camera relative to the world coordinate system, allowing us to establish the gravity vector g within the camera’s reference frame. For any pixel x, the corresponding 3D light ray emitted from the camera center is denoted as R. The target latitude $\varphi _ { \mathbf { x } } ,$ which measures the elevation angle between the light ray R and the horizontal world plane, is computed as:

$$
\varphi _ { \mathbf { x } } = \arcsin \left( { \frac { \mathbf { R } \cdot \mathbf { g } } { \| \mathbf { R } \| _ { 2 } } } \right)
$$

## 3. Up-Vector Field Derivation

The target up-vector $\mathbf { u _ { x } }$ at pixel x represents the 2D image-plane projection of the absolute upward direction in the 3D world (which is strictly opposite to the gravity vector g). Utilizing the projection function $\mathcal { P } _ { : }$ the dense up-vector field is mathematically derived by computing the limit of the projection diference:

$$
\mathbf { u _ { x } } = \operatorname* { l i m } _ { c  0 } { \frac { \mathcal { P } ( \mathbf { X } - c \mathbf { g } ) - \mathcal { P } ( \mathbf { X } ) } { | | \mathcal { P } ( \mathbf { X } - c \mathbf { g } ) - \mathcal { P } ( \mathbf { X } ) | | _ { 2 } } }
$$

By generating the target latitude map $\varphi _ { T }$ and up-vector field ${ \bf u } _ { T }$ through this deterministic formulation, we establish a mathematically rigorous anchor. As described in the main text, the absolute geometric precision of the generative models is subsequently quantified by computing the Mean Squared Error (MSE) of the latitude values $E _ { l a t } ^ { T }$ and the cosine similarity of the upward vectors $S _ { u p } ^ { T } .$

## C Additional Experimental Details

## C.1 Baseline Input Parity and Selection

Table S3 summarizes the information available to each method. The two Wan2.1 controls and CameraEditor use the same Wan2.1- T2V-14B backbone, training data, and optimization settings. The no-reference control receives the target camera through the same deterministic prompt template as the instruction-driven editors. The Plücker control instead receives explicit camera embeddings derived from GeoCalib. Both controls retain CoF, so the comparison isolates the efect of the conditioning representation and visual reference sequence.

We do not include CamEdit [37] as a numerical baseline because its released task definition controls a disjoint parameter space (aperture, focal length, and focal plane), and no public implementation was available at the time of evaluation. A geometric warp followed by inpainting cannot provide ground-truth-aligned content in newly exposed regions, while direct panorama reprojection requires a panoramic user input. We therefore report these approaches as task-boundary references rather than numerical comparators for single-perspective-image editing.

## C.2 Instruction Prompt Conversion for Baselines

Existing instruction-driven models struggle to interpret continuous numerical parameters. To ensure a strictly fair comparison, we establish a deterministic mapping (Table S4) that translates our exact target camera configurations $( p _ { e d i t e d } )$ into hybrid natural language prompts comprehensible to baseline models.

Specifically, for extrinsics (����, ����ℎ) and vertical Field of View (����), we partition their mathematical domains into standard cinematographic terminology while dynamically injecting the exact numerical values to avoid quantization penalties. Since standard text-to-image baselines are agnostic to radial distortion (�), we map its coeficient range (0, 1) into explicit optical lens efects (e.g., from "rectilinear lens" to "extreme spherical fisheye").

A critical vulnerability in prompting generic editing models is "concept bleeding," where geometric instructions are overshadowed by original semantic descriptions. To mitigate this, we design a unified instruction template that strictly decouples structural camera modifiers from the core semantic content. For an instance with scene description $S _ { d e s c }$ and mapped camera terms $( t _ { r o l l } , t _ { p i t c h } , t _ { v F o V } , t _ { \xi } )$ , the prompt is constructed as follows:

## Unified Baseline Prompt Architecture

Task Instruction:

"Edit the provided image to match the specified camera perspective

Table S2: Comprehensive comparison of CamEditor-Bench with representative benchmarks. The comparison is strictly categorized into two primary aspects: Data Properties and Evaluation Metrics. Our benchmark uniquely evaluates precise camera editing using a Source Image + Params modality, combined with a fully decoupled, optimization-free geometric evaluation protocol.
<table><tr><td rowspan="2">Benchmark</td><td colspan="4">Data Properties</td><td colspan="2">Evaluation Metrics</td></tr><tr><td>Test Size</td><td>Core Task Focus</td><td>Data Source</td><td>Instruction Modality</td><td>Content &amp; Quality</td><td>Geometry &amp; Camera (Opt-Free)</td></tr><tr><td>RealEstate10K-test [11]</td><td> $7 { , } 2 8 9 \ s e _ { \ P } s$ </td><td>Video Synthesis</td><td>Real YouTube Videos</td><td>Camera Trajectory</td><td>PSNR, SSIM, LPIPS</td><td>N/A</td></tr><tr><td>PreciseCam [1]</td><td>57,380†</td><td>Camera-guided T2I Gen</td><td>Real Panoramas</td><td>Text + Params</td><td>CLIPScore</td><td> $\mathrm { P F } \mathrm { - } \mathrm { U S } \left( u _ { x } , \phi _ { x } \right)$ </td></tr><tr><td>Puffin-Gen [25]</td><td>650</td><td>Unified T2I Gen &amp; Und</td><td>Real Panoramas + Syn</td><td>Text + Params</td><td>N/A</td><td>Up-vector, Latitude, Gravity</td></tr><tr><td>Ours (CamEditor-Bench)</td><td>462</td><td>Camera-Parameterized Edit</td><td>Real Panoramas + UE5</td><td>Source Image + Params</td><td>DINO-v2, CLIP-I, SSIM, LPIPS</td><td> $\mathbf { P F - T } , \mathbf { P F - I } \left( S _ { u p } , E _ { l a t } \right)$ </td></tr></table>

<sup>†</sup>Full dataset size; no independent held-out evaluation split is defined.

Table S3: Input parity and backbone configuration for the evaluated method families. “Det. prompt” denotes the deterministic template in Table S4.
<table><tr><td>Method family</td><td>Camera signal</td><td>GeoCalib</td><td>Reference sequence</td><td>Backbone</td></tr><tr><td>Open-source editors</td><td>Det. prompt</td><td>No</td><td>No</td><td>Various</td></tr><tr><td>Closed APIs</td><td>Det. prompt</td><td>No</td><td>No</td><td>Undisclosed</td></tr><tr><td>Wan2.1, no reference</td><td>Det. prompt</td><td>No</td><td>No</td><td>Wan2.1-T2V-14B</td></tr><tr><td>Wan2.1 + Plücker</td><td>Plücker embedding</td><td>Yes</td><td>No</td><td>Wan2.1-T2V-14B</td></tr><tr><td>CameraEditor</td><td>Camera parameters</td><td>Yes</td><td>Yes</td><td>Wan2.1-T2V-14B</td></tr></table>

Table S4: Deterministic mapping of continuous camera parameters to hybrid natural language templates for baseline evaluation. The exact numerical parameter $( { \bf e . g . } , r o l l , p i t c h )$ is dynamically injected into the text to avoid quantization penalties.
<table><tr><td>Parameter</td><td>Value Range (p)</td><td>Dynamic Natural Language Template (t)</td></tr><tr><td rowspan="5">Roll</td><td>[−90°,-45°)</td><td>Extreme counterclockwise Dutch angle of roll°</td></tr><tr><td> $[ - 4 5 ^ { \circ } , - 1 0 ^ { \circ } )$ </td><td>Counterclockwise Dutch angle of roll°</td></tr><tr><td> $[ - 1 0 ^ { \circ } , 1 0 ^ { \circ } ]$ </td><td>Near level shot of roll°</td></tr><tr><td>(10°, 45°]</td><td>Clockwise Dutch angle of roll°</td></tr><tr><td>(45°, 90°]</td><td>Extreme clockwise Dutch angle of roll</td></tr><tr><td rowspan="5">Pitch</td><td>[−90°,-45°)</td><td>Extreme high-angle (Bird&#x27;s-eye view) of pitch°</td></tr><tr><td> $[ - 4 5 ^ { \circ } , - 1 0 ^ { \circ } )$ </td><td>High-angle shot (Tilt-down) of pitch°</td></tr><tr><td> $[ - 1 0 ^ { \circ } , 1 0 ^ { \circ } ]$ </td><td>Near straight-on shot of pitch°</td></tr><tr><td>(10°,45°]</td><td>Low-angle shot (Tilt-up) of pitch°</td></tr><tr><td>(45°, 90°]</td><td>Extreme low-angle (Worm&#x27;s-eye view) of pitch°</td></tr><tr><td rowspan="5">vFoV</td><td>[30°, 50°)</td><td>Close-up shot with a vFoV° vertical field of view</td></tr><tr><td>[50°, 75°)</td><td>Medium shot with a vFoV° vertical field of view</td></tr><tr><td>[75°, 100°)</td><td>Wide-angle shot with a vFoV° vertical field of view</td></tr><tr><td>[100°,140°]</td><td>Ultra wide-angle shot with a vFoV° vertical field of view</td></tr><tr><td>[0, 0.2)</td><td></td></tr><tr><td rowspan="4">Distortion (ξ)</td><td></td><td>Standard rectilinear lens with a distortion coefficient of ξ</td></tr><tr><td>[0.2, 0.5)</td><td>Mild barrel distortion with a coefficient of ξ</td></tr><tr><td>[0.5, 0.8)</td><td>Pronounced fisheye effect with a coefficient of ξ</td></tr><tr><td>[0.8, 1.0]</td><td>Extreme spherical fisheye with a coefficient of ξ</td></tr></table>

```tcl
and optical lens efects, while strictly maintaining the original
scene identity and structural integrity."
Camera Modifiers:
"Perspective: Apply a [ $t _ { p i t c h }$ ] and a [ $t _ { r o l l } \ " \mathrm { ~ J ~ . ~ }$
"Lens Efect: Use a $[ \ t _ { v F o V } ]$ combined with a $[ \mathrm { \ } t _ { \xi } ] _ { \cdot } ^ { \ast }$
Semantic Content:
"Original Scene: $[ S _ { d e s c } ] "$
```

By standardizing this architecture, we isolate the baselines’ intrinsic geometric alignment capacities from variations in manual

prompt engineering. This ensures that spatial coherence failures stem directly from architectural limitations.

## C.3 Human Evaluation Protocol

The human study contains 300 blind pairwise comparisons: 100 against Step1X-Edit, 100 against Qwen-Image-Editing, and 100 against Nano-Banana-Pro. Each comparison shows the source image, ground truth, target-camera visualization, and two anonymous edited outputs. Evaluators judge the overall better result by jointly considering perceived camera correctness, source-content preservation, and artifact severity. A tie contributes 0.5 preference to each method. The resulting baseline-win, tie, and CameraEditor-win counts are reported in Table 2.

## C.4 Detailed Configurations of Perception Modules

To answer RQ2 (Section 5.4), we evaluate how initial camera perception accuracy $( \Omega _ { e s t } = \{ r o l l , p i t c h , v F o V , \xi \} )$ impacts final generation quality. Table S5 quantifies the estimation errors of the four evaluated strategies.

1. Random Routing: Uniformly samples parameters without visual perception. The catastrophic geometric misalignment (e.g., $E _ { \mathit { p i t c h } } = 5 4 . 0 0 ^ { \circ } )$ causes severe structural collapse during generation $( \mathrm { D I N O - v 2 : } 0 . 6 3 0 1$ , Table 3).

2. Pufin (Base & Thinking) [25]: We utilize the oficial automated pipeline (understanding.py) to regress parameters directly from the source image. While the Chain-of-Thought reasoning in Pufin (Thinking) slightly reduces $E _ { r o l l }$ and $E _ { v F o V }$ , both variants yield an identical $E _ { \xi } \left( 0 . 3 3 8 \right)$ to the Random baseline. This exposes a fundamental limitation: VLM-based models are blind to continuous non-linear radial distortions, inherently capping their absolute geometric alignment capabilities $( S _ { u p } ^ { T } \le 0 . 5 7 5 2 )$

3. GeoCalib (Ours): Our pipeline uses GeoCalib’s geometric calibration optimizer to extract the dense Perspective Field. Specifically designed for non-linear lens physics, it drastically reduces the distortion error $( E _ { \xi } = 0 . 2 0 1 )$ ) and pitch error $( E _ { p i t c h } = 6 . 0 5 ^ { \circ } )$ ). This robust geometric anchor directly translates to our state-of-the-art generation performance, achieving the highest structural alignment $( S _ { u p } ^ { T } : 0 . 6 0 8 9 )$ and content preservation (DINO-v2: 0.8569). This confirms that precise analytical perception is crucial for coherent camera-parameterized editing.

Table S5: Parameter estimation Mean Absolute Error (MAE). $E _ { r o l l } , E _ { p i t c h : }$ , and $E _ { v F o V }$ denote the angular errors, and $E _ { \xi }$ denotes the distortion coeficient error.
<table><tr><td>Strategy</td><td> $E _ { r o l l }$  (↓)</td><td> $E _ { p i t c h } \left( \downarrow \right)$ </td><td> $E _ { v F o V }$  (↓)</td><td> $E _ { \xi } \left( \downarrow \right)$ </td></tr><tr><td>Random</td><td> $4 8 . 4 2 ^ { \circ }$ </td><td> $5 4 . 0 0 ^ { \circ }$ </td><td> $2 8 . 5 1 ^ { \circ }$ </td><td>0.338</td></tr><tr><td>Puffin (Base) [25]</td><td> $2 6 . 4 0 ^ { \circ }$ </td><td> $1 2 . 0 7 ^ { \circ }$ </td><td> $1 8 . 5 5 ^ { \circ }$ </td><td>0.338</td></tr><tr><td>Puffin (Thinking) [25]</td><td> $2 5 . 8 3 ^ { \circ }$ </td><td> $1 2 . 1 0 ^ { \circ }$ </td><td> $1 7 . 5 2 ^ { \circ }$ </td><td>0.338</td></tr><tr><td>GeoCalib (Ours) [48]</td><td> $2 0 . 7 6 ^ { \circ }$ </td><td> ${ \bf 6 . 0 5 ^ { \circ } }$ </td><td> $2 1 . 8 2 ^ { \circ }$ </td><td>0.201</td></tr></table>

## D Visualization

To further substantiate the quantitative findings presented in the main manuscript, this section provides extended qualitative visualizations. We systematically analyze the generative superiority of our framework, the critical role of accurate camera perception, and the structural impact of our Chain-of-Frames (CoF) temporal priors.

## D.1 Extended Baseline Comparisons

Due to space constraints in the main text, Figure S2 presents an extended qualitative comparison against additional state-of-the-art multi-modal and generative editing models, including Step1X-Edit, GPT-Image-1.5, and Flux2. While these models excel at semantic modifications, they fundamentally struggle with absolute spatial conditioning. Under large perspective shifts, they sufer from severe identity drift (hallucinating new structures) and fail to accurately match the target Perspective Field (PF), as evidenced by the misaligned up-vector and latitude heatmaps. In contrast, our method strictly anchors the generative process to the mathematical constraints, ensuring precise geometric alignment while perfectly preserving the source scene identity.

## D.2 Qualitative Impact of Perception Modules

Figure S3 visualizes the downstream generative consequences of diferent initial camera perception strategies $( \Omega _ { e s t } ) _ { : }$ , directly supporting the quantitative results in Table S5. When the routing mechanism relies on Random sampling or VLM-based estimators (Pufin base, Pufin-thinking), the resulting geometric mismatch forces the difusion model to warp the image into highly unnatural and physi cally impossible perspectives. By integrating GeoCalib, our pipeline secures a mathematically rigorous geometric anchor, completely eliminating these structural collapse artifacts and yielding coherent, high-fidelity perspective edits.

## D.3 Ablation on Intermediate Frames (N)

Figure S4 provides a qualitative ablation on the sequence length (�) within our CoF framework. Attempting to bridge a massive perspective gap in a single step $( N = 0 ;$ , standard image editing) leads to severe spatial tearing and outpainting hallucinations, as the non-linear pixel displacement exceeds the model’s structural capacity. By explicitly decomposing the transformation into a continuous sequence $( N = 4 , 8 , 1 2 )$ , the video difusion prior enforces strong inter-frame geometric consistency, significantly improving the structural integrity of the final edited frame. However, this geometric robustness introduces a fundamental trade-of: as � increases, the accumulated temporal smoothing inherent to the sequence generation progressively compromises high-frequency textural clarity, leading to a noticeable loss of fine-grained details.

## E Limitations

While our proposed Chain-of-Frames (CoF) framework introduces a novel paradigm for structurally coherent, camera-parameterized image editing, it is bounded by several algorithmic and physical limitations that warrant future investigation:

• Trade-of Between Spatial Coherence and Textural Clarity: While our CoF framework ensures absolute geometric stability across massive perspective shifts, the sequential generation paradigm inherently sufers from cumulative errors. The temporal smoothing applied at each intermediate frame compounds as sequence length increases, progressively degrading high-frequency textures and singleframe clarity. Future work will explore hybrid latent upscaling to mitigate this degradation.

• Inherent Ambiguities in 2D-to-3D Camera Estimation: Extracting explicit 3D geometry from 2D images remains fundamentally ill-posed, introducing bottlenecks at both ends of our pipeline. On the input side, wild scenes with weak vanishing-point cues, reflective surfaces, or non-rigid content can cause the initial parameter estimation $( \Omega _ { e s t } )$ to drift, misguiding the entire editing sequence. On the evaluation side, downstream estimators used for our PF-T metric can be tricked by complex generation artifacts, occasionally conflating generative failures with metric noise. Developing robust, generation-aware geometric perceivers is crucial.

![](images/58024fe53c107876c0233482b2850273820ad757c7ed83c22df35d64651d0c4b.jpg)

Input Image  
Ground Truth  
Step1X-Edit  
GPT-Image-1.5  
Flux2  
![](images/1c43bae79f71dad33999aa0834d01fccf8460938ce36ac30ff75e1268945aed4.jpg)  
Ours  
Figure S2: Extended qualitative comparison against additional state-of-the-art editing models. Our framework accurately aligns with the target camera geometry—as verified by the corresponding Perspective Field (PF) heatmaps (right columns)—while strictly preventing the identity drift and structural hallucinations commonly observed in standard difusion baselines.

Input Image  
![](images/83b1f42fbbe04144a30749f4f2b37921ca52af81a032646e47b9384c5f28c6dd.jpg)

Ground Truth  
Random  
![](images/6b7f70f4666990be759cdffc91c01345e28ec37e9e86a6e743e2ea897cdb5b9b.jpg)

Puffin-base  
![](images/f7b6227413e6e4673f90359b0b332e006a153a5ca790befc6f5b8c3b7e388340.jpg)

Puffin-thinking  
![](images/d06ee8d75604d648eca10967ffa26415bd18d7b62d468518a7a1641adf70f163.jpg)

GeoCalib  
![](images/7f41ab47b2a1a16a7554b62d05b4cc82089c86d46ebe88ab24c9eeb6ee62a163.jpg)  
Figure S3: Visual impact of diferent camera perception modules on the final generation quality. Erroneous initial estimations from Random routing or VLM-based understanding (Pufin variants) lead to catastrophic structural distortion. Our GeoCalibdriven routing provides a robust geometric anchor, ensuring physically coherent perspective editing.

![](images/2f10b816a251242474196e7f6cf87e17306c198040d1b7f68815d455069c7ac0.jpg)  
Figure S4: Qualitative ablation on the number of intermediate transition frames (�) in our Chain-of-Frames (CoF) pipeline. Direct single-step generation (� = 0) results in spatial tearing and identity loss under large perspective shifts. Increasing the temporal sequence length (� = 4, 8, 12) provides stronger continuous geometric priors, significantly enhancing structural preservation.