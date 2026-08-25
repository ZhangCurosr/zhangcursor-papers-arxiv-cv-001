# Bridge Damage Detection from Low-Light UAV Imagery via Degradation-Aware Mixture-of-Experts Enhancement

Hu Wang<sup>a</sup>, Hongxu Pu<sup>b</sup>, Zhiqi Hu<sup>c,d</sup>, Fangzhou Lin<sup>e</sup> and Wang Wang<sup>b,∗</sup>

<sup>a</sup>School ofComputer Science and Engineering, University ofElectronic Science and Technology ofChina, No. 2006, Xiyuan Avenue, West Hi-Tech Zone, Chengdu, 611731, China

<sup>b</sup>Sustainability X-Lab, The University of Hong Kong, Hong Kong, China

<sup>c</sup>School ofArchitecture, Building, and Civil Engineering, Loughborough University, Loughborough, UK, LE11 3TU

<sup>d</sup>Department of Engineering Science, University of Oxford, Oxford, UK

<sup>e</sup>Department ofCivil and Environmental Engineering, The Hong Kong University ofScience and Technology, Clear Water Bay, Kowloon, Hong Kong, China

## A R T I C L E I N F O

<sup>Keywords:</sup>g UAV-based bridge inspection Low-light image enhancement Bridge damage detection Mixture-of-experts Sim-to-real transfer

## A BS T R AC T

Poor illumination obscures small, low-contrast defects in UAV bridge imagery, reducing the reliability and operational flexibility of automated inspection. This paper investigates whether degradation-aware image restoration can improve bridge damage detection under low-light conditions and transfer from synthetic degradations to real inspection scenes. We propose DaL MoE, a detector-agnostic restoration front end trained with an ISP-aware low-light synthesis pipeline and equipped with degradation-aware guidance estimation and complementary experts for noise suppression, color adjustment, and structural-detail recovery. On paired synthetic data, DaL-MoE achieves 23.12 dB PSNR and 0.8482 SSIM, increasing YOLOv11m box mA $\mathrm { \bf J } _ { 5 0 }$ from 0.3097 to 0.4923 and mask mAP from 0.2281 to 0.3529. On real low-light UAV imagery without paired normal-light references, sim-to-real evaluation shows improved defect visibility and more complete detections than direct inference on raw low-light inputs. Future work will develop low-light-aware bridge damage detectors with stronger cross-scene generalization across bridge sites, imaging conditions, and illumination levels.

![](images/60a7e45f6cb2a5768e68cd2da86ce5561370ccc0103a6238de223bf724185a01.jpg)  
Figure 1: Conceptual illustration of the motivation for low-light bridge damage detection.

## 1. Introduction

Bridges are long-serving assets within transportation infrastructure systems, and their safety and serviceability are continuously afected by trafic loading, environmental exposure, and material aging [1]. Structural deterioration often first manifests as surface damage, including cracking, spalling, corrosion, seepage, and localized material loss [2]. Although externally visible, these manifestations often indicate the degradation of structural durability, declining service performance, and the accumulation of potential maintenance risks. Timely, accurate, and reliable identification of bridge surface damage is therefore essential for structural condition assessment and maintenance and repair decisionmaking. Eficiently and reliably acquiring and interpreting damage information under practical engineering conditions has consequently become a critical issue in intelligent bridge maintenance and automated inspection [3].

Against this background, UAV-assisted visual inspection has emerged as an important approach to field data acquisition and damage recognition for bridges [4]. Existing studies have extensively investigated UAV inspection path planning and field data acquisition [5], with subsequent eforts extending to image-based damage quantification [6], deep-learning-based bridge crack recognition [7, 8], and the detection of multiple types of surface damage [9]. These advances have substantially improved the accessibility of bridge undersides, piers, elevated components, and bridges located in complex terrain, while promoting a transition from manual field observation toward an integrated workflow of “UAV image acquisition–automated damage recognition–condition information extraction” [10]. However, despite the improvements in inspection accessibility and data-acquisition eficiency, the quality of UAV imagery remains susceptible to field imaging conditions. In particular, uncontrollable illumination beneath bridges, within structural interiors, and in locally occluded regions can cause pronounced low-light degradation in the acquired images, thereby compromising subsequent damage information extraction.

More specifically, illumination conditions during bridge inspection are generally dificult to control. Bridge undersides, inter-girder regions, areas near bearings and expansion joints, box-girder interiors, and component surfaces occluded by structural elements are particularly susceptible to shadows, localized underexposure, and substantial brightness variations. Image exposure is also jointly afected by weather conditions, inspection time, and local occlusions. Because these factors are highly site-dependent, it is dificult to consistently acquire images with uniform brightness, suficient contrast, and clearly visible damage details across diferent structural components and viewpoints. Low-light conditions further cause underexposure, contrast reduction, amplified noise in dark regions, and blurred damage boundaries, thereby reducing the visibility of small and low-contrast damage. Consequently, illumination variation afects not only the visual quality of UAV imagery but also the ability of detection and segmentation models to recognize, localize, and delineate damaged regions.

Beyond low-light degradation and the associated dificulty of restoring information for multiple damage categories, the development of low-light image restoration and damage recognition models is further constrained by the limited availability of suitable training data. Supervised low-light image restoration ideally requires normal-light–low-light image pairs depicting the same damage locations from the same viewpoints with strict spatial correspondence. During UAV-based bridge inspection, however, flight pose, imaging distance, shadow distribution, weather conditions, and camera parameters are dificult to maintain consistently, making strictly paired field data dificult to acquire. Even when the objective is limited to training low-light damage detectors, a large number of field images covering diferent bridge types, structural components, damage categories, and illumination conditions must still be collected and precisely annotated. This process is costly and time-consuming, and annotation quality can be adversely afected by severe underexposure and ambiguous damage boundaries. Consequently, constructing controllable training samples in the absence of real paired low-light data, while recovering visual information beneficial to downstream damage recognition, remains a practical challenge for automated bridge inspection.

To address these challenges, this paper presents a task-oriented low-light visual information restoration framework for bridge damage recognition, rather than a generic low-light enhancement method aimed primarily at improving perceptual appearance. At the data level, a physics-grounded ISP-aware synthesis pipeline is used to transform normal-light bridge imagery into spatially aligned low-light samples while retaining the original multi-class object detection and instance segmentation annotations. This provides a basis for supervised restoration training and controlled evaluation of downstream recognition under low illumination. The related synthesis background and detailed implementation are presented in Sections 2 and 4, respectively. At the model level, the Degradation-aware Low-light Mixture-of-Experts Network (DaL-MoE) is proposed to combine degradation-aware guidance estimation, adaptive expert collaboration, and multi-scale feature restoration, thereby balancing noise suppression, illumination regulation, and preservation of damage-relevant structures. The detailed architecture is described in Section 3. DaL-MoE is trained independently of downstream recognition models and produces standardized RGB images. It can therefore be deployed as a plug-and-play, detector-agnostic restoration front end for diferent object detection and instance segmentation frameworks. The efectiveness of the framework is evaluated at the levels of low-level restoration quality, downstream damage recognition, and transfer to real UAV inspection imagery.

The main contributions of this paper are summarized as follows:

1. To address the scarcity of paired low-light bridge damage data, this paper constructs an ISP-aware dataset comprising normal-light UAV images, spatially aligned synthetic low-light counterparts, and shared annotations for object detection and instance segmentation. Exposure attenuation and sensor noise are simulated in a RAW-like domain while preserving the original damage locations and scene geometry, enabling paired image restoration training and controlled downstream damage recognition evaluation.

2. Building on the constructed dataset, we propose the Degradation-aware Low-light Mixture-of-Experts Network (DaL-MoE), which integrates degradation-aware guidance estimation, hierarchical multi-scale restoration, and adaptive fusion of parallel experts specialized in noise suppression, color adjustment, and structural-detail recovery. Trained independently as an image-to-image restoration model, DaL-MoE functions as a plug-andplay, detector-agnostic front end that requires neither modification of nor joint optimization with downstream detection or segmentation models.

3. Extensive experiments provide a multi-level evaluation of full-reference image restoration quality on paired synthetic data, object detection and instance segmentation performance across multiple downstream frameworks, and real-scene transfer using UAV inspection imagery without paired normal-light references. The results demonstrate consistent improvements on the synthetic benchmark and provide both qualitative and quantitative evidence of sim-to-real transfer to practical bridge inspection scenes.

## 2. Related Work

To address the pronounced performance degradation of bridge damage recognition under uncontrollable illumination, prior investigations have largely developed along two complementary trajectories: benchmark dataset construction and low-light visual restoration. At the data level, recent eforts focus on expanding defect coverage from crack-only patterns to multi-class surface deteriorations and synthesizing controllable low-light degradation when real paired observations are unavailable. At the methodological level, the objective has evolved from generic global illumination enhancement toward task-oriented restoration that suppresses sensor noise while preserving subtle textures and fine boundaries vital for damage delineation. Accordingly, this section reviews existing bridge damage datasets, generic low-light benchmarks, and degradation synthesis pipelines, followed by an analysis of low-light enhancement and task-oriented adaptive restoration models. Finally, the remaining research gaps across data, architecture, and evaluation paradigms are summarized to establish the positioning of this paper.

Table 1 systematically summarizes representative research streams across the data and model levels, highlighting the positioning of this paper.

## 2.1. Datasets for Bridge Damage and Low-Light Vision

Datasets for bridge and concrete surface damage recognition have evolved from binary crack classification toward multi-class damage detection and segmentation. Early benchmarks were predominantly designed for crack categorization. For instance, SDNET2018 collected local image patches from concrete bridge decks, walls, and pavements for binary classification [11], while the COncrete DEfect BRidge IMage dataset (CODEBRIM) extended this setting to multi-label classification covering cracks, spalling, eflorescence, exposed reinforcement, and corrosion stains [12]. Although these datasets laid the foundation for learning defect-sensitive visual representations, their imagelevel formulation ofers limited support for evaluating spatial localization, boundary delineation, and instance-level geometric properties.

Driven by the need for fine-grained structural condition assessment, recent datasets have substantially enhanced both annotation granularity and damage diversity. The BridgeDamage dataset provides over 2,800 inspection images covering five major surface defect categories for multi-class detection and segmentation [13]. GYU-DET further contributes 11,123 high-resolution images with bounding-box annotations across six defect categories, capturing diverse field environments and natural illumination variations [14]. These resources have expanded the data scale and structural complexity available for automated bridge inspection.

Nevertheless, natural illumination diversity in field imagery is fundamentally distinct from strictly paired normallight–low-light observations. Existing bridge inspection datasets are primarily tailored for defect perception under standard lighting; they lack pixel-aligned image pairs depicting identical damage instances and scene geometry under both illumination conditions. Furthermore, they do not provide detection and instance segmentation annotations that are shared and verified across paired lighting states. Consequently, these datasets cannot directly supervise low-light image restoration models, nor can they support controlled benchmark analyses to quantitatively isolate the adverse impact of low illumination on downstream defect recognition while holding spatial geometry constant. This limitation is directly related to the dificulty of acquiring strictly paired field observations under practical UAV inspection conditions.

Positioning of the present study relative to representative data- and model-level research streams for low-light bridge damage analysis.
<table><tr><td>Research level</td><td>Research stream</td><td>Typical data or methods</td><td>Representative capability</td><td>Main limitation relative to the present study</td></tr><tr><td rowspan="4">Data level</td><td>Bridge damage datasets</td><td>Crack classification, multi-label damage classification, multi-class detection, and segmentation [11-14]</td><td>Provide bridge-specific damage categories, inspection scenes, and spatial annotations at diverse granularities</td><td>Lack strictly pixel-aligned normal-light-low-light pairs and shared detection/segmentation annotations across both illumination conditions</td></tr><tr><td>Generic low-light datasets</td><td>Paired low-light restoration, nighttime object detection, and visible-infrared multimodal perception [15-18]</td><td>Support full-reference restoration or high-level visual perception under low illumination</td><td>Focus on natural, traffic, and surveillance domains, lacking civil engineering semantics and bridge defect contexts</td></tr><tr><td>Synthetic low-light data</td><td>sRGB-domain degradation, RAW- domain exposure/noise simulation, and day-to-night synthesis [19-21]</td><td>Provide controllable physical parameters and spatially aligned normal-light-low-light samples</td><td>Are not tailored to multiple bridge defect categories and lack mechanisms to preserve and evaluate instance-level annotations</td></tr><tr><td>Low-light infrastructure datasets</td><td>Normal- and low-light crack images and multi-darkness crack segmentation benchmarks [22, 23]</td><td>Provide specialized data for low-light enhancement and crack segmentation in civil structures</td><td>Remain strictly crack-centric and do not support simultaneous object detection and instance segmentation across multiple defect categories</td></tr><tr><td rowspan="3">Model level</td><td>Generic low-light enhancement and restoration</td><td>Retinex decomposition, illumination estimation, deep restoration backbones, and compound degradation recovery [24–32]</td><td>Recover illumination, suppress noise, and restore color and fine details</td><td>Are optimized and evaluated on pixel reconstruction or perceptual quality, without explicitly preserving defect-critical boundaries and textures</td></tr><tr><td>Low-light infrastructure perception</td><td>Low-light crack enhancement, reflectance modeling, and enhancement-assisted crack segmentation [22, 23, 33]</td><td>Demonstrate the efficacy of illumination restoration for civil infrastructure crack identification</td><td>Target a single crack category with backbone-coupled pipelines, lacking multi-class evaluation and cross- framework adaptability</td></tr><tr><td>Adaptive and degradation-aware restoration</td><td>Degradation prompts, conditional restoration, dynamic routing, and mixture-of-experts architectures [34, 35]</td><td>Adapt restoration behavior dynamically based on input degradation characteristics</td><td>Target generic restoration, with limited exploration of defect-tailored expert specialization, degradation guidance, and detector-agnostic deployment</td></tr><tr><td>This paper</td><td>Low-light multi- class bridge damage restoration and recognition</td><td>ISP-aware paired low-light synthesis, degradation-aware MoE restoration, and multi-framework evaluation</td><td>Bridges physics-grounded synthesis, multi-class visual information recovery, object detection, and instance segmentation</td><td>Addresses the underexplored intersection of bridge- specific paired data construction, degradation-adaptive restoration, and cross-framework defect recognition</td></tr></table>

In parallel, the low-light vision community has developed specialized benchmarks for image restoration and highlevel visual perception. For low-level restoration, the LOL dataset provides real-world paired low-light and normallight images [15], while See-in-the-Dark (SID) pairs short-exposure RAW observations with long-exposure reference images for extreme low-light recovery [16]. For high-level perception, ExDark establishes a multi-class object detection benchmark under low illumination [17], and LLVIP contributes spatially aligned visible–infrared image pairs for lowlight pedestrian perception [18]. While efective in their intended domains, these benchmarks predominantly feature natural landscapes, urban trafic, and surveillance scenes, exhibiting a substantial domain gap from bridge surface defects and civil engineering semantics. They therefore do not directly satisfy the data requirements of low-light bridge damage restoration and multi-class recognition.

To circumvent the dificulty of collecting real paired data, physics-based low-light synthesis ofers a controllable alternative for generating spatially aligned training samples. LOL-v2 demonstrated the eficacy of synthetic lowlight subsets for algorithm validation [19]. Unprocessing pipelines invert the camera image signal processing (ISP) pipeline into a RAW-like domain, realistically simulating exposure attenuation, signal-dependent shot noise, and signalindependent read noise [20]. Similar principles have been applied to synthesize paired daytime and nighttime images for training neural ISPs [21]. However, existing synthesis frameworks are designed for generic natural imagery rather than civil structures, lacking mechanisms to preserve and reuse fine-grained bounding-box and instance-mask annotations across degraded and restored counterparts. This limitation restricts their direct application to multi-class bridge damage restoration and downstream recognition evaluation.

Currently, low-light datasets dedicated to civil infrastructure remain scarce and strictly crack-centric. CrackNex introduced the LCSD dataset for few-shot crack segmentation under varying illumination [22], and GalleryLL-1240 provided crack images across distinct darkness levels for underlit dam galleries [23]. Nonetheless, actual bridge visual inspection encounters a heterogeneous spectrum of compound deterioration; beyond linear cracks, it involves spalling, seepage, honeycombing, exposed reinforcement, reinforcement corrosion, rust staining, and eflorescence. These defect categories exhibit distinct scale distributions, color contrasts, texture roughness, and topological boundaries, imposing conflicting demands on restoration and feature preservation. Therefore, a low-light bridge dataset that simultaneously provides strict spatial pairing, physics-grounded degradation simulation, and multi-class annotations for both detection and instance segmentation remains an unresolved gap in the literature.

## 2.2. Low-Light Image Enhancement and Task-Oriented Restoration

Low-light image processing has evolved from conventional heuristic illumination adjustment and Retinex decomposition toward data-driven restoration of compound degradations. Classical Retinex models formulate a low-light observation as the product of reflectance and illumination layers, enhancing visibility by estimating either component [24]. Illumination-map estimation methods such as LIME further recover dark-region content via spatially varying illumination estimation paired with structure-preserving constraints [25]. Although these approaches established the theoretical foundation for illumination modeling, their performance in complex real-world environments is often degraded by sensor noise amplification, severe color casts, and non-uniform lighting.

With the advent of deep learning, illumination compensation, noise suppression, and detail reconstruction can be jointly optimized. Zero-DCE frames low-light enhancement as a zero-reference pixel-wise curve estimation task, eliminating the dependency on paired training data [26]. MIRNet preserves complementary representations across spatial scales through recursive multi-scale feature interaction [27], while Restormer introduces an eficient transposed Transformer architecture to capture long-range token dependencies in high-resolution image restoration [28]. These methodologies have advanced low-light processing from empirical intensity stretching to deep, multi-scale feature modeling.

Recent studies increasingly recognize that low-light degradation is rarely a simple global luminance reduction, but rather an entangled compound process involving severe noise, contrast loss, chromatic distortion, and structural degradation. SNR-Aware enhancement introduces spatial signal-to-noise-ratio priors to dynamically balance local convolutional filtering and non-local feature interactions across heterogeneous degradation zones [29]. Retinexformer explicitly estimates illumination cues to guide non-local self-attention and corruption recovery [30]. HVI-CIDNet designs a decoupled color space to alleviate chromatic distortion and refine photometric mapping under complex lighting [31], whereas DarkIR systematically addresses the co-occurrence of insuficient illumination, noise, and motion blur in real-world nighttime scenarios [32]. Collectively, these advances highlight a paradigm shift toward the coordinated restoration of multi-faceted physical degradations.

Despite these achievements, prevailing low-light restoration models remain predominantly optimized and benchmarked for pixel-level reconstruction fidelity (e.g., PSNR and SSIM) or human perceptual appearance. However, improvements in visual brightness or generic fidelity metrics do not necessarily guarantee the preservation of weak textures, subtle boundaries, and low-contrast morphological cues indispensable to downstream recognition models. In short, a visually pleasing or brightened image does not inherently translate into superior defect localization and boundary delineation.

This metric discrepancy is particularly pronounced in bridge surface damage inspection. Overly aggressive denoising tends to obliterate thin cracks and subtle spalling edges; unconstrained brightness amplification frequently exaggerates concrete surface roughness, stains, and environmental background clutter; and excessive high-frequency sharpening can introduce pseudo-structural artifacts that misdirect downstream detectors. Consequently, low-light processing for civil inspection must move beyond cosmetic visual enhancement to achieve a rigorous balance among illumination recovery, artifact suppression, and structural preservation, with its utility quantitatively verified on downstream tasks such as object detection and instance segmentation. This requirement is consistent with the taskoriented low-light visual information restoration objective introduced in the Introduction.

A few recent eforts have begun integrating low-light restoration with civil infrastructure inspection. CrackNex derives illumination-invariant reflectance features via Retinex decomposition for few-shot crack segmentation under weak illumination [22]. Related work has utilized conditional generative models to recover concrete crack images degraded by low light, overexposure, and blur prior to automated assessment [33]. For underlit dam galleries, LL-CrackSeg integrates unsupervised enhancement with a customized segmentation network to improve crack extraction accuracy [23]. Nevertheless, these methods remain strictly crack-centric, and their enhancement modules are typically tightly coupled with specific segmentation backbones. Unlike relatively regular, linear crack topologies, non-crack bridge deteriorations (such as cavity depths in spalling, difuse scattering in seepage, or chromatic shifts in corrosion and eflorescence) display much stronger heterogeneity in roughness, local contrast, and region topology. These diverse defects impose conflicting requirements on illumination compensation and structural preservation. Crack-specific strategies cannot be directly generalized to multi-class scenarios, and whether restoration front ends can stably benefit diverse detection and instance segmentation architectures remains insuficiently verified.

Furthermore, the spatial heterogeneity of compound low-light degradation imposes strict demands on the adaptive capacity of restoration networks. A single monolithic pathway often faces conflicting optimization objectives when simultaneously performing denoising, brightness lifting, color correction, and detail enhancement. PromptIR introduces learnable degradation prompts to conditionally modulate network representations across diverse corruption types [34]. The Mixture-of-Experts (MoE) paradigm provides an alternative adaptive framework by allocating specialized subnetworks to learn complementary restoration behaviors, dynamically arbitrated by a lightweight gating network [35]. This mechanism is well-suited for low-light bridge imagery, where the severity of noise, underexposure, and boundary degradation varies across structural elements and spatial regions. Nonetheless, explicitly designing functional experts tailored to bridge defect characteristics, guiding their feature recovery via estimated degradation cues, and deploying the model as a detector-agnostic front end for cross-framework evaluation remain largely unexplored in multi-class bridge inspection.

## 2.3. Research Gap and Positioning of This Paper

In summary, existing low-light vision research remains misaligned with the practical requirements of UAV-based bridge damage inspection. Although existing bridge damage datasets provide domain-specific annotations for multiple damage categories, they generally lack strictly spatially aligned normal-light–low-light image pairs. By contrast, generic low-light datasets and image restoration methods do not adequately reflect the domain semantics of bridge damage. Existing infrastructure-oriented studies also remain largely focused on cracks as a single damage category or are tightly coupled with specific perception backbones. Therefore, the unresolved issue is not merely how to increase the brightness of low-light bridge images, but how to recover damage-relevant visual information under spatially heterogeneous and compound low-light degradation, such that the restored results can support multi-class damage localization and contour delineation across diferent recognition frameworks.

Based on the research gaps identified above, this paper seeks to answer the following two questions:

1. In the absence of real paired normal-light–low-light UAV bridge imagery, how can controllable training and evaluation conditions be established?

2. How can a detector-agnostic image restoration model recover visual cues relevant to damage recognition for diferent object detection and instance segmentation frameworks?

Accordingly, this paper does not focus on low-light enhancement aimed solely at improving perceptual brightness, but rather on recovering the visual evidence required for reliable multi-class bridge damage recognition under adverse illumination conditions.

## 3. Methodology

This section presents the Degradation-aware Low-light Mixture-of-Experts Network (DaL-MoE) for task-oriented bridge image enhancement under poor illumination. In computer-aided bridge inspection, low-light degradation is rarely a simple reduction in overall image brightness. It typically manifests as a compound of spatially coupled phenomena: dark-region noise amplification, loss of local contrast, tonal and chromatic instability, and weakened structural boundaries. These degradation factors can collectively obscure damage-relevant evidence, including crack discontinuities, spalling contours, corrosion traces, seepage regions, exposed rebar edges, and subtle material texture variations. Accordingly, an enhancement model intended to support bridge damage analysis must go beyond global brightness lifting: it must suppress low-light artifacts while preserving the structural cues that are informative for subsequent detection and instance segmentation.

The term task-oriented in this work indicates that the enhancement model is designed and evaluated with downstream bridge damage analysis as the target application, while the enhancement network itself remains a detectoragnostic restoration front-end. DaL-MoE therefore does not require a detection loss or an embedded detector during training. Instead, its architecture is organized around the visual degradation factors that most directly impair bridge damage recognition under poor illumination.

The design follows a two-stage principle. In the first stage, the network estimates illumination- and noise-related guidance from the low-light input image, identifying spatial regions where degradation is likely to be more severe. In the second stage, a U-shaped restoration backbone employs an adaptive mixture-of-experts mechanism to balance complementary restoration behaviors, since noise suppression, brightness correction, and structural detail preservation may be needed to diferent degrees across bridge scenes, viewpoints, and illumination conditions. This principle leads to the two-module architecture illustrated in Fig. 2, consisting of a Degradation-Aware Guidance Estimation (DAGE) module and an Adaptive Mixture-of-Experts Restoration (AMER) module.

![](images/658942d6897b2676aa9f0f59b42556dc5498419e4a4f41cc5742bed58b0003dc.jpg)  
Figure 2: Overview of the proposed DaL-MoE framework. (a) Overall architecture, consisting of a Degradation-Aware Guidance Estimation (DAGE) module and a U-shaped Adaptive Mixture-of-Experts Restoration (AMER) network. (b) Degradation-Aware Guidance Estimation module, which estimates a spatially varying illumination map and generates noise-aware guidance features from the low-light input. (c) Adaptive Mixture-of-Experts Restoration module, where noiseaware, color adjustment, and detail enhancement experts are dynamically integrated by an adaptive gating network for feature restoration.

Problem statement. Given a low-light bridge image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , the goal is to learn an enhancement mapping $\mathcal { N } ( \cdot )$ that produces an enhanced image $\bar { \boldsymbol { I } } \in \mathbb { R } ^ { H \times W \times 3 } ; \bar { \boldsymbol { I } } = \mathcal { N } ( \boldsymbol { I } )$ . The enhanced output is expected to exhibit improved visibility, reduced low-light degradation, and better-preserved damage-relevant structural cues for downstream bridge damage analysis.

Overview. The overall architecture of DaL-MoE, illustrated in Fig. 2(a), consists of two main components: a Degradation-Aware Guidance Estimation (DAGE) Module and an Adaptive Mixture-of-Experts Restoration (AMER) Module. Given a low-light input image �, the degradation-aware guidance estimation module first predicts an illumination map � and extracts a noise-aware guidance feature $F _ { g }$ . The illumination map is used to preliminarily compensate for insuficient illumination through residual illumination scaling, producing a lightened image $I _ { 0 } ,$ , while $F _ { g }$ encodes degradation-related information for subsequent restoration. The lightened image $I _ { 0 }$ is first encoded into the feature space through a $3 \times 3$ convolution, and the resulting image feature, together with the guidance feature $F _ { g }$ , is then fed into the adaptive mixture-of-experts restoration (AMER) module. Within its U-shaped backbone, $F _ { g }$ is resampled to match the spatial resolution at each scale, while adaptive MoE blocks progressively restore the image representation by dynamically integrating complementary noise-suppression, brightness-correction, and structural-detail recovery behaviors. Finally, the decoded feature is projected into the RGB space through a $3 \times 3$ convolution and added to $I _ { 0 }$ via residual reconstruction to obtain the enhanced image �<sup>̂</sup>. In the following, we describe the DAGE module and the AMER module in detail.

## 3.1. Degradation-Aware Guidance Estimation Module

As illustrated in Fig. 2(b), the degradation-aware guidance estimation module is designed to characterize spatially varying low-light degradation and provide explicit guidance for subsequent feature restoration. The module consists of three functional components: a shared feature embedding block, a light-up estimation block, and a noise-aware guidance generation block. Given a low-light image $I \in \mathbb { R } ^ { H \times W \times 3 }$ , the shared feature embedding block first augments the RGB observation with an illumination-sensitive intensity cue and extracts a compact representation $F _ { \mathrm { e } } .$ Based on this shared representation, the light-up estimation block predicts a spatially varying map � and produces a preliminarily compensated image $I _ { 0 }$ through residual illumination scaling. Meanwhile, a noise-related degradation prior $P _ { n }$ is derived from � and combined with $F _ { \mathrm { e } }$ in the noise-aware guidance generation block to obtain the guidance feature $F _ { g }$ . The overall mapping of the module is expressed as

$$
( F _ { g } , M ) = \mathcal { P } ( I ) ,\tag{1}
$$

where $M \in [ 0 , 1 ] ^ { H \times W \times 3 }$ denotes the predicted light-up map and $F _ { g } \in \mathbb { R } ^ { H \times W \times C }$ denotes the noise-aware guidance feature. Consequently, the module provides two complementary outputs for subsequent restoration: the preliminarily compensated image $I _ { 0 }$ as the restoration input and the degradation-aware feature $F _ { g }$ as conditional guidance.

Shared Feature Embedding Block. Directly estimating degradation characteristics from the RGB input alone may provide insuficient illumination context, particularly in spatially non-uniform low-light regions. To introduce an explicit intensity cue while retaining the original chromatic information, the input image is augmented with its channel-wise mean map:

$$
I _ { \mathrm { c a t } } = \operatorname { C o n c a t } \left( I , \operatorname { M e a n } _ { c } ( I ) \right) \in \mathbb { R } ^ { H \times W \times 4 } ,\tag{2}
$$

where $\mathrm { M e a n } _ { c } ( \cdot )$ denotes averaging along the channel dimension. The resulting four-channel representation jointly preserves the RGB appearance and provides a compact description of the local intensity distribution.

The augmented input is subsequently projected into a �-dimensional feature space using a $1 \times 1$ convolution, followed by a $5 \times 5$ depthwise convolution for local spatial aggregation:

$$
F _ { \mathrm { e } } = \eta \big ( D _ { 5 \times 5 } ^ { \mathrm { c t x } } \big ( \eta \big ( C _ { 1 \times 1 } ^ { \mathrm { e m b } } ( I _ { \mathrm { c a t } } ) \big ) \big ) \big ) ,\tag{3}
$$

where �(⋅) denotes the LeakyReLU activation. The $1 \times 1$ convolution performs cross-channel feature mixing, whereas the $5 \times 5$ depthwise convolution aggregates local spatial context with limited computational overhead. The resulting embedding $\dot { F } _ { \mathrm { e } } \in \mathbb { R } ^ { H \times W \times C }$ serves as a shared representation for both illumination compensation and degradation-aware guidance generation.

Light-Up Estimation Block. Based on the shared feature $F _ { \mathrm { e } } ,$ , the light-up estimation block predicts a spatially varying map through a $1 \times 1$ convolution followed by Sigmoid normalization:

$$
M = \sigma \big ( { \cal C } _ { 1 \times 1 } ^ { \mathrm { l u m } } ( F _ { \mathrm { e } } ) \big ) \in [ 0 , 1 ] ^ { H \times W \times 3 } ,\tag{4}
$$

where $\sigma ( \cdot )$ denotes the Sigmoid function. Rather than directly replacing the original image intensity, the estimated map is applied through residual illumination scaling:

$$
I _ { 0 } = { \cal I } \odot ( 1 + M ) ,\tag{5}
$$

where ⊙ denotes element-wise multiplication. This formulation preserves the original low-light observation while introducing spatially adaptive illumination compensation, thereby providing a better-exposed image representation for subsequent restoration.

Noise-Aware Guidance Generation Block. Low-light degradation is generally spatially heterogeneous, with poorly illuminated regions being more susceptible to noise amplification and contrast deterioration. Instead of introducing an additional noise-estimation network or explicit noise supervision, a lightweight degradation prior is derived directly from the predicted map �:

$$
P _ { n } = 1 - \mathrm { M e a n } _ { c } ( M ) \in [ 0 , 1 ] ^ { H \times W \times 1 } ,\tag{6}
$$

where $P _ { n }$ provides a complementary spatial cue for noise-aware restoration. This complementary representation allows degradation information to be introduced without additional supervisory signals or substantial computational overhead.

The degradation prior $P _ { n }$ is then concatenated with the shared embedding $F _ { \mathrm { e } }$ and transformed by two successive $3 \times 3$ convolutions to generate the final noise-aware guidance feature:

$$
F _ { g } = C _ { 3 \times 3 } ^ { \mathrm { g o u t } } \Bigl ( \eta \Bigl ( C _ { 3 \times 3 } ^ { \mathrm { g i n } } \bigl ( \mathrm { C o n c a t } ( F _ { \mathrm { e } } , P _ { n } ) \Bigr ) \Bigr ) ,\tag{7}
$$

where $c _ { 3 \times 3 } ^ { \mathrm { g i n } }$ and $c _ { 3 \times 3 } ^ { \mathrm { g o u t } }$ denote independently parameterized convolutional layers. By jointly modeling the appearance information contained in $F _ { \mathrm { e } }$ and the degradation cue encoded by $P _ { n } .$ the resulting $F _ { g }$ captures spatially varying information relevant to low-light restoration.

During the subsequent restoration process, $F _ { g }$ is spatially resampled to match the feature resolution at each level of the U-shaped backbone, allowing the restoration blocks to exploit scale-aligned degradation guidance. The preliminarily compensated image $I _ { 0 }$ and the guidance feature $F _ { g }$ are then jointly forwarded to the Adaptive Mixtureof-Experts Restoration Module, which is introduced in the following subsection.

## 3.2. Adaptive Mixture-of-Experts Restoration Module

As illustrated in Fig. 2(c), the Adaptive Mixture-of-Experts Restoration Module is designed to adaptively address the coupled degradations present in low-light bridge imagery. At the �-th backbone scale, the module receives an image feature $\check { F } _ { i } \in \mathbb { R } ^ { H _ { i } \times \check { W } _ { i } \times C _ { i } }$ together with a scale-aligned degradation guidance feature $G _ { i }$ . The latter is obtained by spatially aligning the guidance feature $F _ { g }$ generated by the preceding Degradation-Aware Guidance Estimation Module:

$$
G _ { i } = { \cal C } _ { i } ^ { g } \left( { \cal A } _ { i } ( F _ { g } ) \right) ,\tag{8}
$$

where $\boldsymbol { \mathcal { A } } _ { i } ( \cdot )$ denotes spatial resampling to match the resolution of $F _ { i }$ , and $\mathcal { C } _ { i } ^ { g } ( \cdot )$ denotes a scale-specific 1×1 convolution that projects the resampled guidance feature to $C _ { i }$ channels, yielding $G _ { i } \doteq \mathbb { R } ^ { H _ { i } \times W _ { i } \times C _ { i } }$

The module consists of four functional blocks: a Noise-Aware Expert, a Color Adjustment Expert, a Detail Enhancement Expert, and an Adaptive Gating Network. The three expert blocks operate in parallel on the shared input feature $F _ { i }$ and specialize in complementary restoration objectives. Specifically, the Noise-Aware Expert additionally incorporates the degradation guidance $G _ { i }$ to suppress noise-related corruption, the Color Adjustment Expert regulates illumination-induced chromatic distortions and tonal responses, and the Detail Enhancement Expert focuses on recovering local boundaries and structural details. Their outputs are formulated as

$$
F _ { n } = \mathcal { E } _ { n } ( F _ { i } , G _ { i } ) , \qquad F _ { b } = \mathcal { E } _ { b } ( F _ { i } ) , \qquad F _ { d } = \mathcal { E } _ { d } ( F _ { i } ) ,\tag{9}
$$

where $F _ { n } , F _ { c }$ , and $F _ { d }$ denote the feature representations produced by the Noise-Aware Expert, Color Adjustment Expert, and Detail Enhancement Expert, respectively. Rather than imposing a predefined restoration order on diferent degradation factors, the parallel expert organization enables each branch to independently learn degradation-specific representations from the same input feature. The Adaptive Gating Network subsequently estimates expert-specific routing weights and adaptively integrates the three complementary representations into the restored output feature $F _ { \mathrm { o u t } }$ . The four functional blocks are detailed below.

## 3.2.1. Noise-Aware Expert

The Noise-Aware Expert $\mathcal { E } _ { n }$ is designed to suppress noise-related degradations that are particularly pronounced in poorly illuminated regions. Unlike the other two experts, $\mathcal { E } _ { n }$ explicitly incorporates the scale-aligned guidance feature $G _ { i }$ , which encodes degradation characteristics estimated from the original low-light input. By introducing such degradation-aware information into feature aggregation, $\mathcal { E } _ { n }$ adaptively suppresses noise-sensitive responses while preserving informative structural content.

Specifically, given the input feature $F _ { i }$ and its corresponding guidance feature $G _ { i } , \mathcal { E } _ { n }$ first employs noise-guided multi-head transposed self-attention (NG-MTSA) to perform degradation-aware feature aggregation. The resulting feature is then further refined by a convolutional feed-forward network (FFN):

$$
\widetilde { F } _ { i } = F _ { i } + \mathrm { N G - M T S A } ( F _ { i } , G _ { i } ) , \qquad F _ { n } = \widetilde { F } _ { i } + \mathrm { F F N } \Big ( \mathrm { L N } ( \widetilde { F } _ { i } ) \Big ) ,\tag{10}
$$

where LN(⋅) denotes layer normalization, and $F _ { n }$ is the output of the Noise-Aware Expert. The FFN adopts a convolutional architecture consisting of a $1 \times 1$ convolution, a depthwise $3 \times 3$ convolution, and a subsequent $1 \times 1$ convolution, with GELU nonlinearities inserted between consecutive convolutional layers.

We next describe the NG-MTSA operation in detail. Given an input feature $F \in \mathbb { R } ^ { H _ { i } \times W _ { i } \times C _ { i } }$ , let $C _ { i } = h d$ , where ℎ d h b f i h d d � d h f di i f h h d d fi $N = H _ { i } W _ { i }$ and reshape � into the token representation $\mathbf { F } \in \mathbb { R } ^ { N \times C _ { i } }$ . The query, key, and value representations are generated through learnable projections:

$$
Q = { \bf F } W _ { q } , \qquad K = { \bf F } W _ { k } , \qquad V = { \bf F } W _ { v } ,\tag{11}
$$

where $W _ { q } , W _ { k } , W _ { v } \in \mathbb R ^ { C _ { i } \times C _ { i } }$ are learnable projection matrices. The projected representations are partitioned into ℎ attention heads, yielding $Q _ { j } , K _ { j } , V _ { j } \in \mathbb { R } ^ { N \times d }$ for the �-th head. Similarly, the scale-aligned guidance feature $G _ { i }$ is reshaped and partitioned into $G _ { i } \in \mathbb { R } ^ { N \times d }$

To explicitly incorporate degradation-aware information into feature aggregation, the guidance representation is injected into the value branch through element-wise modulation:

$$
\begin{array} { r } { \bar { V } _ { j } = V _ { j } \odot G _ { j } , } \end{array}\tag{12}
$$

where ⊙ denotes element-wise multiplication. In this manner, the degradation cues encoded in $G _ { i }$ directly regulate the feature responses participating in attention aggregation, allowing the network to attenuate noise-sensitive activations while retaining structurally informative responses.

Instead of constructing spatial token-to-token afinities, whose computational cost grows quadratically with the number of spatial locations, NG-MTSA performs attention in a transposed channel space. Specifically, the query and key representations are first transposed and normalized along the token dimension:

$$
\widehat { Q } _ { j } = \ell _ { 2 } ( Q _ { j } ^ { \top } ) , \qquad \widehat { K } _ { j } = \ell _ { 2 } ( K _ { j } ^ { \top } ) ,\tag{13}
$$

where $\hat { Q } _ { j } , \hat { K } _ { j } \in \mathbb R ^ { d \times N }$ , and $\ell _ { 2 } ( \cdot )$ denotes $\ell _ { 2 }$ normalization along the token dimension. The channel-wise attention matrix for the �-th head is then computed as

$$
A _ { j } = \operatorname { S o f t m a x } \left( \lambda _ { j } \widehat { K } _ { j } \widehat { Q } _ { j } ^ { \intercal } \right) \in \mathbb { R } ^ { d \times d } ,\tag{14}
$$

where $\lambda _ { j }$ is a learnable rescaling parameter associated with the �-th attention head. Consequently, NG-MTSA models global inter-channel dependencies through a � × � attention matrix without explicitly constructing an $N \times N$ spatial afinity matrix.

The degradation-guided attention output of each head is obtained by aggregating the modulated value representation:

$$
\begin{array} { r } { Y _ { j } = \left( A _ { j } \bar { V } _ { j } ^ { \top } \right) ^ { \top } \in \mathbb { R } ^ { N \times d } . } \end{array}\tag{15}
$$

The outputs of all attention heads are subsequently concatenated and projected back to the original feature dimension. Since transposed channel attention mainly captures global inter-channel dependencies, an additional local positional branch is introduced to complement the attention output with local spatial information. In particular, this branch operates on the unmodulated value representation �, such that the degradation guidance only afects the global attention aggregation and does not directly alter the local positional cues. The complete NG-MTSA operation is formulated as

$$
\begin{array} { r } { \mathrm { N G \mathrm { \bf - M T S A } } ( { \boldsymbol F } , { \boldsymbol G } _ { i } ) = \mathcal { R } ^ { - 1 } \biggl ( \mathrm { C o n c a t } _ { j = 1 } ^ { h } ( Y _ { j } ) W _ { o } \biggr ) + \mathrm { P o s } \bigl ( \mathcal { R } ^ { - 1 } ( { \boldsymbol V } ) \bigr ) , } \end{array}\tag{16}
$$

where $W _ { o } \in \mathbb { R } ^ { C _ { i } \times C _ { i } }$ denotes the output projection matrix, $\mathcal { R } ^ { - 1 } ( \cdot )$ reshapes the token representation back to its spatial form, and Pos(⋅) denotes the local positional branch implemented by two successive depthwise $3 \times 3$ convolutions with an intermediate GELU activation.

By combining degradation-guided value modulation, transposed channel-wise attention, and local positional compensation, NG-MTSA enables $\mathcal { E } _ { n }$ to selectively suppress noise-related feature responses while preserving global structural dependencies and local spatial details. The resulting feature $F _ { n }$ serves as the noise-aware representation produced by $\mathcal { E } _ { n }$ for subsequent feature fusion and reconstruction.

## 3.2.2. Color Adjustment Expert

The Color Adjustment Expert $\mathcal { E } _ { c }$ is designed to correct color distortions and illumination-induced chromatic shifts in low-light features. In poorly illuminated regions, insuficient exposure often leads not only to reduced brightness bu also to inaccurate color responses and uneven tonal distributions. To address these degradations, $\mathcal { E } _ { c }$ combines globa channel-wise afine modulation with local chromatic refinement, enabling adaptive adjustment of both global color statistics and spatially varying chromatic responses.

Color–Illumination Embedding. The shared input feature is first projected into a latent color–illumination representation:

$$
F _ { c } = \delta \big ( C _ { 1 \times 1 } ^ { \mathrm { c o l } } ( F _ { i } ) \big ) ,\tag{17}
$$

where $\delta ( \cdot )$ denotes GELU. The $1 \times 1$ convolution reorganizes cross-channel responses and produces a compact representation that facilitates subsequent color and illumination modulation.

Global Afine Modulation. To capture global color and illumination statistics, a channel descriptor is first extracted from $F _ { c }$ through global average pooling. The descriptor is then processed by two successive $1 \times 1$ convolutions with an intermediate ReLU activation to estimate channel-wise afine parameters:

$$
[ \gamma , \beta ] = \mathrm { S p l i t } \left( C _ { 1 \times 1 } ^ { \mathrm { a f f } 2 } \left( \rho { \left( C _ { 1 \times 1 } ^ { \mathrm { a f f } 1 } \left( \mathrm { G A P } ( F _ { c } ) \right) \right) } \right) \right) ,\tag{18}
$$

where $\rho ( \cdot )$ denotes ReLU, $c _ { 1 \times 1 } ^ { \mathrm { a f f 1 } }$ reduces the channel dimension to $C _ { i } / 4 _ { ; }$ , and $c _ { 1 \times 1 } ^ { \mathrm { a f f } 2 }$ expands it to $2 C _ { i }$ . Split(⋅) evenly partitions the output along the channel dimension, yielding $\gamma , \beta \in \mathbb { R } ^ { C _ { i } \times 1 \times 1 }$

The resulting afine parameters are spatially broadcast and applied to $F _ { c }$ through a residual afine transformation:

$$
F _ { a } = F _ { c } \odot ( 1 + \gamma ) + \beta .\tag{19}
$$

The multiplicative term adaptively rescales channel responses associated with color and illumination characteristics, while the additive term compensates for global chromatic and tonal ofsets. The residual parameterization $( 1 + \gamma )$ provides an identity-preserving modulation path, allowing the expert to adjust degraded color responses without unnecessarily perturbing informative feature components.

Local Chromatic Refinement. Global channel-wise modulation alone is insuficient to characterize spatially varying color distortions caused by non-uniform illumination. Therefore, $F _ { a }$ is further processed by a local chromatic refinement operator:

$$
F _ { \mathrm { o u t } } ^ { c } = C _ { 1 \times 1 } ^ { \mathrm { c o u t } } \big ( \delta \big ( { \cal D } _ { 3 \times 3 } ^ { \mathrm { c l o c } } ( F _ { a } ) \big ) \big ) + F _ { i } ,\tag{20}
$$

where $\boldsymbol { D } _ { 3 \times 3 } ^ { \mathrm { c l o c } }$ captures local spatial and chromatic variations, and $c _ { 1 \times 1 } ^ { \mathrm { c o u t } }$ projects the refined representation back to the original channel dimension. The residual connection from $F _ { i }$ preserves the underlying structural representation while allowing $\mathcal { E } _ { c }$ to selectively correct color distortions and locally varying chromatic responses. The resulting $F _ { \mathrm { o u t } } ^ { c }$ provides a color-adjusted representation complementary to the outputs of the other expert branches.

## 3.2.3. Detail Enhancement Expert

The Detail Enhancement Expert $\mathcal { E } _ { d }$ operates directly on the shared input feature $F _ { i }$ and focuses on recovering local structural information that can be weakened under low-light conditions. Bridge damage often contains fine and low-contrast morphological cues, including thin cracks, spalling boundaries, exposed rebar edges, corrosion contours, and local texture discontinuities. To enhance such structure-sensitive information, the expert combines learnable local diference extraction with spatially adaptive structural refinement.

Structural Feature Projection. The input feature $F _ { i }$ is first projected into a compact structural representation with half of the original channel dimension:

$$
F _ { p } = \delta \Big ( C _ { 1 \times 1 } ^ { \mathrm { p r o j } } ( F _ { i } ) \Big ) \in \mathbb { R } ^ { H _ { i } \times W _ { i } \times C _ { i } / 2 } ,\tag{21}
$$

where $\delta ( \cdot )$ denotes GELU. This projection reduces channel redundancy and provides a compact representation for subsequent local structure extraction.

Learnable Diference Extraction. Two independent depthwise diference branches are employed to capture complementary local structural variations:

$$
D _ { x } = \eta \big ( D _ { 3 \times 3 } ^ { x } ( F _ { p } ) - F _ { p } \big ) , \qquad D _ { y } = \eta \Big ( D _ { 3 \times 3 } ^ { y } ( F _ { p } ) - F _ { p } \Big ) ,\tag{22}
$$

where $\eta ( \cdot )$ denotes LeakyReLU and $\mathcal { D } _ { 3 \times 3 } ^ { x }$ and $\mathcal { D } _ { 3 \times 3 } ^ { y }$ denote two independently parameterized depthwise convolutional filters. Subtracting the input feature from each convolutional response suppresses slowly varying components and emphasizes local diferential responses associated with boundaries and texture variations. Unlike fixed gradient operators, the learnable diference branches can adapt their local response patterns to structural characteristics captured during training.

Structural Reconstruction and Spatial Gating. The two diferential responses are concatenated along the channel dimension and reconstructed through a $1 \times 1 { \mathrm { - G E L U } } - 3 \times 3$ transformation:

$$
F _ { s } = C _ { 3 \times 3 } ^ { \mathrm { r e c } } \big ( \delta \big ( C _ { 1 \times 1 } ^ { \mathrm { f u s } } \big ( \mathrm { C o n c a t } ( D _ { x } , D _ { y } ) \big ) \big ) \big ) ,\tag{23}
$$

where $c _ { 1 \times 1 } ^ { \mathrm { f u s } }$ integrates the complementary diference responses across channels and $C _ { 3 \times 3 } ^ { \mathrm { r e c } }$ further aggregates local spatial context to reconstruct structure-sensitive features.

A spatial gating map is subsequently generated to adaptively emphasize informative structural responses:

$$
A _ { s } = \sigma \Big ( C _ { 3 \times 3 } ^ { \mathrm { g a t e } } ( F _ { s } ) \Big ) \in [ 0 , 1 ] ^ { H _ { i } \times W _ { i } \times 1 } ,\tag{24}
$$

where $\sigma ( \cdot )$ denotes the Sigmoid function. The single-channel gating map $A _ { s }$ is broadcast across feature channels to modulate the reconstructed structural representation. The final output of the expert is obtained through gated residual fusion:

$$
F _ { d } = F _ { s } \odot A _ { s } + F _ { i } .\tag{25}
$$

By combining learnable diference extraction, structural reconstruction, and spatially adaptive gating, the Detail Enhancement Expert strengthens boundary- and texture-sensitive responses while preserving the underlying feature representation through the residual connection. The resulting $F _ { d }$ constitutes the structure-oriented representation used in adaptive expert fusion.

## 3.2.4. Adaptive Gating Network

The three expert branches focus on diferent restoration objectives and therefore need not contribute equally under diferent degradation conditions. The relative importance of noise suppression, illumination correction, and structuraldetail recovery can vary across input images and feature scales. A fixed combination of the three expert outputs is therefore insuficient to fully exploit their complementary specialization. The Adaptive Gating Network addresses this issue by estimating sample-adaptive routing weights and dynamically integrating the expert representations.

Since $F _ { n }$ incorporates the scale-aligned degradation guidance $G _ { i } ,$ it is used to construct the routing descriptor. Global average pooling first aggregates its spatial responses into a compact representation:

$$
z = \mathbf { G A P } ( F _ { n } ) \in \mathbb { R } ^ { C _ { i } } .\tag{26}
$$

The descriptor is subsequently processed by two successive $1 \times 1$ convolutions with an intermediate ReLU activation to generate the routing logits:

$$
r = C _ { 1 \times 1 } ^ { \mathrm { r o u t } } \big ( \rho \big ( C _ { 1 \times 1 } ^ { \mathrm { r i n } } ( z ) \big ) \big ) \in \mathbb { R } ^ { 3 } ,\tag{27}
$$

where $\rho ( \cdot )$ denotes ReLU, $c _ { 1 \times 1 } ^ { \mathrm { r i n } }$ reduces the channel dimension to $C _ { i } / 2$ , and $ { c _ { 1 \times 1 } } ^ { \mathrm { r o u t } }$ projects the compact descriptor to three expert-specific logits.

The routing logits are normalized using Softmax to obtain the expert weights:

$$
[ w _ { n } , w _ { b } , w _ { d } ] = \mathrm { S o f t m a x } ( r ) ,\tag{28}
$$

where $w _ { n } , w _ { c }$ , and $w _ { d }$ correspond to the Noise-Aware Expert, Color Adjustment Expert, and Detail Enhancement Expert, respectively, and satisfy

$$
w _ { n } + w _ { b } + w _ { d } = 1 , \qquad w _ { n } , w _ { b } , w _ { d } \geq 0 .\tag{29}
$$

Finally, the output of the Adaptive Mixture-of-Experts Restoration Module is obtained by weighted fusion of the three parallel expert representations:

$$
F _ { \mathrm { o u t } } = w _ { n } F _ { n } + w _ { b } F _ { b } + w _ { d } F _ { d } .\tag{30}
$$

Each scalar routing weight is broadcast across the spatial and channel dimensions of its corresponding expert feature. Through this adaptive fusion mechanism, the module dynamically balances noise suppression, illumination and chromatic adjustment, and structural-detail recovery according to the degradation characteristics of the current input representation, yielding the restored feature $F _ { \mathrm { o u t } }$ for subsequent feature propagation in the restoration network.

## 4. Low-light Data Synthesis Pipeline

Acquiring strictly paired normal-light and low-light images under identical viewpoints is challenging in real-world scenarios. Therefore, we adopt an ISP-aware degradation pipeline to synthesize low-light observations from normallight images. Unlike direct intensity attenuation in the sRGB domain, this pipeline first unprocesses a normal-light image into a RAW-like linear representation, introduces low-light degradation in the sensor domain, and then renders the corrupted signal back to the sRGB space. This procedure better approximates the physical image formation process under low illumination.

Given a normal-light image $I ^ { n } \in [ 0 , 1 ] ^ { H \times W \times 3 }$ , the synthetic low-light image $I ^ { l }$ is generated as

$$
\begin{array} { r } { I ^ { l } = \mathcal { P } _ { \mathrm { I S P } } \left( \mathcal { D } _ { \mathrm { r a w } } \left( \mathcal { P } _ { \mathrm { I S P } } ^ { - 1 } ( I ^ { n } ) \right) \right) , } \end{array}\tag{31}
$$

where $ { \mathcal { P } } _ { \mathrm { I S P } } ^ { - 1 } ( \cdot )$ denotes the inverse ISP unprocessing operation, $D _ { \mathrm { r a w } } ( \cdot )$ represents RAW-domain low-light corruption, and $\mathcal { P } _ { \mathrm { I S P } } ( \cdot )$ denotes the forward ISP rendering process.

RGB-to-RAW unprocessing. The normal-light sRGB image is first mapped back to a RAW-like sensor representation by approximately reversing the camera rendering process. Following the convention of [20], we adopt a threechannel RAW-like linear representation instead of a single-channel Bayer mosaic, which allows color transformations to operate directly on RGB triples. The unprocessing proceeds by inverting the four standard ISP stages in reverse order:

(i) Inverse tone mapping. The nonlinear sRGB signal is first linearized as

$$
I _ { t } = \frac { 1 } { 2 } - \sin \left( \frac { \sin ^ { - 1 } ( 1 - 2 I ^ { n } ) } { 3 } \right) ,\tag{32}
$$

where $I _ { t }$ denotes the inverse-tone-mapped image.

(ii) Inverse gamma correction. The linear RGB signal is then recovered via

$$
I _ { \mathrm { l i n } } = \operatorname* { m a x } ( I _ { t } , \epsilon ) ^ { \gamma } ,\tag{33}
$$

where $\epsilon = 1 0 ^ { - 8 }$ is used for numerical stability and $\gamma \sim \mathcal { V } ( 2 . 0 , 3 . 5 )$ .

(iii) Color space conversion (RGB → camera RGB). To account for camera-dependent color responses, a camera color correction matrix is randomly selected from a predefined pool and combined with the standard RGB-to-XYZ matrix to obtain a row-normalized RGB-to-camera matrix $M _ { \mathrm { r g b \to c a m } }$ . For each pixel location $p ,$ the camera-domain image is computed as

$$
I _ { \mathrm { c a m } } ( p ) = M _ { \mathrm { r g b \to c a m } } I _ { \mathrm { l i n } } ( p ) .\tag{34}
$$

(iv) White balance removal. Finally, the camera-applied white balance is removed to obtain the RAW-like signal:

$$
I _ { \mathrm { r a w } } = I _ { \mathrm { c a m } } \odot \mathbf { g } _ { \mathrm { w b } } , \quad \mathbf { g } _ { \mathrm { w b } } = g _ { \mathrm { r g b } } \cdot \left[ \frac { 1 } { g _ { r } } , 1 , \frac { 1 } { g _ { b } } \right] ^ { \top } ,\tag{35}
$$

where ⊙ denotes element-wise multiplication, $\mathbf { g } _ { \mathrm { w b } } \in \mathbb { R } ^ { 3 }$ is the channel-wise gain vector, $g _ { r } \sim \mathcal { V } ( 1 . 9 , 2 . 4 )$ $g _ { b } \sim$ $\mathcal { V } ( 1 . 5 , 1 . 9 )$ , and $g _ { \mathrm { r g b } } \sim \mathcal { N } ( 0 . 8 , 0 . 1 ^ { 2 } )$ .

RAW-domain low-light corruption. We first attenuate the exposure while introducing spatially varying illumination to simulate non-uniform underexposure caused by structural occlusion and local shadows:

$$
I _ { \mathrm { d a r k } } = k L \odot I _ { \mathrm { r a w } } ,\tag{36}
$$

where the global exposure ratio � is sampled from a truncated normal distribution within [0.01, 0.1], with mean 0.1 and standard deviation 0.08. Here, $\boldsymbol { L } \in \mathbb { R } ^ { \hat { H } \times W \times 1 }$ denotes a smooth spatial illumination field that models locally varying exposure conditions. Specifically, a low-resolution random field is first generated, interpolated to the image resolution, and Gaussian-smoothed to produce gradual illumination variations. The resulting field is rescaled to [0.5, 1.5] and normalized around unity, such that � controls the overall exposure level while � introduces spatially heterogeneous underexposure. Sensor noise is then injected by combining signal-dependent shot noise and signal-independent read noise:

$$
\widetilde { I } _ { \mathrm { r a w } } = I _ { \mathrm { d a r k } } + n , \quad n \sim \mathcal { N } \left( 0 , \lambda _ { s } I _ { \mathrm { d a r k } } + \lambda _ { r } \right) ,\tag{37}
$$

where $\lambda _ { s }$ and $\lambda _ { r }$ denote the shot-noise and read-noise levels, respectively. Since both variances must be positive and span several orders of magnitude, they are sampled in the logarithmic domain. The shot-noise level is drawn as

$$
u _ { s } \sim \mathcal { V } \left( \log ( 1 0 ^ { - 4 } ) , \log ( 1 . 2 \times 1 0 ^ { - 2 } ) \right) , \quad \lambda _ { s } = \exp ( u _ { s } ) ,\tag{38}
$$

and the read-noise level is generated from a log-linear camera noise model conditioned on $u _ { s } \mathrm { : }$

$$
u _ { r } = 2 . 1 8 u _ { s } + 1 . 2 0 + \xi , \quad \xi \sim \mathcal { N } ( 0 , 0 . 2 6 ^ { 2 } ) , \quad \lambda _ { r } = \exp ( u _ { r } ) .\tag{39}
$$

This conditional sampling preserves the empirical correlation between shot and read noise while guaranteeing $\lambda _ { s } > 0$ and $\lambda _ { r } > 0$

RAW-to-RGB rendering. After RAW-domain corruption, the degraded signal is rendered back to the sRGB space through the forward ISP pipeline. Quantization noise is first added to simulate analog-to-digital conversion:

$$
I _ { q } = \widetilde { I } _ { \mathrm { r a w } } + q , \quad q \sim \mathcal { V } \left( - \frac { 1 } { 2 ^ { B + 1 } } , \frac { 1 } { 2 ^ { B + 1 } } \right) ,\tag{40}
$$

where $B \in \{ 1 2 , 1 4 , 1 6 \}$ denotes the simulated bit depth. The forward ISP $\mathcal { P } _ { \mathrm { I S P } } ( \cdot )$ then applies the four reciprocal operations of the unprocessing stage in forward order: (i) white balance restoration, (ii) color space conversion via $M _ { \mathrm { c a m \to r g b } } = M _ { \mathrm { r g b \to c a m } } ^ { - 1 } .$ , (iii) gamma encoding with exponent $1 / \gamma$ , and (iv) tone mapping, followed by clipping to the valid range:

$$
\begin{array} { r } { I ^ { l } = \mathrm { c l i p } \left( \mathcal { P } _ { \mathrm { I S P } } ( I _ { q } ) , 0 , 1 \right) . } \end{array}\tag{41}
$$

The rendered image $I ^ { l } \in [ 0 , 1 ] ^ { H \times W \times 3 }$ serves as the synthetic low-light counterpart of $I ^ { n }$ for model training and evaluation.

## 5. Experiments and Geometric Evaluation

This section presents the experimental setup and evaluates the proposed DaL-MoE in terms of image enhancement quality and downstream bridge damage detection and instance segmentation performance. We first describe the experimental platform, datasets, implementation details, and evaluation protocols, and then present quantitative and qualitative comparisons, ablation studies, and real-world evaluations.

## 5.1. Experimental platform

All experiments were conducted on a workstation equipped with an AMD Ryzen 9 9950X central processing unit (CPU), 96 GB of random access memory (RAM), and an NVIDIA GeForce RTX 5090 graphics processing unit (GPU) with 32 GB of video random access memory (VRAM), running the Linux operating system.

Panel B. Class-wise instance distribution across the training and test sets  
Table 2  
Overview and class-wise instance distribution of the UAV bridge defect dataset and paired low-light benchmark Panel A. Dataset acquisition, partitioning, annotation, and benchmark construction
<table><tr><td>Item</td><td>Content</td></tr><tr><td>Field acquisition</td><td></td></tr><tr><td>Number of bridges</td><td>7</td></tr><tr><td>Inspection contexts</td><td>4, including urban viaducts, mountainous bridges, railway bridges, and long-span highway bridges</td></tr><tr><td>Acquisition system</td><td>UAV equipped with a DJI H30T imaging system</td></tr><tr><td>Original image resolution</td><td>4032 × 3024 pixels</td></tr><tr><td>Dataset composition and partitioning</td><td></td></tr><tr><td>Normal-light source images</td><td>2,895</td></tr><tr><td>Train / test split</td><td>2,413 / 482 images</td></tr><tr><td>Annotated defect instances</td><td>19,582, including 15,731 training instances and 3,851 test instances</td></tr><tr><td>Defect categories Spatial partitioning</td><td>Spalling, Crack, Hole, Honeycombing, Seepage, Rebar Corrosion, Rust, and Efflorescence</td></tr><tr><td></td><td>Group-wise partition using mutually exclusive bridge spans or mileage sections rather than random image-level sampling</td></tr><tr><td>Leakage control</td><td>Adjacent or overlapping views from the same span or mileage section were assigned to one subset; the</td></tr><tr><td colspan="2">Annotation and paired benchmark construction</td></tr><tr><td>Manual annotation source Derived task annotations</td><td>Closed polygon contours manually delineated for individual defect instances</td></tr><tr><td></td><td>Axis-aligned bounding boxes computed from polygon-coordinate extrema; instance masks generated by rasterizing and filling the same polygons</td></tr><tr><td>Paired low-light benchmark</td><td>2,895 spatially aligned normal-light-low-light pairs generated using the ISP-aware synthesis pipeline, with annotations shared across both illumination conditions</td></tr><tr><td>Supported evaluations</td><td>Full-reference image enhancement, object detection, and instance segmentation</td></tr></table>

<table><tr><td>Class ID</td><td>Class name</td><td>Train (2,413 imgs)</td><td>Test (482 imgs)</td><td>Total (2,895 imgs)</td></tr><tr><td>0</td><td>Spalling</td><td>3,568 (22.68%)</td><td>809 (21.01%)</td><td>4,377 (22.35%)</td></tr><tr><td>1</td><td>Crack</td><td>1,373 (8.73%)</td><td>456 (11.84%)</td><td>1,829 (9.34%)</td></tr><tr><td></td><td>Hole</td><td>4,315 (27.43%)</td><td>1,043 (27.08%)</td><td>5,358 (27.36%)</td></tr><tr><td>23</td><td>Honeycombing</td><td>484 (3.08%)</td><td>185 (4.80%)</td><td>669 (3.42%)</td></tr><tr><td>4</td><td>Seepage</td><td>745 (4.74%)</td><td>200 (5.19%)</td><td>945 (4.83%)</td></tr><tr><td>5</td><td>Rebar corrosion</td><td>1,695 (10.77%)</td><td>362 (9.40%)</td><td>2,057 (10.50%)</td></tr><tr><td>6</td><td>Rust</td><td>845 (5.37%)</td><td>239 (6.21%)</td><td>1,084 (5.54%)</td></tr><tr><td>7</td><td>Efflorescence</td><td>2,706 (17.20%)</td><td>557 (14.46%)</td><td>3,263 (16.66%)</td></tr><tr><td>Total</td><td></td><td>15,731 (100.00%)</td><td>3,851 (100.00%)</td><td>19,582 (100.00%)</td></tr></table>

Note: Image counts and split sizes refer to the normal-light source images. One spatially aligned synthetic low-light counterpart was generated for each source image. Training and test images were assigned according to mutually exclusive bridge spans or mileage sections, and adjacent or spatially overlapping UAV views were not shared across subsets. Percentages in Panel B indicate column-wise class proportions.

## 5.2. Datasets

A total of 2,895 images were collected from real-world UAV bridge inspection scenarios covering seven bridges and spatially distinct bridge spans and mileage sections. All images were acquired using a UAV equipped with a DJI H30T imaging system and have an original resolution of 4032 × 3024 pixels. To reduce information leakage caused by adjacent viewpoints or repeated local scenes, the dataset was partitioned according to spatially disjoint bridge spans or mileage sections rather than through random image-level sampling. Adjacent or overlapping views from the same span or mileage section were assigned to the same subset. The final dataset contains 2,413 training images and 482 test images.

Eight representative bridge surface defect categories were considered: spalling, crack, hole, honeycombing, seepage, rebar corrosion, rust, and eflorescence. Each defect instance was manually delineated as a closed polygon contour, from which the axis-aligned bounding box and instance segmentation mask were automatically derived. This procedure ensured spatial consistency between the detection and segmentation annotations. As summarized in Table ??, the dataset contains 19,582 annotated defect instances, including 15,731 instances (80.33%) in the training set and 3,851 instances (19.67%) in the test set. Hole is the most frequent category in both subsets, accounting for 27.43% of the training instances and 27.08% of the test instances, with an overall proportion of 27.36%. It is followed by spalling, which accounts for 22.68% of the training set, 21.01% of the test set, and 22.35% of all instances, and eflorescence, which accounts for 17.20%, 14.46%, and 16.66%, respectively. The remaining categories account for 10.50% (rebar corrosion), 9.34% (crack), 5.54% (rust), 4.83% (seepage), and 3.42% (honeycombing) of all annotated instances. Thus, the training and test sets exhibit broadly comparable, although not identical, class-wise distributions.

Because strictly paired normal-light and low-light images are dificult to acquire in real-world inspection scenarios, an ISP-aware low-light synthesis pipeline was applied to the normal-light images to generate spatially aligned lowlight counterparts. The dataset partition was established before synthesis, such that each source image and its derived low-light image remained in the same subset. Since the synthesis process does not alter the damage locations or scene geometry, the low-light images directly inherit the bounding-box and instance-mask annotations of their corresponding normal-light images. As illustrated in Fig. 3, the normal-light source images and their synthetic low-light counterparts retain identical defect locations, instance masks, and axis-aligned bounding boxes. The resulting image pairs were used for training and full-reference evaluation of the image enhancement model, whereas the synthesized low-light test images and their shared annotations were used for controlled evaluation of object detection and instance segmentation under low-light conditions.

![](images/1d62bc43855d806a264213b5f4cf6147e4fb3f1a1f3b7bd80172c52a62522258.jpg)  
Figure 3: Visualization of defect annotations shared between paired normal-light and synthetic low-light images.

## 5.3. Implementation details

We implement DaL-MoE using PyTorch. For a controlled comparison, all image enhancement methods are trained under a unified protocol. Specifically, each model is optimized using Adam for $3 \bar { \times } 1 0 ^ { 5 }$ iterations. The initial learning rate is set to $2 \times 1 0 ^ { - 4 }$ and gradually decayed to $1 \times 1 0 ^ { - 6 }$ following a cosine annealing schedule. Prior to training, ofline data augmentation is performed on the spatially aligned low-light and normal-light image pairs. Random rotations and horizontal and vertical flips are synchronously applied to each input–reference pair to preserve their spatial correspondence, expanding the original 2,413 training image pairs to a total of 12,065 paired training samples. During training, the augmented image pairs are randomly cropped into patches of size 256 × 256, with a batch size of 8. The training objective is to minimize the mean absolute error between the enhanced image and its corresponding normal-light reference image. The downstream detectors are trained and evaluated with an input resolution of 640 × 640. The architecture configuration of DaL-MoE and the unified training settings of the compared image enhancement methods and downstream detector architectures are summarized in Table 3.

## 5.4. Evaluation Protocol and Metrics

To assess restoration fidelity, downstream bridge damage recognition utility, and transfer to real inspection scenes, three complementary evaluation protocols were adopted. All quantitative evaluations on the paired synthetic benchmark were conducted on the held-out test set containing 482 images. The training–test partition was established before low-light synthesis, ensuring that each normal-light source image and its low-light counterpart remained in the same subset. All compared image enhancement methods were evaluated using the same training and test data.

Table 3  
Panel A. Architecture configuration of the proposed DaL-MoE  
Architecture of the proposed DaL-MoE and unified training configurations of the compared low-light image enhancement methods and downstream detector architectures.
<table><tr><td>Block</td><td>Input size</td><td>Output size</td></tr><tr><td>Light-up image I0</td><td> $\overline { { H \times W \times 3 } }$ </td><td> $\overline { { H \times W \times 3 } }$ </td></tr><tr><td>Feature embedding (3 × 3 Conv)</td><td> $H \times W \times 3$ </td><td> $H \times W \times 4 0$ </td></tr><tr><td>MoE Block (Encoder-1)</td><td> $H \times W \times 4 0$ </td><td> $H \times W \times 4 0$ </td></tr><tr><td>Downsampling</td><td> $H \times W \times 4 0$ </td><td> $H / 2 \times W / 2 \times 8 0$ </td></tr><tr><td>MoE Block (Encoder-2)</td><td> $H / 2 \times W / 2 \times 8 0$ </td><td> $\dot { H / 2 } \times \dot { W / 2 } \times 8 0$ </td></tr><tr><td>Downsampling</td><td> $H / 2 \times W / 2 \times 8 0$ </td><td> $H / 4 \times W \dot { / } 4 \times 1 6 0$ </td></tr><tr><td>MoE Block (Bottleneck)</td><td> $H / 4 \times W / 4 \times 1 6 0$ </td><td> $H ^ { ' } / 4 \times W ^ { ' } / 4 \times 1 6 0$ </td></tr><tr><td>Upsampling</td><td> $\dot { H / 4 } \times \dot { W / 4 } \times 1 6 0$ </td><td> $\dot { H } / 2 \times \dot { W } / 2 \times 8 0$ </td></tr><tr><td>MoE Block (Decoder-2)</td><td> $\dot { H } / 2 \times \dot { W } / 2 \times 8 0$ </td><td> $H / 2 \times W / 2 \times 8 0$ </td></tr><tr><td>Upsampling</td><td> $\dot { H / 2 } \times \dot { W / 2 } \times 8 0$ </td><td> $\dot { H } \times W \times 4 0$ </td></tr><tr><td>MoE Block (Decoder-1)</td><td> $\dot { H } \times \dot { W } \times 4 0$   $H \times W \times 4 0$ </td><td> $H \times W \times 4 0$ </td></tr><tr><td>Output projection and residual reconstruction</td><td></td><td> $H \times W \times 3$ </td></tr></table>

Panel B. Principal training settings of the compared low-light image enhancement methods
<table><tr><td rowspan="2">Method</td><td colspan="3">Training schedule</td><td colspan="4">Optimization</td></tr><tr><td>Patch size</td><td>Batch</td><td>Iterations</td><td>Optimizer</td><td> $\scriptstyle L R _ { 0 }$ </td><td> $L R _ { \mathrm { m i n } }$ </td><td>Schedule</td></tr><tr><td>MIRNet [27]</td><td>256×256</td><td>8</td><td> $3 \times 1 0 ^ { 5 }$ </td><td>Adam</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Cosine</td></tr><tr><td>Restormer [28]</td><td> $2 5 6 \times 2 5 6$ </td><td>8</td><td> $3 \times 1 0 ^ { 5 }$ </td><td>Adam</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Cosine</td></tr><tr><td>LLFormer [36]</td><td> $2 5 6 \times 2 5 6$ </td><td>8</td><td> $3 \times 1 0 ^ { 5 }$ </td><td>Adam</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Cosine</td></tr><tr><td>FourLLIE [37]</td><td> $2 5 6 \times 2 5 6$ </td><td>8</td><td> $3 \times 1 0 ^ { 5 }$ </td><td>Adam</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Cosine</td></tr><tr><td>Retinexformer [30]</td><td> $2 5 6 \times 2 5 6$ </td><td>8</td><td> $3 \times 1 0 ^ { 5 }$ </td><td>Adam</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Cosine</td></tr><tr><td>UHDformer [38]</td><td> $2 5 6 \times 2 5 6$ </td><td>8</td><td> $3 \times 1 0 ^ { 5 }$ </td><td>Adam</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Cosine</td></tr><tr><td>HVI-CIDNet [31]</td><td> $2 5 6 \times 2 5 6$ </td><td>8</td><td> $3 \times 1 0 ^ { 5 }$ </td><td>Adam</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$ </td><td>Cosine</td></tr><tr><td>DarkIR [32] DaL-MoE</td><td> $2 5 6 \times 2 5 6$   $2 5 6 \times 2 5 6$ </td><td>8 8</td><td> $3 \times 1 0 ^ { 5 }$   $\overline { { 3 \times 1 0 ^ { 5 } } }$ </td><td>Adam Adam</td><td> $2 \times 1 0 ^ { - 4 }$   $2 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 6 }$   $1 \times 1 0 ^ { - 6 }$ </td><td>Cosine Cosine</td></tr></table>

Panel C. Principal training settings of the downstream detector architectures
<table><tr><td rowspan="2">Detector</td><td colspan="3">Training schedule</td><td colspan="5">Optimization</td></tr><tr><td>Input size</td><td>Effective batch</td><td>Epochs</td><td>Optimizer</td><td> $\scriptstyle L R _ { 0 }$ </td><td>Momentum</td><td>WD</td><td>Schedule</td></tr><tr><td>YOLOv8m [39]</td><td> $6 4 0 \times 6 4 0$ </td><td>16</td><td>200</td><td>SGD</td><td>0.01</td><td>0.937</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>Cosine</td></tr><tr><td>YOLOv11m [40]</td><td> $6 4 0 \times 6 4 0$ </td><td>16</td><td>200</td><td>SGD</td><td>0.01</td><td>0.937</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>Cosine</td></tr><tr><td>YOLOv8n [39]</td><td> $6 4 0 \times 6 4 0$ </td><td>16</td><td>200</td><td>SGD</td><td>0.01</td><td>0.937</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>Cosine</td></tr><tr><td>YOLOv11n [40]</td><td>640× 640</td><td>16</td><td>200</td><td>SGD</td><td>0.01</td><td>0.937</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>Cosine</td></tr><tr><td>SCNet [41]</td><td>640×640</td><td>16</td><td>200</td><td>SGD</td><td>0.01</td><td>0.900</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>Cosine</td></tr></table>

� and � denote spatial dimensions; $L R _ { 0 } , \ L R _ { \operatorname* { m i n } } ,$ and WD denote the initial learning rate, terminal learning rate, and weight decay, respectively. All enhancement methods use the same paired training set and <sup>0</sup> <sup>min</sup>                         augmentation protocol. All detectors are trained on the same normal-light split and evaluated using fixed checkpoints without enhancement-specific fine-tuning. The YOLO entries denote instance-segmentation variants providing both box and mask predictions.

Full-reference image enhancement evaluation. On the paired synthetic test set, peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM) were used to compare each enhanced image with its spatially aligned normal-light reference. Following Retinexformer [30], both metrics were calculated on the corresponding RGB image pairs. Let �<sup>̂</sup> and $I ^ { n }$ denote an enhanced image and its normal-light reference, respectively, each with a spatial resolution of $H \times W$ and three color channels. Their mean squared error is defined as

$$
\mathrm { M S E } ( \hat { I } , I ^ { n } ) = \frac { 1 } { 3 H W } \sum _ { p = 1 } ^ { H W } \sum _ { c = 1 } ^ { 3 } \left( \hat { I } _ { p , c } - I _ { p , c } ^ { n } \right) ^ { 2 } .\tag{42}
$$

PSNR is subsequently calculated as

$$
\mathrm { P S N R } ( \hat { I } , I ^ { n } ) = 1 0 \log _ { 1 0 } \left( \frac { L ^ { 2 } } { \mathrm { M S E } ( \hat { I } , I ^ { n } ) } \right) ,\tag{43}
$$

where � denotes the maximum possible pixel value; � = 1 for images normalized to [0, 1]. PSNR is reported in decibels (dB), with a higher value indicating lower pixel-wise reconstruction error.

For corresponding local windows � and � from the enhanced and reference images, respectively, SSIM is defined as

$$
\mathrm { S S I M } ( x , y ) = \frac { ( 2 \mu _ { x } \mu _ { y } + C _ { 1 } ) ( 2 \sigma _ { x y } + C _ { 2 } ) } { ( \mu _ { x } ^ { 2 } + \mu _ { y } ^ { 2 } + C _ { 1 } ) ( \sigma _ { x } ^ { 2 } + \sigma _ { y } ^ { 2 } + C _ { 2 } ) } ,\tag{44}
$$

where $\mu _ { x }$ and $\mu _ { y }$ are the local means, $\sigma _ { x } ^ { 2 }$ and $\sigma _ { y } ^ { 2 }$ are the local variances, and $\sigma _ { x y }$ is the local covariance. The stabilizing constants are $C _ { 1 } = ( 0 . 0 1 L ) ^ { 2 }$ and $C _ { 2 } = ( 0 . 0 3 L ) ^ { 2 }$ . The image-level SSIM is obtained by averaging the local-window scores. A higher SSIM indicates greater consistency in luminance, contrast, and structural information. The reported PSNR and SSIM values are arithmetic means over all test image pairs.

Downstream recognition protocol. To isolate the efect of image enhancement on downstream recognition, each object detection or instance segmentation model was trained once on the normal-light training split and then held fixed for all input conditions. No enhancement-specific fine-tuning of the downstream models was performed. Each model was evaluated under three input settings: Normal, denoting the corresponding normal-light test images as a normalillumination reference; $B a s e ,$ , denoting direct inference on the synthesized low-light test images without enhancement; and Enhanced, denoting inference on the outputs of the evaluated enhancement methods. The Normal setting was not included in the ranking of enhancement methods. The same test images, annotations, detector checkpoints, and inference settings were used across all input conditions; therefore, diferences between Base and Enhanced primarily reflect the influence of the restoration front end.

Object detection and instance segmentation metrics. For a predicted region � and a ground-truth region �, intersection over union (IoU) is defined as

$$
\operatorname { I o U } ( A , B ) = { \frac { | A \cap B | } { | A \cup B | } } .\tag{45}
$$

For object detection, � and � denote the predicted and ground-truth bounding boxes, whereas for instance segmentation they denote the predicted and ground-truth masks. At a specified IoU threshold, a prediction is counted as a true positive (TP) when it has the correct class and matches an unmatched ground-truth instance with an IoU not lower than the threshold. Duplicate or unmatched predictions are counted as false positives (FP), and unmatched ground-truth instances are counted as false negatives (FN). Precision and recall are defined as

$$
\mathrm { P r e c i s i o n } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } } , \qquad \mathrm { R e c a l l } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } .\tag{46}
$$

Varying the prediction-confidence threshold produces a precision–recall curve, and the area under its interpolated form gives the average precision (AP) for a given class and IoU threshold. Let $C = 8$ denote the number of defect categories and let $\mathsf { A P } _ { c , \tau }$ denote the AP of class � at IoU threshold �. The mean AP at an IoU threshold of 0.50 is

$$
\operatorname* { m A P } _ { 5 0 } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathrm { A P } _ { c , 0 . 5 0 } .\tag{47}
$$

The stricter multi-threshold metric is defined as

$$
\mathrm { m A P } _ { 5 0 : 9 5 } = \frac { 1 } { C | \mathcal { T } | } \sum _ { \tau \in \mathcal { T } } \sum _ { c = 1 } ^ { C } \mathrm { A P } _ { c , \tau } , \qquad \mathcal { T } = \{ 0 . 5 0 , 0 . 5 5 , \dots , 0 . 9 5 \} ,\tag{48}
$$

where $| \tau | = 1 0$ . Thus, $\mathrm { m A P } _ { 5 0 : 9 5 }$ averages AP over all eight defect categories and ten IoU thresholds ranging from 0.50 to 0.95 in increments of 0.05. We report box $\mathrm { \ m A P { 5 0 } }$ and box $\mathrm { m A P } _ { 5 0 : 9 5 }$ using bounding-box IoU, together with mask $\mathrm { m A P } _ { 5 0 }$ and mask $\mathrm { m A P } _ { 5 0 : 9 5 }$ using mask IoU. Higher values indicate better classification and localization or instance-boundary delineation performance.

Real-world evaluation. The 200 real-world nighttime UAV images do not have paired normal-light references; consequently, full-reference PSNR and SSIM cannot be computed. Dataset-averaged intensity and Canny edge density were therefore used as auxiliary descriptors of visibility changes after enhancement. Let $Y _ { i }$ denote the grayscale intensity image of the �-th sample. Its mean intensity is

$$
\mu _ { i } = \frac { 1 } { H _ { i } W _ { i } } \sum _ { p = 1 } ^ { H _ { i } W _ { i } } Y _ { i } ( p ) .\tag{49}
$$

The dataset-level mean intensity is obtained by averaging $\mu _ { i }$ over all real-world images. Let $E _ { i }$ denote the binary Canny edge map generated using fixed parameters, where edge and non-edge pixels take values of 1 and $^ { 0 , }$ respectively. The edge density is defined as

$$
d _ { i } = \frac { 1 } { H _ { i } W _ { i } } \sum _ { p = 1 } ^ { H _ { i } W _ { i } } E _ { i } ( p ) .\tag{50}
$$

The same grayscale conversion and Canny settings were used for the original and enhanced images. The downstream comparison was also performed using the same fixed YOLOv11m checkpoint and inference settings. Mean intensity and edge density are descriptive rather than full-reference fidelity measures: increased brightness does not necessarily indicate more accurate restoration, and increased edge density may include residual noise or enhancement artifacts. These quantities were therefore excluded from the primary method ranking, and the real-world results were used as complementary evidence of synthetic-to-real transfer.

## Table 4

Comparison of representative low-light image enhancement methods.
<table><tr><td>Method</td><td>Venue</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td></tr><tr><td>MIRNet [27]</td><td>ECCV&#x27;20</td><td>20.2218</td><td>0.7903</td></tr><tr><td>Restormer [28]</td><td>CVPR&#x27;22</td><td>20.6620</td><td>0.7624</td></tr><tr><td>LLFormer [36]</td><td>AAAI&#x27;23</td><td>20.7207</td><td>0.7717</td></tr><tr><td>FourLLIE [37]</td><td>ACM MM&#x27;23</td><td>21.1467</td><td>0.7490</td></tr><tr><td>Retinexformer [30]</td><td>ICCV&#x27;23</td><td>21.9448</td><td>0.7565</td></tr><tr><td>UHDformer [38]</td><td>AAAI&#x27;24</td><td>21.3763</td><td>0.7183</td></tr><tr><td>HVI-CIDNet [31]</td><td>CVPR&#x27;25</td><td>21.6674</td><td>0.7458</td></tr><tr><td>DarkIR [32]</td><td>CVPR&#x27;25</td><td>21.6787</td><td>0.7598</td></tr><tr><td>Ours</td><td></td><td>23.1174</td><td>0.8482</td></tr></table>

Best results are shown in bold. The green row highlights the proposed method.

Table 5  
Downstream bridge damage detection and instance segmentation performance across diferent detector architectures and image enhancement methods.
<table><tr><td rowspan="2">Detector</td><td rowspan="2">Task</td><td rowspan="2">Metric</td><td colspan="2">Reference Inputs</td><td colspan="4">Enhancement Methods</td></tr><tr><td>Base</td><td>Normal</td><td>Retinexformer [30]</td><td>HVI-CIDNet [31]</td><td>DarkIR [32]</td><td>Ours</td></tr><tr><td rowspan="4">YOLOv8m [39]</td><td>Box</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.2787</td><td>0.6710</td><td>0.4336</td><td>0.4193</td><td>0.3908</td><td>0.4806</td></tr><tr><td>detection</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.1837</td><td>0.4744</td><td>0.2745</td><td>0.2656</td><td>0.2466</td><td>0.3341</td></tr><tr><td>Mask</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.1985</td><td>0.3913</td><td>0.2721</td><td>0.2704</td><td>0.2291</td><td>0.2863</td></tr><tr><td>segmentation</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.0853</td><td>0.1644</td><td>0.1169</td><td>0.1105</td><td>0.0953</td><td>0.1251</td></tr><tr><td rowspan="4">YOLOv11m [40]</td><td>Box</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.3097</td><td>0.7723</td><td>0.4495</td><td>0.4232</td><td>0.3140</td><td>0.4923</td></tr><tr><td>detection</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.2105</td><td>0.5855</td><td>0.2899</td><td>0.2701</td><td>0.1904</td><td>0.3578</td></tr><tr><td>Mask</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.2281</td><td>0.5236</td><td>0.2985</td><td>0.2836</td><td>0.2061</td><td>0.3529</td></tr><tr><td>segmentation</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.1051</td><td>0.2409</td><td>0.1260</td><td>0.1209</td><td>0.0844</td><td>0.1655</td></tr><tr><td rowspan="4">YOLOv8n [39]</td><td>Box</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.3267</td><td>0.5720</td><td>0.3611</td><td>0.3606</td><td>0.3099</td><td>0.4167</td></tr><tr><td>detection</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.2210</td><td>0.3759</td><td>0.2169</td><td>0.2191</td><td>0.1815</td><td>0.2718</td></tr><tr><td>Mask</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.2161</td><td>0.3116</td><td>0.2085</td><td>0.1987</td><td>0.1958</td><td>0.2430</td></tr><tr><td>segmentation</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.0961</td><td>0.1240</td><td>0.0844</td><td>0.0828</td><td>0.0749</td><td>0.1031</td></tr><tr><td rowspan="4">YOLOv11n [40]</td><td>Box</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.2764</td><td>0.5493</td><td>0.3807</td><td>0.3508</td><td>0.2997</td><td>0.4063</td></tr><tr><td>detection</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.1763</td><td>0.3556</td><td>0.2350</td><td>0.2192</td><td>0.1803</td><td>0.2729</td></tr><tr><td>Mask</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.1788</td><td>0.2952</td><td>0.2234</td><td>0.2107</td><td>0.1912</td><td>0.2432</td></tr><tr><td>segmentation</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.0819</td><td>0.1133</td><td>0.0901</td><td>0.0852</td><td>0.0804</td><td>0.1027</td></tr><tr><td rowspan="4">SCNet [41]</td><td>Box</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.1115</td><td>0.3682</td><td>0.2141</td><td>0.1783</td><td>0.1748</td><td>0.2242</td></tr><tr><td>detection</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.0474</td><td>0.1688</td><td>0.0934</td><td>0.0788</td><td>0.0762</td><td>0.1025</td></tr><tr><td>Mask</td><td> $\mathrm { m A P } _ { 5 0 }$ </td><td>0.1067</td><td>0.3594</td><td>0.2109</td><td>0.1777</td><td>0.1742</td><td>0.2202</td></tr><tr><td>segmentation</td><td> $\mathrm { m A P } _ { 5 0 : 9 5 }$ </td><td>0.0399</td><td>0.1403</td><td>0.0832</td><td>0.0704</td><td>0.0685</td><td>0.0873</td></tr></table>

Base denotes direct inference on low-light images without enhancement, whereas Normal denotes inference on the corresponding normal-light images and is excluded from ranking. Within each cell, the upper and lower values correspond to m $\mathrm { \ A P } _ { 5 0 }$ and $\mathrm { m A P } _ { 5 0 : 9 5 }$ , respectively. Best results among the enhancement methods are shown in bold. The green cells highlight the proposed method.

## 5.5. Comparison with State-of-the-Art Methods

For the image enhancement task, we compare our method with eight state-of-the-art low-light image enhancement approaches, including MIRNet [27], Restormer [28], LLFormer [36], FourLLIE [37], Retinexformer [30], UHDformer [38], HVI-CIDNet [31], and DarkIR [32]. For downstream bridge damage recognition, we further evaluate the enhanced images using five detector architectures, namely YOLOv8m [39], YOLOv11m [40], YOLOv8n [39], YOLOv11n [40], and SCNet [41]. Because each synthetic low-light image is generated from a normal-light source while preserving the original scene geometry and defect annotations, the same bounding boxes and instance masks can be used to compare recognition performance under diferent illumination conditions. For the downstream comparison reported in Table 5, three representative enhancement baselines, namely Retinexformer, HVI-CIDNet, and DarkIR, are compared with the proposed DaL-MoE. Each detector is trained once on the normal-light training split and then kept fixed for inference on the Base, Normal, and enhanced inputs. Therefore, the diferences in downstream performance primarily reflect the contribution of the restoration front end rather than changes in detector architecture or optimization. In addition, two reference settings are reported: Base denotes direct inference on low-light images without enhancement, whereas Normal denotes inference on the corresponding normal-light images and serves as an upper-bound reference under normal illumination.

## 5.5.1. Quantitative Comparison

Based on the quantitative results reported in Table 4 for image enhancement and Table 5 for downstream bridge damage recognition, four observations can be made.

(1) DaL-MoE achieves the best image enhancement performance on the paired bridge benchmark. As shown in Table 4, DaL-MoE obtains the highest performance among all compared low-light image enhancement methods, achieving a PSNR of 23.1174 dB and an SSIM of 0.8482. Retinexformer is the strongest baseline in terms of PSNR, with a PSNR of 21.9448 dB; therefore, DaL-MoE provides an improvement of 1.1726 dB. In terms of SSIM, MIRNet is the best-performing baseline with a score of 0.7903, whereas DaL-MoE improves this value by 0.0579. These results indicate that the proposed method can simultaneously improve pixel-level reconstruction fidelity and structural consistency. This property is particularly important for low-light UAV bridge inspection, where useful evidence may be distributed across weak crack patterns, spalling contours, corrosion traces, seepage regions, and other low-contrast surface defects. In contrast to methods that may obtain favorable performance on one metric while compromising the other, DaL-MoE provides a more balanced recovery of brightness, color, local contrast, and structural information.

(2) Low-light enhancement improves downstream bridge damage recognition. As reported in Table 5, directly applying the detector to low-light images, denoted as Base, results in a clear performance degradation. This degradation is caused by the loss of local contrast, amplified dark-region noise, and weakened defect boundaries under insuficient illumination. In contrast, the images enhanced by DaL-MoE consistently improve both bounding-box detection and instance segmentation performance. For example, on YOLOv8m, DaL-MoE increases box $\mathrm { m A P } _ { 5 0 }$ from 0.2787 to 0.4806 and box $\mathrm { m A P } _ { 5 0 : 9 5 }$ from 0.1837 to 0.3341. On YOLOv11m, box $\mathrm { m A P } _ { 5 0 }$ increases from 0.3097 to 0.4923, while box $\mathrm { m A P } _ { 5 0 : 9 5 }$ increases from 0.2105 to 0.3578. These improvements show that the proposed restoration front end can recover visual cues that are useful for defect localization and recognition, rather than merely increasing the apparent brightness of the input images.

(3) DaL-MoE provides consistent advantages in both detection and instance segmentation. Among the enhancement methods reported in Table 5, DaL-MoE achieves the highest values in every reported detector–task combination for both bounding-box detection and mask segmentation. For example, on YOLOv11m, DaL-MoE obtains a box $\mathrm { m A P } _ { 5 0 }$ of 0.4923 and a box $\mathrm { m A P } _ { 5 0 : 9 5 }$ of 0.3578, outperforming Retinexformer by 0.0428 and 0.0679, respectively. For instance segmentation, DaL-MoE increases mask $\mathrm { \ m A P } _ { 5 0 }$ from 0.2985 to 0.3529 and mask $\mathrm { m A P } _ { 5 0 : 9 5 }$ from 0.1260 to 0.1655 compared with Retinexformer. These results suggest that DaL-MoE not only improves global visibility but also preserves local defect structures, object boundaries, and fine-grained surface details that are essentia for accurate instance-level delineation. The consistent improvement in both box- and mask-based metrics further supports the task-oriented design of the proposed restoration model.

(4) The proposed method provides stable gains across diferent detector architectures. The improvements brought by DaL-MoE are not restricted to a particular detector family or model scale. Compared with the Base setting, DaL-MoE improves box m $\mathsf { A P } _ { 5 0 }$ by 0.2019, 0.1826, 0.09, 0.1299, and 0.1127 for YOLOv8m, YOLOv11m, YOLOv8n, YOLOv11n, and SCNet, respectively. The same overall trend is also observed for box $\mathrm { m A P } _ { 5 0 : 9 5 }$ , mask $\mathrm { m A P } _ { 5 0 }$ , and mask $\mathrm { m A P } _ { 5 0 : 9 5 }$ . These results indicate that the benefit of the proposed enhancement is not tied to a specific detector architecture and supports the use of DaL-MoE as a detector-agnostic restoration front end. Although the Normal setting remains an upper-bound reference under the evaluated conditions, DaL-MoE consistently narrows the performance gap between low-light and normal-light inputs. This demonstrates the practical value of task-oriented low-light restoration for UAV-based bridge damage inspection.

## 5.5.2. Visual Comparison

To further evaluate the efectiveness of the proposed method, we provide visual comparisons from two complementary perspectives: enhanced bridge image quality and downstream damage recognition performance.

MIRNet  
Restormer  
LLFormer  
FouriLLIE  
![](images/b6d343e31c866fc069044d050bd9d6c22d069cc28fda5c5bd45860038accfbd0.jpg)  
Figure 4: Visual comparison of low-light bridge image enhancement results. Input denotes the original low-light image, GT denotes the corresponding normal-light reference, and the remaining columns show the enhanced results produced by diferent methods.

Visual comparison of enhanced images. Fig. 4 presents representative visual comparisons among the input lowlight UAV images, the corresponding normal-light references, and the enhanced results produced by diferent methods. Here, Input denotes the original low-light image, whereas GT denotes the corresponding normal-light reference image. Across multiple challenging bridge scenes, DaL-MoE restores the overall illumination more faithfully toward the normal-light reference, making bridge surfaces and damage-related regions more visible. More importantly, the proposed method maintains stable color rendition and preserves local contrast without introducing obvious chromatic distortion. In contrast, although MIRNet and Restormer improve the overall visibility to some extent, their results exhibit noticeable color deviations in several scenes, leading to unnatural tones and reduced visual consistency with the reference images. These deviations may afect the appearance of concrete textures and defect regions, particularly when the defects are characterized by subtle color or intensity diferences. Compared with these methods, DaL-MoE achieves a better balance between brightness recovery, color fidelity, and structural preservation. This indicates that the proposed approach is better suited to enhancing low-light bridge imagery while retaining visual cues that are relevant to subsequent damage recognition.

Visual comparison of downstream detection results. Fig. 5 presents the YOLOv11m detection results under diferent input settings. Two observations can be made. (1) Enhancement before detection substantially improves bridge damage recognition under low-light conditions. When the original low-light images are directly fed into YOLOv11m, the detector produces noticeable missed detections and incomplete localization results. This is mainly because insuficient illumination weakens defect visibility, reduces local contrast, blurs damage boundaries, and makes small or low-contrast defects dificult to distinguish from the surrounding bridge surface. Such efects are particularly harmful for fine cracks, shallow spalling, corrosion traces, and other defects whose visual evidence is already weak. After low-light enhancement, the visual quality of the input images is improved and more damage-relevant cues become visible. As a result, YOLOv11m detects more defect regions and produces more complete prediction results. This qualitative observation is consistent with the quantitative improvements reported in Table 5, demonstrating that image enhancement can serve as an efective preprocessing stage for low-light UAV bridge damage recognition.

![](images/95b79101d606a697e67ad50c0fcb267bee60340f705c76aa244f48569f00279a.jpg)  
Figure 5: Visual comparison of YOLOv11m bridge damage detection results under diferent input conditions. The same detector is applied to the low-light input, the enhanced images, and the corresponding normal-light reference.

(2) DaL-MoE provides more detection-friendly enhanced images than the other compared enhancement methods. Although some existing methods can produce visually acceptable results in particular scenes, their enhanced outputs are not always equally beneficial for downstream bridge damage recognition. For example, Retinexformer may recover reasonable brightness and visual appearance in some cases, but its enhanced images can still lead to missed detections when processed by YOLOv11m. This observation indicates that perceptual brightness improvement alone is insuficient for bridge damage inspection. The detector also depends on whether damage-relevant cues, such as crack boundaries, spalling contours, corrosion patterns, seepage regions, and local texture variations, are preserved with suficient fidelity. In contrast, DaL-MoE not only improves image brightness but also preserves damage-relevant structural details while reducing color distortion and boundary degradation. Consequently, the enhanced images generated by DaL-MoE enable YOLOv11m to identify more complete defect regions and produce results closer to those obtained under normal illumination. These results further demonstrate that the proposed method is suitable for task-oriented low-light restoration, where the objective is not merely to improve perceptual quality but to recover visual information that supports accurate downstream bridge damage detection.

## 5.6. Ablation Study

To verify the efectiveness of the proposed components for task-oriented low-light restoration of bridge inspection images, we conduct ablation experiments on the paired synthetic low-light bridge benchmark. The analysis is performed from two perspectives: the overall contribution of the Degradation-Aware Guidance Estimation (DAGE) module and the Adaptive Mixture-of-Experts Restoration (AMER) module, and the internal design of the AMER module. The results are reported in Table 6. Since the benchmark preserves the spatial correspondence between normal-light and low-light images as well as the original bridge damage annotations, the ablation results reflect the ability of each component to recover illumination, suppress degradation, and preserve damage-relevant structural information.

Ablation of DAGE and AMER. We first evaluate the contribution of the two main modules, namely DAGE and AMER. When AMER is removed, the PSNR and SSIM decrease from 23.12 and 0.8482 to 21.65 and 0.7157, respectively. This degradation demonstrates that adaptive expert collaboration is important for restoring bridge images afected by coupled low-light degradations, including insuficient illumination, dark-region noise amplification, contrast loss, and weakened damage boundaries. Without the AMER module, the network has a reduced capacity to coordinate noise suppression, illumination adjustment, and structural-detail recovery, which are jointly required to recover visual cues from cracks, spalling contours, corrosion traces, seepage regions, and other low-contrast bridge defects. Removing DAGE module also results in a clear performance degradation, with the PSNR decreasing to 22.73 and the SSIM decreasing to 0.7716. DAGE estimates spatially varying illumination and noise-related guidance from the low-light input and provides this information to the subsequent restoration stages. When DAGE is removed, the expert branches receive less explicit information about regions afected by severe underexposure and noise amplification. As a result, the restoration network becomes less efective in selectively suppressing low-light degradation while preserving damage-relevant structures. These results demonstrate that DAGE and AMER play complementary roles: DAGE provides degradation-aware guidance, whereas AMER uses this guidance to adaptively integrate diferent restoration behaviors.

Table 6  
Ablation study of the proposed DaL-MoE method.
<table><tr><td>Variant</td><td colspan="2">Main Modules</td><td colspan="4">AMER Components</td><td colspan="2">Performance</td></tr><tr><td></td><td>DAGE</td><td>AMER</td><td> $E _ { 1 }$ </td><td> $E _ { 2 }$ </td><td> $E _ { 3 }$ </td><td>Gate</td><td>PSNR (dB) ↑</td><td>SSIM ↑</td></tr><tr><td colspan="9">A. Main architecture modules</td></tr><tr><td>w/o AMER</td><td>√</td><td>x</td><td></td><td></td><td>一</td><td></td><td>21.65</td><td>0.7157</td></tr><tr><td>w/o DAGE</td><td>x</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>22.73</td><td>0.7716</td></tr><tr><td colspan="9">B. AMER components</td></tr><tr><td>w/o Expert1</td><td></td><td></td><td>x</td><td>√</td><td>√</td><td></td><td>22.03</td><td>0.7682</td></tr><tr><td> $\mathsf { w } / \mathsf { o } \mathsf { E x p e r t } _ { 2 }$ </td><td></td><td></td><td></td><td>x</td><td>√</td><td></td><td>22.89</td><td>0.7449</td></tr><tr><td>w/o Expert3</td><td></td><td></td><td></td><td></td><td>x</td><td></td><td>22.72</td><td>0.7553</td></tr><tr><td>w/o Gate</td><td></td><td></td><td></td><td></td><td>√</td><td>x</td><td>21.14</td><td>0.7397</td></tr><tr><td>Ours</td><td>V</td><td></td><td></td><td></td><td>V</td><td>V</td><td>23.12</td><td>0.8482</td></tr></table>

$" w / \circ "$ denotes “without $\because x , x ,$ and –indicate enabled, removed, and not applicable, respectively. DAGE and AMER denote the Degradation-Aware Guidance Estimation module and the Adaptive Mixture-of-Experts Restoration module, respectively. $E _ { 1 } { - } E _ { 3 }$ denote the Noise-Aware Expert, Color Adjustment Expert, and Detail Enhancement Expert, respectively. The best results are shown in bold, and the green cells highlight the full model and its performance.

Ablation of the AMER components. We further investigate the internal design of AMER by individually removing the three expert branches and the adaptive gating mechanism while retaining the other components. The three experts correspond to the Noise-Aware Expert, the Color Adjustment Expert, and the Detail Enhancement Expert, which are designed to address complementary degradation characteristics in low-light bridge imagery. Removing $E x p e r t _ { 1 }$ , namely the Noise-Aware Expert, reduces the PSNR and SSIM to 22.03 and 0.7682, respectively. This result indicates that degradation-aware noise suppression is important for recovering information from dark regions, where sensor noise and illumination loss can obscure weak damage patterns. In particular, efective noise suppression helps prevent noise responses from interfering with the representation of fine cracks, corrosion traces, and other low-contrast defects. Removing Expert , namely the Color Adjustment Expert, produces a PSNR of 22.89 and an SSIM of 0.7449. Although its PSNR remains relatively high among the incomplete AMER variants, the lower SSIM indicates a loss of structural and tonal consistency. This result suggests that color and illumination adjustment contributes not only to the visual appearance of the restored image but also to the consistency of bridge-surface structures and damage-related regions with the normal-light reference. Removing $E x p e r t _ { 3 }$ , namely the Detail Enhancement Expert, decreases the PSNR and SSIM to 22.72 and 0.7553, respectively. This degradation confirms the importance of the detail-oriented branch for recovering local boundaries and fine structures. Such information is particularly relevant to bridge damage categories with weak or irregular morphology, including thin cracks, spalling boundaries, corrosion contours, exposed reinforcement edges, and local concrete texture variations.

The adaptive gating mechanism is also important for the proposed AMER design. When the gate is removed, the model obtains a PSNR of 21.14 and an SSIM of 0.7397. The PSNR of this variant is the lowest among all evaluated variants, indicating that simply combining multiple experts without degradation-dependent routing is insuficient. The gating network enables the model to dynamically adjust the contribution of the three experts according to the degradation characteristics of the current input and feature scale. This adaptive routing improves the coordination between noise suppression, color and illumination adjustment, and structural-detail recovery.

Overall, removing any individual expert or the gating mechanism leads to lower performance than the full model. The ablation results therefore confirm that the three experts provide complementary restoration capabilities, while the adaptive gating mechanism is necessary to integrate them efectively. Together with the contribution of DAGE, these results support the task-oriented design of DaL-MoE, which aims not only to improve the perceptual quality of low-light bridge images but also to recover visual information that is useful for subsequent multi-class damage detection and instance segmentation.

## 6. Real-World Evaluation

![](images/d1817f70f3dd34380638fea259b398a1d98a159be4378bea8345ada0d9ad4808.jpg)  
Figure 6: Dataset-averaged grayscale intensity distributions of 200 real-world nighttime bridge images. DaL-MoE increases the mean intensity from 40.2 to 135.0.

To further examine the practical applicability and synthetic-to-real transfer capability of the proposed DaL-MoE framework, we evaluate it on real-world low-light UAV bridge inspection images. Unlike the paired synthetic benchmark, these images do not have spatially aligned normal-light references. Therefore, full-reference restoration metrics such as PSNR and SSIM cannot be computed for this real-world set. We instead evaluate real-scene transfer from two complementary perspectives: image-level visibility analysis and downstream bridge damage detection. The former characterizes changes in brightness and local edge responses, while the latter includes both qualitative visualization and quantitative box detection using manual annotations. This evaluation examines whether the degradation-aware restoration learned from the ISP-aware synthetic benchmark can transfer to practical bridge inspection scenes afected by severe underexposure, structural occlusion, and illumination variability.

## 6.1. Visual Analysis of Real-World Nighttime Bridge Images

We evaluate 200 real-world nighttime bridge images captured during UAV inspections. The visual analysis focuses on two aspects: whether DaL-MoE alleviates severe underexposure and whether it reveals local structural cues that are dificult to perceive in the original low-light inputs, as illustrated in Figs. 6 and 7. Since the real-world images do not have paired normal-light references, the following intensity and edge-density measurements serve as auxiliary descriptors of visibility changes rather than full-reference restoration metrics.

## 6.1.1. Brightness Enhancement

We first examine whether DaL-MoE efectively alleviates the severe underexposure commonly observed in realworld nighttime bridge inspection scenes. As shown in Fig. 6, the dataset-averaged grayscale intensity distribution of the original low-light images is concentrated in the dark range, particularly below a gray value of 60. The mean intensity is only 40.2, indicating that a considerable amount of bridge-surface information is poorly exposed and dificult to distinguish from the surrounding background.

After enhancement, the intensity distribution shifts toward a substantially broader and more informative brightness range. The mean intensity increases from 40.2 to 135.0, corresponding to a 3.36× increase. The enhanced distribution does not exhibit a dominant accumulation at the high-intensity end, suggesting that DaL-MoE improves the visibility of underexposed bridge regions without causing obvious global saturation. This brightness recovery is particularly beneficial for surfaces located beneath bridge decks, within locally occluded regions, and around structural components where damage cues may otherwise be submerged in darkness. Together, these observations indicate that the illumination-recovery capability learned from the synthetic low-light benchmark transfers efectively to real nighttime UAV imagery and improves the visibility of underexposed bridge surfaces.

## 6.1.2. Structural Cue Recovery

![](images/c37fe14a54133c8e8de6f33ff0f395032e43de717a5b1e118f0f034831aac216.jpg)  
Figure 7: Canny edge densities of 200 real-world nighttime bridge images, sorted by the density of the low-light inputs. The mean density increases from 1.55% to 4.07%.

Improved brightness alone does not necessarily indicate that damage-relevant structural information has been recovered. We therefore further analyze local structural visibility using Canny edge density [42], defined as the ratio of detected edge pixels to the total number of image pixels. The same grayscale conversion and fixed Canny parameters are applied to each low-light image and its corresponding enhanced output, so that the comparison reflects the change induced by the restoration front end. As shown in Fig. 7, the enhanced outputs generally exhibit higher edge density than the corresponding low-light inputs across the 200-image dataset. The dataset-average edge density increases from 1.55% to 4.07%. This increase indicates that more local intensity transitions and high-frequency responses become detectable after enhancement, which is consistent with improved visibility of surface boundaries, construction joints, local texture variations, and potential damage regions.

However, Canny edge density is an auxiliary visibility descriptor rather than a direct measure of structural fidelity. A higher edge density may reflect the recovery of meaningful boundaries, but it may also include enhanced texture responses or residual noise. For this reason, the edge-density analysis is interpreted together with the qualitative examples and downstream detection results rather than used as an independent measure of restoration accuracy. For a more intuitive comparison, Fig. 8 presents four representative samples together with their corresponding Canny edge maps. In the original low-light inputs, many surface boundaries and local patterns are weak or fragmented because of insuficient illumination. After enhancement, denser and more continuous edge responses can be observed around structural boundaries, construction joints, surface textures, and damage-related regions such as cracks and spalling. The enhanced edge maps may also contain additional fine responses caused by noise or texture amplification; nevertheless, the overall increase in edge continuity and local responses provides useful qualitative evidence that structural cues have become more visible.

## 6.2. Downstream Bridge Damage Detection

The preceding analysis shows that DaL-MoE improves the brightness distribution and exposes richer local edge responses in real-world nighttime bridge images. We next examine whether this recovered visual information can be utilized by a downstream bridge damage detector, thereby providing a real-scene evaluation of the task-oriented restoration objective introduced in this paper. Although paired normal-light references are unavailable, all 200 original low-light images were manually annotated with bounding boxes for the same eight damage categories considered in the paired synthetic benchmark. Because DaL-MoE changes the photometric appearance without altering the spatial geometry of the input images, the same annotations can be used to evaluate detection on both the original low-light images and their enhanced counterparts. This setting enables both qualitative and quantitative evaluation of real-world box detection performance

As shown in Fig. 9, Label denotes the bounding-box annotations manually delineated on the original low-light images. Low-light denotes direct inference on the original low-light images, while Ours denotes inference after enhancement with DaL-MoE. Consistent with the controlled downstream evaluation protocol described in Section 5.4, damage detection in both settings is performed using the same YOLOv11m checkpoint trained on the normal-light training split of the collected bridge damage dataset. The detector architecture, model parameters, preprocessing, and inference procedure are kept fixed, and no enhancement-specific fine-tuning is applied. Therefore, diferences in the visual predictions and quantitative box metrics primarily reflect the influence of the restoration front end on downstream damage perception.

![](images/a10c6282ad2e8da98cf5965bf6b1fae1e8c3500dea7852e6ca756b860e50faf9.jpg)  
Figure 8: Structural-cue comparison on representative real-world nighttime bridge images. From left to right: low-light input, input Canny edges, DaL-MoE output, and output Canny edges.

When the original low-light images are directly fed into YOLOv11m, the detector exhibits missed detections, spurious responses, and inaccurate localization in several challenging examples. Severe underexposure reduces the local contrast between bridge damage and the surrounding concrete surface, while darkness and sensor noise obscure damage boundaries and fine surface textures. These efects make it dificult for the detector to distinguish defects such as cracks, spalling, corrosion-related regions, and other low-contrast surface deteriorations.

In contrast, the detections obtained from the DaL-MoE-enhanced images are generally more consistent with the manual annotations in the illustrated examples. Several previously weak or missed damage regions become detectable after enhancement, and the predicted regions are more closely aligned with visible bridge-surface structures. This improvement is consistent with the preceding image-level analysis: by recovering illumination and exposing local structural cues, DaL-MoE provides a more informative RGB input for the fixed downstream detector. The visual examples also show that some missed detections and spurious predictions remain, reflecting the severity and spatia heterogeneity of real-world low-light degradation as well as the domain gap from the synthetic training distribution.

![](images/7f52f6d19b6e625b01aaff3778ee3e039252491a0f248f1f6a284d29f4e6c36d.jpg)  
Figure 9: Qualitative YOLOv11m detection results on real-world nighttime bridge images. Label, Low-light, and Ours denote manual annotations, predictions on the original low-light images, and predictions after DaL-MoE enhancement, respectively. The detector and inference settings are fixed.

Table 7  
Class-wise box detection performance and absolute gains on 200 manually annotated real-world nighttime bridge images.
<table><tr><td rowspan="2">Class</td><td colspan="2">Low-light</td><td colspan="2">Enhanced</td><td colspan="2">Absolute gain</td></tr><tr><td> $\mathrm { A P } _ { 5 0 }$ </td><td> $\mathrm { A P } _ { 5 0 : 9 5 }$ </td><td> $\mathrm { A P } _ { 5 0 }$ </td><td> $\mathrm { A P } _ { 5 0 : 9 5 }$ </td><td> $\Delta \mathrm { A P } _ { 5 0 }$ </td><td> $\Delta \mathrm { A P } _ { 5 0 : 9 5 }$ </td></tr><tr><td>Spalling</td><td>0.079</td><td>0.033</td><td>0.240</td><td>0.109</td><td>+0.161</td><td>+0.076</td></tr><tr><td>Crack</td><td>0.050</td><td>0.019</td><td>0.125</td><td>0.057</td><td>+0.075</td><td>+0.038</td></tr><tr><td>Hole</td><td>0.224</td><td>0.096</td><td>0.418</td><td>0.179</td><td>+0.194</td><td>+0.083</td></tr><tr><td>Honeycombing</td><td>0.015</td><td>0.007</td><td>0.116</td><td>0.045</td><td>+0.101</td><td>+0.038</td></tr><tr><td>Seepage</td><td>0.186</td><td>0.099</td><td>0.355</td><td>0.205</td><td>+0.169</td><td>+0.106</td></tr><tr><td>Rebar corrosion</td><td>0.071</td><td>0.026</td><td>0.157</td><td>0.066</td><td>+0.086</td><td>+0.040</td></tr><tr><td>Rust</td><td>0.086</td><td>0.053</td><td>0.171</td><td>0.100</td><td>+0.085</td><td>+0.047</td></tr><tr><td>Efflorescence</td><td>0.270</td><td>0.155</td><td>0.436</td><td>0.250</td><td>+0.166</td><td>+0.095</td></tr><tr><td>All</td><td>0.123</td><td>0.061</td><td>0.252</td><td>0.126</td><td>+0.129</td><td>+0.065</td></tr></table>

Class rows report box AP, whereas the All row reports box mAP averaged over the eight damage categories. Absolute gains are calculated as the enhanced score minus the corresponding low-light score. Bounding boxes were manually annotated on the original low-light images and reused for the enhanced counterparts because enhancement preserves spatial geometry. Both conditions use the same fixed YOLOv11m checkpoint and inference settings without enhancement-specific fine-tuning.

Nevertheless, the comparison indicates that the visual information recovered by DaL-MoE contributes to downstream damage recognition rather than merely improving perceived image brightness.

As shown in Table 7, DaL-MoE improves downstream box detection performance across all eight damage categories on the manually annotated real-world nighttime images. Compared with direct inference on the original lowlight images, the overall box $\mathrm { \ m A P } _ { 5 0 }$ increases from 0.123 to 0.252, representing an absolute gain of 0.129, while box $\mathrm { m A P } _ { 5 0 : 9 5 }$ increases from 0.061 to 0.126, representing an absolute gain of 0.065. At the class level, the largest absolute $\mathrm { A P } _ { 5 0 }$ gains are observed for hole, seepage, eflorescence, and spalling. Their $\mathsf { A P } _ { 5 0 }$ values increase from 0.224, 0.186, $0 . 2 7 0 .$ , and 0.079 to 0.418, 0.355, 0.436, and 0.240, corresponding to absolute improvements of 0.194, 0.169, 0.166, and 0.161, respectively. Improvements are also obtained for challenging categories such as crack and honeycombing, whose $\mathrm { A P } _ { 5 0 }$ values increase from 0.050 to 0.125 and from 0.015 to 0.116, respectively. A consistent improvement is observed in $\mathsf { A P } _ { 5 0 : 9 5 }$ for every damage category, indicating that the recovered visual information benefits both damage recognition and localization under stricter IoU thresholds. Because the same fixed YOLOv11m checkpoint and inference settings are used for the low-light and enhanced images without enhancement-specific fine-tuning, these results provide quantitative real-scene evidence that DaL-MoE improves the usability of nighttime UAV bridge images for downstream damage detection.

Overall, the real-world evaluation supports the intended role of DaL-MoE as a detector-agnostic restoration front end for low-light UAV bridge inspection. The brightness and edge-density analyses demonstrate improved image-level visibility, while the qualitative examples and class-wise box detection results show that the recovered visual information can be utilized by a fixed downstream detector. Together with the full-reference restoration and multi-framework recognition results obtained on the paired synthetic benchmark, these findings provide complementary evidence of synthetic-to-real transfer. Broader validation on larger multi-site real-world datasets containing both bounding-box and instance-mask annotations will further establish robustness across bridge sites, camera systems, UAV platforms, and illumination conditions.

## 7. Conclusions

This paper addressed the degradation of UAV-based bridge damage recognition under low illumination by considering two related questions: how to establish controllable training and evaluation conditions without real paired normal-light–low-light images, and how to recover damage-related visual information that remains usable by diferent object detection and instance segmentation models. To establish controlled conditions, an ISP-aware synthesis pipeline was used to construct 2,895 spatially aligned normal-light–low-light image pairs while preserving the original bounding-box and instance-mask annotations for eight bridge damage categories. On this basis, the detectoragnostic DaL-MoE was developed to balance noise suppression, illumination and color adjustment, and structuraldetail recovery through degradation-aware guidance and adaptive expert collaboration. Its standardized RGB output can be directly processed by existing recognition models without modifying their architectures or jointly training them with the restoration network.

On the held-out paired synthetic test set of 482 images, DaL-MoE achieved 23.1174 dB PSNR and 0.8482 SSIM, outperforming the compared low-light image restoration methods. These restoration improvements were also reflected in higher downstream recognition accuracy under the fixed-detector evaluation protocol. For YOLOv11m, box $\mathrm { m A P } _ { 5 0 }$ increased from 0.3097 to 0.4923, while box $\mathrm { m A P } _ { 5 0 : 9 5 }$ increased from 0.2105 to 0.3578. Mask $\mathrm { m A P } _ { 5 0 }$ increased from 0.2281 to 0.3529, and mask m $\mathrm { A P } _ { 5 0 : 9 5 }$ increased from 0.1051 to 0.1655. Similar improvements were observed for YOLOv8m, YOLOv8n, YOLOv11n, and SCNet, indicating that the restored images provide cross-architecture utility under the evaluated conditions. The ablation results further confirmed that degradation-aware guidance, the complementary expert branches, and adaptive gating each made a substantive contribution to the final restoration performance.

On 200 real-world nighttime UAV images without paired normal-light references, DaL-MoE increased the datasetaverage grayscale intensity from 40.2 to 135.0 and the average Canny edge density from 1.55% to 4.07%. All 200 original low-light images were manually annotated with bounding boxes for the same eight damage categories used in the paired synthetic benchmark. Under the controlled protocol using the same fixed YOLOv11m checkpoint and inference settings without enhancement-specific fine-tuning, the overall box m $\mathrm { A P } _ { 5 0 }$ increased from 0.123 to 0.252, corresponding to an absolute gain of 0.129, while box $\mathrm { m A P } _ { 5 0 : 9 5 }$ increased from 0.061 to 0.126, corresponding to an absolute gain of 0.065. Class-wise $\mathrm { A P } _ { 5 0 }$ and $\mathsf { A P } _ { 5 0 : 9 5 }$ improved for all eight damage categories, and the qualitative examples likewise showed clearer damage regions and more complete predictions in representative scenes. Although changes in brightness and edge density are auxiliary visibility descriptors and do not directly measure structural fidelity, the controlled box detection results provide quantitative task-level evidence that the restored visual information can be utilized by a fixed downstream detector. Together, these image-level and task-level findings provide complementary qualitative and quantitative evidence of transfer from the ISP-aware synthetic degradation to real low-light bridge inspection scenes.

The conclusions of this paper should nevertheless be interpreted within the evaluated conditions. The threechannel RAW-like synthesis pipeline approximates exposure attenuation, spatially varying illumination, sensor noise, quantization, and ISP rendering, but it cannot fully reproduce the complexity of field imaging conditions, including more irregular illumination distributions, motion or defocus blur, and camera-dependent ISP and sensor-noise characteristics. The paired synthetic benchmark was derived from one collected bridge dataset acquired using a single imaging system, and only a limited set of downstream recognition architectures was evaluated. The current real-world quantitative evaluation is further limited to 200 images, bounding-box annotations, and one fixed YOLOv11m detector;

it therefore does not yet establish real-scene instance segmentation performance or robustness across multiple detectors and independent inspection sites. In addition, a performance gap remains between the enhanced and normal-light inputs on the paired benchmark. DaL-MoE is primarily trained using an image-reconstruction objective and does not explicitly incorporate detection or segmentation supervision, leaving room for further improvement in downstream task performance. The results therefore indicate that DaL-MoE can serve as an efective low-light restoration front end under the tested conditions and exhibits promising cross-architecture and synthetic-to-real transfer potential, while broader independent validation remains necessary.

Future work will focus on strengthening the real-data foundation and extending the engineering applicability of the framework. On the data side, the current 200-image box-annotated real-world set will be expanded into a larger multi-site low-light bridge benchmark covering diferent bridge types, structural components, camera systems, UAV platforms, and illumination conditions. Complete bounding-box annotations will be retained and extended, while highquality instance-mask annotations will be added to support quantitative real-scene evaluation of both object detection and instance segmentation across multiple recognition architectures. Real low-light observations can also be combined with camera calibration and RAW sensor information to improve the fidelity of degradation modeling and synthetic data generation. On the engineering side, future studies will investigate high-resolution bridge image processing, metric damage quantification, and spatial registration of damage predictions to bridge components. Integrating these results with BIM or digital-twin platforms, historical inspection records, and human verification workflows could further advance image-space damage recognition toward traceable component-level condition representation and maintenance decision support.

## CRediT authorship contribution statement

Hu Wang: Original Draft, Visualization, Validation, Coding, Funding Acquisition. Hongxu Pu: Validation, Coding, Data Curation. Zhiqi Hu: Data Curation, Coding, Visualization. Fangzhou Lin: Data Curation, Figure Preparation. Wang Wang: Conceptualization, Original Draft, Experimental Design, Data Curation, Coding, Supervision.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used GPT5.6 Sol in order to improve language. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## Declaration of competing interest

We declare that we do not have any commercial or associative interest that represents a conflict of interest in connection with the work submitted.

## Data availability

Data will be made available on request.

## References

[1] W. Wang, M. Xu, Z. Cao, J. Guo, C. Liu, H. Zhang, et al., Unified data synthesis for automated 3d visual inspection and digital twinning of bridges, Automation in Construction 182 (2026) pp. 106741. https://doi.org/10.1016/j.autcon.2025.106741.

[2] C. Koch, K. Georgieva, V. Kasireddy, B. Akinci, P. Fieguth, A review on computer vision based defect detection and condition assessment of concrete and asphalt civil infrastructure, Advanced Engineering Informatics 29 (2015) pp. 196–210. https://doi.org/10.1016/j.aei.

[3] T. Panigati, M. Zini, D. Striccoli, P. F. Giordano, D. Tonelli, M. P. Limongelli, et al., Drone-based bridge inspections: Current practices and future directions, Automation in Construction 173 (2025) pp. 106101. https://doi.org/https://doi.org/10.1016/j.autcon. 2025.106101

[4] S. Agnisarman, S. Lopes, K. Chalil Madathil, K. Piratla, A. Gramopadhye, A survey of automation-enabled human-in-the-loop systems for infrastructure visual inspection, Automation in Construction 97 (2019) pp. 52–76. https://doi.org/10.1016/j.autcon.2018.10.019.

[5] L. Duque, J. Seo, J. Wacker, Synthesis of unmanned aerial vehicle applications for infrastructures, Journal of Performance of Constructed Facilities 32 (2018) pp. 04018046. https://doi.org/10.1061/(ASCE)CF.1943-5509.0001185.

[6] A. Ellenberg, L. Branco, A. Krick, I. Bartoli, A. Kontsos, Use of unmanned aerial vehicle for quantitative infrastructure evaluation, Journal of Infrastructure Systems 21 (2015) pp. 04014054. https://doi.org/10.1061/(ASCE)IS.1943-555X.0000246.

[7] C. M. Yeum, S. J. Dyke, Vision-based automated crack detection for bridge inspection, Computer-Aided Civil and Infrastructure Engineering 30 (2015) pp. 759–770. https://doi.org/10.1111/mice.12141.

[8] Y.-J. Cha, W. Choi, O. Büyüköztürk, Deep learning-based crack damage detection using convolutional neural networks, Computer-Aided Civil and Infrastructure Engineering 32 (2017) pp. 361–378. https://doi.org/10.1111/mice.12263.

[9] Y.-J. Cha, W. Choi, G. Suh, S. Mahmoudkhani, O. Büyüköztürk, Autonomous structural visual inspection using region-based deep learning for detecting multiple damage types, Computer-Aided Civil and Infrastructure Engineering 33 (2018) pp. 731–747. https://doi.org/10.

[10] J. Seo, L. Duque, J. Wacker, Drone-enabled bridge inspection methodology and application, Automation in Construction 94 (2018) pp. 112–126. https://doi.org/10.1016/j.autcon.2018.06.006.

[11] S. Dorafshan, R. J. Thomas, M. Maguire, Sdnet2018: An annotated image dataset for non-contact concrete crack detection using deep convolutional neural networks, Data in Brief 21 (2018) pp. 1664–1668. https://doi.org/10.1016/j.dib.2018.11.015.

[12] M. Mundt, S. Majumder, S. Murali, P. Panetsos, V. Ramesh, Meta-learning convolutional neural architectures for multi-target concrete defect classification with the concrete defect bridge image dataset, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 11188–11197. https://doi.org/10.1109/CVPR.2019.01145.

[13] L. Huang, G. Fan, J. Li, H. Hao, Deep learning for automated multiclass surface damage detection in bridge inspections, Automation in Construction 166 (2024) pp. 105601. https://doi.org/https://doi.org/10.1016/j.autcon.2024.105601.

[14] R. Li, L. Zhao, H. Wei, G. Hu, Y. Xu, B. Ouyang, et al., Multi-defect type beam bridge dataset: Gyu-det, Scientific Data 12 (2025) pp. 1101. https://doi.org/10.1038/s41597-025-05395-w.

[15] C. Wei, W. Wang, W. Yang, J. Liu, Deep retinex decomposition for low-light enhancement, 2018. https://doi.org/10.48550/arXiv. 1808.04560.arXiy:1808.04560.

[16] C. Chen, Q. Chen, J. Xu, V. Koltun, Learning to see in the dark, in: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018, pp. 3291–3300. https://doi.or /10.1109/CVPR.2018.00347.

[17] Y. P. Loh, C. S. Chan, Getting to know low-light images with the exclusively dark dataset, Computer Vision and Image Understanding 178 (2019) pp. 30–42. https://doi.org/10.1016/j.cviu.2018.10.010.

[18] X. Jia, C. Zhu, M. Li, W. Tang, W. Zhou, Llvip: A visible-infrared paired dataset for low-light vision, in: 2021 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2021, pp. 3489–3497. https://doi.org/10.1109/ICCVW54120.2021.00389.

[19] W. Yang, S. Wang, Y. Fang, Y. Wang, J. Liu, From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3060–3069. https://doi.org/10.1109/ CVPR42600.2020.00313.

[20] T. Brooks, B. Mildenhall, T. Xue, J. Chen, D. Sharlet, J. T. Barron, Unprocessing images for learned raw denoising, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 11028–11037. https://doi.org/10.1109/CVPR.2019. 01129.

[21] A. Punnappurath, A. Abuolaim, A. Abdelhamed, A. Levinshtein, M. S. Brown, Day-to-night image synthesis for training nighttime neural isps, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10759–10768. https://doi.org/ 10.1109/CVPR52688.2022.01050.

[22] Z. Yao, J. Xu, S. Hou, M. C. Chuah, Cracknex: a few-shot low-light crack segmentation model based on retinex theory for uav inspections, in: 2024 IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 11155–11162. https://doi.org/10.1109/ ICRA57147.2024.10611660.

[23] Q. Ren, J. Feng, M. Li, Y. Yu, J. Yuan, K. Shi, Robust low-light crack segmentation with unsupervised visual enhancement for underlit dam gallery inspection, Automation in Construction 187 (2026) pp. 106963. https://doi.org/10.1016/j.autcon.2026.106963.

[24] D. Jobson, Z. Rahman, G. Woodell, A multiscale retinex for bridging the gap between color images and the human observation of scenes, IEEE Transactions on Image Processing 6 (1997) pp. 965–976. https://doi.org/10.1109/83.597272.

[25] X. Guo, Y. Li, H. Ling, Lime: Low-light image enhancement via illumination map estimation, IEEE Transactions on Image Processing 26 (2017) pp. 982–993. https://doi.org/10.1109/TIP.2016.2639450.

[26] C. Guo, C. Li, J. Guo, C. C. Loy, J. Hou, S. Kwong, et al., Zero-reference deep curve estimation for low-light image enhancement, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 1777–1786. https://doi.org/10.1109/ CVPR42600.2020.00185.

[27] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, et al., Learning enriched features for fast image restoration and enhancement, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (2023) pp. 1934–1948. https://doi.org/10.1109/TPAMI.2022. 3167175.

[28] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, Restormer: Eficient transformer for high-resolution image restoration, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5728–5739. https://doi. org/10.48550/arXiv.2111.09881.

[29] X. Xu, R. Wang, C.-W. Fu, J. Jia, Snr-aware low-light image enhancement, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 17693–17703. https://doi.org/10.1109/CVPR52688.2022.01719.

[30] Y. Cai, H. Bian, J. Lin, H. Wang, R. Timofte, Y. Zhang, Retinexformer: One-stage retinex-based transformer for low-light image enhancement, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 12470–12479. https://doi.org 10.1109/ICCV51070.2023.01149.

[31] Q. Yan, Y. Feng, C. Zhang, G. Pang, K. Shi, P. Wu, et al., Hvi: A new color space for low-light image enhancement, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 5678–5687. https://doi.org/10.1109/CVPR52734.2025. 00533.

[32] D. Feijoo, J. C. Benito, A. Garcia, M. V. Conde, Darkir: Robust low-light image restoration, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 10879–10889. https://doi.org/10.1109/CVPR52734.2025.01016.

[33] P. Guo, X. Meng, W. Meng, Y. Bao, Automatic assessment of concrete cracks in low-light, overexposed, and blurred images restored using a generative ai approach, Automation in Construction 168 (2024) pp. 105787. https://doi.org/10.1016/j.autcon.2024.105787.

[34] V. Potlapalli, S. W. Zamir, S. Khan, F. Shahbaz Khan, Promptir: Prompting for all-in-one image restoration, in: A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, S. Levine (Eds.), Advances in Neural Information Processing Systems, volume 36, Curran Associates, Inc., 2023, pp. 71275–71293. https://doi.org/10.52202/075280-3121.

[35] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, G. E. Hinton, Adaptive mixtures of local experts, Neural Computation 3 (1991) pp. 79–87. https://doi.org/10.1162/neco.1991.3.1.79.

[36] T. Wang, K. Zhang, T. Shen, W. Luo, B. Stenger, T. Lu, Ultra-high-definition low-light image enhancement: A benchmark and transformer based method, in: Proceedings of the AAAI conference on artificial intelligence, volume 37, 2023, pp. 2654–2662. https://doi.org/10. 1609/aaai.v37i3.25364.

[37] C. Wang, H. Wu, Z. Jin, Fourllie: Boosting low-light image enhancement by fourier frequency information, Association for Computing Machinery, 2023, p. 7459–7469. https://doi.org/10.1145/3581783.3611909.

[38] C. Wang, J. Pan, W. Wang, G. Fu. S. Liang, M. Wang, et al., Correlation matching transformation transformers for uhd image restoration in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 2024, pp. 5336–5344. https://doi.org/10.1609/aaai. v38i6.28341

[39] R. Varghese, S. M., Yolov8: A novel object detection algorithm with enhanced performance and robustness, in: 2024 International Conference on Advances in Data Engineering and Intelligent Computing Systems (ADICS), 2024, pp. 1–6. https://doi.org/10.1109/ADICS58448. 2024.10533619

[40] R. Khanam, M. Hussain, Yolov11: An overview of the key architectural enhancements, arXiv preprint arXiv:2410.17725, 2024. https: //doi.org/10.48550/arXiv.2410.17725.

[41] T. Vu, H. Kang, C. D. Yoo, Scnet: Training inference sample consistency for instance segmentation, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, 2021, pp. 2701–2709. https://doi.or /10.1609/aaai.v35i3.16374.

[42] J. Canny, A computational approach to edge detection, IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-8 (1986) pp. 679–698. https://doi.org/10.1109/TPAMI.1986.4767851.