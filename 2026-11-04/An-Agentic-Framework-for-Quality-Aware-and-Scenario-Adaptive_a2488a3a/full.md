# An Agentic Framework for Quality-Aware and Scenario-Adaptive Image Compression

Zhenghao Chen, Lebin Zhou, Zicong Chen, Naifu Xue, Jiahao Li, Bin Li, Dong Xu, Fellow, IEEE, and Yan Lu

Abstract—Despite decades of progress, image codecs remain difficult to adapt to diverse real-world compression scenarios for two main reasons. First, most image codecs are optimized for a specific objective and require substantial redevelopment when the compression goal changes. Second, they typically assume clean inputs, whereas sensor-captured images often contain degradations that alter rate–quality behavior. To address these challenges, we propose Practical Image Coding Agents (PICA), a multi-agent framework that adapts existing codecs to diverse user requirements and potentially degraded inputs. PICA consists of four collaborative agents: a Perception agent that interprets user intent and analyzes image content, an Improvement agent that jointly considers bitrate and reconstruction quality to guide precoding restoration, a Coding agent that refines quality-control settings from global to regional levels, and an After-coding improvement agent that addresses residual degradations and compression-induced artifacts through restoration guided solely by image quality. To enable codec-agnostic regional adaptation, we introduce Agentic ROI Coding (AROIC), which treats the codec as a black box and allows the agent to automatically schedule, execute, evaluate, and reflect ROI coding plans without architectural modifications or codec-specific retraining. We integrate these four agents with a Scenario-Guided Quality–Rate (SGQR) evaluation strategy to enable more effective scenarioadaptive compression across heterogeneous codecs. Comprehensive experiments on clean, noisy, and blurred images across standard, perceptual, and machine-vision compression scenarios demonstrate that PICA consistently enhances the compression performance of both the conventional H.266/VVC codec and the neural Cheng2020 codec, outperforming competing compression methods on the Kodak, CLIC2020, and HEVC datasets.

Index Terms—Image compression, image restoration, agentic systems, perceptual compression, machine-vision compression.

## I. INTRODUCTION

MAGE compression is a fundamental problem in signal processing, as the rapidly growing volume and resolution of visual data impose substantial costs on storage, transmission bandwidth, and communication latency. This has been extensively studied for more than three decades, leading to mature hand-engineered image coding standards such as JPEG [1]. Such traditional codecs remain widely used in practical image storage and communication systems. More recent standards, such as H.266/VVC [2], also support high-efficiency image compression through intra-frame coding. With rapid advances in deep learning, neural image compression (NIC) methods have achieved substantial progress by jointly optimizing nonlinear transforms and entropy models under end-to-end rate– distortion (RD) objectives [3]–[10]. Representative methods such as Cheng2020 [11] have demonstrated RD performance competitive with or superior to those traditional codecs on the standard compression benchmarks.

Despite this tremendous success, most widely deployed codecs are designed primarily for generic pixel-fidelity reconstruction, with their RD performance conventionally optimized and evaluated using distortion metrics such as PSNR, a compression scenario referred to as standard compression. However, modern visual-data ecosystems increasingly involve heterogeneous consumers and objectives with distinct compression intents. For example, some compressed images are primarily expected to preserve visual realism and semantic fidelity for human observers [7], [12]–[14], a compression scenario referred to as perceptual compression. Meanwhile, an increasing proportion of visual data is consumed by downstream machine-learning systems, where compression should preserve task-relevant information and downstream accuracy [15]–[17], a compression scenario referred to as machine-vision compression. Unfortunately, prior studies [7], [14], [16], [17] have shown that commonly used standard codecs can become suboptimal in these specialized compression scenarios.

Developing dedicated codecs for these specialized scenarios provides an intuitive solution. With the flexibility of NIC, several scenario-specific methods have therefore been proposed. For example, machine-vision compression approaches [16], [17] introduce task-aware compression modules to produce latent representations that preserve information relevant to downstream vision tasks. Meanwhile, perceptual NIC methods [7], [13] combine powerful generative models with humanperception-oriented objectives, such as LPIPS [18], to reconstruct visually plausible content. Though these specialized designs achieve substantial improvements in their respective scenarios, they are typically tightly coupled with extensive scenario-specific modules and objectives, thereby requiring considerable architectural redesign and substantial re-optimization when the compression intent changes. Consequently, they exhibit limited generalizability across diverse compression scenarios. This limitation motivates our first research question (RQ): RQ1: Can existing widely-used codecs be rapidly adapted to diverse compression scenarios without architectural redesign or substantial re-optimization?

Beyond their limited generalizability across compression scenarios, most existing codecs are developed under the assumption of clean input images. In practice, however, acquired images may exhibit degradations (e.g., noise) due to limitations of imaging sensors, such as those in smartphone cameras. These degradations not only compromise visual quality but also introduce undesirable signal variations that may increase the number of bits required for compression [19], [20]. To address this issue, recent studies have jointly optimized restoration and compression or introduced restoration-aware compression modules [19]–[25]. However, these approaches still rely on dedicated components and degradation-specific optimization. Moreover, they generally assume that the degradation type and restoration objective are known in advance, rather than autonomously determining whether restoration is necessary and how it should be coordinated before and after coding. It motivates our second RQ: RQ2: Can an existing widely used codec autonomously detect image degradations and adaptively coordinate pre-coding and post-coding restoration without substantial re-development?

To simultaneously address above two RQs, we propose Practical Image Coding Agents (PICA), a multi-agent framework that adapts existing image codecs to diverse degraded inputs and compression scenarios through online and goaloriented optimization. PICA consists of four collaborative agents: a Perception Agent, an Improvement Agent, a Coding Agent, and an After-Coding Improvement Agent. Specifically, the Perception Agent interprets the user-specified compression intent while analyzing the input image, including its visual content, quality condition, and regions of interest, thereby providing structured guidance for subsequent agents.

To address RQ1, we introduce a Coding Agent with a global-to-regional coding quality refinement scheme. Guided by the compression intent summarized by the Perception Agent, it first performs Global Quality Control (QC) Search, applying estimated QC to the target codec and iteratively refining them based on the measured bitrate and reconstruction quality until the user-specified requirements are satisfied. Starting from this feasible global operating point, we further refine the coding quality of individual regions, motivated by prior studies showing that different compression scenarios assign different importance to image regions. For example, machine-vision compression often requires stronger preservation of task-relevant objects than background regions [16], [17]. However, most existing methods [26] rely on codecspecific ROI coding mechanisms, requiring the underlying codec to be explicitly designed or trained with dedicated ROI functionality and thus limiting their generalizability to heterogeneous codecs without native ROI support.

To overcome this limitation, we introduce Agentic ROI Coding (AROIC), which treats the underlying codec as a black box and follows a scheduling–execution–reflection loop. Specifically, the agent schedules and executes candidate coding plans that adjust the QC settings of the ROIs and background and, when beneficial, apply background downsampling. Each region is independently encoded using the target codec with its assigned QC setting, and the decoded regions are assembled into a complete reconstruction. The agent then evaluates the compression results and uses the evaluation feedback to refine subsequent coding plans while retaining the best-performing candidate. By relying on the codec’s native QC interface, AROIC enables ROI-aware adaptation across heterogeneous codecs without codec-specific redesign or retraining.

Moreover, to address RQ2, we introduce the Improvement Agent and the After-Coding Improvement Agent. First, the Improvement Agent performs quality–rate (QR)-aware precoding restoration to mitigate input degradations and improve the overall QR trade-off. Unlike existing restoration agents [27], [28], which focus on correcting all detected degradations, our agent jointly evaluates the coding rate and reconstruction quality resulting from each candidate restoration operation. It retains only operations that improve the QR trade-off under the specified compression objective. Similarly, the After-Coding Improvement Agent performs quality-aware post-coding restoration to mitigate residual degradations and potential compression-induced artifacts in the decoded image. This stage focuses on degradations that primarily affect reconstruction quality and adaptively applies restoration only when it improves the quality of the final decoded image.

To align the collaborative agents with the intended compression scenario, we introduce a Scenario-Guided Quality–Rate (SGQR) evaluation strategy for the Improvement and Coding Agents. Specifically, SGQR evaluates each candidate improvement or coding plan using the bitrate and reconstructed image obtained after compression. It assesses reconstruction quality using scenario-specific metrics and selects the plan that offers the best QR trade-off under the user’s compression requirements. Overall, our PICA integrates four collaborative agents with SGQR evaluation to adapt heterogeneous existing codecs to diverse compression scenarios without codec-specific redevelopment. We conduct extensive experiments on clean, noisy, and blurred images across standard, perceptual, and machinevision compression scenarios. Results on the Kodak [29], CLIC2020 [30], and HEVC [31] datasets demonstrate that PICA consistently improves the compression performance of both H.266/VVC [2] and Cheng2020 [32].

Our main contributions can be summarized as follows:

• Practical Image Coding Agents. We propose a multiagent framework that adapts existing image codecs to diverse compression scenarios and potentially degraded inputs through SGQR-guided optimization.

• Quality–Rate–Aware Improvement Agents. We introduce a pre-coding improvement agent that jointly considers bitrate and reconstruction quality to address input degradations, and a post-coding improvement agent to mitigate residual and compression-induced degradations.

• Codec–Agnostic Coding Agent. We introduce a Coding Agent with global-to-regional coding quality refinement together with AROIC, which treats the underlying codec as a black box and enables ROI-aware adaptation across heterogeneous codecs without codec-specific redesign.

## II. RELATED WORKS

## A. Image Compression

Image compression methods broadly fall into conventional codecs and NIC. Conventional standards include

JPEG [1], JPEG2000 [33], and intra-frame coding mode in H.265/HEVC [31] and H.266/VVC [2]. These standard codecs underpin practical compression systems. Driven by expressive neural architectures and end-to-end optimization, NIC has advanced substantially through hyperpriors [4], autoregressive context models [5], and Gaussian mixture entropy models with attention [11], the latter achieving PSNR performance competitive with those standard codecs.

Beyond conventional compression that prioritizes pixel-wise reconstruction fidelity, growing attention has been devoted to alternative compression objectives. In this work, we focus on two challenging scenarios: perceptual compression [7], [12]–[14] and machine-vision compression [15]–[17]. Existing approaches address these objectives through specialized training strategies and architectural components. For example, HiFiC [7] combines adversarial training and perceptual optimizations to produce visually realistic reconstructions. Adapt-ICMH [17] optimizes a spatial-frequency modulation adapters to suppress task-irrelevant redundancy. Although effective for the specific scenarios, these approaches require objective-specific training or additional trainable components. In contrast, PICA is a training-free framework that adapts existing black-box codecs at inference time without codec reoptimization or scenario-specific module redesign. Through global-to-regional quality control, it allocates bits according to perceptual or task-specific priorities.

## B. Quality-Aware Image Compression

Most existing image compression methods treat the observed input as the reconstruction target, without explicitly accounting for acquisition degradations. In practice, however, images may contain noise, blur, or other degradations [19], [21], [22], [24], [25]. These degradations not only compromise visual quality but may also lead codecs to allocate additional bits to preserving undesirable signal variations rather than prioritizing the underlying clean signal [19], [21].

A common approach to addressing this issue is to jointly optimize compression and restoration [19]–[22], [24]. For example, Cheng et al. [19] introduce a two-branch, weightsharing architecture with feature denoisers for joint denoising and compression, while Wan et al. [24] explore joint compression and deblurring through blur-aware guidance. To avoid retraining the entire codec, Chen et al. [23] freeze a pre-trained codec and train additional latent refinement and prompt-generation modules for joint decoding and denoising. Overall, integrating restoration into these compression frameworks requires training either the full model or additional architectural components. In contrast, PICA coordinates off-theshelf restoration tools and black-box codecs without additional training. At inference time, it diagnoses input degradations and evaluates candidate restoration tools to select suitable operations. Before coding, it addresses input degradations by jointly considering bitrate and reconstruction quality. After decoding, it mitigates residual and compression-induced degradations.

## C. Intelligent Agents for Image Processing

Intelligent agents are systems that autonomously perceive and act upon their environments to pursue specified goals [34].

Recent advances in Large Language Models (LLMs) have enabled frameworks such as ReAct [35], which interleaves reasoning and actions to guide decision-making through environmental feedback. Meanwhile, Vision–Language Models (VLMs) [36]–[38] provide visual understanding capabilities that support perception and assessment in image-processing agents. For example, You et al.. [39] introduce DepictQA, which analyzes image content and distortions to provide descriptive quality assessments beyond scalar scores.

Building on visual assessment and language-based reasoning, recent studies coordinate specialized restoration tools through agentic planning and execution [27], [28], [40]– [42]. Zhu et al. [27] propose AgenticIR to address complex image degradations through perception, scheduling, execution, reflection, and rescheduling. Zuo et al. [28] introduce 4KAgent for image restoration and 4K super-resolution, combining perception-guided planning with quality-driven tool selection and an iterative execution–reflection–rollback process. These works demonstrate the potential of integrating visual assessment, planning, and feedback to adaptively coordinate image restoration and enhancement tools.

For image compression, Gao et al. [26] propose Comp-X, which integrates an LLM agent with a neural codec specifically designed and trained to work with the agent. Although Comp-X supports multiple compression objectives and ROI coding, these capabilities rely on trained task-specific adapters and internal codec mechanisms, including spatial quality maps and group masks. Its agentic control therefore remains coupled to this specialized codec, leaving the adaptation of existing codecs without redesign or retraining (i.e., RQ1) unaddressed. In contrast, PICA adapts ready-to-use conventional and neural codecs as black boxes, enabling scenario-adaptive compression without codec-specific architectural modifications, taskspecific adapters, or retraining. Moreover, Comp-X’s workflow does not explicitly consider or address acquisition degradations (i.e., RQ2), whereas PICA considers the effects of degradations on both rate and quality to guide restoration before encoding or after decoding.

## III. METHODOLOGY

## A. Preliminaries and Formulation

Quality-Aware Image Compression. Following prior work on quality-aware image coding [19], we let $\mathbf { x } ^ { \star } \in \mathbf { \bar { \rho } } [ 0 , 1 ] ^ { H \times W \times 3 }$ denote a latent degradation-free image. In practical compression scenarios, the observed input may already contain one or multiple degradations:

$$
\mathbf { x } = \mathcal { G } _ { \delta } ( \mathbf { x } ^ { \star } ) ,\tag{1}
$$

where $\mathcal { G } _ { \delta }$ denotes an unknown degradation process parameterized by δ such as noise or blur. The clean-input case is included by letting $\mathcal { G } _ { \delta }$ be the identity mapping. During compression, the codec observes only raw image x, neither $\mathbf { x } ^ { \star }$ nor $\delta$ is assumed to be available.

QC-Controllable Black-Box Codec. PICA supports heterogeneous off-the-shelf codecs without accessing or modifying their internal architectures. We use a unified QC factor q to abstract codec-specific quality-control parameters, such as the quantization parameter (QP) in H.266/VVC. Given an input image z and a QC factor q, we define a codec query as

![](images/f468136b0ee78639265db1d6aa819041618464269548da9b3e77c1e898a2d197.jpg)  
Fig. 1: Overview of the Practical Image Compression Agent (PICA) framework, consisting of four collaborative agents: the Perception agent, the Improvement agent, the Coding agent, and the After-coding improvement agent.

$$
\bigl ( \mathbf { b } , \widetilde { \mathbf { z } } \bigr ) = \mathcal { C } ( \mathbf { z } ; q ) ,\tag{2}
$$

where b and ez denote the encoded bitstream and reconstructed image, respectively. We treat C as a black-box codec, without knowing knowledge of its internal architecture or coding procedure, but only assumes that the QC factor can be adjusted and its corresponding rate–quality behavior can be observed. For example, increasing $\mathrm { Q P }$ in H.266/VVC produces stronger compression with lower reconstruction quality and bitrate.

Problem Formulation. Given an observed image x that may contain unknown degradations, a QC-controllable black-box codec ${ \mathcal { C } } ,$ and a user instruction I specifying the desired compression objective (i.e., scenario), PICA then employs an agentic coding function $\mathcal { F }$ to produce an aggregate coded representation $B ^ { \star }$ and reconstructed image $\widetilde { \mathbf { x } } ^ { \star }$

$$
\begin{array} { r l } & { \left( \mathcal { B } ^ { \star } , \widetilde { \mathbf { x } } ^ { \star } \right) = \arg \underset { \mathcal { B } , \widetilde { \mathbf { x } } } { \operatorname* { m i n } } \left[ R ( \mathcal { B } ) + \lambda _ { \mathcal { Z } } Q _ { \mathcal { Z } } \left( \widetilde { \mathbf { x } } , \mathbf { x } ^ { \star } \right) \right] , } \\ & { \quad \quad \quad \mathrm { s . t . , } \quad \quad \left( \mathcal { B } , \widetilde { \mathbf { x } } \right) = \mathcal { F } \left( \mathbf { x } , \mathcal { C } , \mathcal { T } \right) . } \end{array}\tag{3}
$$

Here, B denotes the complete coded representation required for reconstruction, comprising one or more codec-generated bitstreams and any necessary coding metadata. $R ( B )$ measures its total bitrate, while $\lambda _ { \mathcal { T } }$ controls the QR trade-off based on the compression requirements, such as the target bitrate. The instruction-dependent quality term $Q _ { \mathcal { T } } ( \widetilde { \mathbf { x } } , \mathbf { x } ^ { \star } )$ evaluates reconstruction quality relative to $\mathbf { x } ^ { \star }$ , with lower values indicating better quality. Rather than measuring distortion against the potentially degraded input x, as is typical in conventional image compression, we follow other quality-aware compression [19] and assess reconstruction quality with respect to the latent degradation-free image $\mathbf { x } ^ { \star }$ . The quality criterion $Q _ { \mathcal { I } }$ is determined by the user-specified compression intent. For example, perceptual compression may employ perceptual quality metrics, whereas machine-vision compression may define $Q _ { \mathcal { I } }$ according to the performance degradation of a downstream machine-vision task. Here, $\mathbf { x } ^ { \star }$ serves only as an ideal reference for defining and evaluating the desired reconstruction quality and is not assumed to be available during compression procedure.

## B. Practical Image Coding Agents

To realize the optimization in Eq. 3, we implement PICA as a practical multi-agent system that instantiates the agentic coding function $\mathcal { F } ( \mathbf { x } , \mathcal { C } , \mathcal { T } )$ . As illustrated in Fig. 1, the $P e r { _ { - } }$ ception Agent takes the observed image $\mathbf { x } = \mathcal { G } _ { \delta } ( \mathbf { x } ^ { \star } )$ and user instruction I as inputs. Using VLMs, it generates a structured configuration S specifying the instruction-dependent quality criterion $Q _ { \mathcal { Z } } ( \cdot , \cdot )$ and the QR preference $\lambda _ { \mathcal { T } }$ inferred from compression requirements such as the target bitrate. The configuration also contains an input quality assessment, including the diagnosed degradation $\mathcal { G } _ { \delta } ^ { \prime } ( \cdot )$ , and auxiliary information such as image content analysis and identified ROIs.

Taking the observed image x and configuration S as in puts, the Improvement Agent updates x through QR–aware restoration. It encodes the image using the target codec C and applies SGQR-based reflection to improve the QR tradeoff in subsequent coding iterations. The Coding Agent then encodes the updated image x using an off-the-shelf black-box codec C under the configuration S, producing an aggregate coded representation <sup>B</sup> and its decoded reconstruction xe. It first performs a global QC search by initializing a codecspecific quality-control parameter and iteratively refining it based on the measured bitrate and reconstruction quality to meet the specified requirements. To account for spatial variations in content importance under the target compression scenario I, the Coding Agent further employs AROIC to preserve important regions while reducing the coding cost of less critical content. Through SGQR-based plan reflection, AROIC enables region-adaptive compression for the objective, without codec-specific architectural modifications or retraining. Finally, the After-Coding Improvement Agent updates the decoded reconstruction x using a restoration procedure similar to the Improvement Agent, but evaluates restoration candidates using image quality alone. The resulting pair (<sup>B</sup>, xe) constitutes the final output of the agentic coding function F in Eq. 3.

Scenario-Guided Quality–Rate Evaluation. To align candidate plan selection with the compression scenario specified by the user instruction I, we propose an SGQR evaluation strategy. Given a candidate improvement or coding plan, we obtain the resulting bitrate b in bits per pixel and the corresponding decoded reconstruction ${ \widehat { \mathbf { x } } } .$ Using an obtained reference image $\mathbf { x } _ { \mathrm { r e f } }$ for quality assessment, we define it as:

![](images/37e2581168421e4f6dd3a5659c72ef0ae3122ce5ead184e99a7abc971df97f7c.jpg)  
Fig. 2: The Coding Agent workflow comprises two components: Global QC Search and Agentic ROI Coding (AROIC).

$$
\begin{array} { r } { \mathcal { E } _ { \mathrm { S G Q R } } \left( b , \widehat { \mathbf { x } } , \mathbf { x } _ { \mathrm { r e f } } ; \mathbb { Z } \right) = b + \lambda _ { \mathbb { Z } } Q _ { \mathbb { Z } } \left( \widehat { \mathbf { x } } , \mathbf { x } _ { \mathrm { r e f } } \right) . } \end{array}\tag{4}
$$

where $Q _ { \mathcal { T } } ( \widehat { \mathbf { x } } , \mathbf { x } _ { \mathrm { r e f } } )$ denotes the instruction-dependent reconstruction quality, with lower values indicating better quality.

In this work, we primarily consider three representative compression scenarios. For standard compression, $Q _ { \mathcal { I } }$ is defined as the mean squared error between the reconstructed and reference images. For perceptual compression, it quantifies perceptual dissimilarity between these images using DISTS and LPIPS. For machine-vision compression, it is defined as the negative of a task-specific performance score evaluated on the reconstructed image $( e . g .$ , a weighted average of the product of detection confidence and IoU for object detection), with task predictions on the reference image serving as pseudo-ground truth. The procedures for obtaining the bitrate b, reconstructed image xb, and reference image $\mathbf { x } _ { \mathrm { r e f } }$ in the Improvement and Coding Agents are detailed in Secs. III-C and III-D, respectively.

## C. Quality–Rate–Aware Improvement Agents

We develop the Improvement Agent and After-Coding Improvement Agent to adaptively handle image degradations through restoration before encoding or after decoding for improved QR performance.

Improvement Agent. Unlike previous restoration agents [27], [28] that primarily aim to improve image quality, our agent performs pre-coding restoration to improve the QR tradeoff of subsequent compression. For example, denoising can substantially reduce the bits spent on unwanted noise, but excessive denoising may remove meaningful details and impair reconstruction quality. Conversely, deblurring can improve image quality, but the recovered details may require additional bits to encode. These competing effects motivate jointly considering reconstructed image quality and bitrate when deciding which restoration operations to apply. Guided by the diagnosed degradations $\mathcal { G } _ { \delta } ^ { \prime } ( \cdot )$ , the agent follows a scheduling–execution– reflection pipeline. It first schedules restoration subtasks, such as denoising or deblurring. For each subtask, it applies $T - 1$ candidate restoration tools to the subtask input and includes the unchanged input as a no-restoration baseline, producing $T$ uncompressed candidate images $\mathbf { x } ^ { ( t ) } { } _ { t \in { \mathcal { T } } _ { \mathrm { p r e } } }$ , where $\mathcal { T } _ { \mathrm { p r e } } =$ $\{ 1 , \ldots , T \}$ . It then evaluates the compression outcomes of these candidates and uses the resulting feedback to refine the remaining restoration plan.

Specifically, our reflection is QR-aware and follows the SGQR criterion in Eq. 4. As noted in Sec. III-A, the clean ground-truth image $\mathbf { x } ^ { \star }$ is unavailable during compression. We therefore select the uncompressed candidate with the best no-reference image quality assessment (IQA) score from $\{ \mathbf { x } ^ { ( t ) } \} _ { t \in \mathcal { T } _ { \mathrm { p r e } } }$ and use it as a pseudo-reference, denoted by $\mathbf { x } ^ { ( t _ { \mathrm { r e f } } ) }$ . Each candidate is then encoded and decoded using the target codec $\mathcal { C }$ at a common probe quality-control setting q estimated from the QR preference $\lambda _ { \mathcal { T } } ,$ producing a bitstream $\mathbf { b } _ { \mathrm { p } } ^ { ( t ) }$ and a reconstruction ${ \widehat { \mathbf { x } } } _ { \mathrm { p } } ^ { ( t ) }$ . The selection procedure is formulated as:

$$
\begin{array} { r } { \begin{array} { c } { t ^ { \star } = \arg \underset { t \in \mathcal { T } _ { \mathrm { p r e } } } { \operatorname* { m i n } } \ E _ { \mathrm { S G Q R } } \left( R ( \mathbf { b } _ { \mathrm { p } } ^ { ( t ) } ) , \widehat { \mathbf { x } } _ { \mathrm { p } } ^ { ( t ) } , \mathbf { x } ^ { ( t _ { \mathrm { r e f } } ) } ; \mathcal { T } \right) , } \\ { \mathrm { s . t . , } \quad \left( \mathbf { b } _ { \mathrm { p } } ^ { ( t ) } , \widehat { \mathbf { x } } _ { \mathrm { p } } ^ { ( t ) } \right) = \mathcal { C } \left( \mathbf { x } ^ { ( t ) } ; q \right) , t _ { \mathrm { r e f } } = \arg \underset { t \in \mathcal { T } _ { \mathrm { p r e } } } { \operatorname* { m i n } } \ \mathcal { Q } _ { \mathrm { I Q A } } \left( \mathbf { x } ^ { ( t ) } \right) , } \end{array} } \end{array}\tag{5}
$$

where $\mathcal { Q } _ { \mathrm { I Q A } } ( \cdot )$ denotes a composite no-reference IQA score computed, following [39], [43], with lower values indicating better image quality. The terms $R ( \mathbf { b } _ { \mathrm { p } } ^ { ( t ) } )$ and ${ \widehat { \mathbf { x } } } _ { \mathrm { p } } ^ { ( t ) }$ denote the measured bitrate (in bits per pixel) and the decoded reconstruction of the restored candidate $\mathbf { x } ^ { ( t ) }$ , respectively, both obtained using the target codec C. The pseudo-reference $\bf \tilde { x } ^ { ( t _ { \mathrm { r e f } } ) }$ is held fixed when comparing candidates within the current subtask, allowing the SGQR evaluation to assess reconstruction quality against a shared reference according to the specific compression scenario I.

For each subtask, the agent selects the candidate $\mathbf { x } ^ { ( t ^ { \star } ) }$ with the lowest estimated SGQR cost and updates x $ \mathbf { x } ^ { ( t ^ { \star } ) }$ only if this cost is lower than that of the current image. Otherwise, it retains the current image x and refines the remaining restoration plan based on agent feedback. Once restoration is complete, the agent passes the preprocessed image x to the Coding Agent for the encoding procedure.

After-Coding Improvement Agent. This agent addresses potential residual degradations and compression-induced artifacts after decoding. It follows a pipeline nearly identical to that of the Improvement Agent, with two main differences. First, it uses a VLM to reassess the reconstructed image xe produced by the Coding Agent and identify residual degradations and artifacts to guide restoration planning. Second, its reflection considers image quality alone using IQA, following [27], [28]:

![](images/784773725a85645166d052d6b279d36975af5d681ae621c00a02a17100a2740e.jpg)

![](images/113cf5b718a32da1df44984c8c67084d0773e88f355576c4df98b9a69f902a86.jpg)

![](images/896b96315b651ae6a6fe4476a4c74cfd43c015758afa77cf492bdcbef74d4353.jpg)

![](images/e24cd724f7d072f065e602cda9a0f4ddae16173696cd2326bdce64ad5098eace.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompression

(a) Quality-rate curves on the Kodak dataset.  
![](images/39abf49e240b1c43cf119faf34b5aeb0a50138d025e5b404487528c4ec841991.jpg)

![](images/749d1b0db573133626099ef026dd279eab8cdeb31d68b8d389490e9dafd48857.jpg)

![](images/d17a31446597b8b6647ce70fed7b5c535675cee14318d0e3513fac8bc76dd79b.jpg)

![](images/ff7040ffa9a8a4538d73fe674ccbc236b0e554f54bfc02b89daac6edb4fe8cce.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompression

(b) Quality-rate curves on the Noisy Kodak dataset.  
![](images/a883790f65b3da35dfb513b45c4f534e7eb55d04be95fb130f6def2570189f3d.jpg)

![](images/76b0e4efb4be81e41a77082b6a416d110f54f06ff767d6ac7215ca416310f7a4.jpg)

![](images/c0c02ed40c91ca7340fd989132a4666c0a3545e0a79ac0f7cbbd42829d60e1d1.jpg)

![](images/83f8c13ad68b5bbf3635a9eb8912b5cb68e6acbd7477e774e146b2f715c80398.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompressio  
(c) Quality-rate curves on the Blur Kodak dataset.  
Fig. 3: Quality-rate performance on the (a) clean, (b) noisy, and (c) blurred Kodak benchmarks. Bitrate is measured in bits pe pixel (bpps), while reconstruction quality is evaluated against the original clean reference images.

$$
t ^ { \star } = \arg \operatorname* { m i n } _ { t \in \mathcal { T } _ { \mathrm { p o s t } } } \mathcal { Q } _ { \mathrm { I Q A } } \Big ( \widetilde { \mathbf { x } } ^ { ( t ) } \Big ) ,\tag{6}
$$

where $\widetilde { \mathbf { x } } ^ { ( t ) }$ denotes a candidate obtained by restoring the reconstructed image, and $\mathcal { T } _ { \mathrm { p o s t } }$ indexes the candidates for the current subtask, including the unchanged input as a norestoration baseline. Here, $\mathcal { Q } _ { \mathrm { I Q A } } ( \cdot )$ is the same no-reference IQA cost defined in Eq. 5, with lower values indicating better image quality. This IQA cost can be evaluated without access to the clean ground-truth image $\mathbf { x } ^ { \star }$

For each subtask, the agent selects the candidate $\widetilde { { \mathbf x } } ^ { ( t ^ { \star } ) }$ with the lowest IQA cost and updates $\widetilde \mathbf { x } \gets \widetilde \mathbf { x } ^ { ( t ^ { \star } ) }$ only if this cost is lower than that of the current image. Otherwise, it retains xe and refines the remaining restoration plan based on agent feedback. Once restoration is complete, the agent outputs the final reconstruction xe.

## D. Codec-Agnostic Coding Agent

Given the QR preference $\lambda _ { \mathcal { T } } .$ , the compression requirements specified by I, and the updated image x returned by the Improvement Agent, the Coding Agent searches for an appropriate coding configuration of the target black-box codec C to satisfy these requirements while balancing bitrate and reconstruction quality. It employs two codec-agnostic mechanisms: global QC search and AROIC.

Global QC Search. This mechanism identifies an appropriate global QC setting for the target codec C through feedbackdriven refinement. As illustrated in Fig. 2, the agent initializes a global QC parameter $q _ { g }$ based on the QR preference $\lambda _ { \mathcal { T } } ,$ explicit compression requirements (e.g., target bitrate and compression ratio), and codec-specific QC knowledge (e.g., guidance on QP usage in H.266/VVC) provided in the configuration ${ \mathcal { S } } .$ The codec then encodes and decodes x using $q _ { g } ,$ , and the measured bitrate and reconstruction quality are checked against the specified rate requirements and quality thresholds. If these requirements are satisfied, the current setting $q _ { g }$ is retained. Otherwise, the agent generates feedback identifying the unmet requirements and updates $q _ { g } \gets q _ { g } ^ { \mathrm { n e x t } }$ accordingly, aiming to improve reconstruction quality when it is unacceptable or bring the bitrate closer to the target while maintaining acceptable quality. This process continues until both the rate and quality requirements are satisfied or the iteration limit is reached. The resulting global QC parameter $q _ { g }$ is used in the subsequent AROIC procedure.

![](images/efa184ce0998821a12839b4873410331ab454f0728e1ee9d95e82d8b3633277e.jpg)

![](images/c96ad0072dace4c12762eb49c749f3126ee09bc33cc2eb9ea667c265a1d380c4.jpg)

![](images/70ac0d35d8c8356e4dbf9ea01584523433a6bbefe43511cf3df39a1774101511.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompression

![](images/b1e30a1302b66d77b2cb1d944a6e2b270ae52296c5aae052ffe26eb9cbcdd2df.jpg)

(a) Quality-rate curves on the CLIC2020 dataset.  
![](images/1e1ee5d439be289726d15d6924db1a8803382fd000574434fc51e130e11f178c.jpg)

![](images/b6c29515029fb0197d2e60a18e180bdc5169c56f2d59436a2616f781579efb9e.jpg)

![](images/bd297cb5895eed3519ef6ab8a98d4addb1b6f75ccfbb5ed5199e5786316fb918.jpg)

![](images/9e25bbc82d58bcb54c8daf98861c5a9325f07c7ff836029f9e681b2ac896ede5.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompression

(b) Quality-rate curves on the Noisy CLIC2020 dataset.  
![](images/c6046d083c733924839213aeec999e9bd3f245fe881249288dc5fffe6b2feea3.jpg)

![](images/5a5e1bde6f504f451096b248883a15efe5ac040fe304e608a5ba0f6689ccb582.jpg)

![](images/7ebd055c2a9115988e805d3d3928133da860cd9421598d3bfd0fbed34f3e890c.jpg)

![](images/ee9a7fb4ad8ab0497cdb4ab8409cc6a32a4189010cc8eb0ccba51d4ea35cfeda.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompression  
(c) Quality-rate curves on the Blurred CLIC2020 dataset.  
Fig. 4: Quality-rate performance on the (a) clean, (b) noisy, and (c) blurred CLIC2020 benchmarks. Bitrate is measured in bits per pixel (bpps), while reconstruction quality is evaluated against the original clean reference images.

Agentic ROI Coding. The relative importance of image regions varies across compression scenarios, making ROI coding an effective mechanism for allocating bits according to content importance. However, reliance on codec-internal mechanisms [26] can limit the transferability of conventional ROI coding methods across heterogeneous codecs. To enable codec-agnostic regional adaptation, AROIC treats the target codec C as a black box and delegates regional coding decisions, execution, and refinement to the Coding Agent, without requiring codec-specific architectural modifications or retraining. As illustrated in Fig. 2, AROIC follows a scheduling– execution–reflection workflow. Given the preprocessed image x returned by the Improvement Agent and the N ROI coordinates $\{ \Omega _ { i } \} _ { i = 1 } ^ { N }$ provided by the Perception Agent, AROIC separates x into ROI crops and a background image:

$$
\begin{array} { r l } & { \mathbf { x } _ { \mathrm { r o i } , i } = \mathrm { C r o p } ( \mathbf { x } ; \Omega _ { i } ) , \quad i = 1 , \ldots , N , } \\ & { \mathbf { x } _ { \mathrm { b g } } = ( \mathbf { 1 } - \mathbf { M } ) \odot \mathbf { x } , } \end{array}\tag{7}
$$

where M is the binary mask covering the union of the ROIs and ⊙ denotes element-wise multiplication.

First, the agent initializes the ROI and background QC settings from the global QC setting $q _ { g }$ and schedules $T - 1$ candidate coding plans conditioned on the configuration S, including the QR preference, compression requirements, image caption, and ROI information. These plans adjust the background QC, ROI QC, or both to explore different QR trade-offs, and may additionally downsample the background before encoding to reduce its coding cost.

Then, the agent executes each plan t by invoking the target codec to independently encode and decode the background and the N ROI crops using their respective QC settings, producing N + 1 bitstreams and corresponding component reconstructions. When background downsampling is used, the decoded background is upsampled to the original image

![](images/73f2bbd07e7e3c4532f2be03ce946811fef480b1f68158e57aecac62c290f17d.jpg)

![](images/294c4b8c02cf2530ac70316698e388b8525bd4555a4ce9050c1f7f09bfcd30d8.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompression Pseudo GT

(a) Quality-rate curves on the HEVC C and D datasets  
![](images/dfc613bf7918335ed14fa7b6b79bf008518cb11cb21c82afe668043d4b7d1bea.jpg)

![](images/c9a6d795480634354baf5d56ff07580d645a39b86d938821d0be26f3770c2be6.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompression Pseudo GT

(b) Quality-rate curves on the Noisy HEVC C and D datasets  
![](images/bdfb50e227ca8a28bafcc785a106d859923f2936e0bd0919f8b09996aad9771f.jpg)

![](images/b1a32169a75db97c642100875520be7ff3a782301fcc09a8fac682f99e59401e.jpg)  
PICA + Cheng2020 Cheng2020 PICA + VVC VVC AdaptICMH HiFiC DenoiseCompression Pseudo GT  
(c) Quality-rate curves on the Blurred HEVC C and D datasets

Fig. 5: Quality–rate performance on the (a) clean, (b) noisy, and (c) blurred benchmarks derived from HEVC Classes C and D. Bitrate is measured in bits per pixel (bpp), while object detection performance is evaluated on the reconstructed images using $\mathrm { m A P } _ { [ 0 . 5 0 : 0 . 9 5 ] }$ (%). The gray dashed lines indicate detection performance on uncompressed clean inputs.

resolution before the decoded ROI crops are placed at their corresponding spatial locations. The resulting aggregate coded representation $\bar { B ^ { ( t ) } }$ and full-image reconstruction $\widehat { { \mathbf x } ( t ) }$ are

$$
\begin{array} { r l } & { \mathcal { B } ^ { ( t ) } = \left( \mathbf { b } _ { \mathrm { b g } } ^ { ( t ) } , \{ \mathbf { b } _ { \mathrm { r o i } , i } ^ { ( t ) } \} _ { i = 1 } ^ { N } , \mathbf { m } ^ { ( t ) } \right) , } \\ & { \widehat { \mathbf { x } } ^ { ( t ) } = \mathrm { A g g } \left( \widehat { \mathbf { x } } _ { \mathrm { b g } } ^ { ( t ) } , \{ \widehat { \mathbf { x } } _ { \mathrm { r o i } , i } ^ { ( t ) } \} _ { i = 1 } ^ { N } ; \mathbf { m } ^ { ( t ) } \right) , } \end{array}\tag{8}
$$

where $\mathbf { b } _ { \mathrm { b g } } ^ { ( t ) }$ b1°) and ${ \bf b } _ { \mathrm { r o i , i } } ^ { ( t ) }$ denote the background and i-th ROI bitstreams, respectively, and $\mathbf { m } ^ { ( t ) }$ contains the metadata required for reconstruction, including ROI locations and background scaling information. The operator Agg spatially combines the background reconstruction $\widehat { \mathbf { x } } _ { \mathrm { b g } } ^ { ( t ) }$ at the original resolution with the decoded ROI crops $\{ \widehat { \mathbf { x } } _ { \mathrm { r o i } , i } ^ { ( t ) } \} _ { i = 1 } ^ { N }$ according to $\mathbf { m } ^ { ( t ) }$

Next, the agent evaluates the candidate coding plans indexed by $\mathcal { T } _ { \mathrm { r o i } } = \{ 1 , \dots , T \}$ using SGQR, with the uncompressed preprocessed image x serving as the shared quality reference and $R ( B ^ { ( t ) } )$ denoting the total coding rate in bits per pixel. The candidate set also includes a global-QC baseline $( \boldsymbol { B } ^ { ( T ) } , \widehat { \mathbf { x } } ^ { ( T ) } )$ , obtained by encoding the entire image (i.e., both ROIs and the

![](images/e8f58bcf5054436dfdb88e1e5708ca502048df3529e55e9ed02300221f80c9f8.jpg)

![](images/22c183a2ab28e832cbe5026701936ea3083cd5ce978dbeca661fd39f93b70404.jpg)  
Cheng2020 P & I Agents + Cheng2020 P & I & A Agents + Cheng2020 PICA + Cheng202

(a) Effectiveness of each agent with Cheng2020 on Blurred Kodak.  
![](images/ef6bfc3939f68eaadf92816f1b131901e701aed8a2f149f48edd974ab23c0b65.jpg)

![](images/bb3e4db48e51b4e1946130924ee2bf736b1b4a0f317aa0b15d81b5013fd387df.jpg)  
VVC P & I Agents + VVC P & I & A Agents + VVC PICA + VVC

(b) Effectiveness of each agent with VVC on Blurred Kodak.  
![](images/b7264c2804a5344a8dfa8d09f485e4856d5528570493d1e88fce2f11636b4085.jpg)

![](images/fbb3a684f40736b253935c3ce064630b0bd4a500150a5af3acd4f7b9f0a2fb72.jpg)  
PICA+VVC (w Qwen-2.5) PICA+VVC (w Qwen-3.0) PICA+VVC (w GPT-6.1)  
(c) Robustness Across VLM Agent Backbones on Noisy Kodak.

Fig. 6: Ablation studies of (a)–(b) the effectiveness of individual agents on the blurred Kodak dataset, where $\mathrm { ~ P , ~ I , ~ C , }$ and A denote the Perception, Improvement, Coding, and After-coding Improvement Agents, respectively, and (c) PICA with different VLMs as agent backbones on the noisy Kodak dataset.

background) at the unchanged global QC setting $q _ { g } .$ . The best candidate is then selected as:

$$
t ^ { \star } = \arg \operatorname* { m i n } _ { t \in \mathcal { T } _ { \mathrm { r o i } } } \mathcal { E } _ { \mathrm { S G Q R } } \Big ( R ( \mathcal { B } ^ { ( t ) } ) , \widehat { \mathbf { x } } ^ { ( t ) } , \mathbf { x } ; \mathcal { T } \Big ) .\tag{9}
$$

Finally, the agent reflects on the selected candidate $\widehat { \mathbf { x } } ^ { ( t ^ { \star } ) }$ . If the candidate achieves a lower SGQR cost than the global-QC baseline while satisfying both the quality and bitrate requirements, the agent updates $( \vec { B } , \widetilde { \mathbf { x } } ) \gets ( \vec { B ^ { ( t ^ { \star } ) } } , \widehat { \mathbf { x } } ^ { ( t ^ { \star } ) } )$ . Otherwise, it generates actionable feedback and reschedules the coding plans accordingly. This feedback guides further adjustments to ROI QC, background QC, or background downsampling, after which the revised plans are executed and evaluated again. The process terminates when an QR improvement is obtained or the iteration limit is reached. AROIC then returns the selected coded representation B and reconstruction $\widetilde { \mathbf { x } } ,$ or the global-QC baseline if no candidate satisfies the acceptance criteria.

## E. Models and Tools

We summarize the pretrained models and task-specific tools used in PICA and its evaluation below:

Bpps: 0.1095 / LPIPS: 0.3982

![](images/12c0b98ad918a20525563acf7845aa4cb3264c8c6c8ed0043bf7861a40f16db6.jpg)  
Fig. 7: Qualitative comparison of PICA using VVC and Cheng2020 as base codecs against their corresponding baseline codecs.

Denoising: X-Restormer [44], SwinIR [45], MPRNet [46], MAXIM [47], Restormer [48], and NAFNet [49].

Defocus Deblurring: DRBNet [50], IFAN [51], Restormer [48], Diff-Plugin [52], and LaKDNet [53].

Super-resolution: DiffBIR [54], X-Restormer [44],

SwinIR [45], HAT [55], OSEDiff [56], PiSA-SR [57], SwinFIR [58], NAFSSR [59], and HMANet [60].

Compression Artifact Removal: FBCNN [61], SwinIR [45], WildIR [62] and NOLAIR [63].

Detection: YOLOv8 [64] for object-based ROI detection and EasyOCR [65] for text-based ROI detection. Faster R-CNN [66] with a ResNeXt-101-32x8d-FPN backbone for machine-vision compression evaluation.

IQA: CLIP-IQA+ [67], MANIQA [68], MUSIQ [69], and NIQE [70] for no-reference IQA, as [27], [28]. LPIPS [18] and DISTS [71] for full-reference perceptual comparisons.

VLMs: DepictQA-family model [43], [72] for qualitative IQA, including degradation identification and severity estimation. Qwen3-VL-8B-Instruct [38] supports user instruction interpretation, image analysis, task planning, tool invocation, and feedback-guided plan refinement across all agents.

## IV. EXPERIMENTS

## A. Datasets and Protocols.

Datasets. We evaluate our methods on three datasets: Kodak, CLIC2020, and HEVC. Kodak [29] consists of 24 natural images and is a widely used benchmark for image compression. The CLIC2020 validation set [30] contains 102 highresolution images spanning diverse scenes and textures, enabling evaluation on more varied visual content. For machinevision compression, we use images sampled from the HEVC test sequences [31], with ground-truth object categories and bounding boxes provided by SFU-HW-Objects-v3 [73], [74]. Specifically, we select 8 sequences from Classes C and D, divide each sequence into 6 equal temporal segments, and extract the $1 ^ { s t }$ frame from each segment, yielding 48 images. Degradation We consider two common image degradations that can substantially affect compression performance, which are noise and blur [19]–[21], [24]. Following prior agentic image restoration methods [27], [28], we synthetically generate additive Gaussian noise and defocus blur from the original images. Specifically, we add Gaussian noise with a standard deviation of $\sigma ~ = ~ 1 5$ in the 8-bit pixel intensity range. To generate defocus blur, we use a circular disk pointspread function with a radius of 3 pixels, smoothed with a $3 \times 3$ Gaussian filter, to blur each image. We apply each degradation independently to the images in the Kodak dataset, the CLIC2020 validation set, and the HEVC test set. The original images form the clean benchmarks. Thus, each dataset has 3 benchmarks: clean, Gaussian-noise, and defocus-blur. In total, there are 9 compression benchmarks: Kodak, Noisy Kodak, Blurred Kodak, CLIC2020, Noisy CLIC2020, Blurred CLIC2020, HEVC, Noisy HEVC, and Blurred HEVC.

Evaluation Metrics. For the 6 clean, noise and blur benchmarks derived from the Kodak and CLIC2020 validation sets, we evaluate reconstruction fidelity using PSNR, the most widely used objective metric for image compression. We also report 3 perceptual metrics: LPIPS and DISTS, computed per image and averaged over each benchmark, and FID, computed at the domain level by comparing the distributions of reconstructed images and their clean references. Here, clean references denote the original ground-truth images without synthetic degradation. For machine-vision compression, we evaluate object detection performance using mAP<sub>[0.50:0.95]</sub>, following [75]. Faster R-CNN [66] with a ResNeXt-101- 32x8d-FPN backbone is used as the downstream detector for all methods. All bitrates are reported in bits per pixel (Bpps), calculated by dividing the number of coded bits by the number of pixels in the corresponding image.

## B. Baseline Codecs.

We evaluate our PICA using two representative codecs: the conventional H.266/VVC codec in intra-frame coding mode (i.e., VVC) [2] and the neural image codec Cheng2020 [32]. For VVC, we use the VTM-24.0 reference software [76] with encoder\_intra\_vtm.cfg and the reduced-runtime configuration. For Cheng2020, we use the Cheng2020Anchor variant implemented in CompressAI-1.2.8 [77]. For each codec, we assume only that the agent has high-level prior knowledge of how a single control parameter affects bitrate and reconstruction quality. That is the quantization parameter (i.e., QP) for VVC and the quality level (i.e., q) for Cheng2020. Our PICA is training-free and requires no knowledge of codec internals or additional codec-specific expertise

In addition to the two base codecs for standard image compression, we include three specialized codecs designed and optimized for specific compression scenarios: HiFiC [7] for perceptual compression, Adapt-ICMH [17] for machinevision compression, and DenoiseCompression [19] for compression of noisy images. We use an open-source PyTorch reimplementation [78] of HiFiC and the official open-source implementations of Adapt-ICMH and DenoiseCompression.

## C. Experiment Results.

We evaluate compression performance across different compression scenarios and input conditions on Kodak, CLIC2020, and HEVC in Fig. 3, Fig. 4, and Fig. 5, respectively. In general, PICA consistently improves the QR performance of both baseline codecs across the evaluated scenarios and degradations, demonstrating its ability to adapt Cheng2020 and VVC to diverse compression objectives. For example, in perceptual compression, PICA achieves average bitrate savings at matched LPIPS of 34.52%, 44.90%, and 62.66% over Cheng2020, and 28.96%, 43.17%, and 59.95% over VVC, on the clean, noisy, and blurred Kodak benchmarks. For machine-vision compression on HEVC Class D, PICA achieves average relative gains in m $\mathrm { A P _ { [ 0 . 5 0 : 0 . 9 5 ] } }$ of 52.58%, 40.94%, and 55.77% over Cheng2020, and 62.28%, 80.46%, and 55.56% over VVC, under clean, noisy, and blurred input conditions, respectively, at matched bitrates. Notably, even though both baseline codecs are specifically optimized for standard compression, PICA achieves further average bitrate savings at matched PSNR of 8.96%, 22.86%, and 31.07% over

Cheng2020, and 7.49%, 22.99%, and 35.81% over VVC, on the clean, noisy, and blurred Kodak benchmarks.

On the other hand, we observe that codecs optimized for specific compression objectives or input conditions often exhibit limited generalization across scenarios and degradations. For instance, HiFiC achieves strong perceptual performance on clean images, but requires 185.89% and 111.57% more bitrate than PICA+VVC at matched PSNR for standard compression on clean Kodak and CLIC2020, respectively. Its perceptual performance also deteriorates under input degradations. On the degraded Kodak and CLIC2020 benchmarks, HiFiC exhibits substantially higher LPIPS scores than PICA+VVC at comparable bitrates, indicating much lower perceptual quality. Similarly, DenoiseCompression, specifically optimized for standard compression of noisy images, exhibits limited robustness to blur. Adapt-ICMH performs well in machine-vision compression on clean images but remains less competitive in standard and perceptual compression, with its machine-vision performance also falling behind PICA under degradations.

## D. Ablation Study

Effectiveness of Individual Agents. We evaluate the effectiveness of the agents on the blurred Kodak benchmark by progressively incorporating them into Cheng2020 and VVC. Specifically, we introduce two intermediate variants: the baseline codec equipped with the Perception and Improvement Agents (i.e., P & I), and a variant that additionally incorporates the After-coding Improvement Agent (i.e., P & I & A). Fig. 6 (a) and (b) compare these variants with the baseline codecs and the full PICA framework. These results demonstrate the complementary benefits of pre-coding restoration, post-coding restoration, and AROIC. Our observation is that, by enabling QR-aware pre-coding restoration, the P & I variant reduces LPIPS by 6.87% and 13.94%, and DISTS by 15.83% and 27.15%, relative to Cheng2020 and VVC, respectively. Adding the After-coding Improvement Agent for quality-only post-coding restoration further reduces LPIPS by 1.76% and 5.52%, and DISTS by 6.83% and 14.07%, relative to the corresponding P & I variants, which use pre-coding restoration only. The full PICA framework achieves the best QR performance among the evaluated variants. Compared with the restoration-only P & I & A variants, incorporating the Coding Agent with AROIC further reduces LPIPS by 7.62% and 11.56%, and DISTS by 10.05% and 22.66%, for Cheng2020 and VVC, respectively. These results demonstrate the complementary benefits of pre-coding restoration, postcoding restoration, and AROIC.

Robustness Across VLM Agent Backbones. To evaluate PICA’s robustness to the choice of VLM agent backbone, we compare three VLMs: two open-source models, Qwen2.5- VL-7B-Instruct [37] and Qwen3-VL-8B-Instruct [38], and one commercial model, GPT-6.1-Sol [79]. Here, the VLM agent backbone refers to the model that supports user instruction interpretation, image analysis, task planning, tool invocation, and feedback-guided plan refinement across all agents. Fig. 6 (c) presents the results on the noisy Kodak benchmark. The three backbones yield nearly identical QR performance, suggesting that PICA’s effectiveness is robust to the choice of backbone among the evaluated VLMs. These results support flexible deployment with different VLM backbones, allowing the choice to accommodate available computational resources and deployment requirements.

## E. Qualitative Study

In addition to the quantitative evaluation, Fig. 7 presents qualitative comparisons using four samples each from the noisy and blurred Kodak benchmarks. We compare Cheng2020 and VVC with their PICA-enhanced counterparts. Across both degradations, PICA produces reconstructions with better visual quality and lower LPIPS scores while reducing bitrate.

For noisy images, noise introduces unwanted image variations that impair visual quality and consume additional bits during encoding. This is particularly evident on the statue’s face in the $3 ^ { r d }$ row and in the sky in the $7 ^ { t h }$ row. Although prior research suggests that compression itself can have a denoising effect [25], our visual comparisons show that, in practice, Cheng2020 and VVC still retain visible residual noise in their reconstructions. PICA suppresses this noise more effectively, improving perceptual quality while reducing the bits spent on encoding unwanted content.

For blurred images, blur substantially degrades fine details, as illustrated by the water surface textures in the sixth row and the rock textures in the eighth row. The baseline codecs do not effectively restore these details, whereas PICA produces clearer and more discernible textures through deblurring. Despite restoring texture details, PICA achieves lower LPIPS scores without increasing bitrate relative to the corresponding baselines. These observations further support the effectiveness of our SGQR strategy in jointly considering bitrate and reconstruction quality when selecting restoration and coding plans, rather than pursuing restoration quality alone.

## V. CONCLUSION

In this work, we introduced PICA, a training-free multiagent framework that adapts existing image codecs to diverse compression scenarios and potentially degraded inputs. It combines four collaborative agents with SGQR evaluation and feedback-guided reflection to coordinate restoration and coding according to specific user requirements and potential input degradations. Extensive experiments on the Kodak, CLIC2020, and HEVC datasets demonstrate that PICA improves the quality–rate performance of both the conventional codec H.266/VVC and the neural codec Cheng2020 across standard, perceptual, and machine-vision compression under clean, noisy, and blurred input conditions. These improvements are achieved without codec-specific architectural modifications, retraining on large-scale datasets, or detailed knowledge of codec internals. By decoupling scenario adaptation from extensive codec redevelopment and scenario-specific training, PICA enables advances in standard codecs to benefit a broader range of image processing and transmission applications. More broadly, this work connects agentic decision-making with image compression research and motivates further exploration of VLM-based agents for compression and related visual processing tasks.

## REFERENCES

[1] G. K. Wallace, “The JPEG still picture compression standard,” IEEE Transactions on Consumer Electronics, 1992.

[2] B. Bross, Y.-K. Wang, Y. Ye, S. Liu, J. Chen, G. J. Sullivan, and J.- R. Ohm, “Overview of the versatile video coding (VVC) standard and its applications,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 31, no. 10, pp. 3736–3764, 2021.

[3] J. Balle, V. Laparra, and E. P. Simoncelli, “End-to-end optimized image´ compression,” in International Conference on Learning Representations, 2017.

[4] J. Balle, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational´ image compression with a scale hyperprior,” in International Conference on Learning Representations, 2018.

[5] D. Minnen, J. Balle, and G. D. Toderici, “Joint autoregressive and´ hierarchical priors for learned image compression,” in Advances in Neural Information Processing Systems, vol. 31, 2018, pp. 10 794– 10 803.

[6] J. Lee, S. Cho, and S.-K. Beack, “Context-adaptive entropy model for end-to-end optimized image compression,” in International Conference on Learning Representations, 2019.

[7] F. Mentzer, G. Toderici, M. Tschannen, and E. Agustsson, “High-fidelity generative image compression,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[8] Z. Chen, S. Gu, G. Lu, and D. Xu, “Exploiting intra-slice and inter-slice redundancy for learning-based lossless volumetric image compression,” vol. 31, pp. 1697–1707, 2022.

[9] Z. Chen, G. Lu, Z. Hu, S. Liu, W. Jiang, and D. Xu, “Lsvc: A learningbased stereo video compression framework,” June 2022, pp. 6073–6082.

[10] Z. Chen, L. Relic, R. Azevedo, Y. Zhang, M. Gross, D. Xu, L. Zhou, and C. Schroers, “Neural video compression with spatio-temporal crosscovariance transformers,” 2023, pp. 8543–8551.

[11] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized gaussian mixture likelihoods and attention modules,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7939–7948.

[12] Y. Blau and T. Michaeli, “Rethinking lossy compression: The ratedistortion-perception tradeoff,” in Proceedings of the 36th International Conference on Machine Learning, vol. 97. PMLR, 2019, pp. 6755– 6764.

[13] E. Agustsson, M. Tschannen, F. Mentzer, R. Timofte, and L. Van Gool, “Generative adversarial networks for extreme learned image compression,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 221–231.

[14] M. Careil, M. J. Muckley, J. Verbeek, and S. Lathuiliere, “Towards image\` compression with perfect realism at ultra-low bitrates,” in International Conference on Learning Representations, 2024.

[15] L. D. Chamain, F. Racape, J. B ´ egaint, A. Pushparaja, and S. Feltman,´ “End-to-end optimized image compression for machines, a study,” in Data Compression Conference, 2021, pp. 163–172.

[16] L. Liu, Z. Hu, Z. Chen, and D. Xu, “ICMH-Net: Neural image compression towards both machine vision and human vision,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 8047– 8056.

[17] H. Li, S. Li, S. Ding, W. Dai, M. Cao, C. Li, J. Zou, and H. Xiong, “Image compression for machine and human vision with spatial-frequency adaptation,” in European Conference on Computer Vision, 2024, pp. 382–399.

[18] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 586–595.

[19] K. L. Cheng, Y. Xie, and Q. Chen, “Optimizing image compression via joint learning with denoising,” in Computer Vision – ECCV 2022, ser. Lecture Notes in Computer Science, vol. 13679. Springer, 2022, pp. 56–73.

[20] J. Ye, H. Yeo, J. Park, and D. Han, “AccelIR: Task-aware image compression for accelerating neural restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 18 216–18 226.

[21] B. Brummer and C. De Vleeschouwer, “On the importance of denoising when learning to compress images,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, pp. 2440– 2448.

[22] S. Cai, L. Chen, S. Zhong, L. Yan, J. Zhou, and X. Zou, “Make lossy compression meaningful for low-light images,” in Proceedings of the

AAAI Conference on Artificial Intelligence, vol. 38, no. 8, 2024, pp. 8236–8245.

[23] Y.-H. Chen, K.-W. Ho, S.-R. Tsai, G.-H. Lin, A. Gnutti, W.-H. Peng, and R. Leonardi, “Transformer-based learned image compression for joint decoding and denoising,” in 2024 Picture Coding Symposium, 2024, pp. 1–5.

[24] Z. Wan, J. Zhao, Y. Ding, J. Guan, L. Zhang, and T. Ogawa, “Joint optimization for image compression and deblurring via blur-aware guidance,” in Proceedings of the 7th ACM International Conference on Multimedia in Asia, ser. MMAsia ’25. New York, NY, USA: Association for Computing Machinery, 2025. [Online]. Available: https://doi.org/10.1145/3743093.3770957

[25] A. Zafari, X. Chen, and S. Jalali, “Zero-shot denoising via neural compression: Theoretical and algorithmic framework,” Advances in Neural Information Processing Systems, vol. 38, pp. 152 020–152 056, 2026.

[26] Y. Gao, X. Li, X. Pan, R. Feng, B. Li, Y. Qi, Y. Lu, Z. Cheng, Z. Chen, and J. Ostermann, “Comp-x: On defining an interactive learned image compression paradigm with expert-driven llm agent,” arXiv preprint arXiv:2508.15243, 2025.

[27] K. Zhu, J. Gu, Z. You, Y. Qiao, and C. Dong, “An intelligent agentic system for complex image restoration problems,” in International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=3RLxccFPHz

[28] Y. Zuo, Q. Zheng, M. Wu, X. Jiang, R. Li, J. Wang, Y. Zhang, G. Mai, L. V. Wang, J. Zou, X. Wang, M.-H. Yang, and Z. Tu, “4KAgent: Agentic any image to 4K super-resolution,” in Advances in Neural Information Processing Systems, 2025. [Online]. Available: https://arxiv.org/abs/2507.07105

[29] R. Franzen, “Kodak lossless true color image suite,” https://r0k.us/ graphics/kodak/, 1999.

[30] G. Toderici, W. Shi, R. Timofte, L. Theis, J. Balle, E. Agustsson,´ N. Johnston, and F. Mentzer, “Workshop and challenge on learned image compression (CLIC2020),” https://archive.compression.cc/, 2020.

[31] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the high efficiency video coding (HEVC) standard,” IEEE Transactions on circuits and systems for video technology, vol. 22, no. 12, pp. 1649– 1668, 2012.

[32] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized gaussian mixture likelihoods and attention modules,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. [Online]. Available: https://arxiv.org/abs/2001.01568

[33] A. Skodras, C. Christopoulos, and T. Ebrahimi, “The JPEG 2000 still image compression standard,” IEEE Signal Processing Magazine, vol. 18, no. 5, pp. 36–58, 2001. [Online]. Available: https://ieeexplore.ieee.org/document/952804

[34] S. Franklin and A. Graesser, “Is it an agent, or just a program?: A taxonomy for autonomous agents,” in Intelligent Agents III: Agent Theories, Architectures, and Languages. Springer, 1997, pp. 21–35.

[35] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “ReAct: Synergizing reasoning and acting in language models,” in International Conference on Learning Representations, 2023. [Online]. Available: https://arxiv.org/abs/2210.03629

[36] OpenAI, “GPT-4 technical report,” arXiv preprint arXiv:2303.08774, 2023. [Online]. Available: https://arxiv.org/abs/2303.08774

[37] S. Bai, K. Chen, X. Liu et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025. [Online]. Available: https: //arxiv.org/abs/2502.13923

[38] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge et al., “Qwen3-vl technical report,” arXiv preprint arXiv:2511.21631, 2025.

[39] Z. You, Z. Li, J. Gu, Z. Yin, T. Xue, and C. Dong, “Depicting beyond scores: Advancing image quality assessment through multimodal language models,” in European Conference on Computer Vision, 2024, pp. 259–276.

[40] H. Chen, W. Li, J. Gu, J. Ren, S. Chen, T. Ye, R. Pei, K. Zhou, F. Song, and L. Zhu, “RestoreAgent: Autonomous image restoration agent via multimodal large language models,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[41] X. Jiang, G. Li, B. Chen, and J. Zhang, “Multi-agent image restoration,” arXiv preprint arXiv:2503.09403, 2025. [Online]. Available: https: //arxiv.org/abs/2503.09403

[42] Y. Lin, Z. Lin, H. Chen, P. Pan, C. Li, S. Chen, K. Wen, Y. Jin, W. Li, and X. Ding, “JarvisIR: Elevating autonomous driving perception with intelligent image restoration,” in Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, 2025, pp. 22 369–22 380.

[43] Z. You, J. Gu, X. Cai, Z. Li, K. Zhu, C. Dong, and T. Xue, “Enhancing descriptive image quality assessment with a large-scale multi-modal dataset,” IEEE Transactions on Image Processing, 2025. [Online]. Available: https://arxiv.org/abs/2405.18842

[44] X. Chen, Z. Li, Y. Pu, Y. Liu, J. Zhou, Y. Qiao, and C. Dong, “A comparative study of image restoration networks for general backbone network design,” in European Conference on Computer Vision, 2024. [Online]. Available: https://arxiv.org/abs/2310.11881

[45] J. Liang, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, “SwinIR: Image restoration using swin transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2021. [Online]. Available: https://arxiv.org/abs/2108.10257

[46] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, “Multi-stage progressive image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. [Online]. Available: https://arxiv.org/abs/2102.02808

[47] Z. Tu, H. Talebi, H. Zhang, F. Yang, P. Milanfar, A. Bovik, and Y. Li, “MAXIM: Multi-axis MLP for image processing,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. [Online]. Available: https://arxiv.org/abs/2201.02973

[48] S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, and M.-H. Yang, “Restormer: Efficient transformer for high-resolution image restoration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. [Online]. Available: https://arxiv.org/abs/2111.09881

[49] L. Chen, X. Chu, X. Zhang, and J. Sun, “Simple baselines for image restoration,” in European Conference on Computer Vision, 2022. [Online]. Available: https://arxiv.org/abs/2204.04676

[50] L. Ruan, B. Chen, J. Li, and M. Lam, “Learning to deblur using light field generated and real defocus images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16 304–16 313. [Online]. Available: https: //arxiv.org/abs/2204.00367

[51] J. Lee, H. Son, J. Rim, S. Cho, and S. Lee, “Iterative filter adaptive network for single image defocus deblurring,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 2034–2042. [Online]. Available: https://arxiv.org/abs/2108. 13610

[52] Y. Liu, Z. Ke, F. Liu, N. Zhao, and R. W. H. Lau, “Diff-Plugin: Revitalizing details for diffusion-based low-level tasks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. [Online]. Available: https://arxiv.org/abs/2403.00644

[53] L. Ruan, M. Bemana, H.-P. Seidel, K. Myszkowski, and B. Chen, “Revisiting image deblurring with an efficient ConvNet,” arXiv preprint arXiv:2302.02234, 2023. [Online]. Available: https://arxiv.org/abs/2302. 02234

[54] X. Lin, J. He, Z. Chen, Z. Lyu, B. Dai, F. Yu, Y. Qiao, W. Ouyang, and C. Dong, “DiffBIR: Toward blind image restoration with generative diffusion prior,” in European Conference on Computer Vision, 2024. [Online]. Available: https://www.ecva.net/papers/eccv 2024/papers ECCV/html/7690 ECCV 2024 paper.php

[55] X. Chen, X. Wang, J. Zhou, Y. Qiao, and C. Dong, “Activating more pixels in image super-resolution transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22 367–22 377. [Online]. Available: https: //arxiv.org/abs/2205.04437

[56] R. Wu, L. Sun, Z. Ma, and L. Zhang, “One-step effective diffusion network for real-world image super-resolution,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: https://arxiv.org/abs/2406.08177

[57] L. Sun, R. Wu, Z. Ma, S. Liu, Q. Yi, and L. Zhang, “Pixel-level and semantic-level adjustable super-resolution: A dual-LoRA approach,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2025/html/ Sun Pixel-level and Semantic-level Adjustable Super-resolution A Dual-LoRA Approach CVPR 2025 paper.html

[58] D. Zhang, F. Huang, S. Liu, X. Wang, and Z. Jin, “SwinFIR: Revisiting the SwinIR with fast fourier convolution and improved training for image super-resolution,” arXiv preprint arXiv:2208.11247, 2022. [Online]. Available: https://arxiv.org/abs/2208.11247

[59] X. Chu, L. Chen, and W. Yu, “NAFSSR: Stereo image super-resolution using NAFNet,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2022, pp. 1239–1248. [Online]. Available: https://arxiv.org/abs/2204.08714

[60] S.-C. Chu, Z.-C. Dou, J.-S. Pan, S. Weng, and J. Li, “HMANet: Hybrid multi-axis aggregation network for image super-resolution,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024, pp. 6257–6266. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/ html/Chu HMANet Hybrid Multi-Axis Aggregation Network for Image Super-Resolution CVPRW 2024 paper.html

[61] J. Jiang, K. Zhang, and R. Timofte, “Towards flexible blind JPEG artifacts removal,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 4997–5006. [Online]. Available: https://arxiv.org/abs/2109.14573

[62] Z. Luo, F. K. Gustafsson, Z. Zhao, J. Sjolund, and T. B. Sch¨ on, “Photo-¨ realistic image restoration in the wild with controlled vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2024, pp. 6641–6651.

[63] J. Kim and K. M. Lee, “Noise-free one-step LoRA for taskdriven image restoration with diffusion priors,” in Asian Conference on Computer Vision, 2026, accepted. [Online]. Available: https: //arxiv.org/abs/2607.25390

[64] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” Open-source software, 2023, software citation; YOLOv8 has no accompanying formal research paper. [Online]. Available: https: //github.com/ultralytics/ultralytics

[65] Jaided AI, “EasyOCR,” Open-source software, n.d., accessed 2026-10- 27. [Online]. Available: https://github.com/JaidedAI/EasyOCR

[66] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in Advances in Neural Information Processing Systems, vol. 28, 2015. [Online]. Available: https://arxiv.org/abs/1506.01497

[67] J. Wang, K. C. K. Chan, and C. C. Loy, “Exploring CLIP for assessing the look and feel of images,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, 2023, pp. 2555–2563. [Online]. Available: https://arxiv.org/abs/2207.12396

[68] S. Yang, T. Wu, S. Shi, S. Lao, Y. Gong, M. Cao, J. Wang, and Y. Yang, “MANIQA: Multi-dimension attention network for no-reference image quality assessment,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2022, pp. 1191–1200. [Online]. Available: https://arxiv.org/abs/2204.08958

[69] J. Ke, Q. Wang, Y. Wang, P. Milanfar, and F. Yang, “MUSIQ: Multiscale image quality transformer,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 5148–5157. [Online]. Available: https://arxiv.org/abs/2108.05997

[70] A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a “completely blind” image quality analyzer,” IEEE Signal Processing Letters, vol. 20, no. 3, pp. 209–212, 2013. [Online]. Available: https: //live.ece.utexas.edu/research/Quality/niqe spl.pdf

[71] K. Ding, K. Ma, S. Wang, and E. P. Simoncelli, “Image quality assessment: Unifying structure and texture similarity,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 5, pp. 2567–2581, 2022. [Online]. Available: https://arxiv.org/abs/2004.07728

[72] Z. You, Z. Li, J. Gu, Z. Yin, T. Xue, and C. Dong, “Depicting beyond scores: Advancing image quality assessment through multi-modal language models,” in European Conference on Computer Vision, 2024, pp. 259–276. [Online]. Available: https://arxiv.org/abs/2312.08962

[73] H. Choi, E. Hosseini, S. Ranjbar Alvar, R. A. Cohen, and I. V. Bajic,´ “A dataset of labelled objects on raw video sequences,” Data in Brief, vol. 34, p. 106701, 2021.

[74] H. Choi, R. Nguyen, S. Keating, and I. Bajic, “SFU-HW-Objects-v3,” Mendeley Data, V3, 2024.

[75] L. Liu, Z. Chen, Z. Hu, and D. Xu, “An efficient adaptive compression method for human perception and machine vision tasks,” Pattern Recognition, vol. 180, p. 114421, 2026.

[76] Joint Video Experts Team (JVET), “VVC test model (VTM) reference software,” 2026, version 24.0. [Online]. Available: https: //vcgit.hhi.fraunhofer.de/jvet/VVCSoftware VTM/-/tags/VTM-24.0

[77] J. Begaint, F. Racap ´ e, S. Feltman, and A. Pushparaja, “CompressAI:´ A PyTorch library and evaluation platform for end-to-end compression research,” arXiv preprint arXiv:2011.03029, 2020.

[78] G. Han and J. Tan, “High-fidelity generative image compression: PyTorch implementation,” GitHub repository, 2020, accessed: 2026-10-31. [Online]. Available: https://github.com/Justin-Tan/ high-fidelity-generative-compression

[79] OpenAI, “Using GPT-6,” https://developers.openai.com/api/docs/guides/ latest-model, accessed: 2026-10-31.