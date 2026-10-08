# Difusion-Generated Image Watermarking: A Two-Axis Taxonomy and Three Protocol-Bounded Case Studies<sup>⋆</sup>

Sung Ju Lee<sup>1</sup> and Nam Ik Cho<sup>1</sup>

Department of ECE & INMC, Seoul National University, Korea {thomas11809,nicho}@snu.ac.kr

Abstract. Watermarking difusion-generated images requires balancing provenance signals with image quality, robustness, and computational cost. This work organizes methods along two axes: insertion mechanism and primary signal-bearing representation, and formalizes a representative z<sub>T</sub>-Fourier pipeline for verification and identification. We then use the taxonomy to structure three protocol-bounded case studies. The first examines associations among frequency integrity, detection, quality, and cropping behavior. The second revisits persistence under seedlinked and seed-independent editing and formulates a scoped Semantic Imprinting Hypothesis without claiming a localized carrier or causal mechanism. The third studies single-shot VAE-latent phase modulation, including its eficiency, regeneration robustness, and robustness– quality operating points. Finally, we separate four content-level attack families from model/pipeline adaptation, propose corresponding evaluation protocols and testable conjectures for parameter-tuning threats, and identify additional temporal extensions for video. These analyses do not establish a universal ranking; instead, they provide a framework for matched, protocol-aware comparisons of watermarking systems for difusion-generated images.

Keywords: Difusion Model Watermarking · Watermarking Taxonomy · Semantic Watermarking · Generative Editing Robustness · Frequency Integrity

## 1 Introduction

The rapid evolution of generative AI, most notably text-to-image Latent Diffusion Models (LDMs), has democratized high-quality visual content creation. While early defense research focused primarily on synthetic image detection, modern media protection faces a broader challenge: verifying whether content has been misappropriated or altered via prompt-driven inpainting and semantic editing. Traditional pixel-domain watermarks and imperceptible noise injection [2, 5] fail in this paradigm. Because difusion-based editors re-synthesize images using learned generative priors, they tend to suppress subtle watermark perturbations that are weakly coupled to perceptually salient content [22, 30]. This vulnerability requires embedding watermark signals directly into the generative process itself.

To address this, semantic watermarking has emerged as a promising alternative, with the literature referring to these methods as either sampling-based or inversion-based, depending on the underlying design perspective. These methods embed a watermark signal directly into the difusion model’s initial Gaussian noise seed $z _ { T }$ . During generation, the noisy latent is progressively denoised into the VAE latent $z _ { 0 }$ and finally decoded into the pixel space. Existing approaches manipulate frequency patterns via predefined rules [4,25], encode payloads while preserving Gaussian noise statistics [27], or combine inversion with iterative optimization to embed watermark signals [29]. By embedding the watermark into the initial noise and propagating it through the difusion sampling process, semantic watermarking can remain detectable under generative re-synthesis [15,30]. However, this paradigm introduces distinct trade-ofs: improper frequency-spectrum manipulation can degrade generation quality, whereas optimization-heavy methods incur substantial computational overhead.

Rather than merely comparing the performance of individual methods, this work examines difusion-generated image watermarking from the perspective of the conditions under which embedded signals remain recoverable throughout generation, attacks, and detection. We organize the design space along two orthogonal axes, insertion mechanism and target representation space, providing a structured framework for analyzing how design choices shape the tradeofs among robustness, visual quality, and computational eficiency. Within this framework, we clarify how latent-space strategies overcome the limitations of pixel-domain watermarking, identify their remaining challenges, and outline an expanded threat model for future generative media protection.

Our analysis focuses on difusion-generated static images, covering representative state-of-the-art latent watermarking methods together with foundational pixel-domain baselines for reference. Non-difusion generative models (e.g., GANs) are excluded from the scope, while video watermarking is discussed only where temporal dynamics extend the image-centric threat model. The remainder of this paper is organized as follows. Sections 2 and 3 introduce the proposed two-axis taxonomy and formal system model. Sections 4 to 6 present three case studies, under their respective evaluation protocols, on frequency integrity and cropping, watermark persistence under generative editing, and the eficiency and regeneration robustness of phase modulation in the VAE latent space, respectively. Section 7 separates attacks on existing watermarked images from adaptation of the model or pipeline, presents evaluation guidance and three testable conjectures for parameter tuning, and briefly discusses the extension to video. Section 8 concludes with key future research directions.

Table 1: Taxonomy matrix of generative-image watermarking methods by insertion mechanism and primary signal-bearing representation (Sec. 2.3). Reproduced from [18].
<table><tr><td rowspan="2">Representation</td><td colspan="3">Mechanism</td></tr><tr><td>Training-free</td><td>Opt.-based</td><td>Learning-based</td></tr><tr><td>Initial noise (zT)</td><td>Tree-Ring [25] (freq.); RingID [4] (freq.); G.Shading [27] (spatial); HSTR/HSQR [15] (freq.); WIND [1] (freq., seed)</td><td>ZoDiac [29] (freq.)</td><td>DiffuseTrace [19] (spatial)</td></tr><tr><td>Intermediate (zt)</td><td></td><td>ROBIN [14]</td><td></td></tr><tr><td>VAE latent (z0)</td><td>PhaseMark [16] (freq., phase)</td><td>FreqMark [9] (freq.)</td><td>RoSteALS [3] (spatial); Latent Watermark [21] (spatial)</td></tr><tr><td>Pixel</td><td>Classical DCT/DWT; DwtDctSvd [5] (freq.)</td><td></td><td>Stable Signature [7] (spatial, VAE-FT); RivaGAN [28] (spatial, attention); HiDDeN [31] (spatial)</td></tr></table>

## 2 A Two-Axis Taxonomy for Generative-Image Watermarking

Existing literature often conflates embedding mechanisms with target representation spaces, blending labels such as learning-based, initial-noise-based, and pixel-domain onto a single descriptive axis. Such categorization obscures whether two methods share an algorithmic pipeline or simply target the same underlying tensor. To resolve this ambiguity, we establish a two-dimensional taxonomy that disentangles the insertion mechanism (how the signal is embedded) from the insertion representation space (where the signal resides). This orthogonal coordinate system, summarized in Table 1, provides a systematic framework to map the design space and expose the technical trade-ofs inherent in generative image watermarking.

## 2.1 Insertion Mechanism

The mechanism axis characterizes the operational procedure used to embed the watermark, categorized into three distinct paradigms:

– Training-free: Inserts signals via fixed rules or closed-form mathematical transformations (e.g., Fourier frequency modulation) without requiring parameter updates or iterative optimization during embedding.

– Optimization-based: Leaves the generative model parameters frozen while solving an iterative, backpropagation-driven optimization over latent vectors or additive latent perturbations to satisfy a target watermark signal.

– Learning-based: Trains or fine-tunes model parameters (e.g., the VAE decoder or difusion backbone weights) or introduces auxiliary neural network modules to implicitly embed and extract watermark payloads.

Note that these categories focus on the dominant embedding procedure; for instance, a training-free embedding method may still rely on inversion routines for watermark extraction.

## 2.2 Insertion Representation Space

The representation axis identifies the primary computational stage targeted by the embedding design along the generative pipeline: initial Gaussian noise $z _ { T } .$ intermediate denoising states $z _ { t } ,$ clean VAE latents $z _ { 0 } .$ , or output pixels. Where applicable, methods are further subdivided into frequency or spatial domains. Rather than rigid partitions, these representations span a continuous computational spectrum along the generative process:

– Initial Noise $\left( z _ { T } \right)$ : Targets the starting random latent seed, allowing the embedded signal to propagate through the entire denoising trajectory.

– Intermediate States (z<sub>t</sub>): Intervenes directly within the iterative denoising process (e.g., ROBIN [14], which applies adversarial optimization over intermediate time steps, $t \in [ 2 0 0 , 3 0 0 ] \rangle$ ).

– VAE Latent Space (z<sub>0</sub>): Modulates the denoised latent vector $z _ { 0 }$ generated after difusion, prior to or during VAE decoding.

– Pixel Domain: Embeds signals directly into the finalized RGB image.

## 2.3 Assignment Principle: Primary Signal-Bearing Representation

The key principle in categorizing a method is not to confuse the representations traversed along the pipeline with the space where the signal is actually embedded and extracted. For example, Stable Signature [7] processes VAE latents $z _ { 0 }$ during generation, but $z _ { \mathrm { 0 } }$ itself is not constrained to hold a watermark pattern. Instead, it fine-tunes the VAE decoder so that the signal is embedded and extracted in the pixel domain, placing it under Pixel / Spatial Domain / Learning-based. Conversely, ZoDiac [29] operates post-hoc on generated pixels and $z _ { \mathrm { 0 } } ,$ yet its optimization explicitly enforces a Tree-Ring-like Fourier pattern in the inverted initial noise $z _ { T }$ . Thus, despite its post-hoc pipeline, it is assigned to Initial Noise (z<sub>T</sub>) / Frequency Domain / Optimization-based.

Furthermore, while spatial versus frequency domain boundaries are sharp for rule-based or explicit transform approaches, they become subtle in learningbased schemes. For instance, the attention-based encoder in RivaGAN [28] embeds watermark payloads into learned pixel distributions, which difers conceptually from classic spatial noise addition. Rather than reflecting a flaw in our taxonomy, this subtle distinction illustrates how the insertion mechanism and representation space inherently interact.

![](images/50a3f0a707c572aebab8089f4d2edacc73da0949aeeb416b268126723903187a.jpg)  
Fig. 1: Representative generation-time z<sub>T</sub>-Fourier watermarking pipeline used in the case studies: a keyed Fourier pattern is inserted into the initial noise, propagated through difusion generation, recovered from an optionally transformed query via DDIM inversion, and compared with reference keys for verification or closed-set identification. Reproduced from [18], which is itself adapted from [15]. The diagram does not represent post-hoc optimization, spatial-z<sub>T</sub>, z<sub>0</sub>, or pixel-domain methods, and the optional transformation is illustrative rather than a robustness claim.

## 2.4 Matrix Coverage and Open Gaps

As shown in Table 1, mapping existing methods onto this matrix highlights clear technical trends and open gaps. First, direct optimization over the initial noise z is uncommon due to the high computational cost of backpropagating gradients through the entire generation trajectory; instead, existing methods resort to training-free rules or intermediate-stage optimization. Second, targeting intermediate representations (z<sub>t</sub>) via training-free rules or parameter learning remains underexplored. Finally, direct pixel-level optimization is vulnerable to generative re-synthesis [9], leading optimization-based methods to focus on latent representations.

## 3 Representative z<sub>T</sub>-Fourier Watermarking: Setup and Detection Tasks

Among the representation spaces in Tab. 1, Secs. 3 to 5 adopt the z<sub>T</sub>-Fourier family as a concrete case study to formalize semantic watermarking, examine frequency integrity, and analyze survival under generative editing. Evaluating this family requires definitions of the embedding and extraction pipelines as well as two detection tasks: verification (confirming watermark presence) and identification (attributing content to an enrolled user or model key). Below, we outline a representative pipeline; readout metrics and decision thresholds vary across specific methods. Note that both tasks are statistical decisions made under a fixed key and threshold, so neither constitutes proof of ownership on its own.

## 3.1 A Representative z -Fourier Pattern Pipeline

The overall pipeline includes latent Fourier embedding, image generation, inversionbased extraction, and distance-based detection, as summarized in Fig. 1. Formally, a text-to-image difusion model G maps an initial Gaussian noise seed $z _ { T } \sim \mathcal { N } ( 0 , I )$ and a text prompt $p _ { s }$ to a synthesized image $x = G ( z _ { T } , p _ { s } )$ . In z<sub>T</sub>-Fourier semantic watermarking, a watermark pattern w is embedded into a specified key region within the frequency domain $\mathcal { F } ( z _ { T } )$ , yielding a watermarked seed $z _ { T } ^ { \mathrm { w m } }$ and the corresponding watermarked image $x _ { \mathrm { w m } } = G ( z _ { T } ^ { \mathrm { w m } } , p _ { s } )$

During detection, given a query image $x _ { \mathrm { q u e r y } }$ that may have undergone downstream editing, an ODE-based DDIM inversion process reconstructs the inverted latent seed $\hat { z } _ { T } .$ . Applying the Fourier transform to $\hat { z } _ { T }$ enables the readout of the extracted key pattern wˆ from the designated key region.

## 3.2 Detection Tasks: Verification and Identification

Verification. Verification determines whether a specified watermark is present. In a standard distance-based formulation, the extracted pattern wˆ is compared against a reference key $w ,$ typically $L _ { 1 }$ distance in the Fourier domain:

$$
D _ { \mathrm { v e r i f . } } ( x _ { \mathrm { q u e r y } } ; w ) = \mathbf { 1 } [ d ( \hat { w } , w ) < \tau ] ,\tag{1}
$$

where τ represents a predefined decision threshold calibrated on unwatermarked (negative) queries. Performance is evaluated using the True Positive Rate at a 1% False Positive Rate (TPR@1%FPR).

Closed-Set Key Identification. Identification expands verification into a multiclass attribution task. It identifies which key from an enrolled pool $\{ w _ { 1 } , \ldots , w _ { N } \}$ generated the query image under the assumption that a watermark exists. Under the closed-set $z _ { T }$ pattern protocol, the predicted key index <sup>ˆ</sup>i is identified by finding the nearest reference key:

$$
\hat { i } = \arg \operatorname* { m i n } _ { i \in \{ 1 , . . . , N \} } d ( \hat { w } , w _ { i } ) .\tag{2}
$$

Top-1 accuracy serves as the primary evaluation metric for the task. Because identification must distinguish the correct key among N candidates rather than perform a simple presence test, it demands higher signal preservation under severe image distortions. In contrast to this pattern-matching formulation, conventional payload-based methods perform identification by applying source-specific thresholded decoding rules to recover the embedded message.

Table 2: Detection over the harmonized attack subset and generative quality under the PhaseMark configuration. Except for HSTR, values follow [16]; HSTR detection is recomputed from [15] without brightness and noise, and its ∆FID-1k is our matchedcondition measurement. “Post” denotes post-hoc applicability, and “Payload/IDs” reports payload bits and enrolled-pool size. Verification is TPR@1%FPR. Identification follows each source-native rule: nearest-pattern accuracy for zero-bit methods or thresholded bit-decoding TPR@1%FPR for payload methods. Lower ∆FID-1k is better. Adapted from [18].
<table><tr><td>Method</td><td>Post Payload</td><td>IDs</td><td>Verif.</td><td>Ident.</td><td>∆FID-1k</td></tr><tr><td>DwtDctSvd [5]</td><td>√</td><td> $3 2 / 1 0 ^ { 6 }$ </td><td>0.494</td><td>0.206</td><td>-0.389</td></tr><tr><td>RivaGAN [28]]</td><td>√</td><td> $3 2 / { 1 0 } ^ { 6 }$ </td><td>0.692</td><td>0.484</td><td>-0.785</td></tr><tr><td>Stable Signature [7]</td><td>X</td><td> $4 8 / 1 0 ^ { 6 }$ </td><td>0.819</td><td>0.489</td><td>-0.241</td></tr><tr><td>Tree-Ring [25]</td><td>×</td><td>zero 2048</td><td>0.699</td><td>0.125</td><td>+1.184</td></tr><tr><td>HSTR [15]</td><td>×</td><td>zero 2048</td><td>0.993</td><td>0.946</td><td>-0.099</td></tr><tr><td>RingID [4]</td><td>X</td><td>zero 2048</td><td>0.999</td><td>0.977</td><td>+2.054</td></tr><tr><td>HSQR [15]</td><td>×</td><td>zero 8192</td><td>0.999</td><td>0.995</td><td>-0.721</td></tr><tr><td>G.Shading [27]</td><td>X</td><td> $2 5 6 / 1 0 ^ { 6 }$ </td><td>1.000</td><td>0.999</td><td>-0.317</td></tr><tr><td>ZoDiac [29]</td><td>√</td><td>zero / 64</td><td>0.982</td><td>0.031</td><td>+0.091</td></tr><tr><td>PhaseMark-PCQ [16]</td><td>√</td><td> $1 2 8 / 1 0 ^ { 6 }$ </td><td>0.970</td><td>0.816</td><td>-0.517</td></tr><tr><td>PhaseMark-APM [16]</td><td>√</td><td> $1 2 8 / 1 0 ^ { 6 }$ </td><td>0.999</td><td>0.988</td><td>+0.042</td></tr></table>

## 4 Case Study I: Frequency Integrity, Detection, and Image Quality

Early z<sub>T</sub> semantic watermarking schemes [4, 25] embed watermark signals by modifying specific frequency components of the initial noise vector. However, such spectral modifications often violate Hermitian symmetry in the Fourier domain. As a result, the imaginary components produced after the inverse Fourier transform must be discarded to enforce a real-valued spatial latent vector. This truncation disrupts the frequency integrity of the seed and shifts the latent distribution even further from the standard Gaussian prior. This compounded Out-of-Distribution (OOD) shift degrades both watermark detectability and the visual quality of generated images.

## 4.1 Associations with Detection and Image Quality

To resolve this limitation, HSTR and HSQR [15], which we introduced in prior work, enforce Hermitian symmetry during frequency-domain embedding. This symmetry condition preserves seed integrity by ensuring lossless frequency reconstruction.

For the shared attack subset in Tab. 2 [15, 16], HSTR reaches verification of 0.993 and identification of 0.946. Tree-Ring reaches 0.699 and 0.125, respectively. HSTR combines Hermitian symmetry with insertion in the center region, so this comparison does not isolate the efect of symmetry.

Table 3: Closed-set identification accuracy under random-crop attacks at four crop scales; smaller scales indicate more aggressive cropping [15]. Adapted from [18].
<table><tr><td>Method</td><td>0.6</td><td>0.5</td><td>0.4</td><td>0.3</td></tr><tr><td>RingID [4]</td><td>0.971</td><td>0.919</td><td>0.774</td><td>0.559</td></tr><tr><td>HSTR [15]</td><td>1.000</td><td>0.999</td><td>0.992</td><td>0.903</td></tr><tr><td>HSQR [15]</td><td>1.000</td><td>1.000</td><td>0.999</td><td>0.999</td></tr></table>

RingID and HSQR both use structured binary patterns and reach a verification TPR at 1% FPR of about 0.999. Their image quality difers: HSQR records $\Delta \mathrm { F I D - 1 k } = - 0 . 7 2 1$ , while RingID records +2.054. This contrast is consistent with a relation between frequency integrity and image quality when detection is already high.

## 4.2 Identification under Spatial Cropping

Beyond frequency integrity, resilience to spatial distortions such as cropping is essential for practical deployment. Conventional schemes that apply the Fourier transform across the entire spatial latent vector often induce severe information loss under image cropping. To address this, HSTR and HSQR adopt a centeraware insertion strategy, which embeds watermark signals into the central region of the latent vector.

As reported in Tab. 3, HSTR and HSQR maintain high identification accuracy even under aggressive cropping ratios, whereas global-embedding methods like RingID sufer substantial performance drops. These results imply that localized insertion with frequency integrity helps maintain robustness against spatial cropping attacks.

## 5 Case Study II: Persistence under Generative Editing

As noted in Section 1, semantic watermarks show promising robustness against generative regeneration attacks [15]. Inversion-based editing such as Prompt-to-Prompt (P2P) [10] remains anchored to the trajectory linking the source image to the initial seed $z _ { T } .$ . Similarly, difusion regeneration [30] can be approximated as traversing this trajectory, since it adds forward noise to the source image to reach an intermediate latent state $z _ { t }$ before denoising. Because these operations rely on the original latent trajectory, they are categorized as seed-linked. Consequently, semantic watermarks may remain detectable when their structure is preserved in these recovered latent states. However, recent findings show that watermarks also survive seed-independent editing protocols such as InfEdit [26], which synthesize images from newly sampled initial noise while retaining structural guidance. This survival challenges purely seed-linked explanations and leaves the precise watermark carrier unresolved.

Table 4: Verification TPR@1%FPR after Prompt-to-Prompt (P2P) and InfEdit editing [17]. Watermarked source images are generated with SD2.1, whereas InfEdit uses SD1.5, yielding a cross-model editing condition. This editing-specific protocol difers from the harmonized general-attack protocol in Tab. 2, so absolute values are not directly comparable across the two tables. Category labels are study-specific descriptors rather than classes in Tab. 1. Reproduced from [18].
<table><tr><td>Category</td><td>Method</td><td>P2P</td><td>InfEdit</td></tr><tr><td>Spectral Modification</td><td>Tree-Ring [25]</td><td>0.25</td><td>0.32</td></tr><tr><td>Optimization-based</td><td>ZoDiac [29]</td><td>0.75</td><td>0.86</td></tr><tr><td>Frequency-aligned</td><td>HSQR [15]</td><td>0.99</td><td>1.00</td></tr><tr><td>Cryptographic</td><td>G.Shading [27]</td><td>1.00</td><td>0.98</td></tr></table>

## 5.1 Observed Persistence across Editing Protocols

Table 4 shows that HSQR and Gaussian Shading retain TPRs of at least 0.98 under both editors, whereas Tree-Ring reaches 0.25 under P2P and 0.32 under InfEdit, as reported in our prior work [17].

Because P2P and InfEdit difer in algorithmic design beyond seed dependence, their numerical gap does not isolate the efect of replacing the original seed. In addition, detailed evaluations reveal that Tree-Ring already has low detectability on unedited source images, and this performance remains largely unchanged after editing. Since Tree-Ring directly modifies initial latent frequencies without preserving spectral integrity, its embedded pattern forms an out-ofdistribution (OOD) signal that fails to align with the natural generative manifold. This structural limitation hinders reliable signal recovery even prior to editing, indicating that its low post-edit performance reflects poor initial detectability rather than edit-induced signal degradation.

If watermark recovery depended solely on reproducing the exact original seed, survival after editing initialized from newly sampled random noise would be impossible. However, because InfEdit remains source-conditioned, the high resilience of HSQR and Gaussian Shading suggests that well-aligned watermarks are internalized into the image feature space, including semantic structure, texture, or local statistics, and propagate reliably through the editing pipeline even without seed conservation.

## 5.2 The Semantic Imprinting Hypothesis

The results support the hypothesis introduced in that work [17].

Semantic Imprinting Hypothesis (SIH). Under a fixed, sourceconditioned editing protocol, a watermark remains detectable even after seed-independent editing if it is reliably recoverable from the unedited source image and its signal is carried within features preserved by the editor.

![](images/a38a046889623a53b0c5acaf7718a98e4d2085c5dfcd80a14fcaf085a77997f9.jpg)  
Fig. 2: Schematic outcomes used to illustrate SIH. The upper example has low preedit recoverability and is not detected after editing, while the lower example has high pre-edit recoverability and is detected after editing. The figure does not identify the watermark carrier or show that recoverability before editing determines the detection result. Adapted from [18]; the underlying study is [17].

Figure 2 contrasts the failure and survival cases described by SIH according to pre-edit watermark recoverability. The current evaluation does not directly test the proposed feature carrier because it covers one benchmark, one comparison between Stable Difusion versions, and four methods. A stronger test should compare the same images before and after editing and report pre-edit TPR, post-edit TPR, and conditional retention $\operatorname* { P r } [ D ( \operatorname { E d i t } ( x ) ) = 1 \mid D ( x ) = 1 ]$ at a fixed threshold. It should control for editor type, edit strength, and fidelity. The test should also define a feature measure or a direct intervention. If recoverability before editing does not predict conditional retention under these controls, the hypothesis would be weakened.

## 6 Case Study III: Single-Shot VAE-Latent Phase Modulation

Watermarks in $z _ { T }$ can be robust, but detection requires DDIM inversion and several seconds per image. The post-hoc $z _ { T }$ method ZoDiac [29] shares this detection cost and also requires several minutes of optimization for embedding. PhaseMark [16], which we proposed in prior work, avoids DDIM inversion and embedding optimization. It modifies the phase of an intermediate frequency band in $z _ { \mathrm { 0 } }$ in a single step.

## 6.1 Latency and Regeneration Robustness

Table 5 shows that PhaseMark requires less than a second for both embedding and detection while maintaining an average regeneration verification rate (0.998)

Table 5: Embedding procedure or per-image cost, detection latency, and mean verification TPR@1%FPR across VAE-B, VAE-C, and difusion regeneration [16]. “Sampling” denotes generation-integrated insertion; “opt.” and “inv.” denote per-image optimization and DDIM inversion. Reproduced from [18].
<table><tr><td>Method</td><td>Space</td><td>Embedding</td><td>Detection</td><td>Regen. avg.</td></tr><tr><td>ZoDiac [29]</td><td>zT</td><td>~7.3 min (opt.)</td><td> $4 . 1 1 \mathrm { ~ s ~ } ( \mathrm { i n v . } )$ </td><td>0.958</td></tr><tr><td>G.Shading [27]</td><td>zT</td><td>Sampling</td><td> $4 . 1 1 \mathrm { ~ s ~ } ( \mathrm { i n v . } )$ </td><td>0.999</td></tr><tr><td>HSQR [15]</td><td>zT</td><td>Sampling</td><td> $4 . 1 1 \mathrm { ~ s ~ } ( \mathrm { i n v . } )$ </td><td>0.997</td></tr><tr><td>PhaseMark-APM [16]</td><td>z0</td><td>0.14 s (one-shot)</td><td>0.05 s</td><td>0.998</td></tr></table>

Table 6: PhaseMark variants: mean verification and identification TPR@1%FPR over clean data and nine attacks, with image-quality metrics [16]. Higher PSNR and lower LPIPS are better. Reproduced from [18].
<table><tr><td>Modulation</td><td>Strategy Type</td><td>Verif. (T@1%F)</td><td>Ident. (T@1%F)</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>APM (Robust)</td><td>Hard / Absolute</td><td>0.999</td><td>0.988</td><td>31.715</td><td>0.129</td></tr><tr><td>IPS</td><td>Hard / Relative</td><td>0.998</td><td>0.980</td><td>32.437</td><td>0.090</td></tr><tr><td>SPS</td><td>Soft / Relative</td><td>0.996</td><td>0.960</td><td>32.952</td><td>0.068</td></tr><tr><td>PCQ (Quality)</td><td>Soft / Absolute</td><td>0.970</td><td>0.816</td><td>34.156</td><td>0.037</td></tr></table>

close to that of z -based semantic watermarks. This high survival rate demonstrates that regeneration robustness does not require initial noise seeds and is achievable through VAE latent frequency modulation.

## 6.2 Modulation Strategies and Robustness–Quality Trade-ofs

Table 6 shows a robustness–quality trade-of across the evaluated phase modulation strategies. Compared to Absolute Phase Modulation (APM), Phase Constellation Quantization (PCQ) yields lower detection scores but higher PSNR and lower LPIPS values. APM applies a stronger modulation to maximize signal detection, whereas PCQ preserves higher visual fidelity. This trade-of provides flexibility to choose a strategy based on application requirements.

## 7 Threat Surfaces: Content Attacks and Model/Pipeline Adaptation

The analyses in Secs. 4 to 6 evaluate attacks on watermarked images under a fixed embedding and detection pipeline. They do not cover changes to a released model or pipeline. We group threats into five families: signal processing, cropping, generative editing, regeneration, and model/pipeline adaptation. The first four transform existing images, while the last changes the system that produces or embeds future watermarks. We study the last family through parameter tuning. The families may overlap.

Table 7: Operational threat families and evidence scope. Content-level attacks transform an existing watermarked image under a fixed embedding rule, key, and detector; model/pipeline adaptation changes future watermark behavior and is supported here only by method-specific reports. Adapted from [18].
<table><tr><td>Attack family</td><td>Modified object</td><td>Evidence in this paper</td></tr><tr><td colspan="3">Content-level attacks Fixed embedding rule, key, and detector</td></tr><tr><td>Signal processing</td><td>Samples of an existing image</td><td>General attacks [15,16]</td></tr><tr><td>Cropping</td><td>Spatial support of an existing image</td><td>Crop analysis [15]</td></tr><tr><td></td><td>Generative editing Source-conditioned re-synthesis</td><td>P2P; cross-model InfEdit [17]</td></tr><tr><td>Regeneration</td><td>Re-synthesis by VAE or LDM</td><td>Regeneration study [16,22,30]</td></tr><tr><td colspan="3">Model/pipeline-level adaptation Future watermark behavior</td></tr><tr><td>Adaptation</td><td>Generator, adapter, conditioning module, or recovery pipeline</td><td>Method-specific reports [6,24]</td></tr></table>

## 7.1 Attack Families and Applicability Boundary

In the first four families in Tab. 7, the embedding rule, key, and detector remain fixed, even if editing or regeneration uses another generative model. Parameter tuning changes the released generator, conditioning module, or integrated embedding component. Its future outputs are tested with the original key and detector. This threat applies when watermarking is integrated into the released system and remains active after tuning. If embedding is an optional external step, an operator can omit it without changing the model, so adaptation robustness does not cover that bypass.

Attack labels alone do not determine whether pixels, z<sub>0</sub>, or z<sub>T</sub> will fail. Our representation taxonomy (Tab. 1) classifies where a watermark is embedded, whereas its actual vulnerability depends on the protected object, pipeline integration, attacker access, and success criteria. W-Bench evaluates watermark robustness under image regeneration and global and local generative editing [20]. Evidence for model/pipeline adaptation remains limited to individual methods. Accordingly, we focus on parameter tuning.

## 7.2 Evaluation Protocols and Conjectures for Parameter Tuning

Full fine-tuning, LoRA [11], DreamBooth [23], and Textual Inversion [8] cannot be placed on a universal severity scale based only on the number of parameters. Full fine-tuning may update broad portions of a generator; LoRA introduces lowrank adapter updates; DreamBooth is a personalization setting where trainable components vary across implementations; and Textual Inversion learns a conditioning embedding while leaving the base generator fixed. Any fair comparison must state the exact trainable tensors, objective, data, optimization budget, active embedding components, and detector model. The primary attack evaluation should keep the key, detector, and threshold calibrated on the original negative distribution fixed. Researchers may report recalibration on adapted outputs separately as a diagnostic for calibration shift.

AquaLoRA [6] couples watermark behavior to U-Net adaptation and reports a trade-of between utility and removal in a white-box setting; however, this does not establish universal persistence under subsequent adaptation. Sleeper-Mark [24] reports persistence under LoRA style adaptation, DreamBooth personalization, and ControlNet conditioning by separating watermark signals from learned semantic concepts. Because these findings apply only to individual methods rather than a matched ranking across Tab. 1, they motivate testable conjectures instead of general conclusions.

Conjecture 1 (component overlap). Assume a watermark integrated into the system, where the operator cannot trivially detach or omit the embedding module. Here, persistence under adaptation relates more directly to whether the modified components overlap the watermark generation path than to the raw number of updated parameters. Testable by: comparing updates that target the watermark generation path with those that do not, using matched prompts and seeds. Retained utility and adaptation objective performance should also be matched, and the status of the embedding module should be reported.

Conjecture 2 (inversion trajectory drift). An inversion dependent $z _ { T }$ watermark can degrade indirectly when adaptation disrupts compatibility between the generation (G) and inversion (I) processes, even though its key is not stored in model weights. In contrast, direct latent level modulation $\left( z _ { 0 } \right)$ exhibits greater resilience against such trajectory distortion because it relies less on the inversion path. Testable by: evaluating $( G _ { \theta } , I _ { \theta } ) , ( G _ { \theta ^ { \prime } } , I _ { \theta } ) , ( G _ { \theta } , I _ { \theta ^ { \prime } } )$ , and $\left( G _ { \theta ^ { \prime } } , I _ { \theta ^ { \prime } } \right)$ with the same key, prompts, and seeds, and reporting latent recovery error alongside verification and identification. The fixed original inverter $I _ { \theta }$ defines the primary attack result. Conditions using ${ \cal I } _ { \theta ^ { \prime } }$ provide diagnostics to separate changes in the forward model from inversion mismatches, rather than representing additional attacker privileges.

Conjecture 3 (adaptation interaction). Persistence depends on the interaction of adaptation type, objective, and strength. Consequently, training steps or the number of parameters alone do not produce a common severity ordering. Testable by: reporting detection, generation quality, and adaptation objective performance across attack strengths at matched retained utility. These evaluations characterize persistence and compatibility loss, not resistance to omission of the embedding step.

## 7.3 Extension to Video

Video watermarking introduces temporal attacks that do not occur in static images. VideoShield [12] extends Gaussian Shading across time with template bits.

It reports spatiotemporal tamper localization and temporal position recovery under frame exchange, insertion, and deletion, but extraction still uses DDIM inversion. VideoMark [13] embeds PRC coded messages frame by frame and uses edit distance for temporal alignment. It addresses inversion error, temporal distortion, and changes in sequence length. Video evaluations should cover insertion, deletion, exchange, and misalignment in addition to the five families above. They should also test inversion drift after temporal adapter tuning.

## 8 Conclusion and Open Challenges

This paper organizes watermarking methods for images generated by difusion models according to insertion mechanism and representation space. It also formalizes verification and closed set identification for a representative Fourier watermark in z . The three case studies use diferent evaluation protocols, so their results do not provide a direct ranking. The results suggest a relation among frequency integrity, detection, and image quality, but the comparison does not separate Hermitian symmetry from other design changes. HSTR and HSQR also show higher identification accuracy than RingID in the crop tests. The editing results motivate the Semantic Imprinting Hypothesis, but the experiments do not identify which image features carry the watermark or test the proposed factors separately. PhaseMark requires less than one second for both embedding and detection and records an average verification rate of 0.998 in the regeneration tests. Its modulation methods show a trade-of between detection and image quality.

Section 7 separates attacks on an existing watermarked image from changes to the model or watermarking pipeline. Current evidence on adaptation comes from individual methods, so the paper does not rank these attacks by strength. The three conjectures concern component overlap, the relation between generation and inversion, and the efects of adaptation type, objective, and strength. Tests should state which components are trained, keep the key, detector, and threshold fixed, and report generation quality and performance on the adaptation task. Video also requires tests for frame insertion, deletion, exchange, and misalignment. Editing studies should test the same images before and after editing and report conditional retention at a fixed threshold. Adaptation studies should compare settings with similar generation quality and adaptation performance. These questions should also be tested with more difusion models, watermark families, and video models. The verification and identification tasks in this paper do not by themselves establish legal ownership.

## Acknowledgements

This research was supported by Samsung Electronics Co., Ltd.

## References

1. Arabi, K., Feuer, B., Witter, R.T., Hegde, C., Cohen, N.: Hidden in the noise: Two-stage robust watermarking for images. In: Proceedings of the 13th International Conference on Learning Representations (ICLR). Singapore (2025), https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 999fcab97007ebef0cda9949550b4a9e-Abstract-Conference.html

2. Barni, M., Bartolini, F., Cappellini, V., Piva, A.: A DCT-domain system for robust image watermarking. Signal Processing 66(3), 357–372 (1998). https://doi.org/ 10.1016/S0165-1684(98)00015-2

3. Bui, T., Agarwal, S., Yu, N., Collomosse, J.: RoSteALS: Robust steganography using autoencoder latent space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 933–942. Vancouver, Canada (2023). https://doi.org/10.1109/CVPRW59228.2023.00100

4. Ci, H., Yang, P., Song, Y., Shou, M.Z.: RingID: Rethinking tree-ring watermarking for enhanced multi-key identification. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 338–354. Milan, Italy (2024). https://doi.org/ 10.1007/978-3-031-73390-1\_20

5. Cox, I.J., Miller, M.L., Bloom, J.A., Fridrich, J., Kalker, T.: Digital Watermarking and Steganography. Morgan Kaufmann Publishers, Burlington, USA, 2nd edn. (2008). https://doi.org/10.1016/B978-0-12-372585-1.X5001-3

6. Feng, W., Zhou, W., He, J., Zhang, J., Wei, T., Li, G., Zhang, T., Zhang, W., Yu, N.: AquaLoRA: Toward white-box protection for customized stable difusion models via watermark LoRA. In: Proceedings of the 41st International Conference on Machine Learning (ICML). vol. 235, pp. 13423–13444. Vienna, Austria (2024), https://proceedings.mlr.press/v235/feng24k.html

7. Fernandez, P., Couairon, G., Jégou, H., Douze, M., Furon, T.: The stable signature: Rooting watermarks in latent difusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 22409–22420. Paris, France (2023). https://doi.org/10.1109/ICCV51070.2023.02053

8. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion. In: Proceedings of the 11th International Conference on Learning Representations (ICLR) (2023), https://openreview.net/forum?id= NAQvF08TcyG

9. Guo, Y., Li, R., Hui, M., Guo, H., Zhang, C., Cai, C., Wan, L., Wang, S.: FreqMark: Invisible image watermarking via frequency based optimization in latent space. In: Advances in Neural Information Processing Systems. vol. 37, pp. 112237–112261 (2024). https://doi.org/10.52202/079017-3564

10. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross-attention control. In: Proceedings of the 11th International Conference on Learning Representations (ICLR). Kigali, Rwanda (2023), https://openreview.net/forum?id=\_CDixzkzeyb

11. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: Proceedings of the 10th International Conference on Learning Representations (ICLR) (2022), https: //openreview.net/forum?id=nZeVKeeFYf9

12. Hu, R., Zhang, J., Li, Y., Li, J., Guo, Q., Qiu, H., Zhang, T.: VideoShield: Regulating difusion-based video generation models via watermarking. In: Proceedings

of the 13th International Conference on Learning Representations (ICLR). Singapore (2025), https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 8227285e32f70e07fa3a247f3a48006d-Abstract-Conference.html

13. Hu, X., Li, H., Li, J., Huang, Y., Liu, S., Zheng, Q., Chen, J., Liu, A.: VideoMark: A distortion-free robust watermarking framework for video difusion models. arXiv preprint arXiv:2504.16359 (2025). https://doi.org/10.48550/arXiv.2504.16359

14. Huang, H., Wu, Y., Wang, Q.: ROBIN: Robust and invisible watermarks for diffusion models with adversarial optimization. In: Advances in Neural Information Processing Systems. vol. 37, pp. 3937–3963 (2024). https://doi.org/10.52202/ 079017-0129

15. Lee, S.J., Cho, N.I.: Semantic watermarking reinvented: Enhancing robustness and generation quality with fourier integrity. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 18759–18769. Honolulu, USA (2025). https://doi.org/10.1109/ICCV51701.2025.01743

16. Lee, S.J., Cho, N.I.: PhaseMark: A post-hoc, optimization-free watermarking of AIgenerated images in the latent frequency domain. In: Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 13472–13476. Barcelona, Spain (2026). https://doi.org/10.1109/ICASSP55912. 2026.11463051

17. Lee, S.J., Cho, N.I.: The semantic imprinting hypothesis: How semantic watermarks survive prompt-based editing. In: ICLR 2026 Workshop on Principled Design for Trustworthy AI: Interpretability, Robustness, and Safety Across Modalities (2026), https://openreview.net/forum?id=hNbkvSjEAD

18. Lee, S.J., Cho, N.I.: A taxonomy and trade-of analysis of generative media watermarking across latent and pixel domains. Journal of Broadcast Engineering 31(4), 687–699 (2026), in Korean

19. Lei, L., Gai, K., Yu, J., Zhu, L.: DifuseTrace: A transparent and flexible watermarking scheme for latent difusion model. arXiv preprint arXiv:2405.02696 (2024). https://doi.org/10.48550/arXiv.2405.02696

20. Lu, S., Zhou, Z., Lu, J., Zhu, Y., Kong, A.: Robust watermarking using generative priors against image editing: From benchmarking to advances. In: Proceedings of the 13th International Conference on Learning Representations (ICLR) (2025), https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ d077bc9ea82a2998ca6b2d0158b5ac6e-Abstract-Conference.html

21. Meng, Z., Peng, B., Dong, J.: Latent watermark: Inject and detect watermarks in latent difusion space. IEEE Transactions on Multimedia 27, 3399–3410 (2025). https://doi.org/10.1109/TMM.2025.3535300

22. Ni, Y., Yang, Z., Niu, Z., Davis, E., Carter, F.: On the informationtheoretic fragility of robust watermarking under difusion editing. arXiv preprint arXiv:2511.10933 (2025). https://doi.org/10.48550/arXiv.2511.10933

23. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dream-Booth: Fine tuning text-to-image difusion models for subject-driven generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22500–22510 (2023). https://doi.org/10.1109/ CVPR52729.2023.02155

24. Wang, Z., Guo, J., Zhu, J., Li, Y., Huang, H., Chen, M., Tu, Z.: SleeperMark: Towards robust watermark against fine-tuning text-to-image difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8213–8224. Nashville, USA (2025). https://doi.org/10. 1109/CVPR52734.2025.00769

25. Wen, Y., Kirchenbauer, J., Geiping, J., Goldstein, T.: Tree-rings watermarks: Invisible fingerprints for difusion images. In: Advances in Neural Information Processing Systems. vol. 36, pp. 58047–58063 (2023). https://doi.org/10.52202/075280- 2529

26. Xu, S., Huang, Y., Pan, J., Ma, Z., Chai, J.: Inversion-free image editing with language-guided difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 9454–9461. Seattle, USA (2024). https://doi.org/10.1109/CVPR52733.2024.00903

27. Yang, Z., Zeng, K., Chen, K., Fang, H., Zhang, W., Yu, N.: Gaussian shading: Provable performance-lossless image watermarking for difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12162–12171. Seattle, USA (2024). https://doi.org/10.1109/ CVPR52733.2024.01156

28. Zhang, K.A., Xu, L., Cuesta-Infante, A., Veeramachaneni, K.: Robust invisible video watermarking with attention. arXiv preprint arXiv:1909.01285 (2019). https://doi.org/10.48550/arXiv.1909.01285

29. Zhang, L., Liu, X., Viros Martin, A., Bearfield, C.X., Brun, Y., Guan, H.: Attackresilient image watermarking using stable difusion. In: Advances in Neural Information Processing Systems. vol. 37, pp. 38480–38507 (2024). https://doi.org/ 10.52202/079017-1215

30. Zhao, X., Zhang, K., Su, Z., Vasan, S., Grishchenko, I., Kruegel, C., Vigna, G., Wang, Y.X., Li, L.: Invisible image watermarks are provably removable using generative AI. In: Advances in Neural Information Processing Systems. vol. 37, pp. 8643–8672 (2024). https://doi.org/10.52202/079017-0276

31. Zhu, J., Kaplan, R., Johnson, J., Li, F.F.: HiDDeN: Hiding data with deep networks. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 682–697 (2018). https://doi.org/10.1007/978-3-030-01267-0\_40