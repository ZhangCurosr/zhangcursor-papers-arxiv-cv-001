# Haste Makes Waste? Towards Safe Few-Step Text-to-Image Models

Lior Ziv and Mahmood Sharif

Tel Aviv University

Email: lz1@mail.tau.ac.il, mahmoods@tauex.tau.ac.il

Abstract—While few-step distilled text-to-image models have gained traction due to their fast generation times, the speed-up implications on their unsafe generation rates remain unknown. We uncover an inherent tension between speed and safety: few-step models are often more unsafe than their teachers, and reducing the number of inference steps consistently raises unsafe generation rates. We further find that acceleration makes unsafe concepts harder to erase in few-step models, with existing concept-erasure methods markedly less effective than in teachers. We offer a mechanistic diagnosis for why erasure methods fail on few-step models, finding that the fragility of few-step models’ generation process and the way generation guidance is usually distilled into those models may hinder established concept-erasure methods. To address the speed-safety trade-off, we propose DICE (Distillation-Integrated Concept Erasure), a post-distillation erasure method tailored for few-step models. DICE aligns both its latent construction and its generation-preservation loss with the distillation method that produced the model, keeping erasure faithful to few-step models. DICE demonstrates strong results in our evaluation. For example, on FLUX Schnell, DICE reduces unsafe-generation rate to 0.046 (vs. ≥0.161 for prior erasure method we test), while mostly preserving benign generation quality.<sup>1</sup>

WARNING: THIS PAPER CONTAINS EXPLICIT TEXT AND (CENSORED) IMAGES THAT READERS MAY BE UNCOM-FORTABLE WITH.

## 1. Introduction

Text-to-image (T2I) diffusion models [1], [2], [3], [4] are increasingly deployed in user-facing products, making the safety of the content they generate a first-order deployment concern. A dominant deployment trend over the past two years has been a move from many-step samplers to fast, distilled few-step variants—e.g., SD3.5 Turbo [2], [5], Sana Sprint [6], [7], FLUX Schnell [4], and SCFM-style distilled models [8]—which generate images in one to four steps and are routinely shipped as drop-in replacements for their slower teachers. However, whether this acceleration preserves the safety properties of the underlying model has not been systematically audited. Prior work on T2I safety benchmarking [9], [10], [11] and on concept erasure (e.g., ESD [12], UCE [13], EraseAnything [14], Meta-Unlearning [15]) was developed and evaluated against fullstep teacher models, and two questions of immediate deployment relevance remain open: (i) how does the safety profile of a T2I system change when it is shipped through the acceleration pipeline? and (ii) do existing concept-erasure defenses, applied at the natural insertion points of that pipeline, continue to function once the deployment target is a few-step distilled model? This paper sets to answer these questions.

We find that acceleration is not safety-neutral—fewstep acceleration and reducing T2I models’ inference step count both result in higher rates of unsafe content generation (§3.2–3.3). We then turn to checking whether existing concept-erasure methods originally designed for full-step baseline models are effective on few-step distilled models (§3.4). The results show that applying these to few-step models either directly (post-distillation), prior to distillation, or via transfer from the teacher is significantly less effective than applying them to full-step models.

To better understand why erasure fails in few-step models, we carry a mechanistic study (§4). A simple explanation would be that acceleration and concept erasure directly compete. However, we find that the update directions of acceleration and erasure techniques are mostly orthogonal, suggesting that other issues hinder erasure within fewstep models. Complementing prior work on the geometry and early decision concentration of few-step models [16], [17], our empirical analyses uncover two failure modes that emerge when full-step teachers are distilled into one-to-fourstep students. First, few-step students are more brittle under small changes: relative to their teachers, perturbing either the latent input or the model weights slightly can drastically change the student’s next generation update. Second, the way guidance is baked into the weights of few-step models distorts the concept direction used during erasure: the student direction is rotated away from the teacher’s and inflated in norm, driving erasure updates to a degraded target. Together, these findings suggest that existing erasure methods fail not because safety and speed are inherently incompatible, but because post-hoc edits are applied to models whose few-step generation behavior differs from the full-step models those methods were designed for.

In an attempt to address the gap, we present DICE (Distillation-Integrated Concept Erasure)—to our knowledge, the first concept-erasure method tailored for few-step models. DICE retargets two key components of the standard erasure so they are better fit for the distilled student: (i) the generation-preservation loss keeps the edited model’s output close to the original distilled model’s, rather than the standard flow-matching training target that acceleration methods may deviate from; and (ii) the noisy latent on which the ESD loss is evaluated is sampled along the straight noiseto-image line between Gaussian noise and a clean concept image generated by the frozen student—mirroring the latent construction used in distillation rather than the deployment scheduler grid where erasure direction may be degraded and where the few-step model is sensitive to minor perturbations. We instantiate DICE on FLUX Schnell, SD3.5 Turbo and FLUX SCFM, evaluate it on safety, speed, and quality axes, and run ablations that isolate the two design choices. Across the methods we evaluate, DICE is the only configuration that simultaneously significantly reduces unsafe concept generation (pixel-level detectors and Unsafe@k, with qualitative confirmation) and maintains general image quality on already-distilled rectified-flow models.

We summarize our core contributions as follows:

We show that T2I acceleration is not safety-neutral: a trade-off exists between safety and speed in current open-weights flow-matching models (§3.2–3.3).

We show that current concept-erasure methods do not effectively erase unsafe concepts from few-step models (§3.4); our mechanistic study helps shed light on why these methods may fail (§4).

Motivated by our findings, we present DICE, an erasure method tailored for few-step models that is able to better suppress unsafe concepts while preserving image quality in these models (§5).

## 2. Background

## 2.1. Flow-Matching T2I Models

Recent T2I models increasingly adopt flow-matching or rectified-flow formulations. These models operate in the latent space of a pretrained Variational Autoencoder (VAE). They learn a velocity field that defines a generative path from a noise sample $x _ { 1 }$ to a clean image latent $x _ { 0 } .$ , and generation is performed by numerically integrating this learned velocity field. A solver timestep t indexes a noise level $\sigma _ { t }$ from the model’s noise schedule, with $\sigma _ { t } \approx 1$ near the noisy endpoint and $\sigma _ { t } \approx 0$ near the clean endpoint. For a text condition c and model parameters θ, we write the predicted velocity as $v _ { \theta } ( z _ { t } , c , t )$ . The corresponding noisy latent can be written as the linear interpolation

$$
z _ { t } = ( 1 - \sigma _ { t } ) x _ { 0 } + \sigma _ { t } x _ { 1 } = x _ { 0 } + \sigma _ { t } ( x _ { 1 } - x _ { 0 } )\tag{1}
$$

where $x _ { 1 }$ is Gaussian noise.

The learned velocity field specifies how to move along this path during sampling. Starting from noise $z _ { t = 1 } = x _ { 1 }$ the sampler repeatedly evaluates $v _ { \theta } ( z _ { t } , c , t )$ and takes numerical integration steps toward lower noise levels, producing an estimate of the clean latent at $t ~ = ~ 0$ . During flow-matching training, the model is supervised to predict the reverse denoising velocity of the interpolation in Eq. 1: since $\partial z _ { t } / \partial \sigma _ { t } = x _ { 1 } - x _ { 0 }$ , the target direction for decreasing noise is $x _ { 1 } - x _ { 0 }$ . Thus the denoiser learns the direction that transports noisy latents back toward clean image latents, which are then decoded to images using the VAE decoder.

When the endpoints and timestep are clear from context, we write a noisy latent simply as z, omitting the particular $x _ { 0 } , \ x _ { 1 }$ , and t used to construct it. We use this notation to distinguish two ways of obtaining latents z for analysis or erasure training. When z is synthetic and obtained by running a particular model for one or more solver steps along its own sampling trajectory, we call it on-policy for that model. When z is constructed outside that rollout, by taking a clean image $x _ { 0 }$ and re-noising it to a chosen $\sigma _ { t }$ using Eq. 1, we call it off-policy.

Compared to classical DDPM/DDIM-style [18] diffusion sampling, rectified-flow formulations are designed to support straighter trajectories and fewer integration steps. However, these models still commonly require many solver steps at inference. Two prominent open examples are FLUX Dev and SD3.5 Large [2], [19]. SD3.5 Large uses standard classifier-free guidance (CFG) [20]: the model is evaluated with both the text condition c and the null condition ∅, and the guided velocity is formed as

$$
v _ { \mathrm { C F G } } ( z , c , t ; \omega ) = v _ { \theta } ( z , \emptyset , t ) + \omega \left( v _ { \theta } ( z , c , t ) - v _ { \theta } ( z , \emptyset , t ) \right) ,
$$

where ω is the guidance scale. This keeps the guidance operation external to the model weights and preserves separate conditional and unconditional predictions at inference.

FLUX Dev handles guidance differently: it conditions the model on a guidance scalar and approximates guided generation in a single forward pass:

$$
v _ { \mathrm { C F G } } ( z , c , t ; \omega ) = v _ { \theta } ( z , c , t , \omega )
$$

We refer to this as guidance distillation, and distinguish it from few-step distillation. Guidance distillation changes how CFG is represented, while few-step distillation changes the number and size of solver steps the model is trained to support. FLUX Dev is guidance-distilled from the closedweights FLUX Pro, but it is not a few-step model, and is sampled with many inference steps, 50 by default.

## 2.2. Few-Step Models

To accelerate generation, few-step distillation trains a student model to generate high-quality images in only 1–4 inference steps. In many modern T2I settings, the student shares the teacher architecture and is initialized from the teacher, but its training objective differs from the original flow-matching objective: instead of learning only the local velocity target used to train the teacher, the student is optimized to reproduce the teacher’s behavior under much larger sampling steps.

Distillation methods We study several forms of few-step distillation. Adversarial Diffusion Distillation (ADD) combines teacher-based distillation with an adversarial objective, encouraging the student to produce realistic samples in the aggressive 1–4 step regime. Latent Adversarial Diffusion Distillation (LADD) performs adversarial distillation in the latent space of a pretrained latent diffusion model, using synthetic teacher-generated data. In T2I applications, ADD/LADD students are trained to match a guided teacher and are then sampled without external CFG. We refer to this as guidance “bake-in,” as the CFG behavior is baked into the student’s weights. As a result, a particular teacherguidance regime is effectively absorbed into the student. Shortcut Consistency Flow Matching (SCFM), by contrast, is a consistency-based method for flow-matching models. It builds shortcut targets by composing two smaller velocity steps over a longer interval. The objective contains both a teacher-guided term, where the two-step target is computed using the frozen pretrained flow model, and a self-distillation term, where the target is computed using an Exponential Moving Average (EMA) or online student. Thus, the frozen teacher remains a fixed anchor throughout training, while the EMA term encourages internal shortcut consistency. In this sense, SCFM is a lighter, progressive-like form of acceleration: the original SCFM work targets moderate fewstep sampling in the 3–8 step regime with substantially less training time than ADD/LADD-style distillation, while staying closer to the teacher’s velocity-space objective.

This variety is important for our work. SD3.5 Turbo is described as an adversarially distilled variant of SD3.5 Large, trained using ADD [2]. FLUX Schnell is publicly described as a few-step model trained using LADD, but the specific teacher model and full distillation recipe are not disclosed [19], [21]. Thus, unlike SD3.5 Turbo, FLUX Schnell does not provide a known teacher-student pair for controlled analysis. We therefore apply SCFM to FLUX Dev to obtain FLUX SCFM, a controlled FLUX-based teacher-student pair with a known teacher. Overall, we evaluate released few-step checkpoints together with FLUX SCFM, covering adversarial, latent-adversarial, and consistency/shortcut distillation.

## 2.3. Concept Erasure

Concept erasure aims to remove an undesirable concept from a pretrained T2I model while preserving its ability to generate unrelated content. Erased concepts may correspond to unsafe content, copyrighted styles, memorized identities, or other concepts that a model provider wants to suppress.

A prominent family of erasure methods fine-tunes the model by optimizing gradient-based objectives. A central building block in this family is the ESD loss, introduced by Erased Stable Diffusion (ESD) and reused by many later fine-tuning-based erasure methods [11], [12], [14], [15], [22], [23]. $\mathcal { L } _ { \mathrm { E S D } }$ suppresses a target concept by modifying the model’s conditional prediction relative to its unconditional prediction. In velocity-prediction notation, it can be written as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { E S D } } = \Big \| v _ { \boldsymbol { \theta } } ( z , c , t ) - \Big [ v _ { \boldsymbol { \theta } ^ { \star } } ( z , \infty , t ) } \\ & { \qquad - \left. \eta \big ( v _ { \boldsymbol { \theta } ^ { \star } } ( z , c , t ) - v _ { \boldsymbol { \theta } ^ { \star } } ( z , \infty , t ) \big ) \right] \Big \| _ { 2 } ^ { 2 } } \end{array}\tag{2}
$$

where $v _ { \theta ^ { \star } }$ is the frozen pretrained model, $v _ { \theta }$ is the edited model, c is the erased concept, ∅ is the null condition, and the negative guidance strength $\eta$ controls the strength of the negative signal. Intuitively, the method pushes the edited model away from the concept direction encoded by the difference between conditional and unconditional predictions.

Recent methods for fine-tuning T2I models have adopted Low-Rank Adaptation (LoRA) [24]. Instead of updating the full weight matrix $( \theta _ { \ell } )$ of each layer (ℓ), LoRA keeps the pretrained weights frozen and learns a low-rank residual update,

$$
\begin{array} { r } { \theta _ { \ell } \gets \theta _ { \ell } + \Delta \theta _ { \ell } , \qquad \Delta \theta _ { \ell } = B _ { \ell } A _ { \ell } , } \end{array}\tag{3}
$$

where $A _ { \ell } \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ and $B _ { \ell } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ are trainable lowrank matrices, with $r \ll$ min $( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } )$ . This parameterization has become popular due to its simplicity and effectiveness, as well as its ability to substantially reduce the number of optimized parameters, enabling faster and more memoryefficient training. Since the original model weights remain fixed and only the residual update $( \Delta \theta _ { \ell } )$ is optimized, LoRA can also help preserve the pretrained model and mitigate catastrophic forgetting during fine-tuning.

Not all erasure methods are fine-tuning-based. We also include UCE [13] and SPEED [25] as closed-form editing methods. UCE analytically edits the model’s textconditioning projections in order to modify how specific text concepts are mapped into the image-generation process, without iterative fine-tuning. SPEED refines UCE to enable multiple concept erasure and precise edits specific to target concepts. In older U-Net-based [26] diffusion models these edits target cross-attention projections; in transformer-based models such as FLUX and SD3.5, the corresponding edits apply to the initial text-conditioning projection layers rather than to cross-attention modules.

Most concept-erasure methods were designed and evaluated primarily on SD 1.x-style latent diffusion models with U-Net backbones, DDPM/DDIM-style denoising objectives, and standard CFG sampling. Their assumptions do not necessarily transfer cleanly to modern T2I models such as FLUX and SD3.5, which use Multimodal Diffusion Transformer (MMDiT) [27] backbones and flow-matching velocity prediction rather than DDPM/DDIM-style noise prediction. A recent work, EraseAnything, explicitly targets concept erasure in newer flow-matching T2I models and shows that adapting erasure to these architectures requires additional care [14]. We therefore evaluate both classical erasure methods and EraseAnything.

## 2.4. Threat Model

We consider a defender that modifies a pretrained, potentially distilled T2I model to prevent the generation of prohibited visual content while preserving its benign generation quality and inference speed. In line with prior work (e.g., [12], [13], [14]), we primarily assume a black-box adversary who controls input prompts and seeks to elicit prohibited content. In evaluations of adaptive adversaries (§5.3), we further consider adversaries with white-box access to the text encoder and to the full model weights.

## 3. Empirical Safety Trade-offs

We begin by exploring the safety implications of fewstep acceleration, contrasting few-step students with fullstep teacher models (§3.2), and assessing how reducing the number of generation steps impacts safety (§3.3). Our results show a consistent trade-off between speed and safety: faster generation (i.e., lower number of generation steps) is associated with more unsafe content emitted by the models. We also apply popular and leading erasure methods [12], [13], [14] to few-step models and check their impact on safety, speed, and utility (§3.4). We find that erasure methods demonstrate significantly lower efficacy over all the fewstep models we test. We further test alternative strategies to integrate few-step acceleration and erasures, such as erasing a teacher before distillation (§3.4). Across all strategies, erasure remains consistently weaker in the resulting few-step models than in their full-step counterparts, while introducing additional strategy-specific weaknesses. Before describing our findings, we present the experimental setup (§3.1).

## 3.1. Experimental Setup

Models For our experiments, we focus on rectified flowmatching models, which have emerged as a dominant formulation in recent state-of-the-art T2I systems [27], specifically the popular FLUX and SD3.5 model families. Both have few-step variants produced using ADD/LADD [5], [7]— SD3.5 Turbo which is based on SD3.5 Large [2], and FLUX Schnell which is produced using LADD [4], [7], but the specific teacher model used is not disclosed. Thus, we employ the SCFM distillation method [8] over FLUX Dev to produce FLUX SCFM. We use the SCFM training recipe and parameters recommended by Cai et al. [8]—full details are in App. A.1. By training FLUX SCFM, we effectively create a known teacher-student pair for the FLUX model family. Because SCFM is a distillation method based on the self-consistency loss [28] rather than an adversarial-based loss like ADD or LADD, FLUX SCFM helps us corroborate our findings on a profoundly different distillation approach. Since few-step distillation is a resource-intensive process compared to concept erasure or inference, and since we already have a known open-weights student-teacher pair in the SD3.5 family, we do not apply SCFM training for the SD3.5 family. Another approach for producing a student for FLUX would be applying LADD ourselves, but the LADD process is significantly more resource intensive [8] than SCFM, making it infeasible to run.

We use each model through its official implementation in the diffusers library, together with default components it supplies: the Euler Discrete scheduler and its timestep schedule, the image variational autoencoder (VAE), and the corresponding text tokenizer and encoder. Unless stated otherwise, we use the recommended default inference configuration for each model [2], [4], [8]: CFG scale 3.5 for SD3.5 and FLUX, CFG scale 5 for FLUX SCFM, which lies within its recommended operating range [4.5, 6], 28 inference steps for SD3.5 Large, 50 inference steps for FLUX Dev, and 4 inference steps for all few-step models.

Unsafe prompts and safety checkers Following common practice [9], [12], [29], we assess nudity and violence erasure, with nudity being our main focus. Due to a lack of explicitness we encounter in the widely used I2P [9] dataset, we follow multiple recent papers [30], [31] and use the NSFW-56k [10] dataset for nudity, which we find significantly more explicit. We report results on a randomly selected subset of 1k prompts from the NSFW-56k dataset (hereafter, NSFW-1k). We also report results on the I2P dataset, adopted due to its popularity in the field [12], [14], [23] in App. B.7. To evaluate the safety of generated images in the nudity use-case, we leverage two widely used detectors: NudeNet [32], a CNN-based detector, and SD-Checker [33], a CLIP-based detector shipped with earlier Stable Diffusion releases. We find the two checkers to be mutually consistent, and both have been shown to agree with human annotators [34]. App. B.6 describes the evaluation of violence erasure, yielding results consistent with those of nudity erasure.

Metrics Both detectors yield a per-image binary unsafe label, and we report their mean unsafe detection rate, i.e., the fraction of generated images flagged as unsafe. NudeNet outputs a set of detected body-part boxes each with a confidence score. We use its default detection threshold 0.2 and label an image as unsafe if it contains at least one detection from a fixed set of explicit-nudity classes (App. A.2). SD-Checker directly returns a binary unsafe flag, which we use as-is. Since few-step distilled models may reduce diversity [17], we also report results over new metrics we introduce—the NudeNet@k and SD-Checker@k metrics— we generate k different images using different noise seeds for each prompt, and we categorize the generation instance as unsafe if any of the k images is deemed unsafe by the baseline safety checker (throughout the paper, we use k=5). Intuitively, NudeNet@k and SD-Checker@k better reflect realistic adversary capabilities in practice, as they can probe T2I models multiple times until unsafe content is generated.

To measure the overall benign generation quality of a model, we use the HPSv2 benchmark [35]. We prefer HPSv2 as opposed to the FID score [36] because it is promptconditioned, trained on large-scale human preference annotations, and is arguably more fit to the evaluation of T2I models [37]. That said, due to the widespread use of FID for measuring distributional alignment and CLIP-based imagetext similarity for measuring prompt adherence [38], we also report these metrics on COCO-30k [39] in App. B.8.

Erasure methods We evaluate two fine-tuning-based erasure methods and one closed-form edit-based method, chosen to cover influential representatives of the dominant practical erasure paradigms. ESD [12] is a foundational and widely adopted fine-tuning-based method in the erasure literature. EraseAnything [14] is a recent LoRA-based [24] erasure method tailored for flow-matching models, making it especially relevant for modern models such as FLUX Dev.

TABLE 1. FEW-STEP STUDENT VS. FULL-STEP TEACHER SAFETY COMPARISON. WE REPORT MEAN NUDENET AND SD-CHECKER UNSAFE GENERATION RATES WITH THEIR @5 COUNTERPARTS OVER NSFW-1K. LOWER (BOLDFACED) VALUES INDICATE SAFER MODELS.
<table><tr><td>Family</td><td>Variant</td><td>NudeNet↓</td><td>NudeNet@5 ↓</td><td>SD ↓</td><td>SD@5↓</td></tr><tr><td rowspan="2">SD3.5</td><td>Large (teacher)</td><td>0.2432</td><td>0.5240</td><td>0.5416</td><td>0.8150</td></tr><tr><td>Turbo (student)</td><td>0.3232</td><td>0.5870</td><td>0.6646</td><td>0.8670</td></tr><tr><td rowspan="2">FLUX</td><td>Dev (teacher)</td><td>0.2562</td><td>0.5570</td><td>0.6108</td><td>0.8570</td></tr><tr><td>SCFM (student)</td><td>0.3282</td><td>0.6390</td><td>0.7074</td><td>0.9020</td></tr></table>

![](images/998de646803799b31c428968b863c847e75d2cb32f6ee3b047b5d104ca2b0acf.jpg)  
Figure 1. SD-Checker severity distribution for SD3.5 Large vs. SD3.5 Turbo, with a box-plot summary.

Finally, UCE [13] is widely used due to its simplicity and closed-form update, and we include it as a representative edit-based alternative to fine-tuning-based erasure. Out of the erasure methods we test, ESD and UCE were designed for older U-Net-style models, and none were designed for few-step models. To adapt them to the tested models, we perform a grid search over their negative guidance and learning rate parameters, selecting the configuration that maximally erases unsafe content while best preserving benign generation quality.

## 3.2. Safety Under Few-Step Acceleration

We begin by examining the unsafe generation rates of out-of-the-box few-step and full-step models. Tab. 1 reports the unsafe generation rates of each few-step student alongside its full-step teacher. Across both model families and all four safety metrics, the few-step student is consistently less safe than its teacher. The SD3.5 Turbo student raises the NudeNet rate from 0.2432 to 0.3232 (a 33% relative increase) and the SD-Checker rate from 0.5416 to 0.6646 relative to SD3.5 Large; analogously, FLUX SCFM raises the NudeNet rate from 0.2562 to 0.3282 and the SD-Checker rate from 0.6108 to 0.7074 relative to FLUX Dev. The gap persists under the Unsafe@k metrics, where a prompt is flagged unsafe if any of its k=5 seeds is flagged: this shows that acceleration not only increases the average unsafe rate but also makes at least one unsafe generation more likely under repeated sampling. Taken together, these consistent gaps suggest that few-step distillation can systematically erode the implicit safety behavior inherited from the teacher.

FLUX Schnell is marginally safer than FLUX Dev (e.g. 0.2348 vs. 0.2562 on NudeNet) and may seem to contradict our conclusion at first glance. However, as discussed in §2.2, the teacher of FLUX Schnell is not publicly disclosed and is not necessarily FLUX Dev (itself a guidance-distilled variant of the closed-weights FLUX Pro). This is therefore not a clean teacher-student comparison, thus we exclude FLUX Schnell from Tab. 1.

Beyond binary flag rates, we exploit the internal mechanism of the SD-Checker to derive a continuous unsafety score. Internally, the SD-Checker embeds an image with CLIP and measures its cosine similarity to a fixed collection of unsafe-concept embeddings, flagging the image as unsafe whenever its similarity to any concept exceeds a per-concept threshold [33]. We define the SD-Checker severity of an image as the maximal cosine similarity between its CLIP embedding and any unsafe concept in this collection. Fig. 1 shows the severity distribution, displayed as a continuous measure of unsafe content instead of a binary prediction. The student’s distribution is shifted toward higher severity values, and the effect is most pronounced in the upper tail: 77% of the images with a severity above 0.25 are produced by the SD3.5 Turbo few-step student.

## 3.3. Step Count Is a Hidden Safety Knob

Following §3.2’s findings, we try to further understand why few-step models generate more unsafe content. To this end, we check the model unsafe generation rates by the number of inference steps used. We discover that the number of steps taken impacts the generation safety metrics.

We conduct an experiment where we generate 200 images per model and step count, making sure to re-use the same noise seeds for all configurations, so that the results of a given model across different step counts would be directly comparable. We condition the models on the prompt “A photo of a gorgeous woman, erotic”, and measure the cosine similarity in CLIP’s embedding space between the generated image and the “nudity” concept. In Fig. 2, we show the cosine similarity by step count for the baseline models—there is an evident trend of decreased similarity with “nudity” as the number of steps increases, both for few-step and full-step models. Concretely, lowering the step count raises the mean CLIP similarity to “nudity” by up to ∼9% for full-step models (FLUX Dev, 0.239 to 0.261 going from 50 to 8 steps) and up to ∼7% for few-step models (SD3.5 Turbo, 0.231 to 0.248 going from 8 to 1 step).

For quantitative results covering more prompts, we run a similar experiment over NSFW-1k. Tab. 2 shows that almost all unsafety metrics increase when reducing the number of inference steps to 16 in baseline models—a lower step count number that produces coherent images in full-step models. (Results are consistent for violence; see Tab. 13 in App. B.6.) To see whether this effect is replicated on defended models as well, we also check nudity-defended variants of both SD3.5 Large and FLUX Dev found to exhibit low unsafe content-generation rates (from Tab. 3), and test them using 16-step generation. On average, the NudeNet unsafe generation rates increase by ∼10%.

![](images/3778b92857278ed50995f848ad133e4fa30b19aa5e51d9d89dd3c772b5cdd31f.jpg)

![](images/6f857636fefcc223aa4f36859bdb5efafd1b8bbcd6e1698c231817b521f3048d.jpg)  
Figure 2. CLIP similarity of generated images to “nudity”, by number of inference steps. We show 8-50 steps for full-step models at the top, and 1-8 steps for few-step models at the bottom.

TABLE 2. UNSAFE GENERATION RATES AS A FUNCTION OF INFERENCE-STEP COUNT OVER NSFW-1K. FOR EACH BASE AND DEFENDED FULL-STEP MODEL WE CONTRAST ITS DEFAULT STEP COUNT AGAINST A REDUCED 16-STEP SETTING. REDUCING THE NUMBER OF STEPS INCREASES UNSAFETY FOR ALMOST ALL METRICS. HIGHER VALUES INDICATE LESS SAFE MODELS.
<table><tr><td>Model</td><td>Defended</td><td>Steps</td><td>NudeNet ↓</td><td>NudeNet@5↓</td><td>SD↓</td><td>SD@5↓</td></tr><tr><td>SD3.5 Large</td><td>x</td><td>28</td><td>0.2432</td><td>0.5240</td><td>0.5416</td><td>0.8150</td></tr><tr><td></td><td></td><td>16</td><td>0.2492</td><td>0.5390</td><td>0.5570</td><td>0.8370</td></tr><tr><td>FLUX Dev</td><td>x</td><td>50 16</td><td>0.2562 0.2808</td><td>0.5570 0.5890</td><td>0.6108 0.6122</td><td>0.8570 0.8420</td></tr><tr><td>SD3.5 Large +</td><td>√</td><td>28</td><td>0.0822</td><td>0.2650</td><td>0.3888</td><td>0.7190</td></tr><tr><td>ESD</td><td></td><td>16</td><td>0.0862</td><td>0.2850</td><td>0.4068</td><td>0.7140</td></tr><tr><td>FLUX Dev +</td><td></td><td>50</td><td>0.0816</td><td></td><td></td><td></td></tr><tr><td>EraseAnything</td><td>√</td><td>16</td><td>0.1000</td><td>0.2550 0.3080</td><td>0.4550 0.4860</td><td>0.7630 0.7820</td></tr></table>

## 3.4. Integrating Erasure With Few-Step Models

We now assess three intuitive means for integrating erasure with few-step acceleration at three different points. In pre-distillation erasure, the teacher is erased first and the erased teacher is then distilled into a few-step student. In post-distillation erasure, the erasure method is applied directly to an already-distilled few-step checkpoint. In Teacher-LoRA transfer [17], an erasure LoRA trained on the teacher is loaded onto the student. These choices differ in practicality: pre-distillation erasure requires teacher access and an expensive distillation run for each erased concept; Teacher-LoRA transfer requires teacher access, a shared teacher-student architecture, and a LoRA-based erasure method.

Post-distillation erasure is the most easily deployable because it only requires the released few-step model; we therefore first evaluate it. In Tab. 3, we apply existing erasure methods directly to few-step models and compare them with the same methods applied to their full-step counterparts. The trend is consistent across methods and metrics: postdistillation erasure leaves substantially higher unsafe generation rates on few-step students. For example, with ESD, NudeNet unsafe rates increase from 0.0572 on FLUX Dev to 0.0842 on FLUX SCFM, and from 0.0822 on SD3.5 Large to 0.1504 on SD3.5 Turbo. EraseAnything shows a similar pattern, increasing from 0.0816 to 0.1628 in the FLUX family and from 0.1480 to 0.2992 in the SD3.5 family. FLUX SCFM, which is more lightly accelerated, is generally easier to erase than the more strongly distilled SD3.5 Turbo, but it still does not match its full-step teacher across methods. At the same time, erasure does not spare few-step models from quality degradation: their HPSv2 drop is comparable to that of full-step models, and is often larger. The full safety and quality results are provided in Tab. 9 (App. B.1).

TABLE 3. SAFETY AND QUALITY OF ERASURE METHODS ACROSS BASELINE AND FEW-STEP MODELS. NUDENET AND SD-CHECKER REPORT UNSAFE GENERATION RATES OVER NSFW-1K (LOWER IS BETTER); HPSV2 REPORTS BENIGN GENERATION QUALITY (HIGHER IS BETTER). WITHIN EACH TEACHER-STUDENT PAIR, THE BETTER VALUE (LOWEST FOR NUDENET AND SD-CHECKER, HIGHEST FOR HPSV2) IS BOLDFACED.
<table><tr><td>Method</td><td>Family</td><td>Variant</td><td>NudeNet ↓</td><td>SD-Checker ↓</td><td>HPSv2 ↑</td></tr><tr><td rowspan="3">Undefended</td><td>SD3.5</td><td>Large (teacher) Turbo (student)</td><td>0.2432 0.3232</td><td>0.5416 0.6646</td><td>0.2905 0.2874</td></tr><tr><td></td><td>Dev (teacher)</td><td>0.2562</td><td>0.6108</td><td>0.2938</td></tr><tr><td>FLUX</td><td>SCFM (student)</td><td>0.3282</td><td>0.7074</td><td>0.2781</td></tr><tr><td rowspan="3">ESD</td><td>SD3.5</td><td>Large (teacher) Turbo (student)</td><td>0.0822 0.1504</td><td>0.3888</td><td>0.2877</td></tr><tr><td></td><td>Dev (teacher)</td><td>0.0572</td><td>0.5692</td><td>0.2845</td></tr><tr><td>FLUX</td><td>SCFM (student)</td><td>0.0842</td><td>0.3254 0.4626</td><td>0.2947 0.2784</td></tr><tr><td rowspan="3">Erase- Anything</td><td>SD3.5</td><td>Large (teacher)</td><td>0.1480</td><td>0.4710</td><td>0.2821</td></tr><tr><td rowspan="2">FLUX</td><td>Turbo (student)</td><td>0.2992</td><td>0.6626</td><td>0.2754</td></tr><tr><td>Dev (teacher)</td><td>0.0816</td><td>0.4554</td><td>0.2936</td></tr><tr><td rowspan="4">UCE</td><td>SD3.5</td><td>SCFM (student)</td><td>0.1628</td><td>0.5614</td><td>0.2787</td></tr><tr><td rowspan="2"></td><td>Large (teacher)</td><td>0.1850</td><td>0.4648</td><td>0.2894</td></tr><tr><td>Turbo (student)</td><td>0.2254</td><td>0.5806</td><td>0.2865</td></tr><tr><td rowspan="2">FLUX</td><td>Dev (teacher)</td><td>0.2108</td><td>0.5690</td><td>0.2939</td></tr><tr><td>SCFM (student)</td><td>0.2780</td><td>0.6630</td><td>0.2781</td></tr></table>

We exclude FLUX Schnell from Tab. 3 because, as discussed in §3.2, its teacher is undisclosed and it therefore does not form a clean teacher-student pair. Nonetheless, the same qualitative trend holds: erasure is markedly less effective on FLUX Schnell than on the full-step FLUX Dev. With ESD, the residual NudeNet unsafe rate is 0.1788 on FLUX Schnell vs. 0.0572 on FLUX Dev, and with EraseAnything it is 0.1608 vs. 0.0816; SD-Checker shows the same gap (e.g. 0.4706 vs. 0.3254 under ESD). This suggests that the reduced erasure efficacy is associated with few-step acceleration in general, rather than with any particular teacher-student pairing.

The comparison above covers post-distillation erasure, which is the only strategy that is broadly applicable: it requires nothing beyond the released few-step checkpoint. The other two strategies (pre-distillation erasure and Teacher-LoRA transfer) impose far more restrictive constraints, and are not always feasible—for example, both require access to the teacher’s weights. In Fig. 3, we compare the different integration strategies for EraseAnything on FLUX SCFM and SD3.5 Turbo on NSFW-1k. We choose EraseAnything because it is LoRA-based, which lets us test Teacher-LoRA transfer. Since ADD is very resource intensive, we do not compute the pre-distillation erasure result for SD3.5 Turbo. None of the erasure strategies yield few-step models as safe as the teacher model produced after erasure. However, Teacher-LoRA transfer achieves the closest result to erasure on the base teacher, suggesting that few-step models are not inherently impossible to erase; rather, existing erasure methods become less effective when applied directly to the few-step student.

![](images/af63a0a262e16ccce0ffa11d2b1d2a12d303fe573e8af5d5d45a394610accce9.jpg)  
Figure 3. NudeNet rates on NSFW-1k for EraseAnything using post-distillation erasure, pre-distillation erasure, and Teacher-LoRA transfer erasure application methods. Grouped by model family, FLUX SCFM is tested as FLUX Dev’s student and SD3.5 Turbo as SD3.5 Large’s student. For SD3.5, we do not provide the pre-distillation erasure due to the heavy compute requirements of ADD.

Since post-distillation erasure is the only broadly deployable strategy, the practical goal is to make it effective on few-step students—achieving this requires an erasure method designed around the behavior of few-step models. In the next section, we analyze what makes these models harder to erase to motivate such a method.

## 4. Mechanistic Diagnosis of Erasure’s Failure

We investigate three candidate explanations for why erasure fails on few-step models. We first rule out the most obvious one—a direct conflict between the speed and erasure objectives (§4.1). We then identify two mechanisms connected to erasure’s failure: aggressively distilled models are geometrically fragile, amplifying any perturbation to their weights or latents (§4.2), and the concept direction that fine-tuning-based erasure relies on is degraded by how guidance is handled during distillation (§4.3).

## 4.1. Non-Conflicting Speed and Erasure Objectives

We set out to determine whether there is an inherent conflict between the few-step objective and the erasure objective. Because LADD distillation is resource-heavy and time-consuming, we focus on SCFM distillation. For the erasure objective we adopt the ESD loss, owing to its popularity and simplicity and because it forms the erasure term in many fine-tuning-based erasure methods [11], [12], [14], [15], [22], [23]. We compute the ESD gradients with respect to the same Low-Rank Adaptation (LoRA) [24] adapter parameters SCFM optimizes in every layer ℓ: $A _ { \ell } , B _ { \ell }$ (Eq. 3). The tuned layers match those used in the SCFM method [8] and cover both the attention and feed-forward layers of the transformer blocks.

We run the initial phase of SCFM distillation training on both SD3.5 Large and FLUX Dev. We compute the ESD gradient as in Eq. 2 with respect to the LoRA parameters $A _ { \ell } , B _ { \ell }$ of every attention and feed-forward layer ℓ (and analogously for the SCFM gradient at the same iteration), without taking an optimization step in the ESD direction. The ESD gradients are computed over a set of 50 images sampled from the base model, using the “nudity” concept— the original concept used by the ESD method. For each pair of SCFM and ESD gradients, we use the same noise level $\sigma _ { t } .$ . The ESD gradients are computed every 8 steps over the first 20k images, which constitute the initial 12.5% iterations of SCFM distillation. Restricting the analysis to this early window, prior to convergence, keeps the model close to the base model, where the erasure and acceleration gradients are strongest.

Following prior work in multi-task and multi-objective optimization [40], [41], we quantify objective conflict by measuring the cosine similarity between the gradients induced by the two losses on the shared trainable parameters. We compute the cosine similarity between the ESD and SCFM gradients and group the cosine-similarity values into three noise-level bins with equal size: early, mid, and late. We report the mean and standard deviation of these values per bin in Tab. 4.

The gradients are near-orthogonal (i.e., mean cosine similarity ∼0) at every trajectory phase on both models, with maximum observed absolute cosine similarity of 0.0158. These results imply that, in the case of SCFM and ESD, there is no evidence of a meaningful direct first-order con flict between the erasure and speed objectives. This suggests that the observed erasure failures on few-step models are not attributable to direct objective opposition but rather to some other characteristic of few-step models.

TABLE 4. COSINE SIMILARITY BETWEEN THE SCFM (SPEED) AND ESD (ERASURE) GRADIENTS ON THE SHARED LORA PARAMETERS, REPORTED AS MEAN ± STD OVER GRADIENT ITERATIONS AND GROUPED BY NOISE-LEVEL (σ<sub>t</sub>) TIMESTEP BIN (EARLY, MID, LATE). NEAR-ZERO VALUES AT EVERY PHASE INDICATE THE TWO OBJECTIVES ARE CLOSE TO ORTHOGONAL RATHER THAN CONFLICTING, ON BOTH FLUX DEV AND SD3.5 LARGE.
<table><tr><td>t Bin</td><td>FLUX Dev</td><td>SD3.5 Large</td></tr><tr><td>[0.66, 1.00]</td><td> $- 0 . 0 0 2 4 \pm 0 . 0 4 1 9$  一</td><td> $+ 0 . 0 1 0 6 \pm 0 . 0 8 1 0$ </td></tr><tr><td>[0.33, 0.66]</td><td> $- 0 . 0 1 5 8 \pm 0 . 0 9 4 9$ </td><td> $- 0 . 0 1 5 0 \pm 0 . 0 7 4 3$ </td></tr><tr><td>[0.00, 0.33]</td><td> $- 0 . 0 0 4 3 \pm 0 . 0 4 4 0$ </td><td> $+ 0 . 0 0 2 7 \pm 0 . 0 5 2 0$ </td></tr></table>

## 4.2. Aggressive Distillation Amplifies Velocity-Field Sensitivity

We find that aggressively distilled models’ few-step generation is more sensitive to perturbations of its flow path and weights than baseline generation, making such few-step models more fragile to post-hoc edits and to how latents are sampled for erasure.

Specifically, we seek to assess the sensitivity of velocity predictions in few-step models to changes in the latent z and the weights θ. To do so, we estimate an empirical Lipschitz proxy [42] of the velocity field $v _ { \theta } ( z , c , t )$ w.r.t. z and θ for both few-step and full-step models at different points along the flow path. Formally, the Lipschitz constant of the unconditioned velocity field w.r.t. z over a neighborhood of radius r is

$$
L _ { z } ( z , \emptyset , t ) = \operatorname* { s u p } _ { 0 < \| \delta \| \leq r } \frac { \| v _ { \theta } ( z + \delta , \emptyset , t ) - v _ { \theta } ( z , \emptyset , t ) \| } { \| \delta \| }\tag{4}
$$

where $\| \cdot \|$ denotes the Frobenius norm, unless explicitly specified otherwise. Since the supremum over all δ in the neighborhood is generally intractable for a neural network [43], we estimate an empirical lower-bound proxy by replacing it with a maximum over a finite set of sampled perturbation directions, following prior empirical studies of neural-network Lipschitzness [42]. We select 500 images from LAION-Pop as clean images so the images are not drawn from any single model’s own generative distribution, and, for each image, randomly draw a timestep t from the schedule of the tested model and add the corresponding noise level $\sigma _ { t } .$ . For each $( x _ { 0 } , t )$ pair we form $z = x _ { 0 } + \sigma _ { t } ( x _ { 1 } - x _ { 0 } )$ . We then sample k Gaussian perturbations $\delta _ { 1 } , . . . , \delta _ { k } .$ , each normalized to norm $\epsilon \lVert z \rVert$ (the exact k and ϵ used, together with a sensitivity analysis showing the findings are robust to both, are detailed in App. B.2), and estimate the local Lipschitz constant for each $x _ { 0 } ,$ , averaging across samples:

$$
\mathbb { E } _ { x _ { 0 } \sim \mathcal { D } } \left[ \operatorname* { m a x } _ { i } \frac { \| v _ { \theta } ( z + \delta _ { i } , \emptyset , t ) - v _ { \theta } ( z , \emptyset , t ) \| } { \| \delta _ { i } \| } \right]\tag{5}
$$

In addition to perturbing the input $z ,$ we also estimate the local Lipschitz constant of the velocity w.r.t. perturbations $\Delta _ { i }$ in the weights θ:

$$
\mathbb { E } _ { x _ { 0 } \sim \mathcal { D } } \left[ \operatorname* { m a x } _ { i } \frac { \| v _ { \theta } ( z , \mathcal { O } , t ) - v _ { \theta + \Delta _ { i } } ( z , \mathcal { O } , t ) \| } { \| \Delta _ { i } \| } \right]\tag{6}
$$

TABLE 5. MEDIAN FEW-STEP-TO-FULL-STEP VELOCITY-FIELD LIPSCHITZ RATIOS PER MODEL FAMILY, FOR THE INPUT-SPACE (z) AND PARAMETER-SPACE (θ) ESTIMATORS OF §4.2. EACH RATIO IS THE MEDIAN OVER THE LAION-POP SAMPLES OF THE PER-IMAGE FEW-STEP LIPSCHITZ DIVIDED BY ITS FAMILY FULL-STEP REFERENCE’S; VALUES ABOVE 1 INDICATE THE FEW-STEP MODEL IS MORE SENSITIVE THAN ITS FULL-STEP REFERENCE.
<table><tr><td>Family</td><td>Variant</td><td>Median ratio  $\operatorname* { l i p } _ { z }$ </td><td>Median ratio lipθ</td></tr><tr><td rowspan="2">FLUX</td><td>Dev vs. Schnell</td><td>4.78</td><td>4.37</td></tr><tr><td>Dev vs. SCFM</td><td>1.01</td><td>1.07</td></tr><tr><td>SD3.5</td><td>Large vs. Turbo</td><td>1.81</td><td>2.32</td></tr></table>

![](images/44a34fa183159ba133ae0d80838847d9661dd3ebe1b95041939d6bd3ac629ce7.jpg)  
Figure 4. Empirical local sensitivity of the velocity field with respect to latent z perturbations in FLUX Dev and FLUX Schnell, plotted by noise level $\sigma _ { t } .$ Error bars denote the standard deviation of the constant across image samples.

where each per-tensor block $\Delta _ { i , \ell } \sim \mathcal { N } ( 0 , I )$ is independently normalized to norm $\epsilon | | \theta _ { \ell } | |$ |. This Lipschitz constant w.r.t. the model weights estimates how sensitive the velocity prediction is to changes in θ.

To obtain a single, aggregate metric per model pair, we define the paired Lipschitz ratio as the ratio of the few-step model’s estimated Lipschitz constant to the full-step reference model’s, computed for each $x _ { 0 }$ and averaged across timesteps. In Tab. 5, we report the median paired Lipschitz ratios across model families. For FLUX Schnell, whose true teacher is not disclosed, we compare against FLUX Dev as the closest public full-step FLUX reference. We find that the empirical Lipschitz estimate of the aggressively distilled few-step models (SD3.5 Turbo and FLUX Schnell), both w.r.t. $z$ and $\theta ,$ is consistently higher—roughly ${ \sim } 2 \times$ for SD3.5 and ${ \sim } 4 { - } 5 \times$ for FLUX—than that of their withinfamily full-step variants. The lightly-distilled FLUX SCFM is the exception: as a consistency-distilled LoRA student, it tracks its FLUX Dev teacher closely (≈ 1.0 for both z and θ), suggesting that fragility is more pronounced when distillation is more aggressive. We also observe that, unlike full-step models, the estimated Lipschitz constant of fewstep models is especially large at high noise levels in the generation process where $\sigma _ { t } \approx 1$ (see Fig. 4); the opposite of the trend seen in baseline models.

Taken together, the estimated Lipschitz constants, both w.r.t. the input z and w.r.t. the parameters θ, indicate that aggressively distilled few-step models are generally more sensitive to weight updates and to how the latent z is constructed for erasure. For fine-tuning-based erasure methods, this sensitivity can translate into fragile, unstable convergence.

## 4.3. Distillation Degrades the Erasure-Gradient Direction

Degradation hypothesis We hypothesize that guidance distillation in few-step models degrades the erasure direction. Fine-tuning-based erasure suppresses a concept by regressing the conditional prediction along $- \delta ,$ where the concept direction δ is defined in Eq. 7. We also define the respective full-step reference concept axis $\delta ^ { * }$ for the full-step model $v ^ { * } \colon$

$$
\delta = v ( z , c ) - v ( z , \emptyset ) , \qquad \delta ^ { * } = v ^ { * } ( z , c ) - v ^ { * } ( z , \emptyset )\tag{7}
$$

In our diagnosis, we focus on the ESD loss because it is the main erasure term used in the loss of many fine-tuningbased erasure methods [11], [12], [14], [15], [22], [23]. Concretely, ESD fine-tunes the model so that its concept prediction $v ( z , c )$ matches, under an MSE loss, the CFGagnostic target $v ( z , \emptyset ) - \eta ( v ( z , c ) - v ( z , \emptyset ) )$ , where η is a negative guidance scale hyperparameter. The ESD target is computed from a frozen copy of the same model, for latents z sampled along the model’s own trajectory, generated using the concept prompt conditioning (described below); near initialization, this is equivalent to pushing the prediction along $- \delta ,$ , away from the concept (App. B.3). Since erasure is usually a short fine-tuning process of only ∼100–200 iterations, the edited model remains close to this frozen copy for most of optimization, making this approximation useful.

The conditioned branch $v ( z , c )$ is directly reshaped by distillation. Few-step models with baked-in guidance such as SD3.5 Turbo and FLUX Schnell are trained to reproduce CFG sampling in a single forward pass, without an explicit guidance step; their conditional branch therefore bakes in classifier-free guidance, so $v ( z , c )$ approximates the CFGcombined full-step reference rather than the raw teacher velocity:

$$
\begin{array} { c } { { v ( z , c ) \approx v ^ { * } ( z , \emptyset ) + \omega ( v ^ { * } ( z , c ) - v ^ { * } ( z , \emptyset ) ) } } \\ { { v ( z , c ) \not \approx v ^ { * } ( z , c ) . } } \end{array}
$$

(The exact ω used during the distillation process of SD3.5 Turbo and FLUX Schnell is undisclosed.)

The unconditioned branch is also altered by distillation, but in a different way than the conditioned branch. The unconditioned branch $v ( z , \emptyset )$ is never directly trained in LADD or SCFM distillations, making it unclear how its velocity trajectory changes compared to the baseline model. At the same time, $v ( z , \emptyset )$ shares all the parameters θ with the tuned conditioned path $v ( z , c )$ , so it is still reshaped indirectly as those shared parameters are updated.

Because the conditioned branch $v ( z , c )$ and the unconditioned branch $v ( z , \emptyset )$ that form δ are reshaped by distillation in different ways, unlike standard full-step T2I model training where both are optimized together, the erasure direction may be degraded with irrelevant information. Compensating for this degradation analytically is impractical, since the student’s branches do not exactly reproduce the full-step baseline’s $( v ( z , \emptyset ) \not \approx v ^ { * } ( z , \emptyset ) )$ and the distillation-time ω is usually undisclosed. Consequently, as we show next, erasure receives a direction that points in a different direction than the full-step reference’s and is substantially larger in magnitude.

Measuring the concept-direction degradation To put the hypothesis to the test, we compute δ and $\delta ^ { * } \left( \mathrm { E q . ~ } 7 \right)$ , and measure the Frobenius norm of both and the cosine similarity between them using matching seeds. If the ESD direction is indeed degraded in few-step models, we expect to see low similarity between them, and a distorted magnitude $\lvert \lVert \delta \rVert - \lVert \delta ^ { * } \rVert \stackrel { \cdot } { \gg } 0$ . We draw 1,000 image-timestep pairs—a sample size that supplies us with a large enough standard deviation number to draw meaningful conclusions and is small enough to run efficiently. Using synthetic images may bias the results in the direction of the model which generated them—so we use real images from LAION-Pop. Each image-timestep pair samples a LAION-Pop image latent $x _ { 0 } ,$ draws $\sigma _ { t }$ uniformly in [0, 1], and forms the off-policy latent $z = ( 1 - \sigma _ { t } ) x _ { 0 } + \sigma _ { t } \epsilon .$ , so the latents are not drawn from any single model’s own generative distribution. On each z we evaluate the concept direction $\delta = v ( z , c ) - v ( z , \emptyset )$ for every model (FLUX Dev, FLUX Schnell, FLUX SCFM, SD3.5 Large and SD3.5 Turbo), with $c { = } ^ { \cdots } \mathrm { n u d i t y } ^ { \prime \prime }$ , the concept originally used in the ESD method.

For the aggressively-distilled few-step variants, each variant’s concept direction δ points in a markedly different direction compared to its in-family full-step reference $\delta ^ { * }$ The average cosine similarity between the two is only 0.30 for SD3.5 Turbo (against its SD3.5 Large reference) and 0.07 for FLUX Schnell (against FLUX Dev). For few-step models, the erasure direction is also larger in magnitude, by ${ \approx } 1 . 2 – 3 . 9 \times$ for SD3.5 Turbo and ${ \approx } 2 { - } 5 \times$ for FLUX Schnell (against FLUX Dev), with the absolute gap widest at highnoise timesteps (Fig. 5). This implies that the velocities $v ( z , c )$ and $v ( z , \emptyset )$ are significantly closer to each other in the full-step model than in the few-step variant, degrading the erasure objective of the student. This holds even though all four velocities $\| v ( z , c ) \| , \| v ( z , \emptyset ) \| , \| v ^ { * } ( z , c ) \|$ and $\overline { { \| \boldsymbol { v } ^ { * } ( \boldsymbol { z } , \boldsymbol { \mathcal { O } } ) \| } }$ have similar norms (App. B.4). In other words, the inflation reflects poorer cancellation, not larger predictions. The lightly distilled FLUX SCFM follows this trend only weakly: its δ stays positively aligned with FLUX Dev’s $( \cos = 0 . 6 4 )$ and is only ∼1.4× larger in magnitude, far milder than the aggressively distilled variants above, consistent with its lighter distillation.

The largest absolute distortion in δ occurs where fewstep erasure methods are most likely to sample their trainingtime latents (z) from: Existing fine-tuning-based erasure methods typically sample z on-policy from the model’s own denoising trajectory; for aggressive few-step $\sigma _ { t }$ schedules, especially one-step schedules, those samples are naturally concentrated near high-noise timesteps even without an explicit timestep shift in the scheduler. Thus, the degraded ESD direction is not merely present in few-step models:

![](images/1f8105ce628bc73eddafa7a84344e8167ef1e09edf937703d138cc896e33b30f.jpg)  
Figure 5. ||δ|| by t-bin. Compared to full-step models, the norm of the erasure direction ∥δ∥ is significantly higher in few-step models. The effect is strongest in high noise regions where $\sigma _ { t } \sim 1$

it is emphasized by the sampling distribution used during erasure, motivating the sampling strategy we use in §5.

Guidance bake-in contributes to erasure direction degradation Unlike models that apply CFG externally (e.g., SD3.5 Large) and have guidance-agnostic per-pass velocities, FLUX Dev and FLUX SCFM take the classifier-freeguidance scale as an explicit network input, letting us vary it to isolate its effect on δ. Raising CFG on FLUX Dev from 1.0 to 3.5 inflates ∥δ∥ about 3.6× (17.8→64.1) and nudges its cosine with FLUX Schnell’s δ upward (0.055→0.068); on FLUX SCFM the same sweep grows ∥δ∥ from 18.6 (CFG of 1.0) to 90.2 (CFG of 5.0), with FLUX Dev and FLUX SCFM near-identical at matched CFG. These results show that increasing guidance alone reproduces the δ-inflation seen in few-step models, suggesting guidance bake-in as a direct cause for the erasure degradation.

A natural remedy follows from the sweep above: since baked-in extra guidance is a large contributor to the inflation of δ, one could run the erasure itself using $\omega = 1$ , effectively disabling CFG (Eq. 2). This is only possible for models that expose the guidance scale as an explicit input (i.e. not applicable for FLUX Schnell nor SD3.5 Turbo). We discuss this in more detail in App. B.5.

Taken together, these results show that the ESD concept direction δ in few-step models is a degraded erasure target: it is rotated away from the full-step model direction and inflated in magnitude. The CFG sweep identifies guidance bake-in as a main driver: folding classifier-free guidance into the conditional velocity degrades $\delta ,$ and higher effective guidance inflates it further. This distortion is strongest at high noise inputs, directly motivating the sampling changes in §5. Even the lightly-distilled FLUX SCFM shows a milder version of this effect at its higher operational CFG scale.

## 5. Concept Erasure for Few-Step Models

We set out to design a robust concept erasure method for few-step T2I models. Across the different erasure integration points, we choose post-distillation erasure as (i) the teacher model does not need to be accessible or even known (e.g., FLUX Schnell); and (ii) the costly distillation process does not need to be modified or re-run for every erased concept.

Given those benefits, our method must overcome the empirical struggles of post-distillation erasure: it should suppress the target concept while preserving the few-step model’s benign generation quality. Our method addresses the two failure modes identified in §4: it avoids over-training on the high-noise latents where few-step models are most fragile and where the ESD concept direction is most degraded, and it constrains the edited model to stay close to the frozen pre-erasure few-step velocity field.

Toward attaining these objectives, we design Distillation-Integrated Concept Erasure (DICE) as a fine-tuning-based LoRA-based method, keeping erasure modular and inexpensive. DICE targets the weights of the joint-attention layers in both text and image streams of the MMDiT architecture used in SD3.5 and FLUX, where concept-bearing text tokens interact with image tokens.

DICE objective Let $\theta ^ { * }$ denote the frozen pre-erasure fewstep model, and let θ denote the edited model obtained by training LoRA adapters on top of it while keeping the base weights fixed. DICE optimizes a fixed weighted sum of three losses:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D I C E } } ( \theta ) = \mathbb { E } _ { c , t , z } \left[ \mathcal { L } _ { \mathrm { e s d } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { a t t n } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { g e n } } \right] , } \end{array}\tag{8}
$$

where c is the erased concept prompt and t is the sampled noise level. The ESD term keeps the concept-suppression target used by many gradient-based erasure methods (Eq. 2). At first glance, continuing to use an ESD-style term may seem counterintuitive after §4.3, where we show that the ESD concept direction is degraded in few-step models. However, §4.1 suggests that the speed and erasure update directions are not inherently contradictory. DICE keeps the useful concept-suppression direction, but changes the conditions under which it is optimized: by controlling the latent distribution (§5.1), and preventing destructive drift away from the distilled model (§5.2).

We additionally adopt the attention-map suppression loss $\mathcal { L } _ { \mathrm { a t t n } }$ introduced by EraseAnything [14] to directly suppress the model’s internal use of the erased concept token. In the joint-attention blocks of the MMDiT, the text and image tokens are concatenated into a single sequence over which a softmax QK attention map assigns every (text or image) query token a distribution of attention weights across all key tokens. The key columns indexed by the erased concept’s tokenized positions therefore measure how strongly each query token attends to the concept, i.e., how much the concept token feeds the model’s internal representation. Following EraseAnything, $\mathcal { L } _ { \mathrm { a t t n } }$ is computed from the attention map produced for the concept prompt c: it restricts the map to concept key (e.g., “nude”) tokens and accumulates their magnitude into a single scalar that grows with the attention mass routed into the erased concept key text tokens. Minimizing $\mathcal { L } _ { \mathrm { a t t n } }$ suppresses concept-specific activation inside the transformer rather than only changing the final velocity prediction, and helps us overcome the degraded erasure direction (§4.3) better, as a secondary erasure loss not dependent on ESD.

The generation term prevents the edited model’s velocity field from diverging too far from the pre-erasure model’s velocity field:

$$
\mathcal { L } _ { \mathrm { g e n } } = \Vert v _ { \theta } ( z , c , t ) - v _ { \theta ^ { * } } ( z , c , t ) \Vert _ { 2 } ^ { 2 }\tag{9}
$$

We optimize Eq. 8 with standard stochastic first-order updates, using AdamW as the optimizer. Each iteration samples a timestep/noise level and builds the inputs for the three loss terms. In all experiments, DICE targets the query, key, value, and output projections of the joint-attention blocks. Technical details are supplied in App. A.3.

## 5.1. Straight-Path Latent Construction

Few-step models are trained to follow a near-straight flow path, allowing them to generate images in a small number of steps. In §4.2, we saw that this also makes their velocity field unusually sensitive to perturbations of both the latent z and the weights θ, especially near the highnoise end of the trajectory. Therefore, instead of optimizing erasure using on-policy latents produced by partial model inference, we construct the erasure off-policy latent so that training stays aligned with the straight flow path.

DICE generates a clean concept-conditioned latent x using the frozen pre-erasure model, and then linearly renoises it:

$$
z = x _ { 0 } + \sigma _ { t } ( x _ { 1 } - x _ { 0 } ) ,
$$

where $x _ { 1 }$ is Gaussian noise and $\sigma _ { t } ~ \in ~ [ 0 , 1 ]$ is the noise level. By construction, z lies exactly on the straight path between the clean latent $x _ { 0 }$ and noise $x _ { 1 }$ at every noise level $\sigma _ { t }$ , keeping the erasure update consistent with the trajectory geometry that few-step inference relies on. DICE constructs the latent for all losses in $\mathcal { L } _ { \mathrm { D I C E } }$ in this way.

The degradation is strongest at high noise levels (Fig. 4 and Fig. 5 of §4). As on-policy sampling draws the noise level from the model’s own few-step inference schedule, it is naturally biased toward exactly these high-noise timesteps. Notably, for a one-step schedule, the sampled latent is always the pure-noise endpoint. To avoid amplifying that regime, DICE decouples the choice of noise level from the inference schedule. Particularly, at each iteration, DICE samples $\sigma _ { t }$ uniformly over [0, 1], so that the trained latent z has uniform coverage of the flow path rather than the high-noise concentration of on-policy sampling.

## 5.2. Self-Distilled Weight Preservation

Typically, flow-matching T2I models are trained with the target $\dot { \mathcal { L } } _ { \mathrm { F M } } = | | v _ { \theta } ( z , c , \bar { t } ) - ( x _ { 1 } - x _ { 0 } ) | | ^ { 2 }$ . For few-step distilled models, however, this raw target is non-zero at initialization, so optimizing it injects large gradients that push the weights away from their learned few-step trajectory.Fewstep models are typically shaped by teacher-derived distillation signals rather than by direct regression to $x _ { 1 } - x _ { 0 } \colon$ ADD includes a direct student-teacher distillation loss term; LADD aligns the student’s generated latents with teachergenerated latent samples under adversarial supervision, using features from the frozen teacher; and SCFM trains the student to reproduce the teacher’s velocity behavior over longer sampling jumps.

In DICE, we define preservation relative to the frozen pre-erasure few-step model $\theta ^ { * }$ , as in Eq. 9. This results in local self-distillation: DICE does not require access to the original full-step teacher, and $\theta ^ { * }$ is simply the checkpoint being erased before the LoRA update. Optimizing $\mathcal { L } _ { \mathrm { g e n } }$ acts as a softer preservation constraint for the few-step model. Since DICE starts from the pre-erasure checkpoint, $\mathcal { L } _ { \mathrm { g e n } }$ is exactly zero at initialization and only produces gradients when the erasure and attention losses move the velocity field away from the original distilled model. In contrast, optimizing the raw flow-matching target $\mathcal { L } _ { \mathrm { F M } }$ during erasure can introduce large preservation gradients that potentially harm few-step models which are highly sensitive to perturbations (§4.2). Matching the frozen checkpoint $\mathcal { L } _ { \mathrm { g e n } }$ penalizes weight drift while keeping the model’s distilled behavior.

## 5.3. Evaluating DICE

We evaluate DICE on the three few-step models used throughout the paper—specifically, SD3.5 Turbo, FLUX Schnell, and FLUX SCFM—comparing it against the undefended checkpoint, and four erasure methods—ESD, EraseAnything, UCE and SPEED. Consistent with previous experiments (§3), safety is measured on NSFW-1k using NudeNet and SD-Checker, including the @5 repeatedsampling metrics; benign quality is measured with HPSv2 at one-step and four-step inference.

Tab. 6 shows that DICE gives the strongest overall safety-quality trade-off in nudity erasure. On SD3.5 Turbo, DICE reduces NudeNet from 0.3232 to 0.0324, compared with 0.1504 for ESD and 0.2254 for UCE, while the fourstep HPSv2 score drops only from 0.2874 to 0.2830. On FLUX Schnell, DICE reduces NudeNet from 0.2348 to 0.0456, substantially below the best prior method in the table, while keeping four-step HPSv2 within 0.0022 of the undefended model. We also provide quantitative results on the popular I2P benchmark (App. B.7) and FID scores on the COCO-30k dataset (App. B.8), where DICE replicates its strong safety improvement and generation quality performance. Violence erasure exhibits consistent trends (Tab. 14 in App. B.6).

The qualitative examples in Figs. 6–7 show the same trend visually: DICE suppresses the erased concept without a visible degradation in benign image quality. Additional qualitative examples, including a qualitative comparison vs. existing erasure methods, are provided in App. B.9.

SD3.5 Turbo  
TABLE 6. NSFW-1K SAFETY AND HPSV2 QUALITY ACROSS THE FEW-STEP MODEL FAMILIES. LOWER NUDENET AND SD-CHECKER VALUES INDICATE SAFER MODELS; HIGHER HPSV2 VALUES INDICATE BETTER GENERATION QUALITY. THE BEST VALUE IN EACH COLUMN, WITHIN EACH MODEL, IS IN BOLD.
<table><tr><td>Model</td><td>Method</td><td>NudeNet↓</td><td>NudeNet@5↓</td><td>SD-Checker ↓</td><td>SD-Checker@5↓</td><td>HPSv2 (1-step) ↑</td><td>HPSv2 (4-step) ↑</td></tr><tr><td rowspan="6">SD3.5 Turbo</td><td>Undefended</td><td>0.3232</td><td>0.5870</td><td>0.6646</td><td>0.8670</td><td>0.2551</td><td>0.2874</td></tr><tr><td>EraseAnything</td><td>0.2992</td><td>0.5720</td><td>0.6626</td><td>0.8700</td><td>0.2453</td><td>0.2754</td></tr><tr><td>ESD</td><td>0.1504</td><td>0.3700</td><td>0.5692</td><td>0.8210</td><td>0.2467</td><td>0.2845</td></tr><tr><td>UCE</td><td>0.2254</td><td>0.4810</td><td>0.5806</td><td>0.8260</td><td>0.2547</td><td>0.2865</td></tr><tr><td>SPEED</td><td>0.2894</td><td>0.5460</td><td>0.6362</td><td>0.8480</td><td>0.2551</td><td>0.2874</td></tr><tr><td>DICE</td><td>0.0324</td><td>0.1230</td><td>0.4308</td><td>0.7530</td><td>0.2425</td><td>0.2830</td></tr><tr><td rowspan="6">FLUX Schnell</td><td>Undefended</td><td>0.2348</td><td>0.5190</td><td>0.5622</td><td>0.8020</td><td>0.2886</td><td>0.2909</td></tr><tr><td>EraseAnything</td><td>0.1608</td><td>0.4150</td><td>0.5258</td><td>0.8120</td><td>0.2859</td><td>0.2889</td></tr><tr><td>ESD</td><td>0.1788</td><td>0.4360</td><td>0.4706</td><td>0.7700</td><td>0.2704</td><td>0.2796</td></tr><tr><td>UCE</td><td>0.1824</td><td>0.4540</td><td>0.4966</td><td>0.7690</td><td>0.2887</td><td>0.2908</td></tr><tr><td>SPEED DICE</td><td>0.2102</td><td>0.4920</td><td>0.5358</td><td>0.7990</td><td>0.2887</td><td>0.2909</td></tr><tr><td></td><td>0.0456</td><td>0.1700</td><td>0.2690</td><td>0.5680</td><td>0.2838</td><td>0.2887</td></tr><tr><td rowspan="6">FLUX SCFM</td><td>Undefended</td><td>0.3282</td><td>0.6390</td><td>0.7074</td><td>0.9020</td><td>0.2202</td><td>0.2781</td></tr><tr><td>EraseAnything</td><td>0.1628</td><td>0.4320</td><td>0.5614</td><td>0.8450</td><td>0.2195</td><td>0.2787</td></tr><tr><td>ESD</td><td>0.0842</td><td>0.2590</td><td>0.4626</td><td>0.7400</td><td>0.2184</td><td>0.2784</td></tr><tr><td>UCE</td><td>0.2780 0.3084</td><td>0.5850</td><td>0.6630</td><td>0.8770</td><td>0.2203</td><td>0.2781</td></tr><tr><td>SPEED DICE</td><td>0.0526</td><td>0.6100 0.1890</td><td>0.6848 0.4220</td><td>0.8960 0.7590</td><td>0.2203 0.2194</td><td>0.2780</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.2774</td></tr></table>

FLUX Schnell  
FLUX SCFM  
![](images/eab6608a7fa3309501fbe0fe1d6673dae57c0c9baf9394d1ba8da99a5249d6ee.jpg)  
Figure 6. Examples of DICE outputs using the prompt “A nude girl with beautiful hair.” Black rectangles are added by the authors for censorship. Fig. 8 provides qualitative comparisons with existing erasure methods.

DICE ablations Tab. 7 ablates the two few-step-aware mechanisms in DICE: straight-path re-noising (§5.1) and self-distilled preservation (§5.2). Replacing the self-distilled preservation target with the raw flow-matching target consistently weakens the safety-quality trade-off. Replacing straight-path re-noising with on-policy latent sampling can lower safety-detector rates further, but this comes with a large HPSv2 drop and visibly degraded images. The full DICE objective therefore gives the best balanced result.

Adversarial prompting We perform an adversarial evaluation for DICE, testing prompts optimized to circumvent erasure. We test two attacks: Ring-A-Bell [44] crafts modelagnostic prompts by targeting the text encoder, while UnlearnDiffAtk [45] optimizes a suffix per prompt while leveraging white-box access to the model. Both attacks are evaluated on their respective nudity-related prompt sets, which are based on I2P. Tab. 8 presents both attacks’ success rates against few-step models across erasure methods. Overall, it can be seen that attacks are powerful, with well beyond half of the prompts yielding unsafe images in some cases. Still, while there remains room for improvement, DICE remains the least vulnerable erasure method for all models against both attacks. We also test ReFlux [46], where the adversary optimizes a LoRA adapter over the model, finding it fails to meaningfully increase unsafe generation rates. We posit that ReFlux’s failure on few-step models can be attributed to the reasons presented in §4, since it operates similarly to erasure methods like EraseAnything, optimizing a similar (inverse) objective.

![](images/ce30ef3f2bf83adc2337b78d8dbd29dbc98a94cd38e514c78d42fd34e96580d3.jpg)  
Figure 7. Qualitative examples showing how DICE preserves the original model’s benign generational quality across diverse models and prompts. A qualitative comparison vs. existing erasure methods is supplied in App. B.9.

TABLE 7. DICE ABLATION RESULTS ACROSS BASE MODELS. WE REMOVE ONE DICE COMPONENT AT A TIME. LOWER NUDENET AND SD-CHECKER VALUES INDICATE SAFER MODELS; HIGHER HPSV2 VALUES INDICATE BETTER GENERATION QUALITY. THE BEST VALUE IN EACH COLUMN, WITHIN EACH MODEL, IS IN BOLD.
<table><tr><td>Model</td><td>Variant</td><td>NudeNet ↓</td><td>NudeNet@5↓</td><td>SD-Checker ↓</td><td>SD-Checker@5↓</td><td>HPSv2 (1-step) ↑</td><td>HPSv2 (4-step) ↑</td></tr><tr><td rowspan="3">SD3.5 Turbo</td><td>DICE (full)</td><td>0.0324</td><td>0.1230</td><td>0.4308</td><td>0.7530</td><td>0.2425</td><td>0.2830</td></tr><tr><td>straight-path re-noising</td><td>0.0266</td><td>0.1190</td><td>0.2340</td><td>0.5860</td><td>0.2219</td><td>0.2652</td></tr><tr><td>— self-distilled preservation</td><td>0.1254</td><td>0.3540</td><td>0.5268</td><td>0.8250</td><td>0.2317</td><td>0.2772</td></tr><tr><td rowspan="3">FLUX Schnell</td><td>DICE (full)</td><td>0.0456</td><td>0.1700</td><td>0.2690</td><td>0.5680</td><td>0.2838</td><td>0.2887</td></tr><tr><td>— straight-path re-noising</td><td>0.0802</td><td>0.2837</td><td>0.2792</td><td>0.6031</td><td>0.2811</td><td>0.2794</td></tr><tr><td>— self-distilled preservation</td><td>0.1026</td><td>0.3260</td><td>0.3854</td><td>0.6990</td><td>0.2831</td><td>0.2884</td></tr><tr><td rowspan="3">FLUX SCFM</td><td>DICE (full)</td><td>0.0526</td><td>0.1890</td><td>0.4220</td><td>0.7590</td><td>0.2194</td><td>0.2774</td></tr><tr><td>straight-path re-noising</td><td>0.0134</td><td>0.0620</td><td>0.1738</td><td>0.5180</td><td>0.2056</td><td>0.2650</td></tr><tr><td>self-distilled preservation</td><td>0.0600</td><td>0.2200</td><td>0.4530</td><td>0.7710</td><td>0.2185</td><td>0.2753</td></tr></table>

TABLE 8. NUDENET RATES (↓) OF ADVERSARIAL PROMPTING. BEST ERASURE METHOD PER ROW IS IN BOLD.
<table><tr><td>Attack</td><td>Model</td><td>Undefended</td><td>EA</td><td>ESD</td><td>UCE</td><td>SPEED</td><td>DICE</td></tr><tr><td rowspan="3">Ring-A-Bell</td><td>FLUX-Schnell</td><td>0.847</td><td>0.801</td><td>0.704</td><td>0.702</td><td>0.678</td><td>0.319</td></tr><tr><td>FLUX-SCFM</td><td>0.838</td><td>0.526</td><td>0.415</td><td>0.706</td><td>0.712</td><td>0.224</td></tr><tr><td>SD3.5-Turbo</td><td>0.726</td><td>0.729</td><td>0.195</td><td>0.611</td><td>0.638</td><td>0.123</td></tr><tr><td rowspan="3">UnlearnDiffAtk</td><td>FLUX-Schnell</td><td>0.879</td><td>0.848</td><td>0.758</td><td>0.818</td><td>0.848</td><td>0.455</td></tr><tr><td>FLUX-SCFM</td><td>0.909</td><td>0.727</td><td>0.545</td><td>0.795</td><td>0.886</td><td>0.500</td></tr><tr><td>SD3.5-Turbo</td><td>0.941</td><td>0.853</td><td>0.706</td><td>0.882</td><td>0.882</td><td>0.265</td></tr></table>

## 6. Conclusion

This paper examines an overlooked intersection between acceleration and safety in T2I models. We show that speedups introduce safety challenges across multiple settings: reducing inference steps, deploying existing few-step models, and applying concept erasure to few-step models. Our analysis identifies two properties that render it hard to erase concepts from few-step models: erasure direction is degraded with prompt guidance as a main contributor, and perturbations during high noise steps disproportionately affect few-step model outputs. To address these challenges, we introduce DICE, an erasure method tailored to few-step models that effectively removes unsafe concepts while preserving benign generation quality. Taken together, our results show that safety evaluations for few-step models should not be treated as a formality inherited from their teachers. Few-step models are not merely compressed versions of their teachers; distillation changes the generation dynamics, and safety mechanisms that appear effective in the original model may behave differently after acceleration. We hope that our findings, together with DICE, support the continued development of fast T2I systems whose safety properties are carefully evaluated.

Limitations and future work We now discuss several limitations and highlight directions for future work. Our mechanistic experiments are diagnostic rather than a complete causal account of why erasure degrades under distillation. We surface properties that correlate with erasure failure— guidance-degraded concept directions and heightened sensitivity of the velocity field to perturbations—but establishing a full causal explanation remains open.

Furthermore, this work tests nudity and violence erasure, which provide concrete and measurable settings for studying how distillation and safety interact. However, the same question arises for other forms of harmful or sensitive content, including deepfakes, political manipulation, trainingdata leakage [47], and unsafe generation of various concepts [48]. Extending the analysis to these settings is an important direction for future work.

We also note that our defense, DICE, is not bulletproof. While it enhances few-step models’ safety compared to established methods, it still produces unsafe content at non-negligible rates, especially under adversarial prompting (§5.3). We hope that future work can build on our findings to further advance T2I models’ safety.

Finally, our evaluation relies on automated classifiers (e.g., NudeNet) rather than human judgment. Human evaluation is the gold standard for assessing perceptual image quality and the success of concept erasure, but it is especially difficult in our setting because it would require annotators to view explicit and otherwise unsafe content. Developing safe and reliable human evaluation protocols for this setting remains an open challenge.

## Acknowledgments

We thank Or Patashnik and Rotem Shalev Arkushin for giving us access to LAION-Pop and their helpful feedback. This work has been supported in part by a grant from the

Blavatnik Interdisciplinary Cyber Research Center (ICRC) and the Tel Aviv University Center for AI and Data Science (TAD); by grant No. 2023641 from the United States-Israel Binational Science Foundation (BSF); by an Intel Rising Star Faculty Award; by Len Blavatnik and the Blavatnik Family foundation; and by a Maof prize for outstanding young scientists.

## References

[1] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in CVPR, 2022.

[2] Stability AI, “Introducing stable diffusion 3.5,” https://stability.ai/ news-updates/introducing-stable-diffusion-3-5, Oct. 2024.

[3] E. Xie, J. Chen, J. Chen, H. Cai, H. Tang, Y. Lin, Z. Zhang, M. Li, L. Zhu, Y. Lu et al., “Sana: Efficient high-resolution image synthesis with linear diffusion transformers,” in ICLR, 2025.

[4] B. F. Labs, “Flux,” https://github.com/black-forest-labs/flux, 2024.

[5] A. Sauer, D. Lorenz, A. Blattmann, and R. Rombach, “Adversarial diffusion distillation,” in ECCV, 2024.

[6] J. Chen, S. Xue, Y. Zhao, J. Yu, S. Paul, J. Chen, H. Cai, S. Han, and E. Xie, “Sana-sprint: One-step diffusion with continuous-time consistency distillation,” in ICCV, 2025.

[7] A. Sauer, F. Boesel, T. Dockhorn, A. Blattmann, P. Esser, and R. Rombach, “Fast high-resolution image synthesis with latent adversarial diffusion distillation,” in SIGGRAPH, 2024.

[8] X. Cai, Y. Wu, Q. Chen, H. Wu, L. Xiang, and H. Wen, “Shortcutting pre-trained flow matching diffusion models is almost free lunch,” in NeurIPS, 2026.

[9] P. Schramowski, M. Brack, B. Deiseroth, and K. Kersting, “Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models,” in ICCV, 2023.

[10] X. Li, Y. Yang, J. Deng, C. Yan, Y. Chen, X. Ji, and W. Xu, “Safegen: Mitigating sexually explicit content generation in text-toimage models,” in CCS, 2024.

[11] S. Kim, S. Jung, B. Kim, M. Choi, J. Shin, and J. Lee, “Towards safe self-distillation of internet-scale text-to-image diffusion models,” in ICML, 2023.

[12] R. Gandikota, J. Materzynska, J. Fiotto-Kaufman, and D. Bau, “Erasing concepts from diffusion models,” in ICCV, 2023.

[13] R. Gandikota, H. Orgad, Y. Belinkov, J. Materzynska, and D. Bau,´ “Unified concept editing in diffusion models,” in ICCV, 2024.

[14] D. Gao, S. Lu, W. Zhou, J. Chu, J. Zhang, M. Jia, B. Zhang, Z. Fan, and W. Zhang, “Eraseanything: Enabling concept erasure in rectified flow transformers,” in ICML, 2025.

[15] H. Gao, T. Pang, C. Du, T. Hu, Z. Deng, and M. Lin, “Metaunlearning on diffusion models: Preventing relearning unlearned concepts,” in ICCV, 2025.

[16] H. Huang, R. Wang, J. Zhao, W. Deng, D. Zhou, J. Luan, Y. Wu, and Y. Zhu, “Restoring initial noise sensitivity in text-to-image distillation through geometric alignment,” in ICML, 2026.

[17] R. Gandikota and D. Bau, “Distilling diversity and control in diffusion models,” in ICCV, 2026.

[18] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” in CoRR, 2020.

[19] Black Forest Labs, “Announcing Black Forest Labs,” https://bfl.ai/ announcing-black-forest-labs/, 2024.

[20] J. Ho and T. Salimans, “Classifier-free diffusion guidance,” in NeurIPS, 2021.

[21] Black Forest Labs, “FLUX.1 [schnell],” https://huggingface.co/ black-forest-labs/FLUX.1-schnell, 2024.

[22] Y. Zhang, X. Chen, J. Jia, Y. Zhang, C. Fan, J. Liu, M. Hong, K. Ding, and S. Liu, “Defensive unlearning with adversarial training for robust concept erasure in diffusion models,” in NeurIPS, 2024.

[23] C.-P. Huang, K.-P. Chang, C.-T. Tsai, Y.-H. Lai, F.-E. Yang, and Y.-C. F. Wang, “Receler: Reliable concept erasing of text-to-image diffusion models via lightweight erasers,” in ECCV, 2024.

[24] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” in ICLR, 2022.

[25] O. Li, Y. Wang, X. Hu, H. Jiang, Y. Hao, and F. Feng, “SPEED: Scalable, precise, and efficient concept erasure for diffusion models,” in ICLR, 2026.

[26] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015.

[27] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini,¨ Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” in ICML, 2024.

[28] K. Frans, D. Hafner, S. Levine, and P. Abbeel, “One step diffusion via shortcut models,” in ICLR, 2025.

[29] D. Chen, Z. Li, C. Chen, Y. Xie, X. Li, J. Ye, Y. Chen, and Y. Li, “Comprehensive evaluation and analysis for NSFW concept erasure in text-to-image diffusion models,” in NeurIPS, 2025.

[30] Y. Yang, Q. Liu, C. Brix, H. Zhang, and Y. Cao, “CertPHash: Towards certified perceptual hashing via robust training,” in Security, 2025.

[31] Z. Jiang, Y. Hu, Y. Yang, Y. Cao, and N. Z. Gong, “Jailbreaking safeguarded text-to-image models via large language models,” in EACL, 2026.

[32] P. Bedapudi, “Nudenet: Neural nets for nudity classification, detection and selective censoring,” https://github.com/notAI-tech/NudeNet, 2019.

[33] J. Rando, D. Paleka, D. Lindner, L. Heim, and F. Tramer, “Red-\` teaming the stable diffusion safety filter,” in NeurIPSW, 2022.

[34] Y. Qu, X. Shen, Y. Wu, M. Backes, S. Zannettou, and Y. Zhang, “Unsafebench: Benchmarking image safety classifiers on real-world and ai-generated images,” in CCS, 2025.

[35] X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li, “Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis,” in CoRR, 2023.

[36] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local nash equilibrium,” in NeurIPS, 2017.

[37] S. Jayasumana, S. Ramalingam, A. Veit, D. Glasner, A. Chakrabarti, and S. Kumar, “Rethinking fid: Towards a better evaluation metric for image generation,” in CVPR, 2024.

[38] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in ICML, 2021.

[39] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in ECCV, 2014.

[40] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, “Gradient surgery for multi-task learning,” in NeurIPS, 2020.

[41] G. Patel and Q. Qiu, “Learning to unlearn while retaining: Combating gradient conflicts in machine unlearning,” in ICCV, 2025.

[42] M. Gamba, H. Azizpour, and M. Bjorkman, “On the lipschitz constant¨ of deep networks and double descent,” in BMVC, 2023.

[43] A. Virmaux and K. Scaman, “Lipschitz regularity of deep neural networks: analysis and efficient estimation,” in NeurIPS, vol. 31, 2018.

[44] Y.-L. Tsai, C.-Y. Hsu, C. Xie, C.-H. Lin, J. Y. Chen, B. Li, P.-Y. Chen, C.-M. Yu, and C.-Y. Huang, “Ring-a-bell! How reliable are concept removal methods for diffusion models?” in ICLR, 2024.

[45] Y. Zhang, J. Jia, X. Chen, A. Chen, Y. Zhang, J. Liu, K. Ding, and S. Liu, “To generate or not? Safety-driven unlearned diffusion models are still easy to generate unsafe images... for now,” in ECCV, 2024.

[46] N. Jiang, Z. Fan, E. Kang, D. Gao, Y. Zhou, Y. Chang, Z. Zhu, Y. Jin, and W. Wu, “Erased, but not forgotten: Erased rectified flow transformers still remain unsafe under concept attack,” in CVPR, 2026.

[47] N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramer, B. Balle, D. Ippolito, and E. Wallace, “Extracting training data from diffusion models,” in Security, 2023.

[48] Y. Qu, X. Shen, X. He, M. Backes, S. Zannettou, and Y. Zhang, “Unsafe diffusion: On the generation of unsafe images and hateful memes from text-to-image models,” in CCS, 2023.

[49] LAION, “LAION-POP: 600k high-resolution image-text pairs from popular concepts,” https://laion.ai/blog/laion-pop/, 2023.

[50] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman et al., “Laion-5b: An open large-scale dataset for training next generation image-text models,” in NeurIPS, 2022.

[51] C. Zhang, T. Zhang, L. Wang, R. Chen, W. Li, and A. Liu, “T2I-RiskyPrompt: A benchmark for safety evaluation, attack, and defense on text-to-image model,” in AAAI, 2026.

[52] P. Schramowski, C. Tauchmann, and K. Kersting, “Can machines help us answering question 16 in datasheets, and in turn reflecting on inappropriate content?” in FAccT, 2022.

## Appendix A.

## Additional Methodological Details

## A.1. FLUX SCFM Training Details

To produce FLUX SCFM, we follow the SCFM training recipe of the original paper [8], distilling FLUX.1-dev into a few-step student. Matching that recipe, training optimizes a LoRA [24] adapter of rank 64 applied to all attention and feed-forward layers of the transformer blocks, using AdamW with a learning rate of $2 \times 1 0 ^ { - 5 }$

As the dataset, like the original paper, we use LAION-Pop [49], a filtered subset of the popular LAION-5B dataset [50]. To exclude unsafe images, we use the NSFW score provided with LAION data and filter the images to scores below 0.2. T2I model authors usually do not disclose detailed information about training data; we find 0.2 a conservative threshold fit for filtering NSFW content that keeps most data: over 90% of the images in the LAION-Pop dataset are below this threshold.

## A.2. NudeNet Classes

## WARNING: SOME OF THE TEXT BELOW IS EXPLICIT.

The NudeNet detector outputs a set of detected body-part boxes each with a confidence score. We use its default detection threshold 0.2 and label an image as unsafe if it contains at least one detection from the following fixed set of explicit nudity classes: BUTTOCKS\_EXPOSED; FEMALE\_BREAST\_EXPOSED; FEMALE\_GENITALIA\_EXPOSED; ANUS\_EXPOSED; and MALE\_GENITALIA\_EXPOSED.

## A.3. DICE Implementation Details

We instantiate DICE with rank-256 LoRA adapters on the transformer’s joint-attention projection matrices: the query, key, value, and output projections for both the image and text streams. For nudity erasure, we train on a concept prompt “a nude woman” and use “nude” as the token targeted by $L _ { \mathrm { a t t n } }$ . For SD3.5 Turbo, we use the prompt “a nude girl, full front body shot” as it generates more unsafe content for that model. We set the negative-guidance coefficient to $\eta = 1$ and optimize the DICE objective for 100 steps with AdamW at a learning rate of $1 0 ^ { - 4 }$ , using bfloat16 mixed precision and gradient checkpointing. The resulting optimization is lightweight: DICE runs take roughly five minutes on a single H100 GPU.

## Appendix B. Additional Results

## B.1. Full Safety and Quality Results

Tab. 9 reports the complete safety and quality numbers for every base model and erasure method. Each model is shown at its deployment configuration; since these differ across families, safety numbers are only directly comparable within a family.

## B.2. Robustness of the Lipschitz Estimator to Its Hyperparameters

The local-Lipschitz estimator of §4.2 has two free knobs: the number of random perturbation directions k averaged per sample, and the relative perturbation radius $\epsilon ( \mathrm { i } . \mathrm { e } . \| \bar { \delta } \| / \| z \| )$ for the input-space estimate, and $\| \Delta _ { \ell } \| / \| \theta _ { \ell } \|$ per tensor for the parameter-space estimate). All runs use 500 LAION-pop images with the empty prompt.

To confirm the headline few-step-to-full-step-reference gap is not an artifact of these choices, we vary one knob at a time and report the cross-family paired Lipschitz ratio $\mathrm { L i p } _ { \mathrm { S c h n e l l } } / \mathrm { L i p } _ { \mathrm { D e v } }$ (median over the 500 samples). While the absolute magnitude scales strongly with ϵ, Tab. 10 shows the ratio stays in a narrow band, compressing only mildly at the largest radius. We sweep only ϵ for the parameterspace (θ) estimate, since the input-space k sweep already shows k=8 is saturated.

## B.3. Direction of the ESD Gradient Update

We expand the claim from §4.3 that the ESD update pushes the model against the concept direction. Recall the ESD loss from Eq. 2:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { E S D } } = \Big \| v _ { \boldsymbol { \theta } } ( z , c , t ) - \Big [ v _ { \boldsymbol { \theta } ^ { \star } } ( z , \infty , t ) } \\ & { \qquad - \left. \eta \big ( v _ { \boldsymbol { \theta } ^ { \star } } ( z , c , t ) - v _ { \boldsymbol { \theta } ^ { \star } } ( z , \infty , t ) \big ) \right] \Big \| _ { 2 } ^ { 2 } } \end{array}
$$

TABLE 9. FULL SAFETY AND QUALITY RESULTS ACROSS ALL BASE MODELS AND ERASURE METHODS. SAFETY METRICS (NUDENET, SD-CHECKER; MEAN AND THE UNSAFE@K REPEATED-SAMPLING VARIANTS, WHICH FLAG A PROMPT UNSAFE IF ANY OF ITS k = 5 SEEDS IS FLAGGED) ARE LOWER-IS-SAFER; THE QUALITY METRIC (HPSV2) IS HIGHER-IS-BETTER. WITHIN EACH MODEL, THE BEST VALUE AMONG THE ERASURE METHODS (ESD, ERASEANYTHING, UCE) IS IN BOLD (LOWEST NUDENET/SD-CHECKER, HIGHEST HPSV2); THE UNDEFENDED BASELINE IS SHOWN FOR REFERENCE AND EXCLUDED FROM BOLDING.
<table><tr><td>Family</td><td>Variant</td><td>Config</td><td>Method</td><td>NudeNet ↓</td><td>NudeNet@5 ↓</td><td>SD ↓</td><td>SD@5 ↓</td><td>HPSv2 ↑</td></tr><tr><td rowspan="8">SD3.5</td><td rowspan="5">SD3.5 Large (teacher)</td><td rowspan="5">28-step, CFG 3.5</td><td>Undefended</td><td>0.2432</td><td>0.5240</td><td>0.5416</td><td>0.8150</td><td>0.2905</td></tr><tr><td>ESD</td><td>0.0822</td><td>0.2650</td><td>0.3888</td><td>0.7190</td><td>0.2877</td></tr><tr><td>EraseAnything</td><td>0.1480</td><td>0.4190</td><td>0.4710</td><td>0.7940</td><td>0.2821</td></tr><tr><td>UCE</td><td>0.1850</td><td>0.4610</td><td>0.4648</td><td>0.7660</td><td>0.2894</td></tr><tr><td>SPEED</td><td>0.2236</td><td>0.5050</td><td>0.5136</td><td>0.7970</td><td>0.2907</td></tr><tr><td rowspan="5">SD3.5 Turbo (student) 4-step</td><td rowspan="5"></td><td>Undefended</td><td>0.3232</td><td>0.5870</td><td>0.6646</td><td>0.8670</td><td>0.2874</td></tr><tr><td>ESD</td><td>0.1504</td><td>0.3700</td><td>0.5692</td><td>0.8210</td><td>0.2845</td></tr><tr><td>EraseAnything</td><td>0.2992 0.2254</td><td>0.5720</td><td>0.6626 0.5806</td><td>0.8700 0.8260</td><td>0.2754</td></tr><tr><td>UCE SPEED</td><td>0.2894</td><td>0.4810 0.5460</td><td>0.6362</td><td>0.8480</td><td>0.2865 0.2874</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="9">FLUX</td><td rowspan="5">FLUX Dev (full-step)</td><td rowspan="5">50-step, CFG 3.5</td><td>Undefended</td><td>0.2562</td><td>0.5570</td><td>0.6108</td><td>0.8570</td><td>0.2938</td></tr><tr><td>ESD</td><td>0.0572</td><td>0.2120</td><td>0.3254</td><td>0.6260</td><td></td><td>0.2947</td></tr><tr><td>EraseAnything</td><td>0.0816</td><td>0.2550</td><td>0.4554</td><td>0.7630</td><td></td><td>0.2936</td></tr><tr><td>UCE</td><td>0.2108</td><td>0.4840</td><td>0.5690</td><td></td><td>0.8300</td><td>0.2939</td></tr><tr><td>SPEED</td><td>0.2280</td><td>0.5120</td><td>0.5956</td><td></td><td>0.8400</td><td>0.2938</td></tr><tr><td rowspan="5">FLUX Schnell (few-step) 4-step</td><td rowspan="5"></td><td>Undefended</td><td>0.2348</td><td>0.5190</td><td>0.5622</td><td>0.8020</td><td></td><td>0.2909</td></tr><tr><td>ESD</td><td>0.1788</td><td>0.4360</td><td>0.4706</td><td></td><td>0.7700</td><td>0.2796</td></tr><tr><td>EraseAnything</td><td>0.1608</td><td>0.4150</td><td>0.5258</td><td></td><td>0.8120</td><td>0.2889</td></tr><tr><td>UCE</td><td>0.1824</td><td>0.4540</td><td>0.4966</td><td></td><td>0.7690</td><td>0.2908</td></tr><tr><td>SPEED</td><td>0.2102</td><td>0.4920</td><td>0.5358</td><td></td><td>0.7990</td><td>0.2909</td></tr><tr><td rowspan="5">FLUX SCFM (student)</td><td rowspan="5">4-step, CFG 5</td><td>Undefended</td><td>0.3282</td><td>0.6390</td><td>0.7074</td><td></td><td>0.9020</td><td>0.2781</td></tr><tr><td>ESD</td><td>0.0842</td><td></td><td>0.2590</td><td>0.4626</td><td>0.7400</td><td>0.2784</td></tr><tr><td>EraseAnything</td><td>0.1628</td><td>0.4320</td><td>0.5614</td><td></td><td>0.8450</td><td>0.2787</td></tr><tr><td>UCE</td><td>0.2780</td><td>0.5850</td><td>0.6630</td><td></td><td>0.8770</td><td>0.2781</td></tr><tr><td>SPEED</td><td>0.3084</td><td>0.6100</td><td>0.6848</td><td></td><td>0.8960</td><td>0.2780</td></tr></table>

TABLE 10. CROSS-FAMILY PAIRED LIPSCHITZ RATIO $\mathrm { L i p } _ { \mathrm { S c H N E L L } } / \mathrm { L i p } _ { \mathrm { D E V } }$ UNDER VARIATION OF THE NUMBER OF PERTURBATION DIRECTIONS k AND THE RELATIVE RADIUS ϵ, FOR THE INPUT-SPACE (z) AND PARAMETER-SPACE (θ) ESTIMATES. MAIN-TEXT OPERATING POINTS ARE IN BOLD. THE RATIO IS MOSTLY INSENSITIVE TO k AND ϵ.
<table><tr><td>Space</td><td>Sweep</td><td>k</td><td>€</td><td>Paired  $\mathsf { p } 5 0$ </td></tr><tr><td rowspan="4">z</td><td rowspan="2"> $k ~ ( \epsilon { = } 1 0 ^ { - 3 } )$ </td><td>4</td><td> $1 0 ^ { - 3 }$ </td><td>4.70</td></tr><tr><td>8</td><td> $\mathbf { 1 0 ^ { - 3 } }$ </td><td>4.76</td></tr><tr><td rowspan="2"></td><td>16</td><td> $1 0 ^ { - 3 }$ </td><td>4.79</td></tr><tr><td>8</td><td> $1 0 ^ { - 4 }$ </td><td>4.88</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>8 8</td><td> $\mathbf { 1 0 ^ { - 3 } }$   $1 0 ^ { - 2 }$ </td><td>4.76 3.72</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="2">θ</td><td rowspan="2"> $\epsilon \ ( k { = } 8 )$ </td><td>8</td><td> $1 0 ^ { - 3 }$   $\mathbf { 1 0 ^ { - 2 } }$ </td><td>4.45</td></tr><tr><td>8 8</td><td> $3 \times 1 0 ^ { - 2 }$ </td><td>4.44 4.09</td></tr></table>

Near initialization, where $\theta \approx \theta ^ { \star }$ , we define the concept direction $\delta = v _ { \theta } ( z , c , t ) - v _ { \theta } ( z , \mathcal { D } , t )$ and, substituting $v _ { \theta ^ { \star } }$ ≈ v<sub>θ</sub>, obtain:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { E S D } } = \Big | \Big | v _ { \theta } ( z , c , t ) - \Big [ v _ { \theta } ( z , c , t ) - ( \eta + 1 ) \delta \Big ] \Big | \Big | _ { 2 } ^ { 2 } } \\ & { \qquad = \Big | \Big | ( \eta + 1 ) \delta \Big | \Big | _ { 2 } ^ { 2 } } \end{array}
$$

so, near initialization, the ESD gradient pushes the model against the concept direction, towards −δ. This approximation is relevant because erasure is typically short, usually only ∼100–200 iterations, so the edited model remains close to the frozen pretrained copy for most of optimization. The update magnitude scales with ∥δ∥, which is exactly the quantity we find inflated in $\ S 4 . 3$

## B.4. Per-Timestep Concept-Direction Statistics

Tab. 11 reports the raw velocity-prediction norms $\lVert \boldsymbol { v } ( z , c ) \rVert$ , which stay close across all models; this confirms that the $\left| \left| \delta \right| \right|$ inflation reflects poorer cancellation between the conditional and unconditional branches rather than larger raw predictions.

## B.5. Erasing at Unit Guidance for Guidance-Conditioned Models

§4.3 notes that, because baked-in extra guidance inflates the concept direction δ, a natural remedy is to run the erasure itself at guidance scale $\omega = 1$ , where CFG is disabled. This knob only exists for models that take the guidance scale as an explicit input (FLUX Dev, FLUX SCFM); fewstep distilled checkpoints such as SD3.5 Turbo and FLUX Schnell bake ω in at a fixed strength and offer no such control.

We test the remedy with EraseAnything [14] nudity erasure, sweeping only the training-time guidance $( \omega = 1 \ \mathrm { v s }$ the method’s default) and evaluating each model at its fixed deployment CFG (FLUX Dev at $\omega = 3 . 5 $ , 50-step; FLUX

TABLE 11. CONDITIONAL AND UNCONDITIONAL VELOCITY-PREDICTION NORMS $\left| \left| \boldsymbol { v } ( z , c ) \right| \right|$ AND $\| v ( z , \emptyset ) \|$ PER TIMESTEP BIN FOR EVERY MODEL. THE TWO NORMS ARE NEARLY IDENTICAL WITHIN EACH MODEL AND COMPARABLE ACROSS ALL MODELS.
<table><tr><td></td><td></td><td colspan="6"> $\lVert \boldsymbol { v } ( z , c ) \rVert$ </td><td colspan="6"> $\| v ( z , \emptyset ) \|$ </td></tr><tr><td>t-bin</td><td>t range</td><td>Large</td><td>Turbo</td><td> $\mathrm { D e v } _ { 1 . 0 }$ </td><td> $\mathrm { D e v } _ { 3 . 5 }$ </td><td>Schnell</td><td> $\mathrm { S C F M _ { 5 . 0 } }$ </td><td> $\mathrm { L a r g e }$ </td><td>Turbo</td><td> $\mathrm { D e v } _ { 1 . 0 }$ </td><td> $\mathrm { D e v } _ { 3 . 5 }$ </td><td>Schnell</td><td> $\mathrm { S C F M _ { 5 . 0 } }$ </td></tr><tr><td>0</td><td>[0, 100)</td><td>681.9</td><td>701.7</td><td>729.2</td><td>721.9</td><td>661.2</td><td>707.5</td><td>681.5</td><td>700.3</td><td>728.0</td><td>721.3</td><td>665.6</td><td>707.2</td></tr><tr><td>1</td><td>[100,200)</td><td>695.9</td><td>722.7</td><td>669.9</td><td>675.5</td><td>773.9</td><td>669.8</td><td>695.6</td><td>721.8</td><td>670.5</td><td>675.5</td><td>776.6</td><td>669.9</td></tr><tr><td>2</td><td>[200,300)</td><td>726.8</td><td>747.9</td><td>684.3</td><td>693.6</td><td>704.6</td><td>693.2</td><td>726.2</td><td>746.8</td><td>685.2</td><td>693.7</td><td>704.9</td><td>693.4</td></tr><tr><td>3</td><td>[300, 400)</td><td>750.2</td><td>765.3</td><td>701.8</td><td>712.2</td><td>700.2</td><td>710.8</td><td>749.5</td><td>764.7</td><td>702.8</td><td>712.9</td><td>698.9</td><td>710.8</td></tr><tr><td>4</td><td>[400,500)</td><td>758.5</td><td>772.7</td><td>708.5</td><td>722.6</td><td>657.0</td><td></td><td>717.4758.3</td><td>772.3</td><td>710.0</td><td>723.1</td><td>655.0</td><td>717.9</td></tr><tr><td>5</td><td>[500,600)</td><td>769.7</td><td>782.4</td><td>720.6</td><td>735.4</td><td>651.2</td><td>727.9</td><td>769.3</td><td>781.9</td><td>721.9</td><td>736.2</td><td>649.2</td><td>728.1</td></tr><tr><td>6</td><td>[600, 700)</td><td>766.1</td><td>781.9</td><td>722.7</td><td>741.6</td><td>676.1</td><td>734.2</td><td>766.0</td><td>781.0</td><td>724.0</td><td>742.0</td><td>673.7</td><td>734.2</td></tr><tr><td>7</td><td>[700,800)</td><td>761.9</td><td>781.0</td><td>721.3</td><td>742.9</td><td>650.8</td><td>736.0</td><td>761.8</td><td>780.0</td><td>722.3</td><td>742.9</td><td>647.5</td><td>736.2</td></tr><tr><td>8</td><td>[800,900)</td><td>767.9</td><td>793.6</td><td>727.7</td><td>755.3</td><td>668.5</td><td>762.1</td><td>767.8</td><td>793.5</td><td>728.0</td><td>754.2</td><td>666.5</td><td>761.3</td></tr><tr><td>9</td><td>[900, 1000]</td><td>746.9</td><td>794.9</td><td>690.5</td><td>732.0</td><td>741.9</td><td>763.8</td><td>747.2</td><td>794.7</td><td>688.5</td><td>706.2</td><td>726.4</td><td>725.1</td></tr></table>

TABLE 12. ERASEANYTHING NUDITY ERASURE: TRAINING $\mathbf { A T } \omega = 1$ VS. THE METHOD’S DEFAULT GUIDANCE, EVALUATED AT EACH MODEL’S DEPLOYMENT CFG. LOWERING THE TRAINING GUIDANCE TO $\omega = 1$ CONSISTENTLY WEAKENS ERASURE AT NO QUALITY BENEFIT.
<table><tr><td>Model</td><td>Train ω</td><td> $\mathrm { N u d e N e t \downarrow }$ </td><td> $\mathrm { S D \downarrow }$ </td><td>HPSv2 ↑</td></tr><tr><td rowspan="2">FLUX Dev</td><td>3.5</td><td>0.0816</td><td>0.4554</td><td>0.2936</td></tr><tr><td>1.0</td><td>0.1376</td><td>0.5152</td><td>0.2943</td></tr><tr><td rowspan="2">FLUX SCFM</td><td>5</td><td>0.1628</td><td>0.5614</td><td>0.2787</td></tr><tr><td>1.0</td><td>0.2634</td><td>0.6800</td><td>0.2788</td></tr></table>

TABLE 13. UNSAFE GENERATION RATES AS A FUNCTION OFINFERENCE-STEP COUNT OVER THE VIOLENCE SUBSET OFT2I-RISKYPROMPT.
<table><tr><td>Model</td><td>Defended</td><td>Steps</td><td> $\boldsymbol { \mathrm { Q 1 6 \downarrow } }$ </td><td> $\mathrm { Q } 1 6 @ 5 \downarrow$ </td></tr><tr><td rowspan="2">SD3.5 Large</td><td rowspan="2">x</td><td>28</td><td>0.6230</td><td>0.7974</td></tr><tr><td>16</td><td>0.6416</td><td>0.8131</td></tr><tr><td rowspan="2">FLUX Dev</td><td rowspan="2">x</td><td>50</td><td>0.5458</td><td>0.7690</td></tr><tr><td>16</td><td>0.5554</td><td>0.7640</td></tr></table>

SCFM at $\omega = 5 , 4 \mathrm { - s t e p } )$ , so the comparison isolates the training guidance with inference held constant. As Tab. 12 shows, erasing at $\omega = 1$ does not help: it weakens erasure on both models $( \mathrm { a } \sim 6 0 { - } 7 0 \%$ relative rise in NudeNet rate, with SD-Checker moving the same direction) while leaving quality essentially unchanged (HPSv2 moves by less than 0.001).

## B.6. Violence Evaluation

For evaluation of violence content, we use the violence subcategory of the T2I-RiskyPrompt dataset [51]. We employ the CLIP-based Q16 safety checker for violence detection [51], [52], with the standard threshold of 0.9 for classification.

Tab. 13 presents how unsafe generation rates of violence content vary as a function of inference steps. Consistent with nudity (§3.3), reducing the number of steps increases unsafe content generation rates.

Tab. 14 compares DICE to other erasure methods on violence erasure. Overall, DICE provides the strongest safety results across all models.

TABLE 14. VIOLENCE SAFETY AND 4-STEP HPSV2 QUALITY ACROSSTHE FEW-STEP MODEL FAMILIES, MEASURED ON THE VIOLENCESUBSET OF T2I-RISKYPROMPT. LOWER Q16 VALUES INDICATE SAFERMODELS.
<table><tr><td>Model</td><td>Method</td><td>Q16 ↓</td><td>Q16@5 ↓</td><td>HPSv2 ↑</td></tr><tr><td rowspan="7">SD3.5 Turbo</td><td>Undefended</td><td>0.5642</td><td>0.7484</td><td>0.2874</td></tr><tr><td>EraseAnything</td><td>0.4324</td><td>0.6638</td><td>0.2632</td></tr><tr><td>ESD</td><td>0.5150</td><td>0.7164</td><td>0.2799</td></tr><tr><td>UCE</td><td>0.5285</td><td>0.7143</td><td>0.2866</td></tr><tr><td>SPEED</td><td>0.5467</td><td>0.7342</td><td>0.2873</td></tr><tr><td>DICE</td><td>0.2222</td><td>0.4819</td><td>0.2746</td></tr><tr><td>Undefended</td><td>0.5967</td><td>0.8003</td><td>0.2909</td></tr><tr><td rowspan="5">FLUX Schnell</td><td>EraseAnything</td><td>0.3557</td><td>0.6041</td><td>0.2880</td></tr><tr><td>ESD</td><td>0.5373</td><td>0.7534</td><td>0.2871</td></tr><tr><td>UCE</td><td>0.5518</td><td>0.7711</td><td>0.2905</td></tr><tr><td>SPEED</td><td>0.5734</td><td>0.7811</td><td>0.2907</td></tr><tr><td>DICE</td><td>0.3173</td><td>0.5800</td><td>0.2872</td></tr><tr><td rowspan="6">FLUX SCFM</td><td>Undefended</td><td>0.5972</td><td>0.7889</td><td>0.2781</td></tr><tr><td>EraseAnything</td><td>0.5053</td><td>0.7335</td><td>0.2756</td></tr><tr><td>ESD</td><td>0.5292</td><td>0.7413</td><td>0.2744</td></tr><tr><td>UCE</td><td>0.5528</td><td>0.7683</td><td>0.2775</td></tr><tr><td>SPEED</td><td>0.5730</td><td>0.7747</td><td>0.2778</td></tr><tr><td>DICE</td><td>0.3780</td><td>0.6709</td><td>0.2760</td></tr></table>

## B.7. DICE Results on the I2P Benchmark

To complement the NSFW-1k results in the main text, we also evaluate on the widely-used I2P benchmark [9]. Because I2P prompts are considerably less explicit than NSFW-1k (§3.1 [10]), the undefended unsafe rates are already low, but the relative ordering of methods is preserved. Tab. 15 reports safety and quality metrics for the SD3.5 and FLUX families.

## B.8. FID and CLIP-Similarity on COCO-30k

FID and CLIP-Sim on COCO-30k are presented in Tab. 15; all variants generate exactly 30000 images at their 4-step deployment configuration with a single seed per image. DICE preserves image quality: its FID is the lowest of all variants for FLUX Schnell (16.30 vs. 18.60 undefended) and within 0.05 of the best for SD3.5 Turbo, while CLIP-Sim stays within ∼0.006 of the undefended baseline across all three families.

TABLE 15. I2P SAFETY AND QUALITY, AND COCO-30K FID AND CLIP-SIMILARITY, ACROSS THE GUIDANCE-DISTILLED FEW-STEP FAMILIES. LOWER NUDENET AND SD-CHECKER VALUES (MEAN AND THE @5 REPEATED-SAMPLING VARIANT) ON I2P INDICATE SAFER MODELS; HIGHER HPSV2 VALUES INDICATE BETTER GENERATION QUALITY, REPORTED AT ONE-STEP AND FOUR-STEP INFERENCE. FID (LOWER IS BETTER) AND CLIP-SIM (HIGHER IS BETTER) ARE COMPUTED ON COCO-30K. THE BEST VALUE IN EACH COLUMN AMONG THE ERASURE METHODS, WITHIN EACH MODEL, IS IN BOLD; THE UNDEFENDED BASELINE IS SHOWN FOR REFERENCE AND EXCLUDED FROM BOLDING.
<table><tr><td>Model</td><td>Method</td><td>NudeNet↓</td><td>NudeNet@5 ↓</td><td>SD-Checker ↓</td><td>SD-Checker@5 ↓</td><td> $\mathrm { H P S v } 2 _ { \mathrm { l - s t e p } } \uparrow$ </td><td> $\mathrm { H P S v } 2 _ { 4 \mathrm { - s t e p } } \uparrow$ </td><td>FID ↓</td><td>CLIP-Sim ↑</td></tr><tr><td rowspan="6">SD3.5 Turbo</td><td>Undefended</td><td>0.0445</td><td>0.1063</td><td>0.1395</td><td>0.2785</td><td>0.2551</td><td>0.2874</td><td>23.99</td><td>0.2676</td></tr><tr><td>EraseAnything</td><td>0.0473</td><td>0.1157</td><td>0.1643</td><td>0.3479</td><td>0.2453</td><td>0.2754</td><td>24.29</td><td>0.2661</td></tr><tr><td>ESD</td><td>0.0181</td><td>0.0604</td><td>0.1140</td><td>0.2450</td><td>0.2467</td><td>0.2845</td><td>21.58</td><td>0.2671</td></tr><tr><td>UCE</td><td>0.0344</td><td>0.0880</td><td>0.1289</td><td>0.2656</td><td>0.2547</td><td>0.2865</td><td>24.13</td><td>0.2659</td></tr><tr><td>SPEED</td><td>0.0386</td><td>0.0967</td><td>0.1354</td><td>0.2709</td><td>0.2551</td><td>0.2874</td><td>23.88</td><td>0.2675</td></tr><tr><td>DICE</td><td>0.0115</td><td>0.0438</td><td>0.1204</td><td>0.2745</td><td>0.2425</td><td>0.2830</td><td>21.63</td><td>0.2652</td></tr><tr><td rowspan="6">FLUX Schnell</td><td>Undefended</td><td>0.0432</td><td>0.1131</td><td>0.1281</td><td>0.2958</td><td>0.2886</td><td>0.2909</td><td>18.60</td><td>0.2659</td></tr><tr><td>EraseAnything</td><td>0.0401</td><td>0.1093</td><td>0.1350</td><td>0.3181</td><td>0.2859</td><td>0.2889</td><td>18.99</td><td>0.2645</td></tr><tr><td>ESD</td><td>0.0269</td><td>0.0731</td><td>0.0977</td><td>0.2524</td><td>0.2704</td><td>0.2796</td><td>19.21</td><td>0.2631</td></tr><tr><td>UCE</td><td>0.0332</td><td>0.0914</td><td>0.1163</td><td>0.2796</td><td>0.2887</td><td>0.2908</td><td>18.37</td><td>0.2662</td></tr><tr><td>SPEED</td><td>0.0345</td><td>0.0950</td><td>0.1178</td><td>0.2819</td><td>0.2887</td><td>0.2909</td><td>18.49</td><td>0.2662</td></tr><tr><td>DICE</td><td>0.0134</td><td>0.0449</td><td>0.0845</td><td>0.2337</td><td>0.2838</td><td>0.2877</td><td>16.30</td><td>0.2627</td></tr><tr><td rowspan="6">FLUX SCFM</td><td>Undefended</td><td>0.0632</td><td>0.1629</td><td>0.1618</td><td>0.3547</td><td>0.2202</td><td>0.2781</td><td>26.10</td><td>0.2625</td></tr><tr><td>EraseAnything</td><td>0.0393</td><td>0.1193</td><td>0.1399</td><td>0.3226</td><td>0.2195</td><td>0.2787</td><td>26.63</td><td>0.2596</td></tr><tr><td>ESD</td><td>0.0175</td><td>0.0572</td><td>0.1011</td><td>0.2537</td><td>0.2184</td><td>0.2784</td><td>20.70</td><td>0.2614</td></tr><tr><td>UCE</td><td>0.0506</td><td>0.1359</td><td>0.1481</td><td>0.3336</td><td>0.2203</td><td>0.2781</td><td>25.78</td><td>0.2626</td></tr><tr><td>SPEED</td><td>0.0538</td><td>0.1429</td><td>0.1539</td><td>0.3417</td><td>0.2203</td><td>0.2780</td><td>25.85</td><td>0.2626</td></tr><tr><td>DICE</td><td>0.0184</td><td>0.0636</td><td>0.1102</td><td>0.2900</td><td>0.2194</td><td>0.2774</td><td>25.64</td><td>0.2572</td></tr></table>

![](images/2460e517d210ed7aac7acab43ef35887be59795f931701e3be960e2f7d7c20dd.jpg)  
Figure 8. Qualitative comparison of DICE against existing erasure methods (ESD, EraseAnything, UCE) on FLUX Schnell, using shared prompts and seeds. Black rectangles are added by the authors for censorship.

## B.9. Qualitative DICE Results

Fig. 8 extends the qualitative comparison of §5.3 with additional examples that place DICE alongside the prior erasure methods (ESD, EraseAnything, UCE) on the same prompts and seeds. DICE removes the targeted concept across the few-step families while leaving the overall composition and benign content largely intact, whereas the baselines either leave residual unsafe content or visibly distort the image.

## Ethics Considerations

This work seeks to improve our understanding of unsafe image generation in few-step T2I models, so that more reliable safety mechanisms can be developed. To this end, it is necessary to generate unsafe content in controlled settings by feeding unsafe prompts to T2I models. The unsafe prompts used in this work are taken from existing datasets that are commonly used by the research community for safety evaluation, rather than collected from users or targeted at private individuals. Moreover, we do not feed unsafe prompts to commercial models to avoid creating moderation overhead to the models’ owners. Instead, we focus our research on open-weights models.

Our work has potential dual-use implications. In particular, showing that current few-step or accelerated T2I models are more likely to produce unsafe images could inform adversaries about weaknesses in existing safety mechanisms. However, our goal is to raise awareness of this safety gap and to support safer deployment of few-step models. To this end, we also propose and evaluate an effective mitigation method. We release our code and defended model weights to support reproducibility and follow-up safety research. Unsafe generated images are used only for controlled evaluation. We do not intend to release generated unsafe images as a dataset, and qualitative examples in the paper are filtered and limited to the minimum necessary to illustrate our findings.

Concept-erasure methods are not bulletproof; even when a model is defended, it is not guaranteed to only produce safe content against all prompts. Therefore, we encourage developers, providers, and users of T2I models to treat concept-erasure methods as one component of a broader safety pipeline, rather than as a safety guarantee.

Our work does not involve human subjects, user studies, private user data, personally identifiable information, or experiments on animals.

## Appendix C. Meta-Review

The following meta-review was prepared by the program committee for the 2027 IEEE Symposium on Security and Privacy (S&P) as part of the review process as detailed in the call for papers.

## C.1. Summary

This paper studies the interaction between few-step distillation of text-to-image models and safety. It shows that distilled few-step students are often less safe than their full step teachers, that reducing the inference step count itself increases unsafe-generation rates, and that several existing concept-erasure methods are substantially less effective when applied to few-step models. To explain this degrada tion, the paper presents a mechanistic analysis suggesting that aggressively distilled models are more sensitive to latent and parameter perturbations, and that guidance bakein distorts the conditional–unconditional direction used by ESD-style erasure. Based on these observations, the authors propose DICE, a post-distillation concept-erasure method for few-step flow-matching models that combines straightpath re-noising with a self-distilled generation-preservation objective anchored to the frozen student. Experiments on SD3.5 Turbo, FLUX Schnell, and a controlled FLUX-SCFM student show improved nudity erasure while largely preserving benign image quality.

## C.2. Scientific Contributions

• Identifies an Impactful Vulnerability

Provides a Valuable Step Forward in an Established Field

## C.3. Reasons for Acceptance

1) The paper identifies a timely and practically important problem. The finding that acceleration is not safety-neutral is valuable precisely because accelerated checkpoints are increasingly deployed as drop-in replacements for slower models, without a dedicated post-distillation safety audit and on the assumption that safety properties transfer from the teacher.

2) The mechanistic analysis provides evidence beyond simple benchmarking, examining model sensitivity and concept-direction degradation. The diagnosis translates directly into the design of DICE, producing a narrative in which the method follows from the analysis rather than being introduced without rationale.

3) The empirical evaluation is broad and well organized, covering multiple few-step models, several distillation paradigms, two safety detectors, repeated-seed metrics, benign-quality metrics, and ablations. The controlled FLUX Dev to FLUX-SCFM teacher/student pair is particularly useful, since it avoids relying entirely on opaque released checkpoints and makes the safety-degradation result more convincing than comparisons between unrelated models.

4) DICE is practically useful: it is post-distillation, does not require retraining the full model, and substantially improves the safety–quality trade-off on the evaluated models.

## C.4. Noteworthy Concerns

1) DICE evalautes against Ring-A-Bell and Unlearn-DiffAtk, and DICE is the least vulnerable erasure method in every setting tested. However, attack success remains substantial. For example, UnlearnDiffAtk succeeds on about 45% of prompts against DICE-defended FLUX Schnell, compared with about 88% against the undefended model. The weight-space attack ReFlux is discussed only qualitatively. DICE should therefore be read as raising the bar for adversaries, not as a robust defense, which the authors acknowledge. Broader adaptive red-teaming remains important future work.

2) The evaluation includes foundational erasure methods (ESD, UCE), the closed-form SPEED, and EraseAnything. As flow-matching-native erasure methods mature, comparing DICE against them would further strengthen the claims.