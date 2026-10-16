# ACEdit: Stable Multi-Turn Image Editing via Asymmetric Conditioning

Sara Dorfman<sup>\*</sup> Nir Goren<sup>\*</sup>

Gal Barak<sup>\*</sup>

Daniel Cohen-Or

Or Patashnik

Tel Aviv University

Flux.2 Klein

Flux.2 Klein + Ours

Original

![](images/20c4fc19fcf5a636017d6a43fd2b2d54d556689e1dfaf95cddca0617be2ddc90.jpg)  
Edit #3  
Edit #6  
Edit #9  
Edit #15  
Edit #12

Figure 1. Stable Multi-Turn Image Editing. We show selected turns from a 15-step sequence. Each of the five edits is followed by two no-op instructions (“keep the image as is”). While the original FLUX.2 Klein gradually loses detail and develops a strong color shift, fine-tuning it with our training scheme yields a more stable editing process, better preserving the scene and previously edited content while faithfully following the requested changes. The zoom-ins highlight the loss of fine fur texture and the emergence of harsh, unnatural lines in the baseline, while our model preserves the fur’s soft texture and natural appearance.

## Abstract

Image editing models produce impressive results for indi vidual edits, but their quality often deteriorates over multiple editing turns. Each editing turn introduces small distribution shifts that accumulate over successive edits, creating a gap between the input distributions encountered during the editor’s original training and multi-turn inference. We introduce a training scheme that makes multi-turn image editing stable under this accumulating distribution shift. A key is to force the editor to operate on its own outputs from previous editing turns during training, bringing the principle of Self-Forcing to image editing. The main challenge is that self-generated editing trajectories deviatefrom groundtruth trajectories, making direct supervision unreliable. We address this challenge through a distillation objective that leverages a teacher image editor to provide a reliable training signal when the model operates on its own previous outputs. We introduce an asymmetric conditioning scheme: while the student performs edits sequentially on its own out-

puts, the teacher remains anchored to the original clean image and receives a compound editing instruction. Experiments show substantially reduced degradation over long multi-turn sequences while maintaining editing quality and instruction adherence.

## 1. Introduction

Image editing with modern generative models has reached an impressive level of quality, enabling complex modifications while faithfully preserving the appearance and content of the input image [6, 24, 26, 44, 49]. Yet, practical editing is rarely a single operation. Users often refine an image through a sequence of edits, where the output of one edit becomes the input to the next. This setting is commonly referred to as multi-turn image editing. With current image editing models, each editing turn introduces small distribution shifts that accumulate as the model repeatedly operates on its own outputs. These shifts create a growing gap between the input distributions encountered during the editor’s original training and multi-turn inference. This gap leads to accumulating artifacts, loss of fine details, and unintended changes in appearance (see Figure 1). Effective image editing therefore requires not only high-quality individual edits, but also stability across successive editing turns.

Several approaches have been proposed to mitigate degradation in multi-turn image editing. Inference-time methods target specific sources of degradation [27, 38, 47, 62], but may incur computational overhead and are limited by the complexity of distribution shifts in multi-turn editing. Other approaches train models on multi-turn editing sequences [16, 39, 52, 61]. Training on ground-truth sequences leaves a gap between training and inference, while training on self-generated sequences poses supervision challenges, leading to inconsistent targets, restrictions on training sequences, or objectives that do not explicitly address distribution shifts.

We present a training scheme for multi-turn image editing that maintains stability under accumulating distribution shifts. Our scheme is inspired by autoregressive video generation, where models similarly operate on their own outputs and face accumulating distribution shifts. Following the principle of Self-Forcing [17], we train the editor on its own outputs from previous editing turns, allowing it to learn from the distribution shifts encountered during multi-turn inference. However, self-generated editing trajectories inevitably deviate from ground-truth trajectories, making direct supervision with ground-truth images unreliable.

To address this challenge, we employ score distillation in a teacher-student framework to train the editor on its selfgenerated trajectories without requiring ground-truth targets for its intermediate outputs [37, 48, 57]. We introduce an asymmetric conditioning scheme for the student and teacher. The student’s input may already deviate significantly from the teacher’s training distribution, making it unsuitable for reliable teacher guidance. We therefore condition the teacher on the original image and a compound instruction combining all edits up to the current turn, while the student performs edits sequentially on its own outputs. This provides a training signal that reflects the intended edits without inheriting the errors accumulated along the student’s trajectory.

We evaluate our method against existing multi-turn image editing approaches and demonstrate significant qualitative and quantitative improvements in preserving editing capabilities while mitigating accumulated distribution drift. Our training is efficient and introduces no inference-time overhead, while making no assumptions about the nature of distribution shifts or restrictions on editing sequences. Through ablation studies, we further demonstrate the importance of our asymmetric conditioning scheme for stable multi-turn editing.

## 2. Related Work

Text-guided image editing. Early text-driven manipulation largely steered pre-trained GAN latent spaces via vision-language models [11, 36]. The advent of diffusion models broadened semantic coverage and synthesized complex scenes more flexibly. Subsequent approaches improved source-image preservation and fine-grained control through noise perturbation and trajectory inversion [5, 10, 19, 33, 34, 46], alongside spatial blending and crossattention steering [1, 3, 7, 14, 45]. Recent efforts extend inference-time control to continuous-time flow models through structural layer control [4], flow inversion [20, 40], inversion-free sampling trajectories [23], and textembedding control for continuous editing [18, 21]. Complementing inference-time methods, instruction-tuned architectures learn a direct mapping using synthetic pairs [6], human annotations [59], and multi-task supervision [13, 42]. Emerging state-of-the-art transformer-based diffusion and flow backbones further push visual quality and compositional reasoning, demonstrating unprecedented instruction fidelity and fine-grained spatial control [25, 26, 49, 50]. Despite these single-step capabilities, isolated edit accuracy does not prevent error propagation, identity degradation, or semantic drift when applied iteratively across extended editing sequences.

Multi-turn image editing. Maintaining image quality and content consistency across successive edits is a central challenge in multi-turn image editing, as errors can accumulate over the editing sequence. Agentic frameworks tackle this challenge through planning, memory, and reflection [41, 54], while training-free methods improve consistency through inversion and attention control [62], high-frequency feature preservation [27], inversion sharing [38], and low-frequency latent alignment [47]. Other approaches learn sequential editing directly: VINCIE learns in-context editing from video-derived sequences [39], while AnchorEdit combines supervised causal multi-turn training with self-generated histories and identity-anchoring memory [52]. Reinforcement learning methods optimize trajectory-level editing rewards [16] or jointly optimize history-dependent reasoning and image generation [55]. More recently, MT-OPSD distills clean-conditioned editing behavior onto states generated through identity-edit rollouts [61], which accumulate errors while keeping the intended content fixed. Our training instead follows the editing process as it unfolds during inference: successive semantic edits with evolving content and preservation requirements. Through distribution matching, we align the student’s evolving output distribution, modeled by a learned auxiliary editor, with the target distribution defined by a teacher conditioned on the original image and cumulative instructions, without paired intermediate targets.

![](images/cc145b4a6a8a5a071a245b6b7e7271eedc17a7a53950163ae48b92aee0f46286.jpg)  
Figure 2. Method overview. Left: The student performs edits along a self-generated rollout, where each output becomes the input to the next turn. Given the previous output $I _ { e - 1 }$ and the current instruction, the few-step student produces a clean estimate $\hat { x } _ { e } ^ { s } .$ Right: Score distillation supervision with asymmetric conditioning. The student’s estimate $\hat { x } _ { e } ^ { s }$ is noised to level τ and fed to both a frozen teacher and a trainable auxiliary editor. Unlike the student, both are conditioned identically on the original clean image $I _ { 0 }$ and a compound instruction aggregating all edits up to turn $e ;$ however, the teacher represents the desired clean-conditioned distribution, while the auxiliary editor is trained to model the distribution actually produced by the student at edit e. The difference between their predictions yields the distillation gradient, which pushes the student’s multi-turn output distribution toward the teacher’s clean, non-degraded distribution.

## 3. Preliminaries: Score Distillation

Score Distillation Sampling (SDS) [37] uses a pretrained text-to-image diffusion model $\epsilon _ { \phi }$ to optimize a differentiable renderer G parameterized by θ. It relies on the observation that the model’s noise prediction at each noise level is proportional to the score of the noisy image distribution [43]. SDS uses this score to guide images produced by G without requiring paired target images. For a sampled timestep t and Gaussian noise ϵ, the SDS gradient is:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S D S } } = w ( t ) \left( \epsilon _ { \phi } ^ { \gamma } ( \mathbf { z } _ { t } ( \mathbf { x } ) ; c , t ) - \epsilon \right) \frac { \partial \mathbf { x } } { \partial \boldsymbol { \theta } } ,\tag{1}
$$

where x is an image produced by $G , \ \mathbf { z } _ { t } ( \mathbf { x } )$ is obtained by adding noise ϵ to x at timestep $t ,$ and c is a condition such as a text prompt. Here, $\epsilon _ { \phi } ^ { \gamma } ( \mathbf { z } _ { t } ( \mathbf { x } ) ; c , t )$ denotes the model’s noise prediction with classifier-free guidance, and $w ( t )$ weights the gradient at different timesteps.

SDS is known to produce blurry or oversaturated results, which has motivated numerous follow-up works [9, 32, 58, 63]. Some of these works have identified the high variance associated with subtracting the sampled noise ϵ from the model’s prediction as a contributing factor. In particular, Variational Score Distillation (VSD) [48] addresses this issue by replacing the sampled noise ϵ subtracted from the model’s prediction with the prediction of an auxiliary model $\epsilon _ { \mathrm { L o R A } }$ fine-tuned on images produced by G.

Noise-Free Score Distillation (NFSD) [22] provides a new interpretation of score distillation. Specifically, it decomposes the conditional noise prediction into three components: condition alignment, correction toward the model’s image distribution, and denoising: $\epsilon _ { \phi } ( \mathbf { z } _ { t } ; c , t ) =$ $\delta _ { \mathrm { C } } + \delta _ { \mathrm { D } } + \delta _ { \mathrm { N } }$ . Under this interpretation, subtracting the auxiliary prediction in VSD approximately cancels the denoising component, leaving the directions associated with condition alignment and domain correction to guide optimization.

While SDS and VSD optimize individual instances for a given prompt, Distribution Matching Distillation (DMD) [56, 57] extends this approach to training a generative model. DMD uses the VSD loss to train a student model, guided by a pretrained diffusion model referred to as the teacher. The student learns to generate images for arbitrary prompts. Our method adopts this teacher–student formulation for training a stable editor.

Although these methods were originally formulated for noise-predicting diffusion models, score distillation can also be applied to flow-matching models [17]. This extension relies on converting predicted flow velocities into score estimates. We adopt this formulation in our method.

## 4. Method

Image editing models are typically trained on pairs of input and target images [26]. However, in multi-turn editing, the model repeatedly receives inputs generated by its own previous predictions, which it was not trained to handle. This creates a train–test gap that can cause errors and artifacts to accumulate over successive edits.

To address this gap, we train our multi-turn image editing model $G _ { \theta }$ to edit images produced by repeatedly applying the model itself, as illustrated in Figure 2. We refer to the resulting sequences of edited images as rollouts. Supervising the model on these rollouts is challenging even when a ground-truth editing sequence is available. Since an edit can have many valid outcomes, the model’s output at a given turn may differ from the corresponding ground-truth image, even if the edit was applied correctly. As a result, subsequent edits in the rollout are applied to images that differ from those in the ground-truth sequence. Therefore, the ground-truth images cannot serve as paired targets for supervising these edits.

Inspired by Self-Forcing [17], which addresses a similar challenge in autoregressive video generation, we use score distillation to supervise our student model $G _ { \theta } ,$ , which becomes our stable editor after training. We compute the score-distillation signal using a pretrained flow-matching editor $v _ { \phi }$ that serves as the teacher.

A key challenge is that the student’s intermediate outputs may contain accumulated errors, making them unreliable inputs for the teacher. To address this, we introduce asymmetric conditioning of the student and teacher through compound-edit supervision. While the student applies edits sequentially to its own outputs, the teacher is always conditioned on the original image and a compound instruction combining all edits up to the current turn. This allows the teacher to provide a training signal that does not inherit the errors accumulated along the student’s trajectory.

## 4.1. Self-Generated Editing Rollouts

We generate editing rollouts by repeatedly applying the student editor $G _ { \theta }$ to its own outputs (left part of Figure 2). Each rollout starts from a source image $I _ { 0 }$ and a sequence of E instructions $c _ { 1 } , \ldots , c _ { E }$ , with $E = 1 0$ during training. We describe the dataset of instruction sequences in Section $5 .$ At turn e, the editor applies the next instruction $c _ { e }$ to the completed output from the previous turn:

$$
I _ { e } = G _ { \theta } ( I _ { e - 1 } , c _ { e } ) , \qquad e = 1 , \dots , E .
$$

Thus, after the first turn, the editor operates on its own generated images, as it does at inference.

We implement the student with FLUX.2-klein-9B, an image editor that operates in four denoising steps. We denote the complete four-step editing process by $G _ { \theta }$ and a single model prediction within this process by $v _ { \theta } .$ . We use the convention where $\sigma = 1$ corresponds to pure noise and $\sigma = 0$ to a clean image, with the predicted velocity pointing toward noise. At a denoising step with noise level $\sigma ,$ let $x _ { \sigma }$ be the current state. The model’s prediction provides an estimate of the clean output: $\hat { x } = x _ { \sigma } - \sigma v _ { \theta } ( x _ { \sigma } )$ , where, for brevity, we omit the dependence of $v _ { \theta }$ on the noise level, the previous image, and the current instruction.

For a given training rollout, we consider an edit turn $e \in \{ 1 , \ldots , E \}$ and a denoising exit step $s \in \{ 1 , \ldots , 4 \}$

We first obtain $I _ { e - 1 }$ by applying the preceding $e - 1$ edits, each using the full four-step denoising process without tracking gradients. At turn e, we execute the denoising process up to step s, where we compute the clean output estimate $\hat { x } _ { e } ^ { s }$ . Gradients are propagated only through the model’s prediction at this step, without backpropagating through earlier denoising steps or edit turns. Throughout the rollout, we pass the edited images directly between turns in the model’s latent space, avoiding repeated decoding and encoding.

## 4.2. Training

Asymmetric Conditioning. Having obtained the cleanoutput estimate $\hat { x } _ { e } ^ { s }$ at turn $e ,$ we now define the training signal. This signal should encourage adherence to the editing instructions, preservation of relevant image details, and generation of realistic, artifact-free images. We obtain this signal from a pretrained flow-matching image editor $v _ { \phi } ,$ which serves as our teacher. Specifically, we evaluate its score on a noisy version of $\hat { x } _ { e } ^ { s }$ at noise level τ, denoted by $\mathbf { z } _ { \tau } ( \hat { x } _ { e } ^ { s } )$ Unlike conventional text-to-image score distillation, our editing teacher is conditioned on both an image and an instruction. It can therefore guide the output toward results that satisfy the requested edit while preserving relevant details of the conditioning image.

The remaining question is how to condition the pretrained editing teacher $v _ { \phi }$ . Its conditioning image should serve as the source of the edit, while its text input should describe the intended change from that image to $\hat { x } _ { e } ^ { s }$ . A natural choice is the previous output $I _ { e - 1 }$ and the current instruction $c _ { e } ,$ which are also the inputs to the student $G _ { \theta }$ at turn e. However, beyond the first turn, $I _ { e - 1 }$ is generated by the student and may already contain artifacts. This creates a distribution shift for the teacher, which was trained on clean images. Moreover, since the teacher is expected to preserve details of its conditioning image that are not affected by the instruction, it may also preserve these artifacts rather than correct them. We examine this conditioning choice in the ablation study in Section 5.4.

Instead, we condition the teacher on the original clean image $I _ { 0 }$ and the accumulated instructions $\begin{array} { r l } { h _ { e } } & { { } = } \end{array}$ $( c _ { 1 } , \ldots , c _ { e } )$ . This allows the teacher to score the result of multiple edits as a single compound edit, without relying on potentially corrupted intermediate images. For example, a sequence of four editing instructions can be treated as a single compound edit:

$$
\begin{array} { r l } { { \mathrm { S t u d e n t : } } } & { \xrightarrow [ ] { \mathrm { \Delta ~ ` a d d ~  { \mathrm { \ x i d } } ~ } } , \quad \xrightarrow [ ] { \mathrm { \ddots } _ { \mathrm { m a k e ~ i t } } } \quad \xrightarrow [ ] { \mathrm { \ddots } _ { \mathrm { m a k e ~ i t } } } \quad \xrightarrow [ ] { \mathrm { \ddots } _ { \mathrm { a d d ~  { \mathrm { a g o l d } } } } } } \\ { { \mathrm { T e a c h e r : } } } & { \xrightarrow [ ] { \mathrm { \ddots } _ { \mathrm { a d d ~  { \mathrm { \ x i d } } ~ \mathrm { r e d } } \mathrm { l e a t h e r ~ h a n d b a g ~ w i t h ~ a ~ g o l d ~ b u c k l e } } , } } \end{array}
$$

To implement this supervision, the teacher $v _ { \phi }$ evaluates a noisy version of the student’s clean-output estimate $\hat { x } _ { e } ^ { s }$ denoted by $\mathbf { z } _ { \tau } ( \hat { x } _ { e } ^ { s } )$ , and produces the velocity prediction $v _ { \phi } \big ( \mathbf { z } _ { \tau } ( \hat { x } _ { e } ^ { s } ) ; I _ { 0 } , h _ { e } , \tau \big )$ used for score distillation.

This creates an asymmetry between the student and teacher. While the student applies each instruction to its own previous output, the teacher evaluates the result relative to the original image and the accumulated instructions. Consequently, the supervision remains anchored to the clean source image, even when errors accumulate across the student’s editing turns. At the first turn, the teacher’s conditioning $( I _ { 0 } , h _ { 1 } )$ coincides with the student’s input $\left( I _ { 0 } , c _ { 1 } \right)$ , so every rollout also supervises an ordinary single edit; we evaluate single edits separately in Section 5.3.

Loss. Following VSD [48] and its adaptation to generative models in DMD [56, 57], we update the student $G _ { \theta }$ using the difference between predictions of the pretrained teacher $v _ { \phi }$ and an auxiliary flow-matching editor $v _ { \psi }$ trained alongside the student on its generated outputs. For the student’s clean-output estimate $\hat { x } _ { e } ^ { s }$ , both models evaluate the same noisy version $\mathbf { z } _ { \tau } ( \hat { x } _ { e } ^ { s } )$ under the conditioning $( I _ { 0 } , h _ { e } )$ We denote their velocity predictions by:

$$
\begin{array} { r } { \hat { v } _ { \phi } ^ { \gamma } = v _ { \phi } ^ { \gamma } ( \mathbf { z } _ { \tau } ( \hat { x } _ { e } ^ { s } ) ; I _ { 0 } , h _ { e } , \tau ) , \quad \hat { v } _ { \psi } = v _ { \psi } ( \mathbf { z } _ { \tau } ( \hat { x } _ { e } ^ { s } ) ; I _ { 0 } , h _ { e } , \tau ) , } \end{array}
$$

where $\gamma$ is the teacher’s classifier-free guidance scale. The resulting distillation gradient is:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { d i s t i l l } } = w _ { e } ( \tau ) ( \hat { v } _ { \phi } ^ { \gamma } - \hat { v } _ { \psi } ) \frac { \partial \hat { x } _ { e } ^ { s } } { \partial \boldsymbol { \theta } } ,\tag{2}
$$

where the weight $w _ { e } ( \tau )$ accounts for the conversion from velocities to clean-output estimates and normalizes the update. For classifier-free guidance, the teacher’s unconditional branch drops the instructions but retains $I _ { 0 }$ . Following common practice [57], the auxiliary editor does not use classifier-free guidance.

Training Procedure. We alternate between updating the auxiliary editor $v _ { \psi }$ and the student $G _ { \theta }$ . For efficiency, at each training iteration, we uniformly sample a denoising exit step s and average each training loss across all editing turns $e = 1 , \ldots , E$ . Specifically, we generate a new rollout of E turns for each update of either the student or the auxiliary editor. When updating the student, we retain gradients through the clean-output estimate $\hat { x } _ { e } ^ { s }$ at step s of each turn. The completed images $I _ { 1 } , \ldots , I _ { E }$ are passed between turns in latent space, without propagating gradients.

The clean output estimates $\hat { x } _ { 1 } ^ { s } , \ldots , \hat { x } _ { E } ^ { s }$ are used to train the auxiliary editor. Specifically, we add noise to these estimates and optimize $v _ { \psi }$ using the flow-matching loss:

$$
\mathcal { L } _ { \mathrm { F M } } ( \psi ) = \frac { 1 } { E } \sum _ { e = 1 } ^ { E } \| v _ { \psi } \big ( \mathbf { z } _ { \tau } \big ( \hat { x } _ { e } ^ { s } \big ) ; I _ { 0 } , h _ { e } , \tau \big ) - \big ( \epsilon - \hat { x } _ { e } ^ { s } \big ) \| _ { 2 } ^ { 2 } .
$$

We first warm up the auxiliary editor for five iterations without updating the student. Subsequently, we update the student once for every five auxiliary-editor updates, computing the distillation gradients from the retained estimates $\hat { x } _ { e } ^ { s }$ and averaging them over all editing turns. The teacher remains fixed throughout training, while the student and auxiliary editor are updated alternately, with one held fixed while the other is optimized.

## 5. Experiments

## 5.1. Experimental Setup

Implementation details. Our student is FLUX.2-klein-9B [25], a 4-step distilled model, which we fine-tune with a LoRA [15] of rank 16 (27.8M trainable parameters, 0.3% of the model). The teacher and the auxiliary editor are both FLUX.2-klein-base-9B, the undistilled model of the same family. The teacher is frozen and uses classifier-free guidance with scale 2. The auxiliary editor trains its own LoRA of the same size as the student and is not guided. The student, being distilled, runs without classifier-free guidance, with 4 sampling steps in training rollouts as at inference. Each training rollout is a chain of 10 edits, and the student’s output latent is passed directly to the next edit, without going through the VAE. The noise level τ is drawn uniformly over [0, 1), independently for every turn, both for the student’s loss and for training the auxiliary editor.

The student and the auxiliary editor are optimized with AdamW [30] $( \beta = ( 0 , 0 . 9 9 9 )$ , weight decay 0.01, gradient clipping at 10, bf16), with learning rates of $4 \times 1 0 ^ { - 5 }$ for the student and $8 \times 1 0 ^ { - 6 }$ for the auxiliary editor. We train for 505 iterations: 5 warm-up iterations that update only the auxiliary editor, followed by 500 iterations in which the auxiliary editor is updated at every iteration and the student at every fifth, i.e., 100 student updates. Training takes about 4.4 GPU-hours on a single NVIDIA B200. At inference we use the same 4-step unguided sampler.

Training data. The inputs for the training rollouts are 96 synthetic 1024 × 1024 images generated with FLUX.2- klein-9B from short scene descriptions. Each image comes with one cumulative chain of 10 edit instructions (960 in total). All edits are local: each adds, removes, replaces or resizes an object, or changes one of its attributes, such as its color. Our evaluation benchmark, EdiVal-10 (Sec. 5.3), consists of real photographs, so no image is shared between training and evaluation.

Baselines. We compare our stable editor with its pretrained backbone model, training-free strategies for reducing multi-turn drift, and a trained multi-turn editor. Our backbone model baseline is FLUX.2-klein-9B without our fine-tuning, using the same 4-step sampler as our student. We evaluate Emu Edit’s pixel-thresholding strategy [42], VAE-LFA [47], and FreqEdit [27] using the same backbone model. For Emu Edit, we follow MT-OPSD’s implementation and settings (blur σ = 3, threshold α = 0.03) [61]. For VAE-LFA, we follow its official FLUX.2-klein-9B code, which differs from the paper’s description. To match our student’s previous-image-only input setting, we port FreqEdit’s official code for the variant that conditions on the previous turn, without the original-source quality guidance of its Kontext version, and use its Kontext settings. Our model, its backbone model and the training-free methods sample each chain with the same fixed seed, so that their differences come from the method rather than from sampling noise. We also compare with MT-OPSD [61], using its released LoRA on FLUX.2-klein-base-9B, and include the same backbone without this LoRA as its corresponding base-model baseline. Both use the authors’ inference script and its fixed seed, at 512-pixel resolution, with 30 sampling steps and guidance scale 4; their outputs are upsampled to 1024×1024 for evaluation. We fix the seed per chain across all model evaluations.

![](images/39660d496a091a25898a85e3a65ef27a5f7c51a4f53fa21f7f8a441adf50fe6e.jpg)  
Figure 3. Multi-turn image editing. We show selected turns from 20-step editing sequences. Each displayed instruction is followed by four no-op instructions (“keep the image as is”), and the images show the results after those no-ops. While the original FLUX.2 Klein accumulates color shifts and loses detail, fine-tuning it with our training paradigm (+Ours) yields a more stable editing process, preserving the scene and prior edits while following the requested changes.

![](images/ae027f14d52c88d81c619103146ad29d744c65aaccdebb12df313896e4909bd4.jpg)  
Figure 4. Ten successive removals. Each turn removes one object from a synthetic garage scene: the drill, broom, car tire, ladder and bicycle (turns 1–5), then the lawn mower, vise, toolbox, pegboard and shelves (turns 6–10). By the last turn, every other method has visibly damaged the scene, while our model leaves the walls, the floor and the workbench as they were. Emu Edit leaves pale, blocky patches on the wall, VAE-LFA darkens the image under a coarse, mottled texture, FreqEdit casts the walls and floor yellow, and MT-OPSD washes the room out into a blurred, almost white space.

## 5.2. Qualitative Evaluation

Figures 1 and 3 compare our model with its backbone model on editing sequences interspersed with no-op instructions. We further compare against the training-free methods and MT-OPSD on sequences of consecutive semantic edits in Figure 4. The source scenes for these comparisons were generated independently of the training data using FLUX.2- klein-9B. Figure 4 demonstrates that degradation can accumulate even under simple editing instructions: after ten successive object removals, all competing methods visibly damage the surrounding scene, while our model successfully preserves it.

Each training-free method ties every output to part of the previous image, and this is also its main weakness: when an edit has to change exactly that part, content from earlier turns leaks into the result. Emu Edit reverts pixels that change only slightly, so remnants of earlier images survive; VAE-LFA aligns low-frequency latent statistics with earlier outputs, so their colours persist; and FreqEdit steers highfrequency detail toward the previous image, so their textures persist. Figure 8 in Appendix F illustrates these effects over three successive background replacements. MT-OPSD leaves no such traces but blurs the image and washes out its colours, consistent with the degradation observed in Figure 4.

Further comparisons with the backbone model, including a full sequence of fifteen edits, are shown in Figs. 9 and 10 in the appendix.

## 5.3. Quantitative Evaluation

Evaluation benchmark. Released multi-turn editing benchmarks are short: EdiVal [8] chains three edits, MagicBrush [59] at most three, and MSE-Bench [39] five, too few for drift to build up. Omni IIE Bench [53] advertises editing sequences of up to 16 turns, but they average 4.35 turns, and only 9 of its 260 reach ten. The one benchmark we know of that is built around ten turns, LME-Bench [61], contains 100 sessions on synthetic images, and each session opens with a global edit (an art style, a black-andwhite or sepia grade, or a change of weather or time of day) and includes three more later. After such a turn, nothing in the image is meant to stay as it was, so unintended changes can no longer be distinguished from those required by the edit through comparison with the source. We therefore introduce EdiVal-10, in which we extend EdiVal to ten turns of local edits, and report results on LME-Bench in Appendix D. It contains 252 chains of 10 cumulative edits on real photographs from GEdit-Bench [29], center-cropped and resized to 1024 × 1024. Following EdiVal, a language model lists the objects in each photo using EdiVal’s objectlisting prompt, and GroundingDINO [28] grounds them. Every edit targets a grounded object and belongs to one of eight local types: add, remove, replace, color, material, text, count, and position. Each chain also reserves one or two anchor objects that no instruction may edit or mention. The language model then writes the instructions from a perchain plan of objects and edit types. They are checked by rule-based validators and by the language model acting as a critic, and rejected turns are sent back for rewriting. Every language-model step of its construction uses Claude Sonnet 5.5 [2].

Table 1. Quantitative comparison on EdiVal-10. All metrics are computed after 10 cumulative edits and averaged over the 252 chains, except the anchor distance, which is averaged over the 148 chains whose anchor object lies mostly outside the edited objects (Appendix B) Quality is measured by the KRIS-Bench image-quality score of the final image, assigned by a VLM judge (1–5), and by the change in HPSv3 from turn 1 to turn 10. Preservation is measured by the KRIS-Bench consistency score assigned by the same judge on the whole image (1–5; higher means fewer unrequested changes), the DINOv2 distance between the source and the final image on unedited anchor objects, and the LPIPS distance on the background outside all edited objects. Editing is measured by the KRIS-Bench instruction-following score at turn 10 (1–5); it checks that stability does not come from skipping edits. Methods are grouped by backbone; the best result in each column is shown in bold and the second best is underlined. : significantly different from our model (more than two standard errors, paired over chains).
<table><tr><td></td><td colspan="2">Quality</td><td colspan="3">Preservation</td><td>Editing</td></tr><tr><td>Method</td><td>Quality (KRIS, 1–5) ↑</td><td>∆HPSv3 (turn 1→10) ↑</td><td>Consistency (KRIS, 1–5) ↑</td><td>Anchor dist. (DINOv2) ↓</td><td>Background dist. (LPIPS) ↓</td><td>Instruction following (KRIS, 1–5) ↑</td></tr><tr><td colspan="7">FLUX.2-klein-base-9B (undistilled)</td></tr><tr><td>Base model</td><td>1.42*</td><td>-3.17*</td><td>1.08*</td><td>0.660*</td><td>0.531*</td><td>3.10*</td></tr><tr><td>MT-OPSD [61]</td><td>1.95*</td><td>-2.54*</td><td>1.29*</td><td>0.505*</td><td>0.455*</td><td>3.16*</td></tr><tr><td colspan="7">FLUX.2-klein-9B (4-step distilled)</td></tr><tr><td>Base model</td><td>2.73*</td><td>-0.78*</td><td>3.03*</td><td>0.314*</td><td>0.351*</td><td>3.31</td></tr><tr><td>+ Emu Edit [42]</td><td>2.90*</td><td>-0.15</td><td>3.32*</td><td>0.196</td><td>0.264*</td><td>3.29</td></tr><tr><td>+ VAE-LFA [47]</td><td>2.80*</td><td>-0.89*</td><td>3.04*</td><td>0.289*</td><td>0.313*</td><td>3.32</td></tr><tr><td>+ FreqEdit [27]</td><td>2.83*</td><td>-0.82*</td><td>3.18*</td><td>0.283*</td><td>0.331*</td><td>3.39</td></tr><tr><td>Ours</td><td>3.19</td><td>-0.17</td><td>3.85</td><td>0.203</td><td>0.254</td><td>3.38</td></tr></table>

Metrics. We score the final image of each chain (turn 10) on three aspects and average over chains. Quality: a VLM judge (Gemma-4-31B [12]) rates the image with the imagequality prompt of KRIS-Bench [51], and ∆HPSv3 [31] measures how much a learned quality estimate changes from turn 1 to turn 10. Preservation: the same judge rates, with the KRIS-Bench consistency prompt, how well everything the instructions do not ask to change is kept; locally, we measure the DINOv2 [35] distance to the source on the unedited anchor objects and the LPIPS [60] distance on the background outside all edited objects. Editing: the judge’s KRIS-Bench instruction-following score checks that the requested edits are carried out, so that preservation cannot come from skipping them. Prompts, masks and the judge’s controls are detailed in Appendix B.

Table 1 reports quality, preservation and editing after ten cumulative edits. To compare a method with ours, we take the difference between the two on each chain and report the mean difference ± its standard error over the chains; we call a difference significant when it is more than twice its standard error.

Quality. Our method produces the highest-quality final images: its KRIS quality score is significantly higher than that of every other method, the smallest margin being over Emu Edit (0.30±0.08). This result is aligned with the qualitative results in Figure 4. Emu Edit’s ∆HPSv3 does not differ significantly from ours (−0.15 vs. −0.17), and its visual artifacts are demonstrated in Figure 4.

Preservation of unedited content. Judged on the whole image, our method makes the fewest unintended changes: its KRIS consistency score is 3.85, compared with 3.04 to 3.32 for the training-free methods, 3.03 for the base model and 1.29 for MT-OPSD. On the background, our method attains the lowest LPIPS distance (0.254), a massive im-

Teachersees previousimage

![](images/755673500bac0799f2764f2b8763a4ccce5ffd2c54e7f103394819f3c583fd94.jpg)  
Studentsees sourceimage

Figure 5. Conditioning ablations. Turns 3, 6, 8 and 9 of a ten-turn editing sequence on a source scene generated with FLUX.2-klein-9B independently of the training data; the instruction below each column is the one applied at that turn. Our model changes only what each instruction names, and the image stays clean. The base model degrades the image: it turns yellow and washed out over the turns, and the knitted texture requested for the teddy bear at turn 9 also covers the rocking horse. Training with a teacher that also sees the previous image does not prevent this degradation: blotchy noise spreads over the walls and the rug from turn to turn. When the student also sees the source image, earlier edits are undone: the stool made yellow at turn 3 is blue again by turn 6, and the ball made green at turn 6 is red again b turn 8.

provement over the base model at 0.351. Compared to our method, every other method is worse, Emu Edit by the smallest margin (0.264; +0.010 ± 0.005, t = 2.1). On the anchor objects, our method and Emu Edit attain the lowest DINOv2 distances (0.203 and 0.196; the difference, −0.007 ± 0.012, is not significant), while all other methods are significantly worse than ours. Emu Edit is comparable to our method only on two local metrics, not on wholeimage consistency; our model matches or exceeds it on both without copying pixels.

Editing. The gains in quality and preservation do not come at the expense of instruction following, with instruction-following scores remaining comparable. On the KRIS-Bench instruction-following score, our method (3.38) does not differ significantly from the highest-scoring method, FreqEdit (3.39), and scores significantly higher than MT-OPSD (3.16) and its base model (3.10).

Single edits. Stability over many turns should not come at the cost of single edits. We therefore also apply each method once to each of the 606 English instructions of GEdit-Bench [29] and measure the change in HPSv3 from the source to the edited image (Appendix E). A single edit lowers HPSv3 for every method, and least for ours: by 0.18, against 0.41 for the base model (0.23 ± 0.02), while every other method loses significantly more.

Table 2. Ablations on EdiVal-10. Each row changes one aspect of our model; all variants are trained for 100 student updates and evaluated after 10 cumulative edits on all 252 chains (the anchor distance on the 148 chains whose anchor object lies mostly outside the edited objects; Appendix B). : significantly different from our model (more than two standard errors, paired over chains).
<table><tr><td></td><td colspan="2">Quality</td><td colspan="3">Preservation</td><td>Editing</td></tr><tr><td>Variant</td><td>Quality (KRIS, 1–5) ↑</td><td>∆HPSv3 (turn 1→10) ↑</td><td>Consistency (KRIS, 1–5) ↑</td><td>Anchor dist. (DINOv2) ↓</td><td>Background dist. (LPIPS) ↓</td><td>Instruction following (KRIS, 1–5) ↑</td></tr><tr><td>Ours</td><td>3.19</td><td>-0.17</td><td>3.85</td><td>0.203</td><td>0.254</td><td>3.38</td></tr><tr><td>Teacher sees previous image</td><td>2.75*</td><td>-0.86*</td><td>2.81*</td><td>0.307*</td><td>0.366*</td><td>3.32</td></tr><tr><td>Teacher sees only previous image and instruction</td><td>2.71*</td><td>-1.12*</td><td>2.28*</td><td>0.366*</td><td>0.381*</td><td>3.23*</td></tr><tr><td>Student sees source image</td><td>3.46*</td><td>-0.08</td><td>4.08*</td><td>0.188</td><td>0.230*</td><td>2.80*</td></tr><tr><td>5-edit training rollouts (vs. 10)</td><td>3.23</td><td>-0.28*</td><td>3.29*</td><td>0.265*</td><td>0.334*</td><td>3.25*</td></tr></table>

## 5.4. Ablation Study

We evaluate the role of asymmetric conditioning on EdiVal-10 by varying the inputs provided to the teacher, auxiliary editor, and student, and the length of the training rollouts. The teacher and auxiliary editor receive identical conditioning in all variants. All variants are trained for 100 student updates. Table 2 reports the quantitative results, and Figure 5 illustrates the conditioning variants qualitatively.

Teacher and auxiliary conditioning. Our teacher and auxiliary editor are conditioned on the original image and the compound instruction, anchoring score distillation to the clean source. We compare this design with two alternatives: additionally providing the previous student output to both models, and replacing the original image and compound instruction with the previous output and current instruction. Both alternatives substantially reduce preservation: consistency drops from 3.85 to 2.81 and 2.28, respectively, compared with 3.03 for the pretrained base model. Figure 5 illustrates the former variant: as with the base model, the image degrades over the turns, and blotchy noise spreads over the walls and the rug. These results suggest that conditioning the score models on intermediate student outputs can reinforce accumulated artifacts rather than correct them.

Student conditioning. Our student receives only the previous output and current instruction. Providing the original image as an additional reference improves consistency from 3.85 to 4.08, but reduces instruction following from 3.38 to 2.80. As shown in Figure 5, earlier edits are undone as later ones are applied: the recoloured stool and ball return to their original colours. Thus, stronger preservation of the source does not necessarily imply successful multiturn editing. Our asymmetric design maintains both scene consistency and the changes introduced across successive turns.

Rollout length. We train on rollouts of 10 edits, the same length as the evaluated sequences. A student trained on 5- edit rollouts still improves over the base model after ten turns (consistency 3.29 vs. 3.03), but recovers only about a third of our gain (3.85) and is significantly worse than our model on every preservation measure. Training on its own outputs over the full length of the sequences it edits at inference is therefore important.

## 6. Conclusions, Limitations and Future Work

We introduced a training approach for stable multi-turn image editing. Our central idea is that stability should be learned over the entire recursive process, rather than at individual editing steps. We achieve this through asymmetric conditioning: the student learns from its own generated editing trajectories, while the teacher remains anchored to the original clean image and the accumulated editing instructions. This allows us to supervise a long, imperfect sequence of edits through a shorter, clean alternative path, without exposing the teacher to the errors accumulated by the student.

Interestingly, multi-turn editing does not always lead to degradation. When edits repeatedly replace the subject and background, existing models can maintain visual quality even over long sequences (Fig. 7). We attribute this to the fact that such edits generate new content rather than preserving existing details. In contrast, edits that require content to persist across many turns are more susceptible to accumulated errors. Our work therefore focuses on sequences in which unaffected content must remain consistent.

![](images/df6c1c84fc94b0f15e6e1693e0b44a3096237afc075c0a15770c895dedfcc27c.jpg)  
Figure 6. Limitations. We show selected steps from a nine-edit sequence. Both the base model and ours follow the final instruction “Make the hiker smile,” but also remove the sunglasses added in a previous edit. While our method reduces accumulated drift, it inherits the editing capabilities and limitations of the base model.

Our approach substantially reduces accumulated drift, but some challenges remain. Particularly difficult editing instructions may still affect previously edited content (Fig. 6), suggesting that stability and instruction following are related but distinct objectives. Furthermore, although our method improves stability beyond the sequence lengths seen during training, extending this robustness to much longer editing sequences remains an interesting direction for future work.

Beyond image editing, our approach suggests a more general training principle for recursive generative processes. Self-Forcing exposes the model to the distribution it will encounter during inference, while asymmetric teacher supervision provides reliable guidance from a clean reference. Score distillation connects these two paths without requiring paired target outputs, which is particularly useful when many valid results are possible. We believe this combination may be useful in other generative tasks where models repeatedly operate on their own outputs.

## Acknowledgments

We thank Ron Mokady for his valuable feedback. This research was supported in part by the Israel Science Foundation (grants no. 3441/21 and 1473/24), Len Blavatnik and the Blavatnik Family Foundation, and the Bronfman Family Foundation. It was made possible through a GPU compute resource grant funded by the Association of University Heads, the Council for

## References

[1] Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar Averbuch-Elor, and Daniel Cohen-Or. Cross-image attention for zeroshot appearance transfer. In ACM SIGGRAPH 2024 conference papers, pages 1–12, 2024. 2

[2] Anthropic. System Card: Claude Sonnet 5.5. https: //www.anthropic.com/claude-sonnet-5-5- system-card, 2026. 8

[3] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18187–18197. IEEE, 2022. 2

[4] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel Cohen-Or. Stable flow: Vital layers for training-free image editing. In 2025 IEEE/CVF Conference on Computer Vision and Pat tern Recognition (CVPR), pages 7877–7888. IEEE, 2025. 2

[5] Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolinario Passos. Ledits++: Limitless image editing using´ text-to-image models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8861–8870, 2024. 2

[6] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18392–18402. IEEE, 2023. 1, 2

[7] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 22503–22513. IEEE, 2023. 2

[8] Tianyu Chen, Yasi Zhang, Zhi Zhang, Peiyu Yu, Shu Wang, Zhendong Wang, Kevin Lin, Xiaofei Wang, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Jianwen Xie, Oscar Leong, Lijuan Wang, Ying Nian Wu, and Mingyuan Zhou. EdiVal-Agent: An object-centric framework for automated, fine-grained evaluation of multi-turn editing. In International Conference on Learning Representations (ICLR), 2026. 7

[9] Jaeyoung Chung, Suyoung Lee, Hyeongjin Nam, Jaerin Lee, and Kyoung Mu Lee. Luciddreamer: Domain-free gen eration of 3d gaussian splatting scenes. arXiv preprint arXiv:2311.13384, 2023. 3

[10] Gilad Deutch, Rinon Gal, Daniel Garibi, Or Patashnik, and Daniel Cohen-Or. Turboedit: Text-based image editing using few-step diffusion models. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12, 2024. 2

[11] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegan-nada: Clip guided domain adaptation of image generators. ACM Trans actions on Graphics (ToG), 41(4):1–13, 2022. 2

[12] Gemma Team. Gemma 4 technical report. arXiv preprint arXiv:2607.02770, 2026. 8, 15

[13] Zigang Geng, Binxin Yang, Tiankai Hang, Chen Li, Shuyang Gu, Ting Zhang, Jianmin Bao, Zheng Zhang, Houqiang Li, Han Hu, et al. Instructdiffusion: A generalist modeling interface for vision tasks. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12709–12720. IEEE, 2024. 2

[14] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2

[15] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. 5

[16] Jiahui Huang, Yasi Zhang, Tianyu Chen, Shu Wang, Jianwen Xie, Oscar Leong, Mingyuan Zhou, Nanzhu Wang, and Ying Nian Wu. MT-EditFlow: Reinforcement learning for multi-turn image editing with flow matching. arXiv preprint arXiv:2606.01985, 2026. 2

[17] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3, 4

[18] Saar Huberman, Ron Mokady, Or Patashnik, and Daniel Cohen-Or. Token-to-token alignment of text embeddings for semantic blending. arXiv preprint arXiv:2606.24021, 2026. 2

[19] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12469– 12478. IEEE, 2024. 2

[20] Edo Kadosh, Nir Goren, Or Patashnik, Daniel Garibi, and Daniel Cohen-Or. Tight inversion: Image-conditioned inversion for real image editing. In 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 7039–7049. IEEE, 2025. 2

[21] Ronen Kamenetsky, Sara Dorfman, Daniel Garibi, Roni Paiss, Or Patashnik, and Daniel Cohen-Or. Saedit: Tokenlevel control for continuous image editing via sparse autoencoder. In European Conference on Computer Vision, pages 359–377. Springer, 2026. 2

[22] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. Noise-free score distillation. In International conference on learning representations, pages 53344–53365, 2024. 3

[23] Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 19721–19730. IEEE, 2025. 2

[24] Nupur Kumari, Sheng-Yu Wang, Nanxuan Zhao, Yotam Nitzan, Yuheng Li, Krishna Kumar Singh, Richard Zhang, Eli Shechtman, Jun-Yan Zhu, and Xun Huang. Learning an

image editing model without image editing pairs. In ICLR, 2026. 1

[25] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025. 2, 5

[26] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dock horn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025. 1, 2, 3

[27] Yucheng Liao, Jiajun Liang, Kaiqian Cui, Baoquan Zhao, Haoran Xie, Wei Liu, Qing Li, and Xudong Mao. Freqedit: Preserving high-frequency features for robust multi-turn image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 43525– 43535, 2026. 2, 5, 8, 17

[28] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding DINO: Mar rying DINO with grounded pre-training for open-set object detection. In European Conference on Computer Vision (ECCV), 2024. 8

[29] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1X-Edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 8, 10, 16

[30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. 5

[31] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. HPSv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 15086–15095, 2025. 8, 15

[32] David McAllister, Songwei Ge, Jia-Bin Huang, David W. Jacobs, Alexei A. Efros, Aleksander Holynski, and Angjoo Kanazawa. Rethinking score distillation as a bridge between image distributions. In Advances in Neural Information Processing Systems, 2024. 3

[33] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2

[34] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6038–6047. IEEE, 2023. 2

[35] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mah moud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Je-´

gou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (TMLR), 2024. 8, 16

[36] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In 2021 IEEE/CVF international conference on computer vision (ICCV), pages 2065–2074. IEEE, 2021. 2

[37] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 3

[38] Chunyu Qi, Zhuoran Song, Cheng Gu, Zhihui Fu, Jianglong Chang, Jun Wang, Xiaoyao Liang, and Haibing Guan. Excavating consistency across editing steps for effective multistep image editing. In Proceedings of the 63rd Design Automation Conference, 2026. 2

[39] Leigang Qu, Feng Cheng, Ziyan Yang, Qi Zhao, Shanchuan Lin, Yichun Shi, Yicong Li, Wenjie Wang, Tat-Seng Chua, and Lu Jiang. Vincie: Unlocking in-context image editing from video. In International Conference on Learning Representations, pages 114873–114918, 2026. 2, 7

[40] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 2

[41] Fei Shen, Chengyu Xie, Lihong Wang, Zhanyi Zhang, Xin Jiang, Xiaoyu Du, and Jinhui Tang. Imagagent: Orchestrating multi-turn image editing via constraint-aware planning and reflection. arXiv preprint arXiv:2603.29602, 2026. 2

[42] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In 2024 ieee/cvf conference on computer vision and pattern recognition (cvpr), pages 8871–8879. IEEE, 2024. 2, 5, 8, 17

[43] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3

[44] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 14940–14950. IEEE, 2025. 1

[45] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1921–1930. IEEE, 2023. 2

[46] Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22532–22541. IEEE, 2023. 2

[47] Xiaoce Wang, Sifan Zhou, Kaifei Wang, Leli Xu, Xuerui Qiu, Tao He, and Ming Li. Why do dit editors drift? plug-

and-play low frequency alignment in vae latent space. arXiv preprint arXiv:2605.08250, 2026. 2, 5, 8, 17

[48] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. ProlificDreamer: High-fidelity and diverse text-to-3D generation with variational score distillation. In Advances in Neural Information Processing Systems, 2023. 2, 3, 5

[49] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 1, 2

[50] Chenyuan Wu, Jiahao Wang, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, et al. Omnigen2: Towards instruction-aligned multimodal generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21964–21975, 2026. 2

[51] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, Ming-Hsuan Yang, and Xu Yang. KRIS-Bench: Benchmarking next-level intelligent image editing models. In Advances in Neural In formation Processing Systems (NeurIPS), 2025. 8, 15

[52] Hang Xu, Xiaoxiao Ma, Guohui Zhang, Yu Hu, Siming Fu, Jie Huang, Lin Song, Haoyang Huang, Nan Duan, and Feng Zhao. Anchoredit: Maintaining temporal consistency in multi-turn image editing via causal memory. arXiv preprint arXiv:2606.11751, 2026. 2

[53] Yujia Yang, Yuanxiang Wang, Zhenyu Guan, Tiankun Yang, Chenxi Bao, Haopeng Jin, Jinwen Luo, Xinyu Zuo, Lisheng Duan, Haijin Liang, Jin Ma, Xinming Wang, Ruiwen Tao, and Hongzhu Yi. Omni IIE Bench: Benchmarking the practical capabilities of image editing models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1089–1099, 2026. 7

[54] Ruijie Ye, Jiayi Zhang, Zhuoxin Liu, Zihao Zhu, Siyuan Yang, Li Li, Tianfu Fu, Franck Dernoncourt, Yue Zhao, Jiacheng Zhu, et al. Agent banana: High-fidelity image editing with agentic thinking and tooling. arXiv preprint arXiv:2602.09084, 2026. 2

[55] Yuxiao Ye, Haoran He, Fangyuan Kong, Xintao Wang, Pengfei Wan, Kun Gai, and Ling Pan. Edit-r2: Context aware reinforcement learning for multi-turn image editing. arXiv preprint arXiv:2606.05950, 2026. 2

[56] Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang,¨ Eli Shechtman, Fredo Durand, and William T. Freeman. Im-´ proved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, pages 47455–47487, 2024. 3, 5

[57] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shecht-¨ man, Fredo Durand, William T. Freeman, and Taesung Park.´ One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024. 2, 3, 5

[58] Xin Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Song-Hai Zhang, and Xiaojuan Qi. Text-to-3d with classifier score distillation. In International conference on learning representations, pages 19163–19177, 2024. 3

[59] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in neural information processing systems, 36:31428–31449, 2023. 2, 7

[60] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595, 2018. 8, 16

[61] Liangbing Zhao, Le Zhuo, and Mohamed Elhoseiny. Onpolicy self-distillation for multi-turn image editing, 2026. 2, 6, 7, 8, 16, 17

[62] Zijun Zhou, Yingying Deng, Xiangyu He, Weiming Dong, and Fan Tang. Multi-turn consistent image editing. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 15792–15801. IEEE, 2025. 2

[63] Joseph Zhu and Peiye Zhuang. Hifa: High-fidelity textto-3d with advanced diffusion guidance. arXiv preprint arXiv:2305.18766, 2023. 3

## Appendix

## A. Training Pseudocode

Algorithms 1 and 2 summarize the training procedure in Sec. 4. Hyperparameters are the rollout length $E ,$ samplingstep count K, warm-up duration W, student-update interval $R ,$ guidance scale $\gamma ,$ and training-iteration count $T .$ Here, sg denotes stop-gradient, and $\| \cdot \| _ { 2 } ^ { 2 }$ the mean squared error over latent elements. Clean estimates are computed as $\hat { x } _ { 0 } = x _ { \sigma } - \sigma v$ . All images passed between edits are represented as latents. $\mathrm { S a m p l e } _ { < s }$ runs the sampling steps before the exit step, and $\mathrm { S a m p l e } _ { > s }$ completes sampling from the exit-step state and velocity. Update denotes an optimizer step, D the training data, and $p _ { \tau }$ the noise-level distribution. The teacher $v _ { \phi }$ is evaluated with classifier-free guidance, $\begin{array} { r } { v _ { \phi } ^ { \gamma } ( x ; I _ { 0 } , h ) = v _ { \phi } ( x ; I _ { 0 } , \mathcal { D } ) + \gamma \left( v _ { \phi } ( x ; I _ { 0 } , h ) - \right. } \end{array}$ $v _ { \phi } ( x ; I _ { 0 } , \mathcal { O } ) \big )$ , where ∅ is the empty instruction; the auxiliary editor $v _ { \psi }$ is not guided. The compound instruction $h _ { e }$ expresses the accumulated edit intent through turn e.

Algorithm 1 Self-generated editing rollout (Sec. 4.1)   
Require: Student $G _ { \theta } ,$ , source $I _ { 0 } ,$ edits $c _ { 1 : E }$ , sampling-step   
count K, gradient flag b   
Ensure: Clean exit estimates $\hat { x } _ { 1 : E } ^ { s }$   
1 $\mathrm { : } \ s \sim \operatorname { U n i f o r m } \{ 1 , \dots , K \}$   
2: for $e = 1 , \ldots , E$ do   
3: $z \sim \mathcal { N } ( 0 , \mathbf { I } )$   
4: $x _ { s } \gets \mathrm { s g } ( \mathrm { S a m p l e } _ { < s } ( G _ { \theta } , z ; I _ { e - 1 } , c _ { e } ) )$   
5: $v _ { s } \gets v _ { \theta } ( x _ { s } ; I _ { e - 1 } , c _ { e } )$   
6: $\hat { x } _ { e } ^ { s }  x _ { s } - \sigma _ { s } v _ { s }$   
7: if ¬b then   
8: $\hat { x } _ { e } ^ { s } \gets \mathrm { s g } ( \hat { x } _ { e } ^ { s } )$   
9: end if   
10: $\mathbf { i f } e < E$ then   
11: $I _ { e } \gets \mathrm { s g } ( \mathrm { S a m p l e } _ { > s } ( G _ { \theta } , x _ { s } , v _ { s } ; I _ { e - 1 } , c _ { e } ) )$   
12: end if   
13: end for   
14: return $\hat { x } _ { 1 : E } ^ { s }$

## B. Evaluation Metrics

All metrics are computed at the final turn (turn 10) of each EdiVal-10 chain and averaged over chains; comparisons between methods are paired over chains and reported with standard errors. The three judged scores use the prompts of KRIS-Bench [51] unchanged, with Gemma-4-31B [12] as the judge in place of GPT-4o.

Source images. EdiVal-10 draws its chains only from GEdit-Bench photographs whose short side is at least 768 pixels, so the center crop and resize to 1024 × 1024 upsamples a source by at most 1.33×.

Algorithm 2 Training (Sec. 4.2)   
Require: Student $G _ { \theta } ,$ , teacher $v _ { \phi } ,$ data D   
Require: Hyperparameters $E , \dot { K } , W , R , \gamma , T$   
1: $v _ { \psi }  v _ { \phi } ;$ freeze $v _ { \phi }$   
2: for $t = \dot { 1 } , \dots , T$ do   
3: if $t > W$ and $\left( t - W - 1 \right)$ mod $R = 0$ then   
4: $( I _ { 0 } , c _ { 1 : E } , h _ { 1 : E } ) \sim \mathcal { D }$   
5: $\hat { x } _ { 1 : E } ^ { s } \gets \mathrm { R o L L O U T } ( G _ { \theta } , I _ { 0 } , c _ { 1 : E } , K , \mathrm { t r u e } )$   
6: for $e = 1 , \ldots , E$ do   
7: $\tau \sim p _ { \tau } , \epsilon \sim \mathcal { N } ( 0 , \mathbf { I } )$   
8: $x _ { \tau } \gets ( 1 - \sigma _ { \tau } ) \hat { x } _ { e } ^ { s } + \sigma _ { \tau } \epsilon$   
9: $\hat { x } _ { 0 } ^ { \mathrm { t e a c h e r } } \gets \mathrm { s g } ( x _ { \tau } - \sigma _ { \tau } v _ { \phi } ^ { \gamma } ( x _ { \tau } ; I _ { 0 } , h _ { e } ) )$   
10: $\hat { x } _ { 0 } ^ { \mathrm { a u x } } \gets \mathrm { s g } ( x _ { \tau } - \sigma _ { \tau } v _ { \psi } ( \dot { x } _ { \tau } ; I _ { 0 } , h _ { e } ) )$   
$\begin{array} { r l } { . \ll } & { { } \hat { x } _ { 0 } ^ { \mathrm { a u x } } - \hat { x } _ { 0 } ^ { \mathrm { t e a c h e r } } } \end{array}$   
11: g<sub>e</sub> ← mean |xˆ<sup>s</sup> − xˆ <sup>teacher</sup>|   
12: $\begin{array} { r } { \ell _ { e } ^ { G } \gets \frac { 1 } { 2 } \| \hat { x } _ { e } ^ { s } - \mathrm { s g } ( \hat { x } _ { e } ^ { s } - g _ { e } ) \| _ { 2 } ^ { 2 } } \end{array}$   
13: end for   
14: $\begin{array} { r } { \theta \gets \mathrm { U p d a t e } ( \theta , \nabla _ { \theta } \frac { 1 } { E } \sum _ { e } \ell _ { e } ^ { G } ) } \end{array}$   
15: end if   
16: $( I _ { 0 } , c _ { 1 : E } , h _ { 1 : E } ) \sim \mathcal { D }$   
17: $\hat { x } _ { 1 : E } ^ { s } \gets \mathrm { R o L L O U T } ( G _ { \theta } , I _ { 0 } , c _ { 1 : E } , K , \mathrm { f a l s e } )$   
18: for $e = 1 , \ldots , E$ do   
19: $\tau ^ { \prime } \sim p _ { \tau } , \epsilon ^ { \prime } \sim \mathcal { N } ( 0 , \mathbf { I } )$   
20: $x _ { \tau ^ { \prime } } \gets ( 1 - \sigma _ { \tau ^ { \prime } } ) \hat { x } _ { e } ^ { s } + \sigma _ { \tau ^ { \prime } } \epsilon ^ { \prime }$   
21: $\ell _ { e } ^ { \mathrm { a u x } } \gets \| v _ { \psi } ( x _ { \tau ^ { \prime } } ; \bar { I } _ { 0 } , h _ { e } ) - ( \epsilon ^ { \prime } - \mathrm { s g } ( \hat { x } _ { e } ^ { s } ) ) \| _ { 2 } ^ { 2 }$   
22: end for   
23: ψ $\begin{array} { r l } { \mathrm { ~ } } & { { }  \mathrm { U p d a t e } ( \psi , \nabla _ { \psi } \frac { 1 } { E } \sum _ { e } \ell _ { e } ^ { \mathrm { a u x } } ) } \end{array}$   
24: end for

Quality. The judge rates the final image with the imagequality prompt, checking structural coherence, natural appearance, absence of artifacts and texture fidelity, and yielding a score from 1 to 5. The prompt describes every image as AI-generated, and on our real photographs the judge also reports defects in untouched sources (their mean score is 3.57); since all methods are scored on the same photographs and compared chain by chain, this affects the absolute scores but not the comparisons. ∆HPSv3 is the change in HPSv3 [31], a learned estimate of perceived image quality, from the first edited image (turn 1) to the final image. Measuring from turn 1 rather than from the source separates the quality lost over the chain from the effect of a single edit.

Preservation. On the whole image, the judge rates consistency with the source using the visual-consistency prompt. It sees the source, the final image and the chain’s ten instructions as one numbered list, and scores from 1 to 5 how well everything the instructions do not ask to change is preserved (5: all such elements unchanged; 1: most of the scene altered). As controls, on the first 20 EdiVal-10 chains it scores each source compared with itself 5, and each source compared with the final image of a different chain 1. Locally, we measure each chain’s anchor objects, which no instruction edits or mentions (Sec. 5.3). We crop the box of each anchor from the source and from the final image, grey out in both crops the parts that fall inside the box of any edited object, and compute the DINOv2 [35] distance (one minus the cosine similarity of the CLS features) between the two crops. Because the boxes are coarse, an anchor can lie largely inside the box of an edited object (e.g., a shirt worn by a person who is edited), leaving too little of it to compare. We therefore use an anchor only if at least 30% of its box remains after the masking, and take a chain’s anchor distance as the mean over its usable anchors; 148 of the 252 chains have at least one, and the anchor distance is averaged over these chains. We also compute LPIPS [60] between the source and the final image over the background, i.e., the whole image outside the boxes of all edited objects (a median of 57% of the image). Added objects are placed in this region, so the background distance should be read together with the instruction-following score.

Editing. A model can preserve content trivially by not editing, so we check that the requested edits are carried out and survive the chain. The judge applies the instructionfollowing prompt: it sees the source, the final image and the chain’s ten instructions as one numbered list, and scores from 1 to 5 whether the requested changes are present and correct (5: all present and accurate; 3: partially; 1: not at all), leaving unrequested changes to the consistency score. As a negative control, showing the source image instead of the final image yields a score of 1 for all 20 chains.

## C. Long Sequences with Repeated Content Replacement

To distinguish sequence length from long-term preservation demands, we consider a 40-edit sequence produced by the unmodified distilled FLUX.2 Klein 9B model (Fig. 7). Starting from a dog in a living room, we repeat four edits ten times: replace the dog with a cat, move the scene to a beach, replace the cat with a dog, and move the scene back to a living room. The subject and background are each replaced every two turns, so neither must retain its original details throughout the sequence. The outputs remain visually coherent through turn 40, without an obvious progressive loss of visual quality. However, the animal’s appearance and the room layout vary across cycles; returning to the same semantic categories does not imply recovery of the original image. This qualitative example shows that a long editing sequence need not exhibit progressive degradation when content is repeatedly replaced, and motivates evaluating stability in settings that require content to persist across many edits.

## D. Results on LME-Bench

Table 3 reports results on LME-Bench [61] (Sec. 5.3) in MT-OPSD’s format. We score it with MT-OPSD’s evaluation code, using Gemma-4-31B in place of GPT-4o as for our other judged metrics, and show every image to the judge at 512 pixels. A turn succeeds if the judge rates both its instruction following and its consistency above 6 out of 10; the success rate (SR) at turn k is the share of the 100 sessions whose turns 1 to k all succeed. The collapse rate (CR) at turn k is the share of sessions in which, by turn k, two consecutive turns have been judged destroyed (severe content loss or surface corruption relative to the source). MT-OPSD and its base model are sampled as in Sec. 5.1, without upsampling; we run our model and its base model with their own 4-step sampler, also generating at 512 pixels and passing each turn’s output image to the next turn. Absolute numbers depend on the judge: MT-OPSD reports an SR at turn 10 of 0.38 for its FLUX model under GPT-4o, against 0.10 under our judge, so Table 3 is not comparable with the numbers reported in [61].

Our model completes all ten turns in 53 of the 100 sessions, against 10 for MT-OPSD, and collapses in 2, against 16. Most of this gap comes from the backbone and its sampler, as our base model already completes 49 sessions. Compared with it, our model succeeds significantly more often at turns 5 and 8, but not at turn 10 (0.53 vs. 0.49), and the base model rarely collapses on this benchmark (0.04 at turn 10), leaving little room for improvement. At 1024 pixels, the resolution of our other experiments, the two are tied in every column (SR at turn 10: 0.53 vs. 0.46).

Table 3. LME-Bench (100 sessions of 10 turns; Gemma-4-31B judge, 512-pixel protocol). SR: share of sessions whose turns 1 to k all succeed; CR: share of sessions collapsed by turn k. <sup>∗</sup>: significantly different from ours (exact sign test over sessions, $p <$ 0.05).
<table><tr><td></td><td colspan="4">SR at turn k ↑</td><td colspan="4">CR at turn k ↓</td></tr><tr><td>Method</td><td>3</td><td>5</td><td>8</td><td>10</td><td>3</td><td>5</td><td>8</td><td>10</td></tr><tr><td>FLUX.2-klein-base-9B (undistilled) Base model</td><td>0.65*</td><td>0.41*</td><td>0.19*</td><td>0.09*</td><td>0.01</td><td>0.07</td><td>0.35*</td><td>0.47*</td></tr><tr><td>MT-OPSD [61]</td><td>0.64*</td><td>0.50*</td><td>0.27*</td><td>0.10*</td><td>0.03</td><td>0.03</td><td>0.06</td><td>0.16*</td></tr><tr><td>FLUX.2-klein-9B (4-step distilled)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base model Ours</td><td>0.89</td><td>0.79*</td><td>0.63*</td><td>0.49</td><td>0.02</td><td>0.02</td><td>0.04</td><td>0.04</td></tr></table>

## E. Single Edits

Every training rollout starts with an ordinary single edit (Sec. 4), but the training targets long sequences, so we check that stability does not come at the cost of single-edit quality. We apply each method once to every English instruction of GEdit-Bench [29]: 606 instructions over 604 photographs, in eleven edit types that include global edits such as style and tone changes. The photographs keep their aspect ratio. Our model, its base model and the training-free methods generate at the photograph’s resolution, capped at about one megapixel; MT-OPSD and its base model generate at about 512 × 512 pixels with the authors’ script, and their change is measured from the source at that resolution. We measure the change in HPSv3 from the source to the edited image, with the same HPSv3 settings as in Table 1, and average it over the instructions (Table 4).

Dog / living room  
Cat / living room  
Cat / beach  
Dog / beach  
Dog / living room  
![](images/9cf717d716c185e04227f0ef37c3f05cdbf9cc865c2c707ea865beea723a5cd5.jpg)  
Figure 7. Repeated content replacement over 40 edits. The unmodified distilled FLUX.2 Klein 9B model cycles from a dog in a living room to a cat in a living room, a cat on a beach, a dog on a beach, and a dog in a living room. We show the first, fifth, and tenth cycles of this sequential rollout. Visual quality remains coherent at late turns, while subject appearance and scene details change across cycles. Replacing both the subject and background every two turns limits long-term preservation demands, illustrating that sequence length alone does not necessarily lead to progressive visual degradation.

Table 4. Single edits on GEdit-Bench (606 English instructions, one edit each). Change in HPSv3 from the source to the edited image, averaged over instructions. <sup>∗</sup>: significantly different from our model (more than two standard errors, paired over instructions). <sup>†</sup>: VAE-LFA does not act on the first edit.
<table><tr><td>Method</td><td>∆HPSv3 (source→edit) ↑</td></tr><tr><td>FLUX.2-klein-base-9B (undistilled)</td><td></td></tr><tr><td>Base model</td><td>-0.68*</td></tr><tr><td>MT-OPSD [61]</td><td>-0.75*</td></tr><tr><td>FLUX.2-klein-9B (4-step distilled)</td><td></td></tr><tr><td>Base model</td><td>-0.41*</td></tr><tr><td>+ Emu Edit [42]</td><td>-0.56*</td></tr><tr><td>+ VAE-LFA [47]</td><td>same as base model†</td></tr><tr><td>+ FreqEdit [27]</td><td>-0.38*</td></tr><tr><td>Ours</td><td>-0.18</td></tr></table>

A single edit lowers HPSv3 for every method, and least for our model: by 0.18, against 0.41 for its base model (difference $0 . 2 3 \pm 0 . 0 2 )$ . Every other method also loses significantly more than our model, MT-OPSD the most (0.75). VAE-LFA aligns each output with earlier outputs and does not act on the first edit, so on single edits it is identical to the base model.

## F. Successive Background Changes

Figure 8 shows the example discussed in Sec. 5.2: three successive background changes, in which each training-free method carries part of the previous image into the next one.

FreqEdit  
MT-OPSD  
Emu Edit  
VAE-LFA  
![](images/fdab6add805455bc7f1bf14bbdaa12c6606db692ac6837ee549bf8ac0e48bf6d.jpg)  
Figure 8. Successive background changes. Each training-free method carries part of an earlier background into the next turn. The teddy bear, alarm clock and toy train of a synthetic scene are placed in a lavender field at sunset (edit 1), in front of a red brick wall (edit 2), and on a smooth gradient backdrop from warm cream to light grey (edit 3). Our model keeps the three objects and gives a clean gradient. Emu Edit keeps the low, warm light of the sunset in front of the brick wall (edit 2), and at edit 3 leaves a clump of lavender stalks on the floor of the gradient backdrop and turns the blue train grey. VAE-LFA keeps the colour of the sunset: at edit 3 the whole image has a purple-magenta cast, the bear turns reddish and the clock face pink. FreqEdit prints the pattern of the brick wall into the smooth gradient. MT-OPSD leaves no trace of the earlier backgrounds but blurs the image and washes out its colours.

![](images/0e8659d6986610e42c54cc858efc9fc3ef9e153e11690d861d5ff276f1afeed3.jpg)  
Figure 9. Additional multi-turn image editing results. We show selected turns from 20-step editing sequences. Each displayed instruction is followed by four no-op instructions (“keep the image as is”), and the images show the results after those no-ops. While the original FLUX.2 Klein accumulates color shifts and loses detail, fine-tuning it with our training paradigm (+Ours) yields a more stable editing process, preserving the scene and prior edits while following the requested changes.

Edit 5  
Edit 7  
Edit 6  
Edit 4  
Edit 2  
Edit 1  
Edit 3  
![](images/e92d3990424013bc01a183568b9110504aebea3e2e9c9885bd777184de6ad205.jpg)  
Figure 10. Fifteen consecutive semantic edits. We compare the original FLUX.2 Klein with our fine-tuned model (+Ours) over a sequence of 15 edits. Each result becomes the input to the next edit. Our method reduces accumulated drift in color and scene details while retaining the original model’s ability to follow diverse editing instructions, including object additions, removals, and pose and expression changes.