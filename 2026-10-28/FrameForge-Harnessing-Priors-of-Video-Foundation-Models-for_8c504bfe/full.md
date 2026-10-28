# FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation

Yijing Lin<sup>1</sup> Zhendong Mao<sup>1,</sup> Mengqi Huang<sup>1,2</sup>

Yuxin Wang<sup>2</sup> Yongdong Zhang<sup>1</sup>

<sup>1</sup>University of Science and Technology of China <sup>2</sup>FrameX.AI

Corresponding author: Zhendong Mao

lyijing@mail.ustc.edu.cn; zdmao@ustc.edu.cn; huangmq@ustc.edu.cn

wangyuxin@frame‑x.ai; zhyd73@ustc.edu.cn

Unified conditional image generation requires a single model to preserve diferent aspects ofa visual input while producing a target image that follows textual guidance. Pretrained video models provide a promising foundation by encoding visual transformations across time, yet existing approaches typically assign the condition and target to fixed temporal positions. We observe that condition‑target relationships vary substantially across generation tasks, while video models naturally encode visual transformations at diferent temporal scales. This motivates us to model conditional generation through input‑adaptive virtual temporal relations. We propose FrameForge, which captures two complementary aspects of such relations. Mixture ofTemporal Experts (MoTE) models howfar the target lies from the condition by dynamically selecting temporal RoPE experts with diferent virtual intervals. Chain of Temporal Reasoning (CoTR) models how the condition is transformed into the target through a latent intermediate state, with a reasoning‑bottleneck attention mask encouraging information propagation through this pathway. Experiments on DreamBench, COCO2017, and GEdit‑Bench show that FrameForge achieves superior performance in subject‑driven generation, canny‑to‑image generation, and image editing among the compared methods.

Project Page: lyne1.github.io/FrameForge‑page/

## 1 Introduction

With recent advances in generative modeling [8, 19, 21, 33, 35, 57, 67], foundation models such as Stable Difusion [13, 14, 41] and FLUX [25] ofer strong text‑to‑image generation capabilities. However, real‑world generation often involves diverse inputs (e.g., text, reference images, and structural maps) that impose constraints at diferent semantic and spatial levels. Modeling these heterogeneous condition‑target relations within a unified framework remains challenging.

Existing approaches to heterogeneous conditional image generation are predominantly image‑centric. Some train unified models on large‑scale multi‑task datasets [2, 9, 11, 42, 46, 53, 56], while others extend pretrained generators with control branches or lightweight adapters [47, 60, 61, 63, 65]. These approaches typically inject conditions as auxiliary spatial or semantic signals without explicitly modeling how strongly the condition and target difer or how one should be transformed into the other. This can make it dificult for a shared model to accommodate substantially diferent generation behaviors.

Pretrained video generation models provide a natural alternative by learning visual transformations across time. Recent studies exploit these priors by arranging the condition and target as diferent frames [6, 27, 31, 54]. However, existing approaches typically assign the condition and target to fixed temporal positions, resulting in an invariant temporal positional relation across diferent tasks and samples. Such a fixed ofset overlooks the variation in condition‑target relations and may limit a single shared adapter.

Our key observation is that heterogeneous condition‑target relations can be interpreted through a virtual temporal interval. In natural videos, frames separated by diferent temporal intervals exhibit diferent degrees of visual variation. A similar phenomenon arises in conditional image generation: image editing typically requires relatively small changes while preserving most source content, whereas subject‑driven and canny‑to‑image generation often involve larger variations in appearance, structure, or scene composition. As shown in Fig. 1, these tasks exhibit clearly diferent condition‑target LPIPS distributions, while their variation ranges substantially overlap with those observed in natural video transitions. This suggests that diferent conditional generation tasks may correspond to diferent regions of the temporal transformation prior learned by a video model. The broad distributions within individual tasks further suggest that a task label alone cannot specify the required transformation magnitude. An input‑dependent relation can account for both diferences across tasks and variations among samples within the same task. Therefore, instead of assigning all condition‑target pairs a fixed temporal ofset, we model their relation with an input‑adaptive virtual temporal interval.

This perspective reveals two complementary aspects of conditional generation: how far the target lies from the condition under the learned temporal prior, and how the condition should be transformed into the target. An adaptive interval addresses the former but does not uniquely determine the transformation. Modeling the latter requires a latent mechanism that learns condition‑to‑target transformations without explicit intermediate‑frame supervision.

To this end, we propose FrameForge, a unified conditional image generation framework built upon a pretrained video foundation model. To model how far the target lies from the condition, Mixture of Temporal Experts (MoTE) selects among temporal RoPE experts with diferent virtual intervals using the joint textual and visual context. To model how the condition evolves into the target, Chain of Temporal Reasoning (CoTR) inserts a noise‑initialized latent to form a condition‑reasoning‑target chain. The latent is progressively contextualized through attention with textual and visual tokens. During an intermediate training stage, a reasoning‑bottleneck attention

![](images/a787feae8890bc8cb6b98e27b053361a9a1b1b12126e206432824daa4ae587bb.jpg)  
Figure 1 Motivation of FrameForge. Top: The LPIPS distances between frame pairs sampled from natural videos at diferent temporal intervals. Bottom: The condition‑to‑target LPIPS distances for image editing, subject‑driven generation, and canny‑to‑image generation, where the dashed lines denote the median distance of each task.

mask prevents target tokens from directly accessing text or condition tokens, encouraging information propagation through the reasoning latent. The MoTE router and a single shared LoRA arejointly optimized across all tasks, while the pretrained video backbone remains frozen. This design enables heterogeneous conditional generation without task‑specific adapters.

Our main contributions are summarized as follows:

Conceptual contribution. We formulate heterogeneous conditional image generation from the perspective of input‑adaptive virtual temporal relations in pretrained video models. Our analysis shows that diferent generation tasks exhibit distinct condition‑target transformation magnitudes, motivating adaptive temporal relation modeling.

Technical contribution. We propose FrameForge, a unified adaptation framework that jointly models two complementary aspects ofconditional generation. MoTE adaptively determines howfar the target lies from the condition through input‑dependent temporal positional priors, while CoTR provides a latent pathway to model how the condition is transformed into the target without explicit intermediate supervision. The entire framework is jointly trained across tasks with a single shared LoRA.

Experimental contribution. We conduct extensive experiments on subject‑driven generation, canny‑to‑image generation, and instruction‑guided image editing. Results on DreamBench, COCO2017, and GEdit demonstrate that a single unified FrameForge model achieves superior performance across all three tasks compared with task‑specific and unified baselines. Comprehensive ablations further validate the efectiveness of the proposed MoTE and CoTR modules.

## 2 Related Work

## 2.1 Unified Image Generation

Conditional image generation includes structure‑guided synthesis [38, 63], subject‑driven generation [44, 60], and instruction‑guided editing [3, 62]. Unified models use shared representations and parameters to preserve spatial layout, subject identity, or content outside the requested edit.

One line of research develops native unified models through large‑scale multi‑task training. PixWizard [30] adapts a flow‑based difusion transformer with textual and visual conditions. OmniGen [56] represents text, conditions, and targets as a multimodal sequence, while UniReal [9] treats input and output images as discontinuous video frames. OmniGen2 [53] connects a vision‑language model to a difusion transformer through decoupled text and image pathways. BAGEL [11] extends unification to visual understanding through interleaved multimodal pretraining, and Lance [15] combines shared context modeling, separate capability pathways, and staged image/video training.

Another line adapts pretrained image generators. OminiControl [47] integrates condition‑image tokens into a DiT’s multimodal sequence and reuses its attention parameters for spatially aligned and non‑aligned controls. ACE++ [37] organizes references, masks, and target latents into unified context units. EasyControl [65] uses isolated condition branches for plug‑and‑play integration and zero‑shot composition, with position‑aware training and KV caching for flexible resolution and eficient inference. Concurrent work, TAG‑MoE [58], alleviates inter‑task interference through sparse FFN experts and routing aligned with hierarchical task semantics. Unlike TAG‑MoE, which routes image tokens to semantically specialized FFN experts, our MoTE retains shared feature‑processing parameters and selects among temporal RoPE experts to model an input‑dependent virtual temporal interval.

## 2.2 Video Foundation Model Adaptation

Video difusion models learn appearance and inter‑frame variation through motion modules, as in AnimateDif[17], or video difusion transformers, as in Latte [36], SANA‑Video [7], CogVideoX [59], Wan [49], and HunyuanVideo [23, 48]. Adapting these priors to image tasks requires relating the condition and target to the pretrained temporal structure.

RealGeneral [31] formulates conditional generation as two‑frame prediction with unified conditional embeddings, decoupled modulation, and attention masking. DRA‑Ctrl [6] uses mixup‑based shot transitions to bridge continuous video dynamics and discontinuous image pairs. iMontage [16] supports variable‑length many‑to‑many generation by treating images as pseudo‑frames and using Marginal RoPE to distinguish input and output sets from natural temporal ordering. Video4Edit [27] transfers intermediate evolution trajectories from a video‑pretrained teacher to a student editor. ChronoEdit [54] targets physically consistent editing by jointly denoising the target and temporal reasoning tokens, which are discarded during later denoising steps.

Beyond task‑specific adaptation, Wiedemer et al. [52] report emergent zero‑shot abilities. Moreover, Acuaviva et al. [1] examine these capabilities under controlled adaptation, providing evidence that video pretraining ofers transferable priors for structured visual transformations.

Existing studies demonstrate the potential of video models as general foundations for image‑level generation and visual reasoning. However, they are either tailored to individual tasks or fail to explicitly mitigate inter‑task interference within a shared parameter‑eficient adapter. Our framework addresses this issue by adapting the virtual temporal relation to each input and modeling the condition‑to‑target transformation.

## 3 Method

FrameForge adapts a pretrained video difusion model to conditional image generation through two complementary operations: selecting an input‑dependent temporal positional relation and learning an intermediate representation for condition‑to‑target prediction. We first introduce the preliminaries and framework overview in Sections 3.1 and 3.2, and then describe MoTE and CoTR in Sections 3.3 and 3.4. The training stages are detailed in Section 3.5.

![](images/618d08e4c845ad46f24b481e36ee2ef87a628eb5cdd59447d142fd170b63aef0.jpg)  
Figure 2 Overview of FrameForge. Heterogeneous conditional image generation is formulated as target‑frame prediction with a pretrained video difusion model. The MM‑DiT backbone processes text, condition, reasoning, and noisy target tokens to predict the target velocity. Mixture of Temporal Experts (MoTE) selects a temporal RoPE configuration at each denoising step, adapting the virtual interval between the condition and target. Chain of Temporal Reasoning (CoTR) adds a noise‑initialized intermediate latent whose hidden representation is updated through attention. During the intermediate training stage, a stochastic attention mask blocks direct text/condition‑target attention to encourage use of this intermediate pathway.

## 3.1 Preliminaries

Modern video generation models typically combine a 3D VAE with a difusion transformer to model visual content and temporal dynamics in a compressed latent space. We instantiate FrameForge with the first‑stage video difusion model of HunyuanVideo1.5 [48]. Its VAE encoder ℰ(⋅) compresses the spatial and

temporal dimensions by factors of 16 and $^ { 4 , }$ respectively. Given a video $\mathbf { x } \in \mathbb { R } ^ { ( 4 f + 1 ) \times 1 6 h \times 1 6 w \times c }$ , the encoded latent is

$$
\begin{array} { r } { \mathbf { z } _ { 0 } = \mathcal { E } ( \mathbf { x } ) \in \mathbb { R } ^ { ( f + 1 ) \times h \times w \times 3 2 } . } \end{array}\tag{1}
$$

The MM‑DiT jointly processes textual and visual tokens, with 3D rotary positional embeddings (3D‑RoPE) encoding the temporal and spatial coordinates of visual tokens. The backbone is trained with flow matching [32], where the clean latent $\mathbf { z } _ { 0 }$ is interpolated with Gaussian noise to obtain $\mathbf { z } _ { t }$ , and the corresponding velocity v is predicted by minimizing

$$
\mathcal { L } _ { \mathrm { d i f f u s i o n } } = \left. \mathbf { v } - \mathbf { v } _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , t , \mathbf { c } ) \right. _ { 2 } ^ { 2 } ,\tag{2}
$$

where c denotes the conditioning inputs.

## 3.2 Framework Overview

As shown in Fig. 2, FrameForge formulates conditional image generation as target‑frame prediction guided by textual and visual conditions. The target image is encoded and perturbed, while the visual conditions are encoded into $\mathbf { z } ^ { \mathrm { c o n d } }$ using the frozen VAE encoder. In addition, CoTR introduces a noise‑initialized reasoning latent $\mathbf { z } ^ { \mathrm { { r e a s o n } } }$ . Together with the text representations $\mathbf { z } ^ { \mathrm { t e x t } }$ , these inputs are jointly processed by the backbone, while MoTE determines their temporal positional configurations. The overall conditioning is expressed as

$$
\mathbf { c } ^ { \mathrm { { o u r s } } } = \left[ \mathbf { z } ^ { \mathrm { { t e x t } } } , \mathbf { z } ^ { \mathrm { { c o n d } } } , \mathbf { z } ^ { \mathrm { { r e a s o n } } } \right] .\tag{3}
$$

## 3.3 Mixture of Temporal Experts (MoTE)

The 3D‑RoPE in the video backbone encodes the temporal coordinates of visual tokens and thereby determines the positional prior for cross‑frame interaction. Under conditional frame prediction, using a fixed temporal gap assigns the same positional relation to all condition‑target pairs, despite their variations across tasks, samples, and denoising steps. To provide a more flexible temporal prior, we introduce Mixture of Temporal Experts (MoTE), which constructs multiple temporal RoPE experts corresponding to diferent virtual intervals and uses a lightweight router to select one expert according to the text, condition, and noisy target representations.

Temporal RoPE experts. We construct � temporal positional experts with candidate latent spans $\begin{array} { r l } { \mathcal { S } } & { { } = } \end{array}$ $\{ s _ { 1 } , s _ { 2 } , \ldots , s _ { E } \}$ , where $s _ { e }$ denotes the number of positions along the virtual latent temporal axis for expert �. Assigning the condition and target to the first and last positions yields a virtual temporal interval of $\Delta _ { e } = s _ { e } - 1$ . For a causal 3D VAE with temporal compression factor $r _ { t }$ , this latent span corresponds to an equivalent video‑frame span of $F _ { e } = r _ { t } ( s _ { e } - 1 ) + 1$ , including both endpoints. In our main implementation, $r _ { t } = 4 .$ , and the default video‑frame spans {17, 69, 121} correspond to latent spans {5, 18, 31}.

For expert $e ,$ we first construct the rotary embeddings over a virtual latent grid of size $( s _ { e } , h , w )$

$$
\left( \mathbf { R } _ { e } ^ { \mathrm { c o s } } , \mathbf { R } _ { e } ^ { \mathrm { s i n } } \right) = \mathrm { R o P E } \left( s _ { e } , h , w \right) .\tag{4}
$$

For condition‑target prediction, we retain the embeddings at temporal coordinates 0 and $s _ { e } - 1$ . With CoTR, we additionally retain the embeddings at an intermediate temporal position for the reasoning latent.

Sample‑wise Top‑1 routing. For each sample, the router takes the concatenated text, condition, and noisy target representations as input, and produces token‑level routing logits:

$$
\mathbf { A } = G \left( \left[ \mathbf { z } ^ { \mathrm { t e x t } } , \mathbf { z } ^ { \mathrm { c o n d } } , \mathbf { z } _ { t } ^ { \mathrm { t a r g e t } } \right] \right) \in \mathbb { R } ^ { L \times E } ,\tag{5}
$$

where � denotes the routing sequence length and � the number of temporal experts. We then perform masked average pooling over valid tokens to obtain a sample‑level logit vector a $\in \mathbb { R } ^ { E }$ , excluding padded text tokens.

The routing probabilities and Top‑1 expert assignment are given by

$$
{ \bf p } = \mathrm { S o f t m a x } \left( \frac { \bf a } { \tau _ { \mathrm { r o u t e r } } } \right) , \qquad e ^ { * } = \underset { e \in \{ 1 , \dots , E \} } { \arg \operatorname* { m a x } } p _ { e } ,\tag{6}
$$

where $\tau _ { \mathrm { r o u t e r } }$ controls the sharpness of the routing distribution. Moreover, to enable end‑to‑end optimization with discrete routing, we adopt a straight‑through estimator:

$$
\mathbf { g } = \mathrm { s g } \left( \mathbf { g } ^ { \mathrm { h a r d } } - \mathbf { p } \right) + \mathbf { p } ,\tag{7}
$$

where $\mathbf { g } ^ { \mathrm { h a r d } }$ is the one‑hot gate corresponding to the selected expert $e ^ { * }$ , and sg(⋅) denotes the stop‑gradient operation. Therefore, the forward pass uses a single selected expert, while gradients are propagated through the soft routing probabilities. At inference time, we directly set $\mathbf { g } = \mathbf { g } ^ { \mathrm { h a r d } }$

The routed rotary embeddings are then computed as

$$
\mathbf { R } ^ { \mathrm { c o s } } = \sum _ { e = 1 } ^ { E } g _ { e } \mathbf { R } _ { e } ^ { \mathrm { c o s } } , \qquad \mathbf { R } ^ { \mathrm { s i n } } = \sum _ { e = 1 } ^ { E } g _ { e } \mathbf { R } _ { e } ^ { \mathrm { s i n } } ,\tag{8}
$$

and applied to the visual queries and keys in self‑attention.

Load‑balancing regularization. To prevent routing collapse and encourage balanced utilization of the temporal experts, we introduce a batch‑level load‑balancing loss. Let $\bar { g } _ { e }$ denote the mean gate value of expert � over a batch of size �. The balancing loss is defined as

$$
\mathcal { L } _ { \mathrm { b a l a n c e } } = - \sum _ { e = 1 } ^ { E } \log \left( \bar { g } _ { e } + \epsilon \right) ,\tag{9}
$$

where � is a small constant for numerical stability.

The overall training objective is

$$
\mathcal { L } _ { \mathrm { f i n a l } } = \mathcal { L } _ { \mathrm { d i f f u s i o n } } + \lambda _ { \mathrm { b a l } } \mathcal { L } _ { \mathrm { b a l a n c e } } ,\tag{10}
$$

where $\lambda _ { \mathrm { b a l } }$ controls the strength of regularization.

## 3.4 Chain of Temporal Reasoning (CoTR)

Intermediate latent representation. CoTR introduces an intermediate latent state between the condition and target to model the implicit condition‑to‑target transformation. Specifically, we insert a single reasoning latent z<sup>reason</sup>, sampled from a standard Gaussian distribution with the same spatial and channel dimensions as the visual latents. Rather than being explicitly supervised or decoded, this latent is progressively contextualized through the MM‑DiT blocks by interacting with the text, condition, and target representations. It therefore serves as a latent pathway for aggregating conditional information before target prediction. The model is trained solely with the target‑frame objective in Eq. (10), without requiring intermediate supervision.

Reasoning‑bottleneck attention mask. Simply inserting the reasoning latent does not ensure that the model makes efective use of it, since target tokens can still directly attend to the text and condition tokens. To discourage this shortcut, we apply a stochastic reasoning‑bottleneck attention mask with probability $p _ { \mathrm { m a s k } }$ during the intermediate training stage.

Let �, �, ℛ, and � denote the text, condition, reasoning, and target token sets, respectively. The mask $\mathbf { M } ^ { \mathrm { c o t r } }$ is added to the self‑attention logits:

$$
\mathbf { M } _ { i j } ^ { \mathrm { c o t r } } = \left\{ \begin{array} { l l } { - \infty , } & { i \in \mathcal { T } \cup \mathcal { C } , ~ j \in \mathcal { X } , } \\ { - \infty , } & { i \in \mathcal { X } , ~ j \in \mathcal { T } \cup \mathcal { C } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{11}
$$

where � and � index query and key tokens, respectively.

The mask removes direct attention between the target and the text/condition tokens while preserving the reasoning latent as a communication bridge between them. Consequently, target tokens receive conditional information through the evolving reasoning representation rather than through a direct attention path. For samples without masking, the original full‑attention pattern is retained. This masking only modifies attention connectivity and does not afect the temporal positional conditioning introduced by MoTE.

## 3.5 Training Strategy

We freeze the pretrained video‑backbone weights and the VAE encoder, and insert a single LoRA adapter [20] into the attention and feed‑forward projections of the MM‑DiT.

Three‑stage optimization. In the first stage, we optimize only the MoTE router with the condition‑target formulation, keeping the LoRA parameters fixed and omitting the reasoning latent. This stage learns temporal routing before introducing the additional intermediate pathway. In the second stage, we add the reasoning latent, jointly optimize the router and LoRA, and apply the stochastic mask in Eq. (11). In the final stage, we retain both modules and continue joint optimization with full attention. The final stage matches the attention configuration used at inference, where the router selects a temporal expert at each denoising step. Furthermore, the shared adapter is used for every task.

## 4 Experiments

## 4.1 Implementation Details

Tasks. We evaluate FrameForge on three representative conditional image generation tasks: subject‑driven generation, canny‑to‑image generation, and instruction‑guided image editing. These tasks respectively assess subject consistency, structural fidelity, and semantic controllability, covering diverse condition‑to‑target relations within a unified framework.

Training data. For subject‑driven generation, we filter Subject200K [47] and UNO‑1M [55] to obtain high‑quality reference‑target subject pairs with variations in pose, viewpoint, and scene. For canny‑to‑image generation, we use a subset of 512‑2M [47], where the extracted edge map serves as the condition and the corresponding image as the target. For image editing, we sample instruction‑based editing pairs from the HQ‑Edit and UltraEdit subsets of GPT‑Image‑Edit‑1.5M [50]. We use 170K samples per task, resulting in 510K training samples in total. All images are resized to 512 × 512.

Benchmarks and metrics. For subject‑driven generation, we evaluate on DreamBench [44] using one reference image for each of 30 subjects and 25 prompts per subject. We adopt VIEScore [24] with GPT‑4o and report semantic consistency (��), perceptual quality (��), and overall score (�).

For canny‑to‑image generation, we use the 5,000‑image COCO2017 validation set [29], resized to $5 1 2 \times 5 1 2$ with text descriptions generated by GPT‑4o. We report FID [18], SSIM [51], and CLIP‑T [43] for distributional quality, structural fidelity, and text‑image alignment, respectively.

For image editing, we evaluate on the English subset of GEdit‑Bench [34], containing 606 samples across 11 editing categories. Following its oficial protocol, we report semantic consistency (�‑��), perceptual quality (�‑��), and overall score (�‑�) using GPT‑4.1. The �‑ prefix distinguishes GEdit‑Bench metrics from the VIEScore metrics used for subject‑driven generation.

Training and Inference. Our model, based on HunyuanVideo1.5‑T2V‑720P and fine‑tuned using LoRA with a rank and alpha of 32, has 182M trainable parameters. The model is trained for 12,000 iterations on two NVIDIA A800 GPUs, with a global batch size of 60 and a learning rate of $3 \times 1 0 ^ { - 4 }$ , requiring approximately 6 days. For the MoTE router, the softmax temperature is fixed to $\tau _ { \mathrm { r o u t e r } } = 1 . 0 $ , and the load‑balancing loss is weighted by $\lambda _ { \mathrm { b a l } } ~ = ~ 0 . 0 1$ . During the intermediate training stage, we apply the reasoning‑bottleneck attention mask with probability $p _ { \mathrm { m a s k } } ~ = ~ 0 . 5 ;$ the final stage and inference use full attention. During inference, we apply classifier‑free guidance to the text condition, with the guidance scale � set to 6 for subject‑driven generation and editing, and to 2 for canny‑to‑image generation. We use 50 denoising steps for all tasks.

## 4.2 Main Results

## 4.2.1 Subject‑driven image generation

Quantitative results As shown in Table 1, our method achieves the best overall score among all compared methods, reaching an � score of 0.739. It improves over ChronoEdit by 1.37% and outperforms OminiControl, ACE++, and EasyControl. Its subject consistency of 0.739 ranks second to ChronoEdit. Although EasyControl and IP‑Adapter achieve higher perceptual quality scores, their SC scores are substantially lower. Our method thus provides a better balance between subject preservation and image realism.

Table 1 Quantitative comparison on DreamBench for subject‑driven image generation
<table><tr><td>Method</td><td>Base Model</td><td>SC↑</td><td>PQ↑</td><td>0↑</td></tr><tr><td>BLIPDiffusion [26]</td><td>SD 1.5</td><td>0.574</td><td>0.696</td><td>0.613</td></tr><tr><td>SSR-Encoder [64]</td><td>SD 1.5</td><td>0.524</td><td>0.774</td><td>0.608</td></tr><tr><td>IP-Adapter [60]</td><td>SDXL</td><td>0.516</td><td>0.824</td><td>0.629</td></tr><tr><td>ACE++ [37]</td><td>FLUX.1-dev</td><td>0.660</td><td>0.779</td><td>0.688</td></tr><tr><td>EasyControl [65]</td><td>FLUX.1-dev</td><td>0.596</td><td>0.839</td><td>0.679</td></tr><tr><td>OmniGen [56]</td><td>Native</td><td>0.611</td><td>0.762</td><td>0.652</td></tr><tr><td>OminiControl [47]</td><td>FLUX.1-dev</td><td>0.680</td><td>0.816</td><td>0.719</td></tr><tr><td>EchoGen [12]</td><td>Infinity-2B</td><td>0.729</td><td>0.744</td><td>0.719</td></tr><tr><td>RealGeneral [31]</td><td>CogVideoX-1.5</td><td>0.633</td><td>0.767</td><td>0.669</td></tr><tr><td>DRA-Ctrl [6]</td><td>HunyuanVideo-I2V</td><td>0.609</td><td>0.800</td><td>0.656</td></tr><tr><td>ChronoEdit [54]</td><td>Wan2.1-I2V-14B</td><td>0.757</td><td>0.764</td><td>0.729</td></tr><tr><td>Ours</td><td>HunyuanVideo1.5</td><td>0.739</td><td>0.783</td><td>0.739</td></tr></table>

![](images/f9ce2ecef0a7b640bc8fcf7ac97ea0032ba90ce3b52153365f9a8f161e2a570f.jpg)  
Figure 3 Qualitative comparison on DreamBench for subject‑driven image generation. Given a reference subject and a text prompt, our method better preserves instance‑level visual attributes while adapting the subject to the prompt‑specified scene.

Qualitative results Fig. 3 illustrates the balance between subject preservation and text alignment. In the dog example, ChronoEdit places the subject partially within the pink fabric, making the requested “on top of” relation unclear. EasyControl changes the shoe design in the sneaker example. Our results preserve distinctive attributes while following the requested scenes and spatial relations, including placing the dog on top of the fabric.

## 4.2.2 Canny‑to‑image generation

Quantitative results As shown in Table 2, FrameForge achieves the highest SSIM of 0.447, corresponding to a relative improvement of 7.2% over the second‑best EasyControl. Its CLIP‑T score of 0.323 is comparable to ControlNet and RealGeneral, indicating improved structural fidelity with maintained text‑image alignment. Its FID of 23.47 is comparable to those of video‑based DRA‑Ctrl and RealGeneral.

These metrics capture complementary properties: SSIM measures image‑level structural similarity, whereas FID measures distributional diferences in Inception features. Backbone‑specific appearance statistics may therefore afect FID without necessarily implying inferior perceptual quality. In addition, prior studies also report disagreement with human judgments [22, 45]. Therefore, we consider FID together with SSIM, CLIP‑T, and qualitative results.

![](images/60c20fba6fca8aa7d590d51c9d4e76e11f6c66cb367815124ede4098b8d5acc0.jpg)  
Figure 4 Qualitative comparison on COCO2017 for canny‑to‑image generation. Our method more faithfully follows the input geometry while producing semantically consistent and visually coherent images.

Table 2 Quantitative results on COCO2017 for canny‑to‑image task
<table><tr><td>Method</td><td>Base Model</td><td>FID↓</td><td>SSIM↑</td><td>CLIP-T↑</td></tr><tr><td>ControlNet [63]</td><td>SD1.5</td><td>20.42</td><td>0.299</td><td>0.323</td></tr><tr><td>OmniGen [56]</td><td>Native</td><td>22.51</td><td>0.352</td><td>0.329</td></tr><tr><td>EasyControl [65]</td><td>FLUX.1-dev</td><td>22.82</td><td>0.417</td><td>0.319</td></tr><tr><td>RealGeneral [31]</td><td>CogVideoX-1.5</td><td>24.63</td><td>0.414</td><td>0.324</td></tr><tr><td>DRA-Ctrl [6]</td><td>HunyuanVideo-I2V</td><td>23.27</td><td>0.408</td><td>0.314</td></tr><tr><td>Ours</td><td>HunyuanVideo1.5</td><td>23.47</td><td>0.447</td><td>0.323</td></tr></table>

Qualitative results In Fig. 4, FrameForge preserves the poses, relative positions, and major contours specified by the edge maps while producing coherent images. Competing methods sometimes alter object geometry or layout. Although ControlNet achieves the lowest FID, its examples show deviations from the input edges, illustrating the complementary roles of distributional and structural metrics.

## 4.2.3 Image Editing

Quantitative results Table 3 compares FrameForge with recent instruction‑guided image editing methods on GEdit. We report results on both Motion Change and the average over all 11 categories. Existing image‑backbone adaptation methods perform poorly on Motion Change, with low �‑�� and �‑� scores despite competitive $G \mathrm { - } P Q$ in some cases. This highlights the importance of the dynamic visual priors learned by video foundation models.

Among open‑source methods, FrameForge achieves the best �‑� on both Motion Change and Overall GEdit, improving over ChronoEdit from 5.76 to 6.84 and from 5.08 to 6.12, respectively. Notably, its Motion Change �‑� is already comparable to closed‑source reference systems, surpassing Gemini 2.5 and Seedream 4.0 while approaching GPT‑4o.

Table 3 Quantitative comparison on GEdit for instruction‑guided image editing
<table><tr><td colspan="2">Method Backbone</td><td colspan="3">Motion Change</td><td colspan="3">Overall GEdit</td></tr><tr><td></td><td></td><td>G-SC↑</td><td>G-PQ↑</td><td>G-0↑</td><td>G-SC↑</td><td>G-PQ↑</td><td>G-0↑</td></tr><tr><td colspan="2">Closed-source</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 2.5 [10]</td><td>Native</td><td>6.87</td><td>7.79</td><td>6.72</td><td>8.25</td><td>8.29</td><td>7.89</td></tr><tr><td>GPT-40 [40]</td><td>Native</td><td>7.81</td><td>8.53</td><td>7.81</td><td>8.74</td><td>7.67</td><td>8.01</td></tr><tr><td>Seedream 4.0 [4]</td><td>Native</td><td>5.58</td><td>8.53</td><td>5.53</td><td>8.41</td><td>8.04</td><td>7.81</td></tr><tr><td colspan="2">Open-source</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ACE++ [37]</td><td>FLUX.1-dev</td><td>1.80</td><td>6.85</td><td>2.29</td><td>2.27</td><td>6.44</td><td>2.60</td></tr><tr><td>EditAR [39]</td><td>LlamaGen</td><td>2.65</td><td>6.05</td><td>2.91</td><td>4.26</td><td>5.70</td><td>3.89</td></tr><tr><td>ICEdit [66]</td><td>FLUX.1-dev</td><td>0.93</td><td>7.98</td><td>1.13</td><td>4.94</td><td>7.39</td><td>4.87</td></tr><tr><td>OmniGen [56]</td><td>Native</td><td>3.35</td><td>6.68</td><td>3.12</td><td>5.88</td><td>5.87</td><td>5.01</td></tr><tr><td>UniWorld-V1 [28]</td><td>Native</td><td>1.58</td><td>7.55</td><td>1.76</td><td>4.93</td><td>7.43</td><td>4.85</td></tr><tr><td>HiDream-I1 (E1) [5]</td><td>Native</td><td>1.58</td><td>7.23</td><td>1.66</td><td>5.66</td><td>6.06</td><td>5.01</td></tr><tr><td>ChronoEdit [54]</td><td>Wan2.1-I2V-14B</td><td>5.90</td><td>7.33</td><td>5.76</td><td>5.35</td><td>6.88</td><td>5.08</td></tr><tr><td>Ours</td><td>HunyuanVideo1.5</td><td>6.90</td><td>7.18</td><td>6.84</td><td>6.56</td><td>6.52</td><td>6.12</td></tr></table>

Qualitative results Fig. 5 presents qualitative comparisons across three representative editing types. FrameForge accurately follows the instructions while preserving the source identity and scene layout. Its advantage is particularly evident for motion and spatial edits, such as changing the cat into a sleeping pose and placing the balloon strip below the airplane, where competing methods often leave the requested change incomplete, mislocalize the added content, or introduce content drift.

![](images/84d52b6f9ce599fe2733aa0e4a164bc5a2fddbc2acdf380e65d0a98b9270a7e3.jpg)  
Figure 5 Qualitative comparison of image editing methods. The examples cover object addition, motion change, and style transfer. FrameForge more accurately follows the instructions while preserving the content unrelated to the requested edit.

## 4.2.4 MoTE Routing Analysis

Fig. 6 reports expert selection ratios over 50 denoising steps, averaged over 500 cases. The MoTE router exhibits distinct task‑dependent and timestep‑dependent preferences.

![](images/a8ed0baf552875fdb8d2bb273efceadbe00e9574e6a9cc4f6da84486baf94b00.jpg)  
Figure 6 Visualization of MoTE routing behavior for editing, subject‑driven generation, and canny‑to‑image. Gate 0, Gate 1, and Gate 2 denote the short‑, medium‑, and long‑interval temporal RoPE experts, respectively. Each curve reports the average expert usage ratio over 500 cases at each denoising step.

Canny‑to‑image generation, which has the largest condition‑target distance, predominantly selects the long‑interval expert, particularly during early denoising. Subject‑driven generation and editing initially favor the medium‑interval expert while target latents remain dominated by noise, with a higher selection ratio for subject‑driven generation. As target content emerges, both shift toward the short‑interval expert. Editing makes this transition earlier and more strongly, consistent with greater source‑content preservation. These within‑task shifts suggest that expert selection responds to the evolving target representation, rather than assigning each task a fixed temporal interval. These results complement the dataset‑level analysis in Fig. 1, showing that MoTE dynamically adapts the virtual temporal relation to both the task and denoising stage.

## 4.3 Ablations

Unless otherwise specified, ablations report overall scores for subject‑driven generation and editing (higher is better), and FID for canny‑to‑image generation (lower is better).

Main component ablation Table 4 compares task‑specific and shared LoRA adaptation. Joint training improves subject‑driven generation but degrades editing, suggesting greater task interference for editing. Adding MoTE or CoTR individually mainly improves editing, raising its score from 4.94 to 5.77 or 5.56, respectively, with small changes on the other tasks. When combined, the full model achieves the best overall performance, with the highest scores on subject‑driven generation and editing and the lowest FID on canny‑to‑image generation. This demonstrates that MoTE and CoTR are complementary in adapting the video difusion backbone to heterogeneous generation tasks.

Notably, neither module alone recovers the task‑specific editing score of 5.99, whereas their combination reaches 6.12 and also improves the other two tasks. This is consistent with temporal prior selection and intermediate transformation modeling addressing complementary requirements: adapting only one aspect does not yield the strongest joint performance in this comparison.

Number of experts As shown in Table 5, increasing the expert count from one to three improves canny‑to‑image generation and editing while maintaining the subject‑driven score. Five or seven experts slightly improve subject‑driven generation but degrade the other tasks. A larger pool may distribute supervision too sparsely across experts underjoint training. Therefore, we use three temporal RoPE experts for the best overall trade‑of.

Table 4 Main component ablation
<table><tr><td>Variant</td><td>LoRA</td><td>MoTE</td><td>CoTR</td><td>Subject ↑</td><td>Canny ↓</td><td>Editing ↑</td></tr><tr><td>Task-specific baseline</td><td>Task-specific</td><td>x</td><td>x</td><td>0.676</td><td>25.22</td><td>5.99</td></tr><tr><td>Unified baseline</td><td>Shared</td><td>x</td><td>x</td><td>0.722</td><td>25.78</td><td>4.94</td></tr><tr><td>+ MoTE</td><td>Shared</td><td>√</td><td>x</td><td>0.719</td><td>25.62</td><td>5.77</td></tr><tr><td>+ CoTR</td><td>Shared</td><td>x</td><td>√</td><td>0.726</td><td>25.94</td><td>5.56</td></tr><tr><td>Full Model</td><td>Shared</td><td>√</td><td>√</td><td>0.739</td><td>23.47</td><td>6.12</td></tr></table>

Here, we compare task‑specific LoRA with unified shared LoRA and evaluate the contributions of MoTE and CoTR.

Table 5 Ablation on the number of temporal RoPE experts in MoTE
<table><tr><td># Experts</td><td>Subject↑</td><td>Canny ↓</td><td>Editing ↑</td></tr><tr><td>1</td><td>0.739</td><td>25.90</td><td>5.90</td></tr><tr><td>3</td><td>0.739</td><td>23.47</td><td>6.12</td></tr><tr><td>5</td><td>0.743</td><td>24.24</td><td>5.96</td></tr><tr><td>7</td><td>0.743</td><td>24.55</td><td>5.67</td></tr></table>

CoTR is enabled and other settings are fixed. One expert uses virtual video‑frame indices 1 and 121. The CoTR‑only variant in Table 4 uses indices 1 and 17.

Virtual temporal intervals Table 6 compares candidate intervals through their equivalent video‑frame spans (� ). Short‑range experts achieve the highest subject‑driven score, while medium‑range experts achieve the best canny FID but degrade the other tasks. The wide‑range configuration achieves the best editing performance, while remaining comparable to the best results on subject‑driven and canny‑to‑image generation. Its subject‑driven score is only 0.002 below the best, supporting the use of {17, 69, 121} to cover diverse condition‑target relations.

Table 6 Ablation on virtual temporal interval configurations
<table><tr><td>Setting</td><td>Video Spans</td><td>Subject ↑</td><td>Canny ↓</td><td>Editing ↑</td></tr><tr><td>Short</td><td>{17,25, 33}</td><td>0.741</td><td>24.84</td><td>5.71</td></tr><tr><td>Medium</td><td>{13, 37,61}</td><td>0.716</td><td>23.19</td><td>5.42</td></tr><tr><td>Long</td><td>{17, 69, 121}</td><td>0.739</td><td>23.47</td><td>6.12</td></tr></table>

We fix the number of experts and vary their equivalent virtual video spans.

Number of reasoning latents Table 7 shows that introducing one reasoning latent improves all three tasks compared with using none. Two or three latents reduce performance, especially on subject‑driven generation and editing. Without direct intermediate supervision, additional latents may enlarge the under‑constrained transformation space and disperse the target training signal. The results indicate that one reasoning latent already provides suficient transformation capacity while retaining an efective information bottleneck. We therefore use a single reasoning latent by default.

## 4.4 Generalization Across Video Backbones

We further evaluate FrameForge on HunyuanVideo1.0 against a baseline trained with a shared LoRA across the three tasks. We report overall scores for subject‑driven generation and editing, and FID for canny‑to‑image generation.

Table 7 Ablation on the number of reasoning latents in CoTR
<table><tr><td># Reasoning Latents</td><td>Subject ↑</td><td>Canny ↓</td><td>Editing ↑</td></tr><tr><td>0</td><td>0.719</td><td>25.62</td><td>5.77</td></tr><tr><td>1</td><td>0.739</td><td>23.47</td><td>6.12</td></tr><tr><td>2</td><td>0.709</td><td>25.82</td><td>5.62</td></tr><tr><td>3</td><td>0.713</td><td>24.55</td><td>5.09</td></tr></table>

CoTR uses one reasoning latent by default.

As shown in Table 8, FrameForge consistently outperforms the unified baseline across all three tasks. It raises the subject‑driven score from 0.699 to 0.736 and the editing score from 4.38 to 5.00, while reducing canny FID from 24.52 to 23.07. These improvements support the applicability of the proposed adaptation strategy beyond the main backbone.

Table 8 Evaluation on HunyuanVideo1.0
<table><tr><td>Method</td><td>Subject ↑</td><td>Canny ↓</td><td>Editing ↑</td></tr><tr><td>Unified baseline</td><td>0.699</td><td>24.52</td><td>4.38</td></tr><tr><td>FrameForge</td><td>0.736</td><td>23.07</td><td>5.00</td></tr></table>

## 5 Conclusion

In this work, we presented FrameForge, a unified adaptation framework that leverages the temporal priors of pretrained video foundation models for heterogeneous conditional image generation. Rather than assigning a fixed temporal relation to all condition‑target pairs, we introduced an input‑adaptive virtual temporal relation perspective. Based on this formulation, MoTE selects an appropriate temporal positional prior from multiple interval experts according to the textual and visual context. CoTR introduces an unsupervised reasoning latent and a reasoning‑bottleneck attention mask to model the unobserved condition‑to‑target transformation. The proposed framework enables joint adaptation across subject‑driven generation, canny‑to‑image generation, and image editing while keeping the video foundation model frozen. Experiments on DreamBench, COCO2017, and GEdit demonstrate strong performance across all three tasks, and the ablation studies verify the efectiveness of both adaptive temporal relation modeling and implicit temporal reasoning.

## References

[1] Pablo Acuaviva, Aram Davtyan, Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Alexandre Alahi, and Paolo Favaro. Rethinking visual intelligence: Insights from video pretraining. In Forty‑third International Conference on Machine Learning, 2025.

[2] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan L Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22861–22872, 2024.

[3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.

[4] ByteDance Seed. Seedream 4.0. https://seed.bytedance.com/en/seedream4\_0, 2025.

[5] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream‑i1: A high‑eficient image generative foundation model with sparse difusion transformer. arXiv preprint arXiv:2505.22705, 2025.

[6] Hengyuan Cao, Yutong Feng, Biao Gong, Yijing Tian, Yunhong Lu, Chuang Liu, and Bin Wang. Dimension‑reduction attack! video generative models are experts on controllable image synthesis. Advances in Neural Information Processing Systems, 38:81329–81358, 2026.

[7] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana‑video: Eficient video generation with block linear difusion transformer. arXiv preprint arXiv:2509.24695, 2025.

[8] Lifeng Chen, Jiner Wang, Zihao Pan, Beier Zhu, Xiaofeng Yang, and Chi Zhang. Detail++: Training‑free detail enhancer for t2i difusion models. IEEE Transactions on Image Processing, 2026.

[9] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. Unireal: Universal image generation and editing via learning real‑world dynamics. arXivpreprint arXiv:2412.07774, 2024.

[10] Google DeepMind. Gemini 2.5. https://deepmind.google/technologies/gemini/, 2025.

[11] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

[12] Ruixiao Dong, Zhendong Wang, Keli Liu, Li Li, Ying Chen, Kai Li, Daowen Li, and Houqiang Li. Echogen: Generating visual echoes in any scene via feed‑forward subject‑driven auto‑regressive model. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=ctmyCjo18u.

[13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high‑resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.

[14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high‑resolution image synthesis, 2024. URL https://arxiv.org/abs/2403.03206.

[15] Fengyi Fu, Mengqi Huang, Shaojin Wu, Yunsheng Jiang, Yufei Huo, Hao Li, Yinghang Song, Fei Ding, Jianzhu Guo, Qian He, Zheren Fu, Zhendong Mao, and Yongdong Zhang. Lance: Unified multimodal modeling by multi‑task synergy, 2026. URL https://arxiv.org/abs/2605.18678.

[16] Zhoujie Fu, Xianfang Zeng, Jinghong Lan, Xinyao Liao, Cheng Chen, Junyi Chen, Jiacheng Wei, Wei Cheng, Shiyu Liu, Yunuo Chen, Gang Yu, and Guosheng Lin. imontage: Unified, versatile, highly dynamic many‑to‑many image generation. arXiv preprint arXiv:2511.20635, 2025.

[17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatedif: Animate your personalized text‑to‑image difusion models without specific tuning. International Conference on Learning Representations, 2024.

[18] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time‑scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen‑Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low‑rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[21] Mengqi Huang, Zhendong Mao, Zhuowei Chen, and Yongdong Zhang. Towards accurate image coding: Improved autoregressive image generation with dynamic vector quantization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22596–22605, 2023.

[22] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9307–9315, June 2024.

[23] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

[24] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation, 2023.

[25] Black Forest Labs. Flux: Oficial inference repository for flux.1 models. https://github.com/black‑forest‑labs/flux, 2024. Accessed: 2024‑11‑12.

[26] Dongxu Li, Junnan Li, and Steven Hoi. Blip‑difusion: Pre‑trained subject representation for controllable text‑to‑image generation and editing. Advances in Neural Information Processing Systems, 36:30146–30166, 2023.

[27] Xiaofan Li, Yanpeng Sun, Chenming Wu, Fan Duan, YuAn Wang, Weihao Bo, Yumeng Zhang, and Dingkang Liang. Video4edit: Viewing image editing as a degenerate temporal process, 2025. URL https://arxiv.org/abs/2511.18131.

[28] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld‑v1: High‑resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.

[29] Tsung‑Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6‑12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.

[30] Weifeng Lin, Xinyu Wei, Renrui Zhang, Le Zhuo, Shitian Zhao, Siyuan Huang, Junlin Xie, Yu Qiao, Peng Gao, and Hongsheng Li. Pixwizard: Versatile image‑to‑image visual assistant with open‑language instructions. arXiv preprint arXiv:2409.15278, 2024.

[31] Yijing Lin, Mengqi Huang, Shuhan Zhuang, and Zhendong Mao. Realgeneral: Unifying visual generation via temporal in‑context learning with video models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14994–15004, 2025.

[32] Yaron Lipman, Ricky TQ Chen, Heli Ben‑Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

[33] Chang Liu, Rui Li, Kaidong Zhang, Yunwei Lan, Xin Luo, and Dong Liu. Lacon: Late‑constraint controllable visual generation. IEEE Transactions on Image Processing, 2026.

[34] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x‑edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

[35] Tao Liu, Kai Wang, Senmao Li, Joost van de Weijer, Fahad Shahbaz Khan, Shiqi Yang, Yaxing Wang, Jian Yang, and Ming‑Ming Cheng. One‑prompt‑one‑story: Free‑lunch consistent text‑to‑image generation using a single prompt. arXiv preprint arXiv:2501.13554, 2025.

[36] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan‑Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent difusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024.

[37] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction‑based image creation and editing via context‑aware content filling. arXiv preprint arXiv:2501.02487, 2025.

[38] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i‑adapter: Learning adapters to dig out more controllable ability for text‑to‑image difusion models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 4296–4304, 2024.

[39] Jiteng Mu, Nuno Vasconcelos, and Xiaolong Wang. Editar: Unified conditional generation with autoregressive models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 7899–7909, 2025.

[40] OpenAI. Gpt‑4o system card, 2024. URL https://arxiv.org/html/2410.21276v1.

[41] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent difusion models for high‑resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

[42] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, et al. Unicontrol: A unified difusion model for controllable visual generation in the wild. arXiv preprint arXiv:2305.11147, 2023.

[43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text‑to‑image difusion models for subject‑driven generation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.

[45] George Stein, Jesse Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L Caterini, Eric Taylor, and Gabriel Loaiza‑Ganem. Exposing flaws ofgenerative model evaluation metrics and their unfair treatment of difusion models. Advances in Neural Information Processing Systems, 36:3732–3784, 2023.

[46] Zeyi Sun, Ziyang Chu, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. X‑prompt: Towards universal in‑context image generation in auto‑regressive vision language foundation models. arXiv preprint arXiv:2412.01824, 2024.

[47] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for difusion transformer. arXiv preprint arXiv:2411.15098, 2024.

[48] Tencent Hunyuan Foundation Model Team. Hunyuanvideo 1.5 technical report, 2025. URL https://arxiv.org/abs 2511.18870.

[49] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen‑Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi‑Fan Wu, and Ziyu Liu. Wan: Open and advanced large‑scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[50] Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. Gpt‑image‑edit‑1.5m: A million‑scale, gpt‑generated image dataset, 2025. URL https://arxiv.org/abs/2507.21033.

[51] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

[52] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero‑shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

[53] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

[54] Jay Zhangjie Wu, Xuanchi Ren, Tianchang Shen, Tianshi Cao, Kai He, Yifan Lu, Ruiyuan Gao, Enze Xie, Shiyi Lan, Jose M. Alvarez, Jun Gao, Sanja Fidler, Zian Wang, and Huan Ling. Chronoedit: Towards temporal reasoning for image editing and world simulation. arXiv preprint arXiv:2510.04290, 2025.

[55] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less‑to‑more generalization: Unlocking more controllability by in‑context generation. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 18682–18692, October 2025.

[56] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13294–13304, 2025.

[57] Xingqian Xu, Jiayi Guo, Zhangyang Wang, Gao Huang, Irfan Essa, and Humphrey Shi. Prompt‑free difusion: Taking” text” out of text‑to‑image difusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8682–8692, 2024.

[58] Yu Xu, Hongbin Yan, Juan Cao, Yiji Cheng, Tiankai Hang, Runze He, Zijin Yin, Shiyi Zhang, Yuxin Zhang, Jintao Li, Chunyu Wang, Qinglin Lu, Tong‑Yee Lee, and Fan Tang. Tag‑moe: Task‑aware gating for unified generative mixture‑of‑experts, 2026. URL https://arxiv.org/abs/2601.08881.

[59] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text‑to‑video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

[60] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip‑adapter: Text compatible image prompt adapter for text‑to‑image difusion models. arXiv preprint arXiv:2308.06721, 2023.

[61] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high‑quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024.

[62] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction‑guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.

[63] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text‑to‑image difusion models. In Proceedings ofthe IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.

[64] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr‑encoder: Encoding selective subject representation for subject‑driven generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024.

[65] Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. Easycontrol: Adding eficient and flexible control for difusion transformer. arXiv preprint arXiv:2503.07027, 2025.

[66] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In‑context edit: Enabling instructional image editing with in‑context generation in large‑scale difusion transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2504.20690.

[67] Mengyao Zhao, Hao Xiong, Weiqing Min, Sujuan Hou, Mengmeng Zhang, and Shuqiang Jiang. Condfoodgen: A conditional two‑stream network for controllable food image generation. IEEE Transactions on Image Processing, 2026.