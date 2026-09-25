# OREO: Fidelity Alignment in 3D Generation via On-the-fly Rendering-Editing Optimization

Zhiyuan Ma<sup>1,2</sup>, Wenbo Hu<sup>2†</sup>, Wang Zhao<sup>2</sup>, Pengfei Wang<sup>1</sup>, Ying Shan<sup>2</sup>, and Lei Zhang<sup>1†</sup>

<sup>1</sup> The Hong Kong Polytechnic University <sup>2</sup> Tencent ARC Lab

Abstract. Despite recent advancements in 3D generation, models often struggle to produce assets with high visual fidelity. To bridge this gap, we propose OREO, an alignment framework that enhances the realism of 3D generators by leveraging rich 2D difusion priors. Instead of relying on static datasets, OREO establishes a dynamic optimization loop that produces on-the-fly edited renderings as 2D pseudo-targets. At its core, we introduce Reinforced Editing, which utilizes a 2D model to refine rendered views of the 3D output, enhancing their overall visual fidelity while preserving the underlying geometry, viewpoint, and content. These refined views serve as high-quality supervision targets, enabling the 3D generator to learn from its own generated samples and progressively improve its visual quality. Experiments demonstrate that OREO efectively improves upon pre-trained baselines, producing 3D assets with enhanced visual realism. Our project page is at https: //theericma.github.io/oreo/.

Keywords: 3D Generation · Alignment · Image Difusion Prior

## 1 Introduction

Recent 3D generators, such as Trellis [38] and the Hunyuan3D series [13, 16, 43, 51], have greatly simplified the 3D content creation process. However, despite their strong performance in geometry generation, achieving high visual fidelity remains challenging. Generated assets often lack intricate textures and fine-grained details observed in real-world subjects. This gap becomes more pronounced for imaginative and out-of-distribution references, where existing 3D priors provide limited appearance coverage. This limitation largely stems from the limited scale and appearance diversity of high-quality 3D data compared with billion-scale 2D imagery. Consequently, 3D generative models may not fully capture the visual detail needed for photorealism.

To bridge this gap, we propose to distill the rich visual priors of pre-trained 2D difusion models into 3D generators. These models encapsulate a deep understanding of visual fidelity and fine-grained details, far surpassing the scope of existing 3D datasets. To this end, we introduce the On-the-fly Rendering Editing Optimization (OREO) framework, as illustrated in Fig. 1. The core idea is to establish a dynamic alignment mechanism: instead of relying solely on static 3D training data, the generator improves through supervision generated on the fly. Specifically, OREO implements a Render-Edit-Optimize loop, where the model renders views from its current output, receives corrective edits to enhance realism, and uses these refined views as supervision targets.

![](images/e49ad17d0490c7b59e1309b3f08201a4ec10386416f25e1aeee79b10bbc9b409.jpg)  
Fig. 1: Teaser. To enhance visual fidelity of 3D generation, our framework first reinforces rendered views via image editing, then distills the gap back to the training generator with a latent contrastive objective. Detailed workflow in Fig. 2.

At the core of this loop is the Reinforced Editing algorithm, which generates edited views as supervision signals. To ensure reliable guidance, Reinforced Editing prioritizes structure preservation, focusing on improving visual fidelity while maintaining the original spatial layout. Inspired by recent inversion-free editing techniques, this approach mitigates the unwanted geometric deviations common in standard editing methods, helping the edited renderings better preserve the viewpoint and structure of the rendered source. To turn these refined targets into efective supervision, we propose Contrastive Distillation, which treats the edited view as a positive and the original rendering as a negative, and distills the fidelity gap into the 3D generator through a latent-space contrastive objective. Experiments on Trellis show that OREO produces 3D assets with improved visual fidelity. In summary, our contributions are as follows:

1. We introduce OREO, a novel alignment framework that leverages pre-trained 2D difusion priors to enhance the fidelity of 3D generators through an onthe-fly optimization loop with dynamic 2D pseudo-targets.

2. At its core, we propose Reinforced Editing, an image editing algorithm that improves the visual fidelity of rendered views while preserving structure, coupled with Contrastive Distillation that distills the visual changes between edited and original renderings into the 3D generator for post-training.

3. We validate the efectiveness of OREO on Trellis, a state-of-the-art 3D generator. Results show that OREO efectively improves visual fidelity and appearance details over the pre-trained baseline.

## 2 Related Work

Learning-based 3D Generation [26, 42] and Generator Post-training. Recent learning-based 3D generation [5, 8, 9, 17–20, 23–25, 35, 38, 40, 41, 44, 49, 51] synthesizes assets feed-forward in latent spaces, but reliance on synthetic data such as Objaverse [6] caps visual fidelity and texture detail. Photo3D [21] performs ofline detail enhancement on Trellis, but its edits do not explicitly preserve source-view structure. OREO instead transfers 2D appearance priors online through structure-aligned edited renderings.

2D Image Editing for Fidelity-Enhancing Supervision. Efective supervision requires edits that enhance fidelity while preserving viewpoint and structure. Training-based image editors [1, 32, 48, 50] inject appearance details but do not constrain viewpoint or layout, while inversion-based methods [11, 27, 31] reconstruct an input before editing it. Reinforced Editing builds on FlowEdit [15], which couples source and target trajectories to edit images without inversion.

2D Priors for 3D Editing and Generation. Although 2D priors have been explored for 3D editing and generation, existing methods do not directly meet the goal of post-training a feed-forward 3D generator. On the one hand, methods such as Instruct-NeRF2NeRF [10], DreamEditor [52], GaussianEditor [34], DFF-Splat [14], Image Sculpting [45], MvDrag3D [2], and Omni-3DEdit [4] edit individual instances or scenes, rather than post-training a generalizable 3D generator. GeoDifusion [3] instead uses geometric conditions to control image generation for object detection data. On the other hand, score-distillation methods such as DreamFusion [28], Magic3D [22], and ProlificDreamer [36] are optimizationbased text-to-3D methods whose input setting and inference pipeline difer from those used by OREO for learning-based image-to-3D generator post-training, making them unsuitable for direct quantitative comparison with our framework. DMD [46] is closer to a generator-learning paradigm and is therefore retained as a reasonable score-distillation ablation baseline; however, it still relies on implicit score-gradient supervision. OREO uses high-fidelity, structure-aligned edited renderings as explicit 2D pseudo-targets to provide stable online supervision for 3D generator post-training.

## 3 Method

We present OREO, an alignment framework that enhances 3D visual fidelity via on-the-fly optimization. An overview of the full pipeline is illustrated in Fig. 2. We leverage a pre-trained 2D editing model to construct edited renderings as 2D pseudo-targets on the fly, enabling generator post-training without paired 3D ground-truth supervision. Each iteration rolls out the generator to obtain a 3D asset and renders it from a sampled camera pose. Building on the flowmatching and inversion-free editing formulation in Sec. 3.2, Reinforced Editing in Sec. 3.3 enhances the view while preserving structure. Sec. 3.4 then distills this improvement into the 3D generator.

![](images/50e076eb5138ac15491bff34cbbeaa1cf3a921caf42bd0c6c775343bb54fc579.jpg)  
Fig. 2: Overview of OREO. The training generator produces a 3D asset via onpolicy ODE rollout that is rendered into a source view $x ^ { \mathsf { s r c } }$ . Reinforced Editing turns $x ^ { \mathsf { s r c } }$ into a high-fidelity target $x ^ { \tt t g t }$ , and Contrastive Distillation uses $( x ^ { \mathrm { { s r c } } } , x ^ { \mathrm { { t g t } } } )$ as a negative/positive pair to supervise the generator in the latent space.

## 3.1 Problem Formulation

Given a reference image $x ^ { \mathrm { r e f } }$ , the 3D latent generator $G _ { \theta }$ followed by a decoder $\mathcal { D }$ produces a 3D asset $\mathcal { A } = \mathcal { D } ( G _ { \theta } ( x ^ { \tt r e f } ) )$ ). A renderer $\mathcal { P }$ then projects A into a 2D view $x ^ { \mathrm { s r c } } = \mathcal { P } ( \mathcal { A } , \pi )$ under camera pose π. In our training framework, a pre-trained 2D editor $\mathcal { E } _ { \phi }$ then enhances this view into a higher-fidelity target $x ^ { \mathrm { t g t } } = \mathcal { E } _ { \phi } ( x ^ { \mathrm { s r c } } , x ^ { \mathrm { r e f } } )$ . We optimize the generator by minimizing this discrepancy:

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \mathbb { E } _ { x ^ { \mathrm { r e f } } , \pi } \left[ \mathcal { L } ( x ^ { \mathrm { s r c } } , x ^ { \mathrm { t g t } } ) \right]\tag{1}
$$

where $\mathcal { L }$ is a supervision loss that encourages improved visual fidelity.

## 3.2 Preliminaries

Flow Matching Framework and Notation. Flow Matching models generation as an ODE process that transports Gaussian noise to the data distribution:

$$
d x _ { t } = v ( x _ { t } ; t , c ) d t , \quad t : 1 \to 0\tag{2}
$$

where $x _ { t }$ is the latent state at time $t ,$ and c is the condition $( \mathrm { e . g . }$ , text or image). For better alignment to the condition c, Classifier-Free Guidance (CFG) is applied to the velocity field, using a null condition ∅ and a guidance scale w:

$$
v ^ { w } ( x _ { t } ; t , c ) = v ( x _ { t } ; t , \emptyset ) + w \cdot \left( v ( x _ { t } ; t , c ) - v ( x _ { t } ; t , \emptyset ) \right)\tag{3}
$$

Algorithm 1 Reinforced Editing   
Require: Rendered view $x ^ { \mathsf { s r c } }$ , Reference condition $c ^ { \mathrm { r e f } }$ , Editor $\mathcal { E } _ { \phi }$ , Guidance scale $w ,$   
Editing steps $N _ { e } ,$ Total steps N   
Ensure: Edited rendering pseudo-target $x ^ { \tt t g t }$   
1: Initialize $x ^ { \mathsf { e d i t } }  x ^ { \mathsf { s r c } } ,$ , Sample noise $\epsilon \sim \mathcal { N } ( 0 , I )$   
2: Select time steps $\{ t _ { i } \} _ { i = 0 } ^ { N _ { e } }$ as the last $N _ { e }$ steps from the full schedule of N steps   
3: for $i = 0$ to $N _ { e } - 1$ do \triangleright FlowEdit: Reverse-time Euler Integration   
4: $t \gets t _ { i } , \Delta t \gets t _ { i } - t _ { i + 1 }$   
5: $x _ { t } ^ { \mathrm { s r c } } \gets ( 1 - t ) x ^ { \mathrm { s r c } } + t \epsilon$   
6: $x _ { t } ^ { \mathrm { { t g t } } }  x ^ { \mathrm { { e d i t } } } + ( x _ { t } ^ { \mathrm { { s r c } } } - x ^ { \mathrm { { s r c } } } )$ \triangleright Eq. 6   
// Reinforced Editing (Sec. 3.3)   
7: $\tilde { v }  v _ { \phi } ^ { w } ( x _ { t } ^ { \mathrm { t g t } } ; t , c ^ { \mathrm { r e f } } )  v _ { \phi } ( x _ { t } ^ { \mathrm { s r c } } ; t , \emptyset )$ \triangleright Eq. 7   
8: $x ^ { \mathrm { e d i t } } \gets x ^ { \mathrm { e d i t } } - \tilde { v } \varDelta t$ \triangleright FlowEdit: Euler Step   
// Noise Update   
9: $\epsilon  \epsilon - \overline { { ( v _ { \phi } ( x _ { t } ^ { \tt t g t } ; t , c ^ { \tt r e f } ) - v _ { \phi } ( x _ { t } ^ { \tt t g t } ; t , \emptyset ) ) \cdot ( 1 - t ) } }$ \triangleright Eq. 8   
10: end for   
11: return $x ^ { \mathrm { { t g t } } }  x ^ { \mathrm { { e d i t } } }$

We estimate the clean sample $\scriptstyle { \hat { x } } _ { 0 }$ from the intermediate state $x _ { t }$ in one step:

$$
\hat { x } _ { 0 } = x _ { t } - t \cdot v ( x _ { t } ; t , c )\tag{4}
$$

To distinguish modalities, we use z and θ for the 3D generator $v _ { \theta } .$ , where $G _ { \theta }$ denotes its ODE sampling process and the pre-trained frozen weights are denoted $v _ { \theta _ { p r e } }$ . Similarly, for the 2D editor $v _ { \phi }$ , we use x for its latent state and $\phi$ for its parameters; $\mathcal { E } _ { \phi }$ denotes the multi-step editing process described below.

Inversion-Free Image Editing. We leverage FlowEdit [15], an inversionfree method that modifies images by coupling source and target flow trajectories. Starting from $x ^ { \mathsf { s r c } }$ , it updates $\boldsymbol { x } ^ { \mathrm { e d i t } } \gets \boldsymbol { x } ^ { \mathrm { e d i t } } - \boldsymbol { \varDelta t } \cdot \boldsymbol { \tilde { v } } ,$ where $\varDelta t = t _ { i } - t _ { i + 1 } > 0$ for the decreasing ODE schedule, with diferential velocity:

$$
\tilde { v } = v ^ { \mathrm { t g t } } ( x _ { t } ^ { \mathrm { t g t } } ; t , c ^ { \mathrm { t g t } } ) - v ^ { \mathrm { s r c } } ( x _ { t } ^ { \mathrm { s r c } } ; t , c ^ { \mathrm { s r c } } )\tag{5}
$$

where the source branch and the target branch are constructed as

$$
x _ { t } ^ { \mathrm { s r c } } = ( 1 - t ) x ^ { \mathrm { s r c } } + t \epsilon , \quad x _ { t } ^ { \mathrm { t g t } } = x ^ { \mathrm { e d i t } } + ( x _ { t } ^ { \mathrm { s r c } } - x ^ { \mathrm { s r c } } ) , \quad \epsilon \sim \mathcal { N } ( 0 , I ) .\tag{6}
$$

The editing process is performed over the last $N _ { e }$ steps of the full N-step ODE schedule. These steps evolve $\scriptstyle { x ^ { \mathsf { e d i t } } }$ into $x ^ { \mathrm { t g t } }$ , transferring the image from $\boldsymbol { c } ^ { \mathrm { s r c } }$ to $c ^ { \tt t g t }$ while preserving structure without costly inversion to noise.

## 3.3 On-the-fly 2D Pseudo-Targets via Reinforced Editing

A critical step in our pipeline is transforming the rendered view $x ^ { \mathsf { s r c } }$ into an edited rendering $x ^ { \mathrm { t g t } }$ , which serves as a 2D pseudo-target for appearance supervision. Existing image editing models [37] can leverage their learned priors to make rendered views more realistic, but often alter the camera perspective or object pose, as analyzed in Sec. 4.2. We propose Reinforced Editing to enable structure-preserving enhancement on top of pre-trained editing models. Diferent from the original FlowEdit [15] that transfers between separate conditions $c ^ { \mathsf { s r c } }$ and $c ^ { \tt t g t }$ in Eq. 5, our goal is to enhance the rendered view to better resemble a novel view of $x ^ { \mathrm { r e f } }$ , where no independent $c ^ { \mathsf { s r c } }$ exists. We thus guide the target branch toward $c ^ { \tt r e f } = c ^ { \tt t g t }$ and replace the source branch with an unconditional prediction:

Algorithm 2 Closed-loop Optimization via Dynamic 2D Pseudo-Targets   
Require: Generator $v _ { \theta }$ , frozen pretrained velocity $v _ { \theta _ { p r e } }$ , 2D editor $\mathcal { E } _ { \phi } .$ learning rate η   
Ensure: Optimized parameters $\theta ^ { * }$   
1: while not converged do   
2: Given reference image $\boldsymbol { x } ^ { \mathrm { r e f } }$   
3: $z _ { 0 } = G _ { \theta } ( x ^ { \mathrm { r e f } } )$ \triangleright Rollout: Generate clean latent   
4: Sample camera pose π   
5: $x ^ { \mathrm { s r c } } = \mathcal { P } ( \mathcal { D } ( z _ { 0 } ) , \pi )$ \triangleright Render: Decode and project   
6: Obtain $x ^ { \mathrm { { t g t } } }$ via Algorithm 1 from $x ^ { \mathsf { s r c } }$ and $x ^ { \mathrm { r e f } }$ \triangleright Reinforced Editing   
7: Sample $t \sim \mathcal { U } ( 0 , 1 )$   
8: $z _ { t } = ( 1 - t ) z _ { 0 } + t \epsilon , \ \epsilon \sim \mathcal { N } ( 0 , I )$ \triangleright Noise Injection   
9: Predict $\hat { z } _ { 0 }$ from z<sub>t</sub> using v<sub>θ</sub> \triangleright Prediction: Denoise latent, Eq. 4   
10: $\hat { z } _ { t } = ( 1 - t ) \hat { z } _ { 0 } + t \epsilon ^ { \prime } , \ \epsilon ^ { \prime } \sim \mathcal { N } ( 0 , I )$ \triangleright Re-noise anchor prediction   
11: $z _ { 0 } ^ { + } = \hat { z } _ { t } - t \cdot v _ { \theta _ { p r e } } ( \hat { z } _ { t } ; t , x ^ { \mathrm { t g t } } )$ \triangleright Detached positive   
12: $z _ { 0 } ^ { - } = \hat { z } _ { t } - t \cdot v _ { \theta _ { p r e } } ( \hat { z } _ { t } ; t , x ^ { \mathrm { s r c } } )$ \triangleright Detached negative   
13: $\breve { \mathscr { L } } _ { c o n t r a s t } = \| \hat { z } _ { 0 } ^ { \cdots } - z _ { 0 } ^ { + } \| ^ { 2 } - \| \hat { z } _ { 0 } - z _ { 0 } ^ { - } \| ^ { 2 }$ \triangleright Latent contrastive loss, Eq. 11   
14: $\mathcal { L } _ { r e g } = \| v _ { \theta } ( z _ { t } ) - v _ { \theta _ { p r e } } ( z _ { t } ) \| ^ { 2 }$ \triangleright Regularization   
15: $\theta \gets \theta - \eta \nabla _ { \theta } ( \mathcal { L } _ { c o n t r a s t } + \lambda \mathcal { L } _ { r e g } )$ \triangleright Update generator   
16: end while

$$
\widetilde { v } = v _ { \phi } ^ { w } ( { x _ { t } ^ { \sf t g t } } ; t , c ^ { \sf r e f } ) - v _ { \phi } ( { x _ { t } ^ { \sf s r c } } ; t , \emptyset ) ,\tag{7}
$$

where $v _ { \phi }$ is the velocity field of the pre-trained 2D editor, w is the classifier-free guidance scale [12], and $c ^ { \mathrm { r e f } }$ encodes both the reference image $x ^ { \mathrm { r e f } }$ and a text prompt indicating a viewpoint change, as illustrated in Fig. 2.

Noise Update. Random noise can weaken the editing strength [39]. We initialize the shared noise ϵ once per rendered view, then update it at each editing step by injecting the classifier-free guidance signal:

$$
\epsilon  \epsilon - ( v _ { \phi } ( x _ { t } ^ { \mathrm { { t g t } } } ; t , c ^ { \mathrm { { r e f } } } ) - v _ { \phi } ( x _ { t } ^ { \mathrm { { t g t } } } ; t , \emptyset ) ) \cdot ( 1 - t )\tag{8}
$$

This update aligns the noise with the conditional direction to strengthen editing while preserving fine-grained structural details. Algorithm 1 summarizes the complete procedure, including the noise update at each editing step.

## 3.4 Contrastive Distillation with 2D Pseudo-Targets

To instantiate the loss L in Eq. 1 for the 3D latent generator $G _ { \theta }$ , we lift supervision from the rendered views $x ^ { \mathrm { s r c } } , x ^ { \mathrm { t g t } }$ into the latent space of the pretrained generator $v _ { \theta _ { p r e } }$ , and cast it as a contrastive objective: we treat the generator’s predicted clean latent $\hat { z } _ { 0 }$ as an anchor, with the edited rendering $x ^ { \mathrm { { t g t } } }$ inducing a positive latent target and the original rendering $x ^ { \mathsf { s r c } }$ inducing a negative latent target. This closes the Render-Edit-Optimize loop.

Per-Iteration Setup. Concretely, in each training iteration, given a reference image $x ^ { \mathrm { r e f } }$ , we roll out a clean latent $z _ { 0 } = G _ { \theta } ( x ^ { \tt r e f } )$ via ODE sampling. We then sample a camera pose π and render $x ^ { \mathsf { s r c } } = \mathcal { P } ( \mathcal { D } ( z _ { 0 } ) , \pi )$ . From it we obtain an edited rendering $x ^ { \mathrm { t g t } }$ via Reinforced Editing in Algorithm 1. Finally, we sample $t \sim \mathcal { U } ( 0 , 1 )$ and perturb the latent as $z _ { t } = ( 1 - t ) z _ { 0 } + t \epsilon$ , with $\epsilon \sim \mathcal { N } ( 0 , I )$ . A single denoising step predicts the clean latent:

$$
\hat { z } _ { 0 } = { z } _ { t } - t \cdot v _ { \theta } ( { z } _ { t } ; t , { x } ^ { \mathrm { r e f } } ) ,\tag{9}
$$

which serves as the anchor in our contrastive objective below.

Latent-Space Contrastive Supervision. Inspired by [47], we re-noise the anchor with fresh noise $\epsilon ^ { \prime } \sim \mathcal { N } ( 0 , I )$ as $\hat { z } _ { t } = ( 1 - t ) \hat { z } _ { 0 } + t \epsilon ^ { \prime }$ , and run two passes of the frozen pretrained generator with shared $\hat { z } _ { t }$ and diferent visual conditions:

$$
z _ { 0 } ^ { + } = \hat { z } _ { t } - t \cdot v _ { \theta _ { p r e } } ( \hat { z } _ { t } ; t , x ^ { \mathrm { t g t } } ) , \quad z _ { 0 } ^ { - } = \hat { z } _ { t } - t \cdot v _ { \theta _ { p r e } } ( \hat { z } _ { t } ; t , x ^ { \mathrm { s r c } } ) .\tag{10}
$$

Both $z _ { 0 } ^ { + }$ and $z _ { 0 } ^ { - }$ are detached from the computational graph and serve as the positive and the negative, respectively. Under the edited rendering $x ^ { \mathrm { t g t } }$ , the pretrained generator denoises toward the edited view, while under $x ^ { \tt s r c }$ it denoises toward the un-edited rendering. Our contrastive objective pulls the anchor $\hat { z } _ { 0 }$ toward the positive $z _ { 0 } ^ { + }$ and pushes it away from the negative $z _ { 0 } ^ { - }$ :

$$
\mathcal { L } _ { c o n t r a s t } = | | \hat { z } _ { 0 } - z _ { 0 } ^ { + } | | ^ { 2 } - | | \hat { z } _ { 0 } - z _ { 0 } ^ { - } | | ^ { 2 } .\tag{11}
$$

For geometric stability, we regularize the velocity against the pre-trained prior:

$$
\mathcal { L } _ { r e g } = \| v _ { \theta } ( z _ { t } ) - v _ { \theta _ { p r e } } ( z _ { t } ) \| ^ { 2 } ,\tag{12}
$$

giving the total objective $\mathcal { L } _ { t o t a l } = \mathcal { L } _ { c o n t r a s t } + \lambda \mathcal { L } _ { r e g }$ . Algorithm 2 summarizes the complete loop of rollout, editing, and generator updates.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. Existing 3D benchmarks such as Objaverse [6] subsets focus on everyday objects, where pre-trained baselines already do well and fidelity gaps stay hidden. To cover the more complex needs of image-to-3D users, especially imaginative and diverse concept-design tasks, we curate a Conceptual Design Dataset of 2,396 in-the-wild reference images under-served by existing 3D priors. We hold out 100 references for evaluation and use the remaining 2,296 for training. OREO uses these images alone and requires no paired 3D ground truth. Curation and filtering details are in Supp. Material. For a general-distribution reference, we additionally evaluate on Google Scanned Objects (GSO) [7].

Implementation Details. We use Trellis [38] as the 3D backbone and Qwen-Image-Edit [37] for Reinforced Editing, whose hyper-parameters are given in Sec. 4.2. Training details are provided in Supp. Material.

xref

xsrc

Reinforced Editing (ours)

xtgt

Nanobanana-pro

Qwen-Image-Edit

![](images/64843dc71d6b0d5eea8f3d3c7c394f9928b07a27034f6e1bbddcb82102e2b51c.jpg)  
Fig. 3: Qualitative comparison of 2D feedback sources. An ideal $x ^ { \tt t g t }$ keeps the viewpoint of $x ^ { \mathsf { s r c } }$ but lifts its fidelity toward $x ^ { \mathrm { r e f } } ;$ ; we compare Reinforced Editing (Ours), NanoBanana Pro, and Qwen-Image-Edit. Details are discussed in Sec. 4.2.

Table 1: Quantitative comparison of 2D feedback sources. We report CLIP and DINO similarity to $\boldsymbol { x } ^ { \mathrm { r e f } }$ for fidelity and to $x ^ { \mathsf { s r c } }$ for content preservation, together with Mask IoU for viewpoint consistency. See Sec. 4.2.
<table><tr><td rowspan="2">Method</td><td colspan="2"> $\mathrm { w . r . t . ~ } x ^ { \mathrm { r e f } }$ </td><td colspan="2"> $\mathrm { w . r . t . ~ } x ^ { \mathrm { s r c } }$ </td><td rowspan="2">IoU ↑</td></tr><tr><td>CLIP Sim. ↑</td><td>DINO Sim. ↑</td><td></td><td>CLIP Sim. ↑ DINO Sim. ↑</td></tr><tr><td>Unedited</td><td>0.7613</td><td>0.7916</td><td>N/A</td><td>N/A</td><td> $\mathrm { N } / \mathrm { A }$ </td></tr><tr><td>NanoBanana Pro</td><td>0.8803</td><td>0.9001</td><td>0.8199</td><td>0.8404</td><td>0.6830</td></tr><tr><td>Qwen-Edit (Orig.)</td><td>0.7753</td><td>0.8269</td><td>0.7882</td><td>0.8084</td><td>0.6374</td></tr><tr><td>Ours</td><td>0.7992(+0.0379)0</td><td>0.8214(+0.0298)</td><td>0.8833</td><td>0.9099</td><td>0.9520</td></tr></table>

## 4.2 Validating the Editing Feedback

The 3D generator is supervised by edited-rendering pseudo-targets from a 2D editor, so the quality of these edits directly bounds what the 3D loop can achieve. We therefore first evaluate the editor’s feedback at the 2D level.

Evaluation Protocol. For each of the 100 held-out references in the Conceptual Design Dataset, we generate a 3D asset with Trellis and render 5 views at azimuths $- 9 0 ^ { \circ } , - 4 5 ^ { \circ } , 0 ^ { \circ } , 4 5 ^ { \circ }$ , and $9 0 ^ { \circ }$ as source views $x ^ { \mathsf { s r c } }$ . To evaluate the editing performance, we adopt three metrics. To measure the multi-view fidelity improvement, we compute the CLIP-ViT-L/14 [30] and DINOv3-ViT-L/16 [33] embedding similarity of each rendered view to the reference $x ^ { \mathrm { r e f } }$ , and report the mean over the 5 views before and after editing. To check content preservation, we additionally report the similarity to $x ^ { \mathsf { s r c } }$ . Additionally, to quantify viewpoint and silhouette preservation, we compute Mask IoU between the edited output and $x ^ { \mathsf { s r c } }$ , using foreground masks extracted by $\mathrm { U ^ { 2 } }$ -Net [29] as in Trellis.

Compared Methods. We compare our Reinforced Editing (RE) against the original Qwen-Image-Edit [37] pipeline and NanoBanana Pro. The setup for the compared methods is provided in Supp. Material.

Qualitative Analysis. Fig. 3 compares the feedback sources along two requirements: improving appearance while preserving the viewpoint and content of $x ^ { \mathrm { s r c } }$ . NanoBanana Pro often follows the reference image too aggressively, replacing the source viewpoint with a frontal view and altering local content such as the mage’s cape. Qwen-Image-Edit introduces larger structural changes, including pose deformation, cropping, and background drift. In contrast, Reinforced Editing retains the source perspective and layout while adding appearance details, producing edited renderings that are better aligned with the rendered views.

Quantitative Results. Table 1 tests whether each editor provides the desired appearance-improvement direction. Similarity to $x ^ { \mathrm { r e f } }$ measures semantic and appearance alignment, while similarity to $x ^ { \mathsf { s r c } }$ and Mask IoU assess whether the source-view structural conditions remain stable. The unedited row provides the baseline, and the positive ∆CLIP/∆DINO values shown for RE simply confirm that editing improves its alignment with $x ^ { \mathrm { r e f } }$ . NanoBanana Pro scores higher against $x ^ { \mathrm { r e f } }$ but lower against $x ^ { \mathsf { s r c } }$ and on Mask IoU, indicating that its apparent gain is entangled with changes to the rendered view. RE provides a cleaner fidelity-improvement signal: it moves the rendered view toward the reference in appearance while avoiding unwanted changes in viewpoint, silhouette, pose, and spatial layout. This makes its edited renderings more suitable as 2D pseudo-targets for post-training the 3D generator.

Ablation Study on RE Parameters. We study two diferent RE parameters separately: the editing ratio $N _ { e } / N$ , which controls how much of the full reverse-time schedule is used for editing, and the absolute number of editing steps $N _ { e }$ , which controls how many updates are performed within that editing interval. Editing Ratio. With the schedule length fixed, a ratio of 0.25 applies editing only over a short interval and therefore produces limited detail enhancement, while a ratio of 1.0 edits the entire trajectory and causes visible changes in pose, scale, and identity. The intermediate ratio of 0.75 improves texture fi-

![](images/3d919254f2cfea6a8062a17642d6fd00fe17f1e4eb84d4873337b94e88517c77.jpg)

![](images/1d6257e6fa35f3362949fe98c578e7c09f589b207c5175926aca0e21b6a664d5.jpg)  
Fig. 5: Sensitivity Analysis of RE Parameters. (a) Efect of Editing Ratio $( N _ { e } / N )$ A ratio of 0.75 balances semantic improvement (CLIP/DINO ∆) and structural consistency (Mask IoU). (b) Efect of Editing Steps (N<sub>e</sub>): Under a fixed ratio (≈ 0.75), more steps improve CLIP/DINO similarity but reduce Mask IoU and increase cost, motivating our choice of 9 editing steps as a balanced setting.

xref

xsrc

Ratio=0.25

Ratio=0.5

Ratio=0.75

Ratio=1.0

![](images/f3997144fbc576cfc001ae77022a9323991a49fff42960661baca11c5068ac94.jpg)  
(b) Efect of Editing Steps (N<sub>e</sub>)

Fig. 6: Qualitative ablation study on RE parameters. (a) Editing Ratio: A low ratio (e.g., 0.25) limits editing capability, while a full ratio (1.0) destroys the original structure. Our choice of 0.75 strikes a balance. (b) Editing Steps: The examples show changes in appearance detail as the number of editing steps increases.

![](images/9f6e71df244d56fe4d185a94e73584513ac7d9b07bd214ed13fd2327ad2ff6cd.jpg)  
Fig. 4: Editing trajectories with and without the source branch. Removing the source branch (top) leads to progressive color over-saturation and structural drift; keeping it (bottom, Ours) stabilizes the trajectory. Details in Sec. 4.2.

Table 2: Ablation on key components of Reinforced Editing. We report ∆CLIP and ∆DINO, defined as the change in CLIP/DINO similarity to $x ^ { \mathrm { r e f } }$ before and after editing; higher is better. Details are in Sec. 4.2.
<table><tr><td>Variant</td><td>∆CLIP Sim. ↑ ∆DINO Sim. ↑</td></tr><tr><td>Full (Ours)</td><td>+0.0379 +0.0298</td></tr><tr><td>w/o Source Branch</td><td>-0.0523 -0.0497</td></tr><tr><td>w/o Noise Update</td><td>-0.0001 +0.0089</td></tr></table>

delity while preserving the source silhouette and viewpoint, as reflected by the trends in Fig. 5(a) and the examples in Fig. 6(a). Editing Steps. We then vary $N _ { e }$ while keeping the editing ratio approximately fixed. More steps improve CLIP/DINO similarity but reduce Mask IoU and increase computation, with diminishing visual gains beyond 9 steps, as shown in Fig. 5(b) and Fig. 6(b). We therefore use $N _ { e } = 9$ and N = 12 in our 3D experiments to balance improved visual fidelity, source-view structural consistency, and practical editing cost.

Ablation Study on RE Components. RE consists of a source-aware regularization term and a dynamic noise update. Without the source branch, target-conditioned guidance causes progressive color over-saturation and structural drift, as shown in Fig. 4. Without the noise update, the editing efect becomes much weaker and yields only marginal visual improvement. Table 2 reports the corresponding changes in CLIP/DINO similarity, confirming that both components contribute to the desired editing efect. Together, these components provide efective and controlled feedback for downstream 3D supervision.

## 4.3 Main Results

Experimental Setup. We evaluate the 3D generation quality of Trellis distilled with OREO. As baselines, we compare against the pre-trained Trellis and Photo3D [21], an ofline detail-enhancement method built on the same Trellis backbone. Following the protocol in Sec. 4.2, we render five views for each reference at azimuths $- 9 0 ^ { \circ } , - 4 5 ^ { \circ } , 0 ^ { \circ } , 4 5 ^ { \circ }$ , and $9 0 °$ . We report CLIP-ViT-L/14 [30] and DINOv3-ViT-L/16 [33] embedding similarity to the reference x<sup>ref</sup>, averaged over the five views. In addition to our proposed Conceptual Design Dataset, which targets imaginative and out-of-distribution references, we further validate on GSO [7], an in-distribution benchmark of common real-world objects, to assess whether OREO maintains the pre-trained model’s performance on its original distribution. This provides a complementary check of generalization.

![](images/9bc03f67bc958c0a6047fa98efebb1a5602a6847a7dca32bb4b8c7a622e9e0dd.jpg)  
Fig. 7: Main qualitative comparison of 3D generation. OREO delivers higher multi-view visual fidelity than Trellis and Photo3D. Details are in Sec. 4.3.

Table 3: Main quantitative comparison of 3D generation. CLIP and DINO similarity to $x ^ { \mathrm { r e f } }$ on the Conceptual Design Dataset and GSO; see Sec. 4.3.
<table><tr><td rowspan="2">Method</td><td colspan="2">Conceptual Design</td><td colspan="2">GSO</td></tr><tr><td>CLIP Sim. ↑ DINO Sim. ↑ CLIP Sim. ↑ DINO Sim. ↑</td><td></td><td></td><td></td></tr><tr><td>Trellis</td><td>0.7613</td><td>0.7916</td><td>0.7722</td><td>0.7022</td></tr><tr><td>Photo3D</td><td>0.7380</td><td>0.7837</td><td>0.7512</td><td>0.7034</td></tr><tr><td>OREO (Ours)</td><td>0.7834</td><td>0.8065</td><td>0.7764</td><td>0.7069</td></tr></table>

Quantitative and Qualitative Evaluation. Table 3 summarizes the quantitative comparison. OREO achieves clear gains in CLIP and DINO similarity on the Conceptual Design Dataset, our target imaginative distribution, while maintaining comparable performance on GSO. Fig. 7 shows the visual comparison. Compared with the pre-trained Trellis and Photo3D, OREO recovers finer details while preserving geometry and identity across views. Fig. 8 provides more qualitative comparisons across views. Unlike Photo3D’s detail-focused post-training, OREO updates Trellis’s sparse-structure stage to enable such shape corrections.

![](images/a6c49fb3faa4915c6e72260c6bce3084f602c12c2e6adfaa2d70ddd2c81e98dc.jpg)  
Fig. 8: Additional multi-view comparisons. Each example compares Trellis, Photo3D, and OREO using the input reference and five views at azimuths −90<sup>◦</sup>, −45<sup>◦</sup>, 0<sup>◦</sup>, 45<sup>◦</sup>, and 90<sup>◦</sup>.

![](images/18653a7d619b39fea15a41e5405f19aa4c2ee640583ecbe2e4b6bd3cc0e5dd07.jpg)  
Fig. 9: Qualitative ablation of the 3D generator design choices. We compare Full OREO against other training variants. Details are in Sec. 4.4.

Table 4: Ablation on the 3D generator design choices. CLIP and DINO similarity to $x ^ { \mathrm { r e f } }$ on the Conceptual Design Dataset; see Sec. 4.4.
<table><tr><td>Method / Variant CLIP Sim. ↑ DINO Sim. ↑</td></tr><tr><td>Pretrained Trellis</td><td>0.7613 0.7916</td></tr><tr><td>Full OREO (Ours)</td><td>0.7834 0.8065</td></tr><tr><td>w/ Pixel-MSE Supervision</td><td>0.6617 0.6982</td></tr><tr><td>w/ Off-policy Rollout</td><td>0.7279 0.7613</td></tr><tr><td>w/ DMD (Score Distillation)</td><td>0.5393 0.5486</td></tr></table>

More examples are provided in Supp. Material. These results show that the improvements produced by Reinforced Editing can be efectively distilled into the 3D generator. Although each iteration provides only a limited update, these improvements accumulate over training and lead to substantial gains in both appearance and shape. The distilled supervision is applied to both stages of the Trellis generator, including sparse structure generation and sparse feature generation, allowing OREO to progressively improve the shape and appearance of generated assets across multiple viewpoints.

User Preference Study. To complement the automatic metrics, we further conduct a user preference study on the same held-out Conceptual Design Dataset, in which participants compare multi-view renderings from Pretrained Trellis, Photo3D, and OREO under blinded conditions. OREO receives the highest aggregate preference share, with 38% of all responses. The Supp. Material provides the full study protocols and preference results.

## 4.4 Ablation Study and Analysis

Full OREO combines on-policy rollouts, latent contrastive supervision, and explicit RE-edited renderings as 2D pseudo-targets. We compare it with of-policy rollout, pixel MSE, and DMD [46] supervision. Table 4 shows that all three variants underperform pretrained Trellis, while Full OREO improves both CLIP and DINO similarity on the same held-out references in this comparison.

Degradation under Of-policy Rollout. Fixing the rollout to the pretrained generator’s output ties supervision to its initial distribution rather than the generator’s current trajectory. As the student evolves, this fixed supervision becomes stale and cannot correct newly visited states. Fig. 9 shows that the geometry remains largely preserved, but colors desaturate and fine-grained textures are smoothed out. Together with the lower scores in Table 4, this supports on-policy rollouts as a way to keep supervision aligned with the evolving generator throughout training.

Attenuation of Gradient through Diferentiable Rendering. This variant keeps the RE-edited views but replaces latent contrastive supervision with pixel-space regression $\| x ^ { \mathbf { s r c } } - x ^ { \mathbf { t g t } } \| ^ { 2 }$ , sending the loss through the decoder D and renderer P. It also loses the positive/negative latent comparison between fidelity correction and preserved source content. Pixel MSE therefore treats all image discrepancies as direct regression errors, without distinguishing details to enhance from content to preserve. The outputs in Fig. 9 become textureaveraged and color-shifted, and Table 4 shows lower CLIP and DINO scores. This comparison supports latent contrastive supervision as a more efective way to transfer the RE correction to the 3D generator.

Role of Reinforced Editing. The DMD variant replaces the explicit REedited rendering target with an image-editor DMD loss, giving an implicit score signal through the renderer and decoder. Unlike RE, this signal provides no concrete, structure-preserving target specifying appearance changes while keeping viewpoint and source content fixed. Fig. 9 shows geometric distortion and color collapse, while Table 4 reports the lowest scores. This supports explicit RE-edited renderings as stable supervision for the downstream 3D generator.

## 5 Conclusion

This paper proposes OREO, a fidelity alignment framework for 3D generation. Addressing the lack of visual realism in existing 3D models, we introduce an onthe-fly optimization loop that leverages 2D difusion priors. In the Render–Edit– Optimize loop, Reinforced Editing produces structure-preserving 2D pseudotargets. Contrastive Distillation uses edited and source renderings as conditions to construct positive and negative latent targets for direct generator supervision. This design provides dynamic supervision without paired 3D ground truth, refreshing targets from the evolving generator while retaining the pretrained 3D prior for geometric stability. Experimental results show that OREO improves the visual fidelity and texture detail of 3D generators while largely preserving structure. Limitations are provided in Supp. Material.

## References

1. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: CVPR (2023)

2. Chen, H., Lan, Y., Chen, Y., Zhou, Y., Pan, X.: Mvdrag3d: Drag-based creative 3d editing via multi-view generation-reconstruction priors. arXiv preprint arXiv:2410.16272 (2024)

3. Chen, K., Xie, E., Chen, Z., Wang, Y., Hong, L., Li, Z., Yeung, D.Y.: Geodifusion: Text-prompted geometric control for object detection data generation. In: ICLR. vol. 2024, pp. 8979–9001 (2024)

4. Chen, L., Wang, P., Zhang, G., Ma, Z., Zhang, L.: Omni-3dedit: Generalized versatile 3d editing in one-pass. In: CVPR. pp. 12640–12650 (2026)

5. Chen, Z., Tang, J., Dong, Y., Cao, Z., Hong, F., Lan, Y., Wang, T., Xie, H., Wu, T., Saito, S., et al.: 3dtopia-xl: Scaling high-quality 3d asset generation via primitive difusion. In: CVPR. pp. 26576–26586 (2025)

6. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. In: CVPR. pp. 13142–13153 (2023)

7. Downs, L., Francis, A., Koenig, N., Kinman, B., Hickman, R., Reymann, K., McHugh, T.B., Vanhoucke, V.: Google scanned objects: A high-quality dataset of 3d scanned household items. In: ICRA. pp. 2553–2560. IEEE (2022)

8. Guo, J., Gao, S., Bian, J.W., Sun, W., Zheng, H., Jia, R., Gong, M.: Hyper3d: Eficient 3d representation via hybrid triplane and octree feature for enhanced 3d shape variational auto-encoders. arXiv preprint arXiv:2503.10403 (2025)

9. Guo, Y., Zhang, Z., Wang, P., Liang, X., Ma, Z., Zhang, L.: Memorize when needed: Decoupled memory control for spatially consistent long-horizon video generation. arXiv preprint arXiv:2604.18215 (2026)

10. Haque, A., Tancik, M., Efros, A.A., Holynski, A., Kanazawa, A.: Instruct-nerf2nerf: Editing 3d scenes with instructions. In: ICCV. pp. 19740–19750 (2023)

11. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., Cohen-Or, D.: Prompt-to-prompt image editing with cross attention control. In: ICLR (2023)

12. Ho, J., Salimans, T.: Classifier-free difusion guidance. In: NeurIPS Workshop (2021)

13. Hunyuan3D, T., Yang, S., Yang, M., Feng, Y., Huang, X., Zhang, S., He, Z., Luo, D., Liu, H., Zhao, Y., et al.: Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442 (2025)

14. Koh, E., Hyun, S., Lee, M., Chung, J., Seo, K., Heo, J.P.: Difusion feature field for text-based 3D editing with gaussian splatting. In: NeurIPS. vol. 38, pp. 159–178 (2025)

15. Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., Michaeli, T.: FlowEdit: Inversion-free text-based editing using pre-trained flow models. In: ICCV. pp. 19721–19730 (2025)

16. Lai, Z., Zhao, Y., Liu, H., Zhao, Z., Lin, Q., Shi, H., Yang, X., Yang, M., Yang, S., Feng, Y., et al.: Hunyuan3d 2.5: Towards high-fidelity 3d assets generation with ultimate details. arXiv preprint arXiv:2506.16504 (2025)

17. Lan, Y., Zhou, S., Lyu, Z., Hong, F., Yang, S., Dai, B., Pan, X., Loy, C.C.: GaussianAnything: Interactive point cloud flow matching for 3D object generation. In: ICLR (2025)

18. Li, W., Zhang, X., Sun, Z., Qi, D., Li, H., Cheng, W., Cai, W., Wu, S., Liu, J., Wang, Z., et al.: Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets. arXiv preprint arXiv:2505.07747 (2025)

19. Li, Y., Zou, Z.X., Liu, Z., Wang, D., Liang, Y., Yu, Z., Liu, X., Guo, Y.C., Liang, D., Ouyang, W., et al.: Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint arXiv:2502.06608 (2025)

20. Liang, X., Ma, Z., Sun, L., Guo, Y., Zhang, L.: Aligncvc: Aligning cross-view consistency for single-image-to-3d generation. In: AAAI. vol. 40, pp. 6889–6897 (2026)

21. Liang, X., Ma, Z., Sun, L., Guo, Y., Zhang, L.: Photo3d: Advancing photorealistic 3d generation through structure-aligned detail enhancement. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 34237– 34247 (2026)

22. Lin, C.H., Gao, J., Tang, L., Takikawa, T., Zeng, X., Huang, X., Kreis, K., Fidler, S., Liu, M.Y., Lin, T.Y.: Magic3d: High-resolution text-to-3d content creation. In: CVPR. pp. 300–309 (2023)

23. Lin, C., Pan, P., Yang, B., Li, Z., Mu, Y.: Difsplat: Repurposing image difusion models for scalable gaussian splat generation. arXiv preprint arXiv:2501.16764 (2025)

24. Liu, X., Zhang, X., Ma, Z., Zhu, X., Lei, Z.: Mvboost: Boost 3d reconstruction with multi-view refinement. In: CVPR. pp. 21664–21673 (2025)

25. Ma, Z., Liang, X., Wu, R., Zhu, X., Lei, Z., Zhang, L.: Progressive rendering distillation: Adapting stable difusion for instant text-to-mesh generation without 3d data. In: CVPR. pp. 11036–11050 (2025)

26. Ma, Z., Wei, Y., Zhang, Y., Zhu, X., Lei, Z., Zhang, L.: Scaledreamer: Scalable textto-3d synthesis with asynchronous score distillation. In: ECCV. pp. 1–19. Springer (2024)

27. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided difusion models. In: CVPR (2023)

28. Poole, B., Jain, A., Barron, J.T., Mildenhall, B.: Dreamfusion: Text-to-3d using 2d difusion. arXiv preprint arXiv:2209.14988 (2022)

29. Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O.R., Jagersand, M.: U2- net: Going deeper with nested u-structure for salient object detection. Pattern recognition 106, 107404 (2020)

30. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763. PmLR (2021)

31. Rout, L., Chen, Y., Ruiz, N., Caramanis, C., Shakkottai, S., Chu, W.S.: Semantic image inversion and editing using rectified stochastic diferential equations. In: ICLR (2025)

32. Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., Taigman, Y.: Emu edit: Precise image editing via recognition and generation tasks. arXiv preprint arXiv:2311.10089 (2023)

33. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

34. Wang, J., Fang, J., Zhang, X., Xie, L., Tian, Q.: Gaussianeditor: Editing 3d gaussians delicately with text instructions. In: CVPR. pp. 20902–20911 (2024)

35. Wang, P., Chen, L., Ma, Z., Guo, Y., Zhang, G., Zhang, L.: One2scene: Geometric consistent explorable 3d scene generation from a single image. arXiv preprint arXiv:2602.19766 (2026)

36. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: ProlificDreamer: High-fidelity and diverse text-to-3D generation with variational score distillation. In: NeurIPS. vol. 36 (2023)

37. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.m., Bai, S., Xu, X., Chen, Y., et al.: Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025)

38. Xiang, J., Lv, Z., Xu, S., Deng, Y., Wang, R., Zhang, B., Chen, D., Tong, X., Yang, J.: Structured 3d latents for scalable and versatile 3d generation. In: CVPR. pp. 21469–21480 (2025)

39. Xie, C., Li, M., Li, S., Wu, Y., Yi, Q., Zhang, L.: DNAEdit: Direct noise alignment for text-guided rectified flow editing. In: NeurIPS. vol. 38, pp. 124456–124474 (2025)

40. Yang, S., Cun, X., Li, X., Li, Y., Zhang, J.: 4dvd: cascaded dense-view video difusion model for high-quality 4d content generation. International Journal of Computer Vision 134(5), 233 (2026)

41. Yang, S., Li, X., Cun, X., Wang, G., Li, L., Shan, Y., Zhang, J.: Gencompositor: Generative video compositing with difusion transformer. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview. net/forum?id=ynim5u2N4i

42. Yang, S., Wang, Y., Li, H., Meng, J., Wu, Y., Meng, X., Zhang, J.: Hybrid fourier score distillation for eficient one image to 3d object generation. Visual Intelligence 3(1), 17 (2025)

43. Yang, X., Shi, H., Zhang, B., Yang, F., Wang, J., Zhao, H., Liu, X., Wang, X., Lin, Q., Yu, J., et al.: Hunyuan3d-1.0: A unified framework for text-to-3d and image-to-3d generation. arXiv preprint arXiv:2411.02293 (2024)

44. Ye, C., Wu, Y., Lu, Z., Chang, J., Guo, X., Zhou, J., Zhao, H., Han, X.: Hi3DGen: High-fidelity 3D geometry generation from images via normal bridging. In: ICCV. pp. 25050–25061 (2025)

45. Yenphraphai, J., Pan, X., Liu, S., Panozzo, D., Xie, S.: Image sculpting: Precise object editing with 3d geometry control. In: CVPR. pp. 4241–4251 (2024)

46. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step difusion with distribution matching distillation. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 6613–6623. IEEE (2024)

47. Yu, X., Qi, X., Li, Z., Zhang, K., Zhang, R., Lin, Z., Shechtman, E., Wang, T., Nitzan, Y.: Self-evaluation unlocks any-step text-to-image generation. In: CVPR. pp. 7816–7826 (2026)

48. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing. NeurIPS 36, 31428–31449 (2023)

49. Zhang, L., Zhang, Q., Jiang, H., Bai, Y., Yang, W., Xu, L., Yu, J.: Bang: Dividing 3d assets via generative exploded dynamics. ACM Transactions on Graphics (TOG) 44(4), 1–21 (2025)

50. Zhao, H., Ma, X., Chen, L., Si, S., Wu, R., An, K., Yu, P., Zhang, M., Li, Q., Chang, B.: Ultraedit: Instruction-based fine-grained image editing at scale. NeurIPS 37, 3058–3093 (2024)

51. Zhao, Z., Lai, Z., Lin, Q., Zhao, Y., Liu, H., Yang, S., Feng, Y., Yang, M., Zhang, S., Yang, X., et al.: Hunyuan3d 2.0: Scaling difusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202 (2025)

52. Zhuang, J., Wang, C., Lin, L., Liu, L., Li, G.: Dreameditor: Text-driven 3d scene editing with neural fields. In: SIGGRAPH Asia 2023 conference papers. pp. 1–10 (2023)

# OREO: Fidelity Alignment in 3D Generation via On-the-fly Rendering-Editing Optimization Supplementary Material

## Overview

This supplementary material first presents additional multi-view comparisons among OREO, the pretrained Trellis, and Photo3D. We then provide details on dataset construction, training and editing configurations, the user preference study protocol, computational overhead, and limitations and failure cases.

## S1 Additional Multi-view Comparisons

![](images/b3a2dbb1440371e9d06eb11c1f2eae4c92c30e79d8aaf9468b0f64178470e8b2.jpg)  
Fig. S1: Additional multi-view comparisons of the pretrained Trellis, Photo3D, and OREO. Each block shows the input reference, followed by five views rendered at the azimuth angles −90<sup>◦</sup>, −45<sup>◦</sup>, 0<sup>◦</sup>, 45<sup>◦</sup>, and 90<sup>◦</sup>.

![](images/99c187ec0e85d24dbfee20eecb1d5b785e95daa6d779b68340d970e74c85fe8a.jpg)  
Fig. S2: Additional multi-view qualitative comparisons (continued).

![](images/aecf370afae628754433dbce276af3b325bf10e701ed18fb7a584f6746b5094c.jpg)  
Fig. S3: Additional multi-view qualitative comparisons (continued).

![](images/73dfeaa27edab308a4904971bb9b281f15a803ae4a5ba42a4fad3bbb621ae7bd.jpg)  
Fig. S4: Additional multi-view qualitative comparisons (continued).

## S2 Implementation and Dataset Details

This section details dataset construction, training hyper-parameters, and the editing setup for the methods compared in our main experiments.

## S2.1 Conceptual Design Dataset Construction

We construct an image-only collection to support generator post-training without paired 3D supervision. The dataset comprises 2,396 concept-design reference images that we manually curate from public online sources; most images are AIgenerated or user-shared creations posted on public content platforms, reflecting the imaginative and stylized inputs that image-to-3D users bring in practice. We prioritize references with clean foregrounds, well-defined silhouettes, and informative texture or material cues, and manually filter the collection to remove duplicates, low-resolution thumbnails, watermarked images, and content with unclear foregrounds. We will release the dataset on our project page.

## S2.2 Training Details

We post-train Trellis-image-large with OREO on the Conceptual Design Dataset training split, using the frozen Qwen-Image-Edit-2511 [37] as our 2D editor.

Optimization. We use the Adan optimizer with learning rate $1 0 ^ { - 4 }$ , weight decay 0, and $\epsilon = 1 0 ^ { - 4 }$ . We use a per-GPU batch size of 1 with no gradient accumulation, giving an efective batch size of 8 reference images per iteration. Full-parameter updates are applied to both Trellis generation stages, while the decoder D remains frozen. Stage 1 generates the sparse structure from a latent represented on a dense 3D grid, and Stage 2 generates sparse SLAT features at the active coordinates. We jointly post-train both flow models in all experiments.

Rollout and camera sampling. Each iteration performs an on-policy ODE rollout with the current student generator to obtain a clean latent $z _ { 0 } = G _ { \theta } ( x ^ { \tt r e f } )$ The ODE rollout follows the sampling configuration of the pretrained Trellis model. For every reference we sample a single training view with yaw uniformly drawn from [0<sup>◦</sup>, 360<sup>◦</sup>], pitch fixed at $0 ^ { \circ }$ , and field of view $4 0 ^ { \circ }$ . The camera distance is determined by an adaptive-distance rule with a fill ratio of 0.9, so that the object occupies a consistent portion of the rendered view. Rendering uses the Gaussian-splatting renderer at 1024 × 1024 resolution with a white background.

Loss and time-step sampling. For the contrastive objective in Eq. 11 of the main paper, the training time step is sampled from $t \sim \mathcal { U } ( 0 . 0 2 , 0 . 9 8 )$ . The student prediction and the frozen pretrained-generator passes used to construct $z _ { 0 } ^ { + }$ and $z _ { 0 } ^ { - }$ all use classifier-free guidance with the default settings of the pretrained Trellis model. The positive and negative passes share the same re-noised latent $\hat { z } _ { t }$ and fresh noise draw $\epsilon ^ { \prime } ,$ while using $x ^ { \mathrm { t g t } }$ and $x ^ { \mathsf { s r c } }$ as their respective visual conditions. The total loss is $\mathcal { L } _ { t o t a l } = \mathcal { L } _ { c o n t r a s t } + \lambda \mathcal { L } _ { r e g }$ , with $\lambda = 1 . 0$ . The velocity regularizer $\mathcal { L } _ { \boldsymbol { r } \boldsymbol { e } \boldsymbol { g } }$ compares the student and pretrained velocity predictions with the same noisy latent $z _ { t } ,$ time step t, and reference condition $x ^ { \mathrm { r e f } }$

## S2.3 Editing Setup for Compared Methods

Reinforced Editing and the of-the-shelf 2D editors used as baselines share the same underlying editor backbone but difer in their text-conditioning interface. Both prompt families are fixed across the dataset without per-example tuning.

Reinforced Editing. We instantiate Reinforced Editing on top of Qwen-Image-Edit-2511 with $N = 1 2$ total time steps and $N _ { e } = 9$ editing steps applied at the last portion of the schedule (editing ratio 0.75). The target branch uses classifier-free guidance with scale $w = 4 .$ , while the source branch is an unconditional forward pass, in accordance with Eq. 7 of the main paper. The shared noise ϵ is initialized once per view and updated at each step by injecting the CFG signal, as in Eq. 8 of the main paper. The editing resolution matches the renderer at $1 0 2 4 \times 1 0 2 4$ . The text component of $c ^ { \mathrm { { r e f } } }$ is:

Table S1: Prompt selection for Reinforced Editing on the Conceptual Design Dataset. All entries share the same editor and RE hyper-parameters $( N = 1 2 , N _ { e } = 9 , w = 4 )$ ; we report ∆CLIP / ∆DINO similarity to $\bar { x ^ { \mathrm { { r e f } } } }$ relative to the unedited $x ^ { \mathsf { s r c } }$ , and Mask IoU with $x ^ { \mathsf { s r c } }$ . Bold row is our final choice.
<table><tr><td>Prompt</td><td></td><td>ΔCLIP ↑ ∆DINO ↑</td><td>IoU ↑</td></tr><tr><td colspan="4">Fidelity-only</td></tr><tr><td>Super resolution of the input image</td><td>+0.0209</td><td>+0.0371</td><td>0.9398</td></tr><tr><td>Enrich the detail given the reference image</td><td>+0.0171</td><td>+0.0250</td><td>0.9469</td></tr><tr><td>Add the appearance of image 1</td><td>+0.0233</td><td>+0.0346</td><td>0.9406</td></tr><tr><td colspan="4">Viewpoint change</td></tr><tr><td>Move the camera</td><td>+0.0277</td><td>+0.0398</td><td>0.9406</td></tr><tr><td>Obtain another side view</td><td>+0.0250</td><td>+0.0341</td><td>0.9330</td></tr><tr><td>Generate a novel view</td><td>+0.0161</td><td>+0.0264</td><td>0.9437</td></tr><tr><td>Move the camera to a novel view</td><td>+0.0268</td><td>+0.0387</td><td>0.9408</td></tr><tr><td colspan="4">Full 3D reinterpretation</td></tr><tr><td>Generate a 3D model of the image</td><td>-0.0274</td><td>+0.0062</td><td>0.9455</td></tr><tr><td colspan="4">&quot;Rotate the camera&quot; variants</td></tr><tr><td>Rotate the camera</td><td>+0.0314</td><td>+0.0371</td><td>0.9402</td></tr><tr><td>Rotate the camera. Consistent Lighting</td><td>+0.0204</td><td>+0.0357</td><td>0.9417</td></tr><tr><td>Rotate the camera. Consistent visual style</td><td>+0.0206</td><td>+0.0310</td><td>0.9428</td></tr><tr><td>Rotate the camera. Consistent concept design</td><td>+0.0265</td><td>+0.0316</td><td>0.9422</td></tr><tr><td>Rotate the camera. White background. (Ours)</td><td>+0.0339</td><td>+0.0387</td><td>0.9411</td></tr></table>

“Rotate the camera. White background.”

Prompt selection. We select a single fixed prompt by evaluating representative instructions on the Conceptual Design Dataset while keeping the editor and all RE hyper-parameters unchanged. Table S1 reports the change in CLIP and DINO similarity to the reference image relative to the unedited source, together with Mask IoU to the source view. Fidelity-only prompts provide limited gains because they do not specify that the source viewpoint should be retained, while asking the editor to generate a full 3D model substantially changes the input and produces a negative CLIP improvement. Viewpoint instructions perform better overall; among them, “Rotate the camera” directly requests another view of the same object without prescribing a specific azimuth or elevation. As shown in Table S1, adding “White background” gives the highest ∆CLIP, a strong ∆DINO, and high Mask IoU. It also prevents the editor from introducing scene backgrounds that are absent from the white-background source renderings. We therefore use “Rotate the camera. White background.” for all experiments.

Of-the-shelf editor baselines. For the feedback comparison in Sec. 4.1 of the main paper, we compare against the original Qwen-Image-Edit-2511 pipeline and NanoBanana Pro. Because both editors expose a single visual context that concatenates the source view $x ^ { \mathsf { s r c } }$ and the reference $x ^ { \mathrm { r e f } }$ side by side, the prompt must first assign roles to the two images before requesting the edit:

Table S2: User preferences on 100 held-out Conceptual Design Dataset examples, assessing reference fidelity, 3D consistency, and identity preservation across views.
<table><tr><td>Method Preference Ratio ↑</td></tr><tr><td>Pretrained (Trellis) 29%</td></tr><tr><td>Photo3D 33%</td></tr><tr><td>OREO (Ours) 38%</td></tr></table>

“The first image is a 3D rendered view. The second image is the reference. Edit the first image to match the reference object’s appearance. White background.”

The prompt is a minimal adaptation of the efective instruction reported by Photo3D and is applied identically to both editors. Only the editor backbone changes between runs; the released source code lists API-specific parameters, including the sampler, classifier-free guidance scale, and denoising steps.

## S3 User Preference Study Protocols

We conduct a user preference study to complement the automatic metrics in the main paper. The study assesses whether generated 3D assets preserve the input image’s identity and improve visual fidelity across rendered views.

Study setup. We use the same 100 held-out examples from the Conceptual Design Dataset as in the main evaluation. For each example, participants are shown the input reference image and multi-view renderings generated by three methods: Pretrained (Trellis), Photo3D, and OREO. With method names hidden, participants choose the result with the best overall quality.

Evaluation criteria. Participants are instructed to consider three aspects jointly. First, Fidelity measures whether the generated asset contains realistic materials, sharp boundaries, and fine-grained texture details. Second, Consistency measures whether the rendered views remain coherent as a 3D object rather than showing view-dependent artifacts. Third, Identity measures whether the asset preserves the input image’s semantic identity and distinctive details.

Participants. The study includes 20 participants, including participants familiar with 3D reconstruction and computer graphics. All participants evaluate the same 100 examples, and we aggregate their preferences over all responses.

As shown in Table S2, OREO receives the highest aggregate preference share, with 38% of all responses. With only aggregate preference ratios recorded, we report descriptive results without unsupported significance tests.

## S4 Computational Overhead

Each OREO training iteration consists of two costs: a generator update (rollout, render, and backward pass), and a call to the frozen 2D editor to produce the edited rendering x<sup>tgt</sup>. The generator update alone takes roughly 22 s per iteration. If the 9-step editing call is executed sequentially inside the same iteration, it dominates the wall-clock cost, contributing about 81% of the total per-iteration time and stretching one iteration to roughly 5× the generator-only cost.

## S5 Limitations and Failure Cases

OREO inherits its appearance guidance from the 2D editing prior, and thus may also inherit the editor’s compositional or viewpoint biases in rare cases. We discuss three representative failure modes on the Conceptual Design Dataset and a limitation for generators that decouple geometry and appearance.

Viewpoint drift on face-like objects. For face-like objects or characters, the 2D editor may rotate the face toward the camera instead of preserving the intended 3D-facing direction, leading to view-dependent orientation inconsistency. This drift arises from a mismatch between the 2D editor’s canonical portrait prior and the 3D view-consistency requirement, so edited renderings may reorient a face toward the camera; repeated use of such targets can improve local facial details while causing inconsistent face orientation across views.

Editor-inherited color and style bias. The dynamic noise update tightens alignment with the editor’s appearance prior and can therefore inherit editorspecific color bias on unusual materials such as stone or metallic surfaces, shifting the intended hue while preserving geometry. Stylized materials outside the editor’s photographic training distribution may also undergo style shifts.

Remaining artifacts on weakly constrained regions. Rarely observed regions such as occluded surfaces, thin appendages, and sharp material boundaries receive weaker per-view supervision and can retain small local artifacts on back views and along thin structures. These regions are also under-constrained by the reference x<sup>ref</sup>, which only depicts a single canonical viewpoint.

Requirements on the generator architecture. OREO requires the optimized stage to provide a conditional flow-based latent model and a frozen pretrained counterpart for constructing the positive and negative predictions. In Trellis, this requirement is satisfied by both stages: Stage 1 represents the sparse structure with a latent on a dense 3D grid, whereas Stage 2 generates sparse SLAT features at the active coordinates. The contrastive formulation can therefore be applied to Stage 1, Stage 2, or both. For generators that decouple geometry and appearance, however, the improvements brought by OREO may be less consistent, as the editing signal cannot jointly constrain both components.

These observations suggest future directions such as stronger multi-view consistency constraints, 3D-aware editing models, or uncertainty-aware filtering of edited renderings before they are used as 2D pseudo-targets for generator updates.