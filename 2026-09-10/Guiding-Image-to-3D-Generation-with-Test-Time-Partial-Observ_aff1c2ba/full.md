# Guiding Image-to-3D Generation with Test-Time Partial Observations

Jerred Chen University of Oxford jerred.chen@cs.ox.ac.uk

Simon Weber University of Oxford simon.weber@cs.ox.ac.uk

Ronald Clark University of Oxford ronald.clark@cs.ox.ac.uk

## Abstract

Image-to-3D models can generate visually compelling 3D assets from a single RGB image, but their geometry is often only loosely constrained by the available observations, limiting their use in applications that require geometric fidelity. In many real-world settings, however, partial geometric observations of the object may be available at test time. We introduce a training-free framework for incorporating such evidence into pretrained image-to-3D generative models without retraining orfinetuning. To do this, we guide generation using a ray-consistent observation likelihood defined over the model’s occupancy representation, combining surface occupancy andfree-space evidence. Applied to SAM 3D and its multi-view extension, our approach substantially improves geometric fidelity across different levels ofobservability, as well as visual quality. Our results demonstrate that pretrained image-to-3D models can effectively integrate partial geometric observations through explicit test-time guidance, complementing their learned generative priors without modifying the underlying model.

## 1. Introduction

Image-to-3D generation models have recently achieved remarkable progress, producing high-quality 3D meshes and Gaussian splats from a single RGB image [22, 26]. Their impressive visual quality has made them attractive for appli cations ranging from digital content creation to robotics and mixed reality. However, this single-image setting also imposes a fundamental limitation: because a single image provides only ambiguous information about the underlying 3D shape, the generated geometry is largely determined by the model’s learned prior rather than the true object geometry (see Figure 1). While the resulting assets are often visually plausible, they may exhibit incorrect depth, proportions, or hallucinated unseen surfaces.

This ambiguity is acceptable for creative applications, but it becomes problematic in settings where geometric accuracy is essential, such as robotics, digital twins, or augmented reality. In these scenarios, additional geometric information is often available at test time. Rather than relying solely on a single RGB image, one may have access to partial observations of the object, obtained for example from multiple viewpoints. These observations typically cover only a subset of the object, leaving the remaining geometry fundamentally ambiguous. The challenge is therefore to generate a complete 3D asset that simultaneously respects the observed geometry while leveraging the powerful shape prior learned by a pretrained image-to-3D model to plausibly complete the unobserved regions.

![](images/7d3589adbe89f30adcf3842036f8071e96eaa6e6ffc65e223ec3826dfecb72c4.jpg)  
Figure 1. Image-to-3D generation models like SAM 3D [22] can produce plausible outputs due to the model’s learned prior, but available image observations often leave underlying geometry ambiguous: regions unseen from the input viewpoint (e.g. the sofa length) are reconstructed inconsistently with the true shape. With our proposed training-free guidance, the generation is steered towards the available information and recovers the correct geometry.

Existing approaches only partially address this problem. Multi-view extensions such as MV-SAM3D [13] improve consistency across several RGB images by modifying the generative model itself during sampling. However, they are tailored specifically to multiple RGB views and cannot naturally incorporate more general forms of geometric evidence such as partial point clouds or depth observations. More broadly, current image-to-3D models lack a generic mechanism for injecting arbitrary geometric constraints at inference time without retraining. In this work, we address this limitation through a simple training-free guidance framework for image-to-3D generation. We guide a pretrained image-to-3D model using partial geometric observations available only at inference time, without modifying or finetuning the underlying model. Concretely, we interpret the model’s canonical representation as an occupancy grid and construct an energy function directly from the observed geometry. This energy combines an occupancy term that encourages consistency with observed surfaces and a free-space term that prevents geometry from being generated where observations indicate empty space. Because the guidance acts solely on the sampling dynamics, it is entirely independent of the underlying image-to-3D architecture and can be readily integrated into existing models. In particular, it composes directly with multi-view extensions such as MV-SAM3D, combining the benefits of both approaches.

We evaluate our approach on SAM3D [22] using partial geometric observations derived from multiple views. Our method substantially improves geometric fidelity while preserving the visual quality of the generated assets, demonstrating that pretrained image-to-3D foundation models can effectively combine learned generative priors with incomplete geometric evidence through purely test-time guidance.

In particular, our contributions are as follows:

• A training-free guidance framework for incorporating partial geometric observations into pretrained image-to-3D generation models at inference time

• A theoretical perspective for how test-time evidence deforms the instantaneous flow landscape

• An occupancy-grid energy combining occupancy and free-space constraints built on a physically grounded, rayconsistent observation likelihood, allowing for enforcing geometric consistency and plausible completion of unseen regions

• Extensive experiments demonstrating substantial improvements over state-of-the-art image-to-3D generation methods on partial observation reconstruction and novel view synthesis benchmarks.

## 2. Related Works

While image-to-3D generation has progressed rapidly in recent years, surprisingly few works have investigated how to guide a pretrained image-to-3D model using external geometric observations available only at inference time. Existing methods either focus on improving the underlying generative model or on developing general guidance mechanisms for diffusion and flow matching. Our work lies at the intersection of these two directions: we leverage recent advances in training-free guidance to improve the geometric consistency of image-to-3D models without any retraining.

## 2.1. Guidance in Diffusion and Flow Matching

Guidance refers to mechanisms for steering the sampling process of a generative model towards samples satisfying a desired condition or constraint. In diffusion models, early forms of guidance relied either on an additional classifier trained to provide gradients towards a target class [5], or on training the generative model itself to support both conditional and unconditional generation, as in classifier-free guidance [9]. These methods showed that the sampling trajectory can be modified at inference time to trade off sample fidelity and adherence to a desired condition. More recently, several works have generalized this idea beyond semantic conditioning. [1, 4, 28, 29] show that pretrained diffusion models can be guided by arbitrary differentiable objectives without retraining. Our work follows this line of research by designing a geometry-aware energy function tailored to image-to-3D generation. Finally, recent work has extended these ideas from diffusion models to flow matching. Feng et al. [8] established the theoretical foundation of training-free guidance for flow-matching models, showing that guidance can be implemented through an additional velocity field during sampling. Building on this perspective, several methods have proposed practical guidance strategies for flow matching, including FlowChef [19], which steers the sampling trajectory using the gradient of a differentiable objective, and FlowDPS [11], which extends diffusion posterior sampling to flow matching for inverse problems. Our work builds upon this growing line of research. Rather than proposing a new guidance algorithm, we design a geometryaware energy tailored to image-to-3D generation from partial geometric observations.

## 2.2. Image-to-3D Generation

Recent advances in image-to-3D generation have largely been driven by diffusion and flow-based generative models capable of producing high-quality 3D assets from a single image. Early methods such as DreamFusion [20], Magic3D [15], Fantasia3D [3], and ProlificDreamer [24] formulate 3D generation as an optimization problem guided by pretrained diffusion models. These methods primarily rely on text or image guidance to synthesize plausible geometry, but do not incorporate external geometric observations during generation. Another line of work improves geometric quality by exploiting multiple input views. Methods such as MVDream [21], Era3D [14], Wonder3D [18], and the recent MV-SAM3D [13] enforce multi-view consistency during generation, leading to improved 3D reconstruction when several RGB observations are available. In contrast, our method is complementary to these approaches, as it leverages partial geometric observations rather than requiring additional RGB views. More recently, large-scale image-to-3D foundation models have demonstrated remarkable generation quality by learning structured latent representations of geometry and appearance [2, 26, 27]. In particular, SAM 3D [22] can optionally condition its generation on geometric inputs such as point maps. However, we observe that these conditioning mechanisms do not always faithfully adhere to the provided geometry. In this work, we focus on SAM 3D and show that its geometric consistency can instead be substantially improved through training-free guidance at inference time, without retraining or modifying the underlying model.

## 2.3. Partial 3D Conditioning

Recent developments has shown promising directions in providing partial 3D information for image-to-3D generation. In particular, [25] propose training a model to inpaint the unobserved regions of a partial point cloud. [10] demonstrates a finetuned 3D generative model to take in additional condition tokens to complete the 3D generation. Contrary to these works, our method provides a straightforward recipe to guide the generation without any retraining. [7] similarly shows a training-free approach by initializing the start of the generation with an interpolation between the partial 3D geometry encoding and Gaussian noise. We later show how applying our proposed guidance yields better results compared to the latent initialization.

## 3. Problem Formulation and Background

Building on the motivation laid out in the introduction, we now place our objective on formal footing. We first distinguish learned conditioning, which determines the pretrained generative prior, from test-time evidence, which should constrain the generated geometry (Sec. 3.1). We then review the flow-matching formulation of SAM 3D (Sec. 3.2), before showing in Sec. 4 how posterior guidance deforms the instantaneous landscape associated with its velocity field.

## 3.1. Problem Formulation

Let $c _ { I }$ denote the standard conditioning information supplied to the pretrained image-to-3D model (e.g. image and object mask), and let O denote additional geometric evidence available only at test time, such as sparse point cloud. The pretrained model defines a conditional distribution $p _ { \theta } ( x \mid c _ { I } )$ over plausible 3D assets x. Rather than requiring the learned conditioner to encode the observation faithfully, we treat O explicitly as evidence through an observation model $p ( \mathcal { O } \mid x )$ that determines whether that asset agrees with the available measurements.

The desired posterior is therefore

$$
q ( x \mid c _ { I } , \mathcal { O } ) \propto p _ { \theta } ( x \mid c _ { I } ) p ( \mathcal { O } \mid x ) ^ { \beta }\tag{1}
$$

$$
= p _ { \theta } ( \boldsymbol { x } \mid \boldsymbol { c } _ { I } ) \exp [ - \beta J _ { \mathcal { O } } ( \boldsymbol { x } ) ] ,\tag{2}
$$

where

$$
J _ { \mathcal { O } } ( x ) = - \log p ( \mathcal { O } \mid x )\tag{3}
$$

is the negative log-likelihood of the geometric observation and $\beta > 0$ controls the strength of the evidence. This formulation cleanly separates the learned prior from the physical observation model.

For the analysis below, we use c to denote an arbitrary conditioner, e.g. either image-only conditioning $c _ { I }$ or image-plus-geometry conditioning $_ { \mathit { c } _ { I , \mathcal { O } } }$ . Even when the same observation energy $J _ { \mathcal { O } }$ is used, changing c changes the pretrained flow and, consequently, the effective landscape on which the test-time guidance acts.

Our method instantiates this framework for image-to-3D generation by introducing a guidance objective tailored to partial geometric observations. Before presenting our method and the new guided flow landscape in Sec. 4, we briefly review how SAM 3D uses the flow matching for generating 3D samples.

## 3.2. Flow Matching and SAM 3D Generation

Flow Matching. Starting from random noise, flow matching gradually transforms the noise into a realistic sample by following a learned velocity field, one small step at a time. More formally, let $p ( \cdot , \cdot ) : [ 0 , 1 ] \times \mathbb { R } ^ { d } \longrightarrow \mathbb { R } _ { > 0 }$ be a probability path, i.e. for every $t ~ \in ~ [ 0 , 1 ] , \ p ( t , \cdot )$ is a probability density over $\mathbb { R } ^ { d }$ . We write $p _ { 0 } : = p ( 0 , \cdot )$ for the initial (noise) distribution and $p _ { 1 } \ : = \ p ( 1 , \cdot )$ for the target distribution. Flow matching defines a vector field $v ( \cdot , \cdot ) : [ 0 , 1 ] \times \mathbb { R } ^ { d } \longrightarrow \mathbb { R } ^ { d }$ such that, starting from a sample $x _ { 0 } \sim p _ { 0 }$ and solving the ordinary differential equation

$$
\frac { d } { d t } x _ { t } = v ( t , x _ { t } ) ,\tag{4}
$$

the resulting trajectory satisfies $x _ { t } \sim p ( t , \cdot )$ for every $t \in$ [0, 1]; in particular, $x _ { 1 } ~ \sim ~ p _ { 1 }$ is a clean sample from the target distribution. In this sense, v drives the probability path p from $p _ { 0 }$ to $p _ { 1 }$

SAM 3D. SAM 3D instantiates this flow-matching formalism for 3D asset generation, and is decomposed, following [26], into two stages. First, given image and mask, a geometric model predicts coarse shape O and object pose $R , t , s$ (rotation, translation, scale). Second, given image, mask and coarse shape, a texture model predicts the shape and texture $S , T .$ . The refined shape and texture latents are then decoded into Gaussian splats or meshes. Concretely, given a set of prediction modalities M and a set of conditioning modalities c , the model learns a conditional flow matching velocity field $v _ { \theta } ( t , x _ { t } , c ) \ [ 1 6 ] \colon$

$$
\mathcal { L } _ { \mathrm { C F M } } = \sum _ { m \in \mathcal { M } } \lambda _ { m } \mathbb { E } _ { \tau , x _ { \tau } ^ { m } } \left[ \Vert v ^ { m } - v _ { \theta } ^ { m } ( \tau , x _ { \tau } ^ { m } , c ) \Vert \right]\tag{5}
$$

where the target velocity is the linear interpolation [17]

$$
v ^ { m } = x _ { 1 } ^ { m } - x _ { 0 } ^ { m } .\tag{6}
$$

The goal is to generate $\{ x _ { 1 } ^ { m } \} _ { m \in \mathcal { M } } \sim p ( \mathcal { M } | c )$ . In particular, during the first stage (geometry model), the prediction modalities are $\mathcal { M } = \{ O , R , t , s \}$ and the conditioning modalities are $c = ( I , M )$ . During the second stage (texture and refinement model), $\mathcal { M } = \{ S , T \}$ and $c = ( O , I , M )$ At inference time, the generation proceeds by repeatedly applying the learned velocity: integrating Eq. (4) between 0 and 1 leads to the discrete update

$$
x _ { t + \Delta t } = x _ { t } + v _ { \theta } \big ( t , x _ { t } , c \big ) \cdot \Delta t ,\tag{7}
$$

which is applied iteratively from $t = 0$ to $t = 1$ to turn an initial noise sample $x _ { 0 }$ into a final prediction $x _ { 1 }$ . We discuss how this update can be modified at inference time to bias generation towards samples satisfying a desired property.

## 4. Method

The discrete update in Eq. (7) describes how SAM 3D generates a sample from its learned conditional distribution. We extend this process to incorporate partial geometric evidence $\mathcal { O }$ at test time, without retraining the model. We first characterize how posterior guidance modifies the pretrained conditional flow, interpreting it as a deformation of an instantaneous flow landscape (Sec. 4.1). We then translate this perspective into a practical point-estimate guidance rule for the learned SAM 3D velocity (Sec. 4.2). Finally, we instantiate the observation energy through a ray-consistent likelihood, physically grounded, that explicitly accounts for observed surfaces and free space (Sec. 4.3).

## 4.1. Guidance as a Deformation of the Flow Landscape

To understand how test-time evidence modifies the pretrained generative process, we first characterize guidance [1, 8] at the level of the underlying flow. In particular, we show that, for the linear path used by SAM 3D, the conditional velocity admits an instantaneous potential whose deformation under posterior guidance can be made explicit. In particular, for the linear path

$$
\begin{array} { r } { x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } , \qquad x _ { 0 } \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{8}
$$

the optimal conditional velocity can be written in terms of the score of its marginal $p _ { t } ^ { c } ( x ) = p _ { t } ( x \mid c )$ as

$$
v _ { t } ^ { c } ( x ) = \frac { x } { t } + \frac { 1 - t } { t } \nabla _ { x } \log p _ { t } ^ { c } ( x ) .\tag{9}
$$

Note that, at perfect-training limit, this velocity should converge to the velocity $v _ { \theta }$ in Eq. (7). At each fixed $t > 0$ the ideal velocity gives the instantaneous potential

$$
J _ { v } ^ { c } ( t , x ) = - \frac { \| x \| ^ { 2 } } { 2 t } - \frac { 1 - t } { t } \log p _ { t } ^ { c } ( x ) + K _ { t } ,\tag{10}
$$

$$
v _ { t } ^ { c } ( x ) = - \nabla _ { x } J _ { v } ^ { c } ( t , x ) .\tag{11}
$$

We refer to $J _ { v } ^ { c }$ as the native flow landscape. Importantly, this landscape is defined instantaneously at each t: sampling still follows the full time-dependent ODE rather than minimizing a fixed objective. However, this representation provides a convenient way to characterize how posterior guidance modifies the pretrained flow, as shown in the following proposition.

Proposition 1 (Guided flow landscape). Let $J _ { v } ^ { c } ( t , x )$ denote the instantaneous potential associated with the conditional flow. Under posterior reweighting by geometric evidence $p ( \mathcal { O } \mid x _ { 1 } ) ^ { \beta }$ , the corresponding guided potential is:

$$
J _ { v g } ^ { c } ( t , x ) = J _ { v } ^ { c } ( t , x ) - \frac { 1 - t } { t } \log h _ { t } ^ { c } ( x ) + C _ { t } ,\tag{12}
$$

where

$$
h _ { t } ^ { c } ( x ) = \mathbb { E } _ { x _ { 1 } \sim p ( x _ { 1 } | x _ { t } = x , c ) } \left[ \exp ( - \beta J _ { \mathcal { O } } ( x _ { 1 } ) ) \right] .\tag{13}
$$

Under the point-estimate approximation [4, 8] $x _ { 1 } \approx x _ { t }$ + $( 1 - t ) v _ { \theta } ( t , x _ { t } , c )$ , this becomes:

$$
J _ { v g } ^ { c } ( t , x _ { t } ) \approx J _ { v } ^ { c } ( t , x _ { t } ) + \eta _ { t } J _ { \mathcal { O } } ( x _ { 1 } ) + C _ { t } ,\tag{14}
$$

with $\begin{array} { r } { \eta _ { t } : = \beta \frac { 1 - t } { t } } \end{array}$

Changing the conditioner changes both terms in Eq. (14): it changes the landscape $J _ { v } ^ { c }$ through $p _ { t } ^ { c }$ , and it changes the observation landscape $\mathcal { E } _ { t } ^ { c } ( x _ { t } ) \equiv J _ { \mathcal { O } } ( x _ { 1 } ( x _ { t } ) )$ through the condition-dependent map x<sub>1</sub>. Also, note that $C _ { t }$ is independent from $x ,$ and therefore has no effect on the velocity field. Proposition 1 thus reveals that test-time guidance acts as an explicit deformation of the pretrained flow landscape, rather than defining a separate generative process.

## 4.2. Guided SAM 3D

The landscape analysis above gives the exact posterior correction for the ideal flow. For the learned SAM 3D velocity, we use the corresponding point-estimate approximation at every sampling step, similarly as prior training-free guidance methods [4, 8]. Specifically, we predict the clean endpoint

$$
\tilde { x } _ { 1 } ( x _ { t } ) = x _ { t } + v _ { \theta } ( t , x _ { t } , c ) ( 1 - t ) ,\tag{15}
$$

and evaluate the observation energy

$$
\mathcal { E } _ { t } ^ { c } ( x _ { t } ; \mathcal { O } ) = J _ { \mathcal { O } } ( \tilde { x } _ { 1 } ( x _ { t } ) ) .\tag{16}
$$

The guidance velocity is then

$$
\begin{array} { r } { g ( t , x _ { t } , c ) = - \lambda _ { g } ( t ) \nabla _ { x _ { t } } \mathcal { E } _ { t } ^ { c } ( x _ { t } ; \mathcal { O } ) , } \end{array}\tag{17}
$$

where $\lambda _ { g } ( t )$ contains constant arising from the exact posterior-guidance expression as well as the guidancestrength schedule used in practice. The guided velocity is

$$
\begin{array} { r } { v ^ { g } ( t , x _ { t } , c ) = v _ { \theta } ( t , x _ { t } , c ) + g ( t , x _ { t } , c ) , } \end{array}\tag{18}
$$

![](images/9362a09db059a99b4f538045a83cc486ab3318d0d97f9551230f512fb9efbc2a.jpg)  
Figure 2. Overview of Guided SAM 3D. At each step t, the current state $x _ { t }$ is decoded into a predicted clean sample $x _ { t } + v _ { t } ( 1 - t )$ , which is compared against the partial point cloud observation through an occupancy loss and a free-space loss, yielding a guidance signal ${ \mathcal { L } } .$ The velocity is corrected as $v _ { t } ^ { \prime } \gets \lambda _ { v } v _ { t } + \lambda _ { g } \nabla _ { x _ { t } } \mathcal { L }$ and fed back into the generation process, steering the trajectory towards a final sample $x _ { 1 }$ with geometry consistent with the observations.

and the discrete sampling update becomes

$$
x _ { t + \Delta t } = x _ { t } + \left( v _ { \theta } ( t , x _ { t } , c ) - \lambda _ { g } ( t ) \nabla _ { x _ { t } } \mathcal { E } _ { t } ^ { c } ( x _ { t } ; \mathcal { O } ) \right) \Delta t .\tag{19}
$$

Let us now describe our novel contribution, that is to instantiate $J _ { \mathcal { O } }$ as a physically interpretable likelihood of partial geometric observations and to show how the resulting guidance interacts with the pretrained conditional flow landscape.

## 4.3. Ray-Consistent Observation Likelihood

We now instantiate the observation energy $J _ { \mathcal { O } }$ from the given measurement geometry. We derive this energy from the likelihood of an observed camera ray under the occupancy field obtained from the predicted clean sample.

Let $\pi _ { v } ( x ) \in [ 0 , 1 ]$ denote the predicted probability that voxel v is occupied. Consider an observed ray r whose measured surface lies in voxel $s _ { r } .$ , and let ${ \mathcal { F } } _ { r }$ denote the set of voxels traversed by the ray before reaching $s _ { r }$ . The observation implies that every voxel in ${ \mathcal { F } } _ { r }$ is empty and that $s _ { r }$ is occupied. Under a Bernoulli occupancy model, the likelihood of the ray is

$$
p ( \mathcal { O } _ { r } \mid x ) = \pi _ { s _ { r } } ( x ) \prod _ { v \in \mathcal { F } _ { r } } \left( 1 - \pi _ { v } ( x ) \right) .\tag{20}
$$

Assuming conditional independence across observed rays gives

$$
p _ { \mathrm { r a y } } ( \mathcal { O } \mid x ) = \prod _ { r \in \mathcal { R } } p ( \mathcal { O } _ { r } \mid x ) ,\tag{21}
$$

and therefore the negative log-likelihood

$$
J _ { \mathrm { r a y } } ( x ; \mathcal { O } ) = - \sum _ { r \in \mathcal { R } } \log \pi _ { s _ { r } } ( x ) - \sum _ { r \in \mathcal { R } } \sum _ { v \in \mathcal { F } _ { r } } \log \left( 1 - \pi _ { v } ( x ) \right) .\tag{22}
$$

The two terms have a direct physical interpretation as the first is a surface-hit likelihood, while the second is araysurvival/free-space likelihood. A depth observation is probable only when the ray remains empty until the measured surface and becomes occupied at the measured surface.

## 5. Experiments

Baselines. We compare our method against the following:

• SAM 3D [22] is the state-of-the-art image-to-3D generation model, especially in occluded environments. We evaluate using the default proposed hyperparameters.

• MV-SAM3D [13] proposes a multi-view extension on top the original SAM 3D method by performing a weighted average across all of the velocity predictions at each timestep during the generation. We use the default hyperparameters and proposed entropy-based weighting.

GT

SAM3D

MVSAM3D

SpaceControl

Ours-MV

![](images/c0c2595a6b392113bdc54cf45ba479ab99b42f71eb2f8bc8419581106c87635f.jpg)

Figure 3. Qualitative comparison on two examples under high observability. Each example consists of two rows: the generated Gaussian splat and predicted-to-ground-truth one-sided distance heatmap (blue indicates low error, red indicates high error). Our method leads to substantially more accurate geometry while preserving the visual appearance of the generated asset.
<table><tr><td rowspan="2">Method</td><td colspan="3">Low Observability</td><td colspan="3">Medium Observability</td><td colspan="3">High Observability</td></tr><tr><td>CD↓</td><td>SD↓</td><td>LPIPS ↓</td><td>CD↓</td><td>SD↓</td><td>LPIPS↓</td><td>CD↓</td><td>SD↓</td><td>LPIPS↓</td></tr><tr><td>SAM 3D [22]</td><td>16.8</td><td>53.5</td><td>0.326</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ matching 3D cond.</td><td>15.0</td><td>50.4</td><td>0.326</td><td>15.0</td><td>50.4</td><td>0.326</td><td>15.0</td><td>50.4</td><td>0.326</td></tr><tr><td>MV-SAM3D [13]</td><td>17.5</td><td>54.9</td><td>0.326</td><td>10.1</td><td>41.4</td><td>0.300</td><td>9.60</td><td>39.6</td><td>0.282</td></tr><tr><td>+ matching 3D cond.</td><td>15.0</td><td>50.4</td><td>0.326</td><td>10.2</td><td>41.5</td><td>0.290</td><td>8.85</td><td>37.3</td><td>0.286</td></tr><tr><td>SpaceControl [7]</td><td>10.3</td><td>55.9</td><td>0.294</td><td>2.43</td><td>22.5</td><td>0.265</td><td>0.70</td><td>12.3</td><td>0.241</td></tr><tr><td>Ours</td><td>7.84</td><td>36.5</td><td>0.286</td><td>3.43</td><td>24.1</td><td>0.245</td><td>1.45</td><td>16.4</td><td>0.220</td></tr><tr><td>Ours + MV</td><td>8.11</td><td>37.2</td><td>0.289</td><td>2.83</td><td>22.0</td><td>0.224</td><td>1.02</td><td>15.7</td><td>0.201</td></tr></table>

Table 1. Quantitative comparison of 3D reconstruction accuracy using Chamfer Distance (CD) and Ground-Truth Sided Distance (SD) for the geometry, and LPIPS for the rendering. Our method are competitive or outperforms the corresponding baselines across the reported evaluation metrics and numbers of input views. Note that all distance metrics are scaled by 10<sup>3</sup>.

• SpaceControl [7] is a recent method that inputs geometric primitives into encoded latents and initializes the sampling process with these latents. To keep the input modalities consistent, we opt to not use the text conditioning in the provided SpaceControl demo and use TRELLIS original image-to-3D backbone. We initialize at generation step $\tau _ { 0 } = 6$ , the default in its evaluations.

Benchmark. We evaluate on the object-centric dataset GSO-30 [6, 12]. GSO-30 is a suite of 30 real-world objects with a full 360<sup>◦</sup> render obtained from the original GSO dataset. We evaluate in three settings: high, medium, and

SAM3D

MVSAM3D

SpaceControl

Ours-MV

![](images/38d953d0ca41f775762f0d5c453afddfe1f9b767997b12a086ca4c8efca2bab4.jpg)  
Figure 4. Qualitative comparison under medium observability. The left column shows the available RGB images (IMG1, IMG2) and the guidance partial point cloud (PC). For each method, we show the generated Gaussian splat, the predicted-to-ground-truth sided distance error heatmap (left), and ground-truth-to-predicted heatmap (right). Incorporating partial geometric observations through our training-free guidance consistently produces more accurate geometry than the baselines, while remaining complementary to multi-view generation. The colored tags indicate the inputs available to each method: black for IMG1, blue for IMG2, orange for partial point cloud (PC).

low observability. Each observability setting is determined by randomly selecting 5, 2, and 1 image(s) respectively, such that the images in each lower observability setting is strictly a subset of the higher observability’s images. In each setting, we simulate RGB-D measurements by projecting the ground-truth mesh onto the selected views and only using the points visible from those views. In the case of a method accepting multiple RGB images such as MV-SAM3D, all of the selected view images are used as input. Both SAM 3D and MV-SAM3D use MoGe [23] to estimate pointmaps that are embedded and used for conditioning. To keep the measurements consistent, we also evaluate against passing in the same RGB-D measurements into the conditioning embedder, which we denote as “matching 3D cond.” in Table 1. SAM 3D only receives one input image, so we only report for the low observability case, which is . We additionally evaluate on extending our method to MV-SAM3D, in which we perform the velocity weighting scheme proposed and additionally apply classifier guidance.

Metrics. We evaluate on the following geometry and novel view metrics:

• Chamfer Distance measures the bidirectional distance between the source pointcloud and the nearest neighbor in the target pointcloud. We report the squared Chamfer Distance, scaled by 10<sup>3</sup>.

• Ground-truth Sided Distance measures specifically the distance between each ground-truth point to its nearest neighbor in the prediction pointcloud. We report this part of the Chamfer Distance calculation to emphasize any inaccuracies the prediction may have with gaping holes or missing regions. We again report this scaled by 10<sup>3</sup>.

• LPIPS [30] computes the differences between each layer activation of VGG-16, which corresponds well with human perceptual features. We evaluate the mean LPIPS across 8 randomly sampled novel views that are at least 30 degrees away from the selected source views. Each baseline method receives the same selected novel views.

For the geometric metrics, we follow the SAM 3D evaluation protocol by independently normalizing both the ground truth and predicted meshes between [-1, 1] and performing ICP to align with the ground truth before computing the metrics. We randomly sample 30,000 points on each mesh. To evaluate LPIPS, we use the decoded Gaussian splat from each method to render the novel views.

Table 2. Ablations on the guidance design decisions. Run across the GSO30 dataset with high observability, single-view setup.
<table><tr><td>Ablated Feature</td><td>CD↓</td><td>LPIPS↓</td></tr><tr><td rowspan="3">Baseline  $( \lambda _ { g } = 1 , \lambda _ { f } = 1$  , steps=50) (+)  $\lambda _ { g } = 4$ </td><td>15.1</td><td>0.341</td></tr><tr><td>4.01</td><td>0.287</td></tr><tr><td>3.54</td><td>0.275</td></tr><tr><td>(+) (+) Increase steps to 200</td><td>0.98</td><td>0.228</td></tr><tr><td>(+) Guidance weight cooldown</td><td>1.45</td><td>0.220</td></tr></table>

Results. Table 1 shows our quantitative results. In all observabilities, we greatly surpass the performance of both SAM 3D and MV-SAM3D. In low observability, we outperform all baselines in geometric and novel view metrics, and we maintain competitive geometric performance compared to SpaceControl in medium and high observabilities while consistently outperforming in novel view synthesis.

Figures 3 and 4 visualize the Gaussian splat predictions from each method and the corresponding error heatmap compared to the ground truth. These figures further exemplify the importance of explicit guidance of partial observations compared to only using RGB images or the latent initialization approach as in SpaceControl. Since SAM 3D does not take in any additional inputs, the resulting shape is ambiguous and is unable to correct the imperfect geometry from any additional signals. Using more RGB images can help as seen in MV-SAM3D, but is not sufficient and still results in inaccuracies in the predicted geometry. The latent initialization done in SpaceControl performs well under high observability, but the ground-truth-to-predicted heat maps in Figure 4 shows a major limitation with more limited views – without enough coverage, SpaceControl is prone to missing large portions of the geometry, as seen in the missing backs of the alarm clock and backpack, and the empty bottom surface of the wooden blocks. In contrast, our method consistently adheres to the geometry while maintaining more faithful visual appearances in novel views.

## 6. Ablations

Table 2 shows what decision decisions were helpful for successfully applying classifier guidance to SAM 3D. The ablation is performed across the entire GOS30 dataset with high observability without any multiview images for simplicity. We found that without any additional weighting to the guidance gradient, the method performed on par with vanilla SAM 3D, even after normalizing the gradient to match the magnitude of the original velocity. Only by significantly upweighting the guidance gradient did we find major improvements into adhering to the geometry as well as more realistic renderings in novel views. We also found that the performance could be further improved by upweighting the freespace loss, as the magnitude of the occupancy loss largely dominates the total loss. Increasing the number of sampling steps yielded major improvement in the performance as well. We hypothesize that the generation is not accustomed to the application of the test-time guidance, which requires more steps to accommodate. While we obtained good geometric accuracy from this point, we observed that the visual quality of the resulting 3D objects were unsatisfactory, with unsmooth surfaces along with some artifacts such as holes or floaters. Applying a guidance weight cooldown schedule reduced its geometric performance, but this improved the appearances as well as the novel view renderings.

## 7. Conclusion

We presented a training-free framework for incorporating partial geometric observations into pretrained image-to-3D generation models at inference time. Beyond the practical guidance mechanism, we interpreted posterior guidance as a deformation of the pretrained conditional flow landscape. Building on this perspective, we derived a ray-consistent observation likelihood in the model’s occupancy representation, combining surface occupancy and free-space evidence to steer generation towards the observed geometry while retaining the learned prior to complete unobserved regions. Applied to SAM 3D and its multi-view extension, our approach consistently improves geometric fidelity across different levels of observability while preserving visual quality. Our experiments further highlight that explicitly enforcing geometric evidence through guidance can be substantially more effective than providing the same information solely through the model’s learned conditioning pathway. Together, these results show that partial geometric evidence can be incorporated explicitly at test time to complement, rather than replace, the powerful priors learned by image-to-3D foundation models. Our work also has several limitations. First, we assume that the partial observations are accurate and expressed in the model’s canonical coordinate system. Handling noisy observations or unknown coordinate frames would require more robust observation models or joint alignment strategies. Second, our geometric guidance operates on an occupancy-grid representation and therefore inherits its finite spatial resolution. More generally, our formulation opens the possibility of designing observation likelihoods for other forms of geometric evidence and intermediate 3D representations. While we demonstrate our approach on SAM 3D and its multi-view extension, extending explicit geometric guidance to other image-to-3D generative models remains another promising direction.

## References

[1] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Roni Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In International Conference on Learning Representations (ICLR), pages 51304–51323, 2024. 2, 4

[2] Mark Boss, Zixuan Huang, Aaryaman Vasishta, and Varun Jampani. Sf3d: Stable fast 3d mesh reconstruction with uvunwrapping and illumination disentanglement. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16240–16250, 2025. 3

[3] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22246–22256, 2023. 2

[4] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022. 2, 4

[5] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2

[6] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items. In International Conference on Robotics and Automation (ICRA), page 2553–2560. IEEE Press, 2022. 6

[7] Elisabetta Fedele, Francis Engelmann, Ian Huang, Or Litany, Marc Pollefeys, and Leonidas Guibas. SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling. In International Conference on Learning Representations (ICLR), 2026. 3, 6

[8] Ruiqi Feng, Chenglei Yu, Wenhao Deng, Peiyan Hu, and Tailin Wu. On the guidance of flow matching. arXiv preprint arXiv:2502.02150, 2025. 2, 4

[9] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2

[10] Anita Hu and Maria Shugrina. Axolotl3d: a unified framework for faithful 3d shape completion. In European Conference on Computer Vision (ECCV), 2026. 3

[11] Jeongsol Kim, Bryan Sangwoo Kim, and Jong Chul Ye. Flowdps: Flow-driven posterior sampling for inverse problems. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12328–12337, 2025. 2

[12] Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. Eschernet: A generative model for scalable view synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 6

[13] Baicheng Li, Dong Wu, Jun Li, Shunkai Zhou, Zecui Zeng, Lusong Li, and Hongbin Zha. Mv-sam3d: Adaptive multiview fusion for layout-aware 3d generation. arXiv preprint arXiv:2603.11633, 2026. 1, 2, 5, 6

[14] Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang Zhang, Wenhan Luo,

Ping Tan, et al. Era3d: High-resolution multiview diffusion using efficient row-wise attention. Advances in Neural Infor mation Processing Systems, 37:55975–56000, 2024. 2

[15] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 300–309, 2023. 2

[16] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximil ian Nickel, and Matt Le. Flow matching for generative mod eling. arXiv preprint arXiv:2210.02747, 2022. 3

[17] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 3

[18] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Sin gle image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9970–9980, 2024. 2

[19] Maitreya Patel, SongWen, Dimitris N. Metaxas, and Yezhou Yang. Flowchef: Steering of rectified flow models for controlled generations. IEEE/CVF International Conference on Computer Vision (ICCV), pages 15308–15318, 2025. 2

[20] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden hall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2

[21] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d gen eration. In International conference on learning representa tions, pages 39838–39859, 2024. 2

[22] SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jiawei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollar, Georgia Gkioxari,´ Matt Feiszli, and Jitendra Malik. Sam 3d: 3dfy anything in images. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 1, 2, 3, 5, 6

[23] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5261–5271, 2025. 7

[24] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36: 8406–8441, 2023. 2

[25] Jiatong Xia, Zicheng Duan, Anton van den Hengel, and Lingqiao Liu. Points-to-3d: Structure-aware 3d generation with point cloud priors. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. 3

[26] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong

Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 21469–21480, 2025. 1, 3

[27] Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14419–14429, 2026. 3

[28] Haotian Ye, Haowei Lin, Jiaqi Han, Minkai Xu, Sheng Liu, Yitao Liang, Jianzhu Ma, James Zou, and Stefano Ermon. Tfg: Unified training-free guidance for diffusion models. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 2

[29] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided conditional diffusion model. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2

[30] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 7

# Guiding Image-to-3D Generation with Test-Time Partial Observations Supplementary Material

This supplemental material is organized as follows: Sec. A presents a proof for Proposition 1. Sec. B shows an extended set of visualizations.

## A. Proof of Proposition 1

Let us define

$$
h _ { t } ^ { c } ( x ) = \mathbb { E } _ { x _ { 1 } \sim p ( x _ { 1 } | x _ { t } = x , c ) } \left[ \exp ( - \beta J _ { \mathcal { O } } ( x _ { 1 } ) ) \right] .\tag{23}
$$

The corresponding intermediate marginal satisfies $q _ { t } ^ { c } ( x ) \propto$ $p _ { t } ^ { c } ( x ) h _ { t } ^ { c } ( x )$ . Its ideal velocity is therefore

$$
v _ { v g } ^ { c } ( t , x ) = v _ { t } ^ { c } ( x ) + \frac { 1 - t } { t } \nabla _ { x } \log h _ { t } ^ { c } ( x ) .\tag{24}
$$

Equivalently, substituting q<sup>c</sup> for $\mathbf { \nabla } p _ { t } ^ { c }$ in Eq. (11) gives the exact guided potential

$$
J _ { v g } ^ { c } ( t , x ) = J _ { v } ^ { c } ( t , x ) - \frac { 1 - t } { t } \log h _ { t } ^ { c } ( x ) + C _ { t } ,\tag{25}
$$

where $C _ { t }$ does not depend on x and therefore has no effect on the velocity field.

## B. Extended Visualizations

Figures 5 and 6 show more detailed comparisons between all baselines. Notably, we include the visualizations for SAM 3D and MV-SAM 3D using its default MoGe conditioning as well as inputting the partial point cloud for pointmap conditioning, labelled “matching PC cond.”. Even when using the same partial point cloud as conditioning, we have found only minor improvements in the performance, which is also observed quantitatively in Table 1. We additionally show the visualizations from our method Ours-SV (single-view). Figure 5 in particular demonstrates the effectiveness of applying our method to single-view SAM 3D, outperforming all other baselines. SpaceControl here is unable to generate the unobserved regions, highlighting the importance of explicitly applying guidance over the latent initialization. Ours-MV further improves the performance using additional multi-view information to constrain the generation.

![](images/3c0d7b2f19328af331d013345b1cf67047ff63e16080b4e70b2fe2b17f978dcf.jpg)  
Figure 5. Qualitative comparison of the “lunch bag” object under high observability. At the top, we show all potential image inputs and the partial point cloud. Each baseline shows the fronts and backs of the Gaussian splat (GS), the GT-to-prediction sided error heatmap, and the prediction-to-GT sided error heatmap. Each method is color-coded with the corresponding accepted inputs.

![](images/9af44f78c6f20700f7040d360e57822678fad6fad8aac199eba47775bd362dd9.jpg)

IMG1

Inputs

![](images/dc6e5cf9fdebe2ead99c28c9402c9b95aded3f14196805a5e405ce751a8b66b0.jpg)

IMG5

GS Front

GS Back

GT-to-Pred

GT-to-Pred

Pred-to-GT

![](images/f489ffdc3798fa2a88b752fb8df273bc5c3a5c381e0c810d92900fc4e0c8bdb2.jpg)

Pred-to-GT Back

![](images/f20bd9992d311baff258ce6535236cdd10a1535410cb588ec7ecebb153117f26.jpg)

Figure 6. Qualitative comparison of the “grandfather” object under high observability. At the top, we show all potential image inputs and the partial point cloud. Each baseline shows the fronts and backs of the Gaussian splat (GS), the GT-to-prediction sided error heatmap, and the prediction-to-GT sided error heatmap. Each method is color-coded with the corresponding accepted inputs.