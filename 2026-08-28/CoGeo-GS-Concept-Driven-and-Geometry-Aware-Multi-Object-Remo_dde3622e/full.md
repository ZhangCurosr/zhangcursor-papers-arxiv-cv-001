# CoGeo-GS: Concept-Driven and Geometry-Aware Multi-Object Removal in 3D Scenes

Yuanxiang Ni<sup>1</sup> Xianliang Huang<sup>2,†</sup> Chenhang Ma<sup>3</sup> Chen Xiao<sup>4</sup> Yuewen Ma<sup>2</sup> Ruxin Wang<sup>5,\*</sup> Hao Zhang<sup>5,\*</sup>

<sup>1</sup>Southern University of Science and Technology, Shenzhen, China <sup>2</sup>PICO, ByteDance, Beijing, China

<sup>3</sup>Zhejiang University, Hangzhou, China <sup>4</sup>Fudan University, Shanghai, China

<sup>5</sup>Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China

Abstract—Multi-object removal in 3D scenes is challenging due to severe occlusions, semantic entanglement, and the difficulty of maintaining geometric and multi-view consistency. Existing 3D Gaussian Splatting (3DGS) methods perform well for singleobject editing but scale poorly to multi-object scenarios, often requiring repetitive optimization and yielding unstable geometry in removed regions. We propose CoGeo-GS, a concept-driven framework for controllable multi-object removal in 3D scenes. CoGeo-GS assigns concept-aware semantic tags to Gaussians, enabling flexible object selection and reducing interference between foreground objects and background structures within a single optimization stage. To recover plausible geometry, we introduce a geometry-aware completion pipeline that combines monocular depth priors with diffusion-based refinement and boundaryaligned blending. A geometry-regularized refinement strategy further stabilizes reconstruction and preserves multi-view consistency. Experiments demonstrate that CoGeo-GS outperforms existing methods in visual quality and reconstruction fidelity.

Index Terms—3D Gaussian Splatting, multi-object removal, concept-aware tagging, monocular depth priors, multi-view consistency

## I. INTRODUCTION

3D scene editing is becoming a core capability for applications such as virtual reality (VR) [1], autonomous driving [2], and robot interaction [3], [4]. Object removal requires plausible completion of the occluded background for controllable 3D editing. Real-world cluttered scenes involve multiple interacting objects and occlusions, requiring precise multi-target specification and geometric completion. Despite recent advances in neural rendering and scene representation, achieving reliable multi-object removal remains challenging. Moreover, maintaining global multi-view consistency and coherent appearance across reconstructed scene further increases the task difficulty.

Recent works based on 3D Gaussian Splatting (3DGS) [5] have demonstrated promising results for single-object editing [6], [7] and inpainting [8]. However, moving from singleobject to multi-object scenarios exposes two core issues. First, most approaches still rely on repeated per-object optimization, which becomes computationally prohibitive and prone to error accumulation as the number of targets grows. Second, nearby objects and shared background structures are often entangled in the representation, so editing one object easily perturbs others and leaves occluded regions poorly constrained, leading to semantic interference, distorted geometry, and unstable depth where content has been removed.

To tackle these limitations, we introduce CoGeo-GS, a concept-driven geometric completion framework for highquality multi-object removal in 3D Gaussian scenes. CoGeo-GS first performs concept-aware semantic tagging on Gaussians. Each primitive receives a compact concept encoding. This allows flexible, user-controllable selection of arbitrary object subsets within a single optimization stage. It also disentangles foreground objects from background structures. As a result, mutual interference is greatly reduced while the editable, spatially explicit 3D representation is preserved.

On top of this concept-level decomposition, we design a geometry-aware completion pipeline that jointly exploits monocular depth priors and generative refinement. We first extract structural priors from a monocular depth foundation model and align them to the 3DGS coordinate system to obtain seed depth in the removed regions. These seeds are then refined by a diffusion-based inpainting module that recovers dense, fine-grained depth for occluded areas, while a boundary-aligned blending scheme enforces smooth transitions and surface continuity between completed and preserved regions. This hybrid strategy yields geometrically stable, highfidelity reconstructions in the edited areas. To further stabilize appearance and maintain multi-view coherence, we introduce a geometry-regularized refinement stage. We first freeze geometry for a few steps and then allow only small, trust-regionlimited geometric updates while refining appearance under image-level supervision from the edited views.

In summary, our main contributions are: (1) We propose CoGeo-GS, a concept-driven framework for controllable multi-object removal in 3D Gaussian Splatting, performing object-level editing in a single optimization stage while maintaining an explicit, editable 3D representation. (2) We introduce a hybrid geometric completion pipeline anchoring monocular depth priors to scene geometry and refining missing regions via diffusion-based depth completion, enabling accurate, consistent reconstruction without boundary artifacts. (3) We design a geometry-regularized refinement strategy decoupling appearance adaptation from constrained geometric updates via freezing and trust-region clipping, improving cross-view consistency and fusion stability. (4) Extensive experiments on diverse scenes show CoGeo-GS consistently outperforms existing multi-object removal methods in visual quality and reconstruction fidelity.

## II. RELATED WORK

## A. Object-Centric 2D & 3D Inpainting

Image inpainting is a fundamental task in creating appropriate content for the target area. Traditional patch-based approaches [9], [10] and GAN-driven methods [11], [12] are designed to handle regular regions but struggle with complex occlusions. Diffusion models [13], [14] demonstrate superior ability in generating semantically coherent content for large missing areas [15], [16]. Building upon the advancements in 2D image inpainting, the extension to 3D scenes reconstructed by NeRF and 3DGS is still a challenging task. 3D inpainting demands handling intricate spatial structures and multi-view consistency. While NeRF-based approaches [17]–[19] achieve partial success with static volumetric representations, their effectiveness remains constrained by inherent architectural limitations. Explicit 3DGS inpainting methods like InFusion [20] and GaussianEditor [6] primarily address single-object-centric removal from an existing static Gaussian Splatting scene, overlooking multiple incidental objects present during scene capture. These gaps motivate our concept-driven, scale-aligned geometric completion framework in an explicit 3DGS representation.

## B. Multi-Object Removal

Recent advancements in scene editing [21], [22] have investigated various 2D and 3D representations for removing or modifying multiple objects in complex scenes. In the 2D domain, large diffusion models enable high-fidelity inpainting via generative priors from massive image datasets, but they lack 3D consistency and geometric awareness. To address this, techniques like Inpaint3D [21] use diffusion priors to guide NeRF reconstructions, facilitating object removal and viewconsistent scene completion. Gaussian-based methods such as Gaussian Grouping [7] support instance-level segmentation and editing through mask-guided clustering. However, they struggle to remove multiple objects, as optimizing the corresponding Gaussian primitives with consistent and high-quality textures remains difficult. Our work combines concept-aware tagging with depth-guided completion to support controllable multi-object removal in complex 3DGS scenes.

## III. METHOD

We propose CoGeo-GS, a multi-object removal framework built on concept-driven Gaussian representations. As shown in Fig. 1, our framework integrates concept-aware semantic tagging, depth-guided geometric completion, and geometryregularized appearance refinement. Together, these modules produce semantically coherent and geometrically stable 3D reconstructions across multiple views.

## A. Background and Problem Definition

3D Gaussian Splatting (3DGS) represents a scene as a set of Gaussians $\mathcal { G } = \{ g _ { i } \} _ { i = 1 } ^ { N }$ and renders images via differentiable α-blending. Each Gaussian $g _ { i } \ = \ ( \mu _ { i } , \pmb { \Sigma } _ { i } , \alpha _ { i } , \mathbf { c } _ { i } )$ is defined by a 3D center $\mu _ { i } ,$ covariance $\Sigma _ { i }$ , opacity $\alpha _ { i } ,$ and SH-based color coefficients $\mathbf { c } _ { i }$ . After projection onto the image plane, the color $C$ at a pixel is computed via α-blending of depth-sorted Gaussians:

$$
C = \sum _ { i \in \mathcal { N } } \mathbf { c } _ { i } \alpha _ { i } T _ { i } , \quad T _ { i } = \prod _ { j = 1 } ^ { i - 1 } ( 1 - \alpha _ { j } ) ,\tag{1}
$$

where $\mathcal { N }$ is the set of Gaussians overlapping the pixel and $T _ { i }$ is the accumulated transmittance.

Given a reconstructed 3D Gaussian scene $\mathcal { G }$ and a set of concept masks that specify the objects to be removed, our task is to modify $\mathcal { G }$ such that the resulting scene remains geometrically complete and visually consistent. This requires solving three coupled challenges. First, all Gaussians associated with the target objects must be reliably identified and removed. Second, the geometry missing from the removed regions must be plausibly completed while preserving the surrounding background structure. Third, the final scene should support photorealistic rendering with appearance consistent across all views. Our goal is a semantically correct, geometrically stable, and visually coherent Gaussian field after removal.

## B. Concept-Aware Tagging for Multi-Object Removal

To enable fine-grained object-level editing, we assign each Gaussian primitive a unique object identity that is consistent across views. Direct 2D-to-3D projection yields viewinconsistent labels, so we use offline text-driven preprocessing. For each target object, we provide a text prompt and use Grounding DINO to generate text-conditioned spatial prompts (e.g., bounding boxes), which guide SAM to extract the corresponding masks in each view. Since all views share the same set of prompts and indexing order, assigning label q to pixels within the mask of the q-th prompt produces a cross-view consistent label map O. This map satisfies ${ \cal O } ( u ) \in \{ 0 , 1 , \ldots , Q \}$ , where 0 denotes the background and $q \in \{ 1 , \ldots , Q \}$ indexes the $Q$ target objects.

We further distill the 2D label map O into the 3D Gaussian field by learning per-primitive feature vectors. Specifically, we assign a learnable feature vector $f _ { i } \in \mathbb { R } ^ { D }$ to each Gaussian i. During rendering, we apply Gaussian splatting to these features and perform front-to-back α-compositing to obtain a per-pixel feature map $F ( u )$

$$
F ( u ) = \sum _ { i \in \mathcal { N } _ { u } } \alpha _ { i } ( u ) T _ { i } ( u ) f _ { i } , T _ { i } ( u ) = \prod _ { j < i } \bigl ( 1 - \alpha _ { j } ( u ) \bigr ) ,\tag{2}
$$

where $u \ = \ ( x , y )$ denotes pixel coordinates, $\mathcal { N } _ { u }$ is the set of Gaussians contributing to pixel u (sorted by depth), $\alpha _ { i } ( u )$ is the opacity contribution, and $T _ { i } ( u )$ is the accumulated transmittance. We apply a linear classifier $\Phi : \mathbb { R } ^ { D }  \mathbb { R } ^ { Q + 1 }$ followed by the softmax function to obtain per-pixel object identity predictions:

$$
\bar { O } ( u ) = \operatorname { s o f t m a x } \bigl ( \Phi ( F ( u ) ) \bigr ) .\tag{3}
$$

![](images/c6bb71dbfb78c1bdb0535a18621f489b61fb72522255cf247342098ff61e449c.jpg)  
Fig. 1. Overview of the CoGeo-GS framework. (a) We first distill text-driven segmentation masks into 3D Gaussians to assign object-level identities. (b) Then, geometric voids are completed by conditioning a latent diffusion model on scale-aligned monocular depth priors (depth anchors) and VAE-encoded RGB latents. (c) The completed geometry is fused with the background via overlap-aware pruning. (d) A two-stage refinement strategy first optimizes appearance, then geometric constraint was subsequently employed to eliminate inconsistencies in fusion seams while preserving structural integrity.

The predictions $\bar { O } ( u )$ are aligned with $O ( u )$ using a multiclass cross-entropy loss over the pixel set $P \colon$

$$
L _ { \mathrm { o b j } } = - \frac { 1 } { | P | } \sum _ { u \in P } \mathbb { E } _ { c \sim O ( u ) } \big [ \log \bar { O } _ { c } ( u ) \big ] .\tag{4}
$$

To propagate 3D supervision to occluded Gaussians near geometric boundaries, we enforce neighborhood consistency in 3D space. Let $x _ { i }$ be the center of Gaussian i and $\kappa ( \boldsymbol { x } _ { i } )$ denote its k-nearest neighbors. We define the object-identity distribution of each Gaussian as:

$$
p _ { i } = \operatorname { s o f t m a x } \bigl ( \Phi ( f _ { i } ) \bigr ) ,\tag{5}
$$

where Φ shares weights with the 2D pixel classification branch. Let Ω denote the set of Gaussians sampled for computing $L _ { \mathrm { s p a c e } }$ . The spatial consistency loss is defined as:

$$
L _ { \mathrm { s p a c e } } = { \frac { 1 } { | \Omega | } } \sum _ { i \in \Omega } { \frac { 1 } { k } } \sum _ { j \in { \mathcal { K } } ( x _ { i } ) } \operatorname { K L } \bigl ( p _ { i } \| p _ { j } \bigr ) .\tag{6}
$$

The final distillation objective is utilized to optimize each identity in Gaussians:

$$
L _ { \mathrm { D i s } } = L _ { \mathrm { o b j } } + \lambda L _ { \mathrm { s p a c e } } .\tag{7}
$$

Through the synergy of 2D supervision and 3D regularization, each Gaussian gradually acquires a stable, cross-view consistent object identity, enabling subsequent object-level editing.

## C. Depth-Guided Geometric Completion

Removing target Gaussians creates geometric holes. RGB inpainting with back-projection often breaks cross-view consistency, while diffusion-based depth completion without scale anchoring can drift in scale. We therefore propose a two-stage depth recovery pipeline that anchors global scale and then refines local geometric details.

Stage 1: Scale Anchoring with Monocular Priors. Given the inpainted image $I _ { \mathrm { I n p } } ^ { ( v ) }$ and hole mask $M ^ { ( v ) }$ for view v, we apply the monocular depth model Depth Anything 3 to estimate relative depth $D _ { \mathrm { m o n c } } ^ { ( v ) }$ . Since monocular predictions suffer from scale ambiguity and global shift, we align them to the current 3DGS geometry using reliable background regions.

Specifically, we estimate scale and shift parameters $( a ^ { * } , b ^ { * } )$ by least-squares fitting between the monocular prediction and the rendered depth $D _ { \mathrm { r e n d e r } } ^ { \bar { ( } v \bar { ) } }$ over the background region $\Omega _ { \mathrm { b g } } ^ { ( v ) }$

$$
( a ^ { * } , b ^ { * } ) = \arg \operatorname* { m i n } _ { a , b } \sum _ { \boldsymbol { u } \in \Omega _ { \mathrm { b g } } ^ { ( v ) } } \left( a D _ { \mathrm { m o n o } } ^ { ( v ) } ( \boldsymbol { u } ) + b - D _ { \mathrm { r e n d e r } } ^ { ( v ) } ( \boldsymbol { u } ) \right) ^ { 2 } .\tag{8}
$$

The aligned depth is then obtained as $D _ { \mathrm { a l i g n } } ^ { ( v ) } = a ^ { * } D _ { \mathrm { m o n o } } ^ { ( v ) } + \dot { b } ^ { * }$ To ensure smooth transitions at hole boundaries, we blend the aligned depth with the rendered depth using a softly dilated mask $M _ { \mathrm { s o f f } } ^ { ( v ) }$

$$
D _ { \mathrm { a n c h o r } } ^ { ( v ) } = M _ { \mathrm { s o f t } } ^ { ( v ) } \odot D _ { \mathrm { a l i g n } } ^ { ( v ) } + ( 1 - M _ { \mathrm { s o f t } } ^ { ( v ) } ) \odot D _ { \mathrm { r e n d e r } } ^ { ( v ) } .\tag{9}
$$

This anchor depth provides a globally consistent geometric scaffold for subsequent refinement.

Stage 2: Detail Refinement with Diffusion Priors. While $D _ { \mathrm { a n c h o t } } ^ { ( v ) }$ ensures correct scale and coarse structure, monocular priors alone are insufficient to recover fine-grained geometry in heavily occluded regions. We therefore employ a pre-trained depth diffusion model (InFusion) as a plug-and-play prior to refine local details.

The diffusion process is conditioned on the inpainted RGB image, the anchor depth, and the hole mask by concatenating their latent representations:

$$
z _ { \mathrm { i n } } = \mathrm { c o n c a t } \big ( z _ { \mathrm { r g b } } , z _ { d , t } , z _ { \mathrm { m a s k , } } z _ { \mathrm { m a s k _ { - } d } } \big ) .\tag{10}
$$

To prevent geometric drift during sampling, we enforce a scale-anchored masked constraint at each diffusion step, explicitly fixing the background depth to the noised anchor depth:

$$
z _ { d , t } \gets ( 1 - z _ { \mathrm { m a s k } } ) \odot z _ { \mathrm { a n c h o r \_ n o i s e d } , t } + z _ { \mathrm { m a s k } } \odot z _ { d , t } .\tag{11}
$$

This constraint restricts the diffusion model to synthesize high-frequency geometry only within missing regions, while preserving global structure and scale consistency. The final completed depth $D _ { \mathrm { i n p a i n t } } ^ { ( v ) }$ is obtained via reverse diffusion.

## D. Gaussian Fusion and Geometry-Regularized Refinement

After obtaining the inpainted depth $D _ { \mathrm { i n p a i n t } } ^ { ( v ) } ,$ we back-project it into 3D space to generate a point cloud and convert it into a set of Gaussians $\mathcal { G } _ { \mathrm { p a t c h } }$ . To mitigate redundant geometry and floater artifacts at the fusion interface, we implement an overlap-aware pruning strategy targeting the background Gaussian set $\mathcal { G } _ { \mathrm { b g } }$ near the hole boundary. Specifically, a background Gaussian $i \in \mathcal { G } _ { \mathrm { b g } }$ is pruned if it satisfies both of the following criteria:

• Proximity overlap: It is spatially adjacent to the new patch: min $\mathsf { l } _ { k \in \mathcal { G } _ { \mathrm { p a t c h } } } \left\| \mu _ { i } - \mu _ { k } \right\| < \delta _ { \mathrm { n e a r } }$ , where $\mu _ { i }$ and $\mu _ { k }$ denote the 3D center positions of Gaussians i and k, respectively.

• Local sparsity: It is identified as a sparse outlier, i.e., the neighbor count within a radius r falls below $n _ { \mathrm { m i n } }$

The remaining background Gaussians are merged with $\mathcal { G } _ { \mathrm { p a t c h } }$ to form the initial fused scene $\mathcal { G } _ { \mathrm { i n i t } }$

To enhance consistency and prevent structural drift, we adopt a two-stage refinement strategy with geometric regularization. We first fix all geometric parameters and optimize only SH coefficients and opacity, allowing the newly inserted Gaussians to match surrounding appearance without altering the underlying structure. Using the inpainted image $I _ { \mathrm { I n p } } ^ { ( v ) }$ as reference, we minimize the photometric loss:

$$
L _ { \mathrm { r e f } } = \left( 1 - \lambda _ { \mathrm { s s i m } } \right) \| I _ { \mathrm { r e n d e r } } - I _ { \mathrm { I n p } } ^ { ( v ) } \| _ { 1 } + \lambda _ { \mathrm { s s i m } } \left( 1 - \mathrm { S S I M } ( I _ { \mathrm { r e n d e r } } , I _ { \mathrm { I n p } } ^ { ( v ) } ) \right) .\tag{12}
$$

Subsequently, we enable updates to geometric parameters to eliminate residual seams at the fusion boundary. To prevent unintended deformation of unedited regions, we constrain each geometric update $\Delta \theta$ within a trust region by norm clipping:

$$
\Delta \theta \gets \Delta \theta \cdot \operatorname* { m i n } \left( 1 , \frac { \varepsilon } { \| \Delta \theta \| _ { 2 } } \right) .\tag{13}
$$

This constraint is applied independently to each parameter tensor, enabling local seam correction while maintaining global geometric consistency.

## IV. EXPERIMENTS

## A. Experimental Setup

Datasets. We evaluate CoGeo-GS on the standard public benchmarks Mip-NeRF 360 [23] and SPIn-NeRF [17]. These datasets are selected to comprehensively validate the effectiveness of our method across both bounded and unbounded environments. To ensure fair benchmarking, all methods are evaluated using the provided camera intrinsics and initialized SfM point clouds from COLMAP [24].

Baselines and Metrics. We compare our method with four state-of-the-art 3D inpainting and editing methods: InFusion [20], SPIn-NeRF [17], GaussianEditor [6] and Gaussian Grouping [7]. For fair comparison, we retrain all baseline models on the benchmarks using their official open-source implementations. We employ PSNR, SSIM, LPIPS, and FID to evaluate visual quality. Crucially, to strictly assess the synthesis quality of the edited content, all metrics are computed exclusively on the masked regions.

Implementation Details. Experiments run on a single NVIDIA RTX 4090 GPU. 3DGS optimization uses 30k iterations at original resolution. In the concept-aware distillation stage, we set the object feature dimension to $D = 1 6$ and apply spatial consistency regularization using $k = 5$ nearest neighbors with weight $\lambda = 0 . 1$ . For geometric completion, monocular depth estimation is performed at 504×504 resolution to provide global scale anchors, followed by diffusion-based refinement at $7 6 8 \times 7 6 8$ resolution for detail recovery. After fusion, the scene is fine-tuned for 150 iterations using an $\mathcal { L } _ { 1 }$ and SSIM loss with $\lambda _ { \mathrm { s s i m } } = 0 . 2$ . To ensure stability, geometric parameters are frozen during the initial optimization phase and subsequently updated under a trust-region constraint.

## B. Comparison Results

We evaluate our method against state-of-the-art baselines on both multi-object and single-object removal tasks.

Multi-object Removal Results. As shown in Fig. 2, GaussianEditor [6] improves appearance but often introduces obviously black holes in removing multiple objects. InFusion [20] tends to produce similar results and inconsistent completion occurs near boundaries, causing floating artifacts due to scale mismatches. Gaussian Grouping [7] removes multiple objects but suffers from floating artifacts and inconsistent appearance between background and inpainting regions. In contrast, our CoGeo-GS utilizes depth-guided completion to maintain structural integrity across views. Quantitative results are demonstrated in Task 1 of Tab. I. For multi-object removal, CoGeo-GS improves PSNR and SSIM over the strongest baseline, while reducing LPIPS and FID by 56.6% and 50.0%, respectively. Our method effectively handles complex occlusions significantly better than existing approaches.

Single-object Removal Results. Fig. 3 presents the comparison for single-object removal. SPIn-NeRF [17] maintains

![](images/33de11564fa4a3fe8f77acd4b4a81993498514fa191f060680b5740de0f5d7cf.jpg)

TABLE I  
QUANTITATIVE COMPARISON ON MULTI-OBJECT AND SINGLE-OBJECT REMOVAL TASKS. HIGHER PSNR AND SSIM INDICATE BETTER RENDERING FIDELITY, WHILE LOWER LPIPS AND FID INDICATE BETTER PERCEPTUAL QUALITY.
<table><tr><td rowspan="2">Method</td><td colspan="4">Task 1: Multi-Object Removal</td><td colspan="4">Task 2: Single-Object Removal</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td></tr><tr><td>GaussianEditor [6]</td><td>22.4</td><td>0.742</td><td>0.284</td><td>31.6</td><td>24.1</td><td>0.781</td><td>0.247</td><td>28.4</td></tr><tr><td>InFusion [20]</td><td>24.8</td><td>0.803</td><td>0.213</td><td>24.9</td><td>26.5</td><td>0.831</td><td>0.186</td><td>21.7</td></tr><tr><td>Gaussian Grouping [7]</td><td>25.6</td><td>0.821</td><td>0.198</td><td>22.8</td><td>27.3</td><td>0.847</td><td>0.172</td><td>19.6</td></tr><tr><td>SPIn-NeRF [17]</td><td>24.9</td><td>0.812</td><td>0.221</td><td>23.7</td><td>26.8</td><td>0.842</td><td>0.189</td><td>20.9</td></tr><tr><td>CoGeo-GS (Ours)</td><td>28.9</td><td>0.882</td><td>0.086</td><td>11.4</td><td>30.7</td><td>0.903</td><td>0.072</td><td>9.8</td></tr></table>

Scene1

![](images/f9a25d556ab1af3a32372fc690c5b0ee2ba61440e699c73a4cbb3c4a1be66f09.jpg)

![](images/c797dbad47aedcc8619e2a074701a142e61d14a6da4d5cb89814e080a786f372.jpg)

![](images/12f943dd1dec81567c2fed430c9553537fd6a99fc2e42a1b68c3eb8228c1dbe6.jpg)

![](images/e25c5e05d781c311bcc1e1b8a3fa96f5205c50361c1a658a903cbadcedbd6e70.jpg)

![](images/ed88a191997d7d8fc27fbc3bad1c5dd444a6bfd30a086c620f41cb40ae9a2138.jpg)

![](images/544075eb7c5ff622932611097bdf7757395662a0b235f57fbe51a96898339e0d.jpg)

![](images/6a762a3d9c3d91d991368e4188163a2733a778253fa4f3be35dcf0f094d36805.jpg)

![](images/8a9f5632ebd7281d4de8764f916938b13ab8cf271cea6013427ade61b9f7fa6c.jpg)

![](images/ec6309be2a5b669a539119f062a995ebdbd324dabe6ffcee5c7135a0f0a57b66.jpg)

![](images/973b0ca98293f6c49184557ddf11cce2678f7f3fb7e8dd95b9a465a70be027d9.jpg)

![](images/eb1b5d72c0cc4ddd58bca620b98540d23ba6478ee43838f335a348b83394f793.jpg)  
Input Images

![](images/5f11ceaae293cb69645cd76e83b4047e0f2fc3e474cd055246c9605ea86e28e6.jpg)  
GaussianEdit

![](images/e738469440422a728100cfa3ed942692babd67f7d6854b9f52edf2e5e41732d7.jpg)  
Infusion

![](images/b3f391bd83537c28d0736fec230d326d99108eaf768b121cb3c5baa8fc950706.jpg)  
Gaussian Grouping

![](images/90920e43466ad5c6c15f87822a0d291b19668a420c6b114727d40977ab508237.jpg)  
Ours

Fig. 2. Qualitative comparison on multi-object removal. We compare CoGeo-GS with GaussianEditor, InFusion and Gaussian Grouping in a consistent view.  
TABLE II  
ABLATION STUDY ON THE MULTI-OBJECT REMOVAL BENCHMARK.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>FID↓</td></tr><tr><td>w/o concept-aware masking</td><td>26.85</td><td>0.851</td><td>0.112</td><td>16.90</td></tr><tr><td>w/o depth guidance</td><td>25.97</td><td>0.832</td><td>0.128</td><td>18.40</td></tr><tr><td>Full model (Ours)</td><td>28.90</td><td>0.882</td><td>0.086</td><td>11.40</td></tr></table>

3D consistency but requires slow per-scene optimization and often produce blurry textures in large masked regions. InFusion exhibits better texture but lacks geometric stability. Our method preserves clear, high-frequency details consistent with the surrounding texture. This indicates that our concept-aware distillation ensures precise object removal without degrading the visual quality of the remaining scene. As reported in Task 2 of Tab. I, CoGeo-GS outperforms baselines on the SPIn-NeRF dataset, achieving PSNR, SSIM, LPIPS and FID improvement. These gains demonstrate the superiority of CoGeo-GS in improving both geometric fidelity and perceptual quality.

IndoorScene

Input Images

![](images/fc9f4c27d1c7004267f86e0b6ca7d62e853b005c92fbcbf7f3f5015502a49192.jpg)

![](images/a508f74e9d9459003cba757da07bb99f21be7c24f602cb8651e25e773f94736b.jpg)  
SPIn-NeRF  
InFusion

![](images/846871e589e0a542953344c2a8ed71aaa9d6e6b4475c93fd0625d5409f264671.jpg)  
Ours  
Fig. 3. Qualitative comparison on single-object removal. We compare CoGeo-GS with SPIn-NeRF and InFusion in a consistent view.

## C. Ablation Study

We evaluate the effect of two core components: conceptaware 3D masking and depth-guided geometric completion. We replace the proposed 3D concept distillation with a naive 2D projection baseline, where a Gaussian is removed if its projected center lies inside the object mask in any view. As shown in the first row of Tab. II, this leads to noticeable performance degradation, particularly in PSNR. The degradation arises from view-dependent visibility and partial occlusions, which cannot be reliably handled by 2D projections. To assess the role of geometric anchoring, we remove the monocular depth prior

![](images/7b052b5fb8195a6ea6b71cc36c5842821c29e4fa15e5f61d8d89ee2abc62c5aa.jpg)  
w/. concept tagging

![](images/2f94a62e7b80eab12ab0567f7fbd965f8723b9ffaa8faf31288c694d534fb3f3.jpg)  
w/o concept tagging

![](images/933224beae6f0aff6d40635a2d668a7e94d7b37a84c86c2156e7fc34895acc2e.jpg)  
w/. depth guidance

![](images/b4e7f130aef9498d7649aafc84382579ac813838918318257fd04a38e2a68356.jpg)  
w/o depth guidance

Fig. 4. Ablation study. Without concept-aware masking, residual artifacts persist in the edited regions. In the absence of depth guidance, the predicted depth map leads to blurred shadows and degraded structure in occluded areas. and perform diffusion-based depth completion without scale guidance. As reported in the second row of Tab. II, this variant suffers from clear drops in PSNR and FID, accompanied by floating artifacts and geometric misalignment in Fig. 4. Without depth anchoring, the reconstructed Gaussians fail to maintain consistent surface geometry.

## V. CONCLUSION

We introduced CoGeo-GS, a concept-driven framework tailored for multi-object removal in 3D scenes. By unifying concept-aware 3D Gaussian selection, depth-guided geometric completion, and geometry-regularized appearance refinement within a single optimization pipeline, CoGeo-GS reliably reconstructs occluded background geometry while preserving cross-view consistency. Extensive experiments on multi-object removal demonstrate that our method achieves improved geometric fidelity and perceptual quality, consistently outperforming prior Gaussian-based editing approaches. We believe CoGeo-GS provides promising directions toward interactive 3D content manipulation.

## ACKNOWLEDGEMENT

This project was supported by the National Natural Science Foundation of China under Grant No. 62472415.

## REFERENCES

[1] Haotian Mao, Zhuoxiong Xu, Siyue Wei, Yule Quan, Nianchen Deng, and Xubo Yang, “Live-gs: Llm powers interactive vr by enhancing gaussian splatting,” arXiv preprint arXiv:2412.09176, 2024.

[2] Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, and Junchi Yan, “Drivemoe: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving,” arXiv preprint arXiv:2505.16278, 2025.

[3] Siting Zhu, Guangming Wang, Dezhi Kong, and Hesheng Wang, “3d gaussian splatting in robotics: A survey,” arXiv preprint arXiv:2410.12262, 2024.

[4] Xianliang Huang, Chen Xiao, Yuanxiang Ni, Guanming Liu, Mingkai Liu, Dikai Fan, Xiao Liu, and Hao Zhang, “Semantic-guided progressive object removal with gaussian splatting,” arXiv preprint arXiv:2607.04144, 2026.

[5] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George¨ Drettakis, “3d gaussian splatting for real-time radiance field rendering.,” ACM Trans. Graph., vol. 42, no. 4, pp. 139–1, 2023.

[6] Yiwen Chen, Zilong Chen, Chi Zhang, Feng Wang, Xiaofeng Yang, Yikai Wang, Zhongang Cai, Lei Yang, Huaping Liu, and Guosheng Lin, “Gaussianeditor: Swift and controllable 3d editing with gaussian splatting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 21476–21485.

[7] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke, “Gaussian grouping: Segment and edit anything in 3d scenes,” in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 162–179.

[8] Yuxin Wang, Qianyi Wu, Guofeng Zhang, and Dan Xu, “Learning 3d geometry and feature consistent gaussian splatting for object removal,” in Proc. Eur. Conf. Comput. Vis. Springer, 2024, pp. 1–17.

[9] Antonio Criminisi, Patrick Perez, and Kentaro Toyama, “Region filling´ and object removal by exemplar-based image inpainting,” IEEE Trans. Image Process., vol. 13, no. 9, pp. 1200–1212, 2004.

[10] Tijana Ruziˇ c and Aleksandra Pi ´ zurica, “Context-aware patch-basedˇ image inpainting using markov random field modeling,” IEEE Trans. Image Process., vol. 24, no. 1, pp. 444–456, 2014.

[11] Yu Zeng, Zhe Lin, Huchuan Lu, and Vishal M Patel, “Cr-fill: Generative image inpainting with auxiliary contextual reconstruction,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., 2021, pp. 14164–14173.

[12] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky, “Resolution-robust large mask inpainting with fourier convolutions,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., 2022, pp. 2149–2159.

[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising diffusion probabilistic models,” Adv. Neural Inf. Process. Syst., vol. 33, pp. 6840– 6851, 2020.

[14] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.

[15] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool, “Repaint: Inpainting using denoising diffusion probabilistic models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 11461–11471.

[16] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang, “Smartbrush: Text and shape guided object inpainting with diffusion model,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 22428–22437.

[17] Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G Derpanis, Jonathan Kelly, Marcus A Brubaker, Igor Gilitschenski, and Alex Levinshtein, “Spin-nerf: Multiview segmentation and perceptual inpainting with neural radiance fields,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 20669–20679.

[18] Xianliang Huang, Shuhang Chen, Zhizhou Zhong, Jiajie Gou, Jihong Guan, and Shuigeng Zhou, “Hi-nerf: Hybridizing 2d inpainting with neural radiance fields for 3d scene inpainting,” in Proceedings of the Asian Conference on Computer Vision, 2024, pp. 2855–2871.

[19] Xianliang Huang, Zhizhou Zhong, Shuhang Chen, Yi Xu, Jihong Guan, and Shuigeng Zhou, “Nerf-mir: Toward high-quality restoration of masked images with neural radiance fields,” IEEE Transactions on Neural Networks and Learning Systems, 2026.

[20] Zhiheng Liu, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jie Xiao, Kai Zhu, Nan Xue, Yu Liu, Yujun Shen, and Yang Cao, “Infusion: Inpainting 3d gaussians via learning depth completion from diffusion prior,” arXiv preprint arXiv:2404.11613, 2024.

[21] Kira Prabhu, Jane Wu, Lynn Tsai, Peter Hedman, Dan B Goldman, Ben Poole, and Michael Broxton, “Inpaint3d: 3d scene content generation using 2d inpainting diffusion,” arXiv preprint arXiv:2312.03869, 2023.

[22] Xianliang Huang, Jiajie Gou, Shuhang Chen, Zhizhou Zhong, Jihong Guan, and Shuigeng Zhou, “Iddr-ngp: Incorporating detectors for distractors removal with instant neural radiance field,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 1343–1351.

[23] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman, “Mip-nerf 360: Unbounded anti-aliased neural radiance fields,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 5470–5479.

[24] Johannes L Schonberger and Jan-Michael Frahm, “Structure-frommotion revisited,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 4104–4113.