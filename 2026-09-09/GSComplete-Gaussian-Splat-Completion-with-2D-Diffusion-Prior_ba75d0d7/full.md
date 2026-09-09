# GSComplete: Gaussian Splat Completion with 2D Diffusion Priors

E. Brugger<sup>1</sup> and P. Erler<sup>1</sup> and S. Ohrhallinger<sup>1</sup> and P. Guerrero<sup>2</sup>

<sup>1</sup>TU Wien, Austria <sup>2</sup>Adobe Research, United Kingdom

![](images/ceeb6bccdc4b6d37c6bafdbfa28e3f8873a8aa5ef56f30adf7ad8cf26d3c8ef4.jpg)  
Figure 1: We propose GSComplete to generatively complete a partial 3D object represented as a set of Gaussian splats. Our approach generates a plausible completed 3D object that exactly preserves the given partial input using only a 2D diffusion prior. Insets show errors in the preservation of input splats for each view.

## Abstract

Gaussian splats provide afast, high-fidelity representationfor 3D objects but are often constructedfrom incomplete input data in practice, leaving missing regions. Existing completion methods either do not preserve the original splats or require scarcely available 3D training data. We propose GSComplete, which combines 3D generation based on Score Distillation Sampling with a novel preservation loss that encourages the original splats to be preserved where they should be visible. This effectively completes the Gaussian splat object using only 2D diffusion priors while fully preserving existing splats and generating new splats only in missing regions, without occluding the input. To evaluate our approach, we introduce a new dataset of partial Gaussian splat objects and show that GSComplete achieves significantly more accurate preservation of the input than existing methods with comparable plausibility ofthe completed result. Our code and dataset will be made available upon acceptance.

CCS Concepts

• Computing methodologies → Shape modeling;

## 1. Introduction

Gaussian splats provide a fast and high-fidelity representation for 3D objects that can be constructed from image sets, 3D scans, or existing 3D objects. Gaussian splat scenes used in practice are often incomplete due to missing information at construction time, such as incomplete image sets, scan shadows, or partial 3D objects. For example, a robotic agent or scanner may only be able to view the front of an object, or a 3D artist may want to save time by only modeling the important parts of an object. To complete these objects, generative methods have been proposed that extrapolate missing information, such as novel view synthesis methods [WMH<sup>∗</sup>24, LWVH<sup>∗</sup>23, SCZ<sup>∗</sup>23] and native 3D generation methods that employ a 3D prior [DZS<sup>∗</sup>25, XLX<sup>∗</sup>25]. However, a 3D prior requires scarcely available 3D training data, and novel view synthesis methods represent existing splats with one or multiple images, making it difficult to accurately preserve them. Analogously to image completion, we want a Gaussian splat completion method that fully preserves existing splats and only generates new splats in missing regions as needed.

We propose GSComplete to complete a Gaussian splat object using only 2D diffusion priors while fully preserving existing splats. Given a set of Gaussian splats that represent a partial 3D object (obtained from sources such as partial scans, image reconstructions, or partial 3D meshes), our method learns to output a set of Gaussian splats that (i) exactly preserves the original splats, (ii) preserves the appearance of the original object from a given range of viewpoints, making sure it is not hidden by new splats; and (iii) represents a plausible completion of the object.

The central challenge in our problem is that the guidance from the 2D diffusion prior and preservation of the existing object part are at odds to some extent. Typically, slight variations of the existing object part will have a higher probability in the prior than its exact preservation, so the prior will often encourage new splats to cover parts of the existing object. Our solution is to combine a generative approach based on Score Distillation Sampling (SDS) [PJBM22, SWY<sup>∗</sup>24] with a new preservation loss that encourages the original object to be preserved.

To evaluate our approach, we introduce a new dataset of partial Gaussian splat (GS) objects and show that GSComplete can achieve significantly more accurate preservation of the original object part than existing methods with comparable plausibility of the completed result. Our main contributions are: (i) a new approach for GS completion that uses 2D diffusion priors and fully preserves existing splats; and (ii) a new dataset of partial GS splat objects.

## 2. Related work

The completion of missing data in point clouds, often due to artifacts from scans such as scan shadows, occlusions, or incomplete campaign planning, is a topic that has been researched for a long time, with input data as point clouds, or single or few images, more recently, using 3DGS - see here for a detailed survey on point cloud completion [ZZH<sup>∗</sup>24].

Point cloud completion with 3D priors. This approach trains a model specifically for 3D point cloud completion or 3D point cloud generation. It requires scarcely available 3D data, and as a result, these methods typically do not generalize beyond their limited training data. One of the first learning-based methods is Point Completion Network [YKH<sup>∗</sup>18], AdaPoinTr [YRW<sup>∗</sup>] uses transformers with a denoising task, 3D-PCGR [YPZ 24] combines LIDAR and color data, colorizing using a GAN, and recent SuperPC [DZS 25] proposes a unified diffusion framework for completing, upsampling, denoising, and colorizing point clouds.

Single- or few-image 3D reconstruction. An alternative approach to complete 3D Gaussian splats is to render existing splats from one or multiple viewpoints that do not show the missing parts, and reconstruct the full object from these images. However, as the existing splats are represented by one or a few images, this does not accurately preserve existing splats, as we demonstrate in Section 4. Tewari et al. [TYC<sup>∗</sup>23] use a differentiable forward renderer for the denoising of a conditional diffusion model, avoiding 3D supervision. Splatter Image [SRV24] turns each pixel into a splat. Shen et al. [SXW24] extend it to hierarchically map pixels to multiple splats. DiffGS [ZZL24] proposes a continuous Gaussian Splatting function for unstructured 3DGS. pixelSplat [CLTS24] samples Gaussian means from a dense probability distribution. LGM [TCC<sup>∗</sup>24] produces multi-view Gaussian features with an asymmetric U-Net. MVSplat [CXZ<sup>∗</sup>24] stores cross-view feature similarities in a cost volume. LRM [HZG<sup>∗</sup>24], GS-LRM [ZBT<sup>∗</sup>24], and TripoSR [TPL<sup>∗</sup>24] incorporate transformers. Wonder3D [LGL<sup>∗</sup>24] applies cross-domain diffusion. Direct3D [WLZ<sup>∗</sup>24] uses a native 3D generative model, and Wu et al. [WKZ<sup>∗</sup>25] extend it with spatial sparse attention. ReconFusion [WMH<sup>∗</sup>24] uses NeRFs. latentSplat [WRI 24] predicts semantic Gaussians in a 3D latent space and decodes them quickly in 2D. Flash3D [SIZ 25] creates a layer of Gaussians at the predicted depth and further layers behind that. Ouroboros3D [WHW 25] uses a recursive diffusionreconstruction-based process. Trellis [XLX<sup>∗</sup>25] generates native 3D built on a unified structured latent representation and rectified flow transformers.

3D inpainting with 2D priors. This approach directly inpaints missing splats, using a 2D inpainting prior that is projected back to the 3D scene. The difference to completion is that this usually removes objects from a background and only completes missing parts that are small compared to the size of the full scene or object, such as holes or small scan shadows: Weder et al. [WGHM<sup>∗</sup>23], Zhang et al. [ZHD<sup>∗</sup>23], RGBD2 [LTJ23], OR-NeRF [YFYL23], Gaussian Grouping [YDYK24], MVIP-NeRF [CLP24], MALD-NeRF [LKH<sup>∗</sup>24], GScream [WWZX24], GS-RoadPatching [CLD<sup>∗</sup>25], and SplatFill [DPTDB25]. Further, the recent 3DGIC [HCW25] outperforms the state-of-the-art by using depth-guided cross-view consistency among multi-view images. The most recent concurrent work has pushed the quality even further. GOR-IS [ZGY<sup>∗</sup>26] focuses on lighting consistency by decomposing the scene into intrinsic components and explicitly modeling light transport. GP-GS [LC26] uses a point cloud completion model, a coarse-to-fine inference strategy, image refinement, and a fine-tuning phase. GaussFiller [PLL<sup>∗</sup>26] chooses the best view for inpainting with global-local alignment and then selects from multiple attempts the completion result with the highest semantic coherence.

Score Distillation Sampling (SDS). Similar to our approach, these methods generate splats using a 2D diffusion prior. However, these are not directly applicable to completion as there is no mechanism to preserve existing splats out of the box.

![](images/519b4ea0e6dbdd61ec4fa4427a877cc9070a9368178e9ab4eba2d71ee879613b.jpg)  
Figure 2: Overview of GSComplete. The partial input P is preserved while iterating with its masked (M<sub>c</sub>) input preservation loss $\mathcal { L } _ { p }$ and SDS loss $\mathcal { L } _ { S D S }$ to complete the 3DGS object O.

Dreamfusion [PJBM22], DreamGaussian [TRZ<sup>∗</sup>23], and MV-Dream $\scriptstyle [ \mathrm { S W Y } ^ { * } 2 4 ]$ , which uses a multi-view diffusion model that lifts 2D diffusion priors for 3D generation and makes them consistent using inflated 3D self attention, are just a few examples of the many methods available. Similar to our approach, ComPC [HYZL24] and SDS-Complete [KRC23] use SDS to complete a 3D object, but handle only uncolored point clouds, rather than Gaussian splats.

## 3. Method

Given a partial 3D object represented by a set of Gaussian splats $P = \{ p 1 , p 2 , \ldots \}$ , our goal is to generate a new set of Gaussian splats $O = \{ o _ { 1 } , o _ { 2 } , . . . \}$ that (i) represents a plausible completion of the 3D object, and (ii) exactly preserves the given part of the 3D object. An overview of our approach is shown in Figure 2.

To generate a plausible object, we use a Score Distillation Sampling (SDS) approach based on MVDream [SWY<sup>∗</sup>24]. SDS allows us to generate plausible 3D objects using only a 2D diffusion prior, without having to rely on scarcely available 3D training data. We provide a short recap of this approach in Section 3.1. As the SDS approach takes a text prompt y of the desired completed object as additional input, we add y to our inputs.

To preserve the given part of the 3D object, we keep the existing splats P fixed during the SDS-based generation, and only optimize additional splats, effectively making sure that P is a subset of O. However, naively generating splats with this approach typically results in existing splats P being progressively covered and hidden by the new splats in $O ,$ as we show in our ablations. We believe this is due to a preference in the 2D prior for some types of objects that do not necessarily include completions of the given input. To address this problem, we propose a new input preservation loss that makes sure that the partial input object is not hidden by new splats from a given distribution of viewpoints V. The viewpoints V are given as input and could, for example, be based on a known capturing setup, or on user preferences. We describe this loss in Section 3.2.

Completing a 3D object with this approach requires suitable initialization and optimization strategies. In Section 3.3, we describe our approach that makes sure that new splats have sufficient coverage of regions that need to be completed without fully hiding existing splats.

In summary, given a partial 3D object as a set of splats, a text prompt, and a distribution of viewpoints $( P , y , V )$ , respectively, we output the completed object as a set of splats O that are a superset of P and preserve the appearance of P from the viewpoints V.

## 3.1. 3D Object Generation with SDS

Diffusion models [HJA20, SME21, RBL<sup>∗</sup>22] generate images $x _ { 0 }$ by iteratively denoising a full-noise image x into increasingly less noisy versions $x _ { T - 1 } , x _ { T - 2 } , . . . , x _ { 0 }$ . In each step, a denoiser g predicts a fully denoised version of the image $\hat { x } _ { 0 } = g ( x _ { t } , t , y )$ , where t is the time step and y is a text prompt. The denoised image $\hat { x } _ { 0 }$ is only a coarse approximation of $x _ { 0 } .$ , but gives a good direction to follow towards the next less-noisy image $x _ { t - 1 }$

Score Distillation Sampling (SDS) [PJBM22] uses the denoiser as a prior to generate 3D objects, by applying the denoiser to noised renders of the object:

$$
\mathcal { L } _ { \mathrm { S D S } } : = \mathbb { E } _ { t , c } \ \| x _ { c } - g \big ( \epsilon ( x _ { c } , t ) , t , y \big ) \| _ { 2 } ^ { 2 } ,\tag{1}
$$

where $x _ { c } : = r ( O , c )$ is a differentiable render of the Gaussian splats O from viewpoint c and $\boldsymbol { \epsilon } ( \boldsymbol { x } , t )$ noises the image x by an amount corresponding to timestep t. This loss is used to iteratively optimize the splats O.

We base our approach on MVDream $\begin{array} { r } { [ \mathrm { S W Y ^ { * } } 2 4 ] . } \end{array}$ , a variant of SDS that uses a denoiser fine-tuned to generate multi-view images of a 3D object:

$$
\mathcal { L } _ { \mathrm { M V S D S } } : = \mathbb { E } _ { t , c } ~ \| x _ { c } - g _ { \operatorname* { m v } } \left( \epsilon ( x _ { c } , t ) , t , y \right) \| _ { 2 } ^ { 2 } .\tag{2}
$$

Here $x _ { c } : = r _ { \mathrm { m v } } ( O , c )$ is a multi-view image rendered from viewpoint c and three additional viewpoints at the same elevation and equally spaced at 90 degree offsets along the azimuth. This multiview formulation of SDS improves the consistency of the generated object across different views.

## 3.2. Input Preservation Loss

To preserve the input splats P, we can include them in O and freeze them during the SDS optimization. This successfully preserves the splats, but we show in Section 4 that the optimization tends to hide P by covering it with other splats in O. Note that a correct completion does need to cover the input splats from some viewpoints, for example, viewpoints showing the incomplete back side of the partial 3D object, but it should still be visible from other viewpoints V. The choice of viewpoints in V – independent of the abovementioned SDS multi-views – then defines, via their appearance, which splats will be preserved as visible. This may, for example, depend on the capturing setup used to obtain the input splats, or on an artistic choice, e.g., when using completion to ideate alternatives for 3D object parts, analogous to existing workflows in image editing [Pro].

We take this distribution of viewpoints V as input and define a new loss that encourages the input splats P to be visible from these viewpoints:

$$
\mathcal { L } _ { \mathfrak { p } } : = \mathbb { E } _ { c \sim V } \Vert M _ { c } \cdot \left( r ( O , c ) - r ( P , c ) \right) \Vert _ { 2 } ^ { 2 } ,\tag{3}
$$

where · denotes the element-wise product and $M _ { c } : = r _ { a } ( P , c )$ is a 2D mask of the splats P from viewpoint $^ { c , }$ obtained as the alpha channel $r _ { a }$ of the render r(P,C).

## 3.3. Initialization and Optimization

Object completion is a task different than unconstrained generation, resulting in other dynamics of the SDS optimization. To adapt the SDS optimization to object completion, we carefully initialize the added Gaussian splats $O \setminus P$ and define a schedule for the input preservation loss and for the amount of noise added to the renders in the SDS optimization.

Initialization. Given the existing splats P (20k-100k in our experiments), we initialize $| O \setminus P | = 5 0 0 0$ new Gaussian splats by uniformly distributing them in a sphere with diameter equal to the largest side of $P { ^ { \circ } s }$ bounding box and centered at the bounding box center. We found that in the first optimization steps, we can achieve faster and more stable convergence by encouraging splats to gather in regions that the missing parts of the object are likely to occupy. Intuitively, these missing parts are more likely to be visible from views other than those in V. Thus, we encourage new splats to initially gather in regions that are hidden from viewpoints in V. We remove the mask M of the preservation loss in the first $n _ { \mathrm { i n i t } } = 5 0$ iterations (setting M<sub>c</sub> to all-ones), effectively promoting renders from views in V to match renders of the partial input object, thereby encouraging new splats to move behind the existing splats as seen from these viewpoints, or to reduce their opacity.

Optimization. In consecutive steps of the SDS optimization, we alternate between using the SDS loss L<sub>MVSDS</sub> and a weighted sum of the SDS loss and the input preservation loss $\mathcal { L } _ { \mathrm { M V S D S } } + \lambda \mathcal { L } _ { \mathrm { p } } ,$ which reduces runtime without impacting quality. We use $\lambda = 2$ in our experiments and optimize for a total of 6000 steps. Following the original Gaussian Splat implementation [KKLD23], in the first 1500 steps we subdivide the splats based on the gradient magnitude, giving us a total of 100k - 200k splats for $O \backslash P .$ The diffusion time t used in each optimization step (Eq. 2) defines the amount of noise added to each image. The amount of noise determines how much of the original rendering is preserved in the denoised image and how much is generated by the denoiser. Large amounts of noise only preserve coarse structures and encourage making larger changes to the Gaussian splat object, while small amounts of noise preserve everything but fine details and encourage making changes only to smaller details. In our experiments, we choose the following schedule for t that reduces the noise level over time to focus only on refining details in later steps of the optimization:

• Steps 1 to 3000: $t \sim \mathcal { U } ( 0 . 4 T , 0 . 6 T )$

• Steps 3001 to 4500: t ∼ U(20,0.35T)

• Steps 4501 to 6000: t ∼ U(20,0.25T)

U denotes a uniform distribution and T is set to 980. As discussed in Section 5 and shown in Table 2, our method is not sensitive to the choice of t schedule and simpler schedules usually work as well.

## 4. Results

We evaluate GSComplete by comparing it to the most relevant prior work on our dataset of partial objects, showing that our method produces completions of similar or better quality, while preserving the existing partial objects much more accurately, followed by an ablation of our core technical contributions and a discussion of the main limitations.

Metrics. Our goal is to produce completions of high plausibility while exactly preserving the partial input object. We measure plausibility of the completion using the CLIP similarity $S _ { \mathrm { C L I P } }$ between a render of the partial input splats P from one of the input viewpoints V and a render of the completed splats O from the four viewpoints at azimuth offsets 0, 90, 180, 270 from the input viewpoint. We measure input preservation by comparing both color and depth renders of the partial splats P from one of the input viewpoints V to corresponding renders of the completed splats O. We compare with three metrics: the mean-squared error $E _ { \mathrm { M S E } } ^ { \mathrm { C o l o r } }$ and $E _ { \mathrm { M S E } } ^ { \mathrm { D e p t h } }$ of color and depth, respectively, and the LPIPS $\mathrm { [ Z I E ^ { * } }$ 18] error E<sub>LPIPS</sub> of the color renders. Each of these is weighted by the mask $M _ { c }$ to only focus on existing splats P. We average all metrics over all objects of our dataset.

Dataset. We evaluate on our new test set of partial Gaussian splat objects that we call SPLATCOMPLETE. It consists of 39 objects from different sources:

Multiview: 24 real-world objects were captured using using between 25 and 50 images taken from viewpoints distributed around the object. As a reconstruction method, we use the original Gaussian Splatting approach [KKLD23]. To obtain partial GS splat objects, we manually define a region to be preserved with a bounding box and discard all remaining splats. The remaining partial objects each comprise between 20k and 100k splats.

Singleview: 5 real-world objects were captured using a single image of the object. We use Depth Anything $3 \ [ \mathrm { L C L } ^ { * } 2 5 ]$ to obtain a colored point cloud for the visible parts of the object and convert to Gaussian splats using a fixed splat size that is chosen manually for each object based on point cloud density.

LiDAR: 5 real-world objects were extracted from the Redwood Indoor Dataset [PZK17]. We manually selected and cut the objects from room scans, removing clutter and any remaining background. Splat size was determined as in the Singleview subset.

Table 1: Comparison to Baselines. We evaluate preservation of the input splats by comparing depth and color renders of the input splats from the input viewpoint to the corresponding renders of the completions. Plausibility of the completion is evaluated by measuring how similar the semantics of the input splats are to the semantics of the completion from multiple different viewpoints, as measured by the CLIP similarity.
<table><tr><td rowspan="2"></td><td colspan="3">Input Preservation</td><td rowspan="2">Plausibility SCLIP ↑</td></tr><tr><td> $E _ { \mathrm { M S E } } ^ { \mathrm { D e p t h } }$  →</td><td> $E _ { \mathrm { M S E } } ^ { \mathrm { C o l o r } } \downarrow$ </td><td>ELPIPS ↓</td></tr><tr><td>MVDream</td><td>0.0167</td><td>0.2327</td><td>0.1561</td><td>66.985</td></tr><tr><td>Trellis MV</td><td>0.0125</td><td>0.1548</td><td>0.1026</td><td>75.485</td></tr><tr><td>Trellis SV</td><td>0.0116</td><td>0.1737</td><td>0.1050</td><td>75.478</td></tr><tr><td>InstantMesh</td><td>0.0115</td><td>0.1205</td><td>0.0859</td><td>75.918</td></tr><tr><td>TripoSG</td><td>0.0121</td><td>0.1641</td><td>0.0898</td><td>75.370</td></tr><tr><td>GSComplete (ours)</td><td>0.0084</td><td>0.0248</td><td>0.0370</td><td>77.944</td></tr></table>

Mesh: 5 synthetic objects were constructed from partially completed meshes obtained from Objaverse++ $[ \mathrm { L L L } ^ { * } 2 5 ]$ . We converted the mesh files to Gaussian splats with mesh2splat [Sco25].

Comparison. For GSComplete, the user selects an input view c and the distribution V is then defined uniformly in range around c with $+ / { - 5 0 } ^ { \circ }$ azimuth. The 6000 SDS iterations take roughly 35 minutes on a single RTX 3090 GPU.

A first baseline approach to GS completion is single-view reconstruction, by rendering the object from a viewpoint that is as complete as possible (the same viewpoint c that we use as input for our method) and reconstruct the 3D object from the rendered image. Trellis [XLX<sup>∗</sup>25], InstantMesh [XCG<sup>∗</sup>24], and TripoSG [LZL<sup>∗</sup>25] are recent methods that cover this approach. We call Trellis used in this approach Trellis SV to distinguish it from a multi-view version we describe next.

Single-view reconstruction misses any information about the partial input object that can’t be captured by a single view. A different baseline approach thus renders multiple views of the partial object, inpaints any missing parts, and then reconstruct the 3D objects from the inpainted renders. We are not aware of any prior work that attempts this approach for 3D object completion, so we use our best attempt as a baseline: in addition to the viewpoint V, we render from two additional viewpoints with an azimuth offset of 90 and 270 degrees (we avoid 180 degrees since from that viewpoint, the existing part of the object is typically fully hidden and thus the 2D inpainting method has no reliable reference to complete from that viewpoint), inpaint each view using SDXL [PEL<sup>∗</sup>24] with a mask based on the alpha channel of the rendered splats P, and finally use Trellis to reconstruct a 3D object from the inpainted multi-views. Since Trellis outputs 3D objects in a coordinate frame that is not aligned to the input image(s), we manually translate, rotate, and uniformly scale the models to best match the input splats. We call this multi-view baseline Trellis MV.

A third baseline apporach is to use our SDS but without the input preservation loss or our changes to initialization and optimization. As we base our method on MVDream [SWY<sup>∗</sup>24], we run MV-

Table 2: Ablation of core contributions. We ablate several technical contributions, including the input preservation (IP) loss, and our optimization strategy, including warmup of the Gaussian splats in the firstfew iterations.
<table><tr><td rowspan="2"></td><td colspan="3">Input Preservation</td><td rowspan="2">Plausibility  $S _ { \mathrm { C L I P } } \uparrow$ </td></tr><tr><td> $E _ { \mathrm { M S E } } ^ { \mathrm { D e p t h } }$  →</td><td> $E _ { \mathrm { M S E } } ^ { \mathrm { C o l o r } } \downarrow$ </td><td> $E _ { \mathrm { L P I P S } } \downarrow$ </td></tr><tr><td>w/o IP loss, t schedule, warmup</td><td>0.0167</td><td>0.2327</td><td>0.1561</td><td>66.985</td></tr><tr><td>w/o IP loss</td><td>0.0124</td><td>0.1711</td><td>0.1272</td><td>70.489</td></tr><tr><td>w/o t schedule</td><td>0.0074</td><td>0.0295</td><td>0.0421</td><td>78.233</td></tr><tr><td>GSComplete full</td><td>0.0084</td><td>0.0248</td><td>0.0370</td><td>77.944</td></tr></table>

Dream to only optimize the new splats $O \backslash P$ while keeping existing splats P fixed.

A quantitative comparison is shown in Table 1 and qualitative comparisons are given in Figures 6, 7, and 8. MVDream, lacking our input preservation loss, tends to cover existing splats, resulting in significantly different depths and colors when viewed from V, as reflected in the higher input preservation errors and lower clip similarity. Both Trellis SV and MV perform better, however, since they need to represent the partial input splats with image(s), the input’s depth and color are preserved much less accurately, which also results in a slight deviation from the original semantics, as the lower CLIP similarity indicates. The 2D inpainting performed for multiple views in Trellis MV is not guaranteed to be view-consistent, introducing errors in the reconstruction. Completions from Trellis SV, InstantMesh, and TripoSG look reasonably accurate from the input views for several objects, as shown in Figures 6, 7, and 8 (with InstantMesh having slightly lower quality than the other two), but the second view tends to show larger errors in input preservation. GSComplete preserves the input splats accurately as seen from the input view. Differences to the input splats in other views mainly come from valid completions covering the input splats. We show in Figure 4 that GSComplete is not sensitive to different seeds for the random number generator.

Ablation Studies We ablate the core technical contributions of our approach on our dataset in Table 2 and Figure 3. First, when removing the input preservation loss (w/o IP loss), the SDS optimization is free to fully cover the partial object P, effectively generating an object that is based on the given text prompt and ignoring the partial shape to a large extent. Examples are shown in Figure 3, columns 2 and 3. This is reflected in a higher input preservation error, as P is no longer visible, and in lower plausibility of the completion, as the semantics of the completion are less similar to the partial input. Our approach is robust to the choice of the t schedule, so replacing it with a simpler schedule where each optimization chooses t uniformly in [20, 980] (w/o t schedule, adapted from DreamGaussian $[ \mathrm { T R } Z ^ { * } 2 3 ] )$ does not significantly impact the performance. Removing the initialization that warms up added splats with the unmasked input preservation loss (in addition to removing the t schedule and the IP loss) shows mainly a strong increase in the depth error $E _ { \mathrm { M S E } } ^ { \mathrm { D e p t h } }$ , as unpruned splats clutter the depth map.

![](images/dcbd1da490b62fd695f68d2dd34e5acc37add3d497df478bb9c3f39492f91483.jpg)  
Figure 3: Ablation. We ablate several technical contributions, including the input preservation (IP) loss, and our optimization strategy, including warmup of the Gaussian splats in the first few iterations. For each object, we show two views of the partial input splats and the same two views for each ablation. Insets show errors in the preservation of input splats for each view as the difference between renders of the completed 3D objects and non-empty regions of the corresponding input view. We can see that our method preserves the input splats significantly more accurately than the baselines, with high plausibility ofthe completed shapes.

![](images/9052ec0e47b302e777c0d6b61ab6b2b103ffed09507db15e7a36c9ba42fbf875.jpg)  
Figure 4: Variance of Completions. GSComplete produces stable completions with different RNG seeds.

Limitations The quality of our result naturally depends on the richness of information from the input, even if our method manages to get good results from sparse input. If the input is too sparse, GSComplete can have difficulties shaping meaningful information (see Figure 5). The manually defined viewpoints/center-of-mass translations can easily be automated with heuristics, and the resolution of the diffusion output is currently limited to 256<sup>2</sup> by MV-

![](images/26c2bb7eba1d8a079503041680d0525d6c75940b8ea9a9a63aec59342188496b.jpg)  
Figure 5: Limitations of GSComplete. Left: texture and color seams. Right: Input with too little information.

Dream’s model architecture. Similar to other SDS methods, expansive scenes or scenes with complex backgrounds are difficult to handle, as they would require longer optimization and more careful camera placement. Like other SDS methods, we observe a slight amount of over-saturation due to the use of classifier-free guidance in the prior, sometimes creating visible color or texture seams, as shown in Figure 5. Additionally, our implementation is not optimized, but its runtime is competitive with comparable methods such as SDS, MVDream, and 3DGIC.

## 5. Conclusion

We revisited 3D object completion in the setting of Gaussian splats. We have shown that by introducing an input preservation loss to Score Distillation Sampling, we can complete partial objects represented as Gaussian splats in high quality while preserving the partial input more accurately than existing methods. Additionally, we introduced a test set of 18 partial Gaussian splat objects captured from real-world objects that we use to evaluate our approach.

Input

MVDream

Trellis MV

Trellis SV

GSComplete

![](images/c3be9c8e9de7e648b89182913322c1a3f3e23e4d06da2a3db3b514a4794e394b.jpg)

Figure 6: Qualitative results on partial GS objects obtained from multi-view images. We compare completions of GSComplete to several baselines. For each object, we show two views of the partial input splats and the same two views for each completion. Insets show errors in the preservation of input splats for each view as the difference between renders of the completed 3D objects and non-empty regions of the corresponding input view. We can see that our method preserves the input splats significantly more accurately than the baselines, with high plausibility ofthe completed shapes.

In future work, we plan to optimize the runtime of our method by using a faster diffusion algorithm and lowering resolution in the early stages, as well as using higher resolutions to further improve the output. Recomputing our preservation loss less frequently, but with a higher weight, can accelerate our method without losing quality. Iterating longer and densifying again later can address the color/texture seams, and adjusting hue and saturation values of the Gaussian splats could directly improve the texture quality even further. Finally, detecting multiple separate (partial) objects in a scene using heuristics and auto-generating prompts for these will allow feeding the objects individually to the diffusion module and thus effectively permitting completion of complex scenes.

## References

[CLD<sup>∗</sup>25] CHEN G., LIU J., DU S., WU C., LI D., HUANG S.- S., ZHANG G., YANG S.: Gs-roadpatching: Inpainting gaussians via 3d searching and placing for driving scenes. arXiv preprint arXiv:2509.19937 (2025). 2

[CLP24] CHEN H., LOY C. C., PAN X.: Mvip-nerf: Multi-view 3d inpainting on nerf scenes via diffusion prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024), pp. 5344–5353. 2

[CLTS24] CHARATAN D., LI S. L., TAGLIASACCHI A., SITZMANN V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2024), pp. 19457–19467. 2

[CXZ<sup>∗</sup>24] CHEN Y., XU H., ZHENG C., ZHUANG B., POLLEFEYS M., GEIGER A., CHAM T.-J., CAI J.: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European conference on computer vision (2024), Springer, pp. 370–386. 2

[DPTDB25] DAHAGHIN M., PADALKAR M. G., TOSO M., DEL BUE A.: Splatfill: 3d scene inpainting via depth-guided gaussian splatting. arXiv preprint arXiv:2509.07809 (2025). 2

[DZS<sup>∗</sup>25] DU Y., ZHAO Z., SU S., GOLLURI S., ZHENG H., YAO R., WANG C.: Superpc: a single diffusion model for point cloud completion, upsampling, denoising, and colorization. In CVPR (2025), pp. 16953– 16964. 2

[HCW25] HUANG S.-Y., CHOU Z.-T., WANG Y.-C. F.: 3d gaussian inpainting with depth-guided cross-view consistency. In Proceedings of the Computer Vision and Pattern Recognition Conference (2025), pp. 26704–26713. 2

[HJA20] HO J., JAIN A., ABBEEL P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851. 3

[HYZL24] HUANG T., YAN Z., ZHAO Y., LEE G. H.: Compc: Completing a 3d point cloud with 2d diffusion priors. arXiv preprint arXiv:2404.06814 (2024). 3

[HZG<sup>∗</sup>24] HONG Y., ZHANG K., GU J., BI S., ZHOU Y., LIU D., LIU F., SUNKAVALLI K., BUI T., TAN H.: Lrm: Large reconstruction model for single image to 3d. In International Conference on Learning Representations (2024), vol. 2024, pp. 50678–50702. 2

[KKLD23] KERBL B., KOPANAS G., LEIMKÜHLER T., DRETTAKIS G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42, 4 (2023), 139–1. 4

[KRC23] KASTEN Y., RAHAMIM O., CHECHIK G.: Point cloud completion with pretrained text-to-image diffusion models. Advances in Neural Information Processing Systems 36 (2023), 12171–12191. 3

[LC26] LEE Y., CHO D.: Gp-gs: Consistent 3d object removal via geometry-aware 3d inpainting and projected image refinement in 3d gaussian splatting. In Proceedings of the AAAI Conference on Artificial Intelligence (2026), vol. 40, pp. 5927–5935. 2

[LCL<sup>∗</sup>25] LIN H., CHEN S., LIEW J. H., CHEN D. Y., LI Z., SHI G., FENG J., KANG B.: Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647 (2025). 4

[LGL<sup>∗</sup>24] LONG X., GUO Y.-C., LIN C., LIU Y., DOU Z., LIU L., MA Y., ZHANG S.-H., HABERMANN M., THEOBALT C., ET AL.: Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2024), pp. 9970–9980. 2

[LKH<sup>∗</sup>24] LIN C. H., KIM C., HUANG J.-B., LI Q., MA C.-Y., KOPF J., YANG M.-H., TSENG H.-Y.: Taming latent diffusion model for neural radiance field inpainting. In European Conference on Computer Vision (ECCV) (2024). 2

[LLL<sup>∗</sup>25] LIN C., LIU H., LIN Q., BRIGHT Z., TANG S., HE Y., LIU M., ZHU L., LE C.: Objaverse++: Curated 3d object dataset with quality annotations, 2025. URL: https://arxiv.org/abs/2504. 07334, arXiv:2504.07334. 5

[LTJ23] LEI J., TANG J., JIA K.: Rgbd2: Generative scene synthesis via incremental view inpainting using rgbd diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition (2023), pp. 8422–8434. 2

[LWVH<sup>∗</sup>23] LIU R., WU R., VAN HOORICK B., TOKMAKOV P., ZA-KHAROV S., VONDRICK C.: Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision (2023), pp. 9298–9309. 2

[LZL<sup>∗</sup>25] LI Y., ZOU Z.-X., LIU Z., WANG D., LIANG Y., YU Z., LIU X., GUO Y.-C., LIANG D., OUYANG W., ET AL.: Triposg: Highfidelity 3d shape synthesis using large-scale rectified flow models. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025). 5

[PEL<sup>∗</sup>24] PODELL D., ENGLISH Z., LACEY K., BLATTMANN A., DOCKHORN T., MÜLLER J., PENNA J., ROMBACH R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. ICLR (2024). 5

[PJBM22] POOLE B., JAIN A., BARRON J. T., MILDENHALL B.: Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022). 2, 3

[PLL<sup>∗</sup>26] PING Y., LIN C., LIU Y., DOU Z., PAN J., WANG W.: Gaussfiller: Unleashing vlm-expert guidance for 3d scene completion with 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2026), pp. 7132–7142. 2

[Pro] PROJECT I.: InvokeAI: Professional creative tools for stable diffusion. Open-source software under Apache 2.0 license. URL: https: //github.com/invoke-ai/InvokeAI. 4

[PZK17] PARK J., ZHOU Q.-Y., KOLTUN V.: Colored point cloud registration revisited. In ICCV (2017). 4

[RBL<sup>∗</sup>22] ROMBACH R., BLATTMANN A., LORENZ D., ESSER P., OMMER B.: High-resolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition (2022), pp. 10684–10695. 3

[Sco25] SCOLARI S.: Mesh2splat: Fast mesh to 3d gaussian splat conversion. https://github.com/electronicarts/mesh2splat, 2025. Extended and updated version of the author’s Master’s thesis at KTH. 5

[SCZ<sup>∗</sup>23] SHI R., CHEN H., ZHANG Z., LIU M., XU C., WEI X., CHEN L., ZENG C., SU H.: Zero123++: a single image to consistent multi-view diffusion base model, 2023. arXiv:2310.15110. 2

[SIZ<sup>∗</sup>25] SZYMANOWICZ S., INSAFUTDINOV E., ZHENG C., CAMP-BELL D., HENRIQUES J. F., RUPPRECHT C., VEDALDI A.: Flash3d: Feed-forward generalisable 3d scene reconstruction from a single image. In 2025 International Conference on 3D Vision (3DV) (2025), IEEE, pp. 670–681. 2

[SME21] SONG J., MENG C., ERMON S.: Denoising diffusion implicit models. ICLR (2021). 3

[SRV24] SZYMANOWICZ S., RUPPRECHT C., VEDALDI A.: Splatter image: Ultra-fast single-view 3d reconstruction. Conference on Computer Vision and Pattern Recognition (CVPR) (2024). 2

[SWY<sup>∗</sup>24] SHI Y., WANG P., YE J., LONG M., LI K., YANG X.: Mvdream: Multi-view diffusion for 3d generation. ICLR (2024). 2, 3, 5

[SXW24] SHEN J., XUE N., WU T.: A pixel is worth more than one 3d gaussians in single-view 3d reconstruction. arXiv preprint arXiv:2405.20310 (2024). 2

[TCC<sup>∗</sup>24] TANG J., CHEN Z., CHEN X., WANG T., ZENG G., LIU Z.: Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision (2024), Springer, pp. 1–18. 2

[TPL<sup>∗</sup>24] TOCHILKIN D., PANKRATZ D., LIU Z., HUANG Z., , LETTS A., LI Y., LIANG D., LAFORTE C., JAMPANI V., CAO Y.-P.: Triposr: Fast 3d object reconstruction from a single image. arXiv preprint arXiv:2403.02151 (2024). 2

[TRZ<sup>∗</sup>23] TANG J., REN J., ZHOU H., LIU Z., ZENG G.: Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653 (2023). 3, 5

[TYC<sup>∗</sup>23] TEWARI A., YIN T., CAZENAVETTE G., REZCHIKOV S., TENENBAUM J., DURAND F., FREEMAN B., SITZMANN V.: Diffusion with forward models: Solving stochastic inverse problems without direct supervision. Advances in Neural Information Processing Systems 36 (2023), 12349–12362. 2

[WGHM<sup>∗</sup>23] WEDER S., GARCIA-HERNANDO G., MONSZPART Á., POLLEFEYS M., BROSTOW G., FIRMAN M., VICENTE S.: Removing objects from neural radiance fields. In CVPR (2023). 2

[WHW<sup>∗</sup>25] WEN H., HUANG Z., WANG Y., CHEN X., SHENG L.: Ouroboros3d: Image-to-3d generation via 3d-aware recursive diffusion. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (2025), pp. 21631–21641. 2

[WKZ<sup>∗</sup>25] WU H., KARUMURI M. G., ZOU C., BANG S., LI Y., SAMARAS D., HADAP S.: Direct and explicit 3d generation from a single image. In 2025 International Conference on 3D Vision (3DV) (2025), IEEE, pp. 490–501. 2

[WLZ<sup>∗</sup>24] WU S., LIN Y., ZHANG F., ZENG Y., XU J., TORR P., CAO X., YAO Y.: Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. Advances in Neural Information Processing Systems 37 (2024), 121859–121881. 2

[WMH<sup>∗</sup>24] WU R., MILDENHALL B., HENZLER P., PARK K., GAO R., WATSON D., SRINIVASAN P. P., VERBIN D., BARRON J. T., POOLE B., ET AL.: Reconfusion: 3d reconstruction with diffusion priors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (2024), pp. 21551–21561. 2

[WRI<sup>∗</sup>24] WEWER C., RAJ K., ILG E., SCHIELE B., LENSSEN J. E.: latentsplat: Autoencoding variational gaussians for fast generalizable 3d reconstruction. In European conference on computer vision (2024), Springer, pp. 456–473. 2

[WWZX24] WANG Y., WU Q., ZHANG G., XU D.: Learning 3d geometry and feature consistent gaussian splatting for object removal. In European Conference on Computer Vision (2024), Springer, pp. 1–17. 2

[XCG<sup>∗</sup>24] XU J., CHENG W., GAO Y., WANG X., GAO S., SHAN Y.: Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191 (2024). 5

[XLX<sup>∗</sup>25] XIANG J., LV Z., XU S., DENG Y., WANG R., ZHANG B., CHEN D., TONG X., YANG J.: Structured 3d latents for scalable and versatile 3d generation. In Proceedings ofthe Computer Vision and Pattern Recognition Conference (2025), pp. 21469–21480. 2, 5

[YDYK24] YE M., DANELLJAN M., YU F., KE L.: Gaussian grouping: Segment and edit anything in 3d scenes. In European conference on computer vision (2024), Springer, pp. 162–179. 2

[YFYL23] YIN Y., FU Z., YANG F., LIN G.: Or-nerf: Object removing from 3d scenes guided by multiview segmentation with neural radiance fields, 2023. arXiv:2305.10503. 2

[YKH<sup>∗</sup>18] YUAN W., KHOT T., HELD D., MERTZ C., HEBERT M.: Pcn: Point completion network. In 3D Vision (3DV), 2018 International Conference on (2018). 2

[YPZ<sup>∗</sup>24] YUAN C., PAN J., ZHANG Z., QI M., XU Y.: 3d-pcgr: Colored point cloud generation and reconstruction with surface and scale constraints. Remote Sensing 16, 6 (2024), 1004. 2

[YRW<sup>∗</sup>] YU X., RAO Y., WANG Z., LU J., ZHOU J.: Adapointr: diverse point cloud completion with adaptive geometry-aware transformers (2023). arXiv preprint arXiv:2301.04545. 2

[ZBT<sup>∗</sup>24] ZHANG K., BI S., TAN H., XIANGLI Y., ZHAO N., SUNKAVALLI K., XU Z.: Gs-lrm: Large reconstruction model for 3d gaussian splatting. European Conference on Computer Vision (2024). 2

[ZGY<sup>∗</sup>26] ZHAO Y., GAO Y., YANG J., XIE J., WANG B.: Gor-is: 3d gaussian object removal in the intrinsic space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2026), pp. 40896–40906. 2

[ZHD<sup>∗</sup>23] ZHANG Z., HAN X., DONG B., LI T., YIN B., YANG X.: Point cloud scene completion with joint color and semantic estimation from single rgb-d image. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 9 (2023), 11079–11095. 2

[ZIE<sup>∗</sup>18] ZHANG R., ISOLA P., EFROS A. A., SHECHTMAN E., WANG O.: The unreasonable effectiveness of deep features as a perceptual metric. In CVPR (2018). 4

[ZZH<sup>∗</sup>24] ZHUANG Z., ZHI Z., HAN T., CHEN Y., CHEN J., WANG C., CHENG M., ZHANG X., QIN N., MA L.: A survey of point cloud completion. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing 17 (2024), 5691–5711. 2

[ZZL24] ZHOU J., ZHANG W., LIU Y.-S.: Diffgs: Functional gaussian splatting diffusion. Advances in Neural Information Processing Systems 37 (2024), 37535–37560. 2

![](images/bd833dea193db8bb8883a10c080562785ddb0d788de5aa42bee3c9f2b58e042b.jpg)  
Figure 7: Additional qualitative results on partial GS objects obtainedfrom multi-view images. We compare completions of GSComplete to several baselines. For each object, we show two views of the partial input splats and the same two views for each completion. Insets show errors in the preservation ofinput splatsfor each view as the difference between renders ofthe completed 3D objects and non-empty regions of the corresponding input view.

![](images/1d7d2fb75bfb46879393983380c063dde2d38e5f292b7322b10745bdd0679d57.jpg)

Figure 8: Qualitative results on partial GS objects obtained from single views, LiDAR scans, or partial meshes. We compare completions of GSComplete to several baselines. For each object, we show two views ofthe partial input splats and the same two viewsfor each completion. Insets show errors in the preservation of input splats for each view as the difference between renders of the completed 3D objects and nonempty regions ofthe corresponding input view.