# DiT-Garment: Garment Dynamics with Diffusion Transformers

Antoine Dumoulin Inria Centre at the University Grenoble Alpes antoine.dumoulin@inria.fr

Joao Regateiro InterDigital Inc. joao.regateiro@interdigital.com

Laurence Boissieux Inria Centre at the University Grenoble Alpes laurence.boissieux@inria.fr

Pierre Hellier Inria, University of Rennes, CNRS, IRISA-UMR 6074 pierre.hellier@inria.fr

Stefanie Wuhrer Inria Centre at the University Grenoble Alpes stefanie.wuhrer@inria.fr

## Abstract

We present DiT-Garment to model dynamic 3D clothing over human body models in arbitrary motion. Unlike existing methods, DiT-Garment can animate garments with unseen designs and physical materials, while allowing for direct inference of deformations for any target pose. To achieve this, we leverage a 2D diffusion transformer architecture to learn 3D deformations in a 2D UV-space. As the result is non-deterministic, our generative model learns the distribution of possible outcomes. The template garment is represented as a 3D triangle mesh spatially aligned with a 3D human body model in a standardizedpose. To work with different garment designs without the need of a common template or complex graph convolution operations, the diffusion transformer is conditioned on a 3D position map of the template, represented in UV-space, which allows to implicitly learn a deformation ofthe 3D space around the body in standard pose. Further conditioning on body motion and physical parameters allows to physically ground the model. We quantitatively and qualitatively evaluate DiT-Garment on both synthetic and real data. While only trained on synthetic simulations ofautomatically generated cloth designs, our method generalizes to captured and artist-made garment designs. Code and data are available for research purposes at https://dumoulina.github.io/dit-garment/.

## 1. Introduction

Garment modeling is a highly studied topic in computer vision and computer graphics. Dressing 3D avatars is a subject of interest in tele-presence applications such as augmented or virtual reality. Furthermore, in applications such as garment prototyping and virtual try-on, an important use case is to dynamically adapt garment geometry to various body morphologies and poses. In this context, modeling the dynamics of digital garments is an important problem.

![](images/588c8e3b8e56e8a32c38376ce3f6bdc145beba12b118e362f8bea0da207b9852.jpg)  
Figure 1. DiT-Garment animates arbitrary garment designs over any human body in any target pose. Given a body motion and physical parameters, DiT-Garment directly infers dynamic garment deformations for a design seen during training (left) and gen eralizes to three unseen designs (right), without retraining.

We aim to simulate garments over human body models in arbitrary motion, with high control. In particular, it is desirable for users to control the garment design, the physical cloth material, and the wearer’s body shape and motion. Our objective is to animate a given input garment by deforming it to fit the last frame of the input body motion.

The problem of simulating garments over body models is well studied. Most works represent the garment using representations that are graph-based, point cloud-based, or use a 2-dimensional UV-based mapping of the 3D garment. Graph-based methods [2, 13] are highly realistic and generally model dynamics well. However, most of these methods are auto-regressive, i.e. they start from a standard pose and fit a full motion sequence up to the desired pose, rendering the methods complex for posing tasks. Existing methods can generalize to garment designs unseen during training, but lack the mesh connectivity or surface structure needed to model physical constraints such as stretch and bending strain, limiting their physical fidelity [40, 54]. Methods that use a UV-based representation [21, 49] offer the advantage of directly benefiting from standard 2D neural architectures, and have recently been used to allow for fine-grained user control in case of a fixed garment design [12]. However, there is currently no method that allows for input garment designs unseen during training and physical material information, while directly inferring the deformations for the target pose and modeling garment dynamics.

To address this problem, we present DiT-Garment, a physically grounded conditional generative model of garment deformation learned using diffusion transformers. We represent a garment design as a 3D template mesh, and the physical cloth material using three physical quantities describing bending, stretching and area density. Body motion is represented as a discrete sequence of parametric body poses along with a body shape. To work on a common space for all garment designs, we take advantage of a diffusion transformer model [52] in a two-dimensional UVspace to conditionally deform a given garment template. DiT-Garment provides fine-grained control over input conditions related to the person wearing the garment and the garment’s design and material. Since DiT-Garment only depends on a small sequence of body poses leading up to the desired pose, the result is not deterministic. To represent the diverse outcomes of garment deformations under unseen previous states of the garment, we leverage a generative model which learns a distribution of plausible garment deformations conditioned on the input conditions.

The core of our method is a garment representation that generalizes across designs unseen during training. We represent each garment as a 3D triangle mesh aligned to a human body model in a standard pose. Because all templates share the same body alignment, the model can learn corresponding deformations across different garment designs without explicit registration. The 3D mesh is then projected into 2D UV-space as a position map encoding the 3D rest coordinates of each garment vertex. We use this position map as positional encoding of the diffusion transformer, so that attention over UV coordinates conditions on rest-shape geometry rather than UV layout. This design preserves seam connectivity: corresponding seam vertices map to shared 3D coordinates, and the attention mechanism learns smooth spatial correlations across UV seams.

To train dynamic clothing models, we introduce a dataset composed of dynamically simulated clothes paired with human body motions. Existing synthetic cloth datasets such as Cloth3D [3] provide large-scale simulation data, but rely on a limited set of template designs. While resizing, cutting and reshaping the templates allows to generate a large number of garments, the underlying topology remains fixed. Recent advances in garment modeling [18] leverage sewing patterns to generate diverse and complex garment designs. To build our dataset of dynamic garments with diversified designs, we pick templates from GarmentCodeData [19]. We then simulate the templates over various body shapes and motions with different physical materials. This dataset is used to train our model and evaluate its generalization to unseen complex garment designs. We further evaluate DiT-Garment on real captured data from 4D-Dress [51] and 4DHumanOutfit [1]. Quantitative and qualitative results show that DiT-Garment generalizes to unseen garment designs and physical materials, while producing realistic deformations at interactive frame rates.

In summary, our contributions are:

• A positional encoding that maps geometry into UV-space, allowing 2D neural architectures to work on arbitrary mesh templates with varying UV parametrizations.

• DiT-Garment, a conditional generative model based on a UV-based representation: We propose a diffusion transformer that enables direct posing of dynamic garments. Our model produces realistic deformations while providing fine-grained user control on physical properties, body shape and motion.

• An intersection-aware guidance strategy during the reverse diffusion process that effectively penalizes and mitigates cloth-body intersections for complex, unseen poses.

• A sewing-pattern-based dynamic dataset: We provide a comprehensive synthetic dataset featuring complex garment designs (from tight tanks to loose dresses) simulated over diverse body motions and physical materials. This dataset will be publicly released to foster further research.

## 2. Related work

We organize existing works according to how the garment is represented. Common representations are graph-based, point cloud based, and 2-dimensional mappings of the 3- dimensional garments. A parallel line of work using implicit functions to model clothing [9, 10, 24] will not be discussed as these works focus on shape modeling rather than cloth dynamics.

## 2.1. Graph-based representation

Cloth is traditionally simulated using graph operations over 3D surface meshes [22, 47]. Common physics-based approaches model cloth motion with particles [6], masssprings [41] or finite elements [47]. Baraff and Witkin [2] proposed a robust implicit solver for cloth simulation offering a good trade-off between accuracy and speed. Faster alternatives have been proposed such as Projective Dynamics [5, 30], Position Based Dynamics [35, 37] and Vertex Block Descent [7]. However, time integration methods remain computationally expensive and require expertise to tune the physical parameters to achieve stable results [50].

Recently, learning based methods have been proposed to render cloth simulation more accessible. TailorNet [38] learns to predict static deformations of garments given a body pose. To model dynamic deformations, templatespecific methods [4, 8, 43, 44] learn to predict cloth dynamics over body motion. Some existing works generalize to multiple garment designs in a single model. GarSim [48] and HOOD [13] learn to predict cloth dynamics for multiple garment designs leveraging a graph neural network. ContourCraft [14] propose a new contour loss to improve the quality of the generated garments with HOOD making it a strong baseline. Recently, transformer-based models [23, 46] show promising results in modeling cloth dynamics for multiple garment designs.

Most works relying on a graph-based representation are autoregressive starting from a standard pose, and cannot directly infer deformations for an arbitrary target pose. In contrast, DiT-Garment allows for direct posing.

## 2.2. Point cloud representation

Point-based representations provide a flexible alternative to meshes for modeling clothed humans, as they do not impose explicit topological constraints on garment deformations. Early works [32, 54] explored point-based representations for modeling pose-dependent clothed human surfaces. Subsequent methods [33, 34] addressed artifacts appearing with highly deformed regions of loose garments, where sparse point sampling can lead to incomplete or noisy surface reconstructions. FiTe [27] further introduced an implicit representation of the garment surface in a canonical pose, which is then explicitly deformed to the target pose. Some point-based methods additionally use a hybrid UV representation for the body geometry input [32, 33].

Dynamic Point Fields (DPF) [40] instead learn deformation fields over an explicit source garment, using geometric constraints such as as-isometric-as-possible regularization. The method can animate a 3D garment using SMPL vertices as a driving signal while maintaining correspondence with the source point representation. This ability to deform an existing garment template while preserving its geometric structure makes DPF a particularly relevant baseline for point-based garment deformation.

More recently, ClothDiffuse [56] introduced a diffusionbased framework for generating temporally coherent cloth deformations conditioned on body motion, enabling the modeling of dynamic garment motion. However, pointbased representations primarily capture the garment surface geometry and its deformation, without explicit information describing the physical structure of the garment. This limits their ability to model physically grounded cloth dynamics, where quantities such as local connectivity, material properties, and interactions between neighboring surface elements are important. In contrast, DiT-Garment explicitly models physics-inspired dynamics.

Table 1. Positioning of our method compared to existing learnedbased approaches. Gar. indicates whether the method can generalize to garments unseen during training. Motion indicates whether the method models physical dynamics given body motion. Phys. allows for control of the physical material. Posing indicates whether the method can directly infer deformations for any target pose. Bold indicates compared methods in experiments.
<table><tr><td>Method</td><td>Gar.</td><td>Motion</td><td>Phys.</td><td>Posing</td></tr><tr><td colspan="5">Graph representation</td></tr><tr><td>TailorNet [38]</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>Santesteban et al. [43]</td><td>X</td><td></td><td>X</td><td>X</td></tr><tr><td>PBNS [4]</td><td>X</td><td></td><td></td><td>X</td></tr><tr><td>SNUG [44]</td><td>X</td><td></td><td>X</td><td>X</td></tr><tr><td>GAPS [8]</td><td>X</td><td></td><td>X</td><td>X</td></tr><tr><td>Cape [31]</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>GarSim [48]</td><td></td><td></td><td></td><td>X</td></tr><tr><td>HOOD [13]</td><td></td><td></td><td></td><td>X</td></tr><tr><td>Shi et al. [46]</td><td></td><td></td><td>X</td><td></td></tr><tr><td>Li et al. [23]</td><td></td><td></td><td>X</td><td>X</td></tr><tr><td>ContourCraft [14]</td><td></td><td></td><td></td><td>X</td></tr><tr><td colspan="5">Point cloud representation</td></tr><tr><td>SCALE [32]</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>Zakharkin et al. [54]</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>PoP [33]</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>FITE [27]</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>SkiRT [34]</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>ClothDiffuse [56]</td><td></td><td></td><td>X</td><td></td></tr><tr><td>DPF [40]</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td colspan="5">UV-based representation</td></tr><tr><td>DeepWrinkle [21]</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>Zhang et al. [55]</td><td>X</td><td></td><td>X</td><td></td></tr><tr><td>D-Garment [12]</td><td>X</td><td></td><td></td><td></td></tr><tr><td>Shen et al. [45]</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>DiffusedWrinkles [49]√</td><td></td><td>X</td><td>X</td><td></td></tr><tr><td>Pyramid-Drape [20]</td><td>V</td><td>X</td><td>X</td><td></td></tr><tr><td>DiT-Garment</td><td></td><td></td><td></td><td></td></tr></table>

## 2.3. UV-based representation

To leverage efficient neural network architectures in the 2D domain [11, 42], recent approaches encode 3D garment deformations in a 2D UV intermediate representations. Deep-Wrinkles [21] model static pose-dependent wrinkles using normal maps. Shen et al. [45] propose a generative adversarial network to generate garments over body shape and pose. DiffusedWrinkles [49] introduce a diffusion model to generate pose-dependent wrinkles over multiple garment designs. Pyramid-Drape [20] propose a multi-scale architecture to generate pose-dependent wrinkles over multiple garment designs. In addition to pose-dependent effects, [12, 55] aim to model motion-dependent deformations.

UV-based methods are particularly promising for garment modeling as they allow to benefit from standard 2D neural architectures. However, there is currently no method that allows simultaneously for generalization to garments unseen during training, allows for control of physical materials, models dynamics, and allows for posing. DiT-Garment is a UV-based method that unlike previous work learns to deform arbitrary mesh templates with varying topology and UV parametrization.

## 2.4. Positioning

Table 1 summarizes the positioning of DiT-Garment w.r.t. existing works. DiT-Garment is the first UV-based method that simultaneously allows to model unseen garments during training, to control the physical cloth material, to simulate dynamics over body motion, and to infer cloth deformations from any pose. We experimentally compare DiT-Garment to strong baselines from each representation category: graph-based, point-based and UV-based methods.

## 3. Method

We introduce DiT-Garment, a unified model that learns to generate dynamic cloth deformations to dress 3D avatars in motion for multiple garment designs, represented by templates. Figure 2 illustrates the architecture, which is based on a Diffusion Transformer (DiT) [39] operating in UVspace. By aligning all garment templates to a common rest pose and projecting them into a two-dimensional parameterized space, DiT-Garment bypasses the need for shared topology, explicit rigging, or complex graph operations. Instead, DiT-Garment learns to predict three-dimensional deformations by leveraging the spatial layout encoded in the template’s position map.

Given a mesh template representing a garment, the wearer’s body shape, a sequence of body poses, and the cloth’s physical material parameters (Sec. 3.1), our approach deforms the mesh template to fit the body in the last frame of the sequence of body poses. This is achieved by representing the garment geometry in a 2-dimensional UV space, and by learning how to displace each vertex of the template. The body and physical material are represented using low-dimensional vectors as in D-Garment [12].

The model operates in a non-autoregressive manner: at each frame, it predicts the target deformation directly from the current input, without requiring current cloth velocity, or deformation history. Because no predicted state feeds back into subsequent frames, this frame-independent generation avoids the error accumulation typical of autoregressive simulators. Temporal coherence across a motion sequence arises from the smooth variation of the driving input rather than from any temporal recurrence in the model.

It is important that the mesh templates of all garments are aligned in $\mathbb { R } ^ { 3 }$ , as we use this alignment to encode correspondence information between different garments. In practice, we align all templates on a human body model in fixed body shape and pose. The unposed template is then unwrapped to a 2D position map which encodes the rest shape geometry of the garment design in a 2D image. The position map acts as a structural guide, informing the DiT about the template’s layout and ensuring that generated displacement maps align with the input’s UV coordinates. Because the position map encodes the 3D spatial arrangemen of vertices, the model can learn correspondence information across different garment templates without requiring explicit UV-map correspondence. Consequently, a single trained model generalizes to multiple garment designs with varying UV parametrizations.

We train the model on a dataset of simulated cloth featuring diverse designs, motions, and physical parameters (Sec. 4). At inference, DiT-Garment enables dynamic simulation (Sec. 5.3) of unseen garment designs.

## 3.1. Garment modeling and representation

We aim to geometrically deform a 3D model of a garment template M, represented as triangle mesh, to M at a discrete time step t. The dynamic deformations of the garment $\mathcal { M }$ to $\mathcal { M } _ { t }$ are modeled as the offset vector $v _ { t } ~ \in ~ \mathbb { R } ^ { | V | \times 3 }$ where $V \in \mathbb { R } ^ { n \times 3 }$ is the set of n vertices of $\mathcal { M } .$ . To learn a normalized space of garment templates, we align all templates to a standard rest pose in $\bar { \mathbb { R } ^ { 3 } }$ where the garment is unposed and aligned with a human body model in a fixed pose. This alignment enables the model to learn correspondences across different garment designs.

Inspired by geometry images [15] and their application in cloth deformation models [12, 20], we map 3D information onto 2D images. Unifying garment mesh templates with varying connectivity and vertex resolution to a standard 2D grid allows us to use state-of-the-art 2D architectures. Unlike previous work, our method directly learns from arbitrary mesh templates with varying UV parametrization.

We denote the mapping function projecting 3D mesh surface to image pixels by ϕ: $\mathbb { R } ^ { 3 } \mapsto \mathbb { R } ^ { \bar { 2 } }$ , and its inverse applying 2D mapping to mesh vertices by $\phi ^ { - 1 } \colon \mathbb { R } ^ { 2 } \mapsto \mathbb { R } ^ { 3 }$ . We use ϕ to encode geometric features including position map $\phi ( \mathcal { M } )$ and displacement map ${ \phi } ( { \mathcal { M } } _ { t } )$ . Thus, DiT-Garment learns to generate the displacement map ${ \phi } ( { \mathcal { M } } _ { t } )$ from the position map $\phi ( \mathcal { M } )$ and the other conditional inputs. Note that each garment M has its own UV parametrization ϕ.

DiT-Garment represents a dynamic garment on top of a parametric human body model that decouples body shape and pose parameters. By representing all garment templates in the same standard pose, DiT-Garment learns the relationship between the garment and the body without requiring explicit rigging or correspondence information between the different garments. In our implementation, we use SMPL [29], and represent body shape $\beta$ and pose sequence $\pmb { \theta } _ { t - l : t }$ as concatenation of the l preceding poses and the current one at time t. θ includes both the global and local body joint rotations and positions. $\beta$ is represented as a low-dimensional vector of shape coefficients.

![](images/2865cc905cfe11367e63bd4a9f08968fd68f2e444e1780a67b30ad9fcf7e04bf.jpg)  
Figure 2. DiT-Garment generates garment deformations for a given template representing a garment design conditioned on body shape $\beta ,$ motion $\pmb { \theta } _ { t - l : t }$ and cloth material $\gamma$ (see Sec. 3.1). It builds upon a 2D diffusion transformer model (DiT) to learn how to deform a template in UV-space. 3D geometric features are parameterized by the UV parametrization of the template, and we concatenate the position map to the diffusion noise. At inference, our model deforms the garment by iteratively denoising the Gaussian noise to the displacement map.

To physically ground DiT-Garment, the network uses cloth material information. The representation of cloth material is inspired by physics-based cloth simulation [2] and includes stretch coefficient s (in $N / m )$ , mass density coefficient d (in $k g / m ^ { 2 } )$ and bending coefficient b (in $N \cdot m )$ as $\gamma : = [ \mathbf { s } , \mathbf { d } , \mathbf { b } ]$ . Parameter s controls resistance to stretching or compression, d controls the influence of inertia, and b controls resistance to bending or curvature changes.

DiT-Garment, shown in Figure 2, can be formulated as:

$$
\mathcal { M } _ { t } \sim \mathcal { G } ( \mathcal { M } , \gamma , \beta , \pmb { \theta } _ { t - l : t } ) .\tag{1}
$$

## 3.2. Diffusion transformer model

The 3D mesh generator $\mathcal { G }$ is built on a 2D diffusion transformer (DiT) [39], leveraging the recent success of diffusion models for 3D cloth deformation in UV-space [12, 16, 17, 25, 26, 49]. Unlike prior works, the transformer architecture can learn from diverse garment templates and UV parametrizations. This is thanks to the position map $\phi ( \mathcal { M } )$ that encodes the 3D coordinates of the garment vertices in UV-space, which allows to implicitly learn a deformation of the 3D space around the body in standard pose. The position map is concatenated to the diffusion noise and fed to the DiT, giving the necessary information to generate displacement maps aligned with the input $U V$ parametrization. The model is further conditioned on body shape, body motion, and cloth material to produce physically grounded deformations. The output of the model is a displacement map ${ \phi } ( { \mathcal { M } } _ { t } )$ that encodes the deformation of the garment from its rest pose to the target pose.

During training, the conditional denoising transformer ϵ is trained with rectified flow objective [28] using samples $\left\{ \mathcal { M } _ { t } , \mathcal { M } , \gamma , \beta , \pmb { \theta } _ { t - l : t } \right\}$ from our training data corpus as:

$$
\mathbb { E } _ { \mathbf { z } , \epsilon , s } \left[ \lVert ( \epsilon - \mathbf { z } _ { 0 } ) - \epsilon _ { \boldsymbol { \theta } } ( \mathbf { z } _ { s } , s , \mathcal { M } , \gamma , \boldsymbol { \beta } , \boldsymbol { \theta } _ { t - l : t } ) \rVert _ { 2 } ^ { 2 } \right] ,\tag{2}
$$

where $\mathbf { z } _ { 0 } ~ = ~ \phi ( \mathcal { M } _ { t } ) , ~ \epsilon ~ \sim ~ \mathcal { N } ( 0 , I )$ , s the diffusion time step, and the noisy latent obtained with forward diffusion ${ \bf z } _ { s } = \alpha _ { s } { \bf z } + \sigma _ { s } \epsilon$ . The scaling factor and standard deviation of the forward diffusion are

$$
\bar { \alpha } _ { s } = \prod _ { u = 1 } ^ { s } ( 1 - \beta _ { u } ) , \quad \alpha _ { s } = \sqrt { \bar { \alpha } _ { s } } , \quad \sigma _ { s } = \sqrt { 1 - \bar { \alpha } _ { s } } ,\tag{3}
$$

where variance $\beta _ { u }$ is determined by the noise schedule.

At inference, we sample the garment deformation $\mathcal { M } _ { t }$ from the learned distribution by iteratively denoising a Gaussian noise $\mathbf { z } _ { S } \sim \mathcal { N } ( 0 , I )$ to the predicted displacement map $\mathbf { z } _ { 0 }$ using the reverse diffusion process.

## 3.3. Intersection guidance

While the trained model can generate realistic cloth deformations, it may produce intersections with the body. To mitigate this issue, we introduce a guidance strategy that penalizes cloth-body intersections during diffusion.

Inspired by $[ 5 3 ] , \mathcal { L } _ { c }$ computes the distance of points inside the body to the closest point on the body surface:

$$
\mathcal { L } _ { c } = \sum _ { p \sim \mathcal { U } ( \mathcal { M } _ { t } ) } \delta _ { \mathrm { i n } } ( p , \mathcal { B } ) \operatorname* { m i n } _ { b \sim \mathcal { U } ( \mathcal { B } ) } | | p - b | | _ { 2 } ,\tag{4}
$$

where $p$ and b are points uniformly sampled over the mesh surface, and $\delta _ { \mathrm { i n } }$ is an indicator function for colliding points.

This loss guides the diffusion process by computing the gradient of $\mathcal { L } _ { c }$ w.r.t. the generated displacement map ${ \phi } ( { \mathcal { M } } _ { t } )$ and applying it as correction to the predicted flow of the denoiser output. Guidance is applied at each denoising step after a warmup period to remove initial noise.

![](images/162adaf1d7a10661b4497a41a8ab2dc997d8eb65bf58cb23b95b3f30e83a18ae.jpg)  
(a)  
(b)  
Front view  
(c)  
(d)  
(e)  
(a)  
(b)  
(c)  
Back view  
(d)  
(e)  
Figure 3. Visual comparison of the simulation ground truth (a), DiT-Garment (b), Pyramid-Drape [20] (c), DPF [40] (d) and ContourCraf [14] (e). DiT-Garment generates realistic deformations for unseen garment designs, physical materials and motions.

## 4. Dataset

Existing datasets for cloth simulation, such as Cloth3D [3], VTO [43] or D-Garment [12], provide large-scale data but rely on a limited set of template designs. While resizing, cutting and reshaping these templates allows generating a large number of garments, the underlying topology remains fixed, limiting the diversity of garment designs. To address this limitation, we simulate a new dataset with diverse garment designs, physical materials, and body motions.

We build a garment template set of 22 upper clothes from GarmentCodeData [19] with styles ranging from tight tanks to complex loose dresses. These templates are generated from sewing patterns based on GarmentCode [18], which allows for complex garment designs with varying structure, cut and sizing. The templates are aligned to a human body model in a standard pose and unwrapped to a 2D UV-space with Blender. ProjectiveFriction [30] simulates the garments over various body shapes and motions with different physical materials. We select 169 body motions from AMASS [36] that we discretize as pose sequences at 30 frames per second (FPS). We augment the motion set with 3 random body shapes $\beta \sim \bar { \mathcal { U } } ( - 1 , 1 ) ^ { 8 }$

Table 2. Comparison on synthetic test set to Pyramid-Drape [20], DPF [40] and ContourCraft [14]. Best scores among methods without ground truth information are in bold, and - means that the method does not produce the output.

<table><tr><td></td><td colspan="2">Shape Sim.</td><td colspan="5">Phys. Validity</td></tr><tr><td></td><td>E</td><td> $E _ { C D } ^ { \downarrow }$   $E _ { n } ^ { \downarrow }$ </td><td> $E _ { c } ^ { \downarrow }$ </td><td> $\underline { E } _ { b } ^ { \downarrow }$ </td><td> $E _ { s } ^ { \downarrow }$ </td><td> $\underline { { E _ { d } ^ { \downarrow } } } ]$ </td><td>FPS</td></tr><tr><td colspan="8">Method with access to ground truth correspondence and normals</td></tr><tr><td>Pyramid-Drape*</td><td>=</td><td>0.01 0.31</td><td>0.63</td><td>=</td><td>=</td><td>2.24</td><td>12.6</td></tr><tr><td colspan="8">Methods without ground truth information</td></tr><tr><td>DiT-Garment</td><td>5.20</td><td>0.17 0.38</td><td>0.62</td><td>0.51</td><td>1.32</td><td>2.66</td><td>0.4</td></tr><tr><td>DPF</td><td>6.95</td><td>0.18</td><td>0.42 0.64</td><td>0.49</td><td>0.22</td><td>3.94</td><td>0.05</td></tr><tr><td>ContourCraft</td><td>15.35</td><td>1.21</td><td>0.51 0.49</td><td>0.53</td><td>14.44</td><td>11.33</td><td>1.7</td></tr></table>

per sequence. We then randomly sample 3 garments with materials in log(b) ∼ U(−8, −4), s ∼ U(40, 200) and $\mathbf { d } \sim \mathcal { U } ( 0 . 0 1 , 0 . 7 )$ . In total, the dataset contains 1494 simulated sequences for a total duration of 2 hours.

For testing, 3 garment templates and 3 body motions are kept out from the training set. The test set is particularly challenging, as it contains only sequences with unseen garment templates, unseen motions, and unseen body shapes. There are 27 sequences in the test set.

4DHumanOutfit

![](images/97fcfe813577d1e24d4f9bbcb3cd4886a1cbbd8b81d30c10cdb6a7ab14686ecd.jpg)

![](images/94f0fa7c18af004f07cffb367f082e0083f962fdc6cf1284af53ebf42d1b6df7.jpg)  
Figure 4. We use DiT-Garment to reproduce captured sequences from 4D-Dress [51] and 4DHumanOutfit [1]. For each sequence, we show the captured avatar (left) and the predicted garment on top of the fitted body (right). We use reconstructed garment templates for 4D-Dress and an artist designed template for 4DHumanOutfit.

## 5. Experiments

## 5.1. Implementation details

To learn the mapping from the input conditions to the garment deformation, we leverage a DiT architecture [52] based on efficient Linear Attention. This model works without positional encoding which has been required by previous DiT models to infer a spatially coherent and structured image. The conditions $\beta , \pmb \theta _ { t - l : t } , \gamma$ are encoded with a 2- layer MLP and injected through cross-attention. At inference, the guidance is applied after half of the 100 denoising steps with a guidance weight of 5.

Trained on 2 NVIDIA L40S for 100 epochs during 9 days, we use a batch size of 32 and a learning rate of $1 0 ^ { - 4 }$ with Adam optimizer with weight decay of $1 0 ^ { - 2 }$ . The input UV maps are $6 4 \times 6 4$ pixels. We sampled the training diffusion steps s from a uniform distribution U(0, 1000) on a linear noise schedule with $\beta _ { 0 } = 1 0 ^ { - 4 }$ and $\beta _ { 1 0 0 0 } = 0 . 0 2$

## 5.2. Evaluation measures

We follow previous work [12] and evaluate the results using both shape similarity and physical validity. $E _ { v }$ is the average vertex-to-vertex distance between the predicted and ground truth meshes. $E _ { C D }$ is the Chamfer distance comparing shape similarity. $E _ { n }$ is the Chamfer normal distance comparing the wrinkling similarity. $E _ { c }$ is the percentage of cloth inside the body. $E _ { b } , \ E _ { s }$ and $E _ { d }$ are the average bending, stretching and density errors relative to the ground truth, comparing the physical validity of the predicted garment. $E _ { C D } , ~ E _ { v }$ and $E _ { d }$ measures are computed in centimeters. We also indicate the inference speed in frames per second (FPS) for each method without pre-processing steps.

![](images/21311cf977a47f48b810770943b132746a4ed28510b290a811fb69f54666d80a.jpg)  
Figure 5. Effect of our guidance strategy to reduce cloth-body intersections. Left: without guidance, the generated garment intersects with the body. Right: with guidance, the generated garment does not intersect with the body.

## 5.3. Comparison on simulated data

To evaluate the generalization of DiT-Garment to unseen garment designs, we compare to methods that can animate any garment design. Pyramid-Drape [20], DPF [40] and ContourCraft [14] are strong baselines for UV-based, point cloud and graph-based representations, respectively. Pyramid-Drape and DPF do not allow for control of the physical material, while we inform DiT-Garment and ContourCraft of the simulation physical material. We use pretrained models for Pyramid-Drape and ContourCraft when available, and optimize DPF on each body frame following the original implementation.

Figure 3 shows frames from test sequences of unseen garment designs along with predicted deformations from DiT-Garment, Pyramid-Drape, DPF and Contour-Craft. While having access to ground truth information, Pyramid-Drape fails to generate wrinkle details and changes the garment topology. This leads to visual artifacts as in the bottom row (c). DPF generalizes well to complex garments, but lacks physical realism with cloth-body intersections and unrealistic wrinkles. ContourCraft fails to represent pleats and folds in the garment, producing oversmoothed deformations. In contrast, DiT-Garment generates realistic deformations with detailed wrinkles and with few cloth-body intersections. We provide additional qualitative results in the supplementary video.

Table 2 reports the average shape similarity and physical validity metrics of all methods applied over the test set. Unlike all other methods, Pyramid-Drape has access to ground truth information as input. It changes mesh topology; hence, $E _ { v } , \ E _ { b }$ and $E _ { s }$ cannot be measured on the output. Pyramid-Drape achieves the best $E _ { C D }$ and $E _ { n }$ errors. Of the remaining methods, overall, DiT-Garment presents competitive performance in most metrics, namely $E _ { v } , E _ { C D } , E _ { n }$ , and $E _ { d }$ . DPF achieves better $E _ { b }$ and $E _ { s }$ errors, thanks to its isometric loss. ContourCraft achieves less cloth-body intersections $E _ { c } ,$ minimized by its contour loss, but shows high $E _ { s }$ for 10 fast sequences due to the slow propagation of graph neural networks, which leads to a high average error.

Table 3. Comparison of different transformer input patch size without guidance on seen and unseen templates. Smaller patch size improves the performance on unseen templates at the cost of inference speed. Guidance and patch size of 1 are used in the full model.
<table><tr><td></td><td colspan="7">Seen templates</td><td colspan="7">Unseen templates</td><td rowspan="2"></td></tr><tr><td>Patch</td><td colspan="3">Shape Similarity</td><td colspan="3">Physical Validity</td><td></td><td colspan="3">Shape Similarity</td><td colspan="3">Physical Validity</td><td></td></tr><tr><td>size</td><td>E2</td><td> $E _ { C D } ^ { \downarrow }$ </td><td>En</td><td>E</td><td> $E _ { b } ^ { \downarrow }$ </td><td>Es</td><td> $E _ { d } ^ { \downarrow }$ </td><td>E</td><td> $E _ { C D } ^ { \downarrow }$ </td><td>En</td><td>E</td><td>Eb</td><td>ES</td><td> $E _ { d } ^ { \downarrow }$ </td><td>fps</td></tr><tr><td>8</td><td>3.07</td><td>0.10</td><td>0.28</td><td>0.65</td><td>0.48</td><td>0.15</td><td>1.44</td><td>5.99</td><td>0.25</td><td>0.44</td><td>0.64</td><td>0.51</td><td>1.17</td><td>2.89</td><td>11.9</td></tr><tr><td>2</td><td>2.87</td><td>0.09</td><td>0.28</td><td>0.65</td><td>0.48</td><td>0.12</td><td>1.32</td><td>5.67</td><td>0.18</td><td>0.39</td><td>0.63</td><td>0.50</td><td>0.64</td><td>3.38</td><td>6.1</td></tr><tr><td>1</td><td>3.09</td><td>0.09</td><td>0.28</td><td>0.64</td><td>0.48</td><td>0.12</td><td>1.31</td><td>5.38</td><td>0.20</td><td>0.38</td><td>0.62</td><td>0.50</td><td>0.62</td><td>2.99</td><td>2.6</td></tr><tr><td>full</td><td>3.02</td><td>0.09</td><td>0.28</td><td>0.64</td><td>0.48</td><td>0.13</td><td>1.33</td><td>5.20</td><td>0.17</td><td>0.38</td><td>0.62</td><td>0.51</td><td>1.32</td><td>2.66</td><td>0.4</td></tr></table>

DiT-Garment outperforms existing approaches without ground truth correspondence input in shape similarity while maintaining competitive physical validity metrics.

## 5.4. Generalization on real data

Previous experiments have shown that DiT-Garment generalizes to unseen garment designs synthesized from a sewing pattern model [19]. To further evaluate the sim-to-real applicability of our model to real-world garments, we compare the generated deformations to captured garments from 4D-Dress [51] and 4DHumanOutfit [1]. We run our model on the fitted body motion from the captured sequences, using a reconstructed garment template for 4D-Dress and an artist designed template for 4DHumanOutfit. The results are shown in Figure 4. We use the same physical material parameters for all garments, which we arbitrarily set to s = 50, d = 0.01 and ${ \bf b } = 1 0 ^ { - 8 }$ . The animated garments generated by DiT-Garment are visually plausible, showing that it can handle templates coming from real data while being only trained on synthetic data. We provide additional qualitative results in the supplementary video.

## 5.5. Ablations

We evaluate the impact of our guidance strategy and the transformer patch size on the generalization capability of our model to unseen garment templates in Table 3. In this experiment, we compare the results with different patch sizes without guidance and with a patch size of 1 with guidance, which is used in the full model. The patch size is one of the main hyperparameters of our transformer architecture: larger patch size implies smaller number of tokens and thus faster inference speed. However, larger patch size also implies that the attention has less information about the local structure of the garment template, which can affect the generalization capability of the model to unseen templates. The quantitative results shows that smaller patch size improves the model performance on unseen templates. Notably, the patch size does not significantly affect the performance on seen templates showing that the model can learn to generate realistic deformations for seen templates even with a larger patch size. In practice, we use a patch size of 1 for all our experiments. Our guidance approach is effectively removing remaining cloth-body intersections as shown in Figure 5 while maintaining the shape similarity and physical validity of the generated deformations in Table 3. Note that guidance requires additional diffusion steps to converge, which reduces the inference speed of the model. We only use 20 steps for testing our model without guidance compared to the 100 steps in the full model.

## 5.6. Limitations

DiT-Garment is a data-driven model that learns to generate garment deformations from a dataset of simulated cloth. It is limited by the diversity and quality of the training data. The model may not generalize well to garment designs, body shapes, or motions that are significantly different from those seen during training.

In addition, our model does not explicitly enforce physical constraints, such as collision avoidance or energy conservation, which may lead to unrealistic deformations in some cases. While our guidance strategy mitigates clothbody intersections, it does not guarantee collision-free results. Future work could explore strategies to incorporate more physical constraints into the model.

Finally, our model operates in a non-autoregressive manner, which has the advantage of avoiding error accumulation over time while still producing temporally coherent results, but may limit its ability to capture long-term temporal dependencies in garment dynamics.

## 6. Conclusion

We presented DiT-Garment, a diffusion-based method for dynamically posing 3D garments conditioned on body shape, motion, and physical material properties. Our approach leverages a 2D diffusion transformer architecture operating in UV-space, enabling generalization across diverse garment templates without requiring shared topology or explicit rigging. The position map serves as a structural guide, encoding 3D spatial relationships that allow the model to learn correspondences across garment designs.

To train our model, we introduced a new dataset of simulated cloth with various complex garment designs, physical materials, and body motions. Experiments demonstrate that DiT-Garment produces realistic garment deformations at interactive frame rates, generalizing to unseen garments.

## 7. Acknowledgments

This work was partially funded by the Nemo.AI laboratory by InterDigital and Inria. Experiments presented in this paper were carried out using the Abaca infrastructure, supported by Inria (see https://abaca.inria.fr). We thank Adnane Boukhayma for helpful discussions. We thank Hunor Laczko for help with the comparison to Pyramid-Drape.´

## References

[1] Matthieu Armando, Laurence Boissieux, Edmond Boyer, Jean-Sebastien Franco, Martin Humenberger, Christophe´ Legras, Vincent Leroy, Mathieu Marsot, Julien Pansiot, Sergi Pujades, et al. 4dhumanoutfit: a multi-subject 4d dataset of human motion sequences in varying outfits exhibiting large displacements. Computer Vision and Image Understanding, 237:103836, 2023. 2, 7, 8

[2] David Baraff and Andrew Witkin. Large steps in cloth simulation. In ACM Transactions on Graphics (TOG), Proc. SIG-GRAPH, pages 767–778, 1998. 1, 2, 5

[3] Hugo Bertiche, Meysam Madadi, and Sergio Escalera. Cloth3d: clothed 3d humans. In Proc. of European Conference on Computer Vision (ECCV), pages 344–359, 2020. 2, 6

[4] Hugo Bertiche, Meysam Madadi, and Sergio Escalera. Pbns: Physically based neural simulation for unsupervised garment pose space deformation. ACM Transactions on Graphics (TOG), 40(6), 2021. 3

[5] Sofien Bouaziz, Sebastian Martin, Tiantian Liu, Ladislav Kavan, and Mark Pauly. Projective dynamics: fusing constraint projections for fast simulation. ACM Transactions on Graphics (TOG), 33(4), 2014. 3

[6] David E Breen, Donald H House, and Michael J Wozny. Predicting the drape of woven cloth using interacting particles. In ACM Proc. of Computer Graphics and Interactive Techniques (PACMCGIT), pages 365–372, 1994. 2

[7] Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. Vertex block descent. ACM Transactions on Graphics (TOG), 43 (4):1–16, 2024. 3

[8] Ruochen Chen, Liming Chen, and Shaifali Parashar. Gaps: Geometry-aware, physics-based, self-supervised neural garment draping. In Proc. of International Conference on 3D Vision (3DV), pages 116–125, 2024. 3

[9] Enric Corona, Albert Pumarola, G. Alenya, Gerard Pons-\` Moll, and Francesc Moreno-Noguer. Smplicit: Topologyaware generative model for clothed people. Proc. ofConfer ence on Computer Vision and Pattern Recognition (CVPR), pages 11870–11880, 2021. 2

[10] Luca De Luigi, Ren Li, Benoˆıt Guillard, Mathieu Salzmann, and Pascal Fua. Drapenet: Garment generation and selfsupervised draping. In Proc. of Conference on Computer Vision and Pattern Recognition (CVPR), pages 1451–1460, 2023. 2

[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Proc. of International Conference on Learning Representations (ICLR), 2021. 3

[12] Antoine Dumoulin, Adnane Boukhayma, Laurence Boissieux, Bharath Bhushan Damodaran, Pierre Hellier, and Stefanie Wuhrer. D-garment: Physically grounded latent diffusion for dynamic garment deformations. Transactions on Machine Learning Research, 2026. 2, 3, 4, 5, 6, 7

[13] Artur Grigorev, Michael J. Black, and Otmar Hilliges. HOOD: hierarchical graphs for generalized modelling of clothing dynamics. In Proc. ofConference on Computer Vi sion and Pattern Recognition (CVPR), 2023. 1, 3

[14] Artur Grigorev, Giorgio Becherini, Michael Black, Otmar Hilliges, and Bernhard Thomaszewski. Contourcraft: Learn ing to resolve intersections in neural multi-garment simula tions. In ACM Transactions on Graphics (TOG), Proc. SIG GRAPH, pages 1–10, 2024. 3, 6, 7

[15] Xianfeng Gu, Steven J Gortler, and Hugues Hoppe. Geometry images. In ACM Proc. of Computer Graphics and Inter active Techniques (PACMCGIT), pages 355–361, 2002. 4

[16] Jingfan Guo, Fabian Prada, Donglai Xiang, Javier Romero, Chenglei Wu, Hyun Soo Park, Takaaki Shiratori, and Shun suke Saito. Diffusion shape prior for wrinkle-accurate cloth registration. In Proc. of International Conference on 3D Vision (3DV), page 790–799, 2024. 5

[17] Jingfan Guo, Jae Shin Yoon, Shunsuke Saito, Takaaki Shiratori, and Hyun Soo Park. High-fidelity modeling of generalizable wrinkle deformation. In Proc. of European Con ference on Computer Vision (ECCV), pages 429–445, 2025. 5

[18] Maria Korosteleva and Olga Sorkine-Hornung. Garment-Code: Programming parametric sewing patterns. ACM Transactions on Graphics (TOG), Proc. SIGGRAPH Asia, 42(6), 2023. 2, 6

[19] Maria Korosteleva, Timur Levent Kesdogan, Fabian Kem per, Stephan Wenninger, Jasmin Koller, Yuhan Zhang, Mario Botsch, and Olga Sorkine-Hornung. GarmentCodeData: A dataset of 3D made-to-measure garments with sewing pat-

terns. In Proc. of European Conference on Computer Vision (ECCV), 2024. 2, 6, 8

[20] Hunor Laczko, Meysam Madadi, Sergio Escalera, and´ Jordi Gonzalez. A generative multi-resolution pyramid and normal-conditioning 3d cloth draping. In Proc. of the Winter Conference on Applications of Computer Vision (WACV), pages 8709–8718, 2024. 3, 4, 6, 7

[21] Zorah Lahner, Daniel Cremers, and Tony Tung. Deepwrinkles: Accurate and realistic clothing modeling. In Proc. of European Conference on Computer Vision (ECCV), pages 667–684, 2018. 2, 3

[22] Jie Li, Gilles Daviet, Rahul Narain, Florence Bertails-Descoubes, Matthew Overby, George E. Brown, and Laurence Boissieux. An implicit frictional contact solver for adaptive cloth simulation. ACM Transactions on Graphics (TOG), 37(4), 2018. 2

[23] Peizhuo Li, Tuanfeng Y Wang, Timur Levent Kesdogan, Duygu Ceylan, and Olga Sorkine-Hornung. Neural garment dynamics via manifold-aware transformers. Computer Graphics Forum, page e15028, 2024. 3

[24] Ren Li, Benoit Guillard, Edoardo Remelli, and Pascal Fua. DIG: Draping Implicit Garment over the Human Body. In Proc. of Asian Conference on Computer Vision (ACCV), 2022. 2

[25] Ren Li, Corentin Dumery, Zhantao Deng, and Pascal Fua. Reconstruction of manipulated garment with guided deformation prior. In Advances in neural information processing systems, pages 58637–58662, 2024. 5

[26] Ren Li, Corentin Dumery, Benoˆıt Guillard, and Pascal Fua. Garment recovery with shape and deformation priors. In Proc. ofConference on Computer Vision and Pattern Recognition (CVPR), pages 1586–1595, 2024. 5

[27] Siyou Lin, Hongwen Zhang, Zerong Zheng, Ruizhi Shao, and Yebin Liu. Learning implicit templates for point-based clothed human modeling. In Proc. of European Conference on Computer Vision (ECCV), 2022. 3

[28] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. Proc. of International Conference on Learning Representations (ICLR), 2023. 5

[29] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. Smpl: a skinned multiperson linear model. ACM Transactions on Graphics (TOG), 34, 2015. 5

[30] Mickael Ly, Jean Jouve, Laurence Boissieux, and Florence¨ Bertails-Descoubes. Projective dynamics with dry frictional contact. ACM Transactions on Graphics (TOG), 39, 2020. 3, 6

[31] Qianli Ma, Jinlong Yang, Anurag Ranjan, Sergi Pujades, Gerard Pons-Moll, Siyu Tang, and Michael J. Black. Learning to Dress 3D People in Generative Clothing. In Proc. of Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 3

[32] Qianli Ma, Shunsuke Saito, Jinlong Yang, Siyu Tang, and Michael J. Black. SCALE: Modeling clothed humans with a surface codec of articulated local elements. In Proc. of Conference on Computer Vision and Pattern Recognition (CVPR), pages 16082–16093, 2021. 3

[33] Qianli Ma, Jinlong Yang, Siyu Tang, and Michael J. Black. The power of points for modeling humans in clothing. In Proc. of International Conference on Computer Vision (ICCV), pages 10974–10984, 2021. 3

[34] Qianli Ma, Jinlong Yang, Michael J Black, and Siyu Tang. Neural point-based shape modeling of humans in challenging clothing. In Proc. of International Conference on 3D Vision (3DV), pages 679–689, 2022. 3

[35] Miles Macklin, Matthias Muller, and Nuttapong Chentanez.¨ Xpbd: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Confer ence on Motion in Games, pages 49–54, 2016. 3

[36] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black. AMASS: Archive of motion capture as surface shapes. In Proc. of International Conference on Computer Vision (ICCV), pages 5442–5451, 2019. 6

[37] Matthias Muller, Bruno Heidelberger, Marcus Hennix, and¨ John Ratcliff. Position based dynamics. Journal of Visual Communication and Image Representation, 18(2):109–118, 2007. 3

[38] Chaitanya Patel, Zhouyingcheng Liao, and Gerard Pons-Moll. Tailornet: Predicting clothing in 3d as a function of human pose, shape and garment style. In Proc. of Conference on Computer Vision and Pattern Recognition (CVPR), pages 7365–7375, 2020. 3

[39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proc. of International Conference on Computer Vision (ICCV), page 4172–4182. IEEE, 2023. 4, 5

[40] Sergey Prokudin, Qianli Ma, Maxime Raafat, Julien Valentin, and Siyu Tang. Dynamic point fields. In Proc. of International Conference on Computer Vision (ICCV), pages 7964–7976, 2023. 2, 3, 6, 7

[41] Xavier Provot. Deformation constraints in a mass-spring model to describe rigid cloth behaviour. In Graphics Inter face, pages 147–147, 1995. 2

[42] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention, pages 234–241, 2015. 3

[43] Igor Santesteban, Nils Thuerey, Miguel A Otaduy, and Dan Casas. Self-supervised collision handling via generative 3d garment models for virtual try-on. In Proc. of Conference on Computer Vision and Pattern Recognition (CVPR), pages 11763–11773, 2021. 3, 6

[44] Igor Santesteban, Miguel A Otaduy, and Dan Casas. Snug: Self-supervised neural dynamic garments. In Proc. of Conference on Computer Vision and Pattern Recognition (CVPR), pages 8140–8150, 2022. 3

[45] Yu Shen, Junbang Liang, and Ming C Lin. Gan-based garment generation using sewing pattern images. In Proc. of European Conference on Computer Vision (ECCV), pages 225–247, 2020. 3

[46] Min Shi, Wenke Feng, Lin Gao, and Dengming Zhu. Generating diverse clothed 3d human animations via a generative model. Computational Visual Media, 10(2):261–277, 2024. 3

[47] Demetri Terzopoulos, John Platt, Alan Barr, and Kurt Fleischer. Elastically deformable models. In ACM Proc. ofComputer Graphics and Interactive Techniques (PACMCGIT), pages 205–214, 1987. 2

[48] Lokender Tiwari and Brojeshwar Bhowmick. Garsim: Particle based neural garment simulator. In Proc. of the Win ter Conference on Applications ofComputer Vision (WACV), pages 4461–4470, 2023. 3

[49] Raquel Vidaurre, Elena Garces, and Dan Casas. Diffusedwrinkles: A diffusion-based model for data-driven garment animation. In Proc. of the British Machine Vision Conference (BMVC), 2024. 2, 3, 5

[50] Huamin Wang, James F O’Brien, and Ravi Ramamoorthi. Data-driven elastic models for cloth: modeling and measurement. ACM Transactions on Graphics (TOG), 30(4):1–12, 2011. 3

[51] Wenbo Wang, Hsuan-I Ho, Chen Guo, Boxiang Rong, Artur Grigorev, Jie Song, Juan Jose Zarate, and Otmar Hilliges. 4d-dress: A 4d dataset of real-world human clothing with semantic annotations. In Proc. ofConference on Computer Vision and Pattern Recognition (CVPR), pages 550–560, 2024. 2, 7, 8

[52] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution text-to-image synthesis with linear diffusion transformers. In Proc. of International Conference on Learning Representations (ICLR), 2025. 2, 7

[53] Jinlong Yang, Jean-Sebastien Franco, Franck H´ etroy-´ Wheeler, and Stefanie Wuhrer. Estimation of human body shape in motion with wide clothing. In Proc. of European Conference on Computer Vision (ECCV), pages 439–454, 2016. 5

[54] Ilya Zakharkin, Kirill Mazur, Artur Grigorev, and Victor Lempitsky. Point-based modeling of human clothing. In Proc. of International Conference on Computer Vision (ICCV), pages 14718–14727, 2021. 2, 3

[55] Meng Zhang, Duygu Ceylan, and Niloy J. Mitra. Motion guided deep dynamic 3d garments. ACM Transactions on Graphics (TOG), 41(6), 2022. 3, 4

[56] Shihao Zou, Yuanlu Xu, Nikolaos Sarafianos, Federica Bogo, Tony Tung, Weixin Si, and Li Cheng. Generating high-fidelity clothed human dynamics with temporal diffusion. Transactions on Multimedia Computing, Communications, and Applications, 21(3), 2025. 3

# DiT-Garment: Garment Dynamics with Diffusion Transformers

Supplementary Material

## 8. Evaluation on unseen factors

Table 4 evaluates DiT-Garment performance on each input parameter individually. The unseen factors include garment design, body motion, body shape, and physical material. Each test set contains the same 9 sequence input data sampled from our training set, resimulated with one specific factor replaced by an unseen value. We report the mean error for each metric as described in Section 5.2. Results show that DiT-Garment robustly generalizes to each unseen factor, with the lowest shape similarity errors for unseen physical material and the highest errors for unseen garment design. The lower $E _ { c }$ observed for unseen body motions is primarily due to the specific body topology of the sampled test motions, which feature fewer complex selfintersections compared to the broader training distribution. The higher errors in most metrics for unseen garment design can be explained by more complex designs than those in the training set, particularly one loose dress covering most of the body.

Table 4. Evaluation of our method on one unseen factor at a time. Each test set uses the same sampled sequences from our training set, but with one factor replaced by an unseen value (garment design, body motion, body shape, or physical material).
<table><tr><td rowspan="2">Unseen factor</td><td colspan="3">Shape Similarity</td><td colspan="3">Physical Validity</td></tr><tr><td> $E _ { v } ^ { \downarrow }$ </td><td> $E _ { C D } ^ { \downarrow }$ </td><td> $E _ { n } ^ { \downarrow }$ </td><td> $E _ { c } ^ { \downarrow }$ </td><td> $E _ { b } ^ { \downarrow }$   $E _ { s } ^ { \downarrow }$ </td><td> $E _ { d } ^ { \downarrow }$ </td></tr><tr><td rowspan="4">garment design body motion body shape</td><td>4.30</td><td>0.11</td><td>0.36</td><td>1.55</td><td>0.53</td><td>1.47</td><td>2.30</td></tr><tr><td>3.40</td><td>0.03</td><td>0.22</td><td>0.68</td><td>0.40</td><td>0.10</td><td>0.74</td></tr><tr><td>2.07</td><td>0.03</td><td>0.22</td><td>1.44</td><td>0.43</td><td>0.11</td><td>1.01</td></tr><tr><td>0.94</td><td>0.01</td><td>0.19</td><td>1.68</td><td>0.43</td><td>0.11</td><td>0.85</td></tr></table>

## 9. Statement of Broader Impact

The datasets used in this paper contain motion captures of real people. 4D-Dress and 4DHumanOutfit followed $\mathrm { r i g \mathrm { - } }$ orously ethics guidelines and GDPR rules. We used these datasets under permission given by their respective owners.

Positive Societal Impact. DiT-Garment enables realistic, physically grounded animation of 3D garments on human avatars, which has broad applications in virtual try-on, digital fashion, gaming, film, and healthcare. By supporting unseen garment designs and other factors, our method lowers the barrier for creative industries to produce customized, physically realistic virtual clothing without expensive perdesign simulation. The ability to generalize across captured, artist-made, and synthetic designs also promotes accessibility in virtual fashion, potentially reducing waste and promoting more sustainable practices in the apparel industry through more accurate digital prototypes.

Potential Societal Risks. Realistically animating garments on human avatars in arbitrary poses raises privacy and misuse concerns. In particular, realistic virtual humans could be used to generate non-consensual or deceptive imagery, or to infer sensitive body-related information from public or private data. Additionally, because our model is trained on synthetic body models, there is a risk that it may not generalize equitably across diverse body shapes, potentially introducing biases in virtual fashion applications. The environmental cost of training and running large diffusionbased models should also be considered.

Mitigation and Responsible Use. We commit to release code and data to encourage open, transparent research while encouraging downstream users to implement safeguards, such as access controls, watermarks, or consent mechanisms, when deploying models capable of generating realistic virtual humans. We also encourage evaluating fairness across diverse body types to reduce biases in 3D human avatars. Furthermore, by releasing our pre-trained models, we aim to reduce the need for redundant, energy-consuming training cycles by the community, thereby reducing the environmental impact.