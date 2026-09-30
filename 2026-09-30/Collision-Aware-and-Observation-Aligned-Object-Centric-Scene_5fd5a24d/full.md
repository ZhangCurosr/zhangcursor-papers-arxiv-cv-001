# Collision-Aware and Observation-Aligned Object-Centric Scene Reconstruction from Point Cloud

Yuxuan Xie<sup>1</sup>, Xuan Yu<sup>1</sup>, Rong Xiong<sup>1</sup>, Yue Wang<sup>1</sup>

Abstract— Object-centric scene reconstruction requires completing partial object observations while preserving metric alignment and avoiding collisions with the surrounding. Existing generation-based methods are often image-conditioned and suffer from scale ambiguity and insufficient geometric constraints. We propose COOL, a framework for COllisionaware and Observation-aLigned reconstruction. Based on an object generation model, COOL conditions the generation on instance and background point clouds. Instance geometry anchors generation in scene coordinates, while background geometry provides local context for scene-consistent completion. We further introduce an explicit collision loss and use joint optimization and resampling to reduce collisions during inference. Experiments on 3D-Front and Scan2CAD demonstrate strong scene-level fidelity, observation alignment, and collision reduction. Moreover, additional studies validate its robustness to mask errors and its applicability to real-world scene replicas.

## I. INTRODUCTION

Object-centric scene reconstruction is a task to recover high-fidelity and pose-aligned objects within the scene. Unlike scene-centric methods that reconstruct single scene entities with incomplete surfaces, object-centric reconstruction focuses on extracting complete and isolated object models [1], [2]. This capability holds promise for constructing digital replicas of real-world scenes and enabling real-tosim pipelines in robotics. To achieve this, object-centric reconstruction should complete object geometry independently while preserving metric partial observations and local scene compatibility. This joint objective presents two key challenges, the alignment with observation, and collision with other objects and scene.

Existing methods can be broadly categorized into two classes: regression-based and generation-based. Regressionbased methods [3]–[5] learn a one-to-one mapping from partial observations to complete shapes, where the result typically represents an average over all plausible shapes and often becomes overly blurry and smooth. Generationbased methods [2], [6]–[9] offer an alternative by leveraging object generation models [10]–[13] that learn shape priors to synthesize diverse and detailed object geometry.

Existing generation-based methods mainly condition generation on images and can synthesize high-quality geometry. However, their generated objects are often poorly aligned with the actual observations in the scene. Some staged pipelines [2], [6] generate in a canonical coordinate system and estimate pose and scale in the scene, causing alignment errors to accumulate across stages. Others [7]–[9] predict shape and layout jointly, but remain affected by the scale ambiguity of single-view images and can produce inaccurate object locations and relative sizes. In addition, independently generated objects may collide with each other or with the surrounding scene. Implicit scene relations encoded in images are often insufficient to prevent these physical collisions. Such errors are particularly problematic in robotics, where accurate alignment with sensor observations and collisionfree scene geometry are essential for reliable applications.

![](images/6f32850bf7a7cbe985a245247af473113e2b5ecbca422065faa0698dd3c5c909.jpg)  
Fig. 1. Image-conditioned generation-based methods for object-centric scene reconstruction produce inaccurate object locations and relative sizes. Our method fully exploits scene-coordinate metric geometry to gain an advantage in generating observation-aligned and collision-aware objects.

Depth camera and LiDAR in robotic systems provide accessible metric geometry for scene reconstruction. Unlike image features, point clouds contain direct metric cues about visible object geometry, scale, and pose. Thus generation conditioned on point clouds promotes metric alignment and provides a natural basis for checking geometric compatibility with surrounding structures. Existing point-cloudconditioned generation methods [9], [14], however, mainly use point clouds to improve object-level generation quality or provide coarse spatial anchors. Fully exploiting scenecoordinate metric geometry to model constraints during generation remains unsolved.

To this end, we propose COOL, a COllision-aware and Observation-aLigned framework for point-cloud-conditioned object-centric scene reconstruction. The key idea of COOL is to fully exploit scene geometry: in addition to objectlevel geometry condition, we use the surrounding geometry as a context condition, and introduce generation guidance from explicit geometric collision. Specifically, building upon a pretrained object generation model [13], COOL conditions the generation on the observed instance and background point cloud, which are back-projected by masked depth observations. The instance point cloud conditions generation to preserve metric alignment, while the background point cloud provides local spatial context for scene-consistent completion. Both of them play a role in improving observation alignment. To further resolve collisions, we construct an explicit collision loss between the generated object and the reconstructed surroundings. During inference, COOL jointly guides the generation under this loss and uses resampling to escape local optima. In this way, COOL reduces geometric overlap while preserving fidelity to the observed geometry.

We evaluate COOL on synthetic and real-world scene datasets. Comparative experiments demonstrate its effectiveness in reconstructing observation-aligned and collisionaware object geometries. We further evaluate robustness under realistic perception errors, extend the framework to multi-view observations, and conduct ablation studies to analyze the effectiveness of each proposed component. In summary, our main contributions are as follows:

• We propose a point-cloud-conditioned object generation framework that reconstructs objects directly in scene coordinates. By using observed instance geometry and local scene geometry as conditions, COOL improves observation fidelity in scale, pose, and scene consistency.

• We introduce an explicit collision loss between generated objects and reconstructed surroundings, together with a joint optimization and resampling strategy that effectively reduces collisions during inference.

• COOL achieves better performances in reconstructing scenes from both synthetic and real-world scene datasets, and we validate its robustness under realistic perception errors.

## II. RELATED WORK

## A. 3D Shape Completion

Traditional 3D shape completion methods [15], [16] rely on structural regularities, such as the symmetries, to infer missing geometry from partial observations. With the increasing availability of large-scale 3D datasets, approaches [3], [4] turn to design mapping networks under the supervision of ground truths. These methods aim to learn an optimal one-to-one mapping from partial inputs to complete objects, which often leads to an averaged, blurry solution that lacks fine-grained details. Subsequently, shape completion methods based on generative models [17], [18], are developed to produce completions from partial inputs, achieving greater generative flexibility. However, they are mainly trained on synthetic datasets, leading to performance degradation when applied to real-world object completion.

## B. 3D Object Generation

3D object generation focuses on generating object geometry and texture from input condition. Building on the success of 2D generation models [19], [20], 3D object generation evolves from optimization methods based on Score Distil lation Sampling (SDS) [10] to multi-view consistent reconstruction models [11]. However, these approaches struggle to preserve geometric fidelity as they primarily prioritize 2D image quality over explicit 3D structure. Benefiting from the increase of 3D datasets [21], [22] and advances in geometric representations, recent works shift towards training large-scale native 3D generative models [12], [13]. These approaches typically employ a two-stage pipeline: a Variational Autoencoder (VAE) [23] compresses 3D geometry into latent space, followed by a latent generative model, which improves both fidelity and efficiency. Nevertheless, these methods remain largely confined to single-object synthesis and assume clean, unobstructed inputs.

## C. Object-Centric Scene Reconstruction

Object-centric scene reconstruction has evolved from retrieval-based CAD alignment [24] to single-view reconstruction of individual objects and overall scene layout [5], both of which are constrained by limited categories. Recently, the field has witnessed a paradigm shift toward leveraging generative 3D models to generate objects and reassembling them into the scene. Most of these methods use image as condition. Some [2], [6] follow canonical-space object generation with independent pose estimation, suffering from the error accumulation that leads to the misalignment of objects in the scene. Others [7], [8] generate shape together with layout, while they often yield inaccurate relative scales and spatial relations since the single view scale ambiguity. Besides, current methods [9], [14] use point clouds to assist geometric generation. SAM3D [9] uses a point map as the supplementary of image condition, but lacks explicit geometric constraints. 3D-Fixer [14] leverages point cloud as a spatial anchor to preserve scene layout, but lacks effective constraints from surrounding geometry.

## III. METHOD

Given the RGB-D observation (I, D, P) of a scene, where I and D denote the aligned RGB image and depth map and P is the camera projection matrix, together with a targetinstance mask M, our goal is to reconstruct complete object geometry with observation-aligned scale, pose and shape, which can be directly assembled into a 3D scene. We backproject valid depth pixels to obtain the scene point cloud $P _ { \mathrm { s c e n e } } ,$ , and use mask M to segment it into instance point cloud $P _ { \mathrm { i n s } }$ and background point cloud $P _ { \mathrm { b g } } = P _ { \mathrm { s c e n e } } \setminus P _ { \mathrm { i n s } } .$ COOL models the conditional distribution $v _ { \theta } ( y \mid P _ { \mathrm { i n s } } , P _ { \mathrm { b g } } )$ where y is the completed object geometry represented in the original scene coordinate system.

The remainder of this section details our approach. Based on the 3D object generation model TRELLIS [13] (Sec.III-$\mathbf { A } ) .$ we perform geometry-conditioned instance generation in scene coordinates with instance condition and context condition (Sec.III-B). Besides, collisions are optimized through a joint optimization and resampling strategy during inference with a collision loss (Sec.III-C). The training details are introduced in Sec.III-D. Fig. 2 shows the pipeline of COOL.

## A. Preliminary: 3D Object Generation

Our model is built based on TRELLIS [13], a conditional 3D object generation model, employing rectified flow models to denoise in a structured latent space, where the 3D object is represented as $\boldsymbol { z } = \{ ( z _ { i } , p _ { i } ) \} _ { i = 1 } ^ { L }$ , where $p _ { i } \in \{ 0 , 1 , . . . , N -$ $1 \bar  \} ^ { 3 }$ represents the active voxel position of a 3D grid, and $\boldsymbol { z } _ { i } \in \mathbb { R } ^ { C }$ is a local latent attached to the voxel.

![](images/9fc38dc600ddfe94bc534edc192c33b9f6034cfdbcb6decf3e7e192437b4e935.jpg)  
Fig. 2. Overview of COOL. Using segmented and normalized scene point cloud as input, COOL extracts the instance condition and context condition from instance and background point clouds through the condition extraction network. In conditional generation module, these conditions are introduced through MHA layers in the flow transformer. The generated objects are placed back to the reconstructed scene to compute the geometric overlap as a collision loss, optimizing collisions via a joint optimization and resampling strategy during inference.

TRELLIS uses a two-staged pipeline for separate geometry and texture generation, generating the sparse structure first, and then the local latents on it. Both of the stages train a network $v _ { \theta }$ based on rectified flow models to move toward the data distribution $x _ { 0 }$ from $x _ { t } = ( 1 - t ) x _ { 0 } + t \epsilon .$ , which is the interpolation of data samples $x _ { 0 }$ and noises ϵ at timestep t. The objective of training the network $v _ { \theta }$ is to minimize the conditional flow matching:

$$
\mathbb { E } _ { t , { x _ { 0 } } , \epsilon } \lVert { v _ { \theta } ( x _ { t } , t ) - ( \epsilon - x _ { 0 } ) } \rVert _ { 2 } ^ { 2 } .\tag{1}
$$

During the geometry generation stage, the neural network $\mathcal { G } _ { S }$ generates a low-resolution feature grid $S \in$ $\mathbb { R } ^ { D \times D \times \tilde { D } \times \tilde { C } _ { S } }$ , which can be decoded to a N-resolution 3D grid O using a 3D convolutional decoder $\mathcal { D } _ { S }$ . The text and image conditions are inserted as the keys and values of the cross attention layers to guide conditional generation.

## B. Geometry-Conditioned Instance Generation

In this section, we model metric geometry of the object as instance condition to guide generation, which preserves the metric scale and pose of the observation. Besides, this kind of scene-aligned generation makes scene observation surround ing the object an effective context condition. The instance and context condition jointly promote observation-aligned generation, as the instance condition anchors the object-level shape, pose, and metric scale, whereas the context condition provides spatial context for scene-level consistency.

Instance Condition. As illustrated in Fig. 2, we employ an attention-based point cloud encoder $\mathcal { E } _ { i n s }$ to derive the instance condition $\dot { c } _ { \mathrm { i n s } }$ from the instance partial point cloud $P _ { \mathrm { i n s } }$ . This process utilizes the sub-sampled instance point cloud $P _ { \mathrm { i n s } } ^ { s }$ to query the original instance point cloud $P _ { \mathrm { i n s } } \mathrm { : }$

$$
c _ { \mathrm { i n s } } = \mathrm { A t t e n t i o n } ( \mathrm { P o s E m b } ( P _ { \mathrm { i n s } } ^ { s } ) , \mathrm { P o s E m b } ( P _ { \mathrm { i n s } } ) ) ,\tag{2}
$$

where PosEmb(·) denotes the positional encoding applied to the 3D coordinates of the point clouds. The attention function Attention(·) consists of cross-attention layers that aggregate fine-grained geometric details from the original partial points $P _ { \mathrm { i n s } } .$ , and self-attention layers that consolidate these details into a coherent, high-level semantic representation.

Context Condition. With the same structure as $\mathcal { E } _ { i n s }$ , an attention-based encoder $\mathcal { E } _ { c o n t }$ is used to derive the context condition, where the sub-sampled background point cloud $P _ { \mathrm { b g } } ^ { s }$ interacts with the masked scene point cloud $P _ { \mathrm { s c e n e } }$ via cross-attention to capture the instance-aware context:

$$
c _ { \mathrm { c o n t } } = \mathrm { A t t e n t i o n } ( \mathrm { P o s E m b } ( P _ { \mathrm { b g } } ^ { s } ) , \mathrm { P o s E m b } ( P _ { \mathrm { s c e n e } } \# M ) ) ,\tag{3}
$$

where M denotes the binary mask of the instance on the scene point cloud, and # represents concatenation.

By encoding cross attention between instance and background geometry, context condition incorporates the relationship between instances and their surroundings.

Generation under Condition. Based on the shape generation model of text-based TRELLIS, we add two multi-head attention blocks (MHA) to the flow transformer, with one for instance condition $\dot { c } _ { \mathrm { i n s } }$ and the other for context condition $c _ { \mathrm { c o n t } } .$ , as shown in Fig. 2. Within the cross-attention layer, the latent feature $f$ acts as the query to attend to the condition $c ,$ which serves as the key and value. This operation enables $f$ to incorporate relevant information from c:

$$
f ^ { \prime } = \mathrm { A t t e n t i o n } ( f , c ) , \qquad c \in \{ c _ { \mathrm { i n s } } , c _ { \mathrm { c o n t } } \} .\tag{4}
$$

Guided by the geometry conditions, the generated objects are effectively aligned with the instance partial observations, in terms of geometric shape, metric scale and pose, and scene-level consistency. By reassembling the independently generated instances, we obtain the reconstructed scene.

![](images/d5d8e0cd0ffb07e35c28926d663588b54750fc1f65416a89abe50addbb1f4e23.jpg)  
(i)

![](images/e229ce4428e1257dc4ec30bda891bd2ebc254facf40545be81de4911bdc8033b.jpg)  
(ii)

![](images/fec8602b570db1d933f6f7a76a1badac28a11f61fce7c3e8434d555358e418c2.jpg)  
(iii)

![](images/a175ad8d23bfeb4ad963bef65d9a2c8a5519af3b6cd6c5a500ed9a853f1eb0da.jpg)  
(iv)  
Fig. 3. A case of joint optimization and resampling. (i) Initial generation with collision. (ii) Gradient-guided optimization converged to a local optima. (iii) Resampling escaping the local optima. (iv) Gradient-guided optimization achieving a collision-free reconstruction.

## C. Collision Optimization

Despite correct observation alignment and implicit scene context priors, physical collisions are inevitable. To address this, we introduce an explicit collision loss to guide collisionaware denoising trajectory during inference with joint optimization and resampling strategy.

Collision Loss. During inference, we construct an explicit scene-level collision constraint from the observed background geometry. For each target instance, we first remove its pixels from the depth observations using the instance mask, and fuse the remaining depth maps into a background Truncated Signed Distance Function (TSDF) volume [25].

For each generated sparse-structure grid index $p _ { i } ,$ , we recover its corresponding world coordinate $p _ { i } ^ { w }$ through the inverse of the instance normalization transform described in Sec.III-D. Let $d _ { i }$ denote the sampled signed distance value at $p _ { i } ^ { w }$ , and let $m _ { i } \in \{ 0 , 1 \}$ indicate whether this location is observed in the fused TSDF volume.

Given the predicted sparse structure latent $x _ { 0 } ,$ the decoder $D _ { S }$ outputs occupancy logits

$$
o = D _ { S } ( x _ { 0 } ) , \qquad o _ { i } \in \mathbb { R } .\tag{5}
$$

A collision occurs when a generated occupied voxel lies inside the observed background, i.e., $o _ { i } > 0 , d _ { i } < 0$ , and $m _ { i } = 1$ . We define the penetration loss as

$$
\mathcal { L } _ { c o l } = \sum _ { i } \mathbf { 1 } [ m _ { i } = 1 ] \mathbf { 1 } [ d _ { i } < 0 ] \mathbf { 1 } [ o _ { i } > 0 ] ( - d _ { i } ) _ { + } \left( o _ { i } \right) _ { + } .\tag{6}
$$

This loss penalizes both the penetration depth $- d _ { i }$ and the confidence of generated occupancy $o _ { i }$

Gradient and Resampling. We use the collision loss to explicitly guide the denoising trajectory during inference without updating the pretrained generator. At each denoising step t, the flow model predicts a velocity $v _ { t }$ , from which the clean sparse structure latent is estimated as $\hat { x } _ { 0 } = x _ { t } - t v _ { t } ,$ where the predicted $\scriptstyle { \hat { x } } _ { 0 }$ is decoded into occupancy logits and evaluated by the collision loss. Since both the flow model and decoder are frozen, gradients are only used to guide the current sampling trajectory:

$$
\begin{array} { r l r } & { } & { g _ { t } = \nabla _ { \hat { x } _ { 0 } } \mathcal { L } _ { c o l } ( D _ { S } ( \hat { x } _ { 0 } ) ) , } \\ & { } & { \nabla _ { v _ { t } } \mathcal { L } _ { \mathrm { c o l } } = \left( \frac { \partial \hat { x } _ { 0 } } { \partial v _ { t } } \right) ^ { \top } \nabla _ { \hat { x } _ { 0 } } \mathcal { L } _ { \mathrm { c o l } } = - t g _ { t } . } \end{array}\tag{7}
$$

We convert this gradient into a correction of the flow velocity. The step size is normalized by the relative magnitudes of the predicted velocity and collision gradient:

$$
\alpha _ { t } = \mathrm { c l i p } \left( \frac { \rho \| v _ { t } \| _ { 2 } } { t \| g _ { t } \| _ { 2 } + \epsilon } , \alpha _ { \operatorname* { m i n } } , \alpha _ { \operatorname* { m a x } } \right) ,\tag{8}
$$

where $\rho$ controls the relative strength of collision guidance, $\alpha _ { \mathrm { { m i n } } }$ and $\alpha _ { \mathrm { m a x } }$ bound the correction magnitude. A gradientdescent update is performed on the predicted velocity:

$$
\hat { v } _ { t } = v _ { t } - \alpha _ { t } \nabla _ { v _ { t } } \mathcal { L } _ { \mathrm { c o l } } = v _ { t } + \alpha _ { t } t g _ { t } .\tag{9}
$$

The latent is then updated using the rectified-flow Euler step:

$$
x _ { t ^ { \prime } } = x _ { t } - ( t - t ^ { \prime } ) \hat { v } _ { t } .\tag{10}
$$

We apply collision guidance during the later denoising stage to preserve the generative prior in early sampling steps.

Because collision-aware optimization is non-convex, a single initial noise may still converge to a poor local solution. Thus we use resampling as a stochastic multi-start strategy. For each instance, we sample $N _ { s }$ independent initial noises

$$
\epsilon ^ { ( k ) } \sim { \mathcal { N } } ( 0 , I ) , \qquad k = 1 , \dots , N _ { s } ,\tag{11}
$$

run the same gradient-guided sampling procedure for each candidate, and select the final result by prioritizing lower collision loss. This simple resampling strategy complements gradient-based refinement by exploring multiple plausible regions of the latent space, thereby reducing the chance of being trapped in a collision-prone local optimum. Fig. 3 shows a case of joint optimization and resampling.

## D. Training

Input Preparation. We normalize the input instance point cloud using its center c and scale $s ,$ while applying the same transformation to the associated background points. The generator therefore operates in a normalized coordinate system but retains the relative geometry between the target and its surroundings. The generated object is transformed back using (c, s) before scene assembly.

Training Strategy. During training, we augment the data by randomly merging point clouds from 1 to 5 frames of the same instance. We finetune the pretrained model of TRELLIS by introducing instance condition and context condition in stages. After the instance condition finetuning stabilizes, we freeze the model parameters and add a new attention module for the context condition, which is then trained separately. This strategy helps better preserve the priors of the pretrained model, leading to more stable and higher-accuracy results.

## IV. EXPERIMENTS

## A. Setup

Datasets. We trained our model on the 3D-Front dataset [21] and the Scan2CAD dataset [26]. The 3D-Front dataset is a synthetic indoor scene dataset. We follow the split of prior work [7] and use 8K training scenes and 1K test scenes. The Scan2CAD dataset is a real-world dataset that matches CAD models to real-world indoor scene scans. We split it into disjoint training and testing sets, and select 10 views

Ours

TABLE I  
QUANTITATIVE COMPARISONS ON THE 3D-FRONT DATASET AND THE SCAN2CAD DATASET
<table><tr><td rowspan="2">Method</td><td colspan="7">3D-Front</td><td colspan="7">Scan2CAD</td></tr><tr><td> $\mathrm { C D } _ { \mathrm { S } } \downarrow$ </td><td>FSs ↑</td><td>CDo ↓</td><td>FSo ↑</td><td> $\mathrm { I o U } _ { \mathrm { B } } \uparrow$ </td><td> $\mathrm { C o l o \ 4 }$ </td><td> $\mathrm { C o l s \ : \downarrow }$ </td><td> $\mathrm { C D } _ { \mathrm { S } } \downarrow$ </td><td> $\mathrm { F S _ { S } \uparrow }$ </td><td>CDo ↓</td><td>FSo ↑</td><td>IoUB ↑|</td><td> $\mathrm { C o l o ~ } \downarrow$ </td><td>Cols ↓</td></tr><tr><td>AdaPoinTr</td><td>0.002</td><td>0.965</td><td>0.026</td><td>0.895</td><td>0.853</td><td>0.010</td><td>0.078</td><td>0.018</td><td>0.841</td><td>0.062</td><td>0.532</td><td>0.597</td><td>0.329</td><td>0.023</td></tr><tr><td>InstPIFu</td><td>0.065</td><td>0.633</td><td>0.088</td><td>0.507</td><td>0.269</td><td>0.460</td><td>0.503</td><td>0.087</td><td>0.518</td><td>0.072</td><td>0.494</td><td>0.219</td><td>0.510</td><td>0.159</td></tr><tr><td>MIDI</td><td>0.008</td><td>0.929</td><td>0.026</td><td>0.834</td><td>0.579</td><td>0.072</td><td>0.076</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td>Ours</td><td>0.001</td><td>0.989</td><td>0.011</td><td>0.904</td><td>0.895</td><td>0.002</td><td>0.003</td><td>0.015</td><td>0.894</td><td>0.036</td><td>0.745</td><td>0.641</td><td>0.027</td><td>0.000</td></tr></table>

![](images/a99c23d6dc6fce2f9e2851c28bcb3d09f4fc2d30a56e9c0b98e4b9fe0c28cfae.jpg)  
Input view

![](images/39aa23e8fbec0820dc470b1ba1075877f7c5d9794f637278b1e5352ae964eb7a.jpg)

![](images/541b6bbf5a18c303e816b9d738bdf58be8c19c9d3c0b7a799ef03612fdeac97b.jpg)

![](images/26755a4592d49c414091b85ec5341f5b2e68b1461945c8cbf43cca143e54792f.jpg)

![](images/19965016440943949c39f995e24a58bcf020b5ba89cce1108d16faf1e5722698.jpg)  
AdaPoinTr  
Fig. 4. Qualitative Comparisons on the 3D Front Dataset.

for each instance according to visibility and depth quality, constructing 90K training samples with 1.4K scenes.

Input Protocol. Unless otherwise stated, we use groundtruth instance masks to isolate reconstruction quality. For realistic perception evaluation, we replace them with masks predicted by segmentation models [27], [28] while keeping the remaining process unchanged.

Baselines. We compare our method with both regressionbased and generation-based methods. The former includes AdaPoinTr [4] for point cloud completion and InstPIFu [5] for feed-forward reconstruction. Generative baselines comprise the multi-stage method Gen3DSR [6], end-to-end methods including image-conditioned MIDI [7] and RGBDconditioned SAM3D [9] and 3D-Fixer [14]. All the methods in comparative study work under a single-view setting. COOL uses only geometric observations, while SAM3D and 3D-Fixer additionally use RGB information.

Evaluation Metrics. We compute both object-level and scene-level Chamfer Distance (CD) and F-Score (FS) to evaluate the quality of geometric reconstruction. The Volumetric Intersection over Union (IoU) of objects is used to assess the accuracy of spatial layout in generated scenes. Additionally, we employ the inter-object collision rate $\left( \mathrm { C o l _ { O } } \right)$ and objectscene collision rate (Col ) to report the collision situation.

TABLE II  
QUANTITATIVE COMPARISONS ON THE SCAN2CAD DATASET WITH ZERO-SHOT METHODS
<table><tr><td>Method</td><td> $\mathrm { C D } _ { \mathrm { S } } \downarrow$  ↓FSo↑ IoUB ↑|Colo↓Cols ↓</td><td>FSs ↑</td><td> $\mathrm { C D o }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>Gen3DSR</td><td>0.050</td><td>0.655</td><td>0.085</td><td>0.443</td><td>0.398</td><td>0.291</td><td>0.093</td></tr><tr><td>SAM3D</td><td>0.032</td><td>0.747</td><td>0.040</td><td>0.691</td><td>0.382</td><td>0.322</td><td>0.050</td></tr><tr><td>3D-Fixer</td><td>0.042</td><td>0.791</td><td>0.059</td><td>0.595</td><td>0.464</td><td>0.286</td><td>0.067</td></tr><tr><td>Ours(zs)</td><td>0.029</td><td>0.841</td><td>0.068</td><td>0.587</td><td>0.549</td><td>0.041</td><td>0.000</td></tr></table>

Implementation. Before condition extraction, input point clouds are sampled by Farthest Point Sampling (FPS) to 512. The condition extraction module includes 8 attention blocks and the dimension of query is 768. Our method is based on the pretrained base text-to-3D model of TRELLIS [13] and we follow the same sampling schedule and strategy for generation. For collision-aware inference, we use 25 rectified-flow sampling steps and collision guidance is activated during the last 13 sampling steps. We set the relative guidance strength to $\rho ~ = ~ 0 . 1 0$ , with $\alpha _ { \mathrm { m i n } } ~ = ~ 0 . 1$ and $\alpha _ { \mathrm { m a x } } ~ = ~ 1 0 . 0$ . For resampling, we generate $N _ { s } = 4$ candidates for each instance and select the result with the lowest collision loss.

## B. Comparative Study

We evaluate COOL under two complementary protocols. Tab. I reports in-domain comparisons with baselines avail-

Input view

![](images/82023e7fcf4e7bee06d5cc7dbe9992034fbf694391d6389e8ad8940c1fd8f883.jpg)  
AdaPoinTr  
InstPIFu  
Gen3DSR  
SAM3D  
3D-Fixer  
Ours

able training on the corresponding dataset. Each method is evaluated on the test split after training on the training split of 3D-Front or Scan2CAD, respectively. In contrast, Tab. II reports zero-shot transfer to Scan2CAD for methods that are generalizable and only available as pretrained checkpoints. For a comparable setting, Ours(zs) is trained on 3D-Front only and directly evaluated on the Scan2CAD test split.

![](images/5084e1fac86fd93d6e46a26e83ff4adce934f7bcd4a23a76851296dea2a4ec48.jpg)  
w/o collision guidance

Quantitative Results. In Tab. I, COOL achieves superior performance on both 3D-Front and Scan2CAD. It consistently improves object-level reconstruction quality, scenelevel fidelity, and collision rates. Tab. II further evaluates cross-dataset transfer to Scan2CAD without using its training data. SAM3D produces strong object-level geometry under its official zero-shot setting, obtaining the best object-level CD and F-score. Although COOL(zs) obtains lower objectlevel reconstruction quality than SAM3D, it achieves better scene-level reconstruction quality and collision mitigation. These results indicate that, while zero-shot object shape quality remains limited by the training data, the proposed instance and context geometry conditions effectively preserve scenecoordinate scale and pose alignment, producing spatially consistent and physically compatible scene reconstructions than the compared zero-shot methods.

![](images/fea06f904eaa0f4526e6c01e2cccf1b45a0bfe400120c351f8b7b8ed2e52003a.jpg)  
w/ collision guidance

Fig. 5. Qualitative Comparisons on the Scan2CAD Dataset.  
![](images/e73bce62543a36fe7116cff546ceb4b9518cfa88d837e1e0dea7e3d73c7a0f5a.jpg)

![](images/cfbb4432d904d3a00c202f731e91f4f50f616755240f0dfdba54022b2eb07258.jpg)  
w/o collision guidance  
w/ collision guidance  
Fig. 6. Qualitative Results on Collision Loss Guidance.

Qualitative Results. Fig. 4 and Fig. 5 present the qualitative comparisons on the two datasets. For object-level quality, generation-based approaches generally outperform regression-based baselines. AdaPoinTr excels in learning dataset-specific shape correspondences, thus performs well on 3D-Front, whose training and test sets share the CAD library, but its quality visibly degrades on Scan2CAD.

Generation quality, however, does not ensure observationaligned reconstruction. Gen3DSR generates each object through multiple stages, where errors in image completion and pose estimation accumulate. MIDI estimates spatial arrangement primarily from image cues, resulting in inaccurate relative scales and object locations. SAM3D uses RGB-D as condition, but its pointmap acts as a generative condition rather than an explicit observation-fitting constraint. Thus its generated pose and scale fail to tightly align with the observation, leading to lower volumetric IoU and higher collision rates. 3D-Fixer uses partial geometry to anchor object scale and pose, but does not constrain the generation against surrounding scene geometry, insufficient to ensure physically compatible placement.

## C. Ablation Study

We conduct ablation studies on the Scan2CAD dataset to validate the effectiveness of each component. Tab. III reports reconstruction metrics, collision rates and inference time.

TABLE III  
ABLATION STUDIES ON THE SCAN2CAD DATASET
<table><tr><td>C.C.</td><td>C.L.</td><td>R.S.</td><td> $\mathrm { C D } _ { \mathrm { S } } \downarrow$ </td><td> $\mathrm { F S _ { S } \uparrow }$ </td><td> $\mathrm { C D } _ { \mathrm { O } } \downarrow$ </td><td> $\mathrm { F S _ { O } }$  个</td><td> $\mathrm { I o U } _ { \mathrm { B } } \uparrow$ </td><td> ${ \mathrm { C o l } } _ { \mathrm { O } } \downarrow$ </td><td> $\mathrm { C o l } _ { \mathrm { S } } \downarrow$ </td><td> $\mathrm { t } _ { \mathrm { i n f e r } } \downarrow$ </td></tr><tr><td>X</td><td>X</td><td>X</td><td>0.019</td><td>0.876</td><td>0.040</td><td>0.710</td><td>0.618</td><td>0.245</td><td>0.016</td><td>3.2s</td></tr><tr><td>√</td><td>X</td><td>X</td><td>0.014</td><td>0.896</td><td>0.037</td><td>0.744</td><td>0.624</td><td>0.203</td><td>0.015</td><td>3.4s</td></tr><tr><td>√</td><td>√</td><td>X</td><td>0.016</td><td>0.891</td><td>0.038</td><td>0.742</td><td>0.633</td><td>0.095</td><td>0.003</td><td>3.9s</td></tr><tr><td>V</td><td>X</td><td>√</td><td>0.018</td><td>0.885</td><td>0.038</td><td>0.740</td><td>0.622</td><td>0.089</td><td>0.009</td><td>5.9s</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.015</td><td>0.894</td><td>0.036</td><td>0.745</td><td>0.641</td><td>0.027</td><td>0.000</td><td>7.8s</td></tr></table>

C.C. denotes context condition, C.L. denotes collision-loss guidance, R.S. denotes resampling.

TABLE IV  
STUDY ON ROBUSTNESS UNDER PERCEPTION NOISE
<table><tr><td>Mask</td><td> $\mathrm { C D } _ { \mathrm { S } } \downarrow$ </td><td> $\mathrm { F S _ { S } \uparrow }$ </td><td> $\mathrm { C D } _ { \mathrm { O } } \downarrow$ </td><td> $\mathrm { F S _ { O } \uparrow }$ </td><td> $\mathrm { I o U } _ { \mathrm { B } } \uparrow$ </td></tr><tr><td>SAM [27]</td><td>0.010</td><td>0.985</td><td>0.024</td><td>0.810</td><td>0.727</td></tr><tr><td>PanoRecon [28]</td><td>0.002</td><td>0.994</td><td>0.016</td><td>0.810</td><td>0.766</td></tr><tr><td>GT</td><td>0.001</td><td>0.995</td><td>0.015</td><td>0.816</td><td>0.767</td></tr></table>

![](images/b48eb9815805b72998fd23ae4effe8236327f82ab5a7a84867fe72739b4c9b63.jpg)

![](images/ea57d64e9be484bf6177b83c1c8beee32eb61ee4be56bdafb0cdbd75146a37c1.jpg)  
Fig. 7. Qualitative Results under Perception Noise.

Context Condition. The first two rows of Tab. III evaluate the effect of the context condition. Adding the context condition improves both reconstruction and spatial consistency, improving scene and object level reconstruction metrics and reducing collisions. However, the collision rates remain high. Therefore, context condition improves scene-consistent generation but cannot explicitly enforce collision-free geometry.

Collision Loss. The third row evaluates collision-guided denoising using one sample. Introducing collision-loss guidance reduces the object-object collision rate from 0.203 to 0.095 and the object-scene collision rate from 0.015 to 0.003. Meanwhile, the object-level and spatial-layout metrics remain close to those of the unoptimized result. This indicates that collision-loss guidance can modify the denoising trajectory to resolve geometric intersections without compromising the plausibility and observation consistency. Fig. 6 shows cases of collision-guided optimization.

Joint Optimization and Resampling. The last two rows evaluate the effect of resampling with four samples. The fourth row confirms that stochastic sampling can occasionally discover lower-collision solutions while it does not reliably eliminate collisions. When collision-loss-guided op timization is applied to all four samples, collision rates are further reduced. These show that resampling and optimization are complementary: optimization steers each sample toward lower-collision geometry, while resampling explores multiple plausible solutions. The object-level inference time shows that using four samples provides a practical balance between collision reduction and computational cost.

## D. Study on Robustness under Perception Noise

We evaluate COOL under two realistic mask sources: SAM, a 2D segmentation model [27], and PanoRecon, a panoptic reconstruction pipeline [28]. Tab. IV reports the results on 4 scenes in Scan2CAD. Compared with GT masks, SAM masks cause a performance decrease mainly caused by missed detection and incorrect segmentation, which can be substantially solved by PanoRecon, since it uses 3D consistency to correct these types of segmentation errors.

Fig. 7 provides qualitative analysis of different error types under SAM masks. When the mask omits part of the object, COOL can usually recover object structure using generative prior (Fig. 7(a)), while it may cause incomplete details (Fig. 7(b)). For point-cloud noise, depth-discontinuous outliers, commonly caused by RGB-depth misalignment near object boundaries, are largely suppressed by the point-cloud condition encoder (Fig. 7(c)). In contrast, mask errors can introduce depth-continuous points from nearby objects, which have a stronger influence on the generation (Fig. 7(d)). This result shows that COOL is robust to partial observations and isolated depth outliers, while geometric noise from incorrect masks remains a challenging failure mode.

## E. Study on Multi-view Condition

COOL natively supports multi-view point cloud inputs. We conducted experiments on Scan2CAD with 1, 3, 5-frame inputs, and point clouds sampled from reconstructed scene meshes (denoted as ”all”). As shown in Fig. 8, performance improves as more views are integrated, which confirms that richer observations enhance both local object fidelity and global spatial consistency.

## F. Case Study

Fig. 9 shows COOL<sup>′</sup>s performance in constructing geometric replicas for real-world scenes on ScanNet. COOL reconstructs and integrates complete object geometries into the scene to replace the incomplete objects in scene-centric reconstruction, showing its real-to-sim potential.

## V. CONCLUSION

We propose COOL, a framework for object-centric scene reconstruction from geometric observations. Geometry conditions promote observation-aligned and scene-consistent generation. During inference, collision-guided optimization together with resampling, further reduces collisions. Experimental results validate its efficacy and show its applicability in reconstructing real-world replicas for robotics.

Limitation. Compared with image-conditioned generators using richer visual cues, geometry-conditioned generation may lose fine-scale shape details. Limited computing resources restrict current training to relatively small datasets, so the generalization ability has not been fully explored.

## REFERENCES

[1] Y. Siddiqui, D. Frost, S. Aroudj, A. Avetisyan, H. Howard-Jenkins, D. DeTone, P. Moulon, Q. Wu, Z. Li, J. Straub, R. Newcombe, and J. Engel, “Shaper: Robust conditional 3d shape generation from casual captures,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 27 157–27 168.

![](images/fe1725c85fe7e6070bf660eed873041ac1f8fb53a84878264db9c384348e6066.jpg)

![](images/c15481422d34133631a9ef44b571833abbc2a0a8adec4973827655e20df39a8e.jpg)

![](images/3b51f2810d46f522ea4b5f6c9a0495ce12f32be24998dfc4c8720d7fd5dc5266.jpg)  
Fig. 8. Study on Multi-view Condition on the Scan2CAD Dataset.

![](images/37dcbfc14f94bc07ff1445a6bc39e8f700d085ed803b10b2da843ec731c44360.jpg)

![](images/282e1774d0028c724101d43daffda0c325f35b973562643ebef77058ee7e0451.jpg)

![](images/497bd4d181636671e55823e84cd79e54dd90522e594928a4f7ee79de3a768c68.jpg)

![](images/91f1301abbd01240c2312cf8d4ab5c17066ddc6f0d752ec79d72cd9de13c5431.jpg)

![](images/1bc161a687b902d5595d8bad8794062af17c6975166c45ddb034f13dda41a323.jpg)

![](images/100b0078197987ec457793f68b8313b505e7b1f0b148410b8388eaafff69a38a.jpg)  
Fig. 9. Cases on Geometric Replicas of Real-World Scenes.

[2] K. Yao, L. Zhang, X. Yan, Y. Zeng, Q. Zhang, L. Xu, W. Yang, J. Gu, and J. Yu, “Cast: Component-aligned 3d scene reconstruction from an rgb image,” ACM Transactions on Graphics (TOG), vol. 44, no. 4, pp. 1–19, 2025.

[3] Z. Huang, Y. Yu, J. Xu, F. Ni, and X. Le, “Pf-net: Point fractal network for 3d point cloud completion,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7659–7667.

[4] X. Yu, Y. Rao, Z. Wang, J. Lu, and J. Zhou, “Adapointr: Diverse point cloud completion with adaptive geometry-aware transformers,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 12, pp. 14 114–14 130, 2023.

[5] H. Liu, Y. Zheng, G. Chen, S. Cui, and X. Han, “Towards high-fidelity single-view holistic reconstruction of indoor scenes,” in European Conference on Computer Vision, 2022, pp. 429–446.

[6] A. Ardelean, M. Ozer, and B. Egger, “Gen3dsr: Generalizable 3d scene<sup>¨</sup> reconstruction via divide and conquer from a single view,” in 2025 International Conference on 3D Vision (3DV), 2025, pp. 616–626.

[7] Z. Huang, Y.-C. Guo, X. An, Y. Yang, Y. Li, Z.-X. Zou, D. Liang, X. Liu, Y.-P. Cao, and L. Sheng, “Midi: Multi-instance diffusion for single image to 3d scene generation,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 23 646–23 657.

[8] Y. Meng, H. Wu, Y. Zhang, and W. Xie, “Scenegen: Single-image 3d scene generation in one feedforward pass,” in 2026 International Conference on 3D Vision (3DV), 2026, pp. 543–553.

[9] X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li et al., “Sam 3d: 3dfy anything in images,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 7220–7232.

[10] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Textto-3d using 2d diffusion,” arXiv preprint arXiv:2209.14988, 2022.

[11] R. Liu, R. Wu, B. Van Hoorick, P. Tokmakov, S. Zakharov, and C. Vondrick, “Zero-1-to-3: Zero-shot one image to 3d object,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 9264–9275.

[12] B. Zhang, J. Tang, M. Niessner, and P. Wonka, “3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models,” ACM Transactions On Graphics (TOG), vol. 42, no. 4, pp. 1–16, 2023.

[13] J. Xiang, Z. Lv, S. Xu, Y. Deng, R. Wang, B. Zhang, D. Chen, X. Tong, and J. Yang, “Structured 3d latents for scalable and versatile 3d generation,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025, pp. 21 469–21 480.

[14] Z.-X. Yin, L. Liu, X. Wang, W. Sui, Z. Su, J. Yang, and J. Xie, “3dfixer: Coarse-to-fine in-place completion for 3d scenes from a single image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2026, pp. 12 753–12 763.

[15] N. J. Mitra, L. J. Guibas, and M. Pauly, “Partial and approximate symmetry detection for 3d geometry,” ACM Transactions on Graphics (ToG), vol. 25, no. 3, pp. 560–568, 2006.

[16] M. Pauly, N. J. Mitra, J. Wallner, H. Pottmann, and L. J. Guibas, “Dis-

covering structural regularity in 3d geometry,” in ACM SIGGRAPH 2008 papers, 2008, pp. 1–11.

[17] X. Yan, L. Lin, N. J. Mitra, D. Lischinski, D. Cohen-Or, and H. Huang, “Shapeformer: Transformer-based shape completion via sparse representation,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 6229–6239.

[18] Y. Li, Y. Dou, X. Chen, B. Ni, Y. Sun, Y. Liu, and F. Wang, “Generalized deep 3d shape prior via part-discretized diffusion process,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 16 784–16 794.

[19] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., “Photorealistic text-to-image diffusion models with deep language understanding,” Advances in neural information processing systems, vol. 35, pp. 36 479–36 494, 2022.

[20] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 10 674–10 685.

[21] H. Fu, B. Cai, L. Gao, L.-X. Zhang, J. Wang, C. Li, Q. Zeng, C. Sun, R. Jia, B. Zhao, and H. Zhang, “3d-front: 3d furnished rooms with layouts and semantics,” in 2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 10 913–10 922.

[22] M. Deitke, D. Schwenk, J. Salvador, L. Weihs, O. Michel, E. VanderBilt, L. Schmidt, K. Ehsanit, A. Kembhavi, and A. Farhadi, “Objaverse: A universe of annotated 3d objects,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 13 142–13 153.

[23] D. King and M. Welling, “Auto-encoding variational bayes,” in Proceedings of the 2nd International Conference on Learning Representations (ICLR), Banff, AB, Canada, 2014, pp. 14–16.

[24] D. Gao, D. Rozenberszki, S. Leutenegger, and A. Dai, “Diffcad: Weakly-supervised probabilistic cad model retrieval and alignment from an rgb image,” ACM Transactions on Graphics (TOG), vol. 43, no. 4, pp. 1–15, 2024.

[25] A. Zeng, S. Song, M. Nießner, M. Fisher, J. Xiao, and T. Funkhouser, “3dmatch: Learning local geometric descriptors from rgb-d reconstructions,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 199–208.

[26] A. Avetisyan, M. Dahnert, A. Dai, M. Savva, A. X. Chang, and M. Nießner, “Scan2cad: Learning cad model alignment in rgb-d scans,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 2609–2618.

[27] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in 2023 IEEE/CVF international conference on computer vision (ICCV), 2023, pp. 3992–4003.

[28] X. Yu, Y. Xie, Y. Liu, H. Lu, R. Xiong, Y. Liao, and Y. Wang, “Leverage cross-attention for end-to-end open-vocabulary panoptic reconstruction,” arXiv preprint arXiv:2501.01119, 2025.