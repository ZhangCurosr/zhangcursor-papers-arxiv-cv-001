# DynOSeg: Dynamic Object Segmentation via Motion-Geometry Fusion

Xu Han, Yiduo Wang, Yiran Wang, Viorela Ila

Australian Centre for Robotics, School of Aerospace, Mechanical and Mechatronic Engineering, The University of Sydney, NSW, 2006, Australia

xhan0329@uni.sydney.edu.au,{yiduo.wang,yiran.wang,viorela.ila}@sydney.edu.au

## Abstract

Moving Object Segmentation (MOS) detects and separates dynamic objects from the static background in a video, which plays a fundamental role in dynamic-scene understanding. Most MOS methods rely on pure visual motion cues alone, such as optical flow or point tracks, which is inherently ambiguous due to the compounded camera and object motions as well as motion-induced appearance changes, e.g. shadows and reflections. Naive fu sion of additional information such as depth and semantics, however, cannot leverage their complementarity to the fullest extent. We therefore propose DynOSeg, a novel MOS approach that takes on a two-stage fusion structure. DynOSeg first combines a geometry-aware residual map and depth encoding into a geometric representation of motion, delineating camera motions and object dynamics. We then further incorporate semantic features to refine object understanding. We have additionally curated a diverse training corpus including CARLA-MOS, a new synthetic urban-driving dataset for MOS training. Our experiments show that DynOSeg consistently outperforms published state-of-the-art MOS methods across a diverse range of datasets. Our code and pretrained models can be found at https://github.com/ACFR-RPG/DynOSeg

## 1. Introduction

Detecting moving objects with vision alone is a fundamental problem in computer vision and remains an active area of interest for applications such as action recognition [27], Simultaneous Localization And Mapping (SLAM) in dynamic environments [32, 60], virtual reality [48] and autonomous driving [18]. We address Moving Object Segmentation (MOS), the segmentation of dynamic objects in a video. Unlike video object segmentation [58], which segments all visible objects, MOS needs to distinguish dynamic objects from the static background. MOS also differs from motion segmentation, which additionally captures background motion such as moving water and shadows.

![](images/fb09a1ab0cf536810485fdc94cd825f490a544a0b91a483531fc64252216ef60.jpg)  
(a) KITTI [13].

![](images/3fa15841dbe2b5b54666d1f2823e07cccbdd55d18f65116b1b64ee8809175b70.jpg)  
(b) KITTI [13].

![](images/b695a86e1015ccfdfba8bf289eaaafceb5187c9a04d8f15fc07c653a5d940397.jpg)  
(c) SegTrackV2 [29].

![](images/0c65a3cbd346c898d9ad3fcf268c895df10ce755c4da2114f514158f62e9f90b.jpg)  
(d) KITTI [13].  
Figure 1. Challenging scenes. Each panel shows three rows: input RGB, DynOSeg (ours), and SegAnyMo [19], with predicted masks overlaid in the second and third rows. (a–b) Large camera motion produces substantial background flow; in these examples, DynOSeg segments the moving object more accurately than SegAnyMo. (c–d) In the displayed scenes with multiple moving objects, DynOSeg recovers objects that SegAnyMo misses.

Most methods infer motion from optical flow or point tracks [19, 20, 30, 52]. However, these 2D visual motion cues are inherently ambiguous as they reflect the combined effects of camera and object motion, making it difficult to separate them using 2D motion cues alone. This ambiguity becomes more severe under large camera motion as the static background can exhibit stronger apparent motion than dynamic objects. For example, SegAnyMo [19] mistakes the road as dynamic in Fig. 1 in a highway scene. 2D appearance cues are also unreliable under illumination changes, shadows, reflections and occlusions. These limitations motivate the use of complementary geometry and semantic information. Previous works introduce depth and camera geometry to better reason about motion [16, 33], while recent methods use powerful pretrained DINO/SAM features to provide stronger semantic cues [19, 53]. Effective MOS therefore requires combining motion, geometry and semantics to distinguish true object motions and extract accurate object regions.

However, a naive fusion of depth and semantics does not fully exploit unique advantages of these different modalities[39]. Depth and visual features provide better geometric and semantic definitions of object boundaries compared to motion-based visual cues. For example, the ablation studies of SegAnyMo [19] demonstrate that how and where semantic features are fused with motion cues significantly affects MOS performance.

These observations motivate a modality-aware fusion strategy that preserves the complementary roles of motion, geometry and semantics. Therefore, we propose DynOSeg, a new MOS approach that addresses these challenges. First, DynOSeg uses consistent camera poses and depth together with optical flow to construct a residual map that suppresses camera-induced apparent motion. We then employ a two-stage fusion strategy: a Geometry-Aware Spatial Fusion (GSF) block integrates depth and residual encoded features, followed by a Global–Local Semantic-Aware Fusion (GLSF) block that incorporates semantics to refine object regions and boundaries. DynOSeg supports both video and streaming-frame inputs when depth and camera poses are provided by online pipelines.

In summary, our contributions are:

• Geometry-aware encoding. We address motion ambiguity by constructing residual map from camera poses, depth, and optical flow, and encoding depth and residual maps with specific encoders to capture geometric cues.

• Two-stage fusion strategy. Reliable MOS requires combining camera motion, 3D scene geometry, and semantic cues. Integrating these dependencies, we show that a principled two-stage fusion is able to effectively distinguish moving objects.

• Extensive results. We conduct extensive experiments and ablation studies to evaluate our method and its key components. We also introduce a new pipeline for generating synthetic MOS datasets with CARLA simulator [9].

## 2. Related Work

Visual motion cues, such as optical flow, scene flow, and point tracks, provide the primary evidence for segmenting moving objects in videos [19]. Three-dimensional scene geometry and semantic information provide complementary cues for accurate and temporally consistent MOS [16, 26].

Visual motion cues. Optical flow represents the apparent motion of image pixels between consecutive frames in a video, describing how points move across the image because of camera motion, object motion, or both. In scenarios where both camera and scene objects are moving, optical flow captures the difference in apparent motions between independently moving objects and the static background, which can then be leveraged to detect motion boundaries in the 2D image plane [3, 36] as well as infer underlying 3D motions [41]. Some tracking-based MOS methods similarly study the 2D motions between consecutive frames for a sparse collection of points [2, 8, 21], formulating the MOS problem as model-fitting and solving it via energy minimization. Recent progress in machine learning has led to approaches that employ CNNs [4, 12, 31] and transformer-based models [53] to infer motion from optical flow, or to enhance feature representation and model estimation for tracking-based MOS [44, 61].

3D geometry consistency. Although learning-based methods have improved MOS accuracy over classical formulations, reasoning only over consecutive frames provides limited 3D geometry and long-term motion consistency [19], which can lead to the static background being mistakenly classified as dynamic [12]. To mitigate this limitation in flow-based approaches, Wedel et al. [47] and Cao et al. [4] leveraged stereo vision, while Faisal et al. [12] incorporated multi-view geometry for 3D understanding. In comparison, multi-frame tracking-based methods study 2D point motion across more than two frames, producing sparse but temporally extended motion representations. Longer point tracks enable more coherent clustering [34] that is less susceptible to short-term inconsistencies, which in turn supports geometric model fitting [1, 56], subspace fitting [11, 38], and learned motion segmentation [20, 26].

Appearance and semantics. Visual motion cues alone may be insufficient to disentangle object dynamics from apparent motion induced by camera movement or depthdependent motion parallax. To address this ambiguity, Mohamed et al. [31] and Neoral et al. [33] integrate semantic information with motion backbones in CNN architectures to better separate independently moving objects from the background. Similarly, Homeyer [16] used a transformer to enable multi-modal fusion between motion and appearance streams for MOS. Fusing appearance and semantic information in MOS allows for better distinction between foreground and background [31] and clearer definition of object boundaries [33], even enabling the segmentation of objects whose motion is partial or non-uniform.

Recent concurrent work shows that 3D geometry and semantic cues provide complementary information for moving-object segmentation [15, 54]. In particular, Geo-Motion [15] combines latent 4D geometry from $\pi ^ { 3 }$ with motion cues, while GMOS [54] combines geometric features from $\pi ^ { 3 }$ with segmentation features from SAM2 using a single feature-fusion stage. Related studies in dynamicscene SLAM and reconstruction also demonstrate the value of reliable geometry for reasoning about camera and object motion [32, 45, 60]. For example, $\pi _ { \mathrm { m o s } } ^ { 3 } { \mathrm { - S L A M } }$ [60] uses DINOv2 [35] features to assist MOS and jointly performs camera pose estimation and scene reconstruction. ViPE [17], in contrast, estimates camera motion and scene geometry but does not address MOS. DynOSeg builds on these developments with geometry-aware encoders and two-stage fusion design that integrate camera poses and depth with DINOv2 RGB features, progressively leveraging complementary motion, geometric, and semantic cues for effective moving objects segmentation.

## 3. Method

Applications requiring MOS typically employ perception frontends that already estimate camera poses, dense depth and optical flow. DynOSeg is designed to directly leverage these readily available geometric and motion cues. In DynOSeg’s default setup, we use ViPE [17] for offline camera pose and depth estimation, and RAFT [43] for optical flow, but DynOSeg also supports other online inputs such as online a SLAM pipeline, as tested and discussed in Sec. 4.4.

DynOSeg first constructs a residual map from camera poses, depth and optical flow. The depth and residual maps are then processed by dedicated encoding blocks (Secs. 3.1 and 3.2) to produce their corresponding feature representations. RGB features are extracted in parallel using DINOv2 encoder [35], which is fully frozen during training and inference. Finally, DynOSeg integrates these representations through a two-stage fusion strategy that combines residual, depth and RGB features using the proposed GSF and GLSF blocks, described in detail in Secs. 3.3 and 3.4. The overall architecture is illustrated in Fig. 2.

## 3.1. Residual Map Encoding

Residual flow represents the discrepancy between the measured optical flow and the predicted one under the assumption of a fully static environment. Existing literature leverages it for 3D scene understanding and separating object motions from camera-induced ones [28, 42, 63]. DynOSeg also uses this residual flow formulation, and converts residual flow vectors into a scalar residual map as the motion cue input for our model in place of optical flows, to exploit information on 3D scene geometry and camera motion for MOS. Between the current frame t and a previous frame q, let $\mathbf { T } _ { q  t }$ define the relative camera transformation that maps camera coordinates from t to q. Given the depth map of the current frame $D _ { t }$ , the expected rigid flow for pixel p under a static-world assumption is:

$$
{ \displaystyle { \bf f } _ { \mathrm { c a m } } ( { \bf p } ) = \pi \Big ( { \bf T } _ { q  t } [ D _ { t } ( { \bf p } ) \pi ^ { - 1 } ( { \bf p } ) , 1 ] ^ { \top } \Big ) - { \bf p } } ,\tag{1}
$$

where $\pi$ and $\pi ^ { - 1 }$ denote projection and back-projection functions respectively. In our current implementation, $q =$ t − 1 so that DynOSeg studies the motion between consecutive frames, inline with optical flow. The scalar residual of p is:

$$
R _ { t } ( \mathbf { p } ) = \left. \mathbf { f } _ { \mathrm { o p t } } ( \mathbf { p } ) - \mathbf { f } _ { \mathrm { c a m } } ( \mathbf { p } ) \right. _ { 2 } .\tag{2}
$$

Thus, static regions are expected to have residuals with small or zero magnitudes, while large residuals indicate motions not caused by camera’s own movement.

However, residual flow magnitude varies with object speed, depth and viewing geometry, and may contain spurious responses from depth or pose errors. The magnitude alone is therefore insufficient for motion detection. Hence we encode the residual map rather than using its raw values directly. Let $\mathbf { R } \in \mathbb { R } ^ { H \times \mathbf { \bar { W } } }$ denote the residual map. We first instance-normalize R to obtain $\hat { \mathbf { R } } .$ This tensor is then processed by an encoding block composed of strided convolutions and Cross-Stage Partial (CSP) blocks [23], followed by Layer Normalization, as shown in Fig. 2. After that, a spatial attention layer [50] further emphasizes informative motion regions:

$$
\mathbf { F } _ { \mathrm { r e s } } = \mathcal { F } _ { \mathrm { s } } \left( \Phi _ { \mathrm { e n c } } ( \hat { \mathbf { R } } ) \right) ,\tag{3}
$$

where $\Phi _ { \mathrm { e n c } } ( \cdot )$ denotes the residual encoder and $\mathcal { F } _ { \mathrm { s } } ( \cdot )$ the spatial attention layer. The resulting $\mathbf { F } _ { \mathrm { r e s } }$ provides motion features for the subsequent geometry–semantic fusion.

## 3.2. Depth Encoding

Depth maps encode important scene geometry but are often corrupted by noise and missing measurements, as well as scale inconsistencies when estimated by perframe learning-based methods. We therefore introduce a Geometry-Aware Depth Encoder to represent local scene and surface geometry.

Given a raw depth map $\mathbf { D } \in \mathbb { R } ^ { H \times W }$ , we first perform normalization using the mean µ and variance $\sigma ^ { 2 } ~ [ 2 4 ]$ over valid depth values $\mathbf { \left( D \theta > \theta 0 \right) }$ , yielding a normalized depth map D<sup>ˆ</sup> . This operation reduces variation in depth ranges across scenes while preserving relative depth discontinuities. To further encode local geometric structure, we derive an image-space normal representation from D<sup>ˆ</sup> using differentiable Sobel operators [6]. Let $\nabla _ { x }$ and $\nabla _ { y }$ denote the horizontal and vertical depth gradients, respectively. The image-space normal map N is computed as:

$$
\begin{array} { r l } & { \mathbf { N } = \mathrm { C o n c a t } \left( \frac { - \nabla _ { x } \hat { \mathbf { D } } } { \| \mathbf { v } \| } , \frac { - \nabla _ { y } \hat { \mathbf { D } } } { \| \mathbf { v } \| } , \frac { 1 } { \| \mathbf { v } \| } \right) } \\ & { \| \mathbf { v } \| = \sqrt { ( \nabla _ { x } \hat { \mathbf { D } } ) ^ { 2 } + ( \nabla _ { y } \hat { \mathbf { D } } ) ^ { 2 } + 1 } . } \end{array}\tag{4}
$$

The normalized depth and image-space normals are concatenated to form a four-channel geometric tensor $\mathbf { I } _ { \mathrm { g e o } } =$ [D<sup>ˆ</sup> ; N]. This tensor is then processed by an encoding block, which shares the same architecture as the residual encoder but with independent parameters, to yield depth features $\mathbf { F } _ { d }$

## 3.3. Geometry-Aware Spatial Fusion (GSF)

To construct a structural representation, we design a GSF block that fuses depth cues with motion features, as shown in Fig. 2. Since both depth and residual maps encode geometric and motion information about the scene, we prioritize their early interaction to form a unified geometric prior, before incorporating semantic information.

![](images/033b23c9c2c4b28f488a835ce0c8f1cb13ab1460388a061c08f876c11b13c9c5.jpg)  
Figure 2. Overview of the proposed DynOSeg framework. DynOSeg uses depth, camera poses and optical flow to compute a residual map. Depth and residual maps are processed by dedicated encoders, while a frozen DINOv2 encoder [35] extracts high-level semantic features from the RGB image. The resulting multimodal features are fused through Geometry-Aware Spatial Fusion (GSF) and Global– Local Semantic-Aware Fusion (GLSF). Finally, a segmentation decoder processes the fused features to produce moving-object masks.

The depth and residual encoders in Secs. 3.1 and 3.2 produce the feature embeddings $\mathbf { F } _ { d }$ and $\mathbf { F } _ { \mathrm { r e s } } ,$ , respectively. We first project them to a shared dimension using 1×1 convolutions. To fuse these modalities, we concatenate them along the channel dimension and pass them through two Cross-Stage Partial (CSP) blocks [23]. This operation facilitates the fusion of geometry and motion. The resulting geometric feature, $\mathbf { F } _ { \mathrm { g e o } }$ , is formulated as:

$$
\mathbf { F } _ { \mathrm { g e o } } = \Phi _ { \mathrm { c s p } } \left( \operatorname { C o n c a t } ( \mathbf { F } _ { d } , \mathbf { F } _ { \mathrm { r e s } } ) \right) ,\tag{5}
$$

where $\Phi _ { \mathrm { c s p } } ( \cdot )$ denotes the CSP blocks.

## 3.4. Global–Local Semantic-Aware Fusion (GLSF)

Building on the geometric features described in Sec. 3.3, the GLSF block further integrates semantic information and captures cross-modal relationships between semantics and geometry.

DynOSeg first uses DINOv2 [35] to generate RGB features $\mathbf { F } _ { \mathrm { r g b } }$ , which are projected into the same latent space as $\mathbf { F } _ { \mathrm { g e o } }$ using a convolution layer and then refined via a separate Channel Attention gate [50] to highlight semantic regions of interest. In the same block, the geometric features $\mathbf { F } _ { \mathrm { g e o } }$ are refined via another Channel Attention gate [50] to emphasize informative channels that correlate with moving object boundaries. We then concatenate the refined RGB features with $\mathbf { F } _ { \mathrm { g e o } }$ . This process can be written as:

$$
\begin{array} { r } { \hat { \mathbf { F } } _ { \mathrm { r g b } } = \mathcal { F } _ { \mathrm { C A } } ( \mathcal { F } _ { \mathrm { C o n v } } ( \mathbf { F } _ { \mathrm { r g b } } ) ) , \quad \hat { \mathbf { F } } _ { \mathrm { g e o } } = \mathcal { F } _ { \mathrm { C A } } ( \mathbf { F } _ { \mathrm { g e o } } ) , } \end{array}\tag{6}
$$

where $\mathcal { F } _ { \mathrm { { C o n v } } } ( \cdot )$ denotes a convolutional projection layer and $\mathcal { F } _ { \mathrm { { C A } } } ( \cdot )$ represents the channel attention operation.

To overcome the limited receptive field of standard convolutions, we introduce a Transformer Layer [10] to model global context across the fused modalities of geometry and semantics. Following the transformer layer, we adopt the Spatial Pyramid Pooling-Fast (SPPF) module from YOLOv5 [23], combined with a CSP block [23] to aggregate multi-scale local context. Finally, a Spatial Attention mechanism [50] is integrated to generate the fused features $\mathbf { F _ { \mathrm { o u t } } }$

$$
\mathbf { F } _ { \mathrm { o u t } } = \mathcal { F } _ { \mathrm { s } } \left( \Phi _ { \mathrm { c t x } } \left( \mathcal { F } _ { \mathrm { t } } ( \mathrm { C o n c a t } ( \hat { \mathbf { F } } _ { \mathrm { r g b } } , \hat { \mathbf { F } } _ { \mathrm { g e o } } ) ) \right) \right) ,\tag{7}
$$

where $\mathcal { F } _ { \mathrm { t } } ( \cdot )$ denotes the Transformer Layer, $\Phi _ { \mathrm { c t x } } ( \cdot )$ represents the context aggregation layers (SPPF and CSP), and $\mathcal { F } _ { \mathrm { s } } ( \cdot )$ denotes the spatial attention.

Table 1. Summary of datasets used for training.
<table><tr><td>Dataset</td><td>Type</td><td>#Seq.</td><td>#Frames</td></tr><tr><td>Kubric [14]</td><td>Synthetic</td><td>6,000</td><td>138k</td></tr><tr><td>Dynamic Replica [25]</td><td>Synthetic</td><td>483</td><td>145k</td></tr><tr><td>OCLR-Syn [51]</td><td>Synthetic</td><td>4,600</td><td>138k</td></tr><tr><td>CARLA-MOS</td><td>Synthetic</td><td>1,000</td><td>300k</td></tr><tr><td>YTVOS-Moving [7]</td><td>Real-world</td><td>395</td><td>10.2k</td></tr><tr><td>YTVOS-Moving *</td><td>Real-world</td><td>342</td><td>9.5k</td></tr><tr><td>Total</td><td>一</td><td>12,820</td><td>740.7k</td></tr></table>

## 3.5. Loss Function

A lightweight decoder then uses the fused features $\mathbf { F _ { \mathrm { o u t } } }$ to generate a moving-object mask, as shown in Fig. 2. We train the model with a combination of cross-entropy, Dice [22], and IoU losses [59]:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c e } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { d i c e } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { i o u } } , } \end{array}\tag{8}
$$

where $\mathcal { L } _ { \mathrm { c e } }$ supervises static/dynamic classification, while ${ \mathcal { L } } _ { \mathrm { d i c e } }$ and $\mathcal { L } _ { \mathrm { i o u } }$ are applied to the dynamic-class probability map. We set $\lambda _ { 1 } = 0 . 5$ and $\lambda _ { 2 } = 0 . 3$ in all experiments.

## 4. Experiments

## 4.1. Datasets

DynOSeg is trained on data from six datasets, including our custom-curated CARLA-MOS dataset, which cover a range of scene types and motion patterns. Details are provided in Tab. 1. Few datasets are dedicated to MOS, as most video segmentation datasets target video object segmentation. Existing MOS training datasets are predominantly synthetic, which may limit their applicability to realworld scenes. For DynOSeg training, we therefore include real-world data drawn from YTVOS [55], which contributes two subsets, YTVOS-Moving<sup>∗</sup>, an additional manually curated subset containing only sequences with moving target objects, and YTVOS-Moving [7] as shown in Tab. 1.

Among the off-the-shelf datasets, Kubric [14] is a synthetic dataset featuring rigid 3D objects with physically plausible dynamics. OCLR-Syn [51] provides synthetic scenes with highly artificial objects and motions. Both Kubric and OCLR-Syn provide motion masks directly in their ground truth. Dynamic Replica [25], also synthetic, contains only articulated actors such as humans and animals. It only provides ground truth camera motion, depth and optical flow, hence we created motion masks for DynOSeg training using scene geometry. Using the CARLA simulator [9], we created CARLA-MOS, a synthetic urban-driving dataset curated for MOS, which we will release alongside DynOSeg. The details of the pipeline for constructing with CARLA simulator[9] are provided in supplementary materials.

## 4.2. Experimental Setup and Benchmarks

Unless otherwise stated in the runtime scaling study, DynOSeg uses an input resolution of 518 × 518 on a single NVIDIA RTX 4090. We use DINOv2 [35] (ViT-B/14, frozen throughout training), RAFT [43] (the FlyingThings3D-pretrained checkpoint) and ViPE [17] (default configuration) as off-the-shelf components without further fine-tuning. The whole architecture of DynOSeg contains 5.235M trainable parameters.

We optimize with Adam (initial learning rate $1 \times 1 0 ^ { - 4 }$ weight decay $1 \times 1 0 ^ { - 5 } )$ using batch size 4 for 10 epochs. Training pools the 740.7k frames across the six training datasets in Tab. 1 and samples per-frame uniformly without per-dataset reweighting, yielding roughly 185k optimizer steps per epoch and 1.85M in total.

Benchmarks and metrics. We evaluate our method across four representative MOS benchmarks with diverse real-world scenarios. Specifically, we use the DAVIS2017- M dataset, the standard 22-sequence moving-object subset of DAVIS [37] defined by [7], which contains fast motions and occlusions. SegTrackV2 [29] is selected as it provides outdoor videos with strong motion blur and unstructured natural objects. FBMS-59 [34] is a standard movingobject segmentation benchmark containing diverse realworld videos with both camera and object motion. For autonomous driving scenarios, we include the official DSEC-MOS test split [62], featuring densely annotated moving objects such as vehicles and pedestrians.

Local evaluation protocol. For all available methods in Tab. 2, predicted instances are merged into binary foreground masks and mapped to native ground-truth resolution. On the official splits, the same evaluation script from SegAnyMo [19] computes region similarity (J), contour similarity (F), and their arithmetic mean J&F [37].

## 4.3. Comparison Experiments

Reproduction protocol. We compare DynOSeg against the available published baselines in Tab. 2, including EM [30], OCLR-TTA [51], ABR [52], SegAnyMo [19], and Geo-Motion [15]. For completeness, we additionally report results against the recently released GMOS [54], which is currently available only as an arXiv preprint and is therefore not part of the required comparison set. Nevertheless, DynOSeg remains competitive with GMOS across the evaluated benchmarks and outperforms it on several datasets. For all methods, we use the officially released weights and code and evaluate predictions using the same evaluation protocol from SegAnyMo [19].

Standard scenes. We define $\Delta$ as the DynOSeg score minus the named comparator’s score. On DAVIS2017- M [7], DynOSeg leads in $\mathcal { T } \& \mathcal { F }$ at 90.1, ahead of second best GMOS-S at 89.1 $( \Delta = + 1 . 0 )$ . Its J of 89.6 also ranks first, above SegAnyMo at 88.0 $( \Delta = + 1 . 6 )$ and GMOS at

![](images/d203f9ba85f2d8572101d8860fdfa6cb648de36988581e981c3edf31fda0c214.jpg)  
Figure 3. Qualitative comparisons on selected sequences from the DAVIS2017-M, SegTrackV2, FBMS-59 and DSEC-MOS benchmarks including scenes with multiple moving objects.

Table 2. MOS benchmark evaluation. Baseline inference uses official weights and code; all methods use the same SegAnyMo evaluation script [19]. Bold/underline mark best/second-best scores.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Year</td><td colspan="3">DAVIS2017-M</td><td colspan="3">SegTrackV2</td><td colspan="3">FBMS-59</td><td colspan="3">DSEC-MOS</td></tr><tr><td>J&amp;F↑</td><td>J↑</td><td>F↑</td><td>J&amp;F↑</td><td>J↑</td><td>F↑</td><td>J&amp;F↑</td><td>J↑</td><td>F↑</td><td>J&amp;F↑</td><td>J↑</td><td>F↑</td></tr><tr><td>EM [30]</td><td>TPAMI&#x27;23</td><td>62.8</td><td>61.6</td><td>64.0</td><td>54.9</td><td>51.5</td><td>58.3</td><td>58.9</td><td>58.4</td><td>59.4</td><td>5.8</td><td>4.5</td><td>7.0</td></tr><tr><td>OCLR-TTA [51]</td><td>NeurIPS&#x27;22</td><td>69.4</td><td>69.9</td><td>69.0</td><td>63.4</td><td>61.3</td><td>65.6</td><td>63.7</td><td>63.2</td><td>64.3</td><td>6.5</td><td>4.8</td><td>8.1</td></tr><tr><td>ABR [52]</td><td>ECCV’24</td><td>72.3</td><td>72.7</td><td>72.0</td><td>72.4</td><td>71.1</td><td>73.7</td><td>75.0</td><td>75.7</td><td>74.3</td><td>6.3</td><td>3.5</td><td>9.2</td></tr><tr><td>SegAnyMo [19]</td><td>CVPR&#x27;25</td><td>88.5</td><td>88.0</td><td>88.9</td><td>76.3</td><td>73.5</td><td>79.1</td><td>72.6</td><td>69.4</td><td>75.9</td><td>22.6</td><td>19.9</td><td>25.2</td></tr><tr><td>GeoMotion [15]</td><td>CVPR&#x27;26</td><td>81.2</td><td>81.1</td><td>81.3</td><td>75.3</td><td>70.9</td><td>79.8</td><td>78.8</td><td>76.7</td><td>80.9</td><td>20.5</td><td>17.3</td><td>23.8</td></tr><tr><td>GMOS [54]</td><td>arXiv&#x27;26</td><td>88.9</td><td>88.1</td><td>89.8</td><td>81.7</td><td>78.3</td><td>85.0</td><td>92.1</td><td>91.0</td><td>93.2</td><td>72.8</td><td>68.3</td><td>77.2</td></tr><tr><td>GMOS-S [54]</td><td>arXiv&#x27;26</td><td>89.1</td><td>87.6</td><td>90.7</td><td>78.0</td><td>74.6</td><td>81.4</td><td>85.1</td><td>83.7</td><td>86.5</td><td>59.3</td><td>55.3</td><td>63.2</td></tr><tr><td>DynOSeg (Ours)</td><td></td><td>90.1</td><td>89.6</td><td>90.6</td><td>84.2</td><td>81.7</td><td>86.6</td><td>84.6</td><td>82.6</td><td>86.6</td><td>73.0</td><td>64.4</td><td>81.6</td></tr></table>

88.1 $( \Delta = + 1 . 5 )$ . Its $\mathcal { F }$ of 90.6 ranks second to GMOS-S at $9 0 . 7 \left( \Delta = - 0 . 1 \right)$

Challenging scenes. On SegTrackV2 [29], which includes scenes with multiple dynamic objects, DynOSeg obtains the highest $\mathcal { T } \& \mathcal { F }$ score of 84.2, exceeding GMOS at 81.7 (∆ = +2.5 points). It also ranks first in J and ${ \mathcal F } ,$ with gains of 3.4 and 1.6 points over GMOS, respectively. On DSEC-MOS [62], DynOSeg obtains the highest $\mathcal { T } \& \mathcal { F }$ score of 73.0, exceeding GMOS at 72.8 (∆ = +0.2 points).

DynOSeg has a lower J score than GMOS (64.4 versus 68.3) but a higher F score (81.6 versus 77.2). Baselines are evaluated off the shelf from their released checkpoints, whereas DynOSeg is trained on our curated pool. Even without CARLA-MOS, DynOSeg reaches 65.7 on DSEC-MOS, still far above the best published baseline at 22.6, so the margin is not explained by in-domain training data.

The DSEC-MOS results indicate that the improvement in the averaged metric is marginal and is primarily associated with contour accuracy rather than region similarity. Figures 3 and 1 provide qualitative examples of DynOSeg predictions in scenes with multiple moving objects. In the selected example in Fig. 1, SegAnyMo’s point-track stage does not fully separate overlapping objects with similar motion, and its subsequent SAM 2 [40] stage does not recover the occluded object as dynamic.

Table 3. CARLA-MOS training-data ablation. Diff. denotes the signed change in $\mathcal { I } \& \mathcal { F }$ points relative to the w/ CARLA-MOS setting; negative values indicate a performance drop.
<table><tr><td rowspan="2">Training setting</td><td colspan="2">DAVIS2017-M</td><td colspan="2">DSEC-MOS</td></tr><tr><td> $\mathcal { T } \& \mathcal { F } \uparrow$ </td><td> $D i f f .$ </td><td> $\mathcal { T } \& \mathcal { F } \uparrow$ </td><td>Diff.</td></tr><tr><td>w/ CARLA-MOS</td><td>90.1</td><td>0.0</td><td>73.0</td><td>0.0</td></tr><tr><td>w/o CARLA-MOS</td><td>87.5</td><td>-2.6</td><td>65.7</td><td>-7.3</td></tr></table>

## 4.4. Ablation Study

In this section, We reported the ablation studies on CARLA-MOS training data, motion representations, input modalities and depth sources, geometry encoders, fusion structures, and different geometry providers.

Contribution of CARLA-MOS. Tab. 3 studies the contribution of CARLA-MOS by removing it from the full training pool while keeping all other datasets unchanged. Removing it reduces $\mathcal { T } \& \mathcal { F }$ by 2.6 points on DAVIS2017- M and 7.3 points on DSEC-MOS, showing that CARLA-MOS provides useful training data for motion segmentation. We attribute this improvement to two main factors. First, CARLA-MOS contains many scenes with multiple moving objects, such as vehicles and pedestrians, which helps the model learn to segment several independently moving objects at the same time. Second, it provides many examples with camera motion along the viewing direction, which is common in real driving scenarios when the camera moves forward or backward with the vehicle. Such motion produces appearance changes that are less common in generic video datasets but are important for autonomous driving. The gain on DSEC-MOS suggests that these driving specific motion patterns help in real road scenes.

Motion representations. Rows (A) and (F) of Tab. 4 isolate the effect of the motion representation, with RGB and temporally consistent ViPE depth kept unchanged. Replacing optical flow with the residual map improves $\mathcal { T } \& \mathcal { F }$ from 86.0 to 90.1 on DAVIS2017-M and from 63.6 to 73.0 on DSEC-MOS, giving gains of 4.1 and 9.4 points, respectively. This shows that the residual map provides a cleaner cue for object motion by reducing motion caused by the camera. The larger improvement on DSEC-MOS further highlights its benefit in driving scenes, where strong camera motion is common.

Incremental ablation study of DynOSeg. Rows (B)– (F) of Tab. 4 study the contribution of different input modalities. The residual map alone (B) gives limited performance, while adding either ViPE depth (C) or RGB (D) improves both benchmarks, showing that geometry and appearance provide complementary cues. In contrast, using per-frame Depth Anything V2 depth (E) degrades performance compared with (D). We attribute this partly to the scale mismatch between depth and pose translation, as residual-map computation requires depth to be consistent with the metric scale of the pose translation. This change further affects both residual-map construction and the depth input to the depth encoder. Combining the residual map, RGB, and temporally consistent ViPE depth (F) achieves the best results, reaching 90.1/73.0 J&F on DAVIS2017-M/DSEC-MOS.

Table 4. Motion-representation and input-modality ablation. † denotes per-frame depth from Depth Anything V2 [57], while ‡ denotes temporally consistent depth from ViPE [17].
<table><tr><td>Exp.</td><td>Setting</td><td>DAVIS2017-M</td><td>DSEC-MOS</td></tr><tr><td>(A)</td><td></td><td>J&amp;F↑</td><td>J&amp;F↑</td></tr><tr><td></td><td>Optical Flow + RGB + Depth‡</td><td>86.0</td><td>63.6</td></tr><tr><td>(B) (C)</td><td>Residual Map only</td><td>55.4 79.8</td><td>61.2</td></tr><tr><td>(D)</td><td>(B) + Depth‡</td><td>83.5</td><td>64.1 65.6</td></tr><tr><td>(E)</td><td> $\mathbf { \left( B \right) } + \mathbf { R } \mathbf { G } \mathbf { B }$   $\mathrm { ( D ) + D e p t h ^ { \dagger } }$ </td><td>78.6</td><td>63.3</td></tr><tr><td>(F)</td><td> $\mathrm { ( D ) + D e p t h ^ { \ddag } \ ( D y n O S e g ) }$ </td><td>90.1</td><td>73.0</td></tr></table>

Table 5. Encoder ablation. Listed branches use PlainCNN; others retain their original encoders.
<table><tr><td rowspan="2">PlainCNN replacement</td><td>DAVIS2017-M</td><td>DSEC-MOS</td></tr><tr><td>J&amp;F↑</td><td>J&amp;F↑</td></tr><tr><td>Residual encoder only</td><td>86.0</td><td>65.4</td></tr><tr><td>Depth encoder only</td><td>84.8</td><td>64.9</td></tr><tr><td>Both encoders</td><td>85.4</td><td>65.1</td></tr><tr><td>None (DynOSeg)</td><td>90.1</td><td>73.0</td></tr></table>

Encoder ablation. Table 5 compares individual and joint PlainCNN replacements, with inputs, outputs, other modules, and training settings fixed. PlainCNN uses four Conv–BN–SiLU layers and 0.963M/0.961M depth/residual parameters versus 0.948M/0.946M; details are reported in supplementary materials. Residual-only and depth-only replacements reduce $\mathcal { T } \& \mathcal { F }$ by 4.1/7.6 and 5.3/8.1 points, compared with DynOSeg on DAVIS2017-M/DSEC-MOS.

Tab. 6 contains five fusion designs. Concat directly concatenates the residual, depth, and RGB features. Only GSF concatenates the RGB and geometry features from GSF, while Only GLSF concatenates the residual and depth features then fuses with RGB features with GLSF. Swapped order uses GLSF for geometry fusion and GSF for RGBgeometry fusion. Two-stage fusion performs best, outperforming the strongest alternative, Only GLSF, by 5.6 points on DAVIS2017-M and 2.4 points on DSEC-MOS. These results favor the two-stage design under the tested settings.

Table 6. Ablation study on fusion structures. All variants use RGB, depth, and residual flow inputs.
<table><tr><td rowspan="2">Exp.</td><td rowspan="2">Fusion Structure</td><td>DAVIS2017-M</td><td>DSEC-MOS</td></tr><tr><td> $\mathcal { I } \& \mathcal { F } \uparrow$ </td><td> $\mathcal { T } \& \mathcal { F } \uparrow$ </td></tr><tr><td>(A)</td><td>Two-stage</td><td>90.1</td><td>73.0</td></tr><tr><td>(B)</td><td>Concat</td><td>83.6</td><td>69.0</td></tr><tr><td>(C)</td><td>Only GSF</td><td>82.8</td><td>69.0</td></tr><tr><td>(D)</td><td>Only GLSF</td><td>84.5</td><td>70.6</td></tr><tr><td>(E)</td><td>Swapped order</td><td>83.7</td><td>69.5</td></tr></table>

Table 7. Prediction consistency across geometry sources on KITTI-MOS. $\mathcal { C } _ { \mathrm { I o U } }$ denotes the mean pixel-micro IoU between each configuration and the other two configurations.
<table><tr><td>Pose source</td><td>Depth source</td><td>Setting</td><td> $\mathcal { C } _ { \mathrm { I o U } } \uparrow$ </td></tr><tr><td>KITTI OXTS (GT)</td><td>FoundationStereo</td><td>Reference</td><td>89.0</td></tr><tr><td>ViPE</td><td>ViPE</td><td>Offline</td><td>84.0</td></tr><tr><td>DynOSAM</td><td>DynOSAM</td><td>Online</td><td>89.0</td></tr></table>

Plug-and-Play Geometry Sources on KITTI-MOS. We evaluate whether DynOSeg can use depth and camera poses from different sources on KITTI-MOS. KITTI-MOS is constructed from KITTI Raw sequences [13] containing moving objects. The details are provided in supplementary materials. Table 7 compares three geometry settings. The reference setting combines GT camera poses from KITTI OXTS with dense stereo depth estimated by Foundation-Stereo [49]. ViPE [17] estimates both camera poses and depth offline, whereas DynOSAM [32] provides both online. All settings use the same RGB frames, RAFT optical flow, evaluation masks, and DynOSeg weights. The three geometry sources produce highly similar predictions: $\mathcal { C } _ { \mathrm { I o U } }$ ranges from 84.0 to 89.0, with an overall mean pairwise IoU of 87.3. These results show that DynOSeg can use reference, offline, or online geometry without retraining.

Online inference. The interface uses depth and camera poses from the current and previous frames, together with two frame optical flow, to construct the residual map for the current frame. DynOSeg then takes the current RGB, depth, and residual map as input. By leveraging a causal geometry provider such as DynOSAM or LinStereo [46], our framework can be integrated into an online pipeline. End to end real time performance also depends on the latency of geometry and optical flow.

Runtime and Resolution Scaling. We further evaluate the computational cost of DynOSeg and its main input components on an RTX 4090 with batch size 1. At

Table 8. Runtime scaling on an RTX 4090 (batch size 1), in ms/frame. DynOSeg wall-clock timing includes data transfer and post-processing; GPU timings are independent compute-only microbenchmarks and are not additive. RAFT and residual construction are not included in DynOSeg timing.
<table><tr><td>Input resolution  $( \bar { W } \times H )$ </td><td colspan="2">DynOSeg</td><td>RAFT GPU</td><td>Residual GPU</td></tr><tr><td></td><td>Wall-clock</td><td>GPU</td><td></td><td></td></tr><tr><td> $5 1 8 \times 5 1 8$ </td><td>20.13</td><td>19.00</td><td>25.93</td><td>0.43</td></tr><tr><td> $6 3 0 \times 6 3 0$   $7 2 8 \times 7 2 8$ </td><td>32.95 46.93</td><td>32.28 47.15</td><td>31.99 44.89</td><td>0.44</td></tr><tr><td> $1 3 4 4 \times 7 5 6$ </td><td>102.43</td><td>106.20</td><td>84.57</td><td>0.46 0.58</td></tr><tr><td> $1 7 9 2 \times 1 0 0 8$ </td><td>252.69</td><td>270.65</td><td>173.47</td><td>1.02</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

$5 1 8 \ \times \ 5 1 8 ,$ DynOSeg takes 20.13 ms per frame when depth and residual maps are precomputed; this latency includes DINOv2, data transfer, feature fusion, decoding, and mask post-processing, but excludes RAFT, geometry estimation, residual construction, and disk I/O. As shown in Table 8, DynOSeg latency increases from 20.13 ms at $5 1 8 \times 5 1 8$ to 252.69 ms at $1 7 9 2 \times 1 0 0 8 ,$ while RAFT requires 25.93–173.47 ms and residual construction adds only 0.43–1.02 ms across the tested resolutions. This indicates that residual-map construction adds little overhead compared with optical flow and segmentation. Since the GPU times are measured independently, they should not be summed to estimate end-to-end latency and are intended only to characterize the cost of individual components. In real applications, DynOSeg works with various sources of optical flow, depth and camera poses. All tested resolutions are compatible with DINOv2 patch size of 14, and rectangular inputs use 16:9 aspect ratio. A more detailed analysis is provided in supplementary materials.

## 5. Conclusion and Future Work

In this paper, we propose DynOSeg, a MOS framework that explicitly models temporally consistent camera motion and depth to segment moving objects in videos. DynOSeg addresses the challenge of separating independent object motion from camera-induced motion through a geometry-aware residual map and combines camera motion, depth, optical flow, and appearance features through a two-stage fusion strategy. In the local benchmark evaluation, DynOSeg ranks first on DAVIS2017-M, SegTrackV2, and DSEC-MOS and third on FBMS-59 in J&F, with a 0.2-point margin over GMOS on DSEC-MOS. The ablation studies support the use of the residual map and temporally consistent depth in the tested DynOSeg configuration. One current limitation of DynOSeg is that it does not explicitly distinguish between individual movingobject instances, which we plan to address by incorporating instance-level awareness in future work.

## Acknowledgements

This research was funded by the Australian Government through the Australian Research Council (ARC) Research Hub for Intelligent Robotic Systems for Real-Time Asset Management (IH210100030).

## References

[1] Federica Arrigoni, Luca Magri, and Tomas Pajdla. On the usage of the trifocal tensor in motion segmentation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 514–530. Springer International Publishing, 2020. 2

[2] Daniel Bar ´ ath and Jiri Matas. Progressive-x: Efficient, any-´ time, multi-model fitting algorithm. In ICCV, pages 3779– 3787, 2019. 2

[3] Pia Bideau and Erik Learned-Miller. It’s moving! a probabilistic model for causal motion segmentation in moving camera videos. In Proceedings of the European Conference on Computer Vision (ECCV), pages 433–449. Springer International Publishing, 2016. 2

[4] Zhe Cao, Abhishek Kar, Christian Hane, and Jitendra Malik.¨ Learning independent object motion from unlabelled stereoscopic videos. In CVPR, pages 5587–5596, 2019. 2

[5] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Radle, Triantafyllos Afouras, Effrosyni Mavroudi, Kather-¨ ine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollar, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and´ Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2025. 16

[6] Kenneth R Castleman. Digital image processing. Prentice Hall Professional Technical Reference, 1979. 3

[7] Achal Dave, Pavel Tokmakov, and Deva Ramanan. Towards segmenting anything that moves. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (IC-CVW), pages 1493–1502. IEEE, 2019. 5

[8] Andrew Delong, Anton Osokin, Hossam N. Isack, and Yuri Boykov. Fast approximate energy minimization with label costs. In CVPR, pages 2173–2180, 2010. 2

[9] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An open urban driving simulator. In Conference on Robot Learning, pages 1–16, 2017. 2, 5, 15, 18

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 4

[11] Ehsan Elhamifar and Rene Vidal. Sparse subspace cluster-´

ing: Algorithm, theory, and applications. IEEE TPAMI, 35 (11):2765–2781, 2013. 2

[12] Muhammad Faisal, Ijaz Akhter, Mohsen Ali, and Richard Hartley. Epo-net: Exploiting geometric constraints on dense trajectories for motion saliency. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1873–1882, 2019. 2

[13] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. Int. J. Robotics Res., 32(11):1231–1237, 2013. 1, 8, 16

[14] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapra gasam, Florian Golemo, Charles Herrmann, et al. Kubric: a scalable dataset generator. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3739–3751. IEEE, 2022. 5

[15] Xiankang He, Peile Lin, Ying Cui, Dongyan Guo, Chunhua Shen, and Xiaoqin Zhang. Geomotion: Rethinking motion segmentation via latent 4d geometry. arXiv preprint arXiv:2602.21810, 2026. 2, 5, 6

[16] Christian Homeyer and Christoph Schnorr. On moving ob-¨ ject segmentation from monocular video with transformers. In 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 880–891. IEEE, 2023. 1, 2

[17] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv.org, 2025. 2, 3, 5, 7, 8, 17

[18] Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Shanghang Zhang. Sgaussian: Self-supervised street gaussians for autonomous driving. CoRR, 2024. 1

[19] Nan Huang, Wenzhao Zheng, Chenfeng Xu, Kurt Keutzer, Shanghang Zhang, Angjoo Kanazawa, and Qianqian Wang. Segment any motion in videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3406–3416. IEEE, 2025. 1, 2, 5, 6

[20] Yuxiang Huang, Yuhao Chen, and John Zelek. Zero-shot monocular motion segmentation in the wild by combining deep learning with geometric motion model fusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 2733– 2743. IEEE, 2024. 1, 2

[21] Hossam Isack and Yuri Boykov. Energy-based geometric multi-model fitting. International Journal of Computer Vi sion, 97(2):123–147, 2012. 2

[22] Shruti Jadon. A survey of loss functions for semantic segmentation. In IEEE Symposium on Computational Intelli gence in Bioinformatics and Computational Biology, pages 1–7. IEEE, IEEE, 2020. 5

[23] Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLO, 2023. 3, 4

[24] Jinyoung Jun, Jae-Han Lee, Chul Lee, and Chang-Su Kim. Depth map decomposition for monocular depth estimation.

In Proceedings of the European Conference on Computer Vision (ECCV), pages 18–34. Springer, Springer Nature Switzerland, 2022. 3

[25] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13229– 13239. IEEE, 2023. 5

[26] Laurynas Karazija, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. Learning segmentation from point trajectories. Advances in Neural Information Processing Systems, 37:112573–112597, 2024. 2

[27] Yu Kong and Yun Fu. Human action recognition and prediction: A survey. IJCV, 130(5):1366–1401, 2022. 1

[28] Seokju Lee, Sunghoon Im, Stephen Lin, and In So Kweon. Learning residual flow as dynamic motion from stereo videos. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 1180–1186, 2019. 3

[29] Fuxin Li, Taeyoung Kim, Ahmad Humayun, David Tsai, and James M Rehg. Video segmentation by tracking many figureground segments. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 2192– 2199. IEEE, 2013. 1, 5, 6

[30] Etienne Meunier, Ana¨ıs Badoual, and Patrick Bouthemy. Em-driven unsupervised learning for efficient motion segmentation. IEEE TPAMI, 45(4):4462–4473, 2023. 1, 5, 6

[31] Eslam Mohamed, Mahmoud Ewaisha, M. Siam, Hazem Rashed, S. Yogamani, Waleed Hamdy, Mohamed El-Dakdouky, and Ahmad El Sallab. Monocular instance motion segmentation for autonomous driving: Kitti instancemotseg dataset and multi-task baseline. In 2021 IEEE Intelligent Vehicles Symposium (IV), pages 114–121. IEEE, 2020. 2

[32] Jesse Morris, Yiduo Wang, Mikolaj Kliniewski, and Viorela Ila. Dynosam: Open-source smoothing and mapping framework for dynamic slam. IEEE Transactions on robotics, 2025. 1, 2, 8, 17

[33] Michal Neoral, Jan Sochman, and Jiri Matas. Monocular arbitrary moving object discovery and segmentation. In BMVC. British Machine Vision Association, 2021. 1, 2

[34] Peter Ochs, Jitendra Malik, and Thomas Brox. Segmentation of moving objects by long term video analysis. IEEE TPAMI, 36(6):1187–1200, 2014. 2, 5

[35] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy V.´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2023. 2, 3, 4, 5

[36] Anestis Papazoglou and Vittorio Ferrari. Fast object segmentation in unconstrained video. In ICCV, pages 1777–1784, 2013. 2

[37] Federico Perazzi, J. Pont-Tuset, B. McWilliams, L. Gool, M. Gross, and A. Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, pages 724–732. IEEE, 2016. 5

[38] Shankar Rao, Roberto Tron, Rene Vidal, and Yi Ma. Motion segmentation in the presence of outlying, incomplete, or corrupted trajectories. IEEE TPAMI, 32(10):1832–1845, 2010. 2

[39] Hazem Rashed, Ahmad El Sallab, Senthil Yogamani, and Mohamed ElHelw. Motion and depth augmented semantic segmentation for autonomous navigation. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 364–370. IEEE, 2019. 2

[40] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya K. Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chlo¨ e Rolland, Laura Gustafson, et al. Sam 2:´ Segment anything in images and videos. ICCV, 2024. 7

[41] Hicham Sekkati and Amar Mitiche. A variational method for the recovery of dense 3d structure from motion. Robotics Auton. Syst., 55(7):597–607, 2007. 2

[42] Tatsunori Taniai, Sudipta N. Sinha, and Yoichi Sato. Fast multi-frame stereo scene flow with motion segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 3

[43] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In ECCV, pages 402–419. Springer International Publishing, 2020. 3, 5

[44] Pavel Tokmakov, Cordelia Schmid, and Karteek Alahari. Learning to segment moving objects. International Journal ofComputer Vision, 127(3):282–301, 2019. 2

[45] Yiduo Wang, Jesse Morris, Lan Wu, Teresa Vidal-Calleja, and Viorela Ila. Dynorecon: Dynamic object reconstruction for navigation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 3305–3311. IEEE, 2025. 2

[46] Yiran Wang, Oliver Turner, and Viorela Ila. Linstereo: linear-complexity global attention for multi-scale iterative stereo matching. In European Conference on Computer Vi sion, pages 1–20. Springer, 2026. 8

[47] A. Wedel, Annemarie Meißner, C. Rabe, Uwe Franke, and D. Cremers. Detection and segmentation of independently moving objects from dense scene flow. In Energy Minimiza tion Methods in Computer Vision and Pattern Recognition, pages 14–27. Springer, Springer Berlin Heidelberg, 2009. 2

[48] Xiaobao Wei, Peng Chen, Ming Lu, Hui Chen, and Feng Tian. Graphavatar: Compact head avatars with gnngenerated 3d gaussians. In AAAI Conference on Artificial Intelligence, pages 8295–8303, 2024. 1

[49] Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, and Stan Birchfield. Foundationstereo: Zeroshot stereo matching. CVPR, 2025. 8, 17

[50] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. Cbam: Convolutional block attention module. In ECCV, pages 3–19. Springer International Publishing, 2018. 3, 4

[51] Junyu Xie, Weidi Xie, and Andrew Zisserman. Segmenting moving objects via an object-centric layered representation. In NeurIPS, pages 28023–28036. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2022. 5, 6

[52] Junyu Xie, Weidi Xie, and Andrew Zisserman. Appearancebased refinement for object-centric motion segmentation. In ECCV, 2023. 1, 5, 6

[53] Junyu Xie, Charig Yang, Weidi Xie, and Andrew Zisserman. Moving object segmentation: All you need is sam (and flow). In ACCV, 2024. 1, 2

[54] Junyu Xie, Tengda Han, Weidi Xie, and Andrew Zisserman. Gmos: Grounding moving object segmentation in 3d space and time. arXiv preprint arXiv:2605.30352, 2026. 2, 5, 6

[55] Ning Xu, Linjie Yang, Yuchen Fan, Dingcheng Yue, Yuchen Liang, Jianchao Yang, and Thomas S. Huang. Youtubevos: A large-scale video object segmentation benchmark. arXiv.org, abs/1809.03327, 2018. 5

[56] Xun Xu, Loong Fah Cheong, and Zhuwen Li. Motion segmentation by exploiting complementary geometric models. In CVPR, pages 2859–2867. IEEE, 2018. 2

[57] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Sys tems, 37:21875–21911, 2024. 7

[58] Rui Yao, Guosheng Lin, Shixiong Xia, Jiaqi Zhao, and Yong Zhou. Video object segmentation and tracking: A survey. ACM Transactions on Intelligent Systems and Technology (TIST), 11(4):1–47, 2020. 1

[59] Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, and Thomas Huang. Unitbox: An advanced object detection network. In ACM Multimedia, pages 516–520. IEEE, 2016. 5

[60] Xingguang Zhong, Liren Jin, Marija Popovic, Jens Behley,´ and Cyrill Stachniss. Dynamic visual slam using a general 3d prior. arXiv preprint, 2025. 1, 2

[61] Tianfei Zhou, Shunzhou Wang, Yi Zhou, Yazhou Yao, Jianwu Li, and Ling Shao. Motion-attentive transition for zero-shot video object segmentation. In AAAI, pages 13066– 13073, 2020. 2

[62] Zhuyun Zhou, Zongwei Wu, Remi Boutteau, Fan Yang, and´ D. Ginhac. Event-free moving object segmentation from moving ego vehicle. pages 8960–8965. IEEE, 2024. 5, 6

[63] Yuliang Zou, Zelun Luo, and Jia-Bin Huang. Df-net: Unsupervised joint learning of depth and flow using cross-task consistency. In Proceedings of the European Conference on Computer Vision (ECCV), 2018. 3

# Supplementary Materials

# DynOSeg: Dynamic Object Segmentation via Motion-Geometry Fusion

Xu Han, Yiduo Wang, Yiran Wang, Viorela Ila

Australian Centre for Robotics, School of Aerospace, Mechanical and Mechatronic Engineering, The University of Sydney, NSW, 2006, Australia

xhan0329@uni.sydney.edu.au,{yiduo.wang,yiran.wang,viorela.ila}@sydney.edu.au

Table A.1. PlainCNN encoder architecture. $C _ { \mathrm { i n } } = 4$ for depth plus normals and $C _ { \mathrm { i n } } = 1$ for the normalized residual map. Only the numbered convolutional blocks count toward the four-layer depth.
<table><tr><td>Conv.</td><td>Operation</td><td>Channels (in/out)</td><td>Kernel / stride</td></tr><tr><td>1</td><td>Conv-BN-SiLU</td><td> $C _ { \mathrm { i n } } / 6 4$ </td><td> $3 \times 3 / 2$ </td></tr><tr><td>2</td><td>Conv-BN-SiLU</td><td>64/128</td><td> $3 \times 3 / 1$ </td></tr><tr><td>一</td><td>MaxPool</td><td>128/128</td><td> $2 \times 2 / 2$ </td></tr><tr><td>3</td><td>Conv-BN-SiLU</td><td>128/256</td><td>3× 3/1</td></tr><tr><td>4</td><td>Conv-BN-SiLU</td><td>256/256</td><td>3 × 3 / 1</td></tr><tr><td></td><td>LayerNorm2d</td><td>256/256</td><td>一</td></tr><tr><td></td><td>Dropout  $( p = 0 . 1 )$ </td><td>256/256</td><td>一</td></tr></table>

## A. Qualitative Results

DAVIS2017-M. Figure A.1 compares foreground masks produced by DynOSeg and the evaluated baselines on representative DAVIS2017-M frames. Green overlays denote the ground-truth or predicted moving-object regions. In these examples, DynOSeg preserves object extent and boundary detail across articulated motion, thin structures, scale changes, and cluttered backgrounds, whereas competing methods sometimes omit object parts or include nearby background regions.

SegTrackV2. Figure A.2 shows examples containing multiple targets, articulated motion, motion blur, and large appearance changes. DynOSeg retains separate moving regions and follows the visible object silhouettes in these selected frames. Several baselines either miss a target or part of its body, while others introduce foreground responses in nearby static regions.

FBMS-59. Figure A.3 includes moving people, a large vehicle, and a small animal under substantial scale and appearance variation. DynOSeg produces compact masks that remain aligned with the moving targets in these examples. By comparison, some methods miss small targets or absorb adjacent static background into the prediction.

DSEC-MOS. Figure A.4 presents urban-driving scenes with strong ego-motion, low illumination, and small or distant road users. DynOSeg identifies multiple moving vehicles and pedestrians while limiting responses on the road and surrounding structures in the displayed frames. Several baselines omit distant targets or produce large falsepositive regions in the static background. Overall, these selected visualizations complement the quantitative results in Table 2 of the main paper; they are not intended to establish a dataset-wide ranking on their own.

## A.1. Ablation Visualizations

Residual map versus optical flow. Figure A.5 visualizes the comparison between rows (A) and (F) of Table 4 in the main paper. Both variants use RGB and temporally consistent ViPE depth; only the motion representation changes. In these examples, the residual-map variant better preserves complete foreground regions and small or distant moving objects, whereas raw optical flow is more affected by apparent motion induced by the camera.

Different input combinations. Figure A.6 complements the input-modality ablation in Table 4 of the main paper. Using the residual map alone (M1) can miss substantial object regions, while adding ViPE depth (M2) or RGB (M3) improves object extent and boundary coherence in the displayed frames. M4 shows that per-frame Depth Anything V2 depth does not provide the same benefit as temporally consistent ViPE depth. The full DynOSeg configuration combines the residual map, RGB, and ViPE depth and most closely follows the ground-truth masks in these selected examples.

Fusion structures. Figure A.7 visualizes the fusion ablation reported in Table 6 of the main paper with identical residual, depth, and RGB inputs. The alternative designs can omit object parts, introduce nearby background, or fragment small foreground regions in the displayed examples. The full two-stage GSF–GLSF design produces more coherent masks, illustrating how geometry–motion fusion followed by semantic refinement complements the quantitative results.

![](images/3e2c89c84c3914de0ddc29d5063ec89a60ba12fb632b0fbc684c00995722a87b.jpg)  
Figure A.1. Qualitative comparison on DAVIS2017-M. Columns show the ground truth and predictions from DynOSeg, GMOS, Geo Motion, SegAnyMo, ABR, OCLR-TTA, and EM. Green overlays indicate moving-object foreground regions.

GT  
EM  
![](images/8679af4d1b85f53c138ac6af933a53214e9d979ada6f6946f9348294c26a3179.jpg)  
Figure A.2. Qualitative comparison on SegTrackV2. Columns show the ground truth and predictions from DynOSeg, GMOS, GeoMo tion, SegAnyMo, ABR, OCLR-TTA, and EM. Green overlays indicate moving-object foreground regions.

The ablation figures show representative cases and are intended to explain characteristic differences between variants rather than establish a dataset-wide ranking independently of the quantitative tables.

## B. CNN Baseline for Geometry Encoders

Replacement protocol. Table 5 in the main paper separates four configurations: replacing only the residual encoder, replacing only the depth encoder, replacing both, and retaining unmodified DynOSeg. Each selected encoder is replaced with a separate PlainCNN branch; an unselected encoder retains its original design. The replacement does not add a CNN to the original encoders or replace the frozen RGB DINOv2 encoder. The depth branch retains the fourchannel depth-plus-normals input, and the residual branch retains its normalized single-channel residual-map input. Both branches output 256 channels at an output stride of 4. Replacing the depth encoder removes its C2f blocks; replacing the residual encoder removes its C2f blocks and spatial attention. GSF, GLSF, DINOv2, the decoder, loss, and training settings are unchanged.

EM  
GT  
![](images/52c3d50547e00b6fb6d30027bf6c8fbacfcb74ceddc1f514a4129461e50ea46e.jpg)  
Figure A.3. Qualitative comparison on FBMS-59. Columns show the ground truth and predictions from DynOSeg, GMOS, GeoMotion, SegAnyMo, ABR, OCLR-TTA, and EM. Green overlays indicate moving-object foreground regions.

GT  
DynOSeg  
GMOS  
SegAnyMo  
ABR  
OCLR-TTA  
EM  
![](images/3ee32a59586ff23c9af8de50722b8d5d522ae6e2d1177c095658fd1668dbdd12.jpg)  
Figure A.4. Qualitative comparison on DSEC-MOS. Columns show the ground truth and predictions from DynOSeg, GMOS, GeoMo tion, SegAnyMo, ABR, OCLR-TTA, and EM. Green overlays indicate moving-object foreground regions.

Architecture and layer counting. Table A.1 gives the complete layer sequence. Each convolution uses the same Conv–BN–SiLU implementation as the proposed model: convolution followed by batch normalization and the SiLU activation. The 2×2 max-pooling operation follows the second convolution, and the fourth convolution is followed by LayerNorm2d with 256 channels and Dropout with probability 0.1. The term four-layer CNN counts only the four convolutional layers; MaxPool, LayerNorm, Dropout, batch normalization, and activation operators are not counted as additional convolutional layers. This is a layer-count convention, not a claim that normalization layers have no learnable parameters.

Parameter matching and results. Table B.1 compares the encoder parameter counts. PlainCNN has 14,592 additional depth-encoder parameters (1.54%) and 14,494 additional residual-encoder parameters (1.53%), making the comparison closely, but not exactly, parameter-matched. These counts refer to the depth and residual encoders, not the complete DynOSeg system. Table 5 in the main paper reports 90.1/73.0 J&F for unmodified DynOSeg on DAVIS2017-M/DSEC-MOS. Defining ∆ as the replacement score minus this reference, replacing only the residual encoder yields 86.0/65.4 (∆ = −4.1/ − 7.6 points), while replacing only the depth encoder yields 84.8/64.9 (∆ = −5.3/−8.1 points). Replacing both yields 85.4/65.1 (∆ = −4.7/ − 7.9 points). Despite their slightly larger parameter budgets, all three PlainCNN variants score below

![](images/b7f0ac518c55f6d206e9e658cc00cd1c179b161ac6932c649ae0bf3350714c6b.jpg)  
Figure A.5. Residual-map versus optical-flow input. Rows show the ground truth, DynOSeg with the residual map, and an otherwise matched DynOSeg variant using raw optical flow. Columns contain examples from DAVIS2017-M and DSEC-MOS. Green overlays denote moving-object foreground regions.

![](images/bc507f7dd90097507007fb128b57e28c94f8ad82ea031a3da8ccadc2684d4d14.jpg)  
Figure A.6. Qualitative ablation of input modalities. M1 uses only the residual map; M2 uses the residual map and ViPE depth; M3 uses the residual map and RGB; and M4 uses the residual map, RGB, and per-frame Depth Anything V2 depth. DynOSeg uses the residual map, RGB, and temporally consistent ViPE depth. Rows show DSEC-MOS and DAVIS2017-M examples, and green overlays denote predicted or ground-truth foreground.

Table B.1. Geometry-encoder parameter comparison. Exact counts are reported; the relative difference uses the corresponding DynOSeg encoder as the denominator.
<table><tr><td>Encoder branch</td><td>DynOSeg</td><td>PlainCNN</td><td>Difference</td></tr><tr><td>Depth</td><td>948,096</td><td>962,688</td><td>+1.54%</td></tr><tr><td>Residual</td><td>946,466</td><td>960,960</td><td>+1.53%</td></tr></table>

DynOSeg, supporting the original design of each branch in these tested configurations. The joint replacement is 0.6/0.2 points above the depth-only replacement, so the decreases are not additive or monotonic in the number of replaced encoders. These point estimates alone do not establish a statistically reliable interaction between the branches.

## C. Custom-Curated Dataset: CARLA-MOS

To further extend DynOSeg to autonomous driving scenarios, we construct a custom simulated driving dataset, termed CARLA-MOS, using the CARLA simulator [9]. Figure C.2 presents representative samples from the proposed dataset, while Fig. C.1 illustrates the overall CARLA-based dataset collection pipeline. The dataset contains 1,000 sequences and 300k frames, all used as training data; it has no heldout test split in this study. Its contribution is evaluated by transfer to the real-world benchmarks in Table 3 of the main paper, not by in-domain CARLA-MOS accuracy.

The pipeline follows four stages: simulator setup, scene generation, ego-sensor configuration, and a synchronized capture loop. We first configure the simulator environ-

DAVIS2017-M DSEC-MOS

![](images/0c11d2763271d05e009feb921f57d4e536227508844e99edfcda74165f58d2d6.jpg)  
Figure A.7. Qualitative ablation of fusion structures. Columns compare ground truth, the full two-stage DynOSeg model, direct feature concatenation, GSF only, GLSF only, and the swapped GLSF–GSF order. Rows show examples from DAVIS2017-M and DSEC-MOS. Green overlays denote moving-object foreground regions.

ment and traffic manager and then spawn ego vehicles together with non-player trucks, cars, bicycles, motorcycles, and pedestrians with randomized trajectories and speeds. At each simulation step, the pipeline synchronously records RGB images, instance masks, semantic labels, depth maps, camera poses, and intrinsic parameters under varied urban traffic conditions.

Dynamic objects are identified from their motion states in the simulator, and pixel-level moving-object masks are generated from CARLA’s instance annotations. The synchronized multimodal annotations also include depth observations and camera calibration parameters. We will release CARLA-MOS and the corresponding data-collection scripts with the DynOSeg source code.

The collected data are organized in a structured format for efficient training and evaluation. RGB observations are stored as JPEG images, while binary moving-object masks are saved as PNG files. Instance-level semantic segmentation annotations are also collected, where each object instance is represented using a unique RGB color encoding and a unique ID number in the metadata. Depth maps are compressed into a ZIP archive, with each frame stored as a 16-bit PNG image representing metric depth values. Camera trajectories are saved as 4 × 4 camera-to-world transformation matrices, and camera intrinsics are stored as a 3 × 3 intrinsic matrix. In addition, all simulation metadata, including timestamps, actor identities, object statistics, and simulation settings, are serialized into a metadata JSON file for reproducibility and future analysis.

## D. KITTI-MOS

Few datasets simultaneously provide camera poses, dense depth maps, and moving-object masks. To evaluate DynOSeg with geometry cues from different sources, we construct the KITTI-MOS benchmark using KITTI tracklets and SAM 3 [5]. KITTI camera and object poses are used to identify moving tracklets and generate pseudo-GT moving-object masks. The dense depth used by DynOSeg is supplied by the selected configuration; it is not ground-truth dense depth.

KITTI-MOS construction. To support the plug-and-play geometry comparison in the main paper, we construct the KITTI-MOS benchmark from KITTI Raw sequences [13] with available tracklet annotations. For each sequence, we use the left color camera images, KITTI calibration, OXTS ego-motion, Velodyne scans, and tracklet labels. Each tracklet provides an object category, dimensions, and a perframe 6-DoF object pose in the Velodyne frame. We first recover each object’s world-frame trajectory using KITTI calibration and OXTS poses and then classify a tracklet as dynamic if its smoothed maximum horizontal speed exceeds 1.5\,mathr {m/s} at the KITTI Raw frame rate of 10 Hz. For each visible dynamic object, we project the 3D tracklet box to the image 02 plane and use the resulting 2D box, together with an object-class text prompt, to query SAM 3 [5]. The per-object SAM 3 masks are then combined into a binary dynamic mask:

$$
M _ { t } ( \mathbf { p } ) = \bigvee _ { i \in \mathcal { D } _ { t } } \mathrm { S A M 3 } ( I _ { t } , \mathcal { B } _ { i , t } , c _ { i } ) ( \mathbf { p } ) ,\tag{D.1}
$$

![](images/63491924b2a2940cfcbb5ff7efe01c93e5d2cdeeb06cdee2fe075277f8c4d001.jpg)  
Figure C.1. CARLA-based data collection pipeline. Simulator setup, scene generation, and ego-sensor configuration are followed by a synchronized capture loop that records RGB, semantic segmentation, depth, and camera parameters at each step; dynamic-object masks are then derived from actor motion and stored with the other modalities.

Table D.1. KITTI-MOS construction and geometry configurations, supplementing Table 7 in the main paper. The annotation pipeline is separate from the geometry inputs used by DynOSeg.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Dataset source</td><td>KITTI Raw sequences with tracklet annotations</td></tr><tr><td>Camera stream</td><td>Left color camera, image_02</td></tr><tr><td>Number of sequences / frames</td><td>38 sequences / 12,915 frames</td></tr><tr><td>Reference camera poses</td><td>KITTI GT camera poses derived from OXTS and calibration</td></tr><tr><td>Reference dense depth</td><td>FoundationStereo applied to KITTI stereo inputs</td></tr><tr><td>Alternative geometry inputs</td><td>Offline ViPE poses/depth; online DynOSAM poses + completed KITTI depth</td></tr><tr><td>Object motion source</td><td>KITTI tracklet 3D object poses + OXTS ego-motion</td></tr><tr><td>Dynamic criterion</td><td>Smoothed max horizontal object speed &gt; 1.5 m/s</td></tr><tr><td>Frame rate</td><td>10 Hz</td></tr><tr><td>2D prompt generation</td><td>Projected 3D tracklet boxes on image_02</td></tr><tr><td>Invalid projections</td><td>Fewer than 4 visible corners, area below 50 px, or depth below 0.1 m</td></tr><tr><td>Mask generator</td><td>SAM 3 with box and object-type text prompts</td></tr><tr><td>Final pseudo-GT mask</td><td>Union of SAM 3 masks for moving tracklets</td></tr></table>

where $I _ { t }$ is the RGB frame, $\boldsymbol { B } _ { i , t }$ is the projected 2D box of object $i , c _ { i }$ is the text prompt derived from its KITTI object type, and $\mathcal { D } _ { t }$ denotes dynamic objects visible at frame t. We discard invalid projections with fewer than four visible 3D box corners, area below 50 pixels, or depth below 0.1 m. The final benchmark contains 38 KITTI Raw sequences and 12,915 frames; Tab. D.1 summarizes the protocol.

Geometry-provider configurations. The reference configuration in the main paper combines KITTI ground-truth camera poses, obtained from OXTS ego-motion and camera calibration, with dense depth generated by Foundation-Stereo [49] from KITTI stereo inputs. Thus, the reference does not assume access to ground-truth dense depth. The ViPE configuration uses camera poses and depth estimated offline by ViPE [17]. For the online configuration, we extend the DynOSAM [32] data-provider interface to accept KITTI sparse LiDAR depth and expose the depth together with the camera poses estimated online by DynOSAM. The instance masks required by DynOSAM are generated by its built-in online detector. The same trained DynOSeg network and residual construction are used for all three configurations, without provider-specific retraining. Table 7 in the main paper reports prediction consistency $\mathcal { C } _ { \mathrm { I o U } }$ values of 89.0, 84.0, and 89.0 for the KITTI reference, ViPE, and DynOSAM configurations, respectively, with an overall mean pairwise IoU of 87.3. These consistency scores show that the same trained DynOSeg network produces similar masks with the tested geometry providers; they do not measure geometry accuracy, robustness to arbitrary geometry errors, or agreement with human-annotated ground truth.

![](images/4c58c60a2436470fc50db41a0801536449faadba7b6418389eb9bb6b954901d4.jpg)  
Figure C.2. CARLA-MOS: a custom-curated simulated driving-scene MOS dataset generated with CARLA [9] for DynOSeg training.

Nearest-valid depth propagation. KITTI Velodyne points are projected onto the left color image using the provided camera–LiDAR calibration, producing a sparse metric-depth map $D _ { t } ^ { \mathrm { s } }$ at frame t. If multiple projected points fall on the same pixel, we retain the smallest positive depth. Let $\Omega _ { t }$ denote the set of pixels with valid projected depth. For every missing pixel $\textbf { p } \notin \ \Omega _ { t } ,$ the modified DynOSAM interface copies the depth of the spatially nearest valid pixel in the image plane:

$$
\mathbf { q } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { q } \in \Omega _ { t } } \| \mathbf { p } - \mathbf { q } \| _ { 2 } , \qquad D _ { t } ( \mathbf { p } ) = D _ { t } ^ { \mathrm { s } } ( \mathbf { q } ^ { * } ) .\tag{D.2}
$$

The original valid LiDAR measurements are retained unchanged, i.e., $D _ { t } ( \mathbf { p } ) = D _ { t } ^ { \mathrm { s } } ( \mathbf { p } )$ for $\mathbf { p } \in \Omega _ { t }$ . The completed map is stored as a double-precision metric-depth image in the format expected by DynOSAM. This operation is deterministic, uses only the current sparse-depth map, and requires neither training nor future frames. DynOSAM consumes this completed depth to estimate the current camera pose, and DynOSeg uses the same completed depth with the DynOSAM pose estimates.

Causal online interface. At time t, the interface accepts depth and camera poses for frames t − 1 and t and optical

DynOSAM & DynOSeg

ViPE & DynOSeg

GT & FoundationStereo & DynOSeg

![](images/d54fa73f316eacc209271fb66dbe2d2f933b8a30b92fa00482a6e1bf5a770848.jpg)  
Figure D.1. Visualization results on KITTI-MOS. The “GT & FoundationStereo & DynOSeg” configuration uses KITTI GT camera poses and FoundationStereo dense depth from KITTI stereo inputs. “GT” refers only to camera pose; the reference masks are SAM 3 pseudo-GT. The DynOSAM configuration uses online DynOSAM camera poses and nearest-valid completion of KITTI sparse LiDAR depth, while ViPE supplies its offline poses and depth.

flow between the two RGB frames. The previous frame’s geometry can be buffered while the online provider produces the current estimates. The resulting residual map is aligned with the current frame, and the segmentation network receives $\left( I _ { t } , D _ { t } , R _ { t } \right)$ , including the current depth $D _ { t }$ rather than the previous depth as its depth-encoder input. No future frame is needed when the pose/depth provider is causal and the flow is computed from this two-frame pair. Thus, DynOSeg can be integrated into online systems using online geometry sources such as DynOSAM; this does not make the default offline ViPE configuration online. Nor does online compatibility by itself establish real-time throughput: the full system must also account for geometry estimation, optical flow, residual construction, data transfer, and segmentation.

Geometry-induced residual. For all geometry sources, DynOSeg uses the same residual construction. Given source frame t, adjacent target frame q, camera-to-world poses $T _ { w  c _ { t } }$ and $T _ { w  c _ { q } }$ , depth $D _ { t }$ , optical flow $\mathbf { f } _ { t  q } .$ , and intrinsics K, we compute

$$
T _ { c _ { q }  c _ { t } } = T _ { w  c _ { q } } ^ { - 1 } T _ { w  c _ { t } } ,\tag{D.3}
$$

$$
\hat { \mathbf { f } } _ { t  q } ( \mathbf { p } ) = \pi \big ( K \big [ \mathbf { R } _ { c _ { q }  c _ { t } } D _ { t } ( \mathbf { p } ) K ^ { - 1 } \tilde { \mathbf { p } } + \mathbf { v } _ { c _ { q }  c _ { t } } \big ] \big ) - \mathbf { p } ,\tag{D.4}
$$

where $\mathbf { R } _ { c _ { q }  c _ { t } }$ and $\mathbf { v } _ { c _ { q }  c _ { t } }$ are the rotation and translation of $T _ { c _ { q }  c _ { t } } , \tilde { \mathbf { p } }$ is a homogeneous pixel coordinate, and π performs perspective division. Depth and translation must use a common scale. The residual magnitude is

$$
R _ { t } ( \mathbf { p } ) =  \mathbf { f } _ { t  q } ( \mathbf { p } ) - \hat { \mathbf { f } } _ { t  q } ( \mathbf { p } )  _ { 2 } .\tag{D.5}
$$

For online inference, q = t−1 gives a current-grid residual using current-to-previous optical flow, following Section 3.1 of the main paper. This residual map is used as the motion cue for DynOSeg. Importantly, only the geometry source changes across the KITTI-MOS evaluation in the main paper; RGB frames, optical flow, and evaluation masks remain fixed.

Annotation and geometry limitations. KITTI-MOS evaluates binary moving regions induced by the tracklet and SAM 3 annotation pipeline, not manually verified instance masks. Tracklet coverage and prompting errors can affect both apparent false positives and false negatives. The geometry comparison holds these pseudo-GT masks fixed; it does not measure annotation quality. Depth, pose, or flow errors can also produce nonzero residuals in static regions. Consequently, high prediction consistency across the three providers is not evidence that their geometry estimates are equally accurate.

Table D.2. Per-sequence DynOSeg results on FBMS-59. Scores are percentages; the final row reports the unweighted sequence mean, at the same precision as Table 2 of the main paper, and the total number of annotated frames. † marks sequences with $\mathcal { I } \& \mathcal { F } < 7 0 $
<table><tr><td>Dataset</td><td>Sequence</td><td>Annotated frames</td><td>J↑</td><td>F↑</td><td>J&amp;F↑</td></tr><tr><td>FBMS-59</td><td>camel01</td><td>6</td><td>91.28</td><td>99.64</td><td>95.46</td></tr><tr><td></td><td>cars1</td><td>3</td><td>97.42</td><td>98.70</td><td>98.06</td></tr><tr><td></td><td>cars10†</td><td>4</td><td>69.29</td><td>63.50</td><td>66.40</td></tr><tr><td></td><td>cars4†</td><td>6</td><td>28.12</td><td>47.55</td><td>37.84</td></tr><tr><td></td><td>cars5</td><td>4</td><td>63.25</td><td>82.73</td><td>72.99</td></tr><tr><td></td><td>cats01</td><td>9</td><td>94.76</td><td>98.36</td><td>96.56</td></tr><tr><td></td><td>cats03</td><td>6</td><td>88.59</td><td>90.87</td><td>89.73</td></tr><tr><td></td><td>cats06</td><td>18</td><td>61.13</td><td>79.38</td><td>70.25</td></tr><tr><td></td><td>dogs01†</td><td>11</td><td>53.97</td><td>67.52</td><td>60.75</td></tr><tr><td></td><td>dogs02</td><td>22</td><td>94.33</td><td>99.25</td><td>96.79</td></tr><tr><td></td><td>farm01</td><td>14</td><td>88.25</td><td>81.39</td><td>84.82</td></tr><tr><td>goats01</td><td>giraffes01</td><td>17</td><td>92.07</td><td>98.56</td><td>95.31</td></tr><tr><td>horses02</td><td></td><td>15</td><td>85.64</td><td>94.17</td><td>89.90</td></tr><tr><td>horses04</td><td></td><td>13</td><td>90.39</td><td>98.74</td><td>94.56</td></tr><tr><td>horses05</td><td></td><td>41</td><td>91.02</td><td>95.10</td><td>93.06</td></tr><tr><td>lion01</td><td></td><td>23</td><td>75.36</td><td>79.79</td><td>77.57</td></tr><tr><td></td><td></td><td>9</td><td>87.47</td><td>89.40</td><td>88.44</td></tr><tr><td>marple12 marple2</td><td></td><td>8</td><td>80.43</td><td>75.49</td><td>77.96</td></tr><tr><td></td><td></td><td>11</td><td>86.11</td><td>88.68</td><td>87.39</td></tr><tr><td>marple4</td><td></td><td>5</td><td>90.09</td><td>86.59</td><td>88.34</td></tr><tr><td>marple6</td><td></td><td>21</td><td>95.05</td><td>89.79</td><td>92.42</td></tr><tr><td>marple7</td><td></td><td>16</td><td>78.16</td><td>72.27</td><td>75.22</td></tr><tr><td>marple9</td><td></td><td>8</td><td>89.28</td><td>84.20</td><td>86.74</td></tr><tr><td>people03</td><td></td><td>10</td><td>87.99</td><td>83.59</td><td>85.79</td></tr><tr><td>people1</td><td></td><td>5</td><td>92.24</td><td>99.41</td><td>95.83</td></tr><tr><td>people2</td><td></td><td>4</td><td>91.94</td><td>94.63</td><td>93.29</td></tr><tr><td>rabbits02</td><td></td><td>14</td><td>93.31</td><td>97.67</td><td>95.49</td></tr><tr><td>rabbits03</td><td></td><td>12</td><td>91.70</td><td>96.39</td><td>94.04</td></tr><tr><td>rabbits04†</td><td></td><td>16</td><td>57.74</td><td>66.63</td><td>62.19</td></tr><tr><td>tennis</td><td></td><td>16</td><td>90.61</td><td>97.44</td><td>94.02</td></tr><tr><td colspan="2">Sequence mean / total frames</td><td>367</td><td>82.6</td><td>86.6</td><td>84.6</td></tr></table>

## E. FBMS-59 Per-Sequence Results

For completeness, Tab. D.2 reports the DynOSeg result for every evaluated FBMS-59 sequence together with its number of annotated frames. The per-sequence metrics are expressed as percentages and rounded to two decimals. The final row is the unweighted mean over the 30 sequence-level scores, shown to one decimal to exactly match the FBMS-59 entry in Table 2 of the main paper; annotated frames are summed across sequences. A dagger marks the four sequences with $\mathcal { I } \& \mathcal { F } < 7 0 $

Failure analysis on cars4. The cars4 sequence has the lowest J&F score in Tab. D.2. As illustrated in Fig. E.1, a plausible contributor is local camera-pose error around the highlighted target frame. An inaccurate pose perturbs the rigid camera-induced flow synthesized from pose and depth, so static pixels can receive spurious residual responses. DynOSeg may then interpret these residual artifacts as independent motion, which is consistent with the false-positive foreground regions near the bottom of the predictions in the later examples. This visualization supports pose-induced residual error as a failure hypothesis rather than establishing it as the sole cause. Depth and opticalflow errors can produce similar effects.

## F. Runtime and Resolution Scaling

Timing definitions. All runtimes use one NVIDIA RTX 4090 with batch size 1. First, Table 8 in the main paper and Tab. F.1 isolate how the main components scale with input resolution. DynOSeg wall-clock timing starts after the depth and residual maps are available and includes DI-NOv2 feature extraction, data transfer, feature fusion, decoding, and mask post-processing. It therefore excludes geometry estimation, RAFT, residual-map construction, and disk I/O. The separate DynOSeg GPU column measures DINOv2 and segmentation computation only; it excludes data transfer and post-processing. RAFT uses the FlyingThings3D checkpoint with 20 iterations, and the residual column measures camera-induced flow and residual-map construction once depth, poses, and optical flow are available. Because the wall-clock and GPU measurements were collected independently, their difference does not isolate transfer or post-processing overhead, and the columns in Tab. F.1 must not be added to claim measured end-to-end latency.

![](images/daaaad621be9f316e632edcfe125b1f4d4111b0618f0415a03af9203643f1e7d.jpg)

![](images/b591a105b17ef46a6a9294a9a940a3b88b9a754346d0273197922fef074370c0.jpg)  
Figure E.1. Failure analysis on the FBMS-59 cars4 sequence. The left panel shows four representative examples. From left to right each example contains the ground-truth foreground overlay, optical flow, metric depth, motion residual, and DynOSeg prediction. The red rectangle marks target frame 10 and links it to the camera-pose visualization on the right. The trajectory covers frames 5–15, with frame 10 shown in red and adjacent frames 9 and 11 shown in green. Metric depth uses the Turbo colormap, where cooler colors indicate smaller depth and warmer colors indicate larger depth. Motion residuals use Inferno, where darker colors indicate better agreement with rigid camera-induced motion and brighter colors indicate larger deviations. For optical flow, hue indicates direction and saturation indicates magnitude. All maps are normalized independently for each frame, so colors should not be compared across examples.

Causal online and offline pipelines. DynOSeg requires only the current and previous frames, their depth and camera poses, and the optical flow between them, as detailed in Sec. D. The previous frame can be buffered, so no future frame is required when the geometry provider is causal. The evaluated online route uses the modified DynOSAM interface described above, followed by RAFT, DynOSAM pose estimation, residual construction, and DynOSeg inference. The nearest-valid sparse-to-dense preprocessing is outside the scope of this timing analysis and is excluded from the reported online totals. The offline route uses ViPE to estimate camera poses and depth with sequence-level processing, followed by RAFT, residual construction, and DynOSeg. The GT-pose plus FoundationStereo route is a controlled reference rather than a deployable online system because it uses KITTI ground-truth camera poses.

Table F.1. Runtime at five resolutions, in ms/frame. These are independently measured component timings, not additive stages of an end-to-end benchmark. Values match Table 8 in the main paper.
<table><tr><td>Resolution (W × H)</td><td>DynOSeg Wall-clock</td><td>GPU</td><td>RAFT GPU</td><td>Residual GPU</td></tr><tr><td>518 × 518</td><td>20.13</td><td>19.00</td><td>25.93</td><td>0.43</td></tr><tr><td> $6 3 0 \times 6 3 0$ </td><td>32.95</td><td>32.28</td><td>31.99</td><td>0.44</td></tr><tr><td> $7 2 8 \times 7 2 8$ </td><td>46.93</td><td>47.15</td><td>44.89</td><td>0.46</td></tr><tr><td> $1 3 4 4 \times 7 5 6$ </td><td>102.43</td><td>106.20</td><td>84.57</td><td>0.58</td></tr><tr><td>1792 × 1008</td><td>252.69</td><td>270.65</td><td>173.47</td><td>1.02</td></tr></table>

Sequential runtime accounting. Table F.2 summarizes stage-wise latency accounting on KITTI-MOS without repeating the resolution-invariant upstream costs in every row. Panel (a) lists the accounted resolution-invariant stages for each route, while panel (b) varies only the square DynOSeg input resolution and reports the resulting sequential totals. The reported total is the sequential sum of independently averaged stage times without pipeline overlap; it is not a directly measured end-to-end wall-clock latency. FPS is the reciprocal of this sequential estimate.

Component scaling. The tested dimensions in Tab. F.1 are multiples of DINOv2’s patch size 14; the two rectangular settings use a 16:9 aspect ratio. With precomputed depth and residual maps, DynOSeg wall-clock latency increases from 20.13 ms/frame at 518 × 518 to 252.69 ms/frame at 1792 × 1008. The former corresponds to approximately 49.7 FPS for the DynOSeg module alone, whereas the latter corresponds to approximately 4.0 FPS. Across the same settings, RAFT requires 25.93–173.47 ms/frame, while residual construction requires only 0.43–1.02 ms/frame. Thus, once depth, poses, and optical flow are available, converting them into the residual cue is inexpensive relative to opticalflow estimation and segmentation.

Table F.2. Stage-wise KITTI-MOS runtime on an NVIDIA RTX 4090. Panel (a) lists the accounted upstream costs that do not change with the DynOSeg input size; panel (b) reports DynOSeg latency and the resulting sequential total/FPS for each route. FS denotes FoundationStereo.  
(a) Resolution-invariant upstream stages (ms/frame)
<table><tr><td>Route</td><td>Geometry/depth</td><td>RAFT</td><td>DynOSAM</td><td>Residual</td></tr><tr><td>Controlled (GT pose + FS)</td><td>180.71</td><td>27.81</td><td>一</td><td>0.48</td></tr><tr><td>Offline (ViPE)</td><td>529.46</td><td>27.81</td><td>一</td><td>0.48</td></tr><tr><td>Online (DynOSAM)</td><td>Excluded</td><td>27.81</td><td>69.40</td><td>0.48</td></tr></table>

(b) Resolution-dependent DynOSeg latency and accounted sequential estimates
<table><tr><td rowspan="2">DynOSeg input</td><td rowspan="2">DynOSeg (ms)</td><td colspan="3">Sequential total (ms/frame) / FPS</td></tr><tr><td>Controlled</td><td>Offline</td><td>Online</td></tr><tr><td> $1 9 6 \times 1 9 6$ </td><td>4.98</td><td>213.99 / 4.67</td><td>562.73 / 1.78</td><td>102.67 / 9.74</td></tr><tr><td> $3 9 2 \times 3 9 2$ </td><td>9.38</td><td>218.39 / 4.58</td><td>567.14 / 1.76</td><td>107.07 / 9.34</td></tr><tr><td> $5 8 8 \times 5 8 8$ </td><td>13.75</td><td>222.76 / 4.49</td><td>571.51 / 1.75</td><td>111.44 / 8.97</td></tr><tr><td> $7 8 4 \times 7 8 4$ </td><td>20.63</td><td>229.63 / 4.35</td><td>578.38 / 1.73</td><td>118.32 / 8.45</td></tr><tr><td> $9 8 0 \times 9 8 0$ </td><td>32.92</td><td>241.93 / 4.13</td><td>590.68 / 1.69</td><td>130.61 / 7.66</td></tr><tr><td> $1 1 7 6 \times 1 1 7 6$ </td><td>54.23</td><td>263.24 / 3.80</td><td>611.98 / 1.63</td><td>151.92 / 6.58</td></tr><tr><td> $1 3 7 2 \times 1 3 7 2$ </td><td>84.81</td><td>293.81 / 3.40</td><td>642.56 / 1.56</td><td>182.50 / 5.48</td></tr><tr><td> $1 5 6 8 \times 1 5 6 8$ </td><td>126.00</td><td>335.01 / 2.98</td><td>683.76 / 1.46</td><td>223.69 / 4.47</td></tr><tr><td> $1 7 6 4 \times 1 7 6 4$ </td><td>187.50</td><td>396.50 / 2.52</td><td>745.25 / 1.34</td><td>285.19 / 3.51</td></tr><tr><td> $1 9 3 2 \times 1 9 3 2$ </td><td>256.17</td><td>465.17 / 2.15</td><td>813.92 / 1.23</td><td>353.86 / 2.83</td></tr></table>

Protocol. Upstream stages run at the native KITTI resolution (1242 × 375); square DynOSeg inputs use multiples of 14. FoundationStereo is averaged over 50 runs, and RAFT, residual construction, and DynOSeg over 30 runs with batch size 1. Totals sum independently averaged stages without pipeline overlap and exclude disk I/O; the online totals also exclude nearest-valid depth propagation.

Accounted pipeline cost and bottlenecks. At a 196×196 DynOSeg input, the controlled reference, offline ViPE, and online DynOSAM routes have accounted sequential estimates of 213.99, 562.73, and 102.67 ms/frame, respectively, corresponding to 4.67, 1.78, and 9.74 FPS. At $1 9 3 2 \times 1 9 3 2$ , these estimates increase to 465.17, 813.92, and 353.86 ms/frame (2.15, 1.23, and 2.83 FPS). At the smallest DynOSeg resolution, ViPE alone requires 529.46 ms/frame in the offline route, while the accounted online route includes 27.81 ms for RAFT and 69.40 ms for the DynOSAM stage, compared with 4.98 ms for DynOSeg. At the largest resolution, DynOSeg grows to 256.17 ms/frame and becomes a substantial part of the total, although ViPE remains the largest stage in the offline route. Residual construction remains 0.48 ms/frame at native KITTI resolution and contributes less than 0.5% of every total in Tab. F.2.

Online runtime analysis. Here, online means causal: the DynOSAM configuration can produce the current prediction without observing future frames. It does not mean that the current implementation runs at the input frame rate. The fastest accounted online sequential estimate is 102.67 ms/frame (9.74 FPS), which already exceeds the 100 ms/frame budget of KITTI’s 10 Hz stream before the excluded sparse-to-dense preprocessing is considered. Accordingly, our results demonstrate compatibility with online geometry inputs, not real-time system throughput. Achieving real-time operation would require a directly optimized and measured streaming implementation, potentially with faster geometry and flow providers, lower DynOSeg resolution, and pipeline overlap.