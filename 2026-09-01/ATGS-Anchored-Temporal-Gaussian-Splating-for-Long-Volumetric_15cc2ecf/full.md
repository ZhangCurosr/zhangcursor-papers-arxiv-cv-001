# ATGS: Anchored Temporal Gaussian Splating for Long Volumetric Video Representation

JIAHAO WU<sup>∗</sup>, Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Shenzhen Graduate School, Peking University, Pengcheng Laboratory, China

JIE LIANG<sup>∗</sup>, Shenzhen Graduate School, Peking University, Pengcheng Laboratory, China

DIE HU, Shenzhen Graduate School, Peking University, China

JIAYU YANG and XIAOYUN ZHENG, Pengcheng Laboratory, China

KAIQIANG XIONG and XIANG LI, Shenzhen Graduate School, Peking University, Pengcheng Laboratory, China CHAO WANG<sup>†</sup>, Pengcheng Laboratory, China

RONGGANG WANG<sup>†</sup>, Shenzhen Graduate School, Peking University, Pengcheng Laboratory, China

Long Time Varying  
![](images/39b51b940e8ccd13d61cbd0292ffbc0cf2f8ed834281af822ffeec13bd756223.jpg)  
Fig. 1. Based on long-sequence multi-view videos, our method is capable of reconstructing high-quality volumetric videos and enables novel view synthesis (NVS) at arbitrary viewpoints and time steps. Compared to prior approaches, which only support 20 frames fast motion or 300 frames minor hand or head motion (N3DV), our method supports fast motions over thousands.

Volumetric video enables immersive free viewpoint rendering of dynamic real world scenes, yet existing methods struggle with long sequences and complex motions, often leading to temporal instability and visual artifacts. To address these challenges, we propose ATGS, a Gaussian splatting based framework for volumetric video reconstruction. Our key insight is that explicitly tracking long term complex motion with individual Gaussian primitives is inherently unstable. Instead, we organize Gaussians around time conditioned anchors that localize their spatial and temporal support, thereby reducing long range motion complexity. We further introduce a temporal windowing strategy to activate only anchors relevant to the queried time, which improves scalability and temporal coherence. In addition, to ensure spatial and temporal stability, we design a compact set of multi level anchor features that encode global features, local spatial features, and local temporal features, jointly constraining Gaussian generation. Extensive experiments demonstrate that ATGS consistently outperforms prior methods on long sequence volumetric videos with complex motions. Project page: https://github.com/WuJH2001/ATGS.

CCS Concepts: • Computing methodologies → Image-based rendering.

Additional Key Words and Phrases: Immersive volumetric video, neural rendering, neural radiance field, 3d gaussian splatting, multi-view videos.

## ACM Reference Format:

Jiahao Wu<sup>∗</sup>, Jie Liang<sup>∗</sup>, Die Hu, Jiayu Yang, Xiaoyun Zheng, Kaiqiang Xiong, Xiang Li, Chao Wang<sup>†</sup>, and Ronggang Wang<sup>†</sup>. 2026. ATGS: Anchored Temporal Gaussian Splatting for Long Volumetric Video Representation. ACM Trans. Graph. 45, 4, Article 110 (September 2026), 13 pages. https://doi.org/ 10.1145/3811306

## 1 Introduction

Volumetric video enables free viewpoint rendering of dynamic real world scenes, allowing users to observe events from arbitrary perspectives. This capability makes volumetric video a compelling medium for immersive visual experiences in a wide range of appli cations, including sports broadcasting, live events, gaming, VR, AR and XR [Burdea and Coifet 2003; Jin et al. 2023; Rauschnabel et al. 2022; Speicher et al. 2019]. Current volumetric video reconstruction pipelines typically convert multi-view 2D video inputs into 3D volumetric representations. However, they face two fundamental challenges: long-sequence reconstruction and complex motion modeling. Here, long sequences refer to videos with minute-level durations, while complex motions involve large inter-frame variations, such as fast multi-person movements from diverse viewpoints and dynamic lighting changes. The goal of ATGS is to address both challenges in a unified framework.

With the rapid development of neural rendering [Kerbl et al. 2023; Mildenhall et al. 2021], data driven volumetric representations have become the dominant approach for volumetric video reconstruction. By jointly modeling geometry and appearance in a unified framework, neural rendering methods achieve high visual fidelity and flexible view synthesis without relying on explicit surface reconstruction or handcrafted priors. These advantages make neural rendering particularly suitable for reconstructing dynamic scenes from multi view captures. Existing neural rendering based approaches for volumetric video reconstruction can be broadly categorized into three classes. Deformation based approaches [Huang et al. 2024a; Park et al. 2021a,b; Wu et al. 2024, 2025b; Yang et al. 2024] model dynamic scenes by learning deformation fields that warp a canonical space over time. 4D Gaussian based methods [Duan et al. 2024a; Wang et al. 2025; Xu et al. 2024c; Yang et al. 2023] represent dynamics using explicit spatio temporal Gaussian primitives, enabling eficient rendering with high visual fidelity. 4D grid based methods [Cao and Johnson 2023; Fridovich-Keil et al. 2023; Xu et al. 2023] adopt compact spatio temporal grids to store implicit features, which are decoded to recover geometry and appearance at each sample. Despite their success on short sequences and controlled settings, these approaches remain challenged when scaling to large scale dynamic scenes with long temporal durations and complex motions, where temporal instability, motion drift, and visual artifacts often emerge.

These limitations have also been recognized by several recent methods. LongVolCap [Xu et al. 2024c] focuses on long sequence volumetric video reconstruction and demonstrates the ability to model minute level captures, but its performance degrades under complex and fast changing motions [Wang et al. 2025]. In contrast, methods such as FreeTimeGS [Wang et al. 2025] and LocalDyGS [Wu et al. 2025a] are designed to handle complex dynamics, yet are limited to short temporal windows of only one to two seconds. Simply splitting long sequences into independent short clips leads to temporal inconsistencies and flickering artifacts. Consequently, jointly model ing long temporal durations and complex motions remains an open and challenging problem in volumetric video reconstruction.

To address these challenges, we propose ATGS, a volumetric video reconstruction method that jointly targets long temporal durations and complex motions. A fundamental dificulty in this setting lies in explicitly modeling complex motion over long sequences, which often leads to unstable optimization and accumulated errors. Rather than directly tracking long range motion, our method introduces a set of time conditioned anchors that constrain the spatial and temporal Gaussian generation, efectively decomposing long term dynamics into a collection of temporally localized modeling tasks. To further control temporal complexity, we employ a temporal windowing strategy that activates only anchors relevant to the queried time during training and inference. This design limits interference from distant time steps and improves temporal coherence, enabling stable optimization over long sequences.

To further stabilize Gaussian generation, we introduce a hierarchical anchor feature formulation that progressively constrains spatial structure and temporal variation. We start by associating each time conditioned anchor with a global spatial feature, which provides coarse contextual information about its surrounding region. While this representation ofers strong expressive power, it alone does not enforce local spatial consistency, making optimization susceptible to temporal jitter over long sequences. To address this limitation, we augment the anchor with local spatial features that encode locally consistent, time invariant structure. These features complement the global representation by constraining Gaussian generation within a stable spatial neighborhood, thereby improving robustness during training. Finally, to capture dynamic variations without introducing long range temporal dependencies, we incorporate local temporal features. By conditioning these features on the anchor position and local time within short temporal intervals, the model represents motion in a temporally localized manner. Together, the global spatial, local spatial, and local temporal features form a unified anchor representation that is jointly decoded to generate temporally aware Gaussian primitives around each anchor, enabling photorealistic free viewpoint rendering.

We evaluate our method on a diverse set of long-sequence, multiview datasets, including N3DV (Flame Salmon) [Li et al. 2022b], VRU [Wu et al. 2025b], and MeetRoom. In addition, we validate the generalization capability of our approach on the SelfCap dataset [Xu et al. 2024c], which contains several thousand frames of video. Please refer to the demonstration video and the supplementary material for additional results.

Our contributions can be summarized as:

(1) We propose a temporal anchor based Gaussian representation for volumetric video reconstruction, which localizes Gaussian primitives in space and time through time conditioned anchors and temporal windowing. This representation enables efective modeling of long sequence dynamic scenes with complex motions.

(2) We introduce a hierarchical anchor feature formulation that progressively constrains Gaussian generation using global spatial, local spatial, and temporal features. This design improves spatial stability and temporal consistency, leading to higher quality reconstruction of dynamic scenes.

(3) We conduct extensive experiments on multiple challenging benchmarks, demonstrating that our method consistently outperforms existing approaches on volumetric video reconstruction involving long temporal durations and complex motions.

## 2 Related Work

## 2.1 Volumetric video

Volumetric video is a 3D video representation that captures depth and color information of dynamic scenes to produce a sequence of three-dimensional models viewable from arbitrary viewpoints. By leveraging multi-view camera setups, it overcomes the planar limitations of conventional 2D video and enables full 6-DoF freeviewpoint rendering. Early approaches relied heavily on strong priors, such as shape-from-silhouette techniques [Ahmed et al. 2008], deformable template models [Carranza et al. 2003], or template-free dynamic reconstruction systems augmented with depth sensors [Newcombe et al. 2015]. There also exist generative approaches [Bozic et al. 2020; Lombardi et al. 2019] aimed at improving the quality of dynamic reconstruction; however, these methods either struggle with real-world scenes that involve complex structures and severe occlusions, or rely on expensive and sophisticated capture systems. Together, these limitations hinder the practical deployment of volumetric video technologies in real-world applications.

## 2.2 Neural radiance field for dynamic scenes

In recent years, Neural Radiance Fields (NeRF)[Mildenhall et al. 2021] have attracted significant attention due to their ability to deliver photorealistic and immersive visual experiences. Subsequent works [Park et al. 2021a,b; Pumarola et al. 2021] have extended NeRF to dynamic scenarios and volumetric video reconstruction. For example, D-NeRF [Pumarola et al. 2021] and NeRFies [Park et al. 2021a] adopt deformation fields to warp a canonical space over time, enabling dynamic view synthesis, while [Du et al. 2021] explicitly incorporates the temporal dimension into the spatial domain to model spatiotemporal variations. However, the high training cost of NeRF-based representations motivates the development of more eficient alternatives. Grid-based representations that enable faster reconstruction have therefore gained popularity, such as Plenoxel and DVGO [Fridovich-Keil et al. 2022; Sun et al. 2022]. Building upon these representations, several dynamic extensions have been proposed, including MixVoxel [Wang et al. 2023b], which further accelerates reconstruction. Despite their eficiency, these methods sufer from high memory consumption, which limits their scalability and practical deployment. To address this issue, Instant-NGP introduces a multi-resolution hash encoding to store node features eficiently, significantly reducing memory usage. This idea has been further extended to dynamic settings, for example in MSTH [Wang et al. 2023a]. In parallel, researchers have explored factorizing 3D or 4D volumetric representations into lower-dimensional structures. Notably, Tensorf [Chen et al. 2022] decomposes volumetric fields into compact tensor representations, while explicit multi-plane methods such as K-Planes, HexPlane, and 4K4D [Cao and Johnson 2023; Fridovich-Keil et al. 2023; Xu et al. 2024b] represent 4D scenes using a set of structured planes, achieving a favorable balance between eficiency and expressiveness.

## 2.3 Gaussian splating for dynamic scenes

3D Gaussian Splatting (3DGS) [Kerbl et al. 2023] has attracted significant attention due to its ability to eficiently reconstruct high-fidelity 3D scenes from posed images. A series of extensions have been proposed to further improve surface reconstruction quality [Huang et al. 2024b; Zhang et al. 2024], accelerate training [Mallick et al. 2024], and improve rendering quality [Lu et al. 2024; Yu et al. 2024]. In the dynamic setting, several representative research directions have also emerged, including frame-wise training methods [Luiten et al. 2024; Sun et al. 2024], deformation-based methods [Huang et al. 2024a; Wu et al. 2024; Yang et al. 2024], and 4D Gaussian–based approaches [Duan et al. 2024b; Wang et al. 2025; Xu et al. 2024c; Yang et al. 2023]. Frame-wise training methods can handle long video sequences, but they require substantial storage and often sufer from inter-frame error accumulation, which may lead to unstable training or model collapse in later stages. Deformation-based methods, on the other hand, struggle to model complex motions and to robustly track long-term dynamics.

More recently, several 4D Gaussian–based methods [Liang et al. 2026; Wang et al. 2025; Xu et al. 2024c] have attempted to address either long-sequence reconstruction or complex motion modeling. For example, LongVolCap [Xu et al. 2024c] decomposes long-term motion into multiple hierarchical levels using layered 4D Gaussians, but shows limited capability in handling highly complex motions. In contrast, FreeTimeGS [Wang et al. 2025] places a large number of 4D Gaussian primitives in space to better capture complex dynamics, while LocalDyGS [Wu et al. 2025a] adopts an anchor-based formulation to improve motion modeling. However, these methods are restricted to handling only second-level complex motions and fail to scale to minute-level long-sequence volumetric video reconstruction. To address these limitations, we propose ATGS, which is designed to jointly tackle the challenges of complex motion modeling and long-duration volumetric video reconstruction.

## 3 Method

## 3.1 Overview

Our method targets volumetric video reconstruction with long tem poral durations and complex motions by representing dynamic scenes using temporally localized Gaussian primitives. The framework consists of four main components. We first introduce timeconditioned anchors to represent the dynamic scene over long sequences (Sec. 3.2). A temporal windowing strategy is then employed to select anchors relevant to a queried time, ensuring smooth temporal evolution during inference (Sec. 3.3). To enable stable and expressive modeling, each anchor is augmented with a hierarchical spatio-temporal feature representation (Sec. 3.4). Finally, the enriched anchors are decoded into Temporal Gaussians for ren dering (Secs. 3.5), and the entire model is trained end-to-end with image-based losses and regularization (Secs. 3.6).

![](images/e1eefdf3b0581fe7ac1944d6c89ca41e473238367626d025689d6ce14bedbc0d.jpg)  
Fig. 2. Overview of ATGS. Keyframes are extracted from each view to initialize time-conditioned anchors with independent anchor features. Anchors query spatial and temporal grids using their positions and timestamps, and the fused features are decoded into temporal Gaussians. During training, only anchors and temporal grids associated with the current timestamp are updated.

## 3.2 Time-conditioned Anchor Initialization

To model long sequence, large scale spatiotemporal scenes, we represent the dynamic content using a sparse set of anchors initialized from periodically sampled keyframes. This strategy provides sparse yet comprehensive coverage of scene geometry over time, ensuring that dynamic objects throughout the sequence are associated with anchor representations.

Each anchor is defined by a set of spatial and temporal attributes. Specifically, we assign an anchor position $\mu \in \mathbb { R } ^ { 3 }$ to represent its 3D location in space, together with a learnable anchor feature $f _ { a } \in \mathbb { R } ^ { 6 4 }$ that encodes local spatial characteristics around the anchor. In addition, each anchor is associated with a spatial scale parameter $\boldsymbol { v } \in \mathbb { R } ^ { 3 }$ , which defines the spatial extent of its influence and serves as a normalization factor for subsequent Gaussian generation. To incorporate temporal information, we further assign each anchor a discrete temporal index $k \in \{ 1 , \ldots , K \}$ , indicating the keyframe from which it is initialized. Anchors equipped with both spatial and temporal attributes are referred to as time-conditioned anchors. Formally, a time-conditioned anchor is defined as

$$
G _ { a } = \{ \mu , f _ { a } , \upsilon , k \} .\tag{1}
$$

This time-conditioned anchor formulation significantly simplifies the modeling of complex motions. By using anchors as priors, the model no longer needs to explicitly model Gaussian point trajectories; instead, anchors control the appearance and disappearance of Gaussian primitives, which is suficient to achieve perceptually plausible motion modeling.

## 3.3 Temporal Windowing Strategy

As shown in Fig.2, given the set of time-conditioned anchors defined over the entire sequence, we introduce a temporal windowing

strategy to select anchors relevant to a queried time $t \in [ 0 , 1 )$ while ensuring temporally coherent inference.

We define a temporal window of size $W \in \mathbb { N }$ (default $W = 7 )$ and denote its half-width as

$$
h = \left\lfloor { \frac { W } { 2 } } \right\rfloor .\tag{2}
$$

For a given query time �, we identify the nearest keyframe index <sup>¯</sup>� by mapping the normalized time to the keyframe index domain,

$$
\bar { k } = \arg \operatorname* { m i n } _ { j \in \{ 1 , \ldots , K \} } \left| t \cdot \left( K - 1 \right) + 1 - j \right| .\tag{3}
$$

The anchor group associated with time � is then constructed by collecting anchors whose temporal indices fall within a window centered at <sup>¯</sup>�,

$$
G _ { a , t } = \left\{ G _ { a } \{ \mu , f _ { a } , v , k \} \ : \middle | k \in \left[ \operatorname* { m a x } ( 1 , \bar { k } - h ) , \ : \operatorname* { m i n } ( K , \bar { k } + h ) \right] \right\}\tag{4}
$$

This temporal windowing strategy ensures that the anchor set used for inference evolves smoothly over time, avoiding abrupt changes in anchor activation across neighboring frames. As a result, it stabilizes long-sequence rendering and efectively reduces temporal jitter, which is a common challenge in volumetric video reconstruction over extended temporal durations.

## 3.4 Spatial-Temporal Feature Representation

Given the anchor group selected at a queried time �,

$$
G _ { a , t } = \left\{ G _ { a } \{ \mu , f _ { a } , v , k \} \mid k \in \left[ \operatorname* { m a x } ( 1 , \bar { k } - h ) , \operatorname* { m i n } ( K , \bar { k } + h ) \right] \right\} ,\tag{5}
$$

we next enrich each anchor with additional spatial and temporal features. The goal is to enable expressive temporal modeling while enforcing local spatial consistency to stabilize training over long sequences.

![](images/4e3623e546493dca2d9a8cd71f85bcb5c3b4a0d9fef43396af50177e24b0cfc9.jpg)  
（a） $f _ { a } + f _ { s } + f _ { t }$

![](images/72c1bfd1e41117353b8cb36353a4fb5f62c45abf71b6cfe6c2e396af84c47584.jpg)  
(b) �<sub>�</sub>

![](images/c592e131b64a83f794e0f1182957eb70e3ef5bb2438ee0dd625c860fb1d4ba2a.jpg)  
(c) $f _ { s } + f _ { t }$

![](images/48d54fc916ac7a25f345c5d3191df4ac30f0aa02d115e787a39a6389510f3430.jpg)  
(d) $f _ { s } + f _ { a }$

![](images/bcb4fb61ac67ea8288dae40be7193608d70662e6cad4f00ff22907a9c5b001ed.jpg)  
(e) �<sub>�</sub>  
Fig. 3. Visualization of the information captured by each feature component after training. As shown in (a), the combination of all features enables full image rendering. (b,c) indicate that anchor features $f _ { a }$ capture most of the scene information, while the other components $( f _ { s } , f _ { t } )$ contribute marginally. (d,e) show that anchor and static features together encode nearly all scene content (all Gaussians), while dynamic features primarily control the appearance and disappearance of Gaussian primitives rather than explicitly modeling the scene.

Local Consistency Representation. Our anchor representation captures local scene information using discrete anchor features, with each anchor optimized independently. While this design is suficient for static reconstruction or short sequences, it often leads to training instability and motion blur when applied to long volumetric videos. To mitigate this issue, we introduce a shared spatial grid $F _ { s }$ to enforce local spatial consistency across anchors.

Specifically, each anchor queries the spatial grid via trilinear interpolation at its spatial location $\mu$ to obtain a static spatial feature

$$
f _ { s } = F _ { s } ( \mu ) , \quad F _ { s } \in \mathbb { R } ^ { X \times Y \times Z \times D } ,\tag{6}
$$

where $( X , Y , Z )$ denote the spatial resolution of the grid and � is the feature dimension at each grid node. These spatial features provide locally consistent and time-invariant cues that complement the anchor features $f _ { a } ,$ , improving stability during optimization.

Temporal Dynamics Modeling. Modeling temporal variation in volumetric video has been approached using diverse representations, including MLP-based formulations, multi-plane factorizations, and hash-grid structures. However, using a single temporal representation is often insuficient to capture long-range dynamics in extended video sequences.

To address this limitation, we partition the temporal domain into � (we set $M = 1 2$ in VRU dataset, others � = 2, details in appendix) segments and introduce a set of temporal structures

$$
F _ { T } = \{ F _ { T , 1 } , \ldots , F _ { T , M } \} ,
$$

where each structure $F _ { T , m }$ models dynamic variations within the temporal interval $[ ( m - 1 ) / M , m / M ]$ . In our implementation, each $F _ { T , m }$ is instantiated as a 4D hash grid.

Given a query time � ∈ [0, 1), the corresponding temporal segment index is computed as

$$
m = \lfloor M \cdot t \rfloor + 1 ,\tag{7}
$$

and the local time coordinate within the segment is

$$
\hat { t } = M \cdot t - m + 1 .\tag{8}
$$

Each anchor then queries the associated temporal structure via quadrilinear interpolation to obtain a temporal feature

$$
f _ { t } = F _ { T , m } ( \mu , \hat { t } ) .\tag{9}
$$

This formulation decouples local temporal modeling from the global sequence length, enabling eficient and stable representation of long-term dynamics.

Augmented Anchor Representation. Finally, each anchor in the group $G _ { a , t }$ is augmented with both spatial and temporal features, yielding the enriched anchor representation

$$
G _ { a , t } = \left\{ G _ { a } \{ \mu , f _ { a } , f _ { s } , f _ { t } , v , k \} \left| k \in \left[ \operatorname* { m a x } ( 1 , \bar { k } - h ) , \operatorname* { m i n } ( K , \bar { k } + h ) \right] \right\} . \right.\tag{10}
$$

These augmented anchors serve as the input to the subsequent temporal Gaussian decoding stage.

The roles of the three feature components are illustrated in Fig. 3. The anchor and static features are responsible for encoding nearly all scene content, while the dynamic features capture only a small amount of time-varying information and do not explicitly contribute to the representation of static scene geometry. As details below:

(1) Anchor feature $f _ { a } .$ . Following Scafold-GS [Lu et al. 2024], anchors are decoded into Gaussians: $G = D e c o d e r ( f _ { a } )$ . However, time-invarying anchor features alone cannot model dynamic scenes.

(2) Temporal feature $f _ { t } .$ . Therefore, we introduce a shared hash grid $F _ { T } ( x , y , z , t )$ to provide time-aware features, enabling timevarying Gaussians for dynamic scene reconstruction:� ${ \mathrm { \Omega } } _ { t } = F _ { * } ( f _ { a } , f _ { t } )$

(3) Static feature $f _ { s }$ . Since $f _ { a }$ is optimized independently across anchors without spatial locality, it can lead to blurry reconstructions in complex scenes(Fig. 7). We therefore introduce $F _ { S } ( x , y , z )$ to encode low-frequency spatial structure and improve stability.

In summary, $F _ { T }$ models temporal variation while $F _ { S }$ provides stable spatial structure. Ablation on the VRU dataset (Tab. 7) shows that removing any component degrades performance.

## 3.5 Temporal Gaussian Decoding

Given a query time $t ,$ the proposed pipeline yields the corresponding anchor group $G _ { a , t }$ together with enriched spatio-temporal features. We decode these features using a Gaussian decoder $F _ { * }$ to generate a set of Temporal Gaussians, whose parameters vary continuously with time:

$$
G _ { T } \{ \mu _ { t } , q _ { t } , s _ { t } , \sigma _ { t } , c _ { t } \} = F _ { * } ( G _ { a , t } ) .\tag{11}
$$

Each time-conditioned anchor generates $q$ Temporal Gaussians. The mean positions of these Gaussians are computed as

$$
\{ \mu _ { t } ^ { i } \} _ { i = 0 } ^ { q - 1 } = \mu + v \cdot F _ { \mu } ( f _ { w } ) , \quad f _ { w } = [ f _ { a } + f _ { s } ; f _ { t } ] ,\tag{12}
$$

where $\mu$ and � denote the anchor position and spatial scale, respectively, as defined in Sec. 3.2. The function $F _ { \mu } ( \cdot )$ is a lightweight MLP that outputs a vector of size $q \times 3 ,$ , following prior Gaussian-based formulations [Lu et al. 2024; Wu et al. 2025a]. Here, $\mu _ { t } ^ { i }$ represents the mean of the �-th Temporal Gaussian generated by the anchor at time �. We adopt $f _ { w } = \left[ f _ { a } + f _ { s } ; f _ { t } \right]$ instead of $f _ { w } = f _ { a } + f _ { s } + f _ { t }$ , since the latter requires implicit disentanglement of static and dynamic components, which incurs an approximate 10% training slowdown. In contrast, the adopted formulation facilitates decoupled optimization of static and temporal features, leading to improved training eficiency and stability.

All remaining Gaussian attributes, including orientation �<sub>�</sub>, scale $s _ { t } ,$ opacity $\sigma _ { t } ,$ and color $c _ { t } ,$ are decoded in an analogous manner from the same spatio-temporal feature representation. The resulting set of Temporal Gaussians is then used for volumetric rendering.

## 3.6 Loss Function

We train the proposed model using a combination of image-based reconstruction losses and a compactness regularization term, following common practice in Gaussian-based rendering methods [Kerbl et al. 2023; Lu et al. 2024; Wu et al. 2025a]. To encourage each Temporal Gaussian to remain spatially localized around its associated anchor, we impose a volume-based regularization on the Gaussian scales:

$$
\mathcal { L } _ { v } = \sum _ { i = 1 } ^ { P } \operatorname { P r o d } ( s _ { t } ^ { i } ) ,\tag{13}
$$

where $P$ denotes the number of active Temporal Gaussians across all anchors at query time $t , s _ { t } ^ { i } \in \mathbb { R } ^ { 3 }$ represents the scale of the �-th Temporal Gaussian, and Prod(·) computes the product of the scaling components. This regularization penalizes overly large Gaussians and promotes compact, anchor-centered representations. In addition to the compactness constraint, we adopt standard image reconstruction losses, including the $L _ { 1 }$ loss and the structural similarity loss L . The overall training objective is defined as

$$
\mathcal { L } = \left( 1 - \lambda _ { \mathrm { S S I M } } \right) \mathcal { L } _ { 1 } + \lambda _ { \mathrm { S S I M } } \mathcal { L } _ { \mathrm { S S I M } } + \lambda _ { v } \mathcal { L } _ { v } ,\tag{14}
$$

where $\lambda _ { \mathrm { S S I M } } = 0 . 2$ and $\lambda _ { v } = 0$ .001 balance the contributions of the reconstruction and regularization terms.

## 4 Experiments

## 4.1 Implementation

For our method, the dimensions of features $f _ { a } , f _ { s } ,$ and $f _ { t }$ are set to 64. The resolution and feature dimensionality of $F _ { S }$ are set to $\{ X , Y , Z , D \} \ = \ \{ 1 2 8 , 1 2 8 , 1 2 8 , 6 4 \}$ . The hash table size of each $F _ { T }$ is set to $2 ^ { 1 6 }$ to trade of performance against storage, with other settings consistent with INGP [Müller et al. 2022]. All MLPs in the Gaussian Decoder consist of two layers with ReLU activation, followed by Sigmoid or normalization at the output.

Our method uses the Adam optimizer [Kingma and Ba 2014], similarly to 3DGS. We employ a fixed learning rate of 0.0075 for the features $f _ { a }$ and $f _ { s } .$ An independent learning rate decay strategy is applied to each temporal grid $F _ { t } ,$ ensuring that only the currently accessed grid undergoes decay.

In addition, we have two important parameters: the number of keyframes � and the number of temporal grids �, which are adjusted according to the magnitude of scene motion. For scenes with small motion, such as N3DV and MeetRoom, we set $K = 1 5$ and � = 2; only a few keyframes and temporal grids are suficient. In contrast, for scenes with large motion, such as VRU, which contain a large amount of dynamic information, a small number of temporal grids cannot capture the extensive temporal variations. Therefore, we set � = 250 and � = 12 for such cases. The temporal window size � is fixed to 7 for all scenes. All experiments are conducted on NVIDIA A100 GPU(80 GB).

Table 1. Quantitative comparison on the MeetRoom dataset [Li et al. 2022a]. PSNR is averaged across all 300 frames, while training time and storage requirements accumulate over the entire sequence.
<table><tr><td>Method</td><td>PSNR ↑</td><td>Time(hours) ↓</td><td>Size(MB) ↓</td></tr><tr><td>StreamRF [Li et al. 2022a]</td><td>26.72</td><td>0.85</td><td>2700</td></tr><tr><td>4DGaussian [Wu et al. 2024]</td><td>30.27</td><td>0.50</td><td>80</td></tr><tr><td>3DGStream [Sun et al. 2024]</td><td>30.79</td><td>0.60</td><td>1230</td></tr><tr><td>Hicom [Gao et al. 2024]</td><td>26.73</td><td>0.20</td><td>180</td></tr><tr><td>ATGS(Ours)</td><td>32.79</td><td>0.45</td><td>110</td></tr></table>

Table 2. Quantitative comparisons on the Neural 3D Video Dataset [Li et al. 2021]. “Size” is the total model size for 300 frames. DSSIM<sub>1</sub> sets data range to 1.0 while DSSIM<sub>2</sub> to 2.0 [Li et al. 2024]. ∗ indicates online method. Tested on 300 frames.
<table><tr><td>Method</td><td>PSNR↑</td><td>DSSIM1↓</td><td>DSSIM2↓</td><td>LPIPS↓</td><td>FPS↑</td><td>Time↓</td></tr><tr><td>StreamRF * [Li et al. 2022a]</td><td>28.26</td><td></td><td></td><td></td><td>10.9</td><td></td></tr><tr><td>NeRFPlayer [Song et al. 2023]</td><td>30.69</td><td>0.034</td><td></td><td>0.111</td><td>0.05</td><td>6.0 h</td></tr><tr><td>HyperReel [Attal et al. 2023]</td><td>31.10</td><td>0.036</td><td></td><td>0.096</td><td>2</td><td></td></tr><tr><td>K-Planes [Fridovich-Keil et al. 2023]</td><td>31.63</td><td>0.018</td><td></td><td></td><td>0.3</td><td>5.0 h</td></tr><tr><td>HexPlane [Cao and Johnson 2023]</td><td>31.70</td><td>0.014</td><td></td><td>0.075</td><td>0.21</td><td>12.0 h</td></tr><tr><td>MixVoxels [Wang et al. 2022]</td><td>31.73</td><td></td><td>0.015</td><td>0.064</td><td>4.6</td><td></td></tr><tr><td>4DGaussian [Wu et al. 2024]</td><td>31.02</td><td>0.030</td><td></td><td>0.150</td><td>30</td><td>0.67 h</td></tr><tr><td>3DGStream* [Sun et al. 2024]</td><td>31.67</td><td></td><td></td><td></td><td>215</td><td>1.0 h</td></tr><tr><td>RealTimeGS [Yang et al. 2023]</td><td>32.01</td><td></td><td>0.014</td><td>0.055</td><td>114</td><td>9.0 h</td></tr><tr><td>SpaceTimeGS [Li et al. 2024]</td><td>32.05</td><td>0.026</td><td>0.014</td><td>0.044</td><td>140</td><td>&gt; 5 h</td></tr><tr><td>LocalDyGS [Wu et al. 2025a]</td><td>32.28</td><td>0.028</td><td>0.014</td><td>0.044</td><td>105</td><td>0.58 h</td></tr><tr><td>ATGS(Ours)</td><td>32.56</td><td>0.027</td><td>0.013</td><td>0.043</td><td>70</td><td>0.9 h</td></tr></table>

## 4.2 Datasets

To comprehensively evaluate our method, we conduct experiments on several publicly available datasets with diverse temporal lengths and motion complexities, including the N3DV, VRU, and MeetRoom. Notably, the latter two datasets pose significantly greater challenges due to their temporal scale and motion variation.

N3DV [Li et al. 2022b]. Neural3DV is a widely used benchmark for novel view synthesis, captured using a multi-view system with 19–21 cameras. The videos are recorded at a resolution of 2704×2028 at 30 FPS. The dataset includes five scenes with 300 frames and one scene with 1200 frames. Following previous work, we downsample the data and split the cameras into training and testing sets.

Table 3. Quantitative comparison on VRU-basketball [Wu et al. 2025b]. We report PSNR, and SSIM to evaluate rendering quality. Methods marked with <sup>1</sup> are trained on short segments (20 frames or single-frame), while methods marked with <sup>2</sup> are trained on long segments (250 frames or 1400 frames). The best results are highlighted in bold.
<table><tr><td rowspan="3">Methods</td><td colspan="2">VRU GZ</td><td colspan="2">VRU DG</td><td colspan="2">VRU Long</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>3DGS1</td><td>30.50</td><td>0.949</td><td>30.28</td><td>0.945</td><td></td><td></td></tr><tr><td>4DGaussian1</td><td>28.32</td><td>0.927</td><td>28.24</td><td>0.926</td><td>23.01</td><td>0.873</td></tr><tr><td>SpacetimeGS1</td><td>27.42</td><td>0.915</td><td>27.45</td><td>0.920</td><td></td><td></td></tr><tr><td>LocalDyGS2</td><td>29.23</td><td>0.939</td><td>29.01</td><td>0.931</td><td>23.21</td><td>0.875</td></tr><tr><td>ATGS(Ours)²</td><td>30.61</td><td>0.948</td><td>30.37</td><td>0.939</td><td>24.78</td><td>0.881</td></tr></table>

MeetRoom [Li et al. 2022a]. The MeetRoom dataset consists of three forward-facing scenes captured by an array of 11–12 cameras, with a resolution of 1080p and a frame rate of 30 FPS. The dataset is sparse in camera viewpoints, making it well suited for evaluating the performance of our method under sparse-view settings. Following previous work, we use camera 0 for testing.

![](images/d888469c3ffa6172395d144e7562ce86ad5db27db5352ce642d13743872212ca.jpg)  
Fig. 4. Qualitative comparisons on the N3DV dataset. Our model can reconstruct a 40-second video (1,200 frames) in a single run and achieves higher rendering quality than other methods operating on much shorter segments.

VRU basketball [Wu et al. 2025b]. The VRU dataset is a highly challenging benchmark featuring large-scale motion. Since many existing methods are already overfitted to small-motion scenarios such as N3DV, VRU provides a more suitable testbed for evaluating new approaches. Each VRU scene is captured by 36 cameras and includes three scenes in total: two scenes of 250 frames at 1080p resolution and 25 FPS (VRU DG and GZ), and one scene of 1400 frames at 4K resolution and 25 FPS (VRU Long360). Following prior work, we use camera views 0, 10, 20, 30 for testing, with the remaining views used for training.

SelfCap [Xu et al. 2024c]. The SelfCap dataset consists of three dynamic videos, each captured at 60 FPS and 4K resolution using a synchronized array of 22 iPhone cameras. The video lengths range from 2 to 10 minutes, making this dataset particularly well suited for evaluating methods on long video sequences.

PKU-DyMVHumans [Zheng et al. 2024]. The dataset comprises 45 dynamic human-centric scenarios, totaling approximately 8.2 million frames. It captures 32 professional performers (16 males, 11 females, and 5 children) engaging in four distinct action types:

Table 4. The ablation study on the number of temporal grids � on GZ.
<table><tr><td>Metrics</td><td>M=1</td><td>M=4</td><td>M=8</td><td>M=12</td></tr><tr><td>PSNR↑</td><td>29.10</td><td>29.31</td><td>30.11</td><td>30.61</td></tr><tr><td>SSIM↑</td><td>0.933</td><td>0.939</td><td>0.942</td><td>0.948</td></tr><tr><td>LPIPS↓</td><td>0.086</td><td>0.077</td><td>0.062</td><td>0.053</td></tr></table>

dance, kungfu, sport, and fashion show, across varied locations and clothing styles. The data is categorized into two resolution groups: 36 scenes at 1920×1080 (1080P) with durations of 10 to 487 seconds, and 8 scenes at 3840×2160 (4K Studio) with durations of 10 to 231 seconds.

## 4.3 Evaluation

We compare our method against a set of representative classical approaches and recent state-of-the-art methods, including 4DGS [Wu et al. 2024], 3DGStream [Sun et al. 2024], Grid4D [Xu et al. 2024a], LocalDyGS [Wu et al. 2025a], and Swift4D [Wu et al. 2025b]. These methods span diverse design choices for dynamic scene representation and thus constitute a strong benchmark for evaluation. We conduct quantitative evaluations on three real-world datasets, including the classic N3DV dataset, the VRU dataset with large-scale motion, and the MeetRoom dataset featuring sparse camera views, which together cover a wide range of practical scenarios. Our evaluation focuses on three complementary aspects: image quality measured by PSNR, SSIM, and LPIPS, training and rendering time. SelfVol-Cap [Xu et al. 2024c] and FreeTimeGS [Wang et al. 2025] have not released public implementations at the time of writing and are hence not included in our comparisons.

![](images/264c34e094d0c3e433941c86dff547943594c180939f832983f49bab4c3f03f1.jpg)  
Fig. 5. Qualitative comparisons on the VRU dataset with 1,400 frames (about 1 minute). Our method is capable of training large-scale dynamic scenes with thousands of frames in a single run, while achieving superior rendering quality compared to prior approaches.

![](images/3e00acbe9b6e95df4c582b5621cdc45a8970c0f35e53e51e13b418590402b221.jpg)  
(a) Ground Truth

![](images/f0b7b00354c0fb9a79cba064ee116e21ae4837a18da99ae74a0825de1dec8be9.jpg)  
(b) Ours

![](images/30f863b1e0d5f0f25ded243835d7412736819660bbc4f7cb60b841f67dd2416b.jpg)  
(c) LocalDyGS

![](images/b5f8ae5850e5cbdfac5956b3b1ac4f212d8d25949f81f0756702ea11bac9c977.jpg)  
(d) 4DGaussian  
Fig. 6. Qualitative comparisons on the MeetRoom dataset (sparse-view scenario). The results show that our approach preserves more fine details and maintains clarity in regions with large motion compared to prior methods.

4.3.1 Quantitative Evaluation. We first compare our method with prior approaches using standard quantitative metrics across all datasets. As reported in Tab. 2, Tab. 1, and Tab. 3, our method consistently achieves superior reconstruction quality in terms of PSNR, SSIM, and LPIPS across all datasets, indicating improved fidelity in both pixel accuracy and perceptual quality. Importantly, these gains are achieved without sacrificing eficiency, as our method maintains comparable training time and supports real-time rendering .

Table 5. The ablation study on the of window size � on VRU Long.
<table><tr><td>Metrics</td><td>W=1</td><td>W=3</td><td>W=5</td><td>W=7</td><td>w/o</td></tr><tr><td>PSNR↑</td><td>24.40</td><td>24.50</td><td>24.39</td><td>24.42</td><td>24.02</td></tr><tr><td>DSSIM↓</td><td>0.080</td><td>0.079</td><td>0.080</td><td>0.080</td><td>0.085</td></tr><tr><td>LPIPS ↓</td><td>0.147</td><td>0.145</td><td>0.147</td><td>0.147</td><td>0.153</td></tr></table>

Beyond reconstruction quality, our method demonstrates a clear advantage in modeling long temporal sequences. On the N3DV benchmark, existing methods are typically limited to short clips of up to 300 frames, whereas our approach enables one-shot reconstruction of sequences with up to 1200 frames. Similarly, on the VRU dataset, our method successfully models sequences of up to 1400 frames, while prior approaches [Wu et al. 2025a] usually handle only around 20 frames, corresponding to an approximately 70× increase in temporal coverage. These results highlight the scal ability of our approach to long and complex dynamic scenes. On SelfCap[Xu et al. 2024c] dataset, the results can be seen in Tab.8.

Table 6. The ablation study on the number of keyframes �(conducted on VRU Long 250 frames).
<table><tr><td>Metrics</td><td> $\mathrm { K } { = } 2 5 0$ </td><td> $\mathrm { K } { = } 1 2 5$ </td><td> $\mathrm { K } { = } 2 5$ </td><td> $\mathrm { K } { = } 1 2$ </td></tr><tr><td>PSNR↑</td><td>24.42</td><td>24.30</td><td>24.17</td><td>23.76</td></tr><tr><td>DSSIM↓</td><td>0.080</td><td>0.081</td><td>0.083</td><td>0.086</td></tr><tr><td>LPIPS↓</td><td>0.147</td><td>0.150</td><td>0.157</td><td>0.165</td></tr></table>

Table 7. The ablation study on diferent features on VRU Long.
<table><tr><td>Metrics</td><td>PSNR ↑</td><td>DSSIM1↓</td><td>DSSIM2↓</td><td>LPIPS ↓</td></tr><tr><td> $f _ { a } + f _ { t }$ </td><td>24.30</td><td>0.082</td><td>0.038</td><td>0.151</td></tr><tr><td> $f _ { s } + f _ { t }$ </td><td>22.58</td><td>0.116</td><td>0.055</td><td>0.240</td></tr><tr><td> $f _ { a } + f _ { s } + f _ { t }$ </td><td>24.42</td><td>0.080</td><td>0.037</td><td>0.147</td></tr></table>

Table 8. Quantitative results on the SelfCap dataset (Bar) (test views: 1, 7, 13; 2000 frames).
<table><tr><td>Method</td><td>PSNR↑</td><td>DSSIM↓</td><td>LPIPS↓</td></tr><tr><td>4DGaussian</td><td>27.86</td><td>0.0772</td><td>0.145</td></tr><tr><td>Ours</td><td>29.13</td><td>0.0689</td><td>0.110</td></tr></table>

Table 9. Inference time breakdown on VRU-GZ.
<table><tr><td></td><td>Feature Extraction</td><td>MLP Decoding</td><td>Rendering</td><td>Others</td><td>Full</td></tr><tr><td>Time (ms)</td><td>0.7</td><td>5.9</td><td>4.4</td><td>4.6</td><td>15.6</td></tr></table>

4.3.2 Qualitative Evaluation. Qualitative comparisons further validate the advantages of our method across diferent scenarios. As shown in Fig. 4 and Fig. 6, our reconstructions preserve fine-grained details with higher visual fidelity, particularly in challenging regions such as human hands, where competing methods often exhibit noticeable blurring or loss of structure. For scenes with complex and fast motion, such as the rapidly moving athletes in Fig. 5, our method is able to reconstruct dynamic content with high fidelity over long temporal spans. In contrast, LocalDyGS [Wu et al. 2025a] sufers from missing body parts, while streaming-based approaches such as 4DGS [Wu et al. 2024] exhibit more severe degradation, including increased blur and missing geometry, due to accumulated errors over time. Notably, our results are obtained by reconstructing the entire long sequence in a single training process, which is particularly challenging for long and complex dynamic scenes.

4.3.3 Generalization experiment. We conducted comprehensive experiments on large-scale datasets, including PKU-DyMVHumans and SelfCap [Xu et al. 2024c]. These datasets contain challenging scenarios with complex human motions and long video sequences. The extensive evaluation demonstrates the strong generalization capability of our method in dynamic scene modeling. Additional results (such as SelfCap [Xu et al. 2024c], PKU-DyHuman [Zheng et al. 2024]) are provided in the supplementary videos.

## 4.4 Ablation Study

The Number of Temporal Grids. We evaluate the impact of the number of temporal grids � on the 250-frame VRU-gz sequence by setting � = 1, 4, 8, 12. As shown in Tab. 4, the rendering quality consistently improves with larger �, indicating a positive correlation between performance and temporal grid capacity. Increasing the number of grids enables the model to store richer temporal information, resulting in more accurate and stable reconstructions, which aligns with our design motivation.

The ablation of query window size � . We further analyze the efect of query window size $W$ on the VRU-long dataset. As reported in Tab. 5, the performance remains relatively stable across diferent window sizes, suggesting that our method is robust to this hyperparameter and does not heavily rely on a specific temporal context range. However, removing the temporal window leads to a noticeable degradation in rendering quality, which validates the importance of the proposed query window.

The number of keyframes. We investigate the influence of the number of keyframes on reconstruction quality. By varying the number of selected keyframes while keeping all other settings fixed (in Tab. $\left. 6 \right) ,$ we observe that increasing the number of keyframes consistently improves performance, as it provides stronger temporal coverage and more reliable supervision for long sequences.

Ablation of static feature $f _ { s }$ and anchor feature $f _ { a } .$ To assess the efectiveness of the proposed static feature $f _ { s }$ and anchor feature $f _ { a } ,$ we perform an ablation study in Tab. 7 by removing each component while keeping all others unchanged. Excluding either $f _ { s }$ or $f _ { a }$ leads to degraded reconstruction quality, particularly in regions with large motion and fine geometric details. These results indicate that both features are critical for encoding spatial context and stabilizing feature aggregation across frames.

Inference time details. To further analyze the time consumption of our model, we report a detailed breakdown in Tab. 9.

## 5 Discussion and Limitations

Similar to previous works [Wu et al. 2025a; Xu et al. 2024c], our method follows an ofline reconstruction paradigm and therefore does not support real-time processing, which remains an open and challenging problem in volumetric video reconstruction. In addition, our approach relies on camera poses and sparse point clouds estimated by COLMAP. Although COLMAP provides accurate and robust geometric initialization, its relatively high computational cost has become a common bottleneck for many recent 3DGS-based pipelines. With the rapid progress of feed-forward camera pose estimation and scene reconstruction methods, we believe this limitation will be gradually mitigated in future systems.

Furthermore, we observe mild temporal jitter in regions associated with extremely sparse viewpoints, such as distant audience areas in the VRU dataset. This behavior is mainly caused by insuficient multi-view observations in these regions, which leads to under-constrained optimization during training and makes it dificult to consistently enforce temporal coherence. Similar phenomena have also been reported in static reconstruction methods, where Gaussians corresponding to marginal viewpoints are often blurry due to limited observational constraints. Notably, these artifacts are restricted to peripheral areas that are weakly observed and lie outside the primary focus of our reconstruction and evalua tion. Consequently, they have limited impact on the reconstruction quality of the main subjects or the overall conclusions of this work. We leave the incorporation of stronger temporal regularization or additional geometric priors for sparsely observed regions as an interesting direction for future research.

Full  
![](images/ef5a813b4454c5ff41b3da28c695d54e80fd607e911a7a7b2d7ad856e04df2e9.jpg)

w/o static feature  
![](images/5dcd2efd4e721ae67d85f81ded7ab04a80f48f09376a66abddb3fe089cc07031.jpg)

w/o anchor feature  
![](images/683a516119a0dbbe172940ebaa49a814d6c8987660822df8baa011eeaa2ed8b5.jpg)

![](images/9e1620d867e27e8ce28a7c77546551d3c1fe1a94893dff602a271810a5aef3a2.jpg)  
w/o window

![](images/7c953c577ae4c9b7a069ba0018e5c47c339a364b818929ba7bc07aedf7eeeea8.jpg)  
Keyframes K=25

![](images/332a18988e75473182c5136aa11ced6aca4d2bc07447cb5751686cff5325ffab.jpg)  
Keyframes K=12

Fig. 7. Additional qualitative visualizations from the ablation study (On VRU Long 250 frames).  
![](images/2514e5df760a50af4febbaafcfd8aea25617face3ad488d1e1f2a29790543872.jpg)  
Fig. 8. Qualitative results on a broader set of datasets.

Another key factor afecting dynamic reconstruction quality is the limited multi-view coverage in dynamic datasets. These datasets typically provide only 10–40 views. For example, in the VRU dataset, cameras are mainly distributed around the stadium boundary in a large-scale scene, resulting in weak geometric constraints in the inner field region. In addition, dynamic subjects occupy only a small number of pixels, making gradient-based optimization particularly challenging.

## 6 Conclusion

We propose ATGS, a novel framework capable of volumetric video reconstruction for long multi-view sequences with large-scale motion, extending reconstruction from small-motion, short sequences to long sequences with significant motion. Our approach is built on two core ideas: time-conditioned anchors and feature decomposition. Time-conditioned anchors localize all moving objects in space-time and specify their corresponding timestamps. To map anchors to Gaussians, we learn a set of features for each anchor, including anchor features for fine-detail control, static features for spatial stability, and temporal features for temporal modeling. By fusing these features, our model can stably reconstruct volumetric videos spanning thousands of frames while achieving state-of-theart performance in both objective and perceptual quality.

## 7 Acknowledges

This work is financially supported by Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China(Grant No. JYB2025XDXM413), Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology(Grant No. 2024B1212010006), this work is also financially supported for Outstanding Talents Training Fund in Shenzhen, Shenzhen Science and Technology Program(Grant No. SYSPG20241211173 440004 and KJZD20230923114707016), R24115SG MIGU-PKU META VISION TECHNOLOGY INNOVATION LAB.

## References

Naveed Ahmed, Christian Theobalt, Christian Rossl, Sebastian Thrun, and Hans-Peter Seidel. 2008. Dense correspondence finding for parametrization-free animation reconstruction from video. In 2008 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 1–8.

Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil Kim. 2023. HyperReel: High-Fidelity 6-DoF Video with Ray-Conditioned Sampling. arXiv preprint arXiv:2301.02238 (2023).

Aljaz Bozic, Michael Zollhofer, Christian Theobalt, and Matthias Nießner. 2020. Deep deform: Learning non-rigid rgb-d reconstruction with semi-supervised data. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7002–7012.

Grigore C Burdea and Philippe Coifet. 2003. Virtual reality technology. John Wiley & Sons.

Ang Cao and Justin Johnson. 2023. Hexplane: A fast representation for dynamic scenes. In Proceedings ofthe IEEE/CVFConference on ComputerVision and Pattern Recognition. 130–141.

Joel Carranza, Christian Theobalt, Marcus A Magnor, and Hans-Peter Seidel. 2003. Free-viewpoint video of human actors. ACM transactions on graphics (TOG) 22, 3 (2003), 569–577.

Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. 2022. Tensorf: Tensorial radiance fields. In European conference on computer vision. Springer, 333– 350.

Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B Tenenbaum, and Jiajun Wu. 2021. Neural radiance flow for 4d view synthesis and video processing. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE Computer Society, 14304– 14314.

Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 2024a. 4D-Rotor Gaussian Splatting: Towards Eficient Novel-View Synthesis for Dynamic Scenes. In Proc. SIGGRAPH.

Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, and Baoquan Chen. 2024b. 4d-rotor gaussian splatting: towards eficient novel view synthesis for dynamic scenes. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. 2023. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12479–12488.

Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. 2022. Plenoxels: Radiance fields without neural networks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 5501–5510.

Qiankun Gao, Jiarui Meng, Chengxiang Wen, Jie Chen, and Jian Zhang. 2024. HiCoM: Hierarchical Coherent Motion for Dynamic Streamable Scenes with 3D Gaussian Splatting. In Advances in Neural Information Processing Systems (NeurIPS).

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2024b. 2d gaussian splatting for geometrically accurate radiance fields. arXiv preprint arXiv:2403.17888 (2024).

Yi-Hua Huang, Yang-Tian Sun, Ziyi Yang, Xiaoyang Lyu, Yan-Pei Cao, and Xiaojuan Qi. 2024a. Sc-gs: Sparse-controlled gaussian splatting for editable dynamic scenes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4220–4230.

Yili Jin, Kaiyuan Hu, Junhua Liu, Fangxin Wang, and Xue Liu. 2023. From capture to display: A survey on volumetric video. arXiv preprint arXiv:2309.05658 (2023).

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Trans. Graph. 42, 4 (2023), 139–1.

Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

Lingzhi Li, Zhen Shen, Zhongshu Wang, Li Shen, and Ping Tan. 2022a. Streaming radiance fields for 3d video synthesis. Advances in Neural Information Processing Systems 35 (2022), 13485–13498.

Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changi Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. 2022b. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5521–5531.

Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. 2024. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8508–8520.

Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. 2021. Neural scene flow fields for space-time view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6498–6508.

Jie Liang, Jiahao Wu, Chao Wang, Jiayu Yang, Xiaoyun Zheng, Kaiqiang Xiong, Zhanke Wang, Jinbo Yan, Feng Gao, and Ronggang Wang. 2026. ClipGStream: Clip-Stream Gaussian Splatting for Any Length and Any Motion Multi-View Dynamic Scene Reconstruction. arXiv preprint arXiv:2604.13746 (2026).

Stephen Lombardi, Tomas Simon, Jason Saragih, Gabriel Schwartz, Andreas Lehrmann, and Yaser Sheikh. 2019. Neural volumes: Learning dynamic renderable volumes

from images. arXiv preprint arXiv:1906.07751 (2019)

Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. 2024. Scafold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20654–20664.

Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. 2024. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 2024 International Conference on 3D Vision (3DV). IEEE, 800–809

Saswat Subhajyoti Mallick, Rahul Goel, Bernhard Kerbl, Markus Steinberger, Fran cisco Vicente Carrasco, and Fernando De La Torre. 2024. Taming 3DGS: High Quality Radiance Fields with Limited Resources. In SIGGRAPH Asia 2024 Conference Papers (SA ’24). Association for Computing Machinery, New York, NY, USA, Article 2, 11 pages. doi:10.1145/3680528.3687694

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG) 41, 4 (2022), 1–15.

Richard A Newcombe, Dieter Fox, and Steven M Seitz. 2015. Dynamicfusion: Reconstruction and tracking of non-rigid scenes in real-time. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 343–352.

Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. 2021a. Nerfies: Deformable neural radiance fields. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 5865–5874.

Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. 2021b. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228 (2021).

Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2021. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10318–10327.

Philipp A Rauschnabel, Reto Felix, Chris Hinsch, Hamza Shahab, and Florian Alt. 2022. What is XR? Towards a framework for augmented and virtual reality. Computers in human behavior 133 (2022), 107289.

Liangchen Song, Anpei Chen, Zhong Li, Zhang Chen, Lele Chen, Junsong Yuan, Yi Xu, and Andreas Geiger. 2023. Nerfplayer: A streamable dynamic scene representation with decomposed neural radiance fields. IEEE Transactions on Visualization and Computer Graphics 29, 5 (2023), 2732–2742.

Maximilian Speicher, Brian D Hall, and Michael Nebeling. 2019. What is mixed reality?. In Proceedings of the 2019 CHI conference on human factors in computing systems. 1–15.

Cheng Sun, Min Sun, and Hwann-Tzong Chen. 2022. Direct Voxel Grid Optimization: Super-fast Convergence for Radiance Fields Reconstruction. In CVPR.

Jiakai Sun, Han Jiao, Guangyuan Li, Zhanjie Zhang, Lei Zhao, and Wei Xing. 2024. 3dgstream: On-the-fly training of 3d gaussians for eficient streaming of photorealistic free-viewpoint videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20675–20685.

Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, and Huaping Liu. 2023a. Masked Space-Time Hash Encoding for Eficient Dynamic Scene Reconstruction. arXiv:2310.17527 [cs.CV]

Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, and Huaping Liu. 2022. Mixed Neural Voxels for Fast Multi-view Video Synthesis. arXiv preprint arXiv:2212.00190 (2022).

Feng Wang, Sinan Tan, Xinghang Li, Zeyue Tian, Yafei Song, and Huaping Liu. 2023b. Mixed neural voxels for fast multi-view video synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19706–19716.

Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng, and Xiaowei Zhou. 2025. FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction. In CVPR. https://zju3dv. github.io/freetimegs

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 2024. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20310–20320.

Jiahao Wu, Rui Peng, Jianbo Jiao, Jiayu Yang, Luyang Tang, Kaiqiang Xiong, Jie Liang, Jinbo Yan, Runling Liu, and Ronggang Wang. 2025a. LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9519–9529.

Jiahao Wu, Rui Peng, Zhiyan Wang, Lu Xiao, Luyang Tang, Jinbo Yan, Kaiqiang Xiong, and Ronggang Wang. 2025b. Swift4D: Adaptive divide-and-conquer Gaussian Splat ting for compact and eficient reconstruction of dynamic scene. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum? id=c1RhJVTPwT

Jiawei Xu, Zexin Fan, Jian Yang, and Jin Xie. 2024a. Grid4D: 4D Decomposed Hash Encoding for High-fidelity Dynamic Gaussian Splatting. arXiv preprint arXiv:2410.20815 (2024).

Zhen Xu, Sida Peng, Haotong Lin, Guangzhao He, Jiaming Sun, Yujun Shen, Hujun Bao, and Xiaowei Zhou. 2023. 4K4D: Real-Time 4D View Synthesis at 4K Resolution. (2023).

Zhen Xu, Sida Peng, Haotong Lin, Guangzhao He, Jiaming Sun, Yujun Shen, Hujun Bao, and Xiaowei Zhou. 2024b. 4k4d: Real-time 4d view synthesis at 4k resolution. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 20029–20040.

Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun, Hujun Bao, and Xiaowei Zhou. 2024c. Representing Long Volumetric Video with Temporal Gaussian Hierarchy. ACM Transactions on Graphics 43, 6 (November 2024). https: //zju3dv.github.io/longvolcap

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. 2024. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings ofthe IEEE/CVFConference on ComputerVision and Pattern Recognition.

20331–20341.

Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. 2023. Real-time photo realistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint arXiv:2310.10642 (2023).

Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. 2024. Mip splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19447–19456.

Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, and Ping Tan. 2024. RaDe-GS: Rasterizing Depth in Gaussian Splatting. arXiv preprint arXiv:2406.01467 (2024).

Xiaoyun Zheng, Liwei Liao, Xufeng Li, Jianbo Jiao, Rongjie Wang, Feng Gao, Shiqi Wang, and Ronggang Wang. 2024. PKU-DyMVHumans: A Multi-View Video Benchmark for High-Fidelity Dynamic Human Modeling. arXiv:2403.16080 [cs.CV] https: //arxiv.org/abs/2403.16080